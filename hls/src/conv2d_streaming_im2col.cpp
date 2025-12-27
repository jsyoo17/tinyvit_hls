// hls/src/conv2d_streaming_im2col.cpp
//
// Conv2D implementation using:
//   - "Streaming im2col" idea: build only a small tile of the unfolded matrix B
//   - GEMM-style compute: C = A @ B
//   - Tiling over output spatial positions (columns) to avoid storing full im2col
//   - A logical P_PE x Q_PE PE array for the GEMM tile
//
// IMPORTANT: Loop labels are written as C/C++ statement labels. To keep the
// source readable and to correlate with HLS Loop Analysis, we place each label
// on its own line and align it to the SAME indentation as the loop statement.
//
// Build target: Vitis HLS 2022.2
// Data type: float (for now)

#include <ap_int.h>
#include <hls_stream.h>
#include <cstdio>

typedef float data_t;

// -----------------------------------------------------------------------------
// Compile-time configuration (toy sizes for csim; you will later parameterize)
// -----------------------------------------------------------------------------

// Fixed N=1 kernel (batch dimension handled outside for now)
#define N        1

// Input feature map: x[C_IN][H_IN][W_IN]
#define C_IN     8
#define H_IN     8
#define W_IN     8

// Output feature map: y[C_OUT][H_OUT][W_OUT]
#define C_OUT    16

// Convolution hyperparameters (NO dilation, NO groups)
#define K        3     // kernel size KxK
#define STRIDE   1
#define PADDING  1

// Derived output size for standard conv2d
#define H_OUT    ((H_IN + 2*PADDING - K) / STRIDE + 1)
#define W_OUT    ((W_IN + 2*PADDING - K) / STRIDE + 1)

// Total output spatial positions (flattened)
#define L_TOTAL  (H_OUT * W_OUT)

// -----------------------------------------------------------------------------
// GEMM / tiling configuration
// -----------------------------------------------------------------------------

// K_DIM is the inner dimension of GEMM after flattening (C_IN * K * K)
#define K_DIM    (C_IN * K * K)

// TILE_L is the number of output spatial locations processed per outer tile.
// This determines the size of the local "B_tile" (im2col tile) buffer.
// Smaller TILE_L uses less on-chip memory but increases loop overhead.
#define TILE_L   16

// PE array concept for the GEMM tile:
// - P_PE corresponds to number of output channels computed in parallel (rows)
// - Q_PE corresponds to number of output locations computed in parallel (cols)
#define P_PE     4
#define Q_PE     4

// TILE_K is the chunk size of the K_DIM reduction dimension.
// Streaming K in chunks helps control local buffering and can map to a
// pipeline where A/B values stream through a systolic-like datapath.
#define TILE_K   8

// -----------------------------------------------------------------------------
// Small helpers (avoid <algorithm> / std::min to keep HLS clean)
// -----------------------------------------------------------------------------
static inline int i_min(int a, int b) {
#pragma HLS INLINE
    return (a < b) ? a : b;
}

// Convert flattened output location l in [0, L_TOTAL) to 2D coordinates (oh, ow).
// Flattening uses row-major order: l = oh * W_OUT + ow.
static inline void linear_to_2d(int l, int &oh, int &ow) {
#pragma HLS INLINE
    oh = l / W_OUT;
    ow = l % W_OUT;
}

// -----------------------------------------------------------------------------
// im2col_nchw_tile
// -----------------------------------------------------------------------------
// Build a SMALL tile of the "im2col" matrix B for conv2d.
//
// If we wrote full im2col, we'd create:
//   B_full[K_DIM][L_TOTAL]
// where each column corresponds to one output spatial location (oh, ow),
// and each row corresponds to one element in the flattened receptive field
// over (C_IN, K, K).
//
// That is too large to store on-chip for real layers.
// So instead we build only:
//   B_tile[K_DIM][TILE_L]
// but only columns [0..L_eff-1] are used for the current tile.
//
// Inputs:
//   x          : [C_IN][H_IN][W_IN] input feature map
//   l_start    : starting column index into flattened output space
//   L_eff      : number of valid columns in this tile (<= TILE_L)
// Output:
//   B_tile     : [K_DIM][TILE_L] where cols 0..L_eff-1 are valid
//
// Memory / hardware interpretation:
//   - x would typically be in on-chip buffer or streamed.
//   - B_tile is a local SRAM/register buffer representing the unfolded window
//     data for this tile.
static void im2col_nchw_tile(
    data_t x[C_IN][H_IN][W_IN],
    int    l_start,
    int    L_eff,
    data_t B_tile[K_DIM][TILE_L]
) {
#pragma HLS INLINE off

    // 1) Initialize the tile buffer (defensive).
    //    If you guarantee every element is overwritten, you can remove this loop.
    IM2COL_INIT_K:
    for (int k = 0; k < K_DIM; ++k) {
        IM2COL_INIT_T:
        for (int t = 0; t < L_eff; ++t) {
#pragma HLS PIPELINE II=1
            B_tile[k][t] = 0.0f;
        }
    }

    // 2) For each output location in this tile, compute its receptive field
    //    and write a flattened (C_IN*K*K) vector into one column of B_tile.
    IM2COL_LOC_T:
    for (int t = 0; t < L_eff; ++t) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=TILE_L
        const int l = l_start + t;

        int oh, ow;
        linear_to_2d(l, oh, ow);

        // Top-left coordinate of receptive field in input space
        // (accounts for stride and padding)
        const int in_h0 = oh * STRIDE - PADDING;
        const int in_w0 = ow * STRIDE - PADDING;

        // flat_idx enumerates rows of the flattened receptive field:
        // ordering is: c-major, then kh, then kw:
        // flat_idx = ((c * K) + kh) * K + kw
        int flat_idx = 0;

        IM2COL_CH_C:
        for (int c = 0; c < C_IN; ++c) {
            IM2COL_KH:
            for (int kh = 0; kh < K; ++kh) {
                IM2COL_KW:
                for (int kw = 0; kw < K; ++kw) {
#pragma HLS PIPELINE II=1
                    const int ih = in_h0 + kh;
                    const int iw = in_w0 + kw;

                    // Handle padding: out-of-range reads are treated as zero
                    data_t v = 0.0f;
                    if (ih >= 0 && ih < H_IN && iw >= 0 && iw < W_IN) {
                        v = x[c][ih][iw];
                    }

                    // Write into B_tile row=flat_idx, col=t
                    B_tile[flat_idx][t] = v;
                    flat_idx++;
                }
            }
        }
    }
}

// -----------------------------------------------------------------------------
// systolic_gemm_explicit
// -----------------------------------------------------------------------------
// Compute C = A @ B for the current tile.
//
// Shapes:
//   A : [C_OUT][K_DIM]           (weights flattened for GEMM)
//   B : [K_DIM][L_eff]           (im2col tile)
//   C : [C_OUT][L_eff]           (output tile before bias)
//
// This uses a *logical* PE array (P_PE x Q_PE):
//   - P dimension maps to output channels (C_OUT)
//   - Q dimension maps to output locations (columns)
//
// We tile over:
//   M (rows) by P_PE
//   N (cols) by Q_PE
// and reduce over K_DIM in chunks of TILE_K.
static void systolic_gemm_explicit(
    data_t A[C_OUT][K_DIM],
    data_t B[K_DIM][TILE_L],
    int    L_eff,
    data_t C[C_OUT][TILE_L]
) {
#pragma HLS INLINE off

    // 1) Initialize output tile C to zero for cols 0..L_eff-1
    GEMM_INIT_CO:
    for (int co = 0; co < C_OUT; ++co) {
        GEMM_INIT_T:
        for (int t = 0; t < L_eff; ++t) {
#pragma HLS PIPELINE II=1
            C[co][t] = 0.0f;
        }
    }

    // 2) Tile across M (output channels) and N (output locations)
    GEMM_TILE_M0:
    for (int m0 = 0; m0 < C_OUT; m0 += P_PE) {
        GEMM_TILE_N0:
        for (int n0 = 0; n0 < L_eff; n0 += Q_PE) {

            // Effective tile sizes at boundaries
            const int m_max = i_min(m0 + P_PE, C_OUT);
            const int n_max = i_min(n0 + Q_PE, L_eff);
            const int p_eff = m_max - m0;   // <= P_PE
            const int q_eff = n_max - n0;   // <= Q_PE

            // Local accumulator array (models registers in a PE grid)
            data_t C_pe[P_PE][Q_PE];
#pragma HLS ARRAY_PARTITION variable=C_pe complete dim=1
#pragma HLS ARRAY_PARTITION variable=C_pe complete dim=2

            // Initialize PE accumulators
            GEMM_PE_INIT_I:
            for (int i = 0; i < p_eff; ++i) {
                GEMM_PE_INIT_J:
                for (int j = 0; j < q_eff; ++j) {
#pragma HLS PIPELINE II=1
                    C_pe[i][j] = 0.0f;
                }
            }

            // 3) Reduce over K dimension in TILE_K chunks
            // Each iteration conceptually streams one "k slice" of A and B
            // through the PE array.
            GEMM_REDUCE_K0:
            for (int k0 = 0; k0 < K_DIM; k0 += TILE_K) {
                const int k_max = i_min(k0 + TILE_K, K_DIM);
                const int k_eff = k_max - k0;

                GEMM_REDUCE_KK:
                for (int kk = 0; kk < k_eff; ++kk) {
                    const int k_idx = k0 + kk;

                    // For each PE row i: read A once and reuse across Q columns
                    GEMM_PE_ROW_I:
                    for (int i = 0; i < p_eff; ++i) {
                        const data_t a_val = A[m0 + i][k_idx];

                        // For each PE col j: multiply by corresponding B and accumulate
                        GEMM_PE_COL_J:
                        for (int j = 0; j < q_eff; ++j) {
#pragma HLS PIPELINE II=1
                            const data_t b_val = B[k_idx][n0 + j];
                            C_pe[i][j] += a_val * b_val;
                        }
                    }
                }
            }

            // 4) Write PE accumulators back to C tile
            GEMM_WRITEBACK_I:
            for (int i = 0; i < p_eff; ++i) {
                GEMM_WRITEBACK_J:
                for (int j = 0; j < q_eff; ++j) {
#pragma HLS PIPELINE II=1
                    C[m0 + i][n0 + j] += C_pe[i][j];
                }
            }
        }
    }
}

// -----------------------------------------------------------------------------
// conv2d_nchw_streaming_core
// -----------------------------------------------------------------------------
// Orchestrates conv2d using tiled im2col + tiled GEMM.
//
// Steps:
// 1) Flatten weights w[C_OUT][C_IN][K][K] into A[C_OUT][K_DIM] for GEMM.
// 2) For each tile of L locations:
//      a) Build B_tile[K_DIM][L_eff] via im2col
//      b) Compute C_tile[C_OUT][L_eff] = A @ B_tile
//      c) Add bias and scatter results to y[C_OUT][H_OUT][W_OUT]
static void conv2d_nchw_streaming_core(
    data_t x[C_IN][H_IN][W_IN],
    data_t w[C_OUT][C_IN][K][K],
    data_t b[C_OUT],
    data_t y[C_OUT][H_OUT][W_OUT]
) {
#pragma HLS INLINE off

    // Flatten weights into GEMM matrix A
    data_t A[C_OUT][K_DIM];
#pragma HLS ARRAY_PARTITION variable=A dim=1 factor=P_PE cyclic

    FLATTEN_W_CO:
    for (int co = 0; co < C_OUT; ++co) {
        int flat_idx = 0;

        FLATTEN_W_CI:
        for (int ci = 0; ci < C_IN; ++ci) {

            FLATTEN_W_KH:
            for (int kh = 0; kh < K; ++kh) {

                FLATTEN_W_KW:
                for (int kw = 0; kw < K; ++kw) {
#pragma HLS PIPELINE II=1
                    // A[co, flat_idx] corresponds to weight w[co, ci, kh, kw]
                    A[co][flat_idx] = w[co][ci][kh][kw];
                    flat_idx++;
                }
            }
        }
    }

    // Initialize output y to zero (not strictly required if we overwrite all)
    INIT_Y_CO:
    for (int co = 0; co < C_OUT; ++co) {
        INIT_Y_OH:
        for (int oh = 0; oh < H_OUT; ++oh) {
            INIT_Y_OW:
            for (int ow = 0; ow < W_OUT; ++ow) {
#pragma HLS PIPELINE II=1
                y[co][oh][ow] = 0.0f;
            }
        }
    }

    // Local tile buffers:
    // - B_tile holds unfolded activations for TILE_L output positions
    // - C_tile holds GEMM result for those TILE_L positions and all C_OUT channels
    data_t B_tile[K_DIM][TILE_L];
    data_t C_tile[C_OUT][TILE_L];

    // Partitioning hints:
    // - Partition B_tile rows to allow multiple reads per cycle in GEMM
    // - Partition C_tile rows to allow parallel access across output channels
#pragma HLS ARRAY_PARTITION variable=B_tile dim=1 factor=8 cyclic
#pragma HLS ARRAY_PARTITION variable=C_tile dim=1 factor=P_PE cyclic

    // Tile loop over output spatial positions (flattened columns)
    int l_start = 0;

    TILE_L_LOOP:
    while (l_start < L_TOTAL) {
        // How many columns in this tile (last tile may be smaller)
        const int L_eff = i_min(TILE_L, L_TOTAL - l_start);

        // (a) Build im2col tile: B_tile[K_DIM][L_eff]
        im2col_nchw_tile(x, l_start, L_eff, B_tile);

        // (b) GEMM: C_tile[C_OUT][L_eff] = A[C_OUT][K_DIM] @ B_tile[K_DIM][L_eff]
        systolic_gemm_explicit(A, B_tile, L_eff, C_tile);

        // (c) Add bias and scatter tile outputs into y[co][oh][ow]
        WRITEBACK_T:
        for (int t = 0; t < L_eff; ++t) {
            const int l = l_start + t;
            int oh, ow;
            linear_to_2d(l, oh, ow);

            WRITEBACK_CO:
            for (int co = 0; co < C_OUT; ++co) {
#pragma HLS PIPELINE II=1
                // bias add + store
                y[co][oh][ow] = C_tile[co][t] + b[co];
            }
        }

        // Advance to next tile
        l_start += L_eff;
    }
}

// -----------------------------------------------------------------------------
// Top-level function (Vitis HLS entry point)
// -----------------------------------------------------------------------------
// This wrapper provides:
// - AXI4 master interfaces for x_flat, w_flat, b_flat, y_flat (DDR-style)
// - AXI4-Lite control interface for register access
//
// It performs:
// 1) Load flat arrays into on-chip tensors x, w, b
// 2) Call conv2d core
// 3) Store on-chip output y back to flat array y_flat
extern "C" {
void conv2d_nchw_streaming_top(
    const data_t *x_flat,
    const data_t *w_flat,
    const data_t *b_flat,
    data_t       *y_flat
) {
#pragma HLS INTERFACE m_axi port=x_flat offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=w_flat offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi port=b_flat offset=slave bundle=gmem2
#pragma HLS INTERFACE m_axi port=y_flat offset=slave bundle=gmem3

#pragma HLS INTERFACE s_axilite port=x_flat  bundle=control
#pragma HLS INTERFACE s_axilite port=w_flat  bundle=control
#pragma HLS INTERFACE s_axilite port=b_flat  bundle=control
#pragma HLS INTERFACE s_axilite port=y_flat  bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    // On-chip tensors
    data_t x[C_IN][H_IN][W_IN];
    data_t w[C_OUT][C_IN][K][K];
    data_t b[C_OUT];
    data_t y[C_OUT][H_OUT][W_OUT];

    // Partitioning helps parallel accesses on inner loops
#pragma HLS ARRAY_PARTITION variable=x dim=1 factor=4 cyclic
#pragma HLS ARRAY_PARTITION variable=w dim=2 factor=4 cyclic
#pragma HLS ARRAY_PARTITION variable=y dim=1 factor=P_PE cyclic

    // -------------------------------------------------------------------------
    // Load x from AXI memory into on-chip x tensor
    // Layout convention for x_flat: contiguous in [c][h][w] order
    // -------------------------------------------------------------------------
    int idx = 0;

    LOAD_X_C:
    for (int c = 0; c < C_IN; ++c) {
        LOAD_X_H:
        for (int h = 0; h < H_IN; ++h) {
            LOAD_X_W:
            for (int wi = 0; wi < W_IN; ++wi) {
#pragma HLS PIPELINE II=1
                x[c][h][wi] = x_flat[idx++];
            }
        }
    }

    // -------------------------------------------------------------------------
    // Load w from AXI memory into on-chip w tensor
    // Layout convention for w_flat: contiguous in [co][ci][kh][kw] order
    // -------------------------------------------------------------------------
    idx = 0;

    LOAD_W_CO:
    for (int co = 0; co < C_OUT; ++co) {
        LOAD_W_CI:
        for (int ci = 0; ci < C_IN; ++ci) {
            LOAD_W_KH:
            for (int kh = 0; kh < K; ++kh) {
                LOAD_W_KW:
                for (int kw = 0; kw < K; ++kw) {
#pragma HLS PIPELINE II=1
                    w[co][ci][kh][kw] = w_flat[idx++];
                }
            }
        }
    }

    // -------------------------------------------------------------------------
    // Load bias b from AXI memory into on-chip b vector
    // -------------------------------------------------------------------------
    LOAD_B_CO:
    for (int co = 0; co < C_OUT; ++co) {
#pragma HLS PIPELINE II=1
        b[co] = b_flat[co];
    }

    // -------------------------------------------------------------------------
    // Compute conv2d core
    // -------------------------------------------------------------------------
    conv2d_nchw_streaming_core(x, w, b, y);

    // -------------------------------------------------------------------------
    // Store output y back to AXI memory y_flat
    // Layout convention for y_flat: contiguous in [co][oh][ow] order
    // -------------------------------------------------------------------------
    idx = 0;

    STORE_Y_CO:
    for (int co = 0; co < C_OUT; ++co) {
        STORE_Y_OH:
        for (int oh = 0; oh < H_OUT; ++oh) {
            STORE_Y_OW:
            for (int ow = 0; ow < W_OUT; ++ow) {
#pragma HLS PIPELINE II=1
                y_flat[idx++] = y[co][oh][ow];
            }
        }
    }
}
} // extern "C"
