// hls/src/conv2d_streaming_im2col.cpp

#include <ap_int.h>
#include <hls_stream.h>

typedef float data_t;

// --------------------------------------------------------------------
// Configuration (small-ish for csim; tune later)
// --------------------------------------------------------------------

// N fixed to 1 for this kernel
#define N        1
#define C_IN     8
#define C_OUT    16
#define H_IN     8
#define W_IN     8

// Kernel / stride / padding
#define K        3
#define STRIDE   1
#define PADDING  1

// Output spatial size (no dilation)
#define H_OUT    ((H_IN + 2*PADDING - K) / STRIDE + 1)
#define W_OUT    ((W_IN + 2*PADDING - K) / STRIDE + 1)
#define L_TOTAL  (H_OUT * W_OUT)

// im2col / GEMM tiling parameters
#define TILE_L   16                 // number of spatial locations per tile
#define K_DIM    (C_IN * K * K)     // K dimension of GEMM
#define P_PE     4                  // PE rows (C_OUT dimension)
#define Q_PE     4                  // PE cols (L dimension)
#define TILE_K   8                  // K chunk per GEMM iteration

inline int i_min(int a, int b) {
#pragma HLS INLINE
    return (a < b) ? a : b;
}

inline void linear_to_2d(int l, int &oh, int &ow) {
#pragma HLS INLINE
    oh = l / W_OUT;
    ow = l % W_OUT;
}

// --------------------------------------------------------------------
// im2col tile loader (single batch, all channels)
//   x:    [C_IN][H_IN][W_IN]
//   B_t:  [K_DIM][L_eff]
// --------------------------------------------------------------------
void im2col_nchw_tile(
    data_t x[C_IN][H_IN][W_IN],
    int     l_start,
    int     L_eff,
    data_t  B_tile[K_DIM][TILE_L]
) {
#pragma HLS INLINE off

    const int K_h = K;
    const int K_w = K;
    const int stride_h = STRIDE;
    const int stride_w = STRIDE;
    const int pad_h = PADDING;
    const int pad_w = PADDING;

    // Initialize tile
IM2COL_INIT_K:
    for (int k = 0; k < K_DIM; ++k) {
    IM2COL_INIT_T:
        for (int t = 0; t < L_eff; ++t) {
        #pragma HLS PIPELINE II=1
            B_tile[k][t] = 0.0f;
        }
    }

    // For each tile location t (corresponding to linear index l)
IM2COL_LOC_LOOP:
    for (int t = 0; t < L_eff; ++t) {
    #pragma HLS LOOP_TRIPCOUNT min=1 max=TILE_L
        int l = l_start + t;
        int oh, ow;
        linear_to_2d(l, oh, ow);

        int in_h0 = oh * stride_h - pad_h;
        int in_w0 = ow * stride_w - pad_w;

        int flat_idx = 0;

    IM2COL_CH_LOOP:
        for (int c = 0; c < C_IN; ++c) {
        IM2COL_KH_LOOP:
            for (int kh = 0; kh < K_h; ++kh) {
            IM2COL_KW_LOOP:
                for (int kw = 0; kw < K_w; ++kw) {
                #pragma HLS PIPELINE II=1
                    int ih = in_h0 + kh;
                    int iw = in_w0 + kw;

                    data_t v = 0.0f;
                    if (ih >= 0 && ih < H_IN && iw >= 0 && iw < W_IN) {
                        v = x[c][ih][iw];
                    }
                    B_tile[flat_idx][t] = v;
                    flat_idx++;
                }
            }
        }
    }
}

// --------------------------------------------------------------------
// Explicit output-stationary systolic-like GEMM
//   C = A @ B
//   A: [C_OUT][K_DIM]
//   B: [K_DIM][L_eff]
//   C: [C_OUT][L_eff]
// --------------------------------------------------------------------
void systolic_gemm_explicit(
    data_t A[C_OUT][K_DIM],
    data_t B[K_DIM][TILE_L],
    int    L_eff,
    data_t C[C_OUT][TILE_L]
) {
#pragma HLS INLINE off

    // Initialize C
GEMM_INIT_CO:
    for (int co = 0; co < C_OUT; ++co) {
    GEMM_INIT_T:
        for (int t = 0; t < L_eff; ++t) {
        #pragma HLS PIPELINE II=1
            C[co][t] = 0.0f;
        }
    }

    // Tile over (M, N) using P_PE × Q_PE PEs
GEMM_M_TILE:
    for (int m0 = 0; m0 < C_OUT; m0 += P_PE) {
    GEMM_N_TILE:
        for (int n0 = 0; n0 < L_eff; n0 += Q_PE) {
            int m_max = i_min(m0 + P_PE, C_OUT);
            int n_max = i_min(n0 + Q_PE, L_eff);
            int p_eff = m_max - m0;
            int q_eff = n_max - n0;

            data_t C_tile[P_PE][Q_PE];
        #pragma HLS ARRAY_PARTITION variable=C_tile complete dim=1
        #pragma HLS ARRAY_PARTITION variable=C_tile complete dim=2

            // Initialize local tile
GEMM_TILE_INIT_I:
            for (int i = 0; i < p_eff; ++i) {
            GEMM_TILE_INIT_J:
                for (int j = 0; j < q_eff; ++j) {
                #pragma HLS PIPELINE II=1
                    C_tile[i][j] = 0.0f;
                }
            }

            // Stream over K dimension in chunks
GEMM_K_CHUNK:
            for (int k0 = 0; k0 < K_DIM; k0 += TILE_K) {
                int k_max = i_min(k0 + TILE_K, K_DIM);
                int k_eff = k_max - k0;

            GEMM_K_INNER:
                for (int kk = 0; kk < k_eff; ++kk) {
                    int k_idx = k0 + kk;

                GEMM_PE_I:
                    for (int i = 0; i < p_eff; ++i) {
                        data_t a_val = A[m0 + i][k_idx];

                    GEMM_PE_J:
                        for (int j = 0; j < q_eff; ++j) {
                        #pragma HLS PIPELINE II=1
                            data_t b_val = B[k_idx][n0 + j];
                            C_tile[i][j] += a_val * b_val;
                        }
                    }
                }
            }

            // Write back tile
GEMM_WRITEBACK_I:
            for (int i = 0; i < p_eff; ++i) {
            GEMM_WRITEBACK_J:
                for (int j = 0; j < q_eff; ++j) {
                #pragma HLS PIPELINE II=1
                    C[m0 + i][n0 + j] += C_tile[i][j];
                }
            }
        }
    }
}

// --------------------------------------------------------------------
// Conv2D core (N=1, groups=1)
//   x: [C_IN][H_IN][W_IN]
//   w: [C_OUT][C_IN][K][K]
//   b: [C_OUT]
//   y: [C_OUT][H_OUT][W_OUT]
// --------------------------------------------------------------------
void conv2d_nchw_streaming_core(
    data_t x[C_IN][H_IN][W_IN],
    data_t w[C_OUT][C_IN][K][K],
    data_t b[C_OUT],
    data_t y[C_OUT][H_OUT][W_OUT]
) {
#pragma HLS INLINE off

    // Flatten weights to A[C_OUT][K_DIM]
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
                    A[co][flat_idx] = w[co][ci][kh][kw];
                    flat_idx++;
                }
            }
        }
    }

    // Initialize output
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

    // Buffers
    data_t B_tile[K_DIM][TILE_L];
    data_t C_tile[C_OUT][TILE_L];

#pragma HLS ARRAY_PARTITION variable=B_tile dim=1 factor=8 cyclic
#pragma HLS ARRAY_PARTITION variable=C_tile dim=1 factor=P_PE cyclic

    int l_start = 0;
TILE_L_LOOP:
    while (l_start < L_TOTAL) {
        int L_eff = i_min(TILE_L, L_TOTAL - l_start);

        // 1) im2col tile
        im2col_nchw_tile(x, l_start, L_eff, B_tile);

        // 2) GEMM tile
        systolic_gemm_explicit(A, B_tile, L_eff, C_tile);

        // 3) Add bias & write back
WRITEBACK_T_LOOP:
        for (int t = 0; t < L_eff; ++t) {
            int l = l_start + t;
            int oh, ow;
            linear_to_2d(l, oh, ow);

        WRITEBACK_CO_LOOP:
            for (int co = 0; co < C_OUT; ++co) {
            #pragma HLS PIPELINE II=1
                data_t val = C_tile[co][t] + b[co];
                y[co][oh][ow] = val;
            }
        }

        l_start += L_eff;
    }
}

// --------------------------------------------------------------------
// Top-level HLS wrapper with AXI interfaces
//   x_flat: [C_IN * H_IN * W_IN]
//   w_flat: [C_OUT * C_IN * K * K]
//   b_flat: [C_OUT]
//   y_flat: [C_OUT * H_OUT * W_OUT]
// --------------------------------------------------------------------
extern "C" {
void conv2d_nchw_streaming_top(
    const data_t *x_flat,
    const data_t *w_flat,
    const data_t *b_flat,
    data_t *y_flat
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

    data_t x[C_IN][H_IN][W_IN];
    data_t w[C_OUT][C_IN][K][K];
    data_t b[C_OUT];
    data_t y[C_OUT][H_OUT][W_OUT];

#pragma HLS ARRAY_PARTITION variable=x dim=1 factor=4 cyclic
#pragma HLS ARRAY_PARTITION variable=w dim=2 factor=4 cyclic
#pragma HLS ARRAY_PARTITION variable=y dim=1 factor=P_PE cyclic

    // Load input
LOAD_X_CH:
    int idx = 0;
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

    // Load weights
LOAD_W_CO:
    idx = 0;
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

    // Load bias
LOAD_B_CO:
    for (int co = 0; co < C_OUT; ++co) {
    #pragma HLS PIPELINE II=1
        b[co] = b_flat[co];
    }

    // Run core
    conv2d_nchw_streaming_core(x, w, b, y);

    // Store output
STORE_Y_CO:
    idx = 0;
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
