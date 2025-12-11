// hls/src/conv2d_streaming_im2col.hpp
#pragma once

#include <ap_int.h>
#include <hls_stream.h>

typedef float data_t;

// --------------------------------------------------------------------
// Configuration (start with small-ish sizes for csim; tune later)
// --------------------------------------------------------------------

// N fixed to 1 for HLS kernel
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

// Small helpers (inline)
inline int i_min(int a, int b) {
#pragma HLS INLINE
    return (a < b) ? a : b;
}

inline void linear_to_2d(int l, int &oh, int &ow) {
#pragma HLS INLINE
    oh = l / W_OUT;
    ow = l % W_OUT;
}
