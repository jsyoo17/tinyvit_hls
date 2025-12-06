// conv2d_patch_embed.h

#ifndef CONV2D_PATCH_EMBED_H
#define CONV2D_PATCH_EMBED_H

#include <cmath>

// TinyViT-5M patch_embed.conv1.conv parameters
// You can tweak these later or template them.
const int N_BATCH   = 1;
const int C_IN      = 3;
const int H_IN      = 224;
const int W_IN      = 224;

const int C_OUT     = 32;
const int K         = 3;
const int STRIDE    = 2;
const int PADDING   = 1;

const int H_OUT     = (H_IN + 2 * PADDING - K) / STRIDE + 1;  // 112
const int W_OUT     = (W_IN + 2 * PADDING - K) / STRIDE + 1;  // 112

// I/O layout:
//   input:  [C_IN][H_IN][W_IN]
//   weight: [C_OUT][C_IN][K][K]
//   bias:   [C_OUT]
//   output: [C_OUT][H_OUT][W_OUT]
//
// For HLS top-level you usually don't want N_BATCH dimension;
// we assume N=1 and use first image only.

void conv2d_patch_embed(
    const float input[C_IN][H_IN][W_IN],
    const float weight[C_OUT][C_IN][K][K],
    const float bias[C_OUT],
    float output[C_OUT][H_OUT][W_OUT]
) {
#pragma HLS INLINE off

    // Zero-init output with bias
    for (int oc = 0; oc < C_OUT; ++oc) {
        for (int oh = 0; oh < H_OUT; ++oh) {
            for (int ow = 0; ow < W_OUT; ++ow) {
#pragma HLS PIPELINE II=1
                output[oc][oh][ow] = bias[oc];
            }
        }
    }

    // Convolution
    for (int oc = 0; oc < C_OUT; ++oc) {
        for (int oh = 0; oh < H_OUT; ++oh) {
            for (int ow = 0; ow < W_OUT; ++ow) {
#pragma HLS PIPELINE II=1
                float acc = output[oc][oh][ow];

                // Top-left corner in input
                int ih0 = oh * STRIDE - PADDING;
                int iw0 = ow * STRIDE - PADDING;

                for (int ic = 0; ic < C_IN; ++ic) {
                    for (int kh = 0; kh < K; ++kh) {
                        int ih = ih0 + kh;
                        if (ih < 0 || ih >= H_IN) continue;

                        for (int kw = 0; kw < K; ++kw) {
                            int iw = iw0 + kw;
                            if (iw < 0 || iw >= W_IN) continue;

                            float x = input[ic][ih][iw];
                            float w = weight[oc][ic][kh][kw];
                            acc += x * w;
                        }
                    }
                }
                output[oc][oh][ow] = acc;
            }
        }
    }
}

#endif // CONV2D_PATCH_EMBED_H
