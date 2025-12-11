// hls/tb/tb_conv2d_streaming.cpp

#include <cstdio>
#include <cmath>

// Must match the config in conv2d_streaming_im2col.cpp
typedef float data_t;

#define C_IN     8
#define C_OUT    16
#define H_IN     8
#define W_IN     8
#define K        3
#define STRIDE   1
#define PADDING  1
#define H_OUT    ((H_IN + 2*PADDING - K) / STRIDE + 1)
#define W_OUT    ((W_IN + 2*PADDING - K) / STRIDE + 1)

// Prototype of HLS top
extern "C" void conv2d_nchw_streaming_top(
    const data_t *x_flat,
    const data_t *w_flat,
    const data_t *b_flat,
    data_t *y_flat
);

// Simple golden conv2d (N=1, groups=1, NCHW)
void conv2d_golden(
    const data_t x[C_IN][H_IN][W_IN],
    const data_t w[C_OUT][C_IN][K][K],
    const data_t b[C_OUT],
    data_t y[C_OUT][H_OUT][W_OUT]
) {
    // Zero output
    for (int co = 0; co < C_OUT; ++co) {
        for (int oh = 0; oh < H_OUT; ++oh) {
            for (int ow = 0; ow < W_OUT; ++ow) {
                y[co][oh][ow] = 0.0f;
            }
        }
    }

    for (int co = 0; co < C_OUT; ++co) {
        for (int oh = 0; oh < H_OUT; ++oh) {
            for (int ow = 0; ow < W_OUT; ++ow) {
                float acc = 0.0f;
                int in_h0 = oh * STRIDE - PADDING;
                int in_w0 = ow * STRIDE - PADDING;
                for (int ci = 0; ci < C_IN; ++ci) {
                    for (int kh = 0; kh < K; ++kh) {
                        for (int kw = 0; kw < K; ++kw) {
                            int ih = in_h0 + kh;
                            int iw = in_w0 + kw;
                            float xv = 0.0f;
                            if (ih >= 0 && ih < H_IN && iw >= 0 && iw < W_IN) {
                                xv = x[ci][ih][iw];
                            }
                            acc += xv * w[co][ci][kh][kw];
                        }
                    }
                }
                acc += b[co];
                y[co][oh][ow] = acc;
            }
        }
    }
}

int main() {
    const int X_SIZE = C_IN * H_IN * W_IN;
    const int W_SIZE = C_OUT * C_IN * K * K;
    const int B_SIZE = C_OUT;
    const int Y_SIZE = C_OUT * H_OUT * W_OUT;

    static data_t x_flat[X_SIZE];
    static data_t w_flat[W_SIZE];
    static data_t b_flat[B_SIZE];
    static data_t y_flat[Y_SIZE];

    static data_t x[C_IN][H_IN][W_IN];
    static data_t w[C_OUT][C_IN][K][K];
    static data_t b[C_OUT];
    static data_t y_golden[C_OUT][H_OUT][W_OUT];

    // Init input with deterministic values
    int idx = 0;
    for (int c = 0; c < C_IN; ++c) {
        for (int h = 0; h < H_IN; ++h) {
            for (int ww = 0; ww < W_IN; ++ww) {
                float val = (float)((c + 1) * 0.1 + h * 0.01 + ww * 0.001);
                x_flat[idx++] = val;
            }
        }
    }

    // Init weights
    idx = 0;
    for (int co = 0; co < C_OUT; ++co) {
        for (int ci = 0; ci < C_IN; ++ci) {
            for (int kh = 0; kh < K; ++kh) {
                for (int kw = 0; kw < K; ++kw) {
                    float val = (float)((co + 1) * 0.05 + (ci + 1) * 0.02
                                        + kh * 0.01 + kw * 0.005);
                    w_flat[idx++] = val;
                }
            }
        }
    }

    // Init bias
    for (int co = 0; co < C_OUT; ++co) {
        b_flat[co] = (float)(0.01 * (co + 1));
    }

    // Copy to 4D/3D arrays for golden
    idx = 0;
    for (int c = 0; c < C_IN; ++c) {
        for (int h = 0; h < H_IN; ++h) {
            for (int ww = 0; ww < W_IN; ++ww) {
                x[c][h][ww] = x_flat[idx++];
            }
        }
    }

    idx = 0;
    for (int co = 0; co < C_OUT; ++co) {
        for (int ci = 0; ci < C_IN; ++ci) {
            for (int kh = 0; kh < K; ++kh) {
                for (int kw = 0; kw < K; ++kw) {
                    w[co][ci][kh][kw] = w_flat[idx++];
                }
            }
        }
    }

    for (int co = 0; co < C_OUT; ++co) {
        b[co] = b_flat[co];
    }

    // Call HLS top
    conv2d_nchw_streaming_top(x_flat, w_flat, b_flat, y_flat);

    // Golden conv
    conv2d_golden(x, w, b, y_golden);

    // Compare
    float max_diff = 0.0f;
    float mean_diff = 0.0f;
    int count = 0;
    idx = 0;
    for (int co = 0; co < C_OUT; ++co) {
        for (int oh = 0; oh < H_OUT; ++oh) {
            for (int ow = 0; ow < W_OUT; ++ow) {
                float hw_val = y_flat[idx++];
                float gd_val = y_golden[co][oh][ow];
                float diff = std::fabs(hw_val - gd_val);
                if (diff > max_diff) max_diff = diff;
                mean_diff += diff;
                count++;
            }
        }
    }
    mean_diff /= (float)count;

    std::printf("C_OUT=%d, C_IN=%d, H_IN=%d, W_IN=%d, K=%d\n",
                C_OUT, C_IN, H_IN, W_IN, K);
    std::printf("H_OUT=%d, W_OUT=%d\n", H_OUT, W_OUT);
    std::printf("max_diff  = %e\n", max_diff);
    std::printf("mean_diff = %e\n", mean_diff);

    float tol = 1e-4f;
    if (max_diff > tol) {
        std::printf("TEST FAILED (max_diff > tol).\n");
        return 1;
    } else {
        std::printf("TEST PASSED (within tolerance).\n");
        return 0;
    }
}
