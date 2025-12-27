#include <cstdio>
#include <cmath>
#include <vector>
#include <algorithm>
#include <cassert>

// for debugging
#ifdef _WIN32
  #include <direct.h>   // _getcwd
  #define getcwd _getcwd
#else
  #include <unistd.h>   // getcwd
#endif

extern "C" void conv2d_nchwc_stream(
    const float* in_nchwc,
    const float* w_oihw,
    float* out_nchw,
    int Cin, int Hin, int Win,
    int Cout,
    int K, int stride, int pad);

static inline int ceil_div(int a, int b) { return (a + b - 1) / b; }

static inline int idx_in_nchwc(int ci, int h, int w, int Hin, int Win, int Cb) {
  int ci_blk = ci / Cb;
  int c      = ci % Cb;
  return (((ci_blk * Hin + h) * Win + w) * Cb + c);
}

static inline int idx_w_oihw(int co, int ci, int kh, int kw, int Cin, int K) {
  return (((co * Cin + ci) * K + kh) * K + kw);
}

static inline int idx_out(int co, int oh, int ow, int Hout, int Wout) {
  return (co * Hout + oh) * Wout + ow;
}

static void ref_conv_oihw_nchwc(
    const float* in_nchwc, const float* w,
    float* out,
    int Cin, int Hin, int Win,
    int Cout, int K, int stride, int pad,
    int Cb)
{
  int Hout = (Hin + 2*pad - K)/stride + 1;
  int Wout = (Win + 2*pad - K)/stride + 1;

  for (int co = 0; co < Cout; co++) {
    for (int oh = 0; oh < Hout; oh++) {
      for (int ow = 0; ow < Wout; ow++) {
        float acc = 0.0f;
        for (int ci = 0; ci < Cin; ci++) {
          for (int kh = 0; kh < K; kh++) {
            for (int kw = 0; kw < K; kw++) {
              int ih = oh*stride + kh - pad;
              int iw = ow*stride + kw - pad;
              float x = 0.0f;
              if (ih >= 0 && ih < Hin && iw >= 0 && iw < Win) {
                x = in_nchwc[idx_in_nchwc(ci, ih, iw, Hin, Win, Cb)];
              }
              float ww = w[idx_w_oihw(co, ci, kh, kw, Cin, K)];
              acc += ww * x;
            }
          }
        }
        out[idx_out(co, oh, ow, Hout, Wout)] = acc;
      }
    }
  }
}

int main() {
    // for debugging
    char cwd[4096];
    if (getcwd(cwd, sizeof(cwd))) {
        std::printf("[TB] CWD = %s\n", cwd);
    } else {
        std::printf("[TB] CWD = <getcwd failed>\n");
    }

  constexpr int Cb = 4;

  // Small test
  int Cin = 8;
  int Cout = 8;
  int Hin = 8;
  int Win = 10;
  int K = 3;
  int stride = 1;
  int pad = 1;

  int Hout = (Hin + 2*pad - K)/stride + 1;
  int Wout = (Win + 2*pad - K)/stride + 1;

  int Cin_blk = ceil_div(Cin, Cb);

  std::vector<float> in_nchwc(Cin_blk * Hin * Win * Cb, 0.0f);
  std::vector<float> w(Cout * Cin * K * K, 0.0f);
  std::vector<float> out_hw(Cout * Hout * Wout, 0.0f);
  std::vector<float> out_ref(Cout * Hout * Wout, 0.0f);

  // Deterministic init
  for (int ci = 0; ci < Cin; ci++) {
    for (int h = 0; h < Hin; h++) {
      for (int ww = 0; ww < Win; ww++) {
        int idx = idx_in_nchwc(ci, h, ww, Hin, Win, Cb);
        in_nchwc[idx] = (float)((ci+1) * 0.1 + (h+1) * 0.01 + (ww+1) * 0.001);
      }
    }
  }
  for (int co = 0; co < Cout; co++) {
    for (int ci = 0; ci < Cin; ci++) {
      for (int kh = 0; kh < K; kh++) {
        for (int kw = 0; kw < K; kw++) {
          w[idx_w_oihw(co, ci, kh, kw, Cin, K)] =
              (float)((co+1) * 0.01 + (ci+1) * 0.001 + kh * 0.0001 + kw * 0.00001);
        }
      }
    }
  }

  ref_conv_oihw_nchwc(in_nchwc.data(), w.data(), out_ref.data(),
                      Cin, Hin, Win, Cout, K, stride, pad, Cb);

  conv2d_nchwc_stream(in_nchwc.data(), w.data(), out_hw.data(),
                      Cin, Hin, Win, Cout, K, stride, pad);

  float max_abs = 0.0f;
  for (int i = 0; i < (int)out_hw.size(); i++) {
    max_abs = std::max(max_abs, std::fabs(out_hw[i] - out_ref[i]));
  }

  std::printf("Max abs error: %.9g\n", max_abs);
  assert(max_abs < 1e-4f);

  std::puts("PASS");
  return 0;
}
