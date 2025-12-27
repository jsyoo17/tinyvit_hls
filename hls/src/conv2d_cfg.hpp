#pragma once

#include <cstdint>
#include <cassert>

namespace tinyvit_hls {

// Conservative KV260-friendly defaults for initial bring-up.
struct Conv2DConfigDefault {
  using data_t = float;
  using acc_t  = float;

  // Channel blocking / tiling
  static constexpr int Cb      = 4;   // NCHWc channel block
  static constexpr int CI_TILE = Cb;  // fixed by design

  // PE geometry (compile-time)
  static constexpr int P_TILE  = 4;   // output-channel lanes
  static constexpr int Q_PAR   = 4;   // output-width lanes (parallel outputs per cycle)

  // Strip length for burst-friendly writes (time-multiplexed)
  static constexpr int OW_TILE = 64;  // must be >= Q_PAR; prefer multiple of Q_PAR

  // Capacity bounds (compile-time)
  static constexpr int MAX_K   = 3;     // support 1x1 and 3x3 initially
  static constexpr int MAX_WIN = 224;
  static constexpr int MAX_HIN = 224;
  static constexpr int MAX_CIN = 320;
  static constexpr int MAX_COUT= 320;

  static constexpr int MAX_KDIM = MAX_CIN * MAX_K * MAX_K;

  // Stream depths (tunable; conservative)
  static constexpr int X_STREAM_DEPTH = 512;
  static constexpr int Y_STREAM_DEPTH = 128;
};

// -------------------------------
// Helpers: ceil-div, min
// -------------------------------
static inline int ceil_div(int a, int b) {
  return (a + b - 1) / b;
}

static inline int imin(int a, int b) { return (a < b) ? a : b; }

// -------------------------------
// Indexing helpers
// -------------------------------

// Input layout: NCHWc, N=1
// in[ci_blk][h][w][c], where ci_blk = ci/Cb, c = ci%Cb
template<typename CFG>
static inline int idx_in_nchwc(int ci, int h, int w, int Hin, int Win) {
  const int ci_blk = ci / CFG::Cb;
  const int c      = ci - ci_blk * CFG::Cb;
  return (((ci_blk * Hin + h) * Win + w) * CFG::Cb + c);
}

// Weights layout (baseline): OIHW contiguous
// w[co][ci][kh][kw]
static inline int idx_w_oihw(int co, int ci, int kh, int kw, int Cin, int K) {
  return (((co * Cin + ci) * K + kh) * K + kw);
}

// Output layout: NCHW contiguous (N=1)
// out[co][oh][ow]
static inline int idx_out_nchw(int co, int oh, int ow, int Hout, int Wout) {
  return (co * Hout + oh) * Wout + ow;
}

} // namespace tinyvit_hls
