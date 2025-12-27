#include <hls_stream.h>
#include <cassert>

#include "conv2d_cfg.hpp"
#include "conv2d_stream_types.hpp"

namespace tinyvit_hls {

// -------------------------------
// WeightTileLoader: Option A layout W_tile[p][k]
// k = ci*(K*K) + kh*K + kw
// -------------------------------
template<typename CFG>
static void load_weights_tile_oihw(
    const typename CFG::data_t* w, // [Cout][Cin][K][K]
    typename CFG::data_t W_tile[CFG::P_TILE][CFG::MAX_KDIM],
    int co0, int Cout,
    int Cin, int K)
{
#pragma HLS INLINE off

  const int KDIM = Cin * K * K;

  // For each p lane
  for (int p = 0; p < CFG::P_TILE; p++) {
    const int co = co0 + p;

    // Fill used KDIM. If tail channel, fill zeros.
    for (int k = 0; k < KDIM; k++) {
#pragma HLS PIPELINE II=1
      if (co >= Cout) {
        W_tile[p][k] = 0.0f;
      } else {
        // decode k -> (ci, kh, kw)
        int tmp = k;
        const int kw = tmp % K; tmp /= K;
        const int kh = tmp % K; tmp /= K;
        const int ci = tmp;

        W_tile[p][k] = w[idx_w_oihw(co, ci, kh, kw, Cin, K)];
      }
    }
  }
}

// -------------------------------
// Line-buffered Conv XVec Producer for one strip segment (NCHWc):
// - Processes one input channel block (Cb channels) at a time.
// - For each ci_blk, "line-buffers" the K rows needed for this output row oh.
// - Then streams XVec tokens across width groups (Q_PAR) for each (ci,kh,kw).
//
// Stream order produced:
//   for ci_blk:
//     for ci_in (0..ci_eff-1):
//       for kh:
//         for kw:
//           for owg_off (0..ow_seg_eff step Q_PAR):
//             write XVec (x[q] for q lanes)
//
// This order must be matched by the compute core below.
// -------------------------------
template<typename CFG>
static void xvec_producer_conv_linebuf_segment(
    const typename CFG::data_t* in_nchwc, // [Cin_blk][Hin][Win][Cb]
    hls::stream< XVec<typename CFG::data_t, CFG::Q_PAR> >& xstr,
    int Cin, int Hin, int Win,
    int K, int stride, int pad,
    int oh,
    int ow0,
    int ow_seg_eff)
{
#pragma HLS INLINE off

  const int Cin_blk = ceil_div(Cin, CFG::Cb);

  // Line buffer for the current channel block only:
  // linebuf[c_in_blk][kh][iw]
  typename CFG::data_t linebuf[CFG::Cb][CFG::MAX_K][CFG::MAX_WIN];
#pragma HLS ARRAY_PARTITION variable=linebuf complete dim=1
#pragma HLS BIND_STORAGE variable=linebuf type=ram_t2p impl=bram

  // Process one channel block at a time (CI_TILE = Cb)
  for (int ci_blk = 0; ci_blk < Cin_blk; ci_blk++) {

    const int ci_base = ci_blk * CFG::Cb;
    const int ci_eff  = imin(CFG::Cb, Cin - ci_base);

    // ---------------------------
    // Preload the K required rows for this output row (oh)
    // into linebuf for this ci_blk.
    // ---------------------------
    for (int kh = 0; kh < K; kh++) {
      const int ih = oh * stride + kh - pad;

      // For padding rows: fill zeros
      if (ih < 0 || ih >= Hin) {
        for (int iw = 0; iw < Win; iw++) {
#pragma HLS PIPELINE II=1
          for (int c = 0; c < CFG::Cb; c++) {
#pragma HLS UNROLL
            linebuf[c][kh][iw] = 0.0f;
          }
        }
      } else {
        // Valid row: read NCHWc contiguous Cb lanes per pixel
        for (int iw = 0; iw < Win; iw++) {
#pragma HLS PIPELINE II=1
          // Read all Cb lanes (guard ci_eff for tail block)
          for (int c = 0; c < CFG::Cb; c++) {
#pragma HLS UNROLL
            if (c < ci_eff) {
              // direct NCHWc address (ci = ci_base + c)
              const int ci = ci_base + c;
              linebuf[c][kh][iw] = in_nchwc[idx_in_nchwc<CFG>(ci, ih, iw, Hin, Win)];
            } else {
              // padded channel lane (tail): zero
              linebuf[c][kh][iw] = 0.0f;
            }
          }
        }
      }
    }

    // ---------------------------
    // Emit XVec tokens in the agreed streaming-friendly order
    // ---------------------------
    for (int c = 0; c < ci_eff; c++) {
      // ci = ci_base + c
      for (int kh = 0; kh < K; kh++) {
        for (int kw = 0; kw < K; kw++) {
          for (int owg_off = 0; owg_off < ow_seg_eff; owg_off += CFG::Q_PAR) {
#pragma HLS PIPELINE II=1
            const int owg  = ow0 + owg_off;
            const int q_eff = imin(CFG::Q_PAR, ow_seg_eff - owg_off);

            XVec<typename CFG::data_t, CFG::Q_PAR> xv;

            for (int q = 0; q < CFG::Q_PAR; q++) {
#pragma HLS UNROLL
              typename CFG::data_t x = 0.0f;

              if (q < q_eff) {
                const int ow = owg + q;
                const int iw = ow * stride + kw - pad;

                if (iw >= 0 && iw < Win) {
                  x = linebuf[c][kh][iw];
                } else {
                  x = 0.0f;
                }
              } else {
                x = 0.0f;
              }

              xv.x[q] = x;
            }

            xstr.write(xv);
          }
        }
      }
    }
  }
}

// -------------------------------
// Naive Conv XVec Producer for one strip segment:
// Emits XVec tokens for each micro-tile (Q_PAR outputs) and each k-step.
// Later replacement: true line-buffered NCHWc strip streamer.
// -------------------------------
template<typename CFG>
static void xvec_producer_conv_naive_segment(
    const typename CFG::data_t* in_nchwc, // [Cin_blk][Hin][Win][Cb]
    hls::stream< XVec<typename CFG::data_t, CFG::Q_PAR> >& xstr,
    int Cin, int Hin, int Win,
    int Hout, int Wout,
    int K, int stride, int pad,
    int oh,
    int ow0,
    int ow_seg_eff) // effective segment width within OW_TILE
{
#pragma HLS INLINE off

  const int KDIM = Cin * K * K;

  // Iterate micro-tiles across the strip: owg is the group base
  for (int owg_off = 0; owg_off < ow_seg_eff; owg_off += CFG::Q_PAR) {
    const int owg = ow0 + owg_off;
    const int q_eff = imin(CFG::Q_PAR, ow_seg_eff - owg_off);

    // Reduction steps
    for (int k = 0; k < KDIM; k++) {
#pragma HLS PIPELINE II=1
      int tmp = k;
      const int kw = tmp % K; tmp /= K;
      const int kh = tmp % K; tmp /= K;
      const int ci = tmp;

      XVec<typename CFG::data_t, CFG::Q_PAR> xv;

      // Fill Q lanes for this k-step
      for (int q = 0; q < CFG::Q_PAR; q++) {
#pragma HLS UNROLL
        typename CFG::data_t x = 0.0f;

        if (q < q_eff) {
          const int ow = owg + q;

          const int ih = oh * stride + kh - pad;
          const int iw = ow * stride + kw - pad;

          if (ih >= 0 && ih < Hin && iw >= 0 && iw < Win) {
            // NCHWc scalar access
            x = in_nchwc[idx_in_nchwc<CFG>(ci, ih, iw, Hin, Win)];
          } else {
            x = 0.0f;
          }
        } else {
          // invalid lane (tail); keep zero
          x = 0.0f;
        }

        xv.x[q] = x;
      }

      xstr.write(xv);
    }
  }
}

// -------------------------------
// ComputeCore for one strip segment (accumulates across the whole OW segment):
// Consumes XVec tokens in the SAME order as the linebuf producer above:
//   for ci_blk:
//     for ci_in (0..ci_eff-1):
//       for kh:
//         for kw:
//           for owg_off:
//             read XVec and update acc[p][owg_off+q]
//
// After accumulation, emits outputs in q-major then p order per microtile,
// matching the existing OutputWriterBuffered fill order.
// -------------------------------
template<typename CFG>
static void compute_core_os_stripacc_segment(
    const typename CFG::data_t W_tile[CFG::P_TILE][CFG::MAX_KDIM],
    hls::stream< XVec<typename CFG::data_t, CFG::Q_PAR> >& xstr,
    hls::stream<typename CFG::data_t>& ystr,
    int co0, int Cout,
    int Cin, int K,
    int ow_seg_eff)
{
#pragma HLS INLINE off

  const int p_eff   = imin(CFG::P_TILE, Cout - co0);
  const int Cin_blk = ceil_div(Cin, CFG::Cb);

  // Accumulate for the entire segment width
  typename CFG::acc_t acc[CFG::P_TILE][CFG::OW_TILE];
#pragma HLS ARRAY_PARTITION variable=acc complete dim=1
#pragma HLS BIND_STORAGE variable=acc type=ram_1p impl=lutram

  // init acc for used columns
  for (int p = 0; p < CFG::P_TILE; p++) {
#pragma HLS UNROLL
    for (int t = 0; t < ow_seg_eff; t++) {
#pragma HLS PIPELINE II=1
      acc[p][t] = 0.0f;
    }
  }

  // Consume tokens in the producer's order
  for (int ci_blk = 0; ci_blk < Cin_blk; ci_blk++) {
    const int ci_base = ci_blk * CFG::Cb;
    const int ci_eff  = imin(CFG::Cb, Cin - ci_base);

    for (int c = 0; c < ci_eff; c++) {
      const int ci = ci_base + c;

      for (int kh = 0; kh < K; kh++) {
        for (int kw = 0; kw < K; kw++) {

          const int k_global = (ci * K * K) + (kh * K) + kw;

          for (int owg_off = 0; owg_off < ow_seg_eff; owg_off += CFG::Q_PAR) {
#pragma HLS PIPELINE II=1
            const int q_eff = imin(CFG::Q_PAR, ow_seg_eff - owg_off);

            const auto xv = xstr.read();

            for (int p = 0; p < CFG::P_TILE; p++) {
#pragma HLS UNROLL
              if (p < p_eff) {
                const typename CFG::data_t wpk = W_tile[p][k_global];

                for (int q = 0; q < CFG::Q_PAR; q++) {
#pragma HLS UNROLL
                  if (q < q_eff) {
                    acc[p][owg_off + q] += (typename CFG::acc_t)wpk * (typename CFG::acc_t)xv.x[q];
                  }
                }
              }
            }
          }
        }
      }
    }
  }

  // Emit outputs in q-major then p order per microtile (same as before)
  for (int owg_off = 0; owg_off < ow_seg_eff; owg_off += CFG::Q_PAR) {
    const int q_eff = imin(CFG::Q_PAR, ow_seg_eff - owg_off);

    for (int q = 0; q < q_eff; q++) {
      for (int p = 0; p < p_eff; p++) {
#pragma HLS PIPELINE II=1
        ystr.write((typename CFG::data_t)acc[p][owg_off + q]);
      }
    }
  }
}

// -------------------------------
// OutputWriter for one strip segment with buffering for burst-friendly writes:
// Collects outputs into obuf[P][OW_TILE] in spatial order, then writes per p contiguously.
// Stream order is q-major then p per micro-tile.
// -------------------------------
template<typename CFG>
static void output_writer_strip_buffered(
    typename CFG::data_t* out_nchw, // [Cout][Hout][Wout]
    hls::stream<typename CFG::data_t>& ystr,
    int co0, int Cout,
    int Hout, int Wout,
    int oh,
    int ow0,
    int ow_seg_eff)
{
#pragma HLS INLINE off

  const int p_eff = imin(CFG::P_TILE, Cout - co0);

  typename CFG::data_t obuf[CFG::P_TILE][CFG::OW_TILE];
#pragma HLS ARRAY_PARTITION variable=obuf complete dim=1

  // Fill buffer in spatial order across the segment
  for (int owg_off = 0; owg_off < ow_seg_eff; owg_off += CFG::Q_PAR) {
    const int q_eff = imin(CFG::Q_PAR, ow_seg_eff - owg_off);

    for (int q = 0; q < q_eff; q++) {
      for (int p = 0; p < p_eff; p++) {
#pragma HLS PIPELINE II=1
        obuf[p][owg_off + q] = ystr.read();
      }
    }
  }

  // Burst-friendly writes: for each p lane, write contiguous [ow0 : ow0+ow_seg_eff)
  for (int p = 0; p < p_eff; p++) {
    const int co = co0 + p;
    const int base = idx_out_nchw(co, oh, ow0, Hout, Wout);
    for (int t = 0; t < ow_seg_eff; t++) {
#pragma HLS PIPELINE II=1
      out_nchw[base + t] = obuf[p][t];
    }
  }
}

// -------------------------------
// Process one (co0, oh, ow0) strip segment as a DATAFLOW region.
// -------------------------------
template<typename CFG>
static void process_strip_segment_dataflow(
    const typename CFG::data_t* in_nchwc,
    const typename CFG::data_t W_tile[CFG::P_TILE][CFG::MAX_KDIM],
    typename CFG::data_t* out_nchw,
    int Cin, int Hin, int Win,
    int Hout, int Wout,
    int Cout,
    int K, int stride, int pad,
    int co0,
    int oh,
    int ow0,
    int ow_seg_eff)
{
#pragma HLS INLINE off

  hls::stream< XVec<typename CFG::data_t, CFG::Q_PAR> > xstr("xstr");
  hls::stream<typename CFG::data_t> ystr("ystr");
#pragma HLS STREAM variable=xstr depth=CFG::X_STREAM_DEPTH
#pragma HLS STREAM variable=ystr depth=CFG::Y_STREAM_DEPTH

#pragma HLS DATAFLOW
  xvec_producer_conv_linebuf_segment<CFG>(in_nchwc, xstr,
                                          Cin, Hin, Win,
                                          K, stride, pad,
                                          oh, ow0, ow_seg_eff);
  
  compute_core_os_stripacc_segment<CFG>(W_tile, xstr, ystr,
                                        co0, Cout,
                                        Cin, K,
                                        ow_seg_eff);

  output_writer_strip_buffered<CFG>(out_nchw, ystr,
                                   co0, Cout,
                                   Hout, Wout,
                                   oh, ow0, ow_seg_eff);
}

} // namespace tinyvit_hls

// -------------------------------
// Top-level HLS kernel
// Input: NCHWc (Cb=4), N=1
// Weights: OIHW
// Output: NCHW, N=1
// -------------------------------
extern "C" {
void conv2d_nchwc_stream(
    const float* in_nchwc,
    const float* w_oihw,
    float* out_nchw,
    int Cin, int Hin, int Win,
    int Cout,
    int K, int stride, int pad)
{
#pragma HLS INTERFACE m_axi port=in_nchwc  offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=w_oihw    offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi port=out_nchw  offset=slave bundle=gmem2

#pragma HLS INTERFACE s_axilite port=in_nchwc bundle=control
#pragma HLS INTERFACE s_axilite port=w_oihw   bundle=control
#pragma HLS INTERFACE s_axilite port=out_nchw bundle=control
#pragma HLS INTERFACE s_axilite port=Cin      bundle=control
#pragma HLS INTERFACE s_axilite port=Hin      bundle=control
#pragma HLS INTERFACE s_axilite port=Win      bundle=control
#pragma HLS INTERFACE s_axilite port=Cout     bundle=control
#pragma HLS INTERFACE s_axilite port=K        bundle=control
#pragma HLS INTERFACE s_axilite port=stride   bundle=control
#pragma HLS INTERFACE s_axilite port=pad      bundle=control
#pragma HLS INTERFACE s_axilite port=return   bundle=control

  using CFG = tinyvit_hls::Conv2DConfigDefault;

#ifndef __SYNTHESIS__
  // Basic parameter guards for C-sim sanity
  assert(K == 1 || K == 3);
  assert(K <= CFG::MAX_K);
  assert(Hin <= CFG::MAX_HIN);
  assert(Win <= CFG::MAX_WIN);
  assert(Cin <= CFG::MAX_CIN);
  assert(Cout <= CFG::MAX_COUT);
  assert(CFG::OW_TILE % CFG::Q_PAR == 0);
#endif

  const int Hout = (Hin + 2 * pad - K) / stride + 1;
  const int Wout = (Win + 2 * pad - K) / stride + 1;

  // On-chip weight tile cache
  static CFG::data_t W_tile[CFG::P_TILE][CFG::MAX_KDIM];
#pragma HLS ARRAY_PARTITION variable=W_tile complete dim=1
#pragma HLS BIND_STORAGE variable=W_tile type=ram_t2p impl=bram

  for (int co0 = 0; co0 < Cout; co0 += CFG::P_TILE) {
    tinyvit_hls::load_weights_tile_oihw<CFG>(w_oihw, W_tile, co0, Cout, Cin, K);

    // Strip streaming over output rows and width segments
    for (int oh = 0; oh < Hout; oh++) {
      for (int ow0 = 0; ow0 < Wout; ow0 += CFG::OW_TILE) {
        const int ow_seg_eff = tinyvit_hls::imin(CFG::OW_TILE, Wout - ow0);

        tinyvit_hls::process_strip_segment_dataflow<CFG>(
            in_nchwc, W_tile, out_nchw,
            Cin, Hin, Win,
            Hout, Wout,
            Cout,
            K, stride, pad,
            co0, oh, ow0, ow_seg_eff);
      }
    }
  }
}
}
