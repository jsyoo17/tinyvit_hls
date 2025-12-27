#pragma once

namespace tinyvit_hls {

// Unified ComputeCore input token: one K-step worth of activations for Q_PAR lanes.
template<typename T, int Q>
struct XVec {
  T x[Q];
};

} // namespace tinyvit_hls
