// #838: widen Gemma-4 FP8 indexed MoE from T==1 to T<=63.
// Pure host predicates + token offsets. No HIP. Product caches env once;
// tests call the parse/predicate functions directly.
#pragma once

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <vector>

namespace vllm {

constexpr int64_t kGemma4PrefillBatchMinT = 64;
constexpr int64_t kGemma4DecodeIndexedMaxTDefault = 63;
constexpr int64_t kGemma4DecodeIndexedMaxTLo = 1;
constexpr int64_t kGemma4DecodeIndexedMaxTHi = 63;

// Oracle class for the default-on numerical route change (spec #838).
// abs_tol = 2^-7 * max_abs(ref); rel_tol unused beyond that product form.
constexpr float kGemma4IndexedOraclePow2 = 7.0f;

inline int64_t ParseGemma4DecodeIndexedMaxT(const char* e) {
  if (e == nullptr || e[0] == '\0') return kGemma4DecodeIndexedMaxTDefault;
  char* end = nullptr;
  const long v = std::strtol(e, &end, 10);
  if (end == e) return kGemma4DecodeIndexedMaxTDefault;
  if (v < kGemma4DecodeIndexedMaxTLo) return kGemma4DecodeIndexedMaxTLo;
  if (v > kGemma4DecodeIndexedMaxTHi) return kGemma4DecodeIndexedMaxTHi;
  return static_cast<int64_t>(v);
}

inline int64_t Gemma4DecodeIndexedMaxT() {
  static const int64_t n = ParseGemma4DecodeIndexedMaxT(std::getenv("VT_GEMMA4_DECODE_INDEXED_MAX_T"));
  return n;
}

inline bool Gemma4IndexedOkT(int64_t T, int64_t indexed_max_t, int top_k, bool fp8_res) {
  return T >= 1 && T <= indexed_max_t && T < kGemma4PrefillBatchMinT && fp8_res &&
         top_k <= 8 && top_k > 0;
}

struct Gemma4IndexedTokenOff {
  int64_t x_elems = 0;  // t * H into [T,H] bf16
  int64_t y_elems = 0;
  int64_t route = 0;    // t * top_k into [T,K] idx/wts
};

inline Gemma4IndexedTokenOff Gemma4IndexedTokenOffsets(int64_t t, int64_t H, int top_k) {
  return Gemma4IndexedTokenOff{t * H, t * H, t * static_cast<int64_t>(top_k)};
}

// Test-visible route witness: incremented once per successful per-token indexed helper.
inline std::atomic<uint64_t>& Gemma4IndexedHelperHits() {
  static std::atomic<uint64_t> n{0};
  return n;
}

// Host serial/reference mix vs indexed loop. Fake expert: y[h] = x[h] * (e+1) * w.
// Same math both arms — proves token stride / ownership, not HIP numerics.
inline void Gemma4IndexedHostSerialRef(const float* x, const int32_t* idx, const float* wts,
                                       float* y, int64_t T, int64_t H, int top_k) {
  for (int64_t t = 0; t < T; ++t) {
    for (int64_t h = 0; h < H; ++h) y[t * H + h] = 0.f;
    for (int g = 0; g < top_k; ++g) {
      const int32_t e = idx[t * top_k + g];
      const float w = wts[t * top_k + g];
      const float s = (e + 1) * w;
      for (int64_t h = 0; h < H; ++h) y[t * H + h] += x[t * H + h] * s;
    }
  }
}

inline void Gemma4IndexedHostIndexedLoop(const float* x, const int32_t* idx, const float* wts,
                                         float* y, int64_t T, int64_t H, int top_k) {
  for (int64_t t = 0; t < T; ++t) {
    const auto off = Gemma4IndexedTokenOffsets(t, H, top_k);
    for (int64_t h = 0; h < H; ++h) y[off.y_elems + h] = 0.f;
    for (int g = 0; g < top_k; ++g) {
      const int32_t e = idx[off.route + g];
      const float w = wts[off.route + g];
      const float s = (e + 1) * w;
      for (int64_t h = 0; h < H; ++h) y[off.y_elems + h] += x[off.x_elems + h] * s;
    }
  }
}

inline bool Gemma4IndexedOracleClose(const float* cand, const float* ref, int64_t n, float* max_abs_out) {
  float max_abs_ref = 0.f;
  float max_abs_diff = 0.f;
  for (int64_t i = 0; i < n; ++i) {
    if (!std::isfinite(cand[i]) || !std::isfinite(ref[i])) return false;
    max_abs_ref = std::max(max_abs_ref, std::fabs(ref[i]));
    max_abs_diff = std::max(max_abs_diff, std::fabs(cand[i] - ref[i]));
    if (ref[i] == 0.f && cand[i] != 0.f) return false;
  }
  if (max_abs_out) *max_abs_out = max_abs_diff;
  const float tol = std::ldexp(max_abs_ref, -static_cast<int>(kGemma4IndexedOraclePow2));
  return max_abs_diff <= tol;
}

}  // namespace vllm
