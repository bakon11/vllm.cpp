// #838: widen Gemma-4 FP8 indexed MoE from T==1 to T<=63.
// Host-injectable dispatch, tensor oracle, single-scale fallback, scratch retire.
// No HIP.
#pragma once

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <cstddef>
#include <algorithm>
#include <vector>

namespace vllm {

constexpr int64_t kGemma4PrefillBatchMinT = 64;
constexpr int64_t kGemma4DecodeIndexedMaxTDefault = 63;
constexpr int64_t kGemma4DecodeIndexedMaxTLo = 1;
constexpr int64_t kGemma4DecodeIndexedMaxTHi = 63;
constexpr float kGemma4IndexedOraclePow2 = 7.0f;  // abs_tol = 2^-7 * max_abs(ref)

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
  int64_t x_elems = 0;
  int64_t y_elems = 0;
  int64_t route = 0;
};

inline Gemma4IndexedTokenOff Gemma4IndexedTokenOffsets(int64_t t, int64_t H, int top_k) {
  return Gemma4IndexedTokenOff{t * H, t * H, t * static_cast<int64_t>(top_k)};
}

inline std::atomic<uint64_t>& Gemma4IndexedHelperHits() {
  static std::atomic<uint64_t> n{0};
  return n;
}

template <typename YT = uint16_t, typename XT = uint16_t>
struct Gemma4IndexedCall {
  int64_t t = 0;
  Gemma4IndexedTokenOff off{};
  bool peer = false;
  const XT* x = nullptr;
  YT* y = nullptr;
  const int32_t* ri = nullptr;
  const float* rw = nullptr;
};

struct Gemma4IndexedDispatchResult {
  bool ok = false;
  uint64_t hits = 0;
  int restores = 0;
  void* y_owner = nullptr;
  bool enqueued = false;
};

// Production T-loop. `fn(call)` is the same-dev or peer helper. `restore()` runs
// after every helper return (success or fail). Does not free `y`.
template <typename YT, typename XT, typename Fn, typename Restore>
inline Gemma4IndexedDispatchResult Gemma4IndexedDispatchTokens(
    int64_t T, int64_t H, int top_k, bool peer, YT* y, const XT* x, const int32_t* ri,
    const float* rw, Fn fn, Restore restore) {
  Gemma4IndexedDispatchResult r;
  r.y_owner = y;
  if (T <= 0 || !y || !x || !ri || !rw) return r;
  for (int64_t t = 0; t < T; ++t) {
    const auto off = Gemma4IndexedTokenOffsets(t, H, top_k);
    const Gemma4IndexedCall<YT, XT> c{t, off, peer, x + off.x_elems, y + off.y_elems,
                                      ri + off.route, rw + off.route};
    const bool one = fn(c);
    r.enqueued = true;
    restore();
    ++r.restores;
    if (!one) {
      r.ok = false;
      return r;
    }
    ++r.hits;
    Gemma4IndexedHelperHits().fetch_add(1, std::memory_order_relaxed);
  }
  r.ok = true;
  return r;
}

// Serial fallback scale. already_scaled=true skips (indexed must not leave rw mutated).
inline void Gemma4ApplyHostExpertScaleOnce(float* hw, const int32_t* hi, const float* hscale,
                                           int64_t T, int top_k, int64_t E, bool already_scaled) {
  if (already_scaled || !hw || !hi || !hscale || T <= 0 || top_k <= 0) return;
  for (int64_t t = 0; t < T; ++t) {
    for (int i = 0; i < top_k; ++i) {
      const size_t o = static_cast<size_t>(t * top_k + i);
      const int e = hi[o];
      if (e >= 0 && e < static_cast<int>(E)) hw[o] *= hscale[static_cast<size_t>(e)];
    }
  }
}

// Independent serial / product-loop token math (host oracle).
inline void Gemma4IndexedHostApplyToken(float* y, const float* x, const int32_t* ri, const float* rw,
                                        int64_t H, int top_k) {
  for (int64_t h = 0; h < H; ++h) y[h] = 0.f;
  for (int g = 0; g < top_k; ++g) {
    const int32_t e = ri[g];
    const float s = static_cast<float>(e + 1) * rw[g];
    for (int64_t h = 0; h < H; ++h) y[h] += x[h] * s;
  }
}

inline void Gemma4IndexedHostSerialRef(const float* x, const int32_t* idx, const float* wts,
                                       float* y, int64_t T, int64_t H, int top_k) {
  for (int64_t t = 0; t < T; ++t) {
    Gemma4IndexedHostApplyToken(y + t * H, x + t * H, idx + t * top_k, wts + t * top_k, H, top_k);
  }
}

inline bool Gemma4IndexedOracleClose(const float* cand, const float* ref, int64_t n,
                                     float* max_abs_out) {
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

// Scratch may not return to a reusable pool until peer/compute work is retired.
struct Gemma4IndexedScratchLedger {
  bool enqueued = false;
  bool retired = false;
  bool released_to_pool = false;
};

inline void Gemma4IndexedRetireScratch(Gemma4IndexedScratchLedger& L) { L.retired = true; }

inline bool Gemma4IndexedReleaseScratchToPool(Gemma4IndexedScratchLedger& L) {
  if (L.enqueued && !L.retired) return false;
  L.released_to_pool = true;
  return true;
}

template <typename Retire, typename Release>
inline bool Gemma4IndexedOnHelperFail(bool enqueued, Retire retire, Release release) {
  if (enqueued) retire();
  release();
  return true;
}

}  // namespace vllm
