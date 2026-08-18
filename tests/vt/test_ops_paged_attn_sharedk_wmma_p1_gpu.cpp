// #785 P1 GPU product-seam witness. Calls vt::PagedAttention on kROCM.
// Does NOT launch PagedAttnPrefillSharedKWmma directly.
// Run only via tests/scripts/run-785-p1.sh after Researcher GPU GO.
// Missing device or missing VT_785_P1_GPU => exit 5 (fail closed, not SKIP 77).
#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

#include "vt/backend.h"
#include "vt/dtype.h"
#include "vt/ops.h"
#include "vt/rocm/rocm_runtime.h"

using vt::Backend;
using vt::Device;
using vt::DeviceType;
using vt::DType;
using vt::PagedAttentionArgs;
using vt::Queue;
using vt::Tensor;

namespace {

constexpr int64_t kT = 64;
constexpr int64_t kHq = 2;
constexpr int64_t kHk = 1;
constexpr int64_t kD = 256;
constexpr int64_t kBlock = 16;
constexpr float kScale = 0.0625f;
constexpr int64_t kWindowLeft = 32;
constexpr int64_t kWindowRight = 0;
constexpr uint32_t kQSeed = 78525601u;
constexpr uint32_t kKSeed = 78525602u;
constexpr uint32_t kVSeed = 78525603u;
constexpr double kAbsFloor = 1.5e-2;
constexpr double kRel = 1.0e-2;
constexpr double kMinCorr = 0.999;

[[noreturn]] void FailClosed(const char* why) {
  std::fprintf(stderr, "\n*** P1 GPU FAIL CLOSED (exit 5) ***\n%s\n", why);
  std::exit(5);
}

inline uint16_t F32ToBf16Bits(float f) {
  uint32_t x;
  std::memcpy(&x, &f, sizeof(x));
  const uint32_t rounding = 0x7fffu + ((x >> 16) & 1u);
  return static_cast<uint16_t>((x + rounding) >> 16);
}
inline float Bf16BitsToF32(uint16_t b) {
  uint32_t x = static_cast<uint32_t>(b) << 16;
  float f;
  std::memcpy(&f, &x, sizeof(f));
  return f;
}

std::vector<float> RandF32(size_t n, uint32_t seed) {
  std::vector<float> v(n);
  uint32_t s = seed;
  for (auto& x : v) {
    s = s * 1664525u + 1013904223u;
    x = (static_cast<float>(s >> 8) / static_cast<float>(1u << 24)) * 4.0f - 2.0f;
  }
  return v;
}

Tensor Contig(void* data, DType dt, Device dev, const std::vector<int64_t>& shape) {
  Tensor t;
  t.data = data;
  t.dtype = dt;
  t.device = dev;
  t.rank = static_cast<int>(shape.size());
  int64_t stride = 1;
  for (int i = t.rank - 1; i >= 0; --i) {
    t.shape[i] = shape[static_cast<size_t>(i)];
    t.stride[i] = stride;
    stride *= shape[static_cast<size_t>(i)];
  }
  return t;
}

struct DeviceBuf {
  Backend& b;
  void* p = nullptr;
  size_t bytes = 0;
  Tensor t;
  DeviceBuf(Backend& b_, Queue& q, DType dt, const std::vector<int64_t>& shape, const void* host)
      : b(b_) {
    int64_t n = 1;
    for (auto s : shape) n *= s;
    bytes = static_cast<size_t>(n) * vt::SizeOf(dt);
    p = b.Alloc(bytes == 0 ? 1 : bytes);
    if (host != nullptr) b.Copy(q, p, host, bytes);
    t = Contig(p, dt, Device{DeviceType::kROCM, 0}, shape);
  }
  ~DeviceBuf() {
    if (p) b.Free(p);
  }
  void Download(Queue& q, void* dst) {
    b.Copy(q, dst, p, bytes);
    b.Synchronize(q);
  }
};

std::vector<float> Oracle(const std::vector<float>& q, const std::vector<float>& kc,
                          const std::vector<float>& vc, const std::vector<int32_t>& block_table) {
  const int64_t qpk = kHq / kHk;
  std::vector<float> out(static_cast<size_t>(kT * kHq * kD), 0.0f);
  for (int64_t local = 0; local < kT; ++local) {
    const int64_t p = local;
    const int64_t jmin = std::max<int64_t>(0, p - kWindowLeft);
    int64_t jmax = std::min(p, p + kWindowRight);
    jmax = std::min(jmax, kT - 1);
    for (int64_t h = 0; h < kHq; ++h) {
      const int64_t g = h / qpk;
      const int64_t qoff = (local * kHq + h) * kD;
      std::vector<float> sc(static_cast<size_t>(jmax - jmin + 1));
      float m = -std::numeric_limits<float>::infinity();
      for (int64_t j = jmin; j <= jmax; ++j) {
        const int64_t blk = block_table[static_cast<size_t>(j / kBlock)];
        const int64_t off = j % kBlock;
        const int64_t kbase = ((blk * kBlock + off) * kHk + g) * kD;
        float dot = 0.0f;
        for (int64_t e = 0; e < kD; ++e)
          dot += q[static_cast<size_t>(qoff + e)] * kc[static_cast<size_t>(kbase + e)];
        dot *= kScale;
        sc[static_cast<size_t>(j - jmin)] = dot;
        if (dot > m) m = dot;
      }
      float denom = 0.0f;
      for (float& s : sc) {
        s = std::exp(s - m);
        denom += s;
      }
      const float inv = 1.0f / denom;
      for (int64_t e = 0; e < kD; ++e) {
        float a = 0.0f;
        for (int64_t j = jmin; j <= jmax; ++j) {
          const int64_t blk = block_table[static_cast<size_t>(j / kBlock)];
          const int64_t off = j % kBlock;
          const int64_t vbase = ((blk * kBlock + off) * kHk + g) * kD;
          a += sc[static_cast<size_t>(j - jmin)] * inv * vc[static_cast<size_t>(vbase + e)];
        }
        out[static_cast<size_t>(qoff + e)] = a;
      }
    }
  }
  return out;
}

}  // namespace

TEST_CASE("P1 product seam vt::PagedAttention d=256 SharedK") {
  if (std::getenv("VT_785_P1_GPU") == nullptr) {
    std::fprintf(stderr,
                 "\n*** P1 GPU not requested (exit 77 SKIP) — not a pass ***\n");
    std::exit(77);
  }
  if (!vt::rocm::DeviceAvailable()) {
    FailClosed("no ROCm device");
  }
  const char* outdir = std::getenv("VT_785_P1_OUT");
  if (outdir == nullptr || outdir[0] == '\0') {
    FailClosed("VT_785_P1_OUT unset");
  }

  const int64_t num_blocks = (kT + kBlock - 1) / kBlock;
  auto qf = RandF32(static_cast<size_t>(kT * kHq * kD), kQSeed);
  auto kf = RandF32(static_cast<size_t>(num_blocks * kBlock * kHk * kD), kKSeed);
  auto vf = RandF32(static_cast<size_t>(num_blocks * kBlock * kHk * kD), kVSeed);
  std::vector<uint16_t> qb(qf.size()), kb(kf.size()), vb(vf.size());
  std::vector<float> qr(qf.size()), kr(kf.size()), vr(vf.size());
  for (size_t i = 0; i < qf.size(); ++i) {
    qb[i] = F32ToBf16Bits(qf[i]);
    qr[i] = Bf16BitsToF32(qb[i]);
  }
  for (size_t i = 0; i < kf.size(); ++i) {
    kb[i] = F32ToBf16Bits(kf[i]);
    kr[i] = Bf16BitsToF32(kb[i]);
    vb[i] = F32ToBf16Bits(vf[i]);
    vr[i] = Bf16BitsToF32(vb[i]);
  }
  std::vector<int32_t> block_table(static_cast<size_t>(num_blocks));
  for (int64_t i = 0; i < num_blocks; ++i) block_table[static_cast<size_t>(i)] = static_cast<int32_t>(i);
  std::vector<int32_t> seq_lens = {static_cast<int32_t>(kT)};
  std::vector<int32_t> qsl = {0, static_cast<int32_t>(kT)};
  const auto ref = Oracle(qr, kr, vr, block_table);

  Backend& rocm = vt::GetBackend(DeviceType::kROCM);
  Queue q = rocm.CreateQueue();
  DeviceBuf dq(rocm, q, DType::kBF16, {kT, kHq, kD}, qb.data());
  DeviceBuf dk(rocm, q, DType::kBF16, {num_blocks, kBlock, kHk, kD}, kb.data());
  DeviceBuf dv(rocm, q, DType::kBF16, {num_blocks, kBlock, kHk, kD}, vb.data());
  DeviceBuf dbt(rocm, q, DType::kI32, {1, num_blocks}, block_table.data());
  DeviceBuf dsl(rocm, q, DType::kI32, {1}, seq_lens.data());
  DeviceBuf dqsl(rocm, q, DType::kI32, {2}, qsl.data());
  DeviceBuf dout(rocm, q, DType::kBF16, {kT, kHq, kD}, nullptr);

  PagedAttentionArgs args{kScale, /*causal=*/true};
  args.window_size = vt::AttentionWindow{kWindowLeft, kWindowRight};
  args.query_start_loc_host = qsl.data();
  args.max_seq_len = static_cast<int32_t>(kT);
  vt::PagedAttention(q, dout.t, dq.t, dk.t, dv.t, dbt.t, dsl.t, dqsl.t, args);

  std::vector<uint16_t> got(qb.size(), 0);
  dout.Download(q, got.data());

  std::vector<float> got_f(got.size());
  size_t nonfinite = 0, violations = 0;
  double max_abs = 0, mean_g = 0, mean_r = 0;
  for (size_t i = 0; i < got.size(); ++i) {
    got_f[i] = Bf16BitsToF32(got[i]);
    if (!std::isfinite(got_f[i]) || !std::isfinite(ref[i])) ++nonfinite;
    mean_g += got_f[i];
    mean_r += ref[i];
  }
  mean_g /= static_cast<double>(got.size());
  mean_r /= static_cast<double>(got.size());
  double num = 0, dg = 0, dr = 0;
  for (size_t i = 0; i < got.size(); ++i) {
    const double err = std::abs(got_f[i] - ref[i]);
    if (err > max_abs) max_abs = err;
    if (err > kAbsFloor + kRel * std::abs(static_cast<double>(ref[i]))) ++violations;
    const double ag = got_f[i] - mean_g;
    const double ar = ref[i] - mean_r;
    num += ag * ar;
    dg += ag * ag;
    dr += ar * ar;
  }
  const double corr = (dg > 0.0 && dr > 0.0) ? (num / std::sqrt(dg * dr)) : 0.0;
  const int oracle_ok = (nonfinite == 0 && violations == 0 && corr >= kMinCorr) ? 1 : 0;

  const std::string out_path = std::string(outdir) + "/out.bf16";
  const std::string met_path = std::string(outdir) + "/metrics.txt";
  {
    std::ofstream o(out_path, std::ios::binary);
    if (!o) FailClosed("cannot write out.bf16");
    o.write(reinterpret_cast<const char*>(got.data()),
            static_cast<std::streamsize>(got.size() * sizeof(uint16_t)));
  }
  {
    std::ofstream m(met_path);
    if (!m) FailClosed("cannot write metrics.txt");
    m << "max_abs=" << max_abs << "\ncorr=" << corr << "\nnonfinite=" << nonfinite
      << "\nviolations=" << violations << "\noracle_ok=" << oracle_ok
      << "\nn=" << got.size() << "\n";
  }
  if (!oracle_ok) FailClosed("oracle bar miss");
}
