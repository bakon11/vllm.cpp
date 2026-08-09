// ROCm upload of Gemma-4 MoE expert stacks (BF16 fused or FP8→BF16 dequant).
#include <cstring>
// Packs GPU0 first (compute device), then GPU1. FP8 streams per-expert to
// avoid ~30G host OOM from permanent BF16 caches / full-layer host buffers.
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

#include "vllm/model_executor/models/gemma4.h"
#include "vllm/model_executor/models/gemma4_moe.h"

namespace vllm {
namespace {

void Check(hipError_t e, const char* w) {
  if (e != hipSuccess)
    throw std::runtime_error(std::string("gemma4 resident: ") + w + ": " + hipGetErrorString(e));
}

bool DeviceHasRoom(int dev, size_t need_bytes) {
  Check(hipSetDevice(dev), "hipSetDevice");
  size_t free_b = 0, tot_b = 0;
  if (hipMemGetInfo(&free_b, &tot_b) != hipSuccess) return true;
  // GPU0 is compute: dense weights + KV + activations need real headroom.
  // 12 GiB: prior 10 GiB still OOM'd gelu after 42 GiB dual-GPU resident pack.
  const size_t headroom = (dev == 0) ? (12ull << 30) : (512ull << 20);
  return free_b >= need_bytes + headroom;
}

bool AllocLayerDev(int dev, size_t gu_bytes, size_t dn_bytes, void** gu, void** dn) {
  if (!DeviceHasRoom(dev, gu_bytes + dn_bytes)) return false;
  Check(hipSetDevice(dev), "hipSetDevice");
  *gu = nullptr;
  *dn = nullptr;
  if (hipMalloc(gu, gu_bytes) != hipSuccess) return false;
  if (hipMalloc(dn, dn_bytes) != hipSuccess) {
    (void)hipFree(*gu);
    *gu = nullptr;
    return false;
  }
  return true;
}

bool AllocFour(int dev, size_t a, size_t b, size_t c, size_t d, void** pa, void** pb, void** pc,
               void** pd) {
  if (!DeviceHasRoom(dev, a + b + c + d)) return false;
  Check(hipSetDevice(dev), "hipSetDevice");
  *pa = *pb = *pc = *pd = nullptr;
  if (hipMalloc(pa, a) != hipSuccess) return false;
  if (hipMalloc(pb, b) != hipSuccess) {
    (void)hipFree(*pa);
    *pa = nullptr;
    return false;
  }
  if (hipMalloc(pc, c) != hipSuccess) {
    (void)hipFree(*pa);
    (void)hipFree(*pb);
    *pa = *pb = nullptr;
    return false;
  }
  if (hipMalloc(pd, d) != hipSuccess) {
    (void)hipFree(*pa);
    (void)hipFree(*pb);
    (void)hipFree(*pc);
    *pa = *pb = *pc = nullptr;
    return false;
  }
  return true;
}

// Native FP8 resident: keep u8 weights + BF16 channel scales on device (no BF16 expand).
// ~½ VRAM vs UploadFp8LayerStreaming BF16 path. Per-expert pointers alias layer packs.
bool UploadFp8NativeLayer(Gemma4FusedExperts& ex, int dev) {
  const int64_t E = ex.num_experts;
  const int64_t I = ex.intermediate;
  const int64_t H = ex.hidden;
  if (E <= 0 || I <= 0 || H <= 0 || ex.fp8.size() != static_cast<size_t>(E)) return false;

  const size_t gu_one = static_cast<size_t>(2 * I * H);       // u8
  const size_t dn_one = static_cast<size_t>(H * I);           // u8
  const size_t sgu_one = static_cast<size_t>(2 * I) * 2;      // bf16
  const size_t sdn_one = static_cast<size_t>(H) * 2;          // bf16
  const size_t gu_bytes = gu_one * static_cast<size_t>(E);
  const size_t dn_bytes = dn_one * static_cast<size_t>(E);
  const size_t sgu_bytes = sgu_one * static_cast<size_t>(E);
  const size_t sdn_bytes = sdn_one * static_cast<size_t>(E);

  void *gu = nullptr, *dn = nullptr, *sgu = nullptr, *sdn = nullptr;
  if (!AllocFour(dev, gu_bytes, dn_bytes, sgu_bytes, sdn_bytes, &gu, &dn, &sgu, &sdn))
    return false;

  std::vector<uint8_t> gu_pack(gu_one);
  for (int64_t e = 0; e < E; ++e) {
    auto& fex = ex.fp8[static_cast<size_t>(e)];
    VT_CHECK(fex.gate_w.HasHostBytes() && fex.up_w.HasHostBytes() && fex.down_w.HasHostBytes(),
             "fp8 native resident: missing weights");
    VT_CHECK(fex.gate_s.HasHostBytes() && fex.up_s.HasHostBytes() && fex.down_s.HasHostBytes(),
             "fp8 native resident: missing scales");
    // Pack gate|up FP8 contiguous [2I, H]
    const size_t half = static_cast<size_t>(I * H);
    std::memcpy(gu_pack.data(), fex.gate_w.bytes.data(), half);
    std::memcpy(gu_pack.data() + half, fex.up_w.bytes.data(), half);
    Check(hipMemcpy(static_cast<char*>(gu) + static_cast<size_t>(e) * gu_one, gu_pack.data(),
                    gu_one, hipMemcpyHostToDevice),
          "H2D fp8 gu");
    Check(hipMemcpy(static_cast<char*>(dn) + static_cast<size_t>(e) * dn_one,
                    fex.down_w.bytes.data(), dn_one, hipMemcpyHostToDevice),
          "H2D fp8 dn");
    // scales: gate then up → [2I]
    Check(hipMemcpy(static_cast<char*>(sgu) + static_cast<size_t>(e) * sgu_one,
                    fex.gate_s.bytes.data(), static_cast<size_t>(I) * 2, hipMemcpyHostToDevice),
          "H2D sgu g");
    Check(hipMemcpy(static_cast<char*>(sgu) + static_cast<size_t>(e) * sgu_one +
                        static_cast<size_t>(I) * 2,
                    fex.up_s.bytes.data(), static_cast<size_t>(I) * 2, hipMemcpyHostToDevice),
          "H2D sgu u");
    Check(hipMemcpy(static_cast<char*>(sdn) + static_cast<size_t>(e) * sdn_one,
                    fex.down_s.bytes.data(), sdn_one, hipMemcpyHostToDevice),
          "H2D sdn");

    // Drop host BF16 caches if any
    fex.cached_gu.clear();
    fex.cached_gu.shrink_to_fit();
    fex.cached_dn.clear();
    fex.cached_dn.shrink_to_fit();

    fex.dev_fp8_gu = static_cast<char*>(gu) + static_cast<size_t>(e) * gu_one;
    fex.dev_fp8_dn = static_cast<char*>(dn) + static_cast<size_t>(e) * dn_one;
    fex.dev_s_gu = static_cast<char*>(sgu) + static_cast<size_t>(e) * sgu_one;
    fex.dev_s_dn = static_cast<char*>(sdn) + static_cast<size_t>(e) * sdn_one;
    // Clear BF16 device expert slots (owned by LRU if any)
    fex.dev_gu = nullptr;
    fex.dev_dn = nullptr;
  }

  ex.fp8_gu_base = gu;
  ex.fp8_dn_base = dn;
  ex.fp8_sgu_base = sgu;
  ex.fp8_sdn_base = sdn;
  ex.fp8_native_resident = true;
  ex.gate_up_dev = nullptr;
  ex.down_dev = nullptr;
  ex.dev_id = dev;
  return true;
}

// Legacy: FP8 → BF16 expand on device (2× VRAM). Opt-in VT_GEMMA4_RESIDENT_BF16=1.
bool UploadFp8LayerStreaming(Gemma4FusedExperts& ex, int dev) {
  const int64_t E = ex.num_experts;
  const int64_t I = ex.intermediate;
  const int64_t H = ex.hidden;
  const size_t gu_one = static_cast<size_t>(2 * I * H) * sizeof(uint16_t);
  const size_t dn_one = static_cast<size_t>(H * I) * sizeof(uint16_t);
  const size_t gu_bytes = gu_one * static_cast<size_t>(E);
  const size_t dn_bytes = dn_one * static_cast<size_t>(E);

  void* gu = nullptr;
  void* dn = nullptr;
  if (!AllocLayerDev(dev, gu_bytes, dn_bytes, &gu, &dn)) return false;

  std::vector<uint16_t> gu_tmp(static_cast<size_t>(2 * I * H));
  std::vector<uint16_t> dn_tmp(static_cast<size_t>(H * I));
  for (int64_t e = 0; e < E; ++e) {
    DequantGemma4Fp8ExpertToBf16Ephemeral(ex.fp8[static_cast<size_t>(e)], I, H, gu_tmp.data(),
                                          dn_tmp.data());
    // Drop any accidental permanent cache on this expert (decode may have set it).
    ex.fp8[static_cast<size_t>(e)].cached_gu.clear();
    ex.fp8[static_cast<size_t>(e)].cached_gu.shrink_to_fit();
    ex.fp8[static_cast<size_t>(e)].cached_dn.clear();
    ex.fp8[static_cast<size_t>(e)].cached_dn.shrink_to_fit();

    Check(hipMemcpy(static_cast<char*>(gu) + static_cast<size_t>(e) * gu_one, gu_tmp.data(),
                    gu_one, hipMemcpyHostToDevice),
          "H2D gu e");
    Check(hipMemcpy(static_cast<char*>(dn) + static_cast<size_t>(e) * dn_one, dn_tmp.data(),
                    dn_one, hipMemcpyHostToDevice),
          "H2D dn e");
  }
  ex.gate_up_dev = gu;
  ex.down_dev = dn;
  ex.dev_id = dev;
  return true;
}

bool UploadBf16Layer(Gemma4FusedExperts& ex, int dev) {
  const int64_t E = ex.num_experts;
  const int64_t I = ex.intermediate;
  const int64_t H = ex.hidden;
  const size_t gu_bytes = static_cast<size_t>(E * 2 * I * H) * 2;
  const size_t dn_bytes = static_cast<size_t>(E * H * I) * 2;
  VT_CHECK(ex.gate_up.bytes.size() == gu_bytes && ex.down.bytes.size() == dn_bytes,
           "gemma4 resident: bf16 fused size");
  void* gu = nullptr;
  void* dn = nullptr;
  if (!AllocLayerDev(dev, gu_bytes, dn_bytes, &gu, &dn)) return false;
  Check(hipMemcpy(gu, ex.gate_up.bytes.data(), gu_bytes, hipMemcpyHostToDevice), "H2D gu");
  Check(hipMemcpy(dn, ex.down.bytes.data(), dn_bytes, hipMemcpyHostToDevice), "H2D dn");
  ex.gate_up_dev = gu;
  ex.down_dev = dn;
  ex.dev_id = dev;
  return true;
}

}  // namespace

size_t UploadGemma4ExpertsResident(std::vector<Gemma4MoeLayerWeights>& layers,
                                   int num_gpus) {
  if (num_gpus < 1) num_gpus = 1;
  int available = 0;
  Check(hipGetDeviceCount(&available), "hipGetDeviceCount");
  if (available < 1) return 0;
  if (num_gpus > available) num_gpus = available;

  // Enable peer access so off-device resident experts can D2D into compute GPU.
  for (int a = 0; a < num_gpus; ++a) {
    for (int b = 0; b < num_gpus; ++b) {
      if (a == b) continue;
      int can = 0;
      if (hipDeviceCanAccessPeer(&can, a, b) == hipSuccess && can) {
        Check(hipSetDevice(a), "peer set");
        hipError_t pe = hipDeviceEnablePeerAccess(b, 0);
        if (pe != hipSuccess && pe != hipErrorPeerAccessAlreadyEnabled) {
          std::fprintf(stderr, "gemma4 moe: peer access %d->%d failed: %s\n", a, b,
                       hipGetErrorString(pe));
        }
      }
    }
  }

  size_t total = 0;
  int ok_layers = 0;
  int max_layers = 100000;
  if (const char* ml = std::getenv("VT_GEMMA4_RESIDENT_MAX_LAYERS"))
    max_layers = std::max(0, std::atoi(ml));

  int fill_dev = 0;

  // FP8 resident mode:
  // - Default dual-GPU: BF16 expand (hipBLAS; ~2× VRAM, much faster decode).
  // - VT_GEMMA4_RESIDENT_NATIVE=1 → native FP8 packs (~½ VRAM, GEMV kernel).
  // - Single GPU auto-prefers native to fit; override with RESIDENT_BF16=1.
  static const bool force_bf16 = [] {
    const char* e = std::getenv("VT_GEMMA4_RESIDENT_BF16");
    return e && e[0] == '1';
  }();
  static const bool force_native = [] {
    const char* e = std::getenv("VT_GEMMA4_RESIDENT_NATIVE");
    return e && e[0] == '1';
  }();

  for (size_t li = 0; li < layers.size(); ++li) {
    if (ok_layers >= max_layers) break;
    auto& moe = layers[li];
    if (!moe.enabled || moe.experts.Empty()) continue;
    auto& ex = moe.experts;
    const int64_t E = ex.num_experts;
    const int64_t I = ex.intermediate;
    const int64_t H = ex.hidden;
    const bool want_native =
        ex.is_fp8 && !force_bf16 && (force_native || (num_gpus <= 1 && !force_bf16));
    const size_t layer_bytes =
        want_native
            ? (static_cast<size_t>(E) * (static_cast<size_t>(2 * I * H) + static_cast<size_t>(H * I) +
                                         static_cast<size_t>(2 * I) * 2 + static_cast<size_t>(H) * 2))
            : (static_cast<size_t>(E * 2 * I * H) * 2 + static_cast<size_t>(E * H * I) * 2);

    bool ok = false;
    if (ex.is_fp8) {
      if (want_native)
        ok = UploadFp8NativeLayer(ex, fill_dev);
      else
        ok = UploadFp8LayerStreaming(ex, fill_dev);
    } else {
      ok = UploadBf16Layer(ex, fill_dev);
    }

    if (!ok && fill_dev + 1 < num_gpus) {
      ++fill_dev;
      std::fprintf(stderr, "gemma4 moe: GPU%d full after %d layers; continuing on GPU%d\n",
                   fill_dev - 1, ok_layers, fill_dev);
      if (ex.is_fp8) {
        if (want_native)
          ok = UploadFp8NativeLayer(ex, fill_dev);
        else
          ok = UploadFp8LayerStreaming(ex, fill_dev);
      } else {
        ok = UploadBf16Layer(ex, fill_dev);
      }
    }
    if (!ok) {
      std::fprintf(stderr, "gemma4 moe: resident upload stopped at layer %zu (OOM)\n", li);
      break;
    }
    total += layer_bytes;
    ++ok_layers;
    std::fprintf(stderr,
                 "gemma4 moe: resident layer %zu -> gpu %d %s (%.2f GiB cumulative)\n", li,
                 ex.dev_id, ex.fp8_native_resident ? "fp8-native" : "bf16",
                 total / (1024.0 * 1024.0 * 1024.0));
    std::fflush(stderr);
  }
  Check(hipSetDevice(0), "hipSetDevice0");
  std::fprintf(stderr, "gemma4 moe: RESIDENT DONE gpus=%d layers=%d %.2f GiB\n", num_gpus,
               ok_layers, total / (1024.0 * 1024.0 * 1024.0));
  std::fflush(stderr);
  return total;
}

size_t UploadGemma4ExpertsResidentForWeights(Gemma4Weights& weights, int num_gpus) {
  std::vector<Gemma4MoeLayerWeights> flat;
  std::vector<Gemma4MoeLayerWeights*> ptrs;
  for (auto& layer : weights.layers) {
    if (!layer.moe.enabled) continue;
    ptrs.push_back(&layer.moe);
    flat.push_back(std::move(layer.moe));
  }
  size_t n = UploadGemma4ExpertsResident(flat, num_gpus);
  for (size_t i = 0; i < ptrs.size(); ++i) *ptrs[i] = std::move(flat[i]);
  return n;
}

bool PeerCopyGemma4ExpertSlice(int src_dev, const void* gate_up_base, const void* down_base,
                               int expert_id, int64_t I, int64_t H, int compute_dev,
                               void* gate_up_dst, void* down_dst) {
  if (!gate_up_base || !down_base || !gate_up_dst || !down_dst) return false;
  if (src_dev < 0 || compute_dev < 0) return false;
  const size_t gu_one = static_cast<size_t>(2 * I * H) * sizeof(uint16_t);
  const size_t dn_one = static_cast<size_t>(H * I) * sizeof(uint16_t);
  const char* gu_src =
      static_cast<const char*>(gate_up_base) + static_cast<size_t>(expert_id) * gu_one;
  const char* dn_src =
      static_cast<const char*>(down_base) + static_cast<size_t>(expert_id) * dn_one;
  if (src_dev == compute_dev) {
    Check(hipSetDevice(compute_dev), "peer same set");
    Check(hipMemcpy(gate_up_dst, gu_src, gu_one, hipMemcpyDeviceToDevice), "D2D gu");
    Check(hipMemcpy(down_dst, dn_src, dn_one, hipMemcpyDeviceToDevice), "D2D dn");
    return true;
  }
  hipError_t e1 =
      hipMemcpyPeer(gate_up_dst, compute_dev, const_cast<char*>(gu_src), src_dev, gu_one);
  hipError_t e2 =
      hipMemcpyPeer(down_dst, compute_dev, const_cast<char*>(dn_src), src_dev, dn_one);
  if (e1 != hipSuccess || e2 != hipSuccess) {
    // Fallback: staged through host (still avoids FP8 dequant)
    std::vector<uint16_t> gu_tmp(gu_one / 2), dn_tmp(dn_one / 2);
    Check(hipSetDevice(src_dev), "peer fb src");
    Check(hipMemcpy(gu_tmp.data(), gu_src, gu_one, hipMemcpyDeviceToHost), "D2H gu");
    Check(hipMemcpy(dn_tmp.data(), dn_src, dn_one, hipMemcpyDeviceToHost), "D2H dn");
    Check(hipSetDevice(compute_dev), "peer fb dst");
    Check(hipMemcpy(gate_up_dst, gu_tmp.data(), gu_one, hipMemcpyHostToDevice), "H2D gu");
    Check(hipMemcpy(down_dst, dn_tmp.data(), dn_one, hipMemcpyHostToDevice), "H2D dn");
  }
  Check(hipSetDevice(compute_dev), "peer restore compute");
  return true;
}

bool PeerCopyGemma4Fp8ExpertSlice(int src_dev, const void* fp8_gu, const void* fp8_dn,
                                  const void* s_gu, const void* s_dn, int64_t I, int64_t H,
                                  int compute_dev, void* fp8_gu_dst, void* fp8_dn_dst,
                                  void* s_gu_dst, void* s_dn_dst) {
  if (!fp8_gu || !fp8_dn || !s_gu || !s_dn || !fp8_gu_dst || !fp8_dn_dst || !s_gu_dst ||
      !s_dn_dst)
    return false;
  if (src_dev < 0 || compute_dev < 0) return false;
  const size_t gu_b = static_cast<size_t>(2 * I * H);
  const size_t dn_b = static_cast<size_t>(H * I);
  const size_t sgu_b = static_cast<size_t>(2 * I) * 2;
  const size_t sdn_b = static_cast<size_t>(H) * 2;
  auto copy1 = [&](void* dst, const void* src, size_t n) -> bool {
    if (src_dev == compute_dev) {
      Check(hipSetDevice(compute_dev), "fp8 peer same");
      return hipMemcpy(dst, src, n, hipMemcpyDeviceToDevice) == hipSuccess;
    }
    if (hipMemcpyPeer(dst, compute_dev, const_cast<void*>(src), src_dev, n) == hipSuccess)
      return true;
    std::vector<char> tmp(n);
    Check(hipSetDevice(src_dev), "fp8 peer fb src");
    if (hipMemcpy(tmp.data(), src, n, hipMemcpyDeviceToHost) != hipSuccess) return false;
    Check(hipSetDevice(compute_dev), "fp8 peer fb dst");
    return hipMemcpy(dst, tmp.data(), n, hipMemcpyHostToDevice) == hipSuccess;
  };
  if (!copy1(fp8_gu_dst, fp8_gu, gu_b)) return false;
  if (!copy1(fp8_dn_dst, fp8_dn, dn_b)) return false;
  if (!copy1(s_gu_dst, s_gu, sgu_b)) return false;
  if (!copy1(s_dn_dst, s_dn, sdn_b)) return false;
  Check(hipSetDevice(compute_dev), "fp8 peer restore");
  return true;
}

void PinGemma4Fp8ExpertHostCache(const Gemma4Fp8ExpertMats& ex) {
  if (ex.host_pinned) return;
  if (ex.cached_gu.empty() || ex.cached_dn.empty()) return;
  const size_t gu_b = ex.cached_gu.size() * sizeof(uint16_t);
  const size_t dn_b = ex.cached_dn.size() * sizeof(uint16_t);
  hipError_t e1 =
      hipHostRegister(const_cast<uint16_t*>(ex.cached_gu.data()), gu_b, hipHostRegisterDefault);
  hipError_t e2 =
      hipHostRegister(const_cast<uint16_t*>(ex.cached_dn.data()), dn_b, hipHostRegisterDefault);
  if ((e1 == hipSuccess || e1 == hipErrorHostMemoryAlreadyRegistered) &&
      (e2 == hipSuccess || e2 == hipErrorHostMemoryAlreadyRegistered)) {
    ex.host_pinned = true;
  }
}

void UnpinGemma4Fp8ExpertHostCache(const Gemma4Fp8ExpertMats& ex) {
  if (!ex.host_pinned) return;
  if (!ex.cached_gu.empty()) {
    (void)hipHostUnregister(const_cast<uint16_t*>(ex.cached_gu.data()));
  }
  if (!ex.cached_dn.empty()) {
    (void)hipHostUnregister(const_cast<uint16_t*>(ex.cached_dn.data()));
  }
  ex.host_pinned = false;
}

}  // namespace vllm
