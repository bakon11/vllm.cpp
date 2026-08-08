// ROCm upload of Gemma-4 MoE expert stacks (BF16 fused or FP8→BF16 dequant).
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
  const size_t headroom = (dev == 0) ? (3ull << 30) : (1ull << 30);
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

  size_t total = 0;
  int ok_layers = 0;
  int max_layers = 100000;
  if (const char* ml = std::getenv("VT_GEMMA4_RESIDENT_MAX_LAYERS"))
    max_layers = std::max(0, std::atoi(ml));

  int fill_dev = 0;

  for (size_t li = 0; li < layers.size(); ++li) {
    if (ok_layers >= max_layers) break;
    auto& moe = layers[li];
    if (!moe.enabled || moe.experts.Empty()) continue;
    auto& ex = moe.experts;
    const int64_t E = ex.num_experts;
    const int64_t I = ex.intermediate;
    const int64_t H = ex.hidden;
    const size_t layer_bytes =
        static_cast<size_t>(E * 2 * I * H) * 2 + static_cast<size_t>(E * H * I) * 2;

    bool ok = false;
    if (ex.is_fp8)
      ok = UploadFp8LayerStreaming(ex, fill_dev);
    else
      ok = UploadBf16Layer(ex, fill_dev);

    if (!ok && fill_dev + 1 < num_gpus) {
      ++fill_dev;
      std::fprintf(stderr, "gemma4 moe: GPU%d full after %d layers; continuing on GPU%d\n",
                   fill_dev - 1, ok_layers, fill_dev);
      if (ex.is_fp8)
        ok = UploadFp8LayerStreaming(ex, fill_dev);
      else
        ok = UploadBf16Layer(ex, fill_dev);
    }
    if (!ok) {
      std::fprintf(stderr, "gemma4 moe: resident upload stopped at layer %zu (OOM)\n", li);
      break;
    }
    total += layer_bytes;
    ++ok_layers;
    std::fprintf(stderr, "gemma4 moe: resident layer %zu -> gpu %d (%.2f GiB cumulative)\n", li,
                 ex.dev_id, total / (1024.0 * 1024.0 * 1024.0));
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

}  // namespace vllm
