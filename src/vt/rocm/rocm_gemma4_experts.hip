// ROCm upload of Gemma-4 MoE expert stacks (BF16 fused or FP8→BF16 dequant).
// Packs GPU0 first (compute device), then GPU1 — maximizes same-device fast path.
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

bool TryUploadLayer(Gemma4FusedExperts& ex, int dev, const void* gu_src, const void* dn_src,
                    size_t gu_bytes, size_t dn_bytes) {
  Check(hipSetDevice(dev), "hipSetDevice");
  size_t free_b = 0, tot_b = 0;
  if (hipMemGetInfo(&free_b, &tot_b) == hipSuccess) {
    // Keep ~3 GiB headroom for activations/KV/dense on compute GPU.
    const size_t headroom = (dev == 0) ? (3ull << 30) : (1ull << 30);
    if (free_b < gu_bytes + dn_bytes + headroom) return false;
  }
  void* gu = nullptr;
  void* dn = nullptr;
  if (hipMalloc(&gu, gu_bytes) != hipSuccess) return false;
  if (hipMalloc(&dn, dn_bytes) != hipSuccess) {
    (void)hipFree(gu);
    return false;
  }
  if (hipMemcpy(gu, gu_src, gu_bytes, hipMemcpyHostToDevice) != hipSuccess ||
      hipMemcpy(dn, dn_src, dn_bytes, hipMemcpyHostToDevice) != hipSuccess) {
    (void)hipFree(gu);
    (void)hipFree(dn);
    return false;
  }
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

  // Prefer packing compute GPU (0) first so RunGemma4Moe hits same_dev path.
  int fill_dev = 0;

  for (size_t li = 0; li < layers.size(); ++li) {
    if (ok_layers >= max_layers) break;
    auto& moe = layers[li];
    if (!moe.enabled || moe.experts.Empty()) continue;
    auto& ex = moe.experts;
    const int64_t E = ex.num_experts;
    const int64_t I = ex.intermediate;
    const int64_t H = ex.hidden;
    const size_t gu_bytes = static_cast<size_t>(E * 2 * I * H) * 2;
    const size_t dn_bytes = static_cast<size_t>(E * H * I) * 2;

    std::vector<uint16_t> gu_host;
    std::vector<uint16_t> dn_host;
    const void* gu_src = nullptr;
    const void* dn_src = nullptr;
    if (ex.is_fp8) {
      gu_host.resize(static_cast<size_t>(E * 2 * I * H));
      dn_host.resize(static_cast<size_t>(E * H * I));
      for (int64_t e = 0; e < E; ++e) {
        DequantGemma4Fp8ExpertToBf16(ex.fp8[static_cast<size_t>(e)], I, H,
                                     gu_host.data() + e * 2 * I * H,
                                     dn_host.data() + e * H * I);
      }
      gu_src = gu_host.data();
      dn_src = dn_host.data();
    } else {
      gu_src = ex.gate_up.bytes.data();
      dn_src = ex.down.bytes.data();
      VT_CHECK(ex.gate_up.bytes.size() == gu_bytes && ex.down.bytes.size() == dn_bytes,
               "gemma4 resident: bf16 fused size");
    }

    bool ok = TryUploadLayer(ex, fill_dev, gu_src, dn_src, gu_bytes, dn_bytes);
    if (!ok && fill_dev + 1 < num_gpus) {
      // GPU full — spill remaining layers to next GPU
      ++fill_dev;
      std::fprintf(stderr, "gemma4 moe: GPU%d full after %d layers; continuing on GPU%d\n",
                   fill_dev - 1, ok_layers, fill_dev);
      ok = TryUploadLayer(ex, fill_dev, gu_src, dn_src, gu_bytes, dn_bytes);
    }
    if (!ok) {
      std::fprintf(stderr, "gemma4 moe: resident upload stopped at layer %zu (OOM)\n", li);
      break;
    }
    total += gu_bytes + dn_bytes;
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
