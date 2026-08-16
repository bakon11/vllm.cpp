// Isolated SWA_PHYSICAL helpers — scheduler pool sizes → per-layer GPU blocks.
// Fail-closed on malformed non-empty group_num_blocks. Empty = legacy shared pool.
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>

#include "vllm/v1/kv_cache_interface.h"
#include "vt/dtype.h"

namespace vllm::v1 {

inline void ValidateGroupNumBlocks(const KVCacheConfig& cfg) {
  const auto& gnb = cfg.group_num_blocks;
  if (gnb.empty()) return;
  VT_CHECK(gnb.size() == cfg.kv_cache_groups.size(),
           "group_num_blocks must be empty or one entry per kv_cache_group");
  VT_CHECK(gnb[0] == cfg.num_blocks,
           "group_num_blocks[0] must equal KVCacheConfig::num_blocks");
  for (int n : gnb) {
    VT_CHECK(n >= 2, "group_num_blocks entries must be >= 2");
  }
}

inline int FirstSlidingWindowGroupId(const KVCacheConfig& cfg) {
  for (int g = 0; g < static_cast<int>(cfg.kv_cache_groups.size()); ++g) {
    const auto& spec = cfg.kv_cache_groups[static_cast<size_t>(g)].kv_cache_spec;
    if (spec && spec->kind() == KVCacheSpecKind::kSlidingWindow) return g;
  }
  return -1;
}

// Physical block count for one attention layer's CacheBuffer / PagedKvCache view.
// SlidingWindowSpec layers use the slide group's pool; everything else uses
// num_blocks. Requires ValidateGroupNumBlocks first when group_num_blocks set.
inline int64_t AttnLayerPhysicalBlocks(const KVCacheConfig& cfg, int layer,
                                       int slide_group_id, int64_t num_blocks) {
  int64_t layer_blocks = num_blocks;
  if (layer >= 0 &&
      static_cast<size_t>(layer) < cfg.per_layer_attn_specs.size()) {
    const auto& sp = cfg.per_layer_attn_specs[static_cast<size_t>(layer)];
    if (sp && dynamic_cast<const SlidingWindowSpec*>(sp.get()) != nullptr) {
      VT_CHECK(slide_group_id >= 0, "SlidingWindow layer with no slide group");
      VT_CHECK(!cfg.group_num_blocks.empty(),
               "SlidingWindow layer requires group_num_blocks");
      VT_CHECK(static_cast<size_t>(slide_group_id) < cfg.group_num_blocks.size(),
               "slide_group_id out of group_num_blocks");
      layer_blocks = cfg.group_num_blocks[static_cast<size_t>(slide_group_id)];
    }
  }
  return std::max<int64_t>(layer_blocks, 2);
}

}  // namespace vllm::v1
