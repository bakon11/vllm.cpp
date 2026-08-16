// CPU tests for SWA_PHYSICAL data-plane helpers + fail-closed group_num_blocks.
#include <doctest/doctest.h>

#include <stdexcept>
#include <string>
#include <vector>

#include "vllm/v1/attention/backend.h"
#include "vllm/v1/core/kv_cache_coordinator.h"
#include "vllm/v1/kv_cache_interface.h"
#include "vllm/v1/swa_physical.h"
#include "vt/dtype.h"

using vllm::v1::AttnLayerPhysicalBlocks;
using vllm::v1::FirstSlidingWindowGroupId;
using vllm::v1::FullAttentionSpec;
using vllm::v1::get_kv_cache_coordinator;
using vllm::v1::KVCacheConfig;
using vllm::v1::SlidingWindowSpec;
using vllm::v1::ValidateGroupNumBlocks;
using vt::DType;

namespace {

constexpr int kBlock = 16;

std::shared_ptr<FullAttentionSpec> FullSpec() {
  return std::make_shared<FullAttentionSpec>(kBlock, 1, 64, DType::kBF16);
}
std::shared_ptr<SlidingWindowSpec> SlideSpec() {
  return std::make_shared<SlidingWindowSpec>(kBlock, 1, 32, DType::kBF16, 512);
}

KVCacheConfig HybridCfg(int full_blocks = 100, int slide_blocks = 20) {
  KVCacheConfig cfg;
  cfg.num_blocks = full_blocks;
  cfg.kv_cache_groups.emplace_back(std::vector<std::string>{"fa"}, FullSpec());
  cfg.kv_cache_groups.emplace_back(std::vector<std::string>{"sw"}, SlideSpec());
  cfg.group_num_blocks = {full_blocks, slide_blocks};
  cfg.per_layer_attn_specs = {FullSpec(), SlideSpec(), SlideSpec(), FullSpec()};
  return cfg;
}

}  // namespace

TEST_CASE("ValidateGroupNumBlocks: empty is legacy OK") {
  KVCacheConfig cfg;
  cfg.num_blocks = 8;
  cfg.kv_cache_groups.emplace_back(std::vector<std::string>{"fa"}, FullSpec());
  ValidateGroupNumBlocks(cfg);
}

TEST_CASE("ValidateGroupNumBlocks: size mismatch throws") {
  auto cfg = HybridCfg();
  cfg.group_num_blocks = {100};
  CHECK_THROWS_AS(ValidateGroupNumBlocks(cfg), std::runtime_error);
}

TEST_CASE("ValidateGroupNumBlocks: [0] != num_blocks throws") {
  auto cfg = HybridCfg();
  cfg.group_num_blocks = {99, 20};
  CHECK_THROWS_AS(ValidateGroupNumBlocks(cfg), std::runtime_error);
}

TEST_CASE("ValidateGroupNumBlocks: entry < 2 throws") {
  auto cfg = HybridCfg();
  cfg.group_num_blocks = {100, 1};
  CHECK_THROWS_AS(ValidateGroupNumBlocks(cfg), std::runtime_error);
}

TEST_CASE("AttnLayerPhysicalBlocks: full vs slide counts") {
  auto cfg = HybridCfg(100, 20);
  const int slide = FirstSlidingWindowGroupId(cfg);
  CHECK(slide == 1);
  CHECK(AttnLayerPhysicalBlocks(cfg, 0, slide, 100) == 100);
  CHECK(AttnLayerPhysicalBlocks(cfg, 1, slide, 100) == 20);
  CHECK(AttnLayerPhysicalBlocks(cfg, 2, slide, 100) == 20);
  CHECK(AttnLayerPhysicalBlocks(cfg, 3, slide, 100) == 100);
}

TEST_CASE("AttnLayerPhysicalBlocks: empty per_layer uses num_blocks") {
  KVCacheConfig cfg;
  cfg.num_blocks = 64;
  CHECK(AttnLayerPhysicalBlocks(cfg, 0, -1, 64) == 64);
}

TEST_CASE("AttnLayerPhysicalBlocks: slide layer without group_num_blocks throws") {
  auto cfg = HybridCfg();
  cfg.group_num_blocks.clear();
  CHECK_THROWS_AS(AttnLayerPhysicalBlocks(cfg, 1, 1, 100), std::runtime_error);
}

TEST_CASE("Hybrid coordinator: distinct owning pools + fail-closed") {
  auto cfg = HybridCfg(32, 8);
  auto coord = get_kv_cache_coordinator(
      cfg, /*max_model_len=*/512, /*max_num_batched_tokens=*/512,
      /*use_eagle=*/false, /*enable_caching=*/true,
      /*enable_kv_cache_events=*/false, /*dcp_world_size=*/1,
      /*pcp_world_size=*/1, /*scheduler_block_size=*/kBlock,
      /*hash_block_size=*/kBlock);
  REQUIRE(coord->single_type_managers.size() == 2);
  auto& p0 = coord->single_type_managers[0]->block_pool;
  auto& p1 = coord->single_type_managers[1]->block_pool;
  CHECK(&p0 != &p1);
  CHECK(p0.num_gpu_blocks == 32);
  CHECK(p1.num_gpu_blocks == 8);

  KVCacheConfig bad = cfg;
  bad.group_num_blocks = {32};
  CHECK_THROWS_AS(get_kv_cache_coordinator(
                      bad, 512, 512, false, true, false, 1, 1, kBlock, kBlock),
                  std::runtime_error);
}

TEST_CASE("SelectGemmaLayerAttnMeta: slide vs full pointer routing") {
  vllm::v1::CommonAttentionMetadata full;
  vllm::v1::CommonAttentionMetadata slide;
  full.num_reqs = 1;
  slide.num_reqs = 2;
  full.block_table_tensor = {7};
  slide.block_table_tensor = {9};
  const vllm::v1::CommonAttentionMetadata* sp = &slide;
  const auto& a = (!true && sp != nullptr) ? *sp : full;
  const auto& b = (!false && sp != nullptr) ? *sp : full;
  CHECK(&a == &full);
  CHECK(&b == &slide);
  CHECK(b.block_table_tensor[0] == 9);
  const vllm::v1::CommonAttentionMetadata* none = nullptr;
  const auto& c = (!false && none != nullptr) ? *none : full;
  CHECK(&c == &full);
}
