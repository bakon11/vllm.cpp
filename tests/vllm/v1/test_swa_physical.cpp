// CPU tests for SWA_PHYSICAL data-plane helpers + fail-closed group_num_blocks.
#include <doctest/doctest.h>

#include <algorithm>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "vllm/model_executor/models/gemma4.h"
#include "vllm/model_executor/models/model_registry.h"
#include "vllm/v1/attention/backend.h"
#include "vllm/v1/core/block_pool.h"
#include "vllm/v1/core/kv_cache_coordinator.h"
#include "vllm/v1/core/kv_cache_utils.h"
#include "vllm/v1/kv_cache_interface.h"
#include "vllm/v1/swa_physical.h"
#include "vt/dtype.h"

using vllm::v1::AttnLayerPhysicalBlocks;
using vllm::v1::BlockHash;
using vllm::v1::FirstSlidingWindowGroupId;
using vllm::v1::FullAttentionSpec;
using vllm::v1::get_kv_cache_coordinator;
using vllm::v1::KVCacheBlock;
using vllm::v1::KVCacheConfig;
using vllm::v1::SelectGemmaLayerAttnMeta;
using vllm::v1::SlidingWindowSpec;
using vllm::v1::ValidateGroupNumBlocks;
using vllm::v1::make_block_hash_with_group_id;
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

TEST_CASE("SelectGemmaLayerAttnMeta: production helper routes slide vs full") {
  vllm::v1::CommonAttentionMetadata full;
  vllm::v1::CommonAttentionMetadata slide;
  full.num_reqs = 1;
  slide.num_reqs = 2;
  full.block_table_tensor = {7};
  slide.block_table_tensor = {9};
  const auto& a = SelectGemmaLayerAttnMeta(true, full, &slide);
  const auto& b = SelectGemmaLayerAttnMeta(false, full, &slide);
  const auto& c = SelectGemmaLayerAttnMeta(false, full, nullptr);
  const auto& d = SelectGemmaLayerAttnMeta(true, full, nullptr);
  CHECK(&a == &full);
  CHECK(&b == &slide);
  CHECK(b.block_table_tensor[0] == 9);
  CHECK(&c == &full);
  CHECK(&d == &full);
}

void MockCache(vllm::v1::BlockPool& pool, const BlockHash& bh, int group_id,
               KVCacheBlock* blk) {
  pool.cached_block_hash_to_block[make_block_hash_with_group_id(bh, group_id)]
                                 [blk->block_id] = blk;
}

TEST_CASE("Hybrid coordinator: find_longest_cache_hit uses owning slide pool") {
  auto cfg = HybridCfg(32, 8);
  auto coord = get_kv_cache_coordinator(
      cfg, /*max_model_len=*/512, /*max_num_batched_tokens=*/512,
      /*use_eagle=*/false, /*enable_caching=*/true,
      /*enable_kv_cache_events=*/false, /*dcp_world_size=*/1,
      /*pcp_world_size=*/1, /*scheduler_block_size=*/kBlock,
      /*hash_block_size=*/kBlock);
  auto& p0 = coord->single_type_managers[0]->block_pool;
  auto& p1 = coord->single_type_managers[1]->block_pool;
  REQUIRE(p0.blocks.size() > 5);
  REQUIRE(p1.blocks.size() > 3);

  SUBCASE("hit blocks for slide group come from slide pool, not full pool") {
    MockCache(p0, "h0", 0, &p0.blocks[4]);
    MockCache(p1, "h0", 1, &p1.blocks[3]);
    std::vector<BlockHash> bh = {"h0"};
    auto [blocks, hit_length] =
        coord->find_longest_cache_hit(bh, /*max_cache_hit_length=*/kBlock);
    REQUIRE(blocks.size() == 2);
    REQUIRE_FALSE(blocks[0].empty());
    REQUIRE_FALSE(blocks[1].empty());
    CHECK(blocks[0][0] == &p0.blocks[4]);
    CHECK(blocks[1][0] == &p1.blocks[3]);
    CHECK(blocks[1][0] != &p0.blocks[4]);
    CHECK(hit_length >= kBlock);
  }

  SUBCASE("slide hash seeded only in full pool is a miss (owning-pool APC)") {
    MockCache(p0, "h0", 1, &p0.blocks[4]);
    std::vector<BlockHash> bh = {"h0"};
    auto [blocks, hit_length] =
        coord->find_longest_cache_hit(bh, /*max_cache_hit_length=*/kBlock);
    CHECK(hit_length == 0);
    REQUIRE(blocks.size() == 2);
    CHECK(blocks[1].empty());
  }
}

namespace {

struct EnvPin {
  std::string key;
  std::string prev;
  bool had = false;
  EnvPin(const char* k, const char* v) : key(k) {
    if (const char* e = std::getenv(k)) {
      had = true;
      prev = e;
    }
    ::setenv(k, v, 1);
  }
  ~EnvPin() {
    if (had) {
      ::setenv(key.c_str(), prev.c_str(), 1);
    } else {
      ::unsetenv(key.c_str());
    }
  }
};

vllm::HfConfig Gemma4HybridHf() {
  vllm::HfConfig cfg;
  cfg.architectures = {"Gemma4ForConditionalGeneration"};
  cfg.num_hidden_layers = 4;
  cfg.num_key_value_heads = 8;
  cfg.head_dim = 256;
  cfg.raw = {
      {"sliding_window", 512},
      {"global_head_dim", 512},
      {"num_global_key_value_heads", 2},
      {"layer_types",
       nlohmann::json::array(
           {"full_attention", "sliding_attention", "sliding_attention",
            "full_attention"})},
  };
  return cfg;
}

}  // namespace

TEST_CASE("registry MakeKVCache: VT_GEMMA4_SWA_PHYSICAL=0 is one Full group") {
  EnvPin pin("VT_GEMMA4_SWA_PHYSICAL", "0");
  vllm::Gemma4Weights weights;
  auto model = vllm::MakeGemma4LoadedModel(std::move(weights));
  const auto kv = vllm::ModelRegistry::MakeKVCache(*model, Gemma4HybridHf(),
                                                   /*block_size=*/16,
                                                   /*num_blocks=*/64);
  CHECK(kv.kv_cache_groups.size() == 1);
  CHECK(kv.group_num_blocks.empty());
  REQUIRE_FALSE(kv.kv_cache_groups.empty());
  CHECK(dynamic_cast<const FullAttentionSpec*>(
            kv.kv_cache_groups[0].kv_cache_spec.get()) != nullptr);
  CHECK(vllm::v1::FirstSlidingWindowGroupId(kv) == -1);
}

TEST_CASE("registry MakeKVCache: SWA_PHYSICAL default hybrid two groups") {
  EnvPin pin("VT_GEMMA4_SWA_PHYSICAL", "1");
  vllm::Gemma4Weights weights;
  auto model = vllm::MakeGemma4LoadedModel(std::move(weights));
  const auto kv = vllm::ModelRegistry::MakeKVCache(*model, Gemma4HybridHf(),
                                                   /*block_size=*/16,
                                                   /*num_blocks=*/64);
  CHECK(kv.kv_cache_groups.size() == 2);
  REQUIRE(kv.group_num_blocks.size() == 2);
  CHECK(kv.group_num_blocks[0] == 64);
  CHECK(dynamic_cast<const SlidingWindowSpec*>(
            kv.kv_cache_groups[1].kv_cache_spec.get()) != nullptr);
  // Slide pool is window/batch sized, not capped by full num_blocks.
  auto expect_slide = std::make_shared<SlidingWindowSpec>(
      /*block_size=*/16, /*num_kv_heads=*/8, /*head_size=*/256, DType::kBF16,
      /*sliding_window=*/512);
  const int admit =
      expect_slide->max_admission_blocks_per_request(8192, 64 * 16);
  CHECK(kv.group_num_blocks[1] == std::max(admit * 1 + 1, 2));
  CHECK(vllm::v1::FirstSlidingWindowGroupId(kv) == 1);
}
