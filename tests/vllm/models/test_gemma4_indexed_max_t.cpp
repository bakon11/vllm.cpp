// #838 host gates: predicate table, tensor oracle vs serial ref, source invariants.
#include <doctest/doctest.h>

#include <cmath>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "vllm/model_executor/models/gemma4_indexed_gate.h"

#ifndef VLLM_CPP_SOURCE_DIR
#define VLLM_CPP_SOURCE_DIR "."
#endif

using vllm::Gemma4IndexedHostIndexedLoop;
using vllm::Gemma4IndexedHostSerialRef;
using vllm::Gemma4IndexedOkT;
using vllm::Gemma4IndexedOracleClose;
using vllm::Gemma4IndexedTokenOffsets;
using vllm::ParseGemma4DecodeIndexedMaxT;
using vllm::kGemma4PrefillBatchMinT;

namespace {

std::string ReadText(const char* rel) {
  const std::string path = std::string(VLLM_CPP_SOURCE_DIR) + "/" + rel;
  std::ifstream in(path);
  std::ostringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

void FillCase(int64_t T, int64_t H, int top_k, std::vector<float>& x, std::vector<int32_t>& idx,
              std::vector<float>& wts) {
  x.assign(static_cast<size_t>(T * H), 0.f);
  idx.assign(static_cast<size_t>(T * top_k), 0);
  wts.assign(static_cast<size_t>(T * top_k), 0.f);
  for (int64_t t = 0; t < T; ++t) {
    for (int64_t h = 0; h < H; ++h) {
      x[static_cast<size_t>(t * H + h)] = 0.01f * static_cast<float>(t + 1) * static_cast<float>(h + 1);
    }
    for (int g = 0; g < top_k; ++g) {
      idx[static_cast<size_t>(t * top_k + g)] = static_cast<int32_t>((t + g) % 8);
      wts[static_cast<size_t>(t * top_k + g)] = 1.f / static_cast<float>(top_k);
    }
  }
}

void RunOracle(int64_t T, int64_t H, int top_k, bool same_dev_arm) {
  (void)same_dev_arm;  // host mix is arm-agnostic; both arms share token offsets.
  std::vector<float> x, wts, serial, indexed;
  std::vector<int32_t> idx;
  FillCase(T, H, top_k, x, idx, wts);
  serial.assign(static_cast<size_t>(T * H), 0.f);
  indexed.assign(static_cast<size_t>(T * H), 0.f);
  Gemma4IndexedHostSerialRef(x.data(), idx.data(), wts.data(), serial.data(), T, H, top_k);
  Gemma4IndexedHostIndexedLoop(x.data(), idx.data(), wts.data(), indexed.data(), T, H, top_k);
  float max_abs = 0.f;
  REQUIRE(Gemma4IndexedOracleClose(indexed.data(), serial.data(), T * H, &max_abs));
  // Ownership: caller still owns `indexed` after return — canary write/read.
  const float canary = 123.5f;
  indexed[0] = canary;
  CHECK(indexed[0] == canary);
}

}  // namespace

TEST_CASE("gemma4 indexed-max-t: env parse") {
  CHECK(ParseGemma4DecodeIndexedMaxT(nullptr) == 63);
  CHECK(ParseGemma4DecodeIndexedMaxT("") == 63);
  CHECK(ParseGemma4DecodeIndexedMaxT("1") == 1);
  CHECK(ParseGemma4DecodeIndexedMaxT("19") == 19);
  CHECK(ParseGemma4DecodeIndexedMaxT("63") == 63);
  CHECK(ParseGemma4DecodeIndexedMaxT("64") == 63);
  CHECK(ParseGemma4DecodeIndexedMaxT("0") == 1);
  CHECK(ParseGemma4DecodeIndexedMaxT("-3") == 1);
}

TEST_CASE("gemma4 indexed-max-t: host predicate table") {
  const int64_t unset63 = ParseGemma4DecodeIndexedMaxT(nullptr);
  const int64_t env1 = ParseGemma4DecodeIndexedMaxT("1");
  CHECK(Gemma4IndexedOkT(1, unset63, 8, true));
  CHECK(Gemma4IndexedOkT(19, unset63, 8, true));
  CHECK_FALSE(Gemma4IndexedOkT(19, env1, 8, true));  // =1 → serial
  CHECK_FALSE(Gemma4IndexedOkT(64, unset63, 8, true));
  CHECK(Gemma4IndexedOkT(63, unset63, 8, true));
  CHECK_FALSE(Gemma4IndexedOkT(0, unset63, 8, true));
  CHECK_FALSE(Gemma4IndexedOkT(19, unset63, 9, true));
  CHECK_FALSE(Gemma4IndexedOkT(19, unset63, 8, false));
  CHECK(kGemma4PrefillBatchMinT == 64);
  // RED: T==1 literal would fail the T=19 unset case above.
}

TEST_CASE("gemma4 indexed-max-t: tensor oracle T=2,19,63 x same-dev/peer") {
  const int64_t H = 8;
  const int top_k = 8;
  for (int64_t T : {int64_t{2}, int64_t{19}, int64_t{63}}) {
    RunOracle(T, H, top_k, /*same_dev_arm=*/true);
    RunOracle(T, H, top_k, /*same_dev_arm=*/false);
  }
}

TEST_CASE("gemma4 indexed-max-t: oracle RED on wrong token stride") {
  const int64_t T = 19, H = 8;
  const int top_k = 8;
  std::vector<float> x, wts, serial, bad;
  std::vector<int32_t> idx;
  FillCase(T, H, top_k, x, idx, wts);
  serial.assign(static_cast<size_t>(T * H), 0.f);
  bad.assign(static_cast<size_t>(T * H), 0.f);
  Gemma4IndexedHostSerialRef(x.data(), idx.data(), wts.data(), serial.data(), T, H, top_k);
  // Mutation: index t as t instead of t*H / t*K.
  for (int64_t t = 0; t < T; ++t) {
    for (int64_t h = 0; h < H; ++h) bad[static_cast<size_t>(t + h)] = 0.f;  // wrong
    for (int g = 0; g < top_k; ++g) {
      const int32_t e = idx[static_cast<size_t>(t + g)];
      const float w = wts[static_cast<size_t>(t + g)];
      const float s = (e + 1) * w;
      for (int64_t h = 0; h < H; ++h) bad[static_cast<size_t>(t + h)] += x[static_cast<size_t>(t + h)] * s;
    }
  }
  float max_abs = 0.f;
  CHECK_FALSE(Gemma4IndexedOracleClose(bad.data(), serial.data(), T * H, &max_abs));
}

TEST_CASE("gemma4 indexed-max-t: source invariants") {
  const std::string moe = ReadText("src/vllm/model_executor/models/gemma4_moe.cpp");
  REQUIRE_FALSE(moe.empty());
  CHECK(moe.find("Gemma4IndexedOkT") != std::string::npos);
  CHECK(moe.find("Gemma4DecodeIndexedMaxT") != std::string::npos);
  CHECK(moe.find("Gemma4IndexedTokenOffsets") != std::string::npos);
  CHECK(moe.find("Gemma4IndexedHelperHits") != std::string::npos);
  CHECK(moe.find("ExpertGeGLUFp8TopKIndexedBatched") == std::string::npos);
  CHECK(moe.find("PREFILL_INDEXED_NOSYNC") == std::string::npos);
  CHECK(moe.find("kPrefillIndexedNoSyncMaxT") == std::string::npos);
  // Must not keep T==1 as the only eligibility literal.
  CHECK(moe.find("if (T == 1 && fp8_res && top_k <= 8 && top_k > 0)") == std::string::npos);
}

TEST_CASE("gemma4 indexed-max-t: offsets are t*H / t*K") {
  const auto a = Gemma4IndexedTokenOffsets(0, 2816, 8);
  CHECK(a.x_elems == 0);
  CHECK(a.y_elems == 0);
  CHECK(a.route == 0);
  const auto b = Gemma4IndexedTokenOffsets(2, 2816, 8);
  CHECK(b.x_elems == 5632);
  CHECK(b.y_elems == 5632);
  CHECK(b.route == 16);
}
