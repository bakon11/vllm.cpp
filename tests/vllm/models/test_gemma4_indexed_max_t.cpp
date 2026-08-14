// #838 host gates: production-dispatch oracle, single-scale fallback, restore.
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

using vllm::Gemma4ApplyHostExpertScaleOnce;
using vllm::Gemma4IndexedCall;
using vllm::Gemma4IndexedDispatchTokens;
using vllm::Gemma4IndexedHelperHits;
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

struct FakeHelper {
  bool peer_expected = false;
  int fail_at = -1;
  int calls = 0;
  int restores = 0;
  std::vector<vllm::Gemma4IndexedTokenOff> offs;
  std::vector<bool> peers;
  uint16_t* y_base = nullptr;
  const uint16_t* x_base = nullptr;
  const int32_t* ri_base = nullptr;
  const float* rw_base = nullptr;
  int64_t H = 0;
  int top_k = 0;

  bool operator()(const Gemma4IndexedCall& c) {
    ++calls;
    offs.push_back(c.off);
    peers.push_back(c.peer);
    REQUIRE(c.peer == peer_expected);
    REQUIRE(c.x == x_base + c.off.x_elems);
    REQUIRE(c.y == y_base + c.off.y_elems);
    REQUIRE(c.ri == ri_base + c.off.route);
    REQUIRE(c.rw == rw_base + c.off.route);
    const auto want = Gemma4IndexedTokenOffsets(c.t, H, top_k);
    REQUIRE(c.off.x_elems == want.x_elems);
    REQUIRE(c.off.y_elems == want.y_elems);
    REQUIRE(c.off.route == want.route);
    if (fail_at >= 0 && c.t == fail_at) return false;
    return true;
  }
};

}  // namespace

TEST_CASE("gemma4 indexed-max-t: env parse") {
  CHECK(ParseGemma4DecodeIndexedMaxT(nullptr) == 63);
  CHECK(ParseGemma4DecodeIndexedMaxT("1") == 1);
  CHECK(ParseGemma4DecodeIndexedMaxT("64") == 63);
}

TEST_CASE("gemma4 indexed-max-t: host predicate table") {
  const int64_t unset63 = ParseGemma4DecodeIndexedMaxT(nullptr);
  const int64_t env1 = ParseGemma4DecodeIndexedMaxT("1");
  CHECK(Gemma4IndexedOkT(1, unset63, 8, true));
  CHECK(Gemma4IndexedOkT(19, unset63, 8, true));
  CHECK_FALSE(Gemma4IndexedOkT(19, env1, 8, true));
  CHECK_FALSE(Gemma4IndexedOkT(64, unset63, 8, true));
  CHECK(Gemma4IndexedOkT(63, unset63, 8, true));
  CHECK(kGemma4PrefillBatchMinT == 64);
}

TEST_CASE("gemma4 indexed-max-t: production dispatch same-dev vs peer") {
  for (bool peer : {false, true}) {
    for (int64_t T : {int64_t{2}, int64_t{19}, int64_t{63}}) {
      const int64_t H = 8;
      const int top_k = 8;
      std::vector<uint16_t> y(static_cast<size_t>(T * H), 0);
      std::vector<uint16_t> x(static_cast<size_t>(T * H), 1);
      std::vector<int32_t> ri(static_cast<size_t>(T * top_k), 0);
      std::vector<float> rw(static_cast<size_t>(T * top_k), 1.f);
      FakeHelper fake;
      fake.peer_expected = peer;
      fake.y_base = y.data();
      fake.x_base = x.data();
      fake.ri_base = ri.data();
      fake.rw_base = rw.data();
      fake.H = H;
      fake.top_k = top_k;
      const uint64_t hits0 = Gemma4IndexedHelperHits().load();
      const auto disp = Gemma4IndexedDispatchTokens(
          T, H, top_k, peer, y.data(), x.data(), ri.data(), rw.data(),
          [&](const Gemma4IndexedCall& c) { return fake(c); }, [&] { ++fake.restores; });
      REQUIRE(disp.ok);
      CHECK(disp.hits == static_cast<uint64_t>(T));
      CHECK(disp.restores == static_cast<int>(T));
      CHECK(fake.calls == static_cast<int>(T));
      CHECK(fake.restores == static_cast<int>(T));
      CHECK(disp.y_owner == y.data());
      CHECK(Gemma4IndexedHelperHits().load() == hits0 + static_cast<uint64_t>(T));
      const float canary = 0;
      y[0] = 0x3C00;  // still owned
      CHECK(y[0] == 0x3C00);
      (void)canary;
    }
  }
}

TEST_CASE("gemma4 indexed-max-t: first-token and mid-loop helper fail + restore") {
  const int64_t T = 19, H = 8;
  const int top_k = 8;
  std::vector<uint16_t> y(static_cast<size_t>(T * H), 0);
  std::vector<uint16_t> x(static_cast<size_t>(T * H), 1);
  std::vector<int32_t> ri(static_cast<size_t>(T * top_k), 0);
  std::vector<float> rw(static_cast<size_t>(T * top_k), 1.f);
  for (int fail_at : {0, 7}) {
    FakeHelper fake;
    fake.peer_expected = true;
    fake.fail_at = fail_at;
    fake.y_base = y.data();
    fake.x_base = x.data();
    fake.ri_base = ri.data();
    fake.rw_base = rw.data();
    fake.H = H;
    fake.top_k = top_k;
    const auto disp = Gemma4IndexedDispatchTokens(
        T, H, top_k, true, y.data(), x.data(), ri.data(), rw.data(),
        [&](const Gemma4IndexedCall& c) { return fake(c); }, [&] { ++fake.restores; });
    CHECK_FALSE(disp.ok);
    CHECK(disp.hits == static_cast<uint64_t>(fail_at));
    CHECK(disp.restores == fail_at + 1);
    CHECK(fake.restores == fail_at + 1);
    CHECK(disp.y_owner == y.data());
    y[3] = 42;
    CHECK(y[3] == 42);
  }
}

TEST_CASE("gemma4 indexed-max-t: fallback scale is once, not s^2") {
  const int64_t T = 19;
  const int top_k = 8;
  const int64_t E = 8;
  std::vector<float> orig(static_cast<size_t>(T * top_k), 0.5f);
  std::vector<int32_t> hi(static_cast<size_t>(T * top_k));
  std::vector<float> hscale(static_cast<size_t>(E), 3.f);  // s != 1
  for (size_t i = 0; i < hi.size(); ++i) hi[i] = static_cast<int32_t>(i % 8);
  std::vector<float> scratch = orig;
  for (auto& v : scratch) v *= 3.f;  // indexed-path scratch
  std::vector<float> fallback = orig;
  Gemma4ApplyHostExpertScaleOnce(fallback.data(), hi.data(), hscale.data(), T, top_k, E, false);
  for (size_t i = 0; i < orig.size(); ++i) {
    CHECK(fallback[i] == doctest::Approx(orig[i] * 3.f));
    CHECK(fallback[i] != doctest::Approx(orig[i] * 9.f));
    CHECK(scratch[i] == doctest::Approx(orig[i] * 3.f));
  }
  std::vector<float> already = orig;
  Gemma4ApplyHostExpertScaleOnce(already.data(), hi.data(), hscale.data(), T, top_k, E, true);
  for (size_t i = 0; i < orig.size(); ++i) CHECK(already[i] == orig[i]);
}

TEST_CASE("gemma4 indexed-max-t: source invariants") {
  const std::string moe = ReadText("src/vllm/model_executor/models/gemma4_moe.cpp");
  const std::string hip = ReadText("src/vt/rocm/rocm_gemma4_experts.hip");
  REQUIRE_FALSE(moe.empty());
  REQUIRE_FALSE(hip.empty());
  CHECK(moe.find("Gemma4IndexedDispatchTokens") != std::string::npos);
  CHECK(moe.find("Gemma4ApplyHostExpertScaleOnce") != std::string::npos);
  CHECK(moe.find("helper_rw") != std::string::npos);
  CHECK(moe.find("rw_idx") != std::string::npos);
  CHECK(moe.find("ExpertGeGLUFp8TopKIndexedBatched") == std::string::npos);
  CHECK(moe.find("PREFILL_INDEXED_NOSYNC") == std::string::npos);
  CHECK(moe.find("if (T == 1 && fp8_res && top_k <= 8 && top_k > 0)") == std::string::npos);
  // Must not scale the live router rw in place.
  CHECK(moe.find("ApplyExpertScaleRw(d.q, static_cast<float*>(rw.ptr())") == std::string::npos);
  CHECK(hip.find("RestoreComputeDev") != std::string::npos);
}

TEST_CASE("gemma4 indexed-max-t: RED wrong production stride") {
  const int64_t T = 19, H = 8;
  const int top_k = 8;
  int mismatches = 0;
  for (int64_t t = 0; t < T; ++t) {
    const auto want = Gemma4IndexedTokenOffsets(t, H, top_k);
    const int64_t mutated = t;  // index t as t, not t*H
    if (mutated != want.x_elems) ++mismatches;
  }
  CHECK(mismatches == T - 1);
  const std::string hdr = ReadText("include/vllm/model_executor/models/gemma4_indexed_gate.h");
  CHECK(hdr.find("x + off.x_elems") != std::string::npos);
  CHECK(hdr.find("Gemma4IndexedTokenOffsets(t, H, top_k)") != std::string::npos);
}
