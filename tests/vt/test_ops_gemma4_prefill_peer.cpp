// #839 host lifetime + source invariants for prefill peer Launch/Finish.
#include <fstream>
#include <string>

#include <doctest/doctest.h>

#include "vt/rocm/rocm_gemma4_prefill_dequant_cache.h"

#ifndef VLLM_CPP_SOURCE_DIR
#define VLLM_CPP_SOURCE_DIR "."
#endif

namespace {

std::string ReadHip() {
  const std::string path = std::string(VLLM_CPP_SOURCE_DIR) + "/src/vt/rocm/rocm_gemma4_experts.hip";
  std::ifstream in(path);
  REQUIRE(in.good());
  return std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

}  // namespace

TEST_CASE("prefill peer source has Launch/Finish and PeerSlot cache_pin") {
  const std::string hip = ReadHip();
  CHECK(hip.find("LaunchGemma4Fp8ExpertGeGLUPrefillPeer") != std::string::npos);
  CHECK(hip.find("FinishGemma4Fp8ExpertGeGLUPrefillPeer") != std::string::npos);
  CHECK(hip.find("cache_pin") != std::string::npos);
  CHECK(hip.find("PeerSlot") != std::string::npos);
  const auto prefill = hip.find("bool RunGemma4Fp8ExpertGeGLUPrefillOnExpertDevice");
  const auto pinhost = hip.find("void PinGemma4Fp8ExpertHostCache");
  REQUIRE(prefill != std::string::npos);
  REQUIRE(pinhost != std::string::npos);
  const std::string region = hip.substr(prefill, pinhost - prefill);
  CHECK(region.find("static thread_local Tls tls;") == std::string::npos);
  // Prefill wrapper must call Launch, not be the only peer storage.
  const auto wrap = hip.find("RunGemma4Fp8ExpertGeGLUPrefillOnExpertDevice");
  REQUIRE(wrap != std::string::npos);
  CHECK(hip.find("LaunchGemma4Fp8ExpertGeGLUPrefillPeer", wrap) != std::string::npos);
}

TEST_CASE("prefill peer Ensure rejects live pin instead of FreeAll") {
  vt::rocm::PrefillDequantCacheHost cache;
  {
    std::lock_guard<std::mutex> lk(cache.mu);
    REQUIRE(cache.Ensure(0, 4, 8));
  }
  void* gu = nullptr;
  void* dn = nullptr;
  int pin = -1;
  const char key = 'k';
  {
    std::lock_guard<std::mutex> lk(cache.mu);
    REQUIRE(cache.GetLocked(&key, &gu, &dn, &pin));
    CHECK(cache.LivePins() == 1);
    CHECK_FALSE(cache.Ensure(0, 5, 8));
    CHECK(cache.I == 4);
    CHECK(cache.slots[pin].pins == 1);
    CHECK(cache.slots[pin].gu == gu);
    cache.UnpinLocked(pin);
  }
  {
    std::lock_guard<std::mutex> lk(cache.mu);
    CHECK(cache.Ensure(0, 5, 8));
    CHECK(cache.I == 5);
  }
}

TEST_CASE("prefill peer pinned slot cannot be rewritten by second worker") {
  vt::rocm::PrefillDequantCacheHost cache;
  const char ka = 'a';
  const char kb = 'b';
  void* gu = nullptr;
  void* dn = nullptr;
  int pin_a = -1;
  {
    std::lock_guard<std::mutex> lk(cache.mu);
    REQUIRE(cache.Ensure(0, 4, 8));
    REQUIRE(cache.GetLocked(&ka, &gu, &dn, &pin_a));
    int pin_b = -1;
    CHECK_FALSE(cache.GetLocked(&kb, &gu, &dn, &pin_b));
    CHECK(cache.slots[pin_a].key == &ka);
    CHECK(cache.slots[pin_a].pins == 1);
    cache.UnpinLocked(pin_a);
    REQUIRE(cache.GetLocked(&kb, &gu, &dn, &pin_b));
    CHECK(cache.slots[pin_b].key == &kb);
  }
}

TEST_CASE("prefill peer Launch/Finish pairing and fail-after-enqueue unpins") {
  using vt::rocm::PrefillPeerFailAt;
  vt::rocm::PrefillDequantCacheHost cache;
  vt::rocm::PrefillPeerSlotHost slot;
  const char key = 'k';
  REQUIRE(vt::rocm::HostLaunch(cache, slot, 0, &key, 8, PrefillPeerFailAt::None));
  CHECK(slot.pending_M == 8);
  CHECK(slot.cache_pin >= 0);
  CHECK(cache.LivePins() == 1);
  REQUIRE(vt::rocm::HostFinish(cache, slot, 8, PrefillPeerFailAt::None));
  CHECK(slot.pending_M == 0);
  CHECK(cache.LivePins() == 0);
  CHECK(slot.compute_restored);
  CHECK(slot.ev_e_retired);

  vt::rocm::PrefillPeerSlotHost s2;
  REQUIRE_FALSE(vt::rocm::HostLaunch(cache, s2, 0, &key, 8, PrefillPeerFailAt::AfterPinEnqueue));
  CHECK(cache.LivePins() == 0);
  CHECK(s2.work_enqueued);
  CHECK(s2.compute_restored);

  vt::rocm::PrefillPeerSlotHost s3;
  REQUIRE_FALSE(vt::rocm::HostLaunch(cache, s3, 0, &key, 8, PrefillPeerFailAt::RecordEvE));
  CHECK(cache.LivePins() == 0);

  vt::rocm::PrefillPeerSlotHost s3b;
  REQUIRE_FALSE(vt::rocm::HostLaunch(cache, s3b, 0, &key, 8, PrefillPeerFailAt::AfterRecord));
  CHECK(cache.LivePins() == 0);

  vt::rocm::PrefillPeerSlotHost s4;
  REQUIRE(vt::rocm::HostLaunch(cache, s4, 0, &key, 8, PrefillPeerFailAt::None));
  REQUIRE_FALSE(vt::rocm::HostFinish(cache, s4, 8, PrefillPeerFailAt::AfterWait));
  CHECK(cache.LivePins() == 0);

  vt::rocm::PrefillPeerSlotHost s5;
  REQUIRE(vt::rocm::HostLaunch(cache, s5, 0, &key, 8, PrefillPeerFailAt::None));
  REQUIRE_FALSE(vt::rocm::HostFinish(cache, s5, 8, PrefillPeerFailAt::AfterCopy));
  CHECK(cache.LivePins() == 0);
}

TEST_CASE("prefill peer wrapper uses slot 0 only") {
  const std::string hip = ReadHip();
  CHECK(hip.find("LaunchGemma4Fp8ExpertGeGLUPrefillPeer(compute_q, /*slot=*/0") != std::string::npos);
  CHECK(hip.find("FinishGemma4Fp8ExpertGeGLUPrefillPeer(compute_q, /*slot=*/0") != std::string::npos);
}
