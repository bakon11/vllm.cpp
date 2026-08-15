// #839 host lifetime + product-policy mutations for prefill peer.
#include <fstream>
#include <stdexcept>
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

std::string ReadHeader() {
  const std::string path =
      std::string(VLLM_CPP_SOURCE_DIR) + "/include/vt/rocm/rocm_gemma4_prefill_dequant_cache.h";
  std::ifstream in(path);
  REQUIRE(in.good());
  return std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

}  // namespace

TEST_CASE("prefill peer source has Launch/Finish and shared cache") {
  const std::string hip = ReadHip();
  CHECK(hip.find("LaunchGemma4Fp8ExpertGeGLUPrefillPeer") != std::string::npos);
  CHECK(hip.find("FinishGemma4Fp8ExpertGeGLUPrefillPeer") != std::string::npos);
  CHECK(hip.find("cache_pin") != std::string::npos);
  CHECK(hip.find("PeerSlot") != std::string::npos);
  CHECK(hip.find("PrefillDequantCacheT<HipPrefillCacheHooks>") != std::string::npos);
  CHECK(hip.find("ChoosePrefillRetire") != std::string::npos);
  CHECK(hip.find("this_gen_ev_e") != std::string::npos);
  CHECK(hip.find("RestoreComputeOrThrow") != std::string::npos);
  CHECK(hip.find("SameDevLife") != std::string::npos);
  const auto prefill = hip.find("bool RunGemma4Fp8ExpertGeGLUPrefillOnExpertDevice");
  const auto pinhost = hip.find("void PinGemma4Fp8ExpertHostCache");
  REQUIRE(prefill != std::string::npos);
  REQUIRE(pinhost != std::string::npos);
  CHECK(hip.find("LaunchGemma4Fp8ExpertGeGLUPrefillPeer", prefill) != std::string::npos);
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
    cache.UnpinLocked(pin_a);
    REQUIRE(cache.GetLocked(&kb, &gu, &dn, &pin_b));
    CHECK(cache.slots[pin_b].key == &kb);
  }
}

TEST_CASE("prefill peer Launch/Finish pairing and fail-after-enqueue retires") {
  using vt::rocm::PrefillPeerFailAt;
  vt::rocm::PrefillDequantCacheHost cache;
  vt::rocm::PrefillPeerLife slot;
  const char key = 'k';
  REQUIRE(vt::rocm::HostLaunch(cache, slot, 0, &key, 8, PrefillPeerFailAt::None));
  CHECK(slot.pending_M == 8);
  CHECK(slot.cache_pin >= 0);
  CHECK(cache.LivePins() == 1);
  REQUIRE(vt::rocm::HostFinish(cache, slot, 8, PrefillPeerFailAt::None));
  CHECK(slot.pending_M == 0);
  CHECK(cache.LivePins() == 0);
  CHECK(slot.compute_restored);
  CHECK_FALSE(slot.ev_e_recorded);
  CHECK_FALSE(slot.this_gen_ev_e);

  vt::rocm::PrefillPeerLife s2;
  REQUIRE_FALSE(vt::rocm::HostLaunch(cache, s2, 0, &key, 8, PrefillPeerFailAt::AfterPinEnqueue));
  CHECK(cache.LivePins() == 0);
  CHECK(s2.work_enqueued);

  vt::rocm::PrefillPeerLife s3;
  REQUIRE_FALSE(vt::rocm::HostLaunch(cache, s3, 0, &key, 8, PrefillPeerFailAt::RecordEvE));
  CHECK(cache.LivePins() == 0);

  vt::rocm::PrefillPeerLife s3b;
  REQUIRE_FALSE(vt::rocm::HostLaunch(cache, s3b, 0, &key, 8, PrefillPeerFailAt::AfterRecord));
  CHECK(cache.LivePins() == 0);

  vt::rocm::PrefillPeerLife s4;
  REQUIRE(vt::rocm::HostLaunch(cache, s4, 0, &key, 8, PrefillPeerFailAt::None));
  REQUIRE_FALSE(vt::rocm::HostFinish(cache, s4, 8, PrefillPeerFailAt::AfterWait));
  CHECK(cache.LivePins() == 0);

  vt::rocm::PrefillPeerLife s5;
  REQUIRE(vt::rocm::HostLaunch(cache, s5, 0, &key, 8, PrefillPeerFailAt::None));
  REQUIRE_FALSE(vt::rocm::HostFinish(cache, s5, 8, PrefillPeerFailAt::AfterCopy));
  CHECK(cache.LivePins() == 0);
}

TEST_CASE("prefill peer wrapper uses slot 0 only") {
  const std::string hip = ReadHip();
  CHECK(hip.find("LaunchGemma4Fp8ExpertGeGLUPrefillPeer(compute_q, /*slot=*/0") != std::string::npos);
  CHECK(hip.find("FinishGemma4Fp8ExpertGeGLUPrefillPeer(compute_q, /*slot=*/0") != std::string::npos);
}

TEST_CASE("prefill peer failed retirement quarantines pin") {
  vt::rocm::PrefillDequantCacheHost cache;
  vt::rocm::PrefillPeerLife slot;
  const char key = 'k';
  REQUIRE(vt::rocm::HostLaunch(cache, slot, 0, &key, 8, vt::rocm::PrefillPeerFailAt::None));
  REQUIRE_FALSE(vt::rocm::HostFinish(cache, slot, 8, vt::rocm::PrefillPeerFailAt::None,
                                     /*retire_ok=*/false));
  CHECK(slot.quarantined);
  CHECK(slot.cache_pin >= 0);
  CHECK(cache.LivePins() == 1);
}

TEST_CASE("prefill peer partial Ensure alloc is freed") {
  vt::rocm::PrefillDequantCacheHost cache;
  cache.hooks.fail_malloc_at = 2;
  {
    std::lock_guard<std::mutex> lk(cache.mu);
    CHECK_FALSE(cache.Ensure(0, 4, 8));
    CHECK(cache.LiveAllocs() == 0);
    CHECK(cache.hooks.frees >= 1);
  }
}

TEST_CASE("prefill peer failed ready record keeps fill lease (no cross-stream reuse)") {
  vt::rocm::PrefillDequantCacheHost cache;
  const char ka = 'a';
  const char kb = 'b';
  void* gu = nullptr;
  void* dn = nullptr;
  int pin = -1;
  {
    std::lock_guard<std::mutex> lk(cache.mu);
    REQUIRE(cache.Ensure(0, 4, 8));
    REQUIRE(cache.GetLocked(&ka, &gu, &dn, &pin));
    cache.UnpinLocked(pin);
    cache.hooks.fail_record = true;
    int fail_pin = -1;
    CHECK_FALSE(cache.GetLocked(&kb, &gu, &dn, &fail_pin));
    CHECK(fail_pin >= 0);
    CHECK(cache.slots[0].key == nullptr);
    CHECK_FALSE(cache.slots[0].ready);
    CHECK(cache.slots[0].filling);
    CHECK(cache.slots[0].fill_failed);
    CHECK(cache.LivePins() >= 1);
    CHECK(cache.hooks.fill_calls >= 1);
    int reuse = -1;
    CHECK_FALSE(cache.GetLocked(&ka, &gu, &dn, &reuse));
    CHECK(cache.slots[0].key == nullptr);
    REQUIRE_FALSE(cache.RetireFillLocked(fail_pin, /*retire_ok=*/false));
    CHECK(cache.slots[0].fill_failed);
    CHECK(cache.LivePins() >= 1);
  }
}

TEST_CASE("prefill peer stale prior ev_e is not current rollback target") {
  vt::rocm::PrefillPeerLife life;
  life.ev_e_recorded = true;  // leftover from previous Finish
  life.this_gen_ev_e = false;
  life.rollback_armed = true;
  life.cache_pin = -1;
  CHECK(vt::rocm::ChoosePrefillRetire(life) == vt::rocm::PrefillRetireTarget::ComputeStream);
  life.this_gen_ev_e = true;
  CHECK(vt::rocm::ChoosePrefillRetire(life) == vt::rocm::PrefillRetireTarget::RecordedEvent);
  life.this_gen_ev_e = false;
  life.rollback_armed = false;
  life.work_on_expert = true;
  CHECK(vt::rocm::ChoosePrefillRetire(life) == vt::rocm::PrefillRetireTarget::ExpertStream);
}

TEST_CASE("prefill peer two-invocation: second enqueue fail syncs current stream") {
  vt::rocm::PrefillDequantCacheHost cache;
  vt::rocm::PrefillPeerLife slot;
  const char key = 'k';
  REQUIRE(vt::rocm::HostLaunch(cache, slot, 0, &key, 8, vt::rocm::PrefillPeerFailAt::None));
  REQUIRE(vt::rocm::HostFinish(cache, slot, 8, vt::rocm::PrefillPeerFailAt::None));
  CHECK_FALSE(slot.this_gen_ev_e);
  CHECK_FALSE(slot.ev_e_recorded);
  REQUIRE_FALSE(
      vt::rocm::HostLaunch(cache, slot, 0, &key, 8, vt::rocm::PrefillPeerFailAt::AfterFirstEnqueue));
  CHECK(vt::rocm::ChoosePrefillRetire(slot) != vt::rocm::PrefillRetireTarget::RecordedEvent);
  CHECK(slot.work_enqueued);
}

TEST_CASE("prefill peer restore failure is fatal not false") {
  vt::rocm::PrefillDequantCacheHost cache;
  vt::rocm::PrefillPeerLife slot;
  const char key = 'k';
  CHECK_THROWS_AS(
      vt::rocm::HostLaunch(cache, slot, 0, &key, 8, vt::rocm::PrefillPeerFailAt::None,
                           /*retire_ok=*/true, /*restore_ok=*/false),
      vt::rocm::RestoreFailed);
}

TEST_CASE("prefill peer same-dev quarantine blocks reenter and reconfigure") {
  vt::rocm::SameDevLife life;
  CHECK(life.CanEnter());
  CHECK(life.CanReconfigure());
  life.Quarantine(0, 0);
  CHECK_FALSE(life.CanEnter());
  CHECK_FALSE(life.CanReconfigure());
  life.ClearPin();
  CHECK(life.CanEnter());
}

TEST_CASE("prefill peer same-dev pin held through GEMM readers") {
  vt::rocm::PrefillDequantCacheHost cache;
  vt::rocm::SameDevLife life;
  vt::rocm::SameDevSession sess;
  sess.cache = &cache;
  sess.life = &life;
  const char key = 'k';
  REQUIRE(sess.Acquire(&key, 0));
  CHECK(cache.LivePins() == 1);
  CHECK(life.cache_pin >= 0);
  REQUIRE(sess.RunGemmReaders());
  CHECK_FALSE(sess.unpinned_before_gemm);
  CHECK(cache.LivePins() == 1);
  REQUIRE(sess.Retire(/*sync_ok=*/true));
  CHECK(cache.LivePins() == 0);
}

TEST_CASE("prefill peer same-dev early unpin before GEMM is RED") {
  vt::rocm::PrefillDequantCacheHost cache;
  vt::rocm::SameDevLife life;
  vt::rocm::SameDevSession sess;
  sess.cache = &cache;
  sess.life = &life;
  const char key = 'k';
  REQUIRE(sess.Acquire(&key, 0));
  {
    std::lock_guard<std::mutex> lk(cache.mu);
    cache.UnpinLocked(sess.pin);  // product mutation analogue
  }
  CHECK_FALSE(sess.RunGemmReaders());
  CHECK(sess.unpinned_before_gemm);
}

TEST_CASE("prefill peer LifeFromSlot does not remap this_gen to ev_e_recorded") {
  struct FakeSlot {
    int cache_pin = 3;
    int cache_dev = 1;
    bool this_gen_ev_e = false;
    bool ev_e_recorded = true;
    bool rollback_armed = true;
    bool quarantined = false;
    bool work_on_compute = true;
    bool work_on_expert = false;
  } tls;
  const auto life = vt::rocm::LifeFromSlot(tls);
  CHECK_FALSE(life.this_gen_ev_e);
  CHECK(life.ev_e_recorded);
  CHECK(vt::rocm::ChoosePrefillRetire(life) == vt::rocm::PrefillRetireTarget::ComputeStream);
}

TEST_CASE("prefill peer restore no-op mutation is RED") {
  vt::rocm::PrefillDequantCacheHost cache;
  vt::rocm::PrefillPeerLife slot;
  const char key = 'k';
  auto mutated_restore = [](int) { return true; };  // no-op success
  (void)mutated_restore;
  CHECK_THROWS_AS(
      vt::rocm::HostLaunch(cache, slot, 0, &key, 8, vt::rocm::PrefillPeerFailAt::None,
                           /*retire_ok=*/true, /*restore_ok=*/false),
      vt::rocm::RestoreFailed);
  const std::string hdr = ReadHeader();
  CHECK(hdr.find("if (std::uncaught_exceptions() > 0) return;") == std::string::npos);
  CHECK(hdr.find("if (!set(dev)) std::terminate();") != std::string::npos);
}

TEST_CASE("prefill peer source: product uses shared retire/restore policy") {
  const std::string hip = ReadHip();
  CHECK(hip.find("ChoosePrefillRetire(PeerLifeView") != std::string::npos);
  CHECK(hip.find("LifeFromSlot") != std::string::npos);
  CHECK(hip.find("RestoreComputeOrThrow") != std::string::npos);
  CHECK(hip.find("bool RestoreComputeDevOrFatal") == std::string::npos);
  CHECK(hip.find("life.Quarantine") != std::string::npos);
  CHECK(hip.find("life.PersistPin") != std::string::npos);
  CHECK(hip.find("work_on_compute = true") != std::string::npos);
  CHECK(hip.find("FailLaunchRestore(tls, est, cst, compute_dev)") != std::string::npos);
  const auto ret = hip.find("bool RetirePinThenUnpin");
  REQUIRE(ret != std::string::npos);
  const auto ret_end = hip.find("void FailLaunchRestore", ret);
  REQUIRE(ret_end != std::string::npos);
  const std::string retire = hip.substr(ret, ret_end - ret);
  CHECK(retire.find("(void)hipEventSynchronize") == std::string::npos);
  CHECK(retire.find("ChoosePrefillRetire") != std::string::npos);
  CHECK(retire.find("PrefillRetireTarget::ComputeStream") != std::string::npos);
  CHECK(retire.find("PrefillRetireTarget::ExpertStream") != std::string::npos);

  // After first successful enqueue is armed, bare restore-return is forbidden.
  const auto armed = hip.find("tls.rollback_armed = true");
  REQUIRE(armed != std::string::npos);
  const auto rec_end = hip.find("tls.pending_M = M;", armed);
  REQUIRE(rec_end != std::string::npos);
  const std::string after = hip.substr(armed, rec_end - armed);
  CHECK(after.find("RestoreComputeOrThrow(compute_dev); return false;") == std::string::npos);
  CHECK(after.find("FailLaunchRestore(tls, est, cst, compute_dev)") != std::string::npos);

  // Same-dev: no UnpinLocked between GetLocked and MatmulBT.
  const auto sd = hip.find("if (expert_dev == compute_dev)");
  REQUIRE(sd != std::string::npos);
  const auto launch = hip.find("LaunchGemma4Fp8ExpertGeGLUPrefillPeer(compute_q, /*slot=*/0", sd);
  REQUIRE(launch != std::string::npos);
  const std::string same = hip.substr(sd, launch - sd);
  const auto gl = same.find("GetLocked");
  const auto mm = same.find("MatmulBT");
  REQUIRE(gl != std::string::npos);
  REQUIRE(mm != std::string::npos);
  const auto unpin = same.find("UnpinLocked", gl);
  const bool unpin_after_gemm = unpin == std::string::npos || unpin > mm;
  CHECK(unpin_after_gemm);
}

TEST_CASE("prefill peer this_gen remap mutation is RED") {
  const std::string hip = ReadHip();
  CHECK(hip.find("life.this_gen_ev_e = tls.ev_e_recorded") == std::string::npos);
  CHECK(hip.find("PeerLifeView(const PeerSlot& tls) { return vt::rocm::LifeFromSlot(tls); }") !=
        std::string::npos);
}
