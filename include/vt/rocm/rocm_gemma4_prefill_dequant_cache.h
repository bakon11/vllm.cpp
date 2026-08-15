// #839 shared prefill-peer lifetime. Product HIP instantiates this template;
// host tests use the same functions. Do not unpin without observed retirement.
#pragma once

#include <cstdint>
#include <cstdlib>
#include <mutex>

namespace vt::rocm {

inline int PrefillDequantCacheSlots() { return 1; }
inline constexpr int kPrefillDequantCacheMaxSlots = 128;

enum class PrefillRetireOutcome { None, Unpinned, Quarantined };

template <typename Hooks>
struct PrefillDequantCacheT {
  struct Slot {
    const void* key = nullptr;
    void* gu = nullptr;
    void* dn = nullptr;
    uint64_t lru = 0;
    int pins = 0;
    bool ready = false;
    typename Hooks::Event ready_ev{};
  };

  int dev = -1;
  int I = 0, H = 0;
  int nslots = 0;
  uint64_t clock = 0;
  uint64_t hits = 0, misses = 0;
  Slot slots[kPrefillDequantCacheMaxSlots]{};
  std::mutex mu;
  Hooks hooks{};

  void FreeAll() {
    // Walk the full array so a failed Ensure that never published nslots
    // still releases live gu/dn (partial-init leak).
    for (int i = 0; i < kPrefillDequantCacheMaxSlots; ++i) {
      if (slots[i].gu) {
        hooks.Free(slots[i].gu);
        slots[i].gu = nullptr;
      }
      if (slots[i].dn) {
        hooks.Free(slots[i].dn);
        slots[i].dn = nullptr;
      }
      hooks.DestroyEvent(slots[i].ready_ev);
      slots[i] = Slot{};
    }
    nslots = 0;
    dev = -1;
    I = H = 0;
  }

  int LivePins() const {
    int n = 0;
    for (int i = 0; i < kPrefillDequantCacheMaxSlots; ++i) n += slots[i].pins;
    return n;
  }

  int LiveAllocs() const {
    int n = 0;
    for (int i = 0; i < kPrefillDequantCacheMaxSlots; ++i) {
      if (slots[i].gu) ++n;
      if (slots[i].dn) ++n;
    }
    return n;
  }

  bool Ensure(int device, int i_dim, int h_dim) {
    const int want = PrefillDequantCacheSlots();
    if (dev == device && I == i_dim && H == h_dim && nslots == want) return true;
    if (LivePins() > 0) return false;
    FreeAll();
    if (device < 0 || i_dim <= 0 || h_dim <= 0 || want <= 0) return false;
    if (!hooks.SetDevice(device)) return false;
    const size_t gu_b = static_cast<size_t>(2 * i_dim) * static_cast<size_t>(h_dim) * 2;
    const size_t dn_b = static_cast<size_t>(h_dim) * static_cast<size_t>(i_dim) * 2;
    for (int i = 0; i < want; ++i) {
      slots[i].gu = hooks.Malloc(gu_b);
      if (!slots[i].gu) {
        FreeAll();
        return false;
      }
      slots[i].dn = hooks.Malloc(dn_b);
      if (!slots[i].dn) {
        FreeAll();
        return false;
      }
    }
    dev = device;
    I = i_dim;
    H = h_dim;
    nslots = want;
    clock = 0;
    return true;
  }

  bool GetLocked(const void* key, void** gu_out, void** dn_out, int* pin_out) {
    if (!key || !gu_out || !dn_out || nslots <= 0) return false;
    int hit = -1;
    int victim = -1;
    uint64_t oldest = UINT64_MAX;
    for (int i = 0; i < nslots; ++i) {
      if (slots[i].key == key) {
        hit = i;
        break;
      }
      if (slots[i].pins > 0) continue;
      if (slots[i].key == nullptr) {
        if (victim < 0) victim = i;
      } else if (slots[i].lru < oldest) {
        oldest = slots[i].lru;
        if (victim < 0 || slots[victim].key != nullptr) victim = i;
      }
    }
    if (hit >= 0) {
      if (!slots[hit].ready) return false;
      if (!hooks.WaitReady(slots[hit].ready_ev)) return false;
      ++hits;
      slots[hit].lru = ++clock;
      slots[hit].pins++;
      *gu_out = slots[hit].gu;
      *dn_out = slots[hit].dn;
      if (pin_out) *pin_out = hit;
      return true;
    }
    if (victim < 0) {
      for (int i = 0; i < nslots; ++i) {
        if (slots[i].pins == 0) {
          victim = i;
          break;
        }
      }
      if (victim < 0) return false;
    }
    Slot& s = slots[victim];
    if (!hooks.Fill(s.gu, s.dn)) return false;
    if (!hooks.RecordReady(s.ready_ev)) return false;
    s.ready = true;
    s.key = key;
    ++misses;
    s.lru = ++clock;
    s.pins++;
    *gu_out = s.gu;
    *dn_out = s.dn;
    if (pin_out) *pin_out = victim;
    return true;
  }

  void UnpinLocked(int idx) {
    if (idx < 0 || idx >= nslots) return;
    if (slots[idx].pins > 0) slots[idx].pins--;
  }
};

struct HostAlloc {
  int next = 1;
  int mallocs = 0;
  int frees = 0;
  int fail_malloc_at = -1;
  bool fail_record = false;
  bool fail_wait = false;
  struct Event {
    bool recorded = false;
  };
  bool SetDevice(int) { return true; }
  void* Malloc(size_t) {
    ++mallocs;
    if (fail_malloc_at >= 0 && mallocs == fail_malloc_at) return nullptr;
    return reinterpret_cast<void*>(static_cast<intptr_t>(++next));
  }
  void Free(void* p) {
    if (p) ++frees;
  }
  void DestroyEvent(Event& e) { e.recorded = false; }
  bool Fill(void*, void*) { return true; }
  bool RecordReady(Event& e) {
    if (fail_record) return false;
    e.recorded = true;
    return true;
  }
  bool WaitReady(Event& e) { return e.recorded && !fail_wait; }
};

using PrefillDequantCacheHost = PrefillDequantCacheT<HostAlloc>;

inline PrefillRetireOutcome RetirePinIfObserved(PrefillDequantCacheHost& cache, int& cache_pin,
                                                bool retire_ok) {
  if (cache_pin < 0) return PrefillRetireOutcome::None;
  if (!retire_ok) return PrefillRetireOutcome::Quarantined;
  std::lock_guard<std::mutex> lk(cache.mu);
  cache.UnpinLocked(cache_pin);
  cache_pin = -1;
  return PrefillRetireOutcome::Unpinned;
}

inline bool SlotReusable(bool pending, bool rollback_armed, bool quarantined) {
  return !pending && !rollback_armed && !quarantined;
}

inline bool RestoreComputeChecked(bool set_ok, bool& restored) {
  restored = set_ok;
  return set_ok;
}

enum class PrefillPeerFailAt {
  None,
  AfterFirstEnqueue,
  AfterPinEnqueue,
  RecordEvE,
  AfterRecord,
  AfterWait,
  AfterCopy,
};

struct PrefillPeerSlotHost {
  int cache_pin = -1;
  int cache_dev = -1;
  int pending_M = 0;
  bool ev_e_recorded = false;
  bool ev_e_retired = false;
  bool work_enqueued = false;
  bool rollback_armed = false;
  bool quarantined = false;
  bool compute_restored = false;
};

inline bool HostRetireThenUnpin(PrefillDequantCacheHost& cache, PrefillPeerSlotHost& slot,
                                bool retire_ok) {
  if (slot.cache_pin < 0 && !slot.rollback_armed) return true;
  if (slot.ev_e_recorded || slot.work_enqueued) {
    if (!retire_ok) {
      slot.quarantined = true;
      return false;
    }
    slot.ev_e_retired = true;
  }
  const auto out = RetirePinIfObserved(cache, slot.cache_pin, /*retire_ok=*/true);
  if (out == PrefillRetireOutcome::Unpinned || out == PrefillRetireOutcome::None) {
    slot.cache_dev = -1;
    slot.rollback_armed = false;
    return true;
  }
  slot.quarantined = true;
  return false;
}

inline bool HostLaunch(PrefillDequantCacheHost& cache, PrefillPeerSlotHost& slot, int device,
                       const void* key, int M, PrefillPeerFailAt fail, bool retire_ok = true) {
  if (!SlotReusable(slot.pending_M > 0, slot.rollback_armed, slot.quarantined)) return false;
  slot.compute_restored = false;
  slot.work_enqueued = true;
  slot.rollback_armed = true;
  if (fail == PrefillPeerFailAt::AfterFirstEnqueue) {
    if (!HostRetireThenUnpin(cache, slot, retire_ok)) {
      (void)RestoreComputeChecked(true, slot.compute_restored);
      return false;
    }
    return RestoreComputeChecked(true, slot.compute_restored) && false;
  }
  int pin = -1;
  void* gu = nullptr;
  void* dn = nullptr;
  {
    std::lock_guard<std::mutex> lk(cache.mu);
    if (!cache.Ensure(device, 4, 8) || !cache.GetLocked(key, &gu, &dn, &pin)) {
      if (!HostRetireThenUnpin(cache, slot, retire_ok)) {
        (void)RestoreComputeChecked(true, slot.compute_restored);
        return false;
      }
      return RestoreComputeChecked(true, slot.compute_restored) && false;
    }
  }
  slot.cache_pin = pin;
  slot.cache_dev = device;
  if (fail == PrefillPeerFailAt::AfterPinEnqueue || fail == PrefillPeerFailAt::RecordEvE) {
    if (!HostRetireThenUnpin(cache, slot, retire_ok)) {
      (void)RestoreComputeChecked(true, slot.compute_restored);
      return false;
    }
    return RestoreComputeChecked(true, slot.compute_restored) && false;
  }
  slot.ev_e_recorded = true;
  if (fail == PrefillPeerFailAt::AfterRecord) {
    if (!HostRetireThenUnpin(cache, slot, retire_ok)) {
      (void)RestoreComputeChecked(true, slot.compute_restored);
      return false;
    }
    return RestoreComputeChecked(true, slot.compute_restored) && false;
  }
  slot.pending_M = M;
  slot.rollback_armed = false;
  return RestoreComputeChecked(true, slot.compute_restored);
}

inline bool HostFinish(PrefillDequantCacheHost& cache, PrefillPeerSlotHost& slot, int M,
                       PrefillPeerFailAt fail, bool retire_ok = true) {
  if (slot.pending_M <= 0 || M > slot.pending_M) return false;
  if (fail == PrefillPeerFailAt::AfterWait || fail == PrefillPeerFailAt::AfterCopy) {
    if (!HostRetireThenUnpin(cache, slot, retire_ok)) {
      (void)RestoreComputeChecked(true, slot.compute_restored);
      return false;
    }
    slot.pending_M = 0;
    return RestoreComputeChecked(true, slot.compute_restored) && false;
  }
  if (!HostRetireThenUnpin(cache, slot, retire_ok)) {
    (void)RestoreComputeChecked(true, slot.compute_restored);
    return false;
  }
  slot.pending_M = 0;
  return RestoreComputeChecked(true, slot.compute_restored);
}

}  // namespace vt::rocm
