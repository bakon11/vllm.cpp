// #839 host-testable dequant-cache lifetime (product HIP uses the same rules).
// Product must not FreeAll while any pin is live, and must not evict a pinned slot.
#pragma once

#include <cstdint>
#include <cstdlib>
#include <mutex>

namespace vt::rocm {

inline int PrefillDequantCacheSlots() { return 1; }

template <typename Alloc>
struct PrefillDequantCacheT {
  struct Slot {
    const void* key = nullptr;
    void* gu = nullptr;
    void* dn = nullptr;
    uint64_t lru = 0;
    int pins = 0;
  };

  int dev = -1;
  int I = 0, H = 0;
  int nslots = 0;
  uint64_t clock = 0;
  uint64_t hits = 0, misses = 0;
  Slot slots[128]{};
  std::mutex mu;
  Alloc alloc{};

  void FreeAll() {
    for (int i = 0; i < nslots; ++i) {
      alloc.Free(slots[i].gu);
      alloc.Free(slots[i].dn);
      slots[i] = Slot{};
    }
    nslots = 0;
    dev = -1;
    I = H = 0;
  }

  int LivePins() const {
    int n = 0;
    for (int i = 0; i < nslots; ++i) n += slots[i].pins;
    return n;
  }

  bool Ensure(int device, int i_dim, int h_dim) {
    const int want = PrefillDequantCacheSlots();
    if (dev == device && I == i_dim && H == h_dim && nslots == want) return true;
    if (LivePins() > 0) return false;
    FreeAll();
    if (device < 0 || i_dim <= 0 || h_dim <= 0 || want <= 0) return false;
    if (!alloc.SetDevice(device)) return false;
    for (int i = 0; i < want; ++i) {
      slots[i].gu = alloc.Malloc(1);
      slots[i].dn = alloc.Malloc(1);
      if (!slots[i].gu || !slots[i].dn) {
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
    ++misses;
    Slot& s = slots[victim];
    alloc.Fill(s.gu, s.dn, key);
    s.key = key;
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
  bool SetDevice(int) { return true; }
  void* Malloc(int) { return reinterpret_cast<void*>(static_cast<intptr_t>(++next)); }
  void Free(void*&) {}
  void Fill(void*, void*, const void*) {}
};

using PrefillDequantCacheHost = PrefillDequantCacheT<HostAlloc>;

enum class PrefillPeerFailAt {
  None,
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
  bool compute_restored = false;
};

inline bool HostRetireThenUnpin(PrefillDequantCacheHost& cache, PrefillPeerSlotHost& slot) {
  if (slot.cache_pin < 0) return true;
  if (slot.ev_e_recorded) {
    slot.ev_e_retired = true;
  } else if (slot.work_enqueued) {
    slot.ev_e_retired = true;  // stream sync fallback
  }
  {
    std::lock_guard<std::mutex> lk(cache.mu);
    cache.UnpinLocked(slot.cache_pin);
  }
  slot.cache_pin = -1;
  slot.cache_dev = -1;
  return true;
}

inline bool HostLaunch(PrefillDequantCacheHost& cache, PrefillPeerSlotHost& slot, int device,
                       const void* key, int M, PrefillPeerFailAt fail) {
  if (slot.pending_M > 0) return false;
  slot.compute_restored = false;
  int pin = -1;
  void* gu = nullptr;
  void* dn = nullptr;
  {
    std::lock_guard<std::mutex> lk(cache.mu);
    if (!cache.Ensure(device, 4, 8)) return false;
    if (!cache.GetLocked(key, &gu, &dn, &pin)) return false;
  }
  slot.cache_pin = pin;
  slot.cache_dev = device;
  slot.work_enqueued = true;
  if (fail == PrefillPeerFailAt::AfterPinEnqueue) {
    HostRetireThenUnpin(cache, slot);
    slot.compute_restored = true;
    return false;
  }
  if (fail == PrefillPeerFailAt::RecordEvE) {
    HostRetireThenUnpin(cache, slot);
    slot.compute_restored = true;
    return false;
  }
  slot.ev_e_recorded = true;
  if (fail == PrefillPeerFailAt::AfterRecord) {
    HostRetireThenUnpin(cache, slot);
    slot.compute_restored = true;
    return false;
  }
  slot.pending_M = M;
  slot.compute_restored = true;
  return true;
}

inline bool HostFinish(PrefillDequantCacheHost& cache, PrefillPeerSlotHost& slot, int M,
                       PrefillPeerFailAt fail) {
  if (slot.pending_M <= 0 || M > slot.pending_M) return false;
  slot.ev_e_retired = true;  // host-wait ev_e
  if (fail == PrefillPeerFailAt::AfterWait) {
    HostRetireThenUnpin(cache, slot);
    slot.pending_M = 0;
    slot.compute_restored = true;
    return false;
  }
  if (fail == PrefillPeerFailAt::AfterCopy) {
    HostRetireThenUnpin(cache, slot);
    slot.pending_M = 0;
    slot.compute_restored = true;
    return false;
  }
  HostRetireThenUnpin(cache, slot);
  slot.pending_M = 0;
  slot.compute_restored = true;
  return true;
}

}  // namespace vt::rocm
