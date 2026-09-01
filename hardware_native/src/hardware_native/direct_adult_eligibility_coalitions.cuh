#ifndef HARDWARE_NATIVE_DIRECT_ADULT_ELIGIBILITY_COALITIONS_CUH
#define HARDWARE_NATIVE_DIRECT_ADULT_ELIGIBILITY_COALITIONS_CUH

// d.eligibility_plasticity_coalitions (#1506): positive and negative
// eligibility evidence are two INDEPENDENTLY learned coalition masses over
// authorized actual participants, never opposite signs of one scalar.
//
// Law anchors (Revision 12 learning ecology):
//   * participation identity stays exact -- only sources with an actual
//     sensory_contact record in device exact history may enter a coalition;
//   * E+ and E- accumulate through independent saturating EMAs with
//     independent evidence classes; no function in this header forms
//     e_support - e_veto, which is precisely the forbidden one-scalar
//     collapse;
//   * high support with high veto is conflicted plasticity that RETAINS both
//     hypotheses under a guard; it never silently cancels toward zero;
//   * eligibility is separate from credit and from update choice: masses here
//     gate plasticity retention only, they mint no consequence credit.

#include <cstdint>

#include "hardware_native/direct_adult_q16.cuh"
#include "hardware_native/direct_exact_history.cuh"

namespace substrate::direct_network {

using direct_adult_core::kQ16One;
using direct_adult_core::mul_q16;

inline constexpr std::uint32_t kCoalitionTableCapacity = 256u;
inline constexpr std::int32_t kCoalitionRiseGainQ16 = kQ16One / 3;
inline constexpr std::int32_t kCoalitionDecayQ16 = (kQ16One * 15) / 16;
inline constexpr std::int32_t kCoalitionHighThresholdQ16 = kQ16One / 2;
inline constexpr std::int32_t kCoalitionLowThresholdQ16 = kQ16One / 4;

enum class EligibilityCoalitionGate : std::uint32_t {
  neutral = 0u,
  retain_plasticity = 1u,
  veto_plasticity = 2u,
  guarded_conflict = 3u,
};

struct DirectEligibilityCoalitionEntry {
  std::uint64_t source;
  std::uint32_t support_samples;
  std::uint32_t veto_samples;
  std::int32_t e_support_q16;
  std::int32_t e_veto_q16;
  std::int32_t plasticity_retention_q16;
  std::uint32_t last_support_tick;
  std::uint32_t last_veto_tick;
  std::uint32_t flags;
};

struct DirectEligibilityCoalitionTable {
  DirectEligibilityCoalitionEntry entries[kCoalitionTableCapacity];
  std::uint32_t count;
  std::uint32_t fence_refusals;
  std::uint32_t gate_counts[4];
};

inline constexpr std::uint32_t kCoalitionFlagConflictGuard = 1u << 0;

__device__ inline std::int32_t coalition_rise_q16(std::int32_t mass,
                                                  std::int32_t gain_q16) {
  const std::int32_t headroom = kQ16One - mass;
  return mass + direct_adult_core::mul_q16(headroom > 0 ? headroom : 0,
                                           gain_q16);
}

__device__ inline std::int32_t coalition_decay_q16(std::int32_t mass) {
  return direct_adult_core::mul_q16(mass, kCoalitionDecayQ16);
}

// The participation fence: a source is an authorized actual participant only
// if device exact history holds a real sensory contact carrying it. Host
// labels cannot mint membership; every deposit path calls this first.
__device__ inline bool coalition_source_authorized(
    const DirectExactHistoryRecord* records, std::uint32_t count,
    std::uint64_t source) {
  for (std::uint32_t i = 0u; i < count; ++i) {
    if (records[i].kind == DirectExactHistoryKind::sensory_contact &&
        records[i].subject == source)
      return true;
  }
  return false;
}

__device__ inline std::int32_t coalition_find_slot(
    const DirectEligibilityCoalitionTable* table, std::uint64_t source) {
  if (table == nullptr) return -1;
  for (std::uint32_t i = 0u; i < table->count; ++i)
    if (table->entries[i].source == source) return static_cast<std::int32_t>(i);
  return -1;
}

__device__ inline bool coalition_deposit(
    DirectEligibilityCoalitionTable* table,
    const DirectExactHistoryRecord* records, std::uint32_t count,
    std::uint64_t source, bool support, std::uint32_t tick) {
  if (table == nullptr || !coalition_source_authorized(records, count, source)) {
    if (table != nullptr) atomicAdd(&table->fence_refusals, 1u);
    return false;
  }
  std::int32_t slot = coalition_find_slot(table, source);
  if (slot < 0) {
    if (table->count >= kCoalitionTableCapacity) {
      atomicAdd(&table->fence_refusals, 1u);
      return false;
    }
    slot = static_cast<std::int32_t>(table->count);
    DirectEligibilityCoalitionEntry fresh{};
    fresh.source = source;
    fresh.plasticity_retention_q16 = kCoalitionLowThresholdQ16;
    table->entries[slot] = fresh;
    table->count = static_cast<std::uint32_t>(slot) + 1u;
  }
  DirectEligibilityCoalitionEntry& entry = table->entries[slot];
  if (support) {
    entry.e_support_q16 = coalition_rise_q16(entry.e_support_q16,
                                             kCoalitionRiseGainQ16);
    ++entry.support_samples;
    entry.last_support_tick = tick;
  } else {
    entry.e_veto_q16 =
        coalition_rise_q16(entry.e_veto_q16, kCoalitionRiseGainQ16);
    ++entry.veto_samples;
    entry.last_veto_tick = tick;
  }
  return true;
}

// Conflict keeps BOTH hypotheses above threshold under a guard flag; no arm
// of this classification subtracts one mass from the other.
__device__ inline EligibilityCoalitionGate coalition_classify_gate(
    DirectEligibilityCoalitionTable* table, std::int32_t slot) {
  const DirectEligibilityCoalitionEntry& entry = table->entries[slot];
  const bool support_high = entry.e_support_q16 >= kCoalitionHighThresholdQ16;
  const bool veto_high = entry.e_veto_q16 >= kCoalitionHighThresholdQ16;
  EligibilityCoalitionGate gate = EligibilityCoalitionGate::neutral;
  if (support_high && veto_high) {
    gate = EligibilityCoalitionGate::guarded_conflict;
    table->entries[slot].flags |= kCoalitionFlagConflictGuard;
  } else if (support_high) {
    gate = EligibilityCoalitionGate::retain_plasticity;
  } else if (veto_high) {
    gate = EligibilityCoalitionGate::veto_plasticity;
  }
  atomicAdd(&table->gate_counts[static_cast<std::uint32_t>(gate)], 1u);
  return gate;
}

// Equal starting surfaces diverge purely by coalition membership: E+ retains
// plasticity, E- vetoes it, conflict holds the surface under guard, neutral
// decays. This is a plasticity-retention gate only; it neither settles credit
// nor chooses an update.
__device__ inline void coalition_apply_gate(DirectEligibilityCoalitionTable* table,
                                            std::int32_t slot,
                                            EligibilityCoalitionGate gate) {
  DirectEligibilityCoalitionEntry& entry = table->entries[slot];
  switch (gate) {
    case EligibilityCoalitionGate::retain_plasticity:
      entry.plasticity_retention_q16 = coalition_rise_q16(
          entry.plasticity_retention_q16, kCoalitionRiseGainQ16);
      break;
    case EligibilityCoalitionGate::veto_plasticity:
      entry.plasticity_retention_q16 =
          coalition_decay_q16(entry.plasticity_retention_q16);
      break;
    case EligibilityCoalitionGate::guarded_conflict:
      break;
    case EligibilityCoalitionGate::neutral:
    default:
      entry.plasticity_retention_q16 =
          coalition_decay_q16(entry.plasticity_retention_q16);
      break;
  }
}

// One thread per distinct participating source walks that source's exact
// history in record order, so accumulation is deterministic regardless of
// scheduling. Supportive positioning is a preserved ordering continuation of
// the source's own payload sequence; wrong ordering is repeated false
// continuation and deposits independent veto evidence.
__device__ inline void coalition_learn_source(
    DirectEligibilityCoalitionTable* table,
    const DirectExactHistoryRecord* records, std::uint32_t count,
    std::uint64_t source) {
  std::uint32_t previous_value = 0u;
  bool have_previous = false;
  for (std::uint32_t i = 0u; i < count; ++i) {
    const DirectExactHistoryRecord& record = records[i];
    if (record.kind != DirectExactHistoryKind::sensory_contact ||
        record.subject != source)
      continue;
    const bool ordered = !have_previous || record.value >= previous_value;
    if (have_previous)
      coalition_deposit(table, records, count, source, ordered,
                        record.resident_tick);
    previous_value = record.value;
    have_previous = true;
  }
}

}  // namespace substrate::direct_network

#endif
