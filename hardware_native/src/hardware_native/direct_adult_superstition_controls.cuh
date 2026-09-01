#ifndef HARDWARE_NATIVE_DIRECT_ADULT_SUPERSTITION_CONTROLS_CUH
#define HARDWARE_NATIVE_DIRECT_ADULT_SUPERSTITION_CONTROLS_CUH

// h.eligibility_superstition_controls (#1519): one experience may ground
// strong immediate binding and a durable episodic distinction under
// consequential circumstances, while GENERALIZATION BREADTH stays a separate
// learned and corrigible axis.
//
// Law anchors (Revision 12 learning ecology §12, §5):
//   * binding strength and generalization breadth are never one scalar: a
//     single consequential conjunction binds strongly immediately, yet the
//     relation stays confined to its witnessed episodic context;
//   * repetition inside one context is not breadth corroboration; breadth
//     grows only when the conjunction is confirmed again in a NEW context;
//   * discriminating counterevidence contracts breadth under an explicit
//     guard (weakening + guarding) while witnessed history persists intact --
//     the first historical episode remains true as history;
//   * no rationality gate refuses association below an evidence count: every
//     actually-participating source binds on its first contact; scoping only
//     limits how far the relation may gate plasticity as if well-evidenced.

#include <cstdint>

#include "hardware_native/direct_adult_q16.cuh"
#include "hardware_native/direct_adult_eligibility_coalitions.cuh"
#include "hardware_native/direct_exact_history.cuh"

namespace substrate::direct_network {

using direct_adult_core::kQ16One;
using direct_adult_core::mul_q16;

inline constexpr std::uint32_t kSuperstitionLedgerCapacity = 256u;
inline constexpr std::uint32_t kSuperstitionContextCapacity = 8u;
inline constexpr std::int32_t kSuperstitionBindGainQ16 = (kQ16One * 5) / 8;
inline constexpr std::int32_t kSuperstitionBreadthGainQ16 = kQ16One / 3;
inline constexpr std::int32_t kSuperstitionContractionQ16 = (kQ16One * 3) / 4;
inline constexpr std::int32_t kSuperstitionBroadThresholdQ16 = kQ16One / 2;
inline constexpr std::uint32_t kSuperstitionMinDistinctContexts = 2u;

enum class SuperstitionDisposition : std::uint32_t {
  episodic_only = 0u,
  corroborated = 1u,
  guarded_contraction = 2u,
};

struct DirectSuperstitionEntry {
  std::uint64_t source;
  std::int32_t binding_q16;
  std::int32_t breadth_q16;
  std::uint32_t witnessed_contexts[kSuperstitionContextCapacity];
  std::uint32_t witnessed_count;
  std::uint32_t confirmed_contexts;
  std::uint32_t contradiction_samples;
  std::uint32_t last_contact_tick;
  std::uint32_t flags;
  SuperstitionDisposition disposition;
};

struct DirectSuperstitionLedger {
  DirectSuperstitionEntry entries[kSuperstitionLedgerCapacity];
  std::uint32_t count;
  std::uint32_t refusals;
  std::uint32_t guard_holds;
  std::uint32_t reserved;
};
inline constexpr std::uint32_t kSuperstitionFlagGuarded = 1u << 0;
__device__ inline std::int32_t superstition_rise_q16(std::int32_t mass,
                                                     std::int32_t gain_q16) {
  const std::int32_t headroom = kQ16One - mass;
  return mass + direct_adult_core::mul_q16(headroom > 0 ? headroom : 0,
                                           gain_q16);
}
__device__ inline DirectSuperstitionEntry* superstition_find(
    DirectSuperstitionLedger* ledger, std::uint64_t source) {
  if (ledger == nullptr) return nullptr;
  for (std::uint32_t i = 0u; i < ledger->count; ++i)
    if (ledger->entries[i].source == source) return &ledger->entries[i];
  return nullptr;
}
__device__ inline DirectSuperstitionEntry* superstition_slot(
    DirectSuperstitionLedger* ledger, std::uint64_t source) {
  DirectSuperstitionEntry* entry = superstition_find(ledger, source);
  if (entry != nullptr) return entry;
  if (ledger->count >= kSuperstitionLedgerCapacity) {
    ++ledger->refusals;
    return nullptr;
  }
  entry = &ledger->entries[ledger->count++];
  *entry = DirectSuperstitionEntry{};
  entry->source = source;
  return entry;
}
__device__ inline bool superstition_context_is_witnessed(
    const DirectSuperstitionEntry& entry, std::uint32_t context) {
  for (std::uint32_t i = 0u; i < entry.witnessed_count; ++i)
    if (entry.witnessed_contexts[i] == context) return true;
  return false;
}

// Strong immediate binding from one consequential conjunction, recorded with
// the context it occurred in. Association is never refused on evidence count.
__device__ inline void superstition_bind(DirectSuperstitionLedger* ledger,
                                         std::uint64_t source,
                                         std::uint32_t context,
                                         std::uint32_t tick) {
  DirectSuperstitionEntry* entry = superstition_slot(ledger, source);
  if (entry == nullptr) return;
  entry->binding_q16 = superstition_rise_q16(entry->binding_q16,
                                             kSuperstitionBindGainQ16);
  if (entry->witnessed_count < kSuperstitionContextCapacity &&
      !superstition_context_is_witnessed(*entry, context))
    entry->witnessed_contexts[entry->witnessed_count++] = context;
  entry->last_contact_tick = tick;
}

// A later confirming conjunction corroborates breadth only when it lands in a
// context the relation has never held in before; repetition inside the
// witnessed context leaves breadth untouched.
__device__ inline void superstition_confirm(DirectSuperstitionLedger* ledger,
                                            std::uint64_t source,
                                            std::uint32_t context,
                                            std::uint32_t tick) {
  DirectSuperstitionEntry* entry = superstition_find(ledger, source);
  if (entry == nullptr) return;
  entry->last_contact_tick = tick;
  if (superstition_context_is_witnessed(*entry, context)) return;
  if (entry->witnessed_count >= kSuperstitionContextCapacity) {
    ++ledger->refusals;
    return;
  }
  entry->witnessed_contexts[entry->witnessed_count++] = context;
  ++entry->confirmed_contexts;
  entry->breadth_q16 = superstition_rise_q16(entry->breadth_q16,
                                             kSuperstitionBreadthGainQ16);
}

// Discriminating counterevidence: breadth contracts multiplicatively under an
// explicit guard; witnessed history and binding persist untouched.
__device__ inline void superstition_contradict(DirectSuperstitionLedger* ledger,
                                               std::uint64_t source,
                                               std::uint32_t tick) {
  DirectSuperstitionEntry* entry = superstition_find(ledger, source);
  if (entry == nullptr) return;
  entry->breadth_q16 = direct_adult_core::mul_q16(entry->breadth_q16,
                                                  kSuperstitionContractionQ16);
  entry->flags |= kSuperstitionFlagGuarded;
  ++entry->contradiction_samples;
  entry->last_contact_tick = tick;
}
__device__ inline SuperstitionDisposition superstition_classify(
    const DirectSuperstitionEntry& entry) {
  if (entry.contradiction_samples != 0u)
    return SuperstitionDisposition::guarded_contraction;
  if (entry.breadth_q16 >= kSuperstitionBroadThresholdQ16 &&
      entry.confirmed_contexts >= kSuperstitionMinDistinctContexts)
    return SuperstitionDisposition::corroborated;
  return SuperstitionDisposition::episodic_only;
}

// Breadth comes only from a real local conjunction, never payload order:
// the next verified world return on this physical channel before its next contact.
__device__ inline std::int32_t superstition_linked_outcome(
    const DirectExactHistoryRecord* records, std::uint32_t count,
    std::uint32_t contact_index, std::uint64_t source, std::uint32_t* tick) {
  for (std::uint32_t i = contact_index + 1u; i < count; ++i) {
    const DirectExactHistoryRecord& r = records[i];
    if (r.kind == DirectExactHistoryKind::sensory_contact && r.subject == source) return 0;
    if (r.kind != DirectExactHistoryKind::world_return || r.subject != source ||
        (r.flags & kDirectHistoryVerifiedObservation) == 0u) continue;
    if (tick) *tick = r.resident_tick;
    return (r.flags & kDirectHistoryPayloadFlags) == 0u ? 1 : -1;
  }
  return 0;
}
__device__ inline void superstition_learn_source(
    DirectSuperstitionLedger* ledger, const DirectExactHistoryRecord* records,
    std::uint32_t count, std::uint64_t source) {
  bool bound = false;
  for (std::uint32_t i = 0u; i < count; ++i) {
    const DirectExactHistoryRecord& r = records[i];
    if (r.kind != DirectExactHistoryKind::sensory_contact || r.subject != source ||
        (r.flags & kDirectHistoryVerifiedObservation) == 0u) continue;
    std::uint32_t consequence_tick = r.resident_tick;
    const std::int32_t outcome = superstition_linked_outcome(records, count, i, source,
                                                               &consequence_tick);
    if (outcome == 0) continue;
    if (!bound) { superstition_bind(ledger, source, r.context, consequence_tick); bound = true; }
    else if (outcome > 0) superstition_confirm(ledger, source, r.context, consequence_tick);
    else superstition_contradict(ledger, source, consequence_tick);
  }
  DirectSuperstitionEntry* entry = superstition_find(ledger, source);
  if (entry != nullptr) entry->disposition = superstition_classify(*entry);
}

// Interplay with the landed coalition gates (#1506): an E+ coalition may
// raise plasticity retention only when its generalization breadth was
// corroborated across distinct contexts. A thin-breadth or contradicted rule
// is HELD under guard -- it cannot gate plasticity as if well-evidenced --
// while the association itself stays intact and action-influencing. Veto and
// conflict handling is inherited unchanged from the coalition semantics.
__device__ inline void superstition_scope_gate(
    DirectSuperstitionLedger* ledger,
    DirectEligibilityCoalitionTable* coalitions, std::int32_t coalition_slot,
    const DirectSuperstitionEntry& super, EligibilityCoalitionGate gate) {
  DirectEligibilityCoalitionEntry& entry = coalitions->entries[coalition_slot];
  switch (gate) {
    case EligibilityCoalitionGate::retain_plasticity:
      if (super.disposition == SuperstitionDisposition::corroborated) {
        entry.plasticity_retention_q16 = coalition_rise_q16(
            entry.plasticity_retention_q16, kCoalitionRiseGainQ16);
      } else {
        ++ledger->guard_holds;
      }
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

}  // namespace substrate::direct_network

#endif
