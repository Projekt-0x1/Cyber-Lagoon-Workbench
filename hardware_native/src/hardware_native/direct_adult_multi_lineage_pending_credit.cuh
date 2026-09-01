#ifndef HARDWARE_NATIVE_DIRECT_ADULT_MULTI_LINEAGE_PENDING_CREDIT_CUH
#define HARDWARE_NATIVE_DIRECT_ADULT_MULTI_LINEAGE_PENDING_CREDIT_CUH

// d.multi_lineage_pending_credit (#1518): pending credit records maintain
// multi-lineage causal provenance across extended temporal horizons.
//
// Law anchors (Revision 12):
//   * a pending record is attribution DECIDED but NOT settled -- it waits on
//     downstream evidence for its own lineage over an extended horizon;
//     predicted credit cannot confirm itself, so settlement requires
//     independent exact-history evidence recorded strictly after opening;
//   * section 8: EVIDENCE, PARTICIPATION and CONSTRUCTION ancestry are three
//     independent lineage graphs carried as distinct identities inside every
//     record; nothing here collapses them and no function keys any decision
//     on a single ancestry component -- lookup and settlement key only on
//     lineage identity;
//   * occurrence-context isolation: one lineage's settlement never mutates
//     another lineage's record; settle writes exactly one slot;
//   * finite resources: the ledger is bounded state -- admission past
//     capacity is refused and counted, aging past horizon expires with a
//     recorded outcome, provenance bytes are preserved. Nothing is silently
//     dropped.

#include <cstdint>

#include "hardware_native/direct_exact_history.cuh"

namespace substrate::direct_network {

inline constexpr std::uint32_t kPendingCreditLedgerCapacity = 64u;

enum class PendingCreditState : std::uint32_t {
  pending = 0u,
  settled = 1u,
  expired = 2u,
};

inline constexpr std::uint32_t kPendingCreditFlagAgedOut = 1u << 0;

struct DirectPendingCreditAncestry {
  std::uint64_t evidence_sequence;
  std::uint64_t participation_ticket;
  std::uint64_t construction_identity;
};

struct DirectPendingCreditRecord {
  std::uint64_t lineage_id;
  DirectPendingCreditAncestry ancestry;
  std::int32_t provisional_mass_q16;
  std::uint32_t opened_tick;
  std::uint32_t horizon_ticks;
  std::uint32_t state;
  std::uint32_t flags;
  std::uint32_t settled_tick;
};

struct DirectPendingCreditLedger {
  DirectPendingCreditRecord records[kPendingCreditLedgerCapacity];
  std::uint32_t count;
  std::uint32_t fence_refusals;
  std::uint32_t capacity_refusals;
  std::uint32_t duplicate_settle_refusals;
  std::uint32_t unevidenced_settle_refusals;
  std::uint32_t settled_count;
  std::uint32_t expired_count;
};

// Participation fence: a lineage may pend only if device exact history holds
// an actual sensory contact carrying it. Host labels cannot mint records.
__device__ inline const DirectExactHistoryRecord* pending_credit_first_contact(
    const DirectExactHistoryRecord* records, std::uint32_t count,
    std::uint64_t lineage_id) {
  for (std::uint32_t i = 0u; i < count; ++i) {
    const DirectExactHistoryRecord& record = records[i];
    if (record.kind == DirectExactHistoryKind::sensory_contact &&
        record.subject == lineage_id)
      return &record;
  }
  return nullptr;
}

// CONSTRUCTION ancestry: which constructor generation produced the resident
// morphology this lineage may later credit. Callers read it on device from
// resident state -- e.g. the birth incarnation of the first active route --
// never from host labels.
__device__ inline bool pending_credit_construction_anchor(
    std::uint64_t construction_identity) {
  return construction_identity != 0u;
}

// Downstream evidence: an independent exact-history record of this same
// lineage arriving through the membrane strictly after the record opened.
__device__ inline const DirectExactHistoryRecord*
pending_credit_downstream_evidence(const DirectExactHistoryRecord* records,
                                   std::uint32_t count,
                                   const DirectPendingCreditRecord& record) {
  for (std::uint32_t i = 0u; i < count; ++i) {
    const DirectExactHistoryRecord& candidate = records[i];
    if (candidate.kind == DirectExactHistoryKind::sensory_contact &&
        candidate.subject == record.lineage_id &&
        candidate.sequence != record.ancestry.evidence_sequence &&
        candidate.resident_tick > record.opened_tick)
      return &candidate;
  }
  return nullptr;
}

__device__ inline std::int32_t pending_credit_find_slot(
    const DirectPendingCreditLedger* ledger, std::uint64_t lineage_id) {
  if (ledger == nullptr) return -1;
  for (std::uint32_t i = 0u; i < ledger->count; ++i)
    if (ledger->records[i].lineage_id == lineage_id)
      return static_cast<std::int32_t>(i);
  return -1;
}

// Open one per-lineage pending record. EVIDENCE and PARTICIPATION ancestry
// and opened_tick derive from device-owned exact history; construction_identity
// arrives read on device from resident morphology. Refusals are counted.
__device__ inline bool pending_credit_open(
    DirectPendingCreditLedger* ledger, const DirectExactHistoryRecord* records,
    std::uint32_t count, std::uint64_t lineage_id,
    std::int32_t provisional_mass_q16, std::uint32_t horizon_ticks,
    std::uint64_t construction_identity) {
  if (ledger == nullptr) return false;
  const DirectExactHistoryRecord* contact =
      pending_credit_first_contact(records, count, lineage_id);
  if (contact == nullptr ||
      !pending_credit_construction_anchor(construction_identity)) {
    atomicAdd(&ledger->fence_refusals, 1u);
    return false;
  }
  if (pending_credit_find_slot(ledger, lineage_id) >= 0 ||
      ledger->count >= kPendingCreditLedgerCapacity) {
    atomicAdd(&ledger->capacity_refusals, 1u);
    return false;
  }
  DirectPendingCreditRecord fresh{};
  fresh.lineage_id = lineage_id;
  fresh.ancestry.evidence_sequence = contact->sequence;
  fresh.ancestry.participation_ticket = contact->identity;
  fresh.ancestry.construction_identity = construction_identity;
  fresh.provisional_mass_q16 = provisional_mass_q16;
  fresh.opened_tick = contact->resident_tick;
  fresh.horizon_ticks = horizon_ticks;
  fresh.state = static_cast<std::uint32_t>(PendingCreditState::pending);
  ledger->records[ledger->count] = fresh;
  ++ledger->count;
  return true;
}

// Occurrence-local settlement: keyed on lineage identity alone, writes
// exactly one record, and only once downstream evidence for that lineage has
// actually arrived in device history after opening.
__device__ inline bool pending_credit_settle(
    DirectPendingCreditLedger* ledger, const DirectExactHistoryRecord* records,
    std::uint32_t count, std::uint64_t lineage_id) {
  if (ledger == nullptr) return false;
  const std::int32_t slot = pending_credit_find_slot(ledger, lineage_id);
  if (slot < 0) return false;
  DirectPendingCreditRecord& record = ledger->records[slot];
  if (record.state != static_cast<std::uint32_t>(PendingCreditState::pending)) {
    atomicAdd(&ledger->duplicate_settle_refusals, 1u);
    return false;
  }
  const DirectExactHistoryRecord* evidence =
      pending_credit_downstream_evidence(records, count, record);
  if (evidence == nullptr) {
    atomicAdd(&ledger->unevidenced_settle_refusals, 1u);
    return false;
  }
  record.state = static_cast<std::uint32_t>(PendingCreditState::settled);
  record.settled_tick = evidence->resident_tick;
  ++ledger->settled_count;
  return true;
}

// Lawful aging: records still pending past their horizon expire with a
// recorded outcome and preserved provenance bytes. Expired records stay in
// the ledger; nothing is erased or silently dropped.
__device__ inline void pending_credit_age(DirectPendingCreditLedger* ledger,
                                          std::uint32_t current_tick) {
  if (ledger == nullptr) return;
  for (std::uint32_t i = 0u; i < ledger->count; ++i) {
    DirectPendingCreditRecord& record = ledger->records[i];
    if (record.state != static_cast<std::uint32_t>(PendingCreditState::pending))
      continue;
    if (current_tick > record.opened_tick + record.horizon_ticks) {
      record.state = static_cast<std::uint32_t>(PendingCreditState::expired);
      record.flags |= kPendingCreditFlagAgedOut;
      ++ledger->expired_count;
    }
  }
}

// Device-owned aging clock: the newest resident tick in exact history, so
// expiry decisions never depend on host-supplied time.
__device__ inline std::uint32_t pending_credit_device_now(
    const DirectExactHistoryRecord* records, std::uint32_t count) {
  std::uint32_t now = 0u;
  for (std::uint32_t i = 0u; i < count; ++i)
    if (records[i].resident_tick > now) now = records[i].resident_tick;
  return now;
}

}  // namespace substrate::direct_network

#endif
