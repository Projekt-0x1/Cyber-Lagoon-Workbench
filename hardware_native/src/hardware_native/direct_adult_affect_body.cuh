#ifndef HARDWARE_NATIVE_DIRECT_ADULT_AFFECT_BODY_CUH
#define HARDWARE_NATIVE_DIRECT_ADULT_AFFECT_BODY_CUH

// f.affect_body_consequence (#1517): interoceptive visceral signals modulating
// prospective valuation and action selection.
//
// Law anchors (Revision 12):
//   * body consequence is physical ingress -- visceral traffic enters through
//     the same membrane as every other modality and lands in device exact
//     history with provenance and chronology; nothing here reads a host label;
//   * vitality, stress and damage are device-owned derived quantities read off
//     real consequence ledgers (one verified world-return record plus settled
//     ticket reward/mismatch state, ancestry-resolved to the visceral sensory
//     root that caused the action). They
//     are saturating evidence masses, never reward scalars;
//   * fields are not a second brain: affect state MODULATES a prospective
//     valuation surface over already-authorized candidates and feeds the
//     #1503 pre-commit gate through its lawful surprise/precision inputs. It
//     cannot mint participation, rewrite settled credit or receipt identity,
//     or force a commitment past a lawful veto. A visceral channel that exact
//     history cannot show as actually innervated acquires no affect state
//     (the innervation fence).

#include <cstdint>
#include <type_traits>

#include "hardware_native/direct_adult_core.cuh"
#include "hardware_native/direct_adult_q16.cuh"
#include "hardware_native/direct_exact_history.cuh"

namespace substrate::direct_network {

inline constexpr std::uint32_t kAffectBodyTableCapacity = 64u;
inline constexpr std::int32_t kAffectRiseGainQ16 = direct_adult_core::kQ16One / 3;
inline constexpr std::int32_t kAffectHighThresholdQ16 =
    direct_adult_core::kQ16One / 2;
inline constexpr std::int32_t kAffectDamageWeightQ16 =
    direct_adult_core::kQ16One / 2;
inline constexpr std::int32_t kAffectStressWeightQ16 =
    direct_adult_core::kQ16One / 4;
inline constexpr std::int32_t kAffectVitalityWeightQ16 =
    direct_adult_core::kQ16One / 2;

inline constexpr std::uint32_t kAffectEntryLastDepositDamage = 1u << 0;

inline constexpr std::uint32_t kAffectRootTraceCapacity =
    1u + direct_adult_core::kMaxActionParticipationLinks;

enum class AffectBodyGate : std::uint32_t {
  neutral = 0u,
  preserve = 1u,
  seek = 2u,
};

struct alignas(8) DirectAffectBodyEntry {
  std::uint32_t channel;
  std::uint32_t damage_samples;
  std::uint32_t vitality_samples;
  std::uint32_t stress_samples;
  std::int32_t damage_q16;
  std::int32_t stress_q16;
  std::int32_t vitality_q16;
  std::uint32_t last_evidence_tick;
  std::uint32_t flags;
  std::uint32_t reserved;
};
static_assert(std::is_trivial_v<DirectAffectBodyEntry> &&
              std::is_standard_layout_v<DirectAffectBodyEntry> &&
              std::has_unique_object_representations_v<DirectAffectBodyEntry>);

// Resumable derivation state for one action-ticket slot. The exact-history
// journal is append-only between cold archives, so a walk that remembers its
// per-parent scan frontiers reaches the same verdict on every call while only
// ever reading newly appended records. A cold archive shrinks the journal and
// is detected by the frontier guard, which restarts the cursor from scratch.
struct alignas(8) DirectAffectDeriveCursor {
  std::uint64_t ticket_id;
  std::uint64_t pending[kAffectRootTraceCapacity];
  std::uint32_t frontier[kAffectRootTraceCapacity];
  std::uint32_t return_scan_index;
  std::uint32_t return_matches;
  std::uint32_t found_return_resident_tick;
  std::uint32_t pending_count;
  std::int32_t resolved_root_channel;
  std::uint32_t flags;
  std::uint32_t reserved;
};

// Bounded derivation-cursor cache. A settled slot whose ancestry or deposit
// verdict is still open resumes through one of these instead of rescanning
// the whole journal every epoch; an evicted or colliding identity simply
// restarts from a fresh full pass, so eviction costs time and never truth.
inline constexpr std::uint32_t kAffectDeriveCursorCacheEntries = 128u;

inline constexpr std::uint32_t kAffectCursorReturnFound = 1u << 0;
inline constexpr std::uint32_t kAffectCursorReturnRefused = 1u << 1;
inline constexpr std::uint32_t kAffectCursorSawRoot = 1u << 2;
inline constexpr std::uint32_t kAffectCursorAncestryDead = 1u << 3;

struct alignas(8) DirectAffectBodyState {
  DirectAffectBodyEntry entries[kAffectBodyTableCapacity];
  // Exactly one deposit per current occupant of each finite action-ticket slot.
  // Slot reuse is lawful: a different ticket identity may deposit later.
  std::uint64_t processed_ticket_by_slot[direct_adult_core::kMaxAsynchronousTickets];
  DirectAffectDeriveCursor cursor_cache[kAffectDeriveCursorCacheEntries];
  std::uint32_t count;
  std::uint32_t fence_refusals;
  std::int32_t preservation_pressure_q16;
  std::int32_t vitality_aggregate_q16;
  AffectBodyGate gate;
  std::uint32_t reserved;
};
static_assert(std::is_trivial_v<DirectAffectBodyState> &&
              std::is_standard_layout_v<DirectAffectBodyState> &&
              std::has_unique_object_representations_v<DirectAffectBodyState>);

// One prospective action candidate bound to its exact device identities; the
// valuation surface exists only over candidates the runtime actually emitted.
struct DirectAffectCandidate {
  std::uint64_t action_ticket_id;
  std::uint64_t upstream_ticket_id;
  std::uint32_t participation_identity;
  std::uint32_t root_channel;
  std::int32_t base_valuation_q16;
  std::uint32_t flags;
};
static_assert(std::is_trivial_v<DirectAffectCandidate> &&
              std::is_standard_layout_v<DirectAffectCandidate> &&
              std::has_unique_object_representations_v<DirectAffectCandidate>);

struct alignas(8) DirectAffectValuationReceipt {
  std::uint64_t action_ticket_id;
  std::uint64_t upstream_ticket_id;
  std::uint32_t participation_identity;
  std::uint32_t root_channel;
  std::int32_t base_valuation_q16;
  std::int32_t bias_q16;
  std::int32_t modulated_valuation_q16;
  AffectBodyGate gate;
};
static_assert(
    std::is_trivial_v<DirectAffectValuationReceipt> &&
    std::is_standard_layout_v<DirectAffectValuationReceipt> &&
    std::has_unique_object_representations_v<DirectAffectValuationReceipt>);

__device__ inline std::int32_t affect_rise_q16(std::int32_t mass,
                                               std::int32_t gain_q16) {
  const std::int32_t headroom = direct_adult_core::kQ16One - mass;
  return mass + direct_adult_core::mul_q16(headroom > 0 ? headroom : 0,
                                           gain_q16);
}

// The innervation fence: a visceral channel carries affect state only if
// device exact history holds a real sensory contact on it. Host assertions
// cannot mint membership.
__device__ inline bool affect_channel_innervated(
    const DirectExactHistoryRecord* records, std::uint32_t count,
    std::uint32_t channel) {
  for (std::uint32_t i = 0u; i < count; ++i)
    if (records[i].kind == DirectExactHistoryKind::sensory_contact &&
        records[i].subject == channel &&
        (records[i].flags & kDirectHistoryVerifiedObservation) != 0u)
      return true;
  return false;
}

__device__ inline std::int32_t affect_find_entry(
    const DirectAffectBodyState* state, std::uint32_t channel) {
  for (std::uint32_t i = 0u; i < state->count; ++i)
    if (state->entries[i].channel == channel) return static_cast<std::int32_t>(i);
  return -1;
}

// Resolve all exact parent links for a settled action. Multi-participant actions
// intentionally may have no single diagnostic `upstream_ticket_id`; their
// sparse-credit/Recipe witness records still carry the participant identities.
// Affect is admitted only when every discovered sensory boundary agrees on one
// root channel. Ambiguous or over-capacity ancestry fails closed.
__device__ inline bool affect_root_contact_channel(
    const DirectExactHistoryRecord* records, std::uint32_t count,
    std::uint64_t action_ticket_id, std::uint32_t* root_channel) {
  if (records == nullptr || root_channel == nullptr || action_ticket_id == 0u)
    return false;
  std::uint64_t pending[kAffectRootTraceCapacity]{};
  std::uint32_t pending_count = 1u, cursor = 0u;
  pending[0] = action_ticket_id;
  bool saw_root = false;
  std::uint32_t resolved_root = 0u;
  while (cursor < pending_count) {
    const std::uint64_t sought = pending[cursor++];
    for (std::uint32_t i = 0u; i < count; ++i) {
      const DirectExactHistoryRecord& record = records[i];
      if (record.identity != sought || record.kind == DirectExactHistoryKind::empty)
        continue;
      if (record.kind == DirectExactHistoryKind::sensory_contact) {
        if ((record.flags & kDirectHistoryVerifiedObservation) == 0u)
          return false;
        if (!saw_root) {
          resolved_root = record.subject;
          saw_root = true;
        } else if (resolved_root != record.subject) {
          return false;
        }
        continue;
      }
      const std::uint64_t parent = record.parent_identity;
      if (parent == 0u || parent == 0xffffffffffffffffULL) continue;
      bool known = false;
      for (std::uint32_t j = 0u; j < pending_count; ++j)
        known |= pending[j] == parent;
      if (known) continue;
      if (pending_count == kAffectRootTraceCapacity) return false;
      pending[pending_count++] = parent;
    }
  }
  if (!saw_root) return false;
  *root_channel = resolved_root;
  return true;
}

// A settled bit is not itself consequence authority. Accept exactly one
// matching verified world-return record and bind its immutable fields back to
// the same emitted action ticket. Prediction, labels, or hand-written ticket
// bytes therefore cannot manufacture body consequence.
__device__ inline const DirectExactHistoryRecord* affect_verified_world_return(
    const direct_adult_core::AsynchronousTicket& ticket,
    const DirectExactHistoryRecord* records, std::uint32_t count) {
  const DirectExactHistoryRecord* found = nullptr;
  std::uint32_t matches = 0u;
  for (std::uint32_t i = 0u; i < count; ++i) {
    const DirectExactHistoryRecord& record = records[i];
    if (record.kind != DirectExactHistoryKind::world_return ||
        record.identity != ticket.ticket_id)
      continue;
    ++matches;
    const bool exact =
        record.parent_identity == ticket.upstream_ticket_id &&
        record.source == ticket.motor_node && record.subject == ticket.motor_channel &&
        record.context == ticket.context_signature &&
        record.event_tick == record.resident_tick &&
        record.resident_tick >= ticket.emission_tick &&
        (record.flags & kDirectHistoryVerifiedObservation) != 0u &&
        (record.flags & kDirectHistoryPayloadFlags) ==
            (ticket.mismatch_bits & kDirectHistoryPayloadFlags) &&
        record.resource_delta == ticket.settled_reward_q16;
    if (!exact) return nullptr;
    found = &record;
  }
  return matches == 1u ? found : nullptr;
}

// Deposit-slot resolution with the innervation fence: an unauthorized or
// overflowing channel is refused outright, never silently admitted.
__device__ inline DirectAffectBodyEntry* affect_deposit_slot(
    DirectAffectBodyState* state, const DirectExactHistoryRecord* records,
    std::uint32_t count, std::uint32_t channel) {
  std::int32_t slot = affect_find_entry(state, channel);
  if (slot >= 0) return &state->entries[slot];
  if (!affect_channel_innervated(records, count, channel)) {
    ++state->fence_refusals;
    return nullptr;
  }
  if (state->count >= kAffectBodyTableCapacity) {
    ++state->fence_refusals;
    return nullptr;
  }
  DirectAffectBodyEntry fresh{};
  fresh.channel = channel;
  state->entries[state->count] = fresh;
  return &state->entries[state->count++];
}

// Derive every visceral lane's affect masses from settled consequence ledgers
// only. The action must have one exact verified world-return record and its
// causal ancestry must resolve to an innervated sensory channel in the declared
// visceral range. One thread walks ticket slots deterministically. Each slot
// deposits at most once per ticket identity, while later slot reuse by a new
// ticket remains legal. Repeated damage on one lane without an intervening
// vital closure raises stress; the lane-local sequence persists across calls.
__device__ inline void affect_derive_from_ledgers(
    DirectAffectBodyState* state,
    const direct_adult_core::AsynchronousTicket* tickets,
    std::uint32_t ticket_count, const DirectExactHistoryRecord* records,
    std::uint32_t record_count, std::uint32_t visceral_lo,
    std::uint32_t visceral_hi) {
  if (state == nullptr || tickets == nullptr || records == nullptr) return;
  const std::uint32_t bounded_ticket_count =
      ticket_count < direct_adult_core::kMaxAsynchronousTickets
          ? ticket_count
          : direct_adult_core::kMaxAsynchronousTickets;
  for (std::uint32_t slot = 0u; slot < bounded_ticket_count; ++slot) {
    const direct_adult_core::AsynchronousTicket& ticket = tickets[slot];
    if (ticket.ticket_id == 0u || ticket.settled == 0u ||
        state->processed_ticket_by_slot[slot] == ticket.ticket_id)
      continue;
    DirectAffectDeriveCursor& cur = state->cursor_cache[
        static_cast<std::uint32_t>(((ticket.ticket_id *
            0x9E3779B97F4A7C15ull) >> 57) &
            (kAffectDeriveCursorCacheEntries - 1u))];
    if (cur.ticket_id != ticket.ticket_id ||
        cur.return_scan_index > record_count) {
      cur = DirectAffectDeriveCursor{};
      cur.ticket_id = ticket.ticket_id;
      cur.pending[0] = ticket.ticket_id;
      cur.pending_count = 1u;
      cur.resolved_root_channel = -1;
    }
    // Settlement and its journal verdict commit in one consequence
    // transaction before this derive runs, so the world-return verdict is a
    // function of the journal prefix and never reopens: scan each record once.
    if ((cur.flags & (kAffectCursorReturnFound | kAffectCursorReturnRefused)) ==
        0u) {
      bool refused = false;
      for (std::uint32_t i = cur.return_scan_index; i < record_count; ++i) {
        const DirectExactHistoryRecord& record = records[i];
        if (record.kind != DirectExactHistoryKind::world_return ||
            record.identity != ticket.ticket_id)
          continue;
        ++cur.return_matches;
        const bool exact =
            record.parent_identity == ticket.upstream_ticket_id &&
            record.source == ticket.motor_node &&
            record.subject == ticket.motor_channel &&
            record.context == ticket.context_signature &&
            record.event_tick == record.resident_tick &&
            record.resident_tick >= ticket.emission_tick &&
            (record.flags & kDirectHistoryVerifiedObservation) != 0u &&
            (record.flags & kDirectHistoryPayloadFlags) ==
                (ticket.mismatch_bits & kDirectHistoryPayloadFlags) &&
            record.resource_delta == ticket.settled_reward_q16;
        if (!exact) {
          refused = true;
          break;
        }
        cur.found_return_resident_tick = record.resident_tick;
      }
      cur.return_scan_index = record_count;
      if (refused || cur.return_matches > 1u)
        cur.flags |= kAffectCursorReturnRefused;
      else if (cur.return_matches == 1u)
        cur.flags |= kAffectCursorReturnFound;
    }
    if ((cur.flags & kAffectCursorReturnRefused) != 0u) {
      state->processed_ticket_by_slot[slot] = ticket.ticket_id;
      continue;
    }
    if ((cur.flags & kAffectCursorReturnFound) == 0u) continue;
    // Resume the ancestry walk from each parent's recorded frontier. The
    // verdict over [0, record_count) is order-independent -- mixed roots,
    // unverified boundaries and pending overflow refuse regardless of the
    // discovery order, and one resolved root is order-independent -- so the
    // resumed sweep returns exactly what a fresh full rescan would.
    bool ancestry_resolved = false;
    if ((cur.flags & kAffectCursorAncestryDead) == 0u) {
      bool aborted = false;
      while (!aborted) {
        bool scanned = false;
        for (std::uint32_t j = 0u; j < cur.pending_count && !aborted; ++j) {
          if (cur.frontier[j] >= record_count) continue;
          scanned = true;
          std::uint32_t r = cur.frontier[j];
          for (; r < record_count; ++r) {
            const DirectExactHistoryRecord& record = records[r];
            if (record.identity != cur.pending[j] ||
                record.kind == DirectExactHistoryKind::empty)
              continue;
            if (record.kind == DirectExactHistoryKind::sensory_contact) {
              if ((record.flags & kDirectHistoryVerifiedObservation) == 0u ||
                  ((cur.flags & kAffectCursorSawRoot) != 0u &&
                   cur.resolved_root_channel !=
                       static_cast<std::int32_t>(record.subject))) {
                aborted = true;
                break;
              }
              if ((cur.flags & kAffectCursorSawRoot) == 0u) {
                cur.resolved_root_channel =
                    static_cast<std::int32_t>(record.subject);
                cur.flags |= kAffectCursorSawRoot;
              }
              continue;
            }
            const std::uint64_t parent = record.parent_identity;
            if (parent == 0u || parent == 0xffffffffffffffffULL) continue;
            bool known = false;
            for (std::uint32_t q = 0u; q < cur.pending_count; ++q)
              known |= cur.pending[q] == parent;
            if (known) continue;
            if (cur.pending_count == kAffectRootTraceCapacity) {
              aborted = true;
              break;
            }
            cur.pending[cur.pending_count] = parent;
            cur.frontier[cur.pending_count] = 0u;
            ++cur.pending_count;
          }
          if (!aborted) cur.frontier[j] = record_count;
        }
        // One confirming pass after every frontier has drained to the current
        // journal end: newly discovered parents scan their own prefix first.
        if (!scanned) break;
      }
      if (aborted)
        cur.flags |= kAffectCursorAncestryDead;
      else if ((cur.flags & kAffectCursorSawRoot) != 0u)
        ancestry_resolved = true;
    }
    std::uint32_t root_channel = static_cast<std::uint32_t>(
        cur.resolved_root_channel);
    if (!ancestry_resolved || root_channel < visceral_lo ||
        root_channel >= visceral_hi)
      continue;
    DirectAffectBodyEntry* entry =
        affect_deposit_slot(state, records, record_count, root_channel);
    if (entry == nullptr) continue;
    state->processed_ticket_by_slot[slot] = ticket.ticket_id;
    entry->last_evidence_tick = cur.found_return_resident_tick;
    const bool damaged =
        ticket.mismatch_bits != 0u || ticket.settled_reward_q16 < 0;
    if (damaged) {
      entry->damage_q16 =
          affect_rise_q16(entry->damage_q16, kAffectRiseGainQ16);
      ++entry->damage_samples;
      if ((entry->flags & kAffectEntryLastDepositDamage) != 0u) {
        entry->stress_q16 =
            affect_rise_q16(entry->stress_q16, kAffectRiseGainQ16);
        ++entry->stress_samples;
      }
      entry->flags |= kAffectEntryLastDepositDamage;
    } else {
      entry->vitality_q16 =
          affect_rise_q16(entry->vitality_q16, kAffectRiseGainQ16);
      ++entry->vitality_samples;
      entry->flags &= ~kAffectEntryLastDepositDamage;
    }
  }
  std::int32_t pressure = 0;
  std::int32_t vital_aggregate = 0;
  for (std::uint32_t i = 0u; i < state->count; ++i) {
    const DirectAffectBodyEntry& entry = state->entries[i];
    pressure += direct_adult_core::mul_q16(entry.damage_q16,
                                           kAffectDamageWeightQ16) +
                direct_adult_core::mul_q16(entry.stress_q16,
                                           kAffectStressWeightQ16);
    vital_aggregate += direct_adult_core::mul_q16(entry.vitality_q16,
                                                  kAffectVitalityWeightQ16);
  }
  state->preservation_pressure_q16 =
      direct_adult_core::clamp_q16(pressure, 0, direct_adult_core::kQ16One);
  state->vitality_aggregate_q16 =
      direct_adult_core::clamp_q16(vital_aggregate, 0,
                                   direct_adult_core::kQ16One);
  if (state->preservation_pressure_q16 >= kAffectHighThresholdQ16)
    state->gate = AffectBodyGate::preserve;
  else if (state->vitality_aggregate_q16 >= kAffectHighThresholdQ16)
    state->gate = AffectBodyGate::seek;
  else
    state->gate = AffectBodyGate::neutral;
}

// The signed modulation surface of one lane: vitality raises prospective
// valuation, damage and stress lower it. No arm of this header subtracts one
// affect mass from another to form a single scalar valence.
__device__ inline std::int32_t affect_channel_bias_q16(
    const DirectAffectBodyEntry* entry) {
  if (entry == nullptr) return 0;
  const std::int32_t bias =
      direct_adult_core::mul_q16(entry->vitality_q16,
                                 kAffectVitalityWeightQ16) -
      direct_adult_core::mul_q16(entry->damage_q16, kAffectDamageWeightQ16) -
      direct_adult_core::mul_q16(entry->stress_q16, kAffectStressWeightQ16);
  return direct_adult_core::clamp_q16(bias, -direct_adult_core::kQ16One,
                                      direct_adult_core::kQ16One);
}

// Modulate one authorized candidate's prospective valuation by its visceral
// lane's derived state. The candidate keeps its exact identities; modulation
// scales the surface only.
__device__ inline DirectAffectValuationReceipt affect_modulate_candidate(
    const DirectAffectBodyState* state, const DirectAffectCandidate& candidate) {
  DirectAffectValuationReceipt receipt{};
  receipt.action_ticket_id = candidate.action_ticket_id;
  receipt.upstream_ticket_id = candidate.upstream_ticket_id;
  receipt.participation_identity = candidate.participation_identity;
  receipt.root_channel = candidate.root_channel;
  receipt.base_valuation_q16 = candidate.base_valuation_q16;
  receipt.gate = state != nullptr ? state->gate : AffectBodyGate::neutral;
  std::int32_t bias = 0;
  if (state != nullptr) {
    const std::int32_t slot = affect_find_entry(state, candidate.root_channel);
    bias = affect_channel_bias_q16(slot >= 0 ? &state->entries[slot] : nullptr);
  }
  receipt.bias_q16 = bias;
  receipt.modulated_valuation_q16 = direct_adult_core::clamp_q16(
      candidate.base_valuation_q16 + bias, 0, direct_adult_core::kQ16One);
  return receipt;
}

}  // namespace substrate::direct_network

#endif  // HARDWARE_NATIVE_DIRECT_ADULT_AFFECT_BODY_CUH
