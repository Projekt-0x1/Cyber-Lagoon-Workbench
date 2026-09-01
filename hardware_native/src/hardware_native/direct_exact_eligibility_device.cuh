#ifndef HARDWARE_NATIVE_DIRECT_EXACT_ELIGIBILITY_DEVICE_CUH
#define HARDWARE_NATIVE_DIRECT_EXACT_ELIGIBILITY_DEVICE_CUH

// #1176's exact eligibility ledger admission, factored out of
// direct_adult_legacy_oracle.cu's anonymous namespace (internal linkage, one
// private copy per translation unit) into inline __device__ functions any
// translation unit can call identically. #1179 needs this because sparse,
// packed, materialized-implicit and state-owning-implicit participation must
// all feed the *same* record pool/bucket index/batch-admission policy --
// admitting through a second, parallel implementation would silently fork
// settlement authority between representations.
//
// Also owns the small device-visible view into the #1179 state-owner table
// (DirectRepresentationDeviceView) and the handful of inline helpers that let
// apply_return_credit_kernel treat an explicit-route-backed and an
// implicit-virtual-backed eligibility record uniformly.

#include "hardware_native/direct_canonical_evaluator_device.cuh"
#include "hardware_native/direct_execution_fabric.cuh"
#include "hardware_native/direct_representation_compiler_abi.cuh"
#include "hardware_native/direct_resource_ecology_abi.cuh"

namespace substrate::direct_adult {

__device__ inline std::uint32_t eligibility_lifetime(EligibilityHorizonClass horizon) {
  switch (horizon) {
    case EligibilityHorizonClass::short_dialogue:
      return 16u;
    case EligibilityHorizonClass::multi_turn:
      return 64u;
    case EligibilityHorizonClass::asynchronous_return:
      return 256u;
    case EligibilityHorizonClass::long_omission:
      return 1024u;
    case EligibilityHorizonClass::immediate:
    default:
      return kEligibilityLifetime;
  }
}

__device__ inline EligibilityHorizonClass eligibility_horizon_for_event(
    const ActivityEvent& event) {
  if (event.horizon >= 256u)
    return EligibilityHorizonClass::long_omission;
  if (event.horizon >= 64u)
    return EligibilityHorizonClass::asynchronous_return;
  if (event.horizon >= 16u)
    return EligibilityHorizonClass::multi_turn;
  if (event.horizon >= 4u)
    return EligibilityHorizonClass::short_dialogue;
  return EligibilityHorizonClass::immediate;
}

// #1179: deterministic logical identity for a #1187 implicit interaction,
// derived from (source, implicit family, virtual slot) exactly as specified
// -- the same triple always yields the same id regardless of which physical
// route slot/generation currently backs it.
__device__ inline DirectLogicalInteractionId derive_implicit_logical_id(
    std::uint32_t source, std::uint32_t implicit_family, std::uint32_t implicit_slot) {
  std::uint64_t value = (static_cast<std::uint64_t>(source) << 40) ^
                        (static_cast<std::uint64_t>(implicit_family) << 20) ^
                        static_cast<std::uint64_t>(implicit_slot);
  value ^= value >> 33;
  value *= 0xff51afd7ed558ccdull;
  value ^= value >> 33;
  value *= 0xc4ceb9fe1a85ec53ull;
  value ^= value >> 33;
  return DirectLogicalInteractionId{value == 0u ? 1u : value};
}

// `charge_new` distinguishes a genuinely new eligibility record from a
// compaction carry-over. The bank is double-buffered, so a surviving record is
// re-appended into the next bank every compaction; charging that as a fresh
// allocation would bill the same matter once per tick forever. Carry-overs keep
// the charge they already hold; only new admissions pay.
__device__ inline bool append_eligibility_record(
    DirectBrainV01 brain, DirectEligibilityRecord record,
    DirectEligibilityRecord* bank, std::uint32_t* count, std::uint32_t capacity,
    std::uint32_t* bucket_heads, std::uint32_t bucket_count,
    const std::uint32_t* batch_admit, AdultCounters* counters, bool new_participation,
    bool charge_new) {
  if (batch_admit != nullptr && *batch_admit == 0u) {
    if (counters != nullptr)
      atomicAdd(&counters->eligibility_capacity_rejects, 1u);
    return false;
  }
  // A STORE-SITE DUPLICATE FILTER WAS BUILT HERE AT RUNG 13 AND REMOVED AFTER
  // MEASUREMENT, which is worth recording because the idea is the obvious one and
  // will occur to the next reader. compact_eligibility_kernel collapses duplicate
  // claims only at compaction, i.e. after all minting for the tick, so the bank
  // must hold the whole un-collapsed population until then; a best-effort
  // same-claim probe of the bucket chain here would stop a duplicate from ever
  // taking a slot. Measured with propagated events admitted, it moved
  // capacity_rejects from 800 to 795 of 1024 and left the bank pinned at its 8193
  // ceiling exactly as before.
  //
  // It bought nothing because THE OVERFLOW IS NOT DUPLICATION. `eligibility_same_
  // claim` requires the same bound route, and propagated events that reach one
  // node along different paths hold different routes -- they are genuinely
  // different claims, which is the entire point of admitting depth. No correct
  // dedup can shrink that population, and one that did would be destroying causal
  // evidence. The cost of admitting depth is bank capacity and record lifetime,
  // not deduplication.
  //
  // Reservation before the bank slot is claimed. The bank's own `capacity`
  // parameter is a second, lower-level bound; the ledger is the authority and
  // must be able to refuse an admission the bank would have accepted.
  if (charge_new && !device_reserve_pool_units(brain.resource_ecology,
                                               DirectResourcePoolKind::eligibility_record, 1u)) {
    if (counters != nullptr)
      atomicAdd(&counters->eligibility_capacity_rejects, 1u);
    return false;
  }
  const std::uint32_t slot = atomicAdd(count, 1u);
  if (slot >= capacity) {
    atomicSub(count, 1u);
    if (charge_new)
      device_cancel_pool_reservation(brain.resource_ecology,
                                     DirectResourcePoolKind::eligibility_record, 1u);
    if (counters != nullptr)
      atomicAdd(&counters->eligibility_capacity_rejects, 1u);
    return false;
  }
  if (charge_new)
    device_commit_pool_units(brain.resource_ecology,
                             DirectResourcePoolKind::eligibility_record, 1u);
  // #1179: stamp stable logical identity + explicit locator for a record
  // whose bound route came from #1187 materialization. Every other record
  // (implicit_family == kInvalidIndex on its route, or an out-of-range
  // route) keeps logical_id.value == 0 and is untouched by any #1179 code
  // path from here on -- this is purely additive over pre-#1179 behavior.
  if (record.route.slot < brain.route_capacity) {
    const DirectRoute& bound = brain.routes[record.route.slot];
    if (bound.implicit_family != kInvalidIndex) {
      record.logical_id =
          derive_implicit_logical_id(bound.source, bound.implicit_family, bound.implicit_slot);
      record.locator.kind = DirectLocatorKind::explicit_route;
      record.locator.slot = record.route.slot;
      record.locator.generation = record.route.generation;
    }
  }
  record.next_in_bucket = kInvalidIndex;
  bank[slot] = record;
  __threadfence();
  const std::uint32_t bucket =
      eligibility_bucket_slot(record.source, record.context, record.ticket, bucket_count);
  const std::uint32_t prior = atomicExch(bucket_heads + bucket, slot);
  bank[slot].next_in_bucket = prior;
  __threadfence();
  if (counters != nullptr && prior != kInvalidIndex)
    atomicAdd(&counters->eligibility_index_collisions, 1u);
  if (new_participation && record.route.slot < brain.route_capacity) {
    DirectRoute& route = brain.routes[record.route.slot];
    atomicAdd(&route.eligibility_live_count, 1u);
    route.eligibility_q16 = record.strength_q16;
    route.eligibility_context = record.context;
    route.eligibility_expires = record.expiry_tick;
    route.predicted_context = record.predicted_context;
    route.eligibility_history = record.history_signature;
    route.eligibility_root = record.participation_root;
  }
  return true;
}

__device__ inline void release_route_eligibility_summary(
    DirectBrainV01 brain, const DirectEligibilityRecord& record) {
  if (record.route.slot >= brain.route_capacity)
    return;
  DirectRoute& route = brain.routes[record.route.slot];
  const std::uint32_t prior = atomicSub(&route.eligibility_live_count, 1u);
  if (prior <= 1u) {
    route.eligibility_live_count = 0u;
    route.eligibility_q16 = 0;
    route.eligibility_context = kInvalidIndex;
    route.eligibility_expires = 0u;
    route.predicted_context = kInvalidIndex;
    route.eligibility_history = 0u;
    route.eligibility_root = 0u;
  }
}

// -- #1179 state-owner-aware settlement helpers ---------------------------

struct DirectRepresentationDeviceView {
  DirectSourceRepresentationState* source_state;   // [node_count]
  std::uint32_t* claim_ordinal;                    // [node_count], touched-work arbitration scratch
  DirectRepresentationStateOwner* state_owners;     // [state_owner_capacity]
  std::uint32_t state_owner_capacity;
  DirectRepresentationCounters* counters;
  // #1235: Level A's own resident packed cache, added to this view so any
  // pass that already carries a DirectRepresentationDeviceView (currently
  // propagate_sparse_frontier_kernel) can consult resolve_packed_cache_winner
  // below without threading two more raw pointer parameters through its
  // signature. [node_count * kRepresentationResidentReserve] / [node_count].
  DirectPackedEntry* resident_entries;
  std::uint32_t* resident_entry_count;
};

// #1235: resolve a source's winning route from its #1179 Level A packed
// cache, replicating propagate_sparse_frontier_kernel's own context-rank
// tie-break exactly (exact-context beats shared-prefix beats context-free;
// conductance only breaks ties within one specificity tier) -- a second,
// looser notion of "the winner" here would let the packed path and the
// canonical path silently disagree on which route an event actually took.
//
// Returns false when the caller must fall back to the canonical linked-list
// scan: the source is not `active`, or has drifted since promotion
// (`source_revision` mismatch against the snapshot taken at promotion time).
// Returns true with *out_winner_slot == kInvalidIndex when the cache is
// current but no cached entry currently conducts above the floor -- the
// same "no winner" outcome the canonical scan can itself produce, not a
// staleness case.
//
// A cached entry whose own route slot has since been retracted or reused
// (dead or generation-mismatched against brain.topology.slot_meta) fails the
// WHOLE source back to canonical, not just that one entry: a half-stale
// cache that silently serves a mix of current and stale answers is worse
// than no cache at all, per this arc's own header documentation above.
//
// The cached entry itself supplies only structural identity (route_slot,
// route_generation) -- conductance_q16 and context_signature are read live
// from brain.routes at consult time, never trusted from the cached copy.
// Both mutate without bumping source_revision or route_generation (ordinary
// credit settlement changes conductance every tick it fires; a focal lesion
// can clear context_signature directly), so a cached scalar snapshot can
// silently drift out of sync with the live route while the slot itself
// stays perfectly live -- the packed array's real win is only skipping the
// next_route pointer chase to find the candidate set, not caching values.
__device__ inline bool resolve_packed_cache_winner(const DirectBrainV01& brain,
                                                    const DirectRepresentationDeviceView& rep,
                                                    std::uint32_t source_node,
                                                    std::uint64_t event_context_signature,
                                                    std::uint32_t* out_winner_slot,
                                                    std::uint32_t* out_winner_context_rank,
                                                    std::int64_t* out_winner_conductance) {
  if (rep.source_state == nullptr || rep.resident_entries == nullptr ||
      rep.resident_entry_count == nullptr)
    return false;
  const DirectSourceRepresentationState state = rep.source_state[source_node];
  if (state.lifecycle != DirectPackedLifecycle::active)
    return false;
  if (brain.nodes[source_node].source_revision != state.shadow_source_revision) {
    if (rep.counters != nullptr)
      atomicAdd(&rep.counters->packed_cache_fallbacks, 1u);
    return false;
  }

  const std::uint32_t count = rep.resident_entry_count[source_node];
  const DirectPackedEntry* entries =
      rep.resident_entries +
      static_cast<std::size_t>(source_node) * kRepresentationResidentReserve;

  std::uint32_t winner = kInvalidIndex;
  std::uint32_t winner_rank = 0u;
  std::int64_t winner_conductance = -1;
  for (std::uint32_t e = 0u; e < count; ++e) {
    const DirectPackedEntry entry = entries[e];
    const DirectRouteSlotMeta meta = brain.topology.slot_meta[entry.route_slot];
    if (meta.live == 0u || meta.generation != entry.route_generation) {
      if (rep.counters != nullptr)
        atomicAdd(&rep.counters->packed_cache_fallbacks, 1u);
      return false;
    }
    // Structural identity (which physical slot this entry names) is proven
    // live above; the route's own mutable fields are read live from
    // brain.routes rather than trusted from the cached copy. conductance_q16
    // changes every tick this route settles credit and context_signature can
    // be focally lesioned directly (both without bumping source_revision or
    // route_generation, since neither is a structural growth/retract), so a
    // cached scalar can go stale while the slot itself stays perfectly live.
    // The packed array is a flat shortlist of route slots for this source
    // (its real win: no next_route pointer chase) -- not a value cache.
    // #1236: this used to be a hand-copied second transcription of the
    // canonical floor/rank/tie-break. It agreed with the canonical scan only
    // as long as nobody edited one copy, which is not an invariant -- it is a
    // coincidence maintained by review. Both now call the same helper.
    const DirectRoute& route = brain.routes[entry.route_slot];
    if (!direct_canonical_route_admissible(route))
      continue;
    const std::uint32_t rank = direct_canonical_context_rank(route, event_context_signature);
    const std::int64_t conductance = route.conductance_q16;
    if (winner == kInvalidIndex || rank > winner_rank ||
        (rank == winner_rank && conductance > winner_conductance)) {
      winner = entry.route_slot;
      winner_rank = rank;
      winner_conductance = conductance;
    }
  }
  if (rep.counters != nullptr)
    atomicAdd(&rep.counters->packed_cache_consult_hits, 1u);
  *out_winner_slot = winner;
  *out_winner_context_rank = winner_rank;
  *out_winner_conductance = winner_conductance;
  return true;
}

__device__ inline std::uint32_t state_owner_bucket(std::uint64_t logical_id_value,
                                                    std::uint32_t capacity) {
  std::uint64_t v = logical_id_value;
  v ^= v >> 33;
  v *= 0xff51afd7ed558ccdull;
  v ^= v >> 33;
  return capacity == 0u ? 0u : static_cast<std::uint32_t>(v % capacity);
}

// Bounded linear probe to find the (unique, at most one live) state-owner
// record for a logical id. Returns kInvalidIndex if none is currently owned.
__device__ inline std::uint32_t find_state_owner(const DirectRepresentationDeviceView& rep,
                                                  DirectLogicalInteractionId logical_id) {
  if (rep.state_owners == nullptr || rep.state_owner_capacity == 0u || logical_id.value == 0u)
    return kInvalidIndex;
  std::uint32_t slot = state_owner_bucket(logical_id.value, rep.state_owner_capacity);
  for (std::uint32_t probe = 0u; probe < 32u; ++probe) {
    const DirectRepresentationStateOwner& owner = rep.state_owners[slot];
    if (owner.lifecycle != DirectStateOwnerLifecycle::free &&
        owner.logical_id.value == logical_id.value)
      return slot;
    slot = (slot + 1u) & (rep.state_owner_capacity - 1u);
  }
  return kInvalidIndex;
}

// True when this record's *current* physical backing (explicit route or
// state-owning implicit storage) is alive and matches the locator it holds.
// Centralizing this one branch keeps every pass over the eligibility bank
// (ticket scan / leader election / exact settlement) consistent with each
// other -- letting them diverge would corrupt exact-episode selection.
__device__ inline bool eligibility_locator_live(const DirectBrainV01& brain,
                                                const DirectRepresentationDeviceView& rep,
                                                const DirectEligibilityRecord& record) {
  if (record.locator.kind == DirectLocatorKind::implicit_virtual) {
    const std::uint32_t owner_slot = find_state_owner(rep, record.logical_id);
    if (owner_slot == kInvalidIndex)
      return false;
    const DirectRepresentationStateOwner& owner = rep.state_owners[owner_slot];
    // probation_pending_retract is included because records are rebound off
    // the explicit route (to unblock its retract admission) as soon as
    // migration begins, before the owner formally reaches `owned` -- see
    // rebind_representation_eligibility_kernel.
    return (owner.lifecycle == DirectStateOwnerLifecycle::owned ||
           owner.lifecycle == DirectStateOwnerLifecycle::probation_pending_retract) &&
           owner.owner_epoch == record.locator.generation;
  }
  if (record.route.slot >= brain.route_capacity)
    return false;
  const DirectRouteSlotMeta meta = brain.topology.slot_meta[record.route.slot];
  return meta.live != 0u && meta.generation == record.route.generation;
}

// Effective (target, flags, learned_output_word) for compatibility/credit
// evaluation, read from whichever representation currently backs the record.
// Returns false if the backing cannot be resolved (caller must then treat
// the record as unsettleable this event, same as a dead explicit route).
__device__ inline bool eligibility_effective_route(const DirectBrainV01& brain,
                                                    const DirectRepresentationDeviceView& rep,
                                                    const DirectEligibilityRecord& record,
                                                    std::uint32_t* out_target,
                                                    std::uint32_t* out_flags,
                                                    Word* out_learned_output_word,
                                                    std::uint32_t* out_owner_slot) {
  *out_owner_slot = kInvalidIndex;
  if (record.locator.kind == DirectLocatorKind::implicit_virtual) {
    const std::uint32_t owner_slot = find_state_owner(rep, record.logical_id);
    if (owner_slot == kInvalidIndex)
      return false;
    const DirectRepresentationStateOwner& owner = rep.state_owners[owner_slot];
    *out_target = owner.target;
    *out_flags = owner.route_flags;
    *out_learned_output_word = static_cast<Word>(owner.learned_output_word);
    *out_owner_slot = owner_slot;
    return true;
  }
  if (record.route.slot >= brain.route_capacity)
    return false;
  const DirectRoute& route = brain.routes[record.route.slot];
  *out_target = route.target;
  *out_flags = route.flags;
  *out_learned_output_word = route.learned_output_word;
  return true;
}

// Deposits signed credit either into the live explicit route (unchanged
// pre-#1179 behavior) or into the state owner's representation-independent
// residual (#1179). Exactly one of the two atomicAdds below fires.
__device__ inline void eligibility_deposit_credit(DirectBrainV01 brain,
                                                   const DirectRepresentationDeviceView& rep,
                                                   const DirectEligibilityRecord& record,
                                                   std::uint32_t owner_slot,
                                                   std::int32_t signed_credit) {
  if (record.locator.kind == DirectLocatorKind::implicit_virtual) {
    if (owner_slot < rep.state_owner_capacity) {
      DirectRepresentationStateOwner& owner = rep.state_owners[owner_slot];
      atomicAdd(reinterpret_cast<unsigned long long*>(&owner.implicit_credit_accumulator_q16),
                static_cast<unsigned long long>(static_cast<long long>(signed_credit)));
      // #1179 contradiction tracking: persistent negative settlement is what
      // "persistent contradiction invalidates the local state-owner
      // certificate" means operationally -- a streak of negative credit,
      // reset by any positive credit, drives advance_state_owner_lifecycle_kernel's
      // owned -> rematerialize_pending transition once it reaches the limit.
      if (signed_credit < 0)
        atomicAdd(&owner.contradiction_strikes, 1);
      else if (signed_credit > 0)
        atomicExch(&owner.contradiction_strikes, 0);
    }
    return;
  }
  if (record.route.slot < brain.route_capacity) {
    DirectRoute& route = brain.routes[record.route.slot];
    atomicAdd(&route.conductance_q16, signed_credit);
    atomicAdd(&route.last_credit_q16, signed_credit);
  }
}

// Rebinds one live eligibility record to `new_locator` (and, when moving to
// explicit_route, `new_route`) if it belongs to `logical_id`. Callers launch
// this over a bounded live-record range (exactly like every other per-tick
// eligibility pass in this codebase, never a whole-brain scan) from their
// own translation unit's __global__ wrapper -- kept as an inline device
// function rather than a kernel here so this shared header never carries an
// externally-linked __global__ symbol multiple translation units would
// collide on.
__device__ inline bool rebind_eligibility_record_if_owned(
    DirectBrainV01 brain, DirectEligibilityRecord& record, DirectLogicalInteractionId logical_id,
    DirectInteractionLocator new_locator, DirectRouteHandle new_route,
    DirectRepresentationCounters* counters) {
  if (record.logical_id.value != logical_id.value || record.state != EligibilityRecordState::live)
    return false;
  const bool was_explicit = record.locator.kind == DirectLocatorKind::explicit_route;
  const bool becomes_explicit = new_locator.kind == DirectLocatorKind::explicit_route;
  if (was_explicit && !becomes_explicit) {
    // Moving off the explicit route this record was blocking: release its
    // contribution to that route's outstanding #1176 eligibility so the
    // fail-closed retract admission it was blocking can proceed once no
    // other live record still references the route.
    release_route_eligibility_summary(brain, record);
  }
  record.locator = new_locator;
  record.route = new_route;
  if (!was_explicit && becomes_explicit && new_route.slot < brain.route_capacity) {
    const DirectRouteSlotMeta meta = brain.topology.slot_meta[new_route.slot];
    // Only re-establish route-eligibility bookkeeping when the target slot
    // is genuinely this record's live, current occupant -- a rollback whose
    // origin slot was already reused by an unrelated interaction must never
    // graft this record's eligibility onto that unrelated route (a real ABA
    // hazard); the stale generation mismatch instead leaves the record
    // permanently unsettleable, fail-closed rather than fail-open.
    if (meta.live != 0u && meta.generation == new_route.generation) {
      DirectRoute& route = brain.routes[new_route.slot];
      atomicAdd(&route.eligibility_live_count, 1u);
      route.eligibility_q16 = record.strength_q16;
      route.eligibility_context = record.context;
      route.eligibility_expires = record.expiry_tick;
      route.predicted_context = record.predicted_context;
      route.eligibility_history = record.history_signature;
      route.eligibility_root = record.participation_root;
    }
  }
  if (counters != nullptr)
    atomicAdd(&counters->eligibility_rebinds, 1u);
  return true;
}

}  // namespace substrate::direct_adult

#endif  // HARDWARE_NATIVE_DIRECT_EXACT_ELIGIBILITY_DEVICE_CUH
