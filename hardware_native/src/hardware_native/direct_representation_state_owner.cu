#include <cuda_runtime.h>

#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

#include "hardware_native/direct_dynamic_topology_arena.cuh"
#include "hardware_native/direct_representation_compiler.cuh"
#include "hardware_native/direct_resource_ecology_abi.cuh"

// Level B (state-owning representation): a stable materialized #1187
// interaction moves its causal backing from an explicit DirectRoute slot
// into procedural storage while live #1176 eligibility records are rebound
// to follow it. See direct_representation_compiler.cuh's file header for the
// full lifecycle description.

namespace substrate::direct_adult {
namespace {

void check_cuda(cudaError_t status, const char* what) {
  if (status != cudaSuccess)
    throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
}

std::uint32_t grid_for(std::uint32_t count, std::uint32_t block = 128u) {
  return count == 0u ? 1u : (count + block - 1u) / block;
}

constexpr std::int32_t kStateOwnerPriorityQ16 = 1 << 12;

__device__ inline std::uint32_t find_or_create_state_owner(
    DirectRepresentationDeviceView view, DirectLogicalInteractionId logical_id,
    DirectResourceEcologyState* ecology) {
  if (view.state_owners == nullptr || view.state_owner_capacity == 0u || logical_id.value == 0u)
    return kInvalidIndex;
  std::uint32_t slot = state_owner_bucket(logical_id.value, view.state_owner_capacity);
  for (std::uint32_t probe = 0u; probe < 32u; ++probe) {
    DirectRepresentationStateOwner& owner = view.state_owners[slot];
    if (owner.lifecycle != DirectStateOwnerLifecycle::free &&
        owner.logical_id.value == logical_id.value)
      return slot;
    if (owner.lifecycle == DirectStateOwnerLifecycle::free) {
      const unsigned int prior = atomicCAS(
          reinterpret_cast<unsigned int*>(&owner.lifecycle),
          static_cast<unsigned int>(DirectStateOwnerLifecycle::free),
          static_cast<unsigned int>(DirectStateOwnerLifecycle::shadow));
      if (prior == static_cast<unsigned int>(DirectStateOwnerLifecycle::free)) {
        owner.logical_id = logical_id;
        owner.contradiction_strikes = 0;
        owner.shadow_streak = 0;
        owner.implicit_credit_accumulator_q16 = 0;
        // #1179 P0.1 items 5/6: the whole table was charged/reserved in full
        // at creation (the array is real bytes whether or not any slot is
        // occupied); this claim moves one unit from reserved to live so the
        // ledger reads actual logical occupancy, not array capacity. The
        // array slot is already claimed regardless of this call's outcome --
        // committing cannot fail here by construction (a free slot existing
        // for the CAS to win already proves live_units < charged_units).
        device_commit_pool_units(ecology, DirectResourcePoolKind::representation_state_owner, 1u);
        return slot;
      }
      // Lost the claim race; the winner (or a prior owner of this id) may be
      // sitting here now -- fall through the loop to re-check by identity.
    }
    slot = (slot + 1u) & (view.state_owner_capacity - 1u);
  }
  return kInvalidIndex;
}

// One deterministic winning ordinal per touched source drives that source's
// state-owner lifecycle this tick, reusing the exact same touched-work claim
// arbitration Level A already computed this tick (view.claim_ordinal) --
// the two lifecycles share "this thread exclusively owns node N's
// representation decisions this tick," not a scratch array whose meaning
// differs between them.
// gh #1338: bounded to the device-resident live frontier count alone, not
// min(*frontier_count, frontier_work) -- same birth defect #1304/#1331 fixed
// in sibling kernels. frontier_work is the host-tracked bound that ramps up
// over ~8 ticks after birth or an idle-fold reset; the caller's grid now
// covers frontier_capacity, so every thread up to *frontier_count runs.
__global__ void advance_state_owner_lifecycle_kernel(
    DirectBrainV01 brain, const ActivityEvent* frontier, const std::uint32_t* frontier_count,
    DirectRepresentationDeviceView view) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= *frontier_count)
    return;
  const std::uint32_t node = frontier[i].node;
  if (node >= brain.node_count || view.claim_ordinal[node] != i)
    return;

  const DirectNode dn = brain.nodes[node];
  std::uint32_t slot = dn.first_route;
  std::uint32_t candidate_slot = kInvalidIndex;
  DirectRoute candidate{};
  for (std::uint32_t n = 0u; n < dn.route_count && slot != kInvalidIndex; ++n) {
    const DirectRoute route = brain.routes[slot];
    if (route.implicit_family != kInvalidIndex) {
      candidate = route;
      candidate_slot = slot;
      break;
    }
    slot = route.next_route;
  }

  // Also let an already-owned interaction whose explicit backing is long
  // gone keep advancing (contradiction/rematerialization) even though the
  // node no longer shows a matching materialized route -- look it up by
  // scanning existing owners this node originated, bounded to one probe walk
  // per lifecycle already-known logical id. Since candidate discovery above
  // requires a live explicit route, an `owned` interaction is instead found
  // via the record of the winning node the FIRST time it was migrated
  // (origin_source == node); rehash using the same triple whenever the
  // explicit route is still visible, otherwise fall through using whatever
  // owner this source most recently created is tracked by the caller-side
  // rebind pass instead. For this first landing, contradiction/rematerialize
  // progression is driven from the `owned`-branch check below, keyed off the
  // logical id derived from the *last* seen candidate; a source whose
  // explicit route has already been reclaimed simply has no candidate here
  // and its owned record is advanced next time this exact node is touched
  // with owner.lifecycle already `owned` via the id recomputed from the
  // owner's own stored implicit_family/implicit_slot instead.
  DirectLogicalInteractionId logical_id{0u};
  if (candidate_slot != kInvalidIndex) {
    logical_id = derive_implicit_logical_id(node, candidate.implicit_family, candidate.implicit_slot);
  } else if (view.source_state[node].lifecycle == DirectPackedLifecycle::plastic) {
    // No live materialized route on this source. Nothing to look up without
    // already knowing implicit_family/implicit_slot; owned/contradiction
    // progression for this source (if any) resumes via the rebind pass's
    // stored owner_epoch bookkeeping instead of this kernel.
    return;
  } else {
    return;
  }

  const std::uint32_t owner_slot =
      find_or_create_state_owner(view, logical_id, brain.resource_ecology);
  if (owner_slot == kInvalidIndex)
    return;
  DirectRepresentationStateOwner& owner = view.state_owners[owner_slot];

  switch (owner.lifecycle) {
    case DirectStateOwnerLifecycle::shadow: {
      const DirectRouteSlotMeta meta = brain.topology.slot_meta[candidate_slot];
      if (candidate_slot != owner.origin_route_slot ||
          meta.generation != owner.origin_route_generation) {
        owner.origin_source = node;
        owner.origin_route_slot = candidate_slot;
        owner.origin_route_generation = meta.generation;
        owner.target = candidate.target;
        owner.implicit_family = candidate.implicit_family;
        owner.implicit_slot = candidate.implicit_slot;
        owner.delay = candidate.delay;
        owner.route_flags = candidate.flags;
        owner.learned_output_word = candidate.learned_output_word;
        owner.context_signature = candidate.context_signature;
        owner.shadow_streak = 0u;
        break;
      }
      ++owner.shadow_streak;
      if (owner.shadow_streak >= kRepresentationProbationTouches) {
        // Bumped here (not at the later `owned` transition): rebind to
        // implicit_virtual starts this same migration, before the retract
        // itself commits -- see rebind_representation_eligibility_kernel and
        // eligibility_locator_live's probation_pending_retract branch. One
        // epoch value must span the whole explicit->implicit transition.
        owner.owner_epoch += 1u;
        owner.lifecycle = DirectStateOwnerLifecycle::probation_pending_retract;
        // The retract itself is submitted (and, since outstanding #1176
        // eligibility on this exact route can fail-close the first attempt,
        // resubmitted) by advance_state_owner_table_kernel below -- that
        // kernel is not frontier-gated, so it keeps retrying even on ticks
        // where this source itself is not touched, which continued
        // real touches on this same route otherwise would block forever.
        // Outstanding #1176 records on this route are rebound off it by
        // rebind_representation_eligibility_kernel this same tick, which is
        // what actually clears route.eligibility_q16 for the retract below.
      }
      break;
    }
    default:
      break;
  }
}

// Bounded scan over the (small, fixed-capacity) state-owner table -- never a
// whole-brain scan -- that advances probation_pending_retract retries,
// owned's contradiction check, and rematerialize_pending's growth-commit
// check. Deliberately NOT frontier-gated: a source can go quiet (no new
// touches) while its route's outstanding #1176 eligibility is still aging
// out, and the retract retry above must keep running regardless.
__global__ void advance_state_owner_table_kernel(DirectBrainV01 brain,
                                                 DirectRepresentationDeviceView view,
                                                 DirectTopologyDeviceView topology,
                                                 std::uint32_t capacity) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= capacity)
    return;
  DirectRepresentationStateOwner& owner = view.state_owners[i];

  if (owner.lifecycle == DirectStateOwnerLifecycle::probation_pending_retract) {
    if (owner.origin_route_slot >= brain.route_capacity)
      return;
    const DirectRouteSlotMeta meta = brain.topology.slot_meta[owner.origin_route_slot];
    if (meta.live == 0u && meta.generation != owner.origin_route_generation) {
      // owner_epoch was already bumped when this owner entered
      // probation_pending_retract (see advance_state_owner_lifecycle_kernel)
      // -- records rebound to implicit_virtual during that window already
      // carry the correct epoch, so it must not change again here.
      owner.lifecycle = DirectStateOwnerLifecycle::owned;
      atomicAdd(&view.counters->state_owner_migrations, 1u);
    } else if (meta.live != 0u && meta.generation != owner.origin_route_generation) {
      // Slot was retracted and already reused by something unrelated before
      // this migration could be considered committed. Roll back: this
      // interaction never left explicit-route authority.
      owner.lifecycle = DirectStateOwnerLifecycle::free;
      device_uncommit_pool_units(brain.resource_ecology,
                                 DirectResourcePoolKind::representation_state_owner, 1u);
      atomicAdd(&view.counters->state_owner_rollbacks, 1u);
    } else if (meta.live != 0u && meta.generation == owner.origin_route_generation) {
      // Still the exact route we asked to retract. Resubmit: idempotent,
      // and succeeds the moment nothing is still #1176-eligible on it.
      submit_direct_retract_proposal(topology, i, owner.origin_source, owner.origin_route_slot,
                                     owner.origin_route_generation, kStateOwnerPriorityQ16);
    }
    return;
  }

  if (owner.lifecycle == DirectStateOwnerLifecycle::owned) {
    if (owner.contradiction_strikes >= kRepresentationContradictionStrikeLimit) {
      const std::int32_t procedural =
          owner.implicit_family < brain.implicit.family_count
              ? brain.implicit.families[owner.implicit_family].base_conductance_q16
              : 0;
      const std::int64_t exact =
          static_cast<std::int64_t>(procedural) + owner.implicit_credit_accumulator_q16;
      submit_direct_growth_proposal(topology, i, owner.origin_source, owner.target,
                                    owner.context_signature, clamp_conductance(exact), 0,
                                    owner.delay, owner.route_flags, kStateOwnerPriorityQ16,
                                    owner.implicit_family, owner.implicit_slot);
      owner.lifecycle = DirectStateOwnerLifecycle::rematerialize_pending;
    }
    return;
  }

  if (owner.lifecycle == DirectStateOwnerLifecycle::rematerialize_pending) {
    if (owner.origin_source >= brain.node_count)
      return;
    // A fresh materialized route for this exact (source, family, slot),
    // physically distinct from the reclaimed origin slot, means the growth
    // committed.
    const DirectNode dn = brain.nodes[owner.origin_source];
    std::uint32_t rescan = dn.first_route;
    for (std::uint32_t n = 0u; n < dn.route_count && rescan != kInvalidIndex; ++n) {
      const DirectRoute r = brain.routes[rescan];
      if (r.implicit_family == owner.implicit_family && r.implicit_slot == owner.implicit_slot &&
          rescan != owner.origin_route_slot) {
        owner.owner_epoch += 1u;
        owner.lifecycle = DirectStateOwnerLifecycle::free;
        device_uncommit_pool_units(brain.resource_ecology,
                                   DirectResourcePoolKind::representation_state_owner, 1u);
        atomicAdd(&view.counters->state_owner_shatters, 1u);
        break;
      }
      rescan = r.next_route;
    }
  }
}

// Bounded scan over the live eligibility bank (never a whole-brain scan)
// that rebinds every record whose logical id just finished migrating this
// tick, in either direction.
__global__ void rebind_representation_eligibility_kernel(
    DirectBrainV01 brain, DirectEligibilityRecord* bank, const std::uint32_t* count,
    std::uint32_t capacity, DirectRepresentationDeviceView view) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= min(*count, capacity))
    return;
  DirectEligibilityRecord& record = bank[i];
  if (record.logical_id.value == 0u || record.state != EligibilityRecordState::live)
    return;
  const std::uint32_t owner_slot = find_state_owner(view, record.logical_id);
  if (owner_slot == kInvalidIndex)
    return;
  const DirectRepresentationStateOwner& owner = view.state_owners[owner_slot];
  // Rebinding starts as soon as migration begins (probation_pending_retract),
  // not only once `owned` is reached: this is what actually releases the
  // explicit route's outstanding #1176 eligibility so its own fail-closed
  // retract admission can succeed -- see eligibility_locator_live's matching
  // probation_pending_retract branch and rebind_eligibility_record_if_owned's
  // release_route_eligibility_summary call.
  if ((owner.lifecycle == DirectStateOwnerLifecycle::owned ||
      owner.lifecycle == DirectStateOwnerLifecycle::probation_pending_retract) &&
      record.locator.kind != DirectLocatorKind::implicit_virtual) {
    DirectInteractionLocator new_locator{};
    new_locator.kind = DirectLocatorKind::implicit_virtual;
    new_locator.slot = owner_slot;
    new_locator.generation = owner.owner_epoch;
    DirectRouteHandle invalid_route{};
    invalid_route.slot = kInvalidIndex;
    invalid_route.generation = 0u;
    rebind_eligibility_record_if_owned(brain, record, record.logical_id, new_locator,
                                       invalid_route, view.counters);
  } else if (owner.lifecycle == DirectStateOwnerLifecycle::free &&
             record.locator.kind == DirectLocatorKind::implicit_virtual) {
    // Rolled back or shattered back to explicit this tick: point live
    // records at the fresh explicit locator recorded by
    // append_committed_growth_eligibility_kernel's own call to
    // append_eligibility_record for any *new* participation, but existing
    // *live* records must be rebound here since they predate that growth.
    DirectInteractionLocator new_locator{};
    new_locator.kind = DirectLocatorKind::explicit_route;
    new_locator.slot = owner.origin_route_slot;
    new_locator.generation = owner.origin_route_generation;
    DirectRouteHandle explicit_route{};
    explicit_route.slot = owner.origin_route_slot;
    explicit_route.generation = owner.origin_route_generation;
    rebind_eligibility_record_if_owned(brain, record, record.logical_id, new_locator,
                                       explicit_route, view.counters);
  }
}

}  // namespace

void launch_direct_state_owner_step(DirectAdultRuntime* runtime, std::uint32_t frontier_work,
                                    DirectEligibilityRecord* eligibility_bank,
                                    std::uint32_t* eligibility_count,
                                    std::uint32_t block_size) {
  if (runtime == nullptr || runtime->representation == nullptr)
    return;
  DirectRepresentationRuntime& rep = *runtime->representation;
  // Frontier-gated: shadow discovery reads this tick's frontier directly, so
  // it has nothing to do when frontier_work == 0.
  if (frontier_work != 0u) {
    // gh #1338: grid covers frontier_capacity, not frontier_work -- see the
    // kernel's own comment above.
    // advance_state_owner_lifecycle_kernel emits no topology proposal (it
    // only drives free->shadow->probation_pending_retract transitions), so
    // unlike the table kernel below it needs no reset_direct_topology_proposals
    // / launch_direct_topology_epoch bracketing.
    advance_state_owner_lifecycle_kernel<<<grid_for(runtime->frontier_capacity, block_size),
                                           block_size, 0, runtime->stream>>>(
        runtime->brain, runtime->frontier, runtime->frontier_count, rep.view);
    check_cuda(cudaGetLastError(), "launch state-owner lifecycle step");
  }

  // NOT frontier-gated from here down: a probation_pending_retract state
  // owner must keep retrying its retract every tick even when frontier_work
  // is 0 for many consecutive ticks (a pure-idle advance), or "eventually
  // shatters back to explicit" / "eventually claims the freed slot" never
  // becomes true.
  reset_direct_topology_proposals(runtime->topology_runtime, rep.state_owner_capacity,
                                  runtime->stream);
  advance_state_owner_table_kernel<<<grid_for(rep.state_owner_capacity, block_size), block_size, 0,
                                     runtime->stream>>>(
      runtime->brain, rep.view, runtime->topology_runtime->view, rep.state_owner_capacity);
  check_cuda(cudaGetLastError(), "launch state-owner table step");
  launch_direct_topology_epoch(&runtime->brain, runtime->topology_runtime, rep.state_owner_capacity,
                               runtime->counters, block_size, runtime->stream);

  // Grid sized by full capacity, not runtime->eligibility_launch_bound: that
  // bound is a stale pre-tick estimate, but a record can be appended into
  // next_eligibility_bank as late as this SAME tick's own propagation pass
  // (e.g. a fresh explicit-route participation on an already-owned logical
  // interaction) -- launching by the stale bound would leave that record's
  // index uncovered by any thread, so it would never get rebound even though
  // the kernel's own internal min(*count, capacity) check would have allowed
  // it. The kernel body is a cheap no-op past the real live count either way.
  rebind_representation_eligibility_kernel<<<grid_for(runtime->eligibility_capacity, block_size),
                                              block_size, 0, runtime->stream>>>(
      runtime->brain, eligibility_bank, eligibility_count,
      runtime->eligibility_capacity, rep.view);
  check_cuda(cudaGetLastError(), "launch representation eligibility rebind");
}

bool lesion_representation_state_owner(DirectAdultRuntime* runtime,
                                       DirectLogicalInteractionId logical_id) {
  if (runtime == nullptr || runtime->representation == nullptr || logical_id.value == 0u)
    return false;
  DirectRepresentationRuntime& rep = *runtime->representation;
  std::vector<DirectRepresentationStateOwner> host(rep.state_owner_capacity);
  check_cuda(cudaMemcpy(host.data(), rep.view.state_owners,
                        sizeof(DirectRepresentationStateOwner) * rep.state_owner_capacity,
                        cudaMemcpyDeviceToHost),
             "read state owners for lesion");
  for (std::uint32_t i = 0u; i < rep.state_owner_capacity; ++i) {
    if (host[i].lifecycle == DirectStateOwnerLifecycle::owned &&
        host[i].logical_id.value == logical_id.value) {
      host[i].lifecycle = DirectStateOwnerLifecycle::lesioned;
      check_cuda(cudaMemcpy(&rep.view.state_owners[i], &host[i], sizeof(host[i]),
                            cudaMemcpyHostToDevice),
                 "write state owner lesion");
      return true;
    }
  }
  return false;
}

bool unlesion_representation_state_owner(DirectAdultRuntime* runtime,
                                         DirectLogicalInteractionId logical_id) {
  if (runtime == nullptr || runtime->representation == nullptr || logical_id.value == 0u)
    return false;
  DirectRepresentationRuntime& rep = *runtime->representation;
  std::vector<DirectRepresentationStateOwner> host(rep.state_owner_capacity);
  check_cuda(cudaMemcpy(host.data(), rep.view.state_owners,
                        sizeof(DirectRepresentationStateOwner) * rep.state_owner_capacity,
                        cudaMemcpyDeviceToHost),
             "read state owners for unlesion");
  for (std::uint32_t i = 0u; i < rep.state_owner_capacity; ++i) {
    if (host[i].lifecycle == DirectStateOwnerLifecycle::lesioned &&
        host[i].logical_id.value == logical_id.value) {
      host[i].lifecycle = DirectStateOwnerLifecycle::owned;
      check_cuda(cudaMemcpy(&rep.view.state_owners[i], &host[i], sizeof(host[i]),
                            cudaMemcpyHostToDevice),
                 "write state owner unlesion");
      return true;
    }
  }
  return false;
}

}  // namespace substrate::direct_adult
