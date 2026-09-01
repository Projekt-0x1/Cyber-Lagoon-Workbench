#include <cuda_runtime.h>

#include <stdexcept>
#include <string>
#include <vector>

#include "hardware_native/direct_network_recipe_abi.cuh"
#include "hardware_native/direct_representation_compiler.cuh"

namespace substrate::direct_adult {
namespace {

void check_cuda(cudaError_t status, const char* what) {
  if (status != cudaSuccess)
    throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
}

std::uint32_t grid_for(std::uint32_t count, std::uint32_t block = 128u) {
  return count == 0u ? 1u : (count + block - 1u) / block;
}

constexpr std::uint32_t kNoRepresentationClaim = 0xffffffffu;

// gh #1331: bounded to the device-resident live frontier count alone, not
// min(*frontier_count, frontier_work) -- frontier_work is the host-tracked
// bound that ramps up over ~8 ticks after birth or an idle-fold reset (see
// #1304), and during that window it can trail *frontier_count. The min()
// used to cap real processing at the ramping bound even though the caller's
// grid (see launch_direct_representation_compiler_step) now covers the full
// frontier_capacity, so every thread up to *frontier_count gets to run.
__global__ void reset_representation_claims_kernel(const ActivityEvent* frontier,
                                                    const std::uint32_t* frontier_count,
                                                    std::uint32_t node_count,
                                                    std::uint32_t* claim_ordinal) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= *frontier_count)
    return;
  const std::uint32_t node = frontier[i].node;
  if (node < node_count)
    claim_ordinal[node] = kNoRepresentationClaim;
}

// gh #1331: see reset_representation_claims_kernel above.
__global__ void arbitrate_representation_claims_kernel(const ActivityEvent* frontier,
                                                        const std::uint32_t* frontier_count,
                                                        std::uint32_t node_count,
                                                        std::uint32_t* claim_ordinal) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= *frontier_count)
    return;
  const std::uint32_t node = frontier[i].node;
  if (node < node_count)
    atomicMin(&claim_ordinal[node], i);
}

// One deterministic winning ordinal per touched source commits that
// source's packed-cache lifecycle this tick -- CUDA scheduling never
// decides it. Bounded by the current frontier, never a whole-brain scan.
// gh #1331: see reset_representation_claims_kernel above.
__global__ void advance_packed_cache_lifecycle_kernel(
    DirectBrainV01 brain, const ActivityEvent* frontier, const std::uint32_t* frontier_count,
    DirectRepresentationDeviceView view,
    DirectPackedEntry* resident_entries, std::uint32_t* resident_entry_count) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= *frontier_count)
    return;
  const std::uint32_t node = frontier[i].node;
  if (node >= brain.node_count || view.claim_ordinal[node] != i)
    return;

  const DirectNode dn = brain.nodes[node];
  DirectSourceRepresentationState state = view.source_state[node];

  if (state.lifecycle == DirectPackedLifecycle::lesioned)
    return;  // Focal lesion holds until an explicit host unlesion call.

  if (dn.route_count == 0u || dn.route_count > kRepresentationResidentReserve) {
    // Not a stable small-degree WTA/sparse candidate (or currently empty).
    // A source that grew past the resident reserve during shadow/probation
    // demotes back to plastic rather than caching a truncated view.
    if (state.lifecycle != DirectPackedLifecycle::plastic) {
      state.lifecycle = DirectPackedLifecycle::plastic;
      state.shadow_agree_streak = 0u;
      view.source_state[node] = state;
    }
    return;
  }

  switch (state.lifecycle) {
    case DirectPackedLifecycle::plastic: {
      // #1179: a context-conditioned WTA/sparse route set (more than one
      // resident target) is explicitly rejected from #1175 Tensor
      // promotion here -- counted, never silently constructed as a fake
      // dense matrix. A single-target source is left uncounted (it is not
      // a WTA competition in the first place).
      bool multi_target = false;
      std::uint32_t slot = dn.first_route;
      std::uint32_t first_target = kInvalidIndex;
      for (std::uint32_t n = 0u; n < dn.route_count && slot != kInvalidIndex; ++n) {
        const std::uint32_t target = brain.routes[slot].target;
        if (first_target == kInvalidIndex)
          first_target = target;
        else if (target != first_target)
          multi_target = true;
        slot = brain.routes[slot].next_route;
      }
      if (multi_target)
        atomicAdd(&view.counters->tensor_rejections, 1u);
      state.lifecycle = DirectPackedLifecycle::shadow;
      state.shadow_agree_streak = 0u;
      state.shadow_source_revision = dn.source_revision;
      break;
    }
    case DirectPackedLifecycle::shadow: {
      if (dn.source_revision != state.shadow_source_revision) {
        state.shadow_agree_streak = 0u;
        state.shadow_source_revision = dn.source_revision;
      } else {
        ++state.shadow_agree_streak;
        // #1179 A3: a source that has already lost the churn bet
        // (churn_strikes > 0) must clear a longer streak before another
        // expensive probation->active rebuild is trusted to it. The counter
        // fires exactly once per shadow period, at the point promotion would
        // have happened under the base rule, so it isolates "would have
        // promoted, was withheld for churn" from every other shadow tick.
        if (state.shadow_agree_streak == kRepresentationProbationTouches &&
            state.churn_strikes > 0u)
          atomicAdd(&view.counters->churn_promotion_deferred, 1u);
        const std::uint32_t required =
            kRepresentationProbationTouches * (1u + state.churn_strikes);
        if (state.shadow_agree_streak >= required)
          state.lifecycle = DirectPackedLifecycle::probation;
      }
      break;
    }
    case DirectPackedLifecycle::probation: {
      if (dn.source_revision != state.shadow_source_revision) {
        state.lifecycle = DirectPackedLifecycle::shadow;
        state.shadow_agree_streak = 0u;
        state.shadow_source_revision = dn.source_revision;
        break;
      }
      std::uint32_t slot = dn.first_route;
      std::uint32_t n = 0u;
      for (; n < dn.route_count && n < kRepresentationResidentReserve && slot != kInvalidIndex; ++n) {
        const DirectRoute route = brain.routes[slot];
        const DirectRouteSlotMeta meta = brain.topology.slot_meta[slot];
        DirectPackedEntry entry{};
        entry.target = route.target;
        entry.conductance_q16 = route.conductance_q16;
        entry.delay = route.delay;
        entry.route_flags = route.flags;
        entry.context_signature = route.context_signature;
        entry.route_slot = slot;
        entry.route_generation = meta.generation;
        entry.learned_output_word = route.learned_output_word;
        entry.reserved16 = 0u;
        resident_entries[node * kRepresentationResidentReserve + n] = entry;
        slot = route.next_route;
      }
      resident_entry_count[node] = n;
      state.lifecycle = DirectPackedLifecycle::active;
      state.shadow_source_revision = dn.source_revision;
      state.active_stable_touches = 0u;
      atomicAdd(&view.counters->packed_cache_promotions, 1u);
      break;
    }
    case DirectPackedLifecycle::active: {
      if (dn.source_revision != state.shadow_source_revision) {
        atomicAdd(&view.counters->packed_cache_stale_refresh, 1u);
        state.lifecycle = DirectPackedLifecycle::shadow;
        state.shadow_agree_streak = 0u;
        state.shadow_source_revision = dn.source_revision;
        // #1179 A3: this active period ended in a real membership change,
        // not merely a mutable-field update (those never touch
        // source_revision) -- exactly the churn event the backoff exists to
        // price. Capped so a permanently unstable source still gets a
        // bounded, not unbounded, required-streak ceiling.
        state.churn_strikes = min(state.churn_strikes + 1u, kRepresentationChurnStrikeCap);
      } else {
        ++state.active_stable_touches;
        // #1179 A3 recovery: earn back one strike per full backed-off window
        // of touches survived without a fresh demotion, so a source that
        // churned once and then stabilized is not held to the penalty
        // forever -- the backoff decays with demonstrated stability.
        if (state.churn_strikes > 0u) {
          const std::uint32_t recovery_window =
              kRepresentationProbationTouches * (1u + state.churn_strikes);
          if (state.active_stable_touches >= recovery_window) {
            --state.churn_strikes;
            state.active_stable_touches = 0u;
            atomicAdd(&view.counters->churn_strikes_recovered, 1u);
          }
        }
      }
      break;
    }
    default:
      break;
  }
  view.source_state[node] = state;
}

// Independent-evaluation microbenchmark kernels used only by
// cuda_direct_representation_compiler_contract's CUDA-event timing gate.
// Each of the `iterations` threads independently re-derives the winning
// (max-conductance) target exactly as the real per-event evaluation would,
// over the resident flat array vs. the canonical route linked list, so the
// comparison isolates the representation's own access pattern rather than
// any other adult-step overhead. Neither kernel performs a D2H readback.
// `lanes` independent threads each repeat their evaluation `steps` times
// sequentially -- steps is the literal "2048 identical steps" the hard
// contract measures; lanes gives CUDA event timing enough total duration to
// resolve above launch/measurement noise. Each lane's own steps are a
// dependent chain (this step's result does not gate the next, but the same
// entries/route-slot memory is revisited every step), so the array-vs-
// linked-list access pattern this is meant to isolate dominates over lane
// count, not the other way around.
__global__ void resident_cache_probe_kernel(const DirectPackedEntry* entries,
                                            const std::uint32_t* entry_count_ptr,
                                            std::uint32_t lanes, std::uint32_t steps,
                                            std::uint32_t* out_checksum) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= lanes)
    return;
  const std::uint32_t entry_count = *entry_count_ptr;
  std::uint32_t acc = 0u;
  for (std::uint32_t s = 0u; s < steps; ++s) {
    std::int64_t best_conductance = -1;
    std::uint32_t best_target = 0u;
    for (std::uint32_t e = 0u; e < entry_count; ++e) {
      const DirectPackedEntry entry = entries[e];
      if (entry.conductance_q16 > best_conductance) {
        best_conductance = entry.conductance_q16;
        best_target = entry.target;
      }
    }
    acc += best_target;
  }
  atomicAdd(out_checksum, acc);
}

__global__ void canonical_sparse_probe_kernel(DirectBrainV01 brain, std::uint32_t source,
                                              std::uint32_t lanes, std::uint32_t steps,
                                              std::uint32_t* out_checksum) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= lanes || source >= brain.node_count)
    return;
  const DirectNode node = brain.nodes[source];
  std::uint32_t acc = 0u;
  for (std::uint32_t s = 0u; s < steps; ++s) {
    std::int64_t best_conductance = -1;
    std::uint32_t best_target = 0u;
    std::uint32_t slot = node.first_route;
    for (std::uint32_t n = 0u; n < node.route_count && slot != kInvalidIndex; ++n) {
      const DirectRoute route = brain.routes[slot];
      if (route.conductance_q16 > best_conductance) {
        best_conductance = route.conductance_q16;
        best_target = route.target;
      }
      slot = route.next_route;
    }
    acc += best_target;
  }
  atomicAdd(out_checksum, acc);
}

}  // namespace

DirectRepresentationRuntime* create_direct_representation_runtime(
    std::uint32_t node_count, std::uint32_t state_owner_capacity_hint) {
  auto* rep = new DirectRepresentationRuntime{};
  rep->node_count = node_count;
  std::uint32_t cap = 64u;
  while (cap < state_owner_capacity_hint)
    cap <<= 1u;
  rep->state_owner_capacity = cap;
  rep->view.state_owner_capacity = cap;

  check_cuda(cudaMalloc(&rep->view.source_state,
                        sizeof(DirectSourceRepresentationState) * node_count),
             "allocate representation source state");
  check_cuda(cudaMemset(rep->view.source_state, 0,
                        sizeof(DirectSourceRepresentationState) * node_count),
             "clear representation source state");
  check_cuda(cudaMalloc(&rep->view.claim_ordinal, sizeof(std::uint32_t) * node_count),
             "allocate representation claim scratch");
  check_cuda(cudaMalloc(&rep->view.state_owners, sizeof(DirectRepresentationStateOwner) * cap),
             "allocate representation state owners");
  check_cuda(cudaMemset(rep->view.state_owners, 0, sizeof(DirectRepresentationStateOwner) * cap),
             "clear representation state owners");
  check_cuda(cudaMalloc(&rep->view.counters, sizeof(DirectRepresentationCounters)),
             "allocate representation counters");
  check_cuda(cudaMemset(rep->view.counters, 0, sizeof(DirectRepresentationCounters)),
             "clear representation counters");
  check_cuda(cudaMalloc(&rep->resident_entries,
                        sizeof(DirectPackedEntry) * node_count * kRepresentationResidentReserve),
             "allocate representation resident entries");
  check_cuda(cudaMalloc(&rep->resident_entry_count, sizeof(std::uint32_t) * node_count),
             "allocate representation resident entry counts");
  check_cuda(cudaMemset(rep->resident_entry_count, 0, sizeof(std::uint32_t) * node_count),
             "clear representation resident entry counts");
  // #1235: mirrored onto the view so any kernel that already carries a
  // DirectRepresentationDeviceView can call resolve_packed_cache_winner
  // without a separate pair of raw pointer parameters.
  rep->view.resident_entries = rep->resident_entries;
  rep->view.resident_entry_count = rep->resident_entry_count;
  return rep;
}

void destroy_direct_representation_runtime(DirectRepresentationRuntime* rep) {
  if (rep == nullptr)
    return;
  cudaFree(rep->view.source_state);
  cudaFree(rep->view.claim_ordinal);
  cudaFree(rep->view.state_owners);
  cudaFree(rep->view.counters);
  cudaFree(rep->resident_entries);
  cudaFree(rep->resident_entry_count);
  delete rep;
}

bool lesion_representation_packed_source(DirectAdultRuntime* runtime, std::uint32_t source) {
  if (runtime == nullptr || runtime->representation == nullptr ||
      source >= runtime->representation->node_count)
    return false;
  DirectSourceRepresentationState state{};
  check_cuda(cudaMemcpy(&state, &runtime->representation->view.source_state[source],
                        sizeof(state), cudaMemcpyDeviceToHost),
             "read representation source state for lesion");
  state.lifecycle = DirectPackedLifecycle::lesioned;
  check_cuda(cudaMemcpy(&runtime->representation->view.source_state[source], &state, sizeof(state),
                        cudaMemcpyHostToDevice),
             "write representation source lesion");
  // Small host-side counter bump: acceptable here because this is an
  // explicit, infrequent host-initiated control action, not the per-tick
  // hot path the "no D2H readback" landing rule targets.
  DirectRepresentationCounters current{};
  check_cuda(cudaMemcpy(&current, runtime->representation->view.counters, sizeof(current),
                        cudaMemcpyDeviceToHost),
             "read representation counters for lesion");
  current.packed_cache_lesions += 1u;
  check_cuda(cudaMemcpy(runtime->representation->view.counters, &current, sizeof(current),
                        cudaMemcpyHostToDevice),
             "write representation counters for lesion");
  return true;
}

bool unlesion_representation_packed_source(DirectAdultRuntime* runtime, std::uint32_t source) {
  if (runtime == nullptr || runtime->representation == nullptr ||
      source >= runtime->representation->node_count)
    return false;
  DirectSourceRepresentationState state{};
  check_cuda(cudaMemcpy(&state, &runtime->representation->view.source_state[source],
                        sizeof(state), cudaMemcpyDeviceToHost),
             "read representation source state for unlesion");
  if (state.lifecycle != DirectPackedLifecycle::lesioned)
    return false;
  state.lifecycle = DirectPackedLifecycle::plastic;  // Rebuild candidacy from scratch.
  state.shadow_agree_streak = 0u;
  check_cuda(cudaMemcpy(&runtime->representation->view.source_state[source], &state, sizeof(state),
                        cudaMemcpyHostToDevice),
             "write representation source unlesion");
  return true;
}

Root256 direct_representation_state_root(const DirectRepresentationRuntime& rep) {
  std::vector<DirectSourceRepresentationState> host_state(rep.node_count);
  check_cuda(cudaMemcpy(host_state.data(), rep.view.source_state,
                        sizeof(DirectSourceRepresentationState) * rep.node_count,
                        cudaMemcpyDeviceToHost),
             "read representation source state for root");
  std::vector<DirectPackedEntry> host_entries(
      static_cast<std::size_t>(rep.node_count) * kRepresentationResidentReserve);
  check_cuda(cudaMemcpy(host_entries.data(), rep.resident_entries,
                        sizeof(DirectPackedEntry) * host_entries.size(), cudaMemcpyDeviceToHost),
             "read representation resident entries for root");
  std::vector<DirectRepresentationStateOwner> host_owners(rep.state_owner_capacity);
  check_cuda(cudaMemcpy(host_owners.data(), rep.view.state_owners,
                        sizeof(DirectRepresentationStateOwner) * rep.state_owner_capacity,
                        cudaMemcpyDeviceToHost),
             "read representation state owners for root");

  const Root256 state_root =
      direct_network::recipe::content_root(host_state.data(), sizeof(DirectSourceRepresentationState) *
                                                                  host_state.size());
  const Root256 entries_root = direct_network::recipe::content_root(
      host_entries.data(), sizeof(DirectPackedEntry) * host_entries.size());
  const Root256 owners_root = direct_network::recipe::content_root(
      host_owners.data(), sizeof(DirectRepresentationStateOwner) * host_owners.size());

  struct Combined {
    Root256 a, b, c;
  } combined{state_root, entries_root, owners_root};
  return direct_network::recipe::content_root(&combined, sizeof(combined));
}

void launch_direct_representation_compiler_step(DirectAdultRuntime* runtime,
                                                std::uint32_t frontier_work,
                                                std::uint32_t block_size) {
  if (runtime == nullptr || runtime->representation == nullptr)
    return;
  DirectRepresentationRuntime& rep = *runtime->representation;
  // Level A's claim/lifecycle kernels read this tick's frontier directly, so
  // they have nothing to do when frontier_work == 0 -- Level B's state-owner
  // step below still runs unconditionally (see its own header comment).
  if (frontier_work != 0u) {
    // gh #1331: grid covers frontier_capacity, not frontier_work -- the three
    // kernels below now self-clamp per-thread against the device-resident
    // *frontier_count alone (see their own comments), so the grid must be
    // wide enough to give every one of those threads a chance to exist.
    reset_representation_claims_kernel<<<grid_for(runtime->frontier_capacity, block_size),
                                         block_size, 0, runtime->stream>>>(
        runtime->frontier, runtime->frontier_count, rep.node_count, rep.view.claim_ordinal);
    arbitrate_representation_claims_kernel<<<grid_for(runtime->frontier_capacity, block_size),
                                             block_size, 0, runtime->stream>>>(
        runtime->frontier, runtime->frontier_count, rep.node_count, rep.view.claim_ordinal);
    advance_packed_cache_lifecycle_kernel<<<grid_for(runtime->frontier_capacity, block_size),
                                            block_size, 0, runtime->stream>>>(
        runtime->brain, runtime->frontier, runtime->frontier_count, rep.view,
        rep.resident_entries, rep.resident_entry_count);
    check_cuda(cudaGetLastError(), "launch representation packed-cache lifecycle step");
  }
  launch_direct_state_owner_step(runtime, frontier_work, runtime->next_eligibility_bank,
                                 runtime->next_eligibility_count, block_size);
}

void launch_resident_cache_probe(const DirectAdultRuntime& runtime, std::uint32_t source,
                                 std::uint32_t steps, std::uint32_t* out_checksum) {
  if (runtime.representation == nullptr || source >= runtime.representation->node_count)
    return;
  const DirectPackedEntry* entries =
      runtime.representation->resident_entries + static_cast<std::size_t>(source) * kRepresentationResidentReserve;
  const std::uint32_t* entry_count_ptr = runtime.representation->resident_entry_count + source;
  resident_cache_probe_kernel<<<grid_for(kRepresentationProbeLanes), 128>>>(
      entries, entry_count_ptr, kRepresentationProbeLanes, steps, out_checksum);
  check_cuda(cudaGetLastError(), "launch resident cache probe");
}

void launch_canonical_sparse_probe(const DirectAdultRuntime& runtime, std::uint32_t source,
                                   std::uint32_t steps, std::uint32_t* out_checksum) {
  canonical_sparse_probe_kernel<<<grid_for(kRepresentationProbeLanes), 128>>>(
      runtime.brain, source, kRepresentationProbeLanes, steps, out_checksum);
  check_cuda(cudaGetLastError(), "launch canonical sparse probe");
}

}  // namespace substrate::direct_adult
