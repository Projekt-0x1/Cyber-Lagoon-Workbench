#include <cub/cub.cuh>

#include <algorithm>
#include <vector>

#include "hardware_native/direct_canonical_transition_device.cuh"
#include "hardware_native/direct_exact_eligibility_device.cuh"
#include "hardware_native/direct_execution_fabric.cuh"
#include "hardware_native/direct_representation_compiler.cuh"

namespace substrate::direct_adult {

__device__ inline bool fabric_external_participation(const ActivityEvent& event) {
  return (event.origin == CausalOrigin::external_contact ||
          event.origin == CausalOrigin::motor_reafference) &&
         event.external_root != 0u;
}

__device__ inline void append_explicit_eligibility_from_transition(
    DirectBrainV01 brain, const ActivityEvent& event,
    const DirectCanonicalTransitionResult& transition,
    DirectEligibilityRecord* eligibility_bank, std::uint32_t* eligibility_count,
    std::uint32_t eligibility_capacity,
    std::uint32_t* eligibility_bucket_heads,
    std::uint32_t eligibility_bucket_count,
    const std::uint32_t* eligibility_batch_admit, std::uint32_t tick,
    AdultCounters* counters) {
  const std::uint32_t winner = transition.explicit_winner.route_slot;
  if (!fabric_external_participation(event) || winner == kInvalidIndex ||
      winner >= brain.route_capacity)
    return;

  const DirectRouteSlotMeta meta = brain.topology.slot_meta[winner];
  if (meta.live == 0u)
    return;

  DirectEligibilityRecord record{};
  record.route = DirectRouteHandle{winner, meta.generation};
  record.source = event.node;
  record.context = event.context;
  record.history_signature = transition.event_context_signature;
  // Same causal participation, same temporal prediction: the generic fallback
  // stamps the successor history on every record it mints, so a fabric adult
  // must not lose exact reafference settlement merely because a different
  // candidate-discovery backing executed.
  record.successor_history_signature = direct_canonical_successor_history_signature(
      event.history_signature, event.node, event.word);
  record.participation_root = event.external_root;
  record.ticket = event.external_root;
  record.strength_q16 = kEligibilityOneQ16;
  record.participation_tick = tick;
  record.horizon_class = eligibility_horizon_for_event(event);
  record.expiry_tick = tick + eligibility_lifetime(record.horizon_class);
  record.predicted_context = event.context;
  record.state = EligibilityRecordState::live;
  append_eligibility_record(
      brain, record, eligibility_bank, eligibility_count, eligibility_capacity,
      eligibility_bucket_heads, eligibility_bucket_count,
      eligibility_batch_admit, counters, true, /*charge_new=*/true);
}

__device__ inline void append_state_owner_implicit_eligibility_from_transition(
    DirectBrainV01 brain, const ActivityEvent& event,
    const DirectCanonicalTransitionResult& transition,
    const DirectRepresentationDeviceView& rep,
    DirectEligibilityRecord* eligibility_bank, std::uint32_t* eligibility_count,
    std::uint32_t eligibility_capacity,
    std::uint32_t* eligibility_bucket_heads,
    std::uint32_t eligibility_bucket_count,
    const std::uint32_t* eligibility_batch_admit, std::uint32_t tick,
    AdultCounters* counters) {
  if (!fabric_external_participation(event) ||
      transition.explicit_winner.route_slot != kInvalidIndex)
    return;

  for (std::uint32_t s = 0u; s < transition.successor_count; ++s) {
    const DirectCanonicalSuccessor& successor = transition.successors[s];
    if (successor.kind != DirectCanonicalSuccessorKind::implicit_virtual)
      continue;

    const DirectLogicalInteractionId logical_id =
        derive_implicit_logical_id(event.node, successor.implicit_family,
                                   successor.implicit_slot);
    const std::uint32_t owner_slot = find_state_owner(rep, logical_id);
    if (owner_slot == kInvalidIndex)
      continue;
    const DirectRepresentationStateOwner& owner = rep.state_owners[owner_slot];
    if (owner.lifecycle != DirectStateOwnerLifecycle::owned)
      continue;

    DirectEligibilityRecord record{};
    record.logical_id = logical_id;
    record.locator.kind = DirectLocatorKind::implicit_virtual;
    record.locator.slot = owner_slot;
    record.locator.generation = owner.owner_epoch;
    record.route = DirectRouteHandle{kInvalidIndex, 0u};
    record.source = event.node;
    record.context = event.context;
    record.history_signature = transition.event_context_signature;
    record.successor_history_signature = direct_canonical_successor_history_signature(
        event.history_signature, event.node, event.word);
    record.participation_root = event.external_root;
    record.ticket = event.external_root;
    record.strength_q16 = kEligibilityOneQ16;
    record.participation_tick = tick;
    record.horizon_class = eligibility_horizon_for_event(event);
    record.expiry_tick = tick + eligibility_lifetime(record.horizon_class);
    record.predicted_context = event.context;
    record.state = EligibilityRecordState::live;
    append_eligibility_record(
        brain, record, eligibility_bank, eligibility_count,
        eligibility_capacity, eligibility_bucket_heads,
        eligibility_bucket_count, eligibility_batch_admit, counters, true,
        /*charge_new=*/true);
  }
}

__global__ void heterogeneous_dispatch_kernel(
    DirectBrainV01 brain, DirectTopologyDeviceView topology,
    DirectExecutionFabricDeviceView fabric,
    const ActivityEvent* frontier, const std::uint32_t* frontier_count,
    std::uint32_t prepass_work, std::uint32_t frontier_capacity,
    std::uint32_t tick, ActivityEvent* staged_events,
    std::uint32_t* staged_event_valid, MotorEvent* motor_events,
    std::uint32_t* motor_count, std::uint32_t motor_capacity,
    DirectEligibilityRecord* eligibility_bank,
    std::uint32_t* eligibility_count, std::uint32_t eligibility_capacity,
    std::uint32_t* eligibility_bucket_heads,
    std::uint32_t eligibility_bucket_count,
    const std::uint32_t* eligibility_batch_admit,
    DirectRepresentationDeviceView rep, AdultCounters* counters) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= frontier_capacity)
    return;

  const std::uint32_t stage_base = i * kCanonicalSuccessorCapacity;
  for (std::uint32_t s = 0u; s < kCanonicalSuccessorCapacity; ++s)
    staged_event_valid[stage_base + s] = 0u;

  // Match the canonical sparse kernel exactly: on every busy adult step,
  // thread zero advances developmental chronology before discovering whether
  // the resident frontier itself is empty. A backend may not change ontogeny
  // merely because host frontier_launch_bound lagged the device frontier.
  if (i == 0u && brain.development != nullptr) {
    ResidentDevelopmentState& development = *brain.development;
    ++development.developmental_tick;
    if (development.developmental_tick < development.maturation_tick &&
        development.plasticity_q16 > development.mature_plasticity_floor_q16) {
      const std::uint32_t remaining =
          development.plasticity_q16 - development.mature_plasticity_floor_q16;
      const std::uint32_t ticks_left =
          development.maturation_tick - development.developmental_tick + 1u;
      const std::uint32_t decrement =
          remaining / (ticks_left == 0u ? 1u : ticks_left);
      development.plasticity_q16 -= decrement == 0u ? 1u : decrement;
    }
  }

  // #1304 fixed this birth defect in canonical sparse: the host launch bound
  // trails the resident frontier after fanout. Heterogeneous execution obeys
  // the same device-live law. *frontier_count is an attempt counter and may
  // exceed the allocation, hence the independent frontier_capacity bound.
  if (i >= *frontier_count)
    return;

  const ActivityEvent event = frontier[i];
  if (event.node >= brain.node_count)
    return;

  DirectNode& source = brain.nodes[event.node];
  if (is_independent_external(event.origin))
    atomicAdd(&source.external_contacts, 1u);
  else
    atomicAdd(&source.endogenous_visits, 1u);

  const DirectExecutionMembership membership = fabric.node_memberships[event.node];

  DirectCanonicalTransitionResult transition{};
  bool specialized_candidate_discovery = false;
  if (membership.phenotype ==
      static_cast<std::uint32_t>(DirectExecutionPhenotype::packed_sparse)) {
    specialized_candidate_discovery = evaluate_packed_sparse_source(
        brain, fabric, membership.owner_index, event, &transition, counters);
  } else if (membership.phenotype ==
             static_cast<std::uint32_t>(DirectExecutionPhenotype::dense_tensor)) {
    // Dense tiles remain numeric execution primitives until a logical
    // participant mapping exists. Do not let a physical tile author a second
    // causal law merely because an install helper was called.
    if (counters != nullptr)
      atomicAdd(&counters->guard_fallbacks, 1u);
  }

  if (!specialized_candidate_discovery)
    transition = direct_canonical_evaluate_source_transition(brain, event);

  // The historical fabric eligibility pre-pass already calls the same Rung-1
  // canonical winner helper, so it is decision-equivalent, but it is still
  // host-bound and skips several cases. Complete exactly the holes here from
  // this transition result; no backend may invent a different participant.
  const bool prepass_covered = i < prepass_work;
  const bool prepass_skips_source_output =
      source.output_word != 0u && (source.flags & kNodeFlagSensor) == 0u;
  const bool prepass_skips_dense =
      membership.phenotype ==
      static_cast<std::uint32_t>(DirectExecutionPhenotype::dense_tensor);
  bool prepass_skips_for_implicit = false;
  if (direct_canonical_implicit_mesh_eligible(transition.explicit_winner,
                                              source.route_count)) {
    const DirectImplicitCandidate prepass_candidate =
        select_direct_implicit_candidate(brain, event.node,
                                         transition.event_context_signature);
    prepass_skips_for_implicit = direct_canonical_implicit_wins(
        prepass_candidate, transition.explicit_winner, source.route_count);
  }
  if (!prepass_covered || prepass_skips_source_output || prepass_skips_dense ||
      prepass_skips_for_implicit) {
    append_explicit_eligibility_from_transition(
        brain, event, transition, eligibility_bank, eligibility_count,
        eligibility_capacity, eligibility_bucket_heads,
        eligibility_bucket_count, eligibility_batch_admit, tick, counters);
  }

  append_state_owner_implicit_eligibility_from_transition(
      brain, event, transition, rep, eligibility_bank, eligibility_count,
      eligibility_capacity, eligibility_bucket_heads,
      eligibility_bucket_count, eligibility_batch_admit, tick, counters);

  std::uint32_t implicit_ordinal = 0u;
  for (std::uint32_t s = 0u; s < transition.successor_count; ++s) {
    const DirectCanonicalSuccessor successor = transition.successors[s];

    if (successor.kind == DirectCanonicalSuccessorKind::implicit_virtual) {
      if (counters != nullptr)
        atomicAdd(&counters->virtual_participations, 1u);

      if (successor.participating != 0u) {
        if (source.route_count != 0u && counters != nullptr)
          atomicAdd(&counters->implicit_wins, 1u);
        const DirectImplicitParticipationResult participation =
            record_direct_implicit_participation(
                brain, successor.implicit_family, event.node,
                successor.implicit_slot, tick, event.external_root,
                fabric_external_participation(event));
        if (participation.should_materialize && topology.proposals != nullptr) {
          submit_direct_growth_proposal(
              topology, i * kMaxImplicitActiveFanout + implicit_ordinal,
              event.node, successor.effect.successor.node,
              transition.event_context_signature, successor.conductance_q16,
              0, successor.delay, 0u, successor.conductance_q16,
              successor.implicit_family, successor.implicit_slot, event.context,
              tick + eligibility_lifetime(eligibility_horizon_for_event(event)),
              event.context, transition.event_context_signature,
              event.external_root);
        }
      }
      ++implicit_ordinal;
    }

    staged_events[stage_base + s] = successor.effect.successor;
    staged_event_valid[stage_base + s] = 1u;
    if (counters != nullptr)
      atomicAdd(&counters->propagated, 1u);

    if (successor.effect.motor_valid) {
      const std::uint32_t motor_slot = atomicAdd(motor_count, 1u);
      if (motor_slot < motor_capacity)
        motor_events[motor_slot] = successor.effect.motor;
      if (counters != nullptr)
        atomicAdd(&counters->motor_events, 1u);
    }
  }
}

__global__ void commit_compacted_staged_events_kernel(
    const ActivityEvent* staged_events, const std::uint32_t* staged_valid,
    const std::uint32_t* staged_ranks, std::uint32_t total_work,
    ActivityEvent* next_frontier,
    DirectIngressAuthority* next_frontier_authority,
    std::uint32_t* next_frontier_count, std::uint32_t frontier_capacity) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= total_work || staged_valid[i] == 0u)
    return;
  const std::uint32_t rank = staged_ranks[i];
  if (rank < frontier_capacity) {
    next_frontier[rank] = staged_events[i];
    next_frontier_authority[rank] = DirectIngressAuthority::ordinary;
  }
  if (i == total_work - 1u)
    *next_frontier_count = staged_ranks[i] + staged_valid[i];
}

__global__ void update_frontier_count_kernel(
    const std::uint32_t* staged_valid, const std::uint32_t* staged_ranks,
    std::uint32_t total_work, std::uint32_t* next_frontier_count) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    if (total_work == 0u)
      *next_frontier_count = 0u;
    else
      *next_frontier_count = staged_ranks[total_work - 1u] +
                             staged_valid[total_work - 1u];
  }
}

DirectExecutionFabricRuntime* create_direct_execution_fabric(
    std::uint32_t node_count, std::uint32_t route_capacity,
    std::uint32_t frontier_capacity, std::uint32_t max_tract_delay) {
  auto* fabric = new DirectExecutionFabricRuntime{};
  fabric->node_count = node_count;
  fabric->route_capacity = route_capacity;
  fabric->frontier_capacity = frontier_capacity;
  fabric->max_tract_delay = max_tract_delay;
  fabric->tract_ring_ever_written = false;
  fabric->packed_panel_capacity = 1024u;
  fabric->packed_entry_capacity = std::max(route_capacity, 8192u);
  fabric->dense_tile_capacity = 1024u;
  fabric->tract_lane_capacity = 1024u;

  cudaMalloc(&fabric->view.node_memberships,
             sizeof(DirectExecutionMembership) * node_count);
  cudaMemset(fabric->view.node_memberships, 0,
             sizeof(DirectExecutionMembership) * node_count);
  cudaMalloc(&fabric->view.route_memberships,
             sizeof(DirectExecutionMembership) * route_capacity);
  cudaMemset(fabric->view.route_memberships, 0,
             sizeof(DirectExecutionMembership) * route_capacity);
  cudaMalloc(&fabric->view.packed_panels,
             sizeof(DirectPackedSparsePanel) * fabric->packed_panel_capacity);
  cudaMalloc(&fabric->view.packed_source_meta,
             sizeof(DirectPackedSourceMeta) * node_count);
  cudaMemset(fabric->view.packed_source_meta, 0,
             sizeof(DirectPackedSourceMeta) * node_count);
  cudaMalloc(&fabric->view.packed_entries,
             sizeof(DirectPackedEntry) * fabric->packed_entry_capacity);
  cudaMalloc(&fabric->view.dense_tiles,
             sizeof(DirectDenseTile) * fabric->dense_tile_capacity);
  cudaMalloc(&fabric->view.tract_lanes,
             sizeof(DirectTractLane) * fabric->tract_lane_capacity);

  const std::uint32_t ring_buckets = max_tract_delay + 1u;
  const std::uint32_t ring_stride = std::max(node_count, frontier_capacity);
  const std::uint32_t ring_capacity = ring_buckets * ring_stride;
  cudaMalloc(&fabric->view.tract_ring_packets,
             sizeof(DirectTractPacket) * ring_capacity);
  cudaMemset(fabric->view.tract_ring_packets, 0,
             sizeof(DirectTractPacket) * ring_capacity);
  cudaMalloc(&fabric->view.tract_bucket_counts,
             sizeof(std::uint32_t) * ring_buckets);
  cudaMemset(fabric->view.tract_bucket_counts, 0,
             sizeof(std::uint32_t) * ring_buckets);

  const std::uint32_t max_staging =
      frontier_capacity * kCanonicalSuccessorCapacity + ring_stride;
  cudaMalloc(&fabric->view.staged_events, sizeof(ActivityEvent) * max_staging);
  cudaMalloc(&fabric->view.staged_event_valid,
             sizeof(std::uint32_t) * max_staging);
  cudaMalloc(&fabric->view.staged_event_ranks,
             sizeof(std::uint32_t) * max_staging);
  cudaMalloc(&fabric->view.staged_event_total, sizeof(std::uint32_t));
  fabric->view.max_tract_delay = max_tract_delay;

  cub::DeviceScan::ExclusiveSum(nullptr, fabric->scan_storage_bytes,
                                fabric->view.staged_event_valid,
                                fabric->view.staged_event_ranks, max_staging);
  cudaMalloc(&fabric->scan_storage, fabric->scan_storage_bytes);
  cudaMalloc(&fabric->tract_sort_keys_in, sizeof(std::uint64_t) * ring_stride);
  cudaMalloc(&fabric->tract_sort_keys_out, sizeof(std::uint64_t) * ring_stride);
  cudaMalloc(&fabric->tract_sort_values_in, sizeof(std::uint32_t) * ring_stride);
  cudaMalloc(&fabric->tract_sort_values_out, sizeof(std::uint32_t) * ring_stride);
  cub::DeviceRadixSort::SortPairs(nullptr, fabric->tract_sort_storage_bytes,
                                  fabric->tract_sort_keys_in,
                                  fabric->tract_sort_keys_out,
                                  fabric->tract_sort_values_in,
                                  fabric->tract_sort_values_out, ring_stride);
  cudaMalloc(&fabric->tract_sort_storage, fabric->tract_sort_storage_bytes);
  return fabric;
}

void destroy_direct_execution_fabric(DirectExecutionFabricRuntime* fabric) {
  if (fabric == nullptr)
    return;
  cudaFree(fabric->view.node_memberships);
  cudaFree(fabric->view.route_memberships);
  cudaFree(fabric->view.packed_panels);
  cudaFree(fabric->view.packed_source_meta);
  cudaFree(fabric->view.packed_entries);
  cudaFree(fabric->view.dense_tiles);
  cudaFree(fabric->view.tract_lanes);
  cudaFree(fabric->view.tract_ring_packets);
  cudaFree(fabric->view.tract_bucket_counts);
  cudaFree(fabric->view.staged_events);
  cudaFree(fabric->view.staged_event_valid);
  cudaFree(fabric->view.staged_event_ranks);
  cudaFree(fabric->view.staged_event_total);
  cudaFree(fabric->scan_storage);
  cudaFree(fabric->tract_sort_keys_in);
  cudaFree(fabric->tract_sort_keys_out);
  cudaFree(fabric->tract_sort_values_in);
  cudaFree(fabric->tract_sort_values_out);
  cudaFree(fabric->tract_sort_storage);
  delete fabric;
}

bool install_direct_packed_sparse_panel(
    DirectBrainV01* brain, DirectExecutionFabricRuntime* fabric,
    std::uint32_t source_begin, std::uint32_t source_count) {
  if (brain == nullptr || fabric == nullptr ||
      source_begin + source_count > brain->node_count)
    return false;

  std::vector<DirectPackedEntry> host_entries;
  std::vector<DirectPackedSourceMeta> host_meta(source_count);
  std::vector<DirectExecutionMembership> host_mem(source_count);
  std::vector<DirectNode> host_nodes(source_count);
  const std::uint32_t panel_idx = fabric->packed_panel_count++;

  cudaMemcpy(host_nodes.data(), &brain->nodes[source_begin],
             sizeof(DirectNode) * source_count, cudaMemcpyDeviceToHost);

  for (std::uint32_t s = 0; s < source_count; ++s) {
    const DirectNode& node = host_nodes[s];
    host_meta[s].entry_offset =
        fabric->packed_entry_count + static_cast<std::uint32_t>(host_entries.size());
    host_meta[s].entry_count = node.route_count;
    host_meta[s].source_revision = node.source_revision;
    host_meta[s].flags = 0u;

    std::uint32_t slot = node.first_route;
    for (std::uint32_t r = 0; r < node.route_count && slot != kInvalidIndex; ++r) {
      DirectRoute route{};
      cudaMemcpy(&route, &brain->routes[slot], sizeof(DirectRoute),
                 cudaMemcpyDeviceToHost);
      DirectRouteSlotMeta meta{};
      cudaMemcpy(&meta, &brain->topology.slot_meta[slot], sizeof(DirectRouteSlotMeta),
                 cudaMemcpyDeviceToHost);
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
      host_entries.push_back(entry);
      slot = route.next_route;
    }

    host_mem[s].phenotype =
        static_cast<std::uint32_t>(DirectExecutionPhenotype::packed_sparse);
    host_mem[s].owner_index = panel_idx;
    host_mem[s].local_slot = s;
    host_mem[s].reserved = 0u;
    host_mem[s].generation = ++fabric->generation_epoch;
  }

  if (!host_entries.empty()) {
    cudaMemcpy(&fabric->view.packed_entries[fabric->packed_entry_count],
               host_entries.data(), sizeof(DirectPackedEntry) * host_entries.size(),
               cudaMemcpyHostToDevice);
    fabric->packed_entry_count += static_cast<std::uint32_t>(host_entries.size());
  }
  cudaMemcpy(&fabric->view.packed_source_meta[source_begin], host_meta.data(),
             sizeof(DirectPackedSourceMeta) * source_count, cudaMemcpyHostToDevice);
  cudaMemcpy(&fabric->view.node_memberships[source_begin], host_mem.data(),
             sizeof(DirectExecutionMembership) * source_count, cudaMemcpyHostToDevice);

  DirectPackedSparsePanel panel{};
  panel.generation = fabric->generation_epoch;
  panel.source_begin = source_begin;
  panel.source_count = source_count;
  panel.entry_begin =
      fabric->packed_entry_count - static_cast<std::uint32_t>(host_entries.size());
  panel.entry_count = static_cast<std::uint32_t>(host_entries.size());
  panel.revision_root = 0u;
  panel.flags = 0u;
  cudaMemcpy(&fabric->view.packed_panels[panel_idx], &panel,
             sizeof(DirectPackedSparsePanel), cudaMemcpyHostToDevice);
  return true;
}

bool install_direct_dense_tile(DirectBrainV01* brain,
                               DirectExecutionFabricRuntime* fabric,
                               const DirectDenseTile& tile) {
  if (brain == nullptr || fabric == nullptr ||
      fabric->dense_tile_count >= fabric->dense_tile_capacity)
    return false;
  const std::uint32_t tile_idx = fabric->dense_tile_count++;
  DirectDenseTile stored = tile;
  stored.state_flags |= kExecutionFlagFallbackGuard;
  cudaMemcpy(&fabric->view.dense_tiles[tile_idx], &stored,
             sizeof(DirectDenseTile), cudaMemcpyHostToDevice);
  ++fabric->generation_epoch;
  return true;
}

bool install_direct_tract_lane(DirectBrainV01* brain,
                               DirectExecutionFabricRuntime* fabric,
                               std::uint32_t source, std::uint32_t target,
                               std::uint32_t delay) {
  if (brain == nullptr || fabric == nullptr || source >= brain->node_count ||
      fabric->tract_lane_count >= fabric->tract_lane_capacity)
    return false;

  DirectNode node{};
  cudaMemcpy(&node, &brain->nodes[source], sizeof(DirectNode), cudaMemcpyDeviceToHost);
  bool found = false;
  std::uint32_t slot = node.first_route;
  for (std::uint32_t r = 0; r < node.route_count && slot != kInvalidIndex; ++r) {
    DirectRoute route{};
    cudaMemcpy(&route, &brain->routes[slot], sizeof(DirectRoute),
               cudaMemcpyDeviceToHost);
    if (route.target == target) {
      if (route.delay != delay)
        return false;
      found = true;
      break;
    }
    slot = route.next_route;
  }
  if (!found)
    return false;

  // Conservative: a lane is what a device-side producer would route through.
  fabric->tract_ring_ever_written = true;
  const std::uint32_t lane_idx = fabric->tract_lane_count++;
  DirectTractLane lane{};
  lane.generation = ++fabric->generation_epoch;
  lane.source_node = source;
  lane.target_node = target;
  lane.delay = delay;
  lane.flags = kExecutionFlagFallbackGuard;
  cudaMemcpy(&fabric->view.tract_lanes[lane_idx], &lane,
             sizeof(DirectTractLane), cudaMemcpyHostToDevice);
  return true;
}

void launch_direct_heterogeneous_frontier_step(DirectAdultRuntime* runtime,
                                               std::uint32_t frontier_work) {
  if (runtime == nullptr || runtime->fabric == nullptr)
    return;

  DirectExecutionFabricRuntime* fabric = runtime->fabric;
  const std::uint32_t block_size = 128u;
  const DirectRepresentationDeviceView rep =
      runtime->representation != nullptr ? runtime->representation->view
                                         : DirectRepresentationDeviceView{};

  // Canonical sparse launches at the allocation and self-bounds on
  // *frontier_count (#1304). Do the same here. frontier_work is only the
  // historical eligibility-prepass coverage bound, never the execution bound.
  if (runtime->frontier_capacity > 0u) {
    const std::uint32_t grid =
        (runtime->frontier_capacity + block_size - 1u) / block_size;
    heterogeneous_dispatch_kernel<<<grid, block_size>>>(
        runtime->brain, runtime->topology_runtime->view, fabric->view,
        runtime->frontier, runtime->frontier_count, frontier_work,
        runtime->frontier_capacity, runtime->tick, fabric->view.staged_events,
        fabric->view.staged_event_valid, runtime->motor_events,
        runtime->motor_count, runtime->motor_capacity,
        runtime->next_eligibility_bank, runtime->next_eligibility_count,
        runtime->eligibility_capacity, runtime->next_eligibility_bucket_heads,
        runtime->eligibility_bucket_count, runtime->eligibility_batch_admit,
        rep, runtime->counters);
  }

  const std::uint32_t frontier_stage_span =
      runtime->frontier_capacity * kCanonicalSuccessorCapacity;
  std::uint32_t total_staged = frontier_stage_span;

  // #1208: reading the due bucket's count costs a BLOCKING device->host copy,
  // and it ran on every step of every fabric adult -- a host roundtrip inside
  // the step, which is the one thing the resident adult may not have. CUB needs
  // `num_items` on the host, so the read cannot simply move to the device; what
  // it can do is not happen at all when the ring is provably empty. Since
  // `cd3df9c933` removed the in-kernel producer the ring has exactly one writer
  // (`enqueue_direct_tract_packet`) and one filler (checkpoint restore), both on
  // the host, so the host already knows whether anything could be in there. The
  // flag is sticky: it over-approximates, never under-approximates, so no packet
  // can be skipped -- an adult that has touched a tract pays exactly the cost it
  // paid before.
  std::uint32_t due_bucket_count = 0u;
  const std::uint32_t due_bucket =
      runtime->tick % (fabric->max_tract_delay + 1u);
  if (fabric->tract_ring_ever_written) {
    cudaMemcpy(&due_bucket_count, &fabric->view.tract_bucket_counts[due_bucket],
               sizeof(std::uint32_t), cudaMemcpyDeviceToHost);
  }

  if (due_bucket_count > 0u) {
    const std::uint32_t tract_grid =
        (due_bucket_count + block_size - 1u) / block_size;
    const std::uint32_t stride = runtime->brain.node_count;
    const DirectTractPacket* bucket_packets =
        &fabric->view.tract_ring_packets[due_bucket * stride];
    gather_tract_sort_keys_kernel<<<tract_grid, block_size>>>(
        bucket_packets, due_bucket_count, fabric->tract_sort_keys_in,
        fabric->tract_sort_values_in);
    cub::DeviceRadixSort::SortPairs(
        fabric->tract_sort_storage, fabric->tract_sort_storage_bytes,
        fabric->tract_sort_keys_in, fabric->tract_sort_keys_out,
        fabric->tract_sort_values_in, fabric->tract_sort_values_out,
        due_bucket_count);
    deliver_sorted_tract_packets_kernel<<<tract_grid, block_size>>>(
        bucket_packets, fabric->tract_sort_values_out, due_bucket_count,
        fabric->view.staged_events, fabric->view.staged_event_valid,
        frontier_stage_span, runtime->counters);
    clear_tract_bucket_count_kernel<<<1, 32>>>(fabric->view.tract_bucket_counts,
                                               due_bucket);
    total_staged += due_bucket_count;
  }

  if (total_staged > 0u) {
    const std::uint32_t merge_grid =
        (total_staged + block_size - 1u) / block_size;
    cub::DeviceScan::ExclusiveSum(
        fabric->scan_storage, fabric->scan_storage_bytes,
        fabric->view.staged_event_valid, fabric->view.staged_event_ranks,
        total_staged);
    commit_compacted_staged_events_kernel<<<merge_grid, block_size>>>(
        fabric->view.staged_events, fabric->view.staged_event_valid,
        fabric->view.staged_event_ranks, total_staged, runtime->next_frontier,
        runtime->next_frontier_authority, runtime->next_frontier_count,
        runtime->frontier_capacity);
    update_frontier_count_kernel<<<1, 32>>>(
        fabric->view.staged_event_valid, fabric->view.staged_event_ranks,
        total_staged, runtime->next_frontier_count);
  }
}

}  // namespace substrate::direct_adult
