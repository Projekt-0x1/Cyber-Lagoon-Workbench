// Included inside direct_network_resident_development.cu's anonymous namespace
// after the causal-memory helpers: owns the exact-history maintenance sweeps
// (scarcity retraction and homeostatic weight decay) behind
// launch_exact_history_maintenance.
__device__ bool base_maintenance_candidate(
    DirectBrain brain, const std::uint64_t* opportunity_incarnations,
    const std::uint32_t* causal_pins, std::uint32_t node_index,
    std::uint32_t route_index) {
  const DirectNode& node = brain.nodes[node_index];
  const DirectRoute& route = brain.routes[route_index];
  return (route.flags & kRouteFlagActive) != 0u &&
         brain.route_incarnations != nullptr && opportunity_incarnations != nullptr &&
         opportunity_incarnations[route_index] == brain.route_incarnations[route_index] &&
         direct_adult_core::maintenance_abs_q16(node.credit_ema_q16) <
             (direct_adult_core::kQ16One / 64) &&
         direct_adult_core::maintenance_abs_q16(node.activity_ema_q16) <
             (direct_adult_core::kQ16One / 128) &&
         (causal_pins == nullptr ||
          ((causal_pins[route_index >> 5] >> (route_index & 31u)) & 1u) == 0u) &&
         route.conductance_q16 <= direct_adult_core::kMinConductanceQ16;
}

__device__ bool maintenance_candidate(
    DirectBrain brain, const std::uint64_t* opportunity_incarnations,
    const std::uint32_t* causal_pins, std::uint32_t node_index,
    std::uint32_t route_index) {
  if (!base_maintenance_candidate(brain, opportunity_incarnations, causal_pins,
                                  node_index, route_index)) return false;
  if (!route_has_causal_memory(brain, route_index)) return true;
  const DirectNode& node = brain.nodes[node_index];
  const std::uint64_t tick = brain.development == nullptr
                                 ? 0u : brain.development->age_tick;
  for (std::uint32_t other = node.route_offset;
       other < node.route_offset + node.route_capacity; ++other) {
    if (other == route_index ||
        !base_maintenance_candidate(brain, opportunity_incarnations, causal_pins,
                                    node_index, other)) continue;
    if (!route_has_causal_memory(brain, other) ||
        causal_route_more_expendable(brain, other, route_index, tick)) return false;
  }
  return true;
}

__device__ void plan_maintenance_transaction_impl(
    DirectBrain brain, const std::uint64_t* opportunity_incarnations,
    const std::uint32_t* causal_pins, std::int32_t cost_per_route_q16,
    std::uint32_t* candidate_counts, TopologyCommitPlan* plan,
    direct_adult_core::AdultCoreMetrics* metrics, std::uint32_t node_index) {
  if (node_index >= brain.node_count) return;
  const DirectNode& node = brain.nodes[node_index];
  std::uint32_t count = 0u;
  const bool pressure = direct_adult::device_explicit_interaction_pressure(
      brain.resource_ecology);
  const bool weak = direct_adult_core::maintenance_abs_q16(node.credit_ema_q16) <
                        (direct_adult_core::kQ16One / 64) &&
                    direct_adult_core::maintenance_abs_q16(node.activity_ema_q16) <
                        (direct_adult_core::kQ16One / 128);
  for (std::uint32_t route_index = node.route_offset;
       route_index < node.route_offset + node.route_capacity; ++route_index) {
    const DirectRoute& route = brain.routes[route_index];
    if ((route.flags & kRouteFlagActive) == 0u) continue;
    atomicAdd(reinterpret_cast<unsigned long long*>(
                  &metrics->maintenance_energy_debited),
              static_cast<unsigned long long>(cost_per_route_q16));
    if (!pressure || !weak || brain.route_incarnations == nullptr ||
        opportunity_incarnations == nullptr ||
        opportunity_incarnations[route_index] !=
            brain.route_incarnations[route_index]) continue;
    if (causal_pins != nullptr &&
        ((causal_pins[route_index >> 5] >> (route_index & 31u)) & 1u) != 0u) {
      direct_adult::device_defer_pool_units(
          brain.resource_ecology,
          direct_adult::DirectResourcePoolKind::explicit_interaction,
          decode_route_construction_cost(route.flags));
      atomicAdd(reinterpret_cast<unsigned long long*>(
                    &metrics->retractions_deferred_causal_pin), 1ULL);
      continue;
    }
    count += route.conductance_q16 <= direct_adult_core::kMinConductanceQ16;
  }
  candidate_counts[node_index] = count;
  if (count != 0u && plan != nullptr) atomicAdd(&plan->proposal_count, count);
}

__global__ void plan_maintenance_transaction_kernel(
    DirectBrain brain, const std::uint64_t* opportunity_incarnations,
    const std::uint32_t* causal_pins, std::int32_t cost_per_route_q16,
    std::uint32_t* candidate_counts, TopologyCommitPlan* plan,
    direct_adult_core::AdultCoreMetrics* metrics) {
  plan_maintenance_transaction_impl(
      brain, opportunity_incarnations, causal_pins, cost_per_route_q16,
      candidate_counts, plan, metrics, blockIdx.x * blockDim.x + threadIdx.x);
}

__device__ void begin_maintenance_transaction_impl(
    DirectBrain brain, TopologyCommitPlan* plan,
    std::uint32_t current_tick) {
  if (plan == nullptr || brain.development == nullptr) return;
  plan->admitted = 0u;
  const auto& history = brain.development->exact_history;
  if (history.sealed != 0u ||
      history.phase_kind != DirectExactHistoryKind::empty ||
      history.committed_slots >= kDirectExactHistoryHotPageCapacity) return;
  const std::uint32_t history_room =
      kDirectExactHistoryHotPageCapacity - history.committed_slots;
  auto* pool = direct_adult::direct_ecology_pool(
      brain.resource_ecology,
      direct_adult::DirectResourcePoolKind::explicit_interaction);
  if (pool == nullptr || pool->capacity_units < pool->charged_units ||
      pool->reserved_units > pool->charged_units) return;
  const std::uint64_t available =
      pool->capacity_units - pool->charged_units + pool->reserved_units;
  const std::uint64_t required = pool->capacity_units / 10u + 1u;
  const std::uint64_t needed = required > available ? required - available : 0u;
  plan->pressure_retraction_count = needed != 0u ? 1u : 0u;
  std::uint64_t accepted = plan->proposal_count < needed
                               ? plan->proposal_count
                               : needed;
  if (accepted > pool->live_units) accepted = pool->live_units;
  const std::uint64_t uncommit_capacity =
      pool->charged_units - pool->reserved_units;
  if (accepted > uncommit_capacity) accepted = uncommit_capacity;
  if (accepted > history_room) accepted = history_room;
  plan->accepted_count = static_cast<std::uint32_t>(accepted);
  plan->retraction_count = plan->accepted_count;
  if (accepted == 0u || !begin_exact_history_phase(
          &brain.development->exact_history,
          DirectExactHistoryKind::topology_retraction,
          plan->accepted_count, current_tick)) return;
  plan->admitted = 1u;
}

__global__ void begin_maintenance_transaction_kernel(
    DirectBrain brain, TopologyCommitPlan* plan, std::uint32_t current_tick) {
  if (blockIdx.x == 0u && threadIdx.x == 0u)
    begin_maintenance_transaction_impl(brain, plan, current_tick);
}

__device__ void commit_maintenance_transaction_impl(
    DirectBrain brain, const std::uint64_t* opportunity_incarnations,
    const std::uint32_t* causal_pins, const std::uint32_t* candidate_prefix,
    const TopologyCommitPlan* plan, direct_adult_core::AdultCoreMetrics* metrics,
    std::uint32_t node_index) {
  if (node_index >= brain.node_count || plan == nullptr || plan->admitted == 0u) return;
  DirectNode& node = brain.nodes[node_index];
  std::uint32_t local = 0u;
  for (std::uint32_t route_index = node.route_offset;
       route_index < node.route_offset + node.route_capacity; ++route_index) {
    if (!maintenance_candidate(brain, opportunity_incarnations, causal_pins,
                               node_index, route_index)) continue;
    const std::uint32_t ordinal = candidate_prefix[node_index] + local++;
    if (ordinal >= plan->accepted_count) continue;
    DirectRoute& route = brain.routes[route_index];
    const std::uint64_t before = brain.route_incarnations[route_index];
    DirectExactHistoryRecord* record =
        &brain.development->exact_history.records[
            brain.development->exact_history.phase_base + ordinal];
    const std::uint32_t route_cost = decode_route_construction_cost(route.flags);
    stage_topology_history_record(
        record, DirectExactHistoryKind::topology_retraction,
        brain.development->exact_history.phase_tick, node_index, route_index, route.target,
        route.flags, before,
        before + kRouteIncarnationStride, -static_cast<std::int64_t>(route_cost),
        kDirectTopologyHistoryMaintenance);
    __threadfence();
    if (!direct_adult::device_uncommit_pool_units(
            brain.resource_ecology,
            direct_adult::DirectResourcePoolKind::explicit_interaction,
            route_cost)) {
      *record = DirectExactHistoryRecord{};
      continue;
    }
    route.flags &= ~kRouteFlagActive;
    brain.route_incarnations[route_index] = before + kRouteIncarnationStride;
    if (route.target < brain.node_count)
      atomicSub(&brain.nodes[route.target].active_in_degree, 1u);
    if (node.active_route_count > 0u) --node.active_route_count;
    atomicAdd(reinterpret_cast<unsigned long long*>(&brain.development->constructor_reserve), route_cost);
    atomicAdd(reinterpret_cast<unsigned long long*>(&brain.development->reclaimed_resource), route_cost);
    atomicAdd(reinterpret_cast<unsigned long long*>(
                  &brain.development->live_route_matter),
              static_cast<unsigned long long>(-static_cast<long long>(route_cost)));
    atomicAdd(reinterpret_cast<unsigned long long*>(
                  &metrics->routes_retracted_scarcity), 1ULL);
  }
}

__global__ void commit_maintenance_transaction_kernel(
    DirectBrain brain, const std::uint64_t* opportunity_incarnations,
    const std::uint32_t* causal_pins, const std::uint32_t* candidate_prefix,
    const TopologyCommitPlan* plan, direct_adult_core::AdultCoreMetrics* metrics) {
  commit_maintenance_transaction_impl(
      brain, opportunity_incarnations, causal_pins, candidate_prefix, plan,
      metrics, blockIdx.x * blockDim.x + threadIdx.x);
}

__device__ std::int32_t maintenance_decayed_conductance(
    const DirectRoute& route, std::int32_t cost_per_route_q16) {
  const std::int32_t decayed = route.conductance_q16 - cost_per_route_q16 * 2;
  return decayed < direct_adult_core::kMinConductanceQ16
             ? direct_adult_core::kMinConductanceQ16
             : decayed;
}

__device__ bool homeostatic_weight_candidate(
    DirectBrain brain, const std::uint64_t* opportunity_incarnations,
    const std::uint32_t* causal_pins, std::int32_t cost_per_route_q16,
    const TopologyCommitPlan* plan, std::uint32_t node_index,
    std::uint32_t route_index) {
  if (plan == nullptr || brain.development == nullptr) return false;
  (void)opportunity_incarnations;
  const DirectNode& node = brain.nodes[node_index];
  const DirectRoute& route = brain.routes[route_index];
  if (direct_adult_core::maintenance_abs_q16(node.credit_ema_q16) >=
          (direct_adult_core::kQ16One / 64) ||
      direct_adult_core::maintenance_abs_q16(node.activity_ema_q16) >=
          (direct_adult_core::kQ16One / 128) ||
      (route.flags & kRouteFlagActive) == 0u || brain.route_incarnations == nullptr ||
      route.eligibility_expires >= brain.development->age_tick ||
      (causal_pins != nullptr &&
       ((causal_pins[route_index >> 5] >> (route_index & 31u)) & 1u) != 0u) ||
      route.conductance_q16 <= direct_adult_core::kMinConductanceQ16) return false;
  return maintenance_decayed_conductance(route, cost_per_route_q16) != route.conductance_q16;
}

__device__ void plan_homeostatic_weight_transaction_impl(
    DirectBrain brain, const std::uint64_t* opportunity_incarnations,
    const std::uint32_t* causal_pins, std::int32_t cost_per_route_q16,
    const TopologyCommitPlan* plan, std::uint32_t* candidate_counts,
    std::uint32_t node_index) {
  if (node_index >= brain.node_count) return;
  const DirectNode& node = brain.nodes[node_index];
  std::uint32_t count = 0u;
  for (std::uint32_t route_index = node.route_offset;
       route_index < node.route_offset + node.route_capacity; ++route_index)
    count += homeostatic_weight_candidate(
        brain, opportunity_incarnations, causal_pins, cost_per_route_q16,
        plan, node_index, route_index);
  candidate_counts[node_index] = count;
}

__global__ void plan_homeostatic_weight_transaction_kernel(
    DirectBrain brain, const std::uint64_t* opportunity_incarnations,
    const std::uint32_t* causal_pins, std::int32_t cost_per_route_q16,
    const TopologyCommitPlan* plan, std::uint32_t* candidate_counts) {
  plan_homeostatic_weight_transaction_impl(
      brain, opportunity_incarnations, causal_pins, cost_per_route_q16, plan,
      candidate_counts, blockIdx.x * blockDim.x + threadIdx.x);
}

__device__ void begin_homeostatic_weight_transaction_impl(
    DirectBrain brain, const std::uint32_t* candidate_counts,
    const std::uint32_t* candidate_prefix, std::uint32_t node_count,
    TopologyCommitPlan* plan, std::uint32_t current_tick) {
  if (plan == nullptr || brain.development == nullptr) return;
  plan->admitted = 0u;
  if (node_count == 0u) return;
  const std::uint32_t last = node_count - 1u;
  const std::uint32_t pending = candidate_prefix[last] + candidate_counts[last];
  // One sweep claims only what remains of the hot page; residual decay
  // candidates carry to later sweeps instead of tripping the fail-closed seal.
  const std::uint32_t room =
      kDirectExactHistoryHotPageCapacity -
      brain.development->exact_history.committed_slots;
  const std::uint32_t width = pending < room ? pending : room;
  if (width == 0u || !begin_exact_history_phase(
          &brain.development->exact_history,
          DirectExactHistoryKind::homeostatic_weight, width, current_tick)) return;
  plan->admitted = 1u;
}

__global__ void begin_homeostatic_weight_transaction_kernel(
    DirectBrain brain, const std::uint32_t* candidate_counts,
    const std::uint32_t* candidate_prefix, std::uint32_t node_count,
    TopologyCommitPlan* plan, std::uint32_t current_tick) {
  begin_homeostatic_weight_transaction_impl(
      brain, candidate_counts, candidate_prefix, node_count, plan, current_tick);
}

__device__ void decay_maintenance_routes_impl(
    DirectBrain brain, const std::uint64_t* opportunity_incarnations,
    const std::uint32_t* causal_pins, std::int32_t cost_per_route_q16,
    const std::uint32_t* candidate_prefix, const TopologyCommitPlan* plan,
    std::uint32_t node_index) {
  if (node_index >= brain.node_count || plan == nullptr || plan->admitted == 0u ||
      brain.development == nullptr ||
      brain.development->exact_history.phase_kind != DirectExactHistoryKind::homeostatic_weight)
    return;
  DirectNode& node = brain.nodes[node_index];
  std::uint32_t local = 0u;
  for (std::uint32_t route_index = node.route_offset;
       route_index < node.route_offset + node.route_capacity; ++route_index) {
    DirectRoute& route = brain.routes[route_index];
    if ((route.flags & kRouteFlagActive) != 0u &&
        route.eligibility_expires < brain.development->age_tick) {
      route.eligibility_q16 = 0;
      route.eligibility_context = 0u;
      route.eligibility_expires = 0u;
    }
    if (!homeostatic_weight_candidate(
            brain, opportunity_incarnations, causal_pins, cost_per_route_q16,
            plan, node_index, route_index)) continue;
    const std::uint32_t ordinal = candidate_prefix[node_index] + local++;
    if (ordinal >= brain.development->exact_history.phase_width) continue;
    const std::int32_t prior = route.conductance_q16;
    const std::int32_t result = maintenance_decayed_conductance(route, cost_per_route_q16);
    DirectExactHistoryRecord* record =
        &brain.development->exact_history.records[
            brain.development->exact_history.phase_base + ordinal];
    record->resident_tick = brain.development->exact_history.phase_tick;
    record->event_tick = brain.development->exact_history.phase_tick;
    record->kind = DirectExactHistoryKind::homeostatic_weight;
    record->source = node_index;
    record->subject = route_index;
    record->value = static_cast<std::uint32_t>(result);
    record->context = route_index - node.route_offset;
    record->flags = static_cast<std::uint32_t>(prior);
    record->incarnation_before = brain.route_incarnations[route_index];
    record->incarnation_after =
        (static_cast<std::uint64_t>(static_cast<std::uint32_t>(node.credit_ema_q16)) << 32u) |
        static_cast<std::uint32_t>(node.activity_ema_q16);
    record->resource_delta = static_cast<std::int64_t>(result) - prior;
    __threadfence();
    route.conductance_q16 = result;
  }
}

__global__ void decay_maintenance_routes_kernel(
    DirectBrain brain, const std::uint64_t* opportunity_incarnations,
    const std::uint32_t* causal_pins, std::int32_t cost_per_route_q16,
    const std::uint32_t* candidate_prefix, const TopologyCommitPlan* plan) {
  decay_maintenance_routes_impl(
      brain, opportunity_incarnations, causal_pins, cost_per_route_q16,
      candidate_prefix, plan, blockIdx.x * blockDim.x + threadIdx.x);
}
