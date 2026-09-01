#ifndef HARDWARE_NATIVE_DIRECT_NETWORK_CONSTRUCTION_FRONT_TYPES
#define HARDWARE_NATIVE_DIRECT_NETWORK_CONSTRUCTION_FRONT_TYPES
DIRECT_NETWORK_HD inline std::uint64_t next_construction_front_generation(std::uint64_t prior) {
  return prior == ~std::uint64_t{0} ? 0u : prior + 1u;
}
#include "direct_network_postbirth_constructor_ecology.inl"
#include "direct_network_repair_regrowth.inl"
#include "direct_network_route_extension.inl"
#include "direct_network_construction_economy.inl"
enum ResidentConstructionFrontState : std::uint16_t { kConstructionFrontQuiescent = 0u, kConstructionFrontLive = 1u };
struct ResidentConstructionFront {
  std::uint32_t source_node, recipe_cell, rule_index, source_route;
  std::uint64_t generation, source_route_incarnation, successor_sequence;
  std::uint16_t territory_index, state;
  std::uint32_t loss_route; std::uint64_t loss_identity, loss_route_incarnation;
  std::uint32_t loss_target, loss_source;
};
static_assert(std::is_standard_layout_v<ResidentConstructionFront> && std::is_trivial_v<ResidentConstructionFront> && std::has_unique_object_representations_v<ResidentConstructionFront>);
DIRECT_NETWORK_HD inline bool construction_front_has_loss_identity(const ResidentConstructionFront& front, std::uint64_t identity) {
  return identity != 0u && front.loss_identity == identity;
}
#endif
#if defined(DIRECT_NETWORK_CONSTRUCTION_FRONT_COMMIT_IMPLEMENTATION)
__device__ void commit_compact_construction_fronts_impl(
    DirectBrain brain, RouteMutationProposal* proposals, std::uint32_t* costs,
    const std::uint64_t* reserve_snapshot, ResidentDevelopmentCounters* counters) {
  if (brain.development == nullptr || brain.construction_front_count == nullptr ||
      proposals == nullptr || costs == nullptr || reserve_snapshot == nullptr) return;
  const std::uint32_t count = min(*brain.construction_front_count,
                                  brain.construction_front_capacity);
  for (std::uint32_t i = 0u; i < count; ++i) {
    proposals[i] = RouteMutationProposal{};
    costs[i] = 0u;
    propose_construction_front_impl(brain, i, proposals, costs, nullptr, counters);
  }
  auto* pool = substrate::direct_adult::direct_ecology_pool(
      brain.resource_ecology,
      substrate::direct_adult::DirectResourcePoolKind::explicit_interaction);
  std::uint64_t accepted_cost = 0u;
  std::uint32_t accepted = 0u;
  std::uint32_t accepted_branches = 0u;
  std::uint32_t proposed = 0u;
  for (std::uint32_t i = 0u; i < count; ++i) {
    RouteMutationProposal& proposal = proposals[i];
    if (proposal.kind != kProposalGrow) continue;
    ++proposed;
    bool winner = true;
    const std::uint64_t own = packed_target_claim(proposal.score_q16, proposal.node);
    for (std::uint32_t j = 0u; j < count; ++j) {
      if (i == j || proposals[j].kind != kProposalGrow ||
          proposals[j].target != proposal.target) continue;
      const std::uint64_t rival = packed_target_claim(
          proposals[j].score_q16, proposals[j].node);
      if (rival < own || (rival == own && j < i)) { winner = false; break; }
    }
    if (!winner) {
      costs[i] = 0u;
      count_commit_drop(counters, kDroppedLostTargetRace);
      continue;
    }
    if (proposal.node >= brain.node_count || proposal.target >= brain.node_count ||
        proposal.route_slot >= brain.nodes[proposal.node].route_capacity) {
      costs[i] = 0u;
      count_commit_drop(counters, kDroppedTargetInvalid);
      continue;
    }
    if (live_construction_front_at_node(brain, proposal.target)) {
      costs[i] = 0u;
      count_commit_drop(counters, kDroppedLostTargetRace);
      continue;
    }
    const DirectNode& source = brain.nodes[proposal.node];
    if (route_is_active(brain.routes[source.route_offset + proposal.route_slot])) {
      costs[i] = 0u;
      count_commit_drop(counters, kDroppedSlotOccupied);
      continue;
    }
    if (brain.nodes[proposal.target].active_in_degree >= kResidentMaximumInDegree) {
      costs[i] = 0u;
      count_commit_drop(counters, kDroppedTargetInDegreeFull);
      continue;
    }
  }
  for (std::uint32_t rank = 0u; rank < count; ++rank) {
    std::uint32_t best = kInvalidIndex;
    for (std::uint32_t i = 0u; i < count; ++i) {
      if (costs[i] == 0u || proposals[i].kind != kProposalGrow) continue;
      if (best == kInvalidIndex || proposals[i].cost < proposals[best].cost ||
          (proposals[i].cost == proposals[best].cost &&
           (proposals[i].node < proposals[best].node ||
            (proposals[i].node == proposals[best].node &&
             (proposals[i].target < proposals[best].target ||
              (proposals[i].target == proposals[best].target && i < best)))))) {
        best = i;
      }
    }
    if (best == kInvalidIndex) break;
    costs[best] = 0u;
    RouteMutationProposal& proposal = proposals[best];
    const ResidentRecipeCell& recipe = brain.recipe_cells[proposal.recipe_cell];
    const bool branch = recipe.rule_index < brain.resident_rule_count &&
        brain.resident_rules[recipe.rule_index].opcode == RuleOpcode::branch;
    if (branch &&
        (brain.resource_ecology == nullptr || count + accepted_branches >=
         brain.resource_ecology->maintenance_scan_budget)) {
      count_commit_drop(counters, kDroppedBudgetExhausted); continue;
    }
    const std::uint64_t next_cost = accepted_cost + proposal.cost;
    if (next_cost > *reserve_snapshot) {
      count_commit_drop(counters, kDroppedBudgetExhausted); continue;
    }
    if (brain.resource_ecology == nullptr || pool == nullptr ||
        next_cost > pool->charged_units - pool->live_units ||
        next_cost > pool->reserved_units) {
      count_commit_drop(counters, kDroppedWorkBudgetExhausted); continue;
    }
    // Construction runs its own window against the epoch churn budget. The
    // node-path admission prefix may consume the whole budget on its own (a
    // sustained retraction storm commits budget-sized retractions every
    // epoch), so front work cannot draw from that residue without starving
    // morphogenesis permanently.
    if (accepted + 1u > brain.resource_ecology->churn.mutation_budget) {
      count_commit_drop(counters, kDroppedWorkBudgetExhausted);
      ++brain.resource_ecology->churn.deferred;
      ++brain.resource_ecology->work.deferred_work;
      brain.resource_ecology->pressure_flags |= substrate::direct_adult::kDirectPressureWork;
      continue;
    }
    proposal.planned_kind = kPlannedGrow;
    accepted_cost = next_cost;
    accepted_branches += branch ? 1u : 0u;
    ++accepted;
  }
  substrate::direct_adult::device_record_churn_proposal(
      brain.resource_ecology, proposed);
  if (accepted == 0u || !begin_exact_history_phase(
          &brain.development->exact_history,
          DirectExactHistoryKind::topology_growth, accepted,
          brain.development->age_tick)) return;
  std::uint32_t ordinal = 0u;
  for (std::uint32_t i = 0u; i < count; ++i) {
    if (proposals[i].planned_kind != kPlannedGrow) continue;
    DirectExactHistoryRecord* record =
        &brain.development->exact_history.records[
            brain.development->exact_history.phase_base + ordinal++];
    commit_node_impl(brain, i, proposals, nullptr, reserve_snapshot, nullptr,
                     nullptr, counters, record, true);
  }
  finish_exact_history_phase(&brain.development->exact_history);
  advance_committed_construction_fronts_impl(brain, proposals, counters, count);
  commit_recipe_transaction_impl(brain, proposals, counters, count);
}
#endif
#if defined(DIRECT_NETWORK_CONSTRUCTION_FRONT_IMPLEMENTATION)
__device__ __forceinline__ bool live_construction_front_at_node(const DirectBrain& brain, std::uint32_t node_index) {
  if (brain.construction_front_count == nullptr || node_index >= brain.node_count) return false;
  const std::uint64_t generation = brain.construction_front_generation_by_node[node_index];
  const std::uint32_t count = min(*brain.construction_front_count, brain.construction_front_capacity);
  for (std::uint32_t i = 0u; i < count; ++i) {
    const ResidentConstructionFront& front = brain.construction_fronts[i];
    if (front.state == kConstructionFrontLive && front.source_node == node_index &&
        front.generation == generation) return true;
  }
  return false;
}
__device__ __forceinline__ void retire_construction_front_generation(DirectBrain brain, std::uint32_t node_index, std::uint64_t generation) {
  if (node_index >= brain.node_count) return;
  const std::uint64_t tombstone = generation == ~std::uint64_t{0} ? generation : generation + 1u;
  atomicMax(reinterpret_cast<unsigned long long*>(brain.construction_front_generation_by_node + node_index),
            static_cast<unsigned long long>(tombstone));
}
#define DIRECT_NETWORK_ROUTE_EXTENSION_IMPLEMENTATION
#include "direct_network_route_extension.inl"
#undef DIRECT_NETWORK_ROUTE_EXTENSION_IMPLEMENTATION
__device__ __forceinline__ bool construction_front_valid(
    const DirectBrain& brain, const ResidentConstructionFront& front) {
  if (front.state != kConstructionFrontLive || front.source_node >= brain.node_count ||
      front.territory_index != brain.nodes[front.source_node].territory_index ||
      front.recipe_cell >= brain.development->recipe_cell_count ||
      front.rule_index >= brain.resident_rule_count ||
      brain.recipe_cells[front.recipe_cell].rule_index != front.rule_index ||
      brain.construction_front_generation_by_node[front.source_node] != front.generation)
    return false;
  const ResidentConstructorRule& rule = brain.resident_rules[front.rule_index];
  if ((rule.flags & kRuleFlagPostBirthResident) == 0u ||
      !resident_growth_opcode(rule.opcode) ||
      !rule_is_active(rule, brain.nodes[front.source_node], brain.development->age_tick))
    return false;
  if (front.source_route == kInvalidIndex) return true;
  if (front.source_route >= brain.route_capacity ||
      brain.route_incarnations[front.source_route] != front.source_route_incarnation)
    return false;
  const DirectRoute& route = brain.routes[front.source_route];
  if (!route_is_active(route)) return false;
  if (rule.opcode == RuleOpcode::branch)
    return route.source == front.source_node || route.target == front.source_node;
  if (rule.opcode == RuleOpcode::fuse) return route.source == front.source_node;
  return route.target == front.source_node;
}

__device__ void propose_construction_front_impl(DirectBrain brain, std::uint32_t front_index,
    RouteMutationProposal* proposals, std::uint32_t* costs,
    std::uint64_t* target_claims, ResidentDevelopmentCounters* counters) {
  if (brain.construction_front_count == nullptr ||
      front_index >= *brain.construction_front_count) return;
  if (counters != nullptr) atomicAdd(&counters->construction_fronts_examined, 1u);
  const ResidentConstructionFront front = brain.construction_fronts[front_index];
  if (!construction_front_valid(brain, front)) {
    ResidentConstructionFront& stale = brain.construction_fronts[front_index];
    stale.state = kConstructionFrontQuiescent;
    retire_construction_front_generation(brain, stale.source_node, stale.generation);
    if (counters != nullptr) atomicAdd(&counters->construction_front_stale_refusals, 1u);
    return;
  }
  const std::uint32_t node_index = front.source_node;
  const DirectNode& node = brain.nodes[node_index];
  RouteMutationProposal proposal{};
  proposal.node = node_index;
  proposal.route_slot = kInvalidIndex;
  proposal.target = kInvalidIndex;
  proposal.recipe_cell = front.recipe_cell;
  proposal.construction_front_index = front_index;
  proposal.construction_front_generation = front.generation;
  if (node.active_route_count >= node.route_capacity) {
    refuse_growth(proposals, costs, counters, front_index, proposal, kRefusedAtRouteCapacity);
    return;
  }
  std::uint32_t free_slot = kInvalidIndex;
  for (std::uint32_t slot = 0u; slot < node.route_capacity; ++slot) {
    if (!route_is_active(brain.routes[node.route_offset + slot])) { free_slot = slot; break; }
  }
  if (free_slot == kInvalidIndex) {
    refuse_growth(proposals, costs, counters, front_index, proposal, kRefusedNoFreeSlot);
    return;
  }
  const ResidentRecipeCell& recipe = brain.recipe_cells[front.recipe_cell];
  const ResidentConstructorRule& rule = brain.resident_rules[front.rule_index];
  const std::uint32_t age = brain.development->age_tick;
  const std::int32_t attract = bound_field_kind_influence_q16(
      brain, node.territory_index, node, age, DevelopmentFieldKind::attract);
  const std::int32_t repel = bound_field_kind_influence_q16(
      brain, node.territory_index, node, age, DevelopmentFieldKind::repel);
  const std::int32_t resource = bound_field_kind_influence_q16(
      brain, node.territory_index, node, age, DevelopmentFieldKind::resource);
  const std::int32_t repair = bound_field_kind_influence_q16(
      brain, node.territory_index, node, age, DevelopmentFieldKind::repair);
  if (rule.opcode == RuleOpcode::repair && repair <= (1 << 14)) {
    refuse_growth(proposals, costs, counters, front_index, proposal, kRefusedBelowGrowthThreshold);
    return;
  }
  std::int64_t growth_score = node.activity_ema_q16 + node.credit_ema_q16 +
      attract + repel + resource + repair + (recipe.support_q16 >> 2) +
      (recipe.credit_q16 >> 1) + (node.attractor_support_q16 >> 2) +
      (static_cast<std::int32_t>(brain.development->plasticity_q16) >> 2);
  const std::int32_t threshold = rule.threshold_q16 == 0u
      ? kGrowthFloorQ16 : static_cast<std::int32_t>(rule.threshold_q16);
  if (growth_score < threshold) {
    refuse_growth(proposals, costs, counters, front_index, proposal, kRefusedBelowGrowthThreshold);
    return;
  }
  const bool long_tract = rule.opcode == RuleOpcode::long_tract;
  std::uint32_t best_target = kInvalidIndex;
  std::int32_t best_score = INT32_MIN;
  for (std::uint32_t candidate = 0u; candidate < kCandidateTargets; ++candidate) {
    const std::uint64_t h = mix64(static_cast<std::uint64_t>(node_index) << 32 ^
        static_cast<std::uint64_t>(age) << 8 ^ candidate ^
        static_cast<std::uint64_t>(node.lineage) * 0x9e3779b97f4a7c15ull);
    const std::uint32_t target_index = static_cast<std::uint32_t>(h % brain.node_count);
    if (target_index == node_index || route_has_target(brain, node, target_index)) continue;
    if (live_construction_front_at_node(brain, target_index)) continue;
    const DirectNode& target = brain.nodes[target_index];
    if (target.active_in_degree >= kResidentMaximumInDegree ||
        (long_tract && target.lineage == node.lineage)) continue;
    const std::int32_t progress = route_extension_progress_q16(
        brain, front, rule, target_index, target);
    if (progress == kRouteExtensionRejected) continue;
    const std::int32_t score = clamp_i32(
        static_cast<std::int64_t>(target_score_q16(brain, node, target, age, long_tract)) +
        progress, -(16 << 16), 16 << 16);
    if (score > best_score || (score == best_score && target_index < best_target)) {
      best_score = score; best_target = target_index;
    }
  }
  if (best_target == kInvalidIndex) {
    refuse_growth(proposals, costs, counters, front_index, proposal, kRefusedNoEligibleTarget);
    return;
  }
  proposal.route_slot = free_slot;
  proposal.target = best_target;
  proposal.kind = kProposalGrow;
  proposal.route_flags = kRouteFlagActive |
      (long_tract ? kRouteFlagLongTract : kRouteFlagRecurrent);
  if ((node.flags & kNodeFlagInhibitory) != 0u) proposal.route_flags |= kRouteFlagInhibitory;
  proposal.score_q16 = clamp_i32(growth_score + best_score / 2, -(16 << 16), 16 << 16);
  proposal.cost = construction_route_cost(node, brain.nodes[best_target]);
  proposals[front_index] = proposal;
  costs[front_index] = 1u;
  if (counters != nullptr) atomicAdd(&counters->grow_proposals_emitted, 1u);
  if (target_claims != nullptr)
    atomicMin(reinterpret_cast<unsigned long long*>(&target_claims[best_target]),
              static_cast<unsigned long long>(packed_target_claim(proposal.score_q16, node_index)));
}

__global__ void propose_construction_fronts_kernel(
    DirectBrain brain, RouteMutationProposal* proposals, std::uint32_t* costs,
    std::uint64_t* target_claims, ResidentDevelopmentCounters* counters) {
  const std::uint32_t count = brain.construction_front_count == nullptr
      ? 0u : *brain.construction_front_count;
  for (std::uint32_t front = threadIdx.x; front < count; front += blockDim.x)
    propose_construction_front_impl(brain, front, proposals, costs, target_claims, counters);
}
#define DIRECT_NETWORK_REPAIR_REGROWTH_IMPLEMENTATION
#include "direct_network_repair_regrowth.inl"
#undef DIRECT_NETWORK_REPAIR_REGROWTH_IMPLEMENTATION
#endif
