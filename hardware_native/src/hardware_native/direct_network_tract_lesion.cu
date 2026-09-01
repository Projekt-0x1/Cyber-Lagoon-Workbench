#include "hardware_native/direct_network_tract_lesion.cuh"
#include "hardware_native/direct_adult_core.cuh"
#include "hardware_native/direct_resource_ecology.cuh"

#include <cuda_runtime.h>

#include <vector>

namespace substrate::direct_network::tract_lesion {
namespace {

__device__ void nominate_accounted_lesion_repair_front(
    DirectBrain brain, std::uint32_t source_node, std::uint64_t loss_identity,
    std::uint32_t loss_route, std::uint32_t loss_target,
    std::uint64_t loss_route_incarnation) {
  if (loss_identity == 0u || source_node >= brain.node_count ||
      brain.construction_front_count == nullptr) return;
  const DirectNode& node = brain.nodes[source_node];
  if (node.active_route_count >= 2u || node.territory_index >= brain.recipe_range_count) return;
  const ResidentRecipeRange range = brain.recipe_ranges[node.territory_index];
  std::uint32_t repair_cell = kInvalidIndex;
  for (std::uint32_t local = 0u; local < range.index_count; ++local) {
    const std::uint32_t index = range.index_offset + local;
    if (index >= brain.recipe_index_count) break;
    const std::uint32_t cell_index = brain.recipe_indices[index];
    if (cell_index >= brain.recipe_cell_count) continue;
    const ResidentRecipeCell& cell = brain.recipe_cells[cell_index];
    if (cell.rule_index < brain.resident_rule_count &&
        brain.resident_rules[cell.rule_index].opcode == RuleOpcode::repair &&
        (brain.resident_rules[cell.rule_index].flags & kRuleFlagPostBirthResident) != 0u) {
      repair_cell = cell_index;
      break;
    }
  }
  if (repair_cell == kInvalidIndex) return;
  const std::uint32_t count = *brain.construction_front_count;
  for (std::uint32_t i = 0u; i < count; ++i)
    if (construction_front_has_loss_identity(
            brain.construction_fronts[i], loss_identity)) return;
  for (std::uint32_t i = 0u; i < count; ++i) {
    ResidentConstructionFront& front = brain.construction_fronts[i];
    if (front.source_node != source_node || front.recipe_cell != repair_cell) continue;
    front.state = kConstructionFrontLive;
    front.loss_identity = loss_identity;
    front.loss_route = loss_route;
    front.loss_target = loss_target;
    front.loss_source = source_node;
    front.loss_route_incarnation = loss_route_incarnation;
    front.source_route = kInvalidIndex;
    front.source_route_incarnation = 0u;
    const std::uint64_t generation = next_construction_front_generation(
        brain.construction_front_generation_by_node[source_node]);
    if (generation == 0u) return;
    front.generation = generation;
    brain.construction_front_generation_by_node[source_node] = generation;
    return;
  }
  if (count >= brain.construction_front_capacity) return;
  const ResidentRecipeCell& cell = brain.recipe_cells[repair_cell];
  const std::uint64_t generation = next_construction_front_generation(
      brain.construction_front_generation_by_node[source_node]);
  if (generation == 0u) return;
  brain.construction_fronts[count] = ResidentConstructionFront{
      source_node, repair_cell, cell.rule_index, kInvalidIndex,
      generation, 0u, 0u, node.territory_index,
      kConstructionFrontLive, loss_route, loss_identity, loss_route_incarnation,
      loss_target, source_node};
  brain.construction_front_generation_by_node[source_node] =
      brain.construction_fronts[count].generation;
  *brain.construction_front_count = count + 1u;
}

__global__ void fail_source_routes_accounted_kernel(
    DirectBrain brain, std::uint32_t source_node, std::uint32_t viability_floor,
    const std::uint32_t* causal_pin_bits, LesionCounts* receipt) {
  if (blockIdx.x != 0u || threadIdx.x != 0u || receipt == nullptr) return;
  *receipt = LesionCounts{};
  receipt->source_node = source_node;
  if (source_node >= brain.node_count || brain.development == nullptr ||
      brain.resource_ecology == nullptr) return;
  DirectNode& source = brain.nodes[source_node];
  receipt->source_territory = source.territory_index;
  receipt->source_active_before = source.active_route_count;
  const std::uint32_t lesion_budget = source.active_route_count > viability_floor
      ? source.active_route_count - viability_floor : 0u;
  std::uint32_t eligible = 0u;
  for (std::uint32_t slot = 0u; slot < source.route_capacity &&
       eligible < lesion_budget; ++slot) {
    const std::uint32_t route_index = source.route_offset + slot;
    const DirectRoute& route = brain.routes[route_index];
    if (route_is_active(route) && route.source == source_node &&
        (causal_pin_bits == nullptr ||
         (causal_pin_bits[route_index >> 5u] & (1u << (route_index & 31u))) == 0u))
      ++eligible;
  }
  if (eligible == 0u || !begin_exact_history_phase(
          &brain.development->exact_history, DirectExactHistoryKind::topology_retraction,
          eligible, brain.development->age_tick)) return;
  for (std::uint32_t slot = 0u;
       slot < source.route_capacity && source.active_route_count > viability_floor; ++slot) {
    const std::uint32_t route_index = source.route_offset + slot;
    DirectRoute& route = brain.routes[route_index];
    if (!route_is_active(route) || route.source != source_node ||
        (causal_pin_bits != nullptr &&
         (causal_pin_bits[route_index >> 5u] & (1u << (route_index & 31u))) != 0u)) continue;
    const std::uint64_t incarnation_before = brain.route_incarnations == nullptr
        ? 0u : brain.route_incarnations[route_index];
    DirectExactHistoryRecord* history_record =
        &brain.development->exact_history.records[
            brain.development->exact_history.phase_base + receipt->routes_deactivated];
    stage_focal_lesion_history_record(
        history_record, brain.development->age_tick, source_node, route_index,
        route.target, route.flags, incarnation_before,
        incarnation_before + kRouteIncarnationStride);
    __threadfence();
    if (!substrate::direct_adult::device_uncommit_pool_units(
            brain.resource_ecology,
            substrate::direct_adult::DirectResourcePoolKind::explicit_interaction, 1u)) {
      *history_record = DirectExactHistoryRecord{};
      break;
    }
    if (receipt->routes_deactivated == 0u) ++receipt->nodes_written;
    route.flags &= ~kRouteFlagActive;
    route.flags |= kRouteFlagDevelopmentalReserve;
    if (brain.route_incarnations != nullptr)
      brain.route_incarnations[route_index] += kRouteIncarnationStride;
    route.eligibility_q16 = 0; route.eligibility_context = kInvalidIndex;
    route.eligibility_expires = 0u; route.last_credit_q16 = 0;
    --source.active_route_count;
    if (route.target < brain.node_count && brain.nodes[route.target].active_in_degree != 0u)
      { --brain.nodes[route.target].active_in_degree; ++receipt->nodes_written; }
    ++brain.development->constructor_reserve;
    ++brain.development->reclaimed_resource;
    --brain.development->live_route_matter;
    substrate::direct_adult::device_record_churn_commit(brain.resource_ecology, 1u);
    ++brain.resource_ecology->churn.admitted;
    nominate_accounted_lesion_repair_front(
        brain, source_node, history_record->identity, route_index,
        history_record->value, history_record->incarnation_after);
    ++receipt->routes_deactivated;
    ++receipt->matter_returned;
  }
  receipt->history_records = finish_exact_history_phase(
      &brain.development->exact_history);
  receipt->source_active_after = source.active_route_count;
  receipt->applied = receipt->routes_deactivated != 0u;
}

// The tissue is read back to the host, edited, and written back rather than
// mutated by a kernel. Deliberate: `sham_disconnect` has to take the FIRST
// `budget` eligible routes in slot order, and a kernel selecting them by
// atomic counter would pick a different set on every run -- an intervention
// that is not reproducible is not a control. A host pass makes all four
// operators deterministic by construction for a few thousand 48-byte routes,
// which is a probe cost, not a runtime cost.
struct HostTissue {
  std::vector<DirectNode> nodes;
  std::vector<DirectRoute> routes;
  std::vector<std::uint64_t> route_incarnations;
  bool ok = false;
};

HostTissue load(const DirectBrain& brain) {
  HostTissue tissue;
  if (brain.nodes == nullptr || brain.routes == nullptr) return tissue;
  tissue.nodes.resize(brain.node_count);
  tissue.routes.resize(brain.route_capacity);
  if (brain.node_count != 0u &&
      cudaMemcpy(tissue.nodes.data(), brain.nodes, sizeof(DirectNode) * brain.node_count,
                 cudaMemcpyDeviceToHost) != cudaSuccess)
    return tissue;
  if (brain.route_capacity != 0u &&
      cudaMemcpy(tissue.routes.data(), brain.routes, sizeof(DirectRoute) * brain.route_capacity,
                 cudaMemcpyDeviceToHost) != cudaSuccess)
    return tissue;
  if (brain.route_incarnations != nullptr) {
    tissue.route_incarnations.resize(brain.route_capacity);
    if (brain.route_capacity != 0u &&
        cudaMemcpy(tissue.route_incarnations.data(), brain.route_incarnations,
                   sizeof(std::uint64_t) * brain.route_capacity,
                   cudaMemcpyDeviceToHost) != cudaSuccess)
      return tissue;
  }
  tissue.ok = true;
  return tissue;
}

bool store_routes(DirectBrain& brain, const HostTissue& tissue) {
  if (!tissue.routes.empty() &&
      cudaMemcpy(brain.routes, tissue.routes.data(),
                 sizeof(DirectRoute) * tissue.routes.size(),
                 cudaMemcpyHostToDevice) != cudaSuccess)
    return false;
  return tissue.route_incarnations.empty() ||
         cudaMemcpy(brain.route_incarnations, tissue.route_incarnations.data(),
                    sizeof(std::uint64_t) * tissue.route_incarnations.size(),
                    cudaMemcpyHostToDevice) == cudaSuccess;
}

// A route slot that carries live tissue. `route_is_active` is the organism's
// own predicate (direct_network_brain.cuh), not a second opinion about what
// active means; the range checks reject slots the arena never bound.
bool live(const HostTissue& tissue, const DirectRoute& route) {
  return route_is_active(route) && route.source < tissue.nodes.size() &&
         route.target < tissue.nodes.size();
}

bool in_territory(const HostTissue& tissue, std::uint32_t node, std::uint32_t territory) {
  if (territory == kAnyTerritory) return true;
  return node < tissue.nodes.size() &&
         static_cast<std::uint32_t>(tissue.nodes[node].territory_index) == territory;
}

bool is_corridor(const HostTissue& tissue, const DirectRoute& route, std::uint32_t source_territory,
                 std::uint32_t target_territory) {
  return live(tissue, route) && (route.flags & kRouteFlagLongTract) != 0u &&
         in_territory(tissue, route.source, source_territory) &&
         in_territory(tissue, route.target, target_territory);
}

// Keep the brain header honest after an intervention. A declared count that
// silently outlives the tissue it describes is exactly the defect the census
// records `declared_active_routes` beside the measurement to catch, so the
// operators do not create one.
void debit_declared(DirectBrain& brain, std::uint32_t cut) {
  brain.active_route_count = cut >= brain.active_route_count ? 0u : brain.active_route_count - cut;
}

// Shared body of all three interventions: clear kRouteFlagActive wherever
// `select` says, write back once, debit the header.
template <typename Select>
LesionCounts apply(DirectBrain& brain, Select select) {
  LesionCounts counts;
  HostTissue tissue = load(brain);
  if (!tissue.ok) return counts;
  for (std::uint32_t i = 0u; i < tissue.routes.size(); ++i) {
    if (!select(tissue, tissue.routes[i], i, counts.routes_deactivated)) continue;
    tissue.routes[i].flags &= ~kRouteFlagActive;
    if (!tissue.route_incarnations.empty())
      tissue.route_incarnations[i] += kRouteIncarnationStride;
    ++counts.routes_deactivated;
  }
  if (!store_routes(brain, tissue)) return counts;
  debit_declared(brain, counts.routes_deactivated);
  counts.applied = true;
  return counts;
}

}  // namespace

LesionCounts fail_source_routes_accounted(
    DirectBrain& brain, std::uint32_t source_node,
    std::uint32_t viability_floor, const std::uint32_t* causal_pin_bits) {
  LesionCounts* device_receipt = nullptr;
  LesionCounts receipt{};
  if (cudaMalloc(&device_receipt, sizeof(receipt)) != cudaSuccess) return receipt;
  fail_source_routes_accounted_kernel<<<1, 1>>>(
      brain, source_node, viability_floor, causal_pin_bits, device_receipt);
  if (cudaGetLastError() == cudaSuccess &&
      cudaMemcpy(&receipt, device_receipt, sizeof(receipt),
                 cudaMemcpyDeviceToHost) == cudaSuccess && receipt.applied) {
    brain.active_route_count = receipt.routes_deactivated >= brain.active_route_count
        ? 0u : brain.active_route_count - receipt.routes_deactivated;
  }
  cudaFree(device_receipt);
  return receipt;
}

TractCensus census_tract(const DirectBrain& brain, std::uint32_t source_territory,
                         std::uint32_t target_territory) {
  TractCensus census;
  const HostTissue tissue = load(brain);
  if (!tissue.ok) return census;
  census.declared_active_routes = brain.active_route_count;
  std::int64_t corridor_conductance_sum = 0;

  for (const DirectNode& node : tissue.nodes) {
    const std::uint32_t territory = static_cast<std::uint32_t>(node.territory_index);
    if (source_territory != kAnyTerritory && territory == source_territory) ++census.source_nodes;
    if (target_territory != kAnyTerritory && territory == target_territory) ++census.target_nodes;
  }

  for (const DirectRoute& route : tissue.routes) {
    if (!live(tissue, route)) continue;
    ++census.active_routes_total;
    if ((route.flags & kRouteFlagLongTract) != 0u) ++census.long_tract_routes;
    if (tissue.nodes[route.source].territory_index != tissue.nodes[route.target].territory_index)
      ++census.inter_territory_routes;
    const bool corridor = is_corridor(tissue, route, source_territory, target_territory);
    if (corridor) {
      ++census.corridor_routes;
      corridor_conductance_sum += route.conductance_q16;
      continue;
    }
    if (in_territory(tissue, route.source, source_territory)) ++census.source_other_routes;
    if (in_territory(tissue, route.target, target_territory)) ++census.target_incoming_other;
    if (in_territory(tissue, route.source, target_territory)) ++census.target_outgoing_routes;
  }
  if (census.corridor_routes > 0u) {
    census.mean_corridor_conductance_q16 =
        static_cast<std::int32_t>(corridor_conductance_sum / census.corridor_routes);
  }
  census.valid = true;
  return census;
}

LesionCounts disconnect_tract(DirectBrain& brain, std::uint32_t source_territory,
                              std::uint32_t target_territory) {
  return apply(brain, [&](const HostTissue& tissue, const DirectRoute& route, std::uint32_t,
                          std::uint32_t) {
    return is_corridor(tissue, route, source_territory, target_territory);
  });
}

LesionCounts lesion_territory(DirectBrain& brain, std::uint32_t territory) {
  return apply(brain, [&](const HostTissue& tissue, const DirectRoute& route, std::uint32_t,
                          std::uint32_t) {
    if (!live(tissue, route)) return false;
    // Incident in EITHER direction. A lesion that only cut the territory's
    // outgoing routes would leave it still driven by the rest of the organism,
    // which is a partial disconnection wearing a lesion's name.
    return in_territory(tissue, route.source, territory) ||
           in_territory(tissue, route.target, territory);
  });
}

LesionCounts sham_disconnect(DirectBrain& brain, std::uint32_t source_territory,
                             std::uint32_t target_territory, std::uint32_t budget,
                             bool long_tract_only) {
  return apply(brain, [&](const HostTissue& tissue, const DirectRoute& route, std::uint32_t,
                          std::uint32_t already_cut) {
    if (already_cut >= budget) return false;
    if (!live(tissue, route)) return false;
    if (long_tract_only && (route.flags & kRouteFlagLongTract) == 0u) return false;
    return !is_corridor(tissue, route, source_territory, target_territory);
  });
}

}  // namespace substrate::direct_network::tract_lesion
