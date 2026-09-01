#include "hardware_native/direct_network_resident_development.cuh"

#include "hardware_native/direct_adult_resource_maintenance.cuh"
#include "hardware_native/direct_resource_ecology.cuh"
#include "hardware_native/direct_retention_policy.cuh"

#include <cuda_runtime.h>
#include <cub/device/device_scan.cuh>

#include <algorithm>
#include <cstdint>
#include <climits>
#include <stdexcept>
#include <string>

namespace substrate::direct_network {
namespace {

constexpr std::uint32_t kProposalNone = 0u;
constexpr std::uint32_t kProposalGrow = 1u;
constexpr std::uint32_t kProposalRetract = 2u;
constexpr std::uint32_t kPlannedNone = 0u;
constexpr std::uint32_t kPlannedGrow = 1u;
constexpr std::uint32_t kPlannedRetract = 2u;
constexpr std::uint32_t kPlannedPressureRetract = 3u;
constexpr std::uint32_t kPlannedRejectedWork = 4u;
constexpr std::uint32_t kPlannedRejectedCapacity = 5u;
constexpr std::uint32_t kCandidateTargets = 16u;
// Resident-development floor; distinct from the adult-core transmission floor.
constexpr std::int32_t kMinConductanceQ16 = 1 << 10;
constexpr std::int32_t kInitialGrownConductanceQ16 = 1 << 15;
constexpr std::int32_t kRetractionScoreQ16 = -(1 << 14);
constexpr std::int32_t kGrowthFloorQ16 = 1 << 13;
constexpr std::uint32_t kResidentMaximumInDegree = 64u;

__device__ bool route_has_causal_memory(const DirectBrain& brain,
                                        std::uint32_t route_index) {
  if (brain.retention_bank == nullptr || brain.route_incarnations == nullptr ||
      route_index >= brain.route_capacity) return false;
  const DirectRoute& route = brain.routes[route_index];
  const auto& retention = brain.retention_bank[route_index];
  return retention.logical_source == route.source &&
         retention.logical_slot == route_index &&
         retention.logical_generation == brain.route_incarnations[route_index] &&
         (retention.last_support_tick != 0u || retention.support_ema_q16 != 0 ||
          retention.contradiction_ema_q16 != 0u);
}

__device__ bool causal_route_more_expendable(const DirectBrain& brain,
                                             std::uint32_t candidate,
                                             std::uint32_t incumbent,
                                             std::uint64_t tick) {
  const bool candidate_has_memory = route_has_causal_memory(brain, candidate);
  const bool incumbent_has_memory = route_has_causal_memory(brain, incumbent);
  if (candidate_has_memory != incumbent_has_memory) return !candidate_has_memory;
  if (!candidate_has_memory) return false;
  return substrate::direct_adult::compare_reclaim_candidates_rich(
      brain.retention_bank[candidate], brain.retention_bank[incumbent], tick);
}
void check_cuda(cudaError_t status, const char* operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
  }
}
std::uint32_t grid_for(std::uint32_t count, std::uint32_t block_size) {
  return std::max(1u, (count + block_size - 1u) / block_size);
}
__device__ __forceinline__ std::uint64_t mix64(std::uint64_t x) {
  x += 0x9e3779b97f4a7c15ull;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
  return x ^ (x >> 31);
}
__device__ __forceinline__ std::int32_t iabs32(std::int32_t value) {
  return value < 0 ? -value : value;
}
__device__ __forceinline__ std::int32_t clamp_i32(std::int64_t value, std::int32_t lo,
                                                   std::int32_t hi) {
  return value < lo ? lo : (value > hi ? hi : static_cast<std::int32_t>(value));
}
__device__ __forceinline__ std::uint32_t clamp_u32(std::uint32_t value, std::uint32_t lo,
                                                    std::uint32_t hi) {
  return value < lo ? lo : (value > hi ? hi : value);
}
__device__ __forceinline__ std::uint64_t packed_target_claim(std::int32_t score_q16,
                                                              std::uint32_t source_node) {
  // Signed -> monotonically ordered unsigned, then invert so atomicMin picks
  // highest score; source node index is the deterministic tie-break.
  const std::uint32_t ordered = static_cast<std::uint32_t>(score_q16) ^ 0x80000000u;
  const std::uint32_t inverted = 0xffffffffu - ordered;
  return (static_cast<std::uint64_t>(inverted) << 32) | source_node;
}
__device__ __forceinline__ bool field_is_active(const ResidentDevelopmentField& field,
                                                 std::uint64_t logical_tick,
                                                 std::uint32_t chemotype) {
  if (logical_tick < field.begin_tick) return false;
  if (field.end_tick != 0u && logical_tick >= field.end_tick) return false;
  return (chemotype & field.require_mask) == field.require_value;
}
__device__ __forceinline__ std::int32_t field_influence_q16(
    const DirectBrain& brain, const DirectNode& node, std::uint16_t field_index,
    std::uint32_t age) {
  if (field_index == kInvalidFieldIndex16 || field_index >= brain.resident_field_count) return 0;
  const ResidentDevelopmentField& field = brain.resident_fields[field_index];
  const std::uint64_t logical_tick =
      static_cast<std::uint64_t>(brain.development->birth_handoff_tick) + age;
  if (!field_is_active(field, logical_tick, node.chemotype)) return 0;
  const std::int32_t dx = iabs32(node.coordinate[0] - field.center[0]);
  const std::int32_t dy = iabs32(node.coordinate[1] - field.center[1]);
  const std::int32_t dz = iabs32(node.coordinate[2] - field.center[2]);
  const std::uint64_t distance = static_cast<std::uint64_t>(dx) + dy + dz;
  const std::uint64_t radius = static_cast<std::uint64_t>(field.radius) * 3ull;
  if (radius == 0ull || distance > radius) return 0;
  const std::int64_t falloff_q16 =
      static_cast<std::int64_t>((radius - distance) << 16) / static_cast<std::int64_t>(radius);
  std::int64_t value =
      (static_cast<std::int64_t>(resident_field_decayed_strength_q16(field, logical_tick)) *
       falloff_q16) >> 16;
  if (field.kind == DevelopmentFieldKind::repel || field.kind == DevelopmentFieldKind::inhibition) {
    value = -value;
  }
  return clamp_i32(value, -(8 << 16), 8 << 16);
}

__device__ __forceinline__ std::int32_t bound_field_kind_influence_q16(
    const DirectBrain& brain, std::uint16_t territory_index,
    const DirectNode& target, std::uint32_t age, DevelopmentFieldKind kind) {
  if (territory_index >= brain.resident_field_range_count) return 0;
  const ResidentFieldRange range = brain.resident_field_ranges[territory_index];
  if (static_cast<std::uint64_t>(range.index_offset) + range.index_count >
      brain.resident_field_index_count)
    return 0;
  std::int64_t sum = 0;
  for (std::uint32_t i = 0u; i < range.index_count; ++i) {
    const std::uint16_t field_index =
        brain.resident_field_indices[range.index_offset + i];
    if (field_index >= brain.resident_field_count ||
        brain.resident_fields[field_index].kind != kind)
      continue;
    sum += field_influence_q16(brain, target, field_index, age);
  }
  return clamp_i32(sum, -(16 << 16), 16 << 16);
}

__device__ __forceinline__ std::uint32_t apply_resident_chemotype_write_device(
    std::uint32_t current, const ResidentConstructorRule& rule) {
  return (current & ~rule.write_mask) | (rule.write_value & rule.write_mask);
}

__device__ __forceinline__ bool rule_is_active_for_chemotype(
    const ResidentConstructorRule& rule, std::uint32_t chemotype, std::uint32_t age) {
  if (age < rule.begin_age) return false;
  if (rule.end_age != 0u && age >= rule.end_age) return false;
  return (chemotype & rule.require_mask) == rule.require_value;
}

__device__ __forceinline__ bool rule_is_active(const ResidentConstructorRule& rule,
                                                const DirectNode& node,
                                                std::uint32_t age) {
  return rule_is_active_for_chemotype(rule, node.chemotype, age);
}

__device__ __forceinline__ bool route_has_target(const DirectBrain& brain,
                                                  const DirectNode& source,
                                                  std::uint32_t target) {
  for (std::uint32_t slot = 0; slot < source.route_capacity; ++slot) {
    const DirectRoute& route = brain.routes[source.route_offset + slot];
    if (route_is_active(route) && route.target == target) return true;
  }
  return false;
}

__device__ __forceinline__ std::int32_t target_score_q16(const DirectBrain& brain,
                                                          const DirectNode& source,
                                                          const DirectNode& target,
                                                          std::uint32_t age,
                                                          bool long_tract) {
  const std::int64_t distance = static_cast<std::int64_t>(
      iabs32(source.coordinate[0] - target.coordinate[0]) +
      iabs32(source.coordinate[1] - target.coordinate[1]) +
      iabs32(source.coordinate[2] - target.coordinate[2]));
  const std::int64_t locality = long_tract ? (distance << 5) : -(distance << 7);
  std::int64_t score = locality;
  score += bound_field_kind_influence_q16(
      brain, source.territory_index, target, age, DevelopmentFieldKind::attract);
  score += bound_field_kind_influence_q16(
      brain, source.territory_index, target, age, DevelopmentFieldKind::repel);
  score += bound_field_kind_influence_q16(
               brain, source.territory_index, target, age,
               DevelopmentFieldKind::resource) /
           2;
  if (!long_tract && source.lineage == target.lineage) score += 1 << 15;
  if (long_tract && source.lineage != target.lineage) score += 1 << 16;
  score += target.attractor_support_q16 >> 3;
  score -= target.inhibition_q16 >> 2;
  return clamp_i32(score, -(16 << 16), 16 << 16);
}

__device__ __forceinline__ std::uint32_t route_delay(const DirectNode& source,
                                                      const DirectNode& target,
                                                      bool long_tract) {
  const std::uint64_t distance = static_cast<std::uint64_t>(
      iabs32(source.coordinate[0] - target.coordinate[0]) +
      iabs32(source.coordinate[1] - target.coordinate[1]) +
      iabs32(source.coordinate[2] - target.coordinate[2]));
  const std::uint32_t divisor = long_tract ? 128u : 32u;
  const std::uint32_t delay = 1u + static_cast<std::uint32_t>(distance / divisor);
  return clamp_u32(delay, 1u, long_tract ? 64u : 16u);
}

__device__ void advance_development_state_impl(ResidentDevelopmentState* state,
                                               std::uint64_t* reserve_snapshot) {
  // One warp owns tiny epoch metadata; node/route development remains parallel.
  if (state == nullptr) return;
  // Snapshot reserve before this epoch's later structural commits.
  if (reserve_snapshot != nullptr) {
    *reserve_snapshot = state->constructor_reserve;
  }
  ++state->age_tick;
  const std::uint32_t age = state->age_tick;
  state->phase = age < 64u ? 0u : (age < 256u ? 1u : (age < 1024u ? 2u : 3u));
  if (state->critical_period_q16 > state->mature_plasticity_floor_q16) {
    const std::uint32_t decay = 64u + (state->phase << 5);
    state->critical_period_q16 =
        state->critical_period_q16 > decay ? state->critical_period_q16 - decay
                                           : state->mature_plasticity_floor_q16;
  }
  const std::uint64_t mixed =
      (static_cast<std::uint64_t>(state->critical_period_q16) * 3ull +
       static_cast<std::uint64_t>(state->plasticity_q16)) >> 2;
  state->plasticity_q16 = static_cast<std::uint32_t>(
      mixed < state->mature_plasticity_floor_q16 ? state->mature_plasticity_floor_q16 : mixed);
}

__global__ void advance_development_state_kernel(ResidentDevelopmentState* state,
                                                  std::uint64_t* reserve_snapshot) {
  if (blockIdx.x == 0u && threadIdx.x == 0u)
    advance_development_state_impl(state, reserve_snapshot);
}

__device__ std::uint32_t folded_mature_chemotype(
    const DirectBrain& brain, const DirectNode& node, std::uint32_t age) {
  const std::uint32_t pre_chemotype = node.chemotype;
  std::uint32_t next_chemotype = pre_chemotype;
  if (node.territory_index >= brain.recipe_range_count || brain.recipe_ranges == nullptr ||
      brain.recipe_indices == nullptr || brain.recipe_cells == nullptr ||
      brain.resident_rules == nullptr) {
    return next_chemotype;
  }
  const ResidentRecipeRange range = brain.recipe_ranges[node.territory_index];
  for (std::uint32_t local = 0u; local < range.index_count; ++local) {
    const std::uint32_t index = range.index_offset + local;
    if (index >= brain.recipe_index_count) break;
    const std::uint32_t cell_id = brain.recipe_indices[index];
    if (cell_id >= brain.recipe_cell_count) continue;
    const ResidentRecipeCell& cell = brain.recipe_cells[cell_id];
    if (cell.rule_index >= brain.resident_rule_count) continue;
    const ResidentConstructorRule& rule = brain.resident_rules[cell.rule_index];
    if (rule.opcode != RuleOpcode::mature ||
        !rule_is_active_for_chemotype(rule, pre_chemotype, age)) {
      continue;
    }
    if (rule.field_index < brain.resident_field_count &&
        rule.field_index < kInvalidFieldIndex16 &&
        field_influence_q16(brain, node, static_cast<std::uint16_t>(rule.field_index), age) == 0) {
      continue;
    }
    next_chemotype = apply_resident_chemotype_write_device(next_chemotype, rule);
  }
  return next_chemotype;
}

__device__ void mature_node_impl(DirectBrain brain, std::uint32_t node_index,
                                 ResidentDevelopmentCounters* counters) {
  if (node_index >= brain.node_count || brain.development == nullptr) return;
  DirectNode& node = brain.nodes[node_index];
  const std::uint32_t age = brain.development->age_tick;
  const std::uint32_t next_chemotype = folded_mature_chemotype(brain, node, age);

  std::int64_t activity_sum = 0;
  std::int64_t credit_sum = 0;
  std::int64_t support_sum = 0;
  std::uint32_t active = 0u;
  for (std::uint32_t slot = 0; slot < node.route_capacity; ++slot) {
    const DirectRoute& route = brain.routes[node.route_offset + slot];
    if (!route_is_active(route)) continue;
    ++active;
    support_sum += route.conductance_q16;
    credit_sum += route.last_credit_q16;
  }
  if (active != 0u) {
    support_sum /= static_cast<std::int64_t>(active);
    credit_sum /= static_cast<std::int64_t>(active);
  }
  activity_sum = node.activation_q16;
  node.activity_ema_q16 = clamp_i32(
      (static_cast<std::int64_t>(node.activity_ema_q16) * 15 + activity_sum) / 16,
      -(8 << 16), 8 << 16);
  node.credit_ema_q16 = clamp_i32(
      (static_cast<std::int64_t>(node.credit_ema_q16) * 15 + credit_sum) / 16,
      -(8 << 16), 8 << 16);

  const std::int32_t maturation = bound_field_kind_influence_q16(
      brain, node.territory_index, node, age, DevelopmentFieldKind::maturation);
  const std::int32_t resource = bound_field_kind_influence_q16(
      brain, node.territory_index, node, age, DevelopmentFieldKind::resource);
  const std::int32_t repel = bound_field_kind_influence_q16(
      brain, node.territory_index, node, age, DevelopmentFieldKind::repel);
  const std::int32_t inhibition = bound_field_kind_influence_q16(
      brain, node.territory_index, node, age, DevelopmentFieldKind::inhibition);
  const std::int32_t attract = bound_field_kind_influence_q16(
      brain, node.territory_index, node, age, DevelopmentFieldKind::attract);

  const bool was_mature = node.maturation_q16 == kQ16One;
  std::int64_t maturation_step = (1 << 8) + (maturation >> 8) + (node.activity_ema_q16 >> 10);
  if (maturation_step < 1) maturation_step = 1;
  node.maturation_q16 = clamp_u32(
      node.maturation_q16 + static_cast<std::uint32_t>(maturation_step), 0u, kQ16One);
  node.maintenance_q16 = clamp_i32(
      static_cast<std::int64_t>(node.maintenance_q16) + (resource >> 8) +
          (node.activity_ema_q16 >> 11) - (repel < 0 ? (-repel >> 10) : 0),
      0, 4 << 16);
  node.inhibition_q16 = clamp_i32(
      (static_cast<std::int64_t>(node.inhibition_q16) * 7 +
       static_cast<std::int64_t>(brain.development->inhibition_gain_q16) +
       (inhibition < 0 ? -inhibition : 0) + (repel < 0 ? (-repel >> 1) : 0)) /
          8,
      0, 4 << 16);
  node.attractor_support_q16 = clamp_i32(
      (support_sum * 3 + node.activity_ema_q16 + node.credit_ema_q16 + attract) / 4,
      0, 8 << 16);
  if (node.maturation_q16 >= kQ16One / 2u) node.flags &= ~kNodeFlagImmature;
  node.chemotype = next_chemotype;
  if (!was_mature && node.maturation_q16 == kQ16One) atomicAdd(&counters->matured_nodes, 1u);
  atomicAdd(&counters->field_updates, 1u);
}

__global__ void update_node_maturation_kernel(DirectBrain brain,
                                               ResidentDevelopmentCounters* counters) {
  mature_node_impl(brain, blockIdx.x * blockDim.x + threadIdx.x, counters);
}

__device__ __forceinline__ bool resident_growth_opcode(RuleOpcode opcode) {
  return opcode == RuleOpcode::branch || opcode == RuleOpcode::extend ||
         opcode == RuleOpcode::repair || opcode == RuleOpcode::long_tract ||
         opcode == RuleOpcode::fuse;
}

__device__ __forceinline__ std::uint32_t select_recipe_cell(
    const DirectBrain& brain, const DirectNode& node, std::uint32_t age,
    bool growth_mode) {
  if (node.territory_index >= brain.recipe_range_count || brain.recipe_ranges == nullptr ||
      brain.recipe_indices == nullptr || brain.recipe_cells == nullptr) {
    return kInvalidIndex;
  }
  const ResidentRecipeRange range = brain.recipe_ranges[node.territory_index];
  std::uint32_t best_cell = kInvalidIndex;
  std::int32_t best_drive = INT32_MIN;
  for (std::uint32_t local = 0; local < range.index_count; ++local) {
    const std::uint32_t index = range.index_offset + local;
    if (index >= brain.recipe_index_count) break;
    const std::uint32_t cell_id = brain.recipe_indices[index];
    if (cell_id >= brain.recipe_cell_count) continue;
    const ResidentRecipeCell& cell = brain.recipe_cells[cell_id];
    if (cell.rule_index >= brain.resident_rule_count) continue;
    const ResidentConstructorRule& rule = brain.resident_rules[cell.rule_index];
    if (!rule_is_active(rule, node, age)) continue;
    if (growth_mode) {
      if (!resident_growth_opcode(rule.opcode)) continue;
    } else if (rule.opcode != RuleOpcode::retract) {
      continue;
    }
    std::int64_t drive = static_cast<std::int64_t>(cell.support_q16) + cell.credit_q16;
    if (rule.field_index < brain.resident_field_count && rule.field_index < kInvalidFieldIndex16) {
      drive += field_influence_q16(brain, node, static_cast<std::uint16_t>(rule.field_index), age);
    }
    const std::int32_t clipped = clamp_i32(drive, -(16 << 16), 16 << 16);
    if (clipped > best_drive || (clipped == best_drive && cell_id < best_cell)) {
      best_drive = clipped;
      best_cell = cell_id;
    }
  }
  return best_cell;
}

// Every non-proposal records one explicit refusal reason.
__device__ __forceinline__ void refuse_growth(RouteMutationProposal* proposals,
                                              std::uint32_t* costs,
                                              ResidentDevelopmentCounters* counters,
                                              std::uint32_t node_index,
                                              const RouteMutationProposal& proposal,
                                              GrowthRefusalKind kind) {
  proposals[node_index] = proposal;
  costs[node_index] = 0u;
  if (counters != nullptr) atomicAdd(&counters->growth_refusals[kind], 1u);
}

__device__ void propose_node_impl(DirectBrain brain, std::uint32_t node_index,
                                  RouteMutationProposal* proposals,
                                  std::uint32_t* costs,
                                  std::uint64_t* target_claims,
                                  ResidentDevelopmentCounters* counters) {
  if (node_index >= brain.node_count || brain.development == nullptr) return;
  const DirectNode& node = brain.nodes[node_index];
  const std::uint32_t age = brain.development->age_tick;
  RouteMutationProposal proposal{};
  proposal.node = node_index;
  proposal.route_slot = kInvalidIndex;
  proposal.target = kInvalidIndex;
  proposal.recipe_cell = kInvalidIndex;
  proposal.construction_front_index = kInvalidIndex;
  proposals[node_index] = proposal;
  costs[node_index] = 0u;

  // First ask whether an actively harmful redundant route should be retired.
  if (node.active_route_count > 2u) {
    std::int32_t worst = 0;
    std::uint32_t worst_slot = kInvalidIndex;
    for (std::uint32_t slot = 2u; slot < node.route_capacity; ++slot) {
      const DirectRoute& route = brain.routes[node.route_offset + slot];
      if (!route_is_active(route)) continue;
      const DirectNode& target = brain.nodes[route.target];
      std::int64_t score = route.last_credit_q16;
      score += route.developmental_score_q16 >> 2;
      score += node.activity_ema_q16 >> 3;
      score += target.attractor_support_q16 >> 4;
      score -= node.inhibition_q16 >> 3;
      score += bound_field_kind_influence_q16(
          brain, node.territory_index, node, age, DevelopmentFieldKind::repel);
      const std::int32_t bounded_score = clamp_i32(score, -(16 << 16), 16 << 16);
      bool choose = worst_slot == kInvalidIndex;
      if (!choose) {
        const std::uint32_t candidate_index = node.route_offset + slot;
        const std::uint32_t incumbent_index = node.route_offset + worst_slot;
        const bool candidate_memory = route_has_causal_memory(brain, candidate_index);
        const bool incumbent_memory = route_has_causal_memory(brain, incumbent_index);
        if (candidate_memory != incumbent_memory) {
          choose = !candidate_memory;
        } else if (candidate_memory) {
          choose = causal_route_more_expendable(
              brain, candidate_index, incumbent_index, age);
        } else {
          choose = bounded_score < worst ||
                   (bounded_score == worst && slot < worst_slot);
        }
      }
      if (choose) {
        worst = bounded_score;
        worst_slot = slot;
      }
    }
    const bool structural_pressure =
        substrate::direct_adult::device_explicit_interaction_pressure(
            brain.resource_ecology);
    if (worst_slot != kInvalidIndex &&
        (worst <= kRetractionScoreQ16 ||
         (node.maintenance_q16 < (1 << 14) && structural_pressure))) {
      proposal.route_slot = worst_slot;
      proposal.kind = kProposalRetract;
      proposal.recipe_cell = select_recipe_cell(brain, node, age, false);
      proposal.score_q16 = worst;
      proposals[node_index] = proposal;
      costs[node_index] = 0u;
      if (counters != nullptr) atomicAdd(&counters->retract_proposals_emitted, 1u);
      return;
    }
  }

  // Growth is front-local; this node-wide pass owns retraction only.
}

__global__ void propose_route_mutations_kernel(DirectBrain brain,
                                                RouteMutationProposal* proposals,
                                                std::uint32_t* costs,
                                                std::uint64_t* target_claims,
                                                ResidentDevelopmentCounters* counters) {
  propose_node_impl(brain, blockIdx.x * blockDim.x + threadIdx.x, proposals, costs,
                    target_claims, counters);
}

#define DIRECT_NETWORK_CONSTRUCTION_FRONT_IMPLEMENTATION
#include "direct_network_construction_fronts.inl"
#undef DIRECT_NETWORK_CONSTRUCTION_FRONT_IMPLEMENTATION
#define DIRECT_NETWORK_POSTBIRTH_CONSTRUCTOR_IMPLEMENTATION
#include "direct_network_postbirth_constructor_ecology.inl"
#undef DIRECT_NETWORK_POSTBIRTH_CONSTRUCTOR_IMPLEMENTATION

__device__ bool rematerialize_condensed_support_impl(
    DirectBrain brain, std::uint32_t derivation_index,
    const DirectExactHistoryRecord& contradiction) {
  if (brain.development == nullptr || brain.recipe_cells == nullptr ||
      brain.postbirth_derivations == nullptr ||
      brain.postbirth_constructor == nullptr ||
      derivation_index >= brain.postbirth_constructor->derivation_count ||
      contradiction.kind != DirectExactHistoryKind::recipe_revision ||
      contradiction.identity == 0u || contradiction.incarnation_before == 0u ||
      contradiction.incarnation_after == 0u || contradiction.resource_delta >= 0 ||
      contradiction.context >= brain.development->exact_history.committed_slots)
    return false;
  const auto& committed =
      brain.development->exact_history.records[contradiction.context];
  if (committed.identity != contradiction.identity ||
      committed.parent_identity != contradiction.parent_identity ||
      committed.kind != contradiction.kind ||
      committed.source != contradiction.source ||
      committed.resource_delta != contradiction.resource_delta)
    return false;
  const auto& derivation = brain.postbirth_derivations[derivation_index];
  if (derivation.recipe_cell >= brain.development->recipe_cell_count ||
      contradiction.source != derivation.recipe_cell)
    return false;
  const auto& compacted = brain.recipe_cells[derivation.recipe_cell];
  if (compacted.logical_recipe_id != derivation.logical_recipe_id ||
      compacted.revision_identity != derivation.revision_identity ||
      contradiction.parent_identity != compacted.revision_identity)
    return false;
  ResidentNetworkCondensationEvidence evidence{};
  const bool boundary_condensed =
      (derivation.condensation_flags &
       kResidentDerivationBoundaryCondensedNetwork) != 0u;
  std::uint32_t source_count = 0u;
  if (boundary_condensed) {
    source_count = derivation.boundary_condensation_root_count;
    if (source_count < 3u ||
        source_count > kResidentRematerializedSupportCapacity ||
        derivation.witness_identity == 0u)
      return false;
    for (std::uint32_t i = 0u; i < source_count; ++i) {
      const auto& source = derivation.boundary_condensation_sources[i];
      if (source.logical_recipe_id !=
              derivation.boundary_condensation_root_logical_recipe_ids[i] ||
          source.revision_identity !=
              derivation.boundary_condensation_root_revision_identities[i])
        return false;
    }
  } else {
    if (!rematerialize_resident_condensation(derivation, &evidence) ||
        evidence.source_count != kResidentRematerializedSupportMinimum)
      return false;
    source_count = evidence.source_count;
  }
  const auto prior = brain.development->refinement_rematerialization;
  if (prior.active != 0u &&
      prior.contradiction_identity == contradiction.identity)
    return false;
  ResidentRefinementRematerializationState candidate{};
  candidate.compacted_logical_recipe_id = compacted.logical_recipe_id;
  candidate.compacted_revision_identity = compacted.revision_identity;
  candidate.witness_identity = derivation.witness_identity;
  candidate.contradiction_identity = contradiction.identity;
  candidate.support_count = source_count;
  candidate.generation = prior.generation == 0xffffffffu
      ? 0xffffffffu : prior.generation + 1u;
  if (candidate.generation == 0xffffffffu) return false;
  candidate.active = 1u;
  for (std::uint32_t i = 0u; i < source_count; ++i) {
    const auto& source = boundary_condensed
        ? derivation.boundary_condensation_sources[i]
        : evidence.sources[i];
    auto& support = candidate.support[i];
    support.logical_recipe_id = source.logical_recipe_id;
    support.revision_identity = source.revision_identity;
    support.generation = source.generation;
    support.recipe_cell = source.recipe_cell;
    support.relation = source.relation;
    support.parameter_q16 = source.parameter_q16;
    support.input_node = source.input_port.node;
    support.output_node = source.output_port.node;
    support.input_arity = source.input_port.arity;
    support.output_arity = source.output_port.arity;
    support.input_domain = static_cast<std::uint8_t>(source.input_port.domain);
    support.input_direction =
        static_cast<std::uint8_t>(source.input_port.direction);
    support.output_domain = static_cast<std::uint8_t>(source.output_port.domain);
    support.output_direction =
        static_cast<std::uint8_t>(source.output_port.direction);
  }
  candidate.instruction_identity =
      resident_refinement_instruction_identity(candidate);
  candidate.transaction_identity =
      resident_refinement_transaction_identity(candidate);
  if (!resident_refinement_rematerialization_valid(candidate)) return false;
  brain.development->refinement_rematerialization = candidate;
  return true;
}

__device__ void filter_node_impl(const RouteMutationProposal* proposals,
                                 std::uint32_t* costs,
                                 const std::uint64_t* target_claims,
                                 std::uint32_t node_index,
                                 std::uint32_t node_count) {
  if (node_index >= node_count) return;
  const RouteMutationProposal proposal = proposals[node_index];
  if (proposal.kind != kProposalGrow || proposal.target >= node_count) return;
  const std::uint64_t claim = target_claims[proposal.target];
  if (static_cast<std::uint32_t>(claim) != node_index) costs[node_index] = 0u;
}

__global__ void filter_winning_growth_costs_kernel(const RouteMutationProposal* proposals,
                                                   std::uint32_t* costs,
                                                   const std::uint64_t* target_claims,
                                                   std::uint32_t node_count) {
  filter_node_impl(proposals, costs, target_claims,
                   blockIdx.x * blockDim.x + threadIdx.x, node_count);
}

enum class RecipeHistoryField : std::uint32_t {
  credit = 1u,
  support = 2u,
  last_active_age = 3u,
  neighbor_support = 4u,
};

__device__ void stage_recipe_commit_history_record(
    DirectExactHistoryRecord* record, std::uint64_t contributor_identity,
    std::uint32_t tick, std::uint32_t source_node, std::uint32_t source_recipe_cell,
    std::uint32_t target_recipe_cell, std::uint32_t edge_index,
    RecipeHistoryField field, std::int64_t prior_value,
    std::int64_t result_value, std::int64_t delta) {
  if (record == nullptr) return;
  std::uint64_t identity = exact_history_fold_word(0xcbf29ce484222325ull, contributor_identity);
  identity = exact_history_fold_word(identity, tick);
  identity = exact_history_fold_word(identity, source_node);
  identity = exact_history_fold_word(identity, source_recipe_cell);
  identity = exact_history_fold_word(identity, target_recipe_cell);
  identity = exact_history_fold_word(identity, edge_index);
  identity = exact_history_fold_word(identity, static_cast<std::uint32_t>(field));
  identity = exact_history_fold_word(identity, static_cast<std::uint64_t>(prior_value));
  identity = exact_history_fold_word(identity, static_cast<std::uint64_t>(result_value));
  identity = exact_history_fold_word(identity, static_cast<std::uint64_t>(delta));
  record->identity = identity;
  record->parent_identity = contributor_identity;
  record->resident_tick = tick;
  record->event_tick = tick;
  record->kind = DirectExactHistoryKind::recipe_commit;
  record->source = source_node;
  record->subject = source_recipe_cell;
  record->value = target_recipe_cell;
  record->context = edge_index;
  record->flags = static_cast<std::uint32_t>(field);
  record->incarnation_before = static_cast<std::uint64_t>(prior_value);
  record->incarnation_after = static_cast<std::uint64_t>(result_value);
  record->resource_delta = delta;
}

__device__ __forceinline__ std::int32_t recipe_neighbor_delta_q16(
    std::int32_t support_delta_q16, const ResidentRecipeEdge& edge) {
  const std::int64_t propagated =
      (static_cast<std::int64_t>(support_delta_q16) * edge.weight_q16) >> 16;
  return clamp_i32(propagated, -(1 << 13), 1 << 13);
}

__device__ __forceinline__ std::uint64_t resident_topology_occurrence_identity(
    DirectExactHistoryKind kind, std::uint32_t tick, std::uint32_t source,
    std::uint32_t route_index, std::uint32_t target, std::uint32_t flags,
    std::uint64_t incarnation_before, std::uint64_t incarnation_after,
    std::int64_t resource_delta) {
  std::uint64_t identity = 0x7265735f746f706full;
  identity = exact_history_fold_word(identity, static_cast<std::uint32_t>(kind));
  identity = exact_history_fold_word(identity, tick);
  identity = exact_history_fold_word(identity, source);
  identity = exact_history_fold_word(identity, route_index);
  identity = exact_history_fold_word(identity, target);
  identity = exact_history_fold_word(identity, flags);
  identity = exact_history_fold_word(identity, incarnation_before);
  identity = exact_history_fold_word(identity, incarnation_after);
  identity = exact_history_fold_word(identity, static_cast<std::uint64_t>(resource_delta));
  return identity == 0u ? 1u : identity;
}

__device__ __forceinline__ void stage_resident_topology_history_record(
    DirectExactHistoryRecord* record, DirectExactHistoryKind kind,
    std::uint32_t tick, std::uint32_t source, std::uint32_t route_index,
    std::uint32_t target, std::uint32_t flags, std::uint64_t incarnation_before,
    std::uint64_t incarnation_after, std::int64_t resource_delta) {
  stage_topology_history_record(record, kind, tick, source, route_index, target,
                                flags, incarnation_before, incarnation_after,
                                resource_delta);
  if (record != nullptr) {
    record->identity = resident_topology_occurrence_identity(
        kind, tick, source, route_index, target, flags, incarnation_before,
        incarnation_after, resource_delta);
  }
}

__device__ void commit_recipe_transaction_impl(
    DirectBrain brain, RouteMutationProposal* proposals,
    ResidentDevelopmentCounters* counters, std::uint32_t work_count) {
  if (brain.development == nullptr || proposals == nullptr ||
      brain.recipe_cells == nullptr ||
      (brain.recipe_edge_count != 0u && brain.recipe_edges == nullptr)) return;
  const std::uint32_t tick = brain.development->age_tick;
  DirectExactHistoryHotPage* history = &brain.development->exact_history;
  for (std::uint32_t proposal_index = 0u; proposal_index < work_count; ++proposal_index) {
    RouteMutationProposal& proposal = proposals[proposal_index];
    const std::uint32_t node_index = proposal.node;
    const std::int32_t support_delta_q16 = proposal.committed_recipe_support_q16;
    const std::uint64_t topology_identity = proposal.committed_topology_identity;
    proposal.committed_recipe_support_q16 = 0;
    proposal.committed_topology_identity = 0u;
    if (support_delta_q16 == 0 || topology_identity == 0u ||
        proposal.recipe_cell == kInvalidIndex ||
        node_index >= brain.node_count ||
        proposal.recipe_cell >= brain.development->recipe_cell_count ||
        proposal.route_slot >= brain.nodes[node_index].route_capacity) continue;
    ResidentRecipeCell& cell = brain.recipe_cells[proposal.recipe_cell];
    std::uint32_t width = 2u + (cell.last_active_age < tick ? 1u : 0u);
    for (std::uint32_t e = 0u; e < cell.edge_count; ++e) {
      const std::uint32_t edge_index = cell.edge_offset + e;
      if (edge_index >= brain.recipe_edge_count) break;
      const ResidentRecipeEdge& edge = brain.recipe_edges[edge_index];
      if (edge.target_cell < brain.development->recipe_cell_count &&
          recipe_neighbor_delta_q16(support_delta_q16, edge) != 0)
        width += 2u;
    }
    if (!begin_exact_history_phase(
            history, DirectExactHistoryKind::recipe_commit, width, tick)) return;
    std::uint32_t ordinal = 0u;
    const std::uint64_t contributor_identity = topology_identity;
    const std::uint32_t local_edge = kInvalidIndex;

    std::int64_t prior = cell.support_q16;
    std::int64_t result = prior + support_delta_q16;
    stage_recipe_commit_history_record(
        &history->records[history->phase_base + ordinal++], contributor_identity,
        tick, node_index, proposal.recipe_cell, proposal.recipe_cell, local_edge,
        RecipeHistoryField::support, prior, result, support_delta_q16);
    const std::uint64_t source_evidence =
        history->records[history->phase_base + ordinal - 1u].identity;
    stage_resident_recipe_revision_event(
        &history->records[history->phase_base + ordinal++], cell,
        proposal.recipe_cell, ResidentRecipeRevisionAuthority::structural,
        topology_identity, source_evidence, tick, tick, proposal.route_slot,
        proposal.recipe_cell, static_cast<std::uint32_t>(RecipeHistoryField::support),
        support_delta_q16);
    const std::uint64_t prior_revision_identity = cell.revision_identity;
    __threadfence();
    if (!apply_resident_recipe_revision_event(
            &cell, history->records[history->phase_base + ordinal - 1u],
            proposal.recipe_cell))
      continue;
    // Structural support revisions move the Recipe head without changing the
    // executable morphology. Keep resident derivation/incidence locators on
    // that same immutable revision head, but only from the exact parent that
    // was just committed. Older/stale derivations remain stale and fail closed.
    if (brain.postbirth_constructor != nullptr &&
        brain.postbirth_derivations != nullptr) {
      const std::uint32_t derivation_count =
          brain.postbirth_constructor->derivation_count;
      for (std::uint32_t d = 0u; d < derivation_count; ++d) {
        auto& derivation = brain.postbirth_derivations[d];
        if (derivation.recipe_cell != proposal.recipe_cell ||
            derivation.logical_recipe_id != cell.logical_recipe_id ||
            derivation.revision_identity != prior_revision_identity)
          continue;
        derivation.revision_identity = cell.revision_identity;
      }
      if (brain.postbirth_constructor->recipe_incidence_count <=
          kResidentRecipeIncidenceCapacity)
        for (std::uint32_t i = 0u;
             i < brain.postbirth_constructor->recipe_incidence_count; ++i) {
          auto& incidence = brain.postbirth_constructor->recipe_incidence[i];
          if (incidence.derivation_index >= derivation_count ||
              incidence.revision_identity != prior_revision_identity)
            continue;
          const auto& derivation =
              brain.postbirth_derivations[incidence.derivation_index];
          if (derivation.recipe_cell == proposal.recipe_cell &&
              derivation.logical_recipe_id == cell.logical_recipe_id &&
              derivation.revision_identity == cell.revision_identity)
            incidence.revision_identity = cell.revision_identity;
        }
    }

    if (cell.last_active_age < tick) {
      const std::uint32_t prior_age = cell.last_active_age;
      stage_recipe_commit_history_record(
          &history->records[history->phase_base + ordinal++], contributor_identity,
          tick, node_index, proposal.recipe_cell, proposal.recipe_cell, local_edge,
          RecipeHistoryField::last_active_age, prior_age, tick,
          static_cast<std::int64_t>(tick) - prior_age);
      __threadfence();
      cell.last_active_age = tick;
    }
    if (counters != nullptr) ++counters->recipe_updates;

    for (std::uint32_t e = 0u; e < cell.edge_count; ++e) {
      const std::uint32_t edge_index = cell.edge_offset + e;
      if (edge_index >= brain.recipe_edge_count) break;
      const ResidentRecipeEdge& edge = brain.recipe_edges[edge_index];
      if (edge.target_cell >= brain.development->recipe_cell_count) continue;
      const std::int32_t delta = recipe_neighbor_delta_q16(support_delta_q16, edge);
      if (delta == 0) continue;
      ResidentRecipeCell& target = brain.recipe_cells[edge.target_cell];
      prior = target.support_q16;
      result = prior + delta;
      stage_recipe_commit_history_record(
          &history->records[history->phase_base + ordinal++], contributor_identity,
          tick, node_index, proposal.recipe_cell, edge.target_cell, edge_index,
          RecipeHistoryField::neighbor_support, prior, result, delta);
      const std::uint64_t target_evidence =
          history->records[history->phase_base + ordinal - 1u].identity;
      stage_resident_recipe_revision_event(
          &history->records[history->phase_base + ordinal++], target,
          edge.target_cell, ResidentRecipeRevisionAuthority::structural,
          topology_identity, target_evidence, tick, tick, edge_index, e,
          static_cast<std::uint32_t>(RecipeHistoryField::neighbor_support), delta);
      __threadfence();
      apply_resident_recipe_revision_event(
          &target, history->records[history->phase_base + ordinal - 1u],
          edge.target_cell);
      if (counters != nullptr) ++counters->recipe_neighbor_updates;
    }
    finish_exact_history_phase(history);
  }
}

__device__ __forceinline__ void count_commit_drop(ResidentDevelopmentCounters* counters,
                                                  CommitDropKind kind) {
  if (counters != nullptr) atomicAdd(&counters->commit_drops[kind], 1u);
}

__device__ void prepare_topology_plan_node(
    DirectBrain brain, std::uint32_t node_index,
    RouteMutationProposal* proposals, std::uint32_t* costs,
    const std::uint32_t* growth_cost_prefix,
    const std::uint64_t* reserve_snapshot,
    const std::uint64_t* target_claims,
    const std::uint32_t* route_causal_pin_bits,
    ResidentDevelopmentCounters* counters, TopologyCommitPlan* plan) {
  if (node_index >= brain.node_count || brain.development == nullptr) return;
  RouteMutationProposal& proposal = proposals[node_index];
  proposal.planned_kind = kPlannedNone;
  costs[node_index] = 0u;
  if (proposal.kind == kProposalNone) return;
  if (plan != nullptr) atomicAdd(&plan->proposal_count, 1u);
  if (proposal.route_slot >= brain.nodes[node_index].route_capacity) {
    if (proposal.kind == kProposalGrow) count_commit_drop(counters, kDroppedSlotInvalid);
    else if (proposal.kind == kProposalRetract && counters != nullptr)
      atomicAdd(&counters->retract_outcomes[kRetractDroppedSlotInvalid], 1u);
    return;
  }
  DirectNode& node = brain.nodes[node_index];
  const std::uint32_t route_index = node.route_offset + proposal.route_slot;
  DirectRoute& route = brain.routes[route_index];
  auto* ecology = brain.resource_ecology;
  auto* pool = substrate::direct_adult::direct_ecology_pool(
      ecology, substrate::direct_adult::DirectResourcePoolKind::explicit_interaction);

  if (proposal.kind == kProposalRetract) {
    if (route_causal_pin_bits != nullptr &&
        (route_causal_pin_bits[route_index >> 5] & (1u << (route_index & 31u))) != 0u) {
      if (counters != nullptr)
        atomicAdd(&counters->retract_outcomes[kRetractDroppedCausalPin], 1u);
      return;
    }
    const bool pressure_retraction = proposal.score_q16 > kRetractionScoreQ16;
    if (pressure_retraction &&
        !substrate::direct_adult::device_explicit_interaction_pressure(ecology)) {
      if (counters != nullptr)
        atomicAdd(&counters->retract_outcomes[kRetractDroppedNoPressure], 1u);
      return;
    }
    if (!route_is_active(route) || node.active_route_count <= 2u) {
      if (counters != nullptr)
        atomicAdd(&counters->retract_outcomes[kRetractDroppedInactive], 1u);
      return;
    }
    if (pool == nullptr || brain.route_incarnations == nullptr) return;
    proposal.planned_kind = pressure_retraction
                                ? kPlannedPressureRetract
                                : kPlannedRetract;
    if (!pressure_retraction && plan != nullptr)
      atomicAdd(&plan->normal_retraction_count, 1u);
    costs[node_index] = pressure_retraction ? 1u : 0u;
    return;
  }

  if (proposal.kind != kProposalGrow) return;
  if (route_is_active(route)) {
    count_commit_drop(counters, kDroppedSlotOccupied);
    return;
  }
  if (proposal.target >= brain.node_count) {
    count_commit_drop(counters, kDroppedTargetInvalid);
    return;
  }
  const std::uint64_t claim = target_claims[proposal.target];
  if (static_cast<std::uint32_t>(claim) != node_index) {
    count_commit_drop(counters, kDroppedLostTargetRace);
    return;
  }
  if (brain.nodes[proposal.target].active_in_degree >= kResidentMaximumInDegree) {
    count_commit_drop(counters, kDroppedTargetInDegreeFull);
    return;
  }
  const std::uint64_t prefix_cost =
      static_cast<std::uint64_t>(growth_cost_prefix[node_index]) + proposal.cost;
  if (prefix_cost > *reserve_snapshot) {
    count_commit_drop(counters, kDroppedBudgetExhausted);
    return;
  }
  if (ecology == nullptr || pool == nullptr || brain.route_incarnations == nullptr) {
    count_commit_drop(counters, kDroppedWorkBudgetExhausted);
    return;
  }
  const std::uint64_t available = pool->charged_units >= pool->live_units
                                      ? pool->charged_units - pool->live_units
                                      : 0u;
  if (prefix_cost > available || prefix_cost > pool->reserved_units) {
    proposal.planned_kind = kPlannedRejectedCapacity;
    return;
  }
  proposal.planned_kind = kPlannedGrow;
  if (plan != nullptr) {
    atomicAdd(&plan->growth_count, 1u);
    atomicAdd(reinterpret_cast<unsigned long long*>(&plan->growth_cost),
              static_cast<unsigned long long>(proposal.cost));
  }
}

__global__ void prepare_topology_plan_kernel(
    DirectBrain brain, RouteMutationProposal* proposals, std::uint32_t* costs,
    const std::uint32_t* growth_cost_prefix,
    const std::uint64_t* reserve_snapshot,
    const std::uint64_t* target_claims,
    const std::uint32_t* route_causal_pin_bits,
    ResidentDevelopmentCounters* counters, TopologyCommitPlan* plan) {
  prepare_topology_plan_node(
      brain, blockIdx.x * blockDim.x + threadIdx.x, proposals, costs,
      growth_cost_prefix, reserve_snapshot, target_claims,
      route_causal_pin_bits, counters, plan);
}

__device__ void finalize_topology_plan_node(
    DirectBrain brain, std::uint32_t node_index,
    RouteMutationProposal* proposals, std::uint32_t* costs,
    const std::uint32_t* pressure_prefix,
    ResidentDevelopmentCounters* counters, TopologyCommitPlan* plan) {
  if (node_index >= brain.node_count) return;
  RouteMutationProposal& proposal = proposals[node_index];
  if (proposal.planned_kind == kPlannedPressureRetract) {
    auto* pool = substrate::direct_adult::direct_ecology_pool(
        brain.resource_ecology,
        substrate::direct_adult::DirectResourcePoolKind::explicit_interaction);
    std::uint64_t needed = 0u;
    if (pool != nullptr && pool->capacity_units >= pool->charged_units) {
      const std::uint64_t available =
          pool->capacity_units - pool->charged_units + pool->reserved_units;
      const std::uint64_t required = pool->capacity_units / 10u + 1u;
      const std::uint64_t before_growth =
          available + plan->normal_retraction_count;
      const std::uint64_t ordinary_available =
          before_growth > plan->growth_count
              ? before_growth - plan->growth_count
              : 0u;
      needed = required > ordinary_available ? required - ordinary_available : 0u;
    }
    if (pressure_prefix[node_index] >= needed) {
      proposal.planned_kind = kPlannedNone;
      if (counters != nullptr)
        atomicAdd(&counters->retract_outcomes[kRetractDroppedNoPressure], 1u);
    }
  }
  const bool accepted = proposal.planned_kind == kPlannedGrow ||
                        proposal.planned_kind == kPlannedRetract ||
                        proposal.planned_kind == kPlannedPressureRetract;
  costs[node_index] = accepted ? 1u : 0u;
  if (!accepted || plan == nullptr) return;
  atomicAdd(&plan->accepted_count, 1u);
  if (proposal.planned_kind != kPlannedGrow) {
    atomicAdd(&plan->retraction_count, 1u);
    if (proposal.planned_kind == kPlannedPressureRetract)
      atomicAdd(&plan->pressure_retraction_count, 1u);
  }
}

__global__ void finalize_topology_plan_kernel(
    DirectBrain brain, RouteMutationProposal* proposals,
    std::uint32_t* costs, const std::uint32_t* pressure_prefix,
    ResidentDevelopmentCounters* counters, TopologyCommitPlan* plan) {
  finalize_topology_plan_node(brain, blockIdx.x * blockDim.x + threadIdx.x,
                              proposals, costs, pressure_prefix, counters, plan);
}

__device__ void admit_topology_work_node(
    DirectBrain brain, std::uint32_t node_index,
    RouteMutationProposal* proposals, std::uint32_t* costs,
    const std::uint32_t* work_prefix,
    ResidentDevelopmentCounters* counters, TopologyCommitPlan* plan) {
  if (node_index >= brain.node_count || plan == nullptr || costs[node_index] == 0u) return;
  RouteMutationProposal& proposal = proposals[node_index];
  const std::uint32_t planned_kind = proposal.planned_kind;
  if (planned_kind != kPlannedGrow && planned_kind != kPlannedRetract &&
      planned_kind != kPlannedPressureRetract) return;
  auto* ecology = brain.resource_ecology;
  const std::uint64_t work_end =
      static_cast<std::uint64_t>(work_prefix[node_index]) + costs[node_index];
  if (ecology != nullptr && work_end <= ecology->churn.mutation_budget) return;

  proposal.planned_kind = kPlannedRejectedWork;
  costs[node_index] = 0u;
  atomicSub(&plan->accepted_count, 1u);
  if (planned_kind == kPlannedGrow) {
    atomicSub(&plan->growth_count, 1u);
    atomicAdd(reinterpret_cast<unsigned long long*>(&plan->growth_cost),
              static_cast<unsigned long long>(-static_cast<long long>(proposal.cost)));
  } else {
    atomicSub(&plan->retraction_count, 1u);
    if (planned_kind == kPlannedPressureRetract) atomicSub(&plan->pressure_retraction_count, 1u);
    else atomicSub(&plan->normal_retraction_count, 1u);
    if (counters != nullptr)
      atomicAdd(&counters->retract_outcomes[kRetractRefusedWorkBudget], 1u);
  }
  count_commit_drop(counters, kDroppedWorkBudgetExhausted);
  if (ecology != nullptr) {
    atomicAdd(reinterpret_cast<unsigned long long*>(&ecology->churn.deferred), 1ULL);
    atomicAdd(reinterpret_cast<unsigned long long*>(&ecology->work.deferred_work), 1ULL);
    atomicOr(&ecology->pressure_flags,
             static_cast<unsigned int>(substrate::direct_adult::kDirectPressureWork));
  }
}

__global__ void admit_topology_work_kernel(
    DirectBrain brain, RouteMutationProposal* proposals, std::uint32_t* costs,
    const std::uint32_t* work_prefix, ResidentDevelopmentCounters* counters,
    TopologyCommitPlan* plan) {
  admit_topology_work_node(brain, blockIdx.x * blockDim.x + threadIdx.x, proposals,
                           costs, work_prefix, counters, plan);
}

__device__ void begin_topology_transaction(DirectBrain brain,
                                           TopologyCommitPlan* plan) {
  if (plan == nullptr || brain.development == nullptr) return;
  plan->admitted = 0u;
  auto* pool = substrate::direct_adult::direct_ecology_pool(
      brain.resource_ecology,
      substrate::direct_adult::DirectResourcePoolKind::explicit_interaction);
  if (pool == nullptr || pool->live_units > pool->charged_units ||
      pool->reserved_units > pool->charged_units ||
      plan->growth_count > pool->charged_units - pool->live_units ||
      plan->growth_count > pool->reserved_units ||
      plan->retraction_count > pool->live_units ||
      plan->retraction_count > pool->charged_units - pool->reserved_units) return;
  if (!begin_exact_history_phase(
          &brain.development->exact_history,
          DirectExactHistoryKind::topology_growth,
          plan->accepted_count, brain.development->age_tick)) return;
  substrate::direct_adult::device_record_churn_proposal(
      brain.resource_ecology, plan->proposal_count);
  plan->admitted = 1u;
}

__global__ void begin_topology_transaction_kernel(
    DirectBrain brain, TopologyCommitPlan* plan) {
  if (blockIdx.x != 0u || threadIdx.x != 0u) return;
  begin_topology_transaction(brain, plan);
}

__device__ __forceinline__ void apply_committed_recipe_chemotype(
    DirectBrain brain, DirectNode& node, std::uint32_t recipe_cell) {
  if (recipe_cell == kInvalidIndex || recipe_cell >= brain.development->recipe_cell_count ||
      brain.recipe_cells == nullptr || brain.resident_rules == nullptr) return;
  const ResidentRecipeCell& cell = brain.recipe_cells[recipe_cell];
  if (cell.rule_index >= brain.resident_rule_count) return;
  const ResidentConstructorRule& rule = brain.resident_rules[cell.rule_index];
  node.chemotype = apply_resident_chemotype_write_device(node.chemotype, rule);
}

__device__ void commit_node_impl(DirectBrain brain, std::uint32_t node_index,
                                 RouteMutationProposal* proposals,
                                 const std::uint32_t* cost_prefix,
                                 const std::uint64_t* reserve_snapshot,
                                 const std::uint64_t* target_claims,
                                 const std::uint32_t* route_causal_pin_bits,
                                 ResidentDevelopmentCounters* counters,
                                 DirectExactHistoryRecord* history_record = nullptr,
                                 bool resource_preflighted = false) {
  const std::uint32_t proposal_index = node_index;
  if (proposal_index >= brain.node_count || brain.development == nullptr) return;
  const RouteMutationProposal proposal = proposals[proposal_index];
  node_index = proposal.node;
  if (node_index >= brain.node_count) return;
  if (proposal.kind == kProposalNone) return;
  if (!resource_preflighted &&
      proposal.route_slot >= brain.nodes[node_index].route_capacity) {
    if (proposal.kind == kProposalGrow) {
      count_commit_drop(counters, kDroppedSlotInvalid);
    } else if (proposal.kind == kProposalRetract && counters != nullptr) {
      atomicAdd(&counters->retract_outcomes[kRetractDroppedSlotInvalid], 1u);
    }
    return;
  }
  DirectNode& node = brain.nodes[node_index];
  DirectRoute& route = brain.routes[node.route_offset + proposal.route_slot];

  if (proposal.kind == kProposalRetract) {
    const std::uint32_t route_index = node.route_offset + proposal.route_slot;
    if (!resource_preflighted && route_causal_pin_bits != nullptr &&
        (route_causal_pin_bits[route_index >> 5] & (1u << (route_index & 31u))) != 0u) {
      if (counters != nullptr)
        atomicAdd(&counters->retract_outcomes[kRetractDroppedCausalPin], 1u);
      return;
    }
    if (!resource_preflighted && proposal.score_q16 > kRetractionScoreQ16 &&
        !substrate::direct_adult::device_explicit_interaction_pressure(
            brain.resource_ecology)) {
      if (counters != nullptr)
        atomicAdd(&counters->retract_outcomes[kRetractDroppedNoPressure], 1u);
      return;
    }
    if (!resource_preflighted &&
        (!route_is_active(route) || node.active_route_count <= 2u)) {
      if (counters != nullptr) atomicAdd(&counters->retract_outcomes[kRetractDroppedInactive], 1u);
      return;
    }
    // Uncommit live matter here; the arena slot remains physically charged.
    const bool pressure_retraction = proposal.score_q16 > kRetractionScoreQ16;
    const std::uint64_t incarnation_before = brain.route_incarnations == nullptr
                                                  ? 0u
                                                  : brain.route_incarnations[route_index];
    const std::uint32_t route_cost = decode_route_construction_cost(route.flags);
    stage_resident_topology_history_record(
        history_record, DirectExactHistoryKind::topology_retraction,
        brain.development->age_tick, node_index, route_index, route.target,
        route.flags, incarnation_before,
        incarnation_before + kRouteIncarnationStride, -static_cast<std::int64_t>(route_cost));
    __threadfence();
    const bool uncommitted = resource_preflighted
        ? substrate::direct_adult::device_uncommit_pool_units(
              brain.resource_ecology,
              substrate::direct_adult::DirectResourcePoolKind::explicit_interaction,
              route_cost)
        : pressure_retraction
        ? substrate::direct_adult::device_uncommit_explicit_interaction_under_pressure(
              brain.resource_ecology, route_cost)
        : substrate::direct_adult::device_uncommit_pool_units(
              brain.resource_ecology,
              substrate::direct_adult::DirectResourcePoolKind::explicit_interaction,
              route_cost);
    if (!uncommitted) {
      if (history_record != nullptr) *history_record = DirectExactHistoryRecord{};
      if (pressure_retraction && counters != nullptr)
        atomicAdd(&counters->retract_outcomes[kRetractDroppedNoPressure], 1u);
      return;
    }
    atomicAdd(&counters->ledger_uncommits, 1u);
    route.flags &= ~kRouteFlagActive;
    route.flags |= kRouteFlagDevelopmentalReserve;
    if (brain.route_incarnations != nullptr)
      brain.route_incarnations[node.route_offset + proposal.route_slot] +=
          kRouteIncarnationStride;
    route.eligibility_q16 = 0;
    route.eligibility_context = kInvalidIndex;
    route.eligibility_expires = 0u;
    route.last_credit_q16 = 0;
    --node.active_route_count;
    if (route.target < brain.node_count) {
      atomicSub(&brain.nodes[route.target].active_in_degree, 1u);
    }
    atomicAdd(reinterpret_cast<unsigned long long*>(&brain.development->constructor_reserve), route_cost);
    atomicAdd(reinterpret_cast<unsigned long long*>(&brain.development->reclaimed_resource), route_cost);
    atomicAdd(reinterpret_cast<unsigned long long*>(&brain.development->live_route_matter),
              static_cast<unsigned long long>(-static_cast<long long>(route_cost)));
    atomicAdd(&counters->retracted_routes, 1u);
    atomicAdd(reinterpret_cast<unsigned long long*>(&counters->resource_reclaimed), route_cost);
    substrate::direct_adult::device_record_churn_commit(brain.resource_ecology, 1u);
    atomicAdd(reinterpret_cast<unsigned long long*>(&brain.resource_ecology->churn.admitted), 1ULL);
    apply_committed_recipe_chemotype(brain, node, proposal.recipe_cell);
    const std::int32_t recipe_support = clamp_i32(
        -(static_cast<std::int64_t>(proposal.score_q16) >> 3), 1 << 8, 1 << 14);
    proposals[proposal_index].committed_recipe_support_q16 = recipe_support;
    proposals[proposal_index].committed_topology_identity =
        history_record == nullptr ? 0u : history_record->identity;
    if (counters != nullptr) atomicAdd(&counters->retract_outcomes[kRetractCommitted], 1u);
    return;
  }

  if (proposal.kind != kProposalGrow) return;
  substrate::direct_adult::DirectResourceEcologyState* ecology = brain.resource_ecology;
  if (!resource_preflighted) {
    if (route_is_active(route)) {
      count_commit_drop(counters, kDroppedSlotOccupied);
      return;
    }
    if (proposal.target >= brain.node_count) {
      count_commit_drop(counters, kDroppedTargetInvalid);
      return;
    }
    const std::uint64_t claim = target_claims[proposal.target];
    if (static_cast<std::uint32_t>(claim) != node_index) {
      count_commit_drop(counters, kDroppedLostTargetRace);
      return;
    }
    if (brain.nodes[proposal.target].active_in_degree >= kResidentMaximumInDegree) {
      count_commit_drop(counters, kDroppedTargetInDegreeFull);
      return;
    }
    const std::uint64_t admitted = *reserve_snapshot;
    if (static_cast<std::uint64_t>(cost_prefix[node_index]) + proposal.cost > admitted) {
      count_commit_drop(counters, kDroppedBudgetExhausted);
      return;
    }
    if (ecology == nullptr) {
      count_commit_drop(counters, kDroppedWorkBudgetExhausted);
      return;
    }
    if (static_cast<std::uint64_t>(cost_prefix[node_index]) + proposal.cost >
        ecology->churn.mutation_budget) {
      atomicAdd(reinterpret_cast<unsigned long long*>(&ecology->churn.deferred), 1ull);
      atomicAdd(reinterpret_cast<unsigned long long*>(&ecology->work.deferred_work),
                static_cast<unsigned long long>(proposal.cost));
      count_commit_drop(counters, kDroppedWorkBudgetExhausted);
      atomicOr(&ecology->pressure_flags, static_cast<unsigned int>(
                                          substrate::direct_adult::kDirectPressureWork));
      return;
    }
  }

  // Physical pool admission is the final pre-write authority after the developmental budget.
  const std::uint64_t incarnation_before = brain.route_incarnations == nullptr
                                                ? 0u
                                                : brain.route_incarnations[
                                                      node.route_offset + proposal.route_slot];
  const std::uint32_t committed_route_flags = encode_route_construction_cost(
      encode_route_recipe_builder(proposal.route_flags, proposal.recipe_cell), proposal.cost);
  stage_resident_topology_history_record(
      history_record, DirectExactHistoryKind::topology_growth,
      brain.development->age_tick, node_index,
      node.route_offset + proposal.route_slot, proposal.target,
      committed_route_flags, incarnation_before,
      incarnation_before + kRouteIncarnationStride,
      static_cast<std::int64_t>(proposal.cost));
  __threadfence();
  if (!substrate::direct_adult::device_commit_pool_units(
          ecology,
          substrate::direct_adult::DirectResourcePoolKind::explicit_interaction,
          proposal.cost)) {
    if (history_record != nullptr) *history_record = DirectExactHistoryRecord{};
    atomicAdd(reinterpret_cast<unsigned long long*>(&ecology->churn.rejected_capacity), 1ull);
    if (counters != nullptr) atomicAdd(&counters->ledger_refusals, 1u);
    return;
  }

  const std::uint32_t pre_growth_active_route_count = node.active_route_count;
  bool compensatory_repair = false;
  if (proposal.recipe_cell < brain.development->recipe_cell_count) {
    const ResidentRecipeCell& cell = brain.recipe_cells[proposal.recipe_cell];
    if (cell.rule_index < brain.resident_rule_count) {
      const std::int32_t activity =
          node.activity_ema_q16 < 0 ? -node.activity_ema_q16 : node.activity_ema_q16;
      compensatory_repair =
          brain.resident_rules[cell.rule_index].opcode == RuleOpcode::repair &&
          bound_field_kind_influence_q16(
              brain, node.territory_index, node, brain.development->age_tick,
              DevelopmentFieldKind::repair) > 0 &&
          activity < (1 << 16) / 128 && pre_growth_active_route_count < 2u;
    }
  }
  if (brain.route_incarnations != nullptr)
    brain.route_incarnations[node.route_offset + proposal.route_slot] +=
        kRouteIncarnationStride;
  route.source = node_index;
  route.target = proposal.target;
  route.flags = committed_route_flags;
  route.delay = route_delay(node, brain.nodes[proposal.target],
                            (proposal.route_flags & kRouteFlagLongTract) != 0u);
  route.developmental_score_q16 = proposal.score_q16;
  route.conductance_q16 = clamp_i32(
      static_cast<std::int64_t>(kInitialGrownConductanceQ16) + (proposal.score_q16 >> 4),
      kMinConductanceQ16, 4 << 16);
  if (brain.retention_bank != nullptr) {
    const std::uint32_t route_index = node.route_offset + proposal.route_slot;
    substrate::direct_adult::DirectRetentionState fresh{};
    fresh.logical_source = node_index;
    fresh.logical_slot = route_index;
    fresh.logical_generation = brain.route_incarnations == nullptr
                                   ? 0u : brain.route_incarnations[route_index];
    fresh.source_revision = fresh.logical_generation;
    fresh.last_confirmed_conductance_q16 = route.conductance_q16;
    brain.retention_bank[route_index] = fresh;
  }
  route.eligibility_q16 = 0;
  route.last_credit_q16 = 0;
  route.eligibility_context = kInvalidIndex;
  route.eligibility_expires = 0u;
  route.last_credit_ticket = kNoCreditTicket;
  ++node.active_route_count;
  atomicAdd(&brain.nodes[proposal.target].active_in_degree, 1u);
  atomicAdd(reinterpret_cast<unsigned long long*>(&brain.development->constructor_reserve),
            static_cast<unsigned long long>(-static_cast<long long>(proposal.cost)));
  atomicAdd(reinterpret_cast<unsigned long long*>(&brain.development->live_route_matter),
            static_cast<unsigned long long>(proposal.cost));
  atomicAdd(&counters->grown_routes, 1u);
  atomicAdd(reinterpret_cast<unsigned long long*>(&counters->resource_spent),
            static_cast<unsigned long long>(proposal.cost));
  substrate::direct_adult::device_record_churn_commit(ecology, 1u);
  atomicAdd(reinterpret_cast<unsigned long long*>(&ecology->churn.admitted), 1ull);
  apply_committed_recipe_chemotype(brain, node, proposal.recipe_cell);
  const std::int32_t recipe_support = clamp_i32(
      static_cast<std::int64_t>(proposal.score_q16) >> 4, 1 << 8, 1 << 14);
  if (!compensatory_repair) {
    proposals[proposal_index].committed_recipe_support_q16 = recipe_support;
    proposals[proposal_index].committed_topology_identity =
        history_record == nullptr ? 0u : history_record->identity;
  }
}

__device__ void advance_committed_construction_fronts_impl(
    DirectBrain, RouteMutationProposal*, ResidentDevelopmentCounters*,
    std::uint32_t);

#define DIRECT_NETWORK_CONSTRUCTION_FRONT_COMMIT_IMPLEMENTATION
#include "direct_network_construction_fronts.inl"
#undef DIRECT_NETWORK_CONSTRUCTION_FRONT_COMMIT_IMPLEMENTATION

__device__ void commit_topology_node(
    DirectBrain brain, std::uint32_t node_index,
    RouteMutationProposal* proposals, const std::uint32_t* cost_prefix,
    const std::uint32_t* history_prefix,
    const std::uint64_t* reserve_snapshot, const std::uint64_t* target_claims,
    const std::uint32_t* route_causal_pin_bits,
    ResidentDevelopmentCounters* counters, const TopologyCommitPlan* plan) {
  if (node_index >= brain.node_count || plan == nullptr || plan->admitted == 0u) return;
  const RouteMutationProposal& proposal = proposals[node_index];
  if (proposal.planned_kind == kPlannedRejectedWork) return;
  if (proposal.planned_kind == kPlannedRejectedCapacity) {
    atomicAdd(reinterpret_cast<unsigned long long*>(
                  &brain.resource_ecology->churn.rejected_capacity), 1ull);
    if (counters != nullptr) atomicAdd(&counters->ledger_refusals, 1u);
    return;
  }
  if (proposal.planned_kind != kPlannedGrow &&
      proposal.planned_kind != kPlannedRetract &&
      proposal.planned_kind != kPlannedPressureRetract) return;
  DirectExactHistoryRecord* record =
      &brain.development->exact_history.records[
          brain.development->exact_history.phase_base + history_prefix[node_index]];
  commit_node_impl(brain, node_index, proposals, cost_prefix, reserve_snapshot,
                   target_claims, route_causal_pin_bits, counters, record, true);
}

__device__ void advance_committed_construction_fronts_impl(
    DirectBrain brain, RouteMutationProposal* proposals,
    ResidentDevelopmentCounters* counters, std::uint32_t work_count) {
  for (std::uint32_t item = 0u; item < work_count; ++item) {
    const RouteMutationProposal& proposal = proposals[item];
    if (proposal.planned_kind != kPlannedGrow ||
        proposal.construction_front_index == kInvalidIndex) continue;
    const std::uint32_t node = proposal.node;
    if (node >= brain.node_count) continue;
    const DirectRoute& route =
        brain.routes[brain.nodes[node].route_offset + proposal.route_slot];
    if (!route_is_active(route) || route.source != node ||
        route.target != proposal.target) continue;
    advance_construction_front_after_commit(brain, proposal);
    if (counters != nullptr) ++counters->construction_front_successions;
  }
}

__global__ void commit_route_mutations_kernel(DirectBrain brain,
                                               RouteMutationProposal* proposals,
                                               const std::uint32_t* cost_prefix,
                                               const std::uint32_t* history_prefix,
                                               const std::uint64_t* reserve_snapshot,
                                               const std::uint64_t* target_claims,
                                               const std::uint32_t* route_causal_pin_bits,
                                               ResidentDevelopmentCounters* counters,
                                               const TopologyCommitPlan* plan) {
  commit_topology_node(brain, blockIdx.x * blockDim.x + threadIdx.x, proposals,
                       cost_prefix, history_prefix, reserve_snapshot, target_claims,
                       route_causal_pin_bits, counters, plan);
}

__global__ void advance_committed_construction_fronts_kernel(
    DirectBrain brain, RouteMutationProposal* proposals,
    ResidentDevelopmentCounters* counters) {
  if (blockIdx.x != 0u || threadIdx.x != 0u) return;
  advance_committed_construction_fronts_impl(
      brain, proposals, counters, brain.node_count);
}

__global__ void commit_compact_construction_fronts_kernel(
    DirectBrain brain, RouteMutationProposal* proposals, std::uint32_t* costs,
    const std::uint64_t* reserve_snapshot, ResidentDevelopmentCounters* counters) {
  if (blockIdx.x != 0u || threadIdx.x != 0u) return;
  commit_compact_construction_fronts_impl(
      brain, proposals, costs, reserve_snapshot, counters);
}

__device__ void finish_topology_transaction(DirectBrain brain,
                                            TopologyCommitPlan* plan) {
  if (plan == nullptr || plan->admitted == 0u || brain.development == nullptr) return;
  finish_exact_history_phase(&brain.development->exact_history);
  plan->admitted = 0u;
}

__global__ void finish_topology_transaction_kernel(DirectBrain brain,
                                                    TopologyCommitPlan* plan) {
  if (blockIdx.x != 0u || threadIdx.x != 0u) return;
  finish_topology_transaction(brain, plan);
}

__global__ void commit_recipe_transaction_kernel(
    DirectBrain brain, RouteMutationProposal* proposals,
    ResidentDevelopmentCounters* counters) {
  if (blockIdx.x != 0u || threadIdx.x != 0u) return;
  commit_recipe_transaction_impl(brain, proposals, counters, brain.node_count);
}

__device__ void finish_resident_admission_epoch_impl(
    DirectBrain brain, const ResidentDevelopmentCounters* counters) {
  if (brain.development == nullptr || counters == nullptr) return;
  ResidentAdultAdmissionState& admission = brain.development->adult_admission;
  ++admission.development_epochs;
  admission.field_update_events += counters->field_updates;
  admission.structural_revision_events +=
      static_cast<std::uint64_t>(counters->grown_routes) +
      counters->retracted_routes + counters->matured_nodes +
      counters->recipe_updates + counters->recipe_neighbor_updates +
      counters->postbirth_recipes_condensed;
  // Age cannot admit. A resident sweep must have physically revised the
  // complete population through local field/maturation law.
  if (admission.earned == 0u && brain.node_count != 0u &&
      admission.development_epochs != 0u &&
      admission.field_update_events >= brain.node_count) {
    admission.admission_tick = brain.development->age_tick;
    admission.earned = 1u;
  }
}

__global__ void finish_resident_admission_epoch_kernel(
    DirectBrain brain, const ResidentDevelopmentCounters* counters) {
  finish_resident_admission_epoch_impl(brain, counters);
}

#include "direct_network_exact_history_maintenance.inl"

// Clear deterministic target claims, counters and the transaction plan in one launch.
__global__ void clear_resident_development_epoch_state_kernel(
    std::uint64_t* target_claims, std::uint32_t node_count,
    ResidentDevelopmentCounters* counters, TopologyCommitPlan* plan) {
  const std::uint32_t stride = blockDim.x * gridDim.x;
  for (std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < node_count; i += stride) {
    target_claims[i] = 0xffffffffffffffffULL;
  }
  if (blockIdx.x == 0u && threadIdx.x == 0u) {
    *counters = ResidentDevelopmentCounters{};
    if (plan != nullptr) *plan = TopologyCommitPlan{};
  }
}

}  // namespace

__global__ void rematerialize_condensed_support_kernel(
    DirectBrain brain, std::uint32_t derivation_index,
    DirectExactHistoryRecord contradiction) {
  if (blockIdx.x == 0u && threadIdx.x == 0u)
    rematerialize_condensed_support_impl(brain, derivation_index, contradiction);
}

__device__ void resident_maintenance_plan_node(
    DirectBrain brain, const std::uint64_t* opportunity_incarnations,
    const std::uint32_t* causal_pins, std::int32_t cost_per_route_q16,
    std::uint32_t* candidate_counts, TopologyCommitPlan* plan,
    direct_adult_core::AdultCoreMetrics* metrics, std::uint32_t node_index) {
  plan_maintenance_transaction_impl(
      brain, opportunity_incarnations, causal_pins, cost_per_route_q16,
      candidate_counts, plan, metrics, node_index);
}

__device__ void resident_maintenance_plan_weight_node(
    DirectBrain brain, const std::uint64_t* opportunity_incarnations,
    const std::uint32_t* causal_pins, std::int32_t cost_per_route_q16,
    const TopologyCommitPlan* plan, std::uint32_t* candidate_counts,
    std::uint32_t node_index) {
  plan_homeostatic_weight_transaction_impl(
      brain, opportunity_incarnations, causal_pins, cost_per_route_q16,
      plan, candidate_counts, node_index);
}

__device__ void resident_maintenance_begin_transaction(
    DirectBrain brain, TopologyCommitPlan* plan,
    std::uint32_t current_tick) {
  begin_maintenance_transaction_impl(brain, plan, current_tick);
}

__device__ void resident_maintenance_begin_weight_transaction(
    DirectBrain brain, const std::uint32_t* candidate_counts,
    const std::uint32_t* candidate_prefix, std::uint32_t node_count,
    TopologyCommitPlan* plan, std::uint32_t current_tick) {
  begin_homeostatic_weight_transaction_impl(
      brain, candidate_counts, candidate_prefix, node_count, plan, current_tick);
}

__device__ void resident_maintenance_commit_node(
    DirectBrain brain, const std::uint64_t* opportunity_incarnations,
    const std::uint32_t* causal_pins, const std::uint32_t* candidate_prefix,
    const TopologyCommitPlan* plan,
    direct_adult_core::AdultCoreMetrics* metrics, std::uint32_t node_index) {
  commit_maintenance_transaction_impl(
      brain, opportunity_incarnations, causal_pins, candidate_prefix, plan,
      metrics, node_index);
}

__device__ void resident_maintenance_decay_node(
    DirectBrain brain, const std::uint64_t* opportunity_incarnations,
    const std::uint32_t* causal_pins, std::int32_t cost_per_route_q16,
    const std::uint32_t* candidate_prefix, const TopologyCommitPlan* plan,
    std::uint32_t node_index) {
  decay_maintenance_routes_impl(
      brain, opportunity_incarnations, causal_pins, cost_per_route_q16,
      candidate_prefix, plan, node_index);
}

__device__ void resident_development_advance_epoch(ResidentDevelopmentState* state,
                                                   std::uint64_t* reserve_snapshot) {
  advance_development_state_impl(state, reserve_snapshot);
}

__device__ void resident_development_mature_node(DirectBrain brain,
                                                 std::uint32_t node_index,
                                                 ResidentDevelopmentCounters* counters) {
  mature_node_impl(brain, node_index, counters);
}

__device__ void resident_development_propose_node(
    DirectBrain brain, std::uint32_t node_index, RouteMutationProposal* proposals,
    std::uint32_t* costs, std::uint64_t* target_claims,
    ResidentDevelopmentCounters* counters) {
  propose_node_impl(brain, node_index, proposals, costs, target_claims, counters);
}

__device__ void resident_development_nominate_committed_retractions(
    DirectBrain brain, RouteMutationProposal* proposals,
    ResidentDevelopmentCounters* counters) {
  (void)proposals;
  nominate_committed_retractions_impl(brain, counters);
}

__device__ void resident_development_advance_committed_fronts(
    DirectBrain brain, RouteMutationProposal* proposals,
    ResidentDevelopmentCounters* counters) {
  advance_committed_construction_fronts_impl(
      brain, proposals, counters, brain.node_count);
}

__device__ void resident_development_commit_construction_fronts(
    DirectBrain brain, RouteMutationProposal* proposals, std::uint32_t* costs,
    const std::uint64_t* reserve_snapshot, ResidentDevelopmentCounters* counters) {
  commit_compact_construction_fronts_impl(
      brain, proposals, costs, reserve_snapshot, counters);
}

__device__ void resident_development_propose_front(
    DirectBrain brain, std::uint32_t front_index,
    RouteMutationProposal* proposals, std::uint32_t* costs,
    std::uint64_t* target_claims, ResidentDevelopmentCounters* counters) {
  propose_construction_front_impl(
      brain, front_index, proposals, costs, target_claims, counters);
}

__device__ void resident_development_filter_node(
    const RouteMutationProposal* proposals, std::uint32_t* costs,
    const std::uint64_t* target_claims, std::uint32_t node_index,
    std::uint32_t node_count) {
  filter_node_impl(proposals, costs, target_claims, node_index, node_count);
}

__device__ void resident_development_prepare_topology_node(
    DirectBrain brain, std::uint32_t node_index,
    RouteMutationProposal* proposals, std::uint32_t* costs,
    const std::uint32_t* growth_cost_prefix,
    const std::uint64_t* reserve_snapshot, const std::uint64_t* target_claims,
    const std::uint32_t* route_causal_pin_bits,
    ResidentDevelopmentCounters* counters, TopologyCommitPlan* plan) {
  prepare_topology_plan_node(brain, node_index, proposals, costs,
                             growth_cost_prefix, reserve_snapshot, target_claims,
                             route_causal_pin_bits, counters, plan);
}

__device__ void resident_development_finalize_topology_node(
    DirectBrain brain, std::uint32_t node_index,
    RouteMutationProposal* proposals, std::uint32_t* costs,
    const std::uint32_t* pressure_prefix,
    ResidentDevelopmentCounters* counters, TopologyCommitPlan* plan) {
  finalize_topology_plan_node(brain, node_index, proposals, costs,
                              pressure_prefix, counters, plan);
}

__device__ void resident_development_admit_topology_work_node(
    DirectBrain brain, std::uint32_t node_index,
    RouteMutationProposal* proposals, std::uint32_t* costs,
    const std::uint32_t* work_prefix,
    ResidentDevelopmentCounters* counters, TopologyCommitPlan* plan) {
  admit_topology_work_node(brain, node_index, proposals, costs, work_prefix, counters, plan);
}

__device__ void resident_development_begin_topology_transaction(
    DirectBrain brain, TopologyCommitPlan* plan) {
  begin_topology_transaction(brain, plan);
}

__device__ void resident_development_commit_topology_node(
    DirectBrain brain, std::uint32_t node_index,
    RouteMutationProposal* proposals, const std::uint32_t* cost_prefix,
    const std::uint32_t* history_prefix,
    const std::uint64_t* reserve_snapshot, const std::uint64_t* target_claims,
    const std::uint32_t* route_causal_pin_bits,
    ResidentDevelopmentCounters* counters, const TopologyCommitPlan* plan) {
  commit_topology_node(brain, node_index, proposals, cost_prefix, history_prefix,
                       reserve_snapshot, target_claims, route_causal_pin_bits,
                       counters, plan);
}

__device__ void resident_development_finish_topology_transaction(
    DirectBrain brain, TopologyCommitPlan* plan) {
  finish_topology_transaction(brain, plan);
}

__device__ void resident_development_finish_admission_epoch(
    DirectBrain brain, const ResidentDevelopmentCounters* counters) {
  finish_resident_admission_epoch_impl(brain, counters);
}

__device__ void resident_development_commit_recipe_transaction(
    DirectBrain brain, RouteMutationProposal* proposals,
    ResidentDevelopmentCounters* counters) {
  commit_recipe_transaction_impl(brain, proposals, counters, brain.node_count);
}
__device__ void resident_development_condense_postbirth_recipe(DirectBrain brain, ResidentDevelopmentCounters* counters, std::uint32_t* cause_memo) { condense_postbirth_recipe_impl(brain, counters, cause_memo); }

__device__ void resident_development_commit_node(
    DirectBrain brain, std::uint32_t node_index,
    RouteMutationProposal* proposals, const std::uint32_t* cost_prefix,
    const std::uint64_t* reserve_snapshot, const std::uint64_t* target_claims,
    const std::uint32_t* route_causal_pin_bits, ResidentDevelopmentCounters* counters) {
  if (node_index < brain.node_count && proposals[node_index].kind != kProposalNone)
    substrate::direct_adult::device_record_churn_proposal(brain.resource_ecology, 1u);
  commit_node_impl(brain, node_index, proposals, cost_prefix, reserve_snapshot,
                   target_claims, route_causal_pin_bits, counters);
}

std::size_t resident_development_workspace_bytes(std::uint32_t node_capacity) {
  if (node_capacity == 0u) return 0u;
  std::size_t scan_storage_bytes = 0u;
  check_cuda(cub::DeviceScan::ExclusiveSum(
                 nullptr, scan_storage_bytes,
                 static_cast<const std::uint32_t*>(nullptr),
                 static_cast<std::uint32_t*>(nullptr), node_capacity),
             "query resident scan storage");
  const std::uint32_t blocks = std::max(256u, (node_capacity + 255u) / 256u);
  return sizeof(RouteMutationProposal) * node_capacity +
         sizeof(std::uint32_t) * node_capacity * 3u +
         sizeof(std::uint64_t) * node_capacity +
         sizeof(ResidentDevelopmentCounters) + sizeof(TopologyCommitPlan) +
         sizeof(std::uint64_t) +
         sizeof(std::uint32_t) * blocks * 2u + scan_storage_bytes;
}

ResidentDevelopmentWorkspace create_resident_development_workspace(std::uint32_t node_capacity) {
  if (node_capacity == 0u) throw std::invalid_argument("resident development requires nodes");
  ResidentDevelopmentWorkspace workspace{};
  workspace.node_capacity = node_capacity;
  check_cuda(cudaMalloc(&workspace.proposals, sizeof(RouteMutationProposal) * node_capacity),
             "allocate resident proposals");
  check_cuda(cudaMalloc(&workspace.costs, sizeof(std::uint32_t) * node_capacity),
             "allocate resident costs");
  check_cuda(cudaMalloc(&workspace.cost_prefix, sizeof(std::uint32_t) * node_capacity),
             "allocate resident cost prefix");
  check_cuda(cudaMalloc(&workspace.history_prefix, sizeof(std::uint32_t) * node_capacity),
             "allocate resident history prefix");
  check_cuda(cudaMalloc(&workspace.target_claims, sizeof(std::uint64_t) * node_capacity),
             "allocate resident target claims");
  workspace.block_capacity = std::max(256u, (node_capacity + 255u) / 256u);
  check_cuda(cudaMalloc(&workspace.block_sums,
                        sizeof(std::uint32_t) * workspace.block_capacity),
             "allocate resident block sums");
  check_cuda(cudaMalloc(&workspace.block_offsets,
                        sizeof(std::uint32_t) * workspace.block_capacity),
             "allocate resident block offsets");
  check_cuda(cudaMalloc(&workspace.counters, sizeof(ResidentDevelopmentCounters)),
             "allocate resident counters");
  check_cuda(cudaMalloc(&workspace.topology_plan, sizeof(TopologyCommitPlan)),
             "allocate resident topology plan");
  check_cuda(cudaMalloc(&workspace.reserve_snapshot, sizeof(std::uint64_t)),
             "allocate resident reserve snapshot");
  check_cuda(cudaMalloc(&workspace.cause_memo,
                        sizeof(std::uint32_t) *
                            (kDirectExactHistoryHotPageCapacity + 1u)),
             "allocate resident cause memo");
  check_cuda(cudaMemset(workspace.cause_memo, 0,
                        sizeof(std::uint32_t) *
                            (kDirectExactHistoryHotPageCapacity + 1u)),
             "zero resident cause memo");
  cub::DeviceScan::ExclusiveSum(nullptr, workspace.scan_storage_bytes, workspace.costs,
                                workspace.cost_prefix, node_capacity);
  check_cuda(cudaMalloc(&workspace.scan_storage, workspace.scan_storage_bytes),
             "allocate resident scan storage");
  return workspace;
}

void destroy_resident_development_workspace(ResidentDevelopmentWorkspace* workspace) {
  if (workspace == nullptr) return;
  cudaFree(workspace->cause_memo);
  cudaFree(workspace->reserve_snapshot);
  cudaFree(workspace->topology_plan);
  cudaFree(workspace->counters);
  cudaFree(workspace->scan_storage);
  cudaFree(workspace->block_offsets);
  cudaFree(workspace->block_sums);
  cudaFree(workspace->target_claims);
  cudaFree(workspace->history_prefix);
  cudaFree(workspace->cost_prefix);
  cudaFree(workspace->costs);
  cudaFree(workspace->proposals);
  *workspace = ResidentDevelopmentWorkspace{};
}

void launch_resident_development_epoch(DirectBrain* brain,
                                       ResidentDevelopmentWorkspace* workspace,
                                       cudaStream_t stream, std::uint32_t block_size,
                                       const std::uint32_t* route_causal_pin_bits) {
  if (brain == nullptr || workspace == nullptr || brain->development == nullptr ||
      workspace->node_capacity < brain->node_count || workspace->history_prefix == nullptr ||
      workspace->topology_plan == nullptr || block_size == 0u) {
    throw std::invalid_argument("invalid resident development state/workspace");
  }
  clear_resident_development_epoch_state_kernel<<<grid_for(brain->node_count, block_size),
                                                  block_size, 0, stream>>>(
      workspace->target_claims, brain->node_count, workspace->counters,
      workspace->topology_plan);
  check_cuda(cudaGetLastError(), "clear resident development epoch state");
  advance_development_state_kernel<<<1, 32, 0, stream>>>(brain->development,
                                                         workspace->reserve_snapshot);
  update_node_maturation_kernel<<<grid_for(brain->node_count, block_size), block_size, 0, stream>>>(
      *brain, workspace->counters);
  propose_route_mutations_kernel<<<grid_for(brain->node_count, block_size), block_size, 0, stream>>>(
      *brain, workspace->proposals, workspace->costs, workspace->target_claims,
      workspace->counters);
  check_cuda(cudaGetLastError(), "launch resident development proposals");
  filter_winning_growth_costs_kernel<<<grid_for(brain->node_count, block_size), block_size, 0, stream>>>(
      workspace->proposals, workspace->costs, workspace->target_claims, brain->node_count);
  check_cuda(cudaGetLastError(), "filter resident target-claim winners");
  check_cuda(cub::DeviceScan::ExclusiveSum(workspace->scan_storage, workspace->scan_storage_bytes,
                                            workspace->costs, workspace->cost_prefix,
                                            brain->node_count, stream),
             "scan resident construction costs");
  prepare_topology_plan_kernel<<<grid_for(brain->node_count, block_size), block_size, 0, stream>>>(
      *brain, workspace->proposals, workspace->costs, workspace->cost_prefix,
      workspace->reserve_snapshot, workspace->target_claims,
      route_causal_pin_bits, workspace->counters, workspace->topology_plan);
  check_cuda(cudaGetLastError(), "prepare resident topology transaction");
  check_cuda(cub::DeviceScan::ExclusiveSum(workspace->scan_storage, workspace->scan_storage_bytes,
                                            workspace->costs, workspace->history_prefix,
                                            brain->node_count, stream),
             "scan pressure retraction candidates");
  finalize_topology_plan_kernel<<<grid_for(brain->node_count, block_size), block_size, 0, stream>>>(
      *brain, workspace->proposals, workspace->costs, workspace->history_prefix,
      workspace->counters, workspace->topology_plan);
  check_cuda(cudaGetLastError(), "finalize resident topology transaction");
  check_cuda(cub::DeviceScan::ExclusiveSum(workspace->scan_storage, workspace->scan_storage_bytes,
                                            workspace->costs, workspace->cost_prefix,
                                            brain->node_count, stream),
             "scan resident structural work");
  admit_topology_work_kernel<<<grid_for(brain->node_count, block_size), block_size, 0, stream>>>(
      *brain, workspace->proposals, workspace->costs, workspace->cost_prefix,
      workspace->counters, workspace->topology_plan);
  check_cuda(cudaGetLastError(), "admit resident structural work");
  check_cuda(cub::DeviceScan::ExclusiveSum(workspace->scan_storage, workspace->scan_storage_bytes,
                                            workspace->costs, workspace->history_prefix,
                                            brain->node_count, stream),
             "scan resident topology history slots");
  begin_topology_transaction_kernel<<<1, 1, 0, stream>>>(*brain, workspace->topology_plan);
  check_cuda(cudaGetLastError(), "begin resident topology transaction");
  commit_route_mutations_kernel<<<grid_for(brain->node_count, block_size), block_size, 0, stream>>>(
      *brain, workspace->proposals, workspace->cost_prefix, workspace->history_prefix,
      workspace->reserve_snapshot, workspace->target_claims, route_causal_pin_bits,
      workspace->counters, workspace->topology_plan);
  check_cuda(cudaGetLastError(), "commit resident development mutations");
  nominate_committed_retractions_kernel<<<1, 1, 0, stream>>>(
      *brain, workspace->proposals, workspace->counters);
  check_cuda(cudaGetLastError(), "nominate committed local repair fronts");
  finish_topology_transaction_kernel<<<1, 1, 0, stream>>>(*brain, workspace->topology_plan);
  check_cuda(cudaGetLastError(), "finish resident topology transaction");
  commit_recipe_transaction_kernel<<<1, 1, 0, stream>>>(
      *brain, workspace->proposals, workspace->counters);
  check_cuda(cudaGetLastError(), "commit resident Recipe transaction");
  commit_compact_construction_fronts_kernel<<<1, 1, 0, stream>>>(
      *brain, workspace->proposals, workspace->costs,
      workspace->reserve_snapshot, workspace->counters);
  check_cuda(cudaGetLastError(), "commit compact construction-front transaction");
  condense_postbirth_recipe_kernel<<<1, 1, 0, stream>>>(*brain, workspace->counters,
                                                        workspace->cause_memo);
  check_cuda(cudaGetLastError(), "condense postbirth resident Recipe");
  finish_resident_admission_epoch_kernel<<<1, 1, 0, stream>>>(
      *brain, workspace->counters);
  check_cuda(cudaGetLastError(), "finish resident Adult admission epoch");
}

void launch_exact_history_maintenance(
    DirectBrain* brain, ResidentDevelopmentWorkspace* workspace,
    const std::uint64_t* opportunity_incarnations,
    std::int32_t cost_per_route_q16, const std::uint32_t* route_causal_pin_bits,
    direct_adult_core::AdultCoreMetrics* metrics, std::uint32_t current_tick,
    cudaStream_t stream, std::uint32_t block_size) {
  if (brain == nullptr || workspace == nullptr || brain->development == nullptr ||
      workspace->node_capacity < brain->node_count ||
      workspace->history_prefix == nullptr || workspace->topology_plan == nullptr ||
      metrics == nullptr || block_size == 0u)
    throw std::invalid_argument("invalid exact-history maintenance state");
  check_cuda(cudaMemsetAsync(workspace->topology_plan, 0,
                             sizeof(TopologyCommitPlan), stream),
             "clear maintenance topology plan");
  plan_maintenance_transaction_kernel<<<grid_for(brain->node_count, block_size),
                                        block_size, 0, stream>>>(
      *brain, opportunity_incarnations, route_causal_pin_bits,
      cost_per_route_q16, workspace->costs, workspace->topology_plan, metrics);
  check_cuda(cudaGetLastError(), "plan exact-history maintenance transaction");
  check_cuda(cub::DeviceScan::ExclusiveSum(
                 workspace->scan_storage, workspace->scan_storage_bytes,
                 workspace->costs, workspace->history_prefix,
                 brain->node_count, stream),
             "scan maintenance topology history slots");
  begin_maintenance_transaction_kernel<<<1, 1, 0, stream>>>(
      *brain, workspace->topology_plan, current_tick);
  check_cuda(cudaGetLastError(), "begin exact-history maintenance transaction");
  commit_maintenance_transaction_kernel<<<grid_for(brain->node_count, block_size),
                                          block_size, 0, stream>>>(
      *brain, opportunity_incarnations, route_causal_pin_bits,
      workspace->history_prefix, workspace->topology_plan, metrics);
  check_cuda(cudaGetLastError(), "commit exact-history maintenance transaction");
  finish_topology_transaction_kernel<<<1, 1, 0, stream>>>(
      *brain, workspace->topology_plan);
  check_cuda(cudaGetLastError(), "finish exact-history maintenance transaction");
  plan_homeostatic_weight_transaction_kernel<<<grid_for(brain->node_count, block_size),
                                               block_size, 0, stream>>>(
      *brain, opportunity_incarnations, route_causal_pin_bits,
      cost_per_route_q16, workspace->topology_plan, workspace->costs);
  check_cuda(cudaGetLastError(), "plan homeostatic weight transaction");
  check_cuda(cub::DeviceScan::ExclusiveSum(
                 workspace->scan_storage, workspace->scan_storage_bytes,
                 workspace->costs, workspace->history_prefix,
                 brain->node_count, stream),
             "scan homeostatic weight history slots");
  begin_homeostatic_weight_transaction_kernel<<<1, 1, 0, stream>>>(
      *brain, workspace->costs, workspace->history_prefix, brain->node_count,
      workspace->topology_plan, current_tick);
  check_cuda(cudaGetLastError(), "begin homeostatic weight transaction");
  decay_maintenance_routes_kernel<<<grid_for(brain->node_count, block_size),
                                    block_size, 0, stream>>>(
      *brain, opportunity_incarnations, route_causal_pin_bits,
      cost_per_route_q16, workspace->history_prefix, workspace->topology_plan);
  check_cuda(cudaGetLastError(), "commit homeostatic weight transaction");
  finish_topology_transaction_kernel<<<1, 1, 0, stream>>>(
      *brain, workspace->topology_plan);
  check_cuda(cudaGetLastError(), "finish homeostatic weight transaction");
}

ResidentDevelopmentCounters observe_resident_development(
    const ResidentDevelopmentWorkspace& workspace, cudaStream_t stream) {
  ResidentDevelopmentCounters host{};
  check_cuda(cudaMemcpyAsync(&host, workspace.counters, sizeof(host), cudaMemcpyDeviceToHost, stream),
             "observe resident development counters");
  check_cuda(cudaStreamSynchronize(stream), "finish resident development observation");
  return host;
}

}  // namespace substrate::direct_network
