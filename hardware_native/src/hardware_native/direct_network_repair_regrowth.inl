#ifndef HARDWARE_NATIVE_DIRECT_NETWORK_REPAIR_REGROWTH_TYPES
#define HARDWARE_NATIVE_DIRECT_NETWORK_REPAIR_REGROWTH_TYPES
inline constexpr std::uint32_t kDirectTopologyHistoryFocalLesion = 2u;

DIRECT_NETWORK_HD inline std::uint64_t focal_lesion_occurrence_identity(std::uint32_t tick, std::uint32_t source,
    std::uint32_t route, std::uint32_t target, std::uint32_t flags, std::uint64_t before, std::uint64_t after) {
  std::uint64_t identity = 0x7265706169726375ull;
  identity = exact_history_fold_word(identity, tick);
  identity = exact_history_fold_word(identity, source);
  identity = exact_history_fold_word(identity, route);
  identity = exact_history_fold_word(identity, target);
  identity = exact_history_fold_word(identity, flags);
  identity = exact_history_fold_word(identity, before);
  identity = exact_history_fold_word(identity, after);
  return identity == 0u ? 1u : identity;
}
DIRECT_NETWORK_HD inline void stage_focal_lesion_history_record(DirectExactHistoryRecord* record, std::uint32_t tick,
    std::uint32_t source, std::uint32_t route, std::uint32_t target, std::uint32_t flags, std::uint64_t before, std::uint64_t after) {
  stage_topology_history_record(record, DirectExactHistoryKind::topology_retraction,
                                tick, source, route, target, flags, before, after, -1, kDirectTopologyHistoryFocalLesion);
  if (record != nullptr)
    record->identity = focal_lesion_occurrence_identity(tick, source, route, target, flags, before, after);
}
#endif

#if defined(DIRECT_NETWORK_REPAIR_REGROWTH_IMPLEMENTATION)
__device__ bool nominate_local_repair_front(DirectBrain brain, std::uint32_t node_index,
    std::uint64_t loss_identity, std::uint32_t loss_route, std::uint32_t loss_target,
    std::uint64_t loss_route_incarnation, ResidentDevelopmentCounters* counters) {
  if (brain.construction_front_count == nullptr || brain.development == nullptr) return false;
  std::uint32_t count = *brain.construction_front_count;
  if (node_index >= brain.node_count || loss_identity == 0u) return false;
  const DirectNode& node = brain.nodes[node_index];
  if (node.active_route_count >= 2u ||
      bound_field_kind_influence_q16(brain, node.territory_index, node, brain.development->age_tick,
          DevelopmentFieldKind::repair) <= (1 << 14)) return false;
  for (std::uint32_t i = 0u; i < count; ++i)
    if (construction_front_has_loss_identity(brain.construction_fronts[i], loss_identity)) return false;
  for (std::uint32_t i = 0u; i < count; ++i) {
    ResidentConstructionFront& existing = brain.construction_fronts[i];
    if (existing.source_node == node_index && existing.rule_index < brain.resident_rule_count &&
        brain.resident_rules[existing.rule_index].opcode == RuleOpcode::repair) {
      const std::uint64_t watermark = brain.construction_front_generation_by_node[node_index];
      if (live_construction_front_at_node(brain, node_index) &&
          watermark != existing.generation) return false;
      existing.state = kConstructionFrontLive; existing.loss_identity = loss_identity;
      existing.loss_route = loss_route; existing.loss_target = loss_target;
      existing.loss_source = node_index; existing.loss_route_incarnation = loss_route_incarnation;
      existing.source_route = kInvalidIndex; existing.source_route_incarnation = 0u;
      const std::uint64_t generation = next_construction_front_generation(watermark);
      if (generation == 0u) return false;
      existing.generation = generation;
      brain.construction_front_generation_by_node[node_index] = generation;
      return true;
    }
  }
  if (brain.resource_ecology == nullptr ||
      count >= brain.resource_ecology->maintenance_scan_budget ||
      count >= brain.construction_front_capacity) return false;
  const std::uint64_t watermark =
      brain.construction_front_generation_by_node[node_index];
  if (live_construction_front_at_node(brain, node_index)) return false;
  const std::uint32_t cell_index = select_recipe_cell(brain, node, brain.development->age_tick, true);
  if (cell_index == kInvalidIndex) return false;
  const ResidentRecipeCell& cell = brain.recipe_cells[cell_index];
  if (cell.rule_index >= brain.resident_rule_count || brain.resident_rules[cell.rule_index].opcode != RuleOpcode::repair) return false;
  const std::uint64_t generation = next_construction_front_generation(watermark);
  if (generation == 0u) return false;
  brain.construction_fronts[count] = ResidentConstructionFront{
      node_index, cell_index, cell.rule_index, kInvalidIndex, generation,
      0u, 0u, node.territory_index, kConstructionFrontLive, loss_route, loss_identity,
      loss_route_incarnation, loss_target, node_index};
  brain.construction_front_generation_by_node[node_index] = brain.construction_fronts[count].generation;
  *brain.construction_front_count = count + 1u;
  if (counters != nullptr) atomicAdd(&counters->construction_fronts_recruited, 1u);
  return true;
}
__device__ void nominate_committed_retractions_impl(DirectBrain brain, ResidentDevelopmentCounters* counters) {
  if (brain.development == nullptr) return;
  // Consume only this transaction's canonical compact ordinal range.
  // Adult node count and older immutable history never participate in repair
  // discovery work.
  DirectExactHistoryHotPage& history = brain.development->exact_history;
  for (std::uint32_t ordinal = 0u; ordinal < history.phase_width; ++ordinal) {
    if (counters != nullptr) ++counters->repair_nomination_work;
    const DirectExactHistoryRecord& record = history.records[history.phase_base + ordinal];
    if (record.kind == DirectExactHistoryKind::topology_retraction && record.identity != 0u)
      nominate_local_repair_front(brain, record.source, record.identity, record.subject,
                                  record.value, record.incarnation_after, counters);
  }
}

__global__ void nominate_committed_retractions_kernel(DirectBrain brain,
    RouteMutationProposal* proposals, ResidentDevelopmentCounters* counters) {
  if (blockIdx.x != 0u || threadIdx.x != 0u) return;
  (void)proposals;
  nominate_committed_retractions_impl(brain, counters);
}
#endif
