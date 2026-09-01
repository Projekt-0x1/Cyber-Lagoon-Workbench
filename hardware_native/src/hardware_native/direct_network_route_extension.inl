#ifndef HARDWARE_NATIVE_DIRECT_NETWORK_ROUTE_EXTENSION_TYPES
#define HARDWARE_NATIVE_DIRECT_NETWORK_ROUTE_EXTENSION_TYPES
inline constexpr std::int32_t kRouteExtensionRejected = INT32_MIN;
#endif

#if defined(DIRECT_NETWORK_ROUTE_EXTENSION_IMPLEMENTATION)
__device__ std::int32_t route_extension_progress_q16(
    const DirectBrain& brain, const ResidentConstructionFront& front,
    const ResidentConstructorRule& rule, std::uint32_t target_index,
    const DirectNode& target) {
  if (rule.opcode != RuleOpcode::extend) return 0;
  if (rule.field_index >= brain.resident_field_count ||
      brain.resident_fields[rule.field_index].kind != DevelopmentFieldKind::attract ||
      live_construction_front_at_node(brain, target_index))
    return kRouteExtensionRejected;
  const DirectNode& source = brain.nodes[front.source_node];
  const ResidentDevelopmentField& field = brain.resident_fields[rule.field_index];
  if (field_influence_q16(brain, source, static_cast<std::uint16_t>(rule.field_index),
                          brain.development->age_tick) <= 0)
    return kRouteExtensionRejected;
  std::int64_t before = 0, after = 0, alignment = 0;
  for (std::uint32_t axis = 0u; axis < 3u; ++axis) {
    before += iabs32(source.coordinate[axis] - field.center[axis]);
    after += iabs32(target.coordinate[axis] - field.center[axis]);
    const std::int64_t segment = static_cast<std::int64_t>(target.coordinate[axis]) -
                                 source.coordinate[axis];
    if (front.source_route == kInvalidIndex) {
      alignment += segment * (static_cast<std::int64_t>(field.center[axis]) -
                              source.coordinate[axis]);
    } else {
      const DirectRoute& prior = brain.routes[front.source_route];
      alignment += segment * (static_cast<std::int64_t>(source.coordinate[axis]) -
                              brain.nodes[prior.source].coordinate[axis]);
    }
  }
  if (after >= before || alignment <= 0) return kRouteExtensionRejected;
  return clamp_i32((before - after) << 8, 0, 4 << 16);
}

__device__ void advance_construction_front_after_commit(DirectBrain brain, const RouteMutationProposal& proposal) {
  if (proposal.construction_front_index >= brain.construction_front_capacity) return;
  ResidentConstructionFront& front = brain.construction_fronts[proposal.construction_front_index];
  if (front.generation != proposal.construction_front_generation ||
      front.source_node != proposal.node) return;
  if (proposal.target >= brain.node_count ||
      live_construction_front_at_node(brain, proposal.target)) return;
  const std::uint32_t route_index = brain.nodes[proposal.node].route_offset + proposal.route_slot;
  const ResidentRecipeCell& cell = brain.recipe_cells[proposal.recipe_cell];
  if (cell.rule_index >= brain.resident_rule_count) return;
  const RuleOpcode opcode = brain.resident_rules[cell.rule_index].opcode;
  const std::uint32_t old_tip = front.source_node;
  const std::uint64_t old_generation = front.generation;
  const std::uint64_t route_incarnation = brain.route_incarnations[route_index];
  const std::uint64_t target_watermark =
      brain.construction_front_generation_by_node[proposal.target];
  front.source_route = route_index;
  front.source_route_incarnation = route_incarnation;
  ++front.successor_sequence;
  if (opcode == RuleOpcode::branch) {
    const std::uint64_t parent_generation = next_construction_front_generation(
        brain.construction_front_generation_by_node[old_tip]);
    if (parent_generation == 0u) {
      front.state = kConstructionFrontQuiescent;
      retire_construction_front_generation(brain, old_tip, old_generation);
      return;
    }
    front.generation = parent_generation;
    brain.construction_front_generation_by_node[old_tip] = parent_generation;
    if (brain.construction_front_count != nullptr &&
        *brain.construction_front_count < brain.construction_front_capacity &&
        !live_construction_front_at_node(brain, proposal.target)) {
      const std::uint64_t child_generation =
          next_construction_front_generation(target_watermark);
      if (child_generation == 0u) return;
      const std::uint32_t child_index = (*brain.construction_front_count)++;
      brain.construction_fronts[child_index] = ResidentConstructionFront{
          proposal.target, front.recipe_cell, front.rule_index, route_index,
          child_generation, route_incarnation, front.successor_sequence,
          brain.nodes[proposal.target].territory_index, kConstructionFrontLive, 0u};
      brain.construction_front_generation_by_node[proposal.target] = child_generation;
    }
  } else if (opcode == RuleOpcode::fuse) {
    front.state = kConstructionFrontQuiescent;
    retire_construction_front_generation(brain, old_tip, old_generation);
  } else {
    retire_construction_front_generation(brain, old_tip, old_generation);
    const std::uint64_t successor_generation = next_construction_front_generation(target_watermark);
    if (successor_generation == 0u) {
      front.state = kConstructionFrontQuiescent;
      return;
    }
    front.source_node = proposal.target;
    front.territory_index = brain.nodes[proposal.target].territory_index;
    front.generation = successor_generation;
    brain.construction_front_generation_by_node[front.source_node] = successor_generation;
  }
}
#endif
