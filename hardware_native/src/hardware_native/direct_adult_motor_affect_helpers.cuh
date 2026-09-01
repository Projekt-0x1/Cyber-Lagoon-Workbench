#ifndef HARDWARE_NATIVE_DIRECT_ADULT_MOTOR_AFFECT_HELPERS_CUH
#define HARDWARE_NATIVE_DIRECT_ADULT_MOTOR_AFFECT_HELPERS_CUH

// One egress pass reads immutable journal/contribution prefixes. The same memo
// and temporal score are used by both canonical executors.
struct MotorRootChannelMemo {
  std::uint32_t nodes[direct_network::kMaxBoundaryPorts];
  std::uint32_t roots[direct_network::kMaxBoundaryPorts];
  bool verified[direct_network::kMaxBoundaryPorts];
  std::uint32_t size = 0u;
};

__device__ inline bool resolve_motor_root_channel(
    MotorRootChannelMemo& memo, const direct_network::DirectExactHistoryRecord* records,
    std::uint32_t record_count, const DirectParticipationDescriptor* contributions,
    const std::uint32_t* contribution_count, std::uint32_t contribution_capacity,
    std::uint32_t motor_node, std::uint32_t current_tick, std::uint32_t* root_channel) {
  for (std::uint32_t i = 0u; i < memo.size; ++i) {
    if (memo.nodes[i] != motor_node) continue;
    *root_channel = memo.roots[i]; return memo.verified[i];
  }
  const bool resolved = candidate_sensory_root_channel(
      records, record_count, contributions, contribution_count,
      contribution_capacity, motor_node, current_tick, root_channel);
  if (memo.size < direct_network::kMaxBoundaryPorts) {
    memo.nodes[memo.size] = motor_node; memo.roots[memo.size] = *root_channel;
    memo.verified[memo.size] = resolved; ++memo.size;
  }
  return resolved;
}

__device__ inline std::int32_t affect_motor_competition_value_q16(
    MotorRootChannelMemo& root_memo, const DirectNode& node, std::uint32_t motor_node,
    const direct_network::DirectAffectBodyState* state,
    const DirectRoute* /*routes*/, const std::uint64_t* /*route_incarnations*/,
    std::uint32_t /*route_capacity*/,
    const direct_network::DirectExactHistoryRecord* records, std::uint32_t record_count,
    const DirectParticipationDescriptor* contributions,
    const std::uint32_t* contribution_count, std::uint32_t contribution_capacity,
    std::uint32_t current_tick) {
  std::uint32_t root_channel = ::substrate::direct_adult_core::kInvalidIndex;
  (void)resolve_motor_root_channel(root_memo, records, record_count, contributions,
      contribution_count, contribution_capacity, motor_node, current_tick, &root_channel);
  direct_network::DirectAffectCandidate candidate{}; candidate.root_channel = root_channel;
  candidate.base_valuation_q16 = clamp_q16(
      node.activation_q16, 0, ::substrate::direct_adult_core::kQ16One);
  return direct_network::affect_modulate_candidate(state, candidate)
             .modulated_valuation_q16 +
         direct_network::delayed_temporal_motor_value_q16(
             records, record_count, motor_node);
}

__device__ inline bool affect_motor_candidate_wins_local_competition(
    const DirectNode* nodes, const DirectBoundaryPort* ports, std::uint32_t port_count,
    std::uint32_t candidate_index, MotorRootChannelMemo& root_memo,
    const direct_network::DirectAffectBodyState* state, const DirectRoute* routes,
    const std::uint64_t* route_incarnations, std::uint32_t route_capacity,
    const direct_network::DirectExactHistoryRecord* records, std::uint32_t record_count,
    const DirectParticipationDescriptor* contributions,
    const std::uint32_t* contribution_count, std::uint32_t contribution_capacity,
    std::uint32_t current_tick) {
  const DirectBoundaryPort candidate_port = ports[candidate_index];
  const DirectNode candidate = nodes[candidate_port.node];
  if ((candidate.flags & direct_network::kNodeFlagCompetitive) == 0u) return true;
  const std::int32_t candidate_value = affect_motor_competition_value_q16(root_memo,
      candidate, candidate_port.node, state, routes, route_incarnations, route_capacity,
      records, record_count, contributions, contribution_count, contribution_capacity,
      current_tick);
  for (std::uint32_t other_index = 0u; other_index < port_count; ++other_index) {
    if (other_index == candidate_index) continue;
    const DirectBoundaryPort other_port = ports[other_index];
    if ((other_port.role_mask & static_cast<std::uint32_t>(direct_network::BoundaryRole::motor)) == 0u ||
        other_port.parent_route != candidate_port.parent_route) continue;
    const DirectNode other = nodes[other_port.node];
    if ((other.flags & direct_network::kNodeFlagCompetitive) == 0u ||
        other.territory_index != candidate.territory_index ||
        other.activation_q16 <= ::substrate::direct_adult_core::kQ16One / 4)
      continue;
    const std::int32_t other_value = affect_motor_competition_value_q16(root_memo,
        other, other_port.node, state, routes, route_incarnations, route_capacity, records,
        record_count, contributions, contribution_count, contribution_capacity, current_tick);
    if (affect_motor_competition_score_precedes(other_value, other, other_port, other_index,
        candidate_value, candidate, candidate_port, candidate_index)) return false;
  }
  return true;
}
#endif
