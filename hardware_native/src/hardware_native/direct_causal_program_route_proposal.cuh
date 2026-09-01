#ifndef HARDWARE_NATIVE_DIRECT_CAUSAL_PROGRAM_ROUTE_PROPOSAL_CUH
#define HARDWARE_NATIVE_DIRECT_CAUSAL_PROGRAM_ROUTE_PROPOSAL_CUH

#include <cstdint>
#include <type_traits>
#include "hardware_native/direct_adult_core.cuh"
#include "hardware_native/direct_adult_device_ops.cuh"
#include "hardware_native/direct_causal_program_route_lowering.cuh"

namespace substrate::direct_causal_program {

struct DirectRouteProposalReceipt {
  RouteLoweringReceipt lowering;
  std::int32_t drive_q16;
  std::uint32_t claim_count;
  std::uint32_t admitted;
};
static_assert(std::is_standard_layout_v<DirectRouteProposalReceipt> &&
              std::is_trivial_v<DirectRouteProposalReceipt>);

template <typename ParentEligibilityT>
__device__ inline DirectRouteProposalReceipt propose_program_step_via_direct_route(
    const ProgramExecutionState& execution, const ParentEligibilityT& parent,
    std::uint64_t current_parent_eligibility_ref,
    substrate::direct_adult_core::RouteTransportProposal* proposals,
    const substrate::direct_adult_core::NodeCausalParticipation* source_frontier,
    const substrate::direct_adult_core::EligibilityRecord* eligibility_table,
    const std::uint32_t* eligibility_generations,
    std::uint32_t* target_ancestry_incomplete,
    const substrate::direct_network::DirectNode* nodes, std::uint32_t node_count,
    const substrate::direct_network::DirectRoute* routes,
    const std::uint64_t* route_incarnations, std::uint32_t route_capacity,
    std::uint32_t current_tick, bool honor_inhibitory_sign = true) {
  using namespace substrate::direct_adult_core;
  using namespace substrate::direct_network;
  DirectRouteProposalReceipt out{};
  out.lowering = resolve_program_step_route(
      execution, parent, routes, route_incarnations, route_capacity,
      current_parent_eligibility_ref, current_tick);
  if (out.lowering.admitted == 0u || nodes == nullptr ||
      out.lowering.source_node >= node_count || out.lowering.target_node >= node_count ||
      out.lowering.route_index >= route_capacity)
    return out;
  const DirectNode& source = nodes[out.lowering.source_node];
  const DirectRoute& route = routes[out.lowering.route_index];
  if (source.activation_q16 <= 0 || route.delay == 0u ||
      out.lowering.route_index < source.route_offset ||
      out.lowering.route_index - source.route_offset >= bounded_route_scan_count(source))
    return out;
  const std::int32_t drive = signed_sparse_route_delivery_q16(
      source, route, competition_output_gain_q16(source), honor_inhibitory_sign,
      direct_tube_chemistry_q16(source.chemotype, nodes[route.target].chemotype)
          .conductance_gain_q16);
  if (drive == 0) return out;
  const std::uint32_t context =
      route.eligibility_context ^ (route.source * 2654435761U);
  propose_delayed_sparse_packet(
      proposals, route_capacity, source_frontier, eligibility_table,
      eligibility_generations, target_ancestry_incomplete,
      route.source, route.target, out.lowering.route_index,
      out.lowering.route_incarnation, context, source.activation_q16, drive,
      0u, out.lowering.due_tick, current_tick);
  const RouteTransportProposal& proposal = proposals[out.lowering.route_index];
  if (proposal.packet.route_index != out.lowering.route_index ||
      proposal.packet.route_incarnation != out.lowering.route_incarnation ||
      proposal.packet.source_node != route.source || proposal.packet.target_node != route.target ||
      proposal.packet.due_tick != out.lowering.due_tick ||
      proposal.packet.drive_q16 != drive || proposal.packet.claim_count == 0u ||
      proposal.packet.source_ancestry_incomplete != 0u)
    return out;
  out.drive_q16 = drive;
  out.claim_count = proposal.packet.claim_count;
  out.admitted = 1u;
  return out;
}

}  // namespace substrate::direct_causal_program

#endif
