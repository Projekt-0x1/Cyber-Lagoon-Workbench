#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include "hardware_native/direct_adult_core_kernels.cuh"
#include "hardware_native/direct_adult_device_ops.cuh"

namespace substrate::direct_adult_core {
#include "hardware_native/direct_adult_delayed_sparse_schedule.cuh"
#include "hardware_native/direct_adult_delayed_sparse_delivery.cuh"
namespace cg = cooperative_groups;
__global__ void derive_ingress_eligibility_frontier_kernel(
    EligibilityRecord* eligibility_table,
    std::uint32_t* eligibility_record_generations,
    std::uint32_t* live_eligibility_count, std::uint32_t current_tick,
    std::uint32_t* scan, std::uint32_t* scratch,
    std::uint32_t* free_slots, AdultCoreMetrics* metrics) {
  derive_ingress_eligibility_free_frontier(
      cg::this_grid(), eligibility_table, eligibility_record_generations,
      live_eligibility_count, current_tick, scan, scratch, free_slots, metrics);
}
__global__ void deterministic_route_transport_kernel(
    DirectBrain brain, std::int32_t* node_incoming_excitation,
    DelayedSparsePacket* delayed_packets,
    std::uint32_t* delayed_packet_live_count,
    std::uint32_t* delayed_packet_free_head,
    std::uint32_t* delayed_packet_next_free,
    DelayedPacketIdentity* delayed_packet_identities,
    const NodeCausalParticipation* node_active_participation,
    NodeCausalParticipation* node_next_participation,
    const std::uint32_t* node_active_ancestry_incomplete,
    std::uint32_t* node_next_ancestry_incomplete,
    DirectParticipationDescriptor* participation_staging,
    std::uint32_t* participation_staging_count,
    std::uint32_t participation_staging_capacity,
    EligibilityRecord* eligibility_table,
    std::uint32_t* live_eligibility_count,
    std::uint64_t* eligibility_claim_directory,
    std::uint32_t* eligibility_record_generations,
    std::uint64_t* route_opportunity_incarnations,
    RouteTransportPhaseView transport, std::uint32_t current_tick,
    AdultExecutionConfig config, AdultCoreMetrics* metrics) {
  execute_deterministic_route_transport(
      cg::this_grid(), brain.nodes, brain.routes,
      brain.route_incarnations, route_opportunity_incarnations,
      brain.node_count, brain.route_capacity, node_incoming_excitation,
      delayed_packets, delayed_packet_live_count, delayed_packet_free_head,
      delayed_packet_next_free, delayed_packet_identities,
      node_active_participation, node_next_participation,
      node_active_ancestry_incomplete, node_next_ancestry_incomplete,
      participation_staging, participation_staging_count,
      participation_staging_capacity, eligibility_table,
      live_eligibility_count, eligibility_claim_directory,
      eligibility_record_generations, transport, current_tick,
      config.eligibility_decay_q16, config.eligibility_horizon_ticks,
      config.honor_inhibitory_sign, metrics);
}

}  // namespace substrate::direct_adult_core
