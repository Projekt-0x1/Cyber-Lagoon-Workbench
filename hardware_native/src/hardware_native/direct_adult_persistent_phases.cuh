#ifndef HARDWARE_NATIVE_DIRECT_ADULT_PERSISTENT_PHASES_CUH
#define HARDWARE_NATIVE_DIRECT_ADULT_PERSISTENT_PHASES_CUH
// Out-of-line major phases of the cooperative persistent executor.

#include "hardware_native/direct_adult_persistent.cuh"
#include <cooperative_groups.h>

namespace substrate::direct_adult_core {

__device__ const std::uint32_t* rebuild_persistent_route_causal_pins(
    cooperative_groups::grid_group grid, const EligibilityRecord* eligibility_table,
    const std::uint32_t* live_eligibility_count,
    const std::uint64_t* route_incarnations, std::uint32_t route_capacity,
    std::uint32_t current_tick, std::uint32_t* pin_bits,
    std::uint32_t global_tid, std::uint32_t total_threads,
    std::uint32_t honor_causal_pin, AdultCoreMetrics* metrics);

__device__ void persistent_exclusive_scan(
    cooperative_groups::grid_group grid, const std::uint32_t* input,
    std::uint32_t* output, std::uint32_t* block_sums,
    std::uint32_t* block_offsets, std::uint32_t* local_scan,
    std::uint32_t node_count, std::uint32_t global_tid);

__device__ void persistent_ingress_phase_device(
    cooperative_groups::grid_group grid,
    const ResidentAdultEpochSnapshot& epoch_snap,
    std::uint32_t current_tick,
    std::uint32_t global_tid,
    std::uint32_t total_threads,
    direct_network::DirectExactHistoryHotPage* exact_history,
    ResidentMismatchOmissionFrontier* mismatch_omission_frontier,
    DirectNode* nodes,
    std::uint32_t node_count,
    const DirectBoundaryPort* boundary_ports,
    std::uint32_t boundary_port_count,
    ActivityEvent* ingress_queue,
    ResidentContactEpochReceipt* ingress_contact_credentials,
    ResidentActualFrontier* actual_frontier,
    ResidentActivationSoaPlane* activation_plane,
    ResidentMultiHorizonPredictionFrontier* causal_credit_predictions,
    IngressRingControl* ingress_control,
    EligibilityRecord* eligibility_table,
    std::uint32_t* live_eligibility_count,
    std::uint64_t* eligibility_claim_directory,
    std::uint32_t* eligibility_record_generations,
    AsynchronousTicket* ticket_table,
    std::uint32_t* ticket_table_locks,
    std::uint32_t* claim_incarnation_counter,
    DirectActionOccurrence* action_occurrences,
    std::int32_t* node_incoming_excitation,
    std::int32_t* node_slow_context_q16,
    RouteTransportPhaseView route_transport,
    NodeCausalParticipation* node_active_participation,
    std::uint32_t* node_active_participation_locks,
    DirectParticipationDescriptor* participation_staging,
    std::uint32_t* participation_staging_count,
    std::uint32_t participation_staging_capacity,
    AdultCoreMetrics* device_metrics,
    AdultExecutionConfig config,
    DirectBrain brain);

__device__ void persistent_settlement_phase_device(
    cooperative_groups::grid_group grid,
    const ResidentAdultEpochSnapshot& epoch_snap,
    std::uint32_t current_tick,
    std::uint32_t global_tid,
    std::uint32_t total_threads,
    direct_network::DirectExactHistoryHotPage* exact_history,
    DirectNode* nodes,
    DirectRoute* routes,
    std::uint32_t node_count,
    std::uint32_t route_capacity,
    ResidentActualFrontier* actual_frontier,
    ResidentMultiHorizonPredictionFrontier* causal_credit_predictions,
    ConsequenceIngressEvent* consequence_queue,
    IngressRingControl* consequence_control,
    EligibilityRecord* eligibility_table,
    std::uint32_t* live_eligibility_count,
    AsynchronousTicket* ticket_table,
    DirectActionOccurrence* action_occurrences,
    DirectActionParticipationLink* action_participation_links,
    std::uint32_t action_participant_capacity,
    direct_network::DirectAffectBodyState* affect_body_state,
    ResolvedConsequenceContext* resolved_consequence_ctx,
    ResidentCueSalienceTable* cue_salience_table,
    AdultCoreMetrics* device_metrics,
    AdultExecutionConfig config,
    DirectBrain brain);

__device__ void persistent_development_phase_device(
    cooperative_groups::grid_group grid,
    const ResidentAdultEpochSnapshot& epoch_snap,
    std::uint32_t current_tick,
    std::uint32_t global_tid,
    std::uint32_t total_threads,
    std::uint32_t node_count,
    std::uint32_t route_capacity,
    EligibilityRecord* eligibility_table,
    std::uint32_t* live_eligibility_count,
    std::uint32_t* route_causal_pin_bits,
    AdultExecutionConfig config,
    ResidentDevelopmentWorkspace resident_development,
    DirectBrain brain);

__device__ void persistent_execution_phase_device(
    cooperative_groups::grid_group grid,
    std::uint32_t current_tick,
    std::uint32_t global_tid,
    std::uint32_t total_threads,
    std::uint32_t participation_slot_count,
    DirectNode* nodes,
    DirectRoute* routes,
    std::uint32_t node_count,
    std::uint32_t route_capacity,
    ResidentActualFrontier* actual_frontier,
    ResidentActivationSoaPlane* activation_plane,
    EligibilityRecord* eligibility_table,
    std::uint32_t* live_eligibility_count,
    std::uint64_t* eligibility_claim_directory,
    std::uint32_t* eligibility_record_generations,
    AttractorBasinState* attractor_state,
    std::int32_t* node_incoming_excitation,
    std::int32_t* node_slow_context_q16,
    DelayedSparsePacket* delayed_packets,
    std::uint32_t* delayed_packet_live_count,
    std::uint32_t* delayed_packet_free_head,
    std::uint32_t* delayed_packet_next_free,
    DelayedPacketIdentity* delayed_packet_identities,
    RouteTransportPhaseView route_transport,
    std::uint64_t* route_opportunity_incarnations,
    NodeCausalParticipation* node_active_participation,
    NodeCausalParticipation* node_next_participation,
    std::uint32_t* node_next_participation_locks,
    std::uint32_t* node_active_ancestry_incomplete,
    std::uint32_t* node_next_ancestry_incomplete,
    DirectParticipationDescriptor* participation_staging,
    std::uint32_t* participation_staging_count,
    std::uint32_t participation_staging_capacity,
    AdultCoreMetrics* device_metrics,
    AdultExecutionConfig config,
    DirectBrain brain);

}  // namespace substrate::direct_adult_core

#endif  // HARDWARE_NATIVE_DIRECT_ADULT_PERSISTENT_PHASES_CUH
