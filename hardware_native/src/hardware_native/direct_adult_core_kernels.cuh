#pragma once
#include "hardware_native/direct_adult_core.cuh"

namespace substrate::direct_adult_core {

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
    AdultExecutionConfig config, AdultCoreMetrics* metrics);

__global__ void derive_ingress_eligibility_frontier_kernel(
    EligibilityRecord* eligibility_table,
    std::uint32_t* eligibility_record_generations,
    std::uint32_t* live_eligibility_count, std::uint32_t current_tick,
    std::uint32_t* scan, std::uint32_t* scratch,
    std::uint32_t* free_slots, AdultCoreMetrics* metrics);

__global__ void commit_consumed_head_kernel(
    std::uint32_t* ingress_consumed_head,
    IngressRingControl* ingress_control,
    std::uint32_t tail_snapshot);

__global__ void clear_consumed_participation_staging_kernel(
    DirectParticipationDescriptor* staging, const std::uint32_t* count,
    std::uint32_t capacity);

__global__ void ingest_sensory_events_kernel(
    DirectNode* nodes,
    const DirectBoundaryPort* ports,
    std::uint32_t port_count,
    std::uint32_t node_count,
    const ActivityEvent* ingress_queue,
    std::uint32_t head_snapshot,
    std::uint32_t tail_snapshot,
    std::int32_t* node_incoming_excitation,
    std::int32_t* node_slow_context_q16,
    NodeCausalParticipation* node_active_participation,
    std::uint32_t* node_active_participation_locks,
    std::uint32_t* claim_incarnation_counter,
    std::uint32_t* ticket_table_locks,
    AsynchronousTicket* ticket_table,
    DirectActionOccurrence* action_occurrences,
    DirectParticipationDescriptor* current_contributions,
    std::uint32_t* current_contribution_count,
    std::uint32_t current_contribution_capacity,
    EligibilityRecord* eligibility_table,
    std::uint32_t* live_eligibility_count,
    std::uint64_t* eligibility_claim_directory,
    std::uint32_t* eligibility_record_generations,
    const std::uint32_t* eligibility_free_slots,
    std::uint32_t* eligibility_free_state,
    AdultCoreMetrics* metrics,
    std::uint32_t current_tick,
    std::uint32_t horizon_ticks,
    ResidentDevelopmentState* development,
    ResidentContactEpochReceipt* contact_credentials,
    ResidentActualFrontier* actual_frontier, ResidentMismatchOmissionFrontier* mismatch_frontier,
    DirectBrain brain,
    std::uint32_t* /*ingress_head_deprecated*/);

__global__ void condense_resident_actual_frontier_kernel(
    DirectBrain brain, ResidentActualFrontier* frontier);

__global__ void advance_actual_frontier_from_propagation_kernel(
    DirectBrain brain, const DirectParticipationDescriptor* contributions,
    const std::uint32_t* contribution_count,
    std::uint32_t contribution_capacity, std::uint32_t current_tick,
    ResidentActualFrontier* frontier);

__global__ void prepare_participation_frontier_kernel(
    const NodeCausalParticipation* active,
    NodeCausalParticipation* next,
    std::uint32_t* next_locks,
    std::uint32_t* next_ancestry_incomplete,
    std::uint32_t node_count,
    std::uint32_t current_tick);

__global__ void commit_participation_frontier_kernel(
    NodeCausalParticipation* active,
    NodeCausalParticipation* next,
    std::uint32_t* active_locks,
    std::uint32_t* next_locks,
    std::uint32_t* active_ancestry_incomplete,
    std::uint32_t* next_ancestry_incomplete,
    std::uint32_t node_count);

__global__ void advance_and_commit_resident_mismatch_kernel(
    DirectBrain brain, ResidentActualFrontier* actual_frontier,
    ResidentMismatchOmissionFrontier* mismatch, std::uint32_t current_tick);
__global__ void commit_resident_mismatch_credit_kernel(
    DirectBrain brain, ResidentActualFrontier* actual_frontier,
    ResidentMismatchOmissionFrontier* mismatch, std::uint32_t current_tick);

__global__ void publish_admitted_actual_contacts_kernel(
    DirectBrain brain, NodeCausalParticipation* active, std::uint32_t* active_locks,
    EligibilityRecord* eligibility, std::uint32_t* live_eligibility, std::uint64_t* claim_directory,
    std::uint32_t* record_generations,
    const std::uint32_t* eligibility_free_slots,
    std::uint32_t* eligibility_free_state,
    DirectParticipationDescriptor* staging,
    std::uint32_t* staging_count, std::uint32_t staging_capacity, std::int32_t* incoming,
    AdultCoreMetrics* metrics, std::uint32_t tick, ResidentActualFrontier* frontier);

__global__ void resolve_pending_actual_contacts_kernel(
    DirectBrain brain, const NodeCausalParticipation* active,
    std::uint32_t tick, ResidentActualFrontier* frontier, AdultCoreMetrics* metrics);

__global__ void execute_dense_tensor_wmma_kernel(
    const DirectNode* nodes,
    const DirectDenseBlock* dense_blocks,
    std::uint32_t dense_block_count,
    const std::uint16_t* dense_weights_fp16,
    std::int32_t* node_incoming_excitation,
    std::uint32_t* node_next_ancestry_incomplete,
    DirectParticipationDescriptor* participation_staging,
    std::uint32_t* participation_staging_count,
    std::uint32_t participation_staging_capacity,
    std::uint32_t current_tick,
    AdultCoreMetrics* metrics);

__global__ void integrate_node_activation_kernel(
    DirectNode* nodes,
    std::uint32_t node_count,
    std::int32_t* node_incoming_excitation,
    std::int32_t* node_slow_context_q16,
    AttractorBasinState* attractor_state,
    std::uint32_t current_tick,
    std::uint32_t refractory_period,
    std::int32_t attractor_coupling_gain_q16,
    std::int32_t persistent_bias_ceiling_q16,
    std::int32_t slow_context_ceiling_q16);

__global__ void decay_node_slow_context_kernel(
    std::int32_t* node_slow_context_q16, std::uint32_t node_count,
    std::int32_t decay_q16);

__global__ void step_attractor_basins_kernel(
    AttractorBasinState* attractor_state,
    std::uint32_t basin_count);

__global__ void derive_affect_body_state_kernel(
    direct_network::DirectAffectBodyState* state,
    const AsynchronousTicket* tickets, ResidentDevelopmentState* development);

__global__ void derive_wanting_liking_state_kernel(
    direct_network::ResidentWantingLikingProfileV1* state,
    const DirectActionOccurrence* actions,
    const DirectActionParticipationLink* links, std::uint32_t link_count,
    const direct_network::DirectAffectBodyState* affect,
    const direct_network::ResidentRecipeCell* recipes,
    std::uint32_t recipe_count);

__global__ void derive_resident_causal_world_model_kernel(
    direct_network::DirectCausalWorldModel* model,
    const AsynchronousTicket* tickets, const std::uint32_t* ticket_count,
    ResidentDevelopmentState* development,
    const direct_network::ResidentPostbirthConstructorState* constructor_state,
    const DirectActionOccurrence* actions,
    const DirectActionParticipationLink* action_links,
    std::uint32_t action_participant_capacity);

__global__ void resolve_consequence_ticket_kernel(
    AsynchronousTicket* ticket_table,
    const std::uint32_t* /*ticket_count*/,
    DirectActionOccurrence* action_occurrences,
    DirectActionParticipationLink* action_participation_links,
    std::uint32_t action_participant_capacity,
    std::uint64_t target_ticket_id,
    Word returned_word,
    CausalOrigin origin,
    std::uint32_t admission_tick,
    ResidentDevelopmentState* development,
    std::uint32_t transport_cursor,
    DirectBrain brain,
    ResidentActualFrontier* actual_frontier,
    const ResidentMultiHorizonPredictionFrontier* causal_credit_predictions,
    DirectNode* nodes,
    std::uint32_t node_count,
    DirectRoute* routes,
    const std::uint64_t* route_incarnations,
    std::uint32_t route_capacity,
    substrate::direct_adult::DirectRetentionState* retention_bank,
    direct_network::ResidentRecipeCell* recipe_cells,
    std::uint32_t recipe_cell_count,
    DirectDenseBlock* dense_blocks,
    std::uint32_t dense_block_count,
    std::int32_t learning_rate_q16,
    std::int32_t shatter_threshold_q16,
    ResolvedConsequenceContext* out_ctx,
    AdultCoreMetrics* metrics,
    ResidentCueSalienceTable* cue_salience);

__global__ void assimilate_consequence_and_credit_kernel(
    DirectBrain brain,
    DirectNode* nodes,
    std::uint32_t node_count,
    DirectRoute* routes,
    const std::uint64_t* route_incarnations,
    std::uint32_t route_capacity,
    DirectDenseBlock* dense_blocks,
    std::uint32_t dense_block_count,
    EligibilityRecord* eligibility_table,
    const std::uint32_t* live_eligibility_count,
    const ResolvedConsequenceContext* resolved_ctx,
    const DirectActionParticipationLink* action_participation_links,
    std::uint32_t action_participant_capacity,
    std::int32_t learning_rate_q16,
    std::int32_t shatter_threshold_q16,
    AdultCoreMetrics* metrics,
    std::uint32_t current_tick);

__global__ void mark_causally_pinned_routes_kernel(const EligibilityRecord* eligibility_table,
                                                   const std::uint32_t* live_eligibility_count,
                                                   const std::uint64_t* route_incarnations,
                                                   std::uint32_t route_capacity,
                                                   std::uint32_t current_tick,
                                                   std::uint32_t* pin_bits,
                                                   AdultCoreMetrics* metrics);

__global__ void resident_self_compilation_crystallize_kernel(
    const DirectNode* nodes,
    DirectDenseBlock* dense_blocks,
    std::uint32_t dense_block_count,
    AdultCoreMetrics* metrics);

__global__ void charge_adult_runtime_working_set_kernel(
    substrate::direct_adult::DirectResourceEcologyState* ecology,
    std::uint64_t eligibility_units, std::uint64_t packet_units, std::uint64_t ticket_units,
    std::uint64_t unpooled_bytes, std::uint32_t* admitted_out);

__global__ void release_adult_runtime_working_set_kernel(
    substrate::direct_adult::DirectResourceEcologyState* ecology,
    std::uint64_t eligibility_units, std::uint64_t packet_units, std::uint64_t ticket_units,
    std::uint64_t unpooled_bytes);

__global__ void emit_motor_events_kernel(
    DirectBrain brain,
    const DirectNode* nodes,
    const DirectRoute* routes,
    const std::uint64_t* route_incarnations,
    std::uint32_t route_capacity,
    const DirectBoundaryPort* ports,
    const ResidentActualFrontier* actual_frontier,
    ResidentActivationSoaPlane* activation_plane,
    std::uint32_t require_exact_occurrence_identity,
    std::uint32_t port_count,
    MotorEvent* egress_queue,
    const std::uint32_t* egress_head,
    std::uint32_t* egress_tail,
    AsynchronousTicket* ticket_table,
    std::uint32_t* ticket_count,
    std::uint32_t* ticket_table_locks,
    DirectActionOccurrence* action_occurrences,
    DirectActionParticipationLink* action_participation_links,
    std::uint32_t action_participant_capacity,
    ResidentPublicMotorTrajectory* resident_motor_trajectory,
    std::uint32_t resident_motor_trajectory_capacity,
    direct_network::DirectAffectBodyState* affect_body_state,
    std::uint32_t action_horizon_ticks,
    const DirectParticipationDescriptor* current_contributions,
    const std::uint32_t* current_contribution_count,
    std::uint32_t current_contribution_capacity,
    const std::uint32_t* phase_ancestry_incomplete,
    const EligibilityRecord* eligibility_table,
    const std::uint32_t* eligibility_record_generations,
    ResidentDevelopmentState* development,
    AdultCoreMetrics* metrics,
    std::uint32_t current_tick,
    DirectEfferenceCopy* efference_ring,
    std::uint32_t* efference_head,
    std::uint32_t* efference_tail,
    std::int32_t* node_incoming_excitation,
    std::uint32_t route_efference_copies);

}  // namespace substrate::direct_adult_core
