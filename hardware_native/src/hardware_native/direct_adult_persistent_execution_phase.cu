#include "hardware_native/direct_adult_persistent_phases.cuh"
#include "hardware_native/direct_adult_runtime_frontiers.cuh"
#include "hardware_native/direct_adult_device_ops.cuh"
#include "hardware_native/direct_adult_dense_execution.cuh"
#include "hardware_native/direct_adult_actual_frontier_condensation.cuh"
#include "hardware_native/direct_bounded_sm_workspace.cuh"
#include "hardware_native/direct_adult_activation.cuh"
#include "hardware_native/direct_adult_affect_body.cuh"
#include "hardware_native/direct_adult_wanting_liking_dissociation.cuh"
#include "hardware_native/direct_adult_incentive_salience.cuh"
#include "hardware_native/direct_adult_resource_maintenance.cuh"

namespace substrate::direct_adult_core {
#include "hardware_native/direct_adult_delayed_sparse_schedule.cuh"
#include "hardware_native/direct_adult_delayed_sparse_delivery.cuh"

__device__ __noinline__ void persistent_execution_phase_device(
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
    DirectBrain brain) {
    // 5. Freeze frontier_t into a carry-preserving write-only next bank.
    for (std::uint32_t index = global_tid; index < participation_slot_count;
         index += total_threads) {
      NodeCausalParticipation value =
          read_participation_slot(node_active_participation + index);
      if (value.ticket_id == 0ULL || value.ticket_id == kInvalidTicket ||
          value.expiry_tick < current_tick)
        value = NodeCausalParticipation{};
      value.current_drive = 0u;
      publish_participation_slot(node_next_participation + index, value);
    }
    for (std::uint32_t node = global_tid; node < node_count; node += total_threads) {
      node_next_participation_locks[node] = 0u;
      if (node_next_ancestry_incomplete != nullptr)
        node_next_ancestry_incomplete[node] = 0u;
    }
    grid.sync();

    // 6. Due matter retires before current physical routes propose. Both
    // delayed and zero-delay claims pass through the same deterministic batch.
    execute_deterministic_route_transport(
        grid, nodes, routes, brain.route_incarnations,
        route_opportunity_incarnations, node_count, route_capacity,
        node_incoming_excitation, delayed_packets,
        delayed_packet_live_count, delayed_packet_free_head,
        delayed_packet_next_free, delayed_packet_identities,
        node_active_participation, node_next_participation,
        node_active_ancestry_incomplete, node_next_ancestry_incomplete,
        participation_staging, participation_staging_count,
        participation_staging_capacity, eligibility_table,
        live_eligibility_count, eligibility_claim_directory,
        eligibility_record_generations, route_transport, current_tick,
        config.eligibility_decay_q16, config.eligibility_horizon_ticks,
        config.honor_inhibitory_sign, device_metrics);
    if (global_tid == 0u && actual_frontier != nullptr &&
        participation_staging != nullptr && participation_staging_count != nullptr) {
      bool frontier_changed = admit_resident_actual_frontier_route_descendant(
          brain, participation_staging, participation_staging_count,
          participation_staging_capacity, current_tick, actual_frontier);
      if (frontier_changed) {
        condense_resident_actual_frontier_dispatch(brain, actual_frontier);
        if (activation_plane != nullptr)
          ++activation_plane->control.frontier_mutations;
      }
    }
    grid.sync();

    // Dense blocks are ordinary resident morphology. Persistent execution must
    // apply the same warp primitive as the stepped executor before integrating
    // node activation; omitting this phase changes the adult by executor.
    __shared__ __align__(32) half dense_shared_weights[kDirectDenseTileElements];
    __shared__ __align__(32) half dense_shared_activation[kDirectDenseTileElements];
    __shared__ __align__(32) float dense_shared_output[kDirectDenseTileElements];
    if (threadIdx.x < 32u && brain.dense_blocks != nullptr &&
        brain.dense_weight_fp16_bits != nullptr) {
      for (std::uint32_t dense = blockIdx.x; dense < brain.dense_block_count;
           dense += gridDim.x) {
        execute_dense_tensor_block(
            nodes, brain.dense_blocks, dense, brain.dense_weight_fp16_bits,
            node_incoming_excitation, node_next_ancestry_incomplete,
            participation_staging, participation_staging_count,
            participation_staging_capacity, current_tick, device_metrics,
            threadIdx.x, dense_shared_weights, dense_shared_activation,
            dense_shared_output);
      }
    }
    grid.sync();

    // 5. Node Activation Integration
    if (nodes != nullptr && node_incoming_excitation != nullptr) {
      for (std::uint32_t n = global_tid; n < node_count; n += total_threads) {
        DirectNode& node = nodes[n];
        const std::int32_t activation = integrate_adult_node_activation(
            node, node_incoming_excitation + n,
            node_slow_context_q16 != nullptr ? node_slow_context_q16 + n : nullptr,
            current_tick, config.refractory_period_ticks,
            config.attractor_coupling_gain_q16,
            config.persistent_bias_ceiling_q16,
            config.slow_context_ceiling_q16);
        record_activation_observer(attractor_state, node.territory_index, activation);
      }
    }

    grid.sync();

    // 6. Observer-only activation summary
    if (attractor_state != nullptr && global_tid < kMaxBasins) {
      step_activation_observer(attractor_state, global_tid);
    }

    grid.sync();

}

}  // namespace substrate::direct_adult_core
