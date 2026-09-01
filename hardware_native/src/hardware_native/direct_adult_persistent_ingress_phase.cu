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

__device__ __noinline__ void persistent_ingress_phase_device(
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
    DirectBrain brain) {
    // 2. Sensory Ingress Ingestion (Canonical device helper)
    const std::uint32_t ing_count = epoch_snap.ingress_tail - epoch_snap.ingress_head;
    if (grid.thread_rank() == 0u && ing_count != 0u && exact_history != nullptr) {
      direct_network::begin_exact_history_phase(
          exact_history, direct_network::DirectExactHistoryKind::sensory_contact,
          ing_count, current_tick);
    }
    grid.sync();
    const bool ingress_admitted = exact_history == nullptr ||
                                  exact_history->phase_admitted != 0u;
    if (ing_count != 0u && ing_count <= kMaxIngressQueueSize && ingress_queue != nullptr) {
      for (std::uint32_t idx = ingress_admitted ? global_tid : ing_count;
           idx < ing_count; idx += total_threads) {
        const std::uint32_t slot = (epoch_snap.ingress_head + idx) % kMaxIngressQueueSize;
        const ActivityEvent ev = ingress_queue[slot];
        const bool membrane_authenticated =
            ingress_contact_credentials != nullptr &&
            resident_contact_credential_valid(
                brain, ev, ingress_contact_credentials[slot],
                static_cast<std::uint64_t>(epoch_snap.ingress_head) + idx + 1u);
        device_process_single_sensory_event(
            nodes, boundary_ports, boundary_port_count, node_count, ev,
            node_incoming_excitation, node_slow_context_q16, node_active_participation,
            node_active_participation_locks, claim_incarnation_counter,
            ticket_table_locks,
            ticket_table, action_occurrences, participation_staging,
            participation_staging_count, participation_staging_capacity,
            device_metrics, current_tick, config.eligibility_horizon_ticks,
            exact_history != nullptr
                ? &exact_history->records[exact_history->phase_base + idx]
                : nullptr,
            membrane_authenticated,
            membrane_authenticated
                ? resident_contact_authority_incarnation(
                      ingress_contact_credentials[slot], ev)
                : 0u);
      }
    }

    grid.sync();
    if (grid.thread_rank() == 0u && ing_count != 0u &&
        exact_history != nullptr && ingress_admitted)
      direct_network::finish_exact_history_phase(exact_history);
    grid.sync();

    if (grid.thread_rank() == 0u && ingress_admitted &&
        ingress_contact_credentials != nullptr &&
        actual_frontier != nullptr) {
      for (std::uint32_t idx = 0u; idx < ing_count; ++idx) {
        const std::uint32_t slot =
            (epoch_snap.ingress_head + idx) % kMaxIngressQueueSize;
        admit_and_reconcile_resident_contact(
            brain, ingress_queue[slot], ingress_contact_credentials + slot,
            static_cast<std::uint64_t>(epoch_snap.ingress_head) + idx + 1u,
            node_active_participation, current_tick,
            config.eligibility_horizon_ticks, actual_frontier,
            mismatch_omission_frontier,
            device_metrics != nullptr
                ? &device_metrics->actual_frontier_causal
                : nullptr);
      }
      publish_admitted_resident_actual_contacts(
          brain, actual_frontier, node_active_participation,
          node_active_participation_locks, eligibility_table,
          live_eligibility_count, eligibility_claim_directory,
          eligibility_record_generations, route_transport.free_slots,
          route_transport.scan_a, participation_staging,
          participation_staging_count, participation_staging_capacity,
          node_incoming_excitation, device_metrics, current_tick);
    }
    if (grid.thread_rank() == 0u && activation_plane != nullptr &&
        ing_count != 0u)
      ++activation_plane->control.frontier_mutations;
    if (grid.thread_rank() == 0u && ing_count != 0u &&
        actual_frontier != nullptr) {
      condense_resident_actual_frontier_dispatch(brain, actual_frontier);
      if (activation_plane != nullptr)
        ++activation_plane->control.frontier_mutations;
    }
    if (grid.thread_rank() == 0u && actual_frontier != nullptr &&
        activation_plane != nullptr)
      refresh_resident_actual_frontier_activation_soa(
          *actual_frontier, &activation_plane->control,
          &activation_plane->soa);
    if (grid.thread_rank() == 0u && ing_count != 0u &&
        actual_frontier != nullptr)
      refresh_resident_causal_credit_predictions(
          brain, *actual_frontier, current_tick, causal_credit_predictions,
          activation_plane);
    if (grid.thread_rank() == 0u && ing_count != 0u &&
        actual_frontier != nullptr && causal_credit_predictions != nullptr)
      refresh_resident_mismatch_expectations(
          brain, *actual_frontier, *causal_credit_predictions,
          mismatch_omission_frontier);

    grid.sync();

    if (grid.thread_rank() == 0u && ing_count != 0u) {
      commit_resident_mismatch_credit_receipts(brain,
                                               mismatch_omission_frontier,
                                               current_tick,
                                               actual_frontier);
    }
    grid.sync();

    // Advance consumed head ONLY if valid
    if (grid.thread_rank() == 0 && ingress_control != nullptr && ing_count != 0u &&
        epoch_snap.ingress_fault == 0u) {
      ingress_control->consumed_head = epoch_snap.ingress_tail;
    }

}

}  // namespace substrate::direct_adult_core
