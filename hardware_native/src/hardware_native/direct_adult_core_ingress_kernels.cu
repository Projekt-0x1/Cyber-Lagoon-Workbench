#include <cuda_runtime.h>
#include "hardware_native/direct_adult_core_kernels.cuh"
#include "hardware_native/direct_boundary_condensation.cuh"
#include "hardware_native/direct_adult_checkpoint.cuh"
#include "hardware_native/direct_adult_device_ops.cuh"
#include "hardware_native/direct_adult_incentive_salience.cuh"
#include "hardware_native/direct_adult_activation.cuh"
#include "hardware_native/direct_adult_affect_body.cuh"
#include "hardware_native/direct_adult_wanting_liking_dissociation.cuh"
#include "hardware_native/direct_adult_temporal_discounting_competition.cuh"
#include "hardware_native/direct_adult_uncertainty_plateau.cuh"
#include "hardware_native/direct_adult_bounded_fanout.cuh"
#include "hardware_native/direct_adult_resource_maintenance.cuh"

namespace substrate::direct_adult_core {
__global__ void commit_consumed_head_kernel(
    std::uint32_t* ingress_consumed_head,
    IngressRingControl* ingress_control,
    std::uint32_t tail_snapshot) {
  if (ingress_consumed_head != nullptr) {
    *ingress_consumed_head = tail_snapshot;
  }
  if (ingress_control != nullptr) {
    ingress_control->consumed_head = tail_snapshot;
  }
}

// Participation staging is phase scratch.  Every consumer has finished before
// this runs, so retaining its scheduler-order bytes at a checkpoint boundary
// would add no causal state and would make exact replay depend on block order.
__global__ void clear_consumed_participation_staging_kernel(
    DirectParticipationDescriptor* staging, const std::uint32_t* count,
    std::uint32_t capacity) {
  if (staging == nullptr || count == nullptr) return;
  const std::uint32_t width = *count < capacity ? *count : capacity;
  for (std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < width;
       i += blockDim.x * gridDim.x)
    staging[i] = DirectParticipationDescriptor{};
}

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
    std::uint32_t* /*ingress_head_deprecated*/) {
  const std::uint32_t total_events = tail_snapshot - head_snapshot;
  if (total_events == 0u || total_events > kMaxIngressQueueSize) return;

  direct_network::DirectExactHistoryHotPage* history =
      development != nullptr ? &development->exact_history : nullptr;
  if (threadIdx.x == 0u && history != nullptr)
    direct_network::begin_exact_history_phase(
        history, direct_network::DirectExactHistoryKind::sensory_contact,
        total_events, current_tick);
  __syncthreads();
  if (history != nullptr && history->phase_admitted == 0u) return;

  for (std::uint32_t idx = threadIdx.x; idx < total_events; idx += blockDim.x) {
    const std::uint32_t event_idx = (head_snapshot + idx) % kMaxIngressQueueSize;
    const ActivityEvent event = ingress_queue[event_idx];
    const bool membrane_authenticated = contact_credentials != nullptr &&
        resident_contact_credential_valid(
            brain, event, contact_credentials[event_idx],
            static_cast<std::uint64_t>(head_snapshot) + idx + 1u);
    device_process_single_sensory_event(
        nodes, ports, port_count, node_count, event,
        node_incoming_excitation, node_slow_context_q16, node_active_participation,
        node_active_participation_locks, claim_incarnation_counter,
        ticket_table_locks,
        ticket_table, action_occurrences, current_contributions,
        current_contribution_count, current_contribution_capacity,
        metrics, current_tick, horizon_ticks,
        history != nullptr ? &history->records[history->phase_base + idx] : nullptr,
        membrane_authenticated,
        membrane_authenticated
            ? resident_contact_authority_incarnation(
                  contact_credentials[event_idx], event)
            : 0u);
  }
  __syncthreads();
  if (threadIdx.x == 0u && history != nullptr)
    direct_network::finish_exact_history_phase(history);
  __syncthreads();
  if (threadIdx.x == 0u && contact_credentials != nullptr &&
      actual_frontier != nullptr) {
    for (std::uint32_t idx = 0u; idx < total_events; ++idx) {
      const std::uint32_t event_idx =
          (head_snapshot + idx) % kMaxIngressQueueSize;
      admit_and_reconcile_resident_contact(
          brain, ingress_queue[event_idx], contact_credentials + event_idx,
          static_cast<std::uint64_t>(head_snapshot) + idx + 1u,
          node_active_participation, current_tick, horizon_ticks,
          actual_frontier, mismatch_frontier,
          metrics != nullptr ? &metrics->actual_frontier_causal : nullptr);
    }
    publish_admitted_resident_actual_contacts(
        brain, actual_frontier, node_active_participation,
        node_active_participation_locks, eligibility_table,
        live_eligibility_count, eligibility_claim_directory,
        eligibility_record_generations, eligibility_free_slots,
        eligibility_free_state, current_contributions,
        current_contribution_count, current_contribution_capacity,
        node_incoming_excitation, metrics, current_tick);
  }
}

__global__ void prepare_participation_frontier_kernel(
    const NodeCausalParticipation* active,
    NodeCausalParticipation* next,
    std::uint32_t* next_locks,
    std::uint32_t* next_ancestry_incomplete,
    std::uint32_t node_count,
    std::uint32_t current_tick) {
  const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  const std::uint32_t slot_count = node_count * kNodeParticipationAperture;
  if (index < slot_count) {
    NodeCausalParticipation value = read_participation_slot(active + index);
    if (value.ticket_id == 0ULL || value.ticket_id == kInvalidTicket ||
        value.expiry_tick < current_tick)
      value = NodeCausalParticipation{};
    value.current_drive = 0u;
    publish_participation_slot(next + index, value);
  }
  if (index < node_count) {
    if (next_locks != nullptr) next_locks[index] = 0u;
    if (next_ancestry_incomplete != nullptr) next_ancestry_incomplete[index] = 0u;
  }
}
__global__ void commit_participation_frontier_kernel(
    NodeCausalParticipation* active,
    NodeCausalParticipation* next,
    std::uint32_t* active_locks,
    std::uint32_t* next_locks,
    std::uint32_t* active_ancestry_incomplete,
    std::uint32_t* next_ancestry_incomplete,
    std::uint32_t node_count) {
  const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  const std::uint32_t slot_count = node_count * kNodeParticipationAperture;
  if (index < slot_count) {
    publish_participation_slot(active + index,
                               read_participation_slot(next + index));
    publish_participation_slot(next + index, NodeCausalParticipation{});
  }
  if (index < node_count) {
    if (active_locks != nullptr) active_locks[index] = 0u;
    if (next_locks != nullptr) next_locks[index] = 0u;
    if (active_ancestry_incomplete != nullptr && next_ancestry_incomplete != nullptr) {
      active_ancestry_incomplete[index] = next_ancestry_incomplete[index];
      next_ancestry_incomplete[index] = 0u;
    }
  }
}
}  // namespace substrate::direct_adult_core
