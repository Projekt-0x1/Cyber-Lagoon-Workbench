#include <cuda_runtime.h>
#include <algorithm>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#include "hardware_native/direct_adult_core.cuh"
#include "hardware_native/direct_adult_core_kernels.cuh"
#include "hardware_native/direct_adult_runtime_frontiers.cuh"
#include "hardware_native/direct_adult_recipe_credit.cuh"

#include "hardware_native/direct_adult_checkpoint.cuh"
#include "hardware_native/direct_adult_device_ops.cuh"
#include "hardware_native/direct_adult_incentive_salience.cuh"
#include "hardware_native/direct_adult_activation.cuh"
#include "hardware_native/direct_adult_action_control_runtime_abi.cuh"
#include "hardware_native/direct_adult_affect_body.cuh"
#include "hardware_native/direct_adult_causal_world_model.cuh"
#include "hardware_native/direct_adult_language_phase_wiring.cuh"
#include "hardware_native/direct_adult_language_candidate_phase.cuh"
#include "hardware_native/direct_adult_action_commit_phase.cuh"
#include "hardware_native/direct_adult_resident_language_runtime_abi.cuh"
#include "hardware_native/direct_adult_wanting_liking_dissociation.cuh"
#include "hardware_native/direct_adult_temporal_discounting_competition.cuh"
#include "hardware_native/direct_adult_uncertainty_plateau.cuh"
#include "hardware_native/direct_adult_bounded_fanout.cuh"
#include "hardware_native/direct_adult_resource_maintenance.cuh"
namespace substrate::direct_adult_core {
__global__ void advance_and_commit_resident_mismatch_kernel(
    DirectBrain brain, ResidentActualFrontier* actual_frontier,
    ResidentMismatchOmissionFrontier* mismatch, std::uint32_t current_tick) {
  if (blockIdx.x == 0u && threadIdx.x == 0u) {
    advance_resident_mismatch_omissions(current_tick, actual_frontier, mismatch);
    commit_resident_mismatch_credit_receipts(
        brain, mismatch, current_tick, actual_frontier);
    expire_resident_actual_frontier(actual_frontier, current_tick);
  }
}
__global__ void commit_resident_mismatch_credit_kernel(
    DirectBrain brain, ResidentActualFrontier* actual_frontier,
    ResidentMismatchOmissionFrontier* mismatch, std::uint32_t current_tick) {
  if (blockIdx.x == 0u && threadIdx.x == 0u)
    commit_resident_mismatch_credit_receipts(
        brain, mismatch, current_tick, actual_frontier);
}
namespace {
void check_cuda(cudaError_t status, const char* op) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(op) + ": " + cudaGetErrorString(status));
  }
}

#include "hardware_native/direct_adult_route_transport_runtime.inl"

const std::uint32_t* rebuild_route_causal_pins(DirectAdultRuntime* runtime,
                                               std::uint32_t block_dim,
                                               bool count_metrics) {
  const std::uint32_t route_capacity = runtime->brain->route_capacity;
  if (runtime->config.honor_causal_pin == 0u ||
      runtime->route_causal_pin_bits == nullptr || route_capacity == 0u)
    return nullptr;
  const std::uint32_t pin_words = (route_capacity + 31u) / 32u;
  check_cuda(cudaMemsetAsync(runtime->route_causal_pin_bits, 0,
                             sizeof(std::uint32_t) * pin_words, runtime->stream),
             "cudaMemsetAsync route_causal_pin_bits");
  const std::uint32_t mark_grid =
      (kMaxLiveEligibilityRecords + block_dim - 1u) / block_dim;
  mark_causally_pinned_routes_kernel<<<mark_grid, block_dim, 0, runtime->stream>>>(
      runtime->eligibility_table, runtime->live_eligibility_count,
      runtime->brain->route_incarnations, route_capacity, runtime->current_tick,
      runtime->route_causal_pin_bits,
      count_metrics ? runtime->device_metrics : nullptr);
  return runtime->route_causal_pin_bits;
}

}  // namespace

DirectAdultRuntime* create_direct_adult_runtime(DirectBrain* brain, const AdultExecutionConfig& config) {
  if (!brain) throw std::invalid_argument("create_direct_adult_runtime: brain is null");
  if (config.action_participant_capacity > kMaxActionParticipationLinks)
    throw std::invalid_argument(
        "create_direct_adult_runtime: action ancestry capacity exceeds bounded closure");
  if (config.resident_motor_trajectory_capacity >
          config.action_participant_capacity ||
      (config.action_participant_capacity != 0u &&
       config.resident_motor_trajectory_capacity == 0u) ||
      config.resident_motor_trajectory_capacity > kMaxProvenanceSlotsPerNode)
    throw std::invalid_argument(
        "create_direct_adult_runtime: motor trajectory capacity is invalid");

  AdultWorkingSetCharge charge = adult_working_set_charge(
      brain->node_count, brain->route_capacity, config.action_participant_capacity,
      brain->route_opportunity_incarnations == nullptr);
  charge.unpooled_bytes +=
      sizeof(ResidentPublicMotorTrajectory) +
      sizeof(direct_network::DirectCausalWorldModel) +
      direct_network::direct_resident_language_runtime_storage_bytes() +
      direct_adult_action_control_runtime_storage_bytes();
  if (brain->resource_ecology != nullptr) {
    substrate::direct_adult::DirectResourceEcologyState host_ecology{};
    check_cuda(cudaMemcpy(&host_ecology, brain->resource_ecology, sizeof(host_ecology), cudaMemcpyDeviceToHost), "read ecology");
    using substrate::direct_adult::DirectResourcePoolKind;
    auto& pools = host_ecology.pools;
    auto size_pool = [&pools](DirectResourcePoolKind kind, std::uint64_t units, std::uint64_t bytes_per_unit) {
      auto& pool = pools[static_cast<std::uint32_t>(kind)];
      if (pool.capacity_units == 0u) pool.capacity_units = units;
      if (pool.bytes_per_unit == 0u) {
        pool.bytes_per_unit = bytes_per_unit;
      } else if (pool.bytes_per_unit != bytes_per_unit) {
        throw std::runtime_error("create_direct_adult_runtime: resource pool byte ABI mismatch");
      }
    };
    size_pool(DirectResourcePoolKind::eligibility_record, charge.eligibility_units, adult_eligibility_bytes_per_unit());
    size_pool(DirectResourcePoolKind::delayed_packet, charge.packet_units, adult_packet_bytes_per_unit());
    size_pool(DirectResourcePoolKind::pending_consequence_ticket, charge.ticket_units, adult_ticket_bytes_per_unit());
    check_cuda(cudaMemcpy(brain->resource_ecology, &host_ecology, sizeof(host_ecology), cudaMemcpyHostToDevice), "publish bounds");

    std::uint32_t* admitted_device = nullptr;
    check_cuda(cudaMalloc(&admitted_device, sizeof(std::uint32_t)), "cudaMalloc admission");
    charge_adult_runtime_working_set_kernel<<<1, 32>>>(
        brain->resource_ecology, charge.eligibility_units, charge.packet_units,
        charge.ticket_units, charge.unpooled_bytes, admitted_device);
    check_cuda(cudaDeviceSynchronize(), "charge adult runtime");
    std::uint32_t admitted = 0u;
    check_cuda(cudaMemcpy(&admitted, admitted_device, sizeof(admitted), cudaMemcpyDeviceToHost), "read admission");
    cudaFree(admitted_device);
    if (admitted == 0u) {
      throw std::runtime_error("create_direct_adult_runtime: resource ecology refused working set");
    }
  }

  DirectAdultRuntime* runtime = new DirectAdultRuntime();
  check_cuda(cudaStreamCreate(&runtime->stream), "cudaStreamCreate adult step stream");
  check_cuda(cudaStreamCreateWithFlags(&runtime->transport_stream, cudaStreamNonBlocking), "cudaStreamCreate transport_stream");
  check_cuda(cudaStreamCreateWithFlags(&runtime->persistent_stream, cudaStreamNonBlocking), "cudaStreamCreate persistent_stream");
  runtime->resident_language =
      direct_network::create_direct_resident_language_runtime();
  runtime->action_control_runtime = create_direct_adult_action_control_runtime();

  runtime->brain = brain;
  runtime->owns_route_opportunity_incarnations = false;
  runtime->resident_development_epochs = 0u;
  runtime->config = config;
  runtime->host_ingress_write_tail = 0u;
  runtime->host_ingress_publish_tail = 0u;
  runtime->host_ingress_observed_head = 0u;
  runtime->host_ingress_dispatched_tail = 0u;
  runtime->host_ingress_overflow_drops = 0u;
  runtime->host_ingress_protocol_faults = 0u;
  runtime->current_tick = 0u;

  // Ingress Ring Control & Pinned Staging
  check_cuda(cudaMallocHost(reinterpret_cast<void**>(&runtime->host_ingress_staging), sizeof(ActivityEvent) * kMaxIngressQueueSize), "cudaMallocHost host_ingress_staging");
  check_cuda(cudaMallocHost(
      reinterpret_cast<void**>(&runtime->host_ingress_contact_staging),
      sizeof(ResidentContactEpochReceipt) * kMaxIngressQueueSize),
      "cudaMallocHost host_ingress_contact_staging");
  std::memset(runtime->host_ingress_contact_staging, 0,
              sizeof(ResidentContactEpochReceipt) * kMaxIngressQueueSize);
  runtime->host_boundary_ports = brain->boundary_port_count == 0u
      ? nullptr : new DirectBoundaryPort[brain->boundary_port_count];
  if (brain->boundary_port_count != 0u)
    check_cuda(cudaMemcpy(runtime->host_boundary_ports, brain->boundary_ports,
                          sizeof(DirectBoundaryPort) * brain->boundary_port_count,
                          cudaMemcpyDeviceToHost),
               "cache born boundary ports");
  check_cuda(cudaMallocHost(reinterpret_cast<void**>(&runtime->host_ingress_head_snapshot), sizeof(std::uint32_t)), "cudaMallocHost host_ingress_head_snapshot");
  *runtime->host_ingress_head_snapshot = 0u;

  check_cuda(cudaMallocHost(reinterpret_cast<void**>(&runtime->host_ingress_publish_slot_pinned), sizeof(std::uint32_t)), "cudaMallocHost host_ingress_publish_slot_pinned");
  *runtime->host_ingress_publish_slot_pinned = 0u;

  check_cuda(cudaEventCreate(&runtime->ingress_consumed_event), "cudaEventCreate ingress_consumed_event");

  check_cuda(cudaMalloc(&runtime->ingress_control, sizeof(IngressRingControl)), "cudaMalloc ingress_control");
  IngressRingControl init_ctrl{0u, 0u, kMaxIngressQueueSize, 0u};
  check_cuda(cudaMemcpy(runtime->ingress_control, &init_ctrl, sizeof(IngressRingControl), cudaMemcpyHostToDevice), "init ingress_control");

  auto alloc_zero = [&](auto** ptr, std::size_t bytes, const char* name) {
    check_cuda(cudaMalloc(reinterpret_cast<void**>(ptr), bytes), name);
    check_cuda(cudaMemset(*ptr, 0, bytes), name);
  };

  // Ingress queue
  alloc_zero(&runtime->ingress_queue, sizeof(ActivityEvent) * kMaxIngressQueueSize, "ingress_queue");
  alloc_zero(&runtime->ingress_contact_credentials,
             sizeof(ResidentContactEpochReceipt) * kMaxIngressQueueSize,
             "ingress_contact_credentials");
  alloc_zero(&runtime->actual_frontier, sizeof(ResidentActualFrontier),
             "actual_frontier");
  alloc_zero(&runtime->activation_plane, sizeof(ResidentActivationSoaPlane),
             "activation_plane");
  alloc_zero(&runtime->causal_credit_predictions,
             sizeof(ResidentMultiHorizonPredictionFrontier),
             "causal_credit_predictions");
  alloc_zero(&runtime->mismatch_omission,
             sizeof(ResidentMismatchOmissionRuntime), "mismatch_omission");
  runtime->ingress_consumed_head = &runtime->ingress_control->consumed_head;
  runtime->ingress_published_tail = &runtime->ingress_control->published_tail;

  // Consequence Transport Ring (gh #1208 / Patch 3)
  runtime->host_consequence_write_tail = 0u;
  runtime->host_consequence_publish_tail = 0u;
  runtime->host_consequence_observed_head = 0u;
  runtime->host_consequence_overflow_drops = 0u;
  runtime->host_consequence_protocol_faults = 0u;

  check_cuda(
      cudaMallocHost(
          reinterpret_cast<void**>(&runtime->host_consequence_staging),
          sizeof(ConsequenceIngressEvent) * kMaxAsynchronousTickets),
      "cudaMallocHost host_consequence_staging");
  check_cuda(
      cudaMallocHost(
          reinterpret_cast<void**>(&runtime->host_consequence_head_snapshot),
          sizeof(std::uint32_t)),
      "cudaMallocHost host_consequence_head_snapshot");
  *runtime->host_consequence_head_snapshot = 0u;

  check_cuda(
      cudaMallocHost(
          reinterpret_cast<void**>(&runtime->host_consequence_publish_slot_pinned),
          sizeof(std::uint32_t)),
      "cudaMallocHost host_consequence_publish_slot_pinned");
  *runtime->host_consequence_publish_slot_pinned = 0u;

  check_cuda(cudaEventCreate(&runtime->consequence_consumed_event), "cudaEventCreate consequence_consumed_event");

  alloc_zero(&runtime->consequence_queue, sizeof(ConsequenceIngressEvent) * kMaxAsynchronousTickets, "consequence_queue");
  check_cuda(cudaMalloc(&runtime->consequence_control, sizeof(IngressRingControl)), "cudaMalloc consequence_control");
  IngressRingControl init_cctrl{0u, 0u, kMaxAsynchronousTickets, 0u};
  check_cuda(cudaMemcpy(runtime->consequence_control, &init_cctrl, sizeof(IngressRingControl), cudaMemcpyHostToDevice), "init consequence_control");

  // Persistent Execution Control & Authority (gh #1208)
  runtime->execution_authority = AdultExecutionAuthority::host_stepped;
  runtime->is_persistent_running = false;
  alloc_zero(&runtime->device_stop_flag, sizeof(std::uint32_t), "device_stop_flag");
  alloc_zero(&runtime->device_epoch_limit, sizeof(std::uint32_t), "device_epoch_limit");
  check_cuda(cudaMallocManaged(&runtime->device_resident_tick, sizeof(std::uint32_t)), "cudaMallocManaged device_resident_tick");
  *runtime->device_resident_tick = 0u;
  alloc_zero(&runtime->device_epoch_snapshot, sizeof(ResidentAdultEpochSnapshot), "device_epoch_snapshot");
  check_cuda(cudaEventCreate(&runtime->transport_ingress_event), "cudaEventCreate transport_ingress_event");
  check_cuda(cudaEventCreate(&runtime->transport_consequence_event), "cudaEventCreate transport_consequence_event");
  check_cuda(cudaHostAlloc(reinterpret_cast<void**>(&runtime->host_observer_snapshot),
                           sizeof(AdultCoreObserverSnapshot), cudaHostAllocMapped),
             "cudaHostAlloc host_observer_snapshot");
  *runtime->host_observer_snapshot = {};
  check_cuda(cudaHostGetDevicePointer(
                 reinterpret_cast<void**>(&runtime->device_observer_snapshot),
                 runtime->host_observer_snapshot, 0u),
             "cudaHostGetDevicePointer observer snapshot");

  // Egress Ring
  alloc_zero(&runtime->egress_queue, sizeof(MotorEvent) * kMaxEgressQueueSize, "egress_queue");
  alloc_zero(&runtime->egress_head, sizeof(std::uint32_t), "egress_head");
  alloc_zero(&runtime->egress_tail, sizeof(std::uint32_t), "egress_tail");
  alloc_zero(&runtime->efference_ring, sizeof(DirectEfferenceCopy) * kMaxEfferenceRingSize, "efference_ring");
  alloc_zero(&runtime->efference_head, sizeof(std::uint32_t), "efference_head");
  alloc_zero(&runtime->efference_tail, sizeof(std::uint32_t), "efference_tail");

  // Eligibility Table
  alloc_zero(&runtime->eligibility_table, sizeof(EligibilityRecord) * kMaxLiveEligibilityRecords, "eligibility_table");
  alloc_zero(&runtime->live_eligibility_count, sizeof(std::uint32_t), "live_eligibility_count");
  alloc_zero(&runtime->eligibility_claim_directory, sizeof(std::uint64_t) * kEligibilityClaimDirectoryCapacity, "eligibility_claim_directory");
  alloc_zero(&runtime->eligibility_claim_locks, sizeof(std::uint32_t) * kEligibilityClaimLockCount, "eligibility_claim_locks");
  alloc_zero(&runtime->eligibility_record_generations, sizeof(std::uint32_t) * kMaxLiveEligibilityRecords, "eligibility_record_generations");

  // Ticket Table
  alloc_zero(&runtime->ticket_table, sizeof(AsynchronousTicket) * kMaxAsynchronousTickets, "ticket_table");
  alloc_zero(&runtime->ticket_count, sizeof(std::uint32_t), "ticket_count");
  alloc_zero(&runtime->ticket_table_locks, sizeof(std::uint32_t) * kMaxAsynchronousTickets, "ticket_table_locks");
  alloc_zero(&runtime->claim_incarnation_counter, sizeof(std::uint32_t),
             "claim_incarnation_counter");
  alloc_zero(&runtime->action_occurrences,
             sizeof(DirectActionOccurrence) * kMaxAsynchronousTickets,
             "action_occurrences");
  if (config.action_participant_capacity != 0u) {
    alloc_zero(&runtime->action_participation_links,
               sizeof(DirectActionParticipationLink) * kMaxAsynchronousTickets *
                   config.action_participant_capacity,
               "action_participation_links");
  } else {
    runtime->action_participation_links = nullptr;
  }
  alloc_zero(&runtime->resident_motor_trajectory,
             sizeof(ResidentPublicMotorTrajectory),
             "resident_motor_trajectory");

  // Resolved Consequence Context & Attractors
  // One allocation gives checkpoint capture one atomic buffer view while the
  // runtime retains explicit typed ownership of all resident states.
  alloc_zero(&runtime->resolved_consequence_ctx,
             sizeof(ResolvedConsequenceContext) +
                 sizeof(direct_network::DirectAffectBodyState) +
                 sizeof(direct_network::ResidentWantingLikingProfileV1) +
                 sizeof(direct_network::DirectCausalWorldModel),
             "resolved_consequence_affect_wanting_and_causal_state");
  runtime->affect_body_state =
      reinterpret_cast<direct_network::DirectAffectBodyState*>(
          runtime->resolved_consequence_ctx + 1u);
  runtime->wanting_liking_state =
      reinterpret_cast<direct_network::ResidentWantingLikingProfileV1*>(
          runtime->affect_body_state + 1u);
  runtime->causal_world_model =
      reinterpret_cast<direct_network::DirectCausalWorldModel*>(
          runtime->wanting_liking_state + 1u);
  alloc_zero(&runtime->attractor_state, sizeof(AttractorBasinState), "attractor_state");
  alloc_zero(&runtime->cue_salience_table, sizeof(ResidentCueSalienceTable),
             "cue_salience_table");

  // Node working buffers
  alloc_zero(&runtime->node_incoming_excitation, sizeof(std::int32_t) * brain->node_count, "node_incoming_excitation");
  alloc_zero(&runtime->node_slow_context_q16, sizeof(std::int32_t) * brain->node_count, "node_slow_context_q16");
  initialize_delayed_packet_transport(runtime);
  if (brain->route_capacity != 0u) {
    if (brain->route_opportunity_incarnations != nullptr) {
      runtime->route_opportunity_incarnations = brain->route_opportunity_incarnations;
    } else {
      // Compatibility only for manually-authored fixtures that predate the born
      // canonical arena. Canonical production brains always carry this state in-arena.
      alloc_zero(&runtime->route_opportunity_incarnations,
                 sizeof(std::uint64_t) * brain->route_capacity,
                 "route_opportunity_incarnations_fallback");
      runtime->owns_route_opportunity_incarnations = true;
    }
  } else {
    runtime->route_opportunity_incarnations = nullptr;
  }

  // Frozen frontier_t, write-only frontier_(t+1), and per-node writer locks.
  // Sized by node-participation aperture, not provenance producer width.
  const std::size_t prov_slots =
      brain->node_count * kNodeParticipationAperture;
  alloc_zero(&runtime->node_active_participation, sizeof(NodeCausalParticipation) * prov_slots, "node_active_participation");
  alloc_zero(&runtime->node_next_participation, sizeof(NodeCausalParticipation) * prov_slots, "node_next_participation");
  alloc_zero(&runtime->node_active_participation_locks, sizeof(std::uint32_t) * brain->node_count, "node_active_participation_locks");
  alloc_zero(&runtime->node_next_participation_locks, sizeof(std::uint32_t) * brain->node_count, "node_next_participation_locks");
  alloc_zero(&runtime->node_active_ancestry_incomplete, sizeof(std::uint32_t) * brain->node_count, "node_active_ancestry_incomplete");
  alloc_zero(&runtime->node_next_ancestry_incomplete, sizeof(std::uint32_t) * brain->node_count, "node_next_ancestry_incomplete");
  runtime->participation_staging_capacity = kMaxLiveEligibilityRecords;
  alloc_zero(&runtime->participation_staging,
             sizeof(DirectParticipationDescriptor) * runtime->participation_staging_capacity,
             "participation_staging");
  alloc_zero(&runtime->participation_staging_count, sizeof(std::uint32_t),
             "participation_staging_count");

  // Causal pin bitmap, one bit per route slot.
  const std::uint32_t pin_words = (brain->route_capacity + 31u) / 32u;
  if (pin_words != 0u) {
    alloc_zero(&runtime->route_causal_pin_bits, sizeof(std::uint32_t) * pin_words, "route_causal_pin_bits");
  } else {
    runtime->route_causal_pin_bits = nullptr;
  }

  // Device Metrics
  alloc_zero(&runtime->device_metrics, sizeof(AdultCoreMetrics), "device_metrics");

  runtime->resident_development =
      direct_network::create_resident_development_workspace(brain->node_count);

  return runtime;
}

void destroy_direct_adult_runtime(DirectAdultRuntime* runtime) {
  if (!runtime) return;

  if (runtime->is_persistent_running) {
    stop_persistent_direct_adult(runtime);
  }

  // 1. Synchronize all streams that may have work in flight
  if (runtime->persistent_stream != nullptr) {
    cudaStreamSynchronize(runtime->persistent_stream);
  }
  if (runtime->stream != nullptr) {
    cudaStreamSynchronize(runtime->stream);
  }
  if (runtime->transport_stream != nullptr) {
    cudaStreamSynchronize(runtime->transport_stream);
  }

  direct_network::destroy_resident_development_workspace(&runtime->resident_development);

  // 2. Free pinned host staging & snapshots
  if (runtime->host_ingress_staging != nullptr) cudaFreeHost(runtime->host_ingress_staging);
  if (runtime->host_ingress_contact_staging != nullptr)
    cudaFreeHost(runtime->host_ingress_contact_staging);
  delete[] runtime->host_boundary_ports;
  if (runtime->host_ingress_head_snapshot != nullptr) cudaFreeHost(runtime->host_ingress_head_snapshot);
  if (runtime->host_ingress_publish_slot_pinned != nullptr) cudaFreeHost(runtime->host_ingress_publish_slot_pinned);

  if (runtime->host_consequence_staging != nullptr) cudaFreeHost(runtime->host_consequence_staging);
  if (runtime->host_consequence_head_snapshot != nullptr) cudaFreeHost(runtime->host_consequence_head_snapshot);
  if (runtime->host_consequence_publish_slot_pinned != nullptr) cudaFreeHost(runtime->host_consequence_publish_slot_pinned);
  if (runtime->host_observer_snapshot != nullptr) cudaFreeHost(runtime->host_observer_snapshot);

  direct_network::destroy_direct_resident_language_runtime(
      runtime->resident_language);
  runtime->resident_language = nullptr;
  destroy_direct_adult_action_control_runtime(runtime->action_control_runtime);
  runtime->action_control_runtime = nullptr;

  // 3. Free device buffers
  void* dev_buffers[] = {
      runtime->ingress_queue,
      runtime->ingress_contact_credentials, runtime->actual_frontier,
      runtime->activation_plane,
      runtime->causal_credit_predictions,
      runtime->mismatch_omission,
      runtime->ingress_control, runtime->consequence_queue, runtime->consequence_control,
      runtime->egress_queue, runtime->egress_head, runtime->egress_tail,
      runtime->efference_ring, runtime->efference_head, runtime->efference_tail,
      runtime->eligibility_table, runtime->live_eligibility_count,
      runtime->eligibility_claim_directory, runtime->eligibility_claim_locks,
      runtime->eligibility_record_generations, runtime->ticket_table,
      runtime->ticket_count, runtime->ticket_table_locks,
      runtime->claim_incarnation_counter,
      runtime->action_occurrences,
      runtime->action_participation_links, runtime->resident_motor_trajectory,
      runtime->resolved_consequence_ctx,
      runtime->attractor_state, runtime->cue_salience_table,
      runtime->node_incoming_excitation, runtime->node_slow_context_q16,
      runtime->delayed_packets, runtime->delayed_packet_live_count,
      runtime->delayed_packet_free_head, runtime->delayed_packet_next_free,
      runtime->delayed_packet_identities,
      runtime->route_transport_proposals,
      runtime->eligibility_batch_claims,
      runtime->route_transport_scan_a, runtime->route_transport_scan_b,
      runtime->route_transport_free_slots, runtime->eligibility_batch_owners,
      runtime->node_participation_candidate_owners,
      runtime->route_transport_cursor,
      runtime->node_active_participation, runtime->node_next_participation,
      runtime->node_active_participation_locks, runtime->node_next_participation_locks,
      runtime->node_active_ancestry_incomplete,
      runtime->node_next_ancestry_incomplete,
      runtime->participation_staging, runtime->participation_staging_count,
      runtime->route_causal_pin_bits, runtime->device_metrics, runtime->device_stop_flag,
      runtime->device_resident_tick, runtime->device_epoch_limit, runtime->device_epoch_snapshot
  };
  for (void* p : dev_buffers) {
    if (p != nullptr) cudaFree(p);
  }
  if (runtime->owns_route_opportunity_incarnations &&
      runtime->route_opportunity_incarnations != nullptr) {
    cudaFree(runtime->route_opportunity_incarnations);
  }

  // 4. Release resource ecology charge
  if (runtime->brain != nullptr && runtime->brain->resource_ecology != nullptr) {
    AdultWorkingSetCharge charge = adult_working_set_charge(
        runtime->brain->node_count, runtime->brain->route_capacity,
        runtime->config.action_participant_capacity,
        runtime->owns_route_opportunity_incarnations);
    charge.unpooled_bytes +=
        sizeof(ResidentPublicMotorTrajectory) +
        sizeof(direct_network::DirectCausalWorldModel) +
        direct_network::direct_resident_language_runtime_storage_bytes() +
        direct_adult_action_control_runtime_storage_bytes();
      release_adult_runtime_working_set_kernel<<<1, 32>>>(
        runtime->brain->resource_ecology, charge.eligibility_units, charge.packet_units,
        charge.ticket_units, charge.unpooled_bytes);
    cudaDeviceSynchronize();
  }

  // 5. Destroy events and streams
  if (runtime->ingress_consumed_event != nullptr) cudaEventDestroy(runtime->ingress_consumed_event);
  if (runtime->consequence_consumed_event != nullptr) cudaEventDestroy(runtime->consequence_consumed_event);
  if (runtime->transport_ingress_event != nullptr) cudaEventDestroy(runtime->transport_ingress_event);
  if (runtime->transport_consequence_event != nullptr) cudaEventDestroy(runtime->transport_consequence_event);
  if (runtime->stream != nullptr) cudaStreamDestroy(runtime->stream);
  if (runtime->transport_stream != nullptr) cudaStreamDestroy(runtime->transport_stream);
  if (runtime->persistent_stream != nullptr) cudaStreamDestroy(runtime->persistent_stream);

  delete runtime;
}

bool inject_sensory_event(
    DirectAdultRuntime* runtime,
    const ActivityEvent& event) {
  return stage_sensory_events(runtime, &event, nullptr, 1u) == 1u;
}

std::uint32_t inject_sensory_events(
    DirectAdultRuntime* runtime,
    const ActivityEvent* events,
    std::uint32_t count) {
  if (events == nullptr) return 0u;
  return stage_sensory_events(runtime, events, nullptr, count);
}

bool inject_actual_sensory_contact(
    DirectAdultRuntime* runtime, const ActivityEvent& event,
    const ResidentContactEpochReceipt& receipt) {
  return stage_sensory_events(runtime, &event, &receipt, 1u) == 1u;
}

std::uint32_t flush_sensory_ingress(DirectAdultRuntime* runtime) {
  if (runtime == nullptr || runtime->host_ingress_staging == nullptr) return 0u;

  const std::uint32_t write_tail = runtime->host_ingress_write_tail;
  const std::uint32_t publish_tail = runtime->host_ingress_publish_tail;
  const std::uint32_t pending = write_tail - publish_tail;

  if (pending == 0u) return 0u;

  if (pending > kMaxIngressQueueSize) {
    ++runtime->host_ingress_protocol_faults;
    throw std::runtime_error(
        "flush_sensory_ingress: ingress host-ring invariant violated (pending > capacity)");
  }

  const std::uint32_t publish_slot = publish_tail % kMaxIngressQueueSize;
  const std::uint32_t first = std::min(pending, kMaxIngressQueueSize - publish_slot);

  check_cuda(
      cudaMemcpyAsync(
          runtime->ingress_queue + publish_slot,
          runtime->host_ingress_staging + publish_slot,
          sizeof(ActivityEvent) * first,
          cudaMemcpyHostToDevice,
          runtime->stream),
      "flush sensory ingress segment 0");
  check_cuda(cudaMemcpyAsync(
                 runtime->ingress_contact_credentials + publish_slot,
                 runtime->host_ingress_contact_staging + publish_slot,
                 sizeof(ResidentContactEpochReceipt) * first,
                 cudaMemcpyHostToDevice, runtime->stream),
             "flush contact credential segment 0");

  const std::uint32_t second = pending - first;
  if (second != 0u) {
    check_cuda(
        cudaMemcpyAsync(
            runtime->ingress_queue,
            runtime->host_ingress_staging,
            sizeof(ActivityEvent) * second,
            cudaMemcpyHostToDevice,
            runtime->stream),
        "flush sensory ingress segment 1");
    check_cuda(cudaMemcpyAsync(
                   runtime->ingress_contact_credentials,
                   runtime->host_ingress_contact_staging,
                   sizeof(ResidentContactEpochReceipt) * second,
                   cudaMemcpyHostToDevice, runtime->stream),
               "flush contact credential segment 1");
  }

  runtime->host_ingress_publish_tail = write_tail;

  if (runtime->ingress_control != nullptr && runtime->host_ingress_publish_slot_pinned != nullptr) {
    *runtime->host_ingress_publish_slot_pinned = write_tail;
    check_cuda(
        cudaMemcpyAsync(
            &runtime->ingress_control->published_tail,
            runtime->host_ingress_publish_slot_pinned,
            sizeof(std::uint32_t),
            cudaMemcpyHostToDevice,
            runtime->stream),
        "publish ingress_control published_tail");
  }

  return pending;
}

namespace {

bool settle_resolved_consequence(DirectAdultRuntime* runtime,
                                 std::uint64_t ticket_id,
                                 Word returned_word,
                                 CausalOrigin origin,
                                 std::uint32_t admission_tick,
                                 std::uint32_t transport_cursor) {
  resolve_consequence_ticket_kernel<<<1, 1, 0, runtime->stream>>>(
      runtime->ticket_table,
      runtime->ticket_count,
      runtime->action_occurrences,
      runtime->action_participation_links,
      runtime->config.action_participant_capacity,
      ticket_id,
      returned_word,
      origin,
      admission_tick,
      runtime->brain->development,
      transport_cursor,
      *runtime->brain,
      runtime->actual_frontier,
      runtime->causal_credit_predictions,
      runtime->brain->nodes,
      runtime->brain->node_count,
      runtime->brain->routes,
      runtime->brain->route_incarnations,
      runtime->brain->route_capacity,
      runtime->brain->retention_bank,
      runtime->brain->recipe_cells,
      runtime->brain->recipe_cell_count,
      runtime->brain->dense_blocks,
      runtime->brain->dense_block_count,
      runtime->config.learning_rate_q16,
      runtime->config.dense_shatter_threshold_q16,
      runtime->resolved_consequence_ctx,
      runtime->device_metrics,
      runtime->cue_salience_table);
  check_cuda(cudaGetLastError(), "inject_raw_reafferent_contact");

  const std::uint32_t block_dim = runtime->config.block_dim;
  const std::uint32_t grid_dim = (kMaxLiveEligibilityRecords + block_dim - 1) / block_dim;
  assimilate_consequence_and_credit_kernel<<<grid_dim, block_dim, 0, runtime->stream>>>(
      *runtime->brain,
      runtime->brain->nodes,
      runtime->brain->node_count,
      runtime->brain->routes,
      runtime->brain->route_incarnations,
      runtime->brain->route_capacity,
      runtime->brain->dense_blocks,
      runtime->brain->dense_block_count,
      runtime->eligibility_table,
      runtime->live_eligibility_count,
      runtime->resolved_consequence_ctx,
      runtime->action_participation_links,
      runtime->config.action_participant_capacity,
      runtime->config.learning_rate_q16,
      runtime->config.dense_shatter_threshold_q16,
      runtime->device_metrics,
      runtime->current_tick);
  check_cuda(cudaGetLastError(), "inject_raw_reafferent_contact");
  derive_affect_body_state_kernel<<<1, 1, 0, runtime->stream>>>(
      runtime->affect_body_state, runtime->ticket_table,
      runtime->brain->development);
  check_cuda(cudaGetLastError(), "derive affect body consequence");
  derive_wanting_liking_state_kernel<<<1, 1, 0, runtime->stream>>>(
      runtime->wanting_liking_state, runtime->action_occurrences,
      runtime->action_participation_links,
      kMaxAsynchronousTickets * runtime->config.action_participant_capacity,
      runtime->affect_body_state, runtime->brain->recipe_cells,
      runtime->brain->recipe_cell_count);
  check_cuda(cudaGetLastError(), "derive wanting liking profile");
  derive_resident_causal_world_model_kernel<<<1, 1, 0, runtime->stream>>>(
      runtime->causal_world_model, runtime->ticket_table, runtime->ticket_count,
      runtime->brain->development, runtime->brain->postbirth_constructor,
      runtime->action_occurrences, runtime->action_participation_links,
      runtime->config.action_participant_capacity);
  check_cuda(cudaGetLastError(), "derive resident causal world model");
  ResolvedConsequenceContext resolved{};
  check_cuda(cudaMemcpy(&resolved, runtime->resolved_consequence_ctx,
                        sizeof(resolved), cudaMemcpyDeviceToHost),
             "read resolved consequence result");
  return resolved.valid != 0u && resolved.history_refused == 0u;
}

static void refresh_consequence_reclaim_frontier(DirectAdultRuntime* runtime) {
  if (runtime == nullptr || runtime->consequence_control == nullptr || runtime->host_consequence_head_snapshot == nullptr) return;

  const cudaStream_t tstream = runtime->transport_stream ? runtime->transport_stream : runtime->stream;
  if (!runtime->is_persistent_running) {
    if (runtime->consequence_consumed_event != nullptr) {
      cudaStreamWaitEvent(tstream, runtime->consequence_consumed_event, 0);
    }
  }

  check_cuda(
      cudaMemcpyAsync(
          runtime->host_consequence_head_snapshot,
          &runtime->consequence_control->consumed_head,
          sizeof(std::uint32_t),
          cudaMemcpyDeviceToHost,
          tstream),
      "refresh_consequence_reclaim_frontier snapshot D2H");

  check_cuda(cudaStreamSynchronize(tstream), "sync transport_stream on consequence reclaim refresh");

  const std::uint32_t refreshed_head = *runtime->host_consequence_head_snapshot;
  const std::uint32_t used = runtime->host_consequence_write_tail - refreshed_head;
  if (used <= kMaxAsynchronousTickets) {
    runtime->host_consequence_observed_head = refreshed_head;
  }
}

}  // namespace

static bool enqueue_consequence_staging(DirectAdultRuntime* runtime,
                                        std::uint64_t ticket_id,
                                        Word returned_word,
                                        CausalOrigin origin) {
  if (!runtime ||
      (origin != CausalOrigin::world_return &&
       origin != CausalOrigin::motor_reafference))
    return false;
  const std::uint32_t transport_cursor = runtime->host_consequence_write_tail + 1u;
  const std::uint32_t admission_tick =
      runtime->is_persistent_running && runtime->device_resident_tick != nullptr
          ? *runtime->device_resident_tick
          : runtime->current_tick;
  if (runtime->host_consequence_staging != nullptr) {
    std::uint32_t used = runtime->host_consequence_write_tail - runtime->host_consequence_observed_head;
    if (used > kMaxAsynchronousTickets) {
      ++runtime->host_consequence_protocol_faults;
      throw std::runtime_error("enqueue_consequence_staging: ring invariant violated (used > capacity)");
    }
    if (used >= kMaxAsynchronousTickets) {
      refresh_consequence_reclaim_frontier(runtime);
      used = runtime->host_consequence_write_tail - runtime->host_consequence_observed_head;
      if (used > kMaxAsynchronousTickets) {
        ++runtime->host_consequence_protocol_faults;
        throw std::runtime_error("enqueue_consequence_staging: ring invariant violated after refresh");
      }
    }
    if (used < kMaxAsynchronousTickets) {
      const std::uint32_t slot = runtime->host_consequence_write_tail % kMaxAsynchronousTickets;
      runtime->host_consequence_staging[slot] = ConsequenceIngressEvent{
          ticket_id, returned_word, admission_tick, origin, 0u};
      ++runtime->host_consequence_write_tail;
    } else {
      ++runtime->host_consequence_overflow_drops;
      return false;
    }
  }
  if (!runtime->is_persistent_running) {
    if (!settle_resolved_consequence(runtime, ticket_id, returned_word, origin,
                                     admission_tick, transport_cursor)) {
      --runtime->host_consequence_write_tail;
      return false;
    }
    const std::uint32_t settled_tail = runtime->host_consequence_write_tail;
    runtime->host_consequence_observed_head = settled_tail;
    runtime->host_consequence_publish_tail = settled_tail;
    if (runtime->consequence_control != nullptr &&
        runtime->host_consequence_head_snapshot != nullptr &&
        runtime->host_consequence_publish_slot_pinned != nullptr) {
      *runtime->host_consequence_head_snapshot = settled_tail;
      *runtime->host_consequence_publish_slot_pinned = settled_tail;
      check_cuda(cudaMemcpyAsync(
                     &runtime->consequence_control->consumed_head,
                     runtime->host_consequence_head_snapshot,
                     sizeof(settled_tail), cudaMemcpyHostToDevice,
                     runtime->stream),
                 "commit stepped consequence consumed frontier");
      check_cuda(cudaMemcpyAsync(
                     &runtime->consequence_control->published_tail,
                     runtime->host_consequence_publish_slot_pinned,
                     sizeof(settled_tail), cudaMemcpyHostToDevice,
                     runtime->stream),
                 "commit stepped consequence published frontier");
    }
    if (runtime->consequence_consumed_event != nullptr) {
      check_cuda(cudaEventRecord(runtime->consequence_consumed_event,
                                 runtime->stream),
                 "record consequence_consumed_event");
    }
  }
  return true;
}

__global__ void decay_cue_salience_epoch_kernel(
    ResidentCueSalienceTable* table) {
  if (blockIdx.x != 0u || threadIdx.x != 0u) return;
  decay_resident_cue_salience(table);
}

bool inject_raw_reafferent_contact(DirectAdultRuntime* runtime, std::uint64_t ticket_id, Word returned_word) {
  return enqueue_consequence_staging(
      runtime, ticket_id, returned_word, CausalOrigin::motor_reafference);
}

bool inject_raw_world_return(DirectAdultRuntime* runtime,
                             std::uint64_t ticket_id, Word returned_word) {
  return enqueue_consequence_staging(
      runtime, ticket_id, returned_word, CausalOrigin::world_return);
}

std::uint32_t read_motor_events(DirectAdultRuntime* runtime, MotorEvent* out_buffer, std::uint32_t max_count) {
  if (!runtime || !out_buffer || max_count == 0) return 0;
  std::uint32_t head = 0, tail = 0;
  check_cuda(cudaMemcpy(&head, runtime->egress_head, sizeof(std::uint32_t), cudaMemcpyDeviceToHost), "read egress_head");
  check_cuda(cudaMemcpy(&tail, runtime->egress_tail, sizeof(std::uint32_t), cudaMemcpyDeviceToHost), "read egress_tail");
  const std::uint32_t pending = tail - head;
  const std::uint32_t count = std::min(pending, max_count);
  const std::uint32_t first_span =
      std::min(count, kMaxEgressQueueSize - head % kMaxEgressQueueSize);
  check_cuda(cudaMemcpy(out_buffer,
                        runtime->egress_queue + head % kMaxEgressQueueSize,
                        first_span * sizeof(MotorEvent),
                        cudaMemcpyDeviceToHost),
             "read MotorEvent span");
  if (count > first_span) {
    check_cuda(cudaMemcpy(out_buffer + first_span,
                          runtime->egress_queue,
                          (count - first_span) * sizeof(MotorEvent),
                          cudaMemcpyDeviceToHost),
               "read MotorEvent wrapped span");
  }
  head += count;
  check_cuda(cudaMemcpy(runtime->egress_head, &head, sizeof(std::uint32_t), cudaMemcpyHostToDevice), "update egress_head");
  return count;
}

std::uint32_t read_efference_copies(DirectAdultRuntime* runtime, DirectEfferenceCopy* out_buffer, std::uint32_t max_count) {
  if (!runtime || !out_buffer || max_count == 0 || !runtime->efference_ring) return 0;
  std::uint32_t head = 0, tail = 0;
  check_cuda(cudaMemcpy(&head, runtime->efference_head, sizeof(std::uint32_t), cudaMemcpyDeviceToHost), "read efference_head");
  check_cuda(cudaMemcpy(&tail, runtime->efference_tail, sizeof(std::uint32_t), cudaMemcpyDeviceToHost), "read efference_tail");
  const std::uint32_t pending = tail - head;
  const std::uint32_t count = std::min(pending, max_count);
  for (std::uint32_t i = 0; i < count; ++i) {
    const std::uint32_t slot = (head + i) % kMaxEfferenceRingSize;
    check_cuda(cudaMemcpy(out_buffer + i, runtime->efference_ring + slot, sizeof(DirectEfferenceCopy), cudaMemcpyDeviceToHost), "read DirectEfferenceCopy");
  }
  head += count;
  check_cuda(cudaMemcpy(runtime->efference_head, &head, sizeof(std::uint32_t), cudaMemcpyHostToDevice), "update efference_head");
  return count;
}

namespace {
void step_direct_adult_epochs_impl(DirectAdultRuntime* runtime,
                                   std::uint32_t epoch_count,
                                   bool advance_resident_structure) {
  if (!runtime || epoch_count == 0) return;
  if (runtime->execution_authority != AdultExecutionAuthority::host_stepped) {
    throw std::runtime_error("step_direct_adult_epochs: cannot step while persistent execution is active");
  }

  // Cross-stream publication ordering: step stream waits on any in-flight transport DMA
  if (runtime->transport_ingress_event != nullptr) {
    check_cuda(cudaStreamWaitEvent(runtime->stream, runtime->transport_ingress_event, 0), "wait transport_ingress_event");
  }
  if (runtime->transport_consequence_event != nullptr) {
    check_cuda(cudaStreamWaitEvent(runtime->stream, runtime->transport_consequence_event, 0), "wait transport_consequence_event");
  }

  const std::uint32_t block_dim = runtime->config.block_dim;
  const std::uint32_t node_count = runtime->brain->node_count;
  const std::uint32_t node_grid = (node_count + block_dim - 1) / block_dim;
  const std::uint32_t participation_grid =
      (node_count * kNodeParticipationAperture + block_dim - 1u) / block_dim;
  const std::uint32_t port_count = runtime->brain->boundary_port_count;
  const std::uint32_t port_grid = (port_count > 0) ? ((port_count + block_dim - 1) / block_dim) : 1;
  const std::uint32_t dense_count = runtime->brain->dense_block_count;
  for (std::uint32_t ep = 0; ep < epoch_count; ++ep) {
    runtime->current_tick++;
    if (runtime->device_resident_tick != nullptr) {
      *runtime->device_resident_tick = runtime->current_tick;
    }
    check_cuda(cudaMemsetAsync(runtime->participation_staging_count, 0,
                               sizeof(std::uint32_t), runtime->stream),
               "reset participation_staging_count");
    advance_and_commit_resident_mismatch_kernel<<<1, block_dim, 0, runtime->stream>>>(
        *runtime->brain, runtime->actual_frontier,
        &runtime->mismatch_omission->frontier, runtime->current_tick);
    decay_cue_salience_epoch_kernel<<<1, 1, 0, runtime->stream>>>(
        runtime->cue_salience_table);
    dispatch_ingress_eligibility_frontier(runtime, block_dim);
    publish_admitted_actual_contacts_kernel<<<1, 1, 0, runtime->stream>>>(
        *runtime->brain, runtime->node_active_participation, runtime->node_active_participation_locks,
        runtime->eligibility_table, runtime->live_eligibility_count, runtime->eligibility_claim_directory,
        runtime->eligibility_record_generations,
        runtime->route_transport_free_slots, runtime->route_transport_scan_a,
        runtime->participation_staging, runtime->participation_staging_count, runtime->participation_staging_capacity,
        runtime->node_incoming_excitation, runtime->device_metrics, runtime->current_tick, runtime->actual_frontier);
    flush_sensory_ingress(runtime);
    const std::uint32_t head_snap = runtime->host_ingress_dispatched_tail;
    const std::uint32_t tail_snap = runtime->host_ingress_publish_tail;
    const std::uint32_t total_events = tail_snap - head_snap;

    if (total_events > kMaxIngressQueueSize) {
      ++runtime->host_ingress_protocol_faults;
      throw std::runtime_error(
          "step_direct_adult_epochs: ingress host-ring invariant violated (total_events > capacity)");
    }

    if (total_events != 0u) {
      ingest_sensory_events_kernel<<<1, block_dim, 0, runtime->stream>>>(
          runtime->brain->nodes,
          runtime->brain->boundary_ports,
          port_count,
          node_count,
          runtime->ingress_queue,
          head_snap,
          tail_snap,
          runtime->node_incoming_excitation,
          runtime->node_slow_context_q16,
          runtime->node_active_participation,
          runtime->node_active_participation_locks,
          runtime->claim_incarnation_counter,
          runtime->ticket_table_locks,
          runtime->ticket_table,
          runtime->action_occurrences,
          runtime->participation_staging,
          runtime->participation_staging_count,
          runtime->participation_staging_capacity,
          runtime->eligibility_table,
          runtime->live_eligibility_count,
          runtime->eligibility_claim_directory,
          runtime->eligibility_record_generations,
          runtime->route_transport_free_slots,
          runtime->route_transport_scan_a,
          runtime->device_metrics,
          runtime->current_tick,
          runtime->config.eligibility_horizon_ticks,
          runtime->brain->development,
           runtime->ingress_contact_credentials,
           runtime->actual_frontier,
           &runtime->mismatch_omission->frontier,
           *runtime->brain,
          nullptr);

      commit_consumed_head_kernel<<<1, 1, 0, runtime->stream>>>(
          runtime->ingress_consumed_head,
          runtime->ingress_control,
          tail_snap);
      condense_resident_actual_frontier_kernel<<<1, 1, 0, runtime->stream>>>(
          *runtime->brain, runtime->actual_frontier);
      refresh_resident_predictions_and_expectations_kernel<<<1, block_dim, 0, runtime->stream>>>(
          *runtime->brain, runtime->actual_frontier, runtime->current_tick,
          runtime->causal_credit_predictions, &runtime->mismatch_omission->frontier);
      commit_resident_mismatch_credit_kernel<<<1, block_dim, 0, runtime->stream>>>(
          *runtime->brain, runtime->actual_frontier,
          &runtime->mismatch_omission->frontier,
          runtime->current_tick);

      if (runtime->ingress_consumed_event != nullptr) {
        check_cuda(
            cudaEventRecord(runtime->ingress_consumed_event, runtime->stream),
            "record ingress_consumed_event");
      }
      runtime->host_ingress_dispatched_tail = tail_snap;
    }

    // Resident language plasticity reads only the same Adult's exact committed
    // history. No host transcript/context or semantic router participates.
    launch_resident_language_assimilation_phase(runtime);
    launch_resident_language_candidate_admission_phase(runtime);
    launch_resident_action_commit_phase(runtime);

    // 2. Γ-authored resident development is part of the adult clock. It runs
    // after contact ingress and before recurrent propagation, matching the
    // persistent executor's device-owned phase ordering.
    if (advance_resident_structure && runtime->brain->development != nullptr) {
      const std::uint32_t* development_pins =
          rebuild_route_causal_pins(runtime, block_dim, false);
      direct_network::launch_resident_development_epoch(
          runtime->brain, &runtime->resident_development, runtime->stream, block_dim,
          development_pins);
      ++runtime->resident_development_epochs;
    }

    // 3. Freeze frontier_t into a carry-preserving write-only next bank.
    prepare_participation_frontier_kernel<<<participation_grid, block_dim, 0,
                                             runtime->stream>>>(
        runtime->node_active_participation, runtime->node_next_participation,
        runtime->node_next_participation_locks,
        runtime->node_next_ancestry_incomplete, node_count,
        runtime->current_tick);

    // Due packets arrive before this tick's recurrent activation and motor
    // egress. Delivery is device-resident and shared with the persistent path.
    dispatch_deterministic_route_transport(runtime, block_dim);

    // Resolve checkpointed contacts only from the current sparse contribution
    // set, then advance ordinary route descendants before public motor binding.
    advance_actual_frontier_from_propagation_kernel<<<1, 1, 0, runtime->stream>>>(
        *runtime->brain, runtime->participation_staging,
        runtime->participation_staging_count, runtime->participation_staging_capacity,
        runtime->current_tick, runtime->actual_frontier);

    // 4. Dense WMMA Tensor Core Tile Integration
    if (dense_count > 0) {
      execute_dense_tensor_wmma_kernel<<<dense_count, 32, 0, runtime->stream>>>(
          runtime->brain->nodes,
          runtime->brain->dense_blocks,
          dense_count,
          runtime->brain->dense_weight_fp16_bits,
          runtime->node_incoming_excitation,
          runtime->node_next_ancestry_incomplete,
          runtime->participation_staging,
          runtime->participation_staging_count,
          runtime->participation_staging_capacity,
          runtime->current_tick,
          runtime->device_metrics);
    }

    // 5. Node Activation Integration
    integrate_node_activation_kernel<<<node_grid, block_dim, 0, runtime->stream>>>(
        runtime->brain->nodes,
        node_count,
        runtime->node_incoming_excitation,
        runtime->node_slow_context_q16,
        runtime->attractor_state,
        runtime->current_tick,
        runtime->config.refractory_period_ticks,
        runtime->config.attractor_coupling_gain_q16,
        runtime->config.persistent_bias_ceiling_q16,
        runtime->config.slow_context_ceiling_q16);

    // 6. Attractor Basin Dynamics
    step_attractor_basins_kernel<<<1, kMaxBasins, 0, runtime->stream>>>(
        runtime->attractor_state,
        kMaxBasins);

    // 7. Motor Egress & Current Contribution Binding
    if (port_count > 0) {
      emit_motor_events_kernel<<<port_grid, block_dim, 0, runtime->stream>>>(
          *runtime->brain,
          runtime->brain->nodes,
          runtime->brain->routes,
          runtime->brain->route_incarnations,
          runtime->brain->route_capacity,
          runtime->brain->boundary_ports,
          runtime->actual_frontier,
          runtime->activation_plane,
          runtime->brain->postbirth_constructor != nullptr ? 1u : 0u,
          port_count,
          runtime->egress_queue,
          runtime->egress_head,
          runtime->egress_tail,
          runtime->ticket_table,
          runtime->ticket_count,
          runtime->ticket_table_locks,
          runtime->action_occurrences,
          runtime->action_participation_links,
          runtime->config.action_participant_capacity,
          runtime->resident_motor_trajectory,
          runtime->config.resident_motor_trajectory_capacity,
          runtime->affect_body_state,
          runtime->config.eligibility_horizon_ticks,
          runtime->participation_staging,
          runtime->participation_staging_count,
          runtime->participation_staging_capacity,
          runtime->node_next_ancestry_incomplete,
          runtime->eligibility_table,
          runtime->eligibility_record_generations,
          runtime->brain->development,
          runtime->device_metrics,
          runtime->current_tick,
          runtime->efference_ring,
          runtime->efference_head,
          runtime->efference_tail,
          runtime->node_incoming_excitation,
          runtime->config.route_efference_copies);
    }

    launch_resident_language_motor_finalize_phase(runtime);

    derive_wanting_liking_state_kernel<<<1, 1, 0, runtime->stream>>>(
        runtime->wanting_liking_state, runtime->action_occurrences,
        runtime->action_participation_links,
        kMaxAsynchronousTickets * runtime->config.action_participant_capacity,
        runtime->affect_body_state, runtime->brain->recipe_cells,
        runtime->brain->recipe_cell_count);

    // 8. Autopoietic Resource Maintenance (every 4 ticks)
    if (advance_resident_structure && (runtime->current_tick % 4) == 0) {
      // Rebuild the causal pin first: the sweep must read the pin state as of
      // THIS tick, not the previous sweep's. Both dispatches are on the
      // maintenance cadence and on the runtime's own stream, so they cost
      // nothing on the other three ticks and add no synchronisation.
      const std::uint32_t* pin_bits = rebuild_route_causal_pins(runtime, block_dim, true);
      direct_network::launch_exact_history_maintenance(
          runtime->brain, &runtime->resident_development,
          runtime->route_opportunity_incarnations,
          runtime->config.maintenance_cost_per_route_q16, pin_bits,
          runtime->device_metrics, runtime->current_tick, runtime->stream, block_dim);
      decay_node_slow_context_kernel<<<node_grid, block_dim, 0, runtime->stream>>>(
          runtime->node_slow_context_q16, node_count,
          runtime->config.eligibility_decay_q16);
    }

    if (dense_count > 0 && (runtime->current_tick % 8) == 0) {
      resident_self_compilation_crystallize_kernel<<<(dense_count + block_dim - 1) / block_dim, block_dim, 0, runtime->stream>>>(
          runtime->brain->nodes,
          runtime->brain->dense_blocks,
          dense_count,
          runtime->device_metrics);
    }

    commit_participation_frontier_kernel<<<participation_grid, block_dim, 0,
                                            runtime->stream>>>(
        runtime->node_active_participation, runtime->node_next_participation,
        runtime->node_active_participation_locks,
        runtime->node_next_participation_locks,
        runtime->node_active_ancestry_incomplete,
        runtime->node_next_ancestry_incomplete, node_count);
    resolve_pending_actual_contacts_kernel<<<1, 1, 0, runtime->stream>>>(
        *runtime->brain, runtime->node_active_participation,
        runtime->current_tick, runtime->actual_frontier, runtime->device_metrics);
    const std::uint32_t staging_grid =
        (runtime->participation_staging_capacity + block_dim - 1u) / block_dim;
    clear_consumed_participation_staging_kernel<<<staging_grid, block_dim, 0,
                                                   runtime->stream>>>(
        runtime->participation_staging, runtime->participation_staging_count,
        runtime->participation_staging_capacity);
    check_cuda(cudaMemsetAsync(runtime->participation_staging_count, 0,
                               sizeof(std::uint32_t), runtime->stream),
               "clear consumed participation staging count");
    check_cuda(cudaGetLastError(), "step_direct_adult_epochs epoch");
    check_cuda(cudaStreamSynchronize(runtime->stream),
               "sync host-stepped adult epoch");
    if (!page_completed_direct_exact_history(runtime)) {
      throw std::runtime_error(
          "step_direct_adult_epochs: exact-history page could not be archived");
    }
  }
  check_cuda(cudaGetLastError(), "step_direct_adult_epochs");
}
}  // namespace

void step_direct_adult_epochs(DirectAdultRuntime* runtime,
                              std::uint32_t epoch_count) {
  step_direct_adult_epochs_impl(runtime, epoch_count, true);
}

void step_direct_adult_fixed_morphology_epochs(
    DirectAdultRuntime* runtime, std::uint32_t epoch_count) {
  step_direct_adult_epochs_impl(runtime, epoch_count, false);
}

direct_network::ResidentDevelopmentCounters observe_direct_adult_resident_development(
    DirectAdultRuntime* runtime) {
  if (runtime == nullptr) return {};
  if (runtime->is_persistent_running) {
    throw std::runtime_error(
        "observe_direct_adult_resident_development: stop persistent adult first");
  }
  return direct_network::observe_resident_development(runtime->resident_development,
                                                      runtime->stream);
}

#include "hardware_native/direct_adult_delayed_sparse_schedule.cuh"
#include "hardware_native/direct_adult_delayed_sparse_delivery.cuh"
#include "hardware_native/direct_adult_checkpoint.inl"

}  // namespace substrate::direct_adult_core
