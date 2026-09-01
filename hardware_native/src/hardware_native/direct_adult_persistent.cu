#include "direct_adult_persistent.cuh"
#include "hardware_native/direct_adult_runtime_frontiers.cuh"
#include "direct_adult_device_ops.cuh"
#include "hardware_native/direct_adult_persistent_motor_phase.cuh"
#include "hardware_native/direct_adult_persistent_phases.cuh"
#include "hardware_native/direct_adult_language_expression_opportunity.cuh"
#include "hardware_native/direct_adult_language_action_candidate_device.cuh"
#include "hardware_native/direct_adult_action_commit_device.cuh"
#include "hardware_native/direct_adult_resident_language_runtime_abi.cuh"
#include "hardware_native/direct_adult_dense_execution.cuh"
#include "hardware_native/direct_adult_actual_frontier_condensation.cuh"
#include "hardware_native/direct_bounded_sm_workspace.cuh"
#include "direct_adult_activation.cuh"
#include "hardware_native/direct_adult_affect_body.cuh"
#include "hardware_native/direct_adult_wanting_liking_dissociation.cuh"
#include "hardware_native/direct_adult_temporal_discounting_competition.cuh"
#include "hardware_native/direct_adult_uncertainty_plateau.cuh"
#include "hardware_native/direct_adult_incentive_salience.cuh"
#include "hardware_native/direct_adult_bounded_fanout.cuh"
#include "hardware_native/direct_adult_resource_maintenance.cuh"
#include "hardware_native/direct_adult_wanting_liking_dissociation.cuh"
#include "hardware_native/direct_adult_checkpoint.cuh"
#include "hardware_native/direct_exact_history_cold_archive.cuh"
#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cerrno>
#include <stdexcept>
#include <algorithm>
#include <cstring>
#include <cstdio>
#include <string>
#include <sys/stat.h>

namespace cg = cooperative_groups;

namespace substrate::direct_adult_core {
#include "hardware_native/direct_adult_delayed_sparse_schedule.cuh"
#include "hardware_native/direct_adult_delayed_sparse_delivery.cuh"

namespace {

inline void check_cuda(cudaError_t err, const char* msg) {
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string(msg) + ": " + cudaGetErrorString(err));
  }
}

bool ensure_exact_history_archive_directory(DirectAdultRuntime* runtime) {
  if (runtime->exact_history_archive_directory[0] != '\0') return true;
  constexpr const char* base = "/tmp/0x1-exact-history";
  if (::mkdir(base, 0700) != 0 && errno != EEXIST) return false;
  const auto& root = runtime->brain->birth_root;
  const int written = std::snprintf(
      runtime->exact_history_archive_directory,
      sizeof(runtime->exact_history_archive_directory),
      "%s/%08x%08x%08x%08x%08x%08x%08x%08x", base,
      static_cast<unsigned>(root.word[0]), static_cast<unsigned>(root.word[1]),
      static_cast<unsigned>(root.word[2]), static_cast<unsigned>(root.word[3]),
      static_cast<unsigned>(root.word[4]), static_cast<unsigned>(root.word[5]),
      static_cast<unsigned>(root.word[6]), static_cast<unsigned>(root.word[7]));
  if (written <= 0 || static_cast<std::size_t>(written) >=
                          sizeof(runtime->exact_history_archive_directory))
    return false;
  return ::mkdir(runtime->exact_history_archive_directory, 0700) == 0 ||
         errno == EEXIST;
}

bool page_completed_exact_history_impl(DirectAdultRuntime* runtime) {
  if (runtime->brain == nullptr) return false;
  if (runtime->brain->development == nullptr) return true;
  std::uint32_t sealed = 0u;
  check_cuda(cudaMemcpy(
                 &sealed,
                 &runtime->brain->development->exact_history.sealed,
                 sizeof(sealed), cudaMemcpyDeviceToHost),
             "read exact-history page state");
  if (sealed == 0u) return true;
  if (runtime->config.exact_history_archive_capacity_bytes == 0u ||
      !ensure_exact_history_archive_directory(runtime))
    return false;
  return direct_network::archive_direct_exact_history_page(
             runtime, runtime->exact_history_archive_directory,
             runtime->config.exact_history_archive_capacity_bytes)
             .status == direct_network::DirectExactHistoryArchiveStatus::archived;
}

}  // namespace

bool page_completed_direct_exact_history(DirectAdultRuntime* runtime) {
  return runtime != nullptr && page_completed_exact_history_impl(runtime);
}

bool query_cooperative_adult_launch_dims(int& out_grid_blocks, int& out_block_dim) {
  int device = 0;
  if (cudaGetDevice(&device) != cudaSuccess) return false;

  int cooperative = 0;
  if (cudaDeviceGetAttribute(&cooperative, cudaDevAttrCooperativeLaunch, device) != cudaSuccess || cooperative == 0) {
    return false;
  }

  int sm_count = 0;
  if (cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device) != cudaSuccess || sm_count <= 0) {
    return false;
  }

  out_block_dim = 256;
  int blocks_per_sm = 0;
  if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &blocks_per_sm,
          persistent_direct_adult_kernel,
          out_block_dim,
          0) != cudaSuccess || blocks_per_sm <= 0) {
    return false;
  }

  out_grid_blocks = blocks_per_sm * sm_count;
  return true;
}

__device__ const std::uint32_t* rebuild_persistent_route_causal_pins(
    cg::grid_group grid, const EligibilityRecord* eligibility_table,
    const std::uint32_t* live_eligibility_count,
    const std::uint64_t* route_incarnations, std::uint32_t route_capacity,
    std::uint32_t current_tick, std::uint32_t* pin_bits,
    std::uint32_t global_tid, std::uint32_t total_threads,
    std::uint32_t honor_causal_pin, AdultCoreMetrics* metrics) {
  if (honor_causal_pin == 0u || pin_bits == nullptr || route_capacity == 0u)
    return nullptr;
  const std::uint32_t pin_words = (route_capacity + 31u) / 32u;
  for (std::uint32_t i = global_tid; i < pin_words; i += total_threads) pin_bits[i] = 0u;
  grid.sync();
  std::uint32_t count = live_eligibility_count != nullptr ? *live_eligibility_count : 0u;
  if (count > kMaxLiveEligibilityRecords) count = kMaxLiveEligibilityRecords;
  for (std::uint32_t i = global_tid; i < count; i += total_threads) {
    const EligibilityRecord rec = eligibility_table[i];
    if (rec.live == 0u || rec.expiry_tick < current_tick ||
        rec.route_index >= route_capacity || route_incarnations == nullptr ||
        route_incarnations[rec.route_index] != rec.route_incarnation)
      continue;
    const std::uint32_t bit = 1u << (rec.route_index & 31u);
    if ((atomicOr(&pin_bits[rec.route_index >> 5], bit) & bit) == 0u &&
        metrics != nullptr)
      atomicAdd(reinterpret_cast<unsigned long long*>(&metrics->causally_pinned_routes),
                1ULL);
  }
  grid.sync();
  return pin_bits;
}

__device__ void persistent_exclusive_scan(
    cg::grid_group grid, const std::uint32_t* input, std::uint32_t* output,
    std::uint32_t* block_sums, std::uint32_t* block_offsets,
    std::uint32_t* local_scan, std::uint32_t node_count,
    std::uint32_t global_tid) {
  const std::uint32_t local = global_tid < node_count ? input[global_tid] : 0u;
  local_scan[threadIdx.x] = local;
  __syncthreads();
  for (std::uint32_t offset = 1u; offset < blockDim.x; offset <<= 1u) {
    const std::uint32_t add =
        threadIdx.x >= offset ? local_scan[threadIdx.x - offset] : 0u;
    __syncthreads();
    local_scan[threadIdx.x] += add;
    __syncthreads();
  }
  if (threadIdx.x == 0u) block_sums[blockIdx.x] = local_scan[blockDim.x - 1u];
  grid.sync();
  if (grid.thread_rank() == 0u) {
    std::uint32_t prefix = 0u;
    for (std::uint32_t block = 0u; block < gridDim.x; ++block) {
      block_offsets[block] = prefix;
      prefix += block_sums[block];
    }
  }
  grid.sync();
  if (global_tid < node_count)
    output[global_tid] =
        block_offsets[blockIdx.x] + local_scan[threadIdx.x] - local;
  grid.sync();
}

__device__ void publish_observer_snapshot(
    AdultCoreObserverSnapshot* snapshot,
    const AdultCoreMetrics* metrics,
    std::uint32_t resident_tick) {
  if (snapshot == nullptr || metrics == nullptr) return;
  const auto* request =
      reinterpret_cast<volatile const unsigned long long*>(&snapshot->request_id);
  const auto* completion =
      reinterpret_cast<volatile const unsigned long long*>(&snapshot->completion_id);
  const unsigned long long request_id = *request;
  if (request_id == 0ull || request_id == *completion) return;
  snapshot->metrics = *metrics;
  snapshot->resident_tick = resident_tick;
  __threadfence_system();
  atomicExch(reinterpret_cast<unsigned long long*>(&snapshot->completion_id),
             request_id);
}


__global__ void persistent_direct_adult_kernel(
    DirectNode* nodes,
    DirectRoute* routes,
    std::uint32_t node_count,
    std::uint32_t route_capacity,
    const DirectBoundaryPort* boundary_ports,
    std::uint32_t boundary_port_count,
    ActivityEvent* ingress_queue,
    ResidentContactEpochReceipt* ingress_contact_credentials,
    ResidentActualFrontier* actual_frontier,
    ResidentActivationSoaPlane* activation_plane,
    ResidentMultiHorizonPredictionFrontier* causal_credit_predictions,
    ResidentMismatchOmissionRuntime* mismatch_omission,
    IngressRingControl* ingress_control,
    ConsequenceIngressEvent* consequence_queue,
    IngressRingControl* consequence_control,
    MotorEvent* egress_queue,
    std::uint32_t* egress_head,
    std::uint32_t* egress_tail,
    EligibilityRecord* eligibility_table,
    std::uint32_t* live_eligibility_count,
    std::uint64_t* eligibility_claim_directory,
    std::uint32_t* eligibility_claim_locks,
    std::uint32_t* eligibility_record_generations,
    AsynchronousTicket* ticket_table,
    std::uint32_t* ticket_count,
    std::uint32_t* ticket_table_locks,
    std::uint32_t* claim_incarnation_counter,
    DirectActionOccurrence* action_occurrences,
    DirectActionParticipationLink* action_participation_links,
    std::uint32_t action_participant_capacity,
    ResidentPublicMotorTrajectory* resident_motor_trajectory,
    direct_network::DirectResidentLanguageRuntimeBlock* resident_language,
    DirectAdultActionControlRuntimeBlock* action_control_runtime,
    direct_network::DirectAffectBodyState* affect_body_state,
    direct_network::ResidentWantingLikingProfileV1* wanting_liking_state,
    ResolvedConsequenceContext* resolved_consequence_ctx,
    AttractorBasinState* attractor_state,
    ResidentCueSalienceTable* cue_salience_table,
    std::int32_t* node_incoming_excitation,
    std::int32_t* node_slow_context_q16,
    DelayedSparsePacket* delayed_packets,
    std::uint32_t* delayed_packet_live_count,
    std::uint32_t* delayed_packet_free_head,
    std::uint32_t* delayed_packet_next_free,
    DelayedPacketIdentity* delayed_packet_identities,
    RouteTransportPhaseView route_transport,
    std::uint64_t* route_opportunity_incarnations,
    std::uint32_t* route_causal_pin_bits,
    NodeCausalParticipation* node_active_participation,
    NodeCausalParticipation* node_next_participation,
    std::uint32_t* node_active_participation_locks,
    std::uint32_t* node_next_participation_locks,
    std::uint32_t* node_active_ancestry_incomplete,
    std::uint32_t* node_next_ancestry_incomplete,
    DirectParticipationDescriptor* participation_staging,
    std::uint32_t* participation_staging_count,
    std::uint32_t participation_staging_capacity,
    AdultCoreMetrics* device_metrics,
    AdultExecutionConfig config,
    volatile std::uint32_t* stop_flag,
    volatile std::uint32_t* resident_tick,
    volatile std::uint32_t* epoch_limit,
    ResidentAdultEpochSnapshot* device_epoch_snapshot,
    ResidentDevelopmentWorkspace resident_development,
    DirectBrain brain,
    DirectEfferenceCopy* efference_ring,
    std::uint32_t* efference_head,
    std::uint32_t* efference_tail,
    AdultCoreObserverSnapshot* observer_snapshot) {

  cg::grid_group grid = cg::this_grid();
  const std::uint32_t global_tid = grid.thread_rank();
  const std::uint32_t total_threads = grid.size();
  ResidentMismatchOmissionFrontier* mismatch_omission_frontier =
      mismatch_omission != nullptr ? &mismatch_omission->frontier : nullptr;

  for (;;) {
    // 1. Grid rank 0 writes global epoch snapshot
    if (grid.thread_rank() == 0) {
      ResidentAdultEpochSnapshot snap{};
      const std::uint32_t prior_tick = resident_tick != nullptr ? *resident_tick : 0u;
      const bool limit_reached = epoch_limit != nullptr && *epoch_limit != 0u &&
                                 prior_tick >= *epoch_limit;
      const bool wrap_reached = prior_tick == UINT32_MAX;
      snap.stop = ((stop_flag != nullptr) ? *stop_flag : 0u) |
                  (limit_reached || wrap_reached ? 1u : 0u);
      snap.tick = snap.stop == 0u && resident_tick != nullptr
                      ? atomicAdd(const_cast<std::uint32_t*>(
                                      const_cast<volatile std::uint32_t*>(resident_tick)),
                                  1u) + 1u
                      : prior_tick;

      snap.ingress_head = ingress_control ? ingress_control->consumed_head : 0u;
      snap.ingress_tail = ingress_control ? ingress_control->published_tail : 0u;
      snap.consequence_head = consequence_control ? consequence_control->consumed_head : 0u;
      snap.consequence_tail = consequence_control ? consequence_control->published_tail : 0u;

      const std::uint32_t ing_count = snap.ingress_tail - snap.ingress_head;
      if (ing_count > kMaxIngressQueueSize) {
        snap.ingress_fault = 1u;
        if (ingress_control != nullptr) ingress_control->fault = 1u;
      } else {
        snap.ingress_fault = 0u;
      }

      const std::uint32_t c_count = snap.consequence_tail - snap.consequence_head;
      if (c_count > kMaxAsynchronousTickets) {
        snap.consequence_fault = 1u;
        if (consequence_control != nullptr) consequence_control->fault = 1u;
      } else {
        snap.consequence_fault = 0u;
      }

      if (device_epoch_snapshot != nullptr) {
        *device_epoch_snapshot = snap;
      }
    }

    grid.sync();

    const ResidentAdultEpochSnapshot epoch_snap =
        (device_epoch_snapshot != nullptr) ? *device_epoch_snapshot : ResidentAdultEpochSnapshot{};

    // Uniform stop or fault branch
    if (epoch_snap.stop != 0u || epoch_snap.ingress_fault != 0u || epoch_snap.consequence_fault != 0u) {
      break;
    }

    const std::uint32_t current_tick = epoch_snap.tick;

    if (grid.thread_rank() == 0u && participation_staging_count != nullptr) {
      *participation_staging_count = 0u;
    }
    derive_ingress_eligibility_free_frontier(
        grid, eligibility_table, eligibility_record_generations,
        live_eligibility_count, current_tick, route_transport.scan_a,
        route_transport.scan_b, route_transport.free_slots, device_metrics);

    if (grid.thread_rank() == 0u) {
      advance_resident_mismatch_omissions(current_tick, actual_frontier,
                                          mismatch_omission_frontier);
      commit_resident_mismatch_credit_receipts(brain,
                                               mismatch_omission_frontier,
                                               current_tick,
                                               actual_frontier);
      expire_resident_actual_frontier(
          actual_frontier, current_tick,
          device_metrics != nullptr ? &device_metrics->actual_frontier_causal : nullptr);
      decay_resident_cue_salience(cue_salience_table);
      const bool published = publish_admitted_resident_actual_contacts(
          brain, actual_frontier, node_active_participation,
          node_active_participation_locks, eligibility_table,
          live_eligibility_count, eligibility_claim_directory,
          eligibility_record_generations, route_transport.free_slots,
          route_transport.scan_a, participation_staging,
          participation_staging_count, participation_staging_capacity,
          node_incoming_excitation, device_metrics, current_tick);
      if (published &&
          !has_unpublished_resident_actual_contact(actual_frontier))
        condense_resident_actual_frontier_dispatch(brain, actual_frontier);
      if (activation_plane != nullptr)
        ++activation_plane->control.frontier_mutations;
    }
    grid.sync();

    direct_network::DirectExactHistoryHotPage* exact_history =
        brain.development != nullptr ? &brain.development->exact_history : nullptr;
    persistent_ingress_phase_device(
        grid, epoch_snap, current_tick, global_tid, total_threads, exact_history,
        mismatch_omission_frontier, nodes, node_count, boundary_ports,
        boundary_port_count, ingress_queue, ingress_contact_credentials,
        actual_frontier, activation_plane, causal_credit_predictions,
        ingress_control, eligibility_table, live_eligibility_count,
        eligibility_claim_directory, eligibility_record_generations,
        ticket_table, ticket_table_locks, claim_incarnation_counter,
        action_occurrences, node_incoming_excitation, node_slow_context_q16,
        route_transport, node_active_participation,
        node_active_participation_locks, participation_staging,
        participation_staging_count, participation_staging_capacity,
        device_metrics, config, brain);

    persistent_settlement_phase_device(
        grid, epoch_snap, current_tick, global_tid, total_threads, exact_history,
        nodes, routes, node_count, route_capacity, actual_frontier,
        causal_credit_predictions, consequence_queue, consequence_control,
        eligibility_table, live_eligibility_count, ticket_table,
        action_occurrences, action_participation_links,
        action_participant_capacity, affect_body_state,
        resolved_consequence_ctx, cue_salience_table, device_metrics, config,
        brain);

    if (grid.thread_rank() == 0u) {
      direct_network::direct_resident_language_assimilate_owned(
          resident_language, exact_history);
      if (actual_frontier != nullptr) {
        ResidentRecipeOccurrence occurrences[kResidentActualFrontierCapacity]{};
        for (std::uint32_t i = 0u; i < kResidentActualFrontierCapacity; ++i)
          occurrences[i] = actual_frontier->entries[i].occurrence;
        (void)plan_and_admit_resident_language_action_candidate_owned(
            resident_language, action_control_runtime, exact_history, occurrences,
            kResidentActualFrontierCapacity,
            node_active_participation, node_count, current_tick);
        (void)commit_resident_action_control_owned(
            action_control_runtime, occurrences, kResidentActualFrontierCapacity,
            mismatch_omission_frontier, current_tick);
      }
    }
    grid.sync();

    persistent_development_phase_device(
        grid, epoch_snap, current_tick, global_tid, total_threads, node_count, route_capacity, eligibility_table, live_eligibility_count, route_causal_pin_bits, config, resident_development, brain);

    const std::uint32_t participation_slot_count =
        node_count * kNodeParticipationAperture;
    persistent_execution_phase_device(
        grid, current_tick, global_tid, total_threads, participation_slot_count, nodes, routes, node_count, route_capacity, actual_frontier, activation_plane, eligibility_table, live_eligibility_count, eligibility_claim_directory, eligibility_record_generations, attractor_state, node_incoming_excitation, node_slow_context_q16, delayed_packets, delayed_packet_live_count, delayed_packet_free_head, delayed_packet_next_free, delayed_packet_identities, route_transport, route_opportunity_incarnations, node_active_participation, node_next_participation, node_next_participation_locks, node_active_ancestry_incomplete, node_next_ancestry_incomplete, participation_staging, participation_staging_count, participation_staging_capacity, device_metrics, config, brain);

    persistent_motor_phase_device(
        grid, current_tick, nodes, routes, route_capacity, boundary_ports,
        boundary_port_count, actual_frontier, activation_plane, exact_history,
        egress_queue, egress_head, egress_tail, eligibility_table,
        eligibility_record_generations, ticket_table, ticket_count,
        ticket_table_locks, action_occurrences, action_participation_links,
        action_participant_capacity, resident_motor_trajectory,
        affect_body_state, node_incoming_excitation,
        node_next_ancestry_incomplete, participation_staging,
        participation_staging_count, participation_staging_capacity,
        device_metrics, config, brain, efference_ring, efference_head,
        efference_tail);

    grid.sync();
    if (grid.thread_rank() == 0u)
      finalize_resident_language_motor_owned(
          resident_language, brain, actual_frontier, egress_queue, egress_head,
          egress_tail, ticket_table, action_occurrences,
          action_participation_links, brain.development, efference_ring,
          efference_head, efference_tail, config.route_efference_copies,
          current_tick);
    grid.sync();

    if (grid.thread_rank() == 0u && wanting_liking_state != nullptr &&
        action_occurrences != nullptr && action_participation_links != nullptr)
      *wanting_liking_state = direct_network::observe_resident_wanting_liking(
          action_occurrences, kMaxAsynchronousTickets,
          action_participation_links,
          kMaxAsynchronousTickets * action_participant_capacity,
          affect_body_state, brain.recipe_cells, brain.recipe_cell_count);
    grid.sync();

    // 8. Autopoietic Maintenance Cadence
    if ((current_tick % 4u) == 0u) {
      const std::uint32_t* maintenance_pins = rebuild_persistent_route_causal_pins(
          grid, eligibility_table, live_eligibility_count, brain.route_incarnations,
          route_capacity, current_tick, route_causal_pin_bits, global_tid,
          total_threads, config.honor_causal_pin, device_metrics);
      if (resident_development.costs != nullptr &&
          resident_development.history_prefix != nullptr &&
          resident_development.block_sums != nullptr &&
          resident_development.block_offsets != nullptr &&
          resident_development.topology_plan != nullptr &&
          brain.development != nullptr) {
        if (global_tid == 0u)
          *resident_development.topology_plan =
              direct_network::TopologyCommitPlan{};
        grid.sync();
        if (global_tid < node_count)
          direct_network::resident_maintenance_plan_node(
              brain, route_opportunity_incarnations, maintenance_pins,
              config.maintenance_cost_per_route_q16,
              resident_development.costs, resident_development.topology_plan,
              device_metrics, global_tid);
        grid.sync();
        __shared__ std::uint32_t maintenance_scan[kDirectPersistentScanWorkspaceWords];
        persistent_exclusive_scan(
            grid, resident_development.costs,
            resident_development.history_prefix,
            resident_development.block_sums,
            resident_development.block_offsets, maintenance_scan, node_count,
            global_tid);
        if (global_tid == 0u)
          direct_network::resident_maintenance_begin_transaction(
              brain, resident_development.topology_plan, current_tick);
        grid.sync();
        if (global_tid < node_count)
          direct_network::resident_maintenance_commit_node(
              brain, route_opportunity_incarnations, maintenance_pins,
              resident_development.history_prefix,
              resident_development.topology_plan, device_metrics, global_tid);
        grid.sync();
        if (global_tid == 0u)
          direct_network::resident_development_finish_topology_transaction(
              brain, resident_development.topology_plan);
        grid.sync();
        if (global_tid < node_count)
          direct_network::resident_maintenance_plan_weight_node(
              brain, route_opportunity_incarnations, maintenance_pins,
              config.maintenance_cost_per_route_q16,
              resident_development.topology_plan, resident_development.costs,
              global_tid);
        grid.sync();
        persistent_exclusive_scan(
            grid, resident_development.costs,
            resident_development.history_prefix,
            resident_development.block_sums,
            resident_development.block_offsets, maintenance_scan, node_count,
            global_tid);
        if (global_tid == 0u)
          direct_network::resident_maintenance_begin_weight_transaction(
              brain, resident_development.costs,
              resident_development.history_prefix, node_count,
              resident_development.topology_plan, current_tick);
        grid.sync();
        if (global_tid < node_count)
          direct_network::resident_maintenance_decay_node(
              brain, route_opportunity_incarnations, maintenance_pins,
              config.maintenance_cost_per_route_q16,
              resident_development.history_prefix,
              resident_development.topology_plan, global_tid);
        grid.sync();
        if (global_tid == 0u)
          direct_network::resident_development_finish_topology_transaction(
              brain, resident_development.topology_plan);
      }
      grid.sync();
      for (std::uint32_t node = global_tid; node < node_count;
           node += total_threads)
        decay_resident_transient_trace_q16(
            node_slow_context_q16 != nullptr ? node_slow_context_q16 + node
                                             : nullptr,
            config.eligibility_decay_q16);
    }

    grid.sync();

    // End-of-epoch device commit: exactly one recurrent hop becomes visible.
    for (std::uint32_t index = global_tid; index < participation_slot_count;
         index += total_threads) {
      publish_participation_slot(
          node_active_participation + index,
          read_participation_slot(node_next_participation + index));
      publish_participation_slot(node_next_participation + index,
                                 NodeCausalParticipation{});
    }
    for (std::uint32_t node = global_tid; node < node_count; node += total_threads) {
      node_active_participation_locks[node] = 0u;
      node_next_participation_locks[node] = 0u;
      if (node_active_ancestry_incomplete != nullptr &&
          node_next_ancestry_incomplete != nullptr) {
        node_active_ancestry_incomplete[node] =
            node_next_ancestry_incomplete[node];
        node_next_ancestry_incomplete[node] = 0u;
      }
    }
    grid.sync();
    if (global_tid == 0u && actual_frontier != nullptr &&
        admit_resident_actual_frontier_pending_participation(
            brain, node_active_participation, current_tick, actual_frontier,
            device_metrics != nullptr ? &device_metrics->actual_frontier_causal : nullptr) &&
        activation_plane != nullptr)
      ++activation_plane->control.frontier_mutations;
    grid.sync();
    // Match the stepped phase boundary: consumed staging descriptors are
    // scratch, not checkpoint authority.  Clear exactly the published prefix
    // after its final consumer so replay cannot retain scheduler-order bytes.
    const std::uint32_t staged_count = participation_staging_count != nullptr
        ? (*participation_staging_count < participation_staging_capacity
               ? *participation_staging_count
               : participation_staging_capacity)
        : 0u;
    for (std::uint32_t i = global_tid; i < staged_count; i += total_threads)
      participation_staging[i] = DirectParticipationDescriptor{};
    grid.sync();
    if (global_tid == 0u && participation_staging_count != nullptr)
      *participation_staging_count = 0u;
    grid.sync();
    if (global_tid == 0u && resident_tick != nullptr)
      publish_observer_snapshot(observer_snapshot, device_metrics, *resident_tick);
    grid.sync();
  }
}

bool publish_ingress_transport(DirectAdultRuntime* runtime) {
  if (runtime == nullptr || runtime->host_ingress_staging == nullptr) return false;

  const std::uint32_t head_snap = runtime->host_ingress_publish_tail;
  const std::uint32_t tail_snap = runtime->host_ingress_write_tail;
  const std::uint32_t total_events = tail_snap - head_snap;

  if (total_events == 0u) return true;
  if (total_events > kMaxIngressQueueSize) return false;

  const std::uint32_t head_slot = head_snap % kMaxIngressQueueSize;
  const std::uint32_t first = std::min(total_events, kMaxIngressQueueSize - head_slot);

  const cudaStream_t stream = runtime->transport_stream ? runtime->transport_stream : runtime->stream;

  check_cuda(
      cudaMemcpyAsync(
          runtime->ingress_queue + head_slot,
          runtime->host_ingress_staging + head_slot,
          sizeof(ActivityEvent) * first,
          cudaMemcpyHostToDevice,
          stream),
      "publish_ingress_transport segment 0");
  check_cuda(cudaMemcpyAsync(
                 runtime->ingress_contact_credentials + head_slot,
                 runtime->host_ingress_contact_staging + head_slot,
                 sizeof(ResidentContactEpochReceipt) * first,
                 cudaMemcpyHostToDevice, stream),
             "publish_ingress_transport credential segment 0");

  const std::uint32_t second = total_events - first;
  if (second != 0u) {
    check_cuda(
        cudaMemcpyAsync(
            runtime->ingress_queue,
            runtime->host_ingress_staging,
            sizeof(ActivityEvent) * second,
            cudaMemcpyHostToDevice,
            stream),
        "publish_ingress_transport segment 1");
    check_cuda(cudaMemcpyAsync(
                   runtime->ingress_contact_credentials,
                   runtime->host_ingress_contact_staging,
                   sizeof(ResidentContactEpochReceipt) * second,
                   cudaMemcpyHostToDevice, stream),
               "publish_ingress_transport credential segment 1");
  }

  if (runtime->ingress_control != nullptr && runtime->host_ingress_publish_slot_pinned != nullptr) {
    *runtime->host_ingress_publish_slot_pinned = tail_snap;
    check_cuda(
        cudaMemcpyAsync(
            &runtime->ingress_control->published_tail,
            runtime->host_ingress_publish_slot_pinned,
            sizeof(std::uint32_t),
            cudaMemcpyHostToDevice,
            stream),
        "publish_ingress_transport published_tail");
  }

  if (runtime->transport_ingress_event != nullptr) {
    check_cuda(cudaEventRecord(runtime->transport_ingress_event, stream), "record transport_ingress_event");
  }

  runtime->host_ingress_publish_tail = tail_snap;
  return true;
}

IngressRingControl canonicalize_checkpoint_consequence(DirectAdultCheckpoint& checkpoint,
                                         const DirectAdultRuntime& runtime) {
  // Keep these ordinals aligned with checkpoint_buffer_views() in
  // direct_adult_checkpoint.inl. Assay 12 carries a real consequence event so
  // queue/control index drift is observable rather than silently checkpointed.
  constexpr std::size_t kConsequenceQueueCheckpointBuffer = 5u;
  constexpr std::size_t kConsequenceControlCheckpointBuffer = 6u;
  static_assert(kConsequenceControlCheckpointBuffer < kDirectAdultCheckpointBufferCount);
  IngressRingControl control{};
  std::memcpy(&control,
              checkpoint.device_buffers[kConsequenceControlCheckpointBuffer].data(),
              sizeof(control));
  if (runtime.host_consequence_observed_head == runtime.host_consequence_write_tail &&
      runtime.host_consequence_publish_tail != runtime.host_consequence_write_tail)
    control.consumed_head = control.published_tail = runtime.host_consequence_observed_head;
  if (control.capacity != kMaxAsynchronousTickets ||
      control.published_tail - control.consumed_head > control.capacity)
    return control;
  const auto raw = checkpoint.device_buffers[kConsequenceQueueCheckpointBuffer];
  std::memset(checkpoint.device_buffers[kConsequenceQueueCheckpointBuffer].data(), 0,
              checkpoint.device_buffers[kConsequenceQueueCheckpointBuffer].size());
  for (std::uint32_t cursor = control.consumed_head; cursor != control.published_tail; ++cursor) {
    const std::size_t slot = cursor % kMaxAsynchronousTickets;
    std::memcpy(checkpoint.device_buffers[kConsequenceQueueCheckpointBuffer].data() +
                    slot * sizeof(ConsequenceIngressEvent),
                raw.data() + slot * sizeof(ConsequenceIngressEvent),
                sizeof(ConsequenceIngressEvent));
  }
  std::memcpy(checkpoint.device_buffers[kConsequenceControlCheckpointBuffer].data(),
              &control, sizeof(control));
  return control;
}

bool publish_consequence_transport(DirectAdultRuntime* runtime) {
  if (runtime == nullptr || runtime->host_consequence_staging == nullptr) return false;

  const std::uint32_t head_snap = runtime->host_consequence_publish_tail;
  const std::uint32_t tail_snap = runtime->host_consequence_write_tail;
  const std::uint32_t total_events = tail_snap - head_snap;

  if (total_events == 0u) return true;
  if (total_events > kMaxAsynchronousTickets) return false;

  const std::uint32_t head_slot = head_snap % kMaxAsynchronousTickets;
  const std::uint32_t first = std::min(total_events, kMaxAsynchronousTickets - head_slot);

  const cudaStream_t stream = runtime->transport_stream ? runtime->transport_stream : runtime->stream;

  check_cuda(
      cudaMemcpyAsync(
          runtime->consequence_queue + head_slot,
          runtime->host_consequence_staging + head_slot,
          sizeof(ConsequenceIngressEvent) * first,
          cudaMemcpyHostToDevice,
          stream),
      "publish_consequence_transport segment 0");

  const std::uint32_t second = total_events - first;
  if (second != 0u) {
    check_cuda(
        cudaMemcpyAsync(
            runtime->consequence_queue,
            runtime->host_consequence_staging,
            sizeof(ConsequenceIngressEvent) * second,
            cudaMemcpyHostToDevice,
            stream),
        "publish_consequence_transport segment 1");
  }

  if (runtime->consequence_control != nullptr && runtime->host_consequence_publish_slot_pinned != nullptr) {
    *runtime->host_consequence_publish_slot_pinned = tail_snap;
    check_cuda(
        cudaMemcpyAsync(
            &runtime->consequence_control->published_tail,
            runtime->host_consequence_publish_slot_pinned,
            sizeof(std::uint32_t),
            cudaMemcpyHostToDevice,
            stream),
        "publish_consequence_transport published_tail");
  }

  if (runtime->transport_consequence_event != nullptr) {
    check_cuda(cudaEventRecord(runtime->transport_consequence_event, stream), "record transport_consequence_event");
  }

  runtime->host_consequence_publish_tail = tail_snap;
  return true;
}

namespace {

bool launch_persistent_direct_adult(
    DirectAdultRuntime* runtime,
    std::uint32_t epoch_limit) {
  if (runtime == nullptr || runtime->brain == nullptr) return false;
  if (runtime->execution_authority != AdultExecutionAuthority::host_stepped) return false;

  int grid_blocks = 0;
  int block_dim = 256;
  if (!query_cooperative_adult_launch_dims(grid_blocks, block_dim)) {
    return false;
  }
  if (runtime->resident_development.block_capacity <
      static_cast<std::uint32_t>(grid_blocks)) {
    return false;
  }

  runtime->execution_authority = AdultExecutionAuthority::starting;

  // Drain host step stream and transport stream before starting persistent kernel
  if (runtime->stream != nullptr) {
    check_cuda(cudaStreamSynchronize(runtime->stream), "sync stream before persistent start");
  }
  if (runtime->transport_stream != nullptr) {
    check_cuda(cudaStreamSynchronize(runtime->transport_stream), "sync transport_stream before persistent start");
  }

  // Reconcile clock from host to device
  if (runtime->device_resident_tick != nullptr) {
    *runtime->device_resident_tick = runtime->current_tick;
  }

  const DirectBrain brain = *runtime->brain;
  RouteTransportPhaseView route_transport{
      runtime->route_transport_proposals,
      runtime->eligibility_batch_claims,
      runtime->eligibility_batch_capacity,
      runtime->route_transport_scan_a,
      runtime->route_transport_scan_b,
      runtime->route_transport_scan_capacity,
      runtime->route_transport_free_slots,
      runtime->eligibility_batch_owners,
      runtime->node_participation_candidate_owners,
      runtime->route_transport_cursor};

  const std::uint32_t zero = 0u;
  check_cuda(
      cudaMemcpy(runtime->device_stop_flag, &zero, sizeof(zero), cudaMemcpyHostToDevice),
      "init device_stop_flag");
  check_cuda(
      cudaMemcpy(runtime->device_epoch_limit, &epoch_limit, sizeof(epoch_limit), cudaMemcpyHostToDevice),
      "init device_epoch_limit");

  void* kernel_args[] = {
      &runtime->brain->nodes,
      &runtime->brain->routes,
      &runtime->brain->node_count,
      &runtime->brain->route_capacity,
      &runtime->brain->boundary_ports,
      &runtime->brain->boundary_port_count,
      &runtime->ingress_queue,
      &runtime->ingress_contact_credentials,
      &runtime->actual_frontier,
      &runtime->activation_plane,
      &runtime->causal_credit_predictions,
      &runtime->mismatch_omission,
      &runtime->ingress_control,
      &runtime->consequence_queue,
      &runtime->consequence_control,
      &runtime->egress_queue,
      &runtime->egress_head,
      &runtime->egress_tail,
      &runtime->eligibility_table,
      &runtime->live_eligibility_count,
      &runtime->eligibility_claim_directory,
      &runtime->eligibility_claim_locks,
      &runtime->eligibility_record_generations,
      &runtime->ticket_table,
      &runtime->ticket_count,
      &runtime->ticket_table_locks,
      &runtime->claim_incarnation_counter,
      &runtime->action_occurrences,
      &runtime->action_participation_links,
      &runtime->config.action_participant_capacity,
      &runtime->resident_motor_trajectory,
      &runtime->resident_language,
      &runtime->action_control_runtime,
      &runtime->affect_body_state,
      &runtime->wanting_liking_state,
      &runtime->resolved_consequence_ctx,
      &runtime->attractor_state,
      &runtime->cue_salience_table,
      &runtime->node_incoming_excitation,
      &runtime->node_slow_context_q16,
      &runtime->delayed_packets,
      &runtime->delayed_packet_live_count,
      &runtime->delayed_packet_free_head,
      &runtime->delayed_packet_next_free,
      &runtime->delayed_packet_identities,
      &route_transport,
      &runtime->route_opportunity_incarnations,
      &runtime->route_causal_pin_bits,
      &runtime->node_active_participation,
      &runtime->node_next_participation,
      &runtime->node_active_participation_locks,
      &runtime->node_next_participation_locks,
      &runtime->node_active_ancestry_incomplete,
      &runtime->node_next_ancestry_incomplete,
      &runtime->participation_staging,
      &runtime->participation_staging_count,
      &runtime->participation_staging_capacity,
      &runtime->device_metrics,
      &runtime->config,
      &runtime->device_stop_flag,
      &runtime->device_resident_tick,
      &runtime->device_epoch_limit,
      &runtime->device_epoch_snapshot,
      &runtime->resident_development,
      const_cast<DirectBrain*>(&brain),
      &runtime->efference_ring,
      &runtime->efference_head,
      &runtime->efference_tail,
      &runtime->device_observer_snapshot};

  check_cuda(
      cudaLaunchCooperativeKernel(
          reinterpret_cast<void*>(persistent_direct_adult_kernel),
          dim3(grid_blocks),
          dim3(block_dim),
          kernel_args,
          0,
          runtime->persistent_stream),
      "cudaLaunchCooperativeKernel persistent_direct_adult_kernel");

  ++runtime->persistent_bootstrap_launches;
  runtime->execution_authority = AdultExecutionAuthority::persistent;
  runtime->is_persistent_running = true;
  return true;
}

bool reconcile_persistent_boundary(DirectAdultRuntime* runtime) {
  if (runtime->transport_stream != nullptr)
    check_cuda(cudaStreamSynchronize(runtime->transport_stream),
               "sync transport stream at persistent boundary");
  const std::uint32_t final_tick = *runtime->device_resident_tick;
  runtime->resident_development_epochs += final_tick - runtime->current_tick;
  runtime->current_tick = final_tick;

  IngressRingControl ingress{};
  IngressRingControl consequence{};
  check_cuda(cudaMemcpy(&ingress, runtime->ingress_control, sizeof(ingress),
                        cudaMemcpyDeviceToHost),
             "read persistent ingress frontier");
  check_cuda(cudaMemcpy(&consequence, runtime->consequence_control,
                        sizeof(consequence), cudaMemcpyDeviceToHost),
             "read persistent consequence frontier");
  const bool valid_ingress =
      ingress.capacity == kMaxIngressQueueSize && ingress.fault == 0u &&
      ingress.published_tail - ingress.consumed_head <= ingress.capacity &&
      runtime->host_ingress_publish_tail == ingress.published_tail &&
      runtime->host_ingress_write_tail - ingress.consumed_head <= ingress.capacity;
  const bool valid_consequence =
      consequence.capacity == kMaxAsynchronousTickets && consequence.fault == 0u &&
      consequence.published_tail - consequence.consumed_head <= consequence.capacity &&
      runtime->host_consequence_publish_tail == consequence.published_tail &&
      runtime->host_consequence_write_tail - consequence.consumed_head <=
          consequence.capacity;
  if (valid_ingress && valid_consequence) {
    runtime->host_ingress_observed_head = ingress.consumed_head;
    runtime->host_ingress_dispatched_tail = ingress.consumed_head;
    *runtime->host_ingress_head_snapshot = ingress.consumed_head;
    runtime->host_consequence_observed_head = consequence.consumed_head;
    *runtime->host_consequence_head_snapshot = consequence.consumed_head;
  }

  const std::uint32_t zero = 0u;
  check_cuda(cudaMemcpy(runtime->device_stop_flag, &zero, sizeof(zero),
                        cudaMemcpyHostToDevice),
             "reset device_stop_flag at persistent boundary");
  check_cuda(cudaMemcpy(runtime->device_epoch_limit, &zero, sizeof(zero),
                        cudaMemcpyHostToDevice),
             "reset device_epoch_limit at persistent boundary");
  return valid_ingress && valid_consequence;
}

}  // namespace

bool start_persistent_direct_adult(DirectAdultRuntime* runtime) {
  return launch_persistent_direct_adult(runtime, 0u);
}

bool configure_direct_exact_history_archive(
    DirectAdultRuntime* runtime, const char* directory,
    std::uint64_t capacity_bytes) {
  if (runtime == nullptr || directory == nullptr || directory[0] == '\0' ||
      capacity_bytes == 0u || runtime->current_tick != 0u ||
      runtime->execution_authority != AdultExecutionAuthority::host_stepped ||
      runtime->is_persistent_running)
    return false;
  const std::size_t length = std::strlen(directory);
  if (length >= sizeof(runtime->exact_history_archive_directory)) return false;
  std::memcpy(runtime->exact_history_archive_directory, directory, length + 1u);
  runtime->config.exact_history_archive_capacity_bytes = capacity_bytes;
  return true;
}

bool run_persistent_direct_adult_epochs(
    DirectAdultRuntime* runtime,
    std::uint32_t epoch_count) {
  if (runtime == nullptr || epoch_count == 0u ||
      epoch_count > UINT32_MAX - runtime->current_tick) {
    return runtime != nullptr && epoch_count == 0u &&
           runtime->execution_authority == AdultExecutionAuthority::host_stepped;
  }

  const std::uint32_t expected_tick = runtime->current_tick + epoch_count;
  if (!launch_persistent_direct_adult(runtime, expected_tick)) return false;
  check_cuda(cudaStreamSynchronize(runtime->persistent_stream),
             "sync bounded persistent span");
  const bool boundary_valid = reconcile_persistent_boundary(runtime);
  runtime->is_persistent_running = false;
  runtime->execution_authority = boundary_valid
                                     ? AdultExecutionAuthority::host_stepped
                                     : AdultExecutionAuthority::stopping;
  return boundary_valid && runtime->current_tick == expected_tick &&
         page_completed_direct_exact_history(runtime);
}

void stop_persistent_direct_adult(DirectAdultRuntime* runtime) {
  if (runtime == nullptr) return;
  if (runtime->execution_authority != AdultExecutionAuthority::persistent) return;

  runtime->execution_authority = AdultExecutionAuthority::stopping;

  const std::uint32_t one = 1u;
  check_cuda(
      cudaMemcpyAsync(
          runtime->device_stop_flag,
          &one,
          sizeof(one),
          cudaMemcpyHostToDevice,
          runtime->transport_stream ? runtime->transport_stream : runtime->stream),
      "write device_stop_flag");

  if (runtime->persistent_stream != nullptr) {
    check_cuda(
        cudaStreamSynchronize(runtime->persistent_stream),
        "sync persistent_stream on stop");
  }

  const bool boundary_valid = reconcile_persistent_boundary(runtime);
  runtime->is_persistent_running = false;
  runtime->execution_authority = boundary_valid
                                     ? AdultExecutionAuthority::host_stepped
                                     : AdultExecutionAuthority::stopping;
}

std::uint32_t get_persistent_adult_resident_tick(const DirectAdultRuntime* runtime) {
  if (runtime == nullptr || runtime->device_resident_tick == nullptr) return 0u;
  return *runtime->device_resident_tick;
}

AdultExecutionAuthority get_direct_adult_execution_authority(const DirectAdultRuntime* runtime) {
  return runtime ? runtime->execution_authority : AdultExecutionAuthority::host_stepped;
}

}  // namespace substrate::direct_adult_core
