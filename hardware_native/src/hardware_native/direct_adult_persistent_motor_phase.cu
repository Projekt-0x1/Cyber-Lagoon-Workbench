#include "hardware_native/direct_adult_persistent_motor_phase.cuh"
#include "hardware_native/direct_adult_runtime_frontiers.cuh"
#include "hardware_native/direct_adult_lived_expression.cuh"
#include "hardware_native/direct_adult_device_ops.cuh"
#include "hardware_native/direct_adult_affect_body.cuh"
#include "hardware_native/direct_adult_temporal_discounting_competition.cuh"
#include "hardware_native/direct_adult_uncertainty_plateau.cuh"
#include "hardware_native/direct_adult_wanting_liking_dissociation.cuh"

#define HARDWARE_NATIVE_DEFINE_RESIDENT_MOTOR_TRAJECTORY_OPS
#include "hardware_native/direct_adult_resident_motor_trajectory.cuh"
#undef HARDWARE_NATIVE_DEFINE_RESIDENT_MOTOR_TRAJECTORY_OPS

namespace substrate::direct_adult_core {

#include "hardware_native/direct_adult_motor_affect_helpers.cuh"

__device__ __noinline__ void persistent_motor_phase_device(
    cooperative_groups::grid_group grid,
    std::uint32_t current_tick,
    DirectNode* nodes,
    DirectRoute* routes,
    std::uint32_t route_capacity,
    const DirectBoundaryPort* boundary_ports,
    std::uint32_t boundary_port_count,
    ResidentActualFrontier* actual_frontier,
    ResidentActivationSoaPlane* activation_plane,
    direct_network::DirectExactHistoryHotPage* exact_history,
    MotorEvent* egress_queue,
    std::uint32_t* egress_head,
    std::uint32_t* egress_tail,
    EligibilityRecord* eligibility_table,
    std::uint32_t* eligibility_record_generations,
    AsynchronousTicket* ticket_table,
    std::uint32_t* ticket_count,
    std::uint32_t* ticket_table_locks,
    DirectActionOccurrence* action_occurrences,
    DirectActionParticipationLink* action_participation_links,
    std::uint32_t action_participant_capacity,
    ResidentPublicMotorTrajectory* resident_motor_trajectory,
    direct_network::DirectAffectBodyState* affect_body_state,
    std::int32_t* node_incoming_excitation,
    std::uint32_t* node_next_ancestry_incomplete,
    DirectParticipationDescriptor* participation_staging,
    std::uint32_t* participation_staging_count,
    std::uint32_t participation_staging_capacity,
    AdultCoreMetrics* device_metrics,
    AdultExecutionConfig config,
    DirectBrain brain,
    DirectEfferenceCopy* efference_ring,
    std::uint32_t* efference_head,
    std::uint32_t* efference_tail) {
    // 7. Motor Egress & Current Contribution Binding
    if (boundary_ports != nullptr && nodes != nullptr && egress_queue != nullptr &&
        egress_head != nullptr && egress_tail != nullptr && ticket_table != nullptr &&
        ticket_count != nullptr) {
      std::uint32_t motor_tail_base = 0u;
      std::uint32_t motor_emitted = 0u;
      std::uint32_t efference_emitted = 0u;
      bool motor_history_admitted = true;
      bool continued_trajectory = false;
      MotorRootChannelMemo root_memo;
      if (grid.thread_rank() == 0u) {
        motor_tail_base = *egress_tail;
        motor_history_admitted = exact_history == nullptr ||
            direct_network::begin_exact_history_phase(
                exact_history, direct_network::DirectExactHistoryKind::motor_output,
                boundary_port_count, current_tick);
        if (motor_history_admitted && resident_motor_trajectory != nullptr &&
            resident_motor_trajectory->state ==
                kResidentPublicMotorTrajectoryLive &&
            motor_tail_base - *egress_head < kMaxEgressQueueSize) {
          std::uint32_t shadow_target = direct_network::kInvalidIndex;
          bool efference_ready = true;
          if (config.route_efference_copies != 0u) {
            efference_ready =
                efference_ring != nullptr && efference_head != nullptr &&
                efference_tail != nullptr &&
                node_incoming_excitation != nullptr &&
                *efference_tail - *efference_head < kMaxEfferenceRingSize;
            if (efference_ready) {
              shadow_target = direct_network::efference_shadow_target(
                  nodes, routes,
                  nodes[resident_motor_trajectory->motor_node]);
              efference_ready =
                  shadow_target != direct_network::kInvalidIndex;
            }
          }
          MotorEvent continuation{};
          if (efference_ready && continue_resident_motor_trajectory(
                  resident_motor_trajectory, brain, boundary_ports,
                  boundary_port_count, actual_frontier, current_tick,
                  config.eligibility_horizon_ticks, ticket_table, ticket_count,
                  ticket_table_locks, action_occurrences,
                  action_participation_links, action_participant_capacity,
                  device_metrics, activation_plane, eligibility_table,
                  eligibility_record_generations, &continuation)) {
            direct_network::stage_motor_history_record(
                exact_history != nullptr
                    ? &exact_history->records[exact_history->phase_base]
                    : nullptr,
                continuation.ticket_id, continuation.upstream_ticket_id,
                current_tick, continuation.node, continuation.channel,
                continuation.word, continuation.context,
                continuation.reserved);
            egress_queue[motor_tail_base % kMaxEgressQueueSize] =
                continuation;
            if (config.route_efference_copies != 0u) {
              DirectEfferenceCopy copy{};
              copy.ticket_id = continuation.ticket_id;
              copy.acting_node = continuation.node;
              copy.shadow_node = shadow_target;
              copy.channel = continuation.channel;
              copy.word = continuation.word;
              copy.timestamp = current_tick;
              efference_ring[*efference_tail % kMaxEfferenceRingSize] = copy;
              atomicAdd(&node_incoming_excitation[shadow_target],
                        kQ16One / 8);
              ++efference_emitted;
            }
            motor_emitted = 1u;
            continued_trajectory = true;
          }
          continued_trajectory = continued_trajectory ||
              resident_motor_trajectory->state ==
                  kResidentPublicMotorTrajectoryLive;
        }
      }
      for (std::uint32_t p =
               grid.thread_rank() == 0u && !continued_trajectory
                   ? 0u
                   : boundary_port_count;
           p < boundary_port_count; ++p) {
        if (!motor_history_admitted) break;
        const DirectBoundaryPort port = boundary_ports[p];
        if (!(port.role_mask & static_cast<std::uint32_t>(direct_network::BoundaryRole::motor)))
          continue;

        const DirectNode node = nodes[port.node];
        if (node.activation_q16 > 0) {
          if (!affect_motor_candidate_wins_local_competition(
                  nodes, boundary_ports, boundary_port_count, p, root_memo,
                  affect_body_state, routes, brain.route_incarnations,
                  route_capacity,
                  exact_history != nullptr ? exact_history->records : nullptr,
                  exact_history != nullptr ? exact_history->committed_slots : 0u,
                  participation_staging, participation_staging_count,
                  participation_staging_capacity, current_tick)) continue;
          const auto wanting_liking_gate =
              direct_network::resident_wanting_liking_motor_gate(
                  participation_staging,
                  participation_staging_count != nullptr
                      ? *participation_staging_count
                      : 0u,
                  port.node, affect_body_state);
          std::uint32_t affect_root = kInvalidIndex;
          (void)resolve_motor_root_channel(
              root_memo,
              exact_history != nullptr ? exact_history->records : nullptr,
              exact_history != nullptr ? exact_history->committed_slots : 0u,
              participation_staging, participation_staging_count,
              participation_staging_capacity, port.node, current_tick,
              &affect_root);
          direct_network::DirectAffectCandidate affect_candidate{};
          affect_candidate.root_channel = affect_root;
          affect_candidate.base_valuation_q16 =
              clamp_q16(node.activation_q16, 0, kQ16One);
          const direct_network::DirectAffectValuationReceipt affect_receipt =
              direct_network::affect_modulate_candidate(affect_body_state,
                                                        affect_candidate);
          const auto uncertainty = resident_motor_outcome_uncertainty(
              ticket_table, kMaxAsynchronousTickets, port.node);
          const std::int32_t affect_base_threshold_q16 =
              affect_preservation_base_threshold_q16(
                  wanting_liking_gate.activation_threshold_q16,
                  affect_receipt.bias_q16, wanting_liking_gate.live_pursuit);
          const std::int32_t pursuit_threshold_q16 =
              resident_uncertainty_pursuit_threshold_q16(
                  affect_base_threshold_q16, uncertainty.signal_q16);
          if (affect_receipt.modulated_valuation_q16 <= pursuit_threshold_q16)
            continue;
          if (motor_tail_base + motor_emitted - *egress_head >=
              kMaxEgressQueueSize) continue;
          std::uint32_t shadow_target = direct_network::kInvalidIndex;
          if (config.route_efference_copies != 0u) {
            if (efference_ring == nullptr || efference_head == nullptr ||
                efference_tail == nullptr ||
                node_incoming_excitation == nullptr ||
                *efference_tail + efference_emitted - *efference_head >=
                    kMaxEfferenceRingSize)
              continue;
            shadow_target =
                direct_network::efference_shadow_target(nodes, routes, node);
            if (shadow_target == direct_network::kInvalidIndex) continue;
          }
          std::uint32_t ticket_idx = kInvalidIndex;
          for (std::uint32_t attempt = 0u; attempt < kMaxAsynchronousTickets; ++attempt) {
            const std::uint32_t candidate =
                atomicAdd(ticket_count, 1u) % kMaxAsynchronousTickets;
            if (lock_ticket_slot_for_motor(
                    ticket_table_locks, ticket_table, action_occurrences,
                    action_participation_links, action_participant_capacity,
                    brain.development, candidate, current_tick,
                    config.eligibility_horizon_ticks, device_metrics)) {
              ticket_idx = candidate;
              break;
            }
          }
          if (ticket_idx == kInvalidIndex) continue;
          const std::uint64_t tid =
              (static_cast<std::uint64_t>(current_tick) << 32) |
              (port.channel << 16) | ticket_idx;

          AsynchronousTicket ticket{};
          ticket.ticket_id = tid;
          ticket.upstream_ticket_id = kInvalidTicket;
          ticket.motor_node = port.node;
          ticket.motor_channel = port.channel;

          ticket.context_signature = (port.node * 1103515245U) ^ current_tick;
          ticket.emission_tick = current_tick;
          ticket.settled = 0u;
          ticket.settled_reward_q16 = 0;
          ticket.mismatch_bits = 0u;
          DirectActionBindingResult ancestry = bind_action_occurrence(
              participation_staging, participation_staging_count,
              participation_staging_capacity, actual_frontier,
              brain.postbirth_constructor != nullptr ? 1u : 0u,
              port.node, current_tick, tid,
              ticket.context_signature, port.channel,
              current_tick + config.eligibility_horizon_ticks, ticket_idx,
              action_occurrences, action_participation_links,
              action_participant_capacity, device_metrics, activation_plane,
              brain, eligibility_table, eligibility_record_generations);
          if (ancestry.admitted == 0u) {
            unlock_ticket_slot(ticket_table_locks, ticket_idx);
            continue;
          }
          if (node_next_ancestry_incomplete != nullptr &&
              node_next_ancestry_incomplete[port.node] != 0u &&
              ancestry.ancestry_incomplete == 0u) {
            ancestry.ancestry_incomplete = 1u;
            if (device_metrics != nullptr)
              atomicAdd(reinterpret_cast<unsigned long long*>(
                            &device_metrics->action_ancestry_incomplete),
                        1ULL);
          }
          ticket.upstream_ticket_id = ancestry.diagnostic_upstream_ticket;
          const bool trajectory_required =
              brain.postbirth_constructor != nullptr &&
              resident_motor_trajectory_required(
                  brain, actual_frontier, action_occurrences[ticket_idx],
                  action_participation_links, current_tick);
          ResidentPublicMotorTrajectoryArtifact trajectory_artifact{};
          if (trajectory_required) {
            if (!begin_resident_motor_trajectory(
                    resident_motor_trajectory, brain, actual_frontier,
                    action_occurrences[ticket_idx],
                    action_participation_links,
                    port, p,
                    resident_motor_trajectory != nullptr
                        ? config.resident_motor_trajectory_capacity
                        : 0u,
                    tid, ancestry.ancestry_incomplete, device_metrics) ||
                !resident_motor_trajectory_word(
                    *resident_motor_trajectory, brain, actual_frontier,
                    current_tick,
                    &ticket.motor_word)) {
              atomicExch(&action_occurrences[ticket_idx].state,
                         kActionOccurrenceExpired);
              unlock_ticket_slot(ticket_table_locks, ticket_idx);
              continue;
            }
            trajectory_artifact =
                resident_motor_trajectory_artifact(
                    *resident_motor_trajectory, brain, actual_frontier);
          } else {
            ticket.motor_word = resident_bound_action_motor_word(
                brain, node, routes, route_capacity,
                bounded_route_scan_count(node), actual_frontier,
                action_occurrences, action_participation_links, ancestry,
                ticket_idx, action_participant_capacity, tid, port.node,
                port.channel, ticket.context_signature, current_tick);
          }
          ticket_table[ticket_idx] = ticket;
          unlock_ticket_slot(ticket_table_locks, ticket_idx);

          const std::uint32_t slot =
              (motor_tail_base + motor_emitted) % kMaxEgressQueueSize;
          MotorEvent mevt{};
          mevt.ticket_id = tid;
          mevt.upstream_ticket_id = ancestry.diagnostic_upstream_ticket;
          mevt.node = port.node;
          mevt.channel = port.channel;
          mevt.word = ticket.motor_word;
          mevt.context = ticket.context_signature;
          mevt.timestamp = current_tick;
          mevt.reserved = ancestry.ancestry_incomplete;
          mevt.trajectory = trajectory_artifact;
          direct_network::stage_motor_history_record(
              exact_history != nullptr
                  ? &exact_history->records[exact_history->phase_base + p]
                  : nullptr,
              mevt.ticket_id, mevt.upstream_ticket_id, current_tick, mevt.node,
              mevt.channel, mevt.word, mevt.context, mevt.reserved);
          egress_queue[slot] = mevt;
          if (config.route_efference_copies != 0u) {
            DirectEfferenceCopy copy{};
            copy.ticket_id = tid;
            copy.acting_node = port.node;
            copy.shadow_node = shadow_target;
            copy.channel = port.channel;
            copy.word = ticket.motor_word;
            copy.timestamp = current_tick;
            efference_ring[(*efference_tail + efference_emitted) %
                           kMaxEfferenceRingSize] = copy;
            atomicAdd(&node_incoming_excitation[shadow_target], kQ16One / 8);
            ++efference_emitted;
          }
          __threadfence_system();
          ++motor_emitted;
          if (trajectory_required) {
            advance_resident_motor_trajectory(
                resident_motor_trajectory, device_metrics, false);
            if (resident_motor_trajectory->state ==
                kResidentPublicMotorTrajectoryLive)
              break;
          }
        }
      }
      if (grid.thread_rank() == 0u && motor_history_admitted) {
        if (device_metrics != nullptr)
          atomicAdd(reinterpret_cast<unsigned long long*>(
                        &device_metrics->motor_events_emitted),
                    static_cast<unsigned long long>(motor_emitted));
        if (efference_tail != nullptr) *efference_tail += efference_emitted;
        *egress_tail = motor_tail_base +
            (exact_history != nullptr
                 ? direct_network::finish_exact_history_phase(exact_history)
                 : motor_emitted);
      }
    }

}


}  // namespace substrate::direct_adult_core
