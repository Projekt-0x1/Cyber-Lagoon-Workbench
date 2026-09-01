#include "hardware_native/direct_adult_persistent_phases.cuh"
#include "hardware_native/direct_adult_device_ops.cuh"
#include "hardware_native/direct_adult_dense_execution.cuh"
#include "hardware_native/direct_boundary_condensation.cuh"
#include "hardware_native/direct_bounded_sm_workspace.cuh"
#include "hardware_native/direct_adult_activation.cuh"
#include "hardware_native/direct_adult_affect_body.cuh"
#include "hardware_native/direct_adult_incentive_salience.cuh"
#include "hardware_native/direct_adult_resource_maintenance.cuh"

namespace substrate::direct_adult_core {

__device__ __noinline__ void persistent_settlement_phase_device(
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
    DirectBrain brain) {
    // 3. Consequence Settlement & Credit Assimilation (Fail-Closed)
    const std::uint32_t c_count = epoch_snap.consequence_tail - epoch_snap.consequence_head;
    if (c_count != 0u && c_count <= kMaxAsynchronousTickets && consequence_queue != nullptr) {
      for (std::uint32_t i = 0u; i < c_count; ++i) {
        const std::uint32_t slot = (epoch_snap.consequence_head + i) % kMaxAsynchronousTickets;
        const ConsequenceIngressEvent ce = consequence_queue[slot];

        if (grid.thread_rank() == 0) {
          ResolvedConsequenceContext ctx{};
          resolve_consequence_ticket_device(
              ticket_table, action_occurrences, action_participation_links,
              action_participant_capacity, ce.ticket_id, ce.returned_word,
              ce.origin, ce.admission_tick, exact_history,
              epoch_snap.consequence_head + i + 1u, brain, actual_frontier,
              causal_credit_predictions,
              nodes, node_count,
              routes, brain.route_incarnations, route_capacity,
              brain.retention_bank, brain.recipe_cells, brain.development->recipe_cell_count,
              brain.dense_blocks, brain.dense_block_count,
              config.learning_rate_q16, config.dense_shatter_threshold_q16,
              &ctx, device_metrics);
          if (ctx.valid != 0u && cue_salience_table != nullptr) {
            const AsynchronousTicket& settled_ticket =
                ticket_table[static_cast<std::uint32_t>(
                    ctx.target_ticket_id & 0x7ffu)];
            attribute_resident_incentive_salience(
                cue_salience_table, settled_ticket.context_signature,
                ctx.effective_reward_q16);
          }
          if (resolved_consequence_ctx != nullptr)
            *resolved_consequence_ctx = ctx;
        }

        grid.sync();

        if (resolved_consequence_ctx != nullptr &&
            resolved_consequence_ctx->history_refused != 0u)
          break;

        if (resolved_consequence_ctx != nullptr &&
            resolved_consequence_ctx->valid != 0u && routes != nullptr) {
          const ResolvedConsequenceContext ctx = *resolved_consequence_ctx;
          if (ctx.action_bound == 0u && eligibility_table != nullptr) {
            const std::uint32_t total_elig = live_eligibility_count
                ? *live_eligibility_count
                : kMaxLiveEligibilityRecords;
            const std::uint32_t bound_elig =
                total_elig < static_cast<std::uint32_t>(kMaxLiveEligibilityRecords)
                    ? total_elig
                    : static_cast<std::uint32_t>(kMaxLiveEligibilityRecords);

            for (std::uint32_t r_idx = global_tid; r_idx < bound_elig;
                 r_idx += total_threads) {
              EligibilityRecord& rec = eligibility_table[r_idx];
              if (!rec.live) continue;

              const bool match = ctx.target_ticket_id != kInvalidTicket
                  ? rec.ticket_id == ctx.target_ticket_id ||
                        (ctx.upstream_ticket_id != kInvalidTicket &&
                         rec.ticket_id == ctx.upstream_ticket_id)
                  : current_tick <= rec.expiry_tick;
              const bool route_current = brain.route_incarnations != nullptr &&
                  rec.route_index < route_capacity &&
                  brain.route_incarnations[rec.route_index] == rec.route_incarnation;
              if (match && route_current && current_tick <= rec.expiry_tick) {
                std::int32_t delta_w = mul_q16(
                    mul_q16(config.learning_rate_q16, ctx.effective_reward_q16),
                    rec.eligibility_q16);
                if (nodes != nullptr)
                  delta_w = mul_q16(
                      delta_w,
                      direct_network::direct_tube_chemistry_q16(
                          nodes[rec.source_node].chemotype,
                          nodes[rec.target_node].chemotype)
                          .plasticity_gain_q16);
                if (nodes != nullptr)
                  delta_w = mul_q16(delta_w, maturation_plasticity_gain_q16(
                      brain, rec.source_node, brain.development != nullptr
                          ? brain.development->age_tick : current_tick));
                DirectRoute& route = routes[rec.route_index];
                atomic_clamp_add_q16(&route.conductance_q16, delta_w,
                                     kMinConductanceQ16, kMaxConductanceQ16);
                route.last_credit_q16 = delta_w;
                route.last_credit_ticket = ctx.target_ticket_id;

                if (nodes != nullptr) {
                  atomic_ema_update_q16(&nodes[rec.source_node].credit_ema_q16,
                                         delta_w);
                  atomic_ema_update_q16(&nodes[rec.target_node].credit_ema_q16,
                                         delta_w);
                }
                rec.live = 0u;

                if (device_metrics != nullptr) {
                  atomicAdd(reinterpret_cast<unsigned long long*>(
                                &device_metrics->credit_updates_committed), 1ULL);
                  if (delta_w > 0) {
                    atomicAdd(reinterpret_cast<unsigned long long*>(
                                  &device_metrics->total_positive_credit_q16),
                              static_cast<unsigned long long>(delta_w));
                  } else {
                    atomicAdd(reinterpret_cast<unsigned long long*>(
                                  &device_metrics->total_negative_credit_q16),
                              static_cast<unsigned long long>(-delta_w));
                  }
                }
              } else if (match && !route_current) {
                rec.live = 0u;
                if (device_metrics != nullptr)
                  atomicAdd(reinterpret_cast<unsigned long long*>(
                                &device_metrics->stale_route_claim_rejects),
                            1ULL);
              } else if (current_tick > rec.expiry_tick) {
                rec.live = 0u;
              }
            }
          }
        }

        grid.sync();
        if (grid.thread_rank() == 0u && consequence_control != nullptr)
          consequence_control->consumed_head =
              epoch_snap.consequence_head + i + 1u;
        grid.sync();
      }
    }

    grid.sync();

    if (grid.thread_rank() == 0u && affect_body_state != nullptr &&
        ticket_table != nullptr && exact_history != nullptr)
      direct_network::affect_derive_from_ledgers(
          affect_body_state, ticket_table, kMaxAsynchronousTickets,
          exact_history->records, exact_history->committed_slots, 0u,
          kInvalidIndex);

}

}  // namespace substrate::direct_adult_core
