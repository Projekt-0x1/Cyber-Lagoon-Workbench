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
    ResidentCueSalienceTable* cue_salience) {
  if (blockIdx.x != 0u || threadIdx.x != 0u)
    return;
  resolve_consequence_ticket_device(
      ticket_table, action_occurrences, action_participation_links,
      action_participant_capacity, target_ticket_id, returned_word,
      origin, admission_tick,
      development != nullptr ? &development->exact_history : nullptr,
      transport_cursor, brain, actual_frontier, causal_credit_predictions,
      nodes, node_count, routes,
      route_incarnations,
      route_capacity, retention_bank, recipe_cells,
      development != nullptr ? development->recipe_cell_count : recipe_cell_count, dense_blocks,
      dense_block_count, learning_rate_q16, shatter_threshold_q16, out_ctx,
      metrics);
  // Berridge attribution at the settlement boundary: a rewarded lived
  // context earns pursuit-readiness salience, hedonic stores untouched.
  if (cue_salience != nullptr && out_ctx->valid != 0u) {
    const AsynchronousTicket& settled_ticket =
        ticket_table[static_cast<std::uint32_t>(out_ctx->target_ticket_id &
                                                0x7ffu)];
    attribute_resident_incentive_salience(cue_salience,
                                          settled_ticket.context_signature,
                                          out_ctx->effective_reward_q16);
  }
}

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
    std::uint32_t current_tick) {
  // A fail-closed resolve retires before any mutable frontier read.
  if (resolved_ctx->valid == 0u) return;

  const ResolvedConsequenceContext ctx = *resolved_ctx;
  const std::uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (ctx.action_bound != 0u) return;
  const std::uint32_t total_elig = *live_eligibility_count;
  if (tid >= total_elig || tid >= kMaxLiveEligibilityRecords) return;

  EligibilityRecord& rec = eligibility_table[tid];
  if (!rec.live) return;

  bool match = false;
  if (ctx.target_ticket_id != kInvalidTicket)
    match = rec.ticket_id == ctx.target_ticket_id ||
            (ctx.upstream_ticket_id != kInvalidTicket &&
             rec.ticket_id == ctx.upstream_ticket_id);
  else
    match = current_tick <= rec.expiry_tick;

  const bool route_current = route_incarnations != nullptr &&
      rec.route_index < route_capacity &&
      route_incarnations[rec.route_index] == rec.route_incarnation;
  if (match && route_current && current_tick <= rec.expiry_tick) {
    std::int32_t delta_w = mul_q16(mul_q16(learning_rate_q16, ctx.effective_reward_q16), rec.eligibility_q16);
    // #1294: the tube's contextual chemistry scales how much credit moves
    // conductance -- a plasticity threshold modulation, never a reward.
    delta_w = mul_q16(delta_w,
                      direct_network::direct_tube_chemistry_q16(
                          nodes[rec.source_node].chemotype,
                          nodes[rec.target_node].chemotype)
                          .plasticity_gain_q16);
    delta_w = mul_q16(delta_w, maturation_plasticity_gain_q16(
        brain, rec.source_node,
        brain.development != nullptr ? brain.development->age_tick : current_tick));

    DirectRoute& route = routes[rec.route_index];
    atomic_clamp_add_q16(&route.conductance_q16, delta_w, kMinConductanceQ16, kMaxConductanceQ16);
    route.last_credit_q16 = delta_w;
    // The signed return belongs to this many-to-one action occurrence. Exact
    // participant identities remain in its link bank; stamping one participant
    // here would falsely assign the action's sign to that participant alone.
    route.last_credit_ticket = ctx.target_ticket_id;

    atomic_ema_update_q16(&nodes[rec.source_node].credit_ema_q16, delta_w);
    atomic_ema_update_q16(&nodes[rec.target_node].credit_ema_q16, delta_w);

    if (ctx.effective_reward_q16 < shatter_threshold_q16 && dense_blocks != nullptr) {
      for (std::uint32_t b = 0; b < dense_block_count; ++b) {
        if (rec.source_node >= dense_blocks[b].node_begin &&
            rec.source_node < dense_blocks[b].node_begin + dense_blocks[b].node_count) {
          dense_blocks[b].flags &= ~direct_network::kDenseBlockFlagTensorEligible;
          atomicAdd(reinterpret_cast<unsigned long long*>(&metrics->dense_shatters), 1ULL);
        }
      }
    }

    rec.live = 0u;

    atomicAdd(reinterpret_cast<unsigned long long*>(&metrics->credit_updates_committed), 1ULL);
    if (delta_w > 0) {
      atomicAdd(reinterpret_cast<unsigned long long*>(&metrics->total_positive_credit_q16), static_cast<unsigned long long>(delta_w));
    } else {
      atomicAdd(reinterpret_cast<unsigned long long*>(&metrics->total_negative_credit_q16), static_cast<unsigned long long>(-delta_w));
    }
  } else if (match && !route_current) {
    rec.live = 0u;
    atomicAdd(reinterpret_cast<unsigned long long*>(
                  &metrics->stale_route_claim_rejects), 1ULL);
  } else if (current_tick > rec.expiry_tick) {
    rec.live = 0u;
  }
}

// Invert live eligibility once so maintenance gets exact O(1) route pins.
__global__ void mark_causally_pinned_routes_kernel(const EligibilityRecord* eligibility_table,
                                                   const std::uint32_t* live_eligibility_count,
                                                   const std::uint64_t* route_incarnations,
                                                   std::uint32_t route_capacity,
                                                   std::uint32_t current_tick,
                                                   std::uint32_t* pin_bits,
                                                   AdultCoreMetrics* metrics) {
  const std::uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  std::uint32_t count = *live_eligibility_count;
  if (count > kMaxLiveEligibilityRecords) count = kMaxLiveEligibilityRecords;
  if (idx >= count) return;
  const EligibilityRecord rec = eligibility_table[idx];
  if (rec.live == 0u) return;
  if (rec.expiry_tick < current_tick) return;
  if (rec.route_index >= route_capacity) return;
  if (route_incarnations == nullptr ||
      route_incarnations[rec.route_index] != rec.route_incarnation) return;
  const std::uint32_t word = rec.route_index >> 5;
  const std::uint32_t bit = 1u << (rec.route_index & 31u);
  // Report the number of ROUTES pinned, not the number of records: many records
  // name the same route, and a count of records would inflate with episode
  // density rather than describe how much matter is protected.
  if ((atomicOr(&pin_bits[word], bit) & bit) == 0u && metrics != nullptr)
    atomicAdd(reinterpret_cast<unsigned long long*>(&metrics->causally_pinned_routes), 1ULL);
}

__global__ void resident_self_compilation_crystallize_kernel(
    const DirectNode* nodes,
    DirectDenseBlock* dense_blocks,
    std::uint32_t dense_block_count,
    AdultCoreMetrics* metrics) {
  const std::uint32_t b = blockIdx.x * blockDim.x + threadIdx.x;
  if (b >= dense_block_count) return;

  DirectDenseBlock& dblock = dense_blocks[b];
  if (dblock.flags & direct_network::kDenseBlockFlagTensorEligible) return;

  std::int32_t sum_credit = 0;
  for (std::uint32_t k = 0; k < dblock.node_count; ++k) {
    sum_credit += nodes[dblock.node_begin + k].credit_ema_q16;
  }

  if (sum_credit > (kQ16One * 2)) {
    dblock.flags |= direct_network::kDenseBlockFlagTensorEligible;
    atomicAdd(reinterpret_cast<unsigned long long*>(&metrics->dense_crystallizations), 1ULL);
  }
}

// #1178: every runtime allocation is silicon. Eligibility, delayed-packet, and
// pending-ticket arrays use their exact pools; all other runtime storage is
// charged as bytes so no saturated gestation pool is double-booked.
__global__ void charge_adult_runtime_working_set_kernel(
    substrate::direct_adult::DirectResourceEcologyState* ecology,
    std::uint64_t eligibility_units, std::uint64_t packet_units, std::uint64_t ticket_units,
    std::uint64_t unpooled_bytes, std::uint32_t* admitted_out) {
  if (threadIdx.x != 0u || blockIdx.x != 0u) return;
  if (ecology == nullptr) {
    *admitted_out = 1u;  // no ledger to obey; historical behaviour
    return;
  }
  using substrate::direct_adult::DirectResourcePoolKind;
  const bool ok_e = substrate::direct_adult::device_reserve_pool_units(
      ecology, DirectResourcePoolKind::eligibility_record, eligibility_units);
  const bool ok_p = ok_e && substrate::direct_adult::device_reserve_pool_units(
                                ecology, DirectResourcePoolKind::delayed_packet, packet_units);
  const bool ok_t = ok_p && substrate::direct_adult::device_reserve_pool_units(
                                ecology, DirectResourcePoolKind::pending_consequence_ticket,
                                ticket_units);
  // The unpooled remainder is charged as raw bytes against the same global bound,
  // failure-atomically with the rest: if it does not fit, everything above is
  // rolled back so a refused runtime leaves no phantom exhaustion behind.
  bool ok_b = ok_t;
  if (ok_t && unpooled_bytes != 0u && ecology->global_capacity_bytes != 0u) {
    const unsigned long long prev = substrate::direct_adult::direct_ecology_atomic_add_u64(
        &ecology->global_charged_bytes, unpooled_bytes);
    if (prev + unpooled_bytes > ecology->global_capacity_bytes) {
      substrate::direct_adult::direct_ecology_atomic_sub_u64(&ecology->global_charged_bytes,
                                                             unpooled_bytes);
      substrate::direct_adult::direct_ecology_atomic_add_u64(&ecology->global_rejected_bytes,
                                                             unpooled_bytes);
      ok_b = false;
    }
  }
  if (!ok_b) {
    if (ok_t)
      substrate::direct_adult::device_cancel_pool_reservation(
          ecology, DirectResourcePoolKind::pending_consequence_ticket, ticket_units);
    if (ok_p)
      substrate::direct_adult::device_cancel_pool_reservation(
          ecology, DirectResourcePoolKind::delayed_packet, packet_units);
    if (ok_e)
      substrate::direct_adult::device_cancel_pool_reservation(
          ecology, DirectResourcePoolKind::eligibility_record, eligibility_units);
    *admitted_out = 0u;
    return;
  }
  // The buffers are about to exist, so the reservations become live.
  substrate::direct_adult::device_commit_pool_units(
      ecology, DirectResourcePoolKind::eligibility_record, eligibility_units);
  substrate::direct_adult::device_commit_pool_units(
      ecology, DirectResourcePoolKind::delayed_packet, packet_units);
  substrate::direct_adult::device_commit_pool_units(
      ecology, DirectResourcePoolKind::pending_consequence_ticket, ticket_units);
  *admitted_out = 1u;
}

__global__ void release_adult_runtime_working_set_kernel(
    substrate::direct_adult::DirectResourceEcologyState* ecology,
    std::uint64_t eligibility_units, std::uint64_t packet_units, std::uint64_t ticket_units,
    std::uint64_t unpooled_bytes) {
  if (threadIdx.x != 0u || blockIdx.x != 0u || ecology == nullptr) return;
  using substrate::direct_adult::DirectResourcePoolKind;
  // Release, not uncommit: these really are separate cudaFrees, so the silicon
  // genuinely goes back to the driver. This is the case the grown arena is NOT.
  substrate::direct_adult::device_release_pool_units(
      ecology, DirectResourcePoolKind::eligibility_record, eligibility_units);
  substrate::direct_adult::device_release_pool_units(ecology, DirectResourcePoolKind::delayed_packet,
                                                     packet_units);
  substrate::direct_adult::device_release_pool_units(
      ecology, DirectResourcePoolKind::pending_consequence_ticket, ticket_units);
  if (unpooled_bytes != 0u && ecology->global_capacity_bytes != 0u) {
    substrate::direct_adult::direct_ecology_atomic_sub_u64(&ecology->global_charged_bytes,
                                                           unpooled_bytes);
  }
}

}  // namespace substrate::direct_adult_core
