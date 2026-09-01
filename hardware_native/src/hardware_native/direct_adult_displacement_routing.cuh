#ifndef HARDWARE_NATIVE_DIRECT_ADULT_DISPLACEMENT_ROUTING_CUH
#define HARDWARE_NATIVE_DIRECT_ADULT_DISPLACEMENT_ROUTING_CUH

#include <cstdint>

#include "direct_adult_backpressure.cuh"
#include "direct_adult_multi_lineage_pending_credit.cuh"

namespace substrate::direct_network {

inline constexpr std::uint32_t kDisplacementCandidateCapacity = 32u;

struct DirectDisplacementRoute {
  DirectPendingCreditAncestry ancestry;
  std::uint64_t route_id;
  std::uint64_t action_identity;
  std::uint32_t drive_q16;
  std::uint32_t inhibition_q16;
};

struct DirectDisplacementRoutingState {
  std::uint64_t pending_sequence;
  std::uint64_t last_settled_sequence;
  std::uint64_t selected_route_id;
  std::uint64_t selected_action_identity;
  std::uint64_t accumulated_pressure_q16;
  std::uint32_t diversion_count;
  std::uint32_t deferred_count;
  std::uint32_t provenance_refusals;
  std::uint32_t no_alternative_refusals;
  std::uint32_t duplicate_refusals;
};

struct DirectDisplacementDecision {
  direct_adult::ResidentBackpressureDecision backpressure;
  std::uint64_t selected_route_id;
  std::uint64_t selected_action_identity;
  std::uint64_t diverted_pressure_q16;
  bool primary_blocked;
  bool deferred;
  bool diverted;
};

__device__ inline bool displacement_route_has_identity(
    const DirectDisplacementRoute& route) {
  return route.route_id != 0u && route.action_identity != 0u &&
         route.ancestry.evidence_sequence != 0u &&
         route.ancestry.participation_ticket != 0u &&
         route.ancestry.construction_identity != 0u;
}

__device__ inline std::uint64_t displacement_pressure_add(
    std::uint64_t accumulated, std::uint32_t drive_q16) {
  const std::uint64_t next = accumulated + drive_q16;
  return next < accumulated ? ~std::uint64_t{0} : next;
}

// Divert only pressure whose primary route is actually inhibited. Alternatives
// must be resident participants in the same causal opportunity and construction;
// their order is not authority. Finite resource admission happens before the
// selected route becomes executable, while deferred pressure remains resident.
__device__ inline bool route_displaced_action_pressure(
    DirectDisplacementRoutingState* state,
    direct_adult::DirectResourceEcologyState* ecology,
    const DirectDisplacementRoute& primary,
    const DirectDisplacementRoute* alternatives,
    std::uint32_t alternative_count, std::uint32_t threshold_q16,
    DirectDisplacementDecision* decision) {
  if (state == nullptr || ecology == nullptr || alternatives == nullptr ||
      decision == nullptr || alternative_count == 0u ||
      alternative_count > kDisplacementCandidateCapacity ||
      !displacement_route_has_identity(primary))
    return false;
  *decision = {};
  if (primary.ancestry.evidence_sequence == state->last_settled_sequence) {
    ++state->duplicate_refusals;
    return false;
  }
  decision->primary_blocked =
      primary.inhibition_q16 != 0u &&
      primary.inhibition_q16 >= primary.drive_q16;
  if (!decision->primary_blocked) return true;

  if (state->pending_sequence != 0u &&
      state->pending_sequence != primary.ancestry.evidence_sequence) {
    ++state->provenance_refusals;
    return false;
  }
  if (state->pending_sequence == 0u) {
    state->pending_sequence = primary.ancestry.evidence_sequence;
    state->accumulated_pressure_q16 = displacement_pressure_add(
        state->accumulated_pressure_q16, primary.drive_q16);
  }
  std::int32_t selected = -1;
  std::uint32_t selected_margin = 0u;
  bool saw_provenance_mismatch = false;
  for (std::uint32_t i = 0u; i < alternative_count; ++i) {
    const DirectDisplacementRoute& candidate = alternatives[i];
    if (!displacement_route_has_identity(candidate) ||
        candidate.route_id == primary.route_id ||
        candidate.ancestry.evidence_sequence !=
            primary.ancestry.evidence_sequence ||
        candidate.ancestry.construction_identity !=
            primary.ancestry.construction_identity) {
      saw_provenance_mismatch = true;
      continue;
    }
    if (candidate.inhibition_q16 >= candidate.drive_q16) continue;
    const std::uint32_t margin =
        candidate.drive_q16 - candidate.inhibition_q16;
    if (selected < 0 || margin > selected_margin ||
        (margin == selected_margin &&
         candidate.route_id < alternatives[selected].route_id)) {
      selected = static_cast<std::int32_t>(i);
      selected_margin = margin;
    }
  }
  if (selected < 0) {
    if (saw_provenance_mismatch) ++state->provenance_refusals;
    ++state->no_alternative_refusals;
    return false;
  }

  if (!direct_adult::apply_resident_backpressure(
          ecology, direct_adult::DirectResourcePoolKind::explicit_interaction,
          direct_adult::DirectResourcePoolKind::representation_source_state,
          1u, threshold_q16, &decision->backpressure))
    return false;
  const auto& route_state = ecology->pools[static_cast<std::uint32_t>(
      direct_adult::DirectResourcePoolKind::representation_source_state)];
  if (decision->backpressure.dampened ||
      decision->backpressure.deferred_units != 0u ||
      route_state.live_units == 0u) {
    ++state->deferred_count;
    decision->deferred = true;
    return true;
  }

  const DirectDisplacementRoute& winner = alternatives[selected];
  state->last_settled_sequence = primary.ancestry.evidence_sequence;
  state->selected_route_id = winner.route_id;
  state->selected_action_identity = winner.action_identity;
  decision->selected_route_id = winner.route_id;
  decision->selected_action_identity = winner.action_identity;
  decision->diverted_pressure_q16 = state->accumulated_pressure_q16;
  decision->diverted = true;
  state->pending_sequence = 0u;
  state->accumulated_pressure_q16 = 0u;
  ++state->diversion_count;
  return true;
}

}  // namespace substrate::direct_network

#endif
