#ifndef HARDWARE_NATIVE_DIRECT_ADULT_MULTI_HORIZON_PREDICTION_CUH
#define HARDWARE_NATIVE_DIRECT_ADULT_MULTI_HORIZON_PREDICTION_CUH

inline constexpr std::uint32_t kResidentPredictionHorizonCount = 3u;
inline constexpr std::uint32_t kResidentMultiHorizonCapacity = 6u;
inline constexpr std::uint32_t kResidentMultiHorizonWorkBudget = 24u;

enum class ResidentMultiHorizonState : std::uint32_t {
  free = 0u,
  live = 1u,
  expired = 2u,
};

struct alignas(8) ResidentPredictedResidualEnvelope {
  std::int64_t recurrence_residual_q16;
  std::int64_t adjacent_horizon_residual_q16;
  std::uint64_t envelope_q16;
};
static_assert(
    std::is_standard_layout_v<ResidentPredictedResidualEnvelope> &&
    std::is_trivial_v<ResidentPredictedResidualEnvelope> &&
    std::has_unique_object_representations_v<
        ResidentPredictedResidualEnvelope>);

struct alignas(8) ResidentMultiHorizonPredictionEntry {
  ResidentSuccessorShadowEntry shadow;
  ResidentPredictedResidualEnvelope predicted_residual;
  std::uint32_t horizon_ordinal;
  std::uint32_t recurrent_depth;
  std::uint32_t cumulative_work_units;
  ResidentMultiHorizonState state;
};
static_assert(
    std::is_standard_layout_v<ResidentMultiHorizonPredictionEntry> &&
    std::is_trivial_v<ResidentMultiHorizonPredictionEntry> &&
    std::has_unique_object_representations_v<
        ResidentMultiHorizonPredictionEntry>);

struct alignas(8) ResidentMultiHorizonPredictionFrontier {
  ResidentMultiHorizonPredictionEntry entries[kResidentMultiHorizonCapacity];
  std::uint32_t live_count;
  std::uint32_t generations;
  std::uint32_t refusals;
  std::uint32_t total_work_units;
};
static_assert(
    std::is_standard_layout_v<ResidentMultiHorizonPredictionFrontier> &&
    std::is_trivial_v<ResidentMultiHorizonPredictionFrontier> &&
    std::has_unique_object_representations_v<
        ResidentMultiHorizonPredictionFrontier>);

DIRECT_ADULT_HD inline std::uint64_t resident_prediction_magnitude(
    std::int64_t value) {
  return value < 0 ? static_cast<std::uint64_t>(-(value + 1)) + 1u
                   : static_cast<std::uint64_t>(value);
}

DIRECT_ADULT_HD inline std::uint32_t expire_resident_multi_horizon_predictions(
    ResidentMultiHorizonPredictionFrontier* frontier,
    std::uint32_t current_tick) {
  if (frontier == nullptr) return 0u;
  std::uint32_t live = 0u;
  for (std::uint32_t i = 0u; i < kResidentMultiHorizonCapacity; ++i) {
    auto& entry = frontier->entries[i];
    if (entry.state == ResidentMultiHorizonState::live &&
        entry.shadow.horizon_tick < current_tick) {
      entry.state = ResidentMultiHorizonState::expired;
      entry.shadow.state = ResidentSuccessorShadowState::expired;
      entry.shadow.occurrence.state = kResidentRecipeOccurrenceSettled;
    }
    live += entry.state == ResidentMultiHorizonState::live ? 1u : 0u;
  }
  frontier->live_count = live;
  return live;
}

// Generate one fixed, bounded horizon bank from resident executable morphology.
// The caller selects only the touched actual parent. Morphology fixes the
// temporal stride, and later horizons perform strictly more recurrent work.
// The complete bank is assembled off to the side and committed atomically.
DIRECT_ADULT_HD inline bool generate_resident_multi_horizon_prediction(
    const DirectBrain& brain, const ResidentActualFrontier& actual_frontier,
    std::uint32_t parent_frontier_slot, std::uint32_t current_tick,
    ResidentMultiHorizonPredictionFrontier* frontier) {
  using namespace direct_network;
  if (frontier == nullptr ||
      parent_frontier_slot >= kResidentActualFrontierCapacity) {
    if (frontier != nullptr) ++frontier->refusals;
    return false;
  }

  const ResidentActualFrontierEntry& parent =
      actual_frontier.entries[parent_frontier_slot];
  if (parent.state != ResidentActualFrontierState::live ||
      parent.derivation_index >= kResidentPostbirthRecipeReserve ||
      brain.postbirth_constructor == nullptr ||
      brain.postbirth_derivations == nullptr ||
      parent.derivation_index >=
          brain.postbirth_constructor->derivation_count ||
      parent.work.identity != resident_executable_work_identity(parent.work)) {
    ++frontier->refusals;
    return false;
  }

  ResidentMultiHorizonPredictionFrontier candidate = *frontier;
  expire_resident_multi_horizon_predictions(&candidate, current_tick);
  for (std::uint32_t i = 0u; i < kResidentMultiHorizonCapacity; ++i)
    if (candidate.entries[i].state == ResidentMultiHorizonState::live &&
        candidate.entries[i].shadow.parent_occurrence_identity ==
            parent.occurrence.occurrence_identity) {
      ++frontier->refusals;
      return false;
    }

  std::uint32_t slots[kResidentPredictionHorizonCount]{};
  std::uint32_t free_count = 0u;
  for (std::uint32_t i = 0u;
       i < kResidentMultiHorizonCapacity &&
       free_count < kResidentPredictionHorizonCount;
       ++i)
    if (candidate.entries[i].state != ResidentMultiHorizonState::live)
      slots[free_count++] = i;
  if (free_count != kResidentPredictionHorizonCount) {
    ++frontier->refusals;
    return false;
  }

  const ResidentRecipeDerivation& derivation =
      brain.postbirth_derivations[parent.derivation_index];
  ResidentExecutableMorphologyWork exact_parent_work{};
  if (!lower_resident_executable_morphology(
          brain, derivation, parent.occurrence, false, &exact_parent_work) ||
      exact_parent_work.identity != parent.work.identity ||
      exact_parent_work.route_count == 0u ||
      exact_parent_work.route_count > 2u) {
    ++frontier->refusals;
    return false;
  }
  const std::uint32_t morphology_stride = exact_parent_work.route_count;
  std::int32_t previous_projection = parent.output_q16;
  std::uint32_t bank_work = 0u;
  for (std::uint32_t horizon = 0u;
       horizon < kResidentPredictionHorizonCount; ++horizon) {
    const std::uint32_t depth = morphology_stride * (horizon + 1u);
    if (depth == 0u || depth > kResidentMultiHorizonWorkBudget ||
        current_tick > 0xffffffffu - depth) {
      ++frontier->refusals;
      return false;
    }

    ResidentSuccessorShadowFrontier scratch{};
    if (!generate_resident_successor_shadow(
            brain, actual_frontier, parent_frontier_slot, current_tick, depth,
            &scratch) || scratch.live_count != 1u) {
      ++frontier->refusals;
      return false;
    }
    ResidentSuccessorShadowEntry shadow{};
    bool found = false;
    for (std::uint32_t i = 0u; i < kResidentSuccessorShadowCapacity; ++i)
      if (scratch.entries[i].state == ResidentSuccessorShadowState::live) {
        shadow = scratch.entries[i];
        found = true;
      }
    if (!found) {
      ++frontier->refusals;
      return false;
    }

    std::int32_t projection = shadow.projected_state_q16;
    std::int32_t penultimate = parent.output_q16;
    std::uint32_t cumulative_work = shadow.work_units;
    std::uint32_t variables[kResidentDerivationWidth]{};
    for (std::uint32_t i = 0u; i < shadow.occurrence.binding_count; ++i)
      variables[i] = shadow.occurrence.bindings[i].variable_identity;
    for (std::uint32_t step = 1u; step < depth; ++step) {
      penultimate = projection;
      std::uint32_t step_work = 0u;
      if (!execute_resident_executable_morphology(
              brain, derivation, shadow.occurrence, shadow.work, projection,
              &projection, &step_work) ||
          step_work == 0u ||
          cumulative_work > kResidentMultiHorizonWorkBudget - step_work ||
          !apply_resident_occurrence_bound_activation(
              &shadow.occurrence, shadow.occurrence.occurrence_identity,
              shadow.occurrence.context_signature, variables,
              shadow.occurrence.binding_count, current_tick, projection)) {
        ++frontier->refusals;
        return false;
      }
      cumulative_work += step_work;
    }
    if (bank_work > kResidentMultiHorizonWorkBudget - cumulative_work) {
      ++frontier->refusals;
      return false;
    }
    bank_work += cumulative_work;
    shadow.projected_state_q16 = projection;
    shadow.work_units = cumulative_work;

    ResidentMultiHorizonPredictionEntry entry{};
    entry.shadow = shadow;
    entry.predicted_residual.recurrence_residual_q16 =
        static_cast<std::int64_t>(projection) - penultimate;
    entry.predicted_residual.adjacent_horizon_residual_q16 =
        static_cast<std::int64_t>(projection) - previous_projection;
    const std::uint64_t recurrence_magnitude = resident_prediction_magnitude(
        entry.predicted_residual.recurrence_residual_q16);
    const std::uint64_t adjacent_magnitude = resident_prediction_magnitude(
        entry.predicted_residual.adjacent_horizon_residual_q16);
    entry.predicted_residual.envelope_q16 =
        recurrence_magnitude > adjacent_magnitude ? recurrence_magnitude
                                                  : adjacent_magnitude;
    entry.horizon_ordinal = horizon;
    entry.recurrent_depth = depth;
    entry.cumulative_work_units = cumulative_work;
    entry.state = ResidentMultiHorizonState::live;
    candidate.entries[slots[horizon]] = entry;
    previous_projection = projection;
  }

  candidate.live_count += kResidentPredictionHorizonCount;
  ++candidate.generations;
  candidate.total_work_units += bank_work;
  *frontier = candidate;
  return true;
}

DIRECT_ADULT_HD inline std::uint32_t refresh_resident_causal_credit_predictions(
    const DirectBrain& brain, const ResidentActualFrontier& actual_frontier,
    std::uint32_t current_tick,
    ResidentMultiHorizonPredictionFrontier* prediction,
    const ResidentActivationSoaPlane* activation_plane = nullptr) {
  if (prediction == nullptr) return 0u;
  expire_resident_multi_horizon_predictions(prediction, current_tick);
  std::uint32_t generated = 0u;
  // Production consumers read the packed identity lanes when the derived
  // activation plane is current; otherwise the canonical AoS scan serves.
  const bool packed = activation_plane != nullptr &&
                      resident_activation_soa_current(
                          actual_frontier, activation_plane->control,
                          activation_plane->soa);
  const ResidentOccurrenceActivationSoa* soa =
      packed ? &activation_plane->soa : nullptr;
  const std::uint32_t lanes =
      packed ? soa->count : kResidentActualFrontierCapacity;
  for (std::uint32_t lane = 0u; lane < lanes; ++lane) {
    const std::uint32_t slot = packed ? soa->source_slot[lane] : lane;
    const auto& actual = actual_frontier.entries[slot];
    if (actual.state != ResidentActualFrontierState::live) continue;
    bool present = false;
    for (std::uint32_t i = 0u; i < kResidentMultiHorizonCapacity; ++i)
      present |= prediction->entries[i].state ==
                     ResidentMultiHorizonState::live &&
                 prediction->entries[i].shadow.parent_occurrence_identity ==
                     actual.occurrence.occurrence_identity;
    if (!present && generate_resident_multi_horizon_prediction(
                        brain, actual_frontier, slot, current_tick,
                        prediction))
      ++generated;
  }
  return generated;
}

static __global__ void refresh_causal_credit_predictions_kernel(
    DirectBrain brain, const ResidentActualFrontier* actual_frontier,
    std::uint32_t current_tick,
    ResidentMultiHorizonPredictionFrontier* prediction) {
  if (blockIdx.x == 0u && threadIdx.x == 0u && actual_frontier != nullptr)
    refresh_resident_causal_credit_predictions(
        brain, *actual_frontier, current_tick, prediction);
}

struct alignas(8) ResidentCausalCreditPrecision {
  std::uint64_t participation_identity;
  std::uint64_t occurrence_identity;
  std::uint64_t revision_identity;
  std::int32_t verified_delta_q16;
  std::int32_t adjusted_delta_q16;
  std::uint32_t precision_q16;
  std::uint32_t horizon_count;
};
static_assert(std::is_standard_layout_v<ResidentCausalCreditPrecision> &&
              std::is_trivial_v<ResidentCausalCreditPrecision> &&
              std::has_unique_object_representations_v<
                  ResidentCausalCreditPrecision>);

// Prediction is endogenous uncertainty evidence, never causal authority. It
// may attenuate the magnitude already assigned to one exact actual participant
// but cannot create a recipient or reverse the verified world-return sign.
DIRECT_ADULT_HD inline bool refine_resident_causal_credit_precision(
    const ResidentRecipeOccurrence& actual,
    std::uint64_t authorized_participation_identity,
    std::int32_t verified_delta_q16,
    const ResidentMultiHorizonPredictionFrontier& prediction,
    ResidentCausalCreditPrecision* out) {
  if (out == nullptr || verified_delta_q16 == 0 ||
      verified_delta_q16 == (-2147483647 - 1) ||
      actual.state != kResidentRecipeOccurrenceLive ||
      actual.lineage_kind != ResidentOccurrenceLineageKind::actual ||
      actual.authority == DirectParticipationAuthority::none ||
      actual.occurrence_identity == 0u || actual.revision_identity == 0u ||
      actual.participation_identity == 0u ||
      actual.participation_identity != authorized_participation_identity ||
      prediction.live_count == 0u ||
      prediction.live_count > kResidentMultiHorizonCapacity)
    return false;

  std::uint32_t live = 0u;
  std::uint64_t envelope = 0u;
  for (std::uint32_t i = 0u; i < kResidentMultiHorizonCapacity; ++i) {
    const auto& entry = prediction.entries[i];
    if (entry.state != ResidentMultiHorizonState::live) continue;
    const auto& shadow = entry.shadow;
    if (shadow.parent_occurrence_identity != actual.occurrence_identity)
      continue;
    if (shadow.state != ResidentSuccessorShadowState::live ||
        shadow.parent_revision_identity != actual.revision_identity ||
        shadow.parent_context_signature != actual.context_signature ||
        shadow.occurrence.lineage_kind !=
            ResidentOccurrenceLineageKind::endogenous ||
        shadow.occurrence.authority != DirectParticipationAuthority::none ||
        shadow.occurrence.eligibility_q16 != 0)
      return false;
    envelope = entry.predicted_residual.envelope_q16 > envelope
        ? entry.predicted_residual.envelope_q16 : envelope;
    ++live;
  }
  if (live == 0u) return false;

  const std::uint32_t uncertainty_bucket = static_cast<std::uint32_t>(
      (envelope >> 16u) > 3u ? 3u : (envelope >> 16u));
  const std::uint32_t precision_q16 =
      (1u << 16u) / (1u + uncertainty_bucket);
  const std::int64_t magnitude = verified_delta_q16 < 0
      ? -static_cast<std::int64_t>(verified_delta_q16)
      : verified_delta_q16;
  std::int64_t adjusted = (magnitude * precision_q16) >> 16u;
  if (adjusted == 0) adjusted = 1;
  ResidentCausalCreditPrecision candidate{};
  candidate.participation_identity = actual.participation_identity;
  candidate.occurrence_identity = actual.occurrence_identity;
  candidate.revision_identity = actual.revision_identity;
  candidate.verified_delta_q16 = verified_delta_q16;
  candidate.adjusted_delta_q16 = static_cast<std::int32_t>(
      verified_delta_q16 < 0 ? -adjusted : adjusted);
  candidate.precision_q16 = precision_q16;
  candidate.horizon_count = live;
  *out = candidate;
  return true;
}

__device__ inline bool refine_frozen_action_credit_precision(
    const ResidentActualFrontier* actual_frontier,
    const ResidentMultiHorizonPredictionFrontier* predictions,
    const DirectActionParticipationLink& link, std::int32_t requested_delta_q16,
    std::int32_t* adjusted_delta_q16) {
  const ResidentRecipeOccurrence* actual = nullptr;
  for (std::uint32_t slot = 0u;
       actual_frontier != nullptr && slot < kResidentActualFrontierCapacity;
       ++slot) {
    const auto& candidate = actual_frontier->entries[slot];
    if (candidate.state == ResidentActualFrontierState::live &&
        candidate.occurrence.occurrence_identity == link.occurrence_identity) {
      if (actual != nullptr) return false;
      actual = &candidate.occurrence;
    }
  }
  ResidentCausalCreditPrecision precision{};
  if (actual == nullptr || predictions == nullptr ||
      adjusted_delta_q16 == nullptr ||
      !refine_resident_causal_credit_precision(
          *actual, link.participation_identity, requested_delta_q16,
          *predictions, &precision))
    return false;
  *adjusted_delta_q16 = precision.adjusted_delta_q16;
  return true;
}
#include "hardware_native/direct_adult_mismatch_omission.cuh"
#endif
