#ifndef HARDWARE_NATIVE_DIRECT_ADULT_ELIGIBILITY_RESIDUAL_META_CUH
#define HARDWARE_NATIVE_DIRECT_ADULT_ELIGIBILITY_RESIDUAL_META_CUH

// d.eligibility_residual_meta (#1524): a settlement compares a constitutionally
// authorized actual participation with the eligibility state and resident
// multi-horizon prediction that existed before its later evidence arrived.
// The result is structured r_E over membership, horizon, timing, context guard,
// magnitude, route, and resource dimensions. Predictions of r_E and predictions
// of their error are ordinary Recipe networks; there is no hard meta-depth enum.
//
// Authority boundary:
//   actual exact-history participation -> frozen eligibility/prediction
//   -> later verified contact -> r_E -> ordinary Recipe prediction/revision.
// Prediction never creates participation or evidence. Error Recipes inherit an
// already-authorized participation identity when they actually enter a closure,
// and experiential credit requires a later verified independent contact. Their
// Recipe bodies can therefore use the same condensation, fission, replacement,
// and revision machinery as any other Recipe while the participation fence stays
// fixed. Event/transport timestamps are not consulted; resident chronology is.

#include <cstdint>
#include <type_traits>

#include "hardware_native/direct_adult_core.cuh"
#include "hardware_native/direct_adult_runtime_frontiers.cuh"
#include "hardware_native/direct_adult_eligibility_coalitions.cuh"
#include "hardware_native/direct_exact_history.cuh"

#ifdef __CUDACC__
#define DIRECT_RESIDUAL_META_HD __host__ __device__
#else
#define DIRECT_RESIDUAL_META_HD
#endif

namespace substrate::direct_network {

inline constexpr std::uint32_t kEligibilityResidualDimensions = 7u;
inline constexpr std::int32_t kEligibilityResidualTickUnitQ16 =
    direct_adult_core::kQ16One / 16;
inline constexpr std::int32_t kEligibilityResidualMagnitudeUnitQ16 =
    direct_adult_core::kQ16One / 16;
inline constexpr std::int32_t kEligibilityResidualResourceUnitQ16 =
    direct_adult_core::kQ16One / 32;

enum class EligibilityResidualDimension : std::uint32_t {
  membership = 0u,
  horizon = 1u,
  timing = 2u,
  context_guard = 3u,
  magnitude = 4u,
  route = 5u,
  resource = 6u,
};

enum class DirectEligibilityResidualState : std::uint32_t {
  free = 0u,
  settled = 1u,
};

struct alignas(8) DirectEligibilityResidualVector {
  std::int32_t q16[kEligibilityResidualDimensions];
  std::uint32_t reserved;
};
static_assert(std::is_standard_layout_v<DirectEligibilityResidualVector> &&
              std::is_trivial_v<DirectEligibilityResidualVector>);

struct alignas(8) DirectEligibilityResidualSettlement {
  std::uint64_t subject;
  std::uint64_t parent_occurrence_identity;
  std::uint64_t participation_identity;
  std::uint64_t revision_identity;
  std::uint64_t opening_identity;
  std::uint64_t evidence_identity;
  std::uint64_t opening_sequence;
  std::uint64_t evidence_sequence;
  std::uint32_t opened_tick;
  std::uint32_t settled_tick;
  std::uint32_t predicted_horizon_ticks;
  std::uint32_t actual_horizon_ticks;
  std::int32_t horizon_correction_ticks;
  DirectEligibilityResidualState state;
  DirectEligibilityResidualVector residual;
};
static_assert(std::is_standard_layout_v<DirectEligibilityResidualSettlement> &&
              std::is_trivial_v<DirectEligibilityResidualSettlement>);

struct alignas(8) DirectEligibilityResidualScore {
  DirectEligibilityResidualVector predicted;
  DirectEligibilityResidualVector observed;
  DirectEligibilityResidualVector error;
};
static_assert(std::is_standard_layout_v<DirectEligibilityResidualScore> &&
              std::is_trivial_v<DirectEligibilityResidualScore>);

// Every dimension is an ordinary trigger Recipe. The extra network identity
// describes topology only; it is not a second mutable learning implementation.
// The learned expected residual lives in each derivation parameter and advances
// through the ordinary Recipe revision identity when settled evidence arrives.
struct alignas(8) DirectEligibilityErrorRecipeNetwork {
  std::uint64_t network_identity;
  std::uint64_t parent_network_identity;
  std::uint64_t subject;
  std::uint32_t generation;
  std::uint32_t observations;
  ResidentRecipeCell recipes[kEligibilityResidualDimensions];
  ResidentRecipeDerivation derivations[kEligibilityResidualDimensions];
};
static_assert(std::is_standard_layout_v<DirectEligibilityErrorRecipeNetwork> &&
              std::is_trivial_v<DirectEligibilityErrorRecipeNetwork>);

DIRECT_RESIDUAL_META_HD inline std::int32_t residual_meta_abs(std::int32_t value) {
  if (value == (-2147483647 - 1)) return 2147483647;
  return value < 0 ? -value : value;
}

DIRECT_RESIDUAL_META_HD inline std::int32_t residual_meta_clamp_signed_q16(
    std::int64_t value) {
  const std::int64_t limit = direct_adult_core::kQ16One;
  if (value > limit) return static_cast<std::int32_t>(limit);
  if (value < -limit) return static_cast<std::int32_t>(-limit);
  return static_cast<std::int32_t>(value);
}

DIRECT_RESIDUAL_META_HD inline bool eligibility_residual_verified_contact(
    const DirectExactHistoryRecord& record) {
  return record.kind == DirectExactHistoryKind::sensory_contact &&
         (record.flags & kDirectHistoryVerifiedObservation) != 0u &&
         record.identity != 0u;
}

DIRECT_RESIDUAL_META_HD inline const DirectExactHistoryRecord*
eligibility_residual_opening_record(
    const direct_adult_core::ResidentRecipeOccurrence& actual,
    const DirectExactHistoryRecord* records, std::uint32_t count) {
  if (records == nullptr ||
      actual.state != direct_adult_core::kResidentRecipeOccurrenceLive ||
      actual.lineage_kind != direct_adult_core::ResidentOccurrenceLineageKind::actual ||
      actual.authority == direct_adult_core::DirectParticipationAuthority::none ||
      actual.participation_identity == 0u || actual.occurrence_identity == 0u ||
      actual.revision_identity == 0u)
    return nullptr;
  const DirectExactHistoryRecord* found = nullptr;
  for (std::uint32_t i = 0u; i < count; ++i) {
    const auto& record = records[i];
    if (!eligibility_residual_verified_contact(record) ||
        record.identity != actual.participation_identity)
      continue;
    if (found != nullptr || record.context != actual.context_signature)
      return nullptr;
    found = &record;
  }
  return found;
}

DIRECT_RESIDUAL_META_HD inline const DirectExactHistoryRecord*
eligibility_residual_later_evidence(
    const DirectExactHistoryRecord& opening,
    const DirectExactHistoryRecord* records, std::uint32_t count) {
  const DirectExactHistoryRecord* evidence = nullptr;
  for (std::uint32_t i = 0u; i < count; ++i) {
    const auto& record = records[i];
    if (!eligibility_residual_verified_contact(record) ||
        record.subject != opening.subject || record.identity == opening.identity ||
        record.sequence <= opening.sequence ||
        record.resident_tick <= opening.resident_tick)
      continue;
    if (evidence == nullptr || record.sequence < evidence->sequence)
      evidence = &record;
  }
  return evidence;
}

DIRECT_RESIDUAL_META_HD inline const DirectEligibilityCoalitionEntry*
eligibility_residual_coalition_entry(
    const DirectEligibilityCoalitionTable& table, std::uint64_t subject) {
  for (std::uint32_t i = 0u; i < table.count; ++i)
    if (table.entries[i].source == subject) return &table.entries[i];
  return nullptr;
}

// This is the canonical settlement path for this node. All comparisons are
// against frozen predecessor state. The only post-opening input is a later
// verified contact already present in device exact history.
DIRECT_RESIDUAL_META_HD inline bool settle_direct_eligibility_residual(
    const direct_adult_core::ResidentRecipeOccurrence& actual,
    const DirectEligibilityCoalitionTable& coalition,
    const direct_adult_core::ResidentMultiHorizonPredictionFrontier& prediction,
    const DirectExactHistoryRecord* records, std::uint32_t count,
    std::int32_t horizon_correction_ticks,
    DirectEligibilityResidualSettlement* out) {
  if (out == nullptr || count == 0u ||
      prediction.live_count == 0u ||
      prediction.live_count > direct_adult_core::kResidentMultiHorizonCapacity)
    return false;
  const auto* opening = eligibility_residual_opening_record(actual, records, count);
  if (opening == nullptr) return false;
  const auto* evidence = eligibility_residual_later_evidence(*opening, records, count);
  if (evidence == nullptr) return false;
  const auto* coalition_entry =
      eligibility_residual_coalition_entry(coalition, opening->subject);
  if (coalition_entry == nullptr) return false;

  std::uint32_t matched = 0u;
  std::uint32_t maximum_horizon_tick = 0u;
  std::uint32_t prediction_work = 0u;
  std::uint64_t maximum_envelope_q16 = 0u;
  std::int64_t nearest_timing_delta = 0x7fffffffffffffffll;
  for (std::uint32_t i = 0u;
       i < direct_adult_core::kResidentMultiHorizonCapacity; ++i) {
    const auto& entry = prediction.entries[i];
    if (entry.state != direct_adult_core::ResidentMultiHorizonState::live ||
        entry.shadow.parent_occurrence_identity != actual.occurrence_identity)
      continue;
    if (entry.shadow.state != direct_adult_core::ResidentSuccessorShadowState::live ||
        entry.shadow.parent_revision_identity != actual.revision_identity ||
        entry.shadow.parent_context_signature != actual.context_signature ||
        entry.shadow.occurrence.lineage_kind !=
            direct_adult_core::ResidentOccurrenceLineageKind::endogenous ||
        entry.shadow.occurrence.authority !=
            direct_adult_core::DirectParticipationAuthority::none ||
        entry.shadow.occurrence.eligibility_q16 != 0)
      return false;
    ++matched;
    if (entry.shadow.horizon_tick > maximum_horizon_tick)
      maximum_horizon_tick = entry.shadow.horizon_tick;
    prediction_work += entry.cumulative_work_units;
    if (entry.predicted_residual.envelope_q16 > maximum_envelope_q16)
      maximum_envelope_q16 = entry.predicted_residual.envelope_q16;
    const std::int64_t shifted =
        static_cast<std::int64_t>(entry.shadow.horizon_tick) +
        horizon_correction_ticks;
    const std::int64_t timing_delta =
        static_cast<std::int64_t>(evidence->resident_tick) - shifted;
    if (nearest_timing_delta == 0x7fffffffffffffffll ||
        (timing_delta < 0 ? -timing_delta : timing_delta) <
            (nearest_timing_delta < 0 ? -nearest_timing_delta
                                      : nearest_timing_delta))
      nearest_timing_delta = timing_delta;
  }
  if (matched == 0u || maximum_horizon_tick < actual.timestamp) return false;

  const std::int64_t base_horizon =
      static_cast<std::int64_t>(maximum_horizon_tick) - actual.timestamp;
  std::int64_t predicted_horizon = base_horizon + horizon_correction_ticks;
  if (predicted_horizon < 1) predicted_horizon = 1;
  if (predicted_horizon > 0xffffffffll) predicted_horizon = 0xffffffffll;
  const std::uint32_t actual_horizon =
      evidence->resident_tick - opening->resident_tick;
  const bool predicted_member =
      coalition_entry->e_support_q16 >= kCoalitionHighThresholdQ16;
  const std::uint64_t bounded_envelope =
      maximum_envelope_q16 > static_cast<std::uint64_t>(direct_adult_core::kQ16One)
          ? direct_adult_core::kQ16One : maximum_envelope_q16;
  const std::uint64_t observed_magnitude =
      evidence->value >= opening->value ? evidence->value - opening->value
                                        : opening->value - evidence->value;
  std::uint32_t actual_work = 0u;
  for (std::uint32_t i = 0u; i < count; ++i)
    if (eligibility_residual_verified_contact(records[i]) &&
        records[i].sequence > opening->sequence &&
        records[i].sequence <= evidence->sequence)
      ++actual_work;
  actual_work += coalition_entry->support_samples + coalition_entry->veto_samples;

  DirectEligibilityResidualSettlement candidate{};
  candidate.subject = opening->subject;
  candidate.parent_occurrence_identity = actual.occurrence_identity;
  candidate.participation_identity = actual.participation_identity;
  candidate.revision_identity = actual.revision_identity;
  candidate.opening_identity = opening->identity;
  candidate.evidence_identity = evidence->identity;
  candidate.opening_sequence = opening->sequence;
  candidate.evidence_sequence = evidence->sequence;
  candidate.opened_tick = opening->resident_tick;
  candidate.settled_tick = evidence->resident_tick;
  candidate.predicted_horizon_ticks = static_cast<std::uint32_t>(predicted_horizon);
  candidate.actual_horizon_ticks = actual_horizon;
  candidate.horizon_correction_ticks = horizon_correction_ticks;
  candidate.state = DirectEligibilityResidualState::settled;
  candidate.residual.q16[static_cast<std::uint32_t>(
      EligibilityResidualDimension::membership)] =
      predicted_member ? 0 : direct_adult_core::kQ16One;
  candidate.residual.q16[static_cast<std::uint32_t>(
      EligibilityResidualDimension::horizon)] = residual_meta_clamp_signed_q16(
      (static_cast<std::int64_t>(actual_horizon) - predicted_horizon) *
      kEligibilityResidualTickUnitQ16);
  candidate.residual.q16[static_cast<std::uint32_t>(
      EligibilityResidualDimension::timing)] = residual_meta_clamp_signed_q16(
      nearest_timing_delta * kEligibilityResidualTickUnitQ16);
  candidate.residual.q16[static_cast<std::uint32_t>(
      EligibilityResidualDimension::context_guard)] =
      evidence->context == opening->context ? 0 : direct_adult_core::kQ16One;
  candidate.residual.q16[static_cast<std::uint32_t>(
      EligibilityResidualDimension::magnitude)] = residual_meta_clamp_signed_q16(
      static_cast<std::int64_t>(observed_magnitude) *
          kEligibilityResidualMagnitudeUnitQ16 -
      static_cast<std::int64_t>(bounded_envelope));
  candidate.residual.q16[static_cast<std::uint32_t>(
      EligibilityResidualDimension::route)] =
      evidence->source == opening->source ? 0 : direct_adult_core::kQ16One;
  candidate.residual.q16[static_cast<std::uint32_t>(
      EligibilityResidualDimension::resource)] = residual_meta_clamp_signed_q16(
      (static_cast<std::int64_t>(actual_work) - prediction_work) *
      kEligibilityResidualResourceUnitQ16);
  *out = candidate;
  return true;
}

DIRECT_RESIDUAL_META_HD inline std::uint64_t eligibility_error_network_identity(
    std::uint64_t seed, std::uint64_t parent, std::uint64_t source,
    std::uint64_t subject) {
  std::uint64_t identity = exact_history_fold_word(0x656c69676572726full, seed);
  identity = exact_history_fold_word(identity, parent);
  identity = exact_history_fold_word(identity, source);
  identity = exact_history_fold_word(identity, subject);
  return identity == 0u ? 1u : identity;
}

DIRECT_RESIDUAL_META_HD inline bool initialize_eligibility_error_recipe_network(
    const DirectEligibilityErrorRecipeNetwork* parent,
    const ResidentRecipeCell& owner_recipe,
    const ResidentRecipeDerivation& owner_derivation,
    const direct_adult_core::ResidentRecipeOccurrence& actual,
    std::uint64_t subject, std::uint64_t seed,
    DirectEligibilityErrorRecipeNetwork* out) {
  if (out == nullptr || seed == 0u || subject == 0u ||
      actual.state != direct_adult_core::kResidentRecipeOccurrenceLive ||
      actual.lineage_kind != direct_adult_core::ResidentOccurrenceLineageKind::actual ||
      actual.authority == direct_adult_core::DirectParticipationAuthority::none ||
      actual.source_identity == 0u || actual.binding_count == 0u ||
      owner_recipe.logical_recipe_id == 0u ||
      owner_recipe.revision_identity == 0u ||
      owner_derivation.logical_recipe_id != owner_recipe.logical_recipe_id ||
      owner_derivation.revision_identity != owner_recipe.revision_identity ||
      owner_derivation.port_count < 2u || owner_derivation.route_index ==
          direct_adult_core::kInvalidIndex ||
      owner_derivation.route_incarnations[0] == 0u)
    return false;
  DirectEligibilityErrorRecipeNetwork candidate{};
  candidate.parent_network_identity = parent == nullptr ? 0u : parent->network_identity;
  candidate.subject = subject;
  candidate.generation = parent == nullptr ? 0u : parent->generation + 1u;
  candidate.network_identity = eligibility_error_network_identity(
      seed, candidate.parent_network_identity, actual.source_identity, subject);
  for (std::uint32_t d = 0u; d < kEligibilityResidualDimensions; ++d) {
    std::uint64_t logical = exact_history_fold_word(candidate.network_identity, d + 1u);
    if (logical == 0u) logical = d + 1u;
    ResidentRecipeCell cell{};
    cell.logical_recipe_id = logical;
    cell.support_q16 = owner_recipe.support_q16;
    cell.revision = 1u;
    cell.rule_index = owner_recipe.rule_index;
    const std::uint64_t contributor =
        exact_history_fold_word(actual.occurrence_identity, d + 1u);
    cell.revision_identity = resident_recipe_revision_identity(
        logical, owner_recipe.revision_identity, cell.revision,
        contributor == 0u ? 1u : contributor, cell.support_q16, 0);
    if (!initialize_resident_recipe_update_ir(&cell)) return false;
    cell.receptor_state = owner_recipe.receptor_state;
    cell.receptor_state.causal_identity =
        exact_history_fold_word(logical, actual.source_identity);
    if (cell.receptor_state.causal_identity == 0u)
      cell.receptor_state.causal_identity = logical;
    cell.receptor_state.revision_identity = cell.revision_identity;
    candidate.recipes[d] = cell;

    ResidentRecipeDerivation derivation{};
    derivation.logical_recipe_id = logical;
    derivation.revision_identity = cell.revision_identity;
    derivation.parent_logical_recipe_id = owner_recipe.logical_recipe_id;
    derivation.parent_revision_identity = owner_recipe.revision_identity;
    derivation.generation = owner_derivation.generation + 1u + candidate.generation;
    derivation.route_incarnations[0] = owner_derivation.route_incarnations[0];
    derivation.route_incarnations[1] = owner_derivation.route_incarnations[1];
    derivation.witness_identity = contributor == 0u ? 1u : contributor;
    derivation.recipe_cell = d;
    derivation.parent_recipe_cell = owner_derivation.recipe_cell;
    derivation.source_node = owner_derivation.source_node;
    derivation.route_index = owner_derivation.route_index;
    derivation.port_count = 2u;
    derivation.relation_count = 1u;
    derivation.parameter_count = 1u;
    derivation.territory_index = owner_derivation.territory_index;
    derivation.ports[0] = owner_derivation.ports[0];
    derivation.ports[1] = owner_derivation.ports[1];
    derivation.relations[0] =
        static_cast<std::uint32_t>(ResidentRecipeRelation::trigger);
    derivation.parameters_q16[0] = 0;
    candidate.derivations[d] = derivation;
  }
  *out = candidate;
  return true;
}

DIRECT_RESIDUAL_META_HD inline bool score_eligibility_error_recipe_network(
    const DirectEligibilityErrorRecipeNetwork& network,
    const DirectEligibilityResidualVector& observed,
    DirectEligibilityResidualScore* out) {
  if (out == nullptr || network.network_identity == 0u) return false;
  DirectEligibilityResidualScore candidate{};
  candidate.observed = observed;
  for (std::uint32_t d = 0u; d < kEligibilityResidualDimensions; ++d) {
    const auto& cell = network.recipes[d];
    const auto& derivation = network.derivations[d];
    if (cell.logical_recipe_id == 0u || cell.revision_identity == 0u ||
        derivation.logical_recipe_id != cell.logical_recipe_id ||
        derivation.revision_identity != cell.revision_identity ||
        !evaluate_resident_recipe_boundary_q16(
            derivation.relations[0], derivation.parameters_q16[0], 0,
            &candidate.predicted.q16[d]))
      return false;
    candidate.error.q16[d] = residual_meta_clamp_signed_q16(
        static_cast<std::int64_t>(observed.q16[d]) -
        candidate.predicted.q16[d]);
  }
  *out = candidate;
  return true;
}

DIRECT_RESIDUAL_META_HD inline bool learn_eligibility_error_recipe_network(
    DirectEligibilityErrorRecipeNetwork* network,
    const DirectEligibilityResidualScore& score, std::uint64_t evidence_identity) {
  if (network == nullptr || network->network_identity == 0u ||
      evidence_identity == 0u)
    return false;
  for (std::uint32_t d = 0u; d < kEligibilityResidualDimensions; ++d) {
    auto& cell = network->recipes[d];
    auto& derivation = network->derivations[d];
    const std::uint64_t prior_revision = cell.revision_identity;
    const std::int64_t next = static_cast<std::int64_t>(
        derivation.parameters_q16[0]) + score.error.q16[d] / 2;
    derivation.parameters_q16[0] = residual_meta_clamp_signed_q16(next);
    std::uint64_t contributor = exact_history_fold_word(
        evidence_identity, network->network_identity);
    contributor = exact_history_fold_word(contributor, d + 1u);
    advance_resident_recipe_revision(
        &cell, contributor == 0u ? 1u : contributor, cell.support_q16,
        cell.credit_q16);
    derivation.parent_logical_recipe_id = cell.logical_recipe_id;
    derivation.parent_revision_identity = prior_revision;
    derivation.revision_identity = cell.revision_identity;
  }
  ++network->observations;
  return true;
}

DIRECT_RESIDUAL_META_HD inline std::int32_t eligibility_error_horizon_correction_ticks(
    const DirectEligibilityErrorRecipeNetwork& network) {
  const auto& derivation = network.derivations[static_cast<std::uint32_t>(
      EligibilityResidualDimension::horizon)];
  if (network.network_identity == 0u ||
      derivation.parameter_count == 0u || kEligibilityResidualTickUnitQ16 == 0)
    return 0;
  return derivation.parameters_q16[0] / kEligibilityResidualTickUnitQ16;
}

DIRECT_RESIDUAL_META_HD inline bool eligibility_error_participation_authorized(
    const DirectEligibilityErrorRecipeNetwork& network,
    const direct_adult_core::ResidentRecipeOccurrence& actual,
    const DirectExactHistoryRecord* records, std::uint32_t count) {
  const auto* opening = eligibility_residual_opening_record(actual, records, count);
  return opening != nullptr && opening->subject == network.subject;
}

DIRECT_RESIDUAL_META_HD inline bool bind_eligibility_error_recipe_occurrence(
    const DirectEligibilityErrorRecipeNetwork& network, std::uint32_t dimension,
    const direct_adult_core::ResidentRecipeOccurrence& actual,
    const DirectExactHistoryRecord* records, std::uint32_t count,
    direct_adult_core::ResidentRecipeOccurrence* out) {
  if (out == nullptr || dimension >= kEligibilityResidualDimensions ||
      actual.binding_count == 0u ||
      !eligibility_error_participation_authorized(network, actual, records, count))
    return false;
  const auto& cell = network.recipes[dimension];
  const auto& derivation = network.derivations[dimension];
  if (derivation.port_count != 2u || cell.revision_identity != derivation.revision_identity)
    return false;
  const std::uint32_t variable = actual.bindings[0].variable_identity;
  if (variable == 0u) return false;
  const std::uint32_t variables[2] = {variable, variable};
  std::uint64_t occurrence_identity = exact_history_fold_word(
      cell.logical_recipe_id, actual.occurrence_identity);
  occurrence_identity = exact_history_fold_word(occurrence_identity, dimension + 1u);
  if (occurrence_identity == 0u) occurrence_identity = cell.logical_recipe_id;
  if (!direct_adult_core::bind_resident_recipe_occurrence(
          cell, derivation, variables, 2u, occurrence_identity,
          actual.participation_identity, actual.source_identity,
          actual.source_incarnation,
          direct_adult_core::ResidentOccurrenceLineageKind::actual,
          actual.authority, actual.context_signature, actual.timestamp,
          actual.expiry_tick, 0, out))
    return false;
  out->route_incarnation = derivation.route_incarnations[0];
  return true;
}

DIRECT_RESIDUAL_META_HD inline bool credit_eligibility_error_recipe(
    DirectEligibilityErrorRecipeNetwork* network, std::uint32_t dimension,
    const direct_adult_core::ResidentRecipeOccurrence& occurrence,
    const DirectExactHistoryRecord& evidence, std::int32_t credit_delta_q16) {
  if (network == nullptr || dimension >= kEligibilityResidualDimensions ||
      credit_delta_q16 == 0 ||
      occurrence.state != direct_adult_core::kResidentRecipeOccurrenceLive ||
      occurrence.lineage_kind != direct_adult_core::ResidentOccurrenceLineageKind::actual ||
      occurrence.authority == direct_adult_core::DirectParticipationAuthority::none ||
      !eligibility_residual_verified_contact(evidence) ||
      evidence.subject != network->subject ||
      evidence.resident_tick <= occurrence.timestamp)
    return false;
  auto& cell = network->recipes[dimension];
  auto& derivation = network->derivations[dimension];
  if (occurrence.logical_recipe_id != cell.logical_recipe_id ||
      occurrence.revision_identity != cell.revision_identity ||
      derivation.revision_identity != cell.revision_identity)
    return false;
  DirectExactHistoryRecord revision_event{};
  if (!stage_resident_recipe_revision_event(
          &revision_event, cell, dimension,
          ResidentRecipeRevisionAuthority::experience,
          occurrence.occurrence_identity, evidence.identity,
          evidence.resident_tick, evidence.resident_tick, evidence.source,
          dimension, evidence.flags, credit_delta_q16))
    return false;
  const std::uint64_t prior_revision = cell.revision_identity;
  if (!apply_resident_recipe_revision_event(&cell, revision_event, dimension))
    return false;
  derivation.parent_logical_recipe_id = cell.logical_recipe_id;
  derivation.parent_revision_identity = prior_revision;
  derivation.revision_identity = cell.revision_identity;
  return true;
}

DIRECT_RESIDUAL_META_HD inline std::int64_t eligibility_error_network_credit(
    const DirectEligibilityErrorRecipeNetwork& network) {
  std::int64_t credit = 0;
  for (std::uint32_t d = 0u; d < kEligibilityResidualDimensions; ++d)
    credit += network.recipes[d].credit_q16;
  return credit;
}

DIRECT_RESIDUAL_META_HD inline bool fission_eligibility_error_recipe_network(
    const DirectEligibilityErrorRecipeNetwork& parent,
    std::uint64_t branch_identity, DirectEligibilityErrorRecipeNetwork* out) {
  if (out == nullptr || parent.network_identity == 0u || branch_identity == 0u)
    return false;
  DirectEligibilityErrorRecipeNetwork child = parent;
  child.parent_network_identity = parent.network_identity;
  child.network_identity = eligibility_error_network_identity(
      branch_identity, parent.network_identity, parent.subject,
      parent.generation + 1u);
  child.generation = parent.generation + 1u;
  child.observations = parent.observations;
  for (std::uint32_t d = 0u; d < kEligibilityResidualDimensions; ++d) {
    const ResidentRecipeCell prior = parent.recipes[d];
    std::uint64_t logical = exact_history_fold_word(child.network_identity, d + 1u);
    if (logical == 0u) logical = d + 1u;
    auto& cell = child.recipes[d];
    cell.logical_recipe_id = logical;
    cell.credit_q16 = 0;
    cell.revision = 1u;
    cell.revision_identity = resident_recipe_revision_identity(
        logical, prior.revision_identity, cell.revision, branch_identity,
        cell.support_q16, 0);
    cell.receptor_state.causal_identity =
        exact_history_fold_word(logical, prior.receptor_state.causal_identity);
    if (cell.receptor_state.causal_identity == 0u)
      cell.receptor_state.causal_identity = logical;
    cell.receptor_state.revision_identity = cell.revision_identity;
    auto& derivation = child.derivations[d];
    derivation.parent_logical_recipe_id = prior.logical_recipe_id;
    derivation.parent_revision_identity = prior.revision_identity;
    derivation.logical_recipe_id = logical;
    derivation.revision_identity = cell.revision_identity;
    ++derivation.generation;
    derivation.witness_identity = branch_identity;
    derivation.recipe_cell = d;
    derivation.parent_recipe_cell = d;
  }
  *out = child;
  return true;
}

DIRECT_RESIDUAL_META_HD inline bool replace_eligibility_error_recipe_network(
    DirectEligibilityErrorRecipeNetwork* current,
    const DirectEligibilityErrorRecipeNetwork& candidate) {
  if (current == nullptr || current->network_identity == 0u ||
      candidate.network_identity == 0u ||
      candidate.parent_network_identity != current->network_identity ||
      candidate.subject != current->subject ||
      candidate.generation != current->generation + 1u ||
      eligibility_error_network_credit(candidate) <=
          eligibility_error_network_credit(*current))
    return false;
  *current = candidate;
  return true;
}

// Residual dimensions remain independent relations. No scalar joint objective
// is formed here, and no host label, language token, expected answer, or semantic
// route appears in settlement, prediction, credit, fission, or replacement.
// Multi-horizon predictions remain endogenous and non-authoritative; only their
// frozen geometry is compared after a verified contact arrives. Coalition state
// remains separately learned support/veto evidence and is never collapsed into
// the residual predictor. The residual predictor receives only the resulting r_E.
//
// Recursive prediction has no MAX_META_LEVEL: callers create another ordinary
// DirectEligibilityErrorRecipeNetwork with a parent pointer and score the prior
// level's error vector. Finite matter and downstream credit determine whether
// such a network persists, condenses, fissions, or is replaced.
//
// Exact replay identity is byte identity only when the same complete state,
// ordered records, frozen predecessor state, and executor are supplied. Separate
// lived organisms are not required to converge microscopically.

}  // namespace substrate::direct_network

#undef DIRECT_RESIDUAL_META_HD

#endif
