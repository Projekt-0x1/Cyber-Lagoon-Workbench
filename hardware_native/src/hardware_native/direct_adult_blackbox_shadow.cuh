#ifndef HARDWARE_NATIVE_DIRECT_ADULT_BLACKBOX_SHADOW_CUH
#define HARDWARE_NATIVE_DIRECT_ADULT_BLACKBOX_SHADOW_CUH

// d.blackbox_shadow_condensation (#1514): system identification proposing
// candidate recipes under shadow execution and empirical probation.
//
// Law anchors (Revision 12 §7, §10, §11):
//   * black-box condensation is legal only where exact elimination is
//     uneconomic; the landed exact reducer refusing the source is that
//     precondition's executable form;
//   * a candidate shadow is fitted from touched boundary/intervention traces
//     whose episode the speculative-provenance classifier attributes to
//     verified world observation; an endogenous-only episode is refused
//     outright because agreement with resident computation creates no world
//     evidence;
//   * promotion requires held-out equivalence AND interventional traces --
//     do-operated inputs with settled world returns -- never passive
//     agreement alone;
//   * promotion requires measurable resource economics: a shadow that does
//     not undercut the authoritative evaluation it replaces is refused;
//   * promotion materializes a NEW fully plastic recipe with its own logical
//     and revision identities bound to a content-addressed shadow witness;
//     the child is not a label for the parent network.

#include <cstdint>
#include <type_traits>

#include "hardware_native/direct_boundary_condensation.cuh"
#include "hardware_native/direct_exact_history.cuh"

namespace substrate::direct_adult_core {

inline constexpr std::uint32_t kBlackboxShadowTraceCapacity = 8u;
inline constexpr std::uint32_t kBlackboxShadowMinTouchedTraces = 2u;
inline constexpr std::uint32_t kBlackboxShadowMinInterventions = 2u;
inline constexpr std::int32_t kBlackboxEquivalenceToleranceQ16 = 1 << 13;
inline constexpr std::uint16_t kResidentDerivationBlackboxShadowPromoted =
    1u << 3;

enum class DirectBlackboxShadowTraceStage : std::uint32_t {
  touched = 0u,
  held_out = 1u,
};

struct DirectBlackboxShadowTraceSample {
  std::uint64_t occurrence_identity;
  std::int32_t input_q16;
  std::int32_t observed_q16;
  std::uint32_t resident_tick;
  std::uint32_t interventional;
  std::uint32_t held_out;
  std::uint32_t reserved;
};
static_assert(std::is_trivial_v<DirectBlackboxShadowTraceSample> &&
              std::is_standard_layout_v<DirectBlackboxShadowTraceSample> &&
              std::has_unique_object_representations_v<
                  DirectBlackboxShadowTraceSample>);

struct DirectBlackboxShadowTraceSet {
  DirectBlackboxShadowTraceSample samples[kBlackboxShadowTraceCapacity];
  std::uint32_t count;
  std::uint32_t intervention_count;
};
static_assert(std::is_trivial_v<DirectBlackboxShadowTraceSet> &&
              std::is_standard_layout_v<DirectBlackboxShadowTraceSet>);

struct DirectBlackboxShadowModel {
  std::int64_t gain_q16;
  std::int64_t bias_q16;
};

enum class DirectBlackboxShadowRefusal : std::uint32_t {
  none = 0u,
  null_argument = 1u,
  trace_capacity = 2u,
  unordered_or_duplicate_identity = 3u,
  thin_touched_schedule = 4u,
  out_of_guard_trace = 5u,
  zero_variance_schedule = 6u,
  endogenous_provenance = 7u,
  missing_intervention_evidence = 8u,
  intervention_disagreement = 9u,
  held_out_disagreement = 10u,
  economics_refused = 11u,
  already_promoted = 12u,
  capacity_exhausted = 13u,
  duplicate_child_identity = 14u,
  parent_network_unknown = 15u,
};

struct DirectBlackboxShadowCandidate {
  DirectBlackboxShadowModel model;
  std::uint64_t parent_network_identity;
  std::uint64_t witness_identity;
  std::uint64_t child_logical_recipe_id;
  std::int64_t maximum_error_q16;
  std::uint32_t guard_min_q16, guard_max_q16;
  std::uint32_t touched_count, held_out_count, intervention_count;
  std::uint32_t provenance;
  std::uint32_t refusal;
  std::uint32_t nomination_only;
  std::uint32_t promoted;
  std::uint32_t reserved;
};
static_assert(std::is_trivial_v<DirectBlackboxShadowCandidate> &&
              std::is_standard_layout_v<DirectBlackboxShadowCandidate> &&
              std::has_unique_object_representations_v<
                  DirectBlackboxShadowCandidate>);

DIRECT_ADULT_HD inline bool blackbox_div_half_away(std::int64_t numerator,
                                                   std::int64_t denominator,
                                                   std::int64_t* out) {
  if (out == nullptr || denominator <= 0) return false;
  const std::int64_t magnitude =
      numerator >= 0 ? numerator : -numerator;
  const std::int64_t quotient = (2 * magnitude + denominator) / (2 * denominator);
  *out = numerator >= 0 ? quotient : -quotient;
  return true;
}

DIRECT_ADULT_HD inline bool blackbox_shadow_evaluate(
    const DirectBlackboxShadowModel& model, std::int64_t input_q16,
    std::int32_t* output_q16) {
  if (output_q16 == nullptr) return false;
  const std::int64_t scaled = model.gain_q16 * input_q16;
  if (scaled / (1ll << 16) > 0x7fffffffll ||
      scaled / (1ll << 16) < -0x80000000ll)
    return false;
  const std::int64_t value = (scaled >> 16) + model.bias_q16;
  if (value > 0x7fffffffll || value < -0x80000000ll) return false;
  *output_q16 = static_cast<std::int32_t>(value);
  return true;
}

// Trace admission is ordered: occurrence identities strictly increase across
// the whole set so touched and held-out schedules are disjoint by
// construction and every replay walks the same order.
DIRECT_ADULT_HD inline bool blackbox_trace_set_record(
    DirectBlackboxShadowTraceSet* set,
    const DirectBlackboxShadowTraceSample& sample) {
  if (set == nullptr || sample.occurrence_identity == 0u) return false;
  if (set->count >= kBlackboxShadowTraceCapacity) return false;
  for (std::uint32_t i = 0u; i < set->count; ++i)
    if (set->samples[i].occurrence_identity >= sample.occurrence_identity)
      return false;
  set->samples[set->count++] = sample;
  if (sample.interventional != 0u) ++set->intervention_count;
  return true;
}

DIRECT_ADULT_HD inline std::uint64_t blackbox_shadow_witness_identity(
    const DirectBlackboxShadowCandidate& candidate,
    const DirectBlackboxShadowTraceSet& traces) {
  using direct_network::exact_history_fold_word;
  std::uint64_t identity = exact_history_fold_word(0x6262736861646f77ull,
                                                   candidate.parent_network_identity);
  identity = exact_history_fold_word(identity, candidate.provenance);
  identity = exact_history_fold_word(identity, candidate.guard_min_q16);
  identity = exact_history_fold_word(identity, candidate.guard_max_q16);
  identity = exact_history_fold_word(identity, traces.count);
  identity = exact_history_fold_word(identity, traces.intervention_count);
  for (std::uint32_t i = 0u; i < traces.count &&
                             i < kBlackboxShadowTraceCapacity;
       ++i) {
    identity = exact_history_fold_word(identity,
                                       traces.samples[i].occurrence_identity);
    identity = exact_history_fold_word(
        identity, static_cast<std::uint32_t>(traces.samples[i].input_q16));
    identity = exact_history_fold_word(
        identity, static_cast<std::uint32_t>(traces.samples[i].observed_q16));
    identity = exact_history_fold_word(identity, traces.samples[i].resident_tick);
    identity = exact_history_fold_word(identity, traces.samples[i].interventional);
    identity = exact_history_fold_word(identity, traces.samples[i].held_out);
  }
  return identity == 0u ? 1u : identity;
}

// Deterministic integer system identification: one closed-form least-squares
// line through the touched schedule, anchored on its first trace so every
// replay of the same episode produces the same model bytes.
DIRECT_ADULT_HD inline DirectBlackboxShadowRefusal blackbox_shadow_identify(
    const DirectBlackboxShadowTraceSet& traces,
    std::uint64_t parent_network_identity,
    std::uint32_t guard_min_q16, std::uint32_t guard_max_q16,
    direct_network::DirectSpeculativeProvenance provenance,
    DirectBlackboxShadowCandidate* out) {
  if (out == nullptr) return DirectBlackboxShadowRefusal::null_argument;
  if (traces.count > kBlackboxShadowTraceCapacity)
    return DirectBlackboxShadowRefusal::trace_capacity;
  if (provenance != direct_network::DirectSpeculativeProvenance::verified_world_observation)
    return DirectBlackboxShadowRefusal::endogenous_provenance;
  if (parent_network_identity == 0u)
    return DirectBlackboxShadowRefusal::parent_network_unknown;
  if (guard_min_q16 >= guard_max_q16)
    return DirectBlackboxShadowRefusal::out_of_guard_trace;
  std::uint32_t touched_count = 0u, held_out_count = 0u;
  for (std::uint32_t i = 0u; i < traces.count; ++i) {
    if (traces.samples[i].input_q16 <
            static_cast<std::int32_t>(guard_min_q16) ||
        traces.samples[i].input_q16 >
            static_cast<std::int32_t>(guard_max_q16))
      return DirectBlackboxShadowRefusal::out_of_guard_trace;
    if (traces.samples[i].held_out != 0u)
      ++held_out_count;
    else
      ++touched_count;
  }
  if (touched_count < kBlackboxShadowMinTouchedTraces)
    return DirectBlackboxShadowRefusal::thin_touched_schedule;
  const DirectBlackboxShadowTraceSample& anchor = traces.samples[0];
  std::int64_t sxx = 0, sxy = 0;
  for (std::uint32_t i = 0u; i < traces.count; ++i) {
    if (traces.samples[i].held_out != 0u) continue;
    const std::int64_t dx =
        static_cast<std::int64_t>(traces.samples[i].input_q16) - anchor.input_q16;
    const std::int64_t dy = static_cast<std::int64_t>(traces.samples[i].observed_q16) -
                            anchor.observed_q16;
    if (dx > 0x20000000ll || dx < -0x20000000ll || dy > 0x20000000ll ||
        dy < -0x20000000ll)
      return DirectBlackboxShadowRefusal::out_of_guard_trace;
    sxx += dx * dx;
    sxy += dx * dy;
  }
  if (sxx == 0) return DirectBlackboxShadowRefusal::zero_variance_schedule;
  std::int64_t gain = 0, bias = 0, scaled = 0;
  if (sxy > 0x8000000000ll || sxy < -0x8000000000ll)
    return DirectBlackboxShadowRefusal::out_of_guard_trace;
  if (!blackbox_div_half_away(sxy << 16, sxx, &gain))
    return DirectBlackboxShadowRefusal::zero_variance_schedule;
  if (gain > 0x7fffffffll || gain < -0x80000000ll)
    return DirectBlackboxShadowRefusal::out_of_guard_trace;
  if (!blackbox_div_half_away(gain * anchor.input_q16, 1ll << 16, &scaled))
    return DirectBlackboxShadowRefusal::out_of_guard_trace;
  bias = anchor.observed_q16 - scaled;
  if (gain > 0x7fffffffll || gain < -0x80000000ll || bias > 0x7fffffffll ||
      bias < -0x80000000ll)
    return DirectBlackboxShadowRefusal::out_of_guard_trace;
  DirectBlackboxShadowCandidate candidate{};
  candidate.model.gain_q16 = gain;
  candidate.model.bias_q16 = bias;
  candidate.parent_network_identity = parent_network_identity;
  candidate.guard_min_q16 = guard_min_q16;
  candidate.guard_max_q16 = guard_max_q16;
  candidate.touched_count = touched_count;
  candidate.held_out_count = held_out_count;
  candidate.intervention_count = traces.intervention_count;
  candidate.provenance = static_cast<std::uint32_t>(provenance);
  candidate.maximum_error_q16 = 0;
  candidate.refusal = static_cast<std::uint32_t>(DirectBlackboxShadowRefusal::none);
  candidate.nomination_only = 1u;
  candidate.promoted = 0u;
  candidate.witness_identity = blackbox_shadow_witness_identity(candidate, traces);
  using direct_network::exact_history_fold_word;
  std::uint64_t child = exact_history_fold_word(0x6262636f6e64656eull,
                                                candidate.witness_identity);
  child = exact_history_fold_word(child, parent_network_identity);
  candidate.child_logical_recipe_id = child == 0u ? 1u : child;
  *out = candidate;
  return DirectBlackboxShadowRefusal::none;
}

// Held-out probation: no mutation of the model, agreement within tolerance
// required on every probe the fit never saw.
DIRECT_ADULT_HD inline bool blackbox_shadow_held_out_gate(
    DirectBlackboxShadowCandidate* candidate,
    const DirectBlackboxShadowTraceSet& traces,
    std::int32_t tolerance_q16) {
  if (candidate == nullptr || candidate->promoted != 0u) return false;
  std::int64_t maximum = 0;
  for (std::uint32_t i = 0u; i < traces.count; ++i) {
    if (traces.samples[i].held_out == 0u) continue;
    std::int32_t produced = 0;
    if (!blackbox_shadow_evaluate(candidate->model,
                                  traces.samples[i].input_q16, &produced)) {
      candidate->refusal = static_cast<std::uint32_t>(
          DirectBlackboxShadowRefusal::out_of_guard_trace);
      return false;
    }
    const std::int64_t error =
        static_cast<std::int64_t>(produced) - traces.samples[i].observed_q16;
    const std::int64_t magnitude = error >= 0 ? error : -error;
    if (magnitude > maximum) maximum = magnitude;
  }
  if (maximum > tolerance_q16) {
    candidate->refusal = static_cast<std::uint32_t>(
        DirectBlackboxShadowRefusal::held_out_disagreement);
    candidate->maximum_error_q16 = maximum;
    return false;
  }
  if (maximum > candidate->maximum_error_q16)
    candidate->maximum_error_q16 = maximum;
  return true;
}

// Intervention probation: passive agreement never substitutes for do-operated
// traces with settled world returns. A candidate without the minimum
// interventional evidence stays nomination-only no matter how well it agreed.
DIRECT_ADULT_HD inline bool blackbox_shadow_intervention_gate(
    DirectBlackboxShadowCandidate* candidate,
    const DirectBlackboxShadowTraceSet& traces,
    std::int32_t tolerance_q16) {
  if (candidate == nullptr || candidate->promoted != 0u) return false;
  if (traces.intervention_count < kBlackboxShadowMinInterventions) {
    candidate->refusal = static_cast<std::uint32_t>(
        DirectBlackboxShadowRefusal::missing_intervention_evidence);
    return false;
  }
  std::int64_t maximum = 0;
  for (std::uint32_t i = 0u; i < traces.count; ++i) {
    if (traces.samples[i].interventional == 0u) continue;
    std::int32_t produced = 0;
    if (!blackbox_shadow_evaluate(candidate->model,
                                  traces.samples[i].input_q16, &produced)) {
      candidate->refusal = static_cast<std::uint32_t>(
          DirectBlackboxShadowRefusal::out_of_guard_trace);
      return false;
    }
    const std::int64_t error =
        static_cast<std::int64_t>(produced) - traces.samples[i].observed_q16;
    const std::int64_t magnitude = error >= 0 ? error : -error;
    if (magnitude > maximum) maximum = magnitude;
  }
  if (maximum > tolerance_q16) {
    candidate->refusal = static_cast<std::uint32_t>(
        DirectBlackboxShadowRefusal::intervention_disagreement);
    candidate->maximum_error_q16 = maximum;
    return false;
  }
  if (maximum > candidate->maximum_error_q16)
    candidate->maximum_error_q16 = maximum;
  return true;
}

DIRECT_ADULT_HD inline bool blackbox_shadow_economics_viable(
    std::uint64_t authority_cost_steps, std::uint64_t shadow_cost_steps) {
  return shadow_cost_steps != 0u && shadow_cost_steps < authority_cost_steps;
}

// Future Occurrences of the NEW recipe: derived from the child's own logical
// identity, never from the parent network's traced occurrences.
DIRECT_ADULT_HD inline std::uint64_t blackbox_child_occurrence_identity(
    std::uint64_t child_logical_recipe_id, std::uint32_t ordinal) {
  using direct_network::exact_history_fold_word;
  std::uint64_t identity = exact_history_fold_word(0x63686f63637572ull,
                                                   child_logical_recipe_id);
  identity = exact_history_fold_word(identity, ordinal);
  return identity == 0u ? 1u : identity;
}

DIRECT_ADULT_HD inline bool replay_resident_blackbox_shadow_recipe(
    const DirectBlackboxShadowCandidate& candidate,
    const direct_network::ResidentRecipeCell& recipe,
    const direct_network::ResidentRecipeDerivation& derivation) {
  using namespace direct_network;
  if (candidate.promoted == 0u || candidate.witness_identity == 0u ||
      candidate.child_logical_recipe_id == 0u ||
      candidate.child_logical_recipe_id == candidate.parent_network_identity)
    return false;
  if (recipe.logical_recipe_id != candidate.child_logical_recipe_id ||
      derivation.logical_recipe_id != candidate.child_logical_recipe_id ||
      recipe.revision_identity != derivation.revision_identity ||
      derivation.witness_identity != candidate.witness_identity)
    return false;
  if (derivation.port_count != 2u || derivation.relation_count != 1u ||
      derivation.parameter_count != 2u)
    return false;
  if (derivation.relations[0] !=
          static_cast<std::uint32_t>(
              DirectRelationAlgebraFamilyV1::linear) ||
      derivation.parameters_q16[0] !=
          static_cast<std::int32_t>(candidate.model.gain_q16) ||
      derivation.parameters_q16[1] !=
          static_cast<std::int32_t>(candidate.model.bias_q16))
    return false;
  if ((derivation.condensation_flags &
       kResidentDerivationBlackboxShadowPromoted) == 0u)
    return false;
  return recipe.revision == 1u && recipe.credit_q16 == 0 &&
         recipe.revision_identity ==
             resident_recipe_revision_identity(
                 recipe.logical_recipe_id, 0u, 1u, candidate.witness_identity,
                 recipe.support_q16, recipe.credit_q16);
}

// Marriage. Runs the full probation ladder in fixed order -- held-out,
// intervention, economics -- then materializes the NEW fully plastic recipe
// beside the parent. A refusal anywhere leaves the tissue untouched and the
// candidate nomination-only.
DIRECT_ADULT_HD inline bool blackbox_shadow_promote(
    DirectBlackboxShadowCandidate* candidate,
    const DirectBlackboxShadowTraceSet& traces,
    direct_network::ResidentRecipeCell* cells, std::uint32_t* cell_count,
    std::uint32_t cell_capacity,
    direct_network::ResidentRecipeDerivation* derivations,
    direct_network::ResidentPostbirthConstructorState* state,
    std::uint32_t parent_cell, std::uint16_t territory_index,
    std::uint64_t authority_cost_steps, std::uint64_t shadow_cost_steps) {
  using namespace direct_network;
  if (candidate == nullptr || cells == nullptr || cell_count == nullptr ||
      derivations == nullptr || state == nullptr)
    return false;
  if (candidate->promoted != 0u) {
    candidate->refusal =
        static_cast<std::uint32_t>(DirectBlackboxShadowRefusal::already_promoted);
    return false;
  }
  if (!blackbox_shadow_held_out_gate(candidate, traces,
                                     kBlackboxEquivalenceToleranceQ16))
    return false;
  if (!blackbox_shadow_intervention_gate(candidate, traces,
                                         kBlackboxEquivalenceToleranceQ16))
    return false;
  if (!blackbox_shadow_economics_viable(authority_cost_steps,
                                        shadow_cost_steps)) {
    candidate->refusal = static_cast<std::uint32_t>(
        DirectBlackboxShadowRefusal::economics_refused);
    return false;
  }
  if (parent_cell >= *cell_count || *cell_count >= cell_capacity ||
      *cell_count >= state->recipe_cell_capacity ||
      state->derivation_count >= state->derivation_capacity) {
    candidate->refusal = static_cast<std::uint32_t>(
        DirectBlackboxShadowRefusal::capacity_exhausted);
    return false;
  }
  std::uint32_t parent_derivation = state->derivation_count;
  for (std::uint32_t i = 0u; i < state->derivation_count; ++i)
    if (derivations[i].recipe_cell == parent_cell &&
        derivations[i].logical_recipe_id == cells[parent_cell].logical_recipe_id &&
        derivations[i].revision_identity == cells[parent_cell].revision_identity) {
      parent_derivation = i;
      break;
    }
  if (parent_derivation == state->derivation_count ||
      derivations[parent_derivation].port_count < 2u ||
      derivations[parent_derivation].generation == ~std::uint64_t{0}) {
    candidate->refusal = static_cast<std::uint32_t>(
        DirectBlackboxShadowRefusal::parent_network_unknown);
    return false;
  }
  for (std::uint32_t i = 0u; i < *cell_count; ++i)
    if (cells[i].logical_recipe_id == candidate->child_logical_recipe_id) {
      candidate->refusal = static_cast<std::uint32_t>(
          DirectBlackboxShadowRefusal::duplicate_child_identity);
      return false;
    }
  if (candidate->model.gain_q16 > 0x7fffffffll ||
      candidate->model.gain_q16 < -0x80000000ll ||
      candidate->model.bias_q16 > 0x7fffffffll ||
      candidate->model.bias_q16 < -0x80000000ll) {
    candidate->refusal = static_cast<std::uint32_t>(
        DirectBlackboxShadowRefusal::out_of_guard_trace);
    return false;
  }
  const std::uint32_t new_cell = *cell_count;
  ResidentRecipeCell recipe{};
  recipe.logical_recipe_id = candidate->child_logical_recipe_id;
  recipe.support_q16 = cells[parent_cell].support_q16;
  recipe.rule_index = cells[parent_cell].rule_index;
  recipe.revision = 1u;
  recipe.revision_identity = resident_recipe_revision_identity(
      recipe.logical_recipe_id, 0u, 1u, candidate->witness_identity,
      recipe.support_q16, recipe.credit_q16);
  if (!initialize_resident_recipe_update_ir(&recipe)) return false;
  ResidentRecipeDerivation derivation{};
  derivation.logical_recipe_id = recipe.logical_recipe_id;
  derivation.revision_identity = recipe.revision_identity;
  derivation.parent_logical_recipe_id = cells[parent_cell].logical_recipe_id;
  derivation.parent_revision_identity = cells[parent_cell].revision_identity;
  derivation.witness_identity = candidate->witness_identity;
  derivation.generation = derivations[parent_derivation].generation + 1u;
  derivation.recipe_cell = new_cell;
  derivation.parent_recipe_cell = parent_cell;
  derivation.territory_index = territory_index;
  derivation.port_count = 2u;
  derivation.relation_count = 1u;
  derivation.parameter_count = 2u;
  derivation.ports[0] = derivations[parent_derivation].ports[0];
  derivation.ports[1] = derivations[parent_derivation].ports[1];
  derivation.relations[0] = static_cast<std::uint32_t>(
      DirectRelationAlgebraFamilyV1::linear);
  derivation.parameters_q16[0] =
      static_cast<std::int32_t>(candidate->model.gain_q16);
  derivation.parameters_q16[1] =
      static_cast<std::int32_t>(candidate->model.bias_q16);
  derivation.condensation_flags |= kResidentDerivationBlackboxShadowPromoted;
  derivation.condensation_guard_min_q16 = candidate->guard_min_q16;
  derivation.condensation_guard_max_q16 = candidate->guard_max_q16;
  derivation.condensation_maximum_error_q16 =
      static_cast<std::int32_t>(candidate->maximum_error_q16);
  cells[new_cell] = recipe;
  derivations[state->derivation_count] = derivation;
  ++*cell_count;
  ++state->derivation_count;
  state->ports_used += 2u;
  state->relations_used += 1u;
  state->parameters_used += 2u;
  ++state->condensed;
  if (derivation.generation > state->highest_derivation_rank)
    state->highest_derivation_rank = derivation.generation;
  candidate->promoted = 1u;
  candidate->nomination_only = 0u;
  candidate->refusal =
      static_cast<std::uint32_t>(DirectBlackboxShadowRefusal::none);
  return true;
}

}  // namespace substrate::direct_adult_core

#endif  // HARDWARE_NATIVE_DIRECT_ADULT_BLACKBOX_SHADOW_CUH
