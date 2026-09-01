#ifndef HARDWARE_NATIVE_DIRECT_ADULT_REPRESENTATION_EQUIVALENCE_CUH
#define HARDWARE_NATIVE_DIRECT_ADULT_REPRESENTATION_EQUIVALENCE_CUH

#include <cstdint>
#include <type_traits>

#include "hardware_native/direct_network_brain.cuh"

#if defined(__CUDACC__)
#define DIRECT_REP_EQ_HD __host__ __device__
#else
#define DIRECT_REP_EQ_HD
#endif

namespace substrate::direct_adult_core {

// GitHub #1475 d.representation_equivalence_update.
//
// Representation/equivalence learning is its own update authority: a bounded
// resident updater may fit ONE candidate RecipeRevision to the boundary
// behavior a replayable condensation witness names, on frozen touched probes,
// and nothing else may change. The witness and the authoritative tissue are
// read-only here; residuals are structured non-credit comparison records; the
// candidate stays in representation probation and never installs, settles
// eligibility, or retires the sources. Equivalence may only nominate later
// condensation work after a disjoint held-out probe agrees exactly.

inline constexpr std::uint32_t kMaxRepresentationProbes = 8u;

enum class ResidentRepresentationRefusal : std::uint32_t {
  none = 0u,
  null_argument = 1u,
  invalid_witness = 2u,
  stale_witness_identity = 3u,
  changed_source_revision = 4u,
  missing_observation_identity = 5u,
  out_of_guard_probe = 6u,
  unordered_probe_identity = 7u,
  duplicate_probe_identity = 8u,
  zero_capacity = 9u,
  capacity_exceeded = 10u,
  candidate_alias_authoritative = 11u,
  candidate_not_probation = 12u,
  record_capacity_exceeded = 13u,
  changed_candidate_definition = 14u,
};

struct ResidentRepresentationEquivalenceStateV1 {
  std::uint64_t witness_identity;
  std::uint64_t source_revision_identities[direct_network::kResidentCondensationSourceCount];
  std::uint64_t candidate_definition_identity;
  std::uint64_t candidate_parent_revision_identity;
  // Frozen touched-probe schedule: identities strictly increasing, inputs
  // inside the witness guard. Update steps must replay exactly this order.
  std::uint64_t touched_identities[kMaxRepresentationProbes];
  std::int32_t touched_inputs_q16[kMaxRepresentationProbes];
  std::int32_t guard_min_q16, guard_max_q16;
  std::uint32_t probe_count, capacity;
  std::uint32_t update_steps;
  std::uint32_t representation_authority;
  std::uint32_t experiential_authority;
  std::uint32_t constructor_authority;
};
static_assert(std::is_trivial_v<ResidentRepresentationEquivalenceStateV1> &&
              std::is_standard_layout_v<ResidentRepresentationEquivalenceStateV1> &&
              std::has_unique_object_representations_v<ResidentRepresentationEquivalenceStateV1>);

// Structured signed comparison record. It deliberately carries no participant
// identity, ticket lineage, or credit field: a residual cannot mint a revision
// or enter settlement (github #1236 residual discipline).
struct ResidentRepresentationProbeRecordV1 {
  std::uint64_t observation_identity;
  std::int64_t residual_q16;
  std::int32_t input_q16;
  std::int32_t authoritative_q16;
  std::int32_t candidate_q16;
  std::uint32_t step_index;
};
static_assert(std::is_trivially_copyable_v<ResidentRepresentationProbeRecordV1>);

struct ResidentRepresentationReceiptV1 {
  std::uint64_t receipt_identity;
  std::int32_t candidate_parameter_q16;
  std::uint32_t update_steps;
  std::uint32_t held_out_probes;
  std::uint32_t held_out_maximum_error_q16;
  std::uint32_t representation_authority;
  std::uint32_t experiential_authority;
  std::uint32_t constructor_authority;
  std::uint32_t reserved;
};
static_assert(std::is_trivial_v<ResidentRepresentationReceiptV1> &&
              std::is_standard_layout_v<ResidentRepresentationReceiptV1> &&
              std::has_unique_object_representations_v<ResidentRepresentationReceiptV1>);

// Identity of the immutable part of a representation-probation candidate.
// Parameters are deliberately excluded: they are the sole update surface.
DIRECT_REP_EQ_HD inline std::uint64_t resident_representation_candidate_definition(
    const direct_network::ResidentRecipeDerivation& candidate) {
  using namespace direct_network;
  std::uint64_t identity = exact_history_fold_word(
      0x72657063616e6431ull, candidate.logical_recipe_id);
  identity = exact_history_fold_word(identity, candidate.revision_identity);
  identity = exact_history_fold_word(identity, candidate.parent_logical_recipe_id);
  identity = exact_history_fold_word(identity, candidate.parent_revision_identity);
  identity = exact_history_fold_word(identity, candidate.witness_identity);
  identity = exact_history_fold_word(identity, candidate.generation);
  identity = exact_history_fold_word(identity, candidate.recipe_cell);
  identity = exact_history_fold_word(identity, candidate.parent_recipe_cell);
  identity = exact_history_fold_word(identity, candidate.source_node);
  identity = exact_history_fold_word(identity, candidate.route_index);
  identity = exact_history_fold_word(identity, candidate.port_count);
  identity = exact_history_fold_word(identity, candidate.relation_count);
  identity = exact_history_fold_word(identity, candidate.parameter_count);
  identity = exact_history_fold_word(identity, candidate.territory_index);
  identity = exact_history_fold_word(identity, candidate.condensation_flags);
  for (std::uint32_t i = 0u; i < candidate.port_count; ++i) {
    identity = exact_history_fold_word(identity, candidate.ports[i].node);
    identity = exact_history_fold_word(
        identity, static_cast<std::uint32_t>(candidate.ports[i].domain));
    identity = exact_history_fold_word(
        identity, static_cast<std::uint32_t>(candidate.ports[i].direction));
    identity = exact_history_fold_word(identity, candidate.ports[i].arity);
  }
  for (std::uint32_t i = 0u; i < candidate.relation_count; ++i)
    identity = exact_history_fold_word(identity, candidate.relations[i]);
  return identity == 0u ? 1u : identity;
}

DIRECT_REP_EQ_HD inline bool resident_representation_candidate_valid(
    const direct_network::ResidentRecipeDerivation& candidate,
    std::uint64_t witness_identity) {
  using namespace direct_network;
  return candidate.logical_recipe_id != 0u && candidate.revision_identity != 0u &&
         candidate.parent_logical_recipe_id == candidate.logical_recipe_id &&
         candidate.parent_revision_identity != 0u &&
         candidate.parent_revision_identity != candidate.revision_identity &&
         candidate.witness_identity == witness_identity && candidate.generation != 0u &&
         candidate.recipe_cell == 0xffffffffu && candidate.port_count == 2u &&
         candidate.relation_count == 1u && candidate.parameter_count == 1u &&
         candidate.ports[0].direction == ResidentRecipePortDirection::input &&
         candidate.ports[1].direction == ResidentRecipePortDirection::output;
}

// Schedule validity with a discriminated reason: zero identities are missing
// observations, non-increasing identities are unordered, and only in-range
// failures are guard refusals.
DIRECT_REP_EQ_HD inline ResidentRepresentationRefusal
resident_representation_schedule_refusal(const std::uint64_t* identities,
                                         const std::int32_t* inputs,
                                         std::uint32_t count,
                                         std::int32_t guard_min_q16,
                                         std::int32_t guard_max_q16) {
  if (count == 0u || count > kMaxRepresentationProbes || identities == nullptr ||
      inputs == nullptr)
    return ResidentRepresentationRefusal::null_argument;
  for (std::uint32_t i = 0u; i < count; ++i) {
    if (identities[i] == 0u)
      return ResidentRepresentationRefusal::missing_observation_identity;
    if (i != 0u && identities[i - 1u] >= identities[i])
      return ResidentRepresentationRefusal::unordered_probe_identity;
    if (inputs[i] < guard_min_q16 || inputs[i] > guard_max_q16)
      return ResidentRepresentationRefusal::out_of_guard_probe;
  }
  return ResidentRepresentationRefusal::none;
}

// Freezes the witness binding, source revisions, guard, capacity, authority
// split, and ordered touched schedule before any update runs. The candidate
// must be a distinct revision carrying the representation-probation flag and
// must not alias either authoritative recipe cell.
DIRECT_REP_EQ_HD inline ResidentRepresentationRefusal admit_representation_equivalence(
    const direct_network::ResidentNetworkCondensationEvidence& witness,
    const direct_network::ResidentRecipeCell* cells, std::uint32_t cell_count,
    const direct_network::ResidentRecipeDerivation* derivations,
    std::uint32_t derivation_count,
    const direct_network::ResidentRecipeDerivation& candidate,
    const std::uint64_t* touched_identities, const std::int32_t* touched_inputs,
    std::uint32_t probe_count, std::uint32_t capacity,
    ResidentRepresentationEquivalenceStateV1* out) {
  using namespace direct_network;
  if (out == nullptr || cells == nullptr || derivations == nullptr)
    return ResidentRepresentationRefusal::null_argument;
  // Validate into a local stage first: a refused admission must not touch
  // any previously frozen state the caller may still hold.
  const ResidentRepresentationEquivalenceStateV1 cleared{};
  ResidentRepresentationEquivalenceStateV1 staged = cleared;
  if (capacity == 0u)
    return ResidentRepresentationRefusal::zero_capacity;
  if (probe_count == 0u || probe_count > capacity ||
      probe_count > kMaxRepresentationProbes)
    return ResidentRepresentationRefusal::capacity_exceeded;
  const ResidentRepresentationRefusal schedule = resident_representation_schedule_refusal(
      touched_identities, touched_inputs, probe_count,
      static_cast<std::int32_t>(witness.guard_min_q16),
      static_cast<std::int32_t>(witness.guard_max_q16));
  if (schedule != ResidentRepresentationRefusal::none) return schedule;
  if (witness.witness_identity == 0u ||
      witness.witness_identity != resident_network_condensation_witness(witness))
    return ResidentRepresentationRefusal::stale_witness_identity;
  std::uint64_t logical = 0u, revision = 0u;
  if (!replay_resident_condensation_witness(cells, cell_count, derivations,
                                            derivation_count, witness, &logical,
                                            &revision))
    return ResidentRepresentationRefusal::invalid_witness;
  if ((candidate.condensation_flags &
       kResidentDerivationRepresentationProbation) == 0u)
    return ResidentRepresentationRefusal::candidate_not_probation;
  for (std::uint32_t i = 0u; i < kResidentCondensationSourceCount; ++i) {
    if (candidate.revision_identity == witness.sources[i].revision_identity ||
        candidate.logical_recipe_id == witness.sources[i].logical_recipe_id)
      return ResidentRepresentationRefusal::candidate_alias_authoritative;
    if (candidate.revision_identity != 0u &&
        candidate.recipe_cell < cell_count &&
        cells[candidate.recipe_cell].logical_recipe_id ==
            candidate.logical_recipe_id)
      return ResidentRepresentationRefusal::candidate_alias_authoritative;
  }
  if (!resident_representation_candidate_valid(candidate, witness.witness_identity))
    return ResidentRepresentationRefusal::changed_candidate_definition;
  staged.witness_identity = witness.witness_identity;
  staged.candidate_definition_identity =
      resident_representation_candidate_definition(candidate);
  staged.candidate_parent_revision_identity = candidate.parent_revision_identity;
  for (std::uint32_t i = 0u; i < kResidentCondensationSourceCount; ++i)
    staged.source_revision_identities[i] = witness.sources[i].revision_identity;
  for (std::uint32_t i = 0u; i < probe_count; ++i) {
    staged.touched_identities[i] = touched_identities[i];
    staged.touched_inputs_q16[i] = touched_inputs[i];
  }
  staged.guard_min_q16 = static_cast<std::int32_t>(witness.guard_min_q16);
  staged.guard_max_q16 = static_cast<std::int32_t>(witness.guard_max_q16);
  staged.probe_count = probe_count;
  staged.capacity = capacity;
  staged.update_steps = 0u;
  staged.representation_authority = 1u;
  staged.experiential_authority = 0u;
  staged.constructor_authority = 0u;
  *out = staged;
  return ResidentRepresentationRefusal::none;
}

// One authoritative boundary evaluation against LIVE resident tissue through
// the witnessed composition. Returns false when the witness no longer matches
// the tissue (stale identity or drifted source parameters).
DIRECT_REP_EQ_HD inline bool authoritative_boundary_q16(
    const direct_network::ResidentNetworkCondensationEvidence& witness,
    const direct_network::ResidentRecipeCell* cells, std::uint32_t cell_count,
    const direct_network::ResidentRecipeDerivation* derivations,
    std::uint32_t derivation_count, std::int32_t input_q16,
    std::int32_t* output_q16) {
  using namespace direct_network;
  if (output_q16 == nullptr || witness.witness_identity == 0u ||
      witness.witness_identity != resident_network_condensation_witness(witness))
    return false;
  for (std::uint32_t i = 0u; i < kResidentCondensationSourceCount; ++i) {
    const auto& source = witness.sources[i];
    if (source.recipe_cell >= cell_count ||
        source.recipe_cell >= derivation_count ||
        cells[source.recipe_cell].logical_recipe_id != source.logical_recipe_id ||
        cells[source.recipe_cell].revision_identity != source.revision_identity ||
        source.parameter_q16 != derivations[source.recipe_cell].parameters_q16[0])
      return false;
  }
  std::int32_t intermediate = 0;
  return evaluate_resident_recipe_boundary_q16(
             witness.sources[0].relation, witness.sources[0].parameter_q16,
             input_q16, &intermediate) &&
         evaluate_resident_recipe_boundary_q16(
             witness.sources[1].relation, witness.sources[1].parameter_q16,
             intermediate, output_q16);
}

// Deterministic half-step toward zero residual, rounding away from zero so
// convergence is monotone in |theta - S| and identical across replays.
DIRECT_REP_EQ_HD inline std::int32_t representation_half_correction(
    std::int64_t residual_q16) {
  if (residual_q16 > 0)
    return static_cast<std::int32_t>((residual_q16 + 1) / 2);
  if (residual_q16 < 0)
    return -static_cast<std::int32_t>((-residual_q16 + 1) / 2);
  return 0;
}

// One resident update step over the FROZEN touched schedule. Emits one signed
// structured record per probe and updates ONLY the candidate parameter. Any
// schedule/witness/source deviation refuses before the first write.
DIRECT_REP_EQ_HD inline ResidentRepresentationRefusal
representation_equivalence_update_step(
    ResidentRepresentationEquivalenceStateV1* frozen,
    const direct_network::ResidentNetworkCondensationEvidence& witness,
    const direct_network::ResidentRecipeCell* cells, std::uint32_t cell_count,
    const direct_network::ResidentRecipeDerivation* derivations,
    std::uint32_t derivation_count,
    const std::uint64_t* presented_identities,
    direct_network::ResidentRecipeDerivation& candidate,
    ResidentRepresentationProbeRecordV1* records, std::uint32_t record_capacity,
    std::uint32_t step_index) {
  using namespace direct_network;
  if (frozen == nullptr || cells == nullptr || derivations == nullptr ||
      presented_identities == nullptr)
    return ResidentRepresentationRefusal::null_argument;
  if (frozen->probe_count == 0u || frozen->capacity == 0u)
    return ResidentRepresentationRefusal::zero_capacity;
  if (record_capacity < frozen->probe_count)
    return ResidentRepresentationRefusal::record_capacity_exceeded;
  if ((candidate.condensation_flags &
       kResidentDerivationRepresentationProbation) == 0u)
    return ResidentRepresentationRefusal::candidate_not_probation;
  if (!resident_representation_candidate_valid(candidate, frozen->witness_identity) ||
      candidate.parent_revision_identity !=
          frozen->candidate_parent_revision_identity ||
      resident_representation_candidate_definition(candidate) !=
          frozen->candidate_definition_identity)
    return ResidentRepresentationRefusal::changed_candidate_definition;
  if (witness.witness_identity != frozen->witness_identity ||
      witness.witness_identity !=
          resident_network_condensation_witness(witness))
    return ResidentRepresentationRefusal::stale_witness_identity;
  for (std::uint32_t i = 0u; i < kResidentCondensationSourceCount; ++i)
    if (witness.sources[i].revision_identity !=
        frozen->source_revision_identities[i])
      return ResidentRepresentationRefusal::changed_source_revision;
  for (std::uint32_t i = 0u; i < frozen->probe_count; ++i) {
    if (presented_identities[i] != frozen->touched_identities[i])
      return presented_identities[i] == 0u
                 ? ResidentRepresentationRefusal::missing_observation_identity
                 : ResidentRepresentationRefusal::unordered_probe_identity;
  }
  // Validate the whole frozen schedule against live tissue before writing.
  for (std::uint32_t i = 0u; i < frozen->probe_count; ++i) {
    std::int32_t authoritative = 0;
    if (!authoritative_boundary_q16(witness, cells, cell_count, derivations,
                                    derivation_count,
                                    frozen->touched_inputs_q16[i],
                                    &authoritative))
      return ResidentRepresentationRefusal::changed_source_revision;
  }
  std::int64_t residual_sum = 0;
  bool converged_already = true;
  for (std::uint32_t i = 0u; i < frozen->probe_count; ++i) {
    std::int32_t authoritative = 0;
    if (!authoritative_boundary_q16(witness, cells, cell_count, derivations,
                                    derivation_count,
                                    frozen->touched_inputs_q16[i],
                                    &authoritative))
      return ResidentRepresentationRefusal::changed_source_revision;
    std::int32_t produced = 0;
    if (!evaluate_resident_recipe_boundary_q16(candidate.relations[0],
                                               candidate.parameters_q16[0],
                                               frozen->touched_inputs_q16[i],
                                               &produced))
      return ResidentRepresentationRefusal::out_of_guard_probe;
    const std::int64_t residual =
        static_cast<std::int64_t>(produced) - authoritative;
    ResidentRepresentationProbeRecordV1 record{};
    record.observation_identity = frozen->touched_identities[i];
    record.residual_q16 = residual;
    record.input_q16 = frozen->touched_inputs_q16[i];
    record.authoritative_q16 = authoritative;
    record.candidate_q16 = produced;
    record.step_index = step_index;
    records[i] = record;
    residual_sum += residual;
    converged_already = converged_already && residual == 0;
  }
  if (!converged_already) {
    const std::int64_t mean_residual = residual_sum / frozen->probe_count;
    const std::int64_t correction = representation_half_correction(mean_residual);
    const std::int64_t updated =
        static_cast<std::int64_t>(candidate.parameters_q16[0]) - correction;
    if (updated < -0x80000000ll || updated > 0x7fffffffll)
      return ResidentRepresentationRefusal::out_of_guard_probe;
    candidate.parameters_q16[0] = static_cast<std::int32_t>(updated);
  }
  ++frozen->update_steps;
  return ResidentRepresentationRefusal::none;
}

// Disjoint held-out probe agreement: no mutation, exact equality required.
DIRECT_REP_EQ_HD inline ResidentRepresentationRefusal
representation_equivalence_held_out(
    const ResidentRepresentationEquivalenceStateV1& frozen,
    const direct_network::ResidentNetworkCondensationEvidence& witness,
    const direct_network::ResidentRecipeCell* cells, std::uint32_t cell_count,
    const direct_network::ResidentRecipeDerivation* derivations,
    std::uint32_t derivation_count,
    const std::uint64_t* held_out_identities, const std::int32_t* held_out_inputs,
    std::uint32_t held_out_count,
    const direct_network::ResidentRecipeDerivation& candidate,
    ResidentRepresentationProbeRecordV1* records, std::uint32_t record_capacity,
    std::uint32_t* out_maximum_error_q16) {
  using namespace direct_network;
  if (out_maximum_error_q16 == nullptr)
    return ResidentRepresentationRefusal::null_argument;
  *out_maximum_error_q16 = 0;
  if (held_out_count == 0u)
    return ResidentRepresentationRefusal::zero_capacity;
  if (held_out_count > kMaxRepresentationProbes ||
      record_capacity < held_out_count)
    return ResidentRepresentationRefusal::record_capacity_exceeded;
  const ResidentRepresentationRefusal held_schedule =
      resident_representation_schedule_refusal(
          held_out_identities, held_out_inputs, held_out_count,
          frozen.guard_min_q16, frozen.guard_max_q16);
  if (held_schedule != ResidentRepresentationRefusal::none) return held_schedule;
  for (std::uint32_t i = 0u; i < held_out_count; ++i)
    for (std::uint32_t j = 0u; j < frozen.probe_count; ++j)
      if (held_out_identities[i] == frozen.touched_identities[j])
        return ResidentRepresentationRefusal::duplicate_probe_identity;
  if ((candidate.condensation_flags &
       kResidentDerivationRepresentationProbation) == 0u)
    return ResidentRepresentationRefusal::candidate_not_probation;
  if (!resident_representation_candidate_valid(candidate, frozen.witness_identity) ||
      candidate.parent_revision_identity !=
          frozen.candidate_parent_revision_identity ||
      resident_representation_candidate_definition(candidate) !=
          frozen.candidate_definition_identity)
    return ResidentRepresentationRefusal::changed_candidate_definition;
  if (witness.witness_identity != frozen.witness_identity)
    return ResidentRepresentationRefusal::stale_witness_identity;
  std::uint32_t maximum_error = 0u;
  for (std::uint32_t i = 0u; i < held_out_count; ++i) {
    std::int32_t authoritative = 0;
    if (!authoritative_boundary_q16(witness, cells, cell_count, derivations,
                                    derivation_count, held_out_inputs[i],
                                    &authoritative))
      return ResidentRepresentationRefusal::changed_source_revision;
    std::int32_t produced = 0;
    if (!evaluate_resident_recipe_boundary_q16(candidate.relations[0],
                                               candidate.parameters_q16[0],
                                               held_out_inputs[i], &produced))
      return ResidentRepresentationRefusal::out_of_guard_probe;
    const std::int64_t error = static_cast<std::int64_t>(produced) -
                               authoritative;
    const std::uint64_t magnitude =
        error >= 0 ? static_cast<std::uint64_t>(error)
                   : static_cast<std::uint64_t>(-error);
    if (magnitude > 0x7fffffffull) return ResidentRepresentationRefusal::out_of_guard_probe;
    if (magnitude > maximum_error) maximum_error = static_cast<std::uint32_t>(magnitude);
    ResidentRepresentationProbeRecordV1 record{};
    record.observation_identity = held_out_identities[i];
    record.residual_q16 = error;
    record.input_q16 = held_out_inputs[i];
    record.authoritative_q16 = authoritative;
    record.candidate_q16 = produced;
    record.step_index = frozen.update_steps;
    records[i] = record;
  }
  *out_maximum_error_q16 = maximum_error;
  return maximum_error == 0u ? ResidentRepresentationRefusal::none
                             : ResidentRepresentationRefusal::invalid_witness;
}

// Equivalence receipt. Valid only after convergence on the frozen schedule
// AND exact disjoint held-out agreement, with the candidate still in
// probation. Carries the explicit authority split: representation yes,
// experiential/constructor never.
DIRECT_REP_EQ_HD inline bool representation_equivalence_receipt(
    const ResidentRepresentationEquivalenceStateV1& frozen,
    const direct_network::ResidentRecipeDerivation& candidate, std::uint32_t held_out_probes,
    std::uint32_t held_out_maximum_error_q16,
    ResidentRepresentationReceiptV1* out) {
  using namespace direct_network;
  if (out == nullptr) return false;
  *out = ResidentRepresentationReceiptV1{};
  if (held_out_probes == 0u || held_out_maximum_error_q16 != 0u)
    return false;
  if ((candidate.condensation_flags &
       kResidentDerivationRepresentationProbation) == 0u)
    return false;
  if (!resident_representation_candidate_valid(candidate, frozen.witness_identity) ||
      candidate.parent_revision_identity !=
          frozen.candidate_parent_revision_identity ||
      resident_representation_candidate_definition(candidate) !=
          frozen.candidate_definition_identity)
    return false;
  if (frozen.representation_authority != 1u ||
      frozen.experiential_authority != 0u ||
      frozen.constructor_authority != 0u)
    return false;
  out->candidate_parameter_q16 = candidate.parameters_q16[0];
  out->update_steps = frozen.update_steps;
  out->held_out_probes = held_out_probes;
  out->held_out_maximum_error_q16 = held_out_maximum_error_q16;
  out->representation_authority = 1u;
  out->experiential_authority = 0u;
  out->constructor_authority = 0u;
  std::uint64_t identity = exact_history_fold_word(0x7265706571756976ull,
                                                   frozen.witness_identity);
  identity = exact_history_fold_word(identity, frozen.probe_count);
  identity = exact_history_fold_word(identity, frozen.update_steps);
  identity = exact_history_fold_word(identity, held_out_probes);
  identity = exact_history_fold_word(identity, frozen.candidate_definition_identity);
  identity = exact_history_fold_word(identity, frozen.candidate_parent_revision_identity);
  for (std::uint32_t i = 0u; i < frozen.probe_count; ++i)
    identity = exact_history_fold_word(identity, frozen.touched_identities[i]);
  identity = exact_history_fold_word(
      identity, static_cast<std::uint64_t>(
                    static_cast<std::uint32_t>(candidate.parameters_q16[0])));
  out->receipt_identity = identity == 0u ? 1u : identity;
  return true;
}

}  // namespace substrate::direct_adult_core

#undef DIRECT_REP_EQ_HD
#endif  // HARDWARE_NATIVE_DIRECT_ADULT_REPRESENTATION_EQUIVALENCE_CUH
