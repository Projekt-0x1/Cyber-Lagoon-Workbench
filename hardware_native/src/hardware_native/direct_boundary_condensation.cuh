#ifndef HARDWARE_NATIVE_DIRECT_BOUNDARY_CONDENSATION_CUH
#define HARDWARE_NATIVE_DIRECT_BOUNDARY_CONDENSATION_CUH

#include <cstdint>
#include <type_traits>

#include "hardware_native/direct_adult_core.cuh"
#include "hardware_native/direct_adult_actual_frontier_condensation.cuh"
#include "hardware_native/direct_adult_runtime_frontiers.cuh"

#include "hardware_native/direct_mathematical_relation_algebra.cuh"

namespace substrate::direct_adult_core {

inline constexpr std::uint16_t kResidentBoundaryCondensationVersion = 1u;
inline constexpr std::uint16_t kResidentBoundaryCondensationProbeCount = 3u;

struct ResidentBoundaryCondensationProbe {
  std::uint64_t participation_identity;
  std::int32_t input_q16, observed_output_q16;
};

struct ResidentBoundaryCondensationWitness {
  ResidentNetworkBoundaryRelation source_boundary;
  ResidentBoundaryCondensationProbe probes[kResidentBoundaryCondensationProbeCount];
  std::uint64_t proof_identity, witness_identity;
  std::uint64_t refinement_logical_recipe_id, refinement_revision_identity;
  std::uint32_t refinement_recipe_cell;
  std::uint32_t guard_min_q16, guard_max_q16;
  std::uint16_t probe_count, eliminated_variable_count, version, reserved;
};

static_assert(std::is_trivial_v<ResidentBoundaryCondensationProbe> &&
              std::is_standard_layout_v<ResidentBoundaryCondensationProbe>);
static_assert(std::is_trivial_v<ResidentBoundaryCondensationWitness> &&
              std::is_standard_layout_v<ResidentBoundaryCondensationWitness>);

DIRECT_ADULT_HD inline std::uint64_t resident_boundary_behavior_identity(
    const ResidentNetworkBoundaryRelation& boundary) {
  using direct_network::exact_history_fold_word;
  std::uint64_t identity = exact_history_fold_word(
      0x626f756e64626568ull, boundary.relation);
  identity = exact_history_fold_word(
      identity, static_cast<std::uint32_t>(boundary.composed_parameter_q16));
  identity = exact_history_fold_word(
      identity, boundary.input_boundary.variable_identity);
  identity = exact_history_fold_word(
      identity, boundary.output_boundary.variable_identity);
  identity = exact_history_fold_word(
      identity, static_cast<std::uint32_t>(boundary.input_boundary.domain));
  identity = exact_history_fold_word(
      identity, static_cast<std::uint32_t>(boundary.output_boundary.domain));
  identity = exact_history_fold_word(identity, boundary.input_boundary.arity);
  identity = exact_history_fold_word(identity, boundary.output_boundary.arity);
  return identity == 0u ? 1u : identity;
}

DIRECT_ADULT_HD inline std::uint64_t resident_boundary_condensation_proof_identity(
    const ResidentBoundaryCondensationWitness& witness) {
  using direct_network::exact_history_fold_word;
  std::uint64_t identity = exact_history_fold_word(
      0x424e70726f6f6631ull, witness.source_boundary.behavior_identity);
  identity = exact_history_fold_word(identity, witness.guard_min_q16);
  identity = exact_history_fold_word(identity, witness.guard_max_q16);
  identity = exact_history_fold_word(identity, witness.probe_count);
  for (std::uint32_t i = 0u; i < witness.probe_count &&
       i < kResidentBoundaryCondensationProbeCount; ++i) {
    identity = exact_history_fold_word(
        identity, witness.probes[i].participation_identity);
    identity = exact_history_fold_word(
        identity, static_cast<std::uint32_t>(witness.probes[i].input_q16));
    identity = exact_history_fold_word(
        identity, static_cast<std::uint32_t>(witness.probes[i].observed_output_q16));
  }
  return identity == 0u ? 1u : identity;
}

DIRECT_ADULT_HD inline std::uint64_t resident_boundary_condensation_witness_identity(
    const ResidentBoundaryCondensationWitness& witness) {
  using direct_network::exact_history_fold_word;
  std::uint64_t identity = exact_history_fold_word(
      0x424e7769746e6573ull, witness.version);
  identity = exact_history_fold_word(
      identity, witness.source_boundary.witness_identity);
  identity = exact_history_fold_word(identity, witness.proof_identity);
  identity = exact_history_fold_word(
      identity, witness.refinement_logical_recipe_id);
  identity = exact_history_fold_word(
      identity, witness.refinement_revision_identity);
  identity = exact_history_fold_word(identity, witness.refinement_recipe_cell);
  identity = exact_history_fold_word(identity, witness.eliminated_variable_count);
  return identity == 0u ? 1u : identity;
}

DIRECT_ADULT_HD inline bool resident_boundary_condensation_witness_complete(
    const ResidentBoundaryCondensationWitness& witness) {
  if (witness.version != kResidentBoundaryCondensationVersion ||
      witness.probe_count != kResidentBoundaryCondensationProbeCount ||
      witness.source_boundary.member_count < 3u ||
      witness.source_boundary.member_count > kResidentNetworkBoundaryMaxMembers ||
      witness.eliminated_variable_count + 1u !=
          witness.source_boundary.member_count ||
      witness.guard_min_q16 >= witness.guard_max_q16 ||
      witness.refinement_logical_recipe_id == 0u ||
      witness.refinement_revision_identity == 0u ||
      witness.source_boundary.witness_identity == 0u ||
      witness.source_boundary.witness_identity !=
          resident_network_boundary_witness(witness.source_boundary) ||
      witness.source_boundary.behavior_identity !=
          resident_boundary_behavior_identity(witness.source_boundary) ||
      witness.source_boundary.members[0].input_variable_identity !=
          witness.source_boundary.input_boundary.variable_identity ||
      witness.source_boundary.members[witness.source_boundary.member_count - 1u]
              .output_variable_identity !=
          witness.source_boundary.output_boundary.variable_identity ||
      witness.guard_max_q16 > 0x7fffffffu ||
      witness.probes[0].input_q16 !=
          static_cast<std::int32_t>(witness.guard_min_q16) ||
      witness.probes[1].input_q16 != static_cast<std::int32_t>(
          witness.guard_min_q16 +
          (witness.guard_max_q16 - witness.guard_min_q16) / 2u) ||
      witness.probes[2].input_q16 !=
          static_cast<std::int32_t>(witness.guard_max_q16))
    return false;
  for (std::uint32_t i = 0u;
       i + 1u < witness.source_boundary.member_count; ++i)
    if (witness.source_boundary.members[i].output_variable_identity !=
        witness.source_boundary.members[i + 1u].input_variable_identity)
      return false;
  for (std::uint32_t i = 0u; i < witness.probe_count; ++i) {
    if (witness.probes[i].participation_identity == 0u) return false;
    for (std::uint32_t prior = 0u; prior < i; ++prior)
      if (witness.probes[prior].participation_identity ==
          witness.probes[i].participation_identity)
        return false;
    std::int32_t expected = 0;
    if (!evaluate_resident_network_boundary_relation_q16(
            witness.source_boundary, witness.probes[i].input_q16, &expected) ||
        expected != witness.probes[i].observed_output_q16)
      return false;
  }
  return witness.proof_identity ==
             resident_boundary_condensation_proof_identity(witness) &&
      witness.witness_identity ==
             resident_boundary_condensation_witness_identity(witness);
}

DIRECT_ADULT_HD inline bool freeze_resident_boundary_condensation_witness(
    direct_network::ResidentRecipeDerivation* derivation,
    const ResidentBoundaryCondensationWitness& witness,
    const direct_network::ResidentRecipeCell* source_recipes = nullptr,
    const direct_network::ResidentRecipeDerivation* source_derivations = nullptr,
    std::uint32_t source_count = 0u) {
  using namespace direct_network;
  if (derivation == nullptr ||
      !resident_boundary_condensation_witness_complete(witness) ||
      witness.source_boundary.member_count >
          kResidentBoundaryCondensationMaxRoots)
    return false;
  derivation->witness_identity = witness.witness_identity;
  derivation->boundary_condensation_proof_identity = witness.proof_identity;
  derivation->boundary_condensation_refinement_logical_recipe_id =
      witness.refinement_logical_recipe_id;
  derivation->boundary_condensation_refinement_revision_identity =
      witness.refinement_revision_identity;
  derivation->boundary_condensation_refinement_recipe_cell =
      witness.refinement_recipe_cell;
  derivation->boundary_condensation_guard_min_q16 = witness.guard_min_q16;
  derivation->boundary_condensation_guard_max_q16 = witness.guard_max_q16;
  derivation->boundary_condensation_root_count =
      witness.source_boundary.member_count;
  derivation->boundary_condensation_probe_count = witness.probe_count;
  derivation->boundary_condensation_eliminated_variable_count =
      witness.eliminated_variable_count;
  derivation->boundary_condensation_version = witness.version;
  for (std::uint32_t i = 0u; i < witness.source_boundary.member_count; ++i) {
    derivation->boundary_condensation_root_logical_recipe_ids[i] =
        witness.source_boundary.members[i].logical_recipe_id;
    derivation->boundary_condensation_root_revision_identities[i] =
        witness.source_boundary.members[i].revision_identity;
  }
  if (source_recipes != nullptr || source_derivations != nullptr ||
      source_count != 0u) {
    if (source_recipes == nullptr || source_derivations == nullptr ||
        source_count != witness.source_boundary.member_count)
      return false;
    for (std::uint32_t member = 0u; member < source_count; ++member) {
      const auto& root = witness.source_boundary.members[member];
      std::uint32_t match = source_count;
      for (std::uint32_t source = 0u; source < source_count; ++source)
        if (source_recipes[source].logical_recipe_id == root.logical_recipe_id &&
            source_recipes[source].revision_identity == root.revision_identity &&
            source_derivations[source].logical_recipe_id == root.logical_recipe_id &&
            source_derivations[source].revision_identity == root.revision_identity) {
          if (match != source_count) return false;
          match = source;
        }
      if (match == source_count) return false;
      const auto& source = source_derivations[match];
      if (source.port_count < 2u || source.relation_count == 0u ||
          source.parameter_count == 0u)
        return false;
      auto& frozen = derivation->boundary_condensation_sources[member];
      frozen.logical_recipe_id = root.logical_recipe_id;
      frozen.revision_identity = root.revision_identity;
      frozen.generation = source.generation;
      frozen.route_incarnation = source.route_incarnations[0];
      frozen.source_identity = root.occurrence_identity;
      frozen.recipe_cell = source.recipe_cell;
      frozen.relation = source.relations[0];
      frozen.route_index = source.route_index;
      frozen.parameter_q16 = source.parameters_q16[0];
      frozen.input_port = source.ports[0];
      frozen.output_port = source.ports[1];
    }
  }
  for (std::uint32_t i = 0u; i < witness.probe_count; ++i) {
    derivation->boundary_condensation_probe_participation_identities[i] =
        witness.probes[i].participation_identity;
    derivation->boundary_condensation_probe_inputs_q16[i] =
        witness.probes[i].input_q16;
    derivation->boundary_condensation_probe_outputs_q16[i] =
        witness.probes[i].observed_output_q16;
  }
  return true;
}

DIRECT_ADULT_HD inline bool resident_boundary_condensation_metadata_complete(
    const direct_network::ResidentRecipeDerivation& derivation) {
  using namespace direct_network;
  if (derivation.witness_identity == 0u ||
      derivation.boundary_condensation_proof_identity == 0u ||
      derivation.boundary_condensation_refinement_logical_recipe_id == 0u ||
      derivation.boundary_condensation_refinement_revision_identity == 0u ||
      derivation.boundary_condensation_root_count < 3u ||
      derivation.boundary_condensation_root_count >
          kResidentBoundaryCondensationMaxRoots ||
      derivation.boundary_condensation_probe_count !=
          kResidentBoundaryCondensationProbeCount ||
      derivation.boundary_condensation_eliminated_variable_count + 1u !=
          derivation.boundary_condensation_root_count ||
      derivation.boundary_condensation_version !=
          kResidentBoundaryCondensationVersion ||
      derivation.boundary_condensation_guard_min_q16 >=
          derivation.boundary_condensation_guard_max_q16)
    return false;
  for (std::uint32_t i = 0u;
       i < derivation.boundary_condensation_root_count; ++i)
    if (derivation.boundary_condensation_root_logical_recipe_ids[i] == 0u ||
        derivation.boundary_condensation_root_revision_identities[i] == 0u)
      return false;
  for (std::uint32_t i = 0u;
       i < derivation.boundary_condensation_probe_count; ++i)
    if (derivation.boundary_condensation_probe_participation_identities[i] == 0u)
      return false;
  return true;
}

DIRECT_ADULT_HD inline direct_network::ResidentRecipeReceptorState
resident_level_c_boundary_child_receptor(
    const direct_network::ResidentRecipeCell* source_recipes,
    std::uint32_t source_count,
    const ResidentBoundaryCondensationWitness& witness) {
  using namespace direct_network;
  ResidentRecipeReceptorState child{};
  if (source_recipes == nullptr || source_count < 3u ||
      source_count != witness.source_boundary.member_count ||
      !resident_boundary_condensation_witness_complete(witness))
    return child;
  std::int64_t activation = 0, plasticity = 0;
  std::uint64_t identity = 0x6c6576656c63626eull;
  for (std::uint32_t i = 0u; i < source_count; ++i) {
    activation += source_recipes[i].receptor_state.activation_q16;
    plasticity += source_recipes[i].receptor_state.plasticity_q16;
    identity = exact_history_fold_word(
        identity, source_recipes[i].receptor_state.causal_identity);
    identity = exact_history_fold_word(
        identity, source_recipes[i].receptor_state.revision_identity);
  }
  child.activation_q16 = resident_level_c_saturating_q16(
      activation / source_count + witness.source_boundary.composed_parameter_q16);
  child.plasticity_q16 = resident_level_c_saturating_q16(
      plasticity / source_count);
  identity = exact_history_fold_word(
      identity, witness.source_boundary.behavior_identity);
  identity = exact_history_fold_word(identity, witness.witness_identity);
  identity = exact_history_fold_word(
      identity, static_cast<std::uint32_t>(child.activation_q16));
  identity = exact_history_fold_word(
      identity, static_cast<std::uint32_t>(child.plasticity_q16));
  child.causal_identity = identity == 0u ? 1u : identity;
  return child;
}

DIRECT_ADULT_HD inline bool observe_resident_boundary_condensation(
    const ResidentNetworkBoundaryRelation& boundary,
    const direct_network::ResidentRecipeCell* source_recipes,
    const direct_network::ResidentRecipeDerivation* source_derivations,
    const ResidentRecipeOccurrence* source_occurrences,
    std::uint32_t source_count,
    const ResidentOccurrenceCoupling* source_couplings,
    std::uint32_t coupling_count,
    const ResidentRelationalNetworkClosure& closure,
    const ResidentBoundaryCondensationProbe (&probes)
        [kResidentBoundaryCondensationProbeCount],
    std::uint32_t guard_min_q16, std::uint32_t guard_max_q16,
    ResidentBoundaryCondensationWitness* out) {
  if (out == nullptr || source_recipes == nullptr ||
      source_derivations == nullptr || source_occurrences == nullptr ||
      source_couplings == nullptr || source_count != boundary.member_count ||
      coupling_count + 1u != source_count ||
      boundary.member_count < 3u ||
      boundary.member_count > kResidentNetworkBoundaryMaxMembers ||
      boundary.witness_identity != resident_network_boundary_witness(boundary) ||
      guard_min_q16 >= guard_max_q16)
    return false;
  ResidentNetworkBoundaryRelation rebuilt{};
  if (!compose_resident_network_boundary_relation(
          source_recipes, source_derivations, source_occurrences, source_count,
          source_couplings, coupling_count, closure, &rebuilt))
    return false;
  const auto* expected = reinterpret_cast<const unsigned char*>(&boundary);
  const auto* observed = reinterpret_cast<const unsigned char*>(&rebuilt);
  for (std::uint32_t i = 0u; i < sizeof(boundary); ++i)
    if (expected[i] != observed[i]) return false;
  ResidentBoundaryCondensationWitness candidate{};
  candidate.source_boundary = boundary;
  candidate.guard_min_q16 = guard_min_q16;
  candidate.guard_max_q16 = guard_max_q16;
  candidate.probe_count = kResidentBoundaryCondensationProbeCount;
  candidate.eliminated_variable_count = boundary.member_count - 1u;
  candidate.version = kResidentBoundaryCondensationVersion;
  candidate.refinement_logical_recipe_id = boundary.members[0].logical_recipe_id;
  candidate.refinement_revision_identity = boundary.members[0].revision_identity;
  for (std::uint32_t member = 0u; member < source_count; ++member) {
    std::uint32_t matches = 0u;
    for (std::uint32_t source = 0u; source < source_count; ++source) {
      if (source_recipes[source].logical_recipe_id !=
              boundary.members[member].logical_recipe_id ||
          source_recipes[source].revision_identity !=
              boundary.members[member].revision_identity ||
          source_derivations[source].logical_recipe_id !=
              boundary.members[member].logical_recipe_id ||
          source_derivations[source].revision_identity !=
              boundary.members[member].revision_identity)
        continue;
      ++matches;
      if (member == 0u)
        candidate.refinement_recipe_cell =
            source_derivations[source].recipe_cell;
    }
    if (matches != 1u) return false;
  }
  for (std::uint32_t i = 0u; i < candidate.probe_count; ++i)
    candidate.probes[i] = probes[i];
  candidate.proof_identity =
      resident_boundary_condensation_proof_identity(candidate);
  candidate.witness_identity =
      resident_boundary_condensation_witness_identity(candidate);
  if (!resident_boundary_condensation_witness_complete(candidate)) return false;
  *out = candidate;
  return true;
}

DIRECT_ADULT_HD inline bool resident_boundary_condensation_sources_current(
    const ResidentBoundaryCondensationWitness& witness,
    const direct_network::ResidentRecipeCell* recipes,
    const direct_network::ResidentRecipeDerivation* derivations,
    std::uint32_t source_count) {
  if (recipes == nullptr || derivations == nullptr ||
      source_count != witness.source_boundary.member_count ||
      !resident_boundary_condensation_witness_complete(witness))
    return false;
  std::uint32_t refinement_matches = 0u;
  for (std::uint32_t member = 0u; member < source_count; ++member) {
    const auto& root = witness.source_boundary.members[member];
    std::uint32_t matches = 0u;
    for (std::uint32_t source = 0u; source < source_count; ++source) {
      if (recipes[source].logical_recipe_id != root.logical_recipe_id ||
          recipes[source].revision_identity != root.revision_identity ||
          derivations[source].logical_recipe_id != root.logical_recipe_id ||
          derivations[source].revision_identity != root.revision_identity)
        continue;
      ++matches;
      refinement_matches +=
          root.logical_recipe_id == witness.refinement_logical_recipe_id &&
          root.revision_identity == witness.refinement_revision_identity &&
          derivations[source].recipe_cell == witness.refinement_recipe_cell;
    }
    if (matches != 1u) return false;
  }
  return refinement_matches == 1u;
}

DIRECT_ADULT_HD inline std::uint64_t resident_boundary_recipe_logical_identity(
    const ResidentNetworkBoundaryRelation& boundary) {
  using direct_network::exact_history_fold_word;
  std::uint64_t identity = exact_history_fold_word(
      0x424e726563697065ull, boundary.relation);
  identity = exact_history_fold_word(
      identity, static_cast<std::uint32_t>(boundary.composed_parameter_q16));
  identity = exact_history_fold_word(
      identity, static_cast<std::uint32_t>(boundary.input_boundary.domain));
  identity = exact_history_fold_word(
      identity, static_cast<std::uint32_t>(boundary.output_boundary.domain));
  identity = exact_history_fold_word(identity, boundary.input_boundary.arity);
  identity = exact_history_fold_word(identity, boundary.output_boundary.arity);
  return identity == 0u ? 1u : identity;
}

DIRECT_ADULT_HD inline bool materialize_resident_boundary_recipe(
    direct_network::ResidentRecipeCell* cells, std::uint32_t* cell_count,
    std::uint32_t cell_capacity,
    direct_network::ResidentRecipeDerivation* derivations,
    direct_network::ResidentPostbirthConstructorState* state,
    const direct_network::ResidentRecipeCell* source_recipes,
    const direct_network::ResidentRecipeDerivation* source_derivations,
    std::uint32_t source_count,
    const ResidentBoundaryCondensationWitness& witness,
    std::uint16_t territory_index) {
  using namespace direct_network;
  if (cells == nullptr || cell_count == nullptr || derivations == nullptr ||
      state == nullptr || !resident_boundary_condensation_sources_current(
          witness, source_recipes, source_derivations, source_count) ||
      *cell_count >= cell_capacity || *cell_count >= state->recipe_cell_capacity ||
      state->derivation_count >= state->derivation_capacity ||
      state->ports_used + 2u > state->port_capacity ||
      state->relations_used + 1u > state->relation_capacity ||
      state->parameters_used + 1u > state->parameter_capacity)
    return false;
  const auto& boundary = witness.source_boundary;
  std::uint32_t input_source = source_count, output_source = source_count;
  std::uint64_t generation = 0u;
  std::int32_t support_q16 = 0x7fffffff;
  for (std::uint32_t i = 0u; i < source_count; ++i) {
    if (source_derivations[i].generation > generation)
      generation = source_derivations[i].generation;
    if (source_recipes[i].support_q16 < support_q16)
      support_q16 = source_recipes[i].support_q16;
    if (source_recipes[i].logical_recipe_id ==
        boundary.members[0].logical_recipe_id)
      input_source = i;
    if (source_recipes[i].logical_recipe_id ==
        boundary.members[source_count - 1u].logical_recipe_id)
      output_source = i;
  }
  if (input_source == source_count || output_source == source_count ||
      generation == ~std::uint64_t{0})
    return false;
  const std::uint64_t logical_id =
      resident_boundary_recipe_logical_identity(boundary);
  for (std::uint32_t i = 0u; i < *cell_count; ++i)
    if (cells[i].logical_recipe_id == logical_id) return false;
  const std::uint32_t new_cell = *cell_count;
  ResidentRecipeCell recipe{};
  recipe.logical_recipe_id = logical_id;
  recipe.support_q16 = support_q16;
  recipe.revision = 1u;
  recipe.rule_index = source_recipes[input_source].rule_index;
  recipe.revision_identity = resident_recipe_revision_identity(
      logical_id, 0u, recipe.revision, witness.witness_identity,
      recipe.support_q16, recipe.credit_q16);
  if (!initialize_resident_recipe_update_ir(&recipe)) return false;
  recipe.receptor_state = resident_level_c_boundary_child_receptor(
      source_recipes, source_count, witness);
  if (recipe.receptor_state.causal_identity == 0u) return false;
  recipe.receptor_state.revision_identity = recipe.revision_identity;
  ResidentRecipeDerivation derivation{};
  derivation.logical_recipe_id = logical_id;
  derivation.revision_identity = recipe.revision_identity;
  derivation.parent_logical_recipe_id =
      source_recipes[input_source].logical_recipe_id;
  derivation.parent_revision_identity =
      source_recipes[input_source].revision_identity;
  derivation.witness_identity = witness.witness_identity;
  derivation.generation = generation + 1u;
  derivation.recipe_cell = new_cell;
  derivation.parent_recipe_cell = source_derivations[input_source].recipe_cell;
  derivation.territory_index = territory_index;
  derivation.port_count = 2u;
  derivation.relation_count = 1u;
  derivation.parameter_count = 1u;
  derivation.ports[0] = source_derivations[input_source].ports[0];
  derivation.ports[1] = source_derivations[output_source].ports[1];
  derivation.relations[0] = boundary.relation;
  derivation.parameters_q16[0] = boundary.composed_parameter_q16;
  derivation.condensation_flags =
      kResidentDerivationBoundaryCondensedNetwork;
  if (!freeze_resident_boundary_condensation_witness(
          &derivation, witness, source_recipes, source_derivations,
          source_count))
    return false;
  cells[new_cell] = recipe;
  derivations[state->derivation_count] = derivation;
  ++*cell_count;
  ++state->derivation_count;
  state->ports_used += 2u;
  ++state->relations_used;
  ++state->parameters_used;
  ++state->condensed;
  if (derivation.generation > state->highest_derivation_rank)
    state->highest_derivation_rank = derivation.generation;
  return true;
}

DIRECT_ADULT_HD inline bool replay_resident_boundary_recipe(
    const ResidentBoundaryCondensationWitness& witness,
    const direct_network::ResidentRecipeCell* source_recipes,
    const direct_network::ResidentRecipeDerivation* source_derivations,
    std::uint32_t source_count,
    const direct_network::ResidentRecipeCell& recipe,
    const direct_network::ResidentRecipeDerivation& derivation) {
  using namespace direct_network;
  if (!resident_boundary_condensation_sources_current(
          witness, source_recipes, source_derivations, source_count) ||
      recipe.logical_recipe_id !=
          resident_boundary_recipe_logical_identity(witness.source_boundary) ||
      derivation.logical_recipe_id != recipe.logical_recipe_id ||
      derivation.revision_identity != recipe.revision_identity ||
      derivation.witness_identity != witness.witness_identity ||
      derivation.port_count != 2u || derivation.relation_count != 1u ||
      derivation.parameter_count != 1u ||
      derivation.relations[0] != witness.source_boundary.relation ||
      derivation.parameters_q16[0] !=
          witness.source_boundary.composed_parameter_q16)
    return false;
  const std::uint64_t expected_revision = resident_recipe_revision_identity(
      recipe.logical_recipe_id, 0u, 1u, witness.witness_identity,
      recipe.support_q16, 0);
  return recipe.revision == 1u && recipe.credit_q16 == 0 &&
      recipe.revision_identity == expected_revision;
}

DIRECT_ADULT_HD inline bool rematerialize_resident_boundary_relation(
    const ResidentBoundaryCondensationWitness& witness,
    ResidentNetworkBoundaryRelation* out) {
  if (out == nullptr ||
      !resident_boundary_condensation_witness_complete(witness))
    return false;
  *out = witness.source_boundary;
  return true;
}

// Condense only the complete bounded actual frontier.  The frontier supplies
// the members and physical participation roots; the runtime derives the
// boundary, probe domain, proof values, and refinement root on device.
__device__ inline bool condense_resident_actual_frontier_boundary_network(
    direct_network::DirectBrain brain, ResidentActualFrontier* frontier) {
  using namespace direct_network;
  using substrate::direct_adult::DirectResourcePoolKind;
  if (frontier == nullptr || brain.development == nullptr ||
      brain.postbirth_constructor == nullptr ||
      brain.postbirth_derivations == nullptr || brain.recipe_cells == nullptr ||
      brain.resource_ecology == nullptr || brain.nodes == nullptr)
    return false;
  ++frontier->condensation_nominations;
  ResidentRecipeCell recipes[kResidentRelationalNetworkMaxOccurrences]{};
  ResidentRecipeDerivation derivations[kResidentRelationalNetworkMaxOccurrences]{};
  ResidentRecipeOccurrence occurrences[kResidentRelationalNetworkMaxOccurrences]{};
  ResidentOccurrenceCoupling couplings[kResidentRelationalNetworkMaxCouplings]{};
  std::uint32_t occurrence_count = 0u, coupling_count = 0u;
  ResidentRelationalNetworkClosure closure{};
  ResidentNetworkBoundaryRelation boundary{};
  if (!collect_resident_actual_frontier_relational_network(
          brain, *frontier, recipes, derivations, occurrences, couplings,
          &occurrence_count, &coupling_count) ||
      !bind_resident_relational_network_closure(
          recipes, derivations, occurrences, occurrence_count, couplings,
          coupling_count, &closure) ||
      !compose_resident_network_boundary_relation(
          recipes, derivations, occurrences, occurrence_count, couplings,
          coupling_count, closure, &boundary) ||
      occurrence_count > kResidentBoundaryCondensationMaxRoots)
    return false;

  std::int32_t guard_min = 0x7fffffff, guard_max = 0;
  for (std::uint32_t i = 0u; i < occurrence_count; ++i) {
    if (occurrences[i].activation_q16 < 0) return false;
    if (occurrences[i].activation_q16 < guard_min)
      guard_min = occurrences[i].activation_q16;
    if (occurrences[i].activation_q16 > guard_max)
      guard_max = occurrences[i].activation_q16;
  }
  if (guard_min >= guard_max) return false;
  ResidentBoundaryCondensationProbe probes[kResidentBoundaryCondensationProbeCount]{};
  const std::int32_t values[kResidentBoundaryCondensationProbeCount] = {
      guard_min, guard_min + (guard_max - guard_min) / 2, guard_max};
  for (std::uint32_t i = 0u; i < kResidentBoundaryCondensationProbeCount; ++i) {
    probes[i].input_q16 = values[i];
    probes[i].participation_identity = exact_history_fold_word(
        occurrences[i % occurrence_count].participation_identity,
        static_cast<std::uint32_t>(values[i]));
    if (probes[i].participation_identity == 0u ||
        !evaluate_resident_network_boundary_relation_q16(
            boundary, values[i], &probes[i].observed_output_q16))
      return false;
  }
  ResidentBoundaryCondensationWitness witness{};
  if (!observe_resident_boundary_condensation(
          boundary, recipes, derivations, occurrences, occurrence_count,
          couplings, coupling_count, closure, probes,
          static_cast<std::uint32_t>(guard_min),
          static_cast<std::uint32_t>(guard_max), &witness))
    return false;

  auto* recipe_pool = substrate::direct_adult::direct_ecology_pool(
      brain.resource_ecology, DirectResourcePoolKind::derivation_record);
  auto* parent_pool = substrate::direct_adult::direct_ecology_pool(
      brain.resource_ecology, DirectResourcePoolKind::derivation_parent_edge);
  if (recipe_pool == nullptr || parent_pool == nullptr ||
      recipe_pool->reserved_units == 0u ||
      parent_pool->reserved_units < occurrence_count)
    return false;
  const auto recipe_pool_before = *recipe_pool;
  const auto parent_pool_before = *parent_pool;
  if (!substrate::direct_adult::device_commit_pool_units(
          brain.resource_ecology, DirectResourcePoolKind::derivation_record, 1u))
    return false;
  if (!substrate::direct_adult::device_commit_pool_units(
          brain.resource_ecology, DirectResourcePoolKind::derivation_parent_edge,
          occurrence_count)) {
    *recipe_pool = recipe_pool_before;
    *parent_pool = parent_pool_before;
    return false;
  }
  if (derivations[0].ports[0].node >= brain.node_count) {
    *recipe_pool = recipe_pool_before;
    *parent_pool = parent_pool_before;
    return false;
  }
  const std::uint16_t territory =
      brain.nodes[derivations[0].ports[0].node].territory_index;
  if (!materialize_resident_boundary_recipe(
          brain.recipe_cells, &brain.development->recipe_cell_count,
          brain.recipe_cell_capacity, brain.postbirth_derivations,
          brain.postbirth_constructor, recipes, derivations, occurrence_count,
          witness, territory)) {
    *recipe_pool = recipe_pool_before;
    *parent_pool = parent_pool_before;
    return false;
  }
  for (std::uint32_t i = 0u; i < kResidentActualFrontierCapacity; ++i)
    if (frontier->entries[i].state == ResidentActualFrontierState::live) {
      frontier->entries[i].state = ResidentActualFrontierState::settled;
      frontier->entries[i].occurrence.state = kResidentRecipeOccurrenceSettled;
    }
  frontier->live_count = 0u;
  ++frontier->condensation_promotions;
  return true;
}

DIRECT_ADULT_HD inline std::uint64_t resident_whitebox_recipe_logical_identity(
    const direct_network::DirectWhiteboxCondensationV1& witness) {
  using namespace direct_network;
  std::uint64_t identity = exact_history_fold_word(
      0x7762726563697065ull, static_cast<std::uint32_t>(witness.source.kind));
  identity = exact_history_fold_word(identity, witness.result_count);
  for (std::uint32_t i = 0u; i < kDirectWhiteboxReductionWidth; ++i) {
    identity = exact_history_fold_word(
        identity, static_cast<std::uint32_t>(witness.result_q16[i]));
    identity = exact_history_fold_word(identity, witness.result_u32[i]);
  }
  return identity == 0u ? 1u : identity;
}

DIRECT_ADULT_HD inline std::uint32_t resident_whitebox_relation_family(
    direct_network::DirectWhiteboxReductionKindV1 kind) {
  using namespace direct_network;
  if (kind == DirectWhiteboxReductionKindV1::polynomial_substitution)
    return static_cast<std::uint32_t>(DirectRelationAlgebraFamilyV1::polynomial);
  if (kind == DirectWhiteboxReductionKindV1::deterministic_bisimulation)
    return static_cast<std::uint32_t>(DirectRelationAlgebraFamilyV1::discrete);
  return static_cast<std::uint32_t>(DirectRelationAlgebraFamilyV1::linear);
}

DIRECT_ADULT_HD inline bool materialize_resident_whitebox_recipe(
    direct_network::ResidentRecipeCell* cells, std::uint32_t* cell_count,
    std::uint32_t cell_capacity,
    direct_network::ResidentRecipeDerivation* derivations,
    direct_network::ResidentPostbirthConstructorState* state,
    std::uint32_t parent_cell,
    const direct_network::DirectWhiteboxCondensationV1& witness,
    std::uint16_t territory_index) {
  using namespace direct_network;
  if (cells == nullptr || cell_count == nullptr || derivations == nullptr ||
      state == nullptr || !direct_whitebox_condensation_valid(witness) ||
      witness.result_count == 0u ||
      witness.result_count > kResidentDerivationWidth ||
      parent_cell >= *cell_count || *cell_count >= cell_capacity ||
      *cell_count >= state->recipe_cell_capacity ||
      state->derivation_count >= state->derivation_capacity)
    return false;
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
      derivations[parent_derivation].generation == ~std::uint64_t{0})
    return false;
  const std::uint32_t relation_count =
      witness.source.kind == DirectWhiteboxReductionKindV1::deterministic_bisimulation
          ? witness.result_count : 1u;
  if (state->ports_used + 2u > state->port_capacity ||
      state->relations_used + relation_count > state->relation_capacity ||
      state->parameters_used + witness.result_count > state->parameter_capacity)
    return false;
  const std::uint64_t logical_id =
      resident_whitebox_recipe_logical_identity(witness);
  for (std::uint32_t i = 0u; i < *cell_count; ++i)
    if (cells[i].logical_recipe_id == logical_id) return false;
  const std::uint32_t new_cell = *cell_count;
  ResidentRecipeCell recipe{};
  recipe.logical_recipe_id = logical_id;
  recipe.support_q16 = cells[parent_cell].support_q16;
  recipe.rule_index = cells[parent_cell].rule_index;
  recipe.revision = 1u;
  recipe.revision_identity = resident_recipe_revision_identity(
      logical_id, 0u, 1u, witness.witness_identity,
      recipe.support_q16, recipe.credit_q16);
  if (!initialize_resident_recipe_update_ir(&recipe)) return false;
  ResidentRecipeDerivation derivation{};
  derivation.logical_recipe_id = logical_id;
  derivation.revision_identity = recipe.revision_identity;
  derivation.parent_logical_recipe_id = cells[parent_cell].logical_recipe_id;
  derivation.parent_revision_identity = cells[parent_cell].revision_identity;
  derivation.witness_identity = witness.witness_identity;
  derivation.generation = derivations[parent_derivation].generation + 1u;
  derivation.recipe_cell = new_cell;
  derivation.parent_recipe_cell = parent_cell;
  derivation.territory_index = territory_index;
  derivation.port_count = 2u;
  derivation.relation_count = static_cast<std::uint16_t>(relation_count);
  derivation.parameter_count = static_cast<std::uint16_t>(witness.result_count);
  derivation.ports[0] = derivations[parent_derivation].ports[0];
  derivation.ports[1] = derivations[parent_derivation].ports[1];
  for (std::uint32_t i = 0u; i < relation_count; ++i)
    derivation.relations[i] =
        witness.source.kind == DirectWhiteboxReductionKindV1::deterministic_bisimulation
            ? witness.result_u32[i] : resident_whitebox_relation_family(witness.source.kind);
  for (std::uint32_t i = 0u; i < witness.result_count; ++i)
    derivation.parameters_q16[i] = witness.result_q16[i];
  cells[new_cell] = recipe;
  derivations[state->derivation_count] = derivation;
  ++*cell_count;
  ++state->derivation_count;
  state->ports_used += 2u;
  state->relations_used += relation_count;
  state->parameters_used += witness.result_count;
  ++state->condensed;
  if (derivation.generation > state->highest_derivation_rank)
    state->highest_derivation_rank = derivation.generation;
  return true;
}

DIRECT_ADULT_HD inline bool replay_resident_whitebox_recipe(
    const direct_network::DirectWhiteboxCondensationV1& witness,
    const direct_network::ResidentRecipeCell& recipe,
    const direct_network::ResidentRecipeDerivation& derivation) {
  using namespace direct_network;
  if (!direct_whitebox_condensation_valid(witness) ||
      recipe.logical_recipe_id != resident_whitebox_recipe_logical_identity(witness) ||
      recipe.logical_recipe_id != derivation.logical_recipe_id ||
      recipe.revision_identity != derivation.revision_identity ||
      derivation.witness_identity != witness.witness_identity ||
      derivation.port_count != 2u ||
      derivation.parameter_count != witness.result_count)
    return false;
  const std::uint32_t relation_count =
      witness.source.kind == DirectWhiteboxReductionKindV1::deterministic_bisimulation
          ? witness.result_count : 1u;
  if (derivation.relation_count != relation_count) return false;
  for (std::uint32_t i = 0u; i < relation_count; ++i) {
    const std::uint32_t expected =
        witness.source.kind == DirectWhiteboxReductionKindV1::deterministic_bisimulation
            ? witness.result_u32[i] : resident_whitebox_relation_family(witness.source.kind);
    if (derivation.relations[i] != expected) return false;
  }
  for (std::uint32_t i = 0u; i < witness.result_count; ++i)
    if (derivation.parameters_q16[i] != witness.result_q16[i]) return false;
  return recipe.revision == 1u && recipe.credit_q16 == 0 &&
      recipe.revision_identity == resident_recipe_revision_identity(
          recipe.logical_recipe_id, 0u, 1u, witness.witness_identity,
          recipe.support_q16, recipe.credit_q16);
}

// A contradiction addresses the compacted recipe, while its refinement
// authority remains frozen in either the legacy pair witness or B_N metadata.
DIRECT_ADULT_HD inline bool stage_resident_condensation_refinement_event(
    direct_network::DirectExactHistoryRecord* event,
    const direct_network::ResidentRecipeDerivation& derivation,
    const direct_network::ResidentRecipeCell* cells, std::uint32_t cell_count,
    std::uint64_t occurrence_identity, std::uint32_t tick,
    std::uint32_t event_tick, std::uint32_t locator,
    std::uint32_t occurrence_ordinal, std::int64_t delta_q16) {
  using namespace direct_network;
  if (event == nullptr || cells == nullptr || derivation.recipe_cell >= cell_count)
    return false;
  ResidentNetworkCondensationEvidence evidence{};
  const bool legacy = rematerialize_resident_condensation(derivation, &evidence);
  const bool boundary = !legacy &&
      (derivation.condensation_flags & kResidentDerivationBoundaryCondensedNetwork) != 0u &&
      derivation.boundary_condensation_root_count >= 3u &&
      derivation.boundary_condensation_root_count <= kResidentBoundaryCondensationMaxRoots;
  if (!legacy && !boundary) return false;
  const std::uint64_t witness = legacy ? evidence.witness_identity : derivation.witness_identity;
  const std::uint64_t refinement_logical = legacy ? evidence.refinement_logical_recipe_id :
      derivation.boundary_condensation_refinement_logical_recipe_id;
  const std::uint64_t refinement_revision = legacy ? evidence.refinement_revision_identity :
      derivation.boundary_condensation_refinement_revision_identity;
  const std::uint32_t refinement_cell = legacy ? evidence.refinement_recipe_cell :
      derivation.boundary_condensation_refinement_recipe_cell;
  if (witness == 0u || refinement_cell >= cell_count ||
      cells[refinement_cell].logical_recipe_id != refinement_logical ||
      cells[refinement_cell].revision_identity != refinement_revision)
    return false;
  const auto& compacted = cells[derivation.recipe_cell];
  if (compacted.logical_recipe_id != derivation.logical_recipe_id ||
      compacted.revision_identity != derivation.revision_identity)
    return false;
  std::uint64_t refinement_evidence = exact_history_fold_word(witness, refinement_revision);
  if (refinement_evidence == 0u) refinement_evidence = 1u;
  return stage_resident_recipe_revision_event(
      event, compacted, derivation.recipe_cell,
      ResidentRecipeRevisionAuthority::structural, occurrence_identity,
      refinement_evidence, tick, event_tick, locator, occurrence_ordinal, 0u,
      delta_q16);
}

}  // namespace substrate::direct_adult_core

#endif
