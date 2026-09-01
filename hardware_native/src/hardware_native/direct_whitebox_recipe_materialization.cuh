#ifndef HARDWARE_NATIVE_DIRECT_WHITEBOX_RECIPE_MATERIALIZATION_CUH
#define HARDWARE_NATIVE_DIRECT_WHITEBOX_RECIPE_MATERIALIZATION_CUH

#include "hardware_native/direct_network_brain.cuh"
#include "hardware_native/direct_mathematical_relation_algebra.cuh"

#if defined(__CUDACC__)
#define DIRECT_WHITEBOX_MATERIALIZE_HD __host__ __device__
#else
#define DIRECT_WHITEBOX_MATERIALIZE_HD
#endif

namespace substrate::direct_adult_core {

DIRECT_WHITEBOX_MATERIALIZE_HD inline std::uint64_t resident_whitebox_recipe_logical_identity(
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

DIRECT_WHITEBOX_MATERIALIZE_HD inline std::uint32_t resident_whitebox_relation_family(
    direct_network::DirectWhiteboxReductionKindV1 kind) {
  using namespace direct_network;
  if (kind == DirectWhiteboxReductionKindV1::polynomial_substitution)
    return static_cast<std::uint32_t>(DirectRelationAlgebraFamilyV1::polynomial);
  if (kind == DirectWhiteboxReductionKindV1::deterministic_bisimulation)
    return static_cast<std::uint32_t>(DirectRelationAlgebraFamilyV1::discrete);
  return static_cast<std::uint32_t>(DirectRelationAlgebraFamilyV1::linear);
}

DIRECT_WHITEBOX_MATERIALIZE_HD inline bool materialize_resident_whitebox_recipe(
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

DIRECT_WHITEBOX_MATERIALIZE_HD inline bool replay_resident_whitebox_recipe(
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


}  // namespace substrate::direct_adult_core

#undef DIRECT_WHITEBOX_MATERIALIZE_HD
#endif  // HARDWARE_NATIVE_DIRECT_WHITEBOX_RECIPE_MATERIALIZATION_CUH
