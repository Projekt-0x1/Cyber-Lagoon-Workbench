#ifndef HARDWARE_NATIVE_DIRECT_FOUNDRY_CONDENSATION_WORKSPACE_CUH
#define HARDWARE_NATIVE_DIRECT_FOUNDRY_CONDENSATION_WORKSPACE_CUH

// Adult mint transaction only: named parent + whitebox witness -> p(n+1).
// N admission is Adult `admit_resident_network_parent`
// (rebind + membership + actual closure).
// Not a second mind and not 1610 IR.

#include "hardware_native/direct_whitebox_recipe_materialization.cuh"
#include "hardware_native/direct_resident_occurrence_coupling.cuh"

#if defined(__CUDACC__)
#define DIRECT_FOUNDRY_WORKSPACE_HD __host__ __device__
#else
#define DIRECT_FOUNDRY_WORKSPACE_HD
#endif

namespace substrate::direct_adult_core {

using direct_network::exact_history_fold_word;
using direct_network::kResidentDerivationWidth;
using direct_network::resident_recipe_ports_compatible;
#include "hardware_native/direct_adult_resident_relational_network.cuh"
#include "hardware_native/direct_adult_resident_network_whitebox_observe.cuh"

struct FoundryCondensationWorkspace {
  direct_network::ResidentRecipeCell* cells;
  std::uint32_t* cell_count;
  std::uint32_t cell_capacity;
  direct_network::ResidentRecipeDerivation* derivations;
  direct_network::ResidentPostbirthConstructorState* state;
  std::uint32_t parent_cell;
  std::uint64_t parent_logical;
  const direct_network::DirectWhiteboxCondensationV1* witness;
  std::uint16_t territory;
  std::uint64_t claimed_identity;
  const direct_network::ResidentRecipeCell* network_cells;
  const direct_network::ResidentRecipeDerivation* network_ders;
  const ResidentRecipeOccurrence* occurrences;
  const ResidentOccurrenceCoupling* couplings;
  std::uint32_t occurrence_count, coupling_count;
  const direct_network::ResidentNetworkCondensationEvidence* evidence;
};

DIRECT_FOUNDRY_WORKSPACE_HD inline bool foundry_admit_network(
    const FoundryCondensationWorkspace* condensation) {
  if (condensation == nullptr || condensation->claimed_identity == 0u)
    return true;
  return admit_resident_network_parent(
      condensation->network_cells, condensation->network_ders,
      condensation->occurrences, condensation->occurrence_count,
      condensation->couplings, condensation->coupling_count,
      condensation->claimed_identity, condensation->parent_logical);
}

DIRECT_FOUNDRY_WORKSPACE_HD inline bool foundry_workspace_ready(
    const FoundryCondensationWorkspace* condensation) {
  if (condensation == nullptr || condensation->cells == nullptr ||
      condensation->cell_count == nullptr || condensation->derivations == nullptr ||
      condensation->state == nullptr || condensation->witness == nullptr ||
      condensation->parent_cell >= *condensation->cell_count ||
      condensation->parent_logical == 0u)
    return false;
  const std::uint32_t parent = condensation->parent_cell;
  if (condensation->cells[parent].logical_recipe_id != condensation->parent_logical)
    return false;
  if (!direct_network::direct_whitebox_condensation_valid(*condensation->witness) ||
      *condensation->cell_count >= condensation->cell_capacity ||
      *condensation->cell_count >= condensation->state->recipe_cell_capacity ||
      condensation->state->derivation_count >= condensation->state->derivation_capacity)
    return false;
  bool parent_derivation = false;
  for (std::uint32_t i = 0u; i < condensation->state->derivation_count; ++i)
    if (condensation->derivations[i].recipe_cell == parent) parent_derivation = true;
  if (!parent_derivation) return false;
  const std::uint64_t logical =
      resident_whitebox_recipe_logical_identity(*condensation->witness);
  for (std::uint32_t i = 0u; i < *condensation->cell_count; ++i)
    if (condensation->cells[i].logical_recipe_id == logical) return false;
  if (condensation->evidence != nullptr) {
    std::uint64_t replay_logical = 0, replay_rev = 0;
    if (!replay_resident_network_candidate(
            condensation->cells, *condensation->cell_count, condensation->derivations,
            condensation->state->derivation_count, *condensation->evidence,
            condensation->witness, &replay_logical, &replay_rev) ||
        replay_logical != logical)
      return false;
  }
  return foundry_admit_network(condensation);
}

DIRECT_FOUNDRY_WORKSPACE_HD inline bool foundry_materialize_from_workspace(
    FoundryCondensationWorkspace* condensation,
    const direct_network::ResidentRecipeCell* episode_cell) {
  if (condensation == nullptr || condensation->cells == nullptr ||
      condensation->cell_count == nullptr || condensation->derivations == nullptr ||
      condensation->state == nullptr || condensation->witness == nullptr ||
      condensation->parent_cell >= *condensation->cell_count)
    return false;
  const std::uint32_t parent = condensation->parent_cell;
  if (condensation->parent_logical == 0u ||
      condensation->cells[parent].logical_recipe_id != condensation->parent_logical)
    return false;
  std::uint32_t parent_derivation = condensation->state->derivation_count;
  for (std::uint32_t i = 0u; i < condensation->state->derivation_count; ++i)
    if (condensation->derivations[i].recipe_cell == parent) {
      parent_derivation = i;
      break;
    }
  if (parent_derivation == condensation->state->derivation_count) return false;
  const direct_network::ResidentRecipeCell saved_cell = condensation->cells[parent];
  const std::uint64_t saved_logical =
      condensation->derivations[parent_derivation].logical_recipe_id;
  const std::uint64_t saved_revision =
      condensation->derivations[parent_derivation].revision_identity;
  if (episode_cell != nullptr &&
      episode_cell->logical_recipe_id == condensation->cells[parent].logical_recipe_id)
    condensation->cells[parent] = *episode_cell;
  condensation->derivations[parent_derivation].logical_recipe_id =
      condensation->cells[parent].logical_recipe_id;
  condensation->derivations[parent_derivation].revision_identity =
      condensation->cells[parent].revision_identity;
  if (materialize_resident_whitebox_recipe(
          condensation->cells, condensation->cell_count, condensation->cell_capacity,
          condensation->derivations, condensation->state, parent,
          *condensation->witness, condensation->territory))
    return true;
  condensation->cells[parent] = saved_cell;
  condensation->derivations[parent_derivation].logical_recipe_id = saved_logical;
  condensation->derivations[parent_derivation].revision_identity = saved_revision;
  return false;
}

}  // namespace substrate::direct_adult_core

#undef DIRECT_FOUNDRY_WORKSPACE_HD
#endif
