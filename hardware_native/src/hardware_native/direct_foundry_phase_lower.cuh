#ifndef HARDWARE_NATIVE_DIRECT_FOUNDRY_PHASE_LOWER_CUH
#define HARDWARE_NATIVE_DIRECT_FOUNDRY_PHASE_LOWER_CUH

#include "hardware_native/direct_relational_sequence_composition.cuh"
#include "hardware_native/direct_resident_recipe_experience_revision.cuh"
#include "hardware_native/direct_foundry_condensation_workspace.cuh"

namespace substrate::direct_adult_core {

// 1610 phase *names* only. This is not the Adult loop and not IR translation.
// Adult cognition remains:
//   p -> q -> N -> B_N -> candidate/witness -> p(n+1)
// RelSeq owns ARBITRATE/FRONTIER/COMPOSE/PUBLISH. SETTLE writes credit.
// CONDENSE is Adult mint: rebind N, then named-parent whitebox materialize.
// Compact-root storage stays with the 1610 / hierarchy-storage owners.

struct FoundryPhaseLowering {
  std::uint32_t frontier, arbitrate, compose, publish, settle, condense;
  std::uint32_t units, depth;
  std::int64_t credit_q16;
  std::uint64_t revision_identity;
  std::uint64_t condensed_logical_id;
  std::uint64_t condensed_revision_identity;
  std::uint64_t condensed_generation;
};

DIRECT_ADULT_HD inline bool lower_foundry_phase_episode(
    const direct_network::DirectRelSeqRecipe* recipes, std::uint32_t recipe_n,
    direct_network::DirectRelSeqOccurrence* occ, std::uint32_t occ_n,
    const direct_network::DirectRelSeqRecipe* morphs, std::uint32_t morph_n,
    std::uint32_t parent, std::uint32_t port, std::uint64_t root,
    ResidentRecipeOccurrence* actual, direct_network::ResidentRecipeCell* cell,
    FoundryPhaseLowering* out,
    FoundryCondensationWorkspace* condensation = nullptr) {
  if (out == nullptr || recipes == nullptr || occ == nullptr || morphs == nullptr)
    return false;
  *out = FoundryPhaseLowering{};
  const direct_network::DirectRelSeqPortBinding* child_binding =
      direct_network::direct_relseq_binding(occ[parent], port);
  if (child_binding == nullptr || child_binding->child_occurrence_identity == 0u)
    return false;
  const direct_network::DirectRelSeqOccurrence* child = nullptr;
  for (std::uint32_t i = 0u; i < occ_n; ++i)
    if (occ[i].occurrence_identity == child_binding->child_occurrence_identity) {
      child = &occ[i];
      break;
    }
  if (child == nullptr ||
      !direct_network::direct_relseq_inherit_context(*child, &occ[parent]))
    return false;
  out->frontier = 1u;
  const direct_network::DirectRelSeqRecipe* picked =
      direct_network::direct_relseq_select_recipe(morphs, morph_n, 0xF2ull, occ[parent]);
  if (picked == nullptr) return false;
  occ[parent].logical_recipe_id = picked->logical_recipe_id;
  occ[parent].revision_identity = picked->revision_identity;
  out->arbitrate = 1u;
  direct_network::DirectRelSeqRecipe exec[3]{};
  if (recipe_n == 0u || morph_n < 2u) return false;
  exec[0] = recipes[0];
  exec[1] = morphs[0];
  exec[2] = morphs[1];
  direct_network::DirectRelSeqOutput published{};
  if (!direct_network::direct_relseq_evaluate_occurrence(exec, 3u, occ, occ_n, root,
                                                        &published))
    return false;
  out->compose = 1u;
  out->publish = published.count != 0u ? 1u : 0u;
  out->units = published.count;
  out->depth = published.depth_peak;
  if (condensation != nullptr && !foundry_workspace_ready(condensation))
    return false;
  if (!commit_resident_causal_difference_revision(actual, cell, 0u, true, 1, 0))
    return false;
  out->settle = 1u;
  out->credit_q16 = cell->credit_q16;
  out->revision_identity = cell->revision_identity;
  if (condensation == nullptr) {
    out->condense = 0u;
    return out->frontier && out->arbitrate && out->compose && out->publish &&
           out->settle;
  }
  if (!foundry_materialize_from_workspace(condensation, cell))
    return false;
  const std::uint32_t born_cell = *condensation->cell_count - 1u;
  const std::uint32_t born_der = condensation->state->derivation_count - 1u;
  out->condense = 1u;
  out->condensed_logical_id = condensation->cells[born_cell].logical_recipe_id;
  out->condensed_revision_identity = condensation->cells[born_cell].revision_identity;
  out->condensed_generation = condensation->derivations[born_der].generation;
  return out->frontier && out->arbitrate && out->compose && out->publish &&
         out->settle && out->condense;
}

}  // namespace substrate::direct_adult_core

#endif
