#ifndef HARDWARE_NATIVE_DIRECT_ADULT_LIVED_EXPRESSION_IDENTITY_CUH
#define HARDWARE_NATIVE_DIRECT_ADULT_LIVED_EXPRESSION_IDENTITY_CUH

#include "hardware_native/direct_adult_core.cuh"
#include "hardware_native/direct_adult_runtime_frontiers.cuh"

namespace substrate::direct_adult_core {

DIRECT_ADULT_HD inline const ResidentActualFrontierEntry*
resident_current_lived_expression_entry(
    const DirectBrain& brain, const ResidentActualFrontier* frontier,
    const DirectActionParticipationLink& link) {
  if (frontier == nullptr || brain.development == nullptr ||
      brain.postbirth_constructor == nullptr ||
      brain.postbirth_derivations == nullptr || brain.recipe_cells == nullptr ||
      frontier->live_count > kResidentActualFrontierCapacity ||
      link.occurrence_identity == 0u || link.logical_recipe_id == 0u ||
      link.revision_identity == 0u || link.participation_identity == 0u)
    return nullptr;
  for (std::uint32_t slot = 0u; slot < kResidentActualFrontierCapacity; ++slot) {
    const auto& entry = frontier->entries[slot];
    const auto& occurrence = entry.occurrence;
    if (entry.state != ResidentActualFrontierState::live ||
        occurrence.state != kResidentRecipeOccurrenceLive ||
        occurrence.lineage_kind != ResidentOccurrenceLineageKind::actual ||
        occurrence.logical_recipe_id != link.logical_recipe_id ||
        occurrence.occurrence_identity != link.occurrence_identity ||
        occurrence.participation_identity != link.participation_identity ||
        occurrence.context_signature != link.occurrence_context_signature ||
        occurrence.route_incarnation != link.occurrence_route_incarnation ||
        occurrence.source_incarnation != link.claim_incarnation ||
        !resident_occurrence_accepts_causal_authority(
            occurrence.authority, link.authority) ||
        !resident_actual_frontier_entry_current(
            brain, *brain.postbirth_constructor, entry) ||
        entry.derivation_index >= brain.postbirth_constructor->derivation_count)
      continue;
    const auto& derivation = brain.postbirth_derivations[entry.derivation_index];
    if (derivation.recipe_cell >= brain.development->recipe_cell_count) continue;
    const auto& recipe = brain.recipe_cells[derivation.recipe_cell];
    if (derivation.revision_identity != recipe.revision_identity ||
        occurrence.revision_identity != recipe.revision_identity ||
        direct_network::resident_recipe_current_revision_authority(recipe) ==
            direct_network::ResidentRecipeRevisionAuthority::none)
      continue;
    return &entry;
  }
  return nullptr;
}

}  // namespace substrate::direct_adult_core

#endif
