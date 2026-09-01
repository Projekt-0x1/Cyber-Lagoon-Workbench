#ifndef HARDWARE_NATIVE_DIRECT_ADULT_LIVED_EXPRESSION_CUH
#define HARDWARE_NATIVE_DIRECT_ADULT_LIVED_EXPRESSION_CUH

#include "direct_adult_core.cuh"
#include "direct_adult_runtime_frontiers.cuh"

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
        occurrence.authority != DirectParticipationAuthority::independent_external ||
        occurrence.logical_recipe_id != link.logical_recipe_id ||
        occurrence.occurrence_identity != link.occurrence_identity ||
        occurrence.participation_identity != link.participation_identity ||
        occurrence.context_signature != link.occurrence_context_signature ||
        occurrence.route_incarnation != link.occurrence_route_incarnation ||
        occurrence.source_incarnation != link.claim_incarnation ||
        occurrence.authority != link.authority ||
        !resident_actual_frontier_entry_current(
            brain, *brain.postbirth_constructor, entry) ||
        entry.derivation_index >= brain.postbirth_constructor->derivation_count)
      continue;
    const auto& derivation =
        brain.postbirth_derivations[entry.derivation_index];
    if (derivation.recipe_cell >= brain.development->recipe_cell_count) continue;
    const auto& recipe = brain.recipe_cells[derivation.recipe_cell];
    if (derivation.revision_identity != recipe.revision_identity ||
        occurrence.revision_identity != recipe.revision_identity ||
        direct_network::resident_recipe_current_revision_authority(recipe) !=
            direct_network::ResidentRecipeRevisionAuthority::experience)
      continue;
    // The frozen revision is immutable origin lineage, not an instruction to
    // replay obsolete morphology. Exact stable Occurrence identity above plus
    // the current experience-authorized RecipeRevision below owns execution.
    return &entry;
  }
  return nullptr;
}

DIRECT_ADULT_HD inline bool resident_lived_expression_entry(
    const DirectBrain& brain, const ResidentActualFrontierEntry& entry,
    Word* expression) {
  if (expression == nullptr || brain.development == nullptr ||
      brain.postbirth_constructor == nullptr ||
      brain.postbirth_derivations == nullptr || brain.recipe_cells == nullptr ||
      entry.derivation_index >= brain.postbirth_constructor->derivation_count)
    return false;
  const auto& derivation = brain.postbirth_derivations[entry.derivation_index];
  if (derivation.recipe_cell >= brain.development->recipe_cell_count) return false;
  const auto& recipe = brain.recipe_cells[derivation.recipe_cell];
  if (recipe.logical_recipe_id != entry.occurrence.logical_recipe_id ||
      recipe.revision_identity != entry.occurrence.revision_identity ||
      derivation.logical_recipe_id != recipe.logical_recipe_id ||
      derivation.revision_identity != recipe.revision_identity ||
      direct_network::resident_recipe_current_revision_authority(recipe) ==
          direct_network::ResidentRecipeRevisionAuthority::none)
    return false;
  ResidentExecutableMorphologyWork work{};
  std::int32_t base = 0;
  std::uint32_t work_units = 0u;
  if (!lower_resident_executable_morphology(
          brain, derivation, entry.occurrence, false, &work) ||
      !execute_resident_executable_morphology(
          brain, derivation, entry.occurrence, work,
          entry.occurrence.activation_q16, &base, &work_units))
    return false;
  *expression = static_cast<Word>(base);
  return true;
}

DIRECT_ADULT_HD inline bool resident_lived_expression(
    const DirectBrain& brain, const ResidentActualFrontier* frontier,
    const DirectActionParticipationLink& link, Word* expression) {
  const auto* entry =
      resident_current_lived_expression_entry(brain, frontier, link);
  return entry != nullptr &&
      resident_lived_expression_entry(brain, *entry, expression);
}

// Re-execute one exact current actual Occurrence through a resident morphology
// authorized by structural contact or later causal-difference experience. The
// authority only permits execution: it never manufactures credit. Ambiguity or
// stale causal matter refuses; history, credit telemetry, and host semantics
// are never outputs.
// Exact logical, revision, participation, context, route, claim-incarnation,
// and authority identities remain mandatory. Current derivation and incidence
// matter own selection and bounded re-execution. Failure cannot invent a
// learned expression or fall back to stored contact/consequence bytes.
DIRECT_ADULT_HD inline bool resident_lived_expression(
    const DirectBrain& brain, const ResidentActualFrontier* frontier,
    const DirectActionParticipationLink* action_links,
    std::uint32_t participant_offset, std::uint32_t participant_count,
    Word* expression) {
  if (frontier == nullptr || action_links == nullptr || expression == nullptr ||
      brain.development == nullptr || brain.postbirth_constructor == nullptr ||
      brain.postbirth_derivations == nullptr || brain.recipe_cells == nullptr ||
      participant_offset == kInvalidIndex || participant_count == 0u ||
      participant_count > kMaxActionParticipationLinks ||
      frontier->live_count > kResidentActualFrontierCapacity) return false;
  const ResidentActualFrontierEntry* match = nullptr;
  for (std::uint32_t p = 0u; p < participant_count; ++p) {
    const auto& link = action_links[participant_offset + p];
    const auto* entry =
        resident_current_lived_expression_entry(brain, frontier, link);
    if (entry == nullptr) continue;
    if (match != nullptr && match != entry) return false;
    match = entry;
  }
  return match != nullptr &&
      resident_lived_expression_entry(brain, *match, expression);
}
}  // namespace substrate::direct_adult_core

#endif
