#ifndef HARDWARE_NATIVE_DIRECT_RESIDENT_RECIPE_EXPERIENCE_REVISION_CUH
#define HARDWARE_NATIVE_DIRECT_RESIDENT_RECIPE_EXPERIENCE_REVISION_CUH

#include "hardware_native/direct_network_brain.cuh"
#include "hardware_native/direct_adult_core.cuh"
#include "hardware_native/direct_adult_recipe_ir.cuh"

namespace substrate::direct_adult_core {

// Authorization is the Occurrence gate. Persistent credit is a RecipeRevision
// written only by executing the encoded production IR. Expired eligibility and
// zero causal difference settle without a revision.
DIRECT_ADULT_HD inline bool commit_resident_causal_difference_revision(
    ResidentRecipeOccurrence* occurrence,
    direct_network::ResidentRecipeCell* cell, std::uint32_t recipe_cell,
    bool independent, std::int32_t effect, std::int32_t counterfactual,
    std::uint32_t return_tick = 0u) {
  if (occurrence == nullptr || cell == nullptr ||
      occurrence->state != kResidentRecipeOccurrenceLive ||
      occurrence->occurrence_identity == 0u ||
      occurrence->participation_identity == 0u ||
      occurrence->binding_count == 0u ||
      occurrence->lineage_kind != ResidentOccurrenceLineageKind::actual ||
      occurrence->authority != DirectParticipationAuthority::independent_external ||
      occurrence->logical_recipe_id != cell->logical_recipe_id ||
      occurrence->revision_identity != cell->revision_identity)
    return false;
  const std::uint32_t tick =
      return_tick != 0u ? return_tick : occurrence->timestamp;
  const bool live = tick <= occurrence->expiry_tick;
  const bool difference = live && independent && effect != counterfactual;
  if (difference) {
    const std::int64_t delta = effect > counterfactual
        ? static_cast<std::int64_t>(kQ16One)
        : -static_cast<std::int64_t>(kQ16One);
    direct_network::DirectExactHistoryRecord event{};
    ResidentRecipeIrProgram program{};
    if (!direct_network::stage_resident_recipe_revision_event(
            &event, *cell, recipe_cell,
            direct_network::ResidentRecipeRevisionAuthority::experience,
            occurrence->occurrence_identity, occurrence->participation_identity,
            occurrence->timestamp, tick, 0u, 0u, 0u, delta) ||
        !encode_resident_recipe_ir(event, 1u, &program) ||
        !execute_resident_recipe_ir(program, event, cell))
      return false;
  }
  occurrence->state = kResidentRecipeOccurrenceSettled;
  return true;
}

}  // namespace substrate::direct_adult_core

#endif
