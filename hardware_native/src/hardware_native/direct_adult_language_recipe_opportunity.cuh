#ifndef HARDWARE_NATIVE_DIRECT_ADULT_LANGUAGE_RECIPE_OPPORTUNITY_CUH
#define HARDWARE_NATIVE_DIRECT_ADULT_LANGUAGE_RECIPE_OPPORTUNITY_CUH

#include <cstdint>
#include <type_traits>

#include "hardware_native/direct_exact_history.cuh"
#include "hardware_native/direct_adult_core.cuh"

namespace substrate::direct_adult_core {

// Compact learned bridge from one authenticated/familiar public surface to the
// resident RecipeRevision that actually participated when that surface was
// expressed.  This is nomination geometry only: no current activation,
// eligibility, external authority, truth or causal credit is serialized here.
struct ResidentLanguageRecipeOpportunityV1 {
  std::uint64_t identity;
  std::uint64_t surface_identity;
  std::uint64_t logical_recipe_id;
  std::uint64_t revision_identity;
  std::uint64_t source_occurrence_identity;
  std::uint32_t context_signature;
  std::uint16_t binding_count;
  std::uint16_t surface_length;
  std::uint32_t variables[direct_network::kResidentDerivationWidth];
};
static_assert(std::is_standard_layout_v<ResidentLanguageRecipeOpportunityV1> &&
              std::is_trivial_v<ResidentLanguageRecipeOpportunityV1> &&
              std::has_unique_object_representations_v<ResidentLanguageRecipeOpportunityV1>);

DIRECT_ADULT_HD inline std::uint64_t resident_language_recipe_opportunity_identity(
    std::uint64_t surface_identity, const ResidentRecipeOccurrence& occurrence) {
  using direct_network::exact_history_fold_word;
  std::uint64_t h = exact_history_fold_word(0x6c616e676f707076ull, surface_identity);
  h = exact_history_fold_word(h, occurrence.logical_recipe_id);
  h = exact_history_fold_word(h, occurrence.revision_identity);
  h = exact_history_fold_word(h, occurrence.context_signature);
  h = exact_history_fold_word(h, occurrence.binding_count);
  for (std::uint16_t i = 0u; i < occurrence.binding_count; ++i)
    h = exact_history_fold_word(h, occurrence.bindings[i].variable_identity);
  return h == 0u ? 1u : h;
}

DIRECT_ADULT_HD inline bool earn_resident_language_recipe_opportunity(
    std::uint64_t surface_identity, const ResidentRecipeOccurrence& expressed,
    ResidentLanguageRecipeOpportunityV1* out, std::uint16_t surface_length = 0u) {
  if (out == nullptr || surface_identity == 0u ||
      expressed.state != kResidentRecipeOccurrenceLive ||
      expressed.lineage_kind != ResidentOccurrenceLineageKind::actual ||
      expressed.authority == DirectParticipationAuthority::none ||
      expressed.logical_recipe_id == 0u || expressed.revision_identity == 0u ||
      expressed.occurrence_identity == 0u || expressed.binding_count == 0u ||
      expressed.binding_count > direct_network::kResidentDerivationWidth)
    return false;
  ResidentLanguageRecipeOpportunityV1 row{};
  row.surface_identity = surface_identity;
  row.logical_recipe_id = expressed.logical_recipe_id;
  row.revision_identity = expressed.revision_identity;
  row.source_occurrence_identity = expressed.occurrence_identity;
  row.context_signature = expressed.context_signature;
  row.binding_count = expressed.binding_count;
  row.surface_length = surface_length;
  for (std::uint16_t i = 0u; i < expressed.binding_count; ++i) {
    if (expressed.bindings[i].formal_port_index != i ||
        expressed.bindings[i].variable_identity == 0u)
      return false;
    row.variables[i] = expressed.bindings[i].variable_identity;
  }
  row.identity = resident_language_recipe_opportunity_identity(surface_identity, expressed);
  *out = row;
  return true;
}

template <typename RecipeCellT, typename DerivationT>
DIRECT_ADULT_HD inline bool nominate_resident_language_recipe_opportunity(
    const ResidentLanguageRecipeOpportunityV1& row,
    std::uint64_t heard_surface_identity, const RecipeCellT* cells,
    std::uint32_t cell_count, const DerivationT* derivations,
    std::uint32_t derivation_count, std::uint64_t candidate_occurrence_identity,
    ResidentRecipeOccurrence* out) {
  if (out == nullptr || row.identity == 0u || heard_surface_identity == 0u ||
      heard_surface_identity != row.surface_identity || cells == nullptr ||
      derivations == nullptr || row.binding_count == 0u ||
      candidate_occurrence_identity == 0u)
    return false;
  const RecipeCellT* cell = nullptr;
  const DerivationT* derivation = nullptr;
  for (std::uint32_t i = 0u; i < cell_count; ++i) {
    if (cells[i].logical_recipe_id != row.logical_recipe_id ||
        cells[i].revision_identity != row.revision_identity)
      continue;
    if (cell != nullptr) return false;
    cell = &cells[i];
  }
  for (std::uint32_t i = 0u; i < derivation_count; ++i) {
    if (derivations[i].logical_recipe_id != row.logical_recipe_id ||
        derivations[i].revision_identity != row.revision_identity ||
        derivations[i].port_count != row.binding_count)
      continue;
    if (derivation != nullptr) return false;
    derivation = &derivations[i];
  }
  if (cell == nullptr || derivation == nullptr) return false;
  return bind_resident_recipe_occurrence(
      *cell, *derivation, row.variables, row.binding_count,
      candidate_occurrence_identity, candidate_occurrence_identity + 1u,
      row.source_occurrence_identity, 1u, ResidentOccurrenceLineageKind::endogenous,
      DirectParticipationAuthority::none, row.context_signature, 90u, 100u, 0, out);
}

}  // namespace substrate::direct_adult_core

#endif
