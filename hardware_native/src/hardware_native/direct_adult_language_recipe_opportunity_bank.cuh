#ifndef HARDWARE_NATIVE_DIRECT_ADULT_LANGUAGE_RECIPE_OPPORTUNITY_BANK_CUH
#define HARDWARE_NATIVE_DIRECT_ADULT_LANGUAGE_RECIPE_OPPORTUNITY_BANK_CUH

#include <cstdint>
#include <type_traits>

#include "hardware_native/direct_adult_language_recipe_opportunity.cuh"
#include "hardware_native/direct_adult_surface_sequence.cuh"

namespace substrate::direct_adult_core {

inline constexpr std::uint32_t kResidentLanguageRecipeOpportunityCapacity = 1024u;

struct ResidentLanguageRecipeOpportunityBankV1 {
  ResidentLanguageRecipeOpportunityV1 rows[kResidentLanguageRecipeOpportunityCapacity];
  std::uint32_t count;
  std::uint32_t refusals;
};
static_assert(std::is_standard_layout_v<ResidentLanguageRecipeOpportunityBankV1> &&
              std::is_trivial_v<ResidentLanguageRecipeOpportunityBankV1> &&
              std::has_unique_object_representations_v<ResidentLanguageRecipeOpportunityBankV1>);

DIRECT_ADULT_HD inline bool record_resident_language_recipe_opportunity(
    ResidentLanguageRecipeOpportunityBankV1* bank, std::uint64_t surface_identity,
    const ResidentRecipeOccurrence& expressed, std::uint16_t surface_length = 0u) {
  if (bank == nullptr) return false;
  ResidentLanguageRecipeOpportunityV1 candidate{};
  if (!earn_resident_language_recipe_opportunity(surface_identity, expressed,
                                                  &candidate, surface_length)) {
    ++bank->refusals;
    return false;
  }
  const std::uint32_t count =
      bank->count < kResidentLanguageRecipeOpportunityCapacity
          ? bank->count
          : kResidentLanguageRecipeOpportunityCapacity;
  for (std::uint32_t i = 0u; i < count; ++i)
    if (bank->rows[i].identity == candidate.identity) return true;
  if (bank->count >= kResidentLanguageRecipeOpportunityCapacity) {
    ++bank->refusals;
    return false;
  }
  bank->rows[bank->count++] = candidate;
  return true;
}

template <typename RecipeCellT, typename DerivationT>
DIRECT_ADULT_HD inline bool nominate_resident_language_recipe_from_bank(
    const ResidentLanguageRecipeOpportunityBankV1& bank,
    std::uint64_t heard_surface_identity, const RecipeCellT* cells,
    std::uint32_t cell_count, const DerivationT* derivations,
    std::uint32_t derivation_count, std::uint64_t candidate_occurrence_identity,
    ResidentRecipeOccurrence* out) {
  if (out == nullptr || heard_surface_identity == 0u ||
      candidate_occurrence_identity == 0u)
    return false;
  const std::uint32_t count =
      bank.count < kResidentLanguageRecipeOpportunityCapacity
          ? bank.count
          : kResidentLanguageRecipeOpportunityCapacity;
  const ResidentLanguageRecipeOpportunityV1* row = nullptr;
  std::uint32_t row_index = 0u;
  for (std::uint32_t i = 0u; i < count; ++i) {
    if (bank.rows[i].surface_identity != heard_surface_identity) continue;
    // Distinct learned mappings for one heard surface are unresolved by design.
    // Refuse before the much larger Recipe/derivation scans.
    if (row != nullptr) return false;
    row = &bank.rows[i];
    row_index = i;
  }
  if (row == nullptr) return false;
  return nominate_resident_language_recipe_opportunity(
      *row, heard_surface_identity, cells, cell_count, derivations,
      derivation_count, candidate_occurrence_identity + row_index, out);
}

template <typename RecipeCellT, typename DerivationT>
DIRECT_ADULT_HD inline std::uint32_t collect_resident_language_recipes_from_contained_surface(
    const ResidentLanguageRecipeOpportunityBankV1& bank,
    const std::uint32_t* heard_values, std::uint32_t heard_length,
    const RecipeCellT* cells, std::uint32_t cell_count,
    const DerivationT* derivations, std::uint32_t derivation_count,
    std::uint64_t candidate_occurrence_identity,
    std::uint64_t* logical_recipe_ids, std::uint64_t* revision_identities,
    std::uint32_t capacity) {
  constexpr std::uint32_t kContainedSurfaceMaxUnits = 64u;
  if (heard_values == nullptr || heard_length == 0u ||
      heard_length > kContainedSurfaceMaxUnits || candidate_occurrence_identity == 0u ||
      logical_recipe_ids == nullptr || revision_identities == nullptr || capacity == 0u)
    return 0u;
  const std::uint32_t count =
      bank.count < kResidentLanguageRecipeOpportunityCapacity
          ? bank.count
          : kResidentLanguageRecipeOpportunityCapacity;

  // Dynamic segmentation maximizes learned surface coverage, then minimizes the
  // number of constituent spans. Equal-score paths are ambiguous and fail closed.
  std::uint16_t covered[kContainedSurfaceMaxUnits + 1u]{};
  std::uint16_t components[kContainedSurfaceMaxUnits + 1u]{};
  std::uint8_t ways[kContainedSurfaceMaxUnits + 1u]{};
  std::uint16_t previous[kContainedSurfaceMaxUnits + 1u]{};
  std::uint16_t selected_row[kContainedSurfaceMaxUnits + 1u]{};
  for (std::uint32_t i = 0u; i <= heard_length; ++i) selected_row[i] = 0xffffu;
  ways[0] = 1u;

  for (std::uint32_t begin = 0u; begin < heard_length; ++begin) {
    if (ways[begin] == 0u) continue;
    const std::uint32_t skip_end = begin + 1u;
    if (covered[begin] > covered[skip_end] ||
        (covered[begin] == covered[skip_end] && components[begin] < components[skip_end]) ||
        ways[skip_end] == 0u) {
      covered[skip_end] = covered[begin];
      components[skip_end] = components[begin];
      ways[skip_end] = ways[begin] > 1u ? 2u : ways[begin];
      previous[skip_end] = static_cast<std::uint16_t>(begin);
      selected_row[skip_end] = 0xffffu;
    } else if (covered[begin] == covered[skip_end] &&
               components[begin] == components[skip_end]) {
      ways[skip_end] = 2u;
    }

    for (std::uint32_t i = 0u; i < count; ++i) {
      const auto& candidate = bank.rows[i];
      if (candidate.surface_length == 0u ||
          begin + candidate.surface_length > heard_length)
        continue;
      if (direct_network::surface_ecology_content_identity(
              candidate.surface_length, heard_values + begin) != candidate.surface_identity)
        continue;
      bool ambiguous_mapping = false;
      for (std::uint32_t j = 0u; j < count; ++j) {
        if (j == i) continue;
        const auto& other = bank.rows[j];
        ambiguous_mapping |= other.surface_identity == candidate.surface_identity &&
                             other.surface_length == candidate.surface_length &&
                             (other.logical_recipe_id != candidate.logical_recipe_id ||
                              other.revision_identity != candidate.revision_identity);
      }
      if (ambiguous_mapping) return 0u;
      ResidentRecipeOccurrence nominated{};
      if (!nominate_resident_language_recipe_opportunity(
              candidate, candidate.surface_identity, cells, cell_count, derivations,
              derivation_count, candidate_occurrence_identity + i + begin, &nominated))
        continue;
      const std::uint32_t end = begin + candidate.surface_length;
      const std::uint16_t next_covered = static_cast<std::uint16_t>(
          covered[begin] + candidate.surface_length);
      const std::uint16_t next_components =
          static_cast<std::uint16_t>(components[begin] + 1u);
      const bool better = ways[end] == 0u || next_covered > covered[end] ||
                          (next_covered == covered[end] && next_components < components[end]);
      if (better) {
        covered[end] = next_covered;
        components[end] = next_components;
        ways[end] = ways[begin] > 1u ? 2u : ways[begin];
        previous[end] = static_cast<std::uint16_t>(begin);
        selected_row[end] = static_cast<std::uint16_t>(i);
      } else if (next_covered == covered[end] && next_components == components[end]) {
        ways[end] = 2u;
      }
    }
  }

  if (ways[heard_length] != 1u || components[heard_length] == 0u ||
      components[heard_length] > capacity)
    return 0u;
  std::uint16_t rows[kContainedSurfaceMaxUnits]{};
  std::uint32_t admitted = 0u;
  std::uint32_t cursor = heard_length;
  while (cursor != 0u) {
    const std::uint16_t row = selected_row[cursor];
    const std::uint32_t prior = previous[cursor];
    if (prior >= cursor) return 0u;
    if (row != 0xffffu) rows[admitted++] = row;
    cursor = prior;
  }
  if (admitted != components[heard_length]) return 0u;
  for (std::uint32_t n = 0u; n < admitted; ++n) {
    const auto& candidate = bank.rows[rows[admitted - 1u - n]];
    logical_recipe_ids[n] = candidate.logical_recipe_id;
    revision_identities[n] = candidate.revision_identity;
  }
  return admitted;
}

template <typename RecipeCellT, typename DerivationT>
DIRECT_ADULT_HD inline bool nominate_resident_language_recipe_from_contained_surface(
    const ResidentLanguageRecipeOpportunityBankV1& bank,
    const std::uint32_t* heard_values, std::uint32_t heard_length,
    const RecipeCellT* cells, std::uint32_t cell_count,
    const DerivationT* derivations, std::uint32_t derivation_count,
    std::uint64_t candidate_occurrence_identity, ResidentRecipeOccurrence* out) {
  if (out == nullptr || heard_values == nullptr || heard_length == 0u ||
      candidate_occurrence_identity == 0u)
    return false;
  const std::uint32_t count =
      bank.count < kResidentLanguageRecipeOpportunityCapacity
          ? bank.count
          : kResidentLanguageRecipeOpportunityCapacity;
  const ResidentLanguageRecipeOpportunityV1* row = nullptr;
  std::uint32_t row_index = 0u;
  for (std::uint32_t i = 0u; i < count; ++i) {
    const auto& candidate = bank.rows[i];
    if (candidate.surface_length == 0u || candidate.surface_length > heard_length)
      continue;
    bool contained = false;
    for (std::uint32_t begin = 0u;
         begin + candidate.surface_length <= heard_length; ++begin) {
      if (direct_network::surface_ecology_content_identity(
              candidate.surface_length, heard_values + begin) ==
          candidate.surface_identity) {
        contained = true;
        break;
      }
    }
    if (!contained) continue;
    if (row != nullptr && row->identity != candidate.identity) return false;
    row = &candidate;
    row_index = i;
  }
  if (row == nullptr) return false;
  return nominate_resident_language_recipe_opportunity(
      *row, row->surface_identity, cells, cell_count, derivations,
      derivation_count, candidate_occurrence_identity + row_index, out);
}

}  // namespace substrate::direct_adult_core

#endif
