#ifndef HARDWARE_NATIVE_DIRECT_HIGH_RANK_HOT_COLD_SUPPORT_CUH
#define HARDWARE_NATIVE_DIRECT_HIGH_RANK_HOT_COLD_SUPPORT_CUH

#include <cstdint>
#include <string>

#include "hardware_native/direct_boundary_condensation.cuh"
#include "hardware_native/direct_network_recipe_abi.cuh"

namespace substrate::direct_adult_core {

inline constexpr std::uint64_t kHighRankSupportMagic = 0x4852535550505431ull;
inline constexpr std::uint32_t kHighRankSupportVersion = 1u;

enum class HighRankSupportRequest : std::uint32_t {
  none = 0u,
  evict = 1u,
  restore = 2u,
};
enum class HighRankColdStatus : std::uint32_t {
  stored = 0u,
  loaded,
  capacity_exhausted,
  invalid_request,
  io_error,
  content_mismatch,
};

struct DirectHighRankSupportPageV1 {
  std::uint64_t magic;
  std::uint64_t logical_recipe_id;
  std::uint64_t revision_identity;
  std::uint64_t derivation_rank;
  ResidentBoundaryCondensationWitness witness;
  direct_network::ResidentRecipeDerivation derivation;
  std::uint32_t version;
  std::uint32_t object_bytes;
};
struct DirectHighRankHotColdStateV1 {
  std::uint64_t logical_recipe_id;
  std::uint64_t revision_identity;
  std::uint64_t derivation_rank;
  direct_network::recipe::Root256 requested_address;
  std::uint64_t support_capacity_bytes;
  std::uint64_t reclaimed_support_bytes;
  std::uint64_t transition_count;
  HighRankSupportRequest request;
  std::uint32_t support_present;
  std::uint32_t version;
  std::uint32_t reserved;
};
static_assert(std::is_trivially_copyable_v<DirectHighRankSupportPageV1> &&
              std::is_standard_layout_v<DirectHighRankSupportPageV1>);
static_assert(std::is_trivially_copyable_v<DirectHighRankHotColdStateV1> &&
              std::is_standard_layout_v<DirectHighRankHotColdStateV1>);

DIRECT_ADULT_HD inline bool high_rank_root_zero(const direct_network::recipe::Root256& root) {
  for (std::uint32_t i = 0u; i < 8u; ++i)
    if (root.word[i] != 0u)
      return false;
  return true;
}
DIRECT_ADULT_HD inline direct_network::recipe::Root256 high_rank_support_address(
    const DirectHighRankSupportPageV1& page) {
  return direct_network::recipe::content_root(&page, sizeof(page));
}
DIRECT_ADULT_HD inline bool high_rank_support_page_valid(const DirectHighRankSupportPageV1& page) {
  return page.magic == kHighRankSupportMagic && page.version == kHighRankSupportVersion &&
         page.object_bytes == sizeof(page) && page.logical_recipe_id != 0u &&
         page.revision_identity != 0u && page.derivation_rank != 0u &&
         page.derivation.logical_recipe_id == page.logical_recipe_id &&
         page.derivation.revision_identity == page.revision_identity &&
         page.derivation.generation == page.derivation_rank &&
         page.derivation.witness_identity == page.witness.witness_identity &&
         resident_boundary_condensation_witness_complete(page.witness);
}
DIRECT_ADULT_HD inline bool make_high_rank_support_page(
    const direct_network::ResidentRecipeCell& recipe,
    const direct_network::ResidentRecipeDerivation& derivation,
    const ResidentBoundaryCondensationWitness& witness, DirectHighRankSupportPageV1* page) {
  if (page == nullptr || recipe.logical_recipe_id == 0u ||
      recipe.logical_recipe_id != derivation.logical_recipe_id ||
      recipe.revision_identity != derivation.revision_identity || derivation.generation == 0u ||
      derivation.witness_identity != witness.witness_identity ||
      !resident_boundary_condensation_witness_complete(witness))
    return false;
  DirectHighRankSupportPageV1 candidate{};
  candidate.magic = kHighRankSupportMagic;
  candidate.logical_recipe_id = recipe.logical_recipe_id;
  candidate.revision_identity = recipe.revision_identity;
  candidate.derivation_rank = derivation.generation;
  candidate.witness = witness;
  candidate.derivation = derivation;
  candidate.version = kHighRankSupportVersion;
  candidate.object_bytes = sizeof(candidate);
  *page = candidate;
  return true;
}
DIRECT_ADULT_HD inline bool initialize_high_rank_hot_cold_state(
    const direct_network::ResidentRecipeCell& recipe, std::uint64_t derivation_rank,
    std::uint64_t capacity_bytes, DirectHighRankHotColdStateV1* state) {
  if (state == nullptr || recipe.logical_recipe_id == 0u || recipe.revision_identity == 0u ||
      derivation_rank == 0u || capacity_bytes < sizeof(DirectHighRankSupportPageV1))
    return false;
  *state = {};
  state->logical_recipe_id = recipe.logical_recipe_id;
  state->revision_identity = recipe.revision_identity;
  state->derivation_rank = derivation_rank;
  state->support_capacity_bytes = capacity_bytes;
  state->support_present = 1u;
  state->version = kHighRankSupportVersion;
  return true;
}
DIRECT_ADULT_HD inline bool request_high_rank_support_eviction(
    const direct_network::ResidentRecipeCell& hot_recipe, const DirectHighRankSupportPageV1& page,
    DirectHighRankHotColdStateV1* state) {
  if (state == nullptr || !high_rank_support_page_valid(page) ||
      state->version != kHighRankSupportVersion || state->support_present != 1u ||
      state->request != HighRankSupportRequest::none ||
      hot_recipe.logical_recipe_id != state->logical_recipe_id ||
      hot_recipe.revision_identity != state->revision_identity ||
      page.logical_recipe_id != state->logical_recipe_id ||
      page.revision_identity != state->revision_identity ||
      state->support_capacity_bytes < sizeof(page))
    return false;
  state->requested_address = high_rank_support_address(page);
  state->request = HighRankSupportRequest::evict;
  return !high_rank_root_zero(state->requested_address);
}
DIRECT_ADULT_HD inline bool commit_high_rank_support_eviction(
    const direct_network::ResidentRecipeCell& hot_recipe,
    const direct_network::recipe::Root256& persisted_address, DirectHighRankSupportPageV1* page,
    DirectHighRankHotColdStateV1* state) {
  if (page == nullptr || state == nullptr || state->request != HighRankSupportRequest::evict ||
      persisted_address != state->requested_address ||
      persisted_address != high_rank_support_address(*page) ||
      hot_recipe.logical_recipe_id != state->logical_recipe_id ||
      hot_recipe.revision_identity != state->revision_identity)
    return false;
  *page = {};
  state->support_present = 0u;
  state->reclaimed_support_bytes = sizeof(DirectHighRankSupportPageV1);
  state->request = HighRankSupportRequest::none;
  ++state->transition_count;
  return true;
}
DIRECT_ADULT_HD inline bool request_high_rank_support_restore(DirectHighRankHotColdStateV1* state) {
  if (state == nullptr || state->support_present != 0u ||
      state->request != HighRankSupportRequest::none ||
      high_rank_root_zero(state->requested_address))
    return false;
  state->request = HighRankSupportRequest::restore;
  return true;
}
DIRECT_ADULT_HD inline bool commit_high_rank_support_restore(
    const direct_network::ResidentRecipeCell& hot_recipe, const DirectHighRankSupportPageV1& staged,
    DirectHighRankSupportPageV1* page, DirectHighRankHotColdStateV1* state) {
  if (page == nullptr || state == nullptr || state->request != HighRankSupportRequest::restore ||
      !high_rank_support_page_valid(staged) ||
      high_rank_support_address(staged) != state->requested_address ||
      staged.logical_recipe_id != state->logical_recipe_id ||
      staged.revision_identity != state->revision_identity ||
      hot_recipe.logical_recipe_id != state->logical_recipe_id ||
      hot_recipe.revision_identity != state->revision_identity)
    return false;
  *page = staged;
  state->support_present = 1u;
  state->reclaimed_support_bytes = 0u;
  state->request = HighRankSupportRequest::none;
  ++state->transition_count;
  return true;
}

HighRankColdStatus store_high_rank_support_page(const char* directory,
                                                const DirectHighRankHotColdStateV1& request,
                                                const DirectHighRankSupportPageV1& page,
                                                std::string* object_path);
HighRankColdStatus load_high_rank_support_page(const char* directory,
                                               const DirectHighRankHotColdStateV1& request,
                                               DirectHighRankSupportPageV1* page,
                                               std::string* object_path);

}  // namespace substrate::direct_adult_core

#endif
