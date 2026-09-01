#pragma once

// A two-arm, source-only history latch immediately downstream of the canonical
// comparator residual ports.  Each arm turns the comparator's P-2 vacancy
// through the existing P3 carrier corner and into a one-cell processive body.
// Its ordinary 0 -> 1 transition is the retained mismatch state.  The founder
// supplies only geometry; it has no outcome, clock, counter, or host callback.

#include <array>
#include <cstddef>
#include <cstdint>

#include "bcc32_m2_residual_vacancy_projection_seed.cuh"

namespace substrate::bcc32 {

inline constexpr ProcessiveWeightRegionSeedHash kM2EndpointHistoryLatchCellHash =
    make_processive_weight_region_seed_hash(
        1u, 0u, 3u, 2u, kProcessiveWeightGene);
inline constexpr std::size_t kM2EndpointHistoryLatchArms = 2u;
inline constexpr std::size_t kM2EndpointHistoryLatchLocksPerArm =
    kCarrierCornerSiteCount - 1u;
inline constexpr std::size_t kM2EndpointHistoryLatchCellSites =
    kProcessiveWeightSitesPerCell;
inline constexpr std::size_t kM2EndpointHistoryLatchSeedSiteCount =
    kM2EndpointHistoryLatchArms *
    (kM2EndpointHistoryLatchLocksPerArm + kM2EndpointHistoryLatchCellSites);

constexpr Int3 m2_endpoint_history_latch_center(
    ProcessivePredictionComparatorArm arm) {
  return m2_residual_projection_center(arm);
}

constexpr Int3 m2_endpoint_history_latch_body(
    ProcessivePredictionComparatorArm arm) {
  return m2_endpoint_history_latch_center(arm) +
         direction_offset(Direction::positive_u3);
}

constexpr DevelopmentalSeedSite m2_endpoint_history_latch_lock(
    ProcessivePredictionComparatorArm arm, std::uint32_t index) {
  return m2_residual_projection_lock(arm, index);
}

constexpr DevelopmentalSeedSite m2_endpoint_history_latch_cell_site(
    ProcessivePredictionComparatorArm arm, std::uint32_t index) {
  const auto cell = processive_weight_region_seed(kM2EndpointHistoryLatchCellHash);
  const Int3 body = m2_endpoint_history_latch_body(arm);
  return {
      static_cast<std::int8_t>(body.x + cell[index].x),
      static_cast<std::int8_t>(body.y + cell[index].y),
      static_cast<std::int8_t>(body.z + cell[index].z),
      cell[index].word,
  };
}

constexpr std::array<DevelopmentalSeedSite, kM2EndpointHistoryLatchSeedSiteCount>
m2_endpoint_history_latch_seed() {
  std::array<DevelopmentalSeedSite, kM2EndpointHistoryLatchSeedSiteCount> result{};
  std::size_t next = 0u;
  for (std::uint32_t arm_index = 0u; arm_index < kM2EndpointHistoryLatchArms;
       ++arm_index) {
    const auto arm = static_cast<ProcessivePredictionComparatorArm>(arm_index);
    for (std::uint32_t index = 1u; index < kCarrierCornerSiteCount; ++index)
      result[next++] = m2_endpoint_history_latch_lock(arm, index);
    for (std::uint32_t index = 0u; index < kM2EndpointHistoryLatchCellSites;
         ++index)
      result[next++] = m2_endpoint_history_latch_cell_site(arm, index);
  }
  return result;
}

constexpr bool m2_endpoint_history_latch_sites_unique() {
  constexpr auto seed = m2_endpoint_history_latch_seed();
  for (std::size_t left = 0u; left < seed.size(); ++left)
    for (std::size_t right = left + 1u; right < seed.size(); ++right)
      if (seed[left].x == seed[right].x && seed[left].y == seed[right].y &&
          seed[left].z == seed[right].z)
        return false;
  return true;
}

static_assert(valid_processive_weight_region_hash(kM2EndpointHistoryLatchCellHash));
static_assert(processive_weight_length(kM2EndpointHistoryLatchCellHash) == 1u);
static_assert(processive_weight_path(kM2EndpointHistoryLatchCellHash) == 3u);
static_assert(m2_endpoint_history_latch_body(
                  ProcessivePredictionComparatorArm::positive) ==
              Int3{-17, -1, -5});
static_assert(m2_endpoint_history_latch_body(
                  ProcessivePredictionComparatorArm::negative) ==
              Int3{15, -1, -5});
static_assert(m2_endpoint_history_latch_sites_unique());

}  // namespace substrate::bcc32
