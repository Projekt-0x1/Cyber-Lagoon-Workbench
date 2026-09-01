#pragma once

// Four one-cell prediction populations receive one fixed represented contact.
// For each comparator arm, the upper and lower populations store the same
// value. Geometry makes only the value-compatible carrier travel toward the
// comparator: upper P5 for zero, lower P1 for one. The sibling carrier travels
// away, so no runtime fan-out or host-selected route is required.

#include <array>
#include <cstddef>
#include <cstdint>

#include "bcc32_processive_prediction_comparator_seed.cuh"
#include "bcc32_processive_prediction_query_launcher_seed.cuh"

namespace substrate::bcc32 {

inline constexpr std::uint32_t kSameContactProjectionPopulationCount = 4u;
inline constexpr std::uint32_t kSameContactProjectionPopulationDistance = 16u;
inline constexpr std::uint32_t kSameContactProjectionQueryDistance = 10u;
inline constexpr std::uint32_t kSameContactProjectionQueryLaneCount =
    kProcessivePredictionQueryLaneCount;
inline constexpr std::size_t kSameContactProjectionPopulationSeedSiteCount =
    kSameContactProjectionPopulationCount *
    kProcessiveWeightSitesPerCell;
inline constexpr std::size_t kSameContactProjectionLauncherSeedSiteCount =
    kSameContactProjectionPopulationCount *
    kSameContactProjectionQueryLaneCount;

constexpr Int3 same_contact_projection_population_body(
    ProcessivePredictionComparatorArm arm, bool lower) {
  const Int3 comparator = processive_prediction_comparator_body(arm);
  return {comparator.x,
          comparator.y +
              (lower
                   ? -static_cast<std::int32_t>(
                         kSameContactProjectionPopulationDistance)
                   : static_cast<std::int32_t>(
                         kSameContactProjectionPopulationDistance)),
          comparator.z};
}

constexpr std::uint32_t same_contact_projection_population_index(
    ProcessivePredictionComparatorArm arm, bool lower) {
  return 2u * static_cast<std::uint32_t>(arm) +
         (lower ? 1u : 0u);
}

constexpr DevelopmentalSeedSite same_contact_projection_site(
    Int3 coordinate, SiteWord word) {
  return {static_cast<std::int8_t>(coordinate.x),
          static_cast<std::int8_t>(coordinate.y),
          static_cast<std::int8_t>(coordinate.z), word};
}

constexpr std::array<DevelopmentalSeedSite,
                     kSameContactProjectionPopulationSeedSiteCount>
same_contact_projection_population_seed() {
  std::array<DevelopmentalSeedSite,
             kSameContactProjectionPopulationSeedSiteCount>
      result{};
  std::size_t next = 0u;
  for (std::uint32_t arm_index = 0u; arm_index < 2u; ++arm_index) {
    const auto arm =
        static_cast<ProcessivePredictionComparatorArm>(arm_index);
    for (std::uint32_t lower_index = 0u;
         lower_index < 2u; ++lower_index) {
      const Int3 body = same_contact_projection_population_body(
          arm, lower_index != 0u);
      const auto cell =
          processive_prediction_comparator_cell_seed(body);
      for (const DevelopmentalSeedSite& site : cell)
        result[next++] = site;
    }
  }
  return result;
}

constexpr std::array<DevelopmentalSeedSite,
                     kSameContactProjectionLauncherSeedSiteCount>
same_contact_projection_launcher_seed() {
  std::array<DevelopmentalSeedSite,
             kSameContactProjectionLauncherSeedSiteCount>
      result{};
  std::size_t next = 0u;
  const auto query_lanes =
      processive_prediction_projection_query_lanes();
  for (std::uint32_t arm_index = 0u; arm_index < 2u; ++arm_index) {
    const auto arm =
        static_cast<ProcessivePredictionComparatorArm>(arm_index);
    for (std::uint32_t lower_index = 0u;
         lower_index < 2u; ++lower_index) {
      const Int3 body = same_contact_projection_population_body(
          arm, lower_index != 0u);
      for (std::uint32_t lane : query_lanes) {
        const Int3 ray =
            direction_offset(static_cast<Direction>(lane));
        result[next++] = same_contact_projection_site(
            {body.x -
                 static_cast<std::int32_t>(
                     kSameContactProjectionQueryDistance) *
                     ray.x,
             body.y -
                 static_cast<std::int32_t>(
                     kSameContactProjectionQueryDistance) *
                     ray.y,
             body.z -
                 static_cast<std::int32_t>(
                     kSameContactProjectionQueryDistance) *
                     ray.z},
            static_cast<SiteWord>(kQ ^ carrier_bit(lane)));
      }
    }
  }
  return result;
}

template <std::size_t Size>
constexpr bool same_contact_projection_seed_sites_unique(
    const std::array<DevelopmentalSeedSite, Size>& seed) {
  for (std::size_t left = 0u; left < seed.size(); ++left) {
    for (std::size_t right = left + 1u; right < seed.size(); ++right) {
      if (seed[left].x == seed[right].x &&
          seed[left].y == seed[right].y &&
          seed[left].z == seed[right].z)
        return false;
    }
  }
  return true;
}

static_assert(
    same_contact_projection_population_body(
        ProcessivePredictionComparatorArm::positive, false) ==
    Int3{-16, 16, 0});
static_assert(
    same_contact_projection_population_body(
        ProcessivePredictionComparatorArm::negative, true) ==
    Int3{16, -16, 0});
static_assert(same_contact_projection_seed_sites_unique(
    same_contact_projection_population_seed()));
static_assert(same_contact_projection_seed_sites_unique(
    same_contact_projection_launcher_seed()));

}  // namespace substrate::bcc32
