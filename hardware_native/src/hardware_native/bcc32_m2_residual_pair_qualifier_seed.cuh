#pragma once

// Pair qualification for the two residual projection arms. The existing
// residual corner locks remain owned by the projection seed; this seed adds
// only the three persistent pair-splitter locks per arm.

#include <array>
#include <cstddef>
#include <cstdint>

#include "bcc32_m2_residual_vacancy_projection_seed.cuh"

namespace substrate::bcc32 {

inline constexpr std::uint32_t kM2ResidualPairQualifierIncoming = 3u;
inline constexpr std::uint32_t kM2ResidualPairQualifierDiverted = 1u;
inline constexpr std::size_t kM2ResidualPairQualifierLocksPerArm = 3u;
inline constexpr std::size_t kM2ResidualPairQualifierArmCount = 2u;
inline constexpr std::size_t kM2ResidualPairQualifierSeedSiteCount =
    kM2ResidualPairQualifierLocksPerArm * kM2ResidualPairQualifierArmCount;

constexpr Int3 m2_residual_pair_qualifier_center(
    ProcessivePredictionComparatorArm arm) {
  const Int3 projection = m2_residual_projection_center(arm);
  const Int3 direction3 = direction_offset(Direction::positive_u3);
  return {projection.x + 2 * direction3.x, projection.y + 2 * direction3.y,
          projection.z + 2 * direction3.z};
}

constexpr Int3 m2_residual_pair_qualifier_upstream(
    ProcessivePredictionComparatorArm arm) {
  const Int3 center = m2_residual_pair_qualifier_center(arm);
  const CarrierPairSplitterOffset offset = carrier_pair_splitter_offset(
      kM2ResidualPairQualifierIncoming, kM2ResidualPairQualifierDiverted, 1u);
  return {center.x + offset.x, center.y + offset.y, center.z + offset.z};
}

constexpr Int3 m2_residual_pair_qualifier_branch(
    ProcessivePredictionComparatorArm arm) {
  const Int3 upstream = m2_residual_pair_qualifier_upstream(arm);
  const Int3 diverted = direction_offset(
      static_cast<Direction>(kM2ResidualPairQualifierDiverted));
  return {upstream.x + diverted.x, upstream.y + diverted.y,
          upstream.z + diverted.z};
}

constexpr DevelopmentalSeedSite m2_residual_pair_qualifier_lock(
    ProcessivePredictionComparatorArm arm, std::uint32_t index) {
  const Int3 center = m2_residual_pair_qualifier_center(arm);
  const CarrierPairSplitterOffset offset = carrier_pair_splitter_offset(
      kM2ResidualPairQualifierIncoming, kM2ResidualPairQualifierDiverted,
      index);
  return {static_cast<std::int8_t>(center.x + offset.x),
          static_cast<std::int8_t>(center.y + offset.y),
          static_cast<std::int8_t>(center.z + offset.z),
          carrier_pair_splitter_word(kM2ResidualPairQualifierIncoming,
                                      kM2ResidualPairQualifierDiverted, index,
                                      false)};
}

constexpr std::array<DevelopmentalSeedSite,
                     kM2ResidualPairQualifierSeedSiteCount>
m2_residual_pair_qualifier_seed() {
  std::array<DevelopmentalSeedSite,
             kM2ResidualPairQualifierSeedSiteCount>
      result{};
  std::size_t next = 0u;
  for (std::uint32_t arm_index = 0u;
       arm_index < kM2ResidualPairQualifierArmCount; ++arm_index) {
    const auto arm =
        static_cast<ProcessivePredictionComparatorArm>(arm_index);
    for (std::uint32_t index = 2u;
         index < kCarrierPairSplitterSiteCount; ++index)
      result[next++] = m2_residual_pair_qualifier_lock(arm, index);
  }
  return result;
}

constexpr bool m2_residual_pair_qualifier_sites_unique() {
  constexpr auto seed = m2_residual_pair_qualifier_seed();
  for (std::size_t left = 0u; left < seed.size(); ++left)
    for (std::size_t right = left + 1u; right < seed.size(); ++right)
      if (seed[left].x == seed[right].x && seed[left].y == seed[right].y &&
          seed[left].z == seed[right].z)
        return false;
  return true;
}

static_assert(m2_residual_pair_qualifier_center(
                  ProcessivePredictionComparatorArm::positive) ==
              Int3{-18, -2, -6});
static_assert(m2_residual_pair_qualifier_center(
                  ProcessivePredictionComparatorArm::negative) ==
              Int3{14, -2, -6});
static_assert(m2_residual_pair_qualifier_branch(
                  ProcessivePredictionComparatorArm::positive) ==
              Int3{-17, 0, -5});
static_assert(m2_residual_pair_qualifier_branch(
                  ProcessivePredictionComparatorArm::negative) ==
              Int3{15, 0, -5});
static_assert(m2_residual_pair_qualifier_sites_unique());

}  // namespace substrate::bcc32
