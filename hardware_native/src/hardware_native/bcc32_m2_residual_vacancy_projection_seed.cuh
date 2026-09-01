#pragma once

// Collision-free resident projection for the two comparator residuals. The
// seed supplies only persistent carrier-corner locks. A comparator-produced
// P-2 vacancy must arrive at the unseeded center before ordinary F can turn it
// onto the arm's outward road.

#include <array>
#include <cstddef>
#include <cstdint>

#include "bcc32_processive_prediction_comparator_seed.cuh"

namespace substrate::bcc32 {

inline constexpr std::uint32_t kM2ResidualProjectionIncoming = 6u;
inline constexpr std::size_t kM2ResidualProjectionArmCount = 2u;
inline constexpr std::size_t kM2ResidualProjectionLocksPerArm = 2u;
inline constexpr std::size_t kM2ResidualProjectionSeedSiteCount =
    kM2ResidualProjectionArmCount * kM2ResidualProjectionLocksPerArm;

constexpr std::uint32_t m2_residual_projection_outgoing(
    ProcessivePredictionComparatorArm) {
  return 3u;
}

constexpr Int3 m2_residual_projection_center(
    ProcessivePredictionComparatorArm arm) {
  return processive_prediction_comparator_error_port(arm) +
         direction_offset(Direction::negative_u2);
}

constexpr DevelopmentalSeedSite m2_residual_projection_lock(
    ProcessivePredictionComparatorArm arm, std::uint32_t index) {
  const Int3 center = m2_residual_projection_center(arm);
  const std::uint32_t outgoing = m2_residual_projection_outgoing(arm);
  const CarrierCornerOffset offset =
      carrier_corner_offset(kM2ResidualProjectionIncoming, outgoing, index);
  return {
      static_cast<std::int8_t>(center.x + offset.x),
      static_cast<std::int8_t>(center.y + offset.y),
      static_cast<std::int8_t>(center.z + offset.z),
      carrier_corner_word(kM2ResidualProjectionIncoming, outgoing, index, false),
  };
}

constexpr std::array<DevelopmentalSeedSite,
                     kM2ResidualProjectionSeedSiteCount>
m2_residual_vacancy_projection_seed() {
  std::array<DevelopmentalSeedSite, kM2ResidualProjectionSeedSiteCount> result{};
  std::size_t next = 0u;
  for (std::uint32_t arm_index = 0u;
       arm_index < kM2ResidualProjectionArmCount; ++arm_index) {
    const auto arm =
        static_cast<ProcessivePredictionComparatorArm>(arm_index);
    for (std::uint32_t index = 1u; index < kCarrierCornerSiteCount; ++index)
      result[next++] = m2_residual_projection_lock(arm, index);
  }
  return result;
}

constexpr bool m2_residual_projection_sites_unique() {
  constexpr auto seed = m2_residual_vacancy_projection_seed();
  for (std::size_t left = 0u; left < seed.size(); ++left)
    for (std::size_t right = left + 1u; right < seed.size(); ++right)
      if (seed[left].x == seed[right].x &&
          seed[left].y == seed[right].y &&
          seed[left].z == seed[right].z)
        return false;
  return true;
}

static_assert(m2_residual_projection_center(
                  ProcessivePredictionComparatorArm::positive) ==
              Int3{-16, 0, -4});
static_assert(m2_residual_projection_center(
                  ProcessivePredictionComparatorArm::negative) ==
              Int3{16, 0, -4});
static_assert(m2_residual_projection_sites_unique());

}  // namespace substrate::bcc32
