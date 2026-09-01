#pragma once

// Compact fixed-law eligibility hypothesis. A one-cell processive weight
// first grows its ordinary turnover candidate from a resident negative-path
// vacancy. A delayed resident positive-path vacancy then attempts to raise the
// same cell from zero to one; the candidate must return it to zero without a
// host-side clear.

#include <array>
#include <cstddef>
#include <cstdint>

#include "bcc32_processive_weight_region_seed.cuh"

namespace substrate::bcc32 {

inline constexpr ProcessiveWeightRegionSeedHash
    kCompactRouteEligibilityWeightHash =
        make_processive_weight_region_seed_hash(
            1u, 0u, 1u, 2u, kProcessiveWeightGene);
inline constexpr std::int32_t kCompactRouteEligibilityDelay = 8;
inline constexpr std::size_t kCompactRouteEligibilityBodySite = 0u;
inline constexpr std::size_t kCompactRouteEligibilityTurnoverSite =
    kProcessiveWeightSitesPerCell;
inline constexpr std::size_t kCompactRouteEligibilityCarrierSite =
    kCompactRouteEligibilityTurnoverSite + 1u;
inline constexpr std::size_t kCompactRouteEligibilitySeedSiteCount =
    kCompactRouteEligibilityCarrierSite + 1u;

constexpr Int3 compact_route_eligibility_scaled(Int3 value,
                                                std::int32_t factor) {
  return {value.x * factor, value.y * factor, value.z * factor};
}

constexpr DevelopmentalSeedSite compact_route_eligibility_site(
    Int3 coordinate, SiteWord word) {
  return {static_cast<std::int8_t>(coordinate.x),
          static_cast<std::int8_t>(coordinate.y),
          static_cast<std::int8_t>(coordinate.z), word};
}

constexpr std::array<DevelopmentalSeedSite,
                     kCompactRouteEligibilitySeedSiteCount>
compact_route_eligibility_seed() {
  std::array<DevelopmentalSeedSite,
             kCompactRouteEligibilitySeedSiteCount>
      result{};
  const auto weight =
      processive_weight_region_seed(kCompactRouteEligibilityWeightHash);
  const std::uint32_t path =
      processive_weight_path(kCompactRouteEligibilityWeightHash);
  const std::uint32_t waste =
      processive_weight_waste(kCompactRouteEligibilityWeightHash);
  const Int3 path_ray =
      direction_offset(static_cast<Direction>(path));
  const Int3 waste_ray =
      direction_offset(static_cast<Direction>(waste));

  for (std::size_t index = 0u;
       index < kProcessiveWeightSitesPerCell; ++index)
    result[index] = weight[index];

  result[kCompactRouteEligibilityBodySite].word =
      processive_weight_one_word(kCompactRouteEligibilityWeightHash) ^
      carrier_bit(path + 4u);
  result[kCompactRouteEligibilityTurnoverSite] =
      compact_route_eligibility_site(
          compact_route_eligibility_scaled(waste_ray, 10),
          kQ | face_bit(waste));
  result[kCompactRouteEligibilityCarrierSite] =
      compact_route_eligibility_site(
          compact_route_eligibility_scaled(
              path_ray, -kCompactRouteEligibilityDelay),
          kQ ^ carrier_bit(path));
  return result;
}

static_assert(
    valid_processive_weight_region_hash(
        kCompactRouteEligibilityWeightHash));
static_assert(
    processive_weight_length(kCompactRouteEligibilityWeightHash) == 1u);

}  // namespace substrate::bcc32
