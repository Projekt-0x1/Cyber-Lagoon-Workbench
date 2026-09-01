#pragma once

// One bounded M2 founder geometry.  The two comparator error outlets each
// face a resident, pre-armed signed route one BCC edge away.  The founder only
// names geometry and generic route chemistry; it contains no route key or
// selected residual sign.

#include <array>
#include <cstddef>
#include <cstdint>

#include "bcc32_processive_prediction_comparator_seed.cuh"

namespace substrate::bcc32 {

// The comparator weight marker occupies basis 1 and its residual uses the
// waste basis. Eligibility must occupy a third basis or the signed-synapse
// gate rejects the molecule before evaluating credit.
inline constexpr std::uint32_t kPhysicalResidualEligibilityBasis = 3u;
inline constexpr Direction kPhysicalResidualIngress = Direction::positive_u0;
inline constexpr std::size_t kPhysicalResidualRouteCount = 2u;
inline constexpr std::size_t kPhysicalResidualRouteSeedSiteCount =
    kProcessivePredictionComparatorSeedSiteCount + kPhysicalResidualRouteCount;

constexpr Int3 physical_residual_route_target(ProcessivePredictionComparatorArm arm) {
  return processive_prediction_comparator_error_port(arm) +
         direction_offset(kPhysicalResidualIngress);
}

// An active signed synapse with one route-local negative credit hole.  A
// second negative hole on another basis would depress it under the ordinary
// differentiated signed-synapse gate.  The comparator must supply that second
// hole physically; this header never writes it.
constexpr SiteWord physical_residual_route_eligible_word() {
  return processive_weight_one_word(kProcessivePredictionComparatorSeedHash) ^
         carrier_bit(kPhysicalResidualEligibilityBasis + 4u);
}

static_assert(kPhysicalResidualEligibilityBasis !=
              processive_weight_marker(kProcessivePredictionComparatorSeedHash));
static_assert(kPhysicalResidualEligibilityBasis !=
              processive_weight_waste(kProcessivePredictionComparatorSeedHash));

constexpr std::array<DevelopmentalSeedSite, kPhysicalResidualRouteSeedSiteCount>
physical_residual_route_seed() {
  std::array<DevelopmentalSeedSite, kPhysicalResidualRouteSeedSiteCount> result{};
  const auto comparator =
      processive_prediction_comparator_seed(kProcessivePredictionComparatorSeedHash);
  for (std::size_t index = 0; index < comparator.size(); ++index)
    result[index] = comparator[index];
  for (std::uint32_t arm = 0; arm < kPhysicalResidualRouteCount; ++arm) {
    const Int3 target = physical_residual_route_target(
        static_cast<ProcessivePredictionComparatorArm>(arm));
    result[comparator.size() + arm] = {static_cast<std::int8_t>(target.x),
                                       static_cast<std::int8_t>(target.y),
                                       static_cast<std::int8_t>(target.z),
                                       physical_residual_route_eligible_word()};
  }
  return result;
}

}  // namespace substrate::bcc32
