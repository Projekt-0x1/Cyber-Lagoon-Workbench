#pragma once

// Two one-sided processive cells store the predicted binary state. Observation
// one contacts only the positive arm and observation zero only the negative
// arm. A mismatch changes that arm and exits through its spatially distinct
// waste port; a match preserves the stored state and uses the neutral path.

#include <array>
#include <cstdint>

#include "bcc32_processive_weight_region_seed.cuh"

namespace substrate::bcc32 {

using ProcessivePredictionComparatorSeedHash = std::uint32_t;

inline constexpr ProcessivePredictionComparatorSeedHash kProcessivePredictionComparatorSeedHash =
    make_processive_weight_region_seed_hash(1u, 0u, 1u, 2u, kProcessiveWeightGene);
inline constexpr std::uint32_t kProcessivePredictionComparatorSeedSiteCount =
    2u * kProcessiveWeightSitesPerCell;

enum class ProcessivePredictionComparatorArm : std::uint32_t {
  positive = 0u,
  negative = 1u,
};

constexpr Int3 processive_prediction_comparator_origin(ProcessivePredictionComparatorArm arm) {
  return arm == ProcessivePredictionComparatorArm::positive ? Int3{-16, 0, 0} : Int3{16, 0, 0};
}

constexpr std::array<DevelopmentalSeedSite, kProcessiveWeightSitesPerCell>
processive_prediction_comparator_cell_seed(Int3 origin) {
  std::array<DevelopmentalSeedSite, kProcessiveWeightSitesPerCell> result{};
  const auto region = processive_weight_region_seed(kProcessivePredictionComparatorSeedHash);
  for (std::uint32_t index = 0u; index < result.size(); ++index) {
    result[index] = {static_cast<std::int8_t>(origin.x + region[index].x),
                     static_cast<std::int8_t>(origin.y + region[index].y),
                     static_cast<std::int8_t>(origin.z + region[index].z), region[index].word};
  }
  return result;
}

constexpr std::array<DevelopmentalSeedSite, kProcessivePredictionComparatorSeedSiteCount>
processive_prediction_comparator_seed(ProcessivePredictionComparatorSeedHash hash) {
  std::array<DevelopmentalSeedSite, kProcessivePredictionComparatorSeedSiteCount> result{};
  if (hash != kProcessivePredictionComparatorSeedHash)
    return result;
  for (std::uint32_t arm = 0u; arm < 2u; ++arm) {
    const auto cell =
        processive_prediction_comparator_cell_seed(processive_prediction_comparator_origin(
            static_cast<ProcessivePredictionComparatorArm>(arm)));
    for (std::uint32_t index = 0u; index < cell.size(); ++index)
      result[arm * cell.size() + index] = cell[index];
  }
  return result;
}

constexpr Int3 processive_prediction_comparator_body(ProcessivePredictionComparatorArm arm) {
  return processive_prediction_comparator_origin(arm);
}

constexpr std::uint32_t processive_prediction_comparator_lane(bool value) {
  const std::uint32_t path = processive_weight_path(kProcessivePredictionComparatorSeedHash);
  return value ? path : path + 4u;
}

constexpr Int3 processive_prediction_comparator_error_port(ProcessivePredictionComparatorArm arm) {
  const Int3 origin = processive_prediction_comparator_origin(arm);
  const Int3 ray = direction_offset(
      static_cast<Direction>(processive_weight_waste(kProcessivePredictionComparatorSeedHash)));
  return {origin.x - 3 * ray.x, origin.y - 3 * ray.y, origin.z - 3 * ray.z};
}

constexpr Int3 processive_prediction_comparator_match_port(bool value) {
  const ProcessivePredictionComparatorArm arm = value ? ProcessivePredictionComparatorArm::positive
                                                      : ProcessivePredictionComparatorArm::negative;
  const Int3 origin = processive_prediction_comparator_origin(arm);
  const Int3 ray =
      direction_offset(static_cast<Direction>(processive_prediction_comparator_lane(value)));
  return {origin.x + 3 * ray.x, origin.y + 3 * ray.y, origin.z + 3 * ray.z};
}

static_assert(valid_processive_weight_region_hash(kProcessivePredictionComparatorSeedHash));
static_assert(processive_weight_length(kProcessivePredictionComparatorSeedHash) == 1u);
static_assert(processive_prediction_comparator_lane(true) == 1u);
static_assert(processive_prediction_comparator_lane(false) == 5u);
static_assert(
    processive_prediction_comparator_error_port(ProcessivePredictionComparatorArm::positive).x ==
    -16);
static_assert(
    processive_prediction_comparator_error_port(ProcessivePredictionComparatorArm::negative).x ==
    16);

}  // namespace substrate::bcc32
