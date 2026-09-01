#pragma once

// A resident processive scalar weight projects one binary prediction without
// exposing its state to the transaction controller. The same three-hole query
// reaches the threshold cell for both values. Its durable body sends the
// resulting hole to one of two spatially distinct rails.

#include <array>
#include <cstdint>

#include "bcc32_processive_weight_region_seed.cuh"

namespace substrate::bcc32 {

inline constexpr ProcessiveWeightRegionSeedHash kProcessivePredictionProjectionSeedHash =
    kProcessiveWeightRegionSeedHash;
inline constexpr std::uint32_t kProcessivePredictionProjectionThresholdIndex = 0u;
inline constexpr std::uint32_t kProcessivePredictionProjectionProbeBasis = 1u;
inline constexpr Int3 kProcessivePredictionProjectionOrigin{0, 0, 32};
inline constexpr std::uint32_t kProcessivePredictionProjectionSeedSiteCount =
    processive_weight_length(kProcessivePredictionProjectionSeedHash) *
    kProcessiveWeightSitesPerCell;

constexpr Int3 processive_prediction_projection_translate(Int3 value) {
  return {value.x + kProcessivePredictionProjectionOrigin.x,
          value.y + kProcessivePredictionProjectionOrigin.y,
          value.z + kProcessivePredictionProjectionOrigin.z};
}

constexpr std::array<DevelopmentalSeedSite, kProcessivePredictionProjectionSeedSiteCount>
processive_prediction_projection_seed() {
  std::array<DevelopmentalSeedSite, kProcessivePredictionProjectionSeedSiteCount> result{};
  const auto source = processive_weight_region_seed(kProcessivePredictionProjectionSeedHash);
  for (std::uint32_t index = 0u; index < result.size(); ++index) {
    const Int3 translated = processive_prediction_projection_translate(
        {source[index].x, source[index].y, source[index].z});
    result[index] = {static_cast<std::int8_t>(translated.x), static_cast<std::int8_t>(translated.y),
                     static_cast<std::int8_t>(translated.z), source[index].word};
  }
  return result;
}

constexpr Int3 processive_prediction_projection_body(std::uint32_t index) {
  const auto source = processive_weight_region_seed(kProcessivePredictionProjectionSeedHash);
  const DevelopmentalSeedSite site = source[index * kProcessiveWeightSitesPerCell];
  return processive_prediction_projection_translate({site.x, site.y, site.z});
}

constexpr Int3 processive_prediction_projection_training_endpoint(bool positive) {
  const std::uint32_t length = processive_weight_length(kProcessivePredictionProjectionSeedHash);
  return processive_prediction_projection_body(positive ? 0u : length - 1u);
}

constexpr Int3 processive_prediction_projection_output(bool value) {
  const Int3 body =
      processive_prediction_projection_body(kProcessivePredictionProjectionThresholdIndex);
  const std::uint32_t lane = value ? kProcessivePredictionProjectionProbeBasis
                                   : kProcessivePredictionProjectionProbeBasis + 4u;
  const Int3 ray = direction_offset(static_cast<Direction>(lane));
  return {body.x + ray.x, body.y + ray.y, body.z + ray.z};
}

constexpr std::array<std::uint32_t, 3u> processive_prediction_projection_query_lanes() {
  std::array<std::uint32_t, 3u> result{};
  const std::uint32_t marker = processive_weight_marker(kProcessivePredictionProjectionSeedHash);
  std::uint32_t write = 0u;
  for (std::uint32_t basis = 0u; basis < 4u; ++basis)
    if (basis != marker)
      result[write++] = basis + 4u;
  return result;
}

static_assert(kProcessivePredictionProjectionThresholdIndex <
              processive_weight_length(kProcessivePredictionProjectionSeedHash));
static_assert(kProcessivePredictionProjectionProbeBasis ==
              processive_weight_path(kProcessivePredictionProjectionSeedHash));
static_assert(processive_prediction_projection_query_lanes()[0] == 5u);
static_assert(processive_prediction_projection_query_lanes()[1] == 6u);
static_assert(processive_prediction_projection_query_lanes()[2] == 7u);

}  // namespace substrate::bcc32
