#pragma once

// Two complementary processive prediction sources expose either binary value
// on the same negative carrier lane. The first source is founded at zero; the
// translated second source is founded at one. Fixed reciprocal marker
// exchanges change both sources together, so no host-selected route is needed.

#include <array>
#include <cstdint>

#include "bcc32_processive_prediction_query_launcher_seed.cuh"

namespace substrate::bcc32 {

inline constexpr Int3 kDualNegativePredictionOneOffset{32, 0, 0};
inline constexpr std::uint32_t kDualNegativePredictionTrainingTicks = 1u;
inline constexpr std::int32_t kDualNegativePredictionReceptorDistance = 2;
inline constexpr std::uint32_t kDualNegativePredictionSourceSeedSiteCount =
    2u * kProcessiveWeightSitesPerCell;
inline constexpr std::uint32_t kDualNegativePredictionLauncherSeedSiteCount =
    2u * kProcessivePredictionQueryLauncherSeedSiteCount;
inline constexpr std::uint32_t kDualNegativePredictionReceptorSeedSiteCount =
    2u;

constexpr Int3 dual_negative_prediction_translate(Int3 value, bool one_source) {
  if (!one_source)
    return value;
  return {
      value.x + kDualNegativePredictionOneOffset.x,
      value.y + kDualNegativePredictionOneOffset.y,
      value.z + kDualNegativePredictionOneOffset.z,
  };
}

constexpr std::array<DevelopmentalSeedSite,
                     kDualNegativePredictionSourceSeedSiteCount>
dual_negative_prediction_source_seed() {
  std::array<DevelopmentalSeedSite,
             kDualNegativePredictionSourceSeedSiteCount>
      result{};
  const auto source = processive_prediction_projection_seed();
  for (std::uint32_t copy = 0u; copy < 2u; ++copy) {
    const bool one_source = copy != 0u;
    for (std::uint32_t index = 0u;
         index < kProcessiveWeightSitesPerCell; ++index) {
      const Int3 translated = dual_negative_prediction_translate(
          {source[index].x, source[index].y, source[index].z}, one_source);
      SiteWord word = source[index].word;
      if (one_source &&
          index ==
              kProcessivePredictionProjectionThresholdIndex *
                  kProcessiveWeightSitesPerCell) {
        word =
            processive_weight_one_word(kProcessivePredictionProjectionSeedHash);
      }
      result[copy * kProcessiveWeightSitesPerCell + index] = {
          static_cast<std::int8_t>(translated.x),
          static_cast<std::int8_t>(translated.y),
          static_cast<std::int8_t>(translated.z),
          word,
      };
    }
  }
  return result;
}

constexpr std::array<DevelopmentalSeedSite,
                     kDualNegativePredictionLauncherSeedSiteCount>
dual_negative_prediction_launcher_seed(
    ProcessivePredictionQueryLauncherSeedHash hash) {
  std::array<DevelopmentalSeedSite,
             kDualNegativePredictionLauncherSeedSiteCount>
      result{};
  const auto source = processive_prediction_query_launcher_seed(hash);
  for (std::uint32_t copy = 0u; copy < 2u; ++copy) {
    const bool one_source = copy != 0u;
    for (std::uint32_t index = 0u; index < source.size(); ++index) {
      const Int3 translated = dual_negative_prediction_translate(
          {source[index].x, source[index].y, source[index].z},
          one_source);
      result[copy * source.size() + index] = {
          static_cast<std::int8_t>(translated.x),
          static_cast<std::int8_t>(translated.y),
          static_cast<std::int8_t>(translated.z),
          source[index].word,
      };
    }
  }
  return result;
}

constexpr Int3 dual_negative_prediction_body(bool one_source,
                                             std::uint32_t index) {
  return dual_negative_prediction_translate(
      processive_prediction_projection_body(index), one_source);
}

constexpr Int3 dual_negative_prediction_training_endpoint(bool one_source) {
  return dual_negative_prediction_body(
      one_source, kProcessivePredictionProjectionThresholdIndex);
}

constexpr Int3 dual_negative_prediction_receptor(bool one_source) {
  const Int3 source = dual_negative_prediction_body(
      one_source, kProcessivePredictionProjectionThresholdIndex);
  const std::uint32_t lane =
      kProcessivePredictionProjectionProbeBasis + 4u;
  const Int3 ray = direction_offset(static_cast<Direction>(lane));
  return {
      source.x + kDualNegativePredictionReceptorDistance * ray.x,
      source.y + kDualNegativePredictionReceptorDistance * ray.y,
      source.z + kDualNegativePredictionReceptorDistance * ray.z,
  };
}

constexpr Int3 dual_negative_prediction_energy_source(bool one_source) {
  const Int3 receptor = dual_negative_prediction_receptor(one_source);
  const std::uint32_t lane =
      kProcessivePredictionProjectionProbeBasis + 4u;
  const Int3 ray = direction_offset(static_cast<Direction>(lane));
  return {
      receptor.x + ray.x,
      receptor.y + ray.y,
      receptor.z + ray.z,
  };
}

constexpr std::array<DevelopmentalSeedSite,
                     kDualNegativePredictionReceptorSeedSiteCount>
dual_negative_prediction_receptor_seed() {
  std::array<DevelopmentalSeedSite,
             kDualNegativePredictionReceptorSeedSiteCount>
      result{};
  for (std::uint32_t copy = 0u; copy < result.size(); ++copy) {
    const Int3 receptor =
        dual_negative_prediction_receptor(copy != 0u);
    result[copy] = {
        static_cast<std::int8_t>(receptor.x),
        static_cast<std::int8_t>(receptor.y),
        static_cast<std::int8_t>(receptor.z),
        static_cast<SiteWord>(
            kQ | channel_bit(kReactiveShift, 1u)),
    };
  }
  return result;
}

static_assert(dual_negative_prediction_body(false, 0u).x == 0);
static_assert(dual_negative_prediction_body(true, 0u).x == 32);
static_assert(dual_negative_prediction_body(false, 0u).z == 32);
static_assert(dual_negative_prediction_body(true, 0u).z == 32);
static_assert(dual_negative_prediction_receptor(false).y == -2);
static_assert(dual_negative_prediction_receptor(true).y == -2);
static_assert(dual_negative_prediction_energy_source(false).y == -3);
static_assert(dual_negative_prediction_energy_source(true).y == -3);

}  // namespace substrate::bcc32
