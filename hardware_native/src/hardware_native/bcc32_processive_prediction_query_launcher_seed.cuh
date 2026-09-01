#pragma once

// A compact developmental hash expands two value-neutral, three-vacancy
// query packets. Geometry, rather than a host transaction, brings lanes
// 5/6/7 to the processive prediction threshold at two separated phases.

#include <array>
#include <cstdint>

#include "bcc32_processive_prediction_projection_seed.cuh"

namespace substrate::bcc32 {

using ProcessivePredictionQueryLauncherSeedHash = std::uint32_t;

inline constexpr std::uint32_t kProcessivePredictionQueryFirstDelayShift = 0u;
inline constexpr std::uint32_t kProcessivePredictionQuerySpacingShift = 5u;
inline constexpr std::uint32_t kProcessivePredictionQueryPulseCountShift = 10u;
inline constexpr std::uint32_t kProcessivePredictionQueryGeneShift = 12u;
inline constexpr std::uint32_t
    kProcessivePredictionQueryFirstDelayHighShift = 20u;
inline constexpr std::uint32_t kProcessivePredictionQueryGene = 0x6bu;
inline constexpr std::uint32_t kProcessivePredictionQueryPulseCount = 2u;
inline constexpr std::uint32_t kProcessivePredictionQueryLaneCount = 3u;
inline constexpr std::uint32_t kProcessivePredictionQueryLauncherSeedSiteCount =
    kProcessivePredictionQueryPulseCount *
    kProcessivePredictionQueryLaneCount;
inline constexpr std::uint32_t kProcessivePredictionBentQueryIncoming = 0u;
inline constexpr std::uint32_t kProcessivePredictionBentQueryOutgoingLeg = 1u;
inline constexpr std::uint32_t kProcessivePredictionBentQueryLockSiteCount =
    kProcessivePredictionQueryLaneCount * (kCarrierCornerSiteCount - 1u);
inline constexpr std::uint32_t kProcessivePredictionBentQuerySeedSiteCount =
    kProcessivePredictionBentQueryLockSiteCount +
    kProcessivePredictionQueryLauncherSeedSiteCount;

constexpr ProcessivePredictionQueryLauncherSeedHash
make_processive_prediction_query_launcher_seed_hash(
    std::uint32_t first_delay,
    std::uint32_t spacing,
    std::uint32_t pulse_count,
    std::uint32_t gene) {
  return static_cast<ProcessivePredictionQueryLauncherSeedHash>(
             first_delay & 0x1fu) |
         static_cast<ProcessivePredictionQueryLauncherSeedHash>(
             (spacing & 0x1fu)
             << kProcessivePredictionQuerySpacingShift) |
         static_cast<ProcessivePredictionQueryLauncherSeedHash>(
             (pulse_count & 0x3u)
             << kProcessivePredictionQueryPulseCountShift) |
         static_cast<ProcessivePredictionQueryLauncherSeedHash>(
             (gene & 0xffu) << kProcessivePredictionQueryGeneShift) |
         static_cast<ProcessivePredictionQueryLauncherSeedHash>(
             ((first_delay >> 5u) & 0x1u)
             << kProcessivePredictionQueryFirstDelayHighShift);
}

constexpr std::uint32_t processive_prediction_query_first_delay(
    ProcessivePredictionQueryLauncherSeedHash hash) {
  return ((hash >> kProcessivePredictionQueryFirstDelayShift) & 0x1fu) |
         (((hash >> kProcessivePredictionQueryFirstDelayHighShift) & 0x1u)
          << 5u);
}

constexpr std::uint32_t processive_prediction_query_spacing(
    ProcessivePredictionQueryLauncherSeedHash hash) {
  return (hash >> kProcessivePredictionQuerySpacingShift) & 0x1fu;
}

constexpr std::uint32_t processive_prediction_query_pulse_count(
    ProcessivePredictionQueryLauncherSeedHash hash) {
  return (hash >> kProcessivePredictionQueryPulseCountShift) & 0x3u;
}

constexpr std::uint32_t processive_prediction_query_gene(
    ProcessivePredictionQueryLauncherSeedHash hash) {
  return (hash >> kProcessivePredictionQueryGeneShift) & 0xffu;
}

constexpr bool valid_processive_prediction_query_launcher_hash(
    ProcessivePredictionQueryLauncherSeedHash hash) {
  return processive_prediction_query_gene(hash) ==
             kProcessivePredictionQueryGene &&
         processive_prediction_query_first_delay(hash) > 0u &&
         processive_prediction_query_spacing(hash) > 0u &&
         processive_prediction_query_pulse_count(hash) ==
             kProcessivePredictionQueryPulseCount;
}

constexpr std::array<DevelopmentalSeedSite,
                     kProcessivePredictionQueryLauncherSeedSiteCount>
processive_prediction_query_launcher_seed(
    ProcessivePredictionQueryLauncherSeedHash hash) {
  std::array<DevelopmentalSeedSite,
             kProcessivePredictionQueryLauncherSeedSiteCount>
      result{};
  if (!valid_processive_prediction_query_launcher_hash(hash))
    return result;

  const Int3 threshold = processive_prediction_projection_body(
      kProcessivePredictionProjectionThresholdIndex);
  const auto lanes = processive_prediction_projection_query_lanes();
  const std::uint32_t first_delay =
      processive_prediction_query_first_delay(hash);
  const std::uint32_t spacing =
      processive_prediction_query_spacing(hash);

  for (std::uint32_t pulse = 0u;
       pulse < kProcessivePredictionQueryPulseCount; ++pulse) {
    const std::int32_t distance = static_cast<std::int32_t>(
        first_delay + pulse * spacing);
    for (std::uint32_t index = 0u; index < lanes.size(); ++index) {
      const std::uint32_t lane = lanes[index];
      const Int3 ray =
          direction_offset(static_cast<Direction>(lane));
      result[pulse * lanes.size() + index] = {
          static_cast<std::int8_t>(threshold.x - distance * ray.x),
          static_cast<std::int8_t>(threshold.y - distance * ray.y),
          static_cast<std::int8_t>(threshold.z - distance * ray.z),
          static_cast<SiteWord>(kQ ^ carrier_bit(lane))};
    }
  }
  return result;
}

// A straight negative-u1 query ray crosses the mature processive bodies before
// reaching body zero.  These three persistent carrier-corner organs instead
// accept parallel positive-u0 packets on body-disjoint rays and release each
// packet one hop from the threshold on its original query lane.  Only the two
// differentiated locks and the distant packets are Genesis matter; corner
// centres and the path between them are recruited by ordinary carrier motion.
constexpr std::array<DevelopmentalSeedSite,
                     kProcessivePredictionBentQuerySeedSiteCount>
processive_prediction_bent_query_seed(
    ProcessivePredictionQueryLauncherSeedHash hash) {
  std::array<DevelopmentalSeedSite,
             kProcessivePredictionBentQuerySeedSiteCount>
      result{};
  if (!valid_processive_prediction_query_launcher_hash(hash) ||
      processive_prediction_query_first_delay(hash) <
          kProcessivePredictionBentQueryOutgoingLeg)
    return result;

  const Int3 threshold = processive_prediction_projection_body(
      kProcessivePredictionProjectionThresholdIndex);
  const auto lanes = processive_prediction_projection_query_lanes();
  const Int3 incoming = direction_offset(
      static_cast<Direction>(kProcessivePredictionBentQueryIncoming));
  for (std::uint32_t lane_index = 0u; lane_index < lanes.size();
       ++lane_index) {
    const std::uint32_t outgoing = lanes[lane_index];
    const Int3 outgoing_ray =
        direction_offset(static_cast<Direction>(outgoing));
    const Int3 center{
        threshold.x - static_cast<std::int32_t>(
                          kProcessivePredictionBentQueryOutgoingLeg) *
                          outgoing_ray.x,
        threshold.y - static_cast<std::int32_t>(
                          kProcessivePredictionBentQueryOutgoingLeg) *
                          outgoing_ray.y,
        threshold.z - static_cast<std::int32_t>(
                          kProcessivePredictionBentQueryOutgoingLeg) *
                          outgoing_ray.z};
    for (std::uint32_t lock = 1u; lock < kCarrierCornerSiteCount; ++lock) {
      const CarrierCornerOffset offset = carrier_corner_offset(
          kProcessivePredictionBentQueryIncoming, outgoing, lock);
      result[lane_index * (kCarrierCornerSiteCount - 1u) + lock - 1u] = {
          static_cast<std::int8_t>(center.x + offset.x),
          static_cast<std::int8_t>(center.y + offset.y),
          static_cast<std::int8_t>(center.z + offset.z),
          carrier_corner_word(
              kProcessivePredictionBentQueryIncoming, outgoing, lock,
              false)};
    }
  }

  const std::uint32_t first_delay =
      processive_prediction_query_first_delay(hash);
  const std::uint32_t spacing =
      processive_prediction_query_spacing(hash);
  for (std::uint32_t pulse = 0u;
       pulse < kProcessivePredictionQueryPulseCount; ++pulse) {
    const std::int32_t incoming_distance = static_cast<std::int32_t>(
        first_delay + pulse * spacing -
        kProcessivePredictionBentQueryOutgoingLeg);
    for (std::uint32_t lane_index = 0u; lane_index < lanes.size();
         ++lane_index) {
      const std::uint32_t outgoing = lanes[lane_index];
      const Int3 outgoing_ray =
          direction_offset(static_cast<Direction>(outgoing));
      const Int3 center{
          threshold.x - static_cast<std::int32_t>(
                            kProcessivePredictionBentQueryOutgoingLeg) *
                            outgoing_ray.x,
          threshold.y - static_cast<std::int32_t>(
                            kProcessivePredictionBentQueryOutgoingLeg) *
                            outgoing_ray.y,
          threshold.z - static_cast<std::int32_t>(
                            kProcessivePredictionBentQueryOutgoingLeg) *
                            outgoing_ray.z};
      const std::uint32_t index =
          kProcessivePredictionBentQueryLockSiteCount +
          pulse * lanes.size() + lane_index;
      result[index] = {
          static_cast<std::int8_t>(
              center.x - incoming_distance * incoming.x),
          static_cast<std::int8_t>(
              center.y - incoming_distance * incoming.y),
          static_cast<std::int8_t>(
              center.z - incoming_distance * incoming.z),
          static_cast<SiteWord>(
              kQ ^ carrier_bit(kProcessivePredictionBentQueryIncoming))};
    }
  }
  return result;
}

inline constexpr ProcessivePredictionQueryLauncherSeedHash
    kProcessivePredictionQueryLauncherSeedHash =
        make_processive_prediction_query_launcher_seed_hash(
            10u, 3u, kProcessivePredictionQueryPulseCount,
            kProcessivePredictionQueryGene);

static_assert(valid_processive_prediction_query_launcher_hash(
    kProcessivePredictionQueryLauncherSeedHash));
static_assert(processive_prediction_query_first_delay(
                  kProcessivePredictionQueryLauncherSeedHash) == 10u);
static_assert(processive_prediction_query_spacing(
                  kProcessivePredictionQueryLauncherSeedHash) == 3u);
static_assert(kProcessivePredictionQueryLauncherSeedHash == 0x0006b86au);
static_assert(processive_prediction_query_first_delay(
                  make_processive_prediction_query_launcher_seed_hash(
                      48u, 3u, kProcessivePredictionQueryPulseCount,
                      kProcessivePredictionQueryGene)) == 48u);
static_assert(kProcessivePredictionBentQueryOutgoingLeg < 10u);

}  // namespace substrate::bcc32
