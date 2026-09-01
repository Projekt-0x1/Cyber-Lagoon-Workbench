#pragma once

// Resident fan-out for the complementary dual-negative source. Two adjacent
// lane-5 prediction vacancies form one conserved packet. A pair splitter keeps
// one vacancy on lane 5 for the activity bridge and diverts the other onto a
// comparator route. Singleton activity cannot trigger the split.

#include <array>
#include <cstddef>
#include <cstdint>

#include "bcc32_synaptic_route_activity_bridge_seed.cuh"

namespace substrate::bcc32 {

inline constexpr std::size_t kDualAuthorityTransportCornerCount = 5u;
inline constexpr std::size_t kDualAuthorityTransportSplitterCount = 2u;
inline constexpr std::size_t kDualAuthorityTransportSeedSiteCount =
    kDualAuthorityTransportCornerCount * (kCarrierCornerSiteCount - 1u) +
    kDualAuthorityTransportSplitterCount *
        (kCarrierPairSplitterSiteCount - 2u);

inline constexpr ProcessivePredictionQueryLauncherSeedHash
    kDualAuthorityPairedPulseLauncherSeedHash =
        make_processive_prediction_query_launcher_seed_hash(
            10u, 1u, kProcessivePredictionQueryPulseCount,
            kProcessivePredictionQueryGene);
inline constexpr Int3 kDualAuthorityActivityBridgeOffset{0, -8, 0};

constexpr Int3 dual_authority_splitter_center(bool one_source) {
  return dual_negative_prediction_translate({0, -4, 32}, one_source);
}

constexpr Int3 dual_authority_split_activity_bridge_hub(bool one_source) {
  return synaptic_route_activity_bridge_hub(one_source) +
         kDualAuthorityActivityBridgeOffset;
}

constexpr std::array<Int3, kDualAuthorityTransportCornerCount>
dual_authority_transport_corner_centers() {
  return {{
      // Zero source, diverted prediction zero -> positive comparator lane 5.
      {0, -3, 0},
      {-16, -3, 0},
      {-16, 6, 0},
      // One source, diverted prediction one -> negative comparator lane 1.
      {32, -3, 0},
      {16, -3, 0},
  }};
}

inline constexpr std::array<std::array<std::uint32_t, 2u>,
                            kDualAuthorityTransportCornerCount>
    kDualAuthorityTransportCornerLanes{{
        {{6u, 4u}},
        {{4u, 1u}},
        {{1u, 5u}},
        {{6u, 4u}},
        {{4u, 1u}},
    }};

constexpr DevelopmentalSeedSite dual_authority_transport_site(
    Int3 coordinate, SiteWord word) {
  return {static_cast<std::int8_t>(coordinate.x),
          static_cast<std::int8_t>(coordinate.y),
          static_cast<std::int8_t>(coordinate.z), word};
}

constexpr std::array<DevelopmentalSeedSite,
                     kDualAuthorityTransportSeedSiteCount>
dual_authority_transport_seed() {
  std::array<DevelopmentalSeedSite,
             kDualAuthorityTransportSeedSiteCount>
      result{};
  std::size_t next = 0u;
  for (std::uint32_t copy = 0u;
       copy < kDualAuthorityTransportSplitterCount; ++copy) {
    const Int3 center = dual_authority_splitter_center(copy != 0u);
    for (std::uint32_t index = 2u;
         index < kCarrierPairSplitterSiteCount; ++index) {
      const CarrierPairSplitterOffset offset =
          carrier_pair_splitter_offset(5u, 6u, index);
      result[next++] = dual_authority_transport_site(
          center + Int3{offset.x, offset.y, offset.z},
          carrier_pair_splitter_word(5u, 6u, index, false));
    }
  }
  const auto centers = dual_authority_transport_corner_centers();
  for (std::size_t corner = 0u; corner < centers.size(); ++corner) {
    const std::uint32_t incoming =
        kDualAuthorityTransportCornerLanes[corner][0u];
    const std::uint32_t outgoing =
        kDualAuthorityTransportCornerLanes[corner][1u];
    for (std::uint32_t index = 1u;
         index < kCarrierCornerSiteCount; ++index) {
      const CarrierCornerOffset offset =
          carrier_corner_offset(incoming, outgoing, index);
      result[next++] = dual_authority_transport_site(
          centers[corner] + Int3{offset.x, offset.y, offset.z},
          carrier_corner_word(incoming, outgoing, index, false));
    }
  }
  return result;
}

constexpr std::array<DevelopmentalSeedSite,
                     kSynapticRouteActivityBridgeSeedSiteCount>
dual_authority_split_activity_bridge_seed(
    SynapticRouteActivityBridgeSeedHash hash, bool one_source) {
  std::array<DevelopmentalSeedSite,
             kSynapticRouteActivityBridgeSeedSiteCount>
      result{};
  const auto source =
      synaptic_route_activity_bridge_seed(hash, one_source);
  for (std::size_t index = 0u; index < source.size(); ++index) {
    result[index] = dual_authority_transport_site(
        Int3{source[index].x, source[index].y, source[index].z} +
            kDualAuthorityActivityBridgeOffset,
        source[index].word);
  }
  return result;
}

template <std::size_t Size>
constexpr bool dual_authority_seed_sites_unique(
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

static_assert(valid_processive_prediction_query_launcher_hash(
    kDualAuthorityPairedPulseLauncherSeedHash));
static_assert(processive_prediction_query_spacing(
                  kDualAuthorityPairedPulseLauncherSeedHash) == 1u);
static_assert(dual_authority_seed_sites_unique(
    dual_authority_transport_seed()));
static_assert(dual_authority_seed_sites_unique(
    dual_authority_split_activity_bridge_seed(
        kSynapticRouteActivityBridgeSeedHash, false)));
static_assert(dual_authority_seed_sites_unique(
    dual_authority_split_activity_bridge_seed(
        kSynapticRouteActivityBridgeSeedHash, true)));

}  // namespace substrate::bcc32
