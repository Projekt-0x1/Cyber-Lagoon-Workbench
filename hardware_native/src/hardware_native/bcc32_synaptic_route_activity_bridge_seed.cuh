#pragma once

// A compact resident bridge from one M1 prediction source to a distinct
// four-founder delayed-credit synapse. Two signed prediction holes retain
// their lane identity while ordinary carrier corners bring them to one
// non-overlapping R1/R3 activity transducer.

#include <array>
#include <cstddef>
#include <cstdint>

#include "bcc32_synaptic_route_coupler_seed.cuh"

namespace substrate::bcc32 {

using SynapticRouteActivityBridgeSeedHash = std::uint64_t;

inline constexpr std::uint32_t kSynapticRouteActivityBridgeGeneShift = 40u;
inline constexpr std::uint8_t kSynapticRouteActivityBridgeGene = 0xa7u;
inline constexpr std::size_t kSynapticRouteActivityBridgeCornerCount = 6u;
inline constexpr std::size_t kSynapticRouteActivityBridgeSeedSiteCount =
    kSynapticRouteCouplerSeedSiteCount +
    kSynapticRouteActivityBridgeCornerCount *
        (kCarrierCornerSiteCount - 1u);
inline constexpr std::array<std::uint8_t, 4u>
    kSynapticRouteActivityBridgePermutation{{1u, 3u, 2u, 0u}};

constexpr SynapticRouteActivityBridgeSeedHash
make_synaptic_route_activity_bridge_seed_hash(
    SynapticRouteCouplerSeedHash parent, std::uint8_t gene) {
  return (parent & ((SynapticRouteActivityBridgeSeedHash{1u} << 40u) - 1u)) |
         (static_cast<SynapticRouteActivityBridgeSeedHash>(gene)
          << kSynapticRouteActivityBridgeGeneShift);
}

constexpr SynapticRouteCouplerSeedHash
synaptic_route_activity_bridge_parent_hash(
    SynapticRouteActivityBridgeSeedHash hash) {
  return hash & ((SynapticRouteActivityBridgeSeedHash{1u} << 40u) - 1u);
}

constexpr std::uint8_t synaptic_route_activity_bridge_gene(
    SynapticRouteActivityBridgeSeedHash hash) {
  return static_cast<std::uint8_t>(
      hash >> kSynapticRouteActivityBridgeGeneShift);
}

constexpr Int3 synaptic_route_activity_bridge_transform(Int3 coordinate) {
  return {-coordinate.y, coordinate.x - coordinate.y,
          coordinate.z - coordinate.y};
}

constexpr SiteWord synaptic_route_activity_bridge_transform_word(
    SiteWord word) {
  SiteWord result = 0u;
  for (std::uint32_t basis = 0u; basis < 4u; ++basis) {
    const std::uint32_t image =
        kSynapticRouteActivityBridgePermutation[basis];
    if ((word & carrier_bit(basis)) != 0u)
      result |= carrier_bit(image);
    if ((word & carrier_bit(basis + 4u)) != 0u)
      result |= carrier_bit(image + 4u);
    if ((word & face_bit(basis)) != 0u)
      result |= face_bit(image);
    if ((word & face_bit(basis + 4u)) != 0u)
      result |= face_bit(image + 4u);
    if ((word & owned_bond_bit(basis)) != 0u)
      result |= owned_bond_bit(image);
    if ((word & channel_bit(kConformationShift, basis)) != 0u)
      result |= channel_bit(kConformationShift, image);
    if ((word & channel_bit(kReactiveShift, basis)) != 0u)
      result |= channel_bit(kReactiveShift, image);
    if ((word & energy_bit(basis)) != 0u)
      result |= energy_bit(image);
  }
  return result;
}

constexpr Int3 synaptic_route_activity_bridge_translate(Int3 coordinate,
                                                        bool one_source) {
  return dual_negative_prediction_translate(coordinate, one_source);
}

constexpr Int3 synaptic_route_activity_bridge_hub(bool one_source) {
  return synaptic_route_activity_bridge_translate({3, 1, 35}, one_source);
}

constexpr std::array<Int3, kSynapticRouteActivityBridgeCornerCount>
synaptic_route_activity_bridge_corner_centers(bool one_source) {
  return {{
      synaptic_route_activity_bridge_translate({0, -1, 32}, one_source),
      synaptic_route_activity_bridge_translate({3, 2, 35}, one_source),
      synaptic_route_activity_bridge_translate({1, 1, 33}, one_source),
      synaptic_route_activity_bridge_translate({2, 1, 33}, one_source),
      synaptic_route_activity_bridge_translate({2, 0, 33}, one_source),
      synaptic_route_activity_bridge_translate({2, 0, 34}, one_source),
  }};
}

constexpr std::array<std::array<std::uint32_t, 2u>,
                     kSynapticRouteActivityBridgeCornerCount>
    kSynapticRouteActivityBridgeCornerLanes{{
        {{5u, 7u}},
        {{7u, 5u}},
        {{7u, 0u}},
        {{0u, 5u}},
        {{5u, 2u}},
        {{2u, 7u}},
    }};

constexpr DevelopmentalSeedSite synaptic_route_activity_bridge_site(
    Int3 coordinate, SiteWord word) {
  return {static_cast<std::int8_t>(coordinate.x),
          static_cast<std::int8_t>(coordinate.y),
          static_cast<std::int8_t>(coordinate.z), word};
}

constexpr std::array<DevelopmentalSeedSite,
                     kSynapticRouteActivityBridgeSeedSiteCount>
synaptic_route_activity_bridge_seed(
    SynapticRouteActivityBridgeSeedHash hash, bool one_source) {
  std::array<DevelopmentalSeedSite,
             kSynapticRouteActivityBridgeSeedSiteCount>
      result{};
  if (synaptic_route_activity_bridge_gene(hash) !=
      kSynapticRouteActivityBridgeGene)
    return result;

  std::size_t next = 0u;
  const Int3 hub = synaptic_route_activity_bridge_hub(one_source);
  const auto coupler = synaptic_route_coupler_seed(
      synaptic_route_activity_bridge_parent_hash(hash));
  for (const DevelopmentalSeedSite& site : coupler) {
    const Int3 transformed =
        synaptic_route_activity_bridge_transform(
            {site.x, site.y, site.z});
    result[next++] = synaptic_route_activity_bridge_site(
        hub + transformed,
        synaptic_route_activity_bridge_transform_word(site.word));
  }

  const auto centers =
      synaptic_route_activity_bridge_corner_centers(one_source);
  for (std::size_t corner = 0u;
       corner < centers.size(); ++corner) {
    const std::uint32_t incoming =
        kSynapticRouteActivityBridgeCornerLanes[corner][0u];
    const std::uint32_t outgoing =
        kSynapticRouteActivityBridgeCornerLanes[corner][1u];
    for (std::uint32_t index = 1u;
         index < kCarrierCornerSiteCount; ++index) {
      const CarrierCornerOffset offset =
          carrier_corner_offset(incoming, outgoing, index);
      result[next++] = synaptic_route_activity_bridge_site(
          centers[corner] +
              Int3{offset.x, offset.y, offset.z},
          carrier_corner_word(incoming, outgoing, index, false));
    }
  }
  return result;
}

constexpr bool synaptic_route_activity_bridge_seed_is_unique(
    SynapticRouteActivityBridgeSeedHash hash, bool one_source) {
  const auto seed =
      synaptic_route_activity_bridge_seed(hash, one_source);
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

inline constexpr SynapticRouteActivityBridgeSeedHash
    kSynapticRouteActivityBridgeSeedHash =
        make_synaptic_route_activity_bridge_seed_hash(
            kSynapticRouteCouplerSeedHash,
            kSynapticRouteActivityBridgeGene);

static_assert(
    synaptic_route_activity_bridge_parent_hash(
        kSynapticRouteActivityBridgeSeedHash) ==
    kSynapticRouteCouplerSeedHash);
static_assert(
    synaptic_route_activity_bridge_hub(false) == Int3{3, 1, 35});
static_assert(
    synaptic_route_activity_bridge_hub(true) == Int3{35, 1, 35});
static_assert(synaptic_route_activity_bridge_seed_is_unique(
    kSynapticRouteActivityBridgeSeedHash, false));
static_assert(synaptic_route_activity_bridge_seed_is_unique(
    kSynapticRouteActivityBridgeSeedHash, true));

}  // namespace substrate::bcc32
