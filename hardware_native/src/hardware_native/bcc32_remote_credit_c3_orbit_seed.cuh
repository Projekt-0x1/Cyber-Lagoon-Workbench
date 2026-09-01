#pragma once

// A compact developmental composition: preserve the remote route's R3 hub,
// remove C3 from the known two-state-orbit source, and place its companion on
// one of the six S4-equivalent physical ports that map the orbit's old input
// lane 0 to the remote lane 3.  Thus the seed supplies recurrent scaffolding,
// not the claimed C3 write; ordinary F must obtain that C3 from remote B3/E3.

#include <array>
#include <cstdint>

#include "bcc32_remote_credit_hole_transducer_seed.cuh"

namespace substrate::bcc32 {

using RemoteCreditC3OrbitSeedHash = std::uint8_t;
constexpr std::uint32_t kRemoteCreditC3OrbitRouteMask = 0x07u;
constexpr std::uint32_t kRemoteCreditC3OrbitVariantShift = 3u;
constexpr std::array<BasisPermutation, 6u> kRemoteCreditC3OrbitPermutations{{
    {3u, 0u, 1u, 2u}, {3u, 0u, 2u, 1u}, {3u, 1u, 0u, 2u},
    {3u, 1u, 2u, 0u}, {3u, 2u, 0u, 1u}, {3u, 2u, 1u, 0u},
}};
constexpr std::size_t kRemoteCreditC3OrbitSeedSiteCount =
    kRemoteCreditHoleTransducerSeedSiteCount + 1u;

constexpr RemoteCreditC3OrbitSeedHash make_remote_credit_c3_orbit_seed_hash(
    std::uint32_t route_length, std::uint32_t variant) {
  return static_cast<RemoteCreditC3OrbitSeedHash>(
      (route_length & kRemoteCreditC3OrbitRouteMask) |
      (variant << kRemoteCreditC3OrbitVariantShift));
}

constexpr RemoteCreditHoleTransducerSeedHash remote_credit_c3_orbit_route_hash(
    RemoteCreditC3OrbitSeedHash hash) {
  return static_cast<RemoteCreditHoleTransducerSeedHash>(hash & kRemoteCreditC3OrbitRouteMask);
}

constexpr std::uint32_t remote_credit_c3_orbit_variant(RemoteCreditC3OrbitSeedHash hash) {
  return static_cast<std::uint32_t>(hash >> kRemoteCreditC3OrbitVariantShift);
}

inline BasisPermutation remote_credit_c3_orbit_permutation(RemoteCreditC3OrbitSeedHash hash) {
  return kRemoteCreditC3OrbitPermutations[remote_credit_c3_orbit_variant(hash)];
}

inline Z3Coordinate remote_credit_c3_orbit_destination(RemoteCreditC3OrbitSeedHash hash) {
  const Int3 old_lane_zero = direction_offset(static_cast<Direction>(0u));
  const Z3Coordinate relative = transformed_coordinate(
      Z3Coordinate{old_lane_zero.x, old_lane_zero.y, old_lane_zero.z},
      remote_credit_c3_orbit_permutation(hash));
  return {relative.x, relative.y, relative.z};
}

inline std::array<DevelopmentalSeedSite, kRemoteCreditC3OrbitSeedSiteCount>
remote_credit_c3_orbit_seed(CreditOrbitSeedHash parent, RemoteCreditC3OrbitSeedHash hash) {
  std::array<DevelopmentalSeedSite, kRemoteCreditC3OrbitSeedSiteCount> result{};
  const auto base = remote_credit_hole_transducer_seed(parent, remote_credit_c3_orbit_route_hash(hash));
  for (std::size_t index = 0u; index < base.size(); ++index) result[index] = base[index];
  const BasisPermutation permutation = remote_credit_c3_orbit_permutation(hash);
  const SiteWord c3 = channel_bit(kConformationShift, 3u);
  result[base.size() - 1u].word =
      static_cast<SiteWord>((transformed_word(0x101001feu, permutation) & ~c3) |
                            channel_bit(kReactiveShift, 3u));
  const Z3Coordinate destination = remote_credit_c3_orbit_destination(hash);
  result.back() = {static_cast<std::int8_t>(destination.x),
                   static_cast<std::int8_t>(destination.y),
                   static_cast<std::int8_t>(destination.z),
                   transformed_word(0x000000efu, permutation)};
  return result;
}

}  // namespace substrate::bcc32
