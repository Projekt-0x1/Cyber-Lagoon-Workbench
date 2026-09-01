#pragma once

// Compose two compact, already interpreted local grammars: an origin weight germ and
// a translated remote credit emitter.  The hash selects the emitter's diagonal
// separation plus one of the 24 exact S4 orientations of the origin germ.  It contains
// neither a learned value nor a route schedule; F must make the arriving carrier useful
// to the oriented origin germ.

#include <array>
#include <cstdint>

#include "bcc32_remote_credit_route_seed.cuh"
#include "bcc32_synaptic_weight_seed.cuh"

namespace substrate::bcc32 {

using RemoteCreditWeightSeedHash = std::uint8_t;
constexpr std::uint32_t kRemoteCreditWeightRouteMask = 0x07u;
constexpr std::uint32_t kRemoteCreditWeightOrientationShift = 3u;
constexpr std::uint32_t kRemoteCreditWeightOrientationCount = 24u;
constexpr std::size_t kRemoteCreditWeightSeedSiteCount =
    kLocalFounderBasisCount + kRemoteCreditRouteSeedSiteCount;

constexpr RemoteCreditWeightSeedHash make_remote_credit_weight_seed_hash(std::uint32_t route_length,
                                                                           std::uint32_t orientation) {
  return static_cast<RemoteCreditWeightSeedHash>(
      (route_length & kRemoteCreditWeightRouteMask) |
      ((orientation & 0x1fu) << kRemoteCreditWeightOrientationShift));
}

constexpr RemoteCreditRouteSeedHash remote_credit_weight_route_hash(RemoteCreditWeightSeedHash hash) {
  return static_cast<RemoteCreditRouteSeedHash>(hash & kRemoteCreditWeightRouteMask);
}

constexpr std::uint32_t remote_credit_weight_orientation(RemoteCreditWeightSeedHash hash) {
  return static_cast<std::uint32_t>(hash >> kRemoteCreditWeightOrientationShift);
}

inline BasisPermutation remote_credit_weight_permutation(RemoteCreditWeightSeedHash hash) {
  BasisPermutation permutation{0u, 1u, 2u, 3u};
  const std::uint32_t target = remote_credit_weight_orientation(hash);
  for (std::uint32_t index = 0u; index < target; ++index) std::next_permutation(permutation.begin(), permutation.end());
  return permutation;
}

inline std::array<DevelopmentalSeedSite, kRemoteCreditWeightSeedSiteCount>
remote_credit_weight_seed(CreditOrbitSeedHash parent, RemoteCreditWeightSeedHash hash,
                          SynapticWeightSeedHash weight_hash = kSynapticWeightSeedHash) {
  std::array<DevelopmentalSeedSite, kRemoteCreditWeightSeedSiteCount> result{};
  const auto weight = synaptic_weight_seed(weight_hash);
  const auto route = remote_credit_route_seed(parent, remote_credit_weight_route_hash(hash));
  const BasisPermutation permutation = remote_credit_weight_permutation(hash);
  for (std::size_t index = 0u; index < weight.size(); ++index) {
    const Z3Coordinate coordinate = transformed_coordinate(
        Z3Coordinate{weight[index].x, weight[index].y, weight[index].z}, permutation);
    result[index] = {static_cast<std::int8_t>(coordinate.x), static_cast<std::int8_t>(coordinate.y),
                     static_cast<std::int8_t>(coordinate.z),
                     transformed_word(weight[index].word, permutation)};
  }
  for (std::size_t index = 0u; index < route.size(); ++index) result[weight.size() + index] = route[index];
  return result;
}

}  // namespace substrate::bcc32
