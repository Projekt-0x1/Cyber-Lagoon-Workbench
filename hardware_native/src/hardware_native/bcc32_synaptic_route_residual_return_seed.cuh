#pragma once

// Minimal residual-return extension of the existing four-founder synaptic
// route. A reactive basis-2 cell sits one edge beyond the canonical C2
// receptor, so a returning P-2 vacancy can produce consequence energy at the
// receptor without moving or duplicating the weight anatomy.

#include <array>
#include <cstddef>
#include <cstdint>

#include "bcc32_synaptic_route_coupler_seed.cuh"

namespace substrate::bcc32 {

using SynapticRouteResidualReturnSeedHash = std::uint64_t;

inline constexpr std::uint8_t kSynapticRouteResidualReturnGene = 0x72u;
inline constexpr std::uint32_t kSynapticRouteResidualReturnGeneShift = 40u;
inline constexpr std::size_t kSynapticRouteResidualReturnSeedSiteCount =
    kSynapticRouteCouplerSeedSiteCount + 1u;
inline constexpr Int3 kSynapticRouteResidualReturnLocal{0, 0, 2};

constexpr SynapticRouteResidualReturnSeedHash
make_synaptic_route_residual_return_seed_hash(
    SynapticRouteCouplerSeedHash parent, std::uint8_t gene) {
  return static_cast<SynapticRouteResidualReturnSeedHash>(parent) |
         (static_cast<SynapticRouteResidualReturnSeedHash>(gene)
          << kSynapticRouteResidualReturnGeneShift);
}

constexpr SynapticRouteCouplerSeedHash
synaptic_route_residual_return_parent_hash(
    SynapticRouteResidualReturnSeedHash hash) {
  return static_cast<SynapticRouteCouplerSeedHash>(
      hash & ((SynapticRouteResidualReturnSeedHash{1u}
               << kSynapticRouteResidualReturnGeneShift) -
              1u));
}

constexpr std::uint8_t synaptic_route_residual_return_gene(
    SynapticRouteResidualReturnSeedHash hash) {
  return static_cast<std::uint8_t>(
      hash >> kSynapticRouteResidualReturnGeneShift);
}

constexpr std::array<DevelopmentalSeedSite,
                     kSynapticRouteResidualReturnSeedSiteCount>
synaptic_route_residual_return_seed(
    SynapticRouteResidualReturnSeedHash hash) {
  std::array<DevelopmentalSeedSite,
             kSynapticRouteResidualReturnSeedSiteCount>
      result{};
  if (synaptic_route_residual_return_gene(hash) !=
      kSynapticRouteResidualReturnGene)
    return result;

  const auto parent = synaptic_route_coupler_seed(
      synaptic_route_residual_return_parent_hash(hash));
  for (std::size_t index = 0u; index < parent.size(); ++index)
    result[index] = parent[index];
  result[parent.size()] = {
      static_cast<std::int8_t>(kSynapticRouteResidualReturnLocal.x),
      static_cast<std::int8_t>(kSynapticRouteResidualReturnLocal.y),
      static_cast<std::int8_t>(kSynapticRouteResidualReturnLocal.z),
      static_cast<SiteWord>(
          kQ | channel_bit(kReactiveShift, 2u))};
  return result;
}

inline constexpr SynapticRouteResidualReturnSeedHash
    kSynapticRouteResidualReturnSeedHash =
        make_synaptic_route_residual_return_seed_hash(
            kSynapticRouteCouplerSeedHash,
            kSynapticRouteResidualReturnGene);

static_assert(synaptic_route_residual_return_parent_hash(
                  kSynapticRouteResidualReturnSeedHash) ==
              kSynapticRouteCouplerSeedHash);
static_assert(synaptic_route_residual_return_seed(
                  kSynapticRouteResidualReturnSeedHash)
                  .back()
                  .z == 2);
static_assert(synaptic_route_residual_return_seed(
                  kSynapticRouteResidualReturnSeedHash)
                  .back()
                  .word ==
              (kQ | channel_bit(kReactiveShift, 2u)));

}  // namespace substrate::bcc32
