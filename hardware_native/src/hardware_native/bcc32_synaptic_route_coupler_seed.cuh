#pragma once

// Compact, value-neutral germ for coupling a carrier route to the existing
// delayed-credit synapse. The parent five-site germ supplies the synapse and
// its local consequence receptor. Two ordinary reactive cells provide native
// P- -> E transduction for route activity and returning consequence.

#include <array>
#include <cstddef>
#include <cstdint>

#include "bcc32_credit_backcarry_seed.cuh"
#include "bcc32_dual_negative_prediction_adapter_seed.cuh"

namespace substrate::bcc32 {

using SynapticRouteCouplerSeedHash = std::uint64_t;

inline constexpr std::uint32_t kSynapticRouteCouplerParentBits = 32u;
inline constexpr SynapticRouteCouplerSeedHash kSynapticRouteCouplerParentMask =
    (SynapticRouteCouplerSeedHash{1u} << kSynapticRouteCouplerParentBits) - 1u;
inline constexpr std::uint32_t kSynapticRouteCouplerGeneShift = 32u;
inline constexpr std::uint8_t kSynapticRouteCouplerGene = 0x6du;
inline constexpr std::size_t kSynapticRouteCouplerSeedSiteCount =
    kCreditBackcarrySeedSiteCount + 2u;
inline constexpr std::uint32_t kSynapticRoutePredictionActivityFactorCount = 2u;
inline constexpr std::uint32_t kSynapticRoutePredictionSourceCount = 2u;
inline constexpr std::uint32_t kSynapticRoutePredictionReceptorDistance = 2u;
inline constexpr std::size_t kSynapticRoutePredictionReceptorSeedSiteCount =
    kSynapticRoutePredictionActivityFactorCount *
    kSynapticRoutePredictionSourceCount;

constexpr SynapticRouteCouplerSeedHash make_synaptic_route_coupler_seed_hash(
    CreditBackcarrySeedHash parent, std::uint8_t gene) {
  return (static_cast<SynapticRouteCouplerSeedHash>(parent) &
          kSynapticRouteCouplerParentMask) |
         (static_cast<SynapticRouteCouplerSeedHash>(gene)
          << kSynapticRouteCouplerGeneShift);
}

constexpr CreditBackcarrySeedHash synaptic_route_coupler_parent_hash(
    SynapticRouteCouplerSeedHash hash) {
  return static_cast<CreditBackcarrySeedHash>(
      hash & kSynapticRouteCouplerParentMask);
}

constexpr std::uint8_t synaptic_route_coupler_gene(
    SynapticRouteCouplerSeedHash hash) {
  return static_cast<std::uint8_t>(
      hash >> kSynapticRouteCouplerGeneShift);
}

constexpr std::array<DevelopmentalSeedSite,
                     kSynapticRouteCouplerSeedSiteCount>
synaptic_route_coupler_seed(SynapticRouteCouplerSeedHash hash) {
  std::array<DevelopmentalSeedSite,
             kSynapticRouteCouplerSeedSiteCount>
      result{};
  if (synaptic_route_coupler_gene(hash) != kSynapticRouteCouplerGene)
    return result;

  const auto parent =
      credit_backcarry_seed(synaptic_route_coupler_parent_hash(hash));
  for (std::size_t index = 0u; index < parent.size(); ++index)
    result[index] = parent[index];

  // The parent is later basis-permuted 0,1,2,3 -> 1,3,2,0. R0/R1 therefore
  // transduce the resident query's P-1/P-3 holes into E1/E3 pre/post activity.
  result[parent.size()] = {
      0, 0, 0,
      static_cast<SiteWord>(kQ | channel_bit(kReactiveShift, 0u) |
                            channel_bit(kReactiveShift, 1u))};

  // A returning P-2 hole becomes E2 one source step away. Basis 2 is fixed by
  // the permutation, so this is the consequence consumed by the parent.
  result[parent.size() + 1u] = {
      4, 4, 4,
      static_cast<SiteWord>(kQ | channel_bit(kReactiveShift, 2u))};
  return result;
}

constexpr std::uint32_t synaptic_route_prediction_activity_basis(
    std::uint32_t factor) {
  return factor == 0u ? 1u : 3u;
}

constexpr Int3 synaptic_route_prediction_receptor(bool one_source,
                                                  std::uint32_t factor) {
  const std::uint32_t basis =
      synaptic_route_prediction_activity_basis(factor);
  const Int3 source = dual_negative_prediction_body(
      one_source, kProcessivePredictionProjectionThresholdIndex);
  const Int3 ray =
      direction_offset(static_cast<Direction>(basis + 4u));
  return {
      source.x +
          static_cast<std::int32_t>(
              kSynapticRoutePredictionReceptorDistance) *
              ray.x,
      source.y +
          static_cast<std::int32_t>(
              kSynapticRoutePredictionReceptorDistance) *
              ray.y,
      source.z +
          static_cast<std::int32_t>(
              kSynapticRoutePredictionReceptorDistance) *
              ray.z,
  };
}

constexpr Int3 synaptic_route_prediction_energy_source(
    bool one_source, std::uint32_t factor) {
  const std::uint32_t basis =
      synaptic_route_prediction_activity_basis(factor);
  const Int3 receptor =
      synaptic_route_prediction_receptor(one_source, factor);
  const Int3 ray =
      direction_offset(static_cast<Direction>(basis + 4u));
  return {
      receptor.x + ray.x,
      receptor.y + ray.y,
      receptor.z + ray.z,
  };
}

constexpr std::array<DevelopmentalSeedSite,
                     kSynapticRoutePredictionReceptorSeedSiteCount>
synaptic_route_prediction_receptor_seed() {
  std::array<DevelopmentalSeedSite,
             kSynapticRoutePredictionReceptorSeedSiteCount>
      result{};
  for (std::uint32_t copy = 0u;
       copy < kSynapticRoutePredictionSourceCount; ++copy) {
    for (std::uint32_t factor = 0u;
         factor < kSynapticRoutePredictionActivityFactorCount; ++factor) {
      const std::uint32_t basis =
          synaptic_route_prediction_activity_basis(factor);
      const Int3 receptor =
          synaptic_route_prediction_receptor(copy != 0u, factor);
      result[copy * kSynapticRoutePredictionActivityFactorCount + factor] = {
          static_cast<std::int8_t>(receptor.x),
          static_cast<std::int8_t>(receptor.y),
          static_cast<std::int8_t>(receptor.z),
          static_cast<SiteWord>(
              kQ | channel_bit(kReactiveShift, basis)),
      };
    }
  }
  return result;
}

inline constexpr SynapticRouteCouplerSeedHash
    kSynapticRouteCouplerSeedHash =
        make_synaptic_route_coupler_seed_hash(
            kCreditBackcarrySeedHash, kSynapticRouteCouplerGene);

static_assert(synaptic_route_coupler_parent_hash(
                  kSynapticRouteCouplerSeedHash) ==
              kCreditBackcarrySeedHash);
static_assert(synaptic_route_coupler_seed(
                  kSynapticRouteCouplerSeedHash)
                  .size() == kSynapticRouteCouplerSeedSiteCount);
static_assert(synaptic_route_prediction_activity_basis(0u) == 1u);
static_assert(synaptic_route_prediction_activity_basis(1u) == 3u);
static_assert(synaptic_route_prediction_receptor(false, 0u) ==
              dual_negative_prediction_receptor(false));
static_assert(synaptic_route_prediction_receptor(true, 0u) ==
              dual_negative_prediction_receptor(true));

}  // namespace substrate::bcc32
