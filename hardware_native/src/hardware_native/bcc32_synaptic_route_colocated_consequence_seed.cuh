#pragma once

// M2 consequence-chemistry candidate. The canonical parent C2 is founded at
// the post-R2 E2 source instead of at the legacy remote receptor. No second C2
// is added, and the resident R2 transducer remains spatially distinct.

#include <array>
#include <cstdint>

#include "bcc32_synaptic_route_coupler_seed.cuh"

namespace substrate::bcc32 {

inline constexpr CreditBackcarrySeedHash
    kSynapticRouteColocatedBackcarrySeedHash =
        make_credit_backcarry_seed_hash(
            kSynapticWeightSeedHash, 0x2u, 4, 4, 3);

inline constexpr SynapticRouteCouplerSeedHash
    kSynapticRouteColocatedConsequenceSeedHash =
        make_synaptic_route_coupler_seed_hash(
            kSynapticRouteColocatedBackcarrySeedHash,
            kSynapticRouteCouplerGene);

constexpr std::array<DevelopmentalSeedSite,
                     kSynapticRouteCouplerSeedSiteCount>
synaptic_route_colocated_consequence_seed() {
  return synaptic_route_coupler_seed(
      kSynapticRouteColocatedConsequenceSeedHash);
}

inline constexpr auto kSynapticRouteColocatedConsequenceSeed =
    synaptic_route_colocated_consequence_seed();

static_assert(credit_backcarry_receptor_coordinate(
                  kSynapticRouteColocatedBackcarrySeedHash) ==
              std::array<std::int32_t, 3>{{4, 4, 3}});
static_assert(
    kSynapticRouteColocatedConsequenceSeed[
        kCreditBackcarrySeedSiteCount - 1u].x == 4 &&
    kSynapticRouteColocatedConsequenceSeed[
        kCreditBackcarrySeedSiteCount - 1u].y == 4 &&
    kSynapticRouteColocatedConsequenceSeed[
        kCreditBackcarrySeedSiteCount - 1u].z == 3);
static_assert(
    (kSynapticRouteColocatedConsequenceSeed[
         kCreditBackcarrySeedSiteCount - 1u].word &
     channel_bit(kConformationShift, 2u)) != 0u);
static_assert(
    kSynapticRouteColocatedConsequenceSeed[
        kSynapticRouteCouplerSeedSiteCount - 1u].x == 4 &&
    kSynapticRouteColocatedConsequenceSeed[
        kSynapticRouteCouplerSeedSiteCount - 1u].y == 4 &&
    kSynapticRouteColocatedConsequenceSeed[
        kSynapticRouteCouplerSeedSiteCount - 1u].z == 4);
static_assert(
    (kSynapticRouteColocatedConsequenceSeed[
         kSynapticRouteCouplerSeedSiteCount - 1u].word &
     channel_bit(kReactiveShift, 2u)) != 0u);

}  // namespace substrate::bcc32
