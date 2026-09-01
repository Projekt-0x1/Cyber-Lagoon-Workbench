#pragma once

// M2 consequence-only wrapper.  The existing coupler germ owns the canonical
// four-founder synapse, its parent C2, and the resident R2 site.  This wrapper
// adds only a second C2 at that already-existing R2 site; it introduces no new
// route, payload, or runtime interpreter.

#include <array>
#include <cstdint>

#include "bcc32_synaptic_route_coupler_seed.cuh"

namespace substrate::bcc32 {

using SynapticRouteDirectConsequenceSeedHash = SynapticRouteCouplerSeedHash;

inline constexpr SynapticRouteDirectConsequenceSeedHash
    kSynapticRouteDirectConsequenceSeedHash = kSynapticRouteCouplerSeedHash;

constexpr SynapticRouteDirectConsequenceSeedHash
make_synaptic_route_direct_consequence_seed_hash(
    CreditBackcarrySeedHash parent, std::uint8_t gene) {
  return make_synaptic_route_coupler_seed_hash(parent, gene);
}

constexpr CreditBackcarrySeedHash
synaptic_route_direct_consequence_parent_hash(
    SynapticRouteDirectConsequenceSeedHash hash) {
  return synaptic_route_coupler_parent_hash(hash);
}

constexpr std::array<DevelopmentalSeedSite,
                     kSynapticRouteCouplerSeedSiteCount>
synaptic_route_direct_consequence_seed(
    SynapticRouteDirectConsequenceSeedHash hash) {
  auto result = synaptic_route_coupler_seed(hash);
  if (synaptic_route_coupler_gene(hash) == kSynapticRouteCouplerGene) {
    // The final coupler site is local {4,4,4}: retain R2 and add C2 in place.
    result[kSynapticRouteCouplerSeedSiteCount - 1u].word |=
        channel_bit(kConformationShift, 2u);
  }
  return result;
}

inline constexpr auto kSynapticRouteDirectConsequenceSeed =
    synaptic_route_direct_consequence_seed(
        kSynapticRouteDirectConsequenceSeedHash);

static_assert(synaptic_route_direct_consequence_parent_hash(
                  kSynapticRouteDirectConsequenceSeedHash) ==
              kCreditBackcarrySeedHash);
static_assert(kSynapticRouteDirectConsequenceSeed.size() ==
              kSynapticRouteCouplerSeedSiteCount);
static_assert(kSynapticRouteDirectConsequenceSeed[
                  kCreditBackcarrySeedSiteCount - 1u].x == 0 &&
              kSynapticRouteDirectConsequenceSeed[
                  kCreditBackcarrySeedSiteCount - 1u].y == 0 &&
              kSynapticRouteDirectConsequenceSeed[
                  kCreditBackcarrySeedSiteCount - 1u].z == 1);
static_assert((kSynapticRouteDirectConsequenceSeed[
                   kCreditBackcarrySeedSiteCount - 1u].word &
               channel_bit(kConformationShift, 2u)) != 0u);
static_assert(kSynapticRouteDirectConsequenceSeed[
                  kSynapticRouteCouplerSeedSiteCount - 1u].x == 4 &&
              kSynapticRouteDirectConsequenceSeed[
                  kSynapticRouteCouplerSeedSiteCount - 1u].y == 4 &&
              kSynapticRouteDirectConsequenceSeed[
                  kSynapticRouteCouplerSeedSiteCount - 1u].z == 4);
static_assert((kSynapticRouteDirectConsequenceSeed[
                   kSynapticRouteCouplerSeedSiteCount - 1u].word &
               channel_bit(kReactiveShift, 2u)) != 0u);
static_assert((kSynapticRouteDirectConsequenceSeed[
                   kSynapticRouteCouplerSeedSiteCount - 1u].word &
               channel_bit(kConformationShift, 2u)) != 0u);

}  // namespace substrate::bcc32
