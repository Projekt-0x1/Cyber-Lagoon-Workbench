#pragma once

// Minimal four-founder synaptic route for testing direct carrier contact at
// the canonical C2 receptor. The parent return R2 is removed so any consequence
// must arise from P-2 reaching C2 itself under ordinary F.

#include <array>
#include <cstddef>
#include <cstdint>

#include "bcc32_synaptic_route_coupler_seed.cuh"

namespace substrate::bcc32 {

using SynapticRouteDirectC2IntakeSeedHash = std::uint64_t;

inline constexpr std::uint8_t kSynapticRouteDirectC2IntakeGene = 0x73u;
inline constexpr std::uint32_t kSynapticRouteDirectC2IntakeGeneShift = 40u;
inline constexpr std::size_t kSynapticRouteDirectC2IntakeSeedSiteCount =
    kSynapticRouteCouplerSeedSiteCount;

constexpr SynapticRouteDirectC2IntakeSeedHash
make_synaptic_route_direct_c2_intake_seed_hash(
    SynapticRouteCouplerSeedHash parent, std::uint8_t gene) {
  return static_cast<SynapticRouteDirectC2IntakeSeedHash>(parent) |
         (static_cast<SynapticRouteDirectC2IntakeSeedHash>(gene)
          << kSynapticRouteDirectC2IntakeGeneShift);
}

constexpr SynapticRouteCouplerSeedHash
synaptic_route_direct_c2_intake_parent_hash(
    SynapticRouteDirectC2IntakeSeedHash hash) {
  return static_cast<SynapticRouteCouplerSeedHash>(
      hash & ((SynapticRouteDirectC2IntakeSeedHash{1u}
               << kSynapticRouteDirectC2IntakeGeneShift) -
              1u));
}

constexpr std::uint8_t synaptic_route_direct_c2_intake_gene(
    SynapticRouteDirectC2IntakeSeedHash hash) {
  return static_cast<std::uint8_t>(
      hash >> kSynapticRouteDirectC2IntakeGeneShift);
}

constexpr std::array<DevelopmentalSeedSite,
                     kSynapticRouteDirectC2IntakeSeedSiteCount>
synaptic_route_direct_c2_intake_seed(
    SynapticRouteDirectC2IntakeSeedHash hash) {
  std::array<DevelopmentalSeedSite,
             kSynapticRouteDirectC2IntakeSeedSiteCount>
      result{};
  if (synaptic_route_direct_c2_intake_gene(hash) !=
      kSynapticRouteDirectC2IntakeGene)
    return result;

  result = synaptic_route_coupler_seed(
      synaptic_route_direct_c2_intake_parent_hash(hash));
  result.back().word = kQ;
  return result;
}

inline constexpr SynapticRouteDirectC2IntakeSeedHash
    kSynapticRouteDirectC2IntakeSeedHash =
        make_synaptic_route_direct_c2_intake_seed_hash(
            kSynapticRouteCouplerSeedHash,
            kSynapticRouteDirectC2IntakeGene);

static_assert(synaptic_route_direct_c2_intake_parent_hash(
                  kSynapticRouteDirectC2IntakeSeedHash) ==
              kSynapticRouteCouplerSeedHash);
static_assert(synaptic_route_direct_c2_intake_seed(
                  kSynapticRouteDirectC2IntakeSeedHash)
                  .back()
                  .word == kQ);
static_assert(synaptic_route_direct_c2_intake_seed(
                  kSynapticRouteDirectC2IntakeSeedHash)
                  [kCreditBackcarrySeedSiteCount - 1u]
                  .word ==
              (kQ | channel_bit(kConformationShift, 2u)));

}  // namespace substrate::bcc32
