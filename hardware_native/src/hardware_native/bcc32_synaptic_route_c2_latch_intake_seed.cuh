#pragma once

// Compact no-R2 M2 composition. The canonical four-founder synaptic weight is
// translated by -2u2 so its basis-2 founder is the B2 source immediately
// upstream of a C2 latch at the synaptic hub. A comparator P-2 vacancy can
// therefore contact the latch while the companion contract tests whether
// ordinary gate 2 exposes selective E2 at the weight's own consequence
// founder.

#include <array>
#include <cstddef>
#include <cstdint>

#include "bcc32_synaptic_weight_seed.cuh"

namespace substrate::bcc32 {

using SynapticRouteC2LatchIntakeSeedHash = std::uint32_t;

inline constexpr std::uint32_t kSynapticRouteC2LatchWeightShift = 0u;
inline constexpr std::uint32_t kSynapticRouteC2LatchGeneShift = 8u;
inline constexpr std::uint8_t kSynapticRouteC2LatchGene = 0x5du;
inline constexpr std::size_t kSynapticRouteC2LatchIntakeSeedSiteCount =
    kLocalFounderBasisCount + 1u;

constexpr SynapticRouteC2LatchIntakeSeedHash
make_synaptic_route_c2_latch_intake_seed_hash(
    SynapticWeightSeedHash weight, std::uint8_t gene) {
  return static_cast<SynapticRouteC2LatchIntakeSeedHash>(weight) |
         (static_cast<SynapticRouteC2LatchIntakeSeedHash>(gene)
          << kSynapticRouteC2LatchGeneShift);
}

constexpr SynapticWeightSeedHash
synaptic_route_c2_latch_intake_weight_hash(
    SynapticRouteC2LatchIntakeSeedHash hash) {
  return static_cast<SynapticWeightSeedHash>(
      hash & 0xffu);
}

constexpr std::uint8_t synaptic_route_c2_latch_intake_gene(
    SynapticRouteC2LatchIntakeSeedHash hash) {
  return static_cast<std::uint8_t>(
      hash >> kSynapticRouteC2LatchGeneShift);
}

constexpr std::array<DevelopmentalSeedSite,
                     kSynapticRouteC2LatchIntakeSeedSiteCount>
synaptic_route_c2_latch_intake_seed(
    SynapticRouteC2LatchIntakeSeedHash hash) {
  std::array<DevelopmentalSeedSite,
             kSynapticRouteC2LatchIntakeSeedSiteCount>
      result{};
  if (synaptic_route_c2_latch_intake_gene(hash) !=
      kSynapticRouteC2LatchGene)
    return result;

  const auto weight = synaptic_weight_seed(
      synaptic_route_c2_latch_intake_weight_hash(hash));
  for (std::size_t index = 0u; index < weight.size(); ++index) {
    result[index] = {
        weight[index].x,
        weight[index].y,
        static_cast<std::int8_t>(weight[index].z - 2),
        weight[index].word,
    };
  }
  result.back() = {
      0, 0, -2,
      static_cast<SiteWord>(
          kQ | channel_bit(kConformationShift, 2u)),
  };
  return result;
}

inline constexpr SynapticRouteC2LatchIntakeSeedHash
    kSynapticRouteC2LatchIntakeSeedHash =
        make_synaptic_route_c2_latch_intake_seed_hash(
            kSynapticWeightSeedHash,
            kSynapticRouteC2LatchGene);

static_assert(
    synaptic_route_c2_latch_intake_weight_hash(
        kSynapticRouteC2LatchIntakeSeedHash) ==
    kSynapticWeightSeedHash);
static_assert(
    synaptic_route_c2_latch_intake_seed(
        kSynapticRouteC2LatchIntakeSeedHash)[2u].z == -3);
static_assert(
    synaptic_route_c2_latch_intake_seed(
        kSynapticRouteC2LatchIntakeSeedHash)[2u].word ==
    (kQ | owned_bond_bit(2u)));
static_assert(
    synaptic_route_c2_latch_intake_seed(
        kSynapticRouteC2LatchIntakeSeedHash).back().word ==
    (kQ | channel_bit(kConformationShift, 2u)));

}  // namespace substrate::bcc32
