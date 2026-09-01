#pragma once

// One compact developmental hash expands a reusable signed-synapse gene on
// three tetrahedral rays converging on one receiver. The hash contains no site
// words: it selects the gene, branch mask, and radius; the interpreter repeats
// the same two-cell local founder rule in each enabled orientation.

#include <array>
#include <cstddef>
#include <cstdint>

#include "bcc32_developmental_seed.cuh"
#include "bcc32_geometry.cuh"
#include "bcc32_law.cuh"

namespace substrate::bcc32 {

using SignedSynapseRegionSeedHash = std::uint32_t;

constexpr std::uint32_t kSignedRegionBranchCount = 3u;
constexpr std::uint32_t kSignedRegionSitesPerBranch = 2u;
constexpr std::uint32_t kSignedRegionSeedSiteCount =
    kSignedRegionBranchCount * kSignedRegionSitesPerBranch;
constexpr std::uint32_t kSignedRegionGeneId = 0x53u;
constexpr SiteWord kSignedRegionWeightZero = 0x100010eeu;
constexpr SiteWord kSignedRegionCollar = 0x000100eeu;

constexpr SignedSynapseRegionSeedHash make_signed_synapse_region_seed_hash(
    std::uint8_t branch_mask, std::uint8_t radius, std::uint8_t gene) {
  return static_cast<SignedSynapseRegionSeedHash>(branch_mask & 0x7u) |
         (static_cast<SignedSynapseRegionSeedHash>(radius & 0x3fu) << 3u) |
         (static_cast<SignedSynapseRegionSeedHash>(gene) << 9u);
}

constexpr std::uint8_t signed_region_branch_mask(SignedSynapseRegionSeedHash hash) {
  return static_cast<std::uint8_t>(hash & 0x7u);
}

constexpr std::uint8_t signed_region_radius(SignedSynapseRegionSeedHash hash) {
  return static_cast<std::uint8_t>((hash >> 3u) & 0x3fu);
}

constexpr std::uint8_t signed_region_gene(SignedSynapseRegionSeedHash hash) {
  return static_cast<std::uint8_t>(hash >> 9u);
}

constexpr std::array<std::uint32_t, kSignedRegionBranchCount>
kSignedRegionReceiverLanes{{1u, 2u, 3u}};

constexpr std::array<DevelopmentalSeedSite, kSignedRegionBranchCount>
signed_region_origins(SignedSynapseRegionSeedHash hash) {
  std::array<DevelopmentalSeedSite, kSignedRegionBranchCount> result{};
  const std::int32_t radius = signed_region_radius(hash);
  for (std::uint32_t branch = 0u; branch < kSignedRegionBranchCount; ++branch) {
    const Int3 ray =
        direction_offset(static_cast<Direction>(kSignedRegionReceiverLanes[branch]));
    result[branch] = {static_cast<std::int8_t>(ray.x * radius),
                      static_cast<std::int8_t>(ray.y * radius),
                      static_cast<std::int8_t>(ray.z * radius), kQ};
  }
  return result;
}

constexpr std::array<DevelopmentalSeedSite, kSignedRegionSeedSiteCount>
signed_synapse_region_seed(SignedSynapseRegionSeedHash hash) {
  std::array<DevelopmentalSeedSite, kSignedRegionSeedSiteCount> result{};
  const auto origins = signed_region_origins(hash);
  const bool valid = signed_region_gene(hash) == kSignedRegionGeneId;
  const std::uint8_t mask = signed_region_branch_mask(hash);
  const Int3 collar = direction_offset(Direction::positive_u0);
  for (std::uint32_t branch = 0u; branch < kSignedRegionBranchCount; ++branch) {
    if (!valid || (mask & (1u << branch)) == 0u)
      continue;
    const DevelopmentalSeedSite origin = origins[branch];
    result[branch * 2u] = {origin.x, origin.y, origin.z, kSignedRegionWeightZero};
    result[branch * 2u + 1u] = {
        static_cast<std::int8_t>(origin.x + collar.x),
        static_cast<std::int8_t>(origin.y + collar.y),
        static_cast<std::int8_t>(origin.z + collar.z),
        kSignedRegionCollar};
  }
  return result;
}

inline constexpr SignedSynapseRegionSeedHash kSignedSynapseRegionSeedHash =
    make_signed_synapse_region_seed_hash(0x7u, 12u, kSignedRegionGeneId);

static_assert(signed_region_branch_mask(kSignedSynapseRegionSeedHash) == 0x7u);
static_assert(signed_region_radius(kSignedSynapseRegionSeedHash) == 12u);
static_assert(signed_region_gene(kSignedSynapseRegionSeedHash) == kSignedRegionGeneId);

}  // namespace substrate::bcc32
