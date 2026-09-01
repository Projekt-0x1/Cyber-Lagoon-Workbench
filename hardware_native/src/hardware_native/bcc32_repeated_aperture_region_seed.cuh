#pragma once

// A compact branching seed for the first repeated-aperture region.
//
// The hash does not contain finished site words.  It selects a reusable local
// germ, a three-bud enable mask, and one scale.  The decoder applies the same
// developmental germ at a root and two tetrahedral daughter offsets.  Those
// are founder pools only: the E/C/R receptor, aperture, and repeated response
// must still be produced by ordinary contacts and F.

#include <array>
#include <cstddef>
#include <cstdint>

#include "bcc32_credit_latch_weight_seed.cuh"

namespace substrate::bcc32 {

using RepeatedApertureRegionSeedHash = std::uint32_t;

constexpr std::uint32_t kRepeatedApertureRegionBranchMaskShift = 0u;
constexpr std::uint32_t kRepeatedApertureRegionSpacingShift = 3u;
constexpr std::uint32_t kRepeatedApertureRegionGeneShift = 9u;
constexpr std::uint32_t kRepeatedApertureRegionReservedShift = 17u;
constexpr std::uint32_t kRepeatedApertureRegionBranchCount = 3u;
constexpr std::uint32_t kRepeatedApertureRegionGeneId = 1u;
constexpr std::size_t kRepeatedApertureRegionSeedSiteCount =
    kRepeatedApertureRegionBranchCount * kCreditLatchWeightSeedSiteCount;

constexpr RepeatedApertureRegionSeedHash make_repeated_aperture_region_seed_hash(
    std::uint8_t branch_mask, std::uint8_t spacing, std::uint8_t gene_id) {
  return (static_cast<RepeatedApertureRegionSeedHash>(branch_mask & 0x7u)
          << kRepeatedApertureRegionBranchMaskShift) |
         (static_cast<RepeatedApertureRegionSeedHash>(spacing & 0x3fu)
          << kRepeatedApertureRegionSpacingShift) |
         (static_cast<RepeatedApertureRegionSeedHash>(gene_id) << kRepeatedApertureRegionGeneShift);
}

constexpr std::uint8_t repeated_aperture_region_branch_mask(RepeatedApertureRegionSeedHash hash) {
  return static_cast<std::uint8_t>((hash >> kRepeatedApertureRegionBranchMaskShift) & 0x7u);
}

constexpr std::uint8_t repeated_aperture_region_spacing(RepeatedApertureRegionSeedHash hash) {
  return static_cast<std::uint8_t>((hash >> kRepeatedApertureRegionSpacingShift) & 0x3fu);
}

constexpr std::uint8_t repeated_aperture_region_gene(RepeatedApertureRegionSeedHash hash) {
  return static_cast<std::uint8_t>(hash >> kRepeatedApertureRegionGeneShift);
}

constexpr std::array<DevelopmentalSeedSite, kRepeatedApertureRegionBranchCount>
repeated_aperture_region_origins(RepeatedApertureRegionSeedHash hash) {
  const std::int8_t spacing = static_cast<std::int8_t>(repeated_aperture_region_spacing(hash));
  return {{{0, 0, 0, kQ}, {spacing, 0, 0, kQ}, {0, spacing, 0, kQ}}};
}

constexpr std::array<DevelopmentalSeedSite, kRepeatedApertureRegionSeedSiteCount>
repeated_aperture_region_seed(RepeatedApertureRegionSeedHash hash) {
  std::array<DevelopmentalSeedSite, kRepeatedApertureRegionSeedSiteCount> result{};
  const auto germ = credit_latch_weight_seed(kCreditLatchWeightSeedHash);
  const auto origins = repeated_aperture_region_origins(hash);
  const std::uint8_t branch_mask = repeated_aperture_region_branch_mask(hash);
  for (std::size_t branch = 0u; branch < origins.size(); ++branch) {
    for (std::size_t site = 0u; site < germ.size(); ++site) {
      DevelopmentalSeedSite& output = result[branch * germ.size() + site];
      if ((branch_mask & (1u << branch)) == 0u ||
          (hash >> kRepeatedApertureRegionReservedShift) != 0u ||
          repeated_aperture_region_gene(hash) != kRepeatedApertureRegionGeneId) {
        output = {0, 0, 0, kQ};
        continue;
      }
      output = {static_cast<std::int8_t>(origins[branch].x + germ[site].x),
                static_cast<std::int8_t>(origins[branch].y + germ[site].y),
                static_cast<std::int8_t>(origins[branch].z + germ[site].z), germ[site].word};
    }
  }
  return result;
}

inline constexpr RepeatedApertureRegionSeedHash kRepeatedApertureRegionSeedHash =
    make_repeated_aperture_region_seed_hash(0x7u, 24u, kRepeatedApertureRegionGeneId);

static_assert(repeated_aperture_region_branch_mask(kRepeatedApertureRegionSeedHash) == 0x7u);
static_assert(repeated_aperture_region_spacing(kRepeatedApertureRegionSeedHash) == 24u);

}  // namespace substrate::bcc32
