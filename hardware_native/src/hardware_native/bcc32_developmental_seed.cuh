#pragma once

// Literal BCC developmental seeds.
//
// A seed is initial *matter*, not a host-side build plan.  The only payload
// variation is represented in the parent cell's C lanes.  Once placed, full F
// is the sole interpreter: it turns those C lanes into two spatially distinct
// daughter E lanes.  This is deliberately a tiny generic developmental
// alphabet, not a complete constructor or organism.

#include <array>
#include <cstdint>

#include "bcc32_types.cuh"

namespace substrate::bcc32 {

struct DevelopmentalSeedSite {
  // Keep the genome transportable without pulling the arbitrary-precision
  // observer coordinate type into the seed encoding itself.
  std::int8_t x;
  std::int8_t y;
  std::int8_t z;
  SiteWord word;
};

// The first executable witness word.  It is a 2-bit developmental genome held
// physically in the parent C0/C1 lanes.  Its content-addressed digest belongs
// above this layer; this word is the compact material that F actually reads.
using DevelopmentalGenomeWord = std::uint8_t;
constexpr DevelopmentalGenomeWord kTwoBitDevelopmentWitnessGenome = 0x3u;

constexpr std::array<DevelopmentalSeedSite, 5> developmental_two_bit_seed(
    DevelopmentalGenomeWord genome) {
  // Keep the developmental alphabet total and explicit: only the two physical
  // C lanes are readable by this first constructor.  There is no host dispatch
  // from a word/role label to a graph layout.
  genome &= 0x3u;
  SiteWord parent = 0x0200005fu;
  if ((genome & 0x1u) != 0u)
    parent |= channel_bit(kConformationShift, 0u);
  if ((genome & 0x2u) != 0u)
    parent |= channel_bit(kConformationShift, 1u);

  return {{
      {-1, -1, -1, 0x008000ffu},
      {-1, 0, 0, 0xc2400a66u},
      {0, -1, 0, 0x401c04f2u},
      {0, 0, 0, parent},
      {1, 1, 1, 0x000008f7u},
  }};
}

// A portable germ is a compact *physical* genome, not a hash lookup into a
// host-side region recipe.  Every two-bit locus uses the identical material
// alphabet above; the only varying matter is its two C-lane bits.  The fixed
// eight-plane spacing is germ packaging, not a feature graph: after placement
// F is the sole interpreter and each locus develops independently in two
// supersteps.  A content hash may name this word above the substrate, but the
// word itself is what the cells carry.
using CompactDevelopmentalGenome = std::uint32_t;
constexpr std::size_t kCompactGenomeLocusCount = 16u;
constexpr std::int8_t kCompactGenomeLocusStrideZ = 8;
constexpr std::size_t kDevelopmentalSitesPerLocus = developmental_two_bit_seed(0u).size();

constexpr std::array<DevelopmentalSeedSite,
                     kCompactGenomeLocusCount * kDevelopmentalSitesPerLocus>
compact_developmental_genome_seed(CompactDevelopmentalGenome genome) {
  std::array<DevelopmentalSeedSite,
             kCompactGenomeLocusCount * kDevelopmentalSitesPerLocus> result{};
  std::size_t cursor = 0u;
  for (std::size_t locus = 0u; locus < kCompactGenomeLocusCount; ++locus) {
    const DevelopmentalGenomeWord local = static_cast<DevelopmentalGenomeWord>(
        (genome >> (2u * locus)) & 0x3u);
    const std::int8_t z = static_cast<std::int8_t>(locus * kCompactGenomeLocusStrideZ);
    for (DevelopmentalSeedSite site : developmental_two_bit_seed(local)) {
      site.z = static_cast<std::int8_t>(site.z + z);
      result[cursor++] = site;
    }
  }
  return result;
}

}  // namespace substrate::bcc32
