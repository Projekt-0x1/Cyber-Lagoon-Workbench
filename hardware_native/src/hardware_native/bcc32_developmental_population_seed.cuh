#pragma once

// Physical developmental germ plus brain-scale state-mass bounds.
//
// The only variable program is a 256-bit hash carried by ordinary C-lane
// matter.  The 640-site germ packaging is fixed scaffolding: after genesis the
// one reversible BCC law F is the sole interpreter.  No host cell-type list,
// graph recipe, contact table, second CA law, or virtual organism is selected
// by the hash.
//
// The 80B neutral cells below are exact coordinates containing Q.  They have
// stable identity without payload.  A cell becomes stateful only when finite
// non-Q matter reaches that coordinate through F.  All persistent cell,
// molecule, field, contact, weight, scar, and waste state shares the existing
// direct dense chunk pager and the one 100 GB organism-image ceiling.

#include <array>
#include <cstddef>
#include <cstdint>

#include "bcc32_developmental_seed.cuh"
#include "bcc32_law.cuh"
#include "bcc32_types.cuh"

namespace substrate::bcc32 {

struct DevelopmentalHash {
  std::uint64_t lane0;
  std::uint64_t lane1;
  std::uint64_t lane2;
  std::uint64_t lane3;

  friend constexpr bool operator==(const DevelopmentalHash&, const DevelopmentalHash&) = default;
};

// First deliberately designed physical germ.
constexpr DevelopmentalHash kFirstPopulationHash{
    0x6d3a91c5c71b2e49ull,
    0xa24baed4963ee407ull,
    0x9fb21c651e98df25ull,
    0xd1b54a32d192ed03ull,
};

constexpr std::uint32_t kDevelopmentalHashBits = 256u;
constexpr std::size_t kPopulationLocusCount = kDevelopmentalHashBits / 2u;
constexpr std::size_t kPopulationSeedSiteCount =
    kPopulationLocusCount * kDevelopmentalSitesPerLocus;
constexpr std::int8_t kPopulationGridX = 8;
constexpr std::int8_t kPopulationGridY = 8;
constexpr std::int8_t kPopulationGridZ = 2;
constexpr std::int8_t kPopulationLocusStride = 8;
constexpr std::int8_t kPopulationPlaneStride = 16;

constexpr std::uint64_t developmental_hash_lane(DevelopmentalHash hash, std::uint32_t lane) {
  switch (lane & 3u) {
    case 0u:
      return hash.lane0;
    case 1u:
      return hash.lane1;
    case 2u:
      return hash.lane2;
    default:
      return hash.lane3;
  }
}

constexpr DevelopmentalGenomeWord developmental_hash_locus(DevelopmentalHash hash,
                                                           std::size_t locus) {
  const std::size_t lane = locus / 32u;
  const std::size_t within_lane = locus % 32u;
  return static_cast<DevelopmentalGenomeWord>(
      (developmental_hash_lane(hash, static_cast<std::uint32_t>(lane)) >> (2u * within_lane)) &
      0x3u);
}

constexpr std::array<DevelopmentalSeedSite, kPopulationSeedSiteCount> developmental_population_seed(
    DevelopmentalHash hash) {
  std::array<DevelopmentalSeedSite, kPopulationSeedSiteCount> result{};
  std::size_t cursor = 0u;
  for (std::size_t locus = 0u; locus < kPopulationLocusCount; ++locus) {
    const std::size_t plane = locus / static_cast<std::size_t>(kPopulationGridX * kPopulationGridY);
    const std::size_t in_plane =
        locus % static_cast<std::size_t>(kPopulationGridX * kPopulationGridY);
    const std::size_t row = in_plane / static_cast<std::size_t>(kPopulationGridX);
    const std::size_t column = in_plane % static_cast<std::size_t>(kPopulationGridX);
    const std::int8_t offset_x =
        static_cast<std::int8_t>(static_cast<std::int32_t>(column) * kPopulationLocusStride -
                                 ((kPopulationGridX - 1) * kPopulationLocusStride) / 2);
    const std::int8_t offset_y =
        static_cast<std::int8_t>(static_cast<std::int32_t>(row) * kPopulationLocusStride -
                                 ((kPopulationGridY - 1) * kPopulationLocusStride) / 2);
    const std::int8_t offset_z =
        static_cast<std::int8_t>(static_cast<std::int32_t>(plane) * kPopulationPlaneStride -
                                 ((kPopulationGridZ - 1) * kPopulationPlaneStride) / 2);
    for (DevelopmentalSeedSite site :
         developmental_two_bit_seed(developmental_hash_locus(hash, locus))) {
      site.x = static_cast<std::int8_t>(site.x + offset_x);
      site.y = static_cast<std::int8_t>(site.y + offset_y);
      site.z = static_cast<std::int8_t>(site.z + offset_z);
      result[cursor++] = site;
    }
  }
  return result;
}

// One generic precursor matter word.  It is not a cell type tag: F sees only
// its ordinary B/E quanta.  Development may redistribute these quanta but may
// not mint more.
constexpr SiteWord kDevelopmentalPrecursorWord =
    kQuiescentWord | owned_bond_bit(0u) | owned_bond_bit(1u) | energy_bit(0u) | energy_bit(1u);

// Exact neutral-cell coordinate field.  One 80B half is the neuron-scale
// target and one 80B half is the comparable support/glia-scale target.  Cell
// roles are not assigned by coordinate; both halves begin as identical Q and
// later phenotype must be measured.  This is deliberately distinct from the
// existing 80B-bit / 2.5B-site device aperture.
constexpr std::uint32_t kDevelopmentalTissueX = 4'000u;
constexpr std::uint32_t kDevelopmentalTissueY = 4'000u;
constexpr std::uint32_t kDevelopmentalTissueZ = 10'000u;
constexpr std::uint64_t kDevelopmentalLogicalCells =
    static_cast<std::uint64_t>(kDevelopmentalTissueX) * kDevelopmentalTissueY *
    kDevelopmentalTissueZ;
constexpr std::uint64_t kDevelopmentalNeuronScaleTargetCells = 80'000'000'000ull;
constexpr std::uint64_t kDevelopmentalSupportScaleTargetCells = 80'000'000'000ull;

struct LogicalCellCoordinate {
  std::uint32_t x;
  std::uint32_t y;
  std::uint32_t z;
};

using LogicalCellId = std::uint64_t;

constexpr bool valid_cell(LogicalCellId identity) {
  return identity < kDevelopmentalLogicalCells;
}

constexpr LogicalCellCoordinate cell_coordinate(LogicalCellId identity) {
  const std::uint32_t x = static_cast<std::uint32_t>(identity % kDevelopmentalTissueX);
  identity /= kDevelopmentalTissueX;
  const std::uint32_t y = static_cast<std::uint32_t>(identity % kDevelopmentalTissueY);
  identity /= kDevelopmentalTissueY;
  return {x, y, static_cast<std::uint32_t>(identity)};
}

constexpr LogicalCellId cell_identity(LogicalCellCoordinate coordinate) {
  return (static_cast<LogicalCellId>(coordinate.z) * kDevelopmentalTissueY + coordinate.y) *
             kDevelopmentalTissueX +
         coordinate.x;
}

// Human-scale contact count is an explicit target, not a proved current
// phenotype.  A contact becomes real only when F grows a path/contact geometry;
// this arithmetic never substitutes for that tissue.
constexpr std::uint16_t kDevelopmentalTargetContactsPerCell = 1'875u;
constexpr std::uint64_t kDevelopmentalTargetContacts =
    kDevelopmentalNeuronScaleTargetCells * kDevelopmentalTargetContactsPerCell;
constexpr std::uint64_t kDevelopmentalTargetCableKilometresLow = 100'000ull;
constexpr std::uint64_t kDevelopmentalTargetCableKilometresHigh = 150'000ull;
constexpr std::uint32_t kDevelopmentalTargetResourceMilliwatts = 20'000u;

// One AI must remain below 100 GB.  Q chunks are canonical and need no payload;
// every non-Q chunk is still one exact direct 4 MB array.  Four GB are reserved
// for manifests, roots, schedules, and other exact indices.  The remaining
// 96 GB permit at most 24,000 materialized chunks / 24B direct BCC sites.
constexpr std::uint64_t kDevelopmentalImageByteCeiling = 100'000'000'000ull;
constexpr std::uint64_t kDevelopmentalDirectMatterBytes = 96'000'000'000ull;
constexpr std::uint64_t kDevelopmentalIndexAndManifestBytes = 4'000'000'000ull;
constexpr std::uint64_t kDevelopmentalDirectChunkCapacity =
    kDevelopmentalDirectMatterBytes / kChunkBytes;
constexpr std::uint64_t kDevelopmentalMaterializedSiteCapacity =
    kDevelopmentalDirectChunkCapacity * kChunkSites;

static_assert(kPopulationLocusCount == 128u);
static_assert(kPopulationSeedSiteCount == 640u);
static_assert(kDevelopmentalPrecursorWord == 0x300300ffu);
static_assert(kDevelopmentalLogicalCells == 160'000'000'000ull);
static_assert(kDevelopmentalNeuronScaleTargetCells + kDevelopmentalSupportScaleTargetCells ==
              kDevelopmentalLogicalCells);
static_assert(kDevelopmentalTargetContacts == 150'000'000'000'000ull);
static_assert(kDevelopmentalDirectMatterBytes + kDevelopmentalIndexAndManifestBytes ==
              kDevelopmentalImageByteCeiling);
static_assert(kDevelopmentalDirectChunkCapacity == 24'000ull);
static_assert(kDevelopmentalMaterializedSiteCapacity == 24'000'000'000ull);
static_assert(kProductionSites == 2'500'000'000ull);
static_assert(kProductionBits == 80'000'000'000ull);
static_assert(kProductionBytes == 10'000'000'000ull);

}  // namespace substrate::bcc32
