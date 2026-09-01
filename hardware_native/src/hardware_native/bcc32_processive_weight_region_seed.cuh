#pragma once

// A compact developmental hash expands one reusable processive-weight cell
// type along a straight BCC carrier road. Five vacant lattice sites between
// repeated bodies keep their bounded release footprints disjoint while carrier
// holes still cross the road under S_P. Two stable differentiated singleton
// locks name marker, waste, and the unused basis; path is the remaining basis.
// The hash contains no cell words, coordinates,
// weights, training history, or target answer. It selects only length, marker
// basis, road orientation, and the reusable gene.

#include <array>
#include <cstddef>
#include <cstdint>

#include "bcc32_developmental_seed.cuh"
#include "bcc32_geometry.cuh"
#include "bcc32_law.cuh"

namespace substrate::bcc32 {

using ProcessiveWeightRegionSeedHash = std::uint32_t;

inline constexpr std::uint32_t kProcessiveWeightLengthShift = 0u;
inline constexpr std::uint32_t kProcessiveWeightMarkerShift = 6u;
inline constexpr std::uint32_t kProcessiveWeightPathShift = 8u;
inline constexpr std::uint32_t kProcessiveWeightWasteShift = 10u;
inline constexpr std::uint32_t kProcessiveWeightGeneShift = 12u;
inline constexpr std::uint32_t kProcessiveWeightGene = 0x57u;
inline constexpr std::uint32_t kProcessiveWeightMaxLength = 32u;
inline constexpr std::uint32_t kProcessiveWeightSitesPerCell = 4u;
inline constexpr std::uint32_t kProcessiveWeightMaxSeedSites =
    kProcessiveWeightMaxLength * kProcessiveWeightSitesPerCell;

constexpr ProcessiveWeightRegionSeedHash make_processive_weight_region_seed_hash(
    std::uint32_t length, std::uint32_t marker, std::uint32_t path, std::uint32_t waste,
    std::uint32_t gene) {
  return static_cast<ProcessiveWeightRegionSeedHash>(length & 0x3fu) |
         static_cast<ProcessiveWeightRegionSeedHash>((marker & 0x3u)
                                                     << kProcessiveWeightMarkerShift) |
         static_cast<ProcessiveWeightRegionSeedHash>((path & 0x3u) << kProcessiveWeightPathShift) |
         static_cast<ProcessiveWeightRegionSeedHash>((waste & 0x3u)
                                                     << kProcessiveWeightWasteShift) |
         static_cast<ProcessiveWeightRegionSeedHash>((gene & 0xffu) << kProcessiveWeightGeneShift);
}

constexpr std::uint32_t processive_weight_length(ProcessiveWeightRegionSeedHash hash) {
  return (hash >> kProcessiveWeightLengthShift) & 0x3fu;
}

constexpr std::uint32_t processive_weight_marker(ProcessiveWeightRegionSeedHash hash) {
  return (hash >> kProcessiveWeightMarkerShift) & 0x3u;
}

constexpr std::uint32_t processive_weight_path(ProcessiveWeightRegionSeedHash hash) {
  return (hash >> kProcessiveWeightPathShift) & 0x3u;
}

constexpr std::uint32_t processive_weight_waste(ProcessiveWeightRegionSeedHash hash) {
  return (hash >> kProcessiveWeightWasteShift) & 0x3u;
}

constexpr std::uint32_t processive_weight_gene(ProcessiveWeightRegionSeedHash hash) {
  return (hash >> kProcessiveWeightGeneShift) & 0xffu;
}

constexpr bool valid_processive_weight_region_hash(ProcessiveWeightRegionSeedHash hash) {
  const std::uint32_t length = processive_weight_length(hash);
  return processive_weight_gene(hash) == kProcessiveWeightGene && length > 0u &&
         length <= kProcessiveWeightMaxLength &&
         processive_weight_marker(hash) != processive_weight_path(hash) &&
         processive_weight_marker(hash) != processive_weight_waste(hash) &&
         processive_weight_path(hash) != processive_weight_waste(hash);
}

constexpr SiteWord processive_weight_zero_word(ProcessiveWeightRegionSeedHash hash) {
  const std::uint32_t marker = processive_weight_marker(hash);
  return (kCarrierMask ^ carrier_bit(marker) ^ carrier_bit(marker + 4u)) | face_bit(marker + 4u) |
         energy_bit(marker);
}

constexpr SiteWord processive_weight_one_word(ProcessiveWeightRegionSeedHash hash) {
  const std::uint32_t marker = processive_weight_marker(hash);
  return (processive_weight_zero_word(hash) ^ energy_bit(marker)) | owned_bond_bit(marker);
}

constexpr std::array<DevelopmentalSeedSite, kProcessiveWeightMaxSeedSites>
processive_weight_region_seed(ProcessiveWeightRegionSeedHash hash) {
  std::array<DevelopmentalSeedSite, kProcessiveWeightMaxSeedSites> result{};
  if (!valid_processive_weight_region_hash(hash))
    return result;
  const std::uint32_t length = processive_weight_length(hash);
  const Int3 ray = direction_offset(static_cast<Direction>(processive_weight_path(hash)));
  const Int3 waste_ray = direction_offset(static_cast<Direction>(processive_weight_waste(hash)));
  for (std::uint32_t index = 0u; index < length; ++index) {
    const std::int32_t scale = 6 * static_cast<std::int32_t>(index);
    const std::int32_t x = ray.x * scale;
    const std::int32_t y = ray.y * scale;
    const std::int32_t z = ray.z * scale;
    result[index * 4u] = {static_cast<std::int8_t>(x), static_cast<std::int8_t>(y),
                          static_cast<std::int8_t>(z), processive_weight_zero_word(hash)};
    result[index * 4u + 1u] = {
        static_cast<std::int8_t>(
            x + direction_offset(static_cast<Direction>(processive_weight_marker(hash))).x),
        static_cast<std::int8_t>(
            y + direction_offset(static_cast<Direction>(processive_weight_marker(hash))).y),
        static_cast<std::int8_t>(
            z + direction_offset(static_cast<Direction>(processive_weight_marker(hash))).z),
        (kCarrierMask ^ carrier_bit(processive_weight_marker(hash)) ^
         carrier_bit(processive_weight_marker(hash) + 4u)) |
            owned_bond_bit(processive_weight_marker(hash))};
    std::uint32_t free_basis = 0u;
    for (std::uint32_t basis = 0u; basis < 4u; ++basis)
      if (basis != processive_weight_marker(hash) && basis != processive_weight_path(hash) &&
          basis != processive_weight_waste(hash))
        free_basis = basis;
    result[index * 4u + 2u] = {static_cast<std::int8_t>(x + 6 * waste_ray.x),
                               static_cast<std::int8_t>(y + 6 * waste_ray.y),
                               static_cast<std::int8_t>(z + 6 * waste_ray.z),
                               kQ | face_bit(free_basis)};
    result[index * 4u + 3u] = {static_cast<std::int8_t>(x + 8 * waste_ray.x),
                               static_cast<std::int8_t>(y + 8 * waste_ray.y),
                               static_cast<std::int8_t>(z + 8 * waste_ray.z),
                               kQ | face_bit(processive_weight_marker(hash))};
  }
  return result;
}

inline constexpr ProcessiveWeightRegionSeedHash kProcessiveWeightRegionSeedHash = 0x00057908u;

static_assert(processive_weight_length(kProcessiveWeightRegionSeedHash) == 8u);
static_assert(processive_weight_marker(kProcessiveWeightRegionSeedHash) == 0u);
static_assert(processive_weight_path(kProcessiveWeightRegionSeedHash) == 1u);
static_assert(processive_weight_waste(kProcessiveWeightRegionSeedHash) == 2u);
static_assert(processive_weight_gene(kProcessiveWeightRegionSeedHash) == kProcessiveWeightGene);
static_assert(valid_processive_weight_region_hash(kProcessiveWeightRegionSeedHash));

}  // namespace substrate::bcc32
