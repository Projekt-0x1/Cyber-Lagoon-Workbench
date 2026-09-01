#pragma once

// A compact developmental hash grows an eight-cell processive scalar weight
// plus stage-local carrier roads and a candidate receptor. Zero underflows in
// stages seven
// through one take private lateral detours and rejoin below their source so
// the vacancy reaches the next cell. Stage zero instead closes its own
// local candidate receptor. The receptor maps the settled underflow product
// into the next forward-compatible processive phase without a shared road or
// host pulse train.

#include <array>
#include <cstdint>

#include "bcc32_processive_weight_region_seed.cuh"

namespace substrate::bcc32 {

using ProcessiveTurnoverSeedHash = std::uint64_t;

inline constexpr std::uint32_t kProcessiveTurnoverGene = 0x79u;
inline constexpr std::uint32_t kProcessiveTurnoverCornerCount = 35u;
inline constexpr std::uint32_t kProcessiveTurnoverSeedSiteCount =
    8u * kProcessiveWeightSitesPerCell + 2u * kProcessiveTurnoverCornerCount + 1u;

constexpr ProcessiveTurnoverSeedHash make_processive_turnover_seed_hash(
    ProcessiveWeightRegionSeedHash weight_hash, std::uint32_t gene) {
  return static_cast<ProcessiveTurnoverSeedHash>(weight_hash) |
         (static_cast<ProcessiveTurnoverSeedHash>(gene & 0xffu) << 32u);
}

constexpr ProcessiveWeightRegionSeedHash processive_turnover_weight_hash(
    ProcessiveTurnoverSeedHash hash) {
  return static_cast<ProcessiveWeightRegionSeedHash>(hash);
}

constexpr std::uint32_t processive_turnover_gene(ProcessiveTurnoverSeedHash hash) {
  return static_cast<std::uint32_t>((hash >> 32u) & 0xffu);
}

constexpr bool valid_processive_turnover_seed_hash(ProcessiveTurnoverSeedHash hash) {
  return processive_turnover_gene(hash) == kProcessiveTurnoverGene &&
         valid_processive_weight_region_hash(processive_turnover_weight_hash(hash)) &&
         processive_weight_length(processive_turnover_weight_hash(hash)) == 8u;
}

constexpr Int3 scaled(Int3 value, std::int32_t factor) {
  return {value.x * factor, value.y * factor, value.z * factor};
}

constexpr DevelopmentalSeedSite turnover_seed_site(Int3 coordinate, SiteWord word) {
  return {static_cast<std::int8_t>(coordinate.x), static_cast<std::int8_t>(coordinate.y),
          static_cast<std::int8_t>(coordinate.z), word};
}

constexpr std::array<DevelopmentalSeedSite, kProcessiveTurnoverSeedSiteCount>
processive_turnover_seed(ProcessiveTurnoverSeedHash hash) {
  std::array<DevelopmentalSeedSite, kProcessiveTurnoverSeedSiteCount> result{};
  if (!valid_processive_turnover_seed_hash(hash))
    return result;

  const ProcessiveWeightRegionSeedHash weight_hash = processive_turnover_weight_hash(hash);
  const std::uint32_t marker = processive_weight_marker(weight_hash);
  const std::uint32_t path = processive_weight_path(weight_hash);
  const std::uint32_t waste = processive_weight_waste(weight_hash);
  const Int3 marker_ray = direction_offset(static_cast<Direction>(marker));
  const Int3 path_ray = direction_offset(static_cast<Direction>(path));
  const Int3 waste_ray = direction_offset(static_cast<Direction>(waste));

  const auto weight = processive_weight_region_seed(weight_hash);
  std::uint32_t next = 0u;
  for (std::uint32_t index = 0u; index < 8u * kProcessiveWeightSitesPerCell; ++index) {
    DevelopmentalSeedSite site = weight[index];
    if (index % kProcessiveWeightSitesPerCell == 0u)
      site.word = processive_weight_one_word(weight_hash);
    result[next++] = site;
  }
  // Genesis includes one negative-path vacancy at the full-weight tail.
  result[(8u - 1u) * kProcessiveWeightSitesPerCell].word ^= carrier_bit(path + 4u);

  const auto append_corner = [&](Int3 center, std::uint32_t incoming, std::uint32_t outgoing) {
    for (std::uint32_t index = 1u; index < kCarrierCornerSiteCount; ++index) {
      const CarrierCornerOffset relative = carrier_corner_offset(incoming, outgoing, index);
      result[next++] = turnover_seed_site(center + Int3{relative.x, relative.y, relative.z},
                                          carrier_corner_word(incoming, outgoing, index, false));
    }
  };

  // A zero-stage underflow exits three sites down the negative-waste ray.
  // Five corners take a stage-private lateral lane and rejoin the negative
  // path halfway toward the next lower stage. Alternating lateral signs keeps
  // neighbouring lock footprints disjoint. Rejoining ballistically at
  // different coordinates preserves route phase; there is no many-to-one
  // corner.
  for (std::uint32_t stage = 1u; stage < 8u; ++stage) {
    const Int3 body = scaled(path_ray, 6 * static_cast<std::int32_t>(stage));
    const Int3 outlet = body + -scaled(waste_ray, 3);
    const bool positive_lateral = (stage & 1u) == 0u;
    const std::uint32_t lateral = positive_lateral ? marker : marker + 4u;
    const std::uint32_t return_lateral = lateral ^ 4u;
    const std::int32_t reach = 14 + static_cast<std::int32_t>(stage);
    const Int3 lateral_ray = direction_offset(static_cast<Direction>(lateral));
    const Int3 far = outlet + scaled(lateral_ray, reach);
    const Int3 lower = far + -scaled(path_ray, 3);
    const Int3 lower_road = lower + scaled(waste_ray, 3);
    const Int3 rejoin = body + -scaled(path_ray, 3);
    append_corner(outlet, waste + 4u, lateral);
    append_corner(far, lateral, path + 4u);
    append_corner(lower, path + 4u, waste);
    append_corner(lower_road, waste, return_lateral);
    append_corner(rejoin, return_lateral, path + 4u);
  }

  result[next++] = turnover_seed_site(scaled(waste_ray, 10), kQ | face_bit(waste));

  return result;
}

inline constexpr ProcessiveTurnoverSeedHash kProcessiveTurnoverSeedHash =
    make_processive_turnover_seed_hash(kProcessiveWeightRegionSeedHash, kProcessiveTurnoverGene);

static_assert(valid_processive_turnover_seed_hash(kProcessiveTurnoverSeedHash));

}  // namespace substrate::bcc32
