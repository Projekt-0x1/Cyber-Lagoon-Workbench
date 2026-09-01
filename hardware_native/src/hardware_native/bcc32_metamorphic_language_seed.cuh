#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

#include "bcc32_law.cuh"
#include "bcc32_raw_byte_tape.cuh"

namespace substrate::bcc32::metamorphic_seed {

struct SeedSite {
  Int3 coordinate{};
  SiteWord word = kQ;

  friend constexpr bool operator==(const SeedSite&, const SeedSite&) = default;
};

struct RawPortPair {
  Int3 zero{};
  Int3 one{};

  friend constexpr bool operator==(const RawPortPair&, const RawPortPair&) = default;
};

constexpr std::uint32_t kRawBits = 8u;
constexpr std::uint32_t kReplicasPerRawBit = 2u;
constexpr std::uint32_t kModuleCopies = kRawBits * kReplicasPerRawBit;
constexpr std::uint32_t kCoreSiteCount = 10u;
constexpr std::uint32_t kRawPortSiteCount = 2u;
constexpr std::uint32_t kSitesPerModule = kCoreSiteCount + kRawPortSiteCount;
constexpr std::uint32_t kSeedSiteCount = kModuleCopies * kSitesPerModule;
constexpr std::int32_t kModuleStride = 300;
constexpr std::int32_t kModuleCenter = 50;

// This is one ordinary mutable route locus: a bounded intake body, a two-site
// plasticity/reversal locus, and balanced turnover matter. No word is a role
// tag and the law has no dispatch for this layout.
inline constexpr std::array<SeedSite, kCoreSiteCount> kRouteLocusCore{{
    {{0, 0, 0}, kQ | owned_bond_bit(3u)},
    {{1, 0, 0}, kQ | owned_bond_bit(0u) | owned_bond_bit(2u)},
    {{0, 1, 0}, kQ | owned_bond_bit(0u) | owned_bond_bit(1u)},
    {{0, 0, 1}, kQ | owned_bond_bit(2u)},
    {{-1, -1, -1}, (kQ & ~carrier_bit(3u)) | owned_bond_bit(3u)},
    {{-1, -2, 0}, 0x000040ffu},
    {{-1, -1, 0}, 0x140018ffu},
    {{-2, 0, 0}, 0x002002f6u},
    {{0, 1, 1}, 0x002080dbu},
    {{1, 1, 2}, 0x800010ebu},
}};

// Each raw bit has a tested local placement. A canonical RawByteRails pair is
// still presented at every placement; these coordinates merely select which
// directional quantum meets the nearby route locus under production F.
inline constexpr std::array<RawPortPair, kRawBits> kRawPortByBit{{
    {{-1, -1, -2}, {-1, -4, 0}},
    {{-1, -4, 0}, {-1, -1, -2}},
    {{-1, -1, -2}, {-1, -4, 0}},
    {{-1, -1, -2}, {-1, -4, 0}},
    {{-2, -2, -2}, {-1, -1, -2}},
    {{-2, -2, -2}, {-1, -1, -2}},
    {{-2, -2, -2}, {-1, -1, -2}},
    {{-1, -1, -2}, {-1, -1, 1}},
}};

[[nodiscard]] __host__ __device__ constexpr std::uint32_t module_target_bit(std::uint32_t copy) {
  return copy % kRawBits;
}

[[nodiscard]] __host__ __device__ constexpr std::uint32_t module_replica(std::uint32_t copy) {
  return copy / kRawBits;
}

[[nodiscard]] __host__ __device__ constexpr std::uint8_t module_low_byte(std::uint32_t copy) {
  return static_cast<std::uint8_t>(0xa5u & ~(1u << module_target_bit(copy)));
}

[[nodiscard]] __host__ __device__ constexpr std::uint8_t module_high_byte(std::uint32_t copy) {
  return static_cast<std::uint8_t>(module_low_byte(copy) | (1u << module_target_bit(copy)));
}

[[nodiscard]] __host__ __device__ constexpr Int3 module_offset(std::uint32_t copy) {
  return {static_cast<std::int32_t>(copy) * kModuleStride + kModuleCenter, kModuleCenter,
          kModuleCenter};
}

[[nodiscard]] __host__ __device__ constexpr Int3 translated(Int3 coordinate, Int3 offset) {
  return {coordinate.x + offset.x, coordinate.y + offset.y, coordinate.z + offset.z};
}

[[nodiscard]] constexpr RawPortPair module_raw_port(std::uint32_t copy) {
  const Int3 offset = module_offset(copy);
  const RawPortPair local = kRawPortByBit[module_target_bit(copy)];
  return {translated(local.zero, offset), translated(local.one, offset)};
}

[[nodiscard]] constexpr std::array<SeedSite, kSeedSiteCount> make_metamorphic_language_seed() {
  std::array<SeedSite, kSeedSiteCount> result{};
  std::size_t cursor = 0u;
  for (std::uint32_t copy = 0u; copy < kModuleCopies; ++copy) {
    const Int3 offset = module_offset(copy);
    for (const SeedSite& site : kRouteLocusCore)
      result[cursor++] = {translated(site.coordinate, offset), site.word};

    const RawPortPair port = module_raw_port(copy);
    const RawByteRails prediction = with_raw_byte_carriers(RawByteRails{}, module_low_byte(copy));
    result[cursor++] = {port.zero, prediction.zero};
    result[cursor++] = {port.one, prediction.one};
  }
  return result;
}

inline constexpr auto kMetamorphicLanguageSeed = make_metamorphic_language_seed();

static_assert(kSeedSiteCount == 192u);
static_assert(kMetamorphicLanguageSeed.size() == kSeedSiteCount);
static_assert(module_target_bit(0u) == 0u && module_target_bit(7u) == 7u);
static_assert(module_target_bit(8u) == 0u && module_replica(8u) == 1u);

}  // namespace substrate::bcc32::metamorphic_seed
