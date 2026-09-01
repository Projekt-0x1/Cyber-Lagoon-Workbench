#pragma once

// Compact physical founder seed for a local prediction-mismatch trace.
//
// This is the deterministic reduction of the existing route observer: three
// causal core cells and two dual-rail ports.  It is a material recipe, not a
// word/token module, and full F plus a reciprocal raw contact are the only
// runtime dynamics.  Its fingerprint identifies the recipe; it is never read
// by the cells.

#include <array>
#include <cstdint>

#include "bcc32_developmental_seed.cuh"
#include "bcc32_law.cuh"
#include "bcc32_raw_byte_tape.cuh"

namespace substrate::bcc32 {

constexpr std::uint8_t kErrorAttractionExpectedByte = 0xa4u;
constexpr std::size_t kErrorAttractionCoreCount = 3u;
constexpr std::size_t kErrorAttractionPortZeroIndex = kErrorAttractionCoreCount;
constexpr std::size_t kErrorAttractionPortOneIndex = kErrorAttractionCoreCount + 1u;
inline constexpr RawByteRails kErrorAttractionExpectedRails =
    with_raw_byte_carriers(RawByteRails{}, kErrorAttractionExpectedByte);

inline constexpr std::array<DevelopmentalSeedSite, 5> kErrorAttractionSeed{{
    {0, 0, 0, kQ | owned_bond_bit(3u)},
    {-1, -1, -1, (kQ & ~carrier_bit(3u)) | owned_bond_bit(3u)},
    {-1, -1, 0, 0x140018ffu},
    {-1, -1, -2, kErrorAttractionExpectedRails.zero},
    {-1, -4, 0, kErrorAttractionExpectedRails.one},
}};
static_assert(kErrorAttractionPortOneIndex + 1u == kErrorAttractionSeed.size());

constexpr std::uint64_t error_attraction_seed_fingerprint() {
  std::uint64_t hash = 14695981039346656037ull;
  const auto add = [&hash](std::uint8_t byte) constexpr {
    hash ^= byte;
    hash *= 1099511628211ull;
  };
  for (const DevelopmentalSeedSite& site : kErrorAttractionSeed) {
    add(static_cast<std::uint8_t>(site.x));
    add(static_cast<std::uint8_t>(site.y));
    add(static_cast<std::uint8_t>(site.z));
    for (std::uint32_t shift = 0u; shift < 32u; shift += 8u)
      add(static_cast<std::uint8_t>(site.word >> shift));
  }
  return hash;
}

inline constexpr std::uint64_t kErrorAttractionSeedFingerprint =
    error_attraction_seed_fingerprint();

}  // namespace substrate::bcc32
