#pragma once

// Compact physical founder seed for the first BCC attraction phenotype.
//
// The seed is two ordinary B/E cells, not a host-side construction plan and
// not a frozen region layout.  Full F is the sole runtime interpreter: the
// two local founders attract a shared C/R trace on their initially-Q child and
// recruit fresh local anchors.  The fingerprint is an identity for this exact
// material recipe; F reads the material itself, never the fingerprint.

#include <array>
#include <cstdint>

#include "bcc32_developmental_seed.cuh"
#include "bcc32_law.cuh"

namespace substrate::bcc32 {

inline constexpr std::array<DevelopmentalSeedSite, 2> kEligibilityAttractionSeed{{
    {-1, 0, 0, kQ | owned_bond_bit(0u) | energy_bit(0u)},
    {0, -1, 0, kQ | owned_bond_bit(1u) | energy_bit(1u)},
}};

constexpr std::uint64_t attraction_seed_fingerprint() {
  std::uint64_t hash = 14695981039346656037ull;  // FNV-1a identity, not an interpreter.
  const auto add = [&hash](std::uint8_t byte) constexpr {
    hash ^= byte;
    hash *= 1099511628211ull;
  };
  for (const DevelopmentalSeedSite& site : kEligibilityAttractionSeed) {
    add(static_cast<std::uint8_t>(site.x));
    add(static_cast<std::uint8_t>(site.y));
    add(static_cast<std::uint8_t>(site.z));
    for (std::uint32_t shift = 0u; shift < 32u; shift += 8u)
      add(static_cast<std::uint8_t>(site.word >> shift));
  }
  return hash;
}

inline constexpr std::uint64_t kEligibilityAttractionSeedFingerprint =
    attraction_seed_fingerprint();

}  // namespace substrate::bcc32
