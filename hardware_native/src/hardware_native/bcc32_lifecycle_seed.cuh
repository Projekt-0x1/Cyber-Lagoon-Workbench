#pragma once

// Reusable local lifecycle matter.  These are ordinary BCC cells, not a host
// scheduler, TTL, garbage collector, or semantic cell-type dispatch.  F is
// the only runtime interpreter.  Fresh resource is an environmental molecule;
// its terminal waste is deliberately a different molecule and cannot replay as
// fuel.

#include <array>
#include <cstdint>

#include "bcc32_developmental_seed.cuh"
#include "bcc32_law.cuh"

namespace substrate::bcc32 {

inline constexpr std::array<DevelopmentalSeedSite, 2> kMetabolicCatalystSeed{{
    {-1, 0, 0, 0xe0011effu},
    {0, -1, 0, 0x022220ffu},
}};

constexpr SiteWord kMetabolicResourceZero = 0x10000000u;
constexpr SiteWord kMetabolicResourceOne = 0x20000000u;
constexpr SiteWord kMetabolicProduct = 0x00c0c0ffu;
constexpr SiteWord kMetabolicWaste = 0x0000087fu;
constexpr SiteWord kMetabolicProductMask = 0x0fff0000u;

// This is resource-bearing founder matter for a closed local assay.  In an
// adult the two resource bits must arrive through ordinary local transport;
// this helper never chooses a target or injects a later reward.
constexpr std::array<DevelopmentalSeedSite, 2> metabolic_flux_seed() {
  auto seed = kMetabolicCatalystSeed;
  seed[0].word |= kMetabolicResourceZero;
  seed[1].word |= kMetabolicResourceOne;
  return seed;
}

// A separate active, bounded structural body.  Its body changes between
// returns, so persistence is not a frozen crystal.  It is intentionally kept
// distinct from the catalyst until a local contact rule proves composition.
inline constexpr std::array<DevelopmentalSeedSite, 3> kActiveHomeostasisSeed{{
    {0, 0, 0, kQ | owned_bond_bit(0u) | owned_bond_bit(1u)},
    {1, 0, 0, kQ | owned_bond_bit(2u)},
    {0, 1, 0, kQ | owned_bond_bit(3u)},
}};

constexpr std::uint64_t lifecycle_seed_fingerprint() {
  std::uint64_t hash = 14695981039346656037ull;
  const auto add = [&hash](std::uint8_t byte) constexpr {
    hash ^= byte;
    hash *= 1099511628211ull;
  };
  const auto add_seed = [&add](const auto& seed) constexpr {
    for (const DevelopmentalSeedSite& site : seed) {
      add(static_cast<std::uint8_t>(site.x));
      add(static_cast<std::uint8_t>(site.y));
      add(static_cast<std::uint8_t>(site.z));
      for (std::uint32_t shift = 0u; shift < 32u; shift += 8u)
        add(static_cast<std::uint8_t>(site.word >> shift));
    }
  };
  add_seed(kMetabolicCatalystSeed);
  add_seed(kActiveHomeostasisSeed);
  return hash;
}

inline constexpr std::uint64_t kLifecycleSeedFingerprint = lifecycle_seed_fingerprint();

}  // namespace substrate::bcc32
