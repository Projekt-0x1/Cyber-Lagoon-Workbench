#pragma once

// A compact developmental gene made only of two copies of the measured
// reanchor cell type.  Each copy consists of B/E parents; all C/R, face, and
// descendant structure must arise under F.  The low hash bit selects which
// fresh-anchor port receives the second unit's initially-Q child.

#include <array>
#include <cstdint>

#include "bcc32_developmental_seed.cuh"
#include "bcc32_law.cuh"

namespace substrate::bcc32 {

using ReanchorPairHash = std::uint8_t;
constexpr ReanchorPairHash kFirstConstructionAreaHash = 0x00u;

[[nodiscard]] constexpr bool reanchor_pair_hash_valid(ReanchorPairHash hash) {
  return (hash & ~ReanchorPairHash{0x01u}) == 0u;
}

[[nodiscard]] constexpr std::array<DevelopmentalSeedSite, 4u> reanchor_pair_seed(
    ReanchorPairHash hash) {
  const bool port_three = (hash & 0x01u) != 0u;
  const std::int32_t child_x = port_three ? 1 : 0;
  const std::int32_t child_y = port_three ? 1 : 0;
  const std::int32_t child_z = port_three ? 1 : -1;
  constexpr SiteWord parent0 = kQ | owned_bond_bit(0u) | energy_bit(0u);
  constexpr SiteWord parent1 = kQ | owned_bond_bit(1u) | energy_bit(1u);
  return {{{-1, 0, 0, parent0},
           {0, -1, 0, parent1},
           {static_cast<std::int8_t>(child_x - 1),
            static_cast<std::int8_t>(child_y), static_cast<std::int8_t>(child_z), parent0},
           {static_cast<std::int8_t>(child_x), static_cast<std::int8_t>(child_y - 1),
            static_cast<std::int8_t>(child_z), parent1}}};
}

}  // namespace substrate::bcc32
