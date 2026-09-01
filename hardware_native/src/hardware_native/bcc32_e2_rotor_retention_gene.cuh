#pragma once

// Law-derived two-cell collar grammar for the E2 molecular rotor.
//
// The rotor is B0+E1.  The collar varies only the basis-0 endpoint roles
// touched by K_edge after K_site creates E0: C0/R0 and the two opposing face
// controls.  It is the complete 4-bit local complement palette across each of
// the eight BCC neighbours, not a search over arbitrary words or a host rule.

#include <array>
#include <cstdint>

#include "bcc32_developmental_seed.cuh"
#include "bcc32_geometry.cuh"
#include "bcc32_law.cuh"

namespace substrate::bcc32 {

using E2RotorRetentionGene = std::uint8_t;
using E2RotorRetentionHash = std::uint8_t;
constexpr E2RotorRetentionGene kRetentionC0 = 0x01u;
constexpr E2RotorRetentionGene kRetentionR0 = 0x02u;
constexpr E2RotorRetentionGene kRetentionFacePositive0 = 0x04u;
constexpr E2RotorRetentionGene kRetentionFaceNegative0 = 0x08u;
constexpr E2RotorRetentionGene kRetentionPaletteMask = 0x0fu;
constexpr std::uint32_t kRetentionPortBits = 3u;

constexpr E2RotorRetentionHash make_e2_rotor_retention_hash(
    std::uint32_t port, E2RotorRetentionGene gene) {
  return static_cast<E2RotorRetentionHash>(
      (port & 0x07u) | ((static_cast<std::uint32_t>(gene) & kRetentionPaletteMask)
                         << kRetentionPortBits));
}
constexpr std::uint32_t e2_rotor_retention_port(E2RotorRetentionHash hash) {
  return static_cast<std::uint32_t>(hash & 0x07u);
}
constexpr E2RotorRetentionGene e2_rotor_retention_gene(E2RotorRetentionHash hash) {
  return static_cast<E2RotorRetentionGene>(
      (hash >> kRetentionPortBits) & kRetentionPaletteMask);
}

constexpr SiteWord e2_rotor_retention_collar_word(E2RotorRetentionGene gene) {
  SiteWord word = kQ;
  if ((gene & kRetentionC0) != 0u) word |= channel_bit(kConformationShift, 0u);
  if ((gene & kRetentionR0) != 0u) word |= channel_bit(kReactiveShift, 0u);
  if ((gene & kRetentionFacePositive0) != 0u) word |= face_bit(0u);
  if ((gene & kRetentionFaceNegative0) != 0u) word |= face_bit(4u);
  return word;
}

constexpr std::array<DevelopmentalSeedSite, 2u> e2_rotor_retention_seed(
    std::uint32_t port, E2RotorRetentionGene gene, bool prime = true) {
  const Int3 offset = direction_offset(static_cast<Direction>(port));
  SiteWord rotor = kQ | owned_bond_bit(0u);
  if (prime) rotor |= energy_bit(1u);
  return {{{0, 0, 0, rotor},
           {static_cast<std::int8_t>(offset.x), static_cast<std::int8_t>(offset.y),
            static_cast<std::int8_t>(offset.z), e2_rotor_retention_collar_word(gene)}}};
}

constexpr E2RotorRetentionHash kE2RotorRetentionCandidateHash =
    make_e2_rotor_retention_hash(
        0u, static_cast<E2RotorRetentionGene>(kRetentionR0 | kRetentionFaceNegative0));

constexpr std::array<DevelopmentalSeedSite, 2u> e2_rotor_retention_seed(
    E2RotorRetentionHash hash, bool prime = true) {
  return e2_rotor_retention_seed(e2_rotor_retention_port(hash),
                                 e2_rotor_retention_gene(hash), prime);
}

}  // namespace substrate::bcc32
