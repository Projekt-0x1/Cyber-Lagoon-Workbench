#pragma once

// Minimal molecular rotor gene for converting an offered E2 quantum into a
// candidate relay-ready E0 state.  Its only resident matter is B0 plus E1.
// When an E2 quantum arrives, the E tetrad has exactly two occupied bases, so
// K_site flips the tetrad.  The full-F contract is deliberately the arbiter:
// it currently proves that this one-cell gene does *not* retain B0+E0 at any
// site after the complete superstep.  The hash names neither input nor route.

#include <array>
#include <cstdint>

#include "bcc32_developmental_seed.cuh"
#include "bcc32_law.cuh"

namespace substrate::bcc32 {

using E2EnergyRotorHash = std::uint8_t;
constexpr E2EnergyRotorHash kE2EnergyRotorBondBit = 0x01u;
constexpr E2EnergyRotorHash kE2EnergyRotorPrimeBit = 0x02u;
constexpr E2EnergyRotorHash kE2EnergyRotorHash =
    kE2EnergyRotorBondBit | kE2EnergyRotorPrimeBit;
constexpr E2EnergyRotorHash kE2EnergyRotorPrimeLesionHash =
    kE2EnergyRotorBondBit;
constexpr E2EnergyRotorHash kE2EnergyRotorBondLesionHash =
    kE2EnergyRotorPrimeBit;

constexpr SiteWord e2_energy_rotor_word(E2EnergyRotorHash hash) {
  SiteWord word = kQ;
  if ((hash & kE2EnergyRotorBondBit) != 0u) word |= owned_bond_bit(0u);
  if ((hash & kE2EnergyRotorPrimeBit) != 0u) word |= energy_bit(1u);
  return word;
}

constexpr std::array<DevelopmentalSeedSite, 1u> e2_energy_rotor_seed(
    E2EnergyRotorHash hash) {
  return {{{0, 0, 0, e2_energy_rotor_word(hash)}}};
}

}  // namespace substrate::bcc32
