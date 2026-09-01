#pragma once

// Compact phase-aligned credit-to-register germ.
//
// A C2-credit receiver has one measured lane-3 carrier difference at its
// exposed port.  The resident register's only write primitive is the native
// energy pair rotor: a positive-carrier tetrad must have popcount two.  The
// smallest lawful bridge is therefore not a new opcode: it is one local
// energy-bearing word which begins with three positive carriers.  A credit
// carrier absence may make that tetrad two and rotate its E state under F.
//
// The hash names only physical placement, the initially missing carrier, and
// the energy basis.  It contains no tick count, expected state, answer, or
// host update.  A test must still prove that a C2 credit episode, rather than
// quiet history or a C2 lesion, writes and preserves the state.

#include <array>
#include <cstdint>

#include "bcc32_coordinate.hpp"
#include "bcc32_geometry.cuh"
#include "bcc32_law.cuh"

namespace substrate::bcc32 {

using CreditPairRotorSeedHash = std::uint8_t;

inline constexpr std::uint32_t kCreditPairRotorPortShift = 0u;
inline constexpr std::uint32_t kCreditPairRotorHoleShift = 3u;
inline constexpr std::uint32_t kCreditPairRotorEnergyShift = 5u;

constexpr std::uint32_t credit_pairrotor_port(CreditPairRotorSeedHash hash) {
  return (hash >> kCreditPairRotorPortShift) & 0x7u;
}

constexpr std::uint32_t credit_pairrotor_hole(CreditPairRotorSeedHash hash) {
  return (hash >> kCreditPairRotorHoleShift) & 0x3u;
}

constexpr std::uint32_t credit_pairrotor_energy(CreditPairRotorSeedHash hash) {
  return (hash >> kCreditPairRotorEnergyShift) & 0x3u;
}

constexpr CreditPairRotorSeedHash make_credit_pairrotor_seed_hash(
    std::uint32_t port, std::uint32_t hole, std::uint32_t energy) {
  return static_cast<CreditPairRotorSeedHash>((port & 0x7u) << kCreditPairRotorPortShift |
                                              (hole & 0x3u) << kCreditPairRotorHoleShift |
                                              (energy & 0x3u) << kCreditPairRotorEnergyShift);
}

constexpr SiteWord credit_pairrotor_word(CreditPairRotorSeedHash hash) {
  return (kQ & ~carrier_bit(credit_pairrotor_hole(hash))) |
         energy_bit(credit_pairrotor_energy(hash));
}

inline Z3Coordinate credit_pairrotor_coordinate(CreditPairRotorSeedHash hash,
                                                const Z3Coordinate& port) {
  const Int3 direction = direction_offset(static_cast<Direction>(credit_pairrotor_port(hash)));
  return {port.x + direction.x, port.y + direction.y, port.z + direction.z};
}

}  // namespace substrate::bcc32
