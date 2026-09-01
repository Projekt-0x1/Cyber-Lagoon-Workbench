#pragma once

// Nearest-neighbour relay grammar for the measured C2-facing reflector.
//
// A single saturated reflector dimer is a strict full-F fixed point, while a
// consequence episode exposes a lane-3 C2 modulation on its carrier surface.
// This grammar does not add a rule or an output decoder.  It asks the smallest
// physical composition question: can a second, identical stable receptor be
// born at one of the eight BCC neighbours of the first receptor and receive
// that residual?  The three-bit hash names only the neighbour direction.  Both
// dimers and every later transition are ordinary BCC matter/F.

#include <array>
#include <cstdint>

#include "bcc32_credit_reflector_seed.cuh"
#include "bcc32_geometry.cuh"

namespace substrate::bcc32 {

using CreditDimerRelayGeneHash = std::uint8_t;

inline constexpr std::uint32_t kCreditDimerRelayDirectionMask = 0x07u;
inline constexpr CreditReflectorGeneHash kCreditDimerRelayLane =
    kCreditReflectorLane3Gene;

constexpr CreditDimerRelayGeneHash make_credit_dimer_relay_gene(std::uint32_t direction) {
  return static_cast<CreditDimerRelayGeneHash>(direction & kCreditDimerRelayDirectionMask);
}

constexpr std::uint32_t credit_dimer_relay_direction(CreditDimerRelayGeneHash hash) {
  return hash & kCreditDimerRelayDirectionMask;
}

inline Z3Coordinate credit_dimer_relay_symbol(CreditDimerRelayGeneHash hash,
                                               const Z3Coordinate& root_symbol) {
  const Int3 delta = direction_offset(static_cast<Direction>(credit_dimer_relay_direction(hash)));
  return {root_symbol.x + delta.x, root_symbol.y + delta.y, root_symbol.z + delta.z};
}

inline std::array<DevelopmentalSeedSite, 2> credit_dimer_relay_seed(
    CreditDimerRelayGeneHash hash, const Z3Coordinate& root_symbol) {
  return credit_reflector_seed(kCreditDimerRelayLane, credit_dimer_relay_symbol(hash, root_symbol));
}

}  // namespace substrate::bcc32
