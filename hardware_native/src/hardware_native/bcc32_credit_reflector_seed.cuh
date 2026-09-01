#pragma once

// A compact saturated-dimer receiver gene.
//
// The two words are a strict fixed point of full F.  A vacancy on the selected
// positive carrier lane breaks the equalities locally and scatters into the
// reversed lane; other lanes pass through.  This is transport anatomy, not a
// learned weight or a target-aware decoder.

#include <array>
#include <cstdint>

#include "bcc32_coordinate.hpp"
#include "bcc32_geometry.cuh"
#include "bcc32_law.cuh"

namespace substrate::bcc32 {

using CreditReflectorGeneHash = std::uint8_t;

constexpr CreditReflectorGeneHash make_credit_reflector_gene(std::uint32_t lane) {
  return static_cast<CreditReflectorGeneHash>(lane & 0x3u);
}

constexpr std::uint32_t credit_reflector_lane(CreditReflectorGeneHash hash) {
  return hash & 0x3u;
}

constexpr SiteWord credit_reflector_guard_word(CreditReflectorGeneHash hash) {
  const std::uint32_t lane = credit_reflector_lane(hash);
  return kQ | energy_bit(lane) | face_bit(lane) | owned_bond_bit(lane);
}

constexpr SiteWord credit_reflector_symbol_word(CreditReflectorGeneHash hash) {
  const std::uint32_t lane = credit_reflector_lane(hash);
  return kQ | channel_bit(kConformationShift, lane) |
         channel_bit(kReactiveShift, lane) | face_bit(lane + 4u);
}

inline std::array<DevelopmentalSeedSite, 2> credit_reflector_seed(
    CreditReflectorGeneHash hash, const Z3Coordinate& symbol) {
  const Int3 offset = direction_offset(static_cast<Direction>(credit_reflector_lane(hash)));
  return {{{static_cast<std::int8_t>(symbol.x - offset.x),
            static_cast<std::int8_t>(symbol.y - offset.y),
            static_cast<std::int8_t>(symbol.z - offset.z), credit_reflector_guard_word(hash)},
           {static_cast<std::int8_t>(symbol.x), static_cast<std::int8_t>(symbol.y),
            static_cast<std::int8_t>(symbol.z), credit_reflector_symbol_word(hash)}}};
}

inline constexpr CreditReflectorGeneHash kCreditReflectorLane3Gene =
    make_credit_reflector_gene(3u);

}  // namespace substrate::bcc32
