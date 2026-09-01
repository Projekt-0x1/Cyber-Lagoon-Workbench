#pragma once

// Compact reciprocal-inhibition germ.
//
// The seed does not encode an update schedule or a Boolean oscillator.  Its
// three resident sites are the smallest closure of the measured C/R inhibitory
// helper under the 0<->1 BCC basis symmetry:
//
//               B (C1,R1)
//                    +u1
//                     |
//    A (C0,R0) -- +u0 H(B0,B1,face0,face1)
//
// H's face1 arm reads A and suppresses B; its face0 arm is the basis-swapped
// reciprocal arm, reading B and suppressing A.  A full-F transition is the
// only timing mechanism.  The hash chooses only initial local matter; F is
// the sole interpreter after birth.  Whether this closed germ has an
// autonomous anti-phase recurrence is deliberately a contract question, not
// an assertion made by this header.

#include <array>
#include <cstdint>

#include "bcc32_coordinate.hpp"
#include "bcc32_developmental_seed.cuh"
#include "bcc32_law.cuh"

namespace substrate::bcc32 {

using HalfCentreOscillatorSeedHash = std::uint16_t;

inline constexpr std::uint32_t kHalfCentreInitialAShift = 0u;
inline constexpr std::uint32_t kHalfCentreInitialBShift = 1u;
inline constexpr std::uint32_t kHalfCentreFace0Shift = 2u;
inline constexpr std::uint32_t kHalfCentreFace1Shift = 3u;
inline constexpr HalfCentreOscillatorSeedHash kHalfCentreSeedMask = 0x0fu;

inline const Z3Coordinate kHalfCentreHelper{0, 0, 0};
inline const Z3Coordinate kHalfCentreA{1, 0, 0};
inline const Z3Coordinate kHalfCentreB{0, 1, 0};
inline constexpr std::size_t kHalfCentreSeedSiteCount = 3u;

constexpr bool halfcentre_bit(HalfCentreOscillatorSeedHash hash, std::uint32_t shift) {
  return ((hash >> shift) & HalfCentreOscillatorSeedHash{1u}) != 0u;
}

constexpr SiteWord halfcentre_a_word(HalfCentreOscillatorSeedHash hash) {
  return static_cast<SiteWord>(kQ | (halfcentre_bit(hash, kHalfCentreInitialAShift)
                                         ? channel_bit(kConformationShift, 0u) |
                                               channel_bit(kReactiveShift, 0u)
                                         : 0u));
}

constexpr SiteWord halfcentre_b_word(HalfCentreOscillatorSeedHash hash) {
  return static_cast<SiteWord>(kQ | (halfcentre_bit(hash, kHalfCentreInitialBShift)
                                         ? channel_bit(kConformationShift, 1u) |
                                               channel_bit(kReactiveShift, 1u)
                                         : 0u));
}

constexpr SiteWord halfcentre_helper_word(HalfCentreOscillatorSeedHash hash) {
  SiteWord word = static_cast<SiteWord>(kQ | owned_bond_bit(0u) | owned_bond_bit(1u));
  if (halfcentre_bit(hash, kHalfCentreFace0Shift)) word |= face_bit(0u);
  if (halfcentre_bit(hash, kHalfCentreFace1Shift)) word |= face_bit(1u);
  return word;
}

constexpr std::array<DevelopmentalSeedSite, kHalfCentreSeedSiteCount> halfcentre_seed(
    HalfCentreOscillatorSeedHash hash) {
  return {{{0, 0, 0, halfcentre_helper_word(hash)},
           {1, 0, 0, halfcentre_a_word(hash)},
           {0, 1, 0, halfcentre_b_word(hash)}}};
}

// Both arms are present.  The two lower bits choose the initial state only;
// no later phase, timer, source, or schedule is serialized in the hash.
inline constexpr HalfCentreOscillatorSeedHash kHalfCentreClosedHash = 0x0cu;
inline constexpr HalfCentreOscillatorSeedHash kHalfCentreASeededHash = 0x0du;
inline constexpr HalfCentreOscillatorSeedHash kHalfCentreBSeededHash = 0x0eu;
inline constexpr HalfCentreOscillatorSeedHash kHalfCentreBothSeededHash = 0x0fu;
inline constexpr HalfCentreOscillatorSeedHash kHalfCentreNoFace0Hash = 0x08u;
inline constexpr HalfCentreOscillatorSeedHash kHalfCentreNoFace1Hash = 0x04u;
inline constexpr HalfCentreOscillatorSeedHash kHalfCentreNoSourceHash = 0x00u;

static_assert(halfcentre_helper_word(kHalfCentreClosedHash) ==
              (kQ | owned_bond_bit(0u) | owned_bond_bit(1u) | face_bit(0u) |
               face_bit(1u)));
static_assert(halfcentre_a_word(kHalfCentreASeededHash) ==
              (kQ | channel_bit(kConformationShift, 0u) | channel_bit(kReactiveShift, 0u)));
static_assert(halfcentre_b_word(kHalfCentreBSeededHash) ==
              (kQ | channel_bit(kConformationShift, 1u) | channel_bit(kReactiveShift, 1u)));

}  // namespace substrate::bcc32
