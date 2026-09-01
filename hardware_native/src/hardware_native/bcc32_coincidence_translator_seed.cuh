#pragma once

// Coincidence-to-translated-interface germ.
//
// A compact hash grows two adjacent cells.  The basis-0 source is ligand
// capable (`P0` absent) and carries only B0; the receptor has the measured
// C0 molecular phase and the existing E1/C1 translated-interface scaffold.
// The +u1 cell is ordinary Q material.  After birth F alone decides whether
// a joint E0/R0 state restores source B0/E0 and reaches that orthogonal cell.
//
// No hash bit encodes an input, output, tick count, target value, or branch
// choice.  The three bits select resident material only.

#include <array>
#include <cstdint>

#include "bcc32_coordinate.hpp"
#include "bcc32_developmental_seed.cuh"
#include "bcc32_law.cuh"

namespace substrate::bcc32 {

using CoincidenceTranslatorSeedHash = std::uint8_t;
inline constexpr std::uint32_t kCoincidenceTranslatorPhaseShift = 0u;
inline constexpr std::uint32_t kCoincidenceTranslatorSourceBondShift = 1u;
inline constexpr std::uint32_t kCoincidenceTranslatorE1Shift = 2u;
inline constexpr std::uint32_t kCoincidenceTranslatorC1Shift = 3u;

inline const Z3Coordinate kCoincidenceTranslatorSource{-1, 0, 0};
inline const Z3Coordinate kCoincidenceTranslatorReceptor{0, 0, 0};
inline const Z3Coordinate kCoincidenceTranslatorTarget{0, 1, 0};

constexpr bool coincidence_translator_bit(CoincidenceTranslatorSeedHash hash,
                                          std::uint32_t shift) {
  return ((hash >> shift) & CoincidenceTranslatorSeedHash{1u}) != 0u;
}

constexpr SiteWord coincidence_translator_source_word(CoincidenceTranslatorSeedHash hash) {
  SiteWord word = static_cast<SiteWord>(kQ & ~carrier_bit(0u));
  if (coincidence_translator_bit(hash, kCoincidenceTranslatorSourceBondShift))
    word |= owned_bond_bit(0u);
  return word;
}

constexpr SiteWord coincidence_translator_receptor_word(CoincidenceTranslatorSeedHash hash) {
  SiteWord word = static_cast<SiteWord>(kQ & ~carrier_bit(4u));
  if (coincidence_translator_bit(hash, kCoincidenceTranslatorPhaseShift))
    word |= channel_bit(kConformationShift, 0u);
  if (coincidence_translator_bit(hash, kCoincidenceTranslatorE1Shift))
    word |= energy_bit(1u);
  if (coincidence_translator_bit(hash, kCoincidenceTranslatorC1Shift))
    word |= channel_bit(kConformationShift, 1u);
  return word;
}

constexpr std::array<DevelopmentalSeedSite, 2> coincidence_translator_seed(
    CoincidenceTranslatorSeedHash hash) {
  return {{{-1, 0, 0, coincidence_translator_source_word(hash)},
           {0, 0, 0, coincidence_translator_receptor_word(hash)}}};
}

inline constexpr CoincidenceTranslatorSeedHash kCoincidenceTranslatorHash = 0x0fu;
inline constexpr CoincidenceTranslatorSeedHash kCoincidenceTranslatorPhaseLesionHash = 0x0eu;
inline constexpr CoincidenceTranslatorSeedHash kCoincidenceTranslatorBondLesionHash = 0x0du;
inline constexpr CoincidenceTranslatorSeedHash kCoincidenceTranslatorE1LesionHash = 0x0bu;
inline constexpr CoincidenceTranslatorSeedHash kCoincidenceTranslatorC1LesionHash = 0x07u;

}  // namespace substrate::bcc32
