#pragma once

// Coincidence-docked inhibitory-edge germ.
//
// This is a single three-cell material seed, not a union of two test rigs:
//   helper (0,0,0) --basis 0--> gate/A (1,0,0)
//          `--basis 1--> opponent/B (0,1,0)
//
// The helper->gate edge is deliberately shared by the two preceding local
// laws.  Its source is the ligand-capable `P0` absence used by the molecular
// gate; its destination is the `C0/R0` limb of the reciprocal-inhibition
// closure.  The two faces remain ordinary BCC matter.  This is an interface
// candidate, not a claim that the opened molecular state already drives the
// sibling limb; the receipt makes that distinction explicit.  The hash has no input,
// output, tick count, target coordinate, or scheduler: it only selects the
// resident phase and the two physical helper faces.  External contact is the
// reversible boundary through which a world may supply an E0 ligand quantum
// and an R0 voltage quantum; it is not a host callback.

#include <array>
#include <cstdint>

#include "bcc32_coordinate.hpp"
#include "bcc32_developmental_seed.cuh"
#include "bcc32_law.cuh"

namespace substrate::bcc32 {

using CoincidenceInhibitionSeedHash = std::uint8_t;

inline constexpr std::uint32_t kCoincidenceInhibitionGateShift = 0u;
inline constexpr std::uint32_t kCoincidenceInhibitionFace0Shift = 1u;
inline constexpr std::uint32_t kCoincidenceInhibitionFace1Shift = 2u;
inline constexpr CoincidenceInhibitionSeedHash kCoincidenceInhibitionSeedMask = 0x07u;

inline const Z3Coordinate kCoincidenceInhibitionHelper{0, 0, 0};
inline const Z3Coordinate kCoincidenceInhibitionGate{1, 0, 0};
inline const Z3Coordinate kCoincidenceInhibitionOpponent{0, 1, 0};
inline constexpr std::size_t kCoincidenceInhibitionSeedSiteCount = 3u;

constexpr bool coincidence_inhibition_bit(CoincidenceInhibitionSeedHash hash,
                                          std::uint32_t shift) {
  return ((hash >> shift) & CoincidenceInhibitionSeedHash{1u}) != 0u;
}

constexpr SiteWord coincidence_inhibition_helper_word(CoincidenceInhibitionSeedHash hash) {
  // The missing P0 is the local ligand-accepting source state (`0xfe` in the
  // isolated molecular gate).  B0/B1 and their faces are retained so this
  // same cell is also the physical reciprocal-inhibition helper.
  SiteWord word = static_cast<SiteWord>((kQ & ~carrier_bit(0u)) | owned_bond_bit(0u) |
                                        owned_bond_bit(1u));
  if (coincidence_inhibition_bit(hash, kCoincidenceInhibitionFace0Shift))
    word |= face_bit(0u);
  if (coincidence_inhibition_bit(hash, kCoincidenceInhibitionFace1Shift))
    word |= face_bit(1u);
  return word;
}

constexpr SiteWord coincidence_inhibition_gate_word(CoincidenceInhibitionSeedHash hash) {
  // The missing P-0 is the receptor's physical blocked state (`0xef`), while
  // C0 is its resident molecular phase.  R0 arrives only as a reversible
  // voltage contact; it is never serialized as a hidden tick or answer.
  SiteWord word = static_cast<SiteWord>((kQ & ~carrier_bit(4u)));
  if (coincidence_inhibition_bit(hash, kCoincidenceInhibitionGateShift))
    word |= channel_bit(kConformationShift, 0u);
  return word;
}

constexpr SiteWord coincidence_inhibition_opponent_word() { return kQ; }

constexpr std::array<DevelopmentalSeedSite, kCoincidenceInhibitionSeedSiteCount>
coincidence_inhibition_seed(CoincidenceInhibitionSeedHash hash) {
  return {{{0, 0, 0, coincidence_inhibition_helper_word(hash)},
           {1, 0, 0, coincidence_inhibition_gate_word(hash)},
           {0, 1, 0, coincidence_inhibition_opponent_word()}}};
}

inline constexpr CoincidenceInhibitionSeedHash kCoincidenceInhibitionHash = 0x07u;
inline constexpr CoincidenceInhibitionSeedHash kCoincidenceInhibitionPhaseLesionHash = 0x06u;
inline constexpr CoincidenceInhibitionSeedHash kCoincidenceInhibitionFace0LesionHash = 0x05u;
inline constexpr CoincidenceInhibitionSeedHash kCoincidenceInhibitionFace1LesionHash = 0x03u;

static_assert(coincidence_inhibition_helper_word(kCoincidenceInhibitionHash) ==
              ((kQ & ~carrier_bit(0u)) | owned_bond_bit(0u) | owned_bond_bit(1u) |
               face_bit(0u) | face_bit(1u)));
static_assert(coincidence_inhibition_gate_word(kCoincidenceInhibitionHash) ==
              ((kQ & ~carrier_bit(4u)) | channel_bit(kConformationShift, 0u)));

}  // namespace substrate::bcc32
