#pragma once

// Compact local molecular-coincidence grammar.
//
// The hash materialises a two-cell receptor complex, not an instruction
// sequence.  The distal cell can receive an E0 quantum (the ligand analogue)
// and the receptor can retain an R0 quantum (the local-voltage analogue).
// Their simultaneous availability is interpreted only by the unchanged BCC
// law F.  There is no timer, scheduler flag, Boolean callback, or host state
// transition in this grammar.
//
// The one structural bit is the receptor's C0 phase.  It is a resident
// molecular gate: remove it and the same two physical exchanges no longer
// reach the open receptor word.  The two runtime conditions themselves are
// represented matter on two independent reciprocal contact tapes.

#include <array>
#include <cstdint>

#include "bcc32_developmental_seed.cuh"
#include "bcc32_law.cuh"

namespace substrate::bcc32 {

using NmdaCoincidenceHash = std::uint8_t;

constexpr NmdaCoincidenceHash kNmdaCoincidenceGateBit = 0x01u;
constexpr NmdaCoincidenceHash kNmdaCoincidenceHash = kNmdaCoincidenceGateBit;
constexpr NmdaCoincidenceHash kNmdaCoincidencePhaseLesionHash = 0x00u;

// `source` and `receptor` are local anatomical roles only.  They encode no
// language feature, target coordinate, output value, or scheduled action.
constexpr std::array<DevelopmentalSeedSite, 2> nmda_coincidence_seed(
    NmdaCoincidenceHash hash) {
  SiteWord receptor = 0x000000efu;
  if ((hash & kNmdaCoincidenceGateBit) != 0u)
    receptor |= channel_bit(kConformationShift, 0u);
  return {{{-1, 0, 0, 0x000000feu}, {0, 0, 0, receptor}}};
}

}  // namespace substrate::bcc32
