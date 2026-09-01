#pragma once

// Compact physical founder seed for the first BCC32 repulsion phenotype.
//
// This composes directly on top of `kEligibilityAttractionSeed`
// (bcc32_attraction_seed.cuh): the same two founders, the same child, the
// same two recruited anchors -- plus one hash-selectable third cell at
// (0,0,-2), one step beyond `kAnchor0` continuing the direction the
// reaction already grows. That coordinate and its exact word were not
// designed; they were found empirically by
// bcc32_repulsion_field_search_probe.cpp's exhaustive search of the
// existing, unmodified law (430 genuine at-a-distance suppression events
// across dozens of coordinates) and then singled out and hand-verified here
// as the cleanest case: a single primitive bit (`energy_bit(2)`), suppressing
// exactly one of the two anchors, leaving the founders' own joint eligibility
// trace and the other anchor untouched.
//
// The measured effect is competitive retraction, not prevention: with the
// suppressor present, anchor0 is recruited on schedule at tick 2 exactly
// like the baseline, then loses its recruited face by tick 5 -- ongoing
// physical activity un-recruits it. This is real, already-present
// contradiction/competition dynamics in the law, not a new mechanism
// authored here (see the handbook's retraction/turnover doctrine: unused or
// contradicted structure returns to reusable matter under ordinary local
// competition, no global usefulness score involved).
//
// Word-specificity is a load-bearing, verified claim, not an assumption:
// bare occupancy at (0,0,-2) with no energy bit does not suppress, and the
// adjacent channel bit `energy_bit(1)` at the same coordinate does not
// suppress either (see bcc32_repulsion_seed_contract.cpp's wrong-bit arm).
// This is independent of and complementary to the existing
// bcc32_coincidence_inhibition_seed.cuh mechanism (a helper-gated reciprocal
// inhibition edge); this seed instead shows at-a-distance competitive
// suppression of a phenotype the suppressor cell never physically touches.
//
// What this does and does not establish: a real, controllable, lesionable
// single-cell repulsion primitive against a real prior phenotype. It does
// NOT establish the underlying causal mechanism (why this exact bit, why
// this exact coordinate, why tick 5 and not tick 2) -- that characterization
// remains open, same as the search probe's own scope note.

#include <array>
#include <cstdint>

#include "bcc32_attraction_seed.cuh"
#include "bcc32_developmental_seed.cuh"
#include "bcc32_law.cuh"

namespace substrate::bcc32 {

using RepulsionSeedHash = std::uint8_t;

inline constexpr std::uint32_t kRepulsionSuppressorPresentShift = 0u;
inline constexpr std::uint32_t kRepulsionSuppressorCorrectBitShift = 1u;
inline constexpr RepulsionSeedHash kRepulsionSeedMask = 0x03u;

inline const Z3Coordinate kRepulsionSuppressorSite{0, 0, -2};

constexpr bool repulsion_seed_bit(RepulsionSeedHash hash, std::uint32_t shift) {
  return ((hash >> shift) & RepulsionSeedHash{1u}) != 0u;
}

// hash bit0 clear -> suppressor site written as plain kQ, physically
// identical to the site being absent (both baseline and bare-occupancy
// controls collapse to this one arm; see the contract's genesis()).
// hash bit0 set, bit1 set   -> the verified suppressing word (energy_bit(2)).
// hash bit0 set, bit1 clear -> adjacent-channel control word (energy_bit(1)),
// confirmed non-suppressing -- proves specificity, not mere occupancy.
constexpr SiteWord repulsion_suppressor_word(RepulsionSeedHash hash) {
  if (!repulsion_seed_bit(hash, kRepulsionSuppressorPresentShift)) return kQ;
  return repulsion_seed_bit(hash, kRepulsionSuppressorCorrectBitShift)
             ? static_cast<SiteWord>(kQ | energy_bit(2u))
             : static_cast<SiteWord>(kQ | energy_bit(1u));
}

inline constexpr std::size_t kRepulsionSeedSiteCount =
    kEligibilityAttractionSeed.size() + 1u;

constexpr std::array<DevelopmentalSeedSite, kRepulsionSeedSiteCount>
repulsion_seed(RepulsionSeedHash hash) {
  return std::array<DevelopmentalSeedSite, kRepulsionSeedSiteCount>{
      kEligibilityAttractionSeed[0], kEligibilityAttractionSeed[1],
      DevelopmentalSeedSite{0, 0, -2, repulsion_suppressor_word(hash)}};
}

inline constexpr RepulsionSeedHash kRepulsionHashAbsent = 0x00u;
inline constexpr RepulsionSeedHash kRepulsionHashPresentCorrect = 0x03u;
inline constexpr RepulsionSeedHash kRepulsionHashPresentWrongBit = 0x01u;

static_assert(repulsion_suppressor_word(kRepulsionHashAbsent) == kQ);
static_assert(repulsion_suppressor_word(kRepulsionHashPresentCorrect) ==
              (kQ | energy_bit(2u)));
static_assert(repulsion_suppressor_word(kRepulsionHashPresentWrongBit) ==
              (kQ | energy_bit(1u)));

}  // namespace substrate::bcc32
