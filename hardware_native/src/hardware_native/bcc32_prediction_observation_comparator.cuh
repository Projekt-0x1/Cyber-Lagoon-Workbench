#pragma once

// Compact developmental grammar for an eight-column prediction/observation
// opponent microarea.
//
// Each raw bit grows two mirrored molecular-coincidence legs.  A fixed
// reciprocal boundary maps prediction-zero and observation-one matter onto the
// positive leg; prediction-one and observation-zero matter onto the negative
// leg.  The seed contains no predicted value, observed value, comparison
// result, delay, or host instruction.  Its sixteen low bits select the
// resident C0 phase of the sixteen legs; the remaining bits identify this
// growth family.

#include <array>
#include <cstddef>
#include <cstdint>

#include "bcc32_developmental_seed.cuh"
#include "bcc32_nmda_coincidence_seed.cuh"

namespace substrate::bcc32 {

using PredictionObservationComparatorHash = std::uint32_t;

inline constexpr std::uint32_t kPredictionObservationBits = 8u;
inline constexpr std::uint32_t kPredictionObservationLegsPerBit = 2u;
inline constexpr std::uint32_t kPredictionObservationSitesPerLeg = 2u;
inline constexpr std::uint32_t kPredictionObservationBitStride = 18u;
inline constexpr std::uint32_t kPredictionObservationOpponentStride = 32u;
inline constexpr std::uint32_t kPredictionObservationPhaseBits = 16u;
inline constexpr PredictionObservationComparatorHash
    kPredictionObservationFamily = 0x6d1c0000u;
inline constexpr PredictionObservationComparatorHash
    kPredictionObservationPhaseMask = 0x0000ffffu;
inline constexpr std::size_t kPredictionObservationComparatorSiteCount =
    kPredictionObservationBits * kPredictionObservationLegsPerBit *
    kPredictionObservationSitesPerLeg;

inline constexpr PredictionObservationComparatorHash
    kPredictionObservationComparatorHash =
        kPredictionObservationFamily | kPredictionObservationPhaseMask;

constexpr std::uint32_t prediction_observation_phase_index(
    std::uint32_t bit, bool positive) {
  return bit * kPredictionObservationLegsPerBit +
         (positive ? 0u : 1u);
}

constexpr bool prediction_observation_phase_present(
    PredictionObservationComparatorHash hash, std::uint32_t bit,
    bool positive) {
  return ((hash >> prediction_observation_phase_index(bit, positive)) &
          1u) != 0u;
}

constexpr Int3 prediction_observation_leg_shift(std::uint32_t bit,
                                                bool positive) {
  return {
      static_cast<std::int32_t>(
          positive ? 0u : kPredictionObservationOpponentStride),
      static_cast<std::int32_t>(bit *
                                kPredictionObservationBitStride),
      0,
  };
}

constexpr Int3 prediction_observation_source(std::uint32_t bit,
                                             bool positive) {
  const Int3 shift = prediction_observation_leg_shift(bit, positive);
  return {shift.x - 1, shift.y, shift.z};
}

constexpr Int3 prediction_observation_receptor(std::uint32_t bit,
                                               bool positive) {
  return prediction_observation_leg_shift(bit, positive);
}

constexpr PredictionObservationComparatorHash
prediction_observation_phase_lesion_hash(
    PredictionObservationComparatorHash hash, std::uint32_t bit,
    bool positive) {
  return hash &
         ~(PredictionObservationComparatorHash{1u}
           << prediction_observation_phase_index(bit, positive));
}

constexpr std::array<DevelopmentalSeedSite,
                     kPredictionObservationComparatorSiteCount>
prediction_observation_comparator_seed(
    PredictionObservationComparatorHash hash) {
  std::array<DevelopmentalSeedSite,
             kPredictionObservationComparatorSiteCount>
      result{};
  std::size_t cursor = 0u;
  for (std::uint32_t bit = 0u; bit < kPredictionObservationBits; ++bit) {
    for (std::uint32_t leg = 0u; leg < 2u; ++leg) {
      const bool positive = leg == 0u;
      const NmdaCoincidenceHash molecular_hash =
          prediction_observation_phase_present(hash, bit, positive)
              ? kNmdaCoincidenceHash
              : kNmdaCoincidencePhaseLesionHash;
      const Int3 shift = prediction_observation_leg_shift(bit, positive);
      const auto molecular = nmda_coincidence_seed(molecular_hash);
      for (const DevelopmentalSeedSite& site : molecular) {
        result[cursor++] = {
            static_cast<std::int8_t>(site.x + shift.x),
            static_cast<std::int8_t>(site.y + shift.y),
            static_cast<std::int8_t>(site.z + shift.z),
            site.word,
        };
      }
    }
  }
  return result;
}

// After a prediction contact, one through thirty ordinary F steps, an
// observation contact, and one more F, this is the physical word on exactly
// the leg named by a valid directional mismatch.
inline constexpr SiteWord kPredictionObservationDelayedOutletWord =
    0x010000ffu;

static_assert(kPredictionObservationPhaseBits ==
              kPredictionObservationBits *
                  kPredictionObservationLegsPerBit);
static_assert(
    prediction_observation_comparator_seed(
        kPredictionObservationComparatorHash)
        .size() == kPredictionObservationComparatorSiteCount);

}  // namespace substrate::bcc32
