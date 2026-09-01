#pragma once

// Phase-aligned tissue for extracting signed residuals from the complete
// prediction/observation validity state.
//
// At tick 20 the composed comparator/validity world exposes two disjoint
// three-molecule conjunctions per bit:
//
//   positive: 00f0 & cfcf & 0660 == 0040  (prediction 0, observation 1)
//   negative: a2f3 & 0f0f & 0660 == 0200  (prediction 1, observation 0)
//
// The hexadecimal masks describe the exhaustive sixteen-row observer census;
// they are not read at runtime. Each conjunction is realized by two native
// two-basis AND cells. Positive and negative use different source molecules,
// including face6 and face7 from the validity receptor, so no represented
// molecule is fanned out or supplied twice.

#include <array>
#include <cstddef>
#include <cstdint>

#include "bcc32_developmental_seed.cuh"
#include "bcc32_law.cuh"
#include "bcc32_prediction_observation_validity_seed.cuh"

namespace substrate::bcc32 {

using PredictionObservationSignedHash = std::uint32_t;

inline constexpr PredictionObservationSignedHash kPredictionObservationSignedHash = 0x5d4effffu;
inline constexpr std::uint32_t kPredictionObservationSignedGatesPerBit = 4u;
inline constexpr std::uint32_t kPredictionObservationSignedPositiveAnd1 = 0u;
inline constexpr std::uint32_t kPredictionObservationSignedPositiveAnd2 = 1u;
inline constexpr std::uint32_t kPredictionObservationSignedNegativeAnd1 = 2u;
inline constexpr std::uint32_t kPredictionObservationSignedNegativeAnd2 = 3u;
inline constexpr std::size_t kPredictionObservationSignedSitesPerGate = 6u;
inline constexpr std::size_t kPredictionObservationSignedSiteCount =
    kPredictionObservationValidityBits * kPredictionObservationSignedGatesPerBit *
    kPredictionObservationSignedSitesPerGate;
inline constexpr SiteWord kPredictionObservationSignedOutlet = face_bit(6u);

constexpr Int3 prediction_observation_signed_gate_origin(std::uint32_t bit, std::uint32_t gate) {
  return {
      -96 + 64 * static_cast<std::int32_t>(gate),
      -112 + 32 * static_cast<std::int32_t>(bit),
      104,
  };
}

constexpr Int3 prediction_observation_signed_gate_source(std::uint32_t bit, std::uint32_t gate) {
  const Int3 origin = prediction_observation_signed_gate_origin(bit, gate);
  return {origin.x - 1, origin.y, origin.z};
}

constexpr Int3 prediction_observation_signed_gate_receptor(std::uint32_t bit, std::uint32_t gate) {
  return prediction_observation_signed_gate_origin(bit, gate);
}

constexpr Int3 prediction_observation_signed_and_field(std::uint32_t bit, std::uint32_t gate) {
  const Int3 origin = prediction_observation_signed_gate_origin(bit, gate);
  return {origin.x, origin.y - 1, origin.z};
}

constexpr Int3 prediction_observation_signed_positive_feature_a(std::uint32_t bit) {
  return prediction_observation_validity_gate_source(bit, kPredictionObservationValidityXor0);
}

constexpr Int3 prediction_observation_signed_positive_feature_b(std::uint32_t bit) {
  Int3 result =
      prediction_observation_validity_gate_origin(bit, kPredictionObservationValidityXor2);
  result.x += 4;
  return result;
}

constexpr Int3 prediction_observation_signed_negative_feature_a(std::uint32_t bit) {
  return prediction_observation_validity_gate_source(bit, kPredictionObservationValidityXor1);
}

constexpr Int3 prediction_observation_signed_negative_feature_b(std::uint32_t bit) {
  Int3 result =
      prediction_observation_validity_gate_origin(bit, kPredictionObservationValidityXor0);
  result.x += 4;
  return result;
}

constexpr Int3 prediction_observation_signed_validity_feature(std::uint32_t bit) {
  return prediction_observation_validity_gate_receptor(bit, kPredictionObservationValidityAnd2);
}

constexpr std::array<DevelopmentalSeedSite, kPredictionObservationSignedSitesPerGate>
prediction_observation_signed_and_template(std::uint32_t ready_tick) {
  return {{
      {static_cast<std::int8_t>(-static_cast<std::int32_t>(ready_tick)), 0, 0, 0x000000feu},
      {-1, 0, 0, 0x100101ffu},
      {0, -1, 0, 0x000200ffu},
      {0, 0, 0, 0x011000ffu},
      {static_cast<std::int8_t>(static_cast<std::int32_t>(ready_tick) - 3), 0, 0, 0x000000efu},
      {static_cast<std::int8_t>(static_cast<std::int32_t>(ready_tick) - 2), 0, 0, 0x000000efu},
  }};
}

template <std::size_t N>
constexpr void append_prediction_observation_signed_template(
    std::array<DevelopmentalSeedSite, kPredictionObservationSignedSiteCount>* result,
    std::size_t* cursor, const std::array<DevelopmentalSeedSite, N>& local, Int3 origin) {
  for (const DevelopmentalSeedSite& site : local) {
    (*result)[(*cursor)++] = {
        static_cast<std::int8_t>(static_cast<std::int32_t>(site.x) + origin.x),
        static_cast<std::int8_t>(static_cast<std::int32_t>(site.y) + origin.y),
        static_cast<std::int8_t>(static_cast<std::int32_t>(site.z) + origin.z),
        site.word,
    };
  }
}

constexpr std::array<DevelopmentalSeedSite, kPredictionObservationSignedSiteCount>
prediction_observation_signed_seed(PredictionObservationSignedHash hash) {
  std::array<DevelopmentalSeedSite, kPredictionObservationSignedSiteCount> result{};
  if (hash != kPredictionObservationSignedHash)
    return result;
  constexpr auto first = prediction_observation_signed_and_template(20u);
  constexpr auto second = prediction_observation_signed_and_template(22u);
  std::size_t cursor = 0u;
  for (std::uint32_t bit = 0u; bit < kPredictionObservationValidityBits; ++bit) {
    append_prediction_observation_signed_template(
        &result, &cursor, first,
        prediction_observation_signed_gate_origin(bit, kPredictionObservationSignedPositiveAnd1));
    append_prediction_observation_signed_template(
        &result, &cursor, second,
        prediction_observation_signed_gate_origin(bit, kPredictionObservationSignedPositiveAnd2));
    append_prediction_observation_signed_template(
        &result, &cursor, first,
        prediction_observation_signed_gate_origin(bit, kPredictionObservationSignedNegativeAnd1));
    append_prediction_observation_signed_template(
        &result, &cursor, second,
        prediction_observation_signed_gate_origin(bit, kPredictionObservationSignedNegativeAnd2));
  }
  return result;
}

static_assert(kPredictionObservationSignedSiteCount == 192u);
static_assert(prediction_observation_signed_gate_source(0u, 0u).x == -97);
static_assert(prediction_observation_signed_gate_receptor(7u, 3u).y == 112);
static_assert(prediction_observation_signed_seed(kPredictionObservationSignedHash).size() ==
              kPredictionObservationSignedSiteCount);

}  // namespace substrate::bcc32
