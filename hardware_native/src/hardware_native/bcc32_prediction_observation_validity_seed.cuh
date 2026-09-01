#pragma once

// Compact developmental grammar for the physical validity reader attached to
// the prediction/observation comparator.
//
// The reader does not count comparator molecules.  At the comparator's fixed
// delay-14 result phase, all sixteen rail combinations factor into three
// independent resident XOR relations:
//
//   positive_source.B0 XOR negative_source.B0
//   positive_source.face0 XOR negative_receptor.R0
//   positive_receptor.R0 XOR negative_source.face0
//
// The three already-measured native parity cells feed two already-measured
// two-basis AND cells.  Every input lane is used once.  The final physical
// outlet is face6 on the second AND receptor.
//
// These templates are developmental phase genes.  They were derived by
// applying exact F inverse to each desired contact state at design time.  The
// runtime never applies inverse to prepare or reset the organ: genesis places
// these words and ordinary forward F brings the three parity cells, first AND,
// second AND, and equal-mass remote control to their contact phases at ticks
// 15, 16, 18, and 18 respectively.

#include <array>
#include <cstddef>
#include <cstdint>

#include "bcc32_developmental_seed.cuh"
#include "bcc32_law.cuh"

namespace substrate::bcc32 {

using PredictionObservationValidityHash = std::uint32_t;

inline constexpr PredictionObservationValidityHash
    kPredictionObservationValidityHash = 0x3a2cffffu;
inline constexpr std::uint32_t kPredictionObservationValidityBits = 8u;
inline constexpr std::uint32_t kPredictionObservationValidityGatesPerBit =
    6u;
inline constexpr std::uint32_t kPredictionObservationValidityXorGates = 3u;
inline constexpr std::size_t kPredictionObservationValidityXorSites = 9u;
inline constexpr std::size_t kPredictionObservationValidityAndSites = 6u;
inline constexpr std::size_t kPredictionObservationValidityRemoteSites = 1u;
inline constexpr std::size_t kPredictionObservationValiditySitesPerBit =
    kPredictionObservationValidityXorGates *
        kPredictionObservationValidityXorSites +
    2u * kPredictionObservationValidityAndSites +
    kPredictionObservationValidityRemoteSites;
inline constexpr std::size_t kPredictionObservationValiditySiteCount =
    kPredictionObservationValidityBits *
    kPredictionObservationValiditySitesPerBit;

inline constexpr std::uint32_t kPredictionObservationValidityXor0 = 0u;
inline constexpr std::uint32_t kPredictionObservationValidityXor1 = 1u;
inline constexpr std::uint32_t kPredictionObservationValidityXor2 = 2u;
inline constexpr std::uint32_t kPredictionObservationValidityAnd1 = 3u;
inline constexpr std::uint32_t kPredictionObservationValidityAnd2 = 4u;
inline constexpr std::uint32_t kPredictionObservationValidityRemote = 5u;

// The shell of a 4x4x4 grid supplies 56 collision-free gate origins.  The
// region needs 48.  Adjacent origins are 48 sites apart while every prephase
// template stays within 18 sites of its origin.
constexpr std::int32_t prediction_observation_validity_grid_coordinate(
    std::uint32_t index) {
  return -72 + 48 * static_cast<std::int32_t>(index);
}

constexpr Int3 prediction_observation_validity_gate_origin(
    std::uint32_t bit, std::uint32_t gate) {
  const std::uint32_t wanted =
      bit * kPredictionObservationValidityGatesPerBit + gate;
  std::uint32_t cursor = 0u;
  for (std::uint32_t z = 0u; z < 4u; ++z) {
    for (std::uint32_t y = 0u; y < 4u; ++y) {
      for (std::uint32_t x = 0u; x < 4u; ++x) {
        const bool shell =
            x == 0u || x == 3u || y == 0u || y == 3u ||
            z == 0u || z == 3u;
        if (!shell)
          continue;
        if (cursor++ == wanted) {
          return {
              prediction_observation_validity_grid_coordinate(x),
              prediction_observation_validity_grid_coordinate(y),
              prediction_observation_validity_grid_coordinate(z),
          };
        }
      }
    }
  }
  return {};
}

constexpr Int3 prediction_observation_validity_gate_source(
    std::uint32_t bit, std::uint32_t gate) {
  const Int3 origin =
      prediction_observation_validity_gate_origin(bit, gate);
  return {origin.x - 1, origin.y, origin.z};
}

constexpr Int3 prediction_observation_validity_gate_receptor(
    std::uint32_t bit, std::uint32_t gate) {
  return prediction_observation_validity_gate_origin(bit, gate);
}

constexpr Int3 prediction_observation_validity_and_field(
    std::uint32_t bit, std::uint32_t gate) {
  const Int3 origin =
      prediction_observation_validity_gate_origin(bit, gate);
  return {origin.x, origin.y - 1, origin.z};
}

inline constexpr SiteWord kPredictionObservationValidityOutlet =
    face_bit(6u);

inline constexpr std::array<DevelopmentalSeedSite,
                            kPredictionObservationValidityXorSites>
    kPredictionObservationValidityXorPrephase15{{
        {-16, 0, 0, 0x000000feu},
        {-14, 0, 0, 0x000000feu},
        {-13, 0, 0, 0x000000feu},
        {-12, 0, 0, 0x000000feu},
        {-1, 0, 0, 0x100101ffu},
        {0, 0, 0, 0x011010ffu},
        {9, 0, 0, 0x000000efu},
        {14, 0, 0, 0x000000efu},
        {15, 0, 0, 0x000000efu},
    }};

inline constexpr std::array<DevelopmentalSeedSite,
                            kPredictionObservationValidityAndSites>
    kPredictionObservationValidityAndPrephase16{{
        {-16, 0, 0, 0x000000feu},
        {-1, 0, 0, 0x100101ffu},
        {0, -1, 0, 0x000200ffu},
        {0, 0, 0, 0x011000ffu},
        {13, 0, 0, 0x000000efu},
        {14, 0, 0, 0x000000efu},
    }};

inline constexpr std::array<DevelopmentalSeedSite,
                            kPredictionObservationValidityAndSites>
    kPredictionObservationValidityAndPrephase18{{
        {-18, 0, 0, 0x000000feu},
        {-1, 0, 0, 0x100101ffu},
        {0, -1, 0, 0x000200ffu},
        {0, 0, 0, 0x011000ffu},
        {15, 0, 0, 0x000000efu},
        {16, 0, 0, 0x000000efu},
    }};

inline constexpr std::array<DevelopmentalSeedSite,
                            kPredictionObservationValidityRemoteSites>
    kPredictionObservationValidityRemotePrephase18{{
        {0, 0, 0, 0x000200ffu},
    }};

template <std::size_t N>
constexpr void append_prediction_observation_validity_template(
    std::array<DevelopmentalSeedSite,
               kPredictionObservationValiditySiteCount>* result,
    std::size_t* cursor,
    const std::array<DevelopmentalSeedSite, N>& local,
    Int3 origin) {
  for (const DevelopmentalSeedSite& site : local) {
    (*result)[(*cursor)++] = {
        static_cast<std::int8_t>(
            static_cast<std::int32_t>(site.x) + origin.x),
        static_cast<std::int8_t>(
            static_cast<std::int32_t>(site.y) + origin.y),
        static_cast<std::int8_t>(
            static_cast<std::int32_t>(site.z) + origin.z),
        site.word,
    };
  }
}

constexpr std::array<DevelopmentalSeedSite,
                     kPredictionObservationValiditySiteCount>
prediction_observation_validity_seed(
    PredictionObservationValidityHash hash) {
  std::array<DevelopmentalSeedSite,
             kPredictionObservationValiditySiteCount>
      result{};
  if (hash != kPredictionObservationValidityHash)
    return result;
  std::size_t cursor = 0u;
  for (std::uint32_t bit = 0u;
       bit < kPredictionObservationValidityBits; ++bit) {
    for (std::uint32_t gate = 0u;
         gate < kPredictionObservationValidityXorGates; ++gate) {
      append_prediction_observation_validity_template(
          &result, &cursor,
          kPredictionObservationValidityXorPrephase15,
          prediction_observation_validity_gate_origin(bit, gate));
    }
    append_prediction_observation_validity_template(
        &result, &cursor,
        kPredictionObservationValidityAndPrephase16,
        prediction_observation_validity_gate_origin(
            bit, kPredictionObservationValidityAnd1));
    append_prediction_observation_validity_template(
        &result, &cursor,
        kPredictionObservationValidityAndPrephase18,
        prediction_observation_validity_gate_origin(
            bit, kPredictionObservationValidityAnd2));
    append_prediction_observation_validity_template(
        &result, &cursor,
        kPredictionObservationValidityRemotePrephase18,
        prediction_observation_validity_gate_origin(
            bit, kPredictionObservationValidityRemote));
  }
  return result;
}

static_assert(kPredictionObservationValiditySiteCount == 320u);
static_assert(
    prediction_observation_validity_gate_origin(0u, 0u).x == -72);
static_assert(
    prediction_observation_validity_gate_origin(7u, 5u).x >= -72 &&
    prediction_observation_validity_gate_origin(7u, 5u).x <= 72);
static_assert(
    prediction_observation_validity_seed(
        kPredictionObservationValidityHash)
        .size() == kPredictionObservationValiditySiteCount);

}  // namespace substrate::bcc32
