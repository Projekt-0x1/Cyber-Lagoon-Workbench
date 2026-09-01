#pragma once

// Compact developmental grammar for a represented validity-rearm pump.
//
// The reversible substrate has no free reset.  This seed therefore grows
// three causally isolated low-entropy fuel banks for target bit 3.  The banks
// are exact F-prephases of the 44-cell comparator/validity suborgan and become
// ready at ticks 20, 40, and 60.  A uniform reciprocal channel swap moves a
// ready state into the live coordinates and exports the used state into the
// same bank as represented spent matter.  Runtime rearm uses forward F only;
// inverse was used at design time to derive these developmental phase genes.

#include <array>
#include <cstddef>
#include <cstdint>

#include "bcc32_law.cuh"

namespace substrate::bcc32 {

using PredictionObservationRearmPumpHash = std::uint32_t;

inline constexpr PredictionObservationRearmPumpHash
    kPredictionObservationRearmPumpHash = 0x7ea4ffffu;
inline constexpr std::uint32_t kPredictionObservationRearmPumpTargetBit = 3u;
inline constexpr std::uint32_t kPredictionObservationRearmPumpBanks = 3u;
inline constexpr std::uint32_t kPredictionObservationRearmPumpEpisodeTicks =
    20u;
inline constexpr std::int32_t kPredictionObservationRearmPumpBankStride =
    4096;
inline constexpr SiteWord kPredictionObservationRearmPumpChannelMask =
    0xffffffffu;

struct PredictionObservationRearmPumpSeedSite {
  std::int32_t x = 0;
  std::int32_t y = 0;
  std::int32_t z = 0;
  SiteWord word = kQ;
};

struct PredictionObservationRearmPumpCoordinate {
  std::int32_t x = 0;
  std::int32_t y = 0;
  std::int32_t z = 0;
};

inline constexpr std::array<PredictionObservationRearmPumpSeedSite, 57>
    kPredictionObservationRearmPumpPrephase20{{
        {-110, 24, -24, 0x000000feu},
        {-108, -24, -24, 0x000000feu},
        {-106, -24, -24, 0x000000feu},
        {-105, -24, -24, 0x000000feu},
        {-104, -24, -24, 0x000000feu},
        {-73, -24, -24, 0x100101ffu},
        {-73, 24, -24, 0x100101ffu},
        {-72, -24, -24, 0x011010ffu},
        {-72, 23, -24, 0x000200ffu},
        {-72, 24, -24, 0x011000ffu},
        {-43, -24, -24, 0x000000efu},
        {-38, -24, -24, 0x000000efu},
        {-37, -24, -24, 0x000000efu},
        {-37, 24, -24, 0x000000efu},
        {-36, 24, -24, 0x000000efu},
        {-21, 54, 0, 0x000000feu},
        {-19, 54, 0, 0x000000feu},
        {-18, 54, 0, 0x000000feu},
        {-17, 54, 0, 0x000000feu},
        {-12, -72, -24, 0x000000feu},
        {-10, -72, -24, 0x000000feu},
        {-9, -72, -24, 0x000000feu},
        {-8, -72, -24, 0x000000feu},
        {-1, 54, 0, 0x100101ffu},
        {0, 54, 0, 0x011010ffu},
        {11, 54, 0, 0x000000feu},
        {13, 54, 0, 0x000000feu},
        {14, 54, 0, 0x000000eeu},
        {15, 54, 0, 0x000000feu},
        {19, 54, 0, 0x000000efu},
        {20, 54, 0, 0x000000efu},
        {23, -72, -24, 0x100101ffu},
        {24, -72, -24, 0x011010ffu},
        {31, 54, 0, 0x100101ffu},
        {32, 54, 0, 0x011010ffu},
        {36, -72, -24, 0x000000feu},
        {36, -24, -24, 0x000000feu},
        {38, -72, -24, 0x000000feu},
        {39, -72, -24, 0x000000feu},
        {40, -72, -24, 0x000000feu},
        {46, 54, 0, 0x000000efu},
        {51, 54, 0, 0x000000efu},
        {52, 54, 0, 0x000000efu},
        {53, -72, -24, 0x000000efu},
        {58, -72, -24, 0x000000efu},
        {59, -72, -24, 0x000000efu},
        {71, -72, -24, 0x100101ffu},
        {71, -24, -24, 0x100101ffu},
        {72, -72, -24, 0x011010ffu},
        {72, -25, -24, 0x000200ffu},
        {72, -24, -24, 0x011000ffu},
        {72, 24, -24, 0x000200ffu},
        {101, -72, -24, 0x000000efu},
        {105, -24, -24, 0x000000efu},
        {106, -72, -24, 0x000000efu},
        {106, -24, -24, 0x000000efu},
        {107, -72, -24, 0x000000efu},
    }};

inline constexpr std::array<PredictionObservationRearmPumpSeedSite, 56>
    kPredictionObservationRearmPumpPrephase40{{
        {-130, 24, -24, 0x000000feu},
        {-128, -24, -24, 0x000000feu},
        {-126, -24, -24, 0x000000feu},
        {-125, -24, -24, 0x000000feu},
        {-124, -24, -24, 0x000000feu},
        {-73, -24, -24, 0x100101ffu},
        {-73, 24, -24, 0x100101ffu},
        {-72, -24, -24, 0x011010ffu},
        {-72, 23, -24, 0x000200ffu},
        {-72, 24, -24, 0x011000ffu},
        {-41, 54, 0, 0x000000feu},
        {-39, 54, 0, 0x000000feu},
        {-38, 54, 0, 0x000000feu},
        {-37, 54, 0, 0x000000feu},
        {-32, -72, -24, 0x000000feu},
        {-30, -72, -24, 0x000000feu},
        {-29, -72, -24, 0x000000feu},
        {-28, -72, -24, 0x000000feu},
        {-23, -24, -24, 0x000000efu},
        {-18, -24, -24, 0x000000efu},
        {-17, -24, -24, 0x000000efu},
        {-17, 24, -24, 0x000000efu},
        {-16, 24, -24, 0x000000efu},
        {-5, 54, 0, 0x000000feu},
        {-1, 54, 0, 0x100100ffu},
        {0, 54, 0, 0x011010ffu},
        {4, 54, 0, 0x000000efu},
        {6, 54, 0, 0x000000efu},
        {16, -24, -24, 0x000000feu},
        {20, -72, -24, 0x000000feu},
        {23, -72, -24, 0x100101ffu},
        {23, 54, 0, 0x000000feu},
        {24, -72, -24, 0x010010ffu},
        {24, 54, 0, 0x000000feu},
        {27, -72, -24, 0x000000efu},
        {29, -72, -24, 0x000000efu},
        {29, 54, 0, 0x000000feu},
        {31, 54, 0, 0x100101ffu},
        {32, 54, 0, 0x011010ffu},
        {64, -72, -24, 0x000000feu},
        {65, -72, -24, 0x000000feu},
        {66, 54, 0, 0x000000efu},
        {70, -72, -24, 0x000000feu},
        {71, -72, -24, 0x100101ffu},
        {71, -24, -24, 0x100101ffu},
        {71, 54, 0, 0x000000efu},
        {72, -72, -24, 0x011010ffu},
        {72, -25, -24, 0x000200ffu},
        {72, -24, -24, 0x011000ffu},
        {72, 24, -24, 0x000200ffu},
        {72, 54, 0, 0x000000efu},
        {121, -72, -24, 0x000000efu},
        {125, -24, -24, 0x000000efu},
        {126, -72, -24, 0x000000efu},
        {126, -24, -24, 0x000000efu},
        {127, -72, -24, 0x000000efu},
    }};

inline constexpr std::array<PredictionObservationRearmPumpSeedSite, 56>
    kPredictionObservationRearmPumpPrephase60{{
        {-150, 24, -24, 0x000000feu},
        {-148, -24, -24, 0x000000feu},
        {-146, -24, -24, 0x000000feu},
        {-145, -24, -24, 0x000000feu},
        {-144, -24, -24, 0x000000feu},
        {-73, -24, -24, 0x100101ffu},
        {-73, 24, -24, 0x100101ffu},
        {-72, -24, -24, 0x011010ffu},
        {-72, 23, -24, 0x000200ffu},
        {-72, 24, -24, 0x011000ffu},
        {-61, 54, 0, 0x000000feu},
        {-59, 54, 0, 0x000000feu},
        {-58, 54, 0, 0x000000feu},
        {-57, 54, 0, 0x000000feu},
        {-52, -72, -24, 0x000000feu},
        {-50, -72, -24, 0x000000feu},
        {-49, -72, -24, 0x000000feu},
        {-48, -72, -24, 0x000000feu},
        {-25, 54, 0, 0x000000feu},
        {-4, -24, -24, 0x000000feu},
        {-3, -24, -24, 0x000000efu},
        {-1, 54, 0, 0x100101ffu},
        {0, -72, -24, 0x000000feu},
        {0, 54, 0, 0x010010ffu},
        {2, -24, -24, 0x000000efu},
        {3, -24, -24, 0x000000efu},
        {3, 24, -24, 0x000000efu},
        {3, 54, 0, 0x000000feu},
        {4, 24, -24, 0x000000efu},
        {4, 54, 0, 0x000000feu},
        {9, 54, 0, 0x000000feu},
        {23, -72, -24, 0x100101ffu},
        {24, -72, -24, 0x001010ffu},
        {24, 54, 0, 0x000000efu},
        {26, 54, 0, 0x000000efu},
        {31, 54, 0, 0x100101ffu},
        {32, 54, 0, 0x011010ffu},
        {44, -72, -24, 0x000000feu},
        {45, -72, -24, 0x000000feu},
        {47, -72, -24, 0x000000efu},
        {49, -72, -24, 0x000000efu},
        {50, -72, -24, 0x000000feu},
        {71, -72, -24, 0x100101ffu},
        {71, -24, -24, 0x100101ffu},
        {72, -72, -24, 0x011010ffu},
        {72, -25, -24, 0x000200ffu},
        {72, -24, -24, 0x011000ffu},
        {72, 24, -24, 0x000200ffu},
        {86, 54, 0, 0x000000efu},
        {91, 54, 0, 0x000000efu},
        {92, 54, 0, 0x000000efu},
        {141, -72, -24, 0x000000efu},
        {145, -24, -24, 0x000000efu},
        {146, -72, -24, 0x000000efu},
        {146, -24, -24, 0x000000efu},
        {147, -72, -24, 0x000000efu},
    }};

inline constexpr std::array<PredictionObservationRearmPumpCoordinate, 93>
    kPredictionObservationRearmPumpFootprint{{
        {-90, 24, -24}, {-88, -24, -24}, {-86, -24, -24},
        {-85, -24, -24}, {-84, -24, -24}, {-77, -24, -24},
        {-76, -24, -24}, {-73, -24, -24}, {-73, 23, -25},
        {-73, 24, -24}, {-72, -24, -24}, {-72, 23, -24},
        {-72, 24, -25}, {-72, 24, -24}, {-72, 24, -23},
        {-72, 25, -24}, {-71, 24, -24}, {-71, 25, -23},
        {-69, -24, -24}, {-68, -24, -24}, {-63, -24, -24},
        {-58, -24, -24}, {-57, -24, -24}, {-57, 24, -24},
        {-56, 24, -24}, {-20, 54, 0}, {-19, 54, 0},
        {-16, 54, 0}, {-14, 54, 0}, {-1, 54, 0},
        {0, 54, 0}, {1, 54, 0}, {4, 54, 0},
        {8, -72, -24}, {10, -72, -24}, {11, -72, -24},
        {12, -72, -24}, {12, 54, 0}, {13, 54, 0},
        {16, 54, 0}, {18, 54, 0}, {19, -72, -24},
        {19, 54, 0}, {20, -72, -24}, {23, -72, -24},
        {24, -72, -24}, {27, -72, -24}, {28, -72, -24},
        {31, 54, 0}, {32, 54, 0}, {33, -72, -24},
        {33, 54, 0}, {36, 54, 0}, {38, -72, -24},
        {39, -72, -24}, {50, 54, 0}, {51, 54, 0},
        {56, -72, -24}, {56, -24, -24}, {58, -72, -24},
        {59, -72, -24}, {60, -72, -24}, {67, -72, -24},
        {68, -72, -24}, {69, -27, -27}, {70, -24, -24},
        {71, -72, -24}, {71, -25, -25}, {71, -24, -24},
        {72, -72, -24}, {72, -26, -24}, {72, -25, -24},
        {72, -24, -27}, {72, -24, -25}, {72, -24, -24},
        {72, -24, -23}, {72, -24, -21}, {72, -23, -24},
        {72, -22, -24}, {72, -21, -24}, {72, 24, -24},
        {73, -24, -24}, {73, -23, -23}, {74, -24, -24},
        {75, -72, -24}, {75, -24, -24}, {75, -21, -21},
        {76, -72, -24}, {81, -72, -24}, {85, -24, -24},
        {86, -72, -24}, {86, -24, -24}, {87, -72, -24},
    }};

inline constexpr std::size_t
    kPredictionObservationRearmPumpSeedSiteCount =
        kPredictionObservationRearmPumpPrephase20.size() +
        kPredictionObservationRearmPumpPrephase40.size() +
        kPredictionObservationRearmPumpPrephase60.size();

template <std::size_t N>
constexpr void append_prediction_observation_rearm_pump_bank(
    std::array<PredictionObservationRearmPumpSeedSite,
               kPredictionObservationRearmPumpSeedSiteCount>* result,
    std::size_t* cursor,
    const std::array<PredictionObservationRearmPumpSeedSite, N>& local,
    std::uint32_t bank) {
  for (const PredictionObservationRearmPumpSeedSite& site : local) {
    (*result)[(*cursor)++] = {
        site.x +
            static_cast<std::int32_t>(bank) *
                kPredictionObservationRearmPumpBankStride,
        site.y, site.z, site.word,
    };
  }
}

constexpr std::array<PredictionObservationRearmPumpSeedSite,
                     kPredictionObservationRearmPumpSeedSiteCount>
prediction_observation_rearm_pump_seed(
    PredictionObservationRearmPumpHash hash) {
  std::array<PredictionObservationRearmPumpSeedSite,
             kPredictionObservationRearmPumpSeedSiteCount>
      result{};
  if (hash != kPredictionObservationRearmPumpHash)
    return result;
  std::size_t cursor = 0u;
  append_prediction_observation_rearm_pump_bank(
      &result, &cursor, kPredictionObservationRearmPumpPrephase20, 1u);
  append_prediction_observation_rearm_pump_bank(
      &result, &cursor, kPredictionObservationRearmPumpPrephase40, 2u);
  append_prediction_observation_rearm_pump_bank(
      &result, &cursor, kPredictionObservationRearmPumpPrephase60, 3u);
  return result;
}

constexpr PredictionObservationRearmPumpCoordinate
prediction_observation_rearm_pump_bank_coordinate(
    PredictionObservationRearmPumpCoordinate local,
    std::uint32_t bank) {
  return {
      local.x +
          static_cast<std::int32_t>(bank) *
              kPredictionObservationRearmPumpBankStride,
      local.y,
      local.z,
  };
}

static_assert(kPredictionObservationRearmPumpSeedSiteCount == 169u);
static_assert(kPredictionObservationRearmPumpFootprint.size() == 93u);

}  // namespace substrate::bcc32
