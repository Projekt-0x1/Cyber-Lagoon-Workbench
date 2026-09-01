#pragma once

// Developmental phase genes for repeated signed residual extraction.
//
// Three causally isolated banks become exact target-bit comparator +
// validity + signed-gate founder states at ticks 24, 48, and 72. Runtime
// exchanges every represented channel with the active 195-site footprint,
// exporting the used state and all twelve represented link words to spent
// matter. Inverse was used only at design time to derive these phase genes.

#include <array>
#include <cstddef>
#include <cstdint>

#include "bcc32_law.cuh"

namespace substrate::bcc32 {

using PredictionObservationSignedRearmPumpHash = std::uint32_t;

inline constexpr PredictionObservationSignedRearmPumpHash
    kPredictionObservationSignedRearmPumpHash = 0x7eb8ffffu;
inline constexpr std::uint32_t kPredictionObservationSignedRearmPumpTargetBit = 3u;
inline constexpr std::uint32_t kPredictionObservationSignedRearmPumpBanks = 3u;
inline constexpr std::uint32_t kPredictionObservationSignedRearmPumpEpisodeTicks = 24u;
inline constexpr std::int32_t kPredictionObservationSignedRearmPumpBankStride = 8192;

struct PredictionObservationSignedRearmPumpSeedSite {
  std::int32_t x = 0;
  std::int32_t y = 0;
  std::int32_t z = 0;
  SiteWord word = kQ;
};

struct PredictionObservationSignedRearmPumpCoordinate {
  std::int32_t x = 0;
  std::int32_t y = 0;
  std::int32_t z = 0;
};

inline constexpr std::array<PredictionObservationSignedRearmPumpSeedSite, 82>
    kPredictionObservationSignedRearmPumpPrephase24{{
        {-140, -16, 104, 0x000000feu}, {-114, 24, -24, 0x000000feu},  {-112, -24, -24, 0x000000feu},
        {-110, -24, -24, 0x000000feu}, {-109, -24, -24, 0x000000feu}, {-108, -24, -24, 0x000000feu},
        {-97, -16, 104, 0x100101ffu},  {-96, -17, 104, 0x000200ffu},  {-96, -16, 104, 0x011000ffu},
        {-78, -16, 104, 0x000000feu},  {-73, -24, -24, 0x100101ffu},  {-73, 24, -24, 0x100101ffu},
        {-72, -24, -24, 0x011010ffu},  {-72, 23, -24, 0x000200ffu},   {-72, 24, -24, 0x011000ffu},
        {-55, -16, 104, 0x000000efu},  {-54, -16, 104, 0x000000efu},  {-39, -24, -24, 0x000000efu},
        {-34, -24, -24, 0x000000efu},  {-33, -24, -24, 0x000000efu},  {-33, -16, 104, 0x100101ffu},
        {-33, 24, -24, 0x000000efu},   {-32, -17, 104, 0x000200ffu},  {-32, -16, 104, 0x011000ffu},
        {-32, 24, -24, 0x000000efu},   {-25, 54, 0, 0x000000feu},     {-23, 54, 0, 0x000000feu},
        {-22, 54, 0, 0x000000feu},     {-21, 54, 0, 0x000000feu},     {-16, -72, -24, 0x000000feu},
        {-14, -72, -24, 0x000000feu},  {-13, -72, -24, 0x000000feu},  {-12, -72, -24, 0x000000feu},
        {-12, -16, 104, 0x000000feu},  {-1, 54, 0, 0x100101ffu},      {0, 54, 0, 0x011010ffu},
        {7, 54, 0, 0x000000feu},       {9, 54, 0, 0x000000feu},       {10, 54, 0, 0x000000feu},
        {11, -16, 104, 0x000000efu},   {11, 54, 0, 0x000000feu},      {12, -16, 104, 0x000000efu},
        {18, 54, 0, 0x000000efu},      {23, -72, -24, 0x100101ffu},   {23, 54, 0, 0x000000efu},
        {24, -72, -24, 0x011010ffu},   {24, 54, 0, 0x000000efu},      {31, -16, 104, 0x100101ffu},
        {31, 54, 0, 0x100101ffu},      {32, -72, -24, 0x000000feu},   {32, -24, -24, 0x000000feu},
        {32, -17, 104, 0x000200ffu},   {32, -16, 104, 0x011000ffu},   {32, 54, 0, 0x011010ffu},
        {34, -72, -24, 0x000000feu},   {35, -72, -24, 0x000000feu},   {36, -72, -24, 0x000000feu},
        {50, -16, 104, 0x000000feu},   {50, 54, 0, 0x000000efu},      {55, 54, 0, 0x000000efu},
        {56, 54, 0, 0x000000efu},      {57, -72, -24, 0x000000efu},   {62, -72, -24, 0x000000efu},
        {63, -72, -24, 0x000000efu},   {71, -72, -24, 0x100101ffu},   {71, -24, -24, 0x100101ffu},
        {72, -72, -24, 0x011010ffu},   {72, -25, -24, 0x000200ffu},   {72, -24, -24, 0x011000ffu},
        {72, 24, -24, 0x000200ffu},    {73, -16, 104, 0x000000efu},   {74, -16, 104, 0x000000efu},
        {95, -16, 104, 0x100101ffu},   {96, -17, 104, 0x000200ffu},   {96, -16, 104, 0x011000ffu},
        {105, -72, -24, 0x000000efu},  {109, -24, -24, 0x000000efu},  {110, -72, -24, 0x000000efu},
        {110, -24, -24, 0x000000efu},  {111, -72, -24, 0x000000efu},  {139, -16, 104, 0x000000efu},
        {140, -16, 104, 0x000000efu},
    }};

inline constexpr std::array<PredictionObservationSignedRearmPumpSeedSite, 78>
    kPredictionObservationSignedRearmPumpPrephase48{{
        {-164, -16, 104, 0x000000feu}, {-138, 24, -24, 0x000000feu},  {-136, -24, -24, 0x000000feu},
        {-134, -24, -24, 0x000000feu}, {-133, -24, -24, 0x000000feu}, {-132, -24, -24, 0x000000feu},
        {-101, -16, 104, 0x000000feu}, {-97, -16, 104, 0x100101ffu},  {-96, -17, 104, 0x000200ffu},
        {-96, -16, 104, 0x011010ffu},  {-94, -16, 104, 0x000000efu},  {-73, -24, -24, 0x100101ffu},
        {-73, 24, -24, 0x100101ffu},   {-72, -24, -24, 0x011010ffu},  {-72, 23, -24, 0x000200ffu},
        {-72, 24, -24, 0x011000ffu},   {-49, 54, 0, 0x000000feu},     {-47, 54, 0, 0x000000feu},
        {-46, 54, 0, 0x000000feu},     {-45, 54, 0, 0x000000feu},     {-40, -72, -24, 0x000000feu},
        {-38, -72, -24, 0x000000feu},  {-37, -72, -24, 0x000000feu},  {-36, -72, -24, 0x000000feu},
        {-35, -16, 104, 0x000000feu},  {-33, -16, 104, 0x000101ffu},  {-32, -17, 104, 0x000200ffu},
        {-32, -16, 104, 0x000010ffu},  {-15, -24, -24, 0x000000efu},  {-13, 54, 0, 0x000000feu},
        {-10, -24, -24, 0x000000efu},  {-9, -24, -24, 0x000000efu},   {-9, 24, -24, 0x000000efu},
        {-8, 24, -24, 0x000000efu},    {-1, 54, 0, 0x100101ffu},      {0, 54, 0, 0x010010ffu},
        {8, -24, -24, 0x000000feu},    {12, -72, -24, 0x000000feu},   {12, 54, 0, 0x000000efu},
        {14, 54, 0, 0x000000efu},      {15, 54, 0, 0x000000feu},      {16, 54, 0, 0x000000feu},
        {21, 54, 0, 0x000000feu},      {23, -72, -24, 0x100101ffu},   {24, -72, -24, 0x001010ffu},
        {27, -16, 104, 0x000000feu},   {30, -16, 104, 0x000000feu},   {31, -16, 104, 0x100101ffu},
        {31, 54, 0, 0x100101ffu},      {32, -17, 104, 0x000200ffu},   {32, -16, 104, 0x001010efu},
        {32, 54, 0, 0x011010ffu},      {35, -72, -24, 0x000000efu},   {37, -72, -24, 0x000000efu},
        {56, -72, -24, 0x000000feu},   {57, -72, -24, 0x000000feu},   {62, -72, -24, 0x000000feu},
        {71, -72, -24, 0x100101ffu},   {71, -24, -24, 0x100101ffu},   {72, -72, -24, 0x011010ffu},
        {72, -25, -24, 0x000200ffu},   {72, -24, -24, 0x011000ffu},   {72, 24, -24, 0x000200ffu},
        {74, 54, 0, 0x000000efu},      {79, 54, 0, 0x000000efu},      {80, 54, 0, 0x000000efu},
        {93, -16, 104, 0x000000feu},   {94, -16, 104, 0x000000feu},   {95, -16, 104, 0x100101ffu},
        {96, -17, 104, 0x000200ffu},   {96, -16, 104, 0x011000ffu},   {129, -72, -24, 0x000000efu},
        {133, -24, -24, 0x000000efu},  {134, -72, -24, 0x000000efu},  {134, -24, -24, 0x000000efu},
        {135, -72, -24, 0x000000efu},  {163, -16, 104, 0x000000efu},  {164, -16, 104, 0x000000efu},
    }};

inline constexpr std::array<PredictionObservationSignedRearmPumpSeedSite, 80>
    kPredictionObservationSignedRearmPumpPrephase72{{
        {-188, -16, 104, 0x000000feu}, {-162, 24, -24, 0x000000feu},  {-160, -24, -24, 0x000000feu},
        {-158, -24, -24, 0x000000feu}, {-157, -24, -24, 0x000000feu}, {-156, -24, -24, 0x000000feu},
        {-125, -16, 104, 0x000000feu}, {-97, -16, 104, 0x100101ffu},  {-96, -17, 104, 0x000200ffu},
        {-96, -16, 104, 0x011010ffu},  {-73, -24, -24, 0x100101ffu},  {-73, 24, -24, 0x100101ffu},
        {-73, 54, 0, 0x000000feu},     {-72, -24, -24, 0x011010ffu},  {-72, 23, -24, 0x000200ffu},
        {-72, 24, -24, 0x011000ffu},   {-71, 54, 0, 0x000000feu},     {-70, -16, 104, 0x000000efu},
        {-70, 54, 0, 0x000000feu},     {-69, 54, 0, 0x000000feu},     {-64, -72, -24, 0x000000feu},
        {-62, -72, -24, 0x000000feu},  {-61, -72, -24, 0x000000feu},  {-60, -72, -24, 0x000000feu},
        {-59, -16, 104, 0x000000feu},  {-56, -16, 104, 0x000000feu},  {-37, 54, 0, 0x000000feu},
        {-33, -16, 104, 0x100101ffu},  {-32, -17, 104, 0x000200ffu},  {-32, -16, 104, 0x010010ffu},
        {-16, -24, -24, 0x000000feu},  {-12, -72, -24, 0x000000feu},  {-10, -16, 104, 0x000000efu},
        {-9, 54, 0, 0x000000feu},      {-8, 54, 0, 0x000000feu},      {-1, 54, 0, 0x100101ffu},
        {0, 54, 0, 0x010000ffu},       {3, -16, 104, 0x000000feu},    {6, -16, 104, 0x000000feu},
        {9, -24, -24, 0x000000efu},    {14, -24, -24, 0x000000efu},   {15, -24, -24, 0x000000efu},
        {15, 24, -24, 0x000000efu},    {16, 24, -24, 0x000000efu},    {23, -72, -24, 0x100101ffu},
        {24, -72, -24, 0x001010ffu},   {25, 54, 0, 0x000000feu},      {27, 54, 0, 0x000000feu},
        {31, -16, 104, 0x100101ffu},   {31, 54, 0, 0x100101ffu},      {32, -72, -24, 0x000000feu},
        {32, -17, 104, 0x000200ffu},   {32, -16, 104, 0x001010ffu},   {32, 54, 0, 0x011010ffu},
        {33, -72, -24, 0x000000feu},   {38, -72, -24, 0x000000feu},   {56, -16, 104, 0x000000efu},
        {59, -72, -24, 0x000000efu},   {61, -72, -24, 0x000000efu},   {69, -16, 104, 0x000000feu},
        {70, -16, 104, 0x000000feu},   {71, -72, -24, 0x100101ffu},   {71, -24, -24, 0x100101ffu},
        {72, -72, -24, 0x011010ffu},   {72, -25, -24, 0x000200ffu},   {72, -24, -24, 0x011000ffu},
        {72, 24, -24, 0x000200ffu},    {95, -16, 104, 0x100101ffu},   {96, -17, 104, 0x000200ffu},
        {96, -16, 104, 0x011000ffu},   {98, 54, 0, 0x000000efu},      {103, 54, 0, 0x000000efu},
        {104, 54, 0, 0x000000efu},     {153, -72, -24, 0x000000efu},  {157, -24, -24, 0x000000efu},
        {158, -72, -24, 0x000000efu},  {158, -24, -24, 0x000000efu},  {159, -72, -24, 0x000000efu},
        {187, -16, 104, 0x000000efu},  {188, -16, 104, 0x000000efu},
    }};

inline constexpr std::array<PredictionObservationSignedRearmPumpCoordinate, 195>
    kPredictionObservationSignedRearmPumpFootprint{{
        {-116, -16, 104}, {-99, -19, 101}, {-98, -16, 104}, {-97, -17, 103}, {-97, -16, 104},
        {-96, -18, 104},  {-96, -17, 104}, {-96, -16, 101}, {-96, -16, 103}, {-96, -16, 104},
        {-96, -16, 105},  {-96, -16, 107}, {-96, -15, 104}, {-96, -14, 104}, {-96, -13, 104},
        {-95, -16, 104},  {-95, -15, 105}, {-94, -16, 104}, {-93, -16, 104}, {-93, -13, 107},
        {-90, 24, -24},   {-88, -24, -24}, {-86, -24, -24}, {-85, -24, -24}, {-84, -24, -24},
        {-81, -24, -24},  {-80, -24, -24}, {-79, -16, 104}, {-78, -16, 104}, {-77, -24, -24},
        {-77, 19, -29},   {-76, 24, -24},  {-75, -24, -24}, {-75, 21, -27},  {-74, 22, -26},
        {-73, -24, -24},  {-73, 23, -25},  {-73, 24, -24},  {-72, -24, -24}, {-72, 20, -24},
        {-72, 23, -24},   {-72, 24, -29},  {-72, 24, -27},  {-72, 24, -25},  {-72, 24, -24},
        {-72, 24, -23},   {-72, 24, -22},  {-72, 24, -21},  {-72, 24, -19},  {-72, 25, -24},
        {-72, 27, -24},   {-72, 28, -24},  {-72, 29, -24},  {-71, 24, -24},  {-71, 25, -23},
        {-69, 24, -24},   {-69, 27, -21},  {-68, 24, -24},  {-67, 24, -24},  {-67, 29, -19},
        {-65, -24, -24},  {-64, -24, -24}, {-63, -24, -24}, {-58, -24, -24}, {-57, -24, -24},
        {-57, 24, -24},   {-56, 24, -24},  {-54, -16, 104}, {-33, -17, 103}, {-33, -16, 104},
        {-32, -17, 104},  {-32, -16, 103}, {-32, -16, 104}, {-32, -16, 105}, {-32, -15, 104},
        {-31, -16, 104},  {-31, -15, 105}, {-24, 54, 0},    {-23, 54, 0},    {-20, 54, 0},
        {-18, 54, 0},     {-13, -16, 104}, {-12, -16, 104}, {-1, 54, 0},     {0, 54, 0},
        {4, 54, 0},       {5, 54, 0},      {8, -72, -24},   {8, 54, 0},      {9, 54, 0},
        {10, -72, -24},   {11, -72, -24},  {12, -72, -24},  {12, -16, 104},  {12, 54, 0},
        {14, 54, 0},      {15, -72, -24},  {16, -72, -24},  {19, -72, -24},  {21, -72, -24},
        {22, 54, 0},      {23, -72, -24},  {23, 54, 0},     {24, -72, -24},  {29, -19, 101},
        {30, -16, 104},   {31, -72, -24},  {31, -17, 103},  {31, -16, 104},  {31, 54, 0},
        {32, -72, -24},   {32, -18, 104},  {32, -17, 104},  {32, -16, 101},  {32, -16, 103},
        {32, -16, 104},   {32, -16, 105},  {32, -16, 107},  {32, -15, 104},  {32, -14, 104},
        {32, -13, 104},   {32, 54, 0},     {33, -72, -24},  {33, -16, 104},  {33, -15, 105},
        {34, -16, 104},   {35, -16, 104},  {35, -13, 107},  {36, 54, 0},     {37, 54, 0},
        {38, -72, -24},   {39, -72, -24},  {40, 54, 0},     {49, -16, 104},  {50, -16, 104},
        {54, 54, 0},      {55, 54, 0},     {56, -72, -24},  {56, -24, -24},  {58, -72, -24},
        {59, -72, -24},   {60, -72, -24},  {63, -72, -24},  {64, -72, -24},  {65, -31, -31},
        {66, -24, -24},   {67, -72, -24},  {67, -29, -29},  {68, -28, -28},  {69, -72, -24},
        {69, -27, -27},   {71, -72, -24},  {71, -24, -24},  {72, -72, -24},  {72, -30, -24},
        {72, -25, -24},   {72, -24, -31},  {72, -24, -29},  {72, -24, -26},  {72, -24, -25},
        {72, -24, -24},   {72, -24, -20},  {72, -24, -19},  {72, -24, -17},  {72, -21, -24},
        {72, -19, -24},   {72, -18, -24},  {72, -17, -24},  {72, 24, -24},   {73, -23, -23},
        {74, -16, 104},   {75, -24, -24},  {77, -24, -24},  {77, -19, -19},  {78, -24, -24},
        {79, -72, -24},   {79, -24, -24},  {79, -17, -17},  {80, -72, -24},  {81, -72, -24},
        {85, -24, -24},   {86, -72, -24},  {86, -24, -24},  {87, -72, -24},  {95, -17, 103},
        {95, -16, 104},   {96, -17, 104},  {96, -16, 103},  {96, -16, 104},  {96, -16, 105},
        {96, -15, 104},   {97, -16, 104},  {97, -15, 105},  {115, -16, 104}, {116, -16, 104},
    }};

inline constexpr std::size_t kPredictionObservationSignedRearmPumpSeedSiteCount =
    kPredictionObservationSignedRearmPumpPrephase24.size() +
    kPredictionObservationSignedRearmPumpPrephase48.size() +
    kPredictionObservationSignedRearmPumpPrephase72.size();

template <std::size_t N>
constexpr void append_prediction_observation_signed_rearm_pump_bank(
    std::array<PredictionObservationSignedRearmPumpSeedSite,
               kPredictionObservationSignedRearmPumpSeedSiteCount>* result,
    std::size_t* cursor, const std::array<PredictionObservationSignedRearmPumpSeedSite, N>& local,
    std::uint32_t bank) {
  for (const PredictionObservationSignedRearmPumpSeedSite& site : local) {
    (*result)[(*cursor)++] = {
        site.x + static_cast<std::int32_t>(bank) * kPredictionObservationSignedRearmPumpBankStride,
        site.y,
        site.z,
        site.word,
    };
  }
}

constexpr std::array<PredictionObservationSignedRearmPumpSeedSite,
                     kPredictionObservationSignedRearmPumpSeedSiteCount>
prediction_observation_signed_rearm_pump_seed(PredictionObservationSignedRearmPumpHash hash) {
  std::array<PredictionObservationSignedRearmPumpSeedSite,
             kPredictionObservationSignedRearmPumpSeedSiteCount>
      result{};
  if (hash != kPredictionObservationSignedRearmPumpHash)
    return result;
  std::size_t cursor = 0u;
  append_prediction_observation_signed_rearm_pump_bank(
      &result, &cursor, kPredictionObservationSignedRearmPumpPrephase24, 1u);
  append_prediction_observation_signed_rearm_pump_bank(
      &result, &cursor, kPredictionObservationSignedRearmPumpPrephase48, 2u);
  append_prediction_observation_signed_rearm_pump_bank(
      &result, &cursor, kPredictionObservationSignedRearmPumpPrephase72, 3u);
  return result;
}

constexpr PredictionObservationSignedRearmPumpCoordinate
prediction_observation_signed_rearm_pump_bank_coordinate(
    PredictionObservationSignedRearmPumpCoordinate local, std::uint32_t bank) {
  return {
      local.x + static_cast<std::int32_t>(bank) * kPredictionObservationSignedRearmPumpBankStride,
      local.y,
      local.z,
  };
}

static_assert(kPredictionObservationSignedRearmPumpSeedSiteCount == 240u);
static_assert(kPredictionObservationSignedRearmPumpFootprint.size() == 195u);

}  // namespace substrate::bcc32
