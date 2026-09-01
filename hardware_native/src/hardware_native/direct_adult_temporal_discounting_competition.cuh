#ifndef HARDWARE_NATIVE_DIRECT_ADULT_TEMPORAL_DISCOUNTING_COMPETITION_CUH
#define HARDWARE_NATIVE_DIRECT_ADULT_TEMPORAL_DISCOUNTING_COMPETITION_CUH

#include <cstdint>

#include "direct_exact_history.cuh"
#include "direct_adult_core_constants.cuh"
#include "direct_exact_history.cuh"

namespace substrate::direct_network {

inline constexpr std::uint32_t kTemporalDiscountOneQ16 = 1u << 16;
inline constexpr std::uint32_t kTemporalDiscountMaxHorizon = 1u << 20;
inline constexpr std::uint32_t kTemporalDiscountRateQ16 = 1u << 14;
inline constexpr std::uint32_t kTemporalHistoryScanLimit = 256u;

DIRECT_ADULT_HD inline std::int32_t temporal_discount_value_q16(
    std::int32_t value_q16, std::uint32_t horizon_ticks,
    std::uint32_t discount_rate_q16 = kTemporalDiscountRateQ16) {
  if (horizon_ticks > kTemporalDiscountMaxHorizon || discount_rate_q16 == 0u ||
      discount_rate_q16 > kTemporalDiscountOneQ16)
    return 0;
  const std::uint64_t denominator_q16 =
      static_cast<std::uint64_t>(kTemporalDiscountOneQ16) +
      static_cast<std::uint64_t>(horizon_ticks) * discount_rate_q16;
  return static_cast<std::int32_t>(
      (static_cast<std::int64_t>(value_q16) * kTemporalDiscountOneQ16) /
      static_cast<std::int64_t>(denominator_q16));
}

// Read only a recent bounded exact-history suffix. A verified world return can
// exist only after exact action settlement, so prediction, current activation,
// and host labels cannot enter this value.
__device__ inline std::int32_t delayed_temporal_motor_value_q16(
    const DirectExactHistoryRecord* records, std::uint32_t record_count,
    std::uint32_t motor_node) {
  if (records == nullptr) return 0;
  std::uint32_t scanned = 0u;
  for (std::uint32_t reverse = record_count;
       reverse != 0u && scanned < kTemporalHistoryScanLimit;
       --reverse, ++scanned) {
    const auto& evidence = records[reverse - 1u];
    if (evidence.kind != DirectExactHistoryKind::world_return ||
        evidence.source != motor_node ||
        (evidence.flags & kDirectHistoryVerifiedObservation) == 0u ||
        evidence.resident_tick < evidence.event_tick)
      continue;
    const std::int64_t bounded =
        evidence.resource_delta > kTemporalDiscountOneQ16
            ? kTemporalDiscountOneQ16
            : evidence.resource_delta <
                      -static_cast<std::int64_t>(kTemporalDiscountOneQ16)
                  ? -static_cast<std::int64_t>(kTemporalDiscountOneQ16)
                  : evidence.resource_delta;
    return temporal_discount_value_q16(
        static_cast<std::int32_t>(bounded),
        evidence.resident_tick - evidence.event_tick);
  }
  return 0;
}

}  // namespace substrate::direct_network

#endif
