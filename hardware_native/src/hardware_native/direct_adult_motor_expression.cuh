#ifndef HARDWARE_NATIVE_DIRECT_ADULT_MOTOR_EXPRESSION_CUH
#define HARDWARE_NATIVE_DIRECT_ADULT_MOTOR_EXPRESSION_CUH

#include "direct_adult_core.cuh"
#include "direct_adult_runtime_frontiers.cuh"

#include "direct_adult_bounded_fanout.cuh"
#include "direct_adult_lived_expression.cuh"

namespace substrate::direct_adult_core {

DIRECT_ADULT_HD inline std::int32_t motor_expression_clamp(
    std::int32_t value, std::int32_t lower, std::int32_t upper) {
  return value < lower ? lower : value > upper ? upper : value;
}

#ifndef HARDWARE_NATIVE_BOUNDED_ROUTE_SCAN_COUNT_DEFINED
#define HARDWARE_NATIVE_BOUNDED_ROUTE_SCAN_COUNT_DEFINED
DIRECT_ADULT_HD inline std::uint32_t bounded_route_scan_count(
    const DirectNode& node) {
  return node.route_capacity < kDirectAdultFanoutCeiling
             ? node.route_capacity
             : kDirectAdultFanoutCeiling;
}
#endif

__device__ inline Word resident_motor_word(
    const DirectBrain& brain, const DirectNode& node, const DirectRoute* routes,
    std::uint32_t route_capacity, std::uint32_t route_scan_count,
    const ResidentActualFrontier* actual_frontier = nullptr,
    const DirectActionParticipationLink* action_links = nullptr,
    std::uint32_t participant_offset = kInvalidIndex,
    std::uint32_t participant_count = 0u) {
  Word expression = 0u;
  if (resident_lived_expression(
          brain, actual_frontier, action_links, participant_offset,
          participant_count, &expression))
    return expression;

  std::int64_t sum = 0;
  std::int32_t minimum = 0x7fffffff;
  std::int32_t maximum = -0x7fffffff - 1;
  std::uint32_t count = 0u;
  const std::uint32_t end = node.route_offset + route_scan_count;
  if (routes != nullptr)
    for (std::uint32_t r = node.route_offset;
         r < end && r < route_capacity; ++r) {
      if ((routes[r].flags & direct_network::kRouteFlagActive) == 0u) continue;
      const std::int32_t conductance = routes[r].conductance_q16;
      sum += conductance;
      minimum = conductance < minimum ? conductance : minimum;
      maximum = conductance > maximum ? conductance : maximum;
      ++count;
    }
  const std::int32_t mean = count == 0u ? 0 :
      static_cast<std::int32_t>(sum / static_cast<std::int64_t>(count));
  const std::int32_t spread = count == 0u ? 0 : maximum - minimum;
  const std::uint32_t spread_octet = static_cast<std::uint32_t>(
      motor_expression_clamp(spread >> 8, 0, 255));
  const std::uint32_t mean_octet = static_cast<std::uint32_t>(
      motor_expression_clamp(mean >> 8, 0, 255));
  // Causal credit authorizes revision; it is not a public byte or answer
  // channel. The neutral final octet keeps this fallback morphology-only.
  return (spread_octet << 16) | (mean_octet << 8) | 128u;
}

// Compatibility surface for contracts that exercise only the morphology
// fallback and deliberately provide no resident occurrence ancestry.
__device__ inline Word resident_motor_word(
    const DirectNode& node, const DirectRoute* routes,
    std::uint32_t route_capacity, std::uint32_t route_scan_count) {
  return resident_motor_word(DirectBrain{}, node, routes, route_capacity,
                             route_scan_count);
}

}  // namespace substrate::direct_adult_core

#endif
