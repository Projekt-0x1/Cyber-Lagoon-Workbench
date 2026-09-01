#ifndef HARDWARE_NATIVE_DIRECT_ADULT_BOUNDED_FANOUT_CUH
#define HARDWARE_NATIVE_DIRECT_ADULT_BOUNDED_FANOUT_CUH

#include <cstdint>

#include "hardware_native/direct_network_brain.cuh"

namespace substrate::direct_adult_core {

inline constexpr std::uint16_t kDirectAdultFanoutCeiling = 64u;

#if defined(__CUDACC__)
__host__ __device__
#endif
inline std::uint32_t bounded_active_route_count(
    const direct_network::DirectNode& node) {
  std::uint32_t count = node.active_route_count < node.route_capacity
                            ? node.active_route_count
                            : node.route_capacity;
  return count < kDirectAdultFanoutCeiling ? count
                                           : kDirectAdultFanoutCeiling;
}

}  // namespace substrate::direct_adult_core

#endif
