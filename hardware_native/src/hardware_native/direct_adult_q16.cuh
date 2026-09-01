#ifndef HARDWARE_NATIVE_DIRECT_ADULT_Q16_CUH
#define HARDWARE_NATIVE_DIRECT_ADULT_Q16_CUH
#include "hardware_native/direct_adult_core_constants.cuh"
namespace substrate::direct_adult_core {
// Shared Q16 arithmetic for phenotype headers. device_ops.cuh keeps identical
// definitions for its own TU graph; include-guard prevents ODR clashes if both
// appear in one translation unit after a future device_ops thin.
#ifndef HARDWARE_NATIVE_DIRECT_ADULT_Q16_OPS_DEFINED
#define HARDWARE_NATIVE_DIRECT_ADULT_Q16_OPS_DEFINED
DIRECT_ADULT_HD inline std::int32_t abs_q16(std::int32_t v) {
  return v < 0 ? -v : v;
}
DIRECT_ADULT_HD inline std::int32_t max_q16(std::int32_t a, std::int32_t b) {
  return a > b ? a : b;
}
DIRECT_ADULT_HD inline std::int32_t clamp_q16(std::int32_t value, std::int32_t min_val,
                                             std::int32_t max_val) {
  return value < min_val ? min_val : (value > max_val ? max_val : value);
}
DIRECT_ADULT_HD inline std::int32_t mul_q16(std::int32_t a, std::int32_t b) {
  return static_cast<std::int32_t>((static_cast<std::int64_t>(a) * b) >> 16);
}
#endif  // HARDWARE_NATIVE_DIRECT_ADULT_Q16_OPS_DEFINED
}  // namespace substrate::direct_adult_core
#endif  // HARDWARE_NATIVE_DIRECT_ADULT_Q16_CUH
