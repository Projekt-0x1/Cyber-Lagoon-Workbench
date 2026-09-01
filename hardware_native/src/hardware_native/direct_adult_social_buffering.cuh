#ifndef HARDWARE_NATIVE_DIRECT_ADULT_SOCIAL_BUFFERING_CUH
#define HARDWARE_NATIVE_DIRECT_ADULT_SOCIAL_BUFFERING_CUH

// f.social_buffering (#1618). A reciprocity-earned trusted partner's presence
// blunts the physiological threat rise on an affect lane and speeds the
// recovery back toward baseline. The gate reads only earned standing --
// unknown or untrusted presence grants nothing. Buffering modulates the
// affect-body evidence masses; it never fabricates vitality.

#include <cstdint>

#include "hardware_native/direct_adult_affect_body.cuh"
#include "hardware_native/direct_adult_partner_reciprocity.cuh"
#include "hardware_native/direct_adult_q16.cuh"

namespace substrate::direct_network {

// Trusted presence halves both the threat rise gain and the recovery step's
// complement: the same episode hurts less and wears off faster.
inline constexpr std::int32_t kBufferingRiseAttenuationQ16 =
    direct_adult_core::kQ16One / 2;
inline constexpr std::uint32_t kRecoveryStepQ16 = direct_adult_core::kQ16One / 8;

__host__ __device__ inline std::int32_t buffered_threat_rise_q16(
    std::int32_t current_mass_q16, bool trusted_presence) {
  const std::int32_t headroom = direct_adult_core::kQ16One - current_mass_q16;
  if (headroom <= 0) return current_mass_q16;
  std::int32_t gain = kAffectRiseGainQ16;
  if (trusted_presence)
    gain = direct_adult_core::mul_q16(gain, kBufferingRiseAttenuationQ16);
  return direct_adult_core::clamp_q16(
      current_mass_q16 + direct_adult_core::mul_q16(headroom, gain), 0,
      direct_adult_core::kQ16One);
}

__host__ __device__ inline std::int32_t buffered_recovery_step_q16(
    bool trusted_presence) {
  return trusted_presence ? static_cast<std::int32_t>(kRecoveryStepQ16) * 2
                          : static_cast<std::int32_t>(kRecoveryStepQ16);
}

__host__ __device__ inline std::int32_t buffered_recovery_decay_q16(
    std::int32_t current_mass_q16, bool trusted_presence) {
  const std::int32_t step = buffered_recovery_step_q16(trusted_presence);
  return current_mass_q16 > step ? current_mass_q16 - step : 0;
}

}  // namespace substrate::direct_network

#endif
