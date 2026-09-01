#ifndef HARDWARE_NATIVE_DIRECT_ADULT_MULTIMODAL_GROUNDING_ABI_CUH
#define HARDWARE_NATIVE_DIRECT_ADULT_MULTIMODAL_GROUNDING_ABI_CUH

#include <cstdint>
#include <type_traits>

#include "hardware_native/direct_adult_core_constants.cuh"

namespace substrate::direct_network {

inline constexpr std::uint32_t kGroundingPairCapacity = 256u;
inline constexpr std::uint32_t kGroundingObjectCapacity = 32u;
inline constexpr std::uint32_t kGroundingLegCapacity = 4u;
inline constexpr std::uint32_t kGroundingExposureCapacity = 16u;
inline constexpr std::uint32_t kGroundingRecentCapacity = 64u;
inline constexpr std::uint32_t kGroundingCoherenceWindowTicks = 16u;
inline constexpr std::uint32_t kGroundingConsequenceWindowTicks = 24u;
inline constexpr std::int32_t kGroundingRiseGainQ16 =
    direct_adult_core::kQ16One / 2;
inline constexpr std::int32_t kGroundingBoundThresholdQ16 =
    (direct_adult_core::kQ16One * 3) / 4;

struct DirectCrossModalContact {
  std::uint32_t channel;
  std::uint32_t value;
  std::uint32_t resident_tick;
};

struct DirectCrossModalPair {
  std::uint32_t channel_a;
  std::uint32_t value_a;
  std::uint32_t channel_b;
  std::uint32_t value_b;
  std::uint32_t cooccurrences;
  std::int32_t bind_mass_q16;
  std::uint32_t first_tick;
  std::uint32_t last_tick;
};

struct DirectGroundedObject {
  DirectCrossModalContact legs[kGroundingLegCapacity];
  std::uint32_t leg_count;
  std::int32_t closure_mass_q16;
  std::uint32_t consequence_samples;
  std::uint32_t first_tick;
  std::uint32_t last_tick;
};

struct DirectChannelExposure {
  std::uint32_t channel;
  std::uint32_t contacts;
};

struct DirectMultimodalGroundingTable {
  DirectCrossModalPair pairs[kGroundingPairCapacity];
  std::uint32_t pair_count;
  DirectGroundedObject objects[kGroundingObjectCapacity];
  std::uint32_t object_count;
  DirectChannelExposure exposures[kGroundingExposureCapacity];
  std::uint32_t exposure_count;
  std::uint32_t fence_refusals;
  std::uint32_t consequences_processed;
  std::uint32_t consequences_unbound;
};

static_assert(std::is_trivially_copyable_v<DirectMultimodalGroundingTable>);
static_assert(sizeof(DirectMultimodalGroundingTable) == 10520u);
static_assert(alignof(DirectMultimodalGroundingTable) == 4u);

}  // namespace substrate::direct_network

#endif
