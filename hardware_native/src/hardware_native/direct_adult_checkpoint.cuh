#ifndef HARDWARE_NATIVE_DIRECT_ADULT_CHECKPOINT_CUH
#define HARDWARE_NATIVE_DIRECT_ADULT_CHECKPOINT_CUH

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "hardware_native/direct_adult_core.cuh"
#include "hardware_native/direct_content_address.cuh"

namespace substrate::direct_adult_core {

inline constexpr std::size_t kDirectAdultCheckpointBufferCount = 51u;
inline constexpr std::size_t kDirectAdultCheckpointActualFrontierBuffer = 2u;
inline constexpr std::size_t kDirectAdultCheckpointEligibilityTableBuffer =
    10u;
inline constexpr std::size_t kDirectAdultCheckpointEligibilityLiveCountBuffer =
    11u;
inline constexpr std::size_t kDirectAdultCheckpointEligibilityDirectoryBuffer =
    12u;
inline constexpr std::size_t kDirectAdultCheckpointEligibilityLocksBuffer =
    13u;
inline constexpr std::size_t
    kDirectAdultCheckpointEligibilityGenerationsBuffer = 14u;
inline constexpr std::size_t kDirectAdultCheckpointDelayedPacketsBuffer = 26u;
inline constexpr std::size_t kDirectAdultCheckpointDelayedLiveCountBuffer =
    27u;
inline constexpr std::size_t kDirectAdultCheckpointDelayedFreeHeadBuffer = 28u;
inline constexpr std::size_t kDirectAdultCheckpointDelayedNextFreeBuffer = 29u;
inline constexpr std::size_t kDirectAdultCheckpointDelayedIdentitiesBuffer =
    30u;
inline constexpr std::size_t kDirectAdultCheckpointRouteTransportCursorBuffer =
    48u;
inline constexpr std::size_t kDirectAdultCheckpointResidentLanguageBuffer = 49u;
inline constexpr std::size_t kDirectAdultCheckpointActionControlBuffer = 50u;
inline constexpr std::size_t
    kDirectAdultCheckpointResidentMotorTrajectoryBuffer = 21u;
inline constexpr std::uint32_t kDirectAdultCheckpointVersion = 24u;

struct DirectAdultCheckpoint {
  std::uint32_t format_version = kDirectAdultCheckpointVersion;
  substrate::direct_network::DirectSha256Address payload_sha256{};
  DirectBrain brain{};
  substrate::direct_adult::DirectResourceEcologyState resource_ecology{};
  std::array<std::uint64_t, 27> arena_pointer_offsets{};
  std::vector<std::byte> arena;
  std::array<std::vector<std::byte>, kDirectAdultCheckpointBufferCount>
      device_buffers;
  std::vector<ActivityEvent> host_ingress_staging;
  std::vector<ResidentContactEpochReceipt> host_ingress_contact_staging;
  std::vector<ConsequenceIngressEvent> host_consequence_staging;
  AdultExecutionConfig config{};
  std::uint64_t resident_development_epochs = 0u;
  std::uint32_t current_tick = 0u;
  std::uint32_t participation_staging_capacity = 0u;
  std::uint32_t host_ingress_write_tail = 0u;
  std::uint32_t host_ingress_publish_tail = 0u;
  std::uint32_t host_ingress_observed_head = 0u;
  std::uint32_t host_ingress_dispatched_tail = 0u;
  std::uint64_t host_ingress_overflow_drops = 0u;
  std::uint64_t host_ingress_protocol_faults = 0u;
  std::uint32_t host_consequence_write_tail = 0u;
  std::uint32_t host_consequence_publish_tail = 0u;
  std::uint32_t host_consequence_observed_head = 0u;
  std::uint64_t host_consequence_overflow_drops = 0u;
  std::uint64_t host_consequence_protocol_faults = 0u;
  std::uint32_t host_ingress_head_snapshot = 0u;
  std::uint32_t host_ingress_publish_slot = 0u;
  std::uint32_t host_consequence_head_snapshot = 0u;
  std::uint32_t host_consequence_publish_slot = 0u;
  bool has_resource_ecology = false;
};

DirectAdultCheckpoint capture_direct_adult_checkpoint(
    const DirectAdultRuntime& runtime);
DirectAdultRuntime* restore_direct_adult_checkpoint(
    const DirectAdultCheckpoint& checkpoint, DirectBrain* out_brain);
// SHA-256 over every payload field except payload_sha256 itself; restore
// recomputes it before any allocation and refuses a mismatch.
substrate::direct_network::DirectSha256Address
direct_adult_checkpoint_payload_digest(
    const DirectAdultCheckpoint& checkpoint);

}  // namespace substrate::direct_adult_core

#endif  // HARDWARE_NATIVE_DIRECT_ADULT_CHECKPOINT_CUH
