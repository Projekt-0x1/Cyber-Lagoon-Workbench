#ifndef HARDWARE_NATIVE_DIRECT_ADULT_CONTACT_RECEIPT_CUH
#define HARDWARE_NATIVE_DIRECT_ADULT_CONTACT_RECEIPT_CUH
#include <cstdint>
#include <type_traits>
namespace substrate::direct_adult_core {
enum class ResidentContactSelection : std::uint32_t {
  resident_owned = 1u, host_forward = 2u, host_selected = 3u, replay = 4u,
};
enum class ResidentContactIntegration : std::uint32_t {
  canonical = 1u, shadow = 2u, test_only = 3u,
};
struct alignas(8) ResidentContactEpochReceipt {
  std::uint64_t identity, source_identity, codec_identity, payload_identity;
  std::uint64_t boundary_session_epoch, ingress_sequence;
  ResidentContactSelection selection;
  ResidentContactIntegration integration;
  std::uint32_t source_available, port_index, consumed, reserved;
};
static_assert(std::is_standard_layout_v<ResidentContactEpochReceipt> &&
              std::is_trivial_v<ResidentContactEpochReceipt> &&
              std::has_unique_object_representations_v<ResidentContactEpochReceipt>);
}  // namespace substrate::direct_adult_core
#endif  // HARDWARE_NATIVE_DIRECT_ADULT_CONTACT_RECEIPT_CUH
