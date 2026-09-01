#ifndef HARDWARE_NATIVE_DIRECT_ADULT_OBSERVER_BUS_CUH
#define HARDWARE_NATIVE_DIRECT_ADULT_OBSERVER_BUS_CUH

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <vector>

#include "hardware_native/direct_adult_core.cuh"

namespace substrate::direct_adult_core {

// A transport observer, never an Adult input.  It reads only host-owned copies
// after their transport cursor is published and owns every byte it records.
class DirectAdultObserverBus {
 public:
  DirectAdultObserverBus() {
    ingress_.reserve(kMaxIngressQueueSize);
    egress_.reserve(kMaxEgressQueueSize);
  }

  [[nodiscard]] bool capture_published_ingress(
      const DirectAdultRuntime& runtime) {
    const std::uint32_t tail = runtime.host_ingress_publish_tail;
    const std::uint32_t count = tail - ingress_cursor_;
    if (runtime.host_ingress_staging == nullptr ||
        count > kMaxIngressQueueSize)
      return false;
    for (std::uint32_t cursor = ingress_cursor_; cursor != tail; ++cursor)
      ingress_.push_back(
          runtime.host_ingress_staging[cursor % kMaxIngressQueueSize]);
    ingress_cursor_ = tail;
    return true;
  }

  [[nodiscard]] bool capture_exported_egress(
      std::span<const MotorEvent> exported) {
    if (exported.size() > kMaxEgressQueueSize - egress_.size()) return false;
    egress_.insert(egress_.end(), exported.begin(), exported.end());
    return true;
  }

  // There is deliberately no runtime argument: observer bytes have no route
  // back to ingress, participation, learning, scheduling, or causal state.
  [[nodiscard]] bool submit(std::span<const std::byte>) noexcept {
    ++refused_submissions_;
    return false;
  }

  [[nodiscard]] const std::vector<ActivityEvent>& ingress() const noexcept {
    return ingress_;
  }
  [[nodiscard]] const std::vector<MotorEvent>& egress() const noexcept {
    return egress_;
  }
  [[nodiscard]] std::uint64_t refused_submissions() const noexcept {
    return refused_submissions_;
  }
  [[nodiscard]] static constexpr std::uint32_t device_api_calls() noexcept {
    return 0u;
  }
  [[nodiscard]] static constexpr std::uint32_t control_authority_paths() noexcept {
    return 0u;
  }

 private:
  std::vector<ActivityEvent> ingress_;
  std::vector<MotorEvent> egress_;
  std::uint32_t ingress_cursor_ = 0u;
  std::uint64_t refused_submissions_ = 0u;
};

}  // namespace substrate::direct_adult_core

#endif  // HARDWARE_NATIVE_DIRECT_ADULT_OBSERVER_BUS_CUH
