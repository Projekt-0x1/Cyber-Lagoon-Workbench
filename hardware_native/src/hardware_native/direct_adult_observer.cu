#include "hardware_native/direct_adult_core.cuh"

#include <cuda_runtime.h>

#include <atomic>
#include <stdexcept>
#include <string>

namespace substrate::direct_adult_core {
namespace {

void require_cuda(cudaError_t status, const char* operation) {
  if (status != cudaSuccess)
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(status));
}

}  // namespace

AdultCoreMetrics get_adult_core_metrics(const DirectAdultRuntime* runtime) {
  AdultCoreMetrics metrics{};
  if (runtime != nullptr && runtime->device_metrics != nullptr) {
    require_cuda(cudaMemcpy(&metrics, runtime->device_metrics, sizeof(metrics),
                            cudaMemcpyDeviceToHost),
                 "cudaMemcpy metrics");
    metrics.tick_count = runtime->current_tick;
    metrics.ingress_overflow_drops += runtime->host_ingress_overflow_drops;
  }
  return metrics;
}

void reset_adult_core_metrics(DirectAdultRuntime* runtime) {
  if (runtime == nullptr) return;
  runtime->host_ingress_overflow_drops = 0u;
  if (runtime->device_metrics != nullptr)
    require_cuda(cudaMemset(runtime->device_metrics, 0, sizeof(AdultCoreMetrics)),
                 "cudaMemset metrics");
}

bool request_adult_core_observer_snapshot(DirectAdultRuntime* runtime) {
  if (runtime == nullptr || runtime->host_observer_snapshot == nullptr ||
      runtime->device_observer_snapshot == nullptr ||
      runtime->execution_authority != AdultExecutionAuthority::persistent ||
      runtime->observer_snapshot_pending)
    return false;

  AdultCoreObserverSnapshot* snapshot = runtime->host_observer_snapshot;
  const std::uint64_t request_id = ++runtime->observer_snapshot_requests;
  std::atomic_ref<std::uint64_t>(snapshot->request_id)
      .store(request_id, std::memory_order_release);
  runtime->observer_snapshot_pending = true;
  return true;
}

bool query_adult_core_observer_snapshot(DirectAdultRuntime* runtime,
                                        AdultCoreObserverSnapshot* out) {
  if (runtime == nullptr || out == nullptr ||
      !runtime->observer_snapshot_pending ||
      runtime->host_observer_snapshot == nullptr)
    return false;
  const AdultCoreObserverSnapshot* snapshot = runtime->host_observer_snapshot;
  const std::uint64_t completion_id =
      std::atomic_ref<const std::uint64_t>(snapshot->completion_id)
          .load(std::memory_order_acquire);
  if (completion_id != runtime->observer_snapshot_requests) return false;
  *out = *runtime->host_observer_snapshot;
  out->metrics.tick_count = out->resident_tick;
  out->metrics.ingress_overflow_drops += runtime->host_ingress_overflow_drops;
  runtime->observer_snapshot_pending = false;
  ++runtime->observer_snapshot_completions;
  return true;
}

}  // namespace substrate::direct_adult_core
