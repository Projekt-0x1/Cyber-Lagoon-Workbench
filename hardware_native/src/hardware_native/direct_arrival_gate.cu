#include "hardware_native/direct_arrival_gate.cuh"

#include <algorithm>
#include <stdexcept>
#include <string>

namespace substrate::direct_adult_core {
namespace {

using direct_network::DirectSha256Address;
using direct_network::detail::DirectSha256State;

void gate_check_cuda(cudaError_t status, const char* operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(status));
  }
}

// One thread recomputes the digest over the landed queue range (wrapping) and
// scrubs every landed byte on mismatch -- the boundary must not retain
// unverified payloads even cursor-deep.
__global__ void arrival_gate_kernel(ActivityEvent* queue, std::uint32_t capacity,
                                    std::uint32_t first_slot, std::uint32_t count,
                                    const DirectSha256Address* declared,
                                    std::uint32_t* match) {
  if (blockIdx.x != 0u || threadIdx.x != 0u) return;
  DirectSha256State state;
  const std::uint32_t first = first_slot % capacity;
  const std::uint32_t linear = (count < capacity - first) ? count : capacity - first;
  state.update(queue + first, sizeof(ActivityEvent) * static_cast<std::size_t>(linear));
  const std::uint32_t rest = count - linear;
  if (rest != 0u) {
    state.update(queue, sizeof(ActivityEvent) * static_cast<std::size_t>(rest));
  }
  const bool intact = state.finish() == *declared;
  *match = intact ? 1u : 0u;
  if (!intact) {
    for (std::uint32_t i = 0; i < linear; ++i) queue[first + i] = ActivityEvent{};
    for (std::uint32_t i = 0; i < rest; ++i) queue[i] = ActivityEvent{};
  }
}

}  // namespace

DirectArrivalSealV1 seal_sensory_arrival_v1(const DirectAdultRuntime* runtime) {
  if (runtime == nullptr || runtime->host_ingress_staging == nullptr) return {};
  DirectArrivalSealV1 seal;
  seal.first_event = runtime->host_ingress_publish_tail;
  const std::uint32_t write_tail = runtime->host_ingress_write_tail;
  if (write_tail < seal.first_event) return {};
  seal.event_count = write_tail - seal.first_event;
  if (seal.event_count == 0u) return seal;

  const std::uint32_t capacity = kMaxIngressQueueSize;
  const std::uint32_t first = seal.first_event % capacity;
  const std::uint32_t linear = std::min(seal.event_count, capacity - first);
  DirectSha256State state;
  state.update(runtime->host_ingress_staging + first,
               sizeof(ActivityEvent) * static_cast<std::size_t>(linear));
  const std::uint32_t rest = seal.event_count - linear;
  if (rest != 0u) {
    state.update(runtime->host_ingress_staging,
                 sizeof(ActivityEvent) * static_cast<std::size_t>(rest));
  }
  seal.payload_digest = state.finish();
  return seal;
}

DirectArrivalGateReceiptV1 enforce_sensory_arrival_v1(
    DirectAdultRuntime* runtime, const DirectArrivalSealV1& seal,
    const DirectSha256Address* declared_digest_device) {
  DirectArrivalGateReceiptV1 receipt;
  if (runtime == nullptr || declared_digest_device == nullptr || seal.event_count == 0u)
    return receipt;

  std::uint32_t* device_match = nullptr;
  gate_check_cuda(cudaMalloc(&device_match, sizeof(*device_match)),
                  "allocate arrival gate verdict");
  arrival_gate_kernel<<<1, 1, 0, runtime->stream>>>(
      runtime->ingress_queue, kMaxIngressQueueSize, seal.first_event,
      seal.event_count, declared_digest_device, device_match);
  gate_check_cuda(cudaGetLastError(), "launch arrival gate kernel");
  gate_check_cuda(cudaStreamSynchronize(runtime->stream),
                  "finish arrival gate kernel");

  std::uint32_t match = 0u;
  gate_check_cuda(cudaMemcpy(&match, device_match, sizeof(match), cudaMemcpyDeviceToHost),
                  "read arrival gate verdict");
  gate_check_cuda(cudaFree(device_match), "free arrival gate verdict");

  if (match != 0u) {
    receipt.verified = 1u;
    return receipt;
  }

  receipt.digest_mismatch = 1u;
  receipt.scrubbed_events = seal.event_count;
  runtime->host_ingress_publish_tail = seal.first_event;
  if (runtime->ingress_control != nullptr &&
      runtime->host_ingress_publish_slot_pinned != nullptr) {
    *runtime->host_ingress_publish_slot_pinned = seal.first_event;
    gate_check_cuda(
        cudaMemcpyAsync(&runtime->ingress_control->published_tail,
                        runtime->host_ingress_publish_slot_pinned,
                        sizeof(std::uint32_t), cudaMemcpyHostToDevice, runtime->stream),
        "retract ingress published_tail");
    gate_check_cuda(cudaStreamSynchronize(runtime->stream),
                    "finish published_tail retraction");
  }
  return receipt;
}

}  // namespace substrate::direct_adult_core
