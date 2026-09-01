#pragma once

#include <cuda_runtime.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace substrate::bcc32::delayed_consequence {

// These labels describe observer arms only. They never enter the resident world.
enum class AuditArm : std::uint32_t {
  preconsequence = 0u,
  delayed_consequence = 1u,
  zero_delay = 2u,
  shuffled_pairing = 3u,
  reversed_contingency = 4u,
  frozen_twin = 5u,
  count = 6u,
};

constexpr std::uint32_t kAuditArmCount = static_cast<std::uint32_t>(AuditArm::count);
constexpr std::uint32_t kMaximumTraceTicks = 128u;

struct DeviceTraceAudit {
  std::uint32_t lengths[kAuditArmCount]{};
  std::uint32_t captures[kAuditArmCount]{};
  std::uint32_t transition_counts[kAuditArmCount]{};
  std::uint32_t differs_from_delayed_mask = 0u;
};

// Read-only observer kernel. The causal episode is production BCC-32 F; this
// kernel merely audits the resulting action traces and has no route back into it.
static __global__ void audit_traces_kernel(const std::uint8_t* traces, const std::uint32_t* offsets,
                                           const std::uint32_t* lengths, DeviceTraceAudit* audit) {
  const std::uint32_t arm = blockIdx.x;
  if (arm >= kAuditArmCount || threadIdx.x != 0u)
    return;

  const std::uint32_t begin = offsets[arm];
  const std::uint32_t length = lengths[arm];
  std::uint32_t captures = 0u;
  std::uint32_t transitions = 0u;
  for (std::uint32_t tick = 0u; tick < length; ++tick) {
    captures += traces[begin + tick] != 0u ? 1u : 0u;
    if (tick != 0u && traces[begin + tick] != traces[begin + tick - 1u])
      ++transitions;
  }
  audit->lengths[arm] = length;
  audit->captures[arm] = captures;
  audit->transition_counts[arm] = transitions;

  if (arm == static_cast<std::uint32_t>(AuditArm::delayed_consequence))
    return;
  const std::uint32_t delayed_arm = static_cast<std::uint32_t>(AuditArm::delayed_consequence);
  const std::uint32_t delayed_begin = offsets[delayed_arm];
  const std::uint32_t delayed_length = lengths[delayed_arm];
  bool differs = length != delayed_length;
  const std::uint32_t common = length < delayed_length ? length : delayed_length;
  for (std::uint32_t tick = 0u; tick < common && !differs; ++tick)
    differs = traces[begin + tick] != traces[delayed_begin + tick];
  if (differs)
    atomicOr(&audit->differs_from_delayed_mask, 1u << arm);
}

inline void require_cuda(cudaError_t result, const char* operation) {
  if (result != cudaSuccess)
    throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(result));
}

inline DeviceTraceAudit audit_traces_cuda(
    const std::array<std::vector<std::uint8_t>, kAuditArmCount>& arms) {
  std::array<std::uint32_t, kAuditArmCount> offsets{};
  std::array<std::uint32_t, kAuditArmCount> lengths{};
  std::vector<std::uint8_t> flattened;
  for (std::uint32_t arm = 0u; arm < kAuditArmCount; ++arm) {
    if (arms[arm].size() > kMaximumTraceTicks)
      throw std::runtime_error("delayed-consequence trace exceeds audit capacity");
    offsets[arm] = static_cast<std::uint32_t>(flattened.size());
    lengths[arm] = static_cast<std::uint32_t>(arms[arm].size());
    flattened.insert(flattened.end(), arms[arm].begin(), arms[arm].end());
  }

  std::uint8_t* device_traces = nullptr;
  std::uint32_t* device_offsets = nullptr;
  std::uint32_t* device_lengths = nullptr;
  DeviceTraceAudit* device_audit = nullptr;
  try {
    require_cuda(cudaMalloc(&device_traces, flattened.size()),
                 "allocate delayed-consequence traces");
    require_cuda(cudaMalloc(&device_offsets, sizeof(offsets)),
                 "allocate delayed-consequence offsets");
    require_cuda(cudaMalloc(&device_lengths, sizeof(lengths)),
                 "allocate delayed-consequence lengths");
    require_cuda(cudaMalloc(&device_audit, sizeof(DeviceTraceAudit)),
                 "allocate delayed-consequence audit");
    require_cuda(
        cudaMemcpy(device_traces, flattened.data(), flattened.size(), cudaMemcpyHostToDevice),
        "upload delayed-consequence traces");
    require_cuda(
        cudaMemcpy(device_offsets, offsets.data(), sizeof(offsets), cudaMemcpyHostToDevice),
        "upload delayed-consequence offsets");
    require_cuda(
        cudaMemcpy(device_lengths, lengths.data(), sizeof(lengths), cudaMemcpyHostToDevice),
        "upload delayed-consequence lengths");
    require_cuda(cudaMemset(device_audit, 0, sizeof(DeviceTraceAudit)),
                 "clear delayed-consequence audit");
    audit_traces_kernel<<<kAuditArmCount, 1u>>>(device_traces, device_offsets, device_lengths,
                                                device_audit);
    require_cuda(cudaGetLastError(), "launch delayed-consequence trace audit");
    DeviceTraceAudit result{};
    require_cuda(cudaMemcpy(&result, device_audit, sizeof(result), cudaMemcpyDeviceToHost),
                 "download delayed-consequence audit");
    cudaFree(device_audit);
    cudaFree(device_lengths);
    cudaFree(device_offsets);
    cudaFree(device_traces);
    return result;
  } catch (...) {
    cudaFree(device_audit);
    cudaFree(device_lengths);
    cudaFree(device_offsets);
    cudaFree(device_traces);
    throw;
  }
}

}  // namespace substrate::bcc32::delayed_consequence
