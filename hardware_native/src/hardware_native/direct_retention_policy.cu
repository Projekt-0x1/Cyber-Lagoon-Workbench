#include "hardware_native/direct_retention_policy.cuh"

#include <cuda_runtime.h>
#include <cstdint>

namespace substrate::direct_adult {
namespace {

__global__ void initialize_retention_state_kernel(
    DirectRetentionState* retention_bank, std::uint32_t route_capacity) {
  const std::uint32_t slot = blockIdx.x * blockDim.x + threadIdx.x;
  if (slot >= route_capacity) return;
  DirectRetentionState state{};
  state.logical_slot = slot;
  state.logical_generation = 1u;
  state.support_ema_q16 = 1 << 16;
  state.usage_ema_q16 = 1 << 16;
  state.contradiction_ema_q16 = 0u;
  state.repair_evidence_q16 = 0u;
  state.last_confirmed_conductance_q16 = 1 << 16;
  state.last_use_tick = 0u;
  state.quiet_protect_until = 0u;
  state.last_support_tick = 0u;
  state.source_revision = 0u;
  state.flags = kDirectRetentionNone;
  state.pin_reasons = kPinNone;
  retention_bank[slot] = state;
}

__global__ void initialize_minimal_retention_state_kernel(
    DirectMinimalRetentionState* minimal_bank, std::uint32_t route_capacity) {
  const std::uint32_t slot = blockIdx.x * blockDim.x + threadIdx.x;
  if (slot >= route_capacity) return;
  DirectMinimalRetentionState state{};
  state.activity_ema_q16 = 1 << 16;
  state.mismatch_count = 0u;
  state.last_use_tick = 0u;
  state.flags = kDirectRetentionNone;
  minimal_bank[slot] = state;
}

}  // namespace

void initialize_direct_retention_bank(
    DirectRetentionState* retention_bank, std::uint32_t route_capacity, cudaStream_t stream) {
  if (retention_bank == nullptr || route_capacity == 0u) return;
  const std::uint32_t block = 256u;
  const std::uint32_t grid = (route_capacity + block - 1u) / block;
  initialize_retention_state_kernel<<<grid, block, 0, stream>>>(retention_bank, route_capacity);
}

void initialize_direct_minimal_retention_bank(
    DirectMinimalRetentionState* minimal_bank, std::uint32_t route_capacity, cudaStream_t stream) {
  if (minimal_bank == nullptr || route_capacity == 0u) return;
  const std::uint32_t block = 256u;
  const std::uint32_t grid = (route_capacity + block - 1u) / block;
  initialize_minimal_retention_state_kernel<<<grid, block, 0, stream>>>(minimal_bank, route_capacity);
}

}  // namespace substrate::direct_adult
