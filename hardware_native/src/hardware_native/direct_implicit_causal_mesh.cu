#include <cuda_runtime.h>

#include <algorithm>
#include <climits>
#include <stdexcept>
#include <string>
#include <vector>

#include "hardware_native/direct_implicit_causal_mesh.cuh"

namespace substrate::direct_adult {
namespace {

void implicit_cuda(cudaError_t status, const char* what) {
  if (status != cudaSuccess)
    throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
}

std::uint32_t next_power_of_two_implicit(std::uint32_t value) {
  value = std::max(2u, value) - 1u;
  value |= value >> 1;
  value |= value >> 2;
  value |= value >> 4;
  value |= value >> 8;
  value |= value >> 16;
  return value + 1u;
}

__device__ std::uint32_t exception_bucket(std::uint64_t key, std::uint32_t capacity) {
  std::uint64_t value = key;
  value ^= value >> 33;
  value *= 0xff51afd7ed558ccdull;
  value ^= value >> 33;
  value *= 0xc4ceb9fe1a85ec53ull;
  value ^= value >> 33;
  return static_cast<std::uint32_t>(value) & (capacity - 1u);
}

__global__ void set_exception_kernel(DirectBrainV01 brain, std::uint64_t key,
                                     std::int32_t delta_q16, std::uint32_t flags,
                                     std::uint32_t* result) {
  if (blockIdx.x != 0u || threadIdx.x != 0u)
    return;
  if (key == 0u || brain.implicit.exception_capacity == 0u) {
    *result = 0u;
    return;
  }
  std::uint32_t slot = direct_implicit_exception_bucket(key, brain.implicit.exception_capacity);
  for (std::uint32_t probe = 0; probe < 64u; ++probe) {
    DirectImplicitException& entry = brain.implicit.exceptions[slot];
    auto* address = reinterpret_cast<unsigned long long*>(&entry.key);
    const unsigned long long observed = atomicCAS(address, 0ull, key);
    if (observed == 0ull || observed == key) {
      if (observed == 0ull)
        atomicAdd(brain.implicit.exception_count, 1u);
      entry.conductance_delta_q16 = delta_q16;
      entry.flags = flags;
      __threadfence();
      *result = 1u;
      return;
    }
    slot = (slot + 1u) & (brain.implicit.exception_capacity - 1u);
  }
  *result = 0u;
}

}  // namespace

void initialize_direct_implicit_state(DirectBrainV01* brain,
                                      const DirectImplicitFamily* host_families,
                                      std::uint32_t family_count,
                                      std::uint64_t virtual_interaction_count,
                                      std::uint32_t exception_capacity) {
  if (brain == nullptr)
    throw std::invalid_argument("implicit state requires a brain");
  DirectImplicitPersistentState implicit{};
  implicit.family_count = family_count;
  implicit.virtual_interaction_count = virtual_interaction_count;
  implicit.exception_capacity = next_power_of_two_implicit(std::max(256u, exception_capacity));
  if (family_count != 0u) {
    implicit_cuda(cudaMalloc(&implicit.families, sizeof(DirectImplicitFamily) * family_count),
                  "allocate implicit families");
    implicit_cuda(cudaMemcpy(implicit.families, host_families,
                             sizeof(DirectImplicitFamily) * family_count,
                             cudaMemcpyHostToDevice),
                  "copy implicit families");
  }
  implicit_cuda(cudaMalloc(&implicit.exceptions,
                           sizeof(DirectImplicitException) * implicit.exception_capacity),
                "allocate implicit exceptions");
  implicit_cuda(cudaMemset(implicit.exceptions, 0,
                           sizeof(DirectImplicitException) * implicit.exception_capacity),
                "clear implicit exceptions");
  implicit_cuda(cudaMalloc(&implicit.exception_count, sizeof(std::uint32_t)),
                "allocate implicit exception count");
  implicit_cuda(cudaMemset(implicit.exception_count, 0, sizeof(std::uint32_t)),
                "clear implicit exception count");
  brain->implicit = implicit;
}

void destroy_direct_implicit_state(DirectBrainV01* brain) {
  if (brain == nullptr)
    return;
  cudaFree(brain->implicit.families);
  cudaFree(brain->implicit.exceptions);
  cudaFree(brain->implicit.exception_count);
  brain->implicit = DirectImplicitPersistentState{};
}

bool set_direct_implicit_exception(DirectBrainV01* brain, std::uint32_t family,
                                   std::uint32_t source, std::uint32_t virtual_slot,
                                   std::int32_t conductance_delta_q16,
                                   std::uint32_t flags) {
  if (brain == nullptr || family >= brain->implicit.family_count || source >= brain->node_count)
    return false;
  const std::uint64_t key = direct_implicit_exception_key(family, source, virtual_slot);
  std::uint32_t* device_result = nullptr;
  std::uint32_t result = 0u;
  implicit_cuda(cudaMalloc(&device_result, sizeof(std::uint32_t)), "allocate implicit result");
  set_exception_kernel<<<1, 32>>>(*brain, key, conductance_delta_q16, flags, device_result);
  implicit_cuda(cudaGetLastError(), "set implicit exception");
  implicit_cuda(cudaMemcpy(&result, device_result, sizeof(result), cudaMemcpyDeviceToHost),
                "read implicit exception result");
  cudaFree(device_result);
  return result != 0u;
}

}  // namespace substrate::direct_adult
