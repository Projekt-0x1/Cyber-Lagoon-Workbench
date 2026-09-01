#ifdef DIRECT_ADULT_DEVICE_OPS_DECLARATIONS_ONLY
#undef DIRECT_ADULT_DEVICE_OPS_DECLARATIONS_ONLY
#endif

#include <cuda_runtime.h>

#include <stdexcept>

#include "hardware_native/direct_adult_resident_language_runtime.cuh"
#include "hardware_native/direct_adult_resident_language_runtime_abi.cuh"

namespace substrate::direct_network {

std::size_t direct_resident_language_runtime_storage_bytes() noexcept {
  return sizeof(DirectResidentLanguageRuntimeBlock);
}

DirectResidentLanguageRuntimeBlock* create_direct_resident_language_runtime() {
  DirectResidentLanguageRuntimeBlock* state = nullptr;
  const cudaError_t allocated = cudaMalloc(&state, sizeof(*state));
  if (allocated != cudaSuccess) {
    throw std::runtime_error("create_direct_resident_language_runtime: cudaMalloc");
  }
  const cudaError_t cleared = cudaMemset(state, 0, sizeof(*state));
  if (cleared != cudaSuccess) {
    cudaFree(state);
    throw std::runtime_error("create_direct_resident_language_runtime: cudaMemset");
  }
  return state;
}

void destroy_direct_resident_language_runtime(
    DirectResidentLanguageRuntimeBlock* state) noexcept {
  if (state != nullptr) (void)cudaFree(state);
}

__device__ void direct_resident_language_assimilate_owned(
    DirectResidentLanguageRuntimeBlock* state,
    const DirectExactHistoryHotPage* history) {
  if (state == nullptr || history == nullptr) return;
  resident_language_runtime_assimilate(
      state, history->records, history->committed_slots);
}

__global__ void direct_resident_language_assimilation_kernel(
    DirectResidentLanguageRuntimeBlock* state,
    const DirectExactHistoryHotPage* history) {
  if (blockIdx.x == 0u && threadIdx.x == 0u)
    direct_resident_language_assimilate_owned(state, history);
}

void launch_direct_resident_language_assimilation(
    DirectResidentLanguageRuntimeBlock* state,
    const DirectExactHistoryHotPage* history, CUstream_st* stream) {
  direct_resident_language_assimilation_kernel<<<1, 32, 0, stream>>>(state, history);
  const cudaError_t status = cudaGetLastError();
  if (status != cudaSuccess)
    throw std::runtime_error(
        "launch_direct_resident_language_assimilation: kernel launch");
}

}  // namespace substrate::direct_network
