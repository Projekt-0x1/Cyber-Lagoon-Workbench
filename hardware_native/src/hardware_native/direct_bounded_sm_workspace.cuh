#ifndef HARDWARE_NATIVE_DIRECT_BOUNDED_SM_WORKSPACE_CUH
#define HARDWARE_NATIVE_DIRECT_BOUNDED_SM_WORKSPACE_CUH

#include <cstdint>

namespace substrate::direct_adult_core {

// One fixed scan tile per resident block. Runtime work never sizes a local
// array; larger frontiers are covered by the existing block-sum hierarchy.
constexpr std::uint32_t kDirectPersistentScanWorkspaceWords = 256u;
constexpr std::uint32_t kDirectPersistentScanWorkspaceBytes =
    kDirectPersistentScanWorkspaceWords * sizeof(std::uint32_t);

__host__ __device__ inline bool direct_persistent_workspace_admits(std::uint32_t requested_words) {
  return requested_words != 0u && requested_words <= kDirectPersistentScanWorkspaceWords;
}

}  // namespace substrate::direct_adult_core

#endif
