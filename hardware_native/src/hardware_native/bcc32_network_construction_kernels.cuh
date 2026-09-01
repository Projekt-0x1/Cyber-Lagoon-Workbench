// Patch 0004: __global__ kernel declarations for one developmental tick.
// Definitions live in bcc32_network_construction_kernels.cu; declared here
// so bcc32_network_life_function.cu can launch them via CUDA's normal
// separable-compilation model (this repository already builds
// network-recipe contracts with CUDA_SEPARABLE_COMPILATION ON).

#ifndef HARDWARE_NATIVE_BCC32_NETWORK_CONSTRUCTION_KERNELS_CUH
#define HARDWARE_NATIVE_BCC32_NETWORK_CONSTRUCTION_KERNELS_CUH

#include "hardware_native/bcc32_network_life_function.cuh"

namespace substrate::bcc32::network_recipe {

__global__ void clear_claims_kernel(TargetClaim* claims);
__global__ void clear_next_frontier_kernel(std::uint32_t* next_frontier_count);
// `report` receives only TickReport::unsupported_opcode from this kernel;
// every other counter is raised by commit_claims_kernel. Both accumulate
// into the same caller-owned TickReport, cleared by the caller before the
// tick (run_one_tick) or before the captured sequence (run_captured_ticks).
__global__ void evaluate_and_claim_kernel(LifeFunctionDeviceState state, TickReport* report);
__global__ void commit_claims_kernel(LifeFunctionDeviceState state, TickReport* report);

}  // namespace substrate::bcc32::network_recipe

#endif  // HARDWARE_NATIVE_BCC32_NETWORK_CONSTRUCTION_KERNELS_CUH
