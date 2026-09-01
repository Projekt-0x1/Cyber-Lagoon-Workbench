#include <cuda_runtime.h>
#include "hardware_native/direct_adult_core_kernels.cuh"
#include "hardware_native/direct_adult_actual_frontier_condensation.cuh"

namespace substrate::direct_adult_core {
__global__ void condense_resident_actual_frontier_kernel(
    DirectBrain brain, ResidentActualFrontier* frontier) {
  condense_resident_actual_frontier_dispatch(brain, frontier);
}

}  // namespace substrate::direct_adult_core
