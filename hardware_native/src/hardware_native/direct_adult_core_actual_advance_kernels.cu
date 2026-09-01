#include <cuda_runtime.h>
#include "hardware_native/direct_adult_core_kernels.cuh"
#include "hardware_native/direct_adult_runtime_frontiers.cuh"
#include "hardware_native/direct_adult_actual_frontier_condensation.cuh"

namespace substrate::direct_adult_core {

__global__ void advance_actual_frontier_from_propagation_kernel(
    DirectBrain brain, const DirectParticipationDescriptor* contributions,
    const std::uint32_t* contribution_count,
    std::uint32_t contribution_capacity, std::uint32_t current_tick,
    ResidentActualFrontier* frontier) {
  if (blockIdx.x != 0u || threadIdx.x != 0u || frontier == nullptr ||
      contributions == nullptr || contribution_count == nullptr)
    return;
  bool changed = admit_resident_actual_frontier_route_descendant(
      brain, contributions, contribution_count, contribution_capacity,
      current_tick, frontier);
  if (changed) condense_resident_actual_frontier_dispatch(brain, frontier);
}

}  // namespace substrate::direct_adult_core
