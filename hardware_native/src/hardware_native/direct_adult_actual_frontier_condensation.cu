#include "hardware_native/direct_boundary_condensation.cuh"

namespace substrate::direct_adult_core {

__device__ bool condense_resident_actual_frontier_dispatch(
    direct_network::DirectBrain brain, ResidentActualFrontier* frontier) {
  if (frontier == nullptr) return false;
  ResidentRelationalNetworkClosure closure{};
  ResidentNetworkBoundaryRelation boundary{};
  if (compose_resident_actual_frontier_network_boundary(
          brain, *frontier, &closure, &boundary))
    return condense_resident_actual_frontier_boundary_network(brain, frontier);
  return condense_resident_actual_frontier_network(brain, frontier);
}

}  // namespace substrate::direct_adult_core
