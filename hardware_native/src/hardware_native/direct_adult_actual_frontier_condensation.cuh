#ifndef HARDWARE_NATIVE_DIRECT_ADULT_ACTUAL_FRONTIER_CONDENSATION_CUH
#define HARDWARE_NATIVE_DIRECT_ADULT_ACTUAL_FRONTIER_CONDENSATION_CUH

namespace substrate::direct_network {
struct DirectBrain;
}

namespace substrate::direct_adult_core {

struct ResidentActualFrontier;

// Stable device boundary for the production actual-frontier condensation pass.
// The implementation owns the heavyweight condensation headers so callers do
// not recompile the whole mechanism when only a contract assertion changes.
__device__ bool condense_resident_actual_frontier_dispatch(
    direct_network::DirectBrain brain, ResidentActualFrontier* frontier);

}  // namespace substrate::direct_adult_core

#endif
