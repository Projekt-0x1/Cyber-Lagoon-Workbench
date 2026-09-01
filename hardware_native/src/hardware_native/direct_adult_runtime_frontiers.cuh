#ifndef HARDWARE_NATIVE_DIRECT_ADULT_RUNTIME_FRONTIERS_CUH
#define HARDWARE_NATIVE_DIRECT_ADULT_RUNTIME_FRONTIERS_CUH
// Complete actual/successor/multi-horizon/mismatch frontier types for TUs that
// dereference them. Requires direct_adult_core.cuh already included so
// ActivityEvent and occurrence types are in scope. Network/morphology splices
// live here, not in core.cuh, so a mechanism edit rebuilds frontier owners.
#include "hardware_native/direct_adult_core.cuh"
namespace substrate::direct_adult_core {
#include "hardware_native/direct_adult_resident_relational_network.cuh"
#include "hardware_native/direct_adult_resident_network_boundary.cuh"
#include "hardware_native/direct_adult_resident_executable_morphology.cuh"
#include "hardware_native/direct_adult_resident_network_condensation.cuh"
#include "hardware_native/direct_adult_actual_frontier.cuh"
#include "hardware_native/direct_adult_successor_shadows.cuh"
#include "hardware_native/direct_adult_multi_horizon_prediction.cuh"
#include "hardware_native/direct_adult_mismatch_omission_runtime.cuh"
}  // namespace substrate::direct_adult_core
#endif  // HARDWARE_NATIVE_DIRECT_ADULT_RUNTIME_FRONTIERS_CUH
