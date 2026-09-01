#ifndef HARDWARE_NATIVE_DIRECT_ADULT_ACTION_COMMIT_DEVICE_CUH
#define HARDWARE_NATIVE_DIRECT_ADULT_ACTION_COMMIT_DEVICE_CUH
#include <cstdint>
namespace substrate::direct_adult_core {
struct DirectAdultActionControlRuntimeBlock;
struct ResidentRecipeOccurrence;
struct ResidentMismatchOmissionFrontier;
#if defined(__CUDACC__)
__device__ bool commit_resident_action_control_owned(
    DirectAdultActionControlRuntimeBlock* control,
    const ResidentRecipeOccurrence* occurrences,std::uint32_t occurrence_count,
    const ResidentMismatchOmissionFrontier* mismatch,std::uint32_t current_tick);
#endif
}
#endif
