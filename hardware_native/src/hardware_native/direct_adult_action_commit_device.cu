#ifdef DIRECT_ADULT_DEVICE_OPS_DECLARATIONS_ONLY
#undef DIRECT_ADULT_DEVICE_OPS_DECLARATIONS_ONLY
#endif
#include "hardware_native/direct_adult_runtime_frontiers.cuh"
#include "hardware_native/direct_adult_action_commit_device.cuh"
#include "hardware_native/direct_causal_program_commit_phase.cuh"
namespace substrate::direct_adult_core {
__device__ bool commit_resident_action_control_owned(
    DirectAdultActionControlRuntimeBlock* control,
    const ResidentRecipeOccurrence* occurrences,std::uint32_t occurrence_count,
    const ResidentMismatchOmissionFrontier* mismatch,std::uint32_t current_tick) {
  DirectProgramCommitPhaseReceipt receipt{};
  return resident_program_competition_commit_phase(
      control,occurrences,occurrence_count,mismatch,current_tick,&receipt);
}
}
