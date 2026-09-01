#include "hardware_native/direct_adult_runtime_frontiers.cuh"
#include "hardware_native/direct_adult_action_commit_device.cuh"
#include "hardware_native/direct_adult_action_commit_phase.cuh"
namespace substrate::direct_adult_core {
__global__ void resident_action_commit_kernel(
    DirectAdultActionControlRuntimeBlock* control,
    const ResidentActualFrontier* frontier,
    const ResidentMismatchOmissionFrontier* mismatch,
    std::uint32_t current_tick) {
  if(blockIdx.x!=0u||threadIdx.x!=0u||control==nullptr||frontier==nullptr)return;
  ResidentRecipeOccurrence occurrences[kResidentActualFrontierCapacity]{};
  for(std::uint32_t i=0u;i<kResidentActualFrontierCapacity;++i)occurrences[i]=frontier->entries[i].occurrence;
  (void)commit_resident_action_control_owned(control,occurrences,kResidentActualFrontierCapacity,mismatch,current_tick);
}
void launch_resident_action_commit_phase(DirectAdultRuntime* runtime){
  if(runtime==nullptr||runtime->action_control_runtime==nullptr||runtime->actual_frontier==nullptr||runtime->mismatch_omission==nullptr)return;
  resident_action_commit_kernel<<<1,32,0,runtime->stream>>>(runtime->action_control_runtime,runtime->actual_frontier,&runtime->mismatch_omission->frontier,runtime->current_tick);
}
}
