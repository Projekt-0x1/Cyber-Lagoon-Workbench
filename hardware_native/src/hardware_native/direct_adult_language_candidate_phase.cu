#include "hardware_native/direct_adult_runtime_frontiers.cuh"
#include "hardware_native/direct_adult_language_action_candidate_device.cuh"
#include "hardware_native/direct_adult_language_candidate_phase.cuh"
namespace substrate::direct_adult_core {
__global__ void resident_language_candidate_admission_kernel(
    direct_network::DirectResidentLanguageRuntimeBlock* language,
    DirectAdultActionControlRuntimeBlock* control,
    const direct_network::DirectExactHistoryHotPage* history,
    const ResidentActualFrontier* frontier,
    const NodeCausalParticipation* active_participation,
    std::uint32_t node_count, std::uint32_t current_tick) {
  if (blockIdx.x != 0u || threadIdx.x != 0u || frontier == nullptr) return;
  ResidentRecipeOccurrence occurrences[kResidentActualFrontierCapacity]{};
  for (std::uint32_t i = 0u; i < kResidentActualFrontierCapacity; ++i)
    occurrences[i] = frontier->entries[i].occurrence;
  (void)plan_and_admit_resident_language_action_candidate_owned(
      language, control, history, occurrences,
      kResidentActualFrontierCapacity, active_participation,
      node_count, current_tick);
}
void launch_resident_language_candidate_admission_phase(DirectAdultRuntime* runtime) {
  if (runtime == nullptr || runtime->brain == nullptr ||
      runtime->brain->development == nullptr || runtime->actual_frontier == nullptr ||
      runtime->action_control_runtime == nullptr) return;
  resident_language_candidate_admission_kernel<<<1, 32, 0, runtime->stream>>>(
      runtime->resident_language, runtime->action_control_runtime,
      &runtime->brain->development->exact_history, runtime->actual_frontier,
      runtime->node_active_participation, runtime->brain->node_count,
      runtime->current_tick);
}
}
