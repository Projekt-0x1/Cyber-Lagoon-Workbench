#ifdef DIRECT_ADULT_DEVICE_OPS_DECLARATIONS_ONLY
#undef DIRECT_ADULT_DEVICE_OPS_DECLARATIONS_ONLY
#endif
#include "hardware_native/direct_adult_language_action_candidate.cuh"
#include "hardware_native/direct_adult_language_action_candidate_device.cuh"
#include "hardware_native/direct_adult_resident_language_runtime.cuh"
namespace substrate::direct_adult_core {
__device__ bool plan_and_admit_resident_language_action_candidate_owned(
    direct_network::DirectResidentLanguageRuntimeBlock* language,
    DirectAdultActionControlRuntimeBlock* control,
    const direct_network::DirectExactHistoryHotPage* history,
    const ResidentRecipeOccurrence* occurrences, std::uint32_t occurrence_count,
    const NodeCausalParticipation* active_participation,
    std::uint32_t node_count, std::uint32_t current_tick) {
  if (language == nullptr || control == nullptr || history == nullptr ||
      history->committed_slots == 0u || occurrences == nullptr ||
      occurrence_count == 0u || active_participation == nullptr || node_count == 0u)
    return false;
  const auto receipt = direct_network::resident_language_runtime_plan_only(
      language, history->records, history->committed_slots, current_tick);
  if (receipt.plan_started == 0u || language->executor.pending.admitted == 0u)
    return false;
  return admit_language_plan_as_action_candidate(
      language->executor.pending, history->records, history->committed_slots,
      occurrences, occurrence_count, active_participation, node_count,
      current_tick, control) != nullptr;
}
}
