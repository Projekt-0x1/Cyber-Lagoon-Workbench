#ifndef HARDWARE_NATIVE_DIRECT_ADULT_LANGUAGE_ACTION_CANDIDATE_DEVICE_CUH
#define HARDWARE_NATIVE_DIRECT_ADULT_LANGUAGE_ACTION_CANDIDATE_DEVICE_CUH
#include <cstdint>
namespace substrate::direct_network {
struct DirectResidentLanguageRuntimeBlock;
struct DirectExactHistoryHotPage;
}
namespace substrate::direct_adult_core {
struct DirectAdultActionControlRuntimeBlock;
struct ResidentRecipeOccurrence;
struct NodeCausalParticipation;
#if defined(__CUDACC__)
__device__ bool plan_and_admit_resident_language_action_candidate_owned(
    direct_network::DirectResidentLanguageRuntimeBlock* language,
    DirectAdultActionControlRuntimeBlock* control,
    const direct_network::DirectExactHistoryHotPage* history,
    const ResidentRecipeOccurrence* occurrences, std::uint32_t occurrence_count,
    const NodeCausalParticipation* active_participation,
    std::uint32_t node_count, std::uint32_t current_tick);
#endif
}
#endif
