#ifndef HARDWARE_NATIVE_DIRECT_ADULT_ACTION_BINDING_API_CUH
#define HARDWARE_NATIVE_DIRECT_ADULT_ACTION_BINDING_API_CUH

// Lightweight action-binding ABI shared by motor policy and device ops.
// Implementations remain in direct_adult_device_ops_heavy.inl.
#include "hardware_native/direct_adult_core.cuh"

namespace substrate::direct_adult_core {

struct ResidentRelationalNetworkClosure;

struct DirectActionBindingResult {
  std::uint32_t admitted;
  std::uint32_t participant_offset;
  std::uint32_t participant_count;
  std::uint32_t ancestry_incomplete;
  std::uint64_t diagnostic_upstream_ticket;
};
static_assert(std::is_standard_layout_v<DirectActionBindingResult> &&
              std::is_trivial_v<DirectActionBindingResult> &&
              std::has_unique_object_representations_v<DirectActionBindingResult>);
static_assert(sizeof(DirectActionBindingResult) == 24u);

__device__ DirectActionBindingResult bind_action_occurrence(
    const DirectParticipationDescriptor* current_contributions,
    const std::uint32_t* current_contribution_count,
    std::uint32_t current_contribution_capacity,
    const ResidentActualFrontier* actual_frontier,
    std::uint32_t require_exact_occurrence_identity,
    std::uint32_t motor_node, std::uint32_t current_tick,
    std::uint64_t action_ticket_id, std::uint32_t action_context,
    std::uint32_t motor_channel, std::uint32_t action_expiry_tick,
    std::uint32_t action_slot, DirectActionOccurrence* action_occurrences,
    DirectActionParticipationLink* action_links,
    std::uint32_t participant_capacity, AdultCoreMetrics* metrics,
    const ResidentActivationSoaPlane* activation_plane = nullptr,
    DirectBrain ancestry_brain = {},
    const EligibilityRecord* eligibility_table = nullptr,
    const std::uint32_t* eligibility_record_generations = nullptr,
    std::uint32_t current_contribution_count_snapshot = kInvalidIndex);

__device__ DirectActionBindingResult bind_network_boundary_action_occurrence(
    const ResidentRelationalNetworkClosure& network,
    const DirectParticipationDescriptor* current_contributions,
    const std::uint32_t* current_contribution_count,
    std::uint32_t current_contribution_capacity,
    const ResidentActualFrontier* actual_frontier,
    std::uint32_t motor_node, std::uint32_t current_tick,
    std::uint64_t action_ticket_id, std::uint32_t action_context,
    std::uint32_t motor_channel, std::uint32_t action_expiry_tick,
    std::uint32_t action_slot, DirectActionOccurrence* action_occurrences,
    DirectActionParticipationLink* action_links,
    std::uint32_t participant_capacity, AdultCoreMetrics* metrics,
    const ResidentActivationSoaPlane* activation_plane = nullptr,
    DirectBrain ancestry_brain = {},
    const EligibilityRecord* eligibility_table = nullptr,
    const std::uint32_t* eligibility_record_generations = nullptr,
    std::int32_t first_use_action_eligibility_q16 = 0);

}  // namespace substrate::direct_adult_core

#endif
