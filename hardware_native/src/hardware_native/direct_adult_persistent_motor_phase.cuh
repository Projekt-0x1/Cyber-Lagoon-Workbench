#ifndef HARDWARE_NATIVE_DIRECT_ADULT_PERSISTENT_MOTOR_PHASE_CUH
#define HARDWARE_NATIVE_DIRECT_ADULT_PERSISTENT_MOTOR_PHASE_CUH
// Out-of-line persistent motor publication phase; order and grid barriers stay
// owned by persistent_direct_adult_kernel.

#include "hardware_native/direct_adult_persistent.cuh"

#include <cooperative_groups.h>

namespace substrate::direct_adult_core {

__device__ void persistent_motor_phase_device(
    cooperative_groups::grid_group grid,
    std::uint32_t current_tick,
    DirectNode* nodes,
    DirectRoute* routes,
    std::uint32_t route_capacity,
    const DirectBoundaryPort* boundary_ports,
    std::uint32_t boundary_port_count,
    ResidentActualFrontier* actual_frontier,
    ResidentActivationSoaPlane* activation_plane,
    direct_network::DirectExactHistoryHotPage* exact_history,
    MotorEvent* egress_queue,
    std::uint32_t* egress_head,
    std::uint32_t* egress_tail,
    EligibilityRecord* eligibility_table,
    std::uint32_t* eligibility_record_generations,
    AsynchronousTicket* ticket_table,
    std::uint32_t* ticket_count,
    std::uint32_t* ticket_table_locks,
    DirectActionOccurrence* action_occurrences,
    DirectActionParticipationLink* action_participation_links,
    std::uint32_t action_participant_capacity,
    ResidentPublicMotorTrajectory* resident_motor_trajectory,
    direct_network::DirectAffectBodyState* affect_body_state,
    std::int32_t* node_incoming_excitation,
    std::uint32_t* node_next_ancestry_incomplete,
    DirectParticipationDescriptor* participation_staging,
    std::uint32_t* participation_staging_count,
    std::uint32_t participation_staging_capacity,
    AdultCoreMetrics* device_metrics,
    AdultExecutionConfig config,
    DirectBrain brain,
    DirectEfferenceCopy* efference_ring,
    std::uint32_t* efference_head,
    std::uint32_t* efference_tail);

}  // namespace substrate::direct_adult_core

#endif  // HARDWARE_NATIVE_DIRECT_ADULT_PERSISTENT_MOTOR_PHASE_CUH
