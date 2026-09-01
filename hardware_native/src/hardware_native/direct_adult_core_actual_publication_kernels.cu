#include <cuda_runtime.h>
#include "hardware_native/direct_adult_core_kernels.cuh"
#include "hardware_native/direct_boundary_condensation.cuh"
#include "hardware_native/direct_adult_checkpoint.cuh"
#include "hardware_native/direct_adult_device_ops.cuh"
#include "hardware_native/direct_adult_incentive_salience.cuh"
#include "hardware_native/direct_adult_activation.cuh"
#include "hardware_native/direct_adult_affect_body.cuh"
#include "hardware_native/direct_adult_wanting_liking_dissociation.cuh"
#include "hardware_native/direct_adult_temporal_discounting_competition.cuh"
#include "hardware_native/direct_adult_uncertainty_plateau.cuh"
#include "hardware_native/direct_adult_bounded_fanout.cuh"
#include "hardware_native/direct_adult_resource_maintenance.cuh"

namespace substrate::direct_adult_core {
__global__ void publish_admitted_actual_contacts_kernel(
    DirectBrain brain, NodeCausalParticipation* active, std::uint32_t* active_locks,
    EligibilityRecord* eligibility, std::uint32_t* live_eligibility, std::uint64_t* claim_directory,
    std::uint32_t* record_generations,
    const std::uint32_t* eligibility_free_slots,
    std::uint32_t* eligibility_free_state,
    DirectParticipationDescriptor* staging,
    std::uint32_t* staging_count, std::uint32_t staging_capacity, std::int32_t* incoming,
    AdultCoreMetrics* metrics, std::uint32_t tick, ResidentActualFrontier* frontier) {
  const bool published = publish_admitted_resident_actual_contacts(
      brain, frontier, active, active_locks, eligibility, live_eligibility,
      claim_directory, record_generations, eligibility_free_slots,
      eligibility_free_state, staging, staging_count, staging_capacity,
      incoming, metrics, tick);
  if (published && !has_unpublished_resident_actual_contact(frontier))
    condense_resident_actual_frontier_dispatch(brain, frontier);
}

__global__ void resolve_pending_actual_contacts_kernel(
    DirectBrain brain, const NodeCausalParticipation* active,
    std::uint32_t tick, ResidentActualFrontier* frontier, AdultCoreMetrics* metrics) {
  if (blockIdx.x != 0u || threadIdx.x != 0u) return;
  admit_resident_actual_frontier_pending_participation(
      brain, active, tick, frontier, metrics != nullptr ? &metrics->actual_frontier_causal : nullptr);
}
}  // namespace substrate::direct_adult_core
