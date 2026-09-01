#ifndef HARDWARE_NATIVE_DIRECT_RESOURCE_ECOLOGY_CUH
#define HARDWARE_NATIVE_DIRECT_RESOURCE_ECOLOGY_CUH

#include <cuda_runtime.h>

#include <cstdint>

#include "hardware_native/direct_adult_core.cuh"
#include "hardware_native/direct_resource_ecology_abi.cuh"
#include "hardware_native/direct_retention_policy.cuh"

namespace substrate::direct_adult {

#if defined(__CUDACC__)
#define DIRECT_ECOLOGY_HD __host__ __device__
#define DIRECT_ECOLOGY_DEVICE __device__
#else
#define DIRECT_ECOLOGY_HD
#define DIRECT_ECOLOGY_DEVICE
#endif

// The pool reserve/commit/cancel/release/reject/defer primitives now live in
// direct_resource_ecology_abi.cuh so that the physical allocators can charge the
// ledger without including this header (which pulls in the whole adult brain).
// Do not reintroduce copies here: two definitions of an admission gate is two
// admission gates.
#if defined(__CUDACC__)
DIRECT_ECOLOGY_DEVICE inline void device_record_work_counters(DirectResourceEcologyState* ecology,
                                                              std::uint64_t frontier_events,
                                                              std::uint64_t evaluations,
                                                              std::uint64_t eligibility_lookups) {
  if (ecology == nullptr)
    return;
  if (frontier_events > 0)
    atomicAdd(reinterpret_cast<unsigned long long*>(&ecology->work.frontier_events),
              static_cast<unsigned long long>(frontier_events));
  if (evaluations > 0)
    atomicAdd(reinterpret_cast<unsigned long long*>(&ecology->work.interaction_evaluations),
              static_cast<unsigned long long>(evaluations));
  if (eligibility_lookups > 0)
    atomicAdd(reinterpret_cast<unsigned long long*>(&ecology->work.eligibility_lookups),
              static_cast<unsigned long long>(eligibility_lookups));
}

// One shipping-path touch, one ledger event, one bounded debit.  Callers invoke
// this from the active incidence they already reached; the accounting path has
// no node/route capacity parameter and performs no compensating brain scan.
DIRECT_ECOLOGY_DEVICE inline bool device_record_active_frontier_cost(
    DirectResourceEcologyState* ecology, direct_adult_core::AdultCoreMetrics* metrics,
    std::uint64_t touched_units, std::uint32_t cost_per_unit_q16) {
  if (ecology == nullptr || metrics == nullptr || touched_units == 0u || cost_per_unit_q16 == 0u ||
      touched_units > (~std::uint64_t{0}) / cost_per_unit_q16)
    return false;
  device_record_work_counters(ecology, 1u, touched_units, 0u);
  atomicAdd(reinterpret_cast<unsigned long long*>(&metrics->maintenance_energy_debited),
            static_cast<unsigned long long>(touched_units * cost_per_unit_q16));
  return true;
}

DIRECT_ECOLOGY_DEVICE inline void device_record_churn_proposal(DirectResourceEcologyState* ecology,
                                                               std::uint64_t proposals) {
  if (ecology == nullptr || proposals == 0u)
    return;
  atomicAdd(reinterpret_cast<unsigned long long*>(&ecology->churn.proposals),
            static_cast<unsigned long long>(proposals));
  atomicAdd(reinterpret_cast<unsigned long long*>(&ecology->work.topology_proposals),
            static_cast<unsigned long long>(proposals));
}

DIRECT_ECOLOGY_DEVICE inline void device_record_churn_commit(DirectResourceEcologyState* ecology,
                                                             std::uint64_t commits) {
  if (ecology == nullptr || commits == 0u)
    return;
  atomicAdd(reinterpret_cast<unsigned long long*>(&ecology->churn.committed),
            static_cast<unsigned long long>(commits));
  atomicAdd(reinterpret_cast<unsigned long long*>(&ecology->work.topology_commits),
            static_cast<unsigned long long>(commits));
}

#endif

#undef DIRECT_ECOLOGY_HD
#undef DIRECT_ECOLOGY_DEVICE

}  // namespace substrate::direct_adult

#endif  // HARDWARE_NATIVE_DIRECT_RESOURCE_ECOLOGY_CUH
