#ifndef HARDWARE_NATIVE_DIRECT_RESOURCE_ECOLOGY_CUH
#define HARDWARE_NATIVE_DIRECT_RESOURCE_ECOLOGY_CUH

#include <cuda_runtime.h>

#include <cstdint>

#include "hardware_native/direct_adult_legacy_oracle.cuh"
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

// Result of asking whether an exact eligibility record still pins a route.
//
// The third state is the load-bearing one. The scan is bounded per candidate, so
// it can run out of budget before it has seen every record. "I did not finish
// looking" is not "I found nothing", and collapsing the two is how a live causal
// reference gets reclaimed under load. `unknown` is treated as pinned by every
// caller: uncertain -> defer, never uncertain -> reclaim.
enum class DirectPinLookup : std::uint32_t {
  not_pinned = 0u,
  pinned = 1u,
  unknown = 2u,
};

// Generation-exact: a bare slot index is not an identity. Slot 42 generation 7
// may be reclaimed and reused as slot 42 generation 8 between two reads; matching
// on the index alone would pin the wrong route, and matching loosely on the other
// side would orphan a live eligibility record.
//
// `scan_bound` caps the records inspected for one candidate so that a bounded
// maintenance pass has a bounded *cost*, not merely a bounded candidate count.
// `lookups_out` reports what was actually spent so the caller can charge it.
DIRECT_ECOLOGY_DEVICE inline DirectPinLookup is_route_hard_pinned_by_eligibility_exact(
    const DirectEligibilityRecord* eligibility_bank, std::uint32_t eligibility_count,
    std::uint32_t route_slot, std::uint64_t route_generation, std::uint32_t scan_bound,
    std::uint32_t* lookups_out) {
  if (lookups_out != nullptr)
    *lookups_out = 0u;
  if (eligibility_bank == nullptr || eligibility_count == 0u)
    return DirectPinLookup::not_pinned;
  const std::uint32_t limit =
      (scan_bound == 0u || scan_bound > eligibility_count) ? eligibility_count : scan_bound;
  for (std::uint32_t i = 0; i < limit; ++i) {
    const DirectEligibilityRecord& rec = eligibility_bank[i];
    if (rec.state == EligibilityRecordState::live && rec.route.slot == route_slot &&
        rec.route.generation == route_generation) {
      if (lookups_out != nullptr)
        *lookups_out = i + 1u;
      return DirectPinLookup::pinned;
    }
  }
  if (lookups_out != nullptr)
    *lookups_out = limit;
  return limit < eligibility_count ? DirectPinLookup::unknown : DirectPinLookup::not_pinned;
}

// Convenience wrapper preserving the original boolean question. `unknown` folds
// into `true` because deferring a reclaim is always the safe direction.
DIRECT_ECOLOGY_DEVICE inline bool is_route_hard_pinned_by_eligibility(
    const DirectEligibilityRecord* eligibility_bank, std::uint32_t eligibility_count,
    std::uint32_t route_slot, std::uint64_t route_generation) {
  return is_route_hard_pinned_by_eligibility_exact(eligibility_bank, eligibility_count, route_slot,
                                                   route_generation, 0u,
                                                   nullptr) != DirectPinLookup::not_pinned;
}
#endif

struct DirectResourceMaintenanceRuntime {
  std::uint32_t* candidate_slots = nullptr;
  std::uint32_t* candidate_actions = nullptr;  // 0: none, 1: reclaim, 2: repair, 3: transition
  std::uint32_t* hard_pin_reasons = nullptr;
  std::uint32_t* winner_count = nullptr;
  // Window start claimed by claim_maintenance_window_kernel and consumed by the
  // scanning kernel. It must be device memory produced by a prior kernel, not a
  // value each scanning thread reads from a cursor another thread is advancing.
  // Two claimed windows per step: [0] routes, [1] context index.
  std::uint64_t* window_start = nullptr;
  // Optional per-slot visit census, sized route_capacity. Null in production.
  // When present the real maintenance kernel stamps every slot it visits, so the
  // fairness claim ("every slot is revisited within ceil(capacity/budget) steps")
  // is measured on the shipping code path rather than on a re-derivation of its
  // cursor arithmetic, which could agree with a broken kernel.
  std::uint32_t* visit_counts = nullptr;
  std::uint32_t scan_capacity = 0u;
};

void initialize_direct_resource_ecology_state(
    DirectResourceEcologyState* state,
    DirectRetentionPolicyKind policy = DirectRetentionPolicyKind::rich);

DirectResourceMaintenanceRuntime* create_resource_maintenance_runtime(std::uint32_t scan_capacity);

void destroy_resource_maintenance_runtime(DirectResourceMaintenanceRuntime* runtime);

void launch_direct_resource_maintenance_step(
    DirectBrainV01* brain, DirectRetentionState* retention_bank,
    DirectMinimalRetentionState* minimal_bank, const DirectEligibilityRecord* eligibility_bank,
    std::uint32_t eligibility_count, DirectResourceMaintenanceRuntime* maintenance_rt,
    std::uint64_t current_tick, std::uint64_t scan_budget, cudaStream_t stream = nullptr,
    std::uint32_t block_size = 128u,
    // 0X1-1175: device-resident live record count for `eligibility_bank`, used to
    // clamp `eligibility_count` on the device. `eligibility_count` is a HOST
    // value, and the adult's only source for it is a launch bound that ratchets
    // to capacity and cannot fall -- passing that bound here tells
    // is_route_hard_pinned_by_eligibility_exact the bank holds records it does
    // not, and a truncated scan over a bank that big returns `unknown`, which
    // folds to "pinned". Every fixture in this repository passes a real count, so
    // only the production adult was affected. Optional and nullptr by default, so
    // callers that already know the true count are byte-identical.
    const std::uint32_t* eligibility_count_device = nullptr);

// Seed the ledger from the physical arena. Legitimate exactly once, at birth or
// restore, when the physical objects already exist and no charge has been booked
// for them yet. It is NOT a periodic reconciliation: calling it on a running
// organism would overwrite real charges with a derived count and turn the ledger
// back into a shadow. Steady-state agreement between ledger and matter is
// maintained by the allocators charging through the ledger, and is checked (not
// enforced) by cuda_direct_resource_physical_authority_contract.
void seed_resource_pool_ledgers_from_matter(DirectBrainV01* brain, cudaStream_t stream = nullptr);

// Count physically live route slots straight out of slot_meta, independent of
// any ledger counter. This is the reconciliation instrument: it must never be
// used to *write* the ledger, only to check it.
void count_physical_live_routes(const DirectBrainV01* brain, std::uint32_t* device_out,
                                cudaStream_t stream = nullptr);

#undef DIRECT_ECOLOGY_HD
#undef DIRECT_ECOLOGY_DEVICE

}  // namespace substrate::direct_adult

#endif  // HARDWARE_NATIVE_DIRECT_RESOURCE_ECOLOGY_CUH
