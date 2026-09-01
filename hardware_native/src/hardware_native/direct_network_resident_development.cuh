#ifndef HARDWARE_NATIVE_DIRECT_NETWORK_RESIDENT_DEVELOPMENT_CUH
#define HARDWARE_NATIVE_DIRECT_NETWORK_RESIDENT_DEVELOPMENT_CUH

#include <cstddef>
#include <cstdint>
#include <cuda_runtime_api.h>

#include "hardware_native/direct_network_brain.cuh"

namespace substrate::direct_adult_core { struct AdultCoreMetrics; }
namespace substrate::direct_network {

struct RouteMutationProposal {
  std::uint32_t node;
  std::uint32_t route_slot;
  std::uint32_t target;
  std::uint32_t kind;  // 0 none, 1 grow/reactivate, 2 retract
  std::uint32_t route_flags;
  std::uint32_t recipe_cell;
  std::int32_t score_q16, committed_recipe_support_q16;
  std::uint32_t cost, planned_kind, construction_front_index, reserved; std::uint64_t committed_topology_identity, construction_front_generation;
};

// Growth proposal refusal sites, counted at their return sites. gh #1294: the
// array keeps later sites from expanding this ABI; kinds live in the .cu.
enum GrowthRefusalKind : std::uint32_t {
  kRefusedNotConstructor = 0u, kRefusedAtRouteCapacity, kRefusedNoFreeSlot,
  kRefusedNoRecipeCell, kRefusedRecipeRuleOutOfRange, kRefusedBelowGrowthThreshold,
  kRefusedNoEligibleTarget, kGrowthRefusalKindCount
};

// Proposal-to-commit grow outcomes. Target arbitration and developmental budget checks are
// admission stages, so this name does not call them post-admission drops. gh #1294.
enum GrowCommitOutcomeKind : std::uint32_t {
  kDroppedSlotInvalid = 0u, kDroppedSlotOccupied, kDroppedTargetInvalid,
  kDroppedLostTargetRace, kDroppedTargetInDegreeFull, kDroppedBudgetExhausted,
  kDroppedWorkBudgetExhausted, kGrowCommitOutcomeKindCount
};

using CommitDropKind = GrowCommitOutcomeKind;
inline constexpr std::uint32_t kCommitDropKindCount = kGrowCommitOutcomeKindCount;

enum RetractionCommitOutcomeKind : std::uint32_t {
  kRetractDroppedSlotInvalid = 0u, kRetractDroppedInactive,
  kRetractDroppedCausalPin, kRetractDroppedNoPressure,
  kRetractRefusedWorkBudget, kRetractCommitted, kRetractionCommitOutcomeKindCount
};
struct ResidentDevelopmentCounters {
  std::uint32_t growth_refusals[kGrowthRefusalKindCount];
  std::uint32_t commit_drops[kCommitDropKindCount];
  std::uint32_t retract_outcomes[kRetractionCommitOutcomeKindCount];
  std::uint32_t grow_proposals_emitted;
  std::uint32_t retract_proposals_emitted;
  std::uint32_t grown_routes;
  std::uint32_t retracted_routes;
  std::uint32_t matured_nodes;
  std::uint32_t field_updates;
  std::uint32_t recipe_updates;
  std::uint32_t recipe_neighbor_updates;
  // Ledger outcomes for post-birth structural change. `ledger_refusals` counts
  // growth the resource ecology said no to after the developmental budget had
  // already said yes -- if that can never be non-zero, the ledger is decoration.
  // `ledger_uncommits` counts live route units returned to the reserve, which is
  // the only way the grown brain's route pool can fall.
  std::uint32_t ledger_refusals;
  std::uint32_t ledger_uncommits;
  std::uint64_t resource_spent;
  std::uint64_t resource_reclaimed; std::uint32_t construction_fronts_examined, construction_fronts_recruited, construction_front_successions, construction_front_stale_refusals, repair_nomination_work, postbirth_recipes_condensed, postbirth_no_history;
};
struct TopologyCommitPlan {
  std::uint32_t proposal_count, accepted_count, growth_count;
  std::uint32_t retraction_count, normal_retraction_count, pressure_retraction_count, admitted;
  std::uint64_t growth_cost;
};
struct ResidentDevelopmentWorkspace {
  RouteMutationProposal* proposals = nullptr;
  std::uint32_t *costs = nullptr, *cost_prefix = nullptr, *history_prefix = nullptr;
  // One packed deterministic growth claim per target node. High 32 bits are
  // inverted signed score rank; low 32 bits are source node index. atomicMin
  // therefore makes the strongest proposal win, with node index as tie-break.
  std::uint64_t* target_claims = nullptr;
  std::uint32_t* block_sums = nullptr;
  std::uint32_t* block_offsets = nullptr;
  void* scan_storage = nullptr;
  std::size_t scan_storage_bytes = 0u;
  ResidentDevelopmentCounters* counters = nullptr; TopologyCommitPlan* topology_plan = nullptr;
  std::uint64_t* reserve_snapshot = nullptr;
  // Postbirth cause-memo scratch: kDirectExactHistoryHotPageCapacity + 1
  // words, last word holding the archived-pages epoch. Runtime-owned and
  // never captured, so the cache is derived on every restore, not stored.
  std::uint32_t* cause_memo = nullptr;
  std::uint32_t node_capacity = 0u;
  std::uint32_t block_capacity = 0u;
};
#if defined(__CUDACC__)
__device__ void resident_maintenance_plan_node(DirectBrain, const std::uint64_t*, const std::uint32_t*, std::int32_t,
                                               std::uint32_t*, TopologyCommitPlan*, direct_adult_core::AdultCoreMetrics*, std::uint32_t); __device__ void resident_maintenance_plan_weight_node(DirectBrain, const std::uint64_t*, const std::uint32_t*, std::int32_t, const TopologyCommitPlan*, std::uint32_t*, std::uint32_t);
__device__ void resident_maintenance_begin_transaction(DirectBrain, TopologyCommitPlan*, std::uint32_t); __device__ void resident_maintenance_begin_weight_transaction(DirectBrain, const std::uint32_t*, const std::uint32_t*, std::uint32_t, TopologyCommitPlan*, std::uint32_t);
__device__ void resident_maintenance_commit_node(DirectBrain, const std::uint64_t*, const std::uint32_t*, const std::uint32_t*,
                                                 const TopologyCommitPlan*, direct_adult_core::AdultCoreMetrics*, std::uint32_t);
__device__ void resident_maintenance_decay_node(DirectBrain, const std::uint64_t*, const std::uint32_t*, std::int32_t,
                                                const std::uint32_t*, const TopologyCommitPlan*, std::uint32_t);
__device__ void resident_development_advance_epoch(ResidentDevelopmentState*, std::uint64_t*);
__device__ void resident_development_mature_node(DirectBrain, std::uint32_t, ResidentDevelopmentCounters*);
__device__ void resident_development_propose_node(DirectBrain, std::uint32_t, RouteMutationProposal*,
                                                   std::uint32_t*, std::uint64_t*, ResidentDevelopmentCounters*); __device__ void resident_development_advance_committed_fronts(DirectBrain, RouteMutationProposal*, ResidentDevelopmentCounters*); __device__ void resident_development_commit_construction_fronts(DirectBrain, RouteMutationProposal*, std::uint32_t*, const std::uint64_t*, ResidentDevelopmentCounters*); __device__ void resident_development_nominate_committed_retractions(DirectBrain, RouteMutationProposal*, ResidentDevelopmentCounters*); __device__ void resident_development_propose_front(DirectBrain, std::uint32_t, RouteMutationProposal*, std::uint32_t*, std::uint64_t*, ResidentDevelopmentCounters*);
__device__ void resident_development_filter_node(const RouteMutationProposal*, std::uint32_t*, const std::uint64_t*, std::uint32_t, std::uint32_t);
__device__ void resident_development_prepare_topology_node(DirectBrain, std::uint32_t, RouteMutationProposal*, std::uint32_t*, const std::uint32_t*, const std::uint64_t*, const std::uint64_t*, const std::uint32_t*, ResidentDevelopmentCounters*, TopologyCommitPlan*);
__device__ void resident_development_finalize_topology_node(DirectBrain, std::uint32_t, RouteMutationProposal*, std::uint32_t*, const std::uint32_t*, ResidentDevelopmentCounters*, TopologyCommitPlan*);
__device__ void resident_development_admit_topology_work_node(DirectBrain, std::uint32_t, RouteMutationProposal*, std::uint32_t*, const std::uint32_t*, ResidentDevelopmentCounters*, TopologyCommitPlan*);
__device__ void resident_development_begin_topology_transaction(DirectBrain, TopologyCommitPlan*);
__device__ void resident_development_commit_topology_node(DirectBrain, std::uint32_t, RouteMutationProposal*, const std::uint32_t*, const std::uint32_t*, const std::uint64_t*, const std::uint64_t*, const std::uint32_t*, ResidentDevelopmentCounters*, const TopologyCommitPlan*);
__device__ void resident_development_finish_topology_transaction(DirectBrain, TopologyCommitPlan*); __device__ void resident_development_commit_recipe_transaction(DirectBrain, RouteMutationProposal*, ResidentDevelopmentCounters*); __device__ void resident_development_condense_postbirth_recipe(DirectBrain, ResidentDevelopmentCounters*, std::uint32_t* cause_memo);
__device__ void resident_development_finish_admission_epoch(
    DirectBrain, const ResidentDevelopmentCounters*);
__device__ void resident_development_commit_node(DirectBrain, std::uint32_t, RouteMutationProposal*, const std::uint32_t*, const std::uint64_t*, const std::uint64_t*, const std::uint32_t*, ResidentDevelopmentCounters*);
#endif
ResidentDevelopmentWorkspace create_resident_development_workspace(
    std::uint32_t node_capacity);
void destroy_resident_development_workspace(ResidentDevelopmentWorkspace* workspace);
std::size_t resident_development_workspace_bytes(std::uint32_t node_capacity);

// Asynchronous resident development epoch.  All proposals are computed in
// parallel, budget admission is deterministic by node index via prefix-scan,
// and commit is parallel.  No external Genome pointer is accepted.
void launch_resident_development_epoch(DirectBrain* brain,
                                       ResidentDevelopmentWorkspace* workspace,
                                       cudaStream_t stream = nullptr, std::uint32_t block_size = 256u,
                                       const std::uint32_t* route_causal_pin_bits = nullptr);
void launch_exact_history_maintenance(DirectBrain*, ResidentDevelopmentWorkspace*, const std::uint64_t*, std::int32_t, const std::uint32_t*, direct_adult_core::AdultCoreMetrics*, std::uint32_t, cudaStream_t, std::uint32_t);

ResidentDevelopmentCounters observe_resident_development(
    const ResidentDevelopmentWorkspace& workspace,
    cudaStream_t stream = nullptr);

}  // namespace substrate::direct_network

#endif  // HARDWARE_NATIVE_DIRECT_NETWORK_RESIDENT_DEVELOPMENT_CUH
