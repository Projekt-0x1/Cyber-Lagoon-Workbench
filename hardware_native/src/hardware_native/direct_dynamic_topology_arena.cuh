#ifndef HARDWARE_NATIVE_DIRECT_DYNAMIC_TOPOLOGY_ARENA_CUH
#define HARDWARE_NATIVE_DIRECT_DYNAMIC_TOPOLOGY_ARENA_CUH

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

#include "hardware_native/direct_adult_legacy_oracle.cuh"

namespace substrate::direct_adult {

struct DirectTopologyConditionalSortCache;

#if defined(__CUDACC__)
#define DIRECT_TOPOLOGY_DEVICE __device__
#else
#define DIRECT_TOPOLOGY_DEVICE
#endif

constexpr std::uint32_t kNoWinnerIndex = 0xffffffffu;

struct DirectTopologyDeviceView {
  DirectTopologyProposal* proposals;
  std::uint32_t proposal_capacity;
  std::uint32_t* proposal_high_water;
};

struct DirectTopologyRuntime {
  DirectTopologyDeviceView view{};
  DirectTopologyProposal* proposals_sorted = nullptr;
  std::uint64_t* sort_keys_in = nullptr;
  std::uint64_t* sort_keys_out = nullptr;
  std::uint32_t* sort_values_in = nullptr;
  std::uint32_t* sort_values_out = nullptr;
  std::uint32_t* growth_source_winner = nullptr;
  std::uint32_t* growth_target_winner = nullptr;
  std::uint32_t* retract_source_winner = nullptr;
  std::uint32_t* retract_route_winner = nullptr;
  std::uint32_t* growth_flags = nullptr;
  std::uint32_t* retract_flags = nullptr;
  // 0X1-1175: the admission-stage rank/total pair that used to live here
  // (`growth_ranks`, `retract_ranks`, `growth_total`, `retract_total`) was
  // written by two rank scans per topology epoch and read by nobody. The live
  // pair below (`*_committed_ranks`) is the one every commit kernel reads.
  std::uint32_t* committed_route_slots = nullptr;
  std::uint32_t* retract_committed_flags = nullptr;
  std::uint32_t* retract_committed_ranks = nullptr;
  std::uint32_t* retract_committed_total = nullptr;
  std::uint32_t* growth_committed_flags = nullptr;
  std::uint32_t* growth_committed_ranks = nullptr;
  std::uint32_t* growth_committed_total = nullptr;
  std::uint32_t* free_count_snapshot = nullptr;
  // 0X1-1205: device-resident live-proposal compaction; `live_indices` holds original
  // indices and `live_count` stays on device, so canonical sort selection needs no D2H.
  std::uint32_t* live_indices = nullptr;
  std::uint32_t* live_count = nullptr;
  void* scan_storage = nullptr;
  std::size_t scan_storage_bytes = 0u;
  void* sort_storage = nullptr;
  std::size_t sort_storage_bytes = 0u;
  // gh #1245: host-only CUDA-Graph cache for the live-bound topology sequence.
  // It is copied inertly in device runtime values. The captured sequence is reused;
  // only per-call node parameters change, because proposal_work and its launch
  // launch extent can vary between
  // topology epochs.
  cudaStream_t topology_epoch_graph_stream = nullptr;
  // Node handles belong to the template graph, so keep it alive for the runtime
  // rather than destroying it after cudaGraphInstantiate.
  cudaGraph_t topology_epoch_graph_template = nullptr;
  cudaGraphExec_t topology_epoch_graph_exec = nullptr;
  // gh #1256: three Phase 1 fast-sort launches (compact/sort/gather) plus
  // #1245's original ten Phase 2-5 nodes, captured as one sequence.
  static constexpr int kTopologyEpochGraphNodeCount = 13;
  cudaGraphNode_t topology_epoch_graph_compact_node = {};
  // gh #1260: host-owned cache for the device-live conditional sort graph used
  // only when a host proposal bound is too large to imply the fast path. One
  // opaque pointer crosses value-copied runtime parameters; graph/cache state does not.
  DirectTopologyConditionalSortCache* conditional_sort_cache = nullptr;
};

DIRECT_TOPOLOGY_DEVICE inline void submit_direct_growth_proposal(
    DirectTopologyDeviceView view, std::uint32_t ordinal, std::uint32_t source,
    std::uint32_t target, std::uint64_t context_signature, std::int32_t conductance_q16,
    std::int32_t credit_q16, std::uint32_t delay, std::uint32_t route_flags,
    std::int32_t priority_q16, std::uint32_t implicit_family = kInvalidIndex,
    std::uint32_t implicit_slot = kInvalidIndex, std::uint32_t eligibility_context = kInvalidIndex,
    std::uint32_t eligibility_expires = 0u, std::uint32_t predicted_context = kInvalidIndex,
    std::uint64_t eligibility_history = 0u, std::uint64_t eligibility_root = 0u,
    Word learned_output_word = 0u) {
#if defined(__CUDA_ARCH__)
  if (ordinal >= view.proposal_capacity)
    return;
  if (view.proposal_high_water != nullptr)
    atomicMax(view.proposal_high_water, ordinal + 1u);
  DirectTopologyProposal proposal{};
  proposal.kind = DirectTopologyProposalKind::grow;
  proposal.source = source;
  proposal.target = target;
  proposal.route_slot = kInvalidIndex;
  proposal.route_generation = 0u;
  proposal.context_signature = context_signature;
  proposal.eligibility_history = eligibility_history;
  proposal.eligibility_root = eligibility_root;
  proposal.learned_output_word = learned_output_word;
  proposal.conductance_q16 = conductance_q16;
  proposal.credit_q16 = credit_q16;
  proposal.priority_q16 = priority_q16;
  proposal.delay = delay;
  proposal.route_flags = route_flags;
  proposal.ordinal = ordinal;
  proposal.implicit_family = implicit_family;
  proposal.implicit_slot = implicit_slot;
  proposal.eligibility_context = eligibility_context;
  proposal.eligibility_expires = eligibility_expires;
  proposal.predicted_context = predicted_context;
  view.proposals[ordinal] = proposal;
#else
  (void)view;
  (void)ordinal;
  (void)source;
  (void)target;
  (void)context_signature;
  (void)conductance_q16;
  (void)credit_q16;
  (void)delay;
  (void)route_flags;
  (void)priority_q16;
  (void)implicit_family;
  (void)implicit_slot;
  (void)eligibility_context;
  (void)eligibility_expires;
  (void)predicted_context;
  (void)eligibility_history;
  (void)eligibility_root;
  (void)learned_output_word;
#endif
}

DIRECT_TOPOLOGY_DEVICE inline void submit_direct_retract_proposal(
    DirectTopologyDeviceView view, std::uint32_t ordinal, std::uint32_t source,
    std::uint32_t route_slot, std::uint64_t route_generation, std::int32_t priority_q16) {
#if defined(__CUDA_ARCH__)
  if (ordinal >= view.proposal_capacity)
    return;
  if (view.proposal_high_water != nullptr)
    atomicMax(view.proposal_high_water, ordinal + 1u);
  DirectTopologyProposal proposal{};
  proposal.kind = DirectTopologyProposalKind::retract;
  proposal.source = source;
  proposal.target = kInvalidIndex;
  proposal.route_slot = route_slot;
  proposal.route_generation = route_generation;
  proposal.priority_q16 = priority_q16;
  proposal.ordinal = ordinal;
  proposal.implicit_family = kInvalidIndex;
  proposal.implicit_slot = kInvalidIndex;
  proposal.eligibility_context = kInvalidIndex;
  proposal.predicted_context = kInvalidIndex;
  view.proposals[ordinal] = proposal;
#else
  (void)view;
  (void)ordinal;
  (void)source;
  (void)route_slot;
  (void)route_generation;
  (void)priority_q16;
#endif
}

void initialize_direct_topology_state(DirectBrainV01* brain);
void destroy_direct_topology_state(DirectBrainV01* brain);
DirectTopologyRuntime* create_direct_topology_runtime(const DirectBrainV01& brain,
                                                      std::uint32_t proposal_capacity);
void destroy_direct_topology_runtime(DirectTopologyRuntime* runtime);
// github #1208 item 1: `stream` defaults to nullptr (the legacy stream), so
// every existing caller that does not pass one stays byte-identical. Callers
// that hold a DirectAdultRuntime pass its `runtime->stream` so this reset and
// the epoch it precedes land on the same stream as the rest of the adult
// step, instead of refusing capture at the first bare launch here.
void reset_direct_topology_proposals(DirectTopologyRuntime* runtime,
                                     std::uint32_t proposal_work,
                                     cudaStream_t stream = nullptr);
void launch_direct_topology_epoch(DirectBrainV01* brain, DirectTopologyRuntime* runtime,
                                  std::uint32_t proposal_work, AdultCounters* counters,
                                  std::uint32_t block_size = 128u,
                                  cudaStream_t stream = nullptr);

#undef DIRECT_TOPOLOGY_DEVICE

}  // namespace substrate::direct_adult

#endif  // HARDWARE_NATIVE_DIRECT_DYNAMIC_TOPOLOGY_ARENA_CUH
