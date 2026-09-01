#include <cuda_runtime.h>
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_scan.cuh>

#include <algorithm>
#include <climits>
#include <stdexcept>
#include <string>
#include <vector>

#include "hardware_native/direct_dynamic_topology_arena.cuh"
#include "hardware_native/direct_implicit_causal_mesh.cuh"
#include "hardware_native/direct_resource_ecology_abi.cuh"

namespace substrate::direct_adult {

struct DirectTopologyConditionalSortGraph {
  std::uint32_t sort_extent = 0u;
  std::uint32_t block_size = 0u;
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t exec = nullptr;
  cudaGraphNode_t compact_node = nullptr;
};

struct DirectTopologyConditionalSortCache {
  // One graph per geometric extent/block geometry, not per proposal_work. The
  // extents double from kFastSortSpan to proposal_capacity, so this cache is
  // finite for the runtime's lifetime even when the host launch bound changes
  // every epoch.
  std::vector<DirectTopologyConditionalSortGraph> graphs;
  // [0] = current real proposal_work, [1] = test/observer path receipt.
  std::uint32_t* sort_control = nullptr;
};

namespace {

void topology_cuda(cudaError_t status, const char* what) {
  if (status != cudaSuccess)
    throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
}

std::uint32_t grid_for_topology(std::uint32_t count, std::uint32_t block) {
  return std::max(1u, (count + block - 1u) / block);
}

// 0X1-1205: largest live-proposal run one block sorts in shared memory. Must be
// a power of two so the bitonic padding never exceeds the shared array.
constexpr std::uint32_t kFastSortCapacity = 8192u;
constexpr std::uint32_t kFastSortBlock = 1024u;
constexpr std::uint32_t kFastSortChunks = 16u;
constexpr std::uint32_t kFastSortSpan = kFastSortCapacity * kFastSortChunks;
constexpr std::uint32_t kSortPathFast = 1u;
constexpr std::uint32_t kSortPathRadix = 2u;
// gh #1260: `proposal_work` is an address/launch bound, not the resident live
// proposal count. A bound <= kFastSortSpan proves the compacted path is safe.
// Above that bound only the GPU may choose: a conditional graph reads live_count
// after compaction, keeps sparse epochs on the compacted path, and retains the
// nine-pass radix chain for genuinely dense epochs where the multi-run merge is
// more expensive. No live-count D2H readback enters the decision.
// 0X1-1175: block width of the single-block rank scan that replaces the CUB
// scan trio whenever a device-resident live-proposal count exists.
constexpr std::uint32_t kLiveScanBlock = 1024u;

std::uint32_t conditional_sort_extent(std::uint32_t proposal_work,
                                      std::uint32_t proposal_capacity) {
  std::uint32_t extent = kFastSortSpan;
  while (extent < proposal_work && extent <= proposal_capacity / 2u)
    extent <<= 1u;
  if (extent < proposal_work)
    extent = proposal_capacity;
  return std::min(extent, proposal_capacity);
}

__device__ bool route_is_live(const DirectBrainV01& brain, std::uint32_t slot,
                              std::uint64_t generation) {
  if (slot >= brain.route_capacity)
    return false;
  const DirectRouteSlotMeta meta = brain.topology.slot_meta[slot];
  return meta.live != 0u && meta.generation == generation;
}

__device__ bool source_contains_route_slot(const DirectBrainV01& brain, std::uint32_t source_index,
                                           std::uint32_t route_slot) {
  if (source_index >= brain.node_count)
    return false;
  const DirectNode source = brain.nodes[source_index];
  std::uint32_t current = source.first_route;
  for (std::uint32_t visited = 0; visited < source.route_count && current != kInvalidIndex;
       ++visited) {
    if (current == route_slot)
      return true;
    if (current >= brain.route_capacity)
      return false;
    current = brain.routes[current].next_route;
  }
  return false;
}

__device__ bool source_has_duplicate(const DirectBrainV01& brain,
                                     const DirectTopologyProposal& proposal) {
  if (proposal.source >= brain.node_count)
    return true;
  const DirectNode source = brain.nodes[proposal.source];
  std::uint32_t route = source.first_route;
  for (std::uint32_t visited = 0; visited < source.route_count && route != kInvalidIndex;
       ++visited) {
    if (route >= brain.route_capacity)
      return true;
    const DirectRouteSlotMeta meta = brain.topology.slot_meta[route];
    const DirectRoute current = brain.routes[route];
    const bool is_learned_motor =
        ((proposal.route_flags | current.flags) & kRouteFlagLearnedOutput) != 0u;
    const bool word_match =
        !is_learned_motor || current.learned_output_word == proposal.learned_output_word;
    if (meta.live != 0u && current.target == proposal.target &&
        current.context_signature == proposal.context_signature &&
        current.implicit_family == proposal.implicit_family &&
        current.implicit_slot == proposal.implicit_slot && word_match) {
      return true;
    }
    route = current.next_route;
  }
  return false;
}

__global__ void initialize_topology_slots_kernel(DirectBrainV01 brain) {
  const std::uint32_t slot = blockIdx.x * blockDim.x + threadIdx.x;
  if (slot >= brain.route_capacity)
    return;
  DirectRouteSlotMeta meta{};
  meta.generation = 1u;
  meta.live = slot < brain.route_count ? 1u : 0u;
  meta.logical_epoch = 0u;
  meta.logical_rank = slot;
  brain.topology.slot_meta[slot] = meta;
  if (slot >= brain.route_count) {
    brain.topology.free_slots[slot - brain.route_count] = slot;
  } else {
    const DirectRoute route = brain.routes[slot];
    if (route.target < brain.node_count)
      atomicAdd(&brain.topology.incoming_degree[route.target], 1u);
  }
}

// ---------------------------------------------------------------------------
// 0X1-1205 conditional-topology-sort
//
// The canonical proposal order is the ascending lexicographic order on the
// 480-bit tuple (K9, K8, K7, K6, K5, K4, K3, K2, K1, original_index), which the
// legacy nine-pass LSB-first stable radix chain below materialises. The helpers
// here express exactly the same key words so a single comparison sort can
// reproduce the legacy permutation bit-for-bit.
//
// The nine radix passes cost a fixed ~15.6 us each because CUB dispatches the
// single-tile (one block) kernel regardless of how many proposals are live, so
// an epoch with zero live proposals paid the same ~140 us as a full one. The
// fast path compacts the live proposals on device and sorts only those, letting
// the cost track the resident proposal count. No count is ever read back to the
// host: `proposal_work` alone selects the path and it is already host-side.
// ---------------------------------------------------------------------------

__device__ __forceinline__ std::uint64_t canonical_key_word(const DirectTopologyProposal& p,
                                                            int word) {
  switch (word) {
    case 9: {  // priority DESC, `none` last
      if (p.kind == DirectTopologyProposalKind::none)
        return ~0ULL;
      const std::uint32_t biased = static_cast<std::uint32_t>(p.priority_q16) ^ 0x80000000u;
      return static_cast<std::uint64_t>(0xffffffffu - biased) << 32;
    }
    case 8: {
      const std::uint32_t kind_val = p.kind == DirectTopologyProposalKind::none
                                         ? 0xffffffffu
                                         : static_cast<std::uint32_t>(p.kind);
      return (static_cast<std::uint64_t>(kind_val) << 32) | static_cast<std::uint64_t>(p.source);
    }
    case 7:
      return (static_cast<std::uint64_t>(p.route_flags) << 32) |
             static_cast<std::uint64_t>(p.target);
    case 6:
      return (static_cast<std::uint64_t>(p.implicit_family) << 32) |
             static_cast<std::uint64_t>(static_cast<std::uint32_t>(p.learned_output_word));
    case 5:
      return (static_cast<std::uint64_t>(p.route_slot) << 32) |
             static_cast<std::uint64_t>(p.implicit_slot);
    case 4:
      return p.context_signature;
    case 3:
      return p.route_generation;
    case 2:
      return p.eligibility_history;
    default:
      return p.eligibility_root;
  }
}

// Strict total order identical to the legacy stable nine-pass chain: the final
// tie-break on the original index reproduces the stability of that chain.
__device__ __forceinline__ bool canonical_less(const DirectTopologyProposal* proposals,
                                               std::uint32_t lhs_index,
                                               std::uint32_t rhs_index) {
  if (lhs_index == rhs_index)
    return false;
  const DirectTopologyProposal& a = proposals[lhs_index];
  const DirectTopologyProposal& b = proposals[rhs_index];
  for (int word = 9; word >= 1; --word) {
    const std::uint64_t ka = canonical_key_word(a, word);
    const std::uint64_t kb = canonical_key_word(b, word);
    if (ka != kb)
      return ka < kb;
  }
  return lhs_index < rhs_index;
}

__global__ void compact_live_proposals_kernel(const DirectTopologyProposal* proposals,
                                              const std::uint32_t* proposal_high_water,
                                              std::uint32_t hard_capacity,
                                              std::uint32_t* live_indices,
                                              std::uint32_t* live_count,
                                              std::uint32_t* active_work = nullptr) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (active_work != nullptr && i == 0u)
    *active_work = hard_capacity;
  const std::uint32_t high =
      proposal_high_water != nullptr ? min(*proposal_high_water, hard_capacity) : hard_capacity;
  if (i >= high)
    return;
  if (proposals[i].kind == DirectTopologyProposalKind::none)
    return;
  // The append order is arbitrary; `canonical_less` is a strict total order that
  // ends in the original index, so the sorted result is independent of it.
  const std::uint32_t slot = atomicAdd(live_count, 1u);
  live_indices[slot] = i;
}

#if CUDART_VERSION >= 12040
// #1260: the only ambiguous sort decision is made here, after compaction and on
// the GPU. One IF/ELSE conditional handle selects exactly one body.
__global__ void select_topology_sort_path_kernel(
    const std::uint32_t* live_count, cudaGraphConditionalHandle sort_handle,
    std::uint32_t* path_receipt) {
  if (blockIdx.x != 0u || threadIdx.x != 0u)
    return;
  const bool use_fast = *live_count <= kFastSortSpan;
  cudaGraphSetConditional(sort_handle, use_fast ? 1u : 0u);
  if (path_receipt != nullptr)
    *path_receipt = use_fast ? kSortPathFast : kSortPathRadix;
}
#endif

// Chunked bitonic sort over the compacted live indices. Block `blockIdx.x` owns
// the run [blockIdx.x * kFastSortCapacity, ...) and sorts it in shared memory;
// the runs are stitched into one canonical order by rank in the gather kernel
// below. The number of bitonic stages is derived from the device-resident live
// count, so an epoch with no live proposals exits immediately and a small epoch
// runs few stages, and a block whose run begins at or past the live count exits
// without touching memory -- which is what lets the host launch a bound
// (ceil(proposal_work / kFastSortCapacity)) instead of a count it would have to
// read back.
__global__ void sort_live_proposals_kernel(const DirectTopologyProposal* proposals,
                                           const std::uint32_t* live_indices,
                                           const std::uint32_t* live_count,
                                           std::uint32_t* sorted_indices) {
  __shared__ std::uint32_t shared_indices[kFastSortCapacity];
  const std::uint32_t live = *live_count;
  const std::uint32_t run_begin = blockIdx.x * kFastSortCapacity;
  if (run_begin >= live)
    return;
  const std::uint32_t run_end =
      live < run_begin + kFastSortCapacity ? live : run_begin + kFastSortCapacity;
  const std::uint32_t run = run_end - run_begin;

  std::uint32_t padded = 1u;
  while (padded < run)
    padded <<= 1;

  for (std::uint32_t i = threadIdx.x; i < padded; i += blockDim.x)
    shared_indices[i] = i < run ? live_indices[run_begin + i] : kInvalidIndex;
  __syncthreads();

  for (std::uint32_t k = 2u; k <= padded; k <<= 1) {
    for (std::uint32_t j = k >> 1; j > 0u; j >>= 1) {
      for (std::uint32_t i = threadIdx.x; i < padded; i += blockDim.x) {
        const std::uint32_t partner = i ^ j;
        if (partner <= i)
          continue;
        const std::uint32_t a = shared_indices[i];
        const std::uint32_t b = shared_indices[partner];
        // kInvalidIndex is the padding sentinel: it orders above every real
        // proposal so the padding sinks to the tail. `swap_needed` means a > b.
        bool swap_needed;
        if (a == kInvalidIndex) {
          swap_needed = (b != kInvalidIndex);
        } else if (b == kInvalidIndex) {
          swap_needed = false;
        } else {
          swap_needed = canonical_less(proposals, b, a);
        }
        const bool ascending = (i & k) == 0u;
        if (swap_needed == ascending) {
          shared_indices[i] = b;
          shared_indices[partner] = a;
        }
      }
      __syncthreads();
    }
  }

  for (std::uint32_t i = threadIdx.x; i < run; i += blockDim.x)
    sorted_indices[run_begin + i] = shared_indices[i];
}

// 0X1-1175: the runs left by the sort above are each internally canonical but
// not globally so. An element's global rank is its position within its own run
// plus, for every other run, the number of elements of that run that order
// strictly before it -- and every run is sorted, so that count is one binary
// search. `canonical_less` is a strict total order whose last tie-break is the
// original proposal index, so no two live proposals compare equal, every rank
// is distinct, and the ranks are exactly a permutation of [0, live). With a
// single run the loop body never executes and the rank is the position, which
// is bit-for-bit the pre-chunking behaviour.
__global__ void gather_live_sorted_proposals_kernel(DirectTopologyRuntime runtime,
                                                    const std::uint32_t* live_count) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  const std::uint32_t live = *live_count;
  if (i >= live)
    return;
  const std::uint32_t* sorted = runtime.sort_values_out;
  const std::uint32_t self_run = i / kFastSortCapacity;
  const std::uint32_t value = sorted[i];
  std::uint32_t rank = i - self_run * kFastSortCapacity;
  const std::uint32_t runs = (live + kFastSortCapacity - 1u) / kFastSortCapacity;
  for (std::uint32_t r = 0u; r < runs; ++r) {
    if (r == self_run)
      continue;
    const std::uint32_t begin = r * kFastSortCapacity;
    const std::uint32_t end = live < begin + kFastSortCapacity ? live : begin + kFastSortCapacity;
    std::uint32_t lo = begin;
    std::uint32_t hi = end;
    while (lo < hi) {
      const std::uint32_t mid = lo + ((hi - lo) >> 1);
      if (canonical_less(runtime.view.proposals, sorted[mid], value))
        lo = mid + 1u;
      else
        hi = mid;
    }
    rank += lo - begin;
  }
  runtime.proposals_sorted[rank] = runtime.view.proposals[value];
}

__global__ void init_sort_indices_kernel(std::uint32_t* values_in, std::uint32_t proposal_work) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < proposal_work) {
    values_in[i] = i;
  }
}

__global__ void gather_pass1_kernel(const DirectTopologyProposal* proposals,
                                    const std::uint32_t* values_in,
                                    std::uint64_t* keys_in,
                                    std::uint32_t proposal_work) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= proposal_work) return;
  const DirectTopologyProposal p = proposals[values_in[i]];
  keys_in[i] = p.eligibility_root;
}

__global__ void gather_pass2_kernel(const DirectTopologyProposal* proposals,
                                    const std::uint32_t* values_in,
                                    std::uint64_t* keys_in,
                                    std::uint32_t proposal_work) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= proposal_work) return;
  const DirectTopologyProposal p = proposals[values_in[i]];
  keys_in[i] = p.eligibility_history;
}

__global__ void gather_pass3_kernel(const DirectTopologyProposal* proposals,
                                    const std::uint32_t* values_in,
                                    std::uint64_t* keys_in,
                                    std::uint32_t proposal_work) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= proposal_work) return;
  const DirectTopologyProposal p = proposals[values_in[i]];
  keys_in[i] = p.route_generation;
}

__global__ void gather_pass4_kernel(const DirectTopologyProposal* proposals,
                                    const std::uint32_t* values_in,
                                    std::uint64_t* keys_in,
                                    std::uint32_t proposal_work) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= proposal_work) return;
  const DirectTopologyProposal p = proposals[values_in[i]];
  keys_in[i] = p.context_signature;
}

__global__ void gather_pass5_kernel(const DirectTopologyProposal* proposals,
                                    const std::uint32_t* values_in,
                                    std::uint64_t* keys_in,
                                    std::uint32_t proposal_work) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= proposal_work) return;
  const DirectTopologyProposal p = proposals[values_in[i]];
  keys_in[i] = (static_cast<std::uint64_t>(p.route_slot) << 32) | static_cast<std::uint64_t>(p.implicit_slot);
}

__global__ void gather_pass6_kernel(const DirectTopologyProposal* proposals,
                                    const std::uint32_t* values_in,
                                    std::uint64_t* keys_in,
                                    std::uint32_t proposal_work) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= proposal_work) return;
  const DirectTopologyProposal p = proposals[values_in[i]];
  keys_in[i] = (static_cast<std::uint64_t>(p.implicit_family) << 32) | static_cast<std::uint64_t>(static_cast<std::uint32_t>(p.learned_output_word));
}

__global__ void gather_pass7_kernel(const DirectTopologyProposal* proposals,
                                    const std::uint32_t* values_in,
                                    std::uint64_t* keys_in,
                                    std::uint32_t proposal_work) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= proposal_work) return;
  const DirectTopologyProposal p = proposals[values_in[i]];
  keys_in[i] = (static_cast<std::uint64_t>(p.route_flags) << 32) | static_cast<std::uint64_t>(p.target);
}

__global__ void gather_pass8_kernel(const DirectTopologyProposal* proposals,
                                    const std::uint32_t* values_in,
                                    std::uint64_t* keys_in,
                                    std::uint32_t proposal_work) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= proposal_work) return;
  const DirectTopologyProposal p = proposals[values_in[i]];
  const std::uint32_t kind_val = p.kind == DirectTopologyProposalKind::none ? 0xffffffffu : static_cast<std::uint32_t>(p.kind);
  keys_in[i] = (static_cast<std::uint64_t>(kind_val) << 32) | static_cast<std::uint64_t>(p.source);
}

__global__ void gather_pass9_kernel(const DirectTopologyProposal* proposals,
                                    const std::uint32_t* values_in,
                                    std::uint64_t* keys_in,
                                    std::uint32_t proposal_work,
                                    const std::uint32_t* active_work = nullptr) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= proposal_work) return;
  const std::uint32_t proposal_index = values_in[i];
  if (active_work != nullptr && proposal_index >= *active_work) {
    // Conditional graphs are cached at a geometric sort extent. Padding may
    // contain bytes from an older, wider epoch; force it behind the current
    // lawful proposal span without clearing or sorting from a host readback.
    keys_in[i] = ~0ULL;
    return;
  }
  const DirectTopologyProposal p = proposals[proposal_index];
  if (p.kind == DirectTopologyProposalKind::none) {
    keys_in[i] = ~0ULL;
  } else {
    const std::uint32_t biased_priority = static_cast<std::uint32_t>(p.priority_q16) ^ 0x80000000u;
    const std::uint32_t inverted_priority = 0xffffffffu - biased_priority;
    keys_in[i] = static_cast<std::uint64_t>(inverted_priority) << 32;
  }
}

__global__ void gather_sorted_proposals_kernel(DirectTopologyRuntime runtime,
                                              std::uint32_t proposal_work) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= proposal_work)
    return;
  const std::uint32_t orig_idx = runtime.sort_values_out[i];
  runtime.proposals_sorted[i] = runtime.view.proposals[orig_idx];
}

// gh #1299: bounded to live_bound like finalize_implicit_materialization_requests_kernel.
// Every dead-tail proposal (index >= live) is `none` from reset_direct_topology_proposals'
// zero-init, and `none` matches neither branch below, so this kernel is a no-op for the
// whole [live, proposal_work) padding tail regardless of where the loop stops.
__global__ void initialize_touched_winners_kernel(DirectBrainV01 brain, DirectTopologyRuntime runtime,
                                                  std::uint32_t proposal_work,
                                                  const std::uint32_t* live_bound) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  std::uint32_t bound = proposal_work;
  if (live_bound != nullptr) {
    const std::uint32_t live = *live_bound;
    bound = live < proposal_work ? live : proposal_work;
  }
  if (i >= bound)
    return;
  const DirectTopologyProposal proposal = runtime.proposals_sorted[i];
  if (proposal.kind == DirectTopologyProposalKind::grow ||
      proposal.kind == DirectTopologyProposalKind::repair ||
      proposal.kind == DirectTopologyProposalKind::materialize_implicit) {
    if (proposal.source < brain.node_count)
      atomicExch(&runtime.growth_source_winner[proposal.source], kNoWinnerIndex);
    if (proposal.target < brain.node_count)
      atomicExch(&runtime.growth_target_winner[proposal.target], kNoWinnerIndex);
  } else if (proposal.kind == DirectTopologyProposalKind::retract) {
    if (proposal.source < brain.node_count)
      atomicExch(&runtime.retract_source_winner[proposal.source], kNoWinnerIndex);
    if (proposal.route_slot < brain.route_capacity)
      atomicExch(&runtime.retract_route_winner[proposal.route_slot], kNoWinnerIndex);
  }
}

// gh #1299: same live-tail no-op argument as initialize_touched_winners_kernel above.
__global__ void arbitrate_topology_kernel(DirectBrainV01 brain, DirectTopologyRuntime runtime,
                                          std::uint32_t proposal_work,
                                          const std::uint32_t* live_bound) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  std::uint32_t bound = proposal_work;
  if (live_bound != nullptr) {
    const std::uint32_t live = *live_bound;
    bound = live < proposal_work ? live : proposal_work;
  }
  if (i >= bound)
    return;
  const DirectTopologyProposal proposal = runtime.proposals_sorted[i];
  if (proposal.kind == DirectTopologyProposalKind::grow ||
      proposal.kind == DirectTopologyProposalKind::repair ||
      proposal.kind == DirectTopologyProposalKind::materialize_implicit) {
    if (proposal.source < brain.node_count && proposal.target < brain.node_count) {
      atomicMin(&runtime.growth_source_winner[proposal.source], i);
      atomicMin(&runtime.growth_target_winner[proposal.target], i);
    }
  } else if (proposal.kind == DirectTopologyProposalKind::retract) {
    if (proposal.source < brain.node_count && proposal.route_slot < brain.route_capacity) {
      atomicMin(&runtime.retract_source_winner[proposal.source], i);
      atomicMin(&runtime.retract_route_winner[proposal.route_slot], i);
    }
  }
}

// gh #1299: bounded to live_bound. For i >= live this kernel only rewrites
// growth_flags[i]/retract_flags[i] to 0, which reset_direct_topology_proposals
// already set them to for the whole proposal_work range before this epoch ran --
// the write is redundant, not load-bearing, so skipping it changes no state.
__global__ void mark_topology_admission_kernel(DirectBrainV01 brain,
                                               DirectTopologyRuntime runtime,
                                               std::uint32_t proposal_work,
                                               const std::uint32_t* live_bound,
                                               AdultCounters* counters) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  std::uint32_t bound = proposal_work;
  if (live_bound != nullptr) {
    const std::uint32_t live = *live_bound;
    bound = live < proposal_work ? live : proposal_work;
  }
  if (i >= bound)
    return;
  runtime.growth_flags[i] = 0u;
  runtime.retract_flags[i] = 0u;
  const DirectTopologyProposal proposal = runtime.proposals_sorted[i];
  if (proposal.kind == DirectTopologyProposalKind::none)
    return;
  atomicAdd(&counters->structural_proposals, 1u);
  if (proposal.kind == DirectTopologyProposalKind::grow ||
      proposal.kind == DirectTopologyProposalKind::repair ||
      proposal.kind == DirectTopologyProposalKind::materialize_implicit) {
    if (proposal.source >= brain.node_count || proposal.target >= brain.node_count)
      return;
    if (runtime.growth_source_winner[proposal.source] != i ||
        runtime.growth_target_winner[proposal.target] != i)
      return;
    const DirectNode source = brain.nodes[proposal.source];
    if (source.route_count >= brain.topology.max_resident_fanout) {
      atomicAdd(&counters->structural_fanout_reject, 1u);
      return;
    }
    if (brain.topology.incoming_degree[proposal.target] >= brain.topology.max_resident_fanout) {
      atomicAdd(&counters->structural_target_reject, 1u);
      return;
    }
    if (source_has_duplicate(brain, proposal)) {
      atomicAdd(&counters->structural_duplicate_reject, 1u);
      return;
    }
    runtime.growth_flags[i] = 1u;
    return;
  }
  if (proposal.kind == DirectTopologyProposalKind::retract) {
    if (proposal.source >= brain.node_count || proposal.route_slot >= brain.route_capacity)
      return;
    if (runtime.retract_source_winner[proposal.source] != i ||
        runtime.retract_route_winner[proposal.route_slot] != i)
      return;
    if (!route_is_live(brain, proposal.route_slot, proposal.route_generation)) {
      atomicAdd(&counters->structural_stale_reject, 1u);
      return;
    }
    const DirectRoute route = brain.routes[proposal.route_slot];
    if (route.source != proposal.source || route.eligibility_q16 != 0 ||
        !source_contains_route_slot(brain, proposal.source, proposal.route_slot)) {
      atomicAdd(&counters->structural_stale_reject, 1u);
      return;
    }
    runtime.retract_flags[i] = 1u;
  }
}

__global__ void finalize_scan_totals_kernel(const std::uint32_t* flags,
                                            const std::uint32_t* ranks,
                                            std::uint32_t proposal_work,
                                            std::uint32_t* total) {
  if (threadIdx.x != 0u || blockIdx.x != 0u)
    return;
  if (proposal_work == 0u) {
    *total = 0u;
    return;
  }
  const std::uint32_t last = proposal_work - 1u;
  *total = ranks[last] + flags[last];
}

// 0X1-1175: live-bounded exclusive rank scan.
//
// `cub::DeviceScan::ExclusiveSum` takes `num_items` as a HOST value, so with
// `proposal_work` as that value the scan trio (DeviceScanInitKernel,
// DeviceScanKernel, finalize_scan_totals_kernel) ran eight launches per epoch at
// the launch bound no matter how many proposals existed. There is no CUB overload
// that reads the item count from device memory, so a live-sized scan has to be
// written out; one block covers the whole range by partitioning it per thread,
// which also folds the separate total kernel away. #1260 keeps this device-live
// scan on every production epoch; `per_thread` grows with live work, not the host
// proposal bound.
//
// Why the prefix is enough: `mark_topology_admission_kernel`,
// `commit_retractions_kernel` and `validate_growth_candidates_kernel` each store
// 0 into their flag array for EVERY index before any early return, and only ever
// store 1 for a proposal whose kind is not `none`. The live proposals occupy
// exactly the sorted positions [0, live) (#1205), so flags[live, proposal_work)
// are identically zero. An exclusive sum is therefore constant across that tail
// and the total taken at index live-1 equals the total taken at proposal_work-1.
// Every reader of a rank array (`publish_freed_slots_kernel`,
// `commit_growth_kernel`) is guarded by its flag being non-zero, so no reader
// ever observes the tail this kernel leaves untouched.
//
// Determinism: the index partition is a pure function of `bound` and the fixed
// block width, and every partial is an integer addition in a fixed order, so the
// output does not depend on the schedule and is bit-identical to the CUB scan
// over the same prefix.
__global__ void live_exclusive_scan_kernel(const std::uint32_t* flags, std::uint32_t* ranks,
                                           const std::uint32_t* live_bound,
                                           std::uint32_t proposal_work, std::uint32_t* total) {
  __shared__ std::uint32_t chunk_sums[kLiveScanBlock];
  std::uint32_t bound = proposal_work;
  if (live_bound != nullptr) {
    const std::uint32_t live = *live_bound;
    bound = live < proposal_work ? live : proposal_work;
  }
  if (bound == 0u) {
    if (threadIdx.x == 0u)
      *total = 0u;
    return;
  }

  const std::uint32_t threads = blockDim.x;
  const std::uint32_t per_thread = (bound + threads - 1u) / threads;
  const std::uint32_t begin = threadIdx.x * per_thread;
  const std::uint32_t end = (begin + per_thread) < bound ? (begin + per_thread) : bound;

  std::uint32_t chunk = 0u;
  for (std::uint32_t i = begin; i < end; ++i)
    chunk += flags[i];
  chunk_sums[threadIdx.x] = chunk;
  __syncthreads();

  // Hillis-Steele inclusive scan over the per-thread chunk sums.
  for (std::uint32_t offset = 1u; offset < threads; offset <<= 1) {
    std::uint32_t addend = 0u;
    if (threadIdx.x >= offset)
      addend = chunk_sums[threadIdx.x - offset];
    __syncthreads();
    if (threadIdx.x >= offset)
      chunk_sums[threadIdx.x] += addend;
    __syncthreads();
  }

  std::uint32_t running = threadIdx.x == 0u ? 0u : chunk_sums[threadIdx.x - 1u];
  for (std::uint32_t i = begin; i < end; ++i) {
    ranks[i] = running;
    running += flags[i];
  }
  if (threadIdx.x == 0u)
    *total = chunk_sums[threads - 1u];
}

__device__ bool unlink_route_from_source(DirectBrainV01 brain, std::uint32_t source_index,
                                         std::uint32_t route_slot) {
  DirectNode& source = brain.nodes[source_index];
  std::uint32_t previous = kInvalidIndex;
  std::uint32_t current = source.first_route;
  for (std::uint32_t visited = 0; visited < source.route_count && current != kInvalidIndex;
       ++visited) {
    if (current >= brain.route_capacity)
      return false;
    const DirectRoute route = brain.routes[current];
    if (current == route_slot) {
      if (previous == kInvalidIndex) {
        source.first_route = route.next_route;
      } else {
        brain.routes[previous].next_route = route.next_route;
      }
      if (source.route_count != 0u)
        --source.route_count;
      // #1179: this source's route membership just changed, so every cache
      // keyed on (source, membership) is now stale. Bumping the revision is
      // what lets the packed panel's first staleness guard and the
      // representation compiler's promotion-drift check actually fire; before
      // this producer existed they compared 0 to 0 and could not.
      //
      // Deliberately non-atomic, matching the adjacent route_count update:
      // the guards test inequality against a snapshot, and this value only
      // ever increases, so a lost update under concurrent mutation of one
      // source still leaves it different from any earlier snapshot. It is a
      // change detector, not a count.
      ++source.source_revision;
      return true;
    }
    previous = current;
    current = route.next_route;
  }
  return false;
}

// gh #1242: folds the former standalone single-thread snapshot_free_count_kernel
// launch into this kernel's block 0/thread 0, guarded the same way gh #1244
// folds the topology epoch's other two single-thread barriers (see
// publish_freed_slots_kernel and finalize_implicit_materialization_requests_kernel
// below). Safe because commit_retractions_kernel never reads or writes
// free_count/free_count_snapshot itself, and the fold still executes strictly
// before publish_freed_slots_kernel's separate launch (its only reader), which
// remains the correctness-relevant barrier.
// gh #1299: bounded to live_bound. For i >= live, retract_flags[i] is 0 (see
// mark_topology_admission_kernel above), so this thread's only write is
// retract_committed_flags[i] = 0 -- already its reset-kernel default.
__global__ void commit_retractions_kernel(DirectBrainV01 brain, DirectTopologyRuntime runtime,
                                          std::uint32_t proposal_work,
                                          const std::uint32_t* live_bound,
                                          AdultCounters* counters,
                                          const std::uint32_t* free_count,
                                          std::uint32_t* free_count_snapshot) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    *free_count_snapshot = *free_count;
  }
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  std::uint32_t bound = proposal_work;
  if (live_bound != nullptr) {
    const std::uint32_t live = *live_bound;
    bound = live < proposal_work ? live : proposal_work;
  }
  if (i >= bound)
    return;
  runtime.retract_committed_flags[i] = 0u;
  if (runtime.retract_flags[i] == 0u)
    return;
  const DirectTopologyProposal proposal = runtime.proposals_sorted[i];
  const std::uint32_t route_slot = proposal.route_slot;
  DirectRouteSlotMeta& meta = brain.topology.slot_meta[route_slot];
  if (meta.live == 0u || meta.generation != proposal.route_generation ||
      !unlink_route_from_source(brain, proposal.source, route_slot)) {
    atomicAdd(&counters->structural_stale_reject, 1u);
    return;
  }
  const DirectRoute retired = brain.routes[route_slot];
  if (retired.target < brain.node_count)
    atomicSub(&brain.topology.incoming_degree[retired.target], 1u);
  meta.live = 0u;
  ++meta.generation;
  meta.logical_epoch = 0u;
  meta.logical_rank = 0u;
  DirectRoute cleared{};
  cleared.next_route = kInvalidIndex;
  cleared.eligibility_context = kInvalidIndex;
  cleared.predicted_context = kInvalidIndex;
  cleared.implicit_family = kInvalidIndex;
  cleared.implicit_slot = kInvalidIndex;
  brain.routes[route_slot] = cleared;
  atomicSub(brain.live_route_count, 1u);
  if (brain.development != nullptr) {
    atomicAdd(reinterpret_cast<unsigned long long*>(&brain.development->reclaimed_resource), 1ull);
  }
  atomicAdd(&counters->structural_retract, 1u);
  // Exactly-once release. This line is reached only on the path that actually
  // made the physical object unreachable (meta.live cleared, route unlinked from
  // its source, slot generation bumped); every early return above leaves the
  // charge standing. Booking the release anywhere else -- at proposal time, or from
  // the folded free_count aggregate in publish_freed_slots_kernel -- would decouple
  // the ledger from the physical retirement it is supposed to account for.
  device_release_pool_units(brain.resource_ecology, DirectResourcePoolKind::explicit_interaction,
                            1u);
  runtime.retract_committed_flags[i] = 1u;
}

// gh #1244: folded from the former standalone finish_retractions_kernel<<<1,32>>>.
// Writes brain.topology.free_count / brain.development->constructor_reserve --
// disjoint from this kernel's per-thread writes into brain.topology.free_slots,
// which are addressed through free_before_ptr (free_count_snapshot), a different
// pointer. No aliasing, so folding the barrier into thread (0,0) is safe and
// removes a dedicated launch every topology epoch.
// gh #1299: per-thread work below bounded to live_bound. retract_committed_flags[i]
// is 0 for i >= live (see commit_retractions_kernel above), so this thread would
// have exited on that check anyway; the block-0/thread-0 guard above (gh #1244) is
// unaffected -- it runs regardless of proposal_work/live_bound.
__global__ void publish_freed_slots_kernel(DirectBrainV01 brain, DirectTopologyRuntime runtime,
                                           std::uint32_t proposal_work,
                                           const std::uint32_t* live_bound,
                                           const std::uint32_t* free_before_ptr) {
  if (blockIdx.x == 0u && threadIdx.x == 0u) {
    const std::uint32_t actual_reclaimed = *runtime.retract_committed_total;
    const std::uint32_t free_count_before = *brain.topology.free_count;
    *brain.topology.free_count =
        (free_count_before + actual_reclaimed < brain.route_capacity)
            ? (free_count_before + actual_reclaimed)
            : brain.route_capacity;
    if (brain.development != nullptr) {
      brain.development->constructor_reserve += actual_reclaimed;
    }
  }
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  std::uint32_t bound = proposal_work;
  if (live_bound != nullptr) {
    const std::uint32_t live = *live_bound;
    bound = live < proposal_work ? live : proposal_work;
  }
  if (i >= bound || runtime.retract_committed_flags[i] == 0u)
    return;
  const std::uint32_t rank = runtime.retract_committed_ranks[i];
  const DirectTopologyProposal proposal = runtime.proposals_sorted[i];
  const std::uint32_t free_before = *free_before_ptr;
  if (free_before + rank < brain.route_capacity) {
    brain.topology.free_slots[free_before + rank] = proposal.route_slot;
  }
}

// gh #1299: bounded to live_bound. For i >= live, growth_flags[i] is 0 (see
// mark_topology_admission_kernel above), so this thread's only write is
// growth_committed_flags[i] = 0 -- already its reset-kernel default.
__global__ void validate_growth_candidates_kernel(DirectBrainV01 brain,
                                                  DirectTopologyRuntime runtime,
                                                  std::uint32_t proposal_work,
                                                  const std::uint32_t* live_bound) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  std::uint32_t bound = proposal_work;
  if (live_bound != nullptr) {
    const std::uint32_t live = *live_bound;
    bound = live < proposal_work ? live : proposal_work;
  }
  if (i >= bound)
    return;
  runtime.growth_committed_flags[i] = 0u;
  if (runtime.growth_flags[i] == 0u)
    return;
  const DirectTopologyProposal proposal = runtime.proposals_sorted[i];
  const DirectNode source = brain.nodes[proposal.source];
  if (source.route_count >= brain.topology.max_resident_fanout)
    return;
  if (brain.topology.incoming_degree[proposal.target] >= brain.topology.max_resident_fanout)
    return;
  if (source_has_duplicate(brain, proposal))
    return;
  runtime.growth_committed_flags[i] = 1u;
}

// gh #1299: bounded to live_bound. For i >= live, growth_committed_flags[i] is 0
// (see validate_growth_candidates_kernel above), so this thread's only write is
// committed_route_slots[i] = kInvalidIndex -- already reset_direct_topology_proposals'
// 0xff sentinel default for the whole proposal_work range.
__global__ void commit_growth_kernel(DirectBrainV01 brain, DirectTopologyRuntime runtime,
                                     std::uint32_t proposal_work, const std::uint32_t* live_bound,
                                     AdultCounters* counters) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  std::uint32_t bound = proposal_work;
  if (live_bound != nullptr) {
    const std::uint32_t live = *live_bound;
    bound = live < proposal_work ? live : proposal_work;
  }
  if (i >= bound)
    return;
  runtime.committed_route_slots[i] = kInvalidIndex;
  if (runtime.growth_committed_flags[i] == 0u)
    return;
  const std::uint32_t rank = runtime.growth_committed_ranks[i];
  const std::uint32_t free_count = *brain.topology.free_count;
  const std::uint32_t total_to_grow = *runtime.growth_committed_total;

  // Reservation FIRST, before any free slot is claimed. This is the whole point
  // of #1178: the free list is not the authority any more, the ledger is. A pool
  // capacity set below route_capacity must be able to refuse growth that the free
  // list would happily have served, otherwise "finite silicon" is a counter that
  // watches allocation rather than a law that governs it.
  if (!device_reserve_pool_units(brain.resource_ecology,
                                 DirectResourcePoolKind::explicit_interaction, 1u)) {
    atomicAdd(&counters->growth_exhausted, 1u);
    runtime.growth_committed_flags[i] = 0u;
    return;
  }

  if (rank >= free_count || rank >= total_to_grow) {
    device_cancel_pool_reservation(brain.resource_ecology,
                                   DirectResourcePoolKind::explicit_interaction, 1u);
    atomicAdd(&counters->growth_exhausted, 1u);
    runtime.growth_committed_flags[i] = 0u;
    return;
  }
  const std::uint32_t slot = brain.topology.free_slots[free_count - 1u - rank];
  if (slot >= brain.route_capacity) {
    device_cancel_pool_reservation(brain.resource_ecology,
                                   DirectResourcePoolKind::explicit_interaction, 1u);
    atomicAdd(&counters->growth_exhausted, 1u);
    runtime.growth_committed_flags[i] = 0u;
    return;
  }
  DirectRouteSlotMeta& meta = brain.topology.slot_meta[slot];
  if (meta.live != 0u) {
    device_cancel_pool_reservation(brain.resource_ecology,
                                   DirectResourcePoolKind::explicit_interaction, 1u);
    atomicAdd(&counters->structural_stale_reject, 1u);
    runtime.growth_committed_flags[i] = 0u;
    return;
  }
  const DirectTopologyProposal proposal = runtime.proposals_sorted[i];
  DirectRoute grown{};
  grown.source = proposal.source;
  grown.target = proposal.target;
  grown.context_signature = proposal.context_signature;
  grown.eligibility_history = proposal.eligibility_history;
  grown.eligibility_root = proposal.eligibility_root;
  grown.learned_output_word = proposal.learned_output_word;
  grown.conductance_q16 = proposal.conductance_q16;
  grown.eligibility_q16 = proposal.eligibility_context == kInvalidIndex ? 0 : kEligibilityOneQ16;
  grown.last_credit_q16 = proposal.credit_q16;
  grown.eligibility_context = proposal.eligibility_context;
  grown.eligibility_expires = proposal.eligibility_expires;
  grown.predicted_context = proposal.predicted_context;
  grown.delay = proposal.delay;
  grown.flags = proposal.route_flags;
  grown.implicit_family = proposal.implicit_family;
  grown.implicit_slot = proposal.implicit_slot;
  DirectNode& source = brain.nodes[proposal.source];
  grown.next_route = source.first_route;
  brain.routes[slot] = grown;
  runtime.committed_route_slots[i] = slot;
  __threadfence();
  source.first_route = slot;
  if (source.route_count < brain.topology.max_resident_fanout)
    ++source.route_count;
  // #1179: see unlink_route_from_source. A route grown onto a packed source
  // after its panel was installed is invisible to that panel; this is what
  // makes the panel's guard notice and fail the source closed to canonical
  // execution instead of silently competing over a stale candidate set.
  ++source.source_revision;
  meta.live = 1u;
  meta.logical_epoch = *brain.topology.epoch + 1u;
  meta.logical_rank = rank;

  // The physical object now exists: publish the reservation.
  device_commit_pool_units(brain.resource_ecology,
                           DirectResourcePoolKind::explicit_interaction, 1u);

  // Retention state is indexed by slot, but describes a (slot, generation) route.
  // Recruiting into a recycled slot must not inherit the previous occupant's
  // flags -- a stale `damaged` bit would otherwise make maintenance "repair" a
  // route that was never damaged, writing a dead generation's conductance into a
  // brand-new one.
  if (brain.retention_bank != nullptr) {
    DirectRetentionState fresh{};
    fresh.logical_source = proposal.source;
    fresh.logical_slot = slot;
    fresh.logical_generation = meta.generation;
    fresh.last_confirmed_conductance_q16 = proposal.conductance_q16;
    brain.retention_bank[slot] = fresh;
  }
  if (brain.minimal_retention_bank != nullptr) {
    DirectMinimalRetentionState fresh_min{};
    brain.minimal_retention_bank[slot] = fresh_min;
  }
  atomicAdd(&brain.topology.incoming_degree[proposal.target], 1u);
  atomicAdd(brain.live_route_count, 1u);
  if (proposal.credit_q16 != 0)
    atomicAdd(&counters->credit_commits, 1u);
  if (proposal.kind == DirectTopologyProposalKind::repair)
    atomicAdd(&counters->structural_repairs, 1u);
  else
    atomicAdd(&counters->structural_growth, 1u);
  if (proposal.implicit_family != kInvalidIndex) {
    atomicAdd(&counters->implicit_materializations, 1u);
  }
}

// 0X1-1175: `proposal_work` is the host-side launch bound, not the number of
// proposals that exist. `reset_direct_topology_proposals` memsets
// `proposals_sorted` to zero, and zero is `implicit_family == 0`, not
// `kInvalidIndex` -- so every slot of the dead tail used to pass the guard below
// and run the full 64-slot open-addressing probe for the key of the phantom
// triple (family 0, source 0, slot 0). That is why this kernel cost 9.81 us with
// 0.7% variance across 126 launches while every other kernel over the same index
// range cost ~1.2 us: the work was set by the bound, not by the epoch.
//
// `live_bound` is the device-resident live-proposal count produced by
// `compact_live_proposals_kernel`. The live proposals occupy exactly the sorted
// positions [0, live) -- the same #1205 invariant that made the compacted sort
// bit-identical -- so restricting this kernel to that prefix visits every real
// proposal and no phantom one. It is nullptr on the legacy radix path, where no
// live count is materialised and `proposal_work` remains the bound.
// gh #1244: also folds the former standalone finish_growth_kernel<<<1,32>>>
// (thread (0,0) below). commit_growth_kernel has already fully retired in this
// stream by the time any thread here runs -- CUDA guarantees in-stream kernel
// execution order -- so every read of free_count that commit_growth_kernel made
// has already happened before this kernel's write to it. Neither the epoch/
// counters increment nor free_count/constructor_reserve alias anything this
// kernel's per-thread body touches (proposals_sorted, implicit exceptions,
// growth_committed_flags), so the two are order-independent and safe to fuse.
__global__ void finalize_implicit_materialization_requests_kernel(
    DirectBrainV01 brain, DirectTopologyRuntime runtime, std::uint32_t proposal_work,
    const std::uint32_t* live_bound, AdultCounters* counters) {
  if (blockIdx.x == 0u && threadIdx.x == 0u) {
    const std::uint32_t actual_grown = *runtime.growth_committed_total;
    const std::uint32_t free_count_before = *brain.topology.free_count;
    const std::uint32_t consumed =
        actual_grown < free_count_before ? actual_grown : free_count_before;
    *brain.topology.free_count = free_count_before - consumed;
    if (brain.development != nullptr) {
      if (brain.development->constructor_reserve >= consumed)
        brain.development->constructor_reserve -= consumed;
      else
        brain.development->constructor_reserve = 0;
    }
    ++(*brain.topology.epoch);
    ++counters->topology_epochs;
  }
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  std::uint32_t bound = proposal_work;
  if (live_bound != nullptr) {
    const std::uint32_t live = *live_bound;
    bound = live < proposal_work ? live : proposal_work;
  }
  if (i >= bound)
    return;
  const DirectTopologyProposal proposal = runtime.proposals_sorted[i];
  if (proposal.implicit_family == kInvalidIndex || brain.implicit.exception_capacity == 0u)
    return;
  const std::uint64_t key = direct_implicit_exception_key(
      proposal.implicit_family, proposal.source, proposal.implicit_slot);
  const std::uint32_t bucket =
      direct_implicit_exception_bucket(key, brain.implicit.exception_capacity);
  for (std::uint32_t probe = 0; probe < 64u; ++probe) {
    const std::uint32_t exc_slot =
        (bucket + probe) & (brain.implicit.exception_capacity - 1u);
    DirectImplicitException& entry = brain.implicit.exceptions[exc_slot];
    if (entry.key == key) {
      if (runtime.growth_committed_flags[i] != 0u) {
        atomicAnd(&entry.flags, ~kImplicitFlagPendingMaterialize);
        atomicOr(&entry.flags, kImplicitFlagMaterialized);
      } else {
        // Proposal failed arbitration / capacity -> clear pending flag so it can retry on later lived participation!
        atomicAnd(&entry.flags, ~kImplicitFlagPendingMaterialize);
      }
      break;
    }
  }
}

}  // namespace

void initialize_direct_topology_state(DirectBrainV01* brain) {
  if (brain == nullptr || brain->routes == nullptr || brain->nodes == nullptr)
    throw std::invalid_argument("direct topology requires a born brain");
  DirectTopologyPersistentState topology{};
  topology.route_capacity = brain->route_capacity;
  topology.max_resident_fanout = kMaximumResidentFanout;
  topology_cuda(cudaMalloc(&topology.slot_meta, sizeof(DirectRouteSlotMeta) * brain->route_capacity),
                "allocate route slot metadata");
  topology_cuda(cudaMalloc(&topology.free_slots, sizeof(std::uint32_t) * brain->route_capacity),
                "allocate route free slots");
  topology_cuda(cudaMalloc(&topology.free_count, sizeof(std::uint32_t)),
                "allocate route free count");
  topology_cuda(cudaMalloc(&topology.incoming_degree, sizeof(std::uint32_t) * brain->node_count),
                "allocate incoming degree");
  topology_cuda(cudaMalloc(&topology.epoch, sizeof(std::uint64_t)), "allocate topology epoch");
  topology_cuda(cudaMemset(topology.free_slots, 0xff,
                           sizeof(std::uint32_t) * brain->route_capacity),
                "clear route free slots");
  topology_cuda(cudaMemset(topology.incoming_degree, 0,
                           sizeof(std::uint32_t) * brain->node_count),
                "clear incoming degree");
  std::uint32_t free_count = brain->route_capacity - brain->route_count;
  if (brain->development != nullptr) {
    ResidentDevelopmentState dev{};
    topology_cuda(cudaMemcpy(&dev, brain->development, sizeof(dev), cudaMemcpyDeviceToHost),
                  "read development state for topology initialization");
    free_count = std::min<std::uint32_t>(free_count, static_cast<std::uint32_t>(dev.constructor_reserve));
  }
  const std::uint64_t epoch = 0u;
  topology_cuda(cudaMemcpy(topology.free_count, &free_count, sizeof(free_count), cudaMemcpyHostToDevice),
                "initialize route free count");
  topology_cuda(cudaMemcpy(topology.epoch, &epoch, sizeof(epoch), cudaMemcpyHostToDevice),
                "initialize topology epoch");
  brain->topology = topology;
  initialize_topology_slots_kernel<<<grid_for_topology(brain->route_capacity, 128u), 128u>>>(*brain);
  topology_cuda(cudaGetLastError(), "initialize topology slots");
  topology_cuda(cudaDeviceSynchronize(), "finish topology initialization");
}

void destroy_direct_topology_state(DirectBrainV01* brain) {
  if (brain == nullptr)
    return;
  cudaFree(brain->topology.slot_meta);
  cudaFree(brain->topology.free_slots);
  cudaFree(brain->topology.free_count);
  cudaFree(brain->topology.incoming_degree);
  cudaFree(brain->topology.epoch);
  brain->topology = DirectTopologyPersistentState{};
}

DirectTopologyRuntime* create_direct_topology_runtime(const DirectBrainV01& brain,
                                                      std::uint32_t proposal_capacity) {
  if (proposal_capacity == 0u)
    throw std::invalid_argument("topology proposal capacity must be non-zero");
  auto* runtime = new DirectTopologyRuntime{};
  runtime->view.proposal_capacity = proposal_capacity;
  // gh #1245: a *blocking* (not cudaStreamNonBlocking) stream. Every other
  // launch in this codebase runs on the legacy default stream, and a
  // blocking stream implicitly synchronizes with it in both directions, so
  // capturing/replaying the Phase 2-5 graph here preserves the exact
  // issue-order serialization the surrounding default-stream launches
  // (Phase 1 before, install_committed_context_routes_kernel after) already
  // rely on, with no explicit event/sync code needed at any call site.
  topology_cuda(cudaStreamCreate(&runtime->topology_epoch_graph_stream),
                "create topology epoch graph stream");
  // #1260 conditional-sort state is allocated lazily only if this runtime ever
  // reaches an ambiguous host bound (> kFastSortSpan).
  topology_cuda(cudaMalloc(&runtime->view.proposals,
                           sizeof(DirectTopologyProposal) * proposal_capacity),
                "allocate topology proposals");
  topology_cuda(cudaMalloc(&runtime->proposals_sorted,
                           sizeof(DirectTopologyProposal) * proposal_capacity),
                "allocate sorted topology proposals");
  topology_cuda(cudaMalloc(&runtime->sort_keys_in, sizeof(std::uint64_t) * proposal_capacity),
                "allocate sort keys in");
  topology_cuda(cudaMalloc(&runtime->sort_keys_out, sizeof(std::uint64_t) * proposal_capacity),
                "allocate sort keys out");
  topology_cuda(cudaMalloc(&runtime->sort_values_in, sizeof(std::uint32_t) * proposal_capacity),
                "allocate sort values in");
  topology_cuda(cudaMalloc(&runtime->sort_values_out, sizeof(std::uint32_t) * proposal_capacity),
                "allocate sort values out");
  topology_cuda(cudaMalloc(&runtime->growth_source_winner,
                           sizeof(std::uint32_t) * brain.node_count),
                "allocate growth source winners");
  topology_cuda(cudaMalloc(&runtime->growth_target_winner,
                           sizeof(std::uint32_t) * brain.node_count),
                "allocate growth target winners");
  topology_cuda(cudaMalloc(&runtime->retract_source_winner,
                           sizeof(std::uint32_t) * brain.node_count),
                "allocate retract source winners");
  topology_cuda(cudaMalloc(&runtime->retract_route_winner,
                           sizeof(std::uint32_t) * brain.route_capacity),
                "allocate retract route winners");
  topology_cuda(cudaMalloc(&runtime->growth_flags, sizeof(std::uint32_t) * proposal_capacity),
                "allocate growth flags");
  topology_cuda(cudaMalloc(&runtime->retract_flags, sizeof(std::uint32_t) * proposal_capacity),
                "allocate retract flags");
  topology_cuda(cudaMalloc(&runtime->committed_route_slots,
                           sizeof(std::uint32_t) * proposal_capacity),
                "allocate committed topology slots");
  topology_cuda(cudaMalloc(&runtime->retract_committed_flags, sizeof(std::uint32_t) * proposal_capacity),
                "allocate retract committed flags");
  topology_cuda(cudaMalloc(&runtime->retract_committed_ranks, sizeof(std::uint32_t) * proposal_capacity),
                "allocate retract committed ranks");
  topology_cuda(cudaMalloc(&runtime->retract_committed_total, sizeof(std::uint32_t)),
                "allocate retract committed total");
  topology_cuda(cudaMalloc(&runtime->growth_committed_flags, sizeof(std::uint32_t) * proposal_capacity),
                "allocate growth committed flags");
  topology_cuda(cudaMalloc(&runtime->growth_committed_ranks, sizeof(std::uint32_t) * proposal_capacity),
                "allocate growth committed ranks");
  topology_cuda(cudaMalloc(&runtime->growth_committed_total, sizeof(std::uint32_t)),
                "allocate growth committed total");
  topology_cuda(cudaMalloc(&runtime->free_count_snapshot, sizeof(std::uint32_t)),
                "allocate free count snapshot");
  topology_cuda(cudaMalloc(&runtime->live_indices, sizeof(std::uint32_t) * proposal_capacity),
                "allocate live proposal indices");
  topology_cuda(cudaMalloc(&runtime->live_count, sizeof(std::uint32_t)),
                "allocate live proposal count");
  topology_cuda(cudaMalloc(&runtime->view.proposal_high_water, sizeof(std::uint32_t)),
                "allocate proposal high water");
  topology_cuda(cudaMemcpy(runtime->view.proposal_high_water, &proposal_capacity,
                           sizeof(std::uint32_t), cudaMemcpyHostToDevice),
                "initialize proposal high water to capacity");

  // 0X1-1175: sizing query for the CUB fallback scan. The pair named here is
  // arbitrary among the surviving `uint32_t` flag/rank arrays -- CUB's
  // `nullptr` temp-storage query never dereferences either pointer and the
  // byte count is a pure function of the item type and `proposal_capacity` --
  // but it must name arrays that still exist, so it uses the growth-commit
  // pair that `launch_rank_scan` actually runs the fallback over.
  std::size_t scan_bytes = 0u;
  cub::DeviceScan::ExclusiveSum(nullptr, scan_bytes, runtime->growth_committed_flags,
                                runtime->growth_committed_ranks, proposal_capacity);
  runtime->scan_storage_bytes = scan_bytes;
  topology_cuda(cudaMalloc(&runtime->scan_storage, scan_bytes), "allocate topology scan storage");

  std::size_t sort_bytes = 0u;
  cub::DeviceRadixSort::SortPairs(nullptr, sort_bytes, runtime->sort_keys_in, runtime->sort_keys_out,
                                  runtime->sort_values_in, runtime->sort_values_out, proposal_capacity);
  runtime->sort_storage_bytes = sort_bytes;
  topology_cuda(cudaMalloc(&runtime->sort_storage, sort_bytes), "allocate topology sort storage");

  reset_direct_topology_proposals(runtime, proposal_capacity);
  return runtime;
}

void destroy_direct_topology_runtime(DirectTopologyRuntime* runtime) {
  if (runtime == nullptr)
    return;
  cudaFree(runtime->view.proposals);
  cudaFree(runtime->proposals_sorted);
  cudaFree(runtime->sort_keys_in);
  cudaFree(runtime->sort_keys_out);
  cudaFree(runtime->sort_values_in);
  cudaFree(runtime->sort_values_out);
  cudaFree(runtime->growth_source_winner);
  cudaFree(runtime->growth_target_winner);
  cudaFree(runtime->retract_source_winner);
  cudaFree(runtime->retract_route_winner);
  cudaFree(runtime->growth_flags);
  cudaFree(runtime->retract_flags);
  cudaFree(runtime->committed_route_slots);
  cudaFree(runtime->retract_committed_flags);
  cudaFree(runtime->retract_committed_ranks);
  cudaFree(runtime->retract_committed_total);
  cudaFree(runtime->growth_committed_flags);
  cudaFree(runtime->growth_committed_ranks);
  cudaFree(runtime->growth_committed_total);
  cudaFree(runtime->free_count_snapshot);
  cudaFree(runtime->live_indices);
  cudaFree(runtime->live_count);
  cudaFree(runtime->view.proposal_high_water);
  cudaFree(runtime->scan_storage);
  cudaFree(runtime->sort_storage);
  if (runtime->topology_epoch_graph_exec != nullptr)
    cudaGraphExecDestroy(runtime->topology_epoch_graph_exec);
  if (runtime->topology_epoch_graph_template != nullptr)
    cudaGraphDestroy(runtime->topology_epoch_graph_template);
  if (runtime->conditional_sort_cache != nullptr) {
    for (DirectTopologyConditionalSortGraph& cached : runtime->conditional_sort_cache->graphs) {
      if (cached.exec != nullptr)
        cudaGraphExecDestroy(cached.exec);
      if (cached.graph != nullptr)
        cudaGraphDestroy(cached.graph);
    }
    cudaFree(runtime->conditional_sort_cache->sort_control);
    delete runtime->conditional_sort_cache;
  }
  if (runtime->topology_epoch_graph_stream != nullptr)
    cudaStreamDestroy(runtime->topology_epoch_graph_stream);
  delete runtime;
}

namespace {

// 0X1-#1247: the nine cudaMemsetAsync host dispatches this function used to
// issue every topology epoch (called up to 4x/step -- see
// direct_representation_state_owner.cu and direct_adult_legacy_oracle.cu)
// collapse into one kernel launch. Every cleared byte is unchanged: the
// seven uint32 arrays and two proposal-struct arrays still zero exactly
// `proposal_work` elements, `committed_route_slots` keeps its 0xff sentinel
// pattern, and the two committed-total scalars are still cleared
// unconditionally (matching the old fixed-size memsets, which never scaled
// with proposal_work either).
//
// 0X1-#1246/#1260: `live_count` rides along as a fourth scalar cleared by the
// same single-thread block. Production now always consumes that device-live
// value, so this is the one canonical reset before every topology epoch.
__global__ void clear_topology_epoch_reset_state_kernel(
    DirectTopologyProposal* proposals, DirectTopologyProposal* proposals_sorted,
    std::uint32_t* growth_flags, std::uint32_t* retract_flags,
    std::uint32_t* committed_route_slots, std::uint32_t* retract_committed_flags,
    std::uint32_t* retract_committed_total, std::uint32_t* growth_committed_flags,
    std::uint32_t* growth_committed_total, std::uint32_t* live_count,
    const std::uint32_t* proposal_high_water, std::uint32_t hard_capacity) {
  const std::uint32_t high = proposal_high_water != nullptr
                                 ? min(*proposal_high_water, hard_capacity)
                                 : hard_capacity;
  const std::uint32_t stride = blockDim.x * gridDim.x;
  for (std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < high; i += stride) {
    proposals[i] = DirectTopologyProposal{};
    proposals_sorted[i] = DirectTopologyProposal{};
    growth_flags[i] = 0u;
    retract_flags[i] = 0u;
    committed_route_slots[i] = 0xffffffffu;
    retract_committed_flags[i] = 0u;
    growth_committed_flags[i] = 0u;
  }
  if (blockIdx.x == 0u && threadIdx.x == 0u) {
    *retract_committed_total = 0u;
    *growth_committed_total = 0u;
    *live_count = 0u;
  }
}

__global__ void reset_topology_proposal_high_water_kernel(
    std::uint32_t* proposal_high_water) {
  if (proposal_high_water != nullptr) {
    *proposal_high_water = 0u;
  }
}

}  // namespace

void reset_direct_topology_proposals(DirectTopologyRuntime* runtime,
                                     std::uint32_t proposal_work,
                                     cudaStream_t stream) {
  if (runtime == nullptr)
    return;
  const std::uint32_t capacity = runtime->view.proposal_capacity;
  proposal_work = std::min(proposal_work, capacity);
  constexpr std::uint32_t kResetBlock = 128u;
  const std::uint32_t grid = grid_for_topology(proposal_work, kResetBlock);
  clear_topology_epoch_reset_state_kernel<<<grid, kResetBlock, 0, stream>>>(
      runtime->view.proposals, runtime->proposals_sorted, runtime->growth_flags,
      runtime->retract_flags, runtime->committed_route_slots, runtime->retract_committed_flags,
      runtime->retract_committed_total, runtime->growth_committed_flags,
      runtime->growth_committed_total, runtime->live_count,
      runtime->view.proposal_high_water, capacity);
  reset_topology_proposal_high_water_kernel<<<1, 1, 0, stream>>>(runtime->view.proposal_high_water);
  topology_cuda(cudaGetLastError(), "launch topology epoch reset state clear");
}

namespace {

// Historical nine-pass LSB-first stable radix chain. It remains the canonical
// dense-sort primitive and the independent test oracle. Production may execute
// it only from #1260's device-selected conditional branch; a host proposal bound
// no longer selects it directly. `active_work` fences reusable graph padding and
// is null for the historical oracle.
void launch_legacy_radix_proposal_sort(DirectTopologyRuntime* runtime,
                                       std::uint32_t proposal_work, std::uint32_t grid,
                                       std::uint32_t block_size, cudaStream_t stream,
                                       const std::uint32_t* active_work = nullptr) {
  init_sort_indices_kernel<<<grid, block_size, 0, stream>>>(runtime->sort_values_in, proposal_work);

  // Pass 1 (LSB): eligibility_root
  gather_pass1_kernel<<<grid, block_size, 0, stream>>>(runtime->view.proposals, runtime->sort_values_in,
                                            runtime->sort_keys_in, proposal_work);
  cub::DeviceRadixSort::SortPairs(runtime->sort_storage, runtime->sort_storage_bytes,
                                  runtime->sort_keys_in, runtime->sort_keys_out,
                                  runtime->sort_values_in, runtime->sort_values_out,
                                  proposal_work, 0, sizeof(std::uint64_t) * 8, stream);

  // Pass 2: eligibility_history
  gather_pass2_kernel<<<grid, block_size, 0, stream>>>(runtime->view.proposals, runtime->sort_values_out,
                                            runtime->sort_keys_in, proposal_work);
  cub::DeviceRadixSort::SortPairs(runtime->sort_storage, runtime->sort_storage_bytes,
                                  runtime->sort_keys_in, runtime->sort_keys_out,
                                  runtime->sort_values_out, runtime->sort_values_in,
                                  proposal_work, 0, sizeof(std::uint64_t) * 8, stream);

  // Pass 3: route_generation
  gather_pass3_kernel<<<grid, block_size, 0, stream>>>(runtime->view.proposals, runtime->sort_values_in,
                                            runtime->sort_keys_in, proposal_work);
  cub::DeviceRadixSort::SortPairs(runtime->sort_storage, runtime->sort_storage_bytes,
                                  runtime->sort_keys_in, runtime->sort_keys_out,
                                  runtime->sort_values_in, runtime->sort_values_out,
                                  proposal_work, 0, sizeof(std::uint64_t) * 8, stream);

  // Pass 4: context_signature
  gather_pass4_kernel<<<grid, block_size, 0, stream>>>(runtime->view.proposals, runtime->sort_values_out,
                                            runtime->sort_keys_in, proposal_work);
  cub::DeviceRadixSort::SortPairs(runtime->sort_storage, runtime->sort_storage_bytes,
                                  runtime->sort_keys_in, runtime->sort_keys_out,
                                  runtime->sort_values_out, runtime->sort_values_in,
                                  proposal_work, 0, sizeof(std::uint64_t) * 8, stream);

  // Pass 5: route_slot (hi32) | implicit_slot (lo32)
  gather_pass5_kernel<<<grid, block_size, 0, stream>>>(runtime->view.proposals, runtime->sort_values_in,
                                            runtime->sort_keys_in, proposal_work);
  cub::DeviceRadixSort::SortPairs(runtime->sort_storage, runtime->sort_storage_bytes,
                                  runtime->sort_keys_in, runtime->sort_keys_out,
                                  runtime->sort_values_in, runtime->sort_values_out,
                                  proposal_work, 0, sizeof(std::uint64_t) * 8, stream);

  // Pass 6: implicit_family (hi32) | learned_output_word (lo32)
  gather_pass6_kernel<<<grid, block_size, 0, stream>>>(runtime->view.proposals, runtime->sort_values_out,
                                            runtime->sort_keys_in, proposal_work);
  cub::DeviceRadixSort::SortPairs(runtime->sort_storage, runtime->sort_storage_bytes,
                                  runtime->sort_keys_in, runtime->sort_keys_out,
                                  runtime->sort_values_out, runtime->sort_values_in,
                                  proposal_work, 0, sizeof(std::uint64_t) * 8, stream);

  // Pass 7: route_flags (hi32) | target (lo32)
  gather_pass7_kernel<<<grid, block_size, 0, stream>>>(runtime->view.proposals, runtime->sort_values_in,
                                            runtime->sort_keys_in, proposal_work);
  cub::DeviceRadixSort::SortPairs(runtime->sort_storage, runtime->sort_storage_bytes,
                                  runtime->sort_keys_in, runtime->sort_keys_out,
                                  runtime->sort_values_in, runtime->sort_values_out,
                                  proposal_work, 0, sizeof(std::uint64_t) * 8, stream);

  // Pass 8: kind (hi32) | source (lo32)
  gather_pass8_kernel<<<grid, block_size, 0, stream>>>(runtime->view.proposals, runtime->sort_values_out,
                                            runtime->sort_keys_in, proposal_work);
  cub::DeviceRadixSort::SortPairs(runtime->sort_storage, runtime->sort_storage_bytes,
                                  runtime->sort_keys_in, runtime->sort_keys_out,
                                  runtime->sort_values_out, runtime->sort_values_in,
                                  proposal_work, 0, sizeof(std::uint64_t) * 8, stream);

  // Pass 9 (MSB): priority_q16 DESC
  gather_pass9_kernel<<<grid, block_size, 0, stream>>>(runtime->view.proposals, runtime->sort_values_in,
                                            runtime->sort_keys_in, proposal_work, active_work);
  cub::DeviceRadixSort::SortPairs(runtime->sort_storage, runtime->sort_storage_bytes,
                                  runtime->sort_keys_in, runtime->sort_keys_out,
                                  runtime->sort_values_in, runtime->sort_values_out,
                                  proposal_work, 0, sizeof(std::uint64_t) * 8, stream);

  gather_sorted_proposals_kernel<<<grid, block_size, 0, stream>>>(*runtime, proposal_work);
}

// 0X1-1175/#1260: production always supplies the device-resident live count and
// therefore uses the single live-sized block scan. The CUB branch survives only
// for the explicit historical test oracle below.
void launch_rank_scan(DirectTopologyRuntime* runtime, const std::uint32_t* flags,
                      std::uint32_t* ranks, std::uint32_t* total,
                      const std::uint32_t* live_bound, std::uint32_t proposal_work,
                      cudaStream_t stream) {
  if (live_bound != nullptr) {
    live_exclusive_scan_kernel<<<1, kLiveScanBlock, 0, stream>>>(flags, ranks, live_bound,
                                                                  proposal_work, total);
    return;
  }
  cub::DeviceScan::ExclusiveSum(runtime->scan_storage, runtime->scan_storage_bytes, flags, ranks,
                                proposal_work, stream);
  finalize_scan_totals_kernel<<<1, 32, 0, stream>>>(flags, ranks, proposal_work, total);
}

// Shared Phase 2-5 sequence for #1260 conditional bodies and the independent
// radix oracle. `grid` names the scheduled extent; production supplies the
// device-resident live count as the semantic bound on both conditional bodies.
// Only the historical oracle leaves `live_bound` null.
void launch_topology_post_sort_sequence(
    DirectBrainV01* brain, DirectTopologyRuntime* runtime, std::uint32_t proposal_work,
    AdultCounters* counters, std::uint32_t grid, std::uint32_t block_size,
    const std::uint32_t* live_bound, const std::uint32_t* free_count_ptr,
    std::uint32_t* free_count_snapshot_ptr, cudaStream_t stream) {
  initialize_touched_winners_kernel<<<grid, block_size, 0, stream>>>(*brain, *runtime,
                                                                     proposal_work, live_bound);
  arbitrate_topology_kernel<<<grid, block_size, 0, stream>>>(*brain, *runtime, proposal_work,
                                                              live_bound);
  mark_topology_admission_kernel<<<grid, block_size, 0, stream>>>(*brain, *runtime, proposal_work,
                                                                   live_bound, counters);
  commit_retractions_kernel<<<grid, block_size, 0, stream>>>(*brain, *runtime, proposal_work,
                                                              live_bound, counters, free_count_ptr,
                                                              free_count_snapshot_ptr);
  launch_rank_scan(runtime, runtime->retract_committed_flags, runtime->retract_committed_ranks,
                   runtime->retract_committed_total, live_bound, proposal_work, stream);
  publish_freed_slots_kernel<<<grid, block_size, 0, stream>>>(*brain, *runtime, proposal_work,
                                                               live_bound, free_count_snapshot_ptr);
  validate_growth_candidates_kernel<<<grid, block_size, 0, stream>>>(*brain, *runtime,
                                                                     proposal_work, live_bound);
  launch_rank_scan(runtime, runtime->growth_committed_flags, runtime->growth_committed_ranks,
                   runtime->growth_committed_total, live_bound, proposal_work, stream);
  commit_growth_kernel<<<grid, block_size, 0, stream>>>(*brain, *runtime, proposal_work,
                                                        live_bound, counters);
  finalize_implicit_materialization_requests_kernel<<<grid, block_size, 0, stream>>>(
      *brain, *runtime, proposal_work, live_bound, counters);
}

// gh #1245 identified each captured node by querying the stream's dependency
// frontier right after its launch; that query grew an edge-data argument in
// the CUDA 12.x -> 13 transition, and against an older libcudart the count is
// written to the caller's extra nullptr slot instead of num_deps, so every
// capture failed with "expected single-node frontier" (the #1624 ABI-skew
// class). Every degree and edge query gained the same extra parameter, so
// identification reads each node's kernel function instead -- the one graph
// query whose signature never drifted -- and takes the unique node bound to
// compact_live_proposals_kernel, the only node ever refreshed (#1328).
cudaGraphNode_t topology_epoch_compact_node(cudaGraph_t graph) {
  std::size_t node_count = 0u;
  topology_cuda(cudaGraphGetNodes(graph, nullptr, &node_count),
                "size topology epoch phase2-5 capture graph");
  cudaGraphNode_t nodes[DirectTopologyRuntime::kTopologyEpochGraphNodeCount] = {};
  std::size_t fetched = DirectTopologyRuntime::kTopologyEpochGraphNodeCount;
  if (node_count != fetched)
    throw std::runtime_error("topology epoch phase2-5 capture: expected thirteen-kernel chain");
  topology_cuda(cudaGraphGetNodes(graph, nodes, &fetched),
                "read topology epoch phase2-5 capture graph");
  cudaGraphNode_t compact = nullptr;
  int matches = 0;
  for (std::size_t i = 0; i < fetched; ++i) {
    cudaKernelNodeParams params{};
    topology_cuda(cudaGraphKernelNodeGetParams(nodes[i], &params),
                  "read topology epoch capture node function");
    if (params.func != reinterpret_cast<void*>(&compact_live_proposals_kernel))
      continue;
    if (matches == 0)
      compact = nodes[i];
    ++matches;
  }
  if (matches != 1)
    throw std::runtime_error("topology epoch phase2-5 capture: expected one compact node");
  return compact;
}

// gh #1245: Phase 2-5 (arbitration through implicit-materialization commit)
// on the live-bound fast-sort path (Phase 1's compact/sort/gather plus
// #1245's original Phase 2-5 sequence), issued via CUDA Graph replay instead
// of thirteen separate cudaLaunchKernel dispatches. #1245's measurement:
// 98.7% of launches in this sequence run a single block on an 80-SM device,
// i.e. these are sequence points, not parallel work, so the cost this
// removes is host dispatch (~2.2us/launch), not GPU time. Every barrier in
// this sequence -- the grid-wide election between initialize_touched_winners_kernel,
// arbitrate_topology_kernel and mark_topology_admission_kernel (#1242's
// diary), and Phase 1's own compact-then-sort-then-gather chain over
// live_indices/live_count/sort_values_out -- is unchanged: a captured
// graph's node-to-node execution order is the same stream-order dependency
// chain a live launch sequence already has, so this is not a fusion and
// introduces no new race.
//
// proposal_work varies almost every call across launch_direct_topology_epoch's
// four call sites, so the graph is captured once per runtime instance. gh
// #1328: only node 0 (compact_live_proposals_kernel) reads proposal_work as a
// direct bound and must see the real per-call value every time, so only it is
// refreshed via cudaGraphExecKernelNodeSetParams before replay. Nodes 1-12 are
// captured once against runtime->view.proposal_capacity and never refreshed --
// see the capture-branch comment below for why that is exact, not approximate.
void launch_topology_epoch_fast_sort_captured(DirectBrainV01* brain, DirectTopologyRuntime* runtime,
                                              std::uint32_t proposal_work, AdultCounters* counters,
                                              std::uint32_t grid, std::uint32_t block_size,
                                              const std::uint32_t* live_bound,
                                              const std::uint32_t* free_count_ptr,
                                              std::uint32_t* free_count_snapshot_ptr) {
  cudaStream_t stream = runtime->topology_epoch_graph_stream;

  if (runtime->topology_epoch_graph_exec == nullptr) {
    // gh #1328: nodes 1-12 all read the live-bound prefix through `live_bound`
    // (runtime->live_count) via `bound = min(*live_bound, X)`; *live_bound <=
    // every tick's real proposal_work <= runtime->view.proposal_capacity always
    // holds, so X can be the fixed capacity instead of the real per-call
    // proposal_work without changing which indices any node processes. Capturing
    // them once against capacity-sized grids/args means they never need
    // cudaGraphExecKernelNodeSetParams again -- only node 0 (compact_live_proposals_kernel,
    // whose scan bound must match the current reset/writer span) still needs the
    // real per-call proposal_work and refreshes below.
    // ponytail: block_size is assumed constant across this runtime's calls (all
    // 4 call sites pass either a fixed constant or the same default); if that
    // ever stops holding, capacity_grid below goes stale for nodes 1-12 -- add a
    // block_size-changed assertion at that point, not before.
    const std::uint32_t capacity = runtime->view.proposal_capacity;
    const std::uint32_t capacity_grid = grid_for_topology(capacity, block_size);
    const std::uint32_t capacity_sort_runs =
        std::min(kFastSortChunks,
                 std::max(1u, (capacity + kFastSortCapacity - 1u) / kFastSortCapacity));

    topology_cuda(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal),
                  "begin topology epoch fast-sort capture");

    compact_live_proposals_kernel<<<grid, block_size, 0, stream>>>(
        runtime->view.proposals, runtime->view.proposal_high_water, capacity,
        runtime->live_indices, runtime->live_count, nullptr);
    sort_live_proposals_kernel<<<capacity_sort_runs, kFastSortBlock, 0, stream>>>(
        runtime->view.proposals, runtime->live_indices, runtime->live_count,
        runtime->sort_values_out);
    gather_live_sorted_proposals_kernel<<<capacity_grid, block_size, 0, stream>>>(*runtime,
                                                                         runtime->live_count);
    initialize_touched_winners_kernel<<<capacity_grid, block_size, 0, stream>>>(*brain, *runtime,
                                                                        capacity, live_bound);
    arbitrate_topology_kernel<<<capacity_grid, block_size, 0, stream>>>(*brain, *runtime, capacity,
                                                               live_bound);
    mark_topology_admission_kernel<<<capacity_grid, block_size, 0, stream>>>(*brain, *runtime, capacity,
                                                                    live_bound, counters);
    commit_retractions_kernel<<<capacity_grid, block_size, 0, stream>>>(
        *brain, *runtime, capacity, live_bound, counters, free_count_ptr,
        free_count_snapshot_ptr);
    live_exclusive_scan_kernel<<<1, kLiveScanBlock, 0, stream>>>(
        runtime->retract_committed_flags, runtime->retract_committed_ranks, live_bound,
        capacity, runtime->retract_committed_total);
    publish_freed_slots_kernel<<<capacity_grid, block_size, 0, stream>>>(*brain, *runtime, capacity,
                                                                live_bound, free_count_snapshot_ptr);
    validate_growth_candidates_kernel<<<capacity_grid, block_size, 0, stream>>>(*brain, *runtime,
                                                                       capacity, live_bound);
    live_exclusive_scan_kernel<<<1, kLiveScanBlock, 0, stream>>>(
        runtime->growth_committed_flags, runtime->growth_committed_ranks, live_bound, capacity,
        runtime->growth_committed_total);
    commit_growth_kernel<<<capacity_grid, block_size, 0, stream>>>(*brain, *runtime, capacity,
                                                          live_bound, counters);
    finalize_implicit_materialization_requests_kernel<<<capacity_grid, block_size, 0, stream>>>(
        *brain, *runtime, capacity, live_bound, counters);

    topology_cuda(cudaStreamEndCapture(stream, &runtime->topology_epoch_graph_template),
                  "end topology epoch fast-sort capture");
    runtime->topology_epoch_graph_compact_node =
        topology_epoch_compact_node(runtime->topology_epoch_graph_template);
    topology_cuda(cudaGraphInstantiate(&runtime->topology_epoch_graph_exec,
                                       runtime->topology_epoch_graph_template, 0ull),
                  "instantiate topology epoch fast-sort graph");
  }

  cudaGraphExec_t exec = runtime->topology_epoch_graph_exec;
  const dim3 grid_dim(grid);
  const dim3 block_dim(block_size);

  auto refresh = [&](cudaGraphNode_t node, void* func, dim3 g, dim3 b, void** args) {
    cudaKernelNodeParams params{};
    params.func = func;
    params.gridDim = g;
    params.blockDim = b;
    params.sharedMemBytes = 0u;
    params.kernelParams = args;
    params.extra = nullptr;
    topology_cuda(cudaGraphExecKernelNodeSetParams(exec, node, &params),
                  "refresh topology epoch fast-sort graph node");
  };

  // gh #1328/#1335: nodes 1-12 stay frozen at their capture-time (capacity-derived)
  // params for the runtime's lifetime -- see the capture-branch comment above.
  std::uint32_t compact_capacity = proposal_work;
  std::uint32_t* active_work = nullptr;
  void* compact_args[] = {&runtime->view.proposals, &runtime->view.proposal_high_water,
                          &compact_capacity, &runtime->live_indices,
                          &runtime->live_count, &active_work};
  refresh(runtime->topology_epoch_graph_compact_node,
          reinterpret_cast<void*>(&compact_live_proposals_kernel), grid_dim, block_dim,
          compact_args);

  topology_cuda(cudaGraphLaunch(exec, stream), "launch topology epoch fast-sort graph");
}

#if CUDART_VERSION >= 12040
DirectTopologyConditionalSortGraph* conditional_sort_graph_for(
    DirectBrainV01* brain, DirectTopologyRuntime* runtime, std::uint32_t proposal_work,
    AdultCounters* counters, std::uint32_t block_size) {
  if (runtime->conditional_sort_cache == nullptr) {
    runtime->conditional_sort_cache = new DirectTopologyConditionalSortCache{};
    topology_cuda(cudaMalloc(&runtime->conditional_sort_cache->sort_control,
                             sizeof(std::uint32_t) * 2u),
                  "allocate topology conditional sort control");
    topology_cuda(cudaMemset(runtime->conditional_sort_cache->sort_control, 0,
                             sizeof(std::uint32_t) * 2u),
                  "clear topology conditional sort control");
  }
  DirectTopologyConditionalSortCache* cache = runtime->conditional_sort_cache;
  const std::uint32_t sort_extent =
      conditional_sort_extent(proposal_work, runtime->view.proposal_capacity);
  for (DirectTopologyConditionalSortGraph& cached : cache->graphs) {
    if (cached.sort_extent == sort_extent && cached.block_size == block_size)
      return &cached;
  }

  DirectTopologyConditionalSortGraph built{};
  built.sort_extent = sort_extent;
  built.block_size = block_size;
  topology_cuda(cudaGraphCreate(&built.graph, 0u), "create topology conditional sort graph");

  cudaGraphConditionalHandle sort_handle{};
  topology_cuda(cudaGraphConditionalHandleCreate(&sort_handle, built.graph, 0u,
                                                  cudaGraphCondAssignDefault),
                "create topology sort conditional handle");

  const DirectTopologyProposal* proposals = runtime->view.proposals;
  const std::uint32_t* proposal_high_water = runtime->view.proposal_high_water;
  std::uint32_t hard_capacity = sort_extent;
  std::uint32_t* live_indices = runtime->live_indices;
  std::uint32_t* live_count = runtime->live_count;
  std::uint32_t* active_work = cache->sort_control;
  void* compact_args[] = {&proposals, &proposal_high_water, &hard_capacity,
                          &live_indices, &live_count, &active_work};
  cudaKernelNodeParams compact_params{};
  compact_params.func = reinterpret_cast<void*>(&compact_live_proposals_kernel);
  compact_params.gridDim = dim3(grid_for_topology(sort_extent, block_size));
  compact_params.blockDim = dim3(block_size);
  compact_params.kernelParams = compact_args;
  topology_cuda(cudaGraphAddKernelNode(&built.compact_node, built.graph, nullptr, 0u,
                                       &compact_params),
                "add topology conditional compact node");

  const std::uint32_t* selector_live_count = runtime->live_count;
  std::uint32_t* path_receipt = cache->sort_control + 1u;
  void* selector_args[] = {&selector_live_count, &sort_handle, &path_receipt};
  cudaKernelNodeParams selector_params{};
  selector_params.func = reinterpret_cast<void*>(&select_topology_sort_path_kernel);
  selector_params.gridDim = dim3(1u);
  selector_params.blockDim = dim3(1u);
  selector_params.kernelParams = selector_args;
  cudaGraphNode_t selector_node{};
  topology_cuda(cudaGraphAddKernelNode(&selector_node, built.graph, &built.compact_node, 1u,
                                       &selector_params),
                "add topology conditional selector node");

  // One IF/ELSE node is sufficient: body 0 executes when the device predicate is
  // true (fast live set), body 1 when false (genuinely dense live set).
  cudaGraphNodeParams sort_cond_params{};
  sort_cond_params.type = cudaGraphNodeTypeConditional;
  sort_cond_params.conditional.handle = sort_handle;
  sort_cond_params.conditional.type = cudaGraphCondTypeIf;
  sort_cond_params.conditional.size = 2u;
  cudaGraphNode_t sort_cond_node{};
  topology_cuda(cudaGraphAddNode(&sort_cond_node, built.graph, &selector_node, nullptr, 1u,
                                 &sort_cond_params),
                "add topology sort conditional node");
  cudaGraph_t fast_body = sort_cond_params.conditional.phGraph_out[0];
  cudaGraph_t radix_body = sort_cond_params.conditional.phGraph_out[1];

  cudaStream_t capture_stream = runtime->topology_epoch_graph_stream;
  topology_cuda(cudaStreamBeginCaptureToGraph(capture_stream, fast_body, nullptr, nullptr, 0u,
                                               cudaStreamCaptureModeThreadLocal),
                "begin topology conditional fast body capture");
  sort_live_proposals_kernel<<<kFastSortChunks, kFastSortBlock, 0, capture_stream>>>(
      runtime->view.proposals, runtime->live_indices, runtime->live_count,
      runtime->sort_values_out);
  const std::uint32_t fast_grid = grid_for_topology(kFastSortSpan, block_size);
  gather_live_sorted_proposals_kernel<<<fast_grid, block_size, 0, capture_stream>>>(*runtime,
                                                                                    runtime->live_count);
  launch_topology_post_sort_sequence(brain, runtime, sort_extent, counters, fast_grid, block_size,
                                     runtime->live_count, brain->topology.free_count,
                                     runtime->free_count_snapshot, capture_stream);
  cudaGraph_t captured_fast_body = nullptr;
  topology_cuda(cudaStreamEndCapture(capture_stream, &captured_fast_body),
                "end topology conditional fast body capture");
  if (captured_fast_body != fast_body)
    throw std::runtime_error("topology conditional fast body capture returned a different graph");

  const std::uint32_t radix_grid = grid_for_topology(sort_extent, block_size);
  topology_cuda(cudaStreamBeginCaptureToGraph(capture_stream, radix_body, nullptr, nullptr, 0u,
                                               cudaStreamCaptureModeThreadLocal),
                "begin topology conditional radix body capture");
  launch_legacy_radix_proposal_sort(runtime, sort_extent, radix_grid, block_size, capture_stream,
                                    cache->sort_control);
  launch_topology_post_sort_sequence(brain, runtime, sort_extent, counters, radix_grid, block_size,
                                     runtime->live_count, brain->topology.free_count,
                                     runtime->free_count_snapshot, capture_stream);
  cudaGraph_t captured_radix_body = nullptr;
  topology_cuda(cudaStreamEndCapture(capture_stream, &captured_radix_body),
                "end topology conditional radix body capture");
  if (captured_radix_body != radix_body)
    throw std::runtime_error("topology conditional radix body capture returned a different graph");

  topology_cuda(cudaGraphInstantiate(&built.exec, built.graph, 0ull),
                "instantiate topology conditional sort graph");
  cache->graphs.push_back(built);
  return &cache->graphs.back();
}

void launch_topology_epoch_conditional_sort(DirectBrainV01* brain, DirectTopologyRuntime* runtime,
                                            std::uint32_t proposal_work, AdultCounters* counters,
                                            std::uint32_t block_size) {
  DirectTopologyConditionalSortGraph* graph =
      conditional_sort_graph_for(brain, runtime, proposal_work, counters, block_size);
  DirectTopologyConditionalSortCache* cache = runtime->conditional_sort_cache;
  const DirectTopologyProposal* proposals = runtime->view.proposals;
  const std::uint32_t* proposal_high_water = runtime->view.proposal_high_water;
  std::uint32_t hard_capacity = proposal_work;
  std::uint32_t* live_indices = runtime->live_indices;
  std::uint32_t* live_count = runtime->live_count;
  std::uint32_t* active_work = cache->sort_control;
  void* compact_args[] = {&proposals, &proposal_high_water, &hard_capacity,
                          &live_indices, &live_count, &active_work};
  cudaKernelNodeParams compact_params{};
  compact_params.func = reinterpret_cast<void*>(&compact_live_proposals_kernel);
  compact_params.gridDim = dim3(grid_for_topology(proposal_work, block_size));
  compact_params.blockDim = dim3(block_size);
  compact_params.kernelParams = compact_args;
  topology_cuda(cudaGraphExecKernelNodeSetParams(graph->exec, graph->compact_node, &compact_params),
                "refresh topology conditional compact node");
  topology_cuda(cudaGraphLaunch(graph->exec, runtime->topology_epoch_graph_stream),
                "launch topology conditional sort graph");
}
#endif

}  // namespace

void launch_direct_topology_epoch(DirectBrainV01* brain, DirectTopologyRuntime* runtime,
                                  std::uint32_t proposal_work, AdultCounters* counters,
                                  std::uint32_t block_size, cudaStream_t stream) {
  if (brain == nullptr || runtime == nullptr || proposal_work == 0u)
    return;
  proposal_work = std::min(proposal_work, runtime->view.proposal_capacity);
  block_size = std::max(32u, std::min(1024u, block_size));
  const std::uint32_t grid = grid_for_topology(proposal_work, block_size);

  // Every `none` proposal sorts strictly after every live proposal under the
  // historical stable-radix order, so compacting and sorting [0, live_count) is
  // canonically exact. A host bound <= kFastSortSpan proves that live_count also
  // fits there and may take the existing fast graph directly. For a larger bound
  // #1260 forbids guessing from the host: the conditional graph compacts first
  // and the GPU selects fast versus radix from the resident live_count.
  if (proposal_work <= kFastSortSpan) {
    launch_topology_epoch_fast_sort_captured(brain, runtime, proposal_work, counters, grid,
                                             block_size, runtime->live_count,
                                             brain->topology.free_count,
                                             runtime->free_count_snapshot);
  } else {
#if CUDART_VERSION >= 12040
    launch_topology_epoch_conditional_sort(brain, runtime, proposal_work, counters, block_size);
#else
    // Compatibility only for CUDA runtimes that predate conditional graph nodes.
    // Current canonical sm_89 builds are CUDA 13.x and take the device branch above.
    launch_legacy_radix_proposal_sort(runtime, proposal_work, grid, block_size, stream);
    launch_topology_post_sort_sequence(brain, runtime, proposal_work, counters, grid, block_size,
                                       nullptr, brain->topology.free_count,
                                       runtime->free_count_snapshot, stream);
#endif
  }
#if CUDART_VERSION >= 12040
  (void)stream;  // both production graph paths own the existing blocking graph stream.
#endif

  topology_cuda(cudaGetLastError(), "launch direct topology epoch");
}

// Test oracle only: preserve the historical nine-pass order independently from
// production so the large-bound device-live path can be falsified without
// putting a host-bound branch back into the adult.
void launch_direct_topology_epoch_legacy_reference_for_test(
    DirectBrainV01* brain, DirectTopologyRuntime* runtime, std::uint32_t proposal_work,
    AdultCounters* counters, std::uint32_t block_size, cudaStream_t stream) {
  if (brain == nullptr || runtime == nullptr || proposal_work == 0u)
    return;
  proposal_work = std::min(proposal_work, runtime->view.proposal_capacity);
  block_size = std::max(32u, std::min(1024u, block_size));
  const std::uint32_t grid = grid_for_topology(proposal_work, block_size);

  launch_legacy_radix_proposal_sort(runtime, proposal_work, grid, block_size, stream);
  launch_topology_post_sort_sequence(brain, runtime, proposal_work, counters, grid, block_size,
                                     nullptr, brain->topology.free_count,
                                     runtime->free_count_snapshot, stream);
  topology_cuda(cudaGetLastError(), "launch legacy topology epoch test reference");
}

const std::uint32_t* direct_topology_sort_path_receipt_for_test(
    const DirectTopologyRuntime* runtime) {
  if (runtime == nullptr || runtime->conditional_sort_cache == nullptr)
    return nullptr;
  return runtime->conditional_sort_cache->sort_control + 1u;
}

std::uint32_t direct_topology_conditional_sort_graph_count_for_test(
    const DirectTopologyRuntime* runtime) {
  if (runtime == nullptr || runtime->conditional_sort_cache == nullptr)
    return 0u;
  return static_cast<std::uint32_t>(runtime->conditional_sort_cache->graphs.size());
}

}  // namespace substrate::direct_adult
