#include <algorithm>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

#include "bcc32_device_ordinary_f_timeline.cuh"
#include "bcc32_law_netlist.cuh"

namespace substrate::bcc32::device_ordinary_f_timeline {

struct SnapshotControlReceipt {
  // Offset zero intentionally preserves DeviceState::publication as the
  // resident publication address while extending its allocation with
  // observer-only checkpoint control staging.
  OrdinaryFPublication publication{};
  std::uint64_t completed_ticks = 0u;
  std::uint64_t generation = 0u;
  std::uint32_t inverse_head = 0u;
  std::uint32_t inverse_depth = 0u;
  std::uint32_t executing = 0u;
  std::uint32_t fault = 0u;
  std::uint32_t active_count = 0u;
  std::uint32_t attempted_count = 0u;
};

static_assert(std::is_standard_layout_v<SnapshotControlReceipt>);
static_assert(std::is_trivially_copyable_v<SnapshotControlReceipt>);
static_assert(offsetof(SnapshotControlReceipt, publication) == 0u);

namespace {

constexpr std::uint32_t kThreads = 256u;
constexpr std::uint32_t kOrdinarySpatialMacroClosureRadius = 15u;
constexpr std::uint64_t kNoSlot = std::numeric_limits<std::uint64_t>::max();
constexpr std::uint32_t kNoHistorySlot =
    std::numeric_limits<std::uint32_t>::max();

void check_cuda(cudaError_t status, const char* operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
  }
}

std::uint32_t fixed_blocks(std::uint32_t count) {
  const std::uint64_t blocks = (static_cast<std::uint64_t>(count) + kThreads - 1u) / kThreads;
  if (blocks == 0u || blocks > std::numeric_limits<std::uint32_t>::max())
    throw std::overflow_error("ordinary-F timeline fixed grid is invalid");
  return static_cast<std::uint32_t>(blocks);
}

__device__ inline void publish_fault(DeviceState* state, Fault fault) {
  const std::uint32_t value = static_cast<std::uint32_t>(fault);
  atomicMax(state->fault, value);
}

__device__ inline std::uint64_t mix_slot(std::uint64_t value) {
  value ^= value >> 30u;
  value *= 0xbf58476d1ce4e5b9ull;
  value ^= value >> 27u;
  value *= 0x94d049bb133111ebull;
  value ^= value >> 31u;
  return value;
}

__host__ __device__ inline std::uint64_t mix_digest(std::uint64_t value) {
  value ^= value >> 30u;
  value *= 0xbf58476d1ce4e5b9ull;
  value ^= value >> 27u;
  value *= 0x94d049bb133111ebull;
  value ^= value >> 31u;
  return value;
}

__host__ __device__ inline std::uint64_t fold_digest(std::uint64_t lane,
                                                     std::uint64_t value) {
  return mix_digest(lane ^ mix_digest(value + 0x9e3779b97f4a7c15ull));
}

WorldDigest host_world_digest(std::uint32_t chunk_count,
                              std::uint32_t capacity,
                              const std::vector<SiteWord>& words,
                              const std::vector<std::uint64_t>& slots) {
  std::uint64_t lane0 = 0x243f6a8885a308d3ull;
  std::uint64_t lane1 = 0x13198a2e03707344ull;
  std::uint64_t lane2 = 0xa4093822299f31d0ull;
  std::uint64_t lane3 = 0x082efa98ec4e6c89ull;
  lane0 = fold_digest(lane0, chunk_count);
  lane1 = fold_digest(lane1, slots.size());
  lane2 = fold_digest(
      lane2, static_cast<std::uint64_t>(chunk_count) * kChunkSites);
  lane3 = fold_digest(lane3, capacity);
  for (std::size_t index = 0u; index < slots.size(); ++index) {
    const std::uint64_t slot = slots[index];
    if (slot >= words.size())
      throw std::invalid_argument(
          "ordinary-F checkpoint digest slot is outside the world");
    const std::uint64_t word = words[slot];
    lane0 = fold_digest(lane0, slot);
    lane1 = fold_digest(lane1, word);
    lane2 = fold_digest(lane2, slot ^ (word << 32u));
    lane3 = fold_digest(lane3,
                        static_cast<std::uint64_t>(index) ^ (slot << 1u) ^
                            word);
  }
  return {lane0, lane1, lane2, lane3};
}

__device__ inline bool valid_slot(const DeviceState& state, std::uint64_t slot) {
  const std::uint64_t site_count =
      static_cast<std::uint64_t>(state.topology.chunks.chunk_count) * kChunkSites;
  return slot != kNoSlot && slot < site_count;
}

__device__ inline bool insert_unique(DeviceState* state, std::uint64_t slot) {
  if (!valid_slot(*state, slot)) {
    publish_fault(state, Fault::invalid_input);
    return false;
  }
  const std::uint32_t hash_capacity =
      state->closure_workspace_capacity * 2u;
  const std::uint32_t start = static_cast<std::uint32_t>(mix_slot(slot)) & (hash_capacity - 1u);
  auto* hash = reinterpret_cast<unsigned long long*>(state->hash);
  for (std::uint32_t probe = 0u; probe < hash_capacity; ++probe) {
    const std::uint32_t index = (start + probe) & (hash_capacity - 1u);
    const unsigned long long prior = atomicCAS(
        &hash[index], static_cast<unsigned long long>(kNoSlot),
        static_cast<unsigned long long>(slot));
    if (prior == static_cast<unsigned long long>(kNoSlot)) {
      const std::uint32_t count = atomicAdd(state->work_count, 1u) + 1u;
      if (count > state->closure_workspace_capacity)
        publish_fault(state, Fault::capacity_overflow);
      return true;
    }
    if (prior == static_cast<unsigned long long>(slot))
      return false;
  }
  publish_fault(state, Fault::capacity_overflow);
  return false;
}

__device__ inline bool next_neighbor(DeviceState& state, std::uint64_t source,
                                     std::uint32_t direction, std::uint64_t* destination) {
  if (!device_topology::neighbor_slot(state.topology, source, direction, destination)) {
    // A finite grown aperture has lawful outer boundaries. Reaching one
    // terminates this candidate route; it is not malformed chronology and
    // must not poison the whole ordinary-F publication with a closure fault.
    return false;
  }
  return true;
}

__device__ inline void clear_workspace(DeviceState* state) {
  const std::uint32_t lane = threadIdx.x;
  // Only the probe hash carries state across calls: insert_unique searches for
  // the kNoSlot sentinel, so a stale entry from the previous tick would be
  // read as an occupant. `work` and `next` are pure scratch that every
  // consumer writes before it reads:
  //   * seed_and_expand_domain/expand_frontier fill work[0, frontier_count)
  //     and next[0, next_count) and never index past those counters;
  //   * build_domain writes next[0, min(work_count, capacity)) and then
  //     initializes the remainder that sort_and_commit can reach.
  // Pre-clearing them cost one single-block sweep of the whole stored
  // workspace (2 x capacity x 8 B) on every ordinary-F domain refresh, which
  // is morphology-sized work that the live causal frontier never reads.
  for (std::uint32_t index = lane;
       index < state->closure_workspace_capacity * 2u;
       index += blockDim.x)
    state->hash[index] = kNoSlot;
  if (lane == 0u)
    *state->work_count = 0u;
  __syncthreads();
}

__device__ inline void expand_frontier(DeviceState* state, std::uint32_t radius,
                                       std::uint32_t* frontier_count,
                                       std::uint32_t* next_count) {
  for (std::uint32_t depth = 0u; depth < radius; ++depth) {
    if (threadIdx.x == 0u)
      *next_count = 0u;
    __syncthreads();
    for (std::uint32_t index = threadIdx.x; index < *frontier_count;
         index += blockDim.x) {
      const std::uint64_t source = state->work[index];
      for (std::uint32_t direction = 0u; direction < 8u; ++direction) {
        std::uint64_t neighbor = 0u;
        if (!next_neighbor(*state, source, direction, &neighbor))
          continue;
        if (!insert_unique(state, neighbor))
          continue;
        const std::uint32_t output = atomicAdd(next_count, 1u);
        if (output >= state->closure_workspace_capacity) {
          publish_fault(state, Fault::capacity_overflow);
          continue;
        }
        state->next[output] = neighbor;
      }
    }
    __syncthreads();
    for (std::uint32_t index = threadIdx.x;
         index < *next_count && index < state->closure_workspace_capacity;
         index += blockDim.x)
      state->work[index] = state->next[index];
    __syncthreads();
    if (threadIdx.x == 0u)
      *frontier_count =
          *next_count < state->closure_workspace_capacity
              ? *next_count
              : state->closure_workspace_capacity;
    __syncthreads();
    if (*frontier_count == 0u ||
        static_cast<Fault>(*state->fault) != Fault::none)
      return;
  }
}

__device__ inline bool root_is_live(const DeviceState& state, std::uint64_t slot,
                                    bool filter_words) {
  return !filter_words || (state.words != nullptr && state.words[slot] != kQ);
}

__device__ inline void seed_and_expand_domain(
    DeviceState* state, std::uint32_t radius, bool filter_roots,
    std::uint32_t* frontier_count, std::uint32_t* next_count) {
  const std::uint32_t old_count = *state->count;
  if (threadIdx.x == 0u)
    *frontier_count = 0u;
  __syncthreads();
  for (std::uint32_t index = threadIdx.x; index < old_count;
       index += blockDim.x) {
    const std::uint64_t slot = state->slots[index];
    if (filter_roots && !root_is_live(*state, slot, true))
      continue;
    if (!insert_unique(state, slot))
      continue;
    const std::uint32_t output = atomicAdd(frontier_count, 1u);
    if (output >= state->closure_workspace_capacity) {
      publish_fault(state, Fault::capacity_overflow);
      continue;
    }
    state->work[output] = slot;
  }
  // A reciprocal boundary slot is not necessarily resident at bootstrap: its
  // quiescent word is deliberately Q. Once the admitted raw exchange writes
  // a sensory rail, the next F tick must be able to grow from that physical
  // matter. Re-enrol only the fixed topology roots supplied by GrownAdult;
  // this is not a caller-selected route or action coordinate.
  const std::uint64_t boundary_roots[] = {
      state->boundary.sensory_zero_slot,
      state->boundary.sensory_one_slot,
      state->boundary.motor_zero_slot,
      state->boundary.motor_one_slot};
  for (std::uint32_t index = threadIdx.x; index < 4u;
       index += blockDim.x) {
    const std::uint64_t slot = boundary_roots[index];
    const std::uint64_t site_count =
        static_cast<std::uint64_t>(state->topology.chunks.chunk_count) *
        kChunkSites;
    if (slot == kUnboundSlot || slot >= site_count ||
        (filter_roots && !root_is_live(*state, slot, true)))
      continue;
    if (!insert_unique(state, slot))
      continue;
    const std::uint32_t output = atomicAdd(frontier_count, 1u);
    if (output >= state->closure_workspace_capacity) {
      publish_fault(state, Fault::capacity_overflow);
      continue;
    }
    state->work[output] = slot;
  }
  __syncthreads();
  if (*frontier_count > state->closure_workspace_capacity &&
      threadIdx.x == 0u)
    *frontier_count = state->closure_workspace_capacity;
  __syncthreads();
  if (static_cast<Fault>(*state->fault) == Fault::none)
    expand_frontier(state, radius, frontier_count, next_count);
}

// The canonical support order is produced by a bitonic sort over the smallest
// power of two that contains the live count.  Both the producer (build_domain,
// which must initialize every cell the sort can touch) and the sort itself
// derive their span from this one function so the two can never disagree.
__device__ inline std::uint32_t canonical_sort_width(std::uint32_t live_count,
                                                     std::uint32_t capacity) {
  std::uint32_t sort_width = 1u;
  while (sort_width < live_count && sort_width < capacity)
    sort_width <<= 1u;
  return sort_width;
}

__device__ inline void build_domain(DeviceState* state, std::uint32_t radius,
                                    bool filter_words,
                                    std::uint32_t* frontier_count,
                                    std::uint32_t* next_count) {
  seed_and_expand_domain(state, radius, false, frontier_count, next_count);
  if (static_cast<Fault>(*state->fault) != Fault::none)
    return;

  // Hash insertion is deterministic with respect to input ordering only after
  // this fixed-width bitonic sort.  The sorted list is the device canonical
  // support consumed by every subsequent factor window.
  if (threadIdx.x == 0u)
    *state->work_count = 0u;
  __syncthreads();
  for (std::uint32_t index = threadIdx.x;
       index < state->closure_workspace_capacity * 2u;
       index += blockDim.x) {
    const std::uint64_t slot = state->hash[index];
    if (slot != kNoSlot && root_is_live(*state, slot, filter_words)) {
      const std::uint32_t output = atomicAdd(state->work_count, 1u);
      if (output >= state->capacity) {
        publish_fault(state, Fault::capacity_overflow);
        continue;
      }
      state->next[output] = slot;
    }
  }
  __syncthreads();
  // sort_and_commit only ever reads next[0, canonical_sort_width), so padding
  // the whole stored capacity with kNoSlot wrote up to capacity x 8 B that no
  // reader can observe.  Pad exactly the sorted span instead: the live frontier
  // sets the cost, not the size of the resident workspace.
  const std::uint32_t live_count =
      *state->work_count < state->capacity ? *state->work_count
                                           : state->capacity;
  const std::uint32_t sort_width =
      canonical_sort_width(live_count, state->capacity);
  for (std::uint32_t index = threadIdx.x + live_count; index < sort_width;
       index += blockDim.x)
    state->next[index] = kNoSlot;
  __syncthreads();
}

__device__ inline void sort_and_commit(DeviceState* state) {
  const std::uint32_t lane = threadIdx.x;
  const std::uint32_t count =
      *state->work_count < state->capacity ? *state->work_count
                                           : state->capacity;
  // The support window is a fixed-capacity resident buffer, but the live
  // frontier is usually sparse. Sorting the entire 2^21 buffer made every
  // ordinary-F tick pay O(capacity log^2 capacity), even when only a few
  // routes were active. The hash scan has already filled the remainder with
  // kNoSlot, so sorting the next power-of-two containing the live count is
  // equivalent while keeping the canonical deterministic order.
  const std::uint32_t sort_width = canonical_sort_width(count, state->capacity);
  for (std::uint32_t width = 2u; width <= sort_width; width <<= 1u) {
    for (std::uint32_t stride = width >> 1u; stride != 0u; stride >>= 1u) {
      for (std::uint32_t index = lane; index < sort_width; index += blockDim.x) {
        const std::uint32_t partner = index ^ stride;
        if (partner <= index)
          continue;
        const bool ascending = (index & width) == 0u;
        const std::uint64_t left = state->next[index];
        const std::uint64_t right = state->next[partner];
        if ((ascending && left > right) || (!ascending && left < right)) {
          state->next[index] = right;
          state->next[partner] = left;
        }
      }
      __syncthreads();
    }
  }
  if (lane == 0u) {
    for (std::uint32_t index = 0u; index < count; ++index)
      state->slots[index] = state->next[index];
    *state->count = count;
    *state->dispatch_count = count;
  }
  __syncthreads();
}

__global__ void begin_world_scan_kernel(DeviceState* state, bool reset_timeline) {
  const std::uint64_t index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::uint64_t stride = static_cast<std::uint64_t>(gridDim.x) * blockDim.x;
  for (std::uint64_t slot = index; slot < state->capacity; slot += stride) {
    state->slots[slot] = kNoSlot;
    state->next[slot] = kNoSlot;
  }
  if (index == 0u) {
    *state->count = 0u;
    *state->dispatch_count = 0u;
    *state->work_count = 0u;
    *state->executing = 0u;
    *state->fault = static_cast<std::uint32_t>(Fault::none);
    if (reset_timeline) {
      *state->inverse_head = 0u;
      *state->inverse_depth = 0u;
      *state->completed_ticks = 0u;
      *state->requested_target = 0u;
      *state->current_frame = 0u;
      *state->generation = 0u;
    }
  }
}

__global__ void scan_world_support_kernel(DeviceState* state, std::uint64_t site_count) {
  const std::uint64_t index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::uint64_t stride = static_cast<std::uint64_t>(gridDim.x) * blockDim.x;
  for (std::uint64_t slot = index; slot < site_count; slot += stride) {
    if (state->words[slot] == kQ)
      continue;
    const std::uint32_t output = atomicAdd(state->work_count, 1u);
    if (output < state->capacity)
      state->next[output] = slot;
    else
      publish_fault(state, Fault::capacity_overflow);
  }
}

__global__ void commit_world_scan_kernel(DeviceState* state,
                                         bool invalidate_inverse) {
  if (blockIdx.x != 0u || blockIdx.y != 0u || blockIdx.z != 0u)
    return;
  if (*state->fault != static_cast<std::uint32_t>(Fault::none)) {
    if (threadIdx.x == 0u) {
      *state->count = 0u;
      *state->dispatch_count = 0u;
    }
    return;
  }
  sort_and_commit(state);
  if (threadIdx.x == 0u && invalidate_inverse) {
    *state->inverse_depth = 0u;
    *state->current_frame = *state->inverse_head;
    *state->requested_target = *state->completed_ticks;
    *state->dispatch_count = *state->count;
  }
}

__global__ void begin_forward_kernel(DeviceState* state) {
  if (blockIdx.x != 0u || threadIdx.x != 0u)
    return;
  *state->dispatch_count = 0u;
  *state->executing = 0u;
  if (*state->fault != static_cast<std::uint32_t>(Fault::none) ||
      *state->completed_ticks >= *state->requested_target) {
    return;
  }
  *state->current_frame = *state->inverse_head;
  *state->executing = 1u;
  *state->fault = static_cast<std::uint32_t>(Fault::none);
  if (state->publication != nullptr)
    state->publication->continuation_phase = 2u;
}

__global__ void preflight_forward_kernel(DeviceState* state) {
  if (blockIdx.x != 0u || blockIdx.y != 0u || blockIdx.z != 0u)
    return;
  if (*state->executing == 0u)
    return;
  __shared__ std::uint32_t frontier_count;
  __shared__ std::uint32_t next_count;
  clear_workspace(state);
  seed_and_expand_domain(state, kSpatialMacroClosureRadius, true,
                         &frontier_count, &next_count);
  if (static_cast<Fault>(*state->fault) != Fault::none) {
    if (threadIdx.x == 0u)
      *state->executing = 0u;
  }
  __syncthreads();
  if (threadIdx.x == 0u && *state->executing == 0u)
    *state->dispatch_count = 0u;
}

__global__ void refresh_domain_kernel(DeviceState* state, std::uint32_t radius, bool filter_words) {
  if (blockIdx.x != 0u || blockIdx.y != 0u || blockIdx.z != 0u)
    return;
  if (*state->executing == 0u)
    return;
  __shared__ std::uint32_t frontier_count;
  __shared__ std::uint32_t next_count;
  clear_workspace(state);
  build_domain(state, radius, filter_words, &frontier_count, &next_count);
  if (static_cast<Fault>(*state->fault) != Fault::none) {
    if (threadIdx.x == 0u) {
      *state->executing = 0u;
      *state->dispatch_count = 0u;
    }
    return;
  }
  sort_and_commit(state);
}

__global__ void capture_boundary_kernel(DeviceState* state, std::uint32_t boundary) {
  if (blockIdx.y != 0u || blockIdx.z != 0u)
    return;
  if (*state->executing == 0u || boundary >= state->boundary_count)
    return;
  const std::uint32_t index = static_cast<std::uint32_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::uint64_t base =
      (static_cast<std::uint64_t>(*state->current_frame) * state->boundary_count + boundary) *
      state->capacity;
  // Every reader of this journal frame is bounded by the count written just
  // below: restore_boundary_kernel reads history_slots[base, base + count) and
  // the host snapshot copies exactly count entries.  The kNoHistorySlot padding
  // over [count, capacity) was therefore unobservable, and writing it made each
  // of the six journal boundaries per tick cost the stored support capacity
  // rather than the live support.
  if (index < *state->count) {
    const std::uint64_t slot = state->slots[index];
    state->history_slots[base + index] =
        slot == kNoSlot ? kNoHistorySlot
                        : static_cast<std::uint32_t>(slot);
  }
  if (threadIdx.x == 0u)
    state
        ->history_counts[static_cast<std::uint64_t>(*state->current_frame) * state->boundary_count +
                         boundary] = *state->count;
}

__global__ void restore_boundary_kernel(DeviceState* state, std::uint32_t boundary) {
  if (blockIdx.y != 0u || blockIdx.z != 0u)
    return;
  if (*state->executing == 0u || boundary >= state->boundary_count)
    return;
  const std::uint32_t index = static_cast<std::uint32_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::uint64_t base =
      (static_cast<std::uint64_t>(*state->current_frame) * state->boundary_count + boundary) *
      state->capacity;
  const std::uint32_t count =
      state->history_counts[static_cast<std::uint64_t>(*state->current_frame) *
                                state->boundary_count +
                            boundary];
  if (index < state->capacity) {
    const std::uint32_t slot =
        index < count ? state->history_slots[base + index]
                      : kNoHistorySlot;
    state->slots[index] =
        slot == kNoHistorySlot ? kNoSlot : static_cast<std::uint64_t>(slot);
  }
  if (threadIdx.x == 0u) {
    *state->count = count;
    *state->dispatch_count = count;
  }
}

__device__ inline void publish_world(DeviceState* state, bool preserve_continuation) {
  if (state == nullptr || state->publication == nullptr) return;
  const OrdinaryFPublication previous = *state->publication;
  OrdinaryFPublication publication{};
  publication.completed_ticks = *state->completed_ticks;
  publication.generation = *state->generation;
  // On a capacity fault commit_world_scan/refresh_domain deliberately clears
  // the resident window so no truncated support can execute. Preserve the
  // attempted population in the device publication via work_count, making the
  // RED receipt diagnostic instead of silently reporting an empty window.
  const Fault current_fault = static_cast<Fault>(*state->fault);
  publication.active_count =
      current_fault == Fault::capacity_overflow ? *state->work_count
                                                 : *state->count;
  if (preserve_continuation) {
    publication.continuation_phase = previous.continuation_phase;
    publication.return_launch_status = previous.return_launch_status;
  }
  std::uint64_t lane0 = 0x243f6a8885a308d3ull;
  std::uint64_t lane1 = 0x13198a2e03707344ull;
  std::uint64_t lane2 = 0xa4093822299f31d0ull;
  std::uint64_t lane3 = 0x082efa98ec4e6c89ull;
  lane0 = fold_digest(lane0, state->topology.chunks.chunk_count);
  lane1 = fold_digest(lane1, publication.active_count);
  lane2 = fold_digest(
      lane2, static_cast<std::uint64_t>(state->topology.chunks.chunk_count) *
                 kChunkSites);
  lane3 = fold_digest(lane3, state->capacity);
  if (publication.active_count > state->capacity) {
    publish_fault(state, Fault::capacity_overflow);
  } else {
    for (std::uint32_t index = 0u; index < publication.active_count; ++index) {
      const std::uint64_t slot = state->slots[index];
      if (!valid_slot(*state, slot)) {
        publish_fault(state, Fault::invalid_input);
        break;
      }
      const std::uint64_t word = state->words[slot];
      lane0 = fold_digest(lane0, slot);
      lane1 = fold_digest(lane1, word);
      lane2 = fold_digest(lane2, slot ^ (word << 32u));
      lane3 = fold_digest(lane3,
                          static_cast<std::uint64_t>(index) ^
                              (slot << 1u) ^ word);
    }
  }
  publication.fault = *state->fault;
  if (valid_slot(*state, state->boundary.motor_zero_slot) &&
      valid_slot(*state, state->boundary.motor_one_slot)) {
    publication.motor = {state->words[state->boundary.motor_zero_slot],
                         state->words[state->boundary.motor_one_slot]};
  }
  publication.world = {lane0, lane1, lane2, lane3};
  *state->publication = publication;
  __threadfence();
}

__global__ void begin_inverse_kernel(DeviceState* state) {
  if (blockIdx.x != 0u || threadIdx.x != 0u)
    return;
  *state->dispatch_count = 0u;
  *state->executing = 0u;
  if (*state->fault != static_cast<std::uint32_t>(Fault::none) ||
      *state->completed_ticks <= *state->requested_target || *state->inverse_depth == 0u)
    return;
  *state->current_frame =
      (*state->inverse_head + state->history_capacity - 1u) % state->history_capacity;
  *state->executing = 1u;
  *state->dispatch_count = *state->count;
}

__global__ void end_forward_kernel(DeviceState* state) {
  if (blockIdx.x != 0u || threadIdx.x != 0u)
    return;
  if (*state->executing == 0u) {
    if (state->publication != nullptr)
      state->publication->continuation_phase = 3u;
    return;
  }
  if (*state->fault == static_cast<std::uint32_t>(Fault::none)) {
    *state->inverse_head = (*state->inverse_head + 1u) % state->history_capacity;
    if (*state->inverse_depth < state->history_capacity)
      ++(*state->inverse_depth);
    ++(*state->generation);
    cuda::atomic_ref<std::uint64_t, cuda::thread_scope_device> completed(
        *state->completed_ticks);
    completed.fetch_add(1u, cuda::memory_order_release);
    if (state->publication != nullptr)
      state->publication->continuation_phase = 4u;
  } else if (state->publication != nullptr) {
    state->publication->continuation_phase = 3u;
  }
  *state->dispatch_count = 0u;
  *state->executing = 0u;
}

__global__ void end_inverse_kernel(DeviceState* state) {
  if (blockIdx.x != 0u || threadIdx.x != 0u)
    return;
  if (*state->executing == 0u)
    return;
  if (*state->fault == static_cast<std::uint32_t>(Fault::none) && *state->inverse_depth != 0u &&
      *state->completed_ticks != 0u) {
    *state->inverse_head = *state->current_frame;
    --(*state->inverse_depth);
    --(*state->completed_ticks);
    ++(*state->generation);
  }
  *state->dispatch_count = 0u;
  *state->executing = 0u;
}

__global__ void return_to_parent_kernel(DeviceState* state) {
  if (blockIdx.x != 0u || threadIdx.x != 0u || state == nullptr ||
      state->return_graph == nullptr)
    return;
  // The resident root wrote its exact current graph executable into this
  // timeline slot before tail-launching the child. A tail launch here returns
  // to that same root only after this child graph has committed its state;
  // sibling launch would let the root's own tail environment race the child.
  const cudaError_t status =
      cudaGraphLaunch(state->return_graph, cudaStreamGraphTailLaunch);
  if (status != cudaSuccess) {
    if (state->publication != nullptr) {
      state->publication->return_launch_status =
          static_cast<std::uint32_t>(status);
      __threadfence_system();
    }
    publish_fault(state, Fault::invalid_input);
  } else if (state->publication != nullptr) {
    state->publication->continuation_phase = 5u;
    __threadfence_system();
  }
}

__global__ void publish_world_kernel(DeviceState* state, bool preserve_continuation) {
  if (blockIdx.x != 0u || threadIdx.x != 0u || state == nullptr ||
      state->publication == nullptr)
    return;
  publish_world(state, preserve_continuation);
}

__global__ void capture_snapshot_control_kernel(const DeviceState* state) {
  if (blockIdx.x != 0u || threadIdx.x != 0u || state == nullptr ||
      state->publication == nullptr)
    return;
  auto* receipt = reinterpret_cast<SnapshotControlReceipt*>(state->publication);
  receipt->completed_ticks = *state->completed_ticks;
  receipt->generation = *state->generation;
  receipt->inverse_head = *state->inverse_head;
  receipt->inverse_depth = *state->inverse_depth;
  receipt->executing = *state->executing;
  receipt->fault = *state->fault;
  receipt->active_count = *state->count;
  receipt->attempted_count = *state->work_count;
}

}  // namespace

DeviceOrdinaryFTimeline::DeviceOrdinaryFTimeline(CudaBcc32Executor& executor,
                                                 DeviceChunkMap chunks,
                                                 std::uint32_t capacity,
                                                 std::uint32_t history_capacity)
    : DeviceOrdinaryFTimeline(executor, chunks, DeviceBoundaryBinding{},
                              capacity, history_capacity) {}

DeviceOrdinaryFTimeline::DeviceOrdinaryFTimeline(
    CudaBcc32Executor& executor, DeviceChunkMap chunks,
    DeviceBoundaryBinding boundary,
    std::uint32_t capacity,
    std::uint32_t history_capacity)
    : executor_(executor),
      chunks_(chunks),
      launch_capacity_(capacity),
      history_capacity_(history_capacity),
      boundary_(boundary) {
  if (chunks_.slots == nullptr || chunks_.chunk_count == 0u)
    throw std::invalid_argument("ordinary-F timeline topology is invalid");
  if (static_cast<std::uint64_t>(chunks_.chunk_count) * kChunkSites != executor_.site_count())
    throw std::invalid_argument("ordinary-F timeline topology does not own the executor world");
  if (executor_.site_count() >
      static_cast<std::uint64_t>(std::numeric_limits<std::uint32_t>::max()))
    throw std::invalid_argument(
        "ordinary-F inverse journal requires aperture-local 32-bit slots");
  if (launch_capacity_ == 0u || (launch_capacity_ & (launch_capacity_ - 1u)) != 0u ||
      launch_capacity_ > kCapacity)
    throw std::invalid_argument("ordinary-F timeline capacity must be a bounded power of two");
  if (history_capacity_ == 0u || history_capacity_ > kHistoryCapacity)
    throw std::invalid_argument("ordinary-F timeline history depth is invalid");
  allocate_state();
}

void DeviceOrdinaryFTimeline::allocate_state() {
  check_cuda(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking),
             "create ordinary-F timeline stream");
  DeviceState state{};
  auto allocate = [](void** pointer, std::size_t bytes, const char* operation) {
    if (bytes != 0u)
      check_cuda(cudaMalloc(pointer, bytes), operation);
  };
  auto alloc_array = [&allocate](auto** pointer, std::size_t count, const char* operation) {
    allocate(reinterpret_cast<void**>(pointer), count * sizeof(**pointer), operation);
  };
  auto free_state = [&]() noexcept {
    (void)cudaFree(state.slots);
    (void)cudaFree(state.work);
    (void)cudaFree(state.next);
    (void)cudaFree(state.hash);
    (void)cudaFree(state.history_slots);
    (void)cudaFree(state.history_counts);
    (void)cudaFree(state.count);
    (void)cudaFree(state.dispatch_count);
    (void)cudaFree(state.work_count);
    (void)cudaFree(state.inverse_head);
    (void)cudaFree(state.inverse_depth);
    (void)cudaFree(state.completed_ticks);
    (void)cudaFree(state.requested_target);
    (void)cudaFree(state.current_frame);
    (void)cudaFree(state.executing);
    (void)cudaFree(state.fault);
    (void)cudaFree(state.generation);
    (void)cudaFree(state.publication);
    (void)cudaFree(device_resolution_);
  };
  try {
    allocate(reinterpret_cast<void**>(&device_state_), sizeof(DeviceState),
             "allocate ordinary-F timeline state");
    allocate(reinterpret_cast<void**>(&device_window_),
             static_cast<std::size_t>(launch_capacity_) * sizeof(std::uint64_t),
             "allocate ordinary-F timeline active window");
    allocate(reinterpret_cast<void**>(&device_resolution_),
             static_cast<std::size_t>(launch_capacity_) * sizeof(std::uint8_t),
             "allocate ordinary-F timeline collision resolution");
    alloc_array(&state.slots, launch_capacity_, "allocate timeline support slots");
    alloc_array(&state.work, launch_capacity_,
                "allocate timeline closure work slots");
    alloc_array(&state.next, launch_capacity_,
                "allocate timeline closure/sort slots");
    alloc_array(&state.hash,
                static_cast<std::size_t>(launch_capacity_) * 2u,
                "allocate timeline closure hash");
    alloc_array(&state.history_slots,
                static_cast<std::size_t>(history_capacity_) * kBoundaryCount * launch_capacity_,
                "allocate timeline support history");
    alloc_array(&state.history_counts, static_cast<std::size_t>(history_capacity_) * kBoundaryCount,
                "allocate timeline history counts");
    alloc_array(&state.count, 1u, "allocate timeline support count");
    alloc_array(&state.dispatch_count, 1u, "allocate timeline dispatch count");
    alloc_array(&state.work_count, 1u, "allocate timeline work count");
    alloc_array(&state.inverse_head, 1u, "allocate timeline inverse ring head");
    alloc_array(&state.inverse_depth, 1u, "allocate timeline inverse ring depth");
    alloc_array(&state.completed_ticks, 1u, "allocate timeline completed ticks");
    alloc_array(&state.requested_target, 1u, "allocate timeline requested target");
    alloc_array(&state.current_frame, 1u, "allocate timeline current inverse frame");
    alloc_array(&state.executing, 1u, "allocate timeline executing flag");
    alloc_array(&state.fault, 1u, "allocate timeline fault");
    alloc_array(&state.generation, 1u, "allocate timeline generation");
    allocate(reinterpret_cast<void**>(&state.publication),
             sizeof(SnapshotControlReceipt),
             "allocate timeline world publication and snapshot receipt");
    check_cuda(cudaMemset(
                   state.history_counts, 0,
                   static_cast<std::size_t>(history_capacity_) * kBoundaryCount *
                       sizeof(std::uint32_t)),
               "initialize ordinary-F history counts");
    check_cuda(cudaMemset(
                   state.history_slots, 0xff,
                   static_cast<std::size_t>(history_capacity_) * kBoundaryCount *
                       launch_capacity_ * sizeof(std::uint32_t)),
               "initialize ordinary-F history slots");
    check_cuda(cudaMemset(device_resolution_, 0,
                          static_cast<std::size_t>(launch_capacity_) *
                              sizeof(std::uint8_t)),
               "initialize ordinary-F collision resolution");

    state.words = executor_.device_words();
    state.boundary = boundary_;
    state.capacity = launch_capacity_;
    state.closure_workspace_capacity = launch_capacity_;
    state.history_capacity = history_capacity_;
    state.boundary_count = kBoundaryCount;
    state.topology = {chunks_};
    host_state_ = state;
    check_cuda(cudaMemcpy(device_state_, &state, sizeof(state), cudaMemcpyHostToDevice),
               "upload ordinary-F timeline state");
    rebuild_support_from_world(true);
    std::uint32_t fault = 0u;
    check_cuda(cudaMemcpy(&fault, host_state_.fault, sizeof(fault), cudaMemcpyDeviceToHost),
               "read ordinary-F timeline bootstrap fault");
    if (fault != static_cast<std::uint32_t>(Fault::none))
      throw std::runtime_error("ordinary-F timeline bootstrap fault");
  } catch (...) {
    destroy_graphs();
    free_state();
    (void)cudaFree(device_window_);
    (void)cudaFree(device_state_);
    device_window_ = nullptr;
    device_state_ = nullptr;
    host_state_ = DeviceState{};
    if (stream_ != nullptr)
      (void)cudaStreamDestroy(stream_);
    stream_ = nullptr;
    throw;
  }
}

void DeviceOrdinaryFTimeline::rebuild_support_from_world(bool reset_timeline) {
  const std::uint32_t clear_blocks = fixed_blocks(launch_capacity_);
  const std::uint64_t site_count = executor_.site_count();
  const std::uint64_t required_scan_blocks = (site_count + kThreads - 1u) / kThreads;
  const std::uint32_t scan_blocks =
      static_cast<std::uint32_t>(std::min<std::uint64_t>(required_scan_blocks, 65535u));
  begin_world_scan_kernel<<<clear_blocks, kThreads, 0, stream_>>>(device_state_, reset_timeline);
  scan_world_support_kernel<<<scan_blocks, kThreads, 0, stream_>>>(device_state_, site_count);
  commit_world_scan_kernel<<<1u, kThreads, 0, stream_>>>(device_state_,
                                                         !reset_timeline);
  publish_world_kernel<<<1u, 1u, 0, stream_>>>(device_state_, false);
  check_cuda(cudaGetLastError(), "launch ordinary-F device world scan");
  check_cuda(cudaStreamSynchronize(stream_), "synchronize ordinary-F device world scan");
  if (reset_timeline) {
    host_completed_ticks_ = 0u;
    host_inverse_head_ = 0u;
    host_inverse_depth_ = 0u;
    host_fault_ = Fault::none;
    host_generation_ = 0u;
  }
}

DeviceOrdinaryFTimeline::~DeviceOrdinaryFTimeline() {
  destroy_graphs();
  (void)cudaFree(host_state_.slots);
  (void)cudaFree(host_state_.work);
  (void)cudaFree(host_state_.next);
  (void)cudaFree(host_state_.hash);
  (void)cudaFree(host_state_.history_slots);
  (void)cudaFree(host_state_.history_counts);
  (void)cudaFree(host_state_.count);
  (void)cudaFree(host_state_.dispatch_count);
  (void)cudaFree(host_state_.work_count);
  (void)cudaFree(host_state_.inverse_head);
  (void)cudaFree(host_state_.inverse_depth);
  (void)cudaFree(host_state_.completed_ticks);
  (void)cudaFree(host_state_.requested_target);
  (void)cudaFree(host_state_.current_frame);
  (void)cudaFree(host_state_.executing);
  (void)cudaFree(host_state_.fault);
  (void)cudaFree(host_state_.generation);
  (void)cudaFree(host_state_.publication);
  (void)cudaFree(device_window_);
  (void)cudaFree(device_resolution_);
  (void)cudaFree(device_state_);
  if (stream_ != nullptr)
    (void)cudaStreamDestroy(stream_);
}

void DeviceOrdinaryFTimeline::destroy_graphs() noexcept {
  if (forward_graph_ != nullptr)
    (void)cudaGraphExecDestroy(forward_graph_);
  if (device_forward_graph_ != nullptr && device_forward_graph_ != forward_graph_)
    (void)cudaGraphExecDestroy(device_forward_graph_);
  if (device_leaf_forward_graph_ != nullptr &&
      device_leaf_forward_graph_ != forward_graph_ &&
      device_leaf_forward_graph_ != device_forward_graph_)
    (void)cudaGraphExecDestroy(device_leaf_forward_graph_);
  if (inverse_graph_ != nullptr)
    (void)cudaGraphExecDestroy(inverse_graph_);
  forward_graph_ = nullptr;
  device_forward_graph_ = nullptr;
  device_leaf_forward_graph_ = nullptr;
  inverse_graph_ = nullptr;
}

cudaGraphExec_t DeviceOrdinaryFTimeline::capture_forward_graph(
    std::uint32_t tick_capacity, bool append_return) {
  if (tick_capacity == 0u || tick_capacity > kGraphTickCapacity)
    throw std::invalid_argument(
        "ordinary-F forward graph tick capacity is invalid");
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t executable = nullptr;
  check_cuda(cudaStreamBeginCapture(stream_, cudaStreamCaptureModeGlobal),
             "begin ordinary-F forward graph capture");
  for (std::uint32_t tick = 0u; tick < tick_capacity; ++tick) {
    begin_forward_kernel<<<1u, kThreads, 0, stream_>>>(device_state_);
    preflight_forward_kernel<<<1u, kThreads, 0, stream_>>>(device_state_);
    capture_boundary_kernel<<<fixed_blocks(launch_capacity_), kThreads, 0, stream_>>>(device_state_,
                                                                                      0u);
    const DeviceActiveSupportWindow window{device_window_, host_state_.dispatch_count,
                                           device_resolution_, launch_capacity_};
    executor_.graph_safe_prepare_active_window(host_state_.slots, host_state_.dispatch_count,
                                               device_window_, launch_capacity_, stream_);
    refresh_domain_kernel<<<1u, kThreads, 0, stream_>>>(device_state_, 0u, true);
    capture_boundary_kernel<<<fixed_blocks(launch_capacity_), kThreads, 0, stream_>>>(device_state_,
                                                                                      1u);
    executor_.graph_safe_prepare_active_window(host_state_.slots, host_state_.dispatch_count,
                                               device_window_, launch_capacity_, stream_);
    for (std::uint32_t index = 0u; index < kForwardFactorCount; ++index) {
      const LawFactor factor = forward_factor(index);
      if (factor == LawFactor::carrier_pair_splitter) {
        executor_.graph_safe_apply_active_macro_factors(chunks_, false, window, stream_);
        refresh_domain_kernel<<<1u, kThreads, 0, stream_>>>(
            device_state_, kOrdinarySpatialMacroClosureRadius, true);
        capture_boundary_kernel<<<fixed_blocks(launch_capacity_), kThreads, 0, stream_>>>(
            device_state_, 4u);
        executor_.graph_safe_prepare_active_window(host_state_.slots, host_state_.dispatch_count,
                                                   device_window_, launch_capacity_, stream_);
        continue;
      }
      if (factor == LawFactor::carrier_corner || factor == LawFactor::processive_rearm ||
          factor == LawFactor::processive_release)
        continue;
      if (factor == LawFactor::edge) {
        // Dynamic host F expands the edge candidate set by one topology hop
        // without filtering Q before K_edge.  This temporary domain is not a
        // journal boundary; the post-edge refresh below is boundary 3.
        refresh_domain_kernel<<<1u, kThreads, 0, stream_>>>(device_state_, 1u, false);
        executor_.graph_safe_prepare_active_window(host_state_.slots, host_state_.dispatch_count,
                                                   device_window_, launch_capacity_, stream_);
        executor_.graph_safe_apply_active_factor(chunks_, factor, false, window, stream_);
        refresh_domain_kernel<<<1u, kThreads, 0, stream_>>>(device_state_, 1u, true);
        capture_boundary_kernel<<<fixed_blocks(launch_capacity_), kThreads, 0, stream_>>>(
            device_state_, 3u);
        executor_.graph_safe_prepare_active_window(host_state_.slots, host_state_.dispatch_count,
                                                   device_window_, launch_capacity_, stream_);
        continue;
      }
      executor_.graph_safe_apply_active_factor(chunks_, factor, false, window, stream_);
      if (factor == LawFactor::developmental_append) {
        refresh_domain_kernel<<<1u, kThreads, 0, stream_>>>(device_state_,
                                                            kSpatialMacroClosureRadius, true);
        capture_boundary_kernel<<<fixed_blocks(launch_capacity_), kThreads, 0, stream_>>>(
            device_state_, 2u);
        executor_.graph_safe_prepare_active_window(host_state_.slots, host_state_.dispatch_count,
                                                   device_window_, launch_capacity_, stream_);
      } else if (factor == LawFactor::stream) {
        refresh_domain_kernel<<<1u, kThreads, 0, stream_>>>(device_state_, 1u, true);
        capture_boundary_kernel<<<fixed_blocks(launch_capacity_), kThreads, 0, stream_>>>(
            device_state_, 5u);
      }
    }
    end_forward_kernel<<<1u, kThreads, 0, stream_>>>(device_state_);
  }
  if (append_return) {
    // The device-launched production child must publish its terminal state
    // before returning control to the resident root. On capacity overflow
    // publish_world() preserves work_count as active_count even though the
    // executable support window was deliberately cleared, so the parent sees
    // the exact attempted physical population rather than a stale/empty
    // publication.
    publish_world_kernel<<<1u, 1u, 0, stream_>>>(device_state_, true);
    return_to_parent_kernel<<<1u, 1u, 0, stream_>>>(device_state_);
  }
  check_cuda(cudaGetLastError(), "capture ordinary-F forward graph kernels");
  check_cuda(cudaStreamEndCapture(stream_, &graph), "end ordinary-F forward graph capture");
  check_cuda(cudaGraphInstantiateWithFlags(
                 &executable, graph, cudaGraphInstantiateFlagDeviceLaunch),
             "instantiate ordinary-F forward graph");
  check_cuda(cudaGraphDestroy(graph), "destroy ordinary-F forward graph");
  check_cuda(cudaGraphUpload(executable, stream_),
             "upload device-launchable ordinary-F forward graph");
  return executable;
}

void DeviceOrdinaryFTimeline::capture_inverse_graph() {
  cudaGraph_t graph = nullptr;
  check_cuda(cudaStreamBeginCapture(stream_, cudaStreamCaptureModeGlobal),
             "begin ordinary-F inverse graph capture");
  for (std::uint32_t tick = 0u; tick < kGraphTickCapacity; ++tick) {
    begin_inverse_kernel<<<1u, kThreads, 0, stream_>>>(device_state_);
    restore_boundary_kernel<<<fixed_blocks(launch_capacity_), kThreads, 0, stream_>>>(device_state_,
                                                                                      5u);
    const DeviceActiveSupportWindow window{device_window_, host_state_.dispatch_count,
                                           device_resolution_, launch_capacity_};
    executor_.graph_safe_prepare_active_window(host_state_.slots, host_state_.dispatch_count,
                                               device_window_, launch_capacity_, stream_);
    for (std::uint32_t index = kForwardFactorCount; index > 0u; --index) {
      const LawFactor factor = forward_factor(index - 1u);
      if (factor == LawFactor::stream) {
        executor_.graph_safe_apply_active_factor(chunks_, factor, true, window, stream_);
        restore_boundary_kernel<<<fixed_blocks(launch_capacity_), kThreads, 0, stream_>>>(
            device_state_, 4u);
        executor_.graph_safe_prepare_active_window(host_state_.slots, host_state_.dispatch_count,
                                                   device_window_, launch_capacity_, stream_);
      } else if (factor == LawFactor::developmental_credit_service) {
        executor_.graph_safe_prepare_active_window(host_state_.slots, host_state_.dispatch_count,
                                                   device_window_, launch_capacity_, stream_);
        executor_.graph_safe_apply_active_factor(chunks_, factor, true, window, stream_);
      } else if (factor == LawFactor::prediction_residual_route_toggle) {
        executor_.graph_safe_apply_active_factor(chunks_, factor, true, window, stream_);
        restore_boundary_kernel<<<fixed_blocks(launch_capacity_), kThreads, 0, stream_>>>(
            device_state_, 3u);
        executor_.graph_safe_prepare_active_window(host_state_.slots, host_state_.dispatch_count,
                                                   device_window_, launch_capacity_, stream_);
      } else if (factor == LawFactor::processive_release) {
        executor_.graph_safe_apply_active_macro_factors(chunks_, true, window, stream_);
        restore_boundary_kernel<<<fixed_blocks(launch_capacity_), kThreads, 0, stream_>>>(
            device_state_, 2u);
        executor_.graph_safe_prepare_active_window(host_state_.slots, host_state_.dispatch_count,
                                                   device_window_, launch_capacity_, stream_);
      } else if (factor == LawFactor::carrier_pair_splitter ||
                 factor == LawFactor::carrier_corner || factor == LawFactor::processive_rearm) {
        continue;
      } else if (factor == LawFactor::edge) {
        refresh_domain_kernel<<<1u, kThreads, 0, stream_>>>(device_state_, 1u, false);
        executor_.graph_safe_prepare_active_window(host_state_.slots, host_state_.dispatch_count,
                                                   device_window_, launch_capacity_, stream_);
        executor_.graph_safe_apply_active_factor(chunks_, factor, true, window, stream_);
        restore_boundary_kernel<<<fixed_blocks(launch_capacity_), kThreads, 0, stream_>>>(
            device_state_, 2u);
        executor_.graph_safe_prepare_active_window(host_state_.slots, host_state_.dispatch_count,
                                                   device_window_, launch_capacity_, stream_);
      } else if (factor == LawFactor::site || factor == LawFactor::eligibility_residual_junction) {
        executor_.graph_safe_apply_active_factor(chunks_, factor, true, window, stream_);
      } else if (factor == LawFactor::developmental_append) {
        restore_boundary_kernel<<<fixed_blocks(launch_capacity_), kThreads, 0, stream_>>>(
            device_state_, 1u);
        executor_.graph_safe_prepare_active_window(host_state_.slots, host_state_.dispatch_count,
                                                   device_window_, launch_capacity_, stream_);
        executor_.graph_safe_apply_active_factor(chunks_, factor, true, window, stream_);
      } else if (factor == LawFactor::developmental_learned_receptor) {
        executor_.graph_safe_apply_active_factor(chunks_, factor, true, window, stream_);
        restore_boundary_kernel<<<fixed_blocks(launch_capacity_), kThreads, 0, stream_>>>(
            device_state_, 0u);
      }
    }
    end_inverse_kernel<<<1u, kThreads, 0, stream_>>>(device_state_);
  }
  check_cuda(cudaGetLastError(), "capture ordinary-F inverse graph kernels");
  check_cuda(cudaStreamEndCapture(stream_, &graph), "end ordinary-F inverse graph capture");
  check_cuda(cudaGraphInstantiate(&inverse_graph_, graph, nullptr, nullptr, 0),
             "instantiate ordinary-F inverse graph");
  check_cuda(cudaGraphDestroy(graph), "destroy ordinary-F inverse graph");
}

void DeviceOrdinaryFTimeline::bootstrap() {
  if (forward_graph_ != nullptr && device_forward_graph_ != nullptr &&
      inverse_graph_ != nullptr)
    return;
  destroy_graphs();
  // Host assays retain the original 65-body capture, while device-tail
  // production uses the same factor body captured for one requested tick.
  // The first-tick parity contract compares both shapes explicitly.
  forward_graph_ = capture_forward_graph(kGraphTickCapacity, false);
  device_forward_graph_ = capture_forward_graph(1u, true);
  capture_inverse_graph();
  last_status_ = "ordinary_f_timeline=BOOTSTRAPPED device_frontier=1";
}

DeviceLaunchHandle DeviceOrdinaryFTimeline::device_launch_handle() {
  bootstrap();
  DeviceLaunchHandle handle{};
  handle.forward_graph = device_forward_graph_;
  handle.return_graph_slot = reinterpret_cast<cudaGraphExec_t*>(
      reinterpret_cast<std::uint8_t*>(device_state_) +
      offsetof(DeviceState, return_graph));
  handle.requested_target = host_state_.requested_target;
  handle.completed_ticks = host_state_.completed_ticks;
  handle.fault = host_state_.fault;
  handle.publication = host_state_.publication;
  handle.words = host_state_.words;
  handle.mutable_words = executor_.mutable_device_words();
  handle.active_slots = host_state_.slots;
  handle.active_count = host_state_.count;
  handle.capacity = host_state_.capacity;
  handle.chunk_count = host_state_.topology.chunks.chunk_count;
  handle.boundary = boundary_;
  return handle;
}

DeviceLaunchHandle DeviceOrdinaryFTimeline::device_leaf_launch_handle() {
  // The resident device tail needs only a one-tick leaf. Do not bootstrap the
  // host parity graphs or inverse journal here: that captures 65 forward
  // ticks plus a reverse graph and can exhaust a production GPU before the
  // canonical root even receives the handle.
  if (device_leaf_forward_graph_ == nullptr) {
    // The resident root owns the continuation. Its epoch node writes the
    // exact current root executable into return_graph_slot, then enqueues
    // this one-tick leaf. The leaf publishes its commit and tail-launches that
    // exact root graph; no sibling or host scheduler is involved.
    device_leaf_forward_graph_ = capture_forward_graph(1u, true);
    // capture_forward_graph uploads the device-launchable executable on this
    // timeline's private stream. Export the handle only after that upload is
    // complete; the resident root will supply its own exact return graph at
    // each device handoff.
    check_cuda(cudaStreamSynchronize(stream_),
               "synchronize resident ordinary-F device leaf upload");
    last_status_ = "ordinary_f_timeline=DEVICE_LEAF_BOOTSTRAPPED";
  }
  DeviceLaunchHandle handle{};
  handle.forward_graph = device_leaf_forward_graph_;
  handle.return_graph_slot = reinterpret_cast<cudaGraphExec_t*>(
      reinterpret_cast<std::uint8_t*>(device_state_) +
      offsetof(DeviceState, return_graph));
  handle.requested_target = host_state_.requested_target;
  handle.completed_ticks = host_state_.completed_ticks;
  handle.fault = host_state_.fault;
  handle.publication = host_state_.publication;
  handle.words = host_state_.words;
  handle.mutable_words = executor_.mutable_device_words();
  handle.active_slots = host_state_.slots;
  handle.active_count = host_state_.count;
  handle.capacity = host_state_.capacity;
  handle.chunk_count = host_state_.topology.chunks.chunk_count;
  handle.boundary = boundary_;
  return handle;
}

void DeviceOrdinaryFTimeline::attach_return_graph(cudaGraphExec_t parent_graph) {
  if (parent_graph == nullptr)
    throw std::invalid_argument("ordinary-F return graph is null");
  // The resident path deliberately owns only the bounded device leaf. Do not
  // bootstrap the host parity and inverse graphs here: doing so defeats the
  // bounded shell and can exhaust the GPU before the root receives its child
  // handle. The public host-driver path has already called
  // device_launch_handle(), which bootstraps the full graph set.
  if (forward_graph_ == nullptr && device_forward_graph_ == nullptr &&
      device_leaf_forward_graph_ == nullptr)
    bootstrap();
  host_state_.return_graph = parent_graph;
  auto* destination = reinterpret_cast<std::uint8_t*>(device_state_) +
                      offsetof(DeviceState, return_graph);
  check_cuda(cudaMemcpyAsync(destination, &parent_graph, sizeof(parent_graph),
                             cudaMemcpyHostToDevice, stream_),
             "attach ordinary-F return graph");
  check_cuda(cudaStreamSynchronize(stream_),
             "synchronize ordinary-F return graph attachment");
}

void DeviceOrdinaryFTimeline::submit(cudaGraphExec_t graph, std::uint64_t target,
                                     const char* operation) {
  if (graph == nullptr)
    bootstrap();
  // The host shadow contains immutable device addresses established at
  // bootstrap. Runtime support/count authority never crosses this boundary;
  // only the requested generation target and passive receipts do.
  check_cuda(cudaMemcpyAsync(host_state_.requested_target, &target, sizeof(target),
                             cudaMemcpyHostToDevice, stream_),
             "upload ordinary-F timeline target");
  check_cuda(cudaGraphLaunch(graph, stream_), operation);
  // The graph and publication share one stream, so ordering does not require
  // an intermediate host barrier. Synchronize only when the host consumes the
  // terminal publication below.
  publish_world_kernel<<<1u, 1u, 0, stream_>>>(device_state_, false);
  check_cuda(cudaGetLastError(), "launch ordinary-F post-graph publication");
  check_cuda(cudaStreamSynchronize(stream_),
             "synchronize ordinary-F post-graph publication");
  refresh_host_receipt();
  last_status_ = "ordinary_f_timeline=GREEN device_frontier=1 graph_launch=1";
}

void DeviceOrdinaryFTimeline::refresh_host_receipt() {
  const std::uint64_t generation_before = host_generation_;
  // One packed control receipt replaces five scattered blocking reads. The
  // device kernel stages the scalars into the publication allocation's
  // observer-only tail (same overlay snapshot() uses), so the five fields are
  // mutually consistent by construction and cross the bus exactly once.
  SnapshotControlReceipt receipt{};
  capture_snapshot_control_kernel<<<1u, 1u, 0, stream_>>>(device_state_);
  check_cuda(cudaGetLastError(), "launch ordinary-F timeline control receipt");
  check_cuda(cudaMemcpyAsync(&receipt, host_state_.publication, sizeof(receipt),
                             cudaMemcpyDeviceToHost, stream_),
             "read ordinary-F timeline control receipt");
  check_cuda(cudaStreamSynchronize(stream_),
             "synchronize ordinary-F timeline control receipt");
  host_fault_ = static_cast<Fault>(receipt.fault);
  host_completed_ticks_ = receipt.completed_ticks;
  host_inverse_head_ = receipt.inverse_head;
  host_inverse_depth_ = receipt.inverse_depth;
  host_generation_ = receipt.generation;
  if (host_generation_ < generation_before)
    throw std::runtime_error("ordinary-F generation receipt regressed");
  executor_.note_world_advances(host_generation_ - generation_before);
  if (host_fault_ != Fault::none) {
    last_status_ =
        "ordinary_f_timeline=RED fault=" + std::to_string(static_cast<std::uint32_t>(host_fault_));
    throw std::runtime_error(last_status_);
  }
}

void DeviceOrdinaryFTimeline::forward(std::uint64_t ticks) {
  if (ticks == 0u)
    return;
  bootstrap();
  refresh_host_receipt();
  if (ticks > kGraphTickCapacity ||
      ticks > std::numeric_limits<std::uint64_t>::max() - host_completed_ticks_)
    throw std::overflow_error("ordinary-F timeline forward command boundary");
  // One submit retires the whole command. The captured graph carries
  // kGraphTickCapacity tick bodies and begin_forward_kernel gates every body
  // on fault == none && completed_ticks < requested_target, so arming
  // completed + ticks executes exactly `ticks` sequential bodies here -- the
  // same kernel sequence per retired tick as one-submit-per-tick, with the
  // identical terminal publication, receipts, and RED throw on the first
  // faulting tick (later bodies are the same gated no-ops that already run
  // inside today's faulting launch).
  submit(forward_graph_, host_completed_ticks_ + ticks,
         "launch ordinary-F forward timeline graph");
}

void DeviceOrdinaryFTimeline::inverse(std::uint64_t ticks) {
  if (ticks == 0u)
    return;
  bootstrap();
  refresh_host_receipt();
  if (ticks > kGraphTickCapacity || ticks > host_completed_ticks_ || ticks > host_inverse_depth_)
    throw std::invalid_argument("ordinary-F timeline inverse history boundary");
  submit(inverse_graph_, host_completed_ticks_ - ticks, "launch ordinary-F inverse timeline graph");
}

void DeviceOrdinaryFTimeline::reacquire_world_support() {
  refresh_host_receipt();
  if (host_fault_ != Fault::none)
    throw std::logic_error("ordinary-F world support reacquisition crosses a fault");
  std::uint32_t executing = 0u;
  check_cuda(
      cudaMemcpy(&executing, host_state_.executing, sizeof(executing), cudaMemcpyDeviceToHost),
      "read ordinary-F reacquisition idle boundary");
  if (executing != 0u)
    throw std::logic_error("ordinary-F world support reacquisition requires idle boundary");

  rebuild_support_from_world(false);
  std::uint32_t fault = 0u;
  check_cuda(cudaMemcpy(&fault, host_state_.fault, sizeof(fault), cudaMemcpyDeviceToHost),
             "read ordinary-F support reacquisition fault");
  host_fault_ = static_cast<Fault>(fault);
  if (host_fault_ != Fault::none) {
    last_status_ = "ordinary_f_timeline=RED support_reacquisition_fault=" + std::to_string(fault);
    throw std::runtime_error(last_status_);
  }
  host_inverse_depth_ = 0u;
  last_status_ = "ordinary_f_timeline=GREEN device_world_support_reacquired=1";
}

HostSnapshot DeviceOrdinaryFTimeline::snapshot() const {
  const DeviceState& state = host_state_;
  HostSnapshot result;
  result.capacity = launch_capacity_;
  result.world_words.resize(executor_.site_count());
  SnapshotControlReceipt control{};
  executor_.download_words(0u, result.world_words, stream_);
  capture_snapshot_control_kernel<<<1u, 1u, 0, stream_>>>(device_state_);
  check_cuda(cudaGetLastError(), "capture ordinary-F snapshot control receipt");
  check_cuda(cudaMemcpyAsync(&control, state.publication,
                             sizeof(SnapshotControlReceipt), cudaMemcpyDeviceToHost,
                             stream_),
             "enqueue ordinary-F snapshot control receipt");
  check_cuda(cudaStreamSynchronize(stream_),
             "synchronize ordinary-F checkpoint receipt");
  result.completed_ticks = control.completed_ticks;
  result.inverse_head = control.inverse_head;
  result.inverse_depth = control.inverse_depth;
  result.fault = static_cast<Fault>(control.fault);
  result.generation = control.generation;
  result.publication = control.publication;
  if (control.executing != 0u)
    throw std::logic_error("ordinary-F checkpoint requires idle boundary");
  const std::uint32_t fault = control.fault;
  const std::uint32_t count = control.active_count;
  const std::uint32_t published_count =
      fault == static_cast<std::uint32_t>(Fault::capacity_overflow)
          ? control.attempted_count
          : count;
  if (count > launch_capacity_ || result.inverse_head >= history_capacity_ ||
      result.inverse_depth > history_capacity_)
    throw std::runtime_error("ordinary-F timeline snapshot ring invalid");
  if (result.publication.completed_ticks != result.completed_ticks ||
      result.publication.generation != result.generation ||
      result.publication.fault != fault ||
      result.publication.active_count != published_count)
    throw std::runtime_error(
        "ordinary-F timeline publication contradicts control state: ticks=" +
        std::to_string(result.publication.completed_ticks) + "/" +
        std::to_string(result.completed_ticks) + " generation=" +
        std::to_string(result.publication.generation) + "/" +
        std::to_string(result.generation) + " fault=" +
        std::to_string(result.publication.fault) + "/" + std::to_string(fault) +
        " active=" + std::to_string(result.publication.active_count) + "/" +
        std::to_string(published_count) + " live=" + std::to_string(count));
  result.active_slots.resize(count);
  if (count != 0u)
    check_cuda(cudaMemcpy(result.active_slots.data(), state.slots, count * sizeof(std::uint64_t),
                          cudaMemcpyDeviceToHost),
               "read ordinary-F timeline active support");
  result.boundary_counts.resize(static_cast<std::size_t>(result.inverse_depth) * kBoundaryCount);
  std::vector<std::uint32_t> physical_counts(static_cast<std::size_t>(history_capacity_) *
                                             kBoundaryCount);
  check_cuda(cudaMemcpy(physical_counts.data(), state.history_counts,
                        physical_counts.size() * sizeof(std::uint32_t), cudaMemcpyDeviceToHost),
             "read ordinary-F timeline physical boundary counts");
  for (std::uint32_t logical = 0u; logical < result.inverse_depth; ++logical) {
    const std::uint32_t physical =
        (result.inverse_head + history_capacity_ - result.inverse_depth + logical) %
        history_capacity_;
    for (std::uint32_t boundary = 0u; boundary < kBoundaryCount; ++boundary)
      result.boundary_counts[static_cast<std::size_t>(logical) * kBoundaryCount + boundary] =
          physical_counts[static_cast<std::size_t>(physical) * kBoundaryCount + boundary];
  }
  result.boundary_slots.resize(result.boundary_counts.size());
  for (std::size_t frame = 0u; frame < result.boundary_counts.size(); ++frame) {
    const std::uint32_t frame_count = result.boundary_counts[frame];
    if (frame_count > launch_capacity_)
      throw std::runtime_error("ordinary-F timeline snapshot count invalid");
    result.boundary_slots[frame].resize(frame_count);
    if (frame_count != 0u) {
      const std::uint32_t logical = static_cast<std::uint32_t>(frame / kBoundaryCount);
      const std::uint32_t boundary = static_cast<std::uint32_t>(frame % kBoundaryCount);
      const std::uint32_t physical =
          (result.inverse_head + history_capacity_ - result.inverse_depth + logical) %
          history_capacity_;
      std::vector<std::uint32_t> compact_slots(frame_count);
      check_cuda(cudaMemcpy(compact_slots.data(),
                            state.history_slots +
                                (static_cast<std::size_t>(physical) *
                                     kBoundaryCount +
                                 boundary) *
                                    launch_capacity_,
                            frame_count * sizeof(std::uint32_t),
                            cudaMemcpyDeviceToHost),
                 "read ordinary-F timeline boundary support");
      std::transform(compact_slots.begin(), compact_slots.end(),
                     result.boundary_slots[frame].begin(),
                     [](std::uint32_t slot) {
                       return static_cast<std::uint64_t>(slot);
                     });
    }
  }
  return result;
}

void DeviceOrdinaryFTimeline::restore(const HostSnapshot& snapshot) {
  const std::uint32_t fault_value = static_cast<std::uint32_t>(snapshot.fault);
  const bool valid_timeline =
      snapshot.inverse_head < history_capacity_ && snapshot.inverse_depth <= history_capacity_ &&
      snapshot.inverse_depth <= snapshot.completed_ticks &&
      snapshot.inverse_head == snapshot.completed_ticks % history_capacity_ &&
      snapshot.generation >= snapshot.completed_ticks &&
      ((snapshot.generation - snapshot.completed_ticks) & 1u) == 0u &&
      fault_value <= static_cast<std::uint32_t>(Fault::capacity_overflow);
  if (snapshot.capacity != launch_capacity_ ||
      snapshot.world_words.size() != executor_.site_count() || !valid_timeline ||
      snapshot.active_slots.size() > launch_capacity_ ||
      snapshot.boundary_counts.size() !=
          static_cast<std::size_t>(snapshot.inverse_depth) * kBoundaryCount ||
      snapshot.boundary_slots.size() != snapshot.boundary_counts.size())
    throw std::invalid_argument("ordinary-F timeline checkpoint shape invalid");
  const auto valid_ordered_slots = [&](const auto& slots) {
    if (!std::is_sorted(slots.begin(), slots.end()) ||
        std::adjacent_find(slots.begin(), slots.end()) != slots.end())
      return false;
    return std::all_of(slots.begin(), slots.end(),
                       [&](std::uint64_t slot) { return slot < executor_.site_count(); });
  };
  if (!valid_ordered_slots(snapshot.active_slots))
    throw std::invalid_argument("ordinary-F timeline checkpoint active support invalid");
  std::vector<std::uint64_t> checkpoint_world_support;
  checkpoint_world_support.reserve(snapshot.active_slots.size());
  for (std::uint64_t slot = 0u; slot < snapshot.world_words.size(); ++slot)
    if (snapshot.world_words[slot] != kQ)
      checkpoint_world_support.push_back(slot);
  if (checkpoint_world_support != snapshot.active_slots)
    throw std::invalid_argument(
        "ordinary-F timeline checkpoint support contradicts its world");
  if (snapshot.publication.completed_ticks != snapshot.completed_ticks ||
      snapshot.publication.generation != snapshot.generation ||
      snapshot.publication.fault != fault_value ||
      snapshot.publication.active_count !=
          static_cast<std::uint32_t>(snapshot.active_slots.size()) ||
      snapshot.publication.world !=
          host_world_digest(chunks_.chunk_count, launch_capacity_,
                            snapshot.world_words, snapshot.active_slots))
    throw std::invalid_argument(
        "ordinary-F timeline checkpoint publication is invalid");
  for (std::size_t frame = 0u; frame < snapshot.boundary_counts.size(); ++frame) {
    if (snapshot.boundary_counts[frame] > launch_capacity_ ||
        snapshot.boundary_slots[frame].size() != snapshot.boundary_counts[frame] ||
        !valid_ordered_slots(snapshot.boundary_slots[frame]))
      throw std::invalid_argument("ordinary-F timeline checkpoint frame invalid");
  }
  const DeviceState& state = host_state_;
  std::uint32_t executing = 0u;
  check_cuda(cudaMemcpy(&executing, state.executing, sizeof(executing), cudaMemcpyDeviceToHost),
             "read ordinary-F restore idle boundary");
  if (executing != 0u)
    throw std::logic_error("ordinary-F restore requires idle boundary");

  // The serialized active list is evidence, never reconstruction authority.
  // The host-side comparison above is a fail-before-write shape check only.
  // Actual resident support is still derived from the uploaded world on device
  // and checked again before any timeline metadata is restored.
  executor_.upload_words(0u, snapshot.world_words, stream_);
  check_cuda(cudaStreamSynchronize(stream_), "synchronize restored ordinary-F world");
  rebuild_support_from_world(false);
  std::uint32_t scan_fault = 0u;
  std::uint32_t derived_count = 0u;
  check_cuda(cudaMemcpy(&scan_fault, state.fault, sizeof(scan_fault), cudaMemcpyDeviceToHost),
             "read restored ordinary-F device scan fault");
  check_cuda(cudaMemcpy(&derived_count, state.count, sizeof(derived_count), cudaMemcpyDeviceToHost),
             "read restored ordinary-F device support count");
  if (scan_fault != static_cast<std::uint32_t>(Fault::none) || derived_count > launch_capacity_)
    throw std::invalid_argument("ordinary-F checkpoint world exceeds support capacity");
  std::vector<std::uint64_t> derived_support(derived_count);
  if (derived_count != 0u)
    check_cuda(cudaMemcpy(derived_support.data(), state.slots,
                          derived_count * sizeof(std::uint64_t), cudaMemcpyDeviceToHost),
               "read restored ordinary-F device-derived support");
  if (derived_support != snapshot.active_slots)
    throw std::invalid_argument("ordinary-F checkpoint support disagrees with device world scan");
  OrdinaryFPublication derived_publication{};
  check_cuda(cudaMemcpy(&derived_publication, state.publication,
                        sizeof(derived_publication), cudaMemcpyDeviceToHost),
             "read restored ordinary-F device publication");
  if (derived_publication.active_count != snapshot.publication.active_count ||
      derived_publication.world != snapshot.publication.world)
    throw std::invalid_argument(
        "ordinary-F checkpoint digest disagrees with device world scan");

  std::vector<std::uint32_t> physical_counts(
      static_cast<std::size_t>(history_capacity_) * kBoundaryCount, 0u);
  for (std::size_t frame = 0u; frame < snapshot.boundary_counts.size(); ++frame) {
    const std::uint32_t logical = static_cast<std::uint32_t>(frame / kBoundaryCount);
    const std::uint32_t boundary = static_cast<std::uint32_t>(frame % kBoundaryCount);
    const std::uint32_t physical =
        (snapshot.inverse_head + history_capacity_ - snapshot.inverse_depth + logical) %
        history_capacity_;
    const std::size_t physical_boundary =
        static_cast<std::size_t>(physical) * kBoundaryCount + boundary;
    physical_counts[physical_boundary] = snapshot.boundary_counts[frame];
    if (!snapshot.boundary_slots[frame].empty()) {
      std::vector<std::uint32_t> compact_slots(
          snapshot.boundary_slots[frame].size());
      std::transform(snapshot.boundary_slots[frame].begin(),
                     snapshot.boundary_slots[frame].end(),
                     compact_slots.begin(), [](std::uint64_t slot) {
                       return static_cast<std::uint32_t>(slot);
                     });
      check_cuda(cudaMemcpy(
                     state.history_slots +
                         physical_boundary * launch_capacity_,
                     compact_slots.data(),
                     compact_slots.size() * sizeof(std::uint32_t),
                     cudaMemcpyHostToDevice),
                 "restore ordinary-F logical boundary support");
    }
  }
  check_cuda(cudaMemcpy(state.history_counts, physical_counts.data(),
                        physical_counts.size() * sizeof(std::uint32_t), cudaMemcpyHostToDevice),
             "restore ordinary-F physical boundary counts");
  check_cuda(cudaMemcpy(state.inverse_head, &snapshot.inverse_head, sizeof(snapshot.inverse_head),
                        cudaMemcpyHostToDevice),
             "restore ordinary-F inverse ring head");
  check_cuda(cudaMemcpy(state.inverse_depth, &snapshot.inverse_depth,
                        sizeof(snapshot.inverse_depth), cudaMemcpyHostToDevice),
             "restore ordinary-F inverse ring depth");
  check_cuda(cudaMemcpy(state.completed_ticks, &snapshot.completed_ticks,
                        sizeof(snapshot.completed_ticks), cudaMemcpyHostToDevice),
             "restore ordinary-F completed ticks");
  check_cuda(cudaMemcpy(state.fault, &fault_value, sizeof(fault_value), cudaMemcpyHostToDevice),
             "restore ordinary-F fault");
  check_cuda(cudaMemcpy(state.generation, &snapshot.generation, sizeof(snapshot.generation),
                        cudaMemcpyHostToDevice),
             "restore ordinary-F generation");
  const std::uint32_t zero = 0u;
  check_cuda(cudaMemcpy(state.executing, &zero, sizeof(zero), cudaMemcpyHostToDevice),
             "restore ordinary-F executing flag");
  check_cuda(cudaMemcpy(state.current_frame, &snapshot.inverse_head, sizeof(snapshot.inverse_head),
                        cudaMemcpyHostToDevice),
             "restore ordinary-F current inverse frame");
  check_cuda(cudaMemcpy(state.requested_target, &snapshot.completed_ticks,
                        sizeof(snapshot.completed_ticks), cudaMemcpyHostToDevice),
             "restore ordinary-F requested target");
  publish_world_kernel<<<1u, 1u, 0, stream_>>>(device_state_, false);
  check_cuda(cudaGetLastError(),
             "launch restored ordinary-F world publication");
  check_cuda(cudaStreamSynchronize(stream_),
             "synchronize restored ordinary-F world publication");
  host_completed_ticks_ = snapshot.completed_ticks;
  host_inverse_head_ = snapshot.inverse_head;
  host_inverse_depth_ = snapshot.inverse_depth;
  host_fault_ = snapshot.fault;
  host_generation_ = snapshot.generation;
  destroy_graphs();
  bootstrap();
}

}  // namespace substrate::bcc32::device_ordinary_f_timeline
