#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <limits>

#include "bcc32_law.cuh"

namespace substrate::bcc32::device_resident_frontier {

inline constexpr std::uint64_t kNoSlot =
    std::numeric_limits<std::uint64_t>::max();
inline constexpr std::uint32_t kKernelThreads = 256u;
inline constexpr std::uint32_t kProductionCapacity = 4096u;

enum class Fault : std::uint32_t {
  none = 0u,
  invalid_input = 1u,
  capacity_overflow = 2u,
};

// The topology is generic physical adjacency. Its storage and lifetime belong
// to the enclosing device organism; this view carries no region or port name.
// sites is a strictly ascending compact aperture of global site ids. This
// avoids materializing world_site_count * neighbor_stride entries for a paged
// world while keeping every neighbor as an ordinary global physical address.
struct TopologyView {
  const std::uint64_t* sites = nullptr;
  const std::uint64_t* neighbors = nullptr;
  std::uint64_t neighbor_entry_count = 0u;
  std::uint64_t world_site_count = 0u;
  std::uint32_t entry_count = 0u;
  std::uint32_t neighbor_stride = 0u;
};

// The word field is the ordinary-F authority. Pinned slots are raw physical
// contacts that must remain scheduled even when their current word is Q; they
// are not semantic receptors and must be supplied by resident boundary state.
struct ActivityView {
  const SiteWord* words = nullptr;
  std::uint64_t word_count = 0u;
  const std::uint64_t* pinned = nullptr;
  std::uint32_t pinned_count = 0u;
};

template <std::uint32_t Capacity>
struct State {
  static_assert(Capacity != 0u);
  static_assert((Capacity & (Capacity - 1u)) == 0u,
                "frontier capacity must be a power of two");

  std::uint64_t committed[Capacity]{};

  // Transient device work matter. Neither array is an authority surface.
  unsigned long long hash_slots[Capacity * 2u]{};
  std::uint64_t scratch[Capacity]{};
  std::uint32_t committed_count = 0u;
  std::uint32_t scratch_count = 0u;
  std::uint32_t unique_count = 0u;
  std::uint32_t fault = static_cast<std::uint32_t>(Fault::none);
  std::uint64_t generation = 0u;
  std::uint64_t revision = 0u;
  std::uint64_t attempts = 0u;
  std::uint64_t overflow_count = 0u;
};

// A checkpoint contains all durable frontier authority. Hash and sorting
// buffers are transient implementation matter and are rebuilt before use.
template <std::uint32_t Capacity>
struct Snapshot {
  std::uint64_t committed[Capacity]{};
  std::uint32_t committed_count = 0u;
  std::uint32_t fault = static_cast<std::uint32_t>(Fault::none);
  std::uint64_t generation = 0u;
  std::uint64_t revision = 0u;
  std::uint64_t attempts = 0u;
  std::uint64_t overflow_count = 0u;
};

using ProductionState = State<kProductionCapacity>;
using ProductionSnapshot = Snapshot<kProductionCapacity>;

[[nodiscard]] __host__ __device__ constexpr bool durable_slot_valid(
    std::uint64_t slot, std::uint64_t site_count) {
  return slot != kNoSlot && slot < site_count;
}

[[nodiscard]] __device__ inline std::uint64_t mix_slot(std::uint64_t value) {
  value ^= value >> 30u;
  value *= 0xbf58476d1ce4e5b9ull;
  value ^= value >> 27u;
  value *= 0x94d049bb133111ebull;
  value ^= value >> 31u;
  return value;
}

template <std::uint32_t Capacity>
__device__ inline void publish_fault(State<Capacity>* state, Fault fault) {
  atomicOr(&state->fault, static_cast<std::uint32_t>(fault));
}

[[nodiscard]] __device__ inline std::uint32_t topology_entry_for(
    TopologyView topology, std::uint64_t slot) {
  std::uint32_t lower = 0u;
  std::uint32_t upper = topology.entry_count;
  while (lower < upper) {
    const std::uint32_t middle = lower + (upper - lower) / 2u;
    const std::uint64_t candidate = topology.sites[middle];
    if (candidate < slot)
      lower = middle + 1u;
    else
      upper = middle;
  }
  return lower < topology.entry_count && topology.sites[lower] == slot
             ? lower
             : topology.entry_count;
}

template <std::uint32_t Capacity>
__device__ inline void insert_unique(State<Capacity>* state,
                                     std::uint64_t slot) {
  constexpr std::uint32_t kHashCapacity = Capacity * 2u;
  const std::uint32_t start =
      static_cast<std::uint32_t>(mix_slot(slot)) & (kHashCapacity - 1u);
  for (std::uint32_t probe = 0u; probe < kHashCapacity; ++probe) {
    const std::uint32_t index = (start + probe) & (kHashCapacity - 1u);
    const unsigned long long prior = atomicCAS(
        &state->hash_slots[index], static_cast<unsigned long long>(kNoSlot),
        static_cast<unsigned long long>(slot));
    if (prior == static_cast<unsigned long long>(kNoSlot)) {
      const std::uint32_t count = atomicAdd(&state->unique_count, 1u) + 1u;
      if (count > Capacity) publish_fault(state, Fault::capacity_overflow);
      return;
    }
    if (prior == static_cast<unsigned long long>(slot)) return;
  }
  publish_fault(state, Fault::capacity_overflow);
}

template <std::uint32_t Capacity>
__device__ inline void insert_slot_and_neighbors(
    State<Capacity>* state, TopologyView topology, std::uint64_t slot) {
  if (!durable_slot_valid(slot, topology.world_site_count)) {
    publish_fault(state, Fault::invalid_input);
    return;
  }
  insert_unique(state, slot);

  const std::uint32_t entry = topology_entry_for(topology, slot);
  if (entry == topology.entry_count) {
    publish_fault(state, Fault::invalid_input);
    return;
  }
  const std::uint64_t base =
      static_cast<std::uint64_t>(entry) * topology.neighbor_stride;
  if (topology.neighbor_stride != 0u &&
      (base / topology.neighbor_stride != entry ||
       base > topology.neighbor_entry_count ||
       topology.neighbor_stride > topology.neighbor_entry_count - base)) {
    publish_fault(state, Fault::invalid_input);
    return;
  }
  for (std::uint32_t lane = 0u; lane < topology.neighbor_stride; ++lane) {
    const std::uint64_t neighbor = topology.neighbors[base + lane];
    if (neighbor == kNoSlot) continue;
    if (!durable_slot_valid(neighbor, topology.world_site_count)) {
      publish_fault(state, Fault::invalid_input);
      continue;
    }
    insert_unique(state, neighbor);
  }
}

template <std::uint32_t Capacity>
__device__ inline void prepare_workspace(State<Capacity>* state) {
  const std::uint32_t lane = threadIdx.x;
  if (lane == 0u) {
    ++state->attempts;
    ++state->revision;
    state->fault = static_cast<std::uint32_t>(Fault::none);
    state->unique_count = 0u;
    state->scratch_count = 0u;
  }
  for (std::uint32_t index = lane; index < Capacity * 2u;
       index += blockDim.x) {
    state->hash_slots[index] = static_cast<unsigned long long>(kNoSlot);
  }
  for (std::uint32_t index = lane; index < Capacity;
       index += blockDim.x) {
    state->scratch[index] = kNoSlot;
  }
  __syncthreads();
}

template <std::uint32_t Capacity>
__device__ inline bool commit_workspace(State<Capacity>* state) {
  const std::uint32_t lane = threadIdx.x;
  if (state->fault != static_cast<std::uint32_t>(Fault::none) ||
      state->unique_count > Capacity) {
    if (lane == 0u &&
        (state->fault &
         static_cast<std::uint32_t>(Fault::capacity_overflow)) != 0u) {
      ++state->overflow_count;
    }
    __syncthreads();
    return false;
  }

  for (std::uint32_t index = lane; index < Capacity * 2u;
       index += blockDim.x) {
    const std::uint64_t slot = state->hash_slots[index];
    if (slot == kNoSlot) continue;
    const std::uint32_t output = atomicAdd(&state->scratch_count, 1u);
    if (output < Capacity) state->scratch[output] = slot;
  }
  __syncthreads();

  // A fixed-size bitonic network removes all dependence on insertion or
  // thread scheduling order. kNoSlot sorts after every valid physical slot.
  for (std::uint32_t width = 2u; width <= Capacity; width <<= 1u) {
    for (std::uint32_t stride = width >> 1u; stride != 0u; stride >>= 1u) {
      for (std::uint32_t index = lane; index < Capacity;
           index += blockDim.x) {
        const std::uint32_t partner = index ^ stride;
        if (partner <= index) continue;
        const bool ascending = (index & width) == 0u;
        const std::uint64_t left = state->scratch[index];
        const std::uint64_t right = state->scratch[partner];
        if ((ascending && left > right) || (!ascending && left < right)) {
          state->scratch[index] = right;
          state->scratch[partner] = left;
        }
      }
      __syncthreads();
    }
  }

  for (std::uint32_t index = lane; index < Capacity;
       index += blockDim.x) {
    state->committed[index] =
        index < state->scratch_count ? state->scratch[index] : kNoSlot;
  }
  __syncthreads();
  if (lane == 0u) {
    state->committed_count = state->scratch_count;
    ++state->generation;
  }
  __syncthreads();
  return true;
}

template <std::uint32_t Capacity>
__device__ inline bool advance_once(
    State<Capacity>* state, TopologyView topology,
    const std::uint64_t* proposals, std::uint32_t proposal_count) {
  const std::uint32_t lane = threadIdx.x;
  prepare_workspace(state);

  if (state->committed_count > Capacity || topology.sites == nullptr ||
      topology.neighbors == nullptr || topology.world_site_count == 0u ||
      topology.entry_count == 0u || topology.neighbor_stride == 0u ||
      topology.entry_count >
          topology.neighbor_entry_count / topology.neighbor_stride ||
      (proposal_count != 0u && proposals == nullptr) ||
      proposal_count > Capacity) {
    if (lane == 0u) {
      state->fault = static_cast<std::uint32_t>(Fault::invalid_input);
    }
    __syncthreads();
    return false;
  }

  for (std::uint32_t entry = lane; entry < topology.entry_count;
       entry += blockDim.x) {
    const std::uint64_t slot = topology.sites[entry];
    if (!durable_slot_valid(slot, topology.world_site_count) ||
        (entry != 0u && topology.sites[entry - 1u] >= slot)) {
      publish_fault(state, Fault::invalid_input);
    }
  }
  __syncthreads();
  if (state->fault != static_cast<std::uint32_t>(Fault::none)) return false;

  const std::uint32_t source_count = state->committed_count + proposal_count;
  for (std::uint32_t source = lane; source < source_count;
       source += blockDim.x) {
    const std::uint64_t slot =
        source < state->committed_count
            ? state->committed[source]
            : proposals[source - state->committed_count];
    insert_slot_and_neighbors(state, topology, slot);
  }
  __syncthreads();
  return commit_workspace(state);
}

template <std::uint32_t Capacity>
__device__ inline bool refresh_once(State<Capacity>* state,
                                    ActivityView activity) {
  const std::uint32_t lane = threadIdx.x;
  prepare_workspace(state);

  if (state->committed_count > Capacity || activity.words == nullptr ||
      activity.word_count == 0u || activity.pinned_count > Capacity ||
      (activity.pinned_count != 0u && activity.pinned == nullptr)) {
    if (lane == 0u)
      state->fault = static_cast<std::uint32_t>(Fault::invalid_input);
    __syncthreads();
    return false;
  }

  const std::uint32_t source_count =
      state->committed_count + activity.pinned_count;
  for (std::uint32_t source = lane; source < source_count;
       source += blockDim.x) {
    const bool pinned = source >= state->committed_count;
    const std::uint64_t slot =
        pinned ? activity.pinned[source - state->committed_count]
               : state->committed[source];
    if (!durable_slot_valid(slot, activity.word_count)) {
      publish_fault(state, Fault::invalid_input);
      continue;
    }
    if (pinned || activity.words[slot] != kQ) insert_unique(state, slot);
  }
  __syncthreads();
  return commit_workspace(state);
}

template <std::uint32_t Capacity>
__device__ inline void advance_generations(
    State<Capacity>* state, TopologyView topology,
    const std::uint64_t* proposals, std::uint32_t proposal_count,
    std::uint32_t generations) {
  for (std::uint32_t generation = 0u; generation < generations;
       ++generation) {
    if (!advance_once(state, topology, proposals, proposal_count)) return;
  }
}

template <std::uint32_t Capacity>
static __global__ void advance_kernel(
    State<Capacity>* state, TopologyView topology,
    const std::uint64_t* proposals, std::uint32_t proposal_count,
    std::uint32_t generations) {
  if (blockIdx.x != 0u || blockIdx.y != 0u || blockIdx.z != 0u) return;
  advance_generations(state, topology, proposals, proposal_count, generations);
}

template <std::uint32_t Capacity>
static __global__ void refresh_kernel(State<Capacity>* state,
                                      ActivityView activity) {
  if (blockIdx.x != 0u || blockIdx.y != 0u || blockIdx.z != 0u) return;
  (void)refresh_once(state, activity);
}

template <std::uint32_t Capacity>
static __global__ void snapshot_kernel(const State<Capacity>* state,
                                       Snapshot<Capacity>* snapshot) {
  if (blockIdx.x != 0u || blockIdx.y != 0u || blockIdx.z != 0u) return;
  for (std::uint32_t index = threadIdx.x; index < Capacity;
       index += blockDim.x) {
    snapshot->committed[index] = state->committed[index];
  }
  if (threadIdx.x == 0u) {
    snapshot->committed_count = state->committed_count;
    snapshot->fault = state->fault;
    snapshot->generation = state->generation;
    snapshot->revision = state->revision;
    snapshot->attempts = state->attempts;
    snapshot->overflow_count = state->overflow_count;
  }
}

template <std::uint32_t Capacity>
static __global__ void restore_kernel(State<Capacity>* state,
                                      const Snapshot<Capacity>* snapshot) {
  if (blockIdx.x != 0u || blockIdx.y != 0u || blockIdx.z != 0u) return;
  for (std::uint32_t index = threadIdx.x; index < Capacity;
       index += blockDim.x) {
    state->committed[index] = snapshot->committed[index];
  }
  if (threadIdx.x == 0u) {
    state->committed_count = snapshot->committed_count;
    state->fault = snapshot->fault;
    state->generation = snapshot->generation;
    state->revision = snapshot->revision;
    state->attempts = snapshot->attempts;
    state->overflow_count = snapshot->overflow_count;
    state->scratch_count = 0u;
    state->unique_count = 0u;
  }
}

}  // namespace substrate::bcc32::device_resident_frontier
