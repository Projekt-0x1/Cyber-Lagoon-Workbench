#pragma once

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace bcc32_cuda_resident_relation_cortex {

inline void cuda_require(cudaError_t status, const char* action) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(action) + ": " + cudaGetErrorString(status));
  }
}

template <typename T>
class DeviceBuffer {
 public:
  DeviceBuffer() = default;
  explicit DeviceBuffer(std::size_t count) { allocate(count); }
  ~DeviceBuffer() { cudaFree(data_); }

  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  DeviceBuffer(DeviceBuffer&& other) noexcept
      : data_(std::exchange(other.data_, nullptr)), count_(std::exchange(other.count_, 0u)) {}
  DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
    if (this != &other) {
      cudaFree(data_);
      data_ = std::exchange(other.data_, nullptr);
      count_ = std::exchange(other.count_, 0u);
    }
    return *this;
  }

  void allocate(std::size_t count) {
    cudaFree(data_);
    data_ = nullptr;
    count_ = count;
    if (count != 0u) {
      cuda_require(cudaMalloc(&data_, count * sizeof(T)), "allocate resident relation cortex");
    }
  }

  T* get() { return data_; }
  const T* get() const { return data_; }
  std::size_t size() const { return count_; }
  std::size_t bytes() const { return count_ * sizeof(T); }

 private:
  T* data_ = nullptr;
  std::size_t count_ = 0u;
};

enum class Direction : std::uint32_t { kForward = 0u, kReverse = 1u };

enum class EdgeKind : std::uint32_t {
  kRecurrent = 0u,
  kDirectional = 1u,
  kUnionToIndex = 2u,
  kIndexToUnion = 3u,
};

constexpr std::uint32_t edge_kind_mask(EdgeKind kind) {
  return 1u << static_cast<std::uint32_t>(kind);
}

constexpr std::uint32_t kAllEdgeKinds =
    edge_kind_mask(EdgeKind::kRecurrent) | edge_kind_mask(EdgeKind::kDirectional) |
    edge_kind_mask(EdgeKind::kUnionToIndex) | edge_kind_mask(EdgeKind::kIndexToUnion);

enum class Status : std::uint32_t {
  kOk = 0u,
  kOutOfMatter = 1u,
  kEdgeCapacity = 2u,
  kEpisodeCapacity = 3u,
  kCodecCapacity = 4u,
  kInvalidPopulation = 5u,
  kCommitmentBusy = 6u,
};

struct ResidentPolicyState {
  std::uint32_t recurrent_weight = 5u;
  std::uint32_t directional_quantum = 1u;
  std::uint32_t relation_horizon = 3u;
  std::uint32_t binding_weight = 2u;
  std::uint32_t completion_width = 48u;
  std::uint32_t minimum_directional_votes = 1u;
  std::uint32_t minimum_episode_votes = 1u;
  std::uint32_t recurrent_settle_iterations = 1u;
  std::uint32_t minimum_recurrent_votes = 1u;
  std::uint32_t appraisal_quanta = 1u;
  std::uint32_t commitment_minimum_overlap = 1u;
  long long commitment_minimum_net_appraisal = (-9223372036854775807ll - 1ll);
  std::uint32_t commitment_quanta = 32u;
};

struct Config {
  std::uint32_t population = 8192u;
  std::uint32_t active_width = 48u;
  std::uint32_t episode_index_width = 48u;
  std::uint32_t max_edges = 1u << 20u;
  std::uint32_t max_episodes = 2048u;
  std::uint32_t max_settle_iterations = 8u;
  unsigned long long policy_mass = 64ull;
  unsigned long long represented_mass = 1ull << 30u;
  ResidentPolicyState warm_start{};
};

struct DevicePopulationView {
  const std::uint32_t* cells = nullptr;
  std::uint32_t count = 0u;
};

// offsets has part_count + 1 entries and partitions ordered_cells by temporal contact order.
struct DeviceEpisodeView {
  const std::uint32_t* ordered_cells = nullptr;
  const std::uint32_t* offsets = nullptr;
  std::uint32_t ordered_count = 0u;
  std::uint32_t part_count = 0u;
  DevicePopulationView episode_union{};
};

// Non-owning view of AdultState::surface_unit_population and its resident dynamics.
// population is row-major [unit_count][population_width]. Unit indices are mutable
// surface codec addresses; the cortex never stores a second unit-to-code mapping.
struct ExternalUnitPopulationView {
  const std::uint32_t* population = nullptr;
  const unsigned long long* activity = nullptr;
  const std::uint32_t* phase = nullptr;
  std::uint32_t unit_count = 0u;
  std::uint32_t population_width = 0u;
};

struct MatterAccounting {
  unsigned long long free_mass = 0ull;
  unsigned long long policy_mass = 0ull;
  unsigned long long edge_mass = 0ull;
  unsigned long long appraisal_mass = 0ull;
  unsigned long long commitment_mass = 0ull;
  unsigned long long lesion_mass = 0ull;
  unsigned long long represented_mass = 0ull;
};

struct DeviceCommitmentView {
  const std::uint32_t* codec_addresses = nullptr;
  const std::uint32_t* count = nullptr;
};

struct EdgeEvent {
  std::uint32_t source = 0u;
  std::uint32_t target = 0u;
  std::uint32_t weight = 0u;
  std::uint32_t episode = 0u;
  std::uint32_t kind = 0u;
  std::uint32_t state = 0u;
  std::uint32_t lesion_tag = 0u;
};

namespace detail {

constexpr std::uint32_t kInvalid = 0xffffffffu;
constexpr std::uint32_t kEdgeActive = 1u;
constexpr std::uint32_t kEdgeLesioned = 2u;

__device__ __forceinline__ std::uint32_t mix32(std::uint32_t value) {
  value ^= value >> 16u;
  value *= 0x7feb352du;
  value ^= value >> 15u;
  value *= 0x846ca68bu;
  value ^= value >> 16u;
  return value;
}

__device__ __forceinline__ bool in_population(std::uint32_t cell,
                                               const std::uint32_t* cells,
                                               std::uint32_t count) {
  for (std::uint32_t i = 0u; i < count; ++i) {
    if (cells[i] == cell) return true;
  }
  return false;
}

__device__ __forceinline__ void set_status(std::uint32_t* status, Status value) {
  atomicCAS(status, static_cast<std::uint32_t>(Status::kOk),
            static_cast<std::uint32_t>(value));
}

__device__ __forceinline__ bool pay(unsigned long long* free_mass,
                                    unsigned long long amount) {
  unsigned long long available = atomicAdd(free_mass, 0ull);
  while (available >= amount) {
    const unsigned long long observed = atomicCAS(free_mass, available, available - amount);
    if (observed == available) return true;
    available = observed;
  }
  return false;
}

__device__ __forceinline__ bool reserve_edge(std::uint32_t* edge_count,
                                             std::uint32_t edge_capacity,
                                             std::uint32_t* slot) {
  std::uint32_t count = atomicAdd(edge_count, 0u);
  while (count < edge_capacity) {
    const std::uint32_t observed = atomicCAS(edge_count, count, count + 1u);
    if (observed == count) {
      *slot = count;
      return true;
    }
    count = observed;
  }
  return false;
}

__device__ __forceinline__ void append_edge(EdgeEvent* edges, std::uint32_t* edge_count,
                                            std::uint32_t edge_capacity,
                                            unsigned long long* free_mass,
                                            std::uint32_t* status, std::uint32_t source,
                                            std::uint32_t target, std::uint32_t weight,
                                            EdgeKind kind, std::uint32_t episode) {
  if (source == target || weight == 0u) return;
  if (!pay(free_mass, weight)) {
    set_status(status, Status::kOutOfMatter);
    return;
  }
  std::uint32_t slot = 0u;
  if (!reserve_edge(edge_count, edge_capacity, &slot)) {
    atomicAdd(free_mass, static_cast<unsigned long long>(weight));
    set_status(status, Status::kEdgeCapacity);
    return;
  }
  EdgeEvent event;
  event.source = source;
  event.target = target;
  event.weight = weight;
  event.episode = episode;
  event.kind = static_cast<std::uint32_t>(kind);
  event.state = kEdgeActive;
  edges[slot] = event;
}

__global__ void learn_concept_kernel(DevicePopulationView code, std::uint32_t population,
                                     const ResidentPolicyState* policy, EdgeEvent* edges,
                                     std::uint32_t* edge_count, std::uint32_t edge_capacity,
                                     unsigned long long* free_mass, std::uint32_t* status) {
  const std::uint64_t ordinal =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::uint64_t pair_count = static_cast<std::uint64_t>(code.count) * code.count;
  if (ordinal >= pair_count) return;
  const std::uint32_t source = code.cells[ordinal / code.count];
  const std::uint32_t target = code.cells[ordinal % code.count];
  if (source >= population || target >= population) {
    set_status(status, Status::kInvalidPopulation);
    return;
  }
  append_edge(edges, edge_count, edge_capacity, free_mass, status, source, target,
              policy->recurrent_weight,
              EdgeKind::kRecurrent, kInvalid);
}

__global__ void reserve_episode_kernel(std::uint32_t* episode_count,
                                       std::uint32_t episode_capacity,
                                       std::uint32_t* current_episode,
                                       std::uint32_t* status) {
  extern __shared__ std::uint32_t reserved_slot[];
  const std::uint32_t lane = threadIdx.x;
  if (lane == 0u) reserved_slot[0] = *episode_count;
  __syncthreads();
  const std::uint32_t slot = reserved_slot[0];
  if (slot >= episode_capacity) {
    if (lane == 0u) {
      *current_episode = kInvalid;
      set_status(status, Status::kEpisodeCapacity);
    }
    return;
  }
  if (lane == 0u) *episode_count = slot + 1u;
  __syncthreads();
  if (lane == 1u) *current_episode = slot;
}

__global__ void anatomy_drive_kernel(DevicePopulationView episode_union,
                                     std::uint32_t population,
                                     const std::uint32_t* cell_lesion_tag,
                                     unsigned long long* drive) {
  const std::uint32_t candidate = blockIdx.x * blockDim.x + threadIdx.x;
  if (candidate >= population) return;
  if (cell_lesion_tag[candidate] != 0u) {
    drive[candidate] = 0ull;
    return;
  }
  unsigned long long value = 0ull;
  for (std::uint32_t i = 0u; i < episode_union.count; ++i) {
    const std::uint32_t source = episode_union.cells[i];
    const std::uint32_t signature =
        mix32(source * 2654435761u ^ (candidate * 0x9e3779b9u + 0x85ebca6bu));
    value += static_cast<unsigned long long>((signature & 0xffffu) < 3932u);
  }
  drive[candidate] = value;
}

__device__ void select_top_width_kernel(const unsigned long long* scores,
                                        std::uint32_t population,
                                        std::uint32_t width,
                                        std::uint32_t* output,
                                        std::uint32_t* output_count,
                                        unsigned long long* shared_scores,
                                        std::uint32_t* shared_candidates) {
  // Candidate scans are parallel, but selection slots remain ordered so the
  // resident tie rule (score descending, then candidate ID ascending) is
  // deterministic. kThreads is a power of two for this reduction.
  const std::uint32_t lane = threadIdx.x;
  for (std::uint32_t slot = 0u; slot < width; ++slot) {
    std::uint32_t local_candidate = kInvalid;
    unsigned long long local_score = 0ull;
    for (std::uint32_t candidate = lane; candidate < population;
         candidate += blockDim.x) {
      bool selected = false;
      for (std::uint32_t prior = 0u; prior < slot; ++prior) {
        if (output[prior] == candidate) {
          selected = true;
          break;
        }
      }
      if (selected) continue;
      const unsigned long long score = scores[candidate];
      if (score == 0ull ||
          (local_candidate != kInvalid &&
           (score < local_score ||
            (score == local_score && candidate > local_candidate)))) {
        continue;
      }
      local_score = score;
      local_candidate = candidate;
    }
    shared_scores[lane] = local_score;
    shared_candidates[lane] = local_candidate;
    __syncthreads();
    for (std::uint32_t stride = blockDim.x / 2u; stride != 0u; stride >>= 1u) {
      if (lane < stride) {
        const std::uint32_t other_candidate = shared_candidates[lane + stride];
        if (other_candidate != kInvalid &&
            (shared_candidates[lane] == kInvalid ||
             shared_scores[lane + stride] > shared_scores[lane] ||
             (shared_scores[lane + stride] == shared_scores[lane] &&
              other_candidate < shared_candidates[lane]))) {
          shared_scores[lane] = shared_scores[lane + stride];
          shared_candidates[lane] = other_candidate;
        }
      }
      __syncthreads();
    }
    if (lane == 0u) output[slot] = shared_candidates[0];
    __syncthreads();
  }
  if (lane == 0u && output_count != nullptr) {
    std::uint32_t count = 0u;
    for (std::uint32_t slot = 0u; slot < width; ++slot) {
      if (output[slot] == kInvalid) break;
      ++count;
    }
    *output_count = count;
  }
}

__global__ void select_index_kernel(const unsigned long long* drive,
                                    std::uint32_t population, std::uint32_t width,
                                    std::uint32_t* index_scratch) {
  extern __shared__ unsigned long long shared_scores[];
  auto* shared_candidates = reinterpret_cast<std::uint32_t*>(
      shared_scores + blockDim.x);
  select_top_width_kernel(drive, population, width, index_scratch, nullptr,
                          shared_scores, shared_candidates);
}

__global__ void store_episode_index_kernel(const std::uint32_t* current_episode,
                                           std::uint32_t index_width,
                                           const std::uint32_t* index_scratch,
                                           std::uint32_t* episode_indices) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  const std::uint32_t episode = *current_episode;
  if (episode == kInvalid || i >= index_width) return;
  episode_indices[static_cast<std::size_t>(episode) * index_width + i] = index_scratch[i];
}

__device__ __forceinline__ std::uint32_t find_part(const std::uint32_t* offsets,
                                                   std::uint32_t part_count,
                                                   std::uint32_t ordinal) {
  for (std::uint32_t part = 0u; part < part_count; ++part) {
    if (ordinal >= offsets[part] && ordinal < offsets[part + 1u]) return part;
  }
  return kInvalid;
}

__global__ void learn_directional_episode_kernel(
    DeviceEpisodeView episode, std::uint32_t population, const ResidentPolicyState* policy,
    const std::uint32_t* current_episode, EdgeEvent* edges, std::uint32_t* edge_count,
    std::uint32_t edge_capacity, unsigned long long* free_mass, std::uint32_t* status) {
  const std::uint64_t ordinal =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::uint64_t pair_count =
      static_cast<std::uint64_t>(episode.ordered_count) * episode.ordered_count;
  if (ordinal >= pair_count || *current_episode == kInvalid) return;
  const std::uint32_t source_ordinal = static_cast<std::uint32_t>(ordinal / episode.ordered_count);
  const std::uint32_t target_ordinal = static_cast<std::uint32_t>(ordinal % episode.ordered_count);
  const std::uint32_t source_part = find_part(episode.offsets, episode.part_count, source_ordinal);
  const std::uint32_t target_part = find_part(episode.offsets, episode.part_count, target_ordinal);
  if (source_part == kInvalid || target_part == kInvalid || source_part >= target_part) return;
  const std::uint32_t distance = target_part - source_part;
  if (distance > policy->relation_horizon) return;
  const std::uint32_t source = episode.ordered_cells[source_ordinal];
  const std::uint32_t target = episode.ordered_cells[target_ordinal];
  if (source >= population || target >= population) {
    set_status(status, Status::kInvalidPopulation);
    return;
  }
  append_edge(edges, edge_count, edge_capacity, free_mass, status, source, target,
              (policy->relation_horizon + 1u - distance) * policy->directional_quantum,
              EdgeKind::kDirectional, *current_episode);
}

__global__ void learn_binding_kernel(DevicePopulationView episode_union,
                                     const std::uint32_t* index_scratch,
                                     std::uint32_t index_width,
                                     const ResidentPolicyState* policy,
                                     const std::uint32_t* current_episode, EdgeEvent* edges,
                                     std::uint32_t* edge_count, std::uint32_t edge_capacity,
                                     unsigned long long* free_mass, std::uint32_t* status) {
  const std::uint64_t ordinal =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::uint64_t one_direction =
      static_cast<std::uint64_t>(episode_union.count) * index_width;
  if (ordinal >= 2ull * one_direction || *current_episode == kInvalid) return;
  const bool reverse = ordinal >= one_direction;
  const std::uint64_t local = reverse ? ordinal - one_direction : ordinal;
  const std::uint32_t union_cell = episode_union.cells[local / index_width];
  const std::uint32_t index_cell = index_scratch[local % index_width];
  if (index_cell == kInvalid) return;
  append_edge(edges, edge_count, edge_capacity, free_mass, status,
              reverse ? index_cell : union_cell, reverse ? union_cell : index_cell,
              policy->binding_weight,
              reverse ? EdgeKind::kIndexToUnion : EdgeKind::kUnionToIndex,
              *current_episode);
}

__global__ void score_edges_kernel(const EdgeEvent* edges, const std::uint32_t* edge_count,
                                   EdgeKind kind, Direction direction,
                                   const std::uint32_t* active,
                                   const std::uint32_t* resident_active_count,
                                   std::uint32_t immediate_active_count,
                                   unsigned long long* score, std::uint32_t* votes) {
  const std::uint32_t ordinal = blockIdx.x * blockDim.x + threadIdx.x;
  if (ordinal >= *edge_count) return;
  const EdgeEvent edge = edges[ordinal];
  if (edge.state != kEdgeActive || edge.kind != static_cast<std::uint32_t>(kind)) return;
  const std::uint32_t count =
      resident_active_count == nullptr ? immediate_active_count : *resident_active_count;
  const std::uint32_t contact =
      direction == Direction::kForward ? edge.source : edge.target;
  if (!in_population(contact, active, count)) return;
  const std::uint32_t candidate =
      direction == Direction::kForward ? edge.target : edge.source;
  atomicAdd(score + candidate, static_cast<unsigned long long>(edge.weight));
  atomicAdd(votes + candidate, 1u);
}

struct EpisodeCandidate {
  std::uint32_t episode;
  unsigned long long score;
  unsigned long long votes;
};

__device__ __forceinline__ bool episode_candidate_precedes(
    const EpisodeCandidate& candidate, const EpisodeCandidate& resident) {
  if (candidate.episode == kInvalid) return false;
  if (resident.episode == kInvalid) return true;
  if (candidate.score != resident.score) return candidate.score > resident.score;
  if (candidate.votes != resident.votes) return candidate.votes > resident.votes;
  return candidate.episode < resident.episode;
}

__global__ void select_episode_kernel(const unsigned long long* index_score,
                                      const std::uint32_t* index_votes,
                                      const std::uint32_t* episode_count,
                                      const std::uint32_t* episode_indices,
                                      std::uint32_t index_width,
                                      std::uint32_t* selected_episode) {
  extern __shared__ EpisodeCandidate episode_candidates[];
  const std::uint32_t lane = threadIdx.x;
  EpisodeCandidate local{kInvalid, 0ull, 0ull};
  for (std::uint32_t episode = lane; episode < *episode_count;
       episode += blockDim.x) {
    unsigned long long score = 0ull;
    unsigned long long votes = 0ull;
    for (std::uint32_t i = 0u; i < index_width; ++i) {
      const std::uint32_t cell =
          episode_indices[static_cast<std::size_t>(episode) * index_width + i];
      if (cell == kInvalid) continue;
      score += index_score[cell];
      votes += index_votes[cell];
    }
    if (score != 0ull) {
      const EpisodeCandidate candidate{episode, score, votes};
      if (episode_candidate_precedes(candidate, local)) local = candidate;
    }
  }
  episode_candidates[lane] = local;
  __syncthreads();
  for (std::uint32_t stride = blockDim.x / 2u; stride != 0u; stride >>= 1u) {
    if (lane < stride &&
        episode_candidate_precedes(episode_candidates[lane + stride],
                                   episode_candidates[lane])) {
      episode_candidates[lane] = episode_candidates[lane + stride];
    }
    __syncthreads();
  }
  if (lane == 0u) *selected_episode = episode_candidates[0].episode;
}

__global__ void score_selected_index_kernel(
    const EdgeEvent* edges, const std::uint32_t* edge_count,
    const std::uint32_t* selected_episode, const std::uint32_t* episode_indices,
    std::uint32_t index_width, unsigned long long* score, std::uint32_t* votes) {
  const std::uint32_t ordinal = blockIdx.x * blockDim.x + threadIdx.x;
  if (ordinal >= *edge_count || *selected_episode == kInvalid) return;
  const EdgeEvent edge = edges[ordinal];
  if (edge.state != kEdgeActive ||
      edge.kind != static_cast<std::uint32_t>(EdgeKind::kIndexToUnion)) {
    return;
  }
  const std::uint32_t* index =
      episode_indices + static_cast<std::size_t>(*selected_episode) * index_width;
  if (!in_population(edge.source, index, index_width)) return;
  atomicAdd(score + edge.target, static_cast<unsigned long long>(edge.weight));
  atomicAdd(votes + edge.target, 1u);
}

__global__ void rank_conjunctive_kernel(
    std::uint32_t population, const std::uint32_t* cell_lesion_tag,
    DevicePopulationView cue, const unsigned long long* directional_score,
    const std::uint32_t* directional_votes, const unsigned long long* episode_score,
    const std::uint32_t* episode_votes, const ResidentPolicyState* policy,
    unsigned long long* ranks) {
  const std::uint32_t cell = blockIdx.x * blockDim.x + threadIdx.x;
  if (cell >= population) return;
  if (cell_lesion_tag[cell] != 0u || in_population(cell, cue.cells, cue.count) ||
      directional_votes[cell] < policy->minimum_directional_votes ||
      episode_votes[cell] < policy->minimum_episode_votes || directional_score[cell] == 0ull ||
      episode_score[cell] == 0ull) {
    ranks[cell] = 0ull;
    return;
  }
  const unsigned long long episode = min(episode_score[cell], 0xffffffffull);
  const unsigned long long directional = min(directional_score[cell], 0xffffffffull);
  ranks[cell] = (episode << 32u) | directional;
}

__global__ void rank_recurrent_kernel(std::uint32_t population,
                                      const std::uint32_t* cell_lesion_tag,
                                      const unsigned long long* score,
                                      const std::uint32_t* votes,
                                      const ResidentPolicyState* policy,
                                      std::uint32_t iteration,
                                      unsigned long long* ranks) {
  const std::uint32_t cell = blockIdx.x * blockDim.x + threadIdx.x;
  if (cell >= population) return;
  if (iteration >= policy->recurrent_settle_iterations) return;
  ranks[cell] = cell_lesion_tag[cell] == 0u &&
                        votes[cell] >= policy->minimum_recurrent_votes
                    ? score[cell]
                    : 0ull;
}

__global__ void select_active_kernel(const unsigned long long* ranks,
                                     std::uint32_t population, std::uint32_t maximum_width,
                                     const ResidentPolicyState* policy,
                                     std::uint32_t settle_iteration,
                                     std::uint32_t* active, std::uint32_t* active_count) {
  if (settle_iteration != kInvalid &&
      settle_iteration >= policy->recurrent_settle_iterations) return;
  const std::uint32_t width = min(maximum_width, policy->completion_width);
  extern __shared__ unsigned long long shared_scores[];
  auto* shared_candidates = reinterpret_cast<std::uint32_t*>(
      shared_scores + blockDim.x);
  select_top_width_kernel(ranks, population, width, active, active_count,
                          shared_scores, shared_candidates);
}

__global__ void appraise_kernel(const std::uint32_t* active,
                                const std::uint32_t* active_count,
                                long long signed_consequence,
                                const ResidentPolicyState* policy,
                                unsigned long long* positive,
                                unsigned long long* negative,
                                unsigned long long* free_mass,
                                std::uint32_t* status) {
  const std::uint32_t ordinal = blockIdx.x * blockDim.x + threadIdx.x;
  if (ordinal >= *active_count || signed_consequence == 0ll ||
      policy->appraisal_quanta == 0u) return;
  const std::uint32_t cell = active[ordinal];
  const unsigned long long magnitude =
      static_cast<unsigned long long>(signed_consequence < 0ll ? -signed_consequence
                                                               : signed_consequence);
  const unsigned long long amount = magnitude * policy->appraisal_quanta;
  if (!pay(free_mass, amount)) {
    set_status(status, Status::kOutOfMatter);
    return;
  }
  atomicAdd((signed_consequence > 0ll ? positive : negative) + cell, amount);
}

__device__ __forceinline__ long long net_appraisal(const std::uint32_t* cells,
                                                   std::uint32_t count,
                                                   const unsigned long long* positive,
                                                   const unsigned long long* negative) {
  unsigned long long pos = 0ull;
  unsigned long long neg = 0ull;
  for (std::uint32_t i = 0u; i < count; ++i) {
    pos += positive[cells[i]];
    neg += negative[cells[i]];
  }
  const unsigned long long limit = static_cast<unsigned long long>(0x7fffffffffffffffll);
  if (pos >= neg) return static_cast<long long>(min(pos - neg, limit));
  return -static_cast<long long>(min(neg - pos, limit));
}

struct CommitmentCandidate {
  std::uint32_t unit;
  std::uint32_t overlap;
  long long appraisal;
  unsigned long long activity;
  std::uint32_t phase;
};

__device__ __forceinline__ bool commitment_candidate_precedes(
    const CommitmentCandidate& candidate, const CommitmentCandidate& resident) {
  if (candidate.unit == kInvalid) return false;
  if (resident.unit == kInvalid) return true;
  if (candidate.overlap != resident.overlap) return candidate.overlap > resident.overlap;
  if (candidate.appraisal != resident.appraisal) return candidate.appraisal > resident.appraisal;
  if (candidate.activity != resident.activity) return candidate.activity > resident.activity;
  if (candidate.phase != resident.phase) return candidate.phase > resident.phase;
  return candidate.unit < resident.unit;
}

__global__ void commit_kernel(const std::uint32_t* active,
                              const std::uint32_t* active_count,
                              ExternalUnitPopulationView surface,
                              const unsigned long long* positive,
                              const unsigned long long* negative,
                              const ResidentPolicyState* policy,
                              std::uint32_t* commitment_addresses,
                              std::uint32_t* commitment_count,
                              unsigned long long* commitment_mass,
                              unsigned long long* free_mass, std::uint32_t* status) {
  if (*commitment_count != 0u) {
    if (threadIdx.x == 0u) set_status(status, Status::kCommitmentBusy);
    return;
  }
  extern __shared__ CommitmentCandidate candidates[];
  const std::uint32_t lane = threadIdx.x;
  CommitmentCandidate local{
      kInvalid, 0u, (-9223372036854775807ll - 1ll), 0ull, 0u};
  for (std::uint32_t unit = lane; unit < surface.unit_count; unit += blockDim.x) {
    const std::uint32_t* cells = surface.population +
                                 static_cast<std::size_t>(unit) * surface.population_width;
    std::uint32_t overlap = 0u;
    for (std::uint32_t i = 0u; i < surface.population_width; ++i) {
      overlap += static_cast<std::uint32_t>(in_population(cells[i], active, *active_count));
    }
    const long long appraisal =
        net_appraisal(cells, surface.population_width, positive, negative);
    const unsigned long long activity = surface.activity == nullptr ? 0ull : surface.activity[unit];
    const std::uint32_t phase = surface.phase == nullptr ? 0u : surface.phase[unit];
    if (overlap < policy->commitment_minimum_overlap ||
        appraisal < policy->commitment_minimum_net_appraisal) continue;
    const CommitmentCandidate candidate{unit, overlap, appraisal, activity, phase};
    if (commitment_candidate_precedes(candidate, local)) local = candidate;
  }
  candidates[lane] = local;
  __syncthreads();
  for (std::uint32_t stride = blockDim.x / 2u; stride != 0u; stride >>= 1u) {
    if (lane < stride &&
        commitment_candidate_precedes(candidates[lane + stride], candidates[lane])) {
      candidates[lane] = candidates[lane + stride];
    }
    __syncthreads();
  }
  if (lane == 0u) {
    if (candidates[0].unit == kInvalid) return;
    if (!pay(free_mass, policy->commitment_quanta)) {
      set_status(status, Status::kOutOfMatter);
      return;
    }
    commitment_addresses[0] = candidates[0].unit;
    *commitment_count = 1u;
    atomicAdd(commitment_mass, static_cast<unsigned long long>(policy->commitment_quanta));
  }
}

__global__ void release_commitment_kernel(std::uint32_t* commitment_addresses,
                                          std::uint32_t* commitment_count,
                                          unsigned long long* commitment_mass,
                                          unsigned long long* free_mass) {
  extern __shared__ unsigned long long shared_mass[];
  const std::uint32_t lane = threadIdx.x;
  if (lane == 0u) shared_mass[0] = *commitment_mass;
  __syncthreads();
  if (lane == 0u) atomicAdd(free_mass, shared_mass[0]);
  __syncthreads();
  if (lane == 1u) *commitment_mass = 0ull;
  if (lane == 2u) *commitment_count = 0u;
  if (lane == 3u) commitment_addresses[0] = kInvalid;
}

__global__ void tag_lesioned_cells_kernel(DevicePopulationView cells, std::uint32_t population,
                                          std::uint32_t tag,
                                          std::uint32_t* cell_lesion_tag,
                                          std::uint32_t* status) {
  const std::uint32_t ordinal = blockIdx.x * blockDim.x + threadIdx.x;
  if (ordinal >= cells.count) return;
  const std::uint32_t cell = cells.cells[ordinal];
  if (cell >= population) {
    set_status(status, Status::kInvalidPopulation);
    return;
  }
  atomicCAS(cell_lesion_tag + cell, 0u, tag);
}

__global__ void lesion_edges_kernel(EdgeEvent* edges, const std::uint32_t* edge_count,
                                    DevicePopulationView cells, std::uint32_t kind_mask,
                                    std::uint32_t tag, unsigned long long* lesion_mass) {
  const std::uint32_t ordinal = blockIdx.x * blockDim.x + threadIdx.x;
  if (ordinal >= *edge_count) return;
  EdgeEvent* edge = edges + ordinal;
  if ((kind_mask & (1u << edge->kind)) == 0u ||
      (!in_population(edge->source, cells.cells, cells.count) &&
       !in_population(edge->target, cells.cells, cells.count))) {
    return;
  }
  if (atomicCAS(&edge->state, kEdgeActive, kEdgeLesioned) == kEdgeActive) {
    edge->lesion_tag = tag;
    atomicAdd(lesion_mass, static_cast<unsigned long long>(edge->weight));
  }
}

__global__ void restore_edges_kernel(EdgeEvent* edges, const std::uint32_t* edge_count,
                                     std::uint32_t tag, unsigned long long* lesion_mass) {
  const std::uint32_t ordinal = blockIdx.x * blockDim.x + threadIdx.x;
  if (ordinal >= *edge_count) return;
  EdgeEvent* edge = edges + ordinal;
  if (edge->state != kEdgeLesioned || edge->lesion_tag != tag) return;
  edge->lesion_tag = 0u;
  edge->state = kEdgeActive;
  atomicAdd(lesion_mass, 0ull - static_cast<unsigned long long>(edge->weight));
}

__global__ void restore_cells_kernel(std::uint32_t* cell_lesion_tag,
                                     std::uint32_t population, std::uint32_t tag) {
  const std::uint32_t cell = blockIdx.x * blockDim.x + threadIdx.x;
  if (cell < population && cell_lesion_tag[cell] == tag) cell_lesion_tag[cell] = 0u;
}

__global__ void account_edges_kernel(const EdgeEvent* edges, const std::uint32_t* edge_count,
                                     MatterAccounting* accounting) {
  const std::uint32_t ordinal = blockIdx.x * blockDim.x + threadIdx.x;
  if (ordinal < *edge_count && edges[ordinal].state == kEdgeActive) {
    atomicAdd(&accounting->edge_mass, static_cast<unsigned long long>(edges[ordinal].weight));
  }
}

__global__ void account_population_kernel(std::uint32_t population,
                                          const unsigned long long* positive,
                                          const unsigned long long* negative,
                                          MatterAccounting* accounting) {
  const std::uint32_t cell = blockIdx.x * blockDim.x + threadIdx.x;
  if (cell < population) atomicAdd(&accounting->appraisal_mass, positive[cell] + negative[cell]);
}

__global__ void finalize_accounting_kernel(const unsigned long long* free_mass,
                                           const unsigned long long* policy_mass,
                                           const unsigned long long* commitment_mass,
                                           const unsigned long long* lesion_mass,
                                           MatterAccounting* accounting) {
  const std::uint32_t field = blockIdx.x * blockDim.x + threadIdx.x;
  if (field == 0u) accounting->free_mass = *free_mass;
  if (field == 1u) accounting->policy_mass = *policy_mass;
  if (field == 2u) accounting->commitment_mass = *commitment_mass;
  if (field == 3u) accounting->lesion_mass = *lesion_mass;
  __syncthreads();
  if (field == 4u) {
    accounting->represented_mass =
        accounting->free_mass + accounting->policy_mass + accounting->edge_mass +
        accounting->appraisal_mass + accounting->commitment_mass + accounting->lesion_mass;
  }
}

}  // namespace detail

class ResidentRelationCortex {
 public:
  explicit ResidentRelationCortex(Config config = {})
      : config_(config),
        edges_(config.max_edges),
        edge_count_(1u),
        episode_count_(1u),
        current_episode_(1u),
        selected_episode_(1u),
        episode_indices_(static_cast<std::size_t>(config.max_episodes) *
                         config.episode_index_width),
        index_scratch_(config.episode_index_width),
        anatomy_drive_(config.population),
        positive_appraisal_(config.population),
        negative_appraisal_(config.population),
        cell_lesion_tag_(config.population),
        index_score_(config.population),
        index_votes_(config.population),
        episode_score_(config.population),
        episode_votes_(config.population),
        directional_score_(config.population),
        directional_votes_(config.population),
        recurrent_score_(config.population),
        recurrent_votes_(config.population),
        ranks_(config.population),
        active_(config.active_width),
        active_count_(1u),
        policy_(1u),
        policy_backup_(1u),
        policy_mass_(1u),
        policy_lesion_tag_(1u),
        commitment_addresses_(1u),
        commitment_count_(1u),
        commitment_mass_(1u),
        lesion_mass_(1u),
        free_mass_(1u),
        status_(1u),
        accounting_(1u) {
    validate_config();
    cuda_require(cudaMemset(edges_.get(), 0, edges_.bytes()), "clear sparse relation edges");
    cuda_require(cudaMemset(edge_count_.get(), 0, edge_count_.bytes()), "clear edge count");
    cuda_require(cudaMemset(episode_count_.get(), 0, episode_count_.bytes()),
                 "clear episode count");
    cuda_require(cudaMemset(positive_appraisal_.get(), 0, positive_appraisal_.bytes()),
                 "clear positive appraisal");
    cuda_require(cudaMemset(negative_appraisal_.get(), 0, negative_appraisal_.bytes()),
                 "clear negative appraisal");
    cuda_require(cudaMemset(cell_lesion_tag_.get(), 0, cell_lesion_tag_.bytes()),
                 "clear lesion tags");
    cuda_require(cudaMemset(active_count_.get(), 0, active_count_.bytes()),
                 "clear active count");
    cuda_require(cudaMemcpy(policy_.get(), &config_.warm_start, sizeof(config_.warm_start),
                            cudaMemcpyHostToDevice),
                 "initialize resident cortex policy");
    cuda_require(cudaMemset(policy_backup_.get(), 0, policy_backup_.bytes()),
                 "clear resident policy backup");
    cuda_require(cudaMemcpy(policy_mass_.get(), &config_.policy_mass, sizeof(config_.policy_mass),
                            cudaMemcpyHostToDevice),
                 "initialize resident policy matter");
    cuda_require(cudaMemset(policy_lesion_tag_.get(), 0, policy_lesion_tag_.bytes()),
                 "clear resident policy lesion tag");
    cuda_require(cudaMemset(commitment_count_.get(), 0, commitment_count_.bytes()),
                 "clear commitment count");
    cuda_require(cudaMemset(commitment_mass_.get(), 0, commitment_mass_.bytes()),
                 "clear commitment mass");
    cuda_require(cudaMemset(lesion_mass_.get(), 0, lesion_mass_.bytes()),
                 "clear lesion mass");
    cuda_require(cudaMemset(status_.get(), 0, status_.bytes()), "clear cortex status");
    const unsigned long long initial_free = config_.represented_mass - config_.policy_mass;
    cuda_require(cudaMemcpy(free_mass_.get(), &initial_free, sizeof(initial_free),
                            cudaMemcpyHostToDevice),
                 "initialize represented matter");
    const std::uint32_t invalid = detail::kInvalid;
    cuda_require(cudaMemcpy(current_episode_.get(), &invalid, sizeof(invalid),
                            cudaMemcpyHostToDevice),
                 "initialize current episode");
    cuda_require(cudaMemcpy(selected_episode_.get(), &invalid, sizeof(invalid),
                            cudaMemcpyHostToDevice),
                 "initialize selected episode");
    cuda_require(cudaMemcpy(commitment_addresses_.get(), &invalid, sizeof(invalid),
                            cudaMemcpyHostToDevice),
                 "initialize commitment address");
  }

  ResidentRelationCortex(const ResidentRelationCortex&) = delete;
  ResidentRelationCortex& operator=(const ResidentRelationCortex&) = delete;

  const Config& config() const { return config_; }

  // Async methods are stream ordered. A cortex owns one scratch field, so callers serialize
  // operations on one stream; device input views must remain valid until that stream completes.
  void learn_concept_async(DevicePopulationView code, cudaStream_t stream = nullptr) {
    if (code.cells == nullptr || code.count == 0u) {
      throw std::invalid_argument("concept imprint requires resident cells");
    }
    const std::uint64_t pairs = static_cast<std::uint64_t>(code.count) * code.count;
    detail::learn_concept_kernel<<<blocks(pairs), kThreads, 0, stream>>>(
        code, config_.population, policy_.get(), edges_.get(), edge_count_.get(), config_.max_edges,
        free_mass_.get(), status_.get());
    cuda_require(cudaGetLastError(), "launch sparse concept learning");
  }

  void learn_base_population_async(DevicePopulationView code, cudaStream_t stream = nullptr) {
    learn_concept_async(code, stream);
  }

  void assimilate_episode_async(DeviceEpisodeView episode, cudaStream_t stream = nullptr) {
    if (episode.ordered_cells == nullptr || episode.offsets == nullptr ||
        episode.ordered_count == 0u || episode.part_count == 0u ||
        episode.episode_union.cells == nullptr || episode.episode_union.count == 0u) {
      throw std::invalid_argument("episode assimilation requires ordered and union populations");
    }
    detail::reserve_episode_kernel<<<1u, 2u, sizeof(std::uint32_t), stream>>>(
        episode_count_.get(), config_.max_episodes, current_episode_.get(), status_.get());
    detail::anatomy_drive_kernel<<<blocks(config_.population), kThreads, 0, stream>>>(
        episode.episode_union, config_.population, cell_lesion_tag_.get(), anatomy_drive_.get());
    const unsigned long long* drive = anatomy_drive_.get();
    std::uint32_t population = config_.population;
    std::uint32_t width = config_.episode_index_width;
    std::uint32_t* index_scratch = index_scratch_.get();
    void* selection_arguments[] = {&drive, &population, &width, &index_scratch};
    cuda_require(
        cudaLaunchKernel(
            reinterpret_cast<const void*>(detail::select_index_kernel),
            dim3{1u, 1u, 1u}, dim3{kThreads, 1u, 1u}, selection_arguments,
            static_cast<std::size_t>(kThreads) *
                (sizeof(unsigned long long) + sizeof(std::uint32_t)),
            stream),
        "launch parallel relation index selection");
    detail::store_episode_index_kernel<<<blocks(config_.episode_index_width), kThreads, 0,
                                         stream>>>(current_episode_.get(),
                                                  config_.episode_index_width,
                                                  index_scratch_.get(),
                                                  episode_indices_.get());
    const std::uint64_t relation_pairs =
        static_cast<std::uint64_t>(episode.ordered_count) * episode.ordered_count;
    detail::learn_directional_episode_kernel<<<blocks(relation_pairs), kThreads, 0, stream>>>(
        episode, config_.population, policy_.get(), current_episode_.get(),
        edges_.get(), edge_count_.get(), config_.max_edges, free_mass_.get(), status_.get());
    const std::uint64_t binding_pairs =
        2ull * episode.episode_union.count * config_.episode_index_width;
    detail::learn_binding_kernel<<<blocks(binding_pairs), kThreads, 0, stream>>>(
        episode.episode_union, index_scratch_.get(), config_.episode_index_width, policy_.get(),
        current_episode_.get(), edges_.get(), edge_count_.get(),
        config_.max_edges, free_mass_.get(), status_.get());
    cuda_require(cudaGetLastError(), "launch distributed episode assimilation");
  }

  void learn_online_episode_async(DeviceEpisodeView episode, cudaStream_t stream = nullptr) {
    assimilate_episode_async(episode, stream);
  }

  void complete_async(DevicePopulationView cue, Direction direction,
                      cudaStream_t stream = nullptr) {
    if (cue.cells == nullptr || cue.count == 0u) {
      throw std::invalid_argument("completion requires a resident cue population");
    }
    clear_query_fields_async(stream);
    detail::score_edges_kernel<<<blocks(config_.max_edges), kThreads, 0, stream>>>(
        edges_.get(), edge_count_.get(), EdgeKind::kUnionToIndex, Direction::kForward,
        cue.cells, nullptr, cue.count, index_score_.get(), index_votes_.get());
    detail::select_episode_kernel<<<
        1u, kThreads,
        static_cast<std::size_t>(kThreads) * sizeof(detail::EpisodeCandidate), stream>>>(
        index_score_.get(), index_votes_.get(), episode_count_.get(), episode_indices_.get(),
        config_.episode_index_width, selected_episode_.get());
    detail::score_selected_index_kernel<<<blocks(config_.max_edges), kThreads, 0, stream>>>(
        edges_.get(), edge_count_.get(), selected_episode_.get(), episode_indices_.get(),
        config_.episode_index_width, episode_score_.get(), episode_votes_.get());
    detail::score_edges_kernel<<<blocks(config_.max_edges), kThreads, 0, stream>>>(
        edges_.get(), edge_count_.get(), EdgeKind::kDirectional, direction, cue.cells, nullptr,
        cue.count, directional_score_.get(), directional_votes_.get());
    detail::rank_conjunctive_kernel<<<blocks(config_.population), kThreads, 0, stream>>>(
        config_.population, cell_lesion_tag_.get(), cue, directional_score_.get(),
        directional_votes_.get(), episode_score_.get(), episode_votes_.get(), policy_.get(),
        ranks_.get());
    detail::select_active_kernel<<<
        1u, kThreads,
        static_cast<std::size_t>(kThreads) *
            (sizeof(unsigned long long) + sizeof(std::uint32_t)), stream>>>(
        ranks_.get(), config_.population, config_.active_width, policy_.get(), detail::kInvalid,
        active_.get(),
        active_count_.get());
    for (std::uint32_t iteration = 0u; iteration < config_.max_settle_iterations;
         ++iteration) {
      cuda_require(cudaMemsetAsync(recurrent_score_.get(), 0, recurrent_score_.bytes(), stream),
                   "clear recurrent score");
      cuda_require(cudaMemsetAsync(recurrent_votes_.get(), 0, recurrent_votes_.bytes(), stream),
                   "clear recurrent votes");
      detail::score_edges_kernel<<<blocks(config_.max_edges), kThreads, 0, stream>>>(
          edges_.get(), edge_count_.get(), EdgeKind::kRecurrent, Direction::kForward,
          active_.get(), active_count_.get(), 0u, recurrent_score_.get(),
          recurrent_votes_.get());
      detail::rank_recurrent_kernel<<<blocks(config_.population), kThreads, 0, stream>>>(
          config_.population, cell_lesion_tag_.get(), recurrent_score_.get(),
          recurrent_votes_.get(), policy_.get(), iteration, ranks_.get());
      detail::select_active_kernel<<<
          1u, kThreads,
          static_cast<std::size_t>(kThreads) *
              (sizeof(unsigned long long) + sizeof(std::uint32_t)), stream>>>(
          ranks_.get(), config_.population, config_.active_width, policy_.get(), iteration,
          active_.get(),
          active_count_.get());
    }
    cuda_require(cudaGetLastError(), "launch conjunctive relation completion");
  }

  void complete_cue_async(DevicePopulationView cue, Direction direction,
                          cudaStream_t stream = nullptr) {
    complete_async(cue, direction, stream);
  }

  void appraise_async(long long signed_consequence, cudaStream_t stream = nullptr) {
    detail::appraise_kernel<<<blocks(config_.active_width), kThreads, 0, stream>>>(
        active_.get(), active_count_.get(), signed_consequence, policy_.get(),
        positive_appraisal_.get(), negative_appraisal_.get(), free_mass_.get(), status_.get());
    cuda_require(cudaGetLastError(), "launch resident appraisal");
  }

  void commit_async(ExternalUnitPopulationView surface, cudaStream_t stream = nullptr) {
    if (surface.population == nullptr || surface.unit_count == 0u ||
        surface.population_width == 0u) {
      throw std::invalid_argument("commitment requires AdultState surface populations");
    }
    detail::commit_kernel<<<
        1u, kThreads,
        static_cast<std::size_t>(kThreads) * sizeof(detail::CommitmentCandidate), stream>>>(
        active_.get(), active_count_.get(), surface, positive_appraisal_.get(),
        negative_appraisal_.get(), policy_.get(),
        commitment_addresses_.get(), commitment_count_.get(), commitment_mass_.get(),
        free_mass_.get(), status_.get());
    cuda_require(cudaGetLastError(), "launch resident commitment");
  }

  void autonomous_step_async(ExternalUnitPopulationView surface,
                             cudaStream_t stream = nullptr) {
    commit_async(surface, stream);
  }

  void release_commitment_async(cudaStream_t stream = nullptr) {
    detail::release_commitment_kernel<<<1u, 4u, sizeof(unsigned long long), stream>>>(
        commitment_addresses_.get(), commitment_count_.get(), commitment_mass_.get(),
        free_mass_.get());
    cuda_require(cudaGetLastError(), "launch commitment release");
  }

  void lesion_async(std::uint32_t tag, DevicePopulationView cells,
                    std::uint32_t kind_mask = kAllEdgeKinds,
                    cudaStream_t stream = nullptr) {
    if (tag == 0u || cells.cells == nullptr || cells.count == 0u) {
      throw std::invalid_argument("lesion requires a nonzero tag and resident population");
    }
    detail::tag_lesioned_cells_kernel<<<blocks(cells.count), kThreads, 0, stream>>>(
        cells, config_.population, tag, cell_lesion_tag_.get(), status_.get());
    detail::lesion_edges_kernel<<<blocks(config_.max_edges), kThreads, 0, stream>>>(
        edges_.get(), edge_count_.get(), cells, kind_mask, tag, lesion_mass_.get());
    cuda_require(cudaGetLastError(), "launch reversible population lesion");
  }

  void restore_async(std::uint32_t tag, cudaStream_t stream = nullptr) {
    if (tag == 0u) throw std::invalid_argument("restore requires a nonzero lesion tag");
    detail::restore_edges_kernel<<<blocks(config_.max_edges), kThreads, 0, stream>>>(
        edges_.get(), edge_count_.get(), tag, lesion_mass_.get());
    detail::restore_cells_kernel<<<blocks(config_.population), kThreads, 0, stream>>>(
        cell_lesion_tag_.get(), config_.population, tag);
    cuda_require(cudaGetLastError(), "launch population restore");
  }

  void account_async(cudaStream_t stream = nullptr) {
    cuda_require(cudaMemsetAsync(accounting_.get(), 0, accounting_.bytes(), stream),
                 "clear matter accounting");
    detail::account_edges_kernel<<<blocks(config_.max_edges), kThreads, 0, stream>>>(
        edges_.get(), edge_count_.get(), accounting_.get());
    detail::account_population_kernel<<<blocks(config_.population), kThreads, 0, stream>>>(
        config_.population, positive_appraisal_.get(), negative_appraisal_.get(),
        accounting_.get());
    const unsigned long long* free_mass = free_mass_.get();
    const unsigned long long* policy_mass = policy_mass_.get();
    const unsigned long long* commitment_mass = commitment_mass_.get();
    const unsigned long long* lesion_mass = lesion_mass_.get();
    MatterAccounting* accounting = accounting_.get();
    void* accounting_arguments[] = {
        &free_mass, &policy_mass, &commitment_mass, &lesion_mass, &accounting};
    cuda_require(
        cudaLaunchKernel(reinterpret_cast<const void*>(detail::finalize_accounting_kernel),
                         dim3{1u, 1u, 1u}, dim3{5u, 1u, 1u}, accounting_arguments, 0u, stream),
        "launch parallel exact matter accounting");
    cuda_require(cudaGetLastError(), "launch exact matter accounting");
  }

  MatterAccounting read_accounting(cudaStream_t stream = nullptr) {
    account_async(stream);
    MatterAccounting result;
    cuda_require(cudaMemcpyAsync(&result, accounting_.get(), sizeof(result),
                                 cudaMemcpyDeviceToHost, stream),
                 "read exact matter accounting");
    cuda_require(cudaStreamSynchronize(stream), "synchronize matter accounting");
    return result;
  }

  std::vector<std::uint32_t> read_active(cudaStream_t stream = nullptr) const {
    std::uint32_t count = 0u;
    cuda_require(cudaMemcpyAsync(&count, active_count_.get(), sizeof(count),
                                 cudaMemcpyDeviceToHost, stream),
                 "read active population count");
    cuda_require(cudaStreamSynchronize(stream), "synchronize active population count");
    std::vector<std::uint32_t> result(count);
    if (count != 0u) {
      cuda_require(cudaMemcpyAsync(result.data(), active_.get(), count * sizeof(std::uint32_t),
                                   cudaMemcpyDeviceToHost, stream),
                   "read active population");
      cuda_require(cudaStreamSynchronize(stream), "synchronize active population");
      std::sort(result.begin(), result.end());
    }
    return result;
  }

  std::vector<std::uint32_t> read_commitment(cudaStream_t stream = nullptr) const {
    std::uint32_t count = 0u;
    cuda_require(cudaMemcpyAsync(&count, commitment_count_.get(), sizeof(count),
                                 cudaMemcpyDeviceToHost, stream),
                 "read commitment count");
    cuda_require(cudaStreamSynchronize(stream), "synchronize commitment count");
    std::vector<std::uint32_t> result(count);
    if (count != 0u) {
      cuda_require(cudaMemcpyAsync(result.data(), commitment_addresses_.get(),
                                   count * sizeof(std::uint32_t), cudaMemcpyDeviceToHost, stream),
                   "read commitment addresses");
      cuda_require(cudaStreamSynchronize(stream), "synchronize commitment addresses");
    }
    return result;
  }

  DeviceCommitmentView device_commitment() const {
    return {commitment_addresses_.get(), commitment_count_.get()};
  }

  DevicePopulationView device_active() const {
    return {active_.get(), config_.active_width};
  }

  const std::uint32_t* device_active_count() const { return active_count_.get(); }

  ResidentPolicyState* mutable_policy_device() { return policy_.get(); }
  const ResidentPolicyState* policy_device() const { return policy_.get(); }

  Status read_status(cudaStream_t stream = nullptr) const {
    std::uint32_t status = 0u;
    cuda_require(cudaMemcpyAsync(&status, status_.get(), sizeof(status), cudaMemcpyDeviceToHost,
                                 stream),
                 "read cortex status");
    cuda_require(cudaStreamSynchronize(stream), "synchronize cortex status");
    return static_cast<Status>(status);
  }

  void clear_status_async(cudaStream_t stream = nullptr) {
    cuda_require(cudaMemsetAsync(status_.get(), 0, status_.bytes(), stream),
                 "clear cortex status");
  }

  std::uint32_t read_edge_count(cudaStream_t stream = nullptr) const {
    return read_scalar(edge_count_.get(), "read sparse edge count", stream);
  }

  std::uint32_t read_episode_count(cudaStream_t stream = nullptr) const {
    return read_scalar(episode_count_.get(), "read episode count", stream);
  }

  std::uint32_t read_selected_episode(cudaStream_t stream = nullptr) const {
    return read_scalar(selected_episode_.get(), "read selected episode", stream);
  }

  unsigned long long represented_mass(cudaStream_t stream = nullptr) {
    return read_accounting(stream).represented_mass;
  }

  std::size_t allocated_bytes() const {
    return edges_.bytes() + edge_count_.bytes() + episode_count_.bytes() +
           current_episode_.bytes() + selected_episode_.bytes() + episode_indices_.bytes() +
           index_scratch_.bytes() + anatomy_drive_.bytes() + positive_appraisal_.bytes() +
           negative_appraisal_.bytes() + cell_lesion_tag_.bytes() + index_score_.bytes() +
           index_votes_.bytes() + episode_score_.bytes() + episode_votes_.bytes() +
           directional_score_.bytes() + directional_votes_.bytes() + recurrent_score_.bytes() +
           recurrent_votes_.bytes() + ranks_.bytes() + active_.bytes() + active_count_.bytes() +
           policy_.bytes() + policy_backup_.bytes() + policy_mass_.bytes() +
           policy_lesion_tag_.bytes() +
           commitment_addresses_.bytes() + commitment_count_.bytes() +
           commitment_mass_.bytes() + lesion_mass_.bytes() + free_mass_.bytes() +
           status_.bytes() + accounting_.bytes();
  }

  std::size_t one_dense_matrix_bytes() const {
    return static_cast<std::size_t>(config_.population) * config_.population *
           sizeof(std::uint32_t);
  }

 private:
  static constexpr std::uint32_t kThreads = 256u;

  static std::uint32_t blocks(std::uint64_t count) {
    return static_cast<std::uint32_t>((count + kThreads - 1u) / kThreads);
  }

  void validate_config() const {
    if (config_.population == 0u || config_.active_width == 0u ||
        config_.active_width > config_.population || config_.episode_index_width == 0u ||
        config_.episode_index_width > config_.population || config_.max_edges == 0u ||
        config_.max_episodes == 0u || config_.max_settle_iterations == 0u ||
        config_.policy_mass == 0ull || config_.policy_mass >= config_.represented_mass ||
        config_.warm_start.recurrent_weight == 0u ||
        config_.warm_start.directional_quantum == 0u ||
        config_.warm_start.relation_horizon == 0u ||
        config_.warm_start.binding_weight == 0u ||
        config_.warm_start.completion_width == 0u ||
        config_.warm_start.completion_width > config_.active_width ||
        config_.warm_start.recurrent_settle_iterations > config_.max_settle_iterations ||
        config_.warm_start.commitment_quanta == 0u || config_.represented_mass == 0ull) {
      throw std::invalid_argument("invalid resident relation cortex configuration");
    }
  }

  void clear_query_fields_async(cudaStream_t stream) {
    cuda_require(cudaMemsetAsync(index_score_.get(), 0, index_score_.bytes(), stream),
                 "clear index score");
    cuda_require(cudaMemsetAsync(index_votes_.get(), 0, index_votes_.bytes(), stream),
                 "clear index votes");
    cuda_require(cudaMemsetAsync(episode_score_.get(), 0, episode_score_.bytes(), stream),
                 "clear episode score");
    cuda_require(cudaMemsetAsync(episode_votes_.get(), 0, episode_votes_.bytes(), stream),
                 "clear episode votes");
    cuda_require(cudaMemsetAsync(directional_score_.get(), 0, directional_score_.bytes(), stream),
                 "clear directional score");
    cuda_require(cudaMemsetAsync(directional_votes_.get(), 0, directional_votes_.bytes(), stream),
                 "clear directional votes");
    cuda_require(cudaMemsetAsync(ranks_.get(), 0, ranks_.bytes(), stream), "clear ranks");
  }

  std::uint32_t read_scalar(const std::uint32_t* value, const char* action,
                            cudaStream_t stream) const {
    std::uint32_t result = 0u;
    cuda_require(cudaMemcpyAsync(&result, value, sizeof(result), cudaMemcpyDeviceToHost, stream),
                 action);
    cuda_require(cudaStreamSynchronize(stream), action);
    return result;
  }

  Config config_;
  DeviceBuffer<EdgeEvent> edges_;
  DeviceBuffer<std::uint32_t> edge_count_;
  DeviceBuffer<std::uint32_t> episode_count_;
  DeviceBuffer<std::uint32_t> current_episode_;
  DeviceBuffer<std::uint32_t> selected_episode_;
  DeviceBuffer<std::uint32_t> episode_indices_;
  DeviceBuffer<std::uint32_t> index_scratch_;
  DeviceBuffer<unsigned long long> anatomy_drive_;
  DeviceBuffer<unsigned long long> positive_appraisal_;
  DeviceBuffer<unsigned long long> negative_appraisal_;
  DeviceBuffer<std::uint32_t> cell_lesion_tag_;
  DeviceBuffer<unsigned long long> index_score_;
  DeviceBuffer<std::uint32_t> index_votes_;
  DeviceBuffer<unsigned long long> episode_score_;
  DeviceBuffer<std::uint32_t> episode_votes_;
  DeviceBuffer<unsigned long long> directional_score_;
  DeviceBuffer<std::uint32_t> directional_votes_;
  DeviceBuffer<unsigned long long> recurrent_score_;
  DeviceBuffer<std::uint32_t> recurrent_votes_;
  DeviceBuffer<unsigned long long> ranks_;
  DeviceBuffer<std::uint32_t> active_;
  DeviceBuffer<std::uint32_t> active_count_;
  DeviceBuffer<ResidentPolicyState> policy_;
  DeviceBuffer<ResidentPolicyState> policy_backup_;
  DeviceBuffer<unsigned long long> policy_mass_;
  DeviceBuffer<std::uint32_t> policy_lesion_tag_;
  DeviceBuffer<std::uint32_t> commitment_addresses_;
  DeviceBuffer<std::uint32_t> commitment_count_;
  DeviceBuffer<unsigned long long> commitment_mass_;
  DeviceBuffer<unsigned long long> lesion_mass_;
  DeviceBuffer<unsigned long long> free_mass_;
  DeviceBuffer<std::uint32_t> status_;
  DeviceBuffer<MatterAccounting> accounting_;
};

inline std::uint32_t population_overlap(const std::vector<std::uint32_t>& first,
                                        const std::vector<std::uint32_t>& second) {
  std::vector<std::uint32_t> left = first;
  std::vector<std::uint32_t> right = second;
  std::sort(left.begin(), left.end());
  std::sort(right.begin(), right.end());
  std::uint32_t overlap = 0u;
  std::size_t i = 0u;
  std::size_t j = 0u;
  while (i < left.size() && j < right.size()) {
    if (left[i] == right[j]) {
      ++overlap;
      ++i;
      ++j;
    } else if (left[i] < right[j]) {
      ++i;
    } else {
      ++j;
    }
  }
  return overlap;
}

}  // namespace bcc32_cuda_resident_relation_cortex
