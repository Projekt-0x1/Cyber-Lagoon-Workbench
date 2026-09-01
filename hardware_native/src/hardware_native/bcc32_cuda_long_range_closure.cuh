#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

#include "bcc32_cuda_executor.cuh"
#include "bcc32_types.cuh"

// CUDA reference scaffold for a sparse, distributed, span-level closure organ.
//
// This file deliberately does not claim organism promotion. Relation quanta below
// are dense CUDA counters with an exact integer ledger, not SiteWord matter advanced
// by production F. kPromotableFNative therefore remains false. FNativeTranslationSeam
// names the exact replacement boundary: relation, history, and committed-plan state
// must all become regions of CudaBcc32Executor::device_words(), and every causal
// transition must be an apply_superstep transition before this organ is promotable.
//
// The scaffold still enforces the intended integration shape: all inputs and outputs
// are device-resident POD views, learning consumes sparse SDRs, a complete mid-range
// plan is selected before serialization, and serialization is a dumb SDR-overlap
// readout. There is no host-side semantic selection in the causal path.
namespace bcc32_cuda_long_range_closure {

using substrate::bcc32::CudaBcc32Executor;
using substrate::bcc32::DeviceChunkMap;
using substrate::bcc32::SiteWord;

inline constexpr bool kPromotableFNative = false;
inline constexpr std::uint32_t kInvalid = 0xffffffffu;
inline constexpr std::uint32_t kThreads = 256u;

inline void cuda_require(cudaError_t status, const char* action) {
  if (status != cudaSuccess)
    throw std::runtime_error(std::string(action) + ": " + cudaGetErrorString(status));
}

struct Config {
  std::uint32_t population = 512u;
  std::uint32_t active_count = 16u;
  std::uint32_t history_depth = 4u;
  std::uint32_t max_sequence_units = 256u;
  std::uint32_t max_plan_units = 32u;
  std::uint32_t beam_width = 8u;
  unsigned long long represented_mass = 1ull << 30u;
};

struct BufferCounts {
  std::size_t relation_quanta = 0u;
  std::size_t target_quanta = 0u;
  std::size_t context_cells = 0u;
  std::size_t context_counts = 0u;
  std::size_t beam_slots = 0u;
  std::size_t beam_scores = 0u;
  std::size_t committed_slots = 0u;
  std::size_t committed_cells = 0u;
  std::size_t role_enabled = 0u;
  std::size_t scalar_u32 = 0u;
  std::size_t scalar_u64 = 0u;
};

inline BufferCounts required_buffer_counts(const Config& config) {
  if (config.population == 0u || config.active_count == 0u ||
      config.active_count > config.population || config.history_depth == 0u ||
      config.max_sequence_units < 2u || config.max_plan_units == 0u || config.beam_width == 0u ||
      config.beam_width > 64u || config.represented_mass == 0ull) {
    throw std::invalid_argument("invalid CUDA long-range closure configuration");
  }
  BufferCounts result;
  result.relation_quanta =
      static_cast<std::size_t>(config.history_depth) * config.population * config.population;
  result.target_quanta = static_cast<std::size_t>(config.history_depth) * config.population;
  result.context_cells = static_cast<std::size_t>(config.history_depth) * config.active_count;
  result.context_counts = config.history_depth;
  result.beam_slots = static_cast<std::size_t>(2u) * config.beam_width * config.max_plan_units;
  result.beam_scores = static_cast<std::size_t>(2u) * config.beam_width;
  result.committed_slots = config.max_plan_units;
  result.committed_cells = static_cast<std::size_t>(config.max_plan_units) * config.active_count;
  result.role_enabled = config.history_depth;
  result.scalar_u32 = 3u;  // committed_count, committed_ready, diagnostic flags.
  result.scalar_u64 = 5u;  // free mass plus four accounting fields.
  return result;
}

struct DeviceUnitPopulationView {
  const std::uint32_t* cells = nullptr;
  const std::uint32_t* unit_ids = nullptr;
  std::uint32_t unit_count = 0u;
  std::uint32_t stride = 0u;
  std::uint32_t active_count = 0u;
};

struct DeviceSequenceView {
  const std::uint32_t* units = nullptr;
  const std::uint32_t* segment_ids = nullptr;
  std::uint32_t count = 0u;
};

// rows are ordered oldest -> newest. counts may make any row a partial SDR.
struct DeviceCueView {
  const std::uint32_t* cells = nullptr;
  const std::uint32_t* counts = nullptr;
  std::uint32_t row_count = 0u;
  std::uint32_t stride = 0u;
};

struct DeviceCommittedPlanView {
  const std::uint32_t* cells = nullptr;
  const std::uint32_t* count = nullptr;
  const std::uint32_t* ready = nullptr;
  std::uint32_t stride = 0u;
  std::uint32_t capacity = 0u;
};

struct MatterAccounting {
  unsigned long long free_mass = 0ull;
  unsigned long long relation_mass = 0ull;
  unsigned long long target_mass = 0ull;
  unsigned long long represented_mass = 0ull;
};

// AdultState owns and checkpoints these buffers. No allocation occurs in any API.
struct DeviceStateView {
  std::uint32_t* relation_quanta = nullptr;
  std::uint32_t* target_quanta = nullptr;
  std::uint32_t* context_cells = nullptr;
  std::uint32_t* context_counts = nullptr;
  std::uint32_t* beam_slots = nullptr;
  unsigned long long* beam_scores = nullptr;
  std::uint32_t* committed_slots = nullptr;
  std::uint32_t* committed_cells = nullptr;
  std::uint32_t* role_enabled = nullptr;
  std::uint32_t* committed_count = nullptr;
  std::uint32_t* committed_ready = nullptr;
  std::uint32_t* diagnostic_flags = nullptr;
  unsigned long long* free_mass = nullptr;
  MatterAccounting* accounting = nullptr;
};

// Exact promotion boundary. These ranges must be ordinary sites inside executor,
// not separately allocated CUDA arrays. The promoted implementation must replace:
//   relation_quanta -> reciprocal relation/binding SiteWord matter,
//   context_*       -> bounded path/history SiteWord matter,
//   committed_*     -> upstream committed-plan SiteWord matter.
// advance_production_f()/inverse are the only accepted causal advance at that seam.
struct FNativeTranslationSeam {
  SiteWord* executor_words = nullptr;
  std::uint64_t relation_begin = 0ull;
  std::uint64_t relation_words = 0ull;
  std::uint64_t history_begin = 0ull;
  std::uint64_t history_words = 0ull;
  std::uint64_t committed_plan_begin = 0ull;
  std::uint64_t committed_plan_words = 0ull;
  DeviceChunkMap chunks{};
};

inline void validate_f_native_translation_seam(const CudaBcc32Executor& executor,
                                               const FNativeTranslationSeam& seam) {
  const auto valid_range = [&executor](std::uint64_t begin, std::uint64_t count) {
    return count != 0ull && begin <= executor.site_count() &&
           count <= executor.site_count() - begin;
  };
  const auto disjoint = [](std::uint64_t first_begin, std::uint64_t first_count,
                           std::uint64_t second_begin, std::uint64_t second_count) {
    return first_begin + first_count <= second_begin || second_begin + second_count <= first_begin;
  };
  if (seam.executor_words != executor.device_words() || seam.chunks.slots == nullptr ||
      seam.chunks.chunk_count == 0u || !valid_range(seam.relation_begin, seam.relation_words) ||
      !valid_range(seam.history_begin, seam.history_words) ||
      !valid_range(seam.committed_plan_begin, seam.committed_plan_words) ||
      !disjoint(seam.relation_begin, seam.relation_words, seam.history_begin, seam.history_words) ||
      !disjoint(seam.relation_begin, seam.relation_words, seam.committed_plan_begin,
                seam.committed_plan_words) ||
      !disjoint(seam.history_begin, seam.history_words, seam.committed_plan_begin,
                seam.committed_plan_words)) {
    throw std::invalid_argument("invalid or overlapping F-native closure matter seam");
  }
}

inline void advance_production_f(CudaBcc32Executor& executor, const FNativeTranslationSeam& seam,
                                 std::uint32_t supersteps, cudaStream_t stream = nullptr) {
  validate_f_native_translation_seam(executor, seam);
  for (std::uint32_t tick = 0u; tick < supersteps; ++tick)
    executor.apply_superstep(seam.chunks, stream);
}

inline void advance_production_f_inverse(CudaBcc32Executor& executor,
                                         const FNativeTranslationSeam& seam,
                                         std::uint32_t supersteps, cudaStream_t stream = nullptr) {
  validate_f_native_translation_seam(executor, seam);
  for (std::uint32_t tick = 0u; tick < supersteps; ++tick)
    executor.apply_superstep_inverse(seam.chunks, stream);
}

namespace detail {

__host__ __device__ inline std::uint32_t mix32(std::uint32_t value) {
  value ^= value >> 16u;
  value *= 0x7feb352du;
  value ^= value >> 15u;
  value *= 0x846ca68bu;
  value ^= value >> 16u;
  return value;
}

__device__ inline bool pay(unsigned long long* free_mass, unsigned long long amount) {
  unsigned long long available = atomicAdd(free_mass, 0ull);
  while (available >= amount) {
    const unsigned long long observed = atomicCAS(free_mass, available, available - amount);
    if (observed == available)
      return true;
    available = observed;
  }
  return false;
}

__device__ inline std::uint32_t unit_slot(DeviceUnitPopulationView surface, std::uint32_t unit) {
  if (surface.unit_ids == nullptr)
    return unit < surface.unit_count ? unit : kInvalid;
  for (std::uint32_t slot = 0u; slot < surface.unit_count; ++slot)
    if (surface.unit_ids[slot] == unit)
      return slot;
  return kInvalid;
}

__global__ void encode_opaque_units_kernel(const std::uint32_t* unit_ids, std::uint32_t unit_count,
                                           std::uint32_t population, std::uint32_t active_count,
                                           std::uint32_t stride, std::uint32_t* output_cells) {
  const std::uint32_t slot = blockIdx.x * blockDim.x + threadIdx.x;
  if (slot >= unit_count)
    return;
  const std::uint32_t unit = unit_ids == nullptr ? slot : unit_ids[slot];
  for (std::uint32_t bit = 0u; bit < active_count; ++bit) {
    std::uint32_t attempt = 0u;
    std::uint32_t cell = 0u;
    bool unique = false;
    while (!unique) {
      cell = mix32(unit ^ (0x9e3779b9u * (bit + 1u)) ^ (0x85ebca6bu * attempt++)) % population;
      unique = true;
      for (std::uint32_t prior = 0u; prior < bit; ++prior)
        unique = unique && output_cells[static_cast<std::size_t>(slot) * stride + prior] != cell;
    }
    output_cells[static_cast<std::size_t>(slot) * stride + bit] = cell;
  }
  for (std::uint32_t bit = active_count; bit < stride; ++bit)
    output_cells[static_cast<std::size_t>(slot) * stride + bit] = kInvalid;
}

__global__ void initialize_kernel(DeviceStateView state, Config config) {
  const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index == 0u) {
    *state.free_mass = config.represented_mass;
    *state.committed_count = 0u;
    *state.committed_ready = 0u;
    *state.diagnostic_flags = 0u;
    state.accounting->free_mass = config.represented_mass;
    state.accounting->relation_mass = 0ull;
    state.accounting->target_mass = 0ull;
    state.accounting->represented_mass = config.represented_mass;
  }
  if (index < config.history_depth)
    state.role_enabled[index] = 1u;
}

__global__ void learn_sequences_kernel(DeviceStateView state, Config config,
                                       DeviceUnitPopulationView surface,
                                       DeviceSequenceView sequence, std::uint32_t repetitions) {
  const std::size_t per_target =
      static_cast<std::size_t>(config.history_depth) * config.active_count * config.active_count;
  const std::size_t per_repeat = static_cast<std::size_t>(sequence.count) * per_target;
  const std::size_t event = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (event >= per_repeat * repetitions)
    return;
  std::size_t local = event % per_repeat;
  const std::uint32_t target_position = static_cast<std::uint32_t>(local / per_target);
  local %= per_target;
  const std::uint32_t role =
      static_cast<std::uint32_t>(local / (config.active_count * config.active_count));
  local %= static_cast<std::size_t>(config.active_count) * config.active_count;
  const std::uint32_t source_bit = static_cast<std::uint32_t>(local / config.active_count);
  const std::uint32_t target_bit = static_cast<std::uint32_t>(local % config.active_count);
  const std::uint32_t lag = role + 1u;
  if (target_position < lag || state.role_enabled[role] == 0u)
    return;
  const std::uint32_t source_position = target_position - lag;
  if (sequence.segment_ids != nullptr &&
      sequence.segment_ids[source_position] != sequence.segment_ids[target_position])
    return;
  const std::uint32_t source_slot = unit_slot(surface, sequence.units[source_position]);
  const std::uint32_t target_slot = unit_slot(surface, sequence.units[target_position]);
  if (source_slot == kInvalid || target_slot == kInvalid)
    return;
  const std::uint32_t source =
      surface.cells[static_cast<std::size_t>(source_slot) * surface.stride + source_bit];
  const std::uint32_t target =
      surface.cells[static_cast<std::size_t>(target_slot) * surface.stride + target_bit];
  if (source >= config.population || target >= config.population || !pay(state.free_mass, 2ull))
    return;
  const std::size_t relation =
      (static_cast<std::size_t>(role) * config.population + source) * config.population + target;
  atomicAdd(state.relation_quanta + relation, 1u);
  atomicAdd(state.target_quanta + static_cast<std::size_t>(role) * config.population + target, 1u);
}

__global__ void prepare_context_kernel(DeviceStateView state, Config config, DeviceCueView cue) {
  const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  const std::uint32_t cell_count = config.history_depth * config.active_count;
  if (index < cell_count)
    state.context_cells[index] = kInvalid;
  if (index < config.history_depth)
    state.context_counts[index] = 0u;
  if (index == 0u) {
    *state.committed_count = 0u;
    *state.committed_ready = 0u;
  }
  if (index >= cell_count)
    return;
  const std::uint32_t history = index / config.active_count;
  const std::uint32_t bit = index % config.active_count;
  if (history >= cue.row_count)
    return;
  const std::uint32_t source_row = cue.row_count - 1u - history;
  const std::uint32_t count = min(cue.counts[source_row], config.active_count);
  if (bit < count)
    state.context_cells[index] = cue.cells[static_cast<std::size_t>(source_row) * cue.stride + bit];
  if (bit == 0u)
    state.context_counts[history] = count;
}

__device__ inline unsigned long long closure_score(DeviceStateView state, Config config,
                                                   DeviceUnitPopulationView surface,
                                                   const std::uint32_t* path, std::uint32_t step,
                                                   std::uint32_t candidate_slot) {
  unsigned long long result = 0ull;
  bool any_evidence = false;
  unsigned long long weakest_lift = ~0ull;
  std::uint32_t represented_roles = 0u;
  for (std::uint32_t role = 0u; role < config.history_depth; ++role) {
    if (state.role_enabled[role] == 0u)
      continue;
    const std::uint32_t lag = role + 1u;
    const std::uint32_t* source_cells = nullptr;
    std::uint32_t source_count = 0u;
    if (lag <= step) {
      const std::uint32_t source_slot = path[step - lag];
      source_cells = surface.cells + static_cast<std::size_t>(source_slot) * surface.stride;
      source_count = surface.active_count;
    } else {
      const std::uint32_t history = lag - step - 1u;
      if (history >= config.history_depth)
        continue;
      source_cells = state.context_cells + static_cast<std::size_t>(history) * config.active_count;
      source_count = state.context_counts[history];
    }
    if (source_count == 0u)
      continue;
    unsigned long long evidence = 0ull;
    unsigned long long hub_mass = 0ull;
    std::uint32_t represented_target_cells = 0u;
    const std::uint32_t* target_cells =
        surface.cells + static_cast<std::size_t>(candidate_slot) * surface.stride;
    for (std::uint32_t target_bit = 0u; target_bit < surface.active_count; ++target_bit) {
      const std::uint32_t target = target_cells[target_bit];
      if (target >= config.population)
        continue;
      const std::uint32_t cell_mass =
          state.target_quanta[static_cast<std::size_t>(role) * config.population + target];
      hub_mass += cell_mass;
      represented_target_cells += cell_mass != 0u;
      for (std::uint32_t source_bit = 0u; source_bit < source_count; ++source_bit) {
        const std::uint32_t source = source_cells[source_bit];
        if (source >= config.population)
          continue;
        const std::size_t relation =
            (static_cast<std::size_t>(role) * config.population + source) * config.population +
            target;
        evidence += state.relation_quanta[relation];
      }
    }
    const bool represented_role = represented_target_cells * 2u >= surface.active_count;
    if (represented_role) {
      const unsigned long long lift = (evidence << 20u) / (hub_mass + 1ull);
      weakest_lift = min(weakest_lift, lift);
      ++represented_roles;
      result += lift;
    }
    if (represented_role && evidence != 0ull) {
      any_evidence = true;
      // Learned lift: conditional support normalized by all represented incoming
      // support for the candidate's SDR. This suppresses frequency hubs without a
      // refractory or a privileged semantic score.
      result += min(evidence, 0xffffull);  // deterministic evidence tie-break only.
    }
  }
  if (represented_roles > 1u && weakest_lift != ~0ull)
    result += weakest_lift * 32ull * represented_roles;
  return any_evidence ? result : 0ull;
}

__device__ inline bool expansion_better(unsigned long long score, std::uint32_t parent,
                                        std::uint32_t candidate, unsigned long long other_score,
                                        std::uint32_t other_parent, std::uint32_t other_candidate) {
  if (score != other_score)
    return score > other_score;
  if (candidate != other_candidate)
    return candidate < other_candidate;
  return parent < other_parent;
}

// Deterministic bounded beam commitment. This is a full-span upstream decision: no
// motor byte is exposed until every requested position exists. It is intentionally a
// reference implementation, not the F-native promoted path.
//
// One block expands the beam. Every (parent, candidate) pair scores the same pure
// closure_score reads as the single-thread form, and each thread keeps its own top-kept
// list under expansion_better; staged merges then fold those lists into the shared
// top-kept list with the identical insertion rule. expansion_better is a strict total
// order over (score, candidate, parent), pairs are unique within a step, and scores are
// integer additions, so the sorted top-kept content is independent of enumeration order
// and byte-identical to the serial scan. Path reconstruction stays on thread 0.
__device__ inline void insert_expansion(unsigned long long score, std::uint32_t parent,
                                        std::uint32_t candidate, std::uint32_t kept,
                                        unsigned long long* scores, std::uint32_t* parents,
                                        std::uint32_t* candidates) {
  for (std::uint32_t rank = 0u; rank < kept; ++rank) {
    if (parents[rank] == kInvalid ||
        expansion_better(score, parent, candidate, scores[rank], parents[rank],
                         candidates[rank])) {
      for (std::uint32_t move = kept - 1u; move > rank; --move) {
        scores[move] = scores[move - 1u];
        parents[move] = parents[move - 1u];
        candidates[move] = candidates[move - 1u];
      }
      scores[rank] = score;
      parents[rank] = parent;
      candidates[rank] = candidate;
      break;
    }
  }
}

inline constexpr std::uint32_t kExpansionStageThreads = 16u;
inline constexpr std::uint32_t kExpansionKeptCap = 64u;

__global__ void condition_or_complete_kernel(DeviceStateView state, Config config,
                                             DeviceUnitPopulationView surface,
                                             std::uint32_t plan_length) {
  __shared__ bool proceed;
  __shared__ std::uint32_t shared_live_beams;
  __shared__ unsigned long long stage_scores[kExpansionStageThreads * kExpansionKeptCap];
  __shared__ std::uint32_t stage_parents[kExpansionStageThreads * kExpansionKeptCap];
  __shared__ std::uint32_t stage_candidates[kExpansionStageThreads * kExpansionKeptCap];
  __shared__ std::uint32_t stage_counts[kExpansionStageThreads];
  if (blockIdx.x != 0u)
    return;
  const std::uint32_t tid = threadIdx.x;
  if (tid == 0u)
    proceed = plan_length != 0u && plan_length <= config.max_plan_units &&
              surface.active_count == config.active_count &&
              surface.stride >= surface.active_count;
  __syncthreads();
  if (!proceed) {
    if (tid == 0u)
      *state.diagnostic_flags |= 1u;
    return;
  }
  const std::size_t plane = static_cast<std::size_t>(config.beam_width) * config.max_plan_units;
  std::uint32_t* current_paths = state.beam_slots;
  std::uint32_t* next_paths = state.beam_slots + plane;
  unsigned long long* current_scores = state.beam_scores;
  unsigned long long* next_scores = state.beam_scores + config.beam_width;
  for (std::size_t i = tid; i < plane * 2u; i += blockDim.x)
    state.beam_slots[i] = kInvalid;
  for (std::uint32_t beam = tid; beam < config.beam_width; beam += blockDim.x) {
    current_scores[beam] = 0ull;
    next_scores[beam] = 0ull;
  }
  if (tid == 0u)
    shared_live_beams = 1u;
  __syncthreads();
  std::uint32_t live_beams = 1u;

  for (std::uint32_t step = 0u; step < plan_length; ++step) {
    std::uint32_t top_parent[kExpansionKeptCap];
    std::uint32_t top_candidate[kExpansionKeptCap];
    unsigned long long local_scores[kExpansionKeptCap];
    const std::uint32_t kept = min(config.beam_width, kExpansionKeptCap);
    for (std::uint32_t rank = tid; rank < kept; rank += blockDim.x)
      next_scores[rank] = 0ull;
    for (std::uint32_t rank = 0u; rank < kept; ++rank) {
      local_scores[rank] = 0ull;
      top_parent[rank] = kInvalid;
      top_candidate[rank] = kInvalid;
    }
    const std::uint32_t pairs = live_beams * surface.unit_count;
    for (std::uint32_t index = tid; index < pairs; index += blockDim.x) {
      const std::uint32_t parent = index / surface.unit_count;
      const std::uint32_t candidate = index % surface.unit_count;
      const std::uint32_t* path =
          current_paths + static_cast<std::size_t>(parent) * config.max_plan_units;
      const unsigned long long local =
          closure_score(state, config, surface, path, step, candidate);
      if (local == 0ull)
        continue;
      insert_expansion(current_scores[parent] + local, parent, candidate, kept, local_scores,
                       top_parent, top_candidate);
    }
    for (std::uint32_t base = 0u; base < blockDim.x; base += kExpansionStageThreads) {
      if (tid >= base && tid < base + kExpansionStageThreads) {
        const std::uint32_t slot = tid - base;
        std::uint32_t count = 0u;
        for (std::uint32_t rank = 0u; rank < kept; ++rank) {
          if (top_parent[rank] == kInvalid)
            continue;
          stage_scores[slot * kExpansionKeptCap + count] = local_scores[rank];
          stage_parents[slot * kExpansionKeptCap + count] = top_parent[rank];
          stage_candidates[slot * kExpansionKeptCap + count] = top_candidate[rank];
          ++count;
        }
        stage_counts[slot] = count;
      }
      __syncthreads();
      if (tid == 0u) {
        for (std::uint32_t staged = 0u; staged < kExpansionStageThreads; ++staged) {
          for (std::uint32_t entry = 0u; entry < stage_counts[staged]; ++entry) {
            insert_expansion(stage_scores[staged * kExpansionKeptCap + entry],
                             stage_parents[staged * kExpansionKeptCap + entry],
                             stage_candidates[staged * kExpansionKeptCap + entry], kept,
                             next_scores, top_parent, top_candidate);
          }
        }
      }
      __syncthreads();
    }
    if (tid == 0u) {
      std::uint32_t next_live = 0u;
      for (std::uint32_t rank = 0u; rank < kept; ++rank) {
        if (top_parent[rank] == kInvalid)
          break;
        const std::uint32_t* parent_path =
            current_paths + static_cast<std::size_t>(top_parent[rank]) * config.max_plan_units;
        std::uint32_t* target_path =
            next_paths + static_cast<std::size_t>(rank) * config.max_plan_units;
        for (std::uint32_t prior = 0u; prior < step; ++prior)
          target_path[prior] = parent_path[prior];
        target_path[step] = top_candidate[rank];
        ++next_live;
      }
      shared_live_beams = next_live;
      if (next_live == 0u) {
        *state.diagnostic_flags |= 2u;
        proceed = false;
      }
    }
    __syncthreads();
    if (!proceed)
      return;
    live_beams = shared_live_beams;
    std::uint32_t* path_swap = current_paths;
    current_paths = next_paths;
    next_paths = path_swap;
    unsigned long long* score_swap = current_scores;
    current_scores = next_scores;
    next_scores = score_swap;
  }

  if (tid != 0u)
    return;
  const std::uint32_t* best = current_paths;
  for (std::uint32_t step = 0u; step < plan_length; ++step) {
    const std::uint32_t slot = best[step];
    state.committed_slots[step] = slot;
    for (std::uint32_t bit = 0u; bit < config.active_count; ++bit)
      state.committed_cells[static_cast<std::size_t>(step) * config.active_count + bit] =
          surface.cells[static_cast<std::size_t>(slot) * surface.stride + bit];
  }
  *state.committed_count = plan_length;
  __threadfence();
  *state.committed_ready = 1u;
}

__device__ inline std::uint32_t sdr_overlap(const std::uint32_t* first, const std::uint32_t* second,
                                            std::uint32_t count) {
  std::uint32_t overlap = 0u;
  for (std::uint32_t i = 0u; i < count; ++i)
    for (std::uint32_t j = 0u; j < count; ++j)
      overlap += first[i] == second[j];
  return overlap;
}

// One block resolves each committed plan row against the unit population. The
// per-slot overlap scan fans out across the block and folds through
// better_commit_candidate, whose (overlap desc, slot asc) order is total over
// distinct slots and never promotes a zero overlap, so the folded winner - and
// therefore the serialized motor bytes - match the single-thread scan exactly.
__device__ inline bool better_commit_candidate(std::uint32_t overlap, std::uint32_t slot,
                                               std::uint32_t other_overlap,
                                               std::uint32_t other_slot) {
  if (overlap != other_overlap)
    return overlap > other_overlap;
  return overlap != 0u && slot < other_slot;
}

__global__ void commit_plan_kernel(DeviceStateView state, Config config,
                                   DeviceUnitPopulationView surface,
                                   std::uint32_t* motor_completion, std::uint32_t motor_capacity,
                                   std::uint32_t* motor_count) {
  __shared__ bool ready_and_exact;
  __shared__ std::uint32_t shared_count;
  __shared__ std::uint32_t fold_overlaps[kThreads];
  __shared__ std::uint32_t fold_slots[kThreads];
  if (blockIdx.x != 0u)
    return;
  const std::uint32_t tid = threadIdx.x;
  if (tid == 0u) {
    ready_and_exact = *state.committed_ready != 0u && *state.committed_count <= motor_capacity;
    shared_count = min(*state.committed_count, motor_capacity);
  }
  __syncthreads();
  if (!ready_and_exact)
    return;
  const std::uint32_t count = shared_count;
  for (std::uint32_t step = 0u; step < count; ++step) {
    const std::uint32_t* plan =
        state.committed_cells + static_cast<std::size_t>(step) * config.active_count;
    std::uint32_t best_slot = kInvalid;
    std::uint32_t best_overlap = 0u;
    for (std::uint32_t slot = tid; slot < surface.unit_count; slot += blockDim.x) {
      const std::uint32_t overlap =
          sdr_overlap(plan, surface.cells + static_cast<std::size_t>(slot) * surface.stride,
                      config.active_count);
      if (better_commit_candidate(overlap, slot, best_overlap, best_slot)) {
        best_overlap = overlap;
        best_slot = slot;
      }
    }
    fold_overlaps[tid] = best_overlap;
    fold_slots[tid] = best_slot;
    __syncthreads();
    if (tid == 0u) {
      for (std::uint32_t folded = 1u; folded < blockDim.x; ++folded) {
        if (better_commit_candidate(fold_overlaps[folded], fold_slots[folded], best_overlap,
                                    best_slot)) {
          best_overlap = fold_overlaps[folded];
          best_slot = fold_slots[folded];
        }
      }
      if (best_slot == kInvalid) {
        ready_and_exact = false;
      } else {
        motor_completion[step] =
            surface.unit_ids == nullptr ? best_slot : surface.unit_ids[best_slot];
      }
    }
    __syncthreads();
    if (!ready_and_exact)
      return;
  }
  __threadfence();
  if (tid == 0u)
    *motor_count = count;
}

__global__ void lesion_role_kernel(DeviceStateView state, Config config, std::uint32_t role) {
  const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t relations = static_cast<std::size_t>(config.population) * config.population;
  if (index < relations) {
    const std::uint32_t displaced =
        atomicExch(state.relation_quanta + static_cast<std::size_t>(role) * relations + index, 0u);
    if (displaced != 0u)
      atomicAdd(state.free_mass, static_cast<unsigned long long>(displaced));
  }
  if (index < config.population) {
    const std::uint32_t displaced = atomicExch(
        state.target_quanta + static_cast<std::size_t>(role) * config.population + index, 0u);
    if (displaced != 0u)
      atomicAdd(state.free_mass, static_cast<unsigned long long>(displaced));
  }
  if (index == 0u)
    state.role_enabled[role] = 0u;
}

__global__ void account_mass_kernel(DeviceStateView state, Config config) {
  extern __shared__ unsigned long long partial[];
  const std::uint32_t lane = threadIdx.x;
  unsigned long long relation = 0ull;
  unsigned long long target = 0ull;
  const std::size_t relation_count =
      static_cast<std::size_t>(config.history_depth) * config.population * config.population;
  const std::size_t target_count =
      static_cast<std::size_t>(config.history_depth) * config.population;
  for (std::size_t index = lane; index < relation_count; index += blockDim.x)
    relation += state.relation_quanta[index];
  for (std::size_t index = lane; index < target_count; index += blockDim.x)
    target += state.target_quanta[index];
  partial[lane] = relation;
  partial[blockDim.x + lane] = target;
  __syncthreads();
  for (std::uint32_t stride = blockDim.x / 2u; stride != 0u; stride >>= 1u) {
    if (lane < stride) {
      partial[lane] += partial[lane + stride];
      partial[blockDim.x + lane] += partial[blockDim.x + stride];
    }
    __syncthreads();
  }
  if (lane == 0u) {
    state.accounting->free_mass = *state.free_mass;
    state.accounting->relation_mass = partial[0];
    state.accounting->target_mass = partial[blockDim.x];
    state.accounting->represented_mass =
        *state.free_mass + partial[0] + partial[blockDim.x];
  }
}

}  // namespace detail

inline std::uint32_t blocks(std::size_t count) {
  return static_cast<std::uint32_t>((count + kThreads - 1u) / kThreads);
}

inline void encode_opaque_units_async(const std::uint32_t* device_unit_ids,
                                      std::uint32_t unit_count, std::uint32_t population,
                                      std::uint32_t active_count, std::uint32_t stride,
                                      std::uint32_t* device_output_cells,
                                      cudaStream_t stream = nullptr) {
  if (unit_count == 0u || population == 0u || active_count == 0u || active_count > population ||
      stride < active_count || device_output_cells == nullptr)
    throw std::invalid_argument("invalid opaque SDR encoding view");
  detail::encode_opaque_units_kernel<<<blocks(unit_count), kThreads, 0, stream>>>(
      device_unit_ids, unit_count, population, active_count, stride, device_output_cells);
  cuda_require(cudaGetLastError(), "launch opaque unit SDR encoding");
}

inline void initialize_async(DeviceStateView state, const Config& config,
                             cudaStream_t stream = nullptr) {
  const BufferCounts counts = required_buffer_counts(config);
  cuda_require(cudaMemsetAsync(state.relation_quanta, 0,
                               counts.relation_quanta * sizeof(std::uint32_t), stream),
               "clear closure relation quanta");
  cuda_require(
      cudaMemsetAsync(state.target_quanta, 0, counts.target_quanta * sizeof(std::uint32_t), stream),
      "clear closure target quanta");
  cuda_require(cudaMemsetAsync(state.context_cells, 0xff,
                               counts.context_cells * sizeof(std::uint32_t), stream),
               "clear closure context cells");
  cuda_require(cudaMemsetAsync(state.context_counts, 0,
                               counts.context_counts * sizeof(std::uint32_t), stream),
               "clear closure context counts");
  cuda_require(
      cudaMemsetAsync(state.beam_slots, 0xff, counts.beam_slots * sizeof(std::uint32_t), stream),
      "clear closure beam slots");
  cuda_require(cudaMemsetAsync(state.beam_scores, 0,
                               counts.beam_scores * sizeof(unsigned long long), stream),
               "clear closure beam scores");
  cuda_require(cudaMemsetAsync(state.committed_slots, 0xff,
                               counts.committed_slots * sizeof(std::uint32_t), stream),
               "clear closure committed slots");
  cuda_require(cudaMemsetAsync(state.committed_cells, 0xff,
                               counts.committed_cells * sizeof(std::uint32_t), stream),
               "clear closure committed cells");
  cuda_require(
      cudaMemsetAsync(state.role_enabled, 0, counts.role_enabled * sizeof(std::uint32_t), stream),
      "clear closure role words");
  const std::uint32_t role_blocks = config.history_depth == 0u ? 1u : blocks(config.history_depth);
  detail::initialize_kernel<<<role_blocks, kThreads, 0, stream>>>(state, config);
  cuda_require(cudaGetLastError(), "launch closure initialization");
}

inline void learn_sequences_async(DeviceStateView state, const Config& config,
                                  DeviceUnitPopulationView surface, DeviceSequenceView sequence,
                                  std::uint32_t repetitions = 1u, cudaStream_t stream = nullptr) {
  if (sequence.units == nullptr || sequence.count < 2u ||
      sequence.count > config.max_sequence_units || repetitions == 0u || surface.cells == nullptr ||
      surface.unit_count == 0u || surface.active_count != config.active_count ||
      surface.stride < surface.active_count)
    throw std::invalid_argument("invalid closure learning view");
  const std::size_t events = static_cast<std::size_t>(repetitions) * sequence.count *
                             config.history_depth * config.active_count * config.active_count;
  detail::learn_sequences_kernel<<<blocks(events), kThreads, 0, stream>>>(state, config, surface,
                                                                          sequence, repetitions);
  cuda_require(cudaGetLastError(), "launch directional role learning");
}

inline void condition_or_complete_async(DeviceStateView state, const Config& config,
                                        DeviceUnitPopulationView surface, DeviceCueView cue,
                                        std::uint32_t plan_length, cudaStream_t stream = nullptr) {
  if (cue.cells == nullptr || cue.counts == nullptr || cue.row_count == 0u ||
      cue.stride < config.active_count || cue.row_count > config.history_depth)
    throw std::invalid_argument("invalid partial distributed cue");
  const std::uint32_t clear_count = config.history_depth * config.active_count;
  detail::prepare_context_kernel<<<blocks(clear_count), kThreads, 0, stream>>>(state, config, cue);
  detail::condition_or_complete_kernel<<<1u, kThreads, 0, stream>>>(state, config, surface, plan_length);
  cuda_require(cudaGetLastError(), "launch full-span closure commitment");
}

inline void commit_plan_async(DeviceStateView state, const Config& config,
                              DeviceUnitPopulationView surface,
                              std::uint32_t* device_motor_completion, std::uint32_t motor_capacity,
                              std::uint32_t* device_motor_count, cudaStream_t stream = nullptr) {
  if (device_motor_completion == nullptr || device_motor_count == nullptr || motor_capacity == 0u)
    throw std::invalid_argument("invalid closure motor commitment view");
  detail::commit_plan_kernel<<<1u, kThreads, 0, stream>>>(state, config, surface, device_motor_completion,
                                                    motor_capacity, device_motor_count);
  cuda_require(cudaGetLastError(), "launch dumb closure plan serializer");
}

inline DeviceCommittedPlanView committed_plan(DeviceStateView state, const Config& config) {
  return {state.committed_cells, state.committed_count, state.committed_ready, config.active_count,
          config.max_plan_units};
}

inline void lesion_role_async(DeviceStateView state, const Config& config, std::uint32_t lag,
                              cudaStream_t stream = nullptr) {
  if (lag == 0u || lag > config.history_depth)
    throw std::invalid_argument("invalid closure role lesion");
  const std::size_t count = static_cast<std::size_t>(config.population) * config.population;
  detail::lesion_role_kernel<<<blocks(count), kThreads, 0, stream>>>(state, config, lag - 1u);
  cuda_require(cudaGetLastError(), "launch represented closure role lesion");
}

inline void account_mass_async(DeviceStateView state, const Config& config,
                               cudaStream_t stream = nullptr) {
  detail::account_mass_kernel<<<1u, kThreads,
                                2u * kThreads * sizeof(unsigned long long), stream>>>(state,
                                                                                         config);
  cuda_require(cudaGetLastError(), "launch closure mass accounting");
}

}  // namespace bcc32_cuda_long_range_closure
