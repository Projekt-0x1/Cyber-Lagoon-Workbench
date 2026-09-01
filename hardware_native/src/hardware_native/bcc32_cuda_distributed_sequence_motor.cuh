#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <type_traits>

// CUDA algorithm sidecar for a raw-surface, sparse-distributed sequence motor.
// It is intentionally independent of words, tokens, source spans, and semantic
// labels. AdultState owns every buffer and may retain this view across repeated
// online learn/generate calls. This component is not production F or a language
// proof; it is a bounded mechanism intended for immediate adult integration.
namespace bcc32_cuda_distributed_sequence_motor {

constexpr std::uint32_t kMotorChannels = 256u;
constexpr std::uint32_t kMaxActiveWidth = 32u;
constexpr std::uint32_t kMaxWindowBytes = 16u;
constexpr std::uint32_t kInvalidCell = 0xffffffffu;
constexpr std::uint32_t kCueCoverageBuckets = 4u;
constexpr std::uint32_t kBindingSlots = 16u;
constexpr std::uint32_t kTemporalScales = 4u;
constexpr std::uint32_t kCandidateCapacity = 512u;
constexpr std::uint32_t kPathCapacity = 65536u;
constexpr std::uint32_t kPathRecallDepth = 4u;
constexpr std::uint32_t kSurfacePlanMaxUnits = 19u;
constexpr std::uint32_t kDefaultPopulation = 65536u;
constexpr std::uint32_t kDefaultActiveWidth = 24u;
constexpr std::uint32_t kDefaultWindowBytes = 16u;
constexpr std::uint32_t kDefaultScratchSteps = 4096u;
constexpr unsigned long long kDefaultRepresentedMass = 1ull << 32u;

struct DeviceStateView {
  std::uint32_t population = 0u;
  std::uint32_t active_width = 0u;
  std::uint32_t max_window_bytes = 0u;
  std::uint32_t scratch_steps = 0u;

  // Persistent learned organization.
  std::uint32_t* motor_mass = nullptr;          // population x 256, next-event matter.
  std::uint32_t* motor_cell_support = nullptr;  // total next-event matter per cell.
  std::uint32_t* motor_support = nullptr;       // derived global channel evidence.
  std::uint32_t* binding_keys = nullptr;        // population x kBindingSlots.
  std::uint32_t* binding_mass = nullptr;        // conserved evidence for each binding.
  std::uint32_t* binding_support = nullptr;     // committed binding matter per source.
  std::uint32_t* enabled = nullptr;          // lesionable population cells.
  unsigned long long* free_mass = nullptr;   // explicit uncommitted represented matter.

  // Persistent online sensory state.
  std::uint8_t* history = nullptr;            // max_window_bytes raw body bytes.
  std::uint32_t* history_size = nullptr;      // one device scalar.
  std::uint32_t* previous_active = nullptr;   // active_width cells.
  std::uint32_t* previous_valid = nullptr;    // one device scalar.

  // Adult-owned bounded workspace and motor state.
  std::uint32_t* sequence_active = nullptr;   // scratch_steps x active_width.
  std::uint32_t* cue_active = nullptr;        // persistent whole-cue population.
  std::uint32_t* current_active = nullptr;    // rolling local population.
  std::uint32_t* cue_count = nullptr;         // one device scalar.
  std::uint32_t* current_count = nullptr;     // one device scalar.
  unsigned long long* completion_scores = nullptr;  // population scores.
  unsigned long long* motor_scores = nullptr;       // 256 scores.
  std::uint32_t* candidate_cells = nullptr;          // sparse completion hash table.
  unsigned long long* candidate_scores = nullptr;   // sparse completion evidence.
  std::uint32_t* path_cells = nullptr;               // append-only population path.
  std::uint32_t* path_phases = nullptr;              // phase retained for every path entry.
  std::uint8_t* output_tape = nullptr;               // append-only raw motor history.
  std::uint32_t* path_size = nullptr;                 // one device scalar.
  std::uint32_t* output_tape_size = nullptr;          // one device scalar.
  std::uint32_t* generation_phase = nullptr;          // one device scalar.
  std::uint32_t* local_generation_step = nullptr;     // reset by a new cue, not global history.
  std::uint32_t* initiative_energy = nullptr;          // resident motor-burst pressure.
  std::uint32_t* uncertainty_state = nullptr;         // resident evidence-margin pressure.
  std::uint32_t* quiet_ticks = nullptr;               // elapsed ticks since sensory contact.
  std::uint32_t* emission_active = nullptr;            // resident motor-burst latch.
  std::uint32_t* emitted_count = nullptr;            // one device scalar.
  unsigned long long* mass_audit = nullptr;          // one observer-only device scalar.
};

static_assert(std::is_trivially_copyable_v<DeviceStateView>);
using DistributedSequenceMotorView = DeviceStateView;

struct BufferCounts {
  std::size_t motor_mass = 0u;
  std::size_t motor_cell_support = 0u;
  std::size_t motor_support = 0u;
  std::size_t binding_keys = 0u;
  std::size_t binding_mass = 0u;
  std::size_t binding_support = 0u;
  std::size_t enabled = 0u;
  std::size_t history = 0u;
  std::size_t previous_active = 0u;
  std::size_t sequence_active = 0u;
  std::size_t current_active = 0u;
  std::size_t cue_active = 0u;
  std::size_t completion_scores = 0u;
  std::size_t motor_scores = 0u;
  std::size_t candidate_cells = 0u;
  std::size_t candidate_scores = 0u;
  std::size_t path_cells = 0u;
  std::size_t path_phases = 0u;
  std::size_t output_tape = 0u;
  std::size_t scalar_u32 = 0u;
  std::size_t scalar_u64 = 0u;
};

constexpr BufferCounts required_buffer_counts(
    std::uint32_t population = kDefaultPopulation,
    std::uint32_t active_width = kDefaultActiveWidth,
    std::uint32_t max_window_bytes = kDefaultWindowBytes,
    std::uint32_t scratch_steps = kDefaultScratchSteps) {
  return BufferCounts{
      static_cast<std::size_t>(population) * kMotorChannels,
      population,
      kMotorChannels,
      static_cast<std::size_t>(population) * kBindingSlots,
      static_cast<std::size_t>(population) * kBindingSlots,
      population,
      population,
      max_window_bytes,
      active_width,
      static_cast<std::size_t>(scratch_steps) * active_width,
      active_width,
      active_width,
      population,
      kMotorChannels,
      kCandidateCapacity,
      kCandidateCapacity,
      static_cast<std::size_t>(kPathCapacity) * active_width,
      kPathCapacity,
      kPathCapacity,
      13u,  // sensory/cue/current/emission plus persistent path/phase/appraisal state.
      2u   // free_mass and mass_audit.
  };
}

inline bool valid_view(const DistributedSequenceMotorView& view) {
  return view.population != 0u && view.active_width != 0u &&
         view.active_width <= kMaxActiveWidth && view.max_window_bytes != 0u &&
         view.max_window_bytes <= kMaxWindowBytes && view.scratch_steps != 0u &&
         view.motor_mass != nullptr && view.motor_cell_support != nullptr &&
         view.motor_support != nullptr && view.binding_keys != nullptr &&
         view.binding_mass != nullptr && view.binding_support != nullptr &&
         view.enabled != nullptr && view.free_mass != nullptr && view.history != nullptr &&
         view.history_size != nullptr && view.previous_active != nullptr &&
         view.previous_valid != nullptr && view.sequence_active != nullptr &&
         view.cue_active != nullptr && view.current_active != nullptr &&
         view.cue_count != nullptr && view.current_count != nullptr &&
         view.completion_scores != nullptr && view.motor_scores != nullptr &&
         view.candidate_cells != nullptr && view.candidate_scores != nullptr &&
         view.path_cells != nullptr && view.path_phases != nullptr &&
         view.output_tape != nullptr && view.path_size != nullptr &&
          view.output_tape_size != nullptr && view.generation_phase != nullptr &&
          view.local_generation_step != nullptr && view.initiative_energy != nullptr &&
          view.uncertainty_state != nullptr && view.quiet_ticks != nullptr &&
          view.emission_active != nullptr && view.emitted_count != nullptr &&
          view.mass_audit != nullptr && (view.population & (view.population - 1u)) == 0u &&
          view.population <= (1u << 24u);
}

__host__ __device__ constexpr std::size_t motor_elements(std::uint32_t population) {
  return static_cast<std::size_t>(population) * kMotorChannels;
}

__host__ __device__ constexpr std::size_t binding_elements(std::uint32_t population) {
  return static_cast<std::size_t>(population) * kBindingSlots;
}

constexpr std::size_t sequence_active_elements(std::uint32_t scratch_steps,
                                               std::uint32_t active_width) {
  return static_cast<std::size_t>(scratch_steps) * active_width;
}

__device__ __forceinline__ std::uint32_t mix32(std::uint32_t value) {
  value ^= value >> 16u;
  value *= 0x7feb352du;
  value ^= value >> 15u;
  value *= 0x846ca68bu;
  value ^= value >> 16u;
  return value;
}

__device__ __forceinline__ std::uint32_t population_bits(std::uint32_t population) {
  return 31u - static_cast<std::uint32_t>(__clz(population));
}

__device__ __forceinline__ std::uint32_t binding_key(std::uint32_t source,
                                                     std::uint32_t target,
                                                     std::uint32_t phase,
                                                     std::uint32_t population) {
  const std::uint32_t bits = population_bits(population);
  const std::uint32_t mask = population - 1u;
  const std::uint32_t shift = phase % bits;
  const std::uint32_t rotated =
      ((source << shift) | (source >> ((bits - shift) % bits))) & mask;
  const std::uint32_t bound = (target ^ rotated) & mask;
  return ((phase & 15u) << bits) | bound;
}

__device__ __forceinline__ std::uint32_t unbind_target(std::uint32_t source,
                                                       std::uint32_t key,
                                                       std::uint32_t population) {
  const std::uint32_t bits = population_bits(population);
  const std::uint32_t mask = population - 1u;
  const std::uint32_t phase = (key >> bits) & 15u;
  const std::uint32_t shift = phase % bits;
  const std::uint32_t rotated =
      ((source << shift) | (source >> ((bits - shift) % bits))) & mask;
  return ((key & mask) ^ rotated) & mask;
}

__device__ __forceinline__ std::uint8_t virtual_byte(const std::uint8_t* history,
                                                     std::uint32_t history_size,
                                                     const std::uint8_t* bytes,
                                                     std::uint32_t index) {
  return index < history_size ? history[index] : bytes[index - history_size];
}

__device__ __forceinline__ bool already_selected(const std::uint32_t* cells,
                                                 std::uint32_t count,
                                                 std::uint32_t candidate) {
  for (std::uint32_t index = 0u; index < count; ++index) {
    if (cells[index] == candidate)
      return true;
  }
  return false;
}

__device__ __forceinline__ std::uint32_t encode_pattern(
    const std::uint8_t* history, std::uint32_t history_size, const std::uint8_t* bytes,
    std::uint32_t position, std::uint32_t surface, const DistributedSequenceMotorView& view,
    std::uint32_t* output) {
  const std::uint32_t available = history_size + position + 1u;
  std::uint32_t output_count = 0u;
  for (std::uint32_t slot = 0u; slot < view.active_width; ++slot)
    output[slot] = kInvalidCell;
  for (std::uint32_t slot = 0u; slot < view.active_width; ++slot) {
    const std::uint32_t scale_index = slot % 5u;
    const std::uint32_t requested_span = 1u << scale_index;
    const std::uint32_t span = requested_span < available ? requested_span : available;
    std::uint32_t hash = mix32(surface ^ (0x9e3779b9u * (slot + 1u)) ^ span);
    const std::uint32_t begin = available - span;
    for (std::uint32_t offset = 0u; offset < span; ++offset) {
      const std::uint32_t labelled =
          static_cast<std::uint32_t>(virtual_byte(history, history_size, bytes, begin + offset));
      hash = mix32(hash ^ labelled ^ (0x85ebca6bu * (offset + 1u)));
    }
    std::uint32_t candidate = hash % view.population;
    const std::uint32_t stride = (mix32(hash ^ 0xc2b2ae35u) | 1u) % view.population;
    for (std::uint32_t attempt = 0u; attempt < view.population; ++attempt) {
      if (!already_selected(output, slot, candidate))
        break;
      candidate = (candidate + (stride == 0u ? 1u : stride)) % view.population;
    }
    if (view.enabled[candidate] != 0u) {
      output[slot] = candidate;
      ++output_count;
    }
  }
  return output_count;
}

__global__ void enable_population_kernel(std::uint32_t* enabled, std::uint32_t population) {
  const std::uint32_t cell = blockIdx.x * blockDim.x + threadIdx.x;
  if (cell < population)
    enabled[cell] = 1u;
}

__global__ void reserve_learning_mass_kernel(DistributedSequenceMotorView view,
                                             unsigned long long requested_mass) {
  if (threadIdx.x == 0u) {
    unsigned long long available = atomicAdd(view.free_mass, 0ull);
    while (available >= requested_mass) {
      const unsigned long long observed =
          atomicCAS(view.free_mass, available, available - requested_mass);
      if (observed == available) {
        *view.mass_audit = requested_mass;
        break;
      }
      available = observed;
    }
    if (available < requested_mass)
      *view.mass_audit = ~0ull;
  }
  __syncthreads();
}

__global__ void release_learning_reservation_kernel(DistributedSequenceMotorView view) {
  if (threadIdx.x == 0u) {
    const unsigned long long remainder = *view.mass_audit;
    if (remainder != ~0ull && remainder != 0ull)
      atomicAdd(view.free_mass, remainder);
    *view.mass_audit = 0ull;
  }
  __syncthreads();
}

__global__ void initialize_scalars_kernel(DistributedSequenceMotorView view,
                                          unsigned long long represented_mass) {
  const std::uint32_t field = blockIdx.x * blockDim.x + threadIdx.x;
  switch (field) {
    case 0u:
      *view.free_mass = represented_mass;
      break;
    case 1u:
      *view.history_size = 0u;
      break;
    case 2u:
      *view.previous_valid = 0u;
      break;
    case 3u:
      *view.cue_count = 0u;
      break;
    case 4u:
      *view.current_count = 0u;
      break;
    case 5u:
      *view.path_size = 0u;
      break;
    case 6u:
      *view.output_tape_size = 0u;
      break;
    case 7u:
      *view.generation_phase = 0u;
      break;
    case 8u:
      *view.local_generation_step = 0u;
      break;
    case 9u:
      *view.initiative_energy = 0u;
      break;
    case 10u:
      *view.uncertainty_state = 0u;
      break;
    case 11u:
      *view.quiet_ticks = 0u;
      break;
    case 12u:
      *view.emission_active = 0u;
      break;
    case 13u:
      *view.emitted_count = 0u;
      break;
    case 14u:
      *view.mass_audit = 0ull;
      break;
    default:
      break;
  }
}

__global__ void encode_sequence_windows_kernel(DistributedSequenceMotorView view,
                                               const std::uint8_t* bytes,
                                               std::uint32_t byte_count,
                                               std::uint32_t surface) {
  const std::uint32_t position = blockIdx.x * blockDim.x + threadIdx.x;
  if (position >= byte_count)
    return;
  const std::uint32_t history_size = *view.history_size;
  encode_pattern(view.history, history_size, bytes, position, surface, view,
                 view.sequence_active + static_cast<std::size_t>(position) * view.active_width);
}

__global__ void learn_motor_association_kernel(DistributedSequenceMotorView view,
                                               const std::uint8_t* bytes,
                                               std::uint32_t byte_count) {
  if (*view.mass_audit == ~0ull)
    return;
  __shared__ unsigned int block_commits;
  if (threadIdx.x == 0u)
    block_commits = 0u;
  __syncthreads();
  const std::size_t total = static_cast<std::size_t>(byte_count) * view.active_width;
  for (std::size_t association =
           static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       association < total;
       association += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
    const std::uint32_t target_position =
        static_cast<std::uint32_t>(association / view.active_width);
    const std::uint32_t slot = static_cast<std::uint32_t>(association % view.active_width);
    if (target_position == 0u && *view.previous_valid == 0u)
      continue;
    const std::uint32_t* source =
        target_position == 0u
            ? view.previous_active
            : view.sequence_active +
                  static_cast<std::size_t>(target_position - 1u) * view.active_width;
    const std::uint32_t cell = source[slot];
    if (cell == kInvalidCell)
      continue;
    const std::uint8_t motor_channel = bytes[target_position];
    atomicAdd(view.motor_mass + static_cast<std::size_t>(cell) * kMotorChannels +
                  motor_channel,
              1u);
    atomicAdd(view.motor_cell_support + cell, 1u);
    atomicAdd(view.motor_support + motor_channel, 1u);
    atomicAdd(&block_commits, 1u);
  }
  __syncthreads();
  if (threadIdx.x == 0u && block_commits != 0u)
    atomicAdd(view.mass_audit, 0ull - static_cast<unsigned long long>(block_commits));
}

__global__ void learn_temporal_binding_kernel(DistributedSequenceMotorView view,
                                              std::uint32_t byte_count) {
  if (*view.mass_audit == ~0ull)
    return;
  __shared__ unsigned int block_commits;
  if (threadIdx.x == 0u)
    block_commits = 0u;
  __syncthreads();
  const std::size_t total = static_cast<std::size_t>(byte_count) *
                            view.active_width * kTemporalScales;
  for (std::size_t association =
           static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       association < total;
       association += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
    const std::uint32_t relation =
        static_cast<std::uint32_t>(association % kTemporalScales);
    const std::size_t feature = association / kTemporalScales;
    const std::uint32_t position =
        static_cast<std::uint32_t>(feature / view.active_width);
    const std::uint32_t slot = static_cast<std::uint32_t>(feature % view.active_width);
    const std::uint32_t offset = 1u << (2u * relation);
    const std::uint32_t* source_population = nullptr;
    const std::uint32_t* target_population = nullptr;
    if (relation == 0u) {
      if (position == 0u && *view.previous_valid == 0u)
        continue;
      source_population =
          position == 0u
              ? view.previous_active
              : view.sequence_active +
                    static_cast<std::size_t>(position - 1u) * view.active_width;
      target_population =
          view.sequence_active + static_cast<std::size_t>(position) * view.active_width;
    } else {
      if (position + offset >= byte_count)
        continue;
      source_population =
          view.sequence_active + static_cast<std::size_t>(position) * view.active_width;
      target_population =
          view.sequence_active + static_cast<std::size_t>(position + offset) * view.active_width;
    }
    const std::uint32_t source = source_population[slot];
    const std::uint32_t target = target_population[slot];
    if (source == kInvalidCell || target == kInvalidCell)
      continue;
    const std::uint32_t phase = ((slot + 3u * relation) % 15u) + 1u;
    const std::uint32_t key = binding_key(source, target, phase, view.population);
    std::uint32_t committed_slot = kBindingSlots;
    for (std::uint32_t edge = 0u; edge < kBindingSlots; ++edge) {
      std::uint32_t* key_address =
          view.binding_keys + static_cast<std::size_t>(source) * kBindingSlots + edge;
      const std::uint32_t observed = atomicCAS(key_address, 0u, key);
      if (observed == 0u || observed == key) {
        committed_slot = edge;
        break;
      }
    }
    if (committed_slot == kBindingSlots)
      continue;
    const std::uint32_t represented_amount = relation == 0u ? 4u : 1u;
    atomicAdd(view.binding_mass + static_cast<std::size_t>(source) * kBindingSlots +
                  committed_slot,
              represented_amount);
    atomicAdd(view.binding_support + source, represented_amount);
    atomicAdd(&block_commits, represented_amount);
  }
  __syncthreads();
  if (threadIdx.x == 0u && block_commits != 0u)
    atomicAdd(view.mass_audit, 0ull - static_cast<unsigned long long>(block_commits));
}

__global__ void commit_online_tail_kernel(DistributedSequenceMotorView view,
                                          const std::uint8_t* bytes,
                                          std::uint32_t byte_count) {
  if (blockIdx.x != 0u || threadIdx.x != 0u || byte_count == 0u)
    return;
  std::uint8_t prior[kMaxWindowBytes];
  const std::uint32_t old_size = *view.history_size;
  for (std::uint32_t index = 0u; index < old_size; ++index)
    prior[index] = view.history[index];
  const std::uint32_t combined = old_size + byte_count;
  const std::uint32_t retained = combined < view.max_window_bytes ? combined : view.max_window_bytes;
  const std::uint32_t begin = combined - retained;
  for (std::uint32_t index = 0u; index < retained; ++index) {
    const std::uint32_t source = begin + index;
    view.history[index] =
        source < old_size ? prior[source] : bytes[source - old_size];
  }
  *view.history_size = retained;
  const std::uint32_t* last =
      view.sequence_active + static_cast<std::size_t>(byte_count - 1u) * view.active_width;
  std::uint32_t current_count = 0u;
  for (std::uint32_t slot = 0u; slot < view.active_width; ++slot) {
    view.previous_active[slot] = last[slot];
    view.current_active[slot] = last[slot];
    if (last[slot] != kInvalidCell)
      ++current_count;
  }
  *view.previous_valid = 1u;
  *view.current_count = current_count;
  const std::uint32_t contact_pressure = byte_count > 128u ? 128u : byte_count;
  const std::uint32_t accumulated = *view.initiative_energy + contact_pressure;
  *view.initiative_energy = accumulated > 384u ? 384u : accumulated;
  *view.quiet_ticks = 0u;
  *view.emission_active = 0u;
}

__global__ void commit_sensory_history_kernel(DistributedSequenceMotorView view,
                                              const std::uint8_t* bytes,
                                              std::uint32_t byte_count) {
  if (blockIdx.x != 0u || threadIdx.x != 0u || byte_count == 0u)
    return;
  std::uint8_t prior[kMaxWindowBytes];
  const std::uint32_t old_size = *view.history_size;
  for (std::uint32_t index = 0u; index < old_size; ++index)
    prior[index] = view.history[index];
  const std::uint32_t combined = old_size + byte_count;
  const std::uint32_t retained =
      combined < view.max_window_bytes ? combined : view.max_window_bytes;
  const std::uint32_t begin = combined - retained;
  for (std::uint32_t index = 0u; index < retained; ++index) {
    const std::uint32_t source = begin + index;
    view.history[index] =
        source < old_size ? prior[source] : bytes[source - old_size];
  }
  *view.history_size = retained;
}

__device__ __forceinline__ void select_top_population(DistributedSequenceMotorView view,
                                                     std::uint32_t* output,
                                                     std::uint32_t* output_count) {
  std::uint32_t count = 0u;
  for (std::uint32_t slot = 0u; slot < view.active_width; ++slot) {
    unsigned long long best_score = 0ull;
    std::uint32_t best_cell = kInvalidCell;
    for (std::uint32_t cell = 0u; cell < view.population; ++cell) {
      const unsigned long long score = view.completion_scores[cell];
      if (view.enabled[cell] != 0u &&
          (score > best_score ||
           (score == best_score && score != 0ull && cell < best_cell))) {
        best_score = score;
        best_cell = cell;
      }
    }
    if (best_cell == kInvalidCell)
      break;
    output[count++] = best_cell;
    view.completion_scores[best_cell] = 0ull;
  }
  for (std::uint32_t slot = count; slot < view.active_width; ++slot)
    output[slot] = kInvalidCell;
  *output_count = count;
}

__device__ __forceinline__ void score_motor_channels(DistributedSequenceMotorView view,
                                                     const std::uint32_t* active,
                                                     std::uint32_t active_count,
                                                     const std::uint32_t* cue,
                                                     std::uint32_t cue_count) {
  const bool first_local_tick = *view.local_generation_step == 0u;
  for (std::uint32_t channel = threadIdx.x; channel < kMotorChannels;
       channel += blockDim.x) {
    unsigned long long evidence = 0ull;
    for (std::uint32_t slot = 0u; slot < active_count; ++slot) {
      const std::uint32_t cell = active[slot];
      if (cell != kInvalidCell) {
        const unsigned long long count =
            view.motor_mass[static_cast<std::size_t>(cell) * kMotorChannels + channel];
        const unsigned long long support = view.motor_cell_support[cell];
        evidence += (count << 24u) / (support + 8ull);
      }
    }
    for (std::uint32_t slot = 0u; slot < cue_count; ++slot) {
      const std::uint32_t cell = cue[slot];
      if (cell != kInvalidCell) {
        const unsigned long long count =
            view.motor_mass[static_cast<std::size_t>(cell) * kMotorChannels + channel];
        const unsigned long long support = view.motor_cell_support[cell];
        const unsigned long long cue_gain = first_local_tick ? 4ull : 1ull;
        evidence += cue_gain * ((count << 20u) / (support + 8ull));
      }
    }
    view.motor_scores[channel] = evidence;
  }
}

__device__ __forceinline__ std::uint32_t select_motor_channel(
    const DistributedSequenceMotorView& view, unsigned long long* selected_score,
    unsigned long long* runner_up_score) {
  unsigned long long best_score = 0ull;
  unsigned long long second_score = 0ull;
  std::uint32_t best_channel = 0u;
  for (std::uint32_t channel = 0u; channel < kMotorChannels; ++channel) {
    const unsigned long long score = view.motor_scores[channel];
    if (score > best_score) {
      second_score = best_score;
      best_score = score;
      best_channel = channel;
    } else if (score > second_score) {
      second_score = score;
    }
  }
  *selected_score = best_score;
  *runner_up_score = second_score;
  return best_score == 0ull ? kInvalidCell : best_channel;
}

__device__ __forceinline__ std::uint32_t select_spontaneous_motor_channel(
    const DistributedSequenceMotorView& view) {
  const std::uint32_t start =
      mix32(*view.generation_phase ^ *view.uncertainty_state ^ 0x51ed270bu) & 0xffu;
  for (std::uint32_t offset = 0u; offset < kMotorChannels; ++offset) {
    const std::uint32_t channel = (start + offset) & 0xffu;
    if (view.motor_support[channel] != 0u)
      return channel;
  }
  return kInvalidCell;
}

__device__ __forceinline__ void score_spontaneous_reactivation(
    DistributedSequenceMotorView view, std::uint32_t channel) {
  for (std::uint32_t cell = threadIdx.x; cell < view.population;
       cell += blockDim.x) {
    const std::uint32_t support = view.motor_cell_support[cell];
    const std::uint32_t channel_mass =
        channel == kInvalidCell
            ? 0u
            : view.motor_mass[static_cast<std::size_t>(cell) * kMotorChannels + channel];
    if (view.enabled[cell] == 0u || support == 0u || channel_mass == 0u) {
      view.completion_scores[cell] = 0ull;
      continue;
    }
    const unsigned long long normalized =
        (static_cast<unsigned long long>(channel_mass) << 32u) / support;
    const unsigned long long bounded_support = support < 256u ? support : 255u;
    view.completion_scores[cell] =
        (normalized << 16u) | (bounded_support << 8u) |
        (mix32(cell ^ *view.generation_phase) & 0xffu);
  }
}

__device__ __forceinline__ void insert_completion_candidate(
    DistributedSequenceMotorView view, std::uint32_t target,
    unsigned long long evidence) {
  if (target >= view.population || view.enabled[target] == 0u || evidence == 0ull)
    return;
  std::uint32_t bucket = mix32(target) & (kCandidateCapacity - 1u);
  for (std::uint32_t probe = 0u; probe < kCandidateCapacity; ++probe) {
    std::uint32_t* address = view.candidate_cells + bucket;
    const std::uint32_t observed = atomicCAS(address, kInvalidCell, target);
    if (observed == kInvalidCell || observed == target) {
      atomicAdd(view.candidate_scores + bucket, evidence);
      return;
    }
    bucket = (bucket + 1u) & (kCandidateCapacity - 1u);
  }
}

__device__ __forceinline__ void accumulate_binding_candidates(
    DistributedSequenceMotorView view, const std::uint32_t* active,
    std::uint32_t active_count, unsigned long long gain) {
  const std::size_t total = static_cast<std::size_t>(active_count) * kBindingSlots;
  for (std::size_t item = threadIdx.x; item < total; item += blockDim.x) {
    const std::uint32_t source_slot = static_cast<std::uint32_t>(item / kBindingSlots);
    const std::uint32_t edge = static_cast<std::uint32_t>(item % kBindingSlots);
    const std::uint32_t source = active[source_slot];
    if (source == kInvalidCell || view.enabled[source] == 0u)
      continue;
    const std::size_t binding = static_cast<std::size_t>(source) * kBindingSlots + edge;
    const std::uint32_t key = view.binding_keys[binding];
    const unsigned long long amount = view.binding_mass[binding];
    if (key == 0u || amount == 0ull)
      continue;
    const unsigned long long support = view.binding_support[source];
    const unsigned long long evidence = gain * ((amount << 20u) / (support + 4ull));
    insert_completion_candidate(view, unbind_target(source, key, view.population), evidence);
  }
}

__device__ __forceinline__ std::uint32_t select_sparse_successor(
    DistributedSequenceMotorView view) {
  std::uint32_t count = 0u;
  std::uint32_t* successor = view.sequence_active;
  const std::uint32_t direct_limit = view.active_width / 2u;
  for (std::uint32_t source_slot = 0u;
       source_slot < *view.current_count && count < direct_limit;
       ++source_slot) {
    const std::uint32_t source = view.current_active[source_slot];
    if (source == kInvalidCell || view.enabled[source] == 0u)
      continue;
    std::uint32_t best_target = kInvalidCell;
    std::uint32_t best_amount = 0u;
    std::uint32_t best_key = 0u;
    for (std::uint32_t edge = 0u; edge < kBindingSlots; ++edge) {
      const std::size_t binding = static_cast<std::size_t>(source) * kBindingSlots + edge;
      const std::uint32_t key = view.binding_keys[binding];
      const std::uint32_t amount = view.binding_mass[binding];
      if (key == 0u || amount == 0u)
        continue;
      const std::uint32_t target = unbind_target(source, key, view.population);
      if (view.enabled[target] != 0u &&
          (amount > best_amount ||
           (amount == best_amount && (key < best_key || best_key == 0u)))) {
        best_amount = amount;
        best_key = key;
        best_target = target;
      }
    }
    if (best_target != kInvalidCell &&
        !already_selected(successor, count, best_target))
      successor[count++] = best_target;
  }
  for (std::uint32_t slot = count; slot < view.active_width; ++slot) {
    unsigned long long best_score = 0ull;
    std::uint32_t best_cell = kInvalidCell;
    std::uint32_t best_bucket = kCandidateCapacity;
    for (std::uint32_t bucket = 0u; bucket < kCandidateCapacity; ++bucket) {
      const std::uint32_t candidate = view.candidate_cells[bucket];
      const unsigned long long score = view.candidate_scores[bucket];
      if (candidate != kInvalidCell && view.enabled[candidate] != 0u &&
          !already_selected(successor, count, candidate) &&
          (score > best_score ||
           (score == best_score && score != 0ull && candidate < best_cell))) {
        best_score = score;
        best_cell = candidate;
        best_bucket = bucket;
      }
    }
    if (best_bucket == kCandidateCapacity)
      break;
    successor[count++] = best_cell;
    view.candidate_scores[best_bucket] = 0ull;
  }
  for (std::uint32_t slot = 0u; slot < count; ++slot)
    view.current_active[slot] = successor[slot];
  for (std::uint32_t slot = count; slot < view.active_width; ++slot) {
    successor[slot] = kInvalidCell;
    view.current_active[slot] = kInvalidCell;
  }
  *view.current_count = count;
  return count;
}

__global__ void accumulate_raw_cue_kernel(DistributedSequenceMotorView view,
                                          const std::uint8_t* cue,
                                          std::uint32_t cue_size,
                                          std::uint32_t surface) {
  const std::uint32_t position = blockIdx.x * blockDim.x + threadIdx.x;
  if (position >= cue_size)
    return;
  std::uint32_t local[kMaxActiveWidth];
  const std::uint32_t count =
      encode_pattern(nullptr, 0u, cue, position, surface, view, local);
  std::uint32_t* stored =
      view.sequence_active + static_cast<std::size_t>(position) * view.active_width;
  (void)count;
  for (std::uint32_t slot = 0u; slot < view.active_width; ++slot)
    stored[slot] = local[slot];
  for (std::uint32_t slot = 0u; slot < view.active_width; ++slot) {
    if (local[slot] == kInvalidCell)
      continue;
    // Generic multiscale evidence: longer raw windows contribute more matter
    // to the conditioned population, without assigning meaning to any byte.
    const std::uint32_t scale_index = slot % 5u;
    const unsigned long long strength = 1ull << (2u * scale_index);
    atomicAdd(view.completion_scores + local[slot], strength);
  }
}

__global__ void accumulate_encoded_sensory_kernel(DistributedSequenceMotorView view,
                                                   std::uint32_t sensory_size) {
  const std::uint32_t position = blockIdx.x * blockDim.x + threadIdx.x;
  if (position >= sensory_size)
    return;
  const std::uint32_t* active =
      view.sequence_active + static_cast<std::size_t>(position) * view.active_width;
  for (std::uint32_t slot = 0u; slot < view.active_width; ++slot) {
    if (active[slot] == kInvalidCell)
      continue;
    const std::uint32_t scale_index = slot % 5u;
    atomicAdd(view.completion_scores + active[slot], 1ull << (2u * scale_index));
  }
}

__device__ __forceinline__ unsigned long long cue_bucket_score(
    DistributedSequenceMotorView view, std::uint32_t candidate,
    std::uint32_t position_begin, std::uint32_t position_end) {
  unsigned long long score = 0ull;
  for (std::uint32_t position = position_begin; position < position_end; ++position) {
    const std::uint32_t* active =
        view.sequence_active + static_cast<std::size_t>(position) * view.active_width;
    for (std::uint32_t slot = 0u; slot < view.active_width; ++slot) {
      if (active[slot] == candidate)
        score += 1ull << (2u * (slot % 5u));
    }
  }
  return score;
}

__global__ void select_raw_cue_population_kernel(DistributedSequenceMotorView view,
                                                 const std::uint8_t* cue,
                                                 std::uint32_t cue_size) {
  if (blockIdx.x != 0u || threadIdx.x != 0u)
    return;
  const std::uint32_t bucket_count =
      cue_size < kCueCoverageBuckets ? cue_size : kCueCoverageBuckets;
  std::uint32_t selected_count = 0u;
  const std::uint32_t* terminal =
      view.sequence_active + static_cast<std::size_t>(cue_size - 1u) * view.active_width;
  const std::uint32_t coverage_slots = view.active_width;
  for (std::uint32_t bucket = 0u; bucket < bucket_count; ++bucket) {
    const std::uint32_t position_begin = (bucket * cue_size) / bucket_count;
    const std::uint32_t position_end = ((bucket + 1u) * cue_size) / bucket_count;
    const std::uint32_t quota_begin = (bucket * coverage_slots) / bucket_count;
    const std::uint32_t quota_end = ((bucket + 1u) * coverage_slots) / bucket_count;
    for (std::uint32_t quota = quota_begin; quota < quota_end; ++quota) {
      (void)quota;
      unsigned long long best_score = 0ull;
      unsigned long long best_global = 0ull;
      std::uint32_t best_cell = kInvalidCell;
      for (std::uint32_t position = position_begin; position < position_end; ++position) {
        const std::uint32_t* active =
            view.sequence_active + static_cast<std::size_t>(position) * view.active_width;
        for (std::uint32_t slot = 0u; slot < view.active_width; ++slot) {
          const std::uint32_t candidate = active[slot];
          if (candidate == kInvalidCell ||
              already_selected(view.cue_active, selected_count, candidate))
            continue;
          const unsigned long long score =
              cue_bucket_score(view, candidate, position_begin, position_end);
          const unsigned long long global = view.completion_scores[candidate];
          if (score > best_score ||
              (score == best_score && global > best_global) ||
              (score == best_score && global == best_global && candidate < best_cell)) {
            best_score = score;
            best_global = global;
            best_cell = candidate;
          }
        }
      }
      if (best_cell != kInvalidCell)
        view.cue_active[selected_count++] = best_cell;
    }
  }
  for (std::uint32_t slot = selected_count; slot < view.active_width; ++slot)
    view.cue_active[slot] = kInvalidCell;
  *view.cue_count = selected_count;

  std::uint32_t local_count = 0u;
  for (std::uint32_t slot = 0u; slot < view.active_width; ++slot) {
    view.current_active[slot] = terminal[slot];
    if (terminal[slot] != kInvalidCell)
      ++local_count;
  }
  *view.current_count = local_count;
  const std::uint32_t retained =
      cue_size < view.max_window_bytes ? cue_size : view.max_window_bytes;
  const std::uint32_t begin = cue_size - retained;
  for (std::uint32_t index = 0u; index < retained; ++index)
    view.history[index] = cue[begin + index];
  *view.history_size = retained;
  *view.emitted_count = 0u;
  *view.local_generation_step = 0u;
  const std::uint32_t contact_pressure = cue_size > 64u ? 192u : cue_size * 3u;
  const std::uint32_t accumulated = *view.initiative_energy + contact_pressure;
  *view.initiative_energy = accumulated > 384u ? 384u : accumulated;
  *view.quiet_ticks = 0u;
}

__global__ void select_appended_sensory_population_kernel(
    DistributedSequenceMotorView view, std::uint32_t sensory_size) {
  if (blockIdx.x != 0u || threadIdx.x != 0u)
    return;
  const std::uint32_t bucket_count =
      sensory_size < kCueCoverageBuckets ? sensory_size : kCueCoverageBuckets;
  std::uint32_t selected_count = 0u;
  for (std::uint32_t bucket = 0u; bucket < bucket_count; ++bucket) {
    const std::uint32_t position_begin = (bucket * sensory_size) / bucket_count;
    const std::uint32_t position_end = ((bucket + 1u) * sensory_size) / bucket_count;
    const std::uint32_t quota_begin = (bucket * view.active_width) / bucket_count;
    const std::uint32_t quota_end = ((bucket + 1u) * view.active_width) / bucket_count;
    for (std::uint32_t quota = quota_begin; quota < quota_end; ++quota) {
      (void)quota;
      unsigned long long best_score = 0ull;
      unsigned long long best_global = 0ull;
      std::uint32_t best_cell = kInvalidCell;
      for (std::uint32_t position = position_begin; position < position_end; ++position) {
        const std::uint32_t* active =
            view.sequence_active + static_cast<std::size_t>(position) * view.active_width;
        for (std::uint32_t slot = 0u; slot < view.active_width; ++slot) {
          const std::uint32_t candidate = active[slot];
          if (candidate == kInvalidCell ||
              already_selected(view.cue_active, selected_count, candidate))
            continue;
          const unsigned long long score =
              cue_bucket_score(view, candidate, position_begin, position_end);
          const unsigned long long global = view.completion_scores[candidate];
          if (score > best_score ||
              (score == best_score && global > best_global) ||
              (score == best_score && global == best_global && candidate < best_cell)) {
            best_score = score;
            best_global = global;
            best_cell = candidate;
          }
        }
      }
      if (best_cell != kInvalidCell)
        view.cue_active[selected_count++] = best_cell;
    }
  }
  for (std::uint32_t slot = selected_count; slot < view.active_width; ++slot)
    view.cue_active[slot] = kInvalidCell;
  *view.cue_count = selected_count;
}

__global__ void seed_motor_channel_kernel(DistributedSequenceMotorView view,
                                          std::uint8_t motor_channel) {
  for (std::uint32_t cell = threadIdx.x; cell < view.population; cell += blockDim.x) {
    view.completion_scores[cell] =
        view.enabled[cell] == 0u
            ? 0ull
            : static_cast<unsigned long long>(
                  view.motor_mass[static_cast<std::size_t>(cell) * kMotorChannels + motor_channel]);
  }
  __syncthreads();
  if (threadIdx.x == 0u)
    select_top_population(view, view.current_active, view.current_count);
  __syncthreads();
  if (threadIdx.x == 0u)
    *view.emitted_count = 0u;
}

#include "bcc32_cuda_distributed_sequence_motor_generation_kernels.inl"

inline cudaError_t initialize_async(const DistributedSequenceMotorView& view,
                                    unsigned long long represented_mass,
                                    cudaStream_t stream = nullptr) {
  if (!valid_view(view) || represented_mass == 0ull)
    return cudaErrorInvalidValue;
  cudaError_t status = cudaMemsetAsync(view.motor_mass, 0,
                           motor_elements(view.population) * sizeof(std::uint32_t), stream);
  if (status != cudaSuccess)
    return status;
  status = cudaMemsetAsync(view.motor_cell_support, 0,
                           view.population * sizeof(std::uint32_t), stream);
  if (status != cudaSuccess)
    return status;
  status = cudaMemsetAsync(view.motor_support, 0,
                           kMotorChannels * sizeof(std::uint32_t), stream);
  if (status != cudaSuccess)
    return status;
  status = cudaMemsetAsync(view.binding_keys, 0,
                           binding_elements(view.population) * sizeof(std::uint32_t), stream);
  if (status != cudaSuccess)
    return status;
  status = cudaMemsetAsync(view.binding_mass, 0,
                           binding_elements(view.population) * sizeof(std::uint32_t), stream);
  if (status != cudaSuccess)
    return status;
  status = cudaMemsetAsync(view.binding_support, 0,
                           view.population * sizeof(std::uint32_t), stream);
  if (status != cudaSuccess)
    return status;
  status = cudaMemsetAsync(view.sequence_active, 0xff,
                           sequence_active_elements(view.scratch_steps, view.active_width) *
                               sizeof(std::uint32_t),
                           stream);
  if (status != cudaSuccess)
    return status;
  status = cudaMemsetAsync(view.previous_active, 0xff,
                           view.active_width * sizeof(std::uint32_t), stream);
  if (status != cudaSuccess)
    return status;
  status = cudaMemsetAsync(view.current_active, 0xff,
                           view.active_width * sizeof(std::uint32_t), stream);
  if (status != cudaSuccess)
    return status;
  status = cudaMemsetAsync(view.cue_active, 0xff,
                           view.active_width * sizeof(std::uint32_t), stream);
  if (status != cudaSuccess)
    return status;
  status = cudaMemsetAsync(view.candidate_cells, 0xff,
                           kCandidateCapacity * sizeof(std::uint32_t), stream);
  if (status != cudaSuccess)
    return status;
  status = cudaMemsetAsync(view.candidate_scores, 0,
                           kCandidateCapacity * sizeof(unsigned long long), stream);
  if (status != cudaSuccess)
    return status;
  status = cudaMemsetAsync(view.path_cells, 0xff,
                           static_cast<std::size_t>(kPathCapacity) * view.active_width *
                               sizeof(std::uint32_t),
                           stream);
  if (status != cudaSuccess)
    return status;
  status = cudaMemsetAsync(view.path_phases, 0,
                           kPathCapacity * sizeof(std::uint32_t), stream);
  if (status != cudaSuccess)
    return status;
  status = cudaMemsetAsync(view.output_tape, 0,
                           kPathCapacity * sizeof(std::uint8_t), stream);
  if (status != cudaSuccess)
    return status;
  enable_population_kernel<<<(view.population + 255u) / 256u, 256u, 0u, stream>>>(
      view.enabled, view.population);
  DistributedSequenceMotorView initialized_view = view;
  void* initialization_arguments[] = {&initialized_view, &represented_mass};
  return cudaLaunchKernel(
      reinterpret_cast<const void*>(initialize_scalars_kernel), dim3{1u, 1u, 1u},
      dim3{15u, 1u, 1u}, initialization_arguments, 0u, stream);
}

inline cudaError_t reset_online_stream_async(const DistributedSequenceMotorView& view,
                                             cudaStream_t stream = nullptr) {
  if (!valid_view(view))
    return cudaErrorInvalidValue;
  cudaError_t status = cudaMemsetAsync(view.history_size, 0, sizeof(std::uint32_t), stream);
  if (status != cudaSuccess)
    return status;
  status = cudaMemsetAsync(view.previous_valid, 0, sizeof(std::uint32_t), stream);
  if (status != cudaSuccess)
    return status;
  return cudaMemsetAsync(view.previous_active, 0xff,
                         view.active_width * sizeof(std::uint32_t), stream);
}

inline cudaError_t learn_online_async(const DistributedSequenceMotorView& view,
                                      const std::uint8_t* device_bytes,
                                      std::uint32_t byte_count,
                                      std::uint32_t surface = 0u,
                                      cudaStream_t stream = nullptr) {
  if (!valid_view(view) || device_bytes == nullptr || byte_count == 0u ||
      byte_count > view.scratch_steps)
    return cudaErrorInvalidValue;
  encode_sequence_windows_kernel<<<(byte_count + 255u) / 256u, 256u, 0u, stream>>>(
      view, device_bytes, byte_count, surface);
  const std::size_t motor_work = static_cast<std::size_t>(byte_count) * view.active_width;
  const unsigned long long requested_mass =
      8ull * static_cast<unsigned long long>(motor_work);
  DistributedSequenceMotorView reserved_view = view;
  unsigned long long reserved_mass = requested_mass;
  void* reservation_arguments[] = {&reserved_view, &reserved_mass};
  cudaError_t reservation_status = cudaLaunchKernel(
      reinterpret_cast<const void*>(reserve_learning_mass_kernel), dim3{1u, 1u, 1u},
      dim3{2u, 1u, 1u}, reservation_arguments, 0u, stream);
  if (reservation_status != cudaSuccess)
    return reservation_status;
  const std::uint32_t motor_blocks =
      static_cast<std::uint32_t>((motor_work + 255u) / 256u < 4096u
                                     ? (motor_work + 255u) / 256u
                                     : 4096u);
  learn_motor_association_kernel<<<motor_blocks, 256u, 0u, stream>>>(view, device_bytes,
                                                                    byte_count);
  learn_temporal_binding_kernel<<<motor_blocks, 256u, 0u, stream>>>(view, byte_count);
  DistributedSequenceMotorView released_view = view;
  void* release_arguments[] = {&released_view};
  cudaError_t release_status = cudaLaunchKernel(
      reinterpret_cast<const void*>(release_learning_reservation_kernel), dim3{1u, 1u, 1u},
      dim3{2u, 1u, 1u}, release_arguments, 0u, stream);
  if (release_status != cudaSuccess)
    return release_status;
  commit_online_tail_kernel<<<1u, 1u, 0u, stream>>>(view, device_bytes, byte_count);
  return cudaGetLastError();
}

inline cudaError_t seed_raw_async(const DistributedSequenceMotorView& view,
                                  const std::uint8_t* device_cue,
                                  std::uint32_t cue_size,
                                  std::uint32_t surface = 0u,
                                  cudaStream_t stream = nullptr) {
  if (!valid_view(view) || device_cue == nullptr || cue_size == 0u ||
      cue_size > view.scratch_steps)
    return cudaErrorInvalidValue;
  cudaError_t status = cudaMemsetAsync(view.completion_scores, 0,
                                      view.population * sizeof(unsigned long long), stream);
  if (status != cudaSuccess)
    return status;
  accumulate_raw_cue_kernel<<<(cue_size + 255u) / 256u, 256u, 0u, stream>>>(
      view, device_cue, cue_size, surface);
  select_raw_cue_population_kernel<<<1u, 1u, 0u, stream>>>(view, device_cue, cue_size);
  return cudaGetLastError();
}

inline cudaError_t append_sensory_async(const DistributedSequenceMotorView& view,
                                        std::uint32_t sensory_size,
                                        cudaStream_t stream = nullptr) {
  if (!valid_view(view) || sensory_size == 0u || sensory_size > view.scratch_steps)
    return cudaErrorInvalidValue;
  cudaError_t status = cudaMemsetAsync(view.completion_scores, 0,
                                      view.population * sizeof(unsigned long long), stream);
  if (status != cudaSuccess)
    return status;
  accumulate_encoded_sensory_kernel<<<(sensory_size + 255u) / 256u, 256u, 0u, stream>>>(
      view, sensory_size);
  select_appended_sensory_population_kernel<<<1u, 1u, 0u, stream>>>(view, sensory_size);
  return cudaGetLastError();
}

inline cudaError_t sense_raw_async(const DistributedSequenceMotorView& view,
                                   const std::uint8_t* device_bytes,
                                   std::uint32_t byte_count,
                                   std::uint32_t surface = 0u,
                                   cudaStream_t stream = nullptr) {
  if (!valid_view(view) || device_bytes == nullptr || byte_count == 0u ||
      byte_count > view.scratch_steps)
    return cudaErrorInvalidValue;
  encode_sequence_windows_kernel<<<(byte_count + 255u) / 256u, 256u, 0u, stream>>>(
      view, device_bytes, byte_count, surface);
  commit_sensory_history_kernel<<<1u, 1u, 0u, stream>>>(view, device_bytes, byte_count);
  cudaError_t status = cudaGetLastError();
  if (status != cudaSuccess)
    return status;
  return append_sensory_async(view, byte_count, stream);
}

inline cudaError_t clear_cue_async(const DistributedSequenceMotorView& view,
                                   cudaStream_t stream = nullptr) {
  if (!valid_view(view))
    return cudaErrorInvalidValue;
  cudaError_t status = cudaMemsetAsync(
      view.cue_active, 0xff, view.active_width * sizeof(std::uint32_t), stream);
  if (status != cudaSuccess)
    return status;
  status = cudaMemsetAsync(view.cue_count, 0, sizeof(std::uint32_t), stream);
  return status;
}

inline cudaError_t seed_motor_async(const DistributedSequenceMotorView& view,
                                    std::uint8_t motor_channel,
                                    cudaStream_t stream = nullptr) {
  if (!valid_view(view))
    return cudaErrorInvalidValue;
  seed_motor_channel_kernel<<<1u, 256u, 0u, stream>>>(view, motor_channel);
  return cudaGetLastError();
}

inline cudaError_t generate_async(const DistributedSequenceMotorView& view,
                                  std::uint8_t* device_output,
                                  std::uint32_t output_capacity,
                                  std::uint32_t completion_iterations = 2u,
                                  cudaStream_t stream = nullptr) {
  if (!valid_view(view) || device_output == nullptr || output_capacity == 0u)
    return cudaErrorInvalidValue;
  generate_sequence_kernel<<<1u, 256u, 0u, stream>>>(view, device_output, output_capacity,
                                                     completion_iterations);
  return cudaGetLastError();
}

inline cudaError_t lesion_modulo_async(const DistributedSequenceMotorView& view,
                                       std::uint32_t modulo,
                                       std::uint32_t phase,
                                       cudaStream_t stream = nullptr) {
  if (!valid_view(view) || modulo == 0u || phase >= modulo)
    return cudaErrorInvalidValue;
  mark_lesion_kernel<<<(view.population + 255u) / 256u, 256u, 0u, stream>>>(
      view.enabled, view.population, modulo, phase);
  reclaim_lesioned_mass_kernel<<<4096u, 256u, 0u, stream>>>(view);
  return cudaGetLastError();
}

inline cudaError_t audit_mass_async(const DistributedSequenceMotorView& view,
                                    unsigned long long* device_result,
                                    cudaStream_t stream = nullptr) {
  if (!valid_view(view) || device_result == nullptr)
    return cudaErrorInvalidValue;
  cudaError_t status =
      cudaMemsetAsync(device_result, 0, sizeof(unsigned long long), stream);
  if (status != cudaSuccess)
    return status;
  audit_represented_mass_kernel<<<4096u, 256u, 0u, stream>>>(view, device_result);
  return cudaGetLastError();
}

// Integration-facing API. All cue bytes, learned state, completion, selection,
// and emitted bytes remain device-resident. The returned mass value is an
// observer-only synchronous audit and never enters a causal kernel.
inline cudaError_t initialize(const DeviceStateView& view,
                              cudaStream_t stream = nullptr) {
  return initialize_async(view, kDefaultRepresentedMass, stream);
}

inline cudaError_t learn_raw(const DeviceStateView& view,
                             const std::uint8_t* device_bytes,
                             std::uint32_t byte_count,
                             std::uint32_t surface,
                             cudaStream_t stream = nullptr) {
  return learn_online_async(view, device_bytes, byte_count, surface, stream);
}

inline cudaError_t condition_raw(const DeviceStateView& view,
                                 const std::uint8_t* device_bytes,
                                 std::uint32_t byte_count,
                                 std::uint32_t surface,
                                 cudaStream_t stream = nullptr) {
  return seed_raw_async(view, device_bytes, byte_count, surface, stream);
}

inline cudaError_t append_sensory(const DeviceStateView& view,
                                  std::uint32_t sensory_size,
                                  cudaStream_t stream = nullptr) {
  return append_sensory_async(view, sensory_size, stream);
}

inline cudaError_t sense_raw(const DeviceStateView& view,
                             const std::uint8_t* device_bytes,
                             std::uint32_t byte_count,
                             std::uint32_t surface,
                             cudaStream_t stream = nullptr) {
  return sense_raw_async(view, device_bytes, byte_count, surface, stream);
}

inline cudaError_t clear_cue(const DeviceStateView& view,
                             cudaStream_t stream = nullptr) {
  return clear_cue_async(view, stream);
}

inline cudaError_t generate(const DeviceStateView& view,
                            std::uint8_t* device_output,
                            std::uint32_t output_capacity,
                            std::uint32_t* device_output_count,
                            cudaStream_t stream = nullptr) {
  if (device_output_count == nullptr)
    return cudaErrorInvalidValue;
  cudaError_t status = generate_async(view, device_output, output_capacity, 2u, stream);
  if (status != cudaSuccess)
    return status;
  if (device_output_count == view.emitted_count)
    return cudaSuccess;
  return cudaMemcpyAsync(device_output_count, view.emitted_count, sizeof(std::uint32_t),
                         cudaMemcpyDeviceToDevice, stream);
}

inline cudaError_t encode_opaque_units_with_count(
    const DeviceStateView& view, const std::uint32_t* unit_lengths,
    const std::uint32_t* packed_unit_bytes, std::uint32_t unit_words,
    std::uint32_t unit_begin, std::uint32_t unit_count,
    std::uint32_t* unit_population, std::uint32_t* unit_population_count,
    cudaStream_t stream = nullptr) {
  if (!valid_view(view) || unit_lengths == nullptr || packed_unit_bytes == nullptr ||
      unit_words == 0u || unit_count == 0u || unit_population == nullptr)
    return cudaErrorInvalidValue;
  encode_opaque_unit_population_kernel<<<(unit_count + 255u) / 256u, 256u, 0u, stream>>>(
      view, unit_lengths, packed_unit_bytes, unit_words, unit_begin, unit_count,
      unit_population, unit_population_count);
  return cudaPeekAtLastError();
}

inline cudaError_t encode_opaque_units(
    const DeviceStateView& view, const std::uint32_t* unit_lengths,
    const std::uint32_t* packed_unit_bytes, std::uint32_t unit_words,
    std::uint32_t unit_begin, std::uint32_t unit_count,
    std::uint32_t* unit_population, cudaStream_t stream = nullptr) {
  return encode_opaque_units_with_count(
      view, unit_lengths, packed_unit_bytes, unit_words, unit_begin, unit_count,
      unit_population, nullptr, stream);
}

inline cudaError_t project_opaque_unit_plan(
    const DeviceStateView& view, const std::uint32_t* unit_lengths,
    const std::uint32_t* unit_population,
    const std::uint32_t* unit_vitality, std::uint32_t unit_begin,
    std::uint32_t unit_count,
    unsigned long long* unit_activity, std::uint32_t* unit_phase,
    std::uint32_t* projection_state,
    std::uint32_t* motor_context, std::uint32_t* motor_completion,
    std::uint32_t motor_capacity, cudaStream_t stream = nullptr) {
  if (!valid_view(view) || unit_lengths == nullptr || unit_population == nullptr ||
      unit_vitality == nullptr || unit_count == 0u ||
      unit_activity == nullptr || unit_phase == nullptr || motor_context == nullptr ||
      projection_state == nullptr || motor_completion == nullptr || motor_capacity == 0u)
    return cudaErrorInvalidValue;
  cudaError_t status = cudaMemsetAsync(view.completion_scores, 0,
                                      view.population * sizeof(unsigned long long), stream);
  if (status != cudaSuccess)
    return status;
  gate_population_plan_projection_kernel<<<1u, 1u, 0u, stream>>>(view,
                                                                  projection_state);
  build_population_plan_activity_kernel<<<1u, 256u, 0u, stream>>>(view,
                                                                   projection_state);
  status = cudaPeekAtLastError();
  if (status != cudaSuccess)
    return status;
  score_opaque_units_from_population_kernel<<<(unit_count + 255u) / 256u, 256u, 0u, stream>>>(
      view, unit_lengths, unit_population, unit_vitality, unit_begin, unit_count,
      projection_state, unit_activity, unit_phase);
  status = cudaPeekAtLastError();
  if (status != cudaSuccess)
    return status;
  select_opaque_unit_plan_kernel<<<1u, 256u, 0u, stream>>>(
      unit_lengths, unit_activity, unit_phase, unit_count, unit_begin, projection_state,
      motor_context,
      motor_completion, motor_capacity);
  return cudaPeekAtLastError();
}

inline unsigned long long represented_mass(const DeviceStateView& view) {
  if (!valid_view(view))
    return 0ull;
  if (audit_mass_async(view, view.mass_audit) != cudaSuccess)
    return 0ull;
  unsigned long long host_mass = 0ull;
  if (cudaMemcpy(&host_mass, view.mass_audit, sizeof(host_mass), cudaMemcpyDeviceToHost) !=
      cudaSuccess)
    return 0ull;
  return host_mass;
}

}  // namespace bcc32_cuda_distributed_sequence_motor
