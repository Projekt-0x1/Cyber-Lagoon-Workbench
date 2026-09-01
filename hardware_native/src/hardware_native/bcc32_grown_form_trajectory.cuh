#pragma once

#include <cuda_runtime.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <vector>

#include "hardware_native/bcc32_grown_situation_tissue.cuh"
#include "hardware_native/bcc32_grown_form_credit_factor.cuh"
#include "hardware_native/bcc32_grown_sensorimotor_factor.cuh"
#include "hardware_native/bcc32_raw_byte_tape.cuh"

namespace substrate::bcc32::grown_form_trajectory {

using substrate::bcc32::InterventionReason;
using substrate::bcc32::ResidentStageReason;

namespace adult = developmental_adult;
namespace situation = grown_situation_tissue;
namespace credit = grown_form_credit_factor;
namespace sensorimotor = grown_sensorimotor_factor;
using adult::GrownAdult;
using adult::StateEntry;

inline constexpr std::uint32_t kMaxInputPhases = credit::kMaxInputPhases;
inline constexpr std::uint32_t kMaxOutputPhases = credit::kMaxOutputPhases;
inline constexpr std::uint32_t kMaxPhases = kMaxOutputPhases;
inline constexpr std::uint32_t kMapFields =
    kMaxOutputPhases * kMaxInputPhases;
inline constexpr std::uint32_t kCandidateBase = credit::kCandidateBase;
inline constexpr std::uint32_t kPrototypeBase = credit::kPrototypeBase;
inline constexpr std::uint32_t kAgreementBase =
    kPrototypeBase + kMaxOutputPhases;
inline constexpr std::uint32_t kSeenBase =
    kAgreementBase + kMaxOutputPhases;
inline constexpr std::uint32_t kLengthField =
    kSeenBase + kMaxOutputPhases;
inline constexpr std::uint32_t kDispositionField = kLengthField + 1u;
inline constexpr std::uint32_t kFieldsPerRegion = credit::kFormFieldsPerRegion;
inline constexpr std::uint32_t kRailCount = situation::kRegionCount * kFieldsPerRegion * 2u;
inline constexpr std::uint32_t kDirectionalClauseRegionBase = 56u;
inline constexpr std::uint32_t kDirectionalClauseRegionCount = 8u;
inline constexpr std::uint32_t kDirectionalCaptureMask = 3u;
inline constexpr std::uint32_t kFormProbeCount = 3u;
inline constexpr std::uint32_t kLearnThreads = situation::kRegionCount;
using DeviceLayout = credit::FormLayout;

struct ActiveContextLayout {
  std::uint64_t rails[2]{};
};

struct GroundedContextLayout {
  std::uint64_t rails[4]{};
};

enum class FormContextSource : std::uint32_t {
  Situation,
  GroundedSensorimotorRelation,
};

struct LearnReceipt {
  std::uint32_t valid = 0u;
  std::uint32_t region = 0xffffffffu;
  std::uint32_t input_count = 0u;
  std::uint32_t output_count = 0u;
  std::uint32_t matter_before_bits = 0u;
  std::uint32_t matter_after_bits = 0u;
  std::uint64_t active_mask = 0u;
  std::uint32_t parallel_regions_eligible = 0u;
  std::uint32_t parallel_regions_completed = 0u;
  std::uint64_t parallel_region_mask = 0u;
  std::uint64_t parallel_launch_epochs = 0u;
};

struct GenerateReceipt {
  std::uint32_t valid = 0u;
  std::uint32_t region = 0xffffffffu;
  std::uint32_t output_count = 0u;
  std::uint32_t mapped_phases = 0u;
  std::uint32_t literal_phases = 0u;
  std::uint64_t active_mask = 0u;
  std::uint64_t contributing_mask = 0u;
  std::uint8_t output[kMaxOutputPhases]{};
};

struct LesionReceipt {
  std::uint32_t first_region = 0xffffffffu;
  std::uint32_t second_region = 0xffffffffu;
  std::uint32_t moved_words = 0u;
  std::uint32_t matter_before_bits = 0u;
  std::uint32_t matter_after_bits = 0u;
};

inline void require_cuda(cudaError_t status, const char* operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
  }
}

__host__ __device__ inline std::uint32_t rail_index(std::uint32_t region, std::uint32_t field,
                                                    std::uint32_t polarity) {
  return (region * kFieldsPerRegion + field) * 2u + polarity;
}

__host__ __device__ inline std::uint32_t map_field(std::uint32_t output_phase,
                                                   std::uint32_t input_phase) {
  return output_phase * kMaxInputPhases + input_phase;
}

inline DeviceLayout make_layout(const GrownAdult& grown) {
  return credit::make_form_layout(grown);
}

inline GroundedContextLayout make_grounded_context_layout(
    const GrownAdult& grown) {
  GroundedContextLayout layout{};
  for (std::uint32_t index = 0u; index < 4u; ++index) {
    layout.rails[index] = grown.physical_slot(
        {36, 36, -2 + static_cast<std::int32_t>(index)});
  }
  return layout;
}

inline ActiveContextLayout make_situation_active_layout(
    const situation::DeviceLayout& layout) {
  ActiveContextLayout active{};
  for (std::uint32_t word = 0u; word < 2u; ++word) {
    active.rails[word] =
        layout.rails[situation::active_rail_index(word, 0u)];
  }
  return active;
}

inline ActiveContextLayout make_grounded_active_layout(
    const GroundedContextLayout& layout) {
  return {{layout.rails[0], layout.rails[2]}};
}

inline std::vector<StateEntry> founder_entries(const GrownAdult& grown) {
  const DeviceLayout layout = make_layout(grown);
  std::vector<StateEntry> entries;
  entries.reserve(kRailCount + 2u);
  for (std::uint32_t region = 0u; region < situation::kRegionCount; ++region) {
    for (std::uint32_t field = 0u; field < kFieldsPerRegion; ++field) {
      SiteWord first = 0u;
      SiteWord second = 0xffffffffu;
      if (field >= kPrototypeBase &&
          field < kPrototypeBase + kMaxOutputPhases) {
        RawByteRails prototype = with_raw_byte_carriers(RawByteRails{}, 0u);
        prototype.zero |= face_bit(0u);
        prototype.one |= face_bit(1u);
        first = prototype.zero;
        second = prototype.one;
      }
      entries.push_back({layout.rails[rail_index(region, field, 0u)], first});
      entries.push_back({layout.rails[rail_index(region, field, 1u)], second});
    }
  }
  RawByteRails motor = with_raw_byte_carriers(RawByteRails{}, 0u);
  motor.zero |= face_bit(2u);
  motor.one |= face_bit(3u);
  entries.push_back({grown.boundary_port_slot(adult::kRawMotorZeroPort), motor.zero});
  entries.push_back({grown.boundary_port_slot(adult::kRawMotorOnePort), motor.one});
  const std::vector<StateEntry> credit_entries =
      credit::founder_entries<StateEntry>(grown);
  entries.insert(entries.end(), credit_entries.begin(), credit_entries.end());
  const GroundedContextLayout grounded = make_grounded_context_layout(grown);
  entries.push_back({grounded.rails[0], 0u});
  entries.push_back({grounded.rails[1], 0xffffffffu});
  entries.push_back({grounded.rails[2], 0u});
  entries.push_back({grounded.rails[3], 0xffffffffu});
  return entries;
}

__device__ inline SiteWord read_value(const SiteWord* words, const DeviceLayout& layout,
                                      std::uint32_t region, std::uint32_t field) {
  return words[layout.rails[rail_index(region, field, 0u)]];
}

__device__ inline void write_value(SiteWord* words, const DeviceLayout& layout,
                                   std::uint32_t region, std::uint32_t field, SiteWord value) {
  words[layout.rails[rail_index(region, field, 0u)]] = value;
  words[layout.rails[rail_index(region, field, 1u)]] = ~value;
}

__device__ inline std::uint32_t support_count(SiteWord support) {
  return __popc(support);
}

__device__ inline SiteWord increment_support(SiteWord support) {
  if (support == 0xffffffffu)
    return support;
  const SiteWord available = ~support;
  return support | (available & (0u - available));
}

__device__ inline std::uint32_t matter_bits(const SiteWord* words, const DeviceLayout& layout) {
  std::uint32_t total = 0u;
  for (std::uint32_t index = 0u; index < kRailCount; ++index)
    total += __popc(words[layout.rails[index]]);
  return total;
}

__device__ inline std::uint64_t active_regions(
    const SiteWord* words, const ActiveContextLayout& layout) {
  return
      static_cast<std::uint64_t>(words[layout.rails[0]]) |
      (static_cast<std::uint64_t>(words[layout.rails[1]]) << 32u);
}

__device__ inline std::uint32_t mix32(std::uint32_t value) {
  value ^= value << 13u;
  value ^= value >> 17u;
  return value ^ (value << 5u);
}

// Appearance addresses are a bounded physical partition of the form tissue.
// They select no semantic role and are only used to keep independently learned
// dispositions apart when an instance basin is reused by later contact.
__device__ inline std::uint64_t appearance_coalition_mask(
    SiteWord appearance) {
  std::uint64_t mask = 0u;
  for (std::uint32_t lane = 0u; lane < 3u; ++lane) {
    const SiteWord seed = 0x9e3779b9u + lane * 0x7f4a7c15u;
    const std::uint32_t region =
        lane * 16u + (mix32(appearance ^ seed) & 15u);
    mask |= 1ull << region;
  }
  return mask;
}

__device__ inline std::uint64_t directional_clause_coalition_mask() {
  return ((1ull << kDirectionalClauseRegionCount) - 1ull)
         << kDirectionalClauseRegionBase;
}

__device__ inline std::uint32_t coalition_phase_mask(
    std::uint32_t active_count, std::uint32_t active_rank,
    bool allow_single_region_complete) {
  if (active_count <= 1u)
    return allow_single_region_complete
               ? (1u << kMaxOutputPhases) - 1u
               : 0u;
  std::uint32_t mask = 0u;
  for (std::uint32_t phase = 0u; phase < kMaxOutputPhases; ++phase) {
    const std::uint32_t first = phase % active_count;
    const std::uint32_t second = (first + 1u) % active_count;
    if (active_rank == first || active_rank == second)
      mask |= 1u << phase;
  }
  return mask;
}

__device__ inline std::uint32_t acquire_phase_disposition(
    SiteWord* words, DeviceLayout layout, std::uint32_t region,
    std::uint32_t active_count, std::uint32_t active_rank,
    bool allow_single_region_complete) {
  const SiteWord resident =
      read_value(words, layout, region, kDispositionField) &
      ((1u << kMaxOutputPhases) - 1u);
  if (resident != 0u)
    return resident;
  const std::uint32_t acquired = coalition_phase_mask(
      active_count, active_rank, allow_single_region_complete);
  if (acquired != 0u)
    write_value(words, layout, region, kDispositionField, acquired);
  return acquired;
}

static_assert(sensorimotor::kRelationSlotCount <= situation::kRegionCount);

static __global__ void bind_grounded_relation_kernel(
    SiteWord* words, GroundedContextLayout form_context,
    const sensorimotor::DeviceLayout* sensorimotor_layout) {
  if (blockIdx.x != 0u || threadIdx.x != 0u)
    return;
  words[form_context.rails[0]] = 0u;
  words[form_context.rails[1]] = 0xffffffffu;
  words[form_context.rails[2]] = 0u;
  words[form_context.rails[3]] = 0xffffffffu;
  if (sensorimotor_layout == nullptr)
    return;

  const sensorimotor::DeviceLayout& layout = *sensorimotor_layout;
  const std::uint32_t slot = sensorimotor::read_value(
      words, layout, sensorimotor::kActiveRelationSlot);
  if (slot >= sensorimotor::kRelationSlotCount)
    return;
  const SiteWord context = sensorimotor::read_value(
      words, layout, sensorimotor::kActiveContext);
  const SiteWord cue = sensorimotor::read_value(
      words, layout, sensorimotor::kActiveCue);
  const SiteWord motor = sensorimotor::read_value(
      words, layout, sensorimotor::kMotor);
  const std::uint32_t action = sensorimotor::read_value(
      words, layout,
      sensorimotor::relation_rail(slot, sensorimotor::kRelationAction)) & 1u;
  const bool qualified =
      sensorimotor::read_value(
          words, layout,
          sensorimotor::relation_rail(slot, sensorimotor::kRelationOccupied)) != 0u &&
      sensorimotor::read_value(
          words, layout,
          sensorimotor::relation_rail(slot, sensorimotor::kRelationContext)) == context &&
      sensorimotor::read_value(
          words, layout,
          sensorimotor::relation_rail(slot, sensorimotor::kRelationCue)) == cue &&
      motor == (1u << action) &&
      sensorimotor::read_value(
          words, layout,
          sensorimotor::relation_rail(slot, sensorimotor::kRelationPositive)) != 0u &&
      sensorimotor::read_value(
          words, layout,
          sensorimotor::relation_rail(slot, sensorimotor::kRelationNegative)) == 0u;
  if (!qualified)
    return;

  // The learned relation is a selective broker, not sole form authority. Project
  // independent resident evidence into separate developmental territories so
  // later local credit can grow an overlapping reconstructive coalition.
  const std::uint32_t context_region = 16u + (mix32(context) & 15u);
  const std::uint32_t cue_region = 32u + (mix32(cue) & 15u);
  const std::uint32_t consequence_region =
      48u + (mix32(context ^ (cue << 1u) ^ motor) & 15u);
  const std::uint64_t active = (1ull << slot) |
                               (1ull << context_region) |
                               (1ull << cue_region) |
                               (1ull << consequence_region);
  words[form_context.rails[0]] = static_cast<SiteWord>(active);
  words[form_context.rails[1]] = ~static_cast<SiteWord>(active);
  words[form_context.rails[2]] = static_cast<SiteWord>(active >> 32u);
  words[form_context.rails[3]] = ~static_cast<SiteWord>(active >> 32u);
}

static __global__ void bind_unique_active_instance_kernel(
    SiteWord* words, GroundedContextLayout form_context,
    const sensorimotor::DeviceLayout* sensorimotor_layout) {
  if (blockIdx.x != 0u || threadIdx.x != 0u) return;
  words[form_context.rails[0]] = 0u;
  words[form_context.rails[1]] = 0xffffffffu;
  words[form_context.rails[2]] = 0u;
  words[form_context.rails[3]] = 0xffffffffu;
  if (sensorimotor_layout == nullptr) return;

  const auto& context = sensorimotor_layout->context;
  std::uint32_t changed_count = 0u;
  std::uint32_t changed_basin = sensorimotor::instance::kBasinCount;
  for (std::uint32_t basin = 0u;
       basin < sensorimotor::instance::kBasinCount; ++basin) {
    if (sensorimotor::instance::read_field(
            words, context,
            sensorimotor::instance::basin_field(
                basin, sensorimotor::instance::kActive)) == 0u ||
        sensorimotor::instance::read_field(
            words, context,
            sensorimotor::instance::basin_field(
                basin, sensorimotor::instance::kMissingAge)) != 0u ||
        sensorimotor::instance::read_field(
            words, context,
            sensorimotor::instance::basin_field(
                basin, sensorimotor::instance::kEligibility)) == 0u)
      continue;
    const SiteWord previous = sensorimotor::instance::read_field(
        words, context,
        sensorimotor::instance::basin_field(
            basin, sensorimotor::instance::kPreviousSite));
    const SiteWord current = sensorimotor::instance::read_field(
        words, context,
        sensorimotor::instance::basin_field(
            basin, sensorimotor::instance::kCurrentSite));
    if (previous != current) {
      ++changed_count;
      changed_basin = basin;
    }
  }
  if (changed_count != 1u || changed_basin >= sensorimotor::instance::kBasinCount)
    return;
  const SiteWord appearance = sensorimotor::instance::read_field(
      words, context,
      sensorimotor::instance::basin_field(
          changed_basin, sensorimotor::instance::kAppearance));
  const std::uint64_t active = appearance_coalition_mask(appearance);
  words[form_context.rails[0]] = static_cast<SiteWord>(active);
  words[form_context.rails[1]] = ~static_cast<SiteWord>(active);
  words[form_context.rails[2]] = static_cast<SiteWord>(active >> 32u);
  words[form_context.rails[3]] = ~static_cast<SiteWord>(active >> 32u);
}

static __global__ void bind_directional_clause_kernel(
    SiteWord* words, GroundedContextLayout form_context,
    const sensorimotor::DeviceLayout* sensorimotor_layout) {
  if (blockIdx.x != 0u || threadIdx.x != 0u) return;
  words[form_context.rails[0]] = 0u;
  words[form_context.rails[1]] = 0xffffffffu;
  words[form_context.rails[2]] = 0u;
  words[form_context.rails[3]] = 0xffffffffu;
  if (sensorimotor_layout == nullptr) return;

  const auto& layout = *sensorimotor_layout;
  const std::uint32_t slot = sensorimotor::read_value(
      words, layout, sensorimotor::kActiveRelationSlot);
  if (slot >= sensorimotor::kRelationSlotCount) return;
  const SiteWord context = sensorimotor::read_value(
      words, layout, sensorimotor::kActiveContext);
  const SiteWord cue = sensorimotor::read_value(
      words, layout, sensorimotor::kActiveCue);
  const SiteWord motor = sensorimotor::read_value(words, layout,
                                                  sensorimotor::kMotor);
  const auto direction = sensorimotor::derive_directional_evidence(
      words, layout);
  const auto relation = [&](sensorimotor::RelationField field) {
    return sensorimotor::read_value(
        words, layout, sensorimotor::relation_rail(slot, field));
  };
  const std::uint32_t action = relation(sensorimotor::kRelationAction) & 1u;
  const bool qualified =
      relation(sensorimotor::kRelationOccupied) != 0u &&
      relation(sensorimotor::kRelationContext) == context &&
      relation(sensorimotor::kRelationCue) == cue &&
      motor == (1u << action) && relation(sensorimotor::kRelationPositive) != 0u &&
      relation(sensorimotor::kRelationNegative) == 0u &&
      direction.changed == relation(sensorimotor::kRelationChanged) &&
      direction.unchanged == relation(sensorimotor::kRelationUnchanged);
  if (!qualified) return;
  const std::uint64_t active = directional_clause_coalition_mask();
  words[form_context.rails[0]] = static_cast<SiteWord>(active);
  words[form_context.rails[1]] = ~static_cast<SiteWord>(active);
  words[form_context.rails[2]] = static_cast<SiteWord>(active >> 32u);
  words[form_context.rails[3]] = ~static_cast<SiteWord>(active >> 32u);
}

static __global__ void learn_kernel(SiteWord* words,
                                    const DeviceLayout* layout_device,
                                    ActiveContextLayout active_context,
                                    const std::uint8_t* input, std::uint32_t input_count,
                                    const std::uint8_t* output, std::uint32_t output_count,
                                    bool allow_single_region_complete,
                                    LearnReceipt* receipt) {
  __shared__ std::uint64_t active_mask_shared;
  __shared__ std::uint32_t active_count_shared;
  __shared__ std::uint32_t input_count_shared;
  __shared__ std::uint32_t output_count_shared;
  __shared__ std::uint32_t valid_shared;
  __shared__ std::uint32_t completed_shared;
  __shared__ std::uint64_t completed_mask_shared;

  if (threadIdx.x == 0u) {
    active_mask_shared = 0u;
    active_count_shared = 0u;
    input_count_shared = input_count;
    output_count_shared = output_count;
    valid_shared = 0u;
    completed_shared = 0u;
    completed_mask_shared = 0u;
    if (receipt != nullptr)
      *receipt = LearnReceipt{};
    if (words != nullptr && layout_device != nullptr) {
      const DeviceLayout& layout = *layout_device;
      active_mask_shared = active_regions(words, active_context);
      active_count_shared = __popcll(active_mask_shared);
      valid_shared = active_mask_shared != 0u && input != nullptr &&
                     input_count != 0u && input_count <= kMaxInputPhases &&
                     output != nullptr && output_count != 0u &&
                     output_count <= kMaxOutputPhases;
      if (receipt != nullptr) {
        receipt->matter_before_bits = matter_bits(words, layout);
        receipt->active_mask = active_mask_shared;
        receipt->region = active_mask_shared == 0u
                              ? 0xffffffffu
                              : static_cast<std::uint32_t>(
                                    __ffsll(static_cast<long long>(active_mask_shared)) - 1);
        receipt->input_count = input_count;
        receipt->output_count = output_count;
        receipt->parallel_regions_eligible = active_count_shared;
      }
    }
  }
  __syncthreads();

  const std::uint32_t region =
      blockIdx.x * blockDim.x + threadIdx.x;
  if (blockIdx.x == 0u && valid_shared &&
      region < situation::kRegionCount &&
      (active_mask_shared & (1ull << region)) != 0u) {
    const DeviceLayout& layout = *layout_device;
    const std::uint32_t active_rank = static_cast<std::uint32_t>(
        __popcll(active_mask_shared & ((1ull << region) - 1ull)));
    const std::uint32_t phase_mask = acquire_phase_disposition(
        words, layout, region, active_count_shared, active_rank,
        allow_single_region_complete);
    if (phase_mask != 0u) {
      for (std::uint32_t output_phase = 0u;
           output_phase < output_count_shared; ++output_phase) {
        if ((phase_mask & (1u << output_phase)) == 0u)
          continue;
        const std::uint32_t seen_field = kSeenBase + output_phase;
        const SiteWord seen = read_value(words, layout, region, seen_field);
        SiteWord matching = 0u;
        for (std::uint32_t input_phase = 0u;
             input_phase < input_count_shared; ++input_phase) {
          if (output[output_phase] == input[input_phase])
            matching |= 1u << input_phase;
        }
        const SiteWord prior_candidate =
            read_value(words, layout, region, kCandidateBase + output_phase);
        const SiteWord candidate =
            seen == 0u ? matching : prior_candidate & matching;
        write_value(words, layout, region, kCandidateBase + output_phase,
                    candidate);
        const std::uint32_t prototype_field = kPrototypeBase + output_phase;
        const std::uint32_t agreement_field = kAgreementBase + output_phase;
        if (seen == 0u) {
          RawByteRails prototype{
              words[layout.rails[rail_index(region, prototype_field, 0u)]],
              words[layout.rails[rail_index(region, prototype_field, 1u)]]};
          prototype = with_raw_byte_carriers(prototype, output[output_phase]);
          words[layout.rails[rail_index(region, prototype_field, 0u)]] =
              prototype.zero;
          words[layout.rails[rail_index(region, prototype_field, 1u)]] =
              prototype.one;
          write_value(words, layout, region, agreement_field, 0xffu);
        } else {
          const RawByteRails prototype{
              words[layout.rails[rail_index(region, prototype_field, 0u)]],
              words[layout.rails[rail_index(region, prototype_field, 1u)]]};
          const RawByteDecode decoded = decode_raw_byte_carriers(prototype);
          const SiteWord agreement =
              read_value(words, layout, region, agreement_field);
          write_value(
              words, layout, region, agreement_field,
              agreement & static_cast<SiteWord>(
                               ~(decoded.value ^ output[output_phase]) & 0xffu));
        }
        write_value(words, layout, region, seen_field, increment_support(seen));
      }
    }
    const SiteWord length_bit = 1u << (output_count_shared - 1u);
    const SiteWord lengths = read_value(words, layout, region, kLengthField);
    write_value(words, layout, region, kLengthField,
                lengths == 0u ? length_bit : lengths & length_bit);
    atomicAdd(&completed_shared, 1u);
    atomicOr(reinterpret_cast<unsigned long long*>(&completed_mask_shared),
             static_cast<unsigned long long>(1ull << region));
  }
  __syncthreads();

  if (threadIdx.x == 0u && receipt != nullptr &&
      words != nullptr && layout_device != nullptr) {
    receipt->valid = valid_shared ? 1u : 0u;
    receipt->matter_after_bits = matter_bits(words, *layout_device);
    receipt->parallel_regions_completed = completed_shared;
    receipt->parallel_region_mask = completed_mask_shared;
    receipt->parallel_launch_epochs = valid_shared ? 1u : 0u;
  }
}

__device__ inline bool generate_region_phase(
    const SiteWord* words, DeviceLayout layout, std::uint32_t region,
    std::uint32_t output_phase, const std::uint8_t* input,
    std::uint32_t input_count, std::uint8_t* output, bool* mapped) {
  const std::uint32_t seen =
      support_count(read_value(words, layout, region, kSeenBase + output_phase));
  if (seen == 0u)
    return false;
  const SiteWord candidate = read_value(
      words, layout, region, kCandidateBase + output_phase);
  const SiteWord bounded = input_count >= 32u
                               ? candidate
                               : candidate & (input_count == 0u
                                                  ? 0u
                                                  : (1u << input_count) - 1u);
  if (support_count(bounded) == 1u) {
    const std::uint32_t input_phase =
        static_cast<std::uint32_t>(__ffs(static_cast<int>(bounded)) - 1);
    *output = input[input_phase];
    *mapped = true;
    return true;
  }

  const SiteWord agreement =
      read_value(words, layout, region, kAgreementBase + output_phase) & 0xffu;
  const RawByteRails prototype{
      words[layout.rails[rail_index(region, kPrototypeBase + output_phase, 0u)]],
      words[layout.rails[rail_index(region, kPrototypeBase + output_phase, 1u)]]};
  const RawByteDecode decoded = decode_raw_byte_carriers(prototype);
  if (agreement != 0xffu || !decoded.valid)
    return false;
  *output = decoded.value;
  *mapped = false;
  return true;
}

static __global__ void generate_kernel(const SiteWord* words,
                                       const DeviceLayout* layout_device,
                                       ActiveContextLayout active_context,
                                       const std::uint8_t* input, std::uint32_t input_count,
                                       const std::uint32_t* capture_valid,
                                       std::uint32_t required_capture_mask,
                                       GenerateReceipt* receipt) {
  if (blockIdx.x != 0u || threadIdx.x != 0u)
    return;
  const DeviceLayout& layout = *layout_device;
  GenerateReceipt local{};
  local.active_mask = active_regions(words, active_context);
  if (capture_valid != nullptr &&
      ((*capture_valid & required_capture_mask) != required_capture_mask)) {
    if (receipt != nullptr)
      *receipt = local;
    return;
  }
  if (local.active_mask == 0u || input_count == 0u ||
      input_count > kMaxInputPhases) {
    if (receipt != nullptr)
      *receipt = local;
    return;
  }

  for (std::uint32_t region = 0u; region < situation::kRegionCount; ++region) {
    if ((local.active_mask & (1ull << region)) == 0u)
      continue;
    const SiteWord lengths = read_value(words, layout, region, kLengthField);
    if (__popc(lengths) != 1u)
      continue;
    const std::uint32_t count =
        static_cast<std::uint32_t>(__ffs(static_cast<int>(lengths)));
    if (count == 0u || count > kMaxOutputPhases) {
      local.output_count = 0u;
      break;
    }
    if (local.output_count == 0u) {
      local.output_count = count;
      continue;
    }
    if (count != local.output_count) {
      local.output_count = 0u;
      break;
    }
  }
  if (local.output_count == 0u) {
    if (receipt != nullptr)
      *receipt = local;
    return;
  }

  for (std::uint32_t output_phase = 0u; output_phase < local.output_count;
       ++output_phase) {
    bool phase_valid = false;
    bool phase_mapped = false;
    for (std::uint32_t region = 0u; region < situation::kRegionCount; ++region) {
      if ((local.active_mask & (1ull << region)) == 0u)
        continue;
      std::uint8_t candidate = 0u;
      bool mapped = false;
      if (!generate_region_phase(words, layout, region, output_phase, input,
                                 input_count, &candidate, &mapped))
        continue;
      if (phase_valid && candidate != local.output[output_phase]) {
        local = GenerateReceipt{};
        local.active_mask = active_regions(words, active_context);
        if (receipt != nullptr)
          *receipt = local;
        return;
      }
      if (!phase_valid) {
        local.output[output_phase] = candidate;
        if (local.region == 0xffffffffu)
          local.region = region;
      }
      phase_valid = true;
      phase_mapped |= mapped;
      local.contributing_mask |= 1ull << region;
    }
    if (!phase_valid) {
      local = GenerateReceipt{};
      local.active_mask = active_regions(words, active_context);
      if (receipt != nullptr)
        *receipt = local;
      return;
    }
    if (phase_mapped)
      ++local.mapped_phases;
    else
      ++local.literal_phases;
  }
  local.valid = 1u;
  if (receipt != nullptr)
    *receipt = local;
}

static __global__ void reset_capture_kernel(std::uint8_t* capture,
                                            std::uint32_t* valid_mask) {
  if (blockIdx.x != 0u || threadIdx.x != 0u)
    return;
  for (std::uint32_t phase = 0u; phase < kMaxInputPhases; ++phase)
    capture[phase] = 0u;
  *valid_mask = 0u;
}

static __global__ void capture_generated_kernel(
    const GenerateReceipt* generated, std::uint8_t* capture,
    std::uint32_t* valid_mask, std::uint32_t offset,
    std::uint32_t expected_count, std::uint32_t slot_bit) {
  if (blockIdx.x != 0u || threadIdx.x != 0u)
    return;
  if (generated == nullptr || generated->valid == 0u ||
      generated->output_count != expected_count ||
      offset + expected_count > kMaxInputPhases) {
    *valid_mask &= ~slot_bit;
    return;
  }
  for (std::uint32_t phase = 0u; phase < expected_count; ++phase)
    capture[offset + phase] = generated->output[phase];
  *valid_mask |= slot_bit;
}

static __global__ void stage_captured_input_kernel(
    const std::uint8_t* capture, const std::uint32_t* valid_mask,
    std::uint32_t required_capture_mask, std::uint32_t input_count,
    std::uint8_t* credit_input, credit::DeviceInputs* credit_inputs) {
  if (blockIdx.x != 0u || threadIdx.x != 0u)
    return;
  credit_inputs->input_staged = 0u;
  if (input_count == 0u || input_count > kMaxInputPhases ||
      ((*valid_mask & required_capture_mask) != required_capture_mask))
    return;
  for (std::uint32_t phase = 0u; phase < input_count; ++phase)
    credit_input[phase] = capture[phase];
  credit_inputs->input_count = input_count;
  credit_inputs->input_staged = 1u;
}

static __global__ void stage_motor_kernel(SiteWord* words, std::uint64_t motor_zero_slot,
                                          std::uint64_t motor_one_slot,
                                          const GenerateReceipt* generated, std::uint32_t phase) {
  if (blockIdx.x != 0u || threadIdx.x != 0u)
    return;
  RawByteRails motor{words[motor_zero_slot], words[motor_one_slot]};
  if (generated != nullptr && generated->valid != 0u && phase < generated->output_count) {
    motor = with_raw_byte_carriers(motor, generated->output[phase]);
  } else {
    // Poll every bounded phase without asking the host whether it exists.
    // Equal four-bit rails preserve eight represented quanta and decode invalid.
    motor.zero = with_carriers(motor.zero, 0x0fu);
    motor.one = with_carriers(motor.one, 0x0fu);
  }
  words[motor_zero_slot] = motor.zero;
  words[motor_one_slot] = motor.one;
}

static __global__ void select_or_copy_control_kernel(
    const GenerateReceipt* generated, const std::uint8_t* input,
    std::uint8_t* selected) {
  if (blockIdx.x != 0u || threadIdx.x != 0u)
    return;
  const bool transformed = generated != nullptr && generated->valid != 0u &&
                           generated->output_count == 1u;
  selected[0] = transformed ? generated->output[0] : input[0];
}

static __global__ void exchange_regions_kernel(
                                               SiteWord* words,
                                               const DeviceLayout* layout_device,
                                               std::uint32_t first_region,
                                               std::uint32_t second_region,
                                               LesionReceipt* receipt) {
  if (blockIdx.x != 0u || threadIdx.x != 0u)
    return;
  const DeviceLayout& layout = *layout_device;
  LesionReceipt local{};
  local.first_region = first_region;
  local.second_region = second_region;
  local.matter_before_bits = matter_bits(words, layout);
  if (first_region < situation::kRegionCount && second_region < situation::kRegionCount &&
      first_region != second_region) {
    for (std::uint32_t field = 0u; field < kFieldsPerRegion; ++field) {
      for (std::uint32_t polarity = 0u; polarity < 2u; ++polarity) {
        const std::uint64_t first = layout.rails[rail_index(first_region, field, polarity)];
        const std::uint64_t second = layout.rails[rail_index(second_region, field, polarity)];
        const SiteWord held = words[first];
        words[first] = words[second];
        words[second] = held;
        ++local.moved_words;
      }
    }
  }
  local.matter_after_bits = matter_bits(words, layout);
  if (receipt != nullptr)
    *receipt = local;
}

static __global__ void exchange_situation_outcome_kernel(SiteWord* words,
                                                         situation::DeviceLayout layout,
                                                         std::uint32_t region,
                                                         situation::LesionReceipt* receipt) {
  if (blockIdx.x != 0u || threadIdx.x != 0u)
    return;
  situation::LesionReceipt local{};
  local.region = region;
  local.state_before_hash = situation::state_hash(words, layout);
  local.matter_before_bits = situation::state_matter_bits(words, layout);
  if (region < situation::kRegionCount) {
    const SiteWord outcome = situation::read_value(words, layout, region, situation::kOutcome);
    const SiteWord escrow = situation::read_value(words, layout, region, situation::kOutcomeEscrow);
    situation::write_value(words, layout, region, situation::kOutcome, escrow);
    situation::write_value(words, layout, region, situation::kOutcomeEscrow, outcome);
    local.moved_bits = 32u;
    local.applied = 1u;
  }
  local.state_after_hash = situation::state_hash(words, layout);
  local.matter_after_bits = situation::state_matter_bits(words, layout);
  if (receipt != nullptr)
    *receipt = local;
}

class FormTrajectory {
 public:
  explicit FormTrajectory(
      GrownAdult& grown,
      FormContextSource context_source = FormContextSource::Situation)
      : grown_(grown),
        layout_(make_layout(grown)),
        situation_layout_(situation::make_layout(grown)),
        grounded_context_layout_(make_grounded_context_layout(grown)),
        context_source_(context_source) {
    active_context_layout_ =
        context_source_ == FormContextSource::Situation
            ? make_situation_active_layout(situation_layout_)
            : make_grounded_active_layout(grounded_context_layout_);
    require_cuda(cudaMalloc(&layout_device_, sizeof(*layout_device_)),
                 "allocate resident form layout");
    require_cuda(cudaMemcpy(layout_device_, &layout_, sizeof(layout_),
                            cudaMemcpyHostToDevice),
                 "upload resident form layout");
    require_cuda(cudaMalloc(&input_, kMaxInputPhases), "allocate form input");
    require_cuda(cudaMalloc(&output_, kMaxOutputPhases), "allocate form output");
    require_cuda(cudaMalloc(&captured_input_, kMaxInputPhases),
                 "allocate captured form input");
    require_cuda(cudaMalloc(&capture_valid_, sizeof(*capture_valid_)),
                 "allocate captured form validity");
    require_cuda(cudaMalloc(&selected_control_, 1u),
                 "allocate selected form control");
    require_cuda(cudaMalloc(&learn_receipt_, sizeof(LearnReceipt)), "allocate form learn receipt");
    require_cuda(cudaMalloc(&generate_receipt_, sizeof(GenerateReceipt)),
                 "allocate form generate receipt");
    require_cuda(cudaMalloc(&lesion_receipt_, sizeof(LesionReceipt)),
                 "allocate form lesion receipt");
    require_cuda(cudaMalloc(&situation_lesion_receipt_, sizeof(situation::LesionReceipt)),
                 "allocate situation outcome lesion receipt");
    require_cuda(cudaMalloc(&credit_input_, kMaxInputPhases),
                 "allocate form credit input");
    require_cuda(cudaMalloc(&credit_output_, kMaxOutputPhases),
                 "allocate form credit output");
    require_cuda(cudaMalloc(&credit_inputs_, sizeof(*credit_inputs_)),
                 "allocate form credit inputs");
    require_cuda(cudaMalloc(&credit_receipt_, sizeof(*credit_receipt_)),
                 "allocate form credit receipt");
    credit_host_inputs_.input = credit_input_;
    credit_host_inputs_.output = credit_output_;
    credit_host_inputs_.allow_single_region_complete =
        context_source_ == FormContextSource::Situation ? 1u : 0u;
    for (std::uint32_t word = 0u; word < 2u; ++word) {
      credit_host_inputs_.active_rails[word] = active_context_layout_.rails[word];
    }
    require_cuda(cudaMemset(credit_receipt_, 0, sizeof(*credit_receipt_)),
                 "initialize form credit receipt");
    reset_capture();
    upload_credit_inputs();
    grown_.configure_form_credit_factor(credit::make_layout(grown_), credit_inputs_,
                                        credit_receipt_);
  }

  ~FormTrajectory() {
    destroy_graphs();
    grown_.detach_form_credit_factor_buffers();
    cudaFree(credit_receipt_);
    cudaFree(credit_inputs_);
    cudaFree(credit_output_);
    cudaFree(credit_input_);
    cudaFree(capture_valid_);
    cudaFree(captured_input_);
    cudaFree(situation_lesion_receipt_);
    cudaFree(lesion_receipt_);
    cudaFree(generate_receipt_);
    cudaFree(learn_receipt_);
    cudaFree(selected_control_);
    cudaFree(output_);
    cudaFree(input_);
    cudaFree(layout_device_);
  }

  FormTrajectory(const FormTrajectory&) = delete;
  FormTrajectory& operator=(const FormTrajectory&) = delete;

  LearnReceipt learn(std::span<const std::uint8_t> input, std::span<const std::uint8_t> output) {
    if (input.empty() || input.size() > kMaxInputPhases || output.empty() ||
        output.size() > kMaxOutputPhases)
      throw std::invalid_argument("form trajectory exceeds bounded phases");
    require_cuda(cudaMemcpy(input_, input.data(), input.size(), cudaMemcpyHostToDevice),
                 "upload form input");
    require_cuda(cudaMemcpy(output_, output.data(), output.size(), cudaMemcpyHostToDevice),
                 "upload form output");
    grown_.resident_stage(
        ResidentStageReason::learn_trajectory, [&](SiteWord* words) {
          const std::uint32_t input_count =
              static_cast<std::uint32_t>(input.size());
          const std::uint32_t output_count =
              static_cast<std::uint32_t>(output.size());
          const bool allow_single_region_complete =
              context_source_ == FormContextSource::Situation;
          void* arguments[] = {
              kernel_argument(words), kernel_argument(layout_device_),
              kernel_argument(active_context_layout_), kernel_argument(input_),
              kernel_argument(input_count), kernel_argument(output_),
              kernel_argument(output_count),
              kernel_argument(allow_single_region_complete),
              kernel_argument(learn_receipt_)};
          launch_graph(learn_graph_, "launch form trajectory graph",
                       reinterpret_cast<void*>(learn_kernel), dim3(1u),
                       dim3(kLearnThreads), arguments);
        });
    synchronize("learn form trajectory");
    return copy(learn_receipt_);
  }

  GenerateReceipt generate(std::span<const std::uint8_t> input) {
    prepare(input);
    return observe_generation();
  }

  void prepare(std::span<const std::uint8_t> input) {
    if (input.empty() || input.size() > kMaxInputPhases)
      throw std::invalid_argument("form probe exceeds bounded phases");
    require_cuda(cudaMemcpy(input_, input.data(), input.size(), cudaMemcpyHostToDevice),
                 "upload form probe");
    const std::uint32_t input_count = static_cast<std::uint32_t>(input.size());
    const std::uint32_t* capture_valid = nullptr;
    launch_generate(input_, input_count, capture_valid, 0u);
    synchronize("generate form trajectory");
  }

  GenerateReceipt observe_generation() const { return copy(generate_receipt_); }

  void reset_capture() {
    void* arguments[] = {kernel_argument(captured_input_),
                          kernel_argument(capture_valid_)};
    launch_graph(reset_capture_graph_, "launch reset capture graph",
                 reinterpret_cast<void*>(reset_capture_kernel), dim3(1u),
                 dim3(1u), arguments);
    synchronize("reset captured form input");
  }

  void capture_generated(std::uint32_t offset, std::uint32_t expected_count,
                         std::uint32_t slot_bit) {
    void* arguments[] = {
        kernel_argument(generate_receipt_), kernel_argument(captured_input_),
        kernel_argument(capture_valid_), kernel_argument(offset),
        kernel_argument(expected_count), kernel_argument(slot_bit)};
    launch_graph(capture_generated_graph_, "launch capture-generated graph",
                 reinterpret_cast<void*>(capture_generated_kernel), dim3(1u),
                 dim3(1u), arguments);
    synchronize("capture generated form input");
  }

  void prepare_captured(std::uint32_t input_count,
                        std::uint32_t required_capture_mask) {
    if (input_count == 0u || input_count > kMaxInputPhases)
      throw std::invalid_argument("captured form probe exceeds bounded phases");
    launch_generate(captured_input_, input_count, capture_valid_,
                    required_capture_mask);
    synchronize("generate from captured form input");
  }

  [[nodiscard]] const std::uint8_t* device_generated_output() const {
    return reinterpret_cast<const std::uint8_t*>(generate_receipt_) +
           offsetof(GenerateReceipt, output);
  }

  void bind_grounded_relation(
      const sensorimotor::DeviceLayout* sensorimotor_layout) {
    if (context_source_ != FormContextSource::GroundedSensorimotorRelation)
      throw std::logic_error("form trajectory does not use grounded relation context");
    grown_.resident_stage(
        ResidentStageReason::bind_grounded_context, [&](SiteWord* words) {
          void* arguments[] = {
              kernel_argument(words), kernel_argument(grounded_context_layout_),
              kernel_argument(sensorimotor_layout)};
          launch_graph(
              bind_grounded_relation_graph_, "launch grounded-context graph",
              reinterpret_cast<void*>(bind_grounded_relation_kernel), dim3(1u),
              dim3(1u), arguments);
        });
    synchronize("bind grounded sensorimotor relation to form context");
  }

  void bind_unique_active_instance(
      const sensorimotor::DeviceLayout* sensorimotor_layout) {
    grown_.resident_stage(
        ResidentStageReason::bind_grounded_context, [&](SiteWord* words) {
          void* arguments[] = {
              kernel_argument(words), kernel_argument(grounded_context_layout_),
              kernel_argument(sensorimotor_layout)};
          launch_graph(bind_unique_active_instance_graph_,
                       "launch active-instance graph",
                       reinterpret_cast<void*>(bind_unique_active_instance_kernel),
                       dim3(1u), dim3(1u), arguments);
        });
    synchronize("bind unique active instance form context");
  }

  void bind_directional_clause(
      const sensorimotor::DeviceLayout* sensorimotor_layout) {
    grown_.resident_stage(
        ResidentStageReason::bind_grounded_context, [&](SiteWord* words) {
          void* arguments[] = {
              kernel_argument(words), kernel_argument(grounded_context_layout_),
              kernel_argument(sensorimotor_layout)};
          launch_graph(bind_directional_clause_graph_,
                       "launch directional-context graph",
                       reinterpret_cast<void*>(bind_directional_clause_kernel),
                       dim3(1u), dim3(1u), arguments);
        });
    synchronize("bind directional clause form context");
  }

  void prepare_selected_control(std::span<const std::uint8_t> input) {
    if (input.size() != 1u)
      throw std::invalid_argument("form control selection requires one byte");
    prepare(input);
    void* arguments[] = {kernel_argument(generate_receipt_),
                         kernel_argument(input_),
                         kernel_argument(selected_control_)};
    launch_graph(select_control_graph_, "launch form-control graph",
                 reinterpret_cast<void*>(select_or_copy_control_kernel),
                 dim3(1u), dim3(1u), arguments);
    synchronize("select resident form control");
  }

  [[nodiscard]] const std::uint8_t* device_selected_control() const {
    return selected_control_;
  }

  void stage_input(std::span<const std::uint8_t> input) {
    if (input.empty() || input.size() > kMaxInputPhases)
      throw std::invalid_argument("form credit input exceeds bounded phases");
    require_cuda(cudaMemcpy(credit_input_, input.data(), input.size(), cudaMemcpyHostToDevice),
                 "stage form credit input");
    credit_host_inputs_.input_count = static_cast<std::uint32_t>(input.size());
    credit_host_inputs_.input_staged = 1u;
    credit_host_inputs_.output_contact_staged = 0u;
    credit_host_inputs_.output_contact_end = 0u;
    upload_credit_inputs();
  }

  void stage_output(std::span<const std::uint8_t> output) {
    if (output.empty() || output.size() > kMaxOutputPhases)
      throw std::invalid_argument("form credit output exceeds bounded phases");
    require_cuda(cudaMemcpy(credit_output_, output.data(), output.size(), cudaMemcpyHostToDevice),
                 "stage form credit output");
    credit_host_inputs_.output_count = static_cast<std::uint32_t>(output.size());
    credit_host_inputs_.output_staged = 1u;
    credit_host_inputs_.output_contact_staged = 0u;
    credit_host_inputs_.output_contact_end = 0u;
    upload_credit_inputs();
  }

  // Stage one later sensory contact. The resident factor supplies phase and
  // route chronology; this API deliberately copies one byte only.
  void stage_output_contact_byte(std::uint8_t byte, bool contact_end) {
    require_cuda(cudaMemcpy(credit_output_, &byte, sizeof(byte),
                            cudaMemcpyHostToDevice),
                 "stage form contact byte");
    credit_host_inputs_.output_count = 1u;
    credit_host_inputs_.output_staged = 0u;
    credit_host_inputs_.output_contact_staged = 1u;
    credit_host_inputs_.output_contact_end = contact_end ? 1u : 0u;
    upload_credit_inputs();
  }

  void stage_captured_input(std::uint32_t input_count,
                            std::uint32_t required_capture_mask) {
    if (input_count == 0u || input_count > kMaxInputPhases)
      throw std::invalid_argument(
          "captured form credit input exceeds bounded phases");
    void* arguments[] = {
        kernel_argument(captured_input_), kernel_argument(capture_valid_),
        kernel_argument(required_capture_mask), kernel_argument(input_count),
        kernel_argument(credit_input_), kernel_argument(credit_inputs_)};
    launch_graph(stage_captured_input_graph_, "launch captured-input graph",
                 reinterpret_cast<void*>(stage_captured_input_kernel), dim3(1u),
                 dim3(1u), arguments);
    synchronize("stage captured form credit input");
  }

  void ordinary_develop(std::uint64_t ticks = 1u) {
    grown_.develop(ticks);
    require_cuda(cudaMemcpy(&credit_host_inputs_, credit_inputs_,
                            sizeof(credit_host_inputs_), cudaMemcpyDeviceToHost),
                 "read form credit stage state");
  }

  credit::Receipt credit_receipt() const { return copy(credit_receipt_); }

  void withdraw_sources() {
    require_cuda(cudaMemset(input_, 0, kMaxInputPhases),
                 "withdraw form input source");
    require_cuda(cudaMemset(output_, 0, kMaxOutputPhases),
                 "withdraw form output source");
    credit_host_inputs_.input_staged = 0u;
    credit_host_inputs_.output_staged = 0u;
    credit_host_inputs_.output_contact_staged = 0u;
    credit_host_inputs_.output_contact_end = 0u;
    require_cuda(cudaMemset(credit_input_, 0, kMaxInputPhases),
                 "withdraw form credit input");
    require_cuda(cudaMemset(credit_output_, 0, kMaxOutputPhases),
                 "withdraw form credit output");
    require_cuda(cudaMemset(generate_receipt_, 0, sizeof(GenerateReceipt)),
                 "withdraw generated form receipt");
    require_cuda(cudaMemset(selected_control_, 0, 1u),
                 "withdraw selected form control");
    reset_capture();
    upload_credit_inputs();
    synchronize("withdraw form sources");
  }

  RawByteDecode emit(std::uint32_t phase) {
    // ⚠ THIS ONE WRITES AT DECLARED MOTOR PORTS AND IS STILL NOT A BOUNDARY
    // TRANSACTION. It lands on the ports but carries no MembraneReceipt and does
    // not go through exchange_boundary_raw_byte, so it has the location of a
    // transaction without the provenance of one. Named for what it is; earning
    // the port means going through the transaction, not sitting on its slot.
    grown_.resident_stage(
        ResidentStageReason::stage_motor_byte, [&](SiteWord* words) {
          const std::uint64_t motor_zero_slot =
              grown_.boundary_port_slot(adult::kRawMotorZeroPort);
          const std::uint64_t motor_one_slot =
              grown_.boundary_port_slot(adult::kRawMotorOnePort);
          void* arguments[] = {
              kernel_argument(words), kernel_argument(motor_zero_slot),
              kernel_argument(motor_one_slot), kernel_argument(generate_receipt_),
              kernel_argument(phase)};
          launch_graph(stage_motor_graph_, "launch form motor graph",
                       reinterpret_cast<void*>(stage_motor_kernel), dim3(1u),
                       dim3(1u), arguments);
        });
    synchronize("stage form motor byte");
    return grown_.extract_motor_raw_byte();
  }

  LesionReceipt exchange_regions(std::uint32_t first_region, std::uint32_t second_region) {
    grown_.intervene(InterventionReason::region_exchange, [&](SiteWord* words) {
      void* arguments[] = {
          kernel_argument(words), kernel_argument(layout_device_),
          kernel_argument(first_region), kernel_argument(second_region),
          kernel_argument(lesion_receipt_)};
      launch_graph(exchange_regions_graph_, "launch region-exchange graph",
                   reinterpret_cast<void*>(exchange_regions_kernel), dim3(1u),
                   dim3(1u), arguments);
    });
    synchronize("exchange form regions");
    return copy(lesion_receipt_);
  }

  situation::LesionReceipt exchange_situation_outcome(std::uint32_t region) {
    grown_.intervene(
        InterventionReason::situation_outcome_exchange, [&](SiteWord* words) {
          void* arguments[] = {
              kernel_argument(words), kernel_argument(situation_layout_),
              kernel_argument(region), kernel_argument(situation_lesion_receipt_)};
          launch_graph(exchange_situation_graph_,
                       "launch situation-exchange graph",
                       reinterpret_cast<void*>(exchange_situation_outcome_kernel),
                       dim3(1u), dim3(1u), arguments);
        });
    synchronize("exchange situation outcome");
    return copy(situation_lesion_receipt_);
  }

  [[nodiscard]] const DeviceLayout& layout() const { return layout_; }

  [[nodiscard]] std::uint8_t* device_captured_input() const {
    return captured_input_;
  }

  [[nodiscard]] std::uint32_t* device_capture_valid() const {
    return capture_valid_;
  }

 private:
  struct KernelGraph {
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t executable = nullptr;
    cudaGraphNode_t node = nullptr;

    void destroy() {
      if (executable != nullptr) cudaGraphExecDestroy(executable);
      if (graph != nullptr) cudaGraphDestroy(graph);
      executable = nullptr;
      graph = nullptr;
      node = nullptr;
    }

    ~KernelGraph() { destroy(); }
  };

  template <typename T>
  T copy(const T* device) const {
    T host{};
    require_cuda(cudaMemcpy(&host, device, sizeof(T), cudaMemcpyDeviceToHost), "copy form receipt");
    return host;
  }

  static void synchronize(const char* operation) {
    require_cuda(cudaGetLastError(), operation);
    require_cuda(cudaDeviceSynchronize(), operation);
  }

  template <typename T>
  static void* kernel_argument(T& value) {
    return const_cast<void*>(static_cast<const void*>(&value));
  }

  static void launch_graph(KernelGraph& graph, const char* operation,
                           void* function, dim3 grid, dim3 block,
                           void** arguments) {
    cudaKernelNodeParams params{};
    params.func = function;
    params.gridDim = grid;
    params.blockDim = block;
    params.kernelParams = arguments;
    if (graph.executable == nullptr) {
      require_cuda(cudaGraphCreate(&graph.graph, 0u), operation);
      require_cuda(cudaGraphAddKernelNode(&graph.node, graph.graph, nullptr,
                                          0u, &params),
                   operation);
      require_cuda(cudaGraphInstantiate(&graph.executable, graph.graph,
                                        nullptr, nullptr, 0u),
                   operation);
    } else {
      require_cuda(cudaGraphExecKernelNodeSetParams(
                       graph.executable, graph.node, &params),
                   operation);
    }
    require_cuda(cudaGraphLaunch(graph.executable, 0), operation);
  }

  void launch_generate(const std::uint8_t* input, std::uint32_t input_count,
                       const std::uint32_t* capture_valid,
                       std::uint32_t required_capture_mask) {
    const SiteWord* words = grown_.device_words();
    void* arguments[] = {
        kernel_argument(words), kernel_argument(layout_device_),
        kernel_argument(active_context_layout_), kernel_argument(input),
        kernel_argument(input_count), kernel_argument(capture_valid),
        kernel_argument(required_capture_mask), kernel_argument(generate_receipt_)};
    launch_graph(generate_graph_, "launch form generation graph",
                 reinterpret_cast<void*>(generate_kernel), dim3(1u), dim3(1u),
                 arguments);
  }

  void destroy_graphs() {
    learn_graph_.destroy();
    generate_graph_.destroy();
    reset_capture_graph_.destroy();
    capture_generated_graph_.destroy();
    bind_grounded_relation_graph_.destroy();
    bind_unique_active_instance_graph_.destroy();
    bind_directional_clause_graph_.destroy();
    select_control_graph_.destroy();
    stage_captured_input_graph_.destroy();
    stage_motor_graph_.destroy();
    exchange_regions_graph_.destroy();
    exchange_situation_graph_.destroy();
  }

  void upload_credit_inputs() {
    require_cuda(cudaMemcpy(credit_inputs_, &credit_host_inputs_,
                            sizeof(credit_host_inputs_), cudaMemcpyHostToDevice),
                 "upload form credit inputs");
  }

  GrownAdult& grown_;
  DeviceLayout layout_{};
  DeviceLayout* layout_device_ = nullptr;
  situation::DeviceLayout situation_layout_{};
  GroundedContextLayout grounded_context_layout_{};
  ActiveContextLayout active_context_layout_{};
  FormContextSource context_source_ = FormContextSource::Situation;
  std::uint8_t* input_ = nullptr;
  std::uint8_t* output_ = nullptr;
  std::uint8_t* captured_input_ = nullptr;
  std::uint32_t* capture_valid_ = nullptr;
  std::uint8_t* selected_control_ = nullptr;
  std::uint8_t* credit_input_ = nullptr;
  std::uint8_t* credit_output_ = nullptr;
  credit::DeviceInputs credit_host_inputs_{};
  credit::DeviceInputs* credit_inputs_ = nullptr;
  credit::Receipt* credit_receipt_ = nullptr;
  LearnReceipt* learn_receipt_ = nullptr;
  GenerateReceipt* generate_receipt_ = nullptr;
  LesionReceipt* lesion_receipt_ = nullptr;
  situation::LesionReceipt* situation_lesion_receipt_ = nullptr;
  KernelGraph learn_graph_;
  KernelGraph generate_graph_;
  KernelGraph reset_capture_graph_;
  KernelGraph capture_generated_graph_;
  KernelGraph bind_grounded_relation_graph_;
  KernelGraph bind_unique_active_instance_graph_;
  KernelGraph bind_directional_clause_graph_;
  KernelGraph select_control_graph_;
  KernelGraph stage_captured_input_graph_;
  KernelGraph stage_motor_graph_;
  KernelGraph exchange_regions_graph_;
  KernelGraph exchange_situation_graph_;
};

}  // namespace substrate::bcc32::grown_form_trajectory
