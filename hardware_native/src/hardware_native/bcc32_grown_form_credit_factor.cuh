#pragma once

#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "hardware_native/bcc32_aperture_geometry.cuh"
#include "hardware_native/bcc32_law.cuh"
#include "hardware_native/bcc32_raw_byte_tape.cuh"
#include "hardware_native/bcc32_types.cuh"

namespace substrate::bcc32::grown_form_credit_factor {

using substrate::bcc32::SiteWord;

inline constexpr std::uint32_t kMaxInputPhases = 6u;
inline constexpr std::uint32_t kMaxOutputPhases = 20u;
inline constexpr std::uint32_t kMaxPhases = kMaxOutputPhases;
static_assert(kMaxOutputPhases < 32u,
              "form output length is encoded in one SiteWord bitmask");
inline constexpr std::uint32_t kMapFields =
    kMaxOutputPhases * kMaxInputPhases;
// kMapFields is retained as the diagnostic route-space width.  The resident
// form tissue stores one candidate-position mask per output phase instead of
// one physical field for every route.
inline constexpr std::uint32_t kCandidateBase = 0u;
inline constexpr std::uint32_t kPrototypeBase =
    kCandidateBase + kMaxOutputPhases;
inline constexpr std::uint32_t kAgreementBase =
    kPrototypeBase + kMaxOutputPhases;
inline constexpr std::uint32_t kSeenBase =
    kAgreementBase + kMaxOutputPhases;
inline constexpr std::uint32_t kLengthField =
    kSeenBase + kMaxOutputPhases;
inline constexpr std::uint32_t kDispositionField = kLengthField + 1u;
inline constexpr std::uint32_t kFormFieldsPerRegion = kDispositionField + 1u;
inline constexpr std::uint32_t kRegionCount = 64u;
inline constexpr std::uint32_t kFormRailCount =
    kRegionCount * kFormFieldsPerRegion * 2u;

inline constexpr std::uint32_t kGlobalFields = 3u;
inline constexpr std::uint32_t kInputBase = 0u;
inline constexpr std::uint32_t kPredictionBase =
    kInputBase + kMaxInputPhases;
inline constexpr std::uint32_t kEligibilityBase =
    kPredictionBase + kMaxOutputPhases;
inline constexpr std::uint32_t kLiteralEligibilityBase =
    kEligibilityBase + kMaxOutputPhases;
inline constexpr std::uint32_t kEligibilityFieldCount =
    kMaxOutputPhases * 2u;
inline constexpr std::uint32_t kPendingField =
    kEligibilityBase + kEligibilityFieldCount;
inline constexpr std::uint32_t kInputCountField = kPendingField + 1u;
// The cursor is resident chronology, not an observer-provided phase label.
// It advances only after one accepted contact byte and is reset on input.
inline constexpr std::uint32_t kOutputCursorField = kInputCountField + 1u;
inline constexpr std::uint32_t kCreditFieldsPerRegion = kOutputCursorField + 1u;
inline constexpr std::uint32_t kCreditResidentRailCount =
    kGlobalFields * 2u + kRegionCount * kCreditFieldsPerRegion * 2u;
inline constexpr std::uint32_t kResidentRailCount =
    kFormRailCount + kCreditResidentRailCount;
inline constexpr std::uint32_t kEligibilityLifetime = 32u;
inline constexpr SiteWord kEligibilityExpired = kEligibilityLifetime + 1u;
// Eligibility duration is learned-time state. Undo capacity is a separate
// bounded integration scaffold and must never wrap destructively.
inline constexpr std::uint32_t kJournalDepth = kEligibilityLifetime;
inline constexpr std::uint32_t kJournalCountField = 1u;
inline constexpr std::uint32_t kJournalCursorField = 2u;
inline constexpr std::uint32_t kPhysicalRailCount =
    kCreditResidentRailCount + kJournalDepth * kResidentRailCount;
inline constexpr std::uint32_t kPrototypePairCount =
    kRegionCount * kMaxOutputPhases;
inline constexpr std::uint32_t kFounderJournalBits =
    ((kResidentRailCount / 2u - kPrototypePairCount - 1u) * 32u) +
    (kPrototypePairCount * 10u) + 32u;
// The marker is also the resident layout epoch. Capacity 20 reindexes the
// prototype/agreement/credit fields, so an old 14-phase checkpoint must fail
// attachment instead of being interpreted under the new field geometry.
inline constexpr SiteWord kFactorMarkerValue = 0x326d7246u;  // "Frm2"
inline constexpr std::uint32_t kLiteralRoute = kMaxInputPhases;
inline constexpr std::uint32_t kInvalidRoute = kLiteralRoute + 1u;
inline constexpr std::uint32_t kValidPrediction = 1u << 11u;
inline constexpr std::uint32_t kRouteShift = 8u;
inline constexpr std::uint32_t kRouteMask = 0x7u;

struct FormLayout {
  std::uint64_t rails[kFormRailCount]{};
};

struct PhysicalOffset {
  std::int32_t x = 0;
  std::int32_t y = 0;
  std::int32_t z = 0;
};

struct DeviceLayout {
  FormLayout form{};
  std::uint64_t rails[kCreditResidentRailCount]{};
};

struct DeviceInputs {
  const std::uint8_t* input = nullptr;
  const std::uint8_t* output = nullptr;
  std::uint64_t active_rails[2]{};
  std::uint32_t input_count = 0u;
  std::uint32_t output_count = 0u;
  std::uint32_t input_staged = 0u;
  std::uint32_t output_staged = 0u;
  std::uint32_t output_contact_staged = 0u;
  std::uint32_t output_contact_end = 0u;
  std::uint32_t allow_single_region_complete = 0u;
};

struct Receipt {
  std::uint32_t input_staged = 0u;
  std::uint32_t output_staged = 0u;
  std::uint32_t eligibility_created = 0u;
  std::uint32_t eligibility_aged = 0u;
  std::uint32_t eligibility_expired = 0u;
  std::uint32_t mapped_eligibility_expired = 0u;
  std::uint32_t literal_eligibility_expired = 0u;
  std::uint32_t predicted_events = 0u;
  std::uint32_t correct_events = 0u;
  std::uint32_t error_events = 0u;
  std::uint32_t positive_updates = 0u;
  std::uint32_t negative_updates = 0u;
  std::uint32_t expired_output_no_update = 0u;
  std::uint32_t contact_events = 0u;
  std::uint32_t contact_end_events = 0u;
  std::uint32_t contact_rejected = 0u;
  std::uint32_t contact_cursor_advances = 0u;
  std::uint32_t contact_no_update = 0u;
  std::uint32_t staged_rejected = 0u;
  std::uint32_t input_rejected = 0u;
  std::uint32_t output_rejected = 0u;
  std::uint32_t last_contact_phase = 0u;
  std::uint32_t live_eligibility_routes = 0u;
  std::uint32_t ordinary_events = 0u;
  std::uint32_t matter_before_bits = 0u;
  std::uint32_t matter_after_bits = 0u;
  std::uint32_t eligibility_matter_before_bits = 0u;
  std::uint32_t eligibility_matter_after_bits = 0u;
  std::uint32_t eligibility_matter_violations = 0u;
  std::uint32_t journal_count = 0u;
  std::uint8_t last_input[kMaxInputPhases]{};
  std::uint8_t last_output[kMaxOutputPhases]{};
  std::uint8_t last_prediction[kMaxOutputPhases]{};
  std::uint8_t last_stored_input[kMaxInputPhases]{};
  std::uint32_t last_eligible_mask = 0u;
  std::uint32_t last_match_mask = 0u;
  std::uint32_t last_eligible_routes = 0u;
  std::uint32_t last_matching_routes = 0u;
};

inline void require_cuda(cudaError_t status, const char* operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(status));
  }
}

__host__ __device__ inline std::uint32_t form_index(
    std::uint32_t region, std::uint32_t field, std::uint32_t polarity) {
  return (region * kFormFieldsPerRegion + field) * 2u + polarity;
}

__host__ __device__ inline std::uint32_t credit_index(
    std::uint32_t region, std::uint32_t field, std::uint32_t polarity) {
  return kGlobalFields * 2u +
         (region * kCreditFieldsPerRegion + field) * 2u + polarity;
}

__host__ __device__ inline std::uint32_t global_index(
    std::uint32_t field, std::uint32_t polarity) {
  return field * 2u + polarity;
}

__host__ __device__ inline std::uint32_t map_field(
    std::uint32_t output_phase, std::uint32_t input_phase) {
  return output_phase * kMaxInputPhases + input_phase;
}

// region in [0, kRegionCount), field in [0, kFormFieldsPerRegion), polarity
// in {0,1}. z0=-224 keeps this region inside the kSpatialMacroClosureRadius=26
// clearance window (previously z0=-240 put it at global z as low as 10).
__host__ __device__ inline PhysicalOffset form_physical_offset(
    std::uint32_t region, std::uint32_t field, std::uint32_t polarity) {
  const std::int32_t region_x =
      -28 + static_cast<std::int32_t>(region % 8u) * 8;
  return {
      region_x + static_cast<std::int32_t>(field % 4u),
      -28 + static_cast<std::int32_t>(region / 8u) * 8,
      -224 + static_cast<std::int32_t>((field / 4u) * 2u + polarity)};
}

// index in [0, kPhysicalRailCount) (credit-resident rails plus their journal
// copies). y0=24 keeps every site inside the clearance window (previously
// y0=130 let the top of the journal range reach global y=579, outside the
// whole 500-wide aperture, not merely outside the 26-hop clearance).
__host__ __device__ inline PhysicalOffset physical_offset(
    std::uint32_t index) {
  return {-220 + static_cast<std::int32_t>(index % 200u),
          24 + static_cast<std::int32_t>((index / 200u) % 200u),
          20 + static_cast<std::int32_t>(index / 40000u)};
}

__host__ __device__ inline std::uint32_t journal_index(
    std::uint32_t event, std::uint32_t resident_index) {
  return kCreditResidentRailCount + event * kResidentRailCount +
         resident_index;
}

__host__ __device__ inline std::uint64_t fixed_physical_slot(
    std::uint32_t index) {
  // edge_chunks, chunk_edge, chunk_sites, and centre are the shared aperture
  // geometry defined once in bcc32_aperture_geometry.cuh, which this file
  // can include without a cycle (bcc32_developmental_adult.cuh includes
  // this file, not the other way around).
  constexpr std::uint64_t edge_chunks =
      static_cast<std::uint64_t>(kApertureEdgeChunks);
  constexpr std::uint64_t chunk_edge = kApertureChunkEdge;
  constexpr std::uint64_t chunk_sites = kApertureChunkSites;
  constexpr std::int32_t centre = kApertureCentre;
  const PhysicalOffset offset = physical_offset(index);
  const std::int64_t gx = static_cast<std::int64_t>(centre) + offset.x;
  const std::int64_t gy = static_cast<std::int64_t>(centre) + offset.y;
  const std::int64_t gz = static_cast<std::int64_t>(centre) + offset.z;
  const std::uint64_t chunk =
      static_cast<std::uint64_t>(gx / static_cast<std::int64_t>(chunk_edge)) *
          edge_chunks * edge_chunks +
      static_cast<std::uint64_t>(gy / static_cast<std::int64_t>(chunk_edge)) *
          edge_chunks +
      static_cast<std::uint64_t>(gz / static_cast<std::int64_t>(chunk_edge));
  const std::uint64_t local =
      ((static_cast<std::uint64_t>(gx % chunk_edge) * chunk_edge +
        static_cast<std::uint64_t>(gy % chunk_edge)) * chunk_edge) +
      static_cast<std::uint64_t>(gz % chunk_edge);
  return chunk * chunk_sites + local;
}

template <typename Adult>
inline FormLayout make_form_layout(const Adult& grown) {
  FormLayout layout{};
  for (std::uint32_t region = 0u; region < kRegionCount; ++region) {
    for (std::uint32_t field = 0u; field < kFormFieldsPerRegion; ++field) {
      const PhysicalOffset first = form_physical_offset(region, field, 0u);
      const PhysicalOffset second = form_physical_offset(region, field, 1u);
      layout.rails[form_index(region, field, 0u)] =
          grown.physical_slot({first.x, first.y, first.z});
      layout.rails[form_index(region, field, 1u)] =
          grown.physical_slot({second.x, second.y, second.z});
    }
  }
  return layout;
}

template <typename Adult>
inline DeviceLayout make_layout(const Adult& grown) {
  DeviceLayout layout{};
  layout.form = make_form_layout(grown);
  for (std::uint32_t index = 0u; index < kCreditResidentRailCount; ++index) {
    const PhysicalOffset offset = physical_offset(index);
    layout.rails[index] = grown.physical_slot({offset.x, offset.y, offset.z});
  }
  return layout;
}

__host__ __device__ inline void resident_founder_pair(
    std::uint32_t resident, SiteWord* first, SiteWord* second) {
  *first = 0u;
  *second = 0xffffffffu;
  if (resident < kFormRailCount) {
    const std::uint32_t field =
        (resident / 2u) % kFormFieldsPerRegion;
    if (field >= kPrototypeBase &&
        field < kPrototypeBase + kMaxOutputPhases) {
      RawByteRails prototype =
          with_raw_byte_carriers(RawByteRails{}, 0u);
      prototype.zero |= face_bit(0u);
      prototype.one |= face_bit(1u);
      *first = prototype.zero;
      *second = prototype.one;
    }
    return;
  }
  if (resident - kFormRailCount == global_index(0u, 0u)) {
    *first = kFactorMarkerValue;
    *second = ~kFactorMarkerValue;
  }
}

template <typename Entry, typename Adult>
inline std::vector<Entry> founder_entries(const Adult& grown) {
  const DeviceLayout layout = make_layout(grown);
  std::vector<Entry> entries;
  entries.reserve(kPhysicalRailCount);
  for (std::uint32_t index = 0u; index < kPhysicalRailCount; index += 2u) {
    SiteWord value = 0u;
    if (index == global_index(0u, 0u)) value = kFactorMarkerValue;
    const std::uint64_t first =
        index < kCreditResidentRailCount
            ? layout.rails[index]
            : fixed_physical_slot(index);
    const std::uint64_t second =
        index + 1u < kCreditResidentRailCount
            ? layout.rails[index + 1u]
            : fixed_physical_slot(index + 1u);
    SiteWord second_value = ~value;
    if (index >= kCreditResidentRailCount) {
      const std::uint32_t resident =
          (index - kCreditResidentRailCount) % kResidentRailCount;
      resident_founder_pair(resident, &value, &second_value);
    }
    entries.push_back({first, value});
    entries.push_back({second, second_value});
  }
  return entries;
}

__device__ inline SiteWord read_form(const SiteWord* words,
                                     const DeviceLayout& layout,
                                     std::uint32_t region,
                                     std::uint32_t field) {
  return words[layout.form.rails[form_index(region, field, 0u)]];
}

__device__ inline void write_form(SiteWord* words, const DeviceLayout& layout,
                                  std::uint32_t region, std::uint32_t field,
                                  SiteWord value) {
  words[layout.form.rails[form_index(region, field, 0u)]] = value;
  words[layout.form.rails[form_index(region, field, 1u)]] = ~value;
}

__device__ inline SiteWord read_credit(const SiteWord* words,
                                       const DeviceLayout& layout,
                                       std::uint32_t region,
                                       std::uint32_t field) {
  return words[layout.rails[credit_index(region, field, 0u)]];
}

__device__ inline void write_credit(SiteWord* words, const DeviceLayout& layout,
                                    std::uint32_t region, std::uint32_t field,
                                    SiteWord value) {
  words[layout.rails[credit_index(region, field, 0u)]] = value;
  words[layout.rails[credit_index(region, field, 1u)]] = ~value;
}

__device__ inline SiteWord read_global(const SiteWord* words,
                                       const DeviceLayout& layout,
                                       std::uint32_t field) {
  return words[layout.rails[global_index(field, 0u)]];
}

__device__ inline void write_global(SiteWord* words, const DeviceLayout& layout,
                                    std::uint32_t field, SiteWord value) {
  words[layout.rails[global_index(field, 0u)]] = value;
  words[layout.rails[global_index(field, 1u)]] = ~value;
}

__device__ inline SiteWord resident_value(const SiteWord* words,
                                          const DeviceLayout& layout,
                                          std::uint32_t index) {
  if (index < kFormRailCount)
    return words[layout.form.rails[index]];
  return words[layout.rails[index - kFormRailCount]];
}

__device__ inline void write_journal_pair(SiteWord* words,
                                          const DeviceLayout& layout,
                                          std::uint32_t event,
                                          std::uint32_t resident_index,
                                          SiteWord first_value,
                                          SiteWord second_value) {
  const std::uint32_t first = journal_index(event, resident_index);
  words[fixed_physical_slot(first)] = first_value;
  words[fixed_physical_slot(first + 1u)] = second_value;
}

__device__ inline bool journal_available(const SiteWord* words,
                                         const DeviceLayout& layout) {
  const SiteWord count = read_global(words, layout, kJournalCountField);
  const SiteWord cursor = read_global(words, layout, kJournalCursorField);
  return count < kJournalDepth && cursor == count;
}

__device__ inline bool begin_journal(SiteWord* words, const DeviceLayout& layout) {
  const SiteWord count = read_global(words, layout, kJournalCountField);
  const SiteWord cursor = read_global(words, layout, kJournalCursorField);
  if (count >= kJournalDepth || cursor != count) return false;
  const std::uint32_t event = count;
  for (std::uint32_t index = 0u; index < kResidentRailCount; index += 2u)
    write_journal_pair(words, layout, event, index,
                       resident_value(words, layout, index),
                       resident_value(words, layout, index + 1u));
  write_global(words, layout, kJournalCursorField, count + 1u);
  write_global(words, layout, kJournalCountField, count + 1u);
  return true;
}

__device__ inline bool has_live_eligibility(const SiteWord* words,
                                            const DeviceLayout& layout) {
  for (std::uint32_t region = 0u; region < kRegionCount; ++region) {
    for (std::uint32_t route = 0u; route < kEligibilityFieldCount; ++route) {
      const SiteWord age =
          read_credit(words, layout, region, kEligibilityBase + route);
      if (age != 0u && age <= kEligibilityLifetime)
        return true;
    }
  }
  return false;
}

__device__ inline std::uint64_t active_regions(const SiteWord* words,
                                               const DeviceInputs& inputs) {
  return
      static_cast<unsigned long long>(words[inputs.active_rails[0]]) |
      (static_cast<unsigned long long>(words[inputs.active_rails[1]]) << 32u);
}

enum StepMode : std::uint32_t {
  kStepNone = 0u,
  kStepJournaled = 1u,
  kStepEligibilityAge = 2u,
};

__device__ inline StepMode required_step_mode(const SiteWord* words,
                                              const DeviceLayout& layout,
                                              const DeviceInputs* inputs) {
  const std::uint64_t regions =
      inputs != nullptr ? active_regions(words, *inputs) : 0u;
  const bool staged =
      inputs != nullptr &&
      (inputs->input_staged != 0u || inputs->output_staged != 0u ||
       inputs->output_contact_staged != 0u);
  if (staged) return kStepJournaled;
  return has_live_eligibility(words, layout) ? kStepEligibilityAge : kStepNone;
}

__device__ inline bool step_requires_journal(const SiteWord* words,
                                             const DeviceLayout& layout,
                                             const DeviceInputs* inputs) {
  return required_step_mode(words, layout, inputs) == kStepJournaled;
}

__device__ inline bool active_pending(const SiteWord* words,
                                      const DeviceLayout& layout,
                                      std::uint64_t regions) {
  for (std::uint32_t region = 0u; region < kRegionCount; ++region) {
    if ((regions & (1ull << region)) == 0u)
      continue;
    if (read_credit(words, layout, region, kPendingField) != 0u)
      return true;
  }
  return false;
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
    SiteWord* words, const DeviceLayout& layout, std::uint32_t region,
    std::uint32_t active_count, std::uint32_t active_rank,
    bool allow_single_region_complete) {
  const SiteWord resident =
      read_form(words, layout, region, kDispositionField) &
      ((1u << kMaxOutputPhases) - 1u);
  if (resident != 0u)
    return resident;
  const std::uint32_t acquired = coalition_phase_mask(
      active_count, active_rank, allow_single_region_complete);
  if (acquired != 0u)
    write_form(words, layout, region, kDispositionField, acquired);
  return acquired;
}

__device__ inline SiteWord increment_support(SiteWord value) {
  const SiteWord empty = ~value;
  return empty == 0u ? value : value | (empty & (0u - empty));
}

__device__ inline SiteWord decrement_support(SiteWord value) {
  return value == 0u ? 0u : value & (value - 1u);
}

__device__ inline std::uint32_t support_count(SiteWord value) {
  return __popc(value);
}

__device__ inline void intersect_length(SiteWord* words,
                                        const DeviceLayout& layout,
                                        std::uint32_t region,
                                        std::uint32_t output_phase) {
  const SiteWord observed = 1u << output_phase;
  const SiteWord prior = read_form(words, layout, region, kLengthField);
  write_form(words, layout, region, kLengthField,
             prior == 0u ? observed : prior & observed);
}

__device__ inline SiteWord input_phase_mask(std::uint32_t input_count) {
  return input_count >= 32u ? 0xffffffffu
                           : (input_count == 0u ? 0u : (1u << input_count) - 1u);
}

__device__ inline SiteWord matching_input_mask(const SiteWord* words,
                                               const DeviceLayout& layout,
                                               std::uint32_t region,
                                               std::uint32_t input_count,
                                               std::uint8_t observed) {
  SiteWord matches = 0u;
  const std::uint32_t count =
      input_count < kMaxInputPhases ? input_count : kMaxInputPhases;
  for (std::uint32_t input_phase = 0u; input_phase < count; ++input_phase) {
    if (static_cast<std::uint8_t>(read_credit(
            words, layout, region, kInputBase + input_phase)) == observed)
      matches |= 1u << input_phase;
  }
  return matches;
}

__device__ inline std::uint32_t unique_candidate(SiteWord mask) {
  return support_count(mask) == 1u
             ? static_cast<std::uint32_t>(__ffs(static_cast<int>(mask)) - 1)
             : kInvalidRoute;
}

__device__ inline std::uint32_t encode_prediction(std::uint8_t value,
                                                  std::uint32_t route) {
  return static_cast<std::uint32_t>(value) | (route << kRouteShift) | kValidPrediction;
}

__device__ inline void age_eligibility(SiteWord* words, const DeviceLayout& layout,
                                       Receipt* receipt) {
  receipt->eligibility_matter_before_bits = 0u;
  receipt->eligibility_matter_after_bits = 0u;
  SiteWord mapped_aged_phases = 0u;
  SiteWord mapped_expired_phases = 0u;
  SiteWord literal_aged_phases = 0u;
  SiteWord literal_expired_phases = 0u;
  for (std::uint32_t region = 0u; region < kRegionCount; ++region) {
    for (std::uint32_t phase = 0u; phase < kMaxOutputPhases; ++phase) {
      const SiteWord prior =
          read_credit(words, layout, region, kEligibilityBase + phase);
      receipt->eligibility_matter_before_bits +=
          __popc(words[layout.rails[credit_index(
              region, kEligibilityBase + phase, 0u)]]) +
          __popc(words[layout.rails[credit_index(
              region, kEligibilityBase + phase, 1u)]]);
      if (prior == kEligibilityLifetime) {
        if ((mapped_expired_phases & (1u << phase)) == 0u) {
          mapped_expired_phases |= 1u << phase;
          receipt->eligibility_expired += kMaxInputPhases;
          receipt->mapped_eligibility_expired += kMaxInputPhases;
        }
      }
      if (prior != 0u && prior <= kEligibilityLifetime &&
          (mapped_aged_phases & (1u << phase)) == 0u) {
        mapped_aged_phases |= 1u << phase;
        receipt->eligibility_aged += kMaxInputPhases;
      }
      write_credit(words, layout, region, kEligibilityBase + phase,
                   prior != 0u && prior <= kEligibilityLifetime
                       ? prior + 1u
                       : prior);
      receipt->eligibility_matter_after_bits +=
          __popc(words[layout.rails[credit_index(
              region, kEligibilityBase + phase, 0u)]]) +
          __popc(words[layout.rails[credit_index(
              region, kEligibilityBase + phase, 1u)]]);
      const SiteWord literal_prior = read_credit(
          words, layout, region, kLiteralEligibilityBase + phase);
      receipt->eligibility_matter_before_bits +=
          __popc(words[layout.rails[credit_index(
              region, kLiteralEligibilityBase + phase, 0u)]]) +
          __popc(words[layout.rails[credit_index(
              region, kLiteralEligibilityBase + phase, 1u)]]);
      if (literal_prior == kEligibilityLifetime) {
        if ((literal_expired_phases & (1u << phase)) == 0u) {
          literal_expired_phases |= 1u << phase;
          ++receipt->eligibility_expired;
          ++receipt->literal_eligibility_expired;
        }
      }
      if (literal_prior != 0u && literal_prior <= kEligibilityLifetime &&
          (literal_aged_phases & (1u << phase)) == 0u) {
        literal_aged_phases |= 1u << phase;
        ++receipt->eligibility_aged;
      }
      write_credit(words, layout, region, kLiteralEligibilityBase + phase,
                   literal_prior != 0u &&
                           literal_prior <= kEligibilityLifetime
                       ? literal_prior + 1u
                       : literal_prior);
      receipt->eligibility_matter_after_bits +=
          __popc(words[layout.rails[credit_index(
              region, kLiteralEligibilityBase + phase, 0u)]]) +
          __popc(words[layout.rails[credit_index(
              region, kLiteralEligibilityBase + phase, 1u)]]);
    }
  }
  if (receipt->eligibility_matter_before_bits !=
      receipt->eligibility_matter_after_bits)
    ++receipt->eligibility_matter_violations;
}

__device__ inline void inverse_age_eligibility(
    SiteWord* words, const DeviceLayout& layout) {
  for (std::uint32_t region = 0u; region < kRegionCount; ++region) {
    for (std::uint32_t phase = 0u; phase < kMaxOutputPhases; ++phase) {
      const SiteWord mapped =
          read_credit(words, layout, region, kEligibilityBase + phase);
      if (mapped >= 2u && mapped <= kEligibilityExpired)
        write_credit(words, layout, region, kEligibilityBase + phase,
                     mapped - 1u);
      const SiteWord literal = read_credit(
          words, layout, region, kLiteralEligibilityBase + phase);
      if (literal >= 2u && literal <= kEligibilityExpired)
        write_credit(words, layout, region,
                     kLiteralEligibilityBase + phase, literal - 1u);
    }
  }
}

__device__ inline std::uint32_t candidate_route(const SiteWord* words,
                                                const DeviceLayout& layout,
                                                std::uint32_t region,
                                                std::uint32_t output_phase,
                                                std::uint32_t input_count) {
  const SiteWord mask =
      read_form(words, layout, region, kCandidateBase + output_phase) &
      input_phase_mask(input_count);
  return unique_candidate(mask);
}

__device__ inline void stage_prediction(SiteWord* words, const DeviceLayout& layout,
                                         std::uint32_t region,
                                         std::uint32_t phase_mask,
                                         const DeviceInputs& inputs,
                                         Receipt* receipt) {
  const std::uint32_t input_count =
      inputs.input_count < kMaxInputPhases ? inputs.input_count : kMaxInputPhases;
  for (std::uint32_t phase = 0u; phase < kMaxInputPhases; ++phase) {
    const SiteWord value = phase < input_count ? inputs.input[phase] : 0u;
    write_credit(words, layout, region, kInputBase + phase, value);
    receipt->last_input[phase] = static_cast<std::uint8_t>(value);
    receipt->last_stored_input[phase] = static_cast<std::uint8_t>(
        read_credit(words, layout, region, kInputBase + phase));
  }
  write_credit(words, layout, region, kInputCountField, input_count);
  for (std::uint32_t output_phase = 0u; output_phase < kMaxOutputPhases; ++output_phase) {
    if ((phase_mask & (1u << output_phase)) == 0u)
      continue;
    write_credit(words, layout, region, kEligibilityBase + output_phase,
                 1u);
    write_credit(words, layout, region, kLiteralEligibilityBase + output_phase,
                 1u);
  }
  for (std::uint32_t output_phase = 0u; output_phase < kMaxOutputPhases; ++output_phase) {
    if ((phase_mask & (1u << output_phase)) == 0u) {
      write_credit(words, layout, region, kPredictionBase + output_phase, 0u);
      continue;
    }
    const std::uint32_t route = candidate_route(
        words, layout, region, output_phase, input_count);
    if (route != kInvalidRoute) {
      write_credit(words, layout, region, kPredictionBase + output_phase,
                   encode_prediction(static_cast<std::uint8_t>(
                                         read_credit(words, layout, region, kInputBase + route)),
                                     route));
      ++receipt->predicted_events;
    } else {
      const SiteWord agreement =
          read_form(words, layout, region, kAgreementBase + output_phase) & 0xffu;
      const RawByteRails prototype{
          words[layout.form.rails[form_index(region, kPrototypeBase + output_phase, 0u)]],
          words[layout.form.rails[form_index(region, kPrototypeBase + output_phase, 1u)]]};
      const RawByteDecode decoded = decode_raw_byte_carriers(prototype);
      if (agreement == 0xffu && decoded.valid) {
        write_credit(words, layout, region, kPredictionBase + output_phase,
                     encode_prediction(decoded.value, kLiteralRoute));
        ++receipt->predicted_events;
      } else {
        write_credit(words, layout, region, kPredictionBase + output_phase, 0u);
      }
    }
  }
  write_credit(words, layout, region, kOutputCursorField, 0u);
  write_credit(words, layout, region, kPendingField, 1u);
  ++receipt->eligibility_created;
}

__device__ inline void clear_contact_pending(SiteWord* words,
                                             const DeviceLayout& layout,
                                             std::uint32_t region) {
  for (std::uint32_t phase = 0u; phase < kMaxOutputPhases; ++phase) {
    write_credit(words, layout, region, kEligibilityBase + phase, 0u);
    write_credit(words, layout, region, kLiteralEligibilityBase + phase, 0u);
    write_credit(words, layout, region, kPredictionBase + phase, 0u);
  }
  write_credit(words, layout, region, kOutputCursorField, 0u);
  write_credit(words, layout, region, kPendingField, 0u);
}

// Event-local contact path. The only phase authority here is the resident
// cursor. The contact carries one sensory byte and a generic boundary bit;
// it never carries a phase, route, candidate, or teacher tape.
__device__ inline void learn_observation_contact(
    SiteWord* words, const DeviceLayout& layout, std::uint32_t region,
    std::uint32_t phase_mask, const DeviceInputs& inputs, Receipt* receipt) {
  if (inputs.output_count != 1u || inputs.output_contact_end > 1u ||
      inputs.output == nullptr ||
      read_credit(words, layout, region, kPendingField) == 0u) {
    ++receipt->contact_rejected;
    ++receipt->contact_no_update;
    return;
  }

  const std::uint32_t output_phase = static_cast<std::uint32_t>(
      read_credit(words, layout, region, kOutputCursorField));
  if (output_phase >= kMaxOutputPhases ||
      (inputs.output_contact_end == 0u &&
       output_phase + 1u >= kMaxOutputPhases)) {
    ++receipt->contact_rejected;
    ++receipt->contact_no_update;
    clear_contact_pending(words, layout, region);
    return;
  }

  bool episode_live = false;
  for (std::uint32_t phase = 0u; phase < kMaxOutputPhases; ++phase) {
    const SiteWord mapped =
        read_credit(words, layout, region, kEligibilityBase + phase);
    const SiteWord literal =
        read_credit(words, layout, region, kLiteralEligibilityBase + phase);
    episode_live |=
        (mapped != 0u && mapped <= kEligibilityLifetime) ||
        (literal != 0u && literal <= kEligibilityLifetime);
  }
  if (!episode_live) {
    ++receipt->contact_no_update;
    ++receipt->expired_output_no_update;
    clear_contact_pending(words, layout, region);
    return;
  }

  const SiteWord mapped_age =
      read_credit(words, layout, region, kEligibilityBase + output_phase);
  const SiteWord literal_age = read_credit(
      words, layout, region, kLiteralEligibilityBase + output_phase);
  const bool mapped_eligible = mapped_age != 0u && mapped_age <= kEligibilityLifetime;
  const bool literal_eligible = literal_age != 0u && literal_age <= kEligibilityLifetime;
  const bool owns_phase = (phase_mask & (1u << output_phase)) != 0u;
  if (!owns_phase || (!mapped_eligible && !literal_eligible)) {
    ++receipt->contact_no_update;
    if (owns_phase)
      ++receipt->expired_output_no_update;
  } else {
    const std::uint8_t observed = inputs.output[0];
    receipt->last_output[output_phase] = observed;
    receipt->last_contact_phase = output_phase;
    const SiteWord prediction =
        read_credit(words, layout, region, kPredictionBase + output_phase);
    const bool predicted = (prediction & kValidPrediction) != 0u;
    const std::uint8_t predicted_value =
        static_cast<std::uint8_t>(prediction & 0xffu);
    receipt->last_prediction[output_phase] = predicted_value;
    const std::uint32_t predicted_route =
        (prediction >> kRouteShift) & kRouteMask;
    if (predicted) {
      if (predicted_value == observed) {
        ++receipt->correct_events;
      } else {
        ++receipt->error_events;
        if (predicted_route < kMaxInputPhases && mapped_eligible) {
          const SiteWord candidate =
              read_form(words, layout, region, kCandidateBase + output_phase);
          write_form(words, layout, region, kCandidateBase + output_phase,
                     candidate & ~(1u << predicted_route));
          ++receipt->negative_updates;
        } else if (predicted_route == kLiteralRoute && literal_eligible) {
          write_form(words, layout, region, kAgreementBase + output_phase, 0u);
          ++receipt->negative_updates;
        }
      }
    }

    const std::uint32_t input_count =
        read_credit(words, layout, region, kInputCountField) < kMaxInputPhases
            ? static_cast<std::uint32_t>(read_credit(
                  words, layout, region, kInputCountField))
            : kMaxInputPhases;
    const SiteWord matching =
        matching_input_mask(words, layout, region, input_count, observed);
    const SiteWord prior_candidate =
        read_form(words, layout, region, kCandidateBase + output_phase);
    const SiteWord seen =
        read_form(words, layout, region, kSeenBase + output_phase);
    const SiteWord candidate =
        seen == 0u ? matching : prior_candidate & matching;
    if (candidate != prior_candidate) {
      write_form(words, layout, region, kCandidateBase + output_phase,
                 candidate);
      ++receipt->positive_updates;
    }

    if (seen == 0u && candidate == 0u && literal_eligible &&
        (!predicted || predicted_value != observed)) {
      const std::uint32_t prototype_field = kPrototypeBase + output_phase;
      const RawByteRails prior_prototype{
          words[layout.form.rails[form_index(region, prototype_field, 0u)]],
          words[layout.form.rails[form_index(region, prototype_field, 1u)]]};
      const RawByteDecode prior_decoded =
          decode_raw_byte_carriers(prior_prototype);
      const SiteWord prior_agreement =
          read_form(words, layout, region, kAgreementBase + output_phase);
      if (prior_candidate != 0u || prior_agreement == 0u ||
          !prior_decoded.valid) {
        const RawByteRails prototype =
            with_raw_byte_carriers(prior_prototype, observed);
        words[layout.form.rails[form_index(region, prototype_field, 0u)]] =
            prototype.zero;
        words[layout.form.rails[form_index(region, prototype_field, 1u)]] =
            prototype.one;
        write_form(words, layout, region, kAgreementBase + output_phase, 0xffu);
        ++receipt->positive_updates;
      }
    }
    if (seen != 0u) {
      const std::uint32_t prototype_field = kPrototypeBase + output_phase;
      const RawByteRails prototype{
          words[layout.form.rails[form_index(region, prototype_field, 0u)]],
          words[layout.form.rails[form_index(region, prototype_field, 1u)]]};
      const RawByteDecode decoded = decode_raw_byte_carriers(prototype);
      const SiteWord agreement =
          read_form(words, layout, region, kAgreementBase + output_phase);
      write_form(
          words, layout, region, kAgreementBase + output_phase,
          agreement & static_cast<SiteWord>(
                          decoded.valid ? (~(decoded.value ^ observed) & 0xffu)
                                        : 0u));
    }
    write_form(words, layout, region, kSeenBase + output_phase,
               increment_support(read_form(words, layout, region,
                                           kSeenBase + output_phase)));
  }

  write_credit(words, layout, region, kEligibilityBase + output_phase, 0u);
  write_credit(words, layout, region, kLiteralEligibilityBase + output_phase, 0u);
  write_credit(words, layout, region, kPredictionBase + output_phase, 0u);
  ++receipt->contact_cursor_advances;
  if (inputs.output_contact_end != 0u) {
    intersect_length(words, layout, region, output_phase);
    clear_contact_pending(words, layout, region);
  } else {
    write_credit(words, layout, region, kOutputCursorField, output_phase + 1u);
  }
}

__device__ inline void learn_observation(SiteWord* words, const DeviceLayout& layout,
                                         std::uint32_t region,
                                         std::uint32_t phase_mask,
                                         const DeviceInputs& inputs,
                                         Receipt* receipt) {
  if (read_credit(words, layout, region, kPendingField) == 0u) return;
  const std::uint32_t output_count =
      inputs.output_count < kMaxOutputPhases ? inputs.output_count : kMaxOutputPhases;
  bool any_eligible = false;
  for (std::uint32_t route = 0u; route < kEligibilityFieldCount; ++route) {
    const SiteWord age =
        read_credit(words, layout, region, kEligibilityBase + route);
    any_eligible |= age != 0u && age <= kEligibilityLifetime;
  }

  if (!any_eligible) {
    for (std::uint32_t output_phase = 0u;
         output_phase < output_count; ++output_phase) {
      const SiteWord prediction =
          read_credit(words, layout, region, kPredictionBase + output_phase);
      if ((prediction & kValidPrediction) == 0u) continue;
      if (static_cast<std::uint8_t>(prediction & 0xffu) == inputs.output[output_phase])
        ++receipt->correct_events;
      else
        ++receipt->error_events;
    }
    ++receipt->expired_output_no_update;
    for (std::uint32_t route = 0u; route < kEligibilityFieldCount; ++route)
      write_credit(words, layout, region, kEligibilityBase + route, 0u);
    write_credit(words, layout, region, kPendingField, 0u);
    return;
  }

  for (std::uint32_t output_phase = 0u;
       output_phase < output_count; ++output_phase) {
    if ((phase_mask & (1u << output_phase)) == 0u)
      continue;
    const std::uint8_t observed = inputs.output[output_phase];
    receipt->last_output[output_phase] = observed;
    const SiteWord prediction =
        read_credit(words, layout, region, kPredictionBase + output_phase);
    const bool predicted = (prediction & kValidPrediction) != 0u;
    const std::uint8_t predicted_value = static_cast<std::uint8_t>(prediction & 0xffu);
    receipt->last_prediction[output_phase] = predicted_value;
    const std::uint32_t predicted_route =
        (prediction >> kRouteShift) & kRouteMask;
    if (predicted) {
      if (predicted_value == observed)
        ++receipt->correct_events;
      else {
        ++receipt->error_events;
        if (predicted_route < kMaxInputPhases &&
            read_credit(words, layout, region,
                        kEligibilityBase + output_phase) <=
                kEligibilityLifetime &&
            read_credit(words, layout, region,
                        kEligibilityBase + output_phase) != 0u) {
          const SiteWord candidate = read_form(
              words, layout, region, kCandidateBase + output_phase);
          write_form(words, layout, region, kCandidateBase + output_phase,
                     candidate & ~(1u << predicted_route));
          ++receipt->negative_updates;
        } else if (predicted_route == kLiteralRoute &&
                   read_credit(words, layout, region,
                               kLiteralEligibilityBase + output_phase) <=
                       kEligibilityLifetime &&
                   read_credit(words, layout, region,
                               kLiteralEligibilityBase + output_phase) != 0u) {
          const std::uint32_t agreement_field = kAgreementBase + output_phase;
          write_form(words, layout, region, agreement_field, 0u);
          ++receipt->negative_updates;
        }
      }
    }

    const std::uint32_t input_count =
        inputs.input_count < kMaxInputPhases ? inputs.input_count : kMaxInputPhases;
    const SiteWord matching = matching_input_mask(
        words, layout, region, input_count, observed);
    const SiteWord prior_candidate =
        read_form(words, layout, region, kCandidateBase + output_phase);
    const SiteWord seen =
        read_form(words, layout, region, kSeenBase + output_phase);
    SiteWord candidate = seen == 0u ? matching : prior_candidate & matching;
    for (std::uint32_t input_phase = 0u; input_phase < input_count; ++input_phase) {
      const std::uint32_t diagnostic_bit =
          output_phase * kMaxInputPhases + input_phase;
      const SiteWord eligibility_age = read_credit(
          words, layout, region, kEligibilityBase + output_phase);
      const bool eligible = eligibility_age != 0u &&
                            eligibility_age <= kEligibilityLifetime;
      const bool matching =
          static_cast<std::uint8_t>(read_credit(words, layout, region,
                                                kInputBase + input_phase)) == observed;
      if (eligible) {
        receipt->last_eligible_routes += 1u;
        if (diagnostic_bit < 32u)
          receipt->last_eligible_mask |= 1u << diagnostic_bit;
      }
      if (matching) {
        ++receipt->last_matching_routes;
        if (diagnostic_bit < 32u)
          receipt->last_match_mask |= 1u << diagnostic_bit;
      }
      if (!eligible || !matching)
        continue;
    }
    const SiteWord literal_eligibility_age = read_credit(
        words, layout, region, kLiteralEligibilityBase + output_phase);
    if (literal_eligibility_age != 0u &&
        literal_eligibility_age <= kEligibilityLifetime) {
      ++receipt->last_eligible_routes;
      const std::uint32_t diagnostic_bit = kMapFields + output_phase;
      if (diagnostic_bit < 32u)
        receipt->last_eligible_mask |= 1u << diagnostic_bit;
    }
    const SiteWord mapped_eligibility_age = read_credit(
        words, layout, region, kEligibilityBase + output_phase);
    if (mapped_eligibility_age != 0u &&
        mapped_eligibility_age <= kEligibilityLifetime) {
      receipt->last_eligible_routes += kMaxInputPhases - 1u;
    }
    if (matching != 0u) {
      for (std::uint32_t input_phase = 0u;
           input_phase < input_count; ++input_phase) {
        if ((matching & (1u << input_phase)) == 0u)
          continue;
        const std::uint32_t diagnostic_bit =
            output_phase * kMaxInputPhases + input_phase;
        if (diagnostic_bit < 32u)
          receipt->last_match_mask |= 1u << diagnostic_bit;
        ++receipt->last_matching_routes;
      }
    }
    const bool literal_eligible = literal_eligibility_age != 0u &&
                                  literal_eligibility_age <=
                                      kEligibilityLifetime;
    if (candidate != prior_candidate) {
      write_form(words, layout, region, kCandidateBase + output_phase,
                 candidate);
      ++receipt->positive_updates;
    }
    // The first eligible observation retains both live explanations: an input
    // route and a literal byte. Generation still prefers a unique route. If
    // later examples collapse a coincidental route, the literal survives only
    // when those examples agree byte-for-byte; contradiction still zeros its
    // agreement below instead of granting the newest teacher authority.
    if (seen == 0u && literal_eligible &&
        (!predicted || predicted_value != observed)) {
      const std::uint32_t prototype_field = kPrototypeBase + output_phase;
      const std::uint32_t agreement_field = kAgreementBase + output_phase;
      const RawByteRails prior_prototype{
          words[layout.form.rails[form_index(region, prototype_field, 0u)]],
          words[layout.form.rails[form_index(region, prototype_field, 1u)]]};
      const RawByteDecode prior_decoded =
          decode_raw_byte_carriers(prior_prototype);
      const SiteWord prior_agreement =
          read_form(words, layout, region, agreement_field);
      if (prior_candidate != 0u || prior_agreement == 0u ||
          !prior_decoded.valid) {
        const RawByteRails prototype =
            with_raw_byte_carriers(prior_prototype, observed);
        words[layout.form.rails[form_index(region, prototype_field, 0u)]] =
            prototype.zero;
        words[layout.form.rails[form_index(region, prototype_field, 1u)]] =
            prototype.one;
        write_form(words, layout, region, agreement_field, 0xffu);
        ++receipt->positive_updates;
      }
    }
    if (seen != 0u) {
      const std::uint32_t prototype_field = kPrototypeBase + output_phase;
      const RawByteRails prototype{
          words[layout.form.rails[form_index(region, prototype_field, 0u)]],
          words[layout.form.rails[form_index(region, prototype_field, 1u)]]};
      const RawByteDecode decoded = decode_raw_byte_carriers(prototype);
      const SiteWord agreement =
          read_form(words, layout, region, kAgreementBase + output_phase);
      write_form(words, layout, region, kAgreementBase + output_phase,
                 agreement & static_cast<SiteWord>(
                                  decoded.valid ?
                                      (~(decoded.value ^ observed) & 0xffu) : 0u));
    }
    write_form(words, layout, region, kSeenBase + output_phase,
               increment_support(read_form(words, layout, region, kSeenBase + output_phase)));
  }
  if (output_count != 0u) {
    intersect_length(words, layout, region, output_count - 1u);
  }
  if (!any_eligible) ++receipt->expired_output_no_update;
  for (std::uint32_t route = 0u; route < kEligibilityFieldCount; ++route)
    write_credit(words, layout, region, kEligibilityBase + route, 0u);
  write_credit(words, layout, region, kPendingField, 0u);
}

__device__ inline std::uint32_t matter_bits(const SiteWord* words,
                                            const DeviceLayout& layout) {
  std::uint32_t total = 0u;
  for (std::uint32_t index = 0u; index < kResidentRailCount; ++index)
    total += __popc(resident_value(words, layout, index));
  const std::uint32_t journal_count =
      read_global(words, layout, kJournalCountField);
  total += (kJournalDepth - journal_count) * kFounderJournalBits;
  const std::uint32_t journal_slots =
      journal_count < kJournalDepth ? journal_count : kJournalDepth;
  for (std::uint32_t event = 0u; event < journal_slots; ++event) {
    for (std::uint32_t index = 0u; index < kResidentRailCount; index += 2u) {
      total += __popc(words[fixed_physical_slot(journal_index(event, index))]);
      total += __popc(words[fixed_physical_slot(journal_index(event, index) + 1u)]);
    }
  }
  return total;
}

__device__ inline bool step_device(SiteWord* words, const DeviceLayout& layout,
                                   DeviceInputs* inputs, Receipt* receipt,
                                   std::uint32_t* step_mode = nullptr) {
  Receipt local = receipt != nullptr ? *receipt : Receipt{};
  local.live_eligibility_routes = 0u;
  ++local.ordinary_events;
  const std::uint64_t regions =
      inputs != nullptr ? active_regions(words, *inputs) : 0u;
  const StepMode mode = required_step_mode(words, layout, inputs);
  if (step_mode != nullptr) *step_mode = mode;
  if (mode == kStepNone) {
    if (local.matter_after_bits == 0u)
      local.matter_after_bits = matter_bits(words, layout);
    local.matter_before_bits = local.matter_after_bits;
    local.journal_count = read_global(words, layout, kJournalCountField);
    if (receipt != nullptr) *receipt = local;
    return true;
  }
  if (mode == kStepEligibilityAge) {
    if (local.matter_after_bits == 0u)
      local.matter_after_bits = matter_bits(words, layout);
    local.matter_before_bits = local.matter_after_bits;
    age_eligibility(words, layout, &local);
    for (std::uint32_t region = 0u; region < kRegionCount; ++region) {
      if ((regions & (1ull << region)) == 0u) continue;
      for (std::uint32_t route = 0u; route < kEligibilityFieldCount; ++route) {
        const SiteWord age =
            read_credit(words, layout, region, kEligibilityBase + route);
        if (age != 0u && age <= kEligibilityLifetime)
          ++local.live_eligibility_routes;
      }
    }
    local.journal_count = read_global(words, layout, kJournalCountField);
    if (receipt != nullptr) *receipt = local;
    return true;
  }

  // Validate staged source chronology before opening an undo event. Invalid
  // sources are consumed here so they cannot survive into a later context.
  if (inputs != nullptr) {
    const bool input_staged = inputs->input_staged != 0u;
    const bool output_staged = inputs->output_staged != 0u ||
                               inputs->output_contact_staged != 0u;
    if (input_staged || output_staged) {
      const bool pending = active_pending(words, layout, regions);
      const bool mixed = input_staged && output_staged;
      const bool invalid = regions == 0u || mixed ||
                           (input_staged && pending) ||
                           (output_staged && !pending);
      if (invalid) {
        if (input_staged) {
          ++local.input_rejected;
          ++local.staged_rejected;
        }
        if (output_staged) {
          ++local.output_rejected;
          ++local.staged_rejected;
          ++local.contact_rejected;
          ++local.contact_no_update;
        }
        inputs->input_staged = 0u;
        inputs->output_staged = 0u;
        inputs->output_contact_staged = 0u;
        inputs->output_contact_end = 0u;
        local.matter_before_bits = matter_bits(words, layout);
        local.matter_after_bits = local.matter_before_bits;
        local.journal_count = read_global(words, layout, kJournalCountField);
        if (receipt != nullptr) *receipt = local;
        return true;
      }
    }
  }
  local.matter_before_bits = matter_bits(words, layout);
  if (mode == kStepJournaled &&
      (!journal_available(words, layout) || !begin_journal(words, layout))) {
    if (step_mode != nullptr) *step_mode = kStepNone;
    return false;
  }
  age_eligibility(words, layout, &local);
  if (inputs != nullptr) {
    const std::uint32_t active_count = __popcll(regions);
    if (regions != 0u && inputs->input_staged != 0u) {
      ++local.input_staged;
      std::uint32_t rank = 0u;
      for (std::uint32_t region = 0u; region < kRegionCount; ++region) {
        if ((regions & (1ull << region)) == 0u)
          continue;
        const std::uint32_t phase_mask = acquire_phase_disposition(
            words, layout, region,
            active_count, rank, inputs->allow_single_region_complete != 0u);
        if (phase_mask != 0u)
          stage_prediction(words, layout, region, phase_mask, *inputs, &local);
        ++rank;
      }
      inputs->input_staged = 0u;
    }
    if (regions != 0u && inputs->output_contact_staged != 0u) {
      ++local.output_staged;
      ++local.contact_events;
      if (inputs->output_contact_end != 0u)
        ++local.contact_end_events;
      std::uint32_t rank = 0u;
      for (std::uint32_t region = 0u; region < kRegionCount; ++region) {
        if ((regions & (1ull << region)) == 0u)
          continue;
        const std::uint32_t phase_mask =
            read_credit(words, layout, region, kPendingField) != 0u
                ? acquire_phase_disposition(
                      words, layout, region, active_count, rank,
                      inputs->allow_single_region_complete != 0u)
                : 0u;
        learn_observation_contact(words, layout, region, phase_mask, *inputs,
                                  &local);
        ++rank;
      }
      inputs->output_contact_staged = 0u;
      inputs->output_contact_end = 0u;
    } else if (regions != 0u && inputs->output_staged != 0u) {
      ++local.output_staged;
      std::uint32_t rank = 0u;
      for (std::uint32_t region = 0u; region < kRegionCount; ++region) {
        if ((regions & (1ull << region)) == 0u)
          continue;
        const std::uint32_t phase_mask = acquire_phase_disposition(
            words, layout, region,
            active_count, rank, inputs->allow_single_region_complete != 0u);
        if (phase_mask != 0u)
          learn_observation(words, layout, region, phase_mask, *inputs, &local);
        ++rank;
      }
      inputs->output_staged = 0u;
    }
  }
  for (std::uint32_t region = 0u; region < kRegionCount; ++region) {
    if ((regions & (1ull << region)) == 0u)
      continue;
    for (std::uint32_t route = 0u; route < kEligibilityFieldCount; ++route) {
      const SiteWord age =
          read_credit(words, layout, region, kEligibilityBase + route);
      if (age != 0u && age <= kEligibilityLifetime)
        ++local.live_eligibility_routes;
    }
  }
  local.journal_count = read_global(words, layout, 1u);
  local.matter_after_bits = matter_bits(words, layout);
  if (receipt != nullptr) *receipt = local;
  return true;
}

__device__ inline void inverse_step_device(SiteWord* words,
                                           const DeviceLayout& layout) {
  const SiteWord count = read_global(words, layout, kJournalCountField);
  const SiteWord cursor = read_global(words, layout, kJournalCursorField);
  if (count == 0u || count > kJournalDepth || cursor != count) return;
  const std::uint32_t event = count - 1u;
  for (std::uint32_t index = 0u; index < kResidentRailCount; index += 2u) {
    const std::uint32_t journal = journal_index(event, index);
    const SiteWord first_value = words[fixed_physical_slot(journal)];
    const SiteWord second_value = words[fixed_physical_slot(journal + 1u)];
    if (index < kFormRailCount) {
      words[layout.form.rails[index]] = first_value;
      words[layout.form.rails[index + 1u]] = second_value;
    } else {
      words[layout.rails[index - kFormRailCount]] = first_value;
      words[layout.rails[index - kFormRailCount + 1u]] = second_value;
    }
    SiteWord journal_first = 0u;
    SiteWord journal_second = 0u;
    resident_founder_pair(index, &journal_first, &journal_second);
    words[fixed_physical_slot(journal)] = journal_first;
    words[fixed_physical_slot(journal + 1u)] = journal_second;
  }
  // The restored resident image already contains the exact pre-step cursor and
  // count. Recomputing either value would overwrite the journaled state.
}

}  // namespace substrate::bcc32::grown_form_credit_factor
