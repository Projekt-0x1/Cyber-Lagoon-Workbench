#pragma once

#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "hardware_native/bcc32_aperture_geometry.cuh"
#include "hardware_native/bcc32_grown_instance_basin.cuh"
#include "hardware_native/bcc32_law.cuh"
#include "hardware_native/bcc32_raw_byte_tape.cuh"
#include "hardware_native/bcc32_types.cuh"

namespace substrate::bcc32::grown_sparse_event_memory {

using substrate::bcc32::SiteWord;

inline constexpr std::uint32_t kFormAssemblyCount = 64u;
inline constexpr std::uint32_t kTransitionRecordCount = 64u;
inline constexpr std::uint32_t kTrajectoryAssemblyCount = 8u;
inline constexpr std::uint32_t kTrajectoryPhaseCount = 12u;
inline constexpr std::uint32_t kMotorRouteFamilyCount = 9u;
inline constexpr std::uint32_t kSituationBasinCount = 8u;
inline constexpr std::uint32_t kSituationFieldsPerBasin = 5u;
inline constexpr std::uint32_t kSituationChannelCount =
    kSituationBasinCount * kSituationFieldsPerBasin;
inline constexpr std::uint32_t kEligibilityDepth = 32u;
inline constexpr std::uint32_t kJournalDepth = 128u;
inline constexpr std::uint32_t kReservedPhysicalRailCount = 3'600'000u;
inline constexpr SiteWord kFactorMarkerValue = 0x5345564du;
inline constexpr SiteWord kLayoutVersionValue = 4u;
inline constexpr std::int32_t kRouteAmplitude = 1024;
inline constexpr std::int32_t kCreditLimit = 32767;
inline constexpr std::uint32_t kTrajectoryStartSentinel =
    kTrajectoryAssemblyCount;
inline constexpr std::uint32_t kTrajectoryNoneSentinel =
    kTrajectoryAssemblyCount + 1u;
inline constexpr std::uint32_t kTrajectoryPredecessorCount =
    kTrajectoryAssemblyCount + 1u;

static_assert(kFormAssemblyCount >= 64u);
static_assert(kTransitionRecordCount >= 64u);
static_assert(kTrajectoryAssemblyCount > 0u);
static_assert(kTrajectoryPhaseCount > 1u);
static_assert(kSituationChannelCount == 40u);

enum GlobalField : std::uint32_t {
  kFactorMarker = 0u,
  kLayoutVersion,
  kEventId,
  kJournalCount,
  kPendingValid,
  kPendingPredecessor,
  kPendingSuccessor,
  kPredictionCount,
  kObservationCount,
  kSupportCount,
  kCounterevidenceCount,
  kPredictionId,
  kCreditHead,
  kCreditCount,
  kExpiredCount,
  kFormCount,
  kTransitionCount,
  kTrajectoryCount,
  kActiveTrajectory,
  kActiveTrajectoryPhase,
  kAcquisitionLength,
  kPreviousTrajectory,
  kPreviousObservedTrajectory,
  kGlobalFieldCount,
};

inline constexpr std::uint32_t kGlobalBase = 0u;
inline constexpr std::uint32_t kFormActiveBase =
    kGlobalBase + kGlobalFieldCount * 2u;
inline constexpr std::uint32_t kFormRouteBase =
    kFormActiveBase + kFormAssemblyCount * 2u;
inline constexpr std::uint32_t kFormRouteRailCount =
    kFormAssemblyCount * kMotorRouteFamilyCount * 2u;
inline constexpr std::uint32_t kTransitionMetaBase =
    kFormRouteBase + kFormRouteRailCount;
inline constexpr std::uint32_t kTransitionMetaFieldCount = 7u;
inline constexpr std::uint32_t kTransitionMetaRailCount =
    kTransitionRecordCount * kTransitionMetaFieldCount * 2u;
inline constexpr std::uint32_t kTransitionValueBase =
    kTransitionMetaBase + kTransitionMetaRailCount;
inline constexpr std::uint32_t kTransitionValueRailCount =
    kTransitionRecordCount * kSituationChannelCount * 2u;
inline constexpr std::uint32_t kTransitionAdmitBase =
    kTransitionValueBase + kTransitionValueRailCount;
inline constexpr std::uint32_t kTransitionAdmitRailCount =
    kTransitionRecordCount * kSituationChannelCount * 2u;
inline constexpr std::uint32_t kEligibilityBase =
    kTransitionAdmitBase + kTransitionAdmitRailCount;
inline constexpr std::uint32_t kEligibilityFieldCount = 8u;
inline constexpr std::uint32_t kEligibilityRailCount =
    kEligibilityDepth * kEligibilityFieldCount * 2u;
inline constexpr std::uint32_t kTrajectoryMetaBase =
    kEligibilityBase + kEligibilityRailCount;
inline constexpr std::uint32_t kTrajectoryMetaFieldCount = 3u;
inline constexpr std::uint32_t kTrajectoryMetaRailCount =
    kTrajectoryAssemblyCount * kTrajectoryMetaFieldCount * 2u;
inline constexpr std::uint32_t kTrajectoryPhaseBase =
    kTrajectoryMetaBase + kTrajectoryMetaRailCount;
inline constexpr std::uint32_t kTrajectoryPhaseRailCount =
    kTrajectoryAssemblyCount * kTrajectoryPhaseCount * 2u;
inline constexpr std::uint32_t kTrajectoryValueBase =
    kTrajectoryPhaseBase + kTrajectoryPhaseRailCount;
inline constexpr std::uint32_t kTrajectoryValueRailCount =
    kTrajectoryAssemblyCount * kSituationChannelCount * 2u;
inline constexpr std::uint32_t kTrajectoryAdmitBase =
    kTrajectoryValueBase + kTrajectoryValueRailCount;
inline constexpr std::uint32_t kTrajectoryAdmitRailCount =
    kTrajectoryAssemblyCount * kSituationChannelCount * 2u;
inline constexpr std::uint32_t kTrajectoryAdjacencyBase =
    kTrajectoryAdmitBase + kTrajectoryAdmitRailCount;
inline constexpr std::uint32_t kTrajectoryAdjacencyRailCount =
    kTrajectoryPredecessorCount * kTrajectoryAssemblyCount * 2u;
inline constexpr std::uint32_t kAcquisitionPhaseBase =
    kTrajectoryAdjacencyBase + kTrajectoryAdjacencyRailCount;
inline constexpr std::uint32_t kAcquisitionPhaseRailCount =
    kTrajectoryPhaseCount * 2u;
inline constexpr std::uint32_t kMutableRailCount =
    kAcquisitionPhaseBase + kAcquisitionPhaseRailCount;

inline constexpr std::uint32_t kJournalMetaRailCount = 2u;
inline constexpr std::uint32_t kJournalMotorRailCount = 4u;
inline constexpr std::uint32_t kJournalRecordRailCount =
    kJournalMetaRailCount + kMutableRailCount + kJournalMotorRailCount;
inline constexpr std::uint32_t kJournalBase = kMutableRailCount;
inline constexpr std::uint32_t kPhysicalRailCount =
    kMutableRailCount + kJournalDepth * kJournalRecordRailCount;

static_assert(kPhysicalRailCount < kReservedPhysicalRailCount);

enum StepMode : std::uint32_t {
  kStepNone = 0u,
  kStepPredicted = 1u,
  kStepObserved = 2u,
  kStepAged = 3u,
};

enum TransitionMetaField : std::uint32_t {
  kTransitionPredecessor = 0u,
  kTransitionSuccessor,
  kTransitionSupport,
  kTransitionCounterevidence,
  kTransitionAdmittedCount,
  kTransitionCredit,
  kTransitionActive,
};

enum EligibilityField : std::uint32_t {
  kEligibilityEvent = 0u,
  kEligibilityPredecessor,
  kEligibilityCandidateLow,
  kEligibilityCandidateHigh,
  kEligibilitySelectedRecord,
  kEligibilityStatus,
  kEligibilityAge,
  kEligibilityObservedForm,
};

enum TrajectoryMetaField : std::uint32_t {
  kTrajectoryActive = 0u,
  kTrajectoryLength,
  kTrajectoryAdmittedCount,
};

struct PhysicalOffset {
  std::int32_t x = 0;
  std::int32_t y = 0;
  std::int32_t z = 0;
};

struct DeviceLayout {
  std::uint64_t raw_sensory_zero_slot = 0u;
  std::uint64_t raw_sensory_one_slot = 0u;
  std::uint64_t raw_motor_zero_slot = 0u;
  std::uint64_t raw_motor_one_slot = 0u;
  std::uint64_t situation_slots[kSituationChannelCount]{};
  std::uint64_t situation_active_slots[kSituationBasinCount]{};
  std::uint64_t situation_missing_age_slots[kSituationBasinCount]{};
};

inline DeviceLayout connect_resident_layout(
    std::uint64_t raw_sensory_zero_slot,
    std::uint64_t raw_sensory_one_slot,
    std::uint64_t raw_motor_zero_slot,
    std::uint64_t raw_motor_one_slot,
    const grown_instance_basin_factor::DeviceLayout& situation) {
  DeviceLayout layout{};
  layout.raw_sensory_zero_slot = raw_sensory_zero_slot;
  layout.raw_sensory_one_slot = raw_sensory_one_slot;
  layout.raw_motor_zero_slot = raw_motor_zero_slot;
  layout.raw_motor_one_slot = raw_motor_one_slot;
  constexpr std::uint32_t fields[kSituationFieldsPerBasin]{
      grown_instance_basin_factor::kAppearance,
      grown_instance_basin_factor::kCurrentSite,
      grown_instance_basin_factor::kPredictedSite,
      grown_instance_basin_factor::kConfidence,
      grown_instance_basin_factor::kSupport};
  for (std::uint32_t basin = 0u; basin < kSituationBasinCount; ++basin) {
    layout.situation_active_slots[basin] = situation.rails[
        grown_instance_basin_factor::resident_index(
            grown_instance_basin_factor::basin_field(
                basin, grown_instance_basin_factor::kActive))];
    layout.situation_missing_age_slots[basin] = situation.rails[
        grown_instance_basin_factor::resident_index(
            grown_instance_basin_factor::basin_field(
                basin, grown_instance_basin_factor::kMissingAge))];
    for (std::uint32_t field = 0u; field < kSituationFieldsPerBasin; ++field) {
      const std::uint32_t source = grown_instance_basin_factor::resident_index(
          grown_instance_basin_factor::basin_field(basin, fields[field]));
      layout.situation_slots[basin * kSituationFieldsPerBasin + field] =
          situation.rails[source];
    }
  }
  return layout;
}

__device__ inline bool situation_basin_current(const SiteWord* words,
                                                const DeviceLayout& layout,
                                                std::uint32_t basin) {
  return words[layout.situation_active_slots[basin]] != 0u &&
         words[layout.situation_missing_age_slots[basin]] == 0u;
}

// These fields are timing only. Raw content remains on the reciprocal rails.
struct DeviceInputs {
  std::uint32_t predict_staged = 0u;
  std::uint32_t observe_staged = 0u;
};

struct DeviceScratch {
  std::uint32_t reserved = 0u;
  std::uint32_t accepted = 0u;
  std::uint32_t journal_slot = 0u;
  std::uint32_t event_id = 0u;
  std::uint32_t current_form = kFormAssemblyCount;
  std::uint32_t observed_form = kFormAssemblyCount;
  std::uint32_t predicted_form = kFormAssemblyCount;
  std::uint32_t transition_record = kTransitionRecordCount;
  std::uint32_t candidate_low = 0u;
  std::uint32_t candidate_high = 0u;
};

struct Receipt {
  std::uint32_t step_mode = kStepNone;
  std::uint32_t event_id = 0u;
  std::uint32_t current_form = kFormAssemblyCount;
  std::uint32_t observed_form = kFormAssemblyCount;
  std::uint32_t predicted_form = kFormAssemblyCount;
  std::uint32_t predecessor_form = kFormAssemblyCount;
  std::uint32_t transition_record = kTransitionRecordCount;
  std::uint32_t recruited_form = 0u;
  std::uint32_t recruited_transition = 0u;
  std::uint32_t recruited_trajectory = 0u;
  std::uint32_t selected_trajectory = kTrajectoryAssemblyCount;
  std::uint32_t trajectory_phase = kTrajectoryPhaseCount;
  std::uint32_t trajectory_abstained = 0u;
  std::uint32_t abstained = 0u;
  std::uint32_t support_updates = 0u;
  std::uint32_t counterevidence_updates = 0u;
  std::uint32_t positive_credit_updates = 0u;
  std::uint32_t negative_credit_updates = 0u;
  std::uint32_t expired_events = 0u;
  std::uint32_t aged_events = 0u;
  std::uint32_t credited_prediction = 0u;
  std::uint32_t journal_exhausted = 0u;
  std::uint32_t invalid_contact = 0u;
  std::int32_t transition_credit_delta[kTransitionRecordCount]{};
};

struct InverseScratch {
  std::uint32_t valid = 0u;
  std::uint32_t error = 0u;
  std::uint32_t journal_slot = 0u;
  std::uint32_t event_slot = 0u;
};

struct ValidationReceipt {
  std::uint32_t marker_valid = 0u;
  std::uint32_t version_valid = 0u;
  std::uint32_t journal_valid = 0u;
  std::uint32_t form_valid = 0u;
  std::uint32_t transition_valid = 0u;
  std::uint32_t trajectory_valid = 0u;
  std::uint32_t acquisition_valid = 0u;
  std::uint32_t eligibility_valid = 0u;
  std::uint32_t invalid_pairs = 0u;
  std::uint32_t invalid_journal_events = 0u;
  std::uint32_t invalid_eligibility = 0u;
};

struct LesionReceipt {
  std::uint32_t record = kTransitionRecordCount;
  std::uint32_t prior_active = 0u;
  std::uint32_t changed_bits = 0u;
  std::uint32_t matter_before = 0u;
  std::uint32_t matter_after = 0u;
  std::uint32_t valid = 0u;
};

struct TrajectoryLesionReceipt {
  std::uint32_t trajectory = kTrajectoryAssemblyCount;
  std::uint32_t prior_active = 0u;
  std::uint32_t changed_bits = 0u;
  std::uint32_t matter_before = 0u;
  std::uint32_t matter_after = 0u;
  std::uint32_t valid = 0u;
};

struct TrajectoryAdjacencyLesionReceipt {
  std::uint32_t predecessor = kTrajectoryPredecessorCount;
  std::uint32_t successor = kTrajectoryAssemblyCount;
  std::uint32_t prior_support = 0u;
  std::uint32_t changed_bits = 0u;
  std::uint32_t matter_before = 0u;
  std::uint32_t matter_after = 0u;
  std::uint32_t valid = 0u;
};

// Region is a dense 80 x 220 x ceil(kPhysicalRailCount/17600) box, translated
// (not resized) to keep every site inside the kSpatialMacroClosureRadius=26
// clearance window on all three axes.
__host__ __device__ inline PhysicalOffset physical_offset(std::uint32_t index) {
  return {64 + static_cast<std::int32_t>(index % 80u),
          -224 + static_cast<std::int32_t>((index / 80u) % 220u),
          -220 + static_cast<std::int32_t>(index / 17600u)};
}

__host__ __device__ inline std::uint64_t fixed_physical_slot(std::uint32_t index) {
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
      static_cast<std::uint64_t>(gx / chunk_edge) * edge_chunks * edge_chunks +
      static_cast<std::uint64_t>(gy / chunk_edge) * edge_chunks +
      static_cast<std::uint64_t>(gz / chunk_edge);
  const std::uint64_t local =
      ((static_cast<std::uint64_t>(gx % chunk_edge) * chunk_edge +
        static_cast<std::uint64_t>(gy % chunk_edge)) *
       chunk_edge) +
      static_cast<std::uint64_t>(gz % chunk_edge);
  return chunk * chunk_sites + local;
}

__host__ __device__ inline std::uint32_t pair_index(std::uint32_t base,
                                                    std::uint32_t item,
                                                    std::uint32_t polarity) {
  return base + item * 2u + polarity;
}

__host__ __device__ inline std::uint32_t form_active_index(std::uint32_t form,
                                                           std::uint32_t polarity) {
  return pair_index(kFormActiveBase, form, polarity);
}

__host__ __device__ inline std::uint32_t form_route_index(std::uint32_t form,
                                                          std::uint32_t family,
                                                          std::uint32_t polarity) {
  return pair_index(kFormRouteBase, form * kMotorRouteFamilyCount + family, polarity);
}

__host__ __device__ inline std::uint32_t transition_meta_index(
    std::uint32_t record, std::uint32_t field, std::uint32_t polarity) {
  return pair_index(kTransitionMetaBase, record * kTransitionMetaFieldCount + field, polarity);
}

__host__ __device__ inline std::uint32_t transition_value_index(
    std::uint32_t record, std::uint32_t channel, std::uint32_t polarity) {
  return pair_index(kTransitionValueBase, record * kSituationChannelCount + channel, polarity);
}

__host__ __device__ inline std::uint32_t transition_admit_index(
    std::uint32_t record, std::uint32_t channel, std::uint32_t polarity) {
  return pair_index(kTransitionAdmitBase, record * kSituationChannelCount + channel, polarity);
}

__host__ __device__ inline std::uint32_t eligibility_index(
    std::uint32_t slot, std::uint32_t field, std::uint32_t polarity) {
  return pair_index(kEligibilityBase, slot * kEligibilityFieldCount + field,
                    polarity);
}

__host__ __device__ inline std::uint32_t trajectory_meta_index(
    std::uint32_t trajectory, std::uint32_t field, std::uint32_t polarity) {
  return pair_index(kTrajectoryMetaBase,
                    trajectory * kTrajectoryMetaFieldCount + field, polarity);
}

__host__ __device__ inline std::uint32_t trajectory_phase_index(
    std::uint32_t trajectory, std::uint32_t phase, std::uint32_t polarity) {
  return pair_index(kTrajectoryPhaseBase,
                    trajectory * kTrajectoryPhaseCount + phase, polarity);
}

__host__ __device__ inline std::uint32_t trajectory_value_index(
    std::uint32_t trajectory, std::uint32_t channel, std::uint32_t polarity) {
  return pair_index(kTrajectoryValueBase,
                    trajectory * kSituationChannelCount + channel, polarity);
}

__host__ __device__ inline std::uint32_t trajectory_admit_index(
    std::uint32_t trajectory, std::uint32_t channel, std::uint32_t polarity) {
  return pair_index(kTrajectoryAdmitBase,
                    trajectory * kSituationChannelCount + channel, polarity);
}

__host__ __device__ inline std::uint32_t trajectory_adjacency_index(
    std::uint32_t predecessor, std::uint32_t successor,
    std::uint32_t polarity) {
  return pair_index(
      kTrajectoryAdjacencyBase,
      predecessor * kTrajectoryAssemblyCount + successor, polarity);
}

__host__ __device__ inline std::uint32_t acquisition_phase_index(
    std::uint32_t phase, std::uint32_t polarity) {
  return pair_index(kAcquisitionPhaseBase, phase, polarity);
}

__host__ __device__ inline std::uint32_t journal_index(std::uint32_t slot,
                                                       std::uint32_t offset) {
  return kJournalBase + slot * kJournalRecordRailCount + offset;
}

__host__ __device__ inline std::uint32_t mix32(std::uint32_t value) {
  value ^= value >> 16u;
  value *= 0x7feb352du;
  value ^= value >> 15u;
  value *= 0x846ca68bu;
  return value ^ (value >> 16u);
}

// github #1323: this used to open with a self-call on the odd branch, an edge
// nvlink cannot rule out. The recursion was depth ONE by construction -- it was
// taken only for an odd index and passed index - 1, which is even and therefore
// cannot take it again -- but nvlink reasons over the call graph, not over the
// parity, so it reported STACK:UNKNOWN (its 0xffffffff "not statically
// boundable" sentinel) for every kernel reaching this function. That is the
// ABSENCE of a bound, not a large one, so no stack reservation could be proven
// sufficient for grown_sparse_event_memory::clear_inverse_journal_kernel.
//
// Splitting the parity fold out of the recursive position is behaviour-identical
// BY CONSTRUCTION, not by measurement: the odd branch called this exact body
// with this exact argument, and the even path is untouched.
__host__ __device__ inline SiteWord founder_value_even(std::uint32_t index) {
  if (index == pair_index(kGlobalBase, kFactorMarker, 0u))
    return kFactorMarkerValue;
  if (index == pair_index(kGlobalBase, kLayoutVersion, 0u))
    return kLayoutVersionValue;
  if (index == pair_index(kGlobalBase, kActiveTrajectory, 0u))
    return kTrajectoryAssemblyCount;
  if (index == pair_index(kGlobalBase, kPreviousTrajectory, 0u))
    return kTrajectoryNoneSentinel;
  if (index == pair_index(kGlobalBase, kPreviousObservedTrajectory, 0u))
    return kTrajectoryNoneSentinel;
  return 0u;
}

__host__ __device__ inline SiteWord founder_value(std::uint32_t index) {
  return (index & 1u) != 0u ? static_cast<SiteWord>(~founder_value_even(index - 1u))
                            : founder_value_even(index);
}

template <typename Entry>
inline std::vector<Entry> founder_entries() {
  std::vector<Entry> entries;
  entries.reserve(kPhysicalRailCount);
  for (std::uint32_t index = 0u; index < kPhysicalRailCount; ++index)
    entries.push_back({fixed_physical_slot(index), founder_value(index)});
  return entries;
}

__device__ inline SiteWord read_resident(const SiteWord* words, std::uint32_t index) {
  return words[fixed_physical_slot(index)];
}

__device__ inline void write_resident(SiteWord* words, std::uint32_t index, SiteWord value) {
  words[fixed_physical_slot(index)] = value;
  words[fixed_physical_slot(index + 1u)] = ~value;
}

__device__ inline std::uint32_t read_unsigned(const SiteWord* words, std::uint32_t index) {
  return static_cast<std::uint32_t>(read_resident(words, index));
}

__device__ inline std::int32_t read_signed(const SiteWord* words, std::uint32_t index) {
  return static_cast<std::int32_t>(read_resident(words, index));
}

__device__ inline std::int32_t clamp_credit(std::int64_t value) {
  if (value > kCreditLimit)
    return kCreditLimit;
  if (value < -kCreditLimit)
    return -kCreditLimit;
  return static_cast<std::int32_t>(value);
}

__device__ inline std::uint32_t route_sign(const RawByteDecode raw, std::uint32_t family) {
  if (!raw.valid)
    return 0u;
  if (family == 0u)
    return 1u;
  return ((raw.value >> (family - 1u)) & 1u) != 0u ? 1u : 0u;
}

__device__ inline std::int32_t route_value(const RawByteDecode raw, std::uint32_t family) {
  return route_sign(raw, family) != 0u ? kRouteAmplitude : -kRouteAmplitude;
}

__device__ inline RawByteDecode current_prediction_raw(const SiteWord* words,
                                                       const DeviceLayout& layout) {
  const RawByteDecode sensory = decode_raw_byte_carriers(
      {words[layout.raw_sensory_zero_slot], words[layout.raw_sensory_one_slot]});
  if (sensory.valid)
    return sensory;
  return decode_raw_byte_carriers(
      {words[layout.raw_motor_zero_slot], words[layout.raw_motor_one_slot]});
}

__device__ inline RawByteDecode current_observation_raw(const SiteWord* words,
                                                        const DeviceLayout& layout) {
  return decode_raw_byte_carriers(
      {words[layout.raw_sensory_zero_slot], words[layout.raw_sensory_one_slot]});
}

__device__ inline std::uint32_t find_form(const SiteWord* words, const RawByteDecode raw) {
  if (!raw.valid)
    return kFormAssemblyCount;
  std::uint32_t best = kFormAssemblyCount;
  std::uint32_t best_score = 0u;
  for (std::uint32_t form = 0u; form < kFormAssemblyCount; ++form) {
    if (read_unsigned(words, form_active_index(form, 0u)) == 0u)
      continue;
    std::uint32_t score = 0u;
    for (std::uint32_t family = 0u; family < kMotorRouteFamilyCount; ++family) {
      const std::int32_t expected = route_value(raw, family);
      const std::int32_t stored = read_signed(words, form_route_index(form, family, 0u));
      score += (stored == expected) ? 1u : 0u;
    }
    if (score > best_score) {
      best = form;
      best_score = score;
    }
    if (score == kMotorRouteFamilyCount)
      return form;
  }
  return best;
}

__device__ inline std::uint32_t free_form(const SiteWord* words) {
  for (std::uint32_t form = 0u; form < kFormAssemblyCount; ++form)
    if (read_unsigned(words, form_active_index(form, 0u)) == 0u)
      return form;
  return kFormAssemblyCount;
}

__device__ inline std::uint32_t find_or_recruit_form(SiteWord* words,
                                                     const RawByteDecode raw,
                                                     Receipt* receipt) {
  const std::uint32_t found = find_form(words, raw);
  if (found < kFormAssemblyCount &&
      read_unsigned(words, form_active_index(found, 0u)) != 0u) {
    bool exact = true;
    for (std::uint32_t family = 0u; family < kMotorRouteFamilyCount; ++family)
      exact = exact && read_signed(words, form_route_index(found, family, 0u)) ==
                          route_value(raw, family);
    if (exact || free_form(words) == kFormAssemblyCount)
      return found;
  }
  const std::uint32_t form = free_form(words);
  if (form == kFormAssemblyCount)
    return found;
  write_resident(words, form_active_index(form, 0u), 1u);
  for (std::uint32_t family = 0u; family < kMotorRouteFamilyCount; ++family)
    write_resident(words, form_route_index(form, family, 0u), route_value(raw, family));
  write_resident(words, pair_index(kGlobalBase, kFormCount, 0u),
                 read_unsigned(words, pair_index(kGlobalBase, kFormCount, 0u)) + 1u);
  if (receipt != nullptr)
    receipt->recruited_form = 1u;
  return form;
}

__device__ inline RawByteDecode decode_form_raw(const SiteWord* words,
                                                std::uint32_t form) {
  if (form >= kFormAssemblyCount ||
      read_unsigned(words, form_active_index(form, 0u)) == 0u ||
      read_signed(words, form_route_index(form, 0u, 0u)) != kRouteAmplitude)
    return {};
  std::uint8_t bits = 0u;
  for (std::uint32_t bit = 0u; bit < 8u; ++bit) {
    const std::int32_t route =
        read_signed(words, form_route_index(form, bit + 1u, 0u));
    if (route != kRouteAmplitude && route != -kRouteAmplitude)
      return {};
    if (route == kRouteAmplitude)
      bits = static_cast<std::uint8_t>(bits | (1u << bit));
  }
  return {bits, true};
}

__device__ inline bool is_surface_separator(const RawByteDecode raw) {
  if (!raw.valid)
    return false;
  const std::uint32_t value = raw.value;
  return (value >= 0x09u && value <= 0x0du) || value == 0x20u ||
         (value >= 0x21u && value <= 0x2fu) ||
         (value >= 0x3au && value <= 0x40u) ||
         (value >= 0x5bu && value <= 0x60u) ||
         (value >= 0x7bu && value <= 0x7eu);
}

__device__ inline void publish_form(SiteWord* words, const DeviceLayout& layout,
                                    std::uint32_t form) {
  RawByteRails motor{};
  const RawByteDecode raw = decode_form_raw(words, form);
  if (raw.valid)
    motor = with_raw_byte_carriers(motor, raw.value);
  words[layout.raw_motor_zero_slot] = motor.zero;
  words[layout.raw_motor_one_slot] = motor.one;
}

__device__ inline std::uint32_t find_transition(const SiteWord* words,
                                                std::uint32_t predecessor,
                                                std::uint32_t successor) {
  for (std::uint32_t record = 0u; record < kTransitionRecordCount; ++record) {
    if (read_unsigned(words, transition_meta_index(record, kTransitionActive, 0u)) == 0u)
      continue;
    if (read_unsigned(words, transition_meta_index(record, kTransitionPredecessor, 0u)) ==
            predecessor &&
        read_unsigned(words, transition_meta_index(record, kTransitionSuccessor, 0u)) ==
            successor)
      return record;
  }
  return kTransitionRecordCount;
}

__device__ inline std::uint32_t free_transition(const SiteWord* words) {
  for (std::uint32_t record = 0u; record < kTransitionRecordCount; ++record)
    if (read_unsigned(words, transition_meta_index(record, kTransitionActive, 0u)) == 0u)
      return record;
  return kTransitionRecordCount;
}

__device__ inline bool transition_basin_identity_matches(
    const SiteWord* words, const DeviceLayout& layout, std::uint32_t record,
    std::uint32_t basin) {
  const std::uint32_t appearance_channel = basin * kSituationFieldsPerBasin;
  return read_resident(
             words,
             transition_value_index(record, appearance_channel, 0u)) ==
         words[layout.situation_slots[appearance_channel]];
}

__device__ inline void snapshot_situation(SiteWord* words, const DeviceLayout& layout,
                                          std::uint32_t record) {
  std::uint32_t admitted = 0u;
  for (std::uint32_t channel = 0u; channel < kSituationChannelCount; ++channel) {
    const std::uint32_t basin = channel / kSituationFieldsPerBasin;
    const std::uint32_t current =
        situation_basin_current(words, layout, basin) ? 1u : 0u;
    write_resident(words, transition_value_index(record, channel, 0u),
                   words[layout.situation_slots[channel]]);
    write_resident(words, transition_admit_index(record, channel, 0u), current);
    admitted += current;
  }
  write_resident(words,
                 transition_meta_index(record, kTransitionAdmittedCount, 0u),
                 admitted);
}

__device__ inline std::uint32_t recruit_transition(SiteWord* words,
                                                    const DeviceLayout& layout,
                                                    std::uint32_t predecessor,
                                                    std::uint32_t successor,
                                                    Receipt* receipt) {
  std::uint32_t record = free_transition(words);
  if (record == kTransitionRecordCount)
    return record;
  write_resident(words, transition_meta_index(record, kTransitionPredecessor, 0u), predecessor);
  write_resident(words, transition_meta_index(record, kTransitionSuccessor, 0u), successor);
  write_resident(words, transition_meta_index(record, kTransitionSupport, 0u), 1u);
  write_resident(words, transition_meta_index(record, kTransitionCounterevidence, 0u), 0u);
  write_resident(words, transition_meta_index(record, kTransitionCredit, 0u), 0u);
  write_resident(words, transition_meta_index(record, kTransitionActive, 0u), 1u);
  snapshot_situation(words, layout, record);
  write_resident(words, pair_index(kGlobalBase, kTransitionCount, 0u),
                 read_unsigned(words, pair_index(kGlobalBase, kTransitionCount, 0u)) + 1u);
  if (receipt != nullptr)
    receipt->recruited_transition = 1u;
  return record;
}

__device__ inline std::uint32_t situation_score(const SiteWord* words,
                                                const DeviceLayout& layout,
                                                std::uint32_t record,
                                                std::uint32_t* admitted) {
  std::uint32_t matches = 0u;
  std::uint32_t count = 0u;
  for (std::uint32_t channel = 0u; channel < kSituationChannelCount; ++channel) {
    if (read_unsigned(words, transition_admit_index(record, channel, 0u)) == 0u)
      continue;
    ++count;
    const std::uint32_t basin = channel / kSituationFieldsPerBasin;
    matches += situation_basin_current(words, layout, basin) &&
                       transition_basin_identity_matches(words, layout, record,
                                                          basin) &&
                       words[layout.situation_slots[channel]] ==
                       read_resident(words, transition_value_index(record, channel, 0u))
                   ? 1u
                   : 0u;
  }
  if (admitted != nullptr)
    *admitted = count;
  return matches;
}

__device__ inline std::uint32_t select_successor(const SiteWord* words,
                                                  const DeviceLayout& layout,
                                                  std::uint32_t predecessor,
                                                  std::uint32_t* selected_record,
                                                  std::uint32_t* candidate_low,
                                                  std::uint32_t* candidate_high) {
  std::uint32_t best = kTransitionRecordCount;
  std::uint32_t best_matches = 0u;
  std::uint32_t best_admitted = 1u;
  std::int32_t best_credit = 0;
  std::uint32_t low = 0u;
  std::uint32_t high = 0u;
  bool tied = false;
  for (std::uint32_t record = 0u; record < kTransitionRecordCount; ++record) {
    if (read_unsigned(words, transition_meta_index(record, kTransitionActive, 0u)) == 0u ||
        read_unsigned(words, transition_meta_index(record, kTransitionPredecessor, 0u)) !=
            predecessor)
      continue;
    std::uint32_t admitted = 0u;
    const std::uint32_t matches = situation_score(words, layout, record, &admitted);
    if (admitted == 0u)
      continue;
    const std::int32_t credit =
        read_signed(words, transition_meta_index(record, kTransitionCredit, 0u));
    const std::uint64_t left = static_cast<std::uint64_t>(matches) * best_admitted;
    const std::uint64_t right = static_cast<std::uint64_t>(best_matches) * admitted;
    if (best == kTransitionRecordCount || left > right) {
      best = record;
      best_matches = matches;
      best_admitted = admitted;
      best_credit = credit;
      low = record < 32u ? (1u << record) : 0u;
      high = record >= 32u ? (1u << (record - 32u)) : 0u;
      tied = false;
    } else if (left == right) {
      if (record < 32u)
        low |= 1u << record;
      else
        high |= 1u << (record - 32u);
      if (credit > best_credit) {
        best = record;
        best_credit = credit;
        tied = false;
      } else if (credit == best_credit) {
        tied = true;
      }
    }
  }
  if (candidate_low != nullptr)
    *candidate_low = low;
  if (candidate_high != nullptr)
    *candidate_high = high;
  if (best == kTransitionRecordCount || tied) {
    if (selected_record != nullptr)
      *selected_record = kTransitionRecordCount;
    return kFormAssemblyCount;
  }
  if (selected_record != nullptr)
    *selected_record = best;
  return read_unsigned(words, transition_meta_index(best, kTransitionSuccessor, 0u));
}

__device__ inline void update_transition_observation(SiteWord* words,
                                                     const DeviceLayout& layout,
                                                     std::uint32_t record) {
  if (record >= kTransitionRecordCount)
    return;
  std::uint32_t admitted = 0u;
  for (std::uint32_t channel = 0u; channel < kSituationChannelCount; ++channel) {
    if (read_unsigned(words, transition_admit_index(record, channel, 0u)) == 0u)
      continue;
    const std::uint32_t basin = channel / kSituationFieldsPerBasin;
    ++admitted;
    if (!situation_basin_current(words, layout, basin) ||
        !transition_basin_identity_matches(words, layout, record, basin) ||
        words[layout.situation_slots[channel]] !=
        read_resident(words, transition_value_index(record, channel, 0u))) {
      write_resident(words, transition_admit_index(record, channel, 0u), 0u);
      --admitted;
    }
  }
  write_resident(words, transition_meta_index(record, kTransitionAdmittedCount, 0u), admitted);
}

__device__ inline void clear_active_trajectory(SiteWord* words) {
  write_resident(words, pair_index(kGlobalBase, kActiveTrajectory, 0u),
                 kTrajectoryAssemblyCount);
  write_resident(words, pair_index(kGlobalBase, kActiveTrajectoryPhase, 0u), 0u);
}

__device__ inline void clear_acquisition(SiteWord* words) {
  write_resident(words, pair_index(kGlobalBase, kAcquisitionLength, 0u), 0u);
  for (std::uint32_t phase = 0u; phase < kTrajectoryPhaseCount; ++phase)
    write_resident(words, acquisition_phase_index(phase, 0u), 0u);
}

__device__ inline std::uint32_t trajectory_adjacency_support(
    const SiteWord* words, std::uint32_t predecessor,
    std::uint32_t successor) {
  if (predecessor >= kTrajectoryPredecessorCount ||
      successor >= kTrajectoryAssemblyCount)
    return 0u;
  return read_unsigned(words,
                       trajectory_adjacency_index(predecessor, successor, 0u));
}

__device__ inline bool has_learned_trajectory_adjacency(
    const SiteWord* words) {
  for (std::uint32_t predecessor = 0u;
       predecessor < kTrajectoryPredecessorCount; ++predecessor) {
    for (std::uint32_t successor = 0u;
         successor < kTrajectoryAssemblyCount; ++successor) {
      if (trajectory_adjacency_support(words, predecessor, successor) != 0u)
        return true;
    }
  }
  return false;
}

__device__ inline void record_observed_trajectory(
    SiteWord* words, std::uint32_t trajectory) {
  if (trajectory >= kTrajectoryAssemblyCount)
    return;
  const std::uint32_t previous = read_unsigned(
      words, pair_index(kGlobalBase, kPreviousObservedTrajectory, 0u));
  if (previous == kTrajectoryStartSentinel ||
      previous < kTrajectoryAssemblyCount) {
    const std::uint32_t index =
        trajectory_adjacency_index(previous, trajectory, 0u);
    write_resident(words, index, read_unsigned(words, index) + 1u);
  }
  write_resident(
      words, pair_index(kGlobalBase, kPreviousObservedTrajectory, 0u),
      trajectory);
}

__device__ inline std::uint32_t free_trajectory(const SiteWord* words) {
  for (std::uint32_t trajectory = 0u; trajectory < kTrajectoryAssemblyCount;
       ++trajectory) {
    if (read_unsigned(words,
                      trajectory_meta_index(trajectory, kTrajectoryActive, 0u)) == 0u)
      return trajectory;
  }
  return kTrajectoryAssemblyCount;
}

__device__ inline std::uint32_t find_acquisition_trajectory(
    const SiteWord* words, std::uint32_t length) {
  if (length == 0u || length > kTrajectoryPhaseCount)
    return kTrajectoryAssemblyCount;
  for (std::uint32_t trajectory = 0u; trajectory < kTrajectoryAssemblyCount;
       ++trajectory) {
    if (read_unsigned(words,
                      trajectory_meta_index(trajectory, kTrajectoryActive, 0u)) == 0u ||
        read_unsigned(words,
                      trajectory_meta_index(trajectory, kTrajectoryLength, 0u)) != length)
      continue;
    bool exact = true;
    for (std::uint32_t phase = 0u; phase < length; ++phase) {
      exact = exact &&
              read_unsigned(words, trajectory_phase_index(trajectory, phase, 0u)) ==
                  read_unsigned(words, acquisition_phase_index(phase, 0u));
    }
    if (exact)
      return trajectory;
  }
  return kTrajectoryAssemblyCount;
}

__device__ inline bool trajectory_basin_identity_matches(
    const SiteWord* words, const DeviceLayout& layout,
    std::uint32_t trajectory, std::uint32_t basin) {
  const std::uint32_t appearance_channel = basin * kSituationFieldsPerBasin;
  return read_resident(
             words, trajectory_value_index(trajectory, appearance_channel, 0u)) ==
         words[layout.situation_slots[appearance_channel]];
}

__device__ inline void snapshot_trajectory_situation(
    SiteWord* words, const DeviceLayout& layout, std::uint32_t trajectory) {
  std::uint32_t admitted = 0u;
  for (std::uint32_t channel = 0u; channel < kSituationChannelCount; ++channel) {
    const std::uint32_t basin = channel / kSituationFieldsPerBasin;
    const std::uint32_t current =
        situation_basin_current(words, layout, basin) ? 1u : 0u;
    write_resident(words, trajectory_value_index(trajectory, channel, 0u),
                   words[layout.situation_slots[channel]]);
    write_resident(words, trajectory_admit_index(trajectory, channel, 0u), current);
    admitted += current;
  }
  write_resident(
      words, trajectory_meta_index(trajectory, kTrajectoryAdmittedCount, 0u),
      admitted);
}

__device__ inline void update_trajectory_observation(
    SiteWord* words, const DeviceLayout& layout, std::uint32_t trajectory) {
  std::uint32_t admitted = 0u;
  for (std::uint32_t channel = 0u; channel < kSituationChannelCount; ++channel) {
    if (read_unsigned(words,
                      trajectory_admit_index(trajectory, channel, 0u)) == 0u)
      continue;
    const std::uint32_t basin = channel / kSituationFieldsPerBasin;
    ++admitted;
    if (!situation_basin_current(words, layout, basin) ||
        !trajectory_basin_identity_matches(words, layout, trajectory, basin) ||
        words[layout.situation_slots[channel]] !=
            read_resident(words,
                          trajectory_value_index(trajectory, channel, 0u))) {
      write_resident(words, trajectory_admit_index(trajectory, channel, 0u), 0u);
      --admitted;
    }
  }
  write_resident(
      words, trajectory_meta_index(trajectory, kTrajectoryAdmittedCount, 0u),
      admitted);
}

__device__ inline void finalize_acquisition_trajectory(
    SiteWord* words, const DeviceLayout& layout, Receipt* receipt) {
  const std::uint32_t length =
      read_unsigned(words, pair_index(kGlobalBase, kAcquisitionLength, 0u));
  if (length == 0u || length > kTrajectoryPhaseCount) {
    clear_acquisition(words);
    return;
  }
  std::uint32_t trajectory = find_acquisition_trajectory(words, length);
  if (trajectory < kTrajectoryAssemblyCount) {
    update_trajectory_observation(words, layout, trajectory);
    record_observed_trajectory(words, trajectory);
    clear_acquisition(words);
    return;
  }
  trajectory = free_trajectory(words);
  if (trajectory == kTrajectoryAssemblyCount) {
    clear_acquisition(words);
    return;
  }
  write_resident(words,
                 trajectory_meta_index(trajectory, kTrajectoryActive, 0u), 1u);
  write_resident(words,
                 trajectory_meta_index(trajectory, kTrajectoryLength, 0u), length);
  for (std::uint32_t phase = 0u; phase < length; ++phase) {
    write_resident(words, trajectory_phase_index(trajectory, phase, 0u),
                   read_unsigned(words, acquisition_phase_index(phase, 0u)));
  }
  snapshot_trajectory_situation(words, layout, trajectory);
  write_resident(
      words, pair_index(kGlobalBase, kTrajectoryCount, 0u),
      read_unsigned(words, pair_index(kGlobalBase, kTrajectoryCount, 0u)) + 1u);
  if (receipt != nullptr)
    receipt->recruited_trajectory = 1u;
  record_observed_trajectory(words, trajectory);
  clear_acquisition(words);
}

__device__ inline void observe_trajectory_phase(
    SiteWord* words, const DeviceLayout& layout, std::uint32_t predecessor,
    std::uint32_t observed, Receipt* receipt) {
  const RawByteDecode predecessor_raw = decode_form_raw(words, predecessor);
  const RawByteDecode observed_raw = decode_form_raw(words, observed);
  if (!predecessor_raw.valid || !observed_raw.valid) {
    clear_acquisition(words);
    return;
  }
  const bool starts = is_surface_separator(predecessor_raw);
  if (starts)
    clear_acquisition(words);
  std::uint32_t length =
      read_unsigned(words, pair_index(kGlobalBase, kAcquisitionLength, 0u));
  if (!starts && length == 0u)
    return;
  if (length >= kTrajectoryPhaseCount) {
    clear_acquisition(words);
    return;
  }
  write_resident(words, acquisition_phase_index(length, 0u), observed);
  ++length;
  write_resident(words, pair_index(kGlobalBase, kAcquisitionLength, 0u), length);
  if (is_surface_separator(observed_raw))
    finalize_acquisition_trajectory(words, layout, receipt);
}

__device__ inline std::uint32_t trajectory_situation_score(
    const SiteWord* words, const DeviceLayout& layout,
    std::uint32_t trajectory, std::uint32_t* admitted) {
  std::uint32_t matches = 0u;
  std::uint32_t count = 0u;
  for (std::uint32_t channel = 0u; channel < kSituationChannelCount; ++channel) {
    if (read_unsigned(words,
                      trajectory_admit_index(trajectory, channel, 0u)) == 0u)
      continue;
    ++count;
    const std::uint32_t basin = channel / kSituationFieldsPerBasin;
    matches += situation_basin_current(words, layout, basin) &&
                       trajectory_basin_identity_matches(words, layout, trajectory,
                                                          basin) &&
                       words[layout.situation_slots[channel]] ==
                           read_resident(
                               words,
                               trajectory_value_index(trajectory, channel, 0u))
                   ? 1u
                   : 0u;
  }
  if (admitted != nullptr)
    *admitted = count;
  return matches;
}

__device__ inline std::uint32_t select_trajectory(
    const SiteWord* words, const DeviceLayout& layout,
    std::uint32_t first_form, bool* ambiguous) {
  std::uint32_t best = kTrajectoryAssemblyCount;
  std::uint32_t best_matches = 0u;
  std::uint32_t best_admitted = 1u;
  bool tied = false;
  for (std::uint32_t trajectory = 0u; trajectory < kTrajectoryAssemblyCount;
       ++trajectory) {
    const std::uint32_t length = read_unsigned(
        words, trajectory_meta_index(trajectory, kTrajectoryLength, 0u));
    if (read_unsigned(words,
                      trajectory_meta_index(trajectory, kTrajectoryActive, 0u)) == 0u ||
        length == 0u || length > kTrajectoryPhaseCount ||
        read_unsigned(words, trajectory_phase_index(trajectory, 0u, 0u)) !=
            first_form)
      continue;
    std::uint32_t admitted = 0u;
    const std::uint32_t matches =
        trajectory_situation_score(words, layout, trajectory, &admitted);
    if (admitted == 0u)
      continue;
    const std::uint64_t left =
        static_cast<std::uint64_t>(matches) * best_admitted;
    const std::uint64_t right =
        static_cast<std::uint64_t>(best_matches) * admitted;
    if (best == kTrajectoryAssemblyCount || left > right) {
      best = trajectory;
      best_matches = matches;
      best_admitted = admitted;
      tied = false;
    } else if (left == right) {
      tied = true;
    }
  }
  if (ambiguous != nullptr)
    *ambiguous = tied;
  return tied ? kTrajectoryAssemblyCount : best;
}

__device__ inline bool has_learned_trajectory_for_first_form(
    const SiteWord* words, std::uint32_t first_form) {
  for (std::uint32_t trajectory = 0u;
       trajectory < kTrajectoryAssemblyCount; ++trajectory) {
    const std::uint32_t length = read_unsigned(
        words, trajectory_meta_index(trajectory, kTrajectoryLength, 0u));
    if (length != 0u && length <= kTrajectoryPhaseCount &&
        read_unsigned(words,
                      trajectory_phase_index(trajectory, 0u, 0u)) == first_form)
      return true;
  }
  return false;
}

__device__ inline bool trajectories_substitutable(
    const SiteWord* words, std::uint32_t left, std::uint32_t right) {
  if (left >= kTrajectoryAssemblyCount ||
      right >= kTrajectoryAssemblyCount || left == right)
    return false;
  if (read_unsigned(words,
                    trajectory_meta_index(left, kTrajectoryActive, 0u)) == 0u ||
      read_unsigned(words,
                    trajectory_meta_index(right, kTrajectoryActive, 0u)) == 0u)
    return false;
  for (std::uint32_t predecessor = 0u;
       predecessor < kTrajectoryPredecessorCount; ++predecessor) {
    if (trajectory_adjacency_support(words, predecessor, left) != 0u &&
        trajectory_adjacency_support(words, predecessor, right) != 0u)
      return true;
  }
  for (std::uint32_t successor = 0u;
       successor < kTrajectoryAssemblyCount; ++successor) {
    if (trajectory_adjacency_support(words, left, successor) != 0u &&
        trajectory_adjacency_support(words, right, successor) != 0u)
      return true;
  }
  return false;
}

__device__ inline bool trajectory_candidate_allowed(
    const SiteWord* words, std::uint32_t previous,
    std::uint32_t candidate) {
  if (previous >= kTrajectoryPredecessorCount ||
      candidate >= kTrajectoryAssemblyCount)
    return false;
  if (previous < kTrajectoryAssemblyCount &&
      read_unsigned(words,
                    trajectory_meta_index(previous, kTrajectoryActive, 0u)) == 0u)
    return false;
  if (trajectory_adjacency_support(words, previous, candidate) != 0u)
    return true;
  if (previous == kTrajectoryStartSentinel)
    return false;
  for (std::uint32_t peer = 0u; peer < kTrajectoryAssemblyCount; ++peer) {
    if (peer == previous ||
        !trajectories_substitutable(words, previous, peer))
      continue;
    if (trajectory_adjacency_support(words, peer, candidate) != 0u)
      return true;
  }
  return false;
}

__device__ inline std::uint32_t select_adjacent_trajectory(
    const SiteWord* words, const DeviceLayout& layout,
    std::uint32_t current_form, std::uint32_t previous,
    std::uint32_t* selected_record, std::uint32_t* candidate_low,
    std::uint32_t* candidate_high, bool* ambiguous) {
  std::uint32_t best = kTrajectoryAssemblyCount;
  std::uint32_t best_matches = 0u;
  std::uint32_t best_admitted = 1u;
  std::int32_t best_credit = -kCreditLimit;
  std::uint32_t best_support = 0u;
  std::uint32_t best_record = kTransitionRecordCount;
  std::uint32_t low = 0u;
  std::uint32_t high = 0u;
  bool tied = false;
  for (std::uint32_t candidate = 0u;
       candidate < kTrajectoryAssemblyCount; ++candidate) {
    const std::uint32_t length = read_unsigned(
        words, trajectory_meta_index(candidate, kTrajectoryLength, 0u));
    if (read_unsigned(words,
                      trajectory_meta_index(candidate, kTrajectoryActive, 0u)) == 0u ||
        length == 0u || length > kTrajectoryPhaseCount ||
        !trajectory_candidate_allowed(words, previous, candidate))
      continue;
    const std::uint32_t first_form =
        read_unsigned(words, trajectory_phase_index(candidate, 0u, 0u));
    if (!decode_form_raw(words, first_form).valid)
      continue;
    std::uint32_t admitted = 0u;
    const std::uint32_t matches =
        trajectory_situation_score(words, layout, candidate, &admitted);
    if (admitted == 0u)
      continue;
    const std::uint32_t record =
        find_transition(words, current_form, first_form);
    if (record >= kTransitionRecordCount)
      continue;
    const std::int32_t credit =
        read_signed(words,
                    transition_meta_index(record, kTransitionCredit, 0u));
    const std::uint32_t support =
        read_unsigned(words,
                      transition_meta_index(record, kTransitionSupport, 0u));
    const std::uint64_t left =
        static_cast<std::uint64_t>(matches) * best_admitted;
    const std::uint64_t right =
        static_cast<std::uint64_t>(best_matches) * admitted;
    if (best == kTrajectoryAssemblyCount || left > right) {
      best = candidate;
      best_matches = matches;
      best_admitted = admitted;
      best_credit = credit;
      best_support = support;
      best_record = record;
      low = record < 32u ? (1u << record) : 0u;
      high = record >= 32u && record < kTransitionRecordCount
                 ? (1u << (record - 32u))
                 : 0u;
      tied = false;
    } else if (left == right) {
      if (record < 32u)
        low |= 1u << record;
      else if (record < kTransitionRecordCount)
        high |= 1u << (record - 32u);
      if (credit > best_credit ||
          (credit == best_credit && support > best_support)) {
        best = candidate;
        best_credit = credit;
        best_support = support;
        best_record = record;
        tied = false;
      } else if (credit == best_credit && support == best_support) {
        tied = true;
      }
    }
  }
  if (candidate_low != nullptr)
    *candidate_low = low;
  if (candidate_high != nullptr)
    *candidate_high = high;
  if (ambiguous != nullptr)
    *ambiguous = tied;
  if (best == kTrajectoryAssemblyCount || tied) {
    if (selected_record != nullptr)
      *selected_record = kTransitionRecordCount;
    return kTrajectoryAssemblyCount;
  }
  if (selected_record != nullptr)
    *selected_record = best_record;
  return best;
}

__device__ inline bool arm_selected_trajectory(
    SiteWord* words, std::uint32_t trajectory, Receipt* receipt) {
  if (trajectory >= kTrajectoryAssemblyCount ||
      read_unsigned(words,
                    trajectory_meta_index(trajectory, kTrajectoryActive, 0u)) == 0u)
    return false;
  const std::uint32_t length = read_unsigned(
      words, trajectory_meta_index(trajectory, kTrajectoryLength, 0u));
  if (length == 0u || length > kTrajectoryPhaseCount)
    return false;
  if (receipt != nullptr) {
    receipt->selected_trajectory = trajectory;
    receipt->trajectory_phase = 0u;
  }
  if (length <= 1u) {
    write_resident(words, pair_index(kGlobalBase, kPreviousTrajectory, 0u),
                   trajectory);
    clear_active_trajectory(words);
    return true;
  }
  write_resident(words, pair_index(kGlobalBase, kActiveTrajectory, 0u),
                 trajectory);
  write_resident(words, pair_index(kGlobalBase, kActiveTrajectoryPhase, 0u),
                 1u);
  return true;
}

__device__ inline bool arm_trajectory_after_first_form(
    SiteWord* words, const DeviceLayout& layout, std::uint32_t first_form,
    Receipt* receipt) {
  bool ambiguous = false;
  const std::uint32_t trajectory =
      select_trajectory(words, layout, first_form, &ambiguous);
  if (trajectory == kTrajectoryAssemblyCount) {
    clear_active_trajectory(words);
    const bool learned =
        has_learned_trajectory_for_first_form(words, first_form);
    if (receipt != nullptr && (ambiguous || learned))
      receipt->trajectory_abstained = 1u;
    return !ambiguous && !learned;
  }
  return arm_selected_trajectory(words, trajectory, receipt);
}

__device__ inline void retain_prediction_eligibility(
    SiteWord* words, std::uint32_t predecessor, std::uint32_t selected_record,
    std::uint32_t candidate_low, std::uint32_t candidate_high) {
  const std::uint32_t prediction =
      read_unsigned(words, pair_index(kGlobalBase, kPredictionId, 0u)) + 1u;
  const std::uint32_t slot = prediction & (kEligibilityDepth - 1u);
  write_resident(words, eligibility_index(slot, kEligibilityEvent, 0u), prediction);
  write_resident(words, eligibility_index(slot, kEligibilityPredecessor, 0u),
                 predecessor);
  write_resident(words, eligibility_index(slot, kEligibilityCandidateLow, 0u),
                 candidate_low);
  write_resident(words, eligibility_index(slot, kEligibilityCandidateHigh, 0u),
                 candidate_high);
  write_resident(words, eligibility_index(slot, kEligibilitySelectedRecord, 0u),
                 selected_record);
  write_resident(words, eligibility_index(slot, kEligibilityStatus, 0u), 0u);
  write_resident(words, eligibility_index(slot, kEligibilityAge, 0u), 0u);
  write_resident(words, eligibility_index(slot, kEligibilityObservedForm, 0u),
                 kFormAssemblyCount);
  write_resident(words, pair_index(kGlobalBase, kPredictionId, 0u), prediction);
}

__device__ inline bool candidate_contains(std::uint32_t low,
                                          std::uint32_t high,
                                          std::uint32_t record) {
  if (record >= kTransitionRecordCount)
    return false;
  return record < 32u ? (low & (1u << record)) != 0u
                      : (high & (1u << (record - 32u))) != 0u;
}

__device__ inline void apply_oldest_prediction_credit(
    SiteWord* words, std::uint32_t observed, Receipt* receipt) {
  const std::uint32_t prediction_count =
      read_unsigned(words, pair_index(kGlobalBase, kPredictionId, 0u));
  const std::uint32_t prediction =
      read_unsigned(words, pair_index(kGlobalBase, kCreditHead, 0u)) + 1u;
  if (prediction == 0u || prediction_count < prediction)
    return;
  const std::uint32_t slot = prediction & (kEligibilityDepth - 1u);
  const std::uint32_t retained =
      read_unsigned(words, eligibility_index(slot, kEligibilityEvent, 0u));
  const std::uint32_t status =
      read_unsigned(words, eligibility_index(slot, kEligibilityStatus, 0u));
  const bool expired = retained != prediction || status == 2u;
  if (expired) {
    write_resident(words, pair_index(kGlobalBase, kCreditHead, 0u), prediction);
    if (retained != prediction)
      write_resident(words, pair_index(kGlobalBase, kExpiredCount, 0u),
                     read_unsigned(
                         words, pair_index(kGlobalBase, kExpiredCount, 0u)) +
                         1u);
    if (receipt != nullptr) {
      ++receipt->expired_events;
      receipt->credited_prediction = prediction;
    }
    return;
  }
  if (status != 0u)
    return;
  const std::uint32_t low = read_unsigned(
      words, eligibility_index(slot, kEligibilityCandidateLow, 0u));
  const std::uint32_t high = read_unsigned(
      words, eligibility_index(slot, kEligibilityCandidateHigh, 0u));
  for (std::uint32_t record = 0u; record < kTransitionRecordCount; ++record) {
    if (!candidate_contains(low, high, record) ||
        read_unsigned(words,
                      transition_meta_index(record, kTransitionActive, 0u)) ==
            0u)
      continue;
    const std::uint32_t successor = read_unsigned(
        words, transition_meta_index(record, kTransitionSuccessor, 0u));
    const std::int32_t delta = successor == observed ? 1 : -1;
    const std::int32_t before = read_signed(
        words, transition_meta_index(record, kTransitionCredit, 0u));
    const std::int32_t after = clamp_credit(
        static_cast<std::int64_t>(before) + static_cast<std::int64_t>(delta));
    if (after == before)
      continue;
    write_resident(words,
                   transition_meta_index(record, kTransitionCredit, 0u),
                   static_cast<SiteWord>(after));
    if (receipt != nullptr) {
      receipt->transition_credit_delta[record] = delta;
      if (delta > 0)
        ++receipt->positive_credit_updates;
      else
        ++receipt->negative_credit_updates;
    }
  }
  write_resident(words, eligibility_index(slot, kEligibilityStatus, 0u), 1u);
  write_resident(words, eligibility_index(slot, kEligibilityObservedForm, 0u),
                 observed);
  write_resident(words, pair_index(kGlobalBase, kCreditHead, 0u), prediction);
  write_resident(words, pair_index(kGlobalBase, kCreditCount, 0u),
                 read_unsigned(words,
                               pair_index(kGlobalBase, kCreditCount, 0u)) +
                     1u);
  if (receipt != nullptr)
    receipt->credited_prediction = prediction;
}

__device__ inline bool has_live_eligibility(const SiteWord* words) {
  for (std::uint32_t slot = 0u; slot < kEligibilityDepth; ++slot) {
    if (read_unsigned(words, eligibility_index(slot, kEligibilityEvent, 0u)) !=
            0u &&
        read_unsigned(words, eligibility_index(slot, kEligibilityStatus, 0u)) ==
            0u)
      return true;
  }
  return false;
}

__device__ inline void age_prediction_eligibility(SiteWord* words,
                                                  Receipt* receipt) {
  for (std::uint32_t slot = 0u; slot < kEligibilityDepth; ++slot) {
    const std::uint32_t event =
        read_unsigned(words, eligibility_index(slot, kEligibilityEvent, 0u));
    if (event == 0u ||
        read_unsigned(words, eligibility_index(slot, kEligibilityStatus, 0u)) !=
            0u)
      continue;
    const std::uint32_t age =
        read_unsigned(words, eligibility_index(slot, kEligibilityAge, 0u)) + 1u;
    if (age >= kEligibilityDepth) {
      write_resident(words, eligibility_index(slot, kEligibilityAge, 0u),
                     kEligibilityDepth);
      write_resident(words, eligibility_index(slot, kEligibilityStatus, 0u), 2u);
      write_resident(words, pair_index(kGlobalBase, kExpiredCount, 0u),
                     read_unsigned(words,
                                   pair_index(kGlobalBase, kExpiredCount, 0u)) +
                         1u);
      if (receipt != nullptr)
        ++receipt->expired_events;
    } else {
      write_resident(words, eligibility_index(slot, kEligibilityAge, 0u), age);
      if (receipt != nullptr)
        ++receipt->aged_events;
    }
  }
}

#include "bcc32_grown_sparse_event_memory_tail.inl"
