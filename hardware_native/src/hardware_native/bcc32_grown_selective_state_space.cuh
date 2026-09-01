#pragma once

#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "hardware_native/bcc32_aperture_geometry.cuh"
#include "hardware_native/bcc32_law.cuh"
#include "hardware_native/bcc32_grown_instance_basin.cuh"
#include "hardware_native/bcc32_raw_byte_tape.cuh"
#include "hardware_native/bcc32_types.cuh"

namespace substrate::bcc32::grown_selective_state_space {

inline constexpr std::uint32_t kCellCount = 512u;
inline constexpr std::uint32_t kFanIn = 8u;
inline constexpr std::uint32_t kContentRouteCount = kCellCount * kFanIn;
inline constexpr std::uint32_t kOutputLaneCount = 8u;
inline constexpr std::uint32_t kOutputRouteCount = kCellCount * kOutputLaneCount;
inline constexpr std::uint32_t kSituationBasinCount = 8u;
inline constexpr std::uint32_t kSituationFieldsPerBasin = 5u;
inline constexpr std::uint32_t kSituationSlotCount =
    kSituationBasinCount * kSituationFieldsPerBasin;
static_assert(kSituationBasinCount == grown_instance_basin_factor::kBasinCount);
inline constexpr std::uint32_t kEligibilityDepth = 32u;
inline constexpr std::uint32_t kJournalDepth = 64u;
inline constexpr std::uint32_t kLedgerFields = 16u;
inline constexpr SiteWord kFactorMarkerValue = 0x53534d31u;
inline constexpr SiteWord kLayoutVersionValue = 5u;
inline constexpr std::int32_t kTraceLimit = 32767;
inline constexpr std::int32_t kCreditLimit = 256;
inline constexpr std::uint32_t kCreditShift = 15u;

enum GlobalField : std::uint32_t {
  kFactorMarker = 0u,
  kLayoutVersion,
  kEventId,
  kJournalCount,
  kActiveBank,
  kCreditHead,
  kPredictionCount,
  kCreditCount,
  kExpiredCount,
  kGlobalFieldCount,
};

inline constexpr std::uint32_t kGlobalBase = 0u;
inline constexpr std::uint32_t kStateBase = kGlobalBase + kGlobalFieldCount * 2u;
inline constexpr std::uint32_t kStateRailCount = 2u * kCellCount * 2u;
inline constexpr std::uint32_t kContentBase = kStateBase + kStateRailCount;
inline constexpr std::uint32_t kContentRailCount = kContentRouteCount * 2u;
inline constexpr std::uint32_t kOutputBase = kContentBase + kContentRailCount;
inline constexpr std::uint32_t kOutputRailCount = kOutputRouteCount * 2u;
inline constexpr std::uint32_t kRawBitCount = 8u;
inline constexpr std::uint32_t kRawProjectionRouteCount = kCellCount * kRawBitCount;
inline constexpr std::uint32_t kRawProjectionBase = kOutputBase + kOutputRailCount;
inline constexpr std::uint32_t kRawProjectionRailCount = kRawProjectionRouteCount * 2u;
// Each event owns an amplitude trace for every recurrent route and one
// factorized output trace per cell. Every trace is a signed SiteWord pair.
inline constexpr std::uint32_t kRecurrentTraceCount = kContentRouteCount;
inline constexpr std::uint32_t kOutputTraceCount = kCellCount;
inline constexpr std::uint32_t kEligibilityWordsPerEvent =
    kRecurrentTraceCount + kOutputTraceCount;
inline constexpr std::uint32_t kEligibilityBase = kRawProjectionBase + kRawProjectionRailCount;
inline constexpr std::uint32_t kEligibilityRailCount =
    kEligibilityDepth * kEligibilityWordsPerEvent * 2u;
inline constexpr std::uint32_t kLedgerBase = kEligibilityBase + kEligibilityRailCount;
inline constexpr std::uint32_t kLedgerRailCount = kEligibilityDepth * kLedgerFields * 2u;
inline constexpr std::uint32_t kMutableRailCount = kLedgerBase + kLedgerRailCount;
inline constexpr std::uint32_t kJournalMetaRailCount = 2u;
inline constexpr std::uint32_t kJournalGlobalRailCount = kGlobalFieldCount * 2u;
inline constexpr std::uint32_t kJournalStateRailCount = kStateRailCount;
inline constexpr std::uint32_t kJournalEligibilityRailCount = kEligibilityWordsPerEvent * 2u;
inline constexpr std::uint32_t kJournalLedgerRailCount = kLedgerFields * 2u;
// U and V are the mutable weight banks journaled per event. The raw projection
// is founder-resident and remains in the ordinary checkpointed mutable region,
// but is not copied into every event record because this factor never updates it.
inline constexpr std::uint32_t kJournalWeightRailCount = kContentRailCount + kOutputRailCount;
inline constexpr std::uint32_t kJournalMotorRailCount = 4u;
inline constexpr std::uint32_t kJournalGlobalOffset = kJournalMetaRailCount;
inline constexpr std::uint32_t kJournalStateOffset = kJournalGlobalOffset + kJournalGlobalRailCount;
inline constexpr std::uint32_t kJournalEligibilityOffset =
    kJournalStateOffset + kJournalStateRailCount;
inline constexpr std::uint32_t kJournalLedgerOffset =
    kJournalEligibilityOffset + kJournalEligibilityRailCount;
inline constexpr std::uint32_t kJournalWeightOffset =
    kJournalLedgerOffset + kJournalLedgerRailCount;
inline constexpr std::uint32_t kJournalMotorOffset = kJournalWeightOffset + kJournalWeightRailCount;
inline constexpr std::uint32_t kJournalRecordRailCount =
    kJournalMotorOffset + kJournalMotorRailCount;
inline constexpr std::uint32_t kJournalBase = kMutableRailCount;
inline constexpr std::uint32_t kPhysicalRailCount =
    kMutableRailCount + kJournalRecordRailCount * kJournalDepth;

static_assert(kPhysicalRailCount < 3'600'000u);
static_assert(kJournalWeightRailCount == kContentRailCount + kOutputRailCount);
static_assert(kJournalMotorOffset == kJournalWeightOffset + kJournalWeightRailCount);

enum StepMode : std::uint32_t {
  kStepNone = 0u,
  kStepPredicted = 1u,
  kStepCredited = 2u,
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
  std::uint64_t situation_slots[kSituationSlotCount]{};
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
    for (std::uint32_t field = 0u; field < kSituationFieldsPerBasin; ++field) {
      const std::uint32_t source = grown_instance_basin_factor::resident_index(
          grown_instance_basin_factor::basin_field(basin, fields[field]));
      layout.situation_slots[basin * kSituationFieldsPerBasin + field] =
          situation.rails[source];
    }
  }
  return layout;
}

// This transient aperture stages timing only. Observation content is decoded
// from the resident raw-sensory rails; event identity is minted by resident
// state.
struct DeviceInputs {
  std::uint32_t predict_staged = 0u;
  std::uint32_t observe_staged = 0u;
};

struct DeviceScratch {
  std::uint32_t reserved = 0u;
  std::uint32_t accepted = 0u;
  std::uint32_t journal_slot = 0u;
  std::uint32_t event_slot = 0u;
  std::uint32_t credit_event = 0u;
  std::uint32_t credit_valid = 0u;
  std::uint32_t credit_expired = 0u;
  std::uint32_t observed_byte = 0u;
#ifdef BCC32_SELECTIVE_TEST_CONTROL
  std::uint32_t invert_residual = 0u;
#endif
  std::int32_t local_sensitivity[kCellCount]{};
  std::int32_t learning_signal[kCellCount]{};
  unsigned long long output_positive[kOutputLaneCount]{};
  unsigned long long output_negative[kOutputLaneCount]{};
};

struct Receipt {
  std::uint32_t step_mode = 0u;
  std::uint32_t event_id = 0u;
  std::uint32_t selected_byte = 0u;
  std::uint32_t state_hash = 0u;
  std::uint32_t predicted_events = 0u;
  std::uint32_t observed_events = 0u;
  std::uint32_t correct_events = 0u;
  std::uint32_t error_events = 0u;
  std::uint32_t expired_events = 0u;
  std::uint32_t content_updates = 0u;
  std::uint32_t output_updates = 0u;
  std::uint32_t journal_exhausted = 0u;
  std::uint32_t invalid_observations = 0u;
  std::int64_t output_margin[kOutputLaneCount]{};
  // One bit per fan-in lane for each cell; this preserves all changed-route
  // evidence without expanding the receipt for the eight-route fan-in.
  std::uint32_t changed_content_route[kCellCount]{};
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
  std::uint32_t active_bank_valid = 0u;
  std::uint32_t fifo_valid = 0u;
  std::uint32_t invalid_pairs = 0u;
  std::uint32_t invalid_journal_events = 0u;
  std::uint32_t invalid_eligibility = 0u;
  std::uint32_t invalid_ledger = 0u;
};

// Region is a dense 80 x 200 x ceil(kPhysicalRailCount/16000) box, translated
// (not resized) to keep every site inside the kSpatialMacroClosureRadius=26
// clearance window on all three axes, and to stay x-disjoint from the
// sparse_event_memory region immediately below it (which was translated too).
__host__ __device__ inline PhysicalOffset physical_offset(std::uint32_t index) {
  return {144 + static_cast<std::int32_t>(index % 80u),
          -224 + static_cast<std::int32_t>((index / 80u) % 200u),
          -220 + static_cast<std::int32_t>(index / 16000u)};
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

__host__ __device__ inline std::uint32_t pair_index(std::uint32_t base, std::uint32_t item,
                                                    std::uint32_t polarity) {
  return base + item * 2u + polarity;
}

__host__ __device__ inline std::uint32_t state_index(std::uint32_t bank, std::uint32_t cell,
                                                     std::uint32_t polarity) {
  return pair_index(kStateBase, bank * kCellCount + cell, polarity);
}

__host__ __device__ inline std::uint32_t content_index(std::uint32_t route,
                                                       std::uint32_t polarity) {
  return pair_index(kContentBase, route, polarity);
}

__host__ __device__ inline std::uint32_t output_index(std::uint32_t lane, std::uint32_t cell,
                                                      std::uint32_t polarity) {
  return pair_index(kOutputBase, lane * kCellCount + cell, polarity);
}

__host__ __device__ inline std::uint32_t raw_index(std::uint32_t cell, std::uint32_t bit,
                                                   std::uint32_t polarity) {
  return pair_index(kRawProjectionBase, cell * kRawBitCount + bit, polarity);
}

__host__ __device__ inline std::uint32_t trace_index(std::uint32_t event_slot, bool output,
                                                     std::uint32_t item,
                                                     std::uint32_t polarity) {
  const std::uint32_t offset = output ? kRecurrentTraceCount + item : item;
  return pair_index(kEligibilityBase, event_slot * kEligibilityWordsPerEvent + offset, polarity);
}

__host__ __device__ inline std::uint32_t ledger_index(std::uint32_t event_slot, std::uint32_t field,
                                                      std::uint32_t polarity) {
  return pair_index(kLedgerBase, event_slot * kLedgerFields + field, polarity);
}

__host__ __device__ inline std::uint32_t journal_index(std::uint32_t journal_slot,
                                                       std::uint32_t resident_index) {
  return kJournalBase + journal_slot * kJournalRecordRailCount + resident_index;
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
// sufficient for grown_selective_state_space::restore_inverse_kernel.
//
// Splitting the parity fold out of the recursive position is behaviour-identical
// BY CONSTRUCTION, not by measurement: the odd branch called this exact body
// with this exact argument, and the even path is untouched.
__host__ __device__ inline SiteWord founder_value_even(std::uint32_t index) {
  if (index == pair_index(kGlobalBase, kFactorMarker, 0u))
    return kFactorMarkerValue;
  if (index == pair_index(kGlobalBase, kLayoutVersion, 0u))
    return kLayoutVersionValue;
  if (index >= kContentBase && index < kContentBase + kContentRailCount) {
    const std::uint32_t route = (index - kContentBase) / 2u;
    return static_cast<SiteWord>(static_cast<std::int32_t>(
        static_cast<std::int32_t>(mix32(route ^ 0xc001d00du) & 0xffu) - 128));
  }
  if (index >= kOutputBase && index < kOutputBase + kOutputRailCount) {
    const std::uint32_t route = (index - kOutputBase) / 2u;
    return static_cast<SiteWord>(static_cast<std::int32_t>(
        static_cast<std::int32_t>(mix32(route ^ 0x51a7e5u) & 0xffu) - 128));
  }
  if (index >= kRawProjectionBase && index < kRawProjectionBase + kRawProjectionRailCount) {
    const std::uint32_t route = (index - kRawProjectionBase) / 2u;
    return (mix32(route ^ 0x72617770u) & 1u) != 0u ? 1024 : -1024;
  }
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

__device__ inline std::int32_t read_signed(const SiteWord* words, std::uint32_t index) {
  return static_cast<std::int32_t>(read_resident(words, index));
}

__device__ inline std::int32_t clamp_q15(std::int64_t value) {
  if (value > kTraceLimit)
    return kTraceLimit;
  if (value < -kTraceLimit)
    return -kTraceLimit;
  return static_cast<std::int32_t>(value);
}

__device__ inline std::int32_t clamp_credit_delta(std::int64_t value) {
  if (value > kCreditLimit)
    return kCreditLimit;
  if (value < -kCreditLimit)
    return -kCreditLimit;
  return static_cast<std::int32_t>(value);
}

// Credit must be odd-symmetric: truncation toward zero makes the opposite
// residual control the exact negation even when the product is sub-quantum.
__device__ inline std::int32_t scale_credit(std::int64_t value, std::uint32_t shift) {
  const std::int64_t divisor = static_cast<std::int64_t>(1ull << shift);
  return value >= 0 ? static_cast<std::int32_t>(value / divisor)
                    : static_cast<std::int32_t>(-((-value) / divisor));
}

__device__ inline std::uint32_t route_source(std::uint32_t target, std::uint32_t lane) {
  return mix32(target * 0x9e3779b9u + lane * 0x85ebca6bu) % kCellCount;
}

// Both raw boundary rails drive the same resident fixed-point projection.
__device__ inline std::int32_t project_raw_byte(const SiteWord* words, std::uint32_t cell,
                                                const RawByteDecode raw) {
  if (!raw.valid)
    return 0;
  std::int64_t drive = 0;
  for (std::uint32_t bit = 0u; bit < kRawBitCount; ++bit) {
    const std::int32_t weight = read_signed(words, raw_index(cell, bit, 0u));
    const bool observed = ((raw.value >> bit) & 1u) != 0u;
    drive += observed ? weight : -weight;
  }
  return clamp_q15(drive);
}

__device__ inline std::int32_t fuse_raw_boundaries(const SiteWord* words, std::uint32_t cell,
                                                   const RawByteDecode sensory,
                                                   const RawByteDecode reafference) {
  const std::int32_t sensory_drive = project_raw_byte(words, cell, sensory);
  const std::int32_t motor_drive = project_raw_byte(words, cell, reafference);
  if (sensory.valid && reafference.valid)
    return static_cast<std::int32_t>(
        (static_cast<std::int64_t>(sensory_drive) + motor_drive) / 2);
  return sensory.valid ? sensory_drive : motor_drive;
}

static __global__ void reserve_step_kernel(SiteWord* words, const DeviceLayout layout,
                                           const DeviceInputs* inputs,
                                           DeviceScratch* scratch, Receipt* receipt,
                                           const std::uint32_t* advanced) {
  if (blockIdx.x != 0u || threadIdx.x != 0u || scratch == nullptr)
    return;
  scratch->reserved = 0u;
  scratch->accepted = 0u;
  scratch->credit_valid = 0u;
  scratch->credit_expired = 0u;
  scratch->observed_byte = 0u;
  const bool staged = inputs != nullptr && scratch != nullptr && receipt != nullptr &&
                      ((inputs->predict_staged != 0u) != (inputs->observe_staged != 0u));
  if (!staged) {
    if (receipt != nullptr)
      receipt->step_mode = kStepNone;
    return;
  }
  if (inputs->observe_staged != 0u) {
    const RawByteDecode observed = decode_raw_byte_carriers(
        {words[layout.raw_sensory_zero_slot], words[layout.raw_sensory_one_slot]});
    scratch->observed_byte = observed.value;
    if (!observed.valid) {
      ++receipt->invalid_observations;
      receipt->step_mode = kStepNone;
      return;
    }
  }
  if (advanced == nullptr || (*advanced & 1u) == 0u)
    return;
  const std::uint32_t count =
      static_cast<std::uint32_t>(read_resident(words, pair_index(kGlobalBase, kJournalCount, 0u)));
  if (count >= kJournalDepth) {
    receipt->step_mode = kStepNone;
    ++receipt->journal_exhausted;
    return;
  }
  const std::uint32_t event = inputs->predict_staged != 0u
                                  ? static_cast<std::uint32_t>(read_resident(
                                        words, pair_index(kGlobalBase, kEventId, 0u))) +
                                        1u
                                  : static_cast<std::uint32_t>(read_resident(
                                        words, pair_index(kGlobalBase, kCreditHead, 0u))) +
                                        1u;
  scratch->journal_slot = count;
  scratch->credit_event = event;
  scratch->event_slot = event & (kEligibilityDepth - 1u);
  scratch->reserved = 1u;
  for (std::uint32_t lane = 0u; lane < kOutputLaneCount; ++lane) {
    scratch->output_positive[lane] = 0ull;
    scratch->output_negative[lane] = 0ull;
  }
  for (std::uint32_t cell = 0u; cell < kCellCount; ++cell)
    receipt->changed_content_route[cell] = 0u;
}

enum CreditStatus : std::uint32_t {
  kCreditNoPending = 0u,
  kCreditValid = 1u,
  kCreditExpired = 2u,
  kCreditConsumed = 3u,
};

__device__ inline CreditStatus classify_credit(const SiteWord* words) {
  const std::uint32_t current =
      static_cast<std::uint32_t>(read_resident(words, pair_index(kGlobalBase, kEventId, 0u)));
  const std::uint32_t event =
      static_cast<std::uint32_t>(read_resident(words, pair_index(kGlobalBase, kCreditHead, 0u))) +
      1u;
  if (event == 0u || current < event)
    return kCreditNoPending;
  const std::uint32_t slot = event & (kEligibilityDepth - 1u);
  if (current - event >= kEligibilityDepth ||
      read_resident(words, ledger_index(slot, 0u, 0u)) != event)
    return kCreditExpired;
  if (read_resident(words, ledger_index(slot, 12u, 0u)) != 0u)
    return kCreditConsumed;
  return kCreditValid;
}

__device__ inline bool step_available(const SiteWord* words, const DeviceInputs* inputs) {
  if (inputs == nullptr)
    return true;
  const bool requested = inputs->predict_staged != 0u || inputs->observe_staged != 0u;
  if (!requested)
    return true;
  const bool valid = (inputs->predict_staged != 0u) != (inputs->observe_staged != 0u);
  const std::uint32_t count =
      static_cast<std::uint32_t>(read_resident(words, pair_index(kGlobalBase, kJournalCount, 0u)));
  if (!valid || count >= kJournalDepth)
    return false;
  if (inputs->observe_staged == 0u)
    return true;
  const CreditStatus status = classify_credit(words);
  return status == kCreditValid || status == kCreditExpired;
}

static __global__ void preflight_step_kernel(const SiteWord* words, const DeviceInputs* inputs,
                                             std::uint32_t* advanced) {
  if (blockIdx.x != 0u || threadIdx.x != 0u || advanced == nullptr)
    return;
  *advanced = step_available(words, inputs) ? 1u : 0u;
}

static __global__ void snapshot_step_kernel(SiteWord* words, const DeviceLayout layout,
                                            const DeviceScratch* scratch) {
  if (scratch == nullptr || scratch->reserved == 0u)
    return;
  const std::uint32_t count = scratch->journal_slot;
  const std::uint32_t event_slot = scratch->event_slot;
  const std::uint32_t stride = blockDim.x * gridDim.x;
  const std::uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  for (std::uint32_t index = tid; index < kJournalRecordRailCount; index += stride) {
    SiteWord value = 0u;
    if (index < kJournalMetaRailCount) {
      value = (index & 1u) == 0u ? event_slot : ~event_slot;
    } else if (index < kJournalStateOffset) {
      value = read_resident(words, index - kJournalGlobalOffset);
    } else if (index < kJournalEligibilityOffset) {
      value = read_resident(words, kStateBase + index - kJournalStateOffset);
    } else if (index < kJournalLedgerOffset) {
      value = read_resident(
          words, pair_index(kEligibilityBase, event_slot * kEligibilityWordsPerEvent, 0u) + index -
                     kJournalEligibilityOffset);
    } else if (index < kJournalWeightOffset) {
      value = read_resident(words, pair_index(kLedgerBase, event_slot * kLedgerFields, 0u) + index -
                                       kJournalLedgerOffset);
    } else if (index < kJournalMotorOffset) {
      value = read_resident(words, kContentBase + index - kJournalWeightOffset);
    } else {
      const bool one = index >= kJournalMotorOffset + 2u;
      const SiteWord motor = words[one ? layout.raw_motor_one_slot : layout.raw_motor_zero_slot];
      value = (index & 1u) == 0u ? motor : ~motor;
    }
    words[fixed_physical_slot(journal_index(count, index))] = value;
  }
}

static __global__ void commit_step_kernel(SiteWord* words, const DeviceInputs* inputs,
                                          DeviceScratch* scratch, Receipt* receipt) {
  if (blockIdx.x != 0u || threadIdx.x != 0u || scratch == nullptr || scratch->reserved == 0u ||
      inputs == nullptr || receipt == nullptr)
    return;
  write_resident(words, pair_index(kGlobalBase, kJournalCount, 0u), scratch->journal_slot + 1u);
  scratch->accepted = 1u;
  receipt->step_mode = inputs->predict_staged != 0u ? kStepPredicted : kStepCredited;
}

static __global__ void clear_event_eligibility_kernel(SiteWord* words, const DeviceInputs* inputs,
                                                      const DeviceScratch* scratch) {
  if (scratch == nullptr || scratch->accepted == 0u || inputs == nullptr ||
      inputs->predict_staged == 0u)
    return;
  const std::uint32_t event_slot = scratch->event_slot;
  const std::uint32_t total = kEligibilityWordsPerEvent;
  for (std::uint32_t item = blockIdx.x * blockDim.x + threadIdx.x; item < total;
       item += blockDim.x * gridDim.x) {
    const std::uint32_t index =
        pair_index(kEligibilityBase, event_slot * kEligibilityWordsPerEvent + item, 0u);
    write_resident(words, index, 0u);
  }
}

static __global__ void state_step_kernel(SiteWord* words, const DeviceLayout layout,
                                         const DeviceInputs* inputs,
                                         DeviceScratch* scratch) {
  if (scratch == nullptr || scratch->accepted == 0u || inputs == nullptr ||
      inputs->predict_staged == 0u)
    return;
  const std::uint32_t cell = blockIdx.x * blockDim.x + threadIdx.x;
  if (cell >= kCellCount)
    return;
  const std::uint32_t active =
      static_cast<std::uint32_t>(read_resident(words, pair_index(kGlobalBase, kActiveBank, 0u))) &
      1u;
  const std::uint32_t next = active ^ 1u;
  const std::int32_t previous = read_signed(words, state_index(active, cell, 0u));
  std::int64_t drive = 0;
  const RawByteDecode sensory = decode_raw_byte_carriers(
      {words[layout.raw_sensory_zero_slot], words[layout.raw_sensory_one_slot]});
  const RawByteDecode reafference = decode_raw_byte_carriers(
      {words[layout.raw_motor_zero_slot], words[layout.raw_motor_one_slot]});
  drive += fuse_raw_boundaries(words, cell, sensory, reafference);
  for (std::uint32_t channel = 0u; channel < kSituationSlotCount; ++channel) {
    const SiteWord resident = words[layout.situation_slots[channel]];
    const SiteWord mask = mix32(cell + channel * 0x9e3779b9u);
    drive += (static_cast<std::int32_t>(__popc(resident ^ mask)) - 16) * 8;
  }
  for (std::uint32_t lane = 0u; lane < kFanIn; ++lane) {
    const std::uint32_t route = cell * kFanIn + lane;
    const std::uint32_t source = route_source(cell, lane);
    const std::int32_t state = read_signed(words, state_index(active, source, 0u));
    const std::int32_t weight = read_signed(words, content_index(route, 0u));
    drive += (static_cast<std::int64_t>(state) * weight) >> 12u;
  }
  const std::int32_t candidate = clamp_q15(drive);
  const std::uint32_t shift = cell < 192u ? 1u : (cell < 320u ? 2u : (cell < 448u ? 4u : 5u));
  const std::int64_t updated_unclamped = previous + ((candidate - previous) >> shift);
  const std::int32_t updated = clamp_q15(updated_unclamped);
  scratch->local_sensitivity[cell] = updated - previous;
  write_resident(words, state_index(next, cell, 0u), static_cast<SiteWord>(updated));

  const std::uint32_t event_slot = scratch->event_slot;
  const std::uint32_t previous_event_slot =
      (scratch->credit_event - 1u) & (kEligibilityDepth - 1u);
  for (std::uint32_t lane = 0u; lane < kFanIn; ++lane) {
    const std::uint32_t route = cell * kFanIn + lane;
    const std::int32_t source_state =
        read_signed(words, state_index(active, route_source(cell, lane), 0u));
    const std::int32_t previous_trace = read_signed(
        words, trace_index(previous_event_slot, false, route, 0u));
    // The first term is the exact resident relaxation leak for this target
    // population. Direct sensitivity is admitted only while both candidate
    // and final clamps remain locally responsive.
    const bool candidate_sensitive = drive > -kTraceLimit && drive < kTraceLimit;
    const bool final_sensitive = updated_unclamped > -kTraceLimit &&
                                 updated_unclamped < kTraceLimit;
    const std::int32_t direct = candidate_sensitive && final_sensitive && updated != previous
                                    ? source_state
                                    : 0;
    const std::int64_t trace = static_cast<std::int64_t>(previous_trace) -
                               (previous_trace >> shift) + (direct >> shift);
    write_resident(words, trace_index(event_slot, false, route, 0u), clamp_q15(trace));
  }
}

static __global__ void output_step_kernel(SiteWord* words, const DeviceInputs* inputs,
                                          DeviceScratch* scratch) {
  if (scratch == nullptr || scratch->accepted == 0u || inputs == nullptr ||
      inputs->predict_staged == 0u)
    return;
  const std::uint32_t cell = blockIdx.x * blockDim.x + threadIdx.x;
  if (cell >= kCellCount)
    return;
  const std::uint32_t active =
      (static_cast<std::uint32_t>(read_resident(words, pair_index(kGlobalBase, kActiveBank, 0u))) ^
       1u) &
      1u;
  const std::int32_t state = read_signed(words, state_index(active, cell, 0u));
  const std::uint32_t event_slot = scratch->event_slot;
  write_resident(words, trace_index(event_slot, true, cell, 0u), state);
  for (std::uint32_t lane = 0u; lane < kOutputLaneCount; ++lane) {
    const std::int32_t weight = read_signed(words, output_index(lane, cell, 0u));
    const long long contribution = (static_cast<long long>(state) * weight) >> 8u;
    if (contribution >= 0)
      atomicAdd(&scratch->output_positive[lane], static_cast<unsigned long long>(contribution));
    else
      atomicAdd(&scratch->output_negative[lane], static_cast<unsigned long long>(-contribution));
  }
}

static __global__ void publish_kernel(SiteWord* words, const DeviceLayout layout,
                                      const DeviceInputs* inputs, const DeviceScratch* scratch,
                                      Receipt* receipt) {
  if (blockIdx.x != 0u || threadIdx.x != 0u || inputs == nullptr || scratch == nullptr ||
      scratch->accepted == 0u || inputs->predict_staged == 0u)
    return;
  const std::uint32_t event = scratch->credit_event;
  const std::uint32_t slot = scratch->event_slot;
  std::uint32_t selected = 0u;
  std::uint32_t hash = 2166136261u;
  for (std::uint32_t field = 0u; field < kLedgerFields; ++field)
    write_resident(words, ledger_index(slot, field, 0u), 0u);
  for (std::uint32_t lane = 0u; lane < kOutputLaneCount; ++lane) {
    const bool positive = scratch->output_positive[lane] >= scratch->output_negative[lane];
    if (lane < 8u && positive)
      selected |= 1u << lane;
    write_resident(words, ledger_index(slot, 2u + lane, 0u), positive ? 1u : 0u);
    receipt->output_margin[lane] = static_cast<std::int64_t>(scratch->output_positive[lane]) -
                                   static_cast<std::int64_t>(scratch->output_negative[lane]);
  }
  const std::uint32_t active =
      (static_cast<std::uint32_t>(read_resident(words, pair_index(kGlobalBase, kActiveBank, 0u))) ^
       1u) &
      1u;
  for (std::uint32_t cell = 0u; cell < kCellCount; ++cell) {
    hash ^= static_cast<std::uint32_t>(read_signed(words, state_index(active, cell, 0u)));
    hash *= 16777619u;
  }
  write_resident(words, ledger_index(slot, 0u, 0u), event);
  write_resident(words, ledger_index(slot, 1u, 0u), selected);
  write_resident(words, ledger_index(slot, 11u, 0u), hash);
  write_resident(words, pair_index(kGlobalBase, kEventId, 0u), event);
  write_resident(words, pair_index(kGlobalBase, kActiveBank, 0u), active);
  write_resident(words, pair_index(kGlobalBase, kPredictionCount, 0u),
                 read_resident(words, pair_index(kGlobalBase, kPredictionCount, 0u)) + 1u);
  RawByteRails motor{words[layout.raw_motor_zero_slot], words[layout.raw_motor_one_slot]};
  motor = with_raw_byte_carriers(motor, static_cast<std::uint8_t>(selected));
  words[layout.raw_motor_zero_slot] = motor.zero;
  words[layout.raw_motor_one_slot] = motor.one;
  if (receipt != nullptr) {
    receipt->event_id = event;
    receipt->selected_byte = selected;
    receipt->state_hash = hash;
    ++receipt->predicted_events;
  }
}

static __global__ void reserve_credit_kernel(SiteWord* words, const DeviceInputs* inputs,
                                             DeviceScratch* scratch) {
  if (blockIdx.x != 0u || threadIdx.x != 0u || scratch == nullptr || scratch->accepted == 0u ||
      inputs == nullptr || inputs->observe_staged == 0u)
    return;
  scratch->credit_valid = 0u;
  scratch->credit_expired = 0u;
  const CreditStatus status = classify_credit(words);
  if (status == kCreditExpired) {
    scratch->credit_expired = 1u;
    return;
  }
  if (status == kCreditValid)
    scratch->credit_valid = 1u;
}

__device__ inline std::int32_t residual_polarity(const DeviceScratch* scratch) {
#ifdef BCC32_SELECTIVE_TEST_CONTROL
  return scratch->invert_residual != 0u ? -1 : 1;
#else
  (void)scratch;
  return 1;
#endif
}

static __global__ void credit_signal_kernel(SiteWord* words, const DeviceInputs* inputs,
                                            DeviceScratch* scratch, Receipt* receipt) {
  if (scratch == nullptr || scratch->accepted == 0u || inputs == nullptr ||
      inputs->observe_staged == 0u)
    return;
  const std::uint32_t cell = blockIdx.x * blockDim.x + threadIdx.x;
  if (scratch->credit_valid == 0u)
    return;
  const std::uint32_t slot = scratch->event_slot;
  const std::int32_t polarity = residual_polarity(scratch);
  std::int32_t signal = 0;
  for (std::uint32_t lane = 0u; lane < kOutputLaneCount; ++lane) {
    const bool observed = ((scratch->observed_byte >> lane) & 1u) != 0u;
    const bool predicted = read_resident(words, ledger_index(slot, 2u + lane, 0u)) != 0u;
    const std::int32_t delta = observed == predicted ? 0 : ((observed ? 1 : -1) * polarity);
    if (cell < kCellCount && delta != 0) {
      // This is deliberately read before V is updated. All mismatching bits
      // therefore contribute through the pre-update output bank.
      signal += delta * read_signed(words, output_index(lane, cell, 0u));
    }
  }
  if (cell < kCellCount)
    scratch->learning_signal[cell] = signal;
  if (cell == 0u) {
    for (std::uint32_t lane = 0u; lane < kOutputLaneCount; ++lane) {
      const bool observed = ((scratch->observed_byte >> lane) & 1u) != 0u;
      const bool predicted = read_resident(words, ledger_index(slot, 2u + lane, 0u)) != 0u;
      scratch->output_positive[lane] = 0ull;
      scratch->output_negative[lane] = 0ull;
      if (observed != predicted) {
        const std::int32_t delta = ((observed ? 1 : -1) * polarity);
        if (delta > 0)
          scratch->output_positive[lane] = 1ull;
        else
          scratch->output_negative[lane] = 1ull;
      }
    }
  }
}

static __global__ void apply_credit_kernel(SiteWord* words, const DeviceInputs* inputs,
                                           const DeviceScratch* scratch, Receipt* receipt) {
  if (scratch == nullptr || scratch->accepted == 0u || inputs == nullptr ||
      inputs->observe_staged == 0u)
    return;
  const std::uint32_t cell = blockIdx.x * blockDim.x + threadIdx.x;
  if (scratch->credit_valid == 0u)
    return;
  const std::uint32_t slot = scratch->event_slot;
  const std::int32_t signal = cell < kCellCount ? scratch->learning_signal[cell] : 0;
  if (cell < kCellCount && signal != 0) {
    for (std::uint32_t lane = 0u; lane < kFanIn; ++lane) {
      const std::uint32_t route = cell * kFanIn + lane;
      const std::int32_t trace = read_signed(words, trace_index(slot, false, route, 0u));
      if (trace == 0)
        continue;
      const std::int32_t before = read_signed(words, content_index(route, 0u));
      const std::int32_t delta = clamp_credit_delta(
          scale_credit(static_cast<std::int64_t>(signal) * trace, kCreditShift));
      const std::int32_t after = clamp_q15(static_cast<std::int64_t>(before) + delta);
      if (after != before) {
        write_resident(words, content_index(route, 0u), static_cast<SiteWord>(after));
        if (receipt != nullptr) {
          atomicAdd(&receipt->content_updates, 1u);
          atomicOr(&receipt->changed_content_route[cell], 1u << lane);
        }
      }
    }
  }
  if (cell < kCellCount) {
    for (std::uint32_t lane = 0u; lane < kOutputLaneCount; ++lane) {
      const bool mismatch = scratch->output_positive[lane] != 0ull ||
                            scratch->output_negative[lane] != 0ull;
      if (!mismatch)
        continue;
      const std::int32_t trace = read_signed(words, trace_index(slot, true, cell, 0u));
      if (trace == 0)
        continue;
      const std::int32_t residual = scratch->output_positive[lane] != 0ull ? 1 : -1;
      const std::int32_t before = read_signed(words, output_index(lane, cell, 0u));
      const std::int32_t delta = clamp_credit_delta(
          scale_credit(static_cast<std::int64_t>(residual) * trace, 8u));
      const std::int32_t after = clamp_q15(static_cast<std::int64_t>(before) + delta);
      if (after != before) {
        write_resident(words, output_index(lane, cell, 0u), static_cast<SiteWord>(after));
        if (receipt != nullptr)
          atomicAdd(&receipt->output_updates, 1u);
      }
    }
  }
}

static __global__ void finalize_credit_kernel(SiteWord* words, const DeviceInputs* inputs,
                                              DeviceScratch* scratch, Receipt* receipt) {
  if (blockIdx.x != 0u || threadIdx.x != 0u || scratch == nullptr || scratch->accepted == 0u ||
      inputs == nullptr || inputs->observe_staged == 0u)
    return;
  const std::uint32_t event = scratch->credit_event;
  const std::uint32_t slot = scratch->event_slot;
  if (scratch->credit_expired != 0u) {
    ++receipt->expired_events;
    write_resident(words, pair_index(kGlobalBase, kExpiredCount, 0u),
                   read_resident(words, pair_index(kGlobalBase, kExpiredCount, 0u)) + 1u);
    write_resident(words, pair_index(kGlobalBase, kCreditHead, 0u), event);
    return;
  }
  if (scratch->credit_valid != 0u) {
    const std::uint32_t predicted =
        static_cast<std::uint32_t>(read_resident(words, ledger_index(slot, 1u, 0u)));
    const bool correct = (predicted & 0xffu) == (scratch->observed_byte & 0xffu);
    write_resident(words, pair_index(kGlobalBase, kCreditCount, 0u),
                   read_resident(words, pair_index(kGlobalBase, kCreditCount, 0u)) + 1u);
    write_resident(words, ledger_index(slot, 12u, 0u), 1u);
    write_resident(words, pair_index(kGlobalBase, kCreditHead, 0u), event);
    if (receipt != nullptr) {
      ++receipt->observed_events;
      if (correct)
        ++receipt->correct_events;
      else
        ++receipt->error_events;
    }
  }
}

static __global__ void finalize_step_kernel(SiteWord* words, const DeviceInputs* inputs,
                                            const Receipt* receipt, std::uint32_t* advanced,
                                            std::uint32_t advancement_bit) {
  if (blockIdx.x != 0u || threadIdx.x != 0u || receipt == nullptr)
    return;
  const bool requested =
      inputs != nullptr && (inputs->predict_staged != 0u || inputs->observe_staged != 0u);
  if (requested && receipt->step_mode == kStepNone) {
    *advanced = 0u;
  } else if (receipt->step_mode != kStepNone) {
    *advanced |= advancement_bit;
  }
}

static __global__ void prepare_inverse_kernel(SiteWord* words, InverseScratch* scratch) {
  if (blockIdx.x != 0u || threadIdx.x != 0u || scratch == nullptr)
    return;
  scratch->valid = 0u;
  scratch->error = 0u;
  const std::uint32_t count =
      static_cast<std::uint32_t>(read_resident(words, pair_index(kGlobalBase, kJournalCount, 0u)));
  if (count == 0u) {
    scratch->error = 1u;
    return;
  }
  scratch->journal_slot = count - 1u;
  scratch->event_slot = static_cast<std::uint32_t>(
      words[fixed_physical_slot(journal_index(scratch->journal_slot, 0u))]);
  if (scratch->event_slot >= kEligibilityDepth) {
    scratch->error = 1u;
    return;
  }
  scratch->valid = 1u;
}

static __global__ void restore_inverse_kernel(SiteWord* words, const DeviceLayout layout,
                                              const InverseScratch* scratch) {
  if (scratch == nullptr || scratch->valid == 0u)
    return;
  const std::uint32_t slot = scratch->journal_slot;
  const std::uint32_t event_slot = scratch->event_slot;
  const std::uint32_t stride = blockDim.x * gridDim.x;
  for (std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x; index < kJournalRecordRailCount;
       index += stride) {
    const SiteWord value = words[fixed_physical_slot(journal_index(slot, index))];
    if (index >= kJournalGlobalOffset && index < kJournalStateOffset) {
      words[fixed_physical_slot(index - kJournalGlobalOffset)] = value;
    } else if (index >= kJournalStateOffset && index < kJournalEligibilityOffset) {
      words[fixed_physical_slot(kStateBase + index - kJournalStateOffset)] = value;
    } else if (index >= kJournalEligibilityOffset && index < kJournalLedgerOffset) {
      words[fixed_physical_slot(
          pair_index(kEligibilityBase, event_slot * kEligibilityWordsPerEvent, 0u) + index -
          kJournalEligibilityOffset)] = value;
    } else if (index >= kJournalLedgerOffset && index < kJournalWeightOffset) {
      words[fixed_physical_slot(pair_index(kLedgerBase, event_slot * kLedgerFields, 0u) + index -
                                kJournalLedgerOffset)] = value;
    } else if (index >= kJournalWeightOffset && index < kJournalMotorOffset) {
      words[fixed_physical_slot(kContentBase + index - kJournalWeightOffset)] = value;
    } else if (index == kJournalMotorOffset) {
      words[layout.raw_motor_zero_slot] = value;
    } else if (index == kJournalMotorOffset + 2u) {
      words[layout.raw_motor_one_slot] = value;
    }
    words[fixed_physical_slot(journal_index(slot, index))] =
        founder_value(journal_index(slot, index));
  }
}

static __global__ void validate_resident_kernel(const SiteWord* words, ValidationReceipt* receipt) {
  if (receipt == nullptr)
    return;
  if (blockIdx.x == 0u && threadIdx.x == 0u) {
    receipt->marker_valid =
        read_resident(words, pair_index(kGlobalBase, kFactorMarker, 0u)) == kFactorMarkerValue ? 1u
                                                                                               : 0u;
    receipt->version_valid =
        read_resident(words, pair_index(kGlobalBase, kLayoutVersion, 0u)) == kLayoutVersionValue
            ? 1u
            : 0u;
    receipt->journal_valid =
        read_resident(words, pair_index(kGlobalBase, kJournalCount, 0u)) <= kJournalDepth ? 1u : 0u;
    receipt->active_bank_valid =
        read_resident(words, pair_index(kGlobalBase, kActiveBank, 0u)) <= 1u ? 1u : 0u;
    const std::uint32_t event_id =
        static_cast<std::uint32_t>(read_resident(words, pair_index(kGlobalBase, kEventId, 0u)));
    const std::uint32_t credit_head =
        static_cast<std::uint32_t>(read_resident(words, pair_index(kGlobalBase, kCreditHead, 0u)));
    const std::uint32_t prediction_count = static_cast<std::uint32_t>(
        read_resident(words, pair_index(kGlobalBase, kPredictionCount, 0u)));
    const std::uint32_t credit_count =
        static_cast<std::uint32_t>(read_resident(words, pair_index(kGlobalBase, kCreditCount, 0u)));
    const std::uint32_t expired_count = static_cast<std::uint32_t>(
        read_resident(words, pair_index(kGlobalBase, kExpiredCount, 0u)));
    const std::uint32_t active_bank =
        static_cast<std::uint32_t>(read_resident(words, pair_index(kGlobalBase, kActiveBank, 0u)));
    receipt->fifo_valid = credit_head <= event_id && prediction_count == event_id &&
                                  active_bank == (prediction_count & 1u) &&
                                  credit_count + expired_count == credit_head
                              ? 1u
                              : 0u;
    const std::uint32_t represented = event_id < kEligibilityDepth ? event_id : kEligibilityDepth;
    for (std::uint32_t offset = 0u; offset < represented; ++offset) {
      const std::uint32_t event = event_id - offset;
      const std::uint32_t slot = event & (kEligibilityDepth - 1u);
      if (read_resident(words, ledger_index(slot, 0u, 0u)) != event)
        ++receipt->invalid_ledger;
    }
  }
  const std::uint32_t journal_count =
      static_cast<std::uint32_t>(read_resident(words, pair_index(kGlobalBase, kJournalCount, 0u)));
  const std::uint32_t validated_rails =
      kMutableRailCount +
      (journal_count <= kJournalDepth ? journal_count * kJournalRecordRailCount : 0u);
  for (std::uint32_t pair = blockIdx.x * blockDim.x + threadIdx.x; pair < validated_rails / 2u;
       pair += blockDim.x * gridDim.x) {
    const SiteWord value = words[fixed_physical_slot(pair * 2u)];
    const SiteWord complement = words[fixed_physical_slot(pair * 2u + 1u)];
    if (complement != ~value)
      atomicAdd(&receipt->invalid_pairs, 1u);
  }
  for (std::uint32_t slot = blockIdx.x * blockDim.x + threadIdx.x; slot < journal_count;
       slot += blockDim.x * gridDim.x) {
    if (words[fixed_physical_slot(journal_index(slot, 0u))] >= kEligibilityDepth)
      atomicAdd(&receipt->invalid_journal_events, 1u);
  }
  const std::uint32_t trace_words = kEligibilityDepth * kEligibilityWordsPerEvent;
  for (std::uint32_t item = blockIdx.x * blockDim.x + threadIdx.x; item < trace_words;
       item += blockDim.x * gridDim.x) {
    const std::int32_t trace = read_signed(
        words, pair_index(kEligibilityBase, item, 0u));
    if (trace > kTraceLimit || trace < -kTraceLimit)
      atomicAdd(&receipt->invalid_eligibility, 1u);
  }
}

// This file is a collection of free launch functions (no owning class), so
// the lazy graph-dispatch idiom used by the class-based sibling factors is
// adapted here with function-local static graph state instead of member
// state: each single-thread launch site gets its own KernelGraph, created via
// stream capture on first call and updated via cudaGraphExecKernelNodeSetParams
// on every later call. Multi-thread launches (more than one block/thread) are
// left as direct launches; this is a launch-mechanism change only.
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

inline void require_cuda(cudaError_t status, const char* operation) {
  if (status != cudaSuccess)
    throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
}

template <typename T>
inline void* kernel_argument(T& value) {
  return const_cast<void*>(static_cast<const void*>(&value));
}

inline void launch_graph(KernelGraph& graph, const char* operation, void* function,
                         dim3 grid, dim3 block, void** arguments) {
  cudaKernelNodeParams params{};
  params.func = function;
  params.gridDim = grid;
  params.blockDim = block;
  params.kernelParams = arguments;
  if (graph.executable == nullptr) {
    require_cuda(cudaGraphCreate(&graph.graph, 0u), operation);
    require_cuda(
        cudaGraphAddKernelNode(&graph.node, graph.graph, nullptr, 0u, &params), operation);
    require_cuda(
        cudaGraphInstantiate(&graph.executable, graph.graph, nullptr, nullptr, 0u), operation);
  } else {
    require_cuda(
        cudaGraphExecKernelNodeSetParams(graph.executable, graph.node, &params), operation);
  }
  require_cuda(cudaGraphLaunch(graph.executable, 0), operation);
}

inline void launch_step(SiteWord* words, const DeviceLayout& layout, DeviceInputs* inputs,
                        DeviceScratch* scratch, Receipt* receipt, std::uint32_t* advanced,
                        std::uint32_t advancement_bit) {
  static KernelGraph reserve_step_graph;
  static KernelGraph commit_step_graph;
  static KernelGraph publish_graph;
  static KernelGraph reserve_credit_graph;
  static KernelGraph finalize_credit_graph;
  static KernelGraph finalize_step_graph;

  void* reserve_step_arguments[] = {
      kernel_argument(words), kernel_argument(layout), kernel_argument(inputs),
      kernel_argument(scratch), kernel_argument(receipt), kernel_argument(advanced)};
  launch_graph(reserve_step_graph, "launch resident recurrent reserve-step graph",
              reinterpret_cast<void*>(reserve_step_kernel), dim3(1u), dim3(1u),
              reserve_step_arguments);

  snapshot_step_kernel<<<64, 256>>>(words, layout, scratch);

  void* commit_step_arguments[] = {kernel_argument(words), kernel_argument(inputs),
                                   kernel_argument(scratch), kernel_argument(receipt)};
  launch_graph(commit_step_graph, "launch resident recurrent commit-step graph",
              reinterpret_cast<void*>(commit_step_kernel), dim3(1u), dim3(1u),
              commit_step_arguments);

  clear_event_eligibility_kernel<<<8, 256>>>(words, inputs, scratch);
  state_step_kernel<<<2, 256>>>(words, layout, inputs, scratch);
  output_step_kernel<<<2, 256>>>(words, inputs, scratch);

  void* publish_arguments[] = {kernel_argument(words), kernel_argument(layout),
                               kernel_argument(inputs), kernel_argument(scratch),
                               kernel_argument(receipt)};
  launch_graph(publish_graph, "launch resident recurrent publish graph",
              reinterpret_cast<void*>(publish_kernel), dim3(1u), dim3(1u),
              publish_arguments);

  void* reserve_credit_arguments[] = {kernel_argument(words), kernel_argument(inputs),
                                      kernel_argument(scratch)};
  launch_graph(reserve_credit_graph, "launch resident recurrent reserve-credit graph",
              reinterpret_cast<void*>(reserve_credit_kernel), dim3(1u), dim3(1u),
              reserve_credit_arguments);

  credit_signal_kernel<<<2, 256>>>(words, inputs, scratch, receipt);
  apply_credit_kernel<<<2, 256>>>(words, inputs, scratch, receipt);

  void* finalize_credit_arguments[] = {kernel_argument(words), kernel_argument(inputs),
                                       kernel_argument(scratch), kernel_argument(receipt)};
  launch_graph(finalize_credit_graph, "launch resident recurrent finalize-credit graph",
              reinterpret_cast<void*>(finalize_credit_kernel), dim3(1u), dim3(1u),
              finalize_credit_arguments);

  void* finalize_step_arguments[] = {
      kernel_argument(words), kernel_argument(inputs), kernel_argument(receipt),
      kernel_argument(advanced), kernel_argument(advancement_bit)};
  launch_graph(finalize_step_graph, "launch resident recurrent finalize-step graph",
              reinterpret_cast<void*>(finalize_step_kernel), dim3(1u), dim3(1u),
              finalize_step_arguments);
}

inline void launch_prepare_inverse(SiteWord* words, InverseScratch* scratch) {
  static KernelGraph prepare_inverse_graph;
  void* prepare_inverse_arguments[] = {kernel_argument(words), kernel_argument(scratch)};
  launch_graph(prepare_inverse_graph, "launch resident recurrent prepare-inverse graph",
              reinterpret_cast<void*>(prepare_inverse_kernel), dim3(1u), dim3(1u),
              prepare_inverse_arguments);
}

inline void launch_restore_inverse(SiteWord* words, const DeviceLayout& layout,
                                   const InverseScratch* scratch) {
  restore_inverse_kernel<<<64, 256>>>(words, layout, scratch);
}

inline void launch_validate(const SiteWord* words, ValidationReceipt* receipt) {
  validate_resident_kernel<<<64, 256>>>(words, receipt);
}

inline ValidationReceipt validate_resident(const SiteWord* words) {
  auto require = [](cudaError_t status, const char* operation) {
    if (status != cudaSuccess)
      throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
  };
  ValidationReceipt* device = nullptr;
  require(cudaMalloc(&device, sizeof(*device)), "allocate resident recurrent validation");
  require(cudaMemset(device, 0, sizeof(*device)), "clear resident recurrent validation");
  launch_validate(words, device);
  require(cudaGetLastError(), "launch resident recurrent validation");
  ValidationReceipt host{};
  require(cudaMemcpy(&host, device, sizeof(host), cudaMemcpyDeviceToHost),
          "copy resident recurrent validation");
  (void)cudaFree(device);
  return host;
}

}  // namespace substrate::bcc32::grown_selective_state_space
