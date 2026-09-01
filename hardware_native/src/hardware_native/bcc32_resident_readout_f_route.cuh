#pragma once

#include <cuda_runtime.h>

#include <cstdint>

#include "hardware_native/bcc32_grown_cloud_factor.cuh"
#include "hardware_native/bcc32_law.cuh"
#include "hardware_native/bcc32_resident_readout_plastic_owner.cuh"

namespace substrate::bcc32::resident_readout_f_route {

namespace cloud = grown_cloud_factor;
namespace predecessor = ::bcc32::resident_readout_plastic_owner;

inline constexpr std::uint32_t kRouteCapacity = 256u;
inline constexpr std::uint32_t kRouteFields = 7u;
inline constexpr std::uint32_t kSupplyWords = 8u;
inline constexpr std::uint32_t kEventDepth = 16u;
inline constexpr std::uint32_t kJournalCapacity = 512u;
inline constexpr std::uint32_t kJournalFields = 12u;
inline constexpr std::uint32_t kTopK = predecessor::kTopK;
// Preserve one eligibility lane for every bounded motor phase. The world may
// return a delayed residual to any still-live phase without a host-selected
// route or an immediate-only credit shortcut.
inline constexpr std::uint32_t kHorizon = 80u;
inline constexpr std::uint32_t kMaxReceiptRoutes =
    predecessor::kMaxReceiptRoutes;
inline constexpr std::int32_t kPlasticQuantumQ8 =
    predecessor::kPlasticQuantumQ8;
inline constexpr SiteWord kFactorMarkerValue = 0x71f04a2du;

enum RouteField : std::uint32_t {
  kRouteMeta,
  kRouteContext,
  kRouteKey,
  kRoutePositive,
  kRouteNegative,
  kRouteLesionPositive,
  kRouteLesionNegative,
};

enum GlobalField : std::uint32_t {
  kFactorMarker,
  kPositiveSupplyBase,
  kNegativeSupplyBase = kPositiveSupplyBase + kSupplyWords,
  kStateEpoch = kNegativeSupplyBase + kSupplyWords,
  kCommittedTransactions,
  kEventCount,
  kJournalCount,
  kGlobalFieldCount,
};

enum EventField : std::uint32_t {
  kEventJournalStart,
  kEventJournalSize,
  kEventEpochBefore,
  kEventCommittedBefore,
  kEventFieldCount,
};

enum JournalField : std::uint32_t {
  kJournalRoute,
  kJournalSupply,
  kJournalDirection,
  kJournalSupplyBefore,
  kJournalRouteBeforeBase,
};

inline constexpr std::uint32_t kRoutePhysicalRails =
    kRouteCapacity * kRouteFields * 2u;
inline constexpr std::uint32_t kGlobalPhysicalBase = kRoutePhysicalRails;
inline constexpr std::uint32_t kGlobalPhysicalRails =
    kGlobalFieldCount * 2u;
inline constexpr std::uint32_t kEventPhysicalBase =
    kGlobalPhysicalBase + kGlobalPhysicalRails;
inline constexpr std::uint32_t kEventPhysicalRails =
    kEventDepth * kEventFieldCount * 2u;
inline constexpr std::uint32_t kJournalPhysicalBase =
    kEventPhysicalBase + kEventPhysicalRails;
inline constexpr std::uint32_t kJournalPhysicalRails =
    kJournalCapacity * kJournalFields * 2u;
inline constexpr std::uint32_t kPhysicalRailCount =
    kJournalPhysicalBase + kJournalPhysicalRails;

inline constexpr SiteWord kMetaOccupied = 1u << 0u;
inline constexpr SiteWord kMetaEnabled = 1u << 1u;
inline constexpr std::uint32_t kMetaBaselineShift = 8u;
inline constexpr SiteWord kMetaBaselineMask = 0xffu << kMetaBaselineShift;
inline constexpr std::uint32_t kMetaGenerationShift = 16u;

enum class Code : std::uint32_t {
  kEmpty = 0u,
  kCommitted = 1u,
  kMatchedResidual = 2u,
  kMalformedRails = 3u,
  kNoPrediction = 4u,
  kWrongTick = 5u,
  kExpired = 6u,
  kNoSupply = 7u,
  kInvalid = 8u,
  kNoContext = 9u,
  kContextMismatch = 10u,
  kJournalFull = 11u,
};

struct PhysicalOffset {
  std::int32_t x = 0;
  std::int32_t y = 0;
  std::int32_t z = 0;
};

struct DeviceLayout {
  std::uint64_t rails[kPhysicalRailCount]{};
};

struct CaptureView {
  predecessor::PredictionLane* lanes = nullptr;
  std::uint64_t* next_tick = nullptr;
};

struct DeviceView {
  SiteWord* words = nullptr;
  const DeviceLayout* layout = nullptr;
  CaptureView capture{};
};

template <typename Witness>
struct RawObservationT {
  std::uint32_t positive = 0u;
  std::uint32_t negative = 0u;
  std::uint64_t tick = 0u;
  // Zero preserves the latest-eligible-lane contract. A nonzero value names
  // the earlier motor prediction graded by the honestly later arrival tick.
  std::uint64_t target_tick = 0u;
  const Witness* m6_prediction = nullptr;
};

using RawObservation = RawObservationT<cloud::PredictionWitness>;

struct DeviceInputs {
  CaptureView capture{};
  const RawObservation* observation = nullptr;
  std::uint32_t staged = 0u;
};

struct CreditReceipt {
  Code code = Code::kEmpty;
  std::uint32_t requested = 0u;
  std::uint32_t committed = 0u;
  std::uint32_t route_count = 0u;
  std::uint32_t route_ids[kMaxReceiptRoutes]{};
  std::int8_t route_directions[kMaxReceiptRoutes]{};
  std::uint64_t prediction_tick = 0u;
  std::uint64_t observation_tick = 0u;
  std::uint32_t predicted_positive = 0u;
  std::uint32_t observed_positive = 0u;
  std::uint32_t matter_before = 0u;
  std::uint32_t matter_after = 0u;
  std::uint32_t resource_before = 0u;
  std::uint32_t resource_after = 0u;
  std::uint64_t state_hash_before = 0u;
  std::uint64_t state_hash_after = 0u;
};

struct LesionReceipt {
  std::uint32_t applied = 0u;
  std::uint32_t route_count = 0u;
  std::uint32_t matter_before = 0u;
  std::uint32_t matter_after = 0u;
  std::uint32_t resource_before = 0u;
  std::uint32_t resource_after = 0u;
};

__host__ __device__ inline std::uint32_t route_rail(
    std::uint32_t route, std::uint32_t field, std::uint32_t polarity) {
  return (route * kRouteFields + field) * 2u + polarity;
}

__host__ __device__ inline std::uint32_t global_rail(
    std::uint32_t field, std::uint32_t polarity) {
  return kGlobalPhysicalBase + field * 2u + polarity;
}

__host__ __device__ inline std::uint32_t event_rail(
    std::uint32_t event, std::uint32_t field, std::uint32_t polarity) {
  return kEventPhysicalBase +
         (event * kEventFieldCount + field) * 2u + polarity;
}

__host__ __device__ inline std::uint32_t journal_rail(
    std::uint32_t entry, std::uint32_t field, std::uint32_t polarity) {
  return kJournalPhysicalBase +
         (entry * kJournalFields + field) * 2u + polarity;
}

__host__ __device__ inline PhysicalOffset physical_offset(
    std::uint32_t index) {
  return {-155 + static_cast<std::int32_t>(index % 50u),
          -155 + static_cast<std::int32_t>((index / 50u) % 50u),
          -155 + static_cast<std::int32_t>(index / 2500u)};
}

__host__ __device__ inline SiteWord initial_value(std::uint32_t rail) {
  if ((rail & 1u) != 0u) return ~initial_value(rail - 1u);
  if (rail == global_rail(kFactorMarker, 0u))
    return kFactorMarkerValue;
  for (std::uint32_t index = 0u; index < kSupplyWords; ++index) {
    if (rail == global_rail(kPositiveSupplyBase + index, 0u) ||
        rail == global_rail(kNegativeSupplyBase + index, 0u))
      return 0xffffffffu;
  }
  return 0u;
}

__device__ inline SiteWord read_pair(const SiteWord* words,
                                     const DeviceLayout& layout,
                                     std::uint32_t rail) {
  return words[layout.rails[rail]];
}

__device__ inline void write_pair(SiteWord* words,
                                  const DeviceLayout& layout,
                                  std::uint32_t rail, SiteWord value) {
  words[layout.rails[rail]] = value;
  words[layout.rails[rail + 1u]] = ~value;
}

__device__ inline SiteWord read_route(const SiteWord* words,
                                      const DeviceLayout& layout,
                                      std::uint32_t route,
                                      RouteField field) {
  return read_pair(words, layout, route_rail(route, field, 0u));
}

__device__ inline void write_route(SiteWord* words,
                                   const DeviceLayout& layout,
                                   std::uint32_t route, RouteField field,
                                   SiteWord value) {
  write_pair(words, layout, route_rail(route, field, 0u), value);
}

__device__ inline SiteWord read_global(const SiteWord* words,
                                       const DeviceLayout& layout,
                                       GlobalField field) {
  return read_pair(words, layout, global_rail(field, 0u));
}

__device__ inline void write_global(SiteWord* words,
                                    const DeviceLayout& layout,
                                    GlobalField field, SiteWord value) {
  write_pair(words, layout, global_rail(field, 0u), value);
}

__device__ inline SiteWord read_event(const SiteWord* words,
                                      const DeviceLayout& layout,
                                      std::uint32_t event,
                                      EventField field) {
  return read_pair(words, layout, event_rail(event, field, 0u));
}

__device__ inline void write_event(SiteWord* words,
                                   const DeviceLayout& layout,
                                   std::uint32_t event, EventField field,
                                   SiteWord value) {
  write_pair(words, layout, event_rail(event, field, 0u), value);
}

__device__ inline SiteWord read_journal(const SiteWord* words,
                                        const DeviceLayout& layout,
                                        std::uint32_t entry,
                                        std::uint32_t field) {
  return read_pair(words, layout, journal_rail(entry, field, 0u));
}

__device__ inline void write_journal(SiteWord* words,
                                     const DeviceLayout& layout,
                                     std::uint32_t entry,
                                     std::uint32_t field, SiteWord value) {
  write_pair(words, layout, journal_rail(entry, field, 0u), value);
}

__device__ inline bool valid_layout(const SiteWord* words,
                                    const DeviceLayout* layout) {
  return words != nullptr && layout != nullptr &&
         read_global(words, *layout, kFactorMarker) == kFactorMarkerValue;
}

__device__ inline bool valid_capture(CaptureView view) {
  return view.lanes != nullptr && view.next_tick != nullptr;
}

__device__ inline std::uint32_t baseline_from_meta(SiteWord meta) {
  return (meta & kMetaBaselineMask) >> kMetaBaselineShift;
}

__device__ inline SiteWord make_meta(std::uint8_t baseline,
                                     std::uint32_t generation = 0u) {
  return kMetaOccupied | kMetaEnabled |
         (static_cast<SiteWord>(baseline) << kMetaBaselineShift) |
         (generation << kMetaGenerationShift);
}

__device__ inline SiteWord make_key(std::uint32_t row,
                                    std::uint32_t column,
                                    std::uint32_t output_phase) {
  return (row & 0x1fu) | ((column & 0x1ffu) << 5u) |
         ((output_phase & 0x7fu) << 14u);
}

__device__ inline std::uint32_t key_row(SiteWord key) {
  return key & 0x1fu;
}

__device__ inline std::uint32_t key_column(SiteWord key) {
  return (key >> 5u) & 0x1ffu;
}

__device__ inline std::uint32_t key_phase(SiteWord key) {
  return (key >> 14u) & 0x7fu;
}

__device__ inline std::uint32_t matter_bits(const SiteWord* words,
                                            const DeviceLayout& layout) {
  std::uint32_t total = 0u;
  for (std::uint32_t rail = 0u; rail < kPhysicalRailCount; ++rail)
    total += __popc(words[layout.rails[rail]]);
  return total;
}

__device__ inline std::uint32_t resource_bits(const SiteWord* words,
                                              const DeviceLayout& layout) {
  std::uint32_t total = 0u;
  for (std::uint32_t index = 0u; index < kSupplyWords; ++index) {
    total += __popc(read_global(
        words, layout,
        static_cast<GlobalField>(kPositiveSupplyBase + index)));
    total += __popc(read_global(
        words, layout,
        static_cast<GlobalField>(kNegativeSupplyBase + index)));
  }
  for (std::uint32_t route = 0u; route < kRouteCapacity; ++route) {
    total += __popc(read_route(words, layout, route, kRoutePositive));
    total += __popc(read_route(words, layout, route, kRouteNegative));
    total += __popc(read_route(words, layout, route, kRouteLesionPositive));
    total += __popc(read_route(words, layout, route, kRouteLesionNegative));
  }
  return total;
}

__device__ inline std::uint64_t mix_hash(std::uint64_t hash,
                                         std::uint64_t value) {
  constexpr std::uint64_t kPrime = UINT64_C(1099511628211);
  for (std::uint32_t byte = 0u; byte < 8u; ++byte) {
    hash ^= static_cast<std::uint8_t>(value >> (byte * 8u));
    hash *= kPrime;
  }
  return hash;
}

__device__ inline std::uint64_t durable_state_hash(
    const SiteWord* words, const DeviceLayout& layout) {
  std::uint64_t hash = UINT64_C(1469598103934665603);
  for (std::uint32_t route = 0u; route < kRouteCapacity; ++route) {
    for (std::uint32_t field = 0u; field < kRouteFields; ++field) {
      hash = mix_hash(
          hash, read_route(words, layout, route,
                           static_cast<RouteField>(field)));
    }
  }
  for (std::uint32_t field = kPositiveSupplyBase;
       field < kGlobalFieldCount; ++field) {
    hash = mix_hash(
        hash, read_global(words, layout, static_cast<GlobalField>(field)));
  }
  return hash;
}

template <typename Witness>
__device__ inline bool valid_witness(const Witness* witness) {
  return witness != nullptr && witness->valid != 0u &&
         witness->predicted_signature != 0u;
}

template <typename Witness>
__device__ inline std::uint32_t witness_signature(const Witness* witness) {
  return witness == nullptr ? 0u : witness->predicted_signature;
}

template <typename Witness>
__device__ bool capture_prediction_device(
    CaptureView view, const float* hidden, const float* pair_scores,
    const std::uint16_t* selected_rows, const Witness* m6_prediction,
    std::uint8_t emitted_byte, std::uint32_t output_phase) {
  if (!valid_capture(view) || hidden == nullptr || pair_scores == nullptr ||
      selected_rows == nullptr || !valid_witness(m6_prediction))
    return false;
  const std::uint64_t tick =
      atomicAdd(reinterpret_cast<unsigned long long*>(view.next_tick), 1ULL);
  predecessor::PredictionLane& lane = view.lanes[tick % kHorizon];
  lane = predecessor::PredictionLane{};
  lane.tick = tick;
  lane.valid = 1u;
  lane.output_phase = output_phase;
  lane.context_signature = witness_signature(m6_prediction);
  lane.baseline_byte = predecessor::decode_baseline_byte(pair_scores);
  lane.emitted_byte = emitted_byte;
  for (std::uint32_t row = 0u; row < predecessor::kOutputRows; ++row)
    lane.selected_rows[row] = selected_rows[row];
  for (std::uint32_t column = 0u;
       column < predecessor::kHiddenColumns; ++column)
    lane.hidden_prototype[column] =
        predecessor::quantize_hidden(hidden[column]);
  return lane.baseline_byte != 0xffu;
}

template <typename Witness>
__device__ float score_correction(const DeviceView& view,
                                  const Witness* m6, const float* hidden,
                                  const float* pair_scores,
                                  std::uint32_t output_row,
                                  std::uint32_t output_phase) {
  if (!valid_layout(view.words, view.layout) || !valid_witness(m6) ||
      hidden == nullptr || pair_scores == nullptr ||
      output_row >= predecessor::kOutputRows)
    return 0.0f;
  const std::uint8_t baseline =
      predecessor::decode_baseline_byte(pair_scores);
  if (baseline == 0xffu) return 0.0f;
  const std::uint32_t signature = witness_signature(m6);
  float sum = 0.0f;
  for (std::uint32_t route = 0u; route < kRouteCapacity; ++route) {
    const SiteWord meta =
        read_route(view.words, *view.layout, route, kRouteMeta);
    if ((meta & (kMetaOccupied | kMetaEnabled)) !=
            (kMetaOccupied | kMetaEnabled) ||
        baseline_from_meta(meta) != baseline ||
        read_route(view.words, *view.layout, route, kRouteContext) !=
            signature)
      continue;
    const SiteWord key =
        read_route(view.words, *view.layout, route, kRouteKey);
    if (key_row(key) != output_row || key_phase(key) != output_phase) continue;
    const std::uint32_t column = key_column(key);
    if (column >= predecessor::kHiddenColumns) continue;
    const std::int32_t quanta = static_cast<std::int32_t>(__popc(
                                     read_route(view.words, *view.layout, route,
                                                kRoutePositive))) -
                                static_cast<std::int32_t>(__popc(
                                     read_route(view.words, *view.layout, route,
                                                kRouteNegative)));
    sum += static_cast<float>(quanta * kPlasticQuantumQ8) * hidden[column];
  }
  return sum / 256.0f;
}

__device__ inline bool valid_rails(const RawObservation& observation) {
  const std::uint32_t positive =
      observation.positive & predecessor::kByteMask;
  const std::uint32_t negative =
      observation.negative & predecessor::kByteMask;
  return ((observation.positive | observation.negative) &
          ~predecessor::kByteMask) == 0u &&
         (positive | negative) == predecessor::kByteMask &&
         (positive & negative) == 0u;
}

__device__ inline bool journal_available(const SiteWord* words,
                                         const DeviceLayout& layout) {
  return read_global(words, layout, kEventCount) < kEventDepth &&
         read_global(words, layout, kJournalCount) +
                 kMaxReceiptRoutes <=
             kJournalCapacity;
}

__device__ inline bool begin_event(SiteWord* words,
                                   const DeviceLayout& layout) {
  if (!journal_available(words, layout)) return false;
  const std::uint32_t event =
      read_global(words, layout, kEventCount);
  write_event(words, layout, event, kEventJournalStart,
              read_global(words, layout, kJournalCount));
  write_event(words, layout, event, kEventJournalSize, 0u);
  write_event(words, layout, event, kEventEpochBefore,
              read_global(words, layout, kStateEpoch));
  write_event(words, layout, event, kEventCommittedBefore,
              read_global(words, layout, kCommittedTransactions));
  write_global(words, layout, kEventCount, event + 1u);
  return true;
}

__device__ inline void append_journal(
    SiteWord* words, const DeviceLayout& layout, std::uint32_t route,
    std::uint32_t supply, std::int32_t direction, SiteWord supply_before) {
  const std::uint32_t event =
      read_global(words, layout, kEventCount) - 1u;
  const std::uint32_t entry =
      read_global(words, layout, kJournalCount);
  write_journal(words, layout, entry, kJournalRoute, route);
  write_journal(words, layout, entry, kJournalSupply, supply);
  write_journal(words, layout, entry, kJournalDirection,
                static_cast<SiteWord>(direction));
  write_journal(words, layout, entry, kJournalSupplyBefore, supply_before);
  for (std::uint32_t field = 0u; field < kRouteFields; ++field) {
    write_journal(
        words, layout, entry, kJournalRouteBeforeBase + field,
        read_route(words, layout, route, static_cast<RouteField>(field)));
  }
  write_global(words, layout, kJournalCount, entry + 1u);
  write_event(words, layout, event, kEventJournalSize,
              read_event(words, layout, event, kEventJournalSize) + 1u);
}

__device__ inline std::int32_t find_or_empty_route(
    const SiteWord* words, const DeviceLayout& layout,
    std::uint32_t context, std::uint8_t baseline, SiteWord key,
    std::int32_t avoid = -1) {
  std::int32_t empty = -1;
  for (std::uint32_t route = 0u; route < kRouteCapacity; ++route) {
    const SiteWord meta = read_route(words, layout, route, kRouteMeta);
    if ((meta & kMetaOccupied) == 0u) {
      if (empty < 0 && static_cast<std::int32_t>(route) != avoid)
        empty = static_cast<std::int32_t>(route);
      continue;
    }
    if (baseline_from_meta(meta) == baseline &&
        read_route(words, layout, route, kRouteContext) == context &&
        read_route(words, layout, route, kRouteKey) == key)
      return static_cast<std::int32_t>(route);
  }
  return empty;
}

struct QuantumPlan {
  std::uint32_t route = 0u;
  std::uint32_t supply_index = 0u;
  SiteWord supply_before = 0u;
  SiteWord quantum = 0u;
  SiteWord key = 0u;
  std::int32_t direction = 0;
  std::uint32_t create = 0u;
  std::uint32_t valid = 0u;
};

__device__ inline QuantumPlan plan_supply_quantum(
    const SiteWord* words, const DeviceLayout& layout,
    std::uint32_t context, std::uint8_t baseline, SiteWord key,
    std::int32_t direction, std::int32_t avoid = -1) {
  QuantumPlan plan{};
  const std::int32_t route_index =
      find_or_empty_route(words, layout, context, baseline, key, avoid);
  if (route_index < 0) return plan;
  plan.route = static_cast<std::uint32_t>(route_index);
  plan.key = key;
  plan.direction = direction;
  plan.create =
      (read_route(words, layout, plan.route, kRouteMeta) & kMetaOccupied) == 0u;
  const RouteField population_field =
      direction > 0 ? kRoutePositive : kRouteNegative;
  const GlobalField supply_base =
      direction > 0 ? kPositiveSupplyBase : kNegativeSupplyBase;
  SiteWord population =
      read_route(words, layout, plan.route, population_field);
  for (std::uint32_t supply_index = 0u;
       supply_index < kSupplyWords; ++supply_index) {
    const GlobalField supply_field = static_cast<GlobalField>(
        static_cast<std::uint32_t>(supply_base) + supply_index);
    const SiteWord supply = read_global(words, layout, supply_field);
    const SiteWord available = supply & ~population;
    if (available == 0u) continue;
    plan.supply_index = supply_index;
    plan.supply_before = supply;
    plan.quantum = available & (0u - available);
    plan.valid = 1u;
    return plan;
  }
  return plan;
}

__device__ inline void commit_supply_quantum(
    SiteWord* words, const DeviceLayout& layout, const QuantumPlan& plan,
    std::uint32_t context, std::uint8_t baseline, CreditReceipt* receipt) {
  const RouteField population_field =
      plan.direction > 0 ? kRoutePositive : kRouteNegative;
  const GlobalField supply_field = static_cast<GlobalField>(
      (plan.direction > 0 ? kPositiveSupplyBase : kNegativeSupplyBase) +
      plan.supply_index);
  append_journal(words, layout, plan.route, plan.supply_index,
                 plan.direction, plan.supply_before);
  if (plan.create != 0u) {
    write_route(words, layout, plan.route, kRouteMeta, make_meta(baseline));
    write_route(words, layout, plan.route, kRouteContext, context);
    write_route(words, layout, plan.route, kRouteKey, plan.key);
  }
  write_global(words, layout, supply_field,
               plan.supply_before & ~plan.quantum);
  write_route(
      words, layout, plan.route, population_field,
      read_route(words, layout, plan.route, population_field) | plan.quantum);
  const std::uint32_t slot = receipt->route_count++;
  receipt->route_ids[slot] = plan.route;
  receipt->route_directions[slot] =
      static_cast<std::int8_t>(plan.direction);
}

__device__ inline bool apply_credit(
    SiteWord* words, const DeviceLayout& layout, DeviceInputs* inputs,
    CreditReceipt* receipt) {
  CreditReceipt local{};
  local.matter_before = matter_bits(words, layout);
  local.resource_before = resource_bits(words, layout);
  local.state_hash_before = durable_state_hash(words, layout);
  if (!begin_event(words, layout)) {
    local.code = Code::kJournalFull;
    if (receipt != nullptr) *receipt = local;
    return false;
  }
  if (inputs == nullptr || inputs->staged == 0u ||
      inputs->observation == nullptr || !valid_capture(inputs->capture)) {
    local.code = Code::kEmpty;
    local.matter_after = matter_bits(words, layout);
    local.resource_after = resource_bits(words, layout);
    local.state_hash_after = durable_state_hash(words, layout);
    if (receipt != nullptr) *receipt = local;
    return true;
  }
  const RawObservation observed = *inputs->observation;
  local.observation_tick = observed.tick;
  local.observed_positive =
      observed.positive & predecessor::kByteMask;
  if (!valid_rails(observed)) {
    local.code = Code::kMalformedRails;
    inputs->staged = 0u;
  } else if (!valid_witness(observed.m6_prediction)) {
    local.code = Code::kNoContext;
    inputs->staged = 0u;
  } else {
    std::int32_t lane_index = -1;
    const std::uint64_t now = *inputs->capture.next_tick;
    for (std::uint32_t lane = 0u; lane < kHorizon; ++lane) {
      const predecessor::PredictionLane& prediction =
          inputs->capture.lanes[lane];
      if (prediction.valid == 0u || prediction.tick >= observed.tick ||
          (observed.target_tick != 0u &&
           prediction.tick != observed.target_tick))
        continue;
      const std::uint64_t age = observed.tick - prediction.tick;
      if (age <= kHorizon && observed.tick <= now &&
          (lane_index < 0 ||
           prediction.tick >
               inputs->capture.lanes[lane_index].tick))
        lane_index = static_cast<std::int32_t>(lane);
    }
    if (lane_index < 0) {
      local.code = observed.tick > now ? Code::kNoPrediction
                                       : Code::kExpired;
      inputs->staged = 0u;
    } else {
      predecessor::PredictionLane& prediction =
          inputs->capture.lanes[lane_index];
      local.prediction_tick = prediction.tick;
      local.predicted_positive = prediction.emitted_byte;
      if (prediction.context_signature !=
          witness_signature(observed.m6_prediction)) {
        local.code = Code::kContextMismatch;
      } else {
        const std::uint32_t residual =
            (prediction.emitted_byte ^ local.observed_positive) &
            predecessor::kByteMask;
        local.requested = __popc(residual);
        if (residual == 0u) {
          local.code = Code::kMatchedResidual;
        } else {
          for (std::uint32_t bit = 0u; bit < 8u; ++bit) {
            if ((residual & (1u << bit)) == 0u) continue;
            const std::uint32_t low = 2u + 2u * bit;
            const bool observed_one =
                (local.observed_positive & (1u << bit)) != 0u;
            const std::uint32_t desired_row =
                prediction.selected_rows[low + (observed_one ? 1u : 0u)];
            const std::uint32_t rejected_row =
                prediction.selected_rows[low + (observed_one ? 0u : 1u)];
            for (std::uint32_t k = 0u; k < kTopK; ++k) {
              const std::uint32_t column =
                  predecessor::eligible_column(prediction, k);
              if (column >= predecessor::kHiddenColumns ||
                  local.route_count + 2u > kMaxReceiptRoutes)
                continue;
              const std::int32_t hidden_sign =
                  prediction.hidden_prototype[column] > 0 ? 1 : -1;
              const std::uint32_t context = static_cast<std::uint32_t>(
                  prediction.context_signature);
              const QuantumPlan desired = plan_supply_quantum(
                  words, layout, context, prediction.baseline_byte,
                  make_key(desired_row, column, prediction.output_phase),
                  hidden_sign);
              if (desired.valid == 0u) continue;
              const QuantumPlan rejected = plan_supply_quantum(
                  words, layout, context, prediction.baseline_byte,
                  make_key(rejected_row, column, prediction.output_phase),
                  -hidden_sign,
                  static_cast<std::int32_t>(desired.route));
              if (rejected.valid == 0u) continue;
              commit_supply_quantum(words, layout, desired, context,
                                    prediction.baseline_byte, &local);
              commit_supply_quantum(words, layout, rejected, context,
                                    prediction.baseline_byte, &local);
            }
          }
          if (local.route_count != 0u) {
            local.code = Code::kCommitted;
            local.committed = local.route_count;
            write_global(
                words, layout, kCommittedTransactions,
                read_global(words, layout, kCommittedTransactions) + 1u);
            write_global(words, layout, kStateEpoch,
                         read_global(words, layout, kStateEpoch) + 1u);
          } else {
            local.code = Code::kNoSupply;
          }
        }
      }
      prediction.valid = 0u;
      inputs->staged = 0u;
    }
  }
  local.matter_after = matter_bits(words, layout);
  local.resource_after = resource_bits(words, layout);
  local.state_hash_after = durable_state_hash(words, layout);
  if (receipt != nullptr) *receipt = local;
  return true;
}

__device__ inline void inverse_step_device(
    SiteWord* words, const DeviceLayout& layout) {
  const std::uint32_t event_count =
      read_global(words, layout, kEventCount);
  if (event_count == 0u || event_count > kEventDepth) return;
  const std::uint32_t event = event_count - 1u;
  const std::uint32_t start =
      read_event(words, layout, event, kEventJournalStart);
  const std::uint32_t count =
      read_event(words, layout, event, kEventJournalSize);
  if (start + count > kJournalCapacity ||
      start + count != read_global(words, layout, kJournalCount))
    return;
  for (std::uint32_t offset = count; offset > 0u; --offset) {
    const std::uint32_t entry = start + offset - 1u;
    const std::uint32_t route =
        read_journal(words, layout, entry, kJournalRoute);
    const std::uint32_t supply =
        read_journal(words, layout, entry, kJournalSupply);
    const std::int32_t direction =
        static_cast<std::int32_t>(
            read_journal(words, layout, entry, kJournalDirection));
    if (route >= kRouteCapacity || supply >= kSupplyWords) return;
    const GlobalField supply_field = static_cast<GlobalField>(
        (direction > 0 ? kPositiveSupplyBase : kNegativeSupplyBase) +
        supply);
    write_global(
        words, layout, supply_field,
        read_journal(words, layout, entry, kJournalSupplyBefore));
    for (std::uint32_t field = 0u; field < kRouteFields; ++field) {
      write_route(
          words, layout, route, static_cast<RouteField>(field),
          read_journal(words, layout, entry,
                       kJournalRouteBeforeBase + field));
    }
    for (std::uint32_t field = 0u; field < kJournalFields; ++field)
      write_journal(words, layout, entry, field, 0u);
  }
  write_global(
      words, layout, kStateEpoch,
      read_event(words, layout, event, kEventEpochBefore));
  write_global(
      words, layout, kCommittedTransactions,
      read_event(words, layout, event, kEventCommittedBefore));
  for (std::uint32_t field = 0u; field < kEventFieldCount; ++field)
    write_event(words, layout, event, static_cast<EventField>(field), 0u);
  write_global(words, layout, kJournalCount, start);
  write_global(words, layout, kEventCount, event);
}

__device__ inline bool step_device(
    SiteWord* words, const DeviceLayout& layout, DeviceInputs* inputs,
    CreditReceipt* receipt) {
  return apply_credit(words, layout, inputs, receipt);
}

static __global__ void step_kernel(
    SiteWord* words, const DeviceLayout* layout, DeviceInputs* inputs,
    CreditReceipt* receipt, std::uint32_t* advanced) {
  if (blockIdx.x != 0u || threadIdx.x != 0u) return;
  *advanced = valid_layout(words, layout) &&
                      step_device(words, *layout, inputs, receipt)
                  ? 1u
                  : 0u;
}

static __global__ void inverse_step_kernel(
    SiteWord* words, const DeviceLayout* layout) {
  if (blockIdx.x != 0u || threadIdx.x != 0u ||
      !valid_layout(words, layout))
    return;
  inverse_step_device(words, *layout);
}

static __global__ void lesion_all_routes_kernel(
    SiteWord* words, const DeviceLayout* layout, LesionReceipt* receipt) {
  if (blockIdx.x != 0u || threadIdx.x != 0u ||
      !valid_layout(words, layout))
    return;
  LesionReceipt local{};
  local.matter_before = matter_bits(words, *layout);
  local.resource_before = resource_bits(words, *layout);
  for (std::uint32_t route = 0u; route < kRouteCapacity; ++route) {
    SiteWord meta = read_route(words, *layout, route, kRouteMeta);
    if ((meta & kMetaOccupied) == 0u) continue;
    const SiteWord positive =
        read_route(words, *layout, route, kRoutePositive);
    const SiteWord negative =
        read_route(words, *layout, route, kRouteNegative);
    const SiteWord lesion_positive =
        read_route(words, *layout, route, kRouteLesionPositive);
    const SiteWord lesion_negative =
        read_route(words, *layout, route, kRouteLesionNegative);
    write_route(words, *layout, route, kRoutePositive, lesion_positive);
    write_route(words, *layout, route, kRouteNegative, lesion_negative);
    write_route(words, *layout, route, kRouteLesionPositive, positive);
    write_route(words, *layout, route, kRouteLesionNegative, negative);
    meta ^= kMetaEnabled | (1u << kMetaGenerationShift);
    write_route(words, *layout, route, kRouteMeta, meta);
    ++local.route_count;
  }
  local.applied = local.route_count != 0u ? 1u : 0u;
  local.matter_after = matter_bits(words, *layout);
  local.resource_after = resource_bits(words, *layout);
  if (receipt != nullptr) *receipt = local;
}

static __global__ void state_hash_kernel(
    const SiteWord* words, const DeviceLayout* layout,
    std::uint64_t* output) {
  if (blockIdx.x != 0u || threadIdx.x != 0u || output == nullptr) return;
  *output = valid_layout(words, layout)
                ? durable_state_hash(words, *layout)
                : 0u;
}

}  // namespace substrate::bcc32::resident_readout_f_route
