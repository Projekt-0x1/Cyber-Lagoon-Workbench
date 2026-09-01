#pragma once

#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>
#include <vector>

namespace substrate::bcc32::grown_cloud_factor {
struct PredictionWitness;
}

namespace bcc32::resident_readout_plastic_owner {

namespace grown_cloud_factor = substrate::bcc32::grown_cloud_factor;

constexpr std::uint32_t kOutputRows = 18u;
constexpr std::uint32_t kHiddenColumns = 384u;
constexpr std::uint32_t kRouteCount = kOutputRows * kHiddenColumns;
constexpr std::uint32_t kContextSlots = 8u;
constexpr std::uint32_t kContextRouteCount = kContextSlots * kRouteCount;
constexpr std::uint32_t kHorizon = 4u;
constexpr std::uint32_t kTopK = 8u;
constexpr std::uint32_t kMaxReceiptRoutes = 8u * 2u * kTopK;
constexpr std::uint32_t kByteMask = 0xffu;
constexpr std::int32_t kPlasticQuantumQ8 = 192;  // 0.75 signed route unit.

enum class Code : std::uint32_t {
  kEmpty = 0u,
  kCommitted = 1u,
  kMatchedResidual = 2u,
  kMalformedRails = 3u,
  kPrePrediction = 4u,
  kWrongTick = 5u,
  kExpired = 6u,
  kNoSupply = 7u,
  kInvalid = 8u,
  kBusy = 9u,
  kStale = 10u,
  kNoContext = 11u,
  kContextMismatch = 12u,
  kStateMismatch = 13u,
  kNoContextSlot = 14u,
};

template <typename Witness>
struct RawObservationT {
  // These are complementary rails for the eight output bits.  The producer
  // writes them in device memory; credit never receives a decoded byte.
  std::uint32_t positive = 0u;
  std::uint32_t negative = 0u;
  std::uint64_t tick = 0u;
  const Witness* m6_prediction = nullptr;
};

using RawObservation = RawObservationT<grown_cloud_factor::PredictionWitness>;

struct PredictionLane {
  std::uint64_t tick = 0u;
  std::uint32_t valid = 0u;
  std::uint32_t output_phase = 0u;
  std::uint64_t context_signature = 0u;
  std::uint8_t baseline_byte = 0u;
  std::uint8_t emitted_byte = 0u;
  std::uint8_t reserved[6]{};
  std::uint16_t selected_rows[kOutputRows]{};
  std::int8_t hidden_prototype[kHiddenColumns]{};
};

struct ContextSlot {
  std::uint32_t valid = 0u;
  std::uint32_t reserved = 0u;
  std::uint64_t context_signature = 0u;
  std::uint8_t baseline_byte = 0u;
  std::uint8_t reserved_byte[7]{};
  std::int8_t hidden_prototype[kHiddenColumns]{};
};

struct AttemptReceipt {
  Code code = Code::kEmpty;
  std::uint32_t lane = 0u;
  std::uint32_t requested = 0u;
  std::uint32_t committed = 0u;
  std::uint64_t observation_tick = 0u;
  std::uint64_t prediction_tick = 0u;
  std::uint32_t predicted_positive = 0u;
  std::uint32_t predicted_negative = 0u;
  std::uint32_t observed_positive = 0u;
  std::uint32_t observed_negative = 0u;
  std::uint32_t positive_supply_before = 0u;
  std::uint32_t negative_supply_before = 0u;
  std::uint32_t positive_supply_after = 0u;
  std::uint32_t negative_supply_after = 0u;
};

struct CommittedReceipt : AttemptReceipt {
  std::uint32_t context_slot = 0u;
  std::uint32_t slot_created = 0u;
  ContextSlot slot_before{};
  std::uint32_t route_count = 0u;
  std::uint32_t route_ids[kMaxReceiptRoutes]{};
  std::uint32_t route_positive_before[kMaxReceiptRoutes]{};
  std::uint32_t route_negative_before[kMaxReceiptRoutes]{};
  std::int8_t route_direction[kMaxReceiptRoutes]{};
  std::uint64_t attempted_before = 0u;
  std::uint64_t committed_before = 0u;
  std::uint64_t state_hash_before = 0u;
  std::uint64_t state_hash_after = 0u;
};

struct LesionReceipt {
  Code code = Code::kEmpty;
  std::uint32_t route_count = 0u;
  std::uint32_t route_ids[kMaxReceiptRoutes]{};
  std::uint8_t enabled_before[kMaxReceiptRoutes]{};
  std::uint64_t generation_before[kMaxReceiptRoutes]{};
};

struct DeviceView {
  ContextSlot* context_slots = nullptr;
  std::uint32_t* route_positive = nullptr;
  std::uint32_t* route_negative = nullptr;
  std::uint8_t* route_enabled = nullptr;
  std::uint64_t* route_generations = nullptr;
  PredictionLane* lanes = nullptr;
  std::uint64_t* next_tick = nullptr;
  std::uint32_t* positive_supply = nullptr;
  std::uint32_t* negative_supply = nullptr;
  std::uint64_t* attempted = nullptr;
  std::uint64_t* committed = nullptr;
  std::uint32_t* transaction_lock = nullptr;
  AttemptReceipt* attempt_receipt = nullptr;
  CommittedReceipt* committed_receipt = nullptr;
  LesionReceipt* lesion_receipt = nullptr;
};

__device__ __forceinline__ bool valid_view(const DeviceView& view) {
  return view.context_slots != nullptr && view.route_positive != nullptr &&
         view.route_negative != nullptr &&
         view.route_enabled != nullptr &&
         view.route_generations != nullptr && view.lanes != nullptr &&
         view.next_tick != nullptr && view.positive_supply != nullptr &&
         view.negative_supply != nullptr && view.attempted != nullptr &&
         view.committed != nullptr && view.transaction_lock != nullptr &&
         view.attempt_receipt != nullptr && view.committed_receipt != nullptr &&
         view.lesion_receipt != nullptr;
}

__device__ __forceinline__ std::uint32_t route_id(
    std::uint32_t context_slot, std::uint32_t output_row,
    std::uint32_t hidden_col) {
  return context_slot * kRouteCount + output_row * kHiddenColumns + hidden_col;
}

__device__ __forceinline__ std::uint64_t mix_hash(std::uint64_t hash,
                                                   std::uint64_t value) {
  constexpr std::uint64_t kPrime = UINT64_C(1099511628211);
  for (std::uint32_t byte = 0u; byte < 8u; ++byte) {
    hash ^= static_cast<std::uint8_t>(value >> (byte * 8u));
    hash *= kPrime;
  }
  return hash;
}

__device__ inline std::uint64_t durable_state_hash(const DeviceView& view) {
  if (!valid_view(view)) return 0u;
  std::uint64_t hash = UINT64_C(1469598103934665603);
  hash = mix_hash(hash, *view.positive_supply);
  hash = mix_hash(hash, *view.negative_supply);
  hash = mix_hash(hash, *view.committed);
  for (std::uint32_t slot = 0u; slot < kContextSlots; ++slot) {
    const ContextSlot& context = view.context_slots[slot];
    hash = mix_hash(hash, context.valid);
    hash = mix_hash(hash, context.context_signature);
    hash = mix_hash(hash, context.baseline_byte);
    for (std::uint32_t column = 0u; column < kHiddenColumns; ++column)
      hash = mix_hash(hash, static_cast<std::uint8_t>(context.hidden_prototype[column]));
  }
  for (std::uint32_t route = 0u; route < kContextRouteCount; ++route) {
    hash = mix_hash(hash, view.route_positive[route]);
    hash = mix_hash(hash, view.route_negative[route]);
    hash = mix_hash(hash, view.route_enabled[route]);
    hash = mix_hash(hash, view.route_generations[route]);
  }
  return hash;
}

__device__ inline std::uint64_t state_hash(const DeviceView& view) {
  if (!valid_view(view)) return 0u;
  std::uint64_t hash = durable_state_hash(view);
  hash = mix_hash(hash, *view.next_tick);
  hash = mix_hash(hash, *view.attempted);
  for (std::uint32_t lane = 0u; lane < kHorizon; ++lane) {
    const PredictionLane& prediction = view.lanes[lane];
    hash = mix_hash(hash, prediction.tick);
    hash = mix_hash(hash, prediction.valid);
    hash = mix_hash(hash, prediction.context_signature);
    hash = mix_hash(hash, prediction.baseline_byte);
    hash = mix_hash(hash, prediction.emitted_byte);
    for (std::uint32_t column = 0u; column < kHiddenColumns; ++column)
      hash = mix_hash(hash, static_cast<std::uint8_t>(prediction.hidden_prototype[column]));
    for (std::uint32_t row = 0u; row < kOutputRows; ++row) {
      hash = mix_hash(hash, prediction.selected_rows[row]);
    }
  }
  return hash;
}

__device__ __forceinline__ void clear_attempt(AttemptReceipt* receipt) {
  *receipt = AttemptReceipt{};
}

__device__ __forceinline__ void clear_committed(CommittedReceipt* receipt) {
  *receipt = CommittedReceipt{};
}

template <typename Witness>
__device__ __forceinline__ bool valid_rails(
    const RawObservationT<Witness>& observation) {
  if (((observation.positive | observation.negative) & ~kByteMask) != 0u)
    return false;
  const std::uint32_t positive = observation.positive & kByteMask;
  const std::uint32_t negative = observation.negative & kByteMask;
  return positive != 0u || negative != 0u
             ? ((positive | negative) == kByteMask &&
                (positive & negative) == 0u)
             : false;
}

template <typename Witness>
__device__ __forceinline__ bool valid_m6_witness(const Witness* witness) {
  return witness != nullptr && witness->valid != 0u &&
         witness->predicted_signature != 0u;
}

template <typename Witness>
__device__ __forceinline__ std::uint64_t m6_signature(const Witness* witness) {
  return witness == nullptr ? 0u
                            : static_cast<std::uint64_t>(witness->predicted_signature);
}

__device__ __forceinline__ std::int8_t quantize_hidden(float value) {
  if (!isfinite(value) || value == 0.0f) return 0;
  const float scaled = value * 127.0f;
  if (scaled >= 127.0f) return 127;
  if (scaled <= -127.0f) return -127;
  const int rounded =
      scaled > 0.0f ? static_cast<int>(scaled + 0.5f)
                    : static_cast<int>(scaled - 0.5f);
  if (rounded == 0) return value > 0.0f ? 1 : -1;
  return static_cast<std::int8_t>(rounded);
}

// Select the ordinal-th strongest eligible hidden column. Ties prefer the
// lower column, making the factorized H4 witness deterministic without storing
// a 6912-route tensor.
__device__ inline std::uint32_t eligible_column(const PredictionLane& prediction,
                                         std::uint32_t ordinal) {
  if (ordinal >= kTopK) return kHiddenColumns;
  std::uint32_t selected[kTopK];
  for (std::uint32_t i = 0u; i < kTopK; ++i) selected[i] = kHiddenColumns;
  std::uint32_t selected_count = 0u;
  for (std::uint32_t column = 0u; column < kHiddenColumns; ++column) {
    const std::int32_t value = prediction.hidden_prototype[column];
    const std::uint32_t magnitude =
        static_cast<std::uint32_t>(value < 0 ? -value : value);
    if (magnitude == 0u) continue;
    std::uint32_t insert = selected_count < kTopK ? selected_count : kTopK;
    for (std::uint32_t i = 0u; i < selected_count && i < kTopK; ++i) {
      const std::int32_t prior_value =
          prediction.hidden_prototype[selected[i]];
      const std::uint32_t prior_magnitude = static_cast<std::uint32_t>(
          prior_value < 0 ? -prior_value : prior_value);
      if (magnitude > prior_magnitude) {
        insert = i;
        break;
      }
    }
    if (insert >= kTopK) continue;
    const std::uint32_t limit =
        selected_count < kTopK ? selected_count : kTopK - 1u;
    for (std::uint32_t i = limit; i > insert; --i)
      selected[i] = selected[i - 1u];
    selected[insert] = column;
    if (selected_count < kTopK) ++selected_count;
  }
  return ordinal < selected_count ? selected[ordinal] : kHiddenColumns;
}

// Called by the recurrent executor after it has produced the live scores.
// pair_scores contains the executor's 18 scores: rows 2/3, 4/5, ... 16/17
// are the eight byte pairs; rows 0/1 remain available to the executor.
template <typename Witness>
__device__ inline bool capture_prediction_device(DeviceView view, const float* hidden,
                                           const float* pair_scores,
                                           const std::uint16_t* selected_rows,
                                           const Witness* m6_prediction,
                                           std::uint8_t emitted_byte) {
  if (!valid_view(view) || hidden == nullptr || pair_scores == nullptr ||
      selected_rows == nullptr || !valid_m6_witness(m6_prediction))
    return false;
  for (std::uint32_t row = 0u; row < kOutputRows; ++row) {
    if (selected_rows[row] >= kOutputRows || !isfinite(pair_scores[row])) {
      return false;
    }
  }
  const std::uint64_t tick =
      atomicAdd(reinterpret_cast<unsigned long long*>(view.next_tick), 1ULL);
  PredictionLane& prediction = view.lanes[tick % kHorizon];
  prediction = PredictionLane{};
  prediction.tick = tick;
  prediction.valid = 1u;
  prediction.context_signature = m6_signature(m6_prediction);
  for (std::uint32_t row = 0u; row < kOutputRows; ++row)
    prediction.selected_rows[row] = selected_rows[row];
  std::uint32_t positive = 0u;
  for (std::uint32_t bit = 0u; bit < 8u; ++bit) {
    const std::uint32_t low = 2u + 2u * bit;
    if (pair_scores[low + 1u] > pair_scores[low])
      positive |= 1u << bit;
  }
  prediction.baseline_byte = static_cast<std::uint8_t>(positive);
  prediction.emitted_byte = emitted_byte;
  for (std::uint32_t column = 0u; column < kHiddenColumns; ++column)
    prediction.hidden_prototype[column] = quantize_hidden(hidden[column]);
  return true;
}

template <typename Witness>
static __global__ void capture_prediction_kernel(DeviceView view, const float* hidden,
                                          const float* pair_scores,
                                          const std::uint16_t* selected_rows,
                                          const Witness* m6_prediction,
                                          std::uint8_t emitted_byte) {
  if (blockIdx.x == 0u && threadIdx.x == 0u)
    (void)capture_prediction_device(view, hidden, pair_scores, selected_rows,
                                    m6_prediction, emitted_byte);
}

__device__ __forceinline__ std::int8_t residual_sign(
    std::uint32_t predicted_positive, std::uint32_t observed_positive,
    std::uint32_t bit) {
  const bool predicted = (predicted_positive & (1u << bit)) != 0u;
  const bool observed = (observed_positive & (1u << bit)) != 0u;
  if (predicted == observed) return 0;
  return observed ? 1 : -1;
}

// Returns the exact context-and-baseline slot.  The prototype is retained as
// H4 eligibility/receipt state, but is deliberately not an activation key.
__device__ inline std::int32_t find_context_slot(DeviceView view,
                                          std::uint64_t signature,
                                          std::uint8_t baseline_byte,
                                          const std::int8_t* prototype,
                                          ContextSlot* before,
                                          bool* created) {
  std::int32_t empty = -1;
  for (std::uint32_t slot = 0u; slot < kContextSlots; ++slot) {
    ContextSlot& context = view.context_slots[slot];
    if (context.valid == 0u) {
      if (empty < 0) empty = static_cast<std::int32_t>(slot);
      continue;
    }
    if (context.context_signature != signature ||
        context.baseline_byte != baseline_byte)
      continue;
    *created = false;
    return static_cast<std::int32_t>(slot);
  }
  if (empty < 0) return -1;
  *before = view.context_slots[empty];
  ContextSlot& context = view.context_slots[empty];
  context.valid = 1u;
  context.context_signature = signature;
  context.baseline_byte = baseline_byte;
  for (std::uint32_t column = 0u; column < kHiddenColumns; ++column)
    context.hidden_prototype[column] = prototype[column];
  *created = true;
  return empty;
}

__device__ inline std::int32_t find_matching_slot(const DeviceView& view,
                                           std::uint64_t signature,
                                           std::uint8_t baseline_byte) {
  if (signature == 0u) return -1;
  for (std::uint32_t slot = 0u; slot < kContextSlots; ++slot) {
    const ContextSlot& context = view.context_slots[slot];
    if (context.valid != 0u && context.context_signature == signature &&
        context.baseline_byte == baseline_byte)
      return static_cast<std::int32_t>(slot);
  }
  return -1;
}

// Fused executors call this while scoring each output row.  A context delta
// is invisible unless the live M6 basin and current baseline byte match the
// credited slot.  The stored hidden prototype is eligibility evidence only;
// current live hidden signs drive the additive correction.
__device__ __forceinline__ std::uint8_t decode_baseline_byte(
    const float* pair_scores) {
  if (pair_scores == nullptr) return 0xffu;
  std::uint32_t byte = 0u;
  for (std::uint32_t bit = 0u; bit < 8u; ++bit) {
    const std::uint32_t low = 2u + 2u * bit;
    if (!isfinite(pair_scores[low]) || !isfinite(pair_scores[low + 1u]))
      return 0xffu;
    if (pair_scores[low + 1u] > pair_scores[low]) byte |= 1u << bit;
  }
  return static_cast<std::uint8_t>(byte);
}

template <typename Witness>
__device__ inline float gated_readout_correction(
    const DeviceView& view, const Witness* m6, const float* hidden,
    const float* pair_scores, std::uint32_t output_row) {
  if (!valid_view(view) || !valid_m6_witness(m6) || hidden == nullptr ||
      output_row >= kOutputRows)
    return 0.0f;
  const std::uint8_t baseline_byte = decode_baseline_byte(pair_scores);
  if (baseline_byte == 0xffu) return 0.0f;
  const std::int32_t slot =
      find_matching_slot(view, m6_signature(m6), baseline_byte);
  if (slot < 0) return 0.0f;
  float sum = 0.0f;
  for (std::uint32_t column = 0u; column < kHiddenColumns; ++column) {
    const std::uint32_t id =
        route_id(static_cast<std::uint32_t>(slot), output_row, column);
    if (view.route_enabled[id] == 0u) continue;
    const std::int64_t signed_quanta =
        static_cast<std::int64_t>(view.route_positive[id]) -
        static_cast<std::int64_t>(view.route_negative[id]);
    sum += static_cast<float>(signed_quanta * kPlasticQuantumQ8) *
           hidden[column];
  }
  return sum / 256.0f;
}

template <typename Witness>
__device__ __forceinline__ float score_correction(
    const DeviceView& view, const Witness* m6, const float* hidden,
    const float* pair_scores, std::uint32_t output_row) {
  return gated_readout_correction(view, m6, hidden, pair_scores, output_row);
}

template <typename Witness>
__device__ inline bool credit_device(DeviceView view,
                              const RawObservationT<Witness>* observation) {
  if (!valid_view(view) || observation == nullptr) return false;
  clear_attempt(view.attempt_receipt);
  AttemptReceipt& attempt = *view.attempt_receipt;
  const RawObservationT<Witness> observed = *observation;
  attempt.observation_tick = observed.tick;
  attempt.observed_positive = observed.positive & kByteMask;
  attempt.observed_negative = observed.negative & kByteMask;
  const std::uint64_t state_hash_before = durable_state_hash(view);
  const std::uint64_t attempted_before =
      atomicAdd(reinterpret_cast<unsigned long long*>(view.attempted), 1ULL);
  if (!valid_rails(observed)) {
    attempt.code = Code::kMalformedRails;
    return false;
  }
  if (!valid_m6_witness(observed.m6_prediction)) {
    attempt.code = Code::kNoContext;
    return false;
  }
  if (atomicCAS(view.transaction_lock, 0u, 1u) != 0u) {
    attempt.code = Code::kBusy;
    return false;
  }
  const std::uint64_t now = *view.next_tick;
  std::int32_t lane_index = -1;
  for (std::uint32_t lane = 0u; lane < kHorizon; ++lane) {
    const PredictionLane& prediction = view.lanes[lane];
    if (prediction.valid == 0u || prediction.tick >= observed.tick) continue;
    const std::uint64_t age = observed.tick - prediction.tick;
    if (age <= kHorizon && observed.tick == now) {
      // Credit always consumes the newest live lane.  The host cannot choose
      // an age lane or route; older lanes remain bounded H4 history.
      if (lane_index < 0 || prediction.tick > view.lanes[lane_index].tick)
        lane_index = static_cast<std::int32_t>(lane);
    }
  }
  if (lane_index < 0) {
    attempt.code = observed.tick >= now
                       ? Code::kPrePrediction
                       : ((now - observed.tick) > kHorizon ? Code::kExpired
                                                           : Code::kWrongTick);
    view.transaction_lock[0] = 0u;
    return false;
  }
  PredictionLane& prediction = view.lanes[lane_index];
  attempt.lane = static_cast<std::uint32_t>(lane_index);
  attempt.prediction_tick = prediction.tick;
  attempt.predicted_positive = prediction.emitted_byte;
  attempt.predicted_negative = (~prediction.emitted_byte) & kByteMask;
  if (prediction.context_signature != m6_signature(observed.m6_prediction)) {
    prediction.valid = 0u;
    attempt.code = Code::kContextMismatch;
    view.transaction_lock[0] = 0u;
    return false;
  }
  attempt.positive_supply_before = *view.positive_supply;
  attempt.negative_supply_before = *view.negative_supply;
  const std::uint32_t residual =
      (static_cast<std::uint32_t>(prediction.emitted_byte) ^
       attempt.observed_positive) & kByteMask;
  if (residual == 0u) {
    prediction.valid = 0u;
    attempt.code = Code::kMatchedResidual;
    attempt.positive_supply_after = *view.positive_supply;
    attempt.negative_supply_after = *view.negative_supply;
    view.transaction_lock[0] = 0u;
    return false;
  }

  ContextSlot slot_before{};
  bool created = false;
  const std::int32_t context_slot = find_context_slot(
      view, prediction.context_signature, prediction.baseline_byte,
      prediction.hidden_prototype, &slot_before, &created);
  if (context_slot < 0) {
    prediction.valid = 0u;
    attempt.code = Code::kNoContextSlot;
    view.transaction_lock[0] = 0u;
    return false;
  }

  const CommittedReceipt previous_receipt = *view.committed_receipt;
  CommittedReceipt& receipt = *view.committed_receipt;
  receipt = CommittedReceipt{};
  receipt.code = Code::kCommitted;
  receipt.lane = static_cast<std::uint32_t>(lane_index);
  receipt.observation_tick = observed.tick;
  receipt.prediction_tick = prediction.tick;
  receipt.predicted_positive = prediction.emitted_byte;
  receipt.predicted_negative = (~prediction.emitted_byte) & kByteMask;
  receipt.observed_positive = attempt.observed_positive;
  receipt.observed_negative = attempt.observed_negative;
  receipt.positive_supply_before = *view.positive_supply;
  receipt.negative_supply_before = *view.negative_supply;
  receipt.attempted_before = attempted_before;
  receipt.committed_before = *view.committed;
  receipt.state_hash_before = state_hash_before;
  receipt.context_slot = static_cast<std::uint32_t>(context_slot);
  receipt.slot_created = created ? 1u : 0u;
  receipt.slot_before = slot_before;
  for (std::uint32_t bit = 0u; bit < 8u; ++bit) {
    const std::int8_t residual =
        residual_sign(prediction.emitted_byte, attempt.observed_positive, bit);
    if (residual == 0) continue;
    const std::uint32_t low = 2u + 2u * bit;
    const bool observed_one =
        (attempt.observed_positive & (1u << bit)) != 0u;
    const std::uint32_t desired_row =
        prediction.selected_rows[low + (observed_one ? 1u : 0u)];
    const std::uint32_t rejected_row =
        prediction.selected_rows[low + (observed_one ? 0u : 1u)];
    for (std::uint32_t k = 0u; k < kTopK; ++k) {
      const std::uint32_t column = eligible_column(prediction, k);
      if (column >= kHiddenColumns) continue;
      const std::int8_t hidden_sign =
          prediction.hidden_prototype[column] > 0 ? 1 : -1;
      const std::uint32_t rows[2] = {desired_row, rejected_row};
      const std::int8_t directions[2] = {hidden_sign,
                                         static_cast<std::int8_t>(-hidden_sign)};
      if (receipt.route_count + 2u > kMaxReceiptRoutes) break;
      const std::uint32_t desired_id =
          route_id(receipt.context_slot, rows[0], column);
      const std::uint32_t rejected_id =
          route_id(receipt.context_slot, rows[1], column);
      if (view.route_enabled[desired_id] == 0u ||
          view.route_enabled[rejected_id] == 0u ||
          *view.positive_supply == 0u || *view.negative_supply == 0u)
        continue;
      const std::uint32_t ids[2] = {desired_id, rejected_id};
      const std::uint32_t desired_population =
          directions[0] > 0 ? view.route_positive[ids[0]]
                            : view.route_negative[ids[0]];
      const std::uint32_t rejected_population =
          directions[1] > 0 ? view.route_positive[ids[1]]
                            : view.route_negative[ids[1]];
      if (desired_population == 0xffffffffu ||
          rejected_population == 0xffffffffu)
        continue;
      for (std::uint32_t side = 0u; side < 2u; ++side) {
        const std::int8_t direction = directions[side];
        const std::uint32_t id = ids[side];
        std::uint32_t& population =
            direction > 0 ? view.route_positive[id] : view.route_negative[id];
        const std::uint32_t slot = receipt.route_count++;
        receipt.route_ids[slot] = id;
        receipt.route_positive_before[slot] = view.route_positive[id];
        receipt.route_negative_before[slot] = view.route_negative[id];
        receipt.route_direction[slot] = direction;
        ++population;
        if (direction > 0)
          --*view.positive_supply;
        else
          --*view.negative_supply;
      }
    }
  }
  const std::uint32_t committed_routes = receipt.route_count;
  if (committed_routes == 0u) {
    if (receipt.slot_created != 0u)
      view.context_slots[receipt.context_slot] = receipt.slot_before;
    receipt = previous_receipt;
  }
  prediction.valid = 0u;
  if (committed_routes != 0u) {
    receipt.committed = committed_routes;
    receipt.positive_supply_after = *view.positive_supply;
    receipt.negative_supply_after = *view.negative_supply;
    atomicAdd(reinterpret_cast<unsigned long long*>(view.committed), 1ULL);
    receipt.state_hash_after = durable_state_hash(view);
  }
  attempt = committed_routes != 0u
                ? static_cast<const AttemptReceipt&>(receipt)
                : AttemptReceipt{};
  attempt.observation_tick = observed.tick;
  attempt.prediction_tick = prediction.tick;
  attempt.predicted_positive = prediction.emitted_byte;
  attempt.predicted_negative = (~prediction.emitted_byte) & kByteMask;
  attempt.observed_positive = observed.positive & kByteMask;
  attempt.observed_negative = observed.negative & kByteMask;
  attempt.requested = __popc(residual);
  attempt.committed = committed_routes;
  attempt.positive_supply_after = *view.positive_supply;
  attempt.negative_supply_after = *view.negative_supply;
  attempt.code = committed_routes == 0u ? Code::kNoSupply : Code::kCommitted;
  view.transaction_lock[0] = 0u;
  return committed_routes != 0u;
}

template <typename Witness>
static __global__ void credit_kernel(DeviceView view,
                              const RawObservationT<Witness>* observation) {
  if (blockIdx.x == 0u && threadIdx.x == 0u) (void)credit_device(view, observation);
}

__device__ inline bool inverse_last_credit(DeviceView view) {
  if (!valid_view(view)) return false;
  if (atomicCAS(view.transaction_lock, 0u, 1u) != 0u) return false;
  CommittedReceipt& receipt = *view.committed_receipt;
  if (receipt.code != Code::kCommitted ||
      receipt.route_count > kMaxReceiptRoutes ||
      receipt.context_slot >= kContextSlots ||
      durable_state_hash(view) != receipt.state_hash_after) {
    view.transaction_lock[0] = 0u;
    return false;
  }
  for (std::uint32_t i = 0u; i < receipt.route_count; ++i) {
    const std::uint32_t id = receipt.route_ids[i];
    const std::uint32_t expected_positive =
        receipt.route_positive_before[i] +
        (receipt.route_direction[i] > 0 ? 1u : 0u);
    const std::uint32_t expected_negative =
        receipt.route_negative_before[i] +
        (receipt.route_direction[i] < 0 ? 1u : 0u);
    if (id >= kContextRouteCount ||
        view.route_positive[id] != expected_positive ||
        view.route_negative[id] != expected_negative) {
      view.transaction_lock[0] = 0u;
      return false;
    }
    view.route_positive[id] = receipt.route_positive_before[i];
    view.route_negative[id] = receipt.route_negative_before[i];
  }
  *view.positive_supply = receipt.positive_supply_before;
  *view.negative_supply = receipt.negative_supply_before;
  *view.committed = receipt.committed_before;
  if (receipt.slot_created != 0u)
    view.context_slots[receipt.context_slot] = receipt.slot_before;
  if (durable_state_hash(view) != receipt.state_hash_before) {
    view.transaction_lock[0] = 0u;
    return false;
  }
  receipt.code = Code::kStale;
  view.transaction_lock[0] = 0u;
  return true;
}

static __global__ void inverse_last_credit_kernel(DeviceView view) {
  if (blockIdx.x == 0u && threadIdx.x == 0u) (void)inverse_last_credit(view);
}

static __global__ void initialize_kernel(DeviceView view, std::uint32_t positive_supply,
                                  std::uint32_t negative_supply) {
  if (blockIdx.x != 0u || threadIdx.x != 0u || !valid_view(view)) return;
  *view.next_tick = 0u;
  *view.positive_supply = positive_supply;
  *view.negative_supply = negative_supply;
  *view.attempted = 0u;
  *view.committed = 0u;
  *view.transaction_lock = 0u;
  *view.attempt_receipt = AttemptReceipt{};
  *view.committed_receipt = CommittedReceipt{};
  *view.lesion_receipt = LesionReceipt{};
  for (std::uint32_t slot = 0u; slot < kContextSlots; ++slot)
    view.context_slots[slot] = ContextSlot{};
  for (std::uint32_t route = 0u; route < kContextRouteCount; ++route) {
    view.route_positive[route] = 0u;
    view.route_negative[route] = 0u;
    view.route_enabled[route] = 1u;
    view.route_generations[route] = 0u;
  }
  for (std::uint32_t lane = 0u; lane < kHorizon; ++lane)
    view.lanes[lane] = PredictionLane{};
}

static __global__ void lesion_committed_routes_kernel(DeviceView view) {
  if (blockIdx.x != 0u || threadIdx.x != 0u || !valid_view(view)) return;
  LesionReceipt& receipt = *view.lesion_receipt;
  receipt = LesionReceipt{};
  const CommittedReceipt& committed = *view.committed_receipt;
  if (committed.code != Code::kCommitted ||
      committed.route_count == 0u ||
      committed.route_count > kMaxReceiptRoutes) {
    receipt.code = Code::kInvalid;
    return;
  }
  receipt.code = Code::kCommitted;
  receipt.route_count = committed.route_count;
  for (std::uint32_t i = 0u; i < committed.route_count; ++i) {
    const std::uint32_t id = committed.route_ids[i];
    if (id >= kContextRouteCount) {
      receipt = LesionReceipt{};
      receipt.code = Code::kInvalid;
      return;
    }
    receipt.route_ids[i] = id;
    receipt.enabled_before[i] = view.route_enabled[id];
    receipt.generation_before[i] = view.route_generations[id];
  }
  for (std::uint32_t i = 0u; i < receipt.route_count; ++i) {
    const std::uint32_t id = receipt.route_ids[i];
    view.route_enabled[id] = 0u;
    ++view.route_generations[id];
  }
}

static __global__ void restore_lesion_kernel(DeviceView view) {
  if (blockIdx.x != 0u || threadIdx.x != 0u || !valid_view(view)) return;
  LesionReceipt& receipt = *view.lesion_receipt;
  if (receipt.code != Code::kCommitted ||
      receipt.route_count == 0u ||
      receipt.route_count > kMaxReceiptRoutes)
    return;
  for (std::uint32_t i = 0u; i < receipt.route_count; ++i) {
    const std::uint32_t id = receipt.route_ids[i];
    if (id >= kContextRouteCount ||
        view.route_generations[id] != receipt.generation_before[i] + 1u ||
        view.route_enabled[id] != 0u)
      return;
  }
  for (std::uint32_t i = 0u; i < receipt.route_count; ++i) {
    const std::uint32_t id = receipt.route_ids[i];
    view.route_enabled[id] = receipt.enabled_before[i];
    view.route_generations[id] = receipt.generation_before[i];
  }
  receipt.code = Code::kStale;
}

inline void require_cuda(cudaError_t error, const char* message) {
  if (error != cudaSuccess) throw std::runtime_error(message);
}

class DeviceOwner {
 private:
  struct KernelGraph {
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t executable = nullptr;
    cudaGraphNode_t node = nullptr;

    void destroy() noexcept {
      if (executable != nullptr) {
        (void)cudaGraphExecDestroy(executable);
        executable = nullptr;
      }
      if (graph != nullptr) {
        (void)cudaGraphDestroy(graph);
        graph = nullptr;
      }
      node = nullptr;
    }

    void take_from(KernelGraph& other) noexcept {
      destroy();
      graph = other.graph;
      executable = other.executable;
      node = other.node;
      other.graph = nullptr;
      other.executable = nullptr;
      other.node = nullptr;
    }

    ~KernelGraph() { destroy(); }
  };

  template <typename T>
  static void* kernel_argument(T& value) {
    return const_cast<void*>(static_cast<const void*>(&value));
  }

  static void launch_graph(KernelGraph& graph, void* function,
                           void** arguments, cudaStream_t stream,
                           const char* operation) {
    cudaKernelNodeParams params{};
    params.func = function;
    params.gridDim = dim3{1u, 1u, 1u};
    params.blockDim = dim3{1u, 1u, 1u};
    params.sharedMemBytes = 0u;
    params.kernelParams = arguments;
    params.extra = nullptr;
    if (graph.executable == nullptr) {
      require_cuda(cudaGraphCreate(&graph.graph, 0u), operation);
      require_cuda(cudaGraphAddKernelNode(
                       &graph.node, graph.graph, nullptr, 0u, &params),
                   operation);
      require_cuda(cudaGraphInstantiate(&graph.executable, graph.graph,
                                        nullptr, nullptr, 0u),
                   operation);
    } else {
      require_cuda(cudaGraphExecKernelNodeSetParams(
                       graph.executable, graph.node, &params),
                   operation);
    }
    require_cuda(cudaGraphLaunch(graph.executable, stream), operation);
  }

 public:
  DeviceOwner() { allocate(); }
  DeviceOwner(const DeviceOwner&) = delete;
  DeviceOwner& operator=(const DeviceOwner&) = delete;
  DeviceOwner(DeviceOwner&& other) noexcept { move_from(other); }
  DeviceOwner& operator=(DeviceOwner&& other) noexcept {
    if (this != &other) {
      release();
      move_from(other);
    }
    return *this;
  }
  ~DeviceOwner() { release(); }

  DeviceView view() const { return view_; }

  void initialize(std::uint32_t positive_supply = 256u,
                  std::uint32_t negative_supply = 256u,
                  cudaStream_t stream = nullptr) {
    void* arguments[] = {kernel_argument(view_),
                         kernel_argument(positive_supply),
                         kernel_argument(negative_supply)};
    launch_graph(initialize_graph_, reinterpret_cast<void*>(initialize_kernel),
                 arguments, stream, "resident readout owner initialize");
    require_cuda(cudaGetLastError(), "resident readout owner initialize");
  }

  template <typename Witness>
  void capture(const float* hidden, const float* pair_scores,
               const std::uint16_t* selected_rows, const Witness* m6_prediction,
               std::uint8_t emitted_byte,
               cudaStream_t stream = nullptr) {
    void* arguments[] = {kernel_argument(view_),
                         kernel_argument(hidden),
                         kernel_argument(pair_scores),
                         kernel_argument(selected_rows),
                         kernel_argument(m6_prediction),
                         kernel_argument(emitted_byte)};
    launch_graph(
        capture_graph_,
        reinterpret_cast<void*>(capture_prediction_kernel<Witness>), arguments,
        stream, "resident readout owner capture");
    require_cuda(cudaGetLastError(), "resident readout owner capture");
  }

  template <typename Witness>
  void credit(const RawObservationT<Witness>* device_observation,
              cudaStream_t stream = nullptr) {
    void* arguments[] = {kernel_argument(view_),
                         kernel_argument(device_observation)};
    launch_graph(credit_graph_, reinterpret_cast<void*>(credit_kernel<Witness>),
                 arguments, stream, "resident readout owner credit");
    require_cuda(cudaGetLastError(), "resident readout owner credit");
  }

  void inverse(cudaStream_t stream = nullptr) {
    void* arguments[] = {kernel_argument(view_)};
    launch_graph(inverse_graph_,
                 reinterpret_cast<void*>(inverse_last_credit_kernel), arguments,
                 stream, "resident readout owner inverse");
    require_cuda(cudaGetLastError(), "resident readout owner inverse");
  }

  void lesion_committed_routes(cudaStream_t stream = nullptr) {
    void* arguments[] = {kernel_argument(view_)};
    launch_graph(
        lesion_graph_, reinterpret_cast<void*>(lesion_committed_routes_kernel),
        arguments, stream, "resident readout owner route-set lesion");
    require_cuda(cudaGetLastError(), "resident readout owner route-set lesion");
  }

  void restore_lesion(cudaStream_t stream = nullptr) {
    void* arguments[] = {kernel_argument(view_)};
    launch_graph(restore_graph_, reinterpret_cast<void*>(restore_lesion_kernel),
                 arguments, stream, "resident readout owner restore lesion");
    require_cuda(cudaGetLastError(), "resident readout owner restore lesion");
  }

  AttemptReceipt copy_attempt() const { return copy_one(view_.attempt_receipt); }
  CommittedReceipt copy_committed() const {
    return copy_one(view_.committed_receipt);
  }
  LesionReceipt copy_lesion() const { return copy_one(view_.lesion_receipt); }

  std::uint64_t state_hash() const {
    return hash_state(true);
  }

  std::uint64_t durable_state_hash() const {
    return hash_state(false);
  }

  void set_supply(std::uint32_t positive_supply, std::uint32_t negative_supply) {
    require_cuda(cudaMemcpy(view_.positive_supply, &positive_supply,
                            sizeof(positive_supply), cudaMemcpyHostToDevice),
                 "resident readout owner positive supply");
    require_cuda(cudaMemcpy(view_.negative_supply, &negative_supply,
                            sizeof(negative_supply), cudaMemcpyHostToDevice),
                 "resident readout owner negative supply");
  }

 private:
  DeviceView view_{};
  KernelGraph initialize_graph_;
  KernelGraph capture_graph_;
  KernelGraph credit_graph_;
  KernelGraph inverse_graph_;
  KernelGraph lesion_graph_;
  KernelGraph restore_graph_;

  std::uint64_t hash_state(bool include_ephemeral) const {
    require_cuda(cudaDeviceSynchronize(), "resident readout owner hash sync");
    std::vector<ContextSlot> contexts(kContextSlots);
    std::vector<std::uint32_t> route_positive(kContextRouteCount);
    std::vector<std::uint32_t> route_negative(kContextRouteCount);
    std::vector<std::uint8_t> enabled(kContextRouteCount);
    std::vector<std::uint64_t> generations(kContextRouteCount);
    std::vector<PredictionLane> lanes(kHorizon);
    std::uint64_t tick = 0u;
    std::uint32_t positive = 0u;
    std::uint32_t negative = 0u;
    std::uint64_t attempted = 0u;
    std::uint64_t committed = 0u;
    require_cuda(cudaMemcpy(contexts.data(), view_.context_slots,
                            contexts.size() * sizeof(contexts[0]), cudaMemcpyDeviceToHost),
                 "resident readout owner hash contexts");
    require_cuda(cudaMemcpy(route_positive.data(), view_.route_positive,
                            route_positive.size() * sizeof(route_positive[0]),
                            cudaMemcpyDeviceToHost),
                 "resident readout owner hash positive routes");
    require_cuda(cudaMemcpy(route_negative.data(), view_.route_negative,
                            route_negative.size() * sizeof(route_negative[0]),
                            cudaMemcpyDeviceToHost),
                 "resident readout owner hash negative routes");
    require_cuda(cudaMemcpy(enabled.data(), view_.route_enabled, enabled.size(),
                            cudaMemcpyDeviceToHost), "resident readout owner hash enabled");
    require_cuda(cudaMemcpy(generations.data(), view_.route_generations,
                            generations.size() * sizeof(generations[0]), cudaMemcpyDeviceToHost),
                 "resident readout owner hash generations");
    require_cuda(cudaMemcpy(lanes.data(), view_.lanes,
                            lanes.size() * sizeof(lanes[0]), cudaMemcpyDeviceToHost),
                 "resident readout owner hash lanes");
    require_cuda(cudaMemcpy(&tick, view_.next_tick, sizeof(tick), cudaMemcpyDeviceToHost),
                 "resident readout owner hash tick");
    require_cuda(cudaMemcpy(&positive, view_.positive_supply, sizeof(positive), cudaMemcpyDeviceToHost),
                 "resident readout owner hash positive supply");
    require_cuda(cudaMemcpy(&negative, view_.negative_supply, sizeof(negative), cudaMemcpyDeviceToHost),
                 "resident readout owner hash negative supply");
    require_cuda(cudaMemcpy(&attempted, view_.attempted, sizeof(attempted), cudaMemcpyDeviceToHost),
                 "resident readout owner hash attempts");
    require_cuda(cudaMemcpy(&committed, view_.committed, sizeof(committed), cudaMemcpyDeviceToHost),
                 "resident readout owner hash commits");
    std::uint64_t hash = UINT64_C(1469598103934665603);
    auto mix = [&hash](std::uint64_t value) {
      constexpr std::uint64_t prime = UINT64_C(1099511628211);
      for (std::uint32_t byte = 0u; byte < 8u; ++byte) {
        hash ^= static_cast<std::uint8_t>(value >> (byte * 8u));
        hash *= prime;
      }
    };
    mix(positive);
    mix(negative);
    mix(committed);
    for (const ContextSlot& context : contexts) {
      mix(context.valid);
      mix(context.context_signature);
      mix(context.baseline_byte);
      for (std::int8_t value : context.hidden_prototype)
        mix(static_cast<std::uint8_t>(value));
    }
    for (std::uint32_t id = 0u; id < kContextRouteCount; ++id) {
      mix(route_positive[id]);
      mix(route_negative[id]);
      mix(enabled[id]);
      mix(generations[id]);
    }
    if (include_ephemeral) {
      mix(tick);
      mix(attempted);
      for (const PredictionLane& lane : lanes) {
        mix(lane.tick);
        mix(lane.valid);
        mix(lane.context_signature);
        mix(lane.baseline_byte);
        mix(lane.emitted_byte);
        for (std::int8_t value : lane.hidden_prototype)
          mix(static_cast<std::uint8_t>(value));
        for (std::uint32_t row = 0u; row < kOutputRows; ++row) {
          mix(lane.selected_rows[row]);
        }
      }
    }
    return hash;
  }

  template <typename T>
  static T copy_one(const T* device_value) {
    T host{};
    require_cuda(cudaMemcpy(&host, device_value, sizeof(host), cudaMemcpyDeviceToHost),
                 "resident readout owner receipt copy");
    return host;
  }

  void allocate() {
    require_cuda(cudaMalloc(reinterpret_cast<void**>(&view_.context_slots),
                            kContextSlots * sizeof(ContextSlot)), "allocate context slots");
    require_cuda(cudaMalloc(reinterpret_cast<void**>(&view_.route_positive),
                            kContextRouteCount * sizeof(std::uint32_t)),
                 "allocate positive routes");
    require_cuda(cudaMalloc(reinterpret_cast<void**>(&view_.route_negative),
                            kContextRouteCount * sizeof(std::uint32_t)),
                 "allocate negative routes");
    require_cuda(cudaMalloc(reinterpret_cast<void**>(&view_.route_enabled), kContextRouteCount),
                 "allocate route enabled");
    require_cuda(cudaMalloc(reinterpret_cast<void**>(&view_.route_generations),
                            kContextRouteCount * sizeof(std::uint64_t)), "allocate route generations");
    require_cuda(cudaMalloc(reinterpret_cast<void**>(&view_.lanes),
                            kHorizon * sizeof(PredictionLane)), "allocate prediction lanes");
    require_cuda(cudaMalloc(reinterpret_cast<void**>(&view_.next_tick), sizeof(std::uint64_t)),
                 "allocate owner clock");
    require_cuda(cudaMalloc(reinterpret_cast<void**>(&view_.positive_supply), sizeof(std::uint32_t)),
                 "allocate positive supply");
    require_cuda(cudaMalloc(reinterpret_cast<void**>(&view_.negative_supply), sizeof(std::uint32_t)),
                 "allocate negative supply");
    require_cuda(cudaMalloc(reinterpret_cast<void**>(&view_.attempted), sizeof(std::uint64_t)),
                 "allocate attempt count");
    require_cuda(cudaMalloc(reinterpret_cast<void**>(&view_.committed), sizeof(std::uint64_t)),
                 "allocate commit count");
    require_cuda(cudaMalloc(reinterpret_cast<void**>(&view_.transaction_lock), sizeof(std::uint32_t)),
                 "allocate transaction lock");
    require_cuda(cudaMalloc(reinterpret_cast<void**>(&view_.attempt_receipt), sizeof(AttemptReceipt)),
                 "allocate attempt receipt");
    require_cuda(cudaMalloc(reinterpret_cast<void**>(&view_.committed_receipt), sizeof(CommittedReceipt)),
                 "allocate committed receipt");
    require_cuda(cudaMalloc(reinterpret_cast<void**>(&view_.lesion_receipt), sizeof(LesionReceipt)),
                 "allocate lesion receipt");
  }

  void release() noexcept {
    initialize_graph_.destroy();
    capture_graph_.destroy();
    credit_graph_.destroy();
    inverse_graph_.destroy();
    lesion_graph_.destroy();
    restore_graph_.destroy();
    (void)cudaFree(view_.context_slots);
    (void)cudaFree(view_.route_positive);
    (void)cudaFree(view_.route_negative);
    (void)cudaFree(view_.route_enabled);
    (void)cudaFree(view_.route_generations);
    (void)cudaFree(view_.lanes);
    (void)cudaFree(view_.next_tick);
    (void)cudaFree(view_.positive_supply);
    (void)cudaFree(view_.negative_supply);
    (void)cudaFree(view_.attempted);
    (void)cudaFree(view_.committed);
    (void)cudaFree(view_.transaction_lock);
    (void)cudaFree(view_.attempt_receipt);
    (void)cudaFree(view_.committed_receipt);
    (void)cudaFree(view_.lesion_receipt);
    view_ = DeviceView{};
  }

  void move_from(DeviceOwner& other) noexcept {
    view_ = other.view_;
    other.view_ = DeviceView{};
    initialize_graph_.take_from(other.initialize_graph_);
    capture_graph_.take_from(other.capture_graph_);
    credit_graph_.take_from(other.credit_graph_);
    inverse_graph_.take_from(other.inverse_graph_);
    lesion_graph_.take_from(other.lesion_graph_);
    restore_graph_.take_from(other.restore_graph_);
  }
};

}  // namespace bcc32::resident_readout_plastic_owner
