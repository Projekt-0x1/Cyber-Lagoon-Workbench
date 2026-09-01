#pragma once

#include <cuda_runtime.h>

#include <cstdint>
#include <limits>
#include <stdexcept>

#include "bcc32_cuda_resident_credit_bank.cuh"
#include "bcc32_cuda_resident_synthesis.cuh"

// Shared conditioned-prediction ABI and kernels. Keeping this family outside
// bcc32_cuda_adult_v1.cuh lets prediction/credit contracts compile against the
// resident predictor without importing the complete adult implementation.
namespace bcc32_cuda_adult_v1 {

namespace resident_credit = bcc32::resident_credit;

using ConditionedTransitionKey =
    bcc32_cuda_resident_synthesis::ResidentSubjectTransitionKey;

constexpr std::uint32_t kConditionedTransitionLagCount = 16u;

// Exact device-side account of the route that the production predictor
// actually selected. The record carries no credit and names no host slot.
// Its factor index is a deterministic physical landing site within the
// selected route region; the resident owner supplies and conserves eligibility.
struct ConditionedPredictionWitness {
  ConditionedTransitionKey key{};
  std::uint32_t region = 0u;
  std::uint32_t factor_index = 0xffffffffu;
  std::uint32_t source_event = 0u;
  std::uint32_t valid = 0u;
};

// The transition table is the resident predictor. Before the current contact
// changes it, each learned (anchor, previous) context casts exactly one
// prediction. A different observed next unit emits a signed exact-key carrier
// for the physical organ owner.
struct ConditionedPredictionReceipt {
  std::uint32_t observed_events = 0u;
  std::uint32_t predicted_events = 0u;
  std::uint32_t correct_events = 0u;
  std::uint32_t error_events = 0u;
  std::uint32_t somatic_error_events = 0u;
  std::uint32_t positive_credit_events = 0u;
  std::uint32_t negative_credit_events = 0u;
};

// Deterministic bridge record from the adult predictor into physical learning
// matter. Each decoded event owns slots 2*event (observed positive credit) and
// 2*event+1 (uniquely predicted negative credit). The fixed slot mapping avoids
// atomic append order and preserves the exact conditioned key.
struct ConditionedCreditEvent {
  ConditionedTransitionKey key{};
  std::int32_t polarity = 0;
  std::uint32_t source_event = 0u;
  std::uint32_t valid = 0u;
};

struct AdultState;

// The producer synchronizes its kernels before invoking these callbacks and
// keeps the device allocation alive for the complete call.
using ConditionedDeviceCreditConsumer = void (*)(
    void*, const ConditionedCreditEvent*, std::uint32_t);
using ConditionedPredictionWitnessConsumer = void (*)(
    void*, const ConditionedPredictionWitness*, std::uint32_t);
using ConditionedConductancePublisher = void (*)(void*, AdultState&);

inline std::uint32_t conditioned_transition_event_count(
    std::uint32_t sequence_count, std::uint32_t prefix_count = 0u) {
  unsigned long long events = 0u;
  for (std::uint32_t lag = 0u;
       lag < kConditionedTransitionLagCount; ++lag) {
    if (sequence_count > lag + 1u) {
      const std::uint32_t start = prefix_count > lag + 1u
          ? prefix_count - lag - 1u : 0u;
      events += sequence_count - lag - 1u - start;
    }
  }
  if (events > std::numeric_limits<std::uint32_t>::max()) {
    throw std::runtime_error("conditioned transition event extent overflow");
  }
  return static_cast<std::uint32_t>(events);
}

#if !defined(BCC32_CUDA_ADULT_STATE_ONLY)
#include "bcc32_cuda_adult_conditioned_credit_kernels.inl"
#endif

}  // namespace bcc32_cuda_adult_v1
