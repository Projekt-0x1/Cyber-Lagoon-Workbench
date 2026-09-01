#pragma once

#include <cstdint>
#include <type_traits>

namespace substrate::bcc32::persistent_kernel::egress_history {

// `append` is the one egress-history mutation invoked directly from the
// continuing resident graph root.  Keep its bounded bookkeeping in its own
// device frame: the complete history still lives in DeviceState's device
// allocation, while the graph root carries only the State pointer.  This is a
// placement boundary, not a second history, clock, writer, or allocation path.
#if defined(__CUDACC__)
#define BCC32_EGRESS_HISTORY_ROOT_CALL __noinline__
#else
#define BCC32_EGRESS_HISTORY_ROOT_CALL
#endif

// Capacity retains both former public windows plus one maximum raw-contact
// width of mixed overlap. It bounds storage only; sequence identity and
// explicit gaps continue across wrap.
inline constexpr std::uint64_t capacity = 512u + 256u + 64u;
inline constexpr std::uint64_t max_sequence = ~std::uint64_t{0};
// This header intentionally does not include the rewrite-universe constants.
// Integration must pass a resident rewrite locus in [0, kRecordCapacity) and
// must never use this local sentinel as a producer locus.
inline constexpr std::uint32_t kInvalidProducerLocus = ~std::uint32_t{0};

struct Event {
  std::uint64_t sequence;
  std::uint64_t completed_tick;
  std::uint32_t raw_word;
  std::uint32_t producer_locus;
};

struct State {
  Event events[capacity]{};
  std::uint64_t next_sequence = 1;
  std::uint64_t oldest_sequence = 1;
  std::uint64_t overwrite_count = 0;
  std::uint32_t fault = 0;
  std::uint32_t padding = 0;
};

static_assert(std::is_trivially_copyable_v<Event>);
static_assert(std::is_trivially_copyable_v<State>);
static_assert(sizeof(Event) == 24u,
              "producer lineage must reuse the existing event padding");
static_assert(capacity == 832u,
              "the graph-stack repair must not alter egress retention capacity");
static_assert(sizeof(State) == capacity * sizeof(Event) + 32u,
              "egress history must remain one bounded resident State allocation");

[[nodiscard]] __host__ __device__ BCC32_EGRESS_HISTORY_ROOT_CALL inline bool append(
    State* state, std::uint32_t raw_word, std::uint64_t completed_tick,
    std::uint32_t producer_locus) {
  if (state == nullptr) {
    return false;
  }
  if (producer_locus == kInvalidProducerLocus) {
    return false;
  }
  if (state->fault != 0u) {
    return false;
  }
  if (state->next_sequence == 0u || state->oldest_sequence == 0u ||
      state->oldest_sequence > state->next_sequence ||
      state->next_sequence - state->oldest_sequence > capacity ||
      state->next_sequence >= max_sequence - 1u) {
    state->fault = 1;
    return false;
  }

  const std::uint64_t sequence = state->next_sequence;
  const bool overwrites = sequence - state->oldest_sequence >= capacity;
  if (overwrites && state->overwrite_count == max_sequence) {
    state->fault = 1;
    return false;
  }
  const std::uint64_t index = (sequence - 1) % capacity;
  state->events[index] = Event{sequence, completed_tick, raw_word, producer_locus};
  ++state->next_sequence;
  if (overwrites) {
    ++state->oldest_sequence;
    ++state->overwrite_count;
  }
  return true;
}

[[nodiscard]] __host__ __device__ inline bool has_producer_locus(
    const Event* event, std::uint32_t producer_locus) {
  return event != nullptr && producer_locus != kInvalidProducerLocus &&
         event->producer_locus == producer_locus;
}

[[nodiscard]] __host__ __device__ inline std::uint64_t newest_sequence(const State* state) {
  return state == nullptr || state->next_sequence == 0 ? 0 : state->next_sequence - 1;
}

[[nodiscard]] __host__ __device__ inline std::uint64_t retained_count(const State* state) {
  if (state == nullptr || state->next_sequence <= state->oldest_sequence) {
    return 0;
  }
  return state->next_sequence - state->oldest_sequence;
}

[[nodiscard]] __host__ __device__ inline bool gap_before(const State* state,
                                                         std::uint64_t sequence) {
  return state != nullptr && sequence != 0 && sequence < state->oldest_sequence;
}

[[nodiscard]] __host__ __device__ inline bool lookup(const State* state, std::uint64_t sequence,
                                                     Event* event) {
  if (state == nullptr || event == nullptr || sequence == 0 || sequence < state->oldest_sequence ||
      sequence >= state->next_sequence) {
    return false;
  }
  const Event candidate = state->events[(sequence - 1) % capacity];
  if (candidate.sequence != sequence) {
    return false;
  }
  *event = candidate;
  return true;
}

}  // namespace substrate::bcc32::persistent_kernel::egress_history

#undef BCC32_EGRESS_HISTORY_ROOT_CALL
