#pragma once

// Generic resident two-source composition primitive.
//
// The boundary exposes only opaque raw source words and a contact order.  The
// device learns one local key for each source, then binds an order-sensitive
// pair route in resident state.  A later combined probe can emit only when
// both learned keys and the resident pair route are present.  No host code
// supplies a pair id, answer value, or semantic lookup.

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace substrate::bcc32::resident_pair_composition {

inline constexpr std::uint32_t kMagic = 0x32505231u;  // "2PR1"
inline constexpr std::uint32_t kMatterCapacity = 64u;
inline constexpr std::uint32_t kNoKey = 0u;
inline constexpr std::uint32_t kComposeDisabled = 0u;
inline constexpr std::uint32_t kComposeEnabled = 1u;

// Channel is physical boundary metadata (0/1), not a semantic source label.
// Pair order is derived from the resident contact history below; it is never
// supplied as an operation argument by the host.
inline constexpr std::uint32_t kChannelA = 0u;
inline constexpr std::uint32_t kChannelB = 1u;
inline constexpr std::uint32_t kHistoryAB = 0x41b2u;
inline constexpr std::uint32_t kHistoryBA = 0xb241u;
inline constexpr std::uint32_t kHistoryInvalid = 0u;

// All fields are ordinary resident state.  `free_matter + a_matter +
// b_matter` is conserved by contact and source-cut operations; the route and
// output fields are organization over the same fixed resident allocation.
struct PairState {
  std::uint32_t magic = 0u;
  std::uint32_t free_matter = 0u;
  std::uint32_t a_matter = 0u;
  std::uint32_t b_matter = 0u;
  std::uint32_t a_key = kNoKey;
  std::uint32_t b_key = kNoKey;
  std::uint32_t pair_route = kNoKey;
  std::uint32_t pair_epoch = 0u;
  std::uint32_t interaction_events = 0u;
  std::uint32_t first_channel = 0xffffffffu;
  std::uint32_t last_channel = 0xffffffffu;
  std::uint32_t contact_count = 0u;
  std::uint32_t composition_enabled = kComposeEnabled;
  std::uint32_t output = 0u;
  std::uint32_t output_events = 0u;
};

__host__ __device__ inline std::uint32_t rotate_left(std::uint32_t x,
                                                       std::uint32_t r) {
  r &= 31u;
  return (x << r) | (x >> ((32u - r) & 31u));
}

__host__ __device__ inline std::uint32_t mix(std::uint32_t x) {
  x ^= x >> 16u;
  x *= 0x7feb352du;
  x ^= x >> 15u;
  x *= 0x846ca68bu;
  x ^= x >> 16u;
  return x == kNoKey ? 1u : x;
}

__host__ __device__ inline std::uint32_t source_key(std::uint32_t channel,
                                                     std::uint32_t raw) {
  const std::uint32_t seed = channel == kChannelA ? 0x13579bdfu
                                                  : 0x2468ace1u;
  return mix(seed ^ raw ^ (channel * 0x9e3779b9u));
}

__host__ __device__ inline std::uint32_t history_order(
    std::uint32_t first_channel, std::uint32_t last_channel) {
  if (first_channel == kChannelA && last_channel == kChannelB) return kHistoryAB;
  if (first_channel == kChannelB && last_channel == kChannelA) return kHistoryBA;
  return kHistoryInvalid;
}

__host__ __device__ inline std::uint32_t pair_route_for(
    std::uint32_t a_key, std::uint32_t b_key, std::uint32_t history) {
  const std::uint32_t order_word = history;
  return mix(a_key ^ rotate_left(b_key, 11u) ^ order_word ^ 0xa5a5a5a5u);
}

__global__ inline void initialize_kernel(PairState* state) {
  if (threadIdx.x != 0u || blockIdx.x != 0u) return;
  *state = PairState{};
  state->magic = kMagic;
  state->free_matter = kMatterCapacity;
  state->composition_enabled = kComposeEnabled;
}

// A contact transfers one resident matter unit from the common pool into the
// contacted source population.  The raw word is never interpreted by host
// code; only the resident device keying path sees it.
__global__ inline void source_contact_kernel(PairState* state,
                                              std::uint32_t channel,
                                              std::uint32_t raw) {
  if (threadIdx.x != 0u || blockIdx.x != 0u || state->magic != kMagic ||
      state->free_matter == 0u || channel > kChannelB)
    return;
  --state->free_matter;
  if (channel == kChannelA) {
    ++state->a_matter;
    state->a_key = source_key(channel, raw);
  } else {
    ++state->b_matter;
    state->b_key = source_key(channel, raw);
  }
  if (state->contact_count == 0u) state->first_channel = channel;
  state->last_channel = channel;
  state->contact_count += 1u;
}

// Pair binding is a resident operation after independent contacts.  The
// order is part of the physical contact history, so AB and BA are distinct
// routes even with equal source inventory.
__global__ inline void bind_pair_kernel(PairState* state) {
  if (threadIdx.x != 0u || blockIdx.x != 0u || state->magic != kMagic ||
      state->a_matter == 0u || state->b_matter == 0u ||
      state->a_key == kNoKey || state->b_key == kNoKey ||
      history_order(state->first_channel, state->last_channel) ==
          kHistoryInvalid)
    return;
  state->pair_route = pair_route_for(
      state->a_key, state->b_key,
      history_order(state->first_channel, state->last_channel));
  state->pair_epoch += 1u;
  state->interaction_events += 1u;
}

// The combined probe is raw and held out as a pair: it is checked against the
// two resident source keys and the resident order-sensitive route.  The
// output is generated on device from that route; the host cannot inject it.
__global__ inline void compose_probe_kernel(PairState* state,
                                             std::uint32_t raw_a,
                                             std::uint32_t raw_b) {
  if (threadIdx.x != 0u || blockIdx.x != 0u || state->magic != kMagic) return;
  state->output = 0u;
  if (state->composition_enabled != kComposeEnabled ||
      state->a_matter == 0u || state->b_matter == 0u ||
      state->a_key != source_key(kChannelA, raw_a) ||
      state->b_key != source_key(kChannelB, raw_b))
    return;
  const std::uint32_t expected = pair_route_for(
      state->a_key, state->b_key,
      history_order(state->first_channel, state->last_channel));
  if (state->pair_route != expected || state->interaction_events == 0u) return;
  state->output = mix(expected ^ 0x5bd1e995u);
  state->output_events += 1u;
}

__global__ inline void cut_source_kernel(PairState* state,
                                          std::uint32_t channel) {
  if (threadIdx.x != 0u || blockIdx.x != 0u || state->magic != kMagic) return;
  if (channel == kChannelA) {
    state->free_matter += state->a_matter;
    state->a_matter = 0u;
    state->a_key = kNoKey;
  } else if (channel == kChannelB) {
    state->free_matter += state->b_matter;
    state->b_matter = 0u;
    state->b_key = kNoKey;
  }
  else return;
  state->pair_route = kNoKey;
  state->interaction_events = 0u;
  state->first_channel = 0xffffffffu;
  state->last_channel = 0xffffffffu;
  state->contact_count = 0u;
  state->output = 0u;
}

// Equal-inventory sham: reserves the same amount of matter as a real AB adult
// but does not create source keys or an interaction route.
__global__ inline void equal_inventory_sham_kernel(PairState* state) {
  if (threadIdx.x != 0u || blockIdx.x != 0u || state->magic != kMagic) return;
  state->free_matter = kMatterCapacity - 2u;
  state->a_matter = 1u;
  state->b_matter = 1u;
  state->a_key = kNoKey;
  state->b_key = kNoKey;
  state->pair_route = kNoKey;
  state->interaction_events = 0u;
  state->first_channel = 0xffffffffu;
  state->last_channel = 0xffffffffu;
  state->contact_count = 0u;
  state->output = 0u;
}

// A lesion/control is applied through the same device boundary as ordinary
// resident operations; the host never patches a semantic result into state.
__global__ inline void disable_interaction_kernel(PairState* state) {
  if (threadIdx.x != 0u || blockIdx.x != 0u || state->magic != kMagic) return;
  state->composition_enabled = kComposeDisabled;
  state->output = 0u;
}

}  // namespace substrate::bcc32::resident_pair_composition
