#pragma once

#include <cuda_runtime.h>

#include <cstdint>

namespace bcc32::resident_credit {

constexpr std::uint32_t kSupportLimit = 3u;
constexpr std::uint32_t kEligibilityLifetime = 3u;
constexpr std::uint32_t kInvalidSlot = 0xffffffffu;
constexpr std::uint32_t kPredictContact = 0u;
constexpr std::uint32_t kObserveContact = 1u;
// A predict-phase contact used to overwrite the SINGLE pending scalar below,
// so any earlier still-eligible prediction was silently dropped -- never
// credited positively or negatively -- the instant a later predict call
// happened before the earlier one's observation arrived. kHorizonDepth
// predictions can now stay pending simultaneously; each still-eligible entry
// is offered credit at observe time, not just the most recent one.
constexpr std::uint32_t kHorizonDepth = 3u;

struct RouteKey {
  std::uint32_t anchor;
  std::uint32_t previous;
  std::uint32_t next;
  std::uint32_t region;
};

// Plain route records are caller-owned. The adult can provide a larger array
// than the focused contract without changing this API or the routing law.
struct RouteState {
  RouteKey key;
  std::uint8_t occupied;
  std::uint8_t positive;
  std::uint8_t negative;
  std::uint8_t eligible;
  std::uint32_t expiry_tick;
  // Total positive-credit ATTEMPTS this route has ever seen, uncapped (saturating at
  // 0xffff), independent of kSupportLimit's clamp on `positive`. Two routes that both
  // saturate `positive` at the cap are otherwise permanently indistinguishable even if
  // one was taught 3 times and the other 30 -- this preserves that difference so a tie
  // at the cap can still be broken by which route was taught more, rather than refusing
  // to decide once both sides reach the ceiling.
  std::uint16_t exposure;
};

// The scalar block is plain and serializable alongside the route array.
// pending_* form a fixed-depth FIFO ring of recent predict-phase selections,
// oldest evicted first when full. Each entry ages and expires independently
// by its OWN tick, exactly as the single scalar did before.
struct BankScalars {
  std::uint32_t free_quanta;
  std::uint32_t escrow_quanta;
  std::uint32_t pending_slot[kHorizonDepth];
  std::uint32_t pending_tick[kHorizonDepth];
  std::uint8_t pending_valid[kHorizonDepth];
  std::uint32_t pending_head;  // ring write cursor; next push lands here
  std::uint32_t pending_count;  // number of currently-valid entries, <= kHorizonDepth
};

// Non-owning device view. No fixed-capacity state or host-side route table is
// part of the bank; callers allocate routes/scalars and pass this view.
struct BankView {
  RouteState* routes;
  BankScalars* scalars;
  std::uint32_t capacity;
  std::uint32_t support_limit;
};

// Raw contact: the device decodes six u32 fields from the byte stream:
// anchor, previous, observed next, candidate next 0, candidate next 1, region.
struct RawRouteSequence {
  std::uint8_t bytes[24];
  std::uint32_t length;
};

struct ContactReceipt {
  std::uint32_t active;
  std::uint32_t expired;
  std::uint32_t mismatch;
  std::uint32_t predicted_slot;
  std::uint32_t observed_slot;
  std::uint32_t positive_update;
  std::uint32_t negative_update;
  RouteKey predicted_key;
  RouteKey observed_key;
  std::int32_t predicted_conductance;
  std::int32_t observed_conductance;
};

struct LesionReceipt {
  std::uint32_t valid;
  std::uint32_t slot;
  std::uint8_t positive;
  std::uint8_t negative;
  std::uint8_t reserved[2];
  std::uint32_t escrow_before;
};

__device__ inline std::uint64_t mix_key(std::uint64_t value) {
  value ^= value >> 30u;
  value *= UINT64_C(0xbf58476d1ce4e5b9);
  value ^= value >> 27u;
  value *= UINT64_C(0x94d049bb133111eb);
  return value ^ (value >> 31u);
}

__device__ inline bool same_key(const RouteKey& left, const RouteKey& right) {
  return left.anchor == right.anchor && left.previous == right.previous &&
         left.next == right.next && left.region == right.region;
}

__device__ inline std::uint64_t key_hash(const RouteKey& key) {
  std::uint64_t value = UINT64_C(0x9e3779b97f4a7c15) ^ key.anchor;
  value = mix_key(value ^ key.previous);
  value = mix_key(value ^ key.next);
  return mix_key(value ^ key.region);
}

__device__ inline std::uint32_t find_route(const BankView& view,
                                           const RouteKey& key) {
  if (view.routes == nullptr || view.capacity == 0u) return kInvalidSlot;
  const std::uint32_t home =
      static_cast<std::uint32_t>(key_hash(key) % view.capacity);
  for (std::uint32_t probe = 0u; probe < view.capacity; ++probe) {
    const std::uint32_t slot = (home + probe) % view.capacity;
    if (view.routes[slot].occupied == 0u) return kInvalidSlot;
    if (same_key(view.routes[slot].key, key)) return slot;
  }
  return kInvalidSlot;
}

__device__ inline std::uint32_t route_slot(BankView& view, const RouteKey& key,
                                            bool insert) {
  const std::uint32_t existing = find_route(view, key);
  if (existing != kInvalidSlot || !insert || view.routes == nullptr ||
      view.capacity == 0u)
    return existing;
  const std::uint32_t home =
      static_cast<std::uint32_t>(key_hash(key) % view.capacity);
  for (std::uint32_t probe = 0u; probe < view.capacity; ++probe) {
    const std::uint32_t slot = (home + probe) % view.capacity;
    if (view.routes[slot].occupied == 0u) {
      view.routes[slot] = RouteState{};
      view.routes[slot].key = key;
      view.routes[slot].occupied = 1u;
      return slot;
    }
  }
  return kInvalidSlot;
}

__device__ inline std::int32_t conductance(const RouteState& route) {
  return static_cast<std::int32_t>(route.positive) -
         static_cast<std::int32_t>(route.negative);
}

__host__ __device__ inline RouteKey make_route_key(
    std::uint32_t anchor, std::uint32_t previous, std::uint32_t next,
    std::uint32_t region) {
  return RouteKey{anchor, previous, next, region};
}

__device__ inline std::uint32_t route_region(std::uint32_t anchor,
                                             std::uint32_t previous) {
  std::uint32_t value = anchor ^ (previous + 0x9e3779b9u +
                                  (anchor << 6u) + (anchor >> 2u));
  value ^= value >> 16u;
  return value & 0xffu;
}

// Apply one signed pulse to a route that participated in the current local
// context. The caller supplies a key derived on device; no host route index is
// accepted. Positive and negative support consume the same conserved pool.
__device__ inline bool apply_signed_credit(BankView& view, const RouteKey& key,
                                           std::int32_t polarity,
                                           std::uint32_t tick) {
  if (view.scalars == nullptr || polarity == 0) return false;
  // Exhaustion must abstain before open addressing claims a slot. Otherwise
  // zero-support routes accumulate as permanent hash-table ghosts.
  if (view.scalars->free_quanta == 0u) return false;
  const std::uint32_t slot = route_slot(view, key, true);
  if (slot >= view.capacity) return false;
  RouteState& route = view.routes[slot];
  route.eligible = 1u;
  route.expiry_tick = tick + kEligibilityLifetime;
  if (polarity > 0) {
    if (route.exposure < 0xffffu) ++route.exposure;
    if (route.positive < view.support_limit) {
      ++route.positive;
      --view.scalars->free_quanta;
      route.eligible = 0u;
      return true;
    }
  }
  if (polarity < 0 && route.negative < view.support_limit) {
    ++route.negative;
    --view.scalars->free_quanta;
    route.eligible = 0u;
    return true;
  }
  route.eligible = 0u;
  return false;
}

__device__ inline std::uint32_t decode_u32(const RawRouteSequence& sequence,
                                            std::uint32_t offset) {
  return static_cast<std::uint32_t>(sequence.bytes[offset]) |
         (static_cast<std::uint32_t>(sequence.bytes[offset + 1u]) << 8u) |
         (static_cast<std::uint32_t>(sequence.bytes[offset + 2u]) << 16u) |
         (static_cast<std::uint32_t>(sequence.bytes[offset + 3u]) << 24u);
}

__device__ inline bool decode_route_sequence(const RawRouteSequence& sequence,
                                             RouteKey* observed,
                                             RouteKey* candidates) {
  if (sequence.length < 24u || observed == nullptr || candidates == nullptr)
    return false;
  const std::uint32_t anchor = decode_u32(sequence, 0u);
  const std::uint32_t previous = decode_u32(sequence, 4u);
  const std::uint32_t observed_next = decode_u32(sequence, 8u);
  const std::uint32_t candidate0 = decode_u32(sequence, 12u);
  const std::uint32_t candidate1 = decode_u32(sequence, 16u);
  const std::uint32_t region = decode_u32(sequence, 20u);
  if (anchor == 0u || previous == 0u || candidate0 == 0u || candidate1 == 0u)
    return false;
  observed->anchor = anchor;
  observed->previous = previous;
  observed->next = observed_next;
  observed->region = region;
  candidates[0] = *observed;
  candidates[0].next = candidate0;
  candidates[1] = *observed;
  candidates[1].next = candidate1;
  return observed_next != 0u;
}

__device__ inline void clear_contact_receipt(ContactReceipt* receipt) {
  if (receipt != nullptr) *receipt = ContactReceipt{};
}

__device__ inline void clear_lesion_receipt(LesionReceipt* receipt) {
  if (receipt != nullptr) *receipt = LesionReceipt{};
}

static __global__ void resident_credit_bank_init_kernel(
    BankView view, std::uint32_t free_quanta) {
  if (view.routes == nullptr || view.scalars == nullptr) return;
  for (std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < view.capacity;
       i += blockDim.x * gridDim.x) {
    view.routes[i] = RouteState{};
  }
  if (blockIdx.x == 0u && threadIdx.x == 0u) {
    *view.scalars = BankScalars{};
    view.scalars->free_quanta = free_quanta;
    for (std::uint32_t i = 0u; i < kHorizonDepth; ++i)
      view.scalars->pending_slot[i] = kInvalidSlot;
  }
}

// One resident contact kernel handles both phases. The prediction phase marks
// all candidate routes eligible, selects the max-conductance next route, and
// PUSHES that route onto a fixed-depth pending ring (evicting the oldest
// entry, uncredited, if the ring is full) rather than overwriting a single
// scalar. The later observation phase decodes the observed next route and
// walks every still-eligible, unexpired ring entry: any entry whose own
// predicted route matches the observed outcome is credited positive; any
// other still-eligible entry is credited negative. This is what lets credit
// reach back through several hops instead of only ever crediting the single
// most recent prediction.
static __global__ void resident_credit_bank_contact_kernel(
    BankView view, const RawRouteSequence* sequences, std::uint32_t sequence_index,
    std::uint32_t tick, std::uint32_t phase, ContactReceipt* receipt) {
  if (blockIdx.x != 0u || threadIdx.x != 0u || view.routes == nullptr ||
      view.scalars == nullptr || sequences == nullptr)
    return;
  clear_contact_receipt(receipt);
  RouteKey observed{};
  RouteKey candidates[2]{};
  if (!decode_route_sequence(sequences[sequence_index], &observed, candidates)) return;
  if (receipt != nullptr) {
    receipt->active = 1u;
    receipt->observed_key = observed;
  }

  if (phase == kPredictContact) {
    std::uint32_t slots[2] = {kInvalidSlot, kInvalidSlot};
    std::int32_t scores[2] = {0, 0};
    for (std::uint32_t i = 0u; i < 2u; ++i) {
      slots[i] = route_slot(view, candidates[i], true);
      if (slots[i] >= view.capacity) continue;
      RouteState& route = view.routes[slots[i]];
      if (route.eligible != 0u && tick > route.expiry_tick) route.eligible = 0u;
      route.eligible = 1u;
      route.expiry_tick = tick + kEligibilityLifetime;
      scores[i] = conductance(route);
    }
    const std::uint32_t selected =
        (slots[1] < view.capacity && scores[1] > scores[0]) ? 1u : 0u;
    if (slots[selected] < view.capacity) {
      const std::uint32_t write = view.scalars->pending_head;
      if (view.scalars->pending_valid[write] != 0u) {
        // Ring full: the entry being evicted never resolved. Its route stops
        // being eligible so a later, unrelated observe cannot mistakenly
        // credit it -- the same outcome an expiry already produces.
        const std::uint32_t evicted = view.scalars->pending_slot[write];
        if (evicted < view.capacity) view.routes[evicted].eligible = 0u;
      } else {
        ++view.scalars->pending_count;
      }
      view.scalars->pending_slot[write] = slots[selected];
      view.scalars->pending_tick[write] = tick;
      view.scalars->pending_valid[write] = 1u;
      view.scalars->pending_head = (write + 1u) % kHorizonDepth;
      if (receipt != nullptr) {
        receipt->predicted_slot = slots[selected];
        receipt->predicted_key = candidates[selected];
        receipt->predicted_conductance = scores[selected];
      }
    }
    return;
  }

  const std::uint32_t observed_slot = find_route(view, observed);
  const std::uint32_t candidate0_slot = find_route(view, candidates[0]);
  const std::uint32_t candidate1_slot = find_route(view, candidates[1]);
  const bool observed_is_candidate = observed_slot == candidate0_slot ||
                                     observed_slot == candidate1_slot;
  if (receipt != nullptr) {
    receipt->observed_slot = observed_slot;
    receipt->observed_conductance =
        observed_slot < view.capacity ? conductance(view.routes[observed_slot]) : 0;
  }

  bool any_pending = false;
  std::int32_t newest_predicted_conductance = 0;
  RouteKey newest_predicted_key{};
  bool mismatch_seen = false;
  for (std::uint32_t offset = 0u; offset < kHorizonDepth; ++offset) {
    // Walk newest-first: (head - 1 - offset) mod depth.
    const std::uint32_t index =
        (view.scalars->pending_head + kHorizonDepth - 1u - offset) % kHorizonDepth;
    if (view.scalars->pending_valid[index] == 0u) continue;
    const std::uint32_t predicted_slot = view.scalars->pending_slot[index];
    if (predicted_slot >= view.capacity) {
      view.scalars->pending_valid[index] = 0u;
      --view.scalars->pending_count;
      continue;
    }
    // CONTEXT GATE: this observe call is about ONE local situation (observed's
    // own anchor/previous/region). A pending entry from a DIFFERENT situation
    // is not confirmed OR refuted by it -- leave it untouched, still pending,
    // for its own future matching observe. Without this gate an unrelated
    // observation would credit or debit guesses about a completely different
    // context, which is not what "reaches back through several hops of the
    // SAME unresolved thread" means.
    const RouteKey& predicted_key = view.routes[predicted_slot].key;
    const bool same_context = predicted_key.anchor == observed.anchor &&
                              predicted_key.previous == observed.previous &&
                              predicted_key.region == observed.region;
    if (!same_context) continue;
    any_pending = true;
    if (receipt != nullptr && offset == 0u) {
      receipt->predicted_slot = predicted_slot;
      newest_predicted_key = predicted_key;
      newest_predicted_conductance = conductance(view.routes[predicted_slot]);
    }
    const bool expired = tick > view.scalars->pending_tick[index] + kEligibilityLifetime;
    if (expired || view.routes[predicted_slot].eligible == 0u ||
        !observed_is_candidate) {
      if (offset == 0u && receipt != nullptr && expired) receipt->expired = 1u;
      view.routes[predicted_slot].eligible = 0u;
      view.scalars->pending_valid[index] = 0u;
      --view.scalars->pending_count;
      continue;
    }
    const bool mismatch = predicted_slot != observed_slot;
    if (mismatch) mismatch_seen = true;
    // Observed gets positive credit whenever it actually happened and was a
    // live candidate -- true on a correct guess (observed==predicted) AND on
    // a mismatch (what actually happened is reinforced regardless of what
    // was guessed). Negative credit is separate and applies ONLY to the
    // wrong guess. Gating the positive branch on "!mismatch" (as this read
    // before) meant a wrong guess taught nothing about what actually
    // happened -- only punished the guess -- which is not a prediction-error
    // signal, just a punishment signal.
    if (view.routes[observed_slot].eligible != 0u &&
        view.routes[observed_slot].positive < view.support_limit &&
        view.scalars->free_quanta != 0u) {
      ++view.routes[observed_slot].positive;
      --view.scalars->free_quanta;
      if (receipt != nullptr) receipt->positive_update = 1u;
    }
    if (mismatch && view.routes[predicted_slot].negative < view.support_limit &&
        view.scalars->free_quanta != 0u) {
      ++view.routes[predicted_slot].negative;
      --view.scalars->free_quanta;
      if (receipt != nullptr) receipt->negative_update = 1u;
    }
    view.routes[predicted_slot].eligible = 0u;
    view.scalars->pending_valid[index] = 0u;
    --view.scalars->pending_count;
  }
  if (receipt != nullptr) {
    receipt->predicted_key = newest_predicted_key;
    receipt->predicted_conductance = newest_predicted_conductance;
    receipt->mismatch = mismatch_seen ? 1u : 0u;
  }
  if (!any_pending) return;
  if (candidate0_slot < view.capacity)
    view.routes[candidate0_slot].eligible = 0u;
  if (candidate1_slot < view.capacity)
    view.routes[candidate1_slot].eligible = 0u;
}

static __global__ void resident_credit_bank_read_candidate_conductance_kernel(
    BankView view, const RawRouteSequence* sequences, std::uint32_t sequence_index,
    std::uint32_t candidate_index, std::int32_t* output) {
  if (blockIdx.x != 0u || threadIdx.x != 0u || output == nullptr ||
      view.routes == nullptr || sequences == nullptr || candidate_index >= 2u)
    return;
  *output = 0;
  RouteKey observed{};
  RouteKey candidates[2]{};
  if (!decode_route_sequence(sequences[sequence_index], &observed, candidates)) return;
  const std::uint32_t slot = find_route(view, candidates[candidate_index]);
  if (slot < view.capacity) *output = conductance(view.routes[slot]);
}

static __global__ void resident_credit_bank_capture_kernel(
    BankView view, RouteState* route_snapshot, BankScalars* scalar_snapshot) {
  if (view.routes == nullptr || view.scalars == nullptr || route_snapshot == nullptr ||
      scalar_snapshot == nullptr)
    return;
  for (std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < view.capacity;
       i += blockDim.x * gridDim.x)
    route_snapshot[i] = view.routes[i];
  if (blockIdx.x == 0u && threadIdx.x == 0u) *scalar_snapshot = *view.scalars;
}

static __global__ void resident_credit_bank_restore_kernel(
    BankView view, const RouteState* route_snapshot,
    const BankScalars* scalar_snapshot) {
  if (view.routes == nullptr || view.scalars == nullptr || route_snapshot == nullptr ||
      scalar_snapshot == nullptr)
    return;
  for (std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < view.capacity;
       i += blockDim.x * gridDim.x)
    view.routes[i] = route_snapshot[i];
  if (blockIdx.x == 0u && threadIdx.x == 0u) *view.scalars = *scalar_snapshot;
}

static __global__ void resident_credit_bank_lesion_kernel(
    BankView view, const RawRouteSequence* sequences, std::uint32_t sequence_index,
    LesionReceipt* receipt) {
  if (blockIdx.x != 0u || threadIdx.x != 0u || view.routes == nullptr ||
      view.scalars == nullptr || sequences == nullptr)
    return;
  clear_lesion_receipt(receipt);
  RouteKey observed{};
  RouteKey candidates[2]{};
  if (!decode_route_sequence(sequences[sequence_index], &observed, candidates)) return;
  const std::uint32_t slot = find_route(view, observed);
  if (slot >= view.capacity) return;
  if (receipt != nullptr) {
    receipt->valid = 1u;
    receipt->slot = slot;
    receipt->positive = view.routes[slot].positive;
    receipt->negative = view.routes[slot].negative;
    receipt->escrow_before = view.scalars->escrow_quanta;
  }
  view.scalars->escrow_quanta +=
      view.routes[slot].positive + view.routes[slot].negative;
  view.routes[slot].positive = 0u;
  view.routes[slot].negative = 0u;
  view.routes[slot].eligible = 0u;
}

static __global__ void resident_credit_bank_restore_lesion_kernel(
    BankView view, const LesionReceipt* receipt) {
  if (blockIdx.x != 0u || threadIdx.x != 0u || view.routes == nullptr ||
      view.scalars == nullptr || receipt == nullptr || receipt->valid == 0u ||
      receipt->slot >= view.capacity)
    return;
  view.routes[receipt->slot].positive = receipt->positive;
  view.routes[receipt->slot].negative = receipt->negative;
  view.scalars->escrow_quanta = receipt->escrow_before;
}

}  // namespace bcc32::resident_credit
