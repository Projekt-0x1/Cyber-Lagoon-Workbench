#pragma once

#include "causal_rewrite_universe.cuh"

#include <cstdint>

#if defined(__CUDACC__)
#define BCC32_COMPETITIVE_MEDIUM_HD __host__ __device__
#define BCC32_COMPETITIVE_MEDIUM_NOINLINE __host__ __device__ __noinline__
#else
#define BCC32_COMPETITIVE_MEDIUM_HD
#if defined(__GNUC__) || defined(__clang__)
#define BCC32_COMPETITIVE_MEDIUM_NOINLINE __attribute__((noinline))
#else
#define BCC32_COMPETITIVE_MEDIUM_NOINLINE
#endif
#endif

namespace substrate::bcc32::resident_competitive_medium {
namespace rewrite = substrate::bcc32::causal_rewrite;

// 0X1-253 Assay-A fixture chemistry only.
//
// These forms deliberately do NOT live in causal_rewrite_universe.cuh yet.
// The assay is intended to falsify one candidate generic local chemistry
// before that chemistry earns a production ABI or developmental path.
//
// Route:
// lane[0] fixture route form
// lane[1] opaque route owner
// lane[2] opaque source locator
// lane[3] opaque target locator
// lane[4] opaque shared-medium locator
// lane[5] matter draw per traversal, q8 integer units
// lane[6] transport delay
// lane[7] zero
//
// Medium:
// lane[0] fixture medium form
// lane[1] opaque medium locator
// lane[2] available matter q8
// lane[3] refractory/spent matter q8
// lane[4] recovery matter q8 per epoch
// lane[5] invariant total capacity q8
// lane[6..7] zero
//
// Carrier:
// lane[0] fixture carrier form
// lane[1] opaque route owner
// lane[2] current opaque locator
// lane[3] local age
// lane[4] remaining transport delay
// lane[5..7] zero
//
// The core physical invariant is:
//
//   available_q8 + refractory_q8 == capacity_q8
//
// Traversal does not destroy resource. It moves draw_q8 from available to
// refractory matter. Epoch recovery moves at most recovery_q8 back. Thus
// simultaneously active routes that reference the same medium physically
// compete for one conserved local pool, while routes referencing a disjoint
// medium cannot observe that depletion.
//
// No value here is a confidence, answer score, top-k budget, semantic class,
// excitation/inhibition scalar, or host-selected behavioral objective.

inline constexpr std::uint32_t kFormCompetitiveRoute = 0x4f3c8a21u;
inline constexpr std::uint32_t kFormCompetitiveMedium = 0x9a61d5e7u;
inline constexpr std::uint32_t kFormCompetitiveCarrier = 0xc2476b35u;

inline constexpr std::uint32_t kScratchCapacity = 64u;

struct AdvanceReceipt {
  std::uint32_t carriers_before = 0u;
  std::uint32_t carriers_after = 0u;
  std::uint32_t attempted = 0u;
  std::uint32_t transferred = 0u;
  std::uint32_t blocked = 0u;
  std::uint32_t delay_advanced = 0u;
  std::uint32_t extinguished = 0u;
  std::uint32_t malformed = 0u;
  std::uint32_t depleted_q8 = 0u;
  std::uint32_t recovered_q8 = 0u;
};

BCC32_COMPETITIVE_MEDIUM_HD inline bool valid_key(std::uint32_t value) {
  return value != 0u && value != rewrite::kInvalid;
}

BCC32_COMPETITIVE_MEDIUM_HD inline bool is_route(
    const rewrite::Record& record) {
  return record.matter_q8 != 0u &&
         record.lane[0] == kFormCompetitiveRoute &&
         valid_key(record.lane[1]) &&
         valid_key(record.lane[2]) &&
         valid_key(record.lane[3]) &&
         valid_key(record.lane[4]) &&
         record.lane[5] != 0u &&
         record.lane[7] == 0u &&
         record.reserved[0] == 0u &&
         record.reserved[1] == 0u;
}

BCC32_COMPETITIVE_MEDIUM_HD inline bool is_medium(
    const rewrite::Record& record) {
  if (record.matter_q8 == 0u ||
      record.lane[0] != kFormCompetitiveMedium ||
      !valid_key(record.lane[1]) ||
      record.lane[4] == 0u ||
      record.lane[5] == 0u ||
      record.lane[6] != 0u ||
      record.lane[7] != 0u ||
      record.reserved[0] != 0u ||
      record.reserved[1] != 0u)
    return false;

  const std::uint64_t represented =
      static_cast<std::uint64_t>(record.lane[2]) +
      static_cast<std::uint64_t>(record.lane[3]);
  return represented == static_cast<std::uint64_t>(record.lane[5]);
}

BCC32_COMPETITIVE_MEDIUM_HD inline bool is_carrier(
    const rewrite::Record& record) {
  return record.matter_q8 != 0u &&
         record.lane[0] == kFormCompetitiveCarrier &&
         valid_key(record.lane[1]) &&
         valid_key(record.lane[2]) &&
         record.lane[5] == 0u &&
         record.lane[6] == 0u &&
         record.lane[7] == 0u &&
         record.reserved[0] == 0u &&
         record.reserved[1] == 0u;
}

BCC32_COMPETITIVE_MEDIUM_HD inline bool seed_medium_for_contract(
    rewrite::ResidentRewriteState* state,
    std::uint32_t medium_key,
    std::uint32_t capacity_q8,
    std::uint32_t recovery_q8) {
  if (state == nullptr ||
      !valid_key(medium_key) ||
      capacity_q8 == 0u ||
      recovery_q8 == 0u ||
      recovery_q8 > capacity_q8)
    return false;

  const std::uint32_t slot = rewrite::allocate_record(state);
  if (slot == rewrite::kInvalid) return false;

  rewrite::Record& record = state->records[slot];
  const std::uint32_t revision = record.revision + 1u;
  record = rewrite::Record{};
  record.lane[0] = kFormCompetitiveMedium;
  record.lane[1] = medium_key;
  record.lane[2] = capacity_q8;
  record.lane[3] = 0u;
  record.lane[4] = recovery_q8;
  record.lane[5] = capacity_q8;
  record.revision = revision;
  return true;
}

BCC32_COMPETITIVE_MEDIUM_HD inline bool seed_route_for_contract(
    rewrite::ResidentRewriteState* state,
    std::uint32_t owner,
    std::uint32_t from,
    std::uint32_t to,
    std::uint32_t medium_key,
    std::uint32_t draw_q8,
    std::uint32_t delay) {
  if (state == nullptr ||
      !valid_key(owner) ||
      !valid_key(from) ||
      !valid_key(to) ||
      !valid_key(medium_key) ||
      draw_q8 == 0u)
    return false;

  const std::uint32_t slot = rewrite::allocate_record(state);
  if (slot == rewrite::kInvalid) return false;

  rewrite::Record& record = state->records[slot];
  const std::uint32_t revision = record.revision + 1u;
  record = rewrite::Record{};
  record.lane[0] = kFormCompetitiveRoute;
  record.lane[1] = owner;
  record.lane[2] = from;
  record.lane[3] = to;
  record.lane[4] = medium_key;
  record.lane[5] = draw_q8;
  record.lane[6] = delay;
  record.revision = revision;
  return true;
}

BCC32_COMPETITIVE_MEDIUM_HD inline bool seed_carrier_for_contract(
    rewrite::ResidentRewriteState* state,
    std::uint32_t owner,
    std::uint32_t state_key) {
  if (state == nullptr || !valid_key(owner) || !valid_key(state_key))
    return false;

  const std::uint32_t slot = rewrite::allocate_record(state);
  if (slot == rewrite::kInvalid) return false;

  rewrite::Record& record = state->records[slot];
  const std::uint32_t revision = record.revision + 1u;
  record = rewrite::Record{};
  record.lane[0] = kFormCompetitiveCarrier;
  record.lane[1] = owner;
  record.lane[2] = state_key;
  record.revision = revision;
  return true;
}

BCC32_COMPETITIVE_MEDIUM_NOINLINE std::uint32_t find_unique_route(
    const rewrite::ResidentRewriteState& state,
    std::uint32_t owner,
    std::uint32_t from,
    bool* ambiguous) {
  if (ambiguous != nullptr) *ambiguous = false;
  std::uint32_t found = rewrite::kInvalid;
  const std::uint32_t capacity = rewrite::live_record_capacity(&state);
  for (std::uint32_t slot = 0u; slot < capacity; ++slot) {
    const rewrite::Record& record = state.records[slot];
    if (!is_route(record) ||
        record.lane[1] != owner ||
        record.lane[2] != from)
      continue;
    if (found != rewrite::kInvalid) {
      if (ambiguous != nullptr) *ambiguous = true;
      return rewrite::kInvalid;
    }
    found = slot;
  }
  return found;
}

BCC32_COMPETITIVE_MEDIUM_NOINLINE std::uint32_t find_unique_medium(
    const rewrite::ResidentRewriteState& state,
    std::uint32_t medium_key,
    bool* ambiguous) {
  if (ambiguous != nullptr) *ambiguous = false;
  std::uint32_t found = rewrite::kInvalid;
  const std::uint32_t capacity = rewrite::live_record_capacity(&state);
  for (std::uint32_t slot = 0u; slot < capacity; ++slot) {
    const rewrite::Record& record = state.records[slot];
    if (!is_medium(record) || record.lane[1] != medium_key) continue;
    if (found != rewrite::kInvalid) {
      if (ambiguous != nullptr) *ambiguous = true;
      return rewrite::kInvalid;
    }
    found = slot;
  }
  return found;
}

BCC32_COMPETITIVE_MEDIUM_NOINLINE void recover_media(
    rewrite::ResidentRewriteState* state,
    AdvanceReceipt* receipt) {
  if (state == nullptr || receipt == nullptr) return;
  const std::uint32_t capacity = rewrite::live_record_capacity(state);
  for (std::uint32_t slot = 0u; slot < capacity; ++slot) {
    rewrite::Record& medium = state->records[slot];
    if (!is_medium(medium)) continue;

    const std::uint32_t recover =
        medium.lane[3] < medium.lane[4] ? medium.lane[3] : medium.lane[4];
    if (recover == 0u) continue;

    medium.lane[3] -= recover;
    medium.lane[2] += recover;
    ++medium.revision;
    receipt->recovered_q8 += recover;
  }
}

// `depletion_enabled` exists only as the Assay-A mechanism knockout. The
// positive chemistry always passes true. Passing false leaves traversal
// topology and activation unchanged while removing exactly the candidate
// available->refractory conversion, so the falsifier can prove that the
// competition phenotype depends on this chemistry rather than the fixture's
// route identities or carrier census.
BCC32_COMPETITIVE_MEDIUM_NOINLINE AdvanceReceipt advance(
    rewrite::ResidentRewriteState* state,
    bool depletion_enabled) {
  AdvanceReceipt receipt{};
  if (state == nullptr) return receipt;

  recover_media(state, &receipt);

  const std::uint32_t capacity = rewrite::live_record_capacity(state);
  std::uint32_t slots[kScratchCapacity]{};
  std::uint32_t owners[kScratchCapacity]{};
  std::uint32_t revisions[kScratchCapacity]{};
  std::uint32_t count = 0u;

  for (std::uint32_t slot = 0u; slot < capacity; ++slot) {
    const rewrite::Record& carrier = state->records[slot];
    if (!is_carrier(carrier)) continue;
    ++receipt.carriers_before;
    if (count == kScratchCapacity) {
      ++receipt.malformed;
      continue;
    }
    slots[count] = slot;
    owners[count] = carrier.lane[1];
    revisions[count] = carrier.revision;
    ++count;
  }

  // Stable logical order rather than allocation-slot order. This prevents a
  // Record permutation from changing which simultaneous microscopic event is
  // visited first. The ordering uses only opaque physical identity.
  for (std::uint32_t i = 1u; i < count; ++i) {
    const std::uint32_t slot = slots[i];
    const std::uint32_t owner = owners[i];
    const std::uint32_t revision = revisions[i];
    std::uint32_t j = i;
    while (j > 0u &&
           (owners[j - 1u] > owner ||
            (owners[j - 1u] == owner &&
             revisions[j - 1u] > revision))) {
      slots[j] = slots[j - 1u];
      owners[j] = owners[j - 1u];
      revisions[j] = revisions[j - 1u];
      --j;
    }
    slots[j] = slot;
    owners[j] = owner;
    revisions[j] = revision;
  }

  for (std::uint32_t i = 0u; i < count; ++i) {
    rewrite::Record& carrier = state->records[slots[i]];
    if (!is_carrier(carrier) || carrier.revision != revisions[i]) {
      ++receipt.malformed;
      continue;
    }

    if (carrier.lane[4] != 0u) {
      --carrier.lane[4];
      ++carrier.lane[3];
      ++carrier.revision;
      ++receipt.delay_advanced;
      continue;
    }

    bool route_ambiguous = false;
    const std::uint32_t route_slot =
        find_unique_route(*state, carrier.lane[1], carrier.lane[2],
                          &route_ambiguous);
    if (route_ambiguous) {
      ++receipt.malformed;
      continue;
    }
    if (route_slot == rewrite::kInvalid) {
      rewrite::clear_record(&carrier);
      ++receipt.extinguished;
      continue;
    }

    const rewrite::Record& route = state->records[route_slot];
    bool medium_ambiguous = false;
    const std::uint32_t medium_slot =
        find_unique_medium(*state, route.lane[4], &medium_ambiguous);
    if (medium_ambiguous || medium_slot == rewrite::kInvalid) {
      ++receipt.malformed;
      continue;
    }

    rewrite::Record& medium = state->records[medium_slot];
    const std::uint32_t draw_q8 = route.lane[5];
    ++receipt.attempted;

    if (depletion_enabled) {
      if (medium.lane[2] < draw_q8) {
        ++receipt.blocked;
        continue;
      }

      medium.lane[2] -= draw_q8;
      medium.lane[3] += draw_q8;
      ++medium.revision;
      receipt.depleted_q8 += draw_q8;

      if (!is_medium(medium)) {
        ++receipt.malformed;
        continue;
      }
    }

    carrier.lane[2] = route.lane[3];
    carrier.lane[3] = 0u;
    carrier.lane[4] = route.lane[6];
    ++carrier.revision;
    ++receipt.transferred;
  }

  for (std::uint32_t slot = 0u; slot < capacity; ++slot)
    if (is_carrier(state->records[slot])) ++receipt.carriers_after;

  return receipt;
}

}  // namespace substrate::bcc32::resident_competitive_medium

#undef BCC32_COMPETITIVE_MEDIUM_HD
#undef BCC32_COMPETITIVE_MEDIUM_NOINLINE
