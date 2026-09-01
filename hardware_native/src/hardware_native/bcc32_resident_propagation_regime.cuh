#pragma once

#include "causal_rewrite_universe.cuh"

#include <cstdint>

#if defined(__CUDACC__)
#define BCC32_PROPAGATION_REGIME_HD __host__ __device__ __forceinline__
#define BCC32_PROPAGATION_REGIME_NOINLINE __host__ __device__ __noinline__
#else
#define BCC32_PROPAGATION_REGIME_HD inline
#if defined(__GNUC__) || defined(__clang__)
#define BCC32_PROPAGATION_REGIME_NOINLINE __attribute__((noinline))
#else
#define BCC32_PROPAGATION_REGIME_NOINLINE
#endif
#endif

namespace substrate::bcc32::resident_propagation_regime {
namespace rewrite = substrate::bcc32::causal_rewrite;

// 0X1-253 Assay-C fixture chemistry only.
//
// This is a THIRD sibling fixture-only header, not a modification of either
// landed engine: bcc32_resident_recurrent_carrier.cuh (Assay B) tests finite
// recurrent renewal; bcc32_resident_competitive_medium.cuh (Assay A) tests
// competitive depletion. Neither alone gives a topology-independent bounded
// "supported" propagation regime -- Assay B's branching has no decay (any
// topology where a branch point is revisited by more than one independently
// branching carrier multiplies population without bound over cycles), and
// Assay A's engine requires exactly one route per (owner, from) with no
// branching at all. This header composes both mechanisms.
//
// Route:
//   lane[0] form
//   lane[1] opaque owner
//   lane[2] opaque source locator
//   lane[3] opaque target locator
//   lane[4] opaque shared-medium locator
//   lane[5] matter draw per traversal, q8
//   lane[6] transport delay
//   lane[7] route strength_q8 (renewal floor, from Assay B's engine)
//
// Medium: identical physical representation to Assay A.
//   lane[0] form
//   lane[1] opaque medium locator
//   lane[2] available_q8
//   lane[3] refractory_q8
//   lane[4] recovery_q8 per epoch
//   lane[5] invariant capacity_q8
//   lane[6..7] zero
//
// Carrier:
//   lane[0] form
//   lane[1] opaque owner
//   lane[2] current opaque locator
//   lane[3] local generation age
//   lane[4] remaining delay
//   lane[5] transient energy_q8
//   lane[6] originating resident revision / lineage
//   lane[7] zero
//
// Physical invariant for every live medium:
//
//   available_q8 + refractory_q8 == capacity_q8
//
// The critical correction over a naive Assay-A-style "block on insufficient
// medium" composition: if a carrier whose every candidate branch is blocked
// simply stayed resident (an Assay-A-style outcome), depletion would only
// bound successful-traversal FLUX, not resident carrier POPULATION -- a
// positive-net branching event (k>=2 successors) can still occur at any
// positive asymptotic rate as long as the medium eventually recovers enough
// to fund it again, so the population sum could still diverge (see the
// analysis in docs/diary for the exact argument). The fix: a resting
// carrier's outgoing routes are evaluated exactly once per epoch, and that
// local generation ends after that evaluation -- including when every
// candidate route is blocked. A carrier that cannot fund any branch this
// epoch extinguishes rather than persisting to retry later. This turns each
// medium into a genuine local dissipative sink rather than a pure gate, and
// yields a topology-independent bound on instantaneous carrier occupancy:
// for each medium m with capacity C_m and minimum route draw d_m > 0 among
// routes referencing it, at most floor(C_m / d_m) successors can be funded
// through m in one epoch (since recovery never raises available matter above
// C_m). Summing over all media gives Q = sum_m floor(C_m / d_m); with the
// largest finite transport delay D in the fixture, instantaneous population
// after the initial transient satisfies P(t) <= (D+1) * Q, independent of
// cyclic topology or branching factor. This is the exact claim the Assay-C
// contract measures against, not an arbitrary population threshold.
//
// depletion_enabled exists only as the Assay-C mechanism knockout: it proves
// the measured boundedness depends on the depletion chemistry, not on
// fixture topology alone.

inline constexpr std::uint32_t kFormPropagationRoute = 0x5bd72193u;
inline constexpr std::uint32_t kFormPropagationMedium = 0xa62ce841u;
inline constexpr std::uint32_t kFormPropagationCarrier = 0xd1743b65u;

inline constexpr std::uint32_t kScratchCapacity = 64u;

struct AdvanceReceipt {
  std::uint32_t carriers_before = 0u;
  std::uint32_t carriers_after = 0u;

  std::uint32_t attempted = 0u;
  std::uint32_t transferred = 0u;
  std::uint32_t branched = 0u;
  std::uint32_t blocked = 0u;

  std::uint32_t renewed = 0u;
  std::uint32_t delay_advanced = 0u;
  std::uint32_t extinguished = 0u;

  std::uint32_t depleted_q8 = 0u;
  std::uint32_t recovered_q8 = 0u;

  std::uint32_t allocation_failed = 0u;
  std::uint32_t malformed = 0u;
};

BCC32_PROPAGATION_REGIME_HD bool valid_key(std::uint32_t value) {
  return value != 0u && value != rewrite::kInvalid;
}

BCC32_PROPAGATION_REGIME_HD bool is_route(const rewrite::Record& record) {
  return record.matter_q8 != 0u &&
         record.lane[0] == kFormPropagationRoute &&
         valid_key(record.lane[1]) &&
         valid_key(record.lane[2]) &&
         valid_key(record.lane[3]) &&
         valid_key(record.lane[4]) &&
         record.lane[5] != 0u &&
         record.lane[7] != 0u &&
         record.reserved[0] == 0u &&
         record.reserved[1] == 0u;
}

BCC32_PROPAGATION_REGIME_HD bool is_medium(const rewrite::Record& record) {
  if (record.matter_q8 == 0u ||
      record.lane[0] != kFormPropagationMedium ||
      !valid_key(record.lane[1]) ||
      record.lane[4] == 0u ||
      record.lane[5] == 0u ||
      record.lane[4] > record.lane[5] ||
      record.lane[6] != 0u ||
      record.lane[7] != 0u ||
      record.reserved[0] != 0u ||
      record.reserved[1] != 0u) {
    return false;
  }

  const std::uint64_t represented =
      static_cast<std::uint64_t>(record.lane[2]) +
      static_cast<std::uint64_t>(record.lane[3]);
  return represented == static_cast<std::uint64_t>(record.lane[5]);
}

BCC32_PROPAGATION_REGIME_HD bool is_carrier(const rewrite::Record& record) {
  return record.matter_q8 != 0u &&
         record.lane[0] == kFormPropagationCarrier &&
         valid_key(record.lane[1]) &&
         valid_key(record.lane[2]) &&
         record.lane[5] != 0u &&
         record.lane[7] == 0u &&
         record.reserved[0] == 0u &&
         record.reserved[1] == 0u;
}

BCC32_PROPAGATION_REGIME_HD std::uint32_t renew_energy_q8(
    std::uint32_t incoming_q8, std::uint32_t route_strength_q8) {
  if (incoming_q8 == 0u || route_strength_q8 == 0u) return 0u;
  const std::uint32_t residual = incoming_q8 - 1u;
  return residual > route_strength_q8 ? residual : route_strength_q8;
}

BCC32_PROPAGATION_REGIME_NOINLINE std::uint32_t find_unique_medium(
    const rewrite::ResidentRewriteState& state, std::uint32_t medium_key,
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

BCC32_PROPAGATION_REGIME_NOINLINE void recover_media(
    rewrite::ResidentRewriteState* state, AdvanceReceipt* receipt) {
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

    if (!is_medium(medium)) ++receipt->malformed;
  }
}

BCC32_PROPAGATION_REGIME_HD bool seed_medium_for_contract(
    rewrite::ResidentRewriteState* state, std::uint32_t medium_key,
    std::uint32_t capacity_q8, std::uint32_t recovery_q8) {
  if (state == nullptr || !valid_key(medium_key) || capacity_q8 == 0u ||
      recovery_q8 == 0u || recovery_q8 > capacity_q8)
    return false;

  const std::uint32_t slot = rewrite::allocate_record(state);
  if (slot == rewrite::kInvalid) return false;

  rewrite::Record& record = state->records[slot];
  const std::uint32_t revision = record.revision + 1u;
  record = rewrite::Record{};
  record.lane[0] = kFormPropagationMedium;
  record.lane[1] = medium_key;
  record.lane[2] = capacity_q8;
  record.lane[3] = 0u;
  record.lane[4] = recovery_q8;
  record.lane[5] = capacity_q8;
  record.revision = revision;
  return true;
}

BCC32_PROPAGATION_REGIME_HD bool seed_route_for_contract(
    rewrite::ResidentRewriteState* state, std::uint32_t owner,
    std::uint32_t from, std::uint32_t to, std::uint32_t medium_key,
    std::uint32_t draw_q8, std::uint32_t delay, std::uint32_t strength_q8) {
  if (state == nullptr || !valid_key(owner) || !valid_key(from) ||
      !valid_key(to) || !valid_key(medium_key) || draw_q8 == 0u ||
      strength_q8 == 0u)
    return false;

  const std::uint32_t slot = rewrite::allocate_record(state);
  if (slot == rewrite::kInvalid) return false;

  rewrite::Record& record = state->records[slot];
  const std::uint32_t revision = record.revision + 1u;
  record = rewrite::Record{};
  record.lane[0] = kFormPropagationRoute;
  record.lane[1] = owner;
  record.lane[2] = from;
  record.lane[3] = to;
  record.lane[4] = medium_key;
  record.lane[5] = draw_q8;
  record.lane[6] = delay;
  record.lane[7] = strength_q8;
  record.revision = revision;
  return true;
}

BCC32_PROPAGATION_REGIME_HD bool seed_carrier_for_contract(
    rewrite::ResidentRewriteState* state, std::uint32_t owner,
    std::uint32_t state_key, std::uint32_t energy_q8) {
  if (state == nullptr || !valid_key(owner) || !valid_key(state_key) ||
      energy_q8 == 0u)
    return false;

  const std::uint32_t slot = rewrite::allocate_record(state);
  if (slot == rewrite::kInvalid) return false;

  rewrite::Record& record = state->records[slot];
  const std::uint32_t revision = record.revision + 1u;
  record = rewrite::Record{};
  record.lane[0] = kFormPropagationCarrier;
  record.lane[1] = owner;
  record.lane[2] = state_key;
  record.lane[3] = 0u;
  record.lane[4] = 0u;
  record.lane[5] = energy_q8;
  record.lane[6] = static_cast<std::uint32_t>(state->revision);
  record.revision = revision;
  return true;
}

// `depletion_enabled` is the Assay-C mechanism knockout: with it false, every
// candidate branch renews unconditionally (subject only to route strength),
// reproducing Assay B's unbounded-branching phenotype on the same topology.
BCC32_PROPAGATION_REGIME_NOINLINE AdvanceReceipt advance(
    rewrite::ResidentRewriteState* state, bool depletion_enabled) {
  AdvanceReceipt receipt{};
  if (state == nullptr) return receipt;

  recover_media(state, &receipt);

  const std::uint32_t capacity = rewrite::live_record_capacity(state);

  // Opening carrier snapshot. Newly generated carriers cannot advance again
  // during this epoch.
  std::uint32_t carrier_slot[kScratchCapacity]{};
  std::uint32_t carrier_owner[kScratchCapacity]{};
  std::uint32_t carrier_key[kScratchCapacity]{};
  std::uint32_t carrier_energy[kScratchCapacity]{};
  std::uint32_t carrier_origin[kScratchCapacity]{};
  std::uint32_t carrier_revision[kScratchCapacity]{};
  std::uint32_t carrier_count = 0u;

  for (std::uint32_t slot = 0u; slot < capacity; ++slot) {
    const rewrite::Record& carrier = state->records[slot];
    if (!is_carrier(carrier)) continue;
    ++receipt.carriers_before;
    if (carrier_count == kScratchCapacity) {
      ++receipt.malformed;
      continue;
    }
    carrier_slot[carrier_count] = slot;
    carrier_owner[carrier_count] = carrier.lane[1];
    carrier_key[carrier_count] = carrier.lane[2];
    carrier_energy[carrier_count] = carrier.lane[5];
    carrier_origin[carrier_count] = carrier.lane[6];
    carrier_revision[carrier_count] = carrier.revision;
    ++carrier_count;
  }

  // Stable physical/logical tuple ordering. Slot identity never
  // participates. Exact duplicate carriers are physically interchangeable,
  // so their tie does not change aggregate chemistry.
  for (std::uint32_t i = 1u; i < carrier_count; ++i) {
    const std::uint32_t slot = carrier_slot[i];
    const std::uint32_t owner = carrier_owner[i];
    const std::uint32_t key = carrier_key[i];
    const std::uint32_t energy = carrier_energy[i];
    const std::uint32_t origin = carrier_origin[i];
    const std::uint32_t revision = carrier_revision[i];
    std::uint32_t j = i;
    while (j > 0u) {
      const std::uint32_t p = j - 1u;
      const bool greater =
          carrier_owner[p] > owner ||
          (carrier_owner[p] == owner && carrier_key[p] > key) ||
          (carrier_owner[p] == owner && carrier_key[p] == key &&
           carrier_energy[p] > energy) ||
          (carrier_owner[p] == owner && carrier_key[p] == key &&
           carrier_energy[p] == energy && carrier_origin[p] > origin) ||
          (carrier_owner[p] == owner && carrier_key[p] == key &&
           carrier_energy[p] == energy && carrier_origin[p] == origin &&
           carrier_revision[p] > revision);
      if (!greater) break;
      carrier_slot[j] = carrier_slot[p];
      carrier_owner[j] = carrier_owner[p];
      carrier_key[j] = carrier_key[p];
      carrier_energy[j] = carrier_energy[p];
      carrier_origin[j] = carrier_origin[p];
      carrier_revision[j] = carrier_revision[p];
      --j;
    }
    carrier_slot[j] = slot;
    carrier_owner[j] = owner;
    carrier_key[j] = key;
    carrier_energy[j] = energy;
    carrier_origin[j] = origin;
    carrier_revision[j] = revision;
  }

  for (std::uint32_t i = 0u; i < carrier_count; ++i) {
    const std::uint32_t parent_slot = carrier_slot[i];
    rewrite::Record& parent = state->records[parent_slot];
    if (!is_carrier(parent) || parent.revision != carrier_revision[i]) {
      ++receipt.malformed;
      continue;
    }

    if (parent.lane[4] != 0u) {
      --parent.lane[4];
      ++parent.lane[3];
      ++parent.revision;
      ++receipt.delay_advanced;
      continue;
    }

    const std::uint32_t owner = parent.lane[1];
    const std::uint32_t current_key = parent.lane[2];
    const std::uint32_t incoming_energy = parent.lane[5];
    const std::uint32_t origin_revision = parent.lane[6];
    const std::uint32_t parent_revision = parent.revision;

    std::uint32_t route_slot[kScratchCapacity]{};
    std::uint32_t route_target[kScratchCapacity]{};
    std::uint32_t route_medium[kScratchCapacity]{};
    std::uint32_t route_draw[kScratchCapacity]{};
    std::uint32_t route_delay[kScratchCapacity]{};
    std::uint32_t route_strength[kScratchCapacity]{};
    std::uint32_t route_revision[kScratchCapacity]{};
    std::uint32_t route_count = 0u;
    bool route_overflow = false;

    for (std::uint32_t rslot = 0u; rslot < capacity; ++rslot) {
      const rewrite::Record& route = state->records[rslot];
      if (!is_route(route) || route.lane[1] != owner ||
          route.lane[2] != current_key)
        continue;
      if (route_count == kScratchCapacity) {
        route_overflow = true;
        break;
      }
      route_slot[route_count] = rslot;
      route_target[route_count] = route.lane[3];
      route_medium[route_count] = route.lane[4];
      route_draw[route_count] = route.lane[5];
      route_delay[route_count] = route.lane[6];
      route_strength[route_count] = route.lane[7];
      route_revision[route_count] = route.revision;
      ++route_count;
    }

    if (route_overflow) {
      rewrite::clear_record(&parent);
      ++receipt.malformed;
      ++receipt.extinguished;
      continue;
    }

    // Route order is independent of allocation slot:
    // (target, medium, draw, delay, strength, revision).
    for (std::uint32_t a = 1u; a < route_count; ++a) {
      const std::uint32_t slot = route_slot[a];
      const std::uint32_t target = route_target[a];
      const std::uint32_t medium = route_medium[a];
      const std::uint32_t draw = route_draw[a];
      const std::uint32_t delay = route_delay[a];
      const std::uint32_t strength = route_strength[a];
      const std::uint32_t revision = route_revision[a];
      std::uint32_t b = a;
      while (b > 0u) {
        const std::uint32_t p = b - 1u;
        const bool greater =
            route_target[p] > target ||
            (route_target[p] == target && route_medium[p] > medium) ||
            (route_target[p] == target && route_medium[p] == medium &&
             route_draw[p] > draw) ||
            (route_target[p] == target && route_medium[p] == medium &&
             route_draw[p] == draw && route_delay[p] > delay) ||
            (route_target[p] == target && route_medium[p] == medium &&
             route_draw[p] == draw && route_delay[p] == delay &&
             route_strength[p] > strength) ||
            (route_target[p] == target && route_medium[p] == medium &&
             route_draw[p] == draw && route_delay[p] == delay &&
             route_strength[p] == strength && route_revision[p] > revision);
        if (!greater) break;
        route_slot[b] = route_slot[p];
        route_target[b] = route_target[p];
        route_medium[b] = route_medium[p];
        route_draw[b] = route_draw[p];
        route_delay[b] = route_delay[p];
        route_strength[b] = route_strength[p];
        route_revision[b] = route_revision[p];
        --b;
      }
      route_slot[b] = slot;
      route_target[b] = target;
      route_medium[b] = medium;
      route_draw[b] = draw;
      route_delay[b] = delay;
      route_strength[b] = strength;
      route_revision[b] = revision;
    }

    if (route_count == 0u) {
      rewrite::clear_record(&parent);
      ++receipt.extinguished;
      continue;
    }

    bool produced_any = false;

    for (std::uint32_t r = 0u; r < route_count; ++r) {
      const rewrite::Record& route = state->records[route_slot[r]];
      if (!is_route(route) || route.revision != route_revision[r] ||
          route.lane[1] != owner || route.lane[2] != current_key) {
        ++receipt.malformed;
        continue;
      }
      ++receipt.attempted;

      std::uint32_t medium_slot = rewrite::kInvalid;
      if (depletion_enabled) {
        bool ambiguous = false;
        medium_slot =
            find_unique_medium(*state, route_medium[r], &ambiguous);
        if (ambiguous || medium_slot == rewrite::kInvalid) {
          ++receipt.malformed;
          continue;
        }
        const rewrite::Record& medium = state->records[medium_slot];
        if (!is_medium(medium)) {
          ++receipt.malformed;
          continue;
        }
        if (medium.lane[2] < route_draw[r]) {
          ++receipt.blocked;
          continue;
        }
      }

      const std::uint32_t successor_energy =
          renew_energy_q8(incoming_energy, route_strength[r]);
      if (successor_energy == 0u) {
        ++receipt.blocked;
        continue;
      }

      std::uint32_t successor_slot = parent_slot;
      std::uint32_t successor_revision = parent_revision + 1u;
      if (produced_any) {
        successor_slot = rewrite::allocate_record(state);
        if (successor_slot == rewrite::kInvalid) {
          ++receipt.allocation_failed;
          continue;
        }
        successor_revision = state->records[successor_slot].revision + 1u;
      }

      // Allocation happens before physical depletion for sibling branches,
      // so allocator exhaustion cannot consume medium without producing the
      // corresponding successor. The medium is revalidated here (rather than
      // trusting the earlier check) because an EARLIER route in this same
      // loop may reference the same medium and have already spent it.
      if (depletion_enabled) {
        rewrite::Record& medium = state->records[medium_slot];
        if (!is_medium(medium) || medium.lane[2] < route_draw[r]) {
          if (produced_any)
            rewrite::clear_record(&state->records[successor_slot]);
          ++receipt.malformed;
          continue;
        }
        medium.lane[2] -= route_draw[r];
        medium.lane[3] += route_draw[r];
        ++medium.revision;
        receipt.depleted_q8 += route_draw[r];
        if (!is_medium(medium)) {
          if (produced_any)
            rewrite::clear_record(&state->records[successor_slot]);
          ++receipt.malformed;
          continue;
        }
      }

      rewrite::Record& successor = state->records[successor_slot];
      successor = rewrite::Record{};
      successor.lane[0] = kFormPropagationCarrier;
      successor.lane[1] = owner;
      successor.lane[2] = route_target[r];
      successor.lane[3] = 0u;
      successor.lane[4] = route_delay[r];
      successor.lane[5] = successor_energy;
      successor.lane[6] = origin_revision;
      successor.lane[7] = 0u;
      successor.revision = successor_revision;

      ++receipt.transferred;
      ++receipt.renewed;
      if (produced_any) ++receipt.branched;
      produced_any = true;
    }

    // Critical Assay-C rule: the opening generation has ended after its
    // resting interaction even when every candidate edge was blocked.
    // Without this local sink, finite medium matter bounds only traversal
    // flux, while blocked carrier generations could accumulate without
    // bound (see the header-level comment for the full argument).
    if (!produced_any) {
      rewrite::clear_record(&state->records[parent_slot]);
      ++receipt.extinguished;
    }
  }

  for (std::uint32_t slot = 0u; slot < capacity; ++slot)
    if (is_carrier(state->records[slot])) ++receipt.carriers_after;

  return receipt;
}

}  // namespace substrate::bcc32::resident_propagation_regime

#undef BCC32_PROPAGATION_REGIME_HD
#undef BCC32_PROPAGATION_REGIME_NOINLINE
