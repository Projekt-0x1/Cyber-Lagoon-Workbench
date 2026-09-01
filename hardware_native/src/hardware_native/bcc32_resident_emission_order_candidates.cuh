#pragma once

#include <cuda_runtime.h>

#include <cstdint>

#include "hardware_native/bcc32_types.cuh"

// EMISSION-ORDER SEARCH HARNESS -- substrate contract, NOT an M7/attachment_
// correct rung claim, and NOT a change to select_trajectory,
// arm_selected_trajectory, trajectory_adjacency_index, or kRenderOrder.
//
// bcc32_resident_factor_renderer.cuh's kRenderOrder is a hand-authored
// constexpr array (actor->action->modifier->object). THE GOVERNING CRITERION
// asks: could a LOCAL rule over the carriers' own resident fields reproduce
// "which carrier emits next" without hand-authoring the sequence and without
// any constant equal to the role count? This file builds five candidate
// local rules and lets bcc32_cuda_emission_order_search_contract.cu measure
// each one against a world whose role COUNT (3, 4, 5) and role
// PRESENTATION ORDER (canonical vs. permuted) both vary at runtime, same
// binary, no recompilation between worlds.
//
// SCOPE DISCLOSURE (same shape as bcc32_resident_recruited_carriers.cuh's):
// this is a standalone carrier-pool primitive, not wired into GrownAdult's
// founder aperture or InstanceBasin::observe(). It reuses the
// match-else-recruit / two-rail-XOR discipline that file already
// established, applied here to a pool that additionally records WHEN
// (as a runtime tick counter, never a literal) each carrier was first
// touched. "Identity" is a caller-supplied SiteWord token, exactly as in
// that file.
//
// DISQUALIFICATION CHECK: none of the five candidates below read
// trajectory_phase_index, select_trajectory's stored assembly, or any other
// stored-trajectory phase order -- each reads ONLY fields written onto the
// carrier itself by present_role_device below (identity, recruit_tick,
// last_touch_tick). See the contract file for which candidates pass, which
// fail, and why.
namespace substrate::bcc32::resident_emission_order_candidates {

using substrate::bcc32::SiteWord;

// Matter-capacity bound on the carrier POOL, not a role-count constant: no
// world exercised by the contract presents this many roles (worlds use 3,
// 4, or 5), so this ceiling is never reached and never equals a role count.
inline constexpr std::uint32_t kCarrierCapacity = 8u;

enum CarrierField : std::uint32_t {
  kIdentity = 0u,
  kOccupied = 1u,
  kRecruitTick = 2u,
  kLastTouchTick = 3u,
  kCarrierFieldCount = 4u,
};

enum TouchAction : std::uint32_t {
  kActionNone = 0u,
  kActionReused = 1u,
  kActionRecruited = 2u,
  kActionCapacityAbstained = 3u,
};

inline constexpr std::uint32_t kRailCount =
    kCarrierCapacity * kCarrierFieldCount * 2u;

__host__ __device__ inline std::uint32_t carrier_field_rail(
    std::uint32_t carrier, std::uint32_t field) {
  return (carrier * kCarrierFieldCount + field) * 2u;
}

__device__ inline SiteWord read_field(const SiteWord* words,
                                      std::uint32_t carrier,
                                      std::uint32_t field) {
  return words[carrier_field_rail(carrier, field)];
}

// Self-inverse: XOR-ing the identical delta twice restores both rails
// exactly (bcc32_resident_edge_bank.cuh's convention, reused verbatim by
// bcc32_resident_recruited_carriers.cuh).
__device__ inline void xor_field(SiteWord* words, std::uint32_t carrier,
                                 std::uint32_t field, SiteWord delta) {
  const std::uint32_t rail = carrier_field_rail(carrier, field);
  words[rail] ^= delta;
  words[rail + 1u] ^= delta;
}

struct TouchOutcome {
  std::uint32_t action = kActionNone;
  std::uint32_t carrier_index = 0xffffffffu;
};

// THE recruitment/touch rule -- identical in shape to
// bcc32_resident_recruited_carriers.cuh's recruit_or_reuse_device: duplicate
// scan first (reuse), then first-free scan (recruit), then abstain. `tick`
// is a runtime counter supplied by the caller (the harness increments it
// once per world-presentation event) -- never a literal, never a role
// count.
__device__ inline TouchOutcome touch_or_reuse_device(SiteWord* words,
                                                     SiteWord identity,
                                                     SiteWord tick) {
  TouchOutcome outcome{};
  for (std::uint32_t carrier = 0u; carrier < kCarrierCapacity; ++carrier) {
    if (read_field(words, carrier, kOccupied) != 0u &&
        read_field(words, carrier, kIdentity) == identity) {
      // Reuse: contact recency updates, recruitment tick does not.
      const SiteWord old_touch = read_field(words, carrier, kLastTouchTick);
      xor_field(words, carrier, kLastTouchTick, old_touch ^ tick);
      outcome.action = kActionReused;
      outcome.carrier_index = carrier;
      return outcome;
    }
  }
  for (std::uint32_t carrier = 0u; carrier < kCarrierCapacity; ++carrier) {
    if (read_field(words, carrier, kOccupied) == 0u) {
      xor_field(words, carrier, kIdentity, identity);
      xor_field(words, carrier, kOccupied, 1u);
      xor_field(words, carrier, kRecruitTick, tick);
      xor_field(words, carrier, kLastTouchTick, tick);
      outcome.action = kActionRecruited;
      outcome.carrier_index = carrier;
      return outcome;
    }
  }
  outcome.action = kActionCapacityAbstained;
  return outcome;
}

__device__ inline void undo_touch_device(SiteWord* words,
                                         TouchOutcome outcome,
                                         SiteWord identity, SiteWord tick,
                                         SiteWord touch_tick_before_reuse) {
  if (outcome.action == kActionReused) {
    const SiteWord current = read_field(words, outcome.carrier_index,
                                        kLastTouchTick);
    xor_field(words, outcome.carrier_index, kLastTouchTick,
             current ^ touch_tick_before_reuse);
    return;
  }
  if (outcome.action != kActionRecruited) return;
  xor_field(words, outcome.carrier_index, kLastTouchTick, tick);
  xor_field(words, outcome.carrier_index, kRecruitTick, tick);
  xor_field(words, outcome.carrier_index, kOccupied, 1u);
  xor_field(words, outcome.carrier_index, kIdentity, identity);
}

// One emitted order: which carriers, in which sequence, for however many
// carriers happen to be occupied (never a compiled-in count).
struct EmittedOrder {
  std::uint32_t carrier_index[kCarrierCapacity]{};
  std::uint32_t count = 0u;
};

// ---------------------------------------------------------------------------
// CANDIDATE 1 -- recruitment index. Emit carriers in ascending order of
// kRecruitTick (the tick at which each carrier was FIRST recruited). No
// role name, no role-count constant: the loop bound is kCarrierCapacity (a
// matter bound, identical for every world), and the comparison reads only
// the carrier's own kRecruitTick rail.
// ---------------------------------------------------------------------------
__device__ inline EmittedOrder rank_by_recruit_index_device(
    const SiteWord* words) {
  EmittedOrder order{};
  bool used[kCarrierCapacity] = {};
  for (std::uint32_t slot = 0u; slot < kCarrierCapacity; ++slot) {
    std::uint32_t best = 0xffffffffu;
    SiteWord best_tick = 0u;
    for (std::uint32_t carrier = 0u; carrier < kCarrierCapacity; ++carrier) {
      if (used[carrier] || read_field(words, carrier, kOccupied) == 0u)
        continue;
      const SiteWord tick = read_field(words, carrier, kRecruitTick);
      if (best == 0xffffffffu || tick < best_tick) {
        best = carrier;
        best_tick = tick;
      }
    }
    if (best == 0xffffffffu) break;
    used[best] = true;
    order.carrier_index[order.count++] = best;
  }
  return order;
}

// ---------------------------------------------------------------------------
// CANDIDATE 2 -- resident age. "Age" is derived from the carrier's own
// occupancy field (kRecruitTick) relative to `now` (a runtime scalar the
// caller measures as the current tick count -- never a literal): age =
// now - kRecruitTick. Emits OLDEST first, i.e. descending age. This reads a
// different quantity than candidate 1 (a subtraction against `now`, not a
// raw comparison of the stored field), even though in a world with no
// carrier recycling the resulting ORDER coincides with candidate 1's --
// disclosed explicitly in the contract's report, not hidden.
// ---------------------------------------------------------------------------
__device__ inline EmittedOrder rank_by_resident_age_device(
    const SiteWord* words, SiteWord now_tick) {
  EmittedOrder order{};
  bool used[kCarrierCapacity] = {};
  for (std::uint32_t slot = 0u; slot < kCarrierCapacity; ++slot) {
    std::uint32_t best = 0xffffffffu;
    SiteWord best_age = 0u;
    for (std::uint32_t carrier = 0u; carrier < kCarrierCapacity; ++carrier) {
      if (used[carrier] || read_field(words, carrier, kOccupied) == 0u)
        continue;
      const SiteWord age = now_tick - read_field(words, carrier, kRecruitTick);
      if (best == 0xffffffffu || age > best_age) {
        best = carrier;
        best_age = age;
      }
    }
    if (best == 0xffffffffu) break;
    used[best] = true;
    order.carrier_index[order.count++] = best;
  }
  return order;
}

// ---------------------------------------------------------------------------
// CANDIDATE 3 -- contact recency. Emits the carrier MOST RECENTLY touched by
// the observation gate FIRST (descending kLastTouchTick), exactly as the
// brief's suggested wording states it ("emit by which carrier was most
// recently touched"). Reads only kLastTouchTick.
// ---------------------------------------------------------------------------
__device__ inline EmittedOrder rank_by_contact_recency_device(
    const SiteWord* words) {
  EmittedOrder order{};
  bool used[kCarrierCapacity] = {};
  for (std::uint32_t slot = 0u; slot < kCarrierCapacity; ++slot) {
    std::uint32_t best = 0xffffffffu;
    SiteWord best_touch = 0u;
    for (std::uint32_t carrier = 0u; carrier < kCarrierCapacity; ++carrier) {
      if (used[carrier] || read_field(words, carrier, kOccupied) == 0u)
        continue;
      const SiteWord touch = read_field(words, carrier, kLastTouchTick);
      if (best == 0xffffffffu || touch > best_touch) {
        best = carrier;
        best_touch = touch;
      }
    }
    if (best == 0xffffffffu) break;
    used[best] = true;
    order.carrier_index[order.count++] = best;
  }
  return order;
}

// FALSIFIER SWITCH (required by the brief): flip to true, rebuild, rerun --
// this replaces candidate 1's comparison key (the carrier's own
// kRecruitTick field) with a FIXED constant table indexed by the carrier's
// identity value's low nibble, reproducing a hand-authored canonical order
// regardless of actual arrival tick. The permuted-order world (world D in
// the contract) must go RED under this switch: the falsified candidate
// keeps emitting the canonical actor/action/modifier/object order even
// though world D presented them object/modifier/action/actor. Revert to
// false to restore the PASS baseline byte-for-byte (this is the only
// difference between the two builds).
inline constexpr bool kFalsifierOrderByFixedConstant = false;

// Fixed constant table used ONLY by the falsifier above -- keyed by the
// low nibble of the five canonical identity tokens the contract defines
// (0x1..0x5), giving the fixed canonical rank of each. This exists purely
// to demonstrate the failure the falsifier is supposed to cause; it is
// never consulted unless kFalsifierOrderByFixedConstant is true.
__device__ inline std::uint32_t falsifier_fixed_rank(SiteWord identity) {
  return identity & 0xfu;
}

// ---------------------------------------------------------------------------
// CANDIDATE 1's device entry point, with the falsifier switch applied. This
// is the ONLY candidate the falsifier corrupts (per the brief: "take
// whichever candidate passes... and corrupt its comparison").
// ---------------------------------------------------------------------------
__device__ inline EmittedOrder rank_by_recruit_index_falsifiable_device(
    const SiteWord* words) {
  if (!kFalsifierOrderByFixedConstant) return rank_by_recruit_index_device(words);
  EmittedOrder order{};
  bool used[kCarrierCapacity] = {};
  for (std::uint32_t slot = 0u; slot < kCarrierCapacity; ++slot) {
    std::uint32_t best = 0xffffffffu;
    std::uint32_t best_rank = 0u;
    for (std::uint32_t carrier = 0u; carrier < kCarrierCapacity; ++carrier) {
      if (used[carrier] || read_field(words, carrier, kOccupied) == 0u)
        continue;
      const std::uint32_t rank =
          falsifier_fixed_rank(read_field(words, carrier, kIdentity));
      if (best == 0xffffffffu || rank < best_rank) {
        best = carrier;
        best_rank = rank;
      }
    }
    if (best == 0xffffffffu) break;
    used[best] = true;
    order.carrier_index[order.count++] = best;
  }
  return order;
}

// ---------------------------------------------------------------------------
// CANDIDATE 4 -- identity magnitude. Emits carriers in ascending order of
// their own kIdentity token value. Reads only kIdentity -- never touches
// recruit tick or touch tick.
// ---------------------------------------------------------------------------
__device__ inline EmittedOrder rank_by_identity_magnitude_device(
    const SiteWord* words) {
  EmittedOrder order{};
  bool used[kCarrierCapacity] = {};
  for (std::uint32_t slot = 0u; slot < kCarrierCapacity; ++slot) {
    std::uint32_t best = 0xffffffffu;
    SiteWord best_identity = 0u;
    for (std::uint32_t carrier = 0u; carrier < kCarrierCapacity; ++carrier) {
      if (used[carrier] || read_field(words, carrier, kOccupied) == 0u)
        continue;
      const SiteWord identity = read_field(words, carrier, kIdentity);
      if (best == 0xffffffffu || identity < best_identity) {
        best = carrier;
        best_identity = identity;
      }
    }
    if (best == 0xffffffffu) break;
    used[best] = true;
    order.carrier_index[order.count++] = best;
  }
  return order;
}

// ---------------------------------------------------------------------------
// CANDIDATE 5 -- pairwise resident comparison. A local "who goes before
// whom" comparator (A before B iff A's own identity < B's own identity),
// applied TRANSITIVELY across every pair of occupied carriers via an
// explicit pairwise compare-and-count-inversions network (an
// insertion-rank, not a key-extraction sort): each occupied carrier's final
// rank is the number of OTHER occupied carriers whose identity compares
// before it. This is mechanically distinct from candidate 4 (no single
// sort pass over an extracted key; the ordering falls out of n*(n-1)/2
// pairwise comparisons run to completion) even though it reads the same
// per-carrier field and is therefore predicted to fail under exactly the
// same condition candidate 4 fails under -- see the contract for whether
// that prediction holds.
// ---------------------------------------------------------------------------
__device__ inline EmittedOrder rank_by_pairwise_comparison_device(
    const SiteWord* words) {
  EmittedOrder order{};
  std::uint32_t occupied_carriers[kCarrierCapacity]{};
  std::uint32_t occupied_count = 0u;
  for (std::uint32_t carrier = 0u; carrier < kCarrierCapacity; ++carrier) {
    if (read_field(words, carrier, kOccupied) != 0u)
      occupied_carriers[occupied_count++] = carrier;
  }
  std::uint32_t rank[kCarrierCapacity]{};
  for (std::uint32_t i = 0u; i < occupied_count; ++i) {
    std::uint32_t before_count = 0u;
    const SiteWord identity_i =
        read_field(words, occupied_carriers[i], kIdentity);
    for (std::uint32_t j = 0u; j < occupied_count; ++j) {
      if (i == j) continue;
      const SiteWord identity_j =
          read_field(words, occupied_carriers[j], kIdentity);
      // Pairwise "who goes before whom": strictly-less identity goes
      // before; a tie is broken by carrier index so the comparator is a
      // valid strict total order (never consults presentation order).
      const bool j_before_i = (identity_j < identity_i) ||
                              (identity_j == identity_i && j < i);
      before_count += j_before_i ? 1u : 0u;
    }
    rank[i] = before_count;
  }
  for (std::uint32_t i = 0u; i < occupied_count; ++i)
    order.carrier_index[rank[i]] = occupied_carriers[i];
  order.count = occupied_count;
  return order;
}

}  // namespace substrate::bcc32::resident_emission_order_candidates
