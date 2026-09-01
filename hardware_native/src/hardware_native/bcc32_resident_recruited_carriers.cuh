#pragma once

#include <cuda_runtime.h>

#include <cstdint>

#include "hardware_native/bcc32_types.cuh"

// RECRUITMENT: a single LOCAL rule -- match-else-recruit -- replacing the two
// hardcoded pieces named in the brief (bcc32_resident_factor_basins.cuh's
// fixed kFactorSlotCount=4, and bcc32_resident_factor_perception.cuh's fixed
// kPatchToSlot table). Neither of those two facts appears anywhere below:
//
//   for each active patch (in whatever order the caller presents them):
//       if some existing carrier already holds this patch's identity
//           -> that carrier is reused
//       else
//           -> recruit a free carrier
//
// The shape is the one this repo already uses for match-else-recruit:
// bcc32_resident_edge_bank.cuh's form_edge_device() (duplicate-scan-then-
// allocate, self-inverse XOR writes, atomic capacity-abstain) and
// bcc32_grown_cloud_factor.cuh's recruit_pair()/first_empty_neighbor()
// (scan for a free site, take it, abstain if none exists). This file reuses
// that exact discipline: duplicate -> reuse, free -> recruit, full ->
// abstain atomically -- applied to a flat pool of carriers instead of a
// cell/route topology.
//
// SCOPE DISCLOSURE (read before citing this as wired into the organism's
// perception path): this is a standalone carrier-pool primitive, matching
// the precedent of bcc32_cuda_paged_resident_route_bank_contract.cu's
// DeviceBank -- a directly host-allocated device buffer, not matter attached
// through GrownAdult::attach_founder_matter() into the shared aperture. It
// is not wired into bcc32_resident_factor_perception.cuh's real
// InstanceBasin::observe() occupied-mask gate, and it does not call
// find_or_recruit_form/decode_form_raw -- "identity" here is whatever
// SiteWord token the caller supplies per patch (in the production perception
// path that would be the decoded form byte, exactly as
// resident_factor_perception.cuh's perceive_scene_device() already produces
// per patch; re-deriving that channel here was out of the 75-minute
// time-box on top of building this mechanism and its contract from scratch).
// The claim this file supports is scoped to the RECRUITMENT TOPOLOGY itself
// (carrier count tracks presented-role count, with no compiled-in role
// count and no patch-index-to-slot table) -- not to a full sensor-to-slot
// production rollout.
//
// This file does not modify select_trajectory, arm_selected_trajectory,
// trajectory_adjacency_index, or kRenderOrder.
namespace substrate::bcc32::resident_recruited_carriers {

using substrate::bcc32::SiteWord;

// Capacity bound on the carrier POOL -- matter is finite, so some ceiling is
// required. This is NOT a role-count constant: it bounds how much matter
// exists, never how many roles a scene may have. A scene with fewer roles
// than kCarrierCapacity recruits exactly that many carriers and leaves the
// rest free; a scene with more roles than kCarrierCapacity is the capacity-
// abstain case exercised by check 5.
inline constexpr std::uint32_t kCarrierCapacity = 6u;

enum CarrierField : std::uint32_t {
  kIdentity = 0u,
  kOccupied = 1u,
  kCarrierFieldCount = 2u,
};

enum RecruitAction : std::uint32_t {
  kActionNone = 0u,
  kActionReused = 1u,
  kActionRecruited = 2u,
  kActionCapacityAbstained = 3u,
};

// Two rails per field (value + complement), matching the convention used by
// every other resident factor in this tree (bcc32_resident_edge_bank.cuh,
// bcc32_resident_factor_basins.cuh, bcc32_grown_instance_basin.cuh).
inline constexpr std::uint32_t kRailCount =
    kCarrierCapacity * kCarrierFieldCount * 2u;

__host__ __device__ inline std::uint32_t carrier_field_rail(
    std::uint32_t carrier, std::uint32_t field) {
  return (carrier * kCarrierFieldCount + field) * 2u;
}

__host__ __device__ inline SiteWord read_field(const SiteWord* words,
                                               std::uint32_t carrier,
                                               std::uint32_t field) {
  return words[carrier_field_rail(carrier, field)];
}

// Self-inverse: XOR-ing the identical delta twice restores both rails
// exactly (same reasoning as bcc32_resident_edge_bank.cuh's xor_field --
// XOR-ing the same delta into the complement rail keeps it the exact
// complement of the value rail automatically).
__host__ __device__ inline void xor_field(SiteWord* words,
                                          std::uint32_t carrier,
                                          std::uint32_t field, SiteWord delta) {
  const std::uint32_t rail = carrier_field_rail(carrier, field);
  words[rail] ^= delta;
  words[rail + 1u] ^= delta;
}

struct RecruitOutcome {
  std::uint32_t action = kActionNone;
  std::uint32_t carrier_index = 0xffffffffu;
};

// THE local rule. No constant equal to a role count appears here: the two
// loops both run 0..kCarrierCapacity (a matter-capacity bound, not a role
// count), and which carrier a patch lands in is decided purely by identity
// match (first loop) or first-free-index (second loop) -- never by patch
// position.
// Same local rule, with the pool bound supplied by the CALLER as a runtime
// argument instead of read from kCarrierCapacity.
//
// Why this exists: kCarrierCapacity is a compile-time constant, so a payload
// longer than it cannot be carried without editing this file. That is exactly
// the property START.md section 4 forbids for anything the organism must
// acquire -- "does it have to be re-authored when the task grows?" -- and it is
// what caps the same-adult grounded payload today. Passing the bound in makes
// the ceiling a property of HOW MUCH MATTER WAS SUPPLIED, which is a legitimate
// finite-matter bound, rather than of how much matter was declared in source.
//
// The rule itself is unchanged and still contains no role-count constant: both
// loops run 0..capacity, and which carrier a patch lands in is decided purely
// by identity match then first-free-index. kCarrierCapacity remains the default
// for every existing caller via the wrapper below, so no current behaviour
// moves.
__host__ __device__ inline RecruitOutcome recruit_or_reuse_in(
    SiteWord* words, std::uint32_t capacity, SiteWord identity) {
  RecruitOutcome outcome{};
  for (std::uint32_t carrier = 0u; carrier < capacity; ++carrier) {
    if (read_field(words, carrier, kOccupied) != 0u &&
        read_field(words, carrier, kIdentity) == identity) {
      outcome.action = kActionReused;
      outcome.carrier_index = carrier;
      return outcome;
    }
  }
  for (std::uint32_t carrier = 0u; carrier < capacity; ++carrier) {
    if (read_field(words, carrier, kOccupied) == 0u) {
      xor_field(words, carrier, kIdentity, identity);
      xor_field(words, carrier, kOccupied, 1u);
      outcome.action = kActionRecruited;
      outcome.carrier_index = carrier;
      return outcome;
    }
  }
  outcome.action = kActionCapacityAbstained;
  return outcome;
}

__host__ __device__ inline RecruitOutcome recruit_or_reuse_device(
    SiteWord* words, SiteWord identity) {
  RecruitOutcome outcome{};
  // Duplicate scan FIRST -- same discipline as form_edge_device(): a patch
  // whose identity already occupies a carrier must reuse it, never allocate
  // a second one.
  for (std::uint32_t carrier = 0u; carrier < kCarrierCapacity; ++carrier) {
    if (read_field(words, carrier, kOccupied) != 0u &&
        read_field(words, carrier, kIdentity) == identity) {
      outcome.action = kActionReused;
      outcome.carrier_index = carrier;
      return outcome;
    }
  }
  // Free scan -- first empty carrier, same discipline as
  // first_empty_neighbor()/recruit_pair(): take the first free site, no
  // preference for any particular index beyond "first free."
  for (std::uint32_t carrier = 0u; carrier < kCarrierCapacity; ++carrier) {
    if (read_field(words, carrier, kOccupied) == 0u) {
      xor_field(words, carrier, kIdentity, identity);
      xor_field(words, carrier, kOccupied, 1u);
      outcome.action = kActionRecruited;
      outcome.carrier_index = carrier;
      return outcome;
    }
  }
  // Every carrier occupied and none matches: abstain atomically. The
  // duplicate scan only reads and the free scan never ran a write before
  // failing to find a slot, so no rail above this point was touched.
  outcome.action = kActionCapacityAbstained;
  return outcome;
}

__host__ __device__ inline void undo_recruit_device(SiteWord* words,
                                                    RecruitOutcome outcome,
                                                    SiteWord identity) {
  // kActionReused and kActionCapacityAbstained touched no rail going
  // forward, so there is nothing to undo for either.
  if (outcome.action != kActionRecruited) return;
  xor_field(words, outcome.carrier_index, kOccupied, 1u);
  xor_field(words, outcome.carrier_index, kIdentity, identity);
}

// Output-buffer bound on how many patches ONE present_scene_device() call
// can report outcomes for. This is NOT a role-count constant either: it is
// sized generously above kCarrierCapacity so that every scene size this
// contract exercises (three, four, five roles) fits with headroom, and nothing
// in recruit_or_reuse_device or the loop below is aware of it -- only the
// caller-supplied `patch_count` runtime parameter governs how many patches
// present_scene_device actually processes.
struct PresentOutcome {
  static constexpr std::uint32_t kMaxPatchesPerCall = 16u;
  RecruitOutcome outcomes[kMaxPatchesPerCall]{};
  SiteWord identities[kMaxPatchesPerCall]{};
  std::uint32_t reported_count = 0u;
};

// FALSIFIER SWITCH (required by the brief): flip to true, rebuild, rerun
// bcc32_cuda_recruited_carriers_contract -- this clamps the loop below to
// process at most 4 patches regardless of how many the caller presented,
// reintroducing the exact fixed-role-count defect this file exists to
// remove. With this true, check 2 (the five-role scene) must go RED: only
// four carriers get recruited, not five. Revert to false to restore the
// PASS baseline byte-for-byte (this is the only line that differs between
// the two builds).
inline constexpr bool kFalsifierClampToFour = false;
inline constexpr std::uint32_t kFalsifierClampCount = 4u;

// present_scene_device(): applies recruit_or_reuse_device IDENTICALLY to
// every patch in the caller-supplied array, in array order (the order the
// caller's own occupied-mask-style enumeration presented them -- see
// bcc32_resident_factor_perception.cuh's perceive_scene_device() for the
// production analogue of that enumeration, not reproduced here per the
// scope disclosure above). `patch_count` is an ordinary runtime value, never
// a compiled-in constant -- a 3-, 4-, or 5-role scene is the identical
// binary taking a different value of this one parameter.
__device__ inline PresentOutcome present_scene_device(
    SiteWord* words, const SiteWord* patch_identities,
    std::uint32_t patch_count) {
  PresentOutcome result{};
  const std::uint32_t effective_count =
      kFalsifierClampToFour
          ? (patch_count < kFalsifierClampCount ? patch_count
                                                : kFalsifierClampCount)
          : patch_count;
  for (std::uint32_t patch = 0u;
       patch < effective_count && patch < PresentOutcome::kMaxPatchesPerCall;
       ++patch) {
    const SiteWord identity = patch_identities[patch];
    result.outcomes[patch] = recruit_or_reuse_device(words, identity);
    result.identities[patch] = identity;
    result.reported_count = patch + 1u;
  }
  return result;
}

// Exact LIFO inverse of present_scene_device(): undo in reverse patch order,
// mirroring the forward order exactly (same discipline as
// resident_factor_perception.cuh's undo_perceive_device()).
__device__ inline void undo_scene_device(SiteWord* words,
                                         const PresentOutcome& presented) {
  for (std::uint32_t i = 0u; i < presented.reported_count; ++i) {
    const std::uint32_t patch = presented.reported_count - 1u - i;
    undo_recruit_device(words, presented.outcomes[patch],
                        presented.identities[patch]);
  }
}

// Measured witness helper: counts carriers currently occupied. Always a
// scan over resident state, never a literal -- the brief requires every
// status field to be a measured variable.
__device__ inline std::uint32_t count_occupied_device(const SiteWord* words) {
  std::uint32_t count = 0u;
  for (std::uint32_t carrier = 0u; carrier < kCarrierCapacity; ++carrier) {
    count += read_field(words, carrier, kOccupied) != 0u ? 1u : 0u;
  }
  return count;
}

}  // namespace substrate::bcc32::resident_recruited_carriers
