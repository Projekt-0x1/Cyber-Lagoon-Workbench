#pragma once

// Must NOT include causal_rewrite_universe.cuh: that closed a cycle. gh #1215.

#include <cstdint>

#if defined(__CUDACC__)
#define BCC32_TURN_WORLD_BINDING_HD __host__ __device__
#else
#define BCC32_TURN_WORLD_BINDING_HD
#endif

// A resident "world cell" Record is written by a physically-grounded action
// and bound, through an explicit owner-token binding Record, to a claim
// Record. A later held-out query resolves the claim by first finding its
// unique binding, then re-reading the world cell's CURRENT value from the
// world cell Record at query time. Nothing along that path caches a value:
// the binding stores only the world cell's owner token, never a copy of its
// payload, so a physical write to the world cell between two calls changes
// what the next resolution returns. Severing only the binding Record breaks
// resolution while the world cell and claim remain physically intact; a full
// resident reset erases every Record and therefore also breaks resolution.
#if defined(BCC32_TURN_WORLD_BINDING_INSIDE_REWRITE_NAMESPACE)
namespace turn_world_consequence_binding {
#else
namespace substrate::bcc32::causal_rewrite::turn_world_consequence_binding {
#endif

using causal_rewrite::allocate_record;
using causal_rewrite::kInvalid;
using causal_rewrite::kRecordCapacity;
using causal_rewrite::live_record_capacity;
using causal_rewrite::Record;
using causal_rewrite::refresh_receipt;
using causal_rewrite::ResidentRewriteState;
using causal_rewrite::rewrite_mix;

inline constexpr std::uint32_t kFormWorldCell = 0x2f6b19d4u;
inline constexpr std::uint32_t kFormWorldCellClaim = 0x5c8e3a71u;
inline constexpr std::uint32_t kFormWorldCellBinding = 0x71d4f2a8u;

// World cell lanes: [1] owner; [2] current physically written value;
//   [3] monotone physical-write count (never a semantic score).
// Claim lanes: [1] owner; [2] optional subject token (see
//   find_claim_for_subject/ensure_subject_claim below), 0 when unused. A
//   claim carries no payload of its own; it exists only as an addressable
//   token a binding can be attached to.
// Binding lanes: [1] owner (the binding's own token); [2] claim owner;
//   [3] world cell owner. The binding is a live pointer pair, not a snapshot:
//   resolution always re-reads the world cell Record named by lane[3].

BCC32_TURN_WORLD_BINDING_HD inline std::uint32_t find_world_cell(
    const ResidentRewriteState* state, std::uint32_t world_owner,
    bool* ambiguous = nullptr) {
  if (ambiguous != nullptr) *ambiguous = false;
  std::uint32_t found = kInvalid;
  if (state == nullptr || world_owner == 0u || world_owner == kInvalid)
    return found;
  for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
    const Record& record = state->records[slot];
    if (record.matter_q8 == 0u || record.lane[0] != kFormWorldCell ||
        record.lane[1] != world_owner)
      continue;
    if (found != kInvalid) {
      if (ambiguous != nullptr) *ambiguous = true;
      return kInvalid;
    }
    found = slot;
  }
  return found;
}

BCC32_TURN_WORLD_BINDING_HD inline std::uint32_t find_claim(
    const ResidentRewriteState* state, std::uint32_t claim_owner) {
  std::uint32_t found = kInvalid;
  if (state == nullptr || claim_owner == 0u || claim_owner == kInvalid)
    return found;
  for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
    const Record& record = state->records[slot];
    if (record.matter_q8 == 0u || record.lane[0] != kFormWorldCellClaim ||
        record.lane[1] != claim_owner)
      continue;
    if (found != kInvalid) return kInvalid;
    found = slot;
  }
  return found;
}

BCC32_TURN_WORLD_BINDING_HD inline std::uint32_t find_binding_for_claim(
    const ResidentRewriteState* state, std::uint32_t claim_owner) {
  std::uint32_t found = kInvalid;
  if (state == nullptr || claim_owner == 0u || claim_owner == kInvalid)
    return found;
  for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
    const Record& record = state->records[slot];
    if (record.matter_q8 == 0u || record.lane[0] != kFormWorldCellBinding ||
        record.lane[2] != claim_owner)
      continue;
    if (found != kInvalid) return kInvalid;
    found = slot;
  }
  return found;
}

BCC32_TURN_WORLD_BINDING_HD inline bool owner_free(
    const ResidentRewriteState* state, std::uint32_t owner) {
  if (state == nullptr || owner == 0u || owner == kInvalid) return false;
  for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
    const Record& record = state->records[slot];
    if (record.matter_q8 != 0u && record.lane[1] == owner) return false;
  }
  return true;
}

BCC32_TURN_WORLD_BINDING_HD inline std::uint32_t make_owner_token(
    const ResidentRewriteState* state, std::uint32_t form,
    std::uint32_t salt) {
  if (state == nullptr) return kInvalid;
  std::uint32_t owner = rewrite_mix(form, salt, state->revision);
  for (std::uint32_t attempt = 0u; attempt < kRecordCapacity; ++attempt) {
    if (owner != 0u && owner != kInvalid && owner_free(state, owner))
      return owner;
    owner = rewrite_mix(owner, salt, attempt + 1u);
  }
  return kInvalid;
}

// Turn 1: a physically-grounded action writes the resident world cell. The
// first call allocates the cell; every later call updates the SAME Record's
// lane[2] in place and advances its monotone write receipt in lane[3] -- no
// second, host-side copy of the value ever exists. Ambiguous prior matter
// (a malformed multi-Record cell) refuses to guess and fails closed.
BCC32_TURN_WORLD_BINDING_HD inline bool apply_grounded_world_write(
    ResidentRewriteState* state, std::uint32_t world_owner,
    std::uint32_t value, std::uint32_t* world_slot = nullptr) {
  if (world_slot != nullptr) *world_slot = kInvalid;
  if (state == nullptr || state->fault != 0u || world_owner == 0u ||
      world_owner == kInvalid)
    return false;
  bool ambiguous = false;
  std::uint32_t slot = find_world_cell(state, world_owner, &ambiguous);
  if (ambiguous) return false;
  if (slot == kInvalid) {
    slot = allocate_record(state);
    if (slot == kInvalid) return false;
    Record& cell = state->records[slot];
    cell.lane[0] = kFormWorldCell;
    cell.lane[1] = world_owner;
    cell.lane[2] = value;
    cell.lane[3] = 1u;
    ++cell.revision;
  } else {
    Record& cell = state->records[slot];
    cell.lane[2] = value;
    ++cell.lane[3];
    ++cell.revision;
  }
  ++state->revision;
  refresh_receipt(state);
  if (world_slot != nullptr) *world_slot = slot;
  return true;
}

BCC32_TURN_WORLD_BINDING_HD inline bool world_cell_current_value(
    const ResidentRewriteState* state, std::uint32_t world_owner,
    std::uint32_t* value) {
  if (value != nullptr) *value = kInvalid;
  bool ambiguous = false;
  const std::uint32_t slot = find_world_cell(state, world_owner, &ambiguous);
  if (slot == kInvalid || ambiguous) return false;
  if (value != nullptr) *value = state->records[slot].lane[2];
  return true;
}

// Allocates a fresh, contentless claim token that a later binding can be
// attached to.
BCC32_TURN_WORLD_BINDING_HD inline bool make_claim(
    ResidentRewriteState* state, std::uint32_t salt,
    std::uint32_t* claim_owner) {
  if (claim_owner != nullptr) *claim_owner = kInvalid;
  if (state == nullptr || state->fault != 0u) return false;
  const std::uint32_t owner = make_owner_token(state, kFormWorldCellClaim, salt);
  if (owner == kInvalid) return false;
  const std::uint32_t slot = allocate_record(state);
  if (slot == kInvalid) return false;
  Record& claim = state->records[slot];
  claim.lane[0] = kFormWorldCellClaim;
  claim.lane[1] = owner;
  ++claim.revision;
  ++state->revision;
  refresh_receipt(state);
  if (claim_owner != nullptr) *claim_owner = owner;
  return true;
}

// Claim lane[2]: an opaque subject token (e.g. a producer locus) this claim
// is scoped to, or 0/kInvalid for a subject-less claim (make_claim()'s
// existing behavior, unchanged). This header never guesses or invents a
// subject; it only stores what a caller supplies and matches it back
// verbatim, the same discipline apply_grounded_world_write already uses for
// world_owner.
BCC32_TURN_WORLD_BINDING_HD inline std::uint32_t find_claim_for_subject(
    const ResidentRewriteState* state, std::uint32_t subject,
    bool* ambiguous = nullptr) {
  if (ambiguous != nullptr) *ambiguous = false;
  std::uint32_t found = kInvalid;
  if (state == nullptr || subject == 0u || subject == kInvalid)
    return found;
  for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
    const Record& record = state->records[slot];
    if (record.matter_q8 == 0u || record.lane[0] != kFormWorldCellClaim ||
        record.lane[2] != subject)
      continue;
    if (found != kInvalid) {
      if (ambiguous != nullptr) *ambiguous = true;
      return kInvalid;
    }
    found = slot;
  }
  return found;
}

// Idempotent claim discovery scoped to an opaque subject token. make_claim()
// alone hands back a fresh, collision-probed owner token with no way to
// recompute it from a subject on a later, independent call -- forcing a
// caller who wants one persistent claim per subject (e.g. one claim per
// producer locus, reused across turns rather than recreated) to keep an
// external subject->claim_owner lookup table. That table would itself be
// exactly the kind of non-resident cognitive authority this project's
// doctrine forbids. ensure_subject_claim rediscovers an existing claim
// already scoped to `subject` by scanning resident matter the same way
// find_world_cell already does, and only allocates a new one when none
// exists. An ambiguous prior claim (more than one Record scoped to the same
// subject, which this function itself can never produce but a malformed
// caller elsewhere could) refuses to guess and fails closed.
BCC32_TURN_WORLD_BINDING_HD inline bool ensure_subject_claim(
    ResidentRewriteState* state, std::uint32_t subject, std::uint32_t salt,
    std::uint32_t* claim_owner) {
  if (claim_owner != nullptr) *claim_owner = kInvalid;
  if (state == nullptr || state->fault != 0u || subject == 0u ||
      subject == kInvalid)
    return false;
  bool ambiguous = false;
  const std::uint32_t existing_slot =
      find_claim_for_subject(state, subject, &ambiguous);
  if (ambiguous) return false;
  if (existing_slot != kInvalid) {
    if (claim_owner != nullptr)
      *claim_owner = state->records[existing_slot].lane[1];
    return true;
  }
  std::uint32_t owner = kInvalid;
  if (!make_claim(state, salt, &owner)) return false;
  const std::uint32_t slot = find_claim(state, owner);
  if (slot == kInvalid) return false;
  state->records[slot].lane[2] = subject;
  ++state->records[slot].revision;
  ++state->revision;
  refresh_receipt(state);
  if (claim_owner != nullptr) *claim_owner = owner;
  return true;
}

// Turn 1: bind an existing claim to an existing, uniquely identified world
// cell. The binding stores only the two owner tokens -- it is the live
// pointer, not a value snapshot. At most one binding may exist per claim so
// resolution can never become ambiguous by construction.
BCC32_TURN_WORLD_BINDING_HD inline bool bind_claim_to_world_cell(
    ResidentRewriteState* state, std::uint32_t claim_owner,
    std::uint32_t world_owner) {
  if (state == nullptr || state->fault != 0u) return false;
  if (find_claim(state, claim_owner) == kInvalid) return false;
  bool ambiguous = false;
  if (find_world_cell(state, world_owner, &ambiguous) == kInvalid || ambiguous)
    return false;
  if (find_binding_for_claim(state, claim_owner) != kInvalid) return false;
  const std::uint32_t owner = make_owner_token(
      state, kFormWorldCellBinding, rewrite_mix(claim_owner, world_owner, 0u));
  if (owner == kInvalid) return false;
  const std::uint32_t slot = allocate_record(state);
  if (slot == kInvalid) return false;
  Record& binding = state->records[slot];
  binding.lane[0] = kFormWorldCellBinding;
  binding.lane[1] = owner;
  binding.lane[2] = claim_owner;
  binding.lane[3] = world_owner;
  ++binding.revision;
  ++state->revision;
  refresh_receipt(state);
  return true;
}

// Turn 2: the held-out query. It resolves the claim's unique binding and
// then re-reads the world cell Record the binding currently names -- always
// the physically CURRENT value, never a value captured at bind time. If the
// binding was severed, or the world cell it names is gone or ambiguous, or
// a full reset removed every Record, resolution fails rather than returning
// a stale or guessed value.
BCC32_TURN_WORLD_BINDING_HD inline bool resolve_claim_current_value(
    const ResidentRewriteState* state, std::uint32_t claim_owner,
    std::uint32_t* value) {
  if (value != nullptr) *value = kInvalid;
  if (state == nullptr) return false;
  const std::uint32_t binding_slot = find_binding_for_claim(state, claim_owner);
  if (binding_slot == kInvalid) return false;
  const Record& binding = state->records[binding_slot];
  return world_cell_current_value(state, binding.lane[3], value);
}

// Resolves the current world value reachable from a resident source-revision
// identity, through its subject-scoped claim (see find_claim_for_subject /
// ensure_subject_claim above) and that claim's live binding. Unlike
// resolve_claim_current_value, this starts from the subject identity, not
// from an already-known claim owner.
struct SourceClaimResolution {
  std::uint32_t claim_owner = kInvalid;
  std::uint32_t world_owner = kInvalid;
  std::uint32_t value = kInvalid;
  bool present = false;
  bool valid = false;
  bool ambiguous = false;
};
BCC32_TURN_WORLD_BINDING_HD inline SourceClaimResolution
resolve_source_claim_current_value(const ResidentRewriteState* state,
                                   std::uint32_t source_owner) {
  SourceClaimResolution result{};
  bool ambiguous = false;
  const std::uint32_t claim_slot =
      find_claim_for_subject(state, source_owner, &ambiguous);
  result.ambiguous = ambiguous;
  if (ambiguous || claim_slot == kInvalid) return result;
  result.present = true;
  result.claim_owner = state->records[claim_slot].lane[1];
  const std::uint32_t binding_slot =
      find_binding_for_claim(state, result.claim_owner);
  if (binding_slot == kInvalid) return result;
  result.world_owner = state->records[binding_slot].lane[3];
  if (!resolve_claim_current_value(state, result.claim_owner, &result.value))
    return result;
  result.valid = true;
  return result;
}

}  // namespace turn_world_consequence_binding

#undef BCC32_TURN_WORLD_BINDING_HD
