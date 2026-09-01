#pragma once

#include "bcc32_resident_mixed_provenance_evidence.cuh"

#include <cstdint>

#if defined(__CUDACC__)
#define BCC32_RELATION_SURFACE_HD __host__ __device__
#else
#define BCC32_RELATION_SURFACE_HD
#endif

// SCAFFOLD, NOT FINAL ARCHITECTURE. Independent review (2026-08-14, see
// docs/diary) found this mechanism is, underneath the framing comments, an
// exact (operand_a, relation_tag, operand_b) -> result associative table row
// implemented as a Record: form_relation_from_lived_contact assigns
// result = words[3] directly from the fourth teaching word, and lookup is an
// exact-match slot scan returning record.lane[5]. The three "surface
// invariance" paths are real code-path diversity but all query the same
// single authoritative Record, so this is a localist ledger and a form of
// episode-remainder-as-answer (the source trajectory is deleted, but the
// fourth word survives losslessly as lane[5]), not learned cross-surface
// abstraction. Useful engineering scaffold proving several render paths can
// reach one resident relation; do not build further capabilities on
// kFormRelationSurfaceInvariance as if held-out surfaces recruiting
// overlapping resident organization had been demonstrated.
//
// The distributed replacement this needs already exists and is landed,
// green, and comprehensively tested:
// bcc32_resident_causal_constraint_participation.cuh
// (substrate::bcc32::resident_causal_constraint_participation). It splits a
// relation into separate antecedent/consequent participation-fragment
// Records with no Record ever holding a complete relation
// ("no Record holds a complete local relation, and complete paths are
// never materialized: they exist only while a probe gathers overlapping
// independently sourced pairs" -- its own header comment), and its contract
// (bcc32_cuda_resident_causal_constraint_participation_contract.cu) already
// measures single-record-sweep independence (single_record_sweep=48),
// graded lesion response (damage_curve=1,2,4,8), focal-vs-remote-lesion
// contrast, source withdrawal and regrowth, and held-out cross-language
// transfer -- exactly the properties this file's own re-render paths do not
// establish. Do not design a new participation primitive from scratch; wire
// M6 relation persistence onto this existing mechanism instead of inventing
// a parallel one.
//
// M6 cross-surface relation persistence.  One lived external contact
// (operand_a, relation_tag, operand_b, result) becomes exactly one persistent
// bound-relation Record.  That Record is the entire causal content: the
// mechanism below never stores a second copy of the relation and never lets a
// query mutate it.  Three structural re-render paths (operand-order swap, an
// additive-offset identity remap of the surface encoding, and a reversed
// operand presentation order) query and render that one Record through
// distinct code paths and must all resolve to the identical raw result word.
// None of them names a cell, field, layer, network, region, or gradient; each
// is an ordinary Record-slot scan over opaque uint32 lanes.
namespace substrate::bcc32::causal_rewrite::relation_surface_invariance {

inline constexpr std::uint32_t kFormRelationSurfaceInvariance = 0x51fa8b2cu;

// A lived teaching contact is exactly four raw words: operand_a, the
// relation's opaque tag, operand_b, and the taught result.  This is generic
// physical framing (an episode length), not a parser or a semantic slot.
inline constexpr std::uint32_t kRelationSurfaceEpisodeLength = 4u;
inline constexpr std::uint32_t kRelationSurfaceBound = 1u;

// A fixed additive constant used only to prove that a surface encoding can be
// applied and reversed without touching the underlying Record.  It carries no
// meaning of its own; any nonzero uint32 would do the same structural job.
inline constexpr std::uint32_t kRelationSurfaceOffset = 0x2f6a1c85u;

// Bound-relation Record lanes:
//   [1] owner; [2] operand_a; [3] operand_b; [4] relation_tag; [5] result;
//   [6] source contact owner (provenance only, never matched or compared);
//   [7] bound-state flag.

BCC32_RELATION_SURFACE_HD inline bool relation_owner_free(
    const ResidentRewriteState* state, std::uint32_t owner) {
  if (state == nullptr || owner == 0u || owner == kInvalid) return false;
  for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
    const Record& record = state->records[slot];
    if (record.matter_q8 != 0u && record.lane[1] == owner) return false;
  }
  return true;
}

BCC32_RELATION_SURFACE_HD inline std::uint32_t make_relation_owner(
    const ResidentRewriteState* state, std::uint32_t salt) {
  if (state == nullptr) return kInvalid;
  std::uint32_t owner = rewrite_mix(
      kFormRelationSurfaceInvariance, salt,
      static_cast<std::uint32_t>(state->revision));
  for (std::uint32_t attempt = 0u; attempt < kRecordCapacity; ++attempt) {
    if (relation_owner_free(state, owner)) return owner;
    owner = rewrite_mix(owner, salt, attempt + 1u);
  }
  return kInvalid;
}

BCC32_RELATION_SURFACE_HD inline bool relation_record_valid(
    const ResidentRewriteState* state, std::uint32_t slot) {
  if (state == nullptr || slot >= live_record_capacity(state)) return false;
  const Record& record = state->records[slot];
  return record.matter_q8 != 0u &&
         record.lane[0] == kFormRelationSurfaceInvariance &&
         record.lane[1] != 0u && record.lane[1] != kInvalid &&
         record.lane[7] == kRelationSurfaceBound;
}

BCC32_RELATION_SURFACE_HD inline std::uint32_t unique_relation_slot_by_owner(
    const ResidentRewriteState* state, std::uint32_t owner) {
  std::uint32_t found = kInvalid;
  if (state == nullptr || owner == 0u || owner == kInvalid) return found;
  for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
    const Record& record = state->records[slot];
    if (record.matter_q8 == 0u ||
        record.lane[0] != kFormRelationSurfaceInvariance ||
        record.lane[1] != owner)
      continue;
    if (found != kInvalid) return kInvalid;
    found = slot;
  }
  return found;
}

BCC32_RELATION_SURFACE_HD inline std::uint32_t
unique_relation_slot_by_operands(const ResidentRewriteState* state,
                                 std::uint32_t operand_a,
                                 std::uint32_t relation_tag,
                                 std::uint32_t operand_b) {
  std::uint32_t found = kInvalid;
  if (state == nullptr) return found;
  for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
    const Record& record = state->records[slot];
    if (record.matter_q8 == 0u ||
        record.lane[0] != kFormRelationSurfaceInvariance ||
        record.lane[2] != operand_a || record.lane[4] != relation_tag ||
        record.lane[3] != operand_b)
      continue;
    if (found != kInvalid) return kInvalid;
    found = slot;
  }
  return found;
}

// The presented pair is deliberately taken in the opposite order from the
// stored (operand_a, operand_b) pair: presented_first must equal the stored
// operand_b and presented_second must equal the stored operand_a.  This is a
// distinct query-order code path, not a normalized re-sort of one pair.
BCC32_RELATION_SURFACE_HD inline std::uint32_t
unique_relation_slot_by_swapped_operands(const ResidentRewriteState* state,
                                         std::uint32_t presented_first,
                                         std::uint32_t relation_tag,
                                         std::uint32_t presented_second) {
  std::uint32_t found = kInvalid;
  if (state == nullptr) return found;
  for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
    const Record& record = state->records[slot];
    if (record.matter_q8 == 0u ||
        record.lane[0] != kFormRelationSurfaceInvariance ||
        record.lane[3] != presented_first || record.lane[4] != relation_tag ||
        record.lane[2] != presented_second)
      continue;
    if (found != kInvalid) return kInvalid;
    found = slot;
  }
  return found;
}

// Forms exactly one persistent bound-relation Record from the currently open
// wholly-external trajectory, if and only if that trajectory is precisely one
// lived four-word teaching contact.  This mirrors the open-inquiry teacher-
// surface capture: it must run before the physical END event closes the
// trajectory, so the generic Program-induction path never also folds this
// contact into unrelated evidence.  The lived contact's source trajectory is
// retired here; it never survives as duplicate resident matter.
BCC32_RELATION_SURFACE_HD inline bool form_relation_from_lived_contact(
    ResidentRewriteState* state, std::uint32_t* out_owner = nullptr) {
  if (out_owner != nullptr) *out_owner = kInvalid;
  if (state == nullptr || state->fault != 0u) return false;
  const std::uint32_t trajectory_slot = find_current_trajectory(state);
  if (trajectory_slot == kInvalid) return false;
  const Record& trajectory = state->records[trajectory_slot];
  if (!open_inquiry::wholly_external_trajectory(state, trajectory) ||
      trajectory.lane[2] != kRelationSurfaceEpisodeLength)
    return false;

  std::uint32_t words[kRelationSurfaceEpisodeLength]{};
  for (std::uint32_t index = 0u; index < kRelationSurfaceEpisodeLength;
       ++index) {
    if (!trajectory_word_at(state, trajectory.lane[1], index, &words[index]))
      return false;
  }
  const std::uint32_t operand_a = words[0];
  const std::uint32_t relation_tag = words[1];
  const std::uint32_t operand_b = words[2];
  const std::uint32_t result = words[3];
  if (operand_a == 0u || operand_a == kInvalid || operand_b == 0u ||
      operand_b == kInvalid || relation_tag == 0u ||
      relation_tag == kInvalid || result == 0u || result == kInvalid)
    return false;

  // Refuse a second binding for one already-bound operand/tag identity: the
  // surface encoding always addresses exactly one persistent relation
  // Record, never a growing table the host could pick among.
  if (unique_relation_slot_by_operands(state, operand_a, relation_tag,
                                       operand_b) != kInvalid)
    return false;

  const std::uint32_t salt = rewrite_mix(operand_a, relation_tag, operand_b);
  const std::uint32_t owner = make_relation_owner(state, salt);
  if (owner == kInvalid) return false;
  if (free_record_count(state) == 0u) return false;

  const std::uint32_t slot = allocate_record(state);
  if (slot == kInvalid) return false;
  Record& record = state->records[slot];
  record.lane[0] = kFormRelationSurfaceInvariance;
  record.lane[1] = owner;
  record.lane[2] = operand_a;
  record.lane[3] = operand_b;
  record.lane[4] = relation_tag;
  record.lane[5] = result;
  record.lane[6] = trajectory.lane[1];
  record.lane[7] = kRelationSurfaceBound;
  ++record.revision;

  const std::uint32_t source_owner = trajectory.lane[1];
  mixed_provenance::clear_provenance(state, source_owner);
  clear_trajectory(state, trajectory_slot);
  ++state->revision;
  refresh_receipt(state);
  if (out_owner != nullptr) *out_owner = owner;
  return true;
}

// Canonical re-render: query with operands in the exact order they were
// taught.  This is the baseline the other three surfaces are checked against.
BCC32_RELATION_SURFACE_HD inline bool render_relation_result_canonical(
    const ResidentRewriteState* state, std::uint32_t operand_a,
    std::uint32_t relation_tag, std::uint32_t operand_b,
    std::uint32_t* result) {
  if (result == nullptr) return false;
  const std::uint32_t slot =
      unique_relation_slot_by_operands(state, operand_a, relation_tag,
                                       operand_b);
  if (!relation_record_valid(state, slot)) return false;
  *result = state->records[slot].lane[5];
  return true;
}

// Surface 1: operand-order swap.  The caller presents the two operands in
// reversed order; the query path matches them against the reversed lanes.
BCC32_RELATION_SURFACE_HD inline bool render_relation_result_operand_swapped(
    const ResidentRewriteState* state, std::uint32_t presented_first,
    std::uint32_t relation_tag, std::uint32_t presented_second,
    std::uint32_t* result) {
  if (result == nullptr) return false;
  const std::uint32_t slot = unique_relation_slot_by_swapped_operands(
      state, presented_first, relation_tag, presented_second);
  if (!relation_record_valid(state, slot)) return false;
  *result = state->records[slot].lane[5];
  return true;
}

// Surface 2: additive-offset identity remap.  The query surface is encoded
// with a fixed additive offset and decoded again before it ever touches the
// resident Record; the offset's net effect on identity is exactly zero, so
// this exercises a genuinely different render path while remaining causally
// inert.
BCC32_RELATION_SURFACE_HD inline std::uint32_t surface_offset_encode(
    std::uint32_t raw_word) {
  return raw_word + kRelationSurfaceOffset;
}

BCC32_RELATION_SURFACE_HD inline std::uint32_t surface_offset_decode(
    std::uint32_t encoded_word) {
  return encoded_word - kRelationSurfaceOffset;
}

BCC32_RELATION_SURFACE_HD inline bool render_relation_result_offset_surface(
    const ResidentRewriteState* state, std::uint32_t operand_a,
    std::uint32_t relation_tag, std::uint32_t operand_b,
    std::uint32_t* result) {
  if (result == nullptr) return false;
  const std::uint32_t recovered_a =
      surface_offset_decode(surface_offset_encode(operand_a));
  const std::uint32_t recovered_tag =
      surface_offset_decode(surface_offset_encode(relation_tag));
  const std::uint32_t recovered_b =
      surface_offset_decode(surface_offset_encode(operand_b));
  const std::uint32_t slot = unique_relation_slot_by_operands(
      state, recovered_a, recovered_tag, recovered_b);
  if (!relation_record_valid(state, slot)) return false;
  *result = surface_offset_decode(
      surface_offset_encode(state->records[slot].lane[5]));
  return true;
}

// Surface 3: reversed operand presentation order.  The emitted surface array
// lists operand_b before operand_a even though the query is addressed by the
// relation's resident owner identity, not by re-sorted operand values.
struct RelationSurface {
  std::uint32_t first_operand = kInvalid;
  std::uint32_t second_operand = kInvalid;
  std::uint32_t relation_tag = kInvalid;
  std::uint32_t result = kInvalid;
};

BCC32_RELATION_SURFACE_HD inline bool render_relation_surface_reversed(
    const ResidentRewriteState* state, std::uint32_t owner,
    RelationSurface* surface) {
  if (surface != nullptr) *surface = RelationSurface{};
  if (surface == nullptr) return false;
  const std::uint32_t slot = unique_relation_slot_by_owner(state, owner);
  if (!relation_record_valid(state, slot)) return false;
  const Record& record = state->records[slot];
  surface->first_operand = record.lane[3];   // operand_b presented first
  surface->second_operand = record.lane[2];  // operand_a presented second
  surface->relation_tag = record.lane[4];
  surface->result = record.lane[5];
  return true;
}

}  // namespace substrate::bcc32::causal_rewrite::relation_surface_invariance

#undef BCC32_RELATION_SURFACE_HD
