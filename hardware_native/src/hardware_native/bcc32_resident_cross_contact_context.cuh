#pragma once

#include "causal_rewrite_universe.cuh"

#include <cstdint>

#if defined(__CUDACC__)
#define BCC32_CROSS_CONTACT_HD __host__ __device__
#else
#define BCC32_CROSS_CONTACT_HD
#endif

// Device stack is reserved per *resident thread slot*, not per launched
// thread: on the sm_89 production device that is 80 SMs x 1536 slots, so each
// KiB of cudaLimitStackSize costs 120 MiB of the card before any work runs.
// The dormant-compaction helpers below each hold multi-KiB scratch arrays
// (payload/term_slots/joined_words are kMaximumTrajectoryEvents words apiece).
// While they are merely `inline`, nvcc folds several of them into one caller
// frame and their locals *sum* instead of overlapping -- measured on the
// linked cubin, detach_active_carry_before_new_contact carries no locals of
// its own yet reported a 31,896-byte frame, which outlining takes to 2,768.
// Outlining keeps each scratch array in its own frame, so a chain pays the
// maximum rather than the sum.
//
// Do not evaluate this change on its own. Measured alone it moves the
// whole-program requirement by *zero*, because a second chain of near-equal
// length runs through learned_cost_search::select_resident_learned_cost_route
// and simply becomes the new maximum. Paired with hoisting that search's
// working set off the stack, the two together took the acyclic floor from
// 103,120 to 68,112 bytes. Either one alone reads as a null result, and
// reverting "the one that did nothing" restores the whole 34 KiB.
//
// These are cold boundary-event paths (END/PAUSE, carry detach, dormant
// join), so the added call overhead sits on no hot loop. This is a
// code-generation directive only: it changes no logic, no ordering, and no
// resident state. It mirrors BCC32_CROSS_CONTEXT_DISPATCH in
// bcc32_resident_cross_context_anti_unification.cuh, which exists for exactly
// the same reason -- see its comment about keeping the bounded structural
// compiler "independent of kMaximumTrajectoryEvents [to make] the CUDA stack
// cost explicit."
#if defined(__CUDACC__)
#define BCC32_CROSS_CONTACT_OUTLINE \
  [[maybe_unused]] static __host__ __device__ __noinline__
#else
#define BCC32_CROSS_CONTACT_OUTLINE [[maybe_unused]] static inline
#endif

namespace substrate::bcc32::causal_rewrite::cross_contact {

// Carry is provenance only.  It is deliberately stored in the existing
// trajectory Record, beside the generated-evidence bit, and never participates
// in matching, support, induction, or output selection.
// The carry bit is core trajectory ABI, so the close path can enforce its
// zero-authority provenance before developmental processing.

using causal_rewrite::Record;
using causal_rewrite::ResidentRewriteEngine;
using causal_rewrite::ResidentRewriteState;
using causal_rewrite::kTrajectoryHasCarry;

inline constexpr std::uint32_t kCrossContactCurrentMultiplicityFault =
    0x4358434du;

BCC32_CROSS_CONTACT_HD inline bool unique_current_trajectory(
    const ResidentRewriteState* state) {
  if (state == nullptr) return false;
  std::uint32_t count = 0u;
  for (std::uint32_t slot = 0u; slot < causal_rewrite::live_record_capacity(state);
       ++slot) {
    const Record& record = state->records[slot];
    if (record.matter_q8 != 0u &&
        record.lane[0] == causal_rewrite::kFormTrajectory &&
        record.lane[3] == 0u)
      ++count;
  }
  return count <= 1u;
}

struct SpanPrefixWitness {
  std::uint32_t trajectory_slot = causal_rewrite::kInvalid;
  std::uint32_t program_slot = causal_rewrite::kInvalid;
  std::uint32_t next_word = 0u;
  bool found = false;
};

BCC32_CROSS_CONTACT_HD inline bool is_carried_history(
    const Record& trajectory) {
  return (trajectory.lane[7] & kTrajectoryHasCarry) != 0u;
}

BCC32_CROSS_CONTACT_HD inline bool is_generated_or_carried_history(
    const Record& trajectory) {
  return trajectory.lane[7] != 0u;
}

BCC32_CROSS_CONTACT_HD inline std::uint32_t find_trajectory_by_owner(
    const ResidentRewriteState* state, std::uint32_t owner) {
  if (state == nullptr || owner == causal_rewrite::kInvalid) return causal_rewrite::kInvalid;
  for (std::uint32_t slot = 0u; slot < causal_rewrite::live_record_capacity(state);
       ++slot) {
    const Record& trajectory = state->records[slot];
    if (trajectory.matter_q8 != 0u &&
        trajectory.lane[0] == causal_rewrite::kFormTrajectory &&
        trajectory.lane[1] == owner)
      return slot;
  }
  return causal_rewrite::kInvalid;
}

BCC32_CROSS_CONTACT_HD inline bool find_unique_bound_span_prefix_impl(
    const ResidentRewriteState* state, std::uint32_t trajectory_slot,
    SpanPrefixWitness* witness, bool allow_carried,
    bool allow_generated = false) {
  if (witness != nullptr) *witness = SpanPrefixWitness{};
  if (state == nullptr || witness == nullptr ||
      trajectory_slot >= causal_rewrite::live_record_capacity(state))
    return false;

  const Record& trajectory = state->records[trajectory_slot];
  const std::uint32_t flags = trajectory.lane[7];
  const bool carried = (flags & kTrajectoryHasCarry) != 0u;
  const bool generated =
      (flags & causal_rewrite::kTrajectoryHasGenerated) != 0u;
  if (trajectory.matter_q8 == 0u ||
      trajectory.lane[0] != causal_rewrite::kFormTrajectory ||
      trajectory.lane[2] == 0u ||
      (generated && !allow_generated) ||
      (flags & ~(kTrajectoryHasCarry |
                 causal_rewrite::kTrajectoryHasGenerated)) != 0u ||
      (carried && !allow_carried))
    return false;

  bool multiple = false;
  // A carried contact must be a contact-prefix from its own beginning.  This
  // prevents unrelated preceding bytes from becoming a hidden suffix context.
  constexpr std::uint32_t kPrefixStart = 0u;
  for (std::uint32_t program_slot = 0u;
       program_slot < causal_rewrite::live_record_capacity(state); ++program_slot) {
    const Record& program = state->records[program_slot];
    if (program.matter_q8 == 0u ||
        program.lane[0] != causal_rewrite::kFormSpanProgram ||
        !causal_rewrite::resident_program_authoritative(state, program_slot) ||
        program.lane[6] == trajectory.lane[1])
      continue;

    std::uint32_t bindings[causal_rewrite::kMaximumProgramVariables]
                          [causal_rewrite::kMaximumVariableSpanEvents]{};
    std::uint32_t lengths[causal_rewrite::kMaximumProgramVariables]{};
    std::uint32_t next_term = causal_rewrite::kInvalid;
    std::uint32_t next_offset = 0u;
    bool next_unbound = false;
    bool complete = false;
    bool ambiguous = false;
    const std::uint32_t match = causal_rewrite::span_match_prefix_at(
        state, program, trajectory, kPrefixStart, &next_term, &next_offset,
        &next_unbound, &complete, &ambiguous, bindings, lengths);
    if (match != causal_rewrite::kSpanMatchPrefix || complete || next_unbound ||
        ambiguous || next_term == causal_rewrite::kInvalid)
      continue;

    std::uint32_t kind = 0u;
    std::uint32_t value = 0u;
    std::uint32_t ignored_left = 0u;
    std::uint32_t ignored_right = 0u;
    if (!causal_rewrite::span_program_term_at(
            state, program.lane[1], next_term, &kind, &value, &ignored_left,
            &ignored_right))
      continue;

    std::uint32_t next_word = 0u;
    if (kind == causal_rewrite::kSpanTermLiteral) {
      next_word = value;
    } else if (value < causal_rewrite::kMaximumProgramVariables &&
               lengths[value] != 0u && next_offset < lengths[value]) {
      next_word = bindings[value][next_offset];
    } else {
      continue;
    }

    if (witness->found) {
      multiple = true;
      continue;
    }
    witness->trajectory_slot = trajectory_slot;
    witness->program_slot = program_slot;
    witness->next_word = next_word;
    witness->found = true;
  }
  if (multiple) {
    *witness = SpanPrefixWitness{};
    return false;
  }
  return witness->found;
}

BCC32_CROSS_CONTACT_HD inline bool find_unique_bound_span_prefix(
    const ResidentRewriteState* state, std::uint32_t trajectory_slot,
    SpanPrefixWitness* witness) {
  return find_unique_bound_span_prefix_impl(state, trajectory_slot, witness,
                                            false);
}

#include "bcc32_resident_dormant_discourse.inl"

BCC32_CROSS_CONTACT_HD inline bool promote_retained_span_prefix_after_end(
    ResidentRewriteState* state,
    std::uint32_t preflight_candidate = causal_rewrite::kInvalid,
    const SpanPrefixWitness* preflight_witness = nullptr) {
  if (state == nullptr || causal_rewrite::find_current_trajectory(state) !=
                              causal_rewrite::kInvalid)
    return false;

  std::uint32_t candidate = preflight_candidate;
  SpanPrefixWitness witness{};
  if (preflight_witness != nullptr) witness = *preflight_witness;
  if (candidate == causal_rewrite::kInvalid) {
    for (std::uint32_t slot = 0u; slot < causal_rewrite::live_record_capacity(state);
         ++slot) {
      const Record& trajectory = state->records[slot];
      if (trajectory.matter_q8 == 0u ||
          trajectory.lane[0] != causal_rewrite::kFormTrajectory ||
          trajectory.lane[3] == 0u ||
          is_generated_or_carried_history(trajectory))
        continue;
      SpanPrefixWitness local{};
      if (!find_unique_bound_span_prefix(state, slot, &local)) continue;
      if (candidate != causal_rewrite::kInvalid) return false;
      candidate = slot;
      witness = local;
    }
  }
  if (candidate >= causal_rewrite::live_record_capacity(state) ||
      witness.trajectory_slot != candidate ||
      witness.program_slot >= causal_rewrite::live_record_capacity(state) ||
      !causal_rewrite::resident_program_authoritative(
          state, witness.program_slot))
    return false;

  Record& trajectory = state->records[candidate];
  if (trajectory.matter_q8 == 0u ||
      trajectory.lane[0] != causal_rewrite::kFormTrajectory ||
      trajectory.lane[3] == 0u || is_generated_or_carried_history(trajectory))
    return false;
  // Mark the settled external prefix as the active carry. Keep its exact
  // ingress provenance until the next physical contact demotes it: compact
  // matter is a dormant representation and must never masquerade as the
  // current raw trajectory between END and that next contact.
  trajectory.lane[7] = kTrajectoryHasCarry;
  ++trajectory.revision;
  trajectory.lane[3] = 0u;  // current context
  trajectory.lane[4] = 0u;  // not yet yielded after promotion
  ++trajectory.revision;
  ++state->revision;
  causal_rewrite::refresh_receipt(state);
  return true;
}

// This is the bounded integration seam for the canonical epoch consumer.
// The newest carry keeps RWR11's immediate post-END shape. The first event of
// a later contact detaches it into dormant resident matter; END ages dormant
// histories, while PAUSE may resume exactly one uniquely matching history.
BCC32_CROSS_CONTACT_HD inline void consume_cross_contact_event(
    ResidentRewriteEngine engine,
                                        causal_rewrite::RawRewriteEvent event,
                                        bool update_receipt = true) {
  if (engine.state == nullptr) return;
  if (!unique_current_trajectory(engine.state)) {
    if (engine.state->fault == 0u)
      engine.state->fault = kCrossContactCurrentMultiplicityFault;
    causal_rewrite::refresh_receipt(engine.state);
    return;
  }
  if (event.valid != 0u &&
      event.reserved == causal_rewrite::kEventFrameNone) {
    // Demote the prior carried contact before ranking dormant matter. This is
    // allocation-free, and prevents the newest carry from escaping the same
    // pressure law applied to older discourse solely because it was active.
    (void)detach_active_carry_before_new_contact(engine.state);
    reclaim_dormant_for_record_pressure(
        engine.state, kDormantOperationalReserve);
    if (engine.state->fault != 0u) return;
  }

  bool preserve_bound_prefix = false;
  std::uint32_t preserved_prefix_slot = causal_rewrite::kInvalid;
  SpanPrefixWitness preserved_prefix_witness{};
  if (event.reserved == causal_rewrite::kEventFrameEnd) {
    (void)detach_active_carry_before_new_contact(engine.state);
    age_dormant_histories_at_end(engine.state);
    const std::uint32_t prefix_slot =
        causal_rewrite::find_current_trajectory(engine.state);
    if (prefix_slot != causal_rewrite::kInvalid) {
      SpanPrefixWitness witness{};
      const bool carried =
          is_carried_history(engine.state->records[prefix_slot]);
      if (carried) {
        // A generated Carry is zero-authority execution context, not an
        // external prefix. Keep it across this physical END so the next raw
        // connective can reach the resident relation ecology; ordinary
        // external Carry still retires at the boundary and cannot become a
        // hidden teaching episode.
        if ((engine.state->records[prefix_slot].lane[7] &
             causal_rewrite::kTrajectoryHasGenerated) == 0u)
          causal_rewrite::clear_trajectory(engine.state, prefix_slot);
      } else {
        preserve_bound_prefix = find_unique_bound_span_prefix_impl(
            engine.state, prefix_slot, &witness, false);
        if (preserve_bound_prefix) {
          preserved_prefix_slot = prefix_slot;
          preserved_prefix_witness = witness;
        }
      }
    }
  }
  if (event.reserved == causal_rewrite::kEventFramePause)
    (void)prepare_dormant_continuation_before_pause(engine.state);

  causal_rewrite::consume_rewrite_event(engine, event, false,
                                        preserve_bound_prefix);
  if (event.reserved == causal_rewrite::kEventFrameEnd)
    (void)promote_retained_span_prefix_after_end(
        engine.state, preserved_prefix_slot,
        preserve_bound_prefix ? &preserved_prefix_witness : nullptr);
  if (update_receipt)
    causal_rewrite::refresh_receipt(engine.state);
}

}  // namespace substrate::bcc32::causal_rewrite::cross_contact

#undef BCC32_CROSS_CONTACT_HD
