#pragma once

#include "bcc32_resident_cross_contact_context.cuh"
#include "bcc32_resident_causal_constraint_participation_end.cuh"
#include "bcc32_resident_mixed_provenance_evidence.cuh"
#include "bcc32_resident_open_inquiry.cuh"
#include "bcc32_resident_open_inquiry_return_gate.cuh"
#include "causal_rewrite_universe.cuh"

#if defined(__CUDACC__)
#define BCC32_CLOSE_WORK_DEVICE __device__ __noinline__
#else
#define BCC32_CLOSE_WORK_DEVICE
#endif

// Physical source closure is a resident transaction. Each phase is invoked
// at most once per device epoch, but the phase label is not itself a work
// bound: helpers on this path must also avoid extent x whole-ecology scan
// amplification. The phase number is scheduling state only; it names no
// concept, speaker, language, or fixed brain region.
namespace substrate::bcc32::resident_close_work {

namespace rewrite = substrate::bcc32::causal_rewrite;
namespace inquiry_return =
    substrate::bcc32::causal_rewrite::open_inquiry_return_gate;
namespace participation_end =
    substrate::bcc32::resident_causal_constraint_participation::physical_end;

// Keep every close phase behind its own device-call boundary. Merely placing
// helpers in switch cases does not bound CUDA stack: nvcc may inline all of
// their workspaces into advance(). These wrappers keep inquiry, provenance and
// Program-induction frames disjoint while preserving one resident transaction.
BCC32_CLOSE_WORK_DEVICE inline void capture_surface(
    rewrite::ResidentRewriteState* state) {
  (void)rewrite::open_inquiry::capture_teacher_surface_before_end(state);
  const bool reply_just_bound =
      inquiry_return::bind_unemitted_teacher_reply_before_end(state);
  // The canonical trajectory closer runs last in this phase, not in the
  // preceding kCloseWorkExternalRelation phase: capture_teacher_surface_
  // before_end's own contract ("called before ordinary END closes the
  // external teacher contact") requires the one live current trajectory to
  // still exist when it runs. External-relation assimilation
  // (advance_current_before_end, previous phase) only reads the trajectory
  // and does not retire it, so deferring retirement to here does not change
  // when assimilation itself completes -- it only lets both open-inquiry
  // "before_end" hooks in this same phase take first refusal on the still-
  // live trajectory before it is closed into ordinary Program/VersionSpace
  // matter. capture_teacher_surface_before_end's own success path already
  // calls clear_trajectory directly, so close_program_trajectory finds no
  // current trajectory and is a no-op in that case regardless of whether we
  // call it. bind_unemitted_teacher_reply_before_end's success path is
  // different: by contract it must leave the reply trajectory "current"
  // (lane[3]==0) so settle_bound_reply, two close-work phases later, can
  // still read it as the inquiry's bound reply Record -- it deliberately
  // does not clear it itself. close_program_trajectory has no awareness of
  // kInquiryReplyBound anywhere in its own logic, so calling it unconditionally
  // here would retire (or convert into ordinary Program/VersionSpace matter)
  // the reply trajectory this same call just bound, one call after binding
  // it and two full close-work phases before settle_bound_reply ever runs to
  // look for it. Skip the call precisely when bind_unemitted_teacher_reply_
  // before_end just admitted on this call: that is the one case where a
  // sibling hook in this same phase still needs the trajectory current, and
  // it is the only trajectory the skip protects -- any other current
  // trajectory on a future epoch, once bind has declined "already bound" or
  // has nothing to bind, is closed normally on the very next call to
  // capture_surface(). This extends capture_teacher_surface_before_end's own
  // self-protection pattern (clear directly, so close_program_trajectory is
  // a no-op) to bind_unemitted_teacher_reply_before_end, which cannot
  // self-protect the same way because its whole point is to leave the
  // trajectory readable afterward; it does not change close_program_
  // trajectory itself, and it does not touch promote_retained_span_prefix_
  // after_end, detach_active_carry_before_new_contact, or any carry/
  // retention path.
  if (!reply_just_bound) {
    rewrite::close_program_trajectory(state);
  }
}

BCC32_CLOSE_WORK_DEVICE inline void bind_reply(
    rewrite::ResidentRewriteState* state) {
  if (state == nullptr || state->open_inquiry_public_return_pending != 0u)
    return;
  (void)rewrite::open_inquiry::bind_one_fresh_external_reply_before_end(state);
}

BCC32_CLOSE_WORK_DEVICE inline bool settle_bound_reply(
    rewrite::ResidentRewriteState* state) {
  // Phase 2 in this same private close shadow has already established the
  // reply's external provenance, exact continuation/wrapper topology, live
  // selected alternative, and reply terms before writing the bound witness.
  // The close transaction fences unrelated mutation between these adjacent
  // phases.  Repeating extent-sensitive validators here would recreate the
  // reply-length x grown-capacity scan that stalled the passive close clock.
  //
  // Every combined-|| guard below is decomposed into sequential
  // if-return-false statements, each incrementing its own decline counter
  // immediately before returning -- the same technique
  // capture_teacher_surface_before_end's own combined guard used
  // (bcc32_resident_open_inquiry.cuh, 0X1-163/0X1-206 sibling session). An
  // `||` chain returns false on its first true clause; a sequential if-chain
  // in the same clause order returns false on that same first true clause.
  // The two forms are behaviorally identical for every input -- same
  // precondition set, same clause order, same short-circuit behavior, same
  // returned boolean bit-for-bit. Passive instrumentation only: nothing
  // reads these counters to select, gate, or advance settlement.
  if (state == nullptr || state->fault != 0u ||
      state->open_inquiry_public_return_pending != 0u)
    return false;
  ++state->oi_settle_reply_attempts;

  const std::uint32_t inquiry_slot =
      rewrite::open_inquiry::unique_active_inquiry(state);
  const std::uint32_t reply_slot =
      rewrite::open_inquiry::unique_current_trajectory(state);
  if (inquiry_slot == rewrite::kInvalid ||
      reply_slot == rewrite::kInvalid) {
    // Purely local, read-only reclassification of which side was kInvalid
    // and why (no matching candidate vs. more than one -- ambiguous),
    // mirroring capture_teacher_surface_before_end's own reclassification
    // verbatim over the same two helper functions. Does not alter the
    // decision above; only selects which counter to bump.
    if (inquiry_slot == rewrite::kInvalid) {
      std::uint32_t candidates = 0u;
      for (std::uint32_t slot = 0u;
           slot < rewrite::live_record_capacity(state); ++slot) {
        const auto& candidate = state->records[slot];
        if (candidate.matter_q8 == 0u ||
            candidate.lane[0] != rewrite::open_inquiry::kFormOpenInquiry ||
            (candidate.lane[7] &
             rewrite::open_inquiry::kInquiryAwaitingReply) == 0u ||
            (candidate.lane[7] &
             rewrite::open_inquiry::kInquirySettled) != 0u)
          continue;
        ++candidates;
      }
      if (candidates == 0u)
        ++state->oi_settle_decline_no_active_inquiry;
      else
        ++state->oi_settle_decline_ambiguous_inquiry;
    }
    if (reply_slot == rewrite::kInvalid) {
      std::uint32_t candidates = 0u;
      for (std::uint32_t slot = 0u;
           slot < rewrite::live_record_capacity(state); ++slot) {
        const auto& candidate = state->records[slot];
        if (candidate.matter_q8 == 0u ||
            candidate.lane[0] != rewrite::kFormTrajectory ||
            candidate.lane[3] != 0u)
          continue;
        ++candidates;
      }
      if (candidates == 0u)
        ++state->oi_settle_decline_no_current_trajectory;
      else
        ++state->oi_settle_decline_ambiguous_trajectory;
    }
    return false;
  }

  auto& inquiry = state->records[inquiry_slot];
  const auto& reply = state->records[reply_slot];

  // Thirteen-clause combined guard, decomposed into thirteen sequential
  // if-return-false statements in the same order (see the function-top
  // comment for the equivalence argument).
  if ((inquiry.lane[7] & rewrite::open_inquiry::kInquiryReplyBound) == 0u) {
    ++state->oi_settle_decline_not_reply_bound;
    return false;
  }
  if ((inquiry.lane[7] & rewrite::open_inquiry::kInquirySettled) != 0u) {
    ++state->oi_settle_decline_already_settled;
    return false;
  }
  if ((inquiry.lane[7] & rewrite::open_inquiry::kInquiryAwaitingReply) ==
      0u) {
    ++state->oi_settle_decline_not_awaiting_reply;
    return false;
  }
  if (inquiry.lane[5] == 0u) {
    ++state->oi_settle_decline_selected_owner_zero;
    return false;
  }
  if (inquiry.lane[5] == rewrite::kInvalid) {
    ++state->oi_settle_decline_selected_owner_invalid;
    return false;
  }
  if (inquiry.lane[6] == 0u) {
    ++state->oi_settle_decline_selected_revision_zero;
    return false;
  }
  if (inquiry.lane[6] == rewrite::kInvalid) {
    ++state->oi_settle_decline_selected_revision_invalid;
    return false;
  }
  if (reply.matter_q8 == 0u) {
    ++state->oi_settle_decline_reply_matter_zero;
    return false;
  }
  if (reply.lane[0] != rewrite::kFormTrajectory) {
    ++state->oi_settle_decline_reply_not_trajectory_form;
    return false;
  }
  if (reply.lane[1] == 0u) {
    ++state->oi_settle_decline_reply_owner_zero;
    return false;
  }
  if (reply.lane[1] == rewrite::kInvalid) {
    ++state->oi_settle_decline_reply_owner_invalid;
    return false;
  }
  if (reply.lane[2] == 0u) {
    ++state->oi_settle_decline_reply_length_zero;
    return false;
  }
  if (reply.lane[3] != 0u) {
    ++state->oi_settle_decline_reply_not_current;
    return false;
  }

  const std::uint32_t suspended_slot =
      rewrite::cross_contact::find_trajectory_by_owner(
          state, inquiry.lane[2]);
  if (suspended_slot == rewrite::kInvalid) {
    ++state->oi_settle_decline_no_suspended_trajectory;
    return false;
  }

  const auto& suspended = state->records[suspended_slot];
  // Three-clause combined guard, decomposed the same way.
  if (suspended.lane[3] == 0u) {
    ++state->oi_settle_decline_suspended_lane3_zero;
    return false;
  }
  if (suspended.lane[2] != inquiry.lane[3]) {
    ++state->oi_settle_decline_suspended_owner_mismatch;
    return false;
  }
  if ((suspended.lane[7] & rewrite::kTrajectoryOpenInquiry) == 0u) {
    ++state->oi_settle_decline_suspended_not_open_inquiry;
    return false;
  }

  std::uint32_t alternative_count = 0u;
  std::uint32_t selected_binding_count = 0u;
  std::uint32_t selected_consequence = rewrite::kInvalid;

  std::uint32_t witness_count = 0u;
  std::uint32_t witness_consequence = rewrite::kInvalid;
  std::uint32_t emission_count = 0u;
  std::uint32_t selected_emission_count = 0u;
  const std::uint32_t capacity = rewrite::live_record_capacity(state);

  for (std::uint32_t slot = 0u; slot < capacity; ++slot) {
    const auto& record = state->records[slot];
    if (record.matter_q8 == 0u) continue;

    if (record.lane[0] ==
            rewrite::open_inquiry::kFormOpenInquiryAlternative &&
        record.lane[1] == inquiry.lane[1]) {
      ++alternative_count;
      if (record.lane[2] == inquiry.lane[5] &&
          record.lane[3] == inquiry.lane[6]) {
        ++selected_binding_count;
        selected_consequence = record.lane[4];
      }
      continue;
    }

    if (record.lane[0] ==
            rewrite::open_inquiry::kFormOpenInquiryReplyWitness &&
        record.lane[1] == inquiry.lane[1]) {
      // Eight-clause combined guard (previously one `if` with eight
      // `||`-joined clauses), decomposed the same way. It still returns
      // false out of the loop on the same first-failing clause as before.
      if (record.lane[2] != reply.lane[1]) {
        ++state->oi_settle_decline_witness_owner_mismatch;
        return false;
      }
      if (record.lane[3] != inquiry.lane[5]) {
        ++state->oi_settle_decline_witness_selected_owner_mismatch;
        return false;
      }
      if (record.lane[4] != inquiry.lane[6]) {
        ++state->oi_settle_decline_witness_selected_revision_mismatch;
        return false;
      }
      if (record.lane[6] != reply.revision) {
        ++state->oi_settle_decline_witness_reply_revision_mismatch;
        return false;
      }
      if (record.reserved[0] != reply.lane[2]) {
        ++state->oi_settle_decline_witness_reply_length_mismatch;
        return false;
      }
      if (record.reserved[1] != reply.lane[6]) {
        ++state->oi_settle_decline_witness_reply_tail_mismatch;
        return false;
      }
      if (record.lane[7] != rewrite::open_inquiry::kInquiryExternalWitness) {
        ++state->oi_settle_decline_witness_not_external;
        return false;
      }
      if (record.revision != 1u) {
        ++state->oi_settle_decline_witness_revision_not_one;
        return false;
      }
      ++witness_count;
      witness_consequence = record.lane[5];
      continue;
    }

    if (record.lane[0] ==
            rewrite::open_inquiry::kFormOpenInquiryEmission &&
        record.lane[1] == inquiry.lane[1]) {
      ++emission_count;
      if (record.lane[4] == inquiry.lane[5] &&
          record.lane[5] == inquiry.lane[6] &&
          record.lane[6] ==
              rewrite::open_inquiry::kInquiryEmissionReplyContinuation)
        ++selected_emission_count;
    }
  }

  // Seven-clause combined post-loop consensus guard, decomposed the same
  // way.
  if (alternative_count != 2u) {
    ++state->oi_settle_decline_alternative_count;
    return false;
  }
  if (selected_binding_count != 1u) {
    ++state->oi_settle_decline_selected_binding_count;
    return false;
  }
  if (selected_consequence == rewrite::kInvalid) {
    ++state->oi_settle_decline_selected_consequence_invalid;
    return false;
  }
  if (witness_count != 1u) {
    ++state->oi_settle_decline_witness_count;
    return false;
  }
  if (witness_consequence != selected_consequence) {
    ++state->oi_settle_decline_witness_consequence_mismatch;
    return false;
  }
  if (emission_count != 1u) {
    ++state->oi_settle_decline_emission_count;
    return false;
  }
  if (selected_emission_count != 1u) {
    ++state->oi_settle_decline_selected_emission_count;
    return false;
  }

  const std::uint32_t reply_owner = reply.lane[1];
  rewrite::mixed_provenance::clear_provenance(state, reply_owner);
  rewrite::clear_trajectory(state, reply_slot);
  inquiry.lane[7] =
      (inquiry.lane[7] & ~rewrite::open_inquiry::kInquiryAwaitingReply) |
      rewrite::open_inquiry::kInquirySettled;
  ++inquiry.revision;
  ++state->revision;
  rewrite::refresh_receipt(state);
  ++state->oi_settle_reply_admitted;
  return true;
}

BCC32_CLOSE_WORK_DEVICE inline void settle_inquiry(
    rewrite::ResidentRewriteState* state) {
  (void)settle_bound_reply(state);
}

BCC32_CLOSE_WORK_DEVICE inline void qualify_history(
    rewrite::ResidentRewriteState* state) {
  (void)rewrite::mixed_provenance::qualify_current_history(state);
}

BCC32_CLOSE_WORK_DEVICE inline void close_cross_contact(
    rewrite::ResidentRewriteState* state) {
  const rewrite::ResidentRewriteEngine engine(state);
  rewrite::cross_contact::consume_cross_contact_event(
      engine, rewrite::RawRewriteEvent{0u, 0u, rewrite::kEventFrameEnd},
      false);
}

BCC32_CLOSE_WORK_DEVICE inline void clear_provenance(
    rewrite::ResidentRewriteState* state) {
  rewrite::mixed_provenance::clear_orphaned_provenance(state);
}

BCC32_CLOSE_WORK_DEVICE inline void reactivate_inquiry(
    rewrite::ResidentRewriteState* state) {
  (void)rewrite::open_inquiry::reactivate_settled_suspended_after_end(state);
}

BCC32_CLOSE_WORK_DEVICE inline void advance_external_relation(
    rewrite::ResidentRewriteState* state) {
  // A rejected external-observation attempt is an honest category receipt,
  // not a close-transaction fault. Mixed/generated history remains available
  // to its existing provenance path while ordinary external sources proceed
  // incrementally here.
  // Production close work does not publish the StageReceipt returned by this
  // adapter. Keep its exact default observer measurement for focused callers,
  // but do not spend the resident clock on the discarded capacity-cubed
  // factor recount while the adult is waiting for the next contact.
  (void)participation_end::advance_current_before_end(
      state, participation_end::kExternalRelationWorkPerEpoch, false);
  // Relation assimilation must finish while the external trajectory is still
  // intact -- this phase only reads the trajectory via advance_current_
  // before_end, it never retires it. The canonical trajectory closer itself
  // (close_program_trajectory) deliberately does NOT run here anymore: it
  // used to run in this same phase step the moment the bounded source cursor
  // completed, which retired the one live current trajectory a full close-
  // work phase before kCloseWorkCaptureSurface's capture_teacher_surface_
  // before_end ever got to look at it (the phase machine advances at most
  // one phase per epoch, so that was a real one-epoch ordering inversion,
  // not merely a same-transaction detail) -- capture_teacher_surface_
  // before_end's own contract requires running "before ordinary END closes
  // the external teacher contact", so close_program_trajectory now runs at
  // the end of capture_surface (kCloseWorkCaptureSurface), after both
  // open-inquiry "before_end" hooks have had first refusal on the still-live
  // trajectory. See capture_surface() for the call.
}

struct AdvanceReceipt {
  std::uint32_t phase_before = rewrite::kCloseWorkIdle;
  std::uint32_t phase_after = rewrite::kCloseWorkIdle;
  std::uint32_t progressed = 0u;
  std::uint32_t completed = 0u;
  std::uint32_t rejected = 0u;
  std::uint32_t fault = 0u;
};

// Advance at most one bounded close phase on `state`. The caller owns
// transaction semantics (private shadow, commit-on-complete, discard-on-
// reject); this function only ever mutates the state it is given.
BCC32_CLOSE_WORK_DEVICE inline AdvanceReceipt advance(
    rewrite::ResidentRewriteState* state) {
  AdvanceReceipt receipt{};
  if (state == nullptr) {
    receipt.rejected = 1u;
    return receipt;
  }
  receipt.phase_before = state->close_work_phase;
  receipt.phase_after = state->close_work_phase;
  receipt.fault = state->fault;
  if (state->close_work_pending == 0u) return receipt;
  if (state->fault != 0u) {
    receipt.rejected = 1u;
    return receipt;
  }

  switch (state->close_work_phase) {
    case rewrite::kCloseWorkExternalRelation:
      advance_external_relation(state);
      state->close_work_phase = rewrite::kCloseWorkCaptureSurface;
      if (state->external_relation_stage_active != 0u)
        state->close_work_phase = rewrite::kCloseWorkExternalRelation;
      break;
    case rewrite::kCloseWorkCaptureSurface:
      capture_surface(state);
      state->close_work_phase = rewrite::kCloseWorkBindReply;
      break;
    case rewrite::kCloseWorkBindReply:
      bind_reply(state);
      state->close_work_phase = rewrite::kCloseWorkSettleInquiry;
      break;
    case rewrite::kCloseWorkSettleInquiry:
      settle_inquiry(state);
      state->close_work_phase = rewrite::kCloseWorkQualifyHistory;
      break;
    case rewrite::kCloseWorkQualifyHistory:
      qualify_history(state);
      state->close_work_phase = rewrite::kCloseWorkCrossContact;
      break;
    case rewrite::kCloseWorkCrossContact:
      close_cross_contact(state);
      state->close_work_phase = rewrite::kCloseWorkClearProvenance;
      break;
    case rewrite::kCloseWorkClearProvenance:
      clear_provenance(state);
      state->close_work_phase = rewrite::kCloseWorkReactivateInquiry;
      break;
    case rewrite::kCloseWorkReactivateInquiry:
      reactivate_inquiry(state);
      state->close_work_phase = rewrite::kCloseWorkComplete;
      break;
    case rewrite::kCloseWorkComplete:
      state->close_work_pending = 0u;
      state->close_work_phase = rewrite::kCloseWorkIdle;
      receipt.progressed = 1u;
      receipt.completed = 1u;
      receipt.phase_after = state->close_work_phase;
      receipt.fault = state->fault;
      return receipt;
    default:
      state->fault = rewrite::kCloseWorkInvalidPhaseFault;
      state->close_work_pending = 0u;
      state->close_work_phase = rewrite::kCloseWorkIdle;
      state->cross_context_factor_pending = 0u;
      receipt.rejected = 1u;
      receipt.fault = state->fault;
      return receipt;
  }

  receipt.progressed = 1u;
  receipt.phase_after = state->close_work_phase;
  receipt.fault = state->fault;
  if (state->fault != 0u) receipt.rejected = 1u;
  return receipt;
}

}  // namespace substrate::bcc32::resident_close_work

#undef BCC32_CLOSE_WORK_DEVICE
