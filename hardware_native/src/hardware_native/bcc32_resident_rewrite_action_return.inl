// Consume one transport chunk directly into the private staging adult. No
// transcript-sized scratch buffer exists; only passive constraint counters are
// held until the terminal transaction commits.
__device__ __noinline__ void apply_action_return_chunk(
    DeviceState* state, const BoundaryWord* words, std::uint32_t count) {
  rewrite::ResidentRewriteEngine engine(&state->action_return_staging_world);
  for (std::uint32_t index = 0u; index < count; ++index) {
    const BoundaryWord word = words[index];
    if (word == rewrite::kBoundaryPause) {
      (void)rewrite::cross_contact::prepare_dormant_continuation_before_pause(
          &state->action_return_staging_world);
      (void)rewrite::mixed_provenance::consume_external_event(
          engine, rewrite::RawRewriteEvent{0u, 0u, rewrite::kEventFramePause}, false);
      (void)rewrite::open_inquiry::open_from_yielded_version_space(
          &state->action_return_staging_world);
      continue;
    }
    if (word == rewrite::kBoundaryEnd) {
      const constraint_reafferent::Receipt constraint =
          constraint_reafferent::assimilate_accepted_staged_trajectory(
              &state->action_return_staging_world);
      // Provider/cursor identity is diagnostic provenance only. A returned
      // consequence may revise resident matter only after the reafferent
      // adapter rederives the distributed closure from its live component
      // cut; no Program locus, cursor, or host ticket can authorize it.
      const bool accepted = constraint.admission_authorized();
      ActionReturnConstraintDelta& delta =
          state->action_return_stream_constraint_delta;
      ++delta.attempted;
      delta.accepted += accepted ? 1u : 0u;
      delta.rejected += accepted ? 0u : 1u;
      delta.countered_records += constraint.countered_records;
      delta.admitted_records += constraint.admitted_records;
      delta.resident_revision = constraint.resident_revision;
      delta.component_ready = constraint.component_ready;
      delta.component_ambiguous = constraint.component_ambiguous;
      delta.component_records = constraint.component_records;
      delta.component_sources = constraint.component_sources;
      delta.rederived_event = constraint.rederived_event;
      delta.source_revision = constraint.source_revision;
      if (accepted) {
        const std::uint32_t trajectory_slot =
            rewrite::find_current_trajectory(
                &state->action_return_staging_world);
        if (trajectory_slot != rewrite::kInvalid)
          state->action_return_stream_distributed_trajectory_owner =
              state->action_return_staging_world.records[trajectory_slot].lane[1];
      }
      if (!accepted && state->action_return_stream_words > 1u &&
          !rewrite::mark_current_accepted_return_trajectory(
              &state->action_return_staging_world))
        // The ticketed witness is only a compatibility donor when distributed
        // admission did not already convert the returned contact into live
        // resident participation.
        state->action_return_staging_world.fault = 0x52572d30u;
      (void)rewrite::open_inquiry::capture_teacher_surface_before_end(
          &state->action_return_staging_world);
      if (state->action_return_stream_inquiry_reply_required != 0u)
        state->action_return_stream_inquiry_reply_settled =
            inquiry_return::settle_ticketed_reply_before_end(
                &state->action_return_staging_world,
                inquiry_return::bind_completed_public_inquiry(
                    &state->world, &state->egress_history,
                    state->action_return_stream_ticket))
                ? 1u
                : 0u;
      (void)rewrite::mixed_provenance::qualify_current_history(
          &state->action_return_staging_world);
      rewrite::cross_contact::consume_cross_contact_event(
          engine, rewrite::RawRewriteEvent{0u, 0u, rewrite::kEventFrameEnd}, false);
      rewrite::mixed_provenance::clear_orphaned_provenance(
          &state->action_return_staging_world);
      (void)rewrite::open_inquiry::reactivate_settled_suspended_after_end(
          &state->action_return_staging_world);
      state->action_return_stream_saw_physical_end = 1u;
      continue;
    }
    // A validated action return is not an unrelated new contact. It is the
    // external consequence of the resident event already present in this
    // private copy of the continuing adult. Keep that chronology on one
    // resident trajectory until physical END so the reafferent adapter can
    // rederive the generated boundary and its later external consequence.
    // Ticket/route validation above controls entry into this transaction;
    // neither transport metadata nor this continuity authorizes revision.
    (void)rewrite::mixed_provenance::consume_external_event(
        engine, rewrite::RawRewriteEvent{word, 1u, rewrite::kEventFrameNone}, false);
    // Settle against the transaction-private copy. Only a transaction that
    // reaches finalize_action_return_commit's accept path publishes this back
    // over the live matter; a faulted or rejected return discards it.
    resident_predictive_shadow_assay::observe_contact(
        state->action_return_staging_predictive_shadow,
        static_cast<std::uint32_t>(word),
        resident_predictive_shadow_assay::ContactOrigin::external);
  }
}

// PAUSE relation disposition: 0 opens inquiry, 1 leaves incomplete relation
// evidence yielded for an ordinary producer, and 2 suppresses inquiry because
// the query-linked relation ecology is either ready/unique or genuinely
// ambiguous. Ambiguity remains fail-closed; it must never fall through into a
// different authority merely because the relation reader abstained.
// Keeping this probe here leaves the
// oversized runtime parent as a thin ingress composition point.
__device__ __noinline__ std::uint32_t
pause_relation_disposition(DeviceState* state) {
  if (state == nullptr) return 0u;
  const std::uint32_t current =
      rewrite::find_current_trajectory(&state->world);
  if (current == rewrite::kInvalid) return 0u;
  const rewrite::Record& trajectory = state->world.records[current];
  // The PAUSE gate must inspect the same complete relation candidate as the
  // autonomous generator. A distributed continuation is wider than the
  // two-word bootstrap aperture: its resident trajectory contains the
  // original cue/relation, the generated bridge, and the next raw connective.
  // Looking only at word 1 misclassifies that live context as unresolved and
  // restarts the external suffix before the canonical reader can emit it.
  const rewrite::CausalRelationCandidate candidate =
      rewrite::collect_causal_relation_candidate(&state->world, trajectory);
  if (candidate.ambiguous && candidate.query_linked) return 2u;
  if (candidate.ready && !candidate.ambiguous) return 2u;
  if (candidate.applicable && !candidate.ready && !candidate.ambiguous)
    return 1u;
  return 0u;
}
