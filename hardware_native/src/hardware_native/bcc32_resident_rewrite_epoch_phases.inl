// 0X1-175 Patch A: mechanical epoch-body split. Everything below is moved
// verbatim (or, where explicitly commented, restructured only to cross a new
// function boundary without changing any resident/receipt/egress-history
// outcome) from bcc32_resident_rewrite_runtime.cu and the former
// bcc32_resident_rewrite_epoch.inl. This file owns the device-side close,
// external-consumption, cognition, and public-commit phase logic; host
// PersistentKernel construction/destruction, the public API, and CUDA graph
// capture remain in the .cu. No new semantic state, causal clock, or
// ordering is introduced here -- see the two documented, narrowly-scoped
// exceptions inline below (both confined to the non-causal `__nanosleep`
// pacing hint and the publish-cadence heartbeat, never to resident state,
// receipts, or egress history).

inline constexpr std::uint32_t kCloseOriginNone = 0u;
inline constexpr std::uint32_t kCloseOriginIngress = 1u;
inline constexpr std::uint32_t kCloseOriginActionReturn = 2u;
inline constexpr std::uint32_t kCloseSettlementIdle = 0u;
inline constexpr std::uint32_t kCloseSettlementGermline = 1u;
inline constexpr std::uint32_t kCloseSettlementRevision = 2u;
inline constexpr std::uint32_t kCloseSettlementPending = 3u;
inline constexpr std::uint32_t kCloseSettlementCrossContext = 4u;
inline constexpr std::uint32_t kCloseSettlementCommit = 5u;

__device__ __noinline__ bool run_causal_germline_pending_device(
    rewrite::ResidentRewriteState* state);
__device__ __noinline__ bool run_revision_transfer_device(
    rewrite::ResidentRewriteState* state);
__device__ __noinline__ bool run_pending_resume_device(
    rewrite::ResidentRewriteState* state);
__device__ __noinline__ bool run_cross_context_settlement_device(
    rewrite::ResidentRewriteState* state);

// Open one private close transaction from an exact copy of `source`. The
// shadow is complete executor workspace: only the close transaction and its
// private settlement phases may mutate it, and nothing may treat it as
// resident authority until it commits. `source` itself is left untouched.
__device__ __noinline__ bool begin_close_transaction(
    DeviceState* state, const rewrite::ResidentRewriteState* source,
    std::uint32_t origin) {
  if (state == nullptr || source == nullptr || source->fault != 0u ||
      state->close_work_active != 0u || origin == kCloseOriginNone)
    return false;
  if (!clone_rewrite_state(&state->close_work_staging_world, source)) {
    state->close_work_fault = state->close_work_staging_world.fault != 0u
                                  ? state->close_work_staging_world.fault
                                  : rewrite::kCloseWorkTransactionFault;
    return false;
  }
  if (!rewrite::schedule_physical_end(&state->close_work_staging_world)) {
    state->close_work_fault = state->close_work_staging_world.fault != 0u
                                  ? state->close_work_staging_world.fault
                                  : rewrite::kCloseWorkTransactionFault;
    return false;
  }
  if (origin == kCloseOriginIngress) {
    // Observer-only latest-END lifecycle receipt. Reset when the private
    // ingress close transaction opens; no resident path reads these fields.
    state->rewrite_participation_end_materialized_records = 0u;
    state->rewrite_participation_end_precommit_records = 0u;
    state->rewrite_participation_end_committed_records = 0u;
  }
  state->close_work_origin = origin;
  state->close_work_settlement_phase = kCloseSettlementIdle;
  state->close_work_active = 1u;
  return true;
}

__device__ __noinline__ bool fail_close_transaction(DeviceState* state) {
  if (state == nullptr) return false;
  if (state->close_work_fault == 0u)
    state->close_work_fault = state->close_work_staging_world.fault != 0u
                                  ? state->close_work_staging_world.fault
                                  : rewrite::kCloseWorkTransactionFault;
  release_owned_rewrite_pages(&state->close_work_staging_world);
  state->close_work_settlement_phase = kCloseSettlementIdle;
  state->close_work_active = 0u;
  state->close_work_origin = kCloseOriginNone;
  return false;
}

// Advance the active close transaction by at most one bounded phase. Returns
// true whenever the private shadow made forward progress. Canonical commit is
// a separate bounded settlement phase below.
__device__ __noinline__ bool advance_close_transaction(DeviceState* state) {
  if (state == nullptr || state->close_work_active == 0u) return false;
  const resident_close_work::AdvanceReceipt step =
      resident_close_work::advance(&state->close_work_staging_world);
  if (state->close_work_origin == kCloseOriginIngress &&
      step.rejected == 0u &&
      step.phase_before == rewrite::kCloseWorkExternalRelation &&
      step.phase_after != rewrite::kCloseWorkExternalRelation) {
    // The resumable participation-END stage has finished, but no later
    // capture/bind/settle/qualify/cross-context phase has run yet.
    state->rewrite_participation_end_materialized_records =
        substrate::bcc32::resident_causal_constraint_participation::
            participation_count(&state->close_work_staging_world);
  }
  if (step.rejected != 0u || state->close_work_staging_world.fault != 0u) {
    state->close_work_fault = state->close_work_staging_world.fault != 0u
                                  ? state->close_work_staging_world.fault
                                  : rewrite::kCloseWorkTransactionFault;
    return fail_close_transaction(state);
  }
  if (step.completed == 0u) return step.progressed != 0u;

  // The close phases only establish the private successor image. Dependent
  // germline, revision, pending-means, and cross-context work is deliberately
  // advanced by one helper per later device epoch before this image becomes
  // canonical. This preserves the original failure-atomic boundary while
  // preventing four extent-sensitive helpers from sharing one epoch frame.
  state->close_work_settlement_phase = kCloseSettlementGermline;
  return true;
}

__device__ __noinline__ bool commit_close_transaction(
    DeviceState* state, bool* accepted_action_return,
    bool* device_body_return) {
  if (state == nullptr || state->close_work_active == 0u ||
      state->close_work_settlement_phase != kCloseSettlementCommit)
    return false;
  const std::uint32_t completed_origin = state->close_work_origin;
  if (completed_origin == kCloseOriginIngress) {
    // Last observation of the completely settled private successor before
    // ownership moves to the canonical world.
    state->rewrite_participation_end_precommit_records =
        substrate::bcc32::resident_causal_constraint_participation::
            participation_count(&state->close_work_staging_world);
  }
  const std::uint64_t staged_relation_admitted =
      state->close_work_staging_world.external_relation_stage_admitted;
  const std::uint64_t staged_relation_rejected =
      state->close_work_staging_world.external_relation_stage_rejected;
  if (!move_rewrite_state(&state->world,
                          &state->close_work_staging_world)) {
    return fail_close_transaction(state);
  }
  if (completed_origin == kCloseOriginIngress) {
    // First observation of the state that became canonical. If this differs
    // from precommit, the ownership move itself is implicated.
    state->rewrite_participation_end_committed_records =
        substrate::bcc32::resident_causal_constraint_participation::
            participation_count(&state->world);
    state->rewrite_participation_end_admitted += staged_relation_admitted;
    state->rewrite_participation_end_rejected += staged_relation_rejected;
  }
  if (state->close_work_origin == kCloseOriginActionReturn) {
    state->action_return_contact = state->close_work_return_contact;
    state->action_return_contact_words = state->close_work_return_words;
    state->action_return_contact_sequence = state->close_work_return_sequence;
    state->action_return_last_action_sequence =
        state->close_work_action_ticket.action_sequence;
    state->action_return_ticket = ActionReturnTicket{};
    if (state->action_return_accepted != ~std::uint64_t{0})
      ++state->action_return_accepted;
    if (accepted_action_return != nullptr) *accepted_action_return = true;
    state->world.open_inquiry_public_return_receipt =
        state->close_work_device_body_return == 0u ? 1u : 0u;
    if (device_body_return != nullptr)
      *device_body_return = state->close_work_device_body_return != 0u;
  }
  state->close_work_action_ticket = ActionReturnTicket{};
  state->close_work_return_contact = ContentAddress{};
  state->close_work_return_words = 0u;
  state->close_work_return_sequence = 0u;
  state->close_work_device_body_return = 0u;
  state->close_work_settlement_phase = kCloseSettlementIdle;
  state->close_work_origin = kCloseOriginNone;
  state->close_work_active = 0u;
  return true;
}

__device__ __noinline__ bool advance_close_settlement(
    DeviceState* state, bool* committed, bool* settlement_generated,
    bool* accepted_action_return, bool* device_body_return) {
  if (committed != nullptr) *committed = false;
  if (settlement_generated != nullptr) *settlement_generated = false;
  if (state == nullptr || state->close_work_active == 0u ||
      state->close_work_settlement_phase == kCloseSettlementIdle)
    return false;
  if (state->close_work_staging_world.fault != 0u)
    return fail_close_transaction(state);

  switch (state->close_work_settlement_phase) {
    case kCloseSettlementGermline:
      (void)run_causal_germline_pending_device(
          &state->close_work_staging_world);
      state->close_work_settlement_phase = kCloseSettlementRevision;
      break;
    case kCloseSettlementRevision:
      (void)run_revision_transfer_device(&state->close_work_staging_world);
      state->close_work_settlement_phase = kCloseSettlementPending;
      break;
    case kCloseSettlementPending:
      if (settlement_generated != nullptr)
        *settlement_generated = run_pending_resume_device(
            &state->close_work_staging_world);
      else
        (void)run_pending_resume_device(&state->close_work_staging_world);
      state->close_work_settlement_phase = kCloseSettlementCrossContext;
      break;
    case kCloseSettlementCrossContext:
      (void)run_cross_context_settlement_device(
          &state->close_work_staging_world);
      state->close_work_settlement_phase = kCloseSettlementCommit;
      break;
    case kCloseSettlementCommit:
      if (!commit_close_transaction(state, accepted_action_return,
                                    device_body_return))
        return false;
      if (committed != nullptr) *committed = true;
      return true;
    default:
      state->close_work_fault = rewrite::kCloseWorkInvalidPhaseFault;
      return fail_close_transaction(state);
  }
  if (state->close_work_staging_world.fault != 0u)
    return fail_close_transaction(state);
  return true;
}

__device__ __noinline__ bool consume_ingress(DeviceState* state,
                                             IngressRing* ingress,
                                             bool* ended_at_contact) {
  // A physical END has already entered the adult.  Its resident close is
  // ordered work, not permission to admit a later contact into a half-closed
  // ecology.  Returning here leaves the passive clock and observers running
  // while the close transaction advances one bounded phase per epoch.
  if (state == nullptr || state->close_work_active != 0u)
    return false;
  cuda::atomic_ref<std::uint64_t, cuda::thread_scope_system> published(ingress->published);
  cuda::atomic_ref<std::uint64_t, cuda::thread_scope_system> consumed(ingress->consumed);
  const std::uint64_t available = published.load(cuda::memory_order_acquire);
  const std::uint64_t next = consumed.load(cuda::memory_order_relaxed) + 1u;
  if (available < next)
    return false;
  IngressSlot& slot = ingress->slots[next % kIngressSlots];
  cuda::atomic_ref<std::uint64_t, cuda::thread_scope_system> sequence(slot.sequence);
  if (sequence.load(cuda::memory_order_acquire) != next)
    return false;

  // A later public raw contact supersedes any unanswered action before the
  // first packet of that contact enters resident matter. Continuation packets
  // from the same present_raw call are transport-only and cannot invalidate a
  // ticket independently. This changes transport authority only; it does not
  // freeze resident work, erase the egress event, or inject semantic state.
  if (slot.offset == 0u && slot.contact_start != 0u) {
    const bool public_inquiry_return_pending =
        state->world.open_inquiry_public_return_pending != 0u;
    if (!public_inquiry_return_pending) {
      state->action_return_ticket = ActionReturnTicket{};
      state->action_return_autonomy_barrier = 0u;
    }
    // A partial ticketed return is private staging matter. Supersession
    // abandons that uncommitted transaction as well; no staged bytes become
    // resident authority and no later packet can inherit its sequence.
    reset_action_return_stream(state);
  }

  rewrite::ResidentRewriteEngine engine(&state->world);
  bool saw_physical_end = false;
  if (slot.count == 0u) {
    (void)rewrite::mixed_provenance::consume_external_event(engine,
                                                            rewrite::RawRewriteEvent{0u, 0u, 0u});
  } else {
    for (std::uint32_t admitted = 0u;
         admitted < kIngressWordsPerEpoch && slot.offset < slot.count;
         ++admitted) {
      const BoundaryWord word = slot.words[slot.offset++];
      if (word == rewrite::kBoundaryPause) {
        // A yielded two-word query may already be explained by a distributed
        // relation closure. Re-derive the cut-closed, unique closure from a
        // disposable query view before suppressing the older
        // dormant/OpenInquiry constructor. Mere applicability is resident
        // evidence, not authority to monopolize PAUSE scheduling. An
        // applicable but incomplete relation must also leave the yielded
        // query available to the ordinary resident producer: its own later
        // returned consequence is what can grow the missing source. Only a
        // ready closure or a genuinely ambiguous one earns the inquiry
        // boundary below.
        const std::uint32_t pause_disposition =
            pause_relation_disposition(state);
         const bool distributed_relation_pause = pause_disposition == 2u;
        if (!distributed_relation_pause && pause_disposition == 0u)
          (void)rewrite::restart_external_suffix_after_unresolved_context(
              &state->world);
        if (!distributed_relation_pause)
          (void)rewrite::cross_contact::
              prepare_dormant_continuation_before_pause(&state->world);
        (void)rewrite::mixed_provenance::consume_external_event(
            engine, rewrite::RawRewriteEvent{0u, 0u, rewrite::kEventFramePause}, false);
        if (!distributed_relation_pause && pause_disposition == 0u)
          (void)rewrite::open_inquiry::open_from_yielded_version_space(
              &state->world);
      } else if (word == rewrite::kBoundaryEnd) {
        // Must run before capture_teacher_surface_before_end, which can
        // consume/clear the current external trajectory. This is the only
        // production seam for the distributed constraint-participation
        // mechanism to ever develop from ordinary raw external contact.
        {
          const participation_end::StageReceipt participation_stage =
              participation_end::stage_current_before_end(&state->world);
          state->rewrite_participation_end_attempted +=
              participation_stage.attempted;
          state->rewrite_participation_end_admitted +=
              participation_stage.admitted;
          state->rewrite_participation_end_rejected +=
              participation_stage.rejected;
        }
        // END starts a private close transaction from the exact canonical
        // prefix. The END itself is not acknowledged to transport until every
        // close phase has settled and the shadow commits atomically. The
        // inquiry-return-gate bind runs inside the shadow's own capture
        // phase, so a rejected transaction never leaves that formation as
        // canonical.
        if (!begin_close_transaction(state, &state->world,
                                     kCloseOriginIngress)) {
          if (state->close_work_fault == 0u)
            state->close_work_fault = rewrite::kCloseWorkTransactionFault;
          break;
        }
        saw_physical_end = true;
        // A physical END is a causal boundary even when transport coalesces
        // several contacts into one slot. Resolve it before consuming a later
        // word from that same packet.
        break;
      } else {
        (void)rewrite::cross_contact::detach_active_carry_before_new_contact(
            &state->world);
        (void)rewrite::open_inquiry::detach_emitted_surface_before_external_reply(
            &state->world);
        (void)rewrite::mixed_provenance::consume_external_event(
            engine, rewrite::RawRewriteEvent{word, 1u, rewrite::kEventFrameNone}, false);
        // 0X1-267 requirement 1: ordinary accepted raw external contact.
        // BoundaryWord and the assay's SiteWord are both plain uint32_t; the
        // real ingress ticket/source lineage stays entirely in the ordinary
        // resident path above, this only lets the generic predictive-shadow
        // relation assay observe the same raw word.
        resident_predictive_shadow_assay::observe_contact(
            state->predictive_shadow, static_cast<std::uint32_t>(word),
            resident_predictive_shadow_assay::ContactOrigin::external);
      }
    }
  }
  if (ended_at_contact != nullptr) *ended_at_contact = saw_physical_end;
  if (state->close_work_active != 0u || slot.offset < slot.count)
    return true;
  state->contact_sequence = next;
  slot.offset = 0u;
  slot.contact_start = 0u;
  sequence.store(0u, cuda::memory_order_release);
  consumed.store(next, cuda::memory_order_release);
  return true;
}

__device__ bool ingress_pending(IngressRing* ingress) {
  cuda::atomic_ref<std::uint64_t, cuda::thread_scope_system> published(ingress->published);
  cuda::atomic_ref<std::uint64_t, cuda::thread_scope_system> consumed(ingress->consumed);
  const std::uint64_t next = consumed.load(cuda::memory_order_relaxed) + 1u;
  if (published.load(cuda::memory_order_acquire) < next)
    return false;

  IngressSlot& slot = ingress->slots[next % kIngressSlots];
  cuda::atomic_ref<std::uint64_t, cuda::thread_scope_system> sequence(slot.sequence);
  return sequence.load(cuda::memory_order_acquire) == next;
}

// An action-return chunk is a resident transaction, even before its ticket is
// validated. Do not let autonomous reorganization advance the public adult in
// the gap between host publication and the consumer epoch; otherwise a replay
// or forged-return control could observe an unrelated authority step.
__device__ bool action_return_ingress_pending(ActionReturnIngress* returned) {
  if (returned == nullptr) return false;
  cuda::atomic_ref<std::uint64_t, cuda::thread_scope_system> published(
      returned->published);
  cuda::atomic_ref<std::uint64_t, cuda::thread_scope_system> consumed(
      returned->consumed);
  const std::uint64_t next = consumed.load(cuda::memory_order_relaxed) + 1u;
  return published.load(cuda::memory_order_acquire) >= next;
}

__device__ void clear_action_return_ingress(ActionReturnIngress* returned) {
  returned->count = 0u;
  returned->final_chunk = 0u;
  returned->ticket = ActionReturnTicket{};
  returned->chunk_sequence = 0u;
  returned->producer_commitment = ContentAddress{};
  returned->device_body_produced = 0u;
}

__device__ bool same_action_return_ticket(const ActionReturnTicket& left,
                                          const ActionReturnTicket& right) {
  return left.issuer_instance == right.issuer_instance &&
         left.action_sequence == right.action_sequence && left.nonce == right.nonce;
}

__device__ void reject_action_return(DeviceState* state, ActionReturnIngress* returned,
                                     cuda::atomic_ref<std::uint64_t, cuda::thread_scope_system>* consumed,
                                     std::uint64_t sequence) {
  if (state->action_return_rejected != ~std::uint64_t{0})
    ++state->action_return_rejected;
  clear_action_return_ingress(returned);
  consumed->store(sequence, cuda::memory_order_release);
}

#include "bcc32_resident_rewrite_action_return.inl"

__device__ __noinline__ bool finalize_action_return_commit(
    DeviceState* state, std::uint64_t next, const ActionReturnTicket& ticket,
    const egress_history::Event& action, bool* accepted,
    bool* device_body_return, bool* ended_at_contact) {
  if (accepted != nullptr) *accepted = false;
  if (device_body_return != nullptr) *device_body_return = false;
  if (ended_at_contact != nullptr) *ended_at_contact = false;
  if (state == nullptr)
    return false;

  // END can leave cross-context factorization pending while the distributed
  // closure has already been admitted. Settle that ordered resident phase
  // before refreshing the transaction receipt, exactly where the former
  // scalar cleanup function did it.
  (void)rewrite::mixed_provenance::settle_cross_context_factor(
      &state->action_return_staging_world);
  rewrite::refresh_receipt(&state->action_return_staging_world);
  if (!stage_device_body_world_consequence(state))
    state->action_return_staging_world.fault =
        rewrite::kCloseWorkTransactionFault;
  if (state->action_return_staging_world.fault != 0u ||
      (state->action_return_stream_inquiry_reply_required != 0u &&
       state->action_return_stream_inquiry_reply_settled == 0u)) {
    if (state->action_return_rejected != ~std::uint64_t{0})
      ++state->action_return_rejected;
    if (state->close_work_fault == 0u &&
        state->action_return_staging_world.fault == 0u)
      state->close_work_fault = rewrite::kCloseWorkTransactionFault;
    reset_action_return_stream(state);
    return true;
  }

  if (!move_rewrite_state(&state->world,
                          &state->action_return_staging_world)) {
    state->close_work_fault = rewrite::kCloseWorkTransactionFault;
    reset_action_return_stream(state);
    return true;
  }
  // The staging END may have changed record metadata after its last internal
  // receipt refresh. Recompute canonical organization before publication.
  rewrite::refresh_receipt(&state->world);
  // The transaction is now accepted, so its predictive-shadow settlement is
  // ratified together with the world it was computed against. Every earlier
  // return path above leaves the live matter untouched (0X1-267 req 4).
  state->predictive_shadow = state->action_return_staging_predictive_shadow;
  state->action_return_contact = finish_action_return_digest(state);
  state->action_return_contact_words = state->action_return_stream_words;
  state->action_return_contact_sequence = next;
  state->action_return_last_action_sequence = ticket.action_sequence;
  const std::uint32_t mouth_index = rewrite::mouth_compartment_for_locus(
      action.producer_locus);
  const std::uint32_t renewal_matter =
      state->action_return_stream_words > rewrite::kRecordMatterQ8
          ? rewrite::kRecordMatterQ8
          : static_cast<std::uint32_t>(state->action_return_stream_words);
  (void)rewrite::renew_mouth_compartment_consequence(
      &state->world, &state->mouth, mouth_index, renewal_matter, state->tick);
  state->action_return_ticket = ActionReturnTicket{};
  state->action_return_autonomy_barrier = 1u;
  commit_action_return_constraint_delta(state);
  const bool saw_physical_end =
      state->action_return_stream_saw_physical_end != 0u;
  const bool from_device_body =
      state->action_return_stream_from_device_body != 0u;
  const std::uint32_t device_body_consequence_word =
      state->action_return_stream_device_body_consequence_word;
  if (from_device_body)
    state->action_return_device_body_consequence_word =
        device_body_consequence_word;
  // Keep an accepted external return visible after the staged transaction is
  // finalized. The next action-return attempt clears this sticky edge; an
  // internal/device-body consequence never authorizes public egress.
  state->world.open_inquiry_public_return_receipt =
      from_device_body ? 0u : 1u;
  reset_action_return_stream(state);
  ++state->action_return_accepted;
  if (accepted != nullptr) *accepted = true;
  if (device_body_return != nullptr) *device_body_return = from_device_body;
  if (ended_at_contact != nullptr) *ended_at_contact = saw_physical_end;
  return true;
}

__device__ __noinline__ bool consume_action_return(
    DeviceState* state, ActionReturnIngress* returned, DeviceBodyControl* body,
    bool* accepted, bool* device_body_return, bool* ended_at_contact,
    ResidentEpochCleanupScratch* epoch_cleanup) {
  if (accepted != nullptr) *accepted = false;
  if (device_body_return != nullptr) *device_body_return = false;
  // Preserve one ordered adult transaction: a returned body consequence must
  // not copy a staged world over a resident close phase that has not settled.
  if (state == nullptr || state->close_work_active != 0u)
    return false;
  cuda::atomic_ref<std::uint64_t, cuda::thread_scope_system> published(returned->published);
  cuda::atomic_ref<std::uint64_t, cuda::thread_scope_system> consumed(returned->consumed);
  const std::uint64_t next = consumed.load(cuda::memory_order_relaxed) + 1u;
  if (published.load(cuda::memory_order_acquire) < next)
    return false;

  // A new attempt supersedes the prior observable acceptance edge. A
  // rejected replay or forged packet therefore cannot inherit the previous
  // receipt, while a valid acceptance below re-arms it for passive readers.
  state->world.open_inquiry_public_return_receipt = 0u;

  const ActionReturnTicket ticket = returned->ticket;
  const std::uint32_t count = returned->count;
  const bool final_chunk = returned->final_chunk == 1u;
  const bool stream_active = state->action_return_stream_active == 1u;
  egress_history::Event action{};
  const bool body_required = device_body_enabled(body);
  const bool ticket_valid = ticket.issuer_instance != 0u &&
                            ticket.issuer_instance == state->action_return_instance_nonce &&
                            ticket.nonce != 0u &&
                            same_action_return_ticket(ticket, state->action_return_ticket);
  const bool action_valid =
      ticket_valid &&
      egress_history::lookup(&state->egress_history, ticket.action_sequence, &action) &&
      (((action.raw_word & rewrite::kRawChannelMask) >> 24u) == 1u ||
       inquiry_return::publication_self_consistent(
           &state->world,
           inquiry_return::bind_completed_public_inquiry(
               &state->world, &state->egress_history, ticket)));
  const bool stream_valid = stream_active
                                ? (same_action_return_ticket(ticket, state->action_return_stream_ticket) &&
                                   returned->chunk_sequence == state->action_return_stream_next_chunk)
                                : returned->chunk_sequence == 1u;
  const bool body_valid = !body_required ||
                          (!stream_active && final_chunk && returned->chunk_sequence == 1u &&
                           returned->device_body_produced == 1u &&
                           state->device_body_initialized != 0u &&
                           returned->producer_instance == state->device_body_producer_instance &&
                           returned->source_epoch == state->device_body_source_epoch &&
                           returned->route_sequence == state->device_body_last_route_sequence &&
                           same_device_content_address(
                               returned->producer_commitment, device_body_return_commitment(
                               returned->producer_instance, returned->source_epoch,
                               returned->route_sequence, ticket, action, count,
                               returned->words)));
  const std::uint64_t bytes = static_cast<std::uint64_t>(count) * sizeof(BoundaryWord);
  bool words_valid = count <= kMaxContactWords && count != 0u &&
                     state->action_return_stream_words <=
                         ~std::uint64_t{0} - static_cast<std::uint64_t>(count) &&
                     state->action_return_stream_bytes <=
                         ~std::uint64_t{0} - bytes &&
                     state->action_return_stream_next_chunk != ~std::uint64_t{0} &&
                     (!final_chunk || state->action_return_accepted != ~std::uint64_t{0});
  for (std::uint32_t index = 0u; words_valid && index < count; ++index) {
    if (returned->words[index] == rewrite::kBoundaryEnd &&
        (!final_chunk || index + 1u != count))
      words_valid = false;
  }
  if (words_valid && final_chunk && returned->words[count - 1u] != rewrite::kBoundaryEnd)
    words_valid = false;
  if (!ticket_valid || !stream_valid || !body_valid || !action_valid || !words_valid) {
    reject_action_return(state, returned, &consumed, next);
    if (stream_active)
      reset_action_return_stream(state);
    return true;
  }

  if (!stream_active) {
    state->action_return_stream_ticket = ticket;
    state->action_return_stream_next_chunk = 1u;
    state->action_return_stream_active = 1u;
    if (!clone_rewrite_state(&state->action_return_staging_world,
                             &state->world)) {
      state->close_work_fault = state->action_return_staging_world.fault != 0u
                                    ? state->action_return_staging_world.fault
                                    : rewrite::kCloseWorkTransactionFault;
      reset_action_return_stream(state);
      return true;
    }
    // Seed the transaction-private predictive-shadow copy from the live
    // matter, mirroring the staging-world clone above. Returned words settle
    // against this copy until the transaction commits (0X1-267 requirement 4).
    state->action_return_staging_predictive_shadow = state->predictive_shadow;
    state->action_return_stream_constraint_delta = ActionReturnConstraintDelta{};
    state->action_return_stream_saw_physical_end = 0u;
    state->action_return_stream_inquiry_reply_settled = 0u;
    const inquiry_return::Publication publication =
        inquiry_return::bind_completed_public_inquiry(
            &state->world, &state->egress_history, ticket);
    state->action_return_stream_inquiry_reply_required =
        inquiry_return::publication_self_consistent(&state->world, publication)
            ? 1u
            : 0u;
    begin_action_return_digest(state);
  }

  extend_action_return_digest(state, returned->words, count);
  ++state->action_return_stream_next_chunk;
  state->action_return_stream_from_device_body |=
      returned->device_body_produced == 1u ? 1u : 0u;
  if (returned->device_body_produced == 1u)
    state->action_return_stream_device_body_consequence_word = returned->words[0];
  clear_action_return_ingress(returned);
  consumed.store(next, cuda::memory_order_release);
  apply_action_return_chunk(state, returned->words, count);
  if (!final_chunk)
    return true;

  // The staging adult has already consumed every preceding chunk. Only this
  // terminal boundary may publish it or its transaction-local receipt.
  // A distributed reafferent closure is the canonical revision path. The
  // older ticketed Program conversion remains a compatibility donor only when
  // no distributed closure was admitted; it can never authorize a return
  // after the new closure has rejected it.
  if (state->action_return_stream_saw_physical_end != 0u &&
      state->action_return_stream_constraint_delta.accepted == 0u &&
      !rewrite::accept_ticketed_revision_source(
          &state->action_return_staging_world))
    state->action_return_staging_world.fault = 0x52572d31u;
  // Distributed admission may have let the ordinary END qualifier demote
  // this transaction's header to a retained exemplar. Retire by the exact
  // owner captured before qualification; rejected distributed closure keeps
  // the legacy compatibility donor untouched.
  if (state->action_return_stream_saw_physical_end != 0u &&
      state->action_return_stream_constraint_delta.accepted != 0u) {
    if (epoch_cleanup == nullptr ||
        epoch_cleanup->phase != kEpochCleanupIdle) {
      state->world.fault = rewrite::kCloseWorkTransactionFault;
      return true;
    }
    epoch_cleanup->epoch_generation = state->device_epochs;
    epoch_cleanup->packet_generation = ticket.action_sequence;
    epoch_cleanup->continuation_next = next;
    epoch_cleanup->continuation_ticket = ticket;
    epoch_cleanup->scanned_records = 0u;
    epoch_cleanup->committed_records = 0u;
    epoch_cleanup->fault = 0u;
    epoch_cleanup->receipt_bits = 0u;
    __threadfence();
    epoch_cleanup->phase = kEpochCleanupStaged;
    __threadfence();
    return true;
  }
  // The extracted finalizer is the single publication path for a staged
  // action-return transaction. Keep the resident receipt sticky here as well
  // as in the close-transaction path so passive egress observes the same
  // accepted external return after graph cleanup completes.
  return finalize_action_return_commit(state, next, ticket, action, accepted,
                                       device_body_return, ended_at_contact);
}

__device__ __noinline__ bool consume_physical(DeviceState* state,
                                              PhysicalIngress* physical) {
  if (state == nullptr || state->close_work_active != 0u) return false;
  cuda::atomic_ref<std::uint64_t, cuda::thread_scope_system> published(physical->published);
  cuda::atomic_ref<std::uint64_t, cuda::thread_scope_system> consumed(physical->consumed);
  const std::uint64_t sequence = published.load(cuda::memory_order_acquire);
  if (sequence == 0u || sequence <= consumed.load(cuda::memory_order_relaxed))
    return false;
  const RawPhysicalIntervention event = physical->event;
  const std::uint32_t radius = event.radius;
  const std::uint32_t count =
      radius > (rewrite::kRecordCapacity - 1u) / 2u ? rewrite::kRecordCapacity : radius * 2u + 1u;
  const std::uint32_t start =
      (event.center + rewrite::kRecordCapacity - (radius % rewrite::kRecordCapacity)) %
      rewrite::kRecordCapacity;
  std::uint32_t max_records = event.matter_q8 / rewrite::kRecordMatterQ8;
  if (max_records == 0u && event.matter_q8 != 0u)
    max_records = 1u;
  const std::uint32_t removed = rewrite::apply_physical_lesion(
      rewrite::ResidentRewriteEngine(&state->world), start, count, max_records);
  rewrite::withdraw_mouth_compartments_for_lesion(&state->mouth, start, count);
  state->intervention_sequence = sequence;
  state->receipt.intervention_sequence = sequence;
  state->receipt.structural_focus_cell = event.center;
  state->receipt.removed_matter_q8_sum = removed;
  state->receipt.intervention_active = removed != 0u ? 1u : 0u;
  consumed.store(sequence, cuda::memory_order_release);
  return true;
}

// Learned-cost search owns a bounded Record-sized label workspace. Keeping it
// as a device call boundary changes frame ownership only: the same resident
// search, authority checks, and generated consequence remain canonical.
__device__ __noinline__ bool run_learned_cost_search_device(
    rewrite::ResidentRewriteEngine engine) {
  return rewrite::learned_cost_search::advance_resident_learned_cost_search_once(
      engine);
}

// Keep the resident root's stack bounded by preserving the major learned
// language phases as device-call boundaries. These wrappers do not change
// selection, authority, or publication; they prevent every bounded workspace
// from being merged into one graph-launch frame.
__device__ __noinline__ bool run_open_inquiry_device(
    rewrite::ResidentRewriteEngine engine) {
  return rewrite::open_inquiry::advance_surface_once(engine);
}

__device__ __noinline__ bool run_episodic_completion_device(
    rewrite::ResidentRewriteEngine engine) {
  return rewrite::advance_resident_episodic_completion_once(engine);
}

__device__ __noinline__ bool run_mixed_provenance_device(
    rewrite::ResidentRewriteEngine engine) {
  return rewrite::mixed_provenance::advance_once(engine);
}

// The relation candidate reader owns several bounded scratch arrays for
// outcome/source reconciliation. Keep that transient observer workspace out
// of the persistent epoch frame; returning its value is a receipt, not a
// semantic cell or a second writer.
__device__ __noinline__ rewrite::CausalRelationCandidate
run_causal_relation_candidate_device(
    const rewrite::ResidentRewriteState* state,
    const rewrite::Record& trajectory) {
  return rewrite::collect_causal_relation_candidate(state, trajectory);
}

__device__ __noinline__ bool run_causal_germline_pending_device(
    rewrite::ResidentRewriteState* state) {
  return rewrite::settle_causal_germline_pending(state);
}

__device__ __noinline__ bool run_revision_transfer_device(
    rewrite::ResidentRewriteState* state) {
  return rewrite::settle_resident_revision_transfer(state);
}

__device__ __noinline__ bool run_revision_participation_reader_device(
    rewrite::ResidentRewriteEngine engine) {
  return rewrite::advance_resident_revision_participation_reader_once(engine);
}

__device__ __noinline__ bool run_cross_context_settlement_device(
    rewrite::ResidentRewriteState* state) {
  return rewrite::mixed_provenance::settle_cross_context_factor(state);
}

__device__ __noinline__ bool run_means_end_inversion_device(
    rewrite::ResidentRewriteEngine engine) {
  return rewrite::advance_resident_means_end_inversion_once(engine);
}

__device__ __noinline__ bool run_pending_resume_device(
    rewrite::ResidentRewriteState* state) {
  return rewrite::pending_means::resume_one_pending_at_physical_end(state);
}

__device__ __noinline__ bool run_pending_retain_device(
    rewrite::ResidentRewriteState* state) {
  return rewrite::pending_means::retain_current_if_unresolved(state);
}

__device__ __noinline__ bool run_observe_resumed_consequence_device(
    rewrite::ResidentRewriteState* state) {
  return rewrite::open_inquiry::observe_resumed_consequence(state);
}

__device__ __noinline__ bool run_settle_inquiry_constructor_device(
    rewrite::ResidentRewriteState* state) {
  return rewrite::open_inquiry::settle_constructor_from_complete_episodes(state);
}

__device__ __noinline__ bool run_mark_generated_device(
    rewrite::ResidentRewriteState* state, std::uint32_t locus) {
  return rewrite::mixed_provenance::mark_last(
      state, rewrite::mixed_provenance::Origin::generated, locus);
}

__device__ __noinline__ bool run_fail_generated_device(
    rewrite::ResidentRewriteState* state) {
  return rewrite::mixed_provenance::fail_closed(
      state, rewrite::mixed_provenance::kGeneratedStampFault);
}

__device__ __noinline__ bool run_append_generated_device(
    rewrite::ResidentRewriteState* state, BoundaryWord word) {
  return rewrite::append_trajectory_word(state, word, true);
}

__device__ __noinline__ bool run_tagged_history_device(
    const rewrite::ResidentRewriteState* state, std::uint32_t locus) {
  return locus != rewrite::kInvalid &&
         rewrite::mixed_provenance::tagged_history(
             state, state->records[locus]);
}

__device__ __noinline__ void run_refresh_history_digest_device(
    DeviceState* state) {
  refresh_egress_history_digest(state);
}

__device__ __noinline__ void run_update_projection_device(DeviceState* state) {
  update_output_projection(state);
}

__device__ __noinline__ void run_issue_action_ticket_device(DeviceState* state) {
  issue_action_return_ticket(state);
}

__device__ __noinline__ void run_fill_receipt_device(
    DeviceState* state, bool consumed_contact, bool accepted_action_return,
    bool device_body_return, bool device_body_attached,
    const DerivedOutput& derived) {
  fill_receipt(state, consumed_contact, accepted_action_return,
               device_body_return, device_body_attached, derived);
}

__device__ __noinline__ void run_publish_egress_device(
    DeviceState* state, const DerivedOutput& derived, bool generated,
    EgressState* egress) {
  publish_egress(state, derived, generated, egress);
}

// Do not aggregate-initialize ResidentRewriteState from the persistent root.
// Its Record ecology is deliberately large; `*state = ResidentRewriteState{}`
// materializes a full temporary on the root CUDA stack before copying it into
// resident matter. The sm_89 resource receipt measured that temporary as the
// 144760-byte root frame that made graph admission fail. Recreate the same
// default state field-wise, without a host-selected value or a second resident
// copy. This is initialization mechanics only; the canonical record law and
// receipt refresh remain in causal_rewrite_universe.cuh.
__device__ __noinline__ void initialize_resident_world_device(
    rewrite::ResidentRewriteState* state, std::uint32_t permutation) {
  if (state == nullptr) return;
  // `state` is allocated with cudaMalloc and cleared with cudaMemset before
  // this helper runs, so C++ default member initializers do not execute for
  // the resident object.  Page zero is inline, but its directory metadata is
  // not: leaving live_page_count at the memset value (zero) makes the first
  // allocation compute a zero capacity and trap in the captured epoch.  Keep
  // the device bootstrap byte-for-byte equivalent to ResidentRewriteState{}
  // for the page-directory fields without materializing a large temporary on
  // the CUDA stack.
  state->directory.live_page_count = 1u;
  for (std::uint32_t page = 0u;
       page + 1u < rewrite::kMaxResidentPages; ++page)
    state->directory.pages[page] = nullptr;
  for (std::uint32_t i = 0u; i < rewrite::kRecordCapacity; ++i) {
    rewrite::Record& record = state->records[i];
    for (std::uint32_t lane = 0u; lane < rewrite::kLaneCount; ++lane)
      record.lane[lane] = 0u;
    record.revision = 0u;
    record.matter_q8 = rewrite::kRecordMatterQ8;
    record.reserved[0] = 0u;
    record.reserved[1] = 0u;
    record.lane[0] = rewrite::kFormEmpty;
  }
  for (std::uint32_t i = 0u; i < 8u; ++i) {
    rewrite::Record& displaced = state->lesion.displaced[i];
    for (std::uint32_t lane = 0u; lane < rewrite::kLaneCount; ++lane)
      displaced.lane[lane] = 0u;
    displaced.revision = 0u;
    displaced.matter_q8 = rewrite::kRecordMatterQ8;
    displaced.reserved[0] = 0u;
    displaced.reserved[1] = 0u;
    displaced.lane[0] = rewrite::kFormEmpty;
    state->lesion.original_slot[i] = 0u;
  }
  state->lesion.count = 0u;
  state->lesion.removed_matter_q8 = 0u;
  state->revision = 0u;
  state->admitted_events = 0u;
  state->allocation_cursor = 0u;
  state->fault = 0u;
  state->cross_context_factor_pending = 0u;
  state->causal_germline_reflection_program = rewrite::kInvalid;
  state->causal_germline_reflection_source = rewrite::kInvalid;
  state->causal_germline_application_program = rewrite::kInvalid;
  state->causal_germline_construction_pending = 0u;
  state->causal_germline_validation_pending = 0u;
  state->close_work_pending = 0u;
  state->close_work_phase = rewrite::kCloseWorkIdle;
  state->close_induction_cursor = 0u;
  state->close_induction_fixed_identity = rewrite::kInvalid;
  state->close_induction_fixed_left = rewrite::kInvalid;
  state->close_induction_span_identity = rewrite::kInvalid;
  state->close_induction_span_left = rewrite::kInvalid;
  state->close_induction_conflict = 0u;
  state->raw_motor_value = 0u;
  state->raw_motor_valid = 0u;
  state->active_locus = rewrite::kInvalid;
  state->constructor_locus = rewrite::kInvalid;
  state->generated_word = 0u;
  state->generated_word_valid = 0u;
  state->generated_locus = rewrite::kInvalid;
  state->open_inquiry_public_return_pending = 0u;
  state->open_inquiry_public_return_receipt = 0u;
  state->concrete_descriptions = 0u;
  state->mature_descriptions = 0u;
  state->partial_matches = 0u;
  state->direct_fires = 0u;
  state->staged_fires = 0u;
  state->partials_aged_out = 0u;
  state->partials_retired_matched = 0u;
  state->partials_retired_unmatched = 0u;
  state->conflict_abstentions = 0u;
  state->constructor_rewrites = 0u;
  state->relearned_descriptions = 0u;
  state->program_rules = 0u;
  state->mature_program_rules = 0u;
  state->trajectory_records = 0u;
  state->retained_exemplars = 0u;
  state->program_generated_events = 0u;
  state->program_conflict_abstentions = 0u;
  state->rejected_unbound_variables = 0u;
  state->completed_inductions = 0u;
  state->inspected_records = 0u;
  state->span_program_rules = 0u;
  state->mature_span_program_rules = 0u;
  state->span_generated_events = 0u;
  state->span_conflict_abstentions = 0u;
  state->span_rejected_unbound_variables = 0u;
  state->span_ambiguous_abstentions = 0u;
  state->span_completed_inductions = 0u;
  state->version_space_factors = 0u;
  state->version_space_alternatives = 0u;
  state->mature_version_space_alternatives = 0u;
  state->version_space_witnesses = 0u;
  state->version_space_conflict_abstentions = 0u;
  state->causal_germline_episodes = 0u;
  state->causal_germline_constructors = 0u;
  state->causal_germline_applications = 0u;
  state->causal_germline_reconstructions = 0u;
  state->causal_germline_counterevidence = 0u;
  state->causal_germline_product_suppressions = 0u;
  state->causal_germline_constructor_suppressions = 0u;
  state->causal_germline_conflict_abstentions = 0u;
  state->causal_germline_constructor_locus = rewrite::kInvalid;
  state->causal_germline_product_locus = rewrite::kInvalid;
  state->causal_relation_trajectory_revision = 0u;
  state->causal_relation_trajectory_owner = rewrite::kInvalid;
  state->causal_relation_generated_index = rewrite::kInvalid;
  state->organization_digest = 0u;

  const std::uint32_t constructor =
      rewrite::rewrite_mix(0x13u, permutation, 0x71u) %
      rewrite::kRecordCapacity;
  std::uint32_t motor =
      rewrite::rewrite_mix(0x29u, permutation, 0x43u) %
      rewrite::kRecordCapacity;
  if (motor == constructor)
    motor = (motor + 1u) % rewrite::kRecordCapacity;
  state->records[constructor].lane[0] = rewrite::kFormConstructor;
  state->records[constructor].lane[1] = 2u;
  state->records[constructor].lane[2] = rewrite::kMatureSupport;
  state->records[motor].lane[0] = rewrite::kFormMotor;
  state->constructor_locus = constructor;
  state->allocation_cursor =
      rewrite::rewrite_mix(permutation, 0xa5u, 0x5au) %
      rewrite::kRecordCapacity;
  rewrite::refresh_receipt(state);
}

// --- Epoch phase boundary API (0X1-175 Patch A) -----------------------------
//
// EpochPhaseReceipt is a stack-local, device-owned, value-neutral scheduling
// record threaded through one epoch's four phase calls below. It carries no
// resident authority of its own; every field mirrors an existing per-epoch
// transient from the former single-function kernel body.
//
// `progressed` deliberately carries more than one sequential meaning across
// the four phases, exactly reproducing the original data flow without adding
// a persistent DeviceState field:
//   - written by advance_close_phase: this epoch's close-transaction
//     `completed_close` signal (matches resident_close_work's own
//     "progressed" vocabulary for forward close progress);
//   - read+extended by consume_external_phase: combined with the live
//     `state->close_work_active` field to reproduce the original
//     `close_barrier`/`close_barrier_after_ingress`/`close_barrier_after_action`
//     locals, then folded with this epoch's `action_stream_barrier` (which
//     already subsumes `consumed_action_return`) for the next two phases;
//   - read by advance_resident_cognition_phase to reproduce
//     run_autonomous_generation_device's original close/return guard exactly;
//   - read by commit_public_phase to reconstruct `publish_due`, then
//     overwritten with the phase's own final `history_healthy` verdict (which
//     can diverge from any persistent field -- see the mouth/append failure
//     paths below) for the coordinator's fault-stop check.
//
// One narrowly-scoped, explicitly non-causal approximation remains from the
// fixed per-epoch shape below and is called out at its exact site:
//   1. The coordinator's idle `__nanosleep` pacing condition (in the .cu)
//      substitutes the live `action_return_stream_active` flag for the
//      original epoch-local `action_stream_barrier` transient.
//      `__nanosleep` has no resident/receipt/egress-history side effect.
struct EpochPhaseReceipt {
  bool consumed_contact = false;
  bool consumed_physical = false;
  bool accepted_action_return = false;
  bool device_body_return = false;
  bool device_body_attached = false;
  bool generated = false;
  bool progressed = false;
};

// Recomputes the former kernel body's single `history_healthy` local from its
// exact persistent-state sources. Idempotent and safe to call once per phase:
// the corruption-detection branch's side effect (`egress_history.fault = 1`)
// only ever transitions 0->1, and nothing between phases mutates the egress
// history's next_sequence/oldest_sequence fields before commit_public_phase's
// own append. Folding in `state->world.fault == 0u` reproduces the original
// close_work_fault-driven override (world.fault is guaranteed 0 at the start
// of every epoch: a nonzero value always stops the kernel before its next
// launch), so a fresh recompute at each phase boundary exactly reproduces the
// single original local's value at each point it was read.
__device__ __forceinline__ bool resident_epoch_history_healthy(
    DeviceState* state) {
  bool history_healthy = state->egress_history.fault == 0u;
  if (history_healthy &&
      (state->egress_history.next_sequence == 0u ||
       state->egress_history.oldest_sequence == 0u ||
       state->egress_history.oldest_sequence > state->egress_history.next_sequence ||
       state->egress_history.next_sequence - state->egress_history.oldest_sequence >
           egress_history::capacity ||
       state->egress_history.next_sequence >= egress_history::max_sequence - 1u)) {
    state->egress_history.fault = 1u;
    history_healthy = false;
  }
  return history_healthy && state->world.fault == 0u;
}

__device__ __noinline__ void advance_close_phase(DeviceState* state,
                                                  EpochPhaseReceipt* r) {
  const bool history_healthy = resident_epoch_history_healthy(state);
  bool completed_close = false;
  bool settlement_generated = false;
  bool accepted_action_return = r->accepted_action_return;
  bool device_body_return = r->device_body_return;
  if (history_healthy && state->close_work_active != 0u)
    (void)advance_close_transaction(state);
  if (history_healthy && state->close_work_active != 0u)
    (void)advance_close_settlement(state, &completed_close,
                                   &settlement_generated,
                                   &accepted_action_return,
                                   &device_body_return);
  r->accepted_action_return = accepted_action_return;
  r->device_body_return = device_body_return;
  if (settlement_generated) r->generated = true;
  // A completed private shadow never became authority; publish only an
  // explicit fault in the canonical shell so resident records stay at the
  // exact pre-transaction image. resident_epoch_history_healthy's next call
  // observes this immediately (world.fault is checked there).
  if (state->close_work_fault != 0u && state->world.fault == 0u)
    state->world.fault = state->close_work_fault;
  r->progressed = completed_close;
}

// Resident epoch execution unit: generation, close ordering, and the
// single public commitment membrane remain one device-side causal path.
// Keep autonomous generation and its bounded observer work behind a real
// device-call boundary. The epoch kernel owns ordering and the single commit
// membrane; this helper only reads the current resident snapshot, invokes the
// existing resident readers, and records their already-derived result. Its
// transient arrays and engine frames must not accumulate with ingress,
// action-return, close, and public-emission frames in the epoch root.
__device__ __noinline__ bool mark_resident_motor_babble_generated(
    rewrite::ResidentRewriteState* state, std::uint32_t motor) {
  const std::uint32_t header_slot = rewrite::find_current_trajectory(state);
  if (header_slot == rewrite::kInvalid ||
      state->records[header_slot].lane[2] == 0u)
    return false;
  const rewrite::Record& header = state->records[header_slot];
  const std::uint32_t index = header.lane[2] - 1u;
  const std::uint32_t local = index % 2u;
  rewrite::Record* provenance =
      rewrite::mixed_provenance::ensure_provenance_block(
          state, header.lane[1], index / 2u);
  if (provenance == nullptr ||
      (provenance->lane[rewrite::kProvenanceValidityLane] &
       ~0x3u) != 0u ||
      (provenance->lane[rewrite::kProvenanceValidityLane] &
       rewrite::mixed_provenance::valid_bit(local)) != 0u)
    return false;
  provenance->lane[rewrite::mixed_provenance::origin_lane(local)] =
      static_cast<std::uint32_t>(
          rewrite::mixed_provenance::Origin::generated);
  provenance->lane[rewrite::mixed_provenance::producer_lane(local)] = motor;
  provenance->lane[rewrite::kProvenanceValidityLane] |=
      rewrite::mixed_provenance::valid_bit(local);
  ++provenance->revision;
  return true;
}

__device__ __noinline__ bool run_resident_motor_babble_device(
    DeviceState* state) {
  if (state == nullptr || state->world.fault != 0u ||
      state->action_return_ticket.nonce != 0u ||
      state->world.program_rules != 0u ||
      state->world.span_program_rules != 0u)
    return false;
  const std::uint32_t trajectory_slot =
      rewrite::find_current_trajectory(&state->world);
  if (trajectory_slot == rewrite::kInvalid) return false;
  const rewrite::Record& trajectory = state->world.records[trajectory_slot];
  if (trajectory.lane[2] != 2u || trajectory.lane[3] != 0u ||
      (trajectory.lane[7] & rewrite::kTrajectoryWasYielded) == 0u ||
      (trajectory.lane[7] & rewrite::kTrajectoryHasGenerated) != 0u)
    return false;
  std::uint32_t first = 0u;
  std::uint32_t second = 0u;
  rewrite::mixed_provenance::Origin first_origin{};
  rewrite::mixed_provenance::Origin second_origin{};
  std::uint32_t first_producer = rewrite::kInvalid;
  std::uint32_t second_producer = rewrite::kInvalid;
  if (!rewrite::trajectory_word_at(&state->world, trajectory.lane[1], 0u,
                                   &first) ||
      !rewrite::trajectory_word_at(&state->world, trajectory.lane[1], 1u,
                                   &second) ||
      !rewrite::mixed_provenance::origin_at(
          &state->world, trajectory, 0u, &first_origin, &first_producer) ||
      !rewrite::mixed_provenance::origin_at(
          &state->world, trajectory, 1u, &second_origin, &second_producer) ||
      first_origin != rewrite::mixed_provenance::Origin::external ||
      second_origin != rewrite::mixed_provenance::Origin::external ||
      first_producer != rewrite::kInvalid ||
      second_producer != rewrite::kInvalid)
    return false;
  const std::uint32_t motor =
      rewrite::find_form(&state->world, rewrite::kFormMotor);
  std::uint32_t action = 0u;
  if (motor == rewrite::kInvalid ||
      !rewrite::resident_motor_babble_action_word(
          &state->world, motor, first, second, &action))
    return false;
  rewrite::clear_generated_word(&state->world);
  if (!rewrite::append_trajectory_word(&state->world, action, true))
    return false;
  if (!mark_resident_motor_babble_generated(&state->world, motor)) {
    (void)rewrite::mixed_provenance::fail_closed(
        &state->world, rewrite::mixed_provenance::kGeneratedStampFault);
    return false;
  }
  state->world.generated_word = action;
  state->world.generated_word_valid = 1u;
  state->world.generated_locus = motor;
  if (state->resident_motor_babble_actions != ~std::uint64_t{0})
    ++state->resident_motor_babble_actions;
  return true;
}

// Unchanged from the former resident_rewrite_epoch_kernel body, apart from
// collapsing its three original guard-only parameters
// (close_barrier_after_action, action_stream_barrier, consumed_action_return
// -- none referenced again past the early-return guard) into one `blocked`
// boolean; see advance_resident_cognition_phase below for the equivalent
// boolean-algebra proof at the call site.
__device__ __noinline__ bool run_autonomous_generation_device(
    DeviceState* state, bool history_healthy, bool blocked,
    bool consumed_contact, bool device_body_attached, IngressRing* ingress,
    ActionReturnIngress* action_return) {
  if (state == nullptr || !history_healthy || blocked ||
      ingress == nullptr ||
      action_return == nullptr || ingress_pending(ingress) ||
      action_return_ingress_pending(action_return) ||
      (state->world.organization_receipt_deferred != 0u &&
       state->world.revision != state->world.organization_receipt_revision))
    return false;

  const rewrite::ResidentRewriteEngine engine(&state->world);
  bool generated_this_epoch = false;
  // Applicability is checked before every legacy/bootstrap reader can
  // mutate the yielded trajectory (notably OpenInquiry's transient marker).
  // A unique ready distributed closure owns publication. An ambiguous
  // closure abstains. A non-ambiguous but incomplete closure has not earned
  // public authority yet, so learned bootstrap construction may continue
  // while independent contact grows the replacement topology.
  bool relation_generation_attempted = false;
  bool relation_generation_abstained = false;
  bool relation_incomplete = false;
  const std::uint32_t relation_trajectory =
      rewrite::find_current_trajectory(&state->world);
  if (relation_trajectory != rewrite::kInvalid) {
    // Snapshot the actual query before calling the passive reader. The
    // observer receipt lives outside ResidentRewriteState, so recording it
    // cannot perturb canonical world bytes, revision, lineage, or provenance.
    // Keep it sticky across later idle epochs so a timeout cannot erase the
    // last real relation trajectory observed by this runtime translation unit.
    const rewrite::Record relation_query =
        state->world.records[relation_trajectory];
    state->receipt.rewrite_causal_relation_candidate_query_observed = 1u;
    state->receipt.rewrite_causal_relation_candidate_query_owner =
        relation_query.lane[1];
    state->receipt.rewrite_causal_relation_candidate_query_revision =
        relation_query.revision;
    state->receipt.rewrite_causal_relation_candidate_query_extent =
        relation_query.lane[2];
    state->receipt.rewrite_causal_relation_candidate_readiness_stage = 0u;
    state->receipt.rewrite_causal_relation_candidate_source_provenance_failure =
        0u;
    state->receipt.rewrite_causal_relation_candidate_source_provenance_owner =
        rewrite::kInvalid;
    state->receipt.rewrite_causal_relation_candidate_applicable = 0u;
    state->receipt.rewrite_causal_relation_candidate_ready = 0u;
    state->receipt.rewrite_causal_relation_candidate_ambiguous = 0u;
    state->receipt.rewrite_causal_relation_candidate_query_linked = 0u;
    state->receipt.rewrite_causal_relation_candidate_trajectory_owner =
        rewrite::kInvalid;
    state->receipt.rewrite_causal_relation_candidate_trajectory_revision = 0u;
    state->receipt.rewrite_causal_relation_candidate_relation = 0u;
    state->receipt.rewrite_causal_relation_candidate_extent = 0u;
    const rewrite::CausalRelationCandidate relation_probe =
        run_causal_relation_candidate_device(
            &state->world, relation_query);
    state->receipt.rewrite_causal_relation_candidate_applicable =
        relation_probe.applicable ? 1u : 0u;
    state->receipt.rewrite_causal_relation_candidate_ready =
        relation_probe.ready ? 1u : 0u;
    state->receipt.rewrite_causal_relation_candidate_ambiguous =
        relation_probe.ambiguous ? 1u : 0u;
    state->receipt.rewrite_causal_relation_candidate_query_linked =
        relation_probe.query_linked ? 1u : 0u;
    state->receipt.rewrite_causal_relation_candidate_trajectory_owner =
        relation_probe.trajectory_owner;
    state->receipt.rewrite_causal_relation_candidate_trajectory_revision =
        relation_probe.trajectory_revision;
    state->receipt.rewrite_causal_relation_candidate_relation =
        relation_probe.relation;
    state->receipt.rewrite_causal_relation_candidate_extent =
        relation_probe.extent;
    state->receipt.rewrite_causal_relation_candidate_readiness_stage =
        relation_probe.readiness_stage;
    state->receipt.rewrite_causal_relation_candidate_source_provenance_failure =
        relation_probe.source_provenance_failure;
    state->receipt.rewrite_causal_relation_candidate_source_provenance_owner =
        relation_probe.source_provenance_owner;
    if (relation_probe.applicable) {
      state->world.causal_relation_probe_steps =
          relation_probe.ready ? 2u : 1u;
      state->world.causal_relation_participating_records =
          relation_probe.participating_records;
      state->world.causal_relation_independent_sources =
          relation_probe.independent_sources;
      state->world.causal_relation_external_leaves =
          relation_probe.external_leaves;
      if (relation_probe.ready && !relation_probe.ambiguous) {
        relation_generation_attempted = true;
        generated_this_epoch = run_mixed_provenance_device(engine);
        relation_generation_abstained = !generated_this_epoch;
      } else if (relation_probe.ambiguous && relation_probe.query_linked) {
        // Do not let an older provider choose through a conflict that is
        // actually linked to this query's resident context. A generic
        // connective conflict without query linkage is observer evidence for
        // the relation reader, not authority to suppress an independent
        // grounded construction path for a novel antecedent.
        relation_generation_attempted = true;
        // The relation reader abstains, but the ambiguity is not a veto on
        // unrelated ordinary resident construction. Let the legacy donor
        // readers continue to compete without allowing this conflicted
        // relation candidate to publish through them.
        relation_generation_abstained = true;
      } else if (!relation_probe.ready && !relation_probe.ambiguous) {
        // An incomplete relation is resident evidence, not an inquiry
        // transition and not publication authority.  Do not let OpenInquiry
        // mutate this yielded trajectory before the ordinary Program,
        // Span, and VersionSpace readers can reproduce the action whose
        // genuinely returned consequence may supply another independent
        // participation source.
        //
        // This is scheduling only: the relation contributes no word,
        // provider, ticket, expected consequence, or host-selected authority.
        relation_incomplete = true;
      }
    }
  }

  bool inquiry_generated = false;
  // An explicitly declined relation candidate (ambiguous and query-linked, or
  // ready but unable to rederive) is not authority to suppress OpenInquiry's
  // already-granted reply continuation or surface emission -- the same
  // abstain escape hatch already given to the episodic-completion and
  // mixed-provenance readers below. Without it, a query-linked relation that
  // keeps re-abstaining every epoch for the same yielded trajectory silently
  // starves OpenInquiry forever: canonical public emission would never
  // observe the returned consequence it depends on (0X1-206). Only a still-
  // incomplete relation keeps the original fresh-trajectory-hijack guard.
  if ((!relation_generation_attempted || relation_generation_abstained) &&
      !relation_incomplete)
    inquiry_generated = run_open_inquiry_device(engine);
  if (inquiry_generated) {
    generated_this_epoch = true;
    if (!run_mark_generated_device(&state->world,
                                   state->world.generated_locus)) {
      (void)run_fail_generated_device(&state->world);
      inquiry_generated = false;
      generated_this_epoch = false;
    }
  }
  bool revision_reader_generated = false;
  if (!generated_this_epoch &&
      (!relation_generation_attempted || relation_generation_abstained)) {
    revision_reader_generated =
        run_revision_participation_reader_device(engine);
    generated_this_epoch = revision_reader_generated;
    if (revision_reader_generated &&
        !run_mark_generated_device(&state->world,
                                   state->world.generated_locus)) {
      (void)run_fail_generated_device(&state->world);
      revision_reader_generated = false;
      generated_this_epoch = false;
    }
  }
  // RWR11 translated all new valid complete one-episode memory into ordinary
  // Program matter. The retained scanner is a bounded migration fallback for
  // historical/non-convertible retained sources and must yield whenever
  // canonical Program matter is engaged.
  // A ready generalized relation is a candidate authority, not a veto on an
  // already grounded ordinary Program. If its resident emission cannot be
  // rederive for this exact trajectory, fall back to ordinary resident
  // readers. Query-linked ambiguity remains fail-closed above.
  if (!generated_this_epoch &&
      (!relation_generation_attempted || relation_generation_abstained))
    generated_this_epoch = run_episodic_completion_device(engine);
  if (!generated_this_epoch &&
      (!relation_generation_attempted || relation_generation_abstained))
    generated_this_epoch = run_mixed_provenance_device(engine);
  // 0X1-163 diagnostics: decompose the combined `&&` gate below into its own
  // itemized counters, same technique as every oi_*_decline_* set in
  // bcc32_resident_open_inquiry_diagnostic_counters.inl. This preserves the
  // exact original short-circuit call pattern -- run_observe_resumed_
  // consequence_device is still invoked at most once per epoch, and only
  // when the first three terms already held, so no new call or side effect
  // is introduced; the counters are read-only bookkeeping around the same
  // branch.
  const bool constructor_settle_gate_eligible =
      generated_this_epoch && !inquiry_generated && !revision_reader_generated;
  if (constructor_settle_gate_eligible) ++state->world.oi_ctor_gate_eligible;
  if (constructor_settle_gate_eligible &&
      run_observe_resumed_consequence_device(&state->world)) {
    ++state->world.oi_ctor_gate_resume_observed;
    (void)run_settle_inquiry_constructor_device(&state->world);
  }
  const std::uint32_t generated_trajectory =
      rewrite::find_current_trajectory(&state->world);
  const bool provenance_owned =
      generated_trajectory != rewrite::kInvalid &&
      run_tagged_history_device(&state->world, generated_trajectory);
  bool prospective_rewrite = false;
  // Channel-1 output is already an executable public action. Backward search
  // applies only to a generated non-action target; otherwise an unrelated
  // Program ending in the same action can replace the action with its
  // predecessor and restart forward language generation.
  if (generated_this_epoch && !inquiry_generated &&
      !revision_reader_generated &&
      (state->world.generated_word & rewrite::kRawChannelMask) !=
          (1u << 24u)) {
    prospective_rewrite = run_learned_cost_search_device(engine);
    if (!prospective_rewrite)
      prospective_rewrite = run_means_end_inversion_device(engine);
  }
  if (prospective_rewrite) {
    (void)run_append_generated_device(&state->world,
                                      state->world.generated_word);
    if (provenance_owned) {
      if (!run_mark_generated_device(&state->world,
                                     state->world.generated_locus))
        (void)run_fail_generated_device(&state->world);
    }
  } else if (generated_this_epoch && !inquiry_generated) {
    (void)run_pending_retain_device(&state->world);
  }
  // A language-naive adult with an attached body may explore only after all
  // learned resident readers abstain. The body mapping remains outside the
  // world and the generated action still crosses the ordinary public gate.
  if (!generated_this_epoch && !relation_generation_attempted &&
      device_body_attached)
    generated_this_epoch = run_resident_motor_babble_device(state);
  return generated_this_epoch;
}

// The ticket-required signature for this boundary is `(DeviceState*,
// EpochPhaseReceipt*)`. run_autonomous_generation_device's existing guard
// (unchanged above) needs a *live* read of the ingress/action-return ring
// buffers (`ingress_pending`/`action_return_ingress_pending`), which are not
// reachable through DeviceState or EpochPhaseReceipt -- both are transport
// objects allocated outside DeviceState and passed to the persistent kernel
// per launch. Omitting that live check would silently change whether the
// resident generates public output this epoch (an observable
// egress_history/receipt change, not a scheduling nicety), so this function
// takes the two ring pointers as additional trailing parameters rather than
// approximate a causally load-bearing gate. This is the one place Patch A's
// mechanical split could not fit the ticket's literal four-signature list
// without changing behavior; every other boundary matches it exactly.
__device__ __noinline__ void advance_resident_cognition_phase(
    DeviceState* state, IngressRing* ingress, ActionReturnIngress* returned,
    EpochPhaseReceipt* r) {
  const bool history_healthy = resident_epoch_history_healthy(state);
  // Zero-authority recurrent transport (Cloud-1 infrastructure) must run
  // strictly before read-only candidate generation so a carrier that just
  // arrived this epoch cannot be observed by a probe from the same epoch;
  // this is unchanged from the original single-function ordering.
  if (history_healthy)
    (void)resident_recurrent_carrier::advance_recurrent_carriers(&state->world);
  // 0X1-267: the 0X1-248 predictive-shadow relation assay's generic
  // per-epoch clock. Same discipline as advance_recurrent_carriers above --
  // runs every history-healthy epoch including quiet ones, publishes
  // nothing, and only ever touches its own bounded matter.
  if (history_healthy)
    resident_predictive_shadow_assay::advance_matter(state->predictive_shadow);
  // `r->progressed` here holds (completed_close || action_stream_barrier)
  // from advance_close_phase/consume_external_phase. Combined with the live
  // close_work_active field this reproduces
  // (close_barrier_after_action || action_stream_barrier ||
  //  consumed_action_return) exactly: action_stream_barrier's own definition
  // already includes consumed_action_return, and close_barrier_after_action
  // is (completed_close || close_work_active != 0).
  const bool blocked = r->progressed || state->close_work_active != 0u;
  r->generated =
      r->generated ||
      run_autonomous_generation_device(state, history_healthy, blocked,
                                       r->consumed_contact,
                                       r->device_body_attached, ingress,
                                       returned);
  // 0X1-267 requirement 2: resident-generated/endogenous activity becomes
  // non-authoritative shadow ancestry. This never adds external_support to a
  // relation and never gates generation, publication, or the public mouth
  // gate -- it only lets the assay's own match/violation/omission chemistry
  // observe what the adult just said, the same way it observes what arrived.
  if (history_healthy && r->generated && state->world.generated_word_valid != 0u)
    resident_predictive_shadow_assay::observe_contact(
        state->predictive_shadow,
        static_cast<std::uint32_t>(state->world.generated_word),
        resident_predictive_shadow_assay::ContactOrigin::endogenous);

  // 0X1-267 requirement 5, step 1 (Linear 0X1-267 comment c98064fd): a
  // purely diagnostic, non-gating probe of whether this epoch's already-
  // computed predictive-shadow route projection (project_prepared_routes,
  // folded into predictive_shadow.projected_route by advance_matter above)
  // physically intersects the resident morphology run_autonomous_generation_
  // device already judged eligible this epoch. "Eligible morphology" here
  // is the relation-candidate receipt word that reader already snapshotted
  // into state->receipt (rewrite_causal_relation_candidate_relation/
  // _trajectory_owner, populated at bcc32_resident_rewrite_epoch_phases.inl's
  // relation_query block above, sticky across idle epochs by that block's
  // own design). Folding through local_contact_reaction_word reuses the
  // same proved reversible chemistry the reafferent adapter already reuses
  // at bcc32_resident_causal_constraint_reafferent.cuh, rather than
  // inventing a second predictor mechanism. This block only ever writes the
  // new predictive_shadow_route_probe_* DeviceState scalars below -- it
  // reads state->receipt and state->predictive_shadow but never writes
  // either, and run_autonomous_generation_device's dispatch above already
  // completed unconditionally before this block runs, so nothing here can
  // change which reader or route that cascade selected this epoch.
  if (history_healthy) {
    const SiteWord eligible_morphology =
        static_cast<SiteWord>(
            state->receipt.rewrite_causal_relation_candidate_relation) ^
        static_cast<SiteWord>(
            state->receipt.rewrite_causal_relation_candidate_trajectory_owner);
    const SiteWord probe =
        resident_predictive_shadow_assay::local_contact_reaction_word(
            state->predictive_shadow.projected_route, eligible_morphology,
            state->predictive_shadow.pair_epoch);
    const SiteWord intersection = probe & eligible_morphology;
    state->predictive_shadow_route_probe_eligible_morphology =
        eligible_morphology;
    state->predictive_shadow_route_probe_intersection = intersection;
    state->predictive_shadow_route_probe_intersection_popcount =
        static_cast<std::uint32_t>(__popc(intersection));
  }
}

__device__ __noinline__ void consume_external_phase(
    DeviceState* state, IngressRing* ingress, ActionReturnIngress* returned,
    DeviceBodyControl* body, PhysicalIngress* physical,
    EpochPhaseReceipt* r, ResidentEpochCleanupScratch* epoch_cleanup) {
  const bool history_healthy = resident_epoch_history_healthy(state);
  const bool close_barrier = r->progressed || state->close_work_active != 0u;

  const bool action_stream_barrier_before_epoch =
      state->action_return_stream_active != 0u;

  bool ended_contact = false;
  const bool action_return_pending_before_ingress =
      action_return_ingress_pending(returned);
  const bool consumed_contact =
      history_healthy && !close_barrier && !action_stream_barrier_before_epoch
              && !action_return_pending_before_ingress
          ? consume_ingress(state, ingress, &ended_contact)
          : false;
  r->consumed_contact = consumed_contact;

  bool ended_return = false;
  const bool device_body_attached = device_body_enabled(body);
  r->device_body_attached = device_body_attached;
  const bool close_barrier_after_ingress =
      close_barrier || state->close_work_active != 0u;
  (void)(history_healthy && !close_barrier_after_ingress &&
                 !action_stream_barrier_before_epoch
             ? produce_device_body_return(state, body, returned)
             : false);
  bool accepted_action_return = r->accepted_action_return;
  bool device_body_return = r->device_body_return;
  const bool consumed_action_return =
      history_healthy && !close_barrier &&
              state->close_work_active == 0u &&
              !accepted_action_return
          ? consume_action_return(state, returned, body,
                                  &accepted_action_return, &device_body_return,
                                  &ended_return, epoch_cleanup)
          : false;
  r->accepted_action_return = accepted_action_return;
  r->device_body_return = device_body_return;
  const bool action_stream_barrier =
      action_stream_barrier_before_epoch || consumed_action_return;
  if (epoch_cleanup != nullptr &&
      epoch_cleanup->phase == kEpochCleanupStaged) {
    r->progressed = r->progressed || action_stream_barrier;
    return;
  }
  const bool close_barrier_after_action =
      close_barrier_after_ingress || state->close_work_active != 0u;
  bool consumed_physical = false;
  if (history_healthy) {
    consumed_physical =
        !close_barrier_after_action && !action_stream_barrier
            ? consume_physical(state, physical)
            : false;
  }
  r->consumed_physical = consumed_physical;
  const bool awaiting_first_action = state->action_return_issued == 0u;
  const bool resident_quiet =
      !consumed_contact && !consumed_action_return && !consumed_physical &&
      !r->progressed /* completed_close, pre-fold */ &&
      !close_barrier_after_action && !action_stream_barrier &&
      state->close_work_active == 0u &&
      state->action_return_autonomy_barrier == 0u &&
      !ingress_pending(ingress) && !action_return_ingress_pending(returned) &&
      !awaiting_first_action;
  if (resident_quiet)
    rewrite::advance_mouth_compartment_epoch(&state->mouth, state->tick);

  // Fold this epoch's return-stream activity into the shared scheduling
  // signal. advance_resident_cognition_phase and commit_public_phase both
  // read it next -- see the EpochPhaseReceipt comment above for the full
  // cross-phase contract.
  r->progressed = r->progressed || action_stream_barrier;
}

__device__ __noinline__ void commit_public_phase(DeviceState* state,
                                                  EpochPhaseReceipt* r,
                                                  ResidentEpochGraphControl* epoch_control) {
  bool history_healthy = resident_epoch_history_healthy(state);
  bool generated_this_epoch = r->generated;
  const bool consumed_contact = r->consumed_contact;
  const bool consumed_physical = r->consumed_physical;

  // The canonical public-emission gate (0X1-158). Ordinary producers use
  // their resident locus. Distributed closures use an opaque receipt handle
  // derived from the already-committed relation closure; no Record locus is
  // fabricated for them.
  std::uint32_t mouth_index = rewrite::kMouthCompartmentCount;
  const bool locus_valid = generated_this_epoch &&
      state->world.generated_locus != rewrite::kInvalid &&
      state->world.generated_locus < rewrite::kRecordCapacity;
  bool receipt_valid = generated_this_epoch &&
      state->world.generated_receipt_valid != 0u &&
      state->world.generated_receipt_owner != rewrite::kInvalid &&
      state->world.generated_receipt_owner != 0u &&
      state->world.generated_receipt_participant_records != 0u &&
      state->world.generated_receipt_external_leaves != 0u &&
      state->world.generated_receipt_independent_sources >= 2u &&
      state->world.generated_receipt_source_contributions >= 2u &&
      state->world.generated_receipt_topology_digest != 0u &&
      state->world.generated_receipt_revision_digest != 0u &&
      state->world.generated_receipt_provenance_digest != 0u &&
      state->world.generated_receipt_participation_digest != 0u;
  if (receipt_valid) {
    // Grounded resident evidence can reacquire the public membrane after a
    // quiet lease expires. This is existing resident payment, not a host
    // reward or semantic exception.
    const std::uint32_t receipt_index = rewrite::mouth_compartment_for_locus(
        state->world.generated_receipt_owner);
    if (!rewrite::mouth_compartment_may_speak(&state->mouth, receipt_index))
      (void)rewrite::admit_mouth_compartment_consequence(
          &state->world, &state->mouth, receipt_index,
          rewrite::kRecordMatterQ8, state->tick);
  }
  // A real external contact may renew the resident-selected producer's
  // speaking compartment before its consequence crosses the public gate.
  // This is the missing bridge between lease expiry and continued grounded
  // operation: the raw contact is the physical resource event, while the
  // resident-selected locus determines which compartment receives it. Quiet
  // generation cannot renew itself because consumed_contact is false there.
  // Map the resident-selected locus before invoking the canonical gate so a
  // renewal made by this contact can authorize the same epoch's publication.
  if (locus_valid)
    mouth_index = rewrite::mouth_compartment_for_locus(
        state->world.generated_locus);
  if (history_healthy && generated_this_epoch && consumed_contact &&
      locus_valid &&
      !rewrite::mouth_compartment_may_speak(&state->mouth, mouth_index))
    (void)rewrite::renew_mouth_compartment_consequence(
        &state->world, &state->mouth, mouth_index,
        rewrite::kRecordMatterQ8, state->tick);
  bool mouth_may_speak = false;
  if (locus_valid)
    mouth_may_speak = rewrite::mouth_compartment_gate_public_emission(
        &state->world, &state->mouth, state->world.generated_locus,
        &mouth_index);
  else if (receipt_valid) {
    mouth_index = rewrite::mouth_compartment_for_locus(
        state->world.generated_receipt_owner);
    mouth_may_speak = rewrite::mouth_compartment_may_speak(
        &state->mouth, mouth_index);
  }
  const bool public_lineage_valid = locus_valid || receipt_valid;
  const std::uint32_t public_producer = receipt_valid
      ? state->world.generated_receipt_owner
      : state->world.generated_locus;
  if (history_healthy && generated_this_epoch && !public_lineage_valid) {
    history_healthy = false;
  } else if (history_healthy && generated_this_epoch && !mouth_may_speak) {
    generated_this_epoch = false;
  } else if (history_healthy && generated_this_epoch &&
             !egress_history::append(&state->egress_history, state->world.generated_word,
                                     state->tick + 1u, public_producer)) {
    history_healthy = false;
  } else if (history_healthy && generated_this_epoch) {
    if (receipt_valid) {
      if (state->mouth.compartment[mouth_index].speak_count != rewrite::kInvalid)
        ++state->mouth.compartment[mouth_index].speak_count;
    } else {
      (void)rewrite::set_generated_word_in_compartment(
          &state->world, &state->mouth, mouth_index,
          state->world.generated_word, state->world.generated_locus);
    }
  }
  if (history_healthy && generated_this_epoch)
    run_refresh_history_digest_device(state);
  if (history_healthy && generated_this_epoch)
    run_update_projection_device(state);
  if (history_healthy && generated_this_epoch)
    run_issue_action_ticket_device(state);
  if (!history_healthy) {
    generated_this_epoch = false;
    // The fault is committed resident state even when append abstained before
    // writing an Event. Publish its metadata and digest before stopping so a
    // passive observer never sees a healthy history paired with a faulted
    // continuation.
    run_refresh_history_digest_device(state);
  }
  r->generated = generated_this_epoch;

  DerivedOutput derived{};
  derive_output(state, generated_this_epoch, &derived);
  ++state->tick;
  ++state->device_epochs;
  run_fill_receipt_device(state, consumed_contact, r->accepted_action_return,
                          r->device_body_return, r->device_body_attached,
                          derived);
  // The resident clock and commitment advance every epoch, but copying the
  // complete public membrane every idle epoch can starve a passive reader's
  // seqlock when several adults share one GPU. Publish every causal boundary
  // change immediately and expose silent progress through a bounded
  // heartbeat. `r->progressed` reconstructs the original
  // (consumed_action_return || completed_close || action_stream_barrier ||
  // consumed_physical) OR-cluster exactly so a physical lesion's committed
  // receipt crosses the same epoch boundary as the intervention itself.
  const bool publish_due =
      consumed_contact || r->progressed || r->accepted_action_return ||
      consumed_physical ||
      generated_this_epoch || state->close_work_active != 0u ||
      !history_healthy || (state->device_epochs & 63u) == 0u;

  // Carry the final history-healthy verdict (including this phase's own
  // public-lineage/mouth-gate/append failures, which are local-only and never
  // reach a persistent state field) back to the coordinator's fault-stop
  // check, exactly reproducing the original single-function `history_healthy`
  // local's value at the point the original kernel body checked it.
  r->progressed = history_healthy;
  if (epoch_control != nullptr) {
    epoch_control->publish_due = publish_due ? 1u : 0u;
    epoch_control->history_changed =
        (generated_this_epoch || !history_healthy) ? 1u : 0u;
    epoch_control->generated = generated_this_epoch ? 1u : 0u;
    epoch_control->consumed_contact = consumed_contact ? 1u : 0u;
    epoch_control->accepted_action_return = r->accepted_action_return ? 1u : 0u;
    epoch_control->device_body_return = r->device_body_return ? 1u : 0u;
    epoch_control->device_body_attached = r->device_body_attached ? 1u : 0u;
  }
}

// The canonical resident root owns the ordinary-F child clock when the
// developmental demo explicitly attaches it. The child returns only to the
// exact root executable recorded by that root; no host launch or host tick is
// involved in this path.
__device__ __noinline__ bool fail_resident_ordinary_f(
    DeviceState* state, Lifecycle* lifecycle, std::uint32_t code) {
  if (state != nullptr) {
    state->f_fault = code;
    if (lifecycle != nullptr) {
      std::uint32_t active_count = 0u;
      if (state->ordinary_f.publication != nullptr)
        active_count = state->ordinary_f.publication->active_count;
      else if (state->ordinary_f.active_count != nullptr)
        active_count = *state->ordinary_f.active_count;
      lifecycle->ordinary_f_active_count = active_count;
    }
  }
  if (lifecycle != nullptr) {
    cuda::atomic_ref<std::uint32_t, cuda::thread_scope_system> continuation(
        lifecycle->continuation_fault);
    continuation.store(0x52571100u | code, cuda::memory_order_release);
    cuda::atomic_ref<std::uint32_t, cuda::thread_scope_system> stopped(
        lifecycle->stopped);
    stopped.store(1u, cuda::memory_order_release);
  }
  return false;
}

__device__ __noinline__ bool queue_resident_ordinary_f_child(
    DeviceState* state, Lifecycle* lifecycle) {
  if (state == nullptr || lifecycle == nullptr || state->f_owned_clock == 0u)
    return false;
  auto& handle = state->ordinary_f;
  if (handle.forward_graph == nullptr || handle.return_graph_slot == nullptr ||
      handle.requested_target == nullptr || handle.completed_ticks == nullptr ||
      handle.fault == nullptr || handle.publication == nullptr)
    return fail_resident_ordinary_f(state, lifecycle, 1u);
  const std::uint64_t completed = ordinary_f::completed_ticks_acquire(handle);
  if (completed == ~std::uint64_t{0})
    return fail_resident_ordinary_f(state, lifecycle, 2u);
  const cudaGraphExec_t root = cudaGetCurrentGraphExec();
  if (root == nullptr)
    return fail_resident_ordinary_f(state, lifecycle, 13u);
  state->expected_f_tick = completed + 1u;
  handle.publication->continuation_phase = 1u;
  handle.publication->return_launch_status = 0u;
  *handle.return_graph_slot = root;
  __threadfence_system();
  const cudaError_t launch = ordinary_f::request_forward_tick(handle);
  if (launch != cudaSuccess)
    return fail_resident_ordinary_f(
        state, lifecycle, 3u | (static_cast<std::uint32_t>(launch) << 8u));
  return true;
}

__device__ __noinline__ bool advance_resident_ordinary_f(
    DeviceState* state, Lifecycle* lifecycle) {
  if (state == nullptr || lifecycle == nullptr || state->f_owned_clock == 0u)
    return true;
  if (state->f_fault != 0u)
    return fail_resident_ordinary_f(state, lifecycle, state->f_fault);
  const auto& handle = state->ordinary_f;
  if (handle.forward_graph == nullptr || handle.return_graph_slot == nullptr ||
      handle.requested_target == nullptr || handle.completed_ticks == nullptr ||
      handle.fault == nullptr || handle.publication == nullptr)
    return fail_resident_ordinary_f(state, lifecycle, 1u);
  if (state->expected_f_tick == 0u) {
    (void)queue_resident_ordinary_f_child(state, lifecycle);
    return false;
  }
  if (*handle.fault != static_cast<std::uint32_t>(ordinary_f::Fault::none))
    return fail_resident_ordinary_f(state, lifecycle,
                                    5u | (*handle.fault << 8u));
  const std::uint64_t completed = ordinary_f::completed_ticks_acquire(handle);
  if (completed < state->expected_f_tick)
    return false;
  if (completed != state->expected_f_tick)
    return fail_resident_ordinary_f(state, lifecycle, 7u);
  if (ordinary_f::publish_current_world(handle) != cudaSuccess)
    return fail_resident_ordinary_f(state, lifecycle, 8u);
  const ordinary_f::OrdinaryFPublication publication = *handle.publication;
  if (publication.fault != static_cast<std::uint32_t>(ordinary_f::Fault::none) ||
      publication.completed_ticks != state->expected_f_tick)
    return fail_resident_ordinary_f(state, lifecycle,
                                    9u | (publication.fault << 8u));
  state->expected_f_tick = 0u;
  return true;
}

// Short coordinator: bootstrap/shutdown bookkeeping, the four epoch phase
// boundaries in the existing causal order, one device_epochs increment (owned
// by commit_public_phase, immediately adjacent to derive_output/fill_receipt/
// publish_due exactly as in the original single function -- see the
// EpochPhaseReceipt comment above), fault-stop, idle pacing, and the device
// tail relaunch. Unchanged from the original kernel body apart from the
// documented `__nanosleep` pacing approximation below (no
// resident/receipt/egress-history effect).
__global__ void resident_rewrite_epoch_kernel(DeviceState* state, IngressRing* ingress,
                                              ActionReturnIngress* action_return,
                                              DeviceBodyControl* device_body,
                                              PhysicalIngress* physical, EgressState* egress,
                                              Lifecycle* lifecycle, ContentAddress sealed,
                                              ContentAddress law, ContentAddress image,
                                              ContentAddress genesis,
                                              std::uint64_t action_return_instance_nonce,
                                              ResidentEpochCleanupScratch* epoch_cleanup,
                                              ResidentEpochGraphControl* epoch_control) {
  if (blockIdx.x != 0u || threadIdx.x != 0u || state == nullptr || ingress == nullptr ||
      action_return == nullptr || device_body == nullptr || physical == nullptr || egress == nullptr || lifecycle == nullptr)
    return;

  cuda::atomic_ref<std::uint32_t, cuda::thread_scope_system> shutdown(lifecycle->shutdown);
  cuda::atomic_ref<std::uint32_t, cuda::thread_scope_system> stopped(lifecycle->stopped);
  cuda::atomic_ref<std::uint32_t, cuda::thread_scope_system> fault(lifecycle->continuation_fault);

  if (state->initialized == 0u) {
    state->sealed = sealed;
    state->law = law;
    state->image = image;
    state->genesis = genesis;
    state->action_return_instance_nonce = action_return_instance_nonce;
    const std::uint32_t permutation = static_cast<std::uint32_t>(kGenesisHash[0] ^ kGenesisHash[1] ^
                                                                 kGenesisHash[2] ^ kGenesisHash[3]);
    initialize_resident_world_device(&state->world, permutation);
    state->egress_history = egress_history::State{};
    // Pay every mouth compartment once at genesis so default behavior matches
    // the pre-0X1-158 topology exactly (all three compartments speaking,
    // nothing withdrawn). A targeted withdraw_mouth_compartment call is the
    // intervention under test, not something that happens spontaneously here.
    state->mouth = rewrite::MouthCompartmentField{};
    for (std::uint32_t i = 0u; i < rewrite::kMouthCompartmentCount; ++i)
      (void)rewrite::admit_mouth_compartment_consequence(
          &state->world, &state->mouth, i, rewrite::kRecordMatterQ8,
          state->tick);
    run_refresh_history_digest_device(state);
    state->host_bootstrap_launches = 1u;
    state->initialized = 1u;
  }

  if (shutdown.load(cuda::memory_order_acquire) != 0u) {
    stopped.store(1u, cuda::memory_order_release);
    return;
  }

  if (!advance_resident_ordinary_f(state, lifecycle)) {
    // The child owns the return edge and will tail-launch this exact root
    // after its device commit. Publish a mechanical phase receipt so a
    // passive observer can distinguish an armed child from a dead root.
    //
    // This must still go through derive_output rather than publish a
    // default-constructed DerivedOutput{} directly: DerivedOutput's
    // action_count defaults to 0u, but bcc32_cuda_resident_rewrite_production_contract
    // (assertion "canonical action surface is not the three raw rails")
    // requires every published snapshot -- including every warm-up epoch
    // published while F is still advancing, before any real epoch runs --
    // to expose exactly the three-word rail encoding. state->egress_history
    // is freshly zeroed at this point (genesis, above), so derive_output's
    // own "no admitted history yet" branch already yields the correct
    // canonical-but-invalid [0, 0, 0]/count=3 surface here; it just has to
    // actually be called.
    DerivedOutput bootstrap_derived{};
    derive_output(state, false, &bootstrap_derived);
    run_fill_receipt_device(state, false, false, false, false,
                            bootstrap_derived);
    run_publish_egress_device(state, bootstrap_derived, false, egress);
    return;
  }

  // All resident mutation helpers may still call refresh_receipt(), but the
  // canonical graph defers that passive population scan to the parallel census
  // nodes appended after this ordered epoch. A quiet epoch with no revision
  // change is skipped by the census itself, so this does not force a global
  // scan on every passive tick.
  state->world.organization_receipt_deferred = 1u;

  EpochPhaseReceipt r{};
  constexpr std::uint32_t kReceiptConsumedContact = 1u << 0u;
  constexpr std::uint32_t kReceiptConsumedPhysical = 1u << 6u;
  constexpr std::uint32_t kReceiptAcceptedActionReturn = 1u << 1u;
  constexpr std::uint32_t kReceiptDeviceBodyReturn = 1u << 2u;
  constexpr std::uint32_t kReceiptDeviceBodyAttached = 1u << 3u;
  constexpr std::uint32_t kReceiptGenerated = 1u << 4u;
  constexpr std::uint32_t kReceiptProgressed = 1u << 5u;
  if (epoch_cleanup != nullptr &&
      epoch_cleanup->phase == kEpochCleanupCommitted) {
    const bool matching_continuation =
        epoch_cleanup->epoch_generation == state->device_epochs &&
        epoch_cleanup->packet_generation ==
            state->action_return_stream_ticket.action_sequence &&
        same_action_return_ticket(epoch_cleanup->continuation_ticket,
                                  state->action_return_stream_ticket) &&
        state->action_return_stream_active != 0u;
    egress_history::Event action{};
    if (!matching_continuation ||
        !egress_history::lookup(
            &state->egress_history,
            epoch_cleanup->continuation_ticket.action_sequence, &action)) {
      epoch_cleanup->fault = rewrite::kCloseWorkTransactionFault;
      epoch_cleanup->phase = kEpochCleanupFault;
      state->world.fault = rewrite::kCloseWorkTransactionFault;
    } else {
      const std::uint32_t receipt_bits = epoch_cleanup->receipt_bits;
      r.consumed_contact = (receipt_bits & kReceiptConsumedContact) != 0u;
      r.consumed_physical = (receipt_bits & kReceiptConsumedPhysical) != 0u;
      r.accepted_action_return =
          (receipt_bits & kReceiptAcceptedActionReturn) != 0u;
      r.device_body_return =
          (receipt_bits & kReceiptDeviceBodyReturn) != 0u;
      r.device_body_attached =
          (receipt_bits & kReceiptDeviceBodyAttached) != 0u;
      r.generated = (receipt_bits & kReceiptGenerated) != 0u;
      r.progressed = (receipt_bits & kReceiptProgressed) != 0u;
      (void)finalize_action_return_commit(
          state, epoch_cleanup->continuation_next,
          epoch_cleanup->continuation_ticket, action,
          &r.accepted_action_return, &r.device_body_return, nullptr);
      epoch_cleanup->phase = kEpochCleanupIdle;
      __threadfence();
    }
  } else {
    if (epoch_cleanup != nullptr &&
        epoch_cleanup->phase == kEpochCleanupStaged) {
      fault.store(0x52571102u, cuda::memory_order_release);
      stopped.store(1u, cuda::memory_order_release);
      return;
    }
    advance_close_phase(state, &r);
    consume_external_phase(state, ingress, action_return, device_body, physical,
                           &r, epoch_cleanup);
    if (epoch_cleanup != nullptr &&
        epoch_cleanup->phase == kEpochCleanupStaged) {
      std::uint32_t receipt_bits = 0u;
      if (r.consumed_contact) receipt_bits |= kReceiptConsumedContact;
      if (r.consumed_physical) receipt_bits |= kReceiptConsumedPhysical;
      if (r.accepted_action_return)
        receipt_bits |= kReceiptAcceptedActionReturn;
      if (r.device_body_return) receipt_bits |= kReceiptDeviceBodyReturn;
      if (r.device_body_attached)
        receipt_bits |= kReceiptDeviceBodyAttached;
      if (r.generated) receipt_bits |= kReceiptGenerated;
      if (r.progressed) receipt_bits |= kReceiptProgressed;
      epoch_cleanup->receipt_bits = receipt_bits;
      __threadfence();
      if (epoch_control == nullptr ||
          epoch_control->cleanup_graph_exec == nullptr) {
        epoch_cleanup->fault = rewrite::kCloseWorkTransactionFault;
        epoch_cleanup->phase = kEpochCleanupFault;
        fault.store(0x52571103u, cuda::memory_order_release);
        stopped.store(1u, cuda::memory_order_release);
        return;
      }
      const cudaError_t cleanup_status = cudaGraphLaunch(
          epoch_control->cleanup_graph_exec, cudaStreamGraphTailLaunch);
      if (cleanup_status != cudaSuccess) {
        epoch_cleanup->fault = static_cast<std::uint32_t>(cleanup_status);
        epoch_cleanup->phase = kEpochCleanupFault;
        fault.store(0x52571100u | static_cast<std::uint32_t>(cleanup_status),
                    cuda::memory_order_release);
        stopped.store(1u, cuda::memory_order_release);
      }
      return;
    }
  }
  advance_resident_cognition_phase(state, ingress, action_return, &r);
  commit_public_phase(state, &r, epoch_control);

  if (!r.progressed) {
    fault.store(0x52572000u | state->egress_history.fault, cuda::memory_order_release);
    stopped.store(1u, cuda::memory_order_release);
    return;
  }
  if (state->world.fault != 0u) {
    fault.store(0x52570000u | state->world.fault, cuda::memory_order_release);
    stopped.store(1u, cuda::memory_order_release);
    return;
  }

  // Original per-epoch idle pacing also suppressed on `action_stream_barrier`
  // (queued/just-started/just-settled/just-rejected action-return activity);
  // that exact epoch-local transient does not survive the fixed
  // EpochPhaseReceipt shape (r.progressed has already been overwritten to the
  // history-healthy verdict above), and `__nanosleep` has no
  // resident/receipt/egress-history side effect, so the live
  // `action_return_stream_active` flag is used as the closest state-derived
  // proxy here -- a deliberate, narrowly-scoped device-pacing approximation.
  if (!r.consumed_contact && !r.generated && state->close_work_active == 0u &&
      state->action_return_stream_active == 0u)
    __nanosleep(kIdlePacingNanoseconds);
  if (shutdown.load(cuda::memory_order_acquire) == 0u) {
    if (state->f_owned_clock != 0u) {
      if (!queue_resident_ordinary_f_child(state, lifecycle))
        return;
      return;
    }
    if (epoch_control == nullptr || epoch_control->post_graph_exec == nullptr) {
      fault.store(0x52571105u, cuda::memory_order_release);
      stopped.store(1u, cuda::memory_order_release);
      return;
    }
    const cudaError_t status =
        cudaGraphLaunch(epoch_control->post_graph_exec,
                        cudaStreamGraphTailLaunch);
    if (status != cudaSuccess) {
      fault.store(0x52571000u | static_cast<std::uint32_t>(status), cuda::memory_order_release);
      stopped.store(1u, cuda::memory_order_release);
    }
  } else {
    stopped.store(1u, cuda::memory_order_release);
  }
}
