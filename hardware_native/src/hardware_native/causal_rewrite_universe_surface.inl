// Raw-packet admission, lawful lesion, and observer-surface helpers for the
// causal rewrite universe. Included only after the resident transforms are
// complete: causal path is raw packet -> resident admission -> one settled
// receipt, and lesion -> escrowed matter -> resident validation -> receipt.
// This unit owns no independent state, clock, semantic route, or observer write.

// Feed one bounded physical packet without exposing intermediate observer
// state after every byte. The event loop still preserves exact ingress order
// and frame semantics; only the final event settles the public receipt. This
// gives CUDA body adapters a one-pass ingress primitive while keeping the
// existing single-event API unchanged for interactive contact.
BCC32_REWRITE_HD inline RewriteBatchReceipt consume_rewrite_events(
    ResidentRewriteEngine engine, const RawRewriteEvent* events,
    std::uint32_t count, bool preserve_unclaimed_prefix = false) {
  RewriteBatchReceipt receipt{};
  receipt.requested = count;
  ResidentRewriteState* state = engine.state;
  if (state == nullptr) return receipt;
  receipt.fault = state->fault;
  if (count == 0u) {
    receipt.completed = state->fault == 0u ? 1u : 0u;
    receipt.observer_settled =
        state->cross_context_factor_pending == 0u ? 1u : 0u;
    const std::uint32_t current = find_current_trajectory(state);
    if (current != kInvalid) {
      receipt.trajectory_pages =
          trajectory_page_count(state, state->records[current].lane[1]);
      receipt.trajectory_continued = receipt.trajectory_pages > 1u ? 1u : 0u;
    }
    return receipt;
  }
  if (events == nullptr || state->fault != 0u) return receipt;
  for (; receipt.consumed < count && state->fault == 0u;
       ++receipt.consumed) {
    consume_rewrite_event(engine, events[receipt.consumed], false,
                          preserve_unclaimed_prefix);
  }
  // Settle exactly once even when a resident fault stopped the packet early.
  // Callers therefore never inherit the observer fields from an arbitrary
  // intermediate event and can distinguish complete from partial admission.
  refresh_receipt(state);
  receipt.fault = state->fault;
  receipt.observer_settled =
      state->cross_context_factor_pending == 0u ? 1u : 0u;
  receipt.completed =
      receipt.consumed == count && receipt.fault == 0u ? 1u : 0u;
  const std::uint32_t current = find_current_trajectory(state);
  if (current != kInvalid) {
    receipt.trajectory_pages =
        trajectory_page_count(state, state->records[current].lane[1]);
    receipt.trajectory_continued = receipt.trajectory_pages > 1u ? 1u : 0u;
  }
  return receipt;
}

BCC32_REWRITE_HD inline std::uint32_t apply_physical_lesion(
    ResidentRewriteEngine engine, std::uint32_t center,
    std::uint32_t count, std::uint32_t max_records) {
  ResidentRewriteState* state = engine.state;
  if (state == nullptr || max_records == 0u || count == 0u) return 0u;
  std::uint32_t removed = 0u;
  for (std::uint32_t offset = 0u;
       offset < count && state->lesion.count < 8u && max_records != 0u;
       ++offset) {
    const std::uint32_t slot = (center + offset) % live_record_capacity(state);
    Record& record = state->records[slot];
    if (record.matter_q8 == 0u || record.lane[0] == kFormEmpty ||
        record.lane[0] == kFormMotor)
      continue;
    const std::uint32_t escrow = state->lesion.count++;
    state->lesion.displaced[escrow] = record;
    if (record.lane[0] == kFormProgram &&
        (record.lane[7] & kProgramFlagVersionSpace) != 0u)
      state->lesion.displaced[escrow].lane[7] |=
          kProgramFlagVersionSpaceLesioned;
    state->lesion.original_slot[escrow] = slot;
    removed += record.matter_q8;
    state->lesion.removed_matter_q8 += record.matter_q8;
    record = Record{};
    record.matter_q8 = 0u;
    --max_records;
    ++state->revision;
  }
  clear_motor(state);
  if (removed != 0u) state->causal_germline_validation_pending = 1u;
#if !defined(__CUDA_ARCH__)
  (void)settle_causal_germline_pending(state);
#endif
  refresh_receipt(state);
  return removed;
}

BCC32_REWRITE_HD inline bool matter_account_is_closed(
    const ResidentRewriteState& state) {
  // Conservation target must scale with the live population, not the fixed
  // 1,024-Record page size: grow_resident_pages fills every slot on a new
  // page with kRecordMatterQ8 matter (Record's own default member
  // initializer, same as initialize_rewrite_state does for page 0), so
  // growth adds exactly kRecordsPerPage * kRecordMatterQ8 new matter into
  // the closed system. Scanning only page 0 while comparing against the
  // grown total (or vice versa) would make this invariant permanently
  // false the moment the population exceeds one page.
  const std::uint32_t capacity = live_record_capacity(&state);
  std::uint64_t matter = 0u;
  for (std::uint32_t i = 0u; i < capacity; ++i)
    matter += state.records[i].matter_q8;
  for (std::uint32_t i = 0u; i < state.lesion.count; ++i)
    matter += state.lesion.displaced[i].matter_q8;
  return matter == static_cast<std::uint64_t>(capacity) * kRecordMatterQ8;
}

BCC32_REWRITE_HD inline std::uint64_t observable_rewrite_digest(
    const ResidentRewriteState& state) {
  return state.organization_digest ^
         (static_cast<std::uint64_t>(state.raw_motor_value) << 32u) ^
         state.raw_motor_valid;
}
