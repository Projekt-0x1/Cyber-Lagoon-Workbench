// Inquiry emission is one resident progression: a learned constructor may
// surface a question, and a selected exact resident continuation may resume a
// yielded path.  Both produce ordinary generated trajectory words.
#if defined(__CUDACC__)
#define BCC32_OPEN_INQUIRY_SETTLEMENT_DISPATCH \
    [[maybe_unused]] static BCC32_OPEN_INQUIRY_HD __noinline__
#else
#define BCC32_OPEN_INQUIRY_SETTLEMENT_DISPATCH BCC32_OPEN_INQUIRY_HD inline
#endif

// A resumed reply continuation is the only place the runtime observes that a
// held-out episode's consequence has actually returned. Attempt constructor
// settlement immediately here, rather than relying on a later, separately
// scheduled pass; a blank END immediately after resumption must not erase the
// only resident opportunity to settle this inquiry.
BCC32_OPEN_INQUIRY_SETTLEMENT_DISPATCH bool
settle_completed_training_constructor_once(ResidentRewriteState* state) {
  return settle_constructor_from_complete_episodes(state);
}

BCC32_OPEN_INQUIRY_HD inline bool advance_reply_continuation_once(
    ResidentRewriteEngine engine, std::uint32_t emission_slot,
    std::uint32_t inquiry_slot) {
  ResidentRewriteState* state = engine.state;
  if (state == nullptr || emission_slot >= live_record_capacity(state) ||
      inquiry_slot >= live_record_capacity(state))
    return false;
  ++state->oi_reply_continuation_attempts;
  Record& emission = state->records[emission_slot];
  Record& inquiry = state->records[inquiry_slot];
  if (emission.lane[6] != kInquiryEmissionReplyContinuation ||
      emission.lane[1] != inquiry.lane[1] ||
      (inquiry.lane[7] & (kInquiryReplyBound | kInquirySettled)) !=
          (kInquiryReplyBound | kInquirySettled) ||
      (inquiry.lane[7] & kInquiryResumed) != 0u ||
      emission.lane[2] >= live_record_capacity(state))
    return false;
  std::uint32_t target_witness = kInvalid;
  for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
    const Record& witness = state->records[slot];
    if (witness.matter_q8 == 0u ||
        witness.lane[0] != kFormOpenInquiryTargetWitness ||
        witness.lane[1] != inquiry.lane[1])
      continue;
    if (target_witness != kInvalid)
      return false;
    target_witness = slot;
  }
  if (target_witness != kInvalid &&
      !resident_target_witness_live(state, target_witness))
    return false;
  const std::uint32_t suspended_slot =
      cross_contact::find_trajectory_by_owner(state, inquiry.lane[2]);
  if (suspended_slot == kInvalid)
    return false;
  Record yielded_prefix = state->records[suspended_slot];
  yielded_prefix.lane[2] = inquiry.lane[3];
  if (!inquiry_has_exact_snapshots(state, inquiry, yielded_prefix, true,
                                   nullptr))
    return false;
  AlternativeSnapshot selected{};
  if (!live_resident_continuation_alternative(
          state, emission.lane[2], yielded_prefix, &selected) ||
      selected.owner != inquiry.lane[5] || selected.revision != inquiry.lane[6] ||
      selected.owner != emission.lane[4] || selected.revision != emission.lane[5])
    return false;
  std::uint32_t length = 0u;
  std::uint32_t digest = 0u;
  if (!alternative_continuation_extent_for_emission(
          state, selected, inquiry.lane[3], &length, &digest) ||
      length == 0u || emission.lane[3] > length)
    return false;
  if (emission.lane[3] == length) {
    if ((inquiry.lane[7] & kInquiryReplyContinuationEmitted) == 0u)
      return false;
    if (observe_resumed_consequence(state))
      (void)settle_completed_training_constructor_once(state);
    return false;
  }
  std::uint32_t word = 0u;
  if (!alternative_continuation_word_at(state, selected, inquiry.lane[3],
                                        emission.lane[3], &word) ||
      !append_trajectory_word(state, word, true))
    return false;
  state->generated_word = word;
  state->generated_word_valid = 1u;
  state->generated_locus = emission.lane[2];
  state->active_locus = emission.lane[2];
  ++emission.lane[3];
  ++emission.revision;
  if (emission.lane[3] == length) {
    inquiry.lane[7] |= kInquiryReplyContinuationEmitted;
    ++inquiry.revision;
  }
  ++state->revision;
  refresh_receipt(state);
  ++state->oi_reply_continuation_admitted;
  return true;
}

#undef BCC32_OPEN_INQUIRY_SETTLEMENT_DISPATCH

// An inquiry admitted while zero constructors were authoritative
// (open_from_yielded_version_space's constructor_count == 0 branch) is left
// with lane[5], lane[6], and reserved[1] all kInvalid and no
// kFormOpenInquiryEmission Record -- the "if (constructor_count == 1u)"
// bootstrap at admission never ran for it.  bind_one_resident_target_without_
// external_reply cannot rescue that inquiry either: it requires
// kInquirySurfaceEmitted already set, and that flag is set only inside this
// very surface-emission success path, so the inquiry would be permanently
// unreachable once a constructor did become authoritative later. This
// replays the identical admission-time binding (constructor slot -> inquiry
// lane[5]/lane[6]/reserved[1], plus the one kInquiryEmissionSurface Record)
// the moment exactly one constructor becomes authoritative, instead of
// requiring that to have already happened at admission. No fork, no
// alternative, and no forecasted continuation content is created here --
// those already exist from admission -- only the missing constructor binding
// and its Emission Record, so this remains the same causal seam admission
// itself uses, not a second host-selected route.
BCC32_OPEN_INQUIRY_HD inline bool bind_deferred_first_surface_emission(
    ResidentRewriteState* state) {
  if (state == nullptr || state->fault != 0u) return false;
  const std::uint32_t inquiry_slot = unique_active_inquiry(state);
  if (inquiry_slot == kInvalid) return false;
  Record& inquiry = state->records[inquiry_slot];
  // Eligible only for an inquiry still in its as-admitted state: no
  // constructor was ever bound (lane[5]/lane[6]/reserved[1] all kInvalid),
  // and no later route (external capture, reply binding, settlement,
  // resumption) has touched it yet.
  if (inquiry.lane[7] != kInquiryAwaitingReply || inquiry.lane[5] != kInvalid ||
      inquiry.lane[6] != kInvalid || inquiry.reserved[1] != kInvalid)
    return false;
  const std::uint32_t suspended_slot =
      cross_contact::find_trajectory_by_owner(state, inquiry.lane[2]);
  if (suspended_slot == kInvalid) return false;
  const Record& suspended = state->records[suspended_slot];
  if (suspended.lane[3] == 0u || suspended.lane[2] != inquiry.lane[3] ||
      (suspended.lane[7] & kTrajectoryOpenInquiry) == 0u)
    return false;
  const std::uint32_t constructor_slot = unique_authoritative_constructor(state);
  if (constructor_slot == kInvalid) return false;
  for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
    const Record& record = state->records[slot];
    if (record.matter_q8 != 0u && record.lane[0] == kFormOpenInquiryEmission)
      return false;
  }
  if (free_record_count(state) < 1u) return false;
  const Record& constructor = state->records[constructor_slot];
  const std::uint32_t emission_slot = allocate_record(state);
  if (emission_slot == kInvalid) return false;
  Record& emission = state->records[emission_slot];
  emission.lane[0] = kFormOpenInquiryEmission;
  emission.lane[1] = inquiry.lane[1];
  emission.lane[2] = constructor_slot;
  emission.lane[3] = 0u;
  emission.lane[4] = constructor.lane[1];
  emission.lane[5] = constructor.revision;
  emission.lane[6] = kInquiryEmissionSurface;
  ++emission.revision;
  inquiry.lane[5] = constructor.lane[1];
  inquiry.lane[6] = constructor.revision;
  inquiry.reserved[1] = constructor_slot;
  ++inquiry.revision;
  ++state->revision;
  refresh_receipt(state);
  return true;
}

// This advances only resident learned bytes. The runtime remains responsible
// for stamping provenance, egress commitment, and public publication.
//
// Diagnostic-only counters (0X1-163/0X1-206, added following the same
// investigation that instrumented open_from_yielded_version_space): each
// early-return branch below increments one cumulative receipt-exposed
// counter in place, immediately before returning. No precondition or
// returned boolean is changed by this instrumentation -- combined ||
// conditions are decomposed into sequential ifs with identical clause order
// (preserving short-circuit safety, e.g. suspended_slot is still checked
// invalid before any record[suspended_slot] access), and a few closely
// related clauses are grouped under one counter name where splitting them
// further would not add distinguishing diagnostic value on its own.
BCC32_OPEN_INQUIRY_HD inline bool advance_surface_once(
    ResidentRewriteEngine engine) {
  ResidentRewriteState* state = engine.state;
  if (state == nullptr) return false;
  ++state->open_inquiry_surface_attempts;
  if (state->fault != 0u) return false;
  std::uint32_t emission_slot = kInvalid;
  for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
    const Record& emission = state->records[slot];
    if (emission.matter_q8 != 0u && emission.lane[0] == kFormOpenInquiryEmission) {
      if (emission_slot != kInvalid) {
        ++state->open_inquiry_surface_decline_ambiguous_emission;
        return false;
      }
      emission_slot = slot;
    }
  }
  if (emission_slot == kInvalid) {
    // A completed learned question may expose one already-resident physical
    // target. Rescan after binding so this same root epoch emits the first
    // continuation word; returning false would let a lower-priority executor
    // advance the reactivated prefix outside the target witness. This remains
    // iterative rather than recursive because the canonical root is already
    // stack-sensitive.
    // Try the deferred first-surface-emission bootstrap first: it only ever
    // applies to an inquiry admitted with zero constructors authoritative,
    // and it is what makes a constructor's later authority reachable at all
    // for that inquiry. bind_one_resident_target_without_external_reply
    // requires kInquirySurfaceEmitted already set, so it is a no-op for that
    // same inquiry regardless of order; trying the deferred bootstrap first
    // simply reaches the newly-possible admission without an extra tick.
    if (!bind_deferred_first_surface_emission(state) &&
        !bind_one_resident_target_without_external_reply(state)) {
      ++state->open_inquiry_surface_decline_no_emission;
      return false;
    }
    for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
      const Record& emission = state->records[slot];
      if (emission.matter_q8 == 0u ||
          emission.lane[0] != kFormOpenInquiryEmission)
        continue;
      if (emission_slot != kInvalid) {
        ++state->open_inquiry_surface_decline_ambiguous_emission;
        return false;
      }
      emission_slot = slot;
    }
    if (emission_slot == kInvalid) {
      ++state->open_inquiry_surface_decline_no_emission;
      return false;
    }
  }
  Record& emission = state->records[emission_slot];
  const std::uint32_t inquiry_slot = unique_header_by_owner(
      state, kFormOpenInquiry, emission.lane[1]);
  if (inquiry_slot == kInvalid) {
    ++state->open_inquiry_surface_decline_no_inquiry_header;
    return false;
  }
  if (emission.lane[6] == kInquiryEmissionReplyContinuation) {
    ++state->open_inquiry_surface_reply_dispatch;
    return advance_reply_continuation_once(engine, emission_slot, inquiry_slot);
  }
  if (emission.lane[6] != kInquiryEmissionSurface ||
      emission.lane[2] >= live_record_capacity(state)) {
    ++state->open_inquiry_surface_decline_bad_kind;
    return false;
  }
  Record& inquiry = state->records[inquiry_slot];
  const Record& constructor = state->records[emission.lane[2]];
  const std::uint32_t suspended_slot =
      cross_contact::find_trajectory_by_owner(state, inquiry.lane[2]);
  // Group 1: the captured constructor reference (identity, owner, revision,
  // authority) must still be exactly the Record this emission was opened
  // against.
  if (inquiry.reserved[1] != emission.lane[2] ||
      constructor.matter_q8 == 0u ||
      constructor.lane[0] != kFormOpenInquiryConstructor ||
      constructor.lane[1] != emission.lane[4] ||
      constructor.revision != emission.lane[5] ||
      !inquiry_constructor_authoritative(state, emission.lane[2])) {
    ++state->open_inquiry_surface_decline_constructor_stale;
    return false;
  }
  // Group 2: the suspended yielded trajectory this emission is surfacing
  // must still exist and still be at the exact prefix length recorded when
  // the inquiry opened.
  if (suspended_slot == kInvalid ||
      state->records[suspended_slot].lane[3] == 0u ||
      state->records[suspended_slot].lane[2] != inquiry.lane[3]) {
    ++state->open_inquiry_surface_decline_suspended_stale;
    return false;
  }
  // Group 3: the exact-fork snapshot check on its own, isolated because it
  // is the same class of "must match bit-for-bit across revisions" check
  // that was the site of the earlier inquiry-count investigation's finding.
  if (!inquiry_has_exact_snapshots(state, inquiry, state->records[suspended_slot],
                                   true, nullptr)) {
    ++state->open_inquiry_surface_decline_snapshot_mismatch;
    return false;
  }
  if (emission.lane[3] >= constructor.lane[2]) {
    ++state->open_inquiry_surface_decline_exhausted;
    return false;
  }
  std::uint32_t kind = 0u;
  std::uint32_t value = 0u;
  if (!constructor_term_at(state, constructor, emission.lane[3], &kind, &value)) {
    ++state->open_inquiry_surface_decline_term_lookup_failed;
    return false;
  }
  std::uint32_t word = value;
  if (kind != kTermLiteral) {
    AlternativeSnapshot alternative{};
    const std::uint32_t competition_cursor_before = emission.reserved[1];
    bool competition_admitted = distributed_competition_step(
        state, state->records[suspended_slot], kind, &emission.reserved[1],
        &alternative);
    if (!competition_admitted &&
        (kind == kTermFirstAlternative || kind == kTermSecondAlternative)) {
      competition_admitted = distributed_competition_fallback_exact(
          state, state->records[suspended_slot], kind, &emission.reserved[1],
          &alternative);
    }
    if (!competition_admitted) {
      if (emission.reserved[1] != competition_cursor_before) {
        ++emission.revision;
        ++state->revision;
        refresh_receipt(state);
      }
      ++state->open_inquiry_surface_decline_alternative_path_failed;
      return false;
    }
    std::uint32_t label_length = 0u;
    std::uint32_t label_digest = 0u;
    if (!alternative_continuation_extent_for_emission(
            state, alternative, inquiry.lane[3], &label_length, &label_digest) ||
        emission.reserved[0] >= label_length ||
        !alternative_continuation_word_at(state, alternative, inquiry.lane[3],
                                          emission.reserved[0], &word)) {
      ++state->open_inquiry_surface_decline_alternative_path_failed;
      return false;
    }
    ++emission.reserved[0];
    if (emission.reserved[0] < label_length) {
      if (!append_trajectory_word(state, word, true)) {
        ++state->open_inquiry_surface_decline_append_failed;
        return false;
      }
      state->generated_word = word;
      state->generated_word_valid = 1u;
      state->generated_locus = emission.lane[2];
      state->active_locus = emission.lane[2];
      inquiry.lane[7] |= kInquirySurfaceEmitted;
      ++inquiry.revision;
      ++state->revision;
      refresh_receipt(state);
      ++state->open_inquiry_surface_word_emitted;
      return true;
    }
    emission.reserved[0] = 0u;
    emission.reserved[1] = 0u;
  }
  if (!append_trajectory_word(state, word, true)) {
    ++state->open_inquiry_surface_decline_append_failed;
    return false;
  }
  state->generated_word = word;
  state->generated_word_valid = 1u;
  state->generated_locus = emission.lane[2];
  state->active_locus = emission.lane[2];
  ++emission.lane[3];
  ++emission.revision;
  inquiry.lane[7] |= kInquirySurfaceEmitted;
  ++inquiry.revision;
  if (emission.lane[3] == constructor.lane[2]) clear_record(&emission);
  ++state->revision;
  refresh_receipt(state);
  ++state->open_inquiry_surface_word_emitted;
  return true;
}
