// RWR24 open-inquiry scheduling diagnostics (0X1-163/0X1-206). Extracted from
// ResidentRewriteState's inline field list purely to keep
// causal_rewrite_universe.cuh under the repository's 2,000-logical-line
// active-file ceiling (0x1-monolith-ratchet); this file is included exactly
// once, textually, in the middle of ResidentRewriteState's member list, so
// every field below is an ordinary flat member of that struct -- there is no
// nested struct, no second state owner, and no change to any existing call
// site's `state->field` access.
//
// Three related but distinct construction/emission stages are counted here,
// accumulated across separate investigation sessions on the same 0X1-163/
// 0X1-206 stall:
//   - open_inquiry_construction_* / open_inquiry_decline_*: whether
//     open_from_yielded_version_space (bcc32_resident_open_inquiry.cuh)
//     admits a brand-new OpenInquiry Record from a yielded trajectory.
//   - oi_capture_*/oi_bind_*/oi_settle_*/oi_reactivate_*/
//     oi_reply_continuation_*: the five ordered close-work stages between
//     admission and a reply-continuation word reaching egress (capturing
//     the teacher's surface, binding a fresh external reply, settling it,
//     reactivating the suspended trajectory, and advance_surface_once's
//     reply-continuation branch itself).
//   - open_inquiry_surface_*: whether advance_surface_once
//     (bcc32_resident_open_inquiry_emission.inl), invoked every epoch via
//     run_open_inquiry_device, actually emits the next surface (question)
//     word of an already-admitted inquiry's Emission record.
// All three were instrumented because the organism-visible stall moved the
// same way each time: wait_for_inquiry_count(1) passes (construction
// admits), and the next wait, wait_for_generated_word(a), still stalls --
// each session localized a different candidate cause without yet landing a
// fix, and the three counter families are complementary evidence about the
// same still-open gap, not competing theories.
//
// All fields are cumulative event counters, exactly like the causal_germline
// suppression counters above this include: each is incremented once, in
// place, at the exact branch that fired, and the receipt exposes their live
// value verbatim every call. Two real-time-stamped samples can be diffed to
// see which branch (if any) fired between them, distinguishing "the epoch
// scheduler never reached this call" from "it reached it and declined at a
// specific precondition". Passive instrumentation only; nothing reads these
// to select, gate, or advance construction.
std::uint32_t open_inquiry_construction_attempts = 0u;
std::uint32_t open_inquiry_decline_active_inquiry = 0u;
std::uint32_t open_inquiry_decline_no_suspended_trajectory = 0u;
std::uint32_t open_inquiry_decline_not_wholly_external = 0u;
std::uint32_t open_inquiry_decline_not_yielded = 0u;
std::uint32_t open_inquiry_decline_already_open = 0u;
std::uint32_t open_inquiry_decline_fork_failed = 0u;
std::uint32_t open_inquiry_decline_multi_constructor = 0u;
std::uint32_t open_inquiry_decline_free_records = 0u;
std::uint32_t open_inquiry_decline_owner_failed = 0u;
std::uint32_t open_inquiry_construction_admitted = 0u;
// Reply-continuation close-work stage diagnostics (0X1-163/0X1-206, sibling
// session): the five ordered stages between admission and a
// reply-continuation word reaching egress.
std::uint32_t oi_capture_surface_attempts = 0u;
// capture_teacher_surface_before_end's own combined-|| guard, decomposed
// into one counter per precondition. kInvalid on either side of that guard
// conflates "no matching candidate" with "more than one -- ambiguous"; the
// first four counters below separate them (a local, read-only
// reclassification only, per the sibling session's own comment).
std::uint32_t oi_capture_decline_no_active_inquiry = 0u;
std::uint32_t oi_capture_decline_ambiguous_inquiry = 0u;
std::uint32_t oi_capture_decline_no_current_trajectory = 0u;
std::uint32_t oi_capture_decline_ambiguous_trajectory = 0u;
std::uint32_t oi_capture_decline_already_progressed = 0u;
std::uint32_t oi_capture_decline_reserved_pending = 0u;
std::uint32_t oi_capture_decline_surface_owner_is_suspended = 0u;
std::uint32_t oi_capture_decline_not_wholly_external = 0u;
std::uint32_t oi_capture_decline_surface_zero_length = 0u;
std::uint32_t oi_capture_decline_surface_too_long = 0u;
std::uint32_t oi_capture_decline_insufficient_free_records = 0u;
std::uint32_t oi_capture_surface_admitted = 0u;
std::uint32_t oi_bind_reply_attempts = 0u;
std::uint32_t oi_bind_reply_admitted = 0u;
std::uint32_t oi_settle_reply_attempts = 0u;
// settle_bound_reply's (bcc32_resident_close_work.cuh, kCloseWorkSettleInquiry)
// own combined-|| guards, decomposed into one counter per precondition, same
// technique as the oi_capture_decline_* set above (0X1-163/0X1-206, this
// session). settle_bound_reply re-derives unique_active_inquiry/
// unique_current_trajectory exactly like capture_teacher_surface_before_end
// does, so the first four counters below mirror that function's own
// no-candidate-vs-ambiguous reclassification verbatim, over the same two
// helper functions. The remaining counters below decompose, in the same
// clause order with the same short-circuit-preserving semantics, settle's
// three further combined guards (the inquiry/reply-shape guard, the
// suspended-trajectory-shape guard, and the post-loop alternative/witness/
// emission consensus guard) plus the reply-witness malformed-record guard
// that returns false mid-loop.
std::uint32_t oi_settle_decline_no_active_inquiry = 0u;
std::uint32_t oi_settle_decline_ambiguous_inquiry = 0u;
std::uint32_t oi_settle_decline_no_current_trajectory = 0u;
std::uint32_t oi_settle_decline_ambiguous_trajectory = 0u;
std::uint32_t oi_settle_decline_not_reply_bound = 0u;
std::uint32_t oi_settle_decline_already_settled = 0u;
std::uint32_t oi_settle_decline_not_awaiting_reply = 0u;
std::uint32_t oi_settle_decline_selected_owner_zero = 0u;
std::uint32_t oi_settle_decline_selected_owner_invalid = 0u;
std::uint32_t oi_settle_decline_selected_revision_zero = 0u;
std::uint32_t oi_settle_decline_selected_revision_invalid = 0u;
std::uint32_t oi_settle_decline_reply_matter_zero = 0u;
std::uint32_t oi_settle_decline_reply_not_trajectory_form = 0u;
std::uint32_t oi_settle_decline_reply_owner_zero = 0u;
std::uint32_t oi_settle_decline_reply_owner_invalid = 0u;
std::uint32_t oi_settle_decline_reply_length_zero = 0u;
std::uint32_t oi_settle_decline_reply_not_current = 0u;
std::uint32_t oi_settle_decline_no_suspended_trajectory = 0u;
std::uint32_t oi_settle_decline_suspended_lane3_zero = 0u;
std::uint32_t oi_settle_decline_suspended_owner_mismatch = 0u;
std::uint32_t oi_settle_decline_suspended_not_open_inquiry = 0u;
std::uint32_t oi_settle_decline_witness_owner_mismatch = 0u;
std::uint32_t oi_settle_decline_witness_selected_owner_mismatch = 0u;
std::uint32_t oi_settle_decline_witness_selected_revision_mismatch = 0u;
std::uint32_t oi_settle_decline_witness_reply_revision_mismatch = 0u;
std::uint32_t oi_settle_decline_witness_reply_length_mismatch = 0u;
std::uint32_t oi_settle_decline_witness_reply_tail_mismatch = 0u;
std::uint32_t oi_settle_decline_witness_not_external = 0u;
std::uint32_t oi_settle_decline_witness_revision_not_one = 0u;
std::uint32_t oi_settle_decline_alternative_count = 0u;
std::uint32_t oi_settle_decline_selected_binding_count = 0u;
std::uint32_t oi_settle_decline_selected_consequence_invalid = 0u;
std::uint32_t oi_settle_decline_witness_count = 0u;
std::uint32_t oi_settle_decline_witness_consequence_mismatch = 0u;
std::uint32_t oi_settle_decline_emission_count = 0u;
std::uint32_t oi_settle_decline_selected_emission_count = 0u;
std::uint32_t oi_settle_reply_admitted = 0u;
std::uint32_t oi_reactivate_attempts = 0u;
std::uint32_t oi_reactivate_admitted = 0u;
std::uint32_t oi_reply_continuation_attempts = 0u;
std::uint32_t oi_reply_continuation_admitted = 0u;
// advance_surface_once surface-emission diagnostics (0X1-163/0X1-206, this
// session): per-branch attempt/decline/emit outcomes for the surface-word
// emission path that runs after an inquiry has already been admitted above.
std::uint32_t open_inquiry_surface_attempts = 0u;
std::uint32_t open_inquiry_surface_decline_ambiguous_emission = 0u;
std::uint32_t open_inquiry_surface_decline_no_emission = 0u;
std::uint32_t open_inquiry_surface_decline_no_inquiry_header = 0u;
std::uint32_t open_inquiry_surface_reply_dispatch = 0u;
std::uint32_t open_inquiry_surface_decline_bad_kind = 0u;
std::uint32_t open_inquiry_surface_decline_constructor_stale = 0u;
std::uint32_t open_inquiry_surface_decline_suspended_stale = 0u;
std::uint32_t open_inquiry_surface_decline_snapshot_mismatch = 0u;
std::uint32_t open_inquiry_surface_decline_exhausted = 0u;
std::uint32_t open_inquiry_surface_decline_term_lookup_failed = 0u;
std::uint32_t open_inquiry_surface_decline_alternative_path_failed = 0u;
std::uint32_t open_inquiry_surface_decline_append_failed = 0u;
std::uint32_t open_inquiry_surface_word_emitted = 0u;
// settle_constructor_from_complete_episodes diagnostics (0X1-163, this
// session): once wait_for_generated_word(a)/(d)/(e) all pass and every
// admitted inquiry resumes (0X1-163-capture-surface-single-retire landed
// that fix), wait_for(constructors==1) is the next organism-visible stall,
// with rewrite_open_inquiry_constructors==0 throughout.
//
// settle_constructor_from_complete_episodes has *two* call sites
// (bcc32_resident_rewrite_epoch_phases.inl's per-epoch generation phase, and
// bcc32_resident_open_inquiry_emission.inl's advance_reply_continuation_once,
// via settle_completed_training_constructor_once). Live evidence from this
// session (three GPU runs, byte-identical) shows oi_ctor_gate_eligible and
// oi_ctor_gate_resume_observed both stay 0 for the whole run -- the epoch-
// phase call site's own gate (generated_this_epoch && !inquiry_generated &&
// !revision_reader_generated && run_observe_resumed_consequence_device(...),
// decomposed below into these same two counters, preserving the exact
// original short-circuit call pattern with no new call or side effect) never
// fires -- while oi_ctor_settle_attempts reaches 3. So the live path is
// entirely the *other* call site: advance_reply_continuation_once's own
// inline `if (observe_resumed_consequence(state))
// settle_completed_training_constructor_once(state);` at the moment a reply
// continuation finishes emitting, per that file's own comment ("the only
// resident opportunity to settle this inquiry"). The epoch-phase call site
// counters are kept as-is (they remain correct, structurally-dead-in-this-
// run evidence, not a bug in themselves) and settle's own body is decomposed
// one counter per decline branch below, same technique as the
// oi_settle_decline_* set above.
std::uint32_t oi_ctor_gate_eligible = 0u;
std::uint32_t oi_ctor_gate_resume_observed = 0u;
std::uint32_t oi_ctor_settle_attempts = 0u;
std::uint32_t oi_ctor_settle_decline_already_authoritative = 0u;
std::uint32_t oi_ctor_settle_decline_episode_overflow = 0u;
std::uint32_t oi_ctor_settle_decline_insufficient_episodes = 0u;
std::uint32_t oi_ctor_settle_decline_conflicting_template = 0u;
std::uint32_t oi_ctor_settle_decline_no_template = 0u;
std::uint32_t oi_ctor_settle_decline_insufficient_free_records = 0u;
std::uint32_t oi_ctor_settle_decline_owner_failed = 0u;
std::uint32_t oi_ctor_settle_decline_header_alloc_failed = 0u;
std::uint32_t oi_ctor_settle_decline_term_alloc_failed = 0u;
std::uint32_t oi_ctor_settle_decline_witness_alloc_failed = 0u;
std::uint32_t oi_ctor_settle_decline_final_check_failed = 0u;
std::uint32_t oi_ctor_settle_admitted = 0u;
// Live snapshot (last value observed at the most recent attempt, not a
// cumulative count), same convention as rewrite_open_inquiry_prefix_flags
// above: the number of episode_complete_valid slots found on the last
// settle_constructor_from_complete_episodes call, to distinguish "0/1/2 of 3
// complete episodes seen so far" from "3 seen but the template/allocation
// stage declined".
std::uint32_t oi_ctor_settle_last_episode_count = 0u;
// episode_complete_valid diagnostics (0X1-163, this session): three live GPU
// runs all show oi_ctor_settle_decl_insuff_ep=3 (every settle attempt
// declines on "fewer than 3 complete episodes") with
// oi_ctor_settle_last_episode_count=2 on the final attempt -- i.e. even
// though all three inquiries eventually resume (rewrite_open_inquiry_resumed
// reaches 3 in the receipt), never more than two of them are simultaneously
// recognized as a "complete episode" by episode_complete_valid at any single
// settle attempt. These counters decompose episode_complete_valid's own
// combined guards, one counter per precondition in source clause order, same
// technique as the oi_settle_decline_* and oi_ctor_settle_decline_* sets
// above, to localize which specific precondition an otherwise-eligible
// episode fails. episode_complete_valid has three call sites in the tree (
// the settle_constructor_from_complete_episodes scan these counters were
// added to trace, plus two guard checks inside derive_continuation_template/
// derive_template in bcc32_resident_open_inquiry.cuh); these counters are
// cumulative across all three call sites, not scoped to the settle scan
// alone -- a caveat to read alongside oi_ctor_settle_last_episode_count
// rather than a precise per-call-site breakdown.
//
// episode_complete_valid takes a `const ResidentRewriteState*` (it is a pure
// validity check, deliberately callable from const contexts) so these
// counters are declared `mutable`, the standard C++ escape for read-only-
// from-the-caller's-view bookkeeping that does not affect any logical state
// the function's const contract promises not to change -- exactly the same
// passive-instrumentation contract as every counter above, just spelled
// through a const member function instead of a non-const one.
mutable std::uint32_t oi_episode_complete_attempts = 0u;
mutable std::uint32_t oi_episode_complete_decline_not_open_inquiry_form = 0u;
mutable std::uint32_t oi_episode_complete_decline_flags_incomplete = 0u;
mutable std::uint32_t oi_episode_complete_decline_lane2_invalid = 0u;
mutable std::uint32_t oi_episode_complete_decline_lane3_zero = 0u;
mutable std::uint32_t oi_episode_complete_decline_lane4_not_two = 0u;
mutable std::uint32_t oi_episode_complete_decline_lane5_invalid = 0u;
mutable std::uint32_t oi_episode_complete_decline_lane6_invalid = 0u;
mutable std::uint32_t oi_episode_complete_decline_reserved1_not_invalid = 0u;
mutable std::uint32_t oi_episode_complete_decline_reserved0_zero = 0u;
mutable std::uint32_t oi_episode_complete_decline_reserved0_too_long = 0u;
mutable std::uint32_t oi_episode_complete_decline_surface_count_mismatch = 0u;
mutable std::uint32_t
    oi_episode_complete_decline_surface_source_owner_invalid = 0u;
mutable std::uint32_t oi_episode_complete_decline_reply_source_owner_invalid =
    0u;
mutable std::uint32_t
    oi_episode_complete_decline_surface_word_lookup_failed = 0u;
mutable std::uint32_t oi_episode_complete_decline_alternative_not_grounded =
    0u;
mutable std::uint32_t
    oi_episode_complete_decline_alternative_consensus_failed = 0u;
mutable std::uint32_t
    oi_episode_complete_decline_alternatives_lookup_failed = 0u;
mutable std::uint32_t oi_episode_complete_decline_reply_witness_malformed =
    0u;
mutable std::uint32_t oi_episode_complete_decline_resume_witness_malformed =
    0u;
mutable std::uint32_t oi_episode_complete_decline_witness_count_mismatch =
    0u;
mutable std::uint32_t oi_episode_complete_valid_count = 0u;
// inquiry_reply_source_owner diagnostics (0X1-163, this session): three live
// GPU runs show episode_complete_valid's decline is entirely
// reply_source_owner_invalid (oi_episode_complete_decline_
// reply_source_owner_invalid==3, surface_source_owner_invalid==0 on every
// run), so this decomposes inquiry_reply_source_owner's own body, one
// counter per branch, same technique and same mutable-for-a-const-function
// rationale as the episode_complete_valid set above. inquiry_reply_source_
// owner has one other call site (bcc32_resident_open_inquiry.cuh:651, inside
// an unrelated combined guard this session did not instrument), so these
// counters are cumulative across both call sites, same caveat as above.
mutable std::uint32_t oi_reply_source_owner_attempts = 0u;
mutable std::uint32_t oi_reply_source_owner_decline_ambiguous = 0u;
mutable std::uint32_t oi_reply_source_owner_decline_witness_owner_bad = 0u;
mutable std::uint32_t oi_reply_source_owner_decline_witness_lane6_zero = 0u;
mutable std::uint32_t oi_reply_source_owner_decline_witness_not_external =
    0u;
mutable std::uint32_t oi_reply_source_owner_decline_reply_terms_invalid =
    0u;
mutable std::uint32_t oi_reply_source_owner_decline_no_witness_found = 0u;
mutable std::uint32_t oi_reply_source_owner_found = 0u;
// reply_terms_valid diagnostics (0X1-163, this session): three live GPU runs
// show inquiry_reply_source_owner's decline is entirely oi_rso_decl_terms_
// inv (reply_terms_valid returning false), so this decomposes reply_terms_
// valid's own body -- its fast literal-shape path and its fallback
// authoritative-Program path -- one counter per branch, same technique and
// same mutable-for-a-const-function rationale as the sets above.
// reply_terms_valid has three other call sites (bcc32_resident_open_inquiry.
// cuh:839, 1317, 1345, none instrumented this session), so these counters
// are cumulative across all four call sites, same caveat as above.
mutable std::uint32_t oi_reply_terms_attempts = 0u;
mutable std::uint32_t oi_reply_terms_decline_witness_guard = 0u;
mutable std::uint32_t oi_reply_terms_decline_term_lookup_failed = 0u;
mutable std::uint32_t oi_reply_terms_decline_term_lane4_owner_mismatch = 0u;
mutable std::uint32_t oi_reply_terms_decline_term_lane5_revision_mismatch =
    0u;
mutable std::uint32_t oi_reply_terms_decline_term_lane6_length_mismatch =
    0u;
mutable std::uint32_t oi_reply_terms_decline_term_not_external = 0u;
mutable std::uint32_t oi_reply_terms_decline_term_revision_not_one = 0u;
// Live snapshot (last value observed, not cumulative), same convention as
// rewrite_open_inquiry_prefix_flags: the actual term.revision value and slot
// index seen the last time oi_reply_terms_decline_term_revision_not_one
// fired, to distinguish "never incremented (0)" from "incremented more than
// once (>1)" without a second build/run cycle.
mutable std::uint32_t oi_reply_terms_last_bad_term_revision = 0u;
mutable std::uint32_t oi_reply_terms_last_bad_term_slot = 0u;
mutable std::uint32_t oi_reply_terms_decline_term_count_mismatch = 0u;
mutable std::uint32_t oi_reply_terms_fastpath_matched = 0u;
mutable std::uint32_t oi_reply_terms_decline_program_ambiguous = 0u;
mutable std::uint32_t oi_reply_terms_decline_program_not_found = 0u;
mutable std::uint32_t oi_reply_terms_decline_program_exact_mismatch = 0u;
mutable std::uint32_t oi_reply_terms_program_exact_matched = 0u;
