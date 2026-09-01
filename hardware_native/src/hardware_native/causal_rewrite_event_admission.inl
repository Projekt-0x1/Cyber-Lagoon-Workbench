// Resident raw-event admission owns the single history revision transition
// for a boundary event.  The enclosing universe supplies the state, record,
// trajectory, and observer helpers; this extraction keeps the over-ceiling
// universe header as a composition point without introducing another clock.
BCC32_REWRITE_HD inline void consume_rewrite_event(
    ResidentRewriteEngine engine, RawRewriteEvent event,
    bool update_receipt = true,
    bool preserve_unclaimed_prefix = false) {
  ResidentRewriteState* state = engine.state;
  if (state == nullptr || state->fault != 0u) return;
  ++state->admitted_events;
  // Every admitted boundary event changes resident history, even when it
  // only advances a trajectory/termination latch and does not allocate a new
  // description. The production passive census uses this revision as its
  // dirty bit; without the event-level increment, raw contact can change
  // resident matter while revision remains equal to the last census and the
  // public lineage receipt is never republished.
  ++state->revision;
  clear_motor(state);
  clear_generated_word(state);
  age_partials(state);

  if (event.reserved == kEventFramePause) {
    yield_program_trajectory(state);
  } else if (event.reserved == kEventFrameEnd) {
#if defined(__CUDA_ARCH__)
    // Suppress every nested close-time observer refresh until the root quiet
    // epoch has also settled recursive Program factorization.
    state->cross_context_factor_pending = 1u;
#endif
    close_program_trajectory(state, preserve_unclaimed_prefix);
    // Learned Programs are ordinary resident evidence for deeper Programs.
    // CUDA schedules this on the root quiet epoch after nested ingress frames
    // unwind. Host/direct execution can settle immediately without that stack.
#if defined(__CUDA_ARCH__)
    if (state->fault != 0u) state->cross_context_factor_pending = 0u;
#else
    (void)cross_context::cross_context_factor_all_mature_programs(state);
#endif
    const std::uint32_t slot = find_sequence(state);
    if (slot != kInvalid) {
      state->records[slot].lane[5] = 2u;
      close_sequence_if_ready(state);
    }
  } else if (event.valid != 0u) {
    // Admit the raw word into the continuing resident trajectory before any
    // derived observer work. A physical page/record failure therefore cannot
    // leave descriptions or sequence matter claiming a word that never
    // crossed the canonical resident boundary.
    if (!append_trajectory_word(state, event.value, false)) {
      if (update_receipt) refresh_receipt(state);
      return;
    }
    apply_event_to_descriptions(state, event.value);
    append_sequence_value(state, event.value);
  } else {
    const std::uint32_t slot = find_sequence(state);
    if (slot != kInvalid) {
      Record& sequence = state->records[slot];
      if (sequence.lane[5] != 0xffffffffu) ++sequence.lane[5];
      ++sequence.revision;
      close_sequence_if_ready(state);
    }
  }
  if (update_receipt) refresh_receipt(state);
}
