// Resident open-inquiry origination and completed-evidence validation.
// Included inside the open_inquiry namespace after core and alternative helpers.

// Completed inquiry evidence must survive withdrawal of the transient prefix
// trajectory.  Revalidate the exact live Program snapshot directly: the two
// bound Programs retain the prefix and continuation matter that the physical
// inquiry observed, while the inquiry header retains its witnessed extent.
BCC32_OPEN_INQUIRY_HD inline bool snapshot_program_still_grounded(
    const ResidentRewriteState* state, const Record& binding,
    std::uint32_t prefix_length, std::uint32_t* program_slot = nullptr) {
  if (program_slot != nullptr) *program_slot = kInvalid;
  if (state == nullptr || binding.matter_q8 == 0u ||
      binding.lane[0] != kFormOpenInquiryAlternative || prefix_length == 0u)
    return false;
  std::uint32_t found = kInvalid;
  for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
    const Record& program = state->records[slot];
    const bool version_space =
        (program.lane[7] & kProgramFlagVersionSpace) != 0u;
    const bool external_exact =
        (program.lane[7] & kProgramFlagPureExternalExact) != 0u;
    if (program.matter_q8 == 0u || program.lane[0] != kFormProgram ||
        program.lane[1] != binding.lane[2])
      continue;
    // VersionSpace Programs are derived, content-addressed matter. A rebuild
    // may rematerialize the same owner at a new Record revision.
    if (found != kInvalid ||
        (!version_space && program.revision != binding.lane[3]) ||
        (!version_space && !external_exact) ||
        (version_space && !version_space_program_grounded(state, slot, true)) ||
        (external_exact &&
         !pure_external_exact_program_authoritative(state, slot)) ||
        causal_product_has_live_counterevidence(state, slot) ||
        prefix_length >= program.lane[2])
      return false;
    found = slot;
  }
  if (found == kInvalid) return false;
  AlternativeSnapshot alternative{binding.lane[2], binding.lane[3], 0u,
                                  found};
  std::uint32_t word = 0u;
  if (!alternative_continuation_word_at(
          state, alternative, prefix_length, 0u, &word) ||
      word != binding.lane[4])
    return false;
  if (program_slot != nullptr) *program_slot = found;
  return true;
}

BCC32_OPEN_INQUIRY_HD inline bool episode_alternatives(
    const ResidentRewriteState* state, const Record& inquiry,
    AlternativeSnapshot alternatives[2]);

BCC32_OPEN_INQUIRY_HD inline bool episode_complete_valid(
    const ResidentRewriteState* state, std::uint32_t episode_slot) {
  if (state == nullptr || episode_slot >= live_record_capacity(state)) return false;
  ++state->oi_episode_complete_attempts;
  const Record& inquiry = state->records[episode_slot];
  if (inquiry.matter_q8 == 0u || inquiry.lane[0] != kFormOpenInquiry) {
    ++state->oi_episode_complete_decline_not_open_inquiry_form;
    return false;
  }
  if ((inquiry.lane[7] & (kInquirySurfaceCaptured | kInquiryReplyBound |
                          kInquirySettled | kInquiryResumed |
                          kInquiryComplete)) !=
      (kInquirySurfaceCaptured | kInquiryReplyBound | kInquirySettled |
       kInquiryResumed | kInquiryComplete)) {
    ++state->oi_episode_complete_decline_flags_incomplete;
    return false;
  }
  if (inquiry.lane[2] == 0u || inquiry.lane[2] == kInvalid) {
    ++state->oi_episode_complete_decline_lane2_invalid;
    return false;
  }
  if (inquiry.lane[3] == 0u) {
    ++state->oi_episode_complete_decline_lane3_zero;
    return false;
  }
  if (inquiry.lane[4] != 2u) {
    ++state->oi_episode_complete_decline_lane4_not_two;
    return false;
  }
  if (inquiry.lane[5] == 0u || inquiry.lane[5] == kInvalid) {
    ++state->oi_episode_complete_decline_lane5_invalid;
    return false;
  }
  if (inquiry.lane[6] == 0u || inquiry.lane[6] == kInvalid) {
    ++state->oi_episode_complete_decline_lane6_invalid;
    return false;
  }
  if (inquiry.reserved[1] != kInvalid) {
    ++state->oi_episode_complete_decline_reserved1_not_invalid;
    return false;
  }
  if (inquiry.reserved[0] == 0u) {
    ++state->oi_episode_complete_decline_reserved0_zero;
    return false;
  }
  if (inquiry.reserved[0] > kMaximumSurfaceWords) {
    ++state->oi_episode_complete_decline_reserved0_too_long;
    return false;
  }
  std::uint32_t surface_count = 0u;
  for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
    const Record& term = state->records[slot];
    if (term.matter_q8 != 0u && term.lane[0] == kFormOpenInquirySurfaceTerm &&
        term.lane[1] == inquiry.lane[1])
      ++surface_count;
  }
  if (surface_count != inquiry.reserved[0]) {
    ++state->oi_episode_complete_decline_surface_count_mismatch;
    return false;
  }
  if (inquiry_surface_source_owner(state, inquiry) == kInvalid) {
    ++state->oi_episode_complete_decline_surface_source_owner_invalid;
    return false;
  }
  if (inquiry_reply_source_owner(state, inquiry) == kInvalid) {
    ++state->oi_episode_complete_decline_reply_source_owner_invalid;
    return false;
  }
  for (std::uint32_t index = 0u; index < surface_count; ++index) {
    std::uint32_t ignored = 0u;
    if (!inquiry_surface_word_at(state, inquiry, index, &ignored)) {
      ++state->oi_episode_complete_decline_surface_word_lookup_failed;
      return false;
    }
  }
  std::uint32_t selected_word = kInvalid;
  std::uint32_t selected_bindings = 0u;
  std::uint32_t alternative_count = 0u;
  for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
    const Record& binding = state->records[slot];
    if (binding.matter_q8 == 0u ||
        binding.lane[0] != kFormOpenInquiryAlternative ||
        binding.lane[1] != inquiry.lane[1])
      continue;
    if (!snapshot_program_still_grounded(state, binding, inquiry.lane[3])) {
      ++state->oi_episode_complete_decline_alternative_not_grounded;
      return false;
    }
    ++alternative_count;
    if (binding.lane[2] == inquiry.lane[5] &&
        binding.lane[3] == inquiry.lane[6]) {
      selected_word = binding.lane[4];
      ++selected_bindings;
    }
  }
  if (alternative_count != 2u || selected_bindings != 1u ||
      selected_word == kInvalid) {
    ++state->oi_episode_complete_decline_alternative_consensus_failed;
    return false;
  }
  AlternativeSnapshot alternatives[2]{};
  if (!episode_alternatives(state, inquiry, alternatives)) {
    ++state->oi_episode_complete_decline_alternatives_lookup_failed;
    return false;
  }
  std::uint32_t reply_count = 0u;
  std::uint32_t resume_count = 0u;
  for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
    const Record& witness = state->records[slot];
    if (witness.matter_q8 == 0u || witness.lane[1] != inquiry.lane[1]) continue;
    if (witness.lane[0] == kFormOpenInquiryReplyWitness) {
      if (witness.lane[2] == 0u || witness.lane[2] == kInvalid ||
          witness.lane[3] != inquiry.lane[5] ||
          witness.lane[4] != inquiry.lane[6] ||
          witness.lane[5] != selected_word ||
          !reply_terms_valid(state, inquiry, witness)) {
          ++state->oi_episode_complete_decline_reply_witness_malformed;
          return false;
      }
      ++reply_count;
    } else if (witness.lane[0] == kFormOpenInquiryResumeWitness) {
      if (witness.lane[2] != inquiry.lane[5] || witness.lane[3] != inquiry.lane[6] ||
          witness.lane[4] != selected_word) {
        ++state->oi_episode_complete_decline_resume_witness_malformed;
        return false;
      }
      ++resume_count;
    }
  }
  if (reply_count != 1u || resume_count != 1u) {
    ++state->oi_episode_complete_decline_witness_count_mismatch;
    return false;
  }
  ++state->oi_episode_complete_valid_count;
  return true;
}

// Starts acquisition or held-out inquiry from resident alternatives only.
BCC32_OPEN_INQUIRY_HD inline bool open_from_yielded_version_space(
    ResidentRewriteState* state, InquiryView* view = nullptr) {
  if (view != nullptr) *view = InquiryView{};
  if (state == nullptr) return false;
  // Diagnostic only (0X1-163/0X1-206): counts every epoch the scheduler
  // reached this call site at all, independent of state->fault or any
  // precondition below, so "never attempted" is distinguishable from
  // "attempted, declined at branch X" purely from the receipt.
  ++state->open_inquiry_construction_attempts;
  if (state->fault != 0u) return false;
  if (unique_active_inquiry(state) != kInvalid) {
    ++state->open_inquiry_decline_active_inquiry;
    return false;
  }
  const std::uint32_t suspended_slot = unique_current_trajectory(state);
  if (suspended_slot == kInvalid) {
    ++state->open_inquiry_decline_no_suspended_trajectory;
    return false;
  }
  Record& suspended = state->records[suspended_slot];
  if (!wholly_external_trajectory(state, suspended)) {
    ++state->open_inquiry_decline_not_wholly_external;
    return false;
  }
  if ((suspended.lane[7] & kTrajectoryWasYielded) == 0u) {
    ++state->open_inquiry_decline_not_yielded;
    return false;
  }
  if ((suspended.lane[7] & kTrajectoryOpenInquiry) != 0u) {
    ++state->open_inquiry_decline_already_open;
    return false;
  }
  AlternativeSnapshot alternatives[2]{};
  if (!collect_exact_fork(state, suspended, alternatives)) {
    ++state->open_inquiry_decline_fork_failed;
    return false;
  }

  const std::uint32_t constructor = unique_authoritative_constructor(state);
  std::uint32_t constructor_count = 0u;
  for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot)
    constructor_count += inquiry_constructor_authoritative(state, slot);
  if (constructor_count > 1u) {
    ++state->open_inquiry_decline_multi_constructor;
    return false;
  }
  const std::uint32_t needed = 3u + (constructor_count == 1u ? 1u : 0u);
  if (free_record_count(state) < needed) {
    ++state->open_inquiry_decline_free_records;
    return false;
  }
  const std::uint32_t owner = make_inquiry_owner(
      state, kFormOpenInquiry, rewrite_mix(suspended.lane[1], alternatives[0].owner,
                                            alternatives[1].owner));
  if (owner == kInvalid) {
    ++state->open_inquiry_decline_owner_failed;
    return false;
  }
  ++state->open_inquiry_construction_admitted;

  const std::uint32_t inquiry_slot = allocate_record(state);
  if (inquiry_slot == kInvalid) return false;
  Record& inquiry = state->records[inquiry_slot];
  inquiry.lane[0] = kFormOpenInquiry;
  inquiry.lane[1] = owner;
  inquiry.lane[2] = suspended.lane[1];
  inquiry.lane[3] = suspended.lane[2];
  inquiry.lane[4] = 2u;
  inquiry.lane[5] = constructor_count == 1u
                        ? state->records[constructor].lane[1]
                        : kInvalid;
  inquiry.lane[6] = constructor_count == 1u
                        ? state->records[constructor].revision
                        : kInvalid;
  inquiry.lane[7] = kInquiryAwaitingReply;
  inquiry.reserved[0] = 0u;
  inquiry.reserved[1] = constructor;
  ++inquiry.revision;
  for (std::uint32_t index = 0u; index < 2u; ++index) {
    const std::uint32_t slot = allocate_record(state);
    if (slot == kInvalid) return false;
    Record& binding = state->records[slot];
    binding.lane[0] = kFormOpenInquiryAlternative;
    binding.lane[1] = owner;
    binding.lane[2] = alternatives[index].owner;
    binding.lane[3] = alternatives[index].revision;
    binding.lane[4] = alternatives[index].consequence;
    ++binding.revision;
  }
  suspended.lane[3] = 1u;
  suspended.lane[7] |= kTrajectoryOpenInquiry;
  ++suspended.revision;
  if (constructor_count == 1u) {
    const std::uint32_t emission_slot = allocate_record(state);
    if (emission_slot == kInvalid) return false;
    Record& emission = state->records[emission_slot];
    emission.lane[0] = kFormOpenInquiryEmission;
    emission.lane[1] = owner;
    emission.lane[2] = constructor;
    emission.lane[3] = 0u;
    emission.lane[4] = state->records[constructor].lane[1];
    emission.lane[5] = state->records[constructor].revision;
    ++emission.revision;
  }
  ++state->revision;
  refresh_receipt(state);
  if (view != nullptr) {
    view->inquiry_slot = inquiry_slot;
    view->suspended_slot = suspended_slot;
    view->constructor_slot = constructor;
  }
  return true;
}
