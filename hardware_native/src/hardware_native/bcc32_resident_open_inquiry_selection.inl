// Resident target selection is deliberately narrower than route reachability:
// the target must already be present in one retained pending-means trajectory.
// This include owns no clock, writer, or state outside ordinary Records.
inline constexpr std::uint32_t kFormOpenInquiryTargetWitness = 0x1d9b7f43u;
inline constexpr std::uint32_t kInquiryResidentTargetWitness = 0x52544754u;

#if defined(__CUDACC__)
#define BCC32_OPEN_INQUIRY_TARGET_DISPATCH \
  [[maybe_unused]] static BCC32_OPEN_INQUIRY_HD __noinline__
#else
#define BCC32_OPEN_INQUIRY_TARGET_DISPATCH BCC32_OPEN_INQUIRY_HD inline
#endif

// Target witness lanes:
// [1] inquiry owner; [2,3] pending trajectory owner/revision;
// [4,5] generated producer owner/revision; [6] resident target word;
// [7] fixed resident-target provenance.  reserved[0,1] retain selected
// alternative owner/revision.  The exact fields make a later validation
// independent of Record allocation order.
BCC32_OPEN_INQUIRY_TARGET_DISPATCH bool resident_target_witness_live(
    const ResidentRewriteState* state, std::uint32_t witness_slot) {
  if (state == nullptr || witness_slot >= live_record_capacity(state))
    return false;
  const Record& witness = state->records[witness_slot];
  if (witness.matter_q8 == 0u || witness.lane[0] != kFormOpenInquiryTargetWitness ||
      witness.lane[1] == 0u || witness.lane[1] == kInvalid || witness.lane[2] == 0u ||
      witness.lane[2] == kInvalid || witness.lane[3] == 0u || witness.lane[4] == 0u ||
      witness.lane[4] == kInvalid || witness.lane[5] == 0u || witness.reserved[0] == 0u ||
      witness.reserved[0] == kInvalid || witness.reserved[1] == 0u ||
      witness.lane[7] != kInquiryResidentTargetWitness)
    return false;

  const std::uint32_t inquiry_slot =
      unique_header_by_owner(state, kFormOpenInquiry, witness.lane[1]);
  if (inquiry_slot == kInvalid)
    return false;
  const Record& inquiry = state->records[inquiry_slot];
  if ((inquiry.lane[7] & (kInquiryReplyBound | kInquirySettled)) !=
          (kInquiryReplyBound | kInquirySettled) ||
      inquiry.lane[2] == witness.lane[2] ||
      inquiry.lane[5] != witness.reserved[0] || inquiry.lane[6] != witness.reserved[1])
    return false;

  std::uint32_t witness_count = 0u;
  std::uint32_t external_reply_count = 0u;
  for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
    const Record& record = state->records[slot];
    if (record.matter_q8 == 0u)
      continue;
    if (record.lane[0] == kFormOpenInquiryTargetWitness && record.lane[1] == inquiry.lane[1])
      ++witness_count;
    if (record.lane[0] == kFormOpenInquiryReplyWitness && record.lane[1] == inquiry.lane[1])
      ++external_reply_count;
  }
  if (witness_count != 1u || external_reply_count != 0u)
    return false;

  std::uint32_t pending_slot = kInvalid;
  std::uint32_t pending_count = 0u;
  for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
    const Record& header = state->records[slot];
    if (header.matter_q8 == 0u || header.lane[0] != kFormTrajectory || header.lane[3] != 1u ||
        (header.lane[7] & pending_means::kTrajectoryPendingMeans) == 0u)
      continue;
    if (header.lane[1] == inquiry.lane[2]) return false;
    ++pending_count;
    if (header.lane[1] == witness.lane[2] && header.revision == witness.lane[3])
      pending_slot = slot;
  }
  if (pending_count != 1u || pending_slot == kInvalid)
    return false;
  const Record& pending = state->records[pending_slot];
  std::uint32_t target = 0u;
  std::uint32_t producer_slot = kInvalid;
  if (!pending_means::pending_header_target(state, pending, &target, &producer_slot) ||
      target != witness.lane[6] || producer_slot >= live_record_capacity(state))
    return false;
  const Record& producer = state->records[producer_slot];
  if (!resident_program_authoritative(state, producer_slot) ||
      producer.lane[1] != witness.lane[4] || producer.revision != witness.lane[5])
    return false;
  if (pending_means::immature_actionable_fixed_route(state, producer_slot, target) !=
      pending_means::Predicate::yes)
    return false;

  const std::uint32_t suspended_slot =
      cross_contact::find_trajectory_by_owner(state, inquiry.lane[2]);
  if (suspended_slot == kInvalid)
    return false;
  Record prefix = state->records[suspended_slot];
  prefix.lane[2] = inquiry.lane[3];
  if (!inquiry_has_exact_snapshots(state, inquiry, prefix, true, nullptr, false))
    return false;
  std::uint32_t selected_count = 0u;
  for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
    const Record& binding = state->records[slot];
    if (binding.matter_q8 == 0u || binding.lane[0] != kFormOpenInquiryAlternative ||
        binding.lane[1] != inquiry.lane[1] || binding.lane[4] != target)
      continue;
    std::uint32_t live_program = kInvalid;
    if (!snapshot_still_live(state, prefix, binding, &live_program) ||
        binding.lane[2] != witness.reserved[0] || binding.lane[3] != witness.reserved[1])
      return false;
    ++selected_count;
  }
  return selected_count == 1u;
}

BCC32_OPEN_INQUIRY_TARGET_DISPATCH bool bind_one_resident_target_without_external_reply(
    ResidentRewriteState* state) {
  if (state == nullptr || state->fault != 0u)
    return false;
  const std::uint32_t inquiry_slot = unique_active_inquiry(state);
  const std::uint32_t question_slot = unique_current_trajectory(state);
  if (inquiry_slot == kInvalid || question_slot == kInvalid)
    return false;
  Record& inquiry = state->records[inquiry_slot];
  const Record& question = state->records[question_slot];
  if ((inquiry.lane[7] & (kInquirySurfaceEmitted | kInquiryReplyBound | kInquirySettled |
                          kInquiryResumed)) != kInquirySurfaceEmitted ||
      inquiry.reserved[0] != 0u || inquiry.reserved[1] == kInvalid || question.lane[5] != 0u ||
      (question.lane[7] & kTrajectoryHasGenerated) == 0u ||
      !mixed_provenance::tagged_history(state, question))
    return false;
  const std::uint32_t suspended_slot =
      cross_contact::find_trajectory_by_owner(state, inquiry.lane[2]);
  const std::uint32_t constructor_slot = inquiry.reserved[1];
  if (suspended_slot == kInvalid || constructor_slot >= live_record_capacity(state) ||
      state->records[constructor_slot].matter_q8 == 0u ||
      state->records[constructor_slot].lane[0] != kFormOpenInquiryConstructor ||
      state->records[constructor_slot].lane[1] != inquiry.lane[5] ||
      state->records[constructor_slot].revision != inquiry.lane[6] ||
      !inquiry_constructor_authoritative(state, constructor_slot))
    return false;
  std::uint32_t expected_length = 0u;
  if (!emitted_surface_length(state, inquiry, state->records[constructor_slot],
                              state->records[suspended_slot], &expected_length) ||
      question.lane[2] != expected_length)
    return false;
  for (std::uint32_t index = 0u; index < question.lane[2]; ++index) {
    mixed_provenance::Origin origin = mixed_provenance::Origin::external;
    std::uint32_t producer = kInvalid;
    if (!mixed_provenance::origin_at(state, question, index, &origin, &producer) ||
        origin != mixed_provenance::Origin::generated || producer != constructor_slot)
      return false;
  }

  std::uint32_t pending_slot = kInvalid;
  std::uint32_t pending_count = 0u;
  std::uint32_t target = 0u;
  std::uint32_t producer_slot = kInvalid;
  for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
    const Record& header = state->records[slot];
    if (header.matter_q8 == 0u || header.lane[0] != kFormTrajectory || header.lane[3] != 1u ||
        (header.lane[7] & pending_means::kTrajectoryPendingMeans) == 0u)
      continue;
    if (header.lane[1] == inquiry.lane[2]) return false;
    std::uint32_t candidate_target = 0u;
    std::uint32_t candidate_producer = kInvalid;
    if (!pending_means::pending_header_target(state, header, &candidate_target,
                                              &candidate_producer) ||
        candidate_producer >= live_record_capacity(state) ||
        !resident_program_authoritative(state, candidate_producer))
      return false;
    if (pending_means::immature_actionable_fixed_route(
            state, candidate_producer, candidate_target) != pending_means::Predicate::yes)
      return false;
    ++pending_count;
    pending_slot = slot;
    target = candidate_target;
    producer_slot = candidate_producer;
  }
  if (pending_count != 1u || pending_slot == kInvalid)
    return false;

  Record& suspended = state->records[suspended_slot];
  if (suspended.lane[3] == 0u || suspended.lane[2] != inquiry.lane[3] ||
      (suspended.lane[7] & kTrajectoryOpenInquiry) == 0u ||
      !inquiry_has_exact_snapshots(state, inquiry, suspended, true, nullptr, false))
    return false;
  std::uint32_t selected_binding = kInvalid;
  std::uint32_t selected_program = kInvalid;
  std::uint32_t selected_count = 0u;
  for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
    const Record& binding = state->records[slot];
    if (binding.matter_q8 == 0u || binding.lane[0] != kFormOpenInquiryAlternative ||
        binding.lane[1] != inquiry.lane[1] || binding.lane[4] != target)
      continue;
    std::uint32_t program = kInvalid;
    if (!snapshot_still_live(state, suspended, binding, &program))
      return false;
    ++selected_count;
    selected_binding = slot;
    selected_program = program;
  }
  if (selected_count != 1u || selected_binding == kInvalid || selected_program == kInvalid)
    return false;
  for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
    const Record& record = state->records[slot];
    if (record.matter_q8 == 0u)
      continue;
    if ((record.lane[0] == kFormOpenInquiryTargetWitness && record.lane[1] == inquiry.lane[1]) ||
        (record.lane[0] == kFormOpenInquiryReplyWitness && record.lane[1] == inquiry.lane[1]) ||
        record.lane[0] == kFormOpenInquiryEmission)
      return false;
  }
  if (free_record_count(state) < 2u)
    return false;

  const Record& pending = state->records[pending_slot];
  const Record& producer = state->records[producer_slot];
  const Record& selected = state->records[selected_binding];
  const std::uint32_t question_owner = question.lane[1];
  const std::uint32_t witness_slot = allocate_record(state);
  const std::uint32_t emission_slot = allocate_record(state);
  if (witness_slot == kInvalid || emission_slot == kInvalid)
    return false;
  Record& witness = state->records[witness_slot];
  witness.lane[0] = kFormOpenInquiryTargetWitness;
  witness.lane[1] = inquiry.lane[1];
  witness.lane[2] = pending.lane[1];
  witness.lane[3] = pending.revision;
  witness.lane[4] = producer.lane[1];
  witness.lane[5] = producer.revision;
  witness.lane[6] = target;
  witness.lane[7] = kInquiryResidentTargetWitness;
  witness.reserved[0] = selected.lane[2];
  witness.reserved[1] = selected.lane[3];
  ++witness.revision;
  Record& emission = state->records[emission_slot];
  emission.lane[0] = kFormOpenInquiryEmission;
  emission.lane[1] = inquiry.lane[1];
  emission.lane[2] = selected_program;
  emission.lane[3] = 0u;
  emission.lane[4] = selected.lane[2];
  emission.lane[5] = selected.lane[3];
  emission.lane[6] = kInquiryEmissionReplyContinuation;
  ++emission.revision;
  inquiry.lane[5] = selected.lane[2];
  inquiry.lane[6] = selected.lane[3];
  inquiry.lane[7] =
      (inquiry.lane[7] & ~kInquiryAwaitingReply) | kInquiryReplyBound | kInquirySettled;
  ++inquiry.revision;
  mixed_provenance::clear_provenance(state, question_owner);
  clear_trajectory(state, question_slot);
  ++state->revision;
  refresh_receipt(state);
  return reactivate_settled_suspended_after_end(state);
}

#undef BCC32_OPEN_INQUIRY_TARGET_DISPATCH
