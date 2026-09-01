inline constexpr std::uint32_t kTrajectoryOpenInquiry = 1u << 5u;
inline constexpr std::uint32_t kInquiryAwaitingReply = 1u;
inline constexpr std::uint32_t kInquirySurfaceCaptured = 1u << 1u;
inline constexpr std::uint32_t kInquirySurfaceEmitted = 1u << 2u;
inline constexpr std::uint32_t kInquiryReplyBound = 1u << 3u;
inline constexpr std::uint32_t kInquirySettled = 1u << 4u;
inline constexpr std::uint32_t kInquiryResumed = 1u << 5u;
inline constexpr std::uint32_t kInquiryComplete = 1u << 6u;
inline constexpr std::uint32_t kInquiryReplyContinuationEmitted = 1u << 7u;
inline constexpr std::uint32_t kInquiryConstructorEnabled = 1u;
inline constexpr std::uint32_t kTermLiteral = 0u;
inline constexpr std::uint32_t kTermFirstAlternative = 1u;
inline constexpr std::uint32_t kTermSecondAlternative = 2u;
inline constexpr std::uint32_t kReplyEqualsChosenAlternative = 1u;
inline constexpr std::uint32_t kInquiryExternalWitness = 0x45585431u;
inline constexpr std::uint32_t kRequiredIndependentEpisodes = 3u;
inline constexpr std::uint32_t kMaximumSurfaceWords = 32u;
inline constexpr std::uint32_t kMaximumFactorEpisodes = 8u;
inline constexpr std::uint32_t kReplyTemplatePrefix = 1u;
inline constexpr std::uint32_t kReplyTemplateSuffix = 2u;
inline constexpr std::uint32_t kInquiryEmissionSurface = 0u;
inline constexpr std::uint32_t kInquiryEmissionReplyContinuation = 1u;

BCC32_OPEN_INQUIRY_HD inline bool reply_terms_valid(
    const ResidentRewriteState* state, const Record& inquiry,
    const Record& witness, std::uint32_t* chosen_index = nullptr);

BCC32_OPEN_INQUIRY_HD inline std::uint32_t
unique_reply_continuation_emission(const ResidentRewriteState* state,
                                   const Record& inquiry) {
  if (state == nullptr) return kInvalid;
  std::uint32_t found = kInvalid;
  for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
    const Record& emission = state->records[slot];
    if (emission.matter_q8 == 0u ||
        emission.lane[0] != kFormOpenInquiryEmission ||
        emission.lane[1] != inquiry.lane[1] ||
        emission.lane[6] != kInquiryEmissionReplyContinuation)
      continue;
    if (found != kInvalid) return kInvalid;
    found = slot;
  }
  return found;
}

// The label is the exact remaining Program continuation after the yielded
// prefix, bounded by the Program's own effective term count.
BCC32_OPEN_INQUIRY_HD inline bool alternative_continuation_word_at(
    const ResidentRewriteState* state, const AlternativeSnapshot& alternative,
    std::uint32_t suspended_length, std::uint32_t index, std::uint32_t* word) {
  if (state == nullptr || word == nullptr || alternative.slot >= live_record_capacity(state))
    return false;
  const Record& program = state->records[alternative.slot];
  const bool version_space =
      (program.lane[7] & kProgramFlagVersionSpace) != 0u;
  if (program.matter_q8 == 0u || program.lane[0] != kFormProgram ||
      program.lane[1] != alternative.owner ||
      (!version_space && program.revision != alternative.revision) ||
      (version_space &&
       !version_space_program_grounded(state, alternative.slot, true)) ||
      suspended_length >= program.lane[2] ||
      index >= program.lane[2] - suspended_length)
    return false;
  std::uint32_t meta = 0u;
  if (version_space)
    return version_space_effective_program_term_at(
               state, program, suspended_length + index, word, &meta) &&
           meta == 0u;
  return resident_program_term_at(state, alternative.slot,
                                  suspended_length + index, word, &meta) &&
         meta == 0u;
}

BCC32_OPEN_INQUIRY_HD inline bool alternative_continuation_digest(
    const ResidentRewriteState* state, const AlternativeSnapshot& alternative,
    std::uint32_t suspended_length, std::uint32_t* length,
    std::uint32_t* digest) {
  if (length == nullptr || digest == nullptr ||
      alternative.slot >= live_record_capacity(state))
    return false;
  const Record& program = state->records[alternative.slot];
  if (suspended_length >= program.lane[2] ||
      program.lane[2] - suspended_length > kMaximumSurfaceWords)
    return false;
  std::uint32_t result = 0u;
  const std::uint32_t count = program.lane[2] - suspended_length;
  for (std::uint32_t index = 0u; index < count; ++index) {
    std::uint32_t word = 0u;
    if (!alternative_continuation_word_at(state, alternative, suspended_length,
                                          index, &word))
      return false;
    result = rewrite_mix(result, word, index);
  }
  *length = count;
  *digest = result;
  return true;
}

BCC32_OPEN_INQUIRY_HD inline bool alternative_continuations_equal(
    const ResidentRewriteState* state, const AlternativeSnapshot& left,
    const AlternativeSnapshot& right, std::uint32_t suspended_length) {
  std::uint32_t left_length = 0u, left_digest = 0u;
  std::uint32_t right_length = 0u, right_digest = 0u;
  if (!alternative_continuation_digest(state, left, suspended_length,
                                       &left_length, &left_digest) ||
      !alternative_continuation_digest(state, right, suspended_length,
                                       &right_length, &right_digest) ||
      left_length != right_length || left_digest != right_digest)
    return false;
  for (std::uint32_t index = 0u; index < left_length; ++index) {
    std::uint32_t left_word = 0u, right_word = 0u;
    if (!alternative_continuation_word_at(state, left, suspended_length, index,
                                          &left_word) ||
        !alternative_continuation_word_at(state, right, suspended_length, index,
                                          &right_word) ||
        left_word != right_word)
      return false;
  }
  return true;
}

BCC32_OPEN_INQUIRY_HD inline bool alternative_continuation_proper_prefix(
    const ResidentRewriteState* state, const AlternativeSnapshot& prefix,
    const AlternativeSnapshot& longer, std::uint32_t suspended_length) {
  std::uint32_t prefix_length = 0u, prefix_digest = 0u;
  std::uint32_t longer_length = 0u, longer_digest = 0u;
  if (!alternative_continuation_digest(state, prefix, suspended_length,
                                       &prefix_length, &prefix_digest) ||
      !alternative_continuation_digest(state, longer, suspended_length,
                                       &longer_length, &longer_digest) ||
      prefix_length >= longer_length)
    return false;
  for (std::uint32_t index = 0u; index < prefix_length; ++index) {
    std::uint32_t prefix_word = 0u, longer_word = 0u;
    if (!alternative_continuation_word_at(state, prefix, suspended_length,
                                          index, &prefix_word) ||
        !alternative_continuation_word_at(state, longer, suspended_length,
                                          index, &longer_word) ||
        prefix_word != longer_word)
      return false;
  }
  return true;
}

BCC32_OPEN_INQUIRY_HD inline bool inquiry_constructor_authoritative(
    const ResidentRewriteState* state, std::uint32_t constructor_slot);
BCC32_OPEN_INQUIRY_HD inline bool constructor_template_digest(
    const ResidentRewriteState* state, const Record& constructor,
    std::uint32_t* digest);

BCC32_OPEN_INQUIRY_HD inline std::uint32_t unique_authoritative_constructor(
    const ResidentRewriteState* state) {
  if (state == nullptr) return kInvalid;
  std::uint32_t found = kInvalid;
  for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
    if (!inquiry_constructor_authoritative(state, slot)) continue;
    if (found != kInvalid) return kInvalid;
    found = slot;
  }
  return found;
}

BCC32_OPEN_INQUIRY_HD inline bool wholly_external_trajectory(
    const ResidentRewriteState* state, const Record& trajectory) {
  if (state == nullptr || trajectory.matter_q8 == 0u ||
      trajectory.lane[0] != kFormTrajectory || trajectory.lane[2] == 0u ||
      (trajectory.lane[7] & ~kTrajectoryWasYielded) != 0u ||
      !mixed_provenance::tagged_history(state, trajectory))
    return false;
  for (std::uint32_t index = 0u; index < trajectory.lane[2]; ++index) {
    mixed_provenance::Origin origin = mixed_provenance::Origin::generated;
    std::uint32_t producer = kInvalid;
    if (!mixed_provenance::origin_at(state, trajectory, index, &origin,
                                     &producer) ||
        origin != mixed_provenance::Origin::external || producer != kInvalid)
      return false;
  }
  return true;
}

BCC32_OPEN_INQUIRY_HD inline std::uint32_t unique_owned_slot(
    const ResidentRewriteState* state, std::uint32_t form,
    std::uint32_t owner, std::uint32_t ordinal) {
  std::uint32_t found = kInvalid;
  if (state == nullptr) return found;
  for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
    const Record& record = state->records[slot];
    if (record.matter_q8 == 0u || record.lane[0] != form ||
        record.lane[1] != owner || record.lane[2] != ordinal)
      continue;
    if (found != kInvalid) return kInvalid;
    found = slot;
  }
  return found;
}

BCC32_OPEN_INQUIRY_HD inline std::uint32_t unique_header_by_owner(
    const ResidentRewriteState* state, std::uint32_t form,
    std::uint32_t owner) {
  std::uint32_t found = kInvalid;
  if (state == nullptr || owner == 0u || owner == kInvalid) return found;
  for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
    const Record& record = state->records[slot];
    if (record.matter_q8 == 0u || record.lane[0] != form ||
        record.lane[1] != owner)
      continue;
    if (found != kInvalid) return kInvalid;
    found = slot;
  }
  return found;
}

BCC32_OPEN_INQUIRY_HD inline bool inquiry_owner_free(
    const ResidentRewriteState* state, std::uint32_t owner) {
  if (state == nullptr || owner == 0u || owner == kInvalid) return false;
  for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
    const Record& record = state->records[slot];
    if (record.matter_q8 != 0u && record.lane[1] == owner) return false;
  }
  return true;
}

BCC32_OPEN_INQUIRY_HD inline std::uint32_t make_inquiry_owner(
    const ResidentRewriteState* state, std::uint32_t form,
    std::uint32_t salt) {
  if (state == nullptr) return kInvalid;
  std::uint32_t owner = rewrite_mix(form, salt, state->revision);
  for (std::uint32_t attempt = 0u; attempt < kRecordCapacity; ++attempt) {
    if (owner != 0u && owner != kInvalid && inquiry_owner_free(state, owner))
      return owner;
    owner = rewrite_mix(owner, salt, attempt + 1u);
  }
  return kInvalid;
}
