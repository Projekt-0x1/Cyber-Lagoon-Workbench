#pragma once

// Included inside causal_rewrite after ordinary Program helpers. A resident
// difference is a set of owner-relative Record-word replacements. The executor
// knows no Program term positions or cognitive opcodes: it only resolves one
// live target Record, checks its old word, and applies the already resident new
// word after the complete set has preflighted.

inline constexpr std::uint32_t kConstructionDifferenceHeaderOrdinal =
    kInvalid;
inline constexpr std::uint32_t kConstructionDifferenceFirstMutableWord = 3u;
inline constexpr std::uint32_t kConstructionDifferenceLastMutableWord = 7u;

BCC32_REWRITE_HD inline std::uint32_t resident_record_word(
    const Record& record, std::uint32_t word) {
  if (word < kLaneCount) return record.lane[word];
  if (word == kLaneCount) return record.revision;
  if (word == kLaneCount + 1u) return record.matter_q8;
  return word < kLaneCount + 4u
             ? record.reserved[word - kLaneCount - 2u]
             : kInvalid;
}

BCC32_REWRITE_HD inline void set_resident_record_word(
    Record* record, std::uint32_t word, std::uint32_t value) {
  if (record == nullptr) return;
  if (word < kLaneCount)
    record->lane[word] = value;
  else if (word == kLaneCount)
    record->revision = value;
  else if (word == kLaneCount + 1u)
    record->matter_q8 = value;
  else if (word < kLaneCount + 4u)
    record->reserved[word - kLaneCount - 2u] = value;
}

BCC32_CAUSAL_GERMLINE_DISPATCH std::uint32_t
resident_difference_target_slot(
    const ResidentRewriteState* state, std::uint32_t target_owner,
    std::uint32_t target_form, std::uint32_t target_ordinal,
    std::uint32_t target_word = kInvalid) {
  if (state == nullptr || target_owner == 0u || target_owner == kInvalid ||
      target_form == 0u || target_form == kInvalid)
    return kInvalid;
  // A page-spanning exact Program keeps its continuation terms under
  // page-local owners. Constructor deltas retain the logical ProgramTerm
  // block ordinal, so resolve that ordinal through the same resident page
  // directory used by program_term_at() before applying the lane delta.
  if (target_form == kFormProgramTerm && target_word != kInvalid) {
    if (target_word != 3u && target_word != 5u) return kInvalid;
    std::uint32_t program_slot = kInvalid;
    std::uint32_t program_count = 0u;
    for (std::uint32_t slot = 0u; slot < live_record_capacity(state);
         ++slot) {
      const Record& program = state->records[slot];
      if (program.matter_q8 != 0u && program.lane[0] == kFormProgram &&
          (program.lane[7] & kProgramFlagVersionSpace) == 0u &&
          program.lane[1] == target_owner) {
        program_slot = slot;
        ++program_count;
      }
    }
    if (program_count != 1u || program_slot == kInvalid) return kInvalid;
    const Record& program = state->records[program_slot];
    const std::uint32_t logical_index =
        target_ordinal * 2u + (target_word == 5u ? 1u : 0u);
    if (logical_index >= program.lane[2]) return kInvalid;
    std::uint32_t term_owner = target_owner;
    std::uint32_t ordinal = target_ordinal;
    const std::uint32_t page = logical_index / kTrajectoryPageEvents;
    if (page != 0u) {
      std::uint32_t descriptor_slot = kInvalid;
      std::uint32_t descriptor_count = 0u;
      for (std::uint32_t slot = 0u;
           slot < live_record_capacity(state); ++slot) {
        const Record& descriptor = state->records[slot];
        if (descriptor.matter_q8 == 0u ||
            descriptor.lane[0] != kFormTrajectoryPage ||
            descriptor.lane[1] != target_owner || descriptor.lane[2] != page)
          continue;
        descriptor_slot = slot;
        ++descriptor_count;
      }
      if (descriptor_count != 1u || descriptor_slot == kInvalid) return kInvalid;
      const Record& descriptor = state->records[descriptor_slot];
      const std::uint32_t local = logical_index % kTrajectoryPageEvents;
      if (descriptor.lane[3] != page * kTrajectoryPageEvents ||
          descriptor.lane[4] <= local || descriptor.lane[4] > kTrajectoryPageEvents ||
          descriptor.lane[6] == 0u || descriptor.lane[6] == kInvalid ||
          descriptor.lane[6] == target_owner || descriptor.lane[7] != 0u ||
          descriptor.reserved[0] != 0u || descriptor.reserved[1] != 0u)
        return kInvalid;
      term_owner = descriptor.lane[6];
      ordinal = local / 2u;
    }
    std::uint32_t found = kInvalid;
    for (std::uint32_t slot = 0u; slot < live_record_capacity(state);
         ++slot) {
      const Record& term = state->records[slot];
      if (term.matter_q8 == 0u || term.lane[0] != kFormProgramTerm ||
          term.lane[1] != term_owner || term.lane[2] != ordinal)
        continue;
      if (found != kInvalid) return kInvalid;
      found = slot;
    }
    return found;
  }
  std::uint32_t found = kInvalid;
  for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
    const Record& candidate = state->records[slot];
    if (candidate.matter_q8 == 0u || candidate.lane[0] != target_form ||
        candidate.lane[1] != target_owner ||
        (target_ordinal != kConstructionDifferenceHeaderOrdinal &&
         candidate.lane[2] != target_ordinal))
      continue;
    if (found != kInvalid) return kInvalid;
    found = slot;
  }
  return found;
}

BCC32_CAUSAL_GERMLINE_DISPATCH bool resident_constructor_difference_valid(
    const ResidentRewriteState* state, std::uint32_t constructor_slot) {
  if (state == nullptr || constructor_slot >= live_record_capacity(state)) return false;
  const Record& constructor = state->records[constructor_slot];
  if (constructor.matter_q8 == 0u ||
      constructor.lane[0] != kFormCausalConstructor ||
      constructor.lane[1] == 0u || constructor.lane[1] == kInvalid ||
      constructor.reserved[0] == 0u ||
      constructor.reserved[0] > kMaximumSpanProgramTerms * 3u + 4u)
    return false;
  std::uint32_t count = 0u;
  for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
    const Record& delta = state->records[slot];
    if (delta.matter_q8 == 0u ||
        delta.lane[0] != kFormCausalConstructorDelta ||
        delta.lane[1] != constructor.lane[1])
      continue;
    if (delta.lane[2] >= constructor.reserved[0] ||
        delta.lane[3] == 0u || delta.lane[3] == kInvalid ||
        delta.lane[5] < kConstructionDifferenceFirstMutableWord ||
        delta.lane[5] > kConstructionDifferenceLastMutableWord ||
        delta.reserved[0] != kCausalGermlineEnabled)
      return false;
    for (std::uint32_t prior_slot = 0u; prior_slot < slot; ++prior_slot) {
      const Record& prior = state->records[prior_slot];
      if (prior.matter_q8 != 0u &&
          prior.lane[0] == kFormCausalConstructorDelta &&
          prior.lane[1] == constructor.lane[1] &&
          (prior.lane[2] == delta.lane[2] ||
           (prior.lane[3] == delta.lane[3] &&
            prior.lane[4] == delta.lane[4] &&
            prior.lane[5] == delta.lane[5])))
        return false;
    }
    ++count;
  }
  return count == constructor.reserved[0];
}

BCC32_CAUSAL_GERMLINE_DISPATCH bool
preflight_resident_constructor_difference(
    const ResidentRewriteState* state, std::uint32_t constructor_slot,
    std::uint32_t target_owner) {
  if (!resident_constructor_difference_valid(state, constructor_slot))
    return false;
  const Record& constructor = state->records[constructor_slot];
  for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
    const Record& delta = state->records[slot];
    if (delta.matter_q8 == 0u ||
        delta.lane[0] != kFormCausalConstructorDelta ||
        delta.lane[1] != constructor.lane[1])
      continue;
    const std::uint32_t target = resident_difference_target_slot(
        state, target_owner, delta.lane[3], delta.lane[4], delta.lane[5]);
    if (target == kInvalid ||
        resident_record_word(state->records[target], delta.lane[5]) !=
            delta.lane[6])
      return false;
  }
  return true;
}

BCC32_CAUSAL_GERMLINE_DISPATCH void
commit_resident_constructor_difference(
    ResidentRewriteState* state, std::uint32_t constructor_slot,
    std::uint32_t target_owner) {
  const Record& constructor = state->records[constructor_slot];
  for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
    const Record& delta = state->records[slot];
    if (delta.matter_q8 == 0u ||
        delta.lane[0] != kFormCausalConstructorDelta ||
        delta.lane[1] != constructor.lane[1])
      continue;
    const std::uint32_t target = resident_difference_target_slot(
        state, target_owner, delta.lane[3], delta.lane[4], delta.lane[5]);
    Record& record = state->records[target];
    set_resident_record_word(&record, delta.lane[5], delta.lane[7]);
    ++record.revision;
  }
}

BCC32_CAUSAL_GERMLINE_DISPATCH bool
resident_constructor_difference_realized(
    const ResidentRewriteState* state, std::uint32_t constructor_slot,
    std::uint32_t target_owner) {
  if (!resident_constructor_difference_valid(state, constructor_slot))
    return false;
  const Record& constructor = state->records[constructor_slot];
  for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
    const Record& delta = state->records[slot];
    if (delta.matter_q8 == 0u ||
        delta.lane[0] != kFormCausalConstructorDelta ||
        delta.lane[1] != constructor.lane[1])
      continue;
    const std::uint32_t target = resident_difference_target_slot(
        state, target_owner, delta.lane[3], delta.lane[4], delta.lane[5]);
    if (target == kInvalid ||
        resident_record_word(state->records[target], delta.lane[5]) !=
            delta.lane[7])
      return false;
  }
  return true;
}
