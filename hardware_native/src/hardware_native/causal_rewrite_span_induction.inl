
// A span program is still only a chain of ordinary Record matter. A term is
// either one literal raw event or a bounded variable-length raw span. The
// inducer finds common physical anchors by monotone equality; the gaps between
// anchors become variables. No byte is assigned a linguistic role.
BCC32_REWRITE_HD inline bool span_program_term_at(
    const ResidentRewriteState* state, std::uint32_t owner,
    std::uint32_t index, std::uint32_t* kind, std::uint32_t* value,
    std::uint32_t* left_length, std::uint32_t* right_length) {
  const std::uint32_t block = find_owned_block(
      state, kFormSpanProgramTerm, owner, index);
  if (block == kInvalid) return false;
  const Record& term = state->records[block];
  *kind = term.lane[3];
  *value = term.lane[4];
  *left_length = term.lane[5];
  *right_length = term.lane[6];
  return true;
}

// A trajectory can span many physical pages.  Keep the induction workspace
// bounded by the variable/span grammar, not by the number of lived words:
// every read resolves the page containing the requested raw event.  This is
// deliberately a transport cursor, not a token or semantic abstraction.
struct ResidentSpanWordReader {
  const ResidentRewriteState* state = nullptr;
  std::uint32_t slot = kInvalid;
  std::uint32_t count = 0u;

  BCC32_REWRITE_HD bool read(std::uint32_t index,
                             std::uint32_t* word) const {
    return state != nullptr && slot < live_record_capacity(state) &&
           index < count &&
           literal_observation_word_at_prevalidated(state, slot, index, word);
  }
};

BCC32_REWRITE_HD inline bool span_literal_observation_shape(
    const ResidentRewriteState* state, std::uint32_t slot,
    bool resident_literal_prevalidated) {
  if (state == nullptr || slot >= live_record_capacity(state)) return false;
  const Record& source = state->records[slot];
  if (source.matter_q8 == 0u || source.lane[2] == 0u) return false;
  if (source.lane[0] == kFormTrajectory) return source.lane[7] == 0u;
  return resident_literal_prevalidated && source.lane[0] == kFormProgram &&
         (source.lane[7] & kProgramFlagPureExternalExact) != 0u;
}

BCC32_REWRITE_HD inline ResidentSpanWordReader span_word_reader(
    const ResidentRewriteState* state, std::uint32_t slot) {
  ResidentSpanWordReader reader{};
  reader.state = state;
  reader.slot = slot;
  if (state != nullptr && slot < live_record_capacity(state))
    reader.count = state->records[slot].lane[2];
  return reader;
}

BCC32_REWRITE_HD inline std::uint32_t find_span_variable(
    const std::uint32_t left_values[kMaximumProgramVariables]
                              [kMaximumVariableSpanEvents],
    const std::uint32_t right_values[kMaximumProgramVariables]
                               [kMaximumVariableSpanEvents],
    const std::uint32_t left_lengths[kMaximumProgramVariables],
    const std::uint32_t right_lengths[kMaximumProgramVariables],
    std::uint32_t count, const std::uint32_t* left, std::uint32_t left_length,
    const std::uint32_t* right, std::uint32_t right_length) {
  for (std::uint32_t variable = 0u; variable < count; ++variable) {
    if (left_lengths[variable] != left_length ||
        right_lengths[variable] != right_length)
      continue;
    bool same = true;
    for (std::uint32_t i = 0u; i < left_length; ++i)
      same &= left_values[variable][i] == left[i];
    for (std::uint32_t i = 0u; i < right_length; ++i)
      same &= right_values[variable][i] == right[i];
    if (same) return variable;
  }
  return kInvalid;
}

// Count exact contiguous recurrences of one raw span inside its own physical
// trajectory. The count is capped because only the evidence threshold matters.
// No byte class, token boundary, or surface role enters this operation.
BCC32_REWRITE_HD inline std::uint32_t trajectory_span_occurrences(
    const ResidentSpanWordReader& words, std::uint32_t count,
    std::uint32_t start,
    std::uint32_t length, std::uint32_t limit) {
  if (length == 0u || start + length > count) return 0u;
  std::uint32_t start_word = 0u;
  if (!words.read(start, &start_word)) return 0u;
  std::uint32_t occurrences = 0u;
  std::uint32_t position = 0u;
  while (position + length <= count) {
    bool same = true;
    for (std::uint32_t offset = 0u; offset < length; ++offset) {
      std::uint32_t position_word = 0u;
      std::uint32_t candidate_word = 0u;
      if (!words.read(position + offset, &position_word) ||
          !words.read(start + offset, &candidate_word) ||
          position_word != candidate_word) {
        same = false;
        break;
      }
    }
    if (same) {
      if (++occurrences >= limit) return occurrences;
      // One physical word cannot pay support for two overlapping recurrences.
      position += length;
    } else {
      ++position;
    }
  }
  return occurrences;
}

// Host/device contract adapter for raw recurrence controls. Production
// induction uses ResidentSpanWordReader so page ownership and live Record
// provenance are rederived from the resident population. These controls also
// exercise the same recurrence law over an immutable raw fixture; retaining
// this overload keeps that observer-only test independent of resident slot
// allocation and does not introduce a production vocabulary or semantic cell.
struct RawSpanWordReader {
  const std::uint32_t* words = nullptr;
  std::uint32_t count = 0u;

  BCC32_REWRITE_HD bool read(std::uint32_t index,
                             std::uint32_t* word) const {
    if (words == nullptr || word == nullptr || index >= count) return false;
    *word = words[index];
    return true;
  }
};

BCC32_REWRITE_HD inline std::uint32_t trajectory_span_occurrences(
    const std::uint32_t* words, std::uint32_t count, std::uint32_t start,
    std::uint32_t length, std::uint32_t limit) {
  RawSpanWordReader reader{words, count};
  if (length == 0u || start + length > count) return 0u;
  std::uint32_t start_word = 0u;
  if (!reader.read(start, &start_word)) return 0u;
  std::uint32_t occurrences = 0u;
  std::uint32_t position = 0u;
  while (position + length <= count) {
    bool same = true;
    for (std::uint32_t offset = 0u; offset < length; ++offset) {
      std::uint32_t position_word = 0u;
      std::uint32_t candidate_word = 0u;
      if (!reader.read(position + offset, &position_word) ||
          !reader.read(start + offset, &candidate_word) ||
          position_word != candidate_word) {
        same = false;
        break;
      }
    }
    if (same) {
      if (++occurrences >= limit) return occurrences;
      position += length;
    } else {
      ++position;
    }
  }
  return occurrences;
}

BCC32_REWRITE_HD inline std::uint32_t trajectory_equal_run(
    const ResidentSpanWordReader& left, std::uint32_t left_count,
    const ResidentSpanWordReader& right, std::uint32_t right_count,
    std::uint32_t left_index, std::uint32_t right_index) {
  std::uint32_t length = 0u;
  while (left_index + length < left_count &&
         right_index + length < right_count) {
    std::uint32_t left_word = 0u;
    std::uint32_t right_word = 0u;
    if (!left.read(left_index + length, &left_word) ||
        !right.read(right_index + length, &right_word) ||
        left_word != right_word)
      break;
    ++length;
  }
  return length;
}

struct SpanPairCandidate {
  std::uint32_t rewind = 0u;
  std::uint32_t left_length = 0u;
  std::uint32_t right_length = 0u;
  std::uint32_t extent = 0u;
  std::uint32_t recurrence = 0u;
  bool found = false;
  bool tied = false;
};

// Recurrent spans remain the preferred evidence.  When recurrence is absent,
// raw same-channel gaps can still be hypothesized only if exactly one
// minimum-extent cut precedes the uniquely strongest exact raw anchor.
inline constexpr std::uint32_t kMinimumStructuralSpanAnchorEvents = 4u;

BCC32_REWRITE_HD inline SpanPairCandidate find_uniquely_anchored_span_pair(
    const ResidentSpanWordReader& left, std::uint32_t left_count,
    const ResidentSpanWordReader& right, std::uint32_t right_count,
    std::uint32_t left_index, std::uint32_t right_index) {
  SpanPairCandidate best{};
  std::uint32_t left_word = 0u;
  std::uint32_t right_word = 0u;
  if (left_index >= left_count || right_index >= right_count ||
      !left.read(left_index, &left_word) ||
      !right.read(right_index, &right_word) ||
      !same_raw_channel(left_word, right_word))
    return best;
  const std::uint32_t channel = left_word & kRawChannelMask;
  std::uint32_t maximum_left = 0u;
  while (maximum_left < kMaximumVariableSpanEvents &&
         left_index + maximum_left < left_count) {
    std::uint32_t word = 0u;
    if (!left.read(left_index + maximum_left, &word) ||
        (word & kRawChannelMask) != channel)
      break;
    ++maximum_left;
  }
  std::uint32_t maximum_right = 0u;
  while (maximum_right < kMaximumVariableSpanEvents &&
         right_index + maximum_right < right_count) {
    std::uint32_t word = 0u;
    if (!right.read(right_index + maximum_right, &word) ||
        (word & kRawChannelMask) != channel)
      break;
    ++maximum_right;
  }
  std::uint32_t strongest_anchor = 0u;
  for (std::uint32_t left_length = 1u; left_length <= maximum_left;
       ++left_length) {
    const std::uint32_t left_end = left_index + left_length;
    if (left_end >= left_count) continue;
    for (std::uint32_t right_length = 1u; right_length <= maximum_right;
         ++right_length) {
      const std::uint32_t right_end = right_index + right_length;
      if (right_end >= right_count) continue;
      const std::uint32_t anchor = trajectory_equal_run(
          left, left_count, right, right_count, left_end, right_end);
      if (anchor < kMinimumStructuralSpanAnchorEvents) continue;
      const std::uint32_t extent = left_length + right_length;
      if (!best.found || anchor > strongest_anchor ||
          (anchor == strongest_anchor && extent < best.extent)) {
        best.rewind = 0u;
        best.left_length = left_length;
        best.right_length = right_length;
        best.extent = extent;
        best.found = true;
        best.tied = false;
        strongest_anchor = anchor;
      } else if (anchor == strongest_anchor && extent == best.extent &&
                 (best.left_length != left_length ||
                  best.right_length != right_length)) {
        best.tied = true;
      }
    }
  }
  return best;
}

// At a physical mismatch, find the strongest independently recurrent raw-span
// pair that returns both trajectories to the same following event. Equally
// recurrent candidates preserve the greatest observed invariant literal run
// before preferring coarseness: absorbing a shared delimiter into a variable
// can leave two variables adjacent and make later execution genuinely
// ambiguous. A bounded rewind remains available when the rewound span has
// stronger recurrence. Equal recurrence, rewind and extent abstain rather than
// choosing arbitrarily.
//
// Two non-overlapping occurrences inside one trajectory establish recurrence;
// epistemic maturity still requires independent external trajectory owners.
// Reusing kProgramMatureSupport here incorrectly required each candidate span
// to appear three times inside every one of those independent observations.
// That rejected ordinary two-mention reference surfaces even when three lived
// episodes agreed on the same equality topology.
inline constexpr std::uint32_t kMinimumWithinTrajectorySpanRecurrences = 2u;

BCC32_REWRITE_HD inline SpanPairCandidate find_recurrent_span_pair(
    const ResidentSpanWordReader& left, std::uint32_t left_count,
    const ResidentSpanWordReader& right, std::uint32_t right_count,
    std::uint32_t left_index, std::uint32_t right_index,
    std::uint32_t literal_tail) {
  SpanPairCandidate best{};
  const std::uint32_t maximum_rewind =
      literal_tail < kMaximumVariableSpanEvents
          ? literal_tail
          : kMaximumVariableSpanEvents;
  for (std::uint32_t rewind = 0u; rewind <= maximum_rewind; ++rewind) {
    if (rewind > left_index || rewind > right_index) break;
    const std::uint32_t left_start = left_index - rewind;
    const std::uint32_t right_start = right_index - rewind;
    const std::uint32_t left_available = left_count - left_start;
    const std::uint32_t right_available = right_count - right_start;
    std::uint32_t left_start_word = 0u;
    std::uint32_t right_start_word = 0u;
    if (left_available == 0u || right_available == 0u ||
        !left.read(left_start, &left_start_word) ||
        !right.read(right_start, &right_start_word) ||
        !same_raw_channel(left_start_word, right_start_word))
      continue;
    std::uint32_t maximum_left =
        left_available < kMaximumVariableSpanEvents
            ? left_available
            : kMaximumVariableSpanEvents;
    std::uint32_t maximum_right =
        right_available < kMaximumVariableSpanEvents
            ? right_available
            : kMaximumVariableSpanEvents;
    const std::uint32_t variable_channel =
        left_start_word & kRawChannelMask;
    const std::uint32_t left_cap = maximum_left;
    const std::uint32_t right_cap = maximum_right;
    maximum_left = 0u;
    maximum_right = 0u;
    while (maximum_left < left_cap) {
      std::uint32_t word = 0u;
      if (!left.read(left_start + maximum_left, &word) ||
          (word & kRawChannelMask) != variable_channel)
        break;
      ++maximum_left;
    }
    while (maximum_right < right_cap) {
      std::uint32_t word = 0u;
      if (!right.read(right_start + maximum_right, &word) ||
          (word & kRawChannelMask) != variable_channel)
        break;
      ++maximum_right;
    }
    std::uint32_t left_recurrent[kMaximumVariableSpanEvents + 1u]{};
    std::uint32_t right_recurrent[kMaximumVariableSpanEvents + 1u]{};
    for (std::uint32_t length = rewind + 1u; length <= maximum_left;
         ++length) {
      left_recurrent[length] =
          trajectory_span_occurrences(left, left_count, left_start, length,
                                      kProgramMatureSupport);
    }
    for (std::uint32_t length = rewind + 1u; length <= maximum_right;
         ++length) {
      right_recurrent[length] =
          trajectory_span_occurrences(right, right_count, right_start, length,
                                      kProgramMatureSupport);
    }
    for (std::uint32_t left_length = rewind + 1u;
         left_length <= maximum_left; ++left_length) {
      if (left_recurrent[left_length] <
          kMinimumWithinTrajectorySpanRecurrences)
        continue;
      const std::uint32_t left_end = left_start + left_length;
      for (std::uint32_t right_length = rewind + 1u;
           right_length <= maximum_right; ++right_length) {
        if (right_recurrent[right_length] <
            kMinimumWithinTrajectorySpanRecurrences)
          continue;
        const std::uint32_t right_end = right_start + right_length;
        const bool both_end =
            left_end == left_count && right_end == right_count;
        const std::uint32_t following_equal =
            left_end < left_count && right_end < right_count
                ? trajectory_equal_run(left, left_count, right, right_count,
                                       left_end, right_end)
                : 0u;
        const bool both_continue =
            following_equal >= 2u ||
            (following_equal == 1u && left_end + 1u == left_count &&
             right_end + 1u == right_count);
        if (!both_end && !both_continue) continue;
        const std::uint32_t extent = left_length + right_length;
        const std::uint32_t recurrence =
            left_recurrent[left_length] < right_recurrent[right_length]
                ? left_recurrent[left_length]
                : right_recurrent[right_length];
        if (!best.found || recurrence > best.recurrence ||
            (recurrence == best.recurrence && rewind < best.rewind) ||
            (recurrence == best.recurrence && rewind == best.rewind &&
             extent > best.extent)) {
          best.rewind = rewind;
          best.left_length = left_length;
          best.right_length = right_length;
          best.extent = extent;
          best.recurrence = recurrence;
          best.found = true;
          best.tied = false;
        } else if (recurrence == best.recurrence && rewind == best.rewind &&
                   extent == best.extent &&
                   (rewind != best.rewind ||
                    left_length != best.left_length ||
                    right_length != best.right_length)) {
          best.tied = true;
        }
      }
    }
  }
  return best;
}

BCC32_LITERAL_OBSERVATION_HD bool induce_span_program(
    ResidentRewriteState* state, std::uint32_t left_slot,
    std::uint32_t right_slot, std::uint32_t* identity = nullptr,
    bool commit = true, bool resident_literal_prevalidated = false) {
  const Record& left = state->records[left_slot];
  const Record& right = state->records[right_slot];
  if (!span_literal_observation_shape(
          state, left_slot, resident_literal_prevalidated) ||
      !span_literal_observation_shape(
          state, right_slot, resident_literal_prevalidated))
    return false;
  if (left.lane[2] == 0u || right.lane[2] == 0u)
    return false;

  std::uint32_t term_kind[kMaximumSpanProgramTerms]{};
  std::uint32_t term_value[kMaximumSpanProgramTerms]{};
  std::uint32_t term_channel[kMaximumSpanProgramTerms]{};
  std::uint32_t left_values[kMaximumProgramVariables]
                           [kMaximumVariableSpanEvents]{};
  std::uint32_t right_values[kMaximumProgramVariables]
                            [kMaximumVariableSpanEvents]{};
  std::uint32_t left_lengths[kMaximumProgramVariables]{};
  std::uint32_t right_lengths[kMaximumProgramVariables]{};
  const ResidentSpanWordReader left_reader =
      span_word_reader(state, left_slot);
  const ResidentSpanWordReader right_reader =
      span_word_reader(state, right_slot);
  // Equal-width observations ordinarily belong to the compact fixed Program
  // path. Group them into spans only when that path cannot represent the lived
  // equality structure within its bounded variable population. This makes the
  // span path a capacity-preserving generalization, not a competing easy parse.
  bool fixed_width_variable_budget_exceeded = false;
  if (left.lane[2] == right.lane[2]) {
    std::uint32_t fixed_left[kMaximumProgramVariables]{};
    std::uint32_t fixed_right[kMaximumProgramVariables]{};
    std::uint32_t fixed_variables = 0u;
    for (std::uint32_t index = 0u; index < left.lane[2]; ++index) {
      std::uint32_t left_word = 0u;
      std::uint32_t right_word = 0u;
      if (!left_reader.read(index, &left_word) ||
          !right_reader.read(index, &right_word))
        return false;
      if (left_word == right_word) continue;
      if (!same_raw_channel(left_word, right_word)) break;
      if (pair_variable(fixed_left, fixed_right, fixed_variables,
                        left_word, right_word) != kInvalid)
        continue;
      if (fixed_variables == kMaximumProgramVariables) {
        fixed_width_variable_budget_exceeded = true;
        break;
      }
      fixed_left[fixed_variables] = left_word;
      fixed_right[fixed_variables] = right_word;
      ++fixed_variables;
    }
  }
  std::uint32_t variable_count = 0u;
  std::uint32_t term_count = 0u;
  std::uint32_t left_index = 0u;
  std::uint32_t right_index = 0u;
  bool unequal_span = false;
  std::uint32_t literal_tail = 0u;
  while (left_index < left.lane[2] || right_index < right.lane[2]) {
    std::uint32_t left_word = 0u;
    std::uint32_t right_word = 0u;
    const bool have_left =
        left_index < left.lane[2] && left_reader.read(left_index, &left_word);
    const bool have_right = right_index < right.lane[2] &&
                            right_reader.read(right_index, &right_word);
    if ((left_index < left.lane[2] && !have_left) ||
        (right_index < right.lane[2] && !have_right))
      return false;
    if (have_left && have_right && left_word == right_word) {
      if (term_count >= kMaximumSpanProgramTerms)
        return false;
      term_kind[term_count] = kSpanTermLiteral;
      term_value[term_count] = left_word;
      term_channel[term_count] = left_word & kRawChannelMask;
      ++term_count;
      ++literal_tail;
      ++left_index;
      ++right_index;
      continue;
    }

  const SpanPairCandidate candidate = find_recurrent_span_pair(
      left_reader, left.lane[2], right_reader, right.lane[2], left_index,
      right_index, literal_tail);
  SpanPairCandidate selected = candidate;
  if (!selected.found && !selected.tied)
    selected = find_uniquely_anchored_span_pair(
        left_reader, left.lane[2], right_reader, right.lane[2], left_index,
        right_index);
  if (!selected.found || selected.tied) {
    if (selected.tied && commit) ++state->span_ambiguous_abstentions;
    return false;
  }
  if (selected.rewind > term_count ||
      selected.left_length == 0u || selected.right_length == 0u ||
      variable_count >= kMaximumProgramVariables)
    return false;
  term_count -= selected.rewind;
  left_index -= selected.rewind;
  right_index -= selected.rewind;
  const std::uint32_t left_length = selected.left_length;
  const std::uint32_t right_length = selected.right_length;
  std::uint32_t left_span[kMaximumVariableSpanEvents]{};
  std::uint32_t right_span[kMaximumVariableSpanEvents]{};
  for (std::uint32_t i = 0u; i < left_length; ++i)
    if (!left_reader.read(left_index + i, &left_span[i])) return false;
  for (std::uint32_t i = 0u; i < right_length; ++i)
    if (!right_reader.read(right_index + i, &right_span[i])) return false;
  std::uint32_t variable = find_span_variable(
      left_values, right_values, left_lengths, right_lengths, variable_count,
      left_span, left_length, right_span, right_length);
  if (variable == kInvalid) {
    variable = variable_count++;
    left_lengths[variable] = left_length;
    right_lengths[variable] = right_length;
    for (std::uint32_t i = 0u; i < left_length; ++i)
      left_values[variable][i] = left_span[i];
    for (std::uint32_t i = 0u; i < right_length; ++i)
      right_values[variable][i] = right_span[i];
  }
  if (left_length != right_length) unequal_span = true;
  if (term_count >= kMaximumSpanProgramTerms) return false;
  term_kind[term_count] = kSpanTermVariable;
  term_value[term_count] = variable;
  term_channel[term_count] = left_span[0] & kRawChannelMask;
  ++term_count;
  left_index += left_length;
  right_index += right_length;
  literal_tail = 0u;
  }

  if ((!unequal_span && !fixed_width_variable_budget_exceeded) ||
      variable_count == 0u || term_count == 0u ||
      term_count > kMaximumSpanProgramTerms)
    return false;
  if (grounded_record_headroom(state) < term_count + 1u) {
    state->fault = 7u;
    return false;
  }
  std::uint32_t program_identity =
      rewrite_mix(kFormSpanProgram, term_count, variable_count);
  for (std::uint32_t index = 0u; index < term_count; ++index)
    program_identity = rewrite_mix(
        program_identity, term_kind[index],
        term_value[index] ^ term_channel[index]);
  if (identity != nullptr) *identity = program_identity;
  if (!commit) return true;
  GroundedPairReflectionPlan reflection{};
  const GroundedPairReflectionStatus reflection_status =
      preflight_grounded_span_reflection(
          state, left_slot, right_slot, program_identity, term_count + 1u,
          term_count, &reflection);
  if (reflection_status == GroundedPairReflectionStatus::kBlocked)
    return false;
  const std::uint32_t reflection_records =
      reflection_status == GroundedPairReflectionStatus::kReady
          ? reflection.episode_records
          : 0u;
  if (!reserve_grounded_record_matter(
          state, term_count + 1u + reflection_records)) {
    state->fault = 7u;
    return false;
  }
  const std::uint32_t header_slot = allocate_record(state);
  if (header_slot == kInvalid) return false;
  std::uint32_t owner =
      reflection_status == GroundedPairReflectionStatus::kReady
          ? reflection.program_owner
          : program_identity;
  if (reflection_status != GroundedPairReflectionStatus::kReady &&
      (owner == 0u || owner == kInvalid || record_owner_exists(state, owner)))
    owner = make_record_owner(state, owner);
  if (owner == kInvalid) {
    state->fault = 8u;
    return false;
  }
  Record& program = state->records[header_slot];
  program.lane[0] = kFormSpanProgram;
  program.lane[1] = owner;
  program.lane[2] = term_count;
  program.lane[3] = 2u;
  program.lane[4] = variable_count;
  program.lane[5] = program_identity;
  program.lane[7] = kProgramFlagEnabled;
  ++program.revision;
  for (std::uint32_t index = 0u; index < term_count; ++index) {
    const std::uint32_t slot = allocate_record(state);
    if (slot == kInvalid) return false;
    Record& term = state->records[slot];
    term.lane[0] = kFormSpanProgramTerm;
    term.lane[1] = owner;
    term.lane[2] = index;
    term.lane[3] = term_kind[index];
    term.lane[4] = term_value[index];
    // Exemplar bytes and lengths are transient induction evidence. The
    // executable resident form keeps only variable identity and raw channel.
    term.lane[5] = term_channel[index];
    term.lane[6] = 0u;
    ++term.revision;
  }
  if (reflection_status == GroundedPairReflectionStatus::kReady)
    commit_grounded_span_reflection(state, header_slot, reflection);
  retire_literal_observation_source(state, left_slot);
  retire_literal_observation_source(state, right_slot);
  ++state->revision;
  ++state->span_completed_inductions;
  return true;
}

// Match the observed trajectory against a span program. When an unbound span
// has no following anchor yet, the result is an explicit abstention rather
// than an invented boundary. Multiple possible anchor positions are likewise
// ambiguous and never choose a route.
BCC32_REWRITE_HD inline bool indexed_span_program_term_at(
    const ResidentRewriteState* state,
    const std::uint32_t term_slots[kMaximumSpanProgramTerms],
    std::uint32_t index, std::uint32_t* kind, std::uint32_t* value,
    std::uint32_t* left_length, std::uint32_t* right_length) {
  if (index >= kMaximumSpanProgramTerms || term_slots[index] == kInvalid)
    return false;
  const Record& term = state->records[term_slots[index]];
  *kind = term.lane[3];
  *value = term.lane[4];
  *left_length = term.lane[5];
  *right_length = term.lane[6];
  return true;
}
