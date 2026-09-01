#pragma once

// Included by causal_rewrite_universe.cuh after Program induction helpers and
// before Program execution helpers, inside the causal_rewrite namespace.
// This first production germline is deliberately narrow: it reflects the
// externally grounded transaction by which a fixed Program becomes mature,
// factors three independent reflections into resident constructor matter, and
// lets that matter generalize one later exact external Program.  It is a
// bootstrap phenotype to be subsumed by generic Record-graph construction,
// not a permanent Program-specific ontology.

BCC32_REWRITE_HD inline std::uint32_t causal_program_slot_by_owner(
    const ResidentRewriteState* state, std::uint32_t owner) {
  if (state == nullptr || owner == 0u || owner == kInvalid) return kInvalid;
  std::uint32_t found = kInvalid;
  for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
    const Record& record = state->records[slot];
    if (record.matter_q8 == 0u || record.lane[0] != kFormProgram ||
        record.lane[1] != owner)
      continue;
    if (found != kInvalid) return kInvalid;
    found = slot;
  }
  return found;
}

BCC32_REWRITE_HD inline std::uint32_t causal_span_program_slot_by_owner(
    const ResidentRewriteState* state, std::uint32_t owner) {
  if (state == nullptr || owner == 0u || owner == kInvalid) return kInvalid;
  std::uint32_t found = kInvalid;
  for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
    const Record& record = state->records[slot];
    if (record.matter_q8 == 0u || record.lane[0] != kFormSpanProgram ||
        record.lane[1] != owner)
      continue;
    if (found != kInvalid) return kInvalid;
    found = slot;
  }
  return found;
}

BCC32_REWRITE_HD inline std::uint32_t causal_product_slot_by_owner(
    const ResidentRewriteState* state, std::uint32_t owner) {
  const std::uint32_t fixed = causal_program_slot_by_owner(state, owner);
  const std::uint32_t span = causal_span_program_slot_by_owner(state, owner);
  if (fixed != kInvalid && span != kInvalid) return kInvalid;
  return fixed != kInvalid ? fixed : span;
}

BCC32_REWRITE_HD inline std::uint32_t causal_episode_slot_by_identity(
    const ResidentRewriteState* state, std::uint32_t identity) {
  std::uint32_t found = kInvalid;
  for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
    const Record& record = state->records[slot];
    if (record.matter_q8 == 0u ||
        record.lane[0] != kFormConstructionEpisode ||
        record.lane[1] != identity)
      continue;
    if (found != kInvalid) return kInvalid;
    found = slot;
  }
  return found;
}

BCC32_REWRITE_HD inline std::uint32_t causal_episode_slot_by_product(
    const ResidentRewriteState* state, std::uint32_t product_owner) {
  std::uint32_t found = kInvalid;
  for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
    const Record& record = state->records[slot];
    if (record.matter_q8 == 0u ||
        record.lane[0] != kFormConstructionEpisode ||
        record.lane[2] != product_owner)
      continue;
    if (found != kInvalid) return kInvalid;
    found = slot;
  }
  return found;
}

BCC32_REWRITE_HD inline std::uint32_t causal_owned_ordinal(
    const ResidentRewriteState* state, std::uint32_t form,
    std::uint32_t owner, std::uint32_t ordinal) {
  std::uint32_t found = kInvalid;
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

BCC32_REWRITE_HD inline std::uint32_t causal_topology_digest(
    const ResidentRewriteState* state, const Record& program) {
  return grounded_construction_topology_digest(state, program);
}

BCC32_CAUSAL_GERMLINE_DISPATCH bool causal_episode_valid(
    const ResidentRewriteState* state, std::uint32_t header_slot) {
  if (state == nullptr || header_slot >= live_record_capacity(state)) return false;
  const Record& header = state->records[header_slot];
  // Construction-episode terms are resident records addressed by owner and
  // ordinal; their logical extent is not a one-page workspace. The validation
  // loop below therefore remains valid after physical trajectory rollover.
  if (header.matter_q8 == 0u ||
      header.lane[0] != kFormConstructionEpisode ||
      header.lane[1] == 0u || header.lane[1] == kInvalid ||
      header.lane[2] == 0u || header.lane[2] == kInvalid ||
      header.lane[3] == 0u || header.lane[3] == kInvalid ||
      header.lane[2] == header.lane[3] || header.lane[4] == 0u ||
      header.lane[5] == 0u ||
      header.lane[5] > kMaximumProgramVariables ||
      header.lane[6] == 0u || header.lane[6] == kInvalid ||
      (header.lane[7] & kCausalGermlineExternal) == 0u ||
      (header.lane[7] & ~(kCausalGermlineExternal |
                          kConstructionEpisodePairInduction |
                          kConstructionEpisodeSpanInduction)) != 0u)
    return false;
  const bool pair_induction =
      (header.lane[7] & kConstructionEpisodePairInduction) != 0u;
  const bool span_induction =
      (header.lane[7] & kConstructionEpisodeSpanInduction) != 0u;
  if (pair_induction == span_induction ||
      (header.reserved[0] == 0u || header.reserved[0] == kInvalid ||
       header.reserved[0] == header.lane[2] ||
       header.reserved[0] == header.lane[3]))
    return false;
  const std::uint32_t product_slot =
      span_induction
          ? causal_span_program_slot_by_owner(state, header.lane[2])
          : causal_program_slot_by_owner(state, header.lane[2]);
  if (product_slot == kInvalid)
    return false;
  const Record& product = state->records[product_slot];
  if ((product.lane[7] &
       (kProgramFlagPureExternalExact | kProgramFlagResidentEvidenceOnly |
        kProgramFlagCausalGermlineProduct)) != 0u ||
      (product.lane[7] & kProgramFlagEnabled) == 0u ||
      (span_induction
           ? product.lane[3] < kSpanProgramMatureSupport
           : product.lane[3] < kProgramMatureSupport) ||
      product.lane[2] != header.lane[4] ||
      product.lane[4] != header.lane[5] ||
      product.lane[5] != header.reserved[1] ||
      causal_topology_digest(state, product) != header.lane[6])
    return false;
  for (std::uint32_t index = 0u; index < header.lane[4]; ++index) {
    const std::uint32_t term_slot = causal_owned_ordinal(
        state, kFormConstructionEpisodeTerm, header.lane[1], index);
    if (term_slot == kInvalid) return false;
    const Record& term = state->records[term_slot];
    if (term.lane[7] != header.lane[7])
      return false;
    if (span_induction) {
      std::uint32_t kind = 0u;
      std::uint32_t value = 0u;
      std::uint32_t channel = 0u;
      std::uint32_t unused = 0u;
      if (!span_program_term_at(state, header.lane[2], index, &kind, &value,
                                &channel, &unused) ||
          term.lane[3] != kind ||
          term.lane[4] != (kind == kSpanTermVariable ? value : 0u) ||
          term.lane[5] != (channel & kRawChannelMask) ||
          (kind != kSpanTermLiteral && kind != kSpanTermVariable) ||
          (kind == kSpanTermVariable && value >= header.lane[5]))
        return false;
    } else {
      std::uint32_t product_word = 0u;
      std::uint32_t product_meta = 0u;
      if (!program_term_at(state, header.lane[2], index, &product_word,
                           &product_meta))
        return false;
      if (term.lane[3] == kBoundaryPause ||
          term.lane[3] == kBoundaryEnd ||
          term.lane[4] == kBoundaryPause ||
          term.lane[4] == kBoundaryEnd || term.lane[5] != product_word ||
          term.lane[6] != product_meta ||
          (term.lane[3] & kRawChannelMask) !=
              (product_word & kRawChannelMask) ||
          (term.lane[4] & kRawChannelMask) !=
              (product_word & kRawChannelMask) ||
          (product_meta == 0u &&
           (term.lane[3] != product_word ||
            term.lane[4] != product_word)) ||
          (product_meta != 0u && term.lane[3] == term.lane[4]) ||
          product_meta > header.lane[5])
        return false;
    }
  }
  return true;
}

BCC32_REWRITE_HD inline bool causal_episode_topology_equal(
    const ResidentRewriteState* state, const Record& left,
    const Record& right) {
  if (left.lane[4] != right.lane[4] || left.lane[5] != right.lane[5] ||
      left.lane[6] != right.lane[6] || left.lane[7] != right.lane[7])
    return false;
  const bool pair_induction =
      (left.lane[7] & kConstructionEpisodePairInduction) != 0u;
  const bool span_induction =
      (left.lane[7] & kConstructionEpisodeSpanInduction) != 0u;
  for (std::uint32_t index = 0u; index < left.lane[4]; ++index) {
    const std::uint32_t a = causal_owned_ordinal(
        state, kFormConstructionEpisodeTerm, left.lane[1], index);
    const std::uint32_t b = causal_owned_ordinal(
        state, kFormConstructionEpisodeTerm, right.lane[1], index);
    if (a == kInvalid || b == kInvalid) return false;
    const Record& left_term = state->records[a];
    const Record& right_term = state->records[b];
    if (span_induction) {
      if (left_term.lane[3] != right_term.lane[3] ||
          left_term.lane[4] != right_term.lane[4] ||
          left_term.lane[5] != right_term.lane[5])
        return false;
    } else if (pair_induction) {
      if (left_term.lane[6] != right_term.lane[6] ||
          (left_term.lane[3] & kRawChannelMask) !=
              (right_term.lane[3] & kRawChannelMask) ||
          (left_term.lane[4] & kRawChannelMask) !=
              (right_term.lane[4] & kRawChannelMask) ||
          (left_term.lane[5] & kRawChannelMask) !=
              (right_term.lane[5] & kRawChannelMask) ||
          (left_term.lane[3] == left_term.lane[4]) !=
              (right_term.lane[3] == right_term.lane[4]))
        return false;
    } else if (left_term.lane[5] != right_term.lane[5] ||
               (left_term.lane[3] & kRawChannelMask) !=
                   (right_term.lane[3] & kRawChannelMask)) {
      return false;
    }
  }
  return true;
}

BCC32_CAUSAL_GERMLINE_DISPATCH bool reflect_grounded_program_construction(
    ResidentRewriteState* state, std::uint32_t program_slot,
    const Record& source) {
  if (state == nullptr || program_slot >= live_record_capacity(state) ||
      source.matter_q8 == 0u || source.lane[0] != kFormTrajectory ||
      source.lane[7] != 0u || source.lane[1] == 0u ||
      source.lane[1] == kInvalid || source.lane[2] == 0u)
    return false;
  const Record& program = state->records[program_slot];
  if (program.matter_q8 == 0u || program.lane[0] != kFormProgram ||
      program.lane[3] != kProgramMatureSupport || program.lane[4] == 0u ||
      program.lane[2] != source.lane[2] ||
      (program.lane[7] &
       (kProgramFlagVersionSpace | kProgramFlagPureExternalExact |
        kProgramFlagResidentEvidenceOnly | kProgramFlagCausalGermlineProduct)) !=
          0u ||
      causal_episode_slot_by_product(state, program.lane[1]) != kInvalid ||
      !full_program_match(state, program, source))
    return false;
  const std::uint32_t topology = causal_topology_digest(state, program);
  if (topology == kInvalid ||
      !reserve_grounded_record_matter(state, program.lane[2] + 1u))
    return false;
  for (std::uint32_t index = 0u; index < program.lane[2]; ++index) {
    std::uint32_t source_word = 0u;
    std::uint32_t product_word = 0u;
    std::uint32_t product_meta = 0u;
    if (!trajectory_word_at(state, source.lane[1], index, &source_word) ||
        !program_term_at(state, program.lane[1], index, &product_word,
                         &product_meta) ||
        (source_word & kRawChannelMask) !=
            (product_word & kRawChannelMask))
      return false;
  }
  std::uint32_t identity = rewrite_mix(
      kFormConstructionEpisode, program.lane[1], source.lane[1]);
  if (identity == 0u || identity == kInvalid || record_owner_exists(state, identity))
    identity = make_record_owner(state, identity);
  if (identity == kInvalid) return false;
  const std::uint32_t header_slot = allocate_record(state);
  if (header_slot == kInvalid) return false;
  Record& header = state->records[header_slot];
  header.lane[0] = kFormConstructionEpisode;
  header.lane[1] = identity;
  header.lane[2] = program.lane[1];
  header.lane[3] = source.lane[1];
  header.lane[4] = program.lane[2];
  header.lane[5] = program.lane[4];
  header.lane[6] = topology;
  header.lane[7] = kCausalGermlineExternal;
  header.reserved[0] = source.lane[6];
  header.reserved[1] = program.lane[5];
  ++header.revision;
  for (std::uint32_t index = 0u; index < program.lane[2]; ++index) {
    const std::uint32_t term_slot = allocate_record(state);
    if (term_slot == kInvalid) return false;
    Record& term = state->records[term_slot];
    std::uint32_t source_word = 0u;
    std::uint32_t product_word = 0u;
    std::uint32_t product_meta = 0u;
    (void)trajectory_word_at(state, source.lane[1], index, &source_word);
    (void)program_term_at(state, program.lane[1], index, &product_word,
                          &product_meta);
    term.lane[0] = kFormConstructionEpisodeTerm;
    term.lane[1] = identity;
    term.lane[2] = index;
    term.lane[3] = source_word;
    term.lane[4] = product_word;
    term.lane[5] = product_meta;
    term.lane[6] = index + 1u == program.lane[2] ? 1u : 0u;
    term.lane[7] = kCausalGermlineExternal;
    ++term.revision;
  }
  ++state->revision;
  return true;
}

BCC32_REWRITE_HD inline std::uint32_t causal_constructor_slot_by_topology(
    const ResidentRewriteState* state, std::uint32_t topology) {
  std::uint32_t found = kInvalid;
  for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
    const Record& record = state->records[slot];
    if (record.matter_q8 == 0u ||
        record.lane[0] != kFormCausalConstructor ||
        record.lane[5] != topology)
      continue;
    if (found != kInvalid) return kInvalid;
    found = slot;
  }
  return found;
}

BCC32_REWRITE_HD inline bool causal_sequence_word_at(
    const ResidentRewriteState* state, const Record& sequence,
    std::uint32_t index, std::uint32_t* word) {
  if (word == nullptr || index >= sequence.lane[2]) return false;
  if (sequence.lane[0] == kFormTrajectory)
    return trajectory_word_at(state, sequence.lane[1], index, word);
  if (sequence.lane[0] != kFormProgram) return false;
  std::uint32_t meta = 0u;
  return program_term_at(state, sequence.lane[1], index, word, &meta) &&
         meta == 0u;
}

inline constexpr std::uint32_t kCausalSpanPartitionSearchBudget = 16384u;

BCC32_CAUSAL_GERMLINE_DISPATCH bool
span_constructor_first_binding_start(
    const ResidentRewriteState* state, const Record& constructor,
    const std::uint32_t lengths[kMaximumProgramVariables],
    std::uint32_t target_variable, std::uint32_t* start) {
  if (state == nullptr || lengths == nullptr || start == nullptr ||
      target_variable >= constructor.lane[3])
    return false;
  std::uint32_t cursor = 0u;
  for (std::uint32_t index = 0u; index < constructor.lane[2]; ++index) {
    const std::uint32_t term_slot = causal_owned_ordinal(
        state, kFormCausalConstructorTerm, constructor.lane[1], index);
    if (term_slot == kInvalid) return false;
    const Record& term = state->records[term_slot];
    if (term.lane[3] == kSpanTermLiteral) {
      ++cursor;
      continue;
    }
    if (term.lane[3] != kSpanTermVariable ||
        term.lane[4] >= constructor.lane[3])
      return false;
    if (term.lane[4] == target_variable) {
      *start = cursor;
      return true;
    }
    // Variable identities are assigned on first physical occurrence. Seeing a
    // later identity here would make this partial binding address-dependent.
    if (term.lane[4] > target_variable || lengths[term.lane[4]] == 0u)
      return false;
    cursor += lengths[term.lane[4]];
  }
  return false;
}

BCC32_CAUSAL_GERMLINE_DISPATCH bool
causal_sequence_span_repeats(
    const ResidentRewriteState* state, const Record& source,
    std::uint32_t start, std::uint32_t length,
    std::uint32_t required) {
  if (state == nullptr || length == 0u || required == 0u ||
      start + length > source.lane[2])
    return false;
  std::uint32_t occurrences = 0u;
  std::uint32_t position = 0u;
  while (position + length <= source.lane[2]) {
    bool same = true;
    for (std::uint32_t offset = 0u; offset < length; ++offset) {
      std::uint32_t expected = 0u;
      std::uint32_t observed = 0u;
      if (!causal_sequence_word_at(state, source, start + offset,
                                   &expected) ||
          !causal_sequence_word_at(state, source, position + offset,
                                   &observed) ||
          expected != observed) {
        same = false;
        break;
      }
    }
    if (same) {
      if (++occurrences >= required) return true;
      position += length;
    } else {
      ++position;
    }
  }
  return false;
}

BCC32_CAUSAL_GERMLINE_DISPATCH bool
indexed_span_constructor_first_binding_start(
    const std::uint32_t term_kind[kMaximumSpanProgramTerms],
    const std::uint32_t term_variable[kMaximumSpanProgramTerms],
    std::uint32_t term_count,
    const std::uint32_t lengths[kMaximumProgramVariables],
    std::uint32_t target_variable, std::uint32_t* start) {
  if (term_kind == nullptr || term_variable == nullptr ||
      lengths == nullptr || start == nullptr)
    return false;
  std::uint32_t cursor = 0u;
  for (std::uint32_t index = 0u; index < term_count; ++index) {
    if (term_kind[index] == kSpanTermLiteral) {
      ++cursor;
      continue;
    }
    if (term_kind[index] != kSpanTermVariable) return false;
    const std::uint32_t variable = term_variable[index];
    if (variable == target_variable) {
      *start = cursor;
      return true;
    }
    if (variable > target_variable || lengths[variable] == 0u)
      return false;
    cursor += lengths[variable];
  }
  return false;
}

BCC32_CAUSAL_GERMLINE_DISPATCH bool indexed_raw_span_repeats(
    const std::uint32_t words[kMaximumTrajectoryEvents],
    std::uint32_t word_count, std::uint32_t start,
    std::uint32_t length, std::uint32_t required) {
  if (words == nullptr || length == 0u || required == 0u ||
      start + length > word_count)
    return false;
  std::uint32_t occurrences = 0u;
  std::uint32_t position = 0u;
  while (position + length <= word_count) {
    bool same = true;
    for (std::uint32_t offset = 0u; offset < length; ++offset)
      same &= words[start + offset] == words[position + offset];
    if (same) {
      if (++occurrences >= required) return true;
      position += length;
    } else {
      ++position;
    }
  }
  return false;
}

BCC32_CAUSAL_GERMLINE_DISPATCH bool
indexed_span_constructor_partition_matches_lengths(
    const std::uint32_t term_kind[kMaximumSpanProgramTerms],
    const std::uint32_t term_variable[kMaximumSpanProgramTerms],
    const std::uint32_t term_channel[kMaximumSpanProgramTerms],
    std::uint32_t term_count, std::uint32_t variable_count,
    const std::uint32_t words[kMaximumTrajectoryEvents],
    std::uint32_t word_count,
    const std::uint32_t lengths[kMaximumProgramVariables]) {
  if (term_kind == nullptr || term_variable == nullptr ||
      term_channel == nullptr || words == nullptr || lengths == nullptr)
    return false;
  std::uint32_t binding_start[kMaximumProgramVariables];
  for (std::uint32_t variable = 0u; variable < kMaximumProgramVariables;
       ++variable)
    binding_start[variable] = kInvalid;
  std::uint32_t cursor = 0u;
  for (std::uint32_t index = 0u; index < term_count; ++index) {
    if (term_kind[index] == kSpanTermLiteral) {
      if (cursor >= word_count ||
          (words[cursor] & kRawChannelMask) != term_channel[index])
        return false;
      ++cursor;
      continue;
    }
    const std::uint32_t variable = term_variable[index];
    if (term_kind[index] != kSpanTermVariable ||
        variable >= variable_count)
      return false;
    const std::uint32_t length = lengths[variable];
    if (length == 0u || length > kMaximumVariableSpanEvents ||
        cursor + length > word_count)
      return false;
    if (binding_start[variable] == kInvalid) {
      binding_start[variable] = cursor;
    } else {
      for (std::uint32_t offset = 0u; offset < length; ++offset)
        if (words[binding_start[variable] + offset] !=
            words[cursor + offset])
          return false;
    }
    for (std::uint32_t offset = 0u; offset < length; ++offset)
      if ((words[cursor + offset] & kRawChannelMask) != term_channel[index])
        return false;
    cursor += length;
  }
  return cursor == word_count;
}

BCC32_CAUSAL_GERMLINE_DISPATCH bool
span_constructor_partition_matches_lengths(
    const ResidentRewriteState* state, const Record& constructor,
    const Record& source,
    const std::uint32_t lengths[kMaximumProgramVariables]) {
  if (state == nullptr || lengths == nullptr || constructor.lane[3] == 0u ||
      constructor.lane[3] > kMaximumProgramVariables)
    return false;
  std::uint32_t binding_start[kMaximumProgramVariables];
  for (std::uint32_t variable = 0u; variable < kMaximumProgramVariables;
       ++variable)
    binding_start[variable] = kInvalid;
  std::uint32_t cursor = 0u;
  for (std::uint32_t index = 0u; index < constructor.lane[2]; ++index) {
    const std::uint32_t term_slot = causal_owned_ordinal(
        state, kFormCausalConstructorTerm, constructor.lane[1], index);
    if (term_slot == kInvalid) return false;
    const Record& term = state->records[term_slot];
    if (term.lane[3] == kSpanTermLiteral) {
      std::uint32_t word = 0u;
      if (cursor >= source.lane[2] ||
          !causal_sequence_word_at(state, source, cursor, &word) ||
          (word & kRawChannelMask) != term.lane[6])
        return false;
      ++cursor;
      continue;
    }
    if (term.lane[3] != kSpanTermVariable ||
        term.lane[4] >= constructor.lane[3])
      return false;
    const std::uint32_t variable = term.lane[4];
    const std::uint32_t length = lengths[variable];
    if (length == 0u || length > kMaximumVariableSpanEvents ||
        cursor + length > source.lane[2])
      return false;
    if (binding_start[variable] == kInvalid) {
      binding_start[variable] = cursor;
    } else {
      for (std::uint32_t offset = 0u; offset < length; ++offset) {
        std::uint32_t prior = 0u;
        std::uint32_t current = 0u;
        if (!causal_sequence_word_at(
                state, source, binding_start[variable] + offset, &prior) ||
            !causal_sequence_word_at(state, source, cursor + offset,
                                      &current) ||
            prior != current)
          return false;
      }
    }
    for (std::uint32_t offset = 0u; offset < length; ++offset) {
      std::uint32_t word = 0u;
      if (!causal_sequence_word_at(state, source, cursor + offset, &word) ||
          (word & kRawChannelMask) != term.lane[6])
        return false;
    }
    cursor += length;
  }
  return cursor == source.lane[2];
}

BCC32_CAUSAL_GERMLINE_DISPATCH bool
span_constructor_partition_sequence_structural(
    const ResidentRewriteState* state, std::uint32_t constructor_slot,
    const Record& source,
    std::uint32_t output_lengths[kMaximumProgramVariables],
    bool* ambiguous);

BCC32_CAUSAL_GERMLINE_DISPATCH bool
causal_product_single_literal_difference(
    const ResidentRewriteState* state, std::uint32_t product_slot,
    const Record& source, std::uint32_t* mismatch_ordinal) {
  if (state == nullptr || mismatch_ordinal == nullptr ||
      product_slot >= live_record_capacity(state))
    return false;
  const Record& product = state->records[product_slot];
  if (product.matter_q8 == 0u || product.lane[0] != kFormProgram ||
      source.lane[2] != product.lane[2] || product.lane[4] == 0u ||
      product.lane[4] > kMaximumProgramVariables)
    return false;
  std::uint32_t binding[kMaximumProgramVariables]{};
  std::uint32_t bound = 0u;
  std::uint32_t mismatch = kInvalid;
  for (std::uint32_t index = 0u; index < product.lane[2]; ++index) {
    std::uint32_t observed = 0u;
    std::uint32_t expected = 0u;
    std::uint32_t meta = 0u;
    if (!causal_sequence_word_at(state, source, index, &observed) ||
        !program_term_at(state, product.lane[1], index, &expected, &meta))
      return false;
    if (meta == 0u) {
      if (observed == expected) continue;
      if (index == 0u || mismatch != kInvalid ||
          !same_raw_channel(observed, expected))
        return false;
      mismatch = index;
      continue;
    }
    const std::uint32_t variable = meta - 1u;
    if (variable >= kMaximumProgramVariables ||
        !same_raw_channel(observed, expected))
      return false;
    const std::uint32_t bit = 1u << variable;
    if ((bound & bit) != 0u) {
      if (binding[variable] != observed) return false;
    } else {
      binding[variable] = observed;
      bound |= bit;
    }
  }
  if (mismatch == kInvalid) return false;
  *mismatch_ordinal = mismatch;
  return true;
}

BCC32_CAUSAL_GERMLINE_DISPATCH bool
causal_version_space_sequence_relation(
    const ResidentRewriteState* state, std::uint32_t product_slot,
    const Record& source, bool require_difference,
    std::uint32_t* mismatch_ordinal) {
  if (state == nullptr || product_slot >= live_record_capacity(state) ||
      mismatch_ordinal == nullptr)
    return false;
  const Record& product = state->records[product_slot];
  if (!version_space_program_authoritative(state, product_slot) ||
      source.lane[2] != product.lane[2])
    return false;
  std::uint32_t mismatch = kInvalid;
  for (std::uint32_t index = 0u; index < product.lane[2]; ++index) {
    std::uint32_t observed = 0u;
    std::uint32_t expected = 0u;
    std::uint32_t meta = 0u;
    if (!causal_sequence_word_at(state, source, index, &observed) ||
        !version_space_effective_program_term_at(
            state, product, index, &expected, &meta))
      return false;
    if (observed == expected) continue;
    if (meta != 0u || mismatch != kInvalid || index == 0u ||
        !same_raw_channel(observed, expected))
      return false;
    mismatch = index;
  }
  if (require_difference != (mismatch != kInvalid)) return false;
  *mismatch_ordinal = mismatch;
  return true;
}

BCC32_CAUSAL_GERMLINE_DISPATCH bool
causal_version_space_disambiguation_valid(
    const ResidentRewriteState* state, std::uint32_t product_slot,
    const Record& source, std::uint32_t* mismatch_ordinal) {
  if (state == nullptr || product_slot >= live_record_capacity(state) ||
      mismatch_ordinal == nullptr)
    return false;
  const Record& target = state->records[product_slot];
  std::uint32_t mismatch = kInvalid;
  if (!causal_version_space_sequence_relation(
          state, product_slot, source, true, &mismatch) ||
      mismatch + 1u != target.lane[2])
    return false;
  std::uint32_t agreeing = 0u;
  for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
    if (slot == product_slot) continue;
    const Record& candidate = state->records[slot];
    if (candidate.matter_q8 == 0u || candidate.lane[0] != kFormProgram ||
        (candidate.lane[7] & kProgramFlagVersionSpace) == 0u ||
        candidate.lane[5] != target.lane[5])
      continue;
    std::uint32_t unused = kInvalid;
    agreeing += causal_version_space_sequence_relation(
        state, slot, source, false, &unused);
  }
  if (agreeing != 1u) return false;
  *mismatch_ordinal = mismatch;
  return true;
}

BCC32_CAUSAL_GERMLINE_DISPATCH bool
causal_span_product_single_literal_difference(
    const ResidentRewriteState* state, std::uint32_t product_slot,
    std::uint32_t constructor_slot, const Record& source,
    std::uint32_t* mismatch_ordinal) {
  if (state == nullptr || mismatch_ordinal == nullptr ||
      product_slot >= live_record_capacity(state) ||
      constructor_slot >= live_record_capacity(state))
    return false;
  const Record& product = state->records[product_slot];
  const Record& constructor = state->records[constructor_slot];
  if (product.matter_q8 == 0u || product.lane[0] != kFormSpanProgram ||
      constructor.matter_q8 == 0u ||
      constructor.lane[0] != kFormCausalConstructor ||
      constructor.lane[6] != kFormSpanProgram ||
      product.lane[2] != constructor.lane[2] ||
      product.lane[4] != constructor.lane[3] ||
      (source.lane[0] != kFormTrajectory &&
       source.lane[0] != kFormProgram))
    return false;
  std::uint32_t lengths[kMaximumProgramVariables]{};
  bool ambiguous = false;
  if (!span_constructor_partition_sequence_structural(
          state, constructor_slot, source, lengths, &ambiguous) ||
      ambiguous)
    return false;

  std::uint32_t binding_start[kMaximumProgramVariables];
  for (std::uint32_t variable = 0u; variable < kMaximumProgramVariables;
       ++variable)
    binding_start[variable] = kInvalid;
  std::uint32_t cursor = 0u;
  std::uint32_t mismatch = kInvalid;
  for (std::uint32_t index = 0u; index < product.lane[2]; ++index) {
    std::uint32_t kind = 0u;
    std::uint32_t value = 0u;
    std::uint32_t channel = 0u;
    std::uint32_t unused = 0u;
    if (!span_program_term_at(state, product.lane[1], index, &kind, &value,
                              &channel, &unused))
      return false;
    if (kind == kSpanTermLiteral) {
      std::uint32_t observed = 0u;
      if (!causal_sequence_word_at(state, source, cursor, &observed))
        return false;
      if (observed != value) {
        if (index == 0u || mismatch != kInvalid ||
            !same_raw_channel(observed, value))
          return false;
        mismatch = index;
      }
      ++cursor;
      continue;
    }
    if (kind != kSpanTermVariable || value >= product.lane[4] ||
        lengths[value] == 0u ||
        cursor + lengths[value] > source.lane[2])
      return false;
    if (binding_start[value] == kInvalid) {
      binding_start[value] = cursor;
    } else {
      for (std::uint32_t offset = 0u; offset < lengths[value]; ++offset) {
        std::uint32_t prior = 0u;
        std::uint32_t current = 0u;
        if (!causal_sequence_word_at(
                state, source, binding_start[value] + offset, &prior) ||
            !causal_sequence_word_at(
                state, source, cursor + offset, &current) ||
            prior != current)
          return false;
      }
    }
    for (std::uint32_t offset = 0u; offset < lengths[value]; ++offset) {
      std::uint32_t word = 0u;
      if (!causal_sequence_word_at(state, source, cursor + offset, &word) ||
          (word & kRawChannelMask) != channel)
        return false;
    }
    cursor += lengths[value];
  }
  if (cursor != source.lane[2] || mismatch == kInvalid) return false;
  *mismatch_ordinal = mismatch;
  return true;
}

BCC32_CAUSAL_GERMLINE_DISPATCH bool causal_counterevidence_valid(
    const ResidentRewriteState* state, std::uint32_t counter_slot,
    std::uint32_t* product_slot_out = nullptr,
    std::uint32_t* constructor_slot_out = nullptr) {
  if (state == nullptr || counter_slot >= live_record_capacity(state)) return false;
  const Record& counter = state->records[counter_slot];
  if (counter.matter_q8 == 0u ||
      counter.lane[0] != kFormCausalCounterevidence ||
      counter.lane[1] == 0u || counter.lane[1] == kInvalid ||
      counter.lane[2] == 0u || counter.lane[2] == kInvalid ||
      counter.lane[3] == 0u || counter.lane[3] == kInvalid ||
      counter.lane[4] == 0u || counter.lane[5] == 0u ||
      counter.lane[5] == kInvalid || counter.lane[6] == 0u ||
      counter.lane[6] == kInvalid ||
      counter.lane[7] != kCausalGermlineExternal)
    return false;
  const std::uint32_t product_slot =
      causal_product_slot_by_owner(state, counter.lane[1]);
  const std::uint32_t source_slot =
      causal_program_slot_by_owner(state, counter.lane[3]);
  if (product_slot == kInvalid || source_slot == kInvalid ||
      product_slot == source_slot ||
      !pure_external_exact_program_authoritative(state, source_slot))
    return false;
  const Record& product = state->records[product_slot];
  const Record& source = state->records[source_slot];
  const bool version_product =
      product.lane[0] == kFormProgram &&
      (product.lane[7] & kProgramFlagVersionSpace) != 0u;
  if (version_product) {
    std::uint32_t mismatch = kInvalid;
    if (product.lane[7] !=
            (kProgramFlagEnabled | kProgramFlagVersionSpace) ||
        product.lane[5] != counter.lane[2] ||
        product.lane[6] != counter.lane[5] ||
        source.lane[5] != counter.lane[6] ||
        counter.reserved[0] != source.lane[2] ||
        !causal_version_space_disambiguation_valid(
            state, product_slot, source, &mismatch) ||
        mismatch != counter.lane[4])
      return false;
    if (product_slot_out != nullptr) *product_slot_out = product_slot;
    if (constructor_slot_out != nullptr) *constructor_slot_out = kInvalid;
    return true;
  }
  const std::uint32_t required_flags =
      kProgramFlagEnabled | kProgramFlagResidentEvidenceOnly |
      kProgramFlagCausalGermlineProduct;
  const bool fixed_product = product.lane[0] == kFormProgram;
  const bool span_product = product.lane[0] == kFormSpanProgram;
  if ((!fixed_product && !span_product) ||
      product.lane[7] != required_flags ||
      (fixed_product ? product.lane[3] < kProgramMatureSupport
                     : product.lane[3] < kSpanProgramMatureSupport) ||
      product.lane[5] != counter.lane[5] ||
      source.lane[5] != counter.lane[6] ||
      counter.reserved[0] != source.lane[2])
    return false;
  std::uint32_t witness_count = 0u;
  for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
    const Record& witness = state->records[slot];
    witness_count += witness.matter_q8 != 0u &&
                     witness.lane[0] == kFormCausalProductWitness &&
                     witness.lane[1] == product.lane[1] &&
                     witness.lane[2] == counter.lane[2] &&
                     witness.lane[4] != 0u &&
                     witness.lane[4] != kInvalid &&
                     witness.lane[5] == product.lane[2] &&
                     witness.lane[7] == kCausalGermlineExternal;
  }
  if (witness_count != 1u) return false;
  std::uint32_t constructor_slot = kInvalid;
  for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
    const Record& constructor = state->records[slot];
    if (constructor.matter_q8 != 0u &&
        constructor.lane[0] == kFormCausalConstructor &&
        constructor.lane[1] == counter.lane[2]) {
      if (constructor_slot != kInvalid) return false;
      constructor_slot = slot;
    }
  }
  if (constructor_slot == kInvalid ||
      state->records[constructor_slot].lane[6] != product.lane[0])
    return false;
  std::uint32_t mismatch = kInvalid;
  const bool reproduced =
      fixed_product
          ? causal_product_single_literal_difference(
                state, product_slot, source, &mismatch)
          : causal_span_product_single_literal_difference(
                state, product_slot, constructor_slot, source, &mismatch);
  if (!reproduced ||
      mismatch != counter.lane[4])
    return false;
  if (product_slot_out != nullptr) *product_slot_out = product_slot;
  if (constructor_slot_out != nullptr)
    *constructor_slot_out = constructor_slot;
  return true;
}

BCC32_CAUSAL_GERMLINE_DISPATCH bool
causal_product_has_live_counterevidence(
    const ResidentRewriteState* state, std::uint32_t product_slot) {
  if (state == nullptr || product_slot >= live_record_capacity(state)) return false;
  for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
    std::uint32_t counter_product = kInvalid;
    if (causal_counterevidence_valid(state, slot, &counter_product, nullptr) &&
        counter_product == product_slot)
      return true;
  }
  return false;
}

BCC32_CAUSAL_GERMLINE_DISPATCH bool
causal_exact_program_shadowed_by_version_space(
    const ResidentRewriteState* state, std::uint32_t program_slot) {
  if (state == nullptr || program_slot >= live_record_capacity(state) ||
      !pure_external_exact_program_authoritative(state, program_slot))
    return false;
  const Record& exact = state->records[program_slot];
  for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
    const Record& alternative = state->records[slot];
    if (alternative.matter_q8 == 0u ||
        alternative.lane[0] != kFormProgram ||
        (alternative.lane[7] & kProgramFlagVersionSpace) == 0u ||
        !version_space_program_authoritative(state, slot))
      continue;
    // The VersionSpace extent includes its consequence as the final term.
    // Compare only the shared antecedent prefix: an older exact episode may
    // retain additional post-consequence history, yet it must still be
    // shadowed when its opening causal route is the factorized one now being
    // executed.
    const std::uint32_t antecedent_extent =
        alternative.lane[2] == 0u ? 0u : alternative.lane[2] - 1u;
    if (antecedent_extent == 0u || exact.lane[2] < antecedent_extent)
      continue;
    bool prefix_match = true;
    for (std::uint32_t index = 0u; index < antecedent_extent; ++index) {
      std::uint32_t observed = 0u;
      std::uint32_t expected = 0u;
      std::uint32_t meta = 0u;
      if (!causal_sequence_word_at(state, exact, index, &observed) ||
          !version_space_effective_program_term_at(
              state, alternative, index, &expected, &meta) ||
          observed != expected) {
        prefix_match = false;
        break;
      }
    }
    if (!prefix_match) continue;
    // A factorized VersionSpace surface owns a matching prefix even after
    // counterevidence has selected one surviving consequence.  The exact
    // external Program remains provenance-bearing evidence, but must not
    // re-enter execution as a competing owner on the next fresh trajectory.
    // Requiring a divergent peer here only handled unresolved ambiguity; it
    // let the exact episode program return after resolution and made the
    // public action seam conflict with the selected factorized lineage.
    return true;
  }
  return false;
}

BCC32_CAUSAL_GERMLINE_DISPATCH bool
causal_constructor_has_family_counterevidence(
    const ResidentRewriteState* state, std::uint32_t constructor_slot) {
  if (state == nullptr || constructor_slot >= live_record_capacity(state)) return false;
  for (std::uint32_t seed_slot = 0u; seed_slot < live_record_capacity(state);
       ++seed_slot) {
    std::uint32_t seed_constructor = kInvalid;
    if (!causal_counterevidence_valid(
            state, seed_slot, nullptr, &seed_constructor) ||
        seed_constructor != constructor_slot)
      continue;
    const Record& seed = state->records[seed_slot];
    std::uint32_t product_owner[kCausalGermlineMinimumContributors]{};
    std::uint32_t source_owner[kCausalGermlineMinimumContributors]{};
    std::uint32_t count = 0u;
    for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
      std::uint32_t candidate_constructor = kInvalid;
      if (!causal_counterevidence_valid(
              state, slot, nullptr, &candidate_constructor) ||
          candidate_constructor != constructor_slot ||
          state->records[slot].lane[4] != seed.lane[4])
        continue;
      const Record& candidate = state->records[slot];
      bool independent = true;
      for (std::uint32_t prior = 0u; prior < count; ++prior)
        independent &= product_owner[prior] != candidate.lane[1] &&
                       source_owner[prior] != candidate.lane[3];
      if (!independent) continue;
      product_owner[count] = candidate.lane[1];
      source_owner[count] = candidate.lane[3];
      if (++count == kCausalGermlineMinimumContributors) return true;
    }
  }
  return false;
}

BCC32_CAUSAL_GERMLINE_DISPATCH bool causal_constructor_authoritative(
    const ResidentRewriteState* state, std::uint32_t constructor_slot) {
  if (state == nullptr || constructor_slot >= live_record_capacity(state)) return false;
  const Record& header = state->records[constructor_slot];
  if (header.matter_q8 == 0u ||
      header.lane[0] != kFormCausalConstructor ||
      header.lane[1] == 0u || header.lane[1] == kInvalid ||
      header.lane[2] == 0u || header.lane[3] == 0u ||
      header.lane[4] != kCausalGermlineMinimumContributors ||
      header.lane[5] == 0u || header.lane[5] == kInvalid ||
      header.lane[7] != kCausalGermlineEnabled)
    return false;
  if (header.lane[6] == kFormSpanProgram) {
    std::uint32_t variable_occurrence[kMaximumProgramVariables]{};
    for (std::uint32_t index = 0u; index < header.lane[2]; ++index) {
      const std::uint32_t term_slot = causal_owned_ordinal(
          state, kFormCausalConstructorTerm, header.lane[1], index);
      if (term_slot == kInvalid) return false;
      const Record& term = state->records[term_slot];
      if (term.lane[3] != kSpanTermLiteral &&
          term.lane[3] != kSpanTermVariable)
        return false;
      if (term.lane[3] == kSpanTermVariable) {
        if (term.lane[4] >= header.lane[3]) return false;
        ++variable_occurrence[term.lane[4]];
      } else if (term.lane[4] != 0u) {
        return false;
      }
    }
    for (std::uint32_t variable = 0u; variable < header.lane[3]; ++variable)
      if (variable_occurrence[variable] < 2u) return false;
    if (!resident_constructor_difference_valid(state, constructor_slot) ||
        header.reserved[0] != 3u + header.lane[2] * 3u)
      return false;
    std::uint32_t delta_count = 0u;
    for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
      const Record& delta = state->records[slot];
      if (delta.matter_q8 == 0u ||
          delta.lane[0] != kFormCausalConstructorDelta ||
          delta.lane[1] != header.lane[1])
        continue;
      bool matched =
          delta.lane[3] == kFormSpanProgram &&
          delta.lane[4] == kConstructionDifferenceHeaderOrdinal &&
          ((delta.lane[5] == 3u && delta.lane[6] == 0u &&
            delta.lane[7] == kSpanProgramMatureSupport) ||
           (delta.lane[5] == 4u && delta.lane[6] == 0u &&
            delta.lane[7] == header.lane[3]) ||
           (delta.lane[5] == 7u && delta.lane[6] == 0u &&
            delta.lane[7] ==
                (kProgramFlagEnabled | kProgramFlagResidentEvidenceOnly |
                 kProgramFlagCausalGermlineProduct)));
      for (std::uint32_t index = 0u; !matched && index < header.lane[2];
           ++index) {
        const std::uint32_t term_slot = causal_owned_ordinal(
            state, kFormCausalConstructorTerm, header.lane[1], index);
        if (term_slot == kInvalid) return false;
        const Record& term = state->records[term_slot];
        matched = delta.lane[3] == kFormSpanProgramTerm &&
                  delta.lane[4] == index && delta.lane[6] == 0u &&
                  ((delta.lane[5] == 3u &&
                    delta.lane[7] == term.lane[3]) ||
                   (delta.lane[5] == 4u &&
                    delta.lane[7] == term.lane[4]) ||
                   (delta.lane[5] == 5u &&
                    delta.lane[7] == term.lane[6]));
      }
      if (!matched) return false;
      ++delta_count;
    }
    if (delta_count != header.reserved[0]) return false;
    std::uint32_t witness_count = 0u;
    std::uint32_t left_source_owner[kCausalGermlineMinimumContributors]{};
    std::uint32_t right_source_owner[kCausalGermlineMinimumContributors]{};
    std::uint32_t product_owner[kCausalGermlineMinimumContributors]{};
    for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
      const Record& witness = state->records[slot];
      if (witness.matter_q8 == 0u ||
          witness.lane[0] != kFormCausalConstructorWitness ||
          witness.lane[1] != header.lane[1])
        continue;
      if (witness_count >= kCausalGermlineMinimumContributors ||
          witness.lane[3] != header.lane[5] ||
          witness.lane[7] != kCausalGermlineExternal)
        return false;
      const std::uint32_t episode_slot =
          causal_episode_slot_by_identity(state, witness.lane[2]);
      if (episode_slot == kInvalid ||
          !causal_episode_valid(state, episode_slot))
        return false;
      const Record& episode = state->records[episode_slot];
      if (episode.lane[2] != witness.lane[4] ||
          episode.lane[3] != witness.lane[5] ||
          episode.reserved[0] != witness.lane[6] ||
          (episode.lane[7] & kConstructionEpisodeSpanInduction) == 0u ||
          episode.lane[6] != header.lane[5])
        return false;
      for (std::uint32_t prior = 0u; prior < witness_count; ++prior)
        if (left_source_owner[prior] == witness.lane[5] ||
            left_source_owner[prior] == witness.lane[6] ||
            right_source_owner[prior] == witness.lane[5] ||
            right_source_owner[prior] == witness.lane[6] ||
            product_owner[prior] == witness.lane[4])
          return false;
      left_source_owner[witness_count] = witness.lane[5];
      right_source_owner[witness_count] = witness.lane[6];
      product_owner[witness_count] = witness.lane[4];
      ++witness_count;
    }
    return witness_count == kCausalGermlineMinimumContributors &&
           !causal_constructor_has_family_counterevidence(
               state, constructor_slot);
  }
  if (header.lane[6] != kFormProgram) return false;
  std::uint32_t variable_occurrence[kMaximumProgramVariables]{};
  for (std::uint32_t index = 0u; index < header.lane[2]; ++index) {
    const std::uint32_t term_slot = causal_owned_ordinal(
        state, kFormCausalConstructorTerm, header.lane[1], index);
    if (term_slot == kInvalid) return false;
    const Record& term = state->records[term_slot];
    if (term.lane[3] > header.lane[3]) return false;
    if (term.lane[3] != 0u) ++variable_occurrence[term.lane[3] - 1u];
  }
  for (std::uint32_t variable = 0u; variable < header.lane[3]; ++variable)
    if (variable_occurrence[variable] < 2u) return false;
  if (!resident_constructor_difference_valid(state, constructor_slot))
    return false;
  const std::uint32_t exact_flags =
      kProgramFlagEnabled | kProgramFlagPureExternalExact;
  const std::uint32_t product_flags =
      kProgramFlagEnabled | kProgramFlagResidentEvidenceOnly |
      kProgramFlagCausalGermlineProduct;
  bool header_variables = false;
  bool header_flags = false;
  std::uint32_t variable_deltas = 0u;
  for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
    const Record& delta = state->records[slot];
    if (delta.matter_q8 == 0u ||
        delta.lane[0] != kFormCausalConstructorDelta ||
        delta.lane[1] != header.lane[1])
      continue;
    if (delta.lane[3] == kFormProgram &&
        delta.lane[4] == kConstructionDifferenceHeaderOrdinal &&
        delta.lane[5] == 4u && delta.lane[6] == 0u &&
        delta.lane[7] == header.lane[3]) {
      header_variables = true;
      continue;
    }
    if (delta.lane[3] == kFormProgram &&
        delta.lane[4] == kConstructionDifferenceHeaderOrdinal &&
        delta.lane[5] == 7u && delta.lane[6] == exact_flags &&
        delta.lane[7] == product_flags) {
      header_flags = true;
      continue;
    }
    bool matched = false;
    for (std::uint32_t index = 0u; index < header.lane[2]; ++index) {
      const std::uint32_t term_slot = causal_owned_ordinal(
          state, kFormCausalConstructorTerm, header.lane[1], index);
      if (term_slot == kInvalid) return false;
      const std::uint32_t meta = state->records[term_slot].lane[3];
      matched |= meta != 0u && delta.lane[3] == kFormProgramTerm &&
                 delta.lane[4] == index / 2u &&
                 delta.lane[5] == 3u + (index % 2u) * 2u &&
                 delta.lane[6] == 0u && delta.lane[7] == meta;
    }
    if (!matched) return false;
    ++variable_deltas;
  }
  std::uint32_t expected_variable_deltas = 0u;
  for (std::uint32_t variable = 0u; variable < header.lane[3]; ++variable)
    expected_variable_deltas += variable_occurrence[variable];
  if (!header_variables || !header_flags ||
      variable_deltas != expected_variable_deltas)
    return false;
  std::uint32_t witness_count = 0u;
  std::uint32_t left_source_owner[kCausalGermlineMinimumContributors]{};
  std::uint32_t right_source_owner[kCausalGermlineMinimumContributors]{};
  std::uint32_t product_owner[kCausalGermlineMinimumContributors]{};
  for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
    const Record& witness = state->records[slot];
    if (witness.matter_q8 == 0u ||
        witness.lane[0] != kFormCausalConstructorWitness ||
        witness.lane[1] != header.lane[1])
      continue;
    if (witness_count >= kCausalGermlineMinimumContributors ||
        witness.lane[3] != header.lane[5] ||
        witness.lane[7] != kCausalGermlineExternal)
      return false;
    const std::uint32_t episode_slot =
        causal_episode_slot_by_identity(state, witness.lane[2]);
    if (episode_slot == kInvalid || !causal_episode_valid(state, episode_slot))
      return false;
    const Record& episode = state->records[episode_slot];
    if (episode.lane[2] != witness.lane[4] ||
        episode.lane[3] != witness.lane[5] ||
        episode.reserved[0] != witness.lane[6] ||
        (episode.lane[7] & kConstructionEpisodePairInduction) == 0u ||
        episode.lane[6] != header.lane[5])
      return false;
    for (std::uint32_t prior = 0u; prior < witness_count; ++prior)
      if (left_source_owner[prior] == witness.lane[5] ||
          left_source_owner[prior] == witness.lane[6] ||
          right_source_owner[prior] == witness.lane[5] ||
          right_source_owner[prior] == witness.lane[6] ||
          product_owner[prior] == witness.lane[4])
        return false;
    left_source_owner[witness_count] = witness.lane[5];
    right_source_owner[witness_count] = witness.lane[6];
    product_owner[witness_count] = witness.lane[4];
    ++witness_count;
  }
  return witness_count == kCausalGermlineMinimumContributors &&
         !causal_constructor_has_family_counterevidence(
             state, constructor_slot);
}

BCC32_CAUSAL_GERMLINE_DISPATCH void append_causal_constructor_delta(
    ResidentRewriteState* state, std::uint32_t constructor_owner,
    std::uint32_t ordinal, std::uint32_t target_form,
    std::uint32_t target_ordinal, std::uint32_t target_word,
    std::uint32_t before, std::uint32_t after) {
  const std::uint32_t delta_slot = allocate_record(state);
  Record& delta = state->records[delta_slot];
  delta.lane[0] = kFormCausalConstructorDelta;
  delta.lane[1] = constructor_owner;
  delta.lane[2] = ordinal;
  delta.lane[3] = target_form;
  delta.lane[4] = target_ordinal;
  delta.lane[5] = target_word;
  delta.lane[6] = before;
  delta.lane[7] = after;
  delta.reserved[0] = kCausalGermlineEnabled;
  ++delta.revision;
}

BCC32_CAUSAL_GERMLINE_DISPATCH void
withdraw_exact_constructor_source_program(
    ResidentRewriteState* state, std::uint32_t owner) {
  if (state == nullptr || owner == 0u || owner == kInvalid) return;
  const std::uint32_t slot = causal_program_slot_by_owner(state, owner);
  if (slot == kInvalid ||
      !pure_external_exact_program_authoritative(state, slot))
    return;
  clear_owned_records(state, kFormProgramTerm, owner);
  clear_record(&state->records[slot]);
  ++state->revision;
}

BCC32_CAUSAL_GERMLINE_DISPATCH bool settle_causal_germline_constructor(
    ResidentRewriteState* state) {
  if (state == nullptr || state->fault != 0u) return false;
  for (std::uint32_t seed_slot = 0u; seed_slot < live_record_capacity(state); ++seed_slot) {
    const Record& seed = state->records[seed_slot];
    if (seed.matter_q8 == 0u ||
        seed.lane[0] != kFormConstructionEpisode ||
        (seed.lane[7] & (kConstructionEpisodePairInduction |
                         kConstructionEpisodeSpanInduction)) == 0u ||
        !causal_episode_valid(state, seed_slot) ||
        causal_constructor_slot_by_topology(state, seed.lane[6]) != kInvalid)
      continue;
    const bool span_seed =
        (seed.lane[7] & kConstructionEpisodeSpanInduction) != 0u;
    std::uint32_t selected[kCausalGermlineMinimumContributors]{
        kInvalid, kInvalid, kInvalid};
    std::uint32_t selected_count = 0u;
    for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
      const Record& candidate = state->records[slot];
      if (candidate.matter_q8 == 0u ||
          candidate.lane[0] != kFormConstructionEpisode ||
          !causal_episode_valid(state, slot) ||
          !causal_episode_topology_equal(state, seed, candidate))
        continue;
      bool independent = true;
      for (std::uint32_t prior_index = 0u;
           prior_index < selected_count; ++prior_index) {
        const Record& prior = state->records[selected[prior_index]];
        if (prior.lane[2] == candidate.lane[2] ||
            prior.lane[3] == candidate.lane[3] ||
            prior.lane[3] == candidate.reserved[0] ||
            prior.reserved[0] == candidate.lane[3] ||
            prior.reserved[0] == candidate.reserved[0]) {
          independent = false;
          break;
        }
      }
      if (!independent) continue;
      std::uint32_t position = selected_count;
      if (position > kCausalGermlineMinimumContributors)
        position = kCausalGermlineMinimumContributors;
      while (position != 0u) {
        const std::uint32_t prior = selected[position - 1u];
        if (prior != kInvalid &&
            state->records[prior].lane[1] < candidate.lane[1])
          break;
        if (position < kCausalGermlineMinimumContributors)
          selected[position] = prior;
        --position;
      }
      if (position < kCausalGermlineMinimumContributors)
        selected[position] = slot;
      if (selected_count < kCausalGermlineMinimumContributors)
        ++selected_count;
    }
    if (selected_count < kCausalGermlineMinimumContributors) continue;
    std::uint32_t variable_delta_count = 0u;
    if (!span_seed) {
      for (std::uint32_t index = 0u; index < seed.lane[4]; ++index) {
        const std::uint32_t source_term = causal_owned_ordinal(
            state, kFormConstructionEpisodeTerm,
            state->records[selected[0]].lane[1], index);
        if (source_term == kInvalid) return false;
        variable_delta_count += state->records[source_term].lane[6] != 0u;
      }
    }
    const std::uint32_t delta_count =
        span_seed ? 3u + seed.lane[4] * 3u
                  : 2u + variable_delta_count;
    const std::uint32_t required =
        1u + seed.lane[4] + kCausalGermlineMinimumContributors +
        delta_count;
    if (!reserve_grounded_record_matter(state, required)) return false;
    std::uint32_t identity = rewrite_mix(
        kFormCausalConstructor, seed.lane[6], seed.lane[4]);
    if (identity == 0u || identity == kInvalid ||
        record_owner_exists(state, identity))
      identity = make_record_owner(state, identity);
    if (identity == kInvalid) return false;
    const std::uint32_t header_slot = allocate_record(state);
    if (header_slot == kInvalid) return false;
    Record& header = state->records[header_slot];
    header.lane[0] = kFormCausalConstructor;
    header.lane[1] = identity;
    header.lane[2] = seed.lane[4];
    header.lane[3] = seed.lane[5];
    header.lane[4] = kCausalGermlineMinimumContributors;
    header.lane[5] = seed.lane[6];
    header.lane[6] = span_seed ? kFormSpanProgram : kFormProgram;
    header.lane[7] = kCausalGermlineEnabled;
    header.reserved[0] = delta_count;
    ++header.revision;
    const Record& selected_seed = state->records[selected[0]];
    for (std::uint32_t index = 0u; index < seed.lane[4]; ++index) {
      const std::uint32_t source_term = causal_owned_ordinal(
          state, kFormConstructionEpisodeTerm, selected_seed.lane[1], index);
      const std::uint32_t term_slot = allocate_record(state);
      if (source_term == kInvalid || term_slot == kInvalid) return false;
      Record& term = state->records[term_slot];
      term.lane[0] = kFormCausalConstructorTerm;
      term.lane[1] = identity;
      term.lane[2] = index;
      term.lane[3] = span_seed
                         ? state->records[source_term].lane[3]
                         : state->records[source_term].lane[6];
      term.lane[4] = span_seed
                         ? state->records[source_term].lane[4]
                         : state->records[source_term].lane[5] &
                               kRawChannelMask;
      term.lane[5] = index + 1u == seed.lane[4] ? 1u : 0u;
      if (span_seed)
        term.lane[6] = state->records[source_term].lane[5];
      term.lane[7] = kCausalGermlineEnabled;
      ++term.revision;
    }
    std::uint32_t delta_ordinal = 0u;
    const std::uint32_t product_flags =
        kProgramFlagEnabled | kProgramFlagResidentEvidenceOnly |
        kProgramFlagCausalGermlineProduct;
    if (span_seed) {
      append_causal_constructor_delta(
          state, identity, delta_ordinal++, kFormSpanProgram,
          kConstructionDifferenceHeaderOrdinal, 3u, 0u,
          kSpanProgramMatureSupport);
      append_causal_constructor_delta(
          state, identity, delta_ordinal++, kFormSpanProgram,
          kConstructionDifferenceHeaderOrdinal, 4u, 0u, seed.lane[5]);
      append_causal_constructor_delta(
          state, identity, delta_ordinal++, kFormSpanProgram,
          kConstructionDifferenceHeaderOrdinal, 7u, 0u, product_flags);
      for (std::uint32_t index = 0u; index < seed.lane[4]; ++index) {
        const std::uint32_t term_slot = causal_owned_ordinal(
            state, kFormCausalConstructorTerm, identity, index);
        const Record& term = state->records[term_slot];
        append_causal_constructor_delta(
            state, identity, delta_ordinal++, kFormSpanProgramTerm, index,
            3u, 0u, term.lane[3]);
        append_causal_constructor_delta(
            state, identity, delta_ordinal++, kFormSpanProgramTerm, index,
            4u, 0u, term.lane[4]);
        append_causal_constructor_delta(
            state, identity, delta_ordinal++, kFormSpanProgramTerm, index,
            5u, 0u, term.lane[6]);
      }
    } else {
      const std::uint32_t exact_flags =
          kProgramFlagEnabled | kProgramFlagPureExternalExact;
      append_causal_constructor_delta(
          state, identity, delta_ordinal++, kFormProgram,
          kConstructionDifferenceHeaderOrdinal, 4u, 0u, seed.lane[5]);
      append_causal_constructor_delta(
          state, identity, delta_ordinal++, kFormProgram,
          kConstructionDifferenceHeaderOrdinal, 7u, exact_flags,
          product_flags);
      for (std::uint32_t index = 0u; index < seed.lane[4]; ++index) {
        const std::uint32_t term_slot = causal_owned_ordinal(
            state, kFormCausalConstructorTerm, identity, index);
        const std::uint32_t meta = state->records[term_slot].lane[3];
        if (meta != 0u)
          append_causal_constructor_delta(
              state, identity, delta_ordinal++, kFormProgramTerm,
              index / 2u, 3u + (index % 2u) * 2u, 0u, meta);
      }
    }
    for (std::uint32_t contributor = 0u;
         contributor < kCausalGermlineMinimumContributors; ++contributor) {
      const Record& episode = state->records[selected[contributor]];
      const std::uint32_t witness_slot = allocate_record(state);
      if (witness_slot == kInvalid) return false;
      Record& witness = state->records[witness_slot];
      witness.lane[0] = kFormCausalConstructorWitness;
      witness.lane[1] = identity;
      witness.lane[2] = episode.lane[1];
      witness.lane[3] = episode.lane[6];
      witness.lane[4] = episode.lane[2];
      witness.lane[5] = episode.lane[3];
      witness.lane[6] = episode.reserved[0];
      witness.lane[7] = kCausalGermlineExternal;
      ++witness.revision;
    }
    ++state->revision;
    if (span_seed && causal_constructor_authoritative(state, header_slot)) {
      // Once three independent grounded products have crystallized one live
      // Constructor, their literal source Programs are redundant. Preserve
      // the ordinary SpanPrograms, construction episodes, and exact witness
      // topology; withdraw only the source copies that would otherwise make
      // every later discourse compete with the ignition document for matter.
      for (std::uint32_t contributor = 0u;
           contributor < kCausalGermlineMinimumContributors; ++contributor) {
        const Record episode = state->records[selected[contributor]];
        withdraw_exact_constructor_source_program(state, episode.lane[3]);
        withdraw_exact_constructor_source_program(state,
                                                  episode.reserved[0]);
      }
    }
    return true;
  }
  return false;
}

BCC32_CAUSAL_GERMLINE_DISPATCH bool causal_germline_product_authoritative(
    const ResidentRewriteState* state, std::uint32_t program_slot) {
  if (state == nullptr || program_slot >= live_record_capacity(state)) return false;
  const Record& program = state->records[program_slot];
  const std::uint32_t required_flags =
      kProgramFlagEnabled | kProgramFlagResidentEvidenceOnly |
      kProgramFlagCausalGermlineProduct;
  if (program.matter_q8 == 0u || program.lane[0] != kFormProgram ||
      program.lane[7] != required_flags ||
      program.lane[3] < kProgramMatureSupport || program.lane[4] == 0u)
    return false;
  std::uint32_t witness_slot = kInvalid;
  std::uint32_t witness_count = 0u;
  for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
    const Record& witness = state->records[slot];
    if (witness.matter_q8 != 0u &&
        witness.lane[0] == kFormCausalProductWitness &&
        witness.lane[1] == program.lane[1]) {
      witness_slot = slot;
      ++witness_count;
    }
  }
  if (witness_count != 1u) return false;
  const Record& witness = state->records[witness_slot];
  const std::uint32_t source_slot =
      causal_program_slot_by_owner(state, witness.lane[6]);
  // Fixed products transform the exact externally grounded Program in place,
  // so its witness owner intentionally resolves back to the product. Span
  // products allocate separately and retain a distinct live exact source.
  const bool transformed_external_source =
      witness.lane[6] == program.lane[1] && witness.lane[3] != 0u &&
      witness.lane[3] != kInvalid;
  const bool distinct_external_source =
      source_slot != kInvalid &&
      pure_external_exact_program_authoritative(state, source_slot) &&
      state->records[source_slot].lane[5] == witness.lane[3];
  if ((!transformed_external_source && !distinct_external_source) ||
      witness.lane[5] != program.lane[2] ||
      witness.lane[7] != kCausalGermlineExternal ||
      causal_topology_digest(state, program) != witness.lane[4])
    return false;
  return fixed_program_identity(state, program) == program.lane[5] &&
         !causal_product_has_live_counterevidence(state, program_slot);
}

enum class GroundedCounterevidenceStatus : std::uint32_t {
  kNotApplicable = 0u,
  kReady = 1u,
  kBlocked = 2u,
};

struct GroundedCounterevidencePlan {
  std::uint32_t product_slot = kInvalid;
  std::uint32_t product_owner = kInvalid;
  std::uint32_t constructor_slot = kInvalid;
  std::uint32_t constructor_owner = kInvalid;
  std::uint32_t source_owner = kInvalid;
  std::uint32_t mismatch_ordinal = kInvalid;
  std::uint32_t product_identity = kInvalid;
  std::uint32_t source_digest = kInvalid;
  std::uint32_t extent = 0u;
};

BCC32_CAUSAL_GERMLINE_DISPATCH GroundedCounterevidenceStatus
preflight_grounded_counterevidence(
    const ResidentRewriteState* state, std::uint32_t trajectory_slot,
    GroundedCounterevidencePlan* output, bool causal_products_only = false) {
  if (output != nullptr) *output = GroundedCounterevidencePlan{};
  if (state == nullptr || output == nullptr ||
      trajectory_slot >= live_record_capacity(state))
    return GroundedCounterevidenceStatus::kNotApplicable;
  const Record& source = state->records[trajectory_slot];
  // This is an admissibility reader, not a stack-backed word workspace. Every
  // later read goes through trajectory_word_at(), so a page-spanning external
  // source must not be rejected merely because its logical extent exceeds one
  // physical page.
  if (source.matter_q8 == 0u || source.lane[0] != kFormTrajectory ||
      source.lane[1] == 0u || source.lane[1] == kInvalid ||
      source.lane[2] == 0u ||
      source.lane[3] != 0u || source.lane[4] != 0u ||
      source.lane[5] != kInvalid || source.lane[7] != 0u)
    return GroundedCounterevidenceStatus::kNotApplicable;

  GroundedCounterevidencePlan selected{};
  if (!causal_products_only) {
    std::uint32_t version_agreeing = 0u;
    std::uint32_t version_disagreeing = 0u;
    std::uint32_t agreeing_factor = kInvalid;
    for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
      const Record& alternative = state->records[slot];
      if (alternative.matter_q8 == 0u ||
          alternative.lane[0] != kFormProgram ||
          (alternative.lane[7] & kProgramFlagVersionSpace) == 0u ||
          !resident_program_authoritative(state, slot) ||
          alternative.lane[2] != source.lane[2])
        continue;
      std::uint32_t mismatch = kInvalid;
      if (causal_version_space_sequence_relation(
              state, slot, source, false, &mismatch)) {
        ++version_agreeing;
        agreeing_factor = alternative.lane[5];
        continue;
      }
      if (!causal_version_space_sequence_relation(
              state, slot, source, true, &mismatch) ||
          mismatch + 1u != alternative.lane[2])
        continue;
      ++version_disagreeing;
      selected.product_slot = slot;
      selected.product_owner = alternative.lane[1];
      selected.constructor_slot = kInvalid;
      selected.constructor_owner = alternative.lane[5];
      selected.source_owner = source.lane[1];
      selected.mismatch_ordinal = mismatch;
      selected.product_identity = alternative.lane[6];
      selected.extent = source.lane[2];
    }
    if (version_agreeing != 0u &&
        (version_disagreeing != 1u || version_agreeing != 1u ||
         selected.constructor_owner != agreeing_factor))
      return GroundedCounterevidenceStatus::kBlocked;
    if (version_disagreeing == 0u || version_agreeing == 0u)
      selected = GroundedCounterevidencePlan{};
  }
  for (std::uint32_t product_slot = 0u; product_slot < live_record_capacity(state);
       ++product_slot) {
    if (selected.product_slot != kInvalid) break;
    const Record& product = state->records[product_slot];
    const bool fixed_product = product.lane[0] == kFormProgram;
    const bool span_product = product.lane[0] == kFormSpanProgram;
    if (product.matter_q8 == 0u || (!fixed_product && !span_product) ||
        (product.lane[7] & kProgramFlagCausalGermlineProduct) == 0u ||
        (fixed_product && product.lane[2] != source.lane[2]) ||
        (fixed_product
             ? !causal_germline_product_authoritative(state, product_slot)
             : !causal_germline_span_product_authoritative(
                   state, product_slot)))
      continue;
    std::uint32_t witness_slot = kInvalid;
    for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
      const Record& witness = state->records[slot];
      if (witness.matter_q8 == 0u ||
          witness.lane[0] != kFormCausalProductWitness ||
          witness.lane[1] != product.lane[1])
        continue;
      if (witness_slot != kInvalid)
        return GroundedCounterevidenceStatus::kBlocked;
      witness_slot = slot;
    }
    if (witness_slot == kInvalid) continue;
    const std::uint32_t constructor_owner =
        state->records[witness_slot].lane[2];
    std::uint32_t constructor_slot = kInvalid;
    for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
      const Record& constructor = state->records[slot];
      if (constructor.matter_q8 != 0u &&
          constructor.lane[0] == kFormCausalConstructor &&
          constructor.lane[1] == constructor_owner) {
        if (constructor_slot != kInvalid)
          return GroundedCounterevidenceStatus::kBlocked;
        constructor_slot = slot;
      }
    }
    if (constructor_slot == kInvalid ||
        !causal_constructor_authoritative(state, constructor_slot) ||
        state->records[constructor_slot].lane[6] != product.lane[0])
      continue;
    std::uint32_t mismatch = kInvalid;
    const bool reproduced =
        fixed_product
            ? causal_product_single_literal_difference(
                  state, product_slot, source, &mismatch)
            : causal_span_product_single_literal_difference(
                  state, product_slot, constructor_slot, source, &mismatch);
    if (!reproduced) continue;
    if (selected.product_slot != kInvalid &&
        selected.product_slot != product_slot)
      return GroundedCounterevidenceStatus::kBlocked;
    selected.product_slot = product_slot;
    selected.product_owner = product.lane[1];
    selected.constructor_slot = constructor_slot;
    selected.constructor_owner = constructor_owner;
    selected.source_owner = source.lane[1];
    selected.mismatch_ordinal = mismatch;
    selected.product_identity = product.lane[5];
    selected.extent = source.lane[2];
  }
  if (selected.product_slot == kInvalid)
    return GroundedCounterevidenceStatus::kNotApplicable;

  bool post_conversion_slot = free_record_count(state) != 0u;
  std::uint32_t digest = rewrite_mix(kFormProgram, source.lane[2], 0u);
  for (std::uint32_t index = 0u; index < source.lane[2]; ++index) {
    std::uint32_t word = 0u;
    if (!trajectory_word_at(state, source.lane[1], index, &word))
      return GroundedCounterevidenceStatus::kBlocked;
    digest = rewrite_mix(digest, word, index);
  }
  for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
    const Record& record = state->records[slot];
    post_conversion_slot |= record.matter_q8 != 0u &&
                            record.lane[0] == kFormTrajectoryProvenance &&
                            record.lane[1] == source.lane[1] &&
                            record.lane[3] == kProvenanceExternalOrigin;
  }
  if (!post_conversion_slot)
    return GroundedCounterevidenceStatus::kBlocked;
  selected.source_digest = digest;
  *output = selected;
  return GroundedCounterevidenceStatus::kReady;
}

BCC32_CAUSAL_GERMLINE_DISPATCH bool commit_grounded_counterevidence(
    ResidentRewriteState* state, std::uint32_t exact_source_slot,
    const GroundedCounterevidencePlan& plan) {
  const bool version_plan = plan.constructor_slot == kInvalid;
  if (state == nullptr || exact_source_slot >= live_record_capacity(state) ||
      plan.product_slot >= live_record_capacity(state) ||
      (!version_plan && plan.constructor_slot >= live_record_capacity(state)) ||
      !pure_external_exact_program_authoritative(state, exact_source_slot))
    return false;
  const Record& source = state->records[exact_source_slot];
  if (source.lane[1] != plan.source_owner ||
      source.lane[2] != plan.extent ||
      source.lane[5] != plan.source_digest ||
      state->records[plan.product_slot].lane[1] != plan.product_owner)
    return false;
  const Record& product = state->records[plan.product_slot];
  std::uint32_t mismatch = kInvalid;
  bool reproduced = false;
  if (version_plan) {
    reproduced =
        product.lane[0] == kFormProgram &&
        (product.lane[7] & kProgramFlagVersionSpace) != 0u &&
        product.lane[5] == plan.constructor_owner &&
        product.lane[6] == plan.product_identity &&
        causal_version_space_disambiguation_valid(
            state, plan.product_slot, source, &mismatch);
  } else {
    if ((product.lane[0] != kFormProgram &&
         product.lane[0] != kFormSpanProgram) ||
        product.lane[5] != plan.product_identity ||
        state->records[plan.constructor_slot].lane[1] !=
            plan.constructor_owner ||
        state->records[plan.constructor_slot].lane[6] != product.lane[0])
      return false;
    reproduced =
        product.lane[0] == kFormProgram
            ? causal_product_single_literal_difference(
                  state, plan.product_slot, source, &mismatch)
            : causal_span_product_single_literal_difference(
                  state, plan.product_slot, plan.constructor_slot, source,
                  &mismatch);
  }
  if (!reproduced ||
      mismatch != plan.mismatch_ordinal)
    return false;
  const bool constructor_was_authoritative =
      !version_plan &&
      causal_constructor_authoritative(state, plan.constructor_slot);
  const std::uint32_t allocation_cursor_before = state->allocation_cursor;
  const std::uint32_t fault_before = state->fault;
  const std::uint32_t revision_before = state->revision;
  const std::uint32_t product_suppressions_before =
      state->causal_germline_product_suppressions;
  const std::uint32_t constructor_suppressions_before =
      state->causal_germline_constructor_suppressions;
  const std::uint32_t counter_slot = allocate_record(state);
  if (counter_slot == kInvalid) {
    state->allocation_cursor = allocation_cursor_before;
    state->fault = fault_before;
    return false;
  }
  const Record counter_before = state->records[counter_slot];
  Record& counter = state->records[counter_slot];
  counter.lane[0] = kFormCausalCounterevidence;
  counter.lane[1] = plan.product_owner;
  counter.lane[2] = plan.constructor_owner;
  counter.lane[3] = plan.source_owner;
  counter.lane[4] = plan.mismatch_ordinal;
  counter.lane[5] = plan.product_identity;
  counter.lane[6] = plan.source_digest;
  counter.lane[7] = kCausalGermlineExternal;
  counter.reserved[0] = plan.extent;
  ++counter.revision;
  ++state->revision;
  ++state->causal_germline_product_suppressions;
  if (!version_plan && constructor_was_authoritative &&
      causal_constructor_has_family_counterevidence(
          state, plan.constructor_slot))
    ++state->causal_germline_constructor_suppressions;
  if (causal_counterevidence_valid(state, counter_slot)) return true;

  // kReady is expected to make this branch unreachable, but a failed internal
  // proof must still leave no physical history. Restore the exact empty Record
  // and every observer/allocation field touched above before reporting failure.
  state->records[counter_slot] = counter_before;
  state->allocation_cursor = allocation_cursor_before;
  state->fault = fault_before;
  state->revision = revision_before;
  state->causal_germline_product_suppressions =
      product_suppressions_before;
  state->causal_germline_constructor_suppressions =
      constructor_suppressions_before;
  return false;
}

#include "bcc32_resident_causal_germline_partition.inl"

// Device ingress only records opaque loci. The continuing root settles the
// expensive construction work after admission frames unwind, mirroring the
// existing cross-context factor discipline. Host/direct assays settle inline.
BCC32_CAUSAL_GERMLINE_DISPATCH bool reclaim_orphaned_program_terms(
    ResidentRewriteState* state) {
  if (state == nullptr) return false;
  bool changed = false;
  for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
    Record& record = state->records[slot];
    if (record.matter_q8 == 0u ||
        (record.lane[0] != kFormProgramTerm &&
         record.lane[0] != kFormSpanProgramTerm))
      continue;
    bool live_header = false;
    for (std::uint32_t header_slot = 0u;
         header_slot < live_record_capacity(state); ++header_slot) {
      const Record& header = state->records[header_slot];
      if (header.matter_q8 != 0u &&
          (header.lane[0] == kFormProgram ||
           header.lane[0] == kFormSpanProgram ||
           header.lane[0] == kFormProgramFactor) &&
          header.lane[1] == record.lane[1]) {
        live_header = true;
        break;
      }
    }
    // VersionSpace factors own ordinary ProgramTerm Records and alternatives
    // may deliberately share an owner. Orphan reclamation asks only whether
    // any live header/factor owns the terms; it must not use the unique-product
    // lookup and erase a valid shared molecule.
    if (live_header) continue;
    clear_record(&record);
    changed = true;
  }
  if (changed) ++state->revision;
  return changed;
}

BCC32_CAUSAL_GERMLINE_DISPATCH bool settle_causal_germline_pending(
    ResidentRewriteState* state) {
  if (state == nullptr || state->fault != 0u) return false;
  // A focal header lesion immediately removes executable authority. Its
  // owner-bound term molecule can no longer participate in any match, so
  // return those ordinary Records to free matter before a learned Constructor
  // attempts regeneration. Product witnesses remain as grounded lineage.
  bool changed = reclaim_orphaned_program_terms(state);

  const std::uint32_t reflection_program =
      state->causal_germline_reflection_program;
  const std::uint32_t reflection_source =
      state->causal_germline_reflection_source;
  const bool reflection_conflict =
      reflection_program == kCausalGermlineReflectionConflict;
  state->causal_germline_reflection_program = kInvalid;
  state->causal_germline_reflection_source = kInvalid;
  if (reflection_conflict) {
    ++state->causal_germline_conflict_abstentions;
    changed = true;
  } else if (reflection_program != kInvalid && reflection_source != kInvalid) {
    const std::uint32_t source_slot =
        find_header(state, kFormTrajectory, reflection_source);
    if (source_slot != kInvalid) {
      const Record source = state->records[source_slot];
      const bool reflected = reflect_grounded_program_construction(
          state, reflection_program, source);
      changed |= reflected;
      clear_trajectory(state, source_slot);
      changed |= settle_causal_germline_constructor(state);
    } else {
      ++state->causal_germline_conflict_abstentions;
    }
  }

  const std::uint32_t application_program =
      state->causal_germline_application_program;
  state->causal_germline_application_program = kInvalid;
  if (application_program != kInvalid) {
    changed |=
        apply_causal_germline_to_exact_program(state, application_program);
    if (changed) state->causal_germline_validation_pending = 1u;
  }

  if (state->causal_germline_construction_pending != 0u) {
    state->causal_germline_construction_pending = 0u;
    changed |= settle_causal_germline_constructor(state);
  }

  if (state->causal_germline_validation_pending != 0u) {
    state->causal_germline_validation_pending = 0u;
    for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
      Record& program = state->records[slot];
      if (program.matter_q8 == 0u ||
          (program.lane[0] != kFormProgram &&
           program.lane[0] != kFormSpanProgram) ||
          (program.lane[7] & kProgramFlagCausalGermlineProduct) == 0u)
        continue;
      const bool authoritative =
          program.lane[0] == kFormSpanProgram
              ? causal_germline_span_product_authoritative(state, slot)
              : causal_germline_product_authoritative(state, slot);
      if (!authoritative) {
        // Counterevidence revises executable authority without destroying the
        // product. Keeping its physical organization intact is what permits a
        // focal counterevidence lesion to reopen the prior interpretation.
        if (causal_product_has_live_counterevidence(state, slot))
          continue;
        program.lane[7] &= ~kProgramFlagEnabled;
        ++program.revision;
        ++state->revision;
        changed = true;
      }
    }
  }
  if (changed) refresh_receipt(state);
  return changed;
}
