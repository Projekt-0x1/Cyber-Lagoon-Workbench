BCC32_CAUSAL_GERMLINE_DISPATCH bool
causal_exact_program_word_at(const ResidentRewriteState* state,
                             const Record& source, std::uint32_t index,
                             std::uint32_t* word) {
  std::uint32_t meta = 0u;
  return word != nullptr && index < source.lane[2] &&
         program_term_at(state, source.lane[1], index, word, &meta) &&
         meta == 0u;
}

BCC32_CAUSAL_GERMLINE_DISPATCH bool
span_constructor_partition_sequence_structural(
    const ResidentRewriteState* state, std::uint32_t constructor_slot,
    const Record& source,
    std::uint32_t output_lengths[kMaximumProgramVariables],
    bool* ambiguous) {
  if (ambiguous != nullptr) *ambiguous = false;
  if (state == nullptr || constructor_slot >= live_record_capacity(state) ||
      output_lengths == nullptr ||
      (source.lane[0] != kFormTrajectory &&
       source.lane[0] != kFormProgram) ||
      source.lane[2] == 0u)
    return false;
  const Record& constructor = state->records[constructor_slot];
  if (constructor.matter_q8 == 0u ||
      constructor.lane[0] != kFormCausalConstructor ||
      constructor.lane[6] != kFormSpanProgram ||
      constructor.lane[3] == 0u ||
      constructor.lane[3] > kMaximumProgramVariables)
    return false;
  for (std::uint32_t variable = 0u; variable < kMaximumProgramVariables;
       ++variable)
    output_lengths[variable] = 0u;

  if (constructor.lane[2] > kMaximumSpanProgramTerms) return false;
  std::uint32_t literal_count = 0u;
  std::uint32_t variable_occurrence[kMaximumProgramVariables]{};
  for (std::uint32_t index = 0u; index < constructor.lane[2]; ++index) {
    const std::uint32_t term_slot = causal_owned_ordinal(
        state, kFormCausalConstructorTerm, constructor.lane[1], index);
    if (term_slot == kInvalid)
      return false;
    const Record& term = state->records[term_slot];
    if (term.lane[3] == kSpanTermLiteral) {
      ++literal_count;
    } else if (term.lane[3] == kSpanTermVariable &&
               term.lane[4] < constructor.lane[3]) {
      ++variable_occurrence[term.lane[4]];
    } else {
      return false;
    }
  }
  if (source.lane[2] <= literal_count) return false;
  const std::uint32_t variable_extent = source.lane[2] - literal_count;
  for (std::uint32_t variable = 0u; variable < constructor.lane[3];
       ++variable) {
    if (variable_occurrence[variable] < 2u) return false;
  }
  std::uint32_t maximum_variable_extent = 0u;
  for (std::uint32_t variable = 0u; variable < constructor.lane[3];
       ++variable)
    maximum_variable_extent +=
        variable_occurrence[variable] * kMaximumVariableSpanEvents;
  // This is a constructor-topology bound, not a trajectory/page bound.
  // Reject an extent that this constructor physically cannot express before
  // any source-wide scan. A representable sequence may span any number of
  // trajectory pages; work remains bounded by the constructor's finite term
  // and per-variable-span apertures.
  if (variable_extent > maximum_variable_extent) return false;

  std::uint32_t suffix_minimum[kMaximumProgramVariables + 1u]{};
  for (std::uint32_t variable = constructor.lane[3]; variable != 0u;
       --variable)
    suffix_minimum[variable - 1u] =
        suffix_minimum[variable] + variable_occurrence[variable - 1u];
  if (variable_extent < suffix_minimum[0]) return false;
  std::uint32_t lengths[kMaximumProgramVariables]{};
  std::uint32_t next_length[kMaximumProgramVariables]{};
  std::uint32_t remaining[kMaximumProgramVariables + 1u]{};
  remaining[0] = variable_extent;
  next_length[0] = 1u;
  std::uint32_t depth = 0u;
  std::uint32_t examined = 0u;
  std::uint32_t accepted = 0u;
  while (true) {
    if (depth == constructor.lane[3]) {
      if (remaining[depth] == 0u &&
          span_constructor_partition_matches_lengths(
              state, constructor, source, lengths)) {
        if (++accepted != 1u) {
          if (ambiguous != nullptr) *ambiguous = true;
          return false;
        }
        for (std::uint32_t variable = 0u; variable < constructor.lane[3];
             ++variable)
          output_lengths[variable] = lengths[variable];
      }
      --depth;
      continue;
    }
    const std::uint32_t occurrence = variable_occurrence[depth];
    // Once every earlier extent is fixed, conservation of physical source
    // events determines the final extent exactly. Enumerating all shorter
    // impossible values makes five-variable constructions exhaust the bounded
    // search before equality can reject them. This is only a constraint
    // propagation step; it neither inspects bytes nor raises the work bound.
    if (depth + 1u == constructor.lane[3]) {
      if (next_length[depth] != 1u) {
        next_length[depth] = 1u;
        if (depth == 0u) break;
        --depth;
        continue;
      }
      next_length[depth] = 2u;
      if (remaining[depth] % occurrence != 0u) continue;
      const std::uint32_t forced = remaining[depth] / occurrence;
      if (forced == 0u || forced > kMaximumVariableSpanEvents) continue;
      // The budget counts complete physical partitions presented to the exact
      // matcher, not partial arithmetic prefixes that cannot yet write or
      // conflict with anything.
      if (++examined > kCausalSpanPartitionSearchBudget) {
        if (ambiguous != nullptr) *ambiguous = true;
        return false;
      }
      lengths[depth] = forced;
      std::uint32_t first = 0u;
      if (!span_constructor_first_binding_start(
              state, constructor, lengths, depth, &first) ||
          !causal_sequence_span_repeats(
              state, source, first, forced,
              variable_occurrence[depth]))
        continue;
      remaining[depth + 1u] = 0u;
      ++depth;
      continue;
    }
    const std::uint32_t minimum_after = suffix_minimum[depth + 1u];
    std::uint32_t maximum = 0u;
    if (remaining[depth] > minimum_after)
      maximum = (remaining[depth] - minimum_after) / occurrence;
    if (maximum > kMaximumVariableSpanEvents)
      maximum = kMaximumVariableSpanEvents;
    if (next_length[depth] > maximum) {
      next_length[depth] = 1u;
      if (depth == 0u) break;
      --depth;
      continue;
    }
    const std::uint32_t length = next_length[depth]++;
    lengths[depth] = length;
    std::uint32_t first = 0u;
    if (!span_constructor_first_binding_start(
            state, constructor, lengths, depth, &first) ||
        !causal_sequence_span_repeats(
            state, source, first, length,
            variable_occurrence[depth]))
      continue;
    remaining[depth + 1u] = remaining[depth] - occurrence * length;
    ++depth;
    if (depth < constructor.lane[3]) next_length[depth] = 1u;
  }
  return accepted == 1u;
}

BCC32_CAUSAL_GERMLINE_DISPATCH bool
span_constructor_partition_exact_program(
    const ResidentRewriteState* state, std::uint32_t constructor_slot,
    std::uint32_t exact_program_slot,
    std::uint32_t output_lengths[kMaximumProgramVariables],
    bool* ambiguous = nullptr) {
  return state != nullptr &&
         causal_constructor_authoritative(state, constructor_slot) &&
         exact_program_slot < live_record_capacity(state) &&
         pure_external_exact_program_authoritative(state,
                                                   exact_program_slot) &&
         span_constructor_partition_sequence_structural(
             state, constructor_slot, state->records[exact_program_slot],
             output_lengths, ambiguous);
}

BCC32_CAUSAL_GERMLINE_DISPATCH bool
causal_germline_span_product_authoritative(
    const ResidentRewriteState* state, std::uint32_t program_slot) {
  if (state == nullptr || program_slot >= live_record_capacity(state)) return false;
  const Record& program = state->records[program_slot];
  const std::uint32_t required_flags =
      kProgramFlagEnabled | kProgramFlagResidentEvidenceOnly |
      kProgramFlagCausalGermlineProduct;
  if (program.matter_q8 == 0u || program.lane[0] != kFormSpanProgram ||
      program.lane[7] != required_flags ||
      program.lane[3] < kSpanProgramMatureSupport || program.lane[4] == 0u)
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
  if (witness.lane[3] == 0u || witness.lane[3] == kInvalid ||
      witness.lane[6] == 0u || witness.lane[6] == kInvalid ||
      witness.lane[5] != program.lane[2] ||
      witness.lane[7] != kCausalGermlineExternal ||
      causal_topology_digest(state, program) != witness.lane[4])
    return false;
  // The committed witness is the grounded causal receipt. The exact episode
  // may remain temporarily, but learned authority must survive its physical
  // withdrawal; otherwise every abstraction permanently stores its source.
  if (source_slot != kInvalid &&
      (!pure_external_exact_program_authoritative(state, source_slot) ||
       state->records[source_slot].lane[5] != witness.lane[3]))
    return false;
  std::uint32_t identity =
      rewrite_mix(kFormSpanProgram, program.lane[2], program.lane[4]);
  for (std::uint32_t index = 0u; index < program.lane[2]; ++index) {
    std::uint32_t kind = 0u;
    std::uint32_t value = 0u;
    std::uint32_t channel = 0u;
    std::uint32_t unused = 0u;
    if (!span_program_term_at(state, program.lane[1], index, &kind, &value,
                              &channel, &unused) ||
        (kind != kSpanTermLiteral && kind != kSpanTermVariable))
      return false;
    identity = rewrite_mix(identity, kind, value ^ channel);
  }
  return identity == program.lane[5] &&
         !causal_product_has_live_counterevidence(state, program_slot);
}

BCC32_CAUSAL_GERMLINE_DISPATCH bool
apply_span_causal_germline_to_exact_program(
    ResidentRewriteState* state, std::uint32_t exact_program_slot,
    bool* blocked = nullptr) {
  if (blocked != nullptr) *blocked = false;
  if (state == nullptr || exact_program_slot >= live_record_capacity(state) ||
      !pure_external_exact_program_authoritative(state, exact_program_slot))
    return false;
  const Record source = state->records[exact_program_slot];
  std::uint32_t selected = kInvalid;
  std::uint32_t selected_lengths[kMaximumProgramVariables]{};
  for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
    const Record& constructor = state->records[slot];
    if (constructor.matter_q8 == 0u ||
        constructor.lane[0] != kFormCausalConstructor ||
        constructor.lane[6] != kFormSpanProgram ||
        !causal_constructor_authoritative(state, slot))
      continue;
    std::uint32_t lengths[kMaximumProgramVariables]{};
    bool ambiguous = false;
    if (!span_constructor_partition_exact_program(
            state, slot, exact_program_slot, lengths, &ambiguous)) {
      if (ambiguous) {
        if (blocked != nullptr) *blocked = true;
        ++state->causal_germline_conflict_abstentions;
        return false;
      }
      continue;
    }
    if (selected != kInvalid) {
      if (blocked != nullptr) *blocked = true;
      ++state->causal_germline_conflict_abstentions;
      return false;
    }
    selected = slot;
    for (std::uint32_t variable = 0u; variable < constructor.lane[3];
         ++variable)
      selected_lengths[variable] = lengths[variable];
  }
  if (selected == kInvalid) return false;
  const Record constructor = state->records[selected];
  if (free_record_count(state) < constructor.lane[2] + 2u) return false;
  for (std::uint32_t index = 0u; index < source.lane[2]; ++index) {
    std::uint32_t word = 0u;
    if (!causal_exact_program_word_at(state, source, index, &word))
      return false;
  }
  std::uint32_t identity = rewrite_mix(
      kFormSpanProgram, constructor.lane[2], constructor.lane[3]);
  std::uint32_t cursor = 0u;
  for (std::uint32_t index = 0u; index < constructor.lane[2]; ++index) {
    const std::uint32_t term_slot = causal_owned_ordinal(
        state, kFormCausalConstructorTerm, constructor.lane[1], index);
    if (term_slot == kInvalid) return false;
    const Record& term = state->records[term_slot];
    std::uint32_t value = term.lane[4];
    if (term.lane[3] == kSpanTermLiteral &&
        !causal_exact_program_word_at(state, source, cursor, &value))
      return false;
    identity = rewrite_mix(identity, term.lane[3], value ^ term.lane[6]);
    cursor += term.lane[3] == kSpanTermVariable
                  ? selected_lengths[term.lane[4]]
                  : 1u;
  }
  for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot)
    if (state->records[slot].matter_q8 != 0u &&
        state->records[slot].lane[0] == kFormSpanProgram &&
        state->records[slot].lane[5] == identity)
      return false;
  std::uint32_t owner = identity;
  if (owner == 0u || owner == kInvalid || record_owner_exists(state, owner))
    owner = make_record_owner(state, owner);
  if (owner == kInvalid) return false;
  const std::uint32_t header_slot = allocate_record(state);
  Record& product = state->records[header_slot];
  product.lane[0] = kFormSpanProgram;
  product.lane[1] = owner;
  product.lane[2] = constructor.lane[2];
  product.lane[3] = kSpanProgramMatureSupport;
  product.lane[4] = constructor.lane[3];
  product.lane[5] = identity;
  product.lane[7] = kProgramFlagEnabled |
                    kProgramFlagResidentEvidenceOnly |
                    kProgramFlagCausalGermlineProduct;
  ++product.revision;
  cursor = 0u;
  for (std::uint32_t index = 0u; index < constructor.lane[2]; ++index) {
    const std::uint32_t constructor_term = causal_owned_ordinal(
        state, kFormCausalConstructorTerm, constructor.lane[1], index);
    const Record& pattern = state->records[constructor_term];
    const std::uint32_t term_slot = allocate_record(state);
    Record& term = state->records[term_slot];
    term.lane[0] = kFormSpanProgramTerm;
    term.lane[1] = owner;
    term.lane[2] = index;
    term.lane[3] = pattern.lane[3];
    term.lane[4] = pattern.lane[4];
    if (pattern.lane[3] == kSpanTermLiteral &&
        !causal_exact_program_word_at(state, source, cursor,
                                      &term.lane[4]))
      return false;
    term.lane[5] = pattern.lane[6];
    ++term.revision;
    cursor += pattern.lane[3] == kSpanTermVariable
                  ? selected_lengths[pattern.lane[4]]
                  : 1u;
  }
  const std::uint32_t witness_slot = allocate_record(state);
  Record& witness = state->records[witness_slot];
  witness.lane[0] = kFormCausalProductWitness;
  witness.lane[1] = owner;
  witness.lane[2] = constructor.lane[1];
  witness.lane[3] = source.lane[5];
  witness.lane[4] = constructor.lane[5];
  witness.lane[5] = constructor.lane[2];
  witness.lane[6] = source.lane[1];
  witness.lane[7] = kCausalGermlineExternal;
  ++witness.revision;
  ++state->revision;
  ++state->causal_germline_applications;
  state->causal_germline_product_locus = header_slot;
  if (!causal_germline_span_product_authoritative(state, header_slot))
    return false;
  // The new executable abstraction plus its external witness replaces the
  // literal one-episode Program. This is source withdrawal, not forgetting:
  // the product remains executable and a focal witness lesion still removes
  // its authority.
  clear_owned_records(state, kFormProgramTerm, source.lane[1]);
  clear_record(&state->records[exact_program_slot]);
  ++state->revision;
  return causal_germline_span_product_authoritative(state, header_slot);
}

BCC32_CAUSAL_GERMLINE_DISPATCH bool apply_causal_germline_to_exact_program(
    ResidentRewriteState* state, std::uint32_t program_slot) {
  if (state == nullptr || program_slot >= live_record_capacity(state) ||
      !pure_external_exact_program_authoritative(state, program_slot))
    return false;
  bool span_blocked = false;
  if (apply_span_causal_germline_to_exact_program(
          state, program_slot, &span_blocked))
    return true;
  if (span_blocked) return false;
  Record& program = state->records[program_slot];
  std::uint32_t selected = kInvalid;
  std::uint32_t selected_topology = kInvalid;
  for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
    const Record& constructor = state->records[slot];
    if (constructor.matter_q8 == 0u ||
        constructor.lane[0] != kFormCausalConstructor ||
        constructor.lane[2] != program.lane[2] ||
        !causal_constructor_authoritative(state, slot))
      continue;
    bool compatible = true;
    for (std::uint32_t index = 0u; index < program.lane[2]; ++index) {
      const std::uint32_t term_slot = causal_owned_ordinal(
          state, kFormCausalConstructorTerm, constructor.lane[1], index);
      std::uint32_t word = 0u;
      std::uint32_t meta = 0u;
      compatible &= term_slot != kInvalid &&
                    program_term_at(state, program.lane[1], index, &word,
                                    &meta) &&
                    meta == 0u &&
                    (word & kRawChannelMask) ==
                        state->records[term_slot].lane[4];
    }
    if (!compatible) continue;
    if (selected == kInvalid) {
      selected = slot;
      selected_topology = constructor.lane[5];
    } else if (selected_topology != constructor.lane[5]) {
      ++state->causal_germline_conflict_abstentions;
      return false;
    }
  }
  if (selected == kInvalid || free_record_count(state) == 0u ||
      !preflight_resident_constructor_difference(
          state, selected, program.lane[1]))
    return false;
  const Record constructor = state->records[selected];
  const std::uint32_t exact_digest = program.lane[5];
  bool reconstruction = false;
  for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
    const Record& prior = state->records[slot];
    reconstruction |= prior.matter_q8 != 0u &&
                      prior.lane[0] == kFormCausalProductWitness &&
                      prior.lane[3] == exact_digest;
  }
  const std::uint32_t witness_slot = allocate_record(state);
  if (witness_slot == kInvalid) return false;
  Record& witness = state->records[witness_slot];
  witness.lane[0] = kFormCausalProductWitness;
  witness.lane[1] = program.lane[1];
  witness.lane[2] = constructor.lane[1];
  witness.lane[3] = exact_digest;
  witness.lane[4] = constructor.lane[5];
  witness.lane[5] = program.lane[2];
  witness.lane[6] = program.lane[1];
  witness.lane[7] = kCausalGermlineExternal;
  ++witness.revision;
  commit_resident_constructor_difference(
      state, selected, program.lane[1]);
  program.lane[5] = fixed_program_identity(state, program);
  ++program.revision;
  ++state->revision;
  ++state->causal_germline_applications;
  if (reconstruction) ++state->causal_germline_reconstructions;
  return true;
}
