BCC32_REWRITE_HD inline bool same_raw_channel(std::uint32_t left,
                                              std::uint32_t right) {
  return (left & kRawChannelMask) == (right & kRawChannelMask);
}

BCC32_REWRITE_HD inline std::uint32_t pair_variable(
    const std::uint32_t* left_values, const std::uint32_t* right_values,
    std::uint32_t count, std::uint32_t left, std::uint32_t right) {
  for (std::uint32_t i = 0u; i < count; ++i)
    if (left_values[i] == left && right_values[i] == right) return i;
  return kInvalid;
}

// RWR18 keeps its evidence, shared factors, alternatives, and witnesses in
// the canonical Record population.  The alternative is a Program header with
// Enabled|VersionSpace; its lane[5] names the shared factor, while lane[6] is
// its consequence.  Factor terms, alternative bindings, and witnesses remain
// VersionSpace-only forms, so no second atom table or learner is introduced.
BCC32_REWRITE_HD inline std::uint32_t version_space_popcount(
    std::uint32_t mask) {
  std::uint32_t count = 0u;
  for (std::uint32_t bit = 0u; bit < kVersionSpaceMaximumAtoms; ++bit)
    count += (mask >> bit) & 1u;
  return count;
}

BCC32_REWRITE_HD inline std::uint32_t version_space_allocate_record(
    ResidentRewriteState* state) {
  for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
    Record& record = state->records[slot];
    if (record.matter_q8 != 0u && record.lane[0] == kFormEmpty) return slot;
  }
  state->fault = 0x52571801u;
  return kInvalid;
}

BCC32_REWRITE_HD inline std::uint32_t version_space_evidence_digest(
    const std::uint32_t* atoms, std::uint32_t length,
    std::uint32_t owner, std::uint32_t consequence) {
  std::uint32_t digest = rewrite_mix(kFormProgramEvidence, owner, consequence);
  for (std::uint32_t index = 0u; index < length; ++index)
    digest = rewrite_mix(digest, atoms[index], index);
  return digest == 0u || digest == kInvalid ? digest ^ 0x6d2b79f5u : digest;
}

BCC32_REWRITE_HD inline bool version_space_evidence_record_valid(
    const Record& evidence) {
  if (evidence.matter_q8 == 0u ||
      evidence.lane[0] != kFormProgramEvidence ||
      evidence.lane[1] == 0u || evidence.lane[1] == kInvalid ||
      evidence.lane[2] == 0u ||
      evidence.lane[2] > kVersionSpaceMaximumAtoms ||
      evidence.reserved[0] != 0u || evidence.reserved[1] != 0u)
    return false;
  for (std::uint32_t unused = 4u + evidence.lane[2]; unused < 8u;
       ++unused)
    if (evidence.lane[unused] != 0u) return false;
  return true;
}

BCC32_REWRITE_HD inline bool version_space_evidence_population_valid(
    const ResidentRewriteState* state) {
  if (state == nullptr) return false;
  for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
    const Record& evidence = state->records[slot];
    if (evidence.matter_q8 == 0u ||
        evidence.lane[0] != kFormProgramEvidence)
      continue;
    if (!version_space_evidence_record_valid(evidence)) return false;
  }
  return true;
}

BCC32_REWRITE_HD inline bool version_space_evidence_less(const Record& left,
                                                          const Record& right) {
  if (left.lane[2] != right.lane[2]) return left.lane[2] < right.lane[2];
  for (std::uint32_t index = 0u; index < left.lane[2]; ++index) {
    const std::uint32_t left_word = left.lane[4u + index];
    const std::uint32_t right_word = right.lane[4u + index];
    if (left_word != right_word) return left_word < right_word;
  }
  if (left.lane[3] != right.lane[3]) return left.lane[3] < right.lane[3];
  return left.lane[1] < right.lane[1];
}

BCC32_REWRITE_HD inline bool version_space_factor_term_at(
    const ResidentRewriteState* state, std::uint32_t factor_identity,
    std::uint32_t index, std::uint32_t* word, std::uint32_t* meta) {
  const std::uint32_t block = find_owned_block(
      state, kFormProgramTerm, factor_identity, index / 2u);
  if (block == kInvalid ||
      state->records[block].lane[7] != kProgramFlagVersionSpace)
    return false;
  const std::uint32_t offset = (index % 2u) * 2u;
  *meta = state->records[block].lane[3u + offset];
  *word = state->records[block].lane[4u + offset];
  return true;
}

BCC32_REWRITE_HD inline bool version_space_sort_evidence(
    ResidentRewriteState* state) {
  // Evidence is a compact one-Record form. Sorting only these Records makes
  // raw packet order irrelevant without moving unrelated resident matter.
  // The rescan selection pass avoids a kRecordCapacity thread-stack array in
  // the persistent root.
  if (!version_space_evidence_population_valid(state)) return false;
  for (std::uint32_t destination = 0u; destination < live_record_capacity(state);
       ++destination) {
    if (state->records[destination].matter_q8 == 0u ||
        state->records[destination].lane[0] != kFormProgramEvidence)
      continue;
    std::uint32_t selected = destination;
    for (std::uint32_t candidate = destination + 1u;
         candidate < live_record_capacity(state); ++candidate) {
      if (state->records[candidate].matter_q8 == 0u ||
          state->records[candidate].lane[0] != kFormProgramEvidence)
        continue;
      if (version_space_evidence_less(state->records[candidate],
                                      state->records[selected]))
        selected = candidate;
    }
    if (selected != destination) {
      const Record temporary = state->records[destination];
      state->records[destination] = state->records[selected];
      state->records[selected] = temporary;
    }
  }
  return true;
}

BCC32_REWRITE_HD inline bool version_space_factor_matches(
    const ResidentRewriteState* state, const Record& factor,
    std::uint32_t length, std::uint32_t variable_mask,
    const std::uint32_t* literals) {
  if (factor.matter_q8 == 0u || factor.lane[0] != kFormProgramFactor ||
      factor.lane[2] != length || factor.lane[3] != variable_mask)
    return false;
  for (std::uint32_t index = 0u; index < length; ++index) {
    const std::uint32_t block = find_owned_block(
        state, kFormProgramTerm, factor.lane[1], index / 2u);
    if (block == kInvalid) return false;
    const Record& term = state->records[block];
    const std::uint32_t offset = (index % 2u) * 2u;
    const std::uint32_t expected_meta = (variable_mask >> index) & 1u
                                             ? version_space_popcount(
                                                   variable_mask &
                                                   ((1u << index) - 1u)) + 1u
                                             : 0u;
    if (term.lane[3u + offset] != expected_meta ||
        ((!expected_meta && term.lane[4u + offset] != literals[index]) ||
         (expected_meta && term.lane[4u + offset] !=
                              (literals[index] & kRawChannelMask))))
      return false;
  }
  return true;
}

BCC32_REWRITE_HD inline std::uint32_t version_space_factor_identity(
    std::uint32_t length, std::uint32_t variable_mask,
    const std::uint32_t* literals) {
  std::uint32_t identity = rewrite_mix(kFormProgramFactor, length,
                                       variable_mask);
  for (std::uint32_t index = 0u; index < length; ++index) {
    const std::uint32_t meta = (variable_mask >> index) & 1u
                                   ? version_space_popcount(
                                         variable_mask & ((1u << index) - 1u)) + 1u
                                   : 0u;
    const std::uint32_t value = meta == 0u ? literals[index]
                                           : literals[index] & kRawChannelMask;
    identity = rewrite_mix(identity, value, meta ^ index);
  }
  return identity == 0u || identity == kInvalid ? identity ^ 0x9e3779b9u
                                                  : identity;
}

BCC32_REWRITE_HD inline std::uint32_t version_space_find_factor(
    const ResidentRewriteState* state, std::uint32_t length,
    std::uint32_t variable_mask, const std::uint32_t* literals,
    std::uint32_t identity) {
  for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
    const Record& factor = state->records[slot];
    if (factor.matter_q8 != 0u && factor.lane[0] == kFormProgramFactor &&
        factor.lane[1] == identity &&
        version_space_factor_matches(state, factor, length, variable_mask,
                                     literals))
      return slot;
  }
  return kInvalid;
}

BCC32_REWRITE_HD inline std::uint32_t version_space_install_factor(
    ResidentRewriteState* state, std::uint32_t length,
    std::uint32_t variable_mask, const std::uint32_t* literals) {
  const std::uint32_t identity =
      version_space_factor_identity(length, variable_mask, literals);
  const std::uint32_t existing = version_space_find_factor(
      state, length, variable_mask, literals, identity);
  if (existing != kInvalid) return state->records[existing].lane[1];
  const std::uint32_t blocks = (length + 1u) / 2u;
  if (free_record_count(state) < blocks + 1u) {
    state->fault = 0x52571802u;
    return kInvalid;
  }
  const std::uint32_t header_slot = version_space_allocate_record(state);
  if (header_slot == kInvalid) return kInvalid;
  Record& factor = state->records[header_slot];
  factor.lane[0] = kFormProgramFactor;
  factor.lane[1] = identity;
  factor.lane[2] = length;
  factor.lane[3] = variable_mask;
  factor.lane[4] = version_space_popcount(variable_mask);
  factor.lane[5] = identity;
  factor.lane[7] = kProgramFlagVersionSpace;
  ++factor.revision;
  for (std::uint32_t block_index = 0u; block_index < blocks; ++block_index) {
    const std::uint32_t slot = version_space_allocate_record(state);
    if (slot == kInvalid) return kInvalid;
    Record& term = state->records[slot];
    term.lane[0] = kFormProgramTerm;
    term.lane[1] = identity;
    term.lane[2] = block_index;
    term.lane[7] = kProgramFlagVersionSpace;
    for (std::uint32_t local = 0u; local < 2u; ++local) {
      const std::uint32_t index = block_index * 2u + local;
      if (index >= length) break;
      const std::uint32_t meta = (variable_mask >> index) & 1u
                                     ? version_space_popcount(
                                           variable_mask & ((1u << index) - 1u)) +
                                           1u
                                     : 0u;
      term.lane[3u + local * 2u] = meta;
      term.lane[4u + local * 2u] =
          meta == 0u ? literals[index] : literals[index] & kRawChannelMask;
    }
    ++term.revision;
  }
  return identity;
}

BCC32_REWRITE_HD inline bool version_space_binding_at(
    const ResidentRewriteState* state, std::uint32_t alternative,
    std::uint32_t variable, std::uint32_t* value) {
  for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
    const Record& term = state->records[slot];
    if (term.matter_q8 != 0u && term.lane[0] == kFormProgramAlternativeTerm &&
        term.lane[1] == alternative && term.lane[2] == variable &&
        term.lane[7] == kProgramFlagVersionSpace) {
      *value = term.lane[3];
      return true;
    }
  }
  return false;
}

BCC32_REWRITE_HD inline std::uint32_t version_space_alternative_identity(
    std::uint32_t factor, std::uint32_t consequence,
    std::uint32_t variable_mask, const std::uint32_t* bindings) {
  std::uint32_t identity = rewrite_mix(factor, consequence, variable_mask);
  for (std::uint32_t variable = 0u; variable < kVersionSpaceMaximumAtoms;
       ++variable)
    if ((variable_mask >> variable) & 1u)
      identity = rewrite_mix(identity, bindings[variable], variable);
  return identity == 0u || identity == kInvalid ? identity ^ 0x85ebca6bu
                                                  : identity;
}

BCC32_REWRITE_HD inline std::uint32_t version_space_find_alternative(
    const ResidentRewriteState* state, std::uint32_t factor,
    std::uint32_t consequence, std::uint32_t variable_mask,
    const std::uint32_t* bindings, std::uint32_t identity) {
  for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
    const Record& alternative = state->records[slot];
    if (alternative.matter_q8 == 0u || alternative.lane[0] != kFormProgram ||
        (alternative.lane[7] & kProgramFlagVersionSpace) == 0u ||
        alternative.lane[1] != identity || alternative.lane[5] != factor ||
        alternative.lane[6] != consequence)
      continue;
    bool same = true;
    for (std::uint32_t position = 0u; position < kVersionSpaceMaximumAtoms;
         ++position) {
      if (((variable_mask >> position) & 1u) == 0u) continue;
      const std::uint32_t variable = version_space_popcount(
          variable_mask & ((1u << position) - 1u));
      std::uint32_t value = 0u;
      same &= version_space_binding_at(state, identity, variable, &value) &&
              value == bindings[position];
    }
    if (same) return slot;
  }
  return kInvalid;
}

BCC32_REWRITE_HD inline bool version_space_owner_seen(
    const ResidentRewriteState* state, std::uint32_t alternative,
    std::uint32_t owner) {
  for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
    const Record& witness = state->records[slot];
    if (witness.matter_q8 != 0u && witness.lane[0] == kFormProgramWitness &&
        witness.lane[1] == alternative && witness.lane[2] == owner &&
        witness.lane[7] == kProgramFlagVersionSpace)
      return true;
  }
  return false;
}

BCC32_REWRITE_HD inline std::uint32_t version_space_owner_count(
    const ResidentRewriteState* state, std::uint32_t alternative) {
  std::uint32_t count = 0u;
  for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
    const Record& witness = state->records[slot];
    if (witness.matter_q8 == 0u || witness.lane[0] != kFormProgramWitness ||
        witness.lane[1] != alternative ||
        witness.lane[7] != kProgramFlagVersionSpace)
      continue;
    bool duplicate = false;
    for (std::uint32_t prior = 0u; prior < slot; ++prior) {
      const Record& previous = state->records[prior];
      duplicate |= previous.matter_q8 != 0u &&
                   previous.lane[0] == kFormProgramWitness &&
                   previous.lane[1] == alternative &&
                   previous.lane[2] == witness.lane[2] &&
                   previous.lane[7] == kProgramFlagVersionSpace;
    }
    if (!duplicate && count != 0xffffffffu) ++count;
  }
  return count;
}

BCC32_REWRITE_HD inline bool version_space_lesion_key(
    const ResidentRewriteState* state, std::uint32_t alternative) {
  for (std::uint32_t slot = 0u; slot < state->lesion.count; ++slot) {
    const Record& displaced = state->lesion.displaced[slot];
    if (displaced.lane[0] == kFormProgram &&
        (displaced.lane[7] & kProgramFlagVersionSpace) != 0u &&
        displaced.lane[1] == alternative)
      return true;
  }
  return false;
}

BCC32_REWRITE_HD inline bool version_space_add_witness(
    ResidentRewriteState* state, std::uint32_t alternative,
    std::uint32_t owner, std::uint32_t evidence_digest) {
  if (version_space_owner_seen(state, alternative, owner)) return true;
  const std::uint32_t slot = version_space_allocate_record(state);
  if (slot == kInvalid) return false;
  Record& witness = state->records[slot];
  witness.lane[0] = kFormProgramWitness;
  witness.lane[1] = alternative;
  witness.lane[2] = owner;
  witness.lane[3] = evidence_digest;
  witness.lane[7] = kProgramFlagVersionSpace;
  ++witness.revision;
  return true;
}

BCC32_REWRITE_HD inline bool version_space_add_alternative(
    ResidentRewriteState* state, std::uint32_t factor, std::uint32_t length,
    std::uint32_t variable_mask, const std::uint32_t* bindings,
    std::uint32_t consequence, std::uint32_t owner,
    std::uint32_t evidence_digest) {
  const std::uint32_t variable_count = version_space_popcount(variable_mask);
  const std::uint32_t identity = version_space_alternative_identity(
      factor, consequence, variable_mask, bindings);
  const std::uint32_t existing = version_space_find_alternative(
      state, factor, consequence, variable_mask, bindings, identity);
  if (existing != kInvalid) {
    Record& alternative = state->records[existing];
    if (!version_space_add_witness(state, identity, owner, evidence_digest))
      return false;
    alternative.lane[3] = version_space_owner_count(state, identity);
    return true;
  }
  // Header, one binding Record per variable, and the first exact witness must
  // be available before construction begins.  Otherwise a full resident
  // ecology could retain a partially installed alternative.
  if (free_record_count(state) < variable_count + 2u) {
    state->fault = 0x52571803u;
    return false;
  }
  const std::uint32_t slot = version_space_allocate_record(state);
  if (slot == kInvalid) return false;
  Record& alternative = state->records[slot];
  alternative.lane[0] = kFormProgram;
  alternative.lane[1] = identity;
  alternative.lane[2] = length + 1u;
  alternative.lane[3] = 0u;
  alternative.lane[4] = variable_count;
  alternative.lane[5] = factor;
  alternative.lane[6] = consequence;
  alternative.lane[7] = kProgramFlagEnabled | kProgramFlagVersionSpace;
  if (version_space_lesion_key(state, identity))
    alternative.lane[7] |= kProgramFlagVersionSpaceLesioned;
  ++alternative.revision;
  for (std::uint32_t position = 0u; position < kVersionSpaceMaximumAtoms;
       ++position) {
    if (((variable_mask >> position) & 1u) == 0u) continue;
    const std::uint32_t variable = version_space_popcount(
        variable_mask & ((1u << position) - 1u));
    const std::uint32_t binding_slot = version_space_allocate_record(state);
    if (binding_slot == kInvalid) return false;
    Record& binding = state->records[binding_slot];
    binding.lane[0] = kFormProgramAlternativeTerm;
    binding.lane[1] = identity;
    binding.lane[2] = variable;
    binding.lane[3] = bindings[position];
    binding.lane[7] = kProgramFlagVersionSpace;
    ++binding.revision;
  }
  if (!version_space_add_witness(state, identity, owner, evidence_digest))
    return false;
  alternative.lane[3] = version_space_owner_count(state, identity);
  return true;
}

BCC32_REWRITE_HD inline void version_space_clear_derived(
    ResidentRewriteState* state) {
  for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
    Record& record = state->records[slot];
    if (record.matter_q8 == 0u) continue;
    if (record.lane[0] == kFormProgramFactor ||
        (record.lane[0] == kFormProgramTerm &&
         (record.lane[7] & kProgramFlagVersionSpace) != 0u) ||
        (record.lane[0] == kFormProgram &&
         (record.lane[7] & kProgramFlagVersionSpace) != 0u) ||
        (record.lane[0] == kFormProgramAlternativeTerm &&
         record.lane[7] == kProgramFlagVersionSpace) ||
        (record.lane[0] == kFormProgramWitness &&
         record.lane[7] == kProgramFlagVersionSpace))
      clear_record(&record);
  }
}

BCC32_REWRITE_HD inline bool version_space_rebuild(
    ResidentRewriteState* state) {
  // Validate every compact evidence Record before clearing or indexing any
  // derived structure.  Malformed resident matter must abstain without an
  // out-of-bounds lane/stack access or a partial rebuild.
  if (!version_space_evidence_population_valid(state)) return false;
  version_space_clear_derived(state);
  for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
    const Record& evidence = state->records[slot];
    if (evidence.matter_q8 == 0u ||
        evidence.lane[0] != kFormProgramEvidence)
      continue;
    std::uint32_t literals[kVersionSpaceMaximumAtoms]{};
    for (std::uint32_t index = 0u; index < evidence.lane[2]; ++index)
      literals[index] = evidence.lane[4u + index];
    const std::uint32_t factor = version_space_install_factor(
        state, evidence.lane[2], 0u, literals);
    if (factor == kInvalid || !version_space_add_alternative(
                                  state, factor, evidence.lane[2], 0u, literals,
                                  evidence.lane[3], evidence.lane[1],
                                  version_space_evidence_digest(
                                      literals, evidence.lane[2], evidence.lane[1],
                                      evidence.lane[3])))
      return false;
  }
  for (std::uint32_t left_slot = 0u; left_slot < live_record_capacity(state); ++left_slot) {
    const Record& left = state->records[left_slot];
    if (left.matter_q8 == 0u || left.lane[0] != kFormProgramEvidence) continue;
    for (std::uint32_t right_slot = left_slot + 1u;
         right_slot < live_record_capacity(state); ++right_slot) {
      const Record& right = state->records[right_slot];
      if (right.matter_q8 == 0u || right.lane[0] != kFormProgramEvidence ||
          right.lane[2] != left.lane[2])
        continue;
      std::uint32_t variable_mask = 0u;
      std::uint32_t bindings_left[kVersionSpaceMaximumAtoms]{};
      std::uint32_t bindings_right[kVersionSpaceMaximumAtoms]{};
      bool compatible = true;
      for (std::uint32_t index = 0u; index < left.lane[2]; ++index) {
        bindings_left[index] = left.lane[4u + index];
        bindings_right[index] = right.lane[4u + index];
        if (bindings_left[index] == bindings_right[index]) continue;
        if ((bindings_left[index] & kRawChannelMask) !=
            (bindings_right[index] & kRawChannelMask)) {
          compatible = false;
          break;
        }
        variable_mask |= 1u << index;
      }
      if (!compatible || variable_mask == 0u) continue;
      const std::uint32_t factor = version_space_install_factor(
          state, left.lane[2], variable_mask, bindings_left);
      if (factor == kInvalid ||
          !version_space_add_alternative(
              state, factor, left.lane[2], variable_mask, bindings_left,
              left.lane[3], left.lane[1],
              version_space_evidence_digest(bindings_left, left.lane[2],
                                            left.lane[1], left.lane[3])) ||
          !version_space_add_alternative(
              state, factor, left.lane[2], variable_mask, bindings_right,
              right.lane[3], right.lane[1],
              version_space_evidence_digest(bindings_right, right.lane[2],
                                            right.lane[1], right.lane[3])))
        return false;
    }
  }
  return true;
}

BCC32_REWRITE_HD inline ProgramEvidenceReceipt admit_external_program_evidence(
    ResidentRewriteState* state, const ProgramEvidenceView& observation) {
  ProgramEvidenceReceipt receipt{};
  if (state == nullptr) return receipt;
  refresh_receipt(state);
  receipt.digest_before = state->organization_digest;
  if (observation.atoms == nullptr || observation.length == 0u ||
      observation.length > kVersionSpaceMaximumAtoms || observation.owner == 0u ||
      observation.owner == kInvalid) {
    receipt.rejected = 1u;
    receipt.digest_after = receipt.digest_before;
    return receipt;
  }
  if (!version_space_evidence_population_valid(state)) {
    receipt.rejected = 1u;
    receipt.digest_after = receipt.digest_before;
    return receipt;
  }
  bool owner_was_witness = false;
  for (std::uint32_t witness_slot = 0u; witness_slot < live_record_capacity(state);
       ++witness_slot) {
    const Record& witness = state->records[witness_slot];
    owner_was_witness |= witness.matter_q8 != 0u &&
                         witness.lane[0] == kFormProgramWitness &&
                         witness.lane[2] == observation.owner &&
                         witness.lane[7] == kProgramFlagVersionSpace;
  }
  for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
    const Record& evidence = state->records[slot];
    if (evidence.matter_q8 == 0u || evidence.lane[0] != kFormProgramEvidence ||
        evidence.lane[1] != observation.owner ||
        evidence.lane[2] != observation.length ||
        evidence.lane[3] != observation.consequence)
      continue;
    bool same = true;
    for (std::uint32_t index = 0u; index < observation.length; ++index)
      same &= evidence.lane[4u + index] == observation.atoms[index];
    if (same) {
      receipt.duplicate = 1u;
      receipt.factor_count = state->version_space_factors;
      receipt.alternative_count = state->version_space_alternatives;
      receipt.mature_alternative_count =
          state->mature_version_space_alternatives;
      receipt.digest_after = receipt.digest_before;
      return receipt;
    }
  }
  const std::uint32_t slot = version_space_allocate_record(state);
  if (slot == kInvalid) {
    receipt.rejected = 1u;
    receipt.digest_after = receipt.digest_before;
    return receipt;
  }
  Record& evidence = state->records[slot];
  evidence.lane[0] = kFormProgramEvidence;
  evidence.lane[1] = observation.owner;
  evidence.lane[2] = observation.length;
  evidence.lane[3] = observation.consequence;
  for (std::uint32_t index = 0u; index < observation.length; ++index)
    evidence.lane[4u + index] = observation.atoms[index];
  ++evidence.revision;
  if (!version_space_sort_evidence(state) || !version_space_rebuild(state)) {
    receipt.rejected = 1u;
    receipt.digest_after = receipt.digest_before;
    return receipt;
  }
  // A physical lesion stays disabled across unrelated rebuilds.  Only a new,
  // exact external observation matching the displaced alternative may restore
  // its authority; replay of the old owner is rejected above as a duplicate.
  for (std::uint32_t alternative_slot = 0u;
       alternative_slot < live_record_capacity(state); ++alternative_slot) {
    Record& alternative = state->records[alternative_slot];
    if (alternative.matter_q8 == 0u || alternative.lane[0] != kFormProgram ||
        (alternative.lane[7] & kProgramFlagVersionSpace) == 0u ||
        (alternative.lane[7] & kProgramFlagVersionSpaceLesioned) == 0u ||
        owner_was_witness ||
        alternative.lane[6] != observation.consequence ||
        alternative.lane[2] != observation.length + 1u)
      continue;
    std::uint32_t factor_slot = kInvalid;
    std::uint32_t factor_count = 0u;
    for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
      const Record& factor = state->records[slot];
      if (factor.matter_q8 == 0u || factor.lane[0] != kFormProgramFactor ||
          factor.lane[1] != alternative.lane[5])
        continue;
      factor_slot = slot;
      ++factor_count;
    }
    if (factor_count != 1u) continue;
    const Record& factor = state->records[factor_slot];
    if (factor.lane[2] != observation.length) continue;
    bool same = true;
    for (std::uint32_t index = 0u; index < observation.length; ++index) {
      const std::uint32_t block = find_owned_block(
          state, kFormProgramTerm, factor.lane[1], index / 2u);
      if (block == kInvalid) {
        same = false;
        break;
      }
      const Record& term = state->records[block];
      const std::uint32_t offset = (index % 2u) * 2u;
      const std::uint32_t meta = term.lane[3u + offset];
      if (meta == 0u) {
        same &= term.lane[4u + offset] == observation.atoms[index];
      } else {
        std::uint32_t binding = 0u;
        same &= version_space_binding_at(
                    state, alternative.lane[1], meta - 1u, &binding) &&
                binding == observation.atoms[index] &&
                (binding & kRawChannelMask) ==
                    (term.lane[4u + offset] & kRawChannelMask);
      }
      if (!same) break;
    }
    if (!same) continue;
    alternative.lane[7] &= ~kProgramFlagVersionSpaceLesioned;
    alternative.lane[7] |= kProgramFlagEnabled;
    ++alternative.revision;
    ++state->revision;
  }
  ++state->revision;
  refresh_receipt(state);
  receipt.committed = 1u;
  receipt.factor_count = state->version_space_factors;
  receipt.alternative_count = state->version_space_alternatives;
  receipt.mature_alternative_count = state->mature_version_space_alternatives;
  receipt.digest_after = state->organization_digest;
  return receipt;
}

BCC32_REWRITE_HD inline bool version_space_factor_structure_valid(
    const ResidentRewriteState* state, const Record& factor,
    std::uint32_t* literals) {
  if (state == nullptr || literals == nullptr || factor.matter_q8 == 0u ||
      factor.lane[0] != kFormProgramFactor ||
      factor.lane[7] != kProgramFlagVersionSpace || factor.lane[1] == 0u ||
      factor.lane[1] == kInvalid ||
      factor.lane[2] == 0u || factor.lane[2] > kVersionSpaceMaximumAtoms ||
      (factor.lane[3] >> factor.lane[2]) != 0u || factor.lane[6] != 0u ||
      factor.lane[4] != version_space_popcount(factor.lane[3]) ||
      factor.lane[5] != factor.lane[1])
    return false;
  const std::uint32_t blocks = (factor.lane[2] + 1u) / 2u;
  for (std::uint32_t index = 0u; index < factor.lane[2]; ++index) {
    std::uint32_t term_slot = kInvalid;
    std::uint32_t term_count = 0u;
    for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
      const Record& term = state->records[slot];
      if (term.matter_q8 == 0u || term.lane[0] != kFormProgramTerm ||
          term.lane[1] != factor.lane[1] || term.lane[2] != index / 2u ||
          term.lane[7] != kProgramFlagVersionSpace)
        continue;
      term_slot = slot;
      ++term_count;
    }
    if (term_count != 1u) return false;
    const Record& term = state->records[term_slot];
    const std::uint32_t offset = (index % 2u) * 2u;
    const std::uint32_t expected_meta =
        ((factor.lane[3] >> index) & 1u)
            ? version_space_popcount(factor.lane[3] & ((1u << index) - 1u)) + 1u
            : 0u;
    if (term.lane[3u + offset] != expected_meta) return false;
    literals[index] = term.lane[4u + offset];
    if (expected_meta != 0u &&
        (literals[index] & ~kRawChannelMask) != 0u)
      return false;
    if (index + 1u == factor.lane[2] && (factor.lane[2] & 1u) != 0u &&
        (term.lane[5] != 0u || term.lane[6] != 0u))
      return false;
  }
  for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
    const Record& term = state->records[slot];
    if (term.matter_q8 == 0u || term.lane[0] != kFormProgramTerm ||
        term.lane[1] != factor.lane[1])
      continue;
    if (term.lane[7] != kProgramFlagVersionSpace || term.lane[2] >= blocks)
      return false;
  }
  return version_space_factor_identity(factor.lane[2], factor.lane[3], literals) ==
         factor.lane[1];
}

BCC32_REWRITE_HD inline bool version_space_program_matches_observation(
    const ResidentRewriteState* state, const Record& program,
    const std::uint32_t* atoms, std::uint32_t length,
    std::uint32_t consequence) {
  if (state == nullptr || atoms == nullptr || program.matter_q8 == 0u ||
      program.lane[0] != kFormProgram ||
      (program.lane[7] & kProgramFlagVersionSpace) == 0u ||
      program.lane[2] != length + 1u || program.lane[6] != consequence)
    return false;
  const Record* factor = nullptr;
  std::uint32_t factor_count = 0u;
  for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
    const Record& candidate = state->records[slot];
    if (candidate.matter_q8 == 0u ||
        candidate.lane[0] != kFormProgramFactor ||
        candidate.lane[1] != program.lane[5])
      continue;
    factor = &candidate;
    ++factor_count;
  }
  if (factor_count != 1u || factor == nullptr || factor->lane[2] != length)
    return false;
  std::uint32_t literals[kVersionSpaceMaximumAtoms]{};
  if (!version_space_factor_structure_valid(state, *factor, literals))
    return false;
  for (std::uint32_t index = 0u; index < length; ++index) {
    const std::uint32_t meta =
        ((factor->lane[3] >> index) & 1u)
            ? version_space_popcount(factor->lane[3] & ((1u << index) - 1u)) + 1u
            : 0u;
    if (meta == 0u) {
      if (atoms[index] != literals[index]) return false;
    } else {
      std::uint32_t binding = 0u;
      if (!version_space_binding_at(state, program.lane[1], meta - 1u,
                                    &binding) ||
          atoms[index] != binding ||
          (atoms[index] & kRawChannelMask) !=
              (literals[index] & kRawChannelMask))
        return false;
    }
  }
  return true;
}

BCC32_REWRITE_HD inline bool version_space_program_grounded(
    const ResidentRewriteState* state, std::uint32_t program_slot,
    bool require_enabled) {
  if (state == nullptr || program_slot >= live_record_capacity(state)) return false;
  const Record& program = state->records[program_slot];
  const bool flags_valid =
      program.lane[7] == (kProgramFlagEnabled | kProgramFlagVersionSpace) ||
      (!require_enabled && program.lane[7] == kProgramFlagVersionSpace);
  if (program.matter_q8 == 0u || program.lane[0] != kFormProgram ||
      !flags_valid ||
      program.lane[1] == 0u || program.lane[1] == kInvalid ||
      program.lane[5] == 0u || program.lane[5] == kInvalid ||
      program.lane[1] == program.lane[5] ||
      program.lane[4] > kVersionSpaceMaximumAtoms)
    return false;
  const std::uint32_t expected_length = program.lane[2] - 1u;
  if (program.lane[2] == 0u || expected_length > kVersionSpaceMaximumAtoms)
    return false;

  std::uint32_t program_count = 0u;
  for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
    const Record& candidate = state->records[slot];
    program_count += candidate.matter_q8 != 0u &&
                     candidate.lane[0] == kFormProgram &&
                     candidate.lane[1] == program.lane[1];
  }
  if (program_count != 1u) return false;

  const Record* factor = nullptr;
  std::uint32_t factor_count = 0u;
  for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
    const Record& candidate = state->records[slot];
    if (candidate.matter_q8 == 0u ||
        candidate.lane[0] != kFormProgramFactor ||
        candidate.lane[1] != program.lane[5])
      continue;
    factor = &candidate;
    ++factor_count;
  }
  if (factor_count != 1u || factor == nullptr ||
      factor->lane[2] != expected_length)
    return false;
  std::uint32_t literals[kVersionSpaceMaximumAtoms]{};
  if (!version_space_factor_structure_valid(state, *factor, literals) ||
      factor->lane[4] != program.lane[4])
    return false;

  // Bindings are an exact, unique projection of the factor's variable mask;
  // no ordinary term or duplicate binding may satisfy a hash collision.
  const std::uint32_t variable_count =
      version_space_popcount(factor->lane[3]);
  for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
    const Record& binding = state->records[slot];
    if (binding.matter_q8 == 0u ||
        binding.lane[0] != kFormProgramAlternativeTerm ||
        binding.lane[1] != program.lane[1])
      continue;
    if (binding.lane[7] != kProgramFlagVersionSpace ||
        binding.lane[2] >= variable_count || binding.lane[4] != 0u ||
        binding.lane[5] != 0u || binding.lane[6] != 0u)
      return false;
  }
  std::uint32_t bindings[kVersionSpaceMaximumAtoms]{};
  for (std::uint32_t position = 0u; position < expected_length; ++position) {
    if (((factor->lane[3] >> position) & 1u) == 0u) continue;
    const std::uint32_t variable = version_space_popcount(
        factor->lane[3] & ((1u << position) - 1u));
    std::uint32_t binding_count = 0u;
    for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
      const Record& binding = state->records[slot];
      if (binding.matter_q8 == 0u ||
          binding.lane[0] != kFormProgramAlternativeTerm ||
          binding.lane[1] != program.lane[1] ||
          binding.lane[2] != variable)
        continue;
      ++binding_count;
      if ((binding.lane[3] & kRawChannelMask) !=
          (literals[position] & kRawChannelMask))
        return false;
      bindings[position] = binding.lane[3];
    }
    if (binding_count != 1u) return false;
  }
  if (version_space_alternative_identity(
          factor->lane[1], program.lane[6], factor->lane[3], bindings) !=
      program.lane[1])
    return false;

  // Every live witness must be unique by owner and point to one exact,
  // independently admitted evidence record for this factor/consequence.
  std::uint32_t witness_count = 0u;
  for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
    const Record& witness = state->records[slot];
    if (witness.matter_q8 == 0u || witness.lane[0] != kFormProgramWitness ||
        witness.lane[1] != program.lane[1])
      continue;
    if (witness.lane[7] != kProgramFlagVersionSpace || witness.lane[2] == 0u ||
        witness.lane[2] == kInvalid || witness.lane[4] != 0u ||
        witness.lane[5] != 0u || witness.lane[6] != 0u)
      return false;
    for (std::uint32_t prior = 0u; prior < slot; ++prior) {
      const Record& previous = state->records[prior];
      if (previous.matter_q8 != 0u &&
          previous.lane[0] == kFormProgramWitness &&
          previous.lane[1] == program.lane[1] &&
          previous.lane[2] == witness.lane[2])
        return false;
    }
    std::uint32_t owner_evidence_count = 0u;
    std::uint32_t evidence_match_count = 0u;
    for (std::uint32_t evidence_slot = 0u;
         evidence_slot < live_record_capacity(state); ++evidence_slot) {
      const Record& evidence = state->records[evidence_slot];
      if (evidence.matter_q8 == 0u ||
          evidence.lane[0] != kFormProgramEvidence ||
          evidence.lane[1] != witness.lane[2])
        continue;
      ++owner_evidence_count;
      if (!version_space_evidence_record_valid(evidence)) return false;
      if (evidence.lane[2] != expected_length ||
          evidence.lane[3] != program.lane[6] ||
          version_space_evidence_digest(&evidence.lane[4], evidence.lane[2],
                                        evidence.lane[1], evidence.lane[3]) !=
              witness.lane[3])
        continue;
      bool matches = true;
      for (std::uint32_t index = 0u; index < expected_length; ++index) {
        if (((factor->lane[3] >> index) & 1u) == 0u) {
          matches &= evidence.lane[4u + index] == literals[index];
        } else {
          matches &= evidence.lane[4u + index] == bindings[index];
        }
      }
      evidence_match_count += matches;
    }
    if (owner_evidence_count != 1u || evidence_match_count != 1u) return false;
    ++witness_count;
  }
  return witness_count >= kVersionSpaceMatureSupport &&
         program.lane[3] == witness_count;
}

BCC32_REWRITE_HD inline bool version_space_program_authoritative(
    const ResidentRewriteState* state, std::uint32_t program_slot) {
  return version_space_program_grounded(state, program_slot, true);
}

// One subtype-aware effective Program view is shared by later resident
// consumers.  VersionSpace keeps its compact factor/binding storage, but callers
// observe the bound antecedent followed by the consequence.  The unchecked
// helper is used only after the caller has established canonical authority.
BCC32_REWRITE_HD inline bool version_space_effective_program_term_at(
    const ResidentRewriteState* state, const Record& program,
    std::uint32_t index, std::uint32_t* word, std::uint32_t* meta) {
  if (state == nullptr || word == nullptr || meta == nullptr ||
      program.matter_q8 == 0u || program.lane[0] != kFormProgram ||
      (program.lane[7] & kProgramFlagVersionSpace) == 0u ||
      index >= program.lane[2])
    return false;
  if (index + 1u == program.lane[2]) {
    *word = program.lane[6];
    *meta = 0u;
    return true;
  }
  std::uint32_t factor_word = 0u;
  std::uint32_t factor_meta = 0u;
  if (!version_space_factor_term_at(state, program.lane[5], index,
                                    &factor_word, &factor_meta))
    return false;
  if (factor_meta == 0u) {
    *word = factor_word;
    *meta = 0u;
    return true;
  }
  std::uint32_t binding = 0u;
  if (!version_space_binding_at(state, program.lane[1], factor_meta - 1u,
                                &binding) ||
      (binding & kRawChannelMask) != (factor_word & kRawChannelMask))
    return false;
  *word = binding;
  *meta = factor_meta;
  return true;
}

BCC32_REWRITE_HD inline bool resident_program_term_at(
    const ResidentRewriteState* state, std::uint32_t program_slot,
    std::uint32_t index, std::uint32_t* word, std::uint32_t* meta) {
  if (state == nullptr || program_slot >= live_record_capacity(state) ||
      word == nullptr ||
      meta == nullptr ||
      !resident_program_authoritative(state, program_slot))
    return false;
  const Record& program = state->records[program_slot];
  if (program.lane[0] != kFormProgram || index >= program.lane[2]) return false;
  if ((program.lane[7] & kProgramFlagVersionSpace) != 0u)
    return version_space_effective_program_term_at(state, program, index, word,
                                                   meta);
  return program_term_at(state, program.lane[1], index, word, meta);
}

BCC32_REWRITE_HD inline std::uint32_t find_version_space_alternative_slot(
    const ResidentRewriteState* state, std::uint32_t consequence,
    std::uint32_t ordinal = 0u) {
  std::uint32_t seen = 0u;
  for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
    const Record& record = state->records[slot];
    if (record.matter_q8 == 0u || record.lane[0] != kFormProgram) continue;
    if ((record.lane[7] & kProgramFlagVersionSpace) == 0u ||
        record.lane[6] != consequence)
      continue;
    if (seen++ == ordinal) return slot;
  }
  return kInvalid;
}

// A literal one-shot Program is the translated resident form of one completed
// external trajectory.  Later factorization reads that executable matter
// directly; it never recreates a Trajectory copy merely to feed the older pair
// inductors.
#if defined(__CUDACC__)
#define BCC32_LITERAL_OBSERVATION_HD static BCC32_REWRITE_HD __noinline__
#else
#define BCC32_LITERAL_OBSERVATION_HD BCC32_REWRITE_HD inline
#endif

BCC32_LITERAL_OBSERVATION_HD bool pure_external_exact_program_authoritative(
    const ResidentRewriteState* state, std::uint32_t program_slot);
BCC32_REWRITE_HD inline std::uint32_t literal_page_count(
    std::uint32_t length);
BCC32_REWRITE_HD inline bool literal_page_owner_at(
    const ResidentRewriteState* state, std::uint32_t root_owner,
    std::uint32_t length, std::uint32_t page,
    std::uint32_t* term_owner);

BCC32_LITERAL_OBSERVATION_HD bool literal_observation_source(
    const ResidentRewriteState* state, std::uint32_t slot) {
  if (state == nullptr || slot >= live_record_capacity(state)) return false;
  const Record& source = state->records[slot];
  if (source.matter_q8 == 0u || source.lane[2] == 0u)
    return false;
  if (source.lane[0] == kFormTrajectory) return source.lane[7] == 0u;
  return source.lane[0] == kFormProgram &&
         (source.lane[7] & kProgramFlagPureExternalExact) != 0u &&
         pure_external_exact_program_authoritative(state, slot);
}

BCC32_REWRITE_HD inline bool literal_observation_shape(
    const ResidentRewriteState* state, std::uint32_t slot,
    bool resident_literal_prevalidated) {
  if (state == nullptr || slot >= live_record_capacity(state)) return false;
  const Record& source = state->records[slot];
  if (source.matter_q8 == 0u || source.lane[2] == 0u)
    return false;
  if (source.lane[0] == kFormTrajectory) return source.lane[7] == 0u;
  return resident_literal_prevalidated && source.lane[0] == kFormProgram &&
         (source.lane[7] & kProgramFlagPureExternalExact) != 0u;
}

BCC32_REWRITE_HD inline bool literal_observation_word_at_prevalidated(
    const ResidentRewriteState* state, std::uint32_t slot,
    std::uint32_t index, std::uint32_t* word) {
  if (state == nullptr || slot >= live_record_capacity(state) ||
      word == nullptr ||
      index >= state->records[slot].lane[2])
    return false;
  const Record& source = state->records[slot];
  if (source.lane[0] == kFormTrajectory)
    return trajectory_word_at(state, source.lane[1], index, word);
  std::uint32_t meta = 0u;
  return program_term_at(state, source.lane[1], index, word, &meta) &&
         meta == 0u;
}

BCC32_REWRITE_HD inline void retire_literal_observation_source(
    ResidentRewriteState* state, std::uint32_t slot) {
  if (state == nullptr || slot >= live_record_capacity(state)) return;
  Record& source = state->records[slot];
  if (source.matter_q8 == 0u) return;
  if (source.lane[0] == kFormTrajectory) {
    clear_trajectory(state, slot);
    return;
  }
  if (source.lane[0] != kFormProgram ||
      (source.lane[7] & kProgramFlagPureExternalExact) == 0u)
    return;
  const std::uint32_t owner = source.lane[1];
  const std::uint32_t pages = literal_page_count(source.lane[2]);
  for (std::uint32_t page = 0u; page < pages && page < kRecordCapacity;
       ++page) {
    std::uint32_t term_owner = kInvalid;
    if (!literal_page_owner_at(state, owner, source.lane[2], page,
                               &term_owner))
      continue;
    for (std::uint32_t term_slot = 0u; term_slot < live_record_capacity(state);
         ++term_slot) {
      Record& term = state->records[term_slot];
      if (term.matter_q8 != 0u && term.lane[0] == kFormProgramTerm &&
          term.lane[1] == term_owner && term.lane[7] == 0u)
        clear_record(&term);
    }
  }
  clear_owned_records(state, kFormTrajectoryPage, owner);
  clear_record(&source);
}

BCC32_LITERAL_OBSERVATION_HD bool induce_version_space_from_pair(
    ResidentRewriteState* state, std::uint32_t left_slot,
    std::uint32_t right_slot,
    bool resident_literal_prevalidated = false,
    bool* divergent_consequence = nullptr) {
  if (divergent_consequence != nullptr) *divergent_consequence = false;
  if (state == nullptr || left_slot >= live_record_capacity(state) ||
      right_slot >= live_record_capacity(state) || left_slot == right_slot)
    return false;
  const Record& left = state->records[left_slot];
  const Record& right = state->records[right_slot];
  if (!literal_observation_shape(state, left_slot,
                                 resident_literal_prevalidated) ||
      !literal_observation_shape(state, right_slot,
                                 resident_literal_prevalidated) ||
      left.lane[2] < 2u ||
      left.lane[2] != right.lane[2])
    return false;

  // Detect divergent consequences before the compact factor representation's
  // four-atom limit.  This path only rescans resident words, so arbitrarily
  // long identical antecedents cannot fall through to fixed induction and
  // create an unbound output.
  bool same_antecedent = true;
  for (std::uint32_t index = 0u; index + 1u < left.lane[2]; ++index) {
    std::uint32_t left_word = 0u;
    std::uint32_t right_word = 0u;
    if (!literal_observation_word_at_prevalidated(
            state, left_slot, index, &left_word) ||
        !literal_observation_word_at_prevalidated(
            state, right_slot, index, &right_word))
      return false;
    same_antecedent &= left_word == right_word;
  }
  std::uint32_t left_consequence = 0u;
  std::uint32_t right_consequence = 0u;
  if (!literal_observation_word_at_prevalidated(
          state, left_slot, left.lane[2] - 1u, &left_consequence) ||
      !literal_observation_word_at_prevalidated(
          state, right_slot, right.lane[2] - 1u, &right_consequence))
    return false;
  if (divergent_consequence != nullptr)
    *divergent_consequence =
        same_antecedent && left_consequence != right_consequence;
  if (left.lane[2] - 1u > kVersionSpaceMaximumAtoms) return false;

  std::uint32_t left_atoms[kVersionSpaceMaximumAtoms]{};
  std::uint32_t right_atoms[kVersionSpaceMaximumAtoms]{};
  for (std::uint32_t index = 0u; index + 1u < left.lane[2]; ++index) {
    if (!literal_observation_word_at_prevalidated(
            state, left_slot, index, &left_atoms[index]) ||
        !literal_observation_word_at_prevalidated(
            state, right_slot, index, &right_atoms[index]))
      return false;
  }
  same_antecedent = true;
  for (std::uint32_t index = 0u; index + 1u < left.lane[2]; ++index)
    same_antecedent &= left_atoms[index] == right_atoms[index];
  if (divergent_consequence != nullptr)
    *divergent_consequence =
        same_antecedent && left_consequence != right_consequence;
  const ProgramEvidenceReceipt first = admit_external_program_evidence(
      state, {left_atoms, left.lane[2] - 1u, left.lane[1], left_consequence});
  const ProgramEvidenceReceipt second = admit_external_program_evidence(
      state,
      {right_atoms, right.lane[2] - 1u, right.lane[1], right_consequence});
  return first.committed != 0u || second.committed != 0u ||
         first.duplicate != 0u || second.duplicate != 0u;
}

// Match a fixed program at one physical start offset. The offset is a
// resident-history boundary, not a semantic selector; callers must establish
// uniqueness across all possible suffix starts before committing.
BCC32_REWRITE_HD inline bool full_program_match_at(
    const ResidentRewriteState* state, const Record& program,
    const Record& trajectory, std::uint32_t start) {
  if ((program.lane[7] & kProgramFlagVersionSpace) != 0u ||
      start > trajectory.lane[2] ||
      program.lane[2] > trajectory.lane[2] - start)
    return false;
  std::uint32_t binding[kMaximumProgramVariables]{};
  std::uint32_t bound = 0u;
  for (std::uint32_t index = 0u; index < program.lane[2]; ++index) {
    std::uint32_t observed = 0u;
    std::uint32_t expected = 0u;
    std::uint32_t meta = 0u;
    if (!trajectory_word_at(state, trajectory.lane[1], start + index, &observed) ||
        !program_term_at(state, program.lane[1], index, &expected, &meta))
      return false;
    if (meta == 0u) {
      if (observed != expected) return false;
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
  return true;
}

BCC32_REWRITE_HD inline bool full_program_match(
    const ResidentRewriteState* state, const Record& program,
    const Record& trajectory) {
  if (program.lane[2] > trajectory.lane[2]) return false;
  const std::uint32_t start = trajectory.lane[2] - program.lane[2];
  return full_program_match_at(state, program, trajectory, start);
}

BCC32_REWRITE_HD inline std::uint32_t fixed_program_identity(
    const ResidentRewriteState* state, const Record& program) {
  if ((program.lane[7] & kProgramFlagVersionSpace) != 0u) return kInvalid;
  std::uint32_t identity = rewrite_mix(
      kFormProgram, program.lane[2], program.lane[4]);
  for (std::uint32_t index = 0u; index < program.lane[2]; ++index) {
    std::uint32_t word = 0u;
    std::uint32_t meta = 0u;
    if (!program_term_at(state, program.lane[1], index, &word, &meta))
      return kInvalid;
    identity = rewrite_mix(identity, word, meta ^ index);
  }
  return identity;
}

BCC32_REWRITE_HD inline std::uint32_t literal_page_count(
    std::uint32_t length) {
  return length == 0u ? 0u :
      1u + (length - 1u) / kTrajectoryPageEvents;
}

BCC32_REWRITE_HD inline std::uint32_t literal_page_extent(
    std::uint32_t length, std::uint32_t page) {
  const std::uint32_t base = page * kTrajectoryPageEvents;
  if (base >= length) return 0u;
  const std::uint32_t remaining = length - base;
  return remaining < kTrajectoryPageEvents ? remaining :
                                              kTrajectoryPageEvents;
}

// Resolve only physically present continuation matter.  Unlike
// trajectory_page_owner(), this never synthesizes a prospective owner while
// validating or reading an already resident literal chain.
BCC32_REWRITE_HD inline bool literal_page_owner_at(
    const ResidentRewriteState* state, std::uint32_t root_owner,
    std::uint32_t length, std::uint32_t page,
    std::uint32_t* term_owner) {
  if (state == nullptr || term_owner == nullptr ||
      page >= literal_page_count(length))
    return false;
  if (page == 0u) {
    *term_owner = root_owner;
    return true;
  }
  std::uint32_t descriptor_slot = kInvalid;
  std::uint32_t descriptor_count = 0u;
  for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
    const Record& candidate = state->records[slot];
    if (candidate.matter_q8 == 0u ||
        candidate.lane[0] != kFormTrajectoryPage ||
        candidate.lane[1] != root_owner || candidate.lane[2] != page)
      continue;
    descriptor_slot = slot;
    ++descriptor_count;
  }
  if (descriptor_count != 1u) return false;
  const Record& descriptor = state->records[descriptor_slot];
  const std::uint32_t extent = literal_page_extent(length, page);
  if (descriptor.lane[3] != page * kTrajectoryPageEvents ||
      descriptor.lane[4] != extent || descriptor.lane[6] == 0u ||
      descriptor.lane[6] == kInvalid || descriptor.lane[6] == root_owner ||
      descriptor.lane[7] != 0u || descriptor.reserved[0] != 0u ||
      descriptor.reserved[1] != 0u)
    return false;
  for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
    const Record& other = state->records[slot];
    if (slot != descriptor_slot && other.matter_q8 != 0u &&
        other.lane[0] == kFormTrajectoryPage &&
        other.lane[1] == root_owner && other.lane[6] == descriptor.lane[6])
      return false;
  }
  *term_owner = descriptor.lane[6];
  return true;
}

BCC32_REWRITE_HD inline bool exact_literal_page_terms(
    const ResidentRewriteState* state, std::uint32_t form,
    std::uint32_t term_owner, std::uint32_t extent) {
  const std::uint32_t blocks = (extent + 1u) / 2u;
  for (std::uint32_t ordinal = 0u; ordinal < blocks; ++ordinal) {
    std::uint32_t count = 0u;
    for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
      const Record& term = state->records[slot];
      if (term.matter_q8 == 0u || term.lane[0] != form ||
          term.lane[1] != term_owner || term.lane[2] != ordinal)
        continue;
      ++count;
      if (term.lane[3] != 0u || term.lane[7] != 0u ||
          term.reserved[0] != 0u || term.reserved[1] != 0u)
        return false;
      if (form == kFormTrajectoryTerm) {
        if (term.lane[6] != 0u ||
            (ordinal + 1u == blocks && (extent & 1u) != 0u &&
             term.lane[5] != 0u))
          return false;
      } else if (form == kFormProgramTerm) {
        if (term.lane[5] != 0u ||
            (ordinal + 1u == blocks && (extent & 1u) != 0u &&
             term.lane[6] != 0u))
          return false;
      } else {
        return false;
      }
    }
    if (count != 1u) return false;
  }
  for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
    const Record& term = state->records[slot];
    if (term.matter_q8 != 0u && term.lane[0] == form &&
        term.lane[1] == term_owner && term.lane[2] >= blocks)
      return false;
  }
  return true;
}

// Validate and port one wholly external completed trajectory in place.  This
// is deliberately a rescan-based transaction: no trajectory-sized scratch or
// second owner is introduced, and no Record is changed until every locus has
// passed the exact topology checks below.
BCC32_REWRITE_HD inline bool convert_pure_external_exact_trajectory(
    ResidentRewriteState* state, std::uint32_t header_slot) {
  if (state == nullptr || header_slot >= live_record_capacity(state)) return false;
  const Record& header = state->records[header_slot];
  if (header.matter_q8 == 0u || header.lane[0] != kFormTrajectory ||
      header.lane[3] != 0u || header.lane[7] != 0u ||
      header.lane[1] == 0u || header.lane[1] == kInvalid ||
      header.lane[2] == 0u || header.lane[4] != 0u ||
      header.lane[5] != kInvalid || header.reserved[0] != 0u ||
      header.reserved[1] != 0u)
    return false;

  const std::uint32_t owner = header.lane[1];
  const std::uint32_t pages = literal_page_count(header.lane[2]);
  if (pages == 0u || pages > kRecordCapacity) return false;
  std::uint32_t current_headers = 0u;
  std::uint32_t owner_records = 0u;
  std::uint32_t evidence_records = 0u;
  std::uint32_t source_witness_records = 0u;
  bool has_provenance = false;
  for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
    const Record& record = state->records[slot];
    if (record.matter_q8 == 0u) continue;
    if (record.lane[0] == kFormTrajectory && record.lane[3] == 0u) {
      ++current_headers;
      if (slot != header_slot) return false;
    }
    if (record.lane[1] != owner) continue;
    ++owner_records;
    if (slot == header_slot) continue;
    if (record.lane[0] == kFormTrajectoryProvenance) {
      has_provenance = true;
      const std::uint32_t valid_lanes =
          record.lane[kProvenanceValidityLane] & 0x3u;
      if (record.lane[2] >= (header.lane[2] + 1u) / 2u ||
          record.lane[3] != kProvenanceExternalOrigin ||
          record.lane[4] != kInvalid ||
          record.lane[5] != kProvenanceExternalOrigin ||
          record.lane[6] != kInvalid ||
          (record.lane[kProvenanceValidityLane] & ~0x3u) != 0u ||
          ((valid_lanes & 0x1u) == 0u && record.reserved[0] != kInvalid) ||
          ((valid_lanes & 0x2u) == 0u && record.reserved[1] != kInvalid))
        return false;
      continue;
    }
    if (record.lane[0] == kFormTransformationWitness) {
      // The bounded production END relation stage commits one exact source
      // witness before this conversion runs. It is lineage for the consumed
      // external trajectory, not part of the executable Program chain. Keep
      // it strict and owner-bound; the relation reader may need this compact
      // provenance handoff after the trajectory pages become Program matter.
      ++source_witness_records;
      if (source_witness_records > 1u ||
          !resident_causal_relation_source_witness::is_witness(record, owner) ||
          record.lane[2] != header.lane[2])
        return false;
      continue;
    }
    if (record.lane[0] == kFormProgramEvidence) {
      ++evidence_records;
      if (evidence_records > 1u ||
          !version_space_evidence_record_valid(record) ||
          record.lane[2] + 1u != header.lane[2])
        return false;
      for (std::uint32_t index = 0u; index < record.lane[2]; ++index) {
        std::uint32_t word = 0u;
        if (!trajectory_word_at(state, owner, index, &word) ||
            word != record.lane[4u + index])
          return false;
      }
      std::uint32_t consequence = 0u;
      if (!trajectory_word_at(state, owner, header.lane[2] - 1u,
                              &consequence) ||
          consequence != record.lane[3])
        return false;
      continue;
    }
    if (record.lane[0] == kFormTrajectoryPage) {
      if (record.lane[2] == 0u || record.lane[2] >= pages) return false;
      continue;
    }
    if (record.lane[0] != kFormTrajectoryTerm) return false;
    if (record.lane[3] != 0u || record.lane[6] != 0u ||
        record.lane[7] != 0u || record.reserved[0] != 0u ||
        record.reserved[1] != 0u)
      return false;
  }
  if (current_headers != 1u || owner_records == 0u) return false;

  for (std::uint32_t page = 0u; page < pages; ++page) {
    std::uint32_t term_owner = kInvalid;
    const std::uint32_t extent =
        literal_page_extent(header.lane[2], page);
    if (!literal_page_owner_at(state, owner, header.lane[2], page,
                               &term_owner) ||
        !exact_literal_page_terms(state, kFormTrajectoryTerm, term_owner,
                                  extent))
      return false;
    if (page == 0u && has_provenance) {
      const std::uint32_t blocks = (extent + 1u) / 2u;
      for (std::uint32_t ordinal = 0u; ordinal < blocks; ++ordinal) {
        std::uint32_t provenance_count = 0u;
        const std::uint32_t expected_valid =
            ordinal + 1u == blocks && (extent & 1u) != 0u ? 1u : 3u;
        for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
          const Record& provenance = state->records[slot];
          if (provenance.matter_q8 != 0u &&
              provenance.lane[0] == kFormTrajectoryProvenance &&
              provenance.lane[1] == owner &&
              provenance.lane[2] == ordinal) {
            ++provenance_count;
            if (provenance.lane[kProvenanceValidityLane] != expected_valid)
              return false;
          }
        }
        if (provenance_count != 1u) return false;
      }
    }
  }

  // The source header carries a rolling contact receipt.  Recompute it from
  // the validated literal chain before deriving the executable identity.
  std::uint32_t rolling = 0u;
  std::uint32_t digest = rewrite_mix(kFormProgram, header.lane[2], 0u);
  for (std::uint32_t index = 0u; index < header.lane[2]; ++index) {
    std::uint32_t word = 0u;
    if (!trajectory_word_at(state, owner, index, &word)) return false;
    rolling = rewrite_mix(rolling, word, index);
    digest = rewrite_mix(digest, word, index);
  }
  if (header.lane[6] != rolling) return false;

  Record& program = state->records[header_slot];
  program.lane[0] = kFormProgram;
  program.lane[2] = header.lane[2];
  program.lane[3] = kProgramMatureSupport;
  program.lane[4] = 0u;
  program.lane[5] = digest;
  program.lane[6] = 0u;
  program.lane[7] = kProgramFlagEnabled | kProgramFlagPureExternalExact;
  ++program.revision;
  for (std::uint32_t page = 0u; page < pages; ++page) {
    std::uint32_t term_owner = kInvalid;
    if (!literal_page_owner_at(state, owner, program.lane[2], page,
                               &term_owner))
      return false;
    for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
      Record& term = state->records[slot];
      if (term.matter_q8 == 0u || term.lane[0] != kFormTrajectoryTerm ||
          term.lane[1] != term_owner)
        continue;
      const std::uint32_t second_word = term.lane[5];
      term.lane[0] = kFormProgramTerm;
      term.lane[3] = 0u;
      term.lane[5] = 0u;
      term.lane[6] = second_word;
      term.lane[7] = 0u;
      ++term.revision;
    }
    for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
      Record& provenance = state->records[slot];
      if (provenance.matter_q8 != 0u &&
          provenance.lane[0] == kFormTrajectoryProvenance &&
          provenance.lane[1] == term_owner)
        clear_record(&provenance);
    }
  }
  ++state->revision;
  ++state->completed_inductions;
  return true;
}

// Exact authority for the in-place literal-chain Program.  This is separate
// from the resident-evidence and VersionSpace predicates: a pure external
// Program is immediately executable, but its causal donor path remains gated
// by independent raw replay in the cross-context layer.
BCC32_LITERAL_OBSERVATION_HD bool pure_external_exact_program_authoritative(
    const ResidentRewriteState* state, std::uint32_t program_slot) {
  if (state == nullptr || program_slot >= live_record_capacity(state)) return false;
  const Record& program = state->records[program_slot];
  if (program.matter_q8 == 0u || program.lane[0] != kFormProgram ||
      program.lane[7] != (kProgramFlagEnabled | kProgramFlagPureExternalExact) ||
      program.lane[1] == 0u || program.lane[1] == kInvalid ||
      program.lane[2] == 0u ||
      program.lane[3] < kProgramMatureSupport || program.lane[4] != 0u ||
      program.lane[6] == kInvalid || program.lane[6] == program.lane[1] ||
      program.reserved[0] != 0u ||
      program.reserved[1] != 0u)
    return false;

  const std::uint32_t owner = program.lane[1];
  const std::uint32_t pages = literal_page_count(program.lane[2]);
  if (pages == 0u || pages > kRecordCapacity) return false;
  std::uint32_t headers = 0u;
  std::uint32_t owner_records = 0u;
  std::uint32_t evidence_records = 0u;
  for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
    const Record& record = state->records[slot];
    if (record.matter_q8 == 0u || record.lane[1] != owner) continue;
    ++owner_records;
    if (record.lane[0] == kFormProgram) {
      ++headers;
      if (slot != program_slot) return false;
    } else if (record.lane[0] == kFormProgramEvidence) {
      ++evidence_records;
      if (evidence_records > 1u ||
          !version_space_evidence_record_valid(record) ||
          record.lane[2] + 1u != program.lane[2])
        return false;
    } else if (record.lane[0] == kFormTransformationWitness) {
      // Source provenance survives trajectory->Program conversion as
      // non-executable resident matter. It contributes no Program identity,
      // support, term, or output authority; only the canonical strict witness
      // shape for this exact owner is tolerated here.
      if (!resident_causal_relation_source_witness::is_witness(record, owner))
        return false;
    } else if (record.lane[0] == kFormTrajectoryPage) {
      if (record.lane[2] == 0u || record.lane[2] >= pages) return false;
    } else if (record.lane[0] != kFormProgramTerm) {
      return false;
    }
  }
  if (headers != 1u || owner_records == 0u) return false;

  for (std::uint32_t page = 0u; page < pages; ++page) {
    std::uint32_t term_owner = kInvalid;
    if (!literal_page_owner_at(state, owner, program.lane[2], page,
                               &term_owner) ||
        !exact_literal_page_terms(
            state, kFormProgramTerm, term_owner,
            literal_page_extent(program.lane[2], page)))
      return false;
  }
  if (evidence_records != 0u) {
    for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
      const Record& evidence = state->records[slot];
      if (evidence.matter_q8 == 0u ||
          evidence.lane[0] != kFormProgramEvidence ||
          evidence.lane[1] != owner)
        continue;
      for (std::uint32_t index = 0u; index < evidence.lane[2]; ++index) {
        std::uint32_t word = 0u;
        std::uint32_t meta = 0u;
        if (!program_term_at(state, owner, index, &word, &meta) || meta != 0u ||
            word != evidence.lane[4u + index])
          return false;
      }
      std::uint32_t consequence = 0u;
      std::uint32_t meta = 0u;
      if (!program_term_at(state, owner, program.lane[2] - 1u,
                           &consequence, &meta) ||
          meta != 0u || consequence != evidence.lane[3])
        return false;
    }
  }
  return fixed_program_identity(state, program) == program.lane[5];
}

BCC32_LITERAL_OBSERVATION_HD bool induce_program(
    ResidentRewriteState* state, std::uint32_t left_slot,
    std::uint32_t right_slot, std::uint32_t* identity = nullptr,
    bool commit = true, bool resident_literal_prevalidated = false) {
  Record& left = state->records[left_slot];
  Record& right = state->records[right_slot];
  if (!literal_observation_shape(state, left_slot,
                                 resident_literal_prevalidated) ||
      !literal_observation_shape(state, right_slot,
                                 resident_literal_prevalidated))
    return false;
  if (left.lane[2] == 0u || left.lane[2] != right.lane[2])
    return false;

  std::uint32_t variable_left[kMaximumProgramVariables]{};
  std::uint32_t variable_right[kMaximumProgramVariables]{};
  std::uint32_t variable_count = 0u;
  std::uint32_t digest = 0x811c9dc5u;
  for (std::uint32_t index = 0u; index < left.lane[2]; ++index) {
    std::uint32_t a = 0u;
    std::uint32_t b = 0u;
    if (!literal_observation_word_at_prevalidated(state, left_slot, index,
                                                  &a) ||
        !literal_observation_word_at_prevalidated(state, right_slot, index,
                                                  &b))
      return false;
    if (a != b) {
      if (!same_raw_channel(a, b)) return false;
      std::uint32_t variable = pair_variable(
          variable_left, variable_right, variable_count, a, b);
      if (variable == kInvalid) {
        if (variable_count >= kMaximumProgramVariables) return false;
        variable_left[variable_count] = a;
        variable_right[variable_count] = b;
        variable = variable_count++;
      }
      digest = rewrite_mix(digest, variable + 1u, index);
    } else {
      digest = rewrite_mix(digest, a, index);
    }
  }
  if (variable_count == 0u) return false;

  if (identity != nullptr) *identity = digest;
  if (!commit) return true;

  const std::uint32_t blocks = (left.lane[2] + 1u) / 2u;
  GroundedPairReflectionPlan reflection{};
  const GroundedPairReflectionStatus reflection_status =
      preflight_grounded_pair_reflection(
          state, left_slot, right_slot, digest, blocks + 1u, &reflection);
  if (reflection_status == GroundedPairReflectionStatus::kBlocked)
    return false;
  const std::uint32_t reflection_records =
      reflection_status == GroundedPairReflectionStatus::kReady
          ? reflection.episode_records
          : 0u;
  if (!reserve_grounded_record_matter(
          state, blocks + 1u + reflection_records)) {
    state->fault = 5u;
    return false;
  }
  const std::uint32_t header_slot = allocate_record(state);
  if (header_slot == kInvalid) return false;
  std::uint32_t owner =
      reflection_status == GroundedPairReflectionStatus::kReady
          ? reflection.program_owner
          : digest;
  if (reflection_status != GroundedPairReflectionStatus::kReady &&
      (owner == 0u || owner == kInvalid || record_owner_exists(state, owner)))
    owner = make_record_owner(state, digest);
  if (owner == kInvalid) {
    state->fault = 6u;
    return false;
  }
  Record& program = state->records[header_slot];
  program.lane[0] = kFormProgram;
  program.lane[1] = owner;
  program.lane[2] = left.lane[2];
  program.lane[3] = 2u;
  program.lane[4] = variable_count;
  program.lane[5] = digest;
  program.lane[7] = kProgramFlagEnabled;
  ++program.revision;

  for (std::uint32_t ordinal = 0u; ordinal < blocks; ++ordinal) {
    const std::uint32_t block_slot = allocate_record(state);
    if (block_slot == kInvalid) return false;
    Record& block = state->records[block_slot];
    block.lane[0] = kFormProgramTerm;
    block.lane[1] = owner;
    block.lane[2] = ordinal;
    for (std::uint32_t local = 0u; local < 2u; ++local) {
      const std::uint32_t index = ordinal * 2u + local;
      if (index >= left.lane[2]) break;
      std::uint32_t a = 0u;
      std::uint32_t b = 0u;
      (void)literal_observation_word_at_prevalidated(state, left_slot, index,
                                                     &a);
      (void)literal_observation_word_at_prevalidated(state, right_slot, index,
                                                     &b);
      std::uint32_t meta = 0u;
      std::uint32_t value = a;
      if (a != b) {
        const std::uint32_t variable = pair_variable(
            variable_left, variable_right, variable_count, a, b);
        meta = variable + 1u;
        value = a & kRawChannelMask;
      }
      block.lane[3u + local * 2u] = meta;
      block.lane[4u + local * 2u] = value;
    }
    ++block.revision;
  }

  if (reflection_status == GroundedPairReflectionStatus::kReady)
    commit_grounded_pair_reflection(
        state, left_slot, right_slot, header_slot, reflection);

  retire_literal_observation_source(state, left_slot);
  retire_literal_observation_source(state, right_slot);
  ++state->revision;
  ++state->completed_inductions;
  return true;
}

#include "causal_rewrite_span_induction.inl"
#include "causal_rewrite_span_matching.inl"

BCC32_REWRITE_HD inline std::uint32_t span_match_prefix(
    const ResidentRewriteState* state, const Record& program,
    const Record& trajectory, std::uint32_t* next_term,
    std::uint32_t* next_offset, bool* next_unbound, bool* complete,
    bool* ambiguous,
    std::uint32_t bindings[kMaximumProgramVariables]
                    [kMaximumVariableSpanEvents],
    std::uint32_t lengths[kMaximumProgramVariables]) {
  return span_match_prefix_at(state, program, trajectory, 0u, next_term,
                              next_offset, next_unbound, complete, ambiguous,
                              bindings, lengths);
}

struct ProgramSupportConsensus {
  std::uint32_t identity = kInvalid;
  std::uint32_t maximum_fixed_extent = 0u;
  bool have_candidate = false;
  bool conflict = false;
  bool saw_span = false;
  bool saw_ambiguous = false;
};

BCC32_REWRITE_HD inline void merge_program_support(
    ProgramSupportConsensus* consensus, std::uint32_t identity,
    bool from_span) {
  if (identity == kInvalid) return;
  if (!consensus->have_candidate) {
    consensus->identity = identity;
    consensus->have_candidate = true;
  } else if (consensus->identity != identity) {
    consensus->conflict = true;
  }
  if (from_span) consensus->saw_span = true;
}

BCC32_REWRITE_HD inline bool full_span_program_match(
    const ResidentRewriteState* state, const Record& program,
    const Record& trajectory, bool* ambiguous) {
  bool matched = false;
  for (std::uint32_t start = 0u; start < trajectory.lane[2]; ++start) {
    std::uint32_t bindings[kMaximumProgramVariables]
                          [kMaximumVariableSpanEvents]{};
    std::uint32_t lengths[kMaximumProgramVariables]{};
    std::uint32_t next_term = kInvalid;
    std::uint32_t next_offset = 0u;
    bool unbound = false;
    bool complete = false;
    bool local_ambiguous = false;
    const std::uint32_t match = span_match_prefix_at(
        state, program, trajectory, start, &next_term, &next_offset, &unbound,
        &complete, &local_ambiguous, bindings, lengths);
    if (local_ambiguous || match == kSpanMatchAmbiguous) {
      if (ambiguous != nullptr) *ambiguous = true;
      continue;
    }
    if (match == kSpanMatchPrefix && complete && !unbound) matched = true;
  }
  return matched;
}

BCC32_REWRITE_HD inline void collect_program_support_candidates(
    const ResidentRewriteState* state, const Record& trajectory,
    ProgramSupportConsensus* consensus, bool include_fixed = true,
    bool include_span = true) {
  for (std::uint32_t i = 0u; i < live_record_capacity(state); ++i) {
    const Record& program = state->records[i];
    if (program.matter_q8 == 0u || program.lane[7] == 0u) continue;
    if (program.lane[0] == kFormProgram) {
      if (!include_fixed) continue;
      if ((program.lane[7] & kProgramFlagVersionSpace) != 0u) continue;
      if ((program.lane[7] & kProgramFlagResidentEvidenceOnly) != 0u)
        continue;
      if (full_program_match(state, program, trajectory)) {
        if (program.lane[2] > consensus->maximum_fixed_extent)
          consensus->maximum_fixed_extent = program.lane[2];
        merge_program_support(
            consensus,
            rewrite_mix(kFormProgram, fixed_program_identity(state, program),
                        program.lane[2]),
            false);
      }
    } else if (program.lane[0] == kFormSpanProgram) {
      if (!include_span) continue;
      bool ambiguous = false;
      if (full_span_program_match(state, program, trajectory, &ambiguous))
        merge_program_support(
            consensus,
            rewrite_mix(kFormSpanProgram, program.lane[5], program.lane[2]),
            true);
      consensus->saw_ambiguous |= ambiguous;
    }
  }
}

BCC32_REWRITE_HD inline void support_program_candidates(
    ResidentRewriteState* state, const Record& trajectory,
    const ProgramSupportConsensus& consensus, bool include_fixed = true,
    bool include_span = true) {
  if (!consensus.have_candidate || consensus.conflict) return;
#if !defined(__CUDA_ARCH__)
  bool reflected_construction = false;
  bool matured_construction = false;
#endif
  for (std::uint32_t i = 0u; i < live_record_capacity(state); ++i) {
    Record& program = state->records[i];
    if (program.matter_q8 == 0u || program.lane[7] == 0u ||
        program.lane[6] == trajectory.lane[1])
      continue;
    bool matched = false;
    std::uint32_t identity = kInvalid;
    if (program.lane[0] == kFormProgram) {
      if (!include_fixed) continue;
      if ((program.lane[7] & kProgramFlagVersionSpace) != 0u) continue;
      if ((program.lane[7] & kProgramFlagResidentEvidenceOnly) != 0u)
        continue;
      matched = full_program_match(state, program, trajectory);
      identity = rewrite_mix(kFormProgram,
                             fixed_program_identity(state, program),
                             program.lane[2]);
    } else if (program.lane[0] == kFormSpanProgram) {
      if (!include_span) continue;
      bool ambiguous = false;
      matched = full_span_program_match(state, program, trajectory, &ambiguous);
      identity = rewrite_mix(kFormSpanProgram, program.lane[5], program.lane[2]);
    }
    if (!matched || identity != consensus.identity) continue;
    const bool crosses_fixed_maturity =
        program.lane[0] == kFormProgram &&
        program.lane[3] + 1u == kProgramMatureSupport;
    if (program.lane[3] != 0xffffffffu) ++program.lane[3];
    program.lane[6] = trajectory.lane[1];
    ++program.revision;
    ++state->revision;
    if (crosses_fixed_maturity) {
#if defined(__CUDA_ARCH__)
      schedule_causal_germline_reflection(state, i, trajectory.lane[1]);
#else
      matured_construction = true;
      reflected_construction |=
          reflect_grounded_program_construction(state, i, trajectory);
#endif
    }
  }
#if !defined(__CUDA_ARCH__)
  if (reflected_construction || matured_construction)
    (void)settle_causal_germline_constructor(state);
#endif
}

BCC32_REWRITE_HD inline void clear_span_program(
    ResidentRewriteState* state, std::uint32_t header_slot) {
  if (header_slot == kInvalid) return;
  Record& header = state->records[header_slot];
  if (header.matter_q8 == 0u || header.lane[0] != kFormSpanProgram) return;
  clear_owned_records(state, kFormSpanProgramTerm, header.lane[1]);
  clear_record(&header);
}

#undef BCC32_LITERAL_OBSERVATION_HD
