// A missing reader is rederived only from one exact participation population:
// prior distributed terms, the accepted source, its P4 word witnesses, and the
// inquiry/action chain. Close competing populations make the sweep abstain.
BCC32_CAUSAL_GERMLINE_DISPATCH bool regrow_resident_revision_reader(
    ResidentRewriteState* state) {
  if (state == nullptr || state->fault != 0u) return false;
  std::uint32_t chosen = kInvalid;
  std::uint32_t source_slot = kInvalid;
  std::uint32_t variables = 0u;
  for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
    const Record& witness = state->records[slot];
    if (witness.matter_q8 == 0u ||
        witness.lane[0] != kFormRevisionTransferWitness)
      continue;
    std::uint32_t reader = kInvalid;
    std::uint32_t reader_count = 0u;
    for (std::uint32_t candidate = 0u; candidate < live_record_capacity(state);
         ++candidate) {
      const Record& record = state->records[candidate];
      if (record.matter_q8 != 0u &&
          record.lane[0] == kFormRevisionParticipationReader &&
          record.lane[1] == witness.lane[1]) {
        reader = candidate;
        ++reader_count;
      }
    }
    if (reader_count > 1u) return false;
    if (reader_count == 1u &&
        resident_revision_participation_reader_authoritative(state, reader))
      continue;
    std::uint32_t candidate_source = kInvalid;
    std::uint32_t candidate_variables = 0u;
    if (!revision_participation_relation(state, slot, &candidate_source,
                                         &candidate_variables))
      continue;
    if (chosen != kInvalid) return false;
    chosen = slot;
    source_slot = candidate_source;
    variables = candidate_variables;
  }
  if (chosen == kInvalid) return false;
  const Record witness = state->records[chosen];
  const Record& source = state->records[source_slot];
  std::uint32_t reader_slot = kInvalid;
  std::uint32_t reader_count = 0u;
  for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
    const Record& reader = state->records[slot];
    if (reader.matter_q8 != 0u &&
        reader.lane[0] == kFormRevisionParticipationReader &&
        reader.lane[1] == witness.lane[1]) {
      reader_slot = slot;
      ++reader_count;
    }
  }
  if (reader_count > 1u) return false;
  const std::uint32_t blocks = (source.lane[2] + 1u) / 2u;
  std::uint32_t missing_blocks = 0u;
  for (std::uint32_t ordinal = 0u; ordinal < blocks; ++ordinal) {
    std::uint32_t count = 0u;
    for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
      const Record& term = state->records[slot];
      count += term.matter_q8 != 0u &&
               term.lane[0] == kFormRevisionParticipationReaderTerm &&
               term.lane[1] == witness.lane[1] && term.lane[2] == ordinal;
    }
    if (count > 1u) return false;
    missing_blocks += count == 0u;
  }
  if (available_revision_regrowth_records(state) <
      missing_blocks + (reader_count == 0u))
    return false;
  if (reader_count == 0u)
    reader_slot = allocate_revision_regrowth_record(state);
  if (reader_slot == kInvalid) return false;
  for (std::uint32_t ordinal = 0u; ordinal < blocks; ++ordinal) {
    if (find_owned_block(state, kFormRevisionParticipationReaderTerm,
                         witness.lane[1], ordinal) != kInvalid)
      continue;
    const std::uint32_t slot = allocate_revision_regrowth_record(state);
    if (slot == kInvalid) return false;
    Record& term = state->records[slot];
    term.lane[0] = kFormRevisionParticipationReaderTerm;
    term.lane[1] = witness.lane[1];
    term.lane[2] = ordinal;
  }
  Record& reader = state->records[reader_slot];
  reader.lane[0] = kFormRevisionParticipationReader;
  reader.lane[1] = witness.lane[1];
  reader.lane[2] = source.lane[2];
  reader.lane[3] = kProgramMatureSupport;
  reader.lane[4] = variables;
  reader.lane[5] = rewrite_mix(witness.lane[4], witness.lane[5],
                               witness.lane[6]);
  reader.lane[6] = 0u;
  reader.lane[7] = kProgramFlagEnabled | kProgramFlagResidentEvidenceOnly |
                   kProgramFlagRevisionTransferProduct;
  reader.revision = witness.reserved[0] + 1u;
  for (std::uint32_t ordinal = 0u; ordinal < blocks; ++ordinal) {
    Record& term = state->records[find_owned_block(
        state, kFormRevisionParticipationReaderTerm, witness.lane[1],
        ordinal)];
    for (std::uint32_t local = 0u; local < 2u; ++local) {
      const std::uint32_t index = ordinal * 2u + local;
      term.lane[3u + local * 2u] = 0u;
      term.lane[4u + local * 2u] = 0u;
      if (index >= source.lane[2]) continue;
      std::uint32_t word = 0u;
      std::uint32_t meta = 0u;
      bool changed = false;
      for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
        const Record& delta = state->records[slot];
        if (delta.matter_q8 != 0u &&
            delta.lane[0] == kFormRevisionParticipationDelta &&
            delta.lane[1] == witness.lane[1] && delta.lane[2] == index) {
          word = delta.lane[5];
          meta = delta.lane[6];
          changed = true;
        }
        const Record& prior = state->records[slot];
        if (!changed && prior.matter_q8 != 0u &&
            prior.lane[0] == kFormRevisionTransferPriorTerm &&
            prior.lane[1] == witness.lane[1] && prior.lane[2] == index) {
          word = prior.lane[3];
          meta = prior.lane[4];
        }
      }
      term.lane[3u + local * 2u] = meta;
      term.lane[4u + local * 2u] = word;
    }
    ++term.revision;
  }
  ++state->revision;
  refresh_receipt(state);
  return resident_revision_participation_reader_authoritative(state,
                                                               reader_slot);
}

BCC32_CAUSAL_GERMLINE_DISPATCH bool settle_resident_revision_transfer(
    ResidentRewriteState* state) {
  if (state == nullptr || state->fault != 0u) return false;
  if (reacquire_revision_reader_source(state)) return true;
  if (regrow_resident_revision_reader(state)) return true;
  for (std::uint32_t source_slot = 0u; source_slot < live_record_capacity(state);
       ++source_slot) {
    if (!pure_external_exact_program_authoritative(state, source_slot))
      continue;
    const Record& source = state->records[source_slot];
    bool already_used = false;
    for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
      const Record& witness = state->records[slot];
      already_used |=
          witness.matter_q8 != 0u &&
          ((witness.lane[0] == kFormRevisionTransferWitness &&
            witness.lane[3] == source.lane[1]) ||
           (witness.lane[0] == kFormRevisionTransferSourceUse &&
            witness.lane[2] == source.lane[1]));
    }
    if (already_used) continue;

    std::uint32_t donor_slot = kInvalid;
    std::uint32_t mismatch = kInvalid;
    for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
      std::uint32_t candidate_mismatch = kInvalid;
      if (!revision_source_relation(state, slot, source_slot,
                                    &candidate_mismatch))
        continue;
      if (donor_slot != kInvalid) {
        donor_slot = kInvalid;
        break;
      }
      donor_slot = slot;
      mismatch = candidate_mismatch;
    }
    if (donor_slot == kInvalid) continue;

    const Record donor = state->records[donor_slot];
    const std::uint32_t blocks = (source.lane[2] + 1u) / 2u;
    const std::uint32_t donor_blocks = (donor.lane[2] + 1u) / 2u;
    std::uint32_t p4_owner = kInvalid;
    std::uint32_t trace_owner = kInvalid;
    if (!revision_source_return_lineage(state, source_slot, &p4_owner,
                                        &trace_owner))
      return false;
    std::uint32_t delta_count = 0u;
    for (std::uint32_t index = 0u; index < source.lane[2]; ++index) {
      std::uint32_t source_word = 0u;
      std::uint32_t source_meta = 0u;
      if (!program_term_at(state, source.lane[1], index, &source_word,
                           &source_meta) || source_meta != 0u)
        return false;
      std::uint32_t before_word = kInvalid;
      std::uint32_t before_meta = kInvalid;
      if (index < donor.lane[2]) {
        if (!program_term_at(state, donor.lane[1], index, &before_word,
                             &before_meta))
          return false;
      }
      delta_count += index >= donor.lane[2] || before_word != source_word;
    }
    if (free_record_count(state) < donor.lane[2] +
                                      (blocks - donor_blocks) + 2u +
                                      delta_count)
      return false;
    for (std::uint32_t index = 0u; index < donor.lane[2]; ++index) {
      std::uint32_t word = 0u;
      std::uint32_t meta = 0u;
      (void)program_term_at(state, donor.lane[1], index, &word, &meta);
      const std::uint32_t prior_slot = allocate_record(state);
      if (prior_slot == kInvalid) return false;
      Record& prior = state->records[prior_slot];
      prior.lane[0] = kFormRevisionTransferPriorTerm;
      prior.lane[1] = donor.lane[1];
      prior.lane[2] = index;
      prior.lane[3] = word;
      prior.lane[4] = meta;
      prior.lane[5] = donor.revision;
      prior.lane[6] = donor.lane[2];
      prior.lane[7] = kCausalGermlineEnabled;
      ++prior.revision;
    }
    for (std::uint32_t index = 0u; index < source.lane[2]; ++index) {
      std::uint32_t source_word = 0u;
      std::uint32_t source_meta = 0u;
      std::uint32_t before_word = kInvalid;
      std::uint32_t before_meta = kInvalid;
      (void)program_term_at(state, source.lane[1], index, &source_word,
                            &source_meta);
      if (index < donor.lane[2])
        (void)program_term_at(state, donor.lane[1], index, &before_word,
                              &before_meta);
      if (index < donor.lane[2] && before_word == source_word) continue;
      const std::uint32_t delta_slot = allocate_record(state);
      if (delta_slot == kInvalid) return false;
      Record& delta = state->records[delta_slot];
      delta.lane[0] = kFormRevisionParticipationDelta;
      delta.lane[1] = donor.lane[1];
      delta.lane[2] = index;
      delta.lane[3] = before_word;
      delta.lane[4] = before_meta;
      delta.lane[5] = source_word;
      delta.lane[6] = index < donor.lane[2] ? before_meta : 0u;
      delta.lane[7] = index < donor.lane[2] ? 1u : 2u;
      delta.reserved[0] = p4_owner;
      delta.reserved[1] = trace_owner;
      ++delta.revision;
    }
    for (std::uint32_t ordinal = donor_blocks; ordinal < blocks; ++ordinal) {
      const std::uint32_t term_slot = allocate_record(state);
      if (term_slot == kInvalid) return false;
      Record& term = state->records[term_slot];
      term.lane[0] = kFormRevisionParticipationReaderTerm;
      term.lane[1] = donor.lane[1];
      term.lane[2] = ordinal;
      ++term.revision;
    }
    const std::uint32_t witness_slot = allocate_record(state);
    const std::uint32_t source_use_slot = allocate_record(state);
    if (witness_slot == kInvalid || source_use_slot == kInvalid) return false;

    Record& product = state->records[donor_slot];
    product.lane[0] = kFormRevisionParticipationReader;
    product.lane[2] = source.lane[2];
    product.lane[3] = kProgramMatureSupport;
    product.lane[5] = rewrite_mix(donor.lane[5], source.lane[5], mismatch);
    product.lane[6] = 0u;
    product.lane[7] = kProgramFlagEnabled | kProgramFlagResidentEvidenceOnly |
                      kProgramFlagRevisionTransferProduct;
    ++product.revision;
    for (std::uint32_t ordinal = 0u; ordinal < blocks; ++ordinal) {
      const std::uint32_t term_slot = ordinal < donor_blocks
          ? find_owned_block(state, kFormProgramTerm, donor.lane[1], ordinal)
          : find_owned_block(state, kFormRevisionParticipationReaderTerm,
                             donor.lane[1], ordinal);
      if (term_slot == kInvalid) return false;
      Record& term = state->records[term_slot];
      term.lane[0] = kFormRevisionParticipationReaderTerm;
      for (std::uint32_t local = 0u; local < 2u; ++local) {
        const std::uint32_t index = ordinal * 2u + local;
        term.lane[3u + local * 2u] = 0u;
        term.lane[4u + local * 2u] = 0u;
        if (index >= source.lane[2]) continue;
        std::uint32_t word = 0u;
        std::uint32_t meta = 0u;
        bool changed = false;
        for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
          const Record& delta = state->records[slot];
          if (delta.matter_q8 != 0u &&
              delta.lane[0] == kFormRevisionParticipationDelta &&
              delta.lane[1] == donor.lane[1] && delta.lane[2] == index) {
            word = delta.lane[5];
            meta = delta.lane[6];
            changed = true;
          }
          const Record& prior = state->records[slot];
          if (!changed && prior.matter_q8 != 0u &&
              prior.lane[0] == kFormRevisionTransferPriorTerm &&
              prior.lane[1] == donor.lane[1] && prior.lane[2] == index) {
            word = prior.lane[3];
            meta = prior.lane[4];
          }
        }
        term.lane[3u + local * 2u] = meta;
        term.lane[4u + local * 2u] = word;
      }
      ++term.revision;
    }

    Record& witness = state->records[witness_slot];
    witness.lane[0] = kFormRevisionTransferWitness;
    witness.lane[1] = donor.lane[1];
    witness.lane[2] = donor.lane[1];
    witness.lane[3] = source.lane[1];
    witness.lane[4] = donor.lane[5];
    witness.lane[5] = source.lane[5];
    witness.lane[6] = mismatch;
    witness.lane[7] = kCausalGermlineExternal;
    witness.reserved[0] = donor.revision;
    witness.reserved[1] = donor.lane[2];
    ++witness.revision;

    // Source use is separate from the proof witness so a focal lesion cannot
    // replay the same historical correction transaction immediately.  New
    // construction requires a fresh independently owned exact correction.
    Record& source_use = state->records[source_use_slot];
    source_use.lane[0] = kFormRevisionTransferSourceUse;
    source_use.lane[1] = donor.lane[1];
    source_use.lane[2] = source.lane[1];
    source_use.lane[3] = source.lane[5];
    source_use.lane[7] = kCausalGermlineExternal;
    ++source_use.revision;
    ++state->revision;
    refresh_receipt(state);
    return resident_revision_participation_reader_authoritative(state,
                                                                 donor_slot);
  }
  return false;
}
