#pragma once

// Included by causal_rewrite_universe.cuh after Program induction helpers and
// inside the causal_rewrite namespace. Variable-sized transaction matter is
// streamed directly into ordinary Records; the transient preflight remains a
// fixed seven-word value.

// A preflight must account for pages that the resident allocator can still
// grow.  Counting only currently materialized empty Records would reject a
// lawful long transaction before allocate_record() gets a chance to extend
// the same bounded population.  The reserve helper is called immediately
// before a writer starts, so no Record is written before capacity is known;
// a device-heap failure after partial page growth may still retain those
// newly materialized empty pages, but never a partially written transaction.
BCC32_REWRITE_HD inline std::uint64_t grounded_record_headroom(
    const ResidentRewriteState* state) {
  if (state == nullptr) return 0u;
  const std::uint64_t free = free_record_count(state);
  const std::uint64_t growable_pages =
      kMaxResidentPages - state->directory.live_page_count;
  return free + growable_pages * kRecordsPerPage;
}

BCC32_REWRITE_HD inline bool reserve_grounded_record_matter(
    ResidentRewriteState* state, std::uint32_t required) {
  if (state == nullptr) return false;
  while (free_record_count(state) < required) {
    if (!grow_resident_pages(state)) return false;
  }
  return true;
}

BCC32_REWRITE_HD inline std::uint32_t grounded_construction_topology_digest(
    const ResidentRewriteState* state, const Record& program) {
  if (state == nullptr ||
      (program.lane[0] != kFormProgram &&
       program.lane[0] != kFormSpanProgram) ||
      program.lane[2] == 0u || program.lane[4] == 0u)
    return kInvalid;
  std::uint32_t digest =
      rewrite_mix(kFormCausalConstructor, program.lane[2], program.lane[4]);
  for (std::uint32_t index = 0u; index < program.lane[2]; ++index) {
    if (program.lane[0] == kFormProgram) {
      std::uint32_t word = 0u;
      std::uint32_t meta = 0u;
      if (!program_term_at(state, program.lane[1], index, &word, &meta))
        return kInvalid;
      digest = rewrite_mix(digest, meta, (word & kRawChannelMask) ^ index);
    } else {
      std::uint32_t kind = 0u;
      std::uint32_t value = 0u;
      std::uint32_t channel = 0u;
      std::uint32_t unused = 0u;
      if (!span_program_term_at(state, program.lane[1], index, &kind,
                                &value, &channel, &unused))
        return kInvalid;
      const std::uint32_t structural_value =
          kind == kSpanTermVariable ? value : 0u;
      digest = rewrite_mix(digest, kind ^ structural_value,
                           (channel & kRawChannelMask) ^ index);
    }
  }
  return digest == 0u || digest == kInvalid
             ? digest ^ 0x4a91c37du
             : digest;
}

BCC32_REWRITE_HD inline std::uint32_t grounded_construction_owner(
    const ResidentRewriteState* state, std::uint32_t salt,
    std::uint32_t excluded_owner);

BCC32_CAUSAL_GERMLINE_DISPATCH GroundedPairReflectionStatus
preflight_grounded_span_reflection(
    const ResidentRewriteState* state, std::uint32_t left_slot,
    std::uint32_t right_slot, std::uint32_t program_digest,
    std::uint32_t program_records, std::uint32_t product_terms,
    GroundedPairReflectionPlan* output) {
  if (output != nullptr) *output = GroundedPairReflectionPlan{};
  if (state == nullptr || output == nullptr || left_slot >= live_record_capacity(state) ||
      right_slot >= live_record_capacity(state) || left_slot == right_slot)
    return GroundedPairReflectionStatus::kNotApplicable;
  const Record& left = state->records[left_slot];
  const Record& right = state->records[right_slot];
  if (!literal_observation_source(state, left_slot) ||
      !literal_observation_source(state, right_slot))
    return GroundedPairReflectionStatus::kNotApplicable;
  if (left.lane[1] == 0u || left.lane[1] == kInvalid ||
      right.lane[1] == 0u || right.lane[1] == kInvalid ||
      left.lane[1] == right.lane[1] || left.lane[2] == 0u ||
      right.lane[2] == 0u || product_terms == 0u ||
      product_terms > kMaximumSpanProgramTerms || program_records == 0u)
    return GroundedPairReflectionStatus::kBlocked;
  std::uint32_t program_owner = program_digest;
  if (program_owner == 0u || program_owner == kInvalid ||
      record_owner_exists(state, program_owner))
    program_owner = make_record_owner(state, program_digest);
  if (program_owner == kInvalid)
    return GroundedPairReflectionStatus::kBlocked;
  std::uint32_t episode_salt = rewrite_mix(
      kFormConstructionEpisode, left.lane[1], right.lane[1]);
  episode_salt = rewrite_mix(episode_salt, program_owner, product_terms);
  const std::uint32_t episode_owner = grounded_construction_owner(
      state, episode_salt, program_owner);
  const std::uint32_t episode_records = product_terms + 1u;
  if (episode_owner == kInvalid ||
      grounded_record_headroom(state) < program_records + episode_records)
    return GroundedPairReflectionStatus::kBlocked;
  output->episode_owner = episode_owner;
  output->program_owner = program_owner;
  output->left_owner = left.lane[1];
  output->right_owner = right.lane[1];
  output->extent = product_terms;
  output->program_records = program_records;
  output->episode_records = episode_records;
  return GroundedPairReflectionStatus::kReady;
}

BCC32_CAUSAL_GERMLINE_DISPATCH void commit_grounded_span_reflection(
    ResidentRewriteState* state, std::uint32_t program_slot,
    const GroundedPairReflectionPlan& plan) {
  const Record& program = state->records[program_slot];
  const std::uint32_t topology =
      grounded_construction_topology_digest(state, program);
  const std::uint32_t header_slot = allocate_record(state);
  Record& header = state->records[header_slot];
  header.lane[0] = kFormConstructionEpisode;
  header.lane[1] = plan.episode_owner;
  header.lane[2] = plan.program_owner;
  header.lane[3] = plan.left_owner;
  header.lane[4] = plan.extent;
  header.lane[5] = program.lane[4];
  header.lane[6] = topology;
  header.lane[7] = kCausalGermlineExternal |
                   kConstructionEpisodeSpanInduction;
  header.reserved[0] = plan.right_owner;
  header.reserved[1] = program.lane[5];
  ++header.revision;
  for (std::uint32_t index = 0u; index < plan.extent; ++index) {
    std::uint32_t kind = 0u;
    std::uint32_t value = 0u;
    std::uint32_t channel = 0u;
    std::uint32_t unused = 0u;
    (void)span_program_term_at(state, plan.program_owner, index, &kind,
                               &value, &channel, &unused);
    const std::uint32_t term_slot = allocate_record(state);
    Record& term = state->records[term_slot];
    term.lane[0] = kFormConstructionEpisodeTerm;
    term.lane[1] = plan.episode_owner;
    term.lane[2] = index;
    term.lane[3] = kind;
    term.lane[4] = kind == kSpanTermVariable ? value : 0u;
    term.lane[5] = channel & kRawChannelMask;
    term.lane[6] = index + 1u == plan.extent ? 1u : 0u;
    term.lane[7] = header.lane[7];
    ++term.revision;
  }
  state->causal_germline_construction_pending = 1u;
}

BCC32_REWRITE_HD inline std::uint32_t grounded_construction_owner(
    const ResidentRewriteState* state, std::uint32_t salt,
    std::uint32_t excluded_owner) {
  std::uint32_t owner = salt;
  for (std::uint32_t attempt = 0u; attempt < kRecordCapacity; ++attempt) {
    if (owner != 0u && owner != kInvalid && owner != excluded_owner &&
        !record_owner_exists(state, owner))
      return owner;
    owner = rewrite_mix(owner, salt ^ 0x6d2b79f5u, attempt + 1u);
  }
  return kInvalid;
}

BCC32_CAUSAL_GERMLINE_DISPATCH GroundedPairReflectionStatus
preflight_grounded_pair_reflection(
    const ResidentRewriteState* state, std::uint32_t left_slot,
    std::uint32_t right_slot, std::uint32_t program_digest,
    std::uint32_t program_records, GroundedPairReflectionPlan* output) {
  if (output != nullptr) *output = GroundedPairReflectionPlan{};
  if (state == nullptr || output == nullptr || left_slot >= live_record_capacity(state) ||
      right_slot >= live_record_capacity(state) || left_slot == right_slot)
    return GroundedPairReflectionStatus::kNotApplicable;
  const Record& left = state->records[left_slot];
  const Record& right = state->records[right_slot];
  if (!literal_observation_source(state, left_slot) ||
      !literal_observation_source(state, right_slot))
    return GroundedPairReflectionStatus::kNotApplicable;
  if (left.lane[1] == 0u || left.lane[1] == kInvalid ||
      right.lane[1] == 0u || right.lane[1] == kInvalid ||
      left.lane[1] == right.lane[1] || left.lane[2] == 0u ||
      left.lane[2] != right.lane[2] || program_records == 0u)
    return GroundedPairReflectionStatus::kBlocked;

  for (std::uint32_t index = 0u; index < left.lane[2]; ++index) {
    std::uint32_t left_word = 0u;
    std::uint32_t right_word = 0u;
    if (!literal_observation_word_at_prevalidated(
            state, left_slot, index, &left_word) ||
        !literal_observation_word_at_prevalidated(
            state, right_slot, index, &right_word) ||
        !same_raw_channel(left_word, right_word))
      return GroundedPairReflectionStatus::kBlocked;
  }

  std::uint32_t program_owner = program_digest;
  if (program_owner == 0u || program_owner == kInvalid ||
      record_owner_exists(state, program_owner))
    program_owner = make_record_owner(state, program_digest);
  if (program_owner == kInvalid)
    return GroundedPairReflectionStatus::kBlocked;
  std::uint32_t episode_salt = rewrite_mix(
      kFormConstructionEpisode, left.lane[1], right.lane[1]);
  episode_salt = rewrite_mix(episode_salt, program_owner, left.lane[2]);
  const std::uint32_t episode_owner = grounded_construction_owner(
      state, episode_salt, program_owner);
  const std::uint32_t episode_records = left.lane[2] + 1u;
  if (episode_owner == kInvalid ||
      grounded_record_headroom(state) < program_records + episode_records)
    return GroundedPairReflectionStatus::kBlocked;

  output->episode_owner = episode_owner;
  output->program_owner = program_owner;
  output->left_owner = left.lane[1];
  output->right_owner = right.lane[1];
  output->extent = left.lane[2];
  output->program_records = program_records;
  output->episode_records = episode_records;
  return GroundedPairReflectionStatus::kReady;
}

BCC32_CAUSAL_GERMLINE_DISPATCH void commit_grounded_pair_reflection(
    ResidentRewriteState* state, std::uint32_t left_slot,
    std::uint32_t right_slot, std::uint32_t program_slot,
    const GroundedPairReflectionPlan& plan) {
  // kReady fixes every source, owner and target slot before the Program writer
  // begins. No fallible predicate is repeated here: a silent post-write decline
  // would leave an unreflected learning transaction in resident history.
  const Record& program = state->records[program_slot];

  const std::uint32_t topology =
      grounded_construction_topology_digest(state, program);
  const std::uint32_t header_slot = allocate_record(state);
  Record& header = state->records[header_slot];
  header.lane[0] = kFormConstructionEpisode;
  header.lane[1] = plan.episode_owner;
  header.lane[2] = plan.program_owner;
  header.lane[3] = plan.left_owner;
  header.lane[4] = plan.extent;
  header.lane[5] = program.lane[4];
  header.lane[6] = topology;
  header.lane[7] = kCausalGermlineExternal |
                   kConstructionEpisodePairInduction;
  header.reserved[0] = plan.right_owner;
  header.reserved[1] = program.lane[5];
  ++header.revision;

  for (std::uint32_t index = 0u; index < plan.extent; ++index) {
    std::uint32_t left_word = 0u;
    std::uint32_t right_word = 0u;
    std::uint32_t product_word = 0u;
    std::uint32_t product_meta = 0u;
    (void)literal_observation_word_at_prevalidated(
        state, left_slot, index, &left_word);
    (void)literal_observation_word_at_prevalidated(
        state, right_slot, index, &right_word);
    (void)program_term_at(state, plan.program_owner, index, &product_word,
                          &product_meta);
    const std::uint32_t term_slot = allocate_record(state);
    Record& term = state->records[term_slot];
    term.lane[0] = kFormConstructionEpisodeTerm;
    term.lane[1] = plan.episode_owner;
    term.lane[2] = index;
    term.lane[3] = left_word;
    term.lane[4] = right_word;
    term.lane[5] = product_word;
    term.lane[6] = product_meta;
    term.lane[7] = kCausalGermlineExternal |
                   kConstructionEpisodePairInduction;
    ++term.revision;
  }
}
