// AdultState checkpoint responsibility: versioned binary headers, device<->
// host staging, save, and version-dispatched restore. Included from
// bcc32_cuda_adult_v1.cuh at the exact point this code previously lived
// (inside namespace bcc32_cuda_adult_v1, after AdultState and every kernel
// load_checkpoint calls are already defined), so it owns no namespace of its
// own and adds no new checkpoint identity, format, or dispatch path -- this
// is a mechanical extraction, not a behavior change.

struct CheckpointHeaderV4 {
  std::uint32_t magic;
  std::uint32_t version;
  std::uint32_t unit_words;
  std::uint32_t mass_budget;
  std::uint32_t unit_occurrences;
  std::uint32_t unit_count;
  std::uint32_t unit_capacity;
  std::uint32_t bigram_count;
  std::uint32_t trigram_count;
  std::uint32_t online_bigram_count;
  std::uint32_t online_trigram_count;
  std::uint32_t online_association_count;
  std::uint32_t online_episode_count;
  std::uint32_t online_episode_break_count;
  std::uint32_t base_window_count;
  std::uint32_t transitions_lesioned;
  std::uint32_t base_episode_lesioned;
};

struct CheckpointHeaderV8 {
  std::uint32_t magic;
  std::uint32_t version;
  std::uint32_t unit_words;
  std::uint32_t mass_budget;
  std::uint32_t unit_occurrences;
  std::uint32_t unit_count;
  std::uint32_t unit_capacity;
  std::uint32_t bigram_count;
  std::uint32_t trigram_count;
  std::uint32_t online_bigram_count;
  std::uint32_t online_trigram_count;
  std::uint32_t online_association_count;
  std::uint32_t online_episode_count;
  std::uint32_t online_episode_break_count;
  std::uint32_t base_window_count;
  std::uint32_t transitions_lesioned;
  std::uint32_t base_episode_lesioned;
  std::uint32_t novelty_episode_begin;
  std::uint32_t novelty_episode_break_begin;
  std::uint32_t novelty_epoch_active;
};

struct CheckpointHeader {
  std::uint32_t magic;
  std::uint32_t version;
  std::uint32_t unit_words;
  std::uint32_t mass_budget;
  std::uint32_t unit_occurrences;
  std::uint32_t unit_count;
  std::uint32_t unit_capacity;
  std::uint32_t bigram_count;
  std::uint32_t trigram_count;
  std::uint32_t online_bigram_count;
  std::uint32_t online_trigram_count;
  std::uint32_t online_association_count;
  std::uint32_t online_conditioned_transition_count;
  std::uint32_t online_episode_count;
  std::uint32_t online_episode_break_count;
  std::uint32_t base_window_count;
  std::uint32_t transitions_lesioned;
  std::uint32_t base_episode_lesioned;
  std::uint32_t novelty_episode_begin;
  std::uint32_t novelty_episode_break_begin;
  std::uint32_t novelty_epoch_active;
};

template <typename T>
inline void checkpoint_write_device(std::ofstream& output, const T* device,
                                    std::size_t count) {
  std::vector<T> host(count);
  if (count != 0u) {
    cuda_require(cudaMemcpy(host.data(), device, count * sizeof(T), cudaMemcpyDeviceToHost),
                 "stage checkpoint state");
    output.write(reinterpret_cast<const char*>(host.data()), count * sizeof(T));
  }
  if (!output) throw std::runtime_error("checkpoint write failed");
}

template <typename T>
inline void checkpoint_read_device(std::ifstream& input, T* device, std::size_t count) {
  std::vector<T> host(count);
  if (count != 0u) {
    input.read(reinterpret_cast<char*>(host.data()), count * sizeof(T));
    if (!input) throw std::runtime_error("checkpoint is truncated");
    cuda_require(cudaMemcpy(device, host.data(), count * sizeof(T), cudaMemcpyHostToDevice),
                 "restore checkpoint state");
  }
}

inline void save_checkpoint(const AdultState& state, const std::string& path) {
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  if (!output) throw std::runtime_error("cannot create checkpoint: " + path);
  const CheckpointHeader header{0x31434342u, 11u, kUnitWords, kResidentMassBudget,
      state.unit_occurrences, state.unit_count, state.unit_capacity,
      state.bigram_count, state.trigram_count,
      state.online_bigram_count, state.online_trigram_count,
      state.online_association_count, state.online_conditioned_transition_count,
      state.online_episode_count,
      state.online_episode_break_count, state.base_window_count,
      state.transitions_lesioned ? 1u : 0u, state.base_episode_lesioned ? 1u : 0u,
      state.novelty_episode_begin, state.novelty_episode_break_begin,
      state.novelty_epoch_active ? 1u : 0u};
  output.write(reinterpret_cast<const char*>(&header), sizeof(header));
  checkpoint_write_device(output, state.boundary_mask.get(), 256u);
  checkpoint_write_device(output, state.boundary_bytes.get(), kBoundaryCount);
  checkpoint_write_device(output, state.boundary_histogram.get(), 256u);
  checkpoint_write_device(output, state.boundary_pairs.get(), 256u * 256u);
  checkpoint_write_device(output, state.unit_lengths.get(), state.unit_count);
  checkpoint_write_device(output, state.unit_content.get(),
                          static_cast<std::size_t>(state.unit_count) * kUnitWords);
  checkpoint_write_device(output, state.unit_vitality.get(), state.unit_count);
  checkpoint_write_device(output, state.unigram_top_ids.get(), kUnigramTop);
  checkpoint_write_device(output, state.bigrams.get(), state.bigram_count);
  checkpoint_write_device(output, state.bigram_counts.get(), state.bigram_count);
  checkpoint_write_device(output, state.trigrams.get(), state.trigram_count);
  checkpoint_write_device(output, state.trigram_counts.get(), state.trigram_count);
  checkpoint_write_device(output, state.base_episode_units.get(), state.unit_occurrences);
  checkpoint_write_device(output, state.base_posting_positions.get(), state.unit_occurrences);
  checkpoint_write_device(output, state.base_window_signatures.get(), state.base_window_count);
  checkpoint_write_device(output, state.online_bigrams.get(), state.online_bigram_count);
  checkpoint_write_device(output, state.online_bigram_counts.get(), state.online_bigram_count);
  checkpoint_write_device(output, state.online_trigrams.get(), state.online_trigram_count);
  checkpoint_write_device(output, state.online_trigram_counts.get(), state.online_trigram_count);
  checkpoint_write_device(output, state.online_associations.get(), state.online_association_count);
  checkpoint_write_device(output, state.online_association_counts.get(),
                          state.online_association_count);
  checkpoint_write_device(output, state.online_conditioned_transitions.get(),
                          state.online_conditioned_transition_count);
  checkpoint_write_device(output, state.online_conditioned_transition_counts.get(),
                          state.online_conditioned_transition_count);
  checkpoint_write_device(output, state.online_episode_units.get(), state.online_episode_count);
  checkpoint_write_device(output, state.online_episode_breaks.get(),
                          state.online_episode_break_count);
  checkpoint_write_device(output, state.mutable_sizes.get(), 7u);
  checkpoint_write_device(output, state.ledger.get(), 4u);
  checkpoint_write_device(output, state.rng.get(), 1u);
  checkpoint_write_device(output, state.synthesis_policy.get(), 1u);
  checkpoint_write_device(output, state.answer_frame_policy.get(), 1u);
  checkpoint_write_device(output, state.answer_frame_selection.get(), 1u);
  checkpoint_write_device(output, state.efference_state.get(), 1u);
  checkpoint_write_device(output, state.efference_trace.get(), kEfferenceTraceBytes);
  checkpoint_write_device(output, state.interaction_shadow_state.get(), 1u);
  checkpoint_write_device(output, state.interaction_shadow_trace.get(),
                          kInteractionShadowBytes);
  checkpoint_write_device(output, state.conditioned_credit_routes.get(),
                          state.conditioned_credit_routes.size());
  checkpoint_write_device(output, state.conditioned_credit_scalars.get(), 1u);
}

inline AdultState load_checkpoint(const std::string& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) throw std::runtime_error("cannot open checkpoint: " + path);
  std::uint32_t prefix[2] = {};
  input.read(reinterpret_cast<char*>(prefix), sizeof(prefix));
  input.seekg(0, std::ios::beg);
  CheckpointHeader header{};
  if (prefix[1] == 4u) {
    CheckpointHeaderV4 old{};
    input.read(reinterpret_cast<char*>(&old), sizeof(old));
    header = CheckpointHeader{
        old.magic, old.version, old.unit_words, old.mass_budget,
        old.unit_occurrences, old.unit_count, old.unit_capacity,
        old.bigram_count, old.trigram_count, old.online_bigram_count,
        old.online_trigram_count, old.online_association_count,
        0u, old.online_episode_count, old.online_episode_break_count,
        old.base_window_count, old.transitions_lesioned,
        old.base_episode_lesioned, old.online_episode_count,
        old.online_episode_break_count, 0u};
  } else if (prefix[1] <= 8u) {
    CheckpointHeaderV8 old{};
    input.read(reinterpret_cast<char*>(&old), sizeof(old));
    header = CheckpointHeader{
        old.magic, old.version, old.unit_words, old.mass_budget,
        old.unit_occurrences, old.unit_count, old.unit_capacity,
        old.bigram_count, old.trigram_count, old.online_bigram_count,
        old.online_trigram_count, old.online_association_count, 0u,
        old.online_episode_count, old.online_episode_break_count,
        old.base_window_count, old.transitions_lesioned,
        old.base_episode_lesioned, old.novelty_episode_begin,
        old.novelty_episode_break_begin, old.novelty_epoch_active};
  } else {
    input.read(reinterpret_cast<char*>(&header), sizeof(header));
  }
  if (!input || header.magic != 0x31434342u ||
      (header.version != 4u && header.version != 5u && header.version != 6u &&
       header.version != 7u && header.version != 8u && header.version != 9u &&
       header.version != 10u && header.version != 11u) ||
      header.unit_words != kUnitWords || header.mass_budget != kResidentMassBudget) {
    throw std::runtime_error("incompatible bcc32 adult v1 checkpoint");
  }
  if (header.unit_capacity < header.unit_count ||
      header.online_bigram_count > kOnlineNgramCapacity ||
      header.online_trigram_count > kOnlineNgramCapacity ||
      header.online_association_count > kOnlineAssociationCapacity ||
      header.online_conditioned_transition_count >
          kOnlineConditionedTransitionCapacity ||
      header.online_episode_count > kOnlineEpisodeCapacity ||
      header.online_episode_break_count > kOnlineEpisodeBreakCapacity ||
      header.novelty_episode_begin > header.online_episode_count ||
      header.novelty_episode_break_begin > header.online_episode_break_count) {
    throw std::runtime_error("checkpoint extents exceed mutable v1 capacities");
  }

  AdultState state;
  state.unit_occurrences = header.unit_occurrences;
  state.unit_count = header.unit_count;
  state.unit_capacity = header.unit_capacity;
  state.bigram_count = header.bigram_count;
  state.trigram_count = header.trigram_count;
  state.online_bigram_count = header.online_bigram_count;
  state.online_trigram_count = header.online_trigram_count;
  state.online_association_count = header.online_association_count;
  state.online_conditioned_transition_count =
      header.online_conditioned_transition_count;
  state.online_episode_count = header.online_episode_count;
  state.online_episode_break_count = header.online_episode_break_count;
  state.novelty_episode_begin = header.novelty_episode_begin;
  state.novelty_episode_break_begin = header.novelty_episode_break_begin;
  state.novelty_epoch_active = header.novelty_epoch_active != 0u;
  state.base_window_count = header.base_window_count;
  state.transitions_lesioned = header.transitions_lesioned != 0u;
  state.base_episode_lesioned = header.base_episode_lesioned != 0u;
  state.unit_hash_capacity = power_of_two_capacity(state.unit_capacity * 2u);

  state.boundary_mask.allocate(256u);
  state.boundary_bytes.allocate(kBoundaryCount);
  state.closure_bytes.allocate(kClosureCount);
  state.boundary_histogram.allocate(256u);
  state.boundary_pairs.allocate(256u * 256u);
  state.unit_lengths.allocate(state.unit_capacity);
  state.unit_content.allocate(static_cast<std::size_t>(state.unit_capacity) * kUnitWords);
  state.unit_vitality.allocate(state.unit_capacity);
  state.unit_hash_slots.allocate(state.unit_hash_capacity);
  state.unigram_top_ids.allocate(kUnigramTop);
  state.bigrams.allocate(state.bigram_count);
  state.bigram_counts.allocate(state.bigram_count);
  state.trigrams.allocate(state.trigram_count);
  state.trigram_counts.allocate(state.trigram_count);
  state.base_episode_units.allocate(state.unit_occurrences);
  state.base_posting_positions.allocate(state.unit_occurrences);
  state.base_window_signatures.allocate(state.base_window_count);
  state.online_bigrams.allocate(kOnlineNgramCapacity);
  state.online_bigram_counts.allocate(kOnlineNgramCapacity);
  state.online_trigrams.allocate(kOnlineNgramCapacity);
  state.online_trigram_counts.allocate(kOnlineNgramCapacity);
  state.online_associations.allocate(kOnlineAssociationCapacity);
  state.online_association_counts.allocate(kOnlineAssociationCapacity);
  state.online_conditioned_transitions.allocate(
      kOnlineConditionedTransitionCapacity);
  state.online_conditioned_transition_counts.allocate(
      kOnlineConditionedTransitionCapacity);
  state.online_conditioned_transition_conductance.allocate(1u);
  state.online_conditioned_transition_exposure.allocate(1u);
  state.conditioned_credit_routes.allocate(kResidentCreditBankCapacity);
  state.conditioned_credit_scalars.allocate(1u);
  resident_credit::resident_credit_bank_init_kernel<<<
      blocks_for(kResidentCreditBankCapacity), kBlock>>>(
      resident_credit_bank_view(state),
      kResidentCreditBankCapacity * resident_credit::kSupportLimit * 2u);
  cuda_require(cudaGetLastError(),
               "initialize checkpoint resident conditioned credit bank");
  state.online_episode_units.allocate(kOnlineEpisodeCapacity);
  state.online_episode_breaks.allocate(kOnlineEpisodeBreakCapacity);
  state.mutable_sizes.allocate(7u);
  state.motor_context.allocate(kMotorWords);
  state.motor_completion.allocate(kCompositionUnits);
  state.subject_ids.allocate(kSubjectCap);
  state.subject_weights.allocate(kSubjectCap);
  state.subject_count.allocate(1u);
  cuda_require(cudaMemset(state.subject_count.get(), 0, state.subject_count.bytes()),
               "clear resident subject field count");
  state.qonset_count.allocate(state.unit_capacity);
  state.qterm_count.allocate(state.unit_capacity);
  state.qonset_evidence_revision.allocate(state.unit_capacity);
  state.question_gap_field_support.allocate(
      static_cast<std::size_t>(state.unit_capacity) *
      construction::kRelationFieldCount);
  state.question_gap_vote_histogram.allocate(8u);
  state.question_gap_coordinate_histogram.allocate(
      construction::kRelationFieldCount);
  state.question_answer_construction.allocate(
      static_cast<std::size_t>(state.unit_capacity) *
      construction::kRelationFieldCount *
      construction::kQuestionAnswerArityCount);
  state.question_answer_construction_support.allocate(
      static_cast<std::size_t>(state.unit_capacity) *
      construction::kRelationFieldCount *
      construction::kQuestionAnswerArityCount);
  state.question_answer_slot_mapping.allocate(
      static_cast<std::size_t>(state.unit_capacity) *
      construction::kRelationFieldCount *
      construction::kQuestionAnswerArityCount);
  cuda_require(cudaMemset(state.qonset_count.get(), 0, state.qonset_count.bytes()),
               "clear resident qorig onset counts");
  cuda_require(cudaMemset(state.qterm_count.get(), 0, state.qterm_count.bytes()),
               "clear resident qorig terminal counts");
  cuda_require(cudaMemset(state.qonset_evidence_revision.get(), 0,
                          state.qonset_evidence_revision.bytes()),
               "clear resident question-onset evidence revisions");
  cuda_require(cudaMemset(state.question_gap_field_support.get(), 0,
                          state.question_gap_field_support.bytes()),
               "clear resident question-gap field support");
  cuda_require(cudaMemset(state.question_gap_vote_histogram.get(), 0,
                          state.question_gap_vote_histogram.bytes()),
               "clear question-gap vote histogram");
  cuda_require(cudaMemset(state.question_gap_coordinate_histogram.get(), 0,
                          state.question_gap_coordinate_histogram.bytes()),
               "clear question-gap coordinate histogram");
  cuda_require(cudaMemset(state.question_answer_construction.get(), 0xff,
                          state.question_answer_construction.bytes()),
               "clear resident question-answer constructions");
  cuda_require(cudaMemset(state.question_answer_construction_support.get(), 0,
                          state.question_answer_construction_support.bytes()),
               "clear resident question-answer construction support");
  cuda_require(cudaMemset(state.question_answer_slot_mapping.get(), 0,
                          state.question_answer_slot_mapping.bytes()),
               "clear resident question-answer slot mappings");
  state.qorig_onset.allocate(kQOrigInject);
  state.qorig_onset_w.allocate(kQOrigInject);
  state.qorig_onset_n.allocate(1u);
  state.qorig_term.allocate(kQOrigInject);
  state.qorig_term_w.allocate(kQOrigInject);
  state.qorig_term_n.allocate(1u);
  cuda_require(cudaMemset(state.qorig_onset_n.get(), 0, state.qorig_onset_n.bytes()),
               "clear resident qorig onset vocabulary count");
  cuda_require(cudaMemset(state.qorig_term_n.get(), 0, state.qorig_term_n.bytes()),
               "clear resident qorig terminal vocabulary count");
  state.ledger.allocate(4u);
  state.rng.allocate(1u);
  state.efference_trace.allocate(kEfferenceTraceBytes);
  state.efference_state.allocate(1u);
  state.interaction_shadow_trace.allocate(kInteractionShadowBytes);
  state.interaction_shadow_state.allocate(1u);
  state.synthesis_policy.allocate(1u);
  state.answer_frame_policy.allocate(1u);
  state.answer_frame_selection.allocate(1u);
  cuda_require(cudaMemset(state.unit_lengths.get(), 0, state.unit_lengths.bytes()),
               "clear checkpoint unit reserve");
  cuda_require(cudaMemset(state.unit_content.get(), 0, state.unit_content.bytes()),
               "clear checkpoint content reserve");
  cuda_require(cudaMemset(state.unit_vitality.get(), 0, state.unit_vitality.bytes()),
               "clear checkpoint vitality reserve");
  cuda_require(cudaMemset(state.online_bigram_counts.get(), 0,
                          state.online_bigram_counts.bytes()), "clear checkpoint bigram reserve");
  cuda_require(cudaMemset(state.online_trigram_counts.get(), 0,
                          state.online_trigram_counts.bytes()), "clear checkpoint trigram reserve");
  cuda_require(cudaMemset(state.online_association_counts.get(), 0,
                          state.online_association_counts.bytes()),
               "clear checkpoint association reserve");
  cuda_require(
      cudaMemset(state.online_conditioned_transition_counts.get(), 0,
                 state.online_conditioned_transition_counts.bytes()),
      "clear checkpoint conditioned transition reserve");
  cuda_require(cudaMemset(state.online_conditioned_transition_conductance.get(),
                          0,
                          state.online_conditioned_transition_conductance.bytes()),
               "clear checkpoint physical conductance surface");
  cuda_require(cudaMemset(state.online_conditioned_transition_exposure.get(),
                          0,
                          state.online_conditioned_transition_exposure.bytes()),
               "clear checkpoint physical exposure surface");

  checkpoint_read_device(input, state.boundary_mask.get(), 256u);
  checkpoint_read_device(input, state.boundary_bytes.get(), kBoundaryCount);
  checkpoint_read_device(input, state.boundary_histogram.get(), 256u);
  checkpoint_read_device(input, state.boundary_pairs.get(), 256u * 256u);
  discover_closure_kernel<<<1u, 256u>>>(
      state.boundary_histogram.get(), state.boundary_pairs.get(),
      state.boundary_bytes.get(), state.closure_bytes.get());
  checkpoint_read_device(input, state.unit_lengths.get(), state.unit_count);
  checkpoint_read_device(input, state.unit_content.get(),
                         static_cast<std::size_t>(state.unit_count) * kUnitWords);
  checkpoint_read_device(input, state.unit_vitality.get(), state.unit_count);
  cuda_require(cudaMemset(state.unit_hash_slots.get(), 0, state.unit_hash_slots.bytes()),
               "clear checkpoint unit fingerprint index");
  populate_unit_hash_kernel<<<blocks_for(state.unit_count), kBlock>>>(
      state.unit_lengths.get(), state.unit_content.get(), state.unit_count,
      state.unit_hash_slots.get(), state.unit_hash_capacity);
  checkpoint_read_device(input, state.unigram_top_ids.get(), kUnigramTop);
  checkpoint_read_device(input, state.bigrams.get(), state.bigram_count);
  checkpoint_read_device(input, state.bigram_counts.get(), state.bigram_count);
  checkpoint_read_device(input, state.trigrams.get(), state.trigram_count);
  checkpoint_read_device(input, state.trigram_counts.get(), state.trigram_count);
  checkpoint_read_device(input, state.base_episode_units.get(), state.unit_occurrences);
  checkpoint_read_device(input, state.base_posting_positions.get(), state.unit_occurrences);
  checkpoint_read_device(input, state.base_window_signatures.get(), state.base_window_count);
  checkpoint_read_device(input, state.online_bigrams.get(), state.online_bigram_count);
  checkpoint_read_device(input, state.online_bigram_counts.get(), state.online_bigram_count);
  checkpoint_read_device(input, state.online_trigrams.get(), state.online_trigram_count);
  checkpoint_read_device(input, state.online_trigram_counts.get(), state.online_trigram_count);
  checkpoint_read_device(input, state.online_associations.get(), state.online_association_count);
  checkpoint_read_device(input, state.online_association_counts.get(),
                         state.online_association_count);
  if (header.version >= 9u) {
    checkpoint_read_device(input, state.online_conditioned_transitions.get(),
                           state.online_conditioned_transition_count);
    checkpoint_read_device(input, state.online_conditioned_transition_counts.get(),
                           state.online_conditioned_transition_count);
  }
  checkpoint_read_device(input, state.online_episode_units.get(), state.online_episode_count);
  checkpoint_read_device(input, state.online_episode_breaks.get(),
                         state.online_episode_break_count);
  checkpoint_read_device(input, state.mutable_sizes.get(),
                         header.version >= 9u ? 7u : 6u);
  if (header.version < 9u) {
    const std::uint32_t zero = 0u;
    cuda_require(cudaMemcpy(state.mutable_sizes.get() + 6u, &zero, sizeof(zero),
                            cudaMemcpyHostToDevice),
                 "initialize legacy conditioned transition extent");
  }
  checkpoint_read_device(input, state.ledger.get(), 4u);
  checkpoint_read_device(input, state.rng.get(), 1u);
  if (header.version >= 6u) {
    checkpoint_read_device(input, state.synthesis_policy.get(), 1u);
  } else {
    cuda_require(cudaMemset(state.synthesis_policy.get(), 0,
                            state.synthesis_policy.bytes()),
                 "clear legacy checkpoint synthesis policy");
  }
  if (header.version >= 10u) {
    checkpoint_read_device(input, state.answer_frame_policy.get(), 1u);
    checkpoint_read_device(input, state.answer_frame_selection.get(), 1u);
  } else {
    const answer_frame::MutablePolicyState answer_frame_warm_start{};
    const answer_frame::MutableSelectionState answer_frame_empty_selection{};
    cuda_require(cudaMemcpy(state.answer_frame_policy.get(), &answer_frame_warm_start,
                            sizeof(answer_frame_warm_start), cudaMemcpyHostToDevice),
                 "initialize legacy checkpoint answer-frame policy");
    cuda_require(cudaMemcpy(state.answer_frame_selection.get(),
                            &answer_frame_empty_selection,
                            sizeof(answer_frame_empty_selection),
                            cudaMemcpyHostToDevice),
                 "initialize legacy checkpoint answer-frame selection");
  }
  if (header.version >= 7u) {
    checkpoint_read_device(input, state.efference_state.get(), 1u);
    checkpoint_read_device(input, state.efference_trace.get(), kEfferenceTraceBytes);
  } else {
    cuda_require(cudaMemset(state.efference_state.get(), 0,
                            state.efference_state.bytes()),
                 "clear legacy checkpoint efference state");
    cuda_require(cudaMemset(state.efference_trace.get(), 0,
                            state.efference_trace.bytes()),
                 "clear legacy checkpoint efference trace");
  }
  if (header.version >= 8u) {
    checkpoint_read_device(input, state.interaction_shadow_state.get(), 1u);
    checkpoint_read_device(input, state.interaction_shadow_trace.get(),
                           kInteractionShadowBytes);
  } else {
    cuda_require(cudaMemset(state.interaction_shadow_state.get(), 0,
                            state.interaction_shadow_state.bytes()),
                 "clear legacy checkpoint interaction shadow state");
    cuda_require(cudaMemset(state.interaction_shadow_trace.get(), 0,
                            state.interaction_shadow_trace.bytes()),
                 "clear legacy checkpoint interaction shadow trace");
  }
  if (header.version >= 11u) {
    checkpoint_read_device(input, state.conditioned_credit_routes.get(),
                           state.conditioned_credit_routes.size());
    checkpoint_read_device(input, state.conditioned_credit_scalars.get(), 1u);
  }
  cuda_require(cudaMemset(state.motor_context.get(), 0, state.motor_context.bytes()),
               "clear transient checkpoint motor context");
  cuda_require(cudaMemset(state.motor_completion.get(), 0, state.motor_completion.bytes()),
               "clear transient checkpoint completion");
  consolidate_online_ngrams(state);
  publish_resident_credit_conductance(state);
  build_generation_indexes(state);
  build_base_episode_indexes(state, false);
  promote_base_episode_relations(state);

  state.resident_bytes = state.boundary_mask.bytes() + state.boundary_bytes.bytes() +
      state.closure_bytes.bytes() +
      state.boundary_histogram.bytes() + state.boundary_pairs.bytes() +
      state.unit_lengths.bytes() + state.unit_content.bytes() + state.unit_vitality.bytes() +
      state.unit_hash_slots.bytes() +
      state.unigram_top_ids.bytes() + state.bigrams.bytes() + state.bigram_counts.bytes() +
      state.trigrams.bytes() + state.trigram_counts.bytes() +
      state.cached_bigram_contexts.bytes() + state.cached_bigram_entries.bytes() +
      state.cached_trigram_contexts.bytes() + state.cached_trigram_entries.bytes() +
      state.base_episode_units.bytes() + state.base_posting_positions.bytes() +
      state.base_posting_offsets.bytes() + state.base_window_signatures.bytes() +
      state.online_bigrams.bytes() +
      state.online_bigram_counts.bytes() + state.online_trigrams.bytes() +
      state.online_trigram_counts.bytes() + state.online_associations.bytes() +
      state.online_association_counts.bytes() +
      state.online_conditioned_transitions.bytes() +
      state.online_conditioned_transition_counts.bytes() +
      state.online_conditioned_transition_conductance.bytes() +
      state.online_conditioned_transition_exposure.bytes() +
      state.conditioned_credit_routes.bytes() +
      state.conditioned_credit_scalars.bytes() +
      state.online_episode_units.bytes() +
      state.online_episode_breaks.bytes() + state.mutable_sizes.bytes() +
      state.motor_context.bytes() + state.motor_completion.bytes() +
      state.ledger.bytes() + state.rng.bytes() + state.efference_trace.bytes() +
      state.efference_state.bytes() + state.interaction_shadow_trace.bytes() +
      state.interaction_shadow_state.bytes() + state.synthesis_policy.bytes() +
      state.answer_frame_policy.bytes() + state.answer_frame_selection.bytes() +
      state.construction_slot_units.bytes() +
      state.construction_slot_masses.bytes() +
      state.construction_slot_totals.bytes() +
      state.construction_slot_overflow.bytes() +
      state.distributed_motor_mass.bytes() +
      state.distributed_motor_cell_support.bytes() +
      state.distributed_motor_support.bytes() +
      state.distributed_binding_keys.bytes() +
      state.distributed_binding_mass.bytes() +
      state.distributed_binding_support.bytes() +
      state.distributed_enabled.bytes() + state.distributed_history.bytes() +
      state.distributed_previous_active.bytes() +
      state.distributed_sequence_active.bytes() +
      state.distributed_current_active.bytes() +
      state.distributed_cue_active.bytes() +
      state.distributed_completion_scores.bytes() +
      state.distributed_motor_scores.bytes() +
      state.distributed_candidate_cells.bytes() +
      state.distributed_candidate_scores.bytes() +
      state.distributed_path_cells.bytes() + state.distributed_path_phases.bytes() +
      state.distributed_output_tape.bytes() + state.distributed_scalars.bytes() +
      state.distributed_mass_scalars.bytes() + state.relation_roles.bytes() +
      state.relation_role_counts.bytes();
  audit_ledger_kernel<<<1u, kBlock>>>(
      state.unit_vitality.get(), state.unit_count,
      state.bigram_counts.get(), state.bigram_count,
      state.trigram_counts.get(), state.trigram_count,
      state.online_bigram_counts.get(), state.online_bigram_count,
      state.online_trigram_counts.get(), state.online_trigram_count,
      state.online_association_counts.get(), state.online_association_count,
      state.online_conditioned_transition_counts.get(),
      state.online_conditioned_transition_count,
      (state.base_episode_lesioned ? 0u : state.unit_occurrences) +
          state.online_episode_count, state.boundary_histogram.get(),
      state.boundary_pairs.get(), state.ledger.get());
  cuda_require(cudaGetLastError(), "launch checkpoint mass audit");
  cuda_require(cudaDeviceSynchronize(), "complete checkpoint restore");
  return state;
}
