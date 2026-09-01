// Adult raw-contact training and resident bootstrap pipeline.
//
// Included from bcc32_cuda_adult_v1.cuh inside its namespace and state-only
// guard as one dependency-ordered training responsibility.
inline AdultState train_raw_bytes(const std::uint8_t* host_bytes, std::uint32_t byte_count,
                                  bool shuffle_control, std::uint32_t seed,
                                  bool enable_distributed_motor = false,
                                  bool enable_resident_language = true) {
  if (byte_count < 3u) throw std::runtime_error("corpus must contain at least three raw bytes");
  if (byte_count > (std::numeric_limits<std::uint32_t>::max() - 3u) / 3u) {
    throw std::runtime_error("corpus is too large for the exact uint32 mass ledger");
  }

  AdultState state;
  state.distributed_motor_enabled = enable_distributed_motor;
  if (state.distributed_motor_enabled) allocate_distributed_motor(state);
  state.boundary_mask.allocate(256u);
  state.boundary_bytes.allocate(kBoundaryCount);
  state.closure_bytes.allocate(kClosureCount);
  state.boundary_histogram.allocate(256u);
  state.boundary_pairs.allocate(256u * 256u);
  state.ledger.allocate(4u);
  state.rng.allocate(1u);
  state.efference_trace.allocate(kEfferenceTraceBytes);
  state.efference_state.allocate(1u);
  state.interaction_shadow_trace.allocate(kInteractionShadowBytes);
  state.interaction_shadow_state.allocate(1u);
  state.synthesis_policy.allocate(1u);
  state.answer_frame_policy.allocate(1u);
  state.answer_frame_selection.allocate(1u);
  if (enable_resident_language)
    allocate_resident_proposition_tissue(state);
  cuda_require(cudaMemset(state.efference_trace.get(), 0,
                          state.efference_trace.bytes()),
               "clear resident motor efference trace");
  cuda_require(cudaMemset(state.efference_state.get(), 0,
                          state.efference_state.bytes()),
               "clear resident motor efference state");
  cuda_require(cudaMemset(state.interaction_shadow_trace.get(), 0,
                          state.interaction_shadow_trace.bytes()),
               "clear resident interaction shadow trace");
  cuda_require(cudaMemset(state.interaction_shadow_state.get(), 0,
                          state.interaction_shadow_state.bytes()),
               "clear resident interaction shadow state");
  cuda_require(cudaMemset(state.synthesis_policy.get(), 0,
                          state.synthesis_policy.bytes()),
               "clear resident synthesis policy");
  const answer_frame::MutablePolicyState answer_frame_warm_start{};
  const answer_frame::MutableSelectionState answer_frame_empty_selection{};
  cuda_require(cudaMemcpy(state.answer_frame_policy.get(), &answer_frame_warm_start,
                          sizeof(answer_frame_warm_start), cudaMemcpyHostToDevice),
               "initialize resident answer-frame policy");
  cuda_require(cudaMemcpy(state.answer_frame_selection.get(),
                          &answer_frame_empty_selection,
                          sizeof(answer_frame_empty_selection),
                          cudaMemcpyHostToDevice),
               "initialize resident answer-frame selection");

  DeviceArray<std::uint8_t> corpus(byte_count);
  cuda_require(cudaMemcpy(corpus.get(), host_bytes, byte_count, cudaMemcpyHostToDevice),
               "upload raw corpus bytes");
  if (shuffle_control) {
    DeviceArray<std::uint32_t> shuffle_keys(byte_count);
    fill_shuffle_keys_kernel<<<blocks_for(byte_count), kBlock>>>(byte_count, seed,
                                                                 shuffle_keys.get());
    thrust::stable_sort_by_key(thrust::device, shuffle_keys.get(),
                               shuffle_keys.get() + byte_count, corpus.get());
  }

  normalize_surface_whitespace_kernel<<<blocks_for(byte_count), kBlock>>>(
      corpus.get(), byte_count);
  cuda_require(cudaMemset(state.boundary_histogram.get(), 0,
                          state.boundary_histogram.bytes()), "clear byte histogram");
  cuda_require(cudaMemset(state.boundary_pairs.get(), 0,
                          state.boundary_pairs.bytes()), "clear byte pair histogram");
  byte_statistics_kernel<<<std::min(4096u, blocks_for(byte_count)), kBlock>>>(
      corpus.get(), byte_count, state.boundary_histogram.get(), state.boundary_pairs.get());
  discover_boundary_kernel<<<1u, 256u>>>(state.boundary_histogram.get(),
                                         state.boundary_pairs.get(),
                                         state.boundary_mask.get(), state.boundary_bytes.get());
  discover_closure_kernel<<<1u, 256u>>>(
      state.boundary_histogram.get(), state.boundary_pairs.get(),
      state.boundary_bytes.get(), state.closure_bytes.get());

  DeviceArray<std::uint32_t> flags(byte_count);
  DeviceArray<std::uint32_t> anchors(byte_count);
  DeviceArray<std::uint32_t> scanned_ids(byte_count);
  mark_base_boundaries_kernel<<<blocks_for(byte_count), kBlock>>>(
      corpus.get(), byte_count, state.boundary_mask.get(), flags.get(), anchors.get());
  thrust::inclusive_scan(thrust::device, anchors.get(), anchors.get() + byte_count,
                         anchors.get(), thrust::maximum<std::uint32_t>());
  mark_bounded_units_kernel<<<blocks_for(byte_count), kBlock>>>(byte_count, anchors.get(),
                                                                flags.get());
  thrust::inclusive_scan(thrust::device, flags.get(), flags.get() + byte_count,
                         scanned_ids.get());
  cuda_require(cudaMemcpy(&state.unit_occurrences, scanned_ids.get() + byte_count - 1u,
                          sizeof(std::uint32_t), cudaMemcpyDeviceToHost),
               "read unit allocation extent");

  DeviceArray<std::uint32_t> starts(state.unit_occurrences);
  scatter_unit_starts_kernel<<<blocks_for(byte_count), kBlock>>>(
      byte_count, flags.get(), scanned_ids.get(), starts.get());
  flags.reset();
  anchors.reset();
  scanned_ids.reset();

  DeviceArray<UnitKey> sorted_keys(state.unit_occurrences);
  DeviceArray<std::uint32_t> sorted_occurrences(state.unit_occurrences);
  hash_units_kernel<<<blocks_for(state.unit_occurrences), kBlock>>>(
      corpus.get(), byte_count, starts.get(), state.unit_occurrences,
      sorted_keys.get(), sorted_occurrences.get());
  thrust::stable_sort_by_key(thrust::device, sorted_keys.get(),
                             sorted_keys.get() + state.unit_occurrences,
                             sorted_occurrences.get());

  DeviceArray<std::uint32_t> unique_flags(state.unit_occurrences);
  DeviceArray<std::uint32_t> group_ids(state.unit_occurrences);
  mark_unique_units_kernel<<<blocks_for(state.unit_occurrences), kBlock>>>(
      corpus.get(), byte_count, starts.get(), state.unit_occurrences,
      sorted_keys.get(), sorted_occurrences.get(), unique_flags.get());
  thrust::inclusive_scan(thrust::device, unique_flags.get(),
                         unique_flags.get() + state.unit_occurrences, group_ids.get());
  cuda_require(cudaMemcpy(&state.unit_count, group_ids.get() + state.unit_occurrences - 1u,
                          sizeof(std::uint32_t), cudaMemcpyDeviceToHost),
               "read dictionary allocation extent");

  state.unit_capacity = state.unit_count + kOnlineUnitReserve;
  state.surface_content_begin = state.unit_count;
  // Proposition settlement and learned construction both consume this
  // population. It is resident language material, not an optional motor-only
  // auxiliary allocated by an environment escape hatch.
  if (state.distributed_motor_enabled || enable_resident_language)
    allocate_surface_organ(state);
  if (enable_resident_language)
    allocate_construction_organ(state);
  state.unit_lengths.allocate(state.unit_capacity);
  state.unit_content.allocate(static_cast<std::size_t>(state.unit_capacity) * kUnitWords);
  state.unit_vitality.allocate(state.unit_capacity);
  state.unigram_top_ids.allocate(kUnigramTop);
  cuda_require(cudaMemset(state.unit_lengths.get(), 0, state.unit_lengths.bytes()),
               "clear mutable unit lengths");
  cuda_require(cudaMemset(state.unit_content.get(), 0, state.unit_content.bytes()),
               "clear mutable unit content");
  cuda_require(cudaMemset(state.unit_vitality.get(), 0, state.unit_vitality.bytes()),
               "clear unit vitality");
  DeviceArray<std::uint32_t> occurrence_units(state.unit_occurrences);
  materialize_dictionary_kernel<<<blocks_for(state.unit_occurrences), kBlock>>>(
      corpus.get(), byte_count, starts.get(), state.unit_occurrences,
      sorted_occurrences.get(), unique_flags.get(), group_ids.get(), occurrence_units.get(),
      state.unit_lengths.get(), state.unit_content.get(), state.unit_vitality.get());
  state.unit_hash_capacity = power_of_two_capacity(state.unit_capacity * 2u);
  state.unit_hash_slots.allocate(state.unit_hash_capacity);
  cuda_require(cudaMemset(state.unit_hash_slots.get(), 0, state.unit_hash_slots.bytes()),
               "clear resident unit fingerprint index");
  populate_unit_hash_kernel<<<blocks_for(state.unit_count), kBlock>>>(
      state.unit_lengths.get(), state.unit_content.get(), state.unit_count,
      state.unit_hash_slots.get(), state.unit_hash_capacity);
  build_unigram_top_kernel<<<1u, kBlock>>>(state.unit_vitality.get(), state.unit_count,
                                      state.unigram_top_ids.get());

  const std::uint32_t bigram_occurrences = state.unit_occurrences - 1u;
  const std::uint32_t trigram_occurrences = state.unit_occurrences - 2u;
  DeviceArray<BigramKey> bigram_input(bigram_occurrences);
  DeviceArray<TrigramKey> trigram_input(trigram_occurrences);
  build_ngram_keys_kernel<<<blocks_for(state.unit_occurrences), kBlock>>>(
      occurrence_units.get(), state.unit_occurrences, bigram_input.get(), trigram_input.get());

  thrust::sort(thrust::device, bigram_input.get(), bigram_input.get() + bigram_occurrences);
  DeviceArray<BigramKey> bigram_unique(bigram_occurrences);
  DeviceArray<std::uint32_t> bigram_reduced_counts(bigram_occurrences);
  auto bigram_end = thrust::reduce_by_key(
      thrust::device, bigram_input.get(), bigram_input.get() + bigram_occurrences,
      thrust::make_constant_iterator<std::uint32_t>(1u), bigram_unique.get(),
      bigram_reduced_counts.get());
  state.bigram_count = static_cast<std::uint32_t>(bigram_end.first - bigram_unique.get());
  state.bigrams = compact_copy(bigram_unique, state.bigram_count);
  state.bigram_counts = compact_copy(bigram_reduced_counts, state.bigram_count);

  thrust::sort(thrust::device, trigram_input.get(), trigram_input.get() + trigram_occurrences);
  DeviceArray<TrigramKey> trigram_unique(trigram_occurrences);
  DeviceArray<std::uint32_t> trigram_reduced_counts(trigram_occurrences);
  auto trigram_end = thrust::reduce_by_key(
      thrust::device, trigram_input.get(), trigram_input.get() + trigram_occurrences,
      thrust::make_constant_iterator<std::uint32_t>(1u), trigram_unique.get(),
      trigram_reduced_counts.get());
  state.trigram_count = static_cast<std::uint32_t>(trigram_end.first - trigram_unique.get());
  state.trigrams = compact_copy(trigram_unique, state.trigram_count);
  state.trigram_counts = compact_copy(trigram_reduced_counts, state.trigram_count);
  build_generation_indexes(state);
  state.base_episode_units = std::move(occurrence_units);
  build_base_episode_indexes(state, true);

  if (state.distributed_motor_enabled) {
    const distributed_motor::DeviceStateView view = distributed_motor_view(state);
    cuda_require(distributed_motor::initialize_async(
                     view, kDistributedMotorMassBudget),
                 "initialize distributed raw-event sequence motor");
    learn_distributed_motor_bytes(state, corpus.get(), byte_count);
    cuda_require(cudaDeviceSynchronize(),
                 "complete distributed raw-event sequence learning");
  }

  // The source is no longer needed: learned organization has its own device
  // allocations before this buffer is released.
  corpus.reset();

  state.online_bigrams.allocate(kOnlineNgramCapacity);
  state.online_bigram_counts.allocate(kOnlineNgramCapacity);
  state.online_trigrams.allocate(kOnlineNgramCapacity);
  state.online_trigram_counts.allocate(kOnlineNgramCapacity);
  state.online_associations.allocate(kOnlineAssociationCapacity);
  state.online_association_counts.allocate(kOnlineAssociationCapacity);
  state.online_conditioned_transitions.allocate(kOnlineConditionedTransitionCapacity);
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
  cuda_require(cudaGetLastError(), "initialize resident conditioned credit bank");
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
  cuda_require(cudaMemset(state.online_bigram_counts.get(), 0,
                          state.online_bigram_counts.bytes()), "clear online bigrams");
  cuda_require(cudaMemset(state.online_trigram_counts.get(), 0,
                          state.online_trigram_counts.bytes()), "clear online trigrams");
  cuda_require(cudaMemset(state.online_association_counts.get(), 0,
                          state.online_association_counts.bytes()), "clear online associations");
  cuda_require(
      cudaMemset(state.online_conditioned_transition_counts.get(), 0,
                 state.online_conditioned_transition_counts.bytes()),
      "clear online conditioned transitions");
  cuda_require(cudaMemset(state.online_conditioned_transition_conductance.get(),
                          0,
                          state.online_conditioned_transition_conductance.bytes()),
               "clear physical conditioned conductance surface");
  cuda_require(cudaMemset(state.online_conditioned_transition_exposure.get(),
                          0,
                          state.online_conditioned_transition_exposure.bytes()),
               "clear physical conditioned exposure surface");
  const std::uint32_t mutable_sizes[7] = {
      state.unit_count, 0u, 0u, 0u, 0u, 0u, 0u};
  cuda_require(cudaMemcpy(state.mutable_sizes.get(), mutable_sizes, sizeof(mutable_sizes),
                          cudaMemcpyHostToDevice), "initialize mutable state extents");
  cuda_require(cudaMemset(state.motor_context.get(), 0, state.motor_context.bytes()),
               "clear motor context");
  cuda_require(cudaMemset(state.motor_completion.get(), 0, state.motor_completion.bytes()),
               "clear motor completion");

  const unsigned long long occupied_at_birth =
      static_cast<unsigned long long>(state.unit_occurrences) + bigram_occurrences +
      trigram_occurrences + state.unit_occurrences +
      static_cast<unsigned long long>(byte_count) * 2u - 1u;
  if (occupied_at_birth > kResidentMassBudget) {
    throw std::runtime_error("initial learned mass exceeds fixed resident budget");
  }
  initialize_ledger_kernel<<<1u, 32u>>>(kResidentMassBudget, seed,
                                       state.ledger.get(), state.rng.get());
  DeviceArray<std::uint32_t> birth_status(1u);
  reserve_fixed_mass_kernel<<<1u, 32u>>>(state.unit_occurrences, state.ledger.get(),
                                       birth_status.get());
  debit_counts_kernel<<<blocks_for(state.unit_count), kBlock>>>(
      state.unit_vitality.get(), state.unit_count, state.ledger.get());
  debit_counts_kernel<<<blocks_for(state.bigram_count), kBlock>>>(
      state.bigram_counts.get(), state.bigram_count, state.ledger.get());
  debit_counts_kernel<<<blocks_for(state.trigram_count), kBlock>>>(
      state.trigram_counts.get(), state.trigram_count, state.ledger.get());
  debit_counts_kernel<<<1u, kBlock>>>(state.boundary_histogram.get(), 256u,
                                      state.ledger.get());
  debit_counts_kernel<<<blocks_for(256u * 256u), kBlock>>>(
      state.boundary_pairs.get(), 256u * 256u, state.ledger.get());
  audit_ledger_kernel<<<1u, kBlock>>>(
      state.unit_vitality.get(), state.unit_count,
      state.bigram_counts.get(), state.bigram_count,
      state.trigram_counts.get(), state.trigram_count,
      state.online_bigram_counts.get(), 0u,
      state.online_trigram_counts.get(), 0u,
      state.online_association_counts.get(), 0u,
      state.online_conditioned_transition_counts.get(), 0u,
      state.unit_occurrences,
      state.boundary_histogram.get(), state.boundary_pairs.get(), state.ledger.get());
  cuda_require(cudaGetLastError(), "launch raw-byte adult training kernels");
  cuda_require(cudaDeviceSynchronize(), "complete raw-byte adult training");
  if (state.online_association_count == 0u) promote_base_episode_relations(state);
  audit_ledger_kernel<<<1u, kBlock>>>(
      state.unit_vitality.get(), state.unit_count,
      state.bigram_counts.get(), state.bigram_count,
      state.trigram_counts.get(), state.trigram_count,
      state.online_bigram_counts.get(), state.online_bigram_count,
      state.online_trigram_counts.get(), state.online_trigram_count,
      state.online_association_counts.get(), state.online_association_count,
      state.online_conditioned_transition_counts.get(),
      state.online_conditioned_transition_count,
      state.unit_occurrences, state.boundary_histogram.get(),
      state.boundary_pairs.get(), state.ledger.get());
  cuda_require(cudaGetLastError(), "launch promoted relation mass audit");
  cuda_require(cudaDeviceSynchronize(), "complete promoted relation mass audit");
  if (state.surface_organ_enabled) {
    refresh_surface_organ(state);
    learn_base_resident_proposition_sequence(state);
  }

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
      state.distributed_mass_scalars.bytes() + state.surface_role_projection.bytes() +
      state.surface_roles.bytes() + state.surface_unit_population.bytes() +
      state.surface_population_context_mass.bytes() +
      state.surface_unit_activity.bytes() +
      state.surface_unit_phase.bytes() + state.surface_projection_state.bytes() +
      state.surface_unit_mass.bytes() +
      state.surface_unit_start_mass.bytes() + state.surface_unit_end_mass.bytes() +
      state.surface_role_mass.bytes() + state.surface_role_start_mass.bytes() +
      state.surface_role_end_mass.bytes() + state.surface_role_bigram_mass.bytes() +
      state.surface_role_bigram_context_mass.bytes() +
      state.surface_role_trigram_mass.bytes() +
      state.surface_role_trigram_context_mass.bytes() + state.surface_stats.bytes() +
      state.surface_context_cells.bytes() + state.surface_context_memberships.bytes() +
      state.surface_context_transitions.bytes() + state.surface_context_bindings.bytes() +
      state.surface_context_scalars.bytes() + state.surface_context_primary_ranks.bytes() +
      state.surface_context_alternate_ranks.bytes() +
      state.surface_context_state_counts.bytes() +
      state.surface_bridges.bytes() + state.surface_prefixes.bytes() +
      state.surface_suffixes.bytes() + state.surface_permutation_scores.bytes() +
      state.surface_permutation_valid.bytes() + state.surface_output_units.bytes() +
      state.surface_output_anchor_mask.bytes() + state.surface_output_bytes.bytes() +
      state.surface_result.bytes() + state.relation_roles.bytes() +
      state.relation_role_counts.bytes() + resident_proposition_bytes(state);
  return state;
}
