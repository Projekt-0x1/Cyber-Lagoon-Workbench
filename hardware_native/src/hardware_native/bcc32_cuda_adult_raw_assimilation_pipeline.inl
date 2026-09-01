inline void assimilate_raw_bytes(AdultState& state, const std::uint8_t* host_bytes,
                                 std::uint32_t byte_count, bool shuffle_order = false,
                                 std::uint32_t shuffle_seed = 0x9e3779b9u,
                                 bool proposition_efferent_contact = false,
                                 std::int32_t proposition_efferent_polarity = 0,
                                 bool proposition_outcome_present = true,
                                 bool learn_relation_triples = true,
                                 const std::uint32_t* contact_learned_request =
                                     nullptr,
                                 bool initial_exposure = false,
                                 std::vector<ConditionedCreditEvent>*
                                     conditioned_credit_events = nullptr,
                                 void* conditioned_credit_context = nullptr,
                                 ConditionedDeviceCreditConsumer
                                     conditioned_credit_consumer = nullptr,
                                 ConditionedConductancePublisher
                                     conditioned_conductance_publisher =
                                         nullptr,
                                 ConditionedPredictionWitnessConsumer
                                     conditioned_prediction_consumer =
                                         nullptr,
                                 ConditionedPredictionReceipt*
                                     conditioned_prediction_result =
                                         nullptr,
                                 bool refresh_question_goal = true) {
  if (byte_count == 0u) return;
  if (conditioned_credit_events != nullptr)
    conditioned_credit_events->clear();
  if (conditioned_prediction_result != nullptr)
    *conditioned_prediction_result = {};
  const std::uint32_t previous_unit_count = state.unit_count;
  const std::uint32_t previous_episode_count = state.online_episode_count;
  const std::uint32_t previous_episode_break_count = state.online_episode_break_count;
  DeviceArray<std::uint8_t> device_bytes(byte_count);
  DeviceArray<std::uint32_t> sequence(byte_count);
  DeviceArray<std::uint32_t> status(1u);
  DeviceArray<std::uint32_t> exact_replay(1u);
  DeviceArray<std::uint32_t> flags(byte_count);
  DeviceArray<std::uint32_t> anchors(byte_count);
  DeviceArray<std::uint32_t> scanned_ids(byte_count);
  DeviceArray<std::uint32_t> starts(byte_count);
  cuda_require(cudaMemcpy(device_bytes.get(), host_bytes, byte_count, cudaMemcpyHostToDevice),
               "upload online raw bytes");
  normalize_surface_whitespace_kernel<<<blocks_for(byte_count), kBlock>>>(
      device_bytes.get(), byte_count);
  cuda_require(cudaMemset(exact_replay.get(), 0, exact_replay.bytes()),
               "clear resident replay gate");
  // Repeated external contact is still an event in a continuous organism.
  // Motor reafference uses the separate efference discharge path; content
  // equality never gives the host authority to erase a sensory contact.
  const std::uint32_t replay_allowed = 0u;
  if (state.online_episode_count != 0u) {
    const std::uint32_t candidates = state.online_episode_count;
    detect_exact_resident_bytes_replay_kernel<<<blocks_for(candidates), kBlock>>>(
        device_bytes.get(), byte_count, state.online_episode_units.get(),
        state.online_episode_count, state.online_episode_breaks.get(),
        state.online_episode_break_count, state.unit_lengths.get(),
        state.unit_content.get(), state.unit_count, 1u, replay_allowed,
        exact_replay.get());
  }
  if (!state.base_episode_lesioned && state.unit_occurrences != 0u) {
    const std::uint32_t candidates = state.unit_occurrences;
    detect_exact_resident_bytes_replay_kernel<<<blocks_for(candidates), kBlock>>>(
        device_bytes.get(), byte_count, state.base_episode_units.get(),
        state.unit_occurrences, nullptr, 0u, state.unit_lengths.get(),
        state.unit_content.get(), state.unit_count, 0u, replay_allowed,
        exact_replay.get());
  }
  const std::uint32_t boundary_mass = byte_count * 2u - 1u;
  reserve_fixed_mass_if_novel_kernel<<<1u, 32u>>>(
      boundary_mass, exact_replay.get(), state.ledger.get(), status.get());
  cuda_require(cudaDeviceSynchronize(), "reserve mutable boundary evidence mass");
  std::uint32_t boundary_status = 0u;
  std::uint32_t detected_replay = 0u;
  cuda_require(cudaMemcpy(&boundary_status, status.get(), sizeof(boundary_status),
                          cudaMemcpyDeviceToHost), "read boundary evidence status");
  cuda_require(cudaMemcpy(&detected_replay, exact_replay.get(),
                          sizeof(detected_replay), cudaMemcpyDeviceToHost),
               "read resident replay decision");
  if (boundary_status != 0u) {
    throw std::runtime_error("online boundary evidence exceeds fixed resident mass");
  }
  if (detected_replay != 0u) return;
  learn_distributed_motor_bytes(state, device_bytes.get(), byte_count);
  byte_statistics_if_novel_kernel<<<std::min(4096u, blocks_for(byte_count)), kBlock>>>(
      device_bytes.get(), byte_count, exact_replay.get(),
      state.boundary_histogram.get(), state.boundary_pairs.get());
  discover_boundary_kernel<<<1u, 256u>>>(state.boundary_histogram.get(),
                                         state.boundary_pairs.get(),
                                         state.boundary_mask.get(),
                                         state.boundary_bytes.get());
  discover_closure_kernel<<<1u, 256u>>>(
      state.boundary_histogram.get(), state.boundary_pairs.get(),
      state.boundary_bytes.get(), state.closure_bytes.get());
  mark_base_boundaries_kernel<<<blocks_for(byte_count), kBlock>>>(
      device_bytes.get(), byte_count, state.boundary_mask.get(), flags.get(), anchors.get());
  thrust::inclusive_scan(thrust::device, anchors.get(), anchors.get() + byte_count,
                         anchors.get(), thrust::maximum<std::uint32_t>());
  mark_bounded_units_kernel<<<blocks_for(byte_count), kBlock>>>(byte_count, anchors.get(),
                                                                flags.get());
  thrust::inclusive_scan(thrust::device, flags.get(), flags.get() + byte_count,
                         scanned_ids.get());
  std::uint32_t sequence_count = 0u;
  cuda_require(cudaMemcpy(&sequence_count, scanned_ids.get() + byte_count - 1u,
                          sizeof(sequence_count), cudaMemcpyDeviceToHost),
               "read online unit extent");
  scatter_unit_starts_kernel<<<blocks_for(byte_count), kBlock>>>(
      byte_count, flags.get(), scanned_ids.get(), starts.get());
  DeviceArray<UnitKey> sorted_keys(sequence_count);
  DeviceArray<std::uint32_t> sorted_occurrences(sequence_count);
  hash_units_kernel<<<blocks_for(sequence_count), kBlock>>>(
      device_bytes.get(), byte_count, starts.get(), sequence_count,
      sorted_keys.get(), sorted_occurrences.get());
  thrust::stable_sort_by_key(thrust::device, sorted_keys.get(),
                             sorted_keys.get() + sequence_count,
                             sorted_occurrences.get());
  DeviceArray<std::uint32_t> unique_flags(sequence_count);
  DeviceArray<std::uint32_t> group_ids(sequence_count);
  mark_unique_units_kernel<<<blocks_for(sequence_count), kBlock>>>(
      device_bytes.get(), byte_count, starts.get(), sequence_count,
      sorted_keys.get(), sorted_occurrences.get(), unique_flags.get());
  thrust::inclusive_scan(thrust::device, unique_flags.get(),
                         unique_flags.get() + sequence_count, group_ids.get());
  std::uint32_t unique_count = 0u;
  cuda_require(cudaMemcpy(&unique_count, group_ids.get() + sequence_count - 1u,
                          sizeof(unique_count), cudaMemcpyDeviceToHost),
               "read unique online unit extent");
  DeviceArray<std::uint32_t> representatives(unique_count);
  DeviceArray<std::uint32_t> group_units(unique_count);
  DeviceArray<std::uint32_t> novel_flags(unique_count);
  DeviceArray<std::uint32_t> novel_ids(unique_count);
  cuda_require(cudaMemset(group_units.get(), 0xff, group_units.bytes()),
               "clear deterministic online unit groups");
  cuda_require(cudaMemset(novel_flags.get(), 0, novel_flags.bytes()),
               "clear deterministic online novelty flags");
  scatter_unique_assimilation_representatives_kernel<<<
      blocks_for(sequence_count), kBlock>>>(
      sorted_occurrences.get(), unique_flags.get(), group_ids.get(),
      sequence_count, representatives.get());
  resolve_unique_assimilation_units_kernel<<<blocks_for(unique_count), kBlock>>>(
      device_bytes.get(), byte_count, starts.get(), sequence_count,
      representatives.get(), unique_count,
      state.unit_lengths.get(), state.unit_content.get(),
      state.unit_hash_slots.get(), state.unit_hash_capacity,
      group_units.get(), novel_flags.get(), status.get(), exact_replay.get());
  thrust::inclusive_scan(thrust::device, novel_flags.get(),
                         novel_flags.get() + unique_count, novel_ids.get());
  std::uint32_t novel_count = 0u;
  std::uint32_t materialize_status = 0u;
  cuda_require(cudaMemcpy(&novel_count, novel_ids.get() + unique_count - 1u,
                          sizeof(novel_count), cudaMemcpyDeviceToHost),
               "read deterministic online novelty extent");
  cuda_require(cudaMemcpy(&materialize_status, status.get(),
                          sizeof(materialize_status), cudaMemcpyDeviceToHost),
               "read deterministic online lookup status");
  if (materialize_status != 0u) {
    throw std::runtime_error("resident unit fingerprint index is full");
  }
  if (novel_count > state.unit_capacity - state.unit_count) {
    throw std::runtime_error("online unit reserve exhausted");
  }
  materialize_novel_assimilation_units_kernel<<<blocks_for(unique_count), kBlock>>>(
      device_bytes.get(), byte_count, starts.get(), sequence_count,
      representatives.get(), unique_count, novel_flags.get(), novel_ids.get(),
      state.unit_count, state.unit_lengths.get(), state.unit_content.get(),
      state.unit_vitality.get(), group_units.get(), exact_replay.get());
  if (novel_count != 0u) {
    populate_unit_hash_range_kernel<<<blocks_for(novel_count), kBlock>>>(
        state.unit_lengths.get(), state.unit_content.get(), state.unit_count,
        novel_count, state.unit_hash_slots.get(), state.unit_hash_capacity);
    const std::uint32_t updated_unit_count = state.unit_count + novel_count;
    cuda_require(cudaMemcpy(state.mutable_sizes.get(), &updated_unit_count,
                            sizeof(updated_unit_count), cudaMemcpyHostToDevice),
                 "publish deterministic online unit extent");
    // Later resident passes in this same contact (role derivation,
    // construction learning, and proposition admission) consume the host
    // extent. Publishing only the device mirror leaves those passes blind to
    // the just-materialized raw units and permanently fragments recurrence.
    state.unit_count = updated_unit_count;
  }
  map_assimilation_occurrence_groups_kernel<<<blocks_for(sequence_count), kBlock>>>(
      sorted_occurrences.get(), group_ids.get(), sequence_count,
      group_units.get(), sequence.get(), exact_replay.get());
  if (shuffle_order) {
    auto* sequence_ptr = sequence.get();
    std::uint32_t sequence_count_arg = sequence_count;
    std::uint32_t shuffle_seed_arg = shuffle_seed;
    void* shuffle_arguments[] = {&sequence_ptr, &sequence_count_arg,
                                 &shuffle_seed_arg};
    cuda_require(
        cudaLaunchKernel(
            reinterpret_cast<const void*>(shuffle_assimilation_sequence_kernel),
            dim3{1u, 1u, 1u}, dim3{2u, 1u, 1u}, shuffle_arguments, 0u,
            nullptr),
        "launch shuffle online assimilation sequence");
    cuda_require(cudaGetLastError(), "shuffle online assimilation sequence");
  }
  if (!shuffle_order) {
    auto* sequence_ptr = sequence.get();
    std::uint32_t sequence_count_arg = sequence_count;
    auto* base_bigrams = state.bigrams.get();
    auto* base_bigram_counts = state.bigram_counts.get();
    std::uint32_t base_bigram_count = state.bigram_count;
    auto* base_trigrams = state.trigrams.get();
    auto* base_trigram_counts = state.trigram_counts.get();
    std::uint32_t base_trigram_count = state.trigram_count;
    auto* online_bigrams = state.online_bigrams.get();
    auto* online_bigram_counts = state.online_bigram_counts.get();
    std::uint32_t online_bigram_count = state.online_bigram_count;
    auto* online_trigrams = state.online_trigrams.get();
    auto* online_trigram_counts = state.online_trigram_counts.get();
    std::uint32_t online_trigram_count = state.online_trigram_count;
    auto* exact_replay_ptr = exact_replay.get();
    auto* policy = state.synthesis_policy.get();
    void* policy_arguments[] = {
        &sequence_ptr, &sequence_count_arg, &base_bigrams,
        &base_bigram_counts, &base_bigram_count, &base_trigrams,
        &base_trigram_counts, &base_trigram_count, &online_bigrams,
        &online_bigram_counts, &online_bigram_count, &online_trigrams,
        &online_trigram_counts, &online_trigram_count, &exact_replay_ptr,
        &policy};
    cuda_require(
        cudaLaunchKernel(
            reinterpret_cast<const void*>(
                advance_resident_synthesis_policy_kernel),
            dim3{1u, 1u, 1u}, dim3{2u, 1u, 1u}, policy_arguments, 0u,
            nullptr),
        "launch resident synthesis policy advancement");
    cuda_require(cudaGetLastError(),
                 "launch resident synthesis policy advancement");
  }
  DeviceArray<std::uint32_t> segment_starts(sequence_count);
  DeviceArray<std::uint32_t> segment_ids(sequence_count);
  mark_sequence_segment_starts_kernel<<<blocks_for(sequence_count), kBlock>>>(
      sequence.get(), sequence_count, state.unit_lengths.get(),
      state.unit_content.get(), state.closure_bytes.get(),
      state.boundary_bytes.get(), segment_starts.get());
  thrust::inclusive_scan(thrust::device, segment_starts.get(),
                         segment_starts.get() + sequence_count,
                         segment_ids.get());
  std::uint32_t host_status = 0u;
  std::uint32_t host_exact_replay = 0u;
  std::uint32_t sizes[7] = {};
  DeviceArray<std::uint32_t> segment_extent(1u);
  DeviceArray<std::uint32_t> conditioned_appended(1u);
  DeviceArray<ConditionedPredictionReceipt> conditioned_prediction_receipt(1u);
  cuda_require(cudaMemset(segment_extent.get(), 0, segment_extent.bytes()),
               "clear online segment carry");
  const std::uint32_t overlap =
      max(kAssociationRadius, kConditionedTransitionLagCount);
  for (std::uint32_t processed = 0u; processed < sequence_count;) {
    const std::uint32_t prefix = min(overlap, processed);
    const std::uint32_t fresh = min(kOnlineAssimilationChunkUnits,
                                    sequence_count - processed);
    const std::uint32_t begin = processed - prefix;
    const std::uint32_t chunk_count = prefix + fresh;
    const std::uint32_t final_chunk =
        processed + fresh == sequence_count ? 1u : 0u;
    // The resident stores are sorted and reduced after every accepted chunk.
    // Preserve those exact prefixes so this contact only sorts its newly
    // appended suffix and merges it into resident order.  Re-sorting the full
    // lifetime store on every small contact is semantically unnecessary and
    // makes online experience scale with the organism's age.
    const std::uint32_t bigram_prefix = state.online_bigram_count;
    const std::uint32_t trigram_prefix = state.online_trigram_count;
    const std::uint32_t association_prefix = state.online_association_count;
    assimilate_raw_bytes_kernel<<<1u, 32u>>>(
        device_bytes.get(), byte_count, state.boundary_mask.get(),
        state.closure_bytes.get(), state.unit_lengths.get(),
        state.unit_content.get(), state.unit_vitality.get(), state.unit_capacity,
        state.bigrams.get(), state.bigram_counts.get(), state.bigram_count,
        state.trigrams.get(), state.trigram_counts.get(), state.trigram_count,
        state.online_bigrams.get(), state.online_bigram_counts.get(),
        state.online_trigrams.get(), state.online_trigram_counts.get(),
        state.online_associations.get(), state.online_association_counts.get(),
        state.online_episode_units.get(), state.online_episode_breaks.get(),
        state.mutable_sizes.get(), starts.get() + begin, chunk_count, prefix,
        final_chunk, segment_extent.get(), segment_ids.get() + begin,
        sequence.get() + begin,
        state.ledger.get(), status.get(), exact_replay.get());
    cuda_require(cudaGetLastError(), "launch chunked online raw-byte assimilation");
    cuda_require(cudaDeviceSynchronize(),
                 "complete chunked online raw-byte assimilation");
    cuda_require(cudaMemcpy(&host_status, status.get(), sizeof(host_status),
                            cudaMemcpyDeviceToHost),
                 "read chunked online assimilation status");
    cuda_require(cudaMemcpy(sizes, state.mutable_sizes.get(), sizeof(sizes),
                            cudaMemcpyDeviceToHost),
                 "read chunked mutable state extents");
    if (host_status != 0u) {
      throw std::runtime_error("online assimilation capacity or mass failure " +
                               std::to_string(host_status));
    }
    state.unit_count = sizes[0];
    state.online_bigram_count = sizes[1];
    state.online_trigram_count = sizes[2];
    state.online_association_count = sizes[3];
    state.online_episode_count = sizes[4];
    state.online_episode_break_count = sizes[5];

    const std::uint32_t conditioned_prefix =
        state.online_conditioned_transition_count;
    const std::uint32_t event_count =
        conditioned_transition_event_count(chunk_count, prefix);
    if (event_count > kOnlineConditionedTransitionCapacity -
                          state.online_conditioned_transition_count) {
      throw std::runtime_error("online conditioned transition reserve exhausted");
    }
    // Conditioned transition prediction and its physical credit bridge belong
    // to the resident language/conditioned organ.  A transport/appraisal
    // adult with that organ disabled still assimilates raw bytes, n-grams,
    // episodes, and the ordinary contact ledger, but must not allocate and
    // sweep a 2*event_count credit surface for a zero-capacity organ.  Keep
    // the full temporal prediction path for language-enabled adults.
    if (state.surface_organ_enabled && event_count != 0u) {
      DeviceArray<ConditionedCreditEvent> device_credit_events(
          2u * event_count);
      DeviceArray<ConditionedPredictionWitness> device_prediction_witnesses(
          event_count);
      cuda_require(cudaMemset(device_credit_events.get(), 0,
                              device_credit_events.bytes()),
                   "clear resident conditioned credit events");
      cuda_require(cudaMemset(conditioned_prediction_receipt.get(), 0,
                              conditioned_prediction_receipt.bytes()),
                   "clear conditioned prediction receipt");
      cuda_require(cudaMemset(device_prediction_witnesses.get(), 0,
                              device_prediction_witnesses.bytes()),
                   "clear conditioned prediction witnesses");
      // This pass sees only the consolidated state that existed before this
      // contact.  It cannot use the observed transition to predict itself.
      apply_conditioned_prediction_error_kernel<<<
          std::min(4096u, blocks_for(event_count)), kBlock>>>(
          sequence.get() + begin, chunk_count, prefix,
          segment_ids.get() + begin,
          state.online_conditioned_transitions.get(),
          state.online_conditioned_transition_conductance.get(),
          conditioned_prefix,
          event_count, proposition_efferent_contact ? 1u : 0u,
          proposition_efferent_polarity,
          proposition_outcome_present ? 1u : 0u,
          device_credit_events.get(),
          conditioned_prediction_receipt.get(),
          conditioned_conductance_publisher == nullptr
              ? resident_credit_bank_view(state)
              : resident_credit::BankView{},
          device_prediction_witnesses.get());
      cuda_require(cudaGetLastError(),
                   "apply resident conditioned prediction error");
      auto* credit_events = device_credit_events.get();
      std::uint32_t credit_event_count = 2u * event_count;
      auto credit_bank = resident_credit_bank_view(state);
      void* credit_arguments[] = {&credit_events, &credit_event_count,
                                  &credit_bank};
      cuda_require(
          cudaLaunchKernel(
              reinterpret_cast<const void*>(apply_conditioned_credit_bank_kernel),
              dim3{1u, 1u, 1u}, dim3{2u, 1u, 1u}, credit_arguments, 0u,
              nullptr),
          "launch resident conditioned credit bank");
      cuda_require(cudaGetLastError(),
                   "apply resident conditioned credit bank");
      reserve_fixed_mass_kernel<<<1u, 32u>>>(
          event_count, state.ledger.get(), status.get());
      cuda_require(cudaDeviceSynchronize(),
                   "reserve conditioned transition mass");
      if (conditioned_credit_consumer != nullptr) {
        conditioned_credit_consumer(conditioned_credit_context,
                                    device_credit_events.get(),
                                    2u * event_count);
      }
      if (conditioned_prediction_consumer != nullptr) {
        conditioned_prediction_consumer(
            conditioned_credit_context, device_prediction_witnesses.get(),
            event_count);
      }
      if (conditioned_prediction_result != nullptr) {
        ConditionedPredictionReceipt chunk{};
        cuda_require(cudaMemcpy(&chunk, conditioned_prediction_receipt.get(),
                                sizeof(chunk), cudaMemcpyDeviceToHost),
                     "read conditioned prediction receipt");
        conditioned_prediction_result->observed_events +=
            chunk.observed_events;
        conditioned_prediction_result->predicted_events +=
            chunk.predicted_events;
        conditioned_prediction_result->correct_events +=
            chunk.correct_events;
        conditioned_prediction_result->error_events += chunk.error_events;
        conditioned_prediction_result->somatic_error_events +=
            chunk.somatic_error_events;
        conditioned_prediction_result->positive_credit_events +=
            chunk.positive_credit_events;
        conditioned_prediction_result->negative_credit_events +=
            chunk.negative_credit_events;
      }
      if (conditioned_credit_events != nullptr) {
        std::vector<ConditionedCreditEvent> chunk_credit_events(
            2u * event_count);
        cuda_require(cudaMemcpy(chunk_credit_events.data(),
                                device_credit_events.get(),
                                device_credit_events.bytes(),
                                cudaMemcpyDeviceToHost),
                     "read conditioned physical credit bridge");
        conditioned_credit_events->insert(
            conditioned_credit_events->end(), chunk_credit_events.begin(),
            chunk_credit_events.end());
      }
      cuda_require(cudaMemcpy(&host_status, status.get(), sizeof(host_status),
                              cudaMemcpyDeviceToHost),
                   "read conditioned transition reserve status");
      if (host_status != 0u) {
        throw std::runtime_error(
            "conditioned transitions exceed fixed resident mass");
      }
      cuda_require(cudaMemset(conditioned_appended.get(), 0,
                              conditioned_appended.bytes()),
                   "clear conditioned transition append extent");
      append_conditioned_transitions_kernel<<<
          std::min(4096u, blocks_for(event_count)), kBlock>>>(
          sequence.get() + begin, chunk_count, prefix,
          segment_ids.get() + begin,
          state.online_conditioned_transitions.get(),
          state.online_conditioned_transition_counts.get(),
          state.online_conditioned_transition_count, event_count,
          conditioned_appended.get(), state.ledger.get());
      cuda_require(cudaGetLastError(),
                   "append resident conditioned transitions");
      cuda_require(cudaDeviceSynchronize(),
                   "complete resident conditioned transition append");
      std::uint32_t appended = 0u;
      cuda_require(cudaMemcpy(&appended, conditioned_appended.get(),
                              sizeof(appended), cudaMemcpyDeviceToHost),
                   "read boundary-aware conditioned transition extent");
      state.online_conditioned_transition_count += appended;
      sizes[6] = state.online_conditioned_transition_count;
      cuda_require(cudaMemcpy(state.mutable_sizes.get() + 6u, sizes + 6u,
                              sizeof(std::uint32_t), cudaMemcpyHostToDevice),
                   "publish conditioned transition extent");
    }
    consolidate_online_ngrams(state, conditioned_prefix, bigram_prefix,
                              trigram_prefix, association_prefix);
    if (conditioned_conductance_publisher != nullptr) {
      conditioned_conductance_publisher(conditioned_credit_context, state);
    } else {
      publish_resident_credit_conductance(state);
    }
    processed += fresh;
  }
  cuda_require(cudaMemcpy(sizes, state.mutable_sizes.get(), sizeof(sizes),
                          cudaMemcpyDeviceToHost),
               "read final chunked mutable extents");
  cuda_require(cudaMemcpy(&host_exact_replay, exact_replay.get(),
                          sizeof(host_exact_replay), cudaMemcpyDeviceToHost),
               "read resident replay gate");
  state.unit_count = sizes[0];
  state.online_bigram_count = sizes[1];
  state.online_trigram_count = sizes[2];
  state.online_association_count = sizes[3];
  state.online_episode_count = sizes[4];
  state.online_episode_break_count = sizes[5];
  state.online_conditioned_transition_count = sizes[6];
  if (host_exact_replay == 0u && state.novelty_epoch_pending) {
    state.novelty_episode_begin = previous_episode_count;
    state.novelty_episode_break_begin = previous_episode_break_count;
    state.novelty_epoch_active = true;
    state.novelty_epoch_pending = false;
  }
  // Every chunk above has already committed a sorted, reduced resident store.
  // Interrogative boundary evidence is learned whenever contact supplies it.
  // qorig_on gates the legacy autonomous injection policy only; it must not
  // erase construction matter that the resident Plan/surface path may later
  // use as a wording prior.
  if (state.online_episode_count > previous_episode_count) {
    const std::uint32_t appended_episode_count =
        state.online_episode_count - previous_episode_count;
    learn_qonset_terminal_kernel<<<blocks_for(appended_episode_count), kBlock>>>(
        state.unit_lengths.get(), state.unit_content.get(),
        state.online_episode_units.get(), previous_episode_count,
        state.online_episode_count,
        state.online_episode_breaks.get(), state.online_episode_break_count,
        state.role_canon_lesioned ? nullptr
                                  : state.construction_role_canon.get(),
        state.proposition_ordered_evidence_revision.get(),
        state.qonset_evidence_revision.get(),
        state.qonset_count.get(), state.qterm_count.get(), state.unit_count);
    cuda_require(cudaGetLastError(),
                 "launch interrogative construction boundary learning");
    if (state.qorig_on) {
      cuda_require(cudaDeviceSynchronize(),
                   "complete interrogative construction boundary learning");
      inject_qorig_units_deterministic(state);
    }
#ifdef QORIG_DEBUG
    { std::uint32_t on_n = 0u, tm_n = 0u;
      cudaMemcpy(&on_n, state.qorig_onset_n.get(), sizeof(on_n), cudaMemcpyDeviceToHost);
      cudaMemcpy(&tm_n, state.qorig_term_n.get(), sizeof(tm_n), cudaMemcpyDeviceToHost);
      std::vector<std::uint32_t> qo(state.unit_count), qt(state.unit_count);
      cudaMemcpy(qo.data(), state.qonset_count.get(), state.unit_count * sizeof(std::uint32_t), cudaMemcpyDeviceToHost);
      cudaMemcpy(qt.data(), state.qterm_count.get(), state.unit_count * sizeof(std::uint32_t), cudaMemcpyDeviceToHost);
      std::uint64_t qo_sum = 0ull, qt_sum = 0ull; std::uint32_t qo_max = 0u, qt_max = 0u;
      for (std::uint32_t v : qo) { qo_sum += v; if (v > qo_max) qo_max = v; }
      for (std::uint32_t v : qt) { qt_sum += v; if (v > qt_max) qt_max = v; }
      std::cerr << "[qorig-debug] onset_n=" << on_n << " term_n=" << tm_n
                << " online_episode_count=" << state.online_episode_count
                << " qonset_sum=" << qo_sum << " qonset_max=" << qo_max
                << " qterm_sum=" << qt_sum << " qterm_max=" << qt_max
                << " unit_count=" << state.unit_count << "\n";
      std::vector<std::uint32_t> lens(state.unit_count);
      std::vector<std::uint32_t> content(static_cast<std::size_t>(state.unit_count) * kUnitWords);
      cudaMemcpy(lens.data(), state.unit_lengths.get(), state.unit_count * sizeof(std::uint32_t), cudaMemcpyDeviceToHost);
      cudaMemcpy(content.data(), state.unit_content.get(), content.size() * sizeof(std::uint32_t), cudaMemcpyDeviceToHost);
      std::uint32_t units_ending_q = 0u, sample_shown = 0u;
      std::uint32_t end_dot = 0u, end_comma = 0u, end_bang = 0u, len0 = 0u;
      for (std::uint32_t u = 0u; u < state.unit_count; ++u) {
        const std::uint32_t L = lens[u];
        if (L == 0u) { ++len0; continue; }
        const std::uint32_t off = L - 1u;
        const std::uint32_t word = content[u * kUnitWords + off / 4u];
        const std::uint8_t byte = static_cast<std::uint8_t>(word >> ((off % 4u) * 8u));
        if (byte == '?') {
          ++units_ending_q;
          if (sample_shown < 5u) {
            std::string s;
            for (std::uint32_t k = 0u; k < L && k < 24u; ++k) {
              const std::uint32_t w = content[u * kUnitWords + k / 4u];
              s += static_cast<char>(static_cast<std::uint8_t>(w >> ((k % 4u) * 8u)));
            }
            std::cerr << "[qorig-debug] sample unit ending '?': len=" << L << " bytes=[" << s << "]\n";
            ++sample_shown;
          }
        }
        if (byte == '.') ++end_dot;
        if (byte == ',') ++end_comma;
        if (byte == '!') ++end_bang;
      }
      std::cerr << "[qorig-debug] units_ending_in_qmark=" << units_ending_q << "/" << state.unit_count
                << " end_dot=" << end_dot << " end_comma=" << end_comma << " end_bang=" << end_bang
                << " len0=" << len0 << "\n";
      { std::string s0, s1, s2;
        for (std::uint32_t k = 0u; k < lens[0] && k < 24u; ++k) { const std::uint32_t w = content[0u * kUnitWords + k / 4u]; s0 += static_cast<char>(static_cast<std::uint8_t>(w >> ((k % 4u) * 8u))); }
        for (std::uint32_t k = 0u; k < lens[1] && k < 24u; ++k) { const std::uint32_t w = content[1u * kUnitWords + k / 4u]; s1 += static_cast<char>(static_cast<std::uint8_t>(w >> ((k % 4u) * 8u))); }
        for (std::uint32_t k = 0u; k < lens[2] && k < 24u; ++k) { const std::uint32_t w = content[2u * kUnitWords + k / 4u]; s2 += static_cast<char>(static_cast<std::uint8_t>(w >> ((k % 4u) * 8u))); }
        std::cerr << "[qorig-debug] unit0=[" << s0 << "] len=" << lens[0]
                  << " unit1=[" << s1 << "] len=" << lens[1]
                  << " unit2=[" << s2 << "] len=" << lens[2] << "\n"; }
      std::vector<std::uint32_t> eu_sample(std::min<std::uint32_t>(30u, state.online_episode_count));
      cudaMemcpy(eu_sample.data(), state.online_episode_units.get(), eu_sample.size() * sizeof(std::uint32_t), cudaMemcpyDeviceToHost);
      std::cerr << "[qorig-debug] first episode units: ";
      for (std::uint32_t u : eu_sample) std::cerr << u << " ";
      std::cerr << "\n";
    }
#endif
  }
  build_unigram_top_kernel<<<1u, kBlock>>>(state.unit_vitality.get(), state.unit_count,
                                      state.unigram_top_ids.get());
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
  cuda_require(cudaGetLastError(), "audit online learned mass");
  cuda_require(cudaDeviceSynchronize(), "complete online learned mass audit");
  learn_incremental_surface_episodes(state, sequence.get(), sequence_count,
                                     segment_ids.get(), previous_unit_count);
  // The adult-stream transport/appraisal contract can deliberately run
  // without the resident language/surface organ. Construction learning is a
  // language-only developmental sweep: calling it for that configuration
  // needlessly re-scans the whole learned adult on every contact and can make
  // an otherwise small first real contact unbounded. The production language
  // path keeps the exact sweep when its resident surface is present.
  if (state.surface_organ_enabled) {
    learn_resident_constructions(
        state, sequence.get(), sequence_count, segment_ids.get(),
        proposition_efferent_contact, proposition_efferent_polarity,
        proposition_outcome_present, learn_relation_triples, initial_exposure,
        contact_learned_request);
  }
  assimilate_resident_candidate_episodes(
      state, sequence.get(), segment_ids.get(), sequence_count,
      proposition_efferent_contact, proposition_efferent_polarity,
      proposition_outcome_present);
  // Answers are not a side channel. Candidate event populations from this
  // same ordinary contact first update proposition tissue; only that changed
  // resident matter may discharge a latched information-gap goal.  The
  // resident adult stream can defer this one discharge until its emitted
  // question ticket has admitted the raw return.
  if (refresh_question_goal)
    refresh_resident_question_goal_after_contact(state);
}
