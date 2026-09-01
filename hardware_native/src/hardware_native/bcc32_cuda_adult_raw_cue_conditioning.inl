// Adult raw-cue conditioning pipeline.
//
// Included from bcc32_cuda_adult_v1.cuh inside its namespace and state-only
// guard as one dependency-ordered raw-contact conditioning responsibility.
inline void condition_on_raw_cue(AdultState& state, const std::uint8_t* host_bytes,
                                 std::uint32_t byte_count, bool diagnostic = false,
                                 bool resident_plan_only = false,
                                 bool preassembled_contact = false) {
  RawCueScratch& scratch = state.raw_cue_scratch;
  // Compact pre-v9 checkpoints reconstruct learned construction matter but
  // intentionally carry no in-flight cue. Recreate only this empty transient
  // workspace before the first post-restore contact.
  if (state.proposition_cue_sequence.get() == nullptr) {
    state.proposition_cue_sequence.allocate(kCompositionUnits);
    state.resident_bytes += state.proposition_cue_sequence.bytes();
  }
  if (state.proposition_cue_sequence_count.get() == nullptr) {
    state.proposition_cue_sequence_count.allocate(1u);
    state.resident_bytes += state.proposition_cue_sequence_count.bytes();
    cuda_require(cudaMemset(state.proposition_cue_sequence_count.get(), 0,
                            state.proposition_cue_sequence_count.bytes()),
                 "initialize restored ordered proposition cue extent");
  }
  cuda_require(cudaMemset(state.motor_context.get(), 0, state.motor_context.bytes()),
               "clear prior cue plan");
  cuda_require(cudaMemset(state.motor_completion.get(), 0, state.motor_completion.bytes()),
               "clear prior cue completion");
  if (state.subject_count.get() != nullptr) {
    cuda_require(cudaMemset(state.subject_count.get(), 0, state.subject_count.bytes()),
                 "clear prior subject field");
  }
  if (byte_count == 0u) {
    if (state.relation_cue_scores.get() != nullptr) {
      cuda_require(cudaMemset(state.relation_cue_scores.get(), 0,
                              state.relation_cue_scores.bytes()),
                   "clear persisted cue identity scores");
      cuda_require(cudaMemset(state.relation_cue_orders.get(), 0xff,
                              state.relation_cue_orders.bytes()),
                   "clear persisted cue unit orders");
      cuda_require(cudaMemset(state.relation_cue_exact.get(), 0,
                              state.relation_cue_exact.bytes()),
                   "clear persisted exact cue contacts");
      cuda_require(cudaMemset(state.proposition_cue_sequence_count.get(), 0,
                              state.proposition_cue_sequence_count.bytes()),
                   "clear ordered proposition cue extent");
      cuda_require(cudaMemset(state.relation_operator_order.get(), 0xff,
                              state.relation_operator_order.bytes()),
                   "clear learned cue operator order");
    }
    if (state.streaming_cue_mode && state.streaming_cue_meta.get() != nullptr)
      cuda_require(cudaMemset(state.streaming_cue_meta.get(), 0,
                              state.streaming_cue_meta.bytes()),
                   "clear consumed streaming cue");
    return;
  }
  DeviceArray<std::uint8_t>& incoming_bytes = scratch.incoming_bytes.renew(byte_count);
  cuda_require(cudaMemcpy(incoming_bytes.get(), host_bytes, byte_count,
                          cudaMemcpyHostToDevice),
               "upload raw cue bytes");
  std::uint32_t effective_count = byte_count;
  if (state.streaming_cue_mode && !preassembled_contact) {
    auto* incoming = incoming_bytes.get();
    std::uint32_t incoming_count = byte_count;
    auto* closure_bytes = state.construction_closure_bytes.get();
    std::uint32_t closure_count = state.construction_closure_count;
    bool complete_on_closure = true;
    bool flush_contact = false;
    auto* stream = state.streaming_cue_bytes.get();
    std::uint32_t capacity = kStreamingCueCapacity;
    auto* meta = state.streaming_cue_meta.get();
    void* append_arguments[] = {&incoming, &incoming_count, &closure_bytes,
                                &closure_count, &complete_on_closure,
                                &flush_contact, &stream, &capacity, &meta};
    cuda_require(
        cudaLaunchKernel(
            reinterpret_cast<const void*>(append_streaming_cue_kernel),
            dim3{1u, 1u, 1u}, dim3{2u, 1u, 1u}, append_arguments, 0u, nullptr),
        "launch append resident streaming cue");
    cuda_require(cudaGetLastError(), "append resident streaming cue");
    std::uint32_t stream_meta[3] = {};
    cuda_require(cudaMemcpy(stream_meta, state.streaming_cue_meta.get(),
                            sizeof(stream_meta), cudaMemcpyDeviceToHost),
                 "read resident streaming cue state");
    if (stream_meta[2] != 0u)
      throw std::runtime_error("resident streaming cue capacity exceeded");
    if (stream_meta[1] == 0u) return;
    effective_count = stream_meta[0];
  }
  DeviceArray<std::uint8_t>& device_bytes = scratch.device_bytes.renew(effective_count);
  cuda_require(cudaMemcpy(device_bytes.get(),
                          state.streaming_cue_mode && !preassembled_contact
                              ? state.streaming_cue_bytes.get()
                              : incoming_bytes.get(),
                          effective_count, cudaMemcpyDeviceToDevice),
               "materialize resident cue episode");
  // Once closure has materialized the complete cue into working matter, the
  // streaming buffer belongs to the next contact. Retaining a ready extent here
  // concatenates every later sentence onto the first and turns learned
  // settlement into an ever-growing ambiguous conversation-wide probe.
  if (state.streaming_cue_mode && !preassembled_contact) {
    cuda_require(cudaMemset(state.streaming_cue_meta.get(), 0,
                            state.streaming_cue_meta.bytes()),
                 "consume completed resident streaming cue");
  }
  byte_count = effective_count;
  if (state.distributed_motor_enabled) {
    cuda_require(distributed_motor::condition_raw(
                     distributed_motor_view(state), device_bytes.get(),
                     byte_count, 0u),
                 "condition distributed raw-event sequence motor");
  }
  const std::uint32_t no_relation = 2u;
  cuda_require(cudaMemcpy(state.motor_context.get() + 5u, &no_relation,
                          sizeof(no_relation), cudaMemcpyHostToDevice),
               "mark unresolved raw cue");
  const std::uint32_t scoped_episode_begin = 0u;
  const std::uint32_t scoped_break_begin = 0u;
  const std::uint32_t scoped_episode_count = state.online_episode_count;
  const std::uint32_t scoped_break_count = state.online_episode_break_count;
  const bool resident_composition_ready =
      resident_construction_admission_open(state);
  // Observed here, not later: everything below can be skipped by an early
  // return, and a silent skip is what made this chain unreadable before.
  state.subject_admission = SubjectAdmissionCensus{};
  state.subject_admission.observed = 1u;
  state.subject_admission.cue_conditioning_entered = 1u;
  state.subject_admission.scoped_episode_count = scoped_episode_count;

  DeviceArray<std::uint32_t>& flags = scratch.flags.renew(byte_count);
  DeviceArray<std::uint32_t>& anchors = scratch.anchors.renew(byte_count);
  DeviceArray<std::uint32_t>& scanned_ids = scratch.scanned_ids.renew(byte_count);
  mark_base_boundaries_kernel<<<blocks_for(byte_count), kBlock>>>(
      device_bytes.get(), byte_count, state.boundary_mask.get(), flags.get(), anchors.get());
  thrust::inclusive_scan(thrust::device, anchors.get(), anchors.get() + byte_count,
                         anchors.get(), thrust::maximum<std::uint32_t>());
  mark_bounded_units_kernel<<<blocks_for(byte_count), kBlock>>>(byte_count, anchors.get(),
                                                                flags.get());
  thrust::inclusive_scan(thrust::device, flags.get(), flags.get() + byte_count,
                         scanned_ids.get());
  std::uint32_t cue_count = 0u;
  cuda_require(cudaMemcpy(&cue_count, scanned_ids.get() + byte_count - 1u,
                          sizeof(cue_count), cudaMemcpyDeviceToHost),
               "read raw cue unit extent");
  DeviceArray<std::uint32_t>& starts = scratch.starts.renew(cue_count);
  scatter_unit_starts_kernel<<<blocks_for(byte_count), kBlock>>>(
      byte_count, flags.get(), scanned_ids.get(), starts.get());

  DeviceArray<std::uint32_t>& best_ids = scratch.best_ids.renew(cue_count);
  DeviceArray<std::uint32_t>& best_scores = scratch.best_scores.renew(cue_count);
  // The canonical resident-Plan path needs only two lossless products of the
  // raw contact: the distributed cue population formed above and the exact
  // learned-unit marks formed here.  It must not pay for (or become dependent
  // on) the legacy episode-alignment, association-ranking, and generator
  // fields below.  The default/full path remains byte-for-byte unchanged.
  // Exact learned-unit contact is an input to consolidated resident meaning,
  // not a lookup into the removable episode store. Source withdrawal may
  // disable legacy alignment, but cannot make a source-free Plan unreachable.
  if (resident_plan_only || scoped_episode_count == 0u) {
    state.subject_admission.exact_plan_branch_taken = 1u;
    match_cue_segments_parallel_kernel<<<cue_count, kBlock>>>(
        device_bytes.get(), byte_count, starts.get(), cue_count,
        state.unit_lengths.get(), state.unit_content.get(), state.unit_count,
        best_ids.get(), best_scores.get());
    auto* best_ids_ptr = best_ids.get();
    auto* best_scores_ptr = best_scores.get();
    std::uint32_t cue_count_arg = cue_count;
    std::uint32_t unit_count_arg = state.unit_count;
    auto* sequence = state.proposition_cue_sequence.get();
    std::uint32_t sequence_capacity = kCompositionUnits;
    auto* sequence_count = state.proposition_cue_sequence_count.get();
    void* sequence_arguments[] = {
        &best_ids_ptr, &best_scores_ptr, &cue_count_arg, &unit_count_arg,
        &sequence, &sequence_capacity, &sequence_count};
    cuda_require(
        cudaLaunchKernel(
            reinterpret_cast<const void*>(
                retain_exact_proposition_cue_sequence_kernel),
            dim3{1u, 1u, 1u}, dim3{2u, 1u, 1u}, sequence_arguments, 0u,
            nullptr),
        "launch exact resident proposition cue sequence");
    cuda_require(cudaGetLastError(),
                 "launch exact resident proposition cue sequence");
    if (state.relation_cue_exact.get() != nullptr) {
      cuda_require(cudaMemset(state.relation_cue_exact.get(), 0,
                              state.relation_cue_exact.bytes()),
                   "clear exact resident Plan cue contacts");
      cuda_require(cudaMemset(state.relation_cue_scores.get(), 0,
                              state.relation_cue_scores.bytes()),
                   "clear graded resident Plan cue contacts");
      cuda_require(cudaMemset(state.relation_cue_orders.get(), 0xff,
                              state.relation_cue_orders.bytes()),
                   "clear exact resident Plan cue order");
      const unsigned long long comparisons =
          static_cast<unsigned long long>(cue_count) * state.unit_count;
      const std::uint32_t comparison_blocks = static_cast<std::uint32_t>(
          (comparisons + kBlock - 1u) / kBlock);
      retain_close_cue_matches_kernel<<<comparison_blocks, kBlock>>>(
          device_bytes.get(), byte_count, starts.get(), cue_count,
          state.unit_lengths.get(), state.unit_content.get(), state.unit_count,
          best_scores.get(), state.relation_cue_scores.get(),
          state.relation_cue_orders.get());
      assign_cue_orders_kernel<<<comparison_blocks, kBlock>>>(
          device_bytes.get(), byte_count, starts.get(), cue_count,
          state.unit_lengths.get(), state.unit_content.get(), state.unit_count,
          state.relation_cue_scores.get(), state.relation_cue_orders.get());
      mark_exact_cue_surfaces_kernel<<<blocks_for(state.unit_count), kBlock>>>(
          best_ids.get(), best_scores.get(), cue_count,
          state.unit_lengths.get(), state.unit_content.get(), state.unit_count,
          state.relation_cue_exact.get());
      retain_exact_cue_representative_order_kernel<<<blocks_for(cue_count),
                                                     kBlock>>>(
          best_ids.get(), best_scores.get(), cue_count, state.unit_count,
          state.relation_cue_orders.get());
    }
    cuda_require(cudaGetLastError(), "form exact resident Plan cue");
    return;
  }
  DeviceArray<std::uint32_t>& cue_scores = scratch.cue_scores.renew(state.unit_count);
  DeviceArray<std::uint32_t>& cue_orders = scratch.cue_orders.renew(state.unit_count);
  DeviceArray<std::uint32_t>& route_scores = scratch.route_scores.renew(state.unit_count);
  DeviceArray<std::uint32_t>& route_orders = scratch.route_orders.renew(state.unit_count);
  DeviceArray<std::uint32_t>& cue_focus = scratch.cue_focus.renew(state.unit_count);
  DeviceArray<std::uint32_t>& cue_masks = scratch.cue_masks.renew(state.unit_count);
  DeviceArray<std::uint32_t>& salient_masks = scratch.salient_masks.renew(state.unit_count);
  DeviceArray<unsigned long long>& association_mass = scratch.association_mass.renew(state.unit_count);
  DeviceArray<unsigned long long>& forward_scores = scratch.forward_scores.renew(state.unit_count);
  DeviceArray<unsigned long long>& backward_scores = scratch.backward_scores.renew(state.unit_count);
  DeviceArray<std::uint32_t>& forward_support = scratch.forward_support.renew(state.unit_count);
  DeviceArray<std::uint32_t>& backward_support = scratch.backward_support.renew(state.unit_count);
  DeviceArray<std::uint32_t>& forward_salient_support = scratch.forward_salient_support.renew(state.unit_count);
  DeviceArray<unsigned long long>& strongest_evidence = scratch.strongest_evidence.renew(state.unit_count);
  DeviceArray<unsigned long long>& second_evidence = scratch.second_evidence.renew(state.unit_count);
  DeviceArray<unsigned long long>& cue_evidence = scratch.cue_evidence.renew(3u);
  DeviceArray<LocalSeedCandidate>& local_seed_candidates = scratch.local_seed_candidates.renew(scoped_episode_count);
  DeviceArray<LocalSeedCandidate>& clause_seed_candidates = scratch.clause_seed_candidates.renew(scoped_episode_count);
  DeviceArray<std::uint32_t>& alternate_motor_context = scratch.alternate_motor_context.renew(16u);
  DeviceArray<std::uint32_t>& alternate_motor_completion = scratch.alternate_motor_completion.renew(kCompositionUnits);
  DeviceArray<std::uint32_t>& episode_match_mask = scratch.episode_match_mask.renew(scoped_episode_count);
  DeviceArray<std::uint32_t>& episode_exact_match_mask = scratch.episode_exact_match_mask.renew(scoped_episode_count);
  DeviceArray<std::uint32_t>& episode_match_spans = scratch.episode_match_spans.renew(
      static_cast<std::size_t>(cue_count) * scoped_episode_count);
  DeviceArray<std::uint32_t>& scoped_episode_breaks = scratch.scoped_episode_breaks.renew(scoped_break_count);
  DeviceArray<std::uint32_t>& matched_segment_counts = scratch.matched_segment_counts.renew(cue_count);
  DeviceArray<std::uint32_t>& admitted_segment_mask = scratch.admitted_segment_mask.renew(1u);
  DeviceArray<std::uint32_t>& selected_anchor_ids = scratch.selected_anchor_ids.renew(kCueAnchorLimit);
  cuda_require(cudaMemset(cue_scores.get(), 0, cue_scores.bytes()), "clear cue unit field");
  cuda_require(cudaMemset(cue_orders.get(), 0xff, cue_orders.bytes()), "clear cue order field");
  cuda_require(cudaMemset(route_scores.get(), 0, route_scores.bytes()),
               "clear exact cue route field");
  cuda_require(cudaMemset(route_orders.get(), 0xff, route_orders.bytes()),
               "clear exact cue route order");
  cuda_require(cudaMemset(cue_focus.get(), 0, cue_focus.bytes()),
               "clear cue order support field");
  cuda_require(cudaMemset(cue_masks.get(), 0, cue_masks.bytes()),
               "clear informative cue masks");
  cuda_require(cudaMemset(salient_masks.get(), 0, salient_masks.bytes()),
               "clear salient cue masks");
  cuda_require(cudaMemset(association_mass.get(), 0, association_mass.bytes()),
               "clear resident association mass");
  cuda_require(cudaMemset(forward_scores.get(), 0, forward_scores.bytes()),
               "clear forward relation field");
  cuda_require(cudaMemset(backward_scores.get(), 0, backward_scores.bytes()),
               "clear backward relation field");
  cuda_require(cudaMemset(forward_support.get(), 0, forward_support.bytes()),
               "clear forward support field");
  cuda_require(cudaMemset(backward_support.get(), 0, backward_support.bytes()),
               "clear backward support field");
  cuda_require(cudaMemset(forward_salient_support.get(), 0,
                          forward_salient_support.bytes()),
               "clear salient relation support");
  cuda_require(cudaMemset(strongest_evidence.get(), 0, strongest_evidence.bytes()),
               "clear strongest relation evidence");
  cuda_require(cudaMemset(second_evidence.get(), 0, second_evidence.bytes()),
               "clear reliable relation evidence");
  cuda_require(cudaMemset(cue_evidence.get(), 0, cue_evidence.bytes()),
               "clear cue relation evidence");
  cuda_require(cudaMemset(episode_match_mask.get(), 0, episode_match_mask.bytes()),
               "clear cue occurrence masks");
  cuda_require(cudaMemset(episode_exact_match_mask.get(), 0,
                          episode_exact_match_mask.bytes()),
               "clear exact cue occurrence masks");
  cuda_require(cudaMemset(episode_match_spans.get(), 0, episode_match_spans.bytes()),
               "clear cue occurrence spans");
  cuda_require(cudaMemset(matched_segment_counts.get(), 0, matched_segment_counts.bytes()),
               "clear cue occurrence counts");
  cuda_require(cudaMemset(admitted_segment_mask.get(), 0, admitted_segment_mask.bytes()),
               "clear admitted cue segment mask");
  if (scoped_break_count != 0u) {
    normalize_episode_breaks_kernel<<<blocks_for(scoped_break_count), kBlock>>>(
        state.online_episode_breaks.get() + scoped_break_begin,
        scoped_break_count, scoped_episode_begin, scoped_episode_breaks.get());
  }
  match_cue_segments_parallel_kernel<<<cue_count, kBlock>>>(
      device_bytes.get(), byte_count, starts.get(), cue_count, state.unit_lengths.get(),
      state.unit_content.get(), state.unit_count, best_ids.get(), best_scores.get());
  auto* best_ids_ptr = best_ids.get();
  auto* best_scores_ptr = best_scores.get();
  std::uint32_t cue_count_arg = cue_count;
  std::uint32_t unit_count_arg = state.unit_count;
  auto* sequence = state.proposition_cue_sequence.get();
  std::uint32_t sequence_capacity = kCompositionUnits;
  auto* sequence_count = state.proposition_cue_sequence_count.get();
  void* sequence_arguments[] = {
      &best_ids_ptr, &best_scores_ptr, &cue_count_arg, &unit_count_arg,
      &sequence, &sequence_capacity, &sequence_count};
  cuda_require(
      cudaLaunchKernel(
          reinterpret_cast<const void*>(
              retain_exact_proposition_cue_sequence_kernel),
          dim3{1u, 1u, 1u}, dim3{2u, 1u, 1u}, sequence_arguments, 0u, nullptr),
      "launch exact resident proposition cue sequence");
  cuda_require(cudaGetLastError(),
               "launch exact resident proposition cue sequence");
  const unsigned long long comparisons =
      static_cast<unsigned long long>(cue_count) * state.unit_count;
  if (comparisons > static_cast<unsigned long long>(std::numeric_limits<std::uint32_t>::max()) *
                        kBlock) {
    throw std::runtime_error("cue comparison field exceeds CUDA grid extent");
  }
  const std::uint32_t comparison_blocks = static_cast<std::uint32_t>(
      (comparisons + kBlock - 1u) / kBlock);
  retain_close_cue_matches_kernel<<<comparison_blocks, kBlock>>>(
      device_bytes.get(), byte_count, starts.get(), cue_count, state.unit_lengths.get(),
      state.unit_content.get(), state.unit_count, best_scores.get(), cue_scores.get(),
      cue_orders.get());
  assign_cue_orders_kernel<<<comparison_blocks, kBlock>>>(
      device_bytes.get(), byte_count, starts.get(), cue_count, state.unit_lengths.get(),
      state.unit_content.get(), state.unit_count, cue_scores.get(), cue_orders.get());
  const dim3 alignment_grid(blocks_for(scoped_episode_count), cue_count);
  match_cue_segment_sequences_kernel<<<alignment_grid, kBlock>>>(
      device_bytes.get(), starts.get(), byte_count, cue_count,
      state.boundary_mask.get(),
      state.unit_lengths.get(), state.unit_content.get(),
      state.online_episode_units.get() + scoped_episode_begin, scoped_episode_count,
      scoped_episode_breaks.get(), scoped_break_count,
      episode_match_mask.get(), episode_exact_match_mask.get(),
      episode_match_spans.get(),
      matched_segment_counts.get(), admitted_segment_mask.get(),
      state.novelty_epoch_active ? 1u : 0u);
  accumulate_association_mass_kernel<<<blocks_for(state.online_association_count), kBlock>>>(
      state.online_associations.get(), state.online_association_counts.get(),
      state.online_association_count, association_mass.get());
  select_informative_cue_anchors_kernel<<<1u, 1u>>>(
      best_ids.get(), best_scores.get(), matched_segment_counts.get(),
      cue_count, state.unit_count,
      state.online_association_count, state.unit_lengths.get(), state.unit_content.get(),
      state.unit_vitality.get(),
      association_mass.get(), state.bigrams.get(), state.bigram_counts.get(),
      state.bigram_count, state.online_bigrams.get(), state.online_bigram_counts.get(),
      state.online_bigram_count, state.online_conditioned_transitions.get(),
      state.online_conditioned_transition_count, route_scores.get(), route_orders.get(),
      cue_focus.get(),
      cue_masks.get(), salient_masks.get(), admitted_segment_mask.get(),
      selected_anchor_ids.get(), diagnostic ? 1u : 0u);
  propagate_ranked_cue_anchor_bytes_kernel<<<blocks_for(state.unit_count), kBlock>>>(
      selected_anchor_ids.get(), state.unit_lengths.get(), state.unit_content.get(),
      state.unit_count, route_scores.get(), route_orders.get(), cue_focus.get(),
      cue_masks.get(), salient_masks.get());
  // Persist the CURRENT cue's near-identity scores for the relational-
  // triple channel (device-to-device; which glue words the question said).
  if (state.relation_cue_scores.get() != nullptr) {
    cuda_require(cudaMemset(state.relation_cue_exact.get(), 0,
                            state.relation_cue_exact.bytes()),
                 "clear current exact cue contacts");
    mark_exact_cue_surfaces_kernel<<<blocks_for(state.unit_count), kBlock>>>(
        best_ids.get(), best_scores.get(), cue_count, state.unit_lengths.get(),
        state.unit_content.get(), state.unit_count,
        state.relation_cue_exact.get());
    cuda_require(cudaGetLastError(), "mark exact cue contacts");
    if (std::getenv("BCC32_TRIPLE_DIAG") != nullptr) {
      std::vector<std::uint32_t> exact_host(state.unit_count);
      cuda_require(cudaMemcpy(exact_host.data(), state.relation_cue_exact.get(),
                              exact_host.size() * sizeof(std::uint32_t),
                              cudaMemcpyDeviceToHost), "diag: read exact set");
      std::vector<std::uint32_t> lengths_host(state.unit_count);
      cuda_require(cudaMemcpy(lengths_host.data(), state.unit_lengths.get(),
                              lengths_host.size() * sizeof(std::uint32_t),
                              cudaMemcpyDeviceToHost), "diag: read lengths");
      std::fprintf(stderr, "exact_cue_set");
      std::uint32_t shown = 0u;
      for (std::uint32_t u = 0u; u < state.unit_count && shown < 40u; ++u) {
        if (exact_host[u] == 0u) continue;
        std::uint32_t words[kUnitWords] = {};
        cuda_require(cudaMemcpy(words,
                                state.unit_content.get() +
                                    static_cast<std::size_t>(u) * kUnitWords,
                                sizeof(words), cudaMemcpyDeviceToHost),
                     "diag: read exact unit");
        std::fprintf(stderr, " u%u'", u);
        for (std::uint32_t o = 0u; o < std::min(lengths_host[u], 14u); ++o) {
          const std::uint8_t b = static_cast<std::uint8_t>(
              words[o / 4u] >> ((o % 4u) * 8u));
          std::fputc(b >= 32u && b < 127u ? b : '.', stderr);
        }
        std::fputc('\'', stderr);
        ++shown;
      }
      std::fprintf(stderr, "\n");
    }
    cuda_require(cudaMemcpy(state.relation_cue_scores.get(), cue_scores.get(),
                            static_cast<std::size_t>(state.unit_count) *
                                sizeof(std::uint32_t),
                            cudaMemcpyDeviceToDevice),
                 "persist cue identity scores");
    cuda_require(cudaMemcpy(state.relation_cue_orders.get(), cue_orders.get(),
                            static_cast<std::size_t>(state.unit_count) *
                                sizeof(std::uint32_t),
                            cudaMemcpyDeviceToDevice),
                 "persist cue unit orders");
    cuda_require(cudaMemset(state.relation_operator_order.get(), 0xff,
                            state.relation_operator_order.bytes()),
                 "clear learned cue operator order");
    construction::derive_relation_operator_order_kernel<<<
        blocks_for(state.unit_count), kBlock>>>(
        state.relation_cue_scores.get(), state.relation_cue_orders.get(),
        state.role_canon_lesioned ? nullptr
                                  : state.construction_role_canon.get(),
        state.qonset_count.get(), state.relation_triple_type_total.get(),
        kCueNearIdentity, state.unit_count,
        state.relation_operator_order.get());
    cuda_require(cudaGetLastError(), "derive learned cue operator order");
  }
  // Latch the cue's content units into the resident subject field so the
  // generator can keep the question's topic active across the whole answer.
  // The census runs UNCONDITIONALLY and the latch stays gated. Recording it
  // only inside the gated branch would make the two failure modes -- "no unit
  // qualified" and "the latch never ran" -- indistinguishable, which is
  // exactly the confusion this observable exists to remove.
  if (state.subject_count.get() != nullptr) {
    observe_subject_admission(cue_scores.get(), state,
                              resident_composition_ready);
  }
  if (state.subject_count.get() != nullptr && resident_composition_ready) {
    // Deterministic host-side collection (see collect_subject_field_deterministic
    // for why the GPU kernel version was retired). The emergent word-class
    // field (build_unit_pos, the surgical 473af804d sub-diff) is built
    // lazily and rebuilt when online assimilation has grown the unit
    // inventory, so FUNC residue ("then", "how") cannot out-weigh the real
    // topic word in the latch.
    if (state.unit_pos.get() == nullptr ||
        state.unit_pos.size() < state.unit_count)
      build_unit_pos(state);
    // Re-observe now that unit_pos exists: the first census ran with the FUNC
    // predicate DISABLED (unit_pos_available=0), so its cue_units_nonfunc was
    // not a real test. This second pass is the one whose numbers can be
    // compared against admitted.
    observe_subject_admission(cue_scores.get(), state, true);
    collect_subject_field_deterministic(cue_scores.get(), state);
    std::uint32_t admitted = 0u;
    cuda_require(cudaMemcpy(&admitted, state.subject_count.get(),
                            sizeof(admitted), cudaMemcpyDeviceToHost),
                 "admission census: read admitted subject count");
    state.subject_admission.admitted_subject_count = admitted;
  }
  score_composition_relations_kernel<<<blocks_for(state.online_association_count), kBlock>>>(
      state.online_associations.get(), state.online_association_counts.get(),
      state.online_association_count, route_scores.get(), route_orders.get(), cue_focus.get(),
      cue_masks.get(), salient_masks.get(), forward_scores.get(), backward_scores.get(),
      forward_support.get(), backward_support.get(), forward_salient_support.get(),
      strongest_evidence.get(),
      second_evidence.get(), cue_evidence.get());
  assess_composition_cue_kernel<<<1u, kBlock>>>(
      best_ids.get(), best_scores.get(), starts.get(), byte_count, cue_count,
      route_scores.get(), state.bigrams.get(), state.bigram_counts.get(),
      state.bigram_count, state.online_bigrams.get(), state.online_bigram_counts.get(),
      state.online_bigram_count, cue_evidence.get(), state.motor_context.get());
  if (resident_composition_ready) {
    namespace synthesis = bcc32_cuda_resident_synthesis;
    namespace roles = substrate::bcc32::resident_roles;
    DeviceArray<unsigned long long>& synthesis_cue_activation = scratch.synthesis_cue_activation.renew(state.unit_count);
    DeviceArray<unsigned long long>& synthesis_propagated_activation = scratch.synthesis_propagated_activation.renew(state.unit_count);
    DeviceArray<unsigned long long>& synthesis_forward_activation = scratch.synthesis_forward_activation.renew(state.unit_count);
    DeviceArray<unsigned long long>& synthesis_backward_activation = scratch.synthesis_backward_activation.renew(state.unit_count);
    DeviceArray<std::uint32_t>& synthesis_cue_masks = scratch.synthesis_cue_masks.renew(state.unit_count);
    DeviceArray<std::uint32_t>& synthesis_salient_masks = scratch.synthesis_salient_masks.renew(state.unit_count);
    DeviceArray<std::uint32_t>& synthesis_propagated_masks = scratch.synthesis_propagated_masks.renew(state.unit_count);
    DeviceArray<std::uint32_t>& synthesis_propagated_salient_masks = scratch.synthesis_propagated_salient_masks.renew(state.unit_count);
    DeviceArray<std::uint32_t>& synthesis_forward_support = scratch.synthesis_forward_support.renew(state.unit_count);
    DeviceArray<std::uint32_t>& synthesis_backward_support = scratch.synthesis_backward_support.renew(state.unit_count);
    cuda_require(cudaMemset(synthesis_cue_activation.get(), 0,
                            synthesis_cue_activation.bytes()),
                 "clear resident synthesis cue activation");
    cuda_require(cudaMemset(synthesis_propagated_activation.get(), 0,
                            synthesis_propagated_activation.bytes()),
                 "clear propagated synthesis activation");
    cuda_require(cudaMemset(synthesis_forward_activation.get(), 0,
                            synthesis_forward_activation.bytes()),
                 "clear forward synthesis activation");
    cuda_require(cudaMemset(synthesis_backward_activation.get(), 0,
                            synthesis_backward_activation.bytes()),
                 "clear backward synthesis activation");
    cuda_require(cudaMemset(synthesis_cue_masks.get(), 0,
                            synthesis_cue_masks.bytes()),
                 "clear resident synthesis cue masks");
    cuda_require(cudaMemset(synthesis_salient_masks.get(), 0,
                            synthesis_salient_masks.bytes()),
                 "clear resident synthesis salient masks");
    cuda_require(cudaMemset(synthesis_propagated_masks.get(), 0,
                            synthesis_propagated_masks.bytes()),
                 "clear propagated synthesis cue masks");
    cuda_require(cudaMemset(synthesis_propagated_salient_masks.get(), 0,
                            synthesis_propagated_salient_masks.bytes()),
                 "clear propagated synthesis salient masks");
    cuda_require(cudaMemset(synthesis_forward_support.get(), 0,
                            synthesis_forward_support.bytes()),
                 "clear forward synthesis support");
    cuda_require(cudaMemset(synthesis_backward_support.get(), 0,
                            synthesis_backward_support.bytes()),
                 "clear backward synthesis support");
    seed_resident_synthesis_cue_activation_kernel<<<
        blocks_for(cue_count), kBlock>>>(
        best_ids.get(), best_scores.get(), matched_segment_counts.get(), cue_count,
        state.unit_lengths.get(), state.unit_vitality.get(), state.unit_count,
        synthesis_cue_activation.get(), synthesis_cue_masks.get(),
        synthesis_salient_masks.get());
    propagate_resident_synthesis_cue_bytes_kernel<<<
        blocks_for(state.unit_count), kBlock>>>(
        best_ids.get(), best_scores.get(), matched_segment_counts.get(), cue_count,
        state.unit_lengths.get(), state.unit_content.get(),
        state.unit_vitality.get(), state.unit_count,
        synthesis_cue_activation.get(), synthesis_cue_masks.get(),
        synthesis_salient_masks.get());
    propagate_resident_synthesis_cue_activation_kernel<<<
        blocks_for(state.online_association_count), kBlock>>>(
        state.online_associations.get(), state.online_association_counts.get(),
        state.online_association_count, state.unit_vitality.get(),
        synthesis_cue_activation.get(), synthesis_cue_masks.get(),
        synthesis_salient_masks.get(),
        synthesis_propagated_activation.get(), synthesis_propagated_masks.get(),
        synthesis_propagated_salient_masks.get());
    merge_resident_synthesis_cue_activation_kernel<<<
        blocks_for(state.unit_count), kBlock>>>(
        route_scores.get(), cue_focus.get(),
        synthesis_propagated_activation.get(), synthesis_propagated_masks.get(),
        synthesis_propagated_salient_masks.get(), state.unit_count,
        synthesis_cue_activation.get(), synthesis_cue_masks.get(),
        synthesis_salient_masks.get());
    propagate_resident_synthesis_direction_kernel<<<
        blocks_for(state.online_association_count), kBlock>>>(
        state.online_associations.get(), state.online_association_counts.get(),
        state.online_association_count, state.unit_vitality.get(),
        synthesis_cue_activation.get(), synthesis_cue_masks.get(),
        synthesis_forward_activation.get(), synthesis_backward_activation.get(),
        synthesis_forward_support.get(), synthesis_backward_support.get());
    const std::uint32_t base_source_windows = 0u;
    const std::uint32_t online_source_windows =
        state.online_episode_count >= synthesis::kResidentSynthesisNoveltyUnits
            ? state.online_episode_count - synthesis::kResidentSynthesisNoveltyUnits + 1u
            : 0u;
    const std::uint32_t source_window_count =
        base_source_windows + online_source_windows;
    DeviceArray<synthesis::ResidentSourceWindowSignature>& synthesis_source_windows =
        scratch.synthesis_source_windows.renew(source_window_count);
    if (online_source_windows != 0u) {
      build_resident_source_window_signatures_kernel<<<
          blocks_for(online_source_windows), kBlock>>>(
          state.online_episode_units.get(), state.online_episode_count,
          synthesis_source_windows.get(), base_source_windows);
    }
    if (source_window_count != 0u) {
      thrust::sort(thrust::device, synthesis_source_windows.get(),
                   synthesis_source_windows.get() + source_window_count);
    }
    DeviceArray<std::int32_t>& synthesis_role_projection = scratch.synthesis_role_projection.renew(
        roles::role_projection_scratch_words(state.unit_count));
    DeviceArray<roles::MutableStructuralRole>& synthesis_roles = scratch.synthesis_roles.renew(state.unit_count);
    DeviceArray<std::uint64_t>& synthesis_base_role_bigrams = scratch.synthesis_base_role_bigrams.renew(
        roles::kRoleBigramTableSize);
    DeviceArray<std::uint64_t>& synthesis_base_role_trigrams = scratch.synthesis_base_role_trigrams.renew(
        roles::kRoleTrigramTableSize);
    DeviceArray<std::uint64_t>& synthesis_online_role_bigrams = scratch.synthesis_online_role_bigrams.renew(
        roles::kRoleBigramTableSize);
    DeviceArray<std::uint64_t>& synthesis_online_role_trigrams = scratch.synthesis_online_role_trigrams.renew(
        roles::kRoleTrigramTableSize);
    cuda_require(roles::derive_structural_roles_cuda(
                     state.unit_count, state.bigrams.get(),
                     state.bigram_counts.get(), state.bigram_count,
                     state.trigrams.get(), state.trigram_counts.get(),
                     state.trigram_count, state.online_bigrams.get(),
                     state.online_bigram_counts.get(), state.online_bigram_count,
                     state.online_trigrams.get(), state.online_trigram_counts.get(),
                     state.online_trigram_count, synthesis_role_projection.get(),
                     synthesis_roles.get()),
                 "derive mutable resident synthesis roles");
    roles::MutableRoleEvidenceTables synthesis_role_tables{};
    synthesis_role_tables.base_grammar = {
        synthesis_base_role_bigrams.get(), synthesis_base_role_trigrams.get()};
    synthesis_role_tables.online_content = {
        synthesis_online_role_bigrams.get(), synthesis_online_role_trigrams.get()};
    cuda_require(roles::build_role_evidence_tables_cuda(
                     synthesis_roles.get(), state.unit_count, state.bigrams.get(),
                     state.bigram_counts.get(), state.bigram_count,
                     state.trigrams.get(), state.trigram_counts.get(),
                     state.trigram_count, state.online_bigrams.get(),
                     state.online_bigram_counts.get(), state.online_bigram_count,
                     state.online_trigrams.get(), state.online_trigram_counts.get(),
                     state.online_trigram_count, synthesis_role_tables),
                 "build mutable resident synthesis role evidence");
    DeviceArray<std::uint32_t>& learned_relation_predicates = scratch.learned_relation_predicates.renew(
        kLearnedRelationPredicateLimit);
    DeviceArray<std::uint32_t>& learned_relation_predicate_count = scratch.learned_relation_predicate_count.renew(1u);
    DeviceArray<std::uint32_t>& learned_relation_cue_indices = scratch.learned_relation_cue_indices.renew(state.unit_count);
    DeviceArray<std::uint32_t>& learned_relation_cue_scores = scratch.learned_relation_cue_scores.renew(state.unit_count);
    DeviceArray<std::uint32_t>& learned_relation_predicate_owners = scratch.learned_relation_predicate_owners.renew(state.unit_count);
    DeviceArray<std::uint32_t>& learned_relation_binding_enabled = scratch.learned_relation_binding_enabled.renew(1u);
    DeviceArray<std::uint32_t>& learned_relation_subject_anchors = scratch.learned_relation_subject_anchors.renew(kCueAnchorLimit);
    DeviceArray<std::uint32_t>& learned_relation_cue_units = scratch.learned_relation_cue_units.renew(cue_count + 1u);
    DeviceArray<std::uint32_t>& learned_relation_cue_count_device = scratch.learned_relation_cue_count_device.renew(1u);
    DeviceArray<unsigned long long>& learned_relation_bindings = scratch.learned_relation_bindings.renew(state.unit_count);
    DeviceArray<unsigned long long>& learned_relation_strongest_binding = scratch.learned_relation_strongest_binding.renew(1u);
    DeviceArray<std::uint32_t>& learned_relation_selected_predicate = scratch.learned_relation_selected_predicate.renew(1u);
    cuda_require(cudaMemset(learned_relation_predicate_count.get(), 0,
                            learned_relation_predicate_count.bytes()),
                 "clear learned relation predicate count");
    cuda_require(cudaMemset(learned_relation_bindings.get(), 0,
                            learned_relation_bindings.bytes()),
                 "clear learned relation bindings");
    cuda_require(cudaMemset(learned_relation_strongest_binding.get(), 0,
                            learned_relation_strongest_binding.bytes()),
                 "clear strongest learned relation binding");
    cuda_require(cudaMemset(learned_relation_predicate_owners.get(), 0xff,
                            learned_relation_predicate_owners.bytes()),
                 "clear learned relation predicate owners");
    cuda_require(cudaMemset(learned_relation_selected_predicate.get(), 0xff,
                            learned_relation_selected_predicate.bytes()),
                 "clear selected learned relation predicate");
    cuda_require(cudaMemcpy(learned_relation_subject_anchors.get(),
                            selected_anchor_ids.get(),
                            kCueAnchorLimit * sizeof(std::uint32_t),
                            cudaMemcpyDeviceToDevice),
                 "copy learned relation subject view");
    cuda_require(cudaMemset(learned_relation_cue_units.get(), 0xff,
                            learned_relation_cue_units.bytes()),
                 "clear learned relation cue view");
    cuda_require(cudaMemcpy(learned_relation_cue_count_device.get(), &cue_count,
                            sizeof(cue_count), cudaMemcpyHostToDevice),
                 "initialize learned relation cue extent");
    cuda_require(cudaMemset(learned_relation_binding_enabled.get(), 0xff,
                            learned_relation_binding_enabled.bytes()),
                 "enable learned relation binding");
    collect_conditioned_subject_predicates_kernel<<<1u, 1u>>>(
        selected_anchor_ids.get(), state.online_conditioned_transitions.get(),
        state.online_conditioned_transition_conductance.get(),
        state.online_conditioned_transition_count,
        learned_relation_predicates.get(), learned_relation_predicate_count.get(),
        learned_relation_predicate_owners.get());
    map_learned_relation_cue_surfaces_kernel<<<
        blocks_for(state.unit_count), kBlock>>>(
        best_ids.get(), cue_count, state.unit_lengths.get(),
        state.unit_content.get(), state.unit_count,
        learned_relation_cue_indices.get(), learned_relation_cue_scores.get());
    score_learned_conditioned_relation_bindings_kernel<<<
        blocks_for(state.online_conditioned_transition_count), kBlock>>>(
        state.online_conditioned_transitions.get(),
        state.online_conditioned_transition_conductance.get(),
        state.online_conditioned_transition_count, state.unit_vitality.get(),
        learned_relation_cue_indices.get(), learned_relation_cue_scores.get(),
        learned_relation_predicates.get(), learned_relation_predicate_count.get(),
        best_ids.get(),
        route_orders.get(), synthesis_cue_masks.get(),
        learned_relation_predicate_owners.get(), selected_anchor_ids.get(),
        learned_relation_binding_enabled.get(),
        device_bytes.get(), byte_count, state.boundary_mask.get(),
        state.unit_lengths.get(), state.unit_content.get(),
        state.online_episode_units.get(),
        state.online_episode_count, state.online_episode_breaks.get(),
        state.online_episode_break_count,
        learned_relation_bindings.get());
    select_strongest_learned_relation_binding_kernel<<<
        blocks_for(state.unit_count), kBlock>>>(
        learned_relation_bindings.get(), state.unit_count,
        learned_relation_strongest_binding.get());
    {
      auto* predicate_bindings = learned_relation_bindings.get();
      auto* strongest_binding = learned_relation_strongest_binding.get();
      std::uint32_t unit_count = state.unit_count;
      auto* predicate_owners = learned_relation_predicate_owners.get();
      auto* original_subject_anchors = selected_anchor_ids.get();
      std::uint32_t anchor_count = kCueAnchorLimit;
      auto* subject_anchor_view = learned_relation_subject_anchors.get();
      auto* selected_predicate = learned_relation_selected_predicate.get();
      void* subject_arguments[] = {
          &predicate_bindings, &strongest_binding, &unit_count,
          &predicate_owners, &original_subject_anchors, &anchor_count,
          &subject_anchor_view, &selected_predicate};
      cuda_require(
          cudaLaunchKernel(
              reinterpret_cast<const void*>(
                  select_learned_relation_subject_kernel),
              dim3{1u, 1u, 1u}, dim3{2u, 1u, 1u}, subject_arguments, 0u,
              nullptr),
          "launch first conditioned-relation subject selection");
      cuda_require(cudaGetLastError(),
                   "launch first conditioned-relation subject selection");
    }
    {
      auto* predicates = learned_relation_predicates.get();
      auto* predicate_count = learned_relation_predicate_count.get();
      auto* predicate_owners = learned_relation_predicate_owners.get();
      auto* subject_anchors = learned_relation_subject_anchors.get();
      auto* original_subject_anchors = selected_anchor_ids.get();
      auto* raw_cue_bytes = device_bytes.get();
      std::uint32_t raw_cue_byte_count = byte_count;
      auto* boundary_mask = state.boundary_mask.get();
      auto* unit_lengths = state.unit_lengths.get();
      auto* unit_content = state.unit_content.get();
      auto* episode_units = state.online_episode_units.get();
      std::uint32_t episode_count = state.online_episode_count;
      auto* episode_breaks = state.online_episode_breaks.get();
      std::uint32_t episode_break_count = state.online_episode_break_count;
      auto* binding_enabled = learned_relation_binding_enabled.get();
      auto* strongest_binding = learned_relation_strongest_binding.get();
      auto* selected_predicate = learned_relation_selected_predicate.get();
      void* detection_arguments[] = {
          &predicates, &predicate_count, &predicate_owners, &subject_anchors,
          &original_subject_anchors, &raw_cue_bytes, &raw_cue_byte_count,
          &boundary_mask, &unit_lengths, &unit_content, &episode_units,
          &episode_count, &episode_breaks, &episode_break_count,
          &binding_enabled, &strongest_binding, &selected_predicate};
      cuda_require(
          cudaLaunchKernel(
              reinterpret_cast<const void*>(
                  detect_direct_conditioned_relation_kernel),
              dim3{1u, 1u, 1u}, dim3{2u, 1u, 1u}, detection_arguments, 0u,
              nullptr),
          "launch direct conditioned-relation detection");
      cuda_require(cudaGetLastError(),
                   "launch direct conditioned-relation detection");
    }
    {
      auto* predicate_bindings = learned_relation_bindings.get();
      auto* strongest_binding = learned_relation_strongest_binding.get();
      std::uint32_t unit_count = state.unit_count;
      auto* predicate_owners = learned_relation_predicate_owners.get();
      auto* original_subject_anchors = selected_anchor_ids.get();
      std::uint32_t anchor_count = kCueAnchorLimit;
      auto* subject_anchor_view = learned_relation_subject_anchors.get();
      auto* selected_predicate = learned_relation_selected_predicate.get();
      void* subject_arguments[] = {
          &predicate_bindings, &strongest_binding, &unit_count,
          &predicate_owners, &original_subject_anchors, &anchor_count,
          &subject_anchor_view, &selected_predicate};
      cuda_require(
          cudaLaunchKernel(
              reinterpret_cast<const void*>(
                  select_learned_relation_subject_kernel),
              dim3{1u, 1u, 1u}, dim3{2u, 1u, 1u}, subject_arguments, 0u,
              nullptr),
          "launch second conditioned-relation subject selection");
      cuda_require(cudaGetLastError(),
                   "launch second conditioned-relation subject selection");
    }
    apply_learned_conditioned_relation_bindings_kernel<<<
        blocks_for(state.unit_count), kBlock>>>(
        learned_relation_bindings.get(), learned_relation_strongest_binding.get(),
        state.unit_count, learned_relation_selected_predicate.get(),
        learned_relation_subject_anchors.get(),
        best_ids.get(), cue_count,
        synthesis_cue_activation.get(), synthesis_cue_masks.get(), route_orders.get());
    build_learned_relation_cue_view_kernel<<<blocks_for(state.unit_count), kBlock>>>(
        best_ids.get(), cue_count, learned_relation_subject_anchors.get(),
        learned_relation_bindings.get(), learned_relation_strongest_binding.get(),
        state.unit_count, learned_relation_selected_predicate.get(),
        learned_relation_cue_units.get(), learned_relation_cue_count_device.get());
    std::uint32_t learned_relation_cue_count = cue_count;
    cuda_require(cudaMemcpy(&learned_relation_cue_count,
                            learned_relation_cue_count_device.get(),
                            sizeof(learned_relation_cue_count),
                            cudaMemcpyDeviceToHost),
                 "read learned relation cue extent");
    const std::size_t reverse_bigram_count =
        static_cast<std::size_t>(state.bigram_count) + state.online_bigram_count;
    const std::size_t reverse_trigram_count =
        static_cast<std::size_t>(state.trigram_count) + state.online_trigram_count;
    DeviceArray<synthesis::ReverseBigramEdge>& reverse_bigrams = scratch.reverse_bigrams.renew(reverse_bigram_count);
    DeviceArray<synthesis::ReverseTrigramEdge>& reverse_trigrams = scratch.reverse_trigrams.renew(reverse_trigram_count);
    DeviceArray<std::uint32_t>& synthesis_output_closure = scratch.synthesis_output_closure.renew(kClosureCount);
    discover_output_closure_kernel<<<1u, 256u>>>(
        state.boundary_histogram.get(), state.boundary_pairs.get(),
        state.boundary_bytes.get(), synthesis_output_closure.get());
    cuda_require(cudaGetLastError(), "launch resident output closure discovery");
    DeviceArray<std::uint32_t>& synthesis_seeds = scratch.synthesis_seeds.renew(synthesis::kResidentSynthesisMaxSeeds);
    DeviceArray<unsigned long long>& synthesis_seed_scores = scratch.synthesis_seed_scores.renew(
        synthesis::kResidentSynthesisMaxSeeds);
    DeviceArray<std::uint32_t>& synthesis_drafts = scratch.synthesis_drafts.renew(
        synthesis::kResidentSynthesisMaxCandidates *
        synthesis::kResidentSynthesisMaxUnits);
    DeviceArray<std::uint32_t>& synthesis_draft_lengths = scratch.synthesis_draft_lengths.renew(
        synthesis::kResidentSynthesisMaxCandidates);
    DeviceArray<unsigned long long>& synthesis_draft_scores = scratch.synthesis_draft_scores.renew(
        synthesis::kResidentSynthesisMaxCandidates);
    DeviceArray<std::uint32_t>& synthesis_selected = scratch.synthesis_selected.renew(
        synthesis::kResidentSynthesisMaxUnits);
    DeviceArray<synthesis::ResidentSynthesisResult>& synthesis_result = scratch.synthesis_result.renew(1u);

    synthesis::ResidentSynthesisModelView<BigramKey, TrigramKey> model{};
    model.unit_lengths = state.unit_lengths.get();
    model.unit_content = state.unit_content.get();
    model.unit_vitality = state.unit_vitality.get();
    model.unit_count = state.unit_count;
    model.unit_words = kUnitWords;
    model.boundary_bytes = state.boundary_bytes.get();
    model.boundary_count = kBoundaryCount;
    model.closure_bytes = state.closure_bytes.get();
    model.closure_count = kClosureCount;
    model.cue_activation = synthesis_cue_activation.get();
    model.forward_activation = synthesis_forward_activation.get();
    model.backward_activation = synthesis_backward_activation.get();
    model.cue_masks = synthesis_cue_masks.get();
    model.cue_orders = route_orders.get();
    model.salient_masks = synthesis_salient_masks.get();
    model.forward_support = synthesis_forward_support.get();
    model.backward_support = synthesis_backward_support.get();
    model.policy_state = state.synthesis_policy.get();
    model.source_windows = synthesis_source_windows.get();
    model.source_window_count = source_window_count;
    model.unit_roles = synthesis_roles.get();
    model.role_count = roles::kStructuralRoleCount;
    model.base_role_bigrams = synthesis_base_role_bigrams.get();
    model.base_role_trigrams = synthesis_base_role_trigrams.get();
    model.online_role_bigrams = synthesis_online_role_bigrams.get();
    model.online_role_trigrams = synthesis_online_role_trigrams.get();
    model.base_bigrams = state.bigrams.get();
    model.base_bigram_counts = state.bigram_counts.get();
    model.base_bigram_count = state.bigram_count;
    model.base_trigrams = state.trigrams.get();
    model.base_trigram_counts = state.trigram_counts.get();
    model.base_trigram_count = state.trigram_count;
    model.online_bigrams = state.online_bigrams.get();
    model.online_bigram_counts = state.online_bigram_counts.get();
    model.online_bigram_count = state.online_bigram_count;
    model.online_trigrams = state.online_trigrams.get();
    model.online_trigram_counts = state.online_trigram_counts.get();
    model.online_trigram_count = state.online_trigram_count;
    model.subject_transitions = state.online_conditioned_transitions.get();
    model.subject_transition_counts =
        state.online_conditioned_transition_conductance.get();
    model.subject_transition_count =
        state.online_conditioned_transition_count;
    model.subject_anchors = learned_relation_subject_anchors.get();
    model.subject_anchor_count = kCueAnchorLimit;
    model.cue_units = learned_relation_cue_units.get();
    model.cue_unit_count = learned_relation_cue_count;
    model.episode_units = state.online_episode_units.get();
    model.episode_count = state.online_episode_count;
    model.episode_breaks = state.online_episode_breaks.get();
    model.episode_break_count = state.online_episode_break_count;

    synthesis::ResidentSynthesisWorkspaceView workspace{};
    workspace.reverse_bigrams = reverse_bigrams.get();
    workspace.reverse_bigram_capacity = reverse_bigram_count;
    workspace.reverse_trigrams = reverse_trigrams.get();
    workspace.reverse_trigram_capacity = reverse_trigram_count;
    workspace.seed_units = synthesis_seeds.get();
    workspace.seed_scores = synthesis_seed_scores.get();
    workspace.drafts = synthesis_drafts.get();
    workspace.draft_lengths = synthesis_draft_lengths.get();
    workspace.draft_scores = synthesis_draft_scores.get();
    workspace.selected_units = synthesis_selected.get();
    workspace.selected_capacity = synthesis::kResidentSynthesisMaxUnits;
    workspace.result = synthesis_result.get();

    synthesis::ResidentSynthesisConfig config{};
    config.max_units = kCompositionUnits;
    config.min_units = kCompositionMinUnits;
    config.repair_passes = 0u;
    cuda_require(synthesis::launch_resident_synthesis(model, workspace, config),
                 "launch resident bidirectional synthesis");
    extend_resident_synthesis_to_output_closure_kernel<<<1u, 1u>>>(
        model, synthesis_result.get(), synthesis_selected.get(),
        synthesis_output_closure.get());
    cuda_require(cudaGetLastError(), "launch resident synthesis closure extension");
    adopt_resident_synthesis_kernel<<<1u, kBlock>>>(
        synthesis_result.get(), synthesis_selected.get(),
        state.motor_context.get(), state.motor_completion.get());
    cuda_require(cudaGetLastError(), "launch resident synthesis adoption");

    // A conditioned one-unit value is causally useful but not yet a readable
    // utterance.  Let resident grammar surround that value.  The fallback
    // above remains intact when no source-novel frame closes.
    DeviceArray<std::uint32_t>& frame_relation_tail = scratch.frame_relation_tail.renew(
        synthesis::kResidentSynthesisMaxRelationTail);
    DeviceArray<std::uint32_t>& frame_relation_tail_count = scratch.frame_relation_tail_count.renew(1u);
    stage_answer_frame_selection_kernel<<<1u, 1u>>>(
        synthesis_result.get(), synthesis_selected.get(),
        state.answer_frame_selection.get(), frame_relation_tail.get(),
        frame_relation_tail_count.get());

    DeviceArray<std::uint32_t>& frame_unit_flags = scratch.frame_unit_flags.renew(state.unit_count);
    build_answer_frame_unit_flags_kernel<<<blocks_for(state.unit_count), kBlock>>>(
        state.unit_lengths.get(), state.unit_content.get(), state.unit_count,
        state.closure_bytes.get(), state.boundary_bytes.get(), kBoundaryCount,
        frame_unit_flags.get());

    constexpr std::uint32_t frame_candidate_capacity = 192u;
    DeviceArray<std::uint32_t>& frame_candidates = scratch.frame_candidates.renew(frame_candidate_capacity);
    DeviceArray<std::uint32_t>& frame_candidate_count_state = scratch.frame_candidate_count_state.renew(1u);
    build_answer_frame_candidate_set_kernel<<<1u, 1u>>>(
        state.answer_frame_selection.get(), frame_relation_tail.get(),
        frame_relation_tail_count.get(), state.bigrams.get(),
        state.bigram_counts.get(), state.bigram_count,
        state.online_bigrams.get(), state.online_bigram_counts.get(),
        state.online_bigram_count,
        state.unigram_top_ids.get(), kUnigramTop, frame_unit_flags.get(),
        state.unit_lengths.get(), state.unit_vitality.get(), state.unit_count,
        frame_candidates.get(),
        frame_candidate_capacity,
        frame_candidate_count_state.get());
    std::uint32_t frame_candidate_count = 0u;
    cuda_require(cudaMemcpy(&frame_candidate_count,
                            frame_candidate_count_state.get(),
                            sizeof(frame_candidate_count), cudaMemcpyDeviceToHost),
                 "read resident answer-frame candidate extent");

    const std::size_t frame_source_capacity =
        (state.unit_occurrences >= 4u ? state.unit_occurrences - 3u : 0u) +
        (state.online_episode_count >= 4u ? state.online_episode_count - 3u : 0u);
    DeviceArray<answer_frame::SourceWindowSignature>& frame_source_windows =
        scratch.frame_source_windows.renew(frame_source_capacity);
    DeviceArray<std::uint32_t>& frame_source_window_count = scratch.frame_source_window_count.renew(1u);
    cuda_require(cudaMemset(frame_source_window_count.get(), 0,
                            frame_source_window_count.bytes()),
                 "clear answer-frame source-window extent");
    if (state.unit_occurrences >= 4u) {
      build_answer_frame_source_window_signatures_kernel<<<
          blocks_for(state.unit_occurrences - 3u), kBlock>>>(
          state.base_episode_units.get(), state.unit_occurrences, nullptr, 0u,
          frame_source_windows.get(), frame_source_window_count.get());
    }
    if (state.online_episode_count >= 4u) {
      build_answer_frame_source_window_signatures_kernel<<<
          blocks_for(state.online_episode_count - 3u), kBlock>>>(
          state.online_episode_units.get(), state.online_episode_count,
          state.online_episode_breaks.get(), state.online_episode_break_count,
          frame_source_windows.get(), frame_source_window_count.get());
    }
    std::uint32_t frame_source_count = 0u;
    cuda_require(cudaMemcpy(&frame_source_count, frame_source_window_count.get(),
                            sizeof(frame_source_count), cudaMemcpyDeviceToHost),
                 "read answer-frame source-window extent");
    if (frame_source_count != 0u) {
      thrust::sort(thrust::device, frame_source_windows.get(),
                   frame_source_windows.get() + frame_source_count);
    }

    DeviceArray<std::uint64_t>& frame_role_bigrams = scratch.frame_role_bigrams.renew(roles::kRoleBigramTableSize);
    DeviceArray<std::uint64_t>& frame_role_trigrams = scratch.frame_role_trigrams.renew(roles::kRoleTrigramTableSize);
    sum_answer_frame_role_evidence_kernel<<<
        blocks_for(static_cast<std::uint32_t>(roles::kRoleBigramTableSize)), kBlock>>>(
        synthesis_base_role_bigrams.get(), synthesis_online_role_bigrams.get(),
        frame_role_bigrams.get(), roles::kRoleBigramTableSize);
    sum_answer_frame_role_evidence_kernel<<<
        blocks_for(static_cast<std::uint32_t>(roles::kRoleTrigramTableSize)), kBlock>>>(
        synthesis_base_role_trigrams.get(), synthesis_online_role_trigrams.get(),
        frame_role_trigrams.get(), roles::kRoleTrigramTableSize);

    constexpr std::uint32_t frame_candidates_to_try = 192u;
    DeviceArray<std::uint32_t>& frame_drafts = scratch.frame_drafts.renew(
        frame_candidates_to_try * answer_frame::kMaxFrameUnits);
    DeviceArray<std::uint32_t>& frame_draft_lengths = scratch.frame_draft_lengths.renew(frame_candidates_to_try);
    DeviceArray<unsigned long long>& frame_draft_scores = scratch.frame_draft_scores.renew(frame_candidates_to_try);
    DeviceArray<std::uint32_t>& frame_output_units = scratch.frame_output_units.renew(answer_frame::kMaxFrameUnits);
    DeviceArray<std::uint8_t>& frame_output_bytes = scratch.frame_output_bytes.renew(1024u);
    DeviceArray<answer_frame::Result>& frame_result = scratch.frame_result.renew(1u);

    answer_frame::BasicMutableModelView<BigramKey, TrigramKey> frame_model{};
    frame_model.unit_lengths = state.unit_lengths.get();
    frame_model.unit_content = state.unit_content.get();
    frame_model.unit_words = kUnitWords;
    frame_model.unit_vitality = state.unit_vitality.get();
    frame_model.unit_activation = nullptr;
    frame_model.unit_flags = frame_unit_flags.get();
    frame_model.unit_roles = synthesis_roles.get();
    frame_model.unit_count = state.unit_count;
    frame_model.candidate_units = frame_candidates.get();
    frame_model.candidate_unit_count = frame_candidate_count;
    frame_model.relation_tail = frame_relation_tail.get();
    frame_model.relation_tail_count = frame_relation_tail_count.get();
    frame_model.relation_tail_capacity =
        synthesis::kResidentSynthesisMaxRelationTail;
    frame_model.bigrams = state.bigrams.get();
    frame_model.bigram_counts = state.bigram_counts.get();
    frame_model.bigram_count = state.bigram_count;
    frame_model.trigrams = state.trigrams.get();
    frame_model.trigram_counts = state.trigram_counts.get();
    frame_model.trigram_count = state.trigram_count;
    frame_model.online_bigrams = state.online_bigrams.get();
    frame_model.online_bigram_counts = state.online_bigram_counts.get();
    frame_model.online_bigram_count = state.online_bigram_count;
    frame_model.online_trigrams = state.online_trigrams.get();
    frame_model.online_trigram_counts = state.online_trigram_counts.get();
    frame_model.online_trigram_count = state.online_trigram_count;
    frame_model.role_bigrams = frame_role_bigrams.get();
    frame_model.role_trigrams = frame_role_trigrams.get();
    frame_model.role_count = roles::kStructuralRoleCount;
    frame_model.source_windows = frame_source_windows.get();
    frame_model.source_window_count = frame_source_count;
    frame_model.policy = state.answer_frame_policy.get();

    answer_frame::MutableWorkspaceView frame_workspace{};
    frame_workspace.drafts = frame_drafts.get();
    frame_workspace.draft_lengths = frame_draft_lengths.get();
    frame_workspace.draft_scores = frame_draft_scores.get();
    frame_workspace.candidate_capacity = frame_candidates_to_try;
    frame_workspace.output_units = frame_output_units.get();
    frame_workspace.output_unit_capacity = answer_frame::kMaxFrameUnits;
    frame_workspace.output_bytes = frame_output_bytes.get();
    frame_workspace.output_byte_capacity =
        static_cast<std::uint32_t>(frame_output_bytes.size());
    cuda_require(answer_frame::launch_resident_answer_frame(
                     frame_model, state.answer_frame_selection.get(),
                     frame_workspace, frame_result.get()),
                 "launch resident factual answer frame");
    adopt_answer_frame_kernel<<<1u, kBlock>>>(
        frame_result.get(), frame_output_units.get(), state.motor_context.get(),
        state.motor_completion.get());
    cuda_require(cudaGetLastError(), "launch resident answer-frame adoption");

    DeviceArray<std::uint32_t>& role_boundary_evidence = scratch.role_boundary_evidence.renew(state.unit_count);
    DeviceArray<std::uint32_t>& role_closure_evidence = scratch.role_closure_evidence.renew(state.unit_count);
    DeviceArray<unsigned long long>& role_cue_activation = scratch.role_cue_activation.renew(state.unit_count);
    DeviceArray<std::uint32_t>& role_cue_groups = scratch.role_cue_groups.renew(state.unit_count);
    build_role_compositor_evidence_kernel<<<blocks_for(state.unit_count), kBlock>>>(
        frame_unit_flags.get(), synthesis_roles.get(), state.unit_count,
        role_boundary_evidence.get(), role_closure_evidence.get());
    seed_role_compositor_activation_kernel<<<blocks_for(state.unit_count), kBlock>>>(
        state.unit_lengths.get(), state.unit_content.get(),
        state.unit_vitality.get(), state.unit_count,
        state.boundary_bytes.get(),
        device_bytes.get(), byte_count, role_cue_activation.get(),
        role_cue_groups.get());
    cuda_require(cudaGetLastError(), "build resident role-compositor cue matter");

    DeviceArray<role_compositor::SubjectConditionedRelationTrace>& relation_traces =
        scratch.relation_traces.renew(kResidentRelationTraceCapacity);
    DeviceArray<std::uint32_t>& relation_trace_units =
        scratch.relation_trace_units.renew(kResidentRelationTraceUnitCapacity);
    DeviceArray<std::uint32_t>& relation_trace_count = scratch.relation_trace_count.renew(1u);
    DeviceArray<std::uint32_t>& relation_trace_unit_count = scratch.relation_trace_unit_count.renew(1u);
    build_conditioned_role_traces_kernel<<<1u, 1u>>>(
        state.online_episode_units.get(), state.online_episode_count,
        state.online_episode_breaks.get(), state.online_episode_break_count,
        state.unit_lengths.get(), state.unit_content.get(), state.closure_bytes.get(),
        role_cue_activation.get(), role_cue_groups.get(), relation_traces.get(),
        kResidentRelationTraceCapacity, relation_trace_count.get(),
        relation_trace_units.get(), kResidentRelationTraceUnitCapacity,
        relation_trace_unit_count.get());
    cuda_require(cudaGetLastError(), "build resident episode-preserving relation traces");
    std::uint32_t host_relation_trace_count = 0u;
    std::uint32_t host_relation_trace_unit_count = 0u;
    cuda_require(cudaMemcpy(&host_relation_trace_count, relation_trace_count.get(),
                            sizeof(host_relation_trace_count), cudaMemcpyDeviceToHost),
                 "read resident relation trace extent");
    cuda_require(cudaMemcpy(&host_relation_trace_unit_count,
                            relation_trace_unit_count.get(),
                            sizeof(host_relation_trace_unit_count),
                            cudaMemcpyDeviceToHost),
                 "read resident relation unit extent");

    DeviceArray<role_compositor::MutableRoleCompositorPolicy>& role_policy = scratch.role_policy.renew(1u);
    role_compositor::MutableRoleCompositorPolicy role_warm_start{};
    role_warm_start.min_plan_units = kCompositionMinUnits;
    role_warm_start.max_plan_units = 16u;
    role_warm_start.subject_cue_weight = 256u;
    role_warm_start.relation_cue_weight = 256u;
    role_warm_start.relation_evidence_weight = 8u;
    role_warm_start.activation_weight = 256u;
    cuda_require(cudaMemcpy(role_policy.get(), &role_warm_start,
                            sizeof(role_warm_start), cudaMemcpyHostToDevice),
                 "initialize resident role-compositor warm start");

    DeviceArray<unsigned long long>& role_trace_scores = scratch.role_trace_scores.renew(
        max(1u, host_relation_trace_count));
    DeviceArray<std::uint32_t>& role_required_counts = scratch.role_required_counts.renew(state.unit_count);
    DeviceArray<std::uint32_t>& role_selected_counts = scratch.role_selected_counts.renew(state.unit_count);
    DeviceArray<role_compositor::RoleCompositorChoice>& role_choice = scratch.role_choice.renew(1u);
    DeviceArray<std::uint32_t>& role_ordered_units = scratch.role_ordered_units.renew(kCompositionUnits);
    DeviceArray<std::uint8_t>& role_output_bytes = scratch.role_output_bytes.renew(2048u);
    DeviceArray<role_compositor::RoleCompositorResult>& role_result = scratch.role_result.renew(1u);

    role_compositor::RoleCompositorModelView role_model{};
    role_model.unit_lengths = state.unit_lengths.get();
    role_model.unit_content = state.unit_content.get();
    role_model.unit_words = kUnitWords;
    role_model.unit_vitality = state.unit_vitality.get();
    role_model.unit_count = state.unit_count;
    role_model.cue_activation = role_cue_activation.get();
    role_model.cue_groups = role_cue_groups.get();
    role_model.boundary_evidence = role_boundary_evidence.get();
    role_model.closure_evidence = role_closure_evidence.get();
    role_model.unit_roles = synthesis_roles.get();
    role_model.role_evidence = synthesis_role_tables;
    role_model.relation_traces = relation_traces.get();
    role_model.relation_trace_count = host_relation_trace_count;
    role_model.relation_units = relation_trace_units.get();
    role_model.relation_unit_count = host_relation_trace_unit_count;
    role_model.policy = role_policy.get();

    role_compositor::RoleCompositorWorkspaceView role_workspace{};
    role_workspace.trace_scores = role_trace_scores.get();
    role_workspace.trace_capacity = max(1u, host_relation_trace_count);
    role_workspace.required_counts = role_required_counts.get();
    role_workspace.selected_counts = role_selected_counts.get();
    role_workspace.unit_capacity = state.unit_count;
    role_workspace.choice = role_choice.get();
    role_workspace.ordered_units = role_ordered_units.get();
    role_workspace.ordered_unit_capacity = kCompositionUnits;
    role_workspace.output_units = state.motor_completion.get();
    role_workspace.output_unit_capacity = kCompositionUnits;
    role_workspace.output_bytes = role_output_bytes.get();
    role_workspace.output_byte_capacity =
        static_cast<std::uint32_t>(role_output_bytes.size());
    if (host_relation_trace_count != 0u && host_relation_trace_unit_count != 0u) {
      cuda_require(role_compositor::compose_resident_role_plan_cuda(
                       role_model, role_workspace, role_result.get()),
                   "compose resident episode-preserving role plan");
      activate_selected_relation_trace_kernel<<<1u, 1u>>>(
          role_choice.get(), relation_traces.get(), relation_trace_units.get(),
          state.unit_lengths.get(), state.unit_vitality.get(),
          role_cue_activation.get(),
          state.subject_ids.get(), state.subject_weights.get(),
          state.subject_count.get());
      seed_motor_from_selected_relation_trace_kernel<<<1u, 1u>>>(
          role_choice.get(), relation_traces.get(), relation_trace_units.get(),
          role_cue_activation.get(), state.unit_lengths.get(),
          state.unit_vitality.get(), state.motor_context.get(),
          state.motor_completion.get());
      cuda_require(cudaGetLastError(), "activate resident episode-preserving relation");
    } else {
      cuda_require(cudaMemset(role_result.get(), 0, role_result.bytes()),
                   "clear unavailable resident role plan");
    }
    cuda_require(cudaDeviceSynchronize(), "complete resident synthesis conditioning");
    if (diagnostic) {
      role_compositor::RoleCompositorResult host_role{};
      role_compositor::RoleCompositorChoice host_role_choice{};
      cuda_require(cudaMemcpy(&host_role, role_result.get(), sizeof(host_role),
                              cudaMemcpyDeviceToHost),
                   "read resident role-plan diagnostic");
      cuda_require(cudaMemcpy(&host_role_choice, role_choice.get(),
                              sizeof(host_role_choice), cudaMemcpyDeviceToHost),
                   "read resident role-choice diagnostic");
      std::fprintf(stderr,
                   "role_trace_compositor traces=%u units=%u ready=%u output=%u "
                   "cue_hits=%u evidence=%u selected=%u required=%u tied=%u "
                   "choice_hits=%u score=%llu\n",
                   host_relation_trace_count, host_relation_trace_unit_count,
                   host_role.ready, host_role.output_unit_count,
                   host_role.relation_cue_hits, host_role.relation_evidence,
                   host_role_choice.trace, host_role_choice.required_units,
                   host_role_choice.tied_or_invalid, host_role_choice.cue_hits,
                   static_cast<unsigned long long>(host_role_choice.score));
      if (host_role_choice.trace < host_relation_trace_count) {
        role_compositor::SubjectConditionedRelationTrace host_trace{};
        cuda_require(cudaMemcpy(&host_trace,
                                relation_traces.get() + host_role_choice.trace,
                                sizeof(host_trace), cudaMemcpyDeviceToHost),
                     "read selected resident relation trace");
        std::vector<std::uint32_t> host_trace_units(host_trace.unit_count);
        cuda_require(cudaMemcpy(host_trace_units.data(),
                                relation_trace_units.get() + host_trace.unit_begin,
                                host_trace_units.size() * sizeof(std::uint32_t),
                                cudaMemcpyDeviceToHost),
                     "read selected resident relation units");
        std::vector<std::uint32_t> host_lengths(state.unit_count);
        std::vector<std::uint32_t> host_content(
            static_cast<std::size_t>(state.unit_count) * kUnitWords);
        cuda_require(cudaMemcpy(host_lengths.data(), state.unit_lengths.get(),
                                host_lengths.size() * sizeof(std::uint32_t),
                                cudaMemcpyDeviceToHost),
                     "read selected trace unit lengths");
        cuda_require(cudaMemcpy(host_content.data(), state.unit_content.get(),
                                host_content.size() * sizeof(std::uint32_t),
                                cudaMemcpyDeviceToHost),
                     "read selected trace unit content");
        std::fprintf(stderr, "role_trace_episode=\"");
        for (const std::uint32_t unit : host_trace_units) {
          for (std::uint32_t offset = 0u; offset < host_lengths[unit]; ++offset) {
            const std::uint32_t packed = host_content[
                static_cast<std::size_t>(unit) * kUnitWords + offset / 4u];
            std::fputc(static_cast<int>((packed >> (8u * (offset & 3u))) & 0xffu),
                       stderr);
          }
        }
        std::fprintf(stderr, "\"\n");
      }
      std::vector<std::uint32_t> host_role_groups(state.unit_count);
      cuda_require(cudaMemcpy(host_role_groups.data(), role_cue_groups.get(),
                              host_role_groups.size() * sizeof(std::uint32_t),
                              cudaMemcpyDeviceToHost),
                   "read resident cue-group diagnostic");
      std::uint32_t host_role_group_counts[kCueAnchorLimit]{};
      for (const std::uint32_t group : host_role_groups)
        if (group < kCueAnchorLimit) ++host_role_group_counts[group];
      std::fprintf(stderr, "role_cue_group_counts=");
      for (std::uint32_t group = 0u; group < kCueAnchorLimit; ++group)
        if (host_role_group_counts[group] != 0u)
          std::fprintf(stderr, " [%u:%u]", group, host_role_group_counts[group]);
      std::fprintf(stderr, "\n");
      if (host_relation_trace_count != 0u) {
        std::vector<role_compositor::SubjectConditionedRelationTrace> host_traces(
            host_relation_trace_count);
        std::vector<std::uint32_t> host_relation_units(
            host_relation_trace_unit_count);
        std::vector<unsigned long long> host_trace_scores(
            host_relation_trace_count);
        cuda_require(cudaMemcpy(host_traces.data(), relation_traces.get(),
                                host_traces.size() * sizeof(host_traces[0]),
                                cudaMemcpyDeviceToHost),
                     "read resident trace table diagnostic");
        cuda_require(cudaMemcpy(host_relation_units.data(), relation_trace_units.get(),
                                host_relation_units.size() * sizeof(std::uint32_t),
                                cudaMemcpyDeviceToHost),
                     "read resident trace units diagnostic");
        cuda_require(cudaMemcpy(host_trace_scores.data(), role_trace_scores.get(),
                                host_trace_scores.size() * sizeof(unsigned long long),
                                cudaMemcpyDeviceToHost),
                     "read resident trace scores diagnostic");
        std::vector<std::uint32_t> ranked(host_relation_trace_count);
        std::iota(ranked.begin(), ranked.end(), 0u);
        const std::size_t ranked_count = std::min<std::size_t>(8u, ranked.size());
        std::partial_sort(ranked.begin(), ranked.begin() + ranked_count,
                          ranked.end(), [&](std::uint32_t left, std::uint32_t right) {
                            return host_trace_scores[left] > host_trace_scores[right];
                          });
        std::fprintf(stderr, "role_trace_ranked_masks=");
        for (std::size_t rank = 0u; rank < ranked_count; ++rank) {
          const std::uint32_t trace_index = ranked[rank];
          const auto& trace = host_traces[trace_index];
          std::uint32_t mask = 0u;
          for (std::uint32_t offset = 0u; offset < trace.unit_count; ++offset) {
            const std::uint32_t unit =
                host_relation_units[trace.unit_begin + offset];
            if (host_role_groups[unit] < 32u)
              mask |= 1u << host_role_groups[unit];
          }
          std::fprintf(stderr, " [%u:%llx:m%08x]", trace_index,
                       static_cast<unsigned long long>(host_trace_scores[trace_index]),
                       mask);
        }
        std::fprintf(stderr, "\n");
      }
      if (host_role.ready != 0u && host_role.output_byte_count != 0u) {
        std::vector<std::uint8_t> host_role_bytes(host_role.output_byte_count);
        cuda_require(cudaMemcpy(host_role_bytes.data(), role_output_bytes.get(),
                                host_role_bytes.size(), cudaMemcpyDeviceToHost),
                     "read resident role-plan bytes diagnostic");
        std::fprintf(stderr, "role_trace_surface=\"");
        for (const std::uint8_t byte : host_role_bytes)
          std::fputc(static_cast<int>(byte), stderr);
        std::fprintf(stderr, "\"\n");
      }
      answer_frame::Result host_frame{};
      answer_frame::MutableSelectionState host_selection{};
      std::uint32_t host_tail_count = 0u;
      std::uint32_t host_tail[synthesis::kResidentSynthesisMaxRelationTail]{};
      cuda_require(cudaMemcpy(&host_frame, frame_result.get(), sizeof(host_frame),
                              cudaMemcpyDeviceToHost),
                   "read answer-frame diagnostic");
      cuda_require(cudaMemcpy(&host_selection, state.answer_frame_selection.get(),
                              sizeof(host_selection), cudaMemcpyDeviceToHost),
                   "read answer-frame selection diagnostic");
      cuda_require(cudaMemcpy(&host_tail_count, frame_relation_tail_count.get(),
                              sizeof(host_tail_count), cudaMemcpyDeviceToHost),
                   "read answer-frame tail extent diagnostic");
      cuda_require(cudaMemcpy(host_tail, frame_relation_tail.get(),
                              sizeof(host_tail), cudaMemcpyDeviceToHost),
                   "read answer-frame tail diagnostic");
      std::vector<std::uint32_t> host_candidates(frame_candidate_count);
      std::vector<std::uint32_t> host_flags(state.unit_count);
      std::vector<std::uint32_t> host_unit_lengths(state.unit_count);
      std::vector<std::uint32_t> host_unit_content(
          static_cast<std::size_t>(state.unit_count) * kUnitWords);
      std::vector<std::uint32_t> host_unit_vitality(state.unit_count);
      std::vector<roles::MutableStructuralRole> host_roles(state.unit_count);
      std::vector<std::uint64_t> host_role_bigrams(roles::kRoleBigramTableSize);
      std::vector<BigramKey> host_base_bigrams(state.bigram_count);
      std::vector<std::uint32_t> host_base_bigram_counts(state.bigram_count);
      std::vector<BigramKey> host_online_bigrams(state.online_bigram_count);
      std::vector<std::uint32_t> host_online_bigram_counts(
          state.online_bigram_count);
      cuda_require(cudaMemcpy(host_candidates.data(), frame_candidates.get(),
                              static_cast<std::size_t>(frame_candidate_count) *
                                  sizeof(std::uint32_t),
                              cudaMemcpyDeviceToHost),
                   "read answer-frame candidate diagnostic");
      cuda_require(cudaMemcpy(host_roles.data(), synthesis_roles.get(),
                              synthesis_roles.bytes(), cudaMemcpyDeviceToHost),
                   "read answer-frame role diagnostic");
      cuda_require(cudaMemcpy(host_flags.data(), frame_unit_flags.get(),
                              frame_unit_flags.bytes(), cudaMemcpyDeviceToHost),
                   "read answer-frame flag diagnostic");
      cuda_require(cudaMemcpy(host_unit_lengths.data(), state.unit_lengths.get(),
                              state.unit_count * sizeof(std::uint32_t),
                              cudaMemcpyDeviceToHost),
                   "read answer-frame unit-length diagnostic");
      cuda_require(cudaMemcpy(host_unit_content.data(), state.unit_content.get(),
                              host_unit_content.size() * sizeof(std::uint32_t),
                              cudaMemcpyDeviceToHost),
                   "read answer-frame unit-content diagnostic");
      cuda_require(cudaMemcpy(host_unit_vitality.data(), state.unit_vitality.get(),
                              state.unit_count * sizeof(std::uint32_t),
                              cudaMemcpyDeviceToHost),
                   "read answer-frame vitality diagnostic");
      cuda_require(cudaMemcpy(host_role_bigrams.data(), frame_role_bigrams.get(),
                              frame_role_bigrams.bytes(), cudaMemcpyDeviceToHost),
                   "read answer-frame role-edge diagnostic");
      cuda_require(cudaMemcpy(host_base_bigrams.data(), state.bigrams.get(),
                              state.bigram_count * sizeof(BigramKey),
                              cudaMemcpyDeviceToHost),
                   "read answer-frame base-bigram diagnostic");
      cuda_require(cudaMemcpy(host_base_bigram_counts.data(),
                              state.bigram_counts.get(),
                              state.bigram_count * sizeof(std::uint32_t),
                              cudaMemcpyDeviceToHost),
                   "read answer-frame base-count diagnostic");
      cuda_require(cudaMemcpy(host_online_bigrams.data(), state.online_bigrams.get(),
                              state.online_bigram_count * sizeof(BigramKey),
                              cudaMemcpyDeviceToHost),
                   "read answer-frame online-bigram diagnostic");
      cuda_require(cudaMemcpy(host_online_bigram_counts.data(),
                              state.online_bigram_counts.get(),
                              state.online_bigram_count * sizeof(std::uint32_t),
                              cudaMemcpyDeviceToHost),
                   "read answer-frame online-count diagnostic");
      std::uint32_t backward_edges = 0u;
      std::uint32_t forward_edges = 0u;
      std::uint32_t closure_candidates = 0u;
      if (host_selection.value < host_roles.size()) {
        const std::uint32_t value_role = host_roles[host_selection.value].role;
        for (const std::uint32_t candidate : host_candidates) {
          if (candidate >= host_roles.size()) continue;
          const std::uint32_t candidate_role = host_roles[candidate].role;
          closure_candidates +=
              (host_flags[candidate] & answer_frame::kUnitClosure) != 0u;
          backward_edges += host_role_bigrams[
              static_cast<std::size_t>(candidate_role) * roles::kStructuralRoleCount +
              value_role] != 0u;
          forward_edges += host_role_bigrams[
              static_cast<std::size_t>(value_role) * roles::kStructuralRoleCount +
              candidate_role] != 0u;
        }
        std::fprintf(stderr,
                     "answer_frame_selection value=%u subject=%u predicate=%u role=%u candidates=%u backward_edges=%u forward_edges=%u closures=%u tail=%u [%u,%u,%u,%u]\n",
                     host_selection.value, host_selection.subject,
                     host_selection.predicate, value_role, frame_candidate_count,
                     backward_edges, forward_edges, closure_candidates,
                     host_tail_count, host_tail[0], host_tail[1], host_tail[2],
                     host_tail[3]);

        const auto host_bigram_count = [](const std::vector<BigramKey>& keys,
                                           const std::vector<std::uint32_t>& counts,
                                           std::uint32_t previous,
                                           std::uint32_t next) {
          const BigramKey wanted{previous, next};
          const auto found = std::lower_bound(keys.begin(), keys.end(), wanted);
          return found != keys.end() && *found == wanted
              ? counts[static_cast<std::size_t>(found - keys.begin())] : 0u;
        };
        const auto unit_text = [&](std::uint32_t unit) {
          std::string text;
          if (unit >= host_unit_lengths.size()) return text;
          for (std::uint32_t offset = 0u; offset < host_unit_lengths[unit]; ++offset) {
            const std::uint32_t word = host_unit_content[
                static_cast<std::size_t>(unit) * kUnitWords + offset / 4u];
            const std::uint8_t byte = static_cast<std::uint8_t>(
                (word >> (8u * (offset & 3u))) & 0xffu);
            text.push_back(byte >= 0x20u && byte < 0x7fu
                               ? static_cast<char>(byte) : '?');
          }
          return text;
        };
        std::uint32_t host_anchor_ids[kCueAnchorLimit]{};
        std::vector<std::uint32_t> host_cue_focus(state.unit_count);
        cuda_require(cudaMemcpy(host_anchor_ids, selected_anchor_ids.get(),
                                sizeof(host_anchor_ids), cudaMemcpyDeviceToHost),
                     "read role-compositor anchors diagnostic");
        cuda_require(cudaMemcpy(host_cue_focus.data(), cue_focus.get(),
                                host_cue_focus.size() * sizeof(std::uint32_t),
                                cudaMemcpyDeviceToHost),
                     "read role-compositor cue activation diagnostic");
        std::fprintf(stderr, "role_cue_anchors=");
        for (std::uint32_t rank = 0u; rank < kCueAnchorLimit; ++rank) {
          const std::uint32_t unit = host_anchor_ids[rank];
          if (unit >= state.unit_count || host_cue_focus[unit] == 0u) continue;
          std::fprintf(stderr, " [%u:%u:v%u:'%s']", rank, host_cue_focus[unit],
                       host_unit_vitality[unit], unit_text(unit).c_str());
        }
        std::fprintf(stderr, "\n");
        if (host_tail_count != 0u && host_tail[0] < host_roles.size() &&
            host_selection.subject < host_roles.size() &&
            host_selection.predicate < host_roles.size()) {
          const std::uint32_t subject_role =
              host_roles[host_selection.subject].role;
          const std::uint32_t predicate_role =
              host_roles[host_selection.predicate].role;
          const std::uint32_t tail_role = host_roles[host_tail[0]].role;
          for (std::uint32_t index = 0u;
               index < std::min<std::uint32_t>(frame_candidate_count, 64u); ++index) {
            const std::uint32_t unit = host_candidates[index];
            if (unit >= host_roles.size()) continue;
            const std::uint32_t literal =
                host_bigram_count(host_base_bigrams, host_base_bigram_counts,
                                  unit, host_tail[0]) +
                host_bigram_count(host_online_bigrams, host_online_bigram_counts,
                                  unit, host_tail[0]);
            if (literal == 0u) continue;
            const std::uint32_t connector_role = host_roles[unit].role;
            const std::uint64_t subject_predicate_count = host_role_bigrams[
                static_cast<std::size_t>(subject_role) * roles::kStructuralRoleCount +
                predicate_role];
            const std::uint64_t predicate_connector_count = host_role_bigrams[
                static_cast<std::size_t>(predicate_role) * roles::kStructuralRoleCount +
                connector_role];
            const std::uint64_t connector_tail_count = host_role_bigrams[
                static_cast<std::size_t>(connector_role) * roles::kStructuralRoleCount +
                tail_role];
            std::fprintf(stderr,
                         "answer_frame_connector rank=%u id=%u bytes=\"%s\" vitality=%u literal_tail=%u roles=%u/%u/%u role_counts=%llu/%llu/%llu\n",
                         index, unit, unit_text(unit).c_str(),
                         host_unit_vitality[unit], literal, subject_role,
                         predicate_role, connector_role,
                         static_cast<unsigned long long>(subject_predicate_count),
                         static_cast<unsigned long long>(predicate_connector_count),
                         static_cast<unsigned long long>(connector_tail_count));
          }
        }
      }
      std::fprintf(stderr,
                   "answer_frame ready=%u units=%u value_position=%u closed=%u source_novel=%u windows=%u\n",
                   host_frame.ready, host_frame.unit_count,
                   host_frame.value_position, host_frame.closed,
                   host_frame.source_novel, frame_source_count);
    }
    return;
  }
  cuda_require(cudaMemcpy(alternate_motor_context.get(), state.motor_context.get(),
                          alternate_motor_context.bytes(), cudaMemcpyDeviceToDevice),
               "seed alternate resident cue plan");
  cuda_require(cudaMemset(alternate_motor_completion.get(), 0,
                          alternate_motor_completion.bytes()),
               "clear alternate resident completion");
  score_local_provenance_seeds_kernel<<<blocks_for(scoped_episode_count), kBlock>>>(
      state.unit_vitality.get(), state.unit_lengths.get(), state.unit_content.get(),
      state.closure_bytes.get(), cue_count,
      episode_match_mask.get(), episode_exact_match_mask.get(), episode_match_spans.get(),
      state.online_bigrams.get(), state.online_bigram_counts.get(),
      state.online_bigram_count, state.online_trigrams.get(),
      state.online_trigram_counts.get(), state.online_trigram_count,
      state.online_episode_units.get() + scoped_episode_begin, scoped_episode_count,
      scoped_episode_breaks.get(), scoped_break_count,
      matched_segment_counts.get(),
      local_seed_candidates.get(), state.motor_context.get(),
      state.novelty_epoch_active ? 1u : 0u, 0u);
  score_local_provenance_seeds_kernel<<<blocks_for(scoped_episode_count), kBlock>>>(
      state.unit_vitality.get(), state.unit_lengths.get(), state.unit_content.get(),
      state.closure_bytes.get(), cue_count,
      episode_match_mask.get(), episode_exact_match_mask.get(), episode_match_spans.get(),
      state.online_bigrams.get(), state.online_bigram_counts.get(),
      state.online_bigram_count, state.online_trigrams.get(),
      state.online_trigram_counts.get(), state.online_trigram_count,
      state.online_episode_units.get() + scoped_episode_begin, scoped_episode_count,
      scoped_episode_breaks.get(), scoped_break_count, matched_segment_counts.get(),
      clause_seed_candidates.get(), alternate_motor_context.get(),
      state.novelty_epoch_active ? 1u : 0u, 1u);
  if (scoped_episode_begin != 0u) {
    offset_local_seed_candidates_kernel<<<blocks_for(scoped_episode_count), kBlock>>>(
        local_seed_candidates.get(), scoped_episode_count, scoped_episode_begin);
    offset_local_seed_candidates_kernel<<<blocks_for(scoped_episode_count), kBlock>>>(
        clause_seed_candidates.get(), scoped_episode_count, scoped_episode_begin);
  }
  thrust::sort(thrust::device, local_seed_candidates.get(),
               local_seed_candidates.get() + scoped_episode_count,
               LocalSeedRank{state.novelty_epoch_active ? 1u : 0u});
  thrust::sort(thrust::device, clause_seed_candidates.get(),
               clause_seed_candidates.get() + scoped_episode_count,
               LocalSeedRank{state.novelty_epoch_active ? 1u : 0u});
  if (diagnostic) {
    print_local_provenance_seed_kernel<<<1u, 1u>>>(
        local_seed_candidates.get(), episode_match_mask.get(),
        episode_match_spans.get(), state.online_episode_units.get(),
        state.unit_lengths.get(), state.unit_content.get(),
        state.unit_vitality.get(), scoped_episode_count, cue_count, scoped_episode_begin);
    print_local_provenance_seed_kernel<<<1u, 1u>>>(
        clause_seed_candidates.get(), episode_match_mask.get(),
        episode_match_spans.get(), state.online_episode_units.get(),
        state.unit_lengths.get(), state.unit_content.get(),
        state.unit_vitality.get(), scoped_episode_count, cue_count, scoped_episode_begin);
  }
  compose_resident_relation_kernel<<<1u, kBlock>>>(
      state.unit_vitality.get(), state.unit_lengths.get(), state.unit_content.get(),
      state.unit_count, state.boundary_histogram.get(), state.boundary_pairs.get(),
      state.boundary_bytes.get(), state.closure_bytes.get(), device_bytes.get(),
      byte_count, cue_count, cue_scores.get(),
      route_orders.get(),
      cue_focus.get(), cue_masks.get(), salient_masks.get(),
      forward_scores.get(), backward_scores.get(), forward_support.get(),
      backward_support.get(), forward_salient_support.get(), strongest_evidence.get(),
      second_evidence.get(),
      state.bigrams.get(), state.bigram_counts.get(),
      state.bigram_count, state.trigrams.get(), state.trigram_counts.get(),
      state.trigram_count, state.online_bigrams.get(), state.online_bigram_counts.get(),
      state.online_bigram_count, state.online_trigrams.get(),
      state.online_trigram_counts.get(), state.online_trigram_count,
      state.online_associations.get(), state.online_association_counts.get(),
      state.online_association_count,
      association_mass.get(),
      state.base_episode_units.get(), state.unit_occurrences,
      state.online_episode_units.get(), state.online_episode_count,
      state.online_episode_breaks.get(), state.online_episode_break_count,
      episode_match_mask.get(), scoped_episode_begin, scoped_episode_count,
      state.novelty_epoch_active ? 1u : 0u, kCompositionSourceRunLimit,
      local_seed_candidates.get(), scoped_episode_count,
      admitted_segment_mask.get(),
      state.motor_context.get(), state.motor_completion.get());
  compose_resident_relation_kernel<<<1u, kBlock>>>(
      state.unit_vitality.get(), state.unit_lengths.get(), state.unit_content.get(),
      state.unit_count, state.boundary_histogram.get(), state.boundary_pairs.get(),
      state.boundary_bytes.get(), state.closure_bytes.get(), device_bytes.get(),
      byte_count, cue_count, cue_scores.get(), route_orders.get(), cue_focus.get(),
      cue_masks.get(), salient_masks.get(), forward_scores.get(), backward_scores.get(),
      forward_support.get(), backward_support.get(), forward_salient_support.get(),
      strongest_evidence.get(), second_evidence.get(), state.bigrams.get(),
      state.bigram_counts.get(), state.bigram_count, state.trigrams.get(),
      state.trigram_counts.get(), state.trigram_count, state.online_bigrams.get(),
      state.online_bigram_counts.get(), state.online_bigram_count,
      state.online_trigrams.get(), state.online_trigram_counts.get(),
      state.online_trigram_count, state.online_associations.get(),
      state.online_association_counts.get(), state.online_association_count,
      association_mass.get(), state.base_episode_units.get(), state.unit_occurrences,
      state.online_episode_units.get(), state.online_episode_count,
      state.online_episode_breaks.get(), state.online_episode_break_count,
      episode_match_mask.get(), scoped_episode_begin, scoped_episode_count,
      state.novelty_epoch_active ? 1u : 0u, kCompositionSourceRunLimit - 2u,
      clause_seed_candidates.get(),
      scoped_episode_count, admitted_segment_mask.get(), alternate_motor_context.get(),
      alternate_motor_completion.get());
  select_resident_completion_kernel<<<1u, 1u>>>(
      state.unit_lengths.get(), state.unit_content.get(), cue_masks.get(),
      salient_masks.get(), state.closure_bytes.get(), state.boundary_bytes.get(),
      device_bytes.get(), byte_count, kCompositionSourceRunLimit,
      kCompositionSourceRunLimit - 2u,
      state.motor_context.get(), state.motor_completion.get(),
      alternate_motor_context.get(), alternate_motor_completion.get());
  cuda_require(cudaGetLastError(), "launch raw cue conditioning");
  cuda_require(cudaDeviceSynchronize(), "complete raw cue conditioning");
}
