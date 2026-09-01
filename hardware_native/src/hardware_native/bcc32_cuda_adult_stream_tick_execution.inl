// bcc32_cuda_adult_stream_tick_execution.inl
//
// Extracted from bcc32_cuda_adult_stream_v1.cuh (0X1-monolith-ratchet slice,
// 2026-08-15) as a second cohesive causal responsibility, distinct from the
// checkpoint save/load tail already extracted into
// bcc32_cuda_adult_stream_checkpoint_tail.inl.
//
// Causal responsibility owned here: the StreamState *lifecycle and tick
// execution driver* -- constructing/adopting a stream (adopt_adult,
// make_stream), delivering host contact bytes into it (contact_host_bytes,
// the streaming-contact assembly helpers), running exactly one tick end to
// end (tick_untransactional and its tick() overloads), reading its public
// report (read_report), and the coarse experimental lesions (lesion_drive,
// lesion_appraisal, lesion_plasticity).
//
// This is deliberately NOT the state layout (StreamState and its member
// structs stay in the parent, alongside allocate_transport which both this
// driver and checkpoint restore depend on) and NOT the device kernel
// primitives (the __global__ kernels this driver launches, in causal-path
// order, stay in the parent immediately above the include point below).
// Single writer: tick_untransactional is the only place that sequences the
// per-tick kernel launches and produces a TickResult; every other function
// here is a thin, single-purpose entry into that same StreamState by
// reference. Narrow data contract: StreamState&, TickResult, StreamReport,
// StreamConfig -- all defined in the parent above the include point, all
// unchanged by this move.
//
// Included, still inside `namespace bcc32_cuda_adult_stream_v1`, at the
// exact point in the parent this text used to occupy (after allocate_transport,
// before the checkpoint tail). It carries its own
// `#if !defined(BCC32_CUDA_ADULT_STREAM_CHECKPOINT_ONLY)` guard (this whole
// driver is inert in checkpoint-only builds), matching the guard structure
// that stood at this point in the parent before the move.

#if !defined(BCC32_CUDA_ADULT_STREAM_CHECKPOINT_ONLY)

// Install an already-grown adult into the canonical stream without
// serializing and reloading it. This initializes only the transport shell;
// every resident organ and learned state moves intact.
inline StreamState adopt_adult(adult::AdultState&& trained_adult,
                               StreamConfig config = {}) {
  if (config.emission_capacity == 0u || config.chunk_capacity == 0u ||
      config.appraisal_holdout_bytes == 0u) {
    throw std::runtime_error("invalid stream configuration");
  }
  const bool resident_language_enabled =
      config.resident_language_enabled != 0u;
  if (resident_language_enabled != trained_adult.surface_organ_enabled) {
    throw std::runtime_error(
        "adopted adult resident-language configuration mismatch");
  }
  StreamState state;
  if (trained_adult.online_conditioned_transition_count != 0u) {
    throw std::runtime_error(
        "cannot adopt conditioned inventory without its physical organ bank");
  }
  state.conditioned_device_owner =
      bcc32::paged_conditioned_owner::PagedConditionedOwner(
          resident_language_enabled ? config.conditioned_organ_capacity : 0u);
  if (resident_language_enabled) {
    state.conditioned_device_owner.reset_resident_factor_state(
        bcc32::paged_conditioned_owner::kResidentFactorWords,
        bcc32::paged_conditioned_owner::kResidentFactorWords);
  }
  state.adult = std::move(trained_adult);
  state.adult.resident_bytes =
      complete_checkpoint::device_allocation_bytes(state.adult);
  state.adult.streaming_cue_mode = resident_language_enabled;
  state.legacy_generator_enabled = config.legacy_generator_enabled != 0u;
  allocate_transport(state, config.chunk_capacity, config.emission_capacity,
                     config.appraisal_holdout_bytes);
  adult::cuda_require(cudaMemset(state.drive.get(), 0, state.drive.bytes()),
                      "clear resident stream drive");
  adult::cuda_require(cudaMemset(state.appraisal.get(), 0,
                                 state.appraisal.bytes()),
                      "clear resident stream appraisal");
  adult::cuda_require(cudaMemset(state.appraisal_workspace.get(), 0,
                                 state.appraisal_workspace.bytes()),
                      "clear resident appraisal workspace");
  adult::cuda_require(cudaMemset(state.contact_summary.get(), 0,
                                 state.contact_summary.bytes()),
                      "clear resident contact summary");
  initialize_drive_kernel<<<1u, 32u>>>(state.drive.get(), config);
  appraisal::initialize_appraisal_kernel<<<1u, 32u>>>(state.appraisal.get());
  sync_appraisal_parameters_kernel<<<1u, 32u>>>(
      state.appraisal.get(), state.drive.get());
  auto* query_plan = state.query_plan.get();
  auto* query_settlement = state.query_settlement.get();
  auto* query_answer_receipt = state.query_answer_receipt.get();
  void* query_state_arguments[] = {
      &query_plan, &query_settlement, &query_answer_receipt};
  adult::cuda_require(
      cudaLaunchKernel(
          reinterpret_cast<const void*>(initialize_query_answer_state_kernel),
          dim3{1u, 1u, 1u}, dim3{2u, 1u, 1u}, query_state_arguments, 0u,
          nullptr),
      "launch adopted resident query-answer initialization");
  adult::cuda_require(cudaMemset(state.query_grounding.get(), 0,
                                 state.query_grounding.bytes()),
                      "clear resident contact-response grounding");
  adult::cuda_require(cudaMemset(state.query_plan_grounding_observer.get(), 0,
                                 state.query_plan_grounding_observer.bytes()),
                      "clear adopted resident plan grounding observer");
  adult::cuda_require(cudaMemset(state.pending_action_trajectory.get(), 0,
                                 state.pending_action_trajectory.bytes()),
                      "clear pending resident action trajectory");
  adult::cuda_require(cudaGetLastError(),
                      "launch adopted resident stream initialization");
  adult::cuda_require(cudaDeviceSynchronize(),
                      "complete adopted resident stream initialization");
  return state;
}

inline StreamState make_stream(const std::uint8_t* bytes,
                               std::uint32_t byte_count,
                               std::uint32_t seed = 0x41c64e6du,
                               StreamConfig config = {}) {
  if (config.emission_capacity == 0u || config.chunk_capacity == 0u ||
      config.appraisal_holdout_bytes == 0u) {
    throw std::runtime_error("invalid stream configuration");
  }
  StreamState state;
  const bool resident_language_enabled =
      config.resident_language_enabled != 0u;
  state.conditioned_device_owner =
      bcc32::paged_conditioned_owner::PagedConditionedOwner(
          resident_language_enabled ? config.conditioned_organ_capacity : 0u);
  if (resident_language_enabled) {
    state.conditioned_device_owner.reset_resident_factor_state(
        bcc32::paged_conditioned_owner::kResidentFactorWords,
        bcc32::paged_conditioned_owner::kResidentFactorWords);
  }
  state.adult = adult::train_raw_bytes(
      bytes, byte_count, false, seed,
      false,
      config.resident_language_enabled != 0u);
  // Duplex transport frames are not episode boundaries. Resident language
  // learns and queries only a complete contact closed by its learned surface
  // matter (or an explicit quiet flush).
  state.adult.streaming_cue_mode = config.resident_language_enabled != 0u;
  state.legacy_generator_enabled = config.legacy_generator_enabled != 0u;
  allocate_transport(state, config.chunk_capacity, config.emission_capacity,
                     config.appraisal_holdout_bytes);
  adult::cuda_require(cudaMemset(state.drive.get(), 0, state.drive.bytes()),
                      "clear resident stream drive");
  adult::cuda_require(cudaMemset(state.appraisal.get(), 0,
                                 state.appraisal.bytes()),
                      "clear resident stream appraisal");
  adult::cuda_require(cudaMemset(state.appraisal_workspace.get(), 0,
                                 state.appraisal_workspace.bytes()),
                      "clear resident appraisal workspace");
  adult::cuda_require(cudaMemset(state.contact_summary.get(), 0,
                                 state.contact_summary.bytes()),
                      "clear resident contact summary");
  initialize_drive_kernel<<<1u, 32u>>>(state.drive.get(), config);
  appraisal::initialize_appraisal_kernel<<<1u, 32u>>>(state.appraisal.get());
  sync_appraisal_parameters_kernel<<<1u, 32u>>>(
      state.appraisal.get(), state.drive.get());
  auto* query_plan = state.query_plan.get();
  auto* query_settlement = state.query_settlement.get();
  auto* query_answer_receipt = state.query_answer_receipt.get();
  void* query_state_arguments[] = {
      &query_plan, &query_settlement, &query_answer_receipt};
  adult::cuda_require(
      cudaLaunchKernel(
          reinterpret_cast<const void*>(initialize_query_answer_state_kernel),
          dim3{1u, 1u, 1u}, dim3{2u, 1u, 1u}, query_state_arguments, 0u,
          nullptr),
      "launch resident query-answer initialization");
  adult::cuda_require(cudaMemset(state.query_grounding.get(), 0,
                                 state.query_grounding.bytes()),
                      "clear resident contact-response grounding");
  adult::cuda_require(cudaMemset(state.query_plan_grounding_observer.get(), 0,
                                 state.query_plan_grounding_observer.bytes()),
                      "clear resident plan grounding observer");
  adult::cuda_require(cudaMemset(state.pending_action_trajectory.get(), 0,
                                 state.pending_action_trajectory.bytes()),
                      "clear pending resident action trajectory");
  adult::cuda_require(cudaGetLastError(),
                      "launch resident stream initialization");
  adult::cuda_require(cudaDeviceSynchronize(),
                      "complete resident stream initialization");
  return state;
}

inline void contact_host_bytes(StreamState& state,
                               const std::uint8_t* host_bytes,
                               const std::uint8_t* device_bytes,
                               std::uint32_t byte_count, bool reafferent,
                               std::int32_t efferent_polarity = 0,
                               bool outcome_present = true,
                               bool relation_observation = true,
                               const std::uint32_t* contact_learned_request =
                                   nullptr,
                               bool initial_exposure = false,
                               bool synchronize_completion = true) {
  if (byte_count == 0u) return;
  if (host_bytes == nullptr || device_bytes == nullptr)
    throw std::runtime_error("null nonempty resident contact");

  const bool plasticity_applied = !state.plasticity_disabled;
  if (plasticity_applied) {
    const bool physical_learning = conditioned_credit_owner_active(state);
    begin_contact(state);
    adult::assimilate_raw_bytes(state.adult, host_bytes, byte_count, false,
                                0x9e3779b9u, efferent_polarity != 0,
                                efferent_polarity, outcome_present,
                                relation_observation && !reafferent,
                                contact_learned_request, initial_exposure,
                                nullptr,
                                physical_learning ? &state : nullptr,
                                physical_learning
                                    ? consume_conditioned_credit_chunk
                                    : nullptr,
                                physical_learning
                                    ? publish_conditioned_conductance_chunk
                                    : nullptr,
                                physical_learning
                                    ? consume_conditioned_prediction_witness_chunk
                                    : nullptr,
                                nullptr, false);
    if (physical_learning &&
        !state.conditioned_credit_advanced_this_contact) {
      consume_conditioned_credit_chunk(&state, nullptr, 0u);
    } else if (!physical_learning) {
      publish_conditioned_conductance(state);
    }
  }

  adult::cuda_require(cudaMemset(state.contact_summary.get(), 0,
                                 state.contact_summary.bytes()),
                      "clear parallel contact summary");
  summarize_contact_parallel_kernel<<<
      std::min(4096u, adult::blocks_for(byte_count)), adult::kBlock>>>(
      device_bytes, byte_count, state.drive.get(), state.contact_summary.get());

  const std::uint32_t holdout_count =
      std::min(byte_count, state.appraisal_holdout_bytes);
  const std::uint32_t holdout_offset = byte_count - holdout_count;
  const std::uint32_t heldout_begin =
      holdout_count > appraisal::kGenomeParameters
          ? std::max(appraisal::kGenomeParameters, holdout_count / 2u)
          : 0u;
  appraisal::propose_mutations_kernel<<<1u, appraisal::kCandidateCount>>>(
      state.appraisal.get(), state.appraisal_workspace.get());
  appraisal::shadow_evaluate_parallel_kernel<<<
      appraisal::kCandidateCount + 1u, appraisal::kCandidateCount>>>(
      device_bytes + holdout_offset, holdout_count, heldout_begin,
      state.appraisal.get(), state.appraisal_workspace.get());
  appraisal::resident_select_commit_parallel_kernel<<<
      1u, appraisal::kCandidateCount>>>(
      holdout_count, heldout_begin,
      state.chronological_bytes + holdout_offset, reafferent ? 1u : 0u,
      state.appraisal.get(), state.appraisal_workspace.get());
  sync_appraisal_parameters_kernel<<<1u, 32u>>>(
      state.appraisal.get(), state.drive.get());
  finalize_contact_summary_kernel<<<1u, 32u>>>(
      byte_count, reafferent ? 1u : 0u, plasticity_applied ? 1u : 0u,
      state.contact_summary.get(), state.appraisal.get(), state.drive.get());
  adult::cuda_require(cudaGetLastError(),
                      "launch parallel stream assimilation summaries");
  if (synchronize_completion) {
    adult::cuda_require(cudaDeviceSynchronize(),
                        "complete parallel stream assimilation summaries");
  }
  state.chronological_bytes += byte_count;
}

#include "hardware_native/bcc32_cuda_adult_stream_plan_grounding_observer.inl"
#include "hardware_native/bcc32_cuda_adult_stream_tick_surface_phases.inl"

inline std::vector<std::uint8_t> append_streaming_contact_for_mode(
    adult::AdultState& state, const std::uint8_t* host_bytes,
    std::uint32_t byte_count, bool flush_contact,
    bool complete_on_closure);

inline TickResult tick_untransactional(
    StreamState& state, const std::uint8_t* host_bytes,
    std::uint32_t byte_count, bool preassembled_contact = false,
    std::int32_t efferent_polarity = 0, bool outcome_present = true,
    bool initial_exposure = false) {
  // The stream accepts complete host frames. This transport compatibility
  // argument must not be mistaken for reafferent outcome credit.
  (void)preassembled_contact;
  if (byte_count > state.chunk_capacity) {
    throw std::runtime_error("inbound chunk exceeds resident stream capacity");
  }
  // Phase attribution instrument. Kernel launches are asynchronous, so a wall
  // clock can only be charged to a named phase if the device is drained at the
  // phase boundary. Those drains serialize work that normally overlaps and so
  // would perturb the very timing being measured; therefore the whole
  // instrument -- syncs and prints alike -- is inert unless
  // BCC32_TICK_PHASE_DIAG is set. The variable is read once into a function
  // static, never per phase, so an ordinary tick pays one predicted branch per
  // boundary and nothing else.
  static const bool tick_phase_diag =
      std::getenv("BCC32_TICK_PHASE_DIAG") != nullptr;
  static std::uint32_t tick_phase_serial = 0u;
  // POST-increment so this serial starts at 0 and lines up with the tool's
  // `ticks_executed`, which is printed by tick_ms/tick_query BEFORE its own
  // `++ticks_executed`. A pre-increment here reports 1 for the tick everything
  // else calls 0, and the 40.1 s tick would then be attributed to the wrong
  // phase row -- a silent off-by-one between two instruments is worse than no
  // instrument.
  const std::uint32_t tick_phase_tick =
      tick_phase_diag ? tick_phase_serial++ : 0u;
  std::chrono::steady_clock::time_point tick_phase_begin;
  if (tick_phase_diag) tick_phase_begin = std::chrono::steady_clock::now();
  const auto tick_phase_mark = [&](const char* phase) {
    if (!tick_phase_diag) return;
    (void)cudaDeviceSynchronize();
    const std::chrono::steady_clock::time_point now =
        std::chrono::steady_clock::now();
    std::fprintf(stderr, "tick_phase tick=%u phase=%s micros=%llu\n",
                 tick_phase_tick, phase,
                 static_cast<unsigned long long>(
                     std::chrono::duration_cast<std::chrono::microseconds>(
                         now - tick_phase_begin)
                         .count()));
    tick_phase_begin = now;
  };
  std::vector<std::uint8_t> completed_contact;
  if (state.adult.streaming_cue_mode && !preassembled_contact) {
    completed_contact = append_streaming_contact_for_mode(
        state.adult, host_bytes, byte_count, byte_count == 0u,
        !state.explicit_streaming_flush);
    if (completed_contact.empty() && byte_count != 0u) {
      // A transport fragment has no independent linguistic or credit-bearing
      // status. Its bytes remain only in the device-resident contact field.
      return {};
    }
    if (!completed_contact.empty()) {
      if (completed_contact.size() > state.chunk_capacity) {
        throw std::runtime_error("completed resident contact exceeds stream capacity");
      }
      host_bytes = completed_contact.data();
      byte_count = static_cast<std::uint32_t>(completed_contact.size());
      preassembled_contact = true;
    }
  }
  PinnedEmissionPublication* emission_publication = nullptr;
  if (state.emission_capacity <= kPinnedEmissionPublicationLimit) {
    emission_publication = &pinned_emission_publication();
    emission_publication->allocate(state.emission_capacity);
  }
  tick_phase_mark("streaming_contact_assembly");
  if (byte_count != 0u) {
    if (host_bytes == nullptr) throw std::runtime_error("null nonempty inbound chunk");
    adult::cuda_require(cudaMemcpy(state.ingress.get(), host_bytes, byte_count,
                                   cudaMemcpyHostToDevice),
                        "transport inbound raw stream bytes");
    // Form the cue before this contact can alter resident matter. A response
    // therefore cannot manufacture the evidence that selects it.
    adult::condition_on_raw_cue(state.adult, host_bytes, byte_count, false,
                                 true, preassembled_contact);
    tick_phase_mark("cue_conditioning");
    // Reafferent traffic never reaches this stage. A later external contact
    // closes a performed action only as temporal observation, never reward.
    settle_pending_action_trajectory_kernel<<<1u, 1u>>>(
        adult::resident_proposition_tissue_view(state.adult),
        adult::resident_surface_population_view(state.adult),
        state.adult.proposition_cue_sequence.get(),
        state.adult.proposition_cue_sequence_count.get(),
        state.ingress.get(), byte_count,
        state.pending_action_trajectory.get(),
        state.adult.question_goal_state.get(), state.query_plan.get(),
        state.action_transitions.get(),
        static_cast<std::uint32_t>(state.action_transitions.size()),
        state.action_transition_scalars.get());
    capture_contact_source_kernel<<<1u, 1u>>>(
        adult::resident_surface_population_view(state.adult),
        state.adult.proposition_cue_sequence.get(),
        state.adult.proposition_cue_sequence_count.get(),
        state.ingress.get(), byte_count,
        state.pending_action_trajectory.get());
    // Env-gated ablation of the bounded two-binding join in the ordered
    // settlement. Read ONCE into a function static -- device code cannot call
    // getenv, and a per-tick getenv would itself be a measurable cost on the
    // path being measured. Unset == false == join enabled == today's
    // behaviour; the join body is retained, not deleted, because it merely has
    // never succeeded on the two corpora measured.
    static const bool ordered_join_lesioned =
        std::getenv("BCC32_ORDERED_JOIN_LESION") != nullptr;
    stage_contact_response_plan_kernel<<<1u, 1u>>>(
        adult::resident_proposition_tissue_view(state.adult),
        adult::resident_proposition_workspace_view(state.adult),
        adult::resident_surface_population_view(state.adult),
        state.adult.proposition_cue_sequence.get(),
        state.adult.proposition_cue_sequence_count.get(),
        state.adult.relation_cue_exact.get(),
        state.adult.relation_cue_scores.get(),
        state.adult.relation_cue_orders.get(),
        state.adult.qonset_count.get(),
        state.adult.role_canon_lesioned
            ? nullptr : state.adult.construction_role_canon.get(),
        state.adult.construction_closed_class_mask.get(),
        state.adult.proposition_ordered_construction.get(),
        static_cast<std::uint32_t>(state.adult.proposition_ordered_construction.size()),
        state.ingress.get(), byte_count,
        state.adult.proposition_completion_result.get(), state.query_plan.get(),
        state.query_settlement.get(), state.pending_action_trajectory.get(),
        state.action_transitions.get(),
        static_cast<std::uint32_t>(state.action_transitions.size()),
        state.action_transition_scalars.get(),
        state.query_answer_receipt.get(), ordered_join_lesioned);
    // Execute the exact ordered cue against the pre-contact tissue. The raw
    // contact is assimilated only after this selector and its producer receipt
    // are frozen, so a question cannot manufacture its own answer authority.
    adult::ordered_relation::execute_ordered_relation_from_exact_cue_kernel<<<
        1u, 1u>>>(
        adult::resident_proposition_tissue_view(state.adult),
        adult::resident_surface_population_view(state.adult),
        state.adult.relation_cue_exact.get(),
        state.adult.relation_cue_orders.get(), 0u, state.adult.unit_count,
        state.adult.qonset_evidence_revision.get(),
        state.adult.question_goal_state.get() == nullptr
            ? nullptr
            : &state.adult.question_goal_state.get()->answered_evidence_revision,
        state.ordered_relation_output_units.get(),
        static_cast<std::uint32_t>(state.ordered_relation_output_units.size()),
        state.ordered_relation_output_count.get(),
        state.ordered_relation_execution_receipt.get());
    latch_ordered_relation_execution_receipt_kernel<<<1u, 1u>>>(
        adult::resident_proposition_tissue_view(state.adult),
        state.ordered_relation_execution_receipt.get(),
        state.query_answer_receipt.get());
    adult::cuda_require(cudaGetLastError(),
                        "execute pre-contact ordered relation output");
    adult::cuda_require(cudaMemset(state.query_relation_evidence_revision.get(), 0,
                                   state.query_relation_evidence_revision.bytes()),
                        "clear current relation surface evidence");
    adult::cuda_require(cudaMemset(state.query_relation_evidence_count.get(), 0,
                                   state.query_relation_evidence_count.bytes()),
                        "clear current relation surface extent");
    adult::cuda_require(cudaMemset(state.query_surface_selection.get(), 0xff,
                                   state.query_surface_selection.bytes()),
                        "clear current frozen surface selections");
    adult::cuda_require(cudaMemset(state.query_surface_anchors.get(), 0xff,
                                   state.query_surface_anchors.bytes()),
                        "clear current frozen surface anchors");
    adult::cuda_require(cudaMemset(state.query_surface_anchor_counts.get(), 0,
                                   state.query_surface_anchor_counts.bytes()),
                        "clear current frozen surface anchor extents");
    tick_phase_mark("contact_response_plan_staging");
    if (state.adult.surface_organ_enabled) {
    // Form a relation commitment from the cue against the store as it existed
    // before this contact. This is the graph route that can cross episode
    // boundaries through learned subject/value identity, unlike the local
    // ordered-binding settlement above.
    adult::cuda_require(cudaMemset(state.adult.relation_triple_cursor.get(), 0,
                                   state.adult.relation_triple_cursor.bytes()),
                        "clear stream relation commitment cursor");
    adult::cuda_require(cudaMemset(state.adult.relation_operator_order.get(), 0xff,
                                   state.adult.relation_operator_order.bytes()),
                        "clear stream relation operator order");
    // Select the concrete cue endpoint before opening the relation tournament.
    // A learned interrogative form may rank a grounded fact, but it cannot
    // substitute for its endpoint: otherwise common question material opens
    // a large unrelated corpus cohort and suppresses the exact taught fact.
    adult::cuda_require(cudaMemset(state.query_topic_support.get(), 0,
                                   state.query_topic_support.bytes()),
                        "clear stream cue relation support");
    adult::cuda_require(cudaMemset(state.query_topic_key.get(), 0,
                                   state.query_topic_key.bytes()),
                        "clear stream cue relation key");
    accumulate_recent_exact_content_relation_support_kernel<<<
        adult::blocks_for(adult::construction::kWitnessedRelationEventCap),
        adult::kBlock>>>(
        state.adult.witnessed_relation_events.get(),
        state.adult.witnessed_relation_event_cursor.get(),
        state.adult.relation_cue_exact.get(), state.adult.relation_cue_orders.get(),
        state.adult.construction_closed_class_mask.get(),
        state.adult.witnessed_relation_surface_units.get(),
        state.adult.witnessed_relation_surface_counts.get(),
        state.adult.qonset_count.get(),
        state.adult.unit_count,
        state.query_topic_support.get());
    derive_supported_exact_content_key_kernel<<<
        adult::blocks_for(state.adult.unit_count), adult::kBlock>>>(
        state.query_topic_support.get(), state.adult.relation_cue_exact.get(),
        state.adult.relation_cue_orders.get(),
        state.adult.construction_closed_class_mask.get(), state.adult.unit_count,
        state.query_topic_key.get());
    publish_specific_topic_unit_kernel<<<1u, 1u>>>(
        state.query_topic_key.get(), state.adult.relation_operator_order.get());
    adult::cuda_require(cudaGetLastError(), "derive stream relation-supported topic unit");
    tick_phase_mark("relation_topic_derivation");
    adult::construction::gather_relation_triples_kernel<<<
        adult::blocks_for(adult::construction::kRelationTripleHashCap),
        adult::kBlock>>>(
        state.adult.relation_triples.get(), state.adult.relation_triple_counts.get(),
        state.adult.role_canon_lesioned ? nullptr
                                        : state.adult.construction_role_canon.get(),
        state.adult.relation_cue_scores.get(), state.adult.relation_cue_orders.get(),
        state.adult.relation_cue_exact.get(), nullptr,
        state.adult.relation_operator_order.get(), adult::kCueNearIdentity,
        state.adult.relation_triple_candidates.get(),
        state.adult.relation_triple_cursor.get());
    adult::cuda_require(cudaGetLastError(), "gather stream grounded relation triples");
    // The cohort is grounded to one exact learned cue endpoint. If that unit
    // has no stored relation, abstain. Replacing it with another cue position
    // turns a missing fact into an unrelated answer.
    std::uint32_t relation_candidate_count = 0u;
    adult::cuda_require(cudaMemcpy(&relation_candidate_count,
                                   state.adult.relation_triple_cursor.get(),
                                   sizeof(relation_candidate_count),
                                   cudaMemcpyDeviceToHost),
                        "read stream operator relation extent");
    tick_phase_mark("relation_triple_gather");
    const bool relation_topic_fallback = relation_candidate_count == 0u;
    // ANALOGICAL SUBSTITUTION CONSUMER: when the exact cue endpoint carries
    // no direct claim (relation_topic_fallback), the resident relation-
    // triple store still offers licensed-novel substitution through the
    // topic's emergent distributional category -- count_category_mates_
    // kernel finds other subjects sharing >= kAnalogyMinSharedContexts
    // predication contexts with the topic, gather_analogical_triples_kernel
    // then borrows one of THEIR unattested-for-topic claims as a candidate
    // (topic, K, B'). This is the exact retrieval geometry the bare-adult
    // tool path's generate_construction_reply/try_analogy already draws
    // replies from (same table, same seed/mate-count thresholds, no new
    // semantics); before this, neither kernel was ever launched from the
    // real dialogue/stream tick, so the store only gated a reply here, it
    // never SOURCED one when the direct topic was empty.
    std::uint32_t analogical_candidate_count = 0u;
    std::uint32_t analogy_topic = adult::construction::kNoTripleUnit;
    // Declared unallocated. cudaMalloc and cudaFree are device-wide
    // synchronizing operations: constructing these two unit_count-sized
    // scratch buffers unconditionally charged EVERY contact tick four such
    // barriers (2 malloc + 2 free, 2 * 4 * unit_count bytes) for a branch that
    // only runs when the exact cue endpoint carried no direct claim AND the
    // topic resolves. They are allocated at the single point of first use
    // below and are still read afterwards through analogical_candidate_count,
    // which can only be nonzero once that same branch allocated and filled
    // them, so a null get() can never reach form_triple_commitment_kernel.
    adult::DeviceArray<std::uint32_t> mate_counts_buffer;
    adult::DeviceArray<std::uint32_t> subject_degree_buffer;
    if (relation_topic_fallback && !state.adult.relation_triple_lesioned &&
        state.adult.relation_triples.get() != nullptr) {
      // relation_operator_order[0] still holds the exact cue topic unit
      // publish_specific_topic_unit_kernel latched above (it is not
      // overwritten with real per-unit operator-order data until the
      // memset immediately below) -- read it back before that happens.
      adult::cuda_require(
          cudaMemcpy(&analogy_topic, state.adult.relation_operator_order.get(),
                     sizeof(analogy_topic), cudaMemcpyDeviceToHost),
          "read stream analogy topic candidate");
      if (analogy_topic != adult::construction::kNoTripleUnit &&
          analogy_topic < state.adult.unit_count) {
        // Single point of first use. Same extent as the unconditional
        // construction this replaces; the memsets below still establish the
        // same zeroed content, so the kernels see identical inputs.
        mate_counts_buffer.allocate(
            std::max<std::uint32_t>(1u, state.adult.unit_count));
        subject_degree_buffer.allocate(
            std::max<std::uint32_t>(1u, state.adult.unit_count));
        adult::cuda_require(cudaMemset(mate_counts_buffer.get(), 0,
                                       mate_counts_buffer.bytes()),
                            "clear stream category mate counts");
        adult::cuda_require(cudaMemset(subject_degree_buffer.get(), 0,
                                       subject_degree_buffer.bytes()),
                            "clear stream subject context degrees");
        adult::construction::count_category_mates_kernel<<<
            adult::blocks_for(adult::construction::kRelationTripleHashCap),
            adult::kBlock>>>(
            state.adult.relation_triples.get(),
            state.adult.relation_triple_counts.get(), analogy_topic,
            state.adult.role_canon_lesioned
                ? nullptr
                : state.adult.construction_role_canon.get(),
            state.adult.unit_count, mate_counts_buffer.get(),
            subject_degree_buffer.get());
        adult::cuda_require(cudaGetLastError(), "count stream category mates");
        adult::cuda_require(cudaMemset(state.adult.relation_triple_cursor.get(), 0,
                                       state.adult.relation_triple_cursor.bytes()),
                            "reset stream analogical candidate cursor");
        adult::construction::gather_analogical_triples_kernel<<<
            adult::blocks_for(adult::construction::kRelationTripleHashCap),
            adult::kBlock>>>(
            state.adult.relation_triples.get(),
            state.adult.relation_triple_counts.get(), analogy_topic,
            mate_counts_buffer.get(),
            state.adult.role_canon_lesioned
                ? nullptr
                : state.adult.construction_role_canon.get(),
            state.adult.unit_count, state.adult.relation_triple_candidates.get(),
            state.adult.relation_triple_cursor.get());
        adult::cuda_require(cudaGetLastError(),
                            "gather stream analogical triples");
        adult::cuda_require(cudaMemcpy(&analogical_candidate_count,
                                       state.adult.relation_triple_cursor.get(),
                                       sizeof(analogical_candidate_count),
                                       cudaMemcpyDeviceToHost),
                            "read stream analogical candidate extent");
      }
    }
    tick_phase_mark("relation_analogical_substitution");
    // The endpoint was required for retrieval. Recompute the learned question
    // relation now that the cohort is fixed so it can rank, never admit,
    // candidate facts.
    adult::cuda_require(cudaMemset(state.adult.relation_operator_order.get(), 0xff,
                                   state.adult.relation_operator_order.bytes()),
                        "clear stream relation operator after topic retrieval");
    adult::construction::derive_relation_operator_order_kernel<<<
        adult::blocks_for(state.adult.unit_count), adult::kBlock>>>(
        state.adult.relation_cue_scores.get(), state.adult.relation_cue_orders.get(),
        state.adult.role_canon_lesioned ? nullptr
                                        : state.adult.construction_role_canon.get(),
        state.adult.qonset_count.get(), state.adult.relation_triple_type_total.get(),
        adult::kCueNearIdentity, state.adult.unit_count,
        state.adult.relation_operator_order.get());
    adult::cuda_require(cudaGetLastError(), "derive stream relation operator after topic retrieval");
    adult::construction::form_triple_commitment_kernel<adult::BigramKey><<<1u, 1u>>>(
        state.adult.relation_triples.get(), state.adult.relation_triple_counts.get(),
        state.adult.relation_triple_candidates.get(),
        state.adult.relation_triple_cursor.get(),
        state.adult.relation_triple_type_total.get(),
        state.adult.relation_triple_type_mirrored.get(),
        state.adult.relation_cue_scores.get(), state.adult.relation_cue_orders.get(),
        state.adult.relation_cue_exact.get(), state.adult.relation_operator_order.get(),
        adult::kCueNearIdentity, relation_topic_fallback,
        state.adult.role_canon_lesioned ? nullptr
                                        : state.adult.construction_role_canon.get(),
        state.adult.qonset_count.get(), state.adult.unit_count,
        state.adult.unit_vitality.get(), state.adult.construction_closed_class_mask.get(),
        state.adult.unit_lengths.get(), state.adult.unit_content.get(), adult::kUnitWords,
        state.adult.boundary_mask.get(), state.adult.construction_filler_terminal_mask.get(),
        state.adult.online_bigrams.get(), state.adult.online_bigram_counts.get(),
        state.adult.online_bigram_count,
        nullptr, 0u, adult::construction::kNoTripleUnit,
        nullptr, 0ull,
        /*analogical_topic=*/
        analogical_candidate_count != 0u ? analogy_topic
                                         : adult::construction::kNoTripleUnit,
        /*mate_counts=*/
        analogical_candidate_count != 0u ? mate_counts_buffer.get() : nullptr,
        /*subject_degree=*/
        analogical_candidate_count != 0u ? subject_degree_buffer.get()
                                         : nullptr,
        /*unit_pos=*/
        analogical_candidate_count != 0u &&
                state.adult.unit_pos.size() >= state.adult.unit_count
            ? state.adult.unit_pos.get()
            : nullptr,
        adult::construction::kRelationTripleMaxClauses,
        state.adult.relation_triple_plan.get(), state.adult.relation_triple_meta.get());
    // A counted relation may speak only through the construction learned from
    // the same exact acquisition event. The aggregate table alone is not a
    // surface authority, and a multi-clause reply waits for the retained path
    // implementation rather than validating a single convenient clause.
    adult::construction::require_relation_construction_witness_kernel<<<1u, 1u>>>(
        state.adult.relation_triple_evidence_revision.get(),
        state.adult.construction_evidence_revision.get(),
        state.adult.construction_lengths.get(),
        state.adult.construction_store_count.get(),
        adult::construction::kConstructionCap,
        state.adult.relation_triple_meta.get());
    // If a single aggregate edge cannot carry the query, resolve the current
    // cue against retained exact episode order and consume only residual
    // content. This remains fully device-resident and can join one witnessed
    // overlap without a host answer table.
    adult::construction::form_witnessed_relation_plan_kernel<<<1u, 1u>>>(
        state.adult.witnessed_relation_events.get(),
        state.adult.witnessed_relation_event_cursor.get(),
        state.adult.relation_cue_exact.get(),
        state.adult.relation_cue_scores.get(),
        state.adult.relation_cue_orders.get(),
        state.adult.qonset_evidence_revision.get(),
        state.adult.qonset_count.get(),
        state.adult.relation_triples.get(),
        state.adult.relation_triple_counts.get(),
        state.adult.role_canon_lesioned
            ? nullptr
            : state.adult.construction_role_canon.get(),
        state.adult.construction_closed_class_mask.get(),
        state.adult.unit_vitality.get(),
        state.adult.witnessed_relation_surface_units.get(),
        state.adult.witnessed_relation_surface_counts.get(),
        state.adult.unit_count, relation_topic_fallback,
        state.adult.relation_triple_plan.get(), state.adult.relation_triple_meta.get(),
        state.query_relation_evidence_revision.get(),
        adult::construction::kRelationSurfaceEvidenceCap,
        state.query_relation_evidence_count.get(),
        state.query_relation_plan_receipt.get(), state.query_topic_key.get());
    latch_stream_relation_commitment_receipt_kernel<<<1u, 1u>>>(
        state.adult.relation_triple_cursor.get(), state.adult.relation_triple_meta.get(),
        relation_topic_fallback ? 1u : 0u,
        state.adult.relation_cue_exact.get(),
        state.adult.relation_cue_orders.get(),
        state.adult.qonset_evidence_revision.get(),
        state.adult.question_gap_field_support.get(),
        state.adult.role_canon_lesioned
            ? nullptr
            : state.adult.construction_role_canon.get(),
        state.adult.unit_count,
        state.query_answer_receipt.get());
    adult::cuda_require(cudaGetLastError(), "stage stream relation commitment");
    tick_phase_mark("relation_commitment_witness");
    run_query_plan_grounding_phase(state, tick_phase_mark);
    }
    contact_host_bytes(state, host_bytes, state.ingress.get(), byte_count,
                       false, efferent_polarity, outcome_present,
                       // Acquisition follows every external contact after its
                       // answer decision.  The relation retrieval path itself
                       // rejects cue-restating triples, so an overbroad learned
                       // question-form field cannot suppress witnessed facts.
                       true,
                       &state.query_answer_receipt.get()
                            ->learned_question_form,
                       initial_exposure,
                       // The remaining tick consumes these summaries on the
                       // same stream, then its mandatory transport readback
                       // establishes the public synchronous tick boundary.
                       false);
    discharge_ticketed_question_goal_kernel<<<1u, 1u>>>(
        adult::resident_proposition_tissue_view(state.adult),
        state.adult.question_goal_state.get(),
        state.pending_action_trajectory.get());
    adult::cuda_require(cudaGetLastError(),
                        "admit ticketed resident question return");
    tick_phase_mark("contact_assimilation");
  } else {
    // This first bridge realizes only the originating query contact. A later
    // autonomous tick must not replay a completed answer as if it were a new
    // resident query. It also must not inherit the prior contact's unresolved
    // raw-cue sentinel: condition_on_raw_cue() uses motor_context[5] == 2 to
    // suppress generation while that contact is being conditioned, and the
    // generator intentionally returns zero for that mode. Once the contact
    // tick has completed, the per-contact motor plan is stale; clearing it
    // here exposes the already-grown resident generator to the input-free
    // tick without selecting an action or changing learned morphology.
    adult::cuda_require(cudaMemset(state.adult.motor_context.get(), 0,
                                   state.adult.motor_context.bytes()),
                        "clear completed raw-cue motor context");
    adult::cuda_require(cudaMemset(state.adult.motor_completion.get(), 0,
                                   state.adult.motor_completion.bytes()),
                        "clear completed raw-cue motor completion");
    initialize_query_answer_state_kernel<<<1u, 1u>>>(
        state.query_plan.get(), state.query_settlement.get(),
        state.query_answer_receipt.get());
    adult::cuda_require(cudaMemset(state.query_relation_evidence_revision.get(), 0,
                                   state.query_relation_evidence_revision.bytes()),
                        "clear inactive relation surface evidence");
    adult::cuda_require(cudaMemset(state.query_relation_evidence_count.get(), 0,
                                   state.query_relation_evidence_count.bytes()),
                        "clear inactive relation surface extent");
    adult::cuda_require(cudaMemset(state.query_surface_selection.get(), 0xff,
                                   state.query_surface_selection.bytes()),
                        "clear inactive frozen surface selections");
    adult::cuda_require(cudaMemset(state.query_surface_anchor_counts.get(), 0,
                                   state.query_surface_anchor_counts.bytes()),
                        "clear inactive frozen surface anchor extents");
    adult::cuda_require(cudaMemset(state.query_plan_grounding_observer.get(), 0,
                                   state.query_plan_grounding_observer.bytes()),
                        "clear inactive plan grounding observer");
  }

  drive_tick_kernel<<<1u, 32u>>>(
      state.drive.get(), byte_count, state.query_plan.get(),
      state.query_answer_receipt.get());
  // A quiet tick may advance only already-resident route matter. The helper
  // reads its extent to size existing coactivity chemistry; it never reads
  // content, chooses a route, or authorizes public output. Running it after
  // drive_tick_kernel is important because that kernel resets the observer
  // receipt for this epoch.
  if (byte_count == 0u)
    (void)advance_endogenous_resident_step(
        state.adult, state.query_answer_receipt.get());
  adult::cuda_require(cudaMemcpyAsync(state.candidate_rng.get(), state.adult.rng.get(),
                                      sizeof(std::uint32_t),
                                      cudaMemcpyDeviceToDevice, nullptr),
                      "stage resident stream rng");
  QueryAnswerReceipt query_before_surface{};
  adult::cuda_require(cudaMemcpy(&query_before_surface,
                                 state.query_answer_receipt.get(),
                                 sizeof(query_before_surface),
                                 cudaMemcpyDeviceToHost),
                      "read resident query surface authority");
  tick_phase_mark("drive_and_query_readback");
  // A committed learned action either realizes its settled resident plan or
  // abstains. Running the legacy statistical walk first cannot help that path
  // and turns a plan-local response into a corpus-wide scan.
  if (state.legacy_generator_enabled && query_before_surface.attempted == 0u) {
    std::uint32_t subject_count = 0u;
    if (state.adult.subject_count.get() != nullptr) {
      adult::cuda_require(cudaMemcpy(&subject_count, state.adult.subject_count.get(),
                                     sizeof(subject_count), cudaMemcpyDeviceToHost),
                          "read stream subject count");
    }
    adult::generate_kernel<<<1u, adult::kBlock>>>(
        state.adult.unit_lengths.get(), state.adult.unit_content.get(),
        state.adult.unit_vitality.get(), state.adult.unigram_top_ids.get(),
        state.adult.bigrams.get(), state.adult.bigram_counts.get(), state.adult.bigram_count,
        state.adult.trigrams.get(), state.adult.trigram_counts.get(), state.adult.trigram_count,
        state.adult.online_bigrams.get(), state.adult.online_bigram_counts.get(),
        state.adult.online_bigram_count, state.adult.online_trigrams.get(),
        state.adult.online_trigram_counts.get(), state.adult.online_trigram_count,
        state.adult.online_conditioned_transitions.get(),
        state.adult.online_conditioned_transition_conductance.get(),
        state.adult.online_conditioned_transition_count,
        state.adult.motor_context.get(), state.adult.motor_completion.get(),
        state.adult.subject_ids.get(), state.adult.subject_weights.get(), subject_count,
        state.adult.qonset_count.get(), state.adult.qterm_count.get(),
        state.adult.qorig_onset.get(), state.adult.qorig_onset_w.get(),
        state.adult.qorig_onset_n.get(), state.adult.qorig_term.get(),
        state.adult.qorig_term_w.get(), state.adult.qorig_term_n.get(),
        state.adult.qorig_on ? 1u : 0u,
        state.candidate_rng.get(), state.candidate.get(), state.emission_capacity,
        state.generated_count.get());
  } else {
    set_candidate_count_kernel<<<1u, 1u>>>(0u, state.generated_count.get());
  }
  tick_phase_mark("legacy_statistical_generation");
  serialize_ordered_relation_public_output_kernel<<<1u, 1u>>>(
      state.ordered_relation_execution_receipt.get(),
      state.ordered_relation_output_units.get(),
      state.ordered_relation_output_count.get(), state.adult.unit_lengths.get(),
      state.adult.unit_content.get(), adult::kUnitWords, state.adult.unit_count,
      state.adult.boundary_mask.get(), state.candidate.get(),
      state.emission_capacity, state.generated_count.get(),
      state.query_answer_receipt.get());
  adult::cuda_require(cudaGetLastError(),
                      "serialize ordered relation public output");
  tick_phase_mark("ordered_relation_public_output");
  realize_relation_surface_phase(state, query_before_surface);
  tick_phase_mark("relation_surface_realization");
  // Autonomous continuation remains on the retained relation cursor until it
  // receives its own event-to-construction transaction. Initial grounded
  // answers, however, may no longer bypass learned surface realization.
  std::uint32_t autonomous_relation_plan_units = 0u;
  if (query_before_surface.attempted == 0u && byte_count == 0u &&
      state.adult.relation_triple_meta.get() != nullptr) {
    // This is an observer read of resident plan extent, not host selection of
    // output content. A valid retained relation plan may supersede the generic
    // walk; with no such plan, leave the already-generated resident candidate
    // intact instead of invoking a relation-silence kernel that zeroes it.
    adult::cuda_require(
        cudaMemcpy(&autonomous_relation_plan_units,
                   state.adult.relation_triple_meta.get(),
                   sizeof(autonomous_relation_plan_units),
                   cudaMemcpyDeviceToHost),
        "read autonomous relation plan extent");
  }
  if (query_before_surface.attempted == 0u && byte_count == 0u &&
      autonomous_relation_plan_units >= 3u &&
      autonomous_relation_plan_units <= adult::construction::kCommitmentCap) {
    emit_stream_relation_commitment_kernel<<<1u, 1u>>>(
        state.drive.get(), state.query_answer_receipt.get(), true,
        state.adult.unit_lengths.get(),
        state.adult.unit_content.get(), adult::kUnitWords,
        state.adult.relation_triple_plan.get(), state.adult.relation_triple_meta.get(),
        state.adult.boundary_mask.get(), state.candidate.get(),
        state.emission_capacity, state.generated_count.get());
  }
  tick_phase_mark("autonomous_relation_emission");
  // The generic surface organ may realize only a relation plan that survived
  // the exact event-to-construction witness gate above. Without that plan it
  // has no authority to fill a query with an unrelated resident construction.
  realize_stream_surface_phase(state, query_before_surface);
  tick_phase_mark("stream_surface_realization");
  realize_query_answer_plan_kernel<<<1u, 1u>>>(
      state.query_answer_receipt.get(), state.generated_count.get());
  realize_selected_action_surface_kernel<<<1u, 1u>>>(
      state.pending_action_trajectory.get(), state.query_answer_receipt.get(),
      state.candidate.get(), state.emission_capacity, state.generated_count.get());
  enforce_ordered_relation_public_silence_kernel<<<1u, 1u>>>(
      state.ordered_relation_execution_receipt.get(),
      state.generated_count.get(), state.query_answer_receipt.get());
  const bool staged_emission = emission_publication != nullptr &&
                               emission_publication->can_stage(
                                   state.emission_capacity);
  finalize_emission_kernel<<<1u, adult::kBlock>>>(
      state.drive.get(), state.candidate.get(), state.generated_count.get(),
      state.egress.get(), state.transport_counts.get(), state.candidate_rng.get(),
      state.adult.rng.get(), state.query_answer_receipt.get());
  adult::cuda_require(cudaGetLastError(), "launch fixed resident stream tick");
  tick_phase_mark("emission_finalize");

  // Same read-once discipline as the two env gates above: getenv walks the
  // whole environment block, and this one sat on the per-tick path, charging
  // that walk to every contact so an inert observer could stay inert.
  static const char* const resident_plan_diag =
      std::getenv("BCC32_RESIDENT_PLAN_DIAG");
  if (resident_plan_diag != nullptr) {
    QueryAnswerReceipt diagnostic{};
    adult::cuda_require(cudaMemcpy(&diagnostic, state.query_answer_receipt.get(),
                                   sizeof(diagnostic), cudaMemcpyDeviceToHost),
                        "read per-contact construction receipt");
    PlanAnchorGroundingObserverReceipt grounding_observer{};
    adult::cuda_require(
        cudaMemcpy(&grounding_observer,
                   state.query_plan_grounding_observer.get(),
                   sizeof(grounding_observer), cudaMemcpyDeviceToHost),
        "read per-contact plan grounding observer");
    std::fprintf(
        stderr,
        "resident_plan_anchor_grounding pre_valid=%u pre_status=%u pre_steps=%u pre_anchors=%u pre_opaque=%u pre_ordered=%u pre_other=%u pre_zero_population=%u pre_pop_refs=%u pre_question_goal=%u pre_revision=%u materializer_ready=%u materializer_steps=%u materializer_anchors=%u materializer_ambiguous=%u materializer_ungrounded=%u materializer_overlap=%u materializer_revision=%u post_status=%u post_anchors=%u post_revision=%u final_attempted=%u final_staged=%u final_anchors=%u final_plan_valid=%u final_plan_status=%u\n",
        grounding_observer.pre_valid, grounding_observer.pre_status,
        grounding_observer.pre_step_count,
        grounding_observer.pre_anchor_count,
        grounding_observer.pre_opaque_steps,
        grounding_observer.pre_ordered_steps,
        grounding_observer.pre_other_steps,
        grounding_observer.pre_zero_population_steps,
        grounding_observer.pre_population_references,
        grounding_observer.pre_question_goal_dependency,
        grounding_observer.pre_plan_revision,
        grounding_observer.materializer_ready,
        grounding_observer.materializer_grounded_steps,
        grounding_observer.materializer_anchor_count,
        grounding_observer.materializer_ambiguous_step,
        grounding_observer.materializer_ungrounded_step,
        grounding_observer.materializer_weakest_overlap,
        grounding_observer.materializer_plan_revision,
        grounding_observer.post_status, grounding_observer.post_anchor_count,
        grounding_observer.post_plan_revision,
        grounding_observer.final_attempted, grounding_observer.final_staged,
        grounding_observer.final_anchor_count,
        grounding_observer.final_plan_valid,
        grounding_observer.final_plan_status);
    if (diagnostic.attempted != 0u || diagnostic.staged != 0u ||
        diagnostic.construction_count != 0u ||
        diagnostic.question_onset_evidence != 0u) {
      adult::construction::WitnessedRelationPlanReceipt relation_plan_receipt{};
      adult::cuda_require(
          cudaMemcpy(&relation_plan_receipt,
                     state.query_relation_plan_receipt.get(),
                     sizeof(relation_plan_receipt), cudaMemcpyDeviceToHost),
          "read witnessed relation plan receipt");
      std::uint32_t witnessed_event_cursor = 0u;
      adult::cuda_require(
          cudaMemcpy(&witnessed_event_cursor,
                     state.adult.witnessed_relation_event_cursor.get(),
                     sizeof(witnessed_event_cursor), cudaMemcpyDeviceToHost),
          "read witnessed relation event extent");
      std::fprintf(stderr,
                   "resident_plan_tick bytes=%u question_form=%u question_onset=%u witnessed_events=%u attempted=%u staged=%u anchors=%u claimed=%u exact_matches=%u "
                   "qualified=%u witnessed=%u witness_cells=%u witness_overlap=%u selected_binding=%u selected_roles=%u spine_steps=%u spine_terminal=%u spine_ambiguous=%u "
                   "construction_count=%u supported=%u shape=%u mapping=%u tied=%u ready=%u grammar=%u output_bytes=%u relation_candidates=%u relation_units=%u relation_clauses=%u relation_fallback=%u relation_gap_field=%u trajectory_slots=%u trajectory_grounded=%u trajectory_ambiguous=%u relation_surface_events=%u relation_surface_witnesses=%u relation_surface_missing_events=%u relation_surface_missing_constructions=%u relation_surface_anchor_frames=%u\n",
                   byte_count, diagnostic.learned_question_form,
                   diagnostic.question_onset_evidence,
                   witnessed_event_cursor,
                   diagnostic.attempted, diagnostic.staged,
                   diagnostic.anchor_count, diagnostic.claimed_ordered_bindings,
                   diagnostic.exact_topic_matches, diagnostic.qualified_best_count,
                   diagnostic.qualified_witnessed_bindings,
                   diagnostic.witnessed_role_cell_overlap_bindings,
                   diagnostic.max_witnessed_role_cell_overlap,
                   diagnostic.selected_binding_index, diagnostic.selected_present_roles,
                   diagnostic.episode_spine_steps,
                   diagnostic.episode_spine_terminal,
                   diagnostic.episode_spine_ambiguous,
                   diagnostic.construction_count,
                   diagnostic.construction_supported,
                   diagnostic.construction_shape_matched,
                   diagnostic.construction_mapping_matched,
                   diagnostic.construction_tied, diagnostic.construction_ready,
                   diagnostic.construction_grammar_supported,
                   diagnostic.construction_output_bytes,
                   diagnostic.relation_candidate_count,
                   diagnostic.relation_plan_units,
                   diagnostic.relation_plan_clauses,
                   diagnostic.relation_topic_fallback,
                   diagnostic.relation_gap_field,
                   diagnostic.surface_trajectory_slots,
                   diagnostic.surface_trajectory_grounded,
                   diagnostic.surface_trajectory_ambiguous,
                   diagnostic.relation_surface_events,
                   diagnostic.relation_surface_witnesses,
                   diagnostic.relation_surface_missing_events,
                   diagnostic.relation_surface_missing_constructions,
                   diagnostic.relation_surface_anchor_frames);
      if (diagnostic.selected_binding_index <
          state.adult.proposition_ordered_bindings.size()) {
        adult::proposition_tissue::OrderedRoleBindingEvidence selected{};
        adult::cuda_require(
            cudaMemcpy(
                &selected,
                state.adult.proposition_ordered_bindings.get() +
                    diagnostic.selected_binding_index,
                sizeof(selected), cudaMemcpyDeviceToHost),
            "read selected ordered binding evidence");
        std::fprintf(
            stderr,
            "resident_plan_selected_binding index=%u episodic=%llu generalized=%llu intervention=%llu counter=%llu contexts=%u cells=%u,%u,%u units=%u,%u,%u role_mass=%llu,%llu,%llu structure=%llu context_mass=%llu\n",
            diagnostic.selected_binding_index,
            static_cast<unsigned long long>(
                selected.episodic_observation_mass),
            static_cast<unsigned long long>(
                selected.generalized_observation_mass),
            static_cast<unsigned long long>(selected.intervention_support),
            static_cast<unsigned long long>(selected.counterevidence),
            selected.qualifying_context_count, selected.role_counts[0],
            selected.role_counts[1], selected.role_counts[2],
            selected.role_unit_counts[0], selected.role_unit_counts[1],
            selected.role_unit_counts[2],
            static_cast<unsigned long long>(selected.role_structure_mass[0]),
            static_cast<unsigned long long>(selected.role_structure_mass[1]),
            static_cast<unsigned long long>(selected.role_structure_mass[2]),
            static_cast<unsigned long long>(selected.structure_mass),
            static_cast<unsigned long long>(selected.context_mass));
      }
      std::fprintf(
          stderr,
          "resident_plan_witness_decision requested=%u gap_ambiguous=%u best_revision=%llu best_hits=%u best_event=%u residual=%u next_revision=%llu next_hits=%u best_connective=%u next_connective=%u next_event_score=%u next_ambiguous=%u selected_next=%u selected_topic=%u scanned_live=%u topic_events=%u topic_groups=%u zero_residual=%u output=%u\n",
          relation_plan_receipt.requested_field,
          relation_plan_receipt.question_gap_ambiguous,
          static_cast<unsigned long long>(
              relation_plan_receipt.best_revision),
          relation_plan_receipt.best_hits,
          relation_plan_receipt.best_event,
          relation_plan_receipt.residual_count,
          static_cast<unsigned long long>(
              relation_plan_receipt.next_revision),
          relation_plan_receipt.next_hits,
          relation_plan_receipt.best_connective_score,
          relation_plan_receipt.next_connective_score,
          relation_plan_receipt.next_event_score,
          relation_plan_receipt.next_ambiguous,
          relation_plan_receipt.selected_next,
          relation_plan_receipt.selected_topic,
          relation_plan_receipt.scanned_live_events,
          relation_plan_receipt.topic_events,
          relation_plan_receipt.topic_groups,
          relation_plan_receipt.zero_residual_groups,
          relation_plan_receipt.output_count);
      if (diagnostic.construction_output_bytes != 0u &&
          diagnostic.construction_output_bytes <= state.emission_capacity) {
        std::vector<std::uint8_t> surface(diagnostic.construction_output_bytes);
        adult::cuda_require(cudaMemcpy(surface.data(), state.adult.surface_output_bytes.get(),
                                       surface.size(), cudaMemcpyDeviceToHost),
                            "read observer construction surface");
        std::fprintf(stderr, "resident_plan_surface bytes=%u text='%.*s'\n",
                     diagnostic.construction_output_bytes,
                     static_cast<int>(surface.size()),
                     reinterpret_cast<const char*>(surface.data()));
        std::fprintf(stderr, "resident_plan_surface_hex=");
        const std::size_t hex_extent = std::min<std::size_t>(surface.size(), 32u);
        for (std::size_t index = 0u; index < hex_extent; ++index)
          std::fprintf(stderr, "%02x", static_cast<unsigned>(surface[index]));
        std::fprintf(stderr, "\n");
      }
      std::uint64_t topic_key = 0u;
      adult::cuda_require(cudaMemcpy(&topic_key, state.query_topic_key.get(),
                                     sizeof(topic_key), cudaMemcpyDeviceToHost),
                          "read observer cue-specific topic key");
      if (topic_key != 0u) {
      const std::uint32_t topic_unit = recent_topic_unit(topic_key);
        std::vector<std::uint32_t> cue_exact(state.adult.unit_count);
        std::vector<std::uint32_t> cue_orders(state.adult.unit_count);
        adult::cuda_require(cudaMemcpy(cue_exact.data(), state.adult.relation_cue_exact.get(),
                                       cue_exact.size() * sizeof(std::uint32_t),
                                       cudaMemcpyDeviceToHost),
                            "read observer cue exact units");
        adult::cuda_require(cudaMemcpy(cue_orders.data(), state.adult.relation_cue_orders.get(),
                                       cue_orders.size() * sizeof(std::uint32_t),
                                       cudaMemcpyDeviceToHost),
                            "read observer cue unit orders");
        std::vector<std::uint64_t> topic_support(state.adult.unit_count);
        std::vector<std::uint32_t> vitality(state.adult.unit_count);
        adult::cuda_require(
            cudaMemcpy(topic_support.data(), state.query_topic_support.get(),
                       topic_support.size() * sizeof(std::uint64_t),
                       cudaMemcpyDeviceToHost),
            "read observer cue topic support");
        adult::cuda_require(
            cudaMemcpy(vitality.data(), state.adult.unit_vitality.get(),
                       vitality.size() * sizeof(std::uint32_t),
                       cudaMemcpyDeviceToHost),
            "read observer cue topic vitality");
        for (std::uint32_t unit = 0u; unit < state.adult.unit_count; ++unit) {
          if (cue_exact[unit] == 0u || topic_support[unit] == 0u) continue;
          std::uint32_t length = 0u;
          std::array<std::uint32_t, adult::kUnitWords> packed{};
          adult::cuda_require(
              cudaMemcpy(&length, state.adult.unit_lengths.get() + unit,
                         sizeof(length), cudaMemcpyDeviceToHost),
              "read observer cue topic candidate length");
          adult::cuda_require(
              cudaMemcpy(packed.data(),
                         state.adult.unit_content.get() +
                             unit * adult::kUnitWords,
                         sizeof(packed), cudaMemcpyDeviceToHost),
              "read observer cue topic candidate content");
          std::string text;
          for (std::uint32_t offset = 0u;
               offset < length && offset < adult::kUnitWords * 4u; ++offset)
            text.push_back(static_cast<char>(
                packed[offset / 4u] >> ((offset % 4u) * 8u)));
          std::fprintf(
              stderr,
              "resident_plan_topic_candidate unit=%u order=%u vitality=%u "
              "hits=%u revision=%u selected=%u text='%s'\n",
              unit, cue_orders[unit], vitality[unit],
              recent_topic_hits(topic_support[unit]),
              recent_topic_revision(topic_support[unit]),
              unit == topic_unit ? 1u : 0u, text.c_str());
        }
        if (topic_unit < state.adult.unit_count && cue_exact[topic_unit] != 0u) {
          const std::uint32_t unit = topic_unit;
          std::uint32_t length = 0u;
          std::array<std::uint32_t, adult::kUnitWords> packed{};
          adult::cuda_require(cudaMemcpy(&length, state.adult.unit_lengths.get() + unit,
                                         sizeof(length), cudaMemcpyDeviceToHost),
                              "read observer cue topic length");
          adult::cuda_require(cudaMemcpy(packed.data(),
                                         state.adult.unit_content.get() + unit * adult::kUnitWords,
                                         sizeof(packed), cudaMemcpyDeviceToHost),
                              "read observer cue topic content");
          std::string text;
          for (std::uint32_t offset = 0u;
               offset < length && offset < adult::kUnitWords * 4u; ++offset)
            text.push_back(static_cast<char>(packed[offset / 4u] >> ((offset % 4u) * 8u)));
          std::fprintf(stderr,
                       "resident_plan_topic unit=%u order=%u hits=%u revision=%u "
                       "text='%s'\n",
                       unit, cue_orders[unit], recent_topic_hits(topic_key),
                       recent_topic_revision(topic_key),
                       text.c_str());
        }
      }
      if (state.adult.question_gap_field_support.get() != nullptr) {
        std::vector<std::uint32_t> gap_support(
            static_cast<std::size_t>(state.adult.unit_count) *
            adult::construction::kRelationFieldCount);
        std::vector<std::uint32_t> cue_exact(state.adult.unit_count);
        adult::cuda_require(
            cudaMemcpy(gap_support.data(),
                       state.adult.question_gap_field_support.get(),
                       gap_support.size() * sizeof(std::uint32_t),
                       cudaMemcpyDeviceToHost),
            "read observer question-gap support");
        adult::cuda_require(
            cudaMemcpy(cue_exact.data(), state.adult.relation_cue_exact.get(),
                       cue_exact.size() * sizeof(std::uint32_t),
                       cudaMemcpyDeviceToHost),
            "read observer question-gap cue");
        std::uint32_t rows = 0u;
        std::uint32_t cue_rows = 0u;
        for (std::uint32_t unit = 0u; unit < state.adult.unit_count; ++unit) {
          bool populated = false;
          for (std::uint32_t field = 0u;
               field < adult::construction::kRelationFieldCount; ++field)
            populated |=
                gap_support[
                    unit * adult::construction::kRelationFieldCount + field] !=
                0u;
          rows += populated;
          cue_rows += populated && cue_exact[unit] != 0u;
        }
        std::fprintf(stderr,
                     "resident_plan_question_gap rows=%u cue_rows=%u\n",
                     rows, cue_rows);
      }
      {
        const std::uint32_t cursor = witnessed_event_cursor;
        const std::uint32_t extent =
            std::min<std::uint32_t>(
                cursor, adult::construction::kWitnessedRelationEventCap);
        const std::uint32_t first =
            cursor > adult::construction::kWitnessedRelationEventCap
                ? cursor &
                      (adult::construction::kWitnessedRelationEventCap - 1u)
                : 0u;
        std::vector<adult::construction::WitnessedRelationEvent> events(
            adult::construction::kWitnessedRelationEventCap);
        std::vector<std::uint32_t> event_constructions(
            adult::construction::kWitnessedRelationEventCap);
        std::vector<std::uint32_t> cue_exact(state.adult.unit_count);
        std::vector<std::uint32_t> closed_class(state.adult.unit_count);
        std::uint64_t current_revision = 0u;
        adult::cuda_require(
            cudaMemcpy(events.data(),
                       state.adult.witnessed_relation_events.get(),
                       events.size() * sizeof(events[0]),
                       cudaMemcpyDeviceToHost),
            "read observer witnessed relation events");
        adult::cuda_require(
            cudaMemcpy(event_constructions.data(),
                       state.adult.witnessed_relation_constructions.get(),
                       event_constructions.size() *
                           sizeof(event_constructions[0]),
                       cudaMemcpyDeviceToHost),
            "read observer witnessed relation constructions");
        adult::cuda_require(
            cudaMemcpy(cue_exact.data(), state.adult.relation_cue_exact.get(),
                       cue_exact.size() * sizeof(cue_exact[0]),
                       cudaMemcpyDeviceToHost),
            "read observer witnessed relation cue");
        adult::cuda_require(
            cudaMemcpy(closed_class.data(),
                       state.adult.construction_closed_class_mask.get(),
                       closed_class.size() * sizeof(closed_class[0]),
                       cudaMemcpyDeviceToHost),
            "read observer witnessed relation glue mask");
        adult::cuda_require(
            cudaMemcpy(&current_revision,
                       state.adult.proposition_ordered_evidence_revision.get(),
                       sizeof(current_revision), cudaMemcpyDeviceToHost),
            "read observer current contact revision");
        std::uint64_t best_revision = 0u;
        std::uint32_t best_hits = 0u;
        std::uint32_t best_events = 0u;
        std::uint32_t best_constructions = 0u;
        std::uint32_t best_event = adult::construction::kNoTripleUnit;
        std::uint64_t group_revision = 0u;
        std::vector<std::uint32_t> group_hits;
        std::uint32_t group_events = 0u;
        std::uint32_t group_constructions = 0u;
        std::uint32_t group_event = adult::construction::kNoTripleUnit;
        auto finish_group = [&]() {
          if (group_revision != 0u &&
              group_revision != current_revision &&
              (group_hits.size() > best_hits ||
               (group_hits.size() == best_hits &&
                group_revision > best_revision))) {
            best_revision = group_revision;
            best_hits = static_cast<std::uint32_t>(group_hits.size());
            best_events = group_events;
            best_constructions = group_constructions;
            best_event = group_event;
          }
          group_revision = 0u;
          group_hits.clear();
          group_events = 0u;
          group_constructions = 0u;
          group_event = adult::construction::kNoTripleUnit;
        };
        for (std::uint32_t offset = 0u; offset < extent; ++offset) {
          const std::uint32_t index =
              (first + offset) &
              (adult::construction::kWitnessedRelationEventCap - 1u);
          const auto& event = events[index];
          if (event.live == 0u || event.evidence_revision == 0u) continue;
          const std::uint64_t revision = event.evidence_revision >> 32u;
          if (group_revision != 0u && revision != group_revision)
            finish_group();
          if (group_revision == 0u) group_revision = revision;
          if (group_event == adult::construction::kNoTripleUnit)
            group_event = index;
          ++group_events;
          group_constructions +=
              event_constructions[index] !=
              adult::construction::kNoConstruction;
          const std::uint32_t units[4] = {
              event.triple.subject, event.triple.connective,
              event.triple.connective2, event.triple.value};
          for (const std::uint32_t unit : units) {
            if (unit >= state.adult.unit_count ||
                unit == adult::construction::kNoTripleUnit ||
                cue_exact[unit] == 0u)
              continue;
            if (std::find(group_hits.begin(), group_hits.end(), unit) ==
                group_hits.end())
              group_hits.push_back(unit);
          }
        }
        finish_group();
        auto event_unit_text = [&](std::uint32_t unit) {
          if (unit >= state.adult.unit_count)
            return std::string("<none>");
          std::uint32_t length = 0u;
          std::array<std::uint32_t, adult::kUnitWords> content{};
          adult::cuda_require(
              cudaMemcpy(&length, state.adult.unit_lengths.get() + unit,
                         sizeof(length), cudaMemcpyDeviceToHost),
              "read observer witnessed unit length");
          adult::cuda_require(
              cudaMemcpy(content.data(),
                         state.adult.unit_content.get() +
                             unit * adult::kUnitWords,
                         sizeof(content), cudaMemcpyDeviceToHost),
              "read observer witnessed unit content");
          std::string text;
          for (std::uint32_t offset = 0u;
               offset < length && offset < adult::kUnitWords * 4u; ++offset)
            text.push_back(static_cast<char>(
                content[offset / 4u] >> ((offset % 4u) * 8u)));
          return text;
        };
        if (best_event != adult::construction::kNoTripleUnit) {
          const auto& event = events[best_event];
          std::fprintf(
              stderr,
              "resident_plan_witness_ohm revision=%llu hits=%u events=%u constructions=%u text='%s | %s | %s | %s'\n",
              static_cast<unsigned long long>(best_revision), best_hits,
              best_events, best_constructions,
              event_unit_text(event.triple.subject).c_str(),
              event_unit_text(event.triple.connective).c_str(),
              event_unit_text(event.triple.connective2).c_str(),
              event_unit_text(event.triple.value).c_str());
        }
        if (relation_plan_receipt.best_event <
            adult::construction::kWitnessedRelationEventCap) {
          const auto& selected =
              events[relation_plan_receipt.best_event];
          std::vector<std::uint32_t> group_units;
          for (const auto& event : events) {
            if (event.live == 0u ||
                (event.evidence_revision >> 32u) !=
                    (selected.evidence_revision >> 32u) ||
                event.segment_begin != selected.segment_begin)
              continue;
            const std::uint32_t units[5] = {
                event.triple.subject, event.triple.connective,
                event.triple.connective2, event.triple.value,
                event.terminal};
            for (const std::uint32_t unit : units) {
              if (unit >= state.adult.unit_count ||
                  unit == adult::construction::kNoTripleUnit ||
                  std::find(group_units.begin(), group_units.end(), unit) !=
                      group_units.end())
                continue;
              group_units.push_back(unit);
            }
          }
          const std::size_t visible =
              std::min<std::size_t>(group_units.size(), 20u);
          std::fprintf(stderr,
                       "resident_plan_witness_group units=%zu shown=%zu",
                       group_units.size(), visible);
          for (std::size_t index = 0u; index < visible; ++index) {
            const std::uint32_t unit = group_units[index];
            std::fprintf(stderr, " [%s cue=%u glue=%u]",
                         event_unit_text(unit).c_str(),
                         cue_exact[unit] != 0u,
                         closed_class[unit] != 0u);
          }
          std::fprintf(stderr, "\n");
        }
      }
      // The relation cohort is selected entirely on device.  Keep this
      // bounded observer here so a RED grounded-language run can distinguish
      // a missing taught fact from an incorrect device tournament winner.
      // relation_triple_cursor is reused by post-contact learning statistics.
      // The receipt is the frozen pre-contact cohort extent; reading the
      // cursor here would label stale slots as current candidates.
      const std::uint32_t relation_dump_count =
          std::min<std::uint32_t>(diagnostic.relation_candidate_count,
                                  adult::construction::kRelationTripleCandidateCap);
      if (relation_dump_count != 0u) {
        std::vector<std::uint32_t> relation_candidates(relation_dump_count);
        adult::cuda_require(cudaMemcpy(relation_candidates.data(),
                                       state.adult.relation_triple_candidates.get(),
                                       relation_candidates.size() * sizeof(std::uint32_t),
                                       cudaMemcpyDeviceToHost),
                            "read observer relation candidates");
        auto relation_unit_text = [&](std::uint32_t unit) {
          if (unit >= state.adult.unit_count) return std::string("<none>");
          std::uint32_t length = 0u;
          std::array<std::uint32_t, adult::kUnitWords> content{};
          adult::cuda_require(cudaMemcpy(&length, state.adult.unit_lengths.get() + unit,
                                         sizeof(length), cudaMemcpyDeviceToHost),
                              "read observer relation candidate length");
          adult::cuda_require(cudaMemcpy(content.data(),
                                         state.adult.unit_content.get() +
                                             unit * adult::kUnitWords,
                                         sizeof(content), cudaMemcpyDeviceToHost),
                              "read observer relation candidate content");
          std::string text;
          for (std::uint32_t offset = 0u;
               offset < length && offset < adult::kUnitWords * 4u; ++offset) {
            text.push_back(static_cast<char>(
                content[offset / 4u] >> ((offset % 4u) * 8u)));
          }
          return text;
        };
        constexpr std::uint32_t kRelationDiagCap = 24u;
        const std::uint32_t visible =
            std::min<std::uint32_t>(relation_dump_count, kRelationDiagCap);
        std::fprintf(stderr, "resident_plan_relation_cohort total=%u shown=%u\n",
                     relation_dump_count, visible);
        for (std::uint32_t index = 0u; index < visible; ++index) {
          const std::uint32_t entry = relation_candidates[index];
          const std::uint32_t slot =
              entry & ~(0x80000000u | adult::construction::kAnalogyFlag);
          adult::construction::RelationTriple triple{};
          std::uint32_t count = 0u;
          adult::cuda_require(cudaMemcpy(&triple,
                                         state.adult.relation_triples.get() + slot,
                                         sizeof(triple), cudaMemcpyDeviceToHost),
                              "read observer relation candidate triple");
          adult::cuda_require(cudaMemcpy(&count,
                                         state.adult.relation_triple_counts.get() + slot,
                                         sizeof(count), cudaMemcpyDeviceToHost),
                              "read observer relation candidate count");
          std::fprintf(stderr,
                       "resident_plan_relation_candidate index=%u count=%u forward=%u analogy=%u text='%s | %s | %s | %s'\n",
                       index, count, (entry & 0x80000000u) == 0u ? 1u : 0u,
                       (entry & adult::construction::kAnalogyFlag) != 0u ? 1u : 0u,
                       relation_unit_text(triple.subject).c_str(),
                       relation_unit_text(triple.connective).c_str(),
                       relation_unit_text(triple.connective2).c_str(),
                       relation_unit_text(triple.value).c_str());
        }
      }
      if (diagnostic.relation_plan_units >= 3u &&
          diagnostic.relation_plan_units <= adult::construction::kCommitmentCap) {
        std::vector<std::uint32_t> relation_units(diagnostic.relation_plan_units);
        adult::cuda_require(cudaMemcpy(relation_units.data(),
                                       state.adult.relation_triple_plan.get(),
                                       relation_units.size() * sizeof(std::uint32_t),
                                       cudaMemcpyDeviceToHost),
                            "read observer relation plan");
        std::string relation_text;
        for (const std::uint32_t unit : relation_units) {
          if (unit >= state.adult.unit_count) continue;
          std::uint32_t length = 0u;
          std::array<std::uint32_t, adult::kUnitWords> content{};
          adult::cuda_require(cudaMemcpy(&length, state.adult.unit_lengths.get() + unit,
                                         sizeof(length), cudaMemcpyDeviceToHost),
                              "read observer relation unit length");
          adult::cuda_require(cudaMemcpy(content.data(),
                                         state.adult.unit_content.get() +
                                             unit * adult::kUnitWords,
                                         sizeof(content), cudaMemcpyDeviceToHost),
                              "read observer relation unit content");
          if (!relation_text.empty()) relation_text.push_back(' ');
          for (std::uint32_t offset = 0u;
               offset < length && offset < adult::kUnitWords * 4u; ++offset) {
            relation_text.push_back(static_cast<char>(
                content[offset / 4u] >> ((offset % 4u) * 8u)));
          }
        }
        std::fprintf(stderr, "resident_plan_relation units=%u text='%s'\n",
                     diagnostic.relation_plan_units, relation_text.c_str());
      }
      if (diagnostic.selected_binding_index <
          state.adult.proposition_ordered_bindings.size()) {
        adult::proposition_tissue::OrderedRoleBindingEvidence binding{};
        adult::cuda_require(cudaMemcpy(
                            &binding,
                            state.adult.proposition_ordered_bindings.get() +
                                diagnostic.selected_binding_index,
                            sizeof(binding), cudaMemcpyDeviceToHost),
                            "read observer selected binding");
        std::fprintf(stderr, "resident_plan_binding index=%u roles=",
                     diagnostic.selected_binding_index);
        for (std::uint32_t role = 0u;
             role < adult::proposition_tissue::kOrderedBindingRoleCount; ++role) {
          std::fprintf(stderr, "%s", role == 0u ? "" : ";");
          for (std::uint32_t identity = 0u;
               identity < binding.role_unit_counts[role]; ++identity) {
            const std::uint32_t unit = binding.role_units[role][identity];
            std::uint32_t length = 0u;
            std::array<std::uint32_t, adult::kUnitWords> packed{};
            if (unit < state.adult.unit_count) {
              adult::cuda_require(cudaMemcpy(&length,
                                               state.adult.unit_lengths.get() + unit,
                                               sizeof(length), cudaMemcpyDeviceToHost),
                                  "read observer binding unit length");
              adult::cuda_require(cudaMemcpy(packed.data(),
                                               state.adult.unit_content.get() +
                                                   static_cast<std::size_t>(unit) * adult::kUnitWords,
                                               packed.size() * sizeof(std::uint32_t),
                                               cudaMemcpyDeviceToHost),
                                  "read observer binding unit bytes");
            }
            std::array<char, adult::kUnitWords * 4u + 1u> text{};
            const std::uint32_t bounded = std::min(length, adult::kUnitWords * 4u);
            for (std::uint32_t byte = 0u; byte < bounded; ++byte)
              text[byte] = static_cast<char>(
                  (packed[byte / 4u] >> ((byte % 4u) * 8u)) & 0xffu);
            std::fprintf(stderr, "%s%u:'%s'", identity == 0u ? "" : ",",
                         unit, text.data());
          }
        }
        std::fprintf(stderr, "\n");
      }
    }
  }

  std::uint32_t counts[2] = {};
  if (staged_emission) {
    emission_publication->copy_from_device_and_wait(
        state.transport_counts.get(), state.egress.get(), state.emission_capacity);
    emission_publication->read_counts(counts);
  } else {
    adult::cuda_require(cudaMemcpy(counts, state.transport_counts.get(),
                                   sizeof(counts), cudaMemcpyDeviceToHost),
                        "transport resident emission extent");
  }
  if (counts[0] > state.emission_capacity || counts[1] > counts[0]) {
    throw std::runtime_error("invalid resident emission extent");
  }
  TickResult result;
  result.input_free = byte_count == 0u;
  if (counts[0] != 0u) {
    if (staged_emission) {
      const std::uint8_t* payload = emission_publication->payload();
      result.bytes.assign(payload, payload + counts[0]);
    } else {
      result.bytes.resize(counts[0]);
      adult::cuda_require(cudaMemcpy(result.bytes.data(), state.egress.get(),
                                     counts[0], cudaMemcpyDeviceToHost),
                          "transport resident emission bytes");
    }
  }
  const char* const plan_egress_diag = resident_plan_diag;
  if (plan_egress_diag != nullptr && plan_egress_diag[0] == '1' &&
      query_before_surface.attempted != 0u && !result.bytes.empty()) {
    std::fprintf(stderr, "resident_plan_egress bytes=%u hex=",
                 static_cast<unsigned>(result.bytes.size()));
    const std::size_t hex_extent = std::min<std::size_t>(result.bytes.size(), 48u);
    for (std::size_t index = 0u; index < hex_extent; ++index)
      std::fprintf(stderr, "%02x", static_cast<unsigned>(result.bytes[index]));
    std::fprintf(stderr, "\n");
  }
  tick_phase_mark("egress_transport");
  // Every actual emitted surface is reconditioned as opaque sensory material
  // after the performed action is captured. The retained source population,
  // not the presence of input on THIS tick, determines whether an action can
  // open a trajectory: capture_emitted_action_kernel already fails closed
  // when trace->source_byte_count == 0.
  //
  // A contact may establish the resident source, one or more input-free ticks
  // may follow, and only then may the resident cross the motor boundary. The
  // old byte_count != 0 guard discarded precisely those autonomous actions,
  // making a later external consequence impossible to bind to them.
  if (counts[0] != 0u) {
    // Capture before reafferent conditioning clears the per-tick motor
    // workspace.  motor_completion/motor_context[3] is the resident sequence
    // that produced the action (including the input-free generator's newly
    // preserved unit sequence), so the trajectory records performed action
    // provenance rather than reconstructing it from the returned bytes.
    auto* transport_counts = state.transport_counts.get();
    auto* egress = state.egress.get();
    auto surface_population =
        adult::resident_surface_population_view(state.adult);
    auto* motor_completion = state.adult.motor_completion.get();
    auto* motor_context_extent = state.adult.motor_context.get() + 3u;
    auto* pending_action_trajectory = state.pending_action_trajectory.get();
    auto* query_answer_receipt = state.query_answer_receipt.get();
    auto* question_goal_state = state.adult.question_goal_state.get();
    auto* query_plan = state.query_plan.get();
    void* action_capture_arguments[] = {
        &transport_counts, &egress, &surface_population, &motor_completion,
        &motor_context_extent, &pending_action_trajectory, &query_answer_receipt,
        &question_goal_state, &query_plan};
    adult::cuda_require(
        cudaLaunchKernel(
            reinterpret_cast<const void*>(capture_emitted_action_kernel),
            dim3{1u, 1u, 1u}, dim3{2u, 1u, 1u}, action_capture_arguments, 0u,
            nullptr),
        "launch capture emitted resident action");
    adult::cuda_require(cudaGetLastError(), "capture emitted resident action");
    // Reafferent conditioning may clear the motor workspace; it must happen
    // only after the action's resident sequence has entered the trajectory.
    adult::condition_on_raw_cue(state.adult, result.bytes.data(), counts[0], false,
                                true, true);
  }
  tick_phase_mark("emitted_action_capture");
  if (counts[1] != 0u) {
    contact_host_bytes(state, result.bytes.data(), state.egress.get(), counts[1], true);
  }
  tick_phase_mark("reafferent_contact");
  return result;
}

inline std::uint32_t pending_streaming_contact_bytes(
    const StreamState& state) {
  if (!state.adult.streaming_cue_mode) return 0u;
  std::uint32_t meta[3] = {};
  adult::cuda_require(
      cudaMemcpy(meta, state.adult.streaming_cue_meta.get(), sizeof(meta),
                 cudaMemcpyDeviceToHost),
      "read contact transaction extent");
  return meta[0];
}

inline std::vector<std::uint8_t> append_streaming_contact_for_mode(
    adult::AdultState& state, const std::uint8_t* host_bytes,
    std::uint32_t byte_count, bool flush_contact,
    bool complete_on_closure) {
  // Transport staging buffer for ONE fragment. It used to be constructed and
  // destroyed per call, i.e. one cudaMalloc plus one cudaFree -- both
  // device-wide synchronizing operations -- on every tick of resident-language
  // mode, before any resident work could be launched. The buffer holds no
  // resident state: it is written by the host copy below, consumed by the
  // launch that follows, and dead again by the blocking meta readback further
  // down, so reusing one grow-only allocation is byte-for-byte equivalent to
  // reallocating it. Kept function-local rather than moved into AdultState so
  // the resident state layout and its checkpoints are untouched; this driver
  // is single-writer host code on the default stream, so one instance is
  // enough.
  static adult::DeviceArray<std::uint8_t> incoming;
  const std::uint32_t staging_extent = byte_count == 0u ? 1u : byte_count;
  if (incoming.size() < staging_extent) incoming.allocate(staging_extent);
  if (byte_count != 0u) {
    adult::cuda_require(cudaMemcpy(incoming.get(), host_bytes, byte_count,
                                   cudaMemcpyHostToDevice),
                        "upload streaming contact fragment");
  }
  adult::append_streaming_cue_kernel<<<1u, 1u>>>(
      incoming.get(), byte_count, state.construction_closure_bytes.get(),
      state.construction_closure_count, complete_on_closure, flush_contact,
      state.streaming_cue_bytes.get(), adult::kStreamingCueCapacity,
      state.streaming_cue_meta.get());
  adult::cuda_require(cudaGetLastError(), "append resident streaming contact");
  std::uint32_t meta[3] = {};
  adult::cuda_require(cudaMemcpy(meta, state.streaming_cue_meta.get(),
                                 sizeof(meta), cudaMemcpyDeviceToHost),
                      "read resident streaming contact state");
  if (meta[2] != 0u)
    throw std::runtime_error("resident streaming contact capacity exceeded");
  if (meta[1] == 0u || meta[0] == 0u) return {};
  std::vector<std::uint8_t> complete(meta[0]);
  adult::cuda_require(cudaMemcpy(complete.data(), state.streaming_cue_bytes.get(),
                                 complete.size(), cudaMemcpyDeviceToHost),
                      "transport completed resident contact");
  adult::cuda_require(cudaMemset(state.streaming_cue_meta.get(), 0,
                                 state.streaming_cue_meta.bytes()),
                      "consume completed resident contact");
  return complete;
}

inline TickResult tick(StreamState& state, const std::uint8_t* host_bytes,
                       std::uint32_t byte_count,
                       bool preassembled_contact = false,
                       std::int32_t efferent_polarity = 0,
                       bool outcome_present = true,
                       bool initial_exposure = false) {
  const bool physical_learning = conditioned_credit_owner_active(state);
  const bool explicit_flush_mode =
      state.explicit_streaming_flush && state.adult.streaming_cue_mode &&
      !preassembled_contact;
  // Reading the resident transient extent is a blocking device->host copy, so
  // it drains the launch pipeline before the tick has issued any work. Every
  // consumer of this value below is itself gated on explicit_flush_mode or on
  // physical_learning (the three explicit-flush branches, and the capacity
  // snapshot conjunction which also requires physical_learning), so with
  // neither active the value is unreadable by construction and the transfer is
  // pure latency. Skipping it there is exact, not approximate: a resident
  // extent that no branch can consult cannot change this tick.
  const std::uint32_t pending_before =
      (explicit_flush_mode || physical_learning)
          ? pending_streaming_contact_bytes(state)
          : 0u;
  const bool incomplete_explicit_fragment =
      explicit_flush_mode && byte_count != 0u;
  const bool explicit_flush_contact =
      explicit_flush_mode && byte_count == 0u && pending_before != 0u;
  if (explicit_flush_contact && physical_learning &&
      static_cast<std::uint64_t>(pending_before) >
          state.conditioned_device_owner.remaining_capacity()) {
    // The transient contact is untouched. Refuse before checkpointing or
    // launching completed-contact work; a later retry may use a larger bank.
    throw std::runtime_error(
        "explicit streaming flush exceeds conditioned resident capacity");
  }
  // Every conditioned event is induced by at least one input byte, and a
  // negative event names an already-bound competitor. Thus byte extent is a
  // conservative upper bound on new physical pages. Snapshot only when this
  // contact can exceed the remaining bank; the ordinary path stays allocation
  // and checkpoint free.
  const std::uint64_t maximum_contact_bound =
      preassembled_contact ? byte_count
                           : explicit_flush_mode
                                 ? static_cast<std::uint64_t>(pending_before)
                                 : adult::kStreamingCueCapacity;
  if (!incomplete_explicit_fragment &&
      state.contact_rollback_checkpoint.empty() && pending_before == 0u &&
      physical_learning &&
      maximum_contact_bound >
          state.conditioned_device_owner.remaining_capacity()) {
    state.contact_rollback_checkpoint = contact_rollback_path(state);
    try {
      save_checkpoint(state, state.contact_rollback_checkpoint);
    } catch (...) {
      std::remove(state.contact_rollback_checkpoint.c_str());
      state.contact_rollback_checkpoint.clear();
      throw;
    }
  }

  try {
    TickResult result =
        tick_untransactional(state, host_bytes, byte_count,
                             preassembled_contact, efferent_polarity,
                             outcome_present, initial_exposure);
    if (!state.contact_rollback_checkpoint.empty() &&
        pending_streaming_contact_bytes(state) == 0u) {
      std::remove(state.contact_rollback_checkpoint.c_str());
      state.contact_rollback_checkpoint.clear();
    }
    return result;
  } catch (...) {
    const std::exception_ptr failure = std::current_exception();
    if (!state.contact_rollback_checkpoint.empty()) {
      const std::string rollback_path = state.contact_rollback_checkpoint;
      try {
        StreamState restored = load_checkpoint(rollback_path);
        // The transaction is always staged before the first fragment, so the
        // snapshot already owns the exact pre-contact transient bytes and
        // extent as well as learned matter.
        state = std::move(restored);
      } catch (...) {
        std::remove(rollback_path.c_str());
        throw std::runtime_error(
            "adult contact failed and its capacity rollback could not restore");
      }
      std::remove(rollback_path.c_str());
    }
    std::rethrow_exception(failure);
  }
}

inline TickResult tick(StreamState& state, const std::vector<std::uint8_t>& bytes) {
  return tick(state, bytes.data(), static_cast<std::uint32_t>(bytes.size()));
}

inline TickResult tick(StreamState& state) { return tick(state, nullptr, 0u); }

inline StreamReport read_report(const StreamState& state) {
  StreamReport report{};
  adult::cuda_require(cudaMemcpy(&report.drive, state.drive.get(), sizeof(report.drive),
                                 cudaMemcpyDeviceToHost),
                      "read resident stream drive");
  adult::cuda_require(cudaMemcpy(&report.appraisal, state.appraisal.get(),
                                 sizeof(report.appraisal), cudaMemcpyDeviceToHost),
                      "read resident stream appraisal");
  report.adult = adult::read_report(state.adult);
  adult::cuda_require(cudaMemcpy(&report.query, state.query_answer_receipt.get(),
                                 sizeof(report.query), cudaMemcpyDeviceToHost),
                      "read resident query answer receipt");
  report.stream_resident_bytes = state.adult.resident_bytes + state.drive.bytes() +
      state.appraisal.bytes() + state.appraisal_workspace.bytes() +
      state.contact_summary.bytes() +
      state.ingress.bytes() + state.candidate.bytes() + state.egress.bytes() +
      state.generated_count.bytes() + state.transport_counts.bytes() +
      state.candidate_rng.bytes() + state.query_plan.bytes() +
      state.query_settlement.bytes() + state.query_answer_receipt.bytes() +
      state.ordered_relation_output_units.bytes() +
      state.ordered_relation_output_count.bytes() +
      state.ordered_relation_execution_receipt.bytes() +
      state.query_grounding.bytes() + state.query_plan_grounding_observer.bytes() +
      state.query_topic_support.bytes() +
      state.query_topic_key.bytes() +
      state.query_surface_selection.bytes() +
      state.query_surface_anchors.bytes() +
      state.query_surface_anchor_counts.bytes() +
      state.query_relation_evidence_revision.bytes() +
      state.query_relation_evidence_count.bytes() +
      state.query_relation_plan_receipt.bytes() +
      state.query_surface_transaction.bytes() + state.pending_action_trajectory.bytes() +
      state.action_transitions.bytes() + state.action_transition_scalars.bytes();
  report.query_topic_support_slots = state.query_topic_support.size();
  return report;
}

inline void lesion_drive(StreamState& state) {
  lesion_drive_kernel<<<1u, 32u>>>(state.drive.get());
  adult::cuda_require(cudaGetLastError(), "launch resident drive lesion");
  adult::cuda_require(cudaDeviceSynchronize(), "complete resident drive lesion");
}

inline void lesion_appraisal(StreamState& state) {
  appraisal::lesion_appraisal_kernel<<<1u, 32u>>>(state.appraisal.get());
  adult::cuda_require(cudaGetLastError(), "launch resident appraisal lesion");
  adult::cuda_require(cudaDeviceSynchronize(), "complete resident appraisal lesion");
}

inline void lesion_plasticity(StreamState& state) {
  lesion_plasticity_flag_kernel<<<1u, 32u>>>(state.drive.get());
  adult::cuda_require(cudaGetLastError(), "launch resident plasticity flag lesion");
  adult::cuda_require(cudaDeviceSynchronize(), "complete resident plasticity flag lesion");
  state.plasticity_disabled = true;
  adult::lesion_transition_state(state.adult);
  adult::lesion_episode_state(state.adult);
  adult::lesion_boundary_state(state.adult);
  adult::lesion_counts_kernel<<<adult::blocks_for(state.adult.unit_count), adult::kBlock>>>(
      state.adult.unit_vitality.get(), state.adult.unit_count, state.adult.ledger.get());
  adult::lesion_counts_kernel<<<adult::blocks_for(state.adult.online_association_count),
                                  adult::kBlock>>>(
      state.adult.online_association_counts.get(), state.adult.online_association_count,
      state.adult.ledger.get());
  adult::cuda_require(cudaMemset(state.adult.motor_context.get(), 0,
                                 state.adult.motor_context.bytes()),
                      "clear plasticity-lesioned motor context");
  adult::cuda_require(cudaMemset(state.adult.motor_completion.get(), 0,
                                 state.adult.motor_completion.bytes()),
                      "clear plasticity-lesioned motor completion");
  build_unigram_top_parallel_kernel<<<1u, adult::kBlock>>>(
      state.adult.unit_vitality.get(), state.adult.unit_count,
      state.adult.unigram_top_ids.get());
  adult::audit_ledger_kernel<<<1u, adult::kBlock>>>(
      state.adult.unit_vitality.get(), state.adult.unit_count,
      state.adult.bigram_counts.get(), state.adult.bigram_count,
      state.adult.trigram_counts.get(), state.adult.trigram_count,
      state.adult.online_bigram_counts.get(), state.adult.online_bigram_count,
      state.adult.online_trigram_counts.get(), state.adult.online_trigram_count,
      state.adult.online_association_counts.get(), state.adult.online_association_count,
      state.adult.online_conditioned_transition_counts.get(),
      state.adult.online_conditioned_transition_count,
      0u, state.adult.boundary_histogram.get(), state.adult.boundary_pairs.get(),
      state.adult.ledger.get());
  adult::cuda_require(cudaGetLastError(), "launch resident plasticity lesion");
  adult::cuda_require(cudaDeviceSynchronize(), "complete resident plasticity lesion");
}

#else

#endif  // !defined(BCC32_CUDA_ADULT_STREAM_CHECKPOINT_ONLY)
