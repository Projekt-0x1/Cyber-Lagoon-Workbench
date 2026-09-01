// bcc32_cuda_adult_stream_tick_surface_phases.inl
//
// The canonical adult stream keeps relation-plan realization, autonomous
// relation emission, and ordered surface realization on one causal tick.
// This boundary owns those output-stage transformations without changing
// their launch order, 1x1 step loops, or public output authority.
//
// The single-threaded step loops remain an explicit performance RED: each
// iteration depends on the prior resident transaction. Naming the boundary
// makes the future batching/parallelization work measurable and local rather
// than introducing another executive or duplicating StreamState ownership.

inline void realize_relation_surface_phase(
    StreamState& state, const QueryAnswerReceipt& query_before_surface) {
  const bool relation_route_enabled =
      std::getenv("BCC32_DISABLE_RELATION_ROUTE") == nullptr;
  if (query_before_surface.attempted != 0u &&
      query_before_surface.relation_plan_units != 0u &&
      relation_route_enabled &&
      state.adult.surface_organ_enabled &&
      !state.adult.construction_lesioned) {
    adult::cuda_require(cudaMemset(state.adult.surface_result.get(), 0,
                                   state.adult.surface_result.bytes()),
                        "clear relation surface realization");
    const std::uint32_t* relation_evidence_count =
        state.query_relation_evidence_count.get();
    const std::uint32_t* relation_selections = state.query_surface_selection.get();
    SurfaceSpanTransaction* relation_transaction =
        state.query_surface_transaction.get();
    void* relation_span_args[] = {&relation_evidence_count,
                                  &relation_selections,
                                  &relation_transaction};
    adult::cuda_require(cudaLaunchKernel(
                           reinterpret_cast<const void*>(
                               initialize_relation_surface_span_kernel),
                           dim3(1u, 1u, 1u), dim3(128u, 1u, 1u),
                           relation_span_args, 0u, nullptr),
                       "initialize relation surface span");
    const surface_organ::OpaqueConstructionWitnessView relation_witness{
        state.adult.construction_tokens.get(),
        state.adult.construction_lengths.get(),
        state.adult.construction_slot_counts.get(),
        state.adult.construction_supports.get(),
        state.adult.construction_store_count.get(),
        state.adult.construction_roles.get(),
        adult::construction::kConstructionCap,
        state.query_surface_selection.get(),
        state.adult.construction_slot_units.get(),
        state.adult.construction_slot_masses.get(),
        state.adult.construction_slot_overflow.get(),
        state.adult.construction_closed_class_mask.get()};
    const std::uint32_t relation_byte_capacity = std::min(
        state.emission_capacity,
        static_cast<std::uint32_t>(state.adult.surface_output_bytes.size()));
    const surface_organ::SurfaceRealizationWorkspaceView relation_workspace{
        state.adult.surface_bridges.get(), state.adult.surface_prefixes.get(),
        state.adult.surface_suffixes.get(),
        state.adult.surface_permutation_scores.get(),
        state.adult.surface_permutation_valid.get(),
        static_cast<std::uint32_t>(
            state.adult.surface_permutation_scores.size()),
        state.adult.surface_output_units.get(),
        state.adult.surface_output_anchor_mask.get(),
        static_cast<std::uint32_t>(state.adult.surface_output_units.size()),
        state.adult.surface_output_bytes.get(), relation_byte_capacity,
        state.adult.surface_result.get()};
    const bool relation_step_diagnostic =
        std::getenv("BCC32_RESIDENT_PLAN_DIAG") != nullptr;
    for (std::uint32_t step = 0u;
         step < adult::construction::kRelationSurfaceEvidenceCap; ++step) {
      const std::uint32_t* frozen_selection =
          state.query_surface_selection.get() + step * 4u;
      const std::uint32_t* frozen_anchors =
          state.query_surface_anchors.get() +
          step * adult::construction::kConstructionMaxSlots;
      const std::uint32_t* frozen_anchor_count =
          state.query_surface_anchor_counts.get() + step;
      realize_ordered_relation_surface_step_kernel<<<1u, 1u>>>(
          adult::surface_unit_view(state.adult), relation_witness,
          frozen_selection, frozen_anchors, frozen_anchor_count,
          relation_workspace);
      if (relation_step_diagnostic) {
        std::uint64_t revision = 0u;
        std::array<std::uint32_t, 4u> selected{};
        surface_organ::SurfaceOrganResult realized{};
        adult::cuda_require(cudaMemcpy(
                                &revision,
                                state.query_relation_evidence_revision.get() +
                                    step,
                                sizeof(revision), cudaMemcpyDeviceToHost),
                            "read observer relation surface revision");
        adult::cuda_require(cudaMemcpy(
                                selected.data(),
                                frozen_selection,
                                sizeof(selected), cudaMemcpyDeviceToHost),
                            "read observer relation surface selection");
        adult::cuda_require(cudaMemcpy(
                                &realized, state.adult.surface_result.get(),
                                sizeof(realized), cudaMemcpyDeviceToHost),
                            "read observer relation surface result");
        const std::uint32_t diagnostic_bytes =
            realized.output_byte_count <= relation_byte_capacity
                ? realized.output_byte_count
                : 0u;
        std::vector<std::uint8_t> surface(diagnostic_bytes);
        if (!surface.empty())
          adult::cuda_require(cudaMemcpy(
                                  surface.data(),
                                  state.adult.surface_output_bytes.get(),
                                  surface.size(), cudaMemcpyDeviceToHost),
                              "read observer relation surface bytes");
        std::fprintf(
            stderr,
            "resident_plan_relation_surface_step step=%u revision=%llu "
            "event=%u construction=%u transfer=%u slot_map=%u ready=%u anchors=%u bytes=%u "
            "text='%.*s'\n",
            step, static_cast<unsigned long long>(revision), selected[1],
            selected[0], selected[2], selected[3], realized.ready,
            realized.anchors_preserved,
            realized.output_byte_count, static_cast<int>(surface.size()),
            surface.empty() ? "" : reinterpret_cast<const char*>(surface.data()));
      }
      append_relation_surface_step_kernel<<<1u, 1u>>>(
          state.query_relation_evidence_revision.get() + step,
          state.adult.surface_result.get(),
          adult::surface_unit_view(state.adult),
          state.adult.surface_output_units.get(),
          static_cast<std::uint32_t>(state.adult.surface_output_units.size()),
          state.adult.surface_output_bytes.get(),
          state.adult.boundary_mask.get(), state.adult.boundary_bytes.get(),
          state.query_surface_transaction.get(), state.candidate.get(),
          state.emission_capacity);
    }
    finalize_relation_surface_span_kernel<<<1u, 1u>>>(
        state.query_surface_transaction.get(), state.adult.surface_result.get(),
        state.query_answer_receipt.get(), state.generated_count.get());
    if (relation_step_diagnostic) {
      std::uint32_t relation_surface_count = 0u;
      adult::cuda_require(cudaMemcpy(
                              &relation_surface_count,
                              state.generated_count.get(),
                              sizeof(relation_surface_count),
                              cudaMemcpyDeviceToHost),
                          "read observer relation surface transaction extent");
      SurfaceSpanTransaction transaction{};
      adult::cuda_require(
          cudaMemcpy(&transaction, state.query_surface_transaction.get(),
                     sizeof(transaction), cudaMemcpyDeviceToHost),
          "read observer relation surface transaction");
      const std::uint32_t diagnostic_bytes =
          transaction.byte_count <= state.emission_capacity
              ? transaction.byte_count
              : 0u;
      std::vector<std::uint8_t> surface(diagnostic_bytes);
      if (!surface.empty())
        adult::cuda_require(
            cudaMemcpy(surface.data(), state.candidate.get(), surface.size(),
                       cudaMemcpyDeviceToHost),
            "read observer relation surface transaction bytes");
      std::fprintf(
          stderr,
          "resident_plan_relation_surface_transaction expected=%u realized=%u "
          "failed=%u rejected=%u reason=%u bytes=%u anchors=%u evidence=%u units=%u generated=%u "
          "text='%.*s'\n",
          transaction.expected_steps, transaction.realized_steps,
          transaction.failed, transaction.rejected_steps,
          transaction.last_rejection, transaction.byte_count,
          transaction.anchor_count,
          transaction.emitted_evidence_count, transaction.emitted_unit_count,
          relation_surface_count, static_cast<int>(surface.size()),
          surface.empty() ? "" : reinterpret_cast<const char*>(surface.data()));
    }
  }
}

inline void realize_stream_surface_phase(
    StreamState& state, const QueryAnswerReceipt& query_before_surface) {
  if (query_before_surface.attempted != 0u &&
      query_before_surface.staged != 0u &&
      state.adult.surface_organ_enabled &&
      state.adult.construction_lesioned == false &&
      state.adult.surface_output_bytes.get() != nullptr) {
    // A dependency-linked Plan is a bounded ordered span, not an anchor bag.
    // Each step must realize through its own learned construction; the
    // transaction emits only after every selected frame has succeeded.
    adult::cuda_require(cudaMemset(state.adult.surface_result.get(), 0,
                                   state.adult.surface_result.bytes()),
                        "clear prior stream surface realization");
    initialize_stream_surface_span_kernel<<<1u, 1u>>>(
        state.query_plan.get(), state.query_surface_transaction.get(),
        state.generated_count.get());
    const std::uint32_t byte_capacity = std::min(
        state.emission_capacity,
        static_cast<std::uint32_t>(state.adult.surface_output_bytes.size()));
    const surface_organ::OpaqueConstructionWitnessView witness{
        state.adult.construction_tokens.get(),
        state.adult.construction_lengths.get(),
        state.adult.construction_slot_counts.get(),
        state.adult.construction_supports.get(),
        state.adult.construction_store_count.get(),
        state.adult.construction_roles.get(), adult::construction::kConstructionCap,
        state.query_surface_selection.get(), state.adult.construction_slot_units.get(),
        state.adult.construction_slot_masses.get(),
        state.adult.construction_slot_overflow.get(),
        state.adult.construction_closed_class_mask.get()};
    const surface_organ::SurfaceRealizationWorkspaceView workspace{
        state.adult.surface_bridges.get(), state.adult.surface_prefixes.get(),
        state.adult.surface_suffixes.get(),
        state.adult.surface_permutation_scores.get(),
        state.adult.surface_permutation_valid.get(),
        static_cast<std::uint32_t>(state.adult.surface_permutation_scores.size()),
        state.adult.surface_output_units.get(),
        state.adult.surface_output_anchor_mask.get(),
        static_cast<std::uint32_t>(state.adult.surface_output_units.size()),
        state.adult.surface_output_bytes.get(), byte_capacity,
        state.adult.surface_result.get()};
    for (std::uint32_t step = 0u; step < discourse_plan::kMaxSteps; ++step) {
      select_stream_surface_witness_kernel<<<1u, 1u>>>(
          state.query_plan.get(), state.adult.proposition_ordered_construction.get(),
          static_cast<std::uint32_t>(state.adult.proposition_ordered_construction.size()),
          step, state.query_surface_selection.get());
      stage_stream_surface_plan_kernel<<<1u, 1u>>>(
          state.query_plan.get(), state.query_surface_selection.get(),
          state.adult.proposition_ordered_bindings.get(),
          state.adult.proposition_ordered_construction.get(),
          state.adult.construction_slot_counts.get(),
          state.adult.construction_slot_units.get(),
          state.adult.construction_slot_masses.get(),
          state.adult.construction_slot_overflow.get(),
          state.adult.construction_evidence_revision.get(),
          static_cast<std::uint32_t>(state.adult.proposition_ordered_bindings.size()),
          adult::construction::kConstructionCap,
          state.adult.motor_completion.get(),
          static_cast<std::uint32_t>(state.adult.motor_completion.size()),
          state.adult.motor_context.get() + 3u, state.query_answer_receipt.get());
      realize_stream_surface_step_kernel<<<1u, 1u>>>(
          adult::surface_unit_view(state.adult), witness,
          state.adult.motor_completion.get(), state.adult.motor_context.get() + 3u,
          workspace);
      append_stream_surface_step_kernel<<<1u, 1u>>>(
          step, state.adult.surface_result.get(), adult::surface_unit_view(state.adult),
          state.adult.surface_output_units.get(),
          static_cast<std::uint32_t>(state.adult.surface_output_units.size()),
          state.query_surface_transaction.get(), state.candidate.get(),
          state.emission_capacity);
    }
    finalize_stream_surface_span_kernel<<<1u, 1u>>>(
        state.query_surface_transaction.get(), state.adult.surface_result.get(),
        state.query_answer_receipt.get(), state.generated_count.get());
    adult::cuda_require(cudaGetLastError(), "realize ordered stream surface span");
  }
}
