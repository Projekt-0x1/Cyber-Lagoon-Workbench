// Bounded observer and pre-contact relation-surface freeze for the resident
// query Plan. This file is included inside the stream namespace after all
// kernels used by the freeze phase have been declared.

__global__ void capture_plan_anchor_grounding_pre_kernel(
    const discourse_plan::ResidentDiscoursePlanState* plan,
    PlanAnchorGroundingObserverReceipt* observer) {
  if (blockIdx.x != 0u || threadIdx.x != 0u || observer == nullptr) return;
  *observer = PlanAnchorGroundingObserverReceipt{};
  observer->materializer_ambiguous_step = population_surface::kInvalidUnit;
  observer->materializer_ungrounded_step = population_surface::kInvalidUnit;
  if (plan == nullptr) return;

  observer->pre_valid = discourse_plan::valid(*plan) ? 1u : 0u;
  observer->pre_status = static_cast<std::uint32_t>(plan->status);
  observer->pre_step_count = plan->step_count;
  observer->pre_anchor_count = plan->anchor_reference_count;
  observer->pre_population_references = plan->population_reference_count;
  observer->pre_question_goal_dependency = plan->question_goal_dependency;
  observer->pre_plan_revision = plan->revision;
  const std::uint32_t step_count = min(plan->step_count,
                                       discourse_plan::kMaxSteps);
  for (std::uint32_t step = 0u; step < step_count; ++step) {
    const auto& current = plan->steps[step];
    if (current.reference_kind ==
        discourse_plan::PlanReferenceKind::opaque_population) {
      ++observer->pre_opaque_steps;
      if (current.population_count == 0u)
        ++observer->pre_zero_population_steps;
    } else if (current.reference_kind ==
               discourse_plan::PlanReferenceKind::ordered_binding) {
      ++observer->pre_ordered_steps;
    } else {
      ++observer->pre_other_steps;
    }
  }
}

__global__ void capture_plan_anchor_grounding_post_kernel(
    const discourse_plan::ResidentDiscoursePlanState* plan,
    const population_surface::GroundingResult* grounding,
    PlanAnchorGroundingObserverReceipt* observer) {
  if (blockIdx.x != 0u || threadIdx.x != 0u || observer == nullptr) return;
  if (grounding != nullptr) {
    observer->materializer_ready = grounding->ready;
    observer->materializer_grounded_steps = grounding->grounded_steps;
    observer->materializer_anchor_count = grounding->anchor_count;
    observer->materializer_ambiguous_step = grounding->ambiguous_step;
    observer->materializer_ungrounded_step = grounding->ungrounded_step;
    observer->materializer_weakest_overlap = grounding->weakest_overlap;
    observer->materializer_plan_revision = grounding->plan_revision;
  }
  if (plan != nullptr) {
    observer->post_status = static_cast<std::uint32_t>(plan->status);
    observer->post_anchor_count = plan->anchor_reference_count;
    observer->post_plan_revision = plan->revision;
  }
}

__global__ void capture_plan_anchor_grounding_final_kernel(
    const discourse_plan::ResidentDiscoursePlanState* plan,
    const QueryAnswerReceipt* receipt,
    PlanAnchorGroundingObserverReceipt* observer) {
  if (blockIdx.x != 0u || threadIdx.x != 0u || observer == nullptr) return;
  if (receipt != nullptr) {
    observer->final_attempted = receipt->attempted;
    observer->final_staged = receipt->staged;
    observer->final_anchor_count = receipt->anchor_count;
  }
  if (plan != nullptr) {
    observer->final_plan_valid = discourse_plan::valid(*plan) ? 1u : 0u;
    observer->final_plan_status = static_cast<std::uint32_t>(plan->status);
  }
}

template <typename TickPhaseMark>
inline void freeze_query_relation_surface_phase(
    StreamState& state, const TickPhaseMark& tick_phase_mark) {
  // Resolve every relation-to-construction projection against the resident
  // state that preceded this contact. Assimilation below may grow the adult,
  // but it cannot retroactively authorize the answer currently being formed.
  for (std::uint32_t step = 0u;
       step < adult::construction::kRelationSurfaceEvidenceCap; ++step) {
    std::uint32_t* frozen_selection =
        state.query_surface_selection.get() + step * 4u;
    std::uint32_t* frozen_anchors =
        state.query_surface_anchors.get() +
        step * adult::construction::kConstructionMaxSlots;
    std::uint32_t* frozen_anchor_count =
        state.query_surface_anchor_counts.get() + step;
    select_relation_surface_witness_kernel<<<1u, 1u>>>(
        state.query_relation_evidence_revision.get() + step,
        state.adult.witnessed_relation_events.get(),
        state.adult.witnessed_relation_event_cursor.get(),
        state.adult.witnessed_relation_constructions.get(),
        state.adult.witnessed_relation_surface_units.get(),
        state.adult.witnessed_relation_surface_counts.get(),
        state.adult.relation_cue_exact.get(),
        state.adult.qonset_evidence_revision.get(),
        state.adult.question_gap_field_support.get(),
        state.adult.question_answer_construction.get(),
        state.adult.question_answer_construction_support.get(),
        state.adult.question_answer_slot_mapping.get(),
        state.adult.role_canon_lesioned
            ? nullptr
            : state.adult.construction_role_canon.get(),
        state.adult.unit_count, state.adult.construction_tokens.get(),
        state.adult.construction_lengths.get(),
        state.adult.construction_slot_counts.get(),
        state.adult.construction_closed_class_mask.get(),
        state.adult.construction_slot_units.get(),
        state.adult.construction_slot_masses.get(),
        state.adult.construction_store_count.get(),
        adult::construction::kConstructionCap, frozen_selection, nullptr);
    stage_relation_surface_anchors_kernel<<<1u, 1u>>>(
        state.query_relation_evidence_revision.get() + step,
        state.adult.witnessed_relation_events.get(),
        state.adult.witnessed_relation_event_cursor.get(), frozen_selection,
        state.adult.witnessed_relation_surface_units.get(),
        state.adult.witnessed_relation_surface_counts.get(),
        state.adult.construction_tokens.get(),
        state.adult.construction_lengths.get(),
        state.adult.construction_slot_counts.get(),
        state.adult.construction_slot_units.get(),
        state.adult.construction_slot_masses.get(),
        state.adult.construction_slot_overflow.get(),
        state.adult.construction_roles.get(), state.adult.unit_count,
        frozen_anchors, adult::construction::kConstructionMaxSlots,
        frozen_anchor_count, nullptr);
    freeze_selected_relation_surface_closure_kernel<<<1u, 1u>>>(
        state.query_relation_evidence_revision.get() + step,
        state.adult.construction_tokens.get(),
        state.adult.construction_lengths.get(), frozen_anchors,
        frozen_anchor_count, state.adult.unit_lengths.get(),
        state.adult.unit_content.get(), adult::kUnitWords,
        state.adult.unit_count, state.adult.boundary_mask.get(),
        state.adult.construction_closure_bytes.get(),
        state.adult.construction_closure_count,
        state.adult.construction_slot_units.get(),
        state.adult.construction_slot_masses.get(), frozen_selection,
        state.query_answer_receipt.get());
    project_relation_obligations_to_construction_kernel<<<1u, 1u>>>(
        state.query_relation_evidence_revision.get() + step,
        state.adult.witnessed_relation_events.get(),
        state.adult.witnessed_relation_event_cursor.get(),
        state.adult.construction_tokens.get(),
        state.adult.construction_lengths.get(),
        state.adult.construction_slot_counts.get(),
        state.adult.construction_supports.get(),
        state.adult.construction_origin_revision.get(),
        state.adult.construction_closed_class_mask.get(),
        state.adult.construction_slot_units.get(),
        state.adult.construction_slot_masses.get(),
        state.adult.construction_slot_overflow.get(),
        state.adult.construction_store_count.get(),
        adult::construction::kConstructionCap,
        state.adult.unit_lengths.get(), state.adult.unit_content.get(),
        adult::kUnitWords, state.adult.boundary_mask.get(),
        state.adult.construction_closure_bytes.get(),
        state.adult.construction_closure_count,
        adult::resident_surface_population_view(state.adult),
        frozen_selection, frozen_anchors,
        adult::construction::kConstructionMaxSlots,
        frozen_anchor_count,
        state.query_answer_receipt.get());
  }
  adult::cuda_require(cudaGetLastError(),
                      "freeze pre-contact relation surface projections");
  tick_phase_mark("relation_surface_freeze");
}

template <typename TickPhaseMark>
inline void run_query_plan_grounding_phase(
    StreamState& state, const TickPhaseMark& tick_phase_mark) {
  freeze_query_relation_surface_phase(state, tick_phase_mark);
  capture_plan_anchor_grounding_pre_kernel<<<1u, 1u>>>(
      state.query_plan.get(), state.query_plan_grounding_observer.get());
  // Freeze the opaque construction witness before this contact changes the
  // resident store. The later surface pass may read learned matter, but its
  // authority is the exact prior plan, never a record made by the cue now
  // being answered.
  population_surface::materialize_plan_anchors_kernel<<<1u, 1u>>>(
      state.query_plan.get(), adult::resident_surface_population_view(state.adult),
      state.adult.proposition_output_cells.get(),
      state.adult.proposition_output_scores.get(),
      static_cast<std::uint32_t>(state.adult.proposition_output_cells.size()),
      state.adult.unit_occurrences, state.query_grounding.get());
  capture_plan_anchor_grounding_post_kernel<<<1u, 1u>>>(
      state.query_plan.get(), state.query_grounding.get(),
      state.query_plan_grounding_observer.get());
  // The generic materializer intentionally ignores question_goal references.
  // Reuse the same frozen resident surface population and overwrite the
  // observer receipt only when the Plan's exact live Goal authorizes one
  // unique learned surface unit.
  population_surface::materialize_question_goal_anchor_kernel<<<1u, 1u>>>(
      state.query_plan.get(), state.adult.question_goal_state.get(),
      adult::resident_surface_population_view(state.adult),
      adult::kDistributedMotorPopulation,
      static_cast<std::uint32_t>(state.chronological_bytes),
      state.query_grounding.get());
  population_surface::finalize_ordered_plan_grounding_kernel<<<1u, 1u>>>(
      state.query_plan.get(), state.query_grounding.get());
  finalize_contact_response_grounding_kernel<<<1u, 1u>>>(
      state.query_plan.get(), state.adult.question_goal_state.get(),
      state.query_grounding.get(), state.pending_action_trajectory.get(),
      state.query_answer_receipt.get());
  capture_plan_anchor_grounding_final_kernel<<<1u, 1u>>>(
      state.query_plan.get(), state.query_answer_receipt.get(),
      state.query_plan_grounding_observer.get());
  QueryAnswerReceipt query_before_contact{};
  adult::cuda_require(cudaMemcpy(&query_before_contact,
                                 state.query_answer_receipt.get(),
                                 sizeof(query_before_contact),
                                 cudaMemcpyDeviceToHost),
                      "read resident contact relation authority");
  tick_phase_mark("plan_anchor_grounding");
}
