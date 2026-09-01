bool admit_selection_execution_plan(
    const std::filesystem::path& repository,
    const ContentAddress& request_identity,
    ContentAddress* admission_identity,
    ContentAddress* plan_identity,
    std::string* error,
    SelectionExecutorFailurePoint failure) {
    if (admission_identity == nullptr || plan_identity == nullptr) {
        return fail(error, "BCC-32 selection admission requires identity outputs");
    }
    SelectionEvaluationRequest request{};
    ContentAddress operation{};
    ContentAddress resolution_identity{};
    SelectionOperationResolution resolution{};
    SelectionCandidateSet candidate_set{};
    SelectionTranscript transcript{};
    SelectionReplay replay{};
    if (!load_replay_closure(repository, request_identity, &request, &operation,
                             &resolution_identity, &resolution, &candidate_set,
                             &transcript, &replay, error)) {
        return false;
    }
    SelectionExecutionPlan plan{
        .evaluation_request = request_identity,
        .operation_resolution = resolution_identity,
        .transcript = resolution.transcript,
        .scheduler_protocol = request.scheduler_protocol,
        .artifact_kind = request.candidate_artifact_kind,
    };
    plan.tasks.reserve(static_cast<std::size_t>(replay.next_population_slot_count));
    for (std::uint64_t slot = 0u; slot < replay.next_population_slot_count; ++slot) {
        SelectionPopulationSlot resolved{};
        ContentAddress task_head{};
        if (!resolve_selection_population_slot(replay, slot, &resolved, error) ||
            !materialize_selection_population_slot(repository,
                                                   resolution.transcript,
                                                   slot,
                                                   request.candidate_artifact_kind,
                                                   &task_head,
                                                   error)) {
            return false;
        }
        plan.tasks.push_back({
            .population_slot = slot,
            .action = resolved.action,
            .candidate_index = resolved.candidate_index,
            .branch_ordinal = resolved.branch_ordinal,
            .candidate_root = resolved.root,
            .task_head = task_head,
        });
    }
    if (failure == SelectionExecutorFailurePoint::after_task_heads) {
        return fail(error, "injected crash after BCC-32 selection task heads");
    }
    if (!assign_compute(replay, &plan.tasks, error) ||
        !assign_mating(replay, plan.tasks, &plan.mating_assignments, error)) {
        return false;
    }
    plan.total_compute_supersteps = replay.total_compute_units;
    if (!validate_plan_context(repository, request, resolution, candidate_set,
                               transcript, replay, plan,
                               request.candidate_artifact_kind, error)) {
        return false;
    }
    const Bytes plan_bytes = canonical_selection_execution_plan(plan);
    ContentAddress persisted_plan{};
    if (!persist_typed(repository, "selection-execution-plans", plan_bytes,
                       &persisted_plan, error)) {
        return false;
    }
    *plan_identity = persisted_plan;
    if (failure == SelectionExecutorFailurePoint::after_plan_object) {
        return fail(error, "injected crash after BCC-32 selection execution plan");
    }
    const SelectionExecutionAdmission admission{
        .evaluation_request = request_identity,
        .operation_resolution = resolution_identity,
        .execution_plan = persisted_plan,
    };
    const Bytes admission_bytes = canonical_selection_execution_admission(admission);
    ContentAddress persisted_admission{};
    if (!persist_typed(repository, "selection-execution-admissions",
                       admission_bytes, &persisted_admission, error)) {
        return false;
    }
    *admission_identity = persisted_admission;
    if (failure == SelectionExecutorFailurePoint::after_admission_object) {
        return fail(error,
                    "injected crash after BCC-32 selection admission object");
    }
    if (!publish_operation_pointer(repository, operation, "ADMISSION",
                                   kOperationAdmissionPointerDomain,
                                   persisted_admission, error)) {
        return false;
    }
    if (failure == SelectionExecutorFailurePoint::after_plan_admission) {
        return fail(error, "injected crash after BCC-32 selection execution admission");
    }
    return true;
}

bool read_selection_execution_plan(const std::filesystem::path& repository,
                                   const ContentAddress& identity,
                                   SelectionExecutionPlan* plan,
                                   std::string* error) {
    return read_typed_object(
        object_path(repository, "selection-execution-plans", identity),
        identity, plan, decode_selection_execution_plan, error);
}

bool read_selection_execution_admission(
    const std::filesystem::path& repository,
    const ContentAddress& operation_identity,
    ContentAddress* admission_identity,
    SelectionExecutionAdmission* admission,
    std::string* error) {
    if (admission_identity == nullptr || admission == nullptr ||
        !read_operation_pointer(repository, operation_identity, "ADMISSION",
                                kOperationAdmissionPointerDomain,
                                admission_identity, error) ||
        !read_admission_object(repository, *admission_identity, admission, error)) {
        return fail(error, "BCC-32 selection execution admission is missing");
    }
    SelectionEvaluationRequest request{};
    ContentAddress request_identity{};
    ContentAddress resolution_identity{};
    SelectionOperationResolution resolution{};
    SelectionExecutionPlan plan{};
    SelectionCandidateSet candidate_set{};
    SelectionTranscript transcript{};
    SelectionReplay replay{};
    if (!read_precommitted_selection_request(repository, operation_identity,
                                             &request_identity, &request, error) ||
        !read_selection_operation_resolution(repository, operation_identity,
                                             &resolution_identity, &resolution, error) ||
        !read_selection_execution_plan(repository, admission->execution_plan,
                                       &plan, error) ||
        admission->evaluation_request != request_identity ||
        admission->operation_resolution != resolution_identity ||
        plan.evaluation_request != request_identity ||
        plan.operation_resolution != resolution_identity ||
        !read_selection_candidate_set_object(repository,
                                             request.candidate_set_commitment,
                                             &candidate_set, error) ||
        !read_selection_transcript_object(repository, resolution.transcript,
                                          &transcript, error) ||
        !replay_selection_transcript(candidate_set, request, transcript,
                                     &replay, error) ||
        !validate_plan_context(repository, request, resolution, candidate_set,
                               transcript, replay, plan,
                               plan.artifact_kind,
                               error)) {
        return fail(error, "BCC-32 selection execution admission closure is invalid");
    }
    return true;
}

bool advance_admitted_selection_task(
    PagedWorldExecutor* executor,
    const std::filesystem::path& repository,
    const ContentAddress& operation_identity,
    std::uint64_t population_slot,
    const ContentAddress& current_head,
    TransitionReceipt* receipt,
    std::string* error) {
    if (executor == nullptr || receipt == nullptr ||
        !is_valid_content_address(current_head)) {
        return fail(error, "BCC-32 admitted task advance requires exact inputs");
    }
    ContentAddress admission_identity{};
    SelectionExecutionAdmission admission{};
    SelectionExecutionPlan plan{};
    if (!read_selection_execution_admission(repository, operation_identity,
                                            &admission_identity, &admission,
                                            error) ||
        !read_selection_execution_plan(repository, admission.execution_plan,
                                       &plan, error) ||
        population_slot >= plan.tasks.size()) {
        return fail(error, "BCC-32 admitted task slot is absent");
    }
    const SelectionExecutionTask& task =
        plan.tasks[static_cast<std::size_t>(population_slot)];
    if (task.population_slot != population_slot || task.compute_supersteps == 0u) {
        return fail(error, "BCC-32 admitted task has no remaining execution budget");
    }

    ContentAddress cursor = current_head;
    WorldCommit current{};
    if (!load_world_commit_object(repository, plan.artifact_kind, cursor,
                                  &current, error)) {
        return false;
    }
    std::uint64_t completed = 0u;
    while (cursor != task.task_head) {
        if (completed >= task.compute_supersteps ||
            current.metadata.provenance.entry_event.kind !=
                EntryEventKind::law_continuation ||
            !is_valid_content_address(
                current.metadata.replay_boundary.predecessor_commit)) {
            return fail(error,
                        "BCC-32 task head is outside its admitted forward ancestry");
        }
        const ContentAddress predecessor =
            current.metadata.replay_boundary.predecessor_commit;
        WorldCommit prior{};
        if (!load_world_commit_object(repository, plan.artifact_kind,
                                      predecessor, &prior, error) ||
            prior.metadata.replay_boundary.completed_supersteps ==
                std::numeric_limits<std::uint64_t>::max() ||
            current.metadata.replay_boundary.completed_supersteps !=
                prior.metadata.replay_boundary.completed_supersteps + 1u) {
            return fail(error,
                        "BCC-32 task ancestry contains a non-forward transition");
        }
        cursor = predecessor;
        current = std::move(prior);
        ++completed;
    }
    if (completed >= task.compute_supersteps) {
        return fail(error, "BCC-32 admitted task exhausted its F budget");
    }

    TransitionReceipt produced{};
    if (!executor->advance_object(repository, current_head, plan.artifact_kind,
                                  false, {}, &produced, error) ||
        produced.inverse || produced.input_identity != current_head ||
        !is_valid_content_address(produced.output_identity)) {
        return fail(error, "BCC-32 admitted task did not produce one forward F");
    }
    *receipt = produced;
    return true;
}

}  // namespace substrate::bcc32
