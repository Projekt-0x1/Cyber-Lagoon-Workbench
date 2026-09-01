#pragma once

#include "bcc32_checkpoint.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace substrate::bcc32 {

inline constexpr std::string_view kSelectionReplaySemanticsV1 =
    "0x1/bcc32/selection-replay/index-derived-intact-roots/v1";
inline constexpr std::string_view kSelectionSlotMaterializationSemanticsV1 =
    "0x1/bcc32/selection-slot/immutable-intact-entry-edge/v1";
inline constexpr std::string_view kSelectionSchedulerProtocolV1 =
    "0x1/bcc32/selection-scheduler/fixed-forward-complete-superstep/v1";

class PagedWorldExecutor;
struct TransitionReceipt;

enum class SelectionExecutorFailurePoint : std::uint32_t {
    none = 0u,
    after_request_object = 1u,
    after_request_admission = 2u,
    after_transcript_object = 3u,
    after_resolution_object = 4u,
    after_resolution_admission = 5u,
    after_task_heads = 6u,
    after_plan_object = 7u,
    after_plan_admission = 8u,
    after_admission_object = 9u,
};

struct SelectionEvaluatorReceipt {
    static constexpr std::uint32_t kSchemaVersion = 1u;

    std::uint32_t schema_version = kSchemaVersion;
    ContentAddress evaluation_request{};
    ContentAddress evaluator_artifact{};
    ContentAddress input_envelope{};
    ContentAddress output_bytes{};
    ContentAddress sandbox_profile{};
    ContentAddress transcript{};
    std::uint32_t exit_status = 0u;

    friend bool operator==(const SelectionEvaluatorReceipt&,
                           const SelectionEvaluatorReceipt&) = default;
};

struct SelectionOperationResolution {
    static constexpr std::uint32_t kSchemaVersion = 1u;

    std::uint32_t schema_version = kSchemaVersion;
    ContentAddress evaluation_request{};
    ContentAddress transcript{};
    ContentAddress evaluator_receipt{};

    friend bool operator==(const SelectionOperationResolution&,
                           const SelectionOperationResolution&) = default;
};

struct SelectionExecutionTask {
    std::uint64_t population_slot = 0u;
    SelectionAction action = SelectionAction::preserve_candidate;
    std::uint32_t candidate_index = 0u;
    std::uint64_t branch_ordinal = 0u;
    ContentAddress candidate_root{};
    ContentAddress task_head{};
    std::uint64_t compute_supersteps = 0u;

    friend bool operator==(const SelectionExecutionTask&,
                           const SelectionExecutionTask&) = default;
};

struct SelectionMatingAssignment {
    std::uint64_t first_population_slot = 0u;
    std::uint64_t second_population_slot = 0u;
    std::uint32_t first_candidate_index = 0u;
    std::uint32_t second_candidate_index = 0u;

    friend bool operator==(const SelectionMatingAssignment&,
                           const SelectionMatingAssignment&) = default;
};

struct SelectionExecutionPlan {
    static constexpr std::uint32_t kSchemaVersion = 1u;

    std::uint32_t schema_version = kSchemaVersion;
    ContentAddress evaluation_request{};
    ContentAddress operation_resolution{};
    ContentAddress transcript{};
    ContentAddress scheduler_protocol{};
    ArtifactKind artifact_kind = ArtifactKind::propagule_capsule;
    std::vector<SelectionExecutionTask> tasks;
    std::vector<SelectionMatingAssignment> mating_assignments;
    std::uint64_t total_compute_supersteps = 0u;

    friend bool operator==(const SelectionExecutionPlan&,
                           const SelectionExecutionPlan&) = default;
};

struct SelectionExecutionAdmission {
    static constexpr std::uint32_t kSchemaVersion = 1u;

    std::uint32_t schema_version = kSchemaVersion;
    ContentAddress evaluation_request{};
    ContentAddress operation_resolution{};
    ContentAddress execution_plan{};

    friend bool operator==(const SelectionExecutionAdmission&,
                           const SelectionExecutionAdmission&) = default;
};

[[nodiscard]] ContentAddress selection_operation_identity(
    const SelectionEvaluationRequest& request);
[[nodiscard]] ContentAddress canonical_selection_sandbox_profile();

[[nodiscard]] std::vector<std::byte> canonical_selection_evaluator_receipt(
    const SelectionEvaluatorReceipt& receipt);
[[nodiscard]] ContentAddress selection_evaluator_receipt_address(
    const SelectionEvaluatorReceipt& receipt);
bool decode_selection_evaluator_receipt(std::span<const std::byte> bytes,
                                        SelectionEvaluatorReceipt* receipt,
                                        std::string* error);

[[nodiscard]] std::vector<std::byte> canonical_selection_operation_resolution(
    const SelectionOperationResolution& resolution);
[[nodiscard]] ContentAddress selection_operation_resolution_address(
    const SelectionOperationResolution& resolution);
bool decode_selection_operation_resolution(std::span<const std::byte> bytes,
                                           SelectionOperationResolution* resolution,
                                           std::string* error);

[[nodiscard]] std::vector<std::byte> canonical_selection_execution_plan(
    const SelectionExecutionPlan& plan);
[[nodiscard]] ContentAddress selection_execution_plan_address(
    const SelectionExecutionPlan& plan);
bool decode_selection_execution_plan(std::span<const std::byte> bytes,
                                     SelectionExecutionPlan* plan,
                                     std::string* error);

[[nodiscard]] std::vector<std::byte> canonical_selection_execution_admission(
    const SelectionExecutionAdmission& admission);
[[nodiscard]] ContentAddress selection_execution_admission_address(
    const SelectionExecutionAdmission& admission);
bool decode_selection_execution_admission(std::span<const std::byte> bytes,
                                          SelectionExecutionAdmission* admission,
                                          std::string* error);

// Atomically binds one predecessor+round operation to one immutable request.
// Persisting the content-addressed request alone is not admission.
bool precommit_selection_operation(
    const std::filesystem::path& repository,
    const SelectionEvaluationRequest& request,
    ContentAddress* request_identity,
    std::string* error,
    SelectionExecutorFailurePoint failure = SelectionExecutorFailurePoint::none);
bool read_precommitted_selection_request(
    const std::filesystem::path& repository,
    const ContentAddress& operation_identity,
    ContentAddress* request_identity,
    SelectionEvaluationRequest* request,
    std::string* error);

// Loads the precommitted evaluator artifact and inputs, executes it in the
// fail-closed Linux sandbox, and feeds its canonical output into the one-result CAS.
bool run_precommitted_selection_evaluator(
    const std::filesystem::path& repository,
    const ContentAddress& request_identity,
    ContentAddress* resolution_identity,
    ContentAddress* transcript_identity,
    std::string* error,
    SelectionExecutorFailurePoint failure = SelectionExecutorFailurePoint::none);

bool read_selection_operation_resolution(
    const std::filesystem::path& repository,
    const ContentAddress& operation_identity,
    ContentAddress* resolution_identity,
    SelectionOperationResolution* resolution,
    std::string* error);

// Materializes every replayed slot before publishing one immutable plan and one
// operation admission pointer. Therefore a crash cannot expose a partial population.
bool admit_selection_execution_plan(
    const std::filesystem::path& repository,
    const ContentAddress& request_identity,
    ContentAddress* admission_identity,
    ContentAddress* plan_identity,
    std::string* error,
    SelectionExecutorFailurePoint failure = SelectionExecutorFailurePoint::none);
bool read_selection_execution_admission(
    const std::filesystem::path& repository,
    const ContentAddress& operation_identity,
    ContentAddress* admission_identity,
    SelectionExecutionAdmission* admission,
    std::string* error);
bool read_selection_execution_plan(const std::filesystem::path& repository,
                                   const ContentAddress& identity,
                                   SelectionExecutionPlan* plan,
                                   std::string* error);

// Advances exactly one forward F from a head inside an admitted slot. The
// operation, slot, artifact kind, budget, ancestry, direction, and schedule are
// executor-derived; callers cannot substitute any of them.
bool advance_admitted_selection_task(
    PagedWorldExecutor* executor,
    const std::filesystem::path& repository,
    const ContentAddress& operation_identity,
    std::uint64_t population_slot,
    const ContentAddress& current_head,
    TransitionReceipt* receipt,
    std::string* error);

}  // namespace substrate::bcc32
