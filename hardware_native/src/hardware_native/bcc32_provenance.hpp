#pragma once

#include "bcc32_world_store.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace substrate::bcc32 {

enum class ArtifactKind : std::uint32_t {
    propagule_capsule = 1u,
    adult_continuity_checkpoint = 2u,
    cultural_capsule = 3u,
};

using Hash256 = std::array<std::uint8_t, 32>;

struct ContentAddress {
    Hash256 digest{};
    std::uint64_t byte_count = 0;

    friend bool operator==(const ContentAddress&, const ContentAddress&) = default;
};

enum class GenesisClass : std::uint32_t {
    // Generic non-biological genesis.  This is not an ancestor of G1.
    G0 = 0u,
    // Independently authored common-ancestor genesis.  This is not abiogenesis
    // and has no biological parent relationship to G0.
    G1 = 1u,
    // Material produced through an eligible biological-parent handoff.
    G2 = 2u,
};

// Artificial selection stays outside the BCC transition.  This transcript is
// the complete host-side cause of a continuation decision: it may choose among
// intact roots and allocate work, but it cannot name an internal edit.
enum class SelectionAction : std::uint32_t {
    preserve_candidate = 0u,
    branch_candidate = 1u,
    allocate_compute = 2u,
    choose_mates = 3u,
    migrate_intact_candidate = 4u,
    retry = 5u,
    stop = 6u,
};

enum class SelectionComparison : std::int32_t {
    less = -1,
    equal = 0,
    greater = 1,
};

enum class SelectionStopReason : std::uint32_t {
    budget_exhausted = 0u,
    declared_limit = 1u,
    evaluator_completed = 2u,
};

struct SelectionCandidate {
    // The frozen, intact candidate capsule/checkpoint and the immutable source
    // that generated it. Parentage and blind mutation seeds remain explicit.
    ContentAddress root{};
    ContentAddress generator_provenance{};
    std::vector<ContentAddress> parentage;
    std::vector<std::uint64_t> mutation_seeds;

    friend bool operator==(const SelectionCandidate&, const SelectionCandidate&) = default;
};

// The future outer executor must persist this object independently before
// evaluator work. Its address freezes candidate roots and generator causes.
// Validation here binds bytes; it does not prove temporal precommit or
// persistence.
struct SelectionCandidateSet {
    static constexpr std::uint32_t kSchemaVersion = 1u;

    std::uint32_t schema_version = kSchemaVersion;
    std::vector<SelectionCandidate> candidates;

    friend bool operator==(const SelectionCandidateSet&,
                           const SelectionCandidateSet&) = default;
};

enum class SelectionComputeSemantics : std::uint32_t {
    production_supersteps = 0u,
};

struct SelectionMatingPair {
    std::uint32_t first_candidate_index = 0u;
    std::uint32_t second_candidate_index = 0u;

    friend bool operator==(const SelectionMatingPair&,
                           const SelectionMatingPair&) = default;
};

struct SelectionEvaluationLimits {
    std::uint64_t maximum_input_bytes = 0u;
    std::uint64_t maximum_transcript_bytes = 0u;
    std::uint64_t maximum_wall_milliseconds = 0u;
    std::uint64_t maximum_cpu_seconds = 0u;
    std::uint64_t maximum_address_space_bytes = 0u;
    std::uint64_t maximum_evaluator_events = 0u;
    std::uint64_t maximum_action_records = 0u;
    std::uint64_t maximum_score_dimensions = 0u;
    std::uint64_t minimum_population_slots = 0u;
    std::uint64_t maximum_population_slots = 0u;
    std::uint64_t maximum_total_compute_units = 0u;
    std::uint64_t maximum_mating_edges = 0u;
    std::uint64_t maximum_retries = 0u;

    friend bool operator==(const SelectionEvaluationLimits&,
                           const SelectionEvaluationLimits&) = default;
};

// This object is immutable before evaluator execution. It fixes every byte the
// evaluator may observe, every action it may return, and every aggregate the
// trusted replay may construct. The evaluator receives the committed input
// objects over a pipe; it never receives repository access.
struct SelectionEvaluationRequest {
    static constexpr std::uint32_t kSchemaVersion = 2u;

    std::uint32_t schema_version = kSchemaVersion;
    ContentAddress operation_predecessor{};
    std::uint64_t selection_round = 0u;
    ContentAddress candidate_set_commitment{};
    ArtifactKind candidate_artifact_kind = ArtifactKind::propagule_capsule;
    ContentAddress evaluator_identity{};
    ContentAddress evaluator_version{};
    ContentAddress evaluator_artifact{};
    ContentAddress evaluator_target_provenance{};
    std::vector<ContentAddress> evaluation_inputs;
    ContentAddress environment_contact_input{};
    ContentAddress replay_semantics{};
    ContentAddress slot_materialization_semantics{};
    ContentAddress scheduler_protocol{};
    ContentAddress mating_contact_manifest{};
    SelectionComputeSemantics compute_semantics =
        SelectionComputeSemantics::production_supersteps;
    std::vector<SelectionAction> action_alphabet;
    std::vector<SelectionMatingPair> allowed_mating_pairs;
    SelectionEvaluationLimits limits{};

    friend bool operator==(const SelectionEvaluationRequest&,
                           const SelectionEvaluationRequest&) = default;
};

enum class SelectionEvaluatorEventKind : std::uint32_t {
    score = 0u,
    comparison = 1u,
};

// Ordered records are the evaluator operations actually performed. A candidate
// may be rescored. Rank remains audit evidence only: deterministic replay never
// consumes it to choose a candidate, root, or continuation. Irrelevant fields
// have canonical values.
struct SelectionEvaluatorEvent {
    SelectionEvaluatorEventKind kind = SelectionEvaluatorEventKind::score;
    std::uint32_t candidate_index = 0u;
    std::uint32_t compared_candidate_index = 0u;
    std::vector<std::int64_t> scores;
    std::uint32_t rank = 0u;
    SelectionComparison result = SelectionComparison::equal;

    friend bool operator==(const SelectionEvaluatorEvent&,
                           const SelectionEvaluatorEvent&) = default;
};

struct SelectionActionRecord {
    SelectionAction action = SelectionAction::preserve_candidate;
    std::uint32_t candidate_index = 0u;
    std::uint64_t multiplicity = 0u;
    std::uint64_t compute_units = 0u;
    std::vector<std::uint32_t> mate_indices;

    friend bool operator==(const SelectionActionRecord&,
                           const SelectionActionRecord&) = default;
};

struct SelectionContinuation {
    SelectionAction action = SelectionAction::preserve_candidate;
    std::uint32_t candidate_index = 0u;
    ContentAddress root{};
    std::uint64_t multiplicity = 0u;

    friend bool operator==(const SelectionContinuation&,
                           const SelectionContinuation&) = default;
};

// A continuation's multiplicity is a compact declaration of these contiguous
// slots. Resolution is action-order first, then branch_ordinal within an action.
struct SelectionPopulationSlot {
    std::uint64_t slot = 0u;
    SelectionAction action = SelectionAction::preserve_candidate;
    std::uint32_t candidate_index = 0u;
    ContentAddress root{};
    std::uint64_t branch_ordinal = 0u;

    friend bool operator==(const SelectionPopulationSlot&,
                           const SelectionPopulationSlot&) = default;
};

struct SelectionComputeAllocation {
    // Compute is attached once to the frozen candidate index, not separately to
    // every population slot that resolves to that candidate.
    std::uint32_t candidate_index = 0u;
    ContentAddress root{};
    std::uint64_t compute_units = 0u;

    friend bool operator==(const SelectionComputeAllocation&,
                           const SelectionComputeAllocation&) = default;
};

struct SelectionMatingEdge {
    // Mating is likewise candidate-index scoped. Multiplicity does not duplicate
    // an edge and selection does not synthesize a descendant root.
    std::uint32_t parent_candidate_index = 0u;
    std::uint32_t mate_candidate_index = 0u;
    ContentAddress parent_root{};
    ContentAddress mate_root{};

    friend bool operator==(const SelectionMatingEdge&,
                           const SelectionMatingEdge&) = default;
};

struct SelectionReplay {
    // Compact continuations plus a checked total admit deterministic random
    // access without allocating one record per declared multiplicity.
    std::vector<SelectionContinuation> next_population;
    std::uint64_t next_population_slot_count = 0u;
    std::vector<SelectionComputeAllocation> compute_allocations;
    std::uint64_t total_compute_units = 0u;
    std::vector<SelectionMatingEdge> mating_graph;
    std::uint64_t retry_count = 0u;
    SelectionStopReason stopping_reason = SelectionStopReason::budget_exhausted;
    bool stopped = false;

    friend bool operator==(const SelectionReplay&, const SelectionReplay&) = default;
};

struct SelectionTranscript {
    static constexpr std::uint32_t kSchemaVersion = 4u;

    std::uint32_t schema_version = kSchemaVersion;
    // This is the temporal firewall: the request already existed before the
    // evaluator was invoked and fixes all duplicated fields below.
    ContentAddress evaluation_request_commitment{};
    // Every member is a content-addressed immutable artifact. Evaluator identity
    // and version are intentionally separate from the target provenance.
    ContentAddress evaluator_identity{};
    ContentAddress evaluator_version{};
    ContentAddress evaluator_target_provenance{};
    // This immutable combined manifest is fixed by the selected candidate.
    // Evaluator selection cannot replace a body, tutor, corpus, or environment.
    ContentAddress environment_contact_input{};
    ContentAddress candidate_set_commitment{};
    std::vector<SelectionEvaluatorEvent> evaluator_transcript;
    // Sorted and unique actions available to this event. It may be a strict
    // subset of the host enum; every action actually used must be declared.
    std::vector<SelectionAction> action_alphabet;
    SelectionStopReason stopping_reason = SelectionStopReason::budget_exhausted;
    std::vector<SelectionActionRecord> actions;

    friend bool operator==(const SelectionTranscript&, const SelectionTranscript&) = default;
};

enum class EntryEventKind : std::uint32_t {
    genesis_entry = 0u,
    evaluator_selection = 1u,
    law_continuation = 2u,
};

// This tagged value states how this artifact entered the repository. A law
// continuation carries no second parent field because ReplayBoundary owns the
// exact predecessor edge. Irrelevant arm fields have canonical zero values.
struct EntryEvent {
    EntryEventKind kind = EntryEventKind::genesis_entry;
    ContentAddress genesis_entry{};
    ContentAddress evaluator_transcript{};
    std::uint64_t next_population_slot = 0u;

    friend bool operator==(const EntryEvent&, const EntryEvent&) = default;
};

// Every field identifies immutable input bytes.  The persistence layer records
// causes and commitments but never interprets their contents as organism state.
struct Provenance {
    // Biological descent exists only for G2.  Artifact-type eligibility is
    // checked by the propagule handoff before these addresses reach storage.
    std::vector<ContentAddress> biological_parents;
    // Cultural sources and other direct causes are recorded here. Checkpoint
    // continuity has its own ordered ReplayBoundary predecessor edge so it
    // cannot erase or masquerade as these causes.
    std::vector<ContentAddress> causal_inputs;
    GenesisClass genesis_class = GenesisClass::G0;
    ContentAddress law{};
    ContentAddress genesis{};
    ContentAddress environment_contact_manifest{};
    // Genesis entry binds genesis exactly. Evaluator selection binds canonical
    // transcript bytes and one deterministic replay population slot. Later law
    // steps replace the current entry event; the selection remains reachable
    // through the immutable predecessor chain.
    EntryEvent entry_event{};
    ContentAddress replay_commitment{};

    friend bool operator==(const Provenance&, const Provenance&) = default;
};

// Incremental SHA-256 keeps hashing memory bounded independently of the world
// size.  finish() is idempotent; updates after finish() are ignored.
class ContentHasher {
  public:
    ContentHasher();

    void update(std::span<const std::byte> bytes);
    [[nodiscard]] Hash256 finish();

  private:
    void compress(const std::uint8_t* block);

    std::array<std::uint32_t, 8> state_{};
    std::array<std::uint8_t, 64> pending_{};
    std::uint64_t total_bytes_ = 0;
    std::size_t pending_bytes_ = 0;
    bool finished_ = false;
    Hash256 digest_{};
};

[[nodiscard]] Hash256 content_hash(std::span<const std::byte> bytes);
[[nodiscard]] ContentAddress content_address(std::span<const std::byte> bytes);
[[nodiscard]] bool is_valid_content_address(const ContentAddress& address);
[[nodiscard]] std::string hash_hex(const Hash256& hash);
[[nodiscard]] std::vector<std::byte> canonical_selection_candidate_set(
    const SelectionCandidateSet& candidate_set);
[[nodiscard]] ContentAddress selection_candidate_set_address(
    const SelectionCandidateSet& candidate_set);
bool validate_selection_candidate_set(const SelectionCandidateSet& candidate_set,
                                      std::string* error);
bool decode_selection_candidate_set(std::span<const std::byte> bytes,
                                    SelectionCandidateSet* candidate_set,
                                    std::string* error);
[[nodiscard]] std::vector<std::byte> canonical_selection_evaluation_request(
    const SelectionEvaluationRequest& request);
[[nodiscard]] ContentAddress selection_evaluation_request_address(
    const SelectionEvaluationRequest& request);
bool validate_selection_evaluation_request(
    const SelectionCandidateSet& candidate_set,
    const SelectionEvaluationRequest& request,
    std::string* error);
bool decode_selection_evaluation_request(
    std::span<const std::byte> bytes,
    SelectionEvaluationRequest* request,
    std::string* error);
[[nodiscard]] std::vector<std::byte> canonical_selection_transcript(
    const SelectionTranscript& transcript);
[[nodiscard]] ContentAddress selection_transcript_address(
    const SelectionTranscript& transcript);
bool validate_selection_transcript(
    const SelectionCandidateSet& candidate_set,
    const SelectionEvaluationRequest& request,
    const SelectionTranscript& transcript,
    std::string* error);
bool decode_selection_transcript(std::span<const std::byte> bytes,
                                 SelectionTranscript* transcript,
                                 std::string* error);
bool replay_selection_transcript(
    const SelectionCandidateSet& candidate_set,
    const SelectionEvaluationRequest& request,
    const SelectionTranscript& transcript,
    SelectionReplay* replay,
    std::string* error);
bool resolve_selection_population_slot(const SelectionReplay& replay,
                                       std::uint64_t slot,
                                       SelectionPopulationSlot* population_slot,
                                       std::string* error);
bool validate_provenance(const Provenance& provenance, std::string* error);
bool validate_provenance(const Provenance& provenance,
                         const SelectionCandidateSet& candidate_set,
                         const SelectionEvaluationRequest& request,
                         const SelectionTranscript& selection_transcript,
                         std::string* error);

}  // namespace substrate::bcc32
