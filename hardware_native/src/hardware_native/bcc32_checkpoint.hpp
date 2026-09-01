#pragma once

#include "bcc32_provenance.hpp"

#include <cstdint>
#include <filesystem>
#include <span>
#include <string>
#include <vector>

namespace substrate::bcc32 {

class PagedWorldExecutor;

struct ByteCounts {
    std::uint64_t direct_chunk_bytes = 0;
    std::uint64_t manifest_bytes = 0;
    std::uint64_t durable_bytes = 0;

    friend bool operator==(const ByteCounts&, const ByteCounts&) = default;
};

enum class ContactBoundary : std::uint32_t {
    closed = 0u,
    reversible_external_tape = 1u,
};

enum class ContinuationTaint : std::uint32_t {
    natural = 0u,
    operator_intervention = 1u,
    synthetic_import = 2u,
};

// Durable artifacts are published only between complete applications of F.
// An open contact boundary must commit the exact external tape state needed to
// reverse and resume the combined world; a closed world has no such address.
// predecessor_commit links this publication to the immutable commit it
// continued. It is separate from provenance.causal_inputs so cultural sources
// and other direct causes are not replaced by checkpoint history.
struct ReplayBoundary {
    std::uint64_t completed_supersteps = 0u;
    std::uint32_t next_factor = 0u;
    ContactBoundary contact = ContactBoundary::closed;
    ContinuationTaint taint = ContinuationTaint::natural;
    ContentAddress predecessor_commit{};
    ContentAddress external_tape_state{};

    friend bool operator==(const ReplayBoundary&, const ReplayBoundary&) = default;
};

// The word contract is persisted explicitly.  This prevents an artifact from
// being restored under a different word width or a different 80-billion-bit
// physical accounting model.
struct ArtifactMetadata {
    std::uint32_t schema_version = 4u;
    ArtifactKind kind = ArtifactKind::propagule_capsule;
    std::uint32_t site_word_bits = static_cast<std::uint32_t>(kBitsPerSite);
    std::uint32_t chunk_edge = kChunkEdge;
    std::uint64_t production_sites = kProductionSites;
    std::uint64_t production_bits = kProductionBits;
    Provenance provenance{};
    ReplayBoundary replay_boundary{};
    WorldSupport world_support{};
    ByteCounts byte_counts{};

    friend bool operator==(const ArtifactMetadata&, const ArtifactMetadata&) = default;
};

struct PropaguleCapsule {
    ArtifactMetadata metadata{};
    WorldStore world{};
};

// This artifact resumes the same developed material.  Its causal inputs are
// continuity references and can never make it a reproduction artifact.
struct AdultContinuityCheckpoint {
    ArtifactMetadata metadata{.kind = ArtifactKind::adult_continuity_checkpoint};
    WorldStore world{};
};

// This artifact carries material culture.  Producer/source identities remain
// causal inputs and can never make it a propagule.
struct CulturalCapsule {
    ArtifactMetadata metadata{.kind = ArtifactKind::cultural_capsule};
    WorldStore world{};
};

struct ChunkObjectReference {
    ChunkCoord coordinate{};
    Hash256 digest{};
    std::uint64_t non_quiescent_sites = 0u;
    std::int64_t delta_n_q = 0;

    friend bool operator==(const ChunkObjectReference&,
                           const ChunkObjectReference&) = default;
};

// A commit is a small immutable material index. Chunk payloads remain in the
// content-addressed repository and are paged into the one CUDA aperture only
// when a law factor needs them.
struct WorldCommit {
    std::uint64_t publication_sequence = 0u;
    ArtifactMetadata metadata{};
    ContentAddress identity{};
    std::vector<ChunkObjectReference> chunks;
};

enum class PublicationFailurePoint : std::uint32_t {
    none = 0u,
    after_generation_sync = 1u,
    after_generation_publish = 2u,
    after_root_temp_sync = 3u,
};

enum class RootExpectation : std::uint32_t {
    absent = 0u,
    exact = 1u,
};

// Every mutable root publication is anchored. Genesis/import requires no
// existing root; a continuation requires the exact root it was computed from.
// There is deliberately no unconditional "latest wins" mode.
struct PublicationPrecondition {
    RootExpectation expectation = RootExpectation::absent;
    ContentAddress expected_identity{};

    [[nodiscard]] static PublicationPrecondition absent() {
        return {};
    }
    [[nodiscard]] static PublicationPrecondition exact(
        ContentAddress identity) {
        return {RootExpectation::exact, identity};
    }
};

[[nodiscard]] WorldSupport support_for_chunk_objects(
    std::span<const ChunkObjectReference> chunks);
[[nodiscard]] ContentAddress material_state_identity(
    std::span<const ChunkObjectReference> chunks);
bool put_world_chunk_object(const std::filesystem::path& root,
                            const ChunkCoord& coordinate,
                            std::span<const SiteWord> words,
                            ChunkObjectReference* reference,
                            bool* materialized,
                            std::string* error);
bool read_world_chunk_object(const std::filesystem::path& root,
                             const ChunkObjectReference& reference,
                             std::vector<SiteWord>* words,
                             std::string* error);
// Generic protocol and generator descriptions are immutable blobs. Candidate
// publication requires each generator_provenance address to resolve here.
bool put_selection_blob_object(const std::filesystem::path& root,
                               std::span<const std::byte> bytes,
                               ContentAddress* identity,
                               std::string* error);
bool read_selection_blob_object(const std::filesystem::path& root,
                                const ContentAddress& identity,
                                std::vector<std::byte>* bytes,
                                std::string* error);
// Candidate sets are durable before evaluator work begins. A set is accepted
// only when every frozen candidate root resolves to a complete immutable world
// commit in this repository. A transcript is accepted only after its named set
// exists and its deterministic replay validates against that exact set.
bool put_selection_candidate_set_object(
    const std::filesystem::path& root,
    const SelectionCandidateSet& candidate_set,
    ContentAddress* identity,
    std::string* error);
bool read_selection_candidate_set_object(
    const std::filesystem::path& root,
    const ContentAddress& identity,
    SelectionCandidateSet* candidate_set,
    std::string* error);
bool put_selection_evaluation_request_object(
    const std::filesystem::path& root,
    const SelectionEvaluationRequest& request,
    ContentAddress* identity,
    std::string* error);
bool read_selection_evaluation_request_object(
    const std::filesystem::path& root,
    const ContentAddress& identity,
    SelectionEvaluationRequest* request,
    std::string* error);
bool put_selection_transcript_object(
    const std::filesystem::path& root,
    const SelectionTranscript& transcript,
    ContentAddress* identity,
    std::string* error);
bool read_selection_transcript_object(
    const std::filesystem::path& root,
    const ContentAddress& identity,
    SelectionTranscript* transcript,
    std::string* error);
// Materializes one replay-derived population slot as a new immutable world
// commit. It copies the selected candidate's material and frozen metadata
// exactly, changes only the typed evaluator entry edge, and does not mutate a
// lineage root. This is the branch-safe input task for PagedWorldExecutor.
bool materialize_selection_population_slot(
    const std::filesystem::path& root,
    const ContentAddress& transcript_identity,
    std::uint64_t slot,
    ArtifactKind expected_kind,
    ContentAddress* selected_identity,
    std::string* error);

bool publish_world_commit(const ArtifactMetadata& metadata,
                          std::span<const ChunkObjectReference> chunks,
                          ArtifactKind expected_kind,
                          const std::filesystem::path& root,
                          PublicationPrecondition precondition,
                          ContentAddress* published_identity,
                          std::string* error,
                          PublicationFailurePoint failure =
                              PublicationFailurePoint::none);
bool load_world_commit(const std::filesystem::path& root,
                       ArtifactKind expected_kind,
                       WorldCommit* commit,
                       std::string* error);
bool load_world_commit_object(const std::filesystem::path& root,
                              ArtifactKind expected_kind,
                              const ContentAddress& identity,
                              WorldCommit* commit,
                              std::string* error);

// These endpoints are explicit host imports for tests and migration. They
// force synthetic_import taint and may only establish an empty root. Natural
// genesis and law continuation use their anchored internal publication paths.
bool import_propagule_capsule(const PropaguleCapsule& capsule,
                              const std::filesystem::path& root,
                              ContentAddress* published_identity,
                              std::string* error,
                              PublicationFailurePoint failure =
                                  PublicationFailurePoint::none);
bool import_adult_continuity_checkpoint(
    const AdultContinuityCheckpoint& checkpoint,
    const std::filesystem::path& root,
    ContentAddress* published_identity,
    std::string* error,
    PublicationFailurePoint failure = PublicationFailurePoint::none);
bool import_cultural_capsule(const CulturalCapsule& capsule,
                             const std::filesystem::path& root,
                             ContentAddress* published_identity,
                             std::string* error,
                             PublicationFailurePoint failure =
                                 PublicationFailurePoint::none);

bool read_propagule_capsule(const std::filesystem::path& root,
                            PropaguleCapsule* capsule,
                            ContentAddress* published_identity,
                            std::string* error);
bool read_adult_continuity_checkpoint(
    const std::filesystem::path& root,
    AdultContinuityCheckpoint* checkpoint,
    ContentAddress* published_identity,
    std::string* error);
bool read_cultural_capsule(const std::filesystem::path& root,
                           CulturalCapsule* capsule,
                           ContentAddress* published_identity,
                           std::string* error);

}  // namespace substrate::bcc32
