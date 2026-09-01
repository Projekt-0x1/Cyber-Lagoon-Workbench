#include "bcc32_checkpoint.hpp"
#include "bcc32_law_identity.hpp"

#include <fcntl.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cerrno>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <set>
#include <span>
#include <string_view>
#include <utility>
#include <vector>

namespace substrate::bcc32 {
namespace {

constexpr std::array<std::uint8_t, 8> kManifestMagic = {
    'B', '3', '2', 'A', 'R', 'T', '0', '4'};
constexpr std::array<std::uint8_t, 8> kRootMagic = {
    'B', '3', '2', 'R', 'O', 'O', 'T', '4'};
constexpr std::string_view kChunkDomain = "bcc32/chunk/v3\0";
constexpr std::string_view kManifestDomain = "bcc32/manifest/v4\0";
constexpr std::string_view kMaterialStateDomain = "bcc32/material-state/v1\0";
constexpr std::string_view kRootDomain = "bcc32/root/v1\0";
constexpr std::uint64_t kMaximumSelectionBlobBytes = 1ull << 30u;

struct Manifest {
    std::uint64_t sequence = 0;
    ArtifactMetadata metadata{};
    std::vector<ChunkObjectReference> chunks;
};

struct RootRecord {
    std::uint64_t sequence = 0;
    ContentAddress identity{};
};

using Bytes = std::vector<std::uint8_t>;

#include "bcc32_checkpoint_private_codec.inl"

bool parse_manifest(const Bytes& bytes,
                    const ContentAddress& expected_identity,
                    Manifest* manifest,
                    std::string* error) {
    if (domain_hash(kManifestDomain, as_bytes(bytes)) != expected_identity.digest ||
        expected_identity.byte_count != bytes.size()) {
        return fail(error, "BCC-32 manifest content address mismatch");
    }
    std::span<const std::uint8_t> input(bytes);
    if (input.size() < kManifestMagic.size() ||
        !std::equal(kManifestMagic.begin(), kManifestMagic.end(), input.begin())) {
        return fail(error, "BCC-32 manifest magic is invalid");
    }
    std::size_t cursor = kManifestMagic.size();
    std::uint32_t kind = 0;
    if (!take_u32(input, &cursor, &manifest->metadata.schema_version) ||
        !take_u32(input, &cursor, &kind) ||
        !take_u32(input, &cursor, &manifest->metadata.site_word_bits) ||
        !take_u32(input, &cursor, &manifest->metadata.chunk_edge) ||
        !take_u64(input, &cursor, &manifest->metadata.production_sites) ||
        !take_u64(input, &cursor, &manifest->metadata.production_bits) ||
        !take_provenance(input, &cursor, &manifest->metadata.provenance) ||
        !take_replay_boundary(input, &cursor, &manifest->metadata.replay_boundary) ||
        !take_support(input, &cursor, &manifest->metadata.world_support) ||
        !take_u64(input, &cursor, &manifest->metadata.byte_counts.direct_chunk_bytes) ||
        !take_u64(input, &cursor, &manifest->metadata.byte_counts.manifest_bytes) ||
        !take_u64(input, &cursor, &manifest->metadata.byte_counts.durable_bytes)) {
        return fail(error, "BCC-32 manifest is truncated");
    }
    manifest->metadata.kind = static_cast<ArtifactKind>(kind);
    std::uint64_t chunk_count = 0;
    constexpr std::uint64_t kMinimumChunkRecordBytes = 83u;
    if (!take_u64(input, &cursor, &chunk_count) ||
        chunk_count > (input.size() - cursor) / kMinimumChunkRecordBytes ||
        chunk_count > std::numeric_limits<std::size_t>::max()) {
        return fail(error, "BCC-32 manifest has an invalid chunk count");
    }
    manifest->chunks.resize(static_cast<std::size_t>(chunk_count));
    std::uint64_t non_quiescent_sites = 0u;
    for (ChunkObjectReference& chunk : manifest->chunks) {
        std::uint64_t direct_bytes = 0;
        if (!take_coordinate_component(input, &cursor, &chunk.coordinate.x) ||
            !take_coordinate_component(input, &cursor, &chunk.coordinate.y) ||
            !take_coordinate_component(input, &cursor, &chunk.coordinate.z) ||
            !take_u64(input, &cursor, &direct_bytes) ||
            !take_hash(input, &cursor, &chunk.digest) ||
            !take_u64(input, &cursor, &chunk.non_quiescent_sites) ||
            !take_i64(input, &cursor, &chunk.delta_n_q) ||
            direct_bytes != kChunkBytes || chunk.non_quiescent_sites == 0u ||
            chunk.non_quiescent_sites > kChunkSites) {
            return fail(error, "BCC-32 manifest chunk record is invalid");
        }
        if (chunk.digest == Hash256{}) {
            return fail(error, "BCC-32 manifest chunk has no content address");
        }
        if (non_quiescent_sites >
            std::numeric_limits<std::uint64_t>::max() -
                chunk.non_quiescent_sites) {
            return fail(error, "BCC-32 manifest non-Q site count overflows");
        }
        non_quiescent_sites += chunk.non_quiescent_sites;
        if (chunk_count > 1u && &chunk != &manifest->chunks.front() &&
            !CoordinateLess{}(
                manifest->chunks[static_cast<std::size_t>(
                    &chunk - manifest->chunks.data()) - 1u].coordinate,
                chunk.coordinate)) {
            return fail(error, "BCC-32 manifest chunk order is noncanonical");
        }
    }
    if (cursor != input.size()) return fail(error, "BCC-32 manifest has trailing bytes");
    if (manifest->metadata.byte_counts.manifest_bytes != input.size() ||
        manifest->metadata.byte_counts.direct_chunk_bytes != chunk_count * kChunkBytes ||
        manifest->metadata.byte_counts.direct_chunk_bytes >
            std::numeric_limits<std::uint64_t>::max() -
                manifest->metadata.byte_counts.manifest_bytes ||
        manifest->metadata.byte_counts.durable_bytes !=
            manifest->metadata.byte_counts.direct_chunk_bytes +
                manifest->metadata.byte_counts.manifest_bytes) {
        return fail(error, "BCC-32 manifest byte counts do not agree");
    }
    if (manifest->metadata.world_support.materialized_chunks != chunk_count ||
        manifest->metadata.world_support.direct_word_bytes != chunk_count * kChunkBytes ||
        manifest->metadata.world_support.non_quiescent_sites != non_quiescent_sites) {
        return fail(error, "BCC-32 manifest support does not agree with chunks");
    }
    return true;
}

Bytes serialize_root(const RootRecord& root) {
    Bytes bytes;
    bytes.insert(bytes.end(), kRootMagic.begin(), kRootMagic.end());
    put_u64(&bytes, root.sequence);
    put_address(&bytes, root.identity);
    put_hash(&bytes, domain_hash(kRootDomain, as_bytes(bytes)));
    return bytes;
}

bool parse_root(const Bytes& bytes, RootRecord* root) {
    std::span<const std::uint8_t> input(bytes);
    if (input.size() < kRootMagic.size() ||
        !std::equal(kRootMagic.begin(), kRootMagic.end(), input.begin())) {
        return false;
    }
    std::size_t cursor = kRootMagic.size();
    if (!take_u64(input, &cursor, &root->sequence) ||
        !take_address(input, &cursor, &root->identity)) {
        return false;
    }
    const std::size_t signed_bytes = cursor;
    Hash256 checksum{};
    return take_hash(input, &cursor, &checksum) && cursor == input.size() &&
           root->sequence != 0u && is_valid_content_address(root->identity) &&
           checksum == domain_hash(kRootDomain, as_bytes(bytes).first(signed_bytes));
}

std::filesystem::path object_path(const std::filesystem::path& root,
                                  std::string_view subdir,
                                  const Hash256& digest,
                                  std::string_view suffix = {}) {
    return root / "objects" / subdir / (hash_hex(digest) + std::string(suffix));
}

std::filesystem::path manifest_object_path(const std::filesystem::path& root,
                                           const Hash256& digest) {
    return object_path(root, "manifests", digest, ".manifest");
}

std::filesystem::path chunk_object_path(const std::filesystem::path& root,
                                        const Hash256& digest) {
    return object_path(root, "chunks", digest, ".words");
}

std::filesystem::path selection_candidate_set_object_path(
    const std::filesystem::path& root,
    const Hash256& digest) {
    return object_path(root, "selection-candidate-sets", digest, ".candidates");
}

std::filesystem::path selection_blob_object_path(
    const std::filesystem::path& root,
    const Hash256& digest) {
    return object_path(root, "selection-blobs", digest);
}

std::filesystem::path selection_transcript_object_path(
    const std::filesystem::path& root,
    const Hash256& digest) {
    return object_path(root, "selection-transcripts", digest, ".selection");
}

std::filesystem::path selection_evaluation_request_object_path(
    const std::filesystem::path& root,
    const Hash256& digest) {
    return object_path(root, "selection-evaluation-requests", digest, ".request");
}

bool read_root(const std::filesystem::path& root_directory,
               std::string_view root_name,
               Manifest* manifest,
               ContentAddress* identity,
               std::string* error) {
    Bytes root_bytes;
    if (!read_file(root_directory / root_name, &root_bytes, error)) return false;
    RootRecord root{};
    if (!parse_root(root_bytes, &root)) {
        return fail(error, "BCC-32 published root record is invalid");
    }
    Bytes manifest_bytes;
    if (!read_file(manifest_object_path(root_directory, root.identity.digest),
                   &manifest_bytes,
                   error) ||
        !parse_manifest(manifest_bytes, root.identity, manifest, error)) {
        return false;
    }
    manifest->sequence = root.sequence;
    *identity = root.identity;
    return true;
}

bool inspect_root_slot(const std::filesystem::path& root_directory,
                       std::string_view root_name,
                       bool* present,
                       std::string* error) {
    std::error_code filesystem_error;
    const std::filesystem::file_status status = std::filesystem::symlink_status(
        root_directory / root_name, filesystem_error);
    if (filesystem_error == std::errc::no_such_file_or_directory) {
        *present = false;
        return true;
    }
    if (filesystem_error) {
        return fail(error, "cannot inspect BCC-32 published root slot");
    }
    if (!std::filesystem::is_regular_file(status)) {
        return fail(error, "BCC-32 published root slot is not a regular file");
    }
    *present = true;
    return true;
}

bool load_published(const std::filesystem::path& root,
                    Manifest* manifest,
                    ContentAddress* identity,
                    std::string* error) {
    Manifest first{};
    Manifest second{};
    ContentAddress first_identity{};
    ContentAddress second_identity{};
    std::string first_error;
    std::string second_error;
    bool first_present = false;
    bool second_present = false;
    if (!inspect_root_slot(root, "ROOT_A", &first_present, error) ||
        !inspect_root_slot(root, "ROOT_B", &second_present, error)) {
        return false;
    }
    const bool first_ok = first_present &&
        read_root(root, "ROOT_A", &first, &first_identity, &first_error);
    const bool second_ok = second_present &&
        read_root(root, "ROOT_B", &second, &second_identity, &second_error);
    if ((first_present && !first_ok) || (second_present && !second_ok)) {
        return fail(error,
                    "BCC-32 published root slot is corrupt: ROOT_A=" +
                        first_error + "; ROOT_B=" + second_error);
    }
    if (!first_ok && !second_ok) {
        return fail(error,
                    "BCC-32 artifact has no valid published root: ROOT_A=" +
                        first_error + "; ROOT_B=" + second_error);
    }
    if (first_ok != second_ok) {
        const Manifest& bootstrap = first_ok ? first : second;
        if (bootstrap.sequence != 1u) {
            return fail(error,
                        "BCC-32 non-bootstrap publication is missing a root slot");
        }
    }
    if (!second_ok || (first_ok && first.sequence >= second.sequence)) {
        *manifest = std::move(first);
        *identity = first_identity;
    } else {
        *manifest = std::move(second);
        *identity = second_identity;
    }
    return true;
}

bool put_chunk_object_impl(const std::filesystem::path& root,
                           const ChunkCoord& coordinate,
                           std::span<const SiteWord> words,
                           ChunkObjectReference* reference,
                           bool* materialized,
                           std::string* error) {
    if (reference == nullptr || materialized == nullptr || words.size() != kChunkSites) {
        return fail(error, "BCC-32 chunk object input is invalid");
    }
    reference->coordinate = coordinate;
    reference->digest = {};
    reference->non_quiescent_sites = 0u;
    reference->delta_n_q = 0;
    for (SiteWord word : words) {
        reference->non_quiescent_sites += word != kQuiescentWord ? 1u : 0u;
        reference->delta_n_q +=
            static_cast<std::int32_t>(std::popcount(word)) -
            static_cast<std::int32_t>(std::popcount(kQuiescentWord));
    }
    if (reference->non_quiescent_sites == 0u) {
        *materialized = false;
        return true;
    }
    Bytes encoded;
    encoded.reserve(kChunkBytes);
    for (SiteWord word : words) {
        for (std::uint32_t shift = 0u; shift < 32u; shift += 8u) {
            encoded.push_back(static_cast<std::uint8_t>(word >> shift));
        }
    }
    reference->digest = domain_hash(kChunkDomain, as_bytes(encoded));
    *materialized = true;
    return write_immutable_object(
        chunk_object_path(root, reference->digest), encoded, error);
}

bool read_chunk(const std::filesystem::path& path,
                const Hash256& expected_digest,
                std::uint64_t expected_non_quiescent_sites,
                std::int64_t expected_delta_n_q,
                std::vector<SiteWord>* words,
                std::string* error) {
    Bytes bytes;
    if (!read_file(path, &bytes, error)) return false;
    if (bytes.size() != kChunkBytes ||
        domain_hash(kChunkDomain, as_bytes(bytes)) != expected_digest) {
        return fail(error, "BCC-32 direct chunk content address mismatch");
    }
    words->resize(kChunkSites);
    for (std::size_t index = 0; index < words->size(); ++index) {
        std::size_t cursor = index * sizeof(SiteWord);
        if (!take_u32(std::span<const std::uint8_t>(bytes), &cursor, &(*words)[index])) {
            return fail(error, "BCC-32 direct chunk decoding failed");
        }
    }
    std::uint64_t observed = 0u;
    std::int64_t observed_delta = 0;
    for (SiteWord word : *words) {
        observed += word != kQuiescentWord ? 1u : 0u;
        observed_delta += static_cast<std::int32_t>(std::popcount(word)) -
                          static_cast<std::int32_t>(std::popcount(kQuiescentWord));
    }
    return observed == expected_non_quiescent_sites && observed != 0u &&
                   observed_delta == expected_delta_n_q
               ? true
               : fail(error, "BCC-32 chunk support count does not match its manifest");
}

bool ensure_repository(const std::filesystem::path& root, std::string* error) {
    std::error_code filesystem_error;
    std::filesystem::create_directories(root, filesystem_error);
    if (filesystem_error) {
        return fail(error, "cannot create BCC-32 artifact root");
    }
    std::filesystem::create_directories(root / "objects" / "chunks", filesystem_error);
    if (filesystem_error) return fail(error, "cannot create BCC-32 chunk object store");
    std::filesystem::create_directories(root / "objects" / "manifests", filesystem_error);
    if (filesystem_error) return fail(error, "cannot create BCC-32 manifest object store");
    std::filesystem::create_directories(
        root / "objects" / "selection-blobs", filesystem_error);
    if (filesystem_error) {
        return fail(error, "cannot create BCC-32 selection-blob object store");
    }
    std::filesystem::create_directories(
        root / "objects" / "selection-candidate-sets", filesystem_error);
    if (filesystem_error) {
        return fail(error, "cannot create BCC-32 candidate-set object store");
    }
    std::filesystem::create_directories(
        root / "objects" / "selection-transcripts", filesystem_error);
    if (filesystem_error) {
        return fail(error, "cannot create BCC-32 immutable object store");
    }
    std::filesystem::create_directories(
        root / "objects" / "selection-evaluation-requests", filesystem_error);
    if (filesystem_error) {
        return fail(error, "cannot create BCC-32 selection-request object store");
    }
    const std::filesystem::path objects = root / "objects";
    const std::filesystem::path parent =
        root.parent_path().empty() ? std::filesystem::path(".") : root.parent_path();
    return sync_directory(objects / "chunks", error) &&
           sync_directory(objects / "manifests", error) &&
           sync_directory(objects / "selection-blobs", error) &&
           sync_directory(objects / "selection-candidate-sets", error) &&
           sync_directory(objects / "selection-transcripts", error) &&
           sync_directory(objects / "selection-evaluation-requests", error) &&
           sync_directory(objects, error) && sync_directory(root, error) &&
           sync_directory(parent, error);
}

bool load_complete_manifest_object(const std::filesystem::path& root,
                                   const ContentAddress& identity,
                                   Manifest* manifest,
                                   std::string* error) {
    if (manifest == nullptr || !is_valid_content_address(identity)) {
        return fail(error, "BCC-32 immutable world-commit lookup is invalid");
    }
    Bytes bytes;
    Manifest decoded{};
    if (!read_file(manifest_object_path(root, identity.digest), &bytes, error) ||
        !parse_manifest(bytes, identity, &decoded, error) ||
        !validate_metadata(decoded.metadata, decoded.metadata.kind, error)) {
        return false;
    }
    for (const ChunkObjectReference& chunk : decoded.chunks) {
        std::vector<SiteWord> verified;
        if (!read_chunk(chunk_object_path(root, chunk.digest),
                        chunk.digest,
                        chunk.non_quiescent_sites,
                        chunk.delta_n_q,
                        &verified,
                        error)) {
            return false;
        }
    }
    *manifest = std::move(decoded);
    return true;
}

bool read_selection_candidate_set_impl(
    const std::filesystem::path& root,
    const ContentAddress& identity,
    SelectionCandidateSet* candidate_set,
    std::string* error) {
    if (candidate_set == nullptr || !is_valid_content_address(identity)) {
        return fail(error, "BCC-32 candidate-set lookup is invalid");
    }
    Bytes bytes;
    if (!read_file(selection_candidate_set_object_path(root, identity.digest),
                   &bytes,
                   error) ||
        content_address(as_bytes(bytes)) != identity ||
        !decode_selection_candidate_set(as_bytes(bytes), candidate_set, error)) {
        return fail(error, "BCC-32 candidate-set object is missing or corrupt");
    }
    return true;
}

bool read_selection_transcript_impl(const std::filesystem::path& root,
                                    const ContentAddress& identity,
                                    SelectionTranscript* transcript,
                                    std::string* error) {
    if (transcript == nullptr || !is_valid_content_address(identity)) {
        return fail(error, "BCC-32 selection-transcript lookup is invalid");
    }
    Bytes bytes;
    if (!read_file(selection_transcript_object_path(root, identity.digest),
                   &bytes,
                   error) ||
        content_address(as_bytes(bytes)) != identity ||
        !decode_selection_transcript(as_bytes(bytes), transcript, error)) {
        return fail(error, "BCC-32 selection-transcript object is missing or corrupt");
    }
    return true;
}

bool read_selection_evaluation_request_impl(
    const std::filesystem::path& root,
    const ContentAddress& identity,
    SelectionEvaluationRequest* request,
    std::string* error) {
    if (request == nullptr || !is_valid_content_address(identity)) {
        return fail(error, "BCC-32 selection-request lookup is invalid");
    }
    Bytes bytes;
    if (!read_file(selection_evaluation_request_object_path(root, identity.digest),
                   &bytes,
                   error) ||
        content_address(as_bytes(bytes)) != identity ||
        !decode_selection_evaluation_request(as_bytes(bytes), request, error)) {
        return fail(error, "BCC-32 selection-request object is missing or corrupt");
    }
    return true;
}

struct ContentAddressLess {
    bool operator()(const ContentAddress& left,
                    const ContentAddress& right) const {
        if (left.digest != right.digest) return left.digest < right.digest;
        return left.byte_count < right.byte_count;
    }
};

struct ProvenanceClosureWalk {
    static constexpr std::size_t kMaximumArtifacts = 1u << 20u;
    std::set<ContentAddress, ContentAddressLess> active;
    std::set<ContentAddress, ContentAddressLess> validated;
};

bool load_complete_manifest_object_with_provenance(
    const std::filesystem::path& root,
    const ContentAddress& identity,
    Manifest* manifest,
    ProvenanceClosureWalk* walk,
    std::string* error);

[[nodiscard]] bool same_frozen_lineage_metadata(
    const ArtifactMetadata& left,
    const ArtifactMetadata& right) {
    return left.schema_version == right.schema_version && left.kind == right.kind &&
           left.site_word_bits == right.site_word_bits &&
           left.chunk_edge == right.chunk_edge &&
           left.production_sites == right.production_sites &&
           left.production_bits == right.production_bits &&
           left.provenance.biological_parents ==
               right.provenance.biological_parents &&
           left.provenance.causal_inputs == right.provenance.causal_inputs &&
           left.provenance.genesis_class == right.provenance.genesis_class &&
           left.provenance.law == right.provenance.law &&
           left.provenance.genesis == right.provenance.genesis &&
           left.provenance.environment_contact_manifest ==
               right.provenance.environment_contact_manifest;
}

bool validate_selection_metadata_delta(const ArtifactMetadata& metadata,
                                       const Manifest& candidate,
                                       const ContentAddress& selected_root,
                                       std::string* error) {
    const ReplayBoundary& requested = metadata.replay_boundary;
    const ReplayBoundary& frozen = candidate.metadata.replay_boundary;
    if (!same_frozen_lineage_metadata(metadata, candidate.metadata) ||
        requested.completed_supersteps != frozen.completed_supersteps ||
        requested.next_factor != frozen.next_factor ||
        requested.contact != frozen.contact || requested.taint != frozen.taint ||
        requested.external_tape_state != frozen.external_tape_state ||
        requested.predecessor_commit != selected_root) {
        return fail(error,
                    "BCC-32 evaluator selection may change only entry provenance");
    }
    if (metadata.provenance.replay_commitment != selected_root) {
        return fail(error, "BCC-32 selection replay commitment does not name its selected root");
    }
    return true;
}

bool validate_law_continuation_metadata_delta(
    const ArtifactMetadata& metadata,
    const ArtifactMetadata& predecessor,
    std::string* error) {
    const ReplayBoundary& requested = metadata.replay_boundary;
    const ReplayBoundary& prior = predecessor.replay_boundary;
    const bool forward =
        prior.completed_supersteps != std::numeric_limits<std::uint64_t>::max() &&
        requested.completed_supersteps == prior.completed_supersteps + 1u;
    const bool inverse = requested.completed_supersteps !=
                             std::numeric_limits<std::uint64_t>::max() &&
                         prior.completed_supersteps ==
                             requested.completed_supersteps + 1u;
    if (metadata.provenance.entry_event.kind != EntryEventKind::law_continuation ||
        !same_frozen_lineage_metadata(metadata, predecessor) ||
        requested.next_factor != prior.next_factor ||
        requested.contact != prior.contact || requested.taint != prior.taint ||
        requested.external_tape_state != prior.external_tape_state ||
        (!forward && !inverse)) {
        return fail(error,
                    "BCC-32 law continuation changes frozen lineage metadata");
    }
    return true;
}

bool collect_candidate_set_dependencies(
    const std::filesystem::path& root,
    const SelectionCandidateSet& candidate_set,
    std::vector<ContentAddress>* dependencies,
    std::string* error) {
    for (const SelectionCandidate& candidate : candidate_set.candidates) {
        std::vector<std::byte> generator;
        if (!read_selection_blob_object(root, candidate.generator_provenance,
                                        &generator, error)) {
            return fail(error,
                        "BCC-32 candidate set names a missing generator provenance object");
        }
        Manifest frozen{};
        if (!load_complete_manifest_object(root, candidate.root, &frozen, error)) {
            return fail(error,
                        "BCC-32 candidate set names an incomplete provenance closure");
        }
        dependencies->push_back(candidate.root);
        for (const ContentAddress& parent : candidate.parentage) {
            Manifest parent_manifest{};
            if (!load_complete_manifest_object(
                    root, parent, &parent_manifest, error)) {
                return fail(error,
                            "BCC-32 candidate set names an incomplete parent closure");
            }
            dependencies->push_back(parent);
        }
    }
    return true;
}

bool validate_repository_provenance(
    const std::filesystem::path& root,
    const ArtifactMetadata& metadata,
    std::span<const ChunkObjectReference> chunks,
    ArtifactKind expected_kind,
    std::vector<ContentAddress>* dependencies,
    std::string* error) {
    const EntryEvent& event = metadata.provenance.entry_event;
    if (event.kind == EntryEventKind::genesis_entry) {
        return metadata.replay_boundary.predecessor_commit == ContentAddress{}
                   ? true
                   : fail(error, "BCC-32 genesis entry has a predecessor");
    }
    if (event.kind == EntryEventKind::law_continuation) {
        const ContentAddress predecessor =
            metadata.replay_boundary.predecessor_commit;
        if (!is_valid_content_address(predecessor)) {
            return fail(error, "BCC-32 law continuation has no predecessor");
        }
        Manifest prior{};
        if (!load_complete_manifest_object(root, predecessor, &prior, error) ||
            prior.metadata.kind != expected_kind) {
            return fail(error, "BCC-32 law predecessor closure is incomplete");
        }
        dependencies->push_back(predecessor);
        return validate_law_continuation_metadata_delta(
            metadata, prior.metadata, error);
    }

    SelectionTranscript transcript{};
    if (!read_selection_transcript_impl(
            root, event.evaluator_transcript, &transcript, error)) {
        return false;
    }
    SelectionEvaluationRequest request{};
    if (!read_selection_evaluation_request_impl(
            root, transcript.evaluation_request_commitment, &request, error)) {
        return false;
    }
    SelectionCandidateSet candidate_set{};
    if (!read_selection_candidate_set_impl(
            root, request.candidate_set_commitment, &candidate_set, error) ||
        !validate_provenance(
            metadata.provenance, candidate_set, request, transcript, error) ||
        !collect_candidate_set_dependencies(
            root, candidate_set, dependencies, error)) {
        return false;
    }
    SelectionReplay replay{};
    SelectionPopulationSlot selected{};
    if (!replay_selection_transcript(
            candidate_set, request, transcript, &replay, error) ||
        !resolve_selection_population_slot(
            replay, event.next_population_slot, &selected, error)) {
        return false;
    }
    Manifest candidate{};
    if (!load_complete_manifest_object(root, selected.root, &candidate, error)) {
        return fail(error, "BCC-32 selected population slot is not a complete artifact");
    }
    if (candidate.metadata.kind != expected_kind) {
        return fail(error, "BCC-32 selected population slot has the wrong artifact kind");
    }
    if (transcript.environment_contact_input !=
            candidate.metadata.provenance.environment_contact_manifest ||
        metadata.provenance.environment_contact_manifest !=
            transcript.environment_contact_input) {
        return fail(error,
                    "BCC-32 evaluator selection cannot replace the frozen boundary manifest");
    }
    if (!validate_selection_metadata_delta(
            metadata, candidate, selected.root, error)) return false;
    if (material_state_identity(chunks) != material_state_identity(candidate.chunks)) {
        return fail(error, "BCC-32 selected publication edits the intact candidate material");
    }
    return true;
}

bool validate_provenance_identity(const std::filesystem::path& root,
                                  const ContentAddress& identity,
                                  ProvenanceClosureWalk* walk,
                                  std::string* error) {
    struct Frame {
        ContentAddress identity{};
        Manifest manifest{};
        std::vector<ContentAddress> dependencies;
        std::size_t next_dependency = 0u;
        bool expanded = false;
    };

    std::vector<Frame> stack;
    const auto push = [&](const ContentAddress& next,
                          std::vector<Frame>* frames) -> bool {
        if (walk->validated.contains(next)) return true;
        if (walk->active.contains(next)) {
            return fail(error, "BCC-32 provenance closure contains a cycle");
        }
        if (walk->validated.size() + walk->active.size() >=
            ProvenanceClosureWalk::kMaximumArtifacts) {
            return fail(error, "BCC-32 provenance closure exceeds its artifact bound");
        }
        Manifest decoded{};
        if (!load_complete_manifest_object(root, next, &decoded, error)) return false;
        walk->active.insert(next);
        frames->push_back({.identity = next, .manifest = std::move(decoded)});
        return true;
    };

    if (!push(identity, &stack)) return false;
    while (!stack.empty()) {
        Frame& frame = stack.back();
        if (!frame.expanded) {
            if (!validate_repository_provenance(root,
                                                frame.manifest.metadata,
                                                frame.manifest.chunks,
                                                frame.manifest.metadata.kind,
                                                &frame.dependencies,
                                                error)) {
                return false;
            }
            frame.expanded = true;
        }
        if (frame.next_dependency < frame.dependencies.size()) {
            const ContentAddress dependency =
                frame.dependencies[frame.next_dependency++];
            if (walk->validated.contains(dependency)) continue;
            if (!push(dependency, &stack)) return false;
            continue;
        }
        const ContentAddress completed = frame.identity;
        walk->active.erase(completed);
        walk->validated.insert(completed);
        stack.pop_back();
    }
    return true;
}

bool validate_candidate_set_references(
    const std::filesystem::path& root,
    const SelectionCandidateSet& candidate_set,
    ProvenanceClosureWalk* walk,
    std::string* error) {
    std::vector<ContentAddress> dependencies;
    if (!collect_candidate_set_dependencies(
            root, candidate_set, &dependencies, error)) {
        return false;
    }
    for (const ContentAddress& dependency : dependencies) {
        if (!validate_provenance_identity(root, dependency, walk, error)) return false;
    }
    return true;
}

bool load_complete_manifest_object_with_provenance(
    const std::filesystem::path& root,
    const ContentAddress& identity,
    Manifest* manifest,
    ProvenanceClosureWalk* walk,
    std::string* error) {
    if (walk == nullptr) {
        return fail(error, "BCC-32 provenance closure requires traversal state");
    }
    return validate_provenance_identity(root, identity, walk, error) &&
           load_complete_manifest_object(root, identity, manifest, error);
}

bool validate_selection_repository(
    const std::filesystem::path& root,
    const ArtifactMetadata& metadata,
    std::span<const ChunkObjectReference> chunks,
    ArtifactKind expected_kind,
    std::string* error) {
    ProvenanceClosureWalk walk{};
    std::vector<ContentAddress> dependencies;
    if (!validate_repository_provenance(
            root, metadata, chunks, expected_kind, &dependencies, error)) {
        return false;
    }
    for (const ContentAddress& dependency : dependencies) {
        if (!validate_provenance_identity(root, dependency, &walk, error)) return false;
    }
    return true;
}

bool persist_immutable_manifest_object(
    const ArtifactMetadata& requested_metadata,
    std::span<const ChunkObjectReference> chunks,
    std::uint64_t sequence,
    const std::filesystem::path& root,
    ContentAddress* identity,
    std::string* error) {
    if (identity == nullptr || sequence == 0u) {
        return fail(error, "BCC-32 immutable manifest requires an identity and sequence");
    }
    Manifest manifest{};
    manifest.sequence = sequence;
    manifest.metadata = requested_metadata;
    manifest.metadata.world_support = support_for_chunk_objects(chunks);
    manifest.chunks.assign(chunks.begin(), chunks.end());
    if (manifest.chunks.size() >
        std::numeric_limits<std::uint64_t>::max() / kChunkBytes) {
        return fail(error, "BCC-32 direct chunk byte count overflows");
    }
    manifest.metadata.byte_counts.direct_chunk_bytes =
        static_cast<std::uint64_t>(manifest.chunks.size()) * kChunkBytes;
    manifest.metadata.byte_counts.manifest_bytes = serialize_manifest(manifest).size();
    manifest.metadata.byte_counts.durable_bytes =
        manifest.metadata.byte_counts.direct_chunk_bytes +
        manifest.metadata.byte_counts.manifest_bytes;
    const Bytes manifest_bytes = serialize_manifest(manifest);
    if (manifest_bytes.size() != manifest.metadata.byte_counts.manifest_bytes ||
        manifest.metadata.world_support.direct_word_bytes !=
            manifest.metadata.byte_counts.direct_chunk_bytes) {
        return fail(error, "BCC-32 immutable manifest byte accounting is unstable");
    }
    const ContentAddress address{
        domain_hash(kManifestDomain, as_bytes(manifest_bytes)), manifest_bytes.size()};
    if (!write_immutable_object(
            manifest_object_path(root, address.digest), manifest_bytes, error)) {
        return false;
    }
    *identity = address;
    return true;
}

bool publish_commit_impl(const ArtifactMetadata& requested_metadata,
                         std::span<const ChunkObjectReference> chunks,
                         ArtifactKind expected_kind,
                         const std::filesystem::path& root,
                         PublicationPrecondition precondition,
                         ContentAddress* published_identity,
                         std::string* error,
                         PublicationFailurePoint failure) {
    if (!validate_metadata(requested_metadata, expected_kind, error) ||
        !ensure_repository(root, error)) {
        return false;
    }
    for (std::size_t index = 0u; index < chunks.size(); ++index) {
        const ChunkObjectReference& chunk = chunks[index];
        if (chunk.digest == Hash256{} || chunk.non_quiescent_sites == 0u ||
            chunk.non_quiescent_sites > kChunkSites ||
            (index != 0u &&
             !CoordinateLess{}(chunks[index - 1u].coordinate, chunk.coordinate))) {
            return fail(error, "BCC-32 commit chunk index is noncanonical");
        }
        std::vector<SiteWord> verified;
        if (!read_chunk(chunk_object_path(root, chunk.digest),
                        chunk.digest,
                        chunk.non_quiescent_sites,
                        chunk.delta_n_q,
                        &verified,
                        error)) {
            return false;
        }
    }
    if (!validate_selection_repository(
            root, requested_metadata, chunks, expected_kind, error)) {
        return false;
    }

    if (precondition.expectation != RootExpectation::absent &&
        precondition.expectation != RootExpectation::exact) {
        return fail(error, "BCC-32 publication precondition is invalid");
    }
    if (precondition.expectation == RootExpectation::exact &&
        !is_valid_content_address(precondition.expected_identity)) {
        return fail(error, "BCC-32 exact publication precondition has no root identity");
    }
    const ContentAddress predecessor =
        requested_metadata.replay_boundary.predecessor_commit;
    if ((precondition.expectation == RootExpectation::absent &&
         predecessor != ContentAddress{}) ||
        (precondition.expectation == RootExpectation::exact &&
         predecessor != precondition.expected_identity)) {
        return fail(error,
                    "BCC-32 predecessor commit must equal the publication CAS root");
    }

    PublicationLock publication_lock(root, error);
    if (!publication_lock.acquired()) return false;

    Manifest prior{};
    ContentAddress prior_identity{};
    const bool has_prior = load_published(root, &prior, &prior_identity, nullptr);
    const bool has_root_record = std::filesystem::exists(root / "ROOT_A") ||
                                 std::filesystem::exists(root / "ROOT_B");
    if (!has_prior && has_root_record) {
        return fail(error, "BCC-32 refuses to overwrite an unreadable lineage root");
    }
    if (precondition.expectation == RootExpectation::absent && has_prior) {
        return fail(error, "BCC-32 genesis/import cannot replace an established root");
    }
    if (precondition.expectation == RootExpectation::exact &&
        (!has_prior || prior_identity != precondition.expected_identity)) {
        return fail(error, "BCC-32 publication lost its exact parent-root race");
    }
    if (precondition.expectation == RootExpectation::exact &&
        prior.metadata.kind != expected_kind) {
        return fail(error, "BCC-32 predecessor commit has the wrong artifact kind");
    }
    if (precondition.expectation == RootExpectation::absent &&
        requested_metadata.provenance.entry_event.kind !=
            EntryEventKind::genesis_entry) {
        return fail(error, "BCC-32 a new repository requires a genesis entry event");
    }
    if (precondition.expectation == RootExpectation::exact) {
        const EntryEventKind entry_kind =
            requested_metadata.provenance.entry_event.kind;
        if (entry_kind == EntryEventKind::law_continuation) {
            if (!validate_law_continuation_metadata_delta(
                    requested_metadata, prior.metadata, error)) return false;
        } else if (entry_kind != EntryEventKind::evaluator_selection) {
            return fail(error,
                        "BCC-32 continuation requires selection or law entry provenance");
        }
    }
    if (has_prior && prior.sequence == std::numeric_limits<std::uint64_t>::max()) {
        return fail(error, "BCC-32 artifact sequence is exhausted");
    }
    const std::uint64_t sequence = has_prior ? prior.sequence + 1u : 1u;

    if (failure == PublicationFailurePoint::after_generation_sync) {
        return fail(error, "injected BCC-32 crash before manifest publication");
    }
    ContentAddress identity{};
    if (!persist_immutable_manifest_object(
            requested_metadata, chunks, sequence, root, &identity, error)) {
        return false;
    }
    if (failure == PublicationFailurePoint::after_generation_publish) {
        return fail(error, "injected BCC-32 crash before root publication");
    }

    const std::string root_name = sequence & 1u ? "ROOT_A" : "ROOT_B";
    const Bytes root_bytes = serialize_root({sequence, identity});
    const std::filesystem::path root_temporary =
        unique_temporary_path(root / root_name);
    if (!write_synced_file(root_temporary, root_bytes, error)) return false;
    if (failure == PublicationFailurePoint::after_root_temp_sync) {
        return fail(error, "injected BCC-32 crash before root replacement");
    }
    std::error_code filesystem_error;
    std::filesystem::rename(root_temporary, root / root_name, filesystem_error);
    if (filesystem_error || !sync_directory(root, error)) {
        return fail(error, "cannot publish BCC-32 root");
    }
    if (published_identity != nullptr) *published_identity = identity;
    return true;
}

bool import_artifact(const ArtifactMetadata& requested_metadata,
                    const WorldStore& world,
                    ArtifactKind expected_kind,
                    const std::filesystem::path& root,
                    ContentAddress* published_identity,
                    std::string* error,
                    PublicationFailurePoint failure) {
    if (!ensure_repository(root, error)) return false;
    ArtifactMetadata metadata = requested_metadata;
    metadata.replay_boundary.taint = ContinuationTaint::synthetic_import;
    std::vector<ChunkObjectReference> chunks;
    chunks.reserve(world.chunks().size());
    for (const auto& [coordinate, words] : world.chunks()) {
        ChunkObjectReference reference{};
        bool materialized = false;
        if (!put_chunk_object_impl(
                root, coordinate, words, &reference, &materialized, error)) return false;
        if (!materialized) {
            return fail(error, "BCC-32 WorldStore exposed an elided Q chunk");
        }
        chunks.push_back(std::move(reference));
    }
    return publish_commit_impl(metadata,
                               chunks,
                               expected_kind,
                               root,
                               PublicationPrecondition::absent(),
                               published_identity,
                               error,
                               failure);
}

bool restore_world(const std::filesystem::path& root,
                   const Manifest& manifest,
                   WorldStore* world,
                   std::string* error) {
    WorldStore candidate{};
    for (const ChunkObjectReference& chunk : manifest.chunks) {
        std::vector<SiteWord> words;
        if (!read_chunk(chunk_object_path(root, chunk.digest),
                        chunk.digest,
                        chunk.non_quiescent_sites,
                        chunk.delta_n_q,
                        &words,
                        error) ||
            !candidate.replace_chunk(chunk.coordinate, words, error)) {
            return false;
        }
    }
    if (candidate.support() != manifest.metadata.world_support ||
        candidate.support().direct_word_bytes != manifest.metadata.byte_counts.direct_chunk_bytes) {
        return fail(error, "BCC-32 restored world support does not match its manifest");
    }
    *world = std::move(candidate);
    return true;
}

bool try_restore_root(const std::filesystem::path& root,
                      const Manifest& manifest,
                      const ContentAddress& identity,
                      ArtifactKind expected_kind,
                      ArtifactMetadata* metadata,
                      WorldStore* world,
                      ContentAddress* published_identity,
                      std::string* error) {
    if (!validate_metadata(manifest.metadata, expected_kind, error) ||
        !validate_selection_repository(
            root, manifest.metadata, manifest.chunks, expected_kind, error)) {
        return false;
    }
    WorldStore candidate{};
    if (!restore_world(root, manifest, &candidate, error)) return false;
    *metadata = manifest.metadata;
    *world = std::move(candidate);
    if (published_identity != nullptr) *published_identity = identity;
    return true;
}

bool read_artifact(const std::filesystem::path& root,
                   ArtifactKind expected_kind,
                   ArtifactMetadata* metadata,
                   WorldStore* world,
                   ContentAddress* published_identity,
                   std::string* error) {
    Manifest manifest{};
    ContentAddress identity{};
    if (!load_published(root, &manifest, &identity, error)) return false;
    return try_restore_root(root,
                            manifest,
                            identity,
                            expected_kind,
                            metadata,
                            world,
                            published_identity,
                            error);
}

}  // namespace

bool put_selection_blob_object(const std::filesystem::path& root,
                               std::span<const std::byte> bytes,
                               ContentAddress* identity,
                               std::string* error) {
    if (identity == nullptr || bytes.size() > kMaximumSelectionBlobBytes ||
        !ensure_repository(root, error)) {
        return fail(error, "BCC-32 selection blob publication is invalid");
    }
    const ContentAddress address = content_address(bytes);
    if (!write_immutable_object(selection_blob_object_path(root, address.digest),
                                copy_bytes(bytes), error)) {
        return false;
    }
    *identity = address;
    return true;
}

bool read_selection_blob_object(const std::filesystem::path& root,
                                const ContentAddress& identity,
                                std::vector<std::byte>* bytes,
                                std::string* error) {
    if (bytes == nullptr || !is_valid_content_address(identity) ||
        identity.byte_count > kMaximumSelectionBlobBytes) {
        return fail(error, "BCC-32 selection blob lookup is invalid");
    }
    Bytes encoded;
    if (!read_exact_regular_file(
            selection_blob_object_path(root, identity.digest), identity.byte_count,
            &encoded, error) ||
        content_address(as_bytes(encoded)) != identity) {
        return fail(error, "BCC-32 selection blob is missing or corrupt");
    }
    bytes->resize(encoded.size());
    if (!encoded.empty()) {
        std::memcpy(bytes->data(), encoded.data(), encoded.size());
    }
    return true;
}

bool put_selection_candidate_set_object(
    const std::filesystem::path& root,
    const SelectionCandidateSet& candidate_set,
    ContentAddress* identity,
    std::string* error) {
    if (!validate_selection_candidate_set(candidate_set, error) ||
        !ensure_repository(root, error)) {
        return false;
    }
    ProvenanceClosureWalk walk{};
    if (!validate_candidate_set_references(root, candidate_set, &walk, error)) return false;
    const std::vector<std::byte> canonical =
        canonical_selection_candidate_set(candidate_set);
    const ContentAddress address = content_address(canonical);
    const Bytes encoded = copy_bytes(canonical);
    if (!write_immutable_object(
            selection_candidate_set_object_path(root, address.digest),
            encoded,
            error)) {
        return false;
    }
    if (identity != nullptr) *identity = address;
    return true;
}

bool read_selection_candidate_set_object(
    const std::filesystem::path& root,
    const ContentAddress& identity,
    SelectionCandidateSet* candidate_set,
    std::string* error) {
    SelectionCandidateSet decoded{};
    if (!read_selection_candidate_set_impl(root, identity, &decoded, error)) {
        return false;
    }
    ProvenanceClosureWalk walk{};
    if (!validate_candidate_set_references(root, decoded, &walk, error)) return false;
    if (candidate_set == nullptr) {
        return fail(error, "BCC-32 candidate-set read requires an output");
    }
    *candidate_set = std::move(decoded);
    return true;
}

bool put_selection_evaluation_request_object(
    const std::filesystem::path& root,
    const SelectionEvaluationRequest& request,
    ContentAddress* identity,
    std::string* error) {
    if (!ensure_repository(root, error)) return false;
    SelectionCandidateSet candidate_set{};
    if (!read_selection_candidate_set_object(root,
                                             request.candidate_set_commitment,
                                             &candidate_set,
                                             error) ||
        !validate_selection_evaluation_request(candidate_set, request, error)) {
        return false;
    }
    const std::vector<std::byte> canonical =
        canonical_selection_evaluation_request(request);
    const ContentAddress address = content_address(canonical);
    if (!write_immutable_object(
            selection_evaluation_request_object_path(root, address.digest),
            copy_bytes(canonical),
            error)) {
        return false;
    }
    if (identity != nullptr) *identity = address;
    return true;
}

bool read_selection_evaluation_request_object(
    const std::filesystem::path& root,
    const ContentAddress& identity,
    SelectionEvaluationRequest* request,
    std::string* error) {
    SelectionEvaluationRequest decoded{};
    if (!read_selection_evaluation_request_impl(root, identity, &decoded, error)) {
        return false;
    }
    SelectionCandidateSet candidate_set{};
    if (!read_selection_candidate_set_object(root,
                                             decoded.candidate_set_commitment,
                                             &candidate_set,
                                             error) ||
        !validate_selection_evaluation_request(candidate_set, decoded, error)) {
        return false;
    }
    if (request == nullptr) {
        return fail(error, "BCC-32 selection-request read requires an output");
    }
    *request = std::move(decoded);
    return true;
}

bool put_selection_transcript_object(
    const std::filesystem::path& root,
    const SelectionTranscript& transcript,
    ContentAddress* identity,
    std::string* error) {
    if (!ensure_repository(root, error)) return false;
    SelectionEvaluationRequest request{};
    if (!read_selection_evaluation_request_object(
            root, transcript.evaluation_request_commitment, &request, error)) {
        return false;
    }
    SelectionCandidateSet candidate_set{};
    if (!read_selection_candidate_set_object(root,
                                             request.candidate_set_commitment,
                                             &candidate_set,
                                             error) ||
        !validate_selection_transcript(
            candidate_set, request, transcript, error)) {
        return false;
    }
    const std::vector<std::byte> canonical =
        canonical_selection_transcript(transcript);
    const ContentAddress address = content_address(canonical);
    const Bytes encoded = copy_bytes(canonical);
    if (!write_immutable_object(
            selection_transcript_object_path(root, address.digest),
            encoded,
            error)) {
        return false;
    }
    if (identity != nullptr) *identity = address;
    return true;
}

bool read_selection_transcript_object(
    const std::filesystem::path& root,
    const ContentAddress& identity,
    SelectionTranscript* transcript,
    std::string* error) {
    SelectionTranscript decoded{};
    if (!read_selection_transcript_impl(root, identity, &decoded, error)) {
        return false;
    }
    SelectionEvaluationRequest request{};
    if (!read_selection_evaluation_request_object(
            root, decoded.evaluation_request_commitment, &request, error)) {
        return false;
    }
    SelectionCandidateSet candidate_set{};
    if (!read_selection_candidate_set_object(root,
                                             request.candidate_set_commitment,
                                             &candidate_set,
                                             error) ||
        !validate_selection_transcript(
            candidate_set, request, decoded, error)) {
        return false;
    }
    if (transcript == nullptr) {
        return fail(error, "BCC-32 selection-transcript read requires an output");
    }
    *transcript = std::move(decoded);
    return true;
}

bool materialize_selection_population_slot(
    const std::filesystem::path& root,
    const ContentAddress& transcript_identity,
    std::uint64_t slot,
    ArtifactKind expected_kind,
    ContentAddress* selected_identity,
    std::string* error) {
    if (selected_identity == nullptr || !ensure_repository(root, error)) {
        return fail(error, "BCC-32 slot materialization requires an identity output");
    }
    SelectionTranscript transcript{};
    if (!read_selection_transcript_object(
            root, transcript_identity, &transcript, error)) {
        return false;
    }
    SelectionEvaluationRequest request{};
    if (!read_selection_evaluation_request_object(
            root, transcript.evaluation_request_commitment, &request, error)) {
        return false;
    }
    SelectionCandidateSet candidate_set{};
    if (!read_selection_candidate_set_object(root,
                                             request.candidate_set_commitment,
                                             &candidate_set,
                                             error)) {
        return false;
    }
    SelectionReplay replay{};
    SelectionPopulationSlot population_slot{};
    if (!replay_selection_transcript(
            candidate_set, request, transcript, &replay, error) ||
        !resolve_selection_population_slot(
            replay, slot, &population_slot, error)) {
        return false;
    }
    Manifest candidate{};
    if (!load_complete_manifest_object(
            root, population_slot.root, &candidate, error) ||
        candidate.metadata.kind != expected_kind) {
        return fail(error, "BCC-32 selected slot has the wrong artifact kind");
    }
    if (candidate.sequence == std::numeric_limits<std::uint64_t>::max()) {
        return fail(error, "BCC-32 selected slot sequence is exhausted");
    }

    ArtifactMetadata metadata = candidate.metadata;
    metadata.replay_boundary.predecessor_commit = population_slot.root;
    metadata.provenance.entry_event = {
        .kind = EntryEventKind::evaluator_selection,
        .evaluator_transcript = transcript_identity,
        .next_population_slot = slot,
    };
    metadata.provenance.replay_commitment = population_slot.root;
    if (!validate_metadata(metadata, expected_kind, error) ||
        !validate_selection_repository(
            root, metadata, candidate.chunks, expected_kind, error)) {
        return false;
    }

    return persist_immutable_manifest_object(metadata,
                                             candidate.chunks,
                                             candidate.sequence + 1u,
                                             root,
                                             selected_identity,
                                             error);
}

WorldSupport support_for_chunk_objects(
    std::span<const ChunkObjectReference> chunks) {
    WorldSupport support{};
    if (chunks.empty()) return support;
    support.has_chunks = true;
    support.minimum = chunks.front().coordinate;
    support.maximum = chunks.front().coordinate;
    support.materialized_chunks = static_cast<std::uint64_t>(chunks.size());
    support.direct_word_bytes = support.materialized_chunks * kChunkBytes;
    for (const ChunkObjectReference& chunk : chunks) {
        support.minimum.x = std::min(support.minimum.x, chunk.coordinate.x);
        support.minimum.y = std::min(support.minimum.y, chunk.coordinate.y);
        support.minimum.z = std::min(support.minimum.z, chunk.coordinate.z);
        support.maximum.x = std::max(support.maximum.x, chunk.coordinate.x);
        support.maximum.y = std::max(support.maximum.y, chunk.coordinate.y);
        support.maximum.z = std::max(support.maximum.z, chunk.coordinate.z);
        support.non_quiescent_sites += chunk.non_quiescent_sites;
    }
    return support;
}

ContentAddress material_state_identity(
    std::span<const ChunkObjectReference> chunks) {
    std::vector<ChunkObjectReference> sorted;
    if (!std::is_sorted(
            chunks.begin(),
            chunks.end(),
            [](const ChunkObjectReference& left,
               const ChunkObjectReference& right) {
                return CoordinateLess{}(left.coordinate, right.coordinate);
            })) {
        sorted.assign(chunks.begin(), chunks.end());
        std::sort(sorted.begin(),
                  sorted.end(),
                  [](const ChunkObjectReference& left,
                     const ChunkObjectReference& right) {
                      return CoordinateLess{}(left.coordinate, right.coordinate);
                  });
        chunks = sorted;
    }
    Bytes bytes;
    put_u64(&bytes, chunks.size());
    for (const ChunkObjectReference& chunk : chunks) {
        put_coordinate_component(&bytes, chunk.coordinate.x);
        put_coordinate_component(&bytes, chunk.coordinate.y);
        put_coordinate_component(&bytes, chunk.coordinate.z);
        put_hash(&bytes, chunk.digest);
        put_u64(&bytes, chunk.non_quiescent_sites);
        put_i64(&bytes, chunk.delta_n_q);
    }
    return {domain_hash(kMaterialStateDomain, as_bytes(bytes)), bytes.size()};
}

bool put_world_chunk_object(const std::filesystem::path& root,
                            const ChunkCoord& coordinate,
                            std::span<const SiteWord> words,
                            ChunkObjectReference* reference,
                            bool* materialized,
                            std::string* error) {
    return ensure_repository(root, error) &&
           put_chunk_object_impl(
               root, coordinate, words, reference, materialized, error);
}

bool read_world_chunk_object(const std::filesystem::path& root,
                             const ChunkObjectReference& reference,
                             std::vector<SiteWord>* words,
                             std::string* error) {
    if (words == nullptr || reference.digest == Hash256{} ||
        reference.non_quiescent_sites == 0u ||
        reference.non_quiescent_sites > kChunkSites) {
        return fail(error, "BCC-32 chunk object reference is invalid");
    }
    return read_chunk(chunk_object_path(root, reference.digest),
                      reference.digest,
                      reference.non_quiescent_sites,
                      reference.delta_n_q,
                      words,
                      error);
}

bool publish_world_commit(const ArtifactMetadata& metadata,
                          std::span<const ChunkObjectReference> chunks,
                          ArtifactKind expected_kind,
                          const std::filesystem::path& root,
                          PublicationPrecondition precondition,
                          ContentAddress* published_identity,
                          std::string* error,
                          PublicationFailurePoint failure) {
    if (metadata.provenance.entry_event.kind ==
        EntryEventKind::law_continuation) {
        return fail(error,
                    "BCC-32 law continuation requires production-law authority");
    }
    return publish_commit_impl(metadata,
                               chunks,
                               expected_kind,
                               root,
                               precondition,
                               published_identity,
                               error,
                               failure);
}

namespace detail {

__attribute__((visibility("hidden")))
bool publish_law_continuation_from_executor(
    const ArtifactMetadata& metadata,
    std::span<const ChunkObjectReference> chunks,
    ArtifactKind expected_kind,
    const std::filesystem::path& root,
    PublicationPrecondition precondition,
    const ContentAddress& expected_predecessor,
    const ContentAddress& expected_material_state,
    std::uint64_t expected_completed_supersteps,
    ContentAddress* published_identity,
    std::string* error,
    PublicationFailurePoint failure) {
    if (metadata.provenance.entry_event.kind !=
        EntryEventKind::law_continuation) {
        return fail(error,
                    "BCC-32 production-law publication requires a law continuation");
    }
    if (precondition.expectation != RootExpectation::exact ||
        precondition.expected_identity != expected_predecessor ||
        metadata.replay_boundary.predecessor_commit != expected_predecessor ||
        metadata.replay_boundary.completed_supersteps !=
            expected_completed_supersteps ||
        material_state_identity(chunks) != expected_material_state) {
        return fail(error,
                    "BCC-32 production-law authority does not match the publication");
    }
    return publish_commit_impl(metadata,
                               chunks,
                               expected_kind,
                               root,
                               precondition,
                               published_identity,
                               error,
                               failure);
}

__attribute__((visibility("hidden")))
bool publish_immutable_law_continuation_from_executor(
    const ArtifactMetadata& metadata,
    std::span<const ChunkObjectReference> chunks,
    ArtifactKind expected_kind,
    const std::filesystem::path& root,
    const ContentAddress& expected_predecessor,
    const ContentAddress& expected_material_state,
    std::uint64_t expected_completed_supersteps,
    ContentAddress* published_identity,
    std::string* error,
    PublicationFailurePoint failure) {
    if (metadata.provenance.entry_event.kind !=
            EntryEventKind::law_continuation ||
        metadata.replay_boundary.predecessor_commit != expected_predecessor ||
        metadata.replay_boundary.completed_supersteps !=
            expected_completed_supersteps ||
        material_state_identity(chunks) != expected_material_state) {
        return fail(error,
                    "BCC-32 immutable law authority does not match the continuation");
    }
    if (!validate_metadata(metadata, expected_kind, error) ||
        !ensure_repository(root, error)) {
        return false;
    }
    for (std::size_t index = 0u; index < chunks.size(); ++index) {
        const ChunkObjectReference& chunk = chunks[index];
        if (chunk.digest == Hash256{} || chunk.non_quiescent_sites == 0u ||
            chunk.non_quiescent_sites > kChunkSites ||
            (index != 0u &&
             !CoordinateLess{}(chunks[index - 1u].coordinate, chunk.coordinate))) {
            return fail(error, "BCC-32 immutable law chunk index is noncanonical");
        }
        std::vector<SiteWord> verified;
        if (!read_chunk(chunk_object_path(root, chunk.digest),
                        chunk.digest,
                        chunk.non_quiescent_sites,
                        chunk.delta_n_q,
                        &verified,
                        error)) {
            return false;
        }
    }
    if (!validate_selection_repository(
            root, metadata, chunks, expected_kind, error)) {
        return false;
    }
    Manifest predecessor{};
    if (!load_complete_manifest_object(
            root, expected_predecessor, &predecessor, error) ||
        predecessor.metadata.kind != expected_kind) {
        return fail(error, "BCC-32 immutable law predecessor is incomplete");
    }
    if (predecessor.sequence == std::numeric_limits<std::uint64_t>::max()) {
        return fail(error, "BCC-32 immutable law sequence is exhausted");
    }
    if (failure == PublicationFailurePoint::after_generation_sync) {
        return fail(error, "injected BCC-32 crash before immutable law publication");
    }
    if (!persist_immutable_manifest_object(metadata,
                                           chunks,
                                           predecessor.sequence + 1u,
                                           root,
                                           published_identity,
                                           error)) {
        return false;
    }
    if (failure == PublicationFailurePoint::after_generation_publish) {
        return fail(error, "injected BCC-32 crash after immutable law publication");
    }
    if (failure == PublicationFailurePoint::after_root_temp_sync) {
        return fail(error,
                    "BCC-32 immutable law publication has no mutable root phase");
    }
    return true;
}

}  // namespace detail

bool load_world_commit(const std::filesystem::path& root,
                       ArtifactKind expected_kind,
                       WorldCommit* commit,
                       std::string* error) {
    if (commit == nullptr) return fail(error, "BCC-32 world commit output is null");
    Manifest manifest{};
    ContentAddress identity{};
    if (!load_published(root, &manifest, &identity, error) ||
        !validate_metadata(manifest.metadata, expected_kind, error) ||
        !validate_selection_repository(
            root, manifest.metadata, manifest.chunks, expected_kind, error)) {
        return false;
    }
    commit->publication_sequence = manifest.sequence;
    commit->metadata = std::move(manifest.metadata);
    commit->identity = identity;
    commit->chunks = std::move(manifest.chunks);
    return true;
}

bool load_world_commit_object(const std::filesystem::path& root,
                              ArtifactKind expected_kind,
                              const ContentAddress& identity,
                              WorldCommit* commit,
                              std::string* error) {
    if (commit == nullptr || !is_valid_content_address(identity)) {
        return fail(error, "BCC-32 immutable world-commit lookup is invalid");
    }
    Manifest manifest{};
    ProvenanceClosureWalk walk{};
    if (!load_complete_manifest_object_with_provenance(
            root, identity, &manifest, &walk, error)) return false;
    if (manifest.metadata.kind != expected_kind) {
        return fail(error, "BCC-32 immutable world commit has the wrong artifact kind");
    }
    commit->publication_sequence = 0u;
    commit->metadata = std::move(manifest.metadata);
    commit->identity = identity;
    commit->chunks = std::move(manifest.chunks);
    return true;
}

bool import_propagule_capsule(const PropaguleCapsule& capsule,
                              const std::filesystem::path& root,
                              ContentAddress* published_identity,
                              std::string* error,
                              PublicationFailurePoint failure) {
    return import_artifact(capsule.metadata,
                           capsule.world,
                           ArtifactKind::propagule_capsule,
                           root,
                           published_identity,
                           error,
                           failure);
}

bool import_adult_continuity_checkpoint(
    const AdultContinuityCheckpoint& checkpoint,
    const std::filesystem::path& root,
    ContentAddress* published_identity,
    std::string* error,
    PublicationFailurePoint failure) {
    return import_artifact(checkpoint.metadata,
                           checkpoint.world,
                           ArtifactKind::adult_continuity_checkpoint,
                           root,
                           published_identity,
                           error,
                           failure);
}

bool import_cultural_capsule(const CulturalCapsule& capsule,
                             const std::filesystem::path& root,
                             ContentAddress* published_identity,
                             std::string* error,
                             PublicationFailurePoint failure) {
    return import_artifact(capsule.metadata,
                           capsule.world,
                           ArtifactKind::cultural_capsule,
                           root,
                           published_identity,
                           error,
                           failure);
}

bool read_propagule_capsule(const std::filesystem::path& root,
                            PropaguleCapsule* capsule,
                            ContentAddress* published_identity,
                            std::string* error) {
    if (capsule == nullptr) return fail(error, "BCC-32 propagule capsule output is null");
    return read_artifact(root,
                         ArtifactKind::propagule_capsule,
                         &capsule->metadata,
                         &capsule->world,
                         published_identity,
                         error);
}

bool read_adult_continuity_checkpoint(
    const std::filesystem::path& root,
    AdultContinuityCheckpoint* checkpoint,
    ContentAddress* published_identity,
    std::string* error) {
    if (checkpoint == nullptr) {
        return fail(error, "BCC-32 adult continuity output is null");
    }
    return read_artifact(root,
                         ArtifactKind::adult_continuity_checkpoint,
                         &checkpoint->metadata,
                         &checkpoint->world,
                         published_identity,
                         error);
}

bool read_cultural_capsule(const std::filesystem::path& root,
                           CulturalCapsule* capsule,
                           ContentAddress* published_identity,
                           std::string* error) {
    if (capsule == nullptr) return fail(error, "BCC-32 cultural capsule output is null");
    return read_artifact(root,
                         ArtifactKind::cultural_capsule,
                         &capsule->metadata,
                         &capsule->world,
                         published_identity,
                         error);
}

}  // namespace substrate::bcc32
