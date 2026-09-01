#include "bcc32_world_entry.hpp"

#include "bcc32_law_identity.hpp"

#include <algorithm>
#include <string_view>

namespace substrate::bcc32 {
namespace {

bool fail(std::string* error, const std::string& message) {
    if (error != nullptr) *error = message;
    return false;
}

ContentAddress address_of(std::string_view text) {
    return content_address({reinterpret_cast<const std::byte*>(text.data()), text.size()});
}

ContentAddress derive_address(std::string_view domain,
                              const ContentAddress& source) {
    const std::string bytes = std::string(domain) + "|" + hash_hex(source.digest) + "|" +
                              std::to_string(source.byte_count);
    return address_of(bytes);
}

}  // namespace

bool establish_world(const Genesis& genesis,
                     const SiteCoord& origin,
                     const std::filesystem::path& repository,
                     WorldCommit* commit,
                     std::string* error,
                     const std::filesystem::path& lineage_repository) {
    if (commit == nullptr) return fail(error, "BCC32 world-entry output is null");
    if (std::filesystem::exists(repository / "ROOT_A") ||
        std::filesystem::exists(repository / "ROOT_B")) {
        return fail(error, "BCC32 genesis cannot replace an existing lineage root");
    }
    const ContentAddress law = canonical_law_identity();
    if (!validate_genesis(genesis, law).passed()) {
        return fail(error, "BCC32 genesis failed its complete physical audit");
    }
    const std::optional<EncodedGenesis> encoded = encode_genesis(genesis, law);
    if (!encoded.has_value()) return fail(error, "BCC32 genesis is not canonically encodable");
    const ContentAddress genesis_identity = genesis_artifact_identity(*encoded);

    if (genesis.metadata.genesis_class == GenesisClass::G2) {
        if (lineage_repository.empty()) {
            return fail(error, "BCC32 G2 entry requires its immutable lineage repository");
        }
        for (const ContentAddress& parent : genesis.metadata.parent_identities) {
            WorldCommit parent_commit{};
            if (!load_world_commit_object(lineage_repository,
                                          ArtifactKind::adult_continuity_checkpoint,
                                          parent,
                                          &parent_commit,
                                          error)) {
                return fail(error, "BCC32 G2 biological parent is not a complete adult artifact");
            }
        }
        WorldCommit replay_source{};
        if (!load_world_commit_object(lineage_repository,
                                      ArtifactKind::adult_continuity_checkpoint,
                                      genesis.metadata.replay_identity,
                                      &replay_source,
                                      error)) {
            return fail(error, "BCC32 G2 replay source is not a complete adult artifact");
        }
    }

    AdultContinuityCheckpoint checkpoint{};
    checkpoint.metadata.provenance.genesis_class = genesis.metadata.genesis_class;
    checkpoint.metadata.provenance.law = law;
    checkpoint.metadata.provenance.genesis = genesis_identity;
    checkpoint.metadata.provenance.environment_contact_manifest =
        address_of("bcc32/contact-boundary/closed/v1");
    checkpoint.metadata.provenance.entry_event = {
        .kind = EntryEventKind::genesis_entry,
        .genesis_entry = genesis_identity,
    };
    checkpoint.metadata.provenance.replay_commitment =
        genesis.metadata.genesis_class == GenesisClass::G2
            ? genesis.metadata.replay_identity
            : derive_address("bcc32/genesis-replay/v1", genesis_identity);
    if (genesis.metadata.genesis_class == GenesisClass::G2) {
        checkpoint.metadata.provenance.biological_parents =
            genesis.metadata.parent_identities;
        // A portable capsule supplied by the host has not, by reference
        // resolution alone, proved a law-native propagule export. Preserve
        // that distinction monotonically rather than laundering it as nature.
        checkpoint.metadata.replay_boundary.taint =
            ContinuationTaint::synthetic_import;
    }

    for (const SitePlacement& placement : genesis.sites) {
        const SiteCoord coordinate{
            origin.x + placement.coordinate.x,
            origin.y + placement.coordinate.y,
            origin.z + placement.coordinate.z,
        };
        if (!checkpoint.world.write_site(coordinate, placement.word, error)) return false;
    }

    std::vector<ChunkObjectReference> chunks;
    chunks.reserve(checkpoint.world.chunks().size());
    for (const auto& [coordinate, words] : checkpoint.world.chunks()) {
        ChunkObjectReference reference{};
        bool materialized = false;
        if (!put_world_chunk_object(
                repository, coordinate, words, &reference, &materialized, error)) {
            return false;
        }
        if (!materialized) {
            return fail(error, "BCC32 genesis exposed an elided all-Q chunk");
        }
        chunks.push_back(std::move(reference));
    }
    if (!publish_world_commit(checkpoint.metadata,
                              chunks,
                              ArtifactKind::adult_continuity_checkpoint,
                              repository,
                              PublicationPrecondition::absent(),
                              nullptr,
                              error)) {
        return false;
    }
    return load_world_commit(repository,
                             ArtifactKind::adult_continuity_checkpoint,
                             commit,
                             error);
}

}  // namespace substrate::bcc32
