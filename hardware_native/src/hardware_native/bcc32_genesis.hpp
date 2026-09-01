#pragma once

#include "bcc32_geometry.cuh"
#include "bcc32_provenance.hpp"
#include "bcc32_types.cuh"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <type_traits>
#include <utility>
#include <vector>

namespace substrate::bcc32 {

enum class ArtifactType : std::uint32_t {
    uniform_material = 0u,
    authored_material = 1u,
    propagule_capsule = 2u,
    continuity_checkpoint = 3u,
    cultural_capsule = 4u,
};

struct SitePlacement {
    Int3 coordinate{};
    SiteWord word = kQuiescentWord;

    friend bool operator==(const SitePlacement&, const SitePlacement&) = default;
};

constexpr std::uint32_t kGenesisFormat = 0x33474342u;
constexpr std::uint64_t kEncodedHeaderBytes = 184u;
constexpr std::uint64_t kEncodedAddressBytes = 40u;
constexpr std::uint64_t kEncodedSiteBytes = 16u;

struct GenesisMetadata {
    std::uint32_t format = kGenesisFormat;
    GenesisClass genesis_class = GenesisClass::G0;
    ArtifactType artifact_type = ArtifactType::uniform_material;
    ContentAddress law_identity{};
    Hash256 content_hash{};
    Int3 extent_min{};
    Int3 extent_max{};
    std::uint64_t support_components = 0u;
    std::uint32_t contact_boundary_mask = 0u;
    std::vector<ContentAddress> parent_identities;
    ContentAddress replay_identity{};
};

struct Genesis {
    GenesisMetadata metadata{};
    std::vector<SitePlacement> sites;
};

using EncodedGenesis = std::vector<std::byte>;

class G1Capsule {
public:
    G1Capsule(const G1Capsule&) = default;
    G1Capsule& operator=(const G1Capsule&) = default;

private:
    explicit G1Capsule(Genesis genesis) : genesis_(std::move(genesis)) {}

    Genesis genesis_{};

    friend std::optional<G1Capsule> open_g1_capsule(
        std::span<const std::byte> bytes,
        const ContentAddress& law_identity);
    friend Genesis compile_g1(const G1Capsule& capsule);
};

class G2Capsule {
public:
    G2Capsule(const G2Capsule&) = default;
    G2Capsule& operator=(const G2Capsule&) = default;

private:
    explicit G2Capsule(Genesis genesis) : genesis_(std::move(genesis)) {}

    Genesis genesis_{};

    friend std::optional<G2Capsule> open_g2_capsule(
        std::span<const std::byte> bytes,
        const ContentAddress& law_identity);
    friend Genesis compile_g2(const G2Capsule& capsule);
};

enum GenesisAuditBit : std::uint32_t {
    kAuditFinite = 1u << 0u,
    kAuditKnownClass = 1u << 1u,
    kAuditKnownArtifactType = 1u << 2u,
    kAuditLawBinding = 1u << 3u,
    kAuditExactExtent = 1u << 4u,
    kAuditInjectiveCoordinates = 1u << 5u,
    kAuditExactSupport = 1u << 6u,
    kAuditAddressProvenance = 1u << 7u,
    kAuditCanonicalLength = 1u << 8u,
    kAuditContentHash = 1u << 9u,
    kAuditCanonicalOrder = 1u << 10u,
};

constexpr std::uint32_t kRequiredGenesisAudit =
    kAuditFinite | kAuditKnownClass | kAuditKnownArtifactType |
    kAuditLawBinding | kAuditExactExtent | kAuditInjectiveCoordinates |
    kAuditExactSupport | kAuditAddressProvenance |
    kAuditCanonicalLength | kAuditContentHash | kAuditCanonicalOrder;

struct GenesisAudit {
    std::uint32_t bits = 0u;

    [[nodiscard]] constexpr bool passed() const {
        return (bits & kRequiredGenesisAudit) == kRequiredGenesisAudit;
    }
};

[[nodiscard]] std::optional<Genesis> compile_g0(
    const ContentAddress& law_identity);
[[nodiscard]] std::optional<EncodedGenesis> seal_genesis(
    Genesis genesis,
    const ContentAddress& law_identity);
[[nodiscard]] std::optional<G1Capsule> open_g1_capsule(
    std::span<const std::byte> bytes,
    const ContentAddress& law_identity);
[[nodiscard]] std::optional<G2Capsule> open_g2_capsule(
    std::span<const std::byte> bytes,
    const ContentAddress& law_identity);
[[nodiscard]] Genesis compile_g1(const G1Capsule& capsule);
[[nodiscard]] Genesis compile_g2(const G2Capsule& capsule);
[[nodiscard]] GenesisAudit validate_genesis(
    const Genesis& genesis,
    const ContentAddress& law_identity);
[[nodiscard]] std::optional<EncodedGenesis> encode_genesis(
    const Genesis& genesis,
    const ContentAddress& law_identity);
[[nodiscard]] ContentAddress genesis_artifact_identity(
    std::span<const std::byte> encoded_genesis);

static_assert(std::is_trivially_copyable_v<SitePlacement>);
static_assert(std::is_same_v<EncodedGenesis, std::vector<std::byte>>);

}  // namespace substrate::bcc32
