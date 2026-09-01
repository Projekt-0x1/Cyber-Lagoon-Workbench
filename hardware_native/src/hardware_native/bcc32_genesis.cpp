#include "bcc32_genesis.hpp"

#include <algorithm>
#include <bit>
#include <limits>
#include <new>
#include <stdexcept>
#include <string_view>
#include <unordered_map>
#include <utility>

namespace substrate::bcc32 {
namespace {

[[nodiscard]] bool hash_is_zero(const Hash256& value) {
    return std::all_of(value.begin(), value.end(), [](std::uint8_t byte) {
        return byte == 0u;
    });
}

[[nodiscard]] bool valid_address(const ContentAddress& address) {
    return !hash_is_zero(address.digest) && address.byte_count != 0u;
}

[[nodiscard]] bool empty_address(const ContentAddress& address) {
    return hash_is_zero(address.digest) && address.byte_count == 0u;
}

[[nodiscard]] bool known_class(GenesisClass genesis_class) {
    const std::uint32_t raw = static_cast<std::uint32_t>(genesis_class);
    return raw <= static_cast<std::uint32_t>(GenesisClass::G2);
}

[[nodiscard]] bool known_artifact_type(ArtifactType artifact_type) {
    const std::uint32_t raw = static_cast<std::uint32_t>(artifact_type);
    return raw <= static_cast<std::uint32_t>(ArtifactType::cultural_capsule);
}

[[nodiscard]] std::optional<std::uint64_t> encoded_length(
    std::uint64_t parent_count,
    std::uint64_t site_count) {
    constexpr std::uint64_t maximum = std::numeric_limits<std::uint64_t>::max();
    if (parent_count > (maximum - kEncodedHeaderBytes) / kEncodedAddressBytes) {
        return std::nullopt;
    }
    const std::uint64_t after_parents =
        kEncodedHeaderBytes + parent_count * kEncodedAddressBytes;
    if (site_count > (maximum - after_parents) / kEncodedSiteBytes) {
        return std::nullopt;
    }
    return after_parents + site_count * kEncodedSiteBytes;
}

[[nodiscard]] bool finite(const Genesis& genesis) {
    if (genesis.sites.empty() ||
        (genesis.metadata.contact_boundary_mask & ~0xffu) != 0u) {
        return false;
    }
    const auto parent_count =
        static_cast<std::uint64_t>(genesis.metadata.parent_identities.size());
    const auto site_count = static_cast<std::uint64_t>(genesis.sites.size());
    return encoded_length(parent_count, site_count).has_value();
}

struct Int3Hash {
    [[nodiscard]] std::size_t operator()(Int3 coordinate) const {
        std::uint64_t value = static_cast<std::uint32_t>(coordinate.x);
        value ^= static_cast<std::uint64_t>(
                     static_cast<std::uint32_t>(coordinate.y))
                 << 21u;
        value ^= static_cast<std::uint64_t>(
                     static_cast<std::uint32_t>(coordinate.z))
                 << 42u;
        value ^= value >> 29u;
        value *= 0x9e3779b185ebca87ull;
        return static_cast<std::size_t>(value ^ (value >> 32u));
    }
};

using CoordinateIndex =
    std::unordered_map<Int3, std::size_t, Int3Hash>;

[[nodiscard]] bool coordinate_less(Int3 left, Int3 right) {
    if (left.x != right.x) return left.x < right.x;
    if (left.y != right.y) return left.y < right.y;
    return left.z < right.z;
}

[[nodiscard]] bool sites_are_canonical(const Genesis& genesis) {
    return std::adjacent_find(
               genesis.sites.begin(),
               genesis.sites.end(),
               [](const SitePlacement& left, const SitePlacement& right) {
                   return !coordinate_less(left.coordinate, right.coordinate);
               }) == genesis.sites.end();
}

[[nodiscard]] bool build_coordinate_index(const Genesis& genesis,
                                          CoordinateIndex* index) {
    index->clear();
    index->reserve(genesis.sites.size());
    for (std::size_t position = 0u; position < genesis.sites.size(); ++position) {
        if (!index->emplace(genesis.sites[position].coordinate, position).second) {
            return false;
        }
    }
    return true;
}

[[nodiscard]] std::pair<Int3, Int3> exact_extent(const Genesis& genesis) {
    Int3 low = genesis.sites.front().coordinate;
    Int3 high = low;
    for (const SitePlacement& site : genesis.sites) {
        low.x = std::min(low.x, site.coordinate.x);
        low.y = std::min(low.y, site.coordinate.y);
        low.z = std::min(low.z, site.coordinate.z);
        high.x = std::max(high.x, site.coordinate.x);
        high.y = std::max(high.y, site.coordinate.y);
        high.z = std::max(high.z, site.coordinate.z);
    }
    return {low, high};
}

[[nodiscard]] std::uint64_t support_components(
    const Genesis& genesis,
    const CoordinateIndex& index) {
    std::vector<bool> reached(genesis.sites.size(), false);
    std::vector<std::size_t> pending;
    pending.reserve(genesis.sites.size());
    std::uint64_t components = 0u;
    for (std::size_t start = 0u; start < genesis.sites.size(); ++start) {
        if (reached[start]) continue;
        ++components;
        pending.clear();
        pending.push_back(start);
        reached[start] = true;
        for (std::size_t cursor = 0u; cursor < pending.size(); ++cursor) {
            const SitePlacement& current = genesis.sites[pending[cursor]];
            for (std::uint32_t raw = 0u; raw < 8u; ++raw) {
                const auto found = index.find(
                    current.coordinate +
                    direction_offset(static_cast<Direction>(raw)));
                if (found == index.end() || reached[found->second]) continue;
                reached[found->second] = true;
                pending.push_back(found->second);
            }
        }
    }
    return components;
}

[[nodiscard]] bool provenance_is_addressed(const Genesis& genesis) {
    if (!valid_address(genesis.metadata.law_identity)) return false;
    if ((genesis.metadata.genesis_class == GenesisClass::G0 ||
         genesis.metadata.genesis_class == GenesisClass::G1) &&
        !genesis.metadata.parent_identities.empty()) {
        return false;
    }
    if (genesis.metadata.genesis_class == GenesisClass::G2 &&
        (genesis.metadata.parent_identities.empty() ||
         !valid_address(genesis.metadata.replay_identity))) {
        return false;
    }
    if (!empty_address(genesis.metadata.replay_identity) &&
        !valid_address(genesis.metadata.replay_identity)) {
        return false;
    }
    return std::all_of(
        genesis.metadata.parent_identities.begin(),
        genesis.metadata.parent_identities.end(), valid_address);
}

void append_u32(EncodedGenesis* output, std::uint32_t value) {
    for (std::uint32_t shift = 0u; shift < 32u; shift += 8u) {
        output->push_back(static_cast<std::byte>((value >> shift) & 0xffu));
    }
}

void append_i32(EncodedGenesis* output, std::int32_t value) {
    append_u32(output, std::bit_cast<std::uint32_t>(value));
}

void append_u64(EncodedGenesis* output, std::uint64_t value) {
    append_u32(output, static_cast<std::uint32_t>(value));
    append_u32(output, static_cast<std::uint32_t>(value >> 32u));
}

void append_hash(EncodedGenesis* output, const Hash256& value) {
    for (const std::uint8_t byte : value) {
        output->push_back(static_cast<std::byte>(byte));
    }
}

void append_address(EncodedGenesis* output, const ContentAddress& address) {
    append_hash(output, address.digest);
    append_u64(output, address.byte_count);
}

[[nodiscard]] std::optional<EncodedGenesis> encode_raw(
    const Genesis& genesis,
    bool include_content_hash) {
    const std::uint64_t parent_count =
        static_cast<std::uint64_t>(genesis.metadata.parent_identities.size());
    const std::uint64_t site_count =
        static_cast<std::uint64_t>(genesis.sites.size());
    const std::optional<std::uint64_t> total =
        encoded_length(parent_count, site_count);
    if (!total.has_value() ||
        *total > std::numeric_limits<std::size_t>::max()) {
        return std::nullopt;
    }
    EncodedGenesis output;
    output.reserve(static_cast<std::size_t>(*total));
    append_u32(&output, genesis.metadata.format);
    append_u32(&output,
               static_cast<std::uint32_t>(genesis.metadata.genesis_class));
    append_u32(&output,
               static_cast<std::uint32_t>(genesis.metadata.artifact_type));
    append_u64(&output, site_count);
    append_u64(&output, parent_count);
    append_u64(&output, *total);
    append_address(&output, genesis.metadata.law_identity);
    append_hash(&output,
                include_content_hash ? genesis.metadata.content_hash : Hash256{});
    append_i32(&output, genesis.metadata.extent_min.x);
    append_i32(&output, genesis.metadata.extent_min.y);
    append_i32(&output, genesis.metadata.extent_min.z);
    append_i32(&output, genesis.metadata.extent_max.x);
    append_i32(&output, genesis.metadata.extent_max.y);
    append_i32(&output, genesis.metadata.extent_max.z);
    append_u64(&output, genesis.metadata.support_components);
    append_u32(&output, genesis.metadata.contact_boundary_mask);
    append_address(&output, genesis.metadata.replay_identity);
    for (const ContentAddress& parent : genesis.metadata.parent_identities) {
        append_address(&output, parent);
    }
    for (const SitePlacement& site : genesis.sites) {
        append_i32(&output, site.coordinate.x);
        append_i32(&output, site.coordinate.y);
        append_i32(&output, site.coordinate.z);
        append_u32(&output, site.word);
    }
    if (output.size() != static_cast<std::size_t>(*total)) {
        return std::nullopt;
    }
    return output;
}

[[nodiscard]] std::optional<Hash256> genesis_content_hash(
    const Genesis& genesis) {
    const std::optional<EncodedGenesis> bytes = encode_raw(genesis, false);
    if (!bytes.has_value()) return std::nullopt;
    ContentHasher hasher;
    constexpr std::string_view domain = "bcc32/genesis/v3";
    hasher.update({reinterpret_cast<const std::byte*>(domain.data()), domain.size()});
    hasher.update(*bytes);
    return hasher.finish();
}

class Reader {
public:
    explicit Reader(std::span<const std::byte> bytes) : bytes_(bytes) {}

    [[nodiscard]] bool read_u32(std::uint32_t* value) {
        if (!has(4u)) return false;
        std::uint32_t output = 0u;
        for (std::uint32_t offset = 0u; offset < 4u; ++offset) {
            output |= std::to_integer<std::uint32_t>(bytes_[cursor_ + offset])
                      << (offset * 8u);
        }
        cursor_ += 4u;
        *value = output;
        return true;
    }

    [[nodiscard]] bool read_i32(std::int32_t* value) {
        std::uint32_t raw = 0u;
        if (!read_u32(&raw)) return false;
        *value = std::bit_cast<std::int32_t>(raw);
        return true;
    }

    [[nodiscard]] bool read_u64(std::uint64_t* value) {
        std::uint32_t low = 0u;
        std::uint32_t high = 0u;
        if (!read_u32(&low) || !read_u32(&high)) return false;
        *value = static_cast<std::uint64_t>(low) |
                 (static_cast<std::uint64_t>(high) << 32u);
        return true;
    }

    [[nodiscard]] bool read_hash(Hash256* value) {
        if (!has(value->size())) return false;
        for (std::size_t index = 0u; index < value->size(); ++index) {
            (*value)[index] =
                std::to_integer<std::uint8_t>(bytes_[cursor_ + index]);
        }
        cursor_ += value->size();
        return true;
    }

    [[nodiscard]] bool read_address(ContentAddress* value) {
        return read_hash(&value->digest) && read_u64(&value->byte_count);
    }

    [[nodiscard]] bool complete() const { return cursor_ == bytes_.size(); }

private:
    [[nodiscard]] bool has(std::size_t count) const {
        return cursor_ <= bytes_.size() && count <= bytes_.size() - cursor_;
    }

    std::span<const std::byte> bytes_;
    std::size_t cursor_ = 0u;
};

[[nodiscard]] std::optional<Genesis> decode_genesis(
    std::span<const std::byte> bytes,
    const ContentAddress& law_identity) {
    if (bytes.size() < kEncodedHeaderBytes || !valid_address(law_identity)) {
        return std::nullopt;
    }
    Reader reader(bytes);
    Genesis genesis{};
    std::uint32_t raw_class = 0u;
    std::uint32_t raw_artifact = 0u;
    std::uint64_t site_count = 0u;
    std::uint64_t parent_count = 0u;
    std::uint64_t total_bytes = 0u;
    if (!reader.read_u32(&genesis.metadata.format) ||
        !reader.read_u32(&raw_class) ||
        !reader.read_u32(&raw_artifact) ||
        !reader.read_u64(&site_count) ||
        !reader.read_u64(&parent_count) ||
        !reader.read_u64(&total_bytes) ||
        !reader.read_address(&genesis.metadata.law_identity) ||
        !reader.read_hash(&genesis.metadata.content_hash) ||
        !reader.read_i32(&genesis.metadata.extent_min.x) ||
        !reader.read_i32(&genesis.metadata.extent_min.y) ||
        !reader.read_i32(&genesis.metadata.extent_min.z) ||
        !reader.read_i32(&genesis.metadata.extent_max.x) ||
        !reader.read_i32(&genesis.metadata.extent_max.y) ||
        !reader.read_i32(&genesis.metadata.extent_max.z) ||
        !reader.read_u64(&genesis.metadata.support_components) ||
        !reader.read_u32(&genesis.metadata.contact_boundary_mask) ||
        !reader.read_address(&genesis.metadata.replay_identity)) {
        return std::nullopt;
    }
    genesis.metadata.genesis_class = static_cast<GenesisClass>(raw_class);
    genesis.metadata.artifact_type = static_cast<ArtifactType>(raw_artifact);
    const std::optional<std::uint64_t> expected =
        encoded_length(parent_count, site_count);
    if (!expected.has_value() || *expected != total_bytes ||
        total_bytes != bytes.size() ||
        parent_count > std::numeric_limits<std::size_t>::max() ||
        site_count > std::numeric_limits<std::size_t>::max()) {
        return std::nullopt;
    }
    try {
        genesis.metadata.parent_identities.reserve(
            static_cast<std::size_t>(parent_count));
        genesis.sites.reserve(static_cast<std::size_t>(site_count));
        for (std::uint64_t index = 0u; index < parent_count; ++index) {
            ContentAddress parent{};
            if (!reader.read_address(&parent)) return std::nullopt;
            genesis.metadata.parent_identities.push_back(parent);
        }
        for (std::uint64_t index = 0u; index < site_count; ++index) {
            SitePlacement site{};
            if (!reader.read_i32(&site.coordinate.x) ||
                !reader.read_i32(&site.coordinate.y) ||
                !reader.read_i32(&site.coordinate.z) ||
                !reader.read_u32(&site.word)) {
                return std::nullopt;
            }
            genesis.sites.push_back(site);
        }
    } catch (const std::bad_alloc&) {
        return std::nullopt;
    } catch (const std::length_error&) {
        return std::nullopt;
    }
    if (!reader.complete() || !validate_genesis(genesis, law_identity).passed()) {
        return std::nullopt;
    }
    return genesis;
}

[[nodiscard]] std::optional<Genesis> finalize(
    Genesis genesis,
    const ContentAddress& law_identity) {
    if (!valid_address(law_identity) || !finite(genesis)) return std::nullopt;
    std::sort(genesis.sites.begin(),
              genesis.sites.end(),
              [](const SitePlacement& left, const SitePlacement& right) {
                  return coordinate_less(left.coordinate, right.coordinate);
              });
    CoordinateIndex index;
    if (!build_coordinate_index(genesis, &index)) return std::nullopt;
    genesis.metadata.format = kGenesisFormat;
    genesis.metadata.law_identity = law_identity;
    const auto [low, high] = exact_extent(genesis);
    genesis.metadata.extent_min = low;
    genesis.metadata.extent_max = high;
    genesis.metadata.support_components = support_components(genesis, index);
    const std::optional<Hash256> identity = genesis_content_hash(genesis);
    if (!identity.has_value()) return std::nullopt;
    genesis.metadata.content_hash = *identity;
    return genesis;
}

}  // namespace

std::optional<Genesis> compile_g0(const ContentAddress& law_identity) {
    Genesis genesis{};
    genesis.metadata.genesis_class = GenesisClass::G0;
    genesis.metadata.artifact_type = ArtifactType::uniform_material;
    genesis.metadata.contact_boundary_mask = 0xffu;
    genesis.sites.push_back({{0, 0, 0}, kQuiescentWord});
    return finalize(std::move(genesis), law_identity);
}

std::optional<EncodedGenesis> seal_genesis(
    Genesis genesis,
    const ContentAddress& law_identity) {
    genesis.metadata.law_identity = law_identity;
    if (!valid_address(law_identity) || !finite(genesis) ||
        !known_class(genesis.metadata.genesis_class) ||
        !known_artifact_type(genesis.metadata.artifact_type) ||
        !provenance_is_addressed(genesis)) {
        return std::nullopt;
    }
    CoordinateIndex index;
    if (!build_coordinate_index(genesis, &index)) {
        return std::nullopt;
    }
    std::optional<Genesis> finalized =
        finalize(std::move(genesis), law_identity);
    if (!finalized.has_value() ||
        !validate_genesis(*finalized, law_identity).passed()) {
        return std::nullopt;
    }
    return encode_genesis(*finalized, law_identity);
}

std::optional<G1Capsule> open_g1_capsule(
    std::span<const std::byte> bytes,
    const ContentAddress& law_identity) {
    std::optional<Genesis> genesis = decode_genesis(bytes, law_identity);
    if (!genesis.has_value() ||
        genesis->metadata.genesis_class != GenesisClass::G1 ||
        genesis->metadata.artifact_type != ArtifactType::authored_material) {
        return std::nullopt;
    }
    return G1Capsule{std::move(*genesis)};
}

std::optional<G2Capsule> open_g2_capsule(
    std::span<const std::byte> bytes,
    const ContentAddress& law_identity) {
    std::optional<Genesis> genesis = decode_genesis(bytes, law_identity);
    if (!genesis.has_value() ||
        genesis->metadata.genesis_class != GenesisClass::G2 ||
        genesis->metadata.artifact_type != ArtifactType::propagule_capsule ||
        genesis->metadata.parent_identities.empty() ||
        !valid_address(genesis->metadata.replay_identity)) {
        return std::nullopt;
    }
    return G2Capsule{std::move(*genesis)};
}

Genesis compile_g1(const G1Capsule& capsule) {
    return capsule.genesis_;
}

Genesis compile_g2(const G2Capsule& capsule) {
    return capsule.genesis_;
}

GenesisAudit validate_genesis(const Genesis& genesis,
                              const ContentAddress& law_identity) {
    GenesisAudit audit{};
    if (!finite(genesis)) return audit;
    audit.bits |= kAuditFinite;
    if (known_class(genesis.metadata.genesis_class)) {
        audit.bits |= kAuditKnownClass;
    }
    if (known_artifact_type(genesis.metadata.artifact_type)) {
        audit.bits |= kAuditKnownArtifactType;
    }
    if (valid_address(law_identity) &&
        genesis.metadata.format == kGenesisFormat &&
        genesis.metadata.law_identity == law_identity) {
        audit.bits |= kAuditLawBinding;
    }
    CoordinateIndex index;
    const bool injective = build_coordinate_index(genesis, &index);
    if (injective) audit.bits |= kAuditInjectiveCoordinates;
    const auto [low, high] = exact_extent(genesis);
    if (genesis.metadata.extent_min == low &&
        genesis.metadata.extent_max == high) {
        audit.bits |= kAuditExactExtent;
    }
    if (injective && genesis.metadata.support_components ==
                         support_components(genesis, index)) {
        audit.bits |= kAuditExactSupport;
    }
    if (provenance_is_addressed(genesis)) {
        audit.bits |= kAuditAddressProvenance;
    }
    if (encoded_length(
            static_cast<std::uint64_t>(
                genesis.metadata.parent_identities.size()),
            static_cast<std::uint64_t>(genesis.sites.size()))
            .has_value()) {
        audit.bits |= kAuditCanonicalLength;
    }
    if (sites_are_canonical(genesis)) {
        audit.bits |= kAuditCanonicalOrder;
    }
    const std::optional<Hash256> identity = genesis_content_hash(genesis);
    if (identity.has_value() && genesis.metadata.content_hash == *identity) {
        audit.bits |= kAuditContentHash;
    }
    return audit;
}

std::optional<EncodedGenesis> encode_genesis(
    const Genesis& genesis,
    const ContentAddress& law_identity) {
    if (!validate_genesis(genesis, law_identity).passed()) {
        return std::nullopt;
    }
    return encode_raw(genesis, true);
}

ContentAddress genesis_artifact_identity(
    std::span<const std::byte> encoded_genesis) {
    ContentHasher hasher;
    constexpr std::string_view domain = "bcc32/genesis-artifact/v3";
    hasher.update(
        {reinterpret_cast<const std::byte*>(domain.data()), domain.size()});
    hasher.update(encoded_genesis);
    return {hasher.finish(), encoded_genesis.size()};
}

}  // namespace substrate::bcc32
