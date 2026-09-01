#include "bcc32_world_store.hpp"

#include <algorithm>
#include <utility>

namespace substrate::bcc32 {
namespace {

[[nodiscard]] bool valid_local(const std::array<std::uint32_t, 3>& local) {
    return local[0] < kChunkEdge && local[1] < kChunkEdge && local[2] < kChunkEdge;
}

}  // namespace

ChunkCoord chunk_for_site(const SiteCoord& site) {
    return {floor_divide(site.x, kChunkEdge),
            floor_divide(site.y, kChunkEdge),
            floor_divide(site.z, kChunkEdge)};
}

std::array<std::uint32_t, 3> local_site_in_chunk(const SiteCoord& site) {
    return {floor_modulo(site.x, kChunkEdge),
            floor_modulo(site.y, kChunkEdge),
            floor_modulo(site.z, kChunkEdge)};
}

std::uint64_t local_site_index(const std::array<std::uint32_t, 3>& local) {
    if (!valid_local(local)) return kChunkSites;
    return (static_cast<std::uint64_t>(local[0]) * kChunkEdge + local[1]) *
               kChunkEdge +
           local[2];
}

std::string canonical_chunk_name(const ChunkCoord& coordinate) {
    return canonical_coordinate_component(coordinate.x) + "_" +
           canonical_coordinate_component(coordinate.y) + "_" +
           canonical_coordinate_component(coordinate.z);
}

bool is_all_quiescent(std::span<const SiteWord> words) {
    return std::all_of(words.begin(), words.end(), [](SiteWord word) {
        return word == kQuiescentWord;
    });
}

std::vector<SiteWord> WorldStore::read_chunk(const ChunkCoord& coordinate) const {
    const auto found = chunks_.find(coordinate);
    if (found != chunks_.end()) return found->second;
    return std::vector<SiteWord>(kChunkSites, kQuiescentWord);
}

SiteWord WorldStore::read_site(const SiteCoord& coordinate) const {
    const auto found = chunks_.find(chunk_for_site(coordinate));
    if (found == chunks_.end()) return kQuiescentWord;
    return found->second[local_site_index(local_site_in_chunk(coordinate))];
}

WorldSupport WorldStore::support() const {
    WorldSupport result{};
    if (chunks_.empty()) return result;

    result.has_chunks = true;
    result.minimum = chunks_.begin()->first;
    result.maximum = chunks_.begin()->first;
    result.materialized_chunks = static_cast<std::uint64_t>(chunks_.size());
    result.direct_word_bytes = result.materialized_chunks * kChunkBytes;
    for (const auto& [coordinate, words] : chunks_) {
        result.minimum.x = std::min(result.minimum.x, coordinate.x);
        result.minimum.y = std::min(result.minimum.y, coordinate.y);
        result.minimum.z = std::min(result.minimum.z, coordinate.z);
        result.maximum.x = std::max(result.maximum.x, coordinate.x);
        result.maximum.y = std::max(result.maximum.y, coordinate.y);
        result.maximum.z = std::max(result.maximum.z, coordinate.z);
        result.non_quiescent_sites += static_cast<std::uint64_t>(
            std::count_if(words.begin(), words.end(), [](SiteWord word) {
                return word != kQuiescentWord;
            }));
    }
    return result;
}

bool WorldStore::equals(const WorldStore& other) const {
    return chunks_ == other.chunks_;
}

bool WorldStore::replace_chunk(const ChunkCoord& coordinate,
                               std::span<const SiteWord> words,
                               std::string* error) {
    if (words.size() != kChunkSites) {
        if (error != nullptr) *error = "BCC-32 chunk has an invalid direct word count";
        return false;
    }
    if (is_all_quiescent(words)) {
        chunks_.erase(coordinate);
        return true;
    }
    chunks_[coordinate] = std::vector<SiteWord>(words.begin(), words.end());
    return true;
}

bool WorldStore::write_site(const SiteCoord& coordinate,
                            SiteWord word,
                            std::string* error) {
    const ChunkCoord chunk = chunk_for_site(coordinate);
    const std::uint64_t index = local_site_index(local_site_in_chunk(coordinate));
    if (index >= kChunkSites) {
        if (error != nullptr) *error = "BCC-32 local site index is invalid";
        return false;
    }
    auto found = chunks_.find(chunk);
    if (found == chunks_.end()) {
        if (word == kQuiescentWord) return true;
        found = chunks_
                    .emplace(chunk, std::vector<SiteWord>(kChunkSites, kQuiescentWord))
                    .first;
    }
    found->second[index] = word;
    if (is_all_quiescent(found->second)) chunks_.erase(found);
    return true;
}

static_assert(sizeof(SiteWord) == sizeof(std::uint32_t));
static_assert(kProductionBits == 80'000'000'000ull);

}  // namespace substrate::bcc32
