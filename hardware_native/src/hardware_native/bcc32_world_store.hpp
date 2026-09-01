#pragma once

#include "bcc32_coordinate.hpp"
#include "bcc32_types.cuh"

#include <array>
#include <cstdint>
#include <map>
#include <span>
#include <string>
#include <vector>

namespace substrate::bcc32 {

struct WorldSupport {
    bool has_chunks = false;
    ChunkCoord minimum{};
    ChunkCoord maximum{};
    std::uint64_t materialized_chunks = 0;
    std::uint64_t non_quiescent_sites = 0;
    std::uint64_t direct_word_bytes = 0;

    friend bool operator==(const WorldSupport&, const WorldSupport&) = default;
};

[[nodiscard]] ChunkCoord chunk_for_site(const SiteCoord& site);
[[nodiscard]] std::array<std::uint32_t, 3> local_site_in_chunk(
    const SiteCoord& site);
[[nodiscard]] std::uint64_t local_site_index(
    const std::array<std::uint32_t, 3>& local);
[[nodiscard]] std::string canonical_chunk_name(const ChunkCoord& coordinate);
[[nodiscard]] bool is_all_quiescent(std::span<const SiteWord> words);

// This store intentionally has one persisted form: an ordered direct array of
// SiteWord values.  The absence of a record means exactly all kQuiescentWord;
// no other codec, palette, or summary is part of persistence.
class WorldStore {
  public:
    using ChunkMap = std::map<ChunkCoord, std::vector<SiteWord>, CoordinateLess>;

    [[nodiscard]] const ChunkMap& chunks() const {
        return chunks_;
    }
    [[nodiscard]] std::vector<SiteWord> read_chunk(const ChunkCoord& coordinate) const;
    [[nodiscard]] SiteWord read_site(const SiteCoord& coordinate) const;
    [[nodiscard]] WorldSupport support() const;
    [[nodiscard]] bool equals(const WorldStore& other) const;

    bool replace_chunk(const ChunkCoord& coordinate,
                       std::span<const SiteWord> words,
                       std::string* error);
    bool write_site(const SiteCoord& coordinate, SiteWord word, std::string* error);

  private:
    ChunkMap chunks_;
};

}  // namespace substrate::bcc32
