#include "bcc32_checkpoint_hamming_divergence.hpp"

#include <bit>
#include <set>

namespace substrate::bcc32 {

std::uint64_t checkpoint_hamming_distance(const WorldStore& a, const WorldStore& b) {
    std::set<ChunkCoord, CoordinateLess> touched;
    for (const auto& [coordinate, words] : a.chunks()) touched.insert(coordinate);
    for (const auto& [coordinate, words] : b.chunks()) touched.insert(coordinate);

    std::uint64_t distance = 0;
    for (const ChunkCoord& coordinate : touched) {
        const std::vector<SiteWord> words_a = a.read_chunk(coordinate);
        const std::vector<SiteWord> words_b = b.read_chunk(coordinate);
        for (std::size_t i = 0; i < words_a.size(); ++i)
            distance += static_cast<std::uint64_t>(std::popcount(words_a[i] ^ words_b[i]));
    }
    return distance;
}

}  // namespace substrate::bcc32
