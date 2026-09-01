#include "bcc32_transition.cuh"

#include "bcc32_conditioned_learning_matter.hpp"
#include "bcc32_geometry.cuh"
#include "bcc32_law_identity.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <bit>
#include <deque>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

#include <boost/multiprecision/cpp_int.hpp>

namespace substrate::bcc32 {

namespace detail {

// Deliberately absent from the public checkpoint header. The evaluator-facing
// persistence API cannot name or call the trusted executor publication seam.
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
    PublicationFailurePoint failure);

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
    PublicationFailurePoint failure);

}  // namespace detail
namespace {

using CoordinateSet = std::set<ChunkCoord, CoordinateLess>;
using ReferenceMap =
    std::map<ChunkCoord, ChunkObjectReference, CoordinateLess>;

struct Page {
    std::vector<ChunkCoord> core;
    std::vector<ChunkCoord> loaded;
};

struct LocalCoordinate {
    std::int64_t x = 0;
    std::int64_t y = 0;
    std::int64_t z = 0;

    friend bool operator==(const LocalCoordinate&, const LocalCoordinate&) = default;
};

bool fail(std::string* error, const std::string& message) {
    if (error != nullptr) *error = message;
    return false;
}

void check_cuda(cudaError_t status, std::string_view operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " +
                                 cudaGetErrorString(status));
    }
}

ChunkCoord offset_chunk(const ChunkCoord& coordinate,
                        std::int32_t x,
                        std::int32_t y,
                        std::int32_t z) {
    return {coordinate.x + x, coordinate.y + y, coordinate.z + z};
}

ChunkCoord direction_neighbor(const ChunkCoord& coordinate,
                              std::uint32_t direction) {
    const Int3 offset = direction_offset(static_cast<Direction>(direction));
    return offset_chunk(coordinate, offset.x, offset.y, offset.z);
}

CoordinateSet resident_window(const WorldStore& input, std::uint32_t steps);

CoordinateSet factor_candidates(const WorldCommit& input, LawFactor factor) {
    CoordinateSet result;
    for (const ChunkObjectReference& chunk : input.chunks) {
        result.insert(chunk.coordinate);
        if (factor == LawFactor::site) continue;
        for (std::int32_t dx = -1; dx <= 1; ++dx) {
            for (std::int32_t dy = -1; dy <= 1; ++dy) {
                for (std::int32_t dz = -1; dz <= 1; ++dz) {
                    result.insert(offset_chunk(chunk.coordinate, dx, dy, dz));
                }
            }
        }
    }
    return result;
}

CoordinateSet factor_candidates(const WorldStore& input, LawFactor factor) {
    if (factor != LawFactor::site) return resident_window(input, 1u);
    CoordinateSet result;
    for (const auto& [coordinate, words] : input.chunks()) {
        (void)words;
        result.insert(coordinate);
    }
    return result;
}

CoordinateSet resident_window(const WorldStore& input, std::uint32_t steps) {
    const std::uint64_t reach_u64 =
        static_cast<std::uint64_t>(steps) * kSpatialMacroClosureRadius;
    if (reach_u64 > static_cast<std::uint64_t>(
                        std::numeric_limits<std::int32_t>::max() -
                        kChunkEdge)) {
        throw std::runtime_error("BCC32 resident reach overflow");
    }
    const auto floor_chunk_offset = [](std::int64_t value) {
        std::int64_t quotient =
            value / static_cast<std::int64_t>(kChunkEdge);
        const std::int64_t remainder =
            value % static_cast<std::int64_t>(kChunkEdge);
        if (remainder < 0) --quotient;
        return static_cast<std::int32_t>(quotient);
    };
    CoordinateSet window;
    for (const auto& [coordinate, words] : input.chunks()) {
        std::array<std::uint32_t, 3u> minimum{
            kChunkEdge, kChunkEdge, kChunkEdge};
        std::array<std::uint32_t, 3u> maximum{};
        bool occupied = false;
        const std::uint64_t plane = kChunkEdge * kChunkEdge;
        for (std::uint64_t index = 0u; index < words.size(); ++index) {
            if (words[index] == kQ) continue;
            occupied = true;
            const std::uint32_t local_x =
                static_cast<std::uint32_t>(index / plane);
            const std::uint64_t remainder = index % plane;
            const std::uint32_t local_y =
                static_cast<std::uint32_t>(remainder / kChunkEdge);
            const std::uint32_t local_z =
                static_cast<std::uint32_t>(remainder % kChunkEdge);
            minimum[0] = std::min(minimum[0], local_x);
            minimum[1] = std::min(minimum[1], local_y);
            minimum[2] = std::min(minimum[2], local_z);
            maximum[0] = std::max(maximum[0], local_x);
            maximum[1] = std::max(maximum[1], local_y);
            maximum[2] = std::max(maximum[2], local_z);
        }
        if (!occupied) continue;
        const std::int64_t reach = static_cast<std::int64_t>(reach_u64);
        const std::int32_t min_dx =
            floor_chunk_offset(
                static_cast<std::int64_t>(minimum[0]) - reach);
        const std::int32_t max_dx =
            floor_chunk_offset(
                static_cast<std::int64_t>(maximum[0]) + reach);
        const std::int32_t min_dy =
            floor_chunk_offset(
                static_cast<std::int64_t>(minimum[1]) - reach);
        const std::int32_t max_dy =
            floor_chunk_offset(
                static_cast<std::int64_t>(maximum[1]) + reach);
        const std::int32_t min_dz =
            floor_chunk_offset(
                static_cast<std::int64_t>(minimum[2]) - reach);
        const std::int32_t max_dz =
            floor_chunk_offset(
                static_cast<std::int64_t>(maximum[2]) + reach);
        for (std::int32_t dx = min_dx; dx <= max_dx; ++dx) {
            for (std::int32_t dy = min_dy; dy <= max_dy; ++dy) {
                for (std::int32_t dz = min_dz; dz <= max_dz; ++dz) {
                    window.insert(offset_chunk(coordinate, dx, dy, dz));
                }
            }
        }
    }
    return window;
}

std::int64_t chunk_delta(std::span<const SiteWord> words) {
    std::int64_t result = 0;
    for (const SiteWord word : words) {
        result += static_cast<std::int32_t>(std::popcount(word)) -
                  static_cast<std::int32_t>(std::popcount(kQuiescentWord));
    }
    return result;
}

CoordinateSet page_closure(std::span<const ChunkCoord> core, LawFactor factor) {
    CoordinateSet result(core.begin(), core.end());
    if (factor == LawFactor::site) return result;
    for (const ChunkCoord& coordinate : core) {
        for (std::int32_t dx = -1; dx <= 1; ++dx) {
            for (std::int32_t dy = -1; dy <= 1; ++dy) {
                for (std::int32_t dz = -1; dz <= 1; ++dz) {
                    result.insert(offset_chunk(coordinate, dx, dy, dz));
                }
            }
        }
    }
    return result;
}

std::vector<Page> make_pages(const CoordinateSet& candidate_set,
                             LawFactor factor,
                             std::uint32_t aperture_chunks,
                             PageSchedule schedule) {
    std::vector<ChunkCoord> candidates(candidate_set.begin(), candidate_set.end());
    if (schedule.reverse_core_order) std::reverse(candidates.begin(), candidates.end());
    const std::uint32_t core_limit = schedule.maximum_core_chunks == 0u
                                         ? aperture_chunks
                                         : std::min(schedule.maximum_core_chunks,
                                                    aperture_chunks);
    std::vector<Page> pages;
    std::size_t next = 0u;
    while (next < candidates.size()) {
        Page page;
        CoordinateSet loaded;
        while (next < candidates.size() && page.core.size() < core_limit) {
            std::vector<ChunkCoord> trial = page.core;
            trial.push_back(candidates[next]);
            CoordinateSet trial_loaded = page_closure(trial, factor);
            if (trial_loaded.size() + trial.size() > aperture_chunks) {
                if (page.core.empty()) {
                    throw std::runtime_error(
                        "BCC32 aperture cannot hold one core chunk and its exact halo");
                }
                break;
            }
            page.core = std::move(trial);
            loaded = std::move(trial_loaded);
            ++next;
        }
        page.loaded = page.core;
        for (const ChunkCoord& coordinate : loaded) {
            if (std::find(page.core.begin(), page.core.end(), coordinate) ==
                page.core.end()) {
                page.loaded.push_back(coordinate);
            }
        }
        pages.push_back(std::move(page));
    }
    return pages;
}

std::vector<DeviceChunkSlot> make_device_topology(
    const Page& page,
    std::uint32_t aperture_chunks) {
    const std::size_t real_count = page.loaded.size();
    std::map<ChunkCoord, std::uint32_t, CoordinateLess> index;
    for (std::uint32_t slot = 0u; slot < real_count; ++slot) {
        if (!index.emplace(page.loaded[slot], slot).second) {
            throw std::logic_error("BCC32 page contains a duplicate material coordinate");
        }
    }

    std::vector<LocalCoordinate> local(real_count);
    std::vector<bool> assigned(real_count, false);
    std::int64_t component_cursor = 0;
    for (std::uint32_t root = 0u; root < real_count; ++root) {
        if (assigned[root]) continue;
        std::deque<std::uint32_t> pending;
        std::vector<std::uint32_t> component;
        assigned[root] = true;
        local[root] = {};
        pending.push_back(root);
        while (!pending.empty()) {
            const std::uint32_t source = pending.front();
            pending.pop_front();
            component.push_back(source);
            for (std::uint32_t direction = 0u; direction < 8u; ++direction) {
                const auto found = index.find(
                    direction_neighbor(page.loaded[source], direction));
                if (found == index.end()) continue;
                const Int3 offset =
                    direction_offset(static_cast<Direction>(direction));
                const LocalCoordinate expected{
                    local[source].x + offset.x,
                    local[source].y + offset.y,
                    local[source].z + offset.z,
                };
                if (!assigned[found->second]) {
                    assigned[found->second] = true;
                    local[found->second] = expected;
                    pending.push_back(found->second);
                } else if (!(local[found->second] == expected)) {
                    throw std::logic_error("BCC32 page topology is geometrically inconsistent");
                }
            }
        }
        LocalCoordinate minimum = local[component.front()];
        LocalCoordinate maximum = minimum;
        for (std::uint32_t slot : component) {
            minimum.x = std::min(minimum.x, local[slot].x);
            minimum.y = std::min(minimum.y, local[slot].y);
            minimum.z = std::min(minimum.z, local[slot].z);
            maximum.x = std::max(maximum.x, local[slot].x);
            maximum.y = std::max(maximum.y, local[slot].y);
            maximum.z = std::max(maximum.z, local[slot].z);
        }
        for (std::uint32_t slot : component) {
            local[slot].x += component_cursor - minimum.x;
            local[slot].y -= minimum.y;
            local[slot].z -= minimum.z;
        }
        component_cursor += maximum.x - minimum.x + 4;
    }

    std::vector<DeviceChunkSlot> slots(aperture_chunks);
    for (std::uint32_t slot = 0u; slot < real_count; ++slot) {
        slots[slot].chunk_x = local[slot].x;
        slots[slot].chunk_y = local[slot].y;
        slots[slot].chunk_z = local[slot].z;
        for (std::uint32_t direction = 0u; direction < 8u; ++direction) {
            const auto found = index.find(
                direction_neighbor(page.loaded[slot], direction));
            slots[slot].bcc_neighbors[direction] =
                found == index.end()
                    ? DeviceChunkSlot::kMissing
                    : static_cast<std::int32_t>(found->second);
        }
    }
    for (std::uint32_t slot = static_cast<std::uint32_t>(real_count);
         slot < aperture_chunks;
         ++slot) {
        slots[slot].chunk_x = component_cursor +
                              3 * static_cast<std::int64_t>(slot - real_count + 1u);
        slots[slot].chunk_y = 0;
        slots[slot].chunk_z = 0;
    }
    return slots;
}

const ChunkObjectReference* find_reference(const ReferenceMap& references,
                                           const ChunkCoord& coordinate) {
    const auto found = references.find(coordinate);
    return found == references.end() ? nullptr : &found->second;
}

ContentAddress replay_commitment(const WorldCommit& input,
                                 bool inverse,
                                 std::uint64_t completed_supersteps) {
    const std::string bytes =
        "bcc32/factor-paged-transition/v1|" + hash_hex(input.identity.digest) + "|" +
        std::to_string(input.identity.byte_count) + "|" +
        hash_hex(input.metadata.provenance.law.digest) + "|" +
        (inverse ? "inverse|" : "forward|") +
        std::to_string(completed_supersteps);
    return content_address({reinterpret_cast<const std::byte*>(bytes.data()), bytes.size()});
}

}  // namespace

PagedWorldExecutor PagedWorldExecutor::production() {
    return PagedWorldExecutor(CudaBcc32Executor::production());
}

PagedWorldExecutor PagedWorldExecutor::windowed(
    std::uint32_t aperture_chunks) {
    return PagedWorldExecutor(CudaBcc32Executor::testing(aperture_chunks));
}

PagedWorldExecutor PagedWorldExecutor::testing(std::uint32_t aperture_chunks) {
    return windowed(aperture_chunks);
}

PagedWorldExecutor::PagedWorldExecutor(CudaBcc32Executor executor)
    : executor_(std::move(executor)) {
    check_cuda(cudaMalloc(&device_slots_,
                          static_cast<std::size_t>(executor_.chunk_count()) *
                              sizeof(DeviceChunkSlot)),
               "allocate BCC32 paged topology");
}

PagedWorldExecutor::PagedWorldExecutor(PagedWorldExecutor&& other) noexcept
    : executor_(std::move(other.executor_)),
      device_slots_(std::exchange(other.device_slots_, nullptr)) {}

PagedWorldExecutor& PagedWorldExecutor::operator=(PagedWorldExecutor&& other) noexcept {
    if (this == &other) return *this;
    release();
    executor_ = std::move(other.executor_);
    device_slots_ = std::exchange(other.device_slots_, nullptr);
    return *this;
}

PagedWorldExecutor::~PagedWorldExecutor() {
    release();
}

void PagedWorldExecutor::release() noexcept {
    if (device_slots_ != nullptr) {
        (void)cudaFree(device_slots_);
        device_slots_ = nullptr;
    }
}

void PagedWorldExecutor::verify_quiescent_aperture() {
    executor_.initialize_q();
    if (!executor_.all_words_equal(kQuiescentWord)) {
        throw std::runtime_error("BCC32 direct aperture failed its all-Q equality audit");
    }
}

std::uint32_t PagedWorldExecutor::verify_nontrivial_forward_inverse_aperture() {
    if (executor_.chunk_count() != kProductionChunkSlots) {
        throw std::logic_error(
            "BCC32 full-aperture proof requires the production chunk count");
    }

    Page aperture;
    aperture.loaded.reserve(kProductionChunkSlots);
    for (std::int32_t x = 0; x < 10; ++x) {
        for (std::int32_t y = 0; y < 10; ++y) {
            for (std::int32_t z = 0; z < 25; ++z) {
                aperture.loaded.push_back({x, y, z});
            }
        }
    }
    const std::vector<DeviceChunkSlot> topology =
        make_device_topology(aperture, kProductionChunkSlots);
    check_cuda(cudaMemcpy(device_slots_,
                          topology.data(),
                          topology.size() * sizeof(DeviceChunkSlot),
                          cudaMemcpyHostToDevice),
               "upload BCC32 production topology");
    const DeviceChunkMap device_map{device_slots_, kProductionChunkSlots};

    executor_.initialize_q();
    constexpr std::uint32_t kProbeWords = 64u;
    constexpr std::uint32_t kProofSupersteps = 2u;
    std::array<std::uint64_t, kProbeWords> offsets{};
    std::array<SiteWord, kProbeWords> expected{};
    for (std::uint32_t index = 0u; index < kProbeWords; ++index) {
        const std::uint32_t direction = index / 8u;
        const std::uint32_t variant = index % 8u;
        std::uint64_t chunk_x = 5u;
        std::uint64_t chunk_y = 5u;
        std::uint64_t chunk_z = 12u;
        std::uint64_t local_x = 40u + variant;
        std::uint64_t local_y = 52u + variant;
        std::uint64_t local_z = 64u + variant;
        switch (direction) {
            case 0u: chunk_x = 4u; local_x = 99u; break;
            case 1u: chunk_y = 4u; local_y = 99u; break;
            case 2u: chunk_z = 11u; local_z = 99u; break;
            case 3u:
                chunk_x = 4u + (variant & 1u);
                chunk_y = 4u + ((variant >> 1u) & 1u);
                chunk_z = 11u + ((variant >> 2u) & 1u);
                local_x = local_y = local_z = 0u;
                break;
            case 4u: local_x = 0u; break;
            case 5u: local_y = 0u; break;
            case 6u: local_z = 0u; break;
            case 7u:
                chunk_x = 4u + (variant & 1u);
                chunk_y = 4u + ((variant >> 1u) & 1u);
                chunk_z = 11u + ((variant >> 2u) & 1u);
                local_x = local_y = local_z = 99u;
                break;
        }
        const std::uint64_t chunk = (chunk_x * 10u + chunk_y) * 25u + chunk_z;
        const std::uint64_t local =
            (local_x * kChunkEdge + local_y) * kChunkEdge + local_z;
        offsets[index] = chunk * kChunkSites + local;
        expected[index] = kQuiescentWord ^ carrier_bit(direction) ^
                          face_bit((direction + variant * 3u) % 8u) ^
                          static_cast<SiteWord>(1u << (20u + index % 12u));
        executor_.write_word(offsets[index], expected[index]);
    }

    executor_.apply_superstep(device_map);
    bool changed = false;
    for (std::uint32_t index = 0u; index < kProbeWords; ++index) {
        changed = changed || executor_.read_word(offsets[index]) != expected[index];
    }
    if (!changed) {
        throw std::runtime_error(
            "BCC32 production forward law performed no observable material work");
    }
    for (std::uint32_t step = 1u; step < kProofSupersteps; ++step) {
        executor_.apply_superstep(device_map);
    }

    for (std::uint32_t step = 0u; step < kProofSupersteps; ++step) {
        executor_.apply_superstep_inverse(device_map);
    }
    for (std::uint32_t index = 0u; index < kProbeWords; ++index) {
        if (executor_.read_word(offsets[index]) != expected[index]) {
            throw std::runtime_error(
                "BCC32 production inverse failed to restore a seeded material word");
        }
        executor_.write_word(offsets[index], kQuiescentWord);
    }
    if (!executor_.all_words_equal(kQuiescentWord)) {
        throw std::runtime_error(
            "BCC32 production inverse failed exact full-aperture restoration");
    }
    return kProbeWords;
}

bool PagedWorldExecutor::advance(const std::filesystem::path& repository,
                                 ArtifactKind expected_kind,
                                 bool inverse,
                                 PageSchedule schedule,
                                 TransitionReceipt* receipt,
                                 std::string* error,
                                 PublicationFailurePoint failure) {
    WorldCommit input{};
    if (!load_world_commit(repository, expected_kind, &input, error)) return false;
    return advance_loaded(repository,
                          std::move(input),
                          expected_kind,
                          inverse,
                          schedule,
                          true,
                          receipt,
                          error,
                          failure);
}

bool PagedWorldExecutor::advance_object(
    const std::filesystem::path& repository,
    const ContentAddress& input_identity,
    ArtifactKind expected_kind,
    bool inverse,
    PageSchedule schedule,
    TransitionReceipt* receipt,
    std::string* error,
    PublicationFailurePoint failure) {
    WorldCommit input{};
    if (!load_world_commit_object(
            repository, expected_kind, input_identity, &input, error)) {
        return false;
    }
    return advance_loaded(repository,
                          std::move(input),
                          expected_kind,
                          inverse,
                          schedule,
                          false,
                          receipt,
                          error,
                          failure);
}

bool PagedWorldExecutor::advance_store(WorldStore* world, bool inverse,
                                       PageSchedule schedule,
                                       std::uint64_t* page_count,
                                       std::string* error) {
    if (world == nullptr) return fail(error, "null BCC32 sparse world");
    try {
        WorldStore current = *world;
        std::uint64_t pages_advanced = 0u;
        const std::vector<SiteWord> quiescent(kChunkSites, kQuiescentWord);

        // initialize_q() resets the *entire* aperture and always synchronizes
        // internally (it is not stream-deferrable), so it must not sit inside
        // a per-page loop -- that turned every page into a full-aperture
        // device stall. Calling it once here and explicitly uploading the
        // quiescent word for every slot lacking a material entry (below)
        // mirrors advance_loaded()'s already-correct single-clear pattern.
        executor_.initialize_q();

        for (std::uint32_t factor_index = 0u;
             factor_index < kForwardFactorCount; ++factor_index) {
            const std::uint32_t ordered_index =
                inverse ? kForwardFactorCount - factor_index - 1u
                        : factor_index;
            const LawFactor factor = forward_factor(ordered_index);
            const CoordinateSet candidates =
                factor_candidates(current, factor);
            const std::vector<Page> pages =
                make_pages(candidates, factor, executor_.chunk_count(),
                           schedule);
            WorldStore output;
            boost::multiprecision::cpp_int input_delta = 0;
            boost::multiprecision::cpp_int output_delta = 0;

            for (const Page& page : pages) {
                const std::vector<DeviceChunkSlot> slots =
                    make_device_topology(page, executor_.chunk_count());
                check_cuda(cudaMemcpy(device_slots_, slots.data(),
                                      slots.size() * sizeof(DeviceChunkSlot),
                                      cudaMemcpyHostToDevice),
                           "upload BCC32 in-memory paged topology");

                const auto& material = current.chunks();
                try {
                    for (std::uint32_t slot = 0u;
                         slot < page.loaded.size(); ++slot) {
                        const auto found = material.find(page.loaded[slot]);
                        if (found == material.end()) {
                            executor_.upload_words(
                                static_cast<std::uint64_t>(slot) * kChunkSites,
                                quiescent);
                            continue;
                        }
                        if (slot < page.core.size())
                            input_delta += chunk_delta(found->second);
                        executor_.upload_words(
                            static_cast<std::uint64_t>(slot) * kChunkSites,
                            found->second);
                    }
                    executor_.apply_factor_window(
                        {device_slots_, executor_.chunk_count()}, factor,
                        inverse,
                        static_cast<std::uint32_t>(page.core.size()),
                        static_cast<std::uint32_t>(page.loaded.size()));
                } catch (...) {
                    (void)cudaStreamSynchronize(nullptr);
                    throw;
                }

                std::vector<std::vector<SiteWord>> page_outputs;
                page_outputs.reserve(page.core.size());
                try {
                    for (std::uint32_t slot = 0u;
                         slot < page.core.size(); ++slot) {
                        page_outputs.emplace_back(kChunkSites);
                        executor_.download_words(
                            (static_cast<std::uint64_t>(page.loaded.size()) +
                             slot) *
                                kChunkSites,
                            page_outputs.back());
                    }
                    check_cuda(cudaStreamSynchronize(nullptr),
                               "synchronize BCC32 in-memory page downloads");
                } catch (...) {
                    (void)cudaStreamSynchronize(nullptr);
                    throw;
                }
                for (std::uint32_t slot = 0u; slot < page.core.size();
                     ++slot) {
                    output_delta += chunk_delta(page_outputs[slot]);
                    if (!output.replace_chunk(page.core[slot],
                                              page_outputs[slot], error))
                        return false;
                }
                ++pages_advanced;
            }
            if (input_delta != output_delta)
                return fail(error,
                            "BCC32 in-memory pager violated DeltaN_Q");
            current = std::move(output);
        }
        *world = std::move(current);
        if (page_count != nullptr) *page_count = pages_advanced;
        return true;
    } catch (const std::exception& exception) {
        return fail(error,
                    std::string("BCC32 in-memory paged transition failed: ") +
                        exception.what());
    }
}

bool PagedWorldExecutor::advance_store_resident(
    WorldStore* world, std::uint32_t steps, bool inverse,
    std::string* error, PageSchedule schedule) {
    if (world == nullptr) return fail(error, "null BCC32 sparse world");
    if (steps == 0u) return true;
    try {
        WorldStore current = *world;
        const CoordinateSet window = resident_window(current, steps);
        if (window.size() > executor_.chunk_count())
            return fail(error,
                        "BCC32 resident window exceeds CUDA aperture");
        if (window.empty()) return true;

        Page page;
        page.core.assign(window.begin(), window.end());
        if (schedule.reverse_core_order)
            std::reverse(page.core.begin(), page.core.end());
        page.loaded = page.core;
        const std::vector<DeviceChunkSlot> slots =
            make_device_topology(page, executor_.chunk_count());
        check_cuda(cudaMemcpy(device_slots_, slots.data(),
                              slots.size() * sizeof(DeviceChunkSlot),
                              cudaMemcpyHostToDevice),
                   "upload BCC32 resident topology");
        executor_.initialize_q();

        boost::multiprecision::cpp_int input_delta = 0;
        for (std::uint32_t slot = 0u; slot < page.loaded.size(); ++slot) {
            const std::vector<SiteWord> words =
                current.read_chunk(page.loaded[slot]);
            input_delta += chunk_delta(words);
            executor_.upload_words(
                static_cast<std::uint64_t>(slot) * kChunkSites, words);
        }
        check_cuda(cudaStreamSynchronize(nullptr),
                   "synchronize BCC32 resident uploads");

        const DeviceChunkMap map{
            device_slots_,
            static_cast<std::uint32_t>(page.loaded.size())};
        for (std::uint32_t step = 0u; step < steps; ++step) {
            if (inverse)
                executor_.apply_superstep_inverse(map);
            else
                executor_.apply_superstep(map);
        }

        WorldStore output;
        boost::multiprecision::cpp_int output_delta = 0;
        // Issue every slot's async download first and synchronize once for
        // the whole window, instead of stalling the device once per chunk.
        // Each slot owns a fresh host vector, so nothing here aliases across
        // iterations and deferring the sync changes no read/write ordering.
        std::vector<std::vector<SiteWord>> downloaded(page.loaded.size());
        for (std::uint32_t slot = 0u; slot < page.loaded.size(); ++slot) {
            downloaded[slot].resize(kChunkSites);
            executor_.download_words(
                static_cast<std::uint64_t>(slot) * kChunkSites,
                downloaded[slot]);
        }
        check_cuda(cudaStreamSynchronize(nullptr),
                   "synchronize BCC32 resident downloads");
        for (std::uint32_t slot = 0u; slot < page.loaded.size(); ++slot) {
            output_delta += chunk_delta(downloaded[slot]);
            if (!output.replace_chunk(page.loaded[slot], downloaded[slot], error))
                return false;
        }
        if (input_delta != output_delta)
            return fail(error, "BCC32 resident window violated DeltaN_Q");
        *world = std::move(output);
        return true;
    } catch (const std::exception& exception) {
        return fail(error,
                    std::string("BCC32 resident transition failed: ") +
                        exception.what());
    }
}

std::uint32_t PagedWorldExecutor::required_resident_chunks(
    const WorldStore& world, std::uint32_t steps) {
    const std::size_t count = resident_window(world, steps).size();
    if (count > std::numeric_limits<std::uint32_t>::max())
        throw std::runtime_error("BCC32 resident window chunk count overflow");
    return static_cast<std::uint32_t>(count);
}

bool PagedWorldExecutor::advance_loaded(
    const std::filesystem::path& repository,
    WorldCommit input,
    ArtifactKind expected_kind,
    bool inverse,
    PageSchedule schedule,
    bool publish_mutable_root,
    TransitionReceipt* receipt,
    std::string* error,
    PublicationFailurePoint failure) {
    try {
        if (inverse && input.metadata.replay_boundary.completed_supersteps == 0u) {
            return fail(error, "BCC32 cannot reverse before the committed genesis boundary");
        }
        WorldCommit current = input;
        std::uint64_t page_count = 0u;
        std::vector<SiteWord> quiescent(kChunkSites, kQuiescentWord);
        executor_.initialize_q();

        for (std::uint32_t factor_index = 0u;
             factor_index < kForwardFactorCount;
             ++factor_index) {
            const std::uint32_t ordered_index =
                inverse ? kForwardFactorCount - factor_index - 1u : factor_index;
            const LawFactor factor = forward_factor(ordered_index);
            const CoordinateSet candidates = factor_candidates(current, factor);
            const std::vector<Page> pages = make_pages(
                candidates, factor, executor_.chunk_count(), schedule);
            ReferenceMap references;
            for (const ChunkObjectReference& chunk : current.chunks) {
                references.emplace(chunk.coordinate, chunk);
            }
            std::vector<ChunkObjectReference> output;
            output.reserve(candidates.size());
            boost::multiprecision::cpp_int input_delta = 0;
            boost::multiprecision::cpp_int output_delta = 0;
            for (const Page& page : pages) {
                const std::vector<DeviceChunkSlot> slots =
                    make_device_topology(page, executor_.chunk_count());
                check_cuda(cudaMemcpy(device_slots_,
                                      slots.data(),
                                      slots.size() * sizeof(DeviceChunkSlot),
                                      cudaMemcpyHostToDevice),
                           "upload BCC32 paged topology");

                // upload_words() is asynchronous. Keep every non-Q source alive
                // until apply_factor_window() has synchronized the same stream.
                std::vector<std::vector<SiteWord>> page_inputs(page.loaded.size());
                for (std::uint32_t slot = 0u; slot < page.loaded.size(); ++slot) {
                    const ChunkObjectReference* reference =
                        find_reference(references, page.loaded[slot]);
                    if (reference != nullptr) {
                        if (!read_world_chunk_object(
                                repository, *reference, &page_inputs[slot], error)) {
                            return false;
                        }
                        if (slot < page.core.size()) {
                            input_delta += reference->delta_n_q;
                        }
                    }
                }

                try {
                    for (std::uint32_t slot = 0u; slot < page.loaded.size(); ++slot) {
                        const std::span<const SiteWord> source =
                            page_inputs[slot].empty()
                                ? std::span<const SiteWord>(quiescent)
                                : std::span<const SiteWord>(page_inputs[slot]);
                        executor_.upload_words(
                            static_cast<std::uint64_t>(slot) * kChunkSites, source);
                    }
                    executor_.apply_factor_window(
                        {device_slots_, executor_.chunk_count()},
                        factor,
                        inverse,
                        static_cast<std::uint32_t>(page.core.size()),
                        static_cast<std::uint32_t>(page.loaded.size()));
                } catch (...) {
                    (void)cudaStreamSynchronize(nullptr);
                    throw;
                }

                std::vector<std::vector<SiteWord>> page_outputs;
                page_outputs.reserve(page.core.size());
                try {
                    for (std::uint32_t slot = 0u; slot < page.core.size(); ++slot) {
                        page_outputs.emplace_back(kChunkSites);
                        executor_.download_words(
                            (static_cast<std::uint64_t>(page.loaded.size()) + slot) *
                                kChunkSites,
                            page_outputs.back());
                    }
                    check_cuda(cudaStreamSynchronize(nullptr),
                               "synchronize BCC32 paged factor output downloads");
                } catch (...) {
                    (void)cudaStreamSynchronize(nullptr);
                    throw;
                }
                for (std::uint32_t slot = 0u; slot < page.core.size(); ++slot) {
                    ChunkObjectReference reference{};
                    bool materialized = false;
                    if (!put_world_chunk_object(repository,
                                                page.core[slot],
                                                page_outputs[slot],
                                                &reference,
                                                &materialized,
                                                error)) {
                        return false;
                    }
                    if (materialized) {
                        output_delta += reference.delta_n_q;
                        output.push_back(std::move(reference));
                    }
                }
                ++page_count;
            }
            if (input_delta != output_delta) {
                return fail(error,
                            "BCC32 paged factor violated the exact DeltaN_Q ledger");
            }
            std::sort(output.begin(), output.end(), [](const auto& left, const auto& right) {
                return CoordinateLess{}(left.coordinate, right.coordinate);
            });
            current.chunks = std::move(output);
            current.metadata.world_support = support_for_chunk_objects(current.chunks);
            current.identity = {};
        }

        ArtifactMetadata output_metadata = current.metadata;
        if (inverse) {
            --output_metadata.replay_boundary.completed_supersteps;
        } else {
            ++output_metadata.replay_boundary.completed_supersteps;
        }
        output_metadata.replay_boundary.next_factor = 0u;
        output_metadata.replay_boundary.predecessor_commit = input.identity;
        output_metadata.provenance.entry_event = {
            .kind = EntryEventKind::law_continuation,
        };
        output_metadata.provenance.replay_commitment = replay_commitment(
            input, inverse, output_metadata.replay_boundary.completed_supersteps);
        ContentAddress output_identity{};
        const bool published = publish_mutable_root
            ? detail::publish_law_continuation_from_executor(
                  output_metadata,
                  current.chunks,
                  expected_kind,
                  repository,
                  PublicationPrecondition::exact(input.identity),
                  input.identity,
                  material_state_identity(current.chunks),
                  output_metadata.replay_boundary.completed_supersteps,
                  &output_identity,
                  error,
                  failure)
            : detail::publish_immutable_law_continuation_from_executor(
                  output_metadata,
                  current.chunks,
                  expected_kind,
                  repository,
                  input.identity,
                  material_state_identity(current.chunks),
                  output_metadata.replay_boundary.completed_supersteps,
                  &output_identity,
                  error,
                  failure);
        if (!published) {
            return false;
        }
        if (receipt != nullptr) {
            *receipt = {
                .input_identity = input.identity,
                .output_identity = output_identity,
                .input_chunks = static_cast<std::uint64_t>(input.chunks.size()),
                .output_chunks = static_cast<std::uint64_t>(current.chunks.size()),
                .pages = page_count,
                .inverse = inverse,
                .execution_profile =
                    execution_profile_identity(executor_.chunk_count()),
            };
        }
        return true;
    } catch (const std::exception& exception) {
        return fail(error, std::string("BCC32 paged transition failed: ") + exception.what());
    }
}

namespace {

class PagedConditionedMatterExecutor final
    : public ConditionedMatterExecutor {
 public:
    PagedConditionedMatterExecutor(std::uint32_t aperture_chunks,
                                   bool reverse_core_order)
        : executor_(PagedWorldExecutor::windowed(aperture_chunks)),
          schedule_{.maximum_core_chunks = 0u,
                    .reverse_core_order = reverse_core_order} {}

    bool advance(WorldStore* world, bool inverse,
                 std::uint32_t supersteps,
                 std::uint64_t* page_count,
                 std::string* error) override {
        const bool advanced = executor_.advance_store_resident(
            world, supersteps, inverse, error, schedule_);
        if (advanced && page_count != nullptr)
            *page_count = world != nullptr && !world->chunks().empty() ? 1u : 0u;
        return advanced;
    }

    [[nodiscard]] std::uint64_t aperture_bytes() const override {
        return executor_.aperture_bytes();
    }

 private:
    PagedWorldExecutor executor_;
    PageSchedule schedule_{};
};

}  // namespace

std::shared_ptr<ConditionedMatterExecutor>
make_paged_conditioned_matter_executor(
    std::uint32_t aperture_chunks, bool reverse_core_order) {
    return std::make_shared<PagedConditionedMatterExecutor>(
        aperture_chunks, reverse_core_order);
}

}  // namespace substrate::bcc32
