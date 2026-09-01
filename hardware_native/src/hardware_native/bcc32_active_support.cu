#include "bcc32_active_support.cuh"

#include "bcc32_law.cuh"
#include "bcc32_developmental_append.hpp"
#include "bcc32_developmental_credit_service.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <string>

namespace substrate::bcc32 {
namespace {

constexpr std::uint32_t kThreads = 256u;
// Developmental append/service reaches the exact radius declared by the
// shared layout.  The ordinary post-edge spatial macros still reach only the
// pre-service fifteen-hop envelope.  Compose the two only from an actual
// developmental center, never isotropically from every differentiated site.
constexpr std::uint32_t kOrdinarySpatialMacroClosureRadius = 15u;
constexpr std::uint32_t kDevelopmentalCenterClosureRadius =
    kSpatialMacroClosureRadius + kOrdinarySpatialMacroClosureRadius + 2u;

void check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " +
                                 cudaGetErrorString(status));
    }
}

void require_default_stream(cudaStream_t stream) {
    if (stream != nullptr)
        throw std::invalid_argument(
            "BCC32 active support currently requires the serialized default stream");
}

std::uint32_t launch_blocks(std::size_t count) {
    const std::uint64_t blocks =
        (static_cast<std::uint64_t>(count) + kThreads - 1u) / kThreads;
    if (blocks == 0u || blocks > std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error("BCC32 active-support launch is out of range");
    }
    return static_cast<std::uint32_t>(blocks);
}

__host__ __device__ std::uint32_t encode_local(std::uint32_t x,
                                                std::uint32_t y,
                                                std::uint32_t z) {
    return (x * kChunkEdge + y) * kChunkEdge + z;
}

__host__ __device__ void decode_local(std::uint32_t local, std::uint32_t* x,
                                      std::uint32_t* y, std::uint32_t* z) {
    *x = local / (kChunkEdge * kChunkEdge);
    const std::uint32_t remainder = local % (kChunkEdge * kChunkEdge);
    *y = remainder / kChunkEdge;
    *z = remainder % kChunkEdge;
}

template <typename Slots>
__host__ __device__ bool hop_chunk(const Slots& slots, std::uint32_t chunk_count,
                                   std::uint32_t* chunk,
                                   std::uint32_t direction) {
    const std::int32_t next = slots[*chunk].bcc_neighbors[direction];
    if (next < 0 || static_cast<std::uint32_t>(next) >= chunk_count) return false;
    *chunk = static_cast<std::uint32_t>(next);
    return true;
}

template <typename Slots>
__host__ __device__ bool neighbor_slot(const Slots& slots,
                                       std::uint32_t chunk_count,
                                       std::uint64_t source,
                                       std::uint32_t direction,
                                       std::uint64_t* destination) {
    std::uint32_t chunk = static_cast<std::uint32_t>(source / kChunkSites);
    const std::uint32_t local = static_cast<std::uint32_t>(source % kChunkSites);
    if (chunk >= chunk_count) return false;
    std::uint32_t local_x = 0u;
    std::uint32_t local_y = 0u;
    std::uint32_t local_z = 0u;
    decode_local(local, &local_x, &local_y, &local_z);
    const Int3 offset = direction_offset(static_cast<Direction>(direction));
    std::int32_t x = static_cast<std::int32_t>(local_x) + offset.x;
    std::int32_t y = static_cast<std::int32_t>(local_y) + offset.y;
    std::int32_t z = static_cast<std::int32_t>(local_z) + offset.z;

    const bool all_negative = x < 0 && y < 0 && z < 0;
    const bool all_positive = x >= static_cast<std::int32_t>(kChunkEdge) &&
                              y >= static_cast<std::int32_t>(kChunkEdge) &&
                              z >= static_cast<std::int32_t>(kChunkEdge);
    if (direction == 3u && all_negative) {
        if (!hop_chunk(slots, chunk_count, &chunk, 3u)) return false;
        x += static_cast<std::int32_t>(kChunkEdge);
        y += static_cast<std::int32_t>(kChunkEdge);
        z += static_cast<std::int32_t>(kChunkEdge);
    } else if (direction == 7u && all_positive) {
        if (!hop_chunk(slots, chunk_count, &chunk, 7u)) return false;
        x -= static_cast<std::int32_t>(kChunkEdge);
        y -= static_cast<std::int32_t>(kChunkEdge);
        z -= static_cast<std::int32_t>(kChunkEdge);
    } else {
        std::int32_t* coordinates[3] = {&x, &y, &z};
        for (std::uint32_t axis = 0u; axis < 3u; ++axis) {
            if (*coordinates[axis] < 0) {
                if (!hop_chunk(slots, chunk_count, &chunk, axis + 4u)) return false;
                *coordinates[axis] += static_cast<std::int32_t>(kChunkEdge);
            } else if (*coordinates[axis] >= static_cast<std::int32_t>(kChunkEdge)) {
                if (!hop_chunk(slots, chunk_count, &chunk, axis)) return false;
                *coordinates[axis] -= static_cast<std::int32_t>(kChunkEdge);
            }
        }
    }
    if (x < 0 || y < 0 || z < 0 ||
        x >= static_cast<std::int32_t>(kChunkEdge) ||
        y >= static_cast<std::int32_t>(kChunkEdge) ||
        z >= static_cast<std::int32_t>(kChunkEdge)) {
        return false;
    }
    *destination = static_cast<std::uint64_t>(chunk) * kChunkSites +
                   encode_local(static_cast<std::uint32_t>(x),
                                static_cast<std::uint32_t>(y),
                                static_cast<std::uint32_t>(z));
    return true;
}

template <typename Slots>
bool walk_slot(const Slots& slots, std::uint32_t chunk_count,
               std::uint64_t start, std::uint32_t basis,
               std::int32_t steps, std::uint64_t* destination) {
    std::uint64_t current = start;
    const std::uint32_t direction = steps < 0 ? basis + 4u : basis;
    const std::uint32_t count =
        static_cast<std::uint32_t>(steps < 0 ? -steps : steps);
    for (std::uint32_t index = 0u; index < count; ++index)
        if (!neighbor_slot(slots, chunk_count, current, direction, &current))
            return false;
    *destination = current;
    return true;
}

template <typename Slots>
bool developmental_relative_slot(
    const Slots& slots, std::uint32_t chunk_count, std::uint64_t center,
    std::uint32_t marker, std::uint32_t path, std::uint32_t waste,
    std::uint32_t site, std::uint64_t* destination) {
    const DevelopmentalAppendOffset offset = developmental_append_offset(site);
    std::uint64_t current = center;
    if (!walk_slot(slots, chunk_count, current, marker, offset.marker,
                   &current) ||
        !walk_slot(slots, chunk_count, current, path, offset.path, &current) ||
        !walk_slot(slots, chunk_count, current, waste, offset.waste,
                   &current))
        return false;
    *destination = current;
    return true;
}

__device__ std::uint32_t site_parity(std::uint64_t slot) {
    std::uint32_t x = 0u;
    std::uint32_t y = 0u;
    std::uint32_t z = 0u;
    decode_local(static_cast<std::uint32_t>(slot % kChunkSites), &x, &y, &z);
    return (x + y + z) & 1u;
}

__global__ void gather_words_kernel(const SiteWord* words,
                                    const std::uint64_t* slots,
                                    SiteWord* samples, std::uint64_t count) {
    const std::uint64_t index =
        static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) samples[index] = words[slots[index]];
}

__global__ void active_site_kernel(SiteWord* words,
                                   const std::uint64_t* slots,
                                   const std::uint32_t* device_count,
                                   std::uint64_t capacity, bool inverse) {
    const std::uint64_t index =
        static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::uint64_t count =
        device_count == nullptr
            ? capacity
            : min(capacity, static_cast<std::uint64_t>(*device_count));
    if (index >= count) return;
    SiteWord word = words[slots[index]];
    if (inverse) apply_site_word_inverse(word);
    else apply_site_word_forward(word);
    words[slots[index]] = word;
}

__global__ void active_edge_kernel(SiteWord* words,
                                   const std::uint64_t* slots,
                                   const std::uint32_t* device_count,
                                   std::uint64_t capacity,
                                   DeviceChunkMap chunks,
                                   std::uint32_t basis,
                                   std::uint32_t parity, bool inverse) {
    const std::uint64_t index =
        static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::uint64_t count =
        device_count == nullptr
            ? capacity
            : min(capacity, static_cast<std::uint64_t>(*device_count));
    if (index >= count) return;
    const std::uint64_t source = slots[index];
    if (site_parity(source) != parity) return;
    std::uint64_t destination = 0u;
    if (!neighbor_slot(chunks.slots, chunks.chunk_count, source, basis,
                       &destination)) return;
    SiteWord source_word = words[source];
    SiteWord destination_word = words[destination];
    if (inverse) apply_edge_block_inverse(source_word, destination_word, basis);
    else apply_edge_block_forward(source_word, destination_word, basis);
    words[source] = source_word;
    words[destination] = destination_word;
}

__global__ void active_hole_snapshot_kernel(SiteWord* words,
                                            const std::uint64_t* slots,
                                            std::uint8_t* holes,
                                            const std::uint32_t* device_count,
                                            std::uint64_t capacity) {
    const std::uint64_t index =
        static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::uint64_t count =
        device_count == nullptr
            ? capacity
            : min(capacity, static_cast<std::uint64_t>(*device_count));
    if (index >= count) return;
    const std::uint64_t slot = slots[index];
    const SiteWord word = words[slot];
    holes[index] = static_cast<std::uint8_t>((~word) & kCarrierMask);
    words[slot] = static_cast<SiteWord>(word | kCarrierMask);
}

__global__ void active_hole_scatter_kernel(SiteWord* words,
                                           const std::uint64_t* slots,
                                           const std::uint8_t* holes,
                                           const std::uint32_t* device_count,
                                           std::uint64_t capacity,
                                           DeviceChunkMap chunks,
                                           bool inverse) {
    const std::uint64_t index =
        static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::uint64_t count =
        device_count == nullptr
            ? capacity
            : min(capacity, static_cast<std::uint64_t>(*device_count));
    if (index >= count) return;
    const std::uint8_t hole_mask = holes[index];
    for (std::uint32_t channel = 0u; channel < kStreamChannelCount; ++channel) {
        const std::uint8_t bit = static_cast<std::uint8_t>(1u << channel);
        if ((hole_mask & bit) == 0u) continue;
        const StreamChannel mapping = stream_channel(channel);
        std::uint32_t direction = direction_index(mapping.direction);
        if (inverse) direction ^= 4u;
        std::uint64_t destination = 0u;
        if (!neighbor_slot(chunks.slots, chunks.chunk_count, slots[index],
                           direction, &destination)) continue;
        atomicAnd(reinterpret_cast<unsigned int*>(words + destination),
                  ~static_cast<unsigned int>(carrier_bit(mapping.channel)));
    }
}

void normalize(std::vector<std::uint64_t>& slots, std::uint64_t site_count) {
    for (const std::uint64_t slot : slots)
        if (slot >= site_count)
            throw std::out_of_range("BCC32 active-support slot is outside aperture");
    std::sort(slots.begin(), slots.end());
    slots.erase(std::unique(slots.begin(), slots.end()), slots.end());
}

}  // namespace

void launch_active_site_graph_safe(
    SiteWord* words, const std::uint64_t* slots,
    const std::uint32_t* device_count, std::uint32_t capacity, bool inverse,
    cudaStream_t stream) {
    if (capacity == 0u || words == nullptr || slots == nullptr ||
        device_count == nullptr)
        throw std::invalid_argument("BCC32 graph-safe site window is invalid");
    active_site_kernel<<<launch_blocks(capacity), kThreads, 0, stream>>>(
        words, slots, device_count, capacity, inverse);
    check_cuda(cudaGetLastError(), "launch graph-safe BCC32 active K_site");
}

void launch_active_edge_graph_safe(
    SiteWord* words, const std::uint64_t* slots,
    const std::uint32_t* device_count, std::uint32_t capacity,
    const DeviceChunkMap& chunks, bool inverse, cudaStream_t stream) {
    if (capacity == 0u || words == nullptr || slots == nullptr ||
        device_count == nullptr)
        throw std::invalid_argument("BCC32 graph-safe edge window is invalid");
    if (!inverse) {
        for (std::uint32_t basis = 0u; basis < 4u; ++basis)
            for (std::uint32_t parity = 0u; parity < 2u; ++parity) {
                active_edge_kernel<<<launch_blocks(capacity), kThreads, 0,
                                     stream>>>(
                    words, slots, device_count, capacity, chunks, basis, parity,
                    false);
                check_cuda(cudaGetLastError(),
                           "launch graph-safe BCC32 active K_edge");
            }
    } else {
        for (std::uint32_t basis = 4u; basis-- > 0u;)
            for (std::uint32_t parity = 2u; parity-- > 0u;) {
                active_edge_kernel<<<launch_blocks(capacity), kThreads, 0,
                                     stream>>>(
                    words, slots, device_count, capacity, chunks, basis, parity,
                    true);
                check_cuda(cudaGetLastError(),
                           "launch graph-safe inverse BCC32 active K_edge");
            }
    }
}

void launch_active_stream_graph_safe(
    SiteWord* words, const std::uint64_t* slots, std::uint8_t* holes,
    const std::uint32_t* device_count, std::uint32_t capacity,
    const DeviceChunkMap& chunks, bool inverse, cudaStream_t stream) {
    if (capacity == 0u || words == nullptr || slots == nullptr ||
        holes == nullptr || device_count == nullptr)
        throw std::invalid_argument("BCC32 graph-safe stream window is invalid");
    active_hole_snapshot_kernel<<<launch_blocks(capacity), kThreads, 0,
                                  stream>>>(
        words, slots, holes, device_count, capacity);
    check_cuda(cudaGetLastError(),
               "launch graph-safe BCC32 active carrier snapshot");
    active_hole_scatter_kernel<<<launch_blocks(capacity), kThreads, 0,
                                 stream>>>(
        words, slots, holes, device_count, capacity, chunks, inverse);
    check_cuda(cudaGetLastError(),
               "launch graph-safe BCC32 active carrier stream");
}

CudaBcc32ActiveSupport::CudaBcc32ActiveSupport(
    CudaBcc32Executor& executor,
    std::span<const DeviceChunkSlot> host_topology,
    std::span<const std::uint64_t> initial_support,
    std::span<const std::uint64_t> persistent_anchors,
    ActiveSupportPolicy policy)
    : executor_(executor), topology_(host_topology.begin(), host_topology.end()),
      anchors_(persistent_anchors.begin(), persistent_anchors.end()),
      active_(initial_support.begin(), initial_support.end()), policy_(policy) {
    if (topology_.size() != executor_.chunk_count())
        throw std::invalid_argument("BCC32 active-support topology size mismatch");
    normalize(anchors_, executor_.site_count());
    active_.insert(active_.end(), anchors_.begin(), anchors_.end());
    normalize(active_, executor_.site_count());
    if (policy_ == ActiveSupportPolicy::fixed && !active_.empty()) {
        ensure_capacity(active_.size());
        check_cuda(cudaMemcpy(device_slots_, active_.data(),
                              active_.size() * sizeof(std::uint64_t),
                              cudaMemcpyHostToDevice),
                   "upload fixed BCC32 support");
    }
}

CudaBcc32ActiveSupport::~CudaBcc32ActiveSupport() {
    cudaFree(device_slots_);
    cudaFree(device_samples_);
    cudaFree(device_holes_);
    cudaFree(device_resolution_);
}

void CudaBcc32ActiveSupport::ensure_capacity(std::size_t count) {
    if (count <= capacity_) return;
    cudaFree(device_slots_);
    cudaFree(device_samples_);
    cudaFree(device_holes_);
    cudaFree(device_resolution_);
    device_slots_ = nullptr;
    device_samples_ = nullptr;
    device_holes_ = nullptr;
    device_resolution_ = nullptr;
    check_cuda(cudaMalloc(&device_slots_, count * sizeof(std::uint64_t)),
               "allocate BCC32 active slots");
    check_cuda(cudaMalloc(&device_samples_, count * sizeof(SiteWord)),
               "allocate BCC32 active samples");
    check_cuda(cudaMalloc(&device_holes_, count * sizeof(std::uint8_t)),
               "allocate BCC32 active holes");
    check_cuda(cudaMalloc(&device_resolution_, count * sizeof(std::uint8_t)),
               "allocate BCC32 active collision resolution");
    check_cuda(cudaMemset(device_resolution_, 0, count * sizeof(std::uint8_t)),
               "initialize BCC32 active collision resolution");
    capacity_ = count;
}

void CudaBcc32ActiveSupport::include(std::span<const std::uint64_t> slots) {
    if (policy_ == ActiveSupportPolicy::fixed)
        throw std::logic_error("fixed BCC32 support cannot be extended");
    active_.insert(active_.end(), slots.begin(), slots.end());
    normalize(active_, executor_.site_count());
    phase_history_.clear();
}

void CudaBcc32ActiveSupport::reconcile_external_contacts(
    std::span<const std::uint64_t> slots, cudaStream_t stream) {
    require_default_stream(stream);
    if (policy_ == ActiveSupportPolicy::fixed || slots.empty()) return;

    std::vector<std::uint64_t> touched(slots.begin(), slots.end());
    normalize(touched, executor_.site_count());
    if (touched.empty()) return;
    ensure_capacity(touched.size());
    check_cuda(cudaMemcpyAsync(device_slots_, touched.data(),
                               touched.size() * sizeof(std::uint64_t),
                               cudaMemcpyHostToDevice, stream),
               "upload external-contact support slots");
    gather_words_kernel<<<launch_blocks(touched.size()), kThreads, 0, stream>>>(
        executor_.device_words(), device_slots_, device_samples_,
        touched.size());
    check_cuda(cudaGetLastError(), "gather external-contact words");
    std::vector<SiteWord> words(touched.size());
    check_cuda(cudaMemcpyAsync(words.data(), device_samples_,
                               words.size() * sizeof(SiteWord),
                               cudaMemcpyDeviceToHost, stream),
               "download external-contact words");
    check_cuda(cudaStreamSynchronize(stream),
               "synchronize external-contact support reconciliation");

    for (std::size_t index = 0u; index < touched.size(); ++index) {
        const std::uint64_t slot = touched[index];
        const auto at = std::lower_bound(active_.begin(), active_.end(), slot);
        if (words[index] != kQuiescentWord) {
            if (at == active_.end() || *at != slot)
                active_.insert(at, slot);
            continue;
        }
        const bool anchored =
            std::binary_search(anchors_.begin(), anchors_.end(), slot);
        if (!anchored && at != active_.end() && *at == slot)
            active_.erase(at);
    }
    // Do not clear or append phase_history_: the reciprocal material exchange
    // is its own external journal.  Applying it again and reconciling the same
    // slots deterministically restores the pre-contact scheduler state.
}

void CudaBcc32ActiveSupport::reset(std::span<const std::uint64_t> support) {
    if (policy_ == ActiveSupportPolicy::fixed)
        throw std::logic_error("fixed BCC32 support cannot be rebound");
    rebind_active(support);
    phase_history_.clear();
}

void CudaBcc32ActiveSupport::rebind_active(
    std::span<const std::uint64_t> support) {
    active_.assign(support.begin(), support.end());
    active_.insert(active_.end(), anchors_.begin(), anchors_.end());
    normalize(active_, executor_.site_count());
}

CudaBcc32ActiveSupport::SupportDelta
CudaBcc32ActiveSupport::support_delta(
    std::span<const std::uint64_t> before,
    std::span<const std::uint64_t> after) const {
    SupportDelta result;
    std::set_difference(
        after.begin(), after.end(), before.begin(), before.end(),
        std::back_inserter(result.added));
    std::set_difference(
        before.begin(), before.end(), after.begin(), after.end(),
        std::back_inserter(result.removed));
    return result;
}

void CudaBcc32ActiveSupport::reverse_support_delta(
    std::vector<std::uint64_t>* support,
    const SupportDelta& delta) const {
    for (const std::uint64_t slot : delta.added)
        if (!std::binary_search(support->begin(), support->end(), slot))
            throw std::logic_error(
                "dynamic BCC32 inverse support delta lacks added slot");
    for (const std::uint64_t slot : delta.removed)
        if (std::binary_search(support->begin(), support->end(), slot))
            throw std::logic_error(
                "dynamic BCC32 inverse support delta retained removed slot");
    std::vector<std::uint64_t>& without_added = delta_scratch_a_;
    without_added.clear();
    without_added.reserve(support->size());
    std::set_difference(
        support->begin(), support->end(),
        delta.added.begin(), delta.added.end(),
        std::back_inserter(without_added));
    std::vector<std::uint64_t>& restored = delta_scratch_b_;
    restored.clear();
    restored.reserve(without_added.size() + delta.removed.size());
    std::merge(
        without_added.begin(), without_added.end(),
        delta.removed.begin(), delta.removed.end(),
        std::back_inserter(restored));
    *support = restored;
}

const std::uint64_t* CudaBcc32ActiveSupport::device_active_slots(
    cudaStream_t stream) {
    require_default_stream(stream);
    if (active_.empty()) return nullptr;
    if (policy_ == ActiveSupportPolicy::fixed) return device_slots_;
    ensure_capacity(active_.size());
    check_cuda(cudaMemcpyAsync(device_slots_, active_.data(),
                               active_.size() * sizeof(std::uint64_t),
                               cudaMemcpyHostToDevice, stream),
               "upload exact BCC32 active support");
    return device_slots_;
}

void CudaBcc32ActiveSupport::expand_radius_into(
    std::vector<std::uint64_t>& expanded,
    std::span<const std::uint64_t> slots, std::uint32_t radius) const {
    const std::size_t site_count =
        static_cast<std::size_t>(executor_.site_count());
    // One-time sizing only. The whole-array clear that used to run on every
    // call was 82% of total runtime at full aperture size; the BFS below
    // only ever touches a tiny fraction of expansion_seen_, so the reset
    // guard below clears exactly the touched entries instead.
    if (expansion_seen_.size() != site_count)
        expansion_seen_.assign(site_count, 0u);

    expanded.clear();
    expanded.reserve(slots.size());

    // expansion_seen_[s] is set to 1 for a slot s if and only if s is pushed
    // into `expanded` (both in the seed loop below and in the neighbour
    // loop). So resetting expansion_seen_[s] = 0 for every s in `expanded`
    // exactly restores the all-zero invariant, regardless of how the
    // function exits. This guard's destructor runs on both the normal
    // return path and on stack unwinding from the throw below, so a
    // mid-traversal exception cannot leave stale marks for the next call.
    struct SeenReset {
        std::vector<std::uint8_t>& seen;
        const std::vector<std::uint64_t>& touched;
        ~SeenReset() {
            for (const std::uint64_t slot : touched) seen[slot] = 0u;
        }
    } reset_guard{expansion_seen_, expanded};

    std::vector<std::uint64_t>& frontier = expansion_frontier_;
    frontier.clear();
    frontier.reserve(slots.size());
    for (const std::uint64_t slot : slots) {
        if (slot >= site_count)
            throw std::out_of_range(
                "BCC32 active-support slot is outside aperture");
        if (expansion_seen_[slot] == 0u) {
            expansion_seen_[slot] = 1u;
            frontier.push_back(slot);
            expanded.push_back(slot);
        }
    }

    for (std::uint32_t depth = 0u; depth < radius; ++depth) {
        std::vector<std::uint64_t>& next = expansion_next_;
        next.clear();
        next.reserve(std::min(site_count, frontier.size() * 4u));
        for (const std::uint64_t slot : frontier) {
            for (std::uint32_t direction = 0u; direction < 8u; ++direction) {
                std::uint64_t neighbor = 0u;
                if (!neighbor_slot(
                        topology_.data(),
                        static_cast<std::uint32_t>(topology_.size()), slot,
                        direction, &neighbor)) {
                    throw std::runtime_error(
                        "BCC32 active support is missing one-hop causal closure "
                        "at slot " +
                        std::to_string(slot) + ", direction " +
                        std::to_string(direction) + ", depth " +
                        std::to_string(depth));
                }
                if (expansion_seen_[neighbor] == 0u) {
                    expansion_seen_[neighbor] = 1u;
                    next.push_back(neighbor);
                    expanded.push_back(neighbor);
                }
            }
        }
        frontier.swap(next);
        if (frontier.empty()) break;
    }
}

void CudaBcc32ActiveSupport::validate_full_superstep_closure(
    cudaStream_t stream) {
    const std::vector<SiteWord> words = download_active_words(stream);
    const auto active_word = [&](std::uint64_t slot) {
        const auto found = std::lower_bound(active_.begin(), active_.end(), slot);
        if (found == active_.end() || *found != slot) return kQ;
        return words[static_cast<std::size_t>(found - active_.begin())];
    };
    const auto is_developmental_center = [&](std::uint64_t center) {
        for (std::uint32_t marker = 0u; marker < 4u; ++marker)
            for (std::uint32_t path = 0u; path < 4u; ++path) {
                if (path == marker) continue;
                for (std::uint32_t waste = 0u; waste < 4u; ++waste) {
                    if (waste == marker || waste == path) continue;
                    bool resource = true;
                    for (std::uint32_t site = 0u;
                         site < kDevelopmentalAppendSiteCount; ++site) {
                        std::uint64_t slot = 0u;
                        if (!developmental_relative_slot(
                                topology_.data(),
                                static_cast<std::uint32_t>(topology_.size()),
                                center, marker, path, waste, site, &slot) ||
                            active_word(slot) != developmental_append_word(
                                                     false, site, marker, path,
                                                     waste)) {
                            resource = false;
                            break;
                        }
                    }
                    if (resource) return true;

                    bool service_owner = true;
                    for (const std::uint32_t site :
                         {0u, 11u, 12u, 13u, 30u, 31u}) {
                        std::uint64_t slot = 0u;
                        if (!developmental_relative_slot(
                                topology_.data(),
                                static_cast<std::uint32_t>(topology_.size()),
                                center, marker, path, waste, site, &slot) ||
                            (active_word(slot) & ~kCarrierMask) !=
                                (developmental_append_product_word(
                                     site, marker, path, waste, 0u) &
                                 ~kCarrierMask)) {
                            service_owner = false;
                            break;
                        }
                    }
                    std::uint64_t enable = 0u;
                    service_owner =
                        service_owner &&
                        developmental_relative_slot(
                            topology_.data(),
                            static_cast<std::uint32_t>(topology_.size()),
                            center, marker, path, waste,
                            kDevelopmentalCreditServiceEnableSite, &enable) &&
                        developmental_credit_service_enable_word_matches(
                            active_word(enable), marker, path, waste);
                    if (service_owner) return true;
                }
            }
        return false;
    };
    std::vector<std::uint64_t> developmental_centers;
    developmental_centers.reserve(active_.size());
    for (std::size_t index = 0u; index < active_.size(); ++index)
        if (is_developmental_center(active_[index]))
            developmental_centers.push_back(active_[index]);
    std::uint32_t attempted_radius = kSpatialMacroClosureRadius;
    try {
        // This is exactly the candidate domain used by the first append and
        // service-aware refresh.  It is validated before any material factor.
        (void)expand_radius_into(expansion_result_, active_, attempted_radius);
        // Only a real resource/product center can create the remote banked
        // ingresses.  From those centers cover the exact developmental reach,
        // the later ordinary macro reach, and the edge/stream hops.
        if (!developmental_centers.empty()) {
            attempted_radius = kDevelopmentalCenterClosureRadius;
            (void)expand_radius_into(expansion_result_, developmental_centers,
                                     attempted_radius);
        }
    } catch (const std::runtime_error& failure) {
        // The per-root rescan that used to live here re-expanded every
        // active root one at a time (up to ~1.9M full expansions) purely to
        // name which root owned the failing slot in the error string. That
        // cost made the preflight unrunnable. The bulk expand_radius_into()
        // call
        // above already names the failing slot, direction and depth in
        // failure.what(); just add the aperture size and radius being
        // validated so a bounded-aperture failure stays legible without
        // another expansion.
        const std::uint64_t representative_root = active_.front();
        const SiteWord representative_word = active_word(representative_root);
        throw std::runtime_error(
            "BCC32 active-support superstep preflight failed over " +
            std::to_string(active_.size()) + " active roots at radius " +
            std::to_string(attempted_radius) + "; representative active root " +
            std::to_string(representative_root) + " word " +
            std::to_string(representative_word) + ": " + failure.what());
    }
}

void CudaBcc32ActiveSupport::refresh(
    std::span<const std::uint64_t> candidate_span, cudaStream_t stream) {
    require_default_stream(stream);
    // The span may alias active_, which the tail of this call rebuilds.
    std::vector<std::uint64_t>& candidates = refresh_candidates_;
    candidates.assign(candidate_span.begin(), candidate_span.end());
    for (const std::uint64_t slot : candidates)
        if (slot >= executor_.site_count())
            throw std::out_of_range(
                "BCC32 active-support candidate is outside aperture");
    ensure_capacity(candidates.size());
    if (!candidates.empty()) {
        check_cuda(cudaMemcpyAsync(device_slots_, candidates.data(),
                                   candidates.size() * sizeof(std::uint64_t),
                                   cudaMemcpyHostToDevice, stream),
                   "upload BCC32 active candidates");
        gather_words_kernel<<<launch_blocks(candidates.size()), kThreads, 0, stream>>>(
            executor_.device_words(), device_slots_, device_samples_,
            candidates.size());
        check_cuda(cudaGetLastError(), "gather BCC32 active candidates");
    }
    std::vector<SiteWord>& samples = refresh_samples_;
    samples.resize(candidates.size());
    if (!samples.empty())
        check_cuda(cudaMemcpyAsync(samples.data(), device_samples_,
                                   samples.size() * sizeof(SiteWord),
                                   cudaMemcpyDeviceToHost, stream),
                   "download BCC32 active candidates");
    check_cuda(cudaStreamSynchronize(stream),
               "synchronize BCC32 active-support refresh");
    active_ = anchors_;
    for (std::size_t index = 0u; index < candidates.size(); ++index)
        if (samples[index] != kQ) active_.push_back(candidates[index]);
    normalize(active_, executor_.site_count());
}

void CudaBcc32ActiveSupport::apply_site(bool inverse, cudaStream_t stream) {
    require_default_stream(stream);
    if (active_.empty()) return;
    ensure_capacity(active_.size());
    if (policy_ == ActiveSupportPolicy::dynamic)
        check_cuda(cudaMemcpyAsync(device_slots_, active_.data(),
                                   active_.size() * sizeof(std::uint64_t),
                                   cudaMemcpyHostToDevice, stream),
                   "upload BCC32 active site list");
    active_site_kernel<<<launch_blocks(active_.size()), kThreads, 0, stream>>>(
        executor_.mutable_device_words(), device_slots_, nullptr,
        active_.size(), inverse);
    check_cuda(cudaGetLastError(), "launch BCC32 active K_site");
}

void CudaBcc32ActiveSupport::apply_edge(const DeviceChunkMap& chunks,
                                        bool inverse, cudaStream_t stream) {
    require_default_stream(stream);
    std::span<const std::uint64_t> candidates = active_;
    if (policy_ == ActiveSupportPolicy::dynamic) {
        expand_radius_into(expansion_result_, active_, 1u);
        candidates = expansion_result_;
    }
    if (candidates.empty()) return;
    ensure_capacity(candidates.size());
    if (policy_ == ActiveSupportPolicy::dynamic)
        check_cuda(cudaMemcpyAsync(device_slots_, candidates.data(),
                                   candidates.size() * sizeof(std::uint64_t),
                                   cudaMemcpyHostToDevice, stream),
                   "upload BCC32 active edge list");
    if (!inverse) {
        for (std::uint32_t basis = 0u; basis < 4u; ++basis)
            for (std::uint32_t parity = 0u; parity < 2u; ++parity) {
                active_edge_kernel<<<launch_blocks(candidates.size()), kThreads, 0,
                                     stream>>>(
                    executor_.mutable_device_words(), device_slots_, nullptr,
                    candidates.size(), chunks, basis, parity, false);
                check_cuda(cudaGetLastError(), "launch BCC32 active K_edge");
            }
    } else {
        for (std::uint32_t basis = 4u; basis-- > 0u;)
            for (std::uint32_t parity = 2u; parity-- > 0u;) {
                active_edge_kernel<<<launch_blocks(candidates.size()), kThreads, 0,
                                     stream>>>(
                    executor_.mutable_device_words(), device_slots_, nullptr,
                    candidates.size(), chunks, basis, parity, true);
                check_cuda(cudaGetLastError(), "launch inverse BCC32 active K_edge");
            }
    }
    if (policy_ == ActiveSupportPolicy::dynamic) refresh(candidates, stream);
}

void CudaBcc32ActiveSupport::apply_stream(const DeviceChunkMap& chunks,
                                          bool inverse,
                                          cudaStream_t stream) {
    require_default_stream(stream);
    const std::span<const std::uint64_t> sources = active_;
    std::span<const std::uint64_t> candidates = sources;
    if (policy_ == ActiveSupportPolicy::dynamic) {
        expand_radius_into(expansion_result_, sources, 1u);
        candidates = expansion_result_;
    }
    if (sources.empty()) return;
    ensure_capacity(std::max(sources.size(), candidates.size()));
    if (policy_ == ActiveSupportPolicy::dynamic)
        check_cuda(cudaMemcpyAsync(device_slots_, sources.data(),
                                   sources.size() * sizeof(std::uint64_t),
                                   cudaMemcpyHostToDevice, stream),
                   "upload BCC32 active stream list");
    active_hole_snapshot_kernel<<<launch_blocks(sources.size()), kThreads, 0,
                                  stream>>>(
        executor_.mutable_device_words(), device_slots_, device_holes_, nullptr,
        sources.size());
    check_cuda(cudaGetLastError(), "snapshot BCC32 active carrier holes");
    active_hole_scatter_kernel<<<launch_blocks(sources.size()), kThreads, 0,
                                 stream>>>(
        executor_.mutable_device_words(), device_slots_, device_holes_, nullptr,
        sources.size(), chunks, inverse);
    check_cuda(cudaGetLastError(), "scatter BCC32 active carrier holes");
    if (policy_ == ActiveSupportPolicy::dynamic) refresh(candidates, stream);
}

void CudaBcc32ActiveSupport::apply_superstep(const DeviceChunkMap& chunks,
                                             cudaStream_t stream) {
    require_default_stream(stream);
    // The executor's genesis guard counts world advances. This path does not go
    // through CudaBcc32Executor::apply_superstep, so without this the guard is
    // blind to every tick the tree actually runs.
    executor_.note_world_advance();
    PhaseSupportJournal journal;
    std::vector<std::uint64_t>& tick_entry_support = tick_entry_support_;
    if (policy_ == ActiveSupportPolicy::dynamic) {
        // include() is deliberately permissive: an external contact can make a
        // formerly absent site material between ticks. Its dense word is the
        // authority at tick entry. Retire only non-anchor candidates that are
        // exactly Q before their topology reach is inspected; refresh keeps
        // explicit persistent anchors even while they are Q.
        tick_entry_support = active_;
        refresh(tick_entry_support, stream);
        journal.entry_to_append =
            support_delta(tick_entry_support, active_);
    }
    // Every later dynamic closure is topology-only and deterministic. Validate
    // the complete append/edge/macro/stream reach before the first material
    // factor so a missing aperture halo cannot strand a partial forward tick.
    if (policy_ == ActiveSupportPolicy::dynamic && !active_.empty()) {
        try {
            validate_full_superstep_closure(stream);
        } catch (...) {
            // Entry compaction is derived metadata, but a rejected tick is
            // nevertheless failure-atomic for both material and support.
            active_ = tick_entry_support;
            throw;
        }
    }
    std::vector<std::uint64_t>& phase_support = phase_support_;
    if (policy_ == ActiveSupportPolicy::dynamic)
        phase_support = active_;
    std::vector<std::uint64_t>& append_closure = expansion_result_;
    if (policy_ == ActiveSupportPolicy::dynamic && !active_.empty())
        expand_radius_into(append_closure, active_, kSpatialMacroClosureRadius);
    else
        append_closure.clear();
    if (!active_.empty())
        executor_.apply_active_developmental_learned_receptor(
            chunks, false, device_active_slots(stream), active_.size(), stream);
    if (!active_.empty())
        executor_.apply_active_developmental_append(
            chunks, false, device_active_slots(stream), active_.size(), stream);
    if (!append_closure.empty()) refresh(append_closure, stream);
    if (!active_.empty())
        executor_.apply_active_eligibility_residual_junction(
            chunks, false, device_active_slots(stream), active_.size(), stream);
    apply_site(false, stream);
    if (policy_ == ActiveSupportPolicy::dynamic) {
        journal.append_to_edge = support_delta(phase_support, active_);
        phase_support = active_;
    }
    apply_edge(chunks, false, stream);
    if (policy_ == ActiveSupportPolicy::dynamic) {
        journal.edge_to_macro = support_delta(phase_support, active_);
        phase_support = active_;
    }
    if (!active_.empty())
        executor_.apply_active_macro_factors(
            chunks, false, device_active_slots(stream), active_.size(),
            device_resolution_, stream);
    if (policy_ == ActiveSupportPolicy::dynamic && !active_.empty()) {
        expand_radius_into(expansion_result_, active_,
                           kOrdinarySpatialMacroClosureRadius);
        refresh(expansion_result_, stream);
    }
    if (!active_.empty())
        executor_.apply_active_prediction_residual_route_toggle(
            chunks, false, device_active_slots(stream), active_.size(),
            device_resolution_, stream);
    if (!active_.empty())
        executor_.apply_active_developmental_credit_service(
            chunks, false, device_active_slots(stream), active_.size(), stream);
    if (policy_ == ActiveSupportPolicy::dynamic) {
        journal.macro_to_prediction = support_delta(phase_support, active_);
        phase_support = active_;
    }
    apply_stream(chunks, false, stream);
    if (policy_ == ActiveSupportPolicy::dynamic) {
        journal.prediction_to_end = support_delta(phase_support, active_);
        phase_history_.push_back(std::move(journal));
    }
}

void CudaBcc32ActiveSupport::apply_superstep_inverse(
    const DeviceChunkMap& chunks, cudaStream_t stream) {
    require_default_stream(stream);
    executor_.note_world_advance();
    if (policy_ == ActiveSupportPolicy::fixed) {
        apply_stream(chunks, true, stream);
        if (!active_.empty())
            executor_.apply_active_developmental_credit_service(
                chunks, true, device_active_slots(stream), active_.size(),
                stream);
        if (!active_.empty())
            executor_.apply_active_prediction_residual_route_toggle(
                chunks, true, device_active_slots(stream), active_.size(),
                device_resolution_, stream);
        if (!active_.empty())
            executor_.apply_active_macro_factors(
                chunks, true, device_active_slots(stream), active_.size(),
                device_resolution_, stream);
        apply_edge(chunks, true, stream);
        apply_site(true, stream);
        if (!active_.empty())
            executor_.apply_active_eligibility_residual_junction(
                chunks, true, device_active_slots(stream), active_.size(),
                stream);
        if (!active_.empty())
            executor_.apply_active_developmental_append(
                chunks, true, device_active_slots(stream), active_.size(),
                stream);
        if (!active_.empty())
            executor_.apply_active_developmental_learned_receptor(
                chunks, true, device_active_slots(stream), active_.size(),
                stream);
        return;
    }

    if (phase_history_.empty())
        throw std::logic_error(
            "dynamic BCC32 inverse crossed a checkpoint or external-reset "
            "boundary without phase-support history");
    const PhaseSupportJournal& journal = phase_history_.back();
    std::vector<std::uint64_t>& phase_support = phase_support_;
    phase_support = active_;
    reverse_support_delta(&phase_support, journal.prediction_to_end);

    // Stream inverse must begin on the actual tick-end destinations.  Every
    // earlier inverse phase is rebound to the exact sparse domain on which its
    // forward conjugate ran; refresh is derived metadata and is not itself a
    // material operation.
    apply_stream(chunks, true, stream);
    rebind_active(phase_support);
    if (!active_.empty())
        executor_.apply_active_developmental_credit_service(
            chunks, true, device_active_slots(stream), active_.size(), stream);
    if (!active_.empty())
        executor_.apply_active_prediction_residual_route_toggle(
            chunks, true, device_active_slots(stream), active_.size(),
            device_resolution_, stream);
    reverse_support_delta(&phase_support, journal.macro_to_prediction);
    rebind_active(phase_support);
    if (!active_.empty())
        executor_.apply_active_macro_factors(
            chunks, true, device_active_slots(stream), active_.size(),
            device_resolution_, stream);
    reverse_support_delta(&phase_support, journal.edge_to_macro);
    rebind_active(phase_support);
    apply_edge(chunks, true, stream);
    rebind_active(phase_support);
    apply_site(true, stream);
    if (!active_.empty())
        executor_.apply_active_eligibility_residual_junction(
            chunks, true, device_active_slots(stream), active_.size(), stream);
    reverse_support_delta(&phase_support, journal.append_to_edge);
    rebind_active(phase_support);
    if (!active_.empty())
        executor_.apply_active_developmental_append(
            chunks, true, device_active_slots(stream), active_.size(), stream);
    if (!active_.empty())
        executor_.apply_active_developmental_learned_receptor(
            chunks, true, device_active_slots(stream), active_.size(), stream);
    rebind_active(phase_support);
    reverse_support_delta(&phase_support, journal.entry_to_append);
    rebind_active(phase_support);
    phase_history_.pop_back();
}

std::vector<SiteWord> CudaBcc32ActiveSupport::download_active_words(
    cudaStream_t stream) const {
    require_default_stream(stream);
    std::vector<SiteWord> words(active_.size());
    if (words.empty()) return words;
    auto* mutable_self = const_cast<CudaBcc32ActiveSupport*>(this);
    const std::uint64_t* slots = mutable_self->device_active_slots(stream);
    gather_words_kernel<<<launch_blocks(active_.size()), kThreads, 0, stream>>>(
        executor_.device_words(), slots, mutable_self->device_samples_,
        active_.size());
    check_cuda(cudaGetLastError(), "gather BCC32 active words");
    check_cuda(cudaMemcpyAsync(words.data(), mutable_self->device_samples_,
                               words.size() * sizeof(SiteWord),
                               cudaMemcpyDeviceToHost, stream),
               "download BCC32 active words");
    check_cuda(cudaStreamSynchronize(stream),
               "synchronize BCC32 active word download");
    return words;
}

}  // namespace substrate::bcc32
