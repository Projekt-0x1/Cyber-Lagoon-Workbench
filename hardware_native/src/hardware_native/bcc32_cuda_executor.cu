#include <cuda_runtime.h>

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

#include "bcc32_cuda_executor.cuh"
#include "bcc32_active_support.cuh"
#include "bcc32_developmental_append.hpp"
#include "bcc32_developmental_credit_service.hpp"
#include "bcc32_developmental_learned_receptor.hpp"
#include "bcc32_law.cuh"
#include "bcc32_spatial_macro_executor.cuh"

namespace substrate::bcc32 {
namespace {

constexpr std::uint32_t kThreads = 256u;
constexpr std::uint32_t kAuditBlocks = 4'096u;
constexpr std::int64_t kInt64Min = (-9'223'372'036'854'775'807LL - 1LL);
constexpr std::int64_t kInt64Max = 9'223'372'036'854'775'807LL;

__global__ void prepare_active_window_kernel(
    const std::uint64_t* source_slots, const std::uint32_t* source_count,
    std::uint64_t* window_slots, std::uint32_t window_capacity) {
    const std::uint32_t index =
        static_cast<std::uint32_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= window_capacity) return;
    const std::uint32_t count = source_count == nullptr ? 0u : *source_count;
    if (source_slots != nullptr && index < count)
        window_slots[index] = source_slots[index];
}

[[noreturn]] void throw_cuda(cudaError_t status, const char* operation) {
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(status));
}

void check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw_cuda(status, operation);
    }
}

[[nodiscard]] std::uint32_t launch_blocks(std::uint64_t count) {
    const std::uint64_t blocks = (count + kThreads - 1ull) / kThreads;
    if (blocks == 0ull || blocks > std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error("CUDA launch grid exceeds one-dimensional limits");
    }
    return static_cast<std::uint32_t>(blocks);
}

[[nodiscard]] std::uint32_t audit_blocks(std::uint64_t scratch_bytes) {
    if (scratch_bytes == 0ull) {
        throw std::logic_error("P carrier snapshot must contain at least one byte");
    }
    return static_cast<std::uint32_t>(std::min<std::uint64_t>(
        scratch_bytes, kAuditBlocks));
}

[[nodiscard]] __device__ std::uint64_t site_slot(std::uint32_t chunk,
                                                  std::uint32_t local) {
    return static_cast<std::uint64_t>(chunk) * kChunkSites + local;
}

__device__ void decode_local(std::uint32_t local, std::uint32_t* x,
                             std::uint32_t* y, std::uint32_t* z) {
    *x = local / (kChunkEdge * kChunkEdge);
    const std::uint32_t remainder = local % (kChunkEdge * kChunkEdge);
    *y = remainder / kChunkEdge;
    *z = remainder % kChunkEdge;
}

[[nodiscard]] __device__ std::uint32_t encode_local(std::uint32_t x,
                                                      std::uint32_t y,
                                                      std::uint32_t z) {
    return (x * kChunkEdge + y) * kChunkEdge + z;
}

[[nodiscard]] __device__ bool checked_coordinate_step(std::int64_t value,
                                                       std::int32_t offset,
                                                       std::int64_t* result) {
    if ((offset > 0 && value == kInt64Max) ||
        (offset < 0 && value == kInt64Min)) {
        return false;
    }
    *result = value + offset;
    return true;
}

[[nodiscard]] __device__ bool topology_neighbor_valid(
    const DeviceChunkMap& chunks, std::uint32_t chunk, std::uint32_t direction) {
    const DeviceChunkSlot& source = chunks.slots[chunk];
    const std::int32_t neighbor = source.bcc_neighbors[direction];
    if (neighbor == DeviceChunkSlot::kMissing) {
        return true;
    }
    if (neighbor < 0 || static_cast<std::uint32_t>(neighbor) >= chunks.chunk_count) {
        return false;
    }
    const DeviceChunkSlot& destination = chunks.slots[neighbor];
    if (destination.bcc_neighbors[direction ^ 4u] !=
        static_cast<std::int32_t>(chunk)) {
        return false;
    }
    const Int3 offset = direction_offset(static_cast<Direction>(direction));
    std::int64_t x = 0;
    std::int64_t y = 0;
    std::int64_t z = 0;
    return checked_coordinate_step(source.chunk_x, offset.x, &x) &&
           checked_coordinate_step(source.chunk_y, offset.y, &y) &&
           checked_coordinate_step(source.chunk_z, offset.z, &z) &&
           destination.chunk_x == x && destination.chunk_y == y &&
           destination.chunk_z == z;
}

[[nodiscard]] __device__ bool hop_chunk(const DeviceChunkMap& chunks,
                                         std::uint32_t* chunk,
                                         std::uint32_t direction) {
    const std::int32_t next = chunks.slots[*chunk].bcc_neighbors[direction];
    if (next == DeviceChunkSlot::kMissing) {
        return false;
    }
    if (next < 0 || static_cast<std::uint32_t>(next) >= chunks.chunk_count) {
        return false;
    }
    *chunk = static_cast<std::uint32_t>(next);
    return true;
}

// Resolve one BCC site neighbor through direct dense chunk storage.  u3 and
// -u3 can cross a subset of the three Cartesian chunk faces; mixed crossings
// compose the corresponding axial BCC chunk links without a per-site table.
[[nodiscard]] __device__ bool neighbor_slot(const DeviceChunkMap& chunks,
                                            std::uint64_t source,
                                            std::uint32_t direction,
                                            std::uint64_t* destination) {
    std::uint32_t chunk = static_cast<std::uint32_t>(source / kChunkSites);
    std::uint32_t local = static_cast<std::uint32_t>(source % kChunkSites);
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
        if (!hop_chunk(chunks, &chunk, 3u)) {
            return false;
        }
        x += static_cast<std::int32_t>(kChunkEdge);
        y += static_cast<std::int32_t>(kChunkEdge);
        z += static_cast<std::int32_t>(kChunkEdge);
    } else if (direction == 7u && all_positive) {
        if (!hop_chunk(chunks, &chunk, 7u)) {
            return false;
        }
        x -= static_cast<std::int32_t>(kChunkEdge);
        y -= static_cast<std::int32_t>(kChunkEdge);
        z -= static_cast<std::int32_t>(kChunkEdge);
    } else {
        std::int32_t* coordinates[3] = {&x, &y, &z};
        for (std::uint32_t axis = 0u; axis < 3u; ++axis) {
            if (*coordinates[axis] < 0) {
                if (!hop_chunk(chunks, &chunk, axis + 4u)) {
                    return false;
                }
                *coordinates[axis] += static_cast<std::int32_t>(kChunkEdge);
            } else if (*coordinates[axis] >= static_cast<std::int32_t>(kChunkEdge)) {
                if (!hop_chunk(chunks, &chunk, axis)) {
                    return false;
                }
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
    *destination = site_slot(chunk, encode_local(static_cast<std::uint32_t>(x),
                                                  static_cast<std::uint32_t>(y),
                                                  static_cast<std::uint32_t>(z)));
    return true;
}

__global__ void copy_core_kernel(const SiteWord* words, SiteWord* output,
                                 std::uint64_t core_sites) {
  const std::uint64_t slot = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (slot < core_sites)
    output[slot] = words[slot];
}

__global__ void fill_q_kernel(SiteWord* words, std::uint64_t site_count) {
    const std::uint64_t slot = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x +
                               threadIdx.x;
    if (slot < site_count) {
        words[slot] = kQ;
    }
}

__global__ void k_site_kernel(SiteWord* words, std::uint64_t site_count,
                              bool inverse) {
    const std::uint64_t slot = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x +
                               threadIdx.x;
    if (slot >= site_count) {
        return;
    }
    SiteWord word = words[slot];
    if (inverse) {
        apply_site_word_inverse(word);
    } else {
        apply_site_word_forward(word);
    }
    words[slot] = word;
}

__global__ void k_site_page_kernel(const SiteWord* words,
                                   SiteWord* output,
                                   std::uint64_t core_sites,
                                   bool inverse) {
    const std::uint64_t slot = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x +
                               threadIdx.x;
    if (slot >= core_sites) return;
    SiteWord word = words[slot];
    if (inverse) {
        apply_site_word_inverse(word);
    } else {
        apply_site_word_forward(word);
    }
    output[slot] = word;
}

[[nodiscard]] __device__ SiteWord page_source_edge_mask(std::uint32_t basis) {
    return carrier_bit(basis) | face_bit(basis) | owned_bond_bit(basis) |
           energy_bit(basis);
}

[[nodiscard]] __device__ SiteWord page_destination_edge_mask(std::uint32_t basis) {
    return carrier_bit(basis + 4u) | face_bit(basis + 4u) |
           channel_bit(kConformationShift, basis) |
           channel_bit(kReactiveShift, basis);
}

__global__ void k_edge_page_kernel(const SiteWord* words,
                                   SiteWord* output,
                                   std::uint64_t core_sites,
                                   DeviceChunkMap chunks,
                                   bool inverse) {
    const std::uint64_t core = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x +
                               threadIdx.x;
    if (core >= core_sites) return;
    const SiteWord original = words[core];
    SiteWord result = original;
    for (std::uint32_t basis = 0u; basis < 4u; ++basis) {
        std::uint64_t positive = 0u;
        std::uint64_t negative = 0u;
        if (!neighbor_slot(chunks, core, basis, &positive) ||
            !neighbor_slot(chunks, core, basis + 4u, &negative)) {
            return;
        }

        SiteWord source_word = original;
        SiteWord destination_word = words[positive];
        if (inverse) {
            apply_edge_block_inverse(source_word, destination_word, basis);
        } else {
            apply_edge_block_forward(source_word, destination_word, basis);
        }
        const SiteWord source_mask = page_source_edge_mask(basis);
        result = static_cast<SiteWord>((result & ~source_mask) |
                                       (source_word & source_mask));

        source_word = words[negative];
        destination_word = original;
        if (inverse) {
            apply_edge_block_inverse(source_word, destination_word, basis);
        } else {
            apply_edge_block_forward(source_word, destination_word, basis);
        }
        const SiteWord destination_mask = page_destination_edge_mask(basis);
        result = static_cast<SiteWord>((result & ~destination_mask) |
                                       (destination_word & destination_mask));
    }
    output[core] = result;
}

__global__ void stream_page_kernel(const SiteWord* words,
                                   SiteWord* output,
                                   std::uint64_t core_sites,
                                   DeviceChunkMap chunks,
                                   bool inverse) {
    const std::uint64_t destination =
        static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (destination >= core_sites) return;
    SiteWord result = words[destination] & ~kCarrierMask;
    for (std::uint32_t index = 0u; index < kStreamChannelCount; ++index) {
        const StreamChannel mapping = stream_channel(index);
        const std::uint32_t direction = direction_index(mapping.direction);
        const std::uint32_t source_direction =
            inverse ? direction : (direction ^ 4u);
        std::uint64_t source = 0u;
        if (!neighbor_slot(chunks, destination, source_direction, &source)) return;
        const SiteWord bit = carrier_bit(mapping.channel);
        if (bit_is_set(words[source], bit)) result |= bit;
    }
    output[destination] = result;
}

__global__ void k_edge_parallel_kernel(SiteWord* words,
                                       std::uint64_t site_count,
                                       DeviceChunkMap chunks,
                                       bool inverse) {
    const std::uint64_t source =
        static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (source >= site_count) {
        return;
    }
    const SiteWord source_original = words[source];
    if (source_original == kQ) {
        bool all_destinations_quiescent = true;
        for (std::uint32_t basis = 0u; basis < 4u; ++basis) {
            std::uint64_t destination = 0ull;
            if (neighbor_slot(chunks, source, basis, &destination) &&
                words[destination] != kQ) {
                all_destinations_quiescent = false;
                break;
            }
        }
        if (all_destinations_quiescent) {
            return;
        }
    }
    // Positive edges own disjoint source/destination role masks at every site.
    // Commit only each edge's XOR delta so concurrent endpoint updates commute.
    SiteWord source_delta = 0u;
    for (std::uint32_t basis = 0u; basis < 4u; ++basis) {
        std::uint64_t destination = 0ull;
        if (!neighbor_slot(chunks, source, basis, &destination)) {
            continue;
        }
        SiteWord source_word = source_original;
        const SiteWord destination_original = words[destination];
        SiteWord destination_word = destination_original;
        if (inverse) {
            apply_edge_block_inverse(source_word, destination_word, basis);
        } else {
            apply_edge_block_forward(source_word, destination_word, basis);
        }
        source_delta |= static_cast<SiteWord>(
            (source_original ^ source_word) & page_source_edge_mask(basis));
        const SiteWord destination_delta = static_cast<SiteWord>(
            (destination_original ^ destination_word) &
            page_destination_edge_mask(basis));
        if (destination_delta != 0u) {
            atomicXor(reinterpret_cast<unsigned int*>(words + destination),
                      static_cast<unsigned int>(destination_delta));
        }
    }
    if (source_delta != 0u) {
        atomicXor(reinterpret_cast<unsigned int*>(words + source),
                  static_cast<unsigned int>(source_delta));
    }
}

__global__ void snapshot_carriers_kernel(const SiteWord* words,
                                         std::uint8_t* carrier_snapshot,
                                         std::uint64_t site_count) {
    const std::uint64_t slot =
        static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (slot >= site_count) {
        return;
    }
    carrier_snapshot[slot] = static_cast<std::uint8_t>(words[slot] & kCarrierMask);
}

__global__ void gather_carriers_kernel(SiteWord* words,
                                       const std::uint8_t* carrier_snapshot,
                                       std::uint64_t site_count,
                                       DeviceChunkMap chunks,
                                       bool inverse) {
    const std::uint64_t destination =
        static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (destination >= site_count) {
        return;
    }
    SiteWord carriers = 0u;
    for (std::uint32_t index = 0u; index < kStreamChannelCount; ++index) {
        const StreamChannel mapping = stream_channel(index);
        const std::uint32_t direction = direction_index(mapping.direction);
        const std::uint32_t source_direction = inverse ? direction : (direction ^ 4u);
        std::uint64_t source = 0ull;
        const bool source_present =
            neighbor_slot(chunks, destination, source_direction, &source);
        if (!source_present ||
            (carrier_snapshot[source] & static_cast<std::uint8_t>(carrier_bit(mapping.channel))) !=
                0u) {
            carriers |= carrier_bit(mapping.channel);
        }
    }
    words[destination] = static_cast<SiteWord>((words[destination] & ~kCarrierMask) | carriers);
}

__global__ void topology_flags_kernel(std::uint8_t* flags,
                                      DeviceChunkMap chunks) {
    __shared__ std::uint8_t local_flags[kThreads];
    const std::uint64_t global_thread =
        static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::uint64_t stride = static_cast<std::uint64_t>(gridDim.x) * blockDim.x;
    std::uint8_t invalid = 0u;
    for (std::uint64_t chunk = global_thread; chunk < chunks.chunk_count;
         chunk += stride) {
        for (std::uint32_t direction = 0u; direction < 8u; ++direction) {
            if (!topology_neighbor_valid(chunks, static_cast<std::uint32_t>(chunk),
                                         direction)) {
                invalid = 1u;
            }
        }
    }
    local_flags[threadIdx.x] = invalid;
    __syncthreads();
    for (std::uint32_t width = blockDim.x / 2u; width != 0u; width >>= 1u) {
        if (threadIdx.x < width) {
            local_flags[threadIdx.x] |= local_flags[threadIdx.x + width];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0u) {
        flags[blockIdx.x] = local_flags[0];
    }
}

__global__ void core_closure_flags_kernel(std::uint8_t* flags,
                                          DeviceChunkMap chunks,
                                          std::uint32_t core_count) {
    __shared__ std::uint8_t local_flags[kThreads];
    const std::uint64_t global_thread =
        static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::uint64_t stride =
        static_cast<std::uint64_t>(gridDim.x) * blockDim.x;
    std::uint8_t missing = 0u;
    for (std::uint64_t chunk = global_thread; chunk < core_count; chunk += stride) {
        for (std::uint32_t direction = 0u; direction < 8u; ++direction) {
            if (chunks.slots[chunk].bcc_neighbors[direction] ==
                DeviceChunkSlot::kMissing) {
                missing = 1u;
            }
        }
        for (std::int32_t dx = -1; dx <= 1; ++dx) {
            for (std::int32_t dy = -1; dy <= 1; ++dy) {
                for (std::int32_t dz = -1; dz <= 1; ++dz) {
                    if (dx == 0 && dy == 0 && dz == 0) continue;
                    std::uint32_t reached = static_cast<std::uint32_t>(chunk);
                    if ((dx < 0 && !hop_chunk(chunks, &reached, 4u)) ||
                        (dx > 0 && !hop_chunk(chunks, &reached, 0u)) ||
                        (dy < 0 && !hop_chunk(chunks, &reached, 5u)) ||
                        (dy > 0 && !hop_chunk(chunks, &reached, 1u)) ||
                        (dz < 0 && !hop_chunk(chunks, &reached, 6u)) ||
                        (dz > 0 && !hop_chunk(chunks, &reached, 2u))) {
                        missing = 1u;
                    }
                }
            }
        }
    }
    local_flags[threadIdx.x] = missing;
    __syncthreads();
    for (std::uint32_t width = blockDim.x / 2u; width != 0u; width >>= 1u) {
        if (threadIdx.x < width) {
            local_flags[threadIdx.x] |= local_flags[threadIdx.x + width];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0u) flags[blockIdx.x] = local_flags[0];
}

__global__ void closure_flags_kernel(const SiteWord* words,
                                     std::uint8_t* flags,
                                     std::uint64_t site_count,
                                     DeviceChunkMap chunks) {
    __shared__ std::uint8_t local_flags[kThreads];
    const std::uint64_t global_thread =
        static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::uint64_t stride = static_cast<std::uint64_t>(gridDim.x) * blockDim.x;
    std::uint8_t missing_closure = 0u;
    for (std::uint64_t source = global_thread; source < site_count;
         source += stride) {
        if (words[source] == kQ) {
            continue;
        }
        for (std::uint32_t first = 0u; first < 8u; ++first) {
            std::uint64_t first_neighbor = 0ull;
            if (!neighbor_slot(chunks, source, first, &first_neighbor)) {
                missing_closure = 1u;
                continue;
            }
            for (std::uint32_t second = 0u; second < 8u; ++second) {
                std::uint64_t ignored = 0ull;
                if (!neighbor_slot(chunks, first_neighbor, second, &ignored)) {
                    missing_closure = 1u;
                }
            }
        }
        // Spatial macro-factors may read or write farther than the two-hop
        // site/edge/stream stencil.  A resident aperture must therefore carry
        // the complete adjacent chunk halo whenever represented matter lies
        // within the declared macro closure radius of a face, edge, or corner.
        // The developmental append descriptor reaches at most ten logical
        // coordinates; kSpatialMacroClosureRadius intentionally leaves margin.
        std::uint32_t x = 0u;
        std::uint32_t y = 0u;
        std::uint32_t z = 0u;
        decode_local(static_cast<std::uint32_t>(source % kChunkSites), &x, &y,
                     &z);
        const bool near_negative[3] = {
            x < kSpatialMacroClosureRadius,
            y < kSpatialMacroClosureRadius,
            z < kSpatialMacroClosureRadius};
        const bool near_positive[3] = {
            x >= kChunkEdge - kSpatialMacroClosureRadius,
            y >= kChunkEdge - kSpatialMacroClosureRadius,
            z >= kChunkEdge - kSpatialMacroClosureRadius};
        for (std::int32_t dx = -1; dx <= 1; ++dx) {
            if ((dx < 0 && !near_negative[0]) ||
                (dx > 0 && !near_positive[0]))
                continue;
            for (std::int32_t dy = -1; dy <= 1; ++dy) {
                if ((dy < 0 && !near_negative[1]) ||
                    (dy > 0 && !near_positive[1]))
                    continue;
                for (std::int32_t dz = -1; dz <= 1; ++dz) {
                    if ((dz < 0 && !near_negative[2]) ||
                        (dz > 0 && !near_positive[2]) ||
                        (dx == 0 && dy == 0 && dz == 0))
                        continue;
                    std::uint32_t reached =
                        static_cast<std::uint32_t>(source / kChunkSites);
                    if ((dx < 0 && !hop_chunk(chunks, &reached, 4u)) ||
                        (dx > 0 && !hop_chunk(chunks, &reached, 0u)) ||
                        (dy < 0 && !hop_chunk(chunks, &reached, 5u)) ||
                        (dy > 0 && !hop_chunk(chunks, &reached, 1u)) ||
                        (dz < 0 && !hop_chunk(chunks, &reached, 6u)) ||
                        (dz > 0 && !hop_chunk(chunks, &reached, 2u)))
                        missing_closure = 1u;
                }
            }
        }
    }
    local_flags[threadIdx.x] = missing_closure;
    __syncthreads();
    for (std::uint32_t width = blockDim.x / 2u; width != 0u; width >>= 1u) {
        if (threadIdx.x < width) {
            local_flags[threadIdx.x] |= local_flags[threadIdx.x + width];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0u) {
        flags[blockIdx.x] = local_flags[0];
    }
}

__global__ void reduce_or_kernel(std::uint8_t* flags, std::uint32_t count) {
    __shared__ std::uint8_t local_flags[kThreads];
    std::uint8_t value = 0u;
    for (std::uint32_t index = threadIdx.x; index < count; index += blockDim.x) {
        value |= flags[index];
    }
    local_flags[threadIdx.x] = value;
    __syncthreads();
    for (std::uint32_t width = blockDim.x / 2u; width != 0u; width >>= 1u) {
        if (threadIdx.x < width) {
            local_flags[threadIdx.x] |= local_flags[threadIdx.x + width];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0u) {
        flags[0] = local_flags[0];
    }
}

__global__ void equal_flags_kernel(const SiteWord* words, std::uint8_t* flags,
                                   std::uint64_t site_count, SiteWord expected) {
    __shared__ std::uint8_t local_flags[kThreads];
    const std::uint64_t global_thread =
        static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::uint64_t stride = static_cast<std::uint64_t>(gridDim.x) * blockDim.x;
    std::uint8_t different = 0u;
    for (std::uint64_t slot = global_thread; slot < site_count; slot += stride) {
        if (words[slot] != expected) {
            different = 1u;
            break;
        }
    }
    local_flags[threadIdx.x] = different;
    __syncthreads();
    for (std::uint32_t width = blockDim.x / 2u; width != 0u; width >>= 1u) {
        if (threadIdx.x < width) {
            local_flags[threadIdx.x] |= local_flags[threadIdx.x + width];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0u) {
        flags[blockIdx.x] = local_flags[0];
    }
}

}  // namespace

class CudaBcc32Executor::Submission {
public:
    Submission(const CudaBcc32Executor& executor, cudaStream_t stream)
        : executor_(executor), stream_(stream), lock_(executor.submission_mutex_) {
        if (executor_.has_submission_ &&
            executor_.last_submission_stream_ != stream_) {
            check_cuda(cudaStreamWaitEvent(stream_, executor_.completion_event_, 0u),
                       "order BCC32 executor cross-stream submission");
        }
    }

    Submission(const Submission&) = delete;
    Submission& operator=(const Submission&) = delete;

    ~Submission() {
        if (!finished_) {
            const cudaError_t status = cudaEventRecord(executor_.completion_event_, stream_);
            if (status == cudaSuccess) {
                executor_.last_submission_stream_ = stream_;
                executor_.has_submission_ = true;
            }
        }
    }

    void finish() {
        check_cuda(cudaEventRecord(executor_.completion_event_, stream_),
                   "record BCC32 executor submission completion");
        executor_.last_submission_stream_ = stream_;
        executor_.has_submission_ = true;
        finished_ = true;
    }

private:
    const CudaBcc32Executor& executor_;
    cudaStream_t stream_;
    bool finished_ = false;
    std::unique_lock<std::mutex> lock_;
};

std::uint64_t CudaBcc32Executor::checked_site_count(std::uint32_t chunk_count) {
    if (chunk_count == 0u ||
        static_cast<std::uint64_t>(chunk_count) >
            std::numeric_limits<std::uint64_t>::max() / kChunkSites) {
        throw std::overflow_error("chunk count does not form a finite direct aperture");
    }
    return static_cast<std::uint64_t>(chunk_count) * kChunkSites;
}

std::uint64_t CudaBcc32Executor::checked_word_bytes(std::uint32_t chunk_count) {
    const std::uint64_t sites = checked_site_count(chunk_count);
    if (sites > std::numeric_limits<std::uint64_t>::max() / sizeof(SiteWord)) {
        throw std::overflow_error("direct aperture byte count overflows uint64");
    }
    const std::uint64_t bytes = sites * sizeof(SiteWord);
    if (bytes > std::numeric_limits<std::size_t>::max()) {
        throw std::overflow_error("direct aperture byte count exceeds size_t");
    }
    return bytes;
}

CudaBcc32Executor CudaBcc32Executor::production() {
    return CudaBcc32Executor(kProductionChunkSlots, true);
}

CudaBcc32Executor CudaBcc32Executor::testing(std::uint32_t chunk_count) {
    return CudaBcc32Executor(chunk_count, false);
}

CudaBcc32Executor::CudaBcc32Executor(std::uint32_t chunk_count, bool production)
    : chunk_count_(chunk_count),
      site_count_(checked_site_count(chunk_count)),
      aperture_bytes_(checked_word_bytes(chunk_count)),
      carrier_snapshot_bytes_(site_count_) {
    if (carrier_snapshot_bytes_ > std::numeric_limits<std::size_t>::max()) {
        throw std::overflow_error("P carrier snapshot byte count exceeds size_t");
    }
    if (production) {
        std::size_t free_bytes = 0u;
        std::size_t total_bytes = 0u;
        check_cuda(cudaMemGetInfo(&free_bytes, &total_bytes),
                   "query production CUDA memory");
        (void)total_bytes;
        const std::uint64_t required = aperture_bytes_ + carrier_snapshot_bytes_ +
                                       kProductionExecutorHeadroomBytes;
        if (required < aperture_bytes_ ||
            static_cast<std::uint64_t>(free_bytes) < required) {
            throw std::runtime_error(
                "insufficient CUDA memory for 10GB BCC32 aperture, P snapshot, and headroom");
        }
    }
    try {
        check_cuda(cudaMalloc(&words_, static_cast<std::size_t>(aperture_bytes_)),
                   "cudaMalloc direct BCC32 aperture");
        check_cuda(cudaMalloc(&carrier_snapshot_,
                              static_cast<std::size_t>(carrier_snapshot_bytes_)),
                   "cudaMalloc P carrier snapshot");
        check_cuda(cudaEventCreateWithFlags(&completion_event_, cudaEventDisableTiming),
                   "create BCC32 executor completion event");
    } catch (...) {
        release();
        throw;
    }
}

CudaBcc32Executor::CudaBcc32Executor(CudaBcc32Executor&& other) noexcept
    : chunk_count_(0u), site_count_(0ull), aperture_bytes_(0ull),
      carrier_snapshot_bytes_(0ull) {
    std::lock_guard<std::mutex> lock(other.submission_mutex_);
    chunk_count_ = other.chunk_count_;
    site_count_ = other.site_count_;
    aperture_bytes_ = other.aperture_bytes_;
    carrier_snapshot_bytes_ = other.carrier_snapshot_bytes_;
    words_ = std::exchange(other.words_, nullptr);
    carrier_snapshot_ = std::exchange(other.carrier_snapshot_, nullptr);
    completion_event_ = std::exchange(other.completion_event_, nullptr);
    last_submission_stream_ = other.last_submission_stream_;
    has_submission_ = other.has_submission_;
    other.last_submission_stream_ = nullptr;
    other.has_submission_ = false;
}

CudaBcc32Executor& CudaBcc32Executor::operator=(CudaBcc32Executor&& other) noexcept {
    if (this != &other) {
        std::scoped_lock lock(submission_mutex_, other.submission_mutex_);
        release();
        chunk_count_ = other.chunk_count_;
        site_count_ = other.site_count_;
        aperture_bytes_ = other.aperture_bytes_;
        carrier_snapshot_bytes_ = other.carrier_snapshot_bytes_;
        words_ = std::exchange(other.words_, nullptr);
        carrier_snapshot_ = std::exchange(other.carrier_snapshot_, nullptr);
        completion_event_ = std::exchange(other.completion_event_, nullptr);
        last_submission_stream_ = other.last_submission_stream_;
        has_submission_ = other.has_submission_;
        other.last_submission_stream_ = nullptr;
        other.has_submission_ = false;
    }
    return *this;
}

CudaBcc32Executor::~CudaBcc32Executor() {
    release();
}

void CudaBcc32Executor::release() noexcept {
    if (completion_event_ != nullptr) {
        if (has_submission_) {
            (void)cudaEventSynchronize(completion_event_);
        }
        (void)cudaEventDestroy(completion_event_);
        completion_event_ = nullptr;
        last_submission_stream_ = nullptr;
        has_submission_ = false;
    }
    if (carrier_snapshot_ != nullptr) {
        cudaFree(carrier_snapshot_);
        carrier_snapshot_ = nullptr;
    }
    if (bounded_resolution_ != nullptr) {
        cudaFree(bounded_resolution_);
        bounded_resolution_ = nullptr;
        bounded_resolution_capacity_ = 0;
    }
    if (words_ != nullptr) {
        cudaFree(words_);
        words_ = nullptr;
    }
}

void CudaBcc32Executor::initialize_q(cudaStream_t stream) {
    Submission submission(*this, stream);
    fill_q_kernel<<<launch_blocks(site_count_), kThreads, 0, stream>>>(words_, site_count_);
    check_cuda(cudaGetLastError(), "launch BCC32 Q initialization");
    check_cuda(cudaStreamSynchronize(stream), "synchronize BCC32 Q initialization");
    submission.finish();
}

void CudaBcc32Executor::upload_words(std::uint64_t offset,
                                      std::span<const SiteWord> source,
                                      cudaStream_t stream) {
    const std::uint64_t count = static_cast<std::uint64_t>(source.size());
    if (offset > site_count_ || count > site_count_ - offset) {
        throw std::out_of_range("BCC32 upload range is outside the direct aperture");
    }
    if (count == 0ull) {
        return;
    }
    Submission submission(*this, stream);
    check_cuda(cudaMemcpyAsync(words_ + offset, source.data(),
                               static_cast<std::size_t>(count) * sizeof(SiteWord),
                               cudaMemcpyHostToDevice, stream),
               "enqueue BCC32 direct upload");
    submission.finish();
}

void CudaBcc32Executor::download_words(std::uint64_t offset,
                                        std::span<SiteWord> destination,
                                        cudaStream_t stream) const {
    const std::uint64_t count = static_cast<std::uint64_t>(destination.size());
    if (offset > site_count_ || count > site_count_ - offset) {
        throw std::out_of_range("BCC32 download range is outside the direct aperture");
    }
    if (count == 0ull) {
        return;
    }
    Submission submission(*this, stream);
    check_cuda(cudaMemcpyAsync(destination.data(), words_ + offset,
                               static_cast<std::size_t>(count) * sizeof(SiteWord),
                               cudaMemcpyDeviceToHost, stream),
               "enqueue BCC32 direct download");
    submission.finish();
}

void CudaBcc32Executor::validate_topology(const DeviceChunkMap& chunks,
                                          cudaStream_t stream) const {
    if (chunks.slots == nullptr || chunks.chunk_count == 0u ||
        chunks.chunk_count > chunk_count_) {
        throw std::invalid_argument(
            "BCC32 executor topology exceeds its device aperture");
    }
    const std::uint32_t blocks = launch_blocks(chunks.chunk_count);
    topology_flags_kernel<<<blocks, kThreads, 0, stream>>>(carrier_snapshot_, chunks);
    check_cuda(cudaGetLastError(), "launch BCC32 chunk topology validation");
    reduce_or_kernel<<<1u, kThreads, 0, stream>>>(carrier_snapshot_, blocks);
    check_cuda(cudaGetLastError(), "launch BCC32 chunk topology reduction");
    std::uint8_t invalid_topology = 0u;
    check_cuda(cudaMemcpyAsync(&invalid_topology, carrier_snapshot_, sizeof(invalid_topology),
                               cudaMemcpyDeviceToHost, stream),
               "copy BCC32 chunk topology validation");
    check_cuda(cudaStreamSynchronize(stream), "synchronize BCC32 chunk topology validation");
    if (invalid_topology != 0u) {
        throw std::runtime_error("BCC32 chunk map is non-reciprocal or coordinate-inconsistent");
    }
}

void CudaBcc32Executor::validate_core_closure(const DeviceChunkMap& chunks,
                                              std::uint32_t core_count,
                                              cudaStream_t stream) const {
    if (core_count == 0u || core_count > chunks.chunk_count) {
        throw std::invalid_argument("BCC32 page core count is invalid");
    }
    const std::uint32_t blocks = launch_blocks(core_count);
    core_closure_flags_kernel<<<blocks, kThreads, 0, stream>>>(
        carrier_snapshot_, chunks, core_count);
    check_cuda(cudaGetLastError(), "launch BCC32 page-core closure validation");
    reduce_or_kernel<<<1u, kThreads, 0, stream>>>(carrier_snapshot_, blocks);
    check_cuda(cudaGetLastError(), "launch BCC32 page-core closure reduction");
    std::uint8_t missing = 0u;
    check_cuda(cudaMemcpyAsync(&missing,
                               carrier_snapshot_,
                               sizeof(missing),
                               cudaMemcpyDeviceToHost,
                               stream),
               "copy BCC32 page-core closure validation");
    check_cuda(cudaStreamSynchronize(stream),
               "synchronize BCC32 page-core closure validation");
    if (missing != 0u) {
        throw std::runtime_error("BCC32 page omits a core causal-neighbor chunk");
    }
}

void CudaBcc32Executor::validate_superstep_closure(const DeviceChunkMap& chunks,
                                                    cudaStream_t stream) const {
    validate_topology(chunks, stream);

    const std::uint64_t dense_site_count =
        static_cast<std::uint64_t>(chunks.chunk_count) * kChunkSites;
    const std::uint32_t blocks = audit_blocks(dense_site_count);
    closure_flags_kernel<<<blocks, kThreads, 0, stream>>>(
        words_, carrier_snapshot_, dense_site_count, chunks);
    check_cuda(cudaGetLastError(), "launch BCC32 causal closure validation");
    reduce_or_kernel<<<1u, kThreads, 0, stream>>>(carrier_snapshot_, blocks);
    check_cuda(cudaGetLastError(), "launch BCC32 causal closure reduction");
    std::uint8_t missing_closure = 0u;
    check_cuda(cudaMemcpyAsync(&missing_closure, carrier_snapshot_, sizeof(missing_closure),
                               cudaMemcpyDeviceToHost, stream),
               "copy BCC32 causal closure validation");
    check_cuda(cudaStreamSynchronize(stream), "synchronize BCC32 causal closure validation");
    if (missing_closure != 0u) {
        throw std::runtime_error(
            "BCC32 resident causal closure is incomplete for the next superstep");
    }
}

void CudaBcc32Executor::apply_k_site(bool inverse,
                                     std::uint64_t site_count,
                                     cudaStream_t stream) {
    if (site_count == 0u || site_count > site_count_) {
        throw std::invalid_argument("BCC32 K_site site count is invalid");
    }
    k_site_kernel<<<launch_blocks(site_count), kThreads, 0, stream>>>(
        words_, site_count, inverse);
    check_cuda(cudaGetLastError(), "launch BCC32 K_site");
}

void CudaBcc32Executor::apply_k_edge(const DeviceChunkMap& chunks, bool inverse,
                                      std::uint64_t site_count,
                                      cudaStream_t stream) {
    if (site_count == 0u || site_count > site_count_) {
        throw std::invalid_argument("BCC32 K_edge site count is invalid");
    }
    k_edge_parallel_kernel<<<launch_blocks(site_count), kThreads, 0, stream>>>(
        words_, site_count, chunks, inverse);
    check_cuda(cudaGetLastError(), inverse
        ? "launch inverse BCC32 parallel K_edge"
        : "launch BCC32 parallel K_edge");
}

void CudaBcc32Executor::apply_k_carrier_pair_splitter(const DeviceChunkMap& chunks, bool,
                                                      std::uint64_t site_count,
                                                      cudaStream_t stream) {
  if (site_count == 0u || site_count > site_count_) {
    throw std::invalid_argument("BCC32 carrier-pair splitter site count is invalid");
  }
  launch_carrier_pair_splitter_cuda(words_, carrier_snapshot_, site_count,
                                    chunks, stream);
}

void CudaBcc32Executor::apply_carrier_pair_splitter(const DeviceChunkMap& chunks, bool inverse,
                                                    cudaStream_t stream) {
  Submission submission(*this, stream);
  validate_superstep_closure(chunks, stream);
  const std::uint64_t dense_site_count =
      static_cast<std::uint64_t>(chunks.chunk_count) * kChunkSites;
  apply_k_carrier_pair_splitter(chunks, inverse, dense_site_count, stream);
  check_cuda(cudaStreamSynchronize(stream), "synchronize BCC32 carrier-pair splitter");
  submission.finish();
}

void CudaBcc32Executor::apply_k_processive_rearm(const DeviceChunkMap& chunks, bool inverse,
                                                 std::uint64_t site_count, cudaStream_t stream) {
  (void)inverse;
  if (site_count == 0u || site_count > site_count_) {
    throw std::invalid_argument("BCC32 processive rearm site count is invalid");
  }
  launch_processive_rearm_cuda(words_, carrier_snapshot_, site_count, chunks,
                               stream);
}

void CudaBcc32Executor::apply_k_processive_release(const DeviceChunkMap& chunks, bool inverse,
                                                   std::uint64_t site_count, cudaStream_t stream) {
  if (site_count == 0u || site_count > site_count_) {
    throw std::invalid_argument("BCC32 processive release site count is invalid");
  }
  launch_processive_release_cuda(words_, carrier_snapshot_, site_count, chunks,
                                 inverse, stream);
}

void CudaBcc32Executor::apply_processive_rearm(const DeviceChunkMap& chunks, bool inverse,
                                               cudaStream_t stream) {
  Submission submission(*this, stream);
  validate_superstep_closure(chunks, stream);
  const std::uint64_t dense_site_count =
      static_cast<std::uint64_t>(chunks.chunk_count) * kChunkSites;
  apply_k_processive_rearm(chunks, inverse, dense_site_count, stream);
  check_cuda(cudaStreamSynchronize(stream), "synchronize BCC32 processive rearm");
  submission.finish();
}

void CudaBcc32Executor::apply_processive_release(const DeviceChunkMap& chunks, bool inverse,
                                                 cudaStream_t stream) {
  Submission submission(*this, stream);
  validate_superstep_closure(chunks, stream);
  const std::uint64_t dense_site_count =
      static_cast<std::uint64_t>(chunks.chunk_count) * kChunkSites;
  apply_k_processive_release(chunks, inverse, dense_site_count, stream);
  check_cuda(cudaStreamSynchronize(stream), "synchronize BCC32 processive release");
  submission.finish();
}

std::uint8_t* CudaBcc32Executor::ensure_bounded_resolution(
    std::uint64_t active_count, cudaStream_t stream) {
  if (active_count == 0u || active_count > site_count_)
    throw std::invalid_argument("BCC32 bounded resolution extent is invalid");
  if (bounded_resolution_capacity_ < active_count) {
    if (bounded_resolution_ != nullptr)
      check_cuda(cudaFree(bounded_resolution_), "grow bounded resolution");
    check_cuda(cudaMalloc(&bounded_resolution_,
                          static_cast<std::size_t>(active_count)),
               "allocate bounded resolution");
    bounded_resolution_capacity_ = active_count;
  }
  check_cuda(cudaMemsetAsync(bounded_resolution_, 0,
                             static_cast<std::size_t>(active_count), stream),
             "initialize bounded resolution");
  return bounded_resolution_;
}

void CudaBcc32Executor::apply_active_processive_release(
    const DeviceChunkMap& chunks, bool inverse,
    const std::uint64_t* active_slots, std::uint64_t active_count,
    cudaStream_t stream) {
  if (active_count == 0u) return;
  apply_active_processive_release(chunks, inverse, active_slots, active_count,
                                  ensure_bounded_resolution(active_count,
                                                            stream),
                                  stream);
}

void CudaBcc32Executor::apply_active_processive_release(
    const DeviceChunkMap& chunks, bool inverse,
    const std::uint64_t* active_slots, std::uint64_t active_count,
    std::uint8_t* resolved,
    cudaStream_t stream) {
  (void)inverse;
  if (active_count == 0u) return;
  if (active_slots == nullptr || resolved == nullptr || active_count > site_count_)
    throw std::invalid_argument(
        "BCC32 active processive release support is invalid");
  Submission submission(*this, stream);
  launch_active_processive_release_cuda(
      words_, carrier_snapshot_, resolved, site_count_, chunks, active_slots,
      active_count, inverse, stream);
  submission.finish();
}

void CudaBcc32Executor::apply_k_developmental_append(
    const DeviceChunkMap& chunks, bool inverse, std::uint64_t site_count,
    const std::uint64_t* active_slots, std::uint64_t active_count,
    cudaStream_t stream) {
  if (site_count == 0u || site_count > site_count_)
    throw std::invalid_argument("BCC32 developmental append site count is invalid");
  if (active_slots != nullptr && active_count > site_count)
    throw std::invalid_argument("BCC32 developmental append active support is invalid");
  launch_developmental_append_cuda(words_, carrier_snapshot_, site_count,
                                   chunks, inverse, active_slots, active_count,
                                   stream);
}

void CudaBcc32Executor::apply_developmental_append(
    const DeviceChunkMap& chunks, bool inverse, cudaStream_t stream) {
  Submission submission(*this, stream);
  validate_superstep_closure(chunks, stream);
  const std::uint64_t dense_site_count =
      static_cast<std::uint64_t>(chunks.chunk_count) * kChunkSites;
  apply_k_developmental_append(chunks, inverse, dense_site_count, nullptr, 0u,
                               stream);
  check_cuda(cudaStreamSynchronize(stream),
             "synchronize BCC32 developmental append");
  submission.finish();
}

void CudaBcc32Executor::apply_active_developmental_append(
    const DeviceChunkMap& chunks, bool inverse,
    const std::uint64_t* active_slots, std::uint64_t active_count,
    cudaStream_t stream) {
  if (active_count == 0u) return;
  if (active_slots == nullptr || active_count > site_count_)
    throw std::invalid_argument(
        "BCC32 active developmental append support is invalid");
  Submission submission(*this, stream);
  apply_k_developmental_append(chunks, inverse, site_count_, active_slots,
                               active_count, stream);
  submission.finish();
}

void CudaBcc32Executor::apply_k_developmental_learned_receptor(
    const DeviceChunkMap& chunks, bool inverse, std::uint64_t site_count,
    const std::uint64_t* active_slots, std::uint64_t active_count,
    cudaStream_t stream) {
  if (site_count == 0u || site_count > site_count_)
    throw std::invalid_argument(
        "BCC32 developmental learned receptor site count is invalid");
  if (active_slots != nullptr && active_count > site_count)
    throw std::invalid_argument(
        "BCC32 developmental learned receptor active support is invalid");
  launch_developmental_learned_receptor_cuda(
      words_, carrier_snapshot_, site_count, chunks, inverse, active_slots,
      active_count, stream);
}

void CudaBcc32Executor::apply_developmental_learned_receptor(
    const DeviceChunkMap& chunks, bool inverse, cudaStream_t stream) {
  Submission submission(*this, stream);
  validate_superstep_closure(chunks, stream);
  const std::uint64_t dense_site_count =
      static_cast<std::uint64_t>(chunks.chunk_count) * kChunkSites;
  apply_k_developmental_learned_receptor(
      chunks, inverse, dense_site_count, nullptr, 0u, stream);
  check_cuda(cudaStreamSynchronize(stream),
             "synchronize BCC32 developmental learned receptor");
  submission.finish();
}

void CudaBcc32Executor::apply_active_developmental_learned_receptor(
    const DeviceChunkMap& chunks, bool inverse,
    const std::uint64_t* active_slots, std::uint64_t active_count,
    cudaStream_t stream) {
  if (active_count == 0u) return;
  if (active_slots == nullptr || active_count > site_count_)
    throw std::invalid_argument(
        "BCC32 active developmental learned receptor support is invalid");
  Submission submission(*this, stream);
  apply_k_developmental_learned_receptor(
      chunks, inverse, site_count_, active_slots, active_count, stream);
  submission.finish();
}

void CudaBcc32Executor::apply_k_developmental_credit_service(
    const DeviceChunkMap& chunks, bool inverse, std::uint64_t site_count,
    const std::uint64_t* active_slots, std::uint64_t active_count,
    cudaStream_t stream) {
  if (site_count == 0u || site_count > site_count_)
    throw std::invalid_argument(
        "BCC32 developmental credit-service site count is invalid");
  if (active_slots != nullptr && active_count > site_count)
    throw std::invalid_argument(
        "BCC32 developmental credit-service active support is invalid");
  launch_developmental_credit_service_cuda(
      words_, carrier_snapshot_, site_count, chunks, inverse, active_slots,
      active_count, stream);
}

void CudaBcc32Executor::apply_developmental_credit_service(
    const DeviceChunkMap& chunks, bool inverse, cudaStream_t stream) {
  Submission submission(*this, stream);
  validate_superstep_closure(chunks, stream);
  const std::uint64_t dense_site_count =
      static_cast<std::uint64_t>(chunks.chunk_count) * kChunkSites;
  apply_k_developmental_credit_service(
      chunks, inverse, dense_site_count, nullptr, 0u, stream);
  check_cuda(cudaStreamSynchronize(stream),
             "synchronize BCC32 developmental credit service");
  submission.finish();
}

void CudaBcc32Executor::apply_active_developmental_credit_service(
    const DeviceChunkMap& chunks, bool inverse,
    const std::uint64_t* active_slots, std::uint64_t active_count,
    cudaStream_t stream) {
  if (active_count == 0u) return;
  if (active_slots == nullptr || active_count > site_count_)
    throw std::invalid_argument(
        "BCC32 active developmental credit-service support is invalid");
  Submission submission(*this, stream);
  apply_k_developmental_credit_service(
      chunks, inverse, site_count_, active_slots, active_count, stream);
  submission.finish();
}

void CudaBcc32Executor::apply_k_carrier_corner(const DeviceChunkMap& chunks, bool inverse,
                                               std::uint64_t site_count, cudaStream_t stream) {
  if (site_count == 0u || site_count > site_count_) {
    throw std::invalid_argument("BCC32 carrier corner site count is invalid");
  }
  launch_carrier_corner_cuda(words_, carrier_snapshot_, site_count, chunks,
                             inverse, stream);
}

void CudaBcc32Executor::apply_carrier_corner(const DeviceChunkMap& chunks, bool inverse,
                                             cudaStream_t stream) {
  Submission submission(*this, stream);
  validate_superstep_closure(chunks, stream);
  const std::uint64_t dense_site_count =
      static_cast<std::uint64_t>(chunks.chunk_count) * kChunkSites;
  apply_k_carrier_corner(chunks, inverse, dense_site_count, stream);
  check_cuda(cudaStreamSynchronize(stream), "synchronize BCC32 carrier corner");
  submission.finish();
}

void CudaBcc32Executor::apply_k_eligibility_residual_junction(
    const DeviceChunkMap& chunks, bool inverse, std::uint64_t site_count,
    const std::uint64_t* active_slots, std::uint64_t active_count,
    cudaStream_t stream) {
  (void)inverse;
  if (site_count == 0u || site_count > site_count_ ||
      (active_slots != nullptr && active_count > site_count))
    throw std::invalid_argument(
        "BCC32 eligibility residual junction support is invalid");
  launch_eligibility_residual_junction_cuda(
      words_, carrier_snapshot_, site_count, chunks, active_slots,
      active_count, stream);
}

void CudaBcc32Executor::apply_eligibility_residual_junction(
    const DeviceChunkMap& chunks, bool inverse, cudaStream_t stream) {
  Submission submission(*this, stream);
  validate_superstep_closure(chunks, stream);
  const std::uint64_t dense_site_count =
      static_cast<std::uint64_t>(chunks.chunk_count) * kChunkSites;
  apply_k_eligibility_residual_junction(
      chunks, inverse, dense_site_count, nullptr, 0u, stream);
  check_cuda(cudaStreamSynchronize(stream),
             "synchronize BCC32 eligibility residual junction");
  submission.finish();
}

void CudaBcc32Executor::apply_active_eligibility_residual_junction(
    const DeviceChunkMap& chunks, bool inverse,
    const std::uint64_t* active_slots, std::uint64_t active_count,
    cudaStream_t stream) {
  if (active_count == 0u) return;
  if (active_slots == nullptr || active_count > site_count_)
    throw std::invalid_argument(
        "BCC32 active eligibility residual junction support is invalid");
  Submission submission(*this, stream);
  apply_k_eligibility_residual_junction(
      chunks, inverse, site_count_, active_slots, active_count, stream);
  submission.finish();
}

void CudaBcc32Executor::apply_active_macro_factors(
    const DeviceChunkMap& chunks, bool inverse,
    const std::uint64_t* active_slots, std::uint64_t active_count,
    cudaStream_t stream) {
  if (active_count == 0u) return;
  apply_active_macro_factors(chunks, inverse, active_slots, active_count,
                             ensure_bounded_resolution(active_count, stream),
                             stream);
}

void CudaBcc32Executor::apply_active_macro_factors(
    const DeviceChunkMap& chunks, bool inverse,
    const std::uint64_t* active_slots, std::uint64_t active_count,
    std::uint8_t* resolved,
    cudaStream_t stream) {
  if (active_count == 0u) return;
  if (active_slots == nullptr || resolved == nullptr || active_count > site_count_) {
    throw std::invalid_argument(
        "BCC32 active macro support is invalid");
  }
  Submission submission(*this, stream);
  launch_active_spatial_macros_cuda(
      words_, carrier_snapshot_, resolved, site_count_, chunks, inverse,
      active_slots, active_count, stream);
  submission.finish();
}

void CudaBcc32Executor::apply_active_prediction_residual_route_toggle(
    const DeviceChunkMap& chunks, bool inverse,
    const std::uint64_t* active_slots, std::uint64_t active_count,
    cudaStream_t stream) {
  if (active_count == 0u) return;
  apply_active_prediction_residual_route_toggle(
      chunks, inverse, active_slots, active_count,
      ensure_bounded_resolution(active_count, stream), stream);
}

void CudaBcc32Executor::apply_active_prediction_residual_route_toggle(
    const DeviceChunkMap& chunks, bool inverse,
    const std::uint64_t* active_slots, std::uint64_t active_count,
    std::uint8_t* resolved,
    cudaStream_t stream) {
  if (active_count == 0u) return;
  if (active_slots == nullptr || resolved == nullptr || active_count > site_count_)
    throw std::invalid_argument("BCC32 active prediction residual support is invalid");
  Submission submission(*this, stream);
  launch_prediction_residual_route_toggle_cuda(
      words_, carrier_snapshot_, resolved, site_count_, chunks, inverse,
      active_slots, active_count, stream);
  submission.finish();
}

void CudaBcc32Executor::apply_prediction_residual_route_toggle(
    const DeviceChunkMap& chunks, bool inverse, cudaStream_t stream) {
  Submission submission(*this, stream);
  validate_superstep_closure(chunks, stream);
  const std::uint64_t dense_site_count =
      static_cast<std::uint64_t>(chunks.chunk_count) * kChunkSites;
  launch_prediction_residual_route_toggle_cuda(
      words_, carrier_snapshot_, nullptr, dense_site_count, chunks, inverse,
      nullptr, 0u, stream);
  check_cuda(cudaStreamSynchronize(stream),
             "synchronize BCC32 prediction residual route toggle");
  submission.finish();
}

void CudaBcc32Executor::graph_safe_prepare_active_window(
    const std::uint64_t* source_slots, const std::uint32_t* source_count,
    std::uint64_t* window_slots, std::uint32_t window_capacity,
    cudaStream_t stream) const {
    if (source_slots == nullptr || source_count == nullptr ||
        window_slots == nullptr || window_capacity == 0u) {
        throw std::invalid_argument(
            "BCC32 graph-safe active window extent is invalid");
    }
    // The active window is a bounded workspace, not a second physical
    // aperture. Its entries are validated against site_count_ by the device
    // kernels; requiring workspace_capacity <= site_count_ here rejects a
    // lawful power-of-two workspace for a smaller finite aperture.
    prepare_active_window_kernel<<<launch_blocks(window_capacity), kThreads, 0,
                                   stream>>>(
        source_slots, source_count, window_slots, window_capacity);
    check_cuda(cudaGetLastError(), "launch BCC32 graph-safe active window");
}

void CudaBcc32Executor::graph_safe_apply_active_factor(
    const DeviceChunkMap& chunks, LawFactor factor, bool inverse,
    DeviceActiveSupportWindow window, cudaStream_t stream) {
    if (window.slots == nullptr || window.count == nullptr ||
        window.resolved == nullptr || window.capacity == 0u) {
        throw std::invalid_argument(
            "BCC32 graph-safe active factor window is invalid");
    }
    switch (factor) {
        case LawFactor::developmental_learned_receptor:
            launch_developmental_learned_receptor_graph_safe_cuda(
                words_, carrier_snapshot_, site_count_, chunks, inverse,
                window.slots, window.count, window.capacity, stream);
            return;
        case LawFactor::developmental_append:
            launch_developmental_append_graph_safe_cuda(
                words_, carrier_snapshot_, site_count_, chunks, inverse,
                window.slots, window.count, window.capacity, stream);
            return;
        case LawFactor::eligibility_residual_junction:
            launch_eligibility_residual_junction_graph_safe_cuda(
                words_, carrier_snapshot_, site_count_, chunks, window.slots,
                window.count, window.capacity, stream);
            return;
        case LawFactor::site:
            launch_active_site_graph_safe(
                words_, window.slots, window.count, window.capacity, inverse,
                stream);
            return;
        case LawFactor::edge:
            launch_active_edge_graph_safe(words_, window.slots,
                                          window.count, window.capacity, chunks,
                                          inverse, stream);
            return;
        case LawFactor::prediction_residual_route_toggle:
            launch_prediction_residual_route_toggle_graph_safe_cuda(
                words_, carrier_snapshot_, window.resolved, site_count_, chunks,
                inverse,
                window.slots, window.count, window.capacity, stream);
            return;
        case LawFactor::developmental_credit_service:
            launch_developmental_credit_service_graph_safe_cuda(
                words_, carrier_snapshot_, site_count_, chunks, inverse,
                window.slots, window.count, window.capacity, stream);
            return;
        case LawFactor::stream:
            launch_active_stream_graph_safe(
                words_, window.slots, carrier_snapshot_, window.count,
                window.capacity, chunks, inverse, stream);
            return;
        case LawFactor::carrier_pair_splitter:
        case LawFactor::carrier_corner:
        case LawFactor::processive_rearm:
        case LawFactor::processive_release:
            throw std::logic_error(
                "BCC32 graph-safe macro factors require grouped dispatch");
    }
    throw std::invalid_argument("BCC32 graph-safe law factor is invalid");
}

void CudaBcc32Executor::graph_safe_apply_active_macro_factors(
    const DeviceChunkMap& chunks, bool inverse,
    DeviceActiveSupportWindow window, cudaStream_t stream) {
    if (window.slots == nullptr || window.count == nullptr ||
        window.resolved == nullptr || window.capacity == 0u) {
        throw std::invalid_argument(
            "BCC32 graph-safe macro window is invalid");
    }
    launch_active_spatial_macros_graph_safe_cuda(
        words_, carrier_snapshot_, window.resolved, site_count_, chunks, inverse,
        window.slots, window.count, window.capacity, stream);
}

void CudaBcc32Executor::stream_p(const DeviceChunkMap& chunks, bool inverse,
                                 std::uint64_t site_count,
                                 cudaStream_t stream) {
    if (site_count == 0u || site_count > site_count_) {
        throw std::invalid_argument("BCC32 P stream site count is invalid");
    }
    snapshot_carriers_kernel<<<launch_blocks(site_count), kThreads, 0, stream>>>(
        words_, carrier_snapshot_, site_count);
    check_cuda(cudaGetLastError(), "launch BCC32 simultaneous P snapshot");
    gather_carriers_kernel<<<launch_blocks(site_count), kThreads, 0, stream>>>(
        words_, carrier_snapshot_, site_count, chunks, inverse);
    check_cuda(cudaGetLastError(), "launch BCC32 fused P gather");
}

void CudaBcc32Executor::apply_superstep(const DeviceChunkMap& chunks,
                                         cudaStream_t stream) {
    Submission submission(*this, stream);
    // Genesis authority above refuses once this is nonzero. Incremented here so
    // "before the first tick" is a measured fact rather than a caller promise.
    ++supersteps_;
    validate_superstep_closure(chunks, stream);
    // ⛔ THIS IS THE DENSE WHOLE-LATTICE SUPERSTEP AND THAT IS ITS PURPOSE, NOT
    // A MISSING FRONTIER WIRING. Every factor below is deliberately passed
    // `nullptr, 0u` for active_slots: this entry point exists so
    // bcc32_cuda_contract can compare CUDA against the CPU ReferenceLattice
    // word-for-word over the entire aperture and prove F o F-inverse restores
    // every resident word. Narrowing it to the live frontier would make that
    // contract verify only the frontier -- i.e. delete the measurement.
    //
    // The frontier path is NOT missing and does not belong here: the adult
    // advances the world through CudaBcc32ActiveSupport::apply_superstep, which
    // never calls this function and instead calls the apply_active_* family at
    // launch_blocks(active_count). Carrier corner in particular is covered
    // there by apply_active_macro_factors -> launch_active_spatial_macros_cuda
    // -> active_corner_{match,collision,apply}_kernel, batched with
    // carrier_pair_splitter/processive_rearm/processive_release. Do not read
    // "no standalone apply_active_carrier_corner" as "no frontier path"; the
    // four spatial macros are frontier-wired as one batch, by design.
    const std::uint64_t dense_site_count =
        static_cast<std::uint64_t>(chunks.chunk_count) * kChunkSites;
    for (std::uint32_t index = 0u; index < kForwardFactorCount; ++index) {
        switch (forward_factor(index)) {
            case LawFactor::developmental_append:
                apply_k_developmental_append(chunks, false, dense_site_count,
                                             nullptr, 0u, stream);
                break;
            case LawFactor::developmental_learned_receptor:
                apply_k_developmental_learned_receptor(
                    chunks, false, dense_site_count, nullptr, 0u, stream);
                break;
            case LawFactor::eligibility_residual_junction:
                apply_k_eligibility_residual_junction(
                    chunks, false, dense_site_count, nullptr, 0u, stream);
                break;
            case LawFactor::site:
                apply_k_site(false, dense_site_count, stream);
                break;
            case LawFactor::edge:
                apply_k_edge(chunks, false, dense_site_count, stream);
                break;
            case LawFactor::carrier_pair_splitter:
        apply_k_carrier_pair_splitter(chunks, false, dense_site_count, stream);
        break;
      case LawFactor::processive_rearm:
        apply_k_processive_rearm(chunks, false, dense_site_count, stream);
        break;
      case LawFactor::processive_release:
        apply_k_processive_release(chunks, false, dense_site_count, stream);
        break;
            case LawFactor::carrier_corner:
                apply_k_carrier_corner(chunks, false, dense_site_count, stream);
                break;
            case LawFactor::prediction_residual_route_toggle:
                launch_prediction_residual_route_toggle_cuda(
                    words_, carrier_snapshot_, nullptr, dense_site_count, chunks, false,
                    nullptr, 0u, stream);
                break;
            case LawFactor::developmental_credit_service:
                apply_k_developmental_credit_service(
                    chunks, false, dense_site_count, nullptr, 0u, stream);
                break;
            case LawFactor::stream:
        stream_p(chunks, false, dense_site_count, stream);
        break;
        }
    }
    check_cuda(cudaStreamSynchronize(stream), "synchronize BCC32 forward superstep");
    submission.finish();
}

void CudaBcc32Executor::apply_superstep_inverse(const DeviceChunkMap& chunks,
                                                 cudaStream_t stream) {
    Submission submission(*this, stream);
    validate_superstep_closure(chunks, stream);
    const std::uint64_t dense_site_count =
        static_cast<std::uint64_t>(chunks.chunk_count) * kChunkSites;
    for (std::uint32_t index = kForwardFactorCount; index > 0u; --index) {
        switch (forward_factor(index - 1u)) {
            case LawFactor::developmental_append:
                apply_k_developmental_append(chunks, true, dense_site_count,
                                             nullptr, 0u, stream);
                break;
            case LawFactor::developmental_learned_receptor:
                apply_k_developmental_learned_receptor(
                    chunks, true, dense_site_count, nullptr, 0u, stream);
                break;
            case LawFactor::eligibility_residual_junction:
                apply_k_eligibility_residual_junction(
                    chunks, true, dense_site_count, nullptr, 0u, stream);
                break;
            case LawFactor::site:
                apply_k_site(true, dense_site_count, stream);
                break;
            case LawFactor::edge:
                apply_k_edge(chunks, true, dense_site_count, stream);
        break;
      case LawFactor::carrier_pair_splitter:
        apply_k_carrier_pair_splitter(chunks, true, dense_site_count, stream);
        break;
      case LawFactor::processive_rearm:
        apply_k_processive_rearm(chunks, true, dense_site_count, stream);
        break;
      case LawFactor::processive_release:
        apply_k_processive_release(chunks, true, dense_site_count, stream);
        break;
      case LawFactor::carrier_corner:
        apply_k_carrier_corner(chunks, true, dense_site_count, stream); break;
      case LawFactor::prediction_residual_route_toggle:
        launch_prediction_residual_route_toggle_cuda(
            words_, carrier_snapshot_, nullptr, dense_site_count, chunks, true,
            nullptr, 0u, stream);
        break;
      case LawFactor::developmental_credit_service:
        apply_k_developmental_credit_service(
            chunks, true, dense_site_count, nullptr, 0u, stream);
        break;
            case LawFactor::stream:
                stream_p(chunks, true, dense_site_count, stream);
                break;
        }
    }
    check_cuda(cudaStreamSynchronize(stream), "synchronize BCC32 inverse superstep");
    submission.finish();
}

void CudaBcc32Executor::apply_factor_window(const DeviceChunkMap& chunks,
                                            LawFactor factor,
                                            bool inverse,
                                            std::uint32_t core_count,
                                            std::uint32_t input_count,
                                            cudaStream_t stream) {
    Submission submission(*this, stream);
    validate_topology(chunks, stream);
    if (factor != LawFactor::site) {
        validate_core_closure(chunks, core_count, stream);
    } else if (core_count == 0u || core_count > chunks.chunk_count) {
        throw std::invalid_argument("BCC32 page core count is invalid");
    }
    if (input_count < core_count || input_count > chunks.chunk_count ||
        core_count > chunks.chunk_count - input_count) {
        throw std::invalid_argument("BCC32 page input/output slots overlap or overflow");
    }
    const std::uint64_t core_sites =
        static_cast<std::uint64_t>(core_count) * kChunkSites;
    SiteWord* output = words_ + static_cast<std::uint64_t>(input_count) * kChunkSites;
    switch (factor) {
        case LawFactor::developmental_append: {
            const std::uint64_t input_sites =
                static_cast<std::uint64_t>(input_count) * kChunkSites;
            apply_k_developmental_append(chunks, inverse, input_sites,
                                         nullptr, 0u, stream);
            copy_core_kernel<<<launch_blocks(core_sites), kThreads, 0, stream>>>(
                words_, output, core_sites);
            break;
        }
        case LawFactor::developmental_learned_receptor: {
            const std::uint64_t input_sites =
                static_cast<std::uint64_t>(input_count) * kChunkSites;
            apply_k_developmental_learned_receptor(
                chunks, inverse, input_sites, nullptr, 0u, stream);
            copy_core_kernel<<<launch_blocks(core_sites), kThreads, 0, stream>>>(
                words_, output, core_sites);
            break;
        }
        case LawFactor::eligibility_residual_junction: {
            const std::uint64_t input_sites =
                static_cast<std::uint64_t>(input_count) * kChunkSites;
            apply_k_eligibility_residual_junction(
                chunks, inverse, input_sites, nullptr, 0u, stream);
            copy_core_kernel<<<launch_blocks(core_sites), kThreads, 0, stream>>>(
                words_, output, core_sites);
            break;
        }
        case LawFactor::site:
            k_site_page_kernel<<<launch_blocks(core_sites), kThreads, 0, stream>>>(
                words_, output, core_sites, inverse);
            break;
        case LawFactor::edge:
            k_edge_page_kernel<<<launch_blocks(core_sites), kThreads, 0, stream>>>(
                words_, output, core_sites, chunks, inverse);
            break;
    case LawFactor::carrier_pair_splitter: {
      const std::uint64_t input_sites = static_cast<std::uint64_t>(input_count) * kChunkSites;
      apply_k_carrier_pair_splitter(chunks, inverse, input_sites, stream);
      copy_core_kernel<<<launch_blocks(core_sites), kThreads, 0, stream>>>(words_, output,
                                                                           core_sites);
      break;
    }
    case LawFactor::processive_rearm: {
      const std::uint64_t input_sites = static_cast<std::uint64_t>(input_count) * kChunkSites;
      apply_k_processive_rearm(chunks, inverse, input_sites, stream);
      copy_core_kernel<<<launch_blocks(core_sites), kThreads, 0, stream>>>(words_, output,
                                                                           core_sites);
      break;
    }
    case LawFactor::processive_release: {
      const std::uint64_t input_sites = static_cast<std::uint64_t>(input_count) * kChunkSites;
      apply_k_processive_release(chunks, inverse, input_sites, stream);
      copy_core_kernel<<<launch_blocks(core_sites), kThreads, 0, stream>>>(words_, output,
                                                                           core_sites);
      break;
    }
    case LawFactor::carrier_corner: {
      const std::uint64_t input_sites = static_cast<std::uint64_t>(input_count) * kChunkSites;
      apply_k_carrier_corner(chunks, inverse, input_sites, stream);
      copy_core_kernel<<<launch_blocks(core_sites), kThreads, 0, stream>>>(words_, output,
                                                                           core_sites);
      break;
    }
    case LawFactor::prediction_residual_route_toggle: {
      const std::uint64_t input_sites = static_cast<std::uint64_t>(input_count) * kChunkSites;
      launch_prediction_residual_route_toggle_cuda(
          words_, carrier_snapshot_, nullptr, input_sites, chunks, inverse,
          nullptr, 0u, stream);
      copy_core_kernel<<<launch_blocks(core_sites), kThreads, 0, stream>>>(words_, output,
                                                                           core_sites);
      break;
    }
    case LawFactor::developmental_credit_service: {
      const std::uint64_t input_sites =
          static_cast<std::uint64_t>(input_count) * kChunkSites;
      apply_k_developmental_credit_service(
          chunks, inverse, input_sites, nullptr, 0u, stream);
      copy_core_kernel<<<launch_blocks(core_sites), kThreads, 0, stream>>>(
          words_, output, core_sites);
      break;
    }
    case LawFactor::stream:
            stream_page_kernel<<<launch_blocks(core_sites), kThreads, 0, stream>>>(
                words_, output, core_sites, chunks, inverse);
            break;
        default: throw std::invalid_argument("BCC32 page law factor is invalid");
    }
    check_cuda(cudaGetLastError(), "launch BCC32 immutable-input page factor");
    check_cuda(cudaStreamSynchronize(stream), "synchronize BCC32 page law factor");
    submission.finish();
}

void CudaBcc32Executor::write_word(std::uint64_t slot, SiteWord value,
                                    cudaStream_t stream) {
    if (slot >= site_count_) {
        throw std::out_of_range("BCC32 write slot is outside the direct aperture");
    }
    Submission submission(*this, stream);
    check_cuda(cudaMemcpyAsync(words_ + slot, &value, sizeof(value), cudaMemcpyHostToDevice,
                               stream), "write BCC32 aperture word");
    check_cuda(cudaStreamSynchronize(stream), "synchronize BCC32 aperture word write");
    submission.finish();
}

SiteWord CudaBcc32Executor::read_word(std::uint64_t slot, cudaStream_t stream) const {
    if (slot >= site_count_) {
        throw std::out_of_range("BCC32 read slot is outside the direct aperture");
    }
    Submission submission(*this, stream);
    SiteWord value = 0u;
    check_cuda(cudaMemcpyAsync(&value, words_ + slot, sizeof(value), cudaMemcpyDeviceToHost,
                               stream), "read BCC32 aperture word");
    check_cuda(cudaStreamSynchronize(stream), "synchronize BCC32 aperture word read");
    submission.finish();
    return value;
}

bool CudaBcc32Executor::all_words_equal(SiteWord value, cudaStream_t stream) {
    Submission submission(*this, stream);
    const std::uint32_t blocks = audit_blocks(carrier_snapshot_bytes_);
    equal_flags_kernel<<<blocks, kThreads, 0, stream>>>(
        words_, carrier_snapshot_, site_count_, value);
    check_cuda(cudaGetLastError(), "launch BCC32 aperture equality audit");
    reduce_or_kernel<<<1u, kThreads, 0, stream>>>(carrier_snapshot_, blocks);
    check_cuda(cudaGetLastError(), "launch BCC32 aperture equality reduction");
    std::uint8_t different = 0u;
    check_cuda(cudaMemcpyAsync(&different, carrier_snapshot_, sizeof(different),
                               cudaMemcpyDeviceToHost, stream),
               "copy BCC32 aperture equality audit");
    check_cuda(cudaStreamSynchronize(stream), "synchronize BCC32 aperture equality audit");
    submission.finish();
    return different == 0u;
}

}  // namespace substrate::bcc32
