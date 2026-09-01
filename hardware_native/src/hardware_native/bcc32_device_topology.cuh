#pragma once

#include "bcc32_cuda_executor.cuh"

#include <cstdint>

namespace substrate::bcc32::device_topology {

// Sparse production topology view.  The only resident adjacency metadata is
// the executor-owned chunk map; an eight-neighbor table is resolved from the
// 100^3 chunk page and never materialised per aperture site.
struct View {
  DeviceChunkMap chunks{};
};

[[nodiscard]] __host__ __device__ inline std::uint32_t encode_local(
    std::uint32_t x, std::uint32_t y, std::uint32_t z) {
  return (x * kChunkEdge + y) * kChunkEdge + z;
}

__host__ __device__ inline void decode_local(std::uint32_t local,
                                             std::uint32_t* x,
                                             std::uint32_t* y,
                                             std::uint32_t* z) {
  *x = local / (kChunkEdge * kChunkEdge);
  const std::uint32_t remainder = local % (kChunkEdge * kChunkEdge);
  *y = remainder / kChunkEdge;
  *z = remainder % kChunkEdge;
}

[[nodiscard]] __host__ __device__ inline bool hop_chunk(
    const DeviceChunkMap& chunks, std::uint32_t* chunk,
    std::uint32_t direction) {
  if (chunks.slots == nullptr || *chunk >= chunks.chunk_count) return false;
  const std::int32_t next = chunks.slots[*chunk].bcc_neighbors[direction];
  if (next < 0 || static_cast<std::uint32_t>(next) >= chunks.chunk_count)
    return false;
  *chunk = static_cast<std::uint32_t>(next);
  return true;
}

[[nodiscard]] __host__ __device__ inline bool neighbor_slot(
    const View& topology, std::uint64_t source, std::uint32_t direction,
    std::uint64_t* destination) {
  const DeviceChunkMap& chunks = topology.chunks;
  if (chunks.slots == nullptr || chunks.chunk_count == 0u) return false;
  std::uint32_t chunk =
      static_cast<std::uint32_t>(source / static_cast<std::uint64_t>(kChunkSites));
  if (chunk >= chunks.chunk_count) return false;
  const std::uint32_t local =
      static_cast<std::uint32_t>(source % static_cast<std::uint64_t>(kChunkSites));
  std::uint32_t local_x = 0u;
  std::uint32_t local_y = 0u;
  std::uint32_t local_z = 0u;
  decode_local(local, &local_x, &local_y, &local_z);
  const Int3 offset = direction_offset(static_cast<Direction>(direction));
  std::int32_t x = static_cast<std::int32_t>(local_x) + offset.x;
  std::int32_t y = static_cast<std::int32_t>(local_y) + offset.y;
  std::int32_t z = static_cast<std::int32_t>(local_z) + offset.z;

  const bool all_negative = x < 0 && y < 0 && z < 0;
  const bool all_positive =
      x >= static_cast<std::int32_t>(kChunkEdge) &&
      y >= static_cast<std::int32_t>(kChunkEdge) &&
      z >= static_cast<std::int32_t>(kChunkEdge);
  if (direction == 3u && all_negative) {
    if (!hop_chunk(chunks, &chunk, 3u)) return false;
    x += static_cast<std::int32_t>(kChunkEdge);
    y += static_cast<std::int32_t>(kChunkEdge);
    z += static_cast<std::int32_t>(kChunkEdge);
  } else if (direction == 7u && all_positive) {
    if (!hop_chunk(chunks, &chunk, 7u)) return false;
    x -= static_cast<std::int32_t>(kChunkEdge);
    y -= static_cast<std::int32_t>(kChunkEdge);
    z -= static_cast<std::int32_t>(kChunkEdge);
  } else {
    std::int32_t* coordinates[3] = {&x, &y, &z};
    for (std::uint32_t axis = 0u; axis < 3u; ++axis) {
      if (*coordinates[axis] < 0) {
        if (!hop_chunk(chunks, &chunk, axis + 4u)) return false;
        *coordinates[axis] += static_cast<std::int32_t>(kChunkEdge);
      } else if (*coordinates[axis] >= static_cast<std::int32_t>(kChunkEdge)) {
        if (!hop_chunk(chunks, &chunk, axis)) return false;
        *coordinates[axis] -= static_cast<std::int32_t>(kChunkEdge);
      }
    }
  }
  if (x < 0 || y < 0 || z < 0 ||
      x >= static_cast<std::int32_t>(kChunkEdge) ||
      y >= static_cast<std::int32_t>(kChunkEdge) ||
      z >= static_cast<std::int32_t>(kChunkEdge))
    return false;
  *destination = static_cast<std::uint64_t>(chunk) * kChunkSites +
                 encode_local(static_cast<std::uint32_t>(x),
                              static_cast<std::uint32_t>(y),
                              static_cast<std::uint32_t>(z));
  return true;
}

[[nodiscard]] __host__ __device__ inline bool walk(
    const View& topology, std::uint64_t start, std::uint32_t direction,
    std::uint32_t steps, std::uint64_t* destination) {
  std::uint64_t current = start;
  for (std::uint32_t step = 0u; step < steps; ++step)
    if (!neighbor_slot(topology, current, direction, &current)) return false;
  *destination = current;
  return true;
}

}  // namespace substrate::bcc32::device_topology
