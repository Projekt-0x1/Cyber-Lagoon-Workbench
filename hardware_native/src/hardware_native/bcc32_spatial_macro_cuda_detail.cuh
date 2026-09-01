// Private CUDA helpers shared by extracted spatial-law translation units.
// This file is included inside an anonymous namespace.
constexpr std::uint32_t kThreads = 256u;
constexpr std::int64_t kInt64Min = (-9'223'372'036'854'775'807LL - 1LL);
constexpr std::int64_t kInt64Max = 9'223'372'036'854'775'807LL;


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



__device__ void processive_permutation(std::uint32_t marker, std::uint32_t path,
                                       std::uint32_t waste, std::uint32_t permutation[4]) {
  permutation[0u] = marker;
  permutation[1u] = path;
  permutation[2u] = waste;
  for (std::uint32_t basis = 0u; basis < 4u; ++basis) {
    if (basis != marker && basis != path && basis != waste) {
      permutation[3u] = basis;
    }
  }
}


[[nodiscard]] __device__ bool junction_walk(
    const DeviceChunkMap& chunks, std::uint64_t start,
    std::uint32_t direction, std::uint32_t count,
    std::uint64_t* destination) {
  std::uint64_t current = start;
  for (std::uint32_t step = 0u; step < count; ++step) {
    if (!neighbor_slot(chunks, current, direction, &current)) return false;
  }
  *destination = current;
  return true;
}

[[nodiscard]] __device__ bool junction_control_slot(
    const DeviceChunkMap& chunks, std::uint64_t center,
    const std::uint32_t permutation[4], std::uint32_t outgoing_sign,
    std::uint32_t incoming_sign, std::uint32_t control_index,
    std::uint64_t* destination) {
  const auto authority = kEligibilityResidualJunctionAuthority;
  const std::uint32_t outgoing =
      permutation[authority.outgoing_role] + 4u * outgoing_sign;
  const std::uint32_t crossed =
      permutation[control_index == 0u ? authority.control_b_role
                                     : authority.control_a_role];
  const std::uint32_t incoming = permutation[authority.incoming_role];
  std::uint64_t current = center;
  if (!junction_walk(chunks, current, outgoing,
                     static_cast<std::uint32_t>(authority.outgoing_distance),
                     &current) ||
      !junction_walk(chunks, current, crossed, 1u, &current) ||
      (incoming_sign == 0u &&
       !junction_walk(
           chunks, current, incoming,
           kEligibilityResidualJunctionAuthority.positive_incoming_control_offset,
           &current)))
    return false;
  *destination = current;
  return true;
}
