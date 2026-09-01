#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

#include "bcc32_law.cuh"
#include "bcc32_spatial_macro_executor.cuh"
#include "bcc32_prediction_residual_route_toggle.cuh"

namespace substrate::bcc32 {
namespace {

constexpr std::uint32_t kThreads = 256u;
constexpr std::uint32_t kAuditBlocks = 4'096u;
constexpr std::int64_t kInt64Min = (-9'223'372'036'854'775'807LL - 1LL);
constexpr std::int64_t kInt64Max = 9'223'372'036'854'775'807LL;

void check_cuda(cudaError_t status, const char* operation);

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


[[nodiscard]] __device__ SiteWord processive_permute_word(SiteWord word,
                                                          const std::uint32_t permutation[4]) {
  SiteWord result = 0u;
  for (std::uint32_t shift = 0u; shift < 32u; shift += 4u) {
    for (std::uint32_t basis = 0u; basis < 4u; ++basis) {
      if ((word & (SiteWord{1u} << (shift + basis))) != 0u) {
        result |= SiteWord{1u} << (shift + permutation[basis]);
      }
    }
  }
  return result;
}

[[nodiscard]] __device__ bool processive_walk(const DeviceChunkMap& chunks, std::uint64_t start,
                                              std::uint32_t basis, std::int32_t steps,
                                              std::uint64_t* destination) {
  std::uint64_t current = start;
  const std::uint32_t direction = steps < 0 ? basis + 4u : basis;
  const std::uint32_t count = static_cast<std::uint32_t>(steps < 0 ? -steps : steps);
  for (std::uint32_t step = 0u; step < count; ++step) {
    if (!neighbor_slot(chunks, current, direction, &current)) {
      return false;
    }
  }
  *destination = current;
  return true;
}

[[nodiscard]] __device__ bool pair_relative_slot(const DeviceChunkMap& chunks, std::uint64_t center,
                                                 std::uint32_t incoming, std::uint32_t diverted,
                                                 std::uint32_t index, std::uint64_t* destination) {
  std::uint64_t current = center;
  if (index == 0u) {
    *destination = current;
    return true;
  }
  std::uint32_t side = 0u;
  while (side == (incoming & 3u) || side == (diverted & 3u))
    ++side;
  if (index == 1u) {
    if (!processive_walk(chunks, current, incoming & 3u, (incoming < 4u ? -1 : 1), &current)) {
      return false;
    }
  } else if (index == 2u) {
    if (!processive_walk(chunks, current, incoming & 3u, (incoming < 4u ? 7 : -7), &current)) {
      return false;
    }
  } else if (index == 3u) {
    if (!processive_walk(chunks, current, diverted & 3u, (diverted < 4u ? 9 : -9), &current)) {
      return false;
    }
  } else {
    if (!processive_walk(chunks, current, side, 11, &current) ||
        !processive_walk(chunks, current, incoming & 3u, (incoming < 4u ? 4 : -4), &current)) {
      return false;
    }
  }
  *destination = current;
  return true;
}

[[nodiscard]] __device__ bool pair_center_from_site(const DeviceChunkMap& chunks,
                                                    std::uint64_t site, std::uint32_t incoming,
                                                    std::uint32_t diverted, std::uint32_t index,
                                                    std::uint64_t* center) {
  std::uint64_t current = site;
  std::uint32_t side = 0u;
  while (side == (incoming & 3u) || side == (diverted & 3u))
    ++side;
  if (index == 1u) {
    if (!processive_walk(chunks, current, incoming & 3u, (incoming < 4u ? 1 : -1), &current)) {
      return false;
    }
  } else if (index == 2u) {
    if (!processive_walk(chunks, current, incoming & 3u, (incoming < 4u ? -7 : 7), &current)) {
      return false;
    }
  } else if (index == 3u) {
    if (!processive_walk(chunks, current, diverted & 3u, (diverted < 4u ? -9 : 9), &current)) {
      return false;
    }
  } else if (index == 4u) {
    if (!processive_walk(chunks, current, incoming & 3u, (incoming < 4u ? -4 : 4), &current) ||
        !processive_walk(chunks, current, side, -11, &current)) {
      return false;
    }
  }
  *center = current;
  return true;
}

[[nodiscard]] __device__ bool pair_owner_present(const SiteWord* words, std::uint64_t site_count,
                                                 const DeviceChunkMap& chunks, std::uint64_t center,
                                                 std::uint32_t incoming, std::uint32_t diverted) {
  for (std::uint32_t index = 2u; index < kCarrierPairSplitterSiteCount; ++index) {
    std::uint64_t slot = 0u;
    if (!pair_relative_slot(chunks, center, incoming, diverted, index, &slot) ||
        slot >= site_count ||
        words[slot] != carrier_pair_splitter_word(incoming, diverted, index, false)) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] __device__ bool pair_decode_rank(std::uint32_t wanted, std::uint32_t* incoming,
                                               std::uint32_t* diverted) {
  std::uint32_t rank = 0u;
  for (std::uint32_t candidate_in = 0u; candidate_in < 8u; ++candidate_in) {
    for (std::uint32_t candidate_out = 0u; candidate_out < 8u; ++candidate_out) {
      if (candidate_in == candidate_out)
        continue;
      if (rank++ == wanted) {
        *incoming = candidate_in;
        *diverted = candidate_out;
        return true;
      }
    }
  }
  return false;
}

[[nodiscard]] __device__ std::uint8_t pair_match_code(const SiteWord* words,
                                                      std::uint64_t site_count,
                                                      const DeviceChunkMap& chunks,
                                                      std::uint64_t center) {
  if ((words[center] & ~kCarrierMask) != 0u || __popc(words[center] & kCarrierMask) != 7) {
    return 0u;
  }
  std::uint32_t rank = 0u;
  for (std::uint32_t incoming = 0u; incoming < 8u; ++incoming) {
    for (std::uint32_t diverted = 0u; diverted < 8u; ++diverted) {
      if (incoming == diverted)
        continue;
      if (!pair_owner_present(words, site_count, chunks, center, incoming, diverted)) {
        ++rank;
        continue;
      }
      bool staged = true;
      bool released = true;
      for (std::uint32_t index = 0u; index < 2u; ++index) {
        std::uint64_t slot = 0u;
        if (!pair_relative_slot(chunks, center, incoming, diverted, index, &slot) ||
            slot >= site_count) {
          staged = false;
          released = false;
          break;
        }
        staged =
            staged && words[slot] == carrier_pair_splitter_word(incoming, diverted, index, false);
        released =
            released && words[slot] == carrier_pair_splitter_word(incoming, diverted, index, true);
      }
      if (staged)
        return static_cast<std::uint8_t>(rank + 1u);
      if (released)
        return static_cast<std::uint8_t>(56u + rank + 1u);
      ++rank;
    }
  }
  return 0u;
}

__global__ void pair_match_kernel(const SiteWord* words, std::uint8_t* matches,
                                  std::uint64_t site_count, DeviceChunkMap chunks) {
  const std::uint64_t center = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (center < site_count)
    matches[center] = pair_match_code(words, site_count, chunks, center);
}

__global__ void pair_collision_kernel(const SiteWord* words, std::uint8_t* matches,
                                      std::uint64_t site_count, DeviceChunkMap chunks) {
  const std::uint64_t center = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (center >= site_count || matches[center] == 0u)
    return;
  std::uint32_t own_in = 0u;
  std::uint32_t own_out = 0u;
  if (!pair_decode_rank((matches[center] - 1u) % 56u, &own_in, &own_out)) {
    matches[center] = 0u;
    return;
  }
  for (std::uint32_t own_index = 0u; own_index < kCarrierPairSplitterSiteCount; ++own_index) {
    std::uint64_t shared = 0u;
    if (!pair_relative_slot(chunks, center, own_in, own_out, own_index, &shared)) {
      continue;
    }
    for (std::uint32_t incoming = 0u; incoming < 8u; ++incoming) {
      for (std::uint32_t diverted = 0u; diverted < 8u; ++diverted) {
        if (incoming == diverted)
          continue;
        for (std::uint32_t other_index = 0u; other_index < kCarrierPairSplitterSiteCount;
             ++other_index) {
          std::uint64_t other_center = 0u;
          if (!pair_center_from_site(chunks, shared, incoming, diverted, other_index,
                                     &other_center) ||
              other_center == center || other_center >= site_count) {
            continue;
          }
          if (pair_owner_present(words, site_count, chunks, other_center, incoming, diverted)) {
            matches[center] = 0u;
            return;
          }
        }
      }
    }
  }
}

__global__ void pair_apply_kernel(SiteWord* words, const std::uint8_t* matches,
                                  std::uint64_t site_count, DeviceChunkMap chunks) {
  const std::uint64_t center = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (center >= site_count || matches[center] == 0u)
    return;
  const bool released = matches[center] > 56u;
  std::uint32_t incoming = 0u;
  std::uint32_t diverted = 0u;
  if (!pair_decode_rank((matches[center] - 1u) % 56u, &incoming, &diverted)) {
    return;
  }
  for (std::uint32_t index = 0u; index < kCarrierPairSplitterSiteCount; ++index) {
    std::uint64_t slot = 0u;
    if (!pair_relative_slot(chunks, center, incoming, diverted, index, &slot) ||
        slot >= site_count) {
      continue;
    }
    words[slot] = carrier_pair_splitter_word(incoming, diverted, index, !released);
  }
}

[[nodiscard]] __device__ bool processive_relative_slot(const DeviceChunkMap& chunks,
                                                       std::uint64_t center,
                                                       const std::uint32_t permutation[4],
                                                       ProcessiveReleaseOffset offset,
                                                       std::uint64_t* destination) {
  std::uint64_t current = center;
  if (!processive_walk(chunks, current, permutation[0u], offset.marker, &current) ||
      !processive_walk(chunks, current, permutation[1u], offset.path, &current) ||
      !processive_walk(chunks, current, permutation[2u], offset.waste, &current)) {
    return false;
  }
  *destination = current;
  return true;
}

[[nodiscard]] __device__ bool processive_rearm_relative_slot(const DeviceChunkMap& chunks,
                                                             std::uint64_t center,
                                                             const std::uint32_t permutation[4],
                                                             ProcessiveRearmOffset offset,
                                                             std::uint64_t* destination) {
  std::uint64_t current = center;
  if (!processive_walk(chunks, current, permutation[0u], offset.marker, &current) ||
      !processive_walk(chunks, current, permutation[1u], offset.path, &current) ||
      !processive_walk(chunks, current, permutation[2u], offset.waste, &current)) {
    return false;
  }
  *destination = current;
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

[[nodiscard]] __device__ bool processive_row_matches(
    const SiteWord* words, std::uint64_t site_count, const DeviceChunkMap& chunks,
    std::uint64_t center, const std::uint32_t permutation[4], std::uint32_t action,
    bool released) {
  for (std::uint32_t index = 0u; index < kProcessiveReleaseSiteCount; ++index) {
    std::uint64_t slot = 0u;
    const bool present = processive_relative_slot(chunks, center, permutation,
                                                  processive_release_offset(index), &slot);
    if (!present || slot >= site_count) return false;
    const SiteWord expected = processive_permute_word(
        released ? processive_release_released_word(action, index)
                 : processive_release_staged_word(action, index),
        permutation);
    const SiteWord actual = words[slot];
    if (actual != expected) {
      return false;
    }
  }
  return true;
}

struct ProcessiveClaimSelection {
  std::uint32_t digit = kProcessiveReleaseClaimDigitCount;
  ProcessiveReleaseClaim role = ProcessiveReleaseClaim::bare;
  bool directional = false;
  std::uint64_t endpoint = 0u;
  std::uint64_t successor_center = 0u;
  std::uint64_t successor_ingress = 0u;
};

[[nodiscard]] __device__ int processive_claim_role(
    SiteWord word, const std::uint32_t permutation[4]) {
  const SiteWord role_a = processive_permute_word(
      processive_release_claim_word(ProcessiveReleaseClaim::adult_a),
      permutation);
  const SiteWord role_b = processive_permute_word(
      processive_release_claim_word(ProcessiveReleaseClaim::adult_b),
      permutation);
  return word == role_a ? 0 : (word == role_b ? 1 : -1);
}

[[nodiscard]] __device__ bool processive_owner_present(
    const SiteWord* words, std::uint64_t site_count,
    const DeviceChunkMap& chunks, std::uint64_t center,
    const std::uint32_t permutation[4]) {
  for (std::uint32_t index = 10u; index < 12u; ++index) {
    std::uint64_t slot = 0u;
    if (!processive_relative_slot(chunks, center, permutation,
                                  processive_release_offset(index), &slot) ||
        slot >= site_count ||
        words[slot] != processive_permute_word(
                           processive_release_staged_word(0u, index),
                           permutation))
      return false;
  }
  return true;
}

[[nodiscard]] __device__ bool processive_successor_geometry(
    const DeviceChunkMap& chunks, std::uint64_t center,
    const std::uint32_t permutation[4], std::uint64_t* successor_center,
    std::uint64_t* successor_ingress) {
  return processive_walk(chunks, center, permutation[1u], 6,
                         successor_center) &&
         processive_relative_slot(
             chunks, center, permutation,
             processive_successor_claim_offset(kProcessiveRoleIngressDigit),
             successor_ingress);
}

[[nodiscard]] __device__ bool processive_claim_stack_matches(
    const SiteWord* words, std::uint64_t site_count,
    const DeviceChunkMap& chunks, std::uint64_t center,
    const std::uint32_t permutation[4], std::uint32_t action, bool inverse,
    ProcessiveClaimSelection* selection) {
  SiteWord actual[kProcessiveReleaseClaimDigitCount]{};
  const SiteWord bare = processive_permute_word(
      processive_release_claim_word(ProcessiveReleaseClaim::bare),
      permutation);
  const SiteWord empty = processive_permute_word(
      processive_release_claim_word(ProcessiveReleaseClaim::empty),
      permutation);
  bool all_bare = true;
  bool all_empty = true;
  for (std::uint32_t digit = 0u;
       digit < kProcessiveReleaseClaimDigitCount; ++digit) {
    std::uint64_t slot = 0u;
    if (!processive_relative_slot(
            chunks, center, permutation,
            processive_release_claim_offset(digit), &slot) ||
        slot >= site_count)
      return false;
    actual[digit] = words[slot];
    all_bare = all_bare && actual[digit] == bare;
    all_empty = all_empty && actual[digit] == empty;
  }
  selection->digit = kProcessiveReleaseClaimDigitCount;
  selection->role = ProcessiveReleaseClaim::bare;
  selection->directional = !(all_bare || all_empty);
  selection->endpoint = 0u;
  selection->successor_center = 0u;
  selection->successor_ingress = 0u;
  if (!selection->directional)
    return true;

  for (std::uint32_t digit = kProcessiveRoleGuardFirst;
       digit < kProcessiveReleaseClaimDigitCount; ++digit)
    if (actual[digit] != empty)
      return false;
  const int ingress_role = processive_claim_role(
      actual[kProcessiveRoleIngressDigit], permutation);
  if (!inverse) {
    if (ingress_role < 0)
      return false;
    if (processive_action_carries(action)) {
      if (processive_claim_role(actual[kProcessiveRoleLandedDigit],
                                permutation) < 0 ||
          !processive_successor_geometry(
              chunks, center, permutation, &selection->successor_center,
              &selection->successor_ingress) ||
          !processive_owner_present(words, site_count,
                                    chunks, selection->successor_center,
                                    permutation) ||
          words[selection->successor_ingress] != empty)
        return false;
      selection->endpoint = selection->successor_ingress;
    } else {
      if (actual[kProcessiveRoleLandedDigit] != empty ||
          !processive_relative_slot(
              chunks, center, permutation,
              processive_release_claim_offset(kProcessiveRoleLandedDigit),
              &selection->endpoint))
        return false;
    }
    selection->role = ingress_role == 0
                         ? ProcessiveReleaseClaim::adult_a
                         : ProcessiveReleaseClaim::adult_b;
  } else {
    if (actual[kProcessiveRoleIngressDigit] != empty)
      return false;
    int endpoint_role = processive_claim_role(
        actual[kProcessiveRoleLandedDigit], permutation);
    if (processive_action_carries(action)) {
      if (endpoint_role < 0 ||
          !processive_successor_geometry(
              chunks, center, permutation, &selection->successor_center,
              &selection->successor_ingress) ||
          !processive_owner_present(words, site_count,
                                    chunks, selection->successor_center,
                                    permutation))
        return false;
      endpoint_role = processive_claim_role(
          words[selection->successor_ingress], permutation);
      if (endpoint_role < 0)
        return false;
      selection->endpoint = selection->successor_ingress;
    } else {
      if (endpoint_role < 0 ||
          !processive_relative_slot(
              chunks, center, permutation,
              processive_release_claim_offset(kProcessiveRoleLandedDigit),
              &selection->endpoint))
        return false;
    }
    selection->role = endpoint_role == 0
                         ? ProcessiveReleaseClaim::adult_a
                         : ProcessiveReleaseClaim::adult_b;
  }
  selection->digit = kProcessiveRoleIngressDigit;
  return true;
}

[[nodiscard]] __device__ constexpr ProcessiveReleaseOffset
processive_collision_offset(std::uint32_t index) {
  if (index < kProcessiveReleaseSiteCount)
    return processive_release_offset(index);
  if (index < kProcessiveReleaseFootprintCount)
    return processive_release_claim_offset(index -
                                           kProcessiveReleaseSiteCount);
  return processive_successor_claim_offset(kProcessiveRoleIngressDigit);
}

[[nodiscard]] __device__ std::uint8_t processive_match_code(const SiteWord* words,
                                                            std::uint64_t site_count,
                                                            const DeviceChunkMap& chunks,
                                                            std::uint64_t center,
                                                            bool inverse) {
  if ((words[center] & ~kCarrierMask) == 0u) {
    return 0u;
  }
  std::uint32_t rank = 0u;
  for (std::uint32_t marker = 0u; marker < 4u; ++marker) {
    for (std::uint32_t path = 0u; path < 4u; ++path) {
      if (path == marker)
        continue;
      for (std::uint32_t waste = 0u; waste < 4u; ++waste) {
        if (waste == marker || waste == path)
          continue;
        std::uint32_t permutation[4]{};
        processive_permutation(marker, path, waste, permutation);
        for (std::uint32_t action = 0u; action < kProcessiveReleaseActionCount; ++action, ++rank) {
          ProcessiveClaimSelection selection{};
          if (!processive_claim_stack_matches(
                  words, site_count, chunks, center, permutation, action,
                  inverse,
                  &selection))
            continue;
          if (processive_row_matches(words, site_count, chunks, center,
                                     permutation, action, false)) {
            return static_cast<std::uint8_t>(rank + 1u);
          }
          if (processive_row_matches(words, site_count, chunks, center,
                                     permutation, action, true)) {
            return static_cast<std::uint8_t>(
                24u * kProcessiveReleaseActionCount + rank + 1u);
          }
        }
      }
    }
  }
  return 0u;
}

[[nodiscard]] __device__ bool processive_decode_rank(std::uint32_t wanted,
                                                     std::uint32_t permutation[4],
                                                     std::uint32_t* action) {
  std::uint32_t rank = 0u;
  for (std::uint32_t marker = 0u; marker < 4u; ++marker) {
    for (std::uint32_t path = 0u; path < 4u; ++path) {
      if (path == marker)
        continue;
      for (std::uint32_t waste = 0u; waste < 4u; ++waste) {
        if (waste == marker || waste == path)
          continue;
        for (std::uint32_t candidate = 0u; candidate < kProcessiveReleaseActionCount;
             ++candidate, ++rank) {
          if (rank == wanted) {
            processive_permutation(marker, path, waste, permutation);
            *action = candidate;
            return true;
          }
        }
      }
    }
  }
  return false;
}

[[nodiscard]] __device__ bool processive_lock_present(const SiteWord* words,
                                                      std::uint64_t site_count,
                                                      const DeviceChunkMap& chunks,
                                                      std::uint64_t center,
                                                      const std::uint32_t permutation[4]) {
  for (std::uint32_t index = 10u; index < 12u; ++index) {
    std::uint64_t slot = 0u;
    if (!processive_relative_slot(chunks, center, permutation, processive_release_offset(index),
                                  &slot) ||
        slot >= site_count ||
        words[slot] !=
            processive_permute_word(processive_release_staged_word(0u, index), permutation)) {
      return false;
    }
  }
  return true;
}

__global__ void processive_match_kernel(const SiteWord* words, std::uint8_t* matches,
                                        std::uint64_t site_count,
                                        DeviceChunkMap chunks, bool inverse) {
  const std::uint64_t center = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (center >= site_count)
    return;
  matches[center] =
      processive_match_code(words, site_count, chunks, center, inverse);
}

__global__ void processive_collision_kernel(const SiteWord* words, std::uint8_t* matches,
                                            std::uint64_t site_count, DeviceChunkMap chunks) {
  const std::uint64_t center = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (center >= site_count || matches[center] == 0u)
    return;

  std::uint32_t own_permutation[4]{};
  std::uint32_t own_action = 0u;
  const std::uint32_t own_rank = (matches[center] - 1u) % (24u * kProcessiveReleaseActionCount);
  if (!processive_decode_rank(own_rank, own_permutation, &own_action)) {
    matches[center] = 0u;
    return;
  }
  const bool own_inverse =
      matches[center] > 24u * kProcessiveReleaseActionCount;
  ProcessiveClaimSelection own_selection{};
  if (!processive_claim_stack_matches(
          words, site_count, chunks, center, own_permutation, own_action,
          own_inverse, &own_selection)) {
    matches[center] = 0u;
    return;
  }
  if (!own_selection.directional) {
    for (std::uint32_t marker = 0u; marker < 4u; ++marker) {
      for (std::uint32_t path = 0u; path < 4u; ++path) {
        if (path == marker) continue;
        for (std::uint32_t waste = 0u; waste < 4u; ++waste) {
          if (waste == marker || waste == path) continue;
          std::uint32_t source_permutation[4]{};
          processive_permutation(marker, path, waste,
                                 source_permutation);
          std::uint64_t source_center = 0u;
          if (!processive_walk(chunks, center, source_permutation[1u], -6,
                               &source_center) ||
              source_center >= site_count ||
              matches[source_center] == 0u)
            continue;
          std::uint32_t matched_permutation[4]{};
          std::uint32_t matched_action = 0u;
          const std::uint32_t matched_rank =
              (matches[source_center] - 1u) %
              (24u * kProcessiveReleaseActionCount);
          if (!processive_decode_rank(matched_rank, matched_permutation,
                                      &matched_action) ||
              matched_action < 2u || matched_action >=
                                         kProcessiveReleaseActionCount)
            continue;
          bool same_frame = true;
          for (std::uint32_t basis = 0u; basis < 4u; ++basis)
            same_frame = same_frame &&
                         matched_permutation[basis] == source_permutation[basis];
          if (!same_frame) continue;
          ProcessiveClaimSelection source_selection{};
          if (processive_claim_stack_matches(
                  words, site_count, chunks, source_center,
                  matched_permutation, matched_action,
                  matches[source_center] >
                      24u * kProcessiveReleaseActionCount,
                  &source_selection) && source_selection.directional) {
            matches[center] = 0u;
            return;
          }
        }
      }
    }
  }
  const std::uint32_t own_footprint_count =
      kProcessiveReleaseFootprintCount +
      (processive_action_carries(own_action) ? 1u : 0u);
  for (std::uint32_t own_index = 0u;
       own_index < own_footprint_count; ++own_index) {
    std::uint64_t shared = 0u;
    if (!processive_relative_slot(chunks, center, own_permutation,
                                  processive_collision_offset(own_index),
                                  &shared) ||
        shared >= site_count) {
      matches[center] = 0u;
      return;
    }
    for (std::uint32_t marker = 0u; marker < 4u; ++marker) {
      for (std::uint32_t path = 0u; path < 4u; ++path) {
        if (path == marker)
          continue;
        for (std::uint32_t waste = 0u; waste < 4u; ++waste) {
          if (waste == marker || waste == path)
            continue;
          std::uint32_t other_permutation[4]{};
          processive_permutation(marker, path, waste, other_permutation);
          for (std::uint32_t other_index = 0u;
               other_index < kProcessiveReleaseFootprintCount;
               ++other_index) {
            ProcessiveReleaseOffset inverse_offset =
                processive_collision_offset(other_index);
            inverse_offset.marker = -inverse_offset.marker;
            inverse_offset.path = -inverse_offset.path;
            inverse_offset.waste = -inverse_offset.waste;
            std::uint64_t other_center = 0u;
            if (!processive_relative_slot(chunks, shared, other_permutation, inverse_offset,
                                          &other_center) ||
                other_center == center || other_center >= site_count) {
              continue;
            }
            bool intentional_successor = false;
            if (own_index == kProcessiveReleaseFootprintCount &&
                processive_action_carries(own_action) &&
                other_index == kProcessiveRoleIngressDigit) {
              std::uint64_t successor_center = 0u;
              std::uint64_t successor_ingress = 0u;
              if (processive_successor_geometry(
                      chunks, center, own_permutation, &successor_center,
                      &successor_ingress)) {
                intentional_successor =
                    other_center == successor_center &&
                    other_permutation[0u] == own_permutation[0u] &&
                    other_permutation[1u] == own_permutation[1u] &&
                    other_permutation[2u] == own_permutation[2u] &&
                    other_permutation[3u] == own_permutation[3u];
              }
            }
            if (!intentional_successor &&
                processive_owner_present(words, site_count, chunks,
                                         other_center, other_permutation)) {
              matches[center] = 0u;
              return;
            }
          }
        }
      }
    }
  }
}

__global__ void processive_apply_kernel(SiteWord* words, const std::uint8_t* matches,
                                        std::uint64_t site_count, DeviceChunkMap chunks) {
  const std::uint64_t center = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (center >= site_count || matches[center] == 0u) {
    return;
  }
  const std::uint32_t endpoint_count = 24u * kProcessiveReleaseActionCount;
  const bool released = matches[center] > endpoint_count;
  const std::uint32_t rank = (matches[center] - 1u) % endpoint_count;
  std::uint32_t permutation[4]{};
  std::uint32_t action = 0u;
  if (!processive_decode_rank(rank, permutation, &action)) {
    return;
  }
  ProcessiveClaimSelection selection{};
  if (!processive_row_matches(words, site_count, chunks, center, permutation,
                              action, released) ||
      !processive_claim_stack_matches(words, site_count, chunks, center,
                                      permutation, action, released,
                                      &selection) ||
      (selection.directional && selection.digit >=
                                    kProcessiveReleaseClaimDigitCount))
    return;
  std::uint64_t slots[kProcessiveReleaseSiteCount]{};
  for (std::uint32_t index = 0u; index < kProcessiveReleaseSiteCount; ++index) {
    if (!processive_relative_slot(chunks, center, permutation,
                                  processive_release_offset(index),
                                  &slots[index]) ||
        slots[index] >= site_count)
      return;
  }
  std::uint64_t claim_slots[kProcessiveReleaseClaimDigitCount]{};
  for (std::uint32_t digit = 0u;
       digit < kProcessiveReleaseClaimDigitCount; ++digit)
    if (!processive_relative_slot(
            chunks, center, permutation,
            processive_release_claim_offset(digit), &claim_slots[digit]) ||
        claim_slots[digit] >= site_count)
      return;
  for (std::uint32_t index = 0u; index < kProcessiveReleaseSiteCount;
       ++index)
    words[slots[index]] = processive_permute_word(
        released ? processive_release_staged_word(action, index)
                 : processive_release_released_word(action, index),
        permutation);
  if (selection.directional) {
    const SiteWord role_word = processive_permute_word(
        processive_release_claim_word(selection.role), permutation);
    const SiteWord empty_word = processive_permute_word(
        processive_release_claim_word(ProcessiveReleaseClaim::empty),
        permutation);
    words[claim_slots[kProcessiveRoleIngressDigit]] =
        released ? role_word : empty_word;
    words[selection.endpoint] = released ? empty_word : role_word;
  }
}

[[nodiscard]] __device__ bool processive_rearm_pattern_matches(
    const SiteWord* words, std::uint64_t site_count, const DeviceChunkMap& chunks,
    std::uint64_t center, const std::uint32_t permutation[4], bool rearmed) {
  for (std::uint32_t index = 0u; index < kProcessiveRearmSiteCount; ++index) {
    std::uint64_t slot = 0u;
    const bool present = processive_rearm_relative_slot(chunks, center, permutation,
                                                        processive_rearm_offset(index), &slot);
    const SiteWord expected = processive_permute_word(
        rearmed ? processive_rearm_rearmed_word(index) : processive_rearm_candidate_word(index),
        permutation);
    const SiteWord actual = present && slot < site_count ? words[slot] : kQ;
    if (actual != expected) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] __device__ bool processive_rearm_decode_rank(std::uint32_t rank,
                                                           std::uint32_t permutation[4]) {
  std::uint32_t current = 0u;
  for (std::uint32_t marker = 0u; marker < 4u; ++marker) {
    for (std::uint32_t path = 0u; path < 4u; ++path) {
      if (path == marker)
        continue;
      for (std::uint32_t waste = 0u; waste < 4u; ++waste) {
        if (waste == marker || waste == path)
          continue;
        if (current == rank) {
          processive_permutation(marker, path, waste, permutation);
          return true;
        }
        ++current;
      }
    }
  }
  return false;
}

[[nodiscard]] __device__ std::uint8_t processive_rearm_match_code(const SiteWord* words,
                                                                  std::uint64_t site_count,
                                                                  const DeviceChunkMap& chunks,
                                                                  std::uint64_t center) {
  if ((words[center] & ~kCarrierMask) == 0u) {
    return 0u;
  }
  std::uint32_t rank = 0u;
  for (std::uint32_t marker = 0u; marker < 4u; ++marker) {
    for (std::uint32_t path = 0u; path < 4u; ++path) {
      if (path == marker)
        continue;
      for (std::uint32_t waste = 0u; waste < 4u; ++waste) {
        if (waste == marker || waste == path)
          continue;
        std::uint32_t permutation[4]{};
        processive_permutation(marker, path, waste, permutation);
        if (processive_rearm_pattern_matches(words, site_count, chunks, center, permutation,
                                             false)) {
          return static_cast<std::uint8_t>(rank + 1u);
        }
        if (processive_rearm_pattern_matches(words, site_count, chunks, center, permutation,
                                             true)) {
          return static_cast<std::uint8_t>(24u + rank + 1u);
        }
        ++rank;
      }
    }
  }
  return 0u;
}

[[nodiscard]] __device__ bool processive_rearm_lock_present(const SiteWord* words,
                                                            std::uint64_t site_count,
                                                            const DeviceChunkMap& chunks,
                                                            std::uint64_t center,
                                                            const std::uint32_t permutation[4]) {
  for (std::uint32_t index = 6u; index < 9u; ++index) {
    std::uint64_t slot = 0u;
    if (!processive_rearm_relative_slot(chunks, center, permutation, processive_rearm_offset(index),
                                        &slot) ||
        slot >= site_count ||
        words[slot] !=
            processive_permute_word(processive_rearm_candidate_word(index), permutation)) {
      return false;
    }
  }
  return true;
}

__global__ void processive_rearm_match_kernel(const SiteWord* words, std::uint8_t* matches,
                                              std::uint64_t site_count, DeviceChunkMap chunks) {
  const std::uint64_t center = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (center >= site_count)
    return;
  matches[center] = processive_rearm_match_code(words, site_count, chunks, center);
}

__global__ void processive_rearm_collision_kernel(const SiteWord* words, std::uint8_t* matches,
                                                  std::uint64_t site_count, DeviceChunkMap chunks) {
  const std::uint64_t center = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (center >= site_count || matches[center] == 0u)
    return;

  std::uint32_t own_permutation[4]{};
  const std::uint32_t own_rank = (matches[center] - 1u) % 24u;
  if (!processive_rearm_decode_rank(own_rank, own_permutation)) {
    matches[center] = 0u;
    return;
  }
  for (std::uint32_t own_index = 0u; own_index < kProcessiveRearmSiteCount; ++own_index) {
    std::uint64_t shared = 0u;
    if (!processive_rearm_relative_slot(chunks, center, own_permutation,
                                        processive_rearm_offset(own_index), &shared)) {
      continue;
    }
    for (std::uint32_t marker = 0u; marker < 4u; ++marker) {
      for (std::uint32_t path = 0u; path < 4u; ++path) {
        if (path == marker)
          continue;
        for (std::uint32_t waste = 0u; waste < 4u; ++waste) {
          if (waste == marker || waste == path)
            continue;
          std::uint32_t other_permutation[4]{};
          processive_permutation(marker, path, waste, other_permutation);
          for (std::uint32_t other_index = 0u; other_index < kProcessiveRearmSiteCount;
               ++other_index) {
            ProcessiveRearmOffset inverse_offset = processive_rearm_offset(other_index);
            inverse_offset.marker = -inverse_offset.marker;
            inverse_offset.path = -inverse_offset.path;
            inverse_offset.waste = -inverse_offset.waste;
            std::uint64_t other_center = 0u;
            if (!processive_rearm_relative_slot(chunks, shared, other_permutation, inverse_offset,
                                                &other_center) ||
                other_center == center || other_center >= site_count) {
              continue;
            }
            if (processive_rearm_lock_present(words, site_count, chunks, other_center,
                                              other_permutation)) {
              matches[center] = 0u;
              return;
            }
          }
        }
      }
    }
  }
}

__global__ void processive_rearm_apply_kernel(SiteWord* words, const std::uint8_t* matches,
                                              std::uint64_t site_count, DeviceChunkMap chunks) {
  const std::uint64_t center = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (center >= site_count || matches[center] == 0u)
    return;
  const bool rearmed = matches[center] > 24u;
  const std::uint32_t rank = (matches[center] - 1u) % 24u;
  std::uint32_t permutation[4]{};
  if (!processive_rearm_decode_rank(rank, permutation))
    return;
  for (std::uint32_t index = 0u; index < kProcessiveRearmSiteCount; ++index) {
    std::uint64_t slot = 0u;
    if (!processive_rearm_relative_slot(chunks, center, permutation, processive_rearm_offset(index),
                                        &slot) ||
        slot >= site_count) {
      continue;
    }
    words[slot] = processive_permute_word(
        rearmed ? processive_rearm_candidate_word(index) : processive_rearm_rearmed_word(index),
        permutation);
  }
}

[[nodiscard]] __device__ bool corner_relative_slot(const DeviceChunkMap& chunks,
                                                   std::uint64_t center, std::uint32_t incoming,
                                                   std::uint32_t outgoing, std::uint32_t index,
                                                   std::uint64_t* destination) {
  if (index == 0u) {
    *destination = center;
    return true;
  }
  const std::uint32_t direction = index == 1u ? incoming : outgoing;
  const std::uint32_t count = index == 1u ? 6u : 8u;
  std::uint32_t side = 0u;
  while (side == (incoming & 3u) || side == (outgoing & 3u))
    ++side;
  const std::uint32_t side_count = index == 1u ? 2u : 3u;
  std::uint64_t current = center;
  for (std::uint32_t step = 0u; step < count; ++step) {
    if (!neighbor_slot(chunks, current, direction, &current)) {
      return false;
    }
  }
  for (std::uint32_t step = 0u; step < side_count; ++step) {
    if (!neighbor_slot(chunks, current, side, &current)) {
      return false;
    }
  }
  *destination = current;
  return true;
}

[[nodiscard]] __device__ bool corner_owner_present(const SiteWord* words, std::uint64_t site_count,
                                                   const DeviceChunkMap& chunks,
                                                   std::uint64_t center, std::uint32_t incoming,
                                                   std::uint32_t outgoing) {
  for (std::uint32_t index = 1u; index < kCarrierCornerSiteCount; ++index) {
    std::uint64_t slot = 0u;
    if (!corner_relative_slot(chunks, center, incoming, outgoing, index, &slot) ||
        slot >= site_count ||
        words[slot] != carrier_corner_word(incoming, outgoing, index, false)) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] __device__ std::uint8_t corner_match_code(const SiteWord* words,
                                                        std::uint64_t site_count,
                                                        const DeviceChunkMap& chunks,
                                                        std::uint64_t center) {
  const SiteWord center_word = words[center];
  // Quiescent-word shortcut, mirroring service_match_kernel's fix (4bff6e14ab)
  // applied to a different per-pair predicate. carrier_corner_center_matches()
  // rejects any word carrying a structural channel (first disjunct) and
  // otherwise requires incoming_occupied != outgoing_occupied to count a pair
  // toward compatible_owners. carrier_bit(d) == 1u << d for d < 8, so the
  // eight named bits are exactly kCarrierMask: with no carrier bit set, both
  // named bits are always clear; with every carrier bit set (kQuiescentWord ==
  // kCarrierMask), both named bits are always set. Either way
  // incoming_occupied == outgoing_occupied for every one of the 56 ordered
  // (incoming, outgoing) pairs, so compatible_owners cannot leave 0 and
  // carrier_corner_unique_owner_matches() -- which requires compatible_owners
  // == 1 -- can never be true. This does not depend on present_owners or the
  // corner_owner_present() lattice walk at all: compatible_owners is gated on
  // carrier_corner_center_matches() alone, a pure function of center_word.
  // Skipping the sweep for the quiescent bath that covers essentially the
  // whole aperture keeps the emitted byte bit-identical while making the
  // corner-match cost track occupied carrier support instead of aperture size.
  if ((center_word & ~kCarrierMask) != 0u || (center_word & kCarrierMask) == 0u ||
      (center_word & kCarrierMask) == kCarrierMask)
    return 0u;
  std::uint32_t rank = 0u;
  std::uint8_t result = 0u;
  std::uint32_t present_owners = 0u;
  std::uint32_t compatible_owners = 0u;
  for (std::uint32_t incoming = 0u; incoming < 8u; ++incoming) {
    for (std::uint32_t outgoing = 0u; outgoing < 8u; ++outgoing) {
      if (incoming == outgoing)
        continue;
      const bool owner_present = corner_owner_present(
          words, site_count, chunks, center, incoming, outgoing);
      if (owner_present) ++present_owners;
      if (owner_present &&
          carrier_corner_center_matches(center_word, incoming, outgoing)) {
        ++compatible_owners;
        result = static_cast<std::uint8_t>(
            (carrier_corner_center_released(
                 center_word, incoming, outgoing)
                 ? 56u
                 : 0u) +
            rank + 1u);
      }
      ++rank;
    }
  }
  return carrier_corner_unique_owner_matches(
             present_owners, compatible_owners)
      ? result
      : 0u;
}

[[nodiscard]] __device__ bool corner_decode_rank(std::uint32_t wanted, std::uint32_t* incoming,
                                                 std::uint32_t* outgoing) {
  std::uint32_t rank = 0u;
  for (std::uint32_t candidate_in = 0u; candidate_in < 8u; ++candidate_in) {
    for (std::uint32_t candidate_out = 0u; candidate_out < 8u; ++candidate_out) {
      if (candidate_in == candidate_out)
        continue;
      if (rank++ == wanted) {
        *incoming = candidate_in;
        *outgoing = candidate_out;
        return true;
      }
    }
  }
  return false;
}

__global__ void corner_match_kernel(const SiteWord* words, std::uint8_t* matches,
                                    std::uint64_t site_count, DeviceChunkMap chunks) {
  const std::uint64_t center = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (center >= site_count)
    return;
  matches[center] = corner_match_code(words, site_count, chunks, center);
}

__global__ void corner_collision_kernel(const SiteWord* words, std::uint8_t* matches,
                                        std::uint64_t site_count, DeviceChunkMap chunks) {
  const std::uint64_t center = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (center >= site_count || matches[center] == 0u)
    return;
  std::uint32_t own_in = 0u;
  std::uint32_t own_out = 0u;
  if (!corner_decode_rank((matches[center] - 1u) % 56u, &own_in, &own_out)) {
    matches[center] = 0u;
    return;
  }
  for (std::uint32_t own_index = 0u; own_index < kCarrierCornerSiteCount; ++own_index) {
    std::uint64_t shared = 0u;
    if (!corner_relative_slot(chunks, center, own_in, own_out, own_index, &shared)) {
      continue;
    }
    for (std::uint32_t incoming = 0u; incoming < 8u; ++incoming) {
      for (std::uint32_t outgoing = 0u; outgoing < 8u; ++outgoing) {
        if (incoming == outgoing)
          continue;
        for (std::uint32_t other_index = 0u; other_index < kCarrierCornerSiteCount; ++other_index) {
          std::uint64_t other_center = shared;
          if (other_index != 0u) {
            const std::uint32_t direction = other_index == 1u ? incoming : outgoing;
            const std::uint32_t count = other_index == 1u ? 6u : 8u;
            std::uint32_t side = 0u;
            while (side == (incoming & 3u) || side == (outgoing & 3u))
              ++side;
            const std::uint32_t side_count = other_index == 1u ? 2u : 3u;
            for (std::uint32_t step = 0u; step < side_count; ++step) {
              if (!neighbor_slot(chunks, other_center, side ^ 4u, &other_center)) {
                other_center = site_count;
                break;
              }
            }
            for (std::uint32_t step = 0u; step < count; ++step) {
              if (!neighbor_slot(chunks, other_center, direction ^ 4u, &other_center)) {
                other_center = site_count;
                break;
              }
            }
          }
          if (other_center == center || other_center >= site_count) {
            continue;
          }
          if (corner_owner_present(words, site_count, chunks, other_center, incoming, outgoing)) {
            matches[center] = 0u;
            return;
          }
        }
      }
    }
  }
}

__global__ void corner_apply_kernel(SiteWord* words, const std::uint8_t* matches,
                                    std::uint64_t site_count, bool) {
  const std::uint64_t center = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (center >= site_count || matches[center] == 0u) {
    return;
  }
  std::uint32_t incoming = 0u;
  std::uint32_t outgoing = 0u;
  if (!corner_decode_rank((matches[center] - 1u) % 56u, &incoming, &outgoing)) {
    return;
  }
  words[center] = carrier_corner_transpose(
      words[center], incoming, outgoing);
}

[[nodiscard]] __device__ std::uint64_t bounded_active_count(
    std::uint64_t capacity, const std::uint32_t* device_active_count) {
  return device_active_count == nullptr
             ? capacity
             : min(capacity,
                   static_cast<std::uint64_t>(*device_active_count));
}

__global__ void active_pair_match_kernel(
    const SiteWord* words, std::uint8_t* matches,
    const std::uint64_t* active_slots, std::uint64_t active_count,
    std::uint64_t site_count, DeviceChunkMap chunks,
    const std::uint32_t* device_active_count) {
  const std::uint64_t index =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  active_count = bounded_active_count(active_count, device_active_count);
  if (index >= active_count) return;
  if (active_slots[index] >= site_count) {
    matches[index] = 0u;
    return;
  }
  if (index != 0u && active_slots[index - 1u] == active_slots[index]) {
    matches[index] = 0u;
    return;
  }
  matches[index] =
      pair_match_code(words, site_count, chunks, active_slots[index]);
}

__global__ void active_pair_collision_kernel(
    const SiteWord* words, std::uint8_t* matches,
    const std::uint64_t* active_slots, std::uint64_t active_count,
    std::uint64_t site_count, DeviceChunkMap chunks,
    const std::uint32_t* device_active_count) {
  const std::uint64_t index =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  active_count = bounded_active_count(active_count, device_active_count);
  if (index >= active_count || matches[index] == 0u) return;
  const std::uint64_t center = active_slots[index];
  std::uint32_t own_in = 0u;
  std::uint32_t own_out = 0u;
  if (!pair_decode_rank((matches[index] - 1u) % 56u, &own_in, &own_out)) {
    matches[index] = 0u;
    return;
  }
  for (std::uint32_t own_index = 0u;
       own_index < kCarrierPairSplitterSiteCount; ++own_index) {
    std::uint64_t shared = 0u;
    if (!pair_relative_slot(chunks, center, own_in, own_out, own_index,
                            &shared))
      continue;
    for (std::uint32_t incoming = 0u; incoming < 8u; ++incoming) {
      for (std::uint32_t diverted = 0u; diverted < 8u; ++diverted) {
        if (incoming == diverted) continue;
        for (std::uint32_t other_index = 0u;
             other_index < kCarrierPairSplitterSiteCount; ++other_index) {
          std::uint64_t other_center = 0u;
          if (!pair_center_from_site(chunks, shared, incoming, diverted,
                                     other_index, &other_center) ||
              other_center == center || other_center >= site_count)
            continue;
          if (pair_owner_present(words, site_count, chunks, other_center,
                                 incoming, diverted)) {
            matches[index] = 0u;
            return;
          }
        }
      }
    }
  }
}

__global__ void active_pair_apply_kernel(
    SiteWord* words, const std::uint8_t* matches,
    const std::uint64_t* active_slots, std::uint64_t active_count,
    std::uint64_t site_count, DeviceChunkMap chunks,
    const std::uint32_t* device_active_count) {
  const std::uint64_t index =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  active_count = bounded_active_count(active_count, device_active_count);
  if (index >= active_count || matches[index] == 0u) return;
  const std::uint64_t center = active_slots[index];
  const bool released = matches[index] > 56u;
  std::uint32_t incoming = 0u;
  std::uint32_t diverted = 0u;
  if (!pair_decode_rank((matches[index] - 1u) % 56u, &incoming, &diverted))
    return;
  for (std::uint32_t site = 0u;
       site < kCarrierPairSplitterSiteCount; ++site) {
    std::uint64_t slot = 0u;
    if (pair_relative_slot(chunks, center, incoming, diverted, site, &slot) &&
        slot < site_count) {
      words[slot] =
          carrier_pair_splitter_word(incoming, diverted, site, !released);
    }
  }
}

__global__ void active_processive_rearm_match_kernel(
    const SiteWord* words, std::uint8_t* matches,
    const std::uint64_t* active_slots, std::uint64_t active_count,
    std::uint64_t site_count, DeviceChunkMap chunks,
    const std::uint32_t* device_active_count) {
  const std::uint64_t index =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  active_count = bounded_active_count(active_count, device_active_count);
  if (index >= active_count) return;
  if (active_slots[index] >= site_count) {
    matches[index] = 0u;
    return;
  }
  if (index != 0u && active_slots[index - 1u] == active_slots[index]) {
    matches[index] = 0u;
    return;
  }
  matches[index] = processive_rearm_match_code(
      words, site_count, chunks, active_slots[index]);
}

__global__ void active_processive_rearm_collision_kernel(
    const SiteWord* words, std::uint8_t* matches,
    const std::uint64_t* active_slots, std::uint64_t active_count,
    std::uint64_t site_count, DeviceChunkMap chunks,
    const std::uint32_t* device_active_count) {
  const std::uint64_t index =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  active_count = bounded_active_count(active_count, device_active_count);
  if (index >= active_count || matches[index] == 0u) return;
  const std::uint64_t center = active_slots[index];
  std::uint32_t own_permutation[4]{};
  const std::uint32_t own_rank = (matches[index] - 1u) % 24u;
  if (!processive_rearm_decode_rank(own_rank, own_permutation)) {
    matches[index] = 0u;
    return;
  }
  for (std::uint32_t own_index = 0u;
       own_index < kProcessiveRearmSiteCount; ++own_index) {
    std::uint64_t shared = 0u;
    if (!processive_rearm_relative_slot(
            chunks, center, own_permutation,
            processive_rearm_offset(own_index), &shared))
      continue;
    for (std::uint32_t marker = 0u; marker < 4u; ++marker) {
      for (std::uint32_t path = 0u; path < 4u; ++path) {
        if (path == marker) continue;
        for (std::uint32_t waste = 0u; waste < 4u; ++waste) {
          if (waste == marker || waste == path) continue;
          std::uint32_t other_permutation[4]{};
          processive_permutation(marker, path, waste, other_permutation);
          for (std::uint32_t other_index = 0u;
               other_index < kProcessiveRearmSiteCount; ++other_index) {
            ProcessiveRearmOffset offset =
                processive_rearm_offset(other_index);
            offset.marker = -offset.marker;
            offset.path = -offset.path;
            offset.waste = -offset.waste;
            std::uint64_t other_center = 0u;
            if (!processive_rearm_relative_slot(
                    chunks, shared, other_permutation, offset,
                    &other_center) ||
                other_center == center || other_center >= site_count)
              continue;
            if (processive_rearm_lock_present(
                    words, site_count, chunks, other_center,
                    other_permutation)) {
              matches[index] = 0u;
              return;
            }
          }
        }
      }
    }
  }
}

__global__ void active_processive_rearm_apply_kernel(
    SiteWord* words, const std::uint8_t* matches,
    const std::uint64_t* active_slots, std::uint64_t active_count,
    std::uint64_t site_count, DeviceChunkMap chunks,
    const std::uint32_t* device_active_count) {
  const std::uint64_t index =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  active_count = bounded_active_count(active_count, device_active_count);
  if (index >= active_count || matches[index] == 0u) return;
  const bool rearmed = matches[index] > 24u;
  const std::uint32_t rank = (matches[index] - 1u) % 24u;
  std::uint32_t permutation[4]{};
  if (!processive_rearm_decode_rank(rank, permutation)) return;
  const std::uint64_t center = active_slots[index];
  for (std::uint32_t site = 0u; site < kProcessiveRearmSiteCount; ++site) {
    std::uint64_t slot = 0u;
    if (processive_rearm_relative_slot(
            chunks, center, permutation, processive_rearm_offset(site),
            &slot) &&
        slot < site_count) {
      words[slot] = processive_permute_word(
          rearmed ? processive_rearm_candidate_word(site)
                  : processive_rearm_rearmed_word(site),
          permutation);
    }
  }
}

__global__ void active_processive_match_kernel(
    const SiteWord* words, std::uint8_t* matches,
    const std::uint64_t* active_slots, std::uint64_t active_count,
    std::uint64_t site_count, DeviceChunkMap chunks, bool inverse,
    const std::uint32_t* device_active_count) {
  const std::uint64_t index =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  active_count = bounded_active_count(active_count, device_active_count);
  if (index >= active_count) return;
  if (active_slots[index] >= site_count) {
    matches[index] = 0u;
    return;
  }
  if (index != 0u && active_slots[index - 1u] == active_slots[index]) {
    matches[index] = 0u;
    return;
  }
  matches[index] =
      processive_match_code(words, site_count, chunks, active_slots[index],
                            inverse);
}

__global__ void active_processive_collision_kernel(
    const SiteWord* words, const std::uint8_t* matches,
    std::uint8_t* resolved,
    const std::uint64_t* active_slots, std::uint64_t active_count,
    std::uint64_t site_count, DeviceChunkMap chunks,
    const std::uint32_t* device_active_count) {
  const std::uint64_t index =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  active_count = bounded_active_count(active_count, device_active_count);
  if (index >= active_count) return;
  const std::uint8_t decision = matches[index];
  resolved[index] = decision;
  if (decision == 0u) return;
  const std::uint64_t center = active_slots[index];
  std::uint32_t own_permutation[4]{};
  std::uint32_t own_action = 0u;
  const std::uint32_t endpoint_count =
      24u * kProcessiveReleaseActionCount;
  const std::uint32_t own_rank = (decision - 1u) % endpoint_count;
  if (!processive_decode_rank(
          own_rank, own_permutation, &own_action)) {
    resolved[index] = 0u;
    return;
  }
  const bool own_inverse = decision > endpoint_count;
  ProcessiveClaimSelection own_selection{};
  if (!processive_claim_stack_matches(
          words, site_count, chunks, center, own_permutation, own_action,
          own_inverse, &own_selection)) {
    resolved[index] = 0u;
    return;
  }
  if (!own_selection.directional) {
    for (std::uint32_t marker = 0u; marker < 4u; ++marker) {
      for (std::uint32_t path = 0u; path < 4u; ++path) {
        if (path == marker) continue;
        for (std::uint32_t waste = 0u; waste < 4u; ++waste) {
          if (waste == marker || waste == path) continue;
          std::uint32_t source_permutation[4]{};
          processive_permutation(marker, path, waste,
                                 source_permutation);
          std::uint64_t source_center = 0u;
          if (!processive_walk(chunks, center, source_permutation[1u], -6,
                               &source_center) ||
              source_center >= site_count)
            continue;
          for (std::uint64_t source_index = 0u;
               source_index < active_count; ++source_index) {
            if (active_slots[source_index] != source_center ||
                matches[source_index] == 0u)
              continue;
            std::uint32_t matched_permutation[4]{};
            std::uint32_t matched_action = 0u;
            const std::uint32_t matched_rank =
                (matches[source_index] - 1u) % endpoint_count;
            if (!processive_decode_rank(matched_rank, matched_permutation,
                                        &matched_action) ||
                !processive_action_carries(matched_action))
              continue;
            bool same_frame = true;
            for (std::uint32_t basis = 0u; basis < 4u; ++basis)
              same_frame = same_frame &&
                           matched_permutation[basis] ==
                               source_permutation[basis];
            if (!same_frame) continue;
            ProcessiveClaimSelection source_selection{};
            if (processive_claim_stack_matches(
                    words, site_count, chunks, source_center,
                    matched_permutation, matched_action,
                    matches[source_index] > endpoint_count,
                    &source_selection) && source_selection.directional) {
              resolved[index] = 0u;
              return;
            }
          }
        }
      }
    }
  }
  const std::uint32_t own_footprint_count =
      kProcessiveReleaseFootprintCount +
      (processive_action_carries(own_action) ? 1u : 0u);
  for (std::uint32_t own_index = 0u;
       own_index < own_footprint_count; ++own_index) {
    std::uint64_t shared = 0u;
    if (!processive_relative_slot(
            chunks, center, own_permutation,
            processive_collision_offset(own_index), &shared) ||
        shared >= site_count) {
      resolved[index] = 0u;
      return;
    }
    for (std::uint32_t marker = 0u; marker < 4u; ++marker) {
      for (std::uint32_t path = 0u; path < 4u; ++path) {
        if (path == marker) continue;
        for (std::uint32_t waste = 0u; waste < 4u; ++waste) {
          if (waste == marker || waste == path) continue;
          std::uint32_t other_permutation[4]{};
          processive_permutation(marker, path, waste, other_permutation);
          for (std::uint32_t other_index = 0u;
               other_index < kProcessiveReleaseFootprintCount;
               ++other_index) {
            ProcessiveReleaseOffset offset =
                processive_collision_offset(other_index);
            offset.marker = -offset.marker;
            offset.path = -offset.path;
            offset.waste = -offset.waste;
            std::uint64_t other_center = 0u;
            if (!processive_relative_slot(
                    chunks, shared, other_permutation, offset,
                    &other_center) ||
                other_center == center || other_center >= site_count)
              continue;
            bool intentional_successor = false;
            if (own_index == kProcessiveReleaseFootprintCount &&
                processive_action_carries(own_action) &&
                other_index == kProcessiveRoleIngressDigit) {
              std::uint64_t successor_center = 0u;
              std::uint64_t successor_ingress = 0u;
              if (processive_successor_geometry(
                      chunks, center, own_permutation, &successor_center,
                      &successor_ingress)) {
                intentional_successor =
                    other_center == successor_center &&
                    other_permutation[0u] == own_permutation[0u] &&
                    other_permutation[1u] == own_permutation[1u] &&
                    other_permutation[2u] == own_permutation[2u] &&
                    other_permutation[3u] == own_permutation[3u];
              }
            }
            if (!intentional_successor &&
                processive_owner_present(words, site_count, chunks,
                                         other_center, other_permutation)) {
              resolved[index] = 0u;
              return;
            }
          }
        }
      }
    }
  }
}

__global__ void active_processive_apply_kernel(
    SiteWord* words, const std::uint8_t* resolved,
    const std::uint64_t* active_slots, std::uint64_t active_count,
    std::uint64_t site_count, DeviceChunkMap chunks,
    const std::uint32_t* device_active_count) {
  const std::uint64_t index =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  active_count = bounded_active_count(active_count, device_active_count);
  if (index >= active_count || resolved[index] == 0u) return;
  const std::uint32_t endpoint_count =
      24u * kProcessiveReleaseActionCount;
  const bool released = resolved[index] > endpoint_count;
  const std::uint32_t rank = (resolved[index] - 1u) % endpoint_count;
  std::uint32_t permutation[4]{};
  std::uint32_t action = 0u;
  if (!processive_decode_rank(rank, permutation, &action)) return;
  const std::uint64_t center = active_slots[index];
  ProcessiveClaimSelection selection{};
  if (!processive_row_matches(words, site_count, chunks, center, permutation,
                              action, released) ||
      !processive_claim_stack_matches(words, site_count, chunks, center,
                                      permutation, action, released,
                                      &selection) ||
      (selection.directional && selection.digit >=
                                    kProcessiveReleaseClaimDigitCount))
    return;
  std::uint64_t slots[kProcessiveReleaseSiteCount]{};
  for (std::uint32_t site = 0u; site < kProcessiveReleaseSiteCount; ++site)
    if (!processive_relative_slot(
            chunks, center, permutation, processive_release_offset(site),
            &slots[site]) ||
        slots[site] >= site_count)
      return;
  std::uint64_t claim_slots[kProcessiveReleaseClaimDigitCount]{};
  for (std::uint32_t digit = 0u;
       digit < kProcessiveReleaseClaimDigitCount; ++digit)
    if (!processive_relative_slot(
            chunks, center, permutation,
            processive_release_claim_offset(digit), &claim_slots[digit]) ||
        claim_slots[digit] >= site_count)
      return;
  for (std::uint32_t site = 0u; site < kProcessiveReleaseSiteCount; ++site)
    words[slots[site]] = processive_permute_word(
        released ? processive_release_staged_word(action, site)
                 : processive_release_released_word(action, site),
        permutation);
  if (selection.directional) {
    const SiteWord role_word = processive_permute_word(
        processive_release_claim_word(selection.role), permutation);
    const SiteWord empty_word = processive_permute_word(
        processive_release_claim_word(ProcessiveReleaseClaim::empty),
        permutation);
    words[claim_slots[kProcessiveRoleIngressDigit]] =
        released ? role_word : empty_word;
    words[selection.endpoint] = released ? empty_word : role_word;
  }
}

__global__ void active_corner_match_kernel(
    const SiteWord* words, std::uint8_t* matches,
    const std::uint64_t* active_slots, std::uint64_t active_count,
    std::uint64_t site_count, DeviceChunkMap chunks,
    const std::uint32_t* device_active_count) {
  const std::uint64_t index =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  active_count = bounded_active_count(active_count, device_active_count);
  if (index >= active_count) return;
  if (active_slots[index] >= site_count) {
    matches[index] = 0u;
    return;
  }
  if (index != 0u && active_slots[index - 1u] == active_slots[index]) {
    matches[index] = 0u;
    return;
  }
  matches[index] =
      corner_match_code(words, site_count, chunks, active_slots[index]);
}

__global__ void active_corner_collision_kernel(
    const SiteWord* words, std::uint8_t* matches,
    const std::uint64_t* active_slots, std::uint64_t active_count,
    std::uint64_t site_count, DeviceChunkMap chunks,
    const std::uint32_t* device_active_count) {
  const std::uint64_t index =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  active_count = bounded_active_count(active_count, device_active_count);
  if (index >= active_count || matches[index] == 0u) return;
  const std::uint64_t center = active_slots[index];
  std::uint32_t own_in = 0u;
  std::uint32_t own_out = 0u;
  if (!corner_decode_rank(
          (matches[index] - 1u) % 56u, &own_in, &own_out)) {
    matches[index] = 0u;
    return;
  }
  for (std::uint32_t own_index = 0u;
       own_index < kCarrierCornerSiteCount; ++own_index) {
    std::uint64_t shared = 0u;
    if (!corner_relative_slot(
            chunks, center, own_in, own_out, own_index, &shared))
      continue;
    for (std::uint32_t incoming = 0u; incoming < 8u; ++incoming) {
      for (std::uint32_t outgoing = 0u; outgoing < 8u; ++outgoing) {
        if (incoming == outgoing) continue;
        for (std::uint32_t other_index = 0u;
             other_index < kCarrierCornerSiteCount; ++other_index) {
          std::uint64_t other_center = shared;
          if (other_index != 0u) {
            const std::uint32_t direction =
                other_index == 1u ? incoming : outgoing;
            const std::uint32_t count = other_index == 1u ? 6u : 8u;
            std::uint32_t side = 0u;
            while (side == (incoming & 3u) ||
                   side == (outgoing & 3u))
              ++side;
            const std::uint32_t side_count =
                other_index == 1u ? 2u : 3u;
            for (std::uint32_t step = 0u;
                 step < side_count; ++step) {
              if (!neighbor_slot(
                      chunks, other_center, side ^ 4u, &other_center)) {
                other_center = site_count;
                break;
              }
            }
            for (std::uint32_t step = 0u; step < count; ++step) {
              if (!neighbor_slot(chunks, other_center,
                                 direction ^ 4u, &other_center)) {
                other_center = site_count;
                break;
              }
            }
          }
          if (other_center == center || other_center >= site_count) continue;
          if (corner_owner_present(words, site_count, chunks, other_center,
                                   incoming, outgoing)) {
            matches[index] = 0u;
            return;
          }
        }
      }
    }
  }
}

__global__ void active_corner_apply_kernel(
    SiteWord* words, const std::uint8_t* matches,
    const std::uint64_t* active_slots, std::uint64_t active_count,
    const std::uint32_t* device_active_count) {
  const std::uint64_t index =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  active_count = bounded_active_count(active_count, device_active_count);
  if (index >= active_count || matches[index] == 0u) return;
  std::uint32_t incoming = 0u;
  std::uint32_t outgoing = 0u;
  if (!corner_decode_rank(
          (matches[index] - 1u) % 56u, &incoming, &outgoing))
    return;
  const std::uint64_t center = active_slots[index];
  words[center] = carrier_corner_transpose(
      words[center], incoming, outgoing);
}

using PredictionResidualAction = prediction_residual_route_toggle_detail::Action;
using PredictionResidualKind = prediction_residual_route_toggle_detail::ActionKind;

[[nodiscard]] __device__ bool prediction_site_slot(
    const DeviceChunkMap& chunks, std::uint64_t center, std::uint32_t site,
    std::uint64_t* slot) {
  if (site == 0u) { *slot = center; return true; }
  return neighbor_slot(chunks, center, site - 1u, slot);
}

[[nodiscard]] __device__ bool prediction_neighborhood(
    const SiteWord* words, std::uint64_t site_count,
    const DeviceChunkMap& chunks, std::uint64_t center,
    PredictionResidualNeighborhood* value) {
  if (center >= site_count) return false;
  value->center = words[center];
  for (std::uint32_t basis = 0u; basis < 4u; ++basis) {
    std::uint64_t slot = 0u;
    value->positive[basis] = kQ;
    value->negative[basis] = kQ;
    if (neighbor_slot(chunks, center, basis, &slot) && slot < site_count)
      value->positive[basis] = words[slot];
    if (neighbor_slot(chunks, center, basis + 4u, &slot) && slot < site_count)
      value->negative[basis] = words[slot];
  }
  return true;
}

[[nodiscard]] __device__ bool prediction_action(
    const SiteWord* words, std::uint64_t site_count,
    const DeviceChunkMap& chunks, std::uint64_t center, std::uint8_t code,
    PredictionResidualAction* action, std::uint64_t sites[9]) {
  PredictionResidualNeighborhood neighborhood{};
  if (!prediction_neighborhood(words, site_count, chunks, center, &neighborhood) ||
      code == 0u || code > 48u)
    return false;
  const auto candidate = prediction_residual_route_toggle_detail::candidate_at(
      prediction_residual_route_toggle_detail::selected_permutation(code),
      prediction_residual_route_toggle_detail::selected_negative_probe(code));
  const auto result = evaluate_prediction_residual_route_toggle(neighborhood);
  if (result.receipt.selected_candidate != code ||
      result.receipt.selected_kind > 3u)
    return false;
  *action = prediction_residual_route_toggle_detail::make_selected_action(
      candidate, static_cast<PredictionResidualKind>(result.receipt.selected_kind));
  for (std::uint32_t site = 0u; site < 9u; ++site)
    if (!prediction_site_slot(chunks, center, site, &sites[site])) return false;
  return true;
}

[[nodiscard]] __device__ bool prediction_actions_conflict(
    const PredictionResidualAction& left, const std::uint64_t left_sites[9],
    const PredictionResidualAction& right, const std::uint64_t right_sites[9]) {
  for (std::uint32_t l = 0u; l < 9u; ++l) {
    for (std::uint32_t r = 0u; r < 9u; ++r) {
      if (left_sites[l] != right_sites[r]) continue;
      if ((left.write_masks[l] & (right.read_masks[r] | right.write_masks[r])) != 0u ||
          (right.write_masks[r] & (left.read_masks[l] | left.write_masks[l])) != 0u)
        return true;
    }
  }
  return false;
}

#include "bcc32_spatial_macro_executor_tail.inl"
