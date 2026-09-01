#include <cuda_runtime.h>

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

#include "bcc32_developmental_credit_service.hpp"
#include "bcc32_law.cuh"
#include "bcc32_spatial_macro_executor.cuh"

namespace substrate::bcc32 {
namespace {

constexpr std::uint32_t kThreads = 256u;
constexpr std::uint8_t kServiceMatch = 0x40u;

[[nodiscard]] std::uint32_t launch_blocks(std::uint64_t count) {
  const std::uint64_t blocks = (count + kThreads - 1u) / kThreads;
  if (blocks == 0u || blocks > std::numeric_limits<std::uint32_t>::max())
    throw std::overflow_error("developmental credit-service launch grid overflow");
  return static_cast<std::uint32_t>(blocks);
}

void check_cuda(cudaError_t status, const char* operation) {
  if (status != cudaSuccess)
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(status));
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

[[nodiscard]] __device__ bool hop_chunk(const DeviceChunkMap& chunks,
                                         std::uint32_t* chunk,
                                         std::uint32_t direction) {
  const std::int32_t next = chunks.slots[*chunk].bcc_neighbors[direction];
  if (next < 0 || static_cast<std::uint32_t>(next) >= chunks.chunk_count)
    return false;
  *chunk = static_cast<std::uint32_t>(next);
  return true;
}

[[nodiscard]] __device__ bool neighbor_slot(const DeviceChunkMap& chunks,
                                             std::uint64_t source,
                                             std::uint32_t direction,
                                             std::uint64_t* destination) {
  std::uint32_t chunk = static_cast<std::uint32_t>(source / kChunkSites);
  if (chunk >= chunks.chunk_count) return false;
  std::uint32_t x = 0u;
  std::uint32_t y = 0u;
  std::uint32_t z = 0u;
  decode_local(static_cast<std::uint32_t>(source % kChunkSites), &x, &y, &z);
  const Int3 step = direction_offset(static_cast<Direction>(direction));
  std::int32_t next_x = static_cast<std::int32_t>(x) + step.x;
  std::int32_t next_y = static_cast<std::int32_t>(y) + step.y;
  std::int32_t next_z = static_cast<std::int32_t>(z) + step.z;
  const bool all_negative = next_x < 0 && next_y < 0 && next_z < 0;
  const bool all_positive =
      next_x >= static_cast<std::int32_t>(kChunkEdge) &&
      next_y >= static_cast<std::int32_t>(kChunkEdge) &&
      next_z >= static_cast<std::int32_t>(kChunkEdge);
  if (direction == 3u && all_negative) {
    if (!hop_chunk(chunks, &chunk, 3u)) return false;
    next_x += kChunkEdge;
    next_y += kChunkEdge;
    next_z += kChunkEdge;
  } else if (direction == 7u && all_positive) {
    if (!hop_chunk(chunks, &chunk, 7u)) return false;
    next_x -= kChunkEdge;
    next_y -= kChunkEdge;
    next_z -= kChunkEdge;
  } else {
    std::int32_t* coordinates[3]{&next_x, &next_y, &next_z};
    for (std::uint32_t axis = 0u; axis < 3u; ++axis) {
      if (*coordinates[axis] < 0) {
        if (!hop_chunk(chunks, &chunk, axis + 4u)) return false;
        *coordinates[axis] += kChunkEdge;
      } else if (*coordinates[axis] >=
                 static_cast<std::int32_t>(kChunkEdge)) {
        if (!hop_chunk(chunks, &chunk, axis)) return false;
        *coordinates[axis] -= kChunkEdge;
      }
    }
  }
  if (next_x < 0 || next_y < 0 || next_z < 0 ||
      next_x >= static_cast<std::int32_t>(kChunkEdge) ||
      next_y >= static_cast<std::int32_t>(kChunkEdge) ||
      next_z >= static_cast<std::int32_t>(kChunkEdge))
    return false;
  *destination = site_slot(
      chunk, encode_local(static_cast<std::uint32_t>(next_x),
                          static_cast<std::uint32_t>(next_y),
                          static_cast<std::uint32_t>(next_z)));
  return true;
}

[[nodiscard]] __device__ bool walk(const DeviceChunkMap& chunks,
                                    std::uint64_t start,
                                    std::uint32_t basis,
                                    std::int32_t steps,
                                    std::uint64_t* destination) {
  std::uint64_t current = start;
  const std::uint32_t direction = steps < 0 ? basis + 4u : basis;
  const std::uint32_t count =
      static_cast<std::uint32_t>(steps < 0 ? -steps : steps);
  for (std::uint32_t index = 0u; index < count; ++index)
    if (!neighbor_slot(chunks, current, direction, &current)) return false;
  *destination = current;
  return true;
}

[[nodiscard]] __device__ bool shifted_slot(
    const DeviceChunkMap& chunks, std::uint64_t center,
    const std::uint32_t permutation[4], DevelopmentalAppendOffset offset,
    bool negate, std::uint64_t* destination) {
  const std::int32_t sign = negate ? -1 : 1;
  std::uint64_t current = center;
  if (!walk(chunks, current, permutation[0u], sign * offset.marker, &current) ||
      !walk(chunks, current, permutation[1u], sign * offset.path, &current) ||
      !walk(chunks, current, permutation[2u], sign * offset.waste, &current))
    return false;
  *destination = current;
  return true;
}

[[nodiscard]] __device__ bool relative_slot(
    const DeviceChunkMap& chunks, std::uint64_t center,
    const std::uint32_t permutation[4], std::uint32_t site,
    std::uint64_t* destination) {
  return shifted_slot(chunks, center, permutation,
                      developmental_append_offset(site), false, destination);
}

[[nodiscard]] __device__ bool offset_slot(const DeviceChunkMap& chunks,
                                           std::uint64_t start, Int3 offset,
                                           bool negate,
                                           std::uint64_t* destination) {
  const std::int32_t sign = negate ? -1 : 1;
  std::uint64_t current = start;
  if (!walk(chunks, current, 0u, sign * offset.x, &current) ||
      !walk(chunks, current, 1u, sign * offset.y, &current) ||
      !walk(chunks, current, 2u, sign * offset.z, &current))
    return false;
  *destination = current;
  return true;
}

[[nodiscard]] __device__ bool decode_permutation(
    std::uint32_t rank, std::uint32_t permutation[4]) {
  std::uint32_t current = 0u;
  for (std::uint32_t marker = 0u; marker < 4u; ++marker)
    for (std::uint32_t path = 0u; path < 4u; ++path) {
      if (path == marker) continue;
      for (std::uint32_t waste = 0u; waste < 4u; ++waste) {
        if (waste == marker || waste == path) continue;
        if (current++ != rank) continue;
        permutation[0u] = marker;
        permutation[1u] = path;
        permutation[2u] = waste;
        for (std::uint32_t basis = 0u; basis < 4u; ++basis)
          if (basis != marker && basis != path && basis != waste)
            permutation[3u] = basis;
        return true;
      }
    }
  return false;
}

[[nodiscard]] __device__ bool base_owner_present(
    const SiteWord* words, std::uint64_t site_count,
    const DeviceChunkMap& chunks, std::uint64_t center,
    const std::uint32_t permutation[4]) {
  constexpr std::uint32_t signature[]{0u, 11u, 12u, 13u, 30u, 31u};
  for (const std::uint32_t site : signature) {
    std::uint64_t slot = 0u;
    if (!relative_slot(chunks, center, permutation, site, &slot) ||
        slot >= site_count ||
        (words[slot] & ~kCarrierMask) !=
            (developmental_append_product_word(
                 site, permutation[0u], permutation[1u], permutation[2u], 0u) &
             ~kCarrierMask))
      return false;
  }
  return true;
}

[[nodiscard]] __device__ bool ring_enabled(
    const SiteWord* words, std::uint64_t site_count,
    const DeviceChunkMap& chunks, std::uint64_t center,
    const std::uint32_t permutation[4]) {
  std::uint64_t enable = 0u;
  return relative_slot(chunks, center, permutation,
                       kDevelopmentalCreditServiceEnableSite, &enable) &&
         enable < site_count &&
         developmental_credit_service_enable_word_matches(
             words[enable], permutation[0u], permutation[1u],
             permutation[2u]);
}

[[nodiscard]] __device__ bool transaction_live(
    const SiteWord* words, std::uint64_t site_count,
    const DeviceChunkMap& chunks, std::uint64_t center,
    const std::uint32_t permutation[4], std::uint32_t ring) {
  const std::uint32_t leg = developmental_credit_service_leg(ring);
  const SiteWord filled = developmental_append_escrow_word(
      true, leg, permutation[0u], permutation[2u]);
  if (developmental_credit_service_reject_source_ring(ring) ||
      developmental_credit_service_reject_teacher_ring(ring) ||
      developmental_credit_service_reject_clock_ring(ring)) {
    const std::uint32_t bank = developmental_credit_service_reject_bank(ring);
    std::uint64_t clock = 0u;
    std::uint64_t reject = 0u;
    return shifted_slot(
               chunks, center, permutation,
               developmental_append_clock_escrow_offset(leg, bank), false,
               &clock) &&
           shifted_slot(
               chunks, center, permutation,
               developmental_append_reject_escrow_offset(leg, bank), false,
               &reject) &&
           clock < site_count && reject < site_count &&
           words[clock] == filled && words[reject] == filled;
  }
  for (std::uint32_t bank = 0u;
       bank < kDevelopmentalAppendWitnessBankCount; ++bank) {
    std::uint64_t clock = 0u;
    std::uint64_t reject = 0u;
    if (shifted_slot(
            chunks, center, permutation,
            developmental_append_clock_escrow_offset(leg, bank), false,
            &clock) &&
        shifted_slot(
            chunks, center, permutation,
            developmental_append_reject_escrow_offset(leg, bank), false,
            &reject) &&
        clock < site_count && reject < site_count &&
        words[clock] == filled && words[reject] == kQuiescentWord)
      return true;
  }
  return false;
}

[[nodiscard]] __device__ std::uint64_t candidate_slot(
    const std::uint64_t* candidates, std::uint64_t index) {
  return candidates == nullptr ? index : candidates[index];
}

__global__ void service_match_kernel(
    const SiteWord* words, std::uint8_t* scratch,
    std::uint64_t candidate_count, std::uint64_t site_count,
    DeviceChunkMap chunks, const std::uint64_t* candidates,
    const std::uint32_t* device_candidate_count) {
  const std::uint64_t index =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (device_candidate_count != nullptr)
    candidate_count =
        min(candidate_count,
            static_cast<std::uint64_t>(*device_candidate_count));
  if (index >= candidate_count) return;
  const std::uint64_t target = candidate_slot(candidates, index);
  if (target >= site_count) return;
  const SiteWord target_word = words[target];
  // Rank/ring/corner-invariant necessary condition for the centre test below.
  // carrier_corner_center_matches() rejects any word carrying a structural
  // channel, and needs exactly one of its two named carrier bits occupied.
  // carrier_bit(d) == 1u << d for d < 8, so the eight named bits are exactly
  // kCarrierMask: a word with no carrier bit leaves both named bits empty, and
  // a word with every carrier bit occupied leaves both named bits full. Either
  // way `incoming_occupied != outgoing_occupied` is false for every ordered
  // (incoming, outgoing) pair, so the 24x36x4 corner sweep cannot raise
  // match_count and the kernel would write the same zero. kQuiescentWord ==
  // kCarrierMask, so the quiescent bath word covering essentially the whole
  // aperture takes the third branch. Skipping the sweep keeps the emitted
  // scratch byte bit-identical while making the credit-service sweep cost
  // track the occupied carrier support instead of the aperture.
  if ((target_word & ~kCarrierMask) != 0u || (target_word & kCarrierMask) == 0u ||
      (target_word & kCarrierMask) == kCarrierMask) {
    scratch[target] = 0u;
    return;
  }
  std::uint32_t match_count = 0u;
  std::uint32_t selected_incoming = 0u;
  std::uint32_t selected_outgoing = 0u;
  for (std::uint32_t rank = 0u; rank < 24u; ++rank) {
    std::uint32_t permutation[4]{};
    if (!decode_permutation(rank, permutation)) continue;
    for (std::uint32_t ring = 0u;
         ring < kDevelopmentalCreditServiceRingCount; ++ring)
      for (std::uint32_t corner = 0u; corner < 4u; ++corner) {
        const std::uint32_t incoming =
            developmental_credit_service_corner_incoming(
                ring, corner, permutation[0u], permutation[1u],
                permutation[2u]);
        const std::uint32_t outgoing =
            developmental_credit_service_corner_outgoing(
                ring, corner, permutation[0u], permutation[1u],
                permutation[2u]);
        if (!carrier_corner_center_matches(target_word, incoming, outgoing))
          continue;
        const Int3 offset = developmental_credit_service_corner(
            ring, corner, permutation[0u], permutation[1u], permutation[2u]);
        std::uint64_t center = 0u;
        if (!offset_slot(chunks, target, offset, true, &center) ||
            !base_owner_present(words, site_count, chunks, center,
                                permutation) ||
            !ring_enabled(words, site_count, chunks, center, permutation) ||
            !transaction_live(words, site_count, chunks, center, permutation,
                              ring))
          continue;
        ++match_count;
        selected_incoming = incoming;
        selected_outgoing = outgoing;
      }
  }
  scratch[target] = match_count == 1u
                        ? static_cast<std::uint8_t>(
                              kServiceMatch | selected_incoming |
                              (selected_outgoing << 3u))
                        : 0u;
}

__global__ void service_apply_kernel(SiteWord* words,
                                     const std::uint8_t* scratch,
                                     std::uint64_t candidate_count,
                                     std::uint64_t site_count,
                                     const std::uint64_t* candidates,
                                     const std::uint32_t* device_candidate_count) {
  const std::uint64_t index =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (device_candidate_count != nullptr)
    candidate_count =
        min(candidate_count,
            static_cast<std::uint64_t>(*device_candidate_count));
  if (index >= candidate_count) return;
  const std::uint64_t target = candidate_slot(candidates, index);
  if (target >= site_count) return;
  const std::uint8_t match = scratch[target];
  if ((match & kServiceMatch) == 0u) return;
  const std::uint32_t incoming = match & 7u;
  const std::uint32_t outgoing = (match >> 3u) & 7u;
  words[target] = carrier_corner_transpose(words[target], incoming, outgoing);
}

}  // namespace

namespace {

void launch_developmental_credit_service_impl(
    SiteWord* words, std::uint8_t* scratch, std::uint64_t site_count,
    const DeviceChunkMap& chunks, bool inverse,
    const std::uint64_t* active_slots, std::uint64_t active_count,
    const std::uint32_t* device_active_count, cudaStream_t stream) {
  (void)inverse;
  const std::uint64_t candidate_count =
      active_slots == nullptr ? site_count : active_count;
  if (candidate_count == 0u) return;
  const std::uint32_t blocks = launch_blocks(candidate_count);
  service_match_kernel<<<blocks, kThreads, 0, stream>>>(
      words, scratch, candidate_count, site_count, chunks, active_slots,
      device_active_count);
  check_cuda(cudaGetLastError(),
             "launch developmental credit-service match");
  service_apply_kernel<<<blocks, kThreads, 0, stream>>>(
      words, scratch, candidate_count, site_count, active_slots,
      device_active_count);
  check_cuda(cudaGetLastError(),
             "launch developmental credit-service apply");
}

}  // namespace

void launch_developmental_credit_service_cuda(
    SiteWord* words, std::uint8_t* scratch, std::uint64_t site_count,
    const DeviceChunkMap& chunks, bool inverse,
    const std::uint64_t* active_slots, std::uint64_t active_count,
    cudaStream_t stream) {
  launch_developmental_credit_service_impl(
      words, scratch, site_count, chunks, inverse, active_slots, active_count,
      nullptr, stream);
}

void launch_developmental_credit_service_graph_safe_cuda(
    SiteWord* words, std::uint8_t* scratch, std::uint64_t site_count,
    const DeviceChunkMap& chunks, bool inverse,
    const std::uint64_t* active_slots,
    const std::uint32_t* device_active_count, std::uint32_t capacity,
    cudaStream_t stream) {
  launch_developmental_credit_service_impl(
      words, scratch, site_count, chunks, inverse, active_slots, capacity,
      device_active_count, stream);
}

}  // namespace substrate::bcc32
