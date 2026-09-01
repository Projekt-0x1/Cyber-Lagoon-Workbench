#include <cuda_runtime.h>

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

#include "bcc32_developmental_learned_receptor.hpp"
#include "bcc32_spatial_macro_executor.cuh"

namespace substrate::bcc32 {
namespace {

inline constexpr std::uint32_t kLearnedJournalInvalid = 3u;

constexpr std::uint32_t kThreads = 256u;

[[nodiscard]] std::uint32_t launch_blocks(std::uint64_t count) {
  const std::uint64_t blocks = (count + kThreads - 1u) / kThreads;
  if (blocks == 0u || blocks > std::numeric_limits<std::uint32_t>::max())
    throw std::overflow_error("learned receptor launch grid overflow");
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
    std::int32_t* coordinates[3] = {&next_x, &next_y, &next_z};
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
    std::uint64_t* destination) {
  std::uint64_t current = center;
  if (!walk(chunks, current, permutation[0u], offset.marker, &current) ||
      !walk(chunks, current, permutation[1u], offset.path, &current) ||
      !walk(chunks, current, permutation[2u], offset.waste, &current))
    return false;
  *destination = current;
  return true;
}

[[nodiscard]] __device__ bool relative_slot(
    const DeviceChunkMap& chunks, std::uint64_t center,
    const std::uint32_t permutation[4], std::uint32_t site,
    std::uint64_t* destination) {
  return shifted_slot(chunks, center, permutation,
                      developmental_append_offset(site), destination);
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

[[nodiscard]] __device__ std::uint32_t journal_state(
    const SiteWord* words, std::uint64_t site_count,
    const DeviceChunkMap& chunks, std::uint64_t center,
    const std::uint32_t permutation[4], std::uint32_t digit) {
  const std::uint32_t site = developmental_append_journal_state_site(digit);
  std::uint64_t slot = 0u;
  if (!relative_slot(chunks, center, permutation, site, &slot) ||
      slot >= site_count)
    return kLearnedJournalInvalid;
  for (std::uint32_t state = 0u; state <= 2u; ++state)
    if (words[slot] == developmental_append_journal_word(
                           site, permutation[0u], permutation[1u],
                           permutation[2u], state))
      return state;
  return kLearnedJournalInvalid;
}

[[nodiscard]] __device__ bool owner_present(
    const SiteWord* words, std::uint64_t site_count,
    const DeviceChunkMap& chunks, std::uint64_t center,
    const std::uint32_t permutation[4]) {
  // Keep these offsets literal.  NVCC 12 mis-lowered the prior runtime loop
  // over developmental_append_offset(): in the retained adult, site 0 read
  // site 18 and site 12 read site 30.  Literal expansion is checked against
  // the CPU law over the exact checkpoint and all 24 role frames.
#define BCC32_LEARNED_OWNER_SITE(site)                                      \
  do {                                                                       \
    std::uint64_t slot = 0u;                                                 \
    if (!shifted_slot(chunks, center, permutation,                           \
                      developmental_append_offset(site), &slot) ||           \
        slot >= site_count ||                                                \
        words[slot] != developmental_append_product_word(                    \
                           site, permutation[0u], permutation[1u],           \
                           permutation[2u], 0u))                             \
      return false;                                                          \
  } while (false)
  BCC32_LEARNED_OWNER_SITE(11u);
  BCC32_LEARNED_OWNER_SITE(12u);
  BCC32_LEARNED_OWNER_SITE(13u);
  BCC32_LEARNED_OWNER_SITE(30u);
  BCC32_LEARNED_OWNER_SITE(31u);
#undef BCC32_LEARNED_OWNER_SITE
#define BCC32_LEARNED_JOURNAL_DIGIT(digit, authority_state)                  \
  do {                                                                       \
    constexpr std::uint32_t first =                                          \
        kDevelopmentalAppendJournalFirst +                                   \
        (digit) * kDevelopmentalAppendAgeSitesPerDigit;                      \
    std::uint64_t anchor = 0u;                                                \
    std::uint64_t lock = 0u;                                                  \
    std::uint64_t state_slot = 0u;                                            \
    if (!shifted_slot(chunks, center, permutation,                            \
                      developmental_append_offset(first), &anchor) ||         \
        !shifted_slot(chunks, center, permutation,                            \
                      developmental_append_offset(first + 1u), &lock) ||      \
        !shifted_slot(chunks, center, permutation,                            \
                      developmental_append_offset(first + 2u),                \
                      &state_slot) ||                                         \
        anchor >= site_count || lock >= site_count ||                         \
        state_slot >= site_count ||                                           \
        words[anchor] != developmental_append_journal_word(                   \
                             first, permutation[0u], permutation[1u],         \
                             permutation[2u], 0u) ||                          \
        words[lock] != developmental_append_journal_word(                     \
                           first + 1u, permutation[0u], permutation[1u],       \
                           permutation[2u], 0u))                              \
      return false;                                                          \
    std::uint32_t state = kLearnedJournalInvalid;                            \
    for (std::uint32_t candidate_state = 0u; candidate_state <= 2u;           \
         ++candidate_state)                                                   \
      if (words[state_slot] == developmental_append_journal_word(             \
                                    first + 2u, permutation[0u],              \
                                    permutation[1u], permutation[2u],         \
                                    candidate_state))                         \
        state = candidate_state;                                              \
    if constexpr ((authority_state) != 0u) {                                  \
      if (state > kDevelopmentalAppendReceptorJournalB ||                     \
          (state != kDevelopmentalAppendReceptorJournalEmpty &&               \
           state != (authority_state)))                                       \
        return false;                                                         \
    } else if ((digit) == kDevelopmentalAppendEventJournalFirst ||            \
               (digit) == kDevelopmentalAppendEventJournalFirst + 1u) {      \
      if (state > kDevelopmentalAppendReceptorJournalB) return false;         \
    } else {                                                                  \
      if (state != kDevelopmentalAppendReceptorJournalEmpty) return false;    \
    }                                                                         \
  } while (false)
  BCC32_LEARNED_JOURNAL_DIGIT(0u, 1u);
  BCC32_LEARNED_JOURNAL_DIGIT(1u, 2u);
  BCC32_LEARNED_JOURNAL_DIGIT(2u, 0u);
  BCC32_LEARNED_JOURNAL_DIGIT(3u, 0u);
  BCC32_LEARNED_JOURNAL_DIGIT(4u, 0u);
  BCC32_LEARNED_JOURNAL_DIGIT(5u, 0u);
  BCC32_LEARNED_JOURNAL_DIGIT(6u, 0u);
  BCC32_LEARNED_JOURNAL_DIGIT(7u, 0u);
#undef BCC32_LEARNED_JOURNAL_DIGIT
  return true;
}

[[nodiscard]] __device__ std::uint64_t candidate_slot(
    const std::uint64_t* candidates, std::uint64_t index) {
  return candidates == nullptr ? index : candidates[index];
}

__global__ void learned_receptor_kernel(
    SiteWord* words, std::uint64_t candidate_count,
    std::uint64_t site_count, DeviceChunkMap chunks, bool inverse,
    const std::uint64_t* candidates,
    const std::uint32_t* device_candidate_count) {
  const std::uint64_t index =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (device_candidate_count != nullptr)
    candidate_count =
        min(candidate_count,
            static_cast<std::uint64_t>(*device_candidate_count));
  if (index >= candidate_count) return;
  const std::uint64_t center = candidate_slot(candidates, index);
  if (center >= site_count) return;
  for (std::uint32_t rank = 0u; rank < 24u; ++rank) {
    std::uint32_t permutation[4]{};
    if (!decode_permutation(rank, permutation) ||
        !owner_present(words, site_count, chunks, center, permutation))
      continue;
    std::uint64_t teacher = 0u;
    if (!shifted_slot(chunks, center, permutation,
                      developmental_append_teacher_offset(), &teacher) ||
        teacher >= site_count || words[teacher] != kQuiescentWord)
      continue;
    std::uint64_t inlet = 0u;
    if (!relative_slot(chunks, center, permutation,
                       kDevelopmentalAppendReceptorInletSite, &inlet) ||
        inlet >= site_count)
      continue;
    const SiteWord inlet_bit = carrier_bit(permutation[1u]);
    if ((!inverse && (words[inlet] & inlet_bit) == 0u) ||
        (inverse && (words[inlet] & inlet_bit) != 0u))
      continue;
    std::uint64_t ports[kDevelopmentalAppendReceptorLegCount]{};
    bool ports_valid = true;
    for (std::uint32_t leg = 0u;
         leg < kDevelopmentalAppendReceptorLegCount; ++leg)
      ports_valid = ports_valid &&
                    shifted_slot(chunks, center, permutation,
                                 developmental_append_receptor_port_offset(leg),
                                 &ports[leg]) &&
                    ports[leg] < site_count;
    if (!ports_valid) continue;

    std::uint32_t leg = kDevelopmentalAppendReceptorLegCount;
    constexpr std::uint32_t event_digit =
        kDevelopmentalAppendEventJournalFirst;
    if (!inverse) {
      for (std::uint32_t candidate_leg = 0u;
           candidate_leg < kDevelopmentalAppendReceptorLegCount;
           ++candidate_leg) {
        const std::uint32_t basis = developmental_append_receptor_basis(
            candidate_leg, permutation[0u], permutation[2u]);
        if (words[ports[candidate_leg]] ==
            (kQuiescentWord ^ carrier_bit(basis))) {
          if (leg != kDevelopmentalAppendReceptorLegCount) {
            leg = kDevelopmentalAppendReceptorLegCount;
            break;
          }
          leg = candidate_leg;
        } else if (words[ports[candidate_leg]] != kQuiescentWord) {
          leg = kDevelopmentalAppendReceptorLegCount;
          break;
        }
      }
      if (leg == kDevelopmentalAppendReceptorLegCount) continue;
      if (journal_state(words, site_count, chunks, center, permutation,
                        event_digit) !=
          kDevelopmentalAppendReceptorJournalEmpty)
        continue;
    } else {
      for (const std::uint64_t port : ports)
        ports_valid = ports_valid && words[port] == kQuiescentWord;
      if (!ports_valid) continue;
      const std::uint32_t state = journal_state(
          words, site_count, chunks, center, permutation, event_digit);
      if (state == kDevelopmentalAppendReceptorJournalA ||
          state == kDevelopmentalAppendReceptorJournalB)
        leg = state - 1u;
    }
    if (leg == kDevelopmentalAppendReceptorLegCount ||
        journal_state(words, site_count, chunks, center, permutation, leg) !=
            leg + 1u)
      continue;
    const std::uint32_t authority_site =
        developmental_append_receptor_authority_site(leg);
    std::uint64_t authority = 0u;
    if (!relative_slot(chunks, center, permutation, authority_site,
                       &authority) ||
        authority >= site_count ||
        words[authority] != developmental_append_journal_word(
                                authority_site, permutation[0u],
                                permutation[1u], permutation[2u], 0u))
      continue;
    const std::uint32_t event_site =
        developmental_append_journal_state_site(event_digit);
    std::uint64_t event = 0u;
    if (!relative_slot(chunks, center, permutation, event_site, &event) ||
        event >= site_count)
      continue;
    const std::uint32_t basis = developmental_append_receptor_basis(
        leg, permutation[0u], permutation[2u]);
    words[ports[leg]] ^= carrier_bit(basis);
    words[inlet] ^= inlet_bit;
    words[event] = developmental_append_journal_word(
        event_site, permutation[0u], permutation[1u], permutation[2u],
        inverse ? kDevelopmentalAppendReceptorJournalEmpty : leg + 1u);
    return;
  }
}

}  // namespace

namespace {

void launch_developmental_learned_receptor_impl(
    SiteWord* words, std::uint8_t* scratch, std::uint64_t site_count,
    const DeviceChunkMap& chunks, bool inverse,
    const std::uint64_t* active_slots, std::uint64_t active_count,
    const std::uint32_t* device_active_count, cudaStream_t stream) {
  (void)scratch;
  const std::uint64_t candidate_count =
      active_slots == nullptr ? site_count : active_count;
  if (candidate_count == 0u) return;
  learned_receptor_kernel<<<launch_blocks(candidate_count), kThreads, 0,
                            stream>>>(words, candidate_count, site_count,
                                      chunks, inverse, active_slots,
                                      device_active_count);
  check_cuda(cudaGetLastError(),
             inverse ? "launch inverse developmental learned receptor"
                     : "launch developmental learned receptor");
}

}  // namespace

void launch_developmental_learned_receptor_cuda(
    SiteWord* words, std::uint8_t* scratch, std::uint64_t site_count,
    const DeviceChunkMap& chunks, bool inverse,
    const std::uint64_t* active_slots, std::uint64_t active_count,
    cudaStream_t stream) {
  launch_developmental_learned_receptor_impl(
      words, scratch, site_count, chunks, inverse, active_slots, active_count,
      nullptr, stream);
}

void launch_developmental_learned_receptor_graph_safe_cuda(
    SiteWord* words, std::uint8_t* scratch, std::uint64_t site_count,
    const DeviceChunkMap& chunks, bool inverse,
    const std::uint64_t* active_slots,
    const std::uint32_t* device_active_count, std::uint32_t capacity,
    cudaStream_t stream) {
  launch_developmental_learned_receptor_impl(
      words, scratch, site_count, chunks, inverse, active_slots, capacity,
      device_active_count, stream);
}

}  // namespace substrate::bcc32
