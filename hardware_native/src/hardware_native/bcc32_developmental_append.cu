#include <cuda_runtime.h>

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

#include "bcc32_developmental_append.hpp"
#include "bcc32_spatial_macro_executor.cuh"

namespace substrate::bcc32 {
namespace {

constexpr std::uint32_t kThreads = 256u;

[[nodiscard]] std::uint32_t launch_blocks(std::uint64_t count) {
  const std::uint64_t blocks = (count + kThreads - 1u) / kThreads;
  if (blocks == 0u || blocks > std::numeric_limits<std::uint32_t>::max())
    throw std::overflow_error("developmental append launch grid overflow");
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
                                    std::uint64_t start, std::uint32_t basis,
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
    const std::uint32_t permutation[4], std::uint32_t index,
    std::uint64_t* destination) {
  return shifted_slot(chunks, center, permutation,
                      developmental_append_offset(index), false, destination);
}

[[nodiscard]] __device__ bool decode_permutation(
    std::uint32_t rank, std::uint32_t permutation[4]) {
  std::uint32_t current = 0u;
  for (std::uint32_t marker = 0u; marker < 4u; ++marker) {
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
  }
  return false;
}

[[nodiscard]] __device__ bool exact_preimage(
    const SiteWord* words, std::uint64_t site_count,
    const DeviceChunkMap& chunks, std::uint64_t center,
    const std::uint32_t permutation[4]) {
  for (std::uint32_t index = 0u; index < kDevelopmentalAppendSiteCount;
       ++index) {
    std::uint64_t slot = 0u;
    if (!relative_slot(chunks, center, permutation, index, &slot) ||
        slot >= site_count ||
        words[slot] != developmental_append_word(
                           false, index, permutation[0u], permutation[1u],
                           permutation[2u]))
      return false;
  }
  return true;
}

[[nodiscard]] __device__ bool decode_product_age(
    const SiteWord* words, std::uint64_t site_count,
    const DeviceChunkMap& chunks, std::uint64_t center,
    const std::uint32_t permutation[4], std::uint64_t* age) {
  std::uint64_t decoded = 0u;
  for (std::uint32_t index = 0u; index < kDevelopmentalAppendAgeFirst;
       ++index) {
    if (index == 10u) continue;  // mutable processive/receptor inlet
    std::uint64_t slot = 0u;
    if (!relative_slot(chunks, center, permutation, index, &slot) ||
        slot >= site_count ||
        words[slot] != developmental_append_product_word(
                           index, permutation[0u], permutation[1u],
                           permutation[2u], 0u)) {
      if (index < kDevelopmentalAppendChildHead ||
          !developmental_append_handoff_pair(
              index, index - kDevelopmentalAppendChildHead) ||
          words[slot] != developmental_append_product_word(
                             index - kDevelopmentalAppendChildHead,
                             permutation[0u], permutation[1u],
                             permutation[2u], 0u))
        return false;
    }
  }
  for (std::uint32_t digit = 0u;
       digit < kDevelopmentalAppendJournalDigitCount; ++digit) {
    const std::uint32_t first = kDevelopmentalAppendJournalFirst +
                                digit * kDevelopmentalAppendAgeSitesPerDigit;
    for (std::uint32_t component = 0u; component < 2u; ++component) {
      std::uint64_t slot = 0u;
      if (!relative_slot(chunks, center, permutation, first + component,
                         &slot) ||
          slot >= site_count ||
          words[slot] != developmental_append_journal_word(
                             first + component, permutation[0u],
                             permutation[1u], permutation[2u], 0u))
        return false;
    }
    std::uint64_t state_slot = 0u;
    if (!relative_slot(chunks, center, permutation, first + 2u,
                       &state_slot) ||
        state_slot >= site_count)
      return false;
    bool valid = false;
    for (std::uint32_t encoded = 0u; encoded <= 2u; ++encoded)
      valid = valid || words[state_slot] ==
                           developmental_append_journal_word(
                               first + 2u, permutation[0u], permutation[1u],
                               permutation[2u], encoded);
    if (!valid) return false;
  }
  for (std::uint32_t digit = 0u;
       digit < kDevelopmentalAppendAgeDigitCount; ++digit) {
    const std::uint32_t first = kDevelopmentalAppendAgeFirst +
                                digit * kDevelopmentalAppendAgeSitesPerDigit;
    for (std::uint32_t component = 0u; component < 2u; ++component) {
      std::uint64_t slot = 0u;
      if (!relative_slot(chunks, center, permutation, first + component,
                         &slot) ||
          slot >= site_count ||
          words[slot] != developmental_append_product_word(
                             first + component, permutation[0u],
                             permutation[1u], permutation[2u], 0u))
        return false;
    }
    std::uint64_t digit_slot = 0u;
    if (!relative_slot(chunks, center, permutation, first + 2u,
                       &digit_slot) ||
        digit_slot >= site_count)
      return false;
    std::uint32_t encoded = 0u;
    while (encoded < 4u &&
           words[digit_slot] != developmental_append_product_word(
                                      first + 2u, permutation[0u],
                                      permutation[1u], permutation[2u],
                                      std::uint64_t{encoded} <<
                                          (2u * digit)))
      ++encoded;
    if (encoded == 4u) return false;
    decoded |= std::uint64_t{encoded} << (2u * digit);
  }
  *age = decoded;
  return true;
}

[[nodiscard]] __device__ bool owner_present(
    const SiteWord* words, std::uint64_t site_count,
    const DeviceChunkMap& chunks, std::uint64_t center,
    const std::uint32_t permutation[4], bool product) {
  const std::uint32_t locks[2] = {
      product ? 11u
              : kDevelopmentalAppendParentLockFirst,
      product ? 12u
              : kDevelopmentalAppendParentLockSecond};
  for (const std::uint32_t lock : locks) {
    std::uint64_t slot = 0u;
    if (!relative_slot(chunks, center, permutation, lock, &slot) ||
        slot >= site_count ||
        words[slot] != developmental_append_word(
                           product, lock, permutation[0u], permutation[1u],
                           permutation[2u]))
      return false;
  }
  if (product) {
    constexpr std::uint32_t signature_sites[9] = {
        0u, 11u, 12u, 13u, 17u, 30u, 31u,
        kDevelopmentalAppendJournalFirst,
        kDevelopmentalAppendJournalFirst + 1u};
    for (const std::uint32_t signature_site : signature_sites) {
      std::uint64_t slot = 0u;
      if (!relative_slot(chunks, center, permutation, signature_site, &slot) ||
          slot >= site_count ||
          words[slot] != developmental_append_word(
                             true, signature_site, permutation[0u],
                             permutation[1u], permutation[2u]))
        return false;
    }
  }
  return true;
}

[[nodiscard]] __device__ std::uint8_t match_code(
    const SiteWord* words, std::uint64_t site_count,
    const DeviceChunkMap& chunks, std::uint64_t center, bool inverse) {
  (void)inverse;
  for (std::uint32_t rank = 0u; rank < 24u; ++rank) {
    std::uint32_t permutation[4]{};
    if (!decode_permutation(rank, permutation)) continue;
    if (exact_preimage(words, site_count, chunks, center, permutation))
      return static_cast<std::uint8_t>(rank + 1u);
    std::uint64_t age = 0u;
    if (decode_product_age(words, site_count, chunks, center, permutation,
                           &age))
      return static_cast<std::uint8_t>(rank + 25u);
  }
  return 0u;
}

[[nodiscard]] __device__ bool lineage_handoff(
    const DeviceChunkMap& chunks, std::uint64_t first_center,
    const std::uint32_t first_permutation[4], bool first_product,
    std::uint64_t second_center, const std::uint32_t second_permutation[4],
    bool second_product) {
  (void)first_product;
  (void)second_product;
  for (std::uint32_t index = 0u; index < 4u; ++index)
    if (first_permutation[index] != second_permutation[index]) return false;
  std::uint64_t parent_center = first_center;
  std::uint64_t child_center = second_center;
  const std::uint32_t* permutation = first_permutation;
  std::uint64_t expected_child = 0u;
  if (!relative_slot(chunks, parent_center, permutation,
                     kDevelopmentalAppendChildHead, &expected_child) ||
      expected_child != child_center) {
    parent_center = second_center;
    child_center = first_center;
    permutation = second_permutation;
    if (!relative_slot(chunks, parent_center, permutation,
                       kDevelopmentalAppendChildHead, &expected_child) ||
        expected_child != child_center)
      return false;
  }

  // A center relation alone is insufficient: enlarging the append footprint
  // can introduce a new physical alias.  Enumerate the two exact footprints
  // and permit only the nine represented parent->child handoff coordinates.
  std::uint32_t shared = 0u;
  for (std::uint32_t left = 0u;
       left < kDevelopmentalAppendSiteCount; ++left) {
    std::uint64_t parent_slot = 0u;
    if (!relative_slot(chunks, parent_center, permutation, left,
                       &parent_slot))
      return false;
    for (std::uint32_t right = 0u;
         right < kDevelopmentalAppendSiteCount; ++right) {
      std::uint64_t child_slot = 0u;
      if (!relative_slot(chunks, child_center, permutation, right,
                         &child_slot))
        return false;
      if (parent_slot != child_slot) continue;
      if (!developmental_append_handoff_pair(left, right)) return false;
      ++shared;
    }
  }
  return shared == kDevelopmentalAppendHandoffSiteCount;
}

[[nodiscard]] __device__ bool collides_with_foreign_owner(
    const SiteWord* words, std::uint64_t site_count,
    const DeviceChunkMap& chunks, std::uint64_t own_center,
    const std::uint32_t own_permutation[4], bool own_product, bool inverse) {
  for (std::uint32_t own_site = 0u;
       own_site < kDevelopmentalAppendSiteCount; ++own_site) {
    std::uint64_t shared_slot = 0u;
    if (!relative_slot(chunks, own_center, own_permutation, own_site,
                       &shared_slot))
      return true;
    for (const bool product : {false, true}) {
      for (std::uint32_t rank = 0u; rank < 24u; ++rank) {
        std::uint32_t permutation[4]{};
        if (!decode_permutation(rank, permutation)) continue;
        for (std::uint32_t foreign_site = 0u;
             foreign_site < kDevelopmentalAppendSiteCount; ++foreign_site) {
          std::uint64_t foreign_center = 0u;
          if (!shifted_slot(chunks, shared_slot, permutation,
                            developmental_append_offset(foreign_site), true,
                            &foreign_center) ||
              foreign_center >= site_count ||
              !owner_present(words, site_count, chunks, foreign_center,
                             permutation, product))
            continue;
          bool same = own_center == foreign_center &&
                      own_product == product;
          for (std::uint32_t index = 0u; index < 4u && same; ++index)
            same = own_permutation[index] == permutation[index];
          if (same) continue;
          if (!lineage_handoff(chunks, own_center, own_permutation,
                               own_product, foreign_center, permutation,
                               product))
            return true;
          if (inverse && !own_product && product) {
            std::uint64_t parent_age = 0u;
            if (decode_product_age(words, site_count, chunks, foreign_center,
                                   permutation, &parent_age) &&
                parent_age == 0u)
              return true;
          }
        }
      }
    }
  }
  return false;
}

[[nodiscard]] __device__ std::uint64_t candidate_slot(
    const std::uint64_t* candidates, std::uint64_t index) {
  return candidates == nullptr ? index : candidates[index];
}

__global__ void append_match_kernel(
    const SiteWord* words, std::uint8_t* matches, std::uint64_t candidate_count,
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
  matches[index] = center < site_count
                       ? match_code(words, site_count, chunks, center, inverse)
                       : 0u;
}

__global__ void append_collision_kernel(
    const SiteWord* words, std::uint8_t* matches, std::uint64_t candidate_count,
    std::uint64_t site_count, DeviceChunkMap chunks, bool inverse,
    const std::uint64_t* candidates,
    const std::uint32_t* device_candidate_count) {
  const std::uint64_t index =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (device_candidate_count != nullptr)
    candidate_count =
        min(candidate_count,
            static_cast<std::uint64_t>(*device_candidate_count));
  if (index >= candidate_count || matches[index] == 0u) return;
  const std::uint64_t center = candidate_slot(candidates, index);
  std::uint32_t permutation[4]{};
  const bool product = matches[index] > 24u;
  if (!decode_permutation((matches[index] - 1u) % 24u, permutation) ||
      collides_with_foreign_owner(words, site_count, chunks, center,
                                  permutation, product, inverse))
    matches[index] = 0u;
}

__global__ void append_apply_kernel(
    SiteWord* words, const std::uint8_t* matches,
    std::uint64_t candidate_count, std::uint64_t site_count,
    DeviceChunkMap chunks, bool inverse, const std::uint64_t* candidates,
    const std::uint32_t* device_candidate_count) {
  const std::uint64_t index =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (device_candidate_count != nullptr)
    candidate_count =
        min(candidate_count,
            static_cast<std::uint64_t>(*device_candidate_count));
  if (index >= candidate_count || matches[index] == 0u) return;
  const std::uint64_t center = candidate_slot(candidates, index);
  std::uint32_t permutation[4]{};
  const bool product = matches[index] > 24u;
  if (!decode_permutation((matches[index] - 1u) % 24u, permutation)) return;
  std::uint64_t age = 0u;
  if (product &&
      !decode_product_age(words, site_count, chunks, center, permutation,
                          &age))
    return;
  bool after_product = true;
  std::uint64_t after_age = 0u;
  if (!product) {
    after_age = inverse ? kDevelopmentalAppendMaxAge : 0u;
  } else if ((!inverse && age == kDevelopmentalAppendMaxAge) ||
             (inverse && age == 0u)) {
    after_product = false;
  } else {
    after_age = inverse ? age - 1u : age + 1u;
  }
  for (std::uint32_t site = 0u; site < kDevelopmentalAppendSiteCount;
       ++site) {
    if (product && after_product &&
        (site < kDevelopmentalAppendAgeFirst ||
         site >= kDevelopmentalAppendJournalFirst ||
         (site - kDevelopmentalAppendAgeFirst) %
                 kDevelopmentalAppendAgeSitesPerDigit !=
             2u))
      continue;
    std::uint64_t slot = 0u;
    if (!relative_slot(chunks, center, permutation, site, &slot) ||
        slot >= site_count)
      return;
    words[slot] = after_product
                      ? developmental_append_product_word(
                            site, permutation[0u], permutation[1u],
                            permutation[2u], after_age)
                      : developmental_append_word(
                            false, site, permutation[0u], permutation[1u],
                            permutation[2u]);
  }
}

[[nodiscard]] __device__ bool receptor_port_slot(
    const DeviceChunkMap& chunks, std::uint64_t center,
    const std::uint32_t permutation[4], std::uint32_t leg,
    std::uint64_t* destination) {
  return shifted_slot(chunks, center, permutation,
                      developmental_append_receptor_port_offset(leg), false,
                      destination);
}

[[nodiscard]] __device__ bool receptor_teacher_slot(
    const DeviceChunkMap& chunks, std::uint64_t center,
    const std::uint32_t permutation[4], std::uint64_t* destination) {
  return shifted_slot(chunks, center, permutation,
                      developmental_append_teacher_offset(), false,
                      destination);
}

[[nodiscard]] __device__ bool receptor_accepted_source_slot(
    const DeviceChunkMap& chunks, std::uint64_t center,
    const std::uint32_t permutation[4], std::uint32_t leg,
    std::uint64_t* destination) {
  return shifted_slot(
      chunks, center, permutation,
      developmental_append_accepted_source_ingress_offset(leg), false,
      destination);
}

[[nodiscard]] __device__ bool receptor_external_slot(
    const DeviceChunkMap& chunks, std::uint64_t center,
    const std::uint32_t permutation[4], DevelopmentalAppendOffset offset,
    std::uint64_t* destination) {
  return shifted_slot(chunks, center, permutation, offset, false,
                      destination);
}

[[nodiscard]] __device__ std::uint32_t receptor_journal_state(
    const SiteWord* words, std::uint64_t site_count,
    const DeviceChunkMap& chunks, std::uint64_t center,
    const std::uint32_t permutation[4], std::uint32_t digit) {
  const std::uint32_t site = kDevelopmentalAppendJournalFirst +
                             digit * kDevelopmentalAppendAgeSitesPerDigit + 2u;
  std::uint64_t slot = 0u;
  if (!relative_slot(chunks, center, permutation, site, &slot) ||
      slot >= site_count)
    return 3u;
  for (std::uint32_t encoded = 0u; encoded <= 2u; ++encoded)
    if (words[slot] == developmental_append_journal_word(
                           site, permutation[0u], permutation[1u],
                           permutation[2u], encoded))
      return encoded;
  return 3u;
}

[[nodiscard]] __device__ bool receptor_owner_present(
    const SiteWord* words, std::uint64_t site_count,
    const DeviceChunkMap& chunks, std::uint64_t center,
    const std::uint32_t permutation[4]) {
  constexpr std::uint32_t signature[] = {0u, 11u, 12u, 13u, 30u, 31u};
  for (const std::uint32_t site : signature) {
    std::uint64_t slot = 0u;
    if (!relative_slot(chunks, center, permutation, site, &slot) ||
        slot >= site_count ||
        words[slot] != developmental_append_product_word(
                           site, permutation[0u], permutation[1u],
                           permutation[2u], 0u))
      return false;
  }
  bool event_empty_seen = false;
  for (std::uint32_t digit = 0u;
       digit < kDevelopmentalAppendJournalDigitCount; ++digit) {
    const std::uint32_t first = kDevelopmentalAppendJournalFirst +
                                digit * kDevelopmentalAppendAgeSitesPerDigit;
    if (digit >= kDevelopmentalAppendReceptorLegCount) {
      std::uint64_t anchor = 0u;
      if (!relative_slot(chunks, center, permutation, first, &anchor) ||
          anchor >= site_count ||
          words[anchor] != developmental_append_journal_word(
                               first, permutation[0u], permutation[1u],
                               permutation[2u], 0u))
        return false;
    }
    std::uint64_t lock = 0u;
    if (!relative_slot(chunks, center, permutation, first + 1u, &lock) ||
        lock >= site_count ||
        words[lock] != developmental_append_journal_word(
                           first + 1u, permutation[0u], permutation[1u],
                           permutation[2u], 0u))
      return false;
    const std::uint32_t state = receptor_journal_state(
        words, site_count, chunks, center, permutation, digit);
    if (state > kDevelopmentalAppendReceptorJournalB)
      return false;
    if (digit < kDevelopmentalAppendAuthorityDigitCount) {
      if (state != kDevelopmentalAppendReceptorJournalEmpty &&
          state != digit + 1u)
        return false;
    } else {
      if (event_empty_seen &&
          state != kDevelopmentalAppendReceptorJournalEmpty)
        return false;
      event_empty_seen = event_empty_seen ||
                         state == kDevelopmentalAppendReceptorJournalEmpty;
    }
  }
  return true;
}

__global__ void append_receptor_kernel(
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
    if (!decode_permutation(rank, permutation)) continue;
    if (!receptor_owner_present(words, site_count, chunks, center,
                                permutation))
      continue;
    std::uint64_t teacher = 0u;
    if (!receptor_teacher_slot(chunks, center, permutation, &teacher) ||
        teacher >= site_count)
      continue;
    const SiteWord teacher_word = words[teacher];
    std::uint64_t inlet = 0u;
    if (!relative_slot(chunks, center, permutation, 10u, &inlet) ||
        inlet >= site_count)
      continue;
    const SiteWord inlet_bit = carrier_bit(permutation[1u]);
    if ((words[inlet] & inlet_bit) == 0u)
      continue;

    std::uint32_t selected_leg = kDevelopmentalAppendReceptorLegCount;
    std::uint32_t selected_digit = kDevelopmentalAppendJournalDigitCount;
    bool selected_source_accepted = false;
    std::uint64_t ports[kDevelopmentalAppendReceptorLegCount]{};
    std::uint64_t accepted_sources[kDevelopmentalAppendReceptorLegCount]{};
    bool ports_valid = true;
    for (std::uint32_t leg = 0u;
         leg < kDevelopmentalAppendReceptorLegCount; ++leg) {
      ports_valid = ports_valid &&
                    receptor_port_slot(chunks, center, permutation, leg,
                                       &ports[leg]) &&
                    ports[leg] < site_count &&
                    receptor_accepted_source_slot(
                        chunks, center, permutation, leg,
                        &accepted_sources[leg]) &&
                    accepted_sources[leg] < site_count;
    }
    if (!ports_valid) continue;

    for (std::uint32_t leg = 0u;
         leg < kDevelopmentalAppendReceptorLegCount; ++leg) {
      const std::uint32_t basis =
          inverse ? developmental_append_spent_source_basis(
                        leg, permutation[0u], permutation[1u],
                        permutation[2u])
                  : developmental_append_receptor_basis(
                        leg, permutation[0u], permutation[2u]);
      const SiteWord source_word =
          kQuiescentWord ^ carrier_bit(basis);
      const SiteWord port_word = words[ports[leg]];
      const SiteWord accepted_source_word = words[accepted_sources[leg]];
      bool source_matches = port_word == source_word;
      bool source_accepted = false;
      if (inverse && port_word == kQuiescentWord) {
        const SiteWord leg_filled_escrow = developmental_append_escrow_word(
            true, leg, permutation[0u], permutation[2u]);
        std::uint32_t top_bank = kDevelopmentalAppendWitnessBankCount;
        bool valid_bank = true;
        bool empty_seen = false;
        for (std::uint32_t bank = 0u;
             bank < kDevelopmentalAppendWitnessBankCount; ++bank) {
          std::uint64_t clock_slot = 0u;
          std::uint64_t reject_slot = 0u;
          if (!receptor_external_slot(
                  chunks, center, permutation,
                  developmental_append_clock_escrow_offset(leg, bank),
                  &clock_slot) ||
              !receptor_external_slot(
                  chunks, center, permutation,
                  developmental_append_reject_escrow_offset(leg, bank),
                  &reject_slot) ||
              clock_slot >= site_count || reject_slot >= site_count) {
            valid_bank = false;
            break;
          }
          const SiteWord clock_state = words[clock_slot];
          const SiteWord reject_state = words[reject_slot];
          const bool occupied = clock_state == leg_filled_escrow;
          valid_bank = valid_bank &&
                       (occupied || clock_state == kQuiescentWord) &&
                       (reject_state == kQuiescentWord ||
                        reject_state == leg_filled_escrow) &&
                       (!empty_seen || !occupied) &&
                       (occupied || reject_state == kQuiescentWord);
          if (occupied) top_bank = bank;
          empty_seen = empty_seen || !occupied;
        }
        std::uint64_t reject_source = 0u;
        const bool rejected_source =
            valid_bank && top_bank != kDevelopmentalAppendWitnessBankCount &&
            receptor_external_slot(
                chunks, center, permutation,
                developmental_append_reject_source_ingress_offset(
                    leg, top_bank),
                &reject_source) &&
            reject_source < site_count &&
            words[reject_source] == source_word;
        source_accepted = accepted_source_word == source_word;
        source_matches = source_accepted != rejected_source;
      }
      if (source_matches) {
        if (selected_leg != kDevelopmentalAppendReceptorLegCount) {
          selected_leg = kDevelopmentalAppendReceptorLegCount;
          break;
        }
        selected_leg = leg;
        selected_source_accepted = source_accepted;
      } else if (port_word != kQuiescentWord) {
        selected_leg = kDevelopmentalAppendReceptorLegCount;
        break;
      }
    }
    if (selected_leg == kDevelopmentalAppendReceptorLegCount) continue;
    selected_digit = selected_leg;
    const std::uint32_t journal_state = receptor_journal_state(
        words, site_count, chunks, center, permutation, selected_digit);
    std::uint64_t accepted_clock = 0u;
    const std::uint64_t accepted_source = accepted_sources[selected_leg];
    std::uint64_t accepted_teacher = 0u;
    std::uint64_t clock_escrow = 0u;
    std::uint64_t reject_escrow = 0u;
    if (!receptor_external_slot(
            chunks, center, permutation,
            developmental_append_clock_ingress_offset(selected_leg),
            &accepted_clock) ||
        !receptor_external_slot(
            chunks, center, permutation,
            developmental_append_accepted_teacher_ingress_offset(
                selected_leg),
            &accepted_teacher) ||
        accepted_clock >= site_count || accepted_teacher >= site_count)
      continue;
    const SiteWord clock_vacancy = developmental_append_clock_vacancy_word(
        selected_leg, permutation[0u], permutation[1u], permutation[2u]);
    const SiteWord filled_escrow = developmental_append_escrow_word(
        true, selected_leg, permutation[0u], permutation[2u]);
    std::uint32_t witness_bank = kDevelopmentalAppendWitnessBankCount;
    bool empty_seen = false;
    bool valid_bank = true;
    for (std::uint32_t bank = 0u;
         bank < kDevelopmentalAppendWitnessBankCount; ++bank) {
      std::uint64_t clock_slot = 0u;
      std::uint64_t reject_slot = 0u;
      if (!receptor_external_slot(
              chunks, center, permutation,
              developmental_append_clock_escrow_offset(selected_leg, bank),
              &clock_slot) ||
          !receptor_external_slot(
              chunks, center, permutation,
              developmental_append_reject_escrow_offset(selected_leg, bank),
              &reject_slot) ||
          clock_slot >= site_count || reject_slot >= site_count) {
        valid_bank = false;
        break;
      }
      const SiteWord clock_state = words[clock_slot];
      const SiteWord reject_state = words[reject_slot];
      const bool occupied = clock_state == filled_escrow;
      valid_bank = valid_bank &&
                   (occupied || clock_state == kQuiescentWord) &&
                   (reject_state == kQuiescentWord ||
                    reject_state == filled_escrow) &&
                   (!empty_seen || !occupied) &&
                   (occupied || reject_state == kQuiescentWord);
      if (inverse && occupied) witness_bank = bank;
      if (!inverse && !occupied &&
          witness_bank == kDevelopmentalAppendWitnessBankCount)
        witness_bank = bank;
      empty_seen = empty_seen || !occupied;
    }
    std::uint64_t reject_source = 0u;
    std::uint64_t reject_teacher = 0u;
    std::uint64_t reject_clock = 0u;
    if (!valid_bank || witness_bank == kDevelopmentalAppendWitnessBankCount ||
        !receptor_external_slot(
            chunks, center, permutation,
            developmental_append_clock_escrow_offset(selected_leg,
                                                      witness_bank),
            &clock_escrow) ||
        !receptor_external_slot(
            chunks, center, permutation,
            developmental_append_reject_escrow_offset(selected_leg,
                                                       witness_bank),
            &reject_escrow) ||
        !receptor_external_slot(
            chunks, center, permutation,
            developmental_append_reject_source_ingress_offset(
                selected_leg, witness_bank),
            &reject_source) ||
        !receptor_external_slot(
            chunks, center, permutation,
            developmental_append_reject_teacher_ingress_offset(
                selected_leg, witness_bank),
            &reject_teacher) ||
        !receptor_external_slot(
            chunks, center, permutation,
            developmental_append_reject_clock_ingress_offset(
                selected_leg, witness_bank),
            &reject_clock) ||
        clock_escrow >= site_count || reject_escrow >= site_count ||
        reject_source >= site_count || reject_teacher >= site_count ||
        reject_clock >= site_count)
      continue;
    bool teaching = false;
    if (!inverse) {
      if (teacher_word == developmental_append_teacher_vacancy_word(
                              selected_leg, permutation[0u],
                              permutation[2u])) {
        teaching = true;
      } else if (teacher_word != kQuiescentWord) {
        continue;
      }
      if (journal_state != kDevelopmentalAppendReceptorJournalEmpty ||
          words[accepted_source] != kQuiescentWord ||
          words[accepted_clock] != kQuiescentWord ||
          words[accepted_teacher] != kQuiescentWord ||
          words[reject_teacher] != kQuiescentWord ||
          words[reject_clock] != kQuiescentWord ||
          words[clock_escrow] != kQuiescentWord ||
          words[reject_escrow] != kQuiescentWord)
        continue;
    } else {
      if (selected_source_accepted &&
          teacher_word == kQuiescentWord &&
          words[accepted_teacher] ==
              developmental_append_spent_teacher_vacancy_word(
                  selected_leg, permutation[0u], permutation[1u],
                  permutation[2u]) &&
          journal_state == selected_leg + 1u &&
          words[reject_escrow] == kQuiescentWord &&
          words[reject_teacher] == kQuiescentWord &&
          words[reject_clock] == kQuiescentWord) {
        teaching = true;
      } else if (selected_source_accepted ||
                 teacher_word != kQuiescentWord ||
                 words[accepted_teacher] != kQuiescentWord ||
                 journal_state != kDevelopmentalAppendReceptorJournalEmpty ||
                 words[reject_escrow] != filled_escrow ||
                 words[reject_teacher] !=
                     developmental_append_spent_teacher_vacancy_word(
                         selected_leg, permutation[0u], permutation[1u],
                         permutation[2u]) ||
                 words[reject_clock] != clock_vacancy) {
        continue;
      }
      if (words[teaching ? accepted_clock : reject_clock] != clock_vacancy ||
          words[clock_escrow] != filled_escrow)
        continue;
    }
    if (selected_leg == kDevelopmentalAppendReceptorLegCount ||
        selected_digit == kDevelopmentalAppendJournalDigitCount)
      continue;
    const std::uint32_t authority_site =
        developmental_append_receptor_authority_site(selected_leg);
    std::uint64_t authority_slot = 0u;
    if (!relative_slot(chunks, center, permutation, authority_site,
                       &authority_slot) ||
        authority_slot >= site_count ||
        words[authority_slot] != developmental_append_journal_word(
                                    authority_site, permutation[0u],
                                    permutation[1u], permutation[2u], 0u))
      continue;
    const std::uint32_t journal_site =
        developmental_append_journal_state_site(selected_digit);
    std::uint64_t journal_slot = 0u;
    if (!relative_slot(chunks, center, permutation, journal_site,
                       &journal_slot) ||
        journal_slot >= site_count)
      return;
    words[journal_slot] = developmental_append_journal_word(
        journal_site, permutation[0u], permutation[1u], permutation[2u],
        teaching ? (inverse ? kDevelopmentalAppendReceptorJournalEmpty
                            : selected_leg + 1u)
                 : kDevelopmentalAppendReceptorJournalEmpty);
    words[teacher] = inverse && teaching
                         ? developmental_append_teacher_vacancy_word(
                               selected_leg, permutation[0u], permutation[2u])
                         : kQuiescentWord;
    words[ports[selected_leg]] =
        inverse ? (kQuiescentWord ^ carrier_bit(developmental_append_receptor_basis(
                              selected_leg, permutation[0u], permutation[2u])))
                : kQuiescentWord;
    if (teaching)
      words[accepted_source] =
          inverse ? kQuiescentWord
                  : developmental_append_spent_source_vacancy_word(
                        selected_leg, permutation[0u], permutation[1u],
                        permutation[2u]);
    if (!teaching)
      words[reject_source] =
          inverse ? kQuiescentWord
                  : developmental_append_spent_source_vacancy_word(
                        selected_leg, permutation[0u], permutation[1u],
                        permutation[2u]);
    if (teaching) {
      words[accepted_clock] = inverse ? kQuiescentWord : clock_vacancy;
      words[accepted_teacher] =
          inverse ? kQuiescentWord
                  : developmental_append_spent_teacher_vacancy_word(
                        selected_leg, permutation[0u], permutation[1u],
                        permutation[2u]);
    } else {
      words[reject_teacher] =
          inverse ? kQuiescentWord
                  : developmental_append_spent_teacher_vacancy_word(
                        selected_leg, permutation[0u], permutation[1u],
                        permutation[2u]);
      words[reject_clock] = inverse ? kQuiescentWord : clock_vacancy;
    }
    words[clock_escrow] = inverse ? kQuiescentWord : filled_escrow;
    words[reject_escrow] =
        inverse || teaching ? kQuiescentWord : filled_escrow;
    return;
  }
}

}  // namespace

namespace {

void launch_developmental_append_impl(
    SiteWord* words, std::uint8_t* scratch, std::uint64_t site_count,
    const DeviceChunkMap& chunks, bool inverse,
    const std::uint64_t* active_slots, std::uint64_t active_count,
    const std::uint32_t* device_active_count, cudaStream_t stream) {
  const std::uint64_t candidate_count =
      active_slots == nullptr ? site_count : active_count;
  if (candidate_count == 0u) return;
  const std::uint32_t blocks = launch_blocks(candidate_count);
  // Receptor matching precedes append matching in both directions.  For an
  // already-grown product the writes commute with the age-only transition;
  // for a just-born product this preserves the immutable pre-factor snapshot
  // rule and defers a coincident port vacancy until the next K application.
  append_receptor_kernel<<<blocks, kThreads, 0, stream>>>(
      words, candidate_count, site_count, chunks, inverse, active_slots,
      device_active_count);
  check_cuda(cudaGetLastError(),
             inverse ? "launch inverse developmental append receptor"
                     : "launch developmental append receptor");
  append_match_kernel<<<blocks, kThreads, 0, stream>>>(
      words, scratch, candidate_count, site_count, chunks, inverse,
      active_slots, device_active_count);
  check_cuda(cudaGetLastError(), "launch developmental append match");
  append_collision_kernel<<<blocks, kThreads, 0, stream>>>(
      words, scratch, candidate_count, site_count, chunks, inverse,
      active_slots, device_active_count);
  check_cuda(cudaGetLastError(), "launch developmental append collision");
  append_apply_kernel<<<blocks, kThreads, 0, stream>>>(
      words, scratch, candidate_count, site_count, chunks, inverse,
      active_slots, device_active_count);
  check_cuda(cudaGetLastError(), "launch developmental append apply");
}

}  // namespace

void launch_developmental_append_cuda(
    SiteWord* words, std::uint8_t* scratch, std::uint64_t site_count,
    const DeviceChunkMap& chunks, bool inverse,
    const std::uint64_t* active_slots, std::uint64_t active_count,
    cudaStream_t stream) {
  launch_developmental_append_impl(
      words, scratch, site_count, chunks, inverse, active_slots, active_count,
      nullptr, stream);
}

void launch_developmental_append_graph_safe_cuda(
    SiteWord* words, std::uint8_t* scratch, std::uint64_t site_count,
    const DeviceChunkMap& chunks, bool inverse,
    const std::uint64_t* active_slots,
    const std::uint32_t* device_active_count, std::uint32_t capacity,
    cudaStream_t stream) {
  launch_developmental_append_impl(
      words, scratch, site_count, chunks, inverse, active_slots, capacity,
      device_active_count, stream);
}

}  // namespace substrate::bcc32
