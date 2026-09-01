#ifndef HARDWARE_NATIVE_DIRECT_LOCALITY_PAGE_BUCKETING_CUH
#define HARDWARE_NATIVE_DIRECT_LOCALITY_PAGE_BUCKETING_CUH

#include <cstdint>

namespace substrate::direct_network {

constexpr std::uint32_t kLocalityBucketCapacity = 32u;

struct DirectLocalityRecipeIdentity {
  std::uint64_t logical_recipe_id = 0u;
  std::uint64_t revision_identity = 0u;
  std::uint64_t participation_lineage = 0u;
  std::uint64_t exact_history_digest = 0u;
};

struct DirectLocalityPageEntry {
  DirectLocalityRecipeIdentity identity{};
  std::uint32_t physical_slot = 0u;
  std::uint32_t reserved = 0u;
};

// This is a transport cache, never recipe or history authority. Resident
// co-activity may change physical_slot; logical identities and their causal
// lineage remain owned by the canonical recipe and exact-history stores.
struct DirectLocalityPageCache {
  DirectLocalityPageEntry entries[kLocalityBucketCapacity]{};
  std::uint32_t coactivity[kLocalityBucketCapacity * kLocalityBucketCapacity]{};
  std::uint32_t activity[kLocalityBucketCapacity]{};
  std::uint32_t entry_count = 0u;
  std::uint32_t page_width = 0u;
  std::uint64_t layout_epoch = 0u;
};

__device__ inline bool locality_record_coactivity(DirectLocalityPageCache* cache,
                                                  std::uint32_t left, std::uint32_t right) {
  if (cache == nullptr || left >= cache->entry_count || right >= cache->entry_count ||
      left == right || cache->entry_count > kLocalityBucketCapacity)
    return false;
  const std::uint32_t stride = kLocalityBucketCapacity;
  ++cache->coactivity[left * stride + right];
  ++cache->coactivity[right * stride + left];
  ++cache->activity[left];
  ++cache->activity[right];
  return true;
}

__device__ inline bool locality_identity_equal(const DirectLocalityRecipeIdentity& left,
                                               const DirectLocalityRecipeIdentity& right) {
  return left.logical_recipe_id == right.logical_recipe_id &&
         left.revision_identity == right.revision_identity &&
         left.participation_lineage == right.participation_lineage &&
         left.exact_history_digest == right.exact_history_digest;
}

// Re-layout is deterministic and resident: only measured co-activity and
// stable logical identity break ties. No label, token, language, or host
// ordering enters this law. Inactive entries keep their physical slots.
__device__ inline bool locality_repage_from_coactivity(DirectLocalityPageCache* cache) {
  if (cache == nullptr || cache->entry_count == 0u ||
      cache->entry_count > kLocalityBucketCapacity || cache->page_width == 0u ||
      cache->page_width > kLocalityBucketCapacity)
    return false;

  DirectLocalityPageCache candidate = *cache;
  bool placed[kLocalityBucketCapacity]{};
  std::uint32_t active_count = 0u;
  for (std::uint32_t i = 0u; i < candidate.entry_count; ++i)
    active_count += candidate.activity[i] != 0u ? 1u : 0u;
  if (active_count == 0u)
    return false;

  std::uint32_t order[kLocalityBucketCapacity]{};
  for (std::uint32_t position = 0u; position < active_count; ++position) {
    std::uint32_t winner = kLocalityBucketCapacity;
    std::uint64_t winner_affinity = 0u;
    for (std::uint32_t candidate_index = 0u; candidate_index < candidate.entry_count;
         ++candidate_index) {
      if (placed[candidate_index] || candidate.activity[candidate_index] == 0u)
        continue;
      std::uint64_t affinity = 0u;
      for (std::uint32_t prior = 0u; prior < position; ++prior)
        affinity += candidate.coactivity[candidate_index * kLocalityBucketCapacity + order[prior]];
      const bool better = winner == kLocalityBucketCapacity || affinity > winner_affinity ||
                          (affinity == winner_affinity &&
                           (candidate.activity[candidate_index] > candidate.activity[winner] ||
                            (candidate.activity[candidate_index] == candidate.activity[winner] &&
                             candidate.entries[candidate_index].identity.logical_recipe_id <
                                 candidate.entries[winner].identity.logical_recipe_id)));
      if (better) {
        winner = candidate_index;
        winner_affinity = affinity;
      }
    }
    if (winner == kLocalityBucketCapacity)
      return false;
    order[position] = winner;
    placed[winner] = true;
  }

  for (std::uint32_t position = 0u; position < active_count; ++position)
    candidate.entries[order[position]].physical_slot = position;
  ++candidate.layout_epoch;
  *cache = candidate;
  return true;
}

__device__ inline std::uint32_t locality_touched_pages(const DirectLocalityPageCache& cache,
                                                       const std::uint32_t* entry_indices,
                                                       std::uint32_t count) {
  if (entry_indices == nullptr || cache.page_width == 0u)
    return 0u;
  std::uint32_t pages[kLocalityBucketCapacity]{};
  std::uint32_t page_count = 0u;
  for (std::uint32_t i = 0u; i < count; ++i) {
    const std::uint32_t entry = entry_indices[i];
    if (entry >= cache.entry_count)
      return 0u;
    const std::uint32_t page = cache.entries[entry].physical_slot / cache.page_width;
    bool seen = false;
    for (std::uint32_t j = 0u; j < page_count; ++j)
      seen = seen || pages[j] == page;
    if (!seen)
      pages[page_count++] = page;
  }
  return page_count;
}

}  // namespace substrate::direct_network

#endif
