// Device-resident raw-byte unit hashing and assimilation kernels.
// Included inside the adult-v1 namespace after the conditioned-credit bridge.

__device__ bool unit_matches_bytes(const std::uint8_t* bytes, std::uint32_t begin,
                                   std::uint32_t length, std::uint32_t unit,
                                   const std::uint32_t* unit_lengths,
                                   const std::uint32_t* unit_content) {
  if (unit_lengths[unit] != length) return false;
  for (std::uint32_t i = 0u; i < length; ++i) {
    const std::uint32_t packed = unit_content[unit * kUnitWords + i / 4u];
    if (bytes[begin + i] != static_cast<std::uint8_t>(packed >> ((i % 4u) * 8u))) return false;
  }
  return true;
}

__device__ UnitKey fingerprint_bytes(const std::uint8_t* bytes, std::uint32_t begin,
                                     std::uint32_t length) {
  std::uint32_t a = 2166136261u;
  std::uint32_t b = 2246822519u;
  for (std::uint32_t i = 0u; i < length; ++i) {
    const std::uint32_t value = bytes[begin + i];
    a = (a ^ value) * 16777619u;
    b = mix32(b + value + i * 0x9e3779b9u);
  }
  return UnitKey{a, b, length};
}

__device__ UnitKey fingerprint_unit(std::uint32_t unit,
                                    const std::uint32_t* unit_lengths,
                                    const std::uint32_t* unit_content) {
  std::uint32_t a = 2166136261u;
  std::uint32_t b = 2246822519u;
  const std::uint32_t length = unit_lengths[unit];
  for (std::uint32_t i = 0u; i < length; ++i) {
    const std::uint32_t packed = unit_content[unit * kUnitWords + i / 4u];
    const std::uint32_t value = (packed >> ((i % 4u) * 8u)) & 0xffu;
    a = (a ^ value) * 16777619u;
    b = mix32(b + value + i * 0x9e3779b9u);
  }
  return UnitKey{a, b, length};
}

__device__ std::uint32_t fingerprint_slot(UnitKey key, std::uint32_t mask) {
  return mix32(key.hash_a ^ key.hash_b ^ key.length * 0x85ebca6bu) & mask;
}

__global__ void populate_unit_hash_kernel(const std::uint32_t* unit_lengths,
                                          const std::uint32_t* unit_content,
                                          std::uint32_t unit_count,
                                          std::uint32_t* slots,
                                          std::uint32_t slot_count) {
  const std::uint32_t unit = blockIdx.x * blockDim.x + threadIdx.x;
  if (unit >= unit_count) return;
  const UnitKey key = fingerprint_unit(unit, unit_lengths, unit_content);
  const std::uint32_t mask = slot_count - 1u;
  std::uint32_t slot = fingerprint_slot(key, mask);
  for (std::uint32_t probe = 0u; probe < slot_count; ++probe) {
    if (atomicCAS(slots + slot, 0u, unit + 1u) == 0u) return;
    slot = (slot + 1u) & mask;
  }
}

__device__ std::uint32_t find_exact_unit(const std::uint8_t* bytes, std::uint32_t begin,
                                         std::uint32_t length, std::uint32_t unit_count,
                                         const std::uint32_t* unit_lengths,
                                         const std::uint32_t* unit_content) {
  for (std::uint32_t unit = 0u; unit < unit_count; ++unit) {
    if (unit_matches_bytes(bytes, begin, length, unit, unit_lengths, unit_content)) return unit;
  }
  return 0xffffffffu;
}

__device__ std::uint32_t find_base_bigram(const BigramKey* keys, std::uint32_t count,
                                          BigramKey wanted) {
  std::uint32_t lo = 0u, hi = count;
  while (lo < hi) {
    const std::uint32_t mid = lo + (hi - lo) / 2u;
    if (keys[mid] < wanted) lo = mid + 1u; else hi = mid;
  }
  return lo < count && keys[lo] == wanted ? lo : 0xffffffffu;
}

__device__ std::uint32_t find_base_trigram(const TrigramKey* keys, std::uint32_t count,
                                           TrigramKey wanted) {
  std::uint32_t lo = 0u, hi = count;
  while (lo < hi) {
    const std::uint32_t mid = lo + (hi - lo) / 2u;
    if (keys[mid] < wanted) lo = mid + 1u; else hi = mid;
  }
  return lo < count && keys[lo] == wanted ? lo : 0xffffffffu;
}

__global__ void scatter_unique_assimilation_representatives_kernel(
    const std::uint32_t* sorted_occurrences, const std::uint32_t* unique_flags,
    const std::uint32_t* group_ids, std::uint32_t sequence_count,
    std::uint32_t* representatives) {
  const std::uint32_t sorted = blockIdx.x * blockDim.x + threadIdx.x;
  if (sorted >= sequence_count || unique_flags[sorted] == 0u) return;
  representatives[group_ids[sorted] - 1u] = sorted_occurrences[sorted];
}

__global__ void resolve_unique_assimilation_units_kernel(
    const std::uint8_t* bytes, std::uint32_t byte_count, const std::uint32_t* starts,
    std::uint32_t sequence_count, const std::uint32_t* representatives,
    std::uint32_t unique_count,
    const std::uint32_t* unit_lengths, const std::uint32_t* unit_content,
    const std::uint32_t* hash_slots, std::uint32_t hash_capacity,
    std::uint32_t* group_units, std::uint32_t* novel_flags,
    std::uint32_t* status, const std::uint32_t* exact_replay) {
  const std::uint32_t group = blockIdx.x * blockDim.x + threadIdx.x;
  if (group >= unique_count || exact_replay[0] != 0u) return;
  const std::uint32_t segment = representatives[group];
  const std::uint32_t begin = starts[segment];
  const std::uint32_t end = segment + 1u < sequence_count
      ? starts[segment + 1u] : byte_count;
  const UnitKey key = fingerprint_bytes(bytes, begin, end - begin);
  const std::uint32_t mask = hash_capacity - 1u;
  std::uint32_t slot = fingerprint_slot(key, mask);
  for (std::uint32_t probe = 0u; probe < hash_capacity; ++probe) {
    const std::uint32_t resident = hash_slots[slot];
    if (resident == 0u) {
      group_units[group] = 0xffffffffu;
      novel_flags[group] = 1u;
      return;
    }
    const std::uint32_t unit = resident - 1u;
    if (unit_matches_bytes(bytes, begin, key.length, unit,
                           unit_lengths, unit_content)) {
      group_units[group] = unit;
      novel_flags[group] = 0u;
      return;
    }
    slot = (slot + 1u) & mask;
  }
  atomicMax(status, 1u);
}

__global__ void materialize_novel_assimilation_units_kernel(
    const std::uint8_t* bytes, std::uint32_t byte_count,
    const std::uint32_t* starts, std::uint32_t sequence_count,
    const std::uint32_t* representatives, std::uint32_t unique_count,
    const std::uint32_t* novel_flags, const std::uint32_t* novel_ids,
    std::uint32_t unit_base, std::uint32_t* unit_lengths,
    std::uint32_t* unit_content, std::uint32_t* vitality,
    std::uint32_t* group_units, const std::uint32_t* exact_replay) {
  const std::uint32_t group = blockIdx.x * blockDim.x + threadIdx.x;
  if (group >= unique_count || exact_replay[0] != 0u ||
      novel_flags[group] == 0u) {
    return;
  }
  const std::uint32_t segment = representatives[group];
  const std::uint32_t begin = starts[segment];
  const std::uint32_t end = segment + 1u < sequence_count
      ? starts[segment + 1u] : byte_count;
  const std::uint32_t allocated = unit_base + novel_ids[group] - 1u;
  unit_lengths[allocated] = end - begin;
  for (std::uint32_t word = 0u; word < kUnitWords; ++word) {
    std::uint32_t packed = 0u;
    for (std::uint32_t lane = 0u; lane < 4u; ++lane) {
      const std::uint32_t offset = word * 4u + lane;
      if (begin + offset < end) {
        packed |= static_cast<std::uint32_t>(bytes[begin + offset])
            << (lane * 8u);
      }
    }
    unit_content[allocated * kUnitWords + word] = packed;
  }
  vitality[allocated] = 0u;
  group_units[group] = allocated;
}

__global__ void populate_unit_hash_range_kernel(
    const std::uint32_t* unit_lengths, const std::uint32_t* unit_content,
    std::uint32_t unit_begin, std::uint32_t unit_count,
    std::uint32_t* slots, std::uint32_t slot_count) {
  const std::uint32_t offset = blockIdx.x * blockDim.x + threadIdx.x;
  if (offset >= unit_count) return;
  const std::uint32_t unit = unit_begin + offset;
  const UnitKey key = fingerprint_unit(unit, unit_lengths, unit_content);
  const std::uint32_t mask = slot_count - 1u;
  std::uint32_t slot = fingerprint_slot(key, mask);
  for (std::uint32_t probe = 0u; probe < slot_count; ++probe) {
    if (atomicCAS(slots + slot, 0u, unit + 1u) == 0u) return;
    slot = (slot + 1u) & mask;
  }
}

__global__ void map_assimilation_occurrence_groups_kernel(
    const std::uint32_t* sorted_occurrences, const std::uint32_t* group_ids,
    std::uint32_t sequence_count, const std::uint32_t* group_units,
    std::uint32_t* sequence, const std::uint32_t* exact_replay) {
  const std::uint32_t sorted = blockIdx.x * blockDim.x + threadIdx.x;
  if (sorted >= sequence_count || exact_replay[0] != 0u) return;
  sequence[sorted_occurrences[sorted]] = group_units[group_ids[sorted] - 1u];
}

__global__ void shuffle_assimilation_sequence_kernel(
    std::uint32_t* sequence, std::uint32_t sequence_count,
    std::uint32_t shuffle_seed) {
  if (blockIdx.x != 0u || threadIdx.x != 0u) return;
  std::uint32_t random = shuffle_seed == 0u ? 0x9e3779b9u : shuffle_seed;
  for (std::uint32_t i = sequence_count; i > 1u; --i) {
    random ^= random << 13u;
    random ^= random >> 17u;
    random ^= random << 5u;
    const std::uint32_t j = random % i;
    const std::uint32_t temporary = sequence[i - 1u];
    sequence[i - 1u] = sequence[j];
    sequence[j] = temporary;
  }
}

__global__ void assimilate_raw_bytes_kernel(
    const std::uint8_t* bytes, std::uint32_t byte_count, const std::uint32_t* boundary_mask,
    const std::uint32_t* closure_bytes,
    std::uint32_t* unit_lengths, std::uint32_t* unit_content, std::uint32_t* vitality,
    std::uint32_t unit_capacity, const BigramKey* base_bigrams,
    std::uint32_t* base_bigram_counts, std::uint32_t base_bigram_count,
    const TrigramKey* base_trigrams, std::uint32_t* base_trigram_counts,
    std::uint32_t base_trigram_count, BigramKey* online_bigrams,
    std::uint32_t* online_bigram_counts, TrigramKey* online_trigrams,
    std::uint32_t* online_trigram_counts, AssociationKey* associations,
    std::uint32_t* association_counts, std::uint32_t* episode_units,
    std::uint32_t* episode_breaks, std::uint32_t* mutable_sizes,
    const std::uint32_t* starts, std::uint32_t sequence_count,
    std::uint32_t prefix_count, std::uint32_t final_chunk,
    std::uint32_t* segment_extent, const std::uint32_t* segment_ids,
    std::uint32_t* sequence, std::uint32_t* ledger, std::uint32_t* status,
    const std::uint32_t* exact_replay) {
  if (blockIdx.x != 0u || status[0] != 0u || exact_replay[0] != 0u) return;
  __shared__ std::uint32_t bigram_base;
  __shared__ std::uint32_t trigram_base;
  __shared__ std::uint32_t association_base;
  __shared__ std::uint32_t episode_base;
  __shared__ std::uint32_t association_events;
  __shared__ std::uint32_t update_allowed;
  __shared__ std::uint32_t bigram_appended;
  __shared__ std::uint32_t trigram_appended;
  __shared__ std::uint32_t association_appended;
  __shared__ std::uint32_t merge_existing;
  __shared__ std::uint32_t episode_segments;
  (void)bytes;
  (void)byte_count;
  (void)boundary_mask;
  (void)unit_capacity;
  (void)base_bigrams;
  (void)base_bigram_counts;
  (void)base_bigram_count;
  (void)base_trigrams;
  (void)base_trigram_counts;
  (void)base_trigram_count;
  (void)starts;
  __syncthreads();
  // The tally is a pure integer count over an unchanged pair set, so spreading
  // it across the block leaves the sum -- and every downstream decision --
  // bit-identical to the serial fold.
  __shared__ unsigned long long event_partials[kBlock];
  __shared__ unsigned long long weighted_partials[kBlock];
  {
    unsigned long long events = 0u;
    unsigned long long weighted = 0u;
    const std::uint32_t distances = min(kAssociationRadius,
                                        sequence_count > 0u ? sequence_count - 1u : 0u);
    for (std::uint32_t distance = 1u; distance <= distances; ++distance) {
      const std::uint32_t start = prefix_count > distance
          ? prefix_count - distance : 0u;
      const std::uint32_t count = sequence_count - distance - start;
      for (std::uint32_t i = threadIdx.x; i < count; i += blockDim.x) {
        if (segment_ids[start + i] !=
            segment_ids[start + i + distance]) continue;
        ++events;
        weighted += kAssociationRadius + 1u - distance;
      }
    }
    event_partials[threadIdx.x] = events;
    weighted_partials[threadIdx.x] = weighted;
  }
  __syncthreads();
  if (threadIdx.x == 0u) {
    unsigned long long events = event_partials[0];
    unsigned long long weighted = weighted_partials[0];
    for (std::uint32_t lane = 1u; lane < blockDim.x; ++lane) {
      events += event_partials[lane];
      weighted += weighted_partials[lane];
    }
    association_events = static_cast<std::uint32_t>(events);
    bigram_base = mutable_sizes[1];
    trigram_base = mutable_sizes[2];
    association_base = mutable_sizes[3];
    episode_base = mutable_sizes[4];
    bigram_appended = 0u;
    trigram_appended = 0u;
    association_appended = 0u;
    merge_existing = sequence_count <= 1024u;
    episode_segments = 0u;
    std::uint32_t extent = segment_extent[0];
    for (std::uint32_t i = prefix_count; i < sequence_count; ++i) {
      ++extent;
      const bool learned_close = resident_unit_contains_any(
          unit_lengths, unit_content, sequence[i], closure_bytes, kClosureCount);
      if ((final_chunk != 0u && i + 1u == sequence_count) ||
          extent >= 4u * kCompositionUnits ||
          (extent >= 2u * kCompositionUnits && learned_close)) {
        ++episode_segments;
        extent = 0u;
      }
    }
    const std::uint32_t new_count = sequence_count - prefix_count;
    const std::uint32_t bigram_begin = prefix_count > 0u ? prefix_count - 1u : 0u;
    const std::uint32_t trigram_begin = prefix_count > 1u ? prefix_count - 2u : 0u;
    const std::uint32_t bigram_added = sequence_count > 1u
        ? sequence_count - 1u - bigram_begin : 0u;
    const std::uint32_t trigram_added = sequence_count > 2u
        ? sequence_count - 2u - trigram_begin : 0u;
    update_allowed = 1u;
    if (bigram_base + bigram_added > kOnlineNgramCapacity) status[0] = 3u;
    else if (trigram_base + trigram_added > kOnlineNgramCapacity) status[0] = 4u;
    else if (association_base + association_events > kOnlineAssociationCapacity) status[0] = 5u;
    else if (episode_base + new_count > kOnlineEpisodeCapacity ||
             mutable_sizes[5] + episode_segments > kOnlineEpisodeBreakCapacity) status[0] = 6u;
    const unsigned long long added_mass = static_cast<unsigned long long>(new_count) * 2u +
        bigram_added + trigram_added + weighted;
    if (status[0] == 0u && added_mass > ledger[1]) status[0] = 2u;
    if (status[0] != 0u) update_allowed = 0u;
    else {
      ledger[1] -= static_cast<std::uint32_t>(added_mass);
      ledger[2] += static_cast<std::uint32_t>(added_mass);
    }
  }
  __syncthreads();
  if (update_allowed == 0u) return;
  for (std::uint32_t i = threadIdx.x; i < sequence_count; i += blockDim.x) {
    if (i >= prefix_count) {
      atomicAdd(vitality + sequence[i], 1u);
      episode_units[episode_base + i - prefix_count] = sequence[i];
    }
    if (i + 1u < sequence_count && i + 1u >= prefix_count) {
      const BigramKey wanted{sequence[i], sequence[i + 1u]};
      const std::uint32_t found = merge_existing != 0u
          ? find_base_bigram(online_bigrams, bigram_base, wanted) : 0xffffffffu;
      if (found != 0xffffffffu) {
        atomicAdd(online_bigram_counts + found, 1u);
      } else {
        const std::uint32_t out = bigram_base + atomicAdd(&bigram_appended, 1u);
        online_bigrams[out] = wanted;
        online_bigram_counts[out] = 1u;
      }
    }
    if (i + 2u < sequence_count && i + 2u >= prefix_count) {
      const TrigramKey wanted{sequence[i], sequence[i + 1u], sequence[i + 2u]};
      const std::uint32_t found = merge_existing != 0u
          ? find_base_trigram(online_trigrams, trigram_base, wanted) : 0xffffffffu;
      if (found != 0xffffffffu) {
        atomicAdd(online_trigram_counts + found, 1u);
      } else {
        const std::uint32_t out = trigram_base + atomicAdd(&trigram_appended, 1u);
        online_trigrams[out] = wanted;
        online_trigram_counts[out] = 1u;
      }
    }
  }
  const std::uint32_t distances = min(kAssociationRadius,
                                      sequence_count > 0u ? sequence_count - 1u : 0u);
  for (std::uint32_t distance = 1u; distance <= distances; ++distance) {
    const std::uint32_t start = prefix_count > distance
        ? prefix_count - distance : 0u;
    const std::uint32_t count = sequence_count - distance - start;
    for (std::uint32_t i = threadIdx.x; i < count; i += blockDim.x) {
      if (segment_ids[start + i] !=
          segment_ids[start + i + distance]) continue;
      const std::uint32_t out =
          association_base + atomicAdd(&association_appended, 1u);
      associations[out] = AssociationKey{sequence[start + i],
                                         sequence[start + i + distance]};
      association_counts[out] = kAssociationRadius + 1u - distance;
    }
  }
  __syncthreads();
  if (threadIdx.x == 0u) {
    const std::uint32_t episode_count =
        episode_base + sequence_count - prefix_count;
    std::uint32_t extent = segment_extent[0];
    for (std::uint32_t i = prefix_count; i < sequence_count; ++i) {
      ++extent;
      const bool learned_close = resident_unit_contains_any(
          unit_lengths, unit_content, sequence[i], closure_bytes, kClosureCount);
      if ((final_chunk != 0u && i + 1u == sequence_count) ||
          extent >= 4u * kCompositionUnits ||
          (extent >= 2u * kCompositionUnits && learned_close)) {
        episode_breaks[mutable_sizes[5]++] =
            episode_base + i + 1u - prefix_count;
        extent = 0u;
      }
    }
    segment_extent[0] = extent;
    mutable_sizes[1] = bigram_base + bigram_appended;
    mutable_sizes[2] = trigram_base + trigram_appended;
    mutable_sizes[3] = association_base + association_appended;
    mutable_sizes[4] = episode_count;
  }
}
