__global__ void build_ngram_keys_kernel(const std::uint32_t* units,
                                        std::uint32_t occurrence_count,
                                        BigramKey* bigrams, TrigramKey* trigrams) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i + 1u < occurrence_count) bigrams[i] = BigramKey{units[i], units[i + 1u]};
  if (i + 2u < occurrence_count) {
    trigrams[i] = TrigramKey{units[i], units[i + 1u], units[i + 2u]};
  }
}

__global__ void mark_sequence_segment_starts_kernel(
    const std::uint32_t* units, std::uint32_t unit_count,
    const std::uint32_t* unit_lengths, const std::uint32_t* unit_content,
    const std::uint32_t* closure_bytes, const std::uint32_t* boundary_bytes,
    std::uint32_t* segment_starts) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= unit_count) return;
  bool closes = false;
  if (i != 0u && closure_bytes != nullptr && boundary_bytes != nullptr) {
    for (std::uint32_t closure = 0u; closure < kClosureCount; ++closure) {
      closes |= resident_unit_terminal_byte(
          unit_lengths, unit_content, units[i - 1u], closure_bytes[closure],
          boundary_bytes[0]);
    }
  }
  segment_starts[i] = closes ? 1u : 0u;
}

__global__ void build_sequence_associations_kernel(
    const std::uint32_t* units, const std::uint32_t* segment_ids,
    std::uint32_t unit_count, AssociationKey* associations,
    std::uint32_t* counts, std::uint32_t* appended_count,
    std::uint32_t* ledger) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  const std::uint32_t stride = gridDim.x * blockDim.x;
  for (std::uint32_t distance = 1u; distance <= kAssociationRadius; ++distance) {
    const std::uint32_t extent = unit_count > distance ? unit_count - distance : 0u;
    for (std::uint32_t position = i; position < extent; position += stride) {
      const std::uint32_t support = kAssociationRadius + 1u - distance;
      if (segment_ids[position] != segment_ids[position + distance]) {
        atomicAdd(ledger + 1u, support);
        atomicSub(ledger + 2u, support);
        continue;
      }
      const std::uint32_t out = atomicAdd(appended_count, 1u);
      associations[out] = AssociationKey{units[position], units[position + distance]};
      counts[out] = support;
    }
  }
}

__global__ void build_posting_material_kernel(const std::uint32_t* units,
                                              std::uint32_t count,
                                              std::uint32_t* keys,
                                              std::uint32_t* positions,
                                              std::uint32_t* shifted_counts) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count) return;
  const std::uint32_t unit = units[i];
  keys[i] = unit;
  positions[i] = i;
  atomicAdd(shifted_counts + unit + 1u, 1u);
}

__global__ void count_episode_postings_kernel(const std::uint32_t* units,
                                              std::uint32_t count,
                                              std::uint32_t* shifted_counts) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count) atomicAdd(shifted_counts + units[i] + 1u, 1u);
}
