// Adult cue-contact matching and exact-order evidence kernels.
//
// Included inside bcc32_cuda_adult_v1 after fuzzy association helpers and
// before base-completion policy. This unit owns the resident-side contact
// evidence fanout and exact proposition sequence; it adds no state or order.

__global__ void propagate_ranked_cue_anchor_bytes_kernel(
    const std::uint32_t* selected_anchor_ids, const std::uint32_t* unit_lengths,
    const std::uint32_t* unit_content, std::uint32_t unit_count,
    std::uint32_t* cue_scores, std::uint32_t* cue_orders,
    std::uint32_t* cue_weights, std::uint32_t* cue_masks,
    std::uint32_t* salient_masks) {
  const std::uint32_t unit = blockIdx.x * blockDim.x + threadIdx.x;
  if (unit >= unit_count) return;
  for (std::uint32_t rank = 0u; rank < kCueAnchorLimit; ++rank) {
    const std::uint32_t representative = selected_anchor_ids[rank];
    if (representative == 0xffffffffu || representative == unit ||
        unit_lengths[representative] != unit_lengths[unit]) continue;
    bool equal = true;
    for (std::uint32_t word = 0u; word < kUnitWords; ++word) {
      equal &= unit_content[representative * kUnitWords + word] ==
               unit_content[unit * kUnitWords + word];
    }
    if (!equal) continue;
    cue_scores[unit] = max(cue_scores[unit], cue_scores[representative]);
    cue_orders[unit] = cue_orders[representative];
    cue_weights[unit] = max(cue_weights[unit], cue_weights[representative]);
    cue_masks[unit] |= 1u << rank;
    if (salient_masks[representative] != 0u) salient_masks[unit] |= 1u << rank;
  }
}

__global__ void score_composition_relations_kernel(
    const AssociationKey* associations, const std::uint32_t* association_counts,
    std::uint32_t association_count, const std::uint32_t* cue_scores,
    const std::uint32_t* cue_orders, const std::uint32_t* cue_weights,
    const std::uint32_t* cue_masks, const std::uint32_t* salient_masks,
    unsigned long long* forward_scores, unsigned long long* backward_scores,
    std::uint32_t* forward_support, std::uint32_t* backward_support,
    std::uint32_t* forward_salient_support,
    unsigned long long* strongest_evidence, unsigned long long* second_evidence,
    unsigned long long* cue_evidence) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= association_count || association_counts[i] == 0u) return;
  const AssociationKey relation = associations[i];
  if (relation.first == relation.second) return;
  const std::uint32_t first_score = cue_scores[relation.first];
  const std::uint32_t second_score = cue_scores[relation.second];
  const std::uint32_t first_mask = cue_masks[relation.first];
  const std::uint32_t second_mask = cue_masks[relation.second];
  const unsigned long long weight = association_counts[i];
  if (first_mask != 0u && first_score >= kCueNearIdentity) {
    const unsigned long long contribution =
        weight * first_score * max(1u, cue_weights[relation.first]);
    atomicAdd(forward_scores + relation.second,
              contribution);
    atomicOr(forward_support + relation.second, first_mask);
    atomicOr(forward_salient_support + relation.second, salient_masks[relation.first]);
    const unsigned long long prior = atomicMax(strongest_evidence + relation.second,
                                                contribution);
    if (prior != 0u) atomicMax(second_evidence + relation.second,
                               min(prior, contribution));
  }
  if (second_mask != 0u && second_score >= kCueNearIdentity) {
    const unsigned long long contribution =
        weight * second_score * max(1u, cue_weights[relation.second]);
    atomicAdd(backward_scores + relation.first,
              contribution);
    atomicOr(backward_support + relation.first, second_mask);
  }
  if (first_mask != 0u && second_mask != 0u) {
    const std::uint32_t first_order = cue_orders[relation.first];
    const std::uint32_t second_order = cue_orders[relation.second];
    if (first_order < second_order) {
      atomicAdd(cue_evidence, weight);
    } else if (second_order < first_order) {
      atomicAdd(cue_evidence + 1u, weight);
    }
    atomicAdd(cue_evidence + 2u, weight);
  }
}

__device__ std::uint32_t resident_bigram_count(
    const BigramKey* keys, const std::uint32_t* counts, std::uint32_t count,
    std::uint32_t previous, std::uint32_t next);



__global__ void match_cue_segments_parallel_kernel(
    const std::uint8_t* bytes, std::uint32_t byte_count, const std::uint32_t* starts,
    std::uint32_t cue_count, const std::uint32_t* unit_lengths,
    const std::uint32_t* unit_content, std::uint32_t unit_count,
    std::uint32_t* best_ids, std::uint32_t* best_scores) {
  const std::uint32_t cue = blockIdx.x;
  if (cue >= cue_count) return;
  const std::uint32_t begin = starts[cue];
  const std::uint32_t end = cue + 1u < cue_count ? starts[cue + 1u] : byte_count;
  std::uint32_t local_id = 0u;
  std::uint32_t local_score = 0u;
  for (std::uint32_t unit = threadIdx.x; unit < unit_count; unit += blockDim.x) {
    const std::uint32_t score = fuzzy_unit_score(bytes, begin, end - begin, unit,
                                                 unit_lengths, unit_content);
    if (score > local_score || (score == local_score && unit < local_id)) {
      local_score = score;
      local_id = unit;
    }
  }
  __shared__ std::uint32_t ids[kBlock];
  __shared__ std::uint32_t scores[kBlock];
  ids[threadIdx.x] = local_id;
  scores[threadIdx.x] = local_score;
  __syncthreads();
  for (std::uint32_t offset = blockDim.x / 2u; offset != 0u; offset >>= 1u) {
    if (threadIdx.x < offset &&
        (scores[threadIdx.x + offset] > scores[threadIdx.x] ||
         (scores[threadIdx.x + offset] == scores[threadIdx.x] &&
          ids[threadIdx.x + offset] < ids[threadIdx.x]))) {
      scores[threadIdx.x] = scores[threadIdx.x + offset];
      ids[threadIdx.x] = ids[threadIdx.x + offset];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0u) {
    best_ids[cue] = ids[0];
    best_scores[cue] = scores[0];
  }
}

__global__ void retain_close_cue_matches_kernel(
    const std::uint8_t* bytes, std::uint32_t byte_count, const std::uint32_t* starts,
    std::uint32_t cue_count, const std::uint32_t* unit_lengths,
    const std::uint32_t* unit_content, std::uint32_t unit_count,
    const std::uint32_t* best_scores, std::uint32_t* cue_scores,
    std::uint32_t* cue_orders) {
  const unsigned long long flat =
      static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  const unsigned long long total =
      static_cast<unsigned long long>(cue_count) * unit_count;
  if (flat >= total) return;
  const std::uint32_t cue = static_cast<std::uint32_t>(flat / unit_count);
  const std::uint32_t unit = static_cast<std::uint32_t>(flat -
      static_cast<unsigned long long>(cue) * unit_count);
  const std::uint32_t begin = starts[cue];
  const std::uint32_t end = cue + 1u < cue_count ? starts[cue + 1u] : byte_count;
  const std::uint32_t score = fuzzy_unit_score(bytes, begin, end - begin, unit,
                                               unit_lengths, unit_content);
  if (score >= kCueNearIdentity && static_cast<unsigned long long>(score) * 16u >=
                                      static_cast<unsigned long long>(best_scores[cue]) * 15u) {
    atomicMax(cue_scores + unit, score);
  }
}

__global__ void assign_cue_orders_kernel(
    const std::uint8_t* bytes, std::uint32_t byte_count, const std::uint32_t* starts,
    std::uint32_t cue_count, const std::uint32_t* unit_lengths,
    const std::uint32_t* unit_content, std::uint32_t unit_count,
    const std::uint32_t* cue_scores, std::uint32_t* cue_orders) {
  const unsigned long long flat =
      static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  const unsigned long long total =
      static_cast<unsigned long long>(cue_count) * unit_count;
  if (flat >= total) return;
  const std::uint32_t cue = static_cast<std::uint32_t>(flat / unit_count);
  const std::uint32_t unit = static_cast<std::uint32_t>(flat -
      static_cast<unsigned long long>(cue) * unit_count);
  if (cue_scores[unit] == 0u) return;
  const std::uint32_t begin = starts[cue];
  const std::uint32_t end = cue + 1u < cue_count ? starts[cue + 1u] : byte_count;
  const std::uint32_t score = fuzzy_unit_score(bytes, begin, end - begin, unit,
                                               unit_lengths, unit_content);
  if (score == cue_scores[unit]) atomicMin(cue_orders + unit, cue + 1u);
}

// Preserve which resident units were contacted by an exact body-surface
// segment. Fuzzy similarity remains useful associative evidence, but cannot
// stand in for the currently perceived topic (for example, "the" vs "them").
__global__ void mark_exact_cue_surfaces_kernel(
    const std::uint32_t* best_ids, const std::uint32_t* best_scores,
    std::uint32_t cue_count, const std::uint32_t* unit_lengths,
    const std::uint32_t* unit_content, std::uint32_t unit_count,
    std::uint32_t* cue_exact) {
  const std::uint32_t unit = blockIdx.x * blockDim.x + threadIdx.x;
  if (unit >= unit_count) return;
  for (std::uint32_t cue = 0u; cue < cue_count; ++cue) {
    const std::uint32_t representative = best_ids[cue];
    if (representative >= unit_count || best_scores[cue] != 0xffffu)
      continue;
    std::uint32_t representative_length = unit_lengths[representative];
    std::uint32_t unit_length = unit_lengths[unit];
    auto trailing_boundary = [](std::uint32_t byte) {
      return byte == static_cast<std::uint32_t>(' ') ||
             byte == static_cast<std::uint32_t>('\t') ||
             byte == static_cast<std::uint32_t>('\n') ||
             byte == static_cast<std::uint32_t>('\r') ||
             byte == static_cast<std::uint32_t>('.') ||
             byte == static_cast<std::uint32_t>(',') ||
             byte == static_cast<std::uint32_t>('!') ||
             byte == static_cast<std::uint32_t>('?') ||
             byte == static_cast<std::uint32_t>(';') ||
             byte == static_cast<std::uint32_t>(':');
    };
    while (representative_length != 0u) {
      const std::uint32_t offset = representative_length - 1u;
      const std::uint32_t packed =
          unit_content[representative * kUnitWords + offset / 4u];
      if (!trailing_boundary((packed >> ((offset % 4u) * 8u)) & 0xffu)) break;
      --representative_length;
    }
    while (unit_length != 0u) {
      const std::uint32_t offset = unit_length - 1u;
      const std::uint32_t packed =
          unit_content[unit * kUnitWords + offset / 4u];
      if (!trailing_boundary((packed >> ((offset % 4u) * 8u)) & 0xffu)) break;
      --unit_length;
    }
    if (representative_length == 0u ||
        representative_length != unit_length)
      continue;
    bool equal = true;
    for (std::uint32_t offset = 0u; offset < unit_length; ++offset) {
      const std::uint32_t lhs =
          (unit_content[representative * kUnitWords + offset / 4u] >>
           ((offset % 4u) * 8u)) & 0xffu;
      const std::uint32_t rhs =
          (unit_content[unit * kUnitWords + offset / 4u] >>
           ((offset % 4u) * 8u)) & 0xffu;
      equal &= lhs == rhs;
    }
    if (equal) {
      cue_exact[unit] = 1u;
      return;
    }
  }
}

// Retain physical cue order only for the exact representative that won each
// raw segment. The full exact-equivalence mask remains available to ordered
// topic matching; this compact trace lets consolidated tissue settle from the
// same sparse cue after the removable episode store is withdrawn.
__global__ void retain_exact_cue_representative_order_kernel(
    const std::uint32_t* best_ids, const std::uint32_t* best_scores,
    std::uint32_t cue_count, std::uint32_t unit_count,
    std::uint32_t* cue_orders) {
  const std::uint32_t cue = blockIdx.x * blockDim.x + threadIdx.x;
  if (cue >= cue_count || best_scores[cue] != 0xffffu)
    return;
  const std::uint32_t unit = best_ids[cue];
  if (unit < unit_count)
    atomicMin(cue_orders + unit, cue + 1u);
}

// Preserve the complete exact-contact sequence.  The per-unit order field
// above is intentionally a set-like compatibility view and loses duplicate
// units; proposition settlement needs the body event in its observed order.
__global__ void retain_exact_proposition_cue_sequence_kernel(
    const std::uint32_t* best_ids, const std::uint32_t* best_scores,
    std::uint32_t cue_count, std::uint32_t unit_count,
    std::uint32_t* sequence, std::uint32_t sequence_capacity,
    std::uint32_t* sequence_count) {
  if (blockIdx.x != 0u || threadIdx.x != 0u) return;
  sequence_count[0] = 0u;
  for (std::uint32_t cue = 0u; cue < cue_count; ++cue) {
    const std::uint32_t unit = best_ids[cue];
    if (best_scores[cue] != 0xffffu || unit >= unit_count) continue;
    if (sequence_count[0] >= sequence_capacity) return;
    sequence[sequence_count[0]++] = unit;
  }
}
