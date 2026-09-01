// Adult base-anchor completion and resident synthesis-policy kernels.
//
// Included from bcc32_cuda_adult_v1.cuh inside its namespace and state-only
// guard as one dependency-ordered completion responsibility.
__global__ void select_base_anchor_kernel(const std::uint32_t* cue_scores,
                                          const std::uint32_t* vitality,
                                          const std::uint32_t* posting_offsets,
                                          std::uint32_t unit_count,
                                          std::uint32_t* anchor) {
  std::uint32_t local_id = 0xffffffffu;
  unsigned long long local_rank = 0u;
  for (std::uint32_t unit = threadIdx.x; unit < unit_count; unit += blockDim.x) {
    const std::uint32_t frequency = posting_offsets[unit + 1u] - posting_offsets[unit];
    if (cue_scores[unit] < 32u || frequency == 0u || frequency > 2048u) continue;
    const unsigned long long rank =
        static_cast<unsigned long long>(cue_scores[unit]) * 1048576ull /
        max(1u, vitality[unit]);
    if (rank > local_rank || (rank == local_rank && unit < local_id)) {
      local_rank = rank;
      local_id = unit;
    }
  }
  __shared__ std::uint32_t ids[kBlock];
  __shared__ unsigned long long ranks[kBlock];
  ids[threadIdx.x] = local_id;
  ranks[threadIdx.x] = local_rank;
  __syncthreads();
  for (std::uint32_t offset = blockDim.x / 2u; offset != 0u; offset >>= 1u) {
    if (threadIdx.x < offset &&
        (ranks[threadIdx.x + offset] > ranks[threadIdx.x] ||
         (ranks[threadIdx.x + offset] == ranks[threadIdx.x] &&
          ids[threadIdx.x + offset] < ids[threadIdx.x]))) {
      ranks[threadIdx.x] = ranks[threadIdx.x + offset];
      ids[threadIdx.x] = ids[threadIdx.x + offset];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0u) anchor[0] = ranks[0] == 0u ? 0xffffffffu : ids[0];
}

__global__ void select_indexed_base_completion_kernel(
    const std::uint32_t* episode_units, std::uint32_t episode_count,
    const std::uint32_t* posting_positions, const std::uint32_t* posting_offsets,
    const std::uint32_t* anchor, const std::uint32_t* cue_scores,
    const std::uint32_t* cue_orders, const std::uint32_t* vitality,
    std::uint32_t* motor_context, std::uint32_t* motor_completion) {
  const std::uint32_t anchor_unit = anchor[0];
  if (anchor_unit == 0xffffffffu) return;
  const std::uint32_t posting_begin = posting_offsets[anchor_unit];
  const std::uint32_t posting_end = posting_offsets[anchor_unit + 1u];
  std::uint32_t local_position = 0xffffffffu;
  std::uint32_t local_pivot = 0xffffffffu;
  unsigned long long local_score = 0u;
  for (std::uint32_t posting = posting_begin + threadIdx.x; posting < posting_end;
       posting += blockDim.x) {
    const std::uint32_t position = posting_positions[posting];
    const std::uint32_t window_begin = position > kCueWindowRadius
        ? position - kCueWindowRadius : 0u;
    const std::uint32_t window_end = min(episode_count, position + kCueWindowRadius + 1u);
    std::uint32_t matches = 0u;
    std::uint32_t ordered_pairs = 0u;
    std::uint32_t inversions = 0u;
    std::uint32_t earliest_match = position;
    std::uint32_t orders[2u * kCueWindowRadius + 1u] = {};
    unsigned long long score = 0u;
    for (std::uint32_t i = window_begin; i < window_end; ++i) {
      const std::uint32_t unit = episode_units[i];
      const std::uint32_t cue_score = cue_scores[unit];
      if (cue_score == 0u) continue;
      const std::uint32_t distance = i > position ? i - position : position - i;
      score += static_cast<unsigned long long>(cue_score) *
               (kCueWindowRadius + 1u - min(kCueWindowRadius, distance)) *
               (4096u / max(1u, min(vitality[unit], 4096u)));
      if (cue_score < 32u || vitality[unit] > 2048u) continue;
      const std::uint32_t order = cue_orders[unit];
      for (std::uint32_t prior = 0u; prior < matches; ++prior) {
        if (order > orders[prior]) ++ordered_pairs;
        else if (order < orders[prior]) ++inversions;
      }
      orders[matches++] = order;
      earliest_match = min(earliest_match, i);
    }
    if (matches < 2u || inversions > 1u || ordered_pairs < inversions * 2u + 1u) continue;
    score += static_cast<unsigned long long>(matches) * matches * 4194304u;
    score += static_cast<unsigned long long>(ordered_pairs) * 1048576u;
    if (score > local_score || (score == local_score && position < local_position)) {
      local_score = score;
      local_position = position;
      local_pivot = earliest_match;
    }
  }

  __shared__ std::uint32_t positions[kBlock];
  __shared__ std::uint32_t pivots[kBlock];
  __shared__ unsigned long long scores[kBlock];
  positions[threadIdx.x] = local_position;
  pivots[threadIdx.x] = local_pivot;
  scores[threadIdx.x] = local_score;
  __syncthreads();
  for (std::uint32_t offset = blockDim.x / 2u; offset != 0u; offset >>= 1u) {
    if (threadIdx.x < offset &&
        (scores[threadIdx.x + offset] > scores[threadIdx.x] ||
         (scores[threadIdx.x + offset] == scores[threadIdx.x] &&
          positions[threadIdx.x + offset] < positions[threadIdx.x]))) {
      scores[threadIdx.x] = scores[threadIdx.x + offset];
      positions[threadIdx.x] = positions[threadIdx.x + offset];
      pivots[threadIdx.x] = pivots[threadIdx.x + offset];
    }
    __syncthreads();
  }
  if (threadIdx.x != 0u || scores[0] == 0u || pivots[0] == 0xffffffffu) return;
  const std::uint32_t completion_begin = pivots[0] + 1u;
  const std::uint32_t completion_count =
      min(kCompositionUnits, episode_count - completion_begin);
  for (std::uint32_t i = 0u; i < completion_count; ++i) {
    motor_completion[i] = episode_units[completion_begin + i];
  }
  motor_context[0] = 1u;
  motor_context[1] = episode_units[pivots[0]];
  motor_context[2] = static_cast<std::uint32_t>(min(scores[0], 0xffffffffull));
  motor_context[3] = completion_count;
  motor_context[4] = positions[0];
  motor_context[5] = 3u;
}

__device__ std::uint32_t lower_bigram(const BigramKey* keys, std::uint32_t count,
                                     std::uint32_t previous) {
  std::uint32_t lo = 0u, hi = count;
  while (lo < hi) {
    const std::uint32_t mid = lo + (hi - lo) / 2u;
    if (keys[mid].previous < previous) lo = mid + 1u; else hi = mid;
  }
  return lo;
}

__device__ std::uint32_t upper_bigram(const BigramKey* keys, std::uint32_t count,
                                     std::uint32_t previous) {
  std::uint32_t lo = 0u, hi = count;
  while (lo < hi) {
    const std::uint32_t mid = lo + (hi - lo) / 2u;
    if (keys[mid].previous <= previous) lo = mid + 1u; else hi = mid;
  }
  return lo;
}

__device__ std::uint32_t lower_trigram(const TrigramKey* keys, std::uint32_t count,
                                      std::uint32_t first, std::uint32_t second) {
  std::uint32_t lo = 0u, hi = count;
  while (lo < hi) {
    const std::uint32_t mid = lo + (hi - lo) / 2u;
    const bool before = keys[mid].first < first ||
                        (keys[mid].first == first && keys[mid].second < second);
    if (before) lo = mid + 1u; else hi = mid;
  }
  return lo;
}

__device__ std::uint32_t upper_trigram(const TrigramKey* keys, std::uint32_t count,
                                      std::uint32_t first, std::uint32_t second) {
  std::uint32_t lo = 0u, hi = count;
  while (lo < hi) {
    const std::uint32_t mid = lo + (hi - lo) / 2u;
    const bool at_or_before = keys[mid].first < first ||
        (keys[mid].first == first && keys[mid].second <= second);
    if (at_or_before) lo = mid + 1u; else hi = mid;
  }
  return lo;
}

__device__ std::uint32_t find_cached_bigram_context(const std::uint32_t* contexts,
                                                    std::uint32_t count,
                                                    std::uint32_t wanted) {
  std::uint32_t lo = 0u, hi = count;
  while (lo < hi) {
    const std::uint32_t mid = lo + (hi - lo) / 2u;
    if (contexts[mid] < wanted) lo = mid + 1u; else hi = mid;
  }
  return lo < count && contexts[lo] == wanted ? lo : 0xffffffffu;
}

__device__ std::uint32_t find_cached_trigram_context(const BigramKey* contexts,
                                                     std::uint32_t count,
                                                     std::uint32_t first,
                                                     std::uint32_t second) {
  const BigramKey wanted{first, second};
  std::uint32_t lo = 0u, hi = count;
  while (lo < hi) {
    const std::uint32_t mid = lo + (hi - lo) / 2u;
    if (contexts[mid] < wanted) lo = mid + 1u; else hi = mid;
  }
  return lo < count && contexts[lo] == wanted ? lo : 0xffffffffu;
}

__device__ std::uint32_t resident_bigram_count(
    const BigramKey* keys, const std::uint32_t* counts, std::uint32_t count,
    std::uint32_t previous, std::uint32_t next) {
  const BigramKey wanted{previous, next};
  std::uint32_t lo = 0u, hi = count;
  while (lo < hi) {
    const std::uint32_t mid = lo + (hi - lo) / 2u;
    if (keys[mid] < wanted) lo = mid + 1u; else hi = mid;
  }
  return lo < count && keys[lo] == wanted ? counts[lo] : 0u;
}

__device__ std::uint32_t resident_trigram_count(
    const TrigramKey* keys, const std::uint32_t* counts, std::uint32_t count,
    std::uint32_t first, std::uint32_t second, std::uint32_t next) {
  const TrigramKey wanted{first, second, next};
  std::uint32_t lo = 0u, hi = count;
  while (lo < hi) {
    const std::uint32_t mid = lo + (hi - lo) / 2u;
    if (keys[mid] < wanted) lo = mid + 1u; else hi = mid;
  }
  return lo < count && keys[lo] == wanted ? counts[lo] : 0u;
}

__device__ std::uint32_t resident_synthesis_policy_candidate(
    std::uint32_t first, std::uint32_t second, std::uint32_t weight,
    const BigramKey* base_bigrams, const std::uint32_t* base_bigram_counts,
    std::uint32_t base_bigram_count,
    const TrigramKey* base_trigrams, const std::uint32_t* base_trigram_counts,
    std::uint32_t base_trigram_count,
    const BigramKey* online_bigrams, const std::uint32_t* online_bigram_counts,
    std::uint32_t online_bigram_count,
    const TrigramKey* online_trigrams,
    const std::uint32_t* online_trigram_counts,
    std::uint32_t online_trigram_count) {
  std::uint32_t best = 0xffffffffu;
  unsigned long long best_score = 0ull;
  const std::uint32_t base_begin =
      lower_bigram(base_bigrams, base_bigram_count, second);
  const std::uint32_t base_end =
      upper_bigram(base_bigrams, base_bigram_count, second);
  const std::uint32_t online_begin =
      lower_bigram(online_bigrams, online_bigram_count, second);
  const std::uint32_t online_end =
      upper_bigram(online_bigrams, online_bigram_count, second);
  for (std::uint32_t source = 0u; source < 2u; ++source) {
    const std::uint32_t begin = source == 0u ? base_begin : online_begin;
    const std::uint32_t end = source == 0u ? base_end : online_end;
    for (std::uint32_t edge = begin; edge < end; ++edge) {
      const std::uint32_t candidate = source == 0u
          ? base_bigrams[edge].next : online_bigrams[edge].next;
      const unsigned long long bigram =
          resident_bigram_count(base_bigrams, base_bigram_counts,
                                base_bigram_count, second, candidate) +
          resident_bigram_count(online_bigrams, online_bigram_counts,
                                online_bigram_count, second, candidate);
      const unsigned long long trigram =
          resident_trigram_count(base_trigrams, base_trigram_counts,
                                 base_trigram_count, first, second, candidate) +
          resident_trigram_count(online_trigrams, online_trigram_counts,
                                 online_trigram_count, first, second, candidate);
      const unsigned long long score = bigram + weight * trigram;
      if (score > best_score ||
          (score == best_score && score != 0ull && candidate < best)) {
        best = candidate;
        best_score = score;
      }
    }
  }
  return best;
}

__global__ void advance_resident_synthesis_policy_kernel(
    const std::uint32_t* sequence, std::uint32_t sequence_count,
    const BigramKey* base_bigrams, const std::uint32_t* base_bigram_counts,
    std::uint32_t base_bigram_count,
    const TrigramKey* base_trigrams, const std::uint32_t* base_trigram_counts,
    std::uint32_t base_trigram_count,
    const BigramKey* online_bigrams, const std::uint32_t* online_bigram_counts,
    std::uint32_t online_bigram_count,
    const TrigramKey* online_trigrams,
    const std::uint32_t* online_trigram_counts,
    std::uint32_t online_trigram_count, const std::uint32_t* exact_replay,
    bcc32_cuda_resident_synthesis::ResidentSynthesisPolicyState* policy) {
  namespace synthesis = bcc32_cuda_resident_synthesis;
  if (blockIdx.x != 0u || threadIdx.x != 0u ||
      policy == nullptr || sequence_count < 3u || exact_replay[0] != 0u) {
    return;
  }
  const std::uint32_t available = sequence_count - 2u;
  const std::uint32_t event_count =
      min(available, synthesis::kResidentSynthesisPolicyWindow);
  const std::uint32_t begin = sequence_count - event_count;
  for (std::uint32_t index = begin; index < sequence_count; ++index) {
    std::uint32_t candidates[synthesis::kResidentSynthesisPolicyVariants]{};
    for (std::uint32_t variant = 0u;
         variant < synthesis::kResidentSynthesisPolicyVariants; ++variant) {
      candidates[variant] = resident_synthesis_policy_candidate(
          sequence[index - 2u], sequence[index - 1u],
          synthesis::resident_synthesis_policy_weight(variant),
          base_bigrams, base_bigram_counts, base_bigram_count,
          base_trigrams, base_trigram_counts, base_trigram_count,
          online_bigrams, online_bigram_counts, online_bigram_count,
          online_trigrams, online_trigram_counts, online_trigram_count);
    }
    (void)synthesis::resident_synthesis_policy_predict(policy, candidates);
    (void)synthesis::resident_synthesis_policy_observe(policy, sequence[index]);
  }
}
