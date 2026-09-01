// Resident synthesis scoring and lookup helpers.
//
// Included after the synthesis model/workspace views and before reverse-edge
// materialization kernels. This unit owns bounded support lookup and scoring.

__device__ inline bool synthesis_source_window_present(
    const ResidentSourceWindowSignature* windows, std::uint32_t count,
    ResidentSourceWindowSignature target) {
  std::uint32_t lower = 0u;
  std::uint32_t upper = count;
  while (lower < upper) {
    const std::uint32_t middle = lower + (upper - lower) / 2u;
    const ResidentSourceWindowSignature value = windows[middle];
    if (value < target) {
      lower = middle + 1u;
    } else {
      upper = middle;
    }
  }
  return lower < count && windows[lower].hash_a == target.hash_a &&
      windows[lower].hash_b == target.hash_b;
}

__device__ inline unsigned long long synthesis_sat_add(
    unsigned long long a, unsigned long long b) {
  const unsigned long long out = a + b;
  return out < a ? ~0ull : out;
}

__device__ inline unsigned long long synthesis_sat_mul(
    unsigned long long a, unsigned long long b) {
  if (a == 0ull || b == 0ull) return 0ull;
  return a > ~0ull / b ? ~0ull : a * b;
}

__device__ inline std::uint32_t synthesis_depth(unsigned long long value) {
  std::uint32_t depth = 0u;
  while (value != 0ull) {
    ++depth;
    value >>= 1u;
  }
  return depth;
}

__device__ inline unsigned long long synthesis_fit_priority_score(
    unsigned long long relevance, unsigned long long online_support) {
  const unsigned long long bounded_relevance =
      relevance > 0xffffull ? 0xffffull : relevance;
  return synthesis_sat_add(bounded_relevance << 48u, online_support);
}

__device__ inline std::uint8_t synthesis_unit_byte(
    const std::uint32_t* content, std::uint32_t unit_words,
    std::uint32_t unit, std::uint32_t offset) {
  const std::uint32_t packed = content[unit * unit_words + offset / 4u];
  return static_cast<std::uint8_t>((packed >> (8u * (offset & 3u))) & 0xffu);
}

__device__ inline bool synthesis_unit_contains(
    const std::uint32_t* lengths, const std::uint32_t* content,
    std::uint32_t unit_words, std::uint32_t unit, const std::uint32_t* values,
    std::uint32_t value_count) {
  if (values == nullptr) return false;
  for (std::uint32_t i = 0u; i < lengths[unit]; ++i) {
    const std::uint32_t byte = synthesis_unit_byte(content, unit_words, unit, i);
    for (std::uint32_t j = 0u; j < value_count; ++j) {
      if (byte == values[j]) return true;
    }
  }
  return false;
}

__device__ inline bool synthesis_unit_is_boundary(
    const std::uint32_t* lengths, const std::uint32_t* content,
    std::uint32_t unit_words, std::uint32_t unit,
    const std::uint32_t* boundary_bytes, std::uint32_t boundary_count) {
  if (boundary_bytes == nullptr || boundary_count == 0u || lengths[unit] == 0u) {
    return false;
  }
  for (std::uint32_t i = 0u; i < lengths[unit]; ++i) {
    const std::uint32_t byte = synthesis_unit_byte(content, unit_words, unit, i);
    bool boundary = false;
    for (std::uint32_t j = 0u; j < boundary_count; ++j) {
      boundary |= byte == boundary_bytes[j];
    }
    if (!boundary) return false;
  }
  return true;
}

template <typename BigramKeyT>
__device__ inline std::uint32_t lower_forward_bigram(
    const BigramKeyT* keys, std::uint32_t count, std::uint32_t previous) {
  std::uint32_t lo = 0u;
  std::uint32_t hi = count;
  while (lo < hi) {
    const std::uint32_t mid = lo + (hi - lo) / 2u;
    if (keys[mid].previous < previous) lo = mid + 1u;
    else hi = mid;
  }
  return lo;
}

template <typename BigramKeyT>
__device__ inline std::uint32_t upper_forward_bigram(
    const BigramKeyT* keys, std::uint32_t count, std::uint32_t previous) {
  std::uint32_t lo = 0u;
  std::uint32_t hi = count;
  while (lo < hi) {
    const std::uint32_t mid = lo + (hi - lo) / 2u;
    if (keys[mid].previous <= previous) lo = mid + 1u;
    else hi = mid;
  }
  return lo;
}

template <typename TrigramKeyT>
__device__ inline std::uint32_t lower_forward_trigram(
    const TrigramKeyT* keys, std::uint32_t count, std::uint32_t first,
    std::uint32_t second) {
  std::uint32_t lo = 0u;
  std::uint32_t hi = count;
  while (lo < hi) {
    const std::uint32_t mid = lo + (hi - lo) / 2u;
    const bool before = keys[mid].first < first ||
        (keys[mid].first == first && keys[mid].second < second);
    if (before) lo = mid + 1u;
    else hi = mid;
  }
  return lo;
}

template <typename TrigramKeyT>
__device__ inline std::uint32_t upper_forward_trigram(
    const TrigramKeyT* keys, std::uint32_t count, std::uint32_t first,
    std::uint32_t second) {
  std::uint32_t lo = 0u;
  std::uint32_t hi = count;
  while (lo < hi) {
    const std::uint32_t mid = lo + (hi - lo) / 2u;
    const bool at_or_before = keys[mid].first < first ||
        (keys[mid].first == first && keys[mid].second <= second);
    if (at_or_before) lo = mid + 1u;
    else hi = mid;
  }
  return lo;
}

__device__ inline std::uint32_t lower_subject_transition(
    const ResidentSubjectTransitionKey* keys, std::uint32_t count,
    std::uint32_t anchor, std::uint32_t previous) {
  std::uint32_t lo = 0u;
  std::uint32_t hi = count;
  while (lo < hi) {
    const std::uint32_t mid = lo + (hi - lo) / 2u;
    const bool before = keys[mid].anchor < anchor ||
        (keys[mid].anchor == anchor && keys[mid].previous < previous);
    if (before) lo = mid + 1u;
    else hi = mid;
  }
  return lo;
}

__device__ inline std::uint32_t upper_subject_transition(
    const ResidentSubjectTransitionKey* keys, std::uint32_t count,
    std::uint32_t anchor, std::uint32_t previous) {
  std::uint32_t lo = 0u;
  std::uint32_t hi = count;
  while (lo < hi) {
    const std::uint32_t mid = lo + (hi - lo) / 2u;
    const bool at_or_before = keys[mid].anchor < anchor ||
        (keys[mid].anchor == anchor && keys[mid].previous <= previous);
    if (at_or_before) lo = mid + 1u;
    else hi = mid;
  }
  return lo;
}

__device__ inline std::uint32_t synthesis_subject_transition_count(
    const ResidentSubjectTransitionKey* keys, const std::uint32_t* counts,
    std::uint32_t count, std::uint32_t anchor, std::uint32_t previous,
    std::uint32_t next) {
  if (keys == nullptr || counts == nullptr) return 0u;
  std::uint32_t lo = lower_subject_transition(keys, count, anchor, previous);
  std::uint32_t hi = upper_subject_transition(keys, count, anchor, previous);
  while (lo < hi) {
    const std::uint32_t mid = lo + (hi - lo) / 2u;
    if (keys[mid].next < next) lo = mid + 1u;
    else hi = mid;
  }
  if (lo < count && keys[lo].anchor == anchor &&
      keys[lo].previous == previous && keys[lo].next == next) {
    return counts[lo];
  }
  return 0u;
}

template <typename BigramKeyT, typename TrigramKeyT>
__device__ inline bool synthesis_subject_is_selected(
    const ResidentSynthesisModelView<BigramKeyT, TrigramKeyT>& model,
    std::uint32_t subject) {
  if (model.subject_anchors == nullptr || subject >= model.unit_count) return false;
  for (std::uint32_t index = 0u; index < model.subject_anchor_count; ++index) {
    if (model.subject_anchors[index] == subject) return true;
  }
  return false;
}

template <typename BigramKeyT>
__device__ inline std::uint32_t synthesis_bigram_count(
    const BigramKeyT* keys, const std::uint32_t* counts, std::uint32_t count,
    std::uint32_t previous, std::uint32_t next) {
  if (keys == nullptr || counts == nullptr) return 0u;
  std::uint32_t lo = lower_forward_bigram(keys, count, previous);
  std::uint32_t hi = upper_forward_bigram(keys, count, previous);
  while (lo < hi) {
    const std::uint32_t mid = lo + (hi - lo) / 2u;
    if (keys[mid].next < next) lo = mid + 1u;
    else hi = mid;
  }
  if (lo < count && keys[lo].previous == previous && keys[lo].next == next) {
    return counts[lo];
  }
  return 0u;
}

template <typename TrigramKeyT>
__device__ inline std::uint32_t synthesis_trigram_count(
    const TrigramKeyT* keys, const std::uint32_t* counts, std::uint32_t count,
    std::uint32_t first, std::uint32_t second, std::uint32_t next) {
  if (keys == nullptr || counts == nullptr) return 0u;
  std::uint32_t lo = lower_forward_trigram(keys, count, first, second);
  std::uint32_t hi = upper_forward_trigram(keys, count, first, second);
  while (lo < hi) {
    const std::uint32_t mid = lo + (hi - lo) / 2u;
    if (keys[mid].next < next) lo = mid + 1u;
    else hi = mid;
  }
  if (lo < count && keys[lo].first == first && keys[lo].second == second &&
      keys[lo].next == next) {
    return counts[lo];
  }
  return 0u;
}

template <typename BigramKeyT, typename TrigramKeyT>
__device__ inline std::uint32_t synthesis_combined_bigram(
    const ResidentSynthesisModelView<BigramKeyT, TrigramKeyT>& model,
    std::uint32_t previous, std::uint32_t next) {
  const unsigned long long total =
      static_cast<unsigned long long>(synthesis_bigram_count(
          model.base_bigrams, model.base_bigram_counts, model.base_bigram_count,
          previous, next)) +
      synthesis_bigram_count(model.online_bigrams, model.online_bigram_counts,
                             model.online_bigram_count, previous, next);
  return total > 0xffffffffull ? 0xffffffffu : static_cast<std::uint32_t>(total);
}

template <typename BigramKeyT, typename TrigramKeyT>
__device__ inline std::uint32_t synthesis_combined_trigram(
    const ResidentSynthesisModelView<BigramKeyT, TrigramKeyT>& model,
    std::uint32_t first, std::uint32_t second, std::uint32_t next) {
  const unsigned long long total =
      static_cast<unsigned long long>(synthesis_trigram_count(
          model.base_trigrams, model.base_trigram_counts,
          model.base_trigram_count, first, second, next)) +
      synthesis_trigram_count(model.online_trigrams, model.online_trigram_counts,
                              model.online_trigram_count, first, second, next);
  return total > 0xffffffffull ? 0xffffffffu : static_cast<std::uint32_t>(total);
}

template <typename BigramKeyT, typename TrigramKeyT>
__device__ inline unsigned long long synthesis_role_support(
    const ResidentSynthesisModelView<BigramKeyT, TrigramKeyT>& model,
    const std::uint64_t* role_bigrams, const std::uint64_t* role_trigrams,
    std::uint32_t first, std::uint32_t second, std::uint32_t next) {
  if (model.unit_roles == nullptr || model.role_count == 0u ||
      role_bigrams == nullptr || second >= model.unit_count ||
      next >= model.unit_count) {
    return 0ull;
  }
  const auto second_state = model.unit_roles[second];
  const auto next_state = model.unit_roles[next];
  const std::uint32_t second_role = second_state.role;
  const std::uint32_t next_role = next_state.role;
  if (second_role >= model.role_count || next_role >= model.role_count) return 0ull;
  unsigned long long support = role_bigrams[
      static_cast<std::size_t>(second_role) * model.role_count + next_role];
  std::uint32_t confidence = second_state.confidence < next_state.confidence
      ? second_state.confidence : next_state.confidence;
  if (first != kResidentSynthesisInvalid && first < model.unit_count &&
      role_trigrams != nullptr) {
    const auto first_state = model.unit_roles[first];
    const std::uint32_t first_role = first_state.role;
    if (first_role < model.role_count) {
      const std::size_t index =
          (static_cast<std::size_t>(first_role) * model.role_count + second_role) *
              model.role_count +
          next_role;
      support += 4ull * role_trigrams[index];
      if (first_state.confidence < confidence) confidence = first_state.confidence;
    }
  }
  return support > (~0ull) / (64u + confidence)
      ? ~0ull
      : support * (64u + confidence) / 64u;
}

__device__ inline std::uint32_t lower_reverse_bigram(
    const ReverseBigramEdge* edges, std::uint32_t count, std::uint32_t current) {
  std::uint32_t lo = 0u;
  std::uint32_t hi = count;
  while (lo < hi) {
    const std::uint32_t mid = lo + (hi - lo) / 2u;
    if (edges[mid].current < current) lo = mid + 1u;
    else hi = mid;
  }
  return lo;
}

__device__ inline std::uint32_t upper_reverse_bigram(
    const ReverseBigramEdge* edges, std::uint32_t count, std::uint32_t current) {
  std::uint32_t lo = 0u;
  std::uint32_t hi = count;
  while (lo < hi) {
    const std::uint32_t mid = lo + (hi - lo) / 2u;
    if (edges[mid].current <= current) lo = mid + 1u;
    else hi = mid;
  }
  return lo;
}

__device__ inline std::uint32_t lower_reverse_trigram(
    const ReverseTrigramEdge* edges, std::uint32_t count, std::uint32_t second,
    std::uint32_t next) {
  std::uint32_t lo = 0u;
  std::uint32_t hi = count;
  while (lo < hi) {
    const std::uint32_t mid = lo + (hi - lo) / 2u;
    const bool before = edges[mid].second < second ||
        (edges[mid].second == second && edges[mid].next < next);
    if (before) lo = mid + 1u;
    else hi = mid;
  }
  return lo;
}

__device__ inline std::uint32_t upper_reverse_trigram(
    const ReverseTrigramEdge* edges, std::uint32_t count, std::uint32_t second,
    std::uint32_t next) {
  std::uint32_t lo = 0u;
  std::uint32_t hi = count;
  while (lo < hi) {
    const std::uint32_t mid = lo + (hi - lo) / 2u;
    const bool at_or_before = edges[mid].second < second ||
        (edges[mid].second == second && edges[mid].next <= next);
    if (at_or_before) lo = mid + 1u;
    else hi = mid;
  }
  return lo;
}
