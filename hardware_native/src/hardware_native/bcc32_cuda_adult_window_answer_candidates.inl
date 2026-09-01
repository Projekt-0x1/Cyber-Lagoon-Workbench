__global__ void build_window_signatures_kernel(const std::uint32_t* units,
                                               std::uint32_t unit_count,
                                               std::uint32_t window_count,
                                               WindowSignature* signatures) {
  const std::uint32_t window = blockIdx.x * blockDim.x + threadIdx.x;
  if (window >= window_count) return;
  const std::uint32_t position = window * kBaseWindowStride;
  const std::uint32_t end = min(unit_count, position + kBaseWindowUnits);
  std::uint32_t a = 0x811c9dc5u;
  std::uint32_t b = 0x9e3779b9u;
  for (std::uint32_t i = position; i < end; ++i) {
    a = mix32(a ^ units[i] ^ (i - position + 1u));
    b = mix32(b + units[i] * 0x85ebca6bu + (i - position));
  }
  signatures[window] = WindowSignature{a, b, position};
}

__global__ void build_resident_source_window_signatures_kernel(
    const std::uint32_t* units, std::uint32_t unit_count,
    bcc32_cuda_resident_synthesis::ResidentSourceWindowSignature* signatures,
    std::uint32_t output_offset) {
  const std::uint32_t begin = blockIdx.x * blockDim.x + threadIdx.x;
  if (begin + bcc32_cuda_resident_synthesis::kResidentSynthesisNoveltyUnits >
      unit_count) {
    return;
  }
  signatures[output_offset + begin] =
      bcc32_cuda_resident_synthesis::synthesis_source_window_signature(
          units + begin);
}

__global__ void build_answer_frame_source_window_signatures_kernel(
    const std::uint32_t* units, std::uint32_t unit_count,
    const std::uint32_t* episode_breaks, std::uint32_t episode_break_count,
    answer_frame::SourceWindowSignature* signatures,
    std::uint32_t* signature_count) {
  constexpr std::uint32_t window = 4u;
  const std::uint32_t begin = blockIdx.x * blockDim.x + threadIdx.x;
  if (begin + window > unit_count) return;

  if (episode_break_count != 0u) {
    std::uint32_t lo = 0u;
    std::uint32_t hi = episode_break_count;
    while (lo < hi) {
      const std::uint32_t mid = lo + (hi - lo) / 2u;
      if (episode_breaks[mid] <= begin)
        lo = mid + 1u;
      else
        hi = mid;
    }
    if (lo < episode_break_count && episode_breaks[lo] < begin + window)
      return;
  }

  const std::uint32_t output = atomicAdd(signature_count, 1u);
  signatures[output] = answer_frame::source_window_signature(units + begin, window);
}

__global__ void build_answer_frame_unit_flags_kernel(
    const std::uint32_t* unit_lengths, const std::uint32_t* unit_content,
    std::uint32_t unit_count, const std::uint32_t* closure_bytes,
    const std::uint32_t* boundary_bytes, std::uint32_t boundary_count,
    std::uint32_t* unit_flags) {
  const std::uint32_t unit = blockIdx.x * blockDim.x + threadIdx.x;
  if (unit >= unit_count) return;
  std::uint32_t flags = unit_lengths[unit] == 0u ? 0u : answer_frame::kUnitEligible;
  if (flags != 0u && resident_unit_contains_any(
                          unit_lengths, unit_content, unit, closure_bytes,
                          kClosureCount)) {
    flags |= answer_frame::kUnitClosure;
    bool terminal_only = true;
    for (std::uint32_t offset = 0u; offset < unit_lengths[unit]; ++offset) {
      const std::uint32_t packed =
          unit_content[unit * kUnitWords + offset / 4u];
      const std::uint32_t byte = (packed >> (8u * (offset & 3u))) & 0xffu;
      bool structural = false;
      for (std::uint32_t i = 0u; i < kClosureCount; ++i)
        structural |= byte == closure_bytes[i];
      for (std::uint32_t i = 0u; i < boundary_count; ++i)
        structural |= byte == boundary_bytes[i];
      terminal_only &= structural;
    }
    if (terminal_only) flags |= answer_frame::kUnitTerminalOnly;
  }
  unit_flags[unit] = flags;
}

__device__ inline void append_answer_frame_candidate(
    std::uint32_t unit, std::uint32_t unit_count,
    std::uint32_t* candidates, std::uint32_t capacity,
    std::uint32_t* count) {
  if (unit >= unit_count || *count >= capacity) return;
  for (std::uint32_t existing = 0u; existing < *count; ++existing)
    if (candidates[existing] == unit) return;
  candidates[(*count)++] = unit;
}

__device__ inline bool answer_frame_candidate_in_prefix(
    std::uint32_t unit, const std::uint32_t* candidates,
    std::uint32_t prefix_count) {
  for (std::uint32_t index = 0u; index < prefix_count; ++index)
    if (candidates[index] == unit) return true;
  return false;
}

__global__ void build_answer_frame_candidate_set_kernel(
    const answer_frame::MutableSelectionState* selection,
    const std::uint32_t* relation_tail,
    const std::uint32_t* relation_tail_count,
    const BigramKey* base_bigrams, const std::uint32_t* base_bigram_counts,
    std::uint32_t base_bigram_count, const BigramKey* online_bigrams,
    const std::uint32_t* online_bigram_counts,
    std::uint32_t online_bigram_count,
    const std::uint32_t* resident_top, std::uint32_t resident_top_count,
    const std::uint32_t* unit_flags, const std::uint32_t* unit_lengths,
    const std::uint32_t* unit_vitality,
    std::uint32_t unit_count, std::uint32_t* candidates,
    std::uint32_t capacity, std::uint32_t* candidate_count) {
  if (blockIdx.x != 0u || threadIdx.x != 0u) return;
  std::uint32_t count = 0u;
  constexpr std::uint32_t closure_limit = 24u;
  std::uint32_t closure_units[closure_limit];
  std::uint32_t closure_scores[closure_limit];
  for (std::uint32_t i = 0u; i < closure_limit; ++i) {
    closure_units[i] = answer_frame::kInvalidUnit;
    closure_scores[i] = 0u;
  }
  for (std::uint32_t unit = 0u; unit < unit_count; ++unit) {
    if ((unit_flags[unit] & answer_frame::kUnitTerminalOnly) == 0u) continue;
    const std::uint32_t score =
        (64u - min(unit_lengths[unit], 64u)) * 0x01000000u +
        min(unit_vitality[unit], 0x00ffffffu);
    for (std::uint32_t rank = 0u; rank < closure_limit; ++rank) {
      if (score > closure_scores[rank] ||
          (score == closure_scores[rank] && unit < closure_units[rank])) {
        for (std::uint32_t move = closure_limit - 1u; move > rank; --move) {
          closure_units[move] = closure_units[move - 1u];
          closure_scores[move] = closure_scores[move - 1u];
        }
        closure_units[rank] = unit;
        closure_scores[rank] = score;
        break;
      }
    }
  }
  if (selection != nullptr && selection->intact != 0u) {
    const std::uint32_t subject = selection->subject;
    const std::uint32_t predicate = selection->predicate;
    const std::uint32_t value = selection->value;
    append_answer_frame_candidate(
        subject, unit_count, candidates, capacity, &count);
    append_answer_frame_candidate(
        predicate, unit_count, candidates, capacity, &count);
    append_answer_frame_candidate(
        value, unit_count, candidates, capacity, &count);
    std::uint32_t tail_count = 0u;
    if (relation_tail != nullptr && relation_tail_count != nullptr) {
      tail_count = min(
          *relation_tail_count,
          bcc32_cuda_resident_synthesis::kResidentSynthesisMaxRelationTail);
      for (std::uint32_t i = 0u; i < tail_count && count < capacity; ++i) {
        append_answer_frame_candidate(
            relation_tail[i], unit_count, candidates, capacity, &count);
      }
    }
    const BigramKey* tables[2] = {online_bigrams, base_bigrams};
    const std::uint32_t* count_tables[2] = {
        online_bigram_counts, base_bigram_counts};
    const std::uint32_t table_counts[2] = {
        online_bigram_count, base_bigram_count};

    // Candidate threads are bounded. Rank literal predecessors of the first
    // causal-tail unit before broad graph expansion so grammatical connectors
    // and direct-answer closure matter both remain physically reachable.
    constexpr std::uint32_t connector_limit = 32u;
    std::uint32_t connector_units[connector_limit];
    unsigned long long connector_scores[connector_limit];
    for (std::uint32_t i = 0u; i < connector_limit; ++i) {
      connector_units[i] = answer_frame::kInvalidUnit;
      connector_scores[i] = 0ull;
    }
    if (tail_count != 0u && relation_tail[0] < unit_count) {
      const std::uint32_t target = relation_tail[0];
      for (std::uint32_t table = 0u; table < 2u; ++table) {
        for (std::uint32_t edge = 0u; edge < table_counts[table]; ++edge) {
          const BigramKey relation = tables[table][edge];
          if (relation.next != target || relation.previous >= unit_count ||
              (unit_flags[relation.previous] & answer_frame::kUnitClosure) != 0u) {
            continue;
          }
          unsigned long long support = count_tables[table][edge];
          for (std::uint32_t rank = 0u; rank < connector_limit; ++rank) {
            if (connector_units[rank] != relation.previous) continue;
            support += connector_scores[rank];
            for (std::uint32_t move = rank; move + 1u < connector_limit; ++move) {
              connector_units[move] = connector_units[move + 1u];
              connector_scores[move] = connector_scores[move + 1u];
            }
            connector_units[connector_limit - 1u] = answer_frame::kInvalidUnit;
            connector_scores[connector_limit - 1u] = 0ull;
            break;
          }
          for (std::uint32_t rank = 0u; rank < connector_limit; ++rank) {
            if (support > connector_scores[rank] ||
                (support == connector_scores[rank] && support != 0ull &&
                 relation.previous < connector_units[rank])) {
              for (std::uint32_t move = connector_limit - 1u; move > rank; --move) {
                connector_units[move] = connector_units[move - 1u];
                connector_scores[move] = connector_scores[move - 1u];
              }
              connector_units[rank] = relation.previous;
              connector_scores[rank] = support;
              break;
            }
          }
        }
      }
    }
    for (std::uint32_t rank = 0u; rank < connector_limit && count < capacity; ++rank) {
      if (connector_units[rank] != answer_frame::kInvalidUnit) {
        append_answer_frame_candidate(
            connector_units[rank], unit_count, candidates, capacity, &count);
      }
    }
    for (std::uint32_t rank = 0u; rank < closure_limit && count < capacity; ++rank) {
      if (closure_units[rank] != answer_frame::kInvalidUnit) {
        append_answer_frame_candidate(
            closure_units[rank], unit_count, candidates, capacity, &count);
      }
    }

    const std::uint32_t seed_count = count;
    constexpr std::uint32_t local_limit = 128u;
    for (std::uint32_t table = 0u;
         table < 2u && count < local_limit && count < capacity; ++table) {
      for (std::uint32_t edge = 0u;
           edge < table_counts[table] && count < local_limit && count < capacity;
           ++edge) {
        const BigramKey relation = tables[table][edge];
        if (answer_frame_candidate_in_prefix(
                relation.previous, candidates, seed_count))
          append_answer_frame_candidate(
              relation.next, unit_count, candidates, capacity, &count);
        if (answer_frame_candidate_in_prefix(
                relation.next, candidates, seed_count))
          append_answer_frame_candidate(
              relation.previous, unit_count, candidates, capacity, &count);
      }
    }
    for (std::uint32_t i = 0u;
         i < resident_top_count && count < capacity; ++i)
      append_answer_frame_candidate(
          resident_top[i], unit_count, candidates, capacity, &count);
    const std::uint32_t direct_count = count;
    for (std::uint32_t table = 0u;
         table < 2u && count < local_limit && count < capacity; ++table) {
      for (std::uint32_t edge = 0u;
           edge < table_counts[table] && count < local_limit && count < capacity;
           ++edge) {
        const BigramKey relation = tables[table][edge];
        if (!answer_frame_candidate_in_prefix(
                relation.previous, candidates, direct_count) &&
            !answer_frame_candidate_in_prefix(
                relation.next, candidates, direct_count)) {
          continue;
        }
        append_answer_frame_candidate(
            relation.previous, unit_count, candidates, capacity, &count);
        append_answer_frame_candidate(
            relation.next, unit_count, candidates, capacity, &count);
      }
    }
    const std::uint32_t second_count = count;
    for (std::uint32_t table = 0u;
         table < 2u && count < local_limit && count < capacity; ++table) {
      for (std::uint32_t edge = 0u;
           edge < table_counts[table] && count < local_limit && count < capacity;
           ++edge) {
        const BigramKey relation = tables[table][edge];
        if (!answer_frame_candidate_in_prefix(
                relation.previous, candidates, second_count) &&
            !answer_frame_candidate_in_prefix(
                relation.next, candidates, second_count))
          continue;
        append_answer_frame_candidate(
            relation.previous, unit_count, candidates, capacity, &count);
        append_answer_frame_candidate(
            relation.next, unit_count, candidates, capacity, &count);
      }
    }
  }
  for (std::uint32_t i = 0u; i < resident_top_count && count < capacity; ++i)
    append_answer_frame_candidate(
        resident_top[i], unit_count, candidates, capacity, &count);
  *candidate_count = count;
}
