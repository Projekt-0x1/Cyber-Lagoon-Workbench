__global__ void normalize_surface_whitespace_kernel(std::uint8_t* bytes,
                                                    std::uint32_t byte_count) {
  const std::uint32_t stride = gridDim.x * blockDim.x;
  for (std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < byte_count;
       i += stride) {
    const std::uint8_t b = bytes[i];
    if (b < 0x20u || b == 0x7fu) bytes[i] = 0x20u;
  }
}

__global__ void discover_boundary_kernel(const std::uint32_t* histogram,
                                         const std::uint32_t* pairs,
                                         std::uint32_t* boundary_mask,
                                         std::uint32_t* boundary_bytes) {
  __shared__ unsigned long long scores[256];
  const std::uint32_t byte = threadIdx.x;
  std::uint32_t left_diversity = 0u;
  std::uint32_t right_diversity = 0u;
  for (std::uint32_t other = 0u; other < 256u; ++other) {
    left_diversity += pairs[other * 256u + byte] != 0u;
    right_diversity += pairs[byte * 256u + other] != 0u;
  }
  const unsigned long long diversity = left_diversity + right_diversity + 1u;
  scores[byte] = static_cast<unsigned long long>(histogram[byte]) * diversity * diversity;
  boundary_mask[byte] = 0u;
  __syncthreads();

  if (byte == 0u) {
    for (std::uint32_t selected = 0u; selected < kBoundaryCount; ++selected) {
      std::uint32_t best = 0u;
      for (std::uint32_t candidate = 1u; candidate < 256u; ++candidate) {
        if (scores[candidate] > scores[best] ||
            (scores[candidate] == scores[best] && candidate < best)) {
          best = candidate;
        }
      }
      boundary_bytes[selected] = best;
      boundary_mask[best] = 1u;
      scores[best] = 0u;
    }
  }
}
__global__ void discover_closure_kernel(const std::uint32_t* histogram,
                                        const std::uint32_t* pairs,
                                        const std::uint32_t* boundary_bytes,
                                        std::uint32_t* closure_bytes) {
  __shared__ unsigned long long scores[256];
  const std::uint32_t byte = threadIdx.x;
  std::uint32_t left_diversity = 0u;
  std::uint32_t right_diversity = 0u;
  for (std::uint32_t other = 0u; other < 256u; ++other) {
    left_diversity += pairs[other * 256u + byte] != 0u;
    right_diversity += pairs[byte * 256u + other] != 0u;
  }
  const unsigned long long left = left_diversity + 1ull;
  const unsigned long long right = right_diversity + 1ull;
  const unsigned long long support = histogram[byte] + 1ull;
  const unsigned long long terminal_mass =
      pairs[byte * 256u + boundary_bytes[0]];
  const unsigned long long interior_mass =
      histogram[byte] > terminal_mass ? histogram[byte] - terminal_mass : 0ull;
  // A lexical byte can often occur before the discovered separator too.  It
  // becomes a closure only when its boundary-facing evidence outweighs its
  // interior continuation evidence in the same learned byte field.  This is
  // intentionally a relative competition: no literal delimiter or fixed
  // punctuation threshold participates in the decision.
  const bool boundary_dominant = terminal_mass * left > interior_mass * right;
  const unsigned long long followed_boundary = terminal_mass + 1ull;
  scores[byte] = byte == boundary_bytes[0] || !boundary_dominant ? 0ull :
      left * left * left * followed_boundary * 65536ull /
          (support * right * right);
  __syncthreads();

  if (byte == 0u) {
    std::uint32_t best = 0u;
    for (std::uint32_t candidate = 1u; candidate < 256u; ++candidate) {
      if (scores[candidate] > scores[best] ||
          (scores[candidate] == scores[best] && candidate < best)) {
        best = candidate;
      }
    }
    // This ABI has two physical slots but only one closure attractor.  A
    // runner-up is not an independent boundary hypothesis: admitting it makes
    // every lexical ending selected by a weak tie split the same episode.
    // Duplicate the winner rather than inventing a second closure class.
    for (std::uint32_t selected = 0u; selected < kClosureCount; ++selected)
      closure_bytes[selected] = best;
  }
}

__global__ void discover_output_closure_kernel(const std::uint32_t* histogram,
                                               const std::uint32_t* pairs,
                                               const std::uint32_t* boundary_bytes,
                                               std::uint32_t* closure_bytes) {
  __shared__ unsigned long long scores[256];
  const std::uint32_t byte = threadIdx.x;
  std::uint32_t left_diversity = 0u;
  std::uint32_t right_diversity = 0u;
  for (std::uint32_t other = 0u; other < 256u; ++other) {
    left_diversity += pairs[other * 256u + byte] != 0u;
    right_diversity += pairs[byte * 256u + other] != 0u;
  }
  const unsigned long long left = left_diversity + 1ull;
  const unsigned long long right = right_diversity + 1ull;
  const unsigned long long support = histogram[byte] + 1ull;
  scores[byte] = byte == boundary_bytes[0] ? 0ull :
      support * left * left * left * left * left * right;
  __syncthreads();

  if (byte == 0u) {
    std::uint32_t best = 0u;
    for (std::uint32_t candidate = 1u; candidate < 256u; ++candidate) {
      if (scores[candidate] > scores[best] ||
          (scores[candidate] == scores[best] && candidate < best)) {
        best = candidate;
      }
    }
    for (std::uint32_t selected = 0u; selected < kClosureCount; ++selected)
      closure_bytes[selected] = best;
  }
}

__global__ void mark_base_boundaries_kernel(const std::uint8_t* bytes, std::uint32_t count,
                                            const std::uint32_t* boundary_mask,
                                            std::uint32_t* flags,
                                            std::uint32_t* anchors) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count) return;
  // A boundary run separates two units once.  Starting at every member of a
  // run turns repeated whitespace/control-normalized bytes into a population
  // of empty surface units, which then becomes spurious cue and relation
  // evidence.  The boundary remains losslessly attached to its preceding
  // unit; only an actual non-boundary byte may begin the next unit.
  const std::uint32_t current_is_boundary = boundary_mask[bytes[i]] != 0u;
  const std::uint32_t previous_is_boundary =
      i != 0u && boundary_mask[bytes[i - 1u]] != 0u;
  const std::uint32_t start = current_is_boundary == 0u &&
      (i == 0u || previous_is_boundary != 0u);
  flags[i] = start;
  // Keep zero as the scan's "no unit yet" sentinel. Positions are stored
  // one-based so a real unit at byte zero remains distinguishable.
  anchors[i] = start ? i + 1u : 0u;
}

__global__ void mark_bounded_units_kernel(std::uint32_t count,
                                          const std::uint32_t* anchors,
                                          std::uint32_t* flags) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count || flags[i] != 0u || anchors[i] == 0u) return;
  // The cap is only a corpus-agnostic allocation guard; separator evidence still
  // determines ordinary unit boundaries.
  flags[i] = ((i - (anchors[i] - 1u)) % kMaxUnitBytes) == 0u;
}

__global__ void scatter_unit_starts_kernel(std::uint32_t byte_count,
                                           const std::uint32_t* flags,
                                           const std::uint32_t* scanned_ids,
                                           std::uint32_t* starts) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < byte_count && flags[i] != 0u) starts[scanned_ids[i] - 1u] = i;
}

__global__ void hash_units_kernel(const std::uint8_t* bytes, std::uint32_t byte_count,
                                  const std::uint32_t* starts, std::uint32_t unit_count,
                                  UnitKey* keys, std::uint32_t* occurrence_ids) {
  const std::uint32_t unit = blockIdx.x * blockDim.x + threadIdx.x;
  if (unit >= unit_count) return;
  const std::uint32_t begin = starts[unit];
  const std::uint32_t end = unit + 1u < unit_count ? starts[unit + 1u] : byte_count;
  std::uint32_t a = 2166136261u;
  std::uint32_t b = 0x9e3779b9u;
  for (std::uint32_t i = begin; i < end; ++i) {
    a = (a ^ bytes[i]) * 16777619u;
    b = mix32(b ^ (static_cast<std::uint32_t>(bytes[i]) + (i - begin) * 0x85ebca6bu));
  }
  keys[unit] = UnitKey{a, b, end - begin};
  occurrence_ids[unit] = unit;
}

__device__ bool same_unit_bytes(const std::uint8_t* bytes, const std::uint32_t* starts,
                                std::uint32_t byte_count, std::uint32_t unit_count,
                                std::uint32_t first, std::uint32_t second) {
  const std::uint32_t first_end = first + 1u < unit_count ? starts[first + 1u] : byte_count;
  const std::uint32_t second_end = second + 1u < unit_count ? starts[second + 1u] : byte_count;
  const std::uint32_t first_length = first_end - starts[first];
  if (first_length != second_end - starts[second]) return false;
  for (std::uint32_t i = 0u; i < first_length; ++i) {
    if (bytes[starts[first] + i] != bytes[starts[second] + i]) return false;
  }
  return true;
}

__device__ __forceinline__ bool same_resident_unit_bytes(
    const std::uint32_t* unit_lengths, const std::uint32_t* unit_content,
    std::uint32_t first, std::uint32_t second) {
  if (unit_lengths[first] != unit_lengths[second]) return false;
  for (std::uint32_t word = 0u; word < kUnitWords; ++word) {
    if (unit_content[first * kUnitWords + word] !=
        unit_content[second * kUnitWords + word]) return false;
  }
  return true;
}

__device__ __forceinline__ std::uint32_t resident_unit_byte(
    const std::uint32_t* unit_content, std::uint32_t unit,
    std::uint32_t offset) {
  const std::uint32_t word =
      unit_content[unit * kUnitWords + offset / 4u];
  return (word >> ((offset % 4u) * 8u)) & 0xffu;
}

__device__ __forceinline__ bool resident_unit_ends_with(
    const std::uint32_t* unit_lengths, const std::uint32_t* unit_content,
    std::uint32_t unit, std::uint32_t byte) {
  const std::uint32_t length = unit_lengths[unit];
  return length != 0u &&
      resident_unit_byte(unit_content, unit, length - 1u) == byte;
}

// A unit's stored bytes include the discovered BOUNDARY byte (the space
// segmentation delimits on) fused onto its trailing end -- e.g. the unit for
// "its" is stored as "its " (length 4), not "its" (length 3). So a
// structural/terminal byte like '.'/'?'/'!' that ends the WORD sits one
// position back, underneath that trailing boundary byte, not at the literal
// last offset. This checks the last two positions and stops early if
// neither is the boundary byte itself, matching the reference mechanism's
// (experiments/language_port/cuda_resident_adult.cu) "skip the trailing
// boundary byte, then test underneath" pattern for the same reason.
__device__ __forceinline__ bool resident_unit_terminal_byte(
    const std::uint32_t* unit_lengths, const std::uint32_t* unit_content,
    std::uint32_t unit, std::uint32_t byte, std::uint32_t boundary_byte) {
  const std::uint32_t length = unit_lengths[unit];
  for (std::uint32_t back = 0u; back < 2u && back < length; ++back) {
    const std::uint32_t offset = length - 1u - back;
    const std::uint32_t value = resident_unit_byte(unit_content, unit, offset);
    if (value == byte) return true;
    if (value != boundary_byte) break;
  }
  return false;
}

__device__ __forceinline__ bool resident_unit_contains_any(
    const std::uint32_t* unit_lengths, const std::uint32_t* unit_content,
    std::uint32_t unit, const std::uint32_t* bytes,
    std::uint32_t byte_count, std::uint32_t* matched_offset = nullptr) {
  for (std::uint32_t offset = 0u; offset < unit_lengths[unit]; ++offset) {
    const std::uint32_t value = resident_unit_byte(unit_content, unit, offset);
    for (std::uint32_t index = 0u; index < byte_count; ++index) {
      if (value == bytes[index]) {
        if (matched_offset != nullptr) *matched_offset = offset;
        return true;
      }
    }
  }
  return false;
}

__device__ __forceinline__ unsigned long long resident_closure_score(
    const std::uint32_t* histogram, const std::uint32_t* pairs,
    std::uint32_t boundary, std::uint32_t closure) {
  std::uint32_t left_diversity = 0u;
  std::uint32_t right_diversity = 0u;
  for (std::uint32_t other = 0u; other < 256u; ++other) {
    left_diversity += pairs[other * 256u + closure] != 0u;
    right_diversity += pairs[closure * 256u + other] != 0u;
  }
  const unsigned long long left = left_diversity + 1ull;
  const unsigned long long right = right_diversity + 1ull;
  return left * left * left *
      (pairs[closure * 256u + boundary] + 1ull) * 65536ull /
      ((histogram[closure] + 1ull) * right * right);
}

__device__ __forceinline__ std::uint32_t resident_episode_start_variant(
    const std::uint32_t* unit_lengths, const std::uint32_t* unit_content,
    std::uint32_t unit_count, const std::uint32_t* episode_units,
    std::uint32_t episode_count, const std::uint32_t* episode_breaks,
    std::uint32_t episode_break_count, std::uint32_t original) {
  const std::uint32_t length = unit_lengths[original];
  if (length == 0u) return original;
  std::uint32_t best = original;
  std::uint32_t best_starts = 0u;
  for (std::uint32_t candidate_episode = 0u;
       candidate_episode < episode_break_count; ++candidate_episode) {
    const std::uint32_t candidate_position = candidate_episode == 0u
        ? 0u : episode_breaks[candidate_episode - 1u];
    if (candidate_position >= episode_count) continue;
    const std::uint32_t candidate = episode_units[candidate_position];
    if (candidate >= unit_count) continue;
    if (unit_lengths[candidate] != length ||
        resident_unit_byte(unit_content, candidate, 0u) ==
            resident_unit_byte(unit_content, original, 0u)) continue;
    bool same_tail = true;
    for (std::uint32_t offset = 1u; offset < length; ++offset) {
      if (resident_unit_byte(unit_content, candidate, offset) !=
          resident_unit_byte(unit_content, original, offset)) {
        same_tail = false;
        break;
      }
    }
    if (!same_tail) continue;
    std::uint32_t starts = 0u;
    for (std::uint32_t episode = 0u; episode < episode_break_count; ++episode) {
      const std::uint32_t position = episode == 0u ? 0u : episode_breaks[episode - 1u];
      starts += position < episode_count && episode_units[position] == candidate;
    }
    if (starts > best_starts) {
      best = candidate;
      best_starts = starts;
    }
  }
  return best;
}

__device__ __forceinline__ void resident_clause_bounds(
    const std::uint32_t* unit_lengths, const std::uint32_t* unit_content,
    const std::uint32_t* episode_units, std::uint32_t episode_count,
    const std::uint32_t* episode_breaks, std::uint32_t episode_break_count,
    std::uint32_t position, const std::uint32_t* closure_bytes,
    std::uint32_t closure_count,
    std::uint32_t* clause_begin, std::uint32_t* clause_end) {
  std::uint32_t lo = 0u;
  std::uint32_t hi = episode_break_count;
  while (lo < hi) {
    const std::uint32_t middle = lo + (hi - lo) / 2u;
    if (episode_breaks[middle] <= position) lo = middle + 1u;
    else hi = middle;
  }
  const std::uint32_t episode_begin = lo == 0u ? 0u : episode_breaks[lo - 1u];
  const std::uint32_t episode_end =
      lo < episode_break_count ? episode_breaks[lo] : episode_count;
  *clause_begin = episode_begin;
  for (std::uint32_t at = position; at > episode_begin; --at) {
    if (resident_unit_contains_any(unit_lengths, unit_content,
                                  episode_units[at - 1u], closure_bytes,
                                  closure_count)) {
      *clause_begin = at;
      break;
    }
  }
  *clause_end = episode_end;
  for (std::uint32_t at = position; at < episode_end; ++at) {
    if (resident_unit_contains_any(unit_lengths, unit_content,
                                  episode_units[at], closure_bytes,
                                  closure_count)) {
      *clause_end = at + 1u;
      break;
    }
  }
}
