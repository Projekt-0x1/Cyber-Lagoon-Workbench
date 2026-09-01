// Adult cue matching, association mass, and informative-anchor kernels.
//
// Included from bcc32_cuda_adult_v1.cuh inside its namespace and state-only
// guard as one dependency-ordered cue-admission responsibility.
__global__ void match_cue_segment_sequences_kernel(
    const std::uint8_t* bytes, const std::uint32_t* starts, std::uint32_t byte_count,
    std::uint32_t cue_count, const std::uint32_t* boundary_mask,
    const std::uint32_t* unit_lengths,
    const std::uint32_t* unit_content, const std::uint32_t* episode_units,
    std::uint32_t episode_count, const std::uint32_t* episode_breaks,
    std::uint32_t episode_break_count, std::uint32_t* episode_match_mask,
    std::uint32_t* episode_exact_match_mask, std::uint32_t* episode_match_spans,
    std::uint32_t* matched_segment_counts,
    std::uint32_t* admitted_segment_mask, std::uint32_t novelty_scope) {
  const std::uint32_t cue = blockIdx.y;
  const std::uint32_t position = blockIdx.x * blockDim.x + threadIdx.x;
  if (cue >= cue_count || position >= episode_count) return;
  const std::uint32_t begin = starts[cue];
  const std::uint32_t end = cue + 1u < cue_count ? starts[cue + 1u] : byte_count;
  const std::uint32_t length = end - begin;
  if (length == 0u) return;

  std::uint32_t episode_end = episode_count;
  if (episode_break_count != 0u) {
    std::uint32_t lo = 0u, hi = episode_break_count;
    while (lo < hi) {
      const std::uint32_t middle = lo + (hi - lo) / 2u;
      if (episode_breaks[middle] <= position) lo = middle + 1u; else hi = middle;
    }
    if (lo >= episode_break_count) return;
    episode_end = episode_breaks[lo];
  }
  std::uint32_t best_span = 0u;
  std::uint32_t best_rank = 0xffffffffu;
  const std::uint32_t first = episode_units[position];
  for (std::uint32_t start_offset = 0u; start_offset < unit_lengths[first]; ++start_offset) {
    if (novelty_scope != 0u && start_offset != 0u) {
      const std::uint32_t previous_packed =
          unit_content[first * kUnitWords + (start_offset - 1u) / 4u];
      const std::uint8_t previous = static_cast<std::uint8_t>(
          previous_packed >> (((start_offset - 1u) % 4u) * 8u));
      if (boundary_mask[previous] == 0u) continue;
    }
    std::uint32_t consumed = 0u;
    bool terminal_canonicalized = false;
    bool open = true;
    for (std::uint32_t count = 1u; count <= kCueAlignmentUnits &&
         position + count <= episode_end && open; ++count) {
      const std::uint32_t unit = episode_units[position + count - 1u];
      const std::uint32_t unit_begin = count == 1u ? start_offset : 0u;
      std::uint32_t end_offset = unit_begin;
      for (std::uint32_t offset = unit_begin;
           offset < unit_lengths[unit] && consumed < length; ++offset) {
        const std::uint32_t packed = unit_content[unit * kUnitWords + offset / 4u];
        const std::uint8_t resident = static_cast<std::uint8_t>(
            packed >> ((offset % 4u) * 8u));
        if (bytes[begin + consumed] != resident) {
          const bool terminal = consumed + 1u == length;
          const bool bounded_terminal_tail = novelty_scope != 0u && consumed >= 3u &&
              length - consumed <= 2u && unit_lengths[unit] - offset <= 2u;
          if (bounded_terminal_tail && !terminal_canonicalized) {
            terminal_canonicalized = true;
            consumed = length;
            end_offset = unit_lengths[unit];
            break;
          }
          bool suffix_normalized = false;
          if (novelty_scope != 0u && terminal && !terminal_canonicalized &&
              boundary_mask[bytes[begin + consumed]] != 0u &&
              boundary_mask[resident] == 0u) {
            for (std::uint32_t suffix = offset + 1u;
                 suffix < unit_lengths[unit]; ++suffix) {
              const std::uint32_t suffix_packed =
                  unit_content[unit * kUnitWords + suffix / 4u];
              const std::uint8_t suffix_byte = static_cast<std::uint8_t>(
                  suffix_packed >> ((suffix % 4u) * 8u));
              if (suffix_byte == bytes[begin + consumed] ||
                  boundary_mask[suffix_byte] != 0u) {
                terminal_canonicalized = true;
                suffix_normalized = true;
                consumed = length;
                end_offset = suffix + 1u;
                break;
              }
            }
          }
          if (suffix_normalized) break;
          if (terminal && !terminal_canonicalized) terminal_canonicalized = true;
          else {
            open = false;
            break;
          }
        }
        ++consumed;
        end_offset = offset + 1u;
      }
      if (consumed != length) continue;
      const std::uint32_t rank =
          (terminal_canonicalized ? 0x80000000u : 0u) |
          (count << 8u) | start_offset;
      if (rank < best_rank) {
        best_rank = rank;
        best_span = 0x80000000u |
            (terminal_canonicalized ? 0x40000000u : 0u) |
            ((start_offset & 0x3fu) << 24u) |
            ((count & 0x1fu) << 19u) |
            ((end_offset & 0x3fu) << 13u);
      }
      break;
    }
  }
  if (best_span != 0u) {
    episode_match_spans[cue * episode_count + position] = best_span;
    atomicOr(episode_match_mask + position, 1u << cue);
    atomicOr(admitted_segment_mask, 1u << cue);
    if ((best_span & 0x40000000u) == 0u) {
      atomicOr(episode_exact_match_mask + position, 1u << cue);
      atomicAdd(matched_segment_counts + cue, 1u);
    }
  }
}

__global__ void match_cue_units_kernel(
    const std::uint8_t* bytes, std::uint32_t byte_count, const std::uint32_t* boundary_mask,
    const std::uint32_t* unit_lengths, const std::uint32_t* unit_content,
    std::uint32_t unit_count, std::uint32_t* cue_presence, std::uint32_t* fallback) {
  if (blockIdx.x != 0u || threadIdx.x != 0u) return;
  std::uint32_t begin = 0u;
  fallback[0] = 0u;
  while (begin < byte_count) {
    std::uint32_t end = begin + 1u;
    while (end < byte_count && end - begin < kMaxUnitBytes &&
           boundary_mask[bytes[end - 1u]] == 0u) ++end;
    std::uint32_t best = 0u;
    std::uint32_t best_score = 0u;
    for (std::uint32_t unit = 0u; unit < unit_count; ++unit) {
      const std::uint32_t score = fuzzy_unit_score(bytes, begin, end - begin, unit,
                                                   unit_lengths, unit_content);
      if (score > best_score) { best_score = score; best = unit; }
    }
    if (best_score != 0u) {
      cue_presence[best] = 1u;
      fallback[0] = best;
    }
    begin = end;
  }
}

__global__ void score_online_associations_kernel(
    const AssociationKey* associations, const std::uint32_t* association_counts,
    std::uint32_t association_count, const std::uint32_t* cue_presence,
    const std::uint32_t* vitality, std::uint32_t* candidate_scores) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= association_count || association_counts[i] == 0u) return;
  const AssociationKey pair = associations[i];
  if (cue_presence[pair.first] != 0u) {
    const std::uint32_t weight = 4096u / max(1u, vitality[pair.first]);
    atomicAdd(candidate_scores + pair.second,
              association_counts[i] * weight * cue_presence[pair.first]);
  }
  if (cue_presence[pair.second] != 0u) {
    const std::uint32_t weight = 4096u / max(1u, vitality[pair.second]);
    atomicAdd(candidate_scores + pair.first,
              (association_counts[i] * weight * cue_presence[pair.second]) / 2u);
  }
}

__device__ std::uint32_t association_mass_depth(unsigned long long count) {
  std::uint32_t depth = 0u;
  while (count != 0u) {
    ++depth;
    count >>= 1u;
  }
  return depth;
}

__global__ void accumulate_association_mass_kernel(
    const AssociationKey* associations, const std::uint32_t* association_counts,
    std::uint32_t association_count, unsigned long long* association_mass) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= association_count || association_counts[i] == 0u) return;
  const AssociationKey relation = associations[i];
  const unsigned long long mass = association_counts[i];
  atomicAdd(association_mass + relation.first, mass);
  atomicAdd(association_mass + relation.second, mass);
}

__device__ std::uint32_t resident_bigram_count(
    const BigramKey* keys, const std::uint32_t* counts, std::uint32_t count,
    std::uint32_t previous, std::uint32_t next);

__global__ void select_informative_cue_anchors_kernel(
    const std::uint32_t* best_ids, const std::uint32_t* best_scores,
    const std::uint32_t* matched_segment_counts,
    std::uint32_t cue_count, std::uint32_t unit_count,
    std::uint32_t association_count, const std::uint32_t* unit_lengths,
    const std::uint32_t* unit_content, const std::uint32_t* vitality,
    const unsigned long long* association_mass, const BigramKey* base_bigrams,
    const std::uint32_t* base_bigram_counts, std::uint32_t base_bigram_count,
    const BigramKey* online_bigrams, const std::uint32_t* online_bigram_counts,
    std::uint32_t online_bigram_count,
    const ConditionedTransitionKey* conditioned_transitions,
    std::uint32_t conditioned_transition_count, std::uint32_t* cue_scores,
    std::uint32_t* cue_orders, std::uint32_t* cue_weights,
    std::uint32_t* cue_masks, std::uint32_t* salient_masks,
    std::uint32_t* admitted_segment_mask, std::uint32_t* selected_anchor_ids,
    std::uint32_t diagnostic) {
  if (blockIdx.x != 0u || threadIdx.x != 0u) return;
  std::uint32_t selected_ids[kCueAnchorLimit];
  std::uint32_t selected_count = 0u;
  for (std::uint32_t rank = 0u; rank < kCueAnchorLimit; ++rank) {
    selected_anchor_ids[rank] = 0xffffffffu;
  }
  admitted_segment_mask[0] = 0u;
  for (std::uint32_t cue = 0u; cue < min(cue_count, 32u); ++cue) {
    if (matched_segment_counts[cue] != 0u) {
      admitted_segment_mask[0] |= 1u << cue;
      continue;
    }
    const std::uint32_t unit = best_ids[cue];
    if (unit >= unit_count) continue;
    const std::uint32_t tolerance = min(
        65535u - kCueStrongIdentity, max(512u, unit_lengths[unit] * 512u));
    if (best_scores[cue] >= 65535u - tolerance) admitted_segment_mask[0] |= 1u << cue;
  }
  for (std::uint32_t rank = 0u; rank < kCueAnchorLimit; ++rank) {
    std::uint32_t best_candidate = 0xffffffffu;
    unsigned long long best_rank = 0u;
    bool best_has_conditioned = false;
    for (std::uint32_t candidate = 0u; candidate < cue_count; ++candidate) {
      const std::uint32_t cue = candidate;
      const std::uint32_t unit = best_ids[cue];
      const std::uint32_t match_score = best_scores[cue];
      if (unit >= unit_count) continue;
      const std::uint32_t conditioned_begin =
          bcc32_cuda_resident_synthesis::lower_subject_transition(
              conditioned_transitions, conditioned_transition_count,
              unit, 0u);
      const std::uint32_t conditioned_end =
          bcc32_cuda_resident_synthesis::upper_subject_transition(
              conditioned_transitions, conditioned_transition_count,
              unit, 0xffffffffu);
      const bool has_conditioned = conditioned_end > conditioned_begin;
      if (matched_segment_counts[cue] != 0u && !has_conditioned) continue;
      const std::uint32_t tolerance = min(
          65535u - kCueStrongIdentity, max(512u, unit_lengths[unit] * 512u));
      if (match_score < 65535u - tolerance ||
          (!has_conditioned &&
           (vitality[unit] == 0u || association_mass[unit] < 8u))) {
        continue;
      }
      bool duplicate = false;
      for (std::uint32_t prior = 0u; prior < selected_count; ++prior) {
        duplicate |= selected_ids[prior] == unit;
      }
      if (duplicate) continue;
      const unsigned long long denominator =
          static_cast<unsigned long long>(max(1u, vitality[unit])) *
          (has_conditioned
               ? max(1ull, min(static_cast<unsigned long long>(
                                   conditioned_end - conditioned_begin),
                               65535ull))
               : max(8ull, min(association_mass[unit], 65535ull)));
      const unsigned long long informative =
          static_cast<unsigned long long>(match_score) * 65536ull / denominator;
      if ((has_conditioned && !best_has_conditioned) ||
          (has_conditioned == best_has_conditioned &&
           (informative > best_rank ||
            (informative == best_rank && informative != 0u &&
             candidate < best_candidate)))) {
        best_has_conditioned = has_conditioned;
        best_rank = informative;
        best_candidate = candidate;
      }
    }
    if (best_candidate == 0xffffffffu || best_rank == 0u) break;
    const std::uint32_t cue = best_candidate;
    const std::uint32_t unit = best_ids[cue];
    const std::uint32_t match_score = best_scores[cue];
    selected_ids[selected_count++] = unit;
    selected_anchor_ids[rank] = unit;
    cue_scores[unit] = match_score;
    cue_orders[unit] = cue * kCueAlignmentUnits + 1u;
    cue_weights[unit] = static_cast<std::uint32_t>(min(best_rank, 0xffffffffull));
    cue_masks[unit] = 1u << rank;
    const unsigned long long resident_scale = max(
        8ull, static_cast<unsigned long long>(association_count) * kAssociationRadius /
                  max(1u, unit_count));
    const std::uint32_t conditioned_begin =
        bcc32_cuda_resident_synthesis::lower_subject_transition(
            conditioned_transitions, conditioned_transition_count, unit, 0u);
    const std::uint32_t conditioned_end =
        bcc32_cuda_resident_synthesis::upper_subject_transition(
            conditioned_transitions, conditioned_transition_count,
            unit, 0xffffffffu);
    const bool has_conditioned = conditioned_end > conditioned_begin;
    const bool salient = has_conditioned ||
        best_rank >= 16ull * resident_scale;
    if (salient) salient_masks[unit] = 1u << rank;
    if (diagnostic != 0u) {
      printf("cue_anchor slot=%u id=%u bytes=\"", rank, unit);
      for (std::uint32_t offset = 0u; offset < unit_lengths[unit]; ++offset) {
        const std::uint32_t packed = unit_content[unit * kUnitWords + offset / 4u];
        printf("%c", static_cast<int>((packed >> ((offset % 4u) * 8u)) & 0xffu));
      }
      printf("\" length=%u match=%u vitality=%u mass=%llu weight=%u salient=%u conditioned=%u\n",
             unit_lengths[unit], match_score, vitality[unit],
             association_mass[unit], cue_weights[unit], salient ? 1u : 0u,
             has_conditioned ? 1u : 0u);
    }
  }
}
