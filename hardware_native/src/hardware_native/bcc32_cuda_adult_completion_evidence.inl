// Adult resident completion evidence and provenance-seed kernels.
//
// Included after base-completion policy and before resident relation
// composition. This unit owns resident association lookup, cue-derived
// completion evidence, and provenance seed materialization; it adds no state.

__device__ std::uint32_t resident_association_count(
    const AssociationKey* keys, const std::uint32_t* counts, std::uint32_t count,
    std::uint32_t first, std::uint32_t second) {
  const AssociationKey wanted{first, second};
  std::uint32_t lo = 0u, hi = count;
  while (lo < hi) {
    const std::uint32_t mid = lo + (hi - lo) / 2u;
    if (keys[mid] < wanted) lo = mid + 1u; else hi = mid;
  }
  return lo < count && keys[lo] == wanted ? counts[lo] : 0u;
}
__device__ std::uint32_t evidence_depth(std::uint32_t count) {
  std::uint32_t depth = 0u;
  while (count != 0u) {
    ++depth;
    count >>= 1u;
  }
  return depth;
}

__device__ bool resident_unit_contains_byte(
    std::uint32_t unit, std::uint8_t byte, const std::uint32_t* unit_lengths,
    const std::uint32_t* unit_content) {
  for (std::uint32_t i = 0u; i < unit_lengths[unit]; ++i) {
    const std::uint32_t packed = unit_content[unit * kUnitWords + i / 4u];
    if (((packed >> ((i % 4u) * 8u)) & 0xffu) == byte) return true;
  }
  return false;
}

__global__ void assess_composition_cue_kernel(
    const std::uint32_t* best_ids, const std::uint32_t* best_scores,
    const std::uint32_t* starts, std::uint32_t byte_count, std::uint32_t cue_count,
    const std::uint32_t* route_scores, const BigramKey* base_bigrams,
    const std::uint32_t* base_bigram_counts, std::uint32_t base_bigram_count,
    const BigramKey* online_bigrams, const std::uint32_t* online_bigram_counts,
    std::uint32_t online_bigram_count, const unsigned long long* cue_evidence,
    std::uint32_t* motor_context) {
  if (blockIdx.x != 0u || threadIdx.x != 0u) return;
  std::uint32_t matched = 0u;
  std::uint32_t distinct = 0u;
  std::uint32_t adjacent = 0u;
  std::uint32_t exact_adjacent = 0u;
  for (std::uint32_t cue = 0u; cue < cue_count; ++cue) {
    const std::uint32_t id = best_ids[cue];
    const bool strong = best_scores[cue] >= kCueNearIdentity && route_scores[id] != 0u;
    if (!strong) continue;
    ++matched;
    bool first = true;
    for (std::uint32_t prior = 0u; prior < cue; ++prior) {
      if (best_ids[prior] == id && route_scores[id] != 0u) {
        first = false;
        break;
      }
    }
    distinct += first;
    if (cue + 1u >= cue_count) continue;
    const std::uint32_t next_id = best_ids[cue + 1u];
    const bool next_strong = best_scores[cue + 1u] >= kCueNearIdentity &&
        route_scores[next_id] != 0u;
    if (!next_strong) continue;
    ++adjacent;
    const std::uint32_t slow = resident_bigram_count(
        base_bigrams, base_bigram_counts, base_bigram_count, id, next_id);
    const std::uint32_t fast = resident_bigram_count(
        online_bigrams, online_bigram_counts, online_bigram_count, id, next_id);
    exact_adjacent += slow != 0u || fast != 0u;
  }

  const unsigned long long forward = cue_evidence[0];
  const unsigned long long backward = cue_evidence[1];
  motor_context[6] = static_cast<std::uint32_t>(min(forward, 0xffffffffull));
  motor_context[7] = static_cast<std::uint32_t>(min(backward, 0xffffffffull));
  motor_context[8] = matched;
  motor_context[9] = distinct;
  motor_context[10] = adjacent;
  motor_context[11] = exact_adjacent;
  const bool directed = cue_evidence[2] != 0u && forward > backward;
  const bool local_order = exact_adjacent != 0u;
  motor_context[15] = distinct >= 2u && (directed || local_order);
}

__global__ void score_local_provenance_seeds_kernel(
    const std::uint32_t* vitality, const std::uint32_t* unit_lengths,
    const std::uint32_t* unit_content, const std::uint32_t* closure_bytes,
    std::uint32_t cue_count, const std::uint32_t* episode_match_mask,
    const std::uint32_t* episode_exact_match_mask,
    const std::uint32_t* episode_match_spans,
    const BigramKey* online_bigrams, const std::uint32_t* online_bigram_counts,
    std::uint32_t online_bigram_count, const TrigramKey* online_trigrams,
    const std::uint32_t* online_trigram_counts, std::uint32_t online_trigram_count,
    const std::uint32_t* episode_units, std::uint32_t episode_count,
    const std::uint32_t* episode_breaks, std::uint32_t episode_break_count,
    const std::uint32_t* matched_segment_counts,
    LocalSeedCandidate* candidates, std::uint32_t* motor_context,
    std::uint32_t novelty_scope, std::uint32_t bounded_clause_scope) {
  const std::uint32_t position = blockIdx.x * blockDim.x + threadIdx.x;
  if (position >= episode_count) return;
  candidates[position] = LocalSeedCandidate{
      0u, position, 0xffffffffu, 0u, 0u, 0u, position, 0u, 0u, 0u, position, 0u,
      position, position, 0u, 0u};
  if (episode_match_mask[position] == 0u) return;

  std::uint32_t episode_begin = 0u;
  std::uint32_t episode_end = episode_count;
  if (episode_break_count != 0u) {
    std::uint32_t lo = 0u, hi = episode_break_count;
    while (lo < hi) {
      const std::uint32_t middle = lo + (hi - lo) / 2u;
      if (episode_breaks[middle] <= position) lo = middle + 1u; else hi = middle;
    }
    if (lo >= episode_break_count) return;
    episode_begin = lo == 0u ? 0u : episode_breaks[lo - 1u];
    episode_end = episode_breaks[lo];
  }
  if (bounded_clause_scope != 0u &&
      episode_end - episode_begin > 4u * kCompositionUnits) {
    resident_clause_bounds(
        unit_lengths, unit_content, episode_units, episode_count,
        episode_breaks, episode_break_count, position, closure_bytes,
        kClosureCount, &episode_begin, &episode_end);
  }
  if (position + kCompositionSemanticLaunchUnits >= episode_end) {
    std::uint32_t local_mask = 0u;
    std::uint32_t local_exact_mask = 0u;
    const std::uint32_t local_begin = episode_end - episode_begin <= kCompositionUnits
        ? episode_begin
        : (position > kAssociationRadius
            ? max(episode_begin, position - kAssociationRadius) : episode_begin);
    for (std::uint32_t at = local_begin; at < episode_end; ++at) {
      local_mask |= episode_match_mask[at];
      local_exact_mask |= episode_exact_match_mask[at];
    }
    if (local_exact_mask == 0u) return;
    const std::uint32_t distinctive_limit = max(2u, episode_count / 8192u);
    std::uint32_t distinctive_total = 0u;
    std::uint32_t distinctive_coverage = 0u;
    std::uint32_t distinctive_cue_mask = 0u;
    for (std::uint32_t cue = 0u; cue < min(cue_count, 32u); ++cue) {
      const std::uint32_t occurrences = matched_segment_counts[cue];
      if (occurrences == 0u || occurrences > distinctive_limit) continue;
      ++distinctive_total;
      if ((local_exact_mask & (1u << cue)) != 0u) {
        ++distinctive_coverage;
        distinctive_cue_mask |= 1u << cue;
      }
    }
    const unsigned long long score =
        (1ull << 60u) |
        (static_cast<unsigned long long>(min(__popc(local_mask), 63)) << 54u) |
        1ull;
    candidates[position] = LocalSeedCandidate{
        score, position, episode_units[position], local_mask, local_exact_mask,
        distinctive_cue_mask,
        position, static_cast<std::uint32_t>(__ffs(local_exact_mask) - 1),
        1u, 0u, min(episode_end, position + 1u), 1u,
        episode_begin, episode_end, distinctive_coverage, distinctive_total};
    return;
  }
  unsigned long long best_score = 0u;
  std::uint32_t best_seed = 0xffffffffu;
  std::uint32_t best_target = 0xffffffffu;
  std::uint32_t best_mask = 0u;
  std::uint32_t best_exact_mask = 0u;
  std::uint32_t best_cue = 0u;
  std::uint32_t best_run = 0u;
  std::uint32_t best_start_offset = 0u;
  std::uint32_t best_launch_end = position;
  std::uint32_t best_backward_launch = 0u;
  for (std::uint32_t first_cue = 0u;
       first_cue < min(cue_count, 32u); ++first_cue) {
    if ((episode_match_mask[position] & (1u << first_cue)) == 0u) continue;
    std::uint32_t current_position = position;
    std::uint32_t run_end_position = position;
    std::uint32_t run_mask = 0u;
    std::uint32_t run = 0u;
    std::uint32_t run_end_offset = 0u;
    std::uint32_t strongest_rarity = 0u;
    std::uint32_t second_rarity = 0u;
    std::uint32_t target = 0xffffffffu;

    for (std::uint32_t cue = first_cue; cue < min(cue_count, 32u); ++cue) {
      if (current_position >= episode_end ||
          (episode_match_mask[current_position] & (1u << cue)) == 0u) break;
      const std::uint32_t span =
          episode_match_spans[cue * episode_count + current_position];
      if ((span & 0x80000000u) == 0u) break;
      const std::uint32_t span_units = (span >> 19u) & 0x1fu;
      const std::uint32_t end_offset = (span >> 13u) & 0x3fu;
      if (span_units == 0u || current_position + span_units > episode_end) break;

      const std::uint32_t anchor = episode_units[current_position];
      const std::uint32_t rarity = static_cast<std::uint32_t>(min(
          static_cast<unsigned long long>(unit_lengths[anchor]) * 65535ull /
              max(1u, vitality[anchor]), 65535ull));
      if (rarity > strongest_rarity) {
        second_rarity = strongest_rarity;
        strongest_rarity = rarity;
        target = anchor;
      } else if (rarity > second_rarity) {
        second_rarity = rarity;
      }
      run_mask |= 1u << cue;
      ++run;
      run_end_position = current_position + span_units - 1u;
      run_end_offset = end_offset;
      if (cue + 1u >= min(cue_count, 32u)) break;

      const std::uint32_t end_unit = episode_units[run_end_position];
      std::uint32_t next_position = run_end_position;
      std::uint32_t required_start = end_offset;
      if (end_offset >= unit_lengths[end_unit]) {
        next_position = run_end_position + 1u;
        required_start = 0u;
      }
      bool found_next = false;
      const std::uint32_t search_end = novelty_scope != 0u
          ? min(episode_end, next_position + kAssociationRadius + 1u)
          : min(episode_end, next_position + 1u);
      for (std::uint32_t at = next_position; at < search_end; ++at) {
        if ((episode_match_mask[at] & (1u << (cue + 1u))) == 0u) continue;
        const std::uint32_t next_span =
            episode_match_spans[(cue + 1u) * episode_count + at];
        const std::uint32_t wanted_start = at == next_position ? required_start : 0u;
        if (((next_span >> 24u) & 0x3fu) != wanted_start) continue;
        current_position = at;
        found_next = true;
        break;
      }
      if (!found_next) break;
    }
    if (run < 2u || (novelty_scope != 0u
            ? episode_begin + 2u >= episode_end
            : run_end_position + 2u >= episode_end)) continue;

  std::uint32_t coverage_mask = run_mask;
  std::uint32_t exact_coverage_mask = 0u;
    const std::uint32_t coverage_begin = position > kAssociationRadius
        ? max(episode_begin, position - kAssociationRadius) : episode_begin;
  const std::uint32_t coverage_end = min(
      episode_end, run_end_position + kAssociationRadius + 1u);
  for (std::uint32_t at = coverage_begin; at < coverage_end; ++at) {
    coverage_mask |= episode_match_mask[at];
  }
  const std::uint32_t exact_coverage_begin =
      episode_end - episode_begin <= kCompositionUnits ? episode_begin : coverage_begin;
  const std::uint32_t exact_coverage_end =
      episode_end - episode_begin <= kCompositionUnits ? episode_end : coverage_end;
  for (std::uint32_t at = exact_coverage_begin; at < exact_coverage_end; ++at) {
    exact_coverage_mask |= episode_exact_match_mask[at];
  }

    std::uint32_t forward_usable = 0u;
    for (std::uint32_t at = run_end_position + 1u;
         at < min(episode_end, run_end_position + kCompositionMinUnits + 1u); ++at) {
      const std::uint32_t unit = episode_units[at];
      const unsigned long long information =
          static_cast<unsigned long long>(unit_lengths[unit]) * 65536ull /
          max(1u, vitality[unit]);
      forward_usable += information >= 32u;
    }
    const bool forward_meaningful =
        forward_usable * 4u >= kCompositionMinUnits * 3u;
    const std::uint32_t seed = forward_meaningful
        ? run_end_position
        : max(episode_begin, position > 2u * kAssociationRadius
              ? position - 2u * kAssociationRadius : episode_begin);
    const std::uint32_t launch_end = forward_meaningful
        ? min(episode_end, seed + kCompositionSemanticLaunchUnits)
        : min(episode_end, run_end_position + 1u);
    const std::uint32_t first = episode_units[seed];
    const std::uint32_t second = episode_units[seed + 1u];
    const std::uint32_t third = episode_units[seed + 2u];
    const std::uint32_t bigram = resident_bigram_count(
        online_bigrams, online_bigram_counts, online_bigram_count, first, second);
    const std::uint32_t trigram = resident_trigram_count(
        online_trigrams, online_trigram_counts, online_trigram_count,
        first, second, third);
    if (bigram == 0u || trigram == 0u) continue;
    const std::uint32_t robust_rarity = run > 1u ? second_rarity : strongest_rarity;
    const std::uint32_t continuation = min(
        65535u, 16u * evidence_depth(bigram) + 24u * evidence_depth(trigram));
    const unsigned long long score =
        (static_cast<unsigned long long>(min(run, 15u)) << 60u) |
        (static_cast<unsigned long long>(min(__popc(coverage_mask), 63)) << 54u) |
        (static_cast<unsigned long long>(robust_rarity) << 38u) |
        continuation;
    if (score > best_score) {
      best_score = score;
      best_seed = seed;
      best_target = target;
      best_mask = coverage_mask;
      best_exact_mask = exact_coverage_mask;
      best_cue = first_cue;
      best_run = run;
      best_start_offset = run_end_offset;
      best_launch_end = launch_end;
      best_backward_launch = forward_meaningful ? 0u : 1u;
    }
  }
  if (best_score == 0u) return;
  const std::uint32_t distinctive_limit = max(2u, episode_count / 8192u);
  std::uint32_t distinctive_total = 0u;
  std::uint32_t distinctive_coverage = 0u;
  std::uint32_t distinctive_cue_mask = 0u;
  for (std::uint32_t cue = 0u; cue < min(cue_count, 32u); ++cue) {
    const std::uint32_t occurrences = matched_segment_counts[cue];
    if (occurrences == 0u || occurrences > distinctive_limit) continue;
    ++distinctive_total;
    if ((best_exact_mask & (1u << cue)) != 0u) {
      ++distinctive_coverage;
      distinctive_cue_mask |= 1u << cue;
    }
  }
  candidates[position] = LocalSeedCandidate{
      best_score, best_seed, best_target, best_mask, best_exact_mask,
      distinctive_cue_mask,
      position, best_cue, best_run,
      best_backward_launch != 0u ? 0u : best_start_offset,
      best_launch_end, best_backward_launch, episode_begin, episode_end,
      distinctive_coverage, distinctive_total};
  atomicAdd(motor_context + 14u, 1u);
}

__global__ void print_local_provenance_seed_kernel(
    const LocalSeedCandidate* candidates, const std::uint32_t* episode_match_mask,
    const std::uint32_t* episode_match_spans, const std::uint32_t* episode_units,
    const std::uint32_t* unit_lengths, const std::uint32_t* unit_content,
    const std::uint32_t* vitality,
    std::uint32_t episode_count, std::uint32_t cue_count, std::uint32_t episode_begin) {
  if (blockIdx.x != 0u || threadIdx.x != 0u) return;
  const LocalSeedCandidate candidate = candidates[0];
  const bool found = candidate.score != 0u;
  if (found) {
    printf("cue_occurrence anchor=%u seed=%u offset=%u launch_end=%u direction=%s run=%u first_cue=%u mask=%08x exact=%08x distinctive=%08x coverage=%u/%u scope=%u score=%llu\n",
           candidate.anchor_position, candidate.position, candidate.start_offset,
           candidate.local_launch_end,
           candidate.backward_launch != 0u ? "backward" : "forward",
           candidate.contiguous_run,
           candidate.anchor_cue, candidate.cue_mask, candidate.exact_cue_mask,
           candidate.distinctive_cue_mask, candidate.distinctive_coverage,
           candidate.distinctive_total,
           candidate.episode_end - candidate.episode_begin, candidate.score);
  } else {
    printf("cue_occurrence none scope_begin=%u scope_units=%u\n",
           episode_begin, episode_count);
  }
  const std::uint32_t local_anchor = found
      ? candidate.anchor_position - episode_begin : 0u;
  const std::uint32_t end = found
      ? min(episode_count, local_anchor + kCueAlignmentUnits)
      : min(episode_count, 128u);
  for (std::uint32_t position = local_anchor; position < end; ++position) {
    const std::uint32_t mask = episode_match_mask[position];
    const std::uint32_t unit = episode_units[episode_begin + position];
    printf("cue_occurrence_unit position=%u id=%u vitality=%u mask=%08x bytes=\"",
           episode_begin + position, unit, vitality[unit], mask);
    for (std::uint32_t offset = 0u; offset < unit_lengths[unit]; ++offset) {
      const std::uint32_t packed = unit_content[unit * kUnitWords + offset / 4u];
      printf("%c", static_cast<int>((packed >> ((offset % 4u) * 8u)) & 0xffu));
    }
    printf("\"\n");
    for (std::uint32_t cue = 0u; cue < min(cue_count, 32u); ++cue) {
      if ((mask & (1u << cue)) == 0u) continue;
      const std::uint32_t span = episode_match_spans[cue * episode_count + position];
      printf("cue_occurrence_span cue=%u start=%u units=%u end=%u normalized=%u\n",
             cue, (span >> 24u) & 0x3fu, (span >> 19u) & 0x1fu,
             (span >> 13u) & 0x3fu, (span >> 30u) & 1u);
    }
  }
}
