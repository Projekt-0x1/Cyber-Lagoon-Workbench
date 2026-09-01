// Adult resident synthesis and surface-evidence kernels.
//
// Included after resident relation composition and before conditioned
// relation bindings. This unit owns resident completion adoption, forward
// closure, cue activation, and surface scoring; it adds no state.

__global__ void select_resident_completion_kernel(
    const std::uint32_t* unit_lengths, const std::uint32_t* unit_content,
    const std::uint32_t* cue_masks, const std::uint32_t* salient_masks,
    const std::uint32_t* closure_bytes, const std::uint32_t* boundary_bytes,
    const std::uint8_t* cue_bytes, std::uint32_t cue_byte_count,
    std::uint32_t primary_source_run_limit,
    std::uint32_t alternate_source_run_limit,
    std::uint32_t* primary_context, std::uint32_t* primary_completion,
    const std::uint32_t* alternate_context,
    const std::uint32_t* alternate_completion) {
  if (blockIdx.x != 0u || threadIdx.x != 0u) return;
  auto quality = [&](const std::uint32_t* context,
                     const std::uint32_t* completion,
                     std::uint32_t source_run_limit) {
    if (context[0] == 0u || context[12] == 0u) return 0ull;
    const std::uint32_t count = min(context[12], kCompositionUnits);
    std::uint32_t cue_mask = 0u;
    std::uint32_t salient_mask = 0u;
    std::uint32_t lexical_evidence = 0u;
    std::uint32_t closure_count = 0u;
    std::uint32_t informative_bytes = 0u;
    for (std::uint32_t index = 0u; index < count; ++index) {
      const std::uint32_t unit = completion[index];
      cue_mask |= cue_masks[unit];
      salient_mask |= salient_masks[unit];
      std::uint32_t best_overlap = 0u;
      for (std::uint32_t unit_begin = 0u;
           unit_begin < unit_lengths[unit]; ++unit_begin) {
        for (std::uint32_t cue_begin = 0u; cue_begin < cue_byte_count; ++cue_begin) {
          std::uint32_t matched = 0u;
          std::uint32_t current_index = index;
          std::uint32_t current_offset = unit_begin;
          while (current_index < count && cue_begin + matched < cue_byte_count &&
                 matched < 64u &&
                 resident_unit_byte(unit_content, completion[current_index],
                                    current_offset) == cue_bytes[cue_begin + matched]) {
            ++matched;
            ++current_offset;
            if (current_offset >= unit_lengths[completion[current_index]]) {
              ++current_index;
              current_offset = 0u;
            }
          }
          best_overlap = max(best_overlap, matched);
        }
      }
      lexical_evidence = max(lexical_evidence, best_overlap * best_overlap);
      closure_count += resident_unit_contains_any(
          unit_lengths, unit_content, unit, closure_bytes, kClosureCount);
      for (std::uint32_t offset = 0u; offset < unit_lengths[unit]; ++offset) {
        informative_bytes += resident_unit_byte(unit_content, unit, offset) !=
            boundary_bytes[0];
      }
    }
    return (static_cast<unsigned long long>(min(lexical_evidence, 65535u)) << 48u) |
        (static_cast<unsigned long long>(255u - min(source_run_limit, 255u)) << 40u) |
        (static_cast<unsigned long long>(min(__popc(salient_mask), 255)) << 32u) |
        (static_cast<unsigned long long>(min(__popc(cue_mask), 255)) << 24u) |
        (static_cast<unsigned long long>(min(closure_count, 255u)) << 16u) |
        min(static_cast<unsigned long long>(informative_bytes), 0xffffull);
  };

  if (quality(alternate_context, alternate_completion, alternate_source_run_limit) <=
      quality(primary_context, primary_completion, primary_source_run_limit)) return;
  for (std::uint32_t index = 0u; index < 16u; ++index) {
    primary_context[index] = alternate_context[index];
  }
  for (std::uint32_t index = 0u; index < kCompositionUnits; ++index) {
    primary_completion[index] = alternate_completion[index];
  }
}
__global__ void adopt_resident_synthesis_kernel(
    const bcc32_cuda_resident_synthesis::ResidentSynthesisResult* result,
    const std::uint32_t* selected_units, std::uint32_t* motor_context,
    std::uint32_t* motor_completion) {
  if (blockIdx.x != 0u) return;
  const std::uint32_t count = result->ready != 0u
      ? min(result->unit_count, kCompositionUnits) : 0u;
  for (std::uint32_t index = threadIdx.x; index < count;
       index += blockDim.x) {
    motor_completion[index] = selected_units[index];
  }
  __syncthreads();
  const std::uint32_t minimum_count = result->conditioned != 0u
      ? bcc32_cuda_resident_synthesis::kResidentSynthesisConditionedMinUnits
      : kCompositionMinUnits;
  if (threadIdx.x != 0u || count < minimum_count) return;
  motor_context[0] = 1u;
  motor_context[1] = selected_units[0];
  motor_context[2] = result->score_high;
  motor_context[3] = count;
  motor_context[4] = result->cue_coverage;
  motor_context[5] = 4u;
  motor_context[12] = count;
  motor_context[13] = result->cue_coverage;
  motor_context[15] = 1u;
}

template <typename BigramKeyT, typename TrigramKeyT>
__device__ std::uint32_t choose_novel_resident_forward_unit(
    const bcc32_cuda_resident_synthesis::ResidentSynthesisModelView<BigramKeyT,
                                                                    TrigramKeyT>& model,
    const std::uint32_t* selected_units, std::uint32_t length,
    std::uint32_t first, std::uint32_t previous, bool prefer_alternate) {
  namespace synthesis = bcc32_cuda_resident_synthesis;
  std::uint32_t best = synthesis::kResidentSynthesisInvalid;
  std::uint32_t alternate = synthesis::kResidentSynthesisInvalid;
  unsigned long long best_score = 0ull;
  unsigned long long alternate_score = 0ull;
  const TrigramKeyT* tri_keys[2] = {model.base_trigrams, model.online_trigrams};
  const std::uint32_t* tri_counts[2] = {
      model.base_trigram_counts, model.online_trigram_counts};
  const std::uint32_t tri_sizes[2] = {
      model.base_trigram_count, model.online_trigram_count};
  for (std::uint32_t source = 0u; source < 2u; ++source) {
    if (tri_keys[source] == nullptr || tri_counts[source] == nullptr) continue;
    const std::uint32_t begin = synthesis::lower_forward_trigram(
        tri_keys[source], tri_sizes[source], first, previous);
    const std::uint32_t end = synthesis::upper_forward_trigram(
        tri_keys[source], tri_sizes[source], first, previous);
    const std::uint32_t samples = synthesis::synthesis_min(
        end - begin, synthesis::kResidentSynthesisDefaultEdgeScan);
    for (std::uint32_t sample = 0u; sample < samples; ++sample) {
      const std::uint32_t edge = synthesis::synthesis_sample_index(
          begin, end, sample, samples);
      const std::uint32_t candidate = tri_keys[source][edge].next;
      const std::uint32_t support = tri_counts[source][edge];
      if (candidate >= model.unit_count || support == 0u ||
          synthesis::synthesis_forward_transition_seen(
              selected_units, 0u, length, candidate) ||
          synthesis::synthesis_forward_source_window_present(
              model, selected_units, 0u, length, candidate)) {
        continue;
      }
      unsigned long long score = synthesis::synthesis_forward_fit(
          model, first, previous, candidate, 4u);
      if (score == 0ull) continue;
      score = synthesis::synthesis_sat_add(score, support);
      synthesis::synthesis_consider_ranked<BigramKeyT, TrigramKeyT>(
          candidate, score, &best, &best_score, &alternate, &alternate_score);
    }
  }
  if (best != synthesis::kResidentSynthesisInvalid) {
    return prefer_alternate && alternate != synthesis::kResidentSynthesisInvalid
        ? alternate : best;
  }

  const BigramKeyT* bi_keys[2] = {model.base_bigrams, model.online_bigrams};
  const std::uint32_t* bi_counts[2] = {
      model.base_bigram_counts, model.online_bigram_counts};
  const std::uint32_t bi_sizes[2] = {
      model.base_bigram_count, model.online_bigram_count};
  for (std::uint32_t source = 0u; source < 2u; ++source) {
    if (bi_keys[source] == nullptr || bi_counts[source] == nullptr) continue;
    const std::uint32_t begin = synthesis::lower_forward_bigram(
        bi_keys[source], bi_sizes[source], previous);
    const std::uint32_t end = synthesis::upper_forward_bigram(
        bi_keys[source], bi_sizes[source], previous);
    const std::uint32_t samples = synthesis::synthesis_min(
        end - begin, synthesis::kResidentSynthesisDefaultEdgeScan);
    for (std::uint32_t sample = 0u; sample < samples; ++sample) {
      const std::uint32_t edge = synthesis::synthesis_sample_index(
          begin, end, sample, samples);
      const std::uint32_t candidate = bi_keys[source][edge].next;
      const std::uint32_t support = bi_counts[source][edge];
      if (candidate >= model.unit_count || support == 0u ||
          synthesis::synthesis_forward_transition_seen(
              selected_units, 0u, length, candidate) ||
          synthesis::synthesis_forward_source_window_present(
              model, selected_units, 0u, length, candidate)) {
        continue;
      }
      unsigned long long score = synthesis::synthesis_forward_fit(
          model, first, previous, candidate, 4u);
      if (score == 0ull) continue;
      score = synthesis::synthesis_sat_add(score, support);
      synthesis::synthesis_consider_ranked<BigramKeyT, TrigramKeyT>(
          candidate, score, &best, &best_score, &alternate, &alternate_score);
    }
  }
  return prefer_alternate && alternate != synthesis::kResidentSynthesisInvalid
      ? alternate : best;
}

template <typename BigramKeyT, typename TrigramKeyT>
__global__ void extend_resident_synthesis_to_output_closure_kernel(
    bcc32_cuda_resident_synthesis::ResidentSynthesisModelView<BigramKeyT,
                                                               TrigramKeyT> model,
    bcc32_cuda_resident_synthesis::ResidentSynthesisResult* result,
    std::uint32_t* selected_units, const std::uint32_t* output_closure) {
  namespace synthesis = bcc32_cuda_resident_synthesis;
  if (blockIdx.x != 0u || threadIdx.x != 0u || result->ready == 0u) return;
  std::uint32_t original_count = min(result->unit_count, kCompositionUnits);
  if (original_count == 1u && result->conditioned != 0u) {
    result->closed = 1u;
    return;
  }
  if (original_count < 2u) return;
  if (result->conditioned != 0u) {
    for (std::uint32_t at =
             bcc32_cuda_resident_synthesis::kResidentSynthesisNoveltyUnits - 1u;
         at < original_count; ++at) {
      if (resident_unit_contains_any(model.unit_lengths, model.unit_content,
                                     selected_units[at], output_closure,
                                     kClosureCount)) {
        result->unit_count = at + 1u;
        result->closed = 1u;
        return;
      }
    }
  }
  result->closed = 0u;
  if (resident_unit_contains_any(model.unit_lengths, model.unit_content,
                                 selected_units[original_count - 1u],
                                 output_closure, kClosureCount)) {
    result->closed = 1u;
    return;
  }
  std::uint32_t length = original_count;
  while (length < kCompositionUnits) {
    const std::uint32_t first = selected_units[length - 2u];
    const std::uint32_t previous = selected_units[length - 1u];
    const std::uint32_t next = choose_novel_resident_forward_unit(
        model, selected_units, length, first, previous,
        false);
    if (next == synthesis::kResidentSynthesisInvalid) return;
    selected_units[length++] = next;
    if (resident_unit_contains_any(model.unit_lengths, model.unit_content, next,
                                   output_closure, kClosureCount)) {
      std::uint32_t sentence_begin = 0u;
      for (std::uint32_t at = 0u; at + 1u < original_count; ++at) {
        if (resident_unit_contains_any(model.unit_lengths, model.unit_content,
                                       selected_units[at], output_closure,
                                       kClosureCount)) {
          sentence_begin = at + 1u;
        }
      }
      if (sentence_begin != 0u && length - sentence_begin >= kCompositionMinUnits) {
        std::uint32_t suffix_coverage = 0u;
        for (std::uint32_t at = sentence_begin; at < length; ++at) {
          suffix_coverage |= model.cue_masks == nullptr
              ? 0u : model.cue_masks[selected_units[at]];
        }
        if (__popc(suffix_coverage) >= result->cue_coverage) {
          for (std::uint32_t at = sentence_begin; at < length; ++at)
            selected_units[at - sentence_begin] = selected_units[at];
          length -= sentence_begin;
        }
      }
      result->unit_count = length;
      result->closed = 1u;
      return;
    }
  }
}

__global__ void seed_resident_synthesis_cue_activation_kernel(
    const std::uint32_t* best_ids, const std::uint32_t* best_scores,
    const std::uint32_t* matched_segment_counts, std::uint32_t cue_count,
    const std::uint32_t* unit_lengths, const std::uint32_t* unit_vitality,
    std::uint32_t unit_count, unsigned long long* activation,
    std::uint32_t* cue_masks, std::uint32_t* salient_masks) {
  const std::uint32_t cue = blockIdx.x * blockDim.x + threadIdx.x;
  if (cue >= cue_count) return;
  const std::uint32_t unit = best_ids[cue];
  if (unit >= unit_count || best_scores[cue] < kCueNearIdentity) return;
  const unsigned long long extent = 1ull + min(unit_lengths[unit], 32u);
  const unsigned long long recurrence =
      1ull + static_cast<unsigned long long>(matched_segment_counts[cue]);
  const unsigned long long stability =
      1ull + static_cast<unsigned long long>(evidence_depth(unit_vitality[unit]));
  const unsigned long long specificity =
      static_cast<unsigned long long>(best_scores[cue]) * extent * extent *
      stability / recurrence;
  atomicMax(activation + unit, specificity);
  if (cue < 32u) {
    const std::uint32_t mask = 1u << cue;
    atomicOr(cue_masks + unit, mask);
    if (specificity >= (1ull << 20u)) atomicOr(salient_masks + unit, mask);
  }
}

__global__ void propagate_resident_synthesis_cue_bytes_kernel(
    const std::uint32_t* best_ids, const std::uint32_t* best_scores,
    const std::uint32_t* matched_segment_counts, std::uint32_t cue_count,
    const std::uint32_t* unit_lengths, const std::uint32_t* unit_content,
    const std::uint32_t* unit_vitality, std::uint32_t unit_count,
    unsigned long long* activation, std::uint32_t* cue_masks,
    std::uint32_t* salient_masks) {
  const std::uint32_t unit = blockIdx.x * blockDim.x + threadIdx.x;
  if (unit >= unit_count) return;
  unsigned long long strongest = activation[unit];
  std::uint32_t masks = cue_masks[unit];
  std::uint32_t salient = salient_masks[unit];
  for (std::uint32_t cue = 0u; cue < min(cue_count, 32u); ++cue) {
    const std::uint32_t representative = best_ids[cue];
    if (representative >= unit_count || best_scores[cue] < kCueNearIdentity ||
        representative == unit ||
        unit_lengths[representative] != unit_lengths[unit]) {
      continue;
    }
    bool equal = true;
    for (std::uint32_t word = 0u; word < kUnitWords; ++word) {
      equal &= unit_content[representative * kUnitWords + word] ==
               unit_content[unit * kUnitWords + word];
    }
    if (!equal) continue;
    const unsigned long long extent = 1ull + min(unit_lengths[unit], 32u);
    const unsigned long long recurrence =
        1ull + static_cast<unsigned long long>(matched_segment_counts[cue]);
    const unsigned long long stability =
        1ull + static_cast<unsigned long long>(evidence_depth(unit_vitality[unit]));
    const unsigned long long specificity =
        static_cast<unsigned long long>(best_scores[cue]) * extent * extent *
        stability / recurrence;
    strongest = max(strongest, specificity);
    masks |= 1u << cue;
    if (specificity >= (1ull << 20u)) salient |= 1u << cue;
  }
  activation[unit] = strongest;
  cue_masks[unit] = masks;
  salient_masks[unit] = salient;
}

__global__ void propagate_resident_synthesis_cue_activation_kernel(
    const AssociationKey* associations,
    const std::uint32_t* association_counts, std::uint32_t association_count,
    const std::uint32_t* unit_vitality,
    const unsigned long long* direct_activation,
    const std::uint32_t* direct_masks, const std::uint32_t* direct_salient_masks,
    unsigned long long* propagated_activation, std::uint32_t* propagated_masks,
    std::uint32_t* propagated_salient_masks) {
  const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= association_count || association_counts[index] == 0u) return;
  const AssociationKey relation = associations[index];
  const std::uint32_t edge_depth = 1u + evidence_depth(association_counts[index]);
  const unsigned long long first = direct_activation[relation.first];
  const unsigned long long second = direct_activation[relation.second];
  if (first != 0ull && unit_vitality[relation.second] >= 4u) {
    const std::uint32_t cost = 1u + unit_vitality[relation.second];
    atomicAdd(propagated_activation + relation.second,
              first * edge_depth / cost);
    if (static_cast<unsigned long long>(association_counts[index]) * 256ull >=
        unit_vitality[relation.second]) {
      atomicOr(propagated_masks + relation.second, direct_masks[relation.first]);
      atomicOr(propagated_salient_masks + relation.second,
               direct_salient_masks[relation.first]);
    }
  }
  if (second != 0ull && unit_vitality[relation.first] >= 4u) {
    const std::uint32_t cost = 1u + unit_vitality[relation.first];
    atomicAdd(propagated_activation + relation.first,
              second * edge_depth / cost);
    if (static_cast<unsigned long long>(association_counts[index]) * 256ull >=
        unit_vitality[relation.first]) {
      atomicOr(propagated_masks + relation.first, direct_masks[relation.second]);
      atomicOr(propagated_salient_masks + relation.first,
               direct_salient_masks[relation.second]);
    }
  }
}

__global__ void propagate_resident_synthesis_direction_kernel(
    const AssociationKey* associations,
    const std::uint32_t* association_counts, std::uint32_t association_count,
    const std::uint32_t* unit_vitality,
    const unsigned long long* direct_activation,
    const std::uint32_t* direct_masks,
    unsigned long long* forward_activation,
    unsigned long long* backward_activation,
    std::uint32_t* forward_support, std::uint32_t* backward_support) {
  const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= association_count || association_counts[index] == 0u) return;
  const AssociationKey relation = associations[index];
  const std::uint32_t edge_depth = 1u + evidence_depth(association_counts[index]);
  const unsigned long long first = direct_activation[relation.first];
  const unsigned long long second = direct_activation[relation.second];
  if (first != 0ull && unit_vitality[relation.second] >= 4u) {
    const std::uint32_t cost = 1u + unit_vitality[relation.second];
    atomicAdd(forward_activation + relation.second,
              first * edge_depth / cost);
    if (static_cast<unsigned long long>(association_counts[index]) * 256ull >=
        unit_vitality[relation.second]) {
      atomicOr(forward_support + relation.second, direct_masks[relation.first]);
    }
  }
  if (second != 0ull && unit_vitality[relation.first] >= 4u) {
    const std::uint32_t cost = 1u + unit_vitality[relation.first];
    atomicAdd(backward_activation + relation.first,
              second * edge_depth / cost);
    if (static_cast<unsigned long long>(association_counts[index]) * 256ull >=
        unit_vitality[relation.first]) {
      atomicOr(backward_support + relation.first, direct_masks[relation.second]);
    }
  }
}

__global__ void merge_resident_synthesis_cue_activation_kernel(
    const std::uint32_t* route_scores, const std::uint32_t* cue_focus,
    const unsigned long long* propagated_activation,
    const std::uint32_t* propagated_masks,
    const std::uint32_t* propagated_salient_masks, std::uint32_t unit_count,
    unsigned long long* activation, std::uint32_t* cue_masks,
    std::uint32_t* salient_masks) {
  const std::uint32_t unit = blockIdx.x * blockDim.x + threadIdx.x;
  if (unit >= unit_count) return;
  const unsigned long long routed =
      (static_cast<unsigned long long>(route_scores[unit]) << 16u) |
      min(cue_focus[unit], 0xffffu);
  activation[unit] = max(max(activation[unit], propagated_activation[unit]), routed);
}

constexpr std::uint32_t kLearnedRelationPredicateLimit = 128u;
constexpr std::uint32_t kLearnedRelationLag = 4u;
constexpr std::uint32_t kLearnedRelationEpisodeMemory = 32u;

__device__ std::uint32_t learned_relation_surface_score(
    std::uint32_t first, std::uint32_t second,
    const std::uint32_t* unit_lengths, const std::uint32_t* unit_content) {
  const std::uint32_t first_length = unit_lengths[first];
  const std::uint32_t second_length = unit_lengths[second];
  const std::uint32_t common = min(first_length, second_length);
  if (common < 4u) return 0u;
  std::uint32_t prefix = 0u;
  while (prefix < common) {
    const std::uint32_t first_packed = unit_content[first * kUnitWords + prefix / 4u];
    const std::uint32_t second_packed = unit_content[second * kUnitWords + prefix / 4u];
    const std::uint8_t first_byte =
        static_cast<std::uint8_t>(first_packed >> ((prefix % 4u) * 8u));
    const std::uint8_t second_byte =
        static_cast<std::uint8_t>(second_packed >> ((prefix % 4u) * 8u));
    if (first_byte != second_byte) break;
    ++prefix;
  }
  const std::uint32_t extent = max(first_length, second_length);
  if (prefix == common && first_length == second_length) return 65535u;
  if (prefix < 4u || prefix * 3u < common * 2u || extent - common > 4u) return 0u;
  return static_cast<std::uint32_t>(
      static_cast<unsigned long long>(prefix) * 65535ull / extent);
}

__device__ bool resident_unit_stem_occurs_in_raw_cue(
    std::uint32_t unit, const std::uint8_t* raw_cue_bytes,
    std::uint32_t raw_cue_byte_count, const std::uint32_t* boundary_mask,
    const std::uint32_t* unit_lengths, const std::uint32_t* unit_content) {
  const std::uint32_t length = unit_lengths[unit];
  for (std::uint32_t begin = 0u; begin < raw_cue_byte_count; ++begin) {
    if (begin != 0u && boundary_mask[raw_cue_bytes[begin - 1u]] == 0u) continue;
    std::uint32_t prefix = 0u;
    std::uint32_t content_prefix = 0u;
    while (prefix < length && begin + prefix < raw_cue_byte_count) {
      const std::uint32_t packed =
          unit_content[unit * kUnitWords + prefix / 4u];
      if (raw_cue_bytes[begin + prefix] != static_cast<std::uint8_t>(
              packed >> ((prefix % 4u) * 8u))) {
        break;
      }
      content_prefix += boundary_mask[raw_cue_bytes[begin + prefix]] == 0u;
      ++prefix;
    }
    if (content_prefix >= 4u) return true;
  }
  return false;
}
