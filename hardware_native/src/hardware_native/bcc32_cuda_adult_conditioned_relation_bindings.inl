// Learned conditioned-relation binding and cue-view kernels.
//
// Included from bcc32_cuda_adult_v1.cuh inside its namespace and state-only
// guard so this remains one dependency-ordered adult responsibility.
__global__ void collect_conditioned_subject_predicates_kernel(
    const std::uint32_t* subject_anchors,
    const ConditionedTransitionKey* conditioned_transitions,
    const std::uint32_t* conditioned_counts, std::uint32_t conditioned_count,
    std::uint32_t* predicates, std::uint32_t* predicate_count,
    std::uint32_t* predicate_owners) {
  if (blockIdx.x != 0u || threadIdx.x != 0u || subject_anchors[0] == 0xffffffffu) return;
  std::uint32_t count = 0u;
  for (std::uint32_t rank = 0u;
       rank < kCueAnchorLimit && count < kLearnedRelationPredicateLimit; ++rank) {
    const std::uint32_t subject = subject_anchors[rank];
    if (subject == 0xffffffffu) break;
    const std::uint32_t begin = bcc32_cuda_resident_synthesis::lower_subject_transition(
        conditioned_transitions, conditioned_count, subject, 0u);
    const std::uint32_t end = bcc32_cuda_resident_synthesis::upper_subject_transition(
        conditioned_transitions, conditioned_count, subject, 0xffffffffu);
    std::uint32_t prior = 0xffffffffu;
    for (std::uint32_t edge = begin;
         edge < end && count < kLearnedRelationPredicateLimit; ++edge) {
      if (conditioned_counts[edge] == 0u) continue;
      const std::uint32_t predicate = conditioned_transitions[edge].previous;
      if (predicate == subject || predicate == prior) continue;
      prior = predicate;
      bool duplicate = false;
      for (std::uint32_t existing = 0u; existing < count; ++existing) {
        if (predicates[existing] == predicate) {
          duplicate = true;
          break;
        }
      }
      if (duplicate) continue;
      predicates[count++] = predicate;
      predicate_owners[predicate] = subject;
    }
  }
  predicate_count[0] = count;
}

__global__ void map_learned_relation_cue_surfaces_kernel(
    const std::uint32_t* cue_units, std::uint32_t cue_count,
    const std::uint32_t* unit_lengths, const std::uint32_t* unit_content,
    std::uint32_t unit_count, std::uint32_t* cue_equivalent_index,
    std::uint32_t* cue_surface_scores) {
  const std::uint32_t unit = blockIdx.x * blockDim.x + threadIdx.x;
  if (unit >= unit_count) return;
  std::uint32_t best_cue = 0xffffffffu;
  std::uint32_t best_cue_score = 0u;
  for (std::uint32_t cue = 0u; cue < min(cue_count, 32u); ++cue) {
    const std::uint32_t score = learned_relation_surface_score(
        unit, cue_units[cue], unit_lengths, unit_content);
    if (score > best_cue_score ||
        (score == best_cue_score && score != 0u && cue > best_cue)) {
      best_cue = cue;
      best_cue_score = score;
    }
  }
  cue_equivalent_index[unit] = best_cue;
  cue_surface_scores[unit] = best_cue_score;
}

__device__ std::uint32_t recent_episode_pair_evidence(
    std::uint32_t first, std::uint32_t second, std::uint32_t excluded,
    const std::uint32_t* episode_units, std::uint32_t episode_count,
    const std::uint32_t* episode_breaks, std::uint32_t episode_break_count) {
  if (first == second || episode_break_count == 0u) return 0u;
  const std::uint32_t first_episode =
      episode_break_count > kLearnedRelationEpisodeMemory
          ? episode_break_count - kLearnedRelationEpisodeMemory : 0u;
  std::uint32_t best = 0u;
  std::uint32_t supporting_episodes = 0u;
  for (std::uint32_t episode = first_episode;
       episode < episode_break_count; ++episode) {
    std::uint32_t begin = episode == 0u ? 0u : episode_breaks[episode - 1u];
    const std::uint32_t end = min(episode_count, episode_breaks[episode]);
    if (end > begin + 256u) begin = end - 256u;
    bool saw_first = false;
    bool saw_second = false;
    bool saw_excluded = false;
    for (std::uint32_t position = begin; position < end; ++position) {
      saw_first |= episode_units[position] == first;
      saw_second |= episode_units[position] == second;
      saw_excluded |= episode_units[position] == excluded;
    }
    if (saw_first && saw_second && !saw_excluded) {
      best = 1u + episode - first_episode;
      ++supporting_episodes;
    }
  }
  return supporting_episodes * supporting_episodes * 32u + best;
}

__device__ std::uint32_t recent_episode_relation_evidence(
    const ConditionedTransitionKey& key,
    const std::uint32_t* episode_units, std::uint32_t episode_count,
    const std::uint32_t* episode_breaks, std::uint32_t episode_break_count) {
  if (episode_break_count == 0u) return 0u;
  const std::uint32_t first_episode =
      episode_break_count > kLearnedRelationEpisodeMemory
          ? episode_break_count - kLearnedRelationEpisodeMemory : 0u;
  std::uint32_t best = 0u;
  for (std::uint32_t episode = first_episode;
       episode < episode_break_count; ++episode) {
    std::uint32_t begin = episode == 0u ? 0u : episode_breaks[episode - 1u];
    const std::uint32_t end = min(episode_count, episode_breaks[episode]);
    if (end > begin + 256u) begin = end - 256u;
    for (std::uint32_t position = begin + 1u; position < end; ++position) {
      if (episode_units[position - 1u] != key.previous ||
          episode_units[position] != key.next) continue;
      const std::uint32_t previous_position = position - 1u;
      for (std::uint32_t lag = 0u;
           lag <= kLearnedRelationLag && previous_position >= begin + lag;
           ++lag) {
        if (episode_units[previous_position - lag] == key.anchor) {
          best = max(best, 1u + episode - first_episode);
          break;
        }
      }
    }
  }
  return best;
}

__global__ void detect_direct_conditioned_relation_kernel(
    const std::uint32_t* predicates, const std::uint32_t* predicate_count,
    const std::uint32_t* predicate_owners, const std::uint32_t* subject_anchors,
    const std::uint32_t* original_subject_anchors,
    const std::uint8_t* raw_cue_bytes, std::uint32_t raw_cue_byte_count,
    const std::uint32_t* boundary_mask, const std::uint32_t* unit_lengths,
    const std::uint32_t* unit_content, const std::uint32_t* episode_units,
    std::uint32_t episode_count, const std::uint32_t* episode_breaks,
    std::uint32_t episode_break_count, std::uint32_t* binding_enabled,
    unsigned long long* strongest_binding, std::uint32_t* selected_predicate) {
  if (blockIdx.x != 0u || threadIdx.x != 0u) return;
  binding_enabled[0] = 1u;
  const std::uint32_t selected_subject = subject_anchors[0];
  if (strongest_binding[0] == 0ull || selected_subject == 0xffffffffu ||
      selected_subject != original_subject_anchors[0]) return;
  for (std::uint32_t index = 0u; index < predicate_count[0]; ++index) {
    const std::uint32_t predicate = predicates[index];
    const std::uint32_t subject = predicate_owners[predicate];
    if (subject != selected_subject || predicate == subject ||
        !resident_unit_stem_occurs_in_raw_cue(
            predicate, raw_cue_bytes, raw_cue_byte_count, boundary_mask,
            unit_lengths, unit_content)) {
      continue;
    }
    if (!resident_unit_stem_occurs_in_raw_cue(
            subject, raw_cue_bytes, raw_cue_byte_count, boundary_mask,
            unit_lengths, unit_content)) {
      continue;
    }
    const std::uint32_t first_direct_episode =
        episode_break_count > kLearnedRelationEpisodeMemory
            ? episode_break_count - kLearnedRelationEpisodeMemory : 0u;
    for (std::uint32_t episode = first_direct_episode;
         episode < episode_break_count; ++episode) {
      const std::uint32_t begin = episode == 0u ? 0u : episode_breaks[episode - 1u];
      const std::uint32_t end = min(episode_count, episode_breaks[episode]);
      for (std::uint32_t subject_position = begin;
           subject_position + 2u < end; ++subject_position) {
        if (episode_units[subject_position] != subject) continue;
        for (std::uint32_t predicate_position = subject_position + 1u;
             predicate_position + 1u < end; ++predicate_position) {
          if (episode_units[predicate_position] != predicate) continue;
          for (std::uint32_t tail = predicate_position + 1u; tail < end; ++tail) {
            if (unit_lengths[episode_units[tail]] >= 4u &&
                !resident_unit_stem_occurs_in_raw_cue(
                    episode_units[tail], raw_cue_bytes, raw_cue_byte_count,
                    boundary_mask, unit_lengths, unit_content)) {
              binding_enabled[0] = 0u;
              strongest_binding[0] = 0ull;
              selected_predicate[0] = 0xffffffffu;
              return;
            }
          }
        }
      }
    }
  }
}

__global__ void score_learned_conditioned_relation_bindings_kernel(
    const ConditionedTransitionKey* conditioned_transitions,
    const std::uint32_t* conditioned_counts, std::uint32_t conditioned_count,
    const std::uint32_t* unit_vitality,
    const std::uint32_t* cue_equivalent_index,
    const std::uint32_t* cue_surface_scores,
    const std::uint32_t* predicates, const std::uint32_t* predicate_count,
    const std::uint32_t* cue_units,
    const std::uint32_t* cue_orders, const std::uint32_t* existing_cue_masks,
    const std::uint32_t* predicate_owners, const std::uint32_t* subject_anchors,
    const std::uint32_t* binding_enabled,
    const std::uint8_t* raw_cue_bytes, std::uint32_t raw_cue_byte_count,
    const std::uint32_t* boundary_mask,
    const std::uint32_t* unit_lengths, const std::uint32_t* unit_content,
    const std::uint32_t* episode_units, std::uint32_t episode_count,
    const std::uint32_t* episode_breaks, std::uint32_t episode_break_count,
    unsigned long long* predicate_bindings) {
  const std::uint32_t edge = blockIdx.x * blockDim.x + threadIdx.x;
  if (binding_enabled[0] == 0u || edge >= conditioned_count ||
      conditioned_counts[edge] == 0u) return;
  const ConditionedTransitionKey key = conditioned_transitions[edge];
  if (key.anchor == key.previous) return;
  const std::uint32_t cue = cue_equivalent_index[key.anchor];
  if (cue >= 32u || unit_vitality[key.anchor] == 0u || episode_count < 3u) return;
  const std::uint32_t relation_evidence = recent_episode_relation_evidence(
      key, episode_units, episode_count, episode_breaks, episode_break_count);
  if (relation_evidence == 0u) return;
  const std::uint32_t cue_score = cue_surface_scores[key.anchor];
  if (cue_score == 0u) return;
  const std::uint32_t cue_order = cue * kCueAlignmentUnits + 1u;
  const std::uint32_t anchor_begin =
      bcc32_cuda_resident_synthesis::lower_subject_transition(
          conditioned_transitions, conditioned_count, key.anchor, 0u);
  const std::uint32_t anchor_end =
      bcc32_cuda_resident_synthesis::upper_subject_transition(
          conditioned_transitions, conditioned_count, key.anchor, 0xffffffffu);
  const std::uint32_t anchor_breadth =
      1u + min(32u, anchor_end - anchor_begin);
  const unsigned long long relation_depth =
      1ull + static_cast<unsigned long long>(
                 evidence_depth(conditioned_counts[edge]));
  for (std::uint32_t index = 0u; index < predicate_count[0]; ++index) {
    const std::uint32_t predicate = predicates[index];
    const std::uint32_t subject = predicate_owners[predicate];
    if (subject == 0xffffffffu || predicate == subject || key.anchor == subject ||
        unit_vitality[predicate] > 1024u || existing_cue_masks[predicate] != 0u ||
        cue_orders[predicate] != 0xffffffffu) continue;
    if (learned_relation_surface_score(
            subject, cue_units[cue], unit_lengths, unit_content) != 0u) {
      continue;
    }
    if (resident_unit_stem_occurs_in_raw_cue(
            predicate, raw_cue_bytes, raw_cue_byte_count, boundary_mask,
            unit_lengths, unit_content)) {
      continue;
    }
    const std::uint32_t predicate_score = learned_relation_surface_score(
        key.next, predicate, unit_lengths, unit_content);
    if (predicate_score == 0u) continue;
    const std::uint32_t fact_evidence = recent_episode_pair_evidence(
        subject, predicate, cue_units[cue], episode_units, episode_count,
        episode_breaks, episode_break_count);
    if (fact_evidence == 0u) continue;
    const std::uint32_t subject_order = cue_orders[subject];
    if (subject_order == 0xffffffffu) continue;
    const std::uint32_t subject_begin =
        bcc32_cuda_resident_synthesis::lower_subject_transition(
            conditioned_transitions, conditioned_count, subject, 0u);
    const std::uint32_t subject_end =
        bcc32_cuda_resident_synthesis::upper_subject_transition(
            conditioned_transitions, conditioned_count, subject, 0xffffffffu);
    const std::uint32_t subject_breadth =
        1u + min(32u, subject_end - subject_begin);
    std::uint32_t subject_rank = kCueAnchorLimit;
    for (std::uint32_t rank = 0u; rank < kCueAnchorLimit; ++rank) {
      if (subject_anchors[rank] == subject) {
        subject_rank = rank;
        break;
      }
    }
    if (subject_rank == kCueAnchorLimit) continue;
    const std::uint32_t rank_penalty = 1u + subject_rank * subject_rank;
    const std::uint32_t distance = min(
        32u, subject_order > cue_order
                 ? (subject_order - cue_order) / kCueAlignmentUnits
                 : (cue_order - subject_order) / kCueAlignmentUnits);
    const unsigned long long specificity =
        ((static_cast<unsigned long long>(cue_score) * predicate_score) >> 16u) *
        relation_depth * relation_evidence * fact_evidence * (33u - distance) *
        subject_breadth /
        (static_cast<unsigned long long>(anchor_breadth) * rank_penalty *
         (1ull + unit_vitality[key.anchor]));
    if (specificity != 0ull) {
      atomicMax(predicate_bindings + predicate, (specificity << 6u) | cue);
    }
  }
}

__global__ void select_strongest_learned_relation_binding_kernel(
    const unsigned long long* predicate_bindings, std::uint32_t unit_count,
    unsigned long long* strongest_binding) {
  const std::uint32_t unit = blockIdx.x * blockDim.x + threadIdx.x;
  if (unit < unit_count) atomicMax(strongest_binding, predicate_bindings[unit]);
}

__global__ void select_learned_relation_subject_kernel(
    const unsigned long long* predicate_bindings,
    const unsigned long long* strongest_binding, std::uint32_t unit_count,
    const std::uint32_t* predicate_owners,
    const std::uint32_t* selected_anchor_ids, std::uint32_t anchor_count,
    std::uint32_t* subject_anchor_view, std::uint32_t* selected_predicate) {
  if (blockIdx.x != 0u || threadIdx.x != 0u) return;
  selected_predicate[0] = 0xffffffffu;
  std::uint32_t selected_subject = 0xffffffffu;
  if (strongest_binding[0] != 0ull) {
    for (std::uint32_t predicate = 0u; predicate < unit_count; ++predicate) {
      if (predicate_bindings[predicate] != strongest_binding[0] ||
          predicate_owners[predicate] == 0xffffffffu) continue;
      selected_predicate[0] = predicate;
      selected_subject = predicate_owners[predicate];
      break;
    }
  }
  std::uint32_t out = 0u;
  if (selected_subject != 0xffffffffu && out < anchor_count) {
    subject_anchor_view[out++] = selected_subject;
  }
  for (std::uint32_t rank = 0u; rank < anchor_count && out < anchor_count; ++rank) {
    const std::uint32_t candidate = selected_anchor_ids[rank];
    if (candidate == 0xffffffffu || candidate == selected_subject) continue;
    bool duplicate = false;
    for (std::uint32_t prior = 0u; prior < out; ++prior) {
      duplicate |= subject_anchor_view[prior] == candidate;
    }
    if (!duplicate) subject_anchor_view[out++] = candidate;
  }
  while (out < anchor_count) subject_anchor_view[out++] = 0xffffffffu;
}

__global__ void apply_learned_conditioned_relation_bindings_kernel(
    const unsigned long long* predicate_bindings,
    const unsigned long long* strongest_binding, std::uint32_t unit_count,
    const std::uint32_t* selected_predicate,
    const std::uint32_t* subject_anchors, const std::uint32_t* cue_units,
    std::uint32_t cue_count, unsigned long long* activation,
    std::uint32_t* cue_masks, std::uint32_t* cue_orders) {
  const std::uint32_t predicate = blockIdx.x * blockDim.x + threadIdx.x;
  if (predicate >= unit_count || predicate != selected_predicate[0] ||
      predicate == subject_anchors[0] ||
      cue_masks[predicate] != 0u || cue_orders[predicate] != 0xffffffffu) {
    return;
  }
  const unsigned long long binding = predicate_bindings[predicate];
  if (binding == 0ull || binding != strongest_binding[0]) return;
  const std::uint32_t cue = static_cast<std::uint32_t>(binding & 63ull);
  std::uint32_t subject_order = 0xffffffffu;
  for (std::uint32_t index = 0u; index < cue_count; ++index) {
    if (cue_units[index] == subject_anchors[0]) {
      subject_order = index + 1u;
      break;
    }
  }
  cue_masks[predicate] = 1u << cue;
  cue_orders[predicate] = subject_order == 0xffffffffu ? cue + 1u : subject_order + 1u;
  activation[predicate] = max(activation[predicate], binding >> 6u);
}

__global__ void build_learned_relation_cue_view_kernel(
    const std::uint32_t* cue_units, std::uint32_t cue_count,
    const std::uint32_t* subject_anchors,
    const unsigned long long* predicate_bindings,
    const unsigned long long* strongest_binding, std::uint32_t unit_count,
    const std::uint32_t* selected_predicate,
    std::uint32_t* relation_cue_units, std::uint32_t* relation_cue_count) {
  const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < cue_count) relation_cue_units[index] = cue_units[index];
  if (index >= unit_count || index != selected_predicate[0] ||
      predicate_bindings[index] == 0ull ||
      predicate_bindings[index] != strongest_binding[0]) return;
  const std::uint32_t relation_cue =
      static_cast<std::uint32_t>(predicate_bindings[index] & 63ull);
  if (relation_cue < cue_count) relation_cue_units[relation_cue] = index;
  std::uint32_t subject_cue = 0xffffffffu;
  for (std::uint32_t cue = 0u; cue < cue_count; ++cue) {
    if (cue_units[cue] == subject_anchors[0]) {
      subject_cue = cue;
      break;
    }
  }
  if (subject_cue != 0xffffffffu &&
      (relation_cue + 1u < subject_cue || relation_cue > subject_cue + 1u)) {
    relation_cue_units[cue_count] = index;
    atomicMax(relation_cue_count, cue_count + 1u);
  }
}
