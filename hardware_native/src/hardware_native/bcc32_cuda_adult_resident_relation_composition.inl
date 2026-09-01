// Resident relation composition kernel.
//
// Included from bcc32_cuda_adult_v1.cuh inside its namespace and state-only
// guard as one dependency-ordered resident causal responsibility.
__global__ void compose_resident_relation_kernel(
    const std::uint32_t* vitality, const std::uint32_t* unit_lengths,
    const std::uint32_t* unit_content, std::uint32_t unit_count,
    const std::uint32_t* boundary_histogram,
    const std::uint32_t* boundary_pairs, const std::uint32_t* boundary_bytes,
    const std::uint32_t* closure_bytes,
    const std::uint8_t* cue_bytes, std::uint32_t cue_byte_count,
    std::uint32_t cue_count,
    const std::uint32_t* cue_scores, const std::uint32_t* cue_orders,
    const std::uint32_t* cue_weights, const std::uint32_t* cue_masks,
    const std::uint32_t* salient_masks, const unsigned long long* forward_scores,
    const unsigned long long* backward_scores, const std::uint32_t* forward_support,
    const std::uint32_t* backward_support, const std::uint32_t* forward_salient_support,
    const unsigned long long* strongest_evidence,
    const unsigned long long* second_evidence, const BigramKey* base_bigrams,
    const std::uint32_t* base_bigram_counts, std::uint32_t base_bigram_count,
    const TrigramKey* base_trigrams, const std::uint32_t* base_trigram_counts,
    std::uint32_t base_trigram_count, const BigramKey* online_bigrams,
    const std::uint32_t* online_bigram_counts, std::uint32_t online_bigram_count,
    const TrigramKey* online_trigrams, const std::uint32_t* online_trigram_counts,
    std::uint32_t online_trigram_count, const AssociationKey* associations,
    const std::uint32_t* association_counts, std::uint32_t association_count,
    const unsigned long long* association_mass,
    const std::uint32_t* base_episode_units,
    std::uint32_t base_episode_count, const std::uint32_t* online_episode_units,
    std::uint32_t online_episode_count, const std::uint32_t* online_episode_breaks,
    std::uint32_t online_episode_break_count,
    const std::uint32_t* scoped_episode_match_mask,
    std::uint32_t scoped_episode_begin, std::uint32_t scoped_episode_count,
    std::uint32_t novelty_scope, std::uint32_t source_run_limit,
    const LocalSeedCandidate* local_seed_candidates,
    std::uint32_t local_seed_candidate_count,
    const std::uint32_t* admitted_segment_mask, std::uint32_t* motor_context,
    std::uint32_t* completion) {
  constexpr std::uint32_t candidate_count = kBlock * kCompositionBeamExpansions;
  __shared__ std::uint32_t beam_units[kCompositionBeamWidth * kCompositionMaxUnits];
  __shared__ std::uint32_t next_units[kCompositionBeamWidth * kCompositionMaxUnits];
  __shared__ unsigned long long beam_scores[kCompositionBeamWidth];
  __shared__ unsigned long long next_scores[kCompositionBeamWidth];
  __shared__ std::uint32_t beam_anchors[kCompositionBeamWidth];
  __shared__ std::uint32_t next_anchors[kCompositionBeamWidth];
  __shared__ std::uint32_t beam_last_anchor[kCompositionBeamWidth];
  __shared__ std::uint32_t next_last_anchor[kCompositionBeamWidth];
  __shared__ std::uint32_t beam_target_anchor[kCompositionBeamWidth];
  __shared__ std::uint32_t next_target_anchor[kCompositionBeamWidth];
  __shared__ std::uint32_t beam_launch_end[kCompositionBeamWidth];
  __shared__ std::uint32_t next_launch_end[kCompositionBeamWidth];
  __shared__ std::uint32_t candidate_ids[candidate_count];
  __shared__ std::uint32_t candidate_parents[candidate_count];
  __shared__ std::uint32_t candidate_anchors[candidate_count];
  __shared__ std::uint32_t candidate_salient[candidate_count];
  __shared__ std::uint32_t candidate_targets[candidate_count];
  __shared__ unsigned long long candidate_scores[candidate_count];
  __shared__ std::uint32_t selected_candidates[kCompositionBeamWidth];
  __shared__ std::uint32_t beam_copied[kCompositionBeamWidth];
  __shared__ std::uint32_t beam_count;
  __shared__ std::uint32_t beam_length;
  __shared__ std::uint32_t admitted_anchor_mask;
  __shared__ std::uint32_t novel_closure_done;

  auto insert_candidate = [&](std::uint32_t slot_begin, std::uint32_t id,
                              std::uint32_t parent, std::uint32_t anchors,
                              std::uint32_t salient, std::uint32_t target,
                              unsigned long long score) {
    if (id == 0xffffffffu || score == 0u) return;
    std::uint32_t slot = kCompositionBeamExpansions;
    for (std::uint32_t i = 0u; i < kCompositionBeamExpansions; ++i) {
      const std::uint32_t at = slot_begin + i;
      if (score > candidate_scores[at] ||
          (score == candidate_scores[at] && id < candidate_ids[at])) {
        slot = i;
        break;
      }
    }
    if (slot == kCompositionBeamExpansions) return;
    for (std::uint32_t i = kCompositionBeamExpansions - 1u; i > slot; --i) {
      const std::uint32_t to = slot_begin + i;
      const std::uint32_t from = to - 1u;
      candidate_ids[to] = candidate_ids[from];
      candidate_parents[to] = candidate_parents[from];
      candidate_anchors[to] = candidate_anchors[from];
      candidate_salient[to] = candidate_salient[from];
      candidate_targets[to] = candidate_targets[from];
      candidate_scores[to] = candidate_scores[from];
    }
    const std::uint32_t at = slot_begin + slot;
    candidate_ids[at] = id;
    candidate_parents[at] = parent;
    candidate_anchors[at] = anchors;
    candidate_salient[at] = salient;
    candidate_targets[at] = target;
    candidate_scores[at] = score;
  };

  const std::uint32_t local_begin = threadIdx.x * kCompositionBeamExpansions;
  for (std::uint32_t i = 0u; i < kCompositionBeamExpansions; ++i) {
    candidate_ids[local_begin + i] = 0xffffffffu;
    candidate_parents[local_begin + i] = 0xffffffffu;
    candidate_anchors[local_begin + i] = 0u;
    candidate_salient[local_begin + i] = 0u;
    candidate_targets[local_begin + i] = 0xffffffffu;
    candidate_scores[local_begin + i] = 0u;
  }
  if (threadIdx.x == 0u) {
    novel_closure_done = 0u;
    admitted_anchor_mask = admitted_segment_mask[0];
    motor_context[14] = (__popc(admitted_anchor_mask) << 24u) |
        (motor_context[14] & 0x00ffffffu);
    const std::uint32_t bounded_cues = min(cue_count, kCueAnchorLimit);
    const bool candidate_present = local_seed_candidate_count != 0u &&
        local_seed_candidates[0].score != 0u;
    const bool distinctive_supported = candidate_present &&
        local_seed_candidates[0].distinctive_total >= 2u &&
        local_seed_candidates[0].distinctive_coverage >= 2u;
    const bool unsupported_base_cue = novelty_scope == 0u &&
        __popc(admitted_anchor_mask) < bounded_cues;
    const bool legacy_supported = !unsupported_base_cue &&
        __popc(admitted_anchor_mask) * 5u >= bounded_cues * 3u;
    const bool ordered_supported = candidate_present &&
        local_seed_candidates[0].contiguous_run >= 3u;
    const bool bounded_candidate = candidate_present &&
        local_seed_candidates[0].episode_end -
            local_seed_candidates[0].episode_begin <= kCompositionUnits;
    const bool novelty_supported = bounded_candidate
        ? (distinctive_supported || ordered_supported) : legacy_supported;
    const bool cue_supported = novelty_scope != 0u
        ? novelty_supported : legacy_supported;
    if (!cue_supported) {
      motor_context[12] = 0u;
      motor_context[13] = 0u;
      motor_context[15] = 0u;
    } else if (candidate_present) {
      motor_context[15] = 1u;
    } else {
      motor_context[12] = 0u;
      motor_context[13] = 0u;
      motor_context[15] = 0u;
    }
  }
  __syncthreads();
  if (motor_context[15] == 0u && novelty_scope == 0u) return;

  if (threadIdx.x == 0u && novelty_scope != 0u &&
      scoped_episode_count <= 4096u &&
      local_seed_candidate_count != 0u &&
      local_seed_candidates[0].score != 0u) {
    const LocalSeedCandidate candidate = local_seed_candidates[0];
    const std::uint32_t join_scan = min(local_seed_candidate_count, 128u);
    std::uint32_t graph_closure_index = 0u;
    std::uint32_t graph_closure_support = 0u;
    for (std::uint32_t closure = 0u; closure < kClosureCount; ++closure) {
      std::uint32_t support = 0u;
      for (std::uint32_t position = 0u; position < online_episode_count; ++position) {
        support += resident_unit_contains_any(
            unit_lengths, unit_content, online_episode_units[position],
            closure_bytes + closure, 1u);
      }
      if (support > graph_closure_support) {
        graph_closure_support = support;
        graph_closure_index = closure;
      }
    }
    const std::uint32_t* graph_closure_bytes =
        closure_bytes + graph_closure_index;

    // A cue can name the two ends of a learned chain in an order that differs
    // from the order of the clauses that carried it. Treat each resident
    // clause as an undirected edge between byte-identical repeated spans, then
    // walk from the cue span named later to the one named earlier. Direction
    // comes from the cue's own resident order; no relation word or punctuation
    // value is authored here.
    LocalSeedCandidate graph_start = candidate;
    LocalSeedCandidate graph_target = candidate;
    std::uint32_t graph_pair_distance = 0u;
    for (std::uint32_t left_rank = 0u; left_rank < join_scan; ++left_rank) {
      LocalSeedCandidate left = local_seed_candidates[left_rank];
      if (left.score == 0u || left.anchor_cue >= cue_count) break;
      if (left.anchor_position < scoped_episode_begin) continue;
      const std::uint32_t left_local = left.anchor_position - scoped_episode_begin;
      if (left_local >= scoped_episode_count) continue;
      const std::uint32_t left_endpoint_mask =
          scoped_episode_match_mask[left_local] & left.distinctive_cue_mask;
      if (left_endpoint_mask == 0u) continue;
      left.anchor_cue = static_cast<std::uint32_t>(__ffs(left_endpoint_mask) - 1);
      std::uint32_t left_begin = 0u;
      std::uint32_t left_end = 0u;
      resident_clause_bounds(
          unit_lengths, unit_content, online_episode_units, online_episode_count,
          online_episode_breaks, online_episode_break_count, left.anchor_position,
          graph_closure_bytes, 1u, &left_begin, &left_end);
      for (std::uint32_t right_rank = left_rank + 1u;
           right_rank < join_scan; ++right_rank) {
        LocalSeedCandidate right = local_seed_candidates[right_rank];
        if (right.score == 0u || right.anchor_cue >= cue_count) break;
        if (right.anchor_position < scoped_episode_begin) continue;
        const std::uint32_t right_local = right.anchor_position - scoped_episode_begin;
        if (right_local >= scoped_episode_count) continue;
        const std::uint32_t right_endpoint_mask =
            scoped_episode_match_mask[right_local] & right.distinctive_cue_mask;
        if (right_endpoint_mask == 0u) continue;
        right.anchor_cue =
            static_cast<std::uint32_t>(__ffs(right_endpoint_mask) - 1);
        std::uint32_t right_begin = 0u;
        std::uint32_t right_end = 0u;
        resident_clause_bounds(
            unit_lengths, unit_content, online_episode_units, online_episode_count,
            online_episode_breaks, online_episode_break_count, right.anchor_position,
            graph_closure_bytes, 1u, &right_begin, &right_end);
        if (left_begin == right_begin && left_end == right_end) continue;
        const std::uint32_t distance = left.anchor_cue > right.anchor_cue
            ? left.anchor_cue - right.anchor_cue
            : right.anchor_cue - left.anchor_cue;
        if (distance <= graph_pair_distance) continue;
        graph_pair_distance = distance;
        graph_start = left.anchor_cue > right.anchor_cue ? left : right;
        graph_target = left.anchor_cue > right.anchor_cue ? right : left;
      }
    }

    constexpr std::uint32_t kDiscoursePathLimit = 8u;
    if (graph_pair_distance != 0u && novel_closure_done == 0u) {
      std::uint32_t clause_begin[kDiscoursePathLimit + 1u]{};
      std::uint32_t clause_end[kDiscoursePathLimit + 1u]{};
      std::uint32_t edge_begin[kDiscoursePathLimit]{};
      std::uint32_t edge_end[kDiscoursePathLimit]{};
      std::uint32_t next_edge_begin[kDiscoursePathLimit]{};
      std::uint32_t next_edge_end[kDiscoursePathLimit]{};
      std::uint32_t graph_episode_ids[kDiscoursePathLimit + 1u]{};
      resident_clause_bounds(
          unit_lengths, unit_content, online_episode_units, online_episode_count,
          online_episode_breaks, online_episode_break_count,
          graph_start.anchor_position, graph_closure_bytes, 1u, clause_begin,
          clause_end);
      std::uint32_t target_begin = 0u;
      std::uint32_t target_end = 0u;
      resident_clause_bounds(
          unit_lengths, unit_content, online_episode_units, online_episode_count,
          online_episode_breaks, online_episode_break_count,
          graph_target.anchor_position, graph_closure_bytes, 1u, &target_begin,
          &target_end);

      std::uint32_t incoming_begin = graph_start.position;
      std::uint32_t incoming_end = min(
          clause_end[0], incoming_begin + max(1u, graph_start.contiguous_run));
      std::uint32_t graph_edges = 0u;
      const bool source_order_inverted = clause_begin[0] > target_begin;
      bool graph_crossed_episode = false;
      std::uint32_t graph_episode_count = 1u;
      for (std::uint32_t episode = 0u;
           episode < online_episode_break_count; ++episode) {
        graph_episode_ids[0] += clause_begin[0] >= online_episode_breaks[episode];
      }
      bool endpoints_cross_episode = false;
      for (std::uint32_t episode = 0u;
           episode < online_episode_break_count; ++episode) {
        const std::uint32_t boundary = online_episode_breaks[episode];
        if ((clause_begin[0] < boundary) != (target_begin < boundary)) {
          endpoints_cross_episode = true;
          break;
        }
      }
      const bool start_is_earlier = clause_begin[0] < target_begin;
      const LocalSeedCandidate earlier_endpoint =
          start_is_earlier ? graph_start : graph_target;
      const LocalSeedCandidate later_endpoint =
          start_is_earlier ? graph_target : graph_start;
      const std::uint32_t earlier_clause_begin =
          start_is_earlier ? clause_begin[0] : target_begin;
      const std::uint32_t later_clause_end =
          start_is_earlier ? target_end : clause_end[0];
      bool forward_bridge_supported = false;
      for (std::uint32_t left = earlier_clause_begin;
           left < earlier_endpoint.episode_end && !forward_bridge_supported;
           ++left) {
        const std::uint32_t bridge = online_episode_units[left];
        if (unit_lengths[bridge] < 5u ||
            static_cast<unsigned long long>(unit_lengths[bridge]) * 16ull <
                vitality[bridge]) continue;
        for (std::uint32_t right = later_endpoint.episode_begin;
             right < later_clause_end; ++right) {
          if (same_resident_unit_bytes(
                  unit_lengths, unit_content, bridge,
                  online_episode_units[right])) {
            forward_bridge_supported = true;
            break;
          }
        }
      }
      bool graph_complete = clause_begin[0] == target_begin &&
                            clause_end[0] == target_end;
      while (!graph_complete &&
             graph_edges < kDiscoursePathLimit) {
        std::uint32_t best_left = 0u;
        std::uint32_t best_right = 0u;
        std::uint32_t best_run = 0u;
        std::uint32_t best_bytes = 0u;
        std::uint32_t best_clause_begin = 0u;
        std::uint32_t best_clause_end = 0u;
        const std::uint32_t current_begin = clause_begin[graph_edges];
        const std::uint32_t current_end = clause_end[graph_edges];

        for (std::uint32_t left = current_begin; left < current_end; ++left) {
          for (std::uint32_t right = 0u; right < online_episode_count; ++right) {
            std::uint32_t other_begin = 0u;
            std::uint32_t other_end = 0u;
            resident_clause_bounds(
                unit_lengths, unit_content, online_episode_units,
                online_episode_count, online_episode_breaks,
                online_episode_break_count, right, graph_closure_bytes, 1u,
                &other_begin, &other_end);
            if (other_begin == current_begin && other_end == current_end) continue;
            bool visited = false;
            for (std::uint32_t prior = 0u; prior <= graph_edges; ++prior) {
              if (clause_begin[prior] == other_begin &&
                  clause_end[prior] == other_end) {
                visited = true;
                break;
              }
            }
            if (visited) continue;
            if (!same_resident_unit_bytes(
                    unit_lengths, unit_content, online_episode_units[left],
                    online_episode_units[right])) continue;

            std::uint32_t run = 0u;
            std::uint32_t run_bytes = 0u;
            while (run < 8u && left + run < current_end &&
                   right + run < other_end &&
                   same_resident_unit_bytes(
                       unit_lengths, unit_content,
                       online_episode_units[left + run],
                       online_episode_units[right + run])) {
              const std::uint32_t unit = online_episode_units[left + run];
              if (resident_unit_contains_any(unit_lengths, unit_content, unit,
                                             graph_closure_bytes, 1u)) break;
              run_bytes += unit_lengths[unit];
              ++run;
            }
            if (run == 0u || run_bytes < 8u || run_bytes < best_bytes) continue;
            if (left < incoming_end && left + run > incoming_begin) continue;

            std::uint32_t occurrences = 0u;
            for (std::uint32_t probe = 0u;
                 probe + run <= online_episode_count && occurrences <= 2u;
                 ++probe) {
              bool equal = true;
              for (std::uint32_t part = 0u; part < run; ++part) {
                if (!same_resident_unit_bytes(
                        unit_lengths, unit_content,
                        online_episode_units[left + part],
                        online_episode_units[probe + part])) {
                  equal = false;
                  break;
                }
              }
              occurrences += equal;
            }
            if (occurrences != 2u) continue;
            const bool reaches_target = other_begin == target_begin &&
                                        other_end == target_end;
            const bool best_reaches_target = best_clause_begin == target_begin &&
                                             best_clause_end == target_end;
            if (run_bytes == best_bytes &&
                (!reaches_target || best_reaches_target)) continue;
            best_left = left;
            best_right = right;
            best_run = run;
            best_bytes = run_bytes;
            best_clause_begin = other_begin;
            best_clause_end = other_end;
          }
        }
        if (best_run == 0u) break;
        edge_begin[graph_edges] = best_left;
        edge_end[graph_edges] = best_left + best_run;
        next_edge_begin[graph_edges] = best_right;
        next_edge_end[graph_edges] = best_right + best_run;
        ++graph_edges;
        clause_begin[graph_edges] = best_clause_begin;
        clause_end[graph_edges] = best_clause_end;
        for (std::uint32_t episode = 0u;
             episode < online_episode_break_count; ++episode) {
          const std::uint32_t boundary = online_episode_breaks[episode];
          if ((current_begin < boundary) != (best_clause_begin < boundary)) {
            graph_crossed_episode = true;
            break;
          }
        }
        std::uint32_t next_episode_id = 0u;
        for (std::uint32_t episode = 0u;
             episode < online_episode_break_count; ++episode) {
          next_episode_id += best_clause_begin >= online_episode_breaks[episode];
        }
        bool episode_seen = false;
        for (std::uint32_t prior = 0u; prior < graph_episode_count; ++prior) {
          if (graph_episode_ids[prior] == next_episode_id) {
            episode_seen = true;
            break;
          }
        }
        if (!episode_seen && graph_episode_count < kDiscoursePathLimit + 1u)
          graph_episode_ids[graph_episode_count++] = next_episode_id;
        incoming_begin = best_right;
        incoming_end = best_right + best_run;
        graph_complete = best_clause_begin == target_begin &&
                         best_clause_end == target_end;
      }

      if (source_order_inverted && graph_complete && graph_edges != 0u &&
          graph_crossed_episode && graph_episode_count >= 3u) {
        std::uint32_t out = 0u;
        std::uint32_t left_cue_evidence = 0u;
        std::uint32_t right_cue_evidence = 0u;
        for (std::uint32_t at = clause_begin[0]; at < edge_begin[0]; ++at) {
          if (at < scoped_episode_begin) continue;
          const std::uint32_t local = at - scoped_episode_begin;
          if (local >= scoped_episode_count) break;
          left_cue_evidence += __popc(
              scoped_episode_match_mask[local] & graph_start.cue_mask);
        }
        for (std::uint32_t at = edge_end[0]; at < clause_end[0]; ++at) {
          if (at < scoped_episode_begin) continue;
          const std::uint32_t local = at - scoped_episode_begin;
          if (local >= scoped_episode_count) break;
          right_cue_evidence += __popc(
              scoped_episode_match_mask[local] & graph_start.cue_mask);
        }
        const bool start_after_edge = right_cue_evidence != left_cue_evidence
            ? right_cue_evidence > left_cue_evidence
            : graph_start.anchor_position >= edge_end[0];
        const std::uint32_t start_begin = start_after_edge
            ? edge_end[0] : clause_begin[0];
        const std::uint32_t start_end = start_after_edge
            ? clause_end[0] : edge_begin[0];
        for (std::uint32_t at = start_begin;
             at < start_end && out < kCompositionUnits; ++at) {
          completion[out++] = online_episode_units[at];
        }
        for (std::uint32_t edge = 0u;
             edge < graph_edges && out < kCompositionUnits; ++edge) {
          for (std::uint32_t at = edge_begin[edge];
               at < edge_end[edge] && out < kCompositionUnits; ++at) {
            completion[out++] = online_episode_units[at];
          }
          const bool current_successor_available =
              edge_end[edge] < clause_end[edge];
          const bool next_successor_available =
              next_edge_end[edge] < clause_end[edge + 1u];
          if (out < kCompositionUnits &&
              (current_successor_available || next_successor_available)) {
            std::uint32_t seam_unit = current_successor_available
                ? online_episode_units[edge_end[edge]]
                : online_episode_units[next_edge_end[edge]];
            if (current_successor_available && next_successor_available) {
              const std::uint32_t current_unit =
                  online_episode_units[edge_end[edge]];
              const std::uint32_t next_unit =
                  online_episode_units[next_edge_end[edge]];
              const bool current_closes = resident_unit_contains_any(
                  unit_lengths, unit_content, current_unit,
                  graph_closure_bytes, 1u);
              const bool next_closes = resident_unit_contains_any(
                  unit_lengths, unit_content, next_unit,
                  graph_closure_bytes, 1u);
              if (next_closes != current_closes) {
                seam_unit = next_closes ? next_unit : current_unit;
              } else if (unit_lengths[next_unit] < unit_lengths[current_unit]) {
                seam_unit = next_unit;
              }
            }
            completion[out++] = seam_unit;
          }
        }
        const std::uint32_t target_incoming_begin =
            next_edge_begin[graph_edges - 1u];
        const std::uint32_t target_incoming_end =
            next_edge_end[graph_edges - 1u];
        const bool target_before_edge =
            graph_target.anchor_position < target_incoming_begin;
        const std::uint32_t final_begin = target_before_edge
            ? target_begin : target_incoming_end;
        const std::uint32_t final_end = target_before_edge
            ? target_incoming_begin : target_end;
        for (std::uint32_t at = final_begin;
             at < final_end && out < kCompositionUnits; ++at) {
          completion[out++] = online_episode_units[at];
        }
        if (out >= kCompositionMinUnits) {
          motor_context[0] = 1u;
          motor_context[1] = completion[0];
          motor_context[2] = static_cast<std::uint32_t>(min(
              max(graph_start.score, graph_target.score) >> 32u,
              0xffffffffull));
          motor_context[3] = out;
          motor_context[4] = graph_edges;
          motor_context[5] = 4u;
          motor_context[12] = out;
          motor_context[13] = graph_edges;
          novel_closure_done = 1u;
        }
      }
      if (endpoints_cross_episode && !graph_complete &&
          !forward_bridge_supported &&
          novel_closure_done == 0u) {
        // The cue selected endpoints in different episodes, but resident
        // repeated-span matter contained no path between those endpoints.
        // Do not let a later fallback substitute an unrelated co-resident
        // chain that happens to share some other unit.
        novel_closure_done = 1u;
      }
    }

    for (std::uint32_t rank = 1u;
         rank < join_scan && novel_closure_done == 0u; ++rank) {
      const LocalSeedCandidate peer = local_seed_candidates[rank];
      if (peer.score == 0u) break;
      if (peer.episode_begin == candidate.episode_begin ||
          peer.episode_end == candidate.episode_end) continue;
      if (candidate.episode_end - candidate.episode_begin > kCompositionUnits ||
          peer.episode_end - peer.episode_begin > kCompositionUnits) continue;
      const std::uint32_t candidate_private =
          candidate.distinctive_cue_mask & ~peer.distinctive_cue_mask;
      const std::uint32_t peer_private =
          peer.distinctive_cue_mask & ~candidate.distinctive_cue_mask;
      if (candidate_private == 0u || peer_private == 0u) continue;

      const LocalSeedCandidate first = candidate.episode_begin < peer.episode_begin
          ? candidate : peer;
      const LocalSeedCandidate second = candidate.episode_begin < peer.episode_begin
          ? peer : candidate;
      const std::uint32_t first_private = candidate.episode_begin < peer.episode_begin
          ? candidate_private : peer_private;
      bool shared_bridge = false;
      std::uint32_t second_bridge_anchor = second.episode_begin;
      // Prefer the shared resident unit nearest the episode boundary. Earlier
      // setup and later outcome clauses can repeat a topic without carrying
      // the transition that joins the two learned episodes.
      for (std::uint32_t left_cursor = first.episode_end;
           left_cursor > first.episode_begin && !shared_bridge; --left_cursor) {
        const std::uint32_t left = left_cursor - 1u;
        const std::uint32_t bridge = online_episode_units[left];
        if (unit_lengths[bridge] < 5u ||
            static_cast<unsigned long long>(unit_lengths[bridge]) * 16ull <
                vitality[bridge]) continue;
        for (std::uint32_t right = second.episode_begin;
             right < second.episode_end; ++right) {
          if (online_episode_units[right] == bridge) {
            shared_bridge = true;
            second_bridge_anchor = right;
            break;
          }
        }
      }
      std::uint32_t first_private_anchor = first.anchor_position;
      for (std::uint32_t position = first.episode_begin;
           position < first.episode_end; ++position) {
        if (position < scoped_episode_begin) continue;
        const std::uint32_t local = position - scoped_episode_begin;
        if (local >= scoped_episode_count) break;
        if ((scoped_episode_match_mask[local] & first_private) != 0u) {
          first_private_anchor = position;
          break;
        }
      }
      const std::uint32_t first_output_begin =
          first_private_anchor > first.episode_begin
              ? first_private_anchor - 1u : first.episode_begin;
      const std::uint32_t second_output_begin =
          second_bridge_anchor > second.episode_begin
              ? second_bridge_anchor - 1u : second.episode_begin;
      const std::uint32_t first_count = first.episode_end - first_output_begin;
      const std::uint32_t second_count = second.episode_end - second_output_begin;
      const std::uint32_t joined_count = first_count + second_count;
      if (!shared_bridge || joined_count < kCompositionMinUnits ||
          joined_count > kCompositionUnits) continue;
      std::uint32_t out = 0u;
      for (std::uint32_t position = first_output_begin;
           position < first.episode_end; ++position) {
        completion[out++] = online_episode_units[position];
      }
      for (std::uint32_t position = second_output_begin;
           position < second.episode_end; ++position) {
        completion[out++] = online_episode_units[position];
      }
      const unsigned long long primary_closure_score = resident_closure_score(
          boundary_histogram, boundary_pairs, boundary_bytes[0],
          closure_bytes[0]);
      const unsigned long long secondary_closure_score = resident_closure_score(
          boundary_histogram, boundary_pairs, boundary_bytes[0],
          closure_bytes[1]);
      if (out != 0u &&
          primary_closure_score > 2ull * max(1ull, secondary_closure_score)) {
        completion[0] = resident_episode_start_variant(
            unit_lengths, unit_content, unit_count, online_episode_units,
            online_episode_count, online_episode_breaks,
            online_episode_break_count, completion[0]);
      }
      motor_context[0] = 1u;
      motor_context[1] = completion[0];
      motor_context[2] = static_cast<std::uint32_t>(
          min(max(candidate.score, peer.score) >> 32u, 0xffffffffull));
      motor_context[3] = out;
      motor_context[4] = __popc(
          candidate.distinctive_cue_mask | peer.distinctive_cue_mask) - 1u;
      motor_context[5] = 4u;
      motor_context[12] = out;
      motor_context[13] = motor_context[4];
      novel_closure_done = 1u;
    }

    // A bounded learned episode can carry a causal chain in which the result
    // of one clause reappears as the condition of the next. Follow those
    // repeated resident units and omit the material between the two
    // occurrences. This composes a shorter path from ordinary learned matter;
    // it neither names a relation nor consults authored language structure.
    for (std::uint32_t rank = 1u;
         rank < join_scan && novel_closure_done == 0u; ++rank) {
      const LocalSeedCandidate peer = local_seed_candidates[rank];
      if (peer.score == 0u) break;
      if (peer.episode_begin != candidate.episode_begin ||
          peer.episode_end != candidate.episode_end) continue;
      const std::uint32_t candidate_private =
          candidate.distinctive_cue_mask & ~peer.distinctive_cue_mask;
      const std::uint32_t peer_private =
          peer.distinctive_cue_mask & ~candidate.distinctive_cue_mask;
      if (candidate_private == 0u || peer_private == 0u) continue;

      const LocalSeedCandidate first = candidate.anchor_position < peer.anchor_position
          ? candidate : peer;
      const std::uint32_t chain_begin = first.position > first.episode_begin
          ? first.position - 1u : first.position;
      const std::uint32_t chain_end = candidate.episode_end;
      if (chain_begin >= chain_end ||
          chain_end - chain_begin > 4u * kCompositionUnits ||
          chain_end - chain_begin < kCompositionMinUnits) continue;

      std::uint32_t out = 0u;
      std::uint32_t cursor = chain_begin;
      std::uint32_t bridge_jumps = 0u;
      while (cursor < chain_end && out < kCompositionUnits) {
        std::uint32_t bridge_begin = chain_end;
        std::uint32_t bridge_repeat = chain_end;
        std::uint32_t bridge_units = 0u;
        for (std::uint32_t left = cursor; left < chain_end; ++left) {
          const std::uint32_t bridge = online_episode_units[left];
          if (unit_lengths[bridge] < 5u) continue;
          for (std::uint32_t right = left + 1u; right < chain_end; ++right) {
            if (!same_resident_unit_bytes(
                    unit_lengths, unit_content, bridge,
                    online_episode_units[right])) continue;
            std::uint32_t run = 1u;
            while (left + run < right && right + run < chain_end &&
                   same_resident_unit_bytes(
                       unit_lengths, unit_content, online_episode_units[left + run],
                       online_episode_units[right + run])) {
              ++run;
            }
            bridge_begin = left;
            bridge_repeat = right;
            bridge_units = run;
            break;
          }
          if (bridge_units != 0u) break;
        }
        if (bridge_units == 0u) {
          while (cursor < chain_end && out < kCompositionUnits)
            completion[out++] = online_episode_units[cursor++];
          break;
        }
        std::uint32_t segment_end = bridge_begin + bridge_units;
        std::uint32_t closure_score = 0u;
        for (std::uint32_t position = segment_end;
             position < bridge_repeat; ++position) {
          const std::uint32_t unit = online_episode_units[position];
          const std::uint32_t length = unit_lengths[unit];
          if (length < 2u) continue;
          const std::uint32_t final_byte =
              resident_unit_byte(unit_content, unit, length - 1u);
          if (final_byte != boundary_bytes[0]) continue;
          const std::uint32_t prior_byte =
              resident_unit_byte(unit_content, unit, length - 2u);
          const std::uint32_t support = boundary_histogram[prior_byte];
          if (support == 0u) continue;
          const std::uint32_t score = static_cast<std::uint32_t>(min(
              static_cast<unsigned long long>(
                  boundary_pairs[prior_byte * 256u + final_byte]) * 65536ull /
                  support, 65535ull));
          if (score > closure_score) {
            closure_score = score;
            segment_end = position + 1u;
          }
        }
        while (cursor < segment_end && out < kCompositionUnits)
          completion[out++] = online_episode_units[cursor++];
        cursor = closure_score > 32768u ? bridge_repeat
                                        : bridge_repeat + bridge_units;
        ++bridge_jumps;
      }
      if (bridge_jumps == 0u || out < kCompositionMinUnits ||
          cursor < chain_end) continue;
      motor_context[0] = 1u;
      motor_context[1] = completion[0];
      motor_context[2] = static_cast<std::uint32_t>(
          min(max(candidate.score, peer.score) >> 32u, 0xffffffffull));
      motor_context[3] = out;
      motor_context[4] = __popc(
          candidate.distinctive_cue_mask | peer.distinctive_cue_mask) - 1u;
      motor_context[5] = 4u;
      motor_context[12] = out;
      motor_context[13] = bridge_jumps;
      novel_closure_done = 1u;
    }
    const std::uint32_t anchor = candidate.anchor_position;
    const std::uint32_t episode_end = min(online_episode_count, candidate.episode_end);
    std::uint32_t action = 0xffffffffu;
    const std::uint32_t later_cues = candidate.anchor_cue + 2u >= 32u
        ? 0u : ~((1u << (candidate.anchor_cue + 2u)) - 1u);
    if (novel_closure_done == 0u && motor_context[15] != 0u &&
        candidate.backward_launch == 0u) {
      for (std::uint32_t position = anchor + 2u;
           position < episode_end; ++position) {
        const std::uint32_t local = position - scoped_episode_begin;
        if (local >= scoped_episode_count) break;
        if ((scoped_episode_match_mask[local] & later_cues) != 0u) {
          action = position;
          break;
        }
      }
    }
    const bool bounded_episode_scope =
        scoped_episode_count <= 4u * kCompositionUnits;
    if (bounded_episode_scope && action != 0xffffffffu &&
        anchor + 1u < action && action >= 3u) {
      const std::uint32_t placeholder = action - 1u;
      const std::uint32_t entity = online_episode_units[anchor + 1u];
      const std::uint32_t placeholder_unit = online_episode_units[placeholder];
      const std::uint32_t cause_begin = anchor > candidate.episode_begin + 3u
          ? anchor - 3u : candidate.episode_begin;
      const std::uint32_t projected = (placeholder - cause_begin) + 2u +
          (episode_end - action);
      const bool replaceable = unit_lengths[placeholder_unit] <= 4u &&
          placeholder_unit != entity && vitality[placeholder_unit] > vitality[entity];
      if (replaceable && projected >= kCompositionMinUnits &&
          projected <= kCompositionUnits) {
        std::uint32_t out = 0u;
        for (std::uint32_t position = cause_begin; position < placeholder; ++position) {
          completion[out++] = online_episode_units[position];
        }
        completion[out++] = online_episode_units[anchor];
        completion[out++] = entity;
        for (std::uint32_t position = action; position < episode_end; ++position) {
          completion[out++] = online_episode_units[position];
        }
        motor_context[0] = 1u;
        motor_context[1] = completion[0];
        motor_context[2] = static_cast<std::uint32_t>(
            min(candidate.score >> 32u, 0xffffffffull));
        motor_context[3] = out;
        motor_context[4] = __popc(candidate.cue_mask) - 1u;
        motor_context[5] = 4u;
        motor_context[12] = out;
        motor_context[13] = __popc(candidate.cue_mask) - 1u;
        novel_closure_done = 1u;
      }
    }
    if (bounded_episode_scope && novel_closure_done == 0u &&
        motor_context[15] != 0u) {
      const std::uint32_t completion_begin = candidate.episode_begin;
      const std::uint32_t completion_count = episode_end - completion_begin;
      if (completion_count >= kCompositionMinUnits &&
          completion_count <= kCompositionUnits) {
        for (std::uint32_t i = 0u; i < completion_count; ++i) {
          completion[i] = online_episode_units[completion_begin + i];
        }
        motor_context[0] = 1u;
        motor_context[1] = completion[0];
        motor_context[2] = static_cast<std::uint32_t>(
            min(candidate.score >> 32u, 0xffffffffull));
        motor_context[3] = completion_count;
        motor_context[4] = __popc(candidate.cue_mask) - 1u;
        motor_context[5] = 4u;
        motor_context[12] = completion_count;
        motor_context[13] = __popc(candidate.cue_mask) - 1u;
        novel_closure_done = 1u;
      }
    }
  }
  __syncthreads();
  if (novel_closure_done != 0u) return;
  if (motor_context[15] == 0u) return;


  if (threadIdx.x == 0u) {
    beam_count = 0u;
    beam_length = 1u;
    const std::uint32_t available = min(local_seed_candidate_count,
                                        kCompositionBeamWidth);
    for (std::uint32_t selected = 0u; selected < available; ++selected) {
      const LocalSeedCandidate candidate = local_seed_candidates[selected];
      if (candidate.score == 0u) break;
      const std::uint32_t out = beam_count++;
      beam_units[out * kCompositionMaxUnits] = candidate.position;
      beam_scores[out] = candidate.score;
      beam_anchors[out] = candidate.cue_mask;
      beam_last_anchor[out] = candidate.position;
      beam_target_anchor[out] = candidate.target;
      beam_launch_end[out] = candidate.local_launch_end;
    }
  }
  __syncthreads();

  if (threadIdx.x < beam_count) {
    const std::uint32_t beam = threadIdx.x;
    const std::uint32_t seed_position = beam_units[beam * kCompositionMaxUnits];
    const std::uint32_t target = beam_target_anchor[beam];
    const std::uint32_t first = online_episode_units[seed_position];
    const std::uint32_t second = online_episode_units[seed_position + 1u];
    const std::uint32_t third = online_episode_units[seed_position + 2u];
    const std::uint32_t grammar = resident_trigram_count(
        online_trigrams, online_trigram_counts, online_trigram_count,
        first, second, third);
    next_scores[beam] = grammar == 0u ? 0u : beam_scores[beam] +
        static_cast<unsigned long long>(16u + 8u * evidence_depth(grammar)) *
            0x100000000ull + grammar;
    if (grammar != 0u) {
      next_units[beam * kCompositionMaxUnits] = first;
      next_units[beam * kCompositionMaxUnits + 1u] = second;
      next_units[beam * kCompositionMaxUnits + 2u] = third;
      next_anchors[beam] = beam_anchors[beam];
      next_last_anchor[beam] = beam_last_anchor[beam];
      next_target_anchor[beam] = target;
      next_launch_end[beam] = beam_launch_end[beam];
    }
  }
  __syncthreads();
  if (threadIdx.x == 0u) {
    std::uint32_t initialized = 0u;
    for (std::uint32_t beam = 0u; beam < beam_count; ++beam) {
      if (next_scores[beam] == 0u) continue;
      for (std::uint32_t i = 0u; i < 3u; ++i) {
        beam_units[initialized * kCompositionMaxUnits + i] =
            next_units[beam * kCompositionMaxUnits + i];
      }
      beam_scores[initialized] = next_scores[beam];
      beam_anchors[initialized] = next_anchors[beam];
      beam_last_anchor[initialized] = next_last_anchor[beam];
      beam_target_anchor[initialized] = next_target_anchor[beam];
      beam_launch_end[initialized] = next_launch_end[beam];
      ++initialized;
    }
    beam_count = initialized;
    beam_length = 3u;
  }
  __syncthreads();

  for (std::uint32_t depth = 3u; depth < kCompositionMaxUnits && beam_count != 0u; ++depth) {
    for (std::uint32_t i = 0u; i < kCompositionBeamExpansions; ++i) {
      candidate_ids[local_begin + i] = 0xffffffffu;
      candidate_parents[local_begin + i] = 0xffffffffu;
      candidate_anchors[local_begin + i] = 0u;
      candidate_salient[local_begin + i] = 0u;
      candidate_targets[local_begin + i] = 0xffffffffu;
      candidate_scores[local_begin + i] = 0u;
    }
    __syncthreads();

    if (threadIdx.x < beam_count) {
      const std::uint32_t parent = threadIdx.x;
      const std::uint32_t source_state = beam_last_anchor[parent];
      const std::uint32_t source_position = source_state & ~kCompositionSplicedBit;
      const bool already_spliced = (source_state & kCompositionSplicedBit) != 0u;
      const bool local_launch = !already_spliced &&
          source_position + depth < beam_launch_end[parent];
      const bool local_available = source_position + depth < online_episode_count;
      const std::uint32_t local_next = local_available
          ? online_episode_units[source_position + depth] : 0xffffffffu;
      const std::uint32_t previous = beam_units[parent * kCompositionMaxUnits + depth - 1u];
      const std::uint32_t first = depth >= 2u
          ? beam_units[parent * kCompositionMaxUnits + depth - 2u] : 0xffffffffu;
      const std::uint32_t slow_begin = depth >= 2u
          ? lower_trigram(base_trigrams, base_trigram_count, first, previous)
          : lower_bigram(base_bigrams, base_bigram_count, previous);
      const std::uint32_t slow_end = depth >= 2u
          ? upper_trigram(base_trigrams, base_trigram_count, first, previous)
          : upper_bigram(base_bigrams, base_bigram_count, previous);
      const std::uint32_t fast_begin = depth >= 2u
          ? lower_trigram(online_trigrams, online_trigram_count, first, previous)
          : lower_bigram(online_bigrams, online_bigram_count, previous);
      const std::uint32_t fast_end = depth >= 2u
          ? upper_trigram(online_trigrams, online_trigram_count, first, previous)
          : upper_bigram(online_bigrams, online_bigram_count, previous);

      const std::uint32_t source_begin = fast_begin < fast_end ? 1u : 0u;
      for (std::uint32_t source = source_begin; source < 2u; ++source) {
        const std::uint32_t begin = source == 0u ? slow_begin : fast_begin;
        const std::uint32_t end = source == 0u ? slow_end : fast_end;
        for (std::uint32_t edge = begin; edge < end; ++edge) {
          const std::uint32_t next = depth >= 2u
              ? (source == 0u ? base_trigrams[edge].next : online_trigrams[edge].next)
              : (source == 0u ? base_bigrams[edge].next : online_bigrams[edge].next);
          if (local_launch &&
              (!local_available || next != local_next)) continue;
          std::uint32_t repeats = 0u;
          for (std::uint32_t i = 0u; i < depth; ++i) {
            repeats += beam_units[parent * kCompositionMaxUnits + i] == next;
          }

          const std::uint32_t slow_count = depth >= 2u
              ? resident_trigram_count(base_trigrams, base_trigram_counts,
                                        base_trigram_count, first, previous, next)
              : resident_bigram_count(base_bigrams, base_bigram_counts,
                                      base_bigram_count, previous, next);
          const std::uint32_t fast_count = depth >= 2u
              ? resident_trigram_count(online_trigrams, online_trigram_counts,
                                        online_trigram_count, first, previous, next)
              : resident_bigram_count(online_bigrams, online_bigram_counts,
                                      online_bigram_count, previous, next);
          if (slow_count == 0u && fast_count == 0u) continue;
          const std::uint32_t guide = resident_association_count(
              associations, association_counts, association_count,
              beam_target_anchor[parent], next) + resident_association_count(
              associations, association_counts, association_count,
              next, beam_target_anchor[parent]);
          const std::uint32_t grammar = 4u * evidence_depth(slow_count) +
                                        6u * evidence_depth(fast_count);
          const std::uint32_t common_penalty = min(12u, evidence_depth(vitality[next])) +
              min(8u, association_mass_depth(association_mass[next] / 1024ull));
          const std::uint32_t echo_penalty = cue_scores[next] >= kCueNearIdentity ? 2u : 0u;
          const std::uint32_t local_bonus = local_launch &&
              local_available && next == local_next ? 48u : 0u;
          const std::uint32_t reward =
              24u + grammar + 8u * evidence_depth(guide) + local_bonus;
          const std::uint32_t repetition_penalty = min(64u, 24u * repeats);
          const std::uint32_t source_run_penalty =
              source_run_limit < kCompositionSourceRunLimit && !local_launch &&
              !already_spliced && local_available && next == local_next ? 96u : 0u;
          const std::uint32_t penalty = common_penalty + echo_penalty +
              repetition_penalty + source_run_penalty;
          const unsigned long long increment = static_cast<unsigned long long>(
              reward > penalty ? reward - penalty : 1u) * 0x100000000ull +
              slow_count + fast_count;
          const bool spliced = already_spliced ||
              (depth >= kCompositionSemanticLaunchUnits &&
               (!local_available || next != local_next));
          const std::uint32_t next_source_state = source_position |
              (spliced ? kCompositionSplicedBit : 0u);
          insert_candidate(local_begin, next, parent,
                           beam_anchors[parent], next_source_state,
                           beam_target_anchor[parent],
                           beam_scores[parent] + increment);
        }
      }
    }
    __syncthreads();

    if (threadIdx.x == 0u) {
      std::uint32_t next_count = 0u;
      for (std::uint32_t selected = 0u; selected < kCompositionBeamWidth; ++selected) {
        std::uint32_t best = 0xffffffffu;
        for (std::uint32_t candidate = 0u; candidate < candidate_count; ++candidate) {
          if (candidate_scores[candidate] == 0u) continue;
          bool duplicate = false;
          for (std::uint32_t prior = 0u; prior < next_count; ++prior) {
            duplicate |= candidate_parents[candidate] ==
                             candidate_parents[selected_candidates[prior]] &&
                         candidate_ids[candidate] == candidate_ids[selected_candidates[prior]];
          }
          if (duplicate) continue;
          if (best == 0xffffffffu || candidate_scores[candidate] > candidate_scores[best] ||
              (candidate_scores[candidate] == candidate_scores[best] &&
               candidate_ids[candidate] < candidate_ids[best])) best = candidate;
        }
        if (best == 0xffffffffu) break;
        const std::uint32_t parent = candidate_parents[best];
        for (std::uint32_t i = 0u; i < depth; ++i) {
          next_units[next_count * kCompositionMaxUnits + i] =
              beam_units[parent * kCompositionMaxUnits + i];
        }
        next_units[next_count * kCompositionMaxUnits + depth] = candidate_ids[best];
        next_scores[next_count] = candidate_scores[best];
        next_anchors[next_count] = candidate_anchors[best];
        next_last_anchor[next_count] = candidate_salient[best];
        next_target_anchor[next_count] = candidate_targets[best];
        next_launch_end[next_count] = beam_launch_end[parent];
        selected_candidates[next_count] = best;
        candidate_scores[best] = 0u;
        ++next_count;
      }
      beam_count = next_count;
      beam_length = depth + 1u;
      for (std::uint32_t beam = 0u; beam < beam_count; ++beam) {
        for (std::uint32_t i = 0u; i < beam_length; ++i) {
          beam_units[beam * kCompositionMaxUnits + i] =
              next_units[beam * kCompositionMaxUnits + i];
        }
        beam_scores[beam] = next_scores[beam];
        beam_anchors[beam] = next_anchors[beam];
        beam_last_anchor[beam] = next_last_anchor[beam];
        beam_target_anchor[beam] = next_target_anchor[beam];
        beam_launch_end[beam] = next_launch_end[beam];
      }
    }
    __syncthreads();
  }

  if (threadIdx.x < beam_count) beam_copied[threadIdx.x] = 0u;
  __syncthreads();
  for (std::uint32_t beam = 0u; beam < beam_count; ++beam) {
    for (std::uint32_t position = threadIdx.x;
         position + source_run_limit <= base_episode_count;
         position += blockDim.x) {
      std::uint32_t matched = 0u;
      while (matched < min(beam_length, source_run_limit) &&
             base_episode_units[position + matched] ==
             beam_units[beam * kCompositionMaxUnits + matched]) ++matched;
      if (matched == min(beam_length, source_run_limit))
        atomicExch(beam_copied + beam, 1u);
    }
    for (std::uint32_t position = threadIdx.x;
         position + source_run_limit <= online_episode_count;
         position += blockDim.x) {
      std::uint32_t matched = 0u;
      while (matched < min(beam_length, source_run_limit) &&
             online_episode_units[position + matched] ==
             beam_units[beam * kCompositionMaxUnits + matched]) ++matched;
      if (matched == min(beam_length, source_run_limit))
        atomicExch(beam_copied + beam, 1u);
    }
  }
  __syncthreads();

  if (threadIdx.x == 0u) {
    std::uint32_t winner = 0xffffffffu;
    bool winner_closed = false;
    for (std::uint32_t beam = 0u; beam < beam_count; ++beam) {
      if (beam_length < kCompositionMinUnits || __popc(beam_anchors[beam]) < 2u ||
          (beam_last_anchor[beam] & kCompositionSplicedBit) == 0u) continue;
      std::uint32_t cue_overlap = 0u;
      for (std::uint32_t i = 0u; i < beam_length; ++i) {
        cue_overlap += cue_scores[beam_units[beam * kCompositionMaxUnits + i]] >=
            kCueNearIdentity;
      }
      if (cue_overlap * 4u >= beam_length * 3u) continue;
      if (beam_copied[beam] != 0u) continue;
      bool beam_closed = false;
      for (std::uint32_t i = kCompositionMinUnits - 1u;
           i < beam_length; ++i) {
        beam_closed |= resident_unit_contains_any(
            unit_lengths, unit_content,
            beam_units[beam * kCompositionMaxUnits + i],
            closure_bytes, kClosureCount);
      }
      if (winner == 0xffffffffu || (beam_closed && !winner_closed) ||
          (beam_closed == winner_closed && beam_scores[beam] > beam_scores[winner]) ||
          (beam_closed == winner_closed && beam_scores[beam] == beam_scores[winner] &&
           __popc(beam_anchors[beam]) > __popc(beam_anchors[winner]))) {
        winner = beam;
        winner_closed = beam_closed;
      }
    }
    if (winner != 0xffffffffu) {
      std::uint32_t emit_length = beam_length;
      std::uint32_t boundary_run = 0u;
      for (std::uint32_t i = kCompositionMinUnits - 1u; i < beam_length; ++i) {
        const std::uint32_t unit =
            beam_units[winner * kCompositionMaxUnits + i];
        bool boundary_only = unit_lengths[unit] != 0u;
        for (std::uint32_t offset = 0u; offset < unit_lengths[unit]; ++offset) {
          boundary_only &= resident_unit_byte(unit_content, unit, offset) ==
              boundary_bytes[0];
        }
        boundary_run = boundary_only ? boundary_run + 1u : 0u;
        if (boundary_run >= 2u) {
          emit_length = i + 1u - boundary_run;
          break;
        }
        if (resident_unit_contains_any(
                unit_lengths, unit_content, unit,
                closure_bytes, kClosureCount)) {
          emit_length = i + 1u;
          break;
        }
      }
      while (emit_length > kCompositionMinUnits) {
        const std::uint32_t unit =
            beam_units[winner * kCompositionMaxUnits + emit_length - 1u];
        bool boundary_only = unit_lengths[unit] != 0u;
        for (std::uint32_t offset = 0u; offset < unit_lengths[unit]; ++offset) {
          boundary_only &= resident_unit_byte(unit_content, unit, offset) ==
              boundary_bytes[0];
        }
        if (!boundary_only) break;
        --emit_length;
      }
      for (std::uint32_t i = 0u; i < emit_length; ++i) {
        completion[i] = beam_units[winner * kCompositionMaxUnits + i];
      }
      motor_context[0] = 1u;
      motor_context[1] = completion[0];
      motor_context[2] = static_cast<std::uint32_t>(
          min(beam_scores[winner] >> 32u, 0xffffffffull));
      motor_context[3] = emit_length;
      motor_context[4] = __popc(beam_anchors[winner]) - 1u;
      motor_context[5] = 4u;
      motor_context[12] = emit_length;
      motor_context[13] = __popc(beam_anchors[winner]) - 1u;
    } else {
      motor_context[12] = beam_length;
      motor_context[13] = 0u;
      motor_context[15] = 0u;
    }
  }
}
