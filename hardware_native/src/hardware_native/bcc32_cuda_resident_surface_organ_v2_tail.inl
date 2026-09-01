__global__ void realize_construction_witness_kernel(
    SurfaceUnitView units, OpaqueContentPlanView plan,
    OpaqueConstructionWitnessView witness,
    SurfaceRealizationWorkspaceView workspace) {
  if (blockIdx.x != 0u || threadIdx.x != 0u)
    return;
  realize_construction_witness(units, plan, witness, workspace);
}

__device__ inline void surface_emit_anchor_fallback(SurfaceUnitView units,
                                                    OpaqueContentPlanView plan,
                                                    SurfaceRealizationWorkspaceView workspace,
                                                    SurfaceOrganResult* result) {
  surface_clear_output(workspace);
  std::uint32_t count = 0u;
  for (std::uint32_t index = 0u; index < plan.anchor_count; ++index) {
    const std::uint32_t unit = plan.anchor_units[index];
    if (unit >= units.unit_count || units.lengths[unit] == 0u ||
        !surface_append_unit(workspace, unit, true, &count)) {
      result->capacity_exceeded = 1u;
      break;
    }
    ++result->anchors_preserved;
  }
  std::uint32_t byte_count = 0u;
  if (!surface_emit_bytes(units, workspace, count, &byte_count))
    result->capacity_exceeded = 1u;
  result->output_unit_count = count;
  result->output_byte_count = byte_count;
  result->ready =
      result->capacity_exceeded == 0u && result->anchors_preserved == plan.anchor_count ? 1u : 0u;
}

[[nodiscard]] __device__ inline bool surface_build_greedy_anchor_order(
    SurfaceUnitView units, OpaqueContentPlanView plan, const SurfaceRolePathChoice* bridges,
    const SurfaceRolePathChoice* prefixes, const SurfaceRolePathChoice* suffixes,
    std::uint32_t* order, std::uint64_t* path_quality) {
  bool used[kSurfaceOrganMaxAnchors]{};
  std::uint32_t first = kSurfaceOrganNoUnit;
  std::uint32_t first_quality = 0u;
  std::uint32_t first_length = kSurfaceOrganMaxBridgeRoles + 1u;
  for (std::uint32_t candidate = 0u; candidate < plan.anchor_count; ++candidate) {
    const std::uint32_t unit = plan.anchor_units[candidate];
    if (unit >= units.unit_count)
      continue;
    const std::uint32_t role = units.roles[unit].role;
    if (role >= kSurfaceOrganRoleCount || prefixes[role].valid == 0u)
      continue;
    const std::uint32_t quality = prefixes[role].quality_q20;
    if (first == kSurfaceOrganNoUnit || prefixes[role].length < first_length ||
        (prefixes[role].length == first_length && quality > first_quality) ||
        (prefixes[role].length == first_length && quality == first_quality && candidate < first)) {
      first = candidate;
      first_quality = quality;
      first_length = prefixes[role].length;
    }
  }
  if (first == kSurfaceOrganNoUnit)
    return false;
  order[0] = first;
  used[first] = true;
  *path_quality = first_quality;

  for (std::uint32_t position = 1u; position < plan.anchor_count; ++position) {
    const std::uint32_t previous_unit = plan.anchor_units[order[position - 1u]];
    const std::uint32_t previous_role = units.roles[previous_unit].role;
    std::uint32_t winner = kSurfaceOrganNoUnit;
    std::uint32_t winner_quality = 0u;
    std::uint32_t winner_length = kSurfaceOrganMaxBridgeRoles + 1u;
    for (std::uint32_t candidate = 0u; candidate < plan.anchor_count; ++candidate) {
      if (used[candidate])
        continue;
      const std::uint32_t unit = plan.anchor_units[candidate];
      if (unit >= units.unit_count)
        continue;
      const std::uint32_t role = units.roles[unit].role;
      if (role >= kSurfaceOrganRoleCount)
        continue;
      const SurfaceRolePathChoice bridge = bridges[previous_role * kSurfaceOrganRoleCount + role];
      if (bridge.valid == 0u || (position + 1u == plan.anchor_count && suffixes[role].valid == 0u))
        continue;
      const std::uint32_t quality =
          bridge.quality_q20 +
          (position + 1u == plan.anchor_count ? suffixes[role].quality_q20 : 0u);
      if (winner == kSurfaceOrganNoUnit || bridge.length < winner_length ||
          (bridge.length == winner_length && quality > winner_quality) ||
          (bridge.length == winner_length && quality == winner_quality && candidate < winner)) {
        winner = candidate;
        winner_quality = quality;
        winner_length = bridge.length;
      }
    }
    if (winner == kSurfaceOrganNoUnit)
      return false;
    order[position] = winner;
    used[winner] = true;
    *path_quality += winner_quality;
  }
  return true;
}

__global__ void select_and_realize_surface_kernel(
    SurfaceUnitView units, MutableSurfaceGrammarEvidenceView evidence, OpaqueContentPlanView plan,
    const SurfaceRolePathChoice* bridges, const SurfaceRolePathChoice* prefixes,
    const SurfaceRolePathChoice* suffixes, const std::uint64_t* permutation_scores,
    const std::uint32_t* permutation_valid, bool force_greedy,
    SurfaceRealizationWorkspaceView workspace) {
  if (blockIdx.x != 0u || threadIdx.x != 0u)
    return;
  *workspace.result = {};
  surface_clear_output(workspace);

  const bool exhaustive = !force_greedy &&
                          plan.anchor_count <= kSurfaceOrganMaxExhaustiveAnchors;
  const std::uint32_t permutation_count = exhaustive ? surface_factorial(plan.anchor_count) : 0u;
  std::uint32_t winner = kSurfaceOrganNoUnit;
  std::uint64_t winner_score = 0u;
  for (std::uint32_t index = 0u; index < permutation_count; ++index) {
    if (permutation_valid[index] == 0u)
      continue;
    if (winner == kSurfaceOrganNoUnit || permutation_scores[index] > winner_score ||
        (permutation_scores[index] == winner_score && index < winner)) {
      winner = index;
      winner_score = permutation_scores[index];
    }
  }
  std::uint32_t permutation[kSurfaceOrganMaxAnchors]{};
  if (exhaustive) {
    if (winner == kSurfaceOrganNoUnit) {
      surface_emit_anchor_fallback(units, plan, workspace, workspace.result);
      return;
    }
    surface_decode_permutation(winner, plan.anchor_count, permutation);
  } else if (!surface_build_greedy_anchor_order(units, plan, bridges, prefixes, suffixes,
                                                 permutation, &winner_score)) {
    surface_emit_anchor_fallback(units, plan, workspace, workspace.result);
    return;
  }
  std::uint32_t anchor_roles[kSurfaceOrganMaxAnchors]{};
  for (std::uint32_t index = 0u; index < plan.anchor_count; ++index)
    anchor_roles[index] = units.roles[plan.anchor_units[permutation[index]]].role;

  std::uint32_t output_count = 0u;
  bool realized = surface_append_role_path(
      units, evidence, plan, workspace, prefixes[anchor_roles[0]], SurfaceUnitCompetition::kStart,
      SurfaceUnitCompetition::kInterior, plan.anchor_units[permutation[0]], &output_count);
  for (std::uint32_t position = 0u; realized && position < plan.anchor_count; ++position) {
    const std::uint32_t anchor = plan.anchor_units[permutation[position]];
    realized = surface_append_unit(workspace, anchor, true, &output_count);
    if (realized)
      ++workspace.result->anchors_preserved;
    if (realized && position + 1u < plan.anchor_count) {
      const SurfaceRolePathChoice bridge =
          bridges[anchor_roles[position] * kSurfaceOrganRoleCount + anchor_roles[position + 1u]];
      realized = surface_append_role_path(units, evidence, plan, workspace, bridge,
                                          SurfaceUnitCompetition::kInterior,
                                          SurfaceUnitCompetition::kInterior,
                                          plan.anchor_units[permutation[position + 1u]],
                                          &output_count);
    }
  }
  if (realized) {
    realized = surface_append_role_path(
        units, evidence, plan, workspace, suffixes[anchor_roles[plan.anchor_count - 1u]],
        SurfaceUnitCompetition::kInterior, SurfaceUnitCompetition::kEnd, kSurfaceOrganNoUnit,
        &output_count);
  }
  if (!realized) {
    *workspace.result = {};
    surface_emit_anchor_fallback(units, plan, workspace, workspace.result);
    return;
  }

  std::uint32_t byte_count = 0u;
  if (!surface_emit_bytes(units, workspace, output_count, &byte_count)) {
    *workspace.result = {};
    workspace.result->capacity_exceeded = 1u;
    return;
  }
  std::uint32_t connector_count = 0u;
  for (std::uint32_t index = 0u; index < output_count; ++index)
    connector_count += workspace.output_anchor_mask[index] == 0u ? 1u : 0u;
  bool reordered = false;
  for (std::uint32_t index = 0u; index < plan.anchor_count; ++index)
    reordered = reordered || permutation[index] != index;

  workspace.result->ready = 1u;
  workspace.result->grammar_supported = 1u;
  workspace.result->closure_supported = 1u;
  workspace.result->output_unit_count = output_count;
  workspace.result->output_byte_count = byte_count;
  workspace.result->connector_count = connector_count;
  workspace.result->plan_reordered = reordered ? 1u : 0u;
  workspace.result->selected_permutation = exhaustive ? winner : kSurfaceOrganNoUnit;
  workspace.result->path_quality_q20 =
      static_cast<std::uint32_t>(winner_score > 0xffffffffull ? 0xffffffffull : winner_score);
}

template <typename BigramKey, typename TrigramKey,
          typename Access = resident_roles::DefaultNgramAccess>
__global__ void select_and_realize_conditioned_surface_kernel(
    SurfaceUnitView units, MutableSurfaceGrammarEvidenceView evidence,
    OpaqueContentPlanView plan,
    SurfaceSequenceEvidenceView<BigramKey, TrigramKey> sequence,
    const SurfaceRolePathChoice* bridges,
    const SurfaceRolePathChoice* prefixes,
    const SurfaceRolePathChoice* suffixes,
    const std::uint64_t* permutation_scores,
    const std::uint32_t* permutation_valid, bool force_greedy,
    SurfaceRealizationWorkspaceView workspace) {
  if (blockIdx.x != 0u || threadIdx.x != 0u)
    return;
  *workspace.result = {};
  surface_clear_output(workspace);

  const bool exhaustive =
      !force_greedy &&
      plan.anchor_count <= kSurfaceOrganMaxExhaustiveAnchors;
  const std::uint32_t permutation_count =
      exhaustive ? surface_factorial(plan.anchor_count) : 0u;
  std::uint32_t winner = kSurfaceOrganNoUnit;
  std::uint64_t winner_score = 0u;
  for (std::uint32_t index = 0u; index < permutation_count; ++index) {
    if (permutation_valid[index] == 0u)
      continue;
    if (winner == kSurfaceOrganNoUnit ||
        permutation_scores[index] > winner_score ||
        (permutation_scores[index] == winner_score && index < winner)) {
      winner = index;
      winner_score = permutation_scores[index];
    }
  }
  std::uint32_t permutation[kSurfaceOrganMaxAnchors]{};
  if (exhaustive) {
    if (winner == kSurfaceOrganNoUnit) {
      return;
    }
    surface_decode_permutation(winner, plan.anchor_count, permutation);
  } else if (!surface_build_greedy_anchor_order(
                  units, plan, bridges, prefixes, suffixes, permutation,
                  &winner_score)) {
    return;
  }

  std::uint32_t anchor_roles[kSurfaceOrganMaxAnchors]{};
  for (std::uint32_t index = 0u; index < plan.anchor_count; ++index)
    anchor_roles[index] =
        units.roles[plan.anchor_units[permutation[index]]].role;

  const std::uint32_t first_anchor = plan.anchor_units[permutation[0]];
  std::uint32_t output_count = 0u;
  bool realized = surface_append_contextual_role_path<
      BigramKey, TrigramKey, Access>(
      units, evidence, plan, sequence, workspace,
      prefixes[anchor_roles[0]], kSurfaceOrganNoUnit, first_anchor,
      &output_count);
  for (std::uint32_t position = 0u;
       realized && position < plan.anchor_count; ++position) {
    const std::uint32_t anchor = plan.anchor_units[permutation[position]];
    realized = surface_append_unit(workspace, anchor, true, &output_count);
    if (realized)
      ++workspace.result->anchors_preserved;
    if (realized && position + 1u < plan.anchor_count) {
      const std::uint32_t next_anchor =
          plan.anchor_units[permutation[position + 1u]];
      const SurfaceRolePathChoice bridge =
          bridges[anchor_roles[position] * kSurfaceOrganRoleCount +
                  anchor_roles[position + 1u]];
      realized = surface_append_contextual_role_path<
          BigramKey, TrigramKey, Access>(
          units, evidence, plan, sequence, workspace, bridge, anchor,
          next_anchor, &output_count);
    }
  }
  if (realized) {
    const std::uint32_t last_anchor =
        plan.anchor_units[permutation[plan.anchor_count - 1u]];
    realized = surface_append_contextual_role_path<
        BigramKey, TrigramKey, Access>(
        units, evidence, plan, sequence, workspace,
        suffixes[anchor_roles[plan.anchor_count - 1u]], last_anchor,
        kSurfaceOrganNoUnit, &output_count);
  }
  if (!realized) {
    *workspace.result = {};
    return;
  }

  std::uint32_t byte_count = 0u;
  if (!surface_emit_bytes(units, workspace, output_count, &byte_count)) {
    *workspace.result = {};
    workspace.result->capacity_exceeded = 1u;
    return;
  }
  std::uint32_t connector_count = 0u;
  for (std::uint32_t index = 0u; index < output_count; ++index)
    connector_count += workspace.output_anchor_mask[index] == 0u ? 1u : 0u;
  bool reordered = false;
  for (std::uint32_t index = 0u; index < plan.anchor_count; ++index)
    reordered = reordered || permutation[index] != index;

  workspace.result->ready = 1u;
  workspace.result->grammar_supported = 1u;
  workspace.result->closure_supported = 1u;
  workspace.result->output_unit_count = output_count;
  workspace.result->output_byte_count = byte_count;
  workspace.result->connector_count = connector_count;
  workspace.result->plan_reordered = reordered ? 1u : 0u;
  workspace.result->selected_permutation =
      exhaustive ? winner : kSurfaceOrganNoUnit;
  workspace.result->path_quality_q20 = static_cast<std::uint32_t>(
      winner_score > 0xffffffffull ? 0xffffffffull : winner_score);
}

[[nodiscard]] inline bool valid_surface_grammar_evidence(
    MutableSurfaceGrammarEvidenceView evidence) {
  return evidence.unit_capacity != 0u && evidence.unit_mass != nullptr &&
         evidence.unit_start_mass != nullptr && evidence.unit_end_mass != nullptr &&
         evidence.role_mass != nullptr && evidence.role_start_mass != nullptr &&
         evidence.role_end_mass != nullptr && evidence.role_bigram_mass != nullptr &&
         evidence.role_bigram_context_mass != nullptr && evidence.role_trigram_mass != nullptr &&
         evidence.role_trigram_context_mass != nullptr && evidence.stats != nullptr;
}

[[nodiscard]] inline cudaError_t clear_surface_grammar_evidence_cuda(
    MutableSurfaceGrammarEvidenceView evidence, cudaStream_t stream = nullptr) {
  if (!valid_surface_grammar_evidence(evidence))
    return cudaErrorInvalidValue;

  cudaError_t status = cudaMemsetAsync(evidence.unit_mass, 0,
                                       evidence.unit_capacity * sizeof(std::uint64_t), stream);
  if (status != cudaSuccess)
    return status;
  status = cudaMemsetAsync(evidence.unit_start_mass, 0,
                           evidence.unit_capacity * sizeof(std::uint64_t), stream);
  if (status != cudaSuccess)
    return status;
  status = cudaMemsetAsync(evidence.unit_end_mass, 0,
                           evidence.unit_capacity * sizeof(std::uint64_t), stream);
  if (status != cudaSuccess)
    return status;
  status = cudaMemsetAsync(evidence.role_mass, 0, kSurfaceOrganRoleCount * sizeof(std::uint64_t),
                           stream);
  if (status != cudaSuccess)
    return status;
  status = cudaMemsetAsync(evidence.role_start_mass, 0,
                           kSurfaceOrganRoleCount * sizeof(std::uint64_t), stream);
  if (status != cudaSuccess)
    return status;
  status = cudaMemsetAsync(evidence.role_end_mass, 0,
                           kSurfaceOrganRoleCount * sizeof(std::uint64_t), stream);
  if (status != cudaSuccess)
    return status;
  status = cudaMemsetAsync(evidence.role_bigram_mass, 0,
                           kSurfaceOrganRoleBigramCount * sizeof(std::uint64_t), stream);
  if (status != cudaSuccess)
    return status;
  status = cudaMemsetAsync(evidence.role_bigram_context_mass, 0,
                           kSurfaceOrganRoleCount * sizeof(std::uint64_t), stream);
  if (status != cudaSuccess)
    return status;
  status = cudaMemsetAsync(evidence.role_trigram_mass, 0,
                           kSurfaceOrganRoleTrigramCount * sizeof(std::uint64_t), stream);
  if (status != cudaSuccess)
    return status;
  status = cudaMemsetAsync(evidence.role_trigram_context_mass, 0,
                           kSurfaceOrganRoleBigramCount * sizeof(std::uint64_t), stream);
  if (status != cudaSuccess)
    return status;
  return cudaMemsetAsync(evidence.stats, 0, sizeof(SurfaceGrammarStats), stream);
}

[[nodiscard]] inline cudaError_t learn_surface_grammar_cuda(
    SurfaceUnitView units, SurfaceGrammarBatchView batch,
    MutableSurfaceGrammarEvidenceView evidence, SurfaceLearningWorkspaceView workspace,
    cudaStream_t stream = nullptr) {
  if (units.unit_count == 0u || units.lengths == nullptr || units.packed_bytes == nullptr ||
      units.roles == nullptr || units.unit_words == 0u ||
      !valid_surface_grammar_evidence(evidence) || evidence.unit_capacity < units.unit_count ||
      batch.units == nullptr || batch.episode_offsets == nullptr ||
      batch.document_episode_offsets == nullptr || batch.episode_count == 0u ||
      batch.document_count == 0u || workspace.document_capacity < batch.document_count ||
      workspace.document_unit_events == nullptr || workspace.document_bigram_events == nullptr ||
      workspace.document_trigram_events == nullptr)
    return cudaErrorInvalidValue;
  cudaError_t status = cudaMemsetAsync(workspace.document_unit_events, 0,
                                       batch.document_count * sizeof(std::uint64_t), stream);
  if (status != cudaSuccess)
    return status;
  status = cudaMemsetAsync(workspace.document_bigram_events, 0,
                           batch.document_count * sizeof(std::uint64_t), stream);
  if (status != cudaSuccess)
    return status;
  status = cudaMemsetAsync(workspace.document_trigram_events, 0,
                           batch.document_count * sizeof(std::uint64_t), stream);
  if (status != cudaSuccess)
    return status;

  const std::uint32_t blocks = surface_organ_blocks(batch.episode_count);
  count_surface_document_events_kernel<<<blocks, kSurfaceOrganBlockSize, 0, stream>>>(batch,
                                                                                      workspace);
  status = cudaPeekAtLastError();
  if (status != cudaSuccess)
    return status;
  accumulate_surface_grammar_kernel<<<blocks, kSurfaceOrganBlockSize, 0, stream>>>(
      units, batch, evidence, workspace);
  status = cudaPeekAtLastError();
  if (status != cudaSuccess)
    return status;
  finish_surface_grammar_batch_kernel<<<1u, 1u, 0, stream>>>(batch, evidence.stats);
  return cudaPeekAtLastError();
}

template <typename BigramKey, typename TrigramKey,
          typename Access = resident_roles::DefaultNgramAccess>
[[nodiscard]] inline cudaError_t rebuild_surface_grammar_from_ngrams_cuda(
    SurfaceUnitView units, const BigramKey* bigrams, const std::uint32_t* bigram_counts,
    std::uint32_t bigram_count, const TrigramKey* trigrams,
    const std::uint32_t* trigram_counts, std::uint32_t trigram_count,
    SurfaceClosureView closure, MutableSurfaceGrammarEvidenceView evidence,
    cudaStream_t stream = nullptr) {
  if (units.unit_count == 0u || units.lengths == nullptr || units.packed_bytes == nullptr ||
      units.roles == nullptr || units.unit_words == 0u ||
      !valid_surface_grammar_evidence(evidence) || evidence.unit_capacity < units.unit_count ||
      bigrams == nullptr || bigram_counts == nullptr || bigram_count == 0u ||
      closure.learned_bytes == nullptr || closure.byte_count == 0u ||
      (trigram_count != 0u && (trigrams == nullptr || trigram_counts == nullptr)))
    return cudaErrorInvalidValue;
  cudaError_t status = clear_surface_grammar_evidence_cuda(evidence, stream);
  if (status != cudaSuccess)
    return status;
  accumulate_surface_bigram_presence_kernel<BigramKey, Access>
      <<<surface_organ_blocks(bigram_count), kSurfaceOrganBlockSize, 0, stream>>>(
          units, bigrams, bigram_counts, bigram_count, closure, evidence);
  status = cudaPeekAtLastError();
  if (status != cudaSuccess)
    return status;
  if (trigram_count != 0u) {
    accumulate_surface_trigram_presence_kernel<TrigramKey, Access>
        <<<surface_organ_blocks(trigram_count), kSurfaceOrganBlockSize, 0, stream>>>(
            units, trigrams, trigram_counts, trigram_count, closure, evidence);
    status = cudaPeekAtLastError();
    if (status != cudaSuccess)
      return status;
  }
  finish_surface_ngram_rebuild_kernel<<<1u, 1u, 0, stream>>>(evidence.stats, bigram_count);
  return cudaPeekAtLastError();
}

[[nodiscard]] inline cudaError_t realize_surface_construction_cuda(
    SurfaceUnitView units, OpaqueConstructionWitnessView witness,
    OpaqueContentPlanView plan, SurfaceRealizationWorkspaceView workspace,
    cudaStream_t stream = nullptr) {
  if (units.unit_count == 0u || units.lengths == nullptr ||
      units.packed_bytes == nullptr || units.roles == nullptr ||
      units.unit_words == 0u || plan.anchor_units == nullptr ||
      plan.anchor_count == 0u || plan.anchor_count > kSurfaceOrganMaxAnchors ||
      witness.capacity > construction::kConstructionCap ||
      workspace.output_units == nullptr ||
      workspace.output_anchor_mask == nullptr ||
      workspace.output_unit_capacity == 0u ||
      workspace.output_unit_capacity > kSurfaceOrganMaxOutputUnits ||
      workspace.output_bytes == nullptr ||
      workspace.output_byte_capacity == 0u || workspace.result == nullptr)
    return cudaErrorInvalidValue;
  realize_construction_witness_kernel<<<1u, 1u, 0, stream>>>(
      units, plan, witness, workspace);
  return cudaPeekAtLastError();
}

[[nodiscard]] inline cudaError_t realize_surface_cuda(SurfaceUnitView units,
                                                      MutableSurfaceGrammarEvidenceView evidence,
                                                      OpaqueContentPlanView plan,
                                                      SurfaceOrganConfig config,
                                                      SurfaceRealizationWorkspaceView workspace,
                                                      cudaStream_t stream = nullptr) {
  const std::uint32_t permutation_count = plan.anchor_count <= kSurfaceOrganMaxExhaustiveAnchors
                                              ? surface_factorial(plan.anchor_count)
                                              : 1u;
  if (units.unit_count == 0u || units.lengths == nullptr || units.packed_bytes == nullptr ||
      units.roles == nullptr || units.unit_words == 0u ||
      !valid_surface_grammar_evidence(evidence) || evidence.unit_capacity < units.unit_count ||
      plan.anchor_units == nullptr || plan.anchor_count == 0u ||
      plan.anchor_count > kSurfaceOrganMaxAnchors ||
      config.max_bridge_roles > kSurfaceOrganMaxBridgeRoles ||
      config.min_link_probability_q20 > kSurfaceOrganProbabilityOne ||
      workspace.role_bridges == nullptr || workspace.prefixes == nullptr ||
      workspace.suffixes == nullptr || workspace.permutation_scores == nullptr ||
      workspace.permutation_valid == nullptr ||
      workspace.permutation_capacity < permutation_count || workspace.output_units == nullptr ||
      workspace.output_anchor_mask == nullptr || workspace.output_unit_capacity == 0u ||
      workspace.output_unit_capacity > kSurfaceOrganMaxOutputUnits ||
      workspace.output_bytes == nullptr || workspace.output_byte_capacity == 0u ||
      workspace.result == nullptr)
    return cudaErrorInvalidValue;

  build_surface_role_bridges_kernel<<<surface_organ_blocks(kSurfaceOrganRoleBigramCount),
                                      kSurfaceOrganBlockSize, 0, stream>>>(evidence, config,
                                                                           workspace.role_bridges);
  cudaError_t status = cudaPeekAtLastError();
  if (status != cudaSuccess)
    return status;
  build_surface_boundary_paths_kernel<<<1u, kSurfaceOrganBlockSize, 0, stream>>>(
      evidence, config, workspace.prefixes, workspace.suffixes);
  status = cudaPeekAtLastError();
  if (status != cudaSuccess)
    return status;
  if (plan.anchor_count <= kSurfaceOrganMaxExhaustiveAnchors) {
    score_surface_anchor_permutations_kernel<<<surface_organ_blocks(permutation_count),
                                               kSurfaceOrganBlockSize, 0, stream>>>(
        units, evidence, plan, workspace.role_bridges, workspace.prefixes, workspace.suffixes,
        workspace.permutation_scores, workspace.permutation_valid);
    status = cudaPeekAtLastError();
    if (status != cudaSuccess)
      return status;
  }
  select_and_realize_surface_kernel<<<1u, 1u, 0, stream>>>(
      units, evidence, plan, workspace.role_bridges, workspace.prefixes, workspace.suffixes,
      workspace.permutation_scores, workspace.permutation_valid, false, workspace);
  return cudaPeekAtLastError();
}

template <typename BigramKey, typename TrigramKey>
[[nodiscard]] inline bool valid_surface_sequence_evidence(
    SurfaceSequenceEvidenceView<BigramKey, TrigramKey> sequence) {
  const bool base_bigrams =
      sequence.base_bigram_count != 0u &&
      sequence.base_bigrams != nullptr &&
      sequence.base_bigram_counts != nullptr;
  const bool online_bigrams =
      sequence.online_bigram_count != 0u &&
      sequence.online_bigrams != nullptr &&
      sequence.online_bigram_counts != nullptr;
  const bool base_trigrams =
      sequence.base_trigram_count == 0u ||
      (sequence.base_trigrams != nullptr &&
       sequence.base_trigram_counts != nullptr);
  const bool online_trigrams =
      sequence.online_trigram_count == 0u ||
      (sequence.online_trigrams != nullptr &&
       sequence.online_trigram_counts != nullptr);
  return (base_bigrams || online_bigrams) && base_trigrams &&
         online_trigrams;
}

template <typename BigramKey, typename TrigramKey,
          typename Access = resident_roles::DefaultNgramAccess>
[[nodiscard]] inline cudaError_t realize_surface_conditioned_cuda(
    SurfaceUnitView units, MutableSurfaceGrammarEvidenceView evidence,
    SurfaceSequenceEvidenceView<BigramKey, TrigramKey> sequence,
    OpaqueContentPlanView plan, SurfaceOrganConfig config,
    SurfaceRealizationWorkspaceView workspace, bool force_greedy = false,
    cudaStream_t stream = nullptr) {
  const std::uint32_t permutation_count =
      !force_greedy &&
              plan.anchor_count <= kSurfaceOrganMaxExhaustiveAnchors
          ? surface_factorial(plan.anchor_count)
          : 1u;
  if (units.unit_count == 0u || units.lengths == nullptr ||
      units.packed_bytes == nullptr || units.roles == nullptr ||
      units.unit_words == 0u || !valid_surface_grammar_evidence(evidence) ||
      evidence.unit_capacity < units.unit_count ||
      !valid_surface_sequence_evidence(sequence) ||
      plan.anchor_units == nullptr || plan.anchor_count == 0u ||
      plan.anchor_count > kSurfaceOrganMaxAnchors ||
      config.max_bridge_roles > kSurfaceOrganMaxBridgeRoles ||
      config.min_link_probability_q20 > kSurfaceOrganProbabilityOne ||
      workspace.role_bridges == nullptr || workspace.prefixes == nullptr ||
      workspace.suffixes == nullptr ||
      workspace.permutation_scores == nullptr ||
      workspace.permutation_valid == nullptr ||
      workspace.permutation_capacity < permutation_count ||
      workspace.output_units == nullptr ||
      workspace.output_anchor_mask == nullptr ||
      workspace.output_unit_capacity == 0u ||
      workspace.output_unit_capacity > kSurfaceOrganMaxOutputUnits ||
      workspace.output_bytes == nullptr ||
      workspace.output_byte_capacity == 0u || workspace.result == nullptr)
    return cudaErrorInvalidValue;

  build_surface_role_bridges_kernel<<<
      surface_organ_blocks(kSurfaceOrganRoleBigramCount),
      kSurfaceOrganBlockSize, 0, stream>>>(evidence, config,
                                           workspace.role_bridges);
  cudaError_t status = cudaPeekAtLastError();
  if (status != cudaSuccess)
    return status;
  build_surface_boundary_paths_kernel<<<1u, kSurfaceOrganBlockSize, 0,
                                        stream>>>(
      evidence, config, workspace.prefixes, workspace.suffixes);
  status = cudaPeekAtLastError();
  if (status != cudaSuccess)
    return status;
  if (!force_greedy &&
      plan.anchor_count <= kSurfaceOrganMaxExhaustiveAnchors) {
    score_surface_anchor_permutations_kernel<<<
        surface_organ_blocks(permutation_count), kSurfaceOrganBlockSize, 0,
        stream>>>(units, evidence, plan, workspace.role_bridges,
                  workspace.prefixes, workspace.suffixes,
                  workspace.permutation_scores, workspace.permutation_valid);
    status = cudaPeekAtLastError();
    if (status != cudaSuccess)
      return status;
  }
  select_and_realize_conditioned_surface_kernel<
      BigramKey, TrigramKey, Access><<<1u, 1u, 0, stream>>>(
      units, evidence, plan, sequence, workspace.role_bridges,
      workspace.prefixes, workspace.suffixes,
      workspace.permutation_scores, workspace.permutation_valid,
      force_greedy, workspace);
  return cudaPeekAtLastError();
}

[[nodiscard]] inline cudaError_t realize_surface_greedy_cuda(
    SurfaceUnitView units, MutableSurfaceGrammarEvidenceView evidence,
    OpaqueContentPlanView plan, SurfaceOrganConfig config,
    SurfaceRealizationWorkspaceView workspace, cudaStream_t stream = nullptr) {
  if (units.unit_count == 0u || units.lengths == nullptr ||
      units.packed_bytes == nullptr || units.roles == nullptr || units.unit_words == 0u ||
      !valid_surface_grammar_evidence(evidence) ||
      evidence.unit_capacity < units.unit_count || plan.anchor_units == nullptr ||
      plan.anchor_count == 0u || plan.anchor_count > kSurfaceOrganMaxAnchors ||
      config.max_bridge_roles > kSurfaceOrganMaxBridgeRoles ||
      config.min_link_probability_q20 > kSurfaceOrganProbabilityOne ||
      workspace.role_bridges == nullptr || workspace.prefixes == nullptr ||
      workspace.suffixes == nullptr || workspace.permutation_scores == nullptr ||
      workspace.permutation_valid == nullptr || workspace.output_units == nullptr ||
      workspace.output_anchor_mask == nullptr || workspace.output_unit_capacity == 0u ||
      workspace.output_unit_capacity > kSurfaceOrganMaxOutputUnits ||
      workspace.output_bytes == nullptr || workspace.output_byte_capacity == 0u ||
      workspace.result == nullptr)
    return cudaErrorInvalidValue;
  build_surface_role_bridges_kernel<<<surface_organ_blocks(kSurfaceOrganRoleBigramCount),
                                      kSurfaceOrganBlockSize, 0, stream>>>(
      evidence, config, workspace.role_bridges);
  cudaError_t status = cudaPeekAtLastError();
  if (status != cudaSuccess)
    return status;
  build_surface_boundary_paths_kernel<<<1u, kSurfaceOrganBlockSize, 0, stream>>>(
      evidence, config, workspace.prefixes, workspace.suffixes);
  status = cudaPeekAtLastError();
  if (status != cudaSuccess)
    return status;
  select_and_realize_surface_kernel<<<1u, 1u, 0, stream>>>(
      units, evidence, plan, workspace.role_bridges, workspace.prefixes,
      workspace.suffixes, workspace.permutation_scores, workspace.permutation_valid,
      true, workspace);
  return cudaPeekAtLastError();
}

}  // namespace substrate::bcc32::resident_surface_organ_v2
