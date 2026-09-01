inline std::uint32_t synthesis_blocks_for(std::uint32_t count) {
  return (count + kResidentSynthesisBlock - 1u) / kResidentSynthesisBlock;
}

inline bool valid_synthesis_config(const ResidentSynthesisConfig& config) {
  return config.seed_count != 0u &&
      config.seed_count <= kResidentSynthesisMaxSeeds &&
      config.candidate_count != 0u &&
      config.candidate_count <= kResidentSynthesisMaxCandidates &&
      config.max_units != 0u && config.max_units <= kResidentSynthesisMaxUnits &&
      config.min_units != 0u && config.min_units <= config.max_units &&
      config.edge_scan_limit != 0u;
}

template <typename BigramKeyT, typename TrigramKeyT>
inline cudaError_t prepare_reverse_ngram_indexes(
    const ResidentSynthesisModelView<BigramKeyT, TrigramKeyT>& model,
    const ResidentSynthesisWorkspaceView& workspace,
    cudaStream_t stream = nullptr) {
  const std::size_t reverse_bigram_count =
      static_cast<std::size_t>(model.base_bigram_count) + model.online_bigram_count;
  const std::size_t reverse_trigram_count =
      static_cast<std::size_t>(model.base_trigram_count) + model.online_trigram_count;
  if (workspace.reverse_bigram_capacity < reverse_bigram_count ||
      workspace.reverse_trigram_capacity < reverse_trigram_count ||
      (reverse_bigram_count != 0u && workspace.reverse_bigrams == nullptr) ||
      (reverse_trigram_count != 0u && workspace.reverse_trigrams == nullptr)) {
    return cudaErrorInvalidValue;
  }
  if (model.base_bigram_count != 0u) {
    materialize_reverse_bigrams_kernel<<<
        synthesis_blocks_for(model.base_bigram_count), kResidentSynthesisBlock, 0u, stream>>>(
        model.base_bigrams, model.base_bigram_counts, model.base_bigram_count,
        workspace.reverse_bigrams, 0u);
  }
  if (model.online_bigram_count != 0u) {
    materialize_reverse_bigrams_kernel<<<
        synthesis_blocks_for(model.online_bigram_count), kResidentSynthesisBlock, 0u, stream>>>(
        model.online_bigrams, model.online_bigram_counts, model.online_bigram_count,
        workspace.reverse_bigrams, model.base_bigram_count);
  }
  if (model.base_trigram_count != 0u) {
    materialize_reverse_trigrams_kernel<<<
        synthesis_blocks_for(model.base_trigram_count), kResidentSynthesisBlock, 0u, stream>>>(
        model.base_trigrams, model.base_trigram_counts, model.base_trigram_count,
        workspace.reverse_trigrams, 0u);
  }
  if (model.online_trigram_count != 0u) {
    materialize_reverse_trigrams_kernel<<<
        synthesis_blocks_for(model.online_trigram_count), kResidentSynthesisBlock, 0u, stream>>>(
        model.online_trigrams, model.online_trigram_counts, model.online_trigram_count,
        workspace.reverse_trigrams, model.base_trigram_count);
  }
  cudaError_t status = cudaPeekAtLastError();
  if (status != cudaSuccess) return status;
  if (reverse_bigram_count != 0u) {
    thrust::sort(thrust::cuda::par.on(stream),
                 thrust::device_pointer_cast(workspace.reverse_bigrams),
                 thrust::device_pointer_cast(
                     workspace.reverse_bigrams + reverse_bigram_count));
  }
  if (reverse_trigram_count != 0u) {
    thrust::sort(thrust::cuda::par.on(stream),
                 thrust::device_pointer_cast(workspace.reverse_trigrams),
                 thrust::device_pointer_cast(
                     workspace.reverse_trigrams + reverse_trigram_count));
  }
  return cudaPeekAtLastError();
}

template <typename BigramKeyT, typename TrigramKeyT>
inline cudaError_t launch_resident_synthesis(
    const ResidentSynthesisModelView<BigramKeyT, TrigramKeyT>& model,
    const ResidentSynthesisWorkspaceView& workspace,
    ResidentSynthesisConfig config = {}, cudaStream_t stream = nullptr) {
  if (!valid_synthesis_config(config) || model.unit_count == 0u ||
      model.unit_lengths == nullptr || model.unit_content == nullptr ||
      workspace.seed_units == nullptr || workspace.seed_scores == nullptr ||
      workspace.drafts == nullptr || workspace.draft_lengths == nullptr ||
      workspace.draft_scores == nullptr || workspace.selected_units == nullptr ||
      workspace.result == nullptr || workspace.selected_capacity < config.max_units) {
    return cudaErrorInvalidValue;
  }
  cudaError_t status = prepare_reverse_ngram_indexes(model, workspace, stream);
  if (status != cudaSuccess) return status;
  status = cudaMemsetAsync(workspace.seed_units, 0xff,
      kResidentSynthesisMaxSeeds * sizeof(std::uint32_t), stream);
  if (status != cudaSuccess) return status;
  status = cudaMemsetAsync(workspace.seed_scores, 0,
      kResidentSynthesisMaxSeeds * sizeof(unsigned long long), stream);
  if (status != cudaSuccess) return status;
  status = cudaMemsetAsync(workspace.draft_lengths, 0,
      kResidentSynthesisMaxCandidates * sizeof(std::uint32_t), stream);
  if (status != cudaSuccess) return status;
  status = cudaMemsetAsync(workspace.draft_scores, 0,
      kResidentSynthesisMaxCandidates * sizeof(unsigned long long), stream);
  if (status != cudaSuccess) return status;
  status = cudaMemsetAsync(workspace.result, 0, sizeof(ResidentSynthesisResult), stream);
  if (status != cudaSuccess) return status;

  select_semantic_seeds_kernel<<<1u, kResidentSynthesisBlock, 0u, stream>>>(
      model, config.seed_count, workspace.seed_units, workspace.seed_scores);
  const std::uint32_t reverse_bigram_count =
      model.base_bigram_count + model.online_bigram_count;
  const std::uint32_t reverse_trigram_count =
      model.base_trigram_count + model.online_trigram_count;
  generate_bidirectional_drafts_kernel<<<config.candidate_count, 1u, 0u, stream>>>(
      model, workspace.reverse_bigrams, reverse_bigram_count,
      workspace.reverse_trigrams, reverse_trigram_count, workspace.seed_units,
      config, workspace.drafts, workspace.draft_lengths);
  for (std::uint32_t pass = 0u; pass < config.repair_passes; ++pass) {
    repair_draft_interiors_kernel<<<config.candidate_count, 1u, 0u, stream>>>(
        model, workspace.reverse_bigrams, reverse_bigram_count, config,
        workspace.drafts, workspace.draft_lengths);
  }
  appraise_drafts_kernel<<<
      synthesis_blocks_for(config.candidate_count), kResidentSynthesisBlock, 0u, stream>>>(
      model, config, workspace.seed_units, workspace.drafts, workspace.draft_lengths,
      workspace.draft_scores);
  select_synthesis_winner_kernel<<<1u, 1u, 0u, stream>>>(
      model, config, workspace.seed_units, workspace.drafts, workspace.draft_lengths,
      workspace.draft_scores, workspace.selected_units, workspace.result);
  return cudaPeekAtLastError();
}

}  // namespace bcc32_cuda_resident_synthesis
