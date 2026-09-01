inline void learn_incremental_surface_episodes(
    AdultState& state, const std::uint32_t* sequence, std::uint32_t sequence_count,
    const std::uint32_t* segment_ids, std::uint32_t previous_unit_count) {
  if (sequence_count == 0u)
    return;
  // Standalone adults do not own the resident surface organ. Avoid adapting
  // its population workspace on every contact when that organ cannot consume
  // the result; large non-language capacity contacts otherwise serialize
  // millions of adjacency updates through a dormant one-thread kernel.
  if (!state.surface_organ_enabled)
    return;
  const std::uint32_t novel_units = state.unit_count - previous_unit_count;
  if (novel_units != 0u && state.surface_unit_population.get() != nullptr &&
      state.surface_unit_mass.get() != nullptr) {
    encode_resident_surface_populations(state, previous_unit_count, novel_units);
    mark_complete_unit_population_formation_kernel<<<
        (novel_units + kBlock - 1u) / kBlock, kBlock>>>(
        state.surface_unit_population.get(), state.surface_unit_mass.get(),
        previous_unit_count, novel_units, kDistributedMotorActiveWidth,
        kDistributedMotorPopulation);
  }
  adapt_resident_population_coactivity(
      state, sequence, sequence_count, segment_ids);
  accumulate_incremental_surface_roles_kernel<<<
      (sequence_count + kBlock - 1u) / kBlock, kBlock>>>(
      sequence, sequence_count, state.surface_role_projection.get());
  if (novel_units != 0u) {
    finalize_incremental_surface_roles_kernel<<<
        (novel_units + kBlock - 1u) / kBlock, kBlock>>>(
        state.surface_role_projection.get(), previous_unit_count, novel_units,
        state.surface_roles.get());
  }
  std::uint32_t last_segment = 0u;
  cuda_require(cudaMemcpy(&last_segment, segment_ids + sequence_count - 1u,
                          sizeof(last_segment), cudaMemcpyDeviceToHost),
               "read incremental surface episode extent");
  const std::uint32_t episode_count = last_segment + 1u;
  DeviceArray<std::uint32_t> episode_offsets(episode_count + 1u);
  scatter_surface_episode_offsets_kernel<<<
      (sequence_count + kBlock - 1u) / kBlock, kBlock>>>(
      segment_ids, sequence_count, episode_offsets.get());
  const std::uint32_t host_document_offsets[2] = {0u, episode_count};
  DeviceArray<std::uint32_t> document_offsets(2u);
  cuda_require(cudaMemcpy(document_offsets.get(), host_document_offsets,
                          sizeof(host_document_offsets), cudaMemcpyHostToDevice),
               "publish incremental surface document extent");
  DeviceArray<std::uint64_t> document_units(1u);
  DeviceArray<std::uint64_t> document_bigrams(1u);
  DeviceArray<std::uint64_t> document_trigrams(1u);
  const surface_organ::SurfaceGrammarBatchView batch{
      sequence, episode_offsets.get(), document_offsets.get(), episode_count, 1u};
  const surface_organ::SurfaceLearningWorkspaceView learning{
      document_units.get(), document_bigrams.get(), document_trigrams.get(), 1u};
  cuda_require(surface_organ::learn_surface_grammar_cuda(
                   surface_unit_view(state), batch, surface_evidence_view(state), learning),
               "learn incremental resident surface episodes");
  cuda_require(context_state::learn(
                   surface_context_field_view(state), surface_context_workspace_view(state),
                   {sequence, episode_offsets.get(), state.surface_roles.get(), sequence_count,
                    episode_count, state.unit_count}),
               "learn incremental resident surface contexts");
  cuda_require(cudaDeviceSynchronize(), "complete incremental resident surface learning");
}
