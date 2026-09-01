inline void refresh_surface_organ(AdultState& state) {
  if (!state.surface_organ_enabled || state.unit_count == 0u)
    return;
  cuda_require(roles::derive_structural_roles_cuda(
                   state.unit_count, state.bigrams.get(), state.bigram_counts.get(),
                   state.bigram_count, state.trigrams.get(), state.trigram_counts.get(),
                   state.trigram_count, state.online_bigrams.get(),
                   state.online_bigram_counts.get(), state.online_bigram_count,
                   state.online_trigrams.get(), state.online_trigram_counts.get(),
                   state.online_trigram_count, state.surface_role_projection.get(),
                   state.surface_roles.get()),
               "derive persistent surface roles");
  cuda_require(surface_organ::rebuild_surface_grammar_from_ngrams_cuda(
                   surface_unit_view(state), state.bigrams.get(), state.bigram_counts.get(),
                   state.bigram_count, state.trigrams.get(), state.trigram_counts.get(),
                   state.trigram_count,
                   surface_organ::SurfaceClosureView{state.closure_bytes.get(), kClosureCount},
                   surface_evidence_view(state)),
               "rebuild normalized resident surface grammar");
  encode_resident_surface_populations(state, 0u, state.unit_count);
  cuda_require(cudaDeviceSynchronize(), "complete resident surface refresh");
}
inline bool project_distributed_surface_plan(AdultState& state) {
  if (!state.distributed_motor_enabled || !state.surface_organ_enabled)
    return false;
  if (state.unit_count <= state.surface_content_begin) {
    cuda_require(cudaMemset(state.motor_context.get(), 0, state.motor_context.bytes()),
                 "clear unavailable distributed surface plan");
    return false;
  }
  cuda_require(distributed_motor::project_opaque_unit_plan(
                   distributed_motor_view(state), state.unit_lengths.get(),
                   state.surface_unit_population.get(), state.unit_vitality.get(),
                   state.surface_content_begin,
                   state.unit_count - state.surface_content_begin,
                   state.surface_unit_activity.get(),
                   state.surface_unit_phase.get(), state.surface_projection_state.get(),
                   state.motor_context.get(),
                   state.motor_completion.get(), kCompositionUnits),
               "project distributed population into opaque surface plan");
  cuda_require(cudaDeviceSynchronize(),
               "complete distributed opaque surface projection");
  std::uint32_t ready = 0u;
  cuda_require(cudaMemcpy(&ready, state.surface_projection_state.get() + 1u,
                          sizeof(ready), cudaMemcpyDeviceToHost),
               "read distributed completed-plan latch");
  return ready != 0u;
}
