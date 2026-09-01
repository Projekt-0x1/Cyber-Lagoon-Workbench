// Resident adult report readers. Included inside namespace
// bcc32_cuda_adult_v1 and the non-state-only CUDA host API guard.

inline TrainReport read_report(const AdultState& state) {
  TrainReport report{};
  report.unit_occurrences = state.unit_occurrences;
  report.unique_units = state.unit_count;
  report.unique_bigrams = state.bigram_count;
  report.unique_trigrams = state.trigram_count;
  report.online_bigrams = state.online_bigram_count;
  report.online_trigrams = state.online_trigram_count;
  report.online_associations = state.online_association_count;
  report.online_conditioned_transitions =
      state.online_conditioned_transition_count;
  report.online_episode_units = state.online_episode_count;
  report.resident_bytes = state.resident_bytes;
  report.ordered_binding_capacity =
      static_cast<std::uint32_t>(state.proposition_ordered_bindings.size());
  report.ordered_construction_capacity =
      static_cast<std::uint32_t>(state.proposition_ordered_construction.size());
  report.construction_association_capacity = static_cast<std::uint32_t>(
      state.proposition_construction_association.size());
  report.relation_triple_capacity =
      static_cast<std::uint32_t>(state.relation_triples.size());
  std::uint32_t ledger[4] = {};
  cuda_require(cudaMemcpy(ledger, state.ledger.get(), sizeof(ledger), cudaMemcpyDeviceToHost),
               "read resident mass ledger");
  report.mass_budget = ledger[0];
  report.mass_reserve = ledger[1];
  report.occupied_mass = ledger[2];
  report.ledger_ok = ledger[3];
  cuda_require(cudaMemcpy(report.boundary_bytes, state.boundary_bytes.get(),
                          sizeof(report.boundary_bytes), cudaMemcpyDeviceToHost),
               "read discovered boundary bytes");
  cuda_require(cudaMemcpy(report.motor_context, state.motor_context.get(),
                          sizeof(report.motor_context), cudaMemcpyDeviceToHost),
               "read resident motor context");
  cuda_require(cudaMemcpy(report.motor_completion, state.motor_completion.get(),
                          sizeof(report.motor_completion), cudaMemcpyDeviceToHost),
               "read resident motor completion");
  if (state.proposition_scalars.get() != nullptr) {
    cuda_require(cudaMemcpy(&report.proposition_tissue,
                            state.proposition_scalars.get(),
                            sizeof(report.proposition_tissue),
                            cudaMemcpyDeviceToHost),
                 "read resident proposition tissue scalars");
  }
  if (state.proposition_construction_association.get() != nullptr) {
    cuda_require(cudaMemcpy(&report.construction_association,
                            state.proposition_construction_association.get(),
                            sizeof(report.construction_association),
                            cudaMemcpyDeviceToHost),
                 "read resident construction association receipt");
  }
  if (state.conditioned_credit_scalars.get() != nullptr) {
    cuda_require(cudaMemcpy(&report.conditioned_credit,
                            state.conditioned_credit_scalars.get(),
                            sizeof(report.conditioned_credit),
                            cudaMemcpyDeviceToHost),
                 "read resident conditioned credit scalars");
  }
  return report;
}

inline EfferenceReport read_efference_report(const AdultState& state) {
  EfferenceReport report{};
  cuda_require(cudaMemcpy(&report, state.efference_state.get(), sizeof(report),
                          cudaMemcpyDeviceToHost),
               "read resident efference report");
  return report;
}
