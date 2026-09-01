// Resident episode and transition lesion operations. This fragment is included
// from bcc32_cuda_adult_v1.cuh after its kernel and AdultState declarations.

inline void lesion_transition_state(AdultState& state) {
  if (state.transitions_lesioned) return;
  lesion_counts_kernel<<<blocks_for(state.bigram_count), kBlock>>>(
      state.bigram_counts.get(), state.bigram_count, state.ledger.get());
  lesion_counts_kernel<<<blocks_for(state.trigram_count), kBlock>>>(
      state.trigram_counts.get(), state.trigram_count, state.ledger.get());
  lesion_counts_kernel<<<blocks_for(state.online_bigram_count), kBlock>>>(
      state.online_bigram_counts.get(), state.online_bigram_count, state.ledger.get());
  lesion_counts_kernel<<<blocks_for(state.online_trigram_count), kBlock>>>(
      state.online_trigram_counts.get(), state.online_trigram_count, state.ledger.get());
  lesion_counts_kernel<<<blocks_for(state.online_association_count), kBlock>>>(
      state.online_association_counts.get(), state.online_association_count,
      state.ledger.get());
  lesion_counts_kernel<<<blocks_for(state.online_conditioned_transition_count), kBlock>>>(
      state.online_conditioned_transition_counts.get(),
      state.online_conditioned_transition_count, state.ledger.get());
  audit_ledger_kernel<<<1u, kBlock>>>(
      state.unit_vitality.get(), state.unit_count,
      state.bigram_counts.get(), state.bigram_count,
      state.trigram_counts.get(), state.trigram_count,
      state.online_bigram_counts.get(), state.online_bigram_count,
      state.online_trigram_counts.get(), state.online_trigram_count,
      state.online_association_counts.get(), state.online_association_count,
      state.online_conditioned_transition_counts.get(),
      state.online_conditioned_transition_count,
      (state.base_episode_lesioned ? 0u : state.unit_occurrences) +
          state.online_episode_count, state.boundary_histogram.get(),
      state.boundary_pairs.get(), state.ledger.get());
  cuda_require(cudaGetLastError(), "launch transition lesion");
  cuda_require(cudaDeviceSynchronize(), "complete transition lesion");
  state.transitions_lesioned = true;
}

inline void lesion_conditioned_transition_state(AdultState& state) {
  if (state.online_conditioned_transition_count != 0u) {
    lesion_counts_kernel<<<blocks_for(
        state.online_conditioned_transition_count), kBlock>>>(
        state.online_conditioned_transition_counts.get(),
        state.online_conditioned_transition_count, state.ledger.get());
  }
  const resident_credit::BankView bank = resident_credit_bank_view(state);
  lesion_conditioned_credit_bank_kernel<<<blocks_for(bank.capacity), kBlock>>>(
      bank);
  audit_ledger_kernel<<<1u, kBlock>>>(
      state.unit_vitality.get(), state.unit_count,
      state.bigram_counts.get(), state.bigram_count,
      state.trigram_counts.get(), state.trigram_count,
      state.online_bigram_counts.get(), state.online_bigram_count,
      state.online_trigram_counts.get(), state.online_trigram_count,
      state.online_association_counts.get(), state.online_association_count,
      state.online_conditioned_transition_counts.get(),
      state.online_conditioned_transition_count,
      (state.base_episode_lesioned ? 0u : state.unit_occurrences) +
          state.online_episode_count,
      state.boundary_histogram.get(), state.boundary_pairs.get(),
      state.ledger.get());
  publish_resident_credit_conductance(state);
  cuda_require(cudaGetLastError(),
               "launch conditioned transition lesion");
  cuda_require(cudaDeviceSynchronize(),
               "complete conditioned transition lesion");
}

inline void lesion_episode_state(AdultState& state) {
  lesion_episode_kernel<<<1u, 32u>>>(state.mutable_sizes.get(), state.ledger.get());
  if (!state.base_episode_lesioned) {
    return_fixed_mass_kernel<<<1u, 32u>>>(state.unit_occurrences, state.ledger.get());
    state.base_episode_lesioned = true;
  }
  state.online_episode_count = 0u;
  state.online_episode_break_count = 0u;
  cuda_require(cudaMemset(state.motor_context.get(), 0, state.motor_context.bytes()),
               "clear lesioned episode motor context");
  cuda_require(cudaMemset(state.motor_completion.get(), 0, state.motor_completion.bytes()),
               "clear lesioned episode completion");
  audit_ledger_kernel<<<1u, kBlock>>>(
      state.unit_vitality.get(), state.unit_count,
      state.bigram_counts.get(), state.bigram_count,
      state.trigram_counts.get(), state.trigram_count,
      state.online_bigram_counts.get(), state.online_bigram_count,
      state.online_trigram_counts.get(), state.online_trigram_count,
      state.online_association_counts.get(), state.online_association_count,
      state.online_conditioned_transition_counts.get(),
      state.online_conditioned_transition_count,
      0u, state.boundary_histogram.get(), state.boundary_pairs.get(), state.ledger.get());
  cuda_require(cudaGetLastError(), "launch resident episode lesion");
  cuda_require(cudaDeviceSynchronize(), "complete resident episode lesion");
}

inline void begin_novelty_epoch(AdultState& state) {
  state.novelty_epoch_pending = true;
}

inline void lesion_recent_episode_state(AdultState& state) {
  if (!state.novelty_epoch_active ||
      state.novelty_episode_begin > state.online_episode_count ||
      state.novelty_episode_break_begin > state.online_episode_break_count) {
    return;
  }
  lesion_recent_episode_kernel<<<1u, 32u>>>(
      state.mutable_sizes.get(), state.novelty_episode_begin,
      state.novelty_episode_break_begin, state.ledger.get());
  state.online_episode_count = state.novelty_episode_begin;
  state.online_episode_break_count = state.novelty_episode_break_begin;
  state.novelty_epoch_active = false;
  state.novelty_epoch_pending = false;
  cuda_require(cudaMemset(state.motor_context.get(), 0, state.motor_context.bytes()),
               "clear recent-episode lesion motor context");
  cuda_require(cudaMemset(state.motor_completion.get(), 0,
                          state.motor_completion.bytes()),
               "clear recent-episode lesion completion");
  audit_ledger_kernel<<<1u, kBlock>>>(
      state.unit_vitality.get(), state.unit_count,
      state.bigram_counts.get(), state.bigram_count,
      state.trigram_counts.get(), state.trigram_count,
      state.online_bigram_counts.get(), state.online_bigram_count,
      state.online_trigram_counts.get(), state.online_trigram_count,
      state.online_association_counts.get(), state.online_association_count,
      state.online_conditioned_transition_counts.get(),
      state.online_conditioned_transition_count,
      (state.base_episode_lesioned ? 0u : state.unit_occurrences) +
          state.online_episode_count,
      state.boundary_histogram.get(), state.boundary_pairs.get(), state.ledger.get());
  cuda_require(cudaGetLastError(), "launch recent resident episode lesion");
  cuda_require(cudaDeviceSynchronize(), "complete recent resident episode lesion");
}
