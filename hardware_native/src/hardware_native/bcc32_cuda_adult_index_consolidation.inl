// Adult resident index and consolidation helpers.
//
// Included after the generation kernel and before the training pipeline.
// This unit owns capacity, index materialization, suffix consolidation,
// online n-gram consolidation, and base proposition promotion helpers.

inline std::uint32_t blocks_for(std::uint32_t count) {
  return std::max(1u, (count + kBlock - 1u) / kBlock);
}

inline void publish_resident_credit_conductance(AdultState& state) {
  const std::uint32_t count = state.online_conditioned_transition_count;
  DeviceArray<std::uint32_t> published(count == 0u ? 1u : count);
  DeviceArray<std::uint32_t> published_exposure(count == 0u ? 1u : count);
  cuda_require(cudaMemset(published.get(), 0, published.bytes()),
               "clear resident credit conductance");
  cuda_require(cudaMemset(published_exposure.get(), 0, published_exposure.bytes()),
               "clear resident credit exposure");
  if (count != 0u) {
    publish_conditioned_credit_conductance_kernel<<<blocks_for(count), kBlock>>>(
        state.online_conditioned_transitions.get(), count,
        resident_credit_bank_view(state), published.get(),
        published_exposure.get());
    cuda_require(cudaGetLastError(),
                 "publish resident credit conductance");
  }
  state.online_conditioned_transition_conductance = std::move(published);
  state.online_conditioned_transition_exposure = std::move(published_exposure);
}

inline std::uint32_t power_of_two_capacity(std::uint32_t required) {
  std::uint32_t capacity = 1u;
  while (capacity < required) {
    if (capacity > 0x40000000u) throw std::runtime_error("resident hash capacity overflow");
    capacity <<= 1u;
  }
  return capacity;
}

template <typename T>
inline DeviceArray<T> compact_copy(const DeviceArray<T>& source, std::uint32_t count) {
  DeviceArray<T> result(count);
  if (count != 0u) {
    cuda_require(cudaMemcpy(result.get(), source.get(), sizeof(T) * count,
                            cudaMemcpyDeviceToDevice), "compact learned state");
  }
  return result;
}

inline void build_generation_indexes(AdultState& state) {
  DeviceArray<std::uint32_t> flags(state.bigram_count);
  DeviceArray<std::uint32_t> ids(state.bigram_count);
  mark_large_bigram_contexts_kernel<<<blocks_for(state.bigram_count), kBlock>>>(
      state.bigrams.get(), state.bigram_count, flags.get());
  thrust::inclusive_scan(thrust::device, flags.get(), flags.get() + state.bigram_count,
                         ids.get());
  cuda_require(cudaMemcpy(&state.cached_bigram_count, ids.get() + state.bigram_count - 1u,
                          sizeof(std::uint32_t), cudaMemcpyDeviceToHost),
               "read cached bigram context extent");
  state.cached_bigram_contexts.allocate(state.cached_bigram_count);
  state.cached_bigram_entries.allocate(
      static_cast<std::size_t>(state.cached_bigram_count) * kTopK);
  materialize_bigram_cache_kernel<<<blocks_for(state.bigram_count), kBlock>>>(
      state.bigrams.get(), state.bigram_counts.get(), state.bigram_count,
      flags.get(), ids.get(), state.cached_bigram_contexts.get(),
      state.cached_bigram_entries.get());

  flags.allocate(state.trigram_count);
  ids.allocate(state.trigram_count);
  mark_large_trigram_contexts_kernel<<<blocks_for(state.trigram_count), kBlock>>>(
      state.trigrams.get(), state.trigram_count, flags.get());
  thrust::inclusive_scan(thrust::device, flags.get(), flags.get() + state.trigram_count,
                         ids.get());
  cuda_require(cudaMemcpy(&state.cached_trigram_count, ids.get() + state.trigram_count - 1u,
                          sizeof(std::uint32_t), cudaMemcpyDeviceToHost),
               "read cached trigram context extent");
  state.cached_trigram_contexts.allocate(state.cached_trigram_count);
  state.cached_trigram_entries.allocate(
      static_cast<std::size_t>(state.cached_trigram_count) * kTopK);
  materialize_trigram_cache_kernel<<<blocks_for(state.trigram_count), kBlock>>>(
      state.trigrams.get(), state.trigram_counts.get(), state.trigram_count,
      flags.get(), ids.get(), state.cached_trigram_contexts.get(),
      state.cached_trigram_entries.get());
  cuda_require(cudaGetLastError(), "build generation context indexes");
}

inline void build_base_episode_indexes(AdultState& state, bool materialize) {
  DeviceArray<std::uint32_t> shifted_counts(state.unit_capacity + 1u);
  cuda_require(cudaMemset(shifted_counts.get(), 0, shifted_counts.bytes()),
               "clear base posting counts");
  state.base_posting_offsets.allocate(state.unit_capacity + 1u);
  if (materialize) {
    DeviceArray<std::uint32_t> posting_keys(state.unit_occurrences);
    state.base_posting_positions.allocate(state.unit_occurrences);
    build_posting_material_kernel<<<blocks_for(state.unit_occurrences), kBlock>>>(
        state.base_episode_units.get(), state.unit_occurrences, posting_keys.get(),
        state.base_posting_positions.get(), shifted_counts.get());
    thrust::stable_sort_by_key(thrust::device, posting_keys.get(),
                               posting_keys.get() + state.unit_occurrences,
                               state.base_posting_positions.get());
    state.base_window_count =
        (state.unit_occurrences + kBaseWindowStride - 1u) / kBaseWindowStride;
    state.base_window_signatures.allocate(state.base_window_count);
    build_window_signatures_kernel<<<blocks_for(state.base_window_count), kBlock>>>(
        state.base_episode_units.get(), state.unit_occurrences, state.base_window_count,
        state.base_window_signatures.get());
    thrust::sort(thrust::device, state.base_window_signatures.get(),
                 state.base_window_signatures.get() + state.base_window_count);
  } else {
    count_episode_postings_kernel<<<blocks_for(state.unit_occurrences), kBlock>>>(
        state.base_episode_units.get(), state.unit_occurrences, shifted_counts.get());
  }
  thrust::inclusive_scan(thrust::device, shifted_counts.get(),
                         shifted_counts.get() + state.unit_capacity + 1u,
                         state.base_posting_offsets.get());
  cuda_require(cudaGetLastError(), "build resident base episode indexes");
}

template <typename Key>
inline std::uint32_t consolidate_sorted_resident_suffix(
    Key* keys, std::uint32_t* counts, std::uint32_t total_count,
    std::uint32_t sorted_prefix) {
  if (total_count == 0u || sorted_prefix >= total_count) return total_count;
  if (sorted_prefix == 0u) {
    thrust::sort_by_key(thrust::device, keys, keys + total_count, counts);
    DeviceArray<Key> compacted_keys(total_count);
    DeviceArray<std::uint32_t> compacted_counts(total_count);
    auto compacted_end = thrust::reduce_by_key(
        thrust::device, keys, keys + total_count, counts,
        compacted_keys.get(), compacted_counts.get());
    const auto compacted_count = static_cast<std::uint32_t>(
        compacted_end.first - compacted_keys.get());
    cuda_require(cudaMemcpy(keys, compacted_keys.get(),
                            compacted_count * sizeof(Key),
                            cudaMemcpyDeviceToDevice),
                 "compact resident keys");
    cuda_require(cudaMemcpy(counts, compacted_counts.get(),
                            compacted_count * sizeof(std::uint32_t),
                            cudaMemcpyDeviceToDevice),
                 "compact resident counts");
    return compacted_count;
  }

  const std::uint32_t suffix_count = total_count - sorted_prefix;
  Key* suffix_keys = keys + sorted_prefix;
  std::uint32_t* suffix_counts = counts + sorted_prefix;
  thrust::sort_by_key(thrust::device, suffix_keys,
                      suffix_keys + suffix_count, suffix_counts);
  DeviceArray<Key> new_keys(suffix_count);
  DeviceArray<std::uint32_t> new_counts(suffix_count);
  auto new_end = thrust::reduce_by_key(
      thrust::device, suffix_keys, suffix_keys + suffix_count, suffix_counts,
      new_keys.get(), new_counts.get());
  const auto new_count =
      static_cast<std::uint32_t>(new_end.first - new_keys.get());
  DeviceArray<Key> merged_keys(sorted_prefix + new_count);
  DeviceArray<std::uint32_t> merged_counts(sorted_prefix + new_count);
  auto merged_end = thrust::merge_by_key(
      thrust::device, keys, keys + sorted_prefix, new_keys.get(),
      new_keys.get() + new_count, counts, new_counts.get(),
      merged_keys.get(), merged_counts.get());
  auto compacted_end = thrust::reduce_by_key(
      thrust::device, merged_keys.get(), merged_end.first, merged_counts.get(),
      keys, counts);
  return static_cast<std::uint32_t>(compacted_end.first - keys);
}

inline void consolidate_online_ngrams(
    AdultState& state, std::uint32_t conditioned_sorted_prefix = 0u,
    std::uint32_t bigram_sorted_prefix = 0u,
    std::uint32_t trigram_sorted_prefix = 0u,
    std::uint32_t association_sorted_prefix = 0u) {
  if (state.online_bigram_count != 0u) {
    state.online_bigram_count = consolidate_sorted_resident_suffix(
        state.online_bigrams.get(), state.online_bigram_counts.get(),
        state.online_bigram_count, bigram_sorted_prefix);
  }
  if (state.online_trigram_count != 0u) {
    state.online_trigram_count = consolidate_sorted_resident_suffix(
        state.online_trigrams.get(), state.online_trigram_counts.get(),
        state.online_trigram_count, trigram_sorted_prefix);
  }
  if (state.online_association_count != 0u) {
    state.online_association_count = consolidate_sorted_resident_suffix(
        state.online_associations.get(), state.online_association_counts.get(),
        state.online_association_count, association_sorted_prefix);
  }
  if (state.online_conditioned_transition_count != 0u) {
    if (conditioned_sorted_prefix == 0u) {
      thrust::sort_by_key(
          thrust::device, state.online_conditioned_transitions.get(),
          state.online_conditioned_transitions.get() +
              state.online_conditioned_transition_count,
          state.online_conditioned_transition_counts.get());
      DeviceArray<ConditionedTransitionKey> keys(
          state.online_conditioned_transition_count);
      DeviceArray<std::uint32_t> counts(
          state.online_conditioned_transition_count);
      auto end = thrust::reduce_by_key(
          thrust::device, state.online_conditioned_transitions.get(),
          state.online_conditioned_transitions.get() +
              state.online_conditioned_transition_count,
          state.online_conditioned_transition_counts.get(), keys.get(), counts.get());
      state.online_conditioned_transition_count =
          static_cast<std::uint32_t>(end.first - keys.get());
      cuda_require(
          cudaMemcpy(state.online_conditioned_transitions.get(), keys.get(),
                     state.online_conditioned_transition_count *
                         sizeof(ConditionedTransitionKey),
                     cudaMemcpyDeviceToDevice),
          "compact online conditioned transitions");
      cuda_require(
          cudaMemcpy(state.online_conditioned_transition_counts.get(), counts.get(),
                     state.online_conditioned_transition_count *
                         sizeof(std::uint32_t),
                     cudaMemcpyDeviceToDevice),
          "compact online conditioned transition counts");
    } else if (conditioned_sorted_prefix <
               state.online_conditioned_transition_count) {
      const std::uint32_t suffix_count =
          state.online_conditioned_transition_count - conditioned_sorted_prefix;
      ConditionedTransitionKey* suffix =
          state.online_conditioned_transitions.get() + conditioned_sorted_prefix;
      std::uint32_t* suffix_counts =
          state.online_conditioned_transition_counts.get() +
          conditioned_sorted_prefix;
      thrust::sort_by_key(thrust::device, suffix, suffix + suffix_count,
                          suffix_counts);
      DeviceArray<ConditionedTransitionKey> new_keys(suffix_count);
      DeviceArray<std::uint32_t> new_counts(suffix_count);
      auto new_end = thrust::reduce_by_key(
          thrust::device, suffix, suffix + suffix_count, suffix_counts,
          new_keys.get(), new_counts.get());
      const std::uint32_t new_count =
          static_cast<std::uint32_t>(new_end.first - new_keys.get());
      DeviceArray<ConditionedTransitionKey> merged_keys(
          conditioned_sorted_prefix + new_count);
      DeviceArray<std::uint32_t> merged_counts(
          conditioned_sorted_prefix + new_count);
      auto merged_end = thrust::merge_by_key(
          thrust::device, state.online_conditioned_transitions.get(),
          state.online_conditioned_transitions.get() + conditioned_sorted_prefix,
          new_keys.get(), new_keys.get() + new_count,
          state.online_conditioned_transition_counts.get(), new_counts.get(),
          merged_keys.get(), merged_counts.get());
      auto compacted_end = thrust::reduce_by_key(
          thrust::device, merged_keys.get(), merged_end.first,
          merged_counts.get(), state.online_conditioned_transitions.get(),
          state.online_conditioned_transition_counts.get());
      state.online_conditioned_transition_count =
          static_cast<std::uint32_t>(
              compacted_end.first - state.online_conditioned_transitions.get());
    }
  }
  std::uint32_t sizes[7] = {};
  cuda_require(cudaMemcpy(sizes, state.mutable_sizes.get(), sizeof(sizes),
                          cudaMemcpyDeviceToHost), "read consolidated mutable extents");
  sizes[1] = state.online_bigram_count;
  sizes[2] = state.online_trigram_count;
  sizes[3] = state.online_association_count;
  sizes[6] = state.online_conditioned_transition_count;
  cuda_require(cudaMemcpy(state.mutable_sizes.get(), sizes, sizeof(sizes),
                          cudaMemcpyHostToDevice), "publish consolidated mutable extents");
}

inline void promote_base_episode_relations(AdultState& state) {
  if (state.online_association_count != 0u || state.unit_occurrences < 2u ||
      state.base_episode_lesioned) {
    return;
  }
  unsigned long long event_count = 0u;
  unsigned long long mass = 0u;
  for (std::uint32_t distance = 1u; distance <= kAssociationRadius; ++distance) {
    const std::uint32_t extent = state.unit_occurrences > distance
        ? state.unit_occurrences - distance : 0u;
    event_count += extent;
    mass += static_cast<unsigned long long>(extent) *
            (kAssociationRadius + 1u - distance);
  }
  if (event_count > kOnlineAssociationCapacity || mass > 0xffffffffull) {
    throw std::runtime_error("base relation field exceeds mutable resident capacity");
  }
  DeviceArray<std::uint32_t> status(1u);
  reserve_fixed_mass_kernel<<<1u, 32u>>>(static_cast<std::uint32_t>(mass),
                                         state.ledger.get(), status.get());
  cuda_require(cudaDeviceSynchronize(), "reserve base relation evidence mass");
  std::uint32_t host_status = 0u;
  cuda_require(cudaMemcpy(&host_status, status.get(), sizeof(host_status),
                          cudaMemcpyDeviceToHost), "read base relation reserve status");
  if (host_status != 0u) {
    throw std::runtime_error("base relation evidence exceeds fixed resident mass");
  }
  DeviceArray<std::uint32_t> segment_starts(state.unit_occurrences);
  DeviceArray<std::uint32_t> segment_ids(state.unit_occurrences);
  mark_sequence_segment_starts_kernel<<<blocks_for(state.unit_occurrences), kBlock>>>(
      state.base_episode_units.get(), state.unit_occurrences,
      state.unit_lengths.get(), state.unit_content.get(),
      state.closure_bytes.get(), state.boundary_bytes.get(), segment_starts.get());
  thrust::inclusive_scan(thrust::device, segment_starts.get(),
                         segment_starts.get() + state.unit_occurrences,
                         segment_ids.get());
  DeviceArray<std::uint32_t> appended_count(1u);
  cuda_require(cudaMemset(appended_count.get(), 0, appended_count.bytes()),
               "clear base relation append extent");
  build_sequence_associations_kernel<<<
      min(4096u, blocks_for(static_cast<std::uint32_t>(event_count))), kBlock>>>(
      state.base_episode_units.get(), segment_ids.get(), state.unit_occurrences,
      state.online_associations.get(), state.online_association_counts.get(),
      appended_count.get(), state.ledger.get());
  cuda_require(cudaGetLastError(), "launch base relation promotion");
  cuda_require(cudaDeviceSynchronize(), "complete base relation promotion");
  cuda_require(cudaMemcpy(&state.online_association_count, appended_count.get(),
                          sizeof(state.online_association_count),
                          cudaMemcpyDeviceToHost),
               "read boundary-aware base relation extent");
  consolidate_online_ngrams(state);
}

inline void learn_base_resident_proposition_sequence(AdultState& state) {
  if (!state.surface_organ_enabled || state.unit_occurrences < 3u ||
      state.base_episode_units.get() == nullptr)
    return;
  DeviceArray<std::uint32_t> segment_starts(state.unit_occurrences);
  DeviceArray<std::uint32_t> segment_ids(state.unit_occurrences);
  mark_sequence_segment_starts_kernel<<<blocks_for(state.unit_occurrences),
                                        kBlock>>>(
      state.base_episode_units.get(), state.unit_occurrences,
      state.unit_lengths.get(), state.unit_content.get(),
      state.closure_bytes.get(), state.boundary_bytes.get(), segment_starts.get());
  thrust::inclusive_scan(thrust::device, segment_starts.get(),
                         segment_starts.get() + state.unit_occurrences,
                         segment_ids.get());
  adapt_resident_population_coactivity(
      state, state.base_episode_units.get(), state.unit_occurrences,
      segment_ids.get());
  std::uint32_t last_segment = 0u;
  cuda_require(cudaMemcpy(&last_segment, segment_ids.get() + state.unit_occurrences - 1u,
                          sizeof(last_segment), cudaMemcpyDeviceToHost),
               "read base surface context episode extent");
  const std::uint32_t episode_count = last_segment + 1u;
  DeviceArray<std::uint32_t> episode_offsets(episode_count + 1u);
  scatter_surface_episode_offsets_kernel<<<blocks_for(state.unit_occurrences), kBlock>>>(
      segment_ids.get(), state.unit_occurrences, episode_offsets.get());
  cuda_require(context_state::learn(
                   surface_context_field_view(state), surface_context_workspace_view(state),
                   {state.base_episode_units.get(), episode_offsets.get(),
                    state.surface_roles.get(), state.unit_occurrences, episode_count,
                    state.unit_count}),
               "learn base resident surface contexts");
  assimilate_resident_candidate_episodes(
      state, state.base_episode_units.get(), segment_ids.get(),
      state.unit_occurrences);
  learn_resident_constructions(
      state, state.base_episode_units.get(), state.unit_occurrences,
      segment_ids.get(), false, 0, true, true, true);
  cuda_require(cudaDeviceSynchronize(),
               "complete base resident proposition learning");
}
