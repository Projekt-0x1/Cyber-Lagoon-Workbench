// Adult raw-contact, reafference, and interaction-shadow control.
//
// This unit stays inside bcc32_cuda_adult_v1's namespace and owns the
// host/device boundary for streaming contact after assimilation has built
// the adult's resident substrate. Keep it ordered after raw assimilation
// and before checkpoint/report surfaces.
inline void assimilate_raw_intervention_bytes(
    AdultState& state, const std::uint8_t* host_bytes, std::uint32_t byte_count,
    std::int32_t efferent_polarity, bool outcome_present,
    std::uint32_t seed = 0x9e3779b9u) {
  if (efferent_polarity == 0)
    throw std::runtime_error("raw intervention polarity must be nonzero");
  assimilate_raw_bytes(state, host_bytes, byte_count, false, seed, true,
                       efferent_polarity, outcome_present);
}

__global__ void lesion_resident_synthesis_policy_kernel(
    bcc32_cuda_resident_synthesis::ResidentSynthesisPolicyState* policy) {
  if (blockIdx.x == 0u && threadIdx.x == 0u) {
    bcc32_cuda_resident_synthesis::resident_synthesis_policy_lesion(policy);
  }
}

inline void lesion_resident_synthesis_policy(AdultState& state) {
  auto* policy = state.synthesis_policy.get();
  void* policy_arguments[] = {&policy};
  cuda_require(
      cudaLaunchKernel(
          reinterpret_cast<const void*>(lesion_resident_synthesis_policy_kernel),
          dim3{1u, 1u, 1u}, dim3{2u, 1u, 1u}, policy_arguments, 0u, nullptr),
      "launch resident synthesis policy lesion");
  cuda_require(cudaGetLastError(), "launch resident synthesis policy lesion");
  cuda_require(cudaDeviceSynchronize(),
               "complete resident synthesis policy lesion");
}

inline void lesion_boundary_state(AdultState& state) {
  lesion_counts_kernel<<<1u, kBlock>>>(state.boundary_histogram.get(), 256u,
                                       state.ledger.get());
  lesion_counts_kernel<<<blocks_for(256u * 256u), kBlock>>>(
      state.boundary_pairs.get(), 256u * 256u, state.ledger.get());
  cuda_require(cudaMemset(state.boundary_mask.get(), 0, state.boundary_mask.bytes()),
               "clear mutable boundary hypotheses");
  cuda_require(cudaMemset(state.boundary_bytes.get(), 0xff, state.boundary_bytes.bytes()),
               "clear reported boundary hypotheses");
  cuda_require(cudaMemset(state.closure_bytes.get(), 0xff, state.closure_bytes.bytes()),
               "clear reported closure hypotheses");
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
  cuda_require(cudaGetLastError(), "launch mutable boundary lesion");
  cuda_require(cudaDeviceSynchronize(), "complete mutable boundary lesion");
}

__global__ void append_streaming_cue_kernel(
    const std::uint8_t* incoming, std::uint32_t incoming_count,
    const std::uint32_t* closure_bytes, std::uint32_t closure_count,
    bool complete_on_closure, bool flush_contact,
    std::uint8_t* stream, std::uint32_t capacity, std::uint32_t* meta) {
  if (blockIdx.x != 0u || threadIdx.x != 0u) return;
  std::uint32_t count = meta[0];
  if (meta[1] != 0u) count = 0u;
  meta[1] = 0u;
  if (incoming_count > capacity - (count < capacity ? count : capacity)) {
    meta[0] = 0u;
    meta[2] = 1u;
    return;
  }
  for (std::uint32_t i = 0u; i < incoming_count; ++i)
    stream[count + i] = incoming[i];
  count += incoming_count;
  meta[0] = count;
  if (count == 0u) return;
  if (flush_contact) {
    meta[1] = 1u;
    return;
  }
  if (!complete_on_closure) return;
  const std::uint32_t terminal = stream[count - 1u];
  for (std::uint32_t i = 0u; i < closure_count; ++i) {
    if (closure_bytes[i] < 256u && closure_bytes[i] == terminal) {
      meta[1] = 1u;
      return;
    }
  }
}

// Buffer opaque body-surface fragments on device. Transport framing carries no
// episode semantics: learned closure matter or an explicit physical quiet flush
// completes an episode. The host observes readiness and extent, never meaning.
inline std::vector<std::uint8_t> append_streaming_contact(
    AdultState& state, const std::uint8_t* host_bytes,
    std::uint32_t byte_count, bool flush_contact = false) {
  DeviceArray<std::uint8_t> incoming(byte_count == 0u ? 1u : byte_count);
  if (byte_count != 0u) {
    cuda_require(cudaMemcpy(incoming.get(), host_bytes, byte_count,
                            cudaMemcpyHostToDevice),
                 "upload streaming contact fragment");
  }
  auto* incoming_ptr = incoming.get();
  std::uint32_t incoming_count = byte_count;
  auto* closure_bytes = state.construction_closure_bytes.get();
  std::uint32_t closure_count = state.construction_closure_count;
  bool complete_on_closure = true;
  bool flush = flush_contact;
  auto* stream = state.streaming_cue_bytes.get();
  std::uint32_t capacity = kStreamingCueCapacity;
  auto* meta_ptr = state.streaming_cue_meta.get();
  void* append_arguments[] = {
      &incoming_ptr, &incoming_count, &closure_bytes, &closure_count,
      &complete_on_closure, &flush, &stream, &capacity, &meta_ptr};
  cuda_require(
      cudaLaunchKernel(
          reinterpret_cast<const void*>(append_streaming_cue_kernel),
          dim3{1u, 1u, 1u}, dim3{2u, 1u, 1u}, append_arguments, 0u, nullptr),
      "launch append resident streaming contact");
  cuda_require(cudaGetLastError(), "append resident streaming contact");
  std::uint32_t meta[3] = {};
  cuda_require(cudaMemcpy(meta, state.streaming_cue_meta.get(), sizeof(meta),
                          cudaMemcpyDeviceToHost),
               "read resident streaming contact state");
  if (meta[2] != 0u)
    throw std::runtime_error("resident streaming contact capacity exceeded");
  if (meta[1] == 0u || meta[0] == 0u) return {};
  std::vector<std::uint8_t> complete(meta[0]);
  cuda_require(cudaMemcpy(complete.data(), state.streaming_cue_bytes.get(),
                          complete.size(), cudaMemcpyDeviceToHost),
               "transport completed resident contact");
  cuda_require(cudaMemset(state.streaming_cue_meta.get(), 0,
                          state.streaming_cue_meta.bytes()),
               "consume completed resident contact");
  return complete;
}

#include "bcc32_cuda_adult_raw_cue_conditioning.inl"

// --frame-emit: after cue conditioning has built the subject field + composed a
// mode-4 answer, override motor_completion with a multi-clause plan drawn from the
// adult's own subject field + resident relation store (the substrate generator).
// Called AFTER condition_on_raw_cue (which has several early returns), so it always
// runs. A no-op unless the flag is set and usable relations exist.
// Public diagnostic entry point used by bcc32_cuda_adult_v1.cu after cue
// conditioning. It remains a small compatibility wrapper around the existing
// device composer, rather than an unreferenced helper hidden in the header.
inline void apply_frame_emit(AdultState& state) {
  if (!state.frame_emit) return;
  compose_frame_content_kernel<<<1u, kBlock>>>(
      state.subject_ids.get(), state.subject_weights.get(), state.subject_count.get(),
      state.online_conditioned_transitions.get(),
      state.online_conditioned_transition_conductance.get(),
      state.online_conditioned_transition_count, state.unit_lengths.get(),
      state.motor_context.get(), state.motor_completion.get());
  cuda_require(cudaGetLastError(), "launch frame-emit composer");
  cuda_require(cudaDeviceSynchronize(), "complete frame-emit composer");
}

inline void stage_motor_efference(AdultState& state,
                                  const std::uint8_t* host_bytes,
                                  std::uint32_t byte_count) {
  if (byte_count > kEfferenceTraceBytes) {
    throw std::runtime_error("motor efference trace exceeds resident capacity");
  }
  if (byte_count != 0u) {
    cuda_require(cudaMemcpy(state.efference_trace.get(), host_bytes, byte_count,
                            cudaMemcpyHostToDevice),
                 "stage resident motor efference bytes");
  }
  auto* efference_state = state.efference_state.get();
  std::uint32_t efference_byte_count = byte_count;
  void* efference_arguments[] = {&efference_state, &efference_byte_count};
  cuda_require(
      cudaLaunchKernel(
          reinterpret_cast<const void*>(stage_motor_efference_kernel),
          dim3{1u, 1u, 1u}, dim3{2u, 1u, 1u}, efference_arguments, 0u,
          nullptr),
      "launch resident motor efference staging");
  cuda_require(cudaGetLastError(), "launch resident motor efference staging");
  cuda_require(cudaDeviceSynchronize(), "complete resident motor efference staging");
}

inline bool appraise_reafferent_echo(AdultState& state,
                                     const std::uint8_t* host_bytes,
                                     std::uint32_t byte_count) {
  if (byte_count > kEfferenceTraceBytes) return false;
  DeviceArray<std::uint8_t> sensed(byte_count);
  DeviceArray<std::uint32_t> discharged(1u);
  if (byte_count != 0u) {
    cuda_require(cudaMemcpy(sensed.get(), host_bytes, byte_count,
                            cudaMemcpyHostToDevice),
                 "stage sensed reafferent bytes");
  }
  appraise_reafferent_echo_kernel<<<1u, kBlock>>>(
      sensed.get(), byte_count, state.efference_trace.get(),
      state.efference_state.get(), discharged.get());
  cuda_require(cudaGetLastError(), "launch resident reafference appraisal");
  cuda_require(cudaDeviceSynchronize(), "complete resident reafference appraisal");
  std::uint32_t host_discharged = 0u;
  cuda_require(cudaMemcpy(&host_discharged, discharged.get(),
                          sizeof(host_discharged), cudaMemcpyDeviceToHost),
               "read resident reafference decision");
  return host_discharged != 0u;
}

inline bool assimilate_reafferent_bytes(
    AdultState& state, const std::uint8_t* host_bytes, std::uint32_t byte_count,
    std::uint32_t seed = 0x9e3779b9u) {
  if (appraise_reafferent_echo(state, host_bytes, byte_count)) return false;
  assimilate_raw_bytes(state, host_bytes, byte_count, false, seed);
  return true;
}

inline bool appraise_interaction_shadow(AdultState& state,
                                        const std::uint8_t* host_bytes,
                                        std::uint32_t byte_count) {
  if (byte_count > kInteractionShadowBytes) {
    throw std::runtime_error("interaction shadow exceeds resident capacity");
  }
  DeviceArray<std::uint8_t> sensed(byte_count);
  DeviceArray<std::uint32_t> confirmed(1u);
  if (byte_count != 0u) {
    cuda_require(cudaMemcpy(sensed.get(), host_bytes, byte_count,
                            cudaMemcpyHostToDevice),
                 "stage interaction shadow bytes");
  }
  appraise_interaction_shadow_kernel<<<1u, kBlock>>>(
      sensed.get(), byte_count, state.interaction_shadow_trace.get(),
      state.interaction_shadow_state.get(), confirmed.get());
  cuda_require(cudaGetLastError(), "launch resident interaction appraisal");
  cuda_require(cudaDeviceSynchronize(), "complete resident interaction appraisal");
  std::uint32_t host_confirmed = 0u;
  cuda_require(cudaMemcpy(&host_confirmed, confirmed.get(),
                          sizeof(host_confirmed), cudaMemcpyDeviceToHost),
               "read resident interaction decision");
  return host_confirmed != 0u;
}

template <typename ResidentPromotion>
inline bool condition_on_interaction_bytes_with_promotion(
    AdultState& state, const std::uint8_t* host_bytes, std::uint32_t byte_count,
    ResidentPromotion promote, bool diagnostic = false,
    std::uint32_t seed = 0x9e3779b9u) {
  const bool confirmed = appraise_interaction_shadow(state, host_bytes, byte_count);
  if (confirmed) {
    promote(state, host_bytes, byte_count, seed);
  }
  condition_on_raw_cue(state, host_bytes, byte_count, diagnostic);
  return confirmed;
}

inline bool condition_on_interaction_bytes(
    AdultState& state, const std::uint8_t* host_bytes, std::uint32_t byte_count,
    bool diagnostic = false, std::uint32_t seed = 0x9e3779b9u) {
  return condition_on_interaction_bytes_with_promotion(
      state, host_bytes, byte_count,
      [](AdultState& promoted_state, const std::uint8_t* promoted_bytes,
         std::uint32_t promoted_count, std::uint32_t promoted_seed) {
        assimilate_raw_bytes(promoted_state, promoted_bytes, promoted_count,
                             false, promoted_seed);
      },
      diagnostic, seed);
}


inline void lesion_interaction_shadow(AdultState& state) {
  lesion_interaction_shadow_kernel<<<1u, kBlock>>>(
      state.interaction_shadow_trace.get(), state.interaction_shadow_state.get());
  cuda_require(cudaGetLastError(), "launch resident interaction shadow lesion");
  cuda_require(cudaDeviceSynchronize(), "complete resident interaction shadow lesion");
}

inline void lesion_efference_trace(AdultState& state) {
  lesion_efference_trace_kernel<<<1u, kBlock>>>(
      state.efference_trace.get(), state.efference_state.get());
  cuda_require(cudaGetLastError(), "launch resident efference lesion");
  cuda_require(cudaDeviceSynchronize(), "complete resident efference lesion");
}

inline void clear_raw_cue(AdultState& state) {
  cuda_require(cudaMemset(state.motor_context.get(), 0, state.motor_context.bytes()),
               "clear resident motor cue");
  cuda_require(cudaMemset(state.motor_completion.get(), 0, state.motor_completion.bytes()),
               "clear resident motor completion");
  if (state.proposition_cue_sequence_count.get() != nullptr)
    cuda_require(cudaMemset(state.proposition_cue_sequence_count.get(), 0,
                            state.proposition_cue_sequence_count.bytes()),
                 "clear ordered proposition cue");
  if (state.distributed_motor_enabled) {
    cuda_require(distributed_motor::clear_cue(distributed_motor_view(state)),
                 "clear distributed sensory field while retaining resident trajectory");
  }
}
