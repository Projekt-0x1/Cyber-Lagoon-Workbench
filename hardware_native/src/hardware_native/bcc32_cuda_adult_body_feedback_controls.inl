__global__ void stage_motor_efference_kernel(
    ResidentEfferenceState* state, std::uint32_t byte_count) {
  if (blockIdx.x != 0u || threadIdx.x != 0u) return;
  state->trace_bytes = byte_count;
  state->valid = byte_count != 0u ? 1u : 0u;
}

__global__ void appraise_reafferent_echo_kernel(
    const std::uint8_t* sensed_bytes, std::uint32_t byte_count,
    const std::uint8_t* efference_trace, ResidentEfferenceState* state,
    std::uint32_t* discharged) {
  __shared__ std::uint32_t mismatch;
  if (threadIdx.x == 0u) {
    mismatch = state->valid == 0u || state->trace_bytes != byte_count;
    discharged[0] = 0u;
  }
  __syncthreads();
  for (std::uint32_t index = threadIdx.x; index < byte_count;
       index += blockDim.x) {
    if (sensed_bytes[index] != efference_trace[index]) atomicExch(&mismatch, 1u);
  }
  __syncthreads();
  if (threadIdx.x != 0u) return;
  if (mismatch == 0u) {
    ++state->exact_discharges;
    state->discharged_bytes += byte_count;
    discharged[0] = 1u;
  } else {
    ++state->mismatched_echoes;
    state->mismatch_bytes += byte_count;
  }
  state->trace_bytes = 0u;
  state->valid = 0u;
}

__global__ void lesion_efference_trace_kernel(
    std::uint8_t* efference_trace, ResidentEfferenceState* state) {
  for (std::uint32_t index = threadIdx.x; index < kEfferenceTraceBytes;
       index += blockDim.x) {
    efference_trace[index] = 0u;
  }
  __syncthreads();
  if (threadIdx.x != 0u) return;
  const std::uint32_t lesions = state->lesion_count + 1u;
  *state = ResidentEfferenceState{};
  state->lesion_count = lesions;
}

__global__ void appraise_interaction_shadow_kernel(
    const std::uint8_t* sensed_bytes, std::uint32_t byte_count,
    std::uint8_t* shadow_trace, ResidentInteractionShadowState* state,
    std::uint32_t* confirmed) {
  __shared__ std::uint32_t mismatch;
  if (threadIdx.x == 0u) {
    mismatch = state->valid == 0u || state->trace_bytes != byte_count;
    confirmed[0] = 0u;
  }
  __syncthreads();
  for (std::uint32_t index = threadIdx.x; index < byte_count;
       index += blockDim.x) {
    if (state->valid != 0u && index < state->trace_bytes &&
        sensed_bytes[index] != shadow_trace[index]) {
      atomicExch(&mismatch, 1u);
    }
  }
  __syncthreads();
  if (mismatch == 0u) {
    if (threadIdx.x == 0u) {
      ++state->exact_confirmations;
      state->confirmed_bytes += byte_count;
      state->trace_bytes = 0u;
      state->valid = 0u;
      confirmed[0] = 1u;
    }
    return;
  }
  for (std::uint32_t index = threadIdx.x; index < byte_count;
       index += blockDim.x) {
    shadow_trace[index] = sensed_bytes[index];
  }
  __syncthreads();
  if (threadIdx.x != 0u) return;
  state->replaced_contacts += state->valid != 0u ? 1u : 0u;
  ++state->staged_contacts;
  state->trace_bytes = byte_count;
  state->valid = byte_count != 0u ? 1u : 0u;
}

__global__ void lesion_interaction_shadow_kernel(
    std::uint8_t* shadow_trace, ResidentInteractionShadowState* state) {
  for (std::uint32_t index = threadIdx.x; index < kInteractionShadowBytes;
       index += blockDim.x) {
    shadow_trace[index] = 0u;
  }
  __syncthreads();
  if (threadIdx.x != 0u) return;
  const std::uint32_t lesions = state->lesion_count + 1u;
  *state = ResidentInteractionShadowState{};
  state->lesion_count = lesions;
}

// Body-surface signal conditioning: collapse every control/whitespace byte
// (<0x20, plus DEL 0x7f) to a single space before any statistics, boundary
// discovery, or unit formation. PDF page/line breaks otherwise leave a word
// ending one line fused to the word starting the next (the discovered single
// boundary is space, and a newline is not a boundary), producing single units
// like "notcompletely" / "Africandata". This is a generic surface property --
// the sensor sees a normalized byte stream -- not language knowledge: the
// boundary byte itself is still discovered emergently downstream.
