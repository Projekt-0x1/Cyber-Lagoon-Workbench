// Distributed sequence motor: autonomous generation, population-plan
// projection, opaque-unit motor planning, and lesion/mass-audit kernels.
//
// Included from bcc32_cuda_distributed_sequence_motor.cuh inside the
// bcc32_cuda_distributed_sequence_motor namespace, after the motor-channel
// scoring/selection helpers it calls (score_motor_channels,
// select_motor_channel, select_spontaneous_motor_channel,
// score_spontaneous_reactivation, select_top_population,
// accumulate_binding_candidates, select_sparse_successor, encode_pattern)
// and before the host-side *_async wrapper functions that launch these
// kernels. Pure textual extraction; no behavior change.

__global__ void generate_sequence_kernel(DistributedSequenceMotorView view,
                                         std::uint8_t* output,
                                         std::uint32_t output_capacity,
                                         std::uint32_t completion_iterations) {
  __shared__ std::uint32_t provisional_channel;
  __shared__ std::uint32_t path_index;
  __shared__ std::uint32_t next_count;
  __shared__ std::uint32_t emit_this_tick;
  __shared__ std::uint32_t reactivate_this_tick;
  __shared__ std::uint32_t spontaneous_channel;
  __shared__ unsigned long long selected_score;
  __shared__ unsigned long long runner_up_score;
  __shared__ unsigned long long selected_global_support;
  __shared__ unsigned long long maximum_global_support;
  __shared__ unsigned long long second_global_support;
  if (threadIdx.x == 0u)
    *view.emitted_count = 0u;
  __syncthreads();
  (void)completion_iterations;
  for (std::uint32_t tick = 0u; tick < output_capacity; ++tick) {
    score_motor_channels(view, view.current_active, *view.current_count,
                         view.cue_active, *view.cue_count);
    __syncthreads();
    if (threadIdx.x == 0u) {
      provisional_channel =
          select_motor_channel(view, &selected_score, &runner_up_score);
      reactivate_this_tick =
          provisional_channel == kInvalidCell && *view.cue_count == 0u &&
                  *view.initiative_energy == 0u && *view.quiet_ticks + 1u >= 32u
              ? 1u
              : 0u;
      spontaneous_channel = reactivate_this_tick != 0u
                                ? select_spontaneous_motor_channel(view)
                                : kInvalidCell;
    }
    __syncthreads();
    if (reactivate_this_tick != 0u && spontaneous_channel != kInvalidCell) {
      score_spontaneous_reactivation(view, spontaneous_channel);
      __syncthreads();
      if (threadIdx.x == 0u) {
        select_top_population(view, view.current_active, view.current_count);
        *view.local_generation_step = 0u;
        if (*view.current_count != 0u && *view.uncertainty_state < 18u)
          *view.uncertainty_state = 18u;
      }
      __syncthreads();
      score_motor_channels(view, view.current_active, *view.current_count,
                           view.cue_active, *view.cue_count);
      __syncthreads();
      if (threadIdx.x == 0u) {
        provisional_channel =
            select_motor_channel(view, &selected_score, &runner_up_score);
      }
    }
    __syncthreads();
    if (threadIdx.x == 0u) {
      selected_global_support = provisional_channel == kInvalidCell
                                    ? 0ull
                                    : view.motor_support[provisional_channel];
      maximum_global_support = 0ull;
      second_global_support = 0ull;
      for (std::uint32_t channel = 0u; channel < kMotorChannels; ++channel) {
        if (view.motor_support[channel] > maximum_global_support) {
          second_global_support = maximum_global_support;
          maximum_global_support = view.motor_support[channel];
        } else if (view.motor_support[channel] > second_global_support) {
          second_global_support = view.motor_support[channel];
        }
      }
      path_index = *view.generation_phase % kPathCapacity;
      emit_this_tick = 0u;
      const unsigned long long margin = selected_score - runner_up_score;
      if (provisional_channel == kInvalidCell ||
          (runner_up_score != 0ull && margin * 4ull < selected_score)) {
        const std::uint32_t raised = *view.uncertainty_state + 2u;
        *view.uncertainty_state = raised > 255u ? 255u : raised;
      } else if (*view.uncertainty_state != 0u) {
        *view.uncertainty_state -= 1u;
      }
      *view.quiet_ticks += 1u;
      if (*view.initiative_energy == 0u && *view.quiet_ticks >= 32u &&
          *view.uncertainty_state >= 16u && provisional_channel != kInvalidCell) {
        const std::uint32_t burst = 32u + *view.uncertainty_state;
        *view.initiative_energy = burst > 192u ? 192u : burst;
        *view.quiet_ticks = 0u;
      }
      const bool informative_onset = maximum_global_support == 0ull ||
          selected_global_support < maximum_global_support ||
          maximum_global_support * 4ull <= second_global_support * 5ull;
      if (provisional_channel != kInvalidCell && *view.initiative_energy != 0u &&
          (*view.emission_active != 0u || informative_onset) &&
          *view.emitted_count < output_capacity) {
        emit_this_tick = 1u;
        *view.emission_active = 1u;
        *view.initiative_energy -= 1u;
        if (*view.initiative_energy == 0u)
          *view.emission_active = 0u;
      }
    }
    __syncthreads();
    for (std::uint32_t slot = threadIdx.x; slot < view.active_width;
         slot += blockDim.x) {
      view.path_cells[static_cast<std::size_t>(path_index) * view.active_width + slot] =
          view.current_active[slot];
    }
    if (threadIdx.x == 0u) {
      view.path_phases[path_index] = *view.generation_phase;
      if (*view.path_size < kPathCapacity)
        *view.path_size += 1u;
      *view.generation_phase += 1u;
      *view.local_generation_step += 1u;
      if (emit_this_tick != 0u) {
        const std::uint32_t emitted = *view.emitted_count;
        output[emitted] = static_cast<std::uint8_t>(provisional_channel);
        view.output_tape[*view.output_tape_size % kPathCapacity] =
            static_cast<std::uint8_t>(provisional_channel);
        *view.output_tape_size += 1u;
        *view.emitted_count = emitted + 1u;
      }
    }
    __syncthreads();

    for (std::uint32_t bucket = threadIdx.x; bucket < kCandidateCapacity;
         bucket += blockDim.x) {
      view.candidate_cells[bucket] = kInvalidCell;
      view.candidate_scores[bucket] = 0ull;
    }
    __syncthreads();
    accumulate_binding_candidates(view, view.current_active, *view.current_count, 8ull);
    accumulate_binding_candidates(view, view.cue_active, *view.cue_count,
                                  *view.local_generation_step <= 1u ? 4ull : 1ull);
    const std::uint32_t retained_before_current =
        *view.path_size > 0u ? *view.path_size - 1u : 0u;
    const std::uint32_t recalled = retained_before_current < kPathRecallDepth
                                       ? retained_before_current
                                       : kPathRecallDepth;
    for (std::uint32_t age = 0u; age < recalled; ++age) {
      const std::uint32_t recalled_index =
          (path_index + kPathCapacity - 1u - age) % kPathCapacity;
      accumulate_binding_candidates(
          view,
          view.path_cells + static_cast<std::size_t>(recalled_index) * view.active_width,
          view.active_width, 1ull);
    }
    __syncthreads();
    if (threadIdx.x == 0u)
      next_count = select_sparse_successor(view);
    __syncthreads();
    if (next_count == 0u)
      break;
  }
}

__device__ inline void mark_population_plan_cell(DistributedSequenceMotorView view,
                                                 std::uint32_t cell,
                                                 std::uint32_t relevance,
                                                 std::uint32_t phase) {
  if (cell == kInvalidCell || cell >= view.population || relevance == 0u)
    return;
  const unsigned long long encoded =
      (static_cast<unsigned long long>(relevance) << 32u) |
      static_cast<unsigned long long>(0xffffffffu - phase);
  atomicMax(view.completion_scores + cell, encoded);
}

__global__ void gate_population_plan_projection_kernel(
    DistributedSequenceMotorView view, std::uint32_t* projection_state) {
  if (blockIdx.x != 0u || threadIdx.x != 0u)
    return;
  const std::uint32_t phase = *view.generation_phase;
  const std::uint32_t elapsed = phase - projection_state[0];
  const bool initial_completion = projection_state[0] == 0u && phase >= 8u;
  const bool motor_completion = *view.emitted_count != 0u && elapsed >= 16u;
  const bool autonomous_completion = elapsed >= 32u;
  const bool ready = initial_completion || motor_completion || autonomous_completion;
  projection_state[1] = ready ? 1u : 0u;
  if (ready)
    projection_state[0] = phase;
}

__global__ void build_population_plan_activity_kernel(
    DistributedSequenceMotorView view, const std::uint32_t* projection_state) {
  if (projection_state[1] == 0u)
    return;
  for (std::uint32_t slot = threadIdx.x; slot < view.active_width; slot += blockDim.x) {
    mark_population_plan_cell(view, view.cue_active[slot], 2u, 0u);
    mark_population_plan_cell(view, view.current_active[slot], 24u,
                              *view.generation_phase);
  }
  const std::uint32_t retained = *view.path_size < kPathRecallDepth
                                     ? *view.path_size
                                     : kPathRecallDepth;
  for (std::uint32_t age = 0u; age < retained; ++age) {
    const std::uint32_t index =
        (*view.generation_phase + kPathCapacity - 1u - age) % kPathCapacity;
    const std::uint32_t relevance = 20u - (age < 16u ? age : 16u);
    for (std::uint32_t slot = threadIdx.x; slot < view.active_width; slot += blockDim.x) {
      mark_population_plan_cell(
          view, view.path_cells[static_cast<std::size_t>(index) * view.active_width + slot],
          relevance, view.path_phases[index]);
    }
  }
}

__global__ void encode_opaque_unit_population_kernel(
    DistributedSequenceMotorView view, const std::uint32_t* unit_lengths,
    const std::uint32_t* packed_unit_bytes, std::uint32_t unit_words,
    std::uint32_t unit_begin, std::uint32_t unit_count,
    std::uint32_t* unit_population, std::uint32_t* unit_population_count = nullptr) {
  const std::uint32_t offset = blockIdx.x * blockDim.x + threadIdx.x;
  if (offset >= unit_count)
    return;
  const std::uint32_t unit = unit_begin + offset;
  const std::uint32_t length = unit_lengths[unit];
  std::uint32_t* output =
      unit_population + static_cast<std::size_t>(unit) * view.active_width;
  if (length == 0u || length > unit_words * sizeof(std::uint32_t)) {
    for (std::uint32_t slot = 0u; slot < view.active_width; ++slot)
      output[slot] = kInvalidCell;
    if (unit_population_count != nullptr)
      unit_population_count[unit] = 0u;
    return;
  }
  const auto* bytes = reinterpret_cast<const std::uint8_t*>(
      packed_unit_bytes + static_cast<std::size_t>(unit) * unit_words);
  const std::uint32_t formed_count =
      encode_pattern(nullptr, 0u, bytes, length - 1u, 0u, view, output);
  if (unit_population_count != nullptr)
    unit_population_count[unit] = formed_count;
}

__global__ void score_opaque_units_from_population_kernel(
    DistributedSequenceMotorView view, const std::uint32_t* unit_lengths,
    const std::uint32_t* unit_population,
    const std::uint32_t* unit_vitality, std::uint32_t unit_begin,
    std::uint32_t unit_count,
    const std::uint32_t* projection_state, unsigned long long* unit_activity,
    std::uint32_t* unit_phase) {
  if (projection_state[1] == 0u)
    return;
  const std::uint32_t offset = blockIdx.x * blockDim.x + threadIdx.x;
  if (offset >= unit_count)
    return;
  const std::uint32_t unit = unit_begin + offset;
  const std::uint32_t length = unit_lengths[unit];
  if (length < 2u || unit_vitality[unit] == 0u) {
    unit_activity[unit] = 0ull;
    unit_phase[unit] = 0xffffffffu;
    return;
  }
  unsigned long long score = 0ull;
  std::uint32_t earliest = 0xffffffffu;
  std::uint32_t matched_cells = 0u;
  const std::uint32_t* cells =
      unit_population + static_cast<std::size_t>(unit) * view.active_width;
  for (std::uint32_t slot = 0u; slot < view.active_width; ++slot) {
    const std::uint32_t cell = cells[slot];
    if (cell == kInvalidCell)
      continue;
    const unsigned long long activity = view.completion_scores[cell];
    const std::uint32_t relevance = static_cast<std::uint32_t>(activity >> 32u);
    if (relevance < 8u)
      continue;
    ++matched_cells;
    score += static_cast<unsigned long long>(relevance) * (1u + (slot % 5u));
    const std::uint32_t phase = 0xffffffffu - static_cast<std::uint32_t>(activity);
    if (phase < earliest)
      earliest = phase;
  }
  unit_activity[unit] = matched_cells != 0u ? score : 0ull;
  unit_phase[unit] = matched_cells != 0u ? earliest : 0xffffffffu;
}

__global__ void select_opaque_unit_plan_kernel(
    const std::uint32_t* unit_lengths, const unsigned long long* unit_activity,
    const std::uint32_t* unit_phase, std::uint32_t unit_count,
    std::uint32_t unit_begin,
    const std::uint32_t* projection_state,
    std::uint32_t* motor_context, std::uint32_t* motor_completion,
    std::uint32_t motor_capacity) {
  if (blockIdx.x != 0u)
    return;
  if (projection_state[1] == 0u) {
    if (threadIdx.x == 0u)
      motor_context[3] = 0u;
    return;
  }
  constexpr std::uint32_t kLocalCandidates = 4u;
  __shared__ unsigned long long candidate_scores[256u * kLocalCandidates];
  __shared__ std::uint32_t candidate_units[256u * kLocalCandidates];
  unsigned long long local_scores[kLocalCandidates]{};
  std::uint32_t local_units[kLocalCandidates];
  for (std::uint32_t index = 0u; index < kLocalCandidates; ++index)
    local_units[index] = kInvalidCell;
  for (std::uint32_t offset = threadIdx.x; offset < unit_count; offset += blockDim.x) {
    const std::uint32_t unit = unit_begin + offset;
    const unsigned long long score = unit_activity[unit];
    for (std::uint32_t position = 0u; position < kLocalCandidates; ++position) {
      if (score > local_scores[position] ||
          (score == local_scores[position] && score != 0ull && unit < local_units[position])) {
        for (std::uint32_t shift = kLocalCandidates - 1u; shift > position; --shift) {
          local_scores[shift] = local_scores[shift - 1u];
          local_units[shift] = local_units[shift - 1u];
        }
        local_scores[position] = score;
        local_units[position] = unit;
        break;
      }
    }
  }
  for (std::uint32_t index = 0u; index < kLocalCandidates; ++index) {
    const std::uint32_t shared = threadIdx.x * kLocalCandidates + index;
    candidate_scores[shared] = local_scores[index];
    candidate_units[shared] = local_units[index];
  }
  __syncthreads();
  if (threadIdx.x != 0u)
    return;
  for (std::uint32_t index = 0u; index < 16u; ++index)
    motor_context[index] = 0u;
  unsigned long long strongest = 0ull;
  for (std::uint32_t index = 0u; index < 256u * kLocalCandidates; ++index)
    if (candidate_scores[index] > strongest)
      strongest = candidate_scores[index];
  if (strongest == 0ull)
    return;
  const unsigned long long threshold = strongest / 2u + 1u;
  std::uint32_t selected[kSurfacePlanMaxUnits];
  std::uint32_t selected_count = 0u;
  const std::uint32_t capacity = motor_capacity < kSurfacePlanMaxUnits
                                     ? motor_capacity
                                     : kSurfacePlanMaxUnits;
  for (; selected_count < capacity; ++selected_count) {
    unsigned long long best_score = 0ull;
    std::uint32_t best = kInvalidCell;
    for (std::uint32_t index = 0u; index < 256u * kLocalCandidates; ++index) {
      const std::uint32_t unit = candidate_units[index];
      const unsigned long long score = candidate_scores[index];
      if (unit == kInvalidCell || score < threshold)
        continue;
      bool used = false;
      for (std::uint32_t prior = 0u; prior < selected_count; ++prior)
        used = used || selected[prior] == unit;
      if (!used && (score > best_score ||
                    (score == best_score && unit < best))) {
        best_score = score;
        best = unit;
      }
    }
    if (best == kInvalidCell)
      break;
    selected[selected_count] = best;
  }
  for (std::uint32_t left = 1u; left < selected_count; ++left) {
    const std::uint32_t value = selected[left];
    std::uint32_t right = left;
    while (right != 0u &&
           (unit_phase[value] < unit_phase[selected[right - 1u]] ||
            (unit_phase[value] == unit_phase[selected[right - 1u]] &&
             unit_lengths[value] > unit_lengths[selected[right - 1u]]))) {
      selected[right] = selected[right - 1u];
      --right;
    }
    selected[right] = value;
  }
  for (std::uint32_t index = 0u; index < selected_count; ++index)
    motor_completion[index] = selected[index];
  motor_context[0] = selected_count != 0u ? 1u : 0u;
  motor_context[1] = selected_count != 0u ? selected[0] : 0u;
  motor_context[2] = static_cast<std::uint32_t>(strongest > 0xffffffffull
                                                    ? 0xffffffffull
                                                    : strongest);
  motor_context[3] = selected_count;
  motor_context[5] = 6u;
  motor_context[12] = selected_count;
  motor_context[15] = selected_count != 0u ? 1u : 0u;
}

__global__ void mark_lesion_kernel(std::uint32_t* enabled, std::uint32_t population,
                                   std::uint32_t modulo, std::uint32_t phase) {
  const std::uint32_t cell = blockIdx.x * blockDim.x + threadIdx.x;
  if (cell < population && (cell % modulo) == phase)
    enabled[cell] = 0u;
}

__global__ void reclaim_lesioned_mass_kernel(DistributedSequenceMotorView view) {
  const std::size_t motor_count = motor_elements(view.population);
  for (std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < motor_count; index += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
    const std::uint32_t cell = static_cast<std::uint32_t>(index / kMotorChannels);
    if (view.enabled[cell] == 0u) {
      const std::uint32_t released = atomicExch(view.motor_mass + index, 0u);
      if (released != 0u)
        atomicAdd(view.free_mass, static_cast<unsigned long long>(released));
    }
  }
  const std::size_t binding_count = binding_elements(view.population);
  for (std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < binding_count; index += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
    const std::uint32_t source = static_cast<std::uint32_t>(index / kBindingSlots);
    const std::uint32_t key = view.binding_keys[index];
    const std::uint32_t target =
        key == 0u ? kInvalidCell : unbind_target(source, key, view.population);
    if (view.enabled[source] == 0u ||
        (target != kInvalidCell && view.enabled[target] == 0u)) {
      const std::uint32_t released = atomicExch(view.binding_mass + index, 0u);
      if (released != 0u)
        atomicAdd(view.free_mass, static_cast<unsigned long long>(released));
    }
  }
}

__global__ void audit_represented_mass_kernel(DistributedSequenceMotorView view,
                                              unsigned long long* result) {
  const std::size_t motor_count = motor_elements(view.population);
  if (blockIdx.x == 0u && threadIdx.x == 0u)
    atomicAdd(result, atomicAdd(view.free_mass, 0ull));
  for (std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < motor_count; index += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
    const std::uint32_t amount = view.motor_mass[index];
    if (amount != 0u)
      atomicAdd(result, static_cast<unsigned long long>(amount));
  }
  const std::size_t binding_count = binding_elements(view.population);
  for (std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < binding_count; index += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
    const std::uint32_t amount = view.binding_mass[index];
    if (amount != 0u)
      atomicAdd(result, static_cast<unsigned long long>(amount));
  }
}
