__device__ void apply_physical_event(DeviceState* state,
                                     const RawPhysicalIntervention& event,
                                     std::uint64_t sequence) {
  if (state->intervention_active != 0u)
    restore_transient_coupling(state);
  state->intervention = event;
  state->intervention_sequence = sequence;
  apply_field_pulse(state, event);
  state->intervention_active =
      (event.duration_epochs != 0u &&
       (event.matter_q8 < 256u || event.coupling_q8 < 256u))
          ? 1u
          : 0u;
  state->intervention_remaining = event.duration_epochs;
  const std::uint32_t matter = event.matter_q8 > 256u ? 256u : event.matter_q8;
  const std::uint32_t coupling =
      event.coupling_q8 > 256u ? 256u : event.coupling_q8;
  const std::uint32_t removed = 256u - matter;
  const std::uint32_t cut = 256u - coupling;
  const std::uint32_t center = event.center % state->cell_count;
  state->removed_matter_q8_sum = 0u;
  state->cut_coupling_q8_sum = 0u;
  for (std::uint32_t index = 0u; index < state->cell_count; ++index) {
    const std::uint32_t distance = index > center ? index - center : center - index;
    if (distance > event.radius) continue;
    state->removed_matter_q8_sum += removed;
    state->cut_coupling_q8_sum += cut;
    if (matter < 256u) {
      state->tissue_matter_q8[index] = static_cast<std::uint32_t>(
          static_cast<std::uint64_t>(state->tissue_matter_q8[index]) *
          matter / 256u);
      state->state[index] = clamp_state(
          static_cast<std::int64_t>(state->state[index]) * matter / 256u);
      state->trace[index] = clamp_trace(
          static_cast<std::int64_t>(state->trace[index]) * matter / 256u);
    }
  }
  adaptive::apply_damage(&state->adaptive, center, event.radius, removed,
                         event.repair_q8, state->cell_count);
  // Physical damage attenuates packet/receptor matter in the same local
  // aperture.  Recovery is later performed by the resident repair reserve;
  // the host never restores a chosen source or relation.
  for (std::uint32_t index = 0u; index < kFieldSourceCapacity; ++index) {
    auto& field = state->field_sources[index];
    if (field.live == 0u) continue;
    const std::uint32_t distance = field_distance(field.anchor, center);
    if (distance > event.radius) continue;
    if (removed > field.damage_q8) field.damage_q8 = removed;
    const std::uint64_t reserve =
        static_cast<std::uint64_t>(field.repair_reserve_q8) + event.repair_q8;
    field.repair_reserve_q8 = static_cast<std::uint32_t>(
        reserve > static_cast<std::uint64_t>(kFieldFluxLimit)
            ? kFieldFluxLimit
            : reserve);
    field.flux_q8 = clamp_field_flux(
        static_cast<std::int64_t>(field.flux_q8) * matter / 256u);
    field.receptor_q8 = clamp_field_flux(
        static_cast<std::int64_t>(field.receptor_q8) * matter / 256u);
    field.packet_history_flux_q8 = clamp_field_flux(
        static_cast<std::int64_t>(field.packet_history_flux_q8) * matter /
        256u);
    field.packet_history_polarity_q8 = clamp_field_flux(
        static_cast<std::int64_t>(field.packet_history_polarity_q8) * matter /
        256u);
    field.packet_braid_q8 = clamp_field_flux(
        static_cast<std::int64_t>(field.packet_braid_q8) * matter / 256u);
    field.packet_alignment_q8 = clamp_field_flux(
        static_cast<std::int64_t>(field.packet_alignment_q8) * matter /
        256u);
    if (matter == 0u) {
      field.packet_history_age = kFieldPacketHistoryExpired;
      field.packet_history_updates = 0u;
    }
  }
  if (state->intervention_active != 0u && coupling < 256u) {
    for (std::uint32_t index = 0u; index < state->cell_count; ++index) {
      const std::uint32_t distance =
          index > center ? index - center : center - index;
      if (distance <= event.radius) state->tissue_coupling_q8[index] = coupling;
    }
  }
  std::uint64_t lanes[4]{0x510e527fade682d1ull, 0x9b05688c2b3e6c1full,
                         0x1f83d9abfb41bd6bull, 0x5be0cd19137e2179ull};
  lanes[0] = mix64(lanes[0], sequence);
  lanes[1] = mix64(lanes[1], event.center);
  lanes[2] = mix64(lanes[2], event.radius);
  lanes[3] = mix64(lanes[3], event.matter_q8);
  lanes[0] = mix64(lanes[0], event.coupling_q8);
  lanes[1] = mix64(lanes[1], event.duration_epochs);
  lanes[2] = mix64(lanes[2], static_cast<std::uint32_t>(event.flux_q8));
  lanes[3] = mix64(lanes[3], static_cast<std::uint32_t>(event.polarity_q8));
  lanes[0] = mix64(lanes[0], static_cast<std::uint32_t>(event.displacement));
  lanes[1] = mix64(lanes[1], event.transport_q8);
  lanes[2] = mix64(lanes[2], event.diffusion_q8);
  lanes[3] = mix64(lanes[3], event.receptor_gain_q8);
  lanes[0] = mix64(lanes[0], event.repair_q8);
  digest_finish(&state->intervention_digest, lanes, sizeof(event));
}

__device__ void consume_physical(DeviceState* state, PhysicalIngress* ingress) {
  const std::uint64_t published = load_system_u64(&ingress->published);
  if (published <= state->intervention_sequence) return;
  apply_physical_event(state, ingress->event, published);
  cuda::atomic_ref<std::uint64_t, cuda::thread_scope_system> consumed(
      ingress->consumed);
  consumed.store(published, cuda::memory_order_release);
}

__device__ void age_physical(DeviceState* state) {
  if (state->intervention_active == 0u ||
      state->intervention_remaining == 0xffffffffu)
    return;
  if (state->intervention_remaining != 0u) --state->intervention_remaining;
  if (state->intervention_remaining == 0u) {
    restore_transient_coupling(state);
    state->intervention_active = 0u;
  }
}

__device__ void publish_egress(const DeviceState* state, EgressRing* egress) {
  EgressSlot* slot = &egress->slots[state->tick % kEgressSlots];
  cuda::atomic_ref<std::uint32_t, cuda::thread_scope_system> slot_state(
      slot->state);
  std::uint32_t expected = kEgressReady;
  if (!slot_state.compare_exchange_strong(expected, kEgressWriting,
                                          cuda::memory_order_acq_rel,
                                          cuda::memory_order_acquire))
    return;
  slot->generation = state->tick;
  slot->energy = state->energy;
  slot->host_bootstrap_launches = state->host_bootstrap_launches;
  slot->device_epochs = state->device_epochs;
  slot->receipt = state->receipt;
  for (std::uint32_t index = 0u; index < state->cell_count; ++index)
    slot->actions[index] = state->actions[index];
  slot->language_frame = state->language.frame;
  slot->language_contacts = state->language.learned_contacts;
  slot->language_recruited_cells = state->language.recruited_cells;
  slot->language_reused_cells = state->language.reused_cells;
  slot->language_strengthened_edges = state->language.strengthened_edges;
  slot_state.store(kEgressReady, cuda::memory_order_release);
  cuda::atomic_ref<std::uint64_t, cuda::thread_scope_system> published(
      egress->published);
  published.store(state->tick, cuda::memory_order_release);
}

// One bounded causal epoch. The graph tail launch below is the only
// continuation mechanism; each invocation remains far below display-watchdog
// duration and host launch count remains one.
__global__ void autonomous_epoch_kernel(
    DeviceState* state, IngressRing* ingress, PhysicalIngress* physical,
    EgressRing* egress,
    Lifecycle* lifecycle, std::uint64_t founder, std::uint32_t cell_count,
    DeviceDigest sealed, DeviceDigest law, DeviceDigest image) {
  if (blockIdx.x != 0u || threadIdx.x != 0u || state == nullptr ||
      ingress == nullptr || physical == nullptr || egress == nullptr ||
      lifecycle == nullptr)
    return;
  // Initialization and completed causal epochs are distinct. The canonical
  // F-owned path returns early once to launch its first F tick, so using
  // device_epochs as the initialization sentinel would reinitialize on every
  // return and reset the handshake forever.
  if (state->host_bootstrap_launches == 0u)
    initialize_state(state, founder, cell_count, sealed, law, image);
  cuda::atomic_ref<std::uint32_t, cuda::thread_scope_system> shutdown(
      lifecycle->shutdown);
  cuda::atomic_ref<std::uint32_t, cuda::thread_scope_system> stopped(
      lifecycle->stopped);
  if (shutdown.load(cuda::memory_order_acquire) != 0u) {
    stopped.store(1u, cuda::memory_order_release);
    return;
  }
  if (state->f_owned_clock != 0u) {
    if (state->ordinary_f.publication == nullptr ||
        state->ordinary_f.completed_ticks == nullptr ||
        state->ordinary_f.fault == nullptr) {
      cuda::atomic_ref<std::uint32_t, cuda::thread_scope_system> fault(
          lifecycle->continuation_fault);
      fault.store(static_cast<std::uint32_t>(
                      ContinuationFault::ordinary_f_handle),
                  cuda::memory_order_release);
      stopped.store(1u, cuda::memory_order_release);
      return;
    }
    bool f_tick_ready = false;
    if (state->expected_f_tick == 0u) {
      // Ordinary-F is queued first, then the root self-tail is queued. CUDA
      // drains tail launches in order, so the next root invocation regains
      // control only after this F child has committed its completion clock.
      state->expected_f_tick =
          ordinary_f::completed_ticks_acquire(state->ordinary_f) + 1u;
      const cudaError_t launch = ordinary_f::request_forward_tick(
          state->ordinary_f);
      if (launch != cudaSuccess) {
        cuda::atomic_ref<std::uint32_t, cuda::thread_scope_system> fault(
            lifecycle->continuation_fault);
        fault.store(continuation_cuda_fault(
                        ContinuationFault::ordinary_f_launch_base, launch),
                    cuda::memory_order_release);
        stopped.store(1u, cuda::memory_order_release);
        return;
      }
    } else {
      const std::uint32_t f_fault = *state->ordinary_f.fault;
      if (f_fault != static_cast<std::uint32_t>(ordinary_f::Fault::none)) {
        cuda::atomic_ref<std::uint32_t, cuda::thread_scope_system> fault(
            lifecycle->continuation_fault);
        fault.store(static_cast<std::uint32_t>(
                        ContinuationFault::ordinary_f_fault_base) |
                        f_fault,
                    cuda::memory_order_release);
        stopped.store(1u, cuda::memory_order_release);
        return;
      }
      const std::uint64_t completed =
          ordinary_f::completed_ticks_acquire(state->ordinary_f);
      if (completed < state->expected_f_tick) {
        const cudaError_t poll = cudaGraphLaunch(
            cudaGetCurrentGraphExec(), cudaStreamGraphTailLaunch);
        if (poll != cudaSuccess) {
          cuda::atomic_ref<std::uint32_t, cuda::thread_scope_system> fault(
              lifecycle->continuation_fault);
          fault.store(continuation_cuda_fault(
                          ContinuationFault::root_tail_launch_base, poll),
                      cuda::memory_order_release);
          stopped.store(1u, cuda::memory_order_release);
        }
        return;
      }
      f_tick_ready = true;
    }
    if (!f_tick_ready) {
      const cudaError_t poll = cudaGraphLaunch(
          cudaGetCurrentGraphExec(), cudaStreamGraphTailLaunch);
      if (poll != cudaSuccess) {
        cuda::atomic_ref<std::uint32_t, cuda::thread_scope_system> fault(
            lifecycle->continuation_fault);
        fault.store(continuation_cuda_fault(
                        ContinuationFault::root_tail_launch_base, poll),
                    cuda::memory_order_release);
        stopped.store(1u, cuda::memory_order_release);
      }
      return;
    }
    if (ordinary_f::publish_current_world(state->ordinary_f) != cudaSuccess) {
      cuda::atomic_ref<std::uint32_t, cuda::thread_scope_system> fault(
          lifecycle->continuation_fault);
      fault.store(static_cast<std::uint32_t>(
                      ContinuationFault::ordinary_f_publish),
                  cuda::memory_order_release);
      stopped.store(1u, cuda::memory_order_release);
      return;
    }
    const ordinary_f::OrdinaryFPublication publication =
        *state->ordinary_f.publication;
    if (publication.fault !=
        static_cast<std::uint32_t>(ordinary_f::Fault::none)) {
      cuda::atomic_ref<std::uint32_t, cuda::thread_scope_system> fault(
          lifecycle->continuation_fault);
      fault.store(static_cast<std::uint32_t>(
                      ContinuationFault::ordinary_f_fault_base) |
                      publication.fault,
                  cuda::memory_order_release);
      stopped.store(1u, cuda::memory_order_release);
      return;
    }
    if (publication.completed_ticks != state->expected_f_tick) {
      cuda::atomic_ref<std::uint32_t, cuda::thread_scope_system> fault(
          lifecycle->continuation_fault);
      fault.store(static_cast<std::uint32_t>(
                      ContinuationFault::ordinary_f_receipt_tick),
                  cuda::memory_order_release);
      stopped.store(1u, cuda::memory_order_release);
      return;
    }
    if (ordinary_f::completed_ticks_acquire(state->ordinary_f) !=
        publication.completed_ticks) {
      cuda::atomic_ref<std::uint32_t, cuda::thread_scope_system> fault(
          lifecycle->continuation_fault);
      fault.store(static_cast<std::uint32_t>(
                      ContinuationFault::ordinary_f_receipt_clock),
                  cuda::memory_order_release);
      stopped.store(1u, cuda::memory_order_release);
      return;
    }
    state->tick = publication.completed_ticks;
    state->phase = publication.generation;
    state->expected_f_tick = 0u;
  }
  if (!consume_ingress(state, ingress, lifecycle)) {
    stopped.store(1u, cuda::memory_order_release);
    return;
  }
  consume_physical(state, physical);
  age_physical(state);
  ++state->device_epochs;
    language::contact(&state->language, state->contact, state->contact_count,
                      state->contact_revision);
    age_prediction_eligibility(state);
    compare_and_publish_prediction(state);
    process_sparse_source_joint(state);
    refresh_joint_expression(state);
    advance_field_ecology(state);
    adaptive::record_contact_credit(
        &state->adaptive, state->contact, state->contact_count,
        state->founder, state->cell_count, state->contact_revision);
    adaptive::finish_epoch(
        &state->adaptive, state->state, state->trace, state->resource,
        state->reafference, state->tissue_matter_q8, state->cell_count,
        state->field_metrics.response_l1, state->field_metrics.residual_l1,
        state->field_packet_braid_sum, state->field_damage_q8,
        state->contact_count, state->contact_revision);
    adaptive::adapt_routes(
        &state->adaptive, &state->prediction[0u][0u],
        &state->credit_weight[0u][0u], &state->eligibility_age[0u][0u],
        state->cell_count, state->contact_count == 0u);
    adaptive::adapt_connectivity(
        &state->adaptive, state->state, state->trace, state->resource,
        state->tissue_matter_q8, state->tissue_coupling_q8,
        state->cell_count, state->contact_count != 0u);
    adaptive::repair_tissue(&state->adaptive, state->tissue_matter_q8,
                            state->resource, state->cell_count);
    const std::int64_t learned_joint_bias =
        sparse_joint_motor_bias(state);
    const bool quiet_contact = state->contact_count == 0u;
    const bool use_reafference =
        quiet_contact && state->reafference_pending != 0u;
    for (std::uint32_t index = 0u; index < state->cell_count; ++index) {
      const std::uint32_t left =
          (index + state->cell_count - 1u) % state->cell_count;
      const std::uint32_t right = (index + 1u) % state->cell_count;
      const std::int64_t external_pressure = raw_pressure(state, index);
      const std::int64_t echo_pressure = use_reafference
                                             ? state->reafference[index] / 2
                                             : 0;
      const std::int64_t pressure = external_pressure + echo_pressure;
      const std::uint32_t route = prediction_route(state, index);
      const std::int64_t predicted_pressure =
          state->contact_count == 0u
              ? 0
              : state->prediction[route][index];
      const std::int64_t drive =
          static_cast<std::int64_t>(state->state[index]) * 2 +
          state->state[left] + state->state[right] + pressure * 2 +
          predicted_pressure * 2 +
          state->credit_weight[route][index] / 2 +
          state->trace[index] / 8 + learned_joint_bias +
          state->field_response[index] / 8 + state->field_residual[index] / 32 +
          state->field_braid_drive[index] / 8 +
          state->field_alignment_drive[index] / 8 +
          adaptive::edge_drive(
              &state->adaptive, state->state, state->tissue_matter_q8,
              state->tissue_coupling_q8, index, state->cell_count);
      const std::int64_t activity_unbounded =
          static_cast<std::int64_t>(abs_value(state->state[index])) +
          (static_cast<std::int64_t>(abs_value(state->state[left])) +
           abs_value(state->state[right])) /
              2;
      const std::int64_t activity =
          activity_unbounded < 512 ? activity_unbounded : 512;
      const std::int64_t signed_drive =
          pressure * (activity + 64) / 128;
      const std::int64_t neighbour_mean =
          (static_cast<std::int64_t>(state->trace[left]) +
           state->trace[right]) /
          2;
      const std::int64_t competition_unbounded =
          (static_cast<std::int64_t>(state->trace[index]) - neighbour_mean) /
          32;
      const std::int64_t competition =
          max_i64(-64, min_i64(64, competition_unbounded));
      const std::int32_t edge_drive = adaptive::edge_drive(
          &state->adaptive, state->state, state->tissue_matter_q8,
          state->tissue_coupling_q8, index, state->cell_count);
      const std::int64_t resource_cost =
          abs_value(static_cast<std::int32_t>(signed_drive)) / 256 +
          abs_value(edge_drive) / 512;
      const std::int64_t resource_recovery = pressure == 0 ? 8 : 2;
      const std::int64_t learned_drive =
          state->resource[index] > 0 ? signed_drive : 0;
      state->trace_next[index] = clamp_trace(
          static_cast<std::int64_t>(state->trace[index]) + learned_drive +
          decay_to_zero(state->trace[index]) - competition);
      state->resource[index] = clamp_resource(
          static_cast<std::int64_t>(state->resource[index]) +
          resource_recovery - resource_cost);
      state->next[index] = clamp_state(drive / 5);
    }
    for (std::uint32_t index = 0u; index < state->cell_count; ++index) {
      state->trace[index] = state->trace_next[index];
    }
    state->energy = 0u;
    for (std::uint32_t index = 0u; index < state->cell_count; ++index) {
      state->state[index] = state->next[index];
      const std::uint32_t route = prediction_route(state, index);
      const std::int64_t predicted_pressure =
          state->contact_count == 0u ? 0 : state->prediction[route][index];
      const std::int64_t resident_bias =
          state->credit_weight[route][index] +
          adaptive::consolidated_bias(&state->adaptive, index, route);
      const std::int32_t recurrent_edge_drive = adaptive::edge_drive(
          &state->adaptive, state->state, state->tissue_matter_q8,
          state->tissue_coupling_q8, index, state->cell_count);
      const std::int64_t motor = static_cast<std::int64_t>(state->state[index]) +
                                 state->trace[index] / 4 + predicted_pressure +
                                 resident_bias + state->field_response[index] / 8 +
                                 state->field_braid_drive[index] / 4 +
                                 state->field_alignment_drive[index] / 4 +
                                 recurrent_edge_drive;
      const std::int32_t bounded_motor = clamp_state(motor);
      const std::uint32_t magnitude = static_cast<std::uint32_t>(
          bounded_motor < 0 ? -static_cast<std::int64_t>(bounded_motor)
                            : bounded_motor);
      state->energy += magnitude +
                       static_cast<std::uint32_t>(abs_value(state->trace[index]));
      // Transitional legacy authority remains explicit until an admitted raw
      // frame is demonstrated to reach F and change its committed motor
      // boundary. The F publication snapshot is observational only here.
      state->actions[index] =
          (magnitude & 0xffffu) |
          (static_cast<std::uint32_t>(bounded_motor < 0) << 31u);
      if (!quiet_contact)
        state->reafference[index] = clamp_state(bounded_motor / 2);
    }
    if (use_reafference) state->reafference_pending = 0u;
    if (!quiet_contact) state->reafference_pending = 1u;
    if (state->joint_route != 0u)
      state->structural_focus_cell = static_cast<std::uint32_t>(
          state->joint_route % state->cell_count);
    update_structural_receipt(state);
    // Language is another resident motor population. Its byte trajectory
    // re-enters the same body state instead of bypassing the organism through
    // a host renderer.
    for (std::uint32_t index = 0u;
         index < state->cell_count && index < state->language.frame.length;
         ++index) {
      const std::int32_t byte_pressure =
          static_cast<std::int32_t>(state->language.frame.bytes[index]) - 127;
      state->reafference[index] = clamp_state(
          static_cast<std::int64_t>(state->reafference[index]) + byte_pressure);
    }
    if (state->f_owned_clock == 0u) {
      ++state->tick;
      state->phase = (state->phase + 1u) & 7u;
    }
    DeviceDigest input{};
    DeviceDigest output{};
    digest_words(state->contact, state->contact_count, &input);
    digest_state(state, &output);
    const DeviceDigest predecessor = state->receipt.commitment;
    state->receipt.tick = state->tick;
    state->receipt.phase = state->phase;
    state->receipt.contact_sequence = state->contact_revision;
    state->receipt.sealed_execution = state->sealed_execution;
    state->receipt.law = state->law;
    state->receipt.image = state->image;
    state->receipt.input = input;
    state->receipt.output = output;
  state->receipt.predecessor = predecessor;
    state->receipt.joint_output = state->joint_output;
    state->receipt.joint_residual = state->joint_residual;
    state->receipt.joint_output_valid = state->joint_output_valid;
    state->receipt.joint_route = state->joint_route;
    state->receipt.field_response_l1 = state->field_metrics.response_l1;
    state->receipt.field_residual_l1 = state->field_metrics.residual_l1;
    state->receipt.field_peak_response = state->field_metrics.peak_response;
    state->receipt.field_peak_residual = state->field_metrics.peak_residual;
    state->receipt.field_active_routes = state->field_metrics.active_routes;
    state->receipt.field_contact_routes = state->field_metrics.contact_routes;
    state->receipt.field_responsive_routes = state->field_metrics.responsive_routes;
    state->receipt.field_owner_mix = state->field_metrics.owner_mix;
    state->receipt.developmental_pressure_l1 =
        state->field_metrics.developmental_pressure_l1;
    state->receipt.functional_horizon = state->field_metrics.functional_horizon;
    state->receipt.functional_integration =
        state->field_metrics.functional_integration;
    state->receipt.functional_delay = state->field_metrics.functional_delay;
    state->receipt.functional_spatial_tv_l1 =
        state->field_metrics.functional_spatial_tv_l1;
    state->receipt.structure = state->structure;
    state->receipt.intervention = state->intervention_digest;
    state->receipt.intervention_sequence = state->intervention_sequence;
    state->receipt.structural_focus_cell = state->structural_focus_cell;
    state->receipt.removed_matter_q8_sum = state->removed_matter_q8_sum;
    state->receipt.cut_coupling_q8_sum = state->cut_coupling_q8_sum;
    state->receipt.structural_support_q8 = state->structural_support_q8;
    state->receipt.effective_field_response_l1 =
        state->effective_field_response_l1;
    state->receipt.intervention_active = state->intervention_active;
    state->receipt.field_source_count = state->field_source_count;
    state->receipt.field_supported_source_count =
        state->field_supported_source_count;
    state->receipt.field_receptor_count = 0u;
    for (std::uint32_t index = 0u; index < kFieldSourceCapacity; ++index)
      if (state->field_sources[index].live != 0u &&
          state->field_sources[index].receptor_q8 != 0)
        ++state->receipt.field_receptor_count;
    state->receipt.field_packet_flux_l1 = 0;
    state->receipt.field_polarity_l1 = 0;
    for (std::uint32_t index = 0u; index < kFieldSourceCapacity; ++index) {
      if (state->field_sources[index].live == 0u) continue;
      state->receipt.field_packet_flux_l1 +=
          abs_value(state->field_sources[index].flux_q8);
      state->receipt.field_polarity_l1 +=
          abs_value(state->field_sources[index].polarity_q8);
    }
    state->receipt.field_return_packets_q8 = state->field_return_packets_q8;
    state->receipt.field_near_profile = state->field_near_profile;
    state->receipt.field_middle_profile = state->field_middle_profile;
    state->receipt.field_far_profile = state->field_far_profile;
    state->receipt.field_anchor_mix = state->field_anchor_mix;
    state->receipt.field_anchor_moment = state->field_anchor_moment;
    state->receipt.field_profile_tv_l1 = state->field_profile_tv_l1;
    state->receipt.field_relation_mix = state->field_relation_mix;
    state->receipt.field_relation_l1 = state->field_relation_l1;
    state->receipt.field_relation_sum = state->field_relation_sum;
    state->receipt.field_polarity_sum = state->field_polarity_sum;
    state->receipt.field_receptor_gain_q8 = state->field_receptor_gain_q8;
    state->receipt.field_diffusion_l1 = state->field_diffusion_l1;
    state->receipt.field_damage_q8 = state->field_damage_q8;
    state->receipt.field_recruited_sources = state->field_recruited_sources;
    state->receipt.field_repaired_sources = state->field_repaired_sources;
    state->receipt.field_growth_front_count = state->field_growth_front_count;
    state->receipt.field_component_count = state->field_component_count;
    state->receipt.field_component0_response_l1 =
        state->field_component0_response_l1;
    state->receipt.field_component1_response_l1 =
        state->field_component1_response_l1;
    state->receipt.field_component0_residual_l1 =
        state->field_component0_residual_l1;
    state->receipt.field_component1_residual_l1 =
        state->field_component1_residual_l1;
    state->receipt.field_component0_support_q8 =
        state->field_component0_support_q8;
    state->receipt.field_component1_support_q8 =
        state->field_component1_support_q8;
    state->receipt.field_withdrawn_sources = state->field_withdrawn_sources;
    state->receipt.active_joint_locus = state->active_joint_locus;
    state->receipt.tissue_matter_q8 =
        state->tissue_matter_q8[state->active_joint_locus % state->cell_count];
    state->receipt.tissue_coupling_q8 =
        state->tissue_coupling_q8[state->active_joint_locus % state->cell_count];
    state->receipt.field_packet_braid_sum = state->field_packet_braid_sum;
    state->receipt.field_packet_braid_l1 = state->field_packet_braid_l1;
    state->receipt.field_packet_braid_sources =
        state->field_packet_braid_sources;
    state->receipt.alternate_expression_endpoint =
        state->alternate_expression_endpoint;
    state->receipt.joint_expression_endpoints =
        state->joint_expression_endpoints;
    state->receipt.primary_expression_gain_q8 =
        state->primary_expression_gain_q8;
    state->receipt.alternate_expression_gain_q8 =
        state->alternate_expression_gain_q8;
    state->receipt.field_packet_alignment_sum =
        state->field_packet_alignment_sum;
    state->receipt.field_packet_alignment_l1 =
        state->field_packet_alignment_l1;
    state->receipt.field_packet_alignment_sources =
        state->field_packet_alignment_sources;
    state->receipt.field_mature_packet_sources =
        state->field_mature_packet_sources;
    state->receipt.field_effective_alignment_sum =
        state->field_effective_alignment_sum;
    state->receipt.field_effective_braid_sum =
        state->field_effective_braid_sum;
    state->receipt.field_effective_alignment_l1 =
        state->field_effective_alignment_l1;
    state->receipt.field_effective_braid_l1 =
        state->field_effective_braid_l1;
    state->receipt.field_alignment_locus = state->field_alignment_locus;
    state->receipt.field_braid_locus = state->field_braid_locus;
    state->receipt.recurrent_context_l1_q16 =
        state->recurrent_context_l1_q16;
    state->receipt.recurrent_context_mix = state->recurrent_context_mix;
    state->receipt.recurrent_context_revision =
        state->recurrent_context_revision;
    state->receipt.adaptive_context_l1 = adaptive::effective_context_l1(
        &state->adaptive, state->tissue_matter_q8,
        state->tissue_coupling_q8, state->cell_count);
    state->receipt.adaptive_context_anchor = state->adaptive.context_anchor;
    state->receipt.adaptive_context_matter_q8 = adaptive::context_matter_q8(
        &state->adaptive, state->tissue_matter_q8,
        state->tissue_coupling_q8, state->cell_count);
    state->receipt.adaptive_plasticity_q8 = state->adaptive.plasticity_q8;
    state->receipt.adaptive_consolidation_q8 =
        state->adaptive.consolidation_q8;
    state->receipt.adaptive_growth_q8 = state->adaptive.growth_q8;
    state->receipt.adaptive_turnover_q8 = state->adaptive.turnover_q8;
    state->receipt.adaptive_repair_q8 = state->adaptive.repair_q8;
    state->receipt.adaptive_replay_q8 = state->adaptive.replay_q8;
    state->receipt.adaptive_live_edges = state->adaptive.live_edges;
    state->receipt.adaptive_consolidated_routes =
        state->adaptive.consolidated_routes;
    state->receipt.adaptive_structural_revision =
        state->adaptive.structural_revision;
    state->receipt.adaptive_recruited_edges = state->adaptive.recruited_edges;
    state->receipt.adaptive_pruned_edges = state->adaptive.pruned_edges;
    state->receipt.adaptive_repaired_cells = state->adaptive.repaired_cells;
    state->receipt.adaptive_free_structural_matter_q8 =
        state->adaptive.free_structural_matter_q8;
    state->receipt.adaptive_edge_structural_matter_q8 =
        state->adaptive.edge_structural_matter_q8;
    state->receipt.adaptive_cumulative_structural_debit_q8 =
        state->adaptive.structural_debit_q8;
    state->receipt.adaptive_cumulative_structural_refund_q8 =
        state->adaptive.structural_refund_q8;
    state->receipt.adaptive_local_proposal_activity =
        state->adaptive.local_proposal_activity;
    if (state->f_owned_clock != 0u) {
      const ordinary_f::OrdinaryFPublication publication =
          *state->ordinary_f.publication;
      state->receipt.genesis_manifest = state->genesis_manifest;
      state->receipt.f_world = ordinary_f_world_digest(publication.world);
      state->receipt.completed_f_ticks = publication.completed_ticks;
      state->receipt.f_generation = publication.generation;
      state->receipt.f_fault = publication.fault;
      state->receipt.f_owned_clock = 1u;
      state->receipt.f_motor_zero = publication.motor.zero;
      state->receipt.f_motor_one = publication.motor.one;
      // Legacy scalar action authority remains transitional until the bounded
      // raw-boundary-to-F gate closes; recurrent language cannot write F state.
      state->receipt.legacy_action_authority = 1u;
    }
    digest_commitment(state, input, output, predecessor,
                      &state->receipt.commitment);
    publish_egress(state, egress);
    __nanosleep(kDevicePacingNanoseconds);
    if (shutdown.load(cuda::memory_order_acquire) == 0u) {
      cudaError_t tail_status = cudaSuccess;
      if (state->f_owned_clock != 0u) {
        state->expected_f_tick =
            ordinary_f::completed_ticks_acquire(state->ordinary_f) + 1u;
        // Keep the dependency serial: F first, then the root's sole
        // continuation. This avoids the illegal F -> active-root back-edge.
        tail_status = ordinary_f::request_forward_tick(state->ordinary_f);
        if (tail_status == cudaSuccess)
          tail_status = cudaGraphLaunch(cudaGetCurrentGraphExec(),
                                        cudaStreamGraphTailLaunch);
      } else {
        tail_status = cudaGraphLaunch(cudaGetCurrentGraphExec(),
                                      cudaStreamGraphTailLaunch);
      }
      if (tail_status != cudaSuccess) {
        cuda::atomic_ref<std::uint32_t, cuda::thread_scope_system> fault(
            lifecycle->continuation_fault);
        fault.store(continuation_cuda_fault(
                        ContinuationFault::root_tail_launch_base, tail_status),
                    cuda::memory_order_release);
        stopped.store(1u, cuda::memory_order_release);
      }
    } else {
      stopped.store(1u, cuda::memory_order_release);
    }
}
