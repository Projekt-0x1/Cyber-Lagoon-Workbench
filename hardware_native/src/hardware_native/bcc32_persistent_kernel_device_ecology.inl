__device__ __forceinline__ std::int32_t clamp_state(std::int64_t value) {
  return static_cast<std::int32_t>(value < -32768
                                       ? -32768
                                       : value > 32767 ? 32767 : value);
}

__device__ __forceinline__ std::int32_t clamp_trace(std::int64_t value) {
  return static_cast<std::int32_t>(
      value < -kTraceMax ? -kTraceMax
                         : value > kTraceMax ? kTraceMax : value);
}

__device__ __forceinline__ std::int32_t clamp_resource(std::int64_t value) {
  return static_cast<std::int32_t>(value < 0 ? 0 : value > kResourceMax
                                                      ? kResourceMax
                                                      : value);
}

__device__ __forceinline__ std::int32_t abs_value(std::int32_t value) {
  return static_cast<std::int32_t>(
      value < 0 ? -static_cast<std::int64_t>(value) : value);
}

__device__ __forceinline__ std::int64_t min_i64(std::int64_t left,
                                                 std::int64_t right) {
  return left < right ? left : right;
}

__device__ __forceinline__ std::int64_t max_i64(std::int64_t left,
                                                 std::int64_t right) {
  return left > right ? left : right;
}

// The pressure is a generic signed projection of every raw word.  Rotating
// by the cell index and mixing a founder byte keep equal-popcount contacts
// capable of driving different distributed cell pressures without a lookup.
__device__ __forceinline__ std::int32_t raw_pressure(
    const DeviceState* state, std::uint32_t index) {
  if (state->contact_count == 0u) return 0;
  const std::uint32_t word = state->contact[index % state->contact_count];
  const std::uint32_t shift = (index & 15u) + 1u;
  const std::uint32_t rotated = (word << shift) | (word >> (32u - shift));
  const std::uint32_t founder_byte = static_cast<std::uint32_t>(
      (state->founder >> ((index & 7u) * 8u)) & 0xffu);
  const std::uint32_t mixed = rotated ^ founder_byte ^
                              (0x9e3779b9u * (index + 1u));
  const std::int32_t odd = static_cast<std::int32_t>(
      __popc(mixed & 0x55555555u));
  const std::int32_t even = static_cast<std::int32_t>(
      __popc(mixed & 0xaaaaaaaau));
  const std::uint32_t nibble_shift = (index & 3u) * 4u;
  return (odd - even) * 32 +
         static_cast<std::int32_t>((mixed >> nibble_shift) & 15u) - 7;
}

__device__ __forceinline__ std::int32_t decay_to_zero(std::int32_t value) {
  if (value == 0) return 0;
  const std::int32_t amount = abs_value(value) / 16;
  const std::int32_t bounded = amount < 1 ? 1 : amount;
  return value > 0 ? -bounded : bounded;
}

__device__ __forceinline__ std::int32_t clamp_prediction(std::int64_t value) {
  return static_cast<std::int32_t>(
      value < -kPredictionMax
          ? -kPredictionMax
          : value > kPredictionMax ? kPredictionMax : value);
}

__device__ __forceinline__ std::int32_t clamp_credit_weight(
    std::int64_t value) {
  return static_cast<std::int32_t>(
      value < -kCreditWeightMax
          ? -kCreditWeightMax
          : value > kCreditWeightMax ? kCreditWeightMax : value);
}

// Route identity is a local consequence of the raw contact, founder, and cell
// position.  It is deliberately not a semantic cue table or a host choice.
__device__ __forceinline__ std::uint32_t prediction_route(
    const DeviceState* state, std::uint32_t index) {
  const std::uint32_t word = state->contact_count == 0u
                                 ? 0u
                                 : state->contact[index % state->contact_count];
  const std::uint32_t founder_lane = static_cast<std::uint32_t>(
      (state->founder >> ((index & 7u) * 8u)) & 0xffu);
  const std::uint32_t mixed = word ^ founder_lane ^ (index * 0x9e3779b9u);
  return mixed & (kPredictionRoutes - 1u);
}

__device__ void age_prediction_eligibility(DeviceState* state) {
  const std::uint32_t lifetime =
      adaptive::eligibility_lifetime(&state->adaptive);
  for (std::uint32_t route = 0u; route < kPredictionRoutes; ++route) {
    for (std::uint32_t index = 0u; index < state->cell_count; ++index) {
      if (state->eligibility_age[route][index] < lifetime + 1u)
        ++state->eligibility_age[route][index];
      if (state->eligibility_age[route][index] > lifetime)
        state->eligibility[route][index] = 0;
    }
  }
}

// A new external contact first grades the still-eligible route from the prior
// cue, then arms exactly one route for the next external contact.  Revisions
// make a physically held contact one observation rather than one observation
// per recurrent tick; empty contact only advances time.  The signed residual
// changes a resident route weight, not a host-side target or a replay buffer.
__device__ void compare_and_publish_prediction(DeviceState* state) {
  if (state->contact_count == 0u ||
      state->contact_revision == state->last_external_revision)
    return;

  const std::uint32_t lifetime =
      adaptive::eligibility_lifetime(&state->adaptive);
  for (std::uint32_t index = 0u; index < state->cell_count; ++index) {
    const std::int32_t observed = raw_pressure(state, index);
    for (std::uint32_t route = 0u; route < kPredictionRoutes; ++route) {
      if (state->eligibility[route][index] == 0 ||
          state->eligibility_age[route][index] > lifetime)
        continue;
      const std::int32_t consolidated = adaptive::consolidated_bias(
          &state->adaptive, index, route);
      const std::int32_t predicted = clamp_prediction(
          static_cast<std::int64_t>(state->prediction[route][index]) +
          consolidated);
      const std::int32_t residual = observed - predicted;
      adaptive::observe_prediction(&state->adaptive, index, route, observed,
                                   predicted, state->resource[index]);
      const std::int64_t signed_delta =
          static_cast<std::int64_t>(residual) *
          state->adaptive.cells[index].plasticity_q8 / (8ll * 256ll);
      state->credit_weight[route][index] = clamp_credit_weight(
          static_cast<std::int64_t>(state->credit_weight[route][index]) +
          signed_delta);
      state->eligibility[route][index] = 0;
      state->eligibility_age[route][index] = lifetime + 1u;
    }

    const std::uint32_t route = prediction_route(state, index);
    const std::int32_t consolidated = adaptive::consolidated_bias(
        &state->adaptive, index, route);
    state->prediction[route][index] = clamp_prediction(
        static_cast<std::int64_t>(state->state[index]) / 4 +
        state->trace[index] / 16 + state->credit_weight[route][index] +
        consolidated);
    state->eligibility[route][index] = 1;
    state->eligibility_age[route][index] = 0u;
  }
  state->last_external_revision = state->contact_revision;
}

__device__ __forceinline__ std::uint32_t expression_endpoint_gain_q8(
    const DeviceState* state, std::uint32_t locus) {
  if (state == nullptr || locus >= state->cell_count) return 0u;
  if (state->adaptive.cells[locus].damage_q8 != 0u) return 0u;
  return static_cast<std::uint32_t>(
      static_cast<std::uint64_t>(state->tissue_matter_q8[locus]) *
      state->tissue_coupling_q8[locus] / 256u);
}

// Re-express the learned joint through a device-owned physical locus.  The
// host can deliver raw contact and physical matter only; it cannot write the
// answer or restore a saved output after a lesion.
__device__ void refresh_joint_expression(DeviceState* state) {
  state->joint_output = 0;
  state->joint_residual = 0;
  state->joint_output_valid = 0u;
  state->joint_route = 0u;
  state->active_joint_locus = 0u;
  state->alternate_expression_endpoint = 0u;
  state->joint_expression_endpoints = 0u;
  state->primary_expression_gain_q8 = 0u;
  state->alternate_expression_gain_q8 = 0u;
  if (state->active_joint_index == sparse_source_joint::kNoIndex ||
      state->active_joint_index >= sparse_source_joint::kJointCapacity ||
      state->cell_count == 0u)
    return;
  const auto& joint = state->source_joint.joints[state->active_joint_index];
  if (joint.live == 0u || joint.interaction_enabled == 0u ||
      state->source_joint.rule_updates == 0u)
    return;
  // A raw history key allocates an empty resident context before its first
  // consequence.  It is not learned merely because another context on the
  // same adult has updates; an unseen key must abstain rather than exposing a
  // zero-valued pseudo-answer through the global arithmetic receipt.
  if (joint.context_index != sparse_source_joint::kNoIndex &&
      (joint.context_index >= sparse_source_joint::kContextCapacity ||
       state->source_joint.contexts[joint.context_index].updates == 0u))
    return;
  state->joint_route = joint.route;
  const std::uint32_t primary =
      sparse_source_joint::primary_expression_locus(
          joint, state->cell_count);
  const bool committed = joint.alternate_endpoint_committed != 0u;
  const std::uint32_t alternate =
      committed
          ? static_cast<std::uint32_t>(joint.alternate_endpoint %
                                       state->cell_count)
          : primary;
  state->active_joint_locus = primary;
  state->alternate_expression_endpoint = committed ? alternate : 0u;
  state->primary_expression_gain_q8 =
      expression_endpoint_gain_q8(state, primary);
  state->alternate_expression_gain_q8 =
      committed && alternate != primary
          ? expression_endpoint_gain_q8(state, alternate)
          : 0u;
  state->joint_expression_endpoints =
      (state->primary_expression_gain_q8 != 0u ? 1u : 0u) +
      (state->alternate_expression_gain_q8 != 0u ? 1u : 0u);
  const std::uint32_t gain_q8 =
      state->primary_expression_gain_q8 >
              state->alternate_expression_gain_q8
          ? state->primary_expression_gain_q8
          : state->alternate_expression_gain_q8;
  if (gain_q8 == 0u) return;
  state->joint_output = static_cast<std::int32_t>(
      static_cast<std::int64_t>(joint.prediction) * gain_q8 / 256);
  state->joint_residual = static_cast<std::int32_t>(
      static_cast<std::int64_t>(joint.credit) * gain_q8 / 256);
  state->joint_output_valid = 1u;
}

// The sparse source/joint gate is fed by the same raw ingress consumed by the
// resident prediction law.  Surface offsets are physical positions in the
// contact frame; no semantic source selector crosses this boundary.
__device__ void process_sparse_source_joint(DeviceState* state) {
  if (state->contact_revision == state->sparse_last_revision) return;
  state->active_joint_index = sparse_source_joint::kNoIndex;
  state->joint_output = 0;
  state->joint_residual = 0;
  state->joint_output_valid = 0u;
  state->joint_route = 0u;
  state->sparse_last_revision = state->contact_revision;
  sparse_source_joint::RawContact contacts[kMaxContactWords]{};
  if (state->contact_count == 0u) {
    source_joint_field_response::evaluate(
        &state->source_joint, contacts, 0u, state->cell_count,
        state->field_response, state->field_residual, &state->field_metrics);
    return;
  }
  for (std::uint32_t index = 0u; index < state->contact_count; ++index) {
    contacts[index].word = state->contact[index];
    contacts[index].surface_offset = static_cast<std::uint16_t>(index);
    contacts[index].span = 1u;
    contacts[index].sequence =
        state->contact_revision * kMaxContactWords + index + 1u;
  }

  // A preceding one-contact frame is resident physical context, not an
  // operation selector.  It is the only history cue used by the generic
  // context-conditioned learner below.
  if (state->contact_count == 1u)
    sparse_source_joint::observe_history_frame(
        &state->source_joint, contacts, state->contact_count);

  // A later raw contact is the only residual source.  It is consumed before
  // this frame is installed as a new resident source population, so the
  // update cannot be a host-written answer or a replayed pair label.
  for (std::uint32_t index = 0u; index < state->contact_count; ++index) {
    sparse_source_joint::apply_delayed_residual(
        &state->source_joint, contacts[index]);
  }
  for (std::uint32_t index = 0u; index < state->contact_count; ++index) {
    sparse_source_joint::observe_contact(&state->source_joint, contacts[index]);
  }
  for (std::uint32_t index = 1u; index < state->contact_count; ++index) {
    sparse_source_joint::observe_joint(&state->source_joint, contacts[index - 1u],
                                       contacts[index]);
    sparse_source_joint::arm_joint_prediction(
        &state->source_joint, sparse_source_joint::owner_for(contacts[index - 1u]),
        sparse_source_joint::owner_for(contacts[index]));
  }
  if (state->contact_count >= 2u) {
    const std::uint32_t index = sparse_source_joint::find_symmetric_joint(
        state->source_joint,
        sparse_source_joint::owner_for(contacts[0u]),
        sparse_source_joint::owner_for(contacts[1u]));
    if (index != sparse_source_joint::kNoIndex &&
        state->source_joint.joints[index].interaction_enabled != 0u &&
        state->source_joint.rule_updates != 0u) {
      state->active_joint_index = index;
    }
  }
  source_joint_field_response::evaluate(
      &state->source_joint, contacts, state->contact_count, state->cell_count,
      state->field_response, state->field_residual, &state->field_metrics);
}

__device__ __forceinline__ std::int32_t sparse_joint_motor_bias(
    const DeviceState* state) {
  std::int64_t total = 0;
  std::uint32_t count = 0u;
  for (std::uint32_t index = 0u; index <
       sparse_source_joint::kJointCapacity; ++index) {
    const auto& joint = state->source_joint.joints[index];
    if (joint.live == 0u || joint.interaction_enabled == 0u ||
        state->cell_count == 0u)
      continue;
    const std::uint32_t primary =
        sparse_source_joint::primary_expression_locus(
            joint, state->cell_count);
    const std::uint32_t primary_gain = static_cast<std::uint32_t>(
        static_cast<std::uint64_t>(state->tissue_matter_q8[primary]) *
        state->tissue_coupling_q8[primary] / 256u);
    std::uint32_t alternate_gain = 0u;
    if (joint.alternate_endpoint_committed != 0u) {
      const std::uint32_t alternate =
          static_cast<std::uint32_t>(joint.alternate_endpoint %
                                     state->cell_count);
      if (alternate != primary)
        alternate_gain = static_cast<std::uint32_t>(
            static_cast<std::uint64_t>(state->tissue_matter_q8[alternate]) *
            state->tissue_coupling_q8[alternate] / 256u);
    }
    const std::uint32_t gain_q8 =
        primary_gain > alternate_gain ? primary_gain : alternate_gain;
    total += static_cast<std::int64_t>(joint.prediction) * gain_q8 / 256;
    ++count;
  }
  if (count == 0u) return 0;
  const std::int64_t mean = total / count;
  return static_cast<std::int32_t>(
      mean < -4096 ? -4096 : mean > 4096 ? 4096 : mean);
}

__device__ void initialize_state(DeviceState* state, std::uint64_t founder,
                                 std::uint32_t cell_count,
                                 DeviceDigest sealed, DeviceDigest law,
                                 DeviceDigest image) {
  const ordinary_f::DeviceLaunchHandle ordinary_f = state->ordinary_f;
  const DeviceDigest genesis_manifest = state->genesis_manifest;
  const std::uint32_t f_owned_clock = state->f_owned_clock;
  *state = {};
  state->ordinary_f = ordinary_f;
  state->genesis_manifest = genesis_manifest;
  state->f_owned_clock = f_owned_clock;
  language::initialize(&state->language);
  adaptive::initialize(&state->adaptive, founder, cell_count);
  state->founder = founder;
  state->cell_count = cell_count;
  state->active_joint_index = sparse_source_joint::kNoIndex;
  state->active_joint_locus = 0u;
  state->sealed_execution = sealed;
  state->law = law;
  state->image = image;
  state->field_receptor_gain_q8 = 256u;
  sparse_source_joint::initialize(&state->source_joint);
  state->host_bootstrap_launches = 1u;
  for (std::uint32_t index = 0u; index < cell_count; ++index) {
    const std::uint32_t lane =
        static_cast<std::uint32_t>((founder >> ((index & 7u) * 8u)) & 0xffu);
    state->state[index] = static_cast<std::int32_t>(lane) - 127;
    state->resource[index] = kResourceMax;
    state->tissue_matter_q8[index] = 256u;
    state->tissue_coupling_q8[index] = 256u;
  }
  digest_words(nullptr, 0u, &state->receipt.input);
  digest_words(nullptr, 0u, &state->receipt.output);
  digest_words(nullptr, 0u, &state->receipt.predecessor);
  state->receipt.tick = 0u;
  state->receipt.phase = 0u;
  state->receipt.contact_sequence = 0u;
  state->receipt.sealed_execution = sealed;
  state->receipt.law = law;
  state->receipt.image = image;
  state->receipt.joint_output = 0;
  state->receipt.joint_residual = 0;
  state->receipt.joint_output_valid = 0u;
  state->receipt.joint_route = 0u;
  state->receipt.active_joint_locus = 0u;
  state->receipt.tissue_matter_q8 = 256u;
  state->receipt.tissue_coupling_q8 = 256u;
  state->receipt.genesis_manifest = genesis_manifest;
  state->receipt.f_owned_clock = f_owned_clock;
  state->receipt.legacy_action_authority = 1u;
  digest_commitment(state, state->receipt.input, state->receipt.output,
                    state->receipt.predecessor, &state->receipt.commitment);
}

__device__ __forceinline__ std::uint64_t load_system_u64(
    const std::uint64_t* address) {
  cuda::atomic_ref<std::uint64_t, cuda::thread_scope_system> reference(
      *const_cast<std::uint64_t*>(address));
  return reference.load(cuda::memory_order_acquire);
}

__device__ bool consume_ingress(DeviceState* state, IngressRing* ingress,
                                Lifecycle* lifecycle) {
  const std::uint64_t published = load_system_u64(&ingress->published);
  if (state->contact_revision >= published) return true;

  const std::uint64_t next_sequence = state->contact_revision + 1u;
  IngressSlot* slot = &ingress->slots[next_sequence % kIngressSlots];
  cuda::atomic_ref<std::uint32_t, cuda::thread_scope_system> shutdown(
      lifecycle->shutdown);
  while (load_system_u64(&slot->sequence) != next_sequence) {
    if (shutdown.load(cuda::memory_order_acquire) != 0u) return false;
    __nanosleep(1000u);
  }
  // Admit the one canonical raw byte through the fixed GrownAdult sensory
  // pair before the successor ordinary-F graph is launched below. A failed
  // exchange consumes neither the contact nor its ingress receipt.
  if (state->f_owned_clock != 0u && slot->raw_byte_count != 0u) {
    const cudaError_t admission = ordinary_f::admit_raw_sensory_byte(
        state->ordinary_f, slot->raw_bytes[0u]);
    if (admission != cudaSuccess) {
      cuda::atomic_ref<std::uint32_t, cuda::thread_scope_system> fault(
          lifecycle->continuation_fault);
      fault.store(continuation_cuda_fault(
                      ContinuationFault::ordinary_f_boundary, admission),
                  cuda::memory_order_release);
      return false;
    }
  }
  state->contact_count = slot->count;
  for (std::uint32_t index = 0u; index < slot->count; ++index)
    state->contact[index] = slot->words[index];
  state->contact_revision = next_sequence;
  __threadfence_system();
  cuda::atomic_ref<std::uint64_t, cuda::thread_scope_system> consumed(
      ingress->consumed);
  consumed.store(next_sequence, cuda::memory_order_release);
  return true;
}

__device__ __forceinline__ std::int32_t clamp_field_flux(std::int64_t value) {
  return static_cast<std::int32_t>(
      value < -kFieldFluxLimit
          ? -kFieldFluxLimit
          : value > kFieldFluxLimit ? kFieldFluxLimit : value);
}

__device__ __forceinline__ std::uint32_t clamp_q8(std::int64_t value) {
  return static_cast<std::uint32_t>(value < 0 ? 0 : value > 256 ? 256 : value);
}

__device__ __forceinline__ std::uint32_t field_distance(
    std::uint32_t left, std::uint32_t right) {
  return left > right ? left - right : right - left;
}

// These endpoint offsets are explicitly seeded local assay geometry. They
// measure whether two resident products can dissociate causally; they are not
// learned region coordinates and do not constitute grown connectivity.
__device__ __forceinline__ std::uint32_t packet_alignment_locus(
    const FieldSourceState& field, std::uint32_t cell_count) {
  return cell_count == 0u ? 0u : (field.anchor + 1u) % cell_count;
}

__device__ __forceinline__ std::uint32_t packet_braid_locus(
    const FieldSourceState& field, std::uint32_t cell_count) {
  return cell_count == 0u ? 0u : (field.anchor + 2u) % cell_count;
}

// Rebuild the local tissue adjacency from resident source anchors.  This is
// deliberately a physical neighborhood, not a relation/region selector.
__device__ void rebuild_field_adjacency(DeviceState* state) {
  for (std::uint32_t index = 0u; index < kFieldSourceCapacity; ++index) {
    auto& field = state->field_sources[index];
    if (field.live == 0u) continue;
    field.neighbor_mask = 0u;
    for (std::uint32_t other = 0u; other < kFieldSourceCapacity; ++other) {
      if (other == index || state->field_sources[other].live == 0u) continue;
      if (field_distance(field.anchor, state->field_sources[other].anchor) <= 8u)
        field.neighbor_mask |= 1u << other;
    }
  }
}

__device__ bool field_owner_in_current_contact(const DeviceState* state,
                                               std::uint64_t owner) {
  for (std::uint32_t contact_index = 0u;
       contact_index < state->contact_count; ++contact_index) {
    sparse_source_joint::RawContact contact{};
    contact.word = state->contact[contact_index];
    contact.surface_offset = static_cast<std::uint16_t>(contact_index);
    contact.span = 1u;
    if (sparse_source_joint::owner_for(contact) == owner) return true;
  }
  return false;
}

__device__ std::uint32_t reclaimable_derived_field(
    const DeviceState* state) {
  std::uint32_t selected = kFieldSourceCapacity;
  std::uint32_t selected_magnitude = 0xffffffffu;
  for (std::uint32_t index = 0u; index < kFieldSourceCapacity; ++index) {
    const auto& field = state->field_sources[index];
    if (field.live == 0u || field.support_q8 != 0u ||
        sparse_source_joint::find_source(state->source_joint, field.owner) !=
            sparse_source_joint::kNoIndex)
      continue;
    const std::uint32_t magnitude = static_cast<std::uint32_t>(
        abs_value(field.flux_q8) + abs_value(field.receptor_q8));
    if (selected == kFieldSourceCapacity || magnitude < selected_magnitude) {
      selected = index;
      selected_magnitude = magnitude;
    }
  }
  return selected;
}

// Mirror the resident source owners into a generic packet ecology.  The
// mapping is derived from physical source matter and is not a host-selected
// field identity.  Existing sources retain their packet state after contact
// withdrawal; new local matter can recruit an empty slot when a raw packet
// reaches it.
__device__ void sync_field_sources(DeviceState* state) {
  for (std::uint32_t source_index = 0u;
       source_index < sparse_source_joint::kSourceCapacity; ++source_index) {
    const auto& source = state->source_joint.sources[source_index];
    if (source.live == 0u) continue;
    // A durable sparse-source record is memory, not a field-slot lease.  Only
    // the owner present in this ordinary raw contact may acquire a new field
    // slot.  This prevents a retired field from being silently reminted on a
    // quiet device epoch while retaining the source record for later contact.
    const bool contacted = field_owner_in_current_contact(state, source.owner);
    std::uint32_t slot = kFieldSourceCapacity;
    for (std::uint32_t candidate = 0u; candidate < kFieldSourceCapacity;
         ++candidate) {
      if (state->field_sources[candidate].live != 0u &&
          state->field_sources[candidate].owner == source.owner) {
        slot = candidate;
        break;
      }
    }
    bool created = false;
    if (slot == kFieldSourceCapacity && contacted) {
      for (std::uint32_t candidate = 0u; candidate < kFieldSourceCapacity;
           ++candidate) {
        if (state->field_sources[candidate].live == 0u) {
          slot = candidate;
          state->field_sources[slot] = {};
          state->field_sources[slot].live = 1u;
          state->field_sources[slot].owner = source.owner;
          state->field_sources[slot].anchor =
              source.surface_offset % state->cell_count;
          state->field_sources[slot].packet_history_age =
              kFieldPacketHistoryExpired;
          created = true;
          break;
        }
      }
      // Finite field capacity is physical competition. A currently contacted
      // raw source outranks an unsupported child grown from prior packet
      // transport, but never displaces another source with current support or
      // a durable sparse-source owner. This resident provenance test prevents
      // transient growth from making ordinary raw reacquisition timing-racy.
      if (slot == kFieldSourceCapacity) {
        slot = reclaimable_derived_field(state);
        if (slot < kFieldSourceCapacity) {
          state->field_sources[slot] = {};
          state->field_sources[slot].live = 1u;
          state->field_sources[slot].owner = source.owner;
          state->field_sources[slot].anchor =
              source.surface_offset % state->cell_count;
          state->field_sources[slot].packet_history_age =
              kFieldPacketHistoryExpired;
          created = true;
        }
      }
    }
    if (slot < kFieldSourceCapacity && created) {
      auto& field = state->field_sources[slot];
      field.anchor = source.surface_offset % state->cell_count;
    }
  }
  for (std::uint32_t index = 0u; index < kFieldSourceCapacity; ++index) {
    auto& field = state->field_sources[index];
    if (field.live == 0u) continue;
    bool present = false;
    for (std::uint32_t contact_index = 0u;
         contact_index < state->contact_count; ++contact_index) {
      sparse_source_joint::RawContact contact{};
      contact.word = state->contact[contact_index];
      contact.surface_offset = static_cast<std::uint16_t>(contact_index);
      contact.span = 1u;
      if (sparse_source_joint::owner_for(contact) == field.owner) {
        present = true;
        break;
      }
    }
    if (present) {
      field.support_q8 = 256u;
      field.support_age = 0u;
    } else {
      field.support_q8 = 0u;
      if (field.support_age != 0xffffffffu) ++field.support_age;
      // Drain over the same bounded temporal scale as packet history. A 1/8
      // integer step could extinguish a returned packet before the next raw
      // withdrawal comparison, making causality depend on host polling speed.
      // The 1/16 step remains strictly dissipative and reaches the retirement
      // amplitude floor well within the declared history/lifecycle window.
      field.flux_q8 = clamp_field_flux(
          static_cast<std::int64_t>(field.flux_q8) * 15 / 16);
      field.receptor_q8 = clamp_field_flux(
          static_cast<std::int64_t>(field.receptor_q8) * 15 / 16);
      // Withdrawn packet matter is retired only after local in-flight flux
      // drains and the declared temporal packet-history window has expired.
      // The latter is still causal resident state even when the instantaneous
      // amplitudes are small, so clearing it earlier would turn withdrawal
      // into an implicit memory lesion. The owner remains resident in
      // source_joint, allowing later ordinary raw contact to reacquire a fresh
      // physical source without a host remint or authored route reset.
      if (field.support_age >= 8u &&
          field.packet_history_age == kFieldPacketHistoryExpired &&
          abs_value(field.flux_q8) <= 32 &&
          abs_value(field.receptor_q8) <= 32) {
        field = {};
        if (state->field_withdrawn_sources != 0xffffffffu)
          ++state->field_withdrawn_sources;
      }
    }
  }
  rebuild_field_adjacency(state);
}

__device__ void apply_field_pulse(DeviceState* state,
                                  const RawPhysicalIntervention& event) {
  if (event.flux_q8 == 0 && event.polarity_q8 == 0 &&
      event.displacement == 0 && event.diffusion_q8 == 0u &&
      event.receptor_gain_q8 == 256u && event.repair_q8 == 0u)
    return;
  sync_field_sources(state);
  const std::uint32_t center = event.center % state->cell_count;
  const std::uint32_t radius = event.radius == 0u ? 1u : event.radius;
  std::uint32_t chosen = kFieldSourceCapacity;
  std::uint32_t best_distance = 0xffffffffu;
  for (std::uint32_t index = 0u; index < kFieldSourceCapacity; ++index) {
    const auto& field = state->field_sources[index];
    if (field.live == 0u) continue;
    const std::uint32_t distance = field_distance(field.anchor, center);
    if (distance <= radius && distance < best_distance) {
      chosen = index;
      best_distance = distance;
    }
  }
  if (chosen == kFieldSourceCapacity) {
    // A pulse cannot nucleate a durable source at a host-selected empty
    // position. Recruitment is performed below only when resident transport
    // reaches an adjacent empty site.
    return;
  }
  if (chosen == kFieldSourceCapacity) return;
  auto& field = state->field_sources[chosen];
  const std::uint32_t transport = event.transport_q8 == 0u
                                       ? 256u
                                       : event.transport_q8 > 256u
                                             ? 256u
                                             : event.transport_q8;
  const std::int32_t incoming_flux = clamp_field_flux(
      static_cast<std::int64_t>(event.flux_q8) * transport / 256);
  const std::int32_t incoming_polarity = clamp_field_flux(event.polarity_q8);
  if ((incoming_flux != 0 || incoming_polarity != 0) &&
      field.packet_history_age <= kFieldPacketHistoryLifetime) {
    // The signed area is antisymmetric: reversing two successive local
    // packet vectors reverses the sign. A different source has no predecessor
    // in this state, and an expired history contributes no composition.
    const std::int64_t signed_area =
        static_cast<std::int64_t>(field.packet_history_flux_q8) *
            incoming_polarity -
        static_cast<std::int64_t>(field.packet_history_polarity_q8) *
            incoming_flux;
    field.packet_braid_q8 = clamp_field_flux(
        static_cast<std::int64_t>(field.packet_braid_q8) +
        signed_area / kFieldPacketBraidScale);
    const std::int64_t alignment =
        static_cast<std::int64_t>(field.packet_history_flux_q8) *
            incoming_flux +
        static_cast<std::int64_t>(field.packet_history_polarity_q8) *
            incoming_polarity;
    field.packet_alignment_q8 = clamp_field_flux(
        static_cast<std::int64_t>(field.packet_alignment_q8) +
        alignment / kFieldPacketBraidScale);
    if (field.packet_history_updates != 0xffffffffu)
      ++field.packet_history_updates;
  }
  if (incoming_flux != 0 || incoming_polarity != 0) {
    field.packet_history_flux_q8 = incoming_flux;
    field.packet_history_polarity_q8 = incoming_polarity;
    field.packet_history_age = 0u;
  }
  const std::uint32_t gain = clamp_q8(event.receptor_gain_q8);
  state->field_receptor_gain_q8 = gain;
  state->field_diffusion_l1 = 0u;
  field.flux_q8 = clamp_field_flux(
      static_cast<std::int64_t>(field.flux_q8) +
      static_cast<std::int64_t>(event.flux_q8) * transport / 256);
  field.polarity_q8 = clamp_field_flux(
      static_cast<std::int64_t>(field.polarity_q8) + event.polarity_q8);
  field.relation_q8 = clamp_field_flux(
      static_cast<std::int64_t>(field.relation_q8) +
      static_cast<std::int64_t>(event.polarity_q8) +
      static_cast<std::int64_t>(event.flux_q8) / 64);
  field.relation_q8 = static_cast<std::int32_t>(
      static_cast<std::int64_t>(field.relation_q8) * gain / 256);
  field.receptor_gain_q8 = gain;
  if (gain < 256u)
    field.receptor_q8 = clamp_field_flux(
        static_cast<std::int64_t>(field.receptor_q8) * gain / 256u);
  if (event.diffusion_q8 != 0u) {
    const std::uint32_t diffusion = clamp_q8(event.diffusion_q8);
    const std::int32_t source_flux =
        event.flux_q8 != 0 ? event.flux_q8 : field.flux_q8;
    const std::int32_t spread = static_cast<std::int32_t>(
        static_cast<std::int64_t>(source_flux) * diffusion / 256);
    field.flux_q8 = clamp_field_flux(
        static_cast<std::int64_t>(field.flux_q8) - spread);
    std::uint32_t total_weight = 0u;
    for (std::uint32_t index = 0u; index < kFieldSourceCapacity; ++index) {
      if (index == chosen || state->field_sources[index].live == 0u) continue;
      const std::uint32_t distance = field_distance(
          field.anchor, state->field_sources[index].anchor);
      if (distance != 0u && distance <= radius) total_weight += distance + 1u;
    }
    for (std::uint32_t index = 0u; index < kFieldSourceCapacity; ++index) {
      if (index == chosen || state->field_sources[index].live == 0u) continue;
      const std::uint32_t distance = field_distance(
          field.anchor, state->field_sources[index].anchor);
      if (distance == 0u || distance > radius) continue;
      if (total_weight == 0u) continue;
      const std::int32_t share = static_cast<std::int32_t>(
          static_cast<std::int64_t>(spread) * (distance + 1u) /
          total_weight);
      state->field_sources[index].flux_q8 = clamp_field_flux(
          static_cast<std::int64_t>(state->field_sources[index].flux_q8) + share);
      state->field_diffusion_l1 += static_cast<std::uint32_t>(abs_value(share));
    }
  }
  if (event.displacement != 0) {
    // A radius covering the complete resident aperture is a physical body
    // remap: every resident anchor is translated by the same local event.
    // Smaller apertures retain the single-source relocation behavior.
    const bool whole_body = event.radius >= state->cell_count;
    for (std::uint32_t index = 0u; index < kFieldSourceCapacity; ++index) {
      auto& moved_field = state->field_sources[index];
      if (moved_field.live == 0u || (!whole_body && index != chosen)) continue;
      const std::int64_t moved = static_cast<std::int64_t>(moved_field.anchor) +
                                 event.displacement;
      const std::int64_t wrapped = moved % state->cell_count;
      moved_field.anchor = static_cast<std::uint32_t>(
          wrapped < 0 ? wrapped + state->cell_count : wrapped);
    }
  }
  rebuild_field_adjacency(state);
}

__device__ void update_field_components(DeviceState* state);

// Advance packet flux locally.  A fixed fraction reaches a resident receptor
// and a smaller fraction returns to its source, so transport is observable
// without a host-side concentration map or a destructive clear operation.
__device__ void advance_field_ecology(DeviceState* state) {
  sync_field_sources(state);
  rebuild_field_adjacency(state);
  const bool endogenous_growth =
      state->adaptive.growth_q8 > 144u &&
      state->adaptive.damage_q8 < state->cell_count * 128u;
  if ((state->intervention.diffusion_q8 != 0u &&
       state->intervention.repair_q8 != 0u) ||
      endogenous_growth) {
    for (std::uint32_t parent_index = 0u;
         parent_index < kFieldSourceCapacity; ++parent_index) {
      auto& parent = state->field_sources[parent_index];
      const std::int32_t growth_threshold = endogenous_growth ? 32 : 64;
      if (parent.live == 0u ||
          abs_value(parent.flux_q8) < growth_threshold)
        continue;
      const std::uint32_t candidate_anchor =
          (parent.anchor + 1u) % state->cell_count;
      bool occupied = false;
      for (std::uint32_t other = 0u; other < kFieldSourceCapacity; ++other)
        if (state->field_sources[other].live != 0u &&
            state->field_sources[other].anchor == candidate_anchor)
          occupied = true;
      if (occupied) continue;
      for (std::uint32_t slot = 0u; slot < kFieldSourceCapacity; ++slot) {
        if (state->field_sources[slot].live != 0u) continue;
        const std::int32_t child_flux = parent.flux_q8 / 4;
        if (child_flux == 0) break;
        state->field_sources[slot] = {};
        auto& child = state->field_sources[slot];
        child.live = 1u;
        child.owner = mix64(parent.owner,
                            state->tick + state->field_recruited_sources + 1u);
        child.anchor = candidate_anchor;
        child.flux_q8 = child_flux;
        child.relation_q8 = parent.relation_q8 / 2;
        child.receptor_gain_q8 = parent.receptor_gain_q8;
        child.packet_history_age = kFieldPacketHistoryExpired;
        parent.flux_q8 -= child_flux;
        state->resource[parent.anchor] =
            state->resource[parent.anchor] > 8
                ? state->resource[parent.anchor] - 8 : 0;
        ++state->field_recruited_sources;
        break;
      }
      break;
    }
    rebuild_field_adjacency(state);
  }
  state->field_source_count = 0u;
  state->field_supported_source_count = 0u;
  state->field_near_profile = 0u;
  state->field_middle_profile = 0u;
  state->field_far_profile = 0u;
  state->field_anchor_mix = 0x243f6a8885a308d3ull;
  state->field_anchor_moment = 0u;
  std::int32_t flux_l1 = 0;
  std::int32_t polarity_l1 = 0;
  std::int32_t polarity_sum = 0;
  std::uint32_t receptor_count = 0u;
  std::uint32_t damage_q8 = 0u;
  std::uint32_t growth_front_count = 0u;
  std::int32_t packet_braid_sum = 0;
  std::uint32_t packet_braid_l1 = 0u;
  std::uint32_t packet_braid_sources = 0u;
  std::int32_t packet_alignment_sum = 0;
  std::uint32_t packet_alignment_l1 = 0u;
  std::uint32_t packet_alignment_sources = 0u;
  std::uint32_t mature_packet_sources = 0u;
  std::int32_t effective_alignment_sum = 0;
  std::int32_t effective_braid_sum = 0;
  std::uint32_t effective_alignment_l1 = 0u;
  std::uint32_t effective_braid_l1 = 0u;
  std::uint32_t dominant_alignment_l1 = 0u;
  std::uint32_t dominant_braid_l1 = 0u;
  state->field_alignment_locus = 0u;
  state->field_braid_locus = 0u;
  std::uint64_t relation_mix = 0x13198a2e03707344ull;
  std::uint32_t relation_l1 = 0u;
  std::int32_t relation_sum = 0;
  std::int32_t profile[kMaxCells]{};
  for (std::uint32_t index = 0u; index < state->cell_count; ++index)
    state->field_braid_drive[index] = 0;
  for (std::uint32_t index = 0u; index < state->cell_count; ++index)
    state->field_alignment_drive[index] = 0;
  for (std::uint32_t index = 0u; index < kFieldSourceCapacity; ++index) {
    auto& field = state->field_sources[index];
    if (field.live == 0u) continue;
    const std::uint32_t unsupported_flux_ceiling =
        field.support_q8 == 0u
            ? static_cast<std::uint32_t>(abs_value(field.flux_q8))
            : static_cast<std::uint32_t>(kFieldFluxLimit);
    if (field.packet_history_age != kFieldPacketHistoryExpired) {
      if (field.packet_history_age < 0xfffffffeu) ++field.packet_history_age;
      if (field.packet_history_age > kFieldPacketHistoryLifetime) {
        field.packet_history_flux_q8 = 0;
        field.packet_history_polarity_q8 = 0;
        field.packet_history_age = kFieldPacketHistoryExpired;
      }
    }
    if (field.packet_braid_q8 != 0)
      field.packet_braid_q8 = clamp_field_flux(
          static_cast<std::int64_t>(field.packet_braid_q8) +
          decay_to_zero(field.packet_braid_q8));
    if (field.packet_alignment_q8 != 0)
      field.packet_alignment_q8 = clamp_field_flux(
          static_cast<std::int64_t>(field.packet_alignment_q8) +
          decay_to_zero(field.packet_alignment_q8));
    ++state->field_source_count;
    if (field.support_q8 != 0u) ++state->field_supported_source_count;
    state->field_anchor_mix = mix64(state->field_anchor_mix, field.owner);
    state->field_anchor_mix = mix64(state->field_anchor_mix, field.anchor);
    relation_mix = mix64(relation_mix, static_cast<std::uint32_t>(field.relation_q8));
    relation_mix = mix64(relation_mix, field.neighbor_mask);
    relation_l1 += static_cast<std::uint32_t>(abs_value(field.relation_q8));
    relation_sum += field.relation_q8;
    if (__popc(field.neighbor_mask) <= 1u) ++growth_front_count;
    const std::uint32_t tissue_coupling =
        state->tissue_coupling_q8[field.anchor % state->cell_count];
    const std::uint32_t effective_receptor_gain = static_cast<std::uint32_t>(
        static_cast<std::uint64_t>(field.receptor_gain_q8) *
        tissue_coupling / 256u);
    const std::int32_t transfer = static_cast<std::int32_t>(
        static_cast<std::int64_t>(field.flux_q8) *
        static_cast<std::int64_t>(effective_receptor_gain) / 2048);
    field.flux_q8 -= transfer;
    field.receptor_q8 = clamp_field_flux(
        static_cast<std::int64_t>(field.receptor_q8) + transfer);
    ++field.return_age;
    if (field.return_age >= 3u && field.receptor_q8 != 0) {
      const std::int32_t returned = field.receptor_q8 / 4;
      field.receptor_q8 -= returned;
      field.flux_q8 = clamp_field_flux(
          static_cast<std::int64_t>(field.flux_q8) + returned);
      state->field_return_packets_q8 +=
          static_cast<std::uint64_t>(abs_value(returned));
      field.relation_q8 = clamp_field_flux(
          static_cast<std::int64_t>(field.relation_q8) + returned / 2);
      field.return_age = 0u;
    }
    if (field.damage_q8 != 0u) {
      // Resident resource and adaptive repair modulation may pay for field
      // recovery; the raw intervention path remains an accelerator.
      if (field.repair_reserve_q8 < 8u &&
          state->adaptive.repair_q8 > 96u) {
        const std::uint32_t anchor = field.anchor % state->cell_count;
        std::uint32_t contribution =
            1u + state->adaptive.repair_q8 / 64u;
        const std::uint32_t affordable = static_cast<std::uint32_t>(
            state->resource[anchor] > 0 ? state->resource[anchor] / 8 : 0);
        if (contribution > affordable) contribution = affordable;
        if (contribution != 0u) {
          field.repair_reserve_q8 += contribution * 8u;
          state->resource[anchor] -=
              static_cast<std::int32_t>(contribution * 8u);
        }
      }
      const std::uint32_t available = field.repair_reserve_q8 / 8u;
      const std::uint32_t repaired =
          available < field.damage_q8 ? available : field.damage_q8;
      if (repaired != 0u) {
        field.damage_q8 -= repaired;
        field.repair_reserve_q8 -=
            repaired > field.repair_reserve_q8 ? field.repair_reserve_q8
                                                : repaired;
        field.flux_q8 = clamp_field_flux(
            static_cast<std::int64_t>(field.flux_q8) + repaired);
        if (field.damage_q8 == 0u) ++state->field_repaired_sources;
      }
    }
    // Receptor circulation may change where unsupported packet matter sits,
    // but it cannot create a larger in-flight packet than survived this
    // epoch's withdrawal dissipation. This keeps return transport resident
    // while making absence a strictly dissipative boundary condition.
    if (field.support_q8 == 0u &&
        static_cast<std::uint32_t>(abs_value(field.flux_q8)) >
            unsupported_flux_ceiling) {
      field.flux_q8 = field.flux_q8 < 0
                          ? -static_cast<std::int32_t>(unsupported_flux_ceiling)
                          : static_cast<std::int32_t>(unsupported_flux_ceiling);
    }
    if (field.receptor_q8 != 0) ++receptor_count;
    flux_l1 += abs_value(field.flux_q8);
    polarity_l1 += abs_value(field.polarity_q8);
    polarity_sum += field.polarity_q8;
    damage_q8 += field.damage_q8;
    const std::uint32_t distance =
        field_distance(field.anchor, state->structural_focus_cell);
    const std::uint32_t magnitude = static_cast<std::uint32_t>(
        abs_value(field.flux_q8) + abs_value(field.receptor_q8));
    state->field_anchor_moment += magnitude * (field.anchor + 1u);
    profile[field.anchor % state->cell_count] += static_cast<std::int32_t>(magnitude);
    if (distance <= 4u)
      state->field_near_profile += magnitude;
    else if (distance <= 12u)
      state->field_middle_profile += magnitude;
    else
      state->field_far_profile += magnitude;
    const std::uint32_t alignment_locus =
        packet_alignment_locus(field, state->cell_count);
    const std::uint32_t braid_locus =
        packet_braid_locus(field, state->cell_count);
    const std::uint32_t alignment_gain_q8 = static_cast<std::uint32_t>(
        static_cast<std::uint64_t>(
            state->tissue_matter_q8[alignment_locus]) *
        state->tissue_coupling_q8[alignment_locus] / 256u);
    const std::uint32_t braid_gain_q8 = static_cast<std::uint32_t>(
        static_cast<std::uint64_t>(state->tissue_matter_q8[braid_locus]) *
        state->tissue_coupling_q8[braid_locus] / 256u);
    const std::int32_t effective_alignment = static_cast<std::int32_t>(
        static_cast<std::int64_t>(field.packet_alignment_q8) *
        alignment_gain_q8 / 256u);
    const std::int32_t effective_braid = static_cast<std::int32_t>(
        static_cast<std::int64_t>(field.packet_braid_q8) * braid_gain_q8 /
        256u);
    state->field_alignment_drive[alignment_locus] =
        source_joint_field_response::clamp_response(
            static_cast<std::int64_t>(
                state->field_alignment_drive[alignment_locus]) +
            effective_alignment);
    state->field_braid_drive[braid_locus] =
        source_joint_field_response::clamp_response(
            static_cast<std::int64_t>(state->field_braid_drive[braid_locus]) +
            effective_braid);
    effective_alignment_sum += effective_alignment;
    effective_braid_sum += effective_braid;
    const std::uint32_t alignment_abs =
        static_cast<std::uint32_t>(abs_value(effective_alignment));
    const std::uint32_t braid_abs =
        static_cast<std::uint32_t>(abs_value(effective_braid));
    effective_alignment_l1 += alignment_abs;
    effective_braid_l1 += braid_abs;
    if (alignment_abs > dominant_alignment_l1) {
      dominant_alignment_l1 = alignment_abs;
      state->field_alignment_locus = alignment_locus;
    }
    if (braid_abs > dominant_braid_l1) {
      dominant_braid_l1 = braid_abs;
      state->field_braid_locus = braid_locus;
    }

    const std::uint32_t lane = field.anchor % state->cell_count;
    state->field_response[lane] = source_joint_field_response::clamp_response(
        static_cast<std::int64_t>(state->field_response[lane]) +
        field.flux_q8 / 16 + field.polarity_q8 / 32);
    state->field_residual[lane] = source_joint_field_response::clamp_response(
        static_cast<std::int64_t>(state->field_residual[lane]) +
        field.receptor_q8 / 16);
    if (field.packet_braid_q8 != 0) {
      packet_braid_sum += field.packet_braid_q8;
      packet_braid_l1 += static_cast<std::uint32_t>(
          abs_value(field.packet_braid_q8));
      ++packet_braid_sources;
    }
    if (field.packet_alignment_q8 != 0) {
      packet_alignment_sum += field.packet_alignment_q8;
      packet_alignment_l1 += static_cast<std::uint32_t>(
          abs_value(field.packet_alignment_q8));
      ++packet_alignment_sources;
    }
    if (field.packet_history_updates >= kFieldPacketMaturityUpdates)
      ++mature_packet_sources;
  }
  state->field_metrics.response_l1 = 0;
  state->field_metrics.residual_l1 = 0;
  state->field_metrics.peak_response = 0;
  state->field_metrics.peak_residual = 0;
  state->field_metrics.functional_spatial_tv_l1 = 0;
  for (std::uint32_t index = 0u; index < state->cell_count; ++index) {
    const std::int32_t response = abs_value(state->field_response[index]);
    const std::int32_t residual = abs_value(state->field_residual[index]);
    state->field_metrics.response_l1 += response;
    state->field_metrics.residual_l1 += residual;
    if (response > state->field_metrics.peak_response)
      state->field_metrics.peak_response = response;
    if (residual > state->field_metrics.peak_residual)
      state->field_metrics.peak_residual = residual;
    if (index != 0u)
      state->field_metrics.functional_spatial_tv_l1 += abs_value(
          state->field_response[index] - state->field_response[index - 1u]);
  }
  state->receipt.field_source_count = state->field_source_count;
  state->receipt.field_receptor_count = receptor_count;
  state->receipt.field_packet_flux_l1 = flux_l1;
  state->receipt.field_polarity_l1 = polarity_l1;
  state->field_polarity_sum = polarity_sum;
  state->field_relation_mix = relation_mix;
  state->field_relation_l1 = relation_l1;
  state->field_relation_sum = relation_sum;
  state->field_damage_q8 = damage_q8;
  state->field_growth_front_count = growth_front_count;
  state->field_packet_braid_sum = packet_braid_sum;
  state->field_packet_braid_l1 = packet_braid_l1;
  state->field_packet_braid_sources = packet_braid_sources;
  state->field_packet_alignment_sum = packet_alignment_sum;
  state->field_packet_alignment_l1 = packet_alignment_l1;
  state->field_packet_alignment_sources = packet_alignment_sources;
  state->field_mature_packet_sources = mature_packet_sources;
  state->field_effective_alignment_sum = effective_alignment_sum;
  state->field_effective_braid_sum = effective_braid_sum;
  state->field_effective_alignment_l1 = effective_alignment_l1;
  state->field_effective_braid_l1 = effective_braid_l1;
  std::uint32_t profile_tv = 0u;
  for (std::uint32_t index = 1u; index < state->cell_count; ++index)
    profile_tv += static_cast<std::uint32_t>(
        abs_value(profile[index] - profile[index - 1u]));
  profile_tv += static_cast<std::uint32_t>(
      abs_value(profile[state->cell_count - 1u] - profile[0u]));
  state->field_profile_tv_l1 = profile_tv;
  state->receipt.field_return_packets_q8 = state->field_return_packets_q8;
  state->receipt.field_near_profile = state->field_near_profile;
  state->receipt.field_middle_profile = state->field_middle_profile;
  state->receipt.field_far_profile = state->field_far_profile;
  state->receipt.field_anchor_mix = state->field_anchor_mix;
  update_field_components(state);
}

// Derive two observer-only population summaries from resident adjacency. The
// component order is physical (minimum resident anchor), never a host region
// selector. These summaries make reciprocal occupied-component lesions
// testable without exporting a route table or writing a semantic target.
__device__ void update_field_components(DeviceState* state) {
  std::uint32_t component[kFieldSourceCapacity];
  std::uint32_t component_min[kFieldSourceCapacity];
  std::int32_t component_response[kFieldSourceCapacity];
  std::int32_t component_residual[kFieldSourceCapacity];
  std::uint32_t component_support[kFieldSourceCapacity];
  for (std::uint32_t index = 0u; index < kFieldSourceCapacity; ++index) {
    component[index] = 0xffffffffu;
    component_min[index] = 0xffffffffu;
    component_response[index] = 0;
    component_residual[index] = 0;
    component_support[index] = 0u;
  }
  std::uint32_t count = 0u;
  for (std::uint32_t root = 0u; root < kFieldSourceCapacity; ++root) {
    if (state->field_sources[root].live == 0u ||
        component[root] != 0xffffffffu)
      continue;
    const std::uint32_t id = count++;
    std::uint32_t stack[kFieldSourceCapacity];
    std::uint32_t stack_size = 0u;
    stack[stack_size++] = root;
    component[root] = id;
    while (stack_size != 0u) {
      const std::uint32_t current = stack[--stack_size];
      const auto& field = state->field_sources[current];
      if (field.anchor < component_min[id]) component_min[id] = field.anchor;
      // Use the component's resident packet/receptor state, not the global
      // recurrent response array. This keeps the dissociation axis local to
      // the occupied component and prevents a lesion from being scored as
      // spared merely because unrelated tissue changed its global activity.
      component_response[id] += abs_value(field.flux_q8) +
                                abs_value(field.receptor_q8);
      component_residual[id] += abs_value(field.relation_q8);
      component_support[id] += field.support_q8;
      std::uint32_t neighbours = field.neighbor_mask;
      while (neighbours != 0u) {
        const std::uint32_t next = __ffs(neighbours) - 1u;
        neighbours &= neighbours - 1u;
        if (next >= kFieldSourceCapacity ||
            state->field_sources[next].live == 0u ||
            component[next] != 0xffffffffu)
          continue;
        component[next] = id;
        stack[stack_size++] = next;
      }
    }
  }
  state->field_component_count = count;
  state->field_component0_response_l1 = 0;
  state->field_component1_response_l1 = 0;
  state->field_component0_residual_l1 = 0;
  state->field_component1_residual_l1 = 0;
  state->field_component0_support_q8 = 0u;
  state->field_component1_support_q8 = 0u;
  std::uint32_t first = 0xffffffffu;
  std::uint32_t second = 0xffffffffu;
  for (std::uint32_t id = 0u; id < count; ++id) {
    if (first == 0xffffffffu || component_min[id] < component_min[first]) {
      second = first;
      first = id;
    } else if (second == 0xffffffffu || component_min[id] < component_min[second]) {
      second = id;
    }
  }
  if (first != 0xffffffffu) {
    state->field_component0_response_l1 = component_response[first];
    state->field_component0_residual_l1 = component_residual[first];
    state->field_component0_support_q8 = component_support[first];
  }
  if (second != 0xffffffffu) {
    state->field_component1_response_l1 = component_response[second];
    state->field_component1_residual_l1 = component_residual[second];
    state->field_component1_support_q8 = component_support[second];
  }
}

__device__ void update_structural_receipt(DeviceState* state) {
  if (state->cell_count == 0u) return;
  const std::uint32_t focus = state->structural_focus_cell % state->cell_count;
  std::uint64_t support = 0u;
  for (std::uint32_t index = 0u; index < state->cell_count; ++index) {
    const std::uint32_t distance =
        index > focus ? index - focus : focus - index;
    const std::uint32_t proximity = distance > 32u ? 0u : 32u - distance;
    const std::uint32_t tissue_gain_q8 = static_cast<std::uint32_t>(
        static_cast<std::uint64_t>(state->tissue_matter_q8[index]) *
        state->tissue_coupling_q8[index] / 256u);
    support += static_cast<std::uint64_t>(tissue_gain_q8) * (proximity + 1u);
  }
  state->structural_support_q8 = static_cast<std::uint32_t>(
      support > 0xffffffffull ? 0xffffffffu : support);
  state->effective_field_response_l1 = state->field_metrics.response_l1;
  std::uint64_t structure_lanes[4]{0x6a09e667f3bcc909ull,
                                  0xbb67ae8584caa73bull,
                                  0x3c6ef372fe94f82bull,
                                  0xa54ff53a5f1d36f1ull};
  structure_lanes[0] = mix64(structure_lanes[0], state->structural_focus_cell);
  structure_lanes[1] = mix64(structure_lanes[1], state->structural_support_q8);
  structure_lanes[2] = mix64(structure_lanes[2], state->removed_matter_q8_sum);
  structure_lanes[3] = mix64(structure_lanes[3], state->cut_coupling_q8_sum);
  digest_finish(&state->structure, structure_lanes, sizeof(DeviceState));
}

__device__ void restore_transient_coupling(DeviceState* state) {
  if (state->cell_count == 0u || state->intervention.coupling_q8 >= 256u)
    return;
  const std::uint32_t center = state->intervention.center % state->cell_count;
  for (std::uint32_t index = 0u; index < state->cell_count; ++index) {
    const std::uint32_t distance =
        index > center ? index - center : center - index;
    if (distance <= state->intervention.radius)
      state->tissue_coupling_q8[index] = 256u;
  }
}

#include "bcc32_persistent_kernel_device_ecology_tail.inl"
