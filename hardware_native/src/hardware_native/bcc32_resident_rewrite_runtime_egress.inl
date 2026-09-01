// Egress derivation, receipt assembly, and publication for the resident
// rewrite epoch kernel.
//
// Causal path: the epoch kernel (resident_rewrite_epoch_kernel, defined in
// the including translation unit) is the sole writer of DeviceState for a
// given epoch. This unit is the transform that turns that DeviceState into
// the public egress contract: derive_output/update_output_projection turn
// admitted egress-history events into the bounded action/language surface,
// derive_resident_world_lineage and derive_active_open_inquiry_identity
// recompute the two public identities fill_receipt exposes, fill_receipt
// assembles the one TickReceipt for the epoch, and publish_egress commits
// the derived output plus receipt into the generation-fenced, host-visible
// EgressState. copy_snapshot is the paired reader: it performs the
// lock-free, retry-on-torn-generation read of that same EgressState from
// the host side and is the sole producer of a PassiveSnapshot for
// PersistentKernel::read_snapshot(). The invariant preserved across this
// extraction is unchanged: exactly one epoch-fenced EgressState writer
// (publish_egress) and one generation-checked host reader (copy_snapshot),
// with byte-identical receipt and snapshot contents before and after the
// split.
//
// Included (not compiled standalone) from bcc32_resident_rewrite_runtime.cu
// once DeviceState, EgressState, DerivedOutput, and OutputMaterial are fully
// defined. The device-side functions below are forward-declared earlier in
// that file so the epoch kernel and its device wrappers can call them ahead
// of this later definition point in the same translation unit.

__device__ void refresh_egress_history_digest(DeviceState* state) {
  struct HistoryDigestMaterial {
    ContentAddress predecessor{};
    egress_history::Event newest{};
    std::uint64_t next_sequence = 1u;
    std::uint64_t oldest_sequence = 1u;
    std::uint64_t overwrite_count = 0u;
    std::uint32_t fault = 0u;
    std::uint32_t has_newest = 0u;
  } material{};
  material.predecessor = state->egress_history_digest;
  material.next_sequence = state->egress_history.next_sequence;
  material.oldest_sequence = state->egress_history.oldest_sequence;
  material.overwrite_count = state->egress_history.overwrite_count;
  material.fault = state->egress_history.fault;
  if (egress_history::retained_count(&state->egress_history) != 0u &&
      egress_history::lookup(&state->egress_history,
                             egress_history::newest_sequence(&state->egress_history),
                             &material.newest))
    material.has_newest = 1u;
  ContentAddress digest{};
  digest_bytes(&material, sizeof(material), &digest,
               0x4547524553535f48ull ^ material.next_sequence);
  state->egress_history_digest = digest;
}

// The projection rings are only a cache of a resident publication.  Recheck
// the backing event before exposing a cached byte so a stale ring entry or a
// malformed history can never become a public motor-language event by itself.
// The producer locus is the existing resident lineage rail; it is not a
// semantic label and this helper does not select content.
__device__ __forceinline__ bool resident_public_event_matches(
    const egress_history::State* history, std::uint64_t sequence,
    std::uint32_t channel, std::uint32_t payload) {
  egress_history::Event event{};
  if (!egress_history::lookup(history, sequence, &event) ||
      !egress_history::has_producer_locus(&event, event.producer_locus))
    return false;
  return ((event.raw_word & rewrite::kRawChannelMask) >> 24u) == channel &&
         (event.raw_word & rewrite::kRawPayloadMask) == payload;
}

__device__ __noinline__ void derive_output(const DeviceState* state,
                                           bool generated_this_epoch,
                                           DerivedOutput* output) {
  output->language_count = 0u;
  output->action_count = 0u;
  const egress_history::State* history = &state->egress_history;
  const std::uint64_t oldest = history->oldest_sequence;
  const std::uint64_t next = history->next_sequence;
  if (history->fault != 0u || next == 0u || oldest == 0u || oldest > next ||
      next - oldest > egress_history::capacity) {
    output->actions[0] = 0u;
    output->actions[1] = 0u;
    output->actions[2] = 0u;
    output->action_count = 3u;
    return;
  }
  // A quiet epoch must not republish the last generated trajectory.  The
  // cached projection remains resident history, but public output is an
  // edge-triggered publication from the current causal epoch.
  //
  // Two load-bearing shapes share this field. The three-rail fallback below
  // serves a motor consumer; this per-event replay serves the action-return
  // contract, which polls actions.back() for a channel==1 payload only this
  // loop publishes -- de6857fa57 deleted it and stranded that poll for its
  // whole epoch budget. Disjoint: this runs only on a generating epoch.
  if (generated_this_epoch) {
    for (std::uint32_t i = 0u; i < state->projected_language_count; ++i) {
      const std::uint32_t index =
          (state->projected_language_head + i) % kLanguageBytes;
      const std::uint32_t payload = state->projected_language[index];
      if (resident_public_event_matches(
              history, state->projected_language_sequences[index], 0u,
              payload))
        output->language[output->language_count++] =
            static_cast<std::uint8_t>(payload);
    }
    for (std::uint32_t i = 0u; i < state->projected_action_count; ++i) {
      const std::uint32_t index =
          (state->projected_action_head + i) % kActionWords;
      const std::uint32_t payload = state->projected_actions[index];
      if (resident_public_event_matches(
              history, state->projected_action_sequences[index], 1u,
              payload))
        output->actions[output->action_count++] = payload;
    }
  }

  if (output->action_count != 0u)
    return;

  egress_history::Event latest{};
  if (generated_this_epoch &&
      egress_history::lookup(history, egress_history::newest_sequence(history), &latest)) {
    const std::uint32_t channel = (latest.raw_word & rewrite::kRawChannelMask) >> 24u;
    const std::uint32_t payload = latest.raw_word & rewrite::kRawPayloadMask;
    if (channel == 0u && payload <= 0xffu) {
      output->actions[0] = 0u;
      output->actions[1] = 0u;
      output->actions[2] = 0u;
      output->action_count = 3u;
      return;
    }
  }

  if (state->world.raw_motor_valid == 0u) {
    output->actions[0] = 0u;
    output->actions[1] = 0u;
    output->actions[2] = 0u;
    output->action_count = 3u;
    return;
  }
  const BoundaryWord value = state->world.raw_motor_value;
  output->actions[0] = ~value;
  output->actions[1] = value;
  output->actions[2] = 0xffffffffu;
  output->action_count = 3u;
}

__device__ void update_output_projection(DeviceState* state) {
  const std::uint64_t oldest = state->egress_history.oldest_sequence;
  while (state->projected_language_count != 0u &&
         state->projected_language_sequences[state->projected_language_head] < oldest) {
    state->projected_language_head = (state->projected_language_head + 1u) % kLanguageBytes;
    --state->projected_language_count;
  }
  while (state->projected_action_count != 0u &&
         state->projected_action_sequences[state->projected_action_head] < oldest) {
    state->projected_action_head = (state->projected_action_head + 1u) % kActionWords;
    --state->projected_action_count;
  }

  const std::uint64_t sequence = egress_history::newest_sequence(&state->egress_history);
  egress_history::Event event{};
  if (!egress_history::lookup(&state->egress_history, sequence, &event))
    return;
  const std::uint32_t channel = (event.raw_word & rewrite::kRawChannelMask) >> 24u;
  const std::uint32_t payload = event.raw_word & rewrite::kRawPayloadMask;
  if (!egress_history::has_producer_locus(&event, event.producer_locus))
    return;
  if (channel == 0u && payload <= 0xffu) {
    if (state->projected_language_count == kLanguageBytes) {
      state->projected_language_head = (state->projected_language_head + 1u) % kLanguageBytes;
      --state->projected_language_count;
    }
    const std::uint32_t index =
        (state->projected_language_head + state->projected_language_count) % kLanguageBytes;
    state->projected_language[index] = static_cast<std::uint8_t>(payload);
    state->projected_language_sequences[index] = sequence;
    ++state->projected_language_count;
  } else if (channel == 1u) {
    if (state->projected_action_count == kActionWords) {
      state->projected_action_head = (state->projected_action_head + 1u) % kActionWords;
      --state->projected_action_count;
    }
    const std::uint32_t index =
        (state->projected_action_head + state->projected_action_count) % kActionWords;
    state->projected_actions[index] = payload;
    state->projected_action_sequences[index] = sequence;
    ++state->projected_action_count;
  }
}

// Recompute the same slot-order-independent logical Record organization used
// by refresh_receipt. The published lineage is valid only if the resident's
// stored receipt is current and no private close/factor transaction is hiding
// an unpublished successor. Host transport counters are intentionally absent.
__device__ bool derive_resident_world_lineage(
    const DeviceState* state, std::uint64_t* organization_digest) {
  if (organization_digest != nullptr) *organization_digest = 0u;
  if (state == nullptr || organization_digest == nullptr ||
      state->world.fault != 0u ||
      state->world.cross_context_factor_pending != 0u ||
      state->close_work_active != 0u)
    return false;

  std::uint64_t observed = 0u;
  for (std::uint32_t slot = 0u;
       slot < rewrite::live_record_capacity(&state->world); ++slot) {
    const rewrite::Record& record = state->world.records[slot];
    if (record.matter_q8 == 0u || record.lane[0] == rewrite::kFormEmpty)
      continue;
    observed ^= rewrite::logical_record_digest(record);
  }
  if (observed != state->world.organization_digest)
    return false;
  *organization_digest = observed;
  return true;
}

// Do not mint an observer identity. Reuse the resident OpenInquiry owner and
// the same unresolved-fork derivation that gates a real ticketed consequence.
// Ambiguity, stale snapshots, withdrawn Programs, malformed owners, a close
// transaction, or a fault all make the public identity invalid.
__device__ bool derive_active_open_inquiry_identity(
    const DeviceState* state, std::uint32_t* owner,
    std::uint32_t* identity, std::uint32_t* generation) {
  if (owner != nullptr) *owner = rewrite::kInvalid;
  if (identity != nullptr) *identity = 0u;
  if (generation != nullptr) *generation = 0u;
  if (state == nullptr || owner == nullptr || identity == nullptr ||
      generation == nullptr || state->world.fault != 0u ||
      state->world.cross_context_factor_pending != 0u ||
      state->close_work_active != 0u)
    return false;

  const std::uint32_t slot =
      rewrite::open_inquiry::unique_active_inquiry(&state->world);
  if (slot == rewrite::kInvalid ||
      slot >= rewrite::live_record_capacity(&state->world))
    return false;
  const rewrite::Record& header = state->world.records[slot];
  if (header.matter_q8 == 0u ||
      header.lane[0] != rewrite::open_inquiry::kFormOpenInquiry ||
      header.lane[1] == 0u || header.lane[1] == rewrite::kInvalid ||
      header.revision == 0u ||
      rewrite::open_inquiry::unique_header_by_owner(
          &state->world, rewrite::open_inquiry::kFormOpenInquiry,
          header.lane[1]) != slot)
    return false;

  std::uint32_t unresolved = 0u;
  if (!inquiry_return::derive_unresolved_identity(
          &state->world, header, &unresolved) ||
      unresolved == 0u || unresolved == rewrite::kInvalid)
    return false;

  *owner = header.lane[1];
  *identity = unresolved;
  *generation = header.revision;
  return true;
}

__device__ void fill_receipt(DeviceState* state, bool consumed_contact,
                             bool accepted_action_return, bool device_body_return,
                             bool device_body_attached,
                             const DerivedOutput& derived) {
  TickReceipt& receipt = state->receipt;
  receipt.tick = state->tick;
  receipt.phase = state->world.revision;
  receipt.contact_sequence = state->contact_sequence;
  receipt.sealed_execution = state->sealed;
  receipt.law = state->law;
  receipt.image = state->image;
  receipt.genesis_manifest = state->genesis;
  receipt.predecessor = state->predecessor;
  digest_bytes(&state->world, sizeof(state->world), &receipt.rewrite_world, 0x525752305f574f52ull);
  receipt.rewrite_revision = state->world.revision;
  receipt.rewrite_admitted_events = state->world.admitted_events;
  receipt.rewrite_fault = state->world.fault;
  receipt.rewrite_owned_clock = 1u;
  receipt.rewrite_descriptions = state->world.concrete_descriptions;
  receipt.rewrite_mature_descriptions = state->world.mature_descriptions;
  receipt.rewrite_partial_matches = state->world.partial_matches;
  receipt.rewrite_direct_fires = state->world.direct_fires;
  receipt.rewrite_staged_fires = state->world.staged_fires;
  receipt.rewrite_partials_aged_out = state->world.partials_aged_out;
  receipt.rewrite_partials_retired_matched =
      state->world.partials_retired_matched;
  receipt.rewrite_partials_retired_unmatched =
      state->world.partials_retired_unmatched;
  receipt.rewrite_conflict_abstentions = state->world.conflict_abstentions;
  receipt.rewrite_constructor_rewrites = state->world.constructor_rewrites;
  receipt.rewrite_motor_value = state->world.raw_motor_value;
  receipt.rewrite_motor_valid = state->world.raw_motor_valid;
  receipt.rewrite_motor_babble_actions = state->resident_motor_babble_actions;
  receipt.rewrite_active_locus = state->world.active_locus;
  receipt.rewrite_constructor_locus = state->world.constructor_locus;
  receipt.rewrite_removed_matter_q8 = state->world.lesion.removed_matter_q8;
  receipt.rewrite_program_rules = state->world.program_rules;
  receipt.rewrite_mature_program_rules = state->world.mature_program_rules;
  receipt.rewrite_trajectory_records = state->world.trajectory_records;
  receipt.rewrite_version_space_factors = state->world.version_space_factors;
  receipt.rewrite_version_space_alternatives =
      state->world.version_space_alternatives;
  receipt.rewrite_mature_version_space_alternatives =
      state->world.mature_version_space_alternatives;
  receipt.rewrite_version_space_witnesses = state->world.version_space_witnesses;
  receipt.rewrite_version_space_conflict_abstentions =
      state->world.version_space_conflict_abstentions;
  receipt.rewrite_retained_exemplars = state->world.retained_exemplars;
  receipt.rewrite_program_generated_events = state->world.program_generated_events;
  receipt.rewrite_program_conflict_abstentions = state->world.program_conflict_abstentions;
  receipt.rewrite_rejected_unbound_variables = state->world.rejected_unbound_variables;
  receipt.rewrite_completed_inductions = state->world.completed_inductions;
  receipt.rewrite_span_program_rules = state->world.span_program_rules;
  receipt.rewrite_mature_span_program_rules = state->world.mature_span_program_rules;
  receipt.rewrite_span_generated_events = state->world.span_generated_events;
  receipt.rewrite_span_conflict_abstentions = state->world.span_conflict_abstentions;
  receipt.rewrite_span_rejected_unbound_variables = state->world.span_rejected_unbound_variables;
  receipt.rewrite_span_ambiguous_abstentions = state->world.span_ambiguous_abstentions;
  receipt.rewrite_span_completed_inductions = state->world.span_completed_inductions;
  receipt.rewrite_causal_relation_generated_events =
      state->world.causal_relation_generated_events;
  receipt.rewrite_causal_relation_probe_steps =
      state->world.causal_relation_probe_steps;
  receipt.rewrite_causal_relation_participating_records =
      state->world.causal_relation_participating_records;
  receipt.rewrite_causal_relation_live_records =
      substrate::bcc32::resident_causal_constraint_participation::
          participation_count(&state->world);
  // These are derived population receipts, not monotonic organism counters.
  // Rebuild them from the current resident revision on every publication;
  // otherwise a passive poll turns one surviving witness into an apparent
  // ever-growing ecology and can exceed kRecordCapacity within seconds.
  receipt.rewrite_causal_relation_source_witness_records = 0u;
  receipt.rewrite_causal_relation_source_witness_leaves = 0u;
  for (std::uint32_t slot = 0u;
       slot < rewrite::live_record_capacity(&state->world); ++slot) {
    const rewrite::Record& record = state->world.records[slot];
    if (record.matter_q8 == 0u ||
        record.lane[0] != rewrite::kFormTransformationWitness ||
        (record.lane[7] !=
             rewrite::resident_causal_relation_source_witness::kWitnessMarker &&
         record.lane[7] != rewrite::resident_causal_relation_source_witness::
             kReafferentWitnessMarker))
      continue;
    ++receipt.rewrite_causal_relation_source_witness_records;
    receipt.rewrite_causal_relation_source_witness_leaves += record.lane[2];
  }
  receipt.rewrite_causal_relation_independent_sources =
      state->world.causal_relation_independent_sources;
  receipt.rewrite_causal_relation_source_contributions =
      state->world.causal_relation_source_contributions;
  receipt.rewrite_causal_relation_max_source_contribution =
      state->world.causal_relation_max_source_contribution;
  receipt.rewrite_causal_relation_contribution_concentration_q16 =
      state->world.causal_relation_contribution_concentration_q16;
  receipt.rewrite_causal_relation_singleton_supported_steps =
      state->world.causal_relation_singleton_supported_steps;
  receipt.rewrite_causal_relation_minimum_probe_support =
      state->world.causal_relation_minimum_probe_support;
  receipt.rewrite_causal_relation_component_digest =
      state->world.causal_relation_component_digest;
  receipt.rewrite_causal_relation_component_revision_digest =
      state->world.causal_relation_component_revision_digest;
  receipt.rewrite_causal_relation_external_provenance_digest =
      state->world.causal_relation_external_provenance_digest;
  receipt.rewrite_causal_relation_external_leaves =
      state->world.causal_relation_external_leaves;
  receipt.rewrite_public_emission_receipt_valid =
      state->world.generated_receipt_valid;
  receipt.rewrite_public_emission_owner = state->world.generated_receipt_owner;
  receipt.rewrite_public_emission_participant_records =
      state->world.generated_receipt_participant_records;
  receipt.rewrite_public_emission_external_leaves =
      state->world.generated_receipt_external_leaves;
  receipt.rewrite_public_emission_independent_sources =
      state->world.generated_receipt_independent_sources;
  receipt.rewrite_public_emission_source_contributions =
      state->world.generated_receipt_source_contributions;
  receipt.rewrite_public_emission_topology_digest =
      state->world.generated_receipt_topology_digest;
  receipt.rewrite_public_emission_revision_digest =
      state->world.generated_receipt_revision_digest;
  receipt.rewrite_public_emission_provenance_digest =
      state->world.generated_receipt_provenance_digest;
  receipt.rewrite_public_emission_participation_digest =
      state->world.generated_receipt_participation_digest;
  receipt.rewrite_public_emission_epoch = state->world.generated_receipt_epoch;
  receipt.rewrite_close_work_pending =
      state->world.close_work_pending != 0u || state->close_work_active != 0u
          ? 1u
          : 0u;
  receipt.rewrite_close_work_phase = state->close_work_active != 0u
                                         ? state->close_work_staging_world.close_work_phase
                                         : state->world.close_work_phase;
  receipt.rewrite_open_inquiries = 0u;
  receipt.rewrite_open_inquiry_constructors = 0u;
  receipt.rewrite_open_inquiry_captured = 0u;
  receipt.rewrite_open_inquiry_bound = 0u;
  receipt.rewrite_open_inquiry_settled = 0u;
  receipt.rewrite_open_inquiry_resumed = 0u;
  receipt.rewrite_open_inquiry_prefix_current = 0u;
  receipt.rewrite_open_inquiry_prefix_yielded = 0u;
  receipt.rewrite_open_inquiry_prefix_flags = 0u;
  for (std::uint32_t slot = 0u;
       slot < rewrite::live_record_capacity(&state->world); ++slot) {
    const rewrite::Record& record = state->world.records[slot];
    if (record.matter_q8 == 0u) continue;
    if (record.lane[0] == rewrite::open_inquiry::kFormOpenInquiry) {
      ++receipt.rewrite_open_inquiries;
      receipt.rewrite_open_inquiry_captured +=
          (record.lane[7] & rewrite::open_inquiry::kInquirySurfaceCaptured) !=
          0u;
      receipt.rewrite_open_inquiry_bound +=
          (record.lane[7] & rewrite::open_inquiry::kInquiryReplyBound) != 0u;
      receipt.rewrite_open_inquiry_settled +=
          (record.lane[7] & rewrite::open_inquiry::kInquirySettled) != 0u;
      receipt.rewrite_open_inquiry_resumed +=
          (record.lane[7] & rewrite::open_inquiry::kInquiryResumed) != 0u;
      const std::uint32_t prefix =
          rewrite::cross_contact::find_trajectory_by_owner(&state->world,
                                                            record.lane[2]);
      if (prefix != rewrite::kInvalid) {
        receipt.rewrite_open_inquiry_prefix_current +=
            state->world.records[prefix].lane[3] == 0u;
        receipt.rewrite_open_inquiry_prefix_yielded +=
            state->world.records[prefix].lane[4] != 0u;
        receipt.rewrite_open_inquiry_prefix_flags =
            state->world.records[prefix].lane[7];
      }
    } else if (record.lane[0] ==
               rewrite::open_inquiry::kFormOpenInquiryConstructor) {
      ++receipt.rewrite_open_inquiry_constructors;
    }
  }
  receipt.rewrite_open_inquiry_construction_attempts =
      state->world.open_inquiry_construction_attempts;
  receipt.rewrite_open_inquiry_decline_active_inquiry =
      state->world.open_inquiry_decline_active_inquiry;
  receipt.rewrite_open_inquiry_decline_no_suspended_trajectory =
      state->world.open_inquiry_decline_no_suspended_trajectory;
  receipt.rewrite_open_inquiry_decline_not_wholly_external =
      state->world.open_inquiry_decline_not_wholly_external;
  receipt.rewrite_open_inquiry_decline_not_yielded =
      state->world.open_inquiry_decline_not_yielded;
  receipt.rewrite_open_inquiry_decline_already_open =
      state->world.open_inquiry_decline_already_open;
  receipt.rewrite_open_inquiry_decline_fork_failed =
      state->world.open_inquiry_decline_fork_failed;
  receipt.rewrite_open_inquiry_decline_multi_constructor =
      state->world.open_inquiry_decline_multi_constructor;
  receipt.rewrite_open_inquiry_decline_free_records =
      state->world.open_inquiry_decline_free_records;
  receipt.rewrite_open_inquiry_decline_owner_failed =
      state->world.open_inquiry_decline_owner_failed;
  receipt.rewrite_open_inquiry_construction_admitted =
      state->world.open_inquiry_construction_admitted;
  receipt.rewrite_oi_capture_surface_attempts =
      state->world.oi_capture_surface_attempts;
  receipt.rewrite_oi_capture_decline_no_active_inquiry =
      state->world.oi_capture_decline_no_active_inquiry;
  receipt.rewrite_oi_capture_decline_ambiguous_inquiry =
      state->world.oi_capture_decline_ambiguous_inquiry;
  receipt.rewrite_oi_capture_decline_no_current_trajectory =
      state->world.oi_capture_decline_no_current_trajectory;
  receipt.rewrite_oi_capture_decline_ambiguous_trajectory =
      state->world.oi_capture_decline_ambiguous_trajectory;
  receipt.rewrite_oi_capture_decline_already_progressed =
      state->world.oi_capture_decline_already_progressed;
  receipt.rewrite_oi_capture_decline_reserved_pending =
      state->world.oi_capture_decline_reserved_pending;
  receipt.rewrite_oi_capture_decline_surface_owner_is_suspended =
      state->world.oi_capture_decline_surface_owner_is_suspended;
  receipt.rewrite_oi_capture_decline_not_wholly_external =
      state->world.oi_capture_decline_not_wholly_external;
  receipt.rewrite_oi_capture_decline_surface_zero_length =
      state->world.oi_capture_decline_surface_zero_length;
  receipt.rewrite_oi_capture_decline_surface_too_long =
      state->world.oi_capture_decline_surface_too_long;
  receipt.rewrite_oi_capture_decline_insufficient_free_records =
      state->world.oi_capture_decline_insufficient_free_records;
  receipt.rewrite_oi_capture_surface_admitted =
      state->world.oi_capture_surface_admitted;
  receipt.rewrite_oi_bind_reply_attempts = state->world.oi_bind_reply_attempts;
  receipt.rewrite_oi_bind_reply_admitted = state->world.oi_bind_reply_admitted;
  receipt.rewrite_oi_settle_reply_attempts =
      state->world.oi_settle_reply_attempts;
  receipt.rewrite_oi_settle_decline_no_active_inquiry =
      state->world.oi_settle_decline_no_active_inquiry;
  receipt.rewrite_oi_settle_decline_ambiguous_inquiry =
      state->world.oi_settle_decline_ambiguous_inquiry;
  receipt.rewrite_oi_settle_decline_no_current_trajectory =
      state->world.oi_settle_decline_no_current_trajectory;
  receipt.rewrite_oi_settle_decline_ambiguous_trajectory =
      state->world.oi_settle_decline_ambiguous_trajectory;
  receipt.rewrite_oi_settle_decline_not_reply_bound =
      state->world.oi_settle_decline_not_reply_bound;
  receipt.rewrite_oi_settle_decline_already_settled =
      state->world.oi_settle_decline_already_settled;
  receipt.rewrite_oi_settle_decline_not_awaiting_reply =
      state->world.oi_settle_decline_not_awaiting_reply;
  receipt.rewrite_oi_settle_decline_selected_owner_zero =
      state->world.oi_settle_decline_selected_owner_zero;
  receipt.rewrite_oi_settle_decline_selected_owner_invalid =
      state->world.oi_settle_decline_selected_owner_invalid;
  receipt.rewrite_oi_settle_decline_selected_revision_zero =
      state->world.oi_settle_decline_selected_revision_zero;
  receipt.rewrite_oi_settle_decline_selected_revision_invalid =
      state->world.oi_settle_decline_selected_revision_invalid;
  receipt.rewrite_oi_settle_decline_reply_matter_zero =
      state->world.oi_settle_decline_reply_matter_zero;
  receipt.rewrite_oi_settle_decline_reply_not_trajectory_form =
      state->world.oi_settle_decline_reply_not_trajectory_form;
  receipt.rewrite_oi_settle_decline_reply_owner_zero =
      state->world.oi_settle_decline_reply_owner_zero;
  receipt.rewrite_oi_settle_decline_reply_owner_invalid =
      state->world.oi_settle_decline_reply_owner_invalid;
  receipt.rewrite_oi_settle_decline_reply_length_zero =
      state->world.oi_settle_decline_reply_length_zero;
  receipt.rewrite_oi_settle_decline_reply_not_current =
      state->world.oi_settle_decline_reply_not_current;
  receipt.rewrite_oi_settle_decline_no_suspended_trajectory =
      state->world.oi_settle_decline_no_suspended_trajectory;
  receipt.rewrite_oi_settle_decline_suspended_lane3_zero =
      state->world.oi_settle_decline_suspended_lane3_zero;
  receipt.rewrite_oi_settle_decline_suspended_owner_mismatch =
      state->world.oi_settle_decline_suspended_owner_mismatch;
  receipt.rewrite_oi_settle_decline_suspended_not_open_inquiry =
      state->world.oi_settle_decline_suspended_not_open_inquiry;
  receipt.rewrite_oi_settle_decline_witness_owner_mismatch =
      state->world.oi_settle_decline_witness_owner_mismatch;
  receipt.rewrite_oi_settle_decline_witness_selected_owner_mismatch =
      state->world.oi_settle_decline_witness_selected_owner_mismatch;
  receipt.rewrite_oi_settle_decline_witness_selected_revision_mismatch =
      state->world.oi_settle_decline_witness_selected_revision_mismatch;
  receipt.rewrite_oi_settle_decline_witness_reply_revision_mismatch =
      state->world.oi_settle_decline_witness_reply_revision_mismatch;
  receipt.rewrite_oi_settle_decline_witness_reply_length_mismatch =
      state->world.oi_settle_decline_witness_reply_length_mismatch;
  receipt.rewrite_oi_settle_decline_witness_reply_tail_mismatch =
      state->world.oi_settle_decline_witness_reply_tail_mismatch;
  receipt.rewrite_oi_settle_decline_witness_not_external =
      state->world.oi_settle_decline_witness_not_external;
  receipt.rewrite_oi_settle_decline_witness_revision_not_one =
      state->world.oi_settle_decline_witness_revision_not_one;
  receipt.rewrite_oi_settle_decline_alternative_count =
      state->world.oi_settle_decline_alternative_count;
  receipt.rewrite_oi_settle_decline_selected_binding_count =
      state->world.oi_settle_decline_selected_binding_count;
  receipt.rewrite_oi_settle_decline_selected_consequence_invalid =
      state->world.oi_settle_decline_selected_consequence_invalid;
  receipt.rewrite_oi_settle_decline_witness_count =
      state->world.oi_settle_decline_witness_count;
  receipt.rewrite_oi_settle_decline_witness_consequence_mismatch =
      state->world.oi_settle_decline_witness_consequence_mismatch;
  receipt.rewrite_oi_settle_decline_emission_count =
      state->world.oi_settle_decline_emission_count;
  receipt.rewrite_oi_settle_decline_selected_emission_count =
      state->world.oi_settle_decline_selected_emission_count;
  receipt.rewrite_oi_settle_reply_admitted =
      state->world.oi_settle_reply_admitted;
  receipt.rewrite_oi_reactivate_attempts = state->world.oi_reactivate_attempts;
  receipt.rewrite_oi_reactivate_admitted = state->world.oi_reactivate_admitted;
  receipt.rewrite_oi_reply_continuation_attempts =
      state->world.oi_reply_continuation_attempts;
  receipt.rewrite_oi_reply_continuation_admitted =
      state->world.oi_reply_continuation_admitted;
  receipt.rewrite_open_inquiry_surface_attempts =
      state->world.open_inquiry_surface_attempts;
  receipt.rewrite_open_inquiry_surface_decline_ambiguous_emission =
      state->world.open_inquiry_surface_decline_ambiguous_emission;
  receipt.rewrite_open_inquiry_surface_decline_no_emission =
      state->world.open_inquiry_surface_decline_no_emission;
  receipt.rewrite_open_inquiry_surface_decline_no_inquiry_header =
      state->world.open_inquiry_surface_decline_no_inquiry_header;
  receipt.rewrite_open_inquiry_surface_reply_dispatch =
      state->world.open_inquiry_surface_reply_dispatch;
  receipt.rewrite_open_inquiry_surface_decline_bad_kind =
      state->world.open_inquiry_surface_decline_bad_kind;
  receipt.rewrite_open_inquiry_surface_decline_constructor_stale =
      state->world.open_inquiry_surface_decline_constructor_stale;
  receipt.rewrite_open_inquiry_surface_decline_suspended_stale =
      state->world.open_inquiry_surface_decline_suspended_stale;
  receipt.rewrite_open_inquiry_surface_decline_snapshot_mismatch =
      state->world.open_inquiry_surface_decline_snapshot_mismatch;
  receipt.rewrite_open_inquiry_surface_decline_exhausted =
      state->world.open_inquiry_surface_decline_exhausted;
  receipt.rewrite_open_inquiry_surface_decline_term_lookup_failed =
      state->world.open_inquiry_surface_decline_term_lookup_failed;
  receipt.rewrite_open_inquiry_surface_decline_alternative_path_failed =
      state->world.open_inquiry_surface_decline_alternative_path_failed;
  receipt.rewrite_open_inquiry_surface_decline_append_failed =
      state->world.open_inquiry_surface_decline_append_failed;
  receipt.rewrite_open_inquiry_surface_word_emitted =
      state->world.open_inquiry_surface_word_emitted;
  receipt.rewrite_oi_ctor_gate_eligible = state->world.oi_ctor_gate_eligible;
  receipt.rewrite_oi_ctor_gate_resume_observed =
      state->world.oi_ctor_gate_resume_observed;
  receipt.rewrite_oi_ctor_settle_attempts =
      state->world.oi_ctor_settle_attempts;
  receipt.rewrite_oi_ctor_settle_decline_already_authoritative =
      state->world.oi_ctor_settle_decline_already_authoritative;
  receipt.rewrite_oi_ctor_settle_decline_episode_overflow =
      state->world.oi_ctor_settle_decline_episode_overflow;
  receipt.rewrite_oi_ctor_settle_decline_insufficient_episodes =
      state->world.oi_ctor_settle_decline_insufficient_episodes;
  receipt.rewrite_oi_ctor_settle_decline_conflicting_template =
      state->world.oi_ctor_settle_decline_conflicting_template;
  receipt.rewrite_oi_ctor_settle_decline_no_template =
      state->world.oi_ctor_settle_decline_no_template;
  receipt.rewrite_oi_ctor_settle_decline_insufficient_free_records =
      state->world.oi_ctor_settle_decline_insufficient_free_records;
  receipt.rewrite_oi_ctor_settle_decline_owner_failed =
      state->world.oi_ctor_settle_decline_owner_failed;
  receipt.rewrite_oi_ctor_settle_decline_header_alloc_failed =
      state->world.oi_ctor_settle_decline_header_alloc_failed;
  receipt.rewrite_oi_ctor_settle_decline_term_alloc_failed =
      state->world.oi_ctor_settle_decline_term_alloc_failed;
  receipt.rewrite_oi_ctor_settle_decline_witness_alloc_failed =
      state->world.oi_ctor_settle_decline_witness_alloc_failed;
  receipt.rewrite_oi_ctor_settle_decline_final_check_failed =
      state->world.oi_ctor_settle_decline_final_check_failed;
  receipt.rewrite_oi_ctor_settle_admitted = state->world.oi_ctor_settle_admitted;
  receipt.rewrite_oi_ctor_settle_last_episode_count =
      state->world.oi_ctor_settle_last_episode_count;
  receipt.rewrite_oi_episode_complete_attempts =
      state->world.oi_episode_complete_attempts;
  receipt.rewrite_oi_episode_complete_decline_not_open_inquiry_form =
      state->world.oi_episode_complete_decline_not_open_inquiry_form;
  receipt.rewrite_oi_episode_complete_decline_flags_incomplete =
      state->world.oi_episode_complete_decline_flags_incomplete;
  receipt.rewrite_oi_episode_complete_decline_lane2_invalid =
      state->world.oi_episode_complete_decline_lane2_invalid;
  receipt.rewrite_oi_episode_complete_decline_lane3_zero =
      state->world.oi_episode_complete_decline_lane3_zero;
  receipt.rewrite_oi_episode_complete_decline_lane4_not_two =
      state->world.oi_episode_complete_decline_lane4_not_two;
  receipt.rewrite_oi_episode_complete_decline_lane5_invalid =
      state->world.oi_episode_complete_decline_lane5_invalid;
  receipt.rewrite_oi_episode_complete_decline_lane6_invalid =
      state->world.oi_episode_complete_decline_lane6_invalid;
  receipt.rewrite_oi_episode_complete_decline_reserved1_not_invalid =
      state->world.oi_episode_complete_decline_reserved1_not_invalid;
  receipt.rewrite_oi_episode_complete_decline_reserved0_zero =
      state->world.oi_episode_complete_decline_reserved0_zero;
  receipt.rewrite_oi_episode_complete_decline_reserved0_too_long =
      state->world.oi_episode_complete_decline_reserved0_too_long;
  receipt.rewrite_oi_episode_complete_decline_surface_count_mismatch =
      state->world.oi_episode_complete_decline_surface_count_mismatch;
  receipt.rewrite_oi_episode_complete_decline_surface_source_owner_invalid =
      state->world.oi_episode_complete_decline_surface_source_owner_invalid;
  receipt.rewrite_oi_episode_complete_decline_reply_source_owner_invalid =
      state->world.oi_episode_complete_decline_reply_source_owner_invalid;
  receipt.rewrite_oi_episode_complete_decline_surface_word_lookup_failed =
      state->world.oi_episode_complete_decline_surface_word_lookup_failed;
  receipt.rewrite_oi_episode_complete_decline_alternative_not_grounded =
      state->world.oi_episode_complete_decline_alternative_not_grounded;
  receipt.rewrite_oi_episode_complete_decline_alternative_consensus_failed =
      state->world.oi_episode_complete_decline_alternative_consensus_failed;
  receipt.rewrite_oi_episode_complete_decline_alternatives_lookup_failed =
      state->world.oi_episode_complete_decline_alternatives_lookup_failed;
  receipt.rewrite_oi_episode_complete_decline_reply_witness_malformed =
      state->world.oi_episode_complete_decline_reply_witness_malformed;
  receipt.rewrite_oi_episode_complete_decline_resume_witness_malformed =
      state->world.oi_episode_complete_decline_resume_witness_malformed;
  receipt.rewrite_oi_episode_complete_decline_witness_count_mismatch =
      state->world.oi_episode_complete_decline_witness_count_mismatch;
  receipt.rewrite_oi_episode_complete_valid_count =
      state->world.oi_episode_complete_valid_count;
  receipt.rewrite_oi_reply_source_owner_attempts =
      state->world.oi_reply_source_owner_attempts;
  receipt.rewrite_oi_reply_source_owner_decline_ambiguous =
      state->world.oi_reply_source_owner_decline_ambiguous;
  receipt.rewrite_oi_reply_source_owner_decline_witness_owner_bad =
      state->world.oi_reply_source_owner_decline_witness_owner_bad;
  receipt.rewrite_oi_reply_source_owner_decline_witness_lane6_zero =
      state->world.oi_reply_source_owner_decline_witness_lane6_zero;
  receipt.rewrite_oi_reply_source_owner_decline_witness_not_external =
      state->world.oi_reply_source_owner_decline_witness_not_external;
  receipt.rewrite_oi_reply_source_owner_decline_reply_terms_invalid =
      state->world.oi_reply_source_owner_decline_reply_terms_invalid;
  receipt.rewrite_oi_reply_source_owner_decline_no_witness_found =
      state->world.oi_reply_source_owner_decline_no_witness_found;
  receipt.rewrite_oi_reply_source_owner_found =
      state->world.oi_reply_source_owner_found;
  receipt.rewrite_oi_reply_terms_attempts = state->world.oi_reply_terms_attempts;
  receipt.rewrite_oi_reply_terms_decline_witness_guard =
      state->world.oi_reply_terms_decline_witness_guard;
  receipt.rewrite_oi_reply_terms_decline_term_lookup_failed =
      state->world.oi_reply_terms_decline_term_lookup_failed;
  receipt.rewrite_oi_reply_terms_decline_term_lane4_owner_mismatch =
      state->world.oi_reply_terms_decline_term_lane4_owner_mismatch;
  receipt.rewrite_oi_reply_terms_decline_term_lane5_revision_mismatch =
      state->world.oi_reply_terms_decline_term_lane5_revision_mismatch;
  receipt.rewrite_oi_reply_terms_decline_term_lane6_length_mismatch =
      state->world.oi_reply_terms_decline_term_lane6_length_mismatch;
  receipt.rewrite_oi_reply_terms_decline_term_not_external =
      state->world.oi_reply_terms_decline_term_not_external;
  receipt.rewrite_oi_reply_terms_decline_term_revision_not_one =
      state->world.oi_reply_terms_decline_term_revision_not_one;
  receipt.rewrite_oi_reply_terms_last_bad_term_revision =
      state->world.oi_reply_terms_last_bad_term_revision;
  receipt.rewrite_oi_reply_terms_last_bad_term_slot =
      state->world.oi_reply_terms_last_bad_term_slot;
  receipt.rewrite_oi_reply_terms_decline_term_count_mismatch =
      state->world.oi_reply_terms_decline_term_count_mismatch;
  receipt.rewrite_oi_reply_terms_fastpath_matched =
      state->world.oi_reply_terms_fastpath_matched;
  receipt.rewrite_oi_reply_terms_decline_program_ambiguous =
      state->world.oi_reply_terms_decline_program_ambiguous;
  receipt.rewrite_oi_reply_terms_decline_program_not_found =
      state->world.oi_reply_terms_decline_program_not_found;
  receipt.rewrite_oi_reply_terms_decline_program_exact_mismatch =
      state->world.oi_reply_terms_decline_program_exact_mismatch;
  receipt.rewrite_oi_reply_terms_program_exact_matched =
      state->world.oi_reply_terms_program_exact_matched;
  receipt.rewrite_causal_germline_episodes =
      state->world.causal_germline_episodes;
  receipt.rewrite_causal_germline_constructors =
      state->world.causal_germline_constructors;
  receipt.rewrite_causal_germline_applications =
      state->world.causal_germline_applications;
  receipt.rewrite_causal_germline_reconstructions =
      state->world.causal_germline_reconstructions;
  receipt.rewrite_causal_germline_counterevidence =
      state->world.causal_germline_counterevidence;
  receipt.rewrite_causal_germline_product_suppressions =
      state->world.causal_germline_product_suppressions;
  receipt.rewrite_causal_germline_constructor_suppressions =
      state->world.causal_germline_constructor_suppressions;
  receipt.rewrite_causal_germline_conflict_abstentions =
      state->world.causal_germline_conflict_abstentions;
  receipt.rewrite_causal_germline_constructor_locus =
      state->world.causal_germline_constructor_locus;
  receipt.rewrite_causal_germline_product_locus =
      state->world.causal_germline_product_locus;
  receipt.rewrite_organization_digest = state->world.organization_digest;

  receipt.rewrite_world_lineage_version = kResidentLineageReceiptVersion;
  receipt.rewrite_world_lineage_valid = 0u;
  receipt.rewrite_world_lineage_revision = 0u;
  receipt.rewrite_world_lineage_organization_digest = 0u;
  receipt.rewrite_world_lineage_admitted_events = 0u;
  std::uint64_t lineage_organization = 0u;
  if (derive_resident_world_lineage(state, &lineage_organization)) {
    receipt.rewrite_world_lineage_valid = 1u;
    receipt.rewrite_world_lineage_revision = state->world.revision;
    receipt.rewrite_world_lineage_organization_digest = lineage_organization;
    receipt.rewrite_world_lineage_admitted_events = state->world.admitted_events;
  }

  receipt.rewrite_open_inquiry_identity_version =
      kResidentLineageReceiptVersion;
  receipt.rewrite_open_inquiry_identity_valid = 0u;
  receipt.rewrite_open_inquiry_owner = rewrite::kInvalid;
  receipt.rewrite_open_inquiry_identity = 0u;
  receipt.rewrite_open_inquiry_generation = 0u;
  std::uint32_t inquiry_owner = rewrite::kInvalid;
  std::uint32_t inquiry_identity = 0u;
  std::uint32_t inquiry_generation = 0u;
  if (derive_active_open_inquiry_identity(
          state, &inquiry_owner, &inquiry_identity, &inquiry_generation)) {
    receipt.rewrite_open_inquiry_identity_valid = 1u;
    receipt.rewrite_open_inquiry_owner = inquiry_owner;
    receipt.rewrite_open_inquiry_identity = inquiry_identity;
    receipt.rewrite_open_inquiry_generation = inquiry_generation;
  }
  receipt.legacy_action_authority = 0u;
  receipt.f_owned_clock = 0u;
  receipt.f_fault = state->f_fault;
  receipt.f_active_count = 0u;
  receipt.f_continuation_phase = 0u;
  receipt.f_continuation_status = 0u;
  receipt.completed_f_ticks = 0u;
  receipt.f_generation = 0u;
  receipt.f_motor_zero = 0u;
  receipt.f_motor_one = 0u;
  if (state->f_owned_clock != 0u && state->ordinary_f.publication != nullptr) {
    const ordinary_f::OrdinaryFPublication publication =
        *state->ordinary_f.publication;
    receipt.genesis_manifest = state->f_genesis_manifest;
    digest_bytes(&publication.world, sizeof(publication.world),
                 &receipt.f_world, 0x4f5244494e415259ull);
    receipt.completed_f_ticks = publication.completed_ticks;
    receipt.f_generation = publication.generation;
    receipt.f_fault |= publication.fault;
    receipt.f_active_count = publication.active_count;
    receipt.f_owned_clock = 1u;
    receipt.f_continuation_phase = publication.continuation_phase;
    receipt.f_continuation_status = publication.return_launch_status;
    receipt.f_motor_zero = publication.motor.zero;
    receipt.f_motor_one = publication.motor.one;
    receipt.legacy_action_authority = 1u;
  } else {
    receipt.f_world = ContentAddress{};
  }
  // 0X1-267: restores the predictive-shadow receipt-counter wiring
  // (a3d935d1e4/54a42b7a86) that de6857fa57's unrelated canonical-action-
  // surface fix silently dropped from this function. Without this, every
  // predictive_shadow_* receipt field below freezes at 0u for the process
  // lifetime, which is exactly the regression
  // bcc32_cuda_resident_action_return_mixed_provenance_production_contract's
  // "accepted ticketed returns did not commit predictive-shadow contact"
  // assertion (ticketed_training_returns) would now catch on real GPU.
  receipt.predictive_shadow_external_matches =
      state->predictive_shadow.receipt.external_matches;
  receipt.predictive_shadow_external_violations =
      state->predictive_shadow.receipt.external_violations;
  receipt.predictive_shadow_omissions = state->predictive_shadow.receipt.omissions;
  receipt.predictive_shadow_relations_formed =
      state->predictive_shadow.receipt.relations_formed;
  receipt.predictive_shadow_external_contacts =
      state->predictive_shadow.receipt.external_contacts;
  // 0X1-267 requirement 5, step 1 (Linear 0X1-267 comment c98064fd): expose
  // the diagnostic residue/route-morphology intersection probe DeviceState
  // already computed this epoch in advance_resident_cognition_phase. Same
  // idiom as the counters immediately above.
  receipt.predictive_shadow_route_probe_eligible_morphology =
      state->predictive_shadow_route_probe_eligible_morphology;
  receipt.predictive_shadow_route_probe_intersection =
      state->predictive_shadow_route_probe_intersection;
  receipt.predictive_shadow_route_probe_intersection_popcount =
      state->predictive_shadow_route_probe_intersection_popcount;
  receipt.action_return_issued = state->action_return_issued;
  receipt.action_return_accepted = state->action_return_accepted;
  receipt.action_return_rejected = state->action_return_rejected;
  receipt.action_return_action_sequence = state->action_return_last_action_sequence;
  receipt.action_return_pending_action_sequence = state->action_return_ticket.action_sequence;
  receipt.action_return_contact_sequence = state->action_return_contact_sequence;
  receipt.action_return_contact_words = state->action_return_contact_words;
  receipt.action_return_stream_words = state->action_return_stream_words;
  receipt.action_return_stream_next_chunk = state->action_return_stream_next_chunk;
  receipt.action_return_stream_active = state->action_return_stream_active;
  receipt.action_return_pending = state->action_return_ticket.nonce != 0u ? 1u : 0u;
  receipt.action_return_ticketed_external_return =
      (accepted_action_return && !device_body_return) ||
              state->world.open_inquiry_public_return_receipt != 0u
          ? 1u
          : 0u;
  receipt.action_return_constraint_reafferent_attempted =
      state->action_return_constraint_reafferent_attempted;
  receipt.action_return_constraint_reafferent_accepted =
      state->action_return_constraint_reafferent_accepted;
  receipt.action_return_constraint_reafferent_rejected =
      state->action_return_constraint_reafferent_rejected;
  receipt.action_return_constraint_countered_records =
      state->action_return_constraint_countered_records;
  receipt.action_return_constraint_admitted_records =
      state->action_return_constraint_admitted_records;
  receipt.action_return_constraint_resident_revision =
      state->action_return_constraint_resident_revision;
  receipt.action_return_constraint_component_ready =
      state->action_return_constraint_component_ready;
  receipt.action_return_constraint_component_ambiguous =
      state->action_return_constraint_component_ambiguous;
  receipt.action_return_constraint_component_records =
      state->action_return_constraint_component_records;
  receipt.action_return_constraint_component_sources =
      state->action_return_constraint_component_sources;
  receipt.action_return_constraint_rederived_event =
      state->action_return_constraint_rederived_event;
  receipt.rewrite_participation_end_attempted =
      state->rewrite_participation_end_attempted;
  receipt.rewrite_participation_end_admitted =
      state->rewrite_participation_end_admitted;
  receipt.rewrite_participation_end_rejected =
      state->rewrite_participation_end_rejected;
  receipt.rewrite_participation_end_materialized_records =
      state->rewrite_participation_end_materialized_records;
  receipt.rewrite_participation_end_precommit_records =
      state->rewrite_participation_end_precommit_records;
  receipt.rewrite_participation_end_committed_records =
      state->rewrite_participation_end_committed_records;
  // These three are the ONLY assignments to these rails anywhere in the
  // runtime: they are unimplemented placeholders, permanently zero since
  // 4b1678ccb1 introduced them that way. Every `== 0u` assertion on them in the
  // contract suite is therefore vacuous. See the declaration comment in
  // bcc32_persistent_kernel.hpp before treating them as provenance evidence.
  receipt.action_return_physical_consequence_proven = 0u;
  receipt.action_return_body_reafference_proven = 0u;
  receipt.action_return_source_identity_proven = 0u;
  receipt.action_return_device_body_enabled = device_body_attached ? 1u : 0u;
  receipt.action_return_device_body_closed_loop = device_body_return ? 1u : 0u;
  receipt.action_return_device_body_producer_instance = state->device_body_producer_instance;
  receipt.action_return_device_body_source_epoch = state->device_body_source_epoch;
  receipt.action_return_device_body_route_sequence = state->device_body_last_route_sequence;
  receipt.action_return_device_body_state = state->device_body_state;
  receipt.action_return_device_body_transition_count = state->device_body_transition_count;
  receipt.action_return_device_body_consequence_word =
      state->action_return_device_body_consequence_word;
  receipt.action_return_world_cell_slot = state->action_return_world_cell_slot;
  receipt.action_return_world_claim_slot = state->action_return_world_claim_slot;
  receipt.action_return_world_write_count = state->action_return_world_write_count;
  receipt.action_return_contact = state->action_return_contact;

  ContentAddress input{};
  const std::uint64_t admitted_marker = consumed_contact ? state->contact_sequence : 0u;
  digest_bytes(&admitted_marker, sizeof(admitted_marker), &input, 0x494e5055545f5257ull);
  receipt.input = input;
  OutputMaterial output{};
  for (std::uint32_t i = 0u; i < kActionWords; ++i)
    output.actions[i] = derived.actions[i];
  output.action_count = derived.action_count;
  for (std::uint32_t i = 0u; i < kLanguageBytes; ++i)
    output.language[i] = derived.language[i];
  output.language_count = derived.language_count;
  output.egress_history = state->egress_history_digest;
  output.egress_history_next_sequence = state->egress_history.next_sequence;
  output.egress_history_oldest_sequence = state->egress_history.oldest_sequence;
  output.egress_history_overwrite_count = state->egress_history.overwrite_count;
  output.egress_history_fault = state->egress_history.fault;
  digest_bytes(&output, sizeof(output), &receipt.output, 0x4f55545055545f52ull);

  struct CommitmentMaterial {
    ContentAddress world;
    ContentAddress input;
    ContentAddress output;
    ContentAddress predecessor;
    ContentAddress law;
    ContentAddress image;
    ContentAddress genesis;
    std::uint64_t tick;
    std::uint64_t contact_sequence;
    std::uint64_t intervention_sequence;
  } material{receipt.rewrite_world,       receipt.input, receipt.output,
             receipt.predecessor,         receipt.law,   receipt.image,
             receipt.genesis_manifest,    receipt.tick,  receipt.contact_sequence,
             state->intervention_sequence};
  digest_bytes(&material, sizeof(material), &receipt.commitment, 0x434f4d4d49545f52ull);
  state->predecessor = receipt.commitment;
}

__device__ void publish_egress(const DeviceState* state, const DerivedOutput& derived,
                               bool history_changed, EgressState* egress) {
  cuda::atomic_ref<std::uint64_t, cuda::thread_scope_system> generation(egress->generation);
  const std::uint64_t current = generation.load(cuda::memory_order_relaxed);
  generation.store(current + 1u, cuda::memory_order_release);
  for (std::uint32_t i = 0u; i < kActionWords; ++i)
    egress->actions[i] = derived.actions[i];
  egress->action_count = derived.action_count;
  for (std::uint32_t i = 0u; i < kLanguageBytes; ++i)
    egress->language[i] = derived.language[i];
  egress->language_count = derived.language_count;
  egress->action_return_ticket = state->action_return_ticket;
  if (history_changed) {
    cuda::atomic_ref<std::uint64_t, cuda::thread_scope_system> history_generation(
        egress->history_generation);
    const std::uint64_t history_current = history_generation.load(cuda::memory_order_relaxed);
    history_generation.store(history_current + 1u, cuda::memory_order_release);
    if (egress_history::retained_count(&state->egress_history) != 0u) {
      const std::uint64_t newest = egress_history::newest_sequence(&state->egress_history);
      const std::size_t event_index =
          static_cast<std::size_t>((newest - 1u) % egress_history::capacity);
      const std::size_t event_offset = event_index * sizeof(egress_history::Event);
      const auto* event_bytes =
          reinterpret_cast<const std::uint8_t*>(&state->egress_history.events[event_index]);
      for (std::size_t i = 0u; i < sizeof(egress_history::Event); ++i)
        egress->egress_history_bytes[event_offset + i] = event_bytes[i];
    }
    egress->egress_history_next_sequence = state->egress_history.next_sequence;
    egress->egress_history_oldest_sequence = state->egress_history.oldest_sequence;
  egress->egress_history_overwrite_count = state->egress_history.overwrite_count;
  egress->egress_history_fault = state->egress_history.fault;
    __threadfence_system();
    history_generation.store(history_current + 2u, cuda::memory_order_release);
  }
  egress->receipt = state->receipt;
  egress->energy = state->world.concrete_descriptions * rewrite::kRecordMatterQ8;
  egress->host_bootstrap_launches = state->host_bootstrap_launches;
  egress->device_epochs = state->device_epochs;
  __threadfence_system();
  generation.store(current + 2u, cuda::memory_order_release);
}

// Host-side reader paired with publish_egress above: a lock-free,
// retry-on-torn-generation copy of the current EgressState into a
// PassiveSnapshot. This is the sole producer of PersistentKernel snapshots.
void copy_snapshot(const EgressState* egress, const Lifecycle* lifecycle,
                   std::vector<std::uint8_t>* history_cache, std::uint64_t* cache_next,
                   std::uint64_t* cache_oldest, std::uint64_t* cache_overwrites,
                   std::uint32_t* cache_fault, PassiveSnapshot* result) {
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
  for (;;) {
    const std::uint64_t before = load_host(&egress->generation);
    if (before != 0u && (before & 1u) == 0u) {
      const std::uint32_t count = std::min<std::uint32_t>(egress->action_count, kActionWords);
      const std::uint32_t language_count =
          std::min<std::uint32_t>(egress->language_count, kLanguageBytes);
      std::array<BoundaryWord, kActionWords> actions{};
      std::array<std::uint8_t, kLanguageBytes> language{};
      for (std::uint32_t i = 0u; i < count && i < kActionWords; ++i)
        actions[i] = egress->actions[i];
      for (std::uint32_t i = 0u; i < language_count && i < kLanguageBytes; ++i)
        language[i] = egress->language[i];
      const TickReceipt receipt = egress->receipt;
      const std::uint64_t energy = egress->energy;
      const std::uint64_t launches = egress->host_bootstrap_launches;
      const std::uint64_t epochs = egress->device_epochs;
      const std::uint64_t history_next = egress->egress_history_next_sequence;
      const std::uint64_t history_oldest = egress->egress_history_oldest_sequence;
      const std::uint64_t history_overwrites = egress->egress_history_overwrite_count;
      const std::uint32_t history_fault = egress->egress_history_fault;
      const ActionReturnTicket action_return_ticket = egress->action_return_ticket;
      std::atomic_thread_fence(std::memory_order_acquire);
      const std::uint64_t after = load_host(&egress->generation);
      if (before == after && (after & 1u) == 0u) {
        const std::uint64_t history_before = load_host(&egress->history_generation);
        if ((history_before & 1u) == 0u) {
          if (history_cache->size() != kEgressHistoryBytes) {
            history_cache->assign(kEgressHistoryBytes, 0u);
            *cache_next = 0u;
          }
          const bool full_copy = *cache_next == 0u || history_next < *cache_next ||
                                 history_next - *cache_next > egress_history::capacity;
          if (full_copy) {
            std::memcpy(history_cache->data(), egress->egress_history_bytes, kEgressHistoryBytes);
          } else {
            for (std::uint64_t sequence = *cache_next; sequence < history_next; ++sequence) {
              const std::size_t event_offset =
                  static_cast<std::size_t>((sequence - 1u) % egress_history::capacity) *
                  sizeof(egress_history::Event);
              std::memcpy(history_cache->data() + event_offset,
                          egress->egress_history_bytes + event_offset,
                          sizeof(egress_history::Event));
            }
          }
          const std::uint64_t copied_next = egress->egress_history_next_sequence;
          const std::uint64_t copied_oldest = egress->egress_history_oldest_sequence;
          const std::uint64_t copied_overwrites = egress->egress_history_overwrite_count;
          const std::uint32_t copied_fault = egress->egress_history_fault;
          std::atomic_thread_fence(std::memory_order_acquire);
          const std::uint64_t history_after = load_host(&egress->history_generation);
          if (history_before == history_after && (history_after & 1u) == 0u &&
              copied_next == history_next && copied_oldest == history_oldest &&
              copied_overwrites == history_overwrites && copied_fault == history_fault) {
            *cache_next = history_next;
            *cache_oldest = history_oldest;
            *cache_overwrites = history_overwrites;
            *cache_fault = history_fault;
            result->actions.assign(actions.begin(), actions.begin() + count);
            result->language_bytes.assign(language.begin(), language.begin() + language_count);
            result->egress_history_bytes = *history_cache;
            result->egress_history_next_sequence = *cache_next;
            result->egress_history_oldest_sequence = *cache_oldest;
            result->egress_history_overwrite_count = *cache_overwrites;
            result->egress_history_fault = *cache_fault;
            result->action_return_ticket = action_return_ticket;
            result->receipt = receipt;
            result->energy = energy;
            result->host_bootstrap_launches = launches;
            result->device_epochs = epochs;
            result->continuation_fault = load_host(&lifecycle->continuation_fault);
            return;
          }
        }
      }
    }
    if (std::chrono::steady_clock::now() >= deadline)
      throw std::runtime_error(
          "resident rewrite egress snapshot timed out: generation=" +
          std::to_string(load_host(&egress->generation)) +
          " continuation_fault=" +
          std::to_string(load_host(&lifecycle->continuation_fault)) +
          " ordinary_f_active_count=" +
          std::to_string(load_host(&lifecycle->ordinary_f_active_count)) +
          " stopped=" + std::to_string(load_host(&lifecycle->stopped)));
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
}
