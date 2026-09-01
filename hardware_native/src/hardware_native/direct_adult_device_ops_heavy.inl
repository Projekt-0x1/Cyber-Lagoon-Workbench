// Heavy shared adult-runtime operations. This file is included inline only by
// compatibility consumers; direct_adult_core compiles one strong device owner.

DIRECT_ADULT_DEVICE_OPS_HEAVY_QUALIFIER bool publish_admitted_resident_actual_contacts(
    DirectBrain brain, ResidentActualFrontier* frontier,
    NodeCausalParticipation* participation, std::uint32_t* participation_locks,
    EligibilityRecord* eligibility_table, std::uint32_t* live_eligibility_count,
    std::uint64_t* eligibility_claim_directory,
    std::uint32_t* eligibility_record_generations,
    const std::uint32_t* eligibility_free_slots,
    std::uint32_t* eligibility_free_state,
    DirectParticipationDescriptor* staging, std::uint32_t* staging_count,
    std::uint32_t staging_capacity, std::int32_t* incoming_excitation,
    AdultCoreMetrics* metrics, std::uint32_t current_tick) {
  if (frontier == nullptr) return false;
  bool published = false;
  for (std::uint32_t rank = 0u; rank < kResidentActualFrontierCapacity;
       ++rank) {
    std::uint32_t selected = kInvalidIndex;
    std::uint64_t selected_sequence = ~0ULL;
    for (std::uint32_t i = 0u; i < kResidentActualFrontierCapacity; ++i) {
      const auto& candidate = frontier->pending_contacts[i];
      if (candidate.state != ResidentPendingActualContactState::admitted ||
          candidate.receipt.ingress_sequence > selected_sequence)
        continue;
      if (candidate.receipt.ingress_sequence < selected_sequence ||
          selected == kInvalidIndex || i < selected) {
        selected = i;
        selected_sequence = candidate.receipt.ingress_sequence;
      }
    }
    if (selected == kInvalidIndex) break;
    auto& pending = frontier->pending_contacts[selected];
    if (!publish_resident_actual_occurrence_contribution(
            brain, *frontier, pending.receipt, pending.event, participation,
            participation_locks, eligibility_table, live_eligibility_count,
            eligibility_claim_directory, eligibility_record_generations,
            eligibility_free_slots, eligibility_free_state, staging,
            staging_count, staging_capacity, incoming_excitation, metrics,
            current_tick))
      break;
    pending = ResidentPendingActualContact{};
    published = true;
  }
  return published;
}

DIRECT_ADULT_DEVICE_OPS_HEAVY_QUALIFIER DirectActionBindingResult bind_action_occurrence(
    const DirectParticipationDescriptor* current_contributions,
    const std::uint32_t* current_contribution_count,
    std::uint32_t current_contribution_capacity,
    const ResidentActualFrontier* actual_frontier,
    std::uint32_t require_exact_occurrence_identity,
    std::uint32_t motor_node,
    std::uint32_t current_tick,
    std::uint64_t action_ticket_id,
    std::uint32_t action_context,
    std::uint32_t motor_channel,
    std::uint32_t action_expiry_tick,
    std::uint32_t action_slot,
    DirectActionOccurrence* action_occurrences,
    DirectActionParticipationLink* action_links,
    std::uint32_t participant_capacity,
    AdultCoreMetrics* metrics,
    const ResidentActivationSoaPlane* activation_plane,
    DirectBrain ancestry_brain,
    const EligibilityRecord* eligibility_table,
    const std::uint32_t* eligibility_record_generations) {
  DirectActionBindingResult result{};
  result.diagnostic_upstream_ticket = kInvalidTicket;
  if (action_occurrences == nullptr || action_slot >= kMaxAsynchronousTickets ||
      participant_capacity > kMaxActionParticipationLinks) {
    if (metrics != nullptr)
      atomicAdd(reinterpret_cast<unsigned long long*>(
                    &metrics->action_ancestry_backpressure), 1ULL);
    return result;
  }

  DirectParticipationDescriptor participants[kMaxActionParticipationLinks]{};
  std::uint32_t participant_count = 0u;
  bool ancestry_incomplete = current_contribution_count == nullptr;
  bool unrepresented_context = false, participant_overflow = false;
  const std::uint32_t staged = current_contribution_count != nullptr
                                   ? atomicAdd(const_cast<std::uint32_t*>(
                                                   current_contribution_count), 0u)
                                   : 0u;
  if (staged > current_contribution_capacity) ancestry_incomplete = true;
  const std::uint32_t stored = staged < current_contribution_capacity
                                   ? staged
                                   : current_contribution_capacity;
  if (stored != 0u && current_contributions == nullptr) ancestry_incomplete = true;
  if (current_contributions != nullptr) {
    for (std::uint32_t index = 0u; index < stored; ++index) {
      const DirectParticipationDescriptor value = current_contributions[index];
      if (value.target_node != motor_node) continue;
      if (value.contribution_kind == DirectContributionKind::ancestry_incomplete) {
        // Unrepresented context stays incomplete without erasing exact sparse participants.
        unrepresented_context = true; continue;
      }
      if ((value.contribution_kind != DirectContributionKind::direct_ingress &&
           value.contribution_kind != DirectContributionKind::sparse_route) ||
          value.ticket_id == 0ULL || value.ticket_id == kInvalidTicket ||
          value.expiry_tick < current_tick ||
          (value.contribution_kind == DirectContributionKind::sparse_route &&
           (value.eligibility_slot == kInvalidIndex ||
            value.eligibility_generation == 0u ||
            value.frozen_eligibility_q16 <= 0))) {
        ancestry_incomplete = true;
        continue;
      }
      bool duplicate = false;
      for (std::uint32_t i = 0u; i < participant_count; ++i) {
        if (!same_action_claim(value, participants[i])) continue;
        if (value.expiry_tick > participants[i].expiry_tick)
          participants[i].expiry_tick = value.expiry_tick;
        if (value.lineage_expiry_tick >
            participants[i].lineage_expiry_tick)
          participants[i].lineage_expiry_tick =
              value.lineage_expiry_tick;
        duplicate = true;
      }
      if (duplicate) continue;
      if (participant_count == kMaxProvenanceSlotsPerNode) {
        participant_overflow = true;
        continue;
      }
      participants[participant_count++] = value;
    }
  }
  if (participant_count == 0u) ancestry_incomplete = true;

  for (std::uint32_t i = 1u; i < participant_count; ++i) {
    const DirectParticipationDescriptor value = participants[i];
    std::uint32_t j = i;
    while (j != 0u && action_claim_less(value, participants[j - 1u])) {
      participants[j] = participants[j - 1u];
      --j;
    }
    participants[j] = value;
  }

  bool terminal_participant[kMaxActionParticipationLinks]{};
  if (require_exact_occurrence_identity != 0u &&
      (eligibility_table == nullptr ||
       eligibility_record_generations == nullptr))
    ancestry_incomplete = true;
  if (!ancestry_incomplete && require_exact_occurrence_identity != 0u &&
      eligibility_table != nullptr &&
      eligibility_record_generations != nullptr) {
    const std::uint32_t terminal_count = participant_count;
    DirectParticipationDescriptor terminals[kMaxProvenanceSlotsPerNode]{};
    for (std::uint32_t i = 0u; i < terminal_count; ++i) terminals[i] = participants[i];
    participant_count = 0u;
    for (std::uint32_t terminal_index = 0u; terminal_index < terminal_count;
         ++terminal_index) {
      DirectParticipationDescriptor closure[kMaxProvenanceSlotsPerNode]{};
      std::uint32_t closure_count = 0u;
      if (!freeze_eligibility_ancestry(
              terminals[terminal_index], ancestry_brain, eligibility_table,
              eligibility_record_generations, current_tick, closure,
              &closure_count)) {
        ancestry_incomplete = true;
        break;
      }
      for (std::uint32_t closure_index = 0u; closure_index < closure_count;
           ++closure_index) {
        bool duplicate = false;
        for (std::uint32_t i = 0u; i < participant_count; ++i)
          duplicate |= same_action_claim(closure[closure_index], participants[i]);
        if (duplicate) continue;
        if (participant_count == participant_capacity ||
            participant_count == kMaxActionParticipationLinks) {
          participant_overflow = true;
          break;
        }
        participants[participant_count] = closure[closure_index];
        terminal_participant[participant_count] =
            closure_index + 1u == closure_count;
        ++participant_count;
      }
      if (participant_overflow) break;
    }
  } else {
    for (std::uint32_t i = 0u; i < participant_count; ++i)
      terminal_participant[i] = true;
  }

  DirectActionOccurrence& incumbent = action_occurrences[action_slot];
  const std::uint32_t incumbent_state = atomicAdd(&incumbent.state, 0u);
  if (!ancestry_incomplete &&
      (participant_overflow || participant_count > participant_capacity ||
       (participant_count != 0u && action_links == nullptr))) {
    if (metrics != nullptr)
      atomicAdd(reinterpret_cast<unsigned long long*>(
                    &metrics->action_ancestry_backpressure), 1ULL);
    return result;
  }
  if (incumbent_state == kActionOccurrencePending) {
    if (incumbent.expiry_tick >= current_tick ||
        !expire_pending_action_occurrence(&incumbent, current_tick, metrics) ||
        atomicAdd(&incumbent.state, 0u) != kActionOccurrenceExpired) {
      if (metrics != nullptr)
        atomicAdd(reinterpret_cast<unsigned long long*>(
                      &metrics->action_ancestry_backpressure), 1ULL);
      return result;
    }
  } else if (incumbent_state != kActionOccurrenceFree &&
             incumbent_state != kActionOccurrenceSettled &&
             incumbent_state != kActionOccurrenceExpired) {
    if (metrics != nullptr)
      atomicAdd(reinterpret_cast<unsigned long long*>(
                    &metrics->action_ancestry_backpressure), 1ULL);
    return result;
  }

  const std::uint32_t offset =
      ancestry_incomplete ? kInvalidIndex
                          : action_slot * participant_capacity;
  bool occurrence_identity_required = false, occurrence_identity_complete = true;
  for (std::uint32_t i = 0u; i < participant_count && !ancestry_incomplete; ++i) {
    const DirectParticipationDescriptor participant = participants[i];
    DirectActionParticipationLink link{};
    link.participant_ticket_id = participant.ticket_id;
    link.source_node = participant.source_node;
    link.target_node = participant.target_node;
    link.route_index = participant.route_index;
    link.route_incarnation = participant.route_incarnation;
    link.context_signature = participant.context_signature;
    link.expiry_tick = participant.expiry_tick;
    link.claim_incarnation = participant.claim_incarnation;
    link.authority_incarnation = participant.authority_incarnation;
    link.authority = participant.authority;
    link.contribution_kind = participant.contribution_kind;
    link.frozen_eligibility_q16 = participant.frozen_eligibility_q16;
    link.eligibility_slot = participant.eligibility_slot;
    link.eligibility_generation = participant.eligibility_generation;
    const bool occurrence_candidate = require_exact_occurrence_identity != 0u &&
        terminal_participant[i] &&
        (participant.contribution_kind == DirectContributionKind::direct_ingress ||
         (participant.contribution_kind == DirectContributionKind::sparse_route &&
          participant.route_index != kInvalidIndex));
    const bool occurrence_frozen = occurrence_candidate && freeze_actual_occurrence_identity(participant, actual_frontier, &link, activation_plane);
    const bool pending_bootstrap = occurrence_candidate &&
        pending_actual_contact_bootstrap_current(
            actual_frontier, participant, current_tick);
    const bool requires_occurrence = occurrence_candidate &&
        (occurrence_frozen ||
         (participant.contribution_kind == DirectContributionKind::sparse_route &&
          participant.route_incarnation !=
              direct_network::initial_route_incarnation(participant.route_index) &&
          !pending_bootstrap));
    occurrence_identity_required |= requires_occurrence;
    if (requires_occurrence && !occurrence_frozen) occurrence_identity_complete = false;
    action_links[offset + i] = link;
  }

  std::uint64_t network_identity = 0u, recruitment_identity = 0u;
  std::int64_t network_eligibility_signed_q16 = 0;
  std::uint64_t network_eligibility_l1_q16 = 0u;
  if (!ancestry_incomplete && require_exact_occurrence_identity != 0u &&
      participant_count != 0u &&
      !resident_action_recruited_network_identity(
          ancestry_brain, actual_frontier, action_links, offset,
          &participant_count, participant_capacity, current_contributions,
          stored, eligibility_table, eligibility_record_generations,
          current_tick, &network_identity,
          &recruitment_identity, &network_eligibility_signed_q16,
          &network_eligibility_l1_q16)) {
    if (metrics != nullptr)
      atomicAdd(reinterpret_cast<unsigned long long*>(
                    &metrics->action_ancestry_backpressure), 1ULL);
    return result;
  }

  // C is compact relation depth, not causal ancestry D, trajectory extent X,
  // terminal claims T or action-link capacity A. A multi-Occurrence action
  // unfolds one level beyond its deepest resident motor-ground constituent.
  // Refuse the entire prospective action before publication at the bounded C
  // ceiling; no partial ActionOccurrence or motor state is committed.
  if (!ancestry_incomplete && participant_count > 1u) {
    std::uint32_t constituent_depth = 0u;
    for (std::uint32_t i = 0u; i < participant_count; ++i)
      if (action_links[offset + i].composition_depth > constituent_depth)
        constituent_depth = action_links[offset + i].composition_depth;
    const std::uint32_t prospective_depth = constituent_depth + 1u;
    if (prospective_depth > kMaxActiveCompositionDepth) {
      if (metrics != nullptr)
        atomicAdd(reinterpret_cast<unsigned long long*>(
                      &metrics->composition_depth_refusals), 1ULL);
      return result;
    }
    if (metrics != nullptr)
      atomicMax(reinterpret_cast<unsigned long long*>(
                    &metrics->active_composition_depth_peak),
                static_cast<unsigned long long>(prospective_depth));
  }

  DirectActionOccurrence occurrence{};
  occurrence.action_ticket_id = action_ticket_id;
  occurrence.network_identity = network_identity;
  occurrence.recruitment_identity = recruitment_identity;
  occurrence.network_eligibility_signed_q16 =
      network_eligibility_signed_q16;
  occurrence.network_eligibility_l1_q16 = network_eligibility_l1_q16;
  occurrence.participant_offset = offset;
  occurrence.participant_count = ancestry_incomplete ? 0u : participant_count;
  occurrence.emission_tick = current_tick;
  occurrence.context_signature = action_context;
  occurrence.motor_node = motor_node;
  occurrence.motor_channel = motor_channel;
  occurrence.expiry_tick = action_expiry_tick;
  occurrence.occurrence_identity_required = occurrence_identity_required ? 1u : 0u;
  occurrence.occurrence_identity_complete =
      occurrence_identity_complete ? 1u : 0u;
  occurrence.state = kActionOccurrenceFree;
  incumbent = occurrence;
  __threadfence();
  atomicExch(&incumbent.state, kActionOccurrencePending);
  result.admitted = 1u;
  result.participant_offset = offset;
  result.participant_count = occurrence.participant_count;
  result.ancestry_incomplete = (ancestry_incomplete || unrepresented_context) ? 1u : 0u;
  if (participant_count == 1u && !ancestry_incomplete)
    result.diagnostic_upstream_ticket = participants[0].ticket_id;
  if (metrics != nullptr) {
    atomicAdd(reinterpret_cast<unsigned long long*>(
                  &metrics->action_occurrences_committed), 1ULL);
    atomicAdd(reinterpret_cast<unsigned long long*>(
                  &metrics->action_participant_links_committed),
              static_cast<unsigned long long>(occurrence.participant_count));
    if (ancestry_incomplete || unrepresented_context)
      atomicAdd(reinterpret_cast<unsigned long long*>(
                    &metrics->action_ancestry_incomplete), 1ULL);
  }
  return result;
}

DIRECT_ADULT_DEVICE_OPS_HEAVY_QUALIFIER void resolve_consequence_ticket_device(
    AsynchronousTicket* ticket_table, DirectActionOccurrence* action_occurrences, DirectActionParticipationLink* action_links,
    std::uint32_t participant_capacity, std::uint64_t target_ticket_id,
    Word returned_word, CausalOrigin origin, std::uint32_t admission_tick,
    direct_network::DirectExactHistoryHotPage* exact_history,
    std::uint32_t transport_cursor, DirectBrain brain,
    ResidentActualFrontier* actual_frontier,
    const ResidentMultiHorizonPredictionFrontier* causal_credit_predictions,
    DirectNode* nodes, std::uint32_t node_count, DirectRoute* routes,
    const std::uint64_t* route_incarnations, std::uint32_t route_capacity,
    substrate::direct_adult::DirectRetentionState* retention_bank, direct_network::ResidentRecipeCell* recipe_cells,
    std::uint32_t recipe_cell_count, DirectDenseBlock* dense_blocks, std::uint32_t dense_block_count,
    std::int32_t learning_rate_q16, std::int32_t shatter_threshold_q16,
    ResolvedConsequenceContext* out_ctx, AdultCoreMetrics* metrics) {
  ResolvedConsequenceContext ctx{};
  ctx.target_ticket_id = target_ticket_id;
  ctx.upstream_ticket_id = kInvalidTicket;
  if (origin != CausalOrigin::world_return &&
      origin != CausalOrigin::motor_reafference) {
    *out_ctx = ctx;
    return;
  }
  if (target_ticket_id != kInvalidTicket && ticket_table != nullptr) {
    const std::uint32_t ticket_slot =
        static_cast<std::uint32_t>(target_ticket_id & 0x7ffu);
    AsynchronousTicket delayed_ticket{};
    DirectActionOccurrence delayed_action{};
    DirectActionParticipationLink delayed_links[kMaxActionParticipationLinks]{};
    auto* delayed_record = static_cast<
        direct_network::ResidentDevelopmentState::DelayedActionRecord*>(nullptr);
    AsynchronousTicket* selected_ticket = ticket_table + ticket_slot;
    DirectActionOccurrence* action = action_occurrences != nullptr
        ? action_occurrences + ticket_slot : nullptr;
    DirectActionParticipationLink* selected_links = action_links;
    if (selected_ticket->ticket_id != target_ticket_id) {
      delayed_record = load_preserved_action(brain.development,
          target_ticket_id, &delayed_ticket, &delayed_action, delayed_links);
      if (delayed_record != nullptr) {
        selected_ticket = &delayed_ticket;
        action = &delayed_action;
        selected_links = delayed_links;
        participant_capacity = kMaxActionParticipationLinks;
      }
    }
    AsynchronousTicket& ticket = *selected_ticket;
    const std::uint32_t action_state = action != nullptr
        ? atomicAdd(&action->state, 0u) : kActionOccurrenceFree;
    const bool action_exact = action != nullptr &&
                              action_state != kActionOccurrenceFree &&
                              action->action_ticket_id == target_ticket_id;
    const bool action_collision = action != nullptr &&
                                  action_state == kActionOccurrencePending &&
                                  !action_exact;
    const bool action_expired = delayed_record == nullptr && action_exact &&
        ((action_state == kActionOccurrencePending &&
          admission_tick > action->expiry_tick) ||
         action_state == kActionOccurrenceExpired);
    const std::uint32_t expected_offset = delayed_record != nullptr
        ? 0u : ticket_slot * participant_capacity;
    const bool ancestry_incomplete = action_exact && action->participant_count == 0u &&
                                     action->participant_offset == kInvalidIndex;
    const bool action_valid = !action_exact ||
        (action_state == kActionOccurrencePending &&
         action->participant_count <= participant_capacity &&
         (action->participant_offset == expected_offset || ancestry_incomplete) &&
         action->emission_tick == ticket.emission_tick &&
         action->context_signature == ticket.context_signature &&
         action->motor_node == ticket.motor_node &&
         action->motor_channel == ticket.motor_channel &&
         action->occurrence_identity_required <= 1u &&
         action->occurrence_identity_complete <= 1u &&
         (action->occurrence_identity_required == 0u ||
          action->occurrence_identity_complete != 0u) &&
         ((action->network_identity == 0u) ==
          (action->recruitment_identity == 0u)) &&
         (action->network_identity == 0u || action->participant_count >= 2u) &&
         (action->participant_count == 0u || selected_links != nullptr));
    if (ticket.ticket_id != target_ticket_id) {
      if (metrics != nullptr)
        atomicAdd(reinterpret_cast<unsigned long long*>(
                      &metrics->consequence_unknown_ticket_rejects), 1ULL);
    } else if (atomicAdd(&ticket.settled, 0u) != 0u) {
      if (metrics != nullptr)
        atomicAdd(reinterpret_cast<unsigned long long*>(
                      &metrics->consequence_duplicate_ticket_rejects), 1ULL);
    } else if (!action_exact || action_state != kActionOccurrencePending) {
      if (metrics != nullptr) {
        atomicAdd(reinterpret_cast<unsigned long long*>(
                      &metrics->raw_reafferent_authority_rejects),
                  1ULL);
      }
    } else if (action_expired) {
      if (action_state == kActionOccurrenceExpired ||
          expire_pending_action_occurrence(action, admission_tick, metrics)) {
        if (metrics != nullptr)
          atomicAdd(reinterpret_cast<unsigned long long*>(
                        &metrics->action_expired_rejects), 1ULL);
      } else if (metrics != nullptr) {
        atomicAdd(reinterpret_cast<unsigned long long*>(
                      &metrics->action_ancestry_backpressure), 1ULL);
      }
    } else if (action_collision || !action_valid) {
      if (metrics != nullptr)
        atomicAdd(reinterpret_cast<unsigned long long*>(
                      &metrics->action_ancestry_backpressure), 1ULL);
    } else {
      ctx.upstream_ticket_id = ticket.upstream_ticket_id;
      const std::uint32_t mismatch = static_cast<std::uint32_t>(__popc(ticket.motor_word ^ returned_word));
      ctx.mismatch_bits = mismatch;
      ctx.effective_reward_q16 = mismatch == 0u ? kQ16One : -kQ16One;
      if (exact_history == nullptr) { ctx.history_refused = 1u; *out_ctx = ctx; return; }
      direct_network::DirectExactHistoryRecord consequence_record{};
      direct_network::stage_world_return_history_record(
          &consequence_record, target_ticket_id, ticket.upstream_ticket_id,
          admission_tick, ticket.emission_tick, ticket.motor_node,
          ticket.motor_channel, returned_word, ticket.context_signature,
          mismatch, transport_cursor, ctx.effective_reward_q16,
          static_cast<std::uint32_t>(origin));
      std::uint32_t sparse_count = 0u, sparse_links[kMaxActionParticipationLinks]{};
      std::int32_t priors[kMaxActionParticipationLinks]{}, results[kMaxActionParticipationLinks]{}, applied_deltas[kMaxActionParticipationLinks]{};
      for (std::uint32_t i = 0u; i < action->participant_count; ++i) {
        const DirectActionParticipationLink& link = selected_links[action->participant_offset + i];
        const bool has_recipe_identity = link.logical_recipe_id != 0u || link.revision_identity != 0u || link.occurrence_identity != 0u || link.participation_identity != 0u || link.occurrence_route_incarnation != 0u;
        if (action->occurrence_identity_required != 0u && has_recipe_identity &&
            !frozen_action_occurrence_identity_current(
                link, brain, actual_frontier)) {
          *out_ctx = ctx;
          return;
        }
        if (link.contribution_kind == DirectContributionKind::sparse_route &&
            !frozen_sparse_action_link_current(
                link, nodes, node_count, routes, route_incarnations,
                route_capacity)) {
          *out_ctx = ctx;
          return;
        }
      }
      for (std::uint32_t i = 0u; i < action->participant_count; ++i) {
        const DirectActionParticipationLink& link = selected_links[action->participant_offset + i];
        if (link.contribution_kind != DirectContributionKind::sparse_route) continue;
        std::int32_t prior = routes[link.route_index].conductance_q16;
        for (std::uint32_t j = 0u; j < sparse_count; ++j) if (selected_links[action->participant_offset + sparse_links[j]].route_index == link.route_index) prior = results[j];
        // #1294: the tube's contextual chemistry scales how much credit moves
        // conductance -- a plasticity threshold modulation, never a reward.
        const auto tube = direct_network::direct_tube_chemistry_q16(
            nodes[link.source_node].chemotype, nodes[link.target_node].chemotype);
        std::int32_t requested_delta = mul_q16(mul_q16(
            mul_q16(mul_q16(learning_rate_q16, ctx.effective_reward_q16),
                    link.frozen_eligibility_q16), tube.plasticity_gain_q16),
            maturation_plasticity_gain_q16(
                brain, link.source_node,
                brain.development != nullptr ? brain.development->age_tick : admission_tick));
        // Prediction refines magnitude only; absent or inconsistent
        // endogenous evidence leaves the verified world-return delta intact.
        if (action->occurrence_identity_required != 0u && requested_delta != 0)
          refine_frozen_action_credit_precision(
              actual_frontier, causal_credit_predictions, link,
              requested_delta, &requested_delta);
        const std::int64_t requested = static_cast<std::int64_t>(prior) + requested_delta;
        const std::int32_t result = static_cast<std::int32_t>(requested < kMinConductanceQ16 ? kMinConductanceQ16 : requested > kMaxConductanceQ16 ? kMaxConductanceQ16 : requested);
        sparse_links[sparse_count] = i; priors[sparse_count] = prior;
        results[sparse_count] = result; applied_deltas[sparse_count] = result - prior;
        ++sparse_count;
      }
      // A verified world return is the third factor for the compact resident
      // Network recruitment. Exact active membership remains frozen only on the
      // action/history evidence; it is never copied into persistent recruitment.
      ResidentRecruitedNetworkCreditPlan network_credit_plan{};
      std::int32_t network_mean_eligibility_q16 = 0;
      if (action->network_eligibility_l1_q16 != 0u &&
          action->participant_count != 0u) {
        std::int64_t mean = action->network_eligibility_signed_q16 /
            static_cast<std::int64_t>(action->participant_count);
        if (mean < -0x80000000ll) mean = -0x80000000ll;
        if (mean > 0x7fffffffll) mean = 0x7fffffffll;
        network_mean_eligibility_q16 = static_cast<std::int32_t>(mean);
      }
      const std::int64_t requested_network_credit_q16 =
          origin == CausalOrigin::world_return &&
                  action->recruitment_identity != 0u
              ? static_cast<std::int64_t>(mul_q16(
                    mul_q16(learning_rate_q16, ctx.effective_reward_q16),
                    network_mean_eligibility_q16))
              : 0;
      if (!plan_resident_recruited_network_credit(
              brain.development, action->recruitment_identity,
              action->network_identity, requested_network_credit_q16,
              &network_credit_plan)) {
        ctx.history_refused = 1u;
        *out_ctx = ctx;
        return;
      }
      const std::uint32_t network_credit_count =
          network_credit_plan.valid != 0u ? 1u : 0u;
      direct_network::DirectExactHistoryRecord network_credit_record{};
      if (network_credit_count != 0u)
        direct_network::stage_network_credit_history_record(
            &network_credit_record, target_ticket_id,
            network_credit_plan.active_network_identity,
            network_credit_plan.recruitment_identity, admission_tick,
            ticket.emission_tick, network_credit_plan.prior_credit_q16,
            network_credit_plan.next_credit_q16,
            network_credit_plan.applied_delta_q16);
      direct_network::DirectExactHistoryRecord recipe_records[2u * kMaxActionParticipationLinks]{};
      bool recipe_transaction_valid = false;
      const std::uint32_t recipe_count = plan_recipe_credit_transaction(
          brain, actual_frontier, consequence_record,
          recipe_cells, recipe_cell_count, routes, selected_links,
          action->participant_offset, sparse_links, applied_deltas,
          sparse_count, target_ticket_id, admission_tick,
          ticket.emission_tick, recipe_records, &recipe_transaction_valid);
      if (!recipe_transaction_valid) {
        ctx.history_refused = 1u;
        *out_ctx = ctx;
        return;
      }
      direct_network::DirectExactHistoryRecord inverse_record{};
      const std::uint32_t inverse_count = plan_resident_inverse_transformation(brain, actual_frontier, selected_links, action->participant_offset, action->participant_count, target_ticket_id, returned_word, admission_tick, *exact_history, recipe_records, recipe_count, &inverse_record);
      ResidentWorldReturnRecipeRebindPlan recipe_rebind{};
      if (recipe_count != 0u && !stage_world_return_recipe_rebind(
              brain, actual_frontier, consequence_record, recipe_records,
              recipe_count,
              &recipe_rebind)) {
        ctx.history_refused = 1u;
        *out_ctx = ctx;
        return;
      }
      const std::uint32_t shatter_count = ctx.effective_reward_q16 < shatter_threshold_q16
          ? record_dense_shatter_transaction(dense_blocks, dense_block_count, selected_links, action->participant_offset, sparse_links, sparse_count, nullptr, target_ticket_id, ticket.upstream_ticket_id, admission_tick, ticket.emission_tick) : 0u;
      const std::uint32_t history_width = 1u + sparse_count + shatter_count +
          network_credit_count + recipe_count + inverse_count;
      if (network_credit_count != 0u &&
          !resident_recruited_network_credit_plan_current(
              brain.development, network_credit_plan)) {
        ctx.history_refused = 1u;
        *out_ctx = ctx;
        return;
      }
      if (exact_history->sealed || exact_history->phase_kind != direct_network::DirectExactHistoryKind::empty || history_width >
          direct_network::kDirectExactHistoryHotPageCapacity - exact_history->committed_slots)
        { ctx.history_refused = 1u; *out_ctx = ctx; return; }
      if (!direct_network::begin_exact_history_phase(exact_history,
              direct_network::DirectExactHistoryKind::world_return, history_width, admission_tick))
        { ctx.history_refused = 1u; *out_ctx = ctx; return; }
      exact_history->records[exact_history->phase_base] = consequence_record;
      const std::uint32_t history_base = exact_history->phase_base;
      for (std::uint32_t i = 0u; i < sparse_count; ++i) {
        const DirectActionParticipationLink& link = selected_links[
            action->participant_offset + sparse_links[i]];
        direct_network::stage_sparse_credit_history_record(
            &exact_history->records[history_base + 1u + i], target_ticket_id,
            link.participant_ticket_id, admission_tick, ticket.emission_tick, link.source_node,
            link.route_index, link.target_node, link.context_signature,
            priors[i], link.route_incarnation, link.claim_incarnation, applied_deltas[i]);
      }
      if (shatter_count != 0u)
        record_dense_shatter_transaction(dense_blocks, dense_block_count, selected_links, action->participant_offset, sparse_links, sparse_count, &exact_history->records[history_base + 1u + sparse_count], target_ticket_id, ticket.upstream_ticket_id, admission_tick, ticket.emission_tick);
      const std::uint32_t network_credit_offset =
          history_base + 1u + sparse_count + shatter_count;
      if (network_credit_count != 0u)
        exact_history->records[network_credit_offset] = network_credit_record;
      const std::uint32_t recipe_offset =
          network_credit_offset + network_credit_count;
      for (std::uint32_t i = 0u; i < recipe_count; ++i)
        exact_history->records[recipe_offset + i] = recipe_records[i];
      if (inverse_count != 0u)
        exact_history->records[recipe_offset + recipe_count] = inverse_record;
      if (direct_network::finish_exact_history_phase(exact_history) != history_width)
        { ctx.history_refused = 1u; *out_ctx = ctx; return; }
      for (std::uint32_t i = 0u; i < sparse_count; ++i) {
        const DirectActionParticipationLink& link =
            selected_links[action->participant_offset + sparse_links[i]];
        bind_raw_contact_recipe_incidence(
            brain, link,
            exact_history->records[history_base + 1u + i]);
        apply_recorded_sparse_action_link(
            link, ctx,
            nodes, routes, metrics, applied_deltas[i]);
        if (retention_bank != nullptr) {
          const std::uint32_t magnitude = static_cast<std::uint32_t>(
              applied_deltas[i] < 0 ? -applied_deltas[i] : applied_deltas[i]);
          substrate::direct_adult::device_record_causal_difference(
              retention_bank[link.route_index], link.source_node, link.route_index,
              link.route_incarnation, mismatch, magnitude, admission_tick,
              routes[link.route_index].conductance_q16);
        }
      }
      if (shatter_count != 0u)
        apply_dense_shatter_transaction(dense_blocks, dense_block_count, selected_links, action->participant_offset,
                                        sparse_links, sparse_count, metrics);
      if (recipe_count != 0u)
        publish_world_return_recipe_rebind(
            brain, actual_frontier, recipe_rebind);
      if (network_credit_count != 0u)
        commit_resident_recruited_network_credit(
            brain.development, network_credit_plan, admission_tick);
      if (inverse_count != 0u) apply_resident_inverse_transformation_and_bindings(inverse_record, brain, actual_frontier);
      __threadfence(); atomicExch(&ticket.settled, 1u);
      ticket.mismatch_bits = mismatch; ticket.settled_reward_q16 = ctx.effective_reward_q16;
      ctx.valid = 1u;
      if (action_exact) {
        ctx.action_bound = 1u; ctx.ancestry_incomplete = ancestry_incomplete ? 1u : 0u;
        ctx.action_participant_offset = ancestry_incomplete ? 0u : action->participant_offset;
        ctx.action_participant_count = action->participant_count;
        __threadfence(); atomicExch(&action->state, kActionOccurrenceSettled);
        settle_action_actual_occurrences(
            actual_frontier, selected_links, action->participant_offset,
            action->participant_count);
      }
      if (delayed_record != nullptr) {
        __threadfence();
        atomicExch(&delayed_record->state, 2u);
        atomicAdd(reinterpret_cast<unsigned long long*>(
                      &brain.development->delayed_actions_settled), 1ULL);
      }
      if (metrics) { atomicAdd(reinterpret_cast<unsigned long long*>(&metrics->consequences_assimilated), 1ULL); atomicAdd(reinterpret_cast<unsigned long long*>(&metrics->predictive_shadows_evaluated), 1ULL); }
    }
  }
  *out_ctx = ctx;
}

DIRECT_ADULT_DEVICE_OPS_HEAVY_QUALIFIER void device_process_single_sensory_event(
    DirectNode* nodes,
    const DirectBoundaryPort* ports,
    std::uint32_t port_count,
    std::uint32_t node_count,
    const ActivityEvent& event,
    std::int32_t* node_incoming_excitation,
    std::int32_t* node_slow_context_q16,
    NodeCausalParticipation* node_active_participation,
    std::uint32_t* node_active_participation_locks,
    std::uint32_t* claim_incarnation_counter,
    std::uint32_t* ticket_table_locks,
    AsynchronousTicket* ticket_table,
    DirectActionOccurrence* action_occurrences,
    DirectParticipationDescriptor* current_contributions,
    std::uint32_t* current_contribution_count,
    std::uint32_t current_contribution_capacity,
    AdultCoreMetrics* metrics,
    std::uint32_t current_tick,
    std::uint32_t horizon_ticks,
    direct_network::DirectExactHistoryRecord* history_record,
    bool membrane_authenticated,
    std::uint32_t membrane_authority_incarnation) {
  if (!membrane_authenticated && event.origin != CausalOrigin::external_contact) { if (metrics != nullptr) atomicAdd(reinterpret_cast<unsigned long long*>(&metrics->sensory_boundary_rejects), 1ULL); return; }
  std::uint32_t target_node = event.node, sensor_matches = 0u, mapped_node = kInvalidIndex;
  bool rejected = false;
  if (ports != nullptr) {
    for (std::uint32_t p = 0; p < port_count; ++p) {
      if (ports[p].channel == event.channel &&
          (ports[p].role_mask & static_cast<std::uint32_t>(direct_network::BoundaryRole::sensor)))
        { ++sensor_matches;
          mapped_node = ports[p].node; }
    }
    rejected = sensor_matches != 1u || mapped_node >= node_count ||
               (target_node != kInvalidIndex && (target_node >= node_count || target_node != mapped_node));
    target_node = mapped_node;
  } else rejected = target_node == kInvalidIndex || target_node >= node_count;
  if (rejected) { if (metrics != nullptr) atomicAdd(reinterpret_cast<unsigned long long*>(&metrics->sensory_boundary_rejects), 1ULL); return; }
  if (target_node != kInvalidIndex && target_node < node_count) {
    direct_network::stage_sensory_history_record(
        history_record, event.ticket_id, current_tick, event.timestamp, target_node, event.channel,
        event.word, event.context, static_cast<std::uint32_t>(event.origin),
        membrane_authenticated);
    if (node_incoming_excitation != nullptr) {
      atomicAdd(&node_incoming_excitation[target_node], kQ16One);
    }
    if (nodes != nullptr) {
      nodes[target_node].last_actual_tick = current_tick;
      nodes[target_node].activation_q16 = kQ16One;
    }
    const std::uint32_t expiry_tick = current_tick + horizon_ticks;
    const std::uint32_t physical_signature =
        raw_contact_signature(target_node, event.channel, event.word);
    bool contribution_complete = false;
    if (node_active_participation != nullptr) {
      bool authority_collision = false;
      NodeCausalParticipation participant{};
      std::uint32_t participant_slot = kInvalidIndex;
      if (insert_active_participation(node_active_participation,
                                      node_active_participation_locks,
                                      claim_incarnation_counter,
                                      target_node, event.ticket_id,
                                      expiry_tick, current_tick,
                                      metrics ? &metrics->provenance_no_evictable_slot_drops : nullptr,
                                      membrane_authenticated
                                          ? DirectParticipationAuthority::independent_external
                                          : DirectParticipationAuthority::none,
                                      membrane_authenticated
                                          ? membrane_authority_incarnation : 0u,
                                      metrics ? &metrics->authorized_participation_refreshes : nullptr,
                                      metrics ? &metrics->participation_authority_collision_rejects : nullptr,
                                      &authority_collision, 0u, &participant,
                                      &participant_slot)) {
        DirectParticipationDescriptor descriptor{};
        descriptor.ticket_id = participant.ticket_id;
        descriptor.source_node = kInvalidIndex;
        descriptor.target_node = target_node;
        descriptor.route_index = kInvalidIndex;
        descriptor.context_signature = physical_signature;
        descriptor.expiry_tick = participant.expiry_tick;
        descriptor.claim_incarnation = participant.claim_incarnation;
        descriptor.authority = participant.authority;
        descriptor.authority_incarnation = participant.authority_incarnation;
        descriptor.contribution_kind = DirectContributionKind::direct_ingress;
        stage_current_contribution(
            descriptor, current_contributions, current_contribution_count,
            current_contribution_capacity, metrics);
        contribution_complete = participant_slot != kInvalidIndex;
        if (metrics != nullptr) {
          atomicAdd(reinterpret_cast<unsigned long long*>(&metrics->sensory_participation_admitted), 1ULL);
        }
      } else if (metrics != nullptr && !authority_collision) {
        atomicAdd(reinterpret_cast<unsigned long long*>(&metrics->provenance_capacity_drops), 1ULL);
      }
    }

    if (!contribution_complete) {
      stage_incomplete_contribution(
          kInvalidIndex, target_node, kInvalidIndex, physical_signature,
          current_tick, current_contributions, current_contribution_count,
          current_contribution_capacity, metrics);
    }

    bool ticket_identity_complete = true;
    if (ticket_table != nullptr && event.ticket_id != 0ULL && event.ticket_id != kInvalidTicket) {
      const std::uint32_t t_slot = static_cast<std::uint32_t>(event.ticket_id & 0x7ffu);
      const bool locked = ticket_table_locks != nullptr &&
          lock_word(ticket_table_locks + t_slot);
      if (!locked) {
        ticket_identity_complete = false;
        if (metrics != nullptr)
          atomicAdd(reinterpret_cast<unsigned long long*>(
                        &metrics->action_ticket_collision_rejects), 1ULL);
      } else {
        __threadfence();
        const AsynchronousTicket existing = ticket_table[t_slot];
        const bool action_pending = action_occurrences != nullptr &&
            atomicAdd(&action_occurrences[t_slot].state, 0u) == kActionOccurrencePending;
        const bool collision = action_pending ||
            (existing.ticket_id != 0u && existing.ticket_id != event.ticket_id &&
             existing.settled == 0u &&
             (current_tick < existing.emission_tick ||
              current_tick - existing.emission_tick <= horizon_ticks));
        if (collision) {
          ticket_identity_complete = false;
          if (metrics != nullptr)
            atomicAdd(reinterpret_cast<unsigned long long*>(
                          &metrics->action_ticket_collision_rejects), 1ULL);
        } else {
          AsynchronousTicket st{};
          st.ticket_id = st.upstream_ticket_id = event.ticket_id;
          st.motor_node = target_node;
          st.motor_channel = event.channel;
          st.motor_word = event.word;
          st.context_signature = physical_signature;
          st.emission_tick = current_tick;
          ticket_table[t_slot] = st;
        }
        __threadfence();
        atomicExch(ticket_table_locks + t_slot, 0u);
      }
    }
    if (!ticket_identity_complete) {
      stage_incomplete_contribution(
          kInvalidIndex, target_node, kInvalidIndex, physical_signature,
          current_tick, current_contributions, current_contribution_count,
          current_contribution_capacity, metrics);
    }

    if (node_slow_context_q16 != nullptr) {
      const std::int32_t context_delta =
          static_cast<std::int32_t>((physical_signature & 0xffffu) | 1u);
      atomicAdd(&node_slow_context_q16[target_node], context_delta / 16);
    }

    if (metrics != nullptr) {
      atomicAdd(reinterpret_cast<unsigned long long*>(&metrics->sensory_events_ingested), 1ULL);
    }
  }
}
