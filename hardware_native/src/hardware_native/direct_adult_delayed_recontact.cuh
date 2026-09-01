#ifndef HARDWARE_NATIVE_DIRECT_ADULT_DELAYED_RECONTACT_CUH
#define HARDWARE_NATIVE_DIRECT_ADULT_DELAYED_RECONTACT_CUH

#include "direct_adult_core.cuh"

namespace substrate::direct_adult_core {

#ifndef DIRECT_ADULT_DEVICE_OPS_EXPORTED_QUALIFIER
#define DIRECT_ADULT_DEVICE_OPS_EXPORTED_QUALIFIER __device__ inline
#define DIRECT_ADULT_DELAYED_RECONTACT_LOCAL_EXPORT_QUALIFIER 1
#endif

// Bounded slot reuse may displace an unresolved action before its physical
// consequence returns. These operations preserve and reconstruct that exact
// resident lineage; they never search history or infer a replacement action.
__device__ inline void copy_delayed_action_participant(
    direct_network::ResidentDevelopmentState::DelayedActionParticipant* out,
    const DirectActionParticipationLink& in) {
  out->participant_ticket_id = in.participant_ticket_id;
  out->logical_recipe_id = in.logical_recipe_id;
  out->revision_identity = in.revision_identity;
  out->occurrence_identity = in.occurrence_identity;
  out->participation_identity = in.participation_identity;
  out->occurrence_route_incarnation = in.occurrence_route_incarnation;
  out->source_node = in.source_node;
  out->target_node = in.target_node;
  out->route_index = in.route_index;
  out->context_signature = in.context_signature;
  out->occurrence_context_signature = in.occurrence_context_signature;
  out->composition_depth = in.composition_depth;
  out->expiry_tick = in.expiry_tick;
  out->claim_incarnation = in.claim_incarnation;
  out->route_incarnation = in.route_incarnation;
  out->authority_incarnation = in.authority_incarnation;
  out->authority = static_cast<std::uint32_t>(in.authority);
  out->contribution_kind = static_cast<std::uint32_t>(in.contribution_kind);
  out->frozen_eligibility_q16 = in.frozen_eligibility_q16;
  out->eligibility_slot = in.eligibility_slot;
  out->eligibility_generation = in.eligibility_generation;
}

__device__ inline void restore_delayed_action_participant(
    DirectActionParticipationLink* out,
    const direct_network::ResidentDevelopmentState::DelayedActionParticipant& in) {
  out->participant_ticket_id = in.participant_ticket_id;
  out->logical_recipe_id = in.logical_recipe_id;
  out->revision_identity = in.revision_identity;
  out->occurrence_identity = in.occurrence_identity;
  out->participation_identity = in.participation_identity;
  out->occurrence_route_incarnation = in.occurrence_route_incarnation;
  out->source_node = in.source_node;
  out->target_node = in.target_node;
  out->route_index = in.route_index;
  out->context_signature = in.context_signature;
  out->occurrence_context_signature = in.occurrence_context_signature;
  out->composition_depth = in.composition_depth;
  out->expiry_tick = in.expiry_tick;
  out->claim_incarnation = in.claim_incarnation;
  out->route_incarnation = in.route_incarnation;
  out->authority_incarnation = in.authority_incarnation;
  out->authority = static_cast<DirectParticipationAuthority>(in.authority);
  out->contribution_kind = static_cast<DirectContributionKind>(in.contribution_kind);
  out->frozen_eligibility_q16 = in.frozen_eligibility_q16;
  out->eligibility_slot = in.eligibility_slot;
  out->eligibility_generation = in.eligibility_generation;
}

DIRECT_ADULT_DEVICE_OPS_EXPORTED_QUALIFIER bool preserve_displaced_action(
    direct_network::ResidentDevelopmentState* development,
    const AsynchronousTicket& ticket, const DirectActionOccurrence& action,
    const DirectActionParticipationLink* action_links,
    std::uint32_t participant_capacity) {
  if (development == nullptr || ticket.ticket_id == 0u ||
      ticket.ticket_id == kInvalidTicket || ticket.settled != 0u ||
      action.action_ticket_id != ticket.ticket_id ||
      action.participant_count > kMaxActionParticipationLinks ||
      action.participant_count > participant_capacity ||
      (action.participant_count != 0u &&
       (action_links == nullptr ||
        action.participant_offset > participant_capacity - action.participant_count)))
    return false;
  auto* destination =
      static_cast<direct_network::ResidentDevelopmentState::DelayedActionRecord*>(nullptr);
  std::uint32_t oldest =
      direct_network::ResidentDevelopmentState::kDelayedActionCapacity;
  for (std::uint32_t i = 0u;
       i < direct_network::ResidentDevelopmentState::kDelayedActionCapacity; ++i) {
    auto& candidate = development->delayed_actions[i];
    if (atomicCAS(&candidate.state, 0u, 3u) == 0u ||
        atomicCAS(&candidate.state, 2u, 3u) == 2u) {
      destination = &candidate;
      break;
    }
    if (atomicAdd(&candidate.state, 0u) != 1u) continue;
    if (oldest == direct_network::ResidentDevelopmentState::kDelayedActionCapacity) {
      oldest = i;
      continue;
    }
    const auto& incumbent = development->delayed_actions[oldest];
    if (candidate.expiry_tick < incumbent.expiry_tick ||
        (candidate.expiry_tick == incumbent.expiry_tick &&
         (candidate.action_emission_tick < incumbent.action_emission_tick ||
          (candidate.action_emission_tick == incumbent.action_emission_tick &&
           candidate.ticket_id < incumbent.ticket_id))))
      oldest = i;
  }
  // Finite exact delayed-return state must not permanently stop future action.
  // Once every slot is occupied, retain the causally newer expired action and
  // retire only the oldest resident record. A later return for the retired
  // ticket is then explicitly unknown; no semantic or arbitrary tie decides.
  if (destination == nullptr &&
      oldest < direct_network::ResidentDevelopmentState::kDelayedActionCapacity) {
    auto& incumbent = development->delayed_actions[oldest];
    const bool arriving_is_newer =
        action.expiry_tick > incumbent.expiry_tick ||
        (action.expiry_tick == incumbent.expiry_tick &&
         (action.emission_tick > incumbent.action_emission_tick ||
          (action.emission_tick == incumbent.action_emission_tick &&
           action.action_ticket_id > incumbent.ticket_id)));
    if (arriving_is_newer && atomicCAS(&incumbent.state, 1u, 3u) == 1u)
      destination = &incumbent;
  }
  if (destination == nullptr) {
    atomicAdd(reinterpret_cast<unsigned long long*>(
                  &development->delayed_action_backpressure), 1ULL);
    return false;
  }
  direct_network::ResidentDevelopmentState::DelayedActionRecord snapshot{};
  snapshot.state = 3u;
  snapshot.ticket_id = ticket.ticket_id;
  snapshot.upstream_ticket_id = ticket.upstream_ticket_id;
  snapshot.motor_node = ticket.motor_node;
  snapshot.motor_channel = ticket.motor_channel;
  snapshot.motor_word = ticket.motor_word;
  snapshot.context_signature = ticket.context_signature;
  snapshot.emission_tick = ticket.emission_tick;
  snapshot.action_ticket_id = action.action_ticket_id;
  snapshot.action_network_identity = action.network_identity;
  snapshot.action_recruitment_identity = action.recruitment_identity;
  snapshot.action_network_eligibility_signed_q16 =
      action.network_eligibility_signed_q16;
  snapshot.action_network_eligibility_l1_q16 =
      action.network_eligibility_l1_q16;
  snapshot.participant_offset = action.participant_offset;
  snapshot.participant_count = action.participant_count;
  snapshot.action_emission_tick = action.emission_tick;
  snapshot.action_context_signature = action.context_signature;
  snapshot.action_motor_node = action.motor_node;
  snapshot.action_motor_channel = action.motor_channel;
  snapshot.expiry_tick = action.expiry_tick;
  snapshot.occurrence_identity_required = action.occurrence_identity_required;
  snapshot.occurrence_identity_complete = action.occurrence_identity_complete;
  for (std::uint32_t i = 0u; i < action.participant_count; ++i)
    copy_delayed_action_participant(
        &snapshot.participants[i], action_links[action.participant_offset + i]);
  *destination = snapshot;
  __threadfence();
  atomicExch(&destination->state, 1u);
  atomicAdd(reinterpret_cast<unsigned long long*>(
                &development->delayed_actions_preserved), 1ULL);
  return true;
}

__device__ inline direct_network::ResidentDevelopmentState::DelayedActionRecord*
load_preserved_action(
    direct_network::ResidentDevelopmentState* development,
    std::uint64_t target_ticket_id, AsynchronousTicket* ticket,
    DirectActionOccurrence* action, DirectActionParticipationLink* links) {
  if (development == nullptr || ticket == nullptr || action == nullptr ||
      links == nullptr)
    return nullptr;
  for (std::uint32_t i = 0u;
       i < direct_network::ResidentDevelopmentState::kDelayedActionCapacity; ++i) {
    auto& candidate = development->delayed_actions[i];
    if (atomicAdd(&candidate.state, 0u) != 1u ||
        candidate.ticket_id != target_ticket_id ||
        candidate.participant_count > kMaxActionParticipationLinks)
      continue;
    ticket->ticket_id = candidate.ticket_id;
    ticket->upstream_ticket_id = candidate.upstream_ticket_id;
    ticket->motor_node = candidate.motor_node;
    ticket->motor_channel = candidate.motor_channel;
    ticket->motor_word = candidate.motor_word;
    ticket->context_signature = candidate.context_signature;
    ticket->emission_tick = candidate.emission_tick;
    action->action_ticket_id = candidate.action_ticket_id;
    action->network_identity = candidate.action_network_identity;
    action->recruitment_identity = candidate.action_recruitment_identity;
    action->network_eligibility_signed_q16 =
        candidate.action_network_eligibility_signed_q16;
    action->network_eligibility_l1_q16 =
        candidate.action_network_eligibility_l1_q16;
    // `links` is the caller-owned compact replay buffer, not the original
    // ticket-indexed action bank. Nonempty closures rematerialize at zero, but
    // the invalid offset on a zero-participant action is the resident
    // ancestry-incomplete state and must survive delayed displacement.
    action->participant_offset =
        candidate.participant_count == 0u &&
                candidate.participant_offset == kInvalidIndex
            ? kInvalidIndex
            : 0u;
    action->participant_count = candidate.participant_count;
    action->emission_tick = candidate.action_emission_tick;
    action->context_signature = candidate.action_context_signature;
    action->motor_node = candidate.action_motor_node;
    action->motor_channel = candidate.action_motor_channel;
    action->state = kActionOccurrencePending;
    action->expiry_tick = candidate.expiry_tick;
    action->occurrence_identity_required = candidate.occurrence_identity_required;
    action->occurrence_identity_complete = candidate.occurrence_identity_complete;
    for (std::uint32_t j = 0u; j < candidate.participant_count; ++j)
      restore_delayed_action_participant(&links[j], candidate.participants[j]);
    return &candidate;
  }
  return nullptr;
}

#ifdef DIRECT_ADULT_DELAYED_RECONTACT_LOCAL_EXPORT_QUALIFIER
#undef DIRECT_ADULT_DELAYED_RECONTACT_LOCAL_EXPORT_QUALIFIER
#undef DIRECT_ADULT_DEVICE_OPS_EXPORTED_QUALIFIER
#endif

}  // namespace substrate::direct_adult_core

#endif
