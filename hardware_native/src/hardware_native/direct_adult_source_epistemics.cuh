#ifndef HARDWARE_NATIVE_DIRECT_ADULT_SOURCE_EPISTEMICS_CUH
#define HARDWARE_NATIVE_DIRECT_ADULT_SOURCE_EPISTEMICS_CUH

#include <cstdint>
#include <type_traits>

#include "hardware_native/direct_adult_causal_world_model.cuh"
#include "hardware_native/direct_adult_contact_receipt.cuh"
#include "hardware_native/direct_adult_runtime_frontiers.cuh"
#include "hardware_native/direct_exact_history.cuh"
#if defined(__CUDACC__)
#define DIRECT_SOURCE_EPISTEMICS_HD __host__ __device__
#else
#define DIRECT_SOURCE_EPISTEMICS_HD
#endif

namespace substrate::direct_adult_core {

enum class DirectSourceAssertionIntegrity : std::uint32_t {
  none = 0u,
  withdrawn = 1u,
  corrupted = 2u,
  forged = 3u,
  refused = 4u,
};

struct alignas(8) DirectSourceAssertionClaim {
  std::uint64_t source_identity;
  std::uint64_t surface_identity;
  std::uint64_t context_identity;
  std::uint64_t contact_identity;
  std::uint64_t ticket;
  std::uint64_t history_sequence;
  std::uint64_t boundary_session_epoch;
  std::uint64_t ingress_sequence;
  std::uint32_t contact_authenticated;
  std::uint32_t reserved;
};

static_assert(std::is_standard_layout_v<DirectSourceAssertionClaim> &&
              std::is_trivial_v<DirectSourceAssertionClaim> &&
              std::has_unique_object_representations_v<DirectSourceAssertionClaim>);

DIRECT_SOURCE_EPISTEMICS_HD inline std::uint64_t source_assertion_receipt_identity(
    const ResidentContactEpochReceipt& receipt) {
  using direct_network::exact_history_fold_word;
  std::uint64_t identity =
      exact_history_fold_word(0x636f6e7461637472ull, receipt.source_identity);
  identity = exact_history_fold_word(identity, receipt.codec_identity);
  identity = exact_history_fold_word(identity, receipt.payload_identity);
  identity = exact_history_fold_word(identity, receipt.boundary_session_epoch);
  identity = exact_history_fold_word(identity, receipt.ingress_sequence);
  identity = exact_history_fold_word(identity, static_cast<std::uint32_t>(receipt.selection));
  identity = exact_history_fold_word(identity, static_cast<std::uint32_t>(receipt.integration));
  identity = exact_history_fold_word(identity, receipt.source_available);
  identity = exact_history_fold_word(identity, receipt.port_index);
  return identity == 0u ? 1u : identity;
}

DIRECT_SOURCE_EPISTEMICS_HD inline std::uint64_t source_assertion_context_identity(
    std::uint64_t session, std::uint64_t ingress) {
  using direct_network::exact_history_fold_word;
  const std::uint64_t identity =
      exact_history_fold_word(exact_history_fold_word(0x6173737274437874ull, session),
                              ingress);
  return identity == 0u ? 1u : identity;
}

DIRECT_SOURCE_EPISTEMICS_HD inline void stage_source_assertion_history_record(
    direct_network::DirectExactHistoryRecord* record, std::uint64_t contact_identity,
    std::uint64_t ticket, std::uint32_t resident_tick, std::uint32_t event_tick,
    std::uint64_t source_identity, std::uint64_t surface_identity,
    std::uint64_t session, std::uint64_t ingress, bool contact_authenticated) {
  if (record == nullptr) return;
  record->identity = contact_identity;
  record->parent_identity = ticket;
  record->resident_tick = resident_tick;
  record->event_tick = event_tick;
  record->kind = direct_network::DirectExactHistoryKind::source_assertion;
  record->source = static_cast<std::uint32_t>(session);
  record->subject = static_cast<std::uint32_t>(session >> 32);
  record->value = static_cast<std::uint32_t>(ingress);
  record->context = static_cast<std::uint32_t>(ingress >> 32);
  record->flags = contact_authenticated
                      ? direct_network::kDirectHistoryVerifiedObservation
                      : 0u;
  record->incarnation_before = source_identity;
  record->incarnation_after = surface_identity;
  record->resource_delta = 0;
}

DIRECT_SOURCE_EPISTEMICS_HD inline bool capture_source_assertion_from_receipt(
    const ResidentContactEpochReceipt& receipt, std::uint64_t ticket,
    DirectSourceAssertionClaim* claim) {
  if (claim == nullptr || ticket == 0u || receipt.identity == 0u ||
      receipt.source_identity == 0u || receipt.payload_identity == 0u ||
      receipt.boundary_session_epoch == 0u || receipt.ingress_sequence == 0u ||
      receipt.selection != ResidentContactSelection::resident_owned ||
      receipt.source_available == 0u ||
      receipt.identity != source_assertion_receipt_identity(receipt))
    return false;
  *claim = DirectSourceAssertionClaim{};
  claim->source_identity = receipt.source_identity;
  claim->surface_identity = receipt.payload_identity;
  claim->context_identity = source_assertion_context_identity(
      receipt.boundary_session_epoch, receipt.ingress_sequence);
  claim->contact_identity = receipt.identity;
  claim->ticket = ticket;
  claim->boundary_session_epoch = receipt.boundary_session_epoch;
  claim->ingress_sequence = receipt.ingress_sequence;
  claim->contact_authenticated = 1u;
  return true;
}

DIRECT_SOURCE_EPISTEMICS_HD inline bool read_source_assertion_claim(
    const direct_network::DirectExactHistoryRecord& record,
    DirectSourceAssertionClaim* claim) {
  if (claim == nullptr ||
      record.kind != direct_network::DirectExactHistoryKind::source_assertion ||
      record.identity == 0u || record.parent_identity == 0u ||
      record.resource_delta != 0)
    return false;
  const std::uint64_t session =
      static_cast<std::uint64_t>(record.source) |
      (static_cast<std::uint64_t>(record.subject) << 32);
  const std::uint64_t ingress =
      static_cast<std::uint64_t>(record.value) |
      (static_cast<std::uint64_t>(record.context) << 32);
  *claim = DirectSourceAssertionClaim{};
  claim->source_identity = record.incarnation_before;
  claim->surface_identity = record.incarnation_after;
  claim->context_identity = source_assertion_context_identity(session, ingress);
  claim->contact_identity = record.identity;
  claim->ticket = record.parent_identity;
  claim->history_sequence = record.sequence;
  claim->boundary_session_epoch = session;
  claim->ingress_sequence = ingress;
  claim->contact_authenticated =
      (record.flags & direct_network::kDirectHistoryVerifiedObservation) != 0u ? 1u : 0u;
  return true;
}

DIRECT_SOURCE_EPISTEMICS_HD inline bool find_committed_source_assertion(
    const direct_network::DirectExactHistoryHotPage& history,
    std::uint64_t contact_identity,
    direct_network::DirectExactHistoryRecord* record) {
  if (contact_identity == 0u) return false;
  for (std::uint32_t i = 0u; i < history.committed_slots; ++i) {
    const auto& candidate = history.records[i];
    if (candidate.kind != direct_network::DirectExactHistoryKind::source_assertion ||
        candidate.identity != contact_identity)
      continue;
    if (record != nullptr) *record = candidate;
    return true;
  }
  return false;
}

DIRECT_SOURCE_EPISTEMICS_HD inline bool history_has_sensory_ticket(
    const direct_network::DirectExactHistoryHotPage& history, std::uint64_t ticket) {
  if (ticket == 0u) return false;
  for (std::uint32_t i = 0u; i < history.committed_slots; ++i) {
    const auto& record = history.records[i];
    if (record.kind == direct_network::DirectExactHistoryKind::sensory_contact &&
        record.identity == ticket)
      return true;
  }
  return false;
}

DIRECT_SOURCE_EPISTEMICS_HD inline bool commit_source_assertion_to_history(
    direct_network::DirectExactHistoryHotPage* history,
    const ResidentContactEpochReceipt& receipt, std::uint64_t ticket,
    std::uint32_t tick, DirectSourceAssertionClaim* claim) {
  DirectSourceAssertionClaim captured{};
  if (history == nullptr ||
      !capture_source_assertion_from_receipt(receipt, ticket, &captured) ||
      !history_has_sensory_ticket(*history, ticket) ||
      find_committed_source_assertion(*history, captured.contact_identity, nullptr))
    return false;
  if (!direct_network::begin_exact_history_phase(
          history, direct_network::DirectExactHistoryKind::source_assertion, 1u,
          tick))
    return false;
  stage_source_assertion_history_record(
      &history->records[history->phase_base], captured.contact_identity, ticket,
      tick, tick, captured.source_identity, captured.surface_identity,
      captured.boundary_session_epoch, captured.ingress_sequence, true);
  if (direct_network::finish_exact_history_phase(history) != 1u) return false;
  direct_network::DirectExactHistoryRecord committed{};
  if (!find_committed_source_assertion(*history, captured.contact_identity, &committed) ||
      !read_source_assertion_claim(committed, claim))
    return false;
  return claim->source_identity == captured.source_identity &&
      claim->surface_identity == captured.surface_identity &&
      claim->context_identity == captured.context_identity &&
      claim->contact_identity == captured.contact_identity &&
      claim->ticket == ticket;
}

DIRECT_SOURCE_EPISTEMICS_HD inline bool sensory_history_authorizes_source_assertion(
    const DirectBrain& brain, const DirectBoundaryPort& port,
    std::uint32_t port_index,
    const direct_network::DirectExactHistoryRecord& sensory,
    const ResidentContactEpochReceipt& receipt, std::uint64_t ticket) {
  using namespace direct_network;
  if (ticket == 0u || sensory.kind != DirectExactHistoryKind::sensory_contact ||
      sensory.identity != ticket || receipt.identity == 0u ||
      receipt.consumed != 0u || receipt.reserved != 0u ||
      receipt.port_index != port_index ||
      receipt.selection != ResidentContactSelection::resident_owned ||
      receipt.integration != ResidentContactIntegration::canonical ||
      receipt.source_available != 1u ||
      (sensory.flags & kDirectHistoryVerifiedObservation) == 0u ||
      (sensory.flags & kDirectHistoryPayloadFlags) !=
          static_cast<std::uint32_t>(CausalOrigin::external_contact) ||
      port.node != sensory.source || port.channel != sensory.subject ||
      port.physical_route != sensory.context ||
      (port.role_mask & static_cast<std::uint32_t>(BoundaryRole::sensor)) == 0u)
    return false;
  ActivityEvent event{};
  event.ticket_id = ticket;
  event.node = sensory.source;
  event.channel = sensory.subject;
  event.word = sensory.value;
  event.origin = CausalOrigin::external_contact;
  event.context = sensory.context;
  event.timestamp = sensory.event_tick;
  return receipt.identity == source_assertion_receipt_identity(receipt) &&
      receipt.payload_identity == resident_contact_payload_identity(event) &&
      receipt.source_identity == resident_contact_source_identity(brain, port) &&
      receipt.codec_identity == resident_contact_codec_identity(brain, port) &&
      receipt.boundary_session_epoch == resident_contact_session_identity(brain, port);
}

// The sensory authority may be hot or cold. This records provenance only;
// neither the assertion nor history can create experiential credit.
DIRECT_SOURCE_EPISTEMICS_HD inline bool commit_source_assertion_from_sensory_authority(
    direct_network::DirectExactHistoryHotPage* history, const DirectBrain& brain,
    const DirectBoundaryPort& port, std::uint32_t port_index,
    const direct_network::DirectExactHistoryRecord& sensory,
    const ResidentContactEpochReceipt& receipt, std::uint64_t ticket,
    std::uint32_t tick, DirectSourceAssertionClaim* claim) {
  DirectSourceAssertionClaim captured{};
  if (history == nullptr ||
      !capture_source_assertion_from_receipt(receipt, ticket, &captured) ||
      !sensory_history_authorizes_source_assertion(
          brain, port, port_index, sensory, receipt, ticket) ||
      find_committed_source_assertion(*history, captured.contact_identity, nullptr))
    return false;
  if (!direct_network::begin_exact_history_phase(
          history, direct_network::DirectExactHistoryKind::source_assertion, 1u,
          tick))
    return false;
  stage_source_assertion_history_record(
      &history->records[history->phase_base], captured.contact_identity, ticket,
      tick, sensory.event_tick, captured.source_identity, captured.surface_identity,
      captured.boundary_session_epoch, captured.ingress_sequence, true);
  if (direct_network::finish_exact_history_phase(history) != 1u) return false;
  direct_network::DirectExactHistoryRecord committed{};
  if (!find_committed_source_assertion(*history, captured.contact_identity, &committed) ||
      !read_source_assertion_claim(committed, claim))
    return false;
  return claim->source_identity == captured.source_identity &&
      claim->surface_identity == captured.surface_identity &&
      claim->context_identity == captured.context_identity &&
      claim->contact_identity == captured.contact_identity &&
      claim->ticket == ticket;
}

struct alignas(8) DirectSourceWithdrawnRelationReceipt {
  std::uint64_t source_identity;
  std::uint64_t context_identity;
  std::uint64_t assertion_contact_identity;
  std::uint64_t assertion_history_sequence;
  std::uint64_t withdrawal_history_sequence;
  std::uint64_t action_ticket_id;
  std::uint64_t causal_model_identity;
  std::uint32_t action_value;
  std::uint32_t root_channel;
  std::uint32_t current_outcome_value;
  std::uint32_t current_outcome_tick;
  std::uint32_t observations;
  std::uint32_t relation_index;
};
static_assert(std::is_standard_layout_v<DirectSourceWithdrawnRelationReceipt> &&
              std::is_trivial_v<DirectSourceWithdrawnRelationReceipt> &&
              std::has_unique_object_representations_v<DirectSourceWithdrawnRelationReceipt>);

DIRECT_SOURCE_EPISTEMICS_HD inline bool find_source_assertion_integrity_event(
    const direct_network::DirectExactHistoryHotPage& history,
    std::uint64_t contact_identity, DirectSourceAssertionIntegrity integrity,
    direct_network::DirectExactHistoryRecord* record) {
  if (contact_identity == 0u || integrity == DirectSourceAssertionIntegrity::none)
    return false;
  const std::uint32_t expected = static_cast<std::uint32_t>(integrity);
  bool found = false;
  direct_network::DirectExactHistoryRecord match{};
  for (std::uint32_t i = 0u; i < history.committed_slots; ++i) {
    const auto& candidate = history.records[i];
    if (candidate.kind != direct_network::DirectExactHistoryKind::source_assertion ||
        candidate.parent_identity != contact_identity ||
        candidate.identity == contact_identity || candidate.flags != expected)
      continue;
    if (found) return false;
    found = true;
    match = candidate;
  }
  if (found && record != nullptr) *record = match;
  return found;
}

DIRECT_SOURCE_EPISTEMICS_HD inline bool find_sensory_history_record(
    const direct_network::DirectExactHistoryHotPage& history, std::uint64_t ticket,
    direct_network::DirectExactHistoryRecord* record) {
  if (ticket == 0u) return false;
  bool found = false;
  direct_network::DirectExactHistoryRecord match{};
  for (std::uint32_t i = 0u; i < history.committed_slots; ++i) {
    const auto& candidate = history.records[i];
    if (candidate.kind != direct_network::DirectExactHistoryKind::sensory_contact ||
        candidate.identity != ticket)
      continue;
    if (found) return false;
    found = true;
    match = candidate;
  }
  if (found && record != nullptr) *record = match;
  return found;
}

DIRECT_SOURCE_EPISTEMICS_HD inline bool source_history_find_assertion(
    const direct_network::DirectExactHistoryRecord* records, std::uint32_t count,
    std::uint64_t contact_identity,
    direct_network::DirectExactHistoryRecord* record) {
  if (records == nullptr || contact_identity == 0u) return false;
  bool found = false;
  direct_network::DirectExactHistoryRecord match{};
  for (std::uint32_t i = 0u; i < count; ++i) {
    const auto& candidate = records[i];
    if (candidate.kind != direct_network::DirectExactHistoryKind::source_assertion ||
        candidate.identity != contact_identity ||
        (candidate.flags & direct_network::kDirectHistoryVerifiedObservation) == 0u)
      continue;
    if (found) return false;
    found = true;
    match = candidate;
  }
  if (found && record != nullptr) *record = match;
  return found;
}

DIRECT_SOURCE_EPISTEMICS_HD inline bool source_history_find_integrity(
    const direct_network::DirectExactHistoryRecord* records, std::uint32_t count,
    std::uint64_t contact_identity, DirectSourceAssertionIntegrity integrity,
    direct_network::DirectExactHistoryRecord* record) {
  if (records == nullptr || contact_identity == 0u ||
      integrity == DirectSourceAssertionIntegrity::none)
    return false;
  const std::uint32_t expected = static_cast<std::uint32_t>(integrity);
  bool found = false;
  direct_network::DirectExactHistoryRecord match{};
  for (std::uint32_t i = 0u; i < count; ++i) {
    const auto& candidate = records[i];
    if (candidate.kind != direct_network::DirectExactHistoryKind::source_assertion ||
        candidate.parent_identity != contact_identity ||
        candidate.identity == contact_identity || candidate.flags != expected)
      continue;
    if (found) return false;
    found = true;
    match = candidate;
  }
  if (found && record != nullptr) *record = match;
  return found;
}

DIRECT_SOURCE_EPISTEMICS_HD inline bool source_history_find_sensory(
    const direct_network::DirectExactHistoryRecord* records, std::uint32_t count,
    std::uint64_t ticket, direct_network::DirectExactHistoryRecord* record) {
  if (records == nullptr || ticket == 0u) return false;
  bool found = false;
  direct_network::DirectExactHistoryRecord match{};
  for (std::uint32_t i = 0u; i < count; ++i) {
    const auto& candidate = records[i];
    if (candidate.kind != direct_network::DirectExactHistoryKind::sensory_contact ||
        candidate.identity != ticket ||
        (candidate.flags & direct_network::kDirectHistoryVerifiedObservation) == 0u)
      continue;
    if (found) return false;
    found = true;
    match = candidate;
  }
  if (found && record != nullptr) *record = match;
  return found;
}

DIRECT_SOURCE_EPISTEMICS_HD inline bool source_history_find_experience_revision(
    const direct_network::DirectExactHistoryRecord* records, std::uint32_t count,
    std::uint64_t evidence_identity,
    direct_network::DirectExactHistoryRecord* record) {
  if (records == nullptr || evidence_identity == 0u) return false;
  bool found = false;
  direct_network::DirectExactHistoryRecord match{};
  for (std::uint32_t i = 0u; i < count; ++i) {
    const auto& candidate = records[i];
    if (candidate.kind != direct_network::DirectExactHistoryKind::recipe_revision ||
        candidate.subject != static_cast<std::uint32_t>(
            direct_network::ResidentRecipeRevisionAuthority::experience) ||
        candidate.incarnation_after != evidence_identity || candidate.identity == 0u)
      continue;
    if (found) return false;
    found = true;
    match = candidate;
  }
  if (found && record != nullptr) *record = match;
  return found;
}

DIRECT_SOURCE_EPISTEMICS_HD inline bool source_history_revision_chain_contains_selection(
    const direct_network::DirectExactHistoryRecord* records, std::uint32_t count,
    std::uint32_t recipe_cell, std::uint64_t logical_recipe_id,
    std::uint64_t current_revision_identity, std::uint64_t occurrence_identity,
    std::uint64_t participation_identity, const DirectActionOccurrence* actions,
    const DirectActionParticipationLink* action_links,
    direct_network::DirectExactHistoryRecord* evidence_record) {
  using namespace direct_network;
  if (records == nullptr || count == 0u || logical_recipe_id == 0u ||
      current_revision_identity == 0u || occurrence_identity == 0u ||
      participation_identity == 0u || actions == nullptr || action_links == nullptr)
    return false;
  std::uint64_t cursor = current_revision_identity;
  for (std::uint32_t depth = 0u; depth < count && cursor != 0u; ++depth) {
    const DirectExactHistoryRecord* step = nullptr;
    for (std::uint32_t i = 0u; i < count; ++i) {
      const auto& candidate = records[i];
      if (candidate.kind != DirectExactHistoryKind::recipe_revision ||
          candidate.identity != cursor || candidate.source != recipe_cell)
        continue;
      if (step != nullptr) return false;
      step = &candidate;
    }
    if (step == nullptr) return false;
    if (step->subject == static_cast<std::uint32_t>(
            ResidentRecipeRevisionAuthority::experience) &&
        step->incarnation_before == occurrence_identity &&
        step->incarnation_after != 0u) {
      const std::uint64_t action_ticket = step->incarnation_after;
      const std::uint32_t action_slot = static_cast<std::uint32_t>(
          action_ticket & (kMaxAsynchronousTickets - 1u));
      const DirectActionOccurrence& action = actions[action_slot];
      if (action.action_ticket_id == action_ticket &&
          action.state == kActionOccurrenceSettled &&
          action.participant_offset != ::substrate::direct_adult_core::kInvalidIndex &&
          action.participant_count != 0u &&
          action.participant_count <= kMaxActionParticipationLinks) {
        bool matched = false;
        for (std::uint32_t i = 0u; i < action.participant_count; ++i) {
          const DirectActionParticipationLink& link =
              action_links[action.participant_offset + i];
          if (link.occurrence_identity == occurrence_identity &&
              link.participation_identity != participation_identity)
            return false;
          if (link.logical_recipe_id == logical_recipe_id &&
              link.revision_identity == step->parent_identity &&
              link.occurrence_identity == occurrence_identity &&
              link.participation_identity == participation_identity &&
              link.authority == DirectParticipationAuthority::independent_external &&
              link.contribution_kind == DirectContributionKind::sparse_route)
            matched = true;
        }
        if (matched) {
          if (evidence_record != nullptr) *evidence_record = *step;
          return true;
        }
      }
    }
    if (step->parent_identity == cursor) return false;
    cursor = step->parent_identity;
  }
  return false;
}

// Observer-only KG1 join. It creates no resident fact/knowledge state.
// Source history proves only which authenticated contact occurred and that the
// source was later withdrawn. Relation authority is resident network state:
// one consequence-derived DirectCausalRelation whose settled action froze the
// asserted contact as a causal participant. Recipe/Occurrence representations
// may change or disappear without erasing the learned relation.
DIRECT_SOURCE_EPISTEMICS_HD inline bool reconstruct_source_withdrawn_relation(
    const direct_network::DirectExactHistoryRecord* records,
    std::uint32_t record_count,
    const direct_network::DirectBrain& brain,
    const direct_network::ResidentPostbirthConstructorState& state,
    const direct_network::DirectCausalWorldModel& causal_world_model,
    const AsynchronousTicket* tickets, const DirectActionOccurrence* actions,
    const DirectActionParticipationLink* action_links,
    std::uint32_t action_participant_capacity,
    std::uint64_t assertion_contact_identity,
    DirectSourceWithdrawnRelationReceipt* receipt) {
  using namespace direct_network;
  if (records == nullptr || record_count == 0u || receipt == nullptr ||
      tickets == nullptr || actions == nullptr || action_links == nullptr ||
      brain.routes == nullptr || brain.recipe_cells == nullptr ||
      brain.postbirth_derivations == nullptr ||
      action_participant_capacity == 0u || assertion_contact_identity == 0u ||
      state.recipe_incidence_count > kResidentRecipeIncidenceCapacity ||
      state.raw_contact_binding_count > kResidentRecipeIncidenceCapacity ||
      causal_world_model.relation_count > kCausalModelRelationCapacity)
    return false;

  DirectExactHistoryRecord assertion_record{}, withdrawal_record{}, sensory_record{};
  DirectSourceAssertionClaim claim{};
  if (!source_history_find_assertion(records, record_count,
                                     assertion_contact_identity,
                                     &assertion_record) ||
      !read_source_assertion_claim(assertion_record, &claim) ||
      claim.contact_authenticated == 0u || claim.history_sequence == 0u ||
      !source_history_find_integrity(records, record_count,
                                     assertion_contact_identity,
                                     DirectSourceAssertionIntegrity::withdrawn,
                                     &withdrawal_record) ||
      withdrawal_record.sequence <= assertion_record.sequence ||
      !source_history_find_sensory(records, record_count, claim.ticket,
                                   &sensory_record))
    return false;

  const ResidentRawContactKey original_contact{
      sensory_record.source, sensory_record.subject, sensory_record.value,
      sensory_record.context};
  for (std::uint32_t i = 0u; i < state.raw_contact_binding_count; ++i)
    if (state.raw_contact_bindings[i].ticket_id == claim.ticket ||
        resident_raw_contact_key_equal(state.raw_contact_bindings[i].key,
                                       original_contact))
      return false;
  for (std::uint32_t i = 0u; i < state.recipe_incidence_count; ++i)
    if (resident_raw_contact_key_equal(state.recipe_incidence[i].contact,
                                       original_contact))
      return false;

  const std::uint64_t total_action_links =
      static_cast<std::uint64_t>(kMaxAsynchronousTickets) *
      action_participant_capacity;
  std::uint32_t chosen_relation = kInvalidIndex;
  std::uint64_t chosen_action_ticket = 0u;
  std::uint32_t chosen_emission_tick = 0u;
  for (std::uint32_t slot = 0u; slot < kMaxAsynchronousTickets; ++slot) {
    const AsynchronousTicket& ticket = tickets[slot];
    const DirectActionOccurrence& action = actions[slot];
    if (ticket.ticket_id == 0u || ticket.settled == 0u ||
        action.action_ticket_id != ticket.ticket_id ||
        action.state != kActionOccurrenceSettled ||
        action.motor_node != ticket.motor_node ||
        action.motor_channel != ticket.motor_channel ||
        action.context_signature != ticket.context_signature ||
        action.emission_tick != ticket.emission_tick ||
        action.expiry_tick < action.emission_tick ||
        action.participant_offset != slot * action_participant_capacity ||
        action.participant_offset == kInvalidIndex ||
        action.participant_count == 0u ||
        action.participant_count > action_participant_capacity ||
        static_cast<std::uint64_t>(action.participant_offset) +
                action.participant_count >
            total_action_links)
      continue;
    bool assertion_participated = false, participants_valid = true;
    std::uint32_t assertion_claim_incarnation = 0u;
    std::uint32_t assertion_authority_incarnation = 0u;
    for (std::uint32_t i = 0u; i < action.participant_count; ++i) {
      const DirectActionParticipationLink& link =
          action_links[action.participant_offset + i];
      if (link.participant_ticket_id == 0u ||
          link.claim_incarnation == 0u || link.authority_incarnation == 0u ||
          link.expiry_tick < action.emission_tick ||
          link.authority != DirectParticipationAuthority::independent_external ||
          (link.contribution_kind != DirectContributionKind::direct_ingress &&
           link.contribution_kind != DirectContributionKind::sparse_route)) {
        participants_valid = false;
        break;
      }
      if (link.participant_ticket_id != claim.ticket) continue;
      if (assertion_claim_incarnation == 0u) {
        assertion_claim_incarnation = link.claim_incarnation;
        assertion_authority_incarnation = link.authority_incarnation;
      } else if (link.claim_incarnation != assertion_claim_incarnation ||
                 link.authority_incarnation !=
                     assertion_authority_incarnation) {
        participants_valid = false;
        break;
      }
      // One sensory root may traverse several sparse routes. Only the
      // terminal route freezes the actual Occurrence; upstream links remain
      // exact action participation but are not separate Occurrences.
      if (link.occurrence_identity == 0u) continue;
      if (link.participation_identity != claim.ticket ||
          link.logical_recipe_id == 0u || link.revision_identity == 0u ||
          link.route_index == kInvalidIndex ||
          link.route_index >= brain.route_capacity) {
        participants_valid = false;
        break;
      }
      // The terminal route may itself be built by a downstream executor
      // Recipe. Resolve the frozen Occurrence's logical Recipe through the
      // resident derivation directory, not through that route's builder.
      std::uint32_t derivation_matches = 0u, recipe_cell = kInvalidIndex;
      for (std::uint32_t d = 0u; d < state.derivation_count; ++d) {
        const ResidentRecipeDerivation& derivation =
            brain.postbirth_derivations[d];
        if (derivation.logical_recipe_id != link.logical_recipe_id ||
            derivation.recipe_cell >= brain.development->recipe_cell_count)
          continue;
        const ResidentRecipeCell& candidate =
            brain.recipe_cells[derivation.recipe_cell];
        if (candidate.logical_recipe_id != link.logical_recipe_id ||
            candidate.revision_identity != derivation.revision_identity)
          continue;
        recipe_cell = derivation.recipe_cell;
        ++derivation_matches;
      }
      if (derivation_matches != 1u) {
        participants_valid = false;
        break;
      }
      const ResidentRecipeCell& recipe = brain.recipe_cells[recipe_cell];
      if (recipe.logical_recipe_id != link.logical_recipe_id ||
          recipe.revision_identity == link.revision_identity ||
          resident_recipe_current_revision_authority(recipe) !=
              ResidentRecipeRevisionAuthority::experience ||
          recipe.revision_identity == 0u) {
        participants_valid = false;
        break;
      }
      assertion_participated = true;
    }
    if (!participants_valid) continue;
    if (!assertion_participated) continue;

    const std::int32_t relation_index = causal_model_find_relation(
        causal_world_model, ticket.motor_word, sensory_record.subject);
    if (relation_index < 0) continue;
    const DirectCausalRelation& relation =
        causal_world_model.relations[relation_index];
    if (relation.observations < kCausalModelMinimumSupport) continue;
    const std::uint32_t index = static_cast<std::uint32_t>(relation_index);
    if (chosen_relation != kInvalidIndex && chosen_relation != index)
      return false;
    chosen_relation = index;
    if (chosen_action_ticket == 0u || ticket.emission_tick >= chosen_emission_tick) {
      chosen_action_ticket = ticket.ticket_id;
      chosen_emission_tick = ticket.emission_tick;
    }
  }
  if (chosen_relation == kInvalidIndex || chosen_action_ticket == 0u)
    return false;

  const DirectCausalRelation& relation =
      causal_world_model.relations[chosen_relation];
  *receipt = DirectSourceWithdrawnRelationReceipt{};
  receipt->source_identity = claim.source_identity;
  receipt->context_identity = claim.context_identity;
  receipt->assertion_contact_identity = assertion_contact_identity;
  receipt->assertion_history_sequence = assertion_record.sequence;
  receipt->withdrawal_history_sequence = withdrawal_record.sequence;
  receipt->action_ticket_id = chosen_action_ticket;
  receipt->causal_model_identity = causal_world_model.model_identity;
  receipt->action_value = relation.action_value;
  receipt->root_channel = relation.root_channel;
  receipt->current_outcome_value = relation.current_outcome_value;
  receipt->current_outcome_tick = relation.current_outcome_tick;
  receipt->observations = relation.observations;
  receipt->relation_index = chosen_relation;
  return true;
}

DIRECT_SOURCE_EPISTEMICS_HD inline bool commit_source_assertion_integrity_from_authority(
    direct_network::DirectExactHistoryHotPage* history,
    const direct_network::DirectExactHistoryRecord& original,
    DirectSourceAssertionIntegrity integrity, std::uint32_t tick) {
  using direct_network::exact_history_fold_word;
  DirectSourceAssertionClaim claim{};
  if (history == nullptr || integrity == DirectSourceAssertionIntegrity::none ||
      original.kind != direct_network::DirectExactHistoryKind::source_assertion ||
      (original.flags & direct_network::kDirectHistoryVerifiedObservation) == 0u ||
      !read_source_assertion_claim(original, &claim) || claim.contact_identity == 0u)
    return false;
  const std::uint64_t event_identity = exact_history_fold_word(
      exact_history_fold_word(claim.contact_identity,
                              static_cast<std::uint32_t>(integrity)),
      tick);
  if (!direct_network::begin_exact_history_phase(
          history, direct_network::DirectExactHistoryKind::source_assertion, 1u,
          tick))
    return false;
  auto& record = history->records[history->phase_base];
  record = original;
  record.identity = event_identity == 0u ? 1u : event_identity;
  record.parent_identity = claim.contact_identity;
  record.resident_tick = tick;
  record.event_tick = tick;
  record.kind = direct_network::DirectExactHistoryKind::source_assertion;
  record.flags = static_cast<std::uint32_t>(integrity);
  record.resource_delta = 0;
  return direct_network::finish_exact_history_phase(history) == 1u;
}

DIRECT_SOURCE_EPISTEMICS_HD inline bool apply_source_assertion_integrity(
    direct_network::DirectExactHistoryHotPage* history,
    std::uint64_t contact_identity, DirectSourceAssertionIntegrity integrity,
    std::uint32_t tick) {
  direct_network::DirectExactHistoryRecord original{};
  if (history == nullptr ||
      !find_committed_source_assertion(*history, contact_identity, &original))
    return false;
  return commit_source_assertion_integrity_from_authority(
      history, original, integrity, tick);
}

}  // namespace substrate::direct_adult_core

#undef DIRECT_SOURCE_EPISTEMICS_HD
#endif  // HARDWARE_NATIVE_DIRECT_ADULT_SOURCE_EPISTEMICS_CUH
