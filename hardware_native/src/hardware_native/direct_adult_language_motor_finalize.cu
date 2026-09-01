#ifdef DIRECT_ADULT_DEVICE_OPS_DECLARATIONS_ONLY
#undef DIRECT_ADULT_DEVICE_OPS_DECLARATIONS_ONLY
#endif

#include <cuda_runtime.h>

#include <stdexcept>

#include "hardware_native/direct_adult_core_kernels.cuh"
#include "hardware_native/direct_adult_language_expression_opportunity.cuh"
#include "hardware_native/direct_adult_lived_expression_identity.cuh"
#include "hardware_native/direct_adult_resident_language_runtime.cuh"

namespace substrate::direct_adult_core {

__device__ bool resident_language_bind_emitted_step(
    direct_network::DirectResidentLanguageRuntimeBlock* language,
    MotorEvent* event, AsynchronousTicket* ticket,
    direct_network::DirectExactHistoryRecord* history_record,
    direct_network::DirectEfferenceCopy* efference) {
  if (language == nullptr || event == nullptr || ticket == nullptr ||
      language->last_tick_receipt.step_driven == 0u)
    return false;
  if (!direct_network::resident_language_bind_due_step_public_surface(
          language->last_tick_receipt, event, ticket, history_record, efference))
    return false;
  if (language->confirmed_emitted_steps < language->executor.pending.step_count)
    ++language->confirmed_emitted_steps;
  return true;
}

__device__ bool resident_language_commit_expression_opportunity(
    direct_network::DirectResidentLanguageRuntimeBlock* language,
    const DirectBrain& brain, const ResidentActualFrontier* frontier,
    const DirectActionOccurrence& action,
    const DirectActionParticipationLink* action_links) {
  if (language == nullptr || frontier == nullptr || action_links == nullptr ||
      language->last_tick_receipt.plan_completed == 0u ||
      language->pending_spoken_surface_content_identity == 0u ||
      language->executor.pending.step_count == 0u ||
      language->confirmed_emitted_steps != language->executor.pending.step_count ||
      action.participant_offset == kInvalidIndex || action.participant_count == 0u)
    return false;
  const ResidentActualFrontierEntry* match = nullptr;
  for (std::uint32_t p = 0u; p < action.participant_count; ++p) {
    const auto* entry = resident_current_lived_expression_entry(
        brain, frontier, action_links[action.participant_offset + p]);
    if (entry == nullptr) continue;
    if (match != nullptr && match != entry) return false;
    match = entry;
  }
  if (match == nullptr) return false;
  const bool recorded = direct_network::resident_language_record_recipe_opportunity(
      language, language->pending_spoken_surface_content_identity,
      match->occurrence,
      static_cast<std::uint16_t>(language->pending_spoken_surface_length));
  if (recorded) {
    language->pending_spoken_surface_content_identity = 0u;
    language->pending_spoken_surface_length = 0u;
    language->confirmed_emitted_steps = 0u;
  }
  return recorded;
}

__device__ std::uint64_t resident_language_current_discourse_recruitment(
    const direct_network::DirectResidentLanguageRuntimeBlock* language) {
  return language != nullptr
             ? direct_network::direct_discourse_focus_select(
                   language->language.discourse_focus)
             : 0u;
}

__device__ std::uint32_t resident_language_current_discourse_relation_nominations(
    const direct_network::DirectResidentLanguageRuntimeBlock* language,
    std::uint64_t* logical_recipe_ids, std::uint64_t* revision_identities,
    std::uint32_t capacity, bool* relation_blocks_fallback) {
  using namespace direct_network;
  if (relation_blocks_fallback != nullptr) *relation_blocks_fallback = false;
  if (language == nullptr || logical_recipe_ids == nullptr ||
      revision_identities == nullptr || capacity == 0u)
    return 0u;
  const auto& focus = language->language.discourse_focus;
  if (focus.relation_count == 0u) return 0u;
  if (relation_blocks_fallback != nullptr) *relation_blocks_fallback = true;
  std::uint32_t selected_index = kDirectDiscourseRelationFocusCapacity;
  if (direct_discourse_relation_focus_select(focus, &selected_index) !=
          DirectDiscourseRelationSelectionStatus::unique ||
      selected_index >= focus.relation_count)
    return 0u;
  const auto& relation = focus.relation_entries[selected_index];
  if (!direct_discourse_relation_entry_valid(relation) ||
      relation.recipe_count > capacity)
    return 0u;
  for (std::uint32_t i = 0u; i < relation.recipe_count; ++i) {
    logical_recipe_ids[i] = relation.logical_recipe_ids[i];
    revision_identities[i] = relation.revision_identities[i];
  }
  return relation.recipe_count;
}

__device__ std::uint32_t resident_language_current_recipe_nominations(
    direct_network::DirectResidentLanguageRuntimeBlock* language,
    const direct_network::DirectBrain& brain, std::uint32_t current_tick,
    std::uint64_t* logical_recipe_ids, std::uint64_t* revision_identities,
    std::uint32_t capacity, std::uint64_t* recruitment_identity) {
  if (recruitment_identity != nullptr) *recruitment_identity = 0u;
  if (logical_recipe_ids == nullptr || revision_identities == nullptr || capacity == 0u)
    return 0u;
  if (language == nullptr || brain.development == nullptr ||
      brain.recipe_cells == nullptr || brain.postbirth_constructor == nullptr ||
      brain.postbirth_derivations == nullptr ||
      language->last_heard_surface_content_identity == 0u)
    return 0u;
  const std::uint64_t occurrence_identity =
      0x4c4e000000000000ull |
      static_cast<std::uint64_t>(current_tick == 0u ? 1u : current_tick);
  const auto& ecology = language->executor.surface.ecology;
  std::uint32_t count = direct_adult_core::collect_resident_language_recipes_from_contained_surface(
      language->opportunity_bank, ecology.last_closed_values, ecology.last_closed_length,
      brain.recipe_cells, brain.development->recipe_cell_count,
      brain.postbirth_derivations, brain.postbirth_constructor->derivation_count,
      occurrence_identity, logical_recipe_ids, revision_identities, capacity);
  if (count == 0u) {
    ResidentRecipeOccurrence nominated{};
    if (direct_network::resident_language_nominate_heard_recipe(
            *language, brain.recipe_cells, brain.development->recipe_cell_count,
            brain.postbirth_derivations, brain.postbirth_constructor->derivation_count,
            occurrence_identity, &nominated)) {
      logical_recipe_ids[0] = nominated.logical_recipe_id;
      revision_identities[0] = nominated.revision_identity;
      count = 1u;
    }
  }
  if (count >= 2u && recruitment_identity != nullptr) {
    const auto* span = select_resident_language_span_recipe_by_children_and_consequence(
        language->span_bank, brain.development->recruited_networks,
        logical_recipe_ids, revision_identities, count);
    if (span != nullptr &&
        resident_language_span_children_current(
            *span, brain.recipe_cells, brain.development->recipe_cell_count))
      *recruitment_identity = span->recruitment_identity;
  }
  language->last_heard_surface_content_identity = 0u;
  return count;
}

__device__ bool resident_language_record_network_span_candidate(
    direct_network::DirectResidentLanguageRuntimeBlock* language,
    const DirectActionOccurrence& action,
    const ResidentPublicMotorTrajectory& trajectory) {
  if (language == nullptr || action.action_ticket_id == 0u ||
      action.network_identity == 0u || action.recruitment_identity == 0u ||
      trajectory.state != kResidentPublicMotorTrajectoryLive ||
      trajectory.trajectory_identity != action.action_ticket_id ||
      trajectory.extent < 2u ||
      trajectory.extent > kResidentLanguageSpanMaxChildren ||
      trajectory.contributor_count != trajectory.extent)
    return false;
  std::uint64_t logical[kResidentLanguageSpanMaxChildren]{};
  std::uint64_t revision[kResidentLanguageSpanMaxChildren]{};
  for (std::uint32_t i = 0u; i < trajectory.extent; ++i) {
    const auto& link = trajectory.contributors[i];
    if (link.logical_recipe_id == 0u || link.revision_identity == 0u)
      return false;
    logical[i] = link.logical_recipe_id;
    revision[i] = link.revision_identity;
  }
  return record_resident_language_span_candidate(
      &language->span_bank, action.action_ticket_id, action.recruitment_identity,
      logical, revision, trajectory.extent);
}

__device__ bool resident_language_current_recipe_nomination(
    direct_network::DirectResidentLanguageRuntimeBlock* language,
    const direct_network::DirectBrain& brain, std::uint32_t current_tick,
    std::uint64_t* logical_recipe_id, std::uint64_t* revision_identity) {
  if (logical_recipe_id == nullptr || revision_identity == nullptr) return false;
  *logical_recipe_id = 0u;
  *revision_identity = 0u;
  if (language == nullptr || brain.development == nullptr ||
      brain.recipe_cells == nullptr || brain.postbirth_constructor == nullptr ||
      brain.postbirth_derivations == nullptr ||
      language->last_heard_surface_content_identity == 0u)
    return false;
  ResidentRecipeOccurrence nominated{};
  const std::uint64_t occurrence_identity =
      0x4c4e000000000000ull |
      static_cast<std::uint64_t>(current_tick == 0u ? 1u : current_tick);
  const bool nominated_ok = direct_network::resident_language_nominate_heard_recipe(
      *language, brain.recipe_cells, brain.development->recipe_cell_count,
      brain.postbirth_derivations, brain.postbirth_constructor->derivation_count,
      occurrence_identity, &nominated);
  // A heard surface is a current contextual event, not persistent motor bias.
  language->last_heard_surface_content_identity = 0u;
  if (!nominated_ok ||
      nominated.lineage_kind != ResidentOccurrenceLineageKind::endogenous ||
      nominated.authority != DirectParticipationAuthority::none ||
      nominated.eligibility_q16 != 0)
    return false;
  *logical_recipe_id = nominated.logical_recipe_id;
  *revision_identity = nominated.revision_identity;
  return *logical_recipe_id != 0u && *revision_identity != 0u;
}

__device__ void finalize_resident_language_motor_owned(
    direct_network::DirectResidentLanguageRuntimeBlock* resident_language,
    DirectBrain brain, const ResidentActualFrontier* actual_frontier,
    MotorEvent* egress_queue, const std::uint32_t* egress_head,
    const std::uint32_t* egress_tail, AsynchronousTicket* ticket_table,
    DirectActionOccurrence* action_occurrences,
    DirectActionParticipationLink* action_participation_links,
    ResidentDevelopmentState* development, DirectEfferenceCopy* efference_ring,
    const std::uint32_t* efference_head, const std::uint32_t* efference_tail,
    std::uint32_t route_efference_copies, std::uint32_t current_tick) {
  if (resident_language == nullptr || actual_frontier == nullptr ||
      egress_queue == nullptr || egress_head == nullptr || egress_tail == nullptr ||
      ticket_table == nullptr || action_occurrences == nullptr ||
      action_participation_links == nullptr || development == nullptr ||
      resident_language->last_tick_receipt.step_driven == 0u)
    return;

  const std::uint32_t head = *egress_head;
  const std::uint32_t tail = *egress_tail;
  if (tail - head > kMaxEgressQueueSize) return;
  MotorEvent* event = nullptr;
  for (std::uint32_t cursor = tail; cursor > head; --cursor) {
    MotorEvent& candidate = egress_queue[(cursor - 1u) % kMaxEgressQueueSize];
    if (candidate.timestamp != current_tick) break;
    if (candidate.node != resident_language->last_tick_receipt.driven_step.node)
      continue;
    if (event != nullptr) return;
    event = &candidate;
  }
  if (event == nullptr || event->ticket_id == 0u || event->ticket_id == kInvalidTicket)
    return;

  const std::uint32_t ticket_slot =
      static_cast<std::uint32_t>(event->ticket_id % kMaxAsynchronousTickets);
  AsynchronousTicket* ticket = &ticket_table[ticket_slot];
  DirectActionOccurrence* action = &action_occurrences[ticket_slot];
  if (ticket->ticket_id != event->ticket_id || ticket->settled != 0u ||
      action->action_ticket_id != event->ticket_id)
    return;

  auto& history = development->exact_history;
  if (history.last_phase_records == 0u ||
      history.last_phase_records > history.committed_slots)
    return;
  const std::uint32_t history_begin =
      history.committed_slots - history.last_phase_records;
  direct_network::DirectExactHistoryRecord* history_record = nullptr;
  for (std::uint32_t i = history_begin; i < history.committed_slots; ++i) {
    auto& record = history.records[i];
    if (record.kind != direct_network::DirectExactHistoryKind::motor_output ||
        record.identity != event->ticket_id)
      continue;
    if (history_record != nullptr) return;
    history_record = &record;
  }
  if (history_record == nullptr) return;

  DirectEfferenceCopy* efference = nullptr;
  if (route_efference_copies != 0u) {
    if (efference_ring == nullptr || efference_head == nullptr || efference_tail == nullptr ||
        *efference_tail - *efference_head > kMaxEfferenceRingSize)
      return;
    for (std::uint32_t cursor = *efference_head; cursor < *efference_tail; ++cursor) {
      DirectEfferenceCopy& candidate = efference_ring[cursor % kMaxEfferenceRingSize];
      if (candidate.ticket_id != event->ticket_id) continue;
      if (efference != nullptr) return;
      efference = &candidate;
    }
    if (efference == nullptr) return;
  }

  if (!resident_language_bind_emitted_step(
          resident_language, event, ticket, history_record, efference))
    return;
  (void)resident_language_commit_expression_opportunity(
      resident_language, brain, actual_frontier, *action, action_participation_links);
}

__global__ void finalize_resident_language_motor_kernel(
    direct_network::DirectResidentLanguageRuntimeBlock* resident_language,
    DirectBrain brain, const ResidentActualFrontier* actual_frontier,
    MotorEvent* egress_queue, const std::uint32_t* egress_head,
    const std::uint32_t* egress_tail, AsynchronousTicket* ticket_table,
    DirectActionOccurrence* action_occurrences,
    DirectActionParticipationLink* action_participation_links,
    ResidentDevelopmentState* development, DirectEfferenceCopy* efference_ring,
    const std::uint32_t* efference_head, const std::uint32_t* efference_tail,
    std::uint32_t route_efference_copies, std::uint32_t current_tick) {
  if (blockIdx.x == 0u && threadIdx.x == 0u)
    finalize_resident_language_motor_owned(
        resident_language, brain, actual_frontier, egress_queue, egress_head,
        egress_tail, ticket_table, action_occurrences, action_participation_links,
        development, efference_ring, efference_head, efference_tail,
        route_efference_copies, current_tick);
}

void launch_finalize_resident_language_motor(
    direct_network::DirectResidentLanguageRuntimeBlock* resident_language,
    DirectBrain brain, const ResidentActualFrontier* actual_frontier,
    MotorEvent* egress_queue, const std::uint32_t* egress_head,
    const std::uint32_t* egress_tail, AsynchronousTicket* ticket_table,
    DirectActionOccurrence* action_occurrences,
    DirectActionParticipationLink* action_participation_links,
    ResidentDevelopmentState* development, DirectEfferenceCopy* efference_ring,
    const std::uint32_t* efference_head, const std::uint32_t* efference_tail,
    std::uint32_t route_efference_copies, std::uint32_t current_tick,
    CUstream_st* stream) {
  finalize_resident_language_motor_kernel<<<1, 32, 0, stream>>>(
      resident_language, brain, actual_frontier, egress_queue, egress_head,
      egress_tail, ticket_table, action_occurrences, action_participation_links,
      development, efference_ring, efference_head, efference_tail,
      route_efference_copies, current_tick);
  const cudaError_t status = cudaGetLastError();
  if (status != cudaSuccess)
    throw std::runtime_error(
        "launch_finalize_resident_language_motor: kernel launch");
}

}  // namespace substrate::direct_adult_core
