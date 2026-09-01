#ifndef HARDWARE_NATIVE_DIRECT_ADULT_RESIDENT_LANGUAGE_EXECUTOR_CUH
#define HARDWARE_NATIVE_DIRECT_ADULT_RESIDENT_LANGUAGE_EXECUTOR_CUH
#include <cstdint>
#include <type_traits>
#include "hardware_native/direct_adult_resident_language_state.cuh"
#include "hardware_native/direct_adult_surface_sequence.cuh"
namespace substrate::direct_network {
struct DirectResidentLanguageExecutorState {
  DirectSurfaceSequenceState surface;
  DirectLanguageMotorPlan pending;
  std::uint32_t start_tick;
  std::uint32_t active;
  std::uint32_t plans_started;
  std::uint32_t due_steps_driven;
  std::uint32_t plans_completed;
  std::uint64_t last_cue_identity;
  std::uint64_t revision_identity;
};
struct DirectResidentLanguageTickReceipt {
  DirectLanguageMotorStep driven_step;
  DirectDiscourseAction discourse;
  std::uint32_t plan_started;
  std::uint32_t step_driven;
  std::uint32_t plan_completed;
  std::uint32_t elapsed;
  std::uint32_t pending_steps;
  std::uint64_t revision_identity;
};
static_assert(std::is_trivially_copyable_v<DirectResidentLanguageExecutorState>);
static_assert(std::is_trivially_copyable_v<DirectResidentLanguageTickReceipt>);

// Emission-confirmed sequencing: once a step's scheduled time has arrived it
// remains due until the ordinary public motor path confirms that exact step.
// Returning plan.step_count means there is no currently due unconfirmed step.
__host__ __device__ inline std::uint32_t resident_language_next_due_step(
    const DirectLanguageMotorPlan& plan, std::uint32_t confirmed_emitted_steps,
    std::uint32_t elapsed) {
  if (plan.admitted == 0u || plan.step_count == 0u ||
      confirmed_emitted_steps >= plan.step_count)
    return plan.step_count;
  const std::uint32_t step = confirmed_emitted_steps;
  if (step != 0u && plan.steps[step].due_offset < plan.steps[step - 1u].due_offset)
    return plan.step_count;
  return plan.steps[step].due_offset <= elapsed ? step : plan.step_count;
}

__host__ __device__ inline bool resident_language_plan_emission_complete(
    const DirectLanguageMotorPlan& plan, std::uint32_t confirmed_emitted_steps) {
  return plan.admitted != 0u && plan.step_count != 0u &&
         confirmed_emitted_steps >= plan.step_count;
}

__device__ inline std::uint32_t resident_language_plan_max_due(
    const DirectLanguageMotorPlan& plan) {
  std::uint32_t due = 0u;
  for (std::uint32_t i = 0u; i < plan.step_count; ++i)
    due = plan.steps[i].due_offset > due ? plan.steps[i].due_offset : due;
  return due;
}


__device__ inline std::uint64_t resident_language_latest_cue_identity(
    const DirectExactHistoryRecord* records, std::uint32_t count) {
  for (std::uint32_t cursor = count; cursor > 0u; --cursor) {
    const auto& record = records[cursor - 1u];
    if (record.kind != DirectExactHistoryKind::sensory_contact) continue;
    return resident_language_fold(
        resident_language_fold(record.identity, record.subject), record.value);
  }
  return 0u;
}

__host__ __device__ inline bool resident_language_select_due_surface(
    const DirectResidentLanguageTickReceipt& receipt, std::uint32_t motor_node,
    std::uint32_t* channel, direct_adult_core::Word* word) {
  if (receipt.step_driven == 0u || channel == nullptr || word == nullptr ||
      motor_node != receipt.driven_step.node)
    return false;
  *channel = receipt.driven_step.channel;
  *word = receipt.driven_step.word;
  return true;
}

__device__ inline bool resident_language_apply_due_step_to_ticket(
    const DirectResidentLanguageTickReceipt& receipt,
    direct_adult_core::AsynchronousTicket* ticket) {
  if (receipt.step_driven == 0u || ticket == nullptr ||
      ticket->ticket_id == 0u || ticket->ticket_id == direct_adult_core::kInvalidTicket ||
      ticket->settled != 0u || ticket->motor_node != receipt.driven_step.node)
    return false;
  ticket->motor_channel = receipt.driven_step.channel;
  ticket->motor_word = receipt.driven_step.word;
  return true;
}

__device__ inline bool resident_language_bind_due_step_public_surface(
    const DirectResidentLanguageTickReceipt& receipt,
    direct_adult_core::MotorEvent* event,
    direct_adult_core::AsynchronousTicket* ticket,
    DirectExactHistoryRecord* history_record,
    DirectEfferenceCopy* efference) {
  if (receipt.step_driven == 0u || event == nullptr || ticket == nullptr ||
      event->ticket_id == 0u || event->ticket_id == direct_adult_core::kInvalidTicket ||
      ticket->ticket_id != event->ticket_id || ticket->settled != 0u ||
      event->node != receipt.driven_step.node || ticket->motor_node != event->node ||
      ticket->emission_tick != event->timestamp)
    return false;
  if (history_record != nullptr &&
      (history_record->kind != DirectExactHistoryKind::motor_output ||
       history_record->identity != event->ticket_id ||
       history_record->source != event->node))
    return false;
  if (efference != nullptr &&
      (efference->ticket_id != event->ticket_id ||
       efference->acting_node != event->node ||
       efference->timestamp != event->timestamp))
    return false;

  ticket->motor_channel = receipt.driven_step.channel;
  ticket->motor_word = receipt.driven_step.word;
  event->channel = receipt.driven_step.channel;
  event->word = receipt.driven_step.word;
  if (history_record != nullptr) {
    history_record->subject = receipt.driven_step.channel;
    history_record->value = receipt.driven_step.word;
  }
  if (efference != nullptr) {
    efference->channel = receipt.driven_step.channel;
    efference->word = receipt.driven_step.word;
  }
  return true;
}

__device__ inline DirectResidentLanguageTickReceipt resident_language_plan_only_tick(
    DirectResidentLanguageState* language,
    DirectResidentLanguageExecutorState* executor,
    const DirectExactHistoryRecord* records, std::uint32_t count,
    std::uint32_t current_tick, bool partner_return_unresolved) {
  DirectResidentLanguageTickReceipt receipt{};
  receipt.discourse = DirectDiscourseAction::remain_silent;
  if (language == nullptr || executor == nullptr || records == nullptr) return receipt;
  const bool compacted = resident_language_follow_hot_suffix(language, records, count);
  if (compacted || count < executor->surface.cursor) executor->surface.cursor = 0u;
  resident_language_assimilate(language, records, count);
  surface_sequence_assimilate(&executor->surface, records, count);
  if (executor->active == 0u) {
    const std::uint64_t cue = resident_language_latest_cue_identity(records, count);
    if (cue == 0u || cue == executor->last_cue_identity) return receipt;
    executor->last_cue_identity = cue;
    DirectResidentLanguageDecision decision = resident_language_decide_recent(
        language, records, count, partner_return_unresolved);
    const DirectSurfaceSequencePlan surface =
        surface_sequence_plan_recent(&executor->surface, records, count);
    surface_sequence_bind_expression(surface, &decision.expression);
    receipt.discourse = decision.discourse;
    if (decision.expression.admitted == 0u ||
        decision.discourse == DirectDiscourseAction::remain_silent) return receipt;
    executor->pending = decision.expression;
    executor->start_tick = current_tick;
    executor->active = 1u;
    ++executor->plans_started;
    executor->revision_identity = resident_language_fold(
        resident_language_fold(executor->revision_identity, decision.revision_identity),
        surface.revision_identity);
    receipt.plan_started = 1u;
  }
  receipt.pending_steps = executor->pending.step_count;
  receipt.revision_identity = executor->revision_identity;
  return receipt;
}

__device__ inline DirectResidentLanguageTickReceipt resident_language_tick(
    DirectResidentLanguageState* language,
    DirectResidentLanguageExecutorState* executor,
    const DirectExactHistoryRecord* records, std::uint32_t count,
    std::uint32_t current_tick, bool partner_return_unresolved,
    std::uint32_t confirmed_emitted_steps,
    DirectNode* nodes, std::uint32_t node_count) {
  DirectResidentLanguageTickReceipt receipt{};
  receipt.discourse = DirectDiscourseAction::remain_silent;
  if (language == nullptr || executor == nullptr || records == nullptr ||
      nodes == nullptr)
    return receipt;

  const bool compacted =
      resident_language_follow_hot_suffix(language, records, count);
  if (compacted || count < executor->surface.cursor)
    executor->surface.cursor = 0u;
  resident_language_assimilate(language, records, count);
  surface_sequence_assimilate(&executor->surface, records, count);
  if (executor->active == 0u) {
    const std::uint64_t cue = resident_language_latest_cue_identity(records, count);
    if (cue == 0u || cue == executor->last_cue_identity) return receipt;
    executor->last_cue_identity = cue;
    DirectResidentLanguageDecision decision = resident_language_decide_recent(
        language, records, count, partner_return_unresolved);
    const DirectSurfaceSequencePlan surface =
        surface_sequence_plan_recent(&executor->surface, records, count);
    surface_sequence_bind_expression(surface, &decision.expression);
    receipt.discourse = decision.discourse;
    if (decision.expression.admitted == 0u ||
        decision.discourse == DirectDiscourseAction::remain_silent)
      return receipt;
    executor->pending = decision.expression;
    executor->start_tick = current_tick;
    executor->active = 1u;
    ++executor->plans_started;
    executor->revision_identity = resident_language_fold(
        resident_language_fold(executor->revision_identity, decision.revision_identity),
        surface.revision_identity);
    confirmed_emitted_steps = 0u;
    receipt.plan_started = 1u;
  }

  const std::uint32_t elapsed = current_tick - executor->start_tick;
  receipt.elapsed = elapsed;
  receipt.pending_steps = executor->pending.step_count;
  const std::uint32_t due = resident_language_next_due_step(
      executor->pending, confirmed_emitted_steps, elapsed);
  if (due < executor->pending.step_count) {
    const DirectLanguageMotorStep& step = executor->pending.steps[due];
    if (step.node < node_count) {
      atomicAdd(&nodes[step.node].activation_q16,
                static_cast<std::int32_t>(kQ16One));
      atomicAdd(&nodes[step.node].credit_ema_q16,
                static_cast<std::int32_t>(kQ16One / 8u));
      receipt.driven_step = step;
      ++executor->due_steps_driven;
      receipt.step_driven = 1u;
    }
  }
  if (resident_language_plan_emission_complete(
          executor->pending, confirmed_emitted_steps)) {
    executor->active = 0u;
    ++executor->plans_completed;
    receipt.plan_completed = 1u;
  }
  receipt.revision_identity = executor->revision_identity;
  return receipt;
}

}  // namespace substrate::direct_network

#endif
