#ifndef HARDWARE_NATIVE_DIRECT_ADULT_RESIDENT_LANGUAGE_STATE_CUH
#define HARDWARE_NATIVE_DIRECT_ADULT_RESIDENT_LANGUAGE_STATE_CUH

// Bounded connective tissue for the already-certified resident language
// organizations.  This layer adds no language rule: one exact-history prefix
// feeds temporal binding and consequence-backed expression, while discourse
// consumes only their resident plan receipts plus the physical turn gate.

#include <cstdint>
#include <type_traits>

#include "hardware_native/direct_adult_discourse_initiative.cuh"
#include "hardware_native/direct_adult_discourse_focus.cuh"
#include "hardware_native/direct_adult_language_expression_motor.cuh"
#include "hardware_native/direct_adult_language_temporal_binding.cuh"

namespace substrate::direct_network {

struct DirectResidentLanguageState {
  DirectLanguageTemporalBindingState temporal;
  DirectLanguageExpressionMotorState expression;
  DirectDiscourseFocusState discourse_focus;
  DirectTurnGate turn;
  std::uint32_t assimilations;
  std::uint32_t reserved;
  std::uint64_t revision_identity;
  std::uint64_t assimilated_sequence;
};

struct DirectResidentLanguageDecision {
  DirectLanguageTemporalPlan temporal;
  DirectLanguageMotorPlan expression;
  DirectDiscourseAction discourse;
  std::uint32_t topic_grounded;
  std::uint32_t expression_learned;
  std::uint64_t revision_identity;
};

static_assert(std::is_trivially_copyable_v<DirectResidentLanguageState>);
static_assert(std::is_trivially_copyable_v<DirectResidentLanguageDecision>);

__host__ __device__ inline std::uint64_t resident_language_fold(
    std::uint64_t hash, std::uint64_t value) {
  return discourse_fold(hash, value);
}

__host__ __device__ inline std::uint64_t resident_language_contact_identity(
    const DirectExactHistoryRecord* records, std::uint32_t begin,
    std::uint32_t count, std::uint64_t identity) {
  if (records == nullptr || begin > count) return identity;
  for (std::uint32_t i = begin; i < count; ++i) {
    if (records[i].kind != DirectExactHistoryKind::sensory_contact) continue;
    identity = resident_language_fold(
        resident_language_fold(identity, records[i].subject), records[i].value);
  }
  return identity;
}

__device__ inline bool resident_language_follow_hot_suffix(
    DirectResidentLanguageState* state,
    const DirectExactHistoryRecord* records, std::uint32_t count) {
  if (state == nullptr) return false;
  bool compacted =
      count < state->temporal.cursor || count < state->expression.cursor;
  if (!compacted && state->assimilated_sequence != 0u && records != nullptr &&
      count != 0u && records[0].sequence > state->assimilated_sequence)
    compacted = true;
  if (compacted) {
    state->temporal.cursor = 0u;
    state->expression.cursor = 0u;
  }
  return compacted;
}

__device__ inline void resident_language_note_sequence(
    DirectResidentLanguageState* state,
    const DirectExactHistoryRecord* records, std::uint32_t count) {
  if (state == nullptr || records == nullptr) return;
  for (std::uint32_t i = 0u; i < count; ++i)
    if (records[i].kind != DirectExactHistoryKind::empty &&
        records[i].sequence > state->assimilated_sequence)
      state->assimilated_sequence = records[i].sequence;
}

__device__ inline void resident_language_assimilate(
    DirectResidentLanguageState* state,
    const DirectExactHistoryRecord* records, std::uint32_t count) {
  if (state == nullptr || records == nullptr) return;
  resident_language_follow_hot_suffix(state, records, count);
  if (count == state->temporal.cursor && count == state->expression.cursor)
    return;
  const std::uint32_t temporal_before = state->temporal.cursor;
  const std::uint32_t expression_before = state->expression.cursor;
  language_temporal_assimilate(&state->temporal, records, count);
  language_expression_assimilate(&state->expression, records, count);
  direct_discourse_focus_assimilate(&state->discourse_focus, records, count);
  resident_language_note_sequence(state, records, count);
  if (state->temporal.cursor == temporal_before &&
      state->expression.cursor == expression_before)
    return;
  ++state->assimilations;
  state->revision_identity = resident_language_fold(
      resident_language_fold(state->revision_identity,
                             state->temporal.revision_identity),
      state->expression.revision_identity);
}

__device__ inline DirectResidentLanguageDecision resident_language_decide(
    DirectResidentLanguageState* state,
    const DirectExactHistoryRecord* records,
    std::uint32_t temporal_begin, std::uint32_t temporal_end,
    std::uint32_t expression_begin, std::uint32_t count,
    bool partner_return_unresolved) {
  DirectResidentLanguageDecision out{};
  out.discourse = DirectDiscourseAction::remain_silent;
  if (state == nullptr || records == nullptr || temporal_begin > temporal_end ||
      temporal_end > count || expression_begin > count)
    return out;
  out.temporal = language_temporal_plan(
      &state->temporal, records, temporal_begin, temporal_end);
  out.expression = language_expression_plan(
      &state->expression, records, expression_begin, count);
  out.topic_grounded = out.temporal.admitted != 0u ? 1u : 0u;
  out.expression_learned = out.expression.admitted != 0u ? 1u : 0u;
  out.discourse = discourse_decide(
      state->turn.spoken_ticket_id == 0u, out.topic_grounded != 0u,
      out.expression_learned != 0u, partner_return_unresolved);
  out.revision_identity = state->revision_identity;
  return out;
}


__device__ inline DirectResidentLanguageDecision resident_language_decide_recent(
    DirectResidentLanguageState* state,
    const DirectExactHistoryRecord* records, std::uint32_t count,
    bool partner_return_unresolved) {
  DirectResidentLanguageDecision out{};
  out.discourse = DirectDiscourseAction::remain_silent;
  if (state == nullptr || records == nullptr || count == 0u) return out;

  std::uint32_t newest_tick = 0u;
  bool have_sensory = false;
  for (std::uint32_t i = 0u; i < count; ++i)
    if (records[i].kind == DirectExactHistoryKind::sensory_contact) {
      newest_tick = records[i].resident_tick > newest_tick
                        ? records[i].resident_tick
                        : newest_tick;
      have_sensory = true;
    }
  if (!have_sensory) return out;

  std::uint32_t expression_begin = count;
  for (std::uint32_t i = 0u; i < count; ++i) {
    const auto& record = records[i];
    if (record.kind == DirectExactHistoryKind::sensory_contact &&
        newest_tick >= record.resident_tick &&
        newest_tick - record.resident_tick <= kGroundingCoherenceWindowTicks) {
      expression_begin = i;
      break;
    }
  }

  std::uint32_t first = count, second = count;
  for (std::uint32_t cursor = count; cursor > 0u; --cursor) {
    const std::uint32_t i = cursor - 1u;
    if (records[i].kind != DirectExactHistoryKind::sensory_contact) continue;
    if (second == count) {
      second = i;
      continue;
    }
    if (records[i].subject != records[second].subject) {
      first = i;
      break;
    }
  }
  const std::uint32_t temporal_begin = first == count ? 0u : first;
  const std::uint32_t temporal_end = second == count || first == count
                                         ? 0u
                                         : second + 1u;
  return resident_language_decide(
      state, records, temporal_begin, temporal_end, expression_begin, count,
      partner_return_unresolved);
}

__host__ __device__ inline void resident_language_note_spoken(
    DirectResidentLanguageState* state, std::uint64_t ticket_id) {
  if (state == nullptr || ticket_id == 0u) return;
  state->turn.spoken_ticket_id = ticket_id;
}

__device__ inline bool resident_language_reopen_turn(
    DirectResidentLanguageState* state,
    const DirectExactHistoryRecord* records, std::uint32_t count,
    std::uint64_t return_identity) {
  return state != nullptr && discourse_return_reopens(
      &state->turn, records, count, return_identity);
}

}  // namespace substrate::direct_network

#endif  // HARDWARE_NATIVE_DIRECT_ADULT_RESIDENT_LANGUAGE_STATE_CUH
