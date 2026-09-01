#ifndef HARDWARE_NATIVE_DIRECT_ADULT_LANGUAGE_EXPRESSION_MOTOR_CUH
#define HARDWARE_NATIVE_DIRECT_ADULT_LANGUAGE_EXPRESSION_MOTOR_CUH

// f.language_expression_motor_territory (#1571).  The embodied-symbol tissue
// supplies a consequence-grounded motor root.  This adjacent resident tissue
// learns only opaque, verified motor chronology from Direct exact history and
// reconstructs a timed multi-channel trajectory from that root.  Observer
// meanings, language identities, answer rows, and host-selected routes have no
// representation here.

#include <cstdint>
#include <type_traits>

#include "hardware_native/direct_adult_embodied_symbol_transfer.cuh"

namespace substrate::direct_network {

inline constexpr std::uint32_t kLanguageExpressionTransitionCapacity = 64u;
inline constexpr std::uint32_t kLanguageExpressionPlanStepCapacity = 8u;
inline constexpr std::uint32_t kLanguageExpressionTransitionWindow = 6u;
inline constexpr std::uint32_t kLanguageExpressionMinimumSupport = 2u;

struct DirectLanguageMotorStep {
  std::uint32_t node;
  std::uint32_t channel;
  std::uint32_t word;
  std::uint32_t due_offset;
};

struct DirectLanguageMotorTransition {
  DirectLanguageMotorStep from;
  DirectLanguageMotorStep to;
  std::uint32_t delay;
  std::uint32_t settled_support;
  // Signed lived-consequence mass for this continuation. This is accumulated
  // only from verified world returns attached to the destination motor event;
  // it is not a text score, target label, or prediction.
  std::int64_t consequence_credit_q16;
  std::uint32_t matter_identity;
  std::uint32_t active;
};

struct DirectLanguageExpressionMotorState {
  DirectEmbodiedSymbolTransferState embodied;
  DirectLanguageMotorTransition
      transitions[kLanguageExpressionTransitionCapacity];
  std::uint32_t transition_count;
  std::uint32_t cursor;
  std::uint32_t verified_motor_events;
  std::uint32_t refused_motor_events;
  std::uint32_t ambiguity_refusals;
  std::uint32_t capacity_refusals;
  std::uint32_t lesion_events;
  std::uint32_t remote_sham_matter;
  std::uint32_t reacquired_transitions;
  std::uint32_t has_previous_motor;
  DirectLanguageMotorStep previous_motor;
  std::uint32_t previous_motor_tick;
  std::uint64_t motor_history_hash;
  std::uint64_t revision_identity;
};

struct DirectLanguageMotorPlan {
  DirectLanguageMotorStep steps[kLanguageExpressionPlanStepCapacity];
  std::uint32_t step_count;
  std::uint32_t supporting_transitions;
  std::uint32_t distinct_channels;
  std::uint32_t resident_matter;
  std::uint64_t revision_identity;
  std::uint32_t admitted;
};

static_assert(std::is_trivially_copyable_v<DirectLanguageExpressionMotorState>);
static_assert(std::is_trivially_copyable_v<DirectLanguageMotorPlan>);

__device__ inline std::uint64_t language_expression_fold(
    std::uint64_t h, std::uint64_t value) {
  h ^= value + 0x9e3779b97f4a7c15ULL + (h << 6u) + (h >> 2u);
  return h;
}

__device__ inline const DirectExactHistoryRecord*
language_expression_verified_return(
    const DirectExactHistoryRecord* records, std::uint32_t count,
    const DirectExactHistoryRecord& motor) {
  const DirectExactHistoryRecord* found = nullptr;
  std::uint32_t matches = 0u;
  for (std::uint32_t i = 0u; i < count; ++i) {
    const DirectExactHistoryRecord& consequence = records[i];
    if (consequence.kind != DirectExactHistoryKind::world_return ||
        (consequence.flags & kDirectHistoryVerifiedObservation) == 0u)
      continue;
    const bool same_ticket = consequence.identity == motor.identity ||
                             consequence.parent_identity == motor.identity;
    if (!same_ticket || consequence.source != motor.source ||
        consequence.subject != motor.subject)
      continue;
    found = &consequence;
    ++matches;
  }
  return matches == 1u ? found : nullptr;
}

__device__ inline std::int64_t language_expression_credit_add(
    std::int64_t total, std::int64_t delta) {
  constexpr std::int64_t kMax = 0x7fffffffffffffffLL;
  constexpr std::int64_t kMin = (-0x7fffffffffffffffLL - 1LL);
  if (delta > 0 && total > kMax - delta) return kMax;
  if (delta < 0 && total < kMin - delta) return kMin;
  return total + delta;
}

__device__ inline bool language_expression_same_step(
    const DirectLanguageMotorStep& left,
    const DirectLanguageMotorStep& right) {
  return left.node == right.node && left.channel == right.channel &&
         left.word == right.word;
}

__device__ inline bool language_expression_same_transition(
    const DirectLanguageMotorTransition& transition,
    const DirectLanguageMotorStep& from, const DirectLanguageMotorStep& to,
    std::uint32_t delay) {
  return language_expression_same_step(transition.from, from) &&
         language_expression_same_step(transition.to, to) &&
         transition.delay == delay;
}

__device__ inline void language_expression_note_transition(
    DirectLanguageExpressionMotorState* state,
    const DirectLanguageMotorStep& from, const DirectLanguageMotorStep& to,
    std::uint32_t delay, std::int64_t consequence_credit_q16) {
  for (std::uint32_t i = 0u; i < state->transition_count; ++i) {
    DirectLanguageMotorTransition& transition = state->transitions[i];
    if (!language_expression_same_transition(transition, from, to, delay))
      continue;
    if (transition.active == 0u) {
      transition.active = 1u;
      transition.settled_support = 1u;
      transition.consequence_credit_q16 = consequence_credit_q16;
      ++state->reacquired_transitions;
    } else {
      ++transition.settled_support;
      transition.consequence_credit_q16 = language_expression_credit_add(
          transition.consequence_credit_q16, consequence_credit_q16);
    }
    state->revision_identity = language_expression_fold(
        state->revision_identity, transition.matter_identity);
    return;
  }
  if (state->transition_count >= kLanguageExpressionTransitionCapacity) {
    ++state->capacity_refusals;
    return;
  }

  DirectLanguageMotorTransition& transition =
      state->transitions[state->transition_count];
  transition = {};
  transition.from = from;
  transition.to = to;
  transition.to.due_offset = delay;
  transition.delay = delay;
  transition.settled_support = 1u;
  transition.consequence_credit_q16 = consequence_credit_q16;
  transition.matter_identity = state->transition_count + 1u;
  transition.active = 1u;
  ++state->transition_count;
  state->revision_identity = language_expression_fold(
      state->revision_identity, transition.matter_identity);
}

// Only verified public consequences admit motor events.  Chronologically close
// admitted events grow resident transitions; merely replaying an unreturned
// motor prediction cannot create or reinforce the territory.
__device__ inline void language_expression_assimilate(
    DirectLanguageExpressionMotorState* state,
    const DirectExactHistoryRecord* records, std::uint32_t count) {
  if (state == nullptr || records == nullptr || count < state->cursor) return;
  const std::uint32_t begin = state->cursor;

  embodied_symbol_assimilate(&state->embodied, records, count);
  // A motor becomes learning evidence when its world return arrives, not when
  // the motor was first observed.  Under ordinary Adult ticks those events are
  // often in different assimilation suffixes.  Revisit historical motors in
  // motor chronology, but admit one only when this NEW suffix contains its
  // matching verified return; this processes each consequential motor exactly
  // once without replaying older settled evidence.
  for (std::uint32_t i = 0u; i < count; ++i) {
    const DirectExactHistoryRecord& record = records[i];
    if (record.kind != DirectExactHistoryKind::motor_output) continue;
    const DirectExactHistoryRecord* consequence = nullptr;
    for (std::uint32_t j = begin; j < count; ++j) {
      const DirectExactHistoryRecord& candidate = records[j];
      if (candidate.kind != DirectExactHistoryKind::world_return ||
          (candidate.flags & kDirectHistoryVerifiedObservation) == 0u)
        continue;
      const bool same_ticket = candidate.identity == record.identity ||
                               candidate.parent_identity == record.identity;
      if (!same_ticket || candidate.source != record.source ||
          candidate.subject != record.subject)
        continue;
      if (consequence != nullptr) {
        consequence = nullptr;
        break;
      }
      consequence = &candidate;
    }
    if (consequence == nullptr) {
      if (i >= begin) ++state->refused_motor_events;
      continue;
    }

    DirectLanguageMotorStep current = {record.source, record.subject,
                                       record.value, 0u};
    ++state->verified_motor_events;
    state->motor_history_hash = language_expression_fold(
        language_expression_fold(state->motor_history_hash, record.identity),
        record.resident_tick);
    state->revision_identity = language_expression_fold(
        state->revision_identity, record.identity);

    if (state->has_previous_motor != 0u &&
        record.resident_tick > state->previous_motor_tick) {
      const std::uint32_t delay =
          record.resident_tick - state->previous_motor_tick;
      if (delay <= kLanguageExpressionTransitionWindow)
        language_expression_note_transition(
            state, state->previous_motor, current, delay,
            consequence->resource_delta);
    }
    state->previous_motor = current;
    state->previous_motor_tick = record.resident_tick;
    state->has_previous_motor = 1u;
  }
  state->cursor = count;
}

__device__ inline std::uint32_t language_expression_distinct_channels(
    const DirectLanguageMotorPlan& plan) {
  std::uint32_t count = 0u;
  for (std::uint32_t i = 0u; i < plan.step_count; ++i) {
    bool seen = false;
    for (std::uint32_t j = 0u; j < i; ++j)
      if (plan.steps[j].channel == plan.steps[i].channel) {
        seen = true;
        break;
      }
    if (!seen) ++count;
  }
  return count;
}

__device__ inline const DirectLanguageMotorTransition*
language_expression_choose_transition(
    DirectLanguageExpressionMotorState* state,
    const DirectLanguageMotorStep& from, bool* ambiguous) {
  const DirectLanguageMotorTransition* chosen = nullptr;
  std::uint32_t best_support = 0u;
  std::int64_t best_expected_consequence_q16 = 0;
  bool chosen_self = true;
  *ambiguous = false;
  for (std::uint32_t i = 0u; i < state->transition_count; ++i) {
    const DirectLanguageMotorTransition& candidate = state->transitions[i];
    if (candidate.active == 0u ||
        candidate.settled_support < kLanguageExpressionMinimumSupport ||
        !language_expression_same_step(candidate.from, from))
      continue;
    const bool self = language_expression_same_step(candidate.from, candidate.to);
    const std::int64_t expected_consequence_q16 =
        candidate.consequence_credit_q16 /
        static_cast<std::int64_t>(candidate.settled_support);
    // Outward settled matter outranks a recurrent self-loop. Among comparable
    // outward continuations, lived expected consequence selects behavior;
    // repeated support only breaks an equal-consequence tie.
    const bool better = chosen == nullptr || (chosen_self && !self) ||
        (chosen_self == self &&
         (expected_consequence_q16 > best_expected_consequence_q16 ||
          (expected_consequence_q16 == best_expected_consequence_q16 &&
           candidate.settled_support > best_support)));
    if (better) {
      chosen = &candidate;
      best_support = candidate.settled_support;
      best_expected_consequence_q16 = expected_consequence_q16;
      chosen_self = self;
      *ambiguous = false;
      continue;
    }
    if (chosen_self == self &&
        expected_consequence_q16 == best_expected_consequence_q16 &&
        candidate.settled_support == best_support &&
        (!language_expression_same_step(candidate.to, chosen->to) ||
         candidate.delay != chosen->delay))
      *ambiguous = true;
  }
  // Sequence persistence is consequence-sensitive independently of structural
  // admission: a specifically negative learned continuation terminates further
  // expansion. Unknown/neutral consequence (zero) preserves established behavior.
  if (chosen != nullptr && best_expected_consequence_q16 < 0) return nullptr;
  return chosen;
}


// Planning is output-null: it consults learned resident matter and returns a
// POD trajectory but does not touch motor nodes.  Equal-support incompatible
// continuations fail closed instead of letting storage order choose content.
__device__ inline DirectLanguageMotorPlan language_expression_plan(
    DirectLanguageExpressionMotorState* state,
    const DirectExactHistoryRecord* records, std::uint32_t begin,
    std::uint32_t count) {
  DirectLanguageMotorPlan plan = {};
  if (state == nullptr || records == nullptr || begin > count) return plan;
  const DirectEmbodiedActionReceipt root =
      embodied_symbol_select(state->embodied, records, begin, count);
  if (root.admitted == 0u) return plan;

  DirectLanguageMotorStep current = {root.action_node, 0u, 0u, 0u};
  bool found_root = false;
  for (std::uint32_t i = 0u; i < state->transition_count; ++i) {
    const DirectLanguageMotorTransition& candidate = state->transitions[i];
    if (candidate.active != 0u &&
        candidate.settled_support >= kLanguageExpressionMinimumSupport &&
        candidate.from.node == root.action_node) {
      current = candidate.from;
      found_root = true;
      break;
    }
  }
  if (!found_root) return plan;
  plan.steps[0] = current;
  plan.step_count = 1u;

  while (plan.step_count < kLanguageExpressionPlanStepCapacity) {
    bool ambiguous = false;
    const DirectLanguageMotorTransition* transition =
        language_expression_choose_transition(state, current, &ambiguous);
    if (ambiguous) {
      ++state->ambiguity_refusals;
      return {};
    }
    if (transition == nullptr) break;
    DirectLanguageMotorStep next = transition->to;
    next.due_offset = current.due_offset + transition->delay;
    plan.steps[plan.step_count++] = next;
    ++plan.supporting_transitions;
    ++plan.resident_matter;
    current = next;
  }

  plan.distinct_channels = language_expression_distinct_channels(plan);
  plan.revision_identity = state->revision_identity;
  plan.admitted = plan.supporting_transitions >= 2u &&
                  plan.distinct_channels >= 2u;
  if (plan.admitted == 0u) return {};
  return plan;
}

// Generic pressure enters the ordinary motor-node path only at an exactly
// learned relative time.  Surface serialization remains the membrane's job.
__device__ inline bool language_expression_drive_due(
    const DirectLanguageMotorPlan& plan, std::uint32_t elapsed,
    DirectNode* nodes, std::uint32_t node_count) {
  if (plan.admitted == 0u || nodes == nullptr) return false;
  for (std::uint32_t i = 0u; i < plan.step_count; ++i) {
    const DirectLanguageMotorStep& step = plan.steps[i];
    if (step.due_offset != elapsed || step.node >= node_count) continue;
    atomicAdd(&nodes[step.node].activation_q16,
              static_cast<std::int32_t>(kQ16One));
    atomicAdd(&nodes[step.node].credit_ema_q16,
              static_cast<std::int32_t>(kQ16One / 8u));
    return true;
  }
  return false;
}

__device__ inline std::uint32_t language_expression_focal_lesion(
    DirectLanguageExpressionMotorState* state) {
  if (state == nullptr) return 0u;
  for (std::uint32_t i = 0u; i < state->transition_count; ++i) {
    if (state->transitions[i].active == 0u) continue;
    state->transitions[i].active = 0u;
    state->transitions[i].settled_support = 0u;
    ++state->lesion_events;
    return 1u;
  }
  return 0u;
}

__device__ inline std::uint32_t language_expression_remote_sham(
    DirectLanguageExpressionMotorState* state, std::uint32_t matter) {
  if (state == nullptr) return 0u;
  const std::uint32_t touched =
      matter < state->transition_count ? matter : state->transition_count;
  state->remote_sham_matter += touched;
  return touched;
}

}  // namespace substrate::direct_network

#endif  // HARDWARE_NATIVE_DIRECT_ADULT_LANGUAGE_EXPRESSION_MOTOR_CUH
