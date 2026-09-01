#ifndef HARDWARE_NATIVE_DIRECT_ADULT_LANGUAGE_TEMPORAL_BINDING_CUH
#define HARDWARE_NATIVE_DIRECT_ADULT_LANGUAGE_TEMPORAL_BINDING_CUH

// f.language_temporal_binding_territory (#1577). Resident device matter binds
// opaque cross-channel contact timing to consequence-backed motor occurrences.
// No token, punctuation, packet, language identity, semantic route, or host
// episode boundary is represented here.

#include <cstdint>
#include <type_traits>

#include "hardware_native/direct_exact_history.cuh"
#include "hardware_native/direct_network_brain.cuh"

namespace substrate::direct_network {

inline constexpr std::uint32_t kLanguageTemporalBindingCapacity = 1024u;
inline constexpr std::uint32_t kLanguageTemporalMotorIdentityCapacity = 256u;
inline constexpr std::uint32_t kLanguageTemporalEligibilityHorizon = 256u;
inline constexpr std::uint32_t kLanguageTemporalMaximumGap = 32u;
inline constexpr std::uint32_t kLanguageTemporalSupportQuorum = 2u;

struct DirectLanguageTemporalBinding {
  std::uint32_t first_channel;
  std::uint32_t first_value;
  std::uint32_t second_channel;
  std::uint32_t second_value;
  std::uint32_t resident_gap;
  std::uint32_t motor_node;
  std::uint32_t motor_channel;
  std::uint32_t motor_value;
  std::uint32_t support;
  std::uint32_t active;
  std::uint64_t matter_identity;
};

struct DirectLanguageTemporalEligibility {
  std::uint64_t first_identity;
  std::uint64_t second_identity;
  std::uint64_t motor_identity;
  std::uint32_t first_channel;
  std::uint32_t first_value;
  std::uint32_t first_tick;
  std::uint32_t second_channel;
  std::uint32_t second_value;
  std::uint32_t second_tick;
  std::uint32_t motor_node;
  std::uint32_t motor_channel;
  std::uint32_t motor_value;
  std::uint32_t motor_tick;
  std::uint32_t expiry_tick;
};

struct DirectLanguageTemporalBindingState {
  DirectLanguageTemporalBinding bindings[kLanguageTemporalBindingCapacity];
  DirectLanguageTemporalEligibility
      pending_motors[kLanguageTemporalMotorIdentityCapacity];
  std::uint32_t binding_count;
  std::uint32_t pending_motor_count;
  std::uint32_t cursor;
  std::uint32_t q_contacts;
  std::uint32_t verified_motor_events;
  std::uint32_t refused_motor_events;
  std::uint32_t ambiguity_refusals;
  std::uint32_t capacity_refusals;
  std::uint32_t revisions;
  std::uint32_t matter;
  std::uint32_t work;
  std::uint32_t lesion_events;
  std::uint32_t remote_sham_matter;
  std::uint32_t reacquired_bindings;
  std::uint64_t source_hash;
  std::uint64_t revision_identity;
};

struct DirectLanguageTemporalPlan {
  std::uint32_t admitted;
  std::uint32_t motor_node;
  std::uint32_t motor_channel;
  std::uint32_t motor_value;
  std::uint32_t due_offset;
  std::uint32_t supporting_bindings;
  std::uint32_t learned_rates;
  std::uint32_t q;
  std::uint32_t N;
  std::uint32_t matter;
  std::uint32_t work;
  std::uint64_t p_next;
};

static_assert(std::is_trivially_copyable_v<DirectLanguageTemporalBindingState>);
static_assert(std::is_trivially_copyable_v<DirectLanguageTemporalPlan>);

__device__ __host__ inline std::uint64_t language_temporal_fold(
    std::uint64_t hash, std::uint64_t value) {
  return (hash ^ (value + 0x9e3779b97f4a7c15ULL + (hash << 6u) +
                  (hash >> 2u))) *
         1099511628211ULL;
}

__device__ inline bool language_temporal_verified_return(
    const DirectExactHistoryRecord* records, std::uint32_t count,
    const DirectExactHistoryRecord& motor) {
  for (std::uint32_t i = 0u; i < count; ++i) {
    const DirectExactHistoryRecord& consequence = records[i];
    if (consequence.kind != DirectExactHistoryKind::world_return ||
        (consequence.flags & kDirectHistoryVerifiedObservation) == 0u)
      continue;
    const bool same_ticket = consequence.identity == motor.identity ||
                             consequence.parent_identity == motor.identity;
    if (same_ticket && consequence.source == motor.source &&
        consequence.subject == motor.subject)
      return true;
  }
  return false;
}

__device__ inline std::int32_t language_temporal_pending_index(
    const DirectLanguageTemporalBindingState& state, std::uint64_t identity) {
  for (std::uint32_t i = 0u; i < state.pending_motor_count; ++i)
    if (state.pending_motors[i].motor_identity == identity)
      return static_cast<std::int32_t>(i);
  return -1;
}

__device__ inline void language_temporal_remove_pending(
    DirectLanguageTemporalBindingState* state, std::uint32_t index) {
  if (state == nullptr || index >= state->pending_motor_count) return;
  state->pending_motors[index] =
      state->pending_motors[--state->pending_motor_count];
}

__device__ inline bool language_temporal_capture_eligibility(
    DirectLanguageTemporalBindingState* state,
    const DirectExactHistoryRecord& first,
    const DirectExactHistoryRecord& second,
    const DirectExactHistoryRecord& motor) {
  if (state == nullptr || motor.identity == 0u ||
      language_temporal_pending_index(*state, motor.identity) >= 0)
    return false;
  if (state->pending_motor_count >= kLanguageTemporalMotorIdentityCapacity) {
    ++state->capacity_refusals;
    return false;
  }
  auto& eligibility = state->pending_motors[state->pending_motor_count++];
  eligibility = {};
  eligibility.first_identity = first.identity;
  eligibility.second_identity = second.identity;
  eligibility.motor_identity = motor.identity;
  eligibility.first_channel = first.subject;
  eligibility.first_value = first.value;
  eligibility.first_tick = first.resident_tick;
  eligibility.second_channel = second.subject;
  eligibility.second_value = second.value;
  eligibility.second_tick = second.resident_tick;
  eligibility.motor_node = motor.source;
  eligibility.motor_channel = motor.subject;
  eligibility.motor_value = motor.value;
  eligibility.motor_tick = motor.resident_tick;
  eligibility.expiry_tick =
      motor.resident_tick > UINT32_MAX - kLanguageTemporalEligibilityHorizon
          ? UINT32_MAX
          : motor.resident_tick + kLanguageTemporalEligibilityHorizon;
  return true;
}

__device__ inline void language_temporal_eligibility_records(
    const DirectLanguageTemporalEligibility& eligibility,
    DirectExactHistoryRecord* first, DirectExactHistoryRecord* second,
    DirectExactHistoryRecord* motor) {
  *first = {};
  first->identity = eligibility.first_identity;
  first->resident_tick = eligibility.first_tick;
  first->kind = DirectExactHistoryKind::sensory_contact;
  first->subject = eligibility.first_channel;
  first->value = eligibility.first_value;
  *second = {};
  second->identity = eligibility.second_identity;
  second->resident_tick = eligibility.second_tick;
  second->kind = DirectExactHistoryKind::sensory_contact;
  second->subject = eligibility.second_channel;
  second->value = eligibility.second_value;
  *motor = {};
  motor->identity = eligibility.motor_identity;
  motor->resident_tick = eligibility.motor_tick;
  motor->kind = DirectExactHistoryKind::motor_output;
  motor->source = eligibility.motor_node;
  motor->subject = eligibility.motor_channel;
  motor->value = eligibility.motor_value;
}

__device__ inline bool language_temporal_preceding_pair(
    const DirectExactHistoryRecord* records, std::uint32_t motor_index,
    DirectExactHistoryRecord* first, DirectExactHistoryRecord* second) {
  bool have_second = false;
  for (std::uint32_t cursor = motor_index; cursor > 0u; --cursor) {
    const DirectExactHistoryRecord& record = records[cursor - 1u];
    if (record.kind != DirectExactHistoryKind::sensory_contact) continue;
    if (!have_second) {
      *second = record;
      have_second = true;
      continue;
    }
    if (record.subject == second->subject) continue;
    if (second->resident_tick < record.resident_tick ||
        second->resident_tick - record.resident_tick >
            kLanguageTemporalMaximumGap)
      return false;
    *first = record;
    return true;
  }
  return false;
}

__device__ inline bool language_temporal_same_binding(
    const DirectLanguageTemporalBinding& binding,
    const DirectExactHistoryRecord& first,
    const DirectExactHistoryRecord& second) {
  return binding.first_channel == first.subject &&
         binding.first_value == first.value &&
         binding.second_channel == second.subject &&
         binding.second_value == second.value &&
         binding.resident_gap == second.resident_tick - first.resident_tick;
}

__device__ inline void language_temporal_admit_motor(
    DirectLanguageTemporalBindingState* state,
    const DirectExactHistoryRecord& first,
    const DirectExactHistoryRecord& second,
    const DirectExactHistoryRecord& motor) {
  for (std::uint32_t i = 0u; i < state->binding_count; ++i) {
    DirectLanguageTemporalBinding& binding = state->bindings[i];
    if (!language_temporal_same_binding(binding, first, second)) continue;
    ++binding.support;
    if (binding.active == 0u) {
      binding.active = 1u;
      ++state->matter;
      ++state->reacquired_bindings;
    }
    ++state->revisions;
    return;
  }
  std::uint32_t binding_index = state->binding_count;
  if (binding_index >= kLanguageTemporalBindingCapacity) {
    binding_index = kLanguageTemporalBindingCapacity;
    for (std::uint32_t i = 0u; i < state->binding_count; ++i)
      if (state->bindings[i].active == 0u) {
        binding_index = i;
        break;
      }
    if (binding_index == kLanguageTemporalBindingCapacity) {
      ++state->capacity_refusals;
      return;
    }
  } else {
    ++state->binding_count;
  }
  DirectLanguageTemporalBinding& binding = state->bindings[binding_index];
  binding = {};
  binding.first_channel = first.subject;
  binding.first_value = first.value;
  binding.second_channel = second.subject;
  binding.second_value = second.value;
  binding.resident_gap = second.resident_tick - first.resident_tick;
  binding.motor_node = motor.source;
  binding.motor_channel = motor.subject;
  binding.motor_value = motor.value;
  binding.support = 1u;
  binding.active = 1u;
  binding.matter_identity = language_temporal_fold(
      language_temporal_fold(motor.identity, first.identity), second.identity);
  ++state->matter;
  ++state->revisions;
}

// A real motor occurrence leaves a bounded sensorimotor eligibility trace.
// The trace is not learning: only a later authenticated return for that exact
// action converts it into durable temporal matter. Archive rotation may reset
// the scan cursor but cannot erase an otherwise-live causal tag.
__device__ inline void language_temporal_assimilate(
    DirectLanguageTemporalBindingState* state,
    const DirectExactHistoryRecord* records, std::uint32_t count) {
  if (state == nullptr || records == nullptr) return;
  if (count < state->cursor) state->cursor = 0u;
  const std::uint32_t begin = state->cursor;
  std::uint32_t newest_tick = 0u;
  for (std::uint32_t i = begin; i < count; ++i) {
    const DirectExactHistoryRecord& record = records[i];
    if (record.resident_tick > newest_tick) newest_tick = record.resident_tick;
    ++state->work;
    if (record.kind == DirectExactHistoryKind::sensory_contact) {
      ++state->q_contacts;
      state->source_hash = language_temporal_fold(
          language_temporal_fold(
              language_temporal_fold(state->source_hash, record.identity),
              record.subject),
          language_temporal_fold(record.value, record.resident_tick));
    } else if (record.kind == DirectExactHistoryKind::motor_output &&
               language_temporal_pending_index(*state, record.identity) < 0) {
      DirectExactHistoryRecord first{};
      DirectExactHistoryRecord second{};
      if (language_temporal_preceding_pair(records, i, &first, &second))
        (void)language_temporal_capture_eligibility(
            state, first, second, record);
      else
        ++state->refused_motor_events;
    }
  }
  state->cursor = count;

  if (newest_tick != 0u) {
    for (std::uint32_t pending = 0u; pending < state->pending_motor_count;) {
      if (state->pending_motors[pending].expiry_tick < newest_tick) {
        language_temporal_remove_pending(state, pending);
        continue;
      }
      ++pending;
    }
  }

  for (std::uint32_t i = begin; i < count; ++i) {
    const DirectExactHistoryRecord& consequence = records[i];
    if (consequence.kind != DirectExactHistoryKind::world_return ||
        (consequence.flags & kDirectHistoryVerifiedObservation) == 0u)
      continue;
    for (std::uint32_t pending = 0u; pending < state->pending_motor_count;
         ++pending) {
      const auto eligibility = state->pending_motors[pending];
      const bool same_ticket =
          consequence.identity == eligibility.motor_identity ||
          consequence.parent_identity == eligibility.motor_identity;
      if (!same_ticket || consequence.source != eligibility.motor_node ||
          consequence.subject != eligibility.motor_channel ||
          consequence.value != eligibility.motor_value)
        continue;
      if (consequence.resident_tick < eligibility.motor_tick ||
          consequence.resident_tick > eligibility.expiry_tick) {
        ++state->refused_motor_events;
        language_temporal_remove_pending(state, pending);
        break;
      }
      DirectExactHistoryRecord first{};
      DirectExactHistoryRecord second{};
      DirectExactHistoryRecord motor{};
      language_temporal_eligibility_records(
          eligibility, &first, &second, &motor);
      ++state->work;
      ++state->verified_motor_events;
      state->revision_identity =
          language_temporal_fold(state->revision_identity, motor.identity);
      language_temporal_admit_motor(state, first, second, motor);
      language_temporal_remove_pending(state, pending);
      break;
    }
  }
}

__device__ inline std::uint32_t language_temporal_active_bindings(
    const DirectLanguageTemporalBindingState& state) {
  std::uint32_t count = 0u;
  for (std::uint32_t i = 0u; i < state.binding_count; ++i)
    count += state.bindings[i].active != 0u ? 1u : 0u;
  return count;
}

__device__ inline std::uint32_t language_temporal_learned_rates(
    const DirectLanguageTemporalBindingState& state) {
  std::uint32_t rates[kLanguageTemporalBindingCapacity]{};
  std::uint32_t count = 0u;
  for (std::uint32_t i = 0u; i < state.binding_count; ++i) {
    const DirectLanguageTemporalBinding& binding = state.bindings[i];
    if (binding.active == 0u ||
        binding.support < kLanguageTemporalSupportQuorum)
      continue;
    bool seen = false;
    for (std::uint32_t j = 0u; j < count; ++j)
      seen |= rates[j] == binding.resident_gap;
    if (!seen) rates[count++] = binding.resident_gap;
  }
  return count;
}

__device__ inline DirectLanguageTemporalPlan language_temporal_plan(
    DirectLanguageTemporalBindingState* state,
    const DirectExactHistoryRecord* records, std::uint32_t begin,
    std::uint32_t count) {
  DirectLanguageTemporalPlan plan{};
  if (state == nullptr || records == nullptr || begin >= count) return plan;
  DirectExactHistoryRecord first{};
  DirectExactHistoryRecord second{};
  bool have_first = false;
  bool have_second = false;
  for (std::uint32_t i = begin; i < count; ++i) {
    if (records[i].kind != DirectExactHistoryKind::sensory_contact) continue;
    if (!have_first) {
      first = records[i];
      have_first = true;
    } else if (records[i].subject != first.subject) {
      second = records[i];
      have_second = true;
    }
  }
  plan.q = state->q_contacts;
  plan.N = language_temporal_active_bindings(*state);
  plan.learned_rates = language_temporal_learned_rates(*state);
  plan.matter = state->matter;
  plan.work = state->work;
  if (!have_first || !have_second || second.resident_tick < first.resident_tick ||
      second.resident_tick - first.resident_tick >
          kLanguageTemporalMaximumGap ||
      plan.learned_rates < 2u)
    return plan;

  const std::uint32_t gap = second.resident_tick - first.resident_tick;
  const DirectLanguageTemporalBinding* best = nullptr;
  bool ambiguous = false;
  for (std::uint32_t i = 0u; i < state->binding_count; ++i) {
    const DirectLanguageTemporalBinding& binding = state->bindings[i];
    if (binding.active == 0u ||
        binding.support < kLanguageTemporalSupportQuorum ||
        binding.first_channel != first.subject ||
        binding.first_value != first.value ||
        binding.second_channel != second.subject ||
        binding.second_value != second.value || binding.resident_gap != gap)
      continue;
    if (best == nullptr || binding.support > best->support) {
      best = &binding;
      ambiguous = false;
    } else if (binding.support == best->support &&
               (binding.motor_node != best->motor_node ||
                binding.motor_channel != best->motor_channel ||
                binding.motor_value != best->motor_value)) {
      ambiguous = true;
    }
  }
  if (best == nullptr || ambiguous) {
    state->ambiguity_refusals += ambiguous ? 1u : 0u;
    return plan;
  }
  plan.admitted = 1u;
  plan.motor_node = best->motor_node;
  plan.motor_channel = best->motor_channel;
  plan.motor_value = best->motor_value;
  plan.due_offset = best->resident_gap;
  plan.supporting_bindings = best->support;
  plan.p_next = language_temporal_fold(
      language_temporal_fold(state->revision_identity, best->matter_identity),
      gap);
  return plan;
}

__device__ inline bool language_temporal_drive(
    const DirectLanguageTemporalPlan& plan, DirectNode* nodes,
    std::uint32_t node_count) {
  if (plan.admitted == 0u || nodes == nullptr ||
      plan.motor_node >= node_count)
    return false;
  atomicAdd(&nodes[plan.motor_node].activation_q16, 1 << 16);
  return true;
}

__device__ inline std::uint32_t language_temporal_focal_lesion(
    DirectLanguageTemporalBindingState* state) {
  if (state == nullptr) return 0u;
  std::uint32_t removed = 0u;
  for (std::uint32_t i = 0u; i < state->binding_count; ++i) {
    if (state->bindings[i].active == 0u) continue;
    state->bindings[i].active = 0u;
    ++removed;
  }
  state->matter -= removed;
  state->lesion_events += removed != 0u ? 1u : 0u;
  return removed;
}

__device__ inline std::uint32_t language_temporal_remote_sham(
    DirectLanguageTemporalBindingState* state, std::uint32_t matter) {
  if (state == nullptr) return 0u;
  state->remote_sham_matter += matter;
  return matter;
}

}  // namespace substrate::direct_network

#endif  // HARDWARE_NATIVE_DIRECT_ADULT_LANGUAGE_TEMPORAL_BINDING_CUH
