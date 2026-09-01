#ifndef HARDWARE_NATIVE_DIRECT_ADULT_SURFACE_SEQUENCE_CUH
#define HARDWARE_NATIVE_DIRECT_ADULT_SURFACE_SEQUENCE_CUH

// #1633 learned public-surface repertoire. External raw contact remains opaque:
// a transition exists only after the third same-channel byte actually arrives.
// Planning later walks that resident chronology; it never reads an answer key,
// tokenization, language id, expected output, or previously emitted motor word.

#include <cstdint>
#include <type_traits>

#include "hardware_native/direct_adult_language_expression_motor.cuh"
#include "hardware_native/direct_exact_history.cuh"

namespace substrate::direct_network {

inline constexpr std::uint32_t kSurfaceSequenceEdgeCapacity = 1024u;
inline constexpr std::uint32_t kSurfaceSequenceChannelCapacity = 8u;
inline constexpr std::uint32_t kSurfaceSequencePlanWords = 8u;
inline constexpr std::uint32_t kSurfaceSequencePlanBytes = 4u * kSurfaceSequencePlanWords;
inline constexpr std::uint32_t kSurfaceEcologyUnitCapacity = 8u;
inline constexpr std::uint32_t kSurfaceEcologyMaxLength = 8u;
inline constexpr std::uint32_t kSurfaceEcologyTemporalGap = 2u;
inline constexpr std::uint32_t kSurfaceEcologyCueTemporal = 1u;
inline constexpr std::uint32_t kSurfaceEcologyCueCompanion = 2u;

struct DirectSurfaceSequenceEdge {
  std::uint32_t channel, previous2, previous1, next;
  std::uint32_t support, last_tick, active, reserved;
};
struct DirectSurfaceSequenceContext {
  std::uint32_t channel, previous2, previous1, depth;
};
struct DirectSurfaceEcologyUnit {
  std::uint64_t payload_identity;
  std::uint32_t length;
  std::uint32_t cue_mask;
  std::uint32_t support;
  std::uint32_t active;
};
struct DirectSurfaceEcologyState {
  DirectSurfaceEcologyUnit units[kSurfaceEcologyUnitCapacity];
  std::uint32_t open_values[kSurfaceEcologyMaxLength];
  std::uint32_t open_length;
  std::uint32_t open_channel;
  std::uint32_t last_stream_tick;
  std::uint32_t have_stream;
  std::uint32_t unit_count;
  std::uint32_t closed_count;
  std::uint32_t learned_count;
  std::uint32_t refusals;
  std::uint32_t assimilated_contacts;
  std::uint64_t revision_identity;
  std::uint64_t last_closed_payload_identity;
  std::uint64_t last_closed_content_identity;
  std::uint32_t last_closed_length;
  std::uint32_t last_closed_values[kSurfaceEcologyMaxLength];
};
struct DirectSurfaceSequenceState {
  DirectSurfaceSequenceEdge edges[kSurfaceSequenceEdgeCapacity];
  DirectSurfaceSequenceContext contexts[kSurfaceSequenceChannelCapacity];
  DirectSurfaceEcologyState ecology;
  std::uint32_t edge_count, context_count, cursor, assimilated_contacts;
  std::uint32_t learned_transitions, predictions, refusals, capacity_refusals;
  std::uint64_t revision_identity;
};
struct DirectSurfaceSequencePlan {
  std::uint32_t words[kSurfaceSequencePlanWords];
  std::uint32_t word_count, byte_count, supporting_transitions, admitted;
  std::uint64_t revision_identity;
};

static_assert(std::is_trivially_copyable_v<DirectSurfaceSequenceState>);
static_assert(std::is_trivially_copyable_v<DirectSurfaceSequencePlan>);
static_assert(std::is_trivially_copyable_v<DirectSurfaceEcologyState>);

__host__ __device__ inline std::uint64_t surface_sequence_fold(
    std::uint64_t h, std::uint64_t value) {
  h ^= value + 0x9e3779b97f4a7c15ull + (h << 6u) + (h >> 2u);
  return h == 0u ? 1u : h;
}

__host__ __device__ inline std::uint64_t surface_ecology_payload_identity(
    std::uint32_t channel, std::uint32_t length, const std::uint32_t* values) {
  std::uint64_t identity = surface_sequence_fold(0u, channel);
  identity = surface_sequence_fold(identity, length);
  if (values == nullptr) return identity;
  for (std::uint32_t i = 0u; i < length; ++i)
    identity = surface_sequence_fold(identity, values[i]);
  return identity;
}

__host__ __device__ inline std::uint64_t surface_ecology_content_identity(
    std::uint32_t length, const std::uint32_t* values) {
  std::uint64_t identity = surface_sequence_fold(0u, length);
  if (values == nullptr) return identity;
  for (std::uint32_t i = 0u; i < length; ++i)
    identity = surface_sequence_fold(identity, values[i]);
  return identity;
}

__host__ __device__ inline void surface_ecology_reset_open(
    DirectSurfaceEcologyState* eco) {
  if (eco == nullptr) return;
  eco->open_length = 0u;
  eco->open_channel = 0u;
  eco->last_stream_tick = 0u;
  eco->have_stream = 0u;
}

__device__ inline void surface_ecology_close(
    DirectSurfaceEcologyState* eco, std::uint32_t cues) {
  if (eco == nullptr) return;
  if (eco->open_length < 2u || cues == 0u) {
    if (eco->open_length != 0u) ++eco->refusals;
    surface_ecology_reset_open(eco);
    return;
  }
  const std::uint64_t identity = surface_ecology_payload_identity(
      eco->open_channel, eco->open_length, eco->open_values);
  for (std::uint32_t i = 0u; i < eco->unit_count; ++i) {
    DirectSurfaceEcologyUnit& unit = eco->units[i];
    if (unit.payload_identity != identity || unit.length != eco->open_length)
      continue;
    ++unit.support;
    unit.cue_mask = cues;
    unit.active = 1u;
    ++eco->closed_count;
    eco->last_closed_payload_identity = identity;
    eco->last_closed_content_identity = surface_ecology_content_identity(
        eco->open_length, eco->open_values);
    eco->last_closed_length = eco->open_length;
    for (std::uint32_t j = 0u; j < eco->open_length; ++j)
      eco->last_closed_values[j] = eco->open_values[j];
    eco->revision_identity = surface_sequence_fold(
        surface_sequence_fold(eco->revision_identity, identity), cues);
    surface_ecology_reset_open(eco);
    return;
  }
  if (eco->unit_count >= kSurfaceEcologyUnitCapacity) {
    ++eco->refusals;
    surface_ecology_reset_open(eco);
    return;
  }
  DirectSurfaceEcologyUnit& unit = eco->units[eco->unit_count];
  unit = {identity, eco->open_length, cues, 1u, 1u};
  eco->revision_identity = surface_sequence_fold(
      surface_sequence_fold(eco->revision_identity, identity),
      ++eco->unit_count);
  ++eco->closed_count;
  eco->last_closed_payload_identity = identity;
  eco->last_closed_content_identity = surface_ecology_content_identity(
      eco->open_length, eco->open_values);
  eco->last_closed_length = eco->open_length;
  for (std::uint32_t j = 0u; j < eco->open_length; ++j)
    eco->last_closed_values[j] = eco->open_values[j];
  ++eco->learned_count;
  surface_ecology_reset_open(eco);
}

__device__ inline void surface_ecology_assimilate(
    DirectSurfaceEcologyState* eco, const DirectExactHistoryRecord* records,
    std::uint32_t begin, std::uint32_t count) {
  if (eco == nullptr || records == nullptr || begin > count) return;
  for (std::uint32_t i = begin; i < count; ++i) {
    const auto& record = records[i];
    if (record.kind != DirectExactHistoryKind::sensory_contact) continue;
    ++eco->assimilated_contacts;
    if (eco->open_length != 0u && record.subject != eco->open_channel) {
      surface_ecology_close(eco, kSurfaceEcologyCueCompanion);
      continue;
    }
    if (eco->have_stream != 0u &&
        record.resident_tick >=
            eco->last_stream_tick + kSurfaceEcologyTemporalGap)
      surface_ecology_close(eco, kSurfaceEcologyCueTemporal);
    if (eco->open_length >= kSurfaceEcologyMaxLength) {
      ++eco->refusals;
      surface_ecology_reset_open(eco);
    }
    if (eco->open_length == 0u) eco->open_channel = record.subject;
    eco->open_values[eco->open_length++] = record.value;
    eco->last_stream_tick = record.resident_tick;
    eco->have_stream = 1u;
  }
}

__host__ __device__ inline const DirectSurfaceEcologyUnit* surface_ecology_find(
    const DirectSurfaceEcologyState* eco, std::uint64_t identity,
    std::uint32_t length) {
  if (eco == nullptr) return nullptr;
  for (std::uint32_t i = 0u; i < eco->unit_count; ++i) {
    const DirectSurfaceEcologyUnit& unit = eco->units[i];
    if (unit.active != 0u && unit.payload_identity == identity &&
        unit.length == length)
      return &unit;
  }
  return nullptr;
}

__device__ inline DirectSurfaceSequenceContext* surface_sequence_context(
    DirectSurfaceSequenceState* state, std::uint32_t channel) {
  for (std::uint32_t i = 0u; i < state->context_count; ++i)
    if (state->contexts[i].channel == channel) return &state->contexts[i];
  if (state->context_count >= kSurfaceSequenceChannelCapacity) {
    ++state->capacity_refusals; return nullptr;
  }
  auto& fresh = state->contexts[state->context_count++];
  fresh = {}; fresh.channel = channel; return &fresh;
}

__device__ inline void surface_sequence_note_transition(
    DirectSurfaceSequenceState* state, std::uint32_t channel,
    std::uint32_t previous2, std::uint32_t previous1, std::uint32_t next,
    std::uint32_t tick) {
  for (std::uint32_t i = 0u; i < state->edge_count; ++i) {
    auto& edge = state->edges[i];
    if (edge.active != 0u && edge.channel == channel &&
        edge.previous2 == previous2 && edge.previous1 == previous1 &&
        edge.next == next) {
      ++edge.support; edge.last_tick = tick; ++state->learned_transitions;
      state->revision_identity = surface_sequence_fold(state->revision_identity, i + 1u);
      return;
    }
  }
  if (state->edge_count >= kSurfaceSequenceEdgeCapacity) {
    ++state->capacity_refusals; return;
  }
  auto& edge = state->edges[state->edge_count];
  edge = {channel, previous2, previous1, next, 1u, tick, 1u, 0u};
  state->revision_identity = surface_sequence_fold(state->revision_identity,
                                                    ++state->edge_count);
  ++state->learned_transitions;
}

// Every learned edge is confirmed by the later external sensory record itself.
// Values wider than one raw byte belong to another physical surface and are
// deliberately not reinterpreted as text here.
__device__ inline void surface_sequence_assimilate(
    DirectSurfaceSequenceState* state, const DirectExactHistoryRecord* records,
    std::uint32_t count) {
  if (state == nullptr || records == nullptr || count < state->cursor) return;
  for (std::uint32_t i = state->cursor; i < count; ++i) {
    const auto& record = records[i];
    if (record.kind != DirectExactHistoryKind::sensory_contact || record.value > 0xffu)
      continue;
    auto* context = surface_sequence_context(state, record.subject);
    if (context == nullptr) continue;
    ++state->assimilated_contacts;
    if (context->depth >= 2u)
      surface_sequence_note_transition(state, record.subject, context->previous2,
                                       context->previous1, record.value,
                                       record.resident_tick);
    if (context->depth == 0u) {
      context->previous1 = record.value; context->depth = 1u;
    } else if (context->depth == 1u) {
      context->previous2 = context->previous1; context->previous1 = record.value;
      context->depth = 2u;
    } else {
      context->previous2 = context->previous1; context->previous1 = record.value;
    }
  }
  const std::uint32_t begin = state->cursor;
  state->cursor = count;
  surface_ecology_assimilate(&state->ecology, records, begin, count);
}

__device__ inline const DirectSurfaceSequenceEdge* surface_sequence_choose(
    DirectSurfaceSequenceState* state, std::uint32_t channel,
    std::uint32_t previous2, std::uint32_t previous1) {
  const DirectSurfaceSequenceEdge* best = nullptr;
  bool ambiguous = false;
  for (std::uint32_t i = 0u; i < state->edge_count; ++i) {
    const auto& edge = state->edges[i];
    if (edge.active == 0u || edge.channel != channel ||
        edge.previous2 != previous2 || edge.previous1 != previous1) continue;
    if (best == nullptr || edge.support > best->support) {
      best = &edge; ambiguous = false;
    } else if (edge.support == best->support && edge.next != best->next) {
      ambiguous = true;
    }
  }
  return ambiguous ? nullptr : best;
}

__device__ inline bool surface_sequence_recent_seed(
    const DirectExactHistoryRecord* records, std::uint32_t count,
    std::uint32_t* channel, std::uint32_t* previous2, std::uint32_t* previous1) {
  if (records == nullptr || channel == nullptr || previous2 == nullptr || previous1 == nullptr)
    return false;
  std::uint32_t newest = count, older = count;
  for (std::uint32_t cursor = count; cursor > 0u; --cursor) {
    const std::uint32_t i = cursor - 1u; const auto& record = records[i];
    if (record.kind != DirectExactHistoryKind::sensory_contact || record.value > 0xffu) continue;
    if (newest == count) { newest = i; *channel = record.subject; continue; }
    if (record.subject == *channel) { older = i; break; }
  }
  if (older == count) return false;
  *previous2 = records[older].value; *previous1 = records[newest].value; return true;
}

__host__ __device__ inline bool surface_sequence_utf8_complete(
    const std::uint8_t* bytes, std::uint32_t count) {
  for (std::uint32_t i = 0u; i < count;) {
    const std::uint8_t lead = bytes[i]; std::uint32_t width = 0u;
    if (lead < 0x80u) { ++i; continue; }
    if ((lead & 0xe0u) == 0xc0u) width = 2u;
    else if ((lead & 0xf0u) == 0xe0u) width = 3u;
    else if ((lead & 0xf8u) == 0xf0u) width = 4u;
    else return false;
    if (i + width > count) return false;
    for (std::uint32_t j = 1u; j < width; ++j)
      if ((bytes[i + j] & 0xc0u) != 0x80u) return false;
    i += width;
  }
  return true;
}

__device__ inline DirectSurfaceSequencePlan surface_sequence_plan_recent(
    DirectSurfaceSequenceState* state, const DirectExactHistoryRecord* records,
    std::uint32_t count) {
  DirectSurfaceSequencePlan plan{}; std::uint32_t channel=0u, p2=0u, p1=0u;
  if (state == nullptr || !surface_sequence_recent_seed(records, count, &channel, &p2, &p1)) {
    if (state != nullptr) ++state->refusals; return plan;
  }
  std::uint8_t bytes[kSurfaceSequencePlanBytes]{}; std::uint32_t produced = 0u;
  while (produced < kSurfaceSequencePlanBytes) {
    const auto* edge = surface_sequence_choose(state, channel, p2, p1);
    if (edge == nullptr) break;
    bytes[produced++] = static_cast<std::uint8_t>(edge->next);
    plan.supporting_transitions += edge->support; p2 = p1; p1 = edge->next;
  }
  plan.byte_count = produced - (produced % 4u);
  while (plan.byte_count != 0u && !surface_sequence_utf8_complete(bytes, plan.byte_count))
    plan.byte_count -= 4u;
  plan.word_count = plan.byte_count / 4u;
  for (std::uint32_t w = 0u; w < plan.word_count; ++w)
    plan.words[w] = static_cast<std::uint32_t>(bytes[4u*w]) |
                    (static_cast<std::uint32_t>(bytes[4u*w+1u]) << 8u) |
                    (static_cast<std::uint32_t>(bytes[4u*w+2u]) << 16u) |
                    (static_cast<std::uint32_t>(bytes[4u*w+3u]) << 24u);
  plan.admitted = plan.word_count >= 2u ? 1u : 0u;
  plan.revision_identity = state->revision_identity;
  if (plan.admitted != 0u) ++state->predictions; else ++state->refusals;
  return plan;
}

__device__ inline bool surface_sequence_bind_expression(
    const DirectSurfaceSequencePlan& surface, DirectLanguageMotorPlan* expression) {
  if (expression == nullptr || surface.admitted == 0u || expression->admitted == 0u ||
      expression->step_count == 0u) return false;
  const std::uint32_t count = expression->step_count < surface.word_count
                                  ? expression->step_count : surface.word_count;
  for (std::uint32_t i = 0u; i < count; ++i) expression->steps[i].word = surface.words[i];
  return count != 0u;
}

}  // namespace substrate::direct_network
#endif
