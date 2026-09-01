#pragma once

#include "bcc32_resident_response_inhibition.cuh"

#include <cuda_runtime.h>

#include <cstdint>

namespace substrate::bcc32::party_attention {

inline constexpr std::uint32_t kPartyContactCapacity = 8u;

enum class SelectionStatus : std::uint32_t {
  unique = 0u,
  ambiguous = 1u,
  no_admissible_contact = 2u,
  fail_closed = 3u,
};

struct ContactEvidence {
  response_inhibition::Contact contact{};
  // Independent resident evidence axes.  No scalar score or host speaker
  // priority is accepted.  A contact wins only by strict Pareto dominance.
  std::uint32_t resident_relation_q8 = 0u;
  std::uint32_t resident_goal_q8 = 0u;
  std::uint32_t resident_continuity_q8 = 0u;
};

struct Selection {
  SelectionStatus status = SelectionStatus::fail_closed;
  std::uint32_t selected_index = 0xffffffffu;
  std::uint64_t selected_speaker_identity = 0u;
  std::uint32_t nondominated_contacts = 0u;
  response_inhibition::Ticket response_ticket{};
};

struct Axes {
  std::uint32_t body = 0u;
  std::uint32_t unresolved = 0u;
  std::uint32_t goal = 0u;
  std::uint32_t relation = 0u;
  std::uint32_t continuity = 0u;
  std::uint32_t novelty = 0u;
  std::uint32_t inhibition_headroom = 0u;
  bool admissible = false;
};

__host__ __device__ inline bool valid_contact(
    const ContactEvidence& evidence) {
  return evidence.contact.speaker_identity != 0u &&
         evidence.contact.raw_bytes != nullptr &&
         evidence.contact.raw_byte_count != 0u &&
         evidence.contact.ingress_sequence != 0u;
}

__host__ __device__ inline Axes resident_axes(
    const response_inhibition::State& state,
    const ContactEvidence& evidence) {
  Axes axes{};
  if (!valid_contact(evidence)) return axes;
  const std::uint64_t digest = response_inhibition::payload_digest(
      evidence.contact.raw_bytes, evidence.contact.raw_byte_count);
  if (digest == 0u) return axes;
  const std::uint32_t slot = response_inhibition::find_speaker(
      &state, evidence.contact.speaker_identity);
  std::uint32_t inhibition = 0u;
  bool repeated = false;
  if (slot != 0xffffffffu) {
    const response_inhibition::SpeakerState& speaker = state.speaker[slot];
    if (speaker.pending_decision_sequence != 0u ||
        evidence.contact.ingress_sequence <= speaker.last_ingress_sequence)
      return axes;
    inhibition = response_inhibition::clamp_pressure(
        speaker.learned_inhibition_q8);
    repeated = speaker.last_payload_digest == digest;
  }
  axes.body = response_inhibition::clamp_pressure(
      evidence.contact.resident_body_pressure_q8);
  axes.unresolved = response_inhibition::clamp_pressure(
      evidence.contact.resident_unresolved_pressure_q8);
  axes.goal = response_inhibition::clamp_pressure(evidence.resident_goal_q8);
  axes.relation =
      response_inhibition::clamp_pressure(evidence.resident_relation_q8);
  axes.continuity =
      response_inhibition::clamp_pressure(evidence.resident_continuity_q8);
  axes.novelty = repeated ? 0u : response_inhibition::kPressureMaximum;
  axes.inhibition_headroom =
      response_inhibition::kPressureMaximum - inhibition;
  const bool resident_override =
      axes.body >= response_inhibition::kUrgentBodyPressure ||
      axes.unresolved >= response_inhibition::kClarificationPressure;
  if (resident_override) {
    // Under a resident body/unresolved override, stale repetition is no longer
    // evidence against admission.  This does not choose the contact: every
    // other resident evidence axis still participates in Pareto comparison.
    axes.novelty = response_inhibition::kPressureMaximum;
    axes.inhibition_headroom = response_inhibition::kPressureMaximum;
  }
  axes.admissible = resident_override || !repeated ||
                    inhibition < response_inhibition::kInhibitionThreshold;
  return axes;
}

__host__ __device__ inline bool dominates(const Axes& left,
                                          const Axes& right) {
  if (!left.admissible || !right.admissible) return false;
  const bool no_worse =
      left.body >= right.body && left.unresolved >= right.unresolved &&
      left.goal >= right.goal && left.relation >= right.relation &&
      left.continuity >= right.continuity && left.novelty >= right.novelty &&
      left.inhibition_headroom >= right.inhibition_headroom;
  const bool strictly_better =
      left.body > right.body || left.unresolved > right.unresolved ||
      left.goal > right.goal || left.relation > right.relation ||
      left.continuity > right.continuity || left.novelty > right.novelty ||
      left.inhibition_headroom > right.inhibition_headroom;
  return no_worse && strictly_better;
}

// The input array is a membrane snapshot, not a queue priority.  Reordering
// contacts cannot change the selected speaker.  If two resident evidence
// profiles are incomparable, the resident remains unresolved and emits no
// response ticket.
__host__ __device__ inline Selection select_contact(
    response_inhibition::State* state, const ContactEvidence* evidence,
    std::uint32_t count) {
  Selection selection{};
  if (state == nullptr || evidence == nullptr || count == 0u ||
      count > kPartyContactCapacity) {
    return selection;
  }

  Axes axes[kPartyContactCapacity]{};
  std::uint32_t admissible = 0u;
  for (std::uint32_t index = 0u; index < count; ++index) {
    axes[index] = resident_axes(*state, evidence[index]);
    admissible += axes[index].admissible ? 1u : 0u;
  }
  if (admissible == 0u) {
    selection.status = SelectionStatus::no_admissible_contact;
    return selection;
  }

  std::uint32_t winner = 0xffffffffu;
  for (std::uint32_t candidate = 0u; candidate < count; ++candidate) {
    if (!axes[candidate].admissible) continue;
    bool dominated = false;
    for (std::uint32_t other = 0u; other < count; ++other) {
      if (candidate != other && dominates(axes[other], axes[candidate])) {
        dominated = true;
        break;
      }
    }
    if (dominated) continue;
    ++selection.nondominated_contacts;
    winner = candidate;
  }
  if (selection.nondominated_contacts != 1u) {
    selection.status = SelectionStatus::ambiguous;
    return selection;
  }

  selection.response_ticket =
      response_inhibition::observe_and_select(state, evidence[winner].contact);
  if (!selection.response_ticket.valid()) {
    selection.status = SelectionStatus::fail_closed;
    selection.nondominated_contacts = 0u;
    return selection;
  }
  selection.status = SelectionStatus::unique;
  selection.selected_index = winner;
  selection.selected_speaker_identity =
      evidence[winner].contact.speaker_identity;
  return selection;
}

}  // namespace substrate::bcc32::party_attention
