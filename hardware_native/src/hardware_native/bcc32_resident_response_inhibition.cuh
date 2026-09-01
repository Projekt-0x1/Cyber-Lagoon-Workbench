#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace substrate::bcc32::response_inhibition {

inline constexpr std::uint32_t kSpeakerCapacity = 8u;
inline constexpr std::uint32_t kPressureMaximum = 255u;
inline constexpr std::uint32_t kInhibitionThreshold = 128u;
inline constexpr std::uint32_t kUrgentBodyPressure = 192u;
inline constexpr std::uint32_t kClarificationPressure = 160u;
inline constexpr std::uint32_t kScarceResourceReserve = 96u;

enum class Disposition : std::uint32_t {
  attend = 0u,
  clarify = 1u,
  defer = 2u,
  ignore = 3u,
  fail_closed = 4u,
};

struct Contact {
  std::uint64_t speaker_identity = 0u;
  const std::uint8_t* raw_bytes = nullptr;
  std::uint32_t raw_byte_count = 0u;
  std::uint64_t ingress_sequence = 0u;
  // These pressures are outputs of resident state/body paths.  They are not
  // host-authored semantic labels for urgency, spam, or the desired action.
  std::uint32_t resident_unresolved_pressure_q8 = 0u;
  std::uint32_t resident_body_pressure_q8 = 0u;
  std::uint32_t resource_reserve_q8 = kPressureMaximum;
};

struct Ticket {
  std::uint64_t speaker_identity = 0u;
  std::uint64_t payload_digest = 0u;
  std::uint64_t decision_sequence = 0u;
  std::uint64_t ingress_sequence = 0u;
  std::uint32_t speaker_slot = 0xffffffffu;
  std::uint32_t speaker_revision = 0u;
  Disposition disposition = Disposition::fail_closed;

  __host__ __device__ bool valid() const {
    return speaker_identity != 0u && payload_digest != 0u &&
           decision_sequence != 0u && speaker_slot < kSpeakerCapacity &&
           disposition != Disposition::fail_closed;
  }
};

struct Consequence {
  // This is accepted only through the matching decision ticket.  The values
  // describe resident assimilation/body deltas; no answer oracle is present.
  std::uint32_t resident_assimilation_delta_q8 = 0u;
  std::uint32_t resident_unresolved_growth_q8 = 0u;
  std::int32_t resource_delta_q8 = 0;
  std::uint64_t reafference_digest = 0u;
};

struct SpeakerState {
  std::uint64_t identity = 0u;
  std::uint64_t last_payload_digest = 0u;
  std::uint64_t last_ingress_sequence = 0u;
  std::uint64_t pending_decision_sequence = 0u;
  std::uint64_t settled_decision_sequence = 0u;
  std::uint32_t repetition_run = 0u;
  std::uint32_t learned_inhibition_q8 = 0u;
  std::uint32_t missed_consequence_q8 = 0u;
  std::uint32_t conserved_resource_q8 = 0u;
  std::uint32_t revision = 0u;
  std::uint64_t pending_payload_digest = 0u;
  Disposition pending_disposition = Disposition::fail_closed;
};

struct Receipt {
  std::uint64_t decisions = 0u;
  std::uint64_t attended = 0u;
  std::uint64_t clarifications = 0u;
  std::uint64_t deferred = 0u;
  std::uint64_t ignored = 0u;
  std::uint64_t settled = 0u;
  std::uint64_t rejected_contacts = 0u;
  std::uint64_t rejected_consequences = 0u;
  std::uint64_t last_speaker_identity = 0u;
  std::uint64_t last_payload_digest = 0u;
  std::uint64_t last_decision_sequence = 0u;
  Disposition last_disposition = Disposition::fail_closed;
};

struct State {
  SpeakerState speaker[kSpeakerCapacity]{};
  std::uint64_t next_decision_sequence = 1u;
  Receipt receipt{};
};

__host__ __device__ inline std::uint64_t mix_commitment_word(
    std::uint64_t digest, std::uint64_t word) {
  digest ^= word + 0x9e3779b97f4a7c15ull + (digest << 6u) +
            (digest >> 2u);
  digest *= 1099511628211ull;
  return digest;
}

// This is an integrity commitment, not physical-source authentication. It
// prevents a returned delta or ticket field from being swapped after the body
// route commits the consequence, while leaving external route authority RED.
__host__ __device__ inline std::uint64_t expected_reafference_digest(
    const Ticket& ticket, const Consequence& consequence) {
  if (!ticket.valid()) return 0u;
  std::uint64_t digest = 1469598103934665603ull;
  digest = mix_commitment_word(digest, ticket.speaker_identity);
  digest = mix_commitment_word(digest, ticket.payload_digest);
  digest = mix_commitment_word(digest, ticket.decision_sequence);
  digest = mix_commitment_word(digest, ticket.ingress_sequence);
  digest = mix_commitment_word(digest, ticket.speaker_slot);
  digest = mix_commitment_word(digest, ticket.speaker_revision);
  digest = mix_commitment_word(
      digest, static_cast<std::uint32_t>(ticket.disposition));
  digest = mix_commitment_word(
      digest, consequence.resident_assimilation_delta_q8);
  digest = mix_commitment_word(
      digest, consequence.resident_unresolved_growth_q8);
  digest = mix_commitment_word(
      digest, static_cast<std::uint32_t>(consequence.resource_delta_q8));
  return digest == 0u ? 1u : digest;
}

__host__ __device__ inline std::uint32_t clamp_pressure(std::uint32_t value) {
  return value > kPressureMaximum ? kPressureMaximum : value;
}

__host__ __device__ inline std::uint32_t saturating_add_pressure(
    std::uint32_t value, std::uint32_t increment) {
  value = clamp_pressure(value);
  increment = clamp_pressure(increment);
  return increment > kPressureMaximum - value ? kPressureMaximum
                                               : value + increment;
}

__host__ __device__ inline std::uint32_t saturating_sub_pressure(
    std::uint32_t value, std::uint32_t decrement) {
  value = clamp_pressure(value);
  return decrement >= value ? 0u : value - decrement;
}

__host__ __device__ inline std::uint64_t payload_digest(
    const std::uint8_t* bytes, std::uint32_t count) {
  if (bytes == nullptr || count == 0u) return 0u;
  std::uint64_t digest = 1469598103934665603ull;
  for (std::uint32_t index = 0u; index < count; ++index) {
    digest ^= static_cast<std::uint64_t>(bytes[index]);
    digest *= 1099511628211ull;
  }
  digest ^= static_cast<std::uint64_t>(count) * 0x9e3779b97f4a7c15ull;
  return digest == 0u ? 1u : digest;
}

__host__ __device__ inline std::uint32_t find_speaker(
    const State* state, std::uint64_t identity) {
  if (state == nullptr || identity == 0u) return 0xffffffffu;
  for (std::uint32_t slot = 0u; slot < kSpeakerCapacity; ++slot)
    if (state->speaker[slot].identity == identity) return slot;
  return 0xffffffffu;
}

__host__ __device__ inline std::uint32_t find_or_allocate_speaker(
    State* state, std::uint64_t identity) {
  const std::uint32_t existing = find_speaker(state, identity);
  if (existing != 0xffffffffu) return existing;
  if (state == nullptr || identity == 0u) return 0xffffffffu;
  for (std::uint32_t slot = 0u; slot < kSpeakerCapacity; ++slot) {
    if (state->speaker[slot].identity != 0u) continue;
    state->speaker[slot].identity = identity;
    state->speaker[slot].revision = 1u;
    return slot;
  }
  return 0xffffffffu;
}

__host__ __device__ inline void count_disposition(Receipt* receipt,
                                                  Disposition disposition) {
  if (receipt == nullptr) return;
  switch (disposition) {
    case Disposition::attend:
      ++receipt->attended;
      break;
    case Disposition::clarify:
      ++receipt->clarifications;
      break;
    case Disposition::defer:
      ++receipt->deferred;
      break;
    case Disposition::ignore:
      ++receipt->ignored;
      break;
    case Disposition::fail_closed:
      break;
  }
}

// Selects only whether to engage the contact.  It neither supplies an answer
// nor chooses an audience.  Exact repeated raw contact creates local redundancy
// pressure; later ticket-bound consequences can reverse that pressure.
__host__ __device__ inline Ticket observe_and_select(State* state,
                                                     const Contact& contact) {
  Ticket ticket{};
  if (state == nullptr || contact.speaker_identity == 0u ||
      contact.ingress_sequence == 0u || contact.raw_bytes == nullptr ||
      contact.raw_byte_count == 0u) {
    if (state != nullptr) ++state->receipt.rejected_contacts;
    return ticket;
  }
  const std::uint64_t digest =
      payload_digest(contact.raw_bytes, contact.raw_byte_count);
  const std::uint32_t slot =
      find_or_allocate_speaker(state, contact.speaker_identity);
  if (digest == 0u || slot == 0xffffffffu) {
    ++state->receipt.rejected_contacts;
    return ticket;
  }
  SpeakerState& speaker = state->speaker[slot];
  if (speaker.last_ingress_sequence != 0u &&
      contact.ingress_sequence <= speaker.last_ingress_sequence) {
    ++state->receipt.rejected_contacts;
    return ticket;
  }
  if (speaker.pending_decision_sequence != 0u) {
    ++state->receipt.rejected_contacts;
    return ticket;
  }

  const bool repeated = speaker.last_payload_digest == digest;
  if (repeated) {
    if (speaker.repetition_run != 0xffffffffu) ++speaker.repetition_run;
    speaker.learned_inhibition_q8 =
        saturating_add_pressure(speaker.learned_inhibition_q8, 64u);
  } else {
    speaker.repetition_run = 1u;
    speaker.learned_inhibition_q8 =
        saturating_sub_pressure(speaker.learned_inhibition_q8, 96u);
  }

  const std::uint32_t body_pressure =
      clamp_pressure(contact.resident_body_pressure_q8);
  const std::uint32_t unresolved_pressure =
      clamp_pressure(contact.resident_unresolved_pressure_q8);
  const std::uint32_t reserve = clamp_pressure(contact.resource_reserve_q8);
  Disposition disposition = Disposition::attend;
  if (body_pressure >= kUrgentBodyPressure) {
    disposition = Disposition::attend;
  } else if (unresolved_pressure >= kClarificationPressure) {
    disposition = Disposition::clarify;
  } else if (speaker.repetition_run >= 2u &&
             speaker.learned_inhibition_q8 >= kInhibitionThreshold) {
    disposition = reserve < kScarceResourceReserve ||
                          speaker.repetition_run >= 4u
                      ? Disposition::ignore
                      : Disposition::defer;
  }

  const std::uint64_t decision_sequence = state->next_decision_sequence++;
  speaker.last_payload_digest = digest;
  speaker.last_ingress_sequence = contact.ingress_sequence;
  speaker.pending_decision_sequence = decision_sequence;
  speaker.pending_payload_digest = digest;
  speaker.pending_disposition = disposition;
  ++speaker.revision;

  ++state->receipt.decisions;
  count_disposition(&state->receipt, disposition);
  state->receipt.last_speaker_identity = contact.speaker_identity;
  state->receipt.last_payload_digest = digest;
  state->receipt.last_decision_sequence = decision_sequence;
  state->receipt.last_disposition = disposition;

  ticket.speaker_identity = contact.speaker_identity;
  ticket.payload_digest = digest;
  ticket.decision_sequence = decision_sequence;
  ticket.ingress_sequence = contact.ingress_sequence;
  ticket.speaker_slot = slot;
  ticket.speaker_revision = speaker.revision;
  ticket.disposition = disposition;
  return ticket;
}

__host__ __device__ inline bool settle_consequence(
    State* state, const Ticket& ticket, const Consequence& consequence) {
  if (state == nullptr || !ticket.valid() ||
      consequence.reafference_digest !=
          expected_reafference_digest(ticket, consequence) ||
      ticket.speaker_slot >= kSpeakerCapacity) {
    if (state != nullptr) ++state->receipt.rejected_consequences;
    return false;
  }
  SpeakerState& speaker = state->speaker[ticket.speaker_slot];
  if (speaker.identity != ticket.speaker_identity ||
      speaker.last_ingress_sequence != ticket.ingress_sequence ||
      speaker.pending_decision_sequence != ticket.decision_sequence ||
      speaker.pending_payload_digest != ticket.payload_digest ||
      speaker.pending_disposition != ticket.disposition ||
      speaker.revision != ticket.speaker_revision ||
      speaker.settled_decision_sequence >= ticket.decision_sequence) {
    ++state->receipt.rejected_consequences;
    return false;
  }

  const std::uint32_t assimilation =
      clamp_pressure(consequence.resident_assimilation_delta_q8);
  const std::uint32_t unresolved_growth =
      clamp_pressure(consequence.resident_unresolved_growth_q8);
  const std::uint32_t missed =
      saturating_add_pressure(assimilation, unresolved_growth);
  const bool suppressed = ticket.disposition == Disposition::defer ||
                          ticket.disposition == Disposition::ignore;
  if (suppressed && missed != 0u) {
    speaker.missed_consequence_q8 =
        saturating_add_pressure(speaker.missed_consequence_q8, missed);
    speaker.learned_inhibition_q8 = saturating_sub_pressure(
        speaker.learned_inhibition_q8,
        saturating_add_pressure(96u, missed));
  } else if (suppressed && consequence.resource_delta_q8 >= 0) {
    const std::uint32_t conserved = static_cast<std::uint32_t>(
        consequence.resource_delta_q8 > 255 ? 255
                                             : consequence.resource_delta_q8);
    speaker.conserved_resource_q8 =
        saturating_add_pressure(speaker.conserved_resource_q8, conserved);
    speaker.learned_inhibition_q8 =
        saturating_add_pressure(speaker.learned_inhibition_q8, 32u);
  } else if (!suppressed && assimilation != 0u) {
    speaker.learned_inhibition_q8 =
        saturating_sub_pressure(speaker.learned_inhibition_q8, assimilation);
  } else if (!suppressed && consequence.resource_delta_q8 < 0) {
    const std::int64_t cost = -static_cast<std::int64_t>(
        consequence.resource_delta_q8);
    speaker.learned_inhibition_q8 = saturating_add_pressure(
        speaker.learned_inhibition_q8,
        static_cast<std::uint32_t>(cost > 64 ? 64 : cost));
  }

  speaker.settled_decision_sequence = ticket.decision_sequence;
  speaker.pending_decision_sequence = 0u;
  speaker.pending_payload_digest = 0u;
  speaker.pending_disposition = Disposition::fail_closed;
  ++speaker.revision;
  ++state->receipt.settled;
  return true;
}

}  // namespace substrate::bcc32::response_inhibition
