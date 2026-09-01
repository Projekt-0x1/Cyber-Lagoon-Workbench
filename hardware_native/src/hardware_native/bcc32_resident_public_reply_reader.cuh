#pragma once

#include "bcc32_resident_party_queue.cuh"
#include "bcc32_resident_source_calibration.cuh"

#include <cuda_runtime.h>

#include <cstdint>
#include <type_traits>

namespace substrate::bcc32::public_reply_reader {

namespace party = party_queue;
namespace calibration = source_calibration;

inline constexpr std::uint64_t kMagic = 0x5245444145525052ull;
inline constexpr std::uint32_t kVersion = 1u;
inline constexpr std::uint32_t kReplySurfaceCapacity = 512u;
inline constexpr std::uint32_t kRecipientCapacity = 8u;
inline constexpr std::uint32_t kSourceRefuseContradictionQ8 = 64u;

enum class Disclosure : std::uint32_t {
  none = 0u,
  bounded = 1u,
  full = 2u,
};

enum class RecipientIntent : std::uint32_t {
  grant = 0u,
  correct = 1u,
  withdraw = 2u,
  fail_closed = 3u,
};

enum class ReadStatus : std::uint32_t {
  ready = 0u,
  stale_authorization = 1u,
  invalid_surface = 2u,
  fail_closed = 3u,
};

struct ResidentContactReceipt {
  std::uint64_t speaker_identity = 0u;
  std::uint64_t source_identity = 0u;
  std::uint64_t transport_sequence = 0u;
  std::uint64_t ingress_receipt_digest = 0u;
  std::uint64_t payload_digest = 0u;
  std::uint64_t decision_sequence = 0u;
  std::uint64_t response_reafference_digest = 0u;
  std::uint64_t source_reafference_digest = 0u;
  std::uint32_t revision = 0u;

  __host__ __device__ bool valid() const {
    return speaker_identity != 0u && source_identity != 0u &&
           transport_sequence != 0u && ingress_receipt_digest != 0u &&
           payload_digest != 0u && decision_sequence != 0u &&
           response_reafference_digest != 0u &&
           source_reafference_digest != 0u && revision != 0u;
  }
};

struct RecipientSignal {
  ResidentContactReceipt contact{};
  RecipientIntent intent = RecipientIntent::fail_closed;
  Disclosure ceiling = Disclosure::none;
  std::uint64_t referent_digest = 0u;
  std::uint64_t resident_signal_digest = 0u;
};

struct RecipientState {
  std::uint64_t speaker_identity = 0u;
  std::uint64_t source_identity = 0u;
  std::uint64_t last_transport_sequence = 0u;
  std::uint64_t consent_ingress_receipt_digest = 0u;
  std::uint64_t referent_digest = 0u;
  std::uint64_t resident_signal_digest = 0u;
  std::uint32_t revision = 0u;
  Disclosure ceiling = Disclosure::none;
  bool granted = false;
};

struct RecipientRef {
  std::uint32_t slot = 0xffffffffu;
  std::uint32_t revision = 0u;
  std::uint64_t speaker_identity = 0u;
  std::uint64_t source_identity = 0u;
  std::uint64_t consent_ingress_receipt_digest = 0u;
  std::uint64_t referent_digest = 0u;
  Disclosure ceiling = Disclosure::none;

  __host__ __device__ bool valid() const {
    return slot < kRecipientCapacity && revision != 0u &&
           speaker_identity != 0u && source_identity != 0u &&
           consent_ingress_receipt_digest != 0u && referent_digest != 0u &&
           ceiling != Disclosure::none;
  }
};

struct ActionTicket {
  party::QueueTicket output{};
  std::uint64_t action_sequence = 0u;
  std::uint64_t public_sequence = 0u;
  std::uint64_t producer_source_identity = 0u;
  std::uint64_t authorization_digest = 0u;
  std::uint32_t authorization_revision = 0u;
  std::uint32_t recipient_count = 0u;
  RecipientRef recipient[kRecipientCapacity]{};

  __host__ __device__ bool valid() const {
    if (!output.valid() || action_sequence == 0u || public_sequence == 0u ||
        producer_source_identity == 0u || authorization_digest == 0u ||
        authorization_revision == 0u || recipient_count == 0u ||
        recipient_count > kRecipientCapacity)
      return false;
    for (std::uint32_t index = 0u; index < recipient_count; ++index)
      if (!recipient[index].valid()) return false;
    return true;
  }
};

struct ActionState {
  std::uint64_t action_sequence = 0u;
  std::uint64_t public_sequence = 0u;
  std::uint64_t output_ingress_receipt_digest = 0u;
  std::uint64_t output_decision_sequence = 0u;
  std::uint64_t producer_speaker_identity = 0u;
  std::uint64_t producer_source_identity = 0u;
  std::uint64_t authorization_digest = 0u;
  std::uint32_t revision = 0u;
  std::uint32_t recipient_count = 0u;
  RecipientRef recipient[kRecipientCapacity]{};
  bool active = false;
};

struct State {
  std::uint64_t magic = kMagic;
  std::uint32_t version = kVersion;
  std::uint32_t revision = 1u;
  std::uint64_t next_action_sequence = 1u;
  std::uint64_t next_public_sequence = 1u;
  RecipientState recipient[kRecipientCapacity]{};
  ActionState action{};
  std::uint64_t recipient_mutations = 0u;
  std::uint64_t actions_authorized = 0u;
  std::uint64_t actions_invalidated = 0u;
};

struct ResidentSurface {
  const std::uint8_t* raw_bytes = nullptr;
  std::uint32_t raw_byte_count = 0u;
  std::uint64_t resident_surface_digest = 0u;
};

struct Recipient {
  std::uint64_t speaker_identity = 0u;
  std::uint64_t source_identity = 0u;
  std::uint64_t consent_ingress_receipt_digest = 0u;
  std::uint64_t referent_digest = 0u;
  Disclosure disclosure = Disclosure::none;
};

struct PublicReply {
  ReadStatus status = ReadStatus::fail_closed;
  std::uint64_t action_sequence = 0u;
  std::uint64_t public_sequence = 0u;
  std::uint64_t producer_speaker_identity = 0u;
  std::uint64_t producer_source_identity = 0u;
  std::uint64_t output_ingress_receipt_digest = 0u;
  std::uint64_t output_decision_sequence = 0u;
  std::uint64_t resident_surface_digest = 0u;
  std::uint32_t authorization_revision = 0u;
  std::uint32_t raw_byte_count = 0u;
  std::uint32_t recipient_count = 0u;
  Recipient recipient[kRecipientCapacity]{};
  std::uint8_t raw_bytes[kReplySurfaceCapacity]{};

  __host__ __device__ bool ready() const {
    return status == ReadStatus::ready && action_sequence != 0u &&
           public_sequence != 0u && producer_speaker_identity != 0u &&
           producer_source_identity != 0u &&
           output_ingress_receipt_digest != 0u &&
           output_decision_sequence != 0u && resident_surface_digest != 0u &&
           authorization_revision != 0u && raw_byte_count != 0u &&
           recipient_count != 0u && recipient_count <= kRecipientCapacity;
  }
};

static_assert(std::is_trivially_copyable_v<ResidentContactReceipt>);
static_assert(std::is_trivially_copyable_v<RecipientSignal>);
static_assert(std::is_trivially_copyable_v<RecipientState>);
static_assert(std::is_trivially_copyable_v<RecipientRef>);
static_assert(std::is_trivially_copyable_v<ActionTicket>);
static_assert(std::is_trivially_copyable_v<ActionState>);
static_assert(std::is_trivially_copyable_v<State>);
static_assert(std::is_trivially_copyable_v<PublicReply>);

__host__ __device__ inline bool exact_live_output(
    const party::State& party_state, const party::QueueTicket& output) {
  if (!output.valid() || output.entry.slot >= party::kQueueCapacity ||
      output.response.speaker_slot >= response_inhibition::kSpeakerCapacity)
    return false;
  const party::Entry& entry = party_state.entry[output.entry.slot];
  const response_inhibition::SpeakerState& speaker =
      party_state.inhibition.speaker[output.response.speaker_slot];
  return entry.occupied() && entry.revision == output.entry.revision &&
         entry.ingress_receipt_digest ==
             output.entry.ingress_receipt_digest &&
         entry.reserved_decision_sequence ==
             output.response.decision_sequence &&
         entry.speaker_identity == output.response.speaker_identity &&
         speaker.identity == output.response.speaker_identity &&
         speaker.pending_decision_sequence ==
             output.response.decision_sequence &&
         speaker.pending_payload_digest == output.response.payload_digest &&
         speaker.pending_disposition == output.response.disposition &&
         speaker.revision == output.response.speaker_revision;
}

__host__ __device__ inline std::uint32_t source_slot(
    const calibration::State& calibration_state,
    std::uint64_t source_identity) {
  return calibration::find_source(&calibration_state, source_identity);
}

__host__ __device__ inline bool source_authorized(
    const calibration::State& calibration_state,
    std::uint64_t source_identity, Disclosure disclosure) {
  const std::uint32_t slot = source_slot(calibration_state, source_identity);
  if (slot == 0xffffffffu) return false;
  const calibration::SourceState& source = calibration_state.source[slot];
  if (source.pending_decision_sequence != 0u ||
      source.settled_observations == 0u ||
      (source.contradiction_q8 >= kSourceRefuseContradictionQ8 &&
       source.contradiction_q8 > source.support_q8))
    return false;
  if (disclosure == Disclosure::full)
    return source.support_q8 > source.contradiction_q8;
  return disclosure == Disclosure::bounded;
}

__host__ __device__ inline std::uint32_t find_recipient(
    const State& state, std::uint64_t speaker_identity,
    std::uint64_t source_identity) {
  for (std::uint32_t slot = 0u; slot < kRecipientCapacity; ++slot)
    if (state.recipient[slot].speaker_identity == speaker_identity &&
        state.recipient[slot].source_identity == source_identity)
      return slot;
  return 0xffffffffu;
}

__host__ __device__ inline std::uint32_t find_or_allocate_recipient(
    State* state, std::uint64_t speaker_identity,
    std::uint64_t source_identity) {
  const std::uint32_t found =
      find_recipient(*state, speaker_identity, source_identity);
  if (found != 0xffffffffu) return found;
  for (std::uint32_t slot = 0u; slot < kRecipientCapacity; ++slot) {
    if (state->recipient[slot].speaker_identity != 0u) continue;
    state->recipient[slot].speaker_identity = speaker_identity;
    state->recipient[slot].source_identity = source_identity;
    state->recipient[slot].revision = 1u;
    return slot;
  }
  return 0xffffffffu;
}

__host__ __device__ inline RecipientRef current_recipient_ref(
    const State& state, std::uint32_t slot) {
  RecipientRef result{};
  if (slot >= kRecipientCapacity) return result;
  const RecipientState& recipient = state.recipient[slot];
  if (!recipient.granted || recipient.ceiling == Disclosure::none ||
      recipient.referent_digest == 0u) return result;
  result.slot = slot;
  result.revision = recipient.revision;
  result.speaker_identity = recipient.speaker_identity;
  result.source_identity = recipient.source_identity;
  result.consent_ingress_receipt_digest =
      recipient.consent_ingress_receipt_digest;
  result.referent_digest = recipient.referent_digest;
  result.ceiling = recipient.ceiling;
  return result;
}

__host__ __device__ inline bool exact_recipient_ref(
    const State& state, const RecipientRef& ref) {
  if (!ref.valid()) return false;
  const RecipientState& recipient = state.recipient[ref.slot];
  return recipient.revision == ref.revision && recipient.granted &&
         recipient.speaker_identity == ref.speaker_identity &&
         recipient.source_identity == ref.source_identity &&
         recipient.consent_ingress_receipt_digest ==
             ref.consent_ingress_receipt_digest &&
         recipient.referent_digest == ref.referent_digest &&
         recipient.ceiling == ref.ceiling;
}

__host__ __device__ inline void invalidate_active(State* state) {
  if (state->action.active) ++state->actions_invalidated;
  state->action.active = false;
  ++state->action.revision;
  if (state->action.revision == 0u) state->action.revision = 1u;
  ++state->revision;
  if (state->revision == 0u) state->revision = 1u;
}

__host__ __device__ inline RecipientRef apply_recipient_signal(
    State* state, const RecipientSignal& signal) {
  RecipientRef result{};
  if (state == nullptr || state->magic != kMagic ||
      state->version != kVersion || !signal.contact.valid() ||
      signal.intent == RecipientIntent::fail_closed ||
      signal.resident_signal_digest == 0u ||
      signal.resident_signal_digest != signal.contact.payload_digest ||
      ((signal.intent == RecipientIntent::grant ||
        signal.intent == RecipientIntent::correct) &&
       (signal.ceiling == Disclosure::none || signal.referent_digest == 0u)))
    return result;
  State next = *state;
  const std::uint32_t slot = find_or_allocate_recipient(
      &next, signal.contact.speaker_identity, signal.contact.source_identity);
  if (slot == 0xffffffffu) return result;
  RecipientState& recipient = next.recipient[slot];
  if (signal.contact.transport_sequence <= recipient.last_transport_sequence ||
      (signal.intent == RecipientIntent::correct &&
       (!recipient.granted || recipient.referent_digest == 0u ||
        recipient.referent_digest == signal.referent_digest)))
    return result;
  invalidate_active(&next);
  recipient.last_transport_sequence = signal.contact.transport_sequence;
  recipient.consent_ingress_receipt_digest =
      signal.contact.ingress_receipt_digest;
  recipient.resident_signal_digest = signal.resident_signal_digest;
  if (signal.intent == RecipientIntent::withdraw) {
    recipient.granted = false;
    recipient.ceiling = Disclosure::none;
  } else {
    recipient.granted = true;
    recipient.ceiling = signal.ceiling;
    recipient.referent_digest = signal.referent_digest;
  }
  ++recipient.revision;
  if (recipient.revision == 0u) recipient.revision = 1u;
  ++next.recipient_mutations;
  result = current_recipient_ref(next, slot);
  if (signal.intent != RecipientIntent::withdraw && !result.valid()) return {};
  *state = next;
  return result;
}

__host__ __device__ inline ActionTicket authorize(
    State* state, const party::State& party_state,
    const calibration::State& calibration_state,
    const party::QueueTicket& output, std::uint64_t producer_source_identity,
    const RecipientRef* recipients, std::uint32_t recipient_count,
    std::uint64_t resident_authorization_digest) {
  ActionTicket ticket{};
  if (state == nullptr || recipients == nullptr || recipient_count == 0u ||
      recipient_count > kRecipientCapacity ||
      resident_authorization_digest == 0u || state->action.active ||
      !exact_live_output(party_state, output) ||
      !source_authorized(calibration_state, producer_source_identity,
                         Disclosure::bounded))
    return ticket;
  for (std::uint32_t index = 0u; index < recipient_count; ++index) {
    if (!exact_recipient_ref(*state, recipients[index]) ||
        !source_authorized(calibration_state,
                           recipients[index].source_identity,
                           recipients[index].ceiling))
      return ticket;
    for (std::uint32_t prior = 0u; prior < index; ++prior)
      if (recipients[prior].slot == recipients[index].slot) return ticket;
  }
  ActionState next{};
  next.action_sequence = state->next_action_sequence++;
  next.public_sequence = state->next_public_sequence++;
  if (state->next_action_sequence == 0u) state->next_action_sequence = 1u;
  if (state->next_public_sequence == 0u) state->next_public_sequence = 1u;
  next.output_ingress_receipt_digest = output.entry.ingress_receipt_digest;
  next.output_decision_sequence = output.response.decision_sequence;
  next.producer_speaker_identity = output.response.speaker_identity;
  next.producer_source_identity = producer_source_identity;
  next.authorization_digest = resident_authorization_digest;
  next.revision = state->action.revision + 1u;
  if (next.revision == 0u) next.revision = 1u;
  next.recipient_count = recipient_count;
  next.active = true;
  for (std::uint32_t index = 0u; index < recipient_count; ++index)
    next.recipient[index] = recipients[index];
  state->action = next;
  ++state->actions_authorized;
  ++state->revision;
  if (state->revision == 0u) state->revision = 1u;
  ticket.output = output;
  ticket.action_sequence = next.action_sequence;
  ticket.public_sequence = next.public_sequence;
  ticket.producer_source_identity = next.producer_source_identity;
  ticket.authorization_digest = next.authorization_digest;
  ticket.authorization_revision = next.revision;
  ticket.recipient_count = recipient_count;
  for (std::uint32_t index = 0u; index < recipient_count; ++index)
    ticket.recipient[index] = recipients[index];
  return ticket;
}

__host__ __device__ inline bool exact_action(
    const State& state, const party::State& party_state,
    const calibration::State& calibration_state,
    const ActionTicket& ticket) {
  if (!ticket.valid() || !state.action.active ||
      !exact_live_output(party_state, ticket.output) ||
      state.action.action_sequence != ticket.action_sequence ||
      state.action.public_sequence != ticket.public_sequence ||
      state.action.output_ingress_receipt_digest !=
          ticket.output.entry.ingress_receipt_digest ||
      state.action.output_decision_sequence !=
          ticket.output.response.decision_sequence ||
      state.action.producer_speaker_identity !=
          ticket.output.response.speaker_identity ||
      state.action.producer_source_identity !=
          ticket.producer_source_identity ||
      state.action.authorization_digest != ticket.authorization_digest ||
      state.action.revision != ticket.authorization_revision ||
      state.action.recipient_count != ticket.recipient_count ||
      !source_authorized(calibration_state,
                         ticket.producer_source_identity,
                         Disclosure::bounded))
    return false;
  for (std::uint32_t index = 0u; index < ticket.recipient_count; ++index)
    if (!exact_recipient_ref(state, ticket.recipient[index]) ||
        !source_authorized(calibration_state,
                           ticket.recipient[index].source_identity,
                           ticket.recipient[index].ceiling) ||
        state.action.recipient[index].revision !=
            ticket.recipient[index].revision ||
        state.action.recipient[index].referent_digest !=
            ticket.recipient[index].referent_digest)
      return false;
  return true;
}

__host__ __device__ inline PublicReply read(
    const State& state, const party::State& party_state,
    const calibration::State& calibration_state, const ActionTicket& ticket,
    const ResidentSurface& surface) {
  PublicReply result{};
  if (!exact_action(state, party_state, calibration_state, ticket)) {
    result.status = ReadStatus::stale_authorization;
    return result;
  }
  if (surface.raw_bytes == nullptr || surface.raw_byte_count == 0u ||
      surface.raw_byte_count > kReplySurfaceCapacity ||
      surface.resident_surface_digest == 0u ||
      surface.resident_surface_digest != ticket.output.response.payload_digest ||
      response_inhibition::payload_digest(surface.raw_bytes,
                                          surface.raw_byte_count) !=
          surface.resident_surface_digest) {
    result.status = ReadStatus::invalid_surface;
    return result;
  }
  result.status = ReadStatus::ready;
  result.action_sequence = ticket.action_sequence;
  result.public_sequence = ticket.public_sequence;
  result.producer_speaker_identity = ticket.output.response.speaker_identity;
  result.producer_source_identity = ticket.producer_source_identity;
  result.output_ingress_receipt_digest =
      ticket.output.entry.ingress_receipt_digest;
  result.output_decision_sequence = ticket.output.response.decision_sequence;
  result.resident_surface_digest = surface.resident_surface_digest;
  result.authorization_revision = ticket.authorization_revision;
  result.raw_byte_count = surface.raw_byte_count;
  result.recipient_count = ticket.recipient_count;
  for (std::uint32_t index = 0u; index < surface.raw_byte_count; ++index)
    result.raw_bytes[index] = surface.raw_bytes[index];
  for (std::uint32_t index = 0u; index < ticket.recipient_count; ++index) {
    result.recipient[index].speaker_identity =
        ticket.recipient[index].speaker_identity;
    result.recipient[index].source_identity =
        ticket.recipient[index].source_identity;
    result.recipient[index].consent_ingress_receipt_digest =
        ticket.recipient[index].consent_ingress_receipt_digest;
    result.recipient[index].referent_digest =
        ticket.recipient[index].referent_digest;
    result.recipient[index].disclosure = ticket.recipient[index].ceiling;
  }
  return result;
}

__host__ __device__ inline bool invalidate(State* state,
                                           const ActionTicket& ticket) {
  if (state == nullptr || !ticket.valid() || !state->action.active ||
      state->action.action_sequence != ticket.action_sequence ||
      state->action.revision != ticket.authorization_revision)
    return false;
  invalidate_active(state);
  return true;
}

}  // namespace substrate::bcc32::public_reply_reader
