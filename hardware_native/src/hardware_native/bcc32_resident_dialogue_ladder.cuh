#pragma once

#include "bcc32_resident_public_reply_reader.cuh"
#include "bcc32_resident_relation_boundary.cuh"

#include <cuda_runtime.h>

#include <cstdint>
#include <type_traits>

namespace substrate::bcc32::dialogue_ladder {

namespace party = party_queue;
namespace reader = public_reply_reader;
namespace calibration = source_calibration;
namespace boundary = relation_boundary;

inline constexpr std::uint64_t kMagic = 0x52454444414c4744ull;
inline constexpr std::uint32_t kVersion = 2u;
inline constexpr std::uint32_t kCandidateCapacity = 4u;

enum class Phase : std::uint32_t {
  empty = 0u,
  ambiguity_open = 1u,
  clarification_authorized = 2u,
  awaiting_resolution = 3u,
  resolution_assimilated = 4u,
  reply_authorized = 5u,
  reply_settled = 6u,
  continuation_authorized = 7u,
  withdrawn = 8u,
};

struct Handle {
  std::uint64_t episode_id = 0u;
  std::uint64_t state_digest = 0u;
  std::uint32_t revision = 0u;

  __host__ __device__ bool valid() const {
    return episode_id != 0u && state_digest != 0u && revision != 0u;
  }
};

struct ContactSettlement {
  party::QueueTicket contact{};
  calibration::Ticket source{};
  response_inhibition::Consequence response_consequence{};
  calibration::Observation source_observation{};
  std::uint64_t ingress_receipt_digest = 0u;
};

struct AmbiguityCandidate {
  party::EntryRef ingress{};
  std::uint64_t payload_digest = 0u;
};

struct AmbiguityEvidence {
  reader::ResidentContactReceipt origin{};
  AmbiguityCandidate candidate[kCandidateCapacity]{};
  std::uint64_t episode_id = 0u;
  std::uint64_t resident_state_digest = 0u;
  std::uint64_t evidence_revision = 0u;
  std::uint32_t candidate_count = 0u;
  std::uint32_t unresolved_pressure_q8 = 0u;
};

struct ContinuityState {
  std::uint64_t action_sequence = 0u;
  std::uint64_t public_sequence = 0u;
  std::uint64_t surface_digest = 0u;
  std::uint64_t producer_speaker_identity = 0u;
  std::uint64_t producer_source_identity = 0u;
  std::uint32_t revision = 0u;
  std::uint32_t raw_byte_count = 0u;
  std::uint32_t recipient_count = 0u;
  reader::Recipient recipient[reader::kRecipientCapacity]{};
  std::uint8_t raw_bytes[reader::kReplySurfaceCapacity]{};
};

struct State {
  std::uint64_t magic = kMagic;
  std::uint32_t version = kVersion;
  Phase phase = Phase::empty;
  std::uint64_t episode_id = 0u;
  std::uint64_t resident_state_digest = 0u;
  std::uint64_t evidence_revision = 0u;
  std::uint64_t candidate_digest[kCandidateCapacity]{};
  std::uint64_t resolved_referent_digest = 0u;
  std::uint64_t pending_action_sequence = 0u;
  std::uint64_t checkpoint_sequence = 0u;
  std::uint64_t checkpoint_digest = 0u;
  std::uint32_t revision = 0u;
  std::uint32_t candidate_count = 0u;
  std::uint32_t unresolved_pressure_q8 = 0u;
  std::uint32_t questions_asked = 0u;
  std::uint32_t resolutions_assimilated = 0u;
  std::uint32_t replies_authorized = 0u;
  std::uint32_t corrections_assimilated = 0u;
  std::uint32_t withdrawals_observed = 0u;
  reader::ResidentContactReceipt origin{};
  reader::RecipientRef recipient{};
  reader::ActionTicket pending_action{};
  boundary::Ticket pending_body{};
  ContinuityState context{};
  calibration::State calibration{};
  boundary::State boundary{};
  reader::State public_reader{};
};

struct RuntimeImage {
  std::uint64_t magic = kMagic;
  std::uint32_t version = kVersion;
  std::uint32_t reserved = 0u;
  std::uint64_t image_sequence = 0u;
  std::uint64_t previous_image_digest = 0u;
  std::uint64_t dialogue_digest = 0u;
  std::uint64_t party_digest = 0u;
  std::uint64_t image_digest = 0u;
  State dialogue{};
  party::State party{};
};

struct RestoreFence {
  std::uint64_t minimum_image_sequence = 0u;
  std::uint64_t expected_previous_image_digest = 0u;
  std::uint64_t expected_episode_id = 0u;
};

static_assert(std::is_trivially_copyable_v<Handle>);
static_assert(std::is_trivially_copyable_v<ContactSettlement>);
static_assert(std::is_trivially_copyable_v<AmbiguityEvidence>);
static_assert(std::is_trivially_copyable_v<ContinuityState>);
static_assert(std::is_trivially_copyable_v<State>);
static_assert(std::is_trivially_copyable_v<RuntimeImage>);

__host__ __device__ inline std::uint64_t mix(std::uint64_t digest,
                                             std::uint64_t value) {
  digest ^= value + 0x9e3779b97f4a7c15ull + (digest << 6u) +
            (digest >> 2u);
  return digest == 0u ? 1u : digest;
}

__host__ __device__ inline std::uint64_t state_digest(const State& state) {
  std::uint64_t digest = mix(kMagic, state.episode_id);
  digest = mix(digest, state.resident_state_digest);
  digest = mix(digest, state.evidence_revision);
  digest = mix(digest, state.resolved_referent_digest);
  digest = mix(digest, state.pending_action_sequence);
  digest = mix(digest, state.checkpoint_sequence);
  digest = mix(digest, state.revision);
  digest = mix(digest, static_cast<std::uint32_t>(state.phase));
  digest = mix(digest, state.context.surface_digest);
  digest = mix(digest, state.context.public_sequence);
  digest = mix(digest, state.context.revision);
  digest = mix(digest, state.public_reader.revision);
  digest = mix(digest, state.public_reader.action.revision);
  digest = mix(digest, state.calibration.next_decision_sequence);
  for (std::uint32_t slot = 0u; slot < calibration::kSourceCapacity; ++slot) {
    digest = mix(digest, state.calibration.source[slot].identity);
    digest = mix(digest, state.calibration.source[slot].revision);
    digest = mix(digest, state.calibration.source[slot].support_q8);
    digest = mix(digest, state.calibration.source[slot].contradiction_q8);
  }
  digest = mix(digest, state.boundary.next_decision_sequence);
  for (std::uint32_t slot = 0u; slot < boundary::kRelationCapacity; ++slot) {
    digest = mix(digest, state.boundary.relation[slot].speaker_identity);
    digest = mix(digest, state.boundary.relation[slot].revision);
    digest = mix(digest,
                 state.boundary.relation[slot].boundary_pressure_q8);
  }
  for (std::uint32_t index = 0u; index < state.candidate_count; ++index)
    digest = mix(digest, state.candidate_digest[index]);
  return digest;
}

__host__ __device__ inline Handle current_handle(const State& state) {
  if (state.magic != kMagic || state.version != kVersion ||
      state.phase == Phase::empty || state.revision == 0u)
    return {};
  return {state.episode_id, state_digest(state), state.revision};
}

__host__ __device__ inline bool exact_handle(const State& state,
                                             const Handle& handle) {
  const Handle current = current_handle(state);
  return handle.valid() && current.episode_id == handle.episode_id &&
         current.state_digest == handle.state_digest &&
         current.revision == handle.revision;
}

__host__ __device__ inline void advance(State* state, Phase phase) {
  state->phase = phase;
  ++state->revision;
  if (state->revision == 0u) state->revision = 1u;
}

// Atomically closes one current-main queued contact through both its selected
// response consequence and its raw source-observation consequence.  The two
// tickets must bind the same raw payload and ingress sequence.
__host__ __device__ inline reader::ResidentContactReceipt settle_contact(
    State* state, party::State* party_state,
    const ContactSettlement& settlement) {
  reader::ResidentContactReceipt receipt{};
  if (state == nullptr || party_state == nullptr ||
      !settlement.contact.valid() || !settlement.source.valid() ||
      settlement.contact.entry.slot >= party::kQueueCapacity ||
      settlement.ingress_receipt_digest == 0u ||
      settlement.contact.response.ingress_sequence !=
          settlement.source.episode_sequence ||
      settlement.contact.entry.ingress_receipt_digest !=
          settlement.ingress_receipt_digest ||
      settlement.response_consequence.reafference_digest == 0u ||
      settlement.source_observation.reafference_digest == 0u)
    return receipt;
  // Party response and source-calibration organs intentionally use distinct
  // digest domains. Bind both tickets to the exact retained raw ingress
  // bytes; comparing the two digest numbers would reject every lawful contact
  // while accepting no additional provenance.
  const party::Entry& ingress =
      party_state->entry[settlement.contact.entry.slot];
  if (settlement.contact.response.payload_digest !=
          response_inhibition::payload_digest(ingress.raw_bytes,
                                               ingress.raw_byte_count) ||
      settlement.source.claim_digest !=
          calibration::digest_bytes(ingress.raw_bytes,
                                    ingress.raw_byte_count))
    return receipt;
  party::State next_party = *party_state;
  calibration::State next_calibration = state->calibration;
  if (!party::settle_selected(&next_party, settlement.contact,
                              settlement.response_consequence) ||
      !calibration::settle_observation(
          &next_calibration, settlement.source,
          settlement.source_observation))
    return receipt;
  *party_state = next_party;
  state->calibration = next_calibration;
  ++state->revision;
  if (state->revision == 0u) state->revision = 1u;
  receipt.speaker_identity = settlement.contact.response.speaker_identity;
  receipt.source_identity = settlement.source.source_identity;
  receipt.transport_sequence = settlement.contact.response.ingress_sequence;
  receipt.ingress_receipt_digest = settlement.ingress_receipt_digest;
  receipt.payload_digest = settlement.contact.response.payload_digest;
  receipt.decision_sequence = settlement.contact.response.decision_sequence;
  receipt.response_reafference_digest =
      settlement.response_consequence.reafference_digest;
  receipt.source_reafference_digest =
      settlement.source_observation.reafference_digest;
  receipt.revision = state->revision;
  return receipt;
}

__host__ __device__ inline bool exact_candidate(
    const party::State& party_state, const AmbiguityCandidate& candidate,
    const reader::ResidentContactReceipt& origin) {
  if (!candidate.ingress.valid() ||
      candidate.ingress.slot >= party::kQueueCapacity ||
      candidate.payload_digest == 0u)
    return false;
  const party::Entry& entry = party_state.entry[candidate.ingress.slot];
  return entry.selectable() && entry.revision == candidate.ingress.revision &&
         entry.ingress_receipt_digest ==
             candidate.ingress.ingress_receipt_digest &&
         entry.speaker_identity == origin.speaker_identity &&
         entry.transport_sequence > origin.transport_sequence &&
         response_inhibition::payload_digest(entry.raw_bytes,
                                             entry.raw_byte_count) ==
             candidate.payload_digest;
}

__host__ __device__ inline Handle open_ambiguity(
    State* state, const party::State* party_state,
    const AmbiguityEvidence& evidence) {
  if (state == nullptr || party_state == nullptr ||
      (state->phase != Phase::empty && state->phase != Phase::withdrawn) ||
      !evidence.origin.valid() || evidence.episode_id == 0u ||
      evidence.resident_state_digest == 0u ||
      evidence.evidence_revision == 0u ||
      evidence.unresolved_pressure_q8 == 0u ||
      evidence.candidate_count < 2u ||
      evidence.candidate_count > kCandidateCapacity)
    return {};
  for (std::uint32_t index = 0u; index < evidence.candidate_count; ++index) {
    if (!exact_candidate(*party_state, evidence.candidate[index],
                         evidence.origin))
      return {};
    for (std::uint32_t prior = 0u; prior < index; ++prior)
      if (evidence.candidate[prior].payload_digest ==
          evidence.candidate[index].payload_digest)
        return {};
  }
  if (state->phase == Phase::withdrawn &&
      (evidence.episode_id == state->episode_id ||
       evidence.evidence_revision <= state->evidence_revision))
    return {};
  state->episode_id = evidence.episode_id;
  state->resident_state_digest = evidence.resident_state_digest;
  state->evidence_revision = evidence.evidence_revision;
  state->origin = evidence.origin;
  state->candidate_count = evidence.candidate_count;
  state->unresolved_pressure_q8 = evidence.unresolved_pressure_q8;
  for (std::uint32_t index = 0u; index < evidence.candidate_count; ++index)
    state->candidate_digest[index] = evidence.candidate[index].payload_digest;
  advance(state, Phase::ambiguity_open);
  return current_handle(*state);
}

__host__ __device__ inline bool candidate_digest(
    const State& state, std::uint64_t payload_digest) {
  for (std::uint32_t index = 0u; index < state.candidate_count; ++index)
    if (state.candidate_digest[index] == payload_digest) return true;
  return false;
}

__host__ __device__ inline reader::RecipientRef apply_recipient_signal(
    State* state, const reader::RecipientSignal& signal) {
  if (state == nullptr || !signal.contact.valid() ||
      signal.contact.speaker_identity != state->origin.speaker_identity ||
      signal.contact.source_identity != state->origin.source_identity)
    return {};
  const reader::RecipientRef recipient =
      reader::apply_recipient_signal(&state->public_reader, signal);
  if (signal.intent != reader::RecipientIntent::withdraw &&
      !recipient.valid())
    return {};
  if (recipient.valid()) state->recipient = recipient;
  const bool cancelled_pending = state->pending_action_sequence != 0u &&
      (signal.intent == reader::RecipientIntent::correct ||
       signal.intent == reader::RecipientIntent::withdraw);
  if (signal.intent == reader::RecipientIntent::correct) {
    ++state->corrections_assimilated;
    state->resolved_referent_digest = signal.referent_digest;
  }
  if (cancelled_pending) {
    ++state->withdrawals_observed;
    advance(state, Phase::withdrawn);
  }
  ++state->revision;
  if (state->revision == 0u) state->revision = 1u;
  return recipient;
}

__host__ __device__ inline reader::ActionTicket authorize_output(
    State* state, const party::State* party_state, const Handle& handle,
    const party::QueueTicket& output, std::uint64_t producer_source_identity,
    std::uint64_t authorization_digest, std::uint64_t body_route_identity,
    Phase authorized_phase) {
  if (state == nullptr || party_state == nullptr ||
      !exact_handle(*state, handle) || !state->recipient.valid() ||
      body_route_identity == 0u)
    return {};
  reader::State next_reader = state->public_reader;
  boundary::State next_boundary = state->boundary;
  if (boundary::disposition_for(&next_boundary,
                                state->origin.speaker_identity) ==
      boundary::Disposition::refuse)
    return {};
  const reader::ActionTicket action = reader::authorize(
      &next_reader, *party_state, state->calibration, output,
      producer_source_identity, &state->recipient, 1u,
      authorization_digest);
  const boundary::Ticket body = boundary::issue_interaction(
      &next_boundary,
      {.speaker_identity = state->origin.speaker_identity,
       .interaction_sequence = action.action_sequence,
       .expected_body_route_identity = body_route_identity});
  if (!action.valid() || !body.valid()) return {};
  state->public_reader = next_reader;
  state->boundary = next_boundary;
  state->pending_action = action;
  state->pending_body = body;
  state->pending_action_sequence = action.action_sequence;
  ++state->replies_authorized;
  if (authorized_phase == Phase::clarification_authorized)
    ++state->questions_asked;
  advance(state, authorized_phase);
  return action;
}

__host__ __device__ inline reader::ActionTicket authorize_clarification(
    State* state, const party::State* party_state, const Handle& handle,
    const party::QueueTicket& output, std::uint64_t producer_source_identity,
    std::uint64_t authorization_digest, std::uint64_t body_route_identity) {
  if (state == nullptr || state->phase != Phase::ambiguity_open ||
      output.response.disposition != response_inhibition::Disposition::clarify)
    return {};
  return authorize_output(state, party_state, handle, output,
                          producer_source_identity, authorization_digest,
                          body_route_identity,
                          Phase::clarification_authorized);
}

__host__ __device__ inline reader::PublicReply read_public(
    const State* state, const party::State* party_state,
    const reader::ActionTicket& action,
    const reader::ResidentSurface& surface) {
  if (state == nullptr || party_state == nullptr ||
      state->pending_action_sequence != action.action_sequence)
    return {};
  return reader::read(state->public_reader, *party_state, state->calibration,
                      action, surface);
}

__host__ __device__ inline void copy_context(
    ContinuityState* context, const reader::PublicReply& reply) {
  ContinuityState next{};
  next.action_sequence = reply.action_sequence;
  next.public_sequence = reply.public_sequence;
  next.surface_digest = reply.resident_surface_digest;
  next.producer_speaker_identity = reply.producer_speaker_identity;
  next.producer_source_identity = reply.producer_source_identity;
  next.revision = context->revision + 1u;
  if (next.revision == 0u) next.revision = 1u;
  next.raw_byte_count = reply.raw_byte_count;
  next.recipient_count = reply.recipient_count;
  for (std::uint32_t index = 0u; index < reply.raw_byte_count; ++index)
    next.raw_bytes[index] = reply.raw_bytes[index];
  for (std::uint32_t index = 0u; index < reply.recipient_count; ++index)
    next.recipient[index] = reply.recipient[index];
  *context = next;
}

__host__ __device__ inline Handle settle_output(
    State* state, party::State* party_state, const Handle& handle,
    const reader::ActionTicket& action,
    const reader::ResidentSurface& surface,
    const response_inhibition::Consequence& response_consequence,
    const boundary::BodyConsequence& body_consequence,
    Phase settled_phase) {
  if (state == nullptr || party_state == nullptr ||
      !exact_handle(*state, handle) ||
      state->pending_action_sequence != action.action_sequence)
    return {};
  const reader::PublicReply reply =
      read_public(state, party_state, action, surface);
  if (!reply.ready()) return {};
  party::State next_party = *party_state;
  reader::State next_reader = state->public_reader;
  boundary::State next_boundary = state->boundary;
  if (!party::settle_selected(&next_party, action.output,
                              response_consequence) ||
      !boundary::settle_body_consequence(&next_boundary,
                                         state->pending_body,
                                         body_consequence) ||
      !reader::invalidate(&next_reader, action))
    return {};
  *party_state = next_party;
  state->public_reader = next_reader;
  state->boundary = next_boundary;
  copy_context(&state->context, reply);
  state->pending_action = {};
  state->pending_body = {};
  state->pending_action_sequence = 0u;
  advance(state, settled_phase);
  return current_handle(*state);
}

__host__ __device__ inline Handle settle_clarification(
    State* state, party::State* party_state, const Handle& handle,
    const reader::ActionTicket& action,
    const reader::ResidentSurface& surface,
    const response_inhibition::Consequence& response_consequence,
    const boundary::BodyConsequence& body_consequence) {
  if (state == nullptr || state->phase != Phase::clarification_authorized)
    return {};
  return settle_output(state, party_state, handle, action, surface,
                       response_consequence, body_consequence,
                       Phase::awaiting_resolution);
}

__host__ __device__ inline Handle assimilate_resolution(
    State* state, const Handle& handle,
    const reader::ResidentContactReceipt& evidence,
    reader::Disclosure disclosure) {
  if (state == nullptr || state->phase != Phase::awaiting_resolution ||
      !exact_handle(*state, handle) || !evidence.valid() ||
      evidence.speaker_identity != state->origin.speaker_identity ||
      evidence.source_identity != state->origin.source_identity ||
      evidence.transport_sequence <= state->origin.transport_sequence ||
      !candidate_digest(*state, evidence.payload_digest))
    return {};
  const reader::RecipientIntent intent = state->recipient.valid()
      ? reader::RecipientIntent::correct
      : reader::RecipientIntent::grant;
  const reader::RecipientRef recipient = apply_recipient_signal(
      state, {.contact = evidence,
              .intent = intent,
              .ceiling = disclosure,
              .referent_digest = evidence.payload_digest,
              .resident_signal_digest = evidence.payload_digest});
  if (!recipient.valid()) return {};
  state->resolved_referent_digest = evidence.payload_digest;
  ++state->resolutions_assimilated;
  advance(state, Phase::resolution_assimilated);
  return current_handle(*state);
}

__host__ __device__ inline reader::ActionTicket authorize_reply(
    State* state, const party::State* party_state, const Handle& handle,
    const party::QueueTicket& output, std::uint64_t producer_source_identity,
    std::uint64_t authorization_digest, std::uint64_t body_route_identity) {
  if (state == nullptr ||
      (state->phase != Phase::resolution_assimilated &&
       state->phase != Phase::reply_settled &&
       state->phase != Phase::withdrawn) ||
      (state->context.recipient_count != 0u &&
       state->recipient.referent_digest != state->resolved_referent_digest))
    return {};
  const Phase phase = state->phase == Phase::reply_settled
      ? Phase::continuation_authorized
      : Phase::reply_authorized;
  return authorize_output(state, party_state, handle, output,
                          producer_source_identity, authorization_digest,
                          body_route_identity, phase);
}

__host__ __device__ inline Handle settle_reply(
    State* state, party::State* party_state, const Handle& handle,
    const reader::ActionTicket& action,
    const reader::ResidentSurface& surface,
    const response_inhibition::Consequence& response_consequence,
    const boundary::BodyConsequence& body_consequence) {
  if (state == nullptr ||
      (state->phase != Phase::reply_authorized &&
       state->phase != Phase::continuation_authorized))
    return {};
  return settle_output(state, party_state, handle, action, surface,
                       response_consequence, body_consequence,
                       Phase::reply_settled);
}

// Calibration changes are read directly by the canonical public reader.  This
// transition merely retires the now-silent action so a later source consequence
// can authorize a fresh output; it never revives the stale ticket.
__host__ __device__ inline Handle reconcile_source_withdrawal(
    State* state, const party::State* party_state, const Handle& handle,
    const reader::ActionTicket& stale_action) {
  if (state == nullptr || party_state == nullptr ||
      !exact_handle(*state, handle) ||
      state->pending_action_sequence != stale_action.action_sequence ||
      reader::exact_action(state->public_reader, *party_state,
                           state->calibration, stale_action) ||
      !reader::invalidate(&state->public_reader, stale_action))
    return {};
  ++state->withdrawals_observed;
  advance(state, Phase::withdrawn);
  return current_handle(*state);
}

__host__ __device__ inline Handle settle_cancelled_output(
    State* state, party::State* party_state, const Handle& handle,
    const reader::ActionTicket& stale_action,
    const response_inhibition::Consequence& response_consequence,
    const boundary::BodyConsequence& body_consequence) {
  if (state == nullptr || party_state == nullptr ||
      state->phase != Phase::withdrawn || !exact_handle(*state, handle) ||
      state->pending_action_sequence != stale_action.action_sequence ||
      state->public_reader.action.active)
    return {};
  party::State next_party = *party_state;
  boundary::State next_boundary = state->boundary;
  if (!party::settle_selected(&next_party, stale_action.output,
                              response_consequence) ||
      !boundary::settle_body_consequence(&next_boundary,
                                         state->pending_body,
                                         body_consequence))
    return {};
  *party_state = next_party;
  state->boundary = next_boundary;
  state->pending_action = {};
  state->pending_body = {};
  state->pending_action_sequence = 0u;
  ++state->revision;
  if (state->revision == 0u) state->revision = 1u;
  return current_handle(*state);
}

__host__ __device__ inline std::uint64_t byte_digest(
    const void* bytes, std::uint32_t count) {
  if (bytes == nullptr || count == 0u) return 0u;
  const auto* raw = static_cast<const std::uint8_t*>(bytes);
  std::uint64_t digest = 0xcbf29ce484222325ull;
  for (std::uint32_t index = 0u; index < count; ++index) {
    digest ^= static_cast<std::uint64_t>(raw[index]);
    digest *= 0x100000001b3ull;
  }
  return digest == 0u ? 1u : digest;
}

__host__ __device__ inline std::uint64_t image_digest(
    std::uint64_t sequence, std::uint64_t previous, std::uint64_t dialogue,
    std::uint64_t party_state, std::uint64_t episode) {
  std::uint64_t digest = mix(kMagic, kVersion);
  digest = mix(digest, sequence);
  digest = mix(digest, previous);
  digest = mix(digest, dialogue);
  digest = mix(digest, party_state);
  return mix(digest, episode);
}

__host__ __device__ inline RuntimeImage seal_runtime(
    State* state, const party::State* party_state, const Handle& handle,
    std::uint64_t image_sequence) {
  RuntimeImage image{};
  if (state == nullptr || party_state == nullptr ||
      !exact_handle(*state, handle) || state->pending_action_sequence != 0u ||
      image_sequence == 0u || image_sequence <= state->checkpoint_sequence)
    return image;
  State next = *state;
  next.checkpoint_sequence = image_sequence;
  ++next.revision;
  if (next.revision == 0u) next.revision = 1u;
  next.checkpoint_digest = 0u;
  image.image_sequence = image_sequence;
  image.previous_image_digest = state->checkpoint_digest;
  image.dialogue_digest =
      byte_digest(&next, static_cast<std::uint32_t>(sizeof(State)));
  image.party_digest =
      byte_digest(party_state, static_cast<std::uint32_t>(sizeof(party::State)));
  image.image_digest = dialogue_ladder::image_digest(
      image_sequence, image.previous_image_digest, image.dialogue_digest,
      image.party_digest, next.episode_id);
  next.checkpoint_digest = image.image_digest;
  image.dialogue = next;
  image.party = *party_state;
  *state = next;
  return image;
}

__host__ __device__ inline bool restore_runtime(
    State* state, party::State* party_state, const RuntimeImage& image,
    const RestoreFence& fence) {
  if (state == nullptr || party_state == nullptr || image.magic != kMagic ||
      image.version != kVersion || image.reserved != 0u ||
      image.image_sequence < fence.minimum_image_sequence ||
      image.previous_image_digest != fence.expected_previous_image_digest ||
      image.dialogue.episode_id != fence.expected_episode_id ||
      image.dialogue.checkpoint_sequence != image.image_sequence ||
      image.dialogue.checkpoint_digest != image.image_digest ||
      state->checkpoint_sequence >= image.image_sequence ||
      (state->checkpoint_sequence != 0u &&
       state->checkpoint_digest != image.previous_image_digest))
    return false;
  State canonical = image.dialogue;
  canonical.checkpoint_digest = 0u;
  if (byte_digest(&canonical, static_cast<std::uint32_t>(sizeof(State))) !=
          image.dialogue_digest ||
      byte_digest(&image.party,
                  static_cast<std::uint32_t>(sizeof(party::State))) !=
          image.party_digest ||
      dialogue_ladder::image_digest(
          image.image_sequence, image.previous_image_digest,
          image.dialogue_digest, image.party_digest,
          image.dialogue.episode_id) != image.image_digest)
    return false;
  *state = image.dialogue;
  *party_state = image.party;
  return true;
}

}  // namespace substrate::bcc32::dialogue_ladder
