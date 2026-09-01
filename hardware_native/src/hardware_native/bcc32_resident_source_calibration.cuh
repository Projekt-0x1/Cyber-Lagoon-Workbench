#pragma once

#include <cuda_runtime.h>

#include <cstdint>

namespace substrate::bcc32::source_calibration {

inline constexpr std::uint32_t kSourceCapacity = 8u;
inline constexpr std::uint32_t kEvidenceMaximum = 255u;

enum class ConflictStatus : std::uint32_t {
  agreement = 0u,
  unique_source = 1u,
  unresolved = 2u,
  fail_closed = 3u,
};

struct Testimony {
  std::uint64_t source_identity = 0u;
  const std::uint8_t* claim_bytes = nullptr;
  std::uint32_t claim_byte_count = 0u;
  std::uint64_t episode_sequence = 0u;
  std::uint64_t expected_observation_route_identity = 0u;
};

struct Ticket {
  std::uint64_t source_identity = 0u;
  std::uint64_t claim_digest = 0u;
  std::uint64_t episode_sequence = 0u;
  std::uint64_t decision_sequence = 0u;
  std::uint64_t expected_observation_route_identity = 0u;
  std::uint32_t source_slot = 0xffffffffu;
  std::uint32_t source_revision = 0u;

  __host__ __device__ bool valid() const {
    return source_identity != 0u && claim_digest != 0u &&
           episode_sequence != 0u && decision_sequence != 0u &&
           expected_observation_route_identity != 0u &&
           source_slot < kSourceCapacity;
  }
};

struct Observation {
  const std::uint8_t* raw_bytes = nullptr;
  std::uint32_t raw_byte_count = 0u;
  std::uint64_t episode_sequence = 0u;
  std::uint64_t route_identity = 0u;
  std::uint64_t reafference_digest = 0u;
};

struct SourceState {
  std::uint64_t identity = 0u;
  std::uint64_t last_episode_sequence = 0u;
  std::uint64_t pending_decision_sequence = 0u;
  std::uint64_t settled_decision_sequence = 0u;
  std::uint64_t pending_claim_digest = 0u;
  std::uint64_t pending_observation_route_identity = 0u;
  std::uint32_t support_q8 = 0u;
  std::uint32_t contradiction_q8 = 0u;
  std::uint32_t settled_observations = 0u;
  std::uint32_t revision = 0u;
};

struct Receipt {
  std::uint64_t testimony_contacts = 0u;
  std::uint64_t settled_observations = 0u;
  std::uint64_t matched_observations = 0u;
  std::uint64_t contradicted_observations = 0u;
  std::uint64_t rejected_testimony = 0u;
  std::uint64_t rejected_observations = 0u;
  std::uint64_t conflict_queries = 0u;
  std::uint64_t unresolved_conflicts = 0u;
};

struct State {
  SourceState source[kSourceCapacity]{};
  std::uint64_t next_decision_sequence = 1u;
  Receipt receipt{};
};

struct Claim {
  std::uint64_t source_identity = 0u;
  const std::uint8_t* claim_bytes = nullptr;
  std::uint32_t claim_byte_count = 0u;
};

struct ConflictSelection {
  ConflictStatus status = ConflictStatus::fail_closed;
  std::uint64_t selected_source_identity = 0u;
  std::uint64_t selected_claim_digest = 0u;
};

__host__ __device__ inline std::uint64_t digest_bytes(
    const std::uint8_t* bytes, std::uint32_t count) {
  if (bytes == nullptr || count == 0u) return 0u;
  std::uint64_t digest = 0xcbf29ce484222325ull;
  for (std::uint32_t index = 0u; index < count; ++index) {
    digest ^= static_cast<std::uint64_t>(bytes[index]);
    digest *= 0x100000001b3ull;
  }
  digest ^= static_cast<std::uint64_t>(count) * 0x9e3779b97f4a7c15ull;
  return digest == 0u ? 1u : digest;
}

__host__ __device__ inline void mix_integrity(std::uint64_t* digest,
                                              std::uint64_t value) {
  *digest ^= value + 0x9e3779b97f4a7c15ull + (*digest << 6u) +
             (*digest >> 2u);
  *digest *= 0x100000001b3ull;
}

// This is an integrity commitment over a ticket and returned observation.  It
// detects payload substitution but does not authenticate the physical source;
// production ingress still has to establish that boundary independently.
__host__ __device__ inline std::uint64_t expected_reafference_digest(
    const Ticket& ticket, const Observation& observation) {
  if (!ticket.valid() || observation.raw_bytes == nullptr ||
      observation.raw_byte_count == 0u)
    return 0u;
  const std::uint64_t observed_digest =
      digest_bytes(observation.raw_bytes, observation.raw_byte_count);
  if (observed_digest == 0u) return 0u;
  std::uint64_t digest = 0xcbf29ce484222325ull;
  mix_integrity(&digest, ticket.source_identity);
  mix_integrity(&digest, ticket.claim_digest);
  mix_integrity(&digest, ticket.episode_sequence);
  mix_integrity(&digest, ticket.decision_sequence);
  mix_integrity(&digest, ticket.expected_observation_route_identity);
  mix_integrity(&digest, ticket.source_slot);
  mix_integrity(&digest, ticket.source_revision);
  mix_integrity(&digest, observed_digest);
  mix_integrity(&digest, observation.episode_sequence);
  mix_integrity(&digest, observation.route_identity);
  return digest == 0u ? 1u : digest;
}

__host__ __device__ inline std::uint32_t add_evidence(
    std::uint32_t value, std::uint32_t increment) {
  value = value > kEvidenceMaximum ? kEvidenceMaximum : value;
  increment = increment > kEvidenceMaximum ? kEvidenceMaximum : increment;
  return increment > kEvidenceMaximum - value ? kEvidenceMaximum
                                               : value + increment;
}

__host__ __device__ inline std::uint32_t remove_evidence(
    std::uint32_t value, std::uint32_t decrement) {
  value = value > kEvidenceMaximum ? kEvidenceMaximum : value;
  return decrement >= value ? 0u : value - decrement;
}

__host__ __device__ inline std::uint32_t find_source(
    const State* state, std::uint64_t identity) {
  if (state == nullptr || identity == 0u) return 0xffffffffu;
  for (std::uint32_t slot = 0u; slot < kSourceCapacity; ++slot)
    if (state->source[slot].identity == identity) return slot;
  return 0xffffffffu;
}

__host__ __device__ inline std::uint32_t find_or_allocate_source(
    State* state, std::uint64_t identity) {
  const std::uint32_t existing = find_source(state, identity);
  if (existing != 0xffffffffu) return existing;
  if (state == nullptr || identity == 0u) return 0xffffffffu;
  for (std::uint32_t slot = 0u; slot < kSourceCapacity; ++slot) {
    if (state->source[slot].identity != 0u) continue;
    state->source[slot].identity = identity;
    state->source[slot].revision = 1u;
    return slot;
  }
  return 0xffffffffu;
}

// Testimony allocates a consequence ticket but does not change reliability.
// Volume, repetition, register, and arrival order therefore cannot become
// source authority without later route-bound observation.
__host__ __device__ inline Ticket observe_testimony(State* state,
                                                    const Testimony& testimony) {
  Ticket ticket{};
  if (state == nullptr || testimony.source_identity == 0u ||
      testimony.claim_bytes == nullptr || testimony.claim_byte_count == 0u ||
      testimony.episode_sequence == 0u ||
      testimony.expected_observation_route_identity == 0u) {
    if (state != nullptr) ++state->receipt.rejected_testimony;
    return ticket;
  }
  const std::uint64_t claim_digest =
      digest_bytes(testimony.claim_bytes, testimony.claim_byte_count);
  const std::uint32_t slot =
      find_or_allocate_source(state, testimony.source_identity);
  if (claim_digest == 0u || slot == 0xffffffffu) {
    ++state->receipt.rejected_testimony;
    return ticket;
  }
  SourceState& source = state->source[slot];
  if (source.pending_decision_sequence != 0u ||
      (source.last_episode_sequence != 0u &&
       testimony.episode_sequence <= source.last_episode_sequence)) {
    ++state->receipt.rejected_testimony;
    return ticket;
  }
  const std::uint64_t decision_sequence = state->next_decision_sequence++;
  source.last_episode_sequence = testimony.episode_sequence;
  source.pending_decision_sequence = decision_sequence;
  source.pending_claim_digest = claim_digest;
  source.pending_observation_route_identity =
      testimony.expected_observation_route_identity;
  ++source.revision;
  ++state->receipt.testimony_contacts;

  ticket.source_identity = testimony.source_identity;
  ticket.claim_digest = claim_digest;
  ticket.episode_sequence = testimony.episode_sequence;
  ticket.decision_sequence = decision_sequence;
  ticket.expected_observation_route_identity =
      testimony.expected_observation_route_identity;
  ticket.source_slot = slot;
  ticket.source_revision = source.revision;
  return ticket;
}

__host__ __device__ inline bool settle_observation(
    State* state, const Ticket& ticket, const Observation& observation) {
  if (state == nullptr || !ticket.valid() || observation.raw_bytes == nullptr ||
      observation.raw_byte_count == 0u ||
      observation.episode_sequence != ticket.episode_sequence ||
      observation.route_identity !=
          ticket.expected_observation_route_identity ||
      observation.reafference_digest !=
          expected_reafference_digest(ticket, observation)) {
    if (state != nullptr) ++state->receipt.rejected_observations;
    return false;
  }
  SourceState& source = state->source[ticket.source_slot];
  if (source.identity != ticket.source_identity ||
      source.pending_decision_sequence != ticket.decision_sequence ||
      source.pending_claim_digest != ticket.claim_digest ||
      source.pending_observation_route_identity !=
          ticket.expected_observation_route_identity ||
      source.revision != ticket.source_revision ||
      source.settled_decision_sequence >= ticket.decision_sequence) {
    ++state->receipt.rejected_observations;
    return false;
  }
  const std::uint64_t observed_digest =
      digest_bytes(observation.raw_bytes, observation.raw_byte_count);
  if (observed_digest == 0u) {
    ++state->receipt.rejected_observations;
    return false;
  }
  if (observed_digest == ticket.claim_digest) {
    source.support_q8 = add_evidence(source.support_q8, 32u);
    source.contradiction_q8 = remove_evidence(source.contradiction_q8, 16u);
    ++state->receipt.matched_observations;
  } else {
    source.contradiction_q8 = add_evidence(source.contradiction_q8, 32u);
    source.support_q8 = remove_evidence(source.support_q8, 16u);
    ++state->receipt.contradicted_observations;
  }
  source.pending_decision_sequence = 0u;
  source.pending_claim_digest = 0u;
  source.pending_observation_route_identity = 0u;
  source.settled_decision_sequence = ticket.decision_sequence;
  if (source.settled_observations != 0xffffffffu)
    ++source.settled_observations;
  ++source.revision;
  ++state->receipt.settled_observations;
  return true;
}

__host__ __device__ inline bool source_dominates(const SourceState& left,
                                                 const SourceState& right) {
  if (left.settled_observations == 0u) return false;
  const bool no_worse = left.support_q8 >= right.support_q8 &&
                        left.contradiction_q8 <= right.contradiction_q8;
  const bool strictly_better = left.support_q8 > right.support_q8 ||
                               left.contradiction_q8 < right.contradiction_q8;
  return no_worse && strictly_better;
}

__host__ __device__ inline ConflictSelection select_conflicting_claim(
    State* state, const Claim& left, const Claim& right) {
  ConflictSelection selection{};
  if (state == nullptr || left.source_identity == 0u ||
      right.source_identity == 0u || left.source_identity == right.source_identity ||
      left.claim_bytes == nullptr || right.claim_bytes == nullptr ||
      left.claim_byte_count == 0u || right.claim_byte_count == 0u) {
    return selection;
  }
  const std::uint64_t left_digest =
      digest_bytes(left.claim_bytes, left.claim_byte_count);
  const std::uint64_t right_digest =
      digest_bytes(right.claim_bytes, right.claim_byte_count);
  if (left_digest == 0u || right_digest == 0u) return selection;
  ++state->receipt.conflict_queries;
  if (left_digest == right_digest) {
    selection.status = ConflictStatus::agreement;
    selection.selected_claim_digest = left_digest;
    return selection;
  }
  const std::uint32_t left_slot = find_source(state, left.source_identity);
  const std::uint32_t right_slot = find_source(state, right.source_identity);
  if (left_slot == 0xffffffffu || right_slot == 0xffffffffu) {
    selection.status = ConflictStatus::unresolved;
    ++state->receipt.unresolved_conflicts;
    return selection;
  }
  const SourceState& left_source = state->source[left_slot];
  const SourceState& right_source = state->source[right_slot];
  const bool left_wins = source_dominates(left_source, right_source);
  const bool right_wins = source_dominates(right_source, left_source);
  if (left_wins == right_wins) {
    selection.status = ConflictStatus::unresolved;
    ++state->receipt.unresolved_conflicts;
    return selection;
  }
  selection.status = ConflictStatus::unique_source;
  selection.selected_source_identity =
      left_wins ? left.source_identity : right.source_identity;
  selection.selected_claim_digest = left_wins ? left_digest : right_digest;
  return selection;
}

}  // namespace substrate::bcc32::source_calibration
