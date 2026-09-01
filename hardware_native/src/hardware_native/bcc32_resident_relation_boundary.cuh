#pragma once

#include <cuda_runtime.h>

#include <cstdint>

namespace substrate::bcc32::relation_boundary {

inline constexpr std::uint32_t kRelationCapacity = 8u;
inline constexpr std::uint32_t kPressureMaximum = 255u;
inline constexpr std::uint32_t kDeferThreshold = 96u;
inline constexpr std::uint32_t kRefuseThreshold = 192u;
inline constexpr std::uint32_t kMaximumRepairStep = 96u;

enum class Disposition : std::uint32_t {
  engage = 0u,
  defer = 1u,
  refuse = 2u,
  fail_closed = 3u,
};

struct Interaction {
  std::uint64_t speaker_identity = 0u;
  std::uint64_t interaction_sequence = 0u;
  std::uint64_t expected_body_route_identity = 0u;
};

struct Ticket {
  std::uint64_t speaker_identity = 0u;
  std::uint64_t interaction_sequence = 0u;
  std::uint64_t decision_sequence = 0u;
  std::uint64_t expected_body_route_identity = 0u;
  std::uint32_t relation_slot = 0xffffffffu;
  std::uint32_t relation_revision = 0u;

  __host__ __device__ bool valid() const {
    return speaker_identity != 0u && interaction_sequence != 0u &&
           decision_sequence != 0u && expected_body_route_identity != 0u &&
           relation_slot < kRelationCapacity;
  }
};

struct BodyConsequence {
  std::uint64_t interaction_sequence = 0u;
  std::uint64_t route_identity = 0u;
  std::uint64_t reafference_digest = 0u;
  // Resident/body measurements only.  There is no host-provided harm, apology,
  // affection, or desired-boundary label.
  std::uint32_t damage_pressure_q8 = 0u;
  std::uint32_t repair_assimilation_q8 = 0u;
  std::int32_t resource_delta_q8 = 0;
};

__host__ __device__ inline void mix_integrity(std::uint64_t* digest,
                                              std::uint64_t value) {
  *digest ^= value + 0x9e3779b97f4a7c15ull + (*digest << 6u) +
             (*digest >> 2u);
  *digest *= 0x100000001b3ull;
}

// Integrity only: this binds returned measurements to the issued ticket but
// does not authenticate the physical body route that produced them.
__host__ __device__ inline std::uint64_t expected_reafference_digest(
    const Ticket& ticket, const BodyConsequence& consequence) {
  if (!ticket.valid()) return 0u;
  std::uint64_t digest = 0xcbf29ce484222325ull;
  mix_integrity(&digest, ticket.speaker_identity);
  mix_integrity(&digest, ticket.interaction_sequence);
  mix_integrity(&digest, ticket.decision_sequence);
  mix_integrity(&digest, ticket.expected_body_route_identity);
  mix_integrity(&digest, ticket.relation_slot);
  mix_integrity(&digest, ticket.relation_revision);
  mix_integrity(&digest, consequence.interaction_sequence);
  mix_integrity(&digest, consequence.route_identity);
  mix_integrity(&digest, consequence.damage_pressure_q8);
  mix_integrity(&digest, consequence.repair_assimilation_q8);
  mix_integrity(
      &digest,
      static_cast<std::uint64_t>(
          static_cast<std::int64_t>(consequence.resource_delta_q8)));
  return digest == 0u ? 1u : digest;
}

struct RelationState {
  std::uint64_t speaker_identity = 0u;
  std::uint64_t last_interaction_sequence = 0u;
  std::uint64_t pending_decision_sequence = 0u;
  std::uint64_t settled_decision_sequence = 0u;
  std::uint32_t boundary_pressure_q8 = 0u;
  std::uint32_t accumulated_damage_q8 = 0u;
  std::uint32_t accumulated_repair_q8 = 0u;
  std::uint32_t settled_interactions = 0u;
  std::uint32_t revision = 0u;
};

struct Receipt {
  std::uint64_t issued = 0u;
  std::uint64_t settled = 0u;
  std::uint64_t rejected = 0u;
  std::uint64_t damaging = 0u;
  std::uint64_t repairing = 0u;
  std::uint64_t neutral = 0u;
  std::uint64_t engage_decisions = 0u;
  std::uint64_t defer_decisions = 0u;
  std::uint64_t refuse_decisions = 0u;
};

struct State {
  RelationState relation[kRelationCapacity]{};
  std::uint64_t next_decision_sequence = 1u;
  Receipt receipt{};
};

__host__ __device__ inline std::uint32_t clamp(std::uint32_t value) {
  return value > kPressureMaximum ? kPressureMaximum : value;
}

__host__ __device__ inline std::uint32_t add(std::uint32_t value,
                                            std::uint32_t increment) {
  value = clamp(value);
  increment = clamp(increment);
  return increment > kPressureMaximum - value ? kPressureMaximum
                                               : value + increment;
}

__host__ __device__ inline std::uint32_t subtract(std::uint32_t value,
                                                 std::uint32_t decrement) {
  value = clamp(value);
  return decrement >= value ? 0u : value - decrement;
}

__host__ __device__ inline std::uint32_t find_relation(
    const State* state, std::uint64_t speaker_identity) {
  if (state == nullptr || speaker_identity == 0u) return 0xffffffffu;
  for (std::uint32_t slot = 0u; slot < kRelationCapacity; ++slot)
    if (state->relation[slot].speaker_identity == speaker_identity) return slot;
  return 0xffffffffu;
}

__host__ __device__ inline std::uint32_t find_or_allocate_relation(
    State* state, std::uint64_t speaker_identity) {
  const std::uint32_t existing = find_relation(state, speaker_identity);
  if (existing != 0xffffffffu) return existing;
  if (state == nullptr || speaker_identity == 0u) return 0xffffffffu;
  for (std::uint32_t slot = 0u; slot < kRelationCapacity; ++slot) {
    if (state->relation[slot].speaker_identity != 0u) continue;
    state->relation[slot].speaker_identity = speaker_identity;
    state->relation[slot].revision = 1u;
    return slot;
  }
  return 0xffffffffu;
}

__host__ __device__ inline Disposition disposition_for(
    State* state, std::uint64_t speaker_identity) {
  if (state == nullptr || speaker_identity == 0u)
    return Disposition::fail_closed;
  const std::uint32_t slot = find_relation(state, speaker_identity);
  const std::uint32_t pressure =
      slot == 0xffffffffu ? 0u : clamp(state->relation[slot].boundary_pressure_q8);
  Disposition result = Disposition::engage;
  if (pressure >= kRefuseThreshold)
    result = Disposition::refuse;
  else if (pressure >= kDeferThreshold)
    result = Disposition::defer;
  if (result == Disposition::engage)
    ++state->receipt.engage_decisions;
  else if (result == Disposition::defer)
    ++state->receipt.defer_decisions;
  else
    ++state->receipt.refuse_decisions;
  return result;
}

__host__ __device__ inline Ticket issue_interaction(
    State* state, const Interaction& interaction) {
  Ticket ticket{};
  if (state == nullptr || interaction.speaker_identity == 0u ||
      interaction.interaction_sequence == 0u ||
      interaction.expected_body_route_identity == 0u) {
    if (state != nullptr) ++state->receipt.rejected;
    return ticket;
  }
  const std::uint32_t slot =
      find_or_allocate_relation(state, interaction.speaker_identity);
  if (slot == 0xffffffffu) {
    ++state->receipt.rejected;
    return ticket;
  }
  RelationState& relation = state->relation[slot];
  if (relation.pending_decision_sequence != 0u ||
      (relation.last_interaction_sequence != 0u &&
       interaction.interaction_sequence <= relation.last_interaction_sequence)) {
    ++state->receipt.rejected;
    return ticket;
  }
  const std::uint64_t decision_sequence = state->next_decision_sequence++;
  relation.last_interaction_sequence = interaction.interaction_sequence;
  relation.pending_decision_sequence = decision_sequence;
  ++relation.revision;
  ++state->receipt.issued;
  ticket.speaker_identity = interaction.speaker_identity;
  ticket.interaction_sequence = interaction.interaction_sequence;
  ticket.decision_sequence = decision_sequence;
  ticket.expected_body_route_identity =
      interaction.expected_body_route_identity;
  ticket.relation_slot = slot;
  ticket.relation_revision = relation.revision;
  return ticket;
}

__host__ __device__ inline bool settle_body_consequence(
    State* state, const Ticket& ticket, const BodyConsequence& consequence) {
  if (state == nullptr || !ticket.valid() ||
      consequence.interaction_sequence != ticket.interaction_sequence ||
      consequence.route_identity != ticket.expected_body_route_identity ||
      consequence.reafference_digest !=
          expected_reafference_digest(ticket, consequence)) {
    if (state != nullptr) ++state->receipt.rejected;
    return false;
  }
  RelationState& relation = state->relation[ticket.relation_slot];
  if (relation.speaker_identity != ticket.speaker_identity ||
      relation.pending_decision_sequence != ticket.decision_sequence ||
      relation.revision != ticket.relation_revision ||
      relation.settled_decision_sequence >= ticket.decision_sequence) {
    ++state->receipt.rejected;
    return false;
  }
  const std::uint32_t resource_cost = consequence.resource_delta_q8 < 0
      ? static_cast<std::uint32_t>(
            -static_cast<std::int64_t>(consequence.resource_delta_q8) > 255
                ? 255
                : -static_cast<std::int64_t>(consequence.resource_delta_q8))
      : 0u;
  const std::uint32_t resource_return = consequence.resource_delta_q8 > 0
      ? static_cast<std::uint32_t>(consequence.resource_delta_q8 > 255
                                       ? 255
                                       : consequence.resource_delta_q8)
      : 0u;
  const std::uint32_t damaging =
      add(clamp(consequence.damage_pressure_q8), resource_cost);
  const std::uint32_t repairing =
      add(clamp(consequence.repair_assimilation_q8), resource_return);
  if (damaging > repairing) {
    const std::uint32_t delta = damaging - repairing;
    relation.boundary_pressure_q8 = add(relation.boundary_pressure_q8, delta);
    relation.accumulated_damage_q8 = add(relation.accumulated_damage_q8, delta);
    ++state->receipt.damaging;
  } else if (repairing > damaging) {
    std::uint32_t delta = repairing - damaging;
    if (delta > kMaximumRepairStep) delta = kMaximumRepairStep;
    relation.boundary_pressure_q8 =
        subtract(relation.boundary_pressure_q8, delta);
    relation.accumulated_repair_q8 = add(relation.accumulated_repair_q8, delta);
    ++state->receipt.repairing;
  } else {
    ++state->receipt.neutral;
  }
  relation.pending_decision_sequence = 0u;
  relation.settled_decision_sequence = ticket.decision_sequence;
  if (relation.settled_interactions != 0xffffffffu)
    ++relation.settled_interactions;
  ++relation.revision;
  ++state->receipt.settled;
  return true;
}

}  // namespace substrate::bcc32::relation_boundary
