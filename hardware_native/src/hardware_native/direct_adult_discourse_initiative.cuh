#ifndef HARDWARE_NATIVE_DIRECT_ADULT_DISCOURSE_INITIATIVE_CUH
#define HARDWARE_NATIVE_DIRECT_ADULT_DISCOURSE_INITIATIVE_CUH

// h.discourse_initiative (#1623). Conversational initiative composes three
// already-proven resident surfaces: grounded one-shot bindings carry the
// learned clarification trajectory, the causal world model carries the topic
// grounding, and the turn gate carries timing. Own output is reafferent
// evidence only -- it can never reopen the gate or confirm its own
// uncertainty. Silence is a first-class decision, not a failure.

#include <cstdint>

#include "hardware_native/direct_adult_causal_world_model.cuh"
#include "hardware_native/direct_adult_core_constants.cuh"

namespace substrate::direct_network {

enum class DirectDiscourseAction : std::uint32_t {
  remain_silent = 0u,
  continue_topic = 1u,
  ask_clarification = 2u,
};

struct DirectTurnGate {
  std::uint64_t spoken_ticket_id;  // our last emission; 0 = gate open
  std::uint32_t reopen_returns;
};

static_assert(std::is_trivially_copyable_v<DirectTurnGate>);

__host__ __device__ inline std::uint64_t discourse_fold(
    std::uint64_t hash, std::uint64_t value) {
  return (hash ^ (value + 0x9e3779b97f4a7c15ULL + (hash << 6u) +
                  (hash >> 2u))) *
         1099511628211ULL;
}

// A candidate return reopens the turn gate only when it is a verified
// observation whose identity is independent of our own last emission. A
// return that merely echoes our spoken ticket back -- our own reafference --
// never reopens anything and never counts.
__device__ inline bool discourse_return_reopens(
    DirectTurnGate* gate, const DirectExactHistoryRecord* records,
    std::uint32_t record_count, std::uint64_t candidate_return_identity) {
  if (gate == nullptr || records == nullptr || gate->spoken_ticket_id == 0u ||
      candidate_return_identity == gate->spoken_ticket_id)
    return false;
  const DirectExactHistoryRecord* candidate = nullptr;
  for (std::uint32_t i = 0u; i < record_count; ++i)
    if (records[i].identity == candidate_return_identity &&
        records[i].kind == DirectExactHistoryKind::world_return &&
        (records[i].flags & kDirectHistoryVerifiedObservation) != 0u)
      candidate = &records[i];
  if (candidate == nullptr) return false;
  // Independence: follow the return's ancestry; it must reach at least one
  // non-empty ancestor that is not our own spoken emission.
  std::uint64_t pending[4] = {candidate_return_identity, 0u, 0u, 0u};
  std::uint32_t pending_count = 1u;
  bool independent_root = false;
  for (std::uint32_t cursor = 0u;
       cursor < pending_count && !independent_root; ++cursor) {
    for (std::uint32_t r = 0u; r < record_count; ++r) {
      const DirectExactHistoryRecord& record = records[r];
      if (record.identity != pending[cursor] ||
          record.kind == DirectExactHistoryKind::empty)
        continue;
      if (record.identity != gate->spoken_ticket_id)
        independent_root = true;
      const std::uint64_t parent = record.parent_identity;
      if (parent == 0u || parent == direct_adult_core::kInvalidTicket)
        continue;
      bool known = false;
      for (std::uint32_t j = 0u; j < pending_count; ++j)
        known |= pending[j] == parent;
      if (!known && pending_count < 4u) pending[pending_count++] = parent;
    }
  }
  if (!independent_root) return false;
  gate->spoken_ticket_id = 0u;
  ++gate->reopen_returns;
  return true;
}

// The initiative decision. Everything input is resident evidence: whether
// the turn gate stands open, whether the topic relation actually grounds
// (the model can predict it), whether a learned clarification trajectory
// exists, and whether the partner's return left the topic unresolved.
// Fabrication is not an option: without grounding the lane either asks, if
// it ever learned how, or keeps autonomous voluntary silence.
__host__ __device__ inline DirectDiscourseAction discourse_decide(
    bool turn_open, bool topic_grounded, bool expression_learned,
    bool partner_return_unresolved) {
  if (!turn_open || !expression_learned)
    return DirectDiscourseAction::remain_silent;
  if (partner_return_unresolved || !topic_grounded)
    return DirectDiscourseAction::ask_clarification;
  return DirectDiscourseAction::continue_topic;
}

}  // namespace substrate::direct_network

#endif
