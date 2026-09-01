#ifndef HARDWARE_NATIVE_DIRECT_CANONICAL_TRANSITION_DEVICE_CUH
#define HARDWARE_NATIVE_DIRECT_CANONICAL_TRANSITION_DEVICE_CUH

// github #1236 rung 3: one bounded logical transition result.
//
// Rung 1 centralized admission/ranking. Main's rung 2 centralized what an
// explicit/implicit winner MEANS in direct_canonical_*_effect. This header does
// not reimplement either law. It only collects those already-canonical effects
// into the full bounded result the execution fabric must commit: the explicit
// winner (if any) plus every implicit candidate admitted by the canonical mesh
// gate. Physical backends may discover candidates differently; they may not
// change this result shape or rebuild its events themselves.

#include "hardware_native/direct_canonical_evaluator_device.cuh"

namespace substrate::direct_adult {

inline constexpr std::uint32_t kCanonicalSuccessorCapacity =
    1u + kMaxImplicitActiveFanout;

enum class DirectCanonicalSuccessorKind : std::uint32_t {
  explicit_route = 0u,
  implicit_virtual = 1u,
};

struct DirectCanonicalSuccessor {
  DirectCanonicalEffect effect;
  DirectCanonicalSuccessorKind kind;
  std::uint32_t route_slot;
  std::uint32_t implicit_family;
  std::uint32_t implicit_slot;
  std::int32_t conductance_q16;
  std::uint32_t delay;
  std::uint32_t participating;
};

struct DirectCanonicalTransitionResult {
  std::uint64_t event_context_signature;
  DirectCanonicalWinner explicit_winner;
  std::uint32_t explicit_candidate_count;
  std::uint32_t successor_count;
  DirectCanonicalSuccessor successors[kCanonicalSuccessorCapacity];
};

__device__ inline std::uint32_t direct_canonical_resolved_cue_node(
    const ActivityEvent& event) {
  return event.cue_node == kInvalidIndex ? event.node : event.cue_node;
}

__device__ inline void direct_canonical_append_explicit_successor(
    const DirectBrainV01& brain, const ActivityEvent& event,
    DirectCanonicalTransitionResult* result) {
  if (result->explicit_winner.route_slot == kInvalidIndex ||
      result->successor_count >= kCanonicalSuccessorCapacity)
    return;
  const std::uint32_t route_slot = result->explicit_winner.route_slot;
  if (route_slot >= brain.route_capacity)
    return;
  const DirectRoute route = brain.routes[route_slot];
  if (route.target >= brain.node_count)
    return;

  DirectCanonicalSuccessor successor{};
  successor.effect = direct_canonical_explicit_effect(
      brain, event, direct_canonical_resolved_cue_node(event), route_slot);
  successor.kind = DirectCanonicalSuccessorKind::explicit_route;
  successor.route_slot = route_slot;
  successor.implicit_family = kInvalidIndex;
  successor.implicit_slot = kInvalidIndex;
  successor.conductance_q16 = route.conductance_q16;
  successor.delay = route.delay;
  successor.participating = 1u;
  result->successors[result->successor_count++] = successor;
}

__device__ inline void direct_canonical_append_implicit_successor(
    const DirectBrainV01& brain, const ActivityEvent& event,
    const DirectImplicitCandidate& implicit, bool participating,
    DirectCanonicalTransitionResult* result) {
  if (implicit.valid == 0u || implicit.target >= brain.node_count ||
      result->successor_count >= kCanonicalSuccessorCapacity)
    return;

  DirectCanonicalSuccessor successor{};
  successor.effect = direct_canonical_implicit_effect(
      brain, event, direct_canonical_resolved_cue_node(event), implicit);
  successor.kind = DirectCanonicalSuccessorKind::implicit_virtual;
  successor.route_slot = kInvalidIndex;
  successor.implicit_family = implicit.family;
  successor.implicit_slot = implicit.virtual_slot;
  successor.conductance_q16 = implicit.conductance_q16;
  successor.delay = implicit.delay;
  successor.participating = participating ? 1u : 0u;
  result->successors[result->successor_count++] = successor;
}

__device__ inline DirectCanonicalTransitionResult
direct_canonical_finalize_transition(
    const DirectBrainV01& brain, const ActivityEvent& event,
    const DirectCanonicalWinner& explicit_winner,
    std::uint32_t explicit_candidate_count,
    std::uint64_t event_context_signature) {
  DirectCanonicalTransitionResult result{};
  result.event_context_signature = event_context_signature;
  result.explicit_winner = explicit_winner;
  result.explicit_candidate_count = explicit_candidate_count;

  // Preserve the current canonical Direct law exactly: the explicit winner
  // propagates, and when the mesh gate is open every bounded implicit candidate
  // also propagates. implicit_wins controls causal participation/materialization
  // authority; it does not make the remaining candidate effects disappear.
  direct_canonical_append_explicit_successor(brain, event, &result);
  if (direct_canonical_implicit_mesh_eligible(explicit_winner,
                                              explicit_candidate_count)) {
    const DirectImplicitCandidateSet implicit_set =
        enumerate_direct_implicit_candidates(brain, event.node,
                                             event_context_signature);
    for (std::uint32_t c = 0u; c < implicit_set.count; ++c) {
      const DirectImplicitCandidate implicit = implicit_set.candidates[c];
      direct_canonical_append_implicit_successor(
          brain, event, implicit,
          direct_canonical_implicit_wins(implicit, explicit_winner,
                                         explicit_candidate_count),
          &result);
    }
  }
  return result;
}

__device__ inline DirectCanonicalTransitionResult
direct_canonical_evaluate_source_transition(const DirectBrainV01& brain,
                                            const ActivityEvent& event) {
  DirectCanonicalTransitionResult result{};
  if (event.node >= brain.node_count)
    return result;
  const DirectNode source = brain.nodes[event.node];
  const std::uint64_t signature =
      direct_canonical_event_context_signature(event.history_signature,
                                               event.word);
  const DirectCanonicalWinner winner =
      direct_canonical_scan_source_routes(brain, source, signature);
  return direct_canonical_finalize_transition(brain, event, winner,
                                              source.route_count, signature);
}

}  // namespace substrate::direct_adult

#endif  // HARDWARE_NATIVE_DIRECT_CANONICAL_TRANSITION_DEVICE_CUH
