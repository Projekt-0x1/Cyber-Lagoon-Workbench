#ifndef HARDWARE_NATIVE_DIRECT_CANONICAL_EVALUATOR_DEVICE_CUH
#define HARDWARE_NATIVE_DIRECT_CANONICAL_EVALUATOR_DEVICE_CUH

// gh #1236 rung 1 -- the ONE causal admission/ranking law.
//
// A hardware backend may change representation and cost. It may not silently
// change which logical causal transition happened. Before this header there
// were five independent implementations of "which route wins this event":
//
//   1. propagate_sparse_frontier_kernel      (direct_adult_legacy_oracle.cu)
//   2. resolve_packed_cache_winner           (direct_exact_eligibility_device.cuh)
//   3. record_fabric_explicit_eligibility_kernel (direct_adult_legacy_oracle.cu)
//   4. propagate_sparse_reference_single_event   (direct_execution_fabric.cu)
//   5. evaluate_packed_sparse_source             (direct_packed_sparse_execution.cu)
//
// (1) and (2) agreed by hand-copied code. (3), (4) and (5) scored ADDITIVELY
// with kDeepContextMatchBonusQ16 and tested NO conductance floor, and they
// derived the event's context signature by a different rule. Both differences
// are load-bearing and both were measured live by
// cuda_direct_fabric_fallback_divergence_contract.
//
// Everything here is `inline __device__` with no backend argument, so a
// backend cannot pass a flag that changes the law. A backend that wants a
// different decision has to stop calling this header, which is a visible diff
// rather than a silent drift.

#include "hardware_native/direct_execution_fabric.cuh"
#include "hardware_native/direct_implicit_causal_mesh.cuh"

namespace substrate::direct_adult {

// ---------------------------------------------------------------------------
// Canonical event context signature.
//
// The signature an evaluator compares a route against is part of the causal
// law, not a local convenience. A route's context_signature is stamped from
// the eligibility record that grew it (direct_adult_legacy_oracle.cu:581 copies
// record.history_signature into route.context_signature), and the canonical
// path stamps that record with THIS function's output. An evaluator that
// derives the event signature differently is asking a different question of
// the same stored route: the identical learned route matches under one backend
// and not the other, with no field anywhere recording the disagreement.
// ---------------------------------------------------------------------------
__device__ inline std::uint64_t direct_canonical_event_context_signature(
    std::uint64_t predecessor_history, Word current_word) {
  const std::uint32_t predecessor_shallow = static_cast<std::uint32_t>(predecessor_history);
  const std::uint32_t shallow =
      mix_signature((static_cast<std::uint64_t>(predecessor_shallow) << 32) | current_word);
  const std::uint32_t predecessor_deep = static_cast<std::uint32_t>(predecessor_history >> 32);
  const std::uint32_t deep =
      predecessor_deep == 0u
          ? 0u
          : mix_signature((static_cast<std::uint64_t>(predecessor_deep) << 32) | shallow);
  return (static_cast<std::uint64_t>(deep) << 32) | shallow;
}

// ---------------------------------------------------------------------------
// Admission floor.
//
// A route below the floor is not a weak candidate, it is not a candidate. This
// is the difference the divergence contract's assay 2/3 measured: under a
// fabric-enabled tick the adult acted on evidence the canonical evaluator
// considers too weak to act on.
// ---------------------------------------------------------------------------
__device__ inline bool direct_canonical_route_admissible(const DirectRoute& route) {
  return route.conductance_q16 > kMinimumConductanceQ16;
}

// ---------------------------------------------------------------------------
// Context specificity rank.
//
// Structural, not a magic conductance bonus. A route learned for the exact
// resident history outranks one that only shares the immediate predecessor
// suffix; either outranks a context-free fallback. Conductance is the tie-break
// WITHIN one specificity class, so repeated experience can strengthen an
// alternative without erasing a learned context -- which an additive bonus
// cannot express, because a large enough conductance always buys the tier.
// ---------------------------------------------------------------------------
__device__ inline std::uint32_t direct_canonical_context_rank(
    const DirectRoute& route, std::uint64_t event_context_signature) {
  if (!direct_canonical_route_admissible(route))
    return 0u;
  const std::uint32_t route_shallow = static_cast<std::uint32_t>(route.context_signature);
  const std::uint32_t event_shallow = static_cast<std::uint32_t>(event_context_signature);
  const bool shallow_match = route_shallow != 0u && route_shallow == event_shallow;
  const bool deep_match = shallow_match && (route.context_signature >> 32) != 0u &&
                          route.context_signature == event_context_signature;
  return deep_match ? 2u : (shallow_match ? 1u : 0u);
}

// The winner of one logical evaluation. `route_slot == kInvalidIndex` is the
// ordinary "no conducting candidate" outcome, not an error.
struct DirectCanonicalWinner {
  std::uint32_t route_slot;
  std::uint32_t context_rank;
  std::int64_t conductance_q16;
  std::uint32_t admissible_candidates;
};

__device__ inline DirectCanonicalWinner direct_canonical_winner_empty() {
  DirectCanonicalWinner winner;
  winner.route_slot = kInvalidIndex;
  winner.context_rank = 0u;
  winner.conductance_q16 = -1;
  winner.admissible_candidates = 0u;
  return winner;
}

// The comparison rule itself: rank first, conductance only within a rank.
__device__ inline bool direct_canonical_consider(DirectCanonicalWinner* winner,
                                                 std::uint32_t route_slot, std::uint32_t rank,
                                                 std::int64_t conductance_q16) {
  ++winner->admissible_candidates;
  if (winner->route_slot == kInvalidIndex || rank > winner->context_rank ||
      (rank == winner->context_rank && conductance_q16 > winner->conductance_q16)) {
    winner->route_slot = route_slot;
    winner->context_rank = rank;
    winner->conductance_q16 = conductance_q16;
    return true;
  }
  return false;
}

// Offer one route slot to the competition under the full canonical law
// (liveness, floor, rank, tie-break). Returns true when it became the winner.
__device__ inline bool direct_canonical_offer_route(DirectCanonicalWinner* winner,
                                                    const DirectBrainV01& brain,
                                                    std::uint32_t route_slot,
                                                    std::uint64_t event_context_signature) {
  if (route_slot >= brain.route_capacity)
    return false;
  const DirectRouteSlotMeta meta = brain.topology.slot_meta[route_slot];
  if (meta.live == 0u)
    return false;
  const DirectRoute route = brain.routes[route_slot];
  if (!direct_canonical_route_admissible(route))
    return false;
  return direct_canonical_consider(winner, route_slot,
                                   direct_canonical_context_rank(route, event_context_signature),
                                   static_cast<std::int64_t>(route.conductance_q16));
}

// The canonical serial scan of one source's explicit route list.
__device__ inline DirectCanonicalWinner direct_canonical_scan_source_routes(
    const DirectBrainV01& brain, const DirectNode& source,
    std::uint64_t event_context_signature) {
  DirectCanonicalWinner winner = direct_canonical_winner_empty();
  std::uint32_t route_index = source.first_route;
  for (std::uint32_t visited = 0u; visited < source.route_count && route_index != kInvalidIndex;
       ++visited) {
    if (route_index >= brain.route_capacity)
      break;
    direct_canonical_offer_route(&winner, brain, route_index, event_context_signature);
    route_index = brain.routes[route_index].next_route;
  }
  return winner;
}

// ---------------------------------------------------------------------------
// Implicit-vs-explicit arbitration.
//
// The canonical rule is rank-aware: a procedural/implicit candidate may only
// outbid an explicit route that carries NO context specificity at all. A route
// the organism learned for this exact history is not for sale, however strong
// the mesh's generic prior happens to be -- otherwise the implicit mesh, whose
// conductance is a family-level constant, could overwrite an individually
// learned context everywhere at once.
//
// `explicit_candidate_count` is the number of explicit candidates this backend
// actually had available to offer -- the linked-list length for a scanning
// backend, the panel entry count for a packed one. It exists because "this
// source has no explicit routes at all" is a different state from "every
// explicit route lost", and only the first one lets an implicit candidate win
// unconditionally.
//
// The three fallback backends previously used the rank-blind form
// (`conductance > winner_conductance || count == 0`), which sells a deep
// context match to any sufficiently strong generic prior.
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// Does the procedural implicit mesh participate in this evaluation at all?
//
// This is a decision, not a scheduling detail, and it belonged to exactly one
// backend: the canonical kernel gates its whole implicit fanout on it while
// the three fallbacks never had it. A source whose explicit routes ALL fail
// the admission floor is therefore inert under the canonical law -- it does
// not fall through to its mesh -- and a fallback that skips the gate invents
// a successor for an event the organism's own law leaves silent.
//
// #1236 collapses the authority onto the canonical evaluator; it does not
// relitigate whether the canonical answer is the one we want. That the gate
// makes a weak-route source blind to its own mesh is a real question, and it
// is a #1289/#1177 dynamics question, not a backend-equivalence one. Changing
// it here would have quietly changed the organism's behaviour under cover of
// a refactor.
// ---------------------------------------------------------------------------
__device__ inline bool direct_canonical_implicit_mesh_eligible(
    const DirectCanonicalWinner& winner, std::uint32_t explicit_candidate_count) {
  return winner.conductance_q16 > 0ll || explicit_candidate_count == 0u;
}

__device__ inline bool direct_canonical_implicit_wins(const DirectImplicitCandidate& implicit,
                                                      const DirectCanonicalWinner& winner,
                                                      std::uint32_t explicit_candidate_count) {
  if (implicit.valid == 0u)
    return false;
  if (explicit_candidate_count == 0u)
    return true;
  return winner.context_rank == 0u &&
         static_cast<std::int64_t>(implicit.conductance_q16) > winner.conductance_q16;
}


// ---------------------------------------------------------------------------
// #1236 rung 2 -- successor CONSTRUCTION is part of the causal law too.
//
// Rung 1 collapsed "which route wins". That left a second, quieter fork: given
// the same winner, the backends built DIFFERENT successors from it. The fabric
// reference copied the incoming event and overwrote three fields, which meant
// it silently dropped everything the canonical kernel derives:
//
//   horizon            canonical adds route.delay; the fallback kept the
//                      predecessor's horizon, so authoritative Recipe timing
//                      never reached the successor at all
//   history_signature  canonical advances it (this is what the NEXT hop's
//                      context rank is computed against); the fallback carried
//                      the predecessor's signature forward unchanged, so a
//                      trajectory through the fabric could never deepen its
//                      context
//   word               canonical routes the learned/target word only when the
//                      target is motor or world-consequence and the route
//                      carries kRouteFlagLearnedOutput; the fallback always
//                      wrote the route's learned_output_word
//   motor event        canonical emits one when the target is a motor node;
//                      the fallback emitted none for a routed successor
//
// A backend may choose how it finds the winner. It may not choose what the
// winner MEANS.
// ---------------------------------------------------------------------------

__device__ inline std::uint32_t direct_canonical_predecessor_signature(std::uint32_t node,
                                                                       Word word) {
  return mix_signature((static_cast<std::uint64_t>(node) << 32) | word);
}

// #1166: the `history_signature` the NEXT stream contact after this event will
// carry, derived from this event alone -- `append_one_event` stamps a
// same-stream contact with `packed_context_signature(rolling_history, ...)` and
// then folds this contact in, and `rolling_history` before that fold is the high
// half of this event's own signature. The exact settlement kernel needs it
// because a record stores a MIXED context signature of its source while the
// settling contact carries a RAW packed one; see the history_ok comment in
// `direct_adult_legacy_oracle.cu` for why that mattered. The first contact after
// a membrane boundary carries no signature and seeds the accumulator unfolded;
// that case is exact here, not approximated, and unambiguous because
// `direct_canonical_predecessor_signature` never returns zero.
__device__ inline std::uint64_t direct_canonical_successor_history_signature(
    std::uint64_t event_history_signature, std::uint32_t node, Word word) {
  const std::uint32_t shallow = direct_canonical_predecessor_signature(node, word);
  const std::uint32_t rolling_before =
      static_cast<std::uint32_t>(event_history_signature >> 32);
  const std::uint32_t rolling_after =
      event_history_signature == 0u
          ? shallow
          : mix_signature((static_cast<std::uint64_t>(rolling_before) << 32) | shallow);
  return (static_cast<std::uint64_t>(rolling_after) << 32) | shallow;
}

// The successor's own context signature, derived from the predecessor's. Deep
// history is only carried when the predecessor already had some; a first
// contact starts the chain at its shallow value rather than hashing zero.
__device__ inline std::uint64_t direct_canonical_successor_history(const ActivityEvent& event) {
  const std::uint32_t predecessor_deep = static_cast<std::uint32_t>(event.history_signature >> 32);
  const std::uint32_t successor_shallow =
      direct_canonical_predecessor_signature(event.node, event.word);
  const std::uint32_t successor_deep =
      predecessor_deep == 0u
          ? successor_shallow
          : mix_signature((static_cast<std::uint64_t>(predecessor_deep) << 32) | successor_shallow);
  return (static_cast<std::uint64_t>(successor_deep) << 32) | successor_shallow;
}

// One committed effect of a logical evaluation: the frontier successor, plus
// the efferent consequence if the target is a motor node.
struct DirectCanonicalEffect {
  ActivityEvent successor;
  MotorEvent motor;
  bool motor_valid;
};

__device__ inline DirectCanonicalEffect direct_canonical_explicit_effect(
    const DirectBrainV01& brain, const ActivityEvent& event, std::uint32_t resolved_cue_node,
    std::uint32_t winner_slot) {
  DirectCanonicalEffect effect{};
  effect.motor_valid = false;
  const DirectRoute chosen = brain.routes[winner_slot];
  const DirectNode target = brain.nodes[chosen.target];
  const Word routed_word = (chosen.flags & kRouteFlagLearnedOutput) != 0u
                               ? chosen.learned_output_word
                               : target.output_word;
  ActivityEvent successor{};
  successor.node = chosen.target;
  successor.word = (target.flags & (kNodeFlagMotor | kNodeFlagWorldConsequence)) != 0u
                       ? routed_word
                       : event.word;
  successor.origin = CausalOrigin::endogenous_prediction;
  successor.context = event.context;
  successor.cue_node = resolved_cue_node;
  successor.source_id = event.source_id;
  successor.external_root = event.external_root;
  successor.horizon = event.horizon + chosen.delay;
  successor.history_signature = direct_canonical_successor_history(event);
  effect.successor = successor;
  if ((target.flags & kNodeFlagMotor) != 0u) {
    effect.motor = MotorEvent{chosen.target,       target.output_channel, routed_word,
                              event.context,       event.external_root,   resolved_cue_node};
    effect.motor_valid = true;
  }
  return effect;
}

__device__ inline DirectCanonicalEffect direct_canonical_implicit_effect(
    const DirectBrainV01& brain, const ActivityEvent& event, std::uint32_t resolved_cue_node,
    const DirectImplicitCandidate& implicit) {
  DirectCanonicalEffect effect{};
  effect.motor_valid = false;
  const DirectNode implicit_target = brain.nodes[implicit.target];
  const Word implicit_routed_word =
      (implicit_target.flags & (kNodeFlagMotor | kNodeFlagWorldConsequence)) != 0u
          ? implicit_target.output_word
          : event.word;
  ActivityEvent successor{};
  successor.node = implicit.target;
  successor.word = implicit_routed_word;
  successor.origin = CausalOrigin::endogenous_prediction;
  successor.context = event.context;
  successor.cue_node = resolved_cue_node;
  successor.source_id = event.source_id;
  successor.external_root = event.external_root;
  successor.horizon = event.horizon + implicit.delay;
  successor.history_signature = direct_canonical_successor_history(event);
  effect.successor = successor;
  if ((implicit_target.flags & kNodeFlagMotor) != 0u) {
    effect.motor = MotorEvent{implicit.target,     implicit_target.output_channel,
                              implicit_routed_word, event.context,
                              event.external_root,  resolved_cue_node};
    effect.motor_valid = true;
  }
  return effect;
}

}  // namespace substrate::direct_adult

#endif  // HARDWARE_NATIVE_DIRECT_CANONICAL_EVALUATOR_DEVICE_CUH
