#pragma once

#include "bcc32_resident_learned_cost_search.cuh"

// mixed_provenance closes its producer-authority declarations by including
// open_inquiry, whose selection layer consults this pending-means API.  Keep
// the small shared interface visible before entering that include cycle; the
// implementations remain below once mixed provenance is complete.
namespace substrate::bcc32::causal_rewrite::pending_means {

#if defined(__CUDACC__)
#define BCC32_PENDING_SELECTION_DISPATCH \
  [[maybe_unused]] static __host__ __device__ __noinline__
#else
#define BCC32_PENDING_SELECTION_DISPATCH [[maybe_unused]] inline
#endif

inline constexpr std::uint32_t kTrajectoryPendingMeans = 8u;

enum class Predicate : std::uint32_t { no = 0u, yes = 1u, malformed = 2u };

__host__ __device__ Predicate immature_actionable_fixed_route(
    const ResidentRewriteState* state, std::uint32_t producer_locus,
    std::uint32_t target_word);

__host__ __device__ bool pending_header_target(
    const ResidentRewriteState* state, const Record& header,
    std::uint32_t* target, std::uint32_t* producer);

}  // namespace substrate::bcc32::causal_rewrite::pending_means

#include "bcc32_resident_mixed_provenance_evidence.cuh"

namespace substrate::bcc32::causal_rewrite::pending_means {

// A retained ordinary trajectory whose final generated target has an immature
// but actionable fixed predecessor route.  This is a header state, never a
// second persistent object or a semantic request identifier.
// A retained trajectory whose already-public action is a strict prefix of one
// unique mature Program. Its event count is the resident causal cursor.
inline constexpr std::uint32_t kTrajectoryPendingPlan = 16u;
// Resident readiness for one retained goal to enter bounded compatible-action
// arbitration. The production END dispatcher rederives this bit from live
// causal selections; a caller can withdraw it but cannot grant an action.
inline constexpr std::uint32_t kTrajectoryGoalNegotiationConsent = 32u;
inline constexpr std::uint32_t kFormGoalNegotiationCommit = 0x72a3d4b1u;
inline constexpr std::uint32_t kFormGoalNegotiationCommitRevision =
    0x72a3d4b2u;
inline constexpr std::uint32_t kFormGoalNegotiationCommitPrograms =
    0x72a3d4b3u;

// A transient, read-only answer to "which retained means can act now?" Every
// field is resident provenance, not a host request. Record revisions make a
// selection stale as soon as its pending owner, target producer, or selected
// action Program changes.
struct CausalSelection {
  std::uint32_t pending_header = kInvalid;
  std::uint32_t pending_header_revision = 0u;
  std::uint32_t producer_program = kInvalid;
  std::uint32_t producer_revision = 0u;
  std::uint32_t target_word = 0u;
  std::uint32_t trajectory_extent = 0u;
  std::uint32_t action_program = kInvalid;
  std::uint32_t action_revision = 0u;
  std::uint32_t action_word = 0u;
  std::uint64_t route_cost = 0u;
  std::uint32_t route_depth = 0u;
};

enum class CausalSelectionStatus : std::uint32_t {
  unique = 0u,
  absent = 1u,
  conflict = 2u,
  fail_closed = 3u,
};

struct GoalNegotiation {
  CausalSelection first{};
  CausalSelection second{};
  // A bounded third consenting goal is an exact nonselected witness. Its
  // incompatible action is what makes one pair uniquely admissible.
  CausalSelection nonselected{};
  std::uint32_t first_goal_owner = 0u;
  std::uint32_t second_goal_owner = 0u;
  std::uint32_t nonselected_goal_owner = 0u;
  std::uint32_t participant_count = 0u;
  std::uint32_t agreed_action_word = 0u;
};

enum class GoalNegotiationStatus : std::uint32_t {
  unique = 0u,
  absent = 1u,
  compatible_ambiguous = 2u,
  fail_closed = 3u,
};

__host__ __device__ inline bool same_causal_selection(
    const CausalSelection& left, const CausalSelection& right) {
  return left.pending_header == right.pending_header &&
         left.pending_header_revision == right.pending_header_revision &&
         left.producer_program == right.producer_program &&
         left.producer_revision == right.producer_revision &&
         left.target_word == right.target_word &&
         left.trajectory_extent == right.trajectory_extent &&
         left.action_program == right.action_program &&
         left.action_revision == right.action_revision &&
         left.action_word == right.action_word &&
         left.route_cost == right.route_cost &&
         left.route_depth == right.route_depth;
}

__host__ __device__ inline bool same_goal_negotiation(
    const GoalNegotiation& left, const GoalNegotiation& right) {
  return same_causal_selection(left.first, right.first) &&
         same_causal_selection(left.second, right.second) &&
         same_causal_selection(left.nonselected, right.nonselected) &&
         left.first_goal_owner == right.first_goal_owner &&
         left.second_goal_owner == right.second_goal_owner &&
         left.nonselected_goal_owner == right.nonselected_goal_owner &&
         left.participant_count == right.participant_count &&
         left.agreed_action_word == right.agreed_action_word;
}

__host__ __device__ inline Predicate immature_actionable_fixed_route(
    const ResidentRewriteState* state, std::uint32_t producer_locus, std::uint32_t target_word) {
  if (state == nullptr || producer_locus == kInvalid || producer_locus >= live_record_capacity(state))
    return Predicate::malformed;
  bool found = false;
  for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
    const Record& program = state->records[slot];
    if (slot == producer_locus || program.matter_q8 == 0u || program.lane[0] != kFormProgram ||
        (program.lane[7] & kProgramFlagVersionSpace) != 0u)
      continue;
    if (!learned_cost_search::canonical_program_shape(program))
      continue;
    if (program.lane[3] == 0u || program.lane[3] >= kProgramMatureSupport)
      continue;
    std::uint32_t predecessor = 0u;
    std::uint32_t predecessor_meta = 0u;
    std::uint32_t outcome = 0u;
    std::uint32_t outcome_meta = 0u;
    const std::uint32_t final_predecessor = program.lane[2] - 2u;
    const std::uint32_t final_outcome = program.lane[2] - 1u;
    // This predicate intentionally inspects an immature ordinary Program.
    // resident_program_term_at requires canonical maturity, so the ordinary
    // fixed decoder is the correct subtype-local view here.
    if (!program_term_at(state, program.lane[1], final_predecessor, &predecessor,
                         &predecessor_meta) ||
        !program_term_at(state, program.lane[1], final_outcome, &outcome, &outcome_meta))
      return Predicate::malformed;
    if (outcome == target_word && (predecessor_meta != 0u || outcome_meta != 0u))
      return Predicate::malformed;
    if (outcome == target_word && predecessor_meta == 0u && outcome_meta == 0u &&
        (predecessor & kRawChannelMask) == (1u << 24u))
      found = true;
  }
  return found ? Predicate::yes : Predicate::no;
}

__host__ __device__ inline bool pending_header_target(const ResidentRewriteState* state,
                                                      const Record& header, std::uint32_t* target,
                                                      std::uint32_t* producer) {
  if (state == nullptr || target == nullptr || producer == nullptr || header.matter_q8 == 0u ||
      header.lane[0] != kFormTrajectory || header.lane[2] == 0u ||
      !mixed_provenance::tagged_history(state, header))
    return false;
  const std::uint32_t index = header.lane[2] - 1u;
  if (!trajectory_word_at(state, header.lane[1], index, target))
    return false;
  mixed_provenance::Origin origin = mixed_provenance::Origin::external;
  if (!mixed_provenance::origin_at(state, header, index, &origin, producer) ||
      origin != mixed_provenance::Origin::generated)
    return false;
  return *producer != kInvalid && *producer < live_record_capacity(state);
}

__host__ __device__ inline bool retain_current_if_unresolved(ResidentRewriteState* state) {
  if (state == nullptr || state->generated_word_valid == 0u || state->generated_locus == kInvalid)
    return false;
  const std::uint32_t slot = find_current_trajectory(state);
  if (slot == kInvalid)
    return false;
  Record& header = state->records[slot];
  std::uint32_t target = 0u;
  std::uint32_t producer = kInvalid;
  if (!pending_header_target(state, header, &target, &producer) ||
      target != state->generated_word || producer != state->generated_locus)
    return false;
  const Predicate predicate = immature_actionable_fixed_route(state, producer, target);
  if (predicate != Predicate::yes)
    return false;
  header.lane[3] = 1u;
  header.lane[4] = 0u;
  header.lane[7] |= kTrajectoryPendingMeans;
  ++header.revision;
  // Observer-only prior activity evidence for a later generic physical assay.
  // This does not select the route or affect execution.
  state->active_locus = slot;
  return true;
}

__host__ __device__ inline bool generated_event_allocation_requirement(
    const ResidentRewriteState* state, const Record& header,
    std::uint32_t producer_locus, std::uint32_t* required) {
  if (state == nullptr || required == nullptr || header.matter_q8 == 0u ||
      header.lane[0] != kFormTrajectory ||
      header.lane[2] == 0u ||
      !resident_program_authoritative(state, producer_locus))
    return false;
  // The term/provenance blocks are addressed by their resident ordinal and
  // may continue onto later physical pages.  This reader only reserves the
  // next block; the old 512-event guard confused the page work aperture with
  // a semantic trajectory limit and rejected otherwise valid long streams.
  *required = 0u;
  const std::uint32_t ordinal = header.lane[2] / 2u;
  if (find_owned_block(state, kFormTrajectoryTerm, header.lane[1], ordinal) == kInvalid)
    ++*required;
  const std::uint32_t provenance =
      find_owned_block(state, mixed_provenance::kFormTrajectoryProvenance, header.lane[1], ordinal);
  if (provenance == kInvalid) {
    ++*required;
  } else {
    const Record& record = state->records[provenance];
    const std::uint32_t local = header.lane[2] % 2u;
    if ((record.lane[kProvenanceValidityLane] & ~0x3u) != 0u ||
        (record.lane[kProvenanceValidityLane] &
         mixed_provenance::valid_bit(local)) != 0u)
      return false;
  }
  return true;
}

__host__ __device__ inline bool can_append_generated_event(
    const ResidentRewriteState* state, const Record& header,
    std::uint32_t producer_locus) {
  std::uint32_t required = 0u;
  if (!generated_event_allocation_requirement(
          state, header, producer_locus, &required))
    return false;
  for (std::uint32_t slot = 0u; slot < live_record_capacity(state) && required != 0u; ++slot)
    if (state->records[slot].matter_q8 != 0u && state->records[slot].lane[0] == kFormEmpty)
      --required;
  return required == 0u;
}

__host__ __device__ inline CausalSelectionStatus select_plan_continuation(
    const ResidentRewriteState* state, const Record& header,
    ProgramCandidateConsensus* candidate) {
  if (state == nullptr || candidate == nullptr || header.matter_q8 == 0u ||
      header.lane[0] != kFormTrajectory || header.lane[2] == 0u)
    return CausalSelectionStatus::fail_closed;
  *candidate = ProgramCandidateConsensus{};
  collect_version_space_program_candidates(state, header, candidate);
  collect_resident_span_program_candidates(state, header, candidate);
  bool saw_unbound = false;
  collect_fixed_program_candidates(state, header, candidate, &saw_unbound);
  if (saw_unbound || candidate->span_saw_unbound ||
      candidate->span_saw_ambiguous)
    return CausalSelectionStatus::fail_closed;
  if (candidate->conflict)
    return CausalSelectionStatus::conflict;
  if (!candidate->have_candidate)
    return CausalSelectionStatus::absent;
  const std::uint32_t channel = candidate->word & kRawChannelMask;
  if (candidate->selected_from_span) {
    // A retained plan may advance through learned language before its eventual
    // action only when one mature, fully-bound SpanProgram owns the exact
    // cursor. Coextensive non-span matter is a distinct authority surface.
    if (!candidate->span_contributed || !candidate->span_cursor_valid ||
        candidate->span_cursor_program != candidate->diagnostic_locus ||
        candidate->span_cursor_program == kInvalid ||
        candidate->saw_nonspan_at_selected_extent ||
        candidate->selected_support < kSpanProgramMatureSupport)
      return CausalSelectionStatus::fail_closed;
    if (channel != 0u && channel != (1u << 24u))
      return CausalSelectionStatus::absent;
    return CausalSelectionStatus::unique;
  }
  if (candidate->span_contributed || channel != (1u << 24u))
    return CausalSelectionStatus::absent;
  return CausalSelectionStatus::unique;
}

__host__ __device__ inline CausalSelectionStatus select_header_causal_action(
    const ResidentRewriteState* state, std::uint32_t slot,
    CausalSelection* selected) {
  if (selected == nullptr || state == nullptr || slot == kInvalid ||
      slot >= live_record_capacity(state))
    return CausalSelectionStatus::fail_closed;
  *selected = CausalSelection{};
  const Record& header = state->records[slot];
  if (header.matter_q8 == 0u || header.lane[0] != kFormTrajectory ||
      header.lane[3] != 1u ||
      (header.lane[7] &
       (kTrajectoryPendingMeans | kTrajectoryPendingPlan)) == 0u)
    return CausalSelectionStatus::absent;
  std::uint32_t target = 0u;
  std::uint32_t producer = kInvalid;
  if (!pending_header_target(state, header, &target, &producer))
    return CausalSelectionStatus::fail_closed;
  if (!resident_program_authoritative(state, producer)) {
    producer = unique_revision_product_for_base(state, producer);
    if (producer == kInvalid) return CausalSelectionStatus::fail_closed;
  }
  std::uint32_t action_program = kInvalid;
  std::uint32_t action_word = 0u;
  std::uint64_t route_cost = 0u;
  std::uint32_t route_depth = 1u;
  if ((header.lane[7] & kTrajectoryPendingPlan) != 0u) {
    ProgramCandidateConsensus plan{};
    const CausalSelectionStatus status =
        select_plan_continuation(state, header, &plan);
    if (status != CausalSelectionStatus::unique)
      return status;
    action_program = plan.diagnostic_locus;
    action_word = plan.word;
  } else {
    learned_cost_search::Route candidate{};
    const learned_cost_search::SelectionStatus status =
        learned_cost_search::select_resident_learned_cost_route(
            state, producer, target, &candidate);
    if (status == learned_cost_search::SelectionStatus::fail_closed)
      return CausalSelectionStatus::fail_closed;
    if (status != learned_cost_search::SelectionStatus::unique)
      return CausalSelectionStatus::absent;
    if (candidate.depth == 0u)
      return CausalSelectionStatus::fail_closed;
    action_program = candidate.action_program;
    action_word = candidate.action_word;
    route_cost = candidate.total_cost;
    route_depth = candidate.depth;
  }
  if (action_program == kInvalid || action_program >= live_record_capacity(state) ||
      !resident_program_authoritative(state, action_program))
    return CausalSelectionStatus::fail_closed;
  selected->pending_header = slot;
  selected->pending_header_revision = header.revision;
  selected->producer_program = producer;
  selected->producer_revision = state->records[producer].revision;
  selected->target_word = target;
  selected->trajectory_extent = header.lane[2];
  selected->action_program = action_program;
  selected->action_revision = state->records[action_program].revision;
  selected->action_word = action_word;
  selected->route_cost = route_cost;
  selected->route_depth = route_depth;
  return CausalSelectionStatus::unique;
}

// Observer-only canonical selection. Every retained owner is scanned and no
// Record, receipt, clock, or public register is changed on absence, malformed
// provenance, or conflict.
BCC32_PENDING_SELECTION_DISPATCH CausalSelectionStatus select_causal_action(
    const ResidentRewriteState* state, CausalSelection* selected) {
  if (selected == nullptr || state == nullptr || state->fault != 0u)
    return CausalSelectionStatus::fail_closed;
  *selected = CausalSelection{};
  if (find_current_trajectory(state) != kInvalid)
    return CausalSelectionStatus::absent;
  bool blocked = false;
  for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
    CausalSelection observed{};
    const CausalSelectionStatus status =
        select_header_causal_action(state, slot, &observed);
    if (status == CausalSelectionStatus::absent)
      continue;
    if (status == CausalSelectionStatus::fail_closed) {
      blocked = true;
      continue;
    }
    if (status == CausalSelectionStatus::conflict) {
      *selected = CausalSelection{};
      return status;
    }
    if (selected->pending_header != kInvalid) {
      *selected = CausalSelection{};
      return CausalSelectionStatus::conflict;
    }
    *selected = observed;
  }
  if (blocked) {
    *selected = CausalSelection{};
    return CausalSelectionStatus::fail_closed;
  }
  return selected->pending_header == kInvalid ? CausalSelectionStatus::absent
                                              : CausalSelectionStatus::unique;
}

// A control may withdraw readiness, but a grant succeeds only when the exact
// live resident header independently selects one unique causal action. The
// production dispatcher calls the population refresh below, so this Boolean
// cannot manufacture action authority or make arbitration test-only.
__host__ __device__ inline bool record_goal_negotiation_readiness(
    ResidentRewriteState* state, std::uint32_t pending_header, bool granted) {
  if (state == nullptr || pending_header == kInvalid ||
      pending_header >= live_record_capacity(state))
    return false;
  Record& header = state->records[pending_header];
  if (header.matter_q8 == 0u || header.lane[0] != kFormTrajectory ||
      header.lane[3] != 1u ||
      (header.lane[7] &
       (kTrajectoryPendingMeans | kTrajectoryPendingPlan)) == 0u)
    return false;
  const bool current =
      (header.lane[7] & kTrajectoryGoalNegotiationConsent) != 0u;
  if (granted) {
    CausalSelection observed{};
    if (select_header_causal_action(state, pending_header, &observed) !=
        CausalSelectionStatus::unique)
      return false;
  }
  if (current == granted) return true;
  if (header.revision == ~std::uint32_t{0}) return false;
  if (granted)
    header.lane[7] |= kTrajectoryGoalNegotiationConsent;
  else
    header.lane[7] &= ~kTrajectoryGoalNegotiationConsent;
  ++header.revision;
  return true;
}

// Every physical END revalidates the complete pending population. Readiness is
// changed only when the residently derived result changed; a quiet revalidation
// is observer-silent and cannot stale an otherwise exact negotiation receipt.
// The two-pass preflight prevents revision saturation from partially changing
// the population.
__host__ __device__ inline bool refresh_goal_negotiation_readiness(
    ResidentRewriteState* state) {
  if (state == nullptr || state->fault != 0u) return false;
  for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
    const Record& header = state->records[slot];
    if (header.matter_q8 == 0u || header.lane[0] != kFormTrajectory ||
        (header.lane[7] &
         (kTrajectoryPendingMeans | kTrajectoryPendingPlan)) == 0u)
      continue;
    CausalSelection observed{};
    const bool desired =
        select_header_causal_action(state, slot, &observed) ==
        CausalSelectionStatus::unique;
    const bool current =
        (header.lane[7] & kTrajectoryGoalNegotiationConsent) != 0u;
    if (desired != current && header.revision == ~std::uint32_t{0})
      return false;
  }
  for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
    Record& header = state->records[slot];
    if (header.matter_q8 == 0u || header.lane[0] != kFormTrajectory ||
        (header.lane[7] &
         (kTrajectoryPendingMeans | kTrajectoryPendingPlan)) == 0u)
      continue;
    CausalSelection observed{};
    const bool desired =
        select_header_causal_action(state, slot, &observed) ==
        CausalSelectionStatus::unique;
    const bool current =
        (header.lane[7] & kTrajectoryGoalNegotiationConsent) != 0u;
    if (desired == current) continue;
    if (desired)
      header.lane[7] |= kTrajectoryGoalNegotiationConsent;
    else
      header.lane[7] &= ~kTrajectoryGoalNegotiationConsent;
    ++header.revision;
  }
  return true;
}

// A goal's stable resident identity is the learned Program owner that produced
// the first generated event in its retained causal history. Later plan actions
// therefore cannot silently replace the goal identity used for negotiation.
__host__ __device__ inline bool negotiation_goal_owner(
    const ResidentRewriteState* state, const CausalSelection& selection,
    std::uint32_t* owner) {
  if (state == nullptr || owner == nullptr ||
      selection.pending_header == kInvalid ||
      selection.pending_header >= live_record_capacity(state))
    return false;
  const Record& header = state->records[selection.pending_header];
  if (header.revision != selection.pending_header_revision ||
      header.lane[2] != selection.trajectory_extent ||
      !mixed_provenance::tagged_history(state, header))
    return false;
  for (std::uint32_t index = 0u; index < header.lane[2]; ++index) {
    mixed_provenance::Origin origin = mixed_provenance::Origin::external;
    std::uint32_t producer = kInvalid;
    if (!mixed_provenance::origin_at(state, header, index, &origin,
                                     &producer))
      return false;
    if (origin != mixed_provenance::Origin::generated)
      continue;
    if (producer == kInvalid || producer >= live_record_capacity(state) ||
        state->records[producer].matter_q8 == 0u ||
        state->records[producer].lane[0] != kFormProgram ||
        state->records[producer].lane[1] == 0u)
      return false;
    *owner = state->records[producer].lane[1];
    return true;
  }
  return false;
}

// Two or three resident-ready goals may negotiate. With three,
// exactly one equal-action pair is authorized; zero compatible pairs or the
// three equal alternatives remain compatible-action ambiguous. Learned producer owners
// provide allocation-independent ordering and the third goal remains bound in
// the receipt even though it is not selected.
__host__ __device__ inline GoalNegotiationStatus select_goal_negotiation(
    const ResidentRewriteState* state, GoalNegotiation* negotiation) {
  if (state == nullptr || negotiation == nullptr || state->fault != 0u)
    return GoalNegotiationStatus::fail_closed;
  *negotiation = GoalNegotiation{};
  if (find_current_trajectory(state) != kInvalid)
    return GoalNegotiationStatus::absent;
  CausalSelection observed[3]{};
  std::uint32_t count = 0u;
  bool unavailable = false;
  bool blocked = false;
  for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
    const Record& header = state->records[slot];
    if (header.matter_q8 == 0u || header.lane[0] != kFormTrajectory ||
        (header.lane[7] & kTrajectoryGoalNegotiationConsent) == 0u)
      continue;
    CausalSelection goal{};
    const CausalSelectionStatus status =
        select_header_causal_action(state, slot, &goal);
    if (status == CausalSelectionStatus::absent)
      unavailable = true;
    else if (status != CausalSelectionStatus::unique)
      blocked = true;
    else if (count < 3u)
      observed[count] = goal;
    ++count;
  }
  if (count > 3u)
    return GoalNegotiationStatus::compatible_ambiguous;
  if (blocked)
    return GoalNegotiationStatus::fail_closed;
  if (unavailable)
    return GoalNegotiationStatus::absent;
  if (count < 2u)
    return GoalNegotiationStatus::absent;
  std::uint32_t owners[3]{};
  for (std::uint32_t index = 0u; index < count; ++index)
    if (!negotiation_goal_owner(state, observed[index], &owners[index]))
      return GoalNegotiationStatus::fail_closed;
  if (owners[0] == owners[1] ||
      (count == 3u &&
       (owners[0] == owners[2] || owners[1] == owners[2])))
    return GoalNegotiationStatus::fail_closed;

  std::uint32_t first_index = 0u;
  std::uint32_t second_index = 1u;
  std::uint32_t nonselected_index = kInvalid;
  if (count == 2u) {
    if (observed[0].action_word != observed[1].action_word)
      return GoalNegotiationStatus::compatible_ambiguous;
  } else {
    std::uint32_t compatible_pairs = 0u;
    for (std::uint32_t left = 0u; left < 3u; ++left)
      for (std::uint32_t right = left + 1u; right < 3u; ++right)
        if (observed[left].action_word == observed[right].action_word) {
          ++compatible_pairs;
          first_index = left;
          second_index = right;
          nonselected_index = 3u - left - right;
        }
    if (compatible_pairs != 1u) {
      *negotiation = GoalNegotiation{};
      return GoalNegotiationStatus::compatible_ambiguous;
    }
  }
  if (owners[second_index] < owners[first_index]) {
    const std::uint32_t temporary = first_index;
    first_index = second_index;
    second_index = temporary;
  }
  // Canonical extent is a semantic priority key. A joint candidate may not
  // borrow whichever participant extent happens to sort first by goal owner.
  if (observed[first_index].trajectory_extent !=
      observed[second_index].trajectory_extent) {
    *negotiation = GoalNegotiation{};
    return GoalNegotiationStatus::compatible_ambiguous;
  }
  negotiation->first = observed[first_index];
  negotiation->second = observed[second_index];
  negotiation->first_goal_owner = owners[first_index];
  negotiation->second_goal_owner = owners[second_index];
  negotiation->participant_count = count;
  if (nonselected_index != kInvalid) {
    negotiation->nonselected = observed[nonselected_index];
    negotiation->nonselected_goal_owner = owners[nonselected_index];
  }
  negotiation->agreed_action_word = negotiation->first.action_word;
  return GoalNegotiationStatus::unique;
}

// Re-derive the entire ready set, including a bounded nonselected third
// witness, before admitting the selected pair's shared action to the ordinary
// candidate law. This produces no output and deliberately does not masquerade
// as a single-owner pending-means commit receipt.
__host__ __device__ inline bool merge_goal_negotiation_candidate(
    const ResidentRewriteState* state, const GoalNegotiation& negotiation,
    ProgramCandidateConsensus* consensus) {
  if (state == nullptr || consensus == nullptr ||
      negotiation.first.pending_header == kInvalid ||
      negotiation.second.pending_header == kInvalid)
    return false;
  GoalNegotiation observed{};
  if (select_goal_negotiation(state, &observed) !=
          GoalNegotiationStatus::unique ||
      !same_goal_negotiation(negotiation, observed))
    return false;
  merge_program_candidate(consensus, observed.agreed_action_word,
                          observed.first.action_program, false,
                          observed.first.trajectory_extent);
  return consensus->have_candidate && !consensus->conflict &&
         consensus->word == observed.agreed_action_word &&
         consensus->diagnostic_locus == observed.first.action_program;
}

__host__ __device__ inline bool retain_plan_if_continuable(
    ResidentRewriteState* state, std::uint32_t header_slot);

__host__ __device__ inline bool plain_program_consensus(
    const ProgramCandidateConsensus& consensus) {
  if (!consensus.have_candidate || consensus.conflict ||
      consensus.selected_support != 0u || consensus.selected_from_span ||
      !consensus.saw_nonspan_at_selected_extent ||
      consensus.support_resolved_distinct_span || consensus.span_contributed ||
      consensus.span_saw_program || consensus.span_saw_unbound ||
      consensus.span_saw_ambiguous || consensus.version_space_saw_program ||
      consensus.selected_from_pending_means || consensus.span_cursor_valid ||
      consensus.pending_header != kInvalid ||
      consensus.pending_producer != kInvalid ||
      consensus.pending_route_cost != 0u ||
      consensus.pending_route_depth != 0u ||
      consensus.span_cursor_program != kInvalid ||
      consensus.span_cursor_term != kInvalid ||
      consensus.span_cursor_start != 0u ||
      consensus.span_cursor_offset != 0u)
    return false;
  for (std::uint32_t index = 0u; index < kMaximumProgramVariables; ++index)
    if (consensus.span_cursor_lengths[index] != 0u) return false;
  return true;
}

__host__ __device__ inline bool pending_program_consensus(
    const ProgramCandidateConsensus& consensus) {
  if (!consensus.have_candidate || consensus.conflict ||
      !consensus.selected_from_pending_means ||
      consensus.selected_support != 0u || consensus.selected_from_span ||
      !consensus.saw_nonspan_at_selected_extent ||
      consensus.support_resolved_distinct_span || consensus.span_contributed ||
      consensus.span_saw_program || consensus.span_saw_unbound ||
      consensus.span_saw_ambiguous || consensus.version_space_saw_program ||
      consensus.span_cursor_valid || consensus.pending_header == kInvalid ||
      consensus.pending_producer == kInvalid ||
      consensus.pending_route_depth == 0u ||
      consensus.span_cursor_program != kInvalid ||
      consensus.span_cursor_term != kInvalid ||
      consensus.span_cursor_start != 0u ||
      consensus.span_cursor_offset != 0u)
    return false;
  for (std::uint32_t index = 0u; index < kMaximumProgramVariables; ++index)
    if (consensus.span_cursor_lengths[index] != 0u) return false;
  return true;
}

__host__ __device__ inline std::uint32_t install_negotiation_commit_receipt(
    ResidentRewriteState* state, const GoalNegotiation& observed) {
  if (state == nullptr || observed.participant_count < 2u ||
      observed.participant_count > 3u)
    return kInvalid;
  const std::uint32_t header_slot = allocate_record(state);
  const std::uint32_t revision_slot = allocate_record(state);
  const std::uint32_t programs_slot = allocate_record(state);
  if (header_slot == kInvalid || revision_slot == kInvalid ||
      programs_slot == kInvalid)
    return kInvalid;
  Record& header = state->records[header_slot];
  header.lane[0] = kFormGoalNegotiationCommit;
  header.lane[1] = revision_slot;
  header.lane[2] = programs_slot;
  header.lane[3] = observed.participant_count;
  header.lane[4] = observed.agreed_action_word;
  header.lane[5] = observed.first_goal_owner;
  header.lane[6] = observed.second_goal_owner;
  header.lane[7] = observed.nonselected_goal_owner;
  ++header.revision;

  Record& revisions = state->records[revision_slot];
  revisions.lane[0] = kFormGoalNegotiationCommitRevision;
  revisions.lane[1] = header_slot;
  revisions.lane[2] = observed.first.pending_header_revision;
  revisions.lane[3] = observed.second.pending_header_revision;
  revisions.lane[4] = observed.nonselected.pending_header_revision;
  revisions.lane[5] = observed.first.action_revision;
  revisions.lane[6] = observed.second.action_revision;
  revisions.lane[7] = observed.nonselected.action_revision;
  ++revisions.revision;

  Record& programs = state->records[programs_slot];
  programs.lane[0] = kFormGoalNegotiationCommitPrograms;
  programs.lane[1] = header_slot;
  programs.lane[2] = observed.first.action_program;
  programs.lane[3] = observed.second.action_program;
  programs.lane[4] = observed.nonselected.action_program;
  programs.lane[5] = observed.first.producer_program;
  programs.lane[6] = observed.second.producer_program;
  programs.lane[7] = observed.nonselected.producer_program;
  ++programs.revision;
  return header_slot;
}

// Commit one jointly authorized public action while advancing both selected
// resident goal histories with their own exact Program producers. The common
// emitter publishes only the canonical first participant; the second append is
// internal causal history, not a duplicate outward action. Every allocation is
// preflighted before either header changes.
__host__ __device__ inline bool commit_goal_negotiation_candidate(
    ResidentRewriteState* state, const GoalNegotiation& negotiation,
    const ProgramCandidateConsensus& consensus) {
  if (state == nullptr || !plain_program_consensus(consensus) ||
      negotiation.first.pending_header == kInvalid ||
      negotiation.second.pending_header == kInvalid ||
      negotiation.first.pending_header == negotiation.second.pending_header)
    return false;
  GoalNegotiation observed{};
  ProgramCandidateConsensus expected{};
  if (select_goal_negotiation(state, &observed) !=
          GoalNegotiationStatus::unique ||
      !same_goal_negotiation(negotiation, observed) ||
      !merge_goal_negotiation_candidate(state, observed, &expected) ||
      consensus.word != expected.word ||
      consensus.diagnostic_locus != expected.diagnostic_locus ||
      consensus.extent != expected.extent ||
      observed.first.trajectory_extent != observed.second.trajectory_extent ||
      state->program_generated_events == ~std::uint32_t{0})
    return false;
  std::uint32_t first_required = 0u;
  std::uint32_t second_required = 0u;
  const Record& first_before =
      state->records[observed.first.pending_header];
  const Record& second_before =
      state->records[observed.second.pending_header];
  if (!generated_event_allocation_requirement(
          state, first_before, observed.first.action_program,
          &first_required) ||
      !generated_event_allocation_requirement(
          state, second_before, observed.second.action_program,
          &second_required))
    return false;
  std::uint32_t free_records = 0u;
  for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot)
    if (state->records[slot].matter_q8 != 0u &&
        state->records[slot].lane[0] == kFormEmpty)
      ++free_records;
  if (free_records < first_required + second_required + 3u ||
      first_before.revision > ~std::uint32_t{0} - 3u ||
      second_before.revision > ~std::uint32_t{0} - 3u)
    return false;

  const std::uint32_t commit_receipt =
      install_negotiation_commit_receipt(state, observed);
  if (commit_receipt == kInvalid) return false;

  Record& first = state->records[observed.first.pending_header];
  first.lane[3] = 0u;
  first.lane[4] = 0u;
  first.lane[7] &= ~(kTrajectoryPendingMeans | kTrajectoryPendingPlan);
  ++first.revision;
  if (!emit_program_candidate_word(state, expected) ||
      !mixed_provenance::mark_last(
          state, mixed_provenance::Origin::generated,
          observed.first.action_program))
    return false;
  first.lane[3] = 1u;
  first.lane[7] &= ~kTrajectoryGoalNegotiationConsent;
  ++first.revision;

  Record& second = state->records[observed.second.pending_header];
  second.lane[3] = 0u;
  second.lane[4] = 0u;
  second.lane[7] &= ~(kTrajectoryPendingMeans | kTrajectoryPendingPlan);
  ++second.revision;
  if (!append_trajectory_word(state, expected.word, true) ||
      !mixed_provenance::mark_last(
          state, mixed_provenance::Origin::generated,
          observed.second.action_program))
    return false;
  second.lane[3] = 1u;
  second.lane[7] &= ~kTrajectoryGoalNegotiationConsent;
  ++second.revision;

  const bool first_continues = retain_plan_if_continuable(
      state, observed.first.pending_header);
  const bool second_continues = retain_plan_if_continuable(
      state, observed.second.pending_header);
  if (!first_continues)
    mixed_provenance::clear_trajectory_and_provenance(
        state, observed.first.pending_header);
  if (!second_continues)
    mixed_provenance::clear_trajectory_and_provenance(
        state, observed.second.pending_header);
  state->active_locus = commit_receipt;
  refresh_receipt(state);
  return true;
}

// Merge only a freshly rederived CausalSelection into the canonical Program
// consensus. The second derivation rejects stale or forged ABI values before
// the consensus, resident matter, or public observer state can be mutated.
__host__ __device__ inline bool merge_causal_selection(
    const ResidentRewriteState* state, const CausalSelection& selection,
    ProgramCandidateConsensus* consensus) {
  if (state == nullptr || consensus == nullptr ||
      selection.pending_header == kInvalid ||
      selection.pending_header >= live_record_capacity(state))
    return false;
  CausalSelection observed{};
  if (select_causal_action(state, &observed) !=
          CausalSelectionStatus::unique ||
      !same_causal_selection(selection, observed))
    return false;
  const Record& header = state->records[selection.pending_header];
  if (selection.trajectory_extent != header.lane[2])
    return false;
  merge_program_candidate(consensus, selection.action_word,
                          selection.action_program, false,
                          selection.trajectory_extent);
  if (!consensus->have_candidate || consensus->conflict ||
      consensus->word != selection.action_word ||
      consensus->diagnostic_locus != selection.action_program ||
      consensus->extent != header.lane[2])
    return false;
  consensus->selected_from_pending_means = true;
  consensus->pending_header = selection.pending_header;
  consensus->pending_header_revision = selection.pending_header_revision;
  consensus->pending_producer = selection.producer_program;
  consensus->pending_producer_revision = selection.producer_revision;
  consensus->pending_target = selection.target_word;
  consensus->pending_action_revision = selection.action_revision;
  consensus->pending_route_cost = selection.route_cost;
  consensus->pending_route_depth = selection.route_depth;
  return true;
}

__host__ __device__ inline bool retain_plan_if_continuable(
    ResidentRewriteState* state, std::uint32_t header_slot) {
  if (state == nullptr || header_slot == kInvalid ||
      header_slot >= live_record_capacity(state))
    return false;
  Record& header = state->records[header_slot];
  ProgramCandidateConsensus continuation{};
  if (select_plan_continuation(state, header, &continuation) !=
      CausalSelectionStatus::unique)
    return false;
  header.lane[3] = 1u;
  header.lane[4] = 0u;
  header.lane[7] |=
      kTrajectoryPendingPlan | kTrajectoryGoalNegotiationConsent;
  ++header.revision;
  return true;
}

// Resume one and only one retained SpanProgram plan through the common
// generated-word rail. Unlike pending means, a plan need not already end in a
// generated target: its authority is the mature, fully-bound SpanProgram and
// exact cursor rederived above. Multiple retained goals remain the negotiation
// path's responsibility and therefore abstain here.
__host__ __device__ inline bool resume_one_pending_plan_at_physical_end(
    ResidentRewriteState* state) {
  if (state == nullptr || state->fault != 0u ||
      find_current_trajectory(state) != kInvalid)
    return false;
  std::uint32_t selected_header = kInvalid;
  ProgramCandidateConsensus selected{};
  for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
    const Record& header = state->records[slot];
    if (header.matter_q8 == 0u || header.lane[0] != kFormTrajectory ||
        (header.lane[7] &
         (kTrajectoryPendingMeans | kTrajectoryPendingPlan)) == 0u)
      continue;
    if (selected_header != kInvalid ||
        (header.lane[7] & kTrajectoryPendingPlan) == 0u)
      return false;
    ProgramCandidateConsensus candidate{};
    if (select_plan_continuation(state, header, &candidate) !=
        CausalSelectionStatus::unique)
      return false;
    selected_header = slot;
    selected = candidate;
  }
  if (selected_header == kInvalid || !selected.selected_from_span ||
      !selected.span_cursor_valid ||
      state->program_generated_events == ~std::uint32_t{0} ||
      state->span_generated_events == ~std::uint32_t{0} ||
      !can_append_generated_event(state, state->records[selected_header],
                                  selected.diagnostic_locus))
    return false;

  Record& header = state->records[selected_header];
  if (header.revision > ~std::uint32_t{0} - 2u) return false;
  header.lane[3] = 0u;
  header.lane[4] = 1u;
  header.lane[7] =
      (header.lane[7] &
       ~(kTrajectoryPendingMeans | kTrajectoryPendingPlan |
         kTrajectoryGoalNegotiationConsent)) |
      kTrajectoryHasGenerated;
  ++header.revision;
  if (!emit_program_candidate_word(state, selected) ||
      !mixed_provenance::mark_last(
          state, mixed_provenance::Origin::generated,
          selected.diagnostic_locus))
    return false;
  const bool cached = install_span_execution_cursor(state, header, selected);
  if (!cached) settle_program_candidates(state, header);
  refresh_receipt(state);
  return true;
}

__host__ __device__ inline bool commit_causal_consensus(
    ResidentRewriteState* state, const ProgramCandidateConsensus& consensus) {
  if (state == nullptr || !pending_program_consensus(consensus))
    return false;
  CausalSelection expected{};
  expected.pending_header = consensus.pending_header;
  expected.pending_header_revision = consensus.pending_header_revision;
  expected.producer_program = consensus.pending_producer;
  expected.producer_revision = consensus.pending_producer_revision;
  expected.target_word = consensus.pending_target;
  expected.trajectory_extent = consensus.extent;
  expected.action_program = consensus.diagnostic_locus;
  expected.action_revision = consensus.pending_action_revision;
  expected.action_word = consensus.word;
  expected.route_cost = consensus.pending_route_cost;
  expected.route_depth = consensus.pending_route_depth;
  CausalSelection observed{};
  if (select_causal_action(state, &observed) !=
          CausalSelectionStatus::unique ||
      !same_causal_selection(expected, observed))
    return false;
  ProgramCandidateConsensus rederived{};
  if (!merge_causal_selection(state, observed, &rederived) ||
      rederived.word != consensus.word ||
      rederived.diagnostic_locus != consensus.diagnostic_locus ||
      rederived.extent != consensus.extent)
    return false;
  Record& header = state->records[expected.pending_header];
  // Preflight every allocation and provenance constraint before changing the
  // retained header. A capacity or malformed-layout failure is an abstention,
  // never a half-resumed cognitive state.
  if (state->program_generated_events == ~std::uint32_t{0} ||
      header.revision > ~std::uint32_t{0} - 2u ||
      !can_append_generated_event(state, header, expected.action_program))
    return false;
  header.lane[3] = 0u;
  header.lane[4] = 0u;
  header.lane[7] =
      (header.lane[7] &
       ~(kTrajectoryPendingMeans | kTrajectoryPendingPlan |
         kTrajectoryGoalNegotiationConsent)) |
      kTrajectoryHasGenerated;
  ++header.revision;
  if (!emit_program_candidate_word(state, rederived) ||
      !mixed_provenance::mark_last(state, mixed_provenance::Origin::generated,
                                   consensus.diagnostic_locus))
    return false;
  (void)retain_plan_if_continuable(state, expected.pending_header);
  refresh_receipt(state);
  return true;
}

// Called only immediately after a physical END. Selection, conflict handling,
// and publication now travel through ProgramCandidateConsensus instead of a
// pending-only generated-word writer.
__host__ __device__ inline bool resume_one_pending_at_physical_end(
    ResidentRewriteState* state) {
  CausalSelection selection{};
  if (select_causal_action(state, &selection) !=
      CausalSelectionStatus::unique)
    return false;
  ProgramCandidateConsensus consensus{};
  return merge_causal_selection(state, selection, &consensus) &&
         commit_causal_consensus(state, consensus);
}

// Production category dispatcher for the physical-END retry boundary. Once
// two or more resident goals have derived negotiation readiness,
// they may only act through the exact joint receipt. Ambiguity, unavailable
// evidence, malformed topology, or a stale participant therefore stays
// silent; it can never fall back to a traversal-selected single goal.
__host__ __device__ inline bool resume_pending_category_at_physical_end(
    ResidentRewriteState* state) {
  if (state == nullptr || state->fault != 0u)
    return false;
  if (resume_one_pending_plan_at_physical_end(state)) return true;
  if (!refresh_goal_negotiation_readiness(state)) return false;
  std::uint32_t pending_goals = 0u;
  std::uint32_t ready_goals = 0u;
  for (std::uint32_t slot = 0u; slot < live_record_capacity(state); ++slot) {
    const Record& record = state->records[slot];
    if (record.matter_q8 == 0u || record.lane[0] != kFormTrajectory ||
        (record.lane[7] &
         (kTrajectoryPendingMeans | kTrajectoryPendingPlan)) == 0u)
      continue;
    ++pending_goals;
    if ((record.lane[7] & kTrajectoryGoalNegotiationConsent) != 0u)
      ++ready_goals;
  }
  if (pending_goals < 2u)
    return resume_one_pending_at_physical_end(state);
  if (ready_goals != pending_goals) return false;

  GoalNegotiation negotiation{};
  if (select_goal_negotiation(state, &negotiation) !=
      GoalNegotiationStatus::unique)
    return false;
  ProgramCandidateConsensus consensus{};
  return merge_goal_negotiation_candidate(state, negotiation, &consensus) &&
         commit_goal_negotiation_candidate(state, negotiation, consensus);
}

}  // namespace substrate::bcc32::causal_rewrite::pending_means

#undef BCC32_PENDING_SELECTION_DISPATCH
