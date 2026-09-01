#ifndef HARDWARE_NATIVE_DIRECT_ADULT_DAMASIO_GROUNDING_CUH
#define HARDWARE_NATIVE_DIRECT_ADULT_DAMASIO_GROUNDING_CUH

// j.damasio_somatic_marker (#1540): evolutionary lens grounding abstract
// reason in homeostatic bodily regulation.
//
// Law anchors (Revision 12):
//   * observer handle, device mechanism: "Damasio" names an interpretation,
//     never a runtime branch. What lives here is bounded deliberative
//     reasoning whose reachable candidate window is shaped by homeostatic
//     regulation state derived on device from settled consequence ledgers
//     (#1517 affect masses) through the #1527 sequence-stamped somatic
//     exploration order;
//   * fields modulate and nominate: regulation decides which slice of the
//     authorized search space fits inside a finite visit budget. It never
//     touches arbitration scores, settled credit, participation identity or
//     provenance; strong contrary evidence still selects against the best
//     nomination, and commitment passes only through the lawful volitional
//     pre-commit gate;
//   * no authority laundering: an episode reads device state and writes only
//     its own receipt; the resident subject's causal ledgers stay read-only;
//   * ablation is the causal witness: severing the regulation input collapses
//     every marker lane to silence and re-runs the identical stamped
//     machinery, so whatever reasoning quality survives the severance is
//     exactly the uncoupled share;
//   * fail closed: episodes refuse zero-identity sources outright and walk
//     only candidates already admitted by a stamped biasing pass -- the
//     ablated control consumes the same stamped machinery with the input
//     severed, never a fabricated shortcut around it.

#include <cstdint>
#include <type_traits>

#include "hardware_native/direct_adult_affect_body.cuh"
#include "hardware_native/direct_adult_somatic_marker.cuh"

namespace substrate::direct_network {

inline constexpr std::uint32_t kDamasioEpisodeCapacity = kSomaticSearchCapacity;

struct alignas(8) DirectDamasioEpisodeReceipt {
  std::uint64_t bias_sequence;
  std::int32_t regulation_q16;
  std::int32_t best_visited_valence_q16;
  std::int32_t solution_valence_q16;
  std::uint32_t visit_order[kDamasioEpisodeCapacity];
  std::int32_t visited_valence_q16[kDamasioEpisodeCapacity];
  std::int32_t visited_scores_q16[kDamasioEpisodeCapacity];
  std::uint32_t budget;
  std::uint32_t admissible_count;
  std::uint32_t visits_used;
  std::uint32_t best_depth;
  std::uint32_t best_slot;
  std::uint32_t solution_slot;
  std::uint32_t wasted_visits_before_positive;
  std::uint32_t grounded_order;
  std::uint32_t found_valuable;
};
static_assert(std::is_trivial_v<DirectDamasioEpisodeReceipt> &&
              std::is_standard_layout_v<DirectDamasioEpisodeReceipt> &&
              std::has_unique_object_representations_v<
                  DirectDamasioEpisodeReceipt>);

struct alignas(8) DirectDamasioSelectionTally {
  std::int64_t grounded_cumulative_valence_q16;
  std::int64_t ablated_cumulative_valence_q16;
  std::uint32_t rounds;
  std::uint32_t grounded_rounds_won;
  std::uint32_t ablated_rounds_won;
  std::uint32_t ties;
};
static_assert(std::is_trivial_v<DirectDamasioSelectionTally> &&
              std::is_standard_layout_v<DirectDamasioSelectionTally> &&
              std::has_unique_object_representations_v<
                  DirectDamasioSelectionTally>);

// Homeostatic regulation state of the whole body table: aggregate vitality
// minus preservation pressure, both device-derived from settled consequence
// ledgers. Positive means enriched, negative means strained.
__device__ inline std::int32_t damasio_regulation_q16(
    const DirectAffectBodyState* affect) {
  if (affect == nullptr) return 0;
  return affect->vitality_aggregate_q16 - affect->preservation_pressure_q16;
}

// The embodied truth of one candidate: its lane's settled-consequence valence
// (vitality against damage and stress). A candidate whose channel carries no
// derived evidence is valence-silent, never invented as good or bad.
__device__ inline std::int32_t damasio_candidate_valence_q16(
    const DirectAffectBodyState* affect,
    const DirectSomaticCandidate& candidate) {
  if (affect == nullptr) return 0;
  const std::int32_t slot = affect_find_entry(affect, candidate.root_channel);
  return affect_channel_bias_q16(slot >= 0 ? &affect->entries[slot] : nullptr);
}

// Ablation: cut the regulation-to-reasoning coupling by collapsing every
// marker lane to homeostatic silence. The bodily evidence itself (affect
// masses, ledgers, tissue) is untouched -- only the influence is severed.
__device__ inline void damasio_sever_regulation(
    DirectSomaticMarkerState* markers) {
  if (markers == nullptr) return;
  for (std::uint32_t i = 0u; i < markers->count; ++i)
    markers->lanes[i].level_q16 = 0;
}

// One bounded deliberative episode. With a stamped plan the walk consumes the
// somatic-frozen biased order (grounded reasoning); with a null plan it is
// refused -- the uncoupled control severs the marker input and re-plans
// instead, so both arms travel the identical stamped machinery. Selection
// follows the lawful deliberation rule over the visited window: strict
// argmax arbitration score, exact tie to the earliest visit. Regulation
// shapes only which candidates fit inside the visit budget.
__device__ inline bool damasio_run_episode(
    const DirectSomaticCandidate* candidates, std::uint32_t count,
    const DirectSomaticSearchPlan* plan, const DirectAffectBodyState* affect,
    std::uint32_t visit_budget, DirectDamasioEpisodeReceipt* receipt) {
  if (candidates == nullptr || receipt == nullptr || count == 0u ||
      count > kDamasioEpisodeCapacity || visit_budget == 0u)
    return false;
  for (std::uint32_t i = 0u; i < count; ++i)
    if (candidates[i].action_ticket_id == 0u) return false;
  if (plan == nullptr || plan->bias_sequence == 0u ||
      plan->requested_count != count || plan->admissible_count != count)
    return false;
  bool seen[kDamasioEpisodeCapacity]{};
  for (std::uint32_t k = 0u; k < count; ++k) {
    const std::uint32_t slot = plan->candidate_order[k];
    if (slot >= count || seen[slot]) return false;
    seen[slot] = true;
  }

  DirectDamasioEpisodeReceipt out{};
  out.bias_sequence = plan->bias_sequence;
  out.grounded_order = 1u;
  out.budget = visit_budget;
  out.admissible_count = count;
  out.regulation_q16 = damasio_regulation_q16(affect);
  out.visits_used = visit_budget < count ? visit_budget : count;

  bool have_best = false;
  bool nominated = false;
  bool found_positive = false;
  std::int32_t best_score = 0;
  std::uint32_t solution_rank = 0u;
  for (std::uint32_t k = 0u; k < out.visits_used; ++k) {
    const std::uint32_t slot = plan->candidate_order[k];
    const std::int32_t valence =
        damasio_candidate_valence_q16(affect, candidates[slot]);
    const std::int32_t score = candidates[slot].arbitration_score_q16;
    out.visit_order[k] = slot;
    out.visited_valence_q16[k] = valence;
    out.visited_scores_q16[k] = score;
    if (!have_best || valence > out.best_visited_valence_q16) {
      have_best = true;
      out.best_visited_valence_q16 = valence;
      out.best_depth = k + 1u;
      out.best_slot = slot;
    }
    if (valence > 0 && !found_positive) {
      found_positive = true;
      out.wasted_visits_before_positive = k;
    }
    if (!nominated || score > best_score) {
      nominated = true;
      best_score = score;
      solution_rank = k;
    }
  }
  if (!found_positive)
    out.wasted_visits_before_positive = out.visits_used;
  out.found_valuable = found_positive ? 1u : 0u;
  out.solution_slot = out.visit_order[solution_rank];
  out.solution_valence_q16 = out.visited_valence_q16[solution_rank];
  *receipt = out;
  return true;
}

// One round of selection pressure: both episodes solved the same fixed
// evaluation budget on identical bodies; the tally accumulates their
// settled solution values and the win/loss record.
__device__ inline void damasio_tally_round(
    DirectDamasioSelectionTally* tally,
    const DirectDamasioEpisodeReceipt& grounded,
    const DirectDamasioEpisodeReceipt& ablated) {
  if (tally == nullptr) return;
  ++tally->rounds;
  tally->grounded_cumulative_valence_q16 += grounded.solution_valence_q16;
  tally->ablated_cumulative_valence_q16 += ablated.solution_valence_q16;
  if (grounded.solution_valence_q16 > ablated.solution_valence_q16)
    ++tally->grounded_rounds_won;
  else if (ablated.solution_valence_q16 > grounded.solution_valence_q16)
    ++tally->ablated_rounds_won;
  else
    ++tally->ties;
}

}  // namespace substrate::direct_network

#endif  // HARDWARE_NATIVE_DIRECT_ADULT_DAMASIO_GROUNDING_CUH
