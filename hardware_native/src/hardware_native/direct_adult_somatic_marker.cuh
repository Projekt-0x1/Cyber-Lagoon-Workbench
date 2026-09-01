#ifndef HARDWARE_NATIVE_DIRECT_ADULT_SOMATIC_MARKER_CUH
#define HARDWARE_NATIVE_DIRECT_ADULT_SOMATIC_MARKER_CUH

// f.somatic_marker_biasing (#1527): visceral homeostatic bodily feedback
// biasing prospective search spaces prior to explicit deliberative solving.
//
// Law anchors (Revision 12):
//   * anticipatory priming without speculative authority: the somatic marker
//     reshapes the EXPLORATION ORDER of a prospective search space over
//     already-authorized candidates before any deliberative solving runs. It
//     cannot mint participation, settle eligibility, pre-write Delta-rho, or
//     touch the arbitration score that selects the outcome;
//   * self-fulfilling credit firewall: biased recruitment stays internally
//     caused ordering. Deliberation attributes selection to each candidate's
//     own settled consequence evidence alone; exact evidence ties remain
//     unresolved -- bias may order search but cannot become action authority;
//   * homeostasis: per-lane marker levels integrate the visceral differential
//     carried by the #1517 affect masses and relax toward neutral whenever a
//     lane receives no fresh visceral evidence, so the search space recovers
//     its prediction-only order as the bodily signal normalizes;
//   * source coupling: levels are read exclusively from the device-derived
//     DirectAffectBodyState over settled consequence ledgers. Host labels and
//     host composition contribute nothing;
//   * fail closed: a search slot exists only for a candidate bound to a
//     settled action ticket in device exact state. Unbound, unsettled or
//     overflowing candidates are refused and counted, never silently
//     admitted, and deliberation structurally cannot run without a prior
//     sequence-stamped biasing pass.

#include <cstdint>
#include <type_traits>

#include "hardware_native/direct_adult_affect_body.cuh"
#include "hardware_native/direct_adult_q16.cuh"

namespace substrate::direct_network {

inline constexpr std::uint32_t kSomaticMarkerLaneCapacity = 16u;
inline constexpr std::uint32_t kSomaticSearchCapacity = 16u;
inline constexpr std::int32_t kSomaticDamageGainQ16 =
    direct_adult_core::kQ16One / 2;
inline constexpr std::int32_t kSomaticVitalityGainQ16 =
    direct_adult_core::kQ16One / 3;
inline constexpr std::int32_t kSomaticRelaxGainQ16 =
    direct_adult_core::kQ16One / 4;
// Declared neutrality tolerance: a level this close to the floor reads as
// homeostatically silent and contributes nothing to the search order.
inline constexpr std::int32_t kSomaticNeutralDeadbandQ16 =
    direct_adult_core::kQ16One / 64;

struct alignas(8) DirectSomaticMarkerLane {
  std::uint32_t channel;
  std::int32_t level_q16;
  std::uint32_t damage_samples_seen;
  std::uint32_t vitality_samples_seen;
};
static_assert(std::is_trivial_v<DirectSomaticMarkerLane> &&
              std::is_standard_layout_v<DirectSomaticMarkerLane> &&
              std::has_unique_object_representations_v<DirectSomaticMarkerLane>);

struct alignas(8) DirectSomaticMarkerState {
  DirectSomaticMarkerLane lanes[kSomaticMarkerLaneCapacity];
  std::uint64_t bias_sequence;
  std::uint32_t count;
  std::uint32_t integrations;
  std::uint32_t relaxations;
  std::uint32_t refusals;
};
static_assert(std::is_trivial_v<DirectSomaticMarkerState> &&
              std::is_standard_layout_v<DirectSomaticMarkerState> &&
              std::has_unique_object_representations_v<
                  DirectSomaticMarkerState>);

// One prospective search candidate bound to its exact device identities. The
// arbitration score travels through every biasing path unmodified; it is the
// deliberative solver's input, never the marker's output.
struct DirectSomaticCandidate {
  std::uint64_t action_ticket_id;
  std::uint32_t participation_identity;
  std::uint32_t root_channel;
  std::int32_t base_priority_q16;
  std::int32_t arbitration_score_q16;
};
static_assert(std::is_trivial_v<DirectSomaticCandidate> &&
              std::is_standard_layout_v<DirectSomaticCandidate> &&
              std::has_unique_object_representations_v<
                  DirectSomaticCandidate>);

struct alignas(8) DirectSomaticSearchPlan {
  std::uint32_t candidate_order[kSomaticSearchCapacity];
  std::int32_t applied_level_q16[kSomaticSearchCapacity];
  std::uint64_t bias_sequence;
  std::uint32_t requested_count;
  std::uint32_t admissible_count;
  std::uint32_t refused_count;
  std::uint32_t top_candidate_index;
};
static_assert(std::is_trivial_v<DirectSomaticSearchPlan> &&
              std::is_standard_layout_v<DirectSomaticSearchPlan> &&
              std::has_unique_object_representations_v<
                  DirectSomaticSearchPlan>);

struct alignas(8) DirectSomaticDeliberationReceipt {
  std::uint64_t bias_sequence;
  std::uint64_t selected_ticket_id;
  std::uint32_t selected_participation_identity;
  std::uint32_t selected_slot;
  std::uint32_t visit_order[kSomaticSearchCapacity];
  std::int32_t visit_scores_q16[kSomaticSearchCapacity];
  std::uint32_t visit_count;
  std::uint32_t tie_decided_by_nomination;
};
static_assert(std::is_trivial_v<DirectSomaticDeliberationReceipt> &&
              std::is_standard_layout_v<DirectSomaticDeliberationReceipt> &&
              std::has_unique_object_representations_v<
                  DirectSomaticDeliberationReceipt>);

// A level inside the declared deadband is homeostatic silence: the lane
// neither promotes nor demotes anyone.
DIRECT_ADULT_HD inline std::int32_t somatic_effective_level_q16(
    std::int32_t level_q16) {
  const std::int32_t magnitude = level_q16 < 0 ? -level_q16 : level_q16;
  return magnitude <= kSomaticNeutralDeadbandQ16 ? 0 : level_q16;
}

// A search slot exists only behind a settled ticket. An open ticket names a
// real emission but no settled consequence yet, so it carries no marker
// authority either.
__device__ inline const direct_adult_core::AsynchronousTicket*
somatic_find_settled_ticket(
    const direct_adult_core::AsynchronousTicket* tickets,
    std::uint32_t ticket_count, std::uint64_t action_ticket_id) {
  if (action_ticket_id == 0u) return nullptr;
  for (std::uint32_t i = 0u; i < ticket_count; ++i) {
    if (tickets[i].ticket_id != action_ticket_id) continue;
    return tickets[i].settled != 0u ? &tickets[i] : nullptr;
  }
  return nullptr;
}

__device__ inline std::int32_t somatic_lane_level_q16(
    const DirectSomaticMarkerState* state, std::uint32_t channel) {
  for (std::uint32_t i = 0u; i < state->count; ++i)
    if (state->lanes[i].channel == channel)
      return somatic_effective_level_q16(state->lanes[i].level_q16);
  return 0;  // visceral silence: no lane, no invented marker
}

__device__ inline DirectSomaticMarkerLane* somatic_lane_slot(
    DirectSomaticMarkerState* state, std::uint32_t channel) {
  for (std::uint32_t i = 0u; i < state->count; ++i)
    if (state->lanes[i].channel == channel) return &state->lanes[i];
  if (state->count >= kSomaticMarkerLaneCapacity) {
    ++state->refusals;
    return nullptr;
  }
  DirectSomaticMarkerLane fresh{};
  fresh.channel = channel;
  state->lanes[state->count] = fresh;
  return &state->lanes[state->count++];
}

// One homeostatic integration epoch over the device-derived affect table.
// Fresh damage deposits drive a lane's level toward the demotion floor, fresh
// vitality deposits toward the promotion ceiling, and a lane with no fresh
// visceral evidence relaxes toward neutral -- recovery is the null signal,
// never an authored repair. Lanes walk in affect-table order under one thread
// so accumulation is deterministic regardless of scheduling.
__device__ inline void somatic_integrate_markers(
    DirectSomaticMarkerState* state, const DirectAffectBodyState* affect) {
  if (state == nullptr || affect == nullptr) return;
  for (std::uint32_t i = 0u; i < affect->count; ++i) {
    const DirectAffectBodyEntry& entry = affect->entries[i];
    DirectSomaticMarkerLane* lane = somatic_lane_slot(state, entry.channel);
    if (lane == nullptr) continue;
    const std::uint32_t fresh_damage =
        entry.damage_samples - lane->damage_samples_seen;
    const std::uint32_t fresh_vitality =
        entry.vitality_samples - lane->vitality_samples_seen;
    lane->damage_samples_seen += fresh_damage;
    lane->vitality_samples_seen += fresh_vitality;
    for (std::uint32_t d = 0u; d < fresh_damage; ++d) {
      lane->level_q16 -= direct_adult_core::mul_q16(
          lane->level_q16 + direct_adult_core::kQ16One,
          kSomaticDamageGainQ16);
    }
    for (std::uint32_t v = 0u; v < fresh_vitality; ++v) {
      lane->level_q16 += direct_adult_core::mul_q16(
          direct_adult_core::kQ16One - lane->level_q16,
          kSomaticVitalityGainQ16);
    }
    if (fresh_damage == 0u && fresh_vitality == 0u) {
      const std::int32_t relaxed =
          lane->level_q16 -
          direct_adult_core::mul_q16(lane->level_q16, kSomaticRelaxGainQ16);
      if (relaxed != lane->level_q16) ++state->relaxations;
      lane->level_q16 = relaxed;
    }
    lane->level_q16 = direct_adult_core::clamp_q16(
        lane->level_q16, -direct_adult_core::kQ16One,
        direct_adult_core::kQ16One);
  }
  ++state->integrations;
}

// The biasing pass: sequence-stamp and freeze one exploration order over the
// authorized candidates, BEFORE any deliberative solving runs. Every
// candidate must resolve to a settled ticket or it is refused outright; a
// candidate whose visceral lane is silent keeps its prediction priority
// untouched. Biased priority is the exact integer sum of prediction priority
// and effective marker level, sorted stably descending with ties kept in
// declaration order, so the plan is a pure deterministic function of device
// state.
__device__ inline bool somatic_plan_search(
    DirectSomaticMarkerState* state,
    const direct_adult_core::AsynchronousTicket* tickets,
    std::uint32_t ticket_count, const DirectSomaticCandidate* candidates,
    std::uint32_t count, DirectSomaticSearchPlan* plan) {
  if (state == nullptr || tickets == nullptr || candidates == nullptr ||
      plan == nullptr || count == 0u || count > kSomaticSearchCapacity) {
    if (state != nullptr) ++state->refusals;
    return false;
  }
  DirectSomaticSearchPlan out{};
  out.requested_count = count;
  struct Ranked {
    std::int32_t priority_q16;
    std::uint32_t index;
  };
  Ranked ranked[kSomaticSearchCapacity];
  for (std::uint32_t i = 0u; i < count; ++i) {
    out.applied_level_q16[i] =
        somatic_lane_level_q16(state, candidates[i].root_channel);
    if (somatic_find_settled_ticket(tickets, ticket_count,
                                    candidates[i].action_ticket_id) ==
        nullptr) {
      ++state->refusals;
      ++out.refused_count;
      continue;
    }
    ranked[out.admissible_count].priority_q16 =
        candidates[i].base_priority_q16 + out.applied_level_q16[i];
    ranked[out.admissible_count].index = i;
    ++out.admissible_count;
  }
  if (out.admissible_count == 0u) {
    ++state->refusals;
    *plan = out;
    return false;
  }
  for (std::uint32_t i = 1u; i < out.admissible_count; ++i) {
    const Ranked key = ranked[i];
    std::uint32_t j = i;
    while (j > 0u && (ranked[j - 1u].priority_q16 < key.priority_q16 ||
                      (ranked[j - 1u].priority_q16 == key.priority_q16 &&
                       ranked[j - 1u].index > key.index))) {
      ranked[j] = ranked[j - 1u];
      --j;
    }
    ranked[j] = key;
  }
  for (std::uint32_t k = 0u; k < out.admissible_count; ++k)
    out.candidate_order[k] = ranked[k].index;
  out.top_candidate_index = out.candidate_order[0];
  out.bias_sequence = ++state->bias_sequence;
  *plan = out;
  return true;
}

// Explicit deliberative solving AFTER the biasing pass: candidates are
// visited strictly in the plan's frozen order, each contributing only its own
// arbitration score. Only a unique strict maximum selects an action. Exact
// best-score ties remain unresolved: nomination order may shape bounded search
// order but cannot become action authority. Deliberation refuses to run without
// a prior sequence-stamped biasing pass and leaves no partial receipt behind.
__device__ inline bool somatic_deliberate(
    const DirectSomaticCandidate* candidates, std::uint32_t count,
    const DirectSomaticSearchPlan* plan,
    DirectSomaticDeliberationReceipt* receipt) {
  if (candidates == nullptr || plan == nullptr || receipt == nullptr ||
      plan->bias_sequence == 0u || plan->admissible_count == 0u ||
      plan->requested_count != count || count > kSomaticSearchCapacity)
    return false;
  DirectSomaticDeliberationReceipt out{};
  out.bias_sequence = plan->bias_sequence;
  std::uint32_t best_rank = 0u;
  std::int32_t best_score = 0;
  bool nominated = false;
  bool tied = false;
  for (std::uint32_t k = 0u; k < plan->admissible_count; ++k) {
    const std::uint32_t slot = plan->candidate_order[k];
    const std::int32_t score = candidates[slot].arbitration_score_q16;
    out.visit_order[k] = slot;
    out.visit_scores_q16[k] = score;
    if (!nominated || score > best_score) {
      nominated = true;
      best_rank = k;
      best_score = score;
      tied = false;
    } else if (score == best_score && slot != out.visit_order[best_rank]) {
      tied = true;
    }
  }
  out.visit_count = plan->admissible_count;
  if (tied) {
    // ABI-stable legacy field now means "tie observed". A tied receipt names
    // no selected action; nomination did not decide the outcome.
    out.tie_decided_by_nomination = 1u;
    out.selected_slot = 0u;
    out.selected_ticket_id = 0u;
    out.selected_participation_identity = 0u;
  } else {
    const std::uint32_t selected_slot = out.visit_order[best_rank];
    out.selected_slot = selected_slot;
    out.selected_ticket_id = candidates[selected_slot].action_ticket_id;
    out.selected_participation_identity = candidates[selected_slot].participation_identity;
  }
  *receipt = out;
  return true;
}

}  // namespace substrate::direct_network

#endif  // HARDWARE_NATIVE_DIRECT_ADULT_SOMATIC_MARKER_CUH
