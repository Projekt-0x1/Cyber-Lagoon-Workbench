#ifndef HARDWARE_NATIVE_DIRECT_ADULT_ALTERNATIVE_FUTURES_CUH
#define HARDWARE_NATIVE_DIRECT_ADULT_ALTERNATIVE_FUTURES_CUH

#include "hardware_native/direct_adult_core.cuh"
#include "hardware_native/direct_adult_runtime_frontiers.cuh"

namespace substrate::direct_adult_core {

inline constexpr std::uint32_t kResidentAlternativeBranchCount = 2u;
inline constexpr std::uint32_t kResidentAlternativeBankCapacity = 8u;

enum class ResidentAlternativeState : std::uint32_t {
  free = 0u,
  live = 1u,
  expired = 2u,
};

// One prospective future: an endogenous rollout of the shared actual parent
// through ONE executable resolution of the resident revision. Branches from
// one derivation are mutually exclusive by construction -- each names a
// distinct executable work identity -- and none of them is more true than
// another; the ordered envelopes are evidence for later volitional gates.
struct alignas(8) ResidentAlternativeFuture {
  ResidentSuccessorShadowEntry shadow;
  ResidentPredictedResidualEnvelope predicted_residual;
  std::uint64_t branch_identity;
  std::uint32_t branch_ordinal;
  std::uint32_t work_class;
  std::int32_t projected_state_q16;
  std::uint32_t cumulative_work_units;
  ResidentAlternativeState state;
};
static_assert(std::is_standard_layout_v<ResidentAlternativeFuture> &&
              std::is_trivial_v<ResidentAlternativeFuture>);

struct alignas(8) ResidentAlternativeFuturesFrontier {
  ResidentAlternativeFuture branches[kResidentAlternativeBankCapacity];
  std::uint64_t parent_occurrence_identity;
  std::uint32_t live_count;
  std::uint32_t generations;
  std::uint32_t refusals;
};
static_assert(std::is_standard_layout_v<ResidentAlternativeFuturesFrontier> &&
              std::is_trivial_v<ResidentAlternativeFuturesFrontier>);

DIRECT_ADULT_HD inline std::uint64_t resident_alternative_branch_identity(
    const ResidentRecipeOccurrence& parent, std::uint32_t ordinal,
    std::uint32_t work_class, std::uint32_t generation_tick) {
  using direct_network::exact_history_fold_word;
  std::uint64_t identity = exact_history_fold_word(
      0x616c7466757473ull, parent.occurrence_identity);
  identity = exact_history_fold_word(identity, ordinal);
  identity = exact_history_fold_word(identity, work_class);
  identity = exact_history_fold_word(identity, generation_tick);
  return identity == 0u ? 1u : identity;
}

DIRECT_ADULT_HD inline std::uint32_t expire_resident_alternative_futures(
    ResidentAlternativeFuturesFrontier* frontier, std::uint32_t current_tick) {
  if (frontier == nullptr) return 0u;
  std::uint32_t live = 0u;
  for (std::uint32_t i = 0u; i < kResidentAlternativeBankCapacity; ++i) {
    auto& branch = frontier->branches[i];
    if (branch.state == ResidentAlternativeState::live &&
        branch.shadow.horizon_tick < current_tick) {
      branch.state = ResidentAlternativeState::expired;
      branch.shadow.state = ResidentSuccessorShadowState::expired;
      branch.shadow.occurrence.state = kResidentRecipeOccurrenceSettled;
    }
    live += branch.state == ResidentAlternativeState::live ? 1u : 0u;
  }
  frontier->live_count = live;
  return live;
}

// Derives the bounded alternative-branch set from one live actual parent in
// one fail-closed transaction. The two branches resolve the SAME revision
// through DIFFERENT executable morphologies (the compiled macro versus its
// generic route micrograph), so their work identities are disjoint by
// construction; equal projections with equal identities refuse as duplicates.
// Nothing outside the endogenous bank moves.
DIRECT_ADULT_HD inline bool generate_resident_alternative_futures(
    const DirectBrain& brain, const ResidentActualFrontier& actual_frontier,
    std::uint32_t parent_frontier_slot, std::uint32_t current_tick,
    ResidentAlternativeFuturesFrontier* frontier) {
  using namespace direct_network;
  if (frontier == nullptr ||
      parent_frontier_slot >= kResidentActualFrontierCapacity) {
    if (frontier != nullptr) ++frontier->refusals;
    return false;
  }
  const ResidentActualFrontierEntry& parent =
      actual_frontier.entries[parent_frontier_slot];
  if (parent.state != ResidentActualFrontierState::live ||
      parent.derivation_index >= kResidentPostbirthRecipeReserve ||
      brain.postbirth_constructor == nullptr ||
      brain.postbirth_derivations == nullptr ||
      parent.derivation_index >=
          brain.postbirth_constructor->derivation_count ||
      parent.work.identity != resident_executable_work_identity(parent.work)) {
    ++frontier->refusals;
    return false;
  }
  const ResidentRecipeDerivation& derivation =
      brain.postbirth_derivations[parent.derivation_index];

  expire_resident_alternative_futures(frontier, current_tick);
  for (std::uint32_t i = 0u; i < kResidentAlternativeBankCapacity; ++i)
    if (frontier->branches[i].state == ResidentAlternativeState::live &&
        frontier->branches[i].shadow.parent_occurrence_identity ==
            parent.occurrence.occurrence_identity) {
      ++frontier->refusals;
      return false;
    }

  // A revision with exactly one executable resolution offers no
  // alternatives; storing one branch twice would be a duplicate. Each
  // branch's resolutions are lowered against ITS OWN bound occurrence so
  // chained execution validates against the same work identity.
  ResidentExecutableMorphologyWork probe_resolutions[2];
  if (!lower_resident_executable_morphology(
          brain, derivation, parent.occurrence, false,
          &probe_resolutions[0]) ||
      !lower_resident_executable_morphology(
          brain, derivation, parent.occurrence, true,
          &probe_resolutions[1])) {
    ++frontier->refusals;
    return false;
  }
  const bool single_resolution =
      probe_resolutions[0].work_class == probe_resolutions[1].work_class &&
      probe_resolutions[0].route_count == probe_resolutions[1].route_count;
  if (single_resolution) {
    ++frontier->refusals;
    return false;
  }

  std::uint32_t slots[kResidentAlternativeBranchCount]{};
  std::uint32_t free_count = 0u;
  for (std::uint32_t i = 0u;
       i < kResidentAlternativeBankCapacity &&
       free_count < kResidentAlternativeBranchCount;
       ++i)
    if (frontier->branches[i].state != ResidentAlternativeState::live)
      slots[free_count++] = i;
  if (free_count != kResidentAlternativeBranchCount) {
    ++frontier->refusals;
    return false;
  }

  ResidentAlternativeFuturesFrontier candidate = *frontier;
  for (std::uint32_t b = 0u; b < kResidentAlternativeBranchCount; ++b) {
    const std::uint32_t depth =
        probe_resolutions[b].route_count == 0u
            ? 1u
            : probe_resolutions[b].route_count;
    ResidentSuccessorShadowFrontier scratch{};
    if (!generate_resident_successor_shadow(
            brain, actual_frontier, parent_frontier_slot, current_tick, depth,
            &scratch) || scratch.live_count != 1u) {
      ++frontier->refusals;
      return false;
    }
    ResidentSuccessorShadowEntry shadow{};
    for (std::uint32_t i = 0u; i < kResidentSuccessorShadowCapacity; ++i)
      if (scratch.entries[i].state == ResidentSuccessorShadowState::live)
        shadow = scratch.entries[i];

    ResidentExecutableMorphologyWork branch_resolutions[2];
    if (!lower_resident_executable_morphology(
            brain, derivation, shadow.occurrence, false,
            &branch_resolutions[0]) ||
        !lower_resident_executable_morphology(
            brain, derivation, shadow.occurrence, true,
            &branch_resolutions[1])) {
      ++frontier->refusals;
      return false;
    }
    const ResidentExecutableMorphologyWork& resolution = branch_resolutions[b];

    // Roll the branch to its own depth through ITS OWN resolution.
    std::int32_t projection = shadow.projected_state_q16;
    std::int32_t penultimate = parent.output_q16;
    std::uint32_t cumulative = shadow.work_units;
    for (std::uint32_t step = 1u; step < depth; ++step) {
      penultimate = projection;
      std::uint32_t step_work = 0u;
      if (!execute_resident_executable_morphology(
              brain, derivation, shadow.occurrence, resolution, projection,
              &projection, &step_work))
        break;
      cumulative += step_work;
    }

    ResidentAlternativeFuture branch{};
    branch.shadow = shadow;
    branch.shadow.projected_state_q16 = projection;
    branch.shadow.work_units = cumulative;
    branch.predicted_residual.recurrence_residual_q16 =
        static_cast<std::int64_t>(projection) - penultimate;
    branch.predicted_residual.adjacent_horizon_residual_q16 =
        static_cast<std::int64_t>(projection) - parent.output_q16;
    const std::uint64_t recurrence_magnitude = resident_prediction_magnitude(
        branch.predicted_residual.recurrence_residual_q16);
    const std::uint64_t adjacent_magnitude = resident_prediction_magnitude(
        branch.predicted_residual.adjacent_horizon_residual_q16);
    branch.predicted_residual.envelope_q16 =
        recurrence_magnitude > adjacent_magnitude ? recurrence_magnitude
                                                  : adjacent_magnitude;
    branch.branch_ordinal = b;
    branch.work_class =
        static_cast<std::uint32_t>(resolution.work_class);
    branch.projected_state_q16 = projection;
    branch.cumulative_work_units = cumulative;
    branch.branch_identity = resident_alternative_branch_identity(
        parent.occurrence, b, branch.work_class, current_tick);

    // MUTUAL EXCLUSIVITY against every live branch in the bank.
    bool exclusive = true;
    for (std::uint32_t i = 0u; i < kResidentAlternativeBankCapacity; ++i) {
      const auto& other = candidate.branches[i];
      if (other.state != ResidentAlternativeState::live) continue;
      exclusive = exclusive &&
          other.branch_identity != branch.branch_identity &&
          !(other.projected_state_q16 == branch.projected_state_q16 &&
            other.work_class == branch.work_class);
    }
    if (!exclusive) {
      ++frontier->refusals;
      return false;
    }

    branch.state = ResidentAlternativeState::live;
    candidate.branches[slots[b]] = branch;
  }

  candidate.parent_occurrence_identity =
      parent.occurrence.occurrence_identity;
  candidate.live_count += kResidentAlternativeBranchCount;
  ++candidate.generations;
  *frontier = candidate;
  return true;
}

}  // namespace substrate::direct_adult_core

#endif  // HARDWARE_NATIVE_DIRECT_ADULT_ALTERNATIVE_FUTURES_CUH
