#ifndef HARDWARE_NATIVE_DIRECT_ADULT_CONSTRUCTOR_CREDIT_CUH
#define HARDWARE_NATIVE_DIRECT_ADULT_CONSTRUCTOR_CREDIT_CUH

#include "direct_adult_q16.cuh"
#include "direct_network_brain.cuh"
#include "direct_exact_history.cuh"

namespace substrate::direct_adult_core {
using direct_network::DirectBrain;

// Kind-4 update authority: a construction choice stays causally load-bearing
// only while the child route it built keeps settling downstream consequences
// under its still-current physical incarnation. Ancestor responsibility decays
// per derivation hop and is bounded in depth, so ancient builders never
// receive every descendant consequence forever (Revision 12 §7/§8).
inline constexpr std::uint32_t kConstructorCreditMaxAncestryDepth = 2u;
inline constexpr std::uint64_t kConstructorCreditWitnessSeed = 0x636f6e7374727631ull;

struct ConstructorViabilityCensus {
  std::uint32_t settled_children;
  std::uint32_t constructed_children;
  std::uint32_t ancestor_records;
  std::uint32_t stale_incarnation_refusals;
  std::uint32_t duplicate_refusals;
};

__device__ inline std::uint64_t constructor_viability_witness(
    std::uint64_t child_identity, std::uint32_t child_cell,
    std::uint32_t ancestor_cell, std::uint32_t depth) {
  std::uint64_t identity =
      direct_network::exact_history_fold_word(kConstructorCreditWitnessSeed, child_identity);
  identity = direct_network::exact_history_fold_word(identity, child_cell);
  identity = direct_network::exact_history_fold_word(identity, ancestor_cell);
  identity = direct_network::exact_history_fold_word(identity, depth);
  return identity == 0u ? 1u : identity;
}

namespace constructor_credit_detail {

inline constexpr std::uint32_t kPlanCapacity = 4u * kMaxProvenanceSlotsPerNode;

struct Plan {
  direct_network::DirectExactHistoryRecord records[2u * kPlanCapacity];
  std::uint32_t commit_count;
};

}  // namespace constructor_credit_detail

// Plans decaying, depth-bounded ancestor credit for construction ancestry.
// Only experience credits prove downstream child viability: a recipe_commit
// whose parent identity resolves to a topology event is a construction commit
// and attributes nothing. The child's route incarnation recorded at settlement
// must still be the current physical incarnation, or the construction choice
// is no longer load-bearing and the chain is cut.
__device__ inline ConstructorViabilityCensus plan_constructor_viability_credit(
    const DirectBrain& brain, std::uint32_t scan_begin,
    std::uint32_t commit_tick, constructor_credit_detail::Plan* plan) {
  ConstructorViabilityCensus census{};
  const direct_network::ResidentDevelopmentState* development = brain.development;
  const direct_network::ResidentPostbirthConstructorState* state =
      brain.postbirth_constructor;
  if (development == nullptr || state == nullptr || plan == nullptr ||
      brain.recipe_cells == nullptr || brain.postbirth_derivations == nullptr ||
      brain.routes == nullptr || brain.route_incarnations == nullptr)
    return census;
  const auto& history = development->exact_history;
  if (history.phase_kind != direct_network::DirectExactHistoryKind::empty)
    return census;
  const std::uint32_t cell_count = development->recipe_cell_count;
  const std::uint32_t derivation_count = state->derivation_count;
  const std::uint32_t end = history.committed_slots;
  if (scan_begin > end) scan_begin = end;

  for (std::uint32_t cursor = scan_begin; cursor < end; ++cursor) {
    const direct_network::DirectExactHistoryRecord witness =
        history.records[cursor];
    if (witness.kind != direct_network::DirectExactHistoryKind::recipe_commit ||
        witness.resource_delta == 0 || witness.source >= cell_count)
      continue;
    // Direct settlement credits carry an unset value slot; constructor
    // attribution records stamp their ancestry depth there and are never
    // themselves downstream viability evidence.
    if (witness.value != kInvalidIndex) continue;
    bool constructed_elsewhere = false;
    for (std::uint32_t p = 0u; p < end && !constructed_elsewhere; ++p) {
      const auto& candidate = history.records[p];
      constructed_elsewhere =
          (candidate.kind == direct_network::DirectExactHistoryKind::topology_growth ||
           candidate.kind ==
               direct_network::DirectExactHistoryKind::topology_retraction) &&
          candidate.identity == witness.parent_identity;
    }
    if (constructed_elsewhere) continue;
    ++census.settled_children;
    const std::uint32_t child_cell = witness.source;
    const std::uint32_t route_index = witness.subject;
    const std::uint64_t recorded_incarnation =
        (static_cast<std::uint64_t>(witness.flags) << 32u) | witness.context;
    if (route_index >= brain.route_capacity ||
        !direct_network::route_is_active(brain.routes[route_index]) ||
        brain.route_incarnations[route_index] != recorded_incarnation) {
      ++census.stale_incarnation_refusals;
      continue;
    }
    std::uint32_t derivation = kInvalidIndex;
    for (std::uint32_t i = 0u; i < derivation_count && derivation == kInvalidIndex; ++i)
      if (brain.postbirth_derivations[i].recipe_cell == child_cell) derivation = i;
    if (derivation == kInvalidIndex) continue;
    ++census.constructed_children;
    for (std::uint32_t depth = 1u;
         depth <= kConstructorCreditMaxAncestryDepth; ++depth) {
      if (derivation == kInvalidIndex) break;
      const direct_network::ResidentRecipeDerivation& link =
          brain.postbirth_derivations[derivation];
      const std::uint32_t parent_cell = link.parent_recipe_cell;
      if (parent_cell >= cell_count ||
          brain.recipe_cells[parent_cell].logical_recipe_id !=
              link.parent_logical_recipe_id)
        break;
      const std::int64_t share = witness.resource_delta >> depth;
      if (share == 0) break;
      const std::uint64_t identity = constructor_viability_witness(
          witness.identity, child_cell, parent_cell, depth);
      bool already_attributed = false;
      for (std::uint32_t p = 0u; p < end && !already_attributed; ++p)
        already_attributed =
            history.records[p].kind ==
                direct_network::DirectExactHistoryKind::recipe_commit &&
            history.records[p].identity == identity;
      if (already_attributed) {
        ++census.duplicate_refusals;
        break;
      }
      if (plan->commit_count >= constructor_credit_detail::kPlanCapacity) break;
      derivation = kInvalidIndex;
      for (std::uint32_t i = 0u; i < derivation_count && derivation == kInvalidIndex;
           ++i)
        if (brain.postbirth_derivations[i].recipe_cell == parent_cell) derivation = i;
      std::int64_t prior =
          static_cast<std::int64_t>(brain.recipe_cells[parent_cell].credit_q16);
      for (std::uint32_t j = 0u; j < plan->commit_count; ++j)
        if (plan->records[j].source == parent_cell)
          prior = static_cast<std::int64_t>(plan->records[j].incarnation_after);
      direct_network::DirectExactHistoryRecord record{};
      record.identity = identity;
      record.parent_identity = witness.identity;
      record.resident_tick = commit_tick;
      record.event_tick = witness.event_tick;
      record.kind = direct_network::DirectExactHistoryKind::recipe_commit;
      record.source = parent_cell;
      record.subject = child_cell;
      record.value = depth;
      record.context = witness.context;
      record.flags = witness.flags;
      record.incarnation_before = static_cast<std::uint64_t>(prior);
      record.incarnation_after = static_cast<std::uint64_t>(prior + share);
      record.resource_delta = share;
      plan->records[plan->commit_count++] = record;
    }
  }
  return census;
}

// Commits the planned ancestor credit inside one atomic exact-history phase:
// paired immutable RecipeRevision events move the resident bodies only after
// the whole transaction has been accepted into the journal.
__device__ inline std::uint32_t attribute_constructor_viability_credit(
    const DirectBrain& brain, std::uint32_t scan_begin,
    std::uint32_t commit_tick, ConstructorViabilityCensus* out_census) {
  using constructor_credit_detail::kPlanCapacity;
  using constructor_credit_detail::Plan;
  Plan plan{};
  ConstructorViabilityCensus census =
      plan_constructor_viability_credit(brain, scan_begin, commit_tick, &plan);
  if (out_census != nullptr) *out_census = census;
  const std::uint32_t commit_count = plan.commit_count;
  if (commit_count == 0u) return 0u;
  direct_network::ResidentDevelopmentState* development = brain.development;
  if (development == nullptr || brain.recipe_cells == nullptr) return 0u;
  direct_network::DirectExactHistoryHotPage& history =
      development->exact_history;
  if (history.sealed != 0u ||
      history.phase_kind != direct_network::DirectExactHistoryKind::empty ||
      2u * commit_count >
          direct_network::kDirectExactHistoryHotPageCapacity - history.committed_slots)
    return 0u;
  for (std::uint32_t i = 0u; i < commit_count; ++i) {
    const direct_network::DirectExactHistoryRecord& credit = plan.records[i];
    direct_network::ResidentRecipeCell predicted =
        brain.recipe_cells[credit.source];
    for (std::uint32_t j = 0u; j < i; ++j)
      if (plan.records[kPlanCapacity + j].source == credit.source)
        direct_network::apply_resident_recipe_revision_event(
            &predicted, plan.records[kPlanCapacity + j], credit.source);
    if (!direct_network::stage_resident_recipe_revision_event(
            &plan.records[kPlanCapacity + i], predicted, credit.source,
            direct_network::ResidentRecipeRevisionAuthority::experience,
            credit.identity, credit.parent_identity, commit_tick,
            credit.event_tick, credit.subject, i, credit.value,
            credit.resource_delta))
      return 0u;
  }
  if (!direct_network::begin_exact_history_phase(
          &history, direct_network::DirectExactHistoryKind::recipe_commit,
          2u * commit_count, commit_tick))
    return 0u;
  const std::uint32_t base = history.phase_base;
  for (std::uint32_t i = 0u; i < commit_count; ++i)
    history.records[base + i] = plan.records[i];
  for (std::uint32_t i = 0u; i < commit_count; ++i)
    history.records[base + commit_count + i] = plan.records[kPlanCapacity + i];
  if (direct_network::finish_exact_history_phase(&history) != 2u * commit_count)
    return 0u;
  for (std::uint32_t i = 0u; i < 2u * commit_count; ++i) {
    const direct_network::DirectExactHistoryRecord& event =
        history.records[base + i];
    if (event.kind != direct_network::DirectExactHistoryKind::recipe_revision)
      continue;
    direct_network::apply_resident_recipe_revision_event(
        &brain.recipe_cells[event.source], event, event.source);
  }
  census.ancestor_records = 2u * commit_count;
  if (out_census != nullptr) *out_census = census;
  return 2u * commit_count;
}

}  // namespace substrate::direct_adult_core

#endif
