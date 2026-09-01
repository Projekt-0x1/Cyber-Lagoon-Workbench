#ifndef HARDWARE_NATIVE_DIRECT_ADULT_CONSTRUCTOR_META_UPDATE_CUH
#define HARDWARE_NATIVE_DIRECT_ADULT_CONSTRUCTOR_META_UPDATE_CUH

#include "direct_adult_core_constants.cuh"
#include "direct_network_brain.cuh"
#include "direct_network_resident_development.cuh"
#include "direct_exact_history.cuh"

namespace substrate::direct_adult_core {
using direct_network::DirectBrain;

// Kind-4 update authority, second half: accumulated constructor credit
// (#1509 ledger) authorizes a plasticity revision of the constructor's own
// receptor strategy surface. Credit says what mattered; it never prescribes
// what to mutate -- the committed relation moves the resident plasticity law
// one bounded structural step, chained on the cell's revision head like any
// other RecipeRevision event (learning ecology §6/§10, Revision 12 §7/§8).
inline constexpr std::int32_t kConstructorMetaUpdateThresholdQ16 =
    substrate::direct_adult_core::kQ16One / 4;
inline constexpr std::int32_t kConstructorMetaUpdatePlasticityStepQ16 =
    substrate::direct_adult_core::kQ16One / 8;
inline constexpr std::uint32_t kConstructorMetaUpdateCellCapacity = 8u;
// Reserved journal flag marking meta-update revision events; consumption
// bookkeeping rides on their context field (the consumed-through page slot).
inline constexpr std::uint32_t kConstructorMetaUpdateEventFlag = 0x80000000u;

struct ConstructorMetaUpdateCensus {
  std::uint32_t scanned_cells;
  std::uint32_t gated_cells;
  std::uint32_t refused_no_downstream;
  std::uint32_t refused_subthreshold;
  std::uint32_t fired_events;
};

struct ConstructorMetaUpdatePlanEntry {
  std::uint32_t recipe_cell;
  std::int64_t unconsumed_mass_q16;
  std::uint32_t first_unconsumed_slot;
  std::uint32_t evidence_event_tick;
};

struct ConstructorMetaUpdatePlan {
  ConstructorMetaUpdatePlanEntry entries[kConstructorMetaUpdateCellCapacity];
  std::uint32_t entry_count;
  std::uint32_t overflow;
};

namespace constructor_meta_detail {

__device__ inline bool is_constructor_cell(const direct_network::ResidentRecipeCell& cell) {
  return (cell.flags & direct_network::kResidentRecipeLevelCConstructor) != 0u;
}

__device__ inline std::uint32_t candidate_index(
    const std::uint32_t* candidates, std::uint32_t count,
    std::uint32_t recipe_cell) {
  for (std::uint32_t i = 0u; i < count; ++i)
    if (candidates[i] == recipe_cell) return i;
  return kInvalidIndex;
}

}  // namespace constructor_meta_detail

// Pure gate classification: which level-C constructor cells hold enough
// UNCONSUMED downstream child-viability evidence (#1509 depth-stamped
// ancestry-attribution records) to authorize one bounded plasticity step.
// Participation or experience credit alone settles nothing here -- a
// constructor holding massive credit but zero downstream viability evidence
// is refused outright (no self-assessment). Exact-once semantics derive from
// the journal alone: a prior meta-update event consumes every attribution
// record up to its recorded page slot, so replays and repeated passes see
// identical evidence sets without extra resident state.
__device__ inline ConstructorMetaUpdateCensus plan_constructor_meta_updates(
    const DirectBrain& brain, ConstructorMetaUpdatePlan* plan) {
  ConstructorMetaUpdateCensus census{};
  const direct_network::ResidentDevelopmentState* development =
      brain.development;
  if (development == nullptr || brain.recipe_cells == nullptr || plan == nullptr)
    return census;
  const auto& history = development->exact_history;
  if (history.phase_kind != direct_network::DirectExactHistoryKind::empty)
    return census;
  plan->entry_count = 0u;
  plan->overflow = 0u;
  const std::uint32_t cell_count = development->recipe_cell_count;
  const std::uint32_t committed = history.committed_slots;

  std::uint32_t candidates[kConstructorMetaUpdateCellCapacity]{};
  for (std::uint32_t cell = 0u; cell < cell_count; ++cell) {
    if (!constructor_meta_detail::is_constructor_cell(brain.recipe_cells[cell]))
      continue;
    ++census.scanned_cells;
    if (plan->entry_count < kConstructorMetaUpdateCellCapacity)
      candidates[plan->entry_count++] = cell;
    else
      ++plan->overflow;
  }
  const std::uint32_t candidate_count = plan->entry_count;

  // Leading-slot watermark per candidate: every attribution record below it
  // was already consumed by a prior meta-update event. Zero = fresh cell.
  std::uint32_t first_unconsumed[kConstructorMetaUpdateCellCapacity]{};
  for (std::uint32_t cursor = 0u; cursor < committed; ++cursor) {
    const direct_network::DirectExactHistoryRecord& event =
        history.records[cursor];
    if (event.kind != direct_network::DirectExactHistoryKind::recipe_revision ||
        (event.flags & kConstructorMetaUpdateEventFlag) == 0u)
      continue;
    const std::uint32_t slot =
        constructor_meta_detail::candidate_index(candidates, candidate_count,
                                                 event.source);
    if (slot == kInvalidIndex) continue;
    if (event.context > first_unconsumed[slot])
      first_unconsumed[slot] = event.context;
  }

  std::int64_t unconsumed_mass[kConstructorMetaUpdateCellCapacity]{};
  std::uint32_t unconsumed_records[kConstructorMetaUpdateCellCapacity]{};
  std::uint32_t latest_evidence_tick[kConstructorMetaUpdateCellCapacity]{};
  for (std::uint32_t cursor = 0u; cursor < committed; ++cursor) {
    const direct_network::DirectExactHistoryRecord& record =
        history.records[cursor];
    // Depth-stamped ancestry attribution from the #1509 ledger: value carries
    // the ancestry hop, unset (kInvalidIndex) marks direct settlement credit
    // that attributes no ancestor responsibility.
    if (record.kind != direct_network::DirectExactHistoryKind::recipe_commit ||
        record.value == kInvalidIndex)
      continue;
    const std::uint32_t slot = constructor_meta_detail::candidate_index(
        candidates, candidate_count, record.source);
    if (slot == kInvalidIndex || cursor < first_unconsumed[slot]) continue;
    ++unconsumed_records[slot];
    unconsumed_mass[slot] += record.resource_delta;
    if (record.event_tick > latest_evidence_tick[slot])
      latest_evidence_tick[slot] = record.event_tick;
  }

  plan->entry_count = 0u;
  for (std::uint32_t slot = 0u; slot < candidate_count; ++slot) {
    if (unconsumed_records[slot] == 0u) {
      ++census.refused_no_downstream;
      continue;
    }
    if (unconsumed_mass[slot] < kConstructorMetaUpdateThresholdQ16) {
      ++census.refused_subthreshold;
      continue;
    }
    plan->entries[plan->entry_count++] = ConstructorMetaUpdatePlanEntry{
        candidates[slot], unconsumed_mass[slot], first_unconsumed[slot],
        latest_evidence_tick[slot]};
    ++census.gated_cells;
  }
  return census;
}

// Commits one bounded structural step per gated constructor cell inside a
// single atomic exact-history phase: an immutable RecipeRevision event moves
// the cell's revision head, then the resident body applies it and advances
// the receptor plasticity law exactly one step. Fired meta events mint no
// viability credit themselves -- structural authority moves support only.
__device__ inline std::uint32_t commit_constructor_meta_updates(
    const DirectBrain& brain, std::uint32_t commit_tick,
    const ConstructorMetaUpdatePlan& plan) {
  if (brain.development == nullptr || brain.recipe_cells == nullptr ||
      plan.entry_count == 0u ||
      plan.entry_count > kConstructorMetaUpdateCellCapacity)
    return 0u;
  direct_network::DirectExactHistoryHotPage& history =
      brain.development->exact_history;
  if (history.sealed != 0u ||
      history.phase_kind != direct_network::DirectExactHistoryKind::empty ||
      plan.entry_count >
          direct_network::kDirectExactHistoryHotPageCapacity - history.committed_slots)
    return 0u;
  ConstructorMetaUpdatePlan authorized{};
  const ConstructorMetaUpdateCensus census =
      plan_constructor_meta_updates(brain, &authorized);
  if (census.gated_cells != plan.entry_count ||
      authorized.entry_count != plan.entry_count)
    return 0u;
  for (std::uint32_t i = 0u; i < plan.entry_count; ++i) {
    const ConstructorMetaUpdatePlanEntry& supplied = plan.entries[i];
    const ConstructorMetaUpdatePlanEntry& derived = authorized.entries[i];
    if (supplied.recipe_cell != derived.recipe_cell ||
        supplied.unconsumed_mass_q16 != derived.unconsumed_mass_q16 ||
        supplied.first_unconsumed_slot != derived.first_unconsumed_slot ||
        supplied.evidence_event_tick != derived.evidence_event_tick)
      return 0u;
  }
  direct_network::DirectExactHistoryRecord staged[kConstructorMetaUpdateCellCapacity];
  for (std::uint32_t i = 0u; i < plan.entry_count; ++i) {
    const ConstructorMetaUpdatePlanEntry& entry = plan.entries[i];
    const direct_network::ResidentRecipeCell& cell = brain.recipe_cells[entry.recipe_cell];
    std::uint64_t evidence_identity =
        direct_network::exact_history_fold_word(0x6d65746175706431ull,
                                                entry.recipe_cell);
    for (std::uint32_t cursor = entry.first_unconsumed_slot;
         cursor < history.committed_slots; ++cursor) {
      const direct_network::DirectExactHistoryRecord& record =
          history.records[cursor];
      if (record.kind != direct_network::DirectExactHistoryKind::recipe_commit ||
          record.value == kInvalidIndex || record.source != entry.recipe_cell)
        continue;
      evidence_identity = direct_network::exact_history_fold_word(
          evidence_identity, record.identity);
    }
    std::uint64_t occurrence_identity =
        direct_network::exact_history_fold_word(evidence_identity, commit_tick);
    occurrence_identity = direct_network::exact_history_fold_word(
        occurrence_identity, entry.first_unconsumed_slot);
    if (!direct_network::stage_resident_recipe_revision_event(
            staged + i, cell, entry.recipe_cell,
            direct_network::ResidentRecipeRevisionAuthority::structural,
            occurrence_identity, evidence_identity, commit_tick,
            entry.evidence_event_tick, history.committed_slots, i,
            kConstructorMetaUpdateEventFlag,
            kConstructorMetaUpdatePlasticityStepQ16))
      return 0u;
  }
  if (!direct_network::begin_exact_history_phase(
          &history, direct_network::DirectExactHistoryKind::recipe_commit,
          plan.entry_count, commit_tick))
    return 0u;
  const std::uint32_t base = history.phase_base;
  for (std::uint32_t i = 0u; i < plan.entry_count; ++i)
    history.records[base + i] = staged[i];
  if (direct_network::finish_exact_history_phase(&history) !=
      plan.entry_count)
    return 0u;
  for (std::uint32_t i = 0u; i < plan.entry_count; ++i) {
    const ConstructorMetaUpdatePlanEntry& entry = plan.entries[i];
    direct_network::ResidentRecipeCell& cell = brain.recipe_cells[entry.recipe_cell];
    if (!direct_network::apply_resident_recipe_revision_event(
            &cell, history.records[base + i], entry.recipe_cell))
      return 0u;
    std::int32_t plasticity =
        cell.receptor_state.plasticity_q16 +
        kConstructorMetaUpdatePlasticityStepQ16;
    if (plasticity > substrate::direct_adult_core::kQ16One)
      plasticity = substrate::direct_adult_core::kQ16One;
    cell.receptor_state.plasticity_q16 = plasticity;
  }
  return plan.entry_count;
}

}  // namespace substrate::direct_adult_core

#endif
