#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>

#include "bcc32_cuda_resident_discourse_plan.cuh"
#include "bcc32_cuda_resident_multistep_commitment.cuh"
#include "bcc32_cuda_resident_plan_eligibility.cuh"
#include "bcc32_cuda_resident_population_surface.cuh"
#include "bcc32_cuda_resident_proposition_tissue.cuh"

// Device-resident bridge from learned surface populations through proposition
// tissue into the persistent discourse-plan ABI.  Unit ids and cell addresses
// remain opaque.  This organ never reads bytes, assigns semantic labels, or
// writes surface anchors; the independent learned grounding organ does that
// only after the complete population trajectory has settled.
namespace bcc32_cuda_resident_proposition_chain {

namespace plan = bcc32_cuda_resident_discourse_plan;
namespace commitment = bcc32_cuda_resident_multistep_commitment;
namespace population_surface = bcc32_cuda_resident_population_surface;
namespace plan_eligibility = substrate::bcc32::resident_plan_eligibility;
namespace tissue = bcc32_cuda_resident_proposition_tissue;

constexpr std::uint32_t kNoUnit = 0xffffffffu;
// Query conditioning writes a sparse exact-unit mask over the complete learned
// vocabulary.  Cache the marked offsets once so the ordered-binding scan does
// not rescan that whole mask for every role.  Overflow is not an admission
// boundary: it retains the original exhaustive comparison path below.
constexpr std::uint32_t kExactUnitCueCacheCapacity = 256u;

struct SettlementWorkspaceView {
  std::uint64_t* cell_scores = nullptr;
  std::uint32_t cell_capacity = 0u;
};

struct SettlementResult {
  std::uint32_t attempted = 0u;
  std::uint32_t staged = 0u;
  std::uint32_t step_count = 0u;
  std::uint32_t cue_cell_count = 0u;
  std::uint32_t cycle_stopped = 0u;
  std::uint32_t capacity_stopped = 0u;
  std::uint64_t tissue_revision = 0u;
  std::uint64_t strongest_score = 0u;
  std::uint64_t uncertain_mass = 0u;
  std::uint32_t source_unit_count = 0u;
  std::uint32_t expanded_cue_count = 0u;
};

struct OrderedSettlementResult {
  std::uint32_t attempted = 0u;
  std::uint32_t staged = 0u;
  std::uint32_t step_count = 0u;
  std::uint32_t claimed_bindings = 0u;
  std::uint32_t qualified_bindings = 0u;
  std::uint32_t qualified_witnessed_bindings = 0u;
  // Observer-only: records whether a learned construction witness lies in the
  // cue's physical population neighbourhood even when its retained unit ids
  // do not literally recur. This does not admit or rank a binding.
  std::uint32_t witnessed_role_cell_overlap_bindings = 0u;
  std::uint32_t max_witnessed_role_cell_overlap = 0u;
  std::uint32_t exact_topic_matches = 0u;
  std::uint32_t selected_binding_index = 0xffffffffu;
  std::uint32_t selected_information_gain = 0u;
  std::uint32_t selected_role_coverage = 0u;
  // THE DOMINANT SELECTION KEY, PREVIOUSLY UNOBSERVABLE. Candidate ranking is
  // lexicographic with episode_coverage FIRST and role_coverage only breaking
  // ties inside the equal-episode_coverage class, so a low
  // selected_role_coverage never meant "no better-overlapping binding
  // existed" -- it meant "not within the winning episode". Reporting the
  // best role coverage found across ALL topic-complete candidates -- not just
  // the winning episode -- is what makes the two distinguishable. A
  // max_role_coverage_any_episode strictly greater than
  // selected_role_coverage means a fact overlapping the question BETTER
  // existed in the store and the primary key discarded it, which indicts the
  // key order. Equality means no better-overlapping fact was ever stored,
  // which indicts acquisition instead. Neither can be inferred from
  // selected_role_coverage alone, which is why every reading of it so far has
  // been ambiguous.
  std::uint32_t selected_episode_coverage = 0u;
  // The co-best count the two-binding join gate tests, and whether it fired.
  // `qualified_bindings` is the ALL-qualified count and cannot answer this.
  std::uint32_t join_gate_qualified_count = 0u;
  std::uint32_t join_gate_entered = 0u;
  std::uint32_t max_role_coverage_any_episode = 0u;
  std::uint32_t episode_spine_steps = 0u;
  std::uint32_t episode_spine_terminal = 0xffffffffu;
  std::uint32_t episode_spine_ambiguous = 0u;
  std::uint32_t linked_dependencies = 0u;
  std::uint32_t capacity_stopped = 0u;
  std::uint64_t tissue_revision = 0u;
};

struct OrderedCandidateRank {
  std::uint32_t episode_coverage = 0u;
  std::uint32_t information_gain = 0u;
  std::uint32_t role_coverage = 0u;
  std::int64_t learned_credit = 0;
  std::uint64_t evidence_revision = 0u;
  std::uint32_t binding_index = 0u;
};

__device__ inline bool ordered_candidate_precedes(
    const OrderedCandidateRank& left, const OrderedCandidateRank& right) {
  if (left.episode_coverage != right.episode_coverage)
    return left.episode_coverage > right.episode_coverage;
  if (left.role_coverage != right.role_coverage)
    return left.role_coverage > right.role_coverage;
  if (left.information_gain != right.information_gain)
    return left.information_gain > right.information_gain;
  if (left.learned_credit != right.learned_credit)
    return left.learned_credit > right.learned_credit;
  if (left.evidence_revision != right.evidence_revision)
    return left.evidence_revision > right.evidence_revision;
  return left.binding_index < right.binding_index;
}

// Exact cue identity and learned surface populations occupy different address
// spaces.  Keep the identity mask as a view over resident unit ids and compare
// proposition roles only with the complete learned population of a marked
// unit.  Flattening these rows into a bounded cell buffer would either truncate
// the cue or accidentally compare motor-cell addresses with surface cells.
struct ExactUnitCueView {
  const std::uint32_t* exact = nullptr;
  std::uint32_t unit_begin = 0u;
  std::uint32_t unit_count = 0u;
  // Optional learned construction evidence. A marked unit with resident
  // question-onset mass may identify a missing argument placeholder, but the
  // numeric unit id and its bytes never do.
  const std::uint32_t* question_onset_mass = nullptr;
  const std::uint32_t* cue_orders = nullptr;
  // Learned pre-contact surface continuity. Consumers see only opaque contact
  // mass; no lexical or grammatical meaning is assigned here.
  const std::uint32_t* cue_scores = nullptr;
  const std::uint32_t* closed_class_mask = nullptr;
};

struct ExactUnitCueCache {
  std::uint32_t offsets[kExactUnitCueCacheCapacity]{};
  std::uint32_t count = 0u;
  std::uint32_t overflow = 0u;
};

enum class OrderedTopicKind : std::uint32_t {
  population = 0u,
  exact_units = 1u,
};

struct OrderedTopicView {
  OrderedTopicKind kind = OrderedTopicKind::population;
  tissue::SparsePopulationView population{};
  ExactUnitCueView exact_units{};
};

static_assert(sizeof(SettlementResult) == 56u);

__device__ inline bool valid_unit_population_view(
    const population_surface::UnitPopulationView& units,
    std::uint32_t cell_capacity) {
  return units.cells != nullptr && units.contact_mass != nullptr &&
         units.unit_count != 0u && units.population_width != 0u &&
         units.population_width <= tissue::kMaximumPopulationCells &&
         units.unit_begin <= kNoUnit - units.unit_count &&
         cell_capacity != 0u;
}

__device__ inline bool append_unique_cell(std::uint32_t cell,
                                         std::uint32_t cell_capacity,
                                         std::uint32_t* cells,
                                         std::uint32_t* count) {
  if (cells == nullptr || count == nullptr || cell >= cell_capacity)
    return false;
  for (std::uint32_t index = 0u; index < *count; ++index)
    if (cells[index] == cell)
      return true;
  if (*count >= tissue::kMaximumPopulationCells)
    return false;
  cells[(*count)++] = cell;
  return true;
}

__device__ inline bool append_unit_population(
    const population_surface::UnitPopulationView& units,
    std::uint32_t cell_capacity, std::uint32_t unit,
    std::uint32_t* cells, std::uint32_t* count) {
  if (!valid_unit_population_view(units, cell_capacity) || cells == nullptr ||
      count == nullptr || unit < units.unit_begin ||
      unit >= units.unit_begin + units.unit_count ||
      units.contact_mass[unit] == 0u)
    return false;
  const std::uint32_t before = *count;
  const std::uint32_t* population =
      units.cells + static_cast<std::size_t>(unit) * units.population_width;
  for (std::uint32_t slot = 0u; slot < units.population_width; ++slot) {
    if (!append_unique_cell(population[slot], cell_capacity, cells, count)) {
      *count = before;
      return false;
    }
  }
  return *count != before;
}

// A ranked motor completion may contain more units than one sparse cue can
// hold.  Admit whole leading learned populations; never truncate a unit or
// reinterpret its rank on the host.
__device__ inline bool cue_population_from_ranked_units(
    const population_surface::UnitPopulationView& units,
    std::uint32_t cell_capacity, const std::uint32_t* ranked_units,
    std::uint32_t ranked_count, std::uint32_t* cells,
    std::uint32_t* cell_count) {
  if (!valid_unit_population_view(units, cell_capacity) ||
      ranked_units == nullptr || ranked_count == 0u || cells == nullptr ||
      cell_count == nullptr)
    return false;
  *cell_count = 0u;
  for (std::uint32_t index = 0u; index < ranked_count; ++index) {
    const std::uint32_t before = *cell_count;
    if (!append_unit_population(units, cell_capacity, ranked_units[index],
                                cells, cell_count)) {
      *cell_count = before;
      continue;
    }
    if (*cell_count == tissue::kMaximumPopulationCells)
      break;
  }
  return *cell_count != 0u;
}

// Settle every reachable observation-backed discourse step before exposing a
// surface anchor.  Causal readout remains separately intervention-gated in
// proposition_tissue. The trajectory itself is formed by the team's canonical
// multistep commitment organ; this adapter only expands ranked learned units
// into opaque cue cells and selects the discourse admission policy.
__device__ __noinline__ bool stage_plan_from_population(
    tissue::TissueView resident_tissue,
    tissue::SparsePopulationView initial_cue,
    SettlementWorkspaceView workspace,
    plan::ResidentDiscoursePlanState* state, SettlementResult* result,
    std::uint32_t surface_revision,
    tissue::CompletionPolicy policy = tissue::CompletionPolicy::discourse,
    std::uint32_t preserve_empty_settling = 1u) {
  if (result != nullptr)
    *result = SettlementResult{};
  if (state == nullptr || resident_tissue.scalars == nullptr ||
      resident_tissue.synapses == nullptr ||
      resident_tissue.synapse_capacity == 0u || workspace.cell_scores == nullptr ||
      workspace.cell_capacity < resident_tissue.cell_capacity ||
      !tissue::valid_population(initial_cue, resident_tissue.cell_capacity) ||
      !plan::valid(*state) ||
      (state->status != plan::PlanStatus::empty &&
       state->status != plan::PlanStatus::completed &&
       state->status != plan::PlanStatus::invalidated))
    return false;

  if (result != nullptr) {
    result->attempted = 1u;
    result->cue_cell_count = initial_cue.count;
    result->tissue_revision = resident_tissue.scalars->revision;
  }
  std::uint32_t completed_cells[tissue::kDefaultCompletionCells]{};
  std::uint64_t completed_scores[tissue::kDefaultCompletionCells]{};
  commitment::CommitmentResult commitment_result{};
  commitment::form_multistep_plan(
      resident_tissue, initial_cue, tissue::SparsePopulationView{},
      {workspace.cell_scores, workspace.cell_capacity, completed_cells,
       completed_scores, tissue::kDefaultCompletionCells},
      state, surface_revision, plan::kMaxSteps, &commitment_result,
      policy, preserve_empty_settling);
  if (result != nullptr) {
    result->staged = commitment_result.formed;
    result->step_count = commitment_result.planned_steps;
    result->cycle_stopped = commitment_result.cycle_cut;
    result->capacity_stopped = commitment_result.capacity_cut;
    result->strongest_score = commitment_result.strongest_score;
    result->uncertain_mass = commitment_result.uncertain_mass;
  }
  return commitment_result.formed != 0u;
}

__device__ inline bool stage_plan_from_units(
    tissue::TissueView resident_tissue,
    const population_surface::UnitPopulationView& units,
    const std::uint32_t* cue_units, std::uint32_t cue_unit_count,
    SettlementWorkspaceView workspace,
    plan::ResidentDiscoursePlanState* state, SettlementResult* result,
    std::uint32_t surface_revision,
    tissue::CompletionPolicy policy = tissue::CompletionPolicy::discourse,
    std::uint32_t preserve_empty_settling = 1u) {
  if (result != nullptr)
    *result = SettlementResult{};
  std::uint32_t cue_cells[tissue::kMaximumPopulationCells]{};
  std::uint32_t cue_count = 0u;
  if (!cue_population_from_ranked_units(
          units, resident_tissue.cell_capacity, cue_units, cue_unit_count,
          cue_cells, &cue_count)) {
    if (result != nullptr)
      result->source_unit_count = cue_unit_count;
    return false;
  }
  SettlementResult staged{};
  const bool formed = stage_plan_from_population(
      resident_tissue, {cue_cells, cue_count}, workspace, state, &staged,
      surface_revision, policy, preserve_empty_settling);
  staged.source_unit_count = cue_unit_count;
  staged.expanded_cue_count = cue_count;
  if (result != nullptr)
    *result = staged;
  return formed;
}

// Recover the bounded leading sparse cue in its raw-contact order. Exact
// surface representatives and their populations are learned resident matter;
// neither numeric unit order nor the removable episode store selects a topic.
__device__ inline bool stage_plan_from_exact_sequence(
    tissue::TissueView resident_tissue,
    const population_surface::UnitPopulationView& units,
    const std::uint32_t* exact_units, const std::uint32_t* cue_orders,
    std::uint32_t exact_unit_count, SettlementWorkspaceView workspace,
    plan::ResidentDiscoursePlanState* state, SettlementResult* result,
    std::uint32_t surface_revision,
    tissue::CompletionPolicy policy = tissue::CompletionPolicy::causal,
    std::uint32_t preserve_empty_settling = 0u) {
  if (result != nullptr)
    *result = SettlementResult{};
  if (exact_units == nullptr || cue_orders == nullptr ||
      exact_unit_count != units.unit_count || exact_unit_count == 0u)
    return false;
  std::uint32_t ranked_units[tissue::kMaximumPopulationCells]{};
  std::uint32_t ranked_count = 0u;
  std::uint32_t maximum_order = 0u;
  for (std::uint32_t offset = 0u; offset < exact_unit_count; ++offset) {
    if (exact_units[offset] != 0u &&
        cue_orders[units.unit_begin + offset] != kNoUnit &&
        cue_orders[units.unit_begin + offset] > maximum_order)
      maximum_order = cue_orders[units.unit_begin + offset];
  }
  for (std::uint32_t order = 1u;
       order <= maximum_order &&
       ranked_count < tissue::kMaximumPopulationCells;
       ++order) {
    std::uint32_t selected = kNoUnit;
    for (std::uint32_t offset = 0u; offset < exact_unit_count; ++offset) {
      if (exact_units[offset] == 0u ||
          cue_orders[units.unit_begin + offset] != order)
        continue;
      if (selected != kNoUnit)
        return false;
      selected = units.unit_begin + offset;
    }
    if (selected != kNoUnit)
      ranked_units[ranked_count++] = selected;
  }
  if (ranked_count == 0u)
    return false;
  std::uint32_t cue_cells[tissue::kMaximumPopulationCells]{};
  std::uint32_t cue_count = 0u;
  if (!cue_population_from_ranked_units(
          units, resident_tissue.cell_capacity, ranked_units, ranked_count,
          cue_cells, &cue_count)) {
    if (result != nullptr)
      result->source_unit_count = ranked_count;
    return false;
  }
  SettlementResult staged{};
  const bool formed = stage_plan_from_population(
      resident_tissue, {cue_cells, cue_count}, workspace, state, &staged,
      surface_revision, policy, preserve_empty_settling);
  staged.source_unit_count = ranked_count;
  staged.expanded_cue_count = cue_count;
  if (result != nullptr)
    *result = staged;
  return formed;
}

__global__ void stage_plan_from_units_kernel(
    tissue::TissueView resident_tissue,
    population_surface::UnitPopulationView units,
    const std::uint32_t* cue_units, std::uint32_t cue_unit_count,
    SettlementWorkspaceView workspace,
    plan::ResidentDiscoursePlanState* state, SettlementResult* result,
    std::uint32_t surface_revision) {
  if (blockIdx.x != 0u || threadIdx.x != 0u)
    return;
  stage_plan_from_units(resident_tissue, units, cue_units, cue_unit_count,
                        workspace, state, result, surface_revision);
}

// Canonical raw-contact seam. The distributed motor already retains the
// complete cue as an opaque cell population. Proposition settlement consumes
// that resident population directly; it must never reinterpret the legacy
// motor-completion tape (which is downstream candidate content) as a query.
__device__ inline bool stage_plan_from_resident_cue(
    tissue::TissueView resident_tissue, const std::uint32_t* cue_cells,
    const std::uint32_t* cue_count, std::uint32_t cue_capacity,
    SettlementWorkspaceView workspace,
    plan::ResidentDiscoursePlanState* state, SettlementResult* result,
    std::uint32_t surface_revision) {
  if (result != nullptr)
    *result = SettlementResult{};
  if (cue_cells == nullptr || cue_count == nullptr || cue_capacity == 0u ||
      *cue_count == 0u || *cue_count > cue_capacity ||
      *cue_count > tissue::kMaximumPopulationCells)
    return false;
  return stage_plan_from_population(
      resident_tissue, {cue_cells, *cue_count}, workspace, state, result,
      surface_revision);
}

__global__ void stage_plan_from_resident_cue_kernel(
    tissue::TissueView resident_tissue, const std::uint32_t* cue_cells,
    const std::uint32_t* cue_count, std::uint32_t cue_capacity,
    SettlementWorkspaceView workspace,
    plan::ResidentDiscoursePlanState* state, SettlementResult* result,
    std::uint32_t surface_revision) {
  if (blockIdx.x != 0u || threadIdx.x != 0u)
    return;
  stage_plan_from_resident_cue(resident_tissue, cue_cells, cue_count,
                               cue_capacity,
                               workspace, state, result, surface_revision);
}

__device__ inline bool ordered_binding_rheme_links(
    const tissue::OrderedRoleBindingEvidence& earlier,
    const tissue::OrderedRoleBindingEvidence& later) {
  const auto rheme = tissue::ordered_binding_role(earlier, 2u);
  return tissue::exact_population_equals(later.role_cells[0],
                                         later.role_counts[0], rheme) ||
         tissue::exact_population_equals(later.role_cells[2],
                                         later.role_counts[2], rheme);
}

__device__ inline bool ordered_binding_contains_identity(
    const tissue::OrderedRoleBindingEvidence& binding, std::uint32_t unit) {
  for (std::uint32_t role = 0u; role < tissue::kOrderedBindingRoleCount;
       ++role)
    for (std::uint32_t index = 0u; index < binding.role_unit_counts[role];
         ++index)
      if (binding.role_units[role][index] == unit)
        return true;
  return false;
}

__device__ inline bool ordered_binding_precedes(
    const tissue::OrderedRoleBindingEvidence& left, std::uint32_t left_index,
    const tissue::OrderedRoleBindingEvidence& right, std::uint32_t right_index) {
  if (left.last_evidence_revision != right.last_evidence_revision)
    return left.last_evidence_revision > right.last_evidence_revision;
  return left_index < right_index;
}

__device__ inline bool complete_population_contained(
    tissue::SparsePopulationView complete,
    tissue::SparsePopulationView cue_union) {
  if (complete.count == 0u || cue_union.count < complete.count)
    return false;
  for (std::uint32_t cell = 0u; cell < complete.count; ++cell)
    if (!tissue::population_contains(cue_union, complete.cells[cell]))
      return false;
  return true;
}

__device__ inline tissue::SparsePopulationView complete_unit_population(
    const population_surface::UnitPopulationView& units, std::uint32_t unit) {
  if (unit < units.unit_begin || unit >= units.unit_begin + units.unit_count ||
      units.contact_mass[unit] == 0u)
    return {};
  const std::uint32_t count = units.population_count == nullptr
                                  ? units.population_width
                                  : units.population_count[unit];
  if (count == 0u || count > units.population_width)
    return {};
  return {units.cells + static_cast<std::size_t>(unit) * units.population_width,
          count};
}

// Preflight every exact-marked row.  One malformed row abstains the complete
// ordered attempt; silently skipping it would turn partial surface formation
// into a semantic selector.
__device__ inline bool prepare_exact_unit_cue(
    ExactUnitCueView cue,
    const population_surface::UnitPopulationView& units,
    std::uint32_t cell_capacity, ExactUnitCueCache* cache) {
  if (cue.exact == nullptr || cue.unit_count == 0u ||
      cue.unit_begin != units.unit_begin || cue.unit_count != units.unit_count ||
      cache == nullptr)
    return false;
  *cache = ExactUnitCueCache{};
  bool any = false;
  for (std::uint32_t offset = 0u; offset < cue.unit_count; ++offset) {
    const std::uint32_t unit = cue.unit_begin + offset;
    if (cue.exact[offset] == 0u ||
        (cue.closed_class_mask != nullptr && cue.closed_class_mask[unit] != 0u))
      continue;
    any = true;
    const auto population = complete_unit_population(
        units, unit);
    if (!tissue::valid_distinct_population(population, cell_capacity))
      return false;
    if (cache->count < kExactUnitCueCacheCapacity)
      cache->offsets[cache->count++] = offset;
    else
      cache->overflow = 1u;
  }
  return any;
}

__device__ inline bool exact_unit_cue_content_identity(
    const ExactUnitCueView& cue, std::uint32_t unit) {
  return unit >= cue.unit_begin && unit - cue.unit_begin < cue.unit_count &&
         (cue.closed_class_mask == nullptr || cue.closed_class_mask[unit] == 0u);
}

// A query often names several units that belong to different local A/K/B
// events in the same acquired contact. Rank that contact by the union of its
// retained identities, rather than asking one local triple to contain the
// whole cue. Revisions are physical episode provenance, not labels.
__device__ inline std::uint32_t exact_cue_episode_coverage(
    tissue::TissueView resident_tissue, ExactUnitCueView cue,
    const ExactUnitCueCache& cache, std::uint64_t evidence_revision,
    const std::uint32_t* construction_witnesses,
    std::uint32_t construction_witness_capacity) {
  if (cue.exact == nullptr || evidence_revision == 0u ||
      resident_tissue.ordered_bindings == nullptr)
    return 0u;
  const std::uint64_t episode = evidence_revision >> 32u;
  std::uint32_t coverage = 0u;
  const std::uint32_t candidate_count =
      cache.overflow == 0u ? cache.count : cue.unit_count;
  for (std::uint32_t candidate = 0u; candidate < candidate_count; ++candidate) {
    const std::uint32_t offset = cache.overflow == 0u ? cache.offsets[candidate]
                                                       : candidate;
    if (offset >= cue.unit_count || cue.exact[offset] == 0u)
      continue;
    const std::uint32_t unit = cue.unit_begin + offset;
    if (!exact_unit_cue_content_identity(cue, unit))
      continue;
    bool found = false;
    for (std::uint32_t index = 0u;
         index < resident_tissue.ordered_binding_capacity && !found; ++index) {
      const auto& binding = resident_tissue.ordered_bindings[index];
      if (binding.claimed == 0u ||
          (binding.last_evidence_revision >> 32u) != episode ||
          (construction_witnesses != nullptr &&
           (index >= construction_witness_capacity ||
            construction_witnesses[index] == 0xffffffffu)))
        continue;
      for (std::uint32_t role = 0u;
           role < tissue::kOrderedBindingRoleCount && !found; ++role) {
        for (std::uint32_t identity = 0u;
             identity < binding.role_unit_counts[role]; ++identity) {
          if (binding.role_units[role][identity] == unit) {
            found = true;
            break;
          }
        }
      }
    }
    coverage += found ? 1u : 0u;
  }
  return coverage;
}

__device__ inline bool ordered_topic_valid(
    OrderedTopicView topic,
    const population_surface::UnitPopulationView& units,
    std::uint32_t cell_capacity, ExactUnitCueCache* exact_cache) {
  if (topic.kind == OrderedTopicKind::population)
    return tissue::valid_distinct_population(topic.population, cell_capacity);
  return topic.kind == OrderedTopicKind::exact_units &&
         prepare_exact_unit_cue(topic.exact_units, units, cell_capacity,
                                exact_cache);
}

__device__ inline std::uint32_t population_overlap(
    tissue::SparsePopulationView left, tissue::SparsePopulationView right) {
  std::uint32_t overlap = 0u;
  for (std::uint32_t l = 0u; l < left.count; ++l) {
    bool present = false;
    for (std::uint32_t r = 0u; r < right.count; ++r)
      present = present || left.cells[l] == right.cells[r];
    overlap += present;
  }
  return overlap;
}

__device__ inline std::uint32_t exact_unit_cue_population_overlap(
    ExactUnitCueView cue,
    const population_surface::UnitPopulationView& units,
    tissue::SparsePopulationView population,
    const ExactUnitCueCache& cache) {
  std::uint32_t best_overlap = 0u;
  if (cache.overflow == 0u) {
    for (std::uint32_t index = 0u; index < cache.count; ++index) {
      const std::uint32_t offset = cache.offsets[index];
      const auto exact_population = complete_unit_population(
          units, cue.unit_begin + offset);
      const std::uint32_t overlap = population_overlap(exact_population, population);
      if (overlap > best_overlap) best_overlap = overlap;
    }
    return best_overlap;
  }
  for (std::uint32_t offset = 0u; offset < cue.unit_count; ++offset) {
    if (cue.exact[offset] == 0u ||
        !exact_unit_cue_content_identity(cue, cue.unit_begin + offset))
      continue;
    const auto exact_population = complete_unit_population(
        units, cue.unit_begin + offset);
    const std::uint32_t overlap = population_overlap(exact_population, population);
    if (overlap > best_overlap) best_overlap = overlap;
  }
  return best_overlap;
}

__device__ inline bool exact_unit_cue_matches_population(
    ExactUnitCueView cue,
    const population_surface::UnitPopulationView& units,
    tissue::SparsePopulationView population,
    const ExactUnitCueCache& cache) {
  if (cache.overflow == 0u) {
    for (std::uint32_t index = 0u; index < cache.count; ++index) {
      const auto exact_population = complete_unit_population(
          units, cue.unit_begin + cache.offsets[index]);
      if (complete_population_contained(population, exact_population))
        return true;
    }
    return false;
  }
  for (std::uint32_t offset = 0u; offset < cue.unit_count; ++offset) {
    if (cue.exact[offset] == 0u ||
        !exact_unit_cue_content_identity(cue, cue.unit_begin + offset))
      continue;
    const auto exact_population = complete_unit_population(
        units, cue.unit_begin + offset);
    if (complete_population_contained(population, exact_population))
      return true;
  }
  return false;
}

__device__ inline std::uint32_t exact_unit_cue_role_identity_overlap(
    ExactUnitCueView cue, const tissue::OrderedRoleBindingEvidence& binding,
    std::uint32_t role) {
  if (cue.exact == nullptr || role >= tissue::kOrderedBindingRoleCount)
    return 0u;
  std::uint32_t overlap = 0u;
  for (std::uint32_t index = 0u;
       index < binding.role_unit_counts[role]; ++index) {
    const std::uint32_t unit = binding.role_units[role][index];
    if (!exact_unit_cue_content_identity(cue, unit))
      continue;
    overlap += cue.exact[unit - cue.unit_begin] != 0u;
  }
  return overlap;
}

// Preserve a learned near-form contact beside exact identity. The field is
// formed before the current contact is assimilated, so it cannot manufacture
// its own evidence during response selection.
__device__ inline std::uint32_t cue_role_identity_overlap(
    ExactUnitCueView cue, const tissue::OrderedRoleBindingEvidence& binding,
    std::uint32_t role) {
  if (role >= tissue::kOrderedBindingRoleCount)
    return 0u;
  std::uint32_t overlap = 0u;
  for (std::uint32_t index = 0u;
       index < binding.role_unit_counts[role]; ++index) {
    const std::uint32_t unit = binding.role_units[role][index];
    if (!exact_unit_cue_content_identity(cue, unit))
      continue;
    const std::uint32_t offset = unit - cue.unit_begin;
    overlap += (cue.exact != nullptr && cue.exact[offset] != 0u) ||
               (cue.cue_scores != nullptr && cue.cue_scores[offset] != 0u);
  }
  return overlap;
}

__device__ inline bool exact_unit_cue_contains_identity(
    const ExactUnitCueView& cue, std::uint32_t unit) {
  return cue.exact != nullptr && exact_unit_cue_content_identity(cue, unit) &&
         cue.exact[unit - cue.unit_begin] != 0u;
}

__device__ inline std::uint32_t exact_unit_cue_role_identity_novelty(
    const ExactUnitCueView& cue,
    const tissue::OrderedRoleBindingEvidence& binding) {
  std::uint32_t novelty = 0u;
  for (std::uint32_t role = 0u; role < tissue::kOrderedBindingRoleCount;
       ++role) {
    for (std::uint32_t index = 0u;
         index < binding.role_unit_counts[role]; ++index) {
      novelty += !exact_unit_cue_contains_identity(
          cue, binding.role_units[role][index]);
    }
  }
  return novelty;
}

__device__ inline std::uint32_t exact_unit_cue_role_earliest_order(
    ExactUnitCueView cue, const tissue::OrderedRoleBindingEvidence& binding,
    std::uint32_t role) {
  if (cue.cue_orders == nullptr || role >= tissue::kOrderedBindingRoleCount)
    return kNoUnit;
  std::uint32_t earliest = kNoUnit;
  for (std::uint32_t index = 0u;
       index < binding.role_unit_counts[role]; ++index) {
    const std::uint32_t unit = binding.role_units[role][index];
    if (!exact_unit_cue_content_identity(cue, unit) ||
        cue.exact[unit - cue.unit_begin] == 0u)
      continue;
    const std::uint32_t order = cue.cue_orders[unit];
    if (order < earliest)
      earliest = order;
  }
  return earliest;
}

__device__ inline bool exact_unit_cue_question_marks_population(
    ExactUnitCueView cue,
    const population_surface::UnitPopulationView& units,
    tissue::SparsePopulationView population,
    const ExactUnitCueCache& cache) {
  if (cue.question_onset_mass == nullptr)
    return false;
  if (cache.overflow == 0u) {
    for (std::uint32_t index = 0u; index < cache.count; ++index) {
      const std::uint32_t offset = cache.offsets[index];
      const std::uint32_t unit = cue.unit_begin + offset;
      if (cue.question_onset_mass[unit] == 0u)
        continue;
      const auto exact_population = complete_unit_population(units, unit);
      if (tissue::exact_population_equals(exact_population.cells,
                                          exact_population.count, population))
        return true;
    }
    return false;
  }
  for (std::uint32_t offset = 0u; offset < cue.unit_count; ++offset) {
    const std::uint32_t unit = cue.unit_begin + offset;
    if (cue.exact[offset] == 0u || cue.question_onset_mass[unit] == 0u)
      continue;
    const auto exact_population = complete_unit_population(units, unit);
    if (tissue::exact_population_equals(exact_population.cells,
                                        exact_population.count, population))
      return true;
  }
  return false;
}

__device__ inline std::uint32_t ordered_topic_population_overlap(
    OrderedTopicView topic,
    const population_surface::UnitPopulationView& units,
    tissue::SparsePopulationView population,
    const ExactUnitCueCache& exact_cache) {
  if (topic.kind == OrderedTopicKind::population)
    return complete_population_contained(population, topic.population)
               ? population.count
               : 0u;
  return exact_unit_cue_population_overlap(topic.exact_units, units, population,
                                           exact_cache);
}

__device__ inline std::uint32_t exact_anchor_for_population(
    const population_surface::UnitPopulationView& units,
    tissue::SparsePopulationView population) {
  std::uint32_t selected = population_surface::kInvalidUnit;
  std::uint32_t selected_context_mass = 0u;
  std::uint32_t selected_overlap = 0u;
  std::uint64_t selected_mass = 0u;
  bool tied = false;
  for (std::uint32_t offset = 0u; offset < units.unit_count; ++offset) {
    const std::uint32_t unit = units.unit_begin + offset;
    if (units.contact_mass[unit] == 0u)
      continue;
    const std::uint32_t count = units.population_count == nullptr
                                    ? units.population_width
                                    : units.population_count[unit];
    if (count == 0u || count > units.population_width)
      continue;
    const std::uint32_t* cells =
        units.cells + static_cast<std::size_t>(unit) * units.population_width;
    const std::uint32_t* context_cells =
        units.context_cells == nullptr
            ? nullptr
            : units.context_cells +
                  static_cast<std::size_t>(unit) * units.population_width;
    std::uint32_t overlap = 0u;
    std::uint32_t context_mass = 0u;
    for (std::uint32_t slot = 0u; slot < count; ++slot) {
      bool shared = false;
      for (std::uint32_t member = 0u; member < population.count; ++member)
        shared |= cells[slot] == population.cells[member];
      if (!shared) continue;
      ++overlap;
    }
    if (context_cells != nullptr &&
        units.population_context_mass != nullptr) {
      for (std::uint32_t slot = 0u; slot < count; ++slot) {
        bool shared = false;
        for (std::uint32_t member = 0u; member < population.count; ++member)
          shared |= context_cells[slot] == population.cells[member];
        if (!shared) continue;
        context_mass += units.population_context_mass[
            static_cast<std::size_t>(unit) * units.population_width + slot];
      }
    }
    if (overlap == 0u)
      continue;
    if (context_mass > selected_context_mass ||
        (context_mass == selected_context_mass &&
         (overlap > selected_overlap ||
          (overlap == selected_overlap && overlap != 0u &&
           units.contact_mass[unit] > selected_mass)))) {
      selected = unit;
      selected_context_mass = context_mass;
      selected_overlap = overlap;
      selected_mass = units.contact_mass[unit];
      tied = false;
    } else if (context_mass == selected_context_mass &&
               overlap != 0u && overlap == selected_overlap &&
               units.contact_mass[unit] == selected_mass) {
      tied = true;
    }
  }
  return tied ? population_surface::kInvalidUnit : selected;
}

__device__ inline bool ground_ordered_binding_plan(
    tissue::TissueView resident_tissue,
    const population_surface::UnitPopulationView& units,
    plan::ResidentDiscoursePlanState* state, std::uint32_t surface_revision) {
  if (state == nullptr || !plan::valid(*state) ||
      state->status != plan::PlanStatus::settling ||
      !valid_unit_population_view(units, resident_tissue.cell_capacity) ||
      state->step_count == 0u ||
      state->step_count * tissue::kOrderedBindingRoleCount >
          plan::kMaxAnchorReferences)
    return false;
  std::uint32_t anchors[plan::kMaxSteps][tissue::kOrderedBindingRoleCount]{};
  for (std::uint32_t step = 0u; step < state->step_count; ++step) {
    const plan::PlanStep& plan_step = state->steps[step];
    if (plan_step.reference_kind != plan::PlanReferenceKind::ordered_binding ||
        plan_step.population_count != 1u)
      return false;
    const std::uint32_t reference =
        state->population_references[plan_step.population_begin];
    if (reference >= resident_tissue.ordered_binding_capacity)
      return false;
    const auto& binding = resident_tissue.ordered_bindings[reference];
    if (!tissue::ordered_binding_structurally_intact(
            binding, resident_tissue.cell_capacity))
      return false;
    for (std::uint32_t role = 0u; role < tissue::kOrderedBindingRoleCount;
         ++role) {
      anchors[step][role] = population_surface::kInvalidUnit;
      if (binding.role_unit_counts[role] == 1u) {
        const std::uint32_t unit = binding.role_units[role][0];
        if (unit >= units.unit_begin &&
            unit - units.unit_begin < units.unit_count &&
            units.contact_mass[unit] != 0u)
          anchors[step][role] = unit;
      }
      if (anchors[step][role] == population_surface::kInvalidUnit)
        anchors[step][role] = exact_anchor_for_population(
            units, tissue::ordered_binding_role(binding, role));
      if (anchors[step][role] == population_surface::kInvalidUnit)
        return false;
    }
  }
  state->anchor_reference_count = 0u;
  for (std::uint32_t step = 0u; step < state->step_count; ++step) {
    plan::PlanStep& plan_step = state->steps[step];
    plan_step.anchor_begin = state->anchor_reference_count;
    plan_step.anchor_count = tissue::kOrderedBindingRoleCount;
    for (std::uint32_t role = 0u; role < tissue::kOrderedBindingRoleCount;
         ++role)
      state->anchor_references[state->anchor_reference_count++] =
          anchors[step][role];
  }
  state->surface_revision = surface_revision;
  return plan::commit_population_plan(state);
}

// Settle the complete bounded set of topic-owned proposition records before
// emission. Each Plan reference is one record index with an explicit kind; A,
// K, and B are never flattened into an untyped cell list. A dependency is
// written only where the previous record's retained rheme population exactly
// recurs in the next record. Every step remains independently qualified.
__device__ inline bool stage_plan_from_ordered_topic_view(
    tissue::TissueView resident_tissue, OrderedTopicView topic,
    tissue::OrderedBindingQualification qualification,
    const population_surface::UnitPopulationView& units,
    plan::ResidentDiscoursePlanState* state, OrderedSettlementResult* result,
    std::uint32_t surface_revision,
    const plan_eligibility::ResidentPlanEligibilityState* eligibility = nullptr,
    const std::uint32_t* construction_witnesses = nullptr,
    std::uint32_t construction_witness_capacity = 0u,
    // Defaulted false == join enabled == today's behaviour. See the lesion
    // comment at the join gate below.
    bool ordered_join_lesioned = false) {
  if (result != nullptr)
    *result = OrderedSettlementResult{};
  ExactUnitCueCache exact_cache{};
  if (state == nullptr || resident_tissue.scalars == nullptr ||
      resident_tissue.ordered_bindings == nullptr ||
      resident_tissue.ordered_binding_capacity == 0u ||
      !valid_unit_population_view(units, resident_tissue.cell_capacity) ||
      !ordered_topic_valid(topic, units, resident_tissue.cell_capacity,
                           &exact_cache) ||
      !plan::valid(*state) ||
      (state->status != plan::PlanStatus::empty &&
       state->status != plan::PlanStatus::completed &&
       state->status != plan::PlanStatus::invalidated))
    return false;
  if (result != nullptr) {
    result->attempted = 1u;
    result->tissue_revision = resident_tissue.scalars->revision;
  }

  std::uint32_t indices[plan::kMaxSteps]{};
  OrderedCandidateRank ranked[plan::kMaxSteps]{};
  std::uint32_t qualified_count = 0u;
  std::uint32_t exact_topic_matches = 0u;
  std::uint32_t best_episode_coverage = 0u;
  std::uint32_t best_information_gain = 0u;
  std::uint32_t best_role_coverage = 0u;
  // Tracked over EVERY topic-complete candidate, independent of the
  // lexicographic winner set, so it survives the equal-best filter that
  // discards non-winning candidates before they are ranked.
  std::uint32_t max_role_coverage_any_episode = 0u;
  std::int64_t best_learned_credit = INT64_MIN;
  bool overflow = false;
  for (std::uint32_t index = 0u;
       index < resident_tissue.ordered_binding_capacity; ++index) {
    const auto& binding = resident_tissue.ordered_bindings[index];
    if (binding.claimed == 0u)
      continue;
    if (result != nullptr) ++result->claimed_bindings;
    const auto agent_population = tissue::ordered_binding_role(binding, 0u);
    const auto predicate_population = tissue::ordered_binding_role(binding, 1u);
    const auto patient_population = tissue::ordered_binding_role(binding, 2u);
    const bool witnessed = construction_witnesses != nullptr &&
                           index < construction_witness_capacity &&
                           construction_witnesses[index] != 0xffffffffu;
    if (topic.kind == OrderedTopicKind::exact_units && !witnessed)
      continue;
    if (result != nullptr && witnessed &&
        topic.kind == OrderedTopicKind::exact_units) {
      const std::uint32_t role_cell_overlap =
          exact_unit_cue_population_overlap(topic.exact_units, units,
                                             agent_population, exact_cache) +
          exact_unit_cue_population_overlap(topic.exact_units, units,
                                             predicate_population, exact_cache) +
          exact_unit_cue_population_overlap(topic.exact_units, units,
                                             patient_population, exact_cache);
      if (role_cell_overlap != 0u) {
        ++result->witnessed_role_cell_overlap_bindings;
        if (role_cell_overlap > result->max_witnessed_role_cell_overlap)
          result->max_witnessed_role_cell_overlap = role_cell_overlap;
      }
    }
    const bool exact_episode =
        qualification == tissue::OrderedBindingQualification::exact_episode &&
        topic.kind == OrderedTopicKind::exact_units;
    const bool retained_identity = binding.role_unit_counts[0] != 0u &&
                                   binding.role_unit_counts[1] != 0u &&
                                   binding.role_unit_counts[2] != 0u;
    const std::uint32_t agent_overlap =
        exact_episode && retained_identity
            ? cue_role_identity_overlap(topic.exact_units, binding, 0u)
            : exact_episode
            ? (exact_unit_cue_matches_population(
                   topic.exact_units, units, agent_population, exact_cache)
                   ? agent_population.count
                   : 0u)
            : ordered_topic_population_overlap(topic, units, agent_population,
                                               exact_cache);
    const std::uint32_t patient_overlap =
        exact_episode && retained_identity
            ? cue_role_identity_overlap(topic.exact_units, binding, 2u)
            : exact_episode
            ? (exact_unit_cue_matches_population(
                   topic.exact_units, units, patient_population, exact_cache)
                   ? patient_population.count
                   : 0u)
            : ordered_topic_population_overlap(topic, units, patient_population,
                                               exact_cache);
    const std::uint32_t connective_overlap =
        topic.kind == OrderedTopicKind::exact_units
            ? (exact_episode && retained_identity
                   ? cue_role_identity_overlap(topic.exact_units, binding, 1u)
               : exact_episode
                   ? (exact_unit_cue_matches_population(
                          topic.exact_units, units, predicate_population,
                          exact_cache)
                          ? predicate_population.count
                          : 0u)
                   : ordered_topic_population_overlap(
                         topic, units, predicate_population, exact_cache))
            : 0u;
    const bool agent_match = agent_overlap != 0u;
    const bool patient_match = patient_overlap != 0u;
    const bool connective_match = connective_overlap != 0u;
    const bool complete_topic = agent_match || patient_match || connective_match;
    if (!complete_topic)
      continue;
    if (exact_episode && retained_identity) {
      std::uint32_t prior_order = 0u;
      for (std::uint32_t role = 0u; role < tissue::kOrderedBindingRoleCount;
           ++role) {
        const std::uint32_t order = exact_unit_cue_role_earliest_order(
            topic.exact_units, binding, role);
        if (order == kNoUnit)
          continue;
        if (prior_order != 0u && order <= prior_order) {
          prior_order = kNoUnit;
          break;
        }
        prior_order = order;
      }
      if (prior_order == kNoUnit)
        continue;
    }
    ++exact_topic_matches;
    if (!tissue::ordered_binding_qualified(
            binding, qualification, resident_tissue.cell_capacity))
      continue;
    if (result != nullptr) ++result->qualified_bindings;
    if (result != nullptr && witnessed)
      ++result->qualified_witnessed_bindings;
    // An exact cue can identify not only the topic entity but also the
    // relation and the opposite entity.  Prefer the complete set of resident
    // records with maximal role coverage.  This is content-addressing over
    // learned populations, not a host ranking: every best-scoring tie remains
    // admissible, and an over-capacity tie still abstains atomically.
    std::uint32_t role_coverage = agent_overlap + patient_overlap;
    if (topic.kind == OrderedTopicKind::exact_units)
      role_coverage += connective_overlap;
    // Recorded HERE, before the lexicographic comparison below drops every
    // candidate that is not equal-best on all keys. Recording it after that
    // filter would only ever restate selected_role_coverage, which is the
    // ambiguity this field exists to remove.
    if (role_coverage > max_role_coverage_any_episode)
      max_role_coverage_any_episode = role_coverage;
    // Every partial cue is resolved by the same resident geometry: overlap
    // names the currently supported matter; novelty is retained role matter
    // absent from that cue. No punctuation, linguistic act, entity class, or
    // question-specific route participates in this competition.
    const std::uint32_t novel_role_mass =
        topic.kind == OrderedTopicKind::exact_units && retained_identity
            ? exact_unit_cue_role_identity_novelty(topic.exact_units, binding)
            : 0u;
    const std::uint32_t information_gain = role_coverage * novel_role_mass;
    const std::uint32_t episode_coverage =
        exact_episode
            ? exact_cue_episode_coverage(
                  resident_tissue, topic.exact_units, exact_cache,
                  binding.last_evidence_revision, construction_witnesses,
                  construction_witness_capacity)
            : role_coverage;
    const std::int64_t learned_credit =
        plan_eligibility::ordered_binding_credit(eligibility, index);
    // Query relevance is learned from the cue's resident overlap. Recency
    // breaks only an otherwise indistinguishable competition: a later
    // distractor must not eclipse a more completely grounded older episode.
    const std::uint64_t best_evidence = qualified_count == 0u
        ? 0u : ranked[0].evidence_revision;
    if (episode_coverage > best_episode_coverage ||
        (episode_coverage == best_episode_coverage &&
         (role_coverage > best_role_coverage ||
          (role_coverage == best_role_coverage &&
           (information_gain > best_information_gain ||
            (information_gain == best_information_gain &&
             (learned_credit > best_learned_credit ||
              (learned_credit == best_learned_credit &&
               binding.last_evidence_revision > best_evidence)))))))) {
      best_episode_coverage = episode_coverage;
      best_information_gain = information_gain;
      best_role_coverage = role_coverage;
      best_learned_credit = learned_credit;
      qualified_count = 0u;
      overflow = false;
    } else if (episode_coverage != best_episode_coverage ||
               role_coverage != best_role_coverage ||
               information_gain != best_information_gain ||
               learned_credit != best_learned_credit ||
               binding.last_evidence_revision != best_evidence) {
      continue;
    }
    const OrderedCandidateRank candidate{
        episode_coverage, information_gain, role_coverage, learned_credit,
        binding.last_evidence_revision, index};
    std::uint32_t insert = qualified_count;
    while (insert > 0u &&
           ordered_candidate_precedes(candidate, ranked[insert - 1u]))
      --insert;
    if (insert >= plan::kMaxSteps) {
      overflow = true;
      continue;
    }
    const std::uint32_t retained =
        min(qualified_count, plan::kMaxSteps - 1u);
    for (std::uint32_t move = retained; move > insert; --move)
      ranked[move] = ranked[move - 1u];
    ranked[insert] = candidate;
    if (qualified_count < plan::kMaxSteps)
      ++qualified_count;
    else
      overflow = true;
  }
  for (std::uint32_t position = 0u; position < qualified_count; ++position)
    indices[position] = ranked[position].binding_index;
  if (result != nullptr) {
    result->exact_topic_matches = exact_topic_matches;
    result->capacity_stopped = overflow ? 1u : 0u;
    if (qualified_count != 0u) {
      result->selected_binding_index = ranked[0].binding_index;
      result->selected_information_gain = ranked[0].information_gain;
      result->selected_role_coverage = ranked[0].role_coverage;
      result->selected_episode_coverage = ranked[0].episode_coverage;
    }
    // Written unconditionally: a run in which NOTHING qualified must still
    // report the best overlap the store could offer, otherwise "no candidate
    // qualified" and "no candidate overlapped" return the same zero.
    result->max_role_coverage_any_episode = max_role_coverage_any_episode;
  }
  if (qualified_count == 0u)
    return false;

  // Diagnostic only -- reports the two values the join gate below actually
  // tests, immediately before it tests them.  `receipt->qualified_best_count`
  // cannot answer this: it is sourced from `ordered.qualified_bindings` (the
  // all-qualified count), not from the co-best `qualified_count` that gates
  // the bounded two-binding join.  The join's inner loop is nested over the
  // same 65536-entry capacity as its outer loop, so whether this gate is taken
  // is the single discriminator between an O(N) and an O(N^2) single-threaded
  // scan.  `will_enter` repeats the gate expression verbatim rather than a
  // simplification of it, so the instrument cannot disagree with the code.
  // SURFACED THROUGH THE RECEIPT, NOT PRINTED. A device-side printf here goes
  // to STDOUT, and under --duplex-stdio stdout carries the length-prefixed
  // motor frames -- a print on this path injects bytes into the frame stream
  // and the harness stalls waiting for a valid header. Measured 2026-08-07: it
  // hung the gate. Device code also cannot read getenv, so such a print cannot
  // be env-gated. Every device-side observable on this path must therefore
  // travel host-ward as a field.
  if (result != nullptr) {
    result->join_gate_qualified_count = qualified_count;
    // Repeats the gate expression verbatim rather than a simplification, so
    // the instrument cannot disagree with the code it reports on.
    result->join_gate_entered =
        (topic.kind == OrderedTopicKind::exact_units && qualified_count == 1u)
            ? 1u
            : 0u;
  }

  if (topic.kind == OrderedTopicKind::exact_units && qualified_count == 1u) {
    const std::uint32_t root = indices[0];
    const std::uint64_t root_episode =
        resident_tissue.ordered_bindings[root].last_evidence_revision >> 32u;
    // Before falling back to a local chronology, compete bounded two-binding
    // joins.  The first event must belong to the cue-supported episode; the
    // second must be later, independently qualified, and share retained
    // identity with it.  This lets a plan cross learned episode matter without
    // treating temporal proximity as an answer rule.
    std::uint32_t join_first = 0xffffffffu;
    std::uint32_t join_second = 0xffffffffu;
    std::uint32_t join_cue = 0u;
    std::uint32_t join_shared = 0u;
    std::uint32_t join_novelty = 0u;
    std::int64_t join_credit = INT64_MIN;
    bool join_ambiguous = false;
    // ENV-GATED ABLATION, NOT A DELETION. `ordered_join_lesioned` is sourced
    // ONCE from BCC32_ORDERED_JOIN_LESION on the host and defaults to false, so
    // an unset environment runs exactly today's code. Measured: this join is
    // entered on every contact tick and NEVER succeeds on the two corpora we
    // have -- `join_first` stays 0xffffffffu, so the `else` branch below
    // already builds every plan actually emitted -- while its inner loop
    // re-scans the same 65536-entry capacity as its outer loop, a worst case of
    // 4.3e9 single-threaded iterations. But "never succeeded on the two corpora
    // measured" is NOT "never succeeds", so the search is retained in full and
    // merely bypassed, to price it. The lesion is placed on the SEARCH, not on
    // the enclosing gate: the fall-back chronology chain lives in that gate's
    // `else`, so lesioning the gate itself would skip the chain too. Skipping
    // the search leaves `join_first == 0xffffffffu` and `join_ambiguous ==
    // false`, which is bit-for-bit the state the unlesioned search already
    // reaches today, and control falls into the very same `else`. The lesion is
    // deliberately NOT folded into the `join_gate_entered` receipt above: that
    // field is the instrument for the code's own predicate and must keep
    // reporting what the unlesioned gate WOULD have evaluated to.
    for (std::uint32_t first_index = 0u;
         !ordered_join_lesioned &&
         first_index < resident_tissue.ordered_binding_capacity; ++first_index) {
      const auto& first = resident_tissue.ordered_bindings[first_index];
      if (first.claimed == 0u ||
          (first.last_evidence_revision >> 32u) != root_episode ||
          !tissue::ordered_binding_qualified(
              first, qualification, resident_tissue.cell_capacity) ||
          (construction_witnesses != nullptr &&
           (first_index >= construction_witness_capacity ||
            construction_witnesses[first_index] == 0xffffffffu)))
        continue;
      const std::uint32_t first_cue =
          exact_unit_cue_role_identity_overlap(topic.exact_units, first, 0u) +
          exact_unit_cue_role_identity_overlap(topic.exact_units, first, 1u) +
          exact_unit_cue_role_identity_overlap(topic.exact_units, first, 2u);
      if (first_cue == 0u) continue;
      for (std::uint32_t second_index = 0u;
           second_index < resident_tissue.ordered_binding_capacity;
           ++second_index) {
        const auto& second = resident_tissue.ordered_bindings[second_index];
        if (second.claimed == 0u ||
            (second.last_evidence_revision >> 32u) == root_episode ||
            second.last_evidence_revision <= first.last_evidence_revision ||
            !tissue::ordered_binding_qualified(
                second, qualification, resident_tissue.cell_capacity) ||
            (construction_witnesses != nullptr &&
             (second_index >= construction_witness_capacity ||
              construction_witnesses[second_index] == 0xffffffffu)))
          continue;
        // The bridge must be newly traversed resident matter, not one of the
        // cue's own identities.  Otherwise a repeated condition word can
        // short-circuit into an unrelated episode before the plan reaches its
        // learned consequence.
        std::uint32_t shared = 0u;
        for (std::uint32_t role = 0u;
             role < tissue::kOrderedBindingRoleCount; ++role) {
          for (std::uint32_t identity = 0u;
               identity < first.role_unit_counts[role]; ++identity) {
            const std::uint32_t unit = first.role_units[role][identity];
            if (exact_unit_cue_contains_identity(topic.exact_units, unit) ||
                (topic.exact_units.closed_class_mask != nullptr &&
                 topic.exact_units.closed_class_mask[unit] != 0u))
              continue;
            bool seen = false;
            for (std::uint32_t earlier_role = 0u;
                 earlier_role <= role && !seen; ++earlier_role) {
              const std::uint32_t limit = earlier_role == role
                  ? identity : first.role_unit_counts[earlier_role];
              for (std::uint32_t earlier = 0u; earlier < limit; ++earlier)
                seen = seen || first.role_units[earlier_role][earlier] == unit;
            }
            if (!seen && ordered_binding_contains_identity(second, unit))
              ++shared;
          }
        }
        if (shared == 0u) continue;
        // Query coverage identifies the source episode.  Counting cue matter
        // in the later episode would reward a distractor that repeats the
        // condition instead of the learned bridge; the later side competes on
        // shared identity and genuinely new retained matter below.
        const std::uint32_t cue = first_cue;
        const std::uint32_t novelty =
            exact_unit_cue_role_identity_novelty(topic.exact_units, second);
        const std::int64_t credit =
            plan_eligibility::ordered_binding_credit(eligibility, first_index) +
            plan_eligibility::ordered_binding_credit(eligibility, second_index);
        const bool same_score = cue == join_cue && shared == join_shared &&
            novelty == join_novelty && credit == join_credit;
        const bool better = cue > join_cue ||
            (cue == join_cue &&
             (shared > join_shared ||
              (shared == join_shared &&
               (novelty > join_novelty ||
                (novelty == join_novelty && credit > join_credit)))));
        if (join_first == 0xffffffffu || better) {
          join_first = first_index;
          join_second = second_index;
          join_cue = cue;
          join_shared = shared;
          join_novelty = novelty;
          join_credit = credit;
          join_ambiguous = false;
        } else if (same_score) {
          join_ambiguous = true;
        }
      }
    }
    if (join_first != 0xffffffffu && !join_ambiguous) {
      indices[0] = join_first;
      indices[1] = join_second;
      qualified_count = 2u;
      if (result != nullptr) {
        result->episode_spine_steps = 2u;
        result->episode_spine_terminal = join_second;
      }
    } else {
      if (result != nullptr && join_ambiguous)
        result->episode_spine_ambiguous = 1u;
      // A full learned episode is a chronology of local relation events. The
      // root is selected by cue overlap; local physical order is retained
      // before an exact rheme recurrence may cross into a later contact.
    std::uint32_t chain_count = 1u;
    while (chain_count < plan::kMaxSteps) {
      const std::uint32_t previous = indices[chain_count - 1u];
      const auto& previous_binding = resident_tissue.ordered_bindings[previous];
      std::uint32_t successor = 0xffffffffu;
      std::uint64_t successor_revision = UINT64_MAX;
      bool successor_is_local = false;
      bool ambiguous_successor = false;
      for (std::uint32_t candidate = 0u;
           candidate < resident_tissue.ordered_binding_capacity; ++candidate) {
        bool visited = false;
        for (std::uint32_t prior = 0u; prior < chain_count; ++prior)
          visited = visited || indices[prior] == candidate;
        if (visited)
          continue;
        const auto& binding = resident_tissue.ordered_bindings[candidate];
        if (binding.claimed == 0u ||
            !tissue::ordered_binding_qualified(
                binding, qualification, resident_tissue.cell_capacity) ||
            binding.last_evidence_revision <= previous_binding.last_evidence_revision)
          continue;
        const bool local_continuation =
            (binding.last_evidence_revision >> 32u) ==
            (previous_binding.last_evidence_revision >> 32u);
        if (!local_continuation &&
            !ordered_binding_rheme_links(previous_binding, binding))
          continue;
        if (successor == 0xffffffffu ||
            (local_continuation && !successor_is_local) ||
            (local_continuation == successor_is_local &&
             binding.last_evidence_revision < successor_revision)) {
          successor = candidate;
          successor_revision = binding.last_evidence_revision;
          successor_is_local = local_continuation;
          ambiguous_successor = false;
        } else if (local_continuation == successor_is_local &&
                   binding.last_evidence_revision == successor_revision) {
          ambiguous_successor = true;
        }
      }
      if (ambiguous_successor || successor == 0xffffffffu)
      {
        if (result != nullptr && ambiguous_successor)
          result->episode_spine_ambiguous = 1u;
        break;
      }
      indices[chain_count++] = successor;
    }
    qualified_count = chain_count;
    if (result != nullptr) {
      result->episode_spine_steps = chain_count;
      result->episode_spine_terminal = indices[chain_count - 1u];
    }
    }
  }

  // Deterministic device-resident ordering: physical evidence chronology is
  // the fallback; exact rheme recurrence is preferred after the first step.
  for (std::uint32_t position = 0u; position < qualified_count; ++position) {
    std::uint32_t selected = position;
    bool selected_link = false;
    for (std::uint32_t candidate = position; candidate < qualified_count;
         ++candidate) {
      const bool linked = position != 0u && ordered_binding_rheme_links(
          resident_tissue.ordered_bindings[indices[position - 1u]],
          resident_tissue.ordered_bindings[indices[candidate]]);
      if ((linked && !selected_link) ||
          (linked == selected_link && ordered_binding_precedes(
              resident_tissue.ordered_bindings[indices[candidate]],
              indices[candidate],
              resident_tissue.ordered_bindings[indices[selected]],
              indices[selected]))) {
        selected = candidate;
        selected_link = linked;
      }
    }
    const std::uint32_t swap = indices[position];
    indices[position] = indices[selected];
    indices[selected] = swap;
  }

  const plan::PlanModality modality =
      qualification == tissue::OrderedBindingQualification::causal
          ? plan::PlanModality::causal
          : plan::PlanModality::discourse;
  if (!plan::begin_population_plan(
          state, resident_tissue.scalars->revision, surface_revision, modality,
          resident_tissue.scalars->accepted_counterevidence, 1u,
          resident_tissue.scalars->lesion_revision))
    return false;
  std::uint32_t linked_dependencies = 0u;
  for (std::uint32_t step = 0u; step < qualified_count; ++step) {
    const bool linked = step != 0u && ordered_binding_rheme_links(
        resident_tissue.ordered_bindings[indices[step - 1u]],
        resident_tissue.ordered_bindings[indices[step]]);
    const std::uint32_t dependency = linked ? 1u << (step - 1u) : 0u;
    linked_dependencies += linked;
    const auto& binding = resident_tissue.ordered_bindings[indices[step]];
    if (!plan::append_population_step(
            state, indices + step, 1u, dependency,
            binding.last_evidence_revision,
            plan::PlanReferenceKind::ordered_binding)) {
      plan::clear(state);
      if (result != nullptr)
        result->capacity_stopped = 1u;
      return false;
    }
  }
  if (!ground_ordered_binding_plan(resident_tissue, units, state,
                                   surface_revision)) {
    plan::clear(state);
    return false;
  }
  if (result != nullptr) {
    result->staged = 1u;
    result->step_count = state->step_count;
    result->linked_dependencies = linked_dependencies;
  }
  return true;
}

__device__ inline bool stage_plan_from_ordered_topic(
    tissue::TissueView resident_tissue, tissue::SparsePopulationView topic,
    tissue::OrderedBindingQualification qualification,
    const population_surface::UnitPopulationView& units,
    plan::ResidentDiscoursePlanState* state, OrderedSettlementResult* result,
    std::uint32_t surface_revision,
    const plan_eligibility::ResidentPlanEligibilityState* eligibility = nullptr,
    const std::uint32_t* construction_witnesses = nullptr,
    std::uint32_t construction_witness_capacity = 0u) {
  return stage_plan_from_ordered_topic_view(
      resident_tissue,
      {OrderedTopicKind::population, topic, {}}, qualification, units, state,
      result, surface_revision, eligibility);
}

__device__ inline bool stage_plan_from_ordered_exact_units(
    tissue::TissueView resident_tissue, ExactUnitCueView exact_units,
    tissue::OrderedBindingQualification qualification,
    const population_surface::UnitPopulationView& units,
    plan::ResidentDiscoursePlanState* state, OrderedSettlementResult* result,
    std::uint32_t surface_revision,
    const plan_eligibility::ResidentPlanEligibilityState* eligibility = nullptr,
    const std::uint32_t* construction_witnesses = nullptr,
    std::uint32_t construction_witness_capacity = 0u,
    // Defaulted false == join enabled == today's behaviour.
    bool ordered_join_lesioned = false) {
  return stage_plan_from_ordered_topic_view(
      resident_tissue,
      {OrderedTopicKind::exact_units, {}, exact_units}, qualification, units,
      state, result, surface_revision, eligibility, construction_witnesses,
      construction_witness_capacity, ordered_join_lesioned);
}

__global__ void stage_plan_from_ordered_topic_kernel(
    tissue::TissueView resident_tissue, tissue::SparsePopulationView topic,
    tissue::OrderedBindingQualification qualification,
    population_surface::UnitPopulationView units,
    plan::ResidentDiscoursePlanState* state, OrderedSettlementResult* result,
    std::uint32_t surface_revision) {
  if (blockIdx.x != 0u || threadIdx.x != 0u)
    return;
  stage_plan_from_ordered_topic(resident_tissue, topic, qualification, units,
                                state, result, surface_revision);
}

__global__ void stage_plan_from_ordered_exact_units_kernel(
    tissue::TissueView resident_tissue, ExactUnitCueView exact_units,
    tissue::OrderedBindingQualification qualification,
    population_surface::UnitPopulationView units,
    plan::ResidentDiscoursePlanState* state, OrderedSettlementResult* result,
    std::uint32_t surface_revision) {
  if (blockIdx.x != 0u || threadIdx.x != 0u)
    return;
  stage_plan_from_ordered_exact_units(resident_tissue, exact_units,
                                      qualification, units, state, result,
                                      surface_revision);
}

}  // namespace bcc32_cuda_resident_proposition_chain
