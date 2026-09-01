#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <limits>

#include "bcc32_cuda_resident_discourse_plan.cuh"
#include "bcc32_cuda_resident_question_goal.cuh"

namespace bcc32_cuda_resident_population_surface {

namespace plan = bcc32_cuda_resident_discourse_plan;
namespace question_goal = bcc32_cuda_resident_question_goal;

constexpr std::uint32_t kInvalidUnit = std::numeric_limits<std::uint32_t>::max();
constexpr std::uint32_t kMaximumUnitPopulationWidth = 16u;
constexpr std::uint32_t kMaximumPopulationAnchorEdges = 256u;
constexpr std::uint32_t kMaximumDependencyEdges = (plan::kMaxSteps * (plan::kMaxSteps - 1u)) / 2u;

// Surface units are body-side motor chunks. Their sparse populations are
// learned contact addresses, not concept identifiers. This view lets an
// already-formed resident commitment ground into that body without a host
// reading the population or naming the word to emit.
struct UnitPopulationView {
  const std::uint32_t* cells = nullptr;
  const std::uint64_t* contact_mass = nullptr;
  std::uint32_t unit_begin = 0u;
  std::uint32_t unit_count = 0u;
  std::uint32_t population_width = 0u;
  const std::uint32_t* population_count = nullptr;
  // Learned coactivity support for each unit-population slot. Slot zero is
  // unused; contextual support lives in a separate mutable population so
  // learning cannot rewrite the identity cells used by older bindings.
  const std::uint16_t* population_context_mass = nullptr;
  const std::uint32_t* context_cells = nullptr;
};

struct GroundingResult {
  std::uint32_t ready = 0u;
  std::uint32_t grounded_steps = 0u;
  std::uint32_t anchor_count = 0u;
  std::uint32_t ambiguous_step = kInvalidUnit;
  std::uint32_t ungrounded_step = kInvalidUnit;
  std::uint32_t weakest_overlap = 0u;
  std::uint32_t plan_revision = 0u;
};

struct PopulationAnchorEdge {
  std::uint32_t step = 0u;
  std::uint32_t population_reference = 0u;
  std::uint32_t anchor_reference = 0u;
  std::uint32_t reserved = 0u;
};

struct StepDependencyEdge {
  std::uint32_t prerequisite_step = 0u;
  std::uint32_t dependent_step = 0u;
};

// Observer-readable receipt for the lossless bounded grounding transaction.
// The plan remains the causal object; these arrays expose every direct
// population-to-anchor support edge and every declared step dependency without
// collapsing a step to a single winning unit.
struct DependencyGroundingResult {
  std::uint32_t ready = 0u;
  std::uint32_t grounded_steps = 0u;
  std::uint32_t population_anchor_edge_count = 0u;
  std::uint32_t dependency_edge_count = 0u;
  std::uint32_t anchor_count = 0u;
  std::uint32_t ungrounded_step = kInvalidUnit;
  std::uint32_t ungrounded_population = kInvalidUnit;
  std::uint32_t invalid_dependency_step = kInvalidUnit;
  std::uint32_t overflow = 0u;
  std::uint32_t plan_revision = 0u;
  std::uint32_t step_anchor_begin[plan::kMaxSteps]{};
  std::uint32_t step_anchor_count[plan::kMaxSteps]{};
  PopulationAnchorEdge population_anchor_edges[kMaximumPopulationAnchorEdges]{};
  StepDependencyEdge dependency_edges[kMaximumDependencyEdges]{};
};

__device__ inline std::uint32_t population_overlap(const plan::ResidentDiscoursePlanState& state,
                                                   const plan::PlanStep& step,
                                                   const std::uint32_t* unit_cells,
                                                   std::uint32_t unit_width) {
  std::uint32_t overlap = 0u;
  for (std::uint32_t population_index = 0u; population_index < step.population_count;
       ++population_index) {
    const std::uint32_t cell =
        state.population_references[step.population_begin + population_index];
    for (std::uint32_t slot = 0u; slot < unit_width; ++slot) {
      if (unit_cells[slot] != cell)
        continue;
      ++overlap;
      break;
    }
  }
  return overlap;
}

__device__ inline bool sufficient_overlap(std::uint32_t overlap, std::uint32_t plan_width,
                                          std::uint32_t unit_width) {
  const std::uint32_t comparable = plan_width < unit_width ? plan_width : unit_width;
  return overlap >= 2u && overlap * 2u >= comparable;
}

// Materialization is atomic at the plan level. A proposition step may be the
// union of several learned body populations. Grounding therefore constructs a
// bounded cover, repeatedly taking the resident unit that explains the most
// still-uncovered proposition cells. Contact mass resolves unequal evidence;
// equal overlap and mass abstain, so numeric unit id never breaks a semantic
// tie. A mostly covered lesion may retain the established bounded-overlap
// fallback. A single thread is intentional for the bounded (<=8 step)
// commitment; selection remains entirely device resident.
__global__ void materialize_plan_anchors_kernel(plan::ResidentDiscoursePlanState* state,
                                                UnitPopulationView units,
                                                const std::uint32_t* evidence_cells,
                                                const std::uint64_t* evidence_scores,
                                                std::uint32_t evidence_capacity,
                                                std::uint32_t surface_revision,
                                                GroundingResult* result) {
  if (blockIdx.x != 0u || threadIdx.x != 0u || result == nullptr)
    return;
  *result = GroundingResult{};
  if (state == nullptr || !plan::valid(*state) ||
      (state->status != plan::PlanStatus::settling &&
       state->status != plan::PlanStatus::committed) ||
      state->step_count == 0u || units.cells == nullptr || units.contact_mass == nullptr ||
      units.unit_count == 0u || units.population_width == 0u ||
      units.population_width > kMaximumUnitPopulationWidth ||
      units.unit_begin > kInvalidUnit - units.unit_count)
    return;

  std::uint32_t selected[plan::kMaxAnchorReferences]{};
  std::uint32_t step_anchor_begin[plan::kMaxSteps]{};
  std::uint32_t step_anchor_count[plan::kMaxSteps]{};
  std::uint32_t selected_count = 0u;
  std::uint32_t weakest_overlap = kInvalidUnit;
  for (std::uint32_t step_index = 0u; step_index < state->step_count; ++step_index) {
    const plan::PlanStep& step = state->steps[step_index];
    if (step.reference_kind != plan::PlanReferenceKind::opaque_population ||
        step.population_count == 0u) {
      result->ungrounded_step = step_index;
      return;
    }

    step_anchor_begin[step_index] = selected_count;
    bool covered[plan::kMaxPopulationReferences]{};
    std::uint32_t covered_count = 0u;
    while (covered_count < step.population_count) {
      std::uint32_t best_unit = kInvalidUnit;
      std::uint64_t best_evidence = 0u;
      std::uint32_t best_overlap = 0u;
      std::uint64_t best_mass = 0u;
      bool ambiguous = false;
      for (std::uint32_t offset = 0u; offset < units.unit_count; ++offset) {
        const std::uint32_t unit = units.unit_begin + offset;
        if (units.contact_mass[unit] == 0u)
          continue;
        bool already_selected = false;
        for (std::uint32_t index = step_anchor_begin[step_index];
             index < selected_count; ++index)
          already_selected = already_selected || selected[index] == unit;
        if (already_selected)
          continue;
        const std::uint32_t* unit_cells =
            units.cells + static_cast<std::size_t>(unit) * units.population_width;
        std::uint32_t overlap = 0u;
        std::uint64_t evidence = 0u;
        for (std::uint32_t population = 0u; population < step.population_count;
             ++population) {
          if (covered[population])
            continue;
          const std::uint32_t target =
              state->population_references[step.population_begin + population];
          bool present = false;
          for (std::uint32_t slot = 0u; slot < units.population_width; ++slot)
            present = present || unit_cells[slot] == target;
          overlap += present;
          if (present && evidence_cells != nullptr && evidence_scores != nullptr) {
            for (std::uint32_t index = 0u; index < evidence_capacity; ++index)
              if (evidence_cells[index] == target)
                evidence += evidence_scores[index];
          }
        }
        const std::uint64_t mass = units.contact_mass[unit];
        if (evidence > best_evidence ||
            (evidence == best_evidence && overlap > best_overlap) ||
            (evidence == best_evidence && overlap == best_overlap &&
             overlap != 0u && mass > best_mass)) {
          best_unit = unit;
          best_evidence = evidence;
          best_overlap = overlap;
          best_mass = mass;
          ambiguous = false;
        } else if (overlap != 0u && evidence == best_evidence &&
                   overlap == best_overlap && mass == best_mass) {
          ambiguous = true;
        }
      }
      if (best_unit == kInvalidUnit) {
        if (selected_count != step_anchor_begin[step_index] &&
            sufficient_overlap(covered_count, step.population_count,
                               units.population_width))
          break;
        result->ungrounded_step = step_index;
        return;
      }
      if (ambiguous) {
        result->ambiguous_step = step_index;
        return;
      }
      if (selected_count == plan::kMaxAnchorReferences) {
        result->ungrounded_step = step_index;
        return;
      }
      selected[selected_count++] = best_unit;
      const std::uint32_t* unit_cells =
          units.cells + static_cast<std::size_t>(best_unit) * units.population_width;
      for (std::uint32_t population = 0u; population < step.population_count;
           ++population) {
        if (covered[population])
          continue;
        const std::uint32_t target =
            state->population_references[step.population_begin + population];
        for (std::uint32_t slot = 0u; slot < units.population_width; ++slot)
          if (unit_cells[slot] == target) {
            covered[population] = true;
            ++covered_count;
            break;
          }
      }
      if (weakest_overlap == kInvalidUnit || best_overlap < weakest_overlap)
        weakest_overlap = best_overlap;
    }
    step_anchor_count[step_index] = selected_count - step_anchor_begin[step_index];
  }

  state->anchor_reference_count = 0u;
  for (std::uint32_t step_index = 0u; step_index < state->step_count; ++step_index) {
    plan::PlanStep& step = state->steps[step_index];
    step.anchor_begin = state->anchor_reference_count;
    step.anchor_count = step_anchor_count[step_index];
    for (std::uint32_t index = 0u; index < step_anchor_count[step_index]; ++index)
      state->anchor_references[state->anchor_reference_count++] =
          selected[step_anchor_begin[step_index] + index];
  }
  state->surface_revision = surface_revision;
  if (state->status == plan::PlanStatus::settling) {
    state->active_step = 0u;
    state->status = plan::PlanStatus::committed;
  }
  ++state->revision;

  result->ready = 1u;
  result->grounded_steps = state->step_count;
  result->anchor_count = state->anchor_reference_count;
  result->weakest_overlap = weakest_overlap == kInvalidUnit ? 0u : weakest_overlap;
  result->plan_revision = state->revision;
}

// Ordered proposition settlement already commits the complete retained unit
// identities for each binding.  Do not run that identity through the opaque
// population cover again: the cover is intentionally lossy and is only for
// plans whose references are opaque populations.  This is a structural plan
// invariant, not a lexical or semantic admission rule.
__global__ void finalize_ordered_plan_grounding_kernel(
    const plan::ResidentDiscoursePlanState* state, GroundingResult* result) {
  if (blockIdx.x != 0u || threadIdx.x != 0u || state == nullptr ||
      result == nullptr || !plan::valid(*state) ||
      state->status != plan::PlanStatus::committed || state->step_count == 0u ||
      state->anchor_reference_count == 0u)
    return;
  for (std::uint32_t step = 0u; step < state->step_count; ++step) {
    if (state->steps[step].reference_kind != plan::PlanReferenceKind::ordered_binding ||
        state->steps[step].anchor_count == 0u)
      return;
  }
  *result = GroundingResult{};
  result->ready = 1u;
  result->grounded_steps = state->step_count;
  result->anchor_count = state->anchor_reference_count;
  result->plan_revision = state->revision;
}

// A QuestionGoal Plan stores the persistent goal slot, not a copied topic
// population. Grounding dereferences that slot on device and selects one
// learned surface unit whose sparse population is exactly the goal target.
// Contact mass is surface evidence, so it may resolve unequal alternatives;
// equal best alternatives abstain. No Plan byte changes until the complete
// transaction has one supported filler.
__global__ void materialize_question_goal_anchor_kernel(
    plan::ResidentDiscoursePlanState* state,
    question_goal::ResidentQuestionGoalState* goal,
    UnitPopulationView units, std::uint32_t cell_capacity,
    std::uint32_t surface_revision,
    GroundingResult* result) {
  if (blockIdx.x != 0u || threadIdx.x != 0u || result == nullptr)
    return;
  *result = GroundingResult{};
  if (state == nullptr || goal == nullptr ||
      cell_capacity == 0u || !question_goal::valid(*goal, cell_capacity) ||
      !plan::valid(*state) || state->status != plan::PlanStatus::committed ||
      !question_goal::question_plan_matches_goal(*goal, *state) ||
      units.cells == nullptr || units.contact_mass == nullptr ||
      units.unit_count == 0u || units.population_width == 0u ||
      units.population_width > kMaximumUnitPopulationWidth ||
      units.unit_begin > kInvalidUnit - units.unit_count)
    return;

  std::uint32_t best_unit = kInvalidUnit;
  std::uint64_t best_mass = 0u;
  bool ambiguous = false;
  for (std::uint32_t offset = 0u; offset < units.unit_count; ++offset) {
    const std::uint32_t unit = units.unit_begin + offset;
    const std::uint32_t count = units.population_count == nullptr
                                    ? units.population_width
                                    : units.population_count[unit];
    if (units.contact_mass[unit] == 0u || count != goal->target_count ||
        count > units.population_width)
      continue;
    const std::uint32_t* cells =
        units.cells + static_cast<std::size_t>(unit) * units.population_width;
    bool exact = true;
    for (std::uint32_t target = 0u; target < goal->target_count; ++target) {
      bool present = false;
      for (std::uint32_t cell = 0u; cell < count; ++cell)
        present = present || cells[cell] == goal->target_cells[target];
      exact = exact && present;
    }
    if (!exact)
      continue;
    const std::uint64_t mass = units.contact_mass[unit];
    if (best_unit == kInvalidUnit || mass > best_mass) {
      best_unit = unit;
      best_mass = mass;
      ambiguous = false;
    } else if (mass == best_mass) {
      ambiguous = true;
    }
  }
  if (best_unit == kInvalidUnit) {
    result->ungrounded_step = 0u;
    result->plan_revision = state->revision;
    return;
  }
  if (ambiguous) {
    result->ambiguous_step = 0u;
    result->plan_revision = state->revision;
    return;
  }

  state->anchor_reference_count = 1u;
  state->anchor_references[0] = best_unit;
  state->steps[0].anchor_begin = 0u;
  state->steps[0].anchor_count = 1u;
  state->surface_revision = surface_revision;
  ++state->revision;
  goal->plan_revision = state->revision;
  result->ready = 1u;
  result->grounded_steps = 1u;
  result->anchor_count = 1u;
  result->weakest_overlap = goal->target_count;
  result->plan_revision = state->revision;
}

// Lossless grounding keeps all resident support edges.  It uses no overlap
// threshold, winning anchor, authored role, or host-selected tie rule.  The
// transaction mutates plan anchors only after every population and dependency
// fits the declared bounded receipt.
__global__ void materialize_dependency_grounding_kernel(plan::ResidentDiscoursePlanState* state,
                                                        UnitPopulationView units,
                                                        std::uint32_t surface_revision,
                                                        DependencyGroundingResult* result) {
  if (blockIdx.x != 0u || threadIdx.x != 0u || result == nullptr)
    return;
  *result = DependencyGroundingResult{};
  // Full Plan validation rejects an out-of-plan dependency before the
  // grounding loop can name the offending step. Preserve atomic rejection,
  // but publish the bounded observer receipt rather than silently returning a
  // blank result. No anchor or Plan byte is changed on this path.
  if (state != nullptr && state->magic == plan::kPlanMagic &&
      state->version == plan::kPlanVersion &&
      state->step_count <= plan::kMaxSteps) {
    for (std::uint32_t step_index = 0u; step_index < state->step_count;
         ++step_index) {
      const std::uint32_t lawful_prior_mask =
          step_index == 0u ? 0u : ((1u << step_index) - 1u);
      if ((state->steps[step_index].dependency_mask & ~lawful_prior_mask) !=
          0u) {
        result->invalid_dependency_step = step_index;
        result->plan_revision = state->revision;
        return;
      }
    }
  }
  if (state == nullptr || !plan::valid(*state) ||
      (state->status != plan::PlanStatus::settling &&
       state->status != plan::PlanStatus::committed) ||
      state->step_count == 0u || units.cells == nullptr || units.contact_mass == nullptr ||
      units.population_count == nullptr || units.unit_count == 0u || units.population_width == 0u ||
      units.population_width > kMaximumUnitPopulationWidth ||
      units.unit_begin > kInvalidUnit - units.unit_count)
    return;

  std::uint32_t anchors[plan::kMaxAnchorReferences]{};
  std::uint32_t anchor_count = 0u;
  for (std::uint32_t step_index = 0u; step_index < state->step_count; ++step_index) {
    const plan::PlanStep& step = state->steps[step_index];
    result->step_anchor_begin[step_index] = anchor_count;
    if (step.reference_kind != plan::PlanReferenceKind::opaque_population ||
        step.population_count == 0u) {
      result->ungrounded_step = step_index;
      return;
    }
    bool step_supported = false;
    for (std::uint32_t unit_offset = 0u; unit_offset < units.unit_count; ++unit_offset) {
      const std::uint32_t unit = units.unit_begin + unit_offset;
      if (units.contact_mass[unit] == 0u || units.population_count[unit] == 0u ||
          units.population_count[unit] > units.population_width)
        continue;

      bool contains_complete_core = true;
      for (std::uint32_t population_offset = 0u; population_offset < step.population_count;
           ++population_offset) {
        const std::uint32_t population =
            state->population_references[step.population_begin + population_offset];
        bool present = false;
        for (std::uint32_t cell = 0u; cell < units.population_count[unit]; ++cell)
          present = present || units.cells[unit * units.population_width + cell] == population;
        contains_complete_core = contains_complete_core && present;
      }
      if (!contains_complete_core)
        continue;

      step_supported = true;
      for (std::uint32_t population_offset = 0u; population_offset < step.population_count;
           ++population_offset) {
        const std::uint32_t population =
            state->population_references[step.population_begin + population_offset];
        if (result->population_anchor_edge_count == kMaximumPopulationAnchorEdges) {
          result->overflow = 1u;
          return;
        }
        result->population_anchor_edges[result->population_anchor_edge_count++] =
            PopulationAnchorEdge{step_index, population, unit, 0u};

      }
      if (anchor_count == plan::kMaxAnchorReferences) {
        result->overflow = 1u;
        return;
      }
      anchors[anchor_count++] = unit;
    }
    if (!step_supported) {
      result->ungrounded_step = step_index;
      result->ungrounded_population = step.population_begin;
      return;
    }
    result->step_anchor_count[step_index] = anchor_count - result->step_anchor_begin[step_index];

    const std::uint32_t declared_step_mask = (1u << state->step_count) - 1u;
    if ((step.dependency_mask & ~declared_step_mask) != 0u) {
      result->invalid_dependency_step = step_index;
      return;
    }
    for (std::uint32_t prerequisite = 0u; prerequisite < state->step_count; ++prerequisite) {
      if ((step.dependency_mask & (1u << prerequisite)) == 0u)
        continue;
      if (prerequisite >= step_index) {
        result->invalid_dependency_step = step_index;
        return;
      }
      if (result->dependency_edge_count == kMaximumDependencyEdges) {
        result->overflow = 1u;
        return;
      }
      result->dependency_edges[result->dependency_edge_count++] =
          StepDependencyEdge{prerequisite, step_index};
    }
  }

  state->anchor_reference_count = anchor_count;
  for (std::uint32_t index = 0u; index < anchor_count; ++index)
    state->anchor_references[index] = anchors[index];
  for (std::uint32_t step_index = 0u; step_index < state->step_count; ++step_index) {
    state->steps[step_index].anchor_begin = result->step_anchor_begin[step_index];
    state->steps[step_index].anchor_count = result->step_anchor_count[step_index];
  }
  state->surface_revision = surface_revision;
  if (state->status == plan::PlanStatus::settling) {
    state->active_step = 0u;
    state->status = plan::PlanStatus::committed;
  }
  ++state->revision;

  result->ready = 1u;
  result->grounded_steps = state->step_count;
  result->anchor_count = anchor_count;
  result->plan_revision = state->revision;
}

}  // namespace bcc32_cuda_resident_population_surface
