#pragma once

#include <cstdint>

#include "hardware_native/bcc32_cuda_resident_discourse_plan.cuh"
#include "hardware_native/bcc32_cuda_resident_proposition_tissue.cuh"

namespace bcc32_cuda_resident_multistep_commitment {

namespace plan = bcc32_cuda_resident_discourse_plan;
namespace tissue = bcc32_cuda_resident_proposition_tissue;

struct CommitmentResult {
  std::uint32_t formed = 0u;
  std::uint32_t planned_steps = 0u;
  std::uint32_t terminal_reached = 0u;
  std::uint32_t cycle_cut = 0u;
  std::uint32_t capacity_cut = 0u;
  std::uint64_t evidence_revision = 0u;
  std::uint32_t cue_cells_matched = 0u;
  std::uint32_t qualified_synapses = 0u;
  std::uint64_t strongest_score = 0u;
  std::uint64_t uncertain_mass = 0u;
};

__device__ inline plan::PlanModality plan_modality(
    tissue::CompletionPolicy policy) {
  return policy == tissue::CompletionPolicy::causal
             ? plan::PlanModality::causal
             : plan::PlanModality::discourse;
}

__device__ inline bool same_population(tissue::SparsePopulationView left,
                                       const std::uint32_t* right_cells,
                                       std::uint32_t right_count) {
  if (left.count != right_count)
    return false;
  for (std::uint32_t index = 0u; index < left.count; ++index)
    if (!tissue::population_contains(tissue::SparsePopulationView{right_cells, right_count},
                                     left.cells[index]))
      return false;
  for (std::uint32_t index = 0u; index < right_count; ++index)
    if (!tissue::population_contains(left, right_cells[index]))
      return false;
  return true;
}

__device__ inline bool repeats_committed_population(const plan::ResidentDiscoursePlanState& state,
                                                    tissue::SparsePopulationView initial_cue,
                                                    tissue::SparsePopulationView candidate) {
  if (same_population(candidate, initial_cue.cells, initial_cue.count))
    return true;
  for (std::uint32_t step_index = 0u; step_index < state.step_count; ++step_index) {
    const plan::PlanStep& step = state.steps[step_index];
    if (same_population(candidate, state.population_references + step.population_begin,
                        step.population_count))
      return true;
  }
  return false;
}

// Form the entire bounded proposition trajectory before exposing any surface
// anchor. The caller chooses the tissue's readout policy; causal fixtures keep
// the intervention gate, while production discourse may admit independently
// contextualized observation support without relabeling it causal.
__device__ inline void form_multistep_plan(
    tissue::TissueView resident_tissue,
    tissue::SparsePopulationView initial_cue,
    tissue::SparsePopulationView active_context,
    tissue::CompletionWorkspaceView workspace,
    plan::ResidentDiscoursePlanState* state, std::uint32_t surface_revision,
    std::uint32_t maximum_steps, CommitmentResult* result,
    tissue::CompletionPolicy policy = tissue::CompletionPolicy::causal,
    std::uint32_t preserve_empty_settling = 0u) {
  if (result == nullptr)
    return;
  *result = CommitmentResult{};
  if (state == nullptr || resident_tissue.scalars == nullptr ||
      !tissue::valid_population(initial_cue, resident_tissue.cell_capacity) ||
      maximum_steps == 0u || maximum_steps > plan::kMaxSteps)
    return;
  if (active_context.count != 0u &&
      !tissue::valid_population(active_context, resident_tissue.cell_capacity))
    return;

  const std::uint64_t evidence_revision = resident_tissue.scalars->revision;
  result->evidence_revision = evidence_revision;
  if (!plan::begin_population_plan(state, evidence_revision, surface_revision,
                                   plan_modality(policy),
                                   resident_tissue.scalars->accepted_counterevidence,
                                   1u,
                                   resident_tissue.scalars->lesion_revision))
    return;

  tissue::SparsePopulationView cue = initial_cue;
  for (std::uint32_t step_index = 0u; step_index < maximum_steps; ++step_index) {
    tissue::CompletionResult completion{};
    tissue::settle_completion_device(resident_tissue, cue, active_context, 0u,
                                     workspace, &completion, policy);
    if (completion.cue_cells_matched > result->cue_cells_matched)
      result->cue_cells_matched = completion.cue_cells_matched;
    result->qualified_synapses += completion.qualified_synapses;
    if (completion.strongest_score > result->strongest_score)
      result->strongest_score = completion.strongest_score;
    result->uncertain_mass += completion.uncertain_mass;
    if (completion.ready == 0u) {
      result->terminal_reached = 1u;
      break;
    }
    const tissue::SparsePopulationView completed{workspace.output_cells, completion.output_count};
    if (repeats_committed_population(*state, initial_cue, completed)) {
      result->cycle_cut = 1u;
      break;
    }
    const std::uint32_t dependency_mask = step_index == 0u ? 0u : (1u << (step_index - 1u));
    if (!plan::append_population_step(state, completed.cells, completed.count,
                                      dependency_mask, evidence_revision)) {
      result->capacity_cut = 1u;
      break;
    }
    cue = completed;
  }

  result->planned_steps = state->step_count;
  result->formed = state->step_count != 0u ? 1u : 0u;
  if (state->step_count == maximum_steps && result->terminal_reached == 0u &&
      result->cycle_cut == 0u)
    result->capacity_cut = 1u;
  if (state->step_count == 0u && preserve_empty_settling == 0u)
    plan::clear(state);
}

__global__ void form_multistep_plan_kernel(
    tissue::TissueView resident_tissue,
    tissue::SparsePopulationView initial_cue,
    tissue::SparsePopulationView active_context,
    tissue::CompletionWorkspaceView workspace,
    plan::ResidentDiscoursePlanState* state, std::uint32_t surface_revision,
    std::uint32_t maximum_steps, CommitmentResult* result) {
  if (blockIdx.x != 0u || threadIdx.x != 0u)
    return;
  form_multistep_plan(resident_tissue, initial_cue, active_context, workspace,
                      state, surface_revision, maximum_steps, result);
}

}  // namespace bcc32_cuda_resident_multistep_commitment
