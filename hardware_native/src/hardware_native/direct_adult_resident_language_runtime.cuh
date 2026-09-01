#ifndef HARDWARE_NATIVE_DIRECT_ADULT_RESIDENT_LANGUAGE_RUNTIME_CUH
#define HARDWARE_NATIVE_DIRECT_ADULT_RESIDENT_LANGUAGE_RUNTIME_CUH

#include <cstdint>
#include <type_traits>

#include "hardware_native/direct_adult_resident_language_executor.cuh"
#include "hardware_native/direct_adult_language_recipe_opportunity_bank.cuh"
#include "hardware_native/direct_adult_language_span_recipe.cuh"
#include "hardware_native/direct_adult_language_one_shot_abi.cuh"

namespace substrate::direct_network {

struct DirectResidentLanguageRuntimeBlock {
  DirectResidentLanguageState language;
  DirectResidentLanguageExecutorState executor;
  DirectResidentLanguageTickReceipt last_tick_receipt;
  DirectLanguageOneShotState one_shot;
  direct_adult_core::ResidentLanguageRecipeOpportunityBankV1 opportunity_bank;
  direct_adult_core::ResidentLanguageSpanBankV1 span_bank;
  std::uint64_t assimilated_history_identity;
  std::uint64_t last_heard_surface_identity;
  std::uint64_t last_heard_surface_content_identity;
  std::uint64_t pending_spoken_surface_content_identity;
  std::uint32_t pending_spoken_surface_length;
  std::uint32_t one_shot_plan_begin;
  std::uint32_t confirmed_emitted_steps;
  std::uint32_t enabled;
  // One-shot discourse uncertainty from a verified action-outcome prediction
  // error. It changes the next planning decision, then is consumed; it is not
  // valence, reward, or semantic authority.
  std::uint32_t pending_prediction_error_bits;
  std::uint32_t prediction_error_events;
};

inline constexpr std::size_t kDirectResidentLanguageRuntimeBlockBytes =
    sizeof(DirectResidentLanguageRuntimeBlock);

static_assert(std::is_trivially_copyable_v<DirectResidentLanguageRuntimeBlock>);
static_assert(std::is_standard_layout_v<DirectResidentLanguageRuntimeBlock>);
static_assert(alignof(DirectResidentLanguageRuntimeBlock) <= alignof(std::uint64_t));

__device__ inline void resident_language_runtime_assimilate(
    DirectResidentLanguageRuntimeBlock* block,
    const DirectExactHistoryRecord* records, std::uint32_t count) {
  if (block == nullptr || records == nullptr) return;
  const bool compacted = resident_language_follow_hot_suffix(
                             &block->language, records, count) ||
                         count < block->executor.surface.cursor;
  std::uint32_t begin = block->language.temporal.cursor;
  if (block->language.expression.cursor < begin)
    begin = block->language.expression.cursor;
  if (block->executor.surface.cursor < begin)
    begin = block->executor.surface.cursor;
  if (compacted) {
    block->language.temporal.cursor = 0u;
    block->language.expression.cursor = 0u;
    block->language.expression.embodied.cursor = 0u;
    block->executor.surface.cursor = 0u;
    if (count < block->one_shot.cursor) block->one_shot.cursor = 0u;
    begin = 0u;
  }
  const std::uint32_t one_shot_prior_cursor = block->one_shot.cursor;
  if (block->one_shot_plan_begin >= one_shot_prior_cursor)
    block->one_shot_plan_begin = one_shot_prior_cursor;
  if (count > block->one_shot.cursor)
    language_one_shot_assimilate(&block->one_shot, records, count);
  block->assimilated_history_identity = resident_language_contact_identity(
      records, begin, count, block->assimilated_history_identity);
  direct_adult_core::assimilate_resident_language_span_credit(
      &block->span_bank, records, count);
  resident_language_assimilate(&block->language, records, count);
  if (count < block->executor.surface.cursor)
    block->executor.surface.cursor = 0u;
  const std::uint32_t closed_before = block->executor.surface.ecology.closed_count;
  surface_sequence_assimilate(&block->executor.surface, records, count);
  if (block->executor.surface.ecology.closed_count != closed_before) {
    block->last_heard_surface_identity =
        block->executor.surface.ecology.last_closed_payload_identity;
    block->last_heard_surface_content_identity =
        block->executor.surface.ecology.last_closed_content_identity;
  }
  block->enabled = 1u;
}

__device__ inline DirectResidentLanguageTickReceipt
resident_language_runtime_plan_only(
    DirectResidentLanguageRuntimeBlock* block,
    const DirectExactHistoryRecord* records, std::uint32_t count,
    std::uint32_t current_tick) {
  DirectResidentLanguageTickReceipt receipt{};
  if (block == nullptr || records == nullptr) return receipt;
  receipt = resident_language_plan_only_tick(
      &block->language, &block->executor, records, count, current_tick,
      direct_discourse_focus_unresolved(block->language.discourse_focus) ||
          block->pending_prediction_error_bits != 0u);
  if (receipt.plan_started != 0u) block->pending_prediction_error_bits = 0u;
  block->last_tick_receipt = receipt;
  return receipt;
}

__device__ inline DirectResidentLanguageTickReceipt
resident_language_runtime_tick_with_planning(
    DirectResidentLanguageRuntimeBlock* block,
    const DirectExactHistoryRecord* records, std::uint32_t count,
    const DirectExactHistoryRecord* planning_records,
    std::uint32_t planning_count, std::uint32_t current_tick,
    DirectNode* nodes, std::uint32_t node_count) {
  DirectResidentLanguageTickReceipt receipt{};
  if (block == nullptr || records == nullptr || planning_records == nullptr ||
      nodes == nullptr)
    return receipt;
  receipt = resident_language_tick(
      &block->language, &block->executor, planning_records, planning_count,
      current_tick,
      direct_discourse_focus_unresolved(block->language.discourse_focus) ||
          block->pending_prediction_error_bits != 0u,
      block->confirmed_emitted_steps, nodes, node_count);
  if (receipt.plan_started != 0u)
    block->pending_prediction_error_bits = 0u;
  if (block->one_shot_plan_begin < count) {
    const DirectLanguageOneShotPlan one_shot_plan = language_one_shot_plan(
        &block->one_shot, records, block->one_shot_plan_begin, count);
    if (one_shot_plan.admitted != 0u) {
      (void)language_one_shot_drive(one_shot_plan, nodes, node_count);
      block->one_shot_plan_begin = count;
    } else {
      std::uint32_t first_tick = 0u;
      for (std::uint32_t i = block->one_shot_plan_begin; i < count; ++i)
        if (records[i].kind == DirectExactHistoryKind::sensory_contact) {
          first_tick = records[i].resident_tick;
          break;
        }
      if (first_tick == 0u ||
          current_tick > first_tick + kLanguageOneShotMaximumGap)
        block->one_shot_plan_begin = count;
    }
  }
  block->last_tick_receipt = receipt;
  return receipt;
}

__device__ inline DirectResidentLanguageTickReceipt resident_language_runtime_tick(
    DirectResidentLanguageRuntimeBlock* block,
    const DirectExactHistoryRecord* records, std::uint32_t count,
    std::uint32_t current_tick, DirectNode* nodes, std::uint32_t node_count) {
  return resident_language_runtime_tick_with_planning(
      block, records, count, records, count, current_tick, nodes, node_count);
}

__device__ inline DirectResidentLanguageTickReceipt resident_language_runtime_tick(
    DirectResidentLanguageRuntimeBlock* block,
    const DirectExactHistoryHotPage* history,
    const DirectRecentSensoryState* recent_sensory,
    std::uint32_t current_tick, DirectNode* nodes, std::uint32_t node_count) {
  DirectResidentLanguageTickReceipt receipt{};
  if (block == nullptr || history == nullptr || recent_sensory == nullptr ||
      nodes == nullptr)
    return receipt;
  DirectExactHistoryRecord recent[kDirectRecentSensoryCapacity]{};
  const std::uint32_t recent_count = exact_history_recent_sensory_records(
      *recent_sensory, current_tick, kGroundingCoherenceWindowTicks, recent);
  if (recent_count == 0u) {
    // A started public trajectory is resident computation.  It must not require
    // another sensory contact merely to finish speaking; emission confirmation,
    // not fresh ingress, governs its remaining steps.
    if (block->executor.active != 0u && history->committed_slots != 0u)
      return resident_language_runtime_tick(
          block, history->records, history->committed_slots, current_tick,
          nodes, node_count);
    block->last_tick_receipt = receipt;
    return receipt;
  }
  const bool hot_contains_current_event =
      history->committed_slots != 0u &&
      history->records[0].sequence != 0u &&
      recent[0].sequence >= history->records[0].sequence;
  if (hot_contains_current_event)
    return resident_language_runtime_tick(
        block, history->records, history->committed_slots, current_tick,
        nodes, node_count);
  return resident_language_runtime_tick_with_planning(
      block, history->records, history->committed_slots, recent, recent_count,
      current_tick, nodes, node_count);
}

DIRECT_ADULT_HD inline std::uint64_t resident_language_plan_content_identity(
    const DirectLanguageMotorPlan& plan) {
  if (plan.admitted == 0u || plan.step_count == 0u ||
      plan.step_count > kLanguageExpressionPlanStepCapacity)
    return 0u;
  std::uint32_t bytes[4u * kLanguageExpressionPlanStepCapacity]{};
  std::uint32_t count = 0u;
  for (std::uint32_t i = 0u; i < plan.step_count; ++i) {
    const std::uint32_t word = plan.steps[i].word;
    bytes[count++] = word & 0xffu;
    bytes[count++] = (word >> 8u) & 0xffu;
    bytes[count++] = (word >> 16u) & 0xffu;
    bytes[count++] = (word >> 24u) & 0xffu;
  }
  return surface_ecology_content_identity(count, bytes);
}

DIRECT_ADULT_HD inline void resident_language_begin_emission_plan(
    DirectResidentLanguageRuntimeBlock* language) {
  if (language == nullptr || language->last_tick_receipt.plan_started == 0u) return;
  language->confirmed_emitted_steps = 0u;
  language->pending_spoken_surface_content_identity =
      resident_language_plan_content_identity(language->executor.pending);
  language->pending_spoken_surface_length =
      language->executor.pending.step_count * 4u;
}

DIRECT_ADULT_HD inline bool resident_language_record_recipe_opportunity(
    DirectResidentLanguageRuntimeBlock* block, std::uint64_t surface_identity,
    const direct_adult_core::ResidentRecipeOccurrence& expressed,
    std::uint16_t surface_length = 0u) {
  return block != nullptr &&
         direct_adult_core::record_resident_language_recipe_opportunity(
             &block->opportunity_bank, surface_identity, expressed, surface_length);
}

template <typename RecipeCellT, typename DerivationT>
DIRECT_ADULT_HD inline bool resident_language_nominate_heard_recipe(
    const DirectResidentLanguageRuntimeBlock& block, const RecipeCellT* cells,
    std::uint32_t cell_count, const DerivationT* derivations,
    std::uint32_t derivation_count, std::uint64_t candidate_occurrence_identity,
    direct_adult_core::ResidentRecipeOccurrence* out) {
  if (direct_adult_core::nominate_resident_language_recipe_from_bank(
          block.opportunity_bank, block.last_heard_surface_content_identity, cells,
          cell_count, derivations, derivation_count, candidate_occurrence_identity, out))
    return true;
  const auto& ecology = block.executor.surface.ecology;
  return direct_adult_core::nominate_resident_language_recipe_from_contained_surface(
      block.opportunity_bank, ecology.last_closed_values, ecology.last_closed_length,
      cells, cell_count, derivations, derivation_count,
      candidate_occurrence_identity, out);
}

}  // namespace substrate::direct_network

#endif
