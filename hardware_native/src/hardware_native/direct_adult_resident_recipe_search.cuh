#ifndef HARDWARE_NATIVE_DIRECT_ADULT_RESIDENT_RECIPE_SEARCH_CUH
#define HARDWARE_NATIVE_DIRECT_ADULT_RESIDENT_RECIPE_SEARCH_CUH

// c.resident_recipe_search (#1651): bounded, residently originated candidate
// Recipe programs. A settled mismatch may nominate alternatives, but the bank
// has no opcode or authority that can manufacture participation, consequence,
// credit, or a persistent revision. Constructor credit may only mark the
// alternatives evidence-ready for a later ordinary construction transaction.

#include <climits>
#include <cstdint>
#include <type_traits>

#include "hardware_native/direct_adult_core.cuh"
#include "hardware_native/direct_adult_constructor_meta_update.cuh"
#include "hardware_native/direct_adult_recipe_ir.cuh"
#include "hardware_native/direct_adult_runtime_frontiers.cuh"
#include "hardware_native/direct_resource_ecology.cuh"

namespace substrate::direct_adult_core {

inline constexpr std::uint32_t kResidentRecipeSearchBranchCount = 2u;
inline constexpr std::uint32_t kResidentRecipeSearchCapacity = 4u;
inline constexpr std::uint32_t kResidentRecipeSearchLifetimeTicks = 64u;

enum class ResidentRecipeSearchCandidateState : std::uint32_t {
  free = 0u,
  live = 1u,
  expired = 2u,
};

enum class ResidentRecipeSearchMutation : std::uint32_t {
  credit_scale = 1u,
  symmetric_clamp = 2u,
};

struct alignas(8) ResidentRecipeSearchCandidate {
  direct_network::ResidentRecipeIrProgram program;
  std::uint64_t identity;
  std::uint64_t parent_logical_recipe_id;
  std::uint64_t parent_revision_identity;
  std::uint64_t lived_mismatch_identity;
  std::uint64_t actual_occurrence_identity;
  std::uint32_t recipe_cell;
  std::uint32_t proposal_tick;
  std::uint32_t expiry_tick;
  std::uint32_t work_units;
  std::int32_t defect_q16;
  ResidentRecipeSearchMutation mutation;
  ResidentRecipeSearchCandidateState state;
  std::uint32_t constructor_credit_ready;
  std::uint32_t reserved;
};
static_assert(std::is_standard_layout_v<ResidentRecipeSearchCandidate> &&
              std::is_trivial_v<ResidentRecipeSearchCandidate>);

struct alignas(8) ResidentRecipeSearchFrontier {
  ResidentRecipeSearchCandidate candidates[kResidentRecipeSearchCapacity];
  std::uint64_t source_mismatch_identity;
  std::uint32_t live_count;
  std::uint32_t generations;
  std::uint32_t refusals;
  std::uint32_t authorized_candidates;
};
static_assert(std::is_standard_layout_v<ResidentRecipeSearchFrontier> &&
              std::is_trivial_v<ResidentRecipeSearchFrontier>);

DIRECT_ADULT_HD inline std::int32_t resident_recipe_search_scaled_operand(
    std::int32_t operand, std::int32_t defect_q16) {
  if (operand <= 0) return 0;
  const std::int64_t numerator = defect_q16 < 0 ? operand :
      static_cast<std::int64_t>(operand) * 3;
  const std::int64_t value = numerator / 2;
  if (value <= 0 || value > INT32_MAX) return 0;
  return static_cast<std::int32_t>(value);
}

DIRECT_ADULT_HD inline bool resident_recipe_search_lived_evidence(
    const direct_network::DirectBrain& brain,
    const ResidentMismatchCreditReceipt& receipt,
    const direct_network::ResidentRecipeCell& cell) {
  using namespace direct_network;
  if (brain.development == nullptr || receipt.identity == 0u ||
      receipt.target_occurrence_identity == 0u ||
      receipt.target_participation_identity == 0u ||
      receipt.target_logical_recipe_id != cell.logical_recipe_id ||
      receipt.target_revision_identity == 0u ||
      receipt.committed_revision_identity == 0u ||
      receipt.committed_revision_identity != cell.revision_identity ||
      receipt.causal_credit_delta_q16 == 0)
    return false;
  const auto& history = brain.development->exact_history;
  if (history.phase_kind != DirectExactHistoryKind::empty) return false;
  std::uint32_t matches = 0u;
  for (std::uint32_t i = 0u; i < history.committed_slots; ++i) {
    const auto& event = history.records[i];
    matches += event.kind == DirectExactHistoryKind::recipe_revision &&
                       event.identity == receipt.committed_revision_identity &&
                       event.parent_identity == receipt.target_revision_identity &&
                       event.source == receipt.target_recipe_cell &&
                       event.subject == static_cast<std::uint32_t>(
                           ResidentRecipeRevisionAuthority::experience) &&
                       event.incarnation_before ==
                           receipt.target_occurrence_identity
                   ? 1u
                   : 0u;
  }
  return matches == 1u;
}

DIRECT_ADULT_HD inline std::uint64_t resident_recipe_search_identity(
    const ResidentMismatchCreditReceipt& receipt,
    const direct_network::ResidentRecipeCell& cell,
    const direct_network::ResidentRecipeIrProgram& program,
    ResidentRecipeSearchMutation mutation, std::uint32_t proposal_tick) {
  using direct_network::exact_history_fold_word;
  std::uint64_t identity = exact_history_fold_word(
      0x7265636973656172ull, receipt.identity);
  identity = exact_history_fold_word(identity, cell.revision_identity);
  identity = exact_history_fold_word(identity, program.program_identity);
  identity = exact_history_fold_word(identity,
      static_cast<std::uint32_t>(mutation));
  identity = exact_history_fold_word(identity, proposal_tick);
  return identity == 0u ? 1u : identity;
}

// Retires transient candidate matter exactly once. Persistent Recipe cells and
// exact history are deliberately outside this operation.
__device__ inline std::uint32_t expire_resident_recipe_search(
    direct_adult::DirectResourceEcologyState* ecology,
    ResidentRecipeSearchFrontier* frontier, std::uint32_t current_tick) {
  if (ecology == nullptr || frontier == nullptr ||
      frontier->live_count > kResidentRecipeSearchCapacity)
    return 0u;
  std::uint32_t expired = 0u;
  for (std::uint32_t i = 0u; i < kResidentRecipeSearchCapacity; ++i) {
    auto& candidate = frontier->candidates[i];
    if (candidate.state != ResidentRecipeSearchCandidateState::live ||
        candidate.expiry_tick >= current_tick)
      continue;
    candidate.state = ResidentRecipeSearchCandidateState::expired;
    direct_adult::device_release_pool_units(
        ecology, direct_adult::DirectResourcePoolKind::topology_proposal, 1u);
    --frontier->live_count;
    ++expired;
  }
  return expired;
}

// Generates two heterogeneous data-program alternatives from one lived local
// defect/opportunity. The bounded history scan validates that the mismatch was
// already admitted by the ordinary experience-revision transaction. No brain
// scan, host candidate, cached graft, or persistent mutation is involved.
__device__ inline bool generate_resident_recipe_search(
    const direct_network::DirectBrain& brain,
    const ResidentMismatchCreditReceipt& receipt, std::uint32_t proposal_tick,
    ResidentRecipeSearchFrontier* frontier) {
  using namespace direct_network;
  using direct_adult::DirectResourcePoolKind;
  if (frontier == nullptr || brain.resource_ecology == nullptr ||
      brain.development == nullptr || brain.recipe_cells == nullptr ||
      receipt.target_recipe_cell >= brain.development->recipe_cell_count) {
    if (frontier != nullptr) ++frontier->refusals;
    return false;
  }
  const ResidentRecipeCell& cell =
      brain.recipe_cells[receipt.target_recipe_cell];
  if ((cell.flags & kResidentRecipeLevelCConstructor) == 0u ||
      !resident_recipe_ir_intact(cell.update_program) ||
      !resident_recipe_search_lived_evidence(brain, receipt, cell)) {
    ++frontier->refusals;
    return false;
  }
  for (std::uint32_t i = 0u; i < kResidentRecipeSearchCapacity; ++i)
    if (frontier->candidates[i].state ==
            ResidentRecipeSearchCandidateState::live &&
        frontier->candidates[i].lived_mismatch_identity == receipt.identity) {
      ++frontier->refusals;
      return false;
    }

  std::uint32_t slots[kResidentRecipeSearchBranchCount]{};
  std::uint32_t free_count = 0u;
  for (std::uint32_t i = 0u;
       i < kResidentRecipeSearchCapacity &&
       free_count < kResidentRecipeSearchBranchCount; ++i)
    if (frontier->candidates[i].state !=
        ResidentRecipeSearchCandidateState::live)
      slots[free_count++] = i;
  if (free_count != kResidentRecipeSearchBranchCount ||
      !direct_adult::device_reserve_pool_units(
          brain.resource_ecology, DirectResourcePoolKind::topology_proposal,
          kResidentRecipeSearchBranchCount)) {
    ++frontier->refusals;
    return false;
  }

  ResidentRecipeSearchCandidate staged[kResidentRecipeSearchBranchCount]{};
  for (std::uint32_t branch = 0u;
       branch < kResidentRecipeSearchBranchCount; ++branch) {
    auto& candidate = staged[branch];
    candidate.program = cell.update_program;
    candidate.mutation = branch == 0u
        ? ResidentRecipeSearchMutation::credit_scale
        : ResidentRecipeSearchMutation::symmetric_clamp;
    const std::uint32_t instruction = branch == 0u ? 1u : 2u;
    const std::int32_t operand = resident_recipe_search_scaled_operand(
        candidate.program.instructions[instruction].operand_q16,
        receipt.causal_credit_delta_q16);
    if (operand == 0 ||
        operand == candidate.program.instructions[instruction].operand_q16) {
      direct_adult::device_cancel_pool_reservation(
          brain.resource_ecology, DirectResourcePoolKind::topology_proposal,
          kResidentRecipeSearchBranchCount);
      ++frontier->refusals;
      return false;
    }
    candidate.program.instructions[instruction].operand_q16 = operand;
    candidate.program.program_identity = resident_recipe_ir_identity(
        candidate.program);
    if (!resident_recipe_ir_intact(candidate.program)) {
      direct_adult::device_cancel_pool_reservation(
          brain.resource_ecology, DirectResourcePoolKind::topology_proposal,
          kResidentRecipeSearchBranchCount);
      ++frontier->refusals;
      return false;
    }
    candidate.identity = resident_recipe_search_identity(
        receipt, cell, candidate.program, candidate.mutation, proposal_tick);
    candidate.parent_logical_recipe_id = cell.logical_recipe_id;
    candidate.parent_revision_identity = cell.revision_identity;
    candidate.lived_mismatch_identity = receipt.identity;
    candidate.actual_occurrence_identity =
        receipt.target_occurrence_identity;
    candidate.recipe_cell = receipt.target_recipe_cell;
    candidate.proposal_tick = proposal_tick;
    candidate.expiry_tick = proposal_tick + kResidentRecipeSearchLifetimeTicks;
    candidate.work_units = cell.update_program.op_count + 1u;
    candidate.defect_q16 = receipt.causal_credit_delta_q16;
    candidate.state = ResidentRecipeSearchCandidateState::live;
  }
  if (staged[0].identity == staged[1].identity ||
      staged[0].program.program_identity == staged[1].program.program_identity ||
      !direct_adult::device_commit_pool_units(
          brain.resource_ecology, DirectResourcePoolKind::topology_proposal,
          kResidentRecipeSearchBranchCount)) {
    direct_adult::device_cancel_pool_reservation(
        brain.resource_ecology, DirectResourcePoolKind::topology_proposal,
        kResidentRecipeSearchBranchCount);
    ++frontier->refusals;
    return false;
  }
  std::uint64_t work = 0u;
  for (std::uint32_t i = 0u; i < kResidentRecipeSearchBranchCount; ++i) {
    frontier->candidates[slots[i]] = staged[i];
    work += staged[i].work_units;
  }
  frontier->source_mismatch_identity = receipt.identity;
  frontier->live_count += kResidentRecipeSearchBranchCount;
  ++frontier->generations;
  direct_adult::device_record_work_counters(
      brain.resource_ecology, 1u, work, 0u);
  direct_adult::device_record_churn_proposal(
      brain.resource_ecology, kResidentRecipeSearchBranchCount);
  return true;
}

// Constructor viability can authorize every still-unresolved alternative for
// later ordinary construction. It does not choose a winner or change a Recipe.
__device__ inline std::uint32_t authorize_resident_recipe_search(
    const direct_network::DirectBrain& brain,
    ResidentRecipeSearchFrontier* frontier) {
  if (frontier == nullptr) return 0u;
  ConstructorMetaUpdatePlan plan{};
  plan_constructor_meta_updates(brain, &plan);
  std::uint32_t authorized = 0u;
  for (std::uint32_t i = 0u; i < kResidentRecipeSearchCapacity; ++i) {
    auto& candidate = frontier->candidates[i];
    if (candidate.state != ResidentRecipeSearchCandidateState::live ||
        candidate.constructor_credit_ready != 0u)
      continue;
    for (std::uint32_t p = 0u; p < plan.entry_count; ++p)
      if (plan.entries[p].recipe_cell == candidate.recipe_cell) {
        candidate.constructor_credit_ready = 1u;
        ++authorized;
        break;
      }
  }
  frontier->authorized_candidates += authorized;
  return authorized;
}

}  // namespace substrate::direct_adult_core

#endif  // HARDWARE_NATIVE_DIRECT_ADULT_RESIDENT_RECIPE_SEARCH_CUH
