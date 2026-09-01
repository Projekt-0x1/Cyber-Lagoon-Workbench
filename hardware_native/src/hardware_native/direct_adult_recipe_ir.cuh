#ifndef HARDWARE_NATIVE_DIRECT_ADULT_RECIPE_IR_CUH
#define HARDWARE_NATIVE_DIRECT_ADULT_RECIPE_IR_CUH

#include <climits>

#include "hardware_native/direct_network_brain.cuh"

namespace substrate::direct_adult_core {

using direct_network::ResidentRecipeIrInstruction;
using direct_network::ResidentRecipeIrOp;
using direct_network::ResidentRecipeIrProgram;
using direct_network::kResidentRecipeIrAbiV1;
using direct_network::kResidentRecipeIrCapacity;
using direct_network::make_resident_recipe_update_ir;
using direct_network::resident_recipe_ir_identity;

// The one bounded numeric interpreter compiled into the Adult binary.
inline constexpr std::uint64_t kResidentRecipeIrExecutorIdentity =
    0x7265636972657862ull;

struct ResidentRecipeIrEvidence {
  std::uint64_t subject_identity;
  std::uint64_t binding_identity;
  std::uint64_t consequence_identity;
  std::uint64_t occurrence_identity;
  std::uint64_t participation_identity;
  std::uint64_t active_work_identity;
  std::uint64_t logical_recipe_id;
  std::uint64_t prior_revision_identity;
  std::int32_t exact_credit_delta_q16;
  std::uint32_t recipe_cell;
};
struct ResidentRecipeIrResult {
  std::uint64_t execution_identity;
  std::int32_t parameter_delta_q16;
  std::uint32_t work_units;
};
static_assert(std::is_standard_layout_v<ResidentRecipeIrEvidence> &&
              std::is_trivial_v<ResidentRecipeIrEvidence> &&
              std::has_unique_object_representations_v<ResidentRecipeIrEvidence>);
static_assert(std::is_standard_layout_v<ResidentRecipeIrResult> &&
              std::is_trivial_v<ResidentRecipeIrResult> &&
              std::has_unique_object_representations_v<ResidentRecipeIrResult>);

__host__ __device__ inline std::uint64_t resident_recipe_ir_subject_identity(
    const direct_network::Root256& birth_root) {
  std::uint64_t identity = direct_network::exact_history_fold_word(
      0x726563697375626aull, kResidentRecipeIrExecutorIdentity);
  for (std::uint32_t i = 0u; i < 8u; ++i)
    identity = direct_network::exact_history_fold_word(identity,
                                                       birth_root.word[i]);
  return identity == 0u ? 1u : identity;
}

__host__ __device__ inline std::uint64_t resident_recipe_ir_binding_identity(
    std::uint64_t subject_identity, std::uint64_t logical_recipe_id,
    std::uint64_t bound_revision_identity, std::uint64_t program_identity) {
  std::uint64_t identity = direct_network::exact_history_fold_word(
      0x726563697269626eull, subject_identity);
  identity = direct_network::exact_history_fold_word(identity, logical_recipe_id);
  identity = direct_network::exact_history_fold_word(identity,
                                                      bound_revision_identity);
  identity = direct_network::exact_history_fold_word(identity, program_identity);
  identity = direct_network::exact_history_fold_word(
      identity, kResidentRecipeIrExecutorIdentity);
  return identity == 0u ? 1u : identity;
}

__host__ __device__ inline std::uint64_t resident_recipe_ir_binding_identity(
    std::uint64_t subject_identity,
    const direct_network::ResidentRecipeCell& cell) {
  return resident_recipe_ir_binding_identity(
      subject_identity, cell.logical_recipe_id, cell.revision_identity,
      cell.update_program.program_identity);
}

__host__ __device__ inline bool resident_recipe_ir_intact(
    const ResidentRecipeIrProgram& program) {
  if (program.abi_version != kResidentRecipeIrAbiV1 ||
      program.op_count < 3u || program.op_count > kResidentRecipeIrCapacity ||
      program.layout_stride == 0u ||
      program.program_identity != resident_recipe_ir_identity(program) ||
      program.instructions[program.op_count - 1u].op != ResidentRecipeIrOp::halt)
    return false;
  bool loaded = false, emitted = false;
  for (std::uint32_t i = 0u; i < kResidentRecipeIrCapacity; ++i) {
    const auto& instruction = program.instructions[i];
    if (instruction.reserved != 0u) return false;
    if (i >= program.op_count) {
      if (instruction.op != ResidentRecipeIrOp::halt ||
          instruction.operand_q16 != 0)
        return false;
      continue;
    }
    switch (instruction.op) {
      case ResidentRecipeIrOp::load_exact_credit:
        if (loaded || emitted || instruction.operand_q16 != 0) return false;
        loaded = true;
        break;
      case ResidentRecipeIrOp::scale_q16:
        if (!loaded || emitted || instruction.operand_q16 == 0) return false;
        break;
      case ResidentRecipeIrOp::clamp_symmetric_q16:
        if (!loaded || emitted || instruction.operand_q16 <= 0) return false;
        break;
      case ResidentRecipeIrOp::emit_parameter_delta:
        if (!loaded || emitted || instruction.operand_q16 != 0) return false;
        emitted = true;
        break;
      case ResidentRecipeIrOp::halt:
        if (i + 1u != program.op_count || !emitted ||
            instruction.operand_q16 != 0)
          return false;
        break;
      default:
        return false;
    }
  }
  return loaded && emitted;
}

__host__ __device__ inline std::uint64_t resident_recipe_ir_execution_identity(
    const ResidentRecipeIrProgram& program,
    const ResidentRecipeIrEvidence& evidence, std::int32_t parameter_delta,
    std::uint32_t work_units) {
  std::uint64_t identity = direct_network::exact_history_fold_word(
      0x7265636972657865ull, kResidentRecipeIrExecutorIdentity);
  identity = direct_network::exact_history_fold_word(identity,
                                                      program.program_identity);
  identity = direct_network::exact_history_fold_word(identity,
                                                      evidence.subject_identity);
  identity = direct_network::exact_history_fold_word(identity,
                                                      evidence.binding_identity);
  identity = direct_network::exact_history_fold_word(
      identity, evidence.consequence_identity);
  identity = direct_network::exact_history_fold_word(
      identity, evidence.occurrence_identity);
  identity = direct_network::exact_history_fold_word(
      identity, evidence.participation_identity);
  identity = direct_network::exact_history_fold_word(
      identity, evidence.active_work_identity);
  identity = direct_network::exact_history_fold_word(
      identity, evidence.logical_recipe_id);
  identity = direct_network::exact_history_fold_word(
      identity, evidence.prior_revision_identity);
  identity = direct_network::exact_history_fold_word(
      identity, static_cast<std::uint32_t>(evidence.exact_credit_delta_q16));
  identity = direct_network::exact_history_fold_word(identity,
                                                      evidence.recipe_cell);
  identity = direct_network::exact_history_fold_word(
      identity, static_cast<std::uint32_t>(parameter_delta));
  identity = direct_network::exact_history_fold_word(identity, work_units);
  return identity == 0u ? 1u : identity;
}

// The program returns a proposed parameter delta.  It cannot mutate the
// authenticated evidence or the Recipe cell; the enclosing resident
// transaction validates and commits both atomically.
__host__ __device__ inline bool execute_resident_recipe_ir(
    const ResidentRecipeIrProgram& program,
    const ResidentRecipeIrEvidence& evidence, ResidentRecipeIrResult* out) {
  if (out == nullptr || !resident_recipe_ir_intact(program) ||
      evidence.subject_identity == 0u || evidence.binding_identity == 0u ||
      evidence.consequence_identity == 0u ||
      evidence.occurrence_identity == 0u ||
      evidence.participation_identity == 0u ||
      evidence.active_work_identity == 0u || evidence.logical_recipe_id == 0u ||
      evidence.prior_revision_identity == 0u ||
      evidence.exact_credit_delta_q16 == 0)
    return false;
  std::int64_t value = 0;
  std::uint32_t work_units = 0u;
  bool emitted = false;
  for (std::uint32_t pc = 0u; pc < program.op_count; ++pc) {
    const auto& instruction = program.instructions[pc];
    switch (instruction.op) {
      case ResidentRecipeIrOp::load_exact_credit:
        value = evidence.exact_credit_delta_q16;
        ++work_units;
        break;
      case ResidentRecipeIrOp::scale_q16:
        value = (value * static_cast<std::int64_t>(instruction.operand_q16)) >>
                16u;
        if (value < INT32_MIN || value > INT32_MAX) return false;
        ++work_units;
        break;
      case ResidentRecipeIrOp::clamp_symmetric_q16: {
        const std::int64_t bound = instruction.operand_q16;
        value = value < -bound ? -bound : (value > bound ? bound : value);
        ++work_units;
        break;
      }
      case ResidentRecipeIrOp::emit_parameter_delta:
        if (value < INT32_MIN || value > INT32_MAX) return false;
        emitted = true;
        ++work_units;
        break;
      case ResidentRecipeIrOp::halt:
        pc = program.op_count;
        break;
      default:
        return false;
    }
  }
  if (!emitted || work_units == 0u) return false;
  ResidentRecipeIrResult result{};
  result.parameter_delta_q16 = static_cast<std::int32_t>(value);
  result.work_units = work_units;
  result.execution_identity = resident_recipe_ir_execution_identity(
      program, evidence, result.parameter_delta_q16, result.work_units);
  *out = result;
  return true;
}

}  // namespace substrate::direct_adult_core

#endif
