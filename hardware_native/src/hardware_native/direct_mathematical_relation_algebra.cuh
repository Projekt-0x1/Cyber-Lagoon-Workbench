#ifndef HARDWARE_NATIVE_DIRECT_MATHEMATICAL_RELATION_ALGEBRA_CUH
#define HARDWARE_NATIVE_DIRECT_MATHEMATICAL_RELATION_ALGEBRA_CUH

#include <cstdint>
#include <limits>
#include <type_traits>

#include "hardware_native/direct_mathematical_relation_solver.cuh"

#if defined(__CUDACC__)
#define DIRECT_ALGEBRA_HD __host__ __device__
#else
#define DIRECT_ALGEBRA_HD
#endif

namespace substrate::direct_network {

// Mathematical family is resident definition physics, never a semantic name
// or solver choice. New families extend this unit's single evaluator seam;
// callers and numerical executors remain unchanged. GitHub #1406.
enum class DirectRelationAlgebraFamilyV1 : std::uint32_t {
  linear = 0u,
  polynomial = 1u,
  state_space = 2u,
  discrete = 3u,
  semiring = 4u,
};

inline constexpr std::uint32_t kDirectRelationAlgebraFamilyCount = 5u;
inline constexpr std::uint32_t kDirectRelationAlgebraDispatchVersion = 1u;

struct DirectRelationAlgebraDefinitionV1 {
  DirectRelationAlgebraFamilyV1 family;
  std::uint32_t cell_count;
  std::uint32_t edge_count;
  std::uint32_t reserved;
  std::int64_t parameter_q16;
  const ResidentRecipeCell* cells;
  const ResidentRecipeEdge* edges;
};
static_assert(std::is_trivially_copyable_v<DirectRelationAlgebraDefinitionV1>);

struct DirectRelationAlgebraResidualV1 {
  DirectRelationAlgebraFamilyV1 family;
  std::uint32_t variable_index;
  std::int64_t residual_q16;
  std::uint32_t satisfied;
  std::uint32_t reserved;
};
static_assert(std::is_trivially_copyable_v<DirectRelationAlgebraResidualV1>);

// Exact whitebox condensation keeps only the polynomial boundary. Integer
// multipliers act on q16 values without learned coefficients or tolerances.
struct alignas(8) DirectWhiteboxBoundaryQ16 {
  std::int64_t quadratic_multiplier, linear_multiplier, offset_q16;
  std::uint32_t degree, reserved;
};
struct DirectWhiteboxAffineStepQ16 {
  std::int64_t multiplier, offset_q16;
};
struct DirectWhiteboxQuadraticQ16 {
  std::int64_t quadratic_multiplier, linear_multiplier, offset_q16;
};
struct DirectWhiteboxSchurClosureQ16 {
  // h = internal_multiplier*x + internal_offset;
  // y = direct_multiplier*x + coupling_multiplier*h + output_offset.
  std::int64_t internal_multiplier, internal_offset_q16;
  std::int64_t direct_multiplier, coupling_multiplier, output_offset_q16;
};
static_assert(sizeof(DirectWhiteboxBoundaryQ16) == 32u &&
              std::is_trivial_v<DirectWhiteboxBoundaryQ16> &&
              std::has_unique_object_representations_v<DirectWhiteboxBoundaryQ16>);
static_assert(std::has_unique_object_representations_v<DirectWhiteboxAffineStepQ16> &&
              std::has_unique_object_representations_v<DirectWhiteboxQuadraticQ16> &&
              std::has_unique_object_representations_v<DirectWhiteboxSchurClosureQ16>);

DIRECT_ALGEBRA_HD inline bool direct_relation_algebra_family_valid(
    DirectRelationAlgebraFamilyV1 family) {
  return static_cast<std::uint32_t>(family) < kDirectRelationAlgebraFamilyCount;
}

DIRECT_ALGEBRA_HD inline std::int64_t direct_relation_algebra_saturating_mul(
    std::int64_t value, std::int32_t coefficient) {
  constexpr std::int64_t kMax = INT64_MAX;
  constexpr std::int64_t kMin = INT64_MIN;
  if (value == 0 || coefficient == 0)
    return 0;
  if (coefficient > 0) {
    if (value > kMax / coefficient)
      return kMax;
    if (value < kMin / coefficient)
      return kMin;
  } else if (value > 0) {
    if (value > kMin / coefficient)
      return kMin;
  } else {
    if (value < kMax / coefficient)
      return kMax;
  }
  return value * static_cast<std::int64_t>(coefficient);
}

DIRECT_ALGEBRA_HD inline std::int64_t direct_relation_algebra_q16_mul(
    std::int64_t value_q16, std::int32_t coefficient_q16) {
  constexpr std::int64_t kQ16 = std::int64_t{1} << 16;
  const std::int64_t whole = value_q16 / kQ16;
  const std::int64_t remainder = value_q16 % kQ16;
  const std::int64_t whole_product = direct_relation_algebra_saturating_mul(whole, coefficient_q16);
  const std::int64_t remainder_product =
      direct_relation_algebra_saturating_mul(remainder, coefficient_q16) / kQ16;
  return direct_relation_saturating_add(whole_product, remainder_product);
}

DIRECT_ALGEBRA_HD inline bool direct_whitebox_checked_add(
    std::int64_t left, std::int64_t right, std::int64_t* out) {
  if (out == nullptr || (right > 0 && left > INT64_MAX - right) ||
      (right < 0 && left < INT64_MIN - right))
    return false;
  *out = left + right;
  return true;
}

DIRECT_ALGEBRA_HD inline bool direct_whitebox_checked_mul(
    std::int64_t left, std::int64_t right, std::int64_t* out) {
  if (out == nullptr) return false;
  if (left == 0 || right == 0) { *out = 0; return true; }
  if ((left == -1 && right == INT64_MIN) ||
      (right == -1 && left == INT64_MIN))
    return false;
  if ((left > 0 && right > 0 && left > INT64_MAX / right) ||
      (left > 0 && right < 0 && right < INT64_MIN / left) ||
      (left < 0 && right > 0 && left < INT64_MIN / right) ||
      (left < 0 && right < 0 && left < INT64_MAX / right))
    return false;
  *out = left * right;
  return true;
}

DIRECT_ALGEBRA_HD inline bool direct_whitebox_evaluate_boundary_q16(
    const DirectWhiteboxBoundaryQ16& boundary, std::int64_t input_q16,
    std::int64_t* output_q16) {
  if (output_q16 == nullptr || boundary.reserved != 0u ||
      (boundary.degree != 1u && boundary.degree != 2u) ||
      (boundary.degree == 1u && boundary.quadratic_multiplier != 0) ||
      input_q16 < INT32_MIN || input_q16 > INT32_MAX)
    return false;
  const std::int64_t square_q16 = (input_q16 * input_q16) >> 16;
  std::int64_t quadratic = 0, linear = 0, sum = 0;
  return direct_whitebox_checked_mul(
             square_q16, boundary.quadratic_multiplier, &quadratic) &&
      direct_whitebox_checked_mul(
             input_q16, boundary.linear_multiplier, &linear) &&
      direct_whitebox_checked_add(quadratic, linear, &sum) &&
      direct_whitebox_checked_add(sum, boundary.offset_q16, output_q16);
}

DIRECT_ALGEBRA_HD inline bool direct_whitebox_reduce_affine_q16(
    const DirectWhiteboxAffineStepQ16* steps, std::uint32_t step_count,
    DirectWhiteboxBoundaryQ16* out) {
  if (steps == nullptr || out == nullptr || step_count == 0u ||
      step_count > 4u)
    return false;
  DirectWhiteboxBoundaryQ16 boundary{0, 1, 0, 1u, 0u};
  for (std::uint32_t i = 0u; i < step_count; ++i) {
    std::int64_t multiplier = 0, offset = 0;
    if (!direct_whitebox_checked_mul(
            steps[i].multiplier, boundary.linear_multiplier, &multiplier) ||
        !direct_whitebox_checked_mul(
            steps[i].multiplier, boundary.offset_q16, &offset) ||
        !direct_whitebox_checked_add(offset, steps[i].offset_q16, &offset))
      return false;
    boundary.linear_multiplier = multiplier;
    boundary.offset_q16 = offset;
  }
  *out = boundary;
  return true;
}

DIRECT_ALGEBRA_HD inline bool direct_whitebox_reduce_quadratic_q16(
    const DirectWhiteboxQuadraticQ16& inner,
    const DirectWhiteboxAffineStepQ16& outer,
    DirectWhiteboxBoundaryQ16* out) {
  if (out == nullptr || inner.quadratic_multiplier == 0) return false;
  DirectWhiteboxBoundaryQ16 boundary{0, 0, 0, 2u, 0u};
  if (!direct_whitebox_checked_mul(
          inner.quadratic_multiplier, outer.multiplier,
          &boundary.quadratic_multiplier) ||
      !direct_whitebox_checked_mul(
          inner.linear_multiplier, outer.multiplier,
          &boundary.linear_multiplier) ||
      !direct_whitebox_checked_mul(
          inner.offset_q16, outer.multiplier, &boundary.offset_q16) ||
      !direct_whitebox_checked_add(
          boundary.offset_q16, outer.offset_q16, &boundary.offset_q16))
    return false;
  *out = boundary;
  return true;
}

DIRECT_ALGEBRA_HD inline bool direct_whitebox_reduce_schur_q16(
    const DirectWhiteboxSchurClosureQ16& closure,
    DirectWhiteboxBoundaryQ16* out) {
  if (out == nullptr) return false;
  DirectWhiteboxBoundaryQ16 boundary{0, 0, 0, 1u, 0u};
  std::int64_t indirect = 0;
  if (!direct_whitebox_checked_mul(
          closure.coupling_multiplier, closure.internal_multiplier,
          &indirect) ||
      !direct_whitebox_checked_add(
          closure.direct_multiplier, indirect, &boundary.linear_multiplier) ||
      !direct_whitebox_checked_mul(
          closure.coupling_multiplier, closure.internal_offset_q16,
          &indirect) ||
      !direct_whitebox_checked_add(
          closure.output_offset_q16, indirect, &boundary.offset_q16))
    return false;
  *out = boundary;
  return true;
}

DIRECT_ALGEBRA_HD inline bool direct_whitebox_boundary_equal(
    const DirectWhiteboxBoundaryQ16& left,
    const DirectWhiteboxBoundaryQ16& right) {
  return left.quadratic_multiplier == right.quadratic_multiplier &&
      left.linear_multiplier == right.linear_multiplier &&
      left.offset_q16 == right.offset_q16 && left.degree == right.degree &&
      left.reserved == 0u && right.reserved == 0u;
}

DIRECT_ALGEBRA_HD inline std::int64_t direct_relation_algebra_square_q16(std::int64_t value_q16) {
  constexpr std::int64_t kI32Max = INT32_MAX;
  constexpr std::int64_t kI32Min = INT32_MIN;
  if (value_q16 > kI32Max || value_q16 < kI32Min)
    return INT64_MAX;
  return direct_relation_algebra_q16_mul(value_q16, static_cast<std::int32_t>(value_q16));
}

DIRECT_ALGEBRA_HD inline bool direct_relation_algebra_definition_valid(
    const DirectRelationAlgebraDefinitionV1& definition) {
  if (!direct_relation_algebra_family_valid(definition.family) || definition.cells == nullptr ||
      definition.cell_count == 0u || definition.cell_count > 0xffffu ||
      (definition.edge_count != 0u && definition.edges == nullptr) ||
      (definition.family == DirectRelationAlgebraFamilyV1::discrete &&
       (definition.parameter_q16 == 0 || definition.parameter_q16 == INT64_MIN)))
    return false;
  for (std::uint32_t row = 0; row < definition.cell_count; ++row) {
    const ResidentRecipeCell cell = definition.cells[row];
    if (cell.edge_offset > definition.edge_count ||
        static_cast<std::uint32_t>(cell.edge_count) > definition.edge_count - cell.edge_offset)
      return false;
    for (std::uint32_t local = 0; local < cell.edge_count; ++local) {
      const ResidentRecipeEdge edge = definition.edges[cell.edge_offset + local];
      if (edge.source_cell != row || edge.target_cell >= definition.cell_count)
        return false;
    }
  }
  return true;
}

// The sole family dispatch seam. It evaluates one row target without writing
// definition, values, participation, credit, or revision state.
DIRECT_ALGEBRA_HD inline bool direct_relation_algebra_evaluate_row(
    const DirectRelationAlgebraDefinitionV1& definition, std::uint32_t variable_index,
    const std::int64_t* values_q16, std::int64_t* out_target_q16) {
  if (values_q16 == nullptr || out_target_q16 == nullptr ||
      variable_index >= definition.cell_count ||
      !direct_relation_algebra_definition_valid(definition))
    return false;
  const ResidentRecipeCell cell = definition.cells[variable_index];
  std::int64_t target = direct_relation_saturating_add(cell.support_q16, cell.credit_q16);

  if (definition.family == DirectRelationAlgebraFamilyV1::semiring) {
    for (std::uint32_t local = 0; local < cell.edge_count; ++local) {
      const ResidentRecipeEdge edge = definition.edges[cell.edge_offset + local];
      const std::int64_t candidate =
          direct_relation_saturating_add(values_q16[edge.target_cell], edge.weight_q16);
      if (candidate < target)
        target = candidate;
    }
  } else {
    if (definition.family == DirectRelationAlgebraFamilyV1::state_space)
      target = direct_relation_saturating_add(target, definition.parameter_q16);
    for (std::uint32_t local = 0; local < cell.edge_count; ++local) {
      const ResidentRecipeEdge edge = definition.edges[cell.edge_offset + local];
      std::int64_t operand = values_q16[edge.target_cell];
      if (definition.family == DirectRelationAlgebraFamilyV1::polynomial)
        operand = direct_relation_algebra_square_q16(operand);
      target = direct_relation_saturating_add(
          target, direct_relation_algebra_q16_mul(operand, edge.weight_q16));
    }
    if (definition.family == DirectRelationAlgebraFamilyV1::discrete) {
      const std::int64_t step =
          definition.parameter_q16 < 0 ? -definition.parameter_q16 : definition.parameter_q16;
      target = (target / step) * step;
    }
  }
  *out_target_q16 = target;
  return true;
}

DIRECT_ALGEBRA_HD inline std::int64_t direct_relation_algebra_saturating_sub(std::int64_t left,
                                                                             std::int64_t right) {
  if (right == INT64_MIN)
    return left >= 0 ? INT64_MAX : left - right;
  return direct_relation_saturating_add(left, -right);
}

DIRECT_ALGEBRA_HD inline bool direct_relation_algebra_residuals(
    const DirectRelationAlgebraDefinitionV1& definition, const std::int64_t* values_q16,
    DirectRelationAlgebraResidualV1* out_residuals, std::uint32_t out_capacity) {
  if (!direct_relation_algebra_definition_valid(definition) || values_q16 == nullptr ||
      out_residuals == nullptr || out_capacity < definition.cell_count)
    return false;
  for (std::uint32_t row = 0; row < definition.cell_count; ++row) {
    std::int64_t target = 0;
    if (!direct_relation_algebra_evaluate_row(definition, row, values_q16, &target))
      return false;
    DirectRelationAlgebraResidualV1 residual{};
    residual.family = definition.family;
    residual.variable_index = row;
    residual.residual_q16 = direct_relation_algebra_saturating_sub(values_q16[row], target);
    residual.satisfied = residual.residual_q16 == 0 ? 1u : 0u;
    out_residuals[row] = residual;
  }
  return true;
}

// One solver seam for every family. All admission and capacity checks occur
// before the first transient value write; the persistent definition is const.
DIRECT_ALGEBRA_HD inline bool direct_relation_algebra_solve(
    const DirectRelationAlgebraDefinitionV1& definition, DirectRelationSolverKind solver,
    std::uint32_t iterations, std::int64_t* values_q16, std::int64_t* scratch_q16,
    DirectRelationAlgebraResidualV1* out_residuals, std::uint32_t out_capacity) {
  if (!direct_relation_algebra_definition_valid(definition) ||
      static_cast<std::uint32_t>(solver) >= kDirectRelationSolverKindCount || iterations == 0u ||
      values_q16 == nullptr || scratch_q16 == nullptr || out_residuals == nullptr ||
      out_capacity < definition.cell_count)
    return false;
  for (std::uint32_t sweep = 0; sweep < iterations; ++sweep) {
    const bool frozen = solver == DirectRelationSolverKind::jacobi;
    if (frozen)
      for (std::uint32_t row = 0; row < definition.cell_count; ++row)
        scratch_q16[row] = values_q16[row];
    const std::int64_t* read_from = frozen ? scratch_q16 : values_q16;
    for (std::uint32_t row = 0; row < definition.cell_count; ++row) {
      std::int64_t target = 0;
      if (!direct_relation_algebra_evaluate_row(definition, row, read_from, &target))
        return false;
      values_q16[row] = target;
    }
  }
  return direct_relation_algebra_residuals(definition, values_q16, out_residuals, out_capacity);
}

enum class DirectWhiteboxReductionKindV1 : std::uint32_t {
  affine_composition = 0u,
  polynomial_substitution = 1u,
  schur_complement = 2u,
  deterministic_bisimulation = 3u,
};
inline constexpr std::uint32_t kDirectWhiteboxReductionKindCount = 4u;
inline constexpr std::uint32_t kDirectWhiteboxReductionWidth = 4u;

// Fixed resident IR for one bounded exact-reduction transaction. Coefficients
// are Q16. For affine/polynomial/Schur, coefficient_count is four. For unary
// deterministic bisimulation, state_count is 1..4 and each state owns one
// successor and one observable output.
struct DirectWhiteboxReductionSourceV1 {
  DirectWhiteboxReductionKindV1 kind;
  std::uint32_t coefficient_count, state_count, reserved;
  std::int32_t coefficients_q16[kDirectWhiteboxReductionWidth];
  std::uint32_t successors[kDirectWhiteboxReductionWidth];
  std::int32_t outputs_q16[kDirectWhiteboxReductionWidth];
  std::uint64_t source_identity;
};
struct DirectWhiteboxCondensationV1 {
  DirectWhiteboxReductionSourceV1 source;
  std::int32_t result_q16[kDirectWhiteboxReductionWidth];
  std::uint32_t result_u32[kDirectWhiteboxReductionWidth];
  std::uint8_t class_by_state[kDirectWhiteboxReductionWidth];
  std::uint32_t result_count, eliminated_count;
  std::uint64_t witness_identity;
};
static_assert(std::is_trivial_v<DirectWhiteboxReductionSourceV1> &&
              std::is_standard_layout_v<DirectWhiteboxReductionSourceV1>);
static_assert(std::is_trivial_v<DirectWhiteboxCondensationV1> &&
              std::is_standard_layout_v<DirectWhiteboxCondensationV1>);

DIRECT_ALGEBRA_HD inline std::uint64_t direct_whitebox_source_identity(
    const DirectWhiteboxReductionSourceV1& source) {
  std::uint64_t identity = exact_history_fold_word(
      0x7768697465626f78ull, static_cast<std::uint32_t>(source.kind));
  identity = exact_history_fold_word(identity, source.coefficient_count);
  identity = exact_history_fold_word(identity, source.state_count);
  for (std::uint32_t i = 0u; i < kDirectWhiteboxReductionWidth; ++i) {
    identity = exact_history_fold_word(
        identity, static_cast<std::uint32_t>(source.coefficients_q16[i]));
    identity = exact_history_fold_word(identity, source.successors[i]);
    identity = exact_history_fold_word(
        identity, static_cast<std::uint32_t>(source.outputs_q16[i]));
  }
  return identity == 0u ? 1u : identity;
}

DIRECT_ALGEBRA_HD inline bool direct_whitebox_exact_q16_mul(
    std::int32_t left, std::int32_t right, std::int32_t* out) {
  if (out == nullptr) return false;
  constexpr std::int64_t q = std::int64_t{1} << 16;
  const std::int64_t product = static_cast<std::int64_t>(left) * right;
  if (product % q != 0) return false;
  const std::int64_t value = product / q;
  if (value < INT32_MIN || value > INT32_MAX) return false;
  *out = static_cast<std::int32_t>(value);
  return true;
}
DIRECT_ALGEBRA_HD inline bool direct_whitebox_exact_q16_div(
    std::int32_t numerator, std::int32_t denominator, std::int32_t* out) {
  if (out == nullptr || denominator == 0) return false;
  constexpr std::int64_t q = std::int64_t{1} << 16;
  const std::int64_t scaled = static_cast<std::int64_t>(numerator) * q;
  if (scaled % denominator != 0) return false;
  const std::int64_t value = scaled / denominator;
  if (value < INT32_MIN || value > INT32_MAX) return false;
  *out = static_cast<std::int32_t>(value);
  return true;
}
DIRECT_ALGEBRA_HD inline bool direct_whitebox_exact_add(
    std::int32_t left, std::int32_t right, std::int32_t* out) {
  if (out == nullptr) return false;
  const std::int64_t value = static_cast<std::int64_t>(left) + right;
  if (value < INT32_MIN || value > INT32_MAX) return false;
  *out = static_cast<std::int32_t>(value);
  return true;
}

DIRECT_ALGEBRA_HD inline std::uint64_t direct_whitebox_witness_identity(
    const DirectWhiteboxCondensationV1& value) {
  std::uint64_t identity = exact_history_fold_word(
      0x6578616374726564ull, value.source.source_identity);
  identity = exact_history_fold_word(
      identity, static_cast<std::uint32_t>(value.source.kind));
  identity = exact_history_fold_word(identity, value.result_count);
  identity = exact_history_fold_word(identity, value.eliminated_count);
  for (std::uint32_t i = 0u; i < kDirectWhiteboxReductionWidth; ++i) {
    identity = exact_history_fold_word(
        identity, static_cast<std::uint32_t>(value.result_q16[i]));
    identity = exact_history_fold_word(identity, value.result_u32[i]);
    identity = exact_history_fold_word(identity, value.class_by_state[i]);
  }
  return identity == 0u ? 1u : identity;
}

DIRECT_ALGEBRA_HD inline bool direct_whitebox_reduce_exact(
    const DirectWhiteboxReductionSourceV1& source,
    DirectWhiteboxCondensationV1* out) {
  if (out == nullptr ||
      static_cast<std::uint32_t>(source.kind) >=
          kDirectWhiteboxReductionKindCount ||
      source.source_identity == 0u ||
      source.source_identity != direct_whitebox_source_identity(source))
    return false;
  DirectWhiteboxCondensationV1 candidate{};
  candidate.source = source;
  if (source.kind != DirectWhiteboxReductionKindV1::deterministic_bisimulation) {
    if (source.coefficient_count != 4u || source.state_count != 0u) return false;
    const auto* c = source.coefficients_q16;
    if (source.kind == DirectWhiteboxReductionKindV1::affine_composition) {
      std::int32_t scaled_offset = 0;
      if (!direct_whitebox_exact_q16_mul(c[2], c[0], &candidate.result_q16[0]) ||
          !direct_whitebox_exact_q16_mul(c[2], c[1], &scaled_offset) ||
          !direct_whitebox_exact_add(scaled_offset, c[3], &candidate.result_q16[1]))
        return false;
      candidate.result_count = 2u;
      candidate.eliminated_count = 1u;
    } else if (source.kind ==
               DirectWhiteboxReductionKindV1::polynomial_substitution) {
      std::int32_t a2 = 0, ab = 0, b2 = 0, linear = 0;
      if (!direct_whitebox_exact_q16_mul(c[0], c[0], &a2) ||
          !direct_whitebox_exact_q16_mul(c[0], c[1], &ab) ||
          !direct_whitebox_exact_q16_mul(c[1], c[1], &b2) ||
          !direct_whitebox_exact_q16_mul(c[2], a2, &candidate.result_q16[0]) ||
          !direct_whitebox_exact_q16_mul(c[2], ab, &linear) ||
          !direct_whitebox_exact_add(linear, linear, &candidate.result_q16[1]) ||
          !direct_whitebox_exact_q16_mul(c[2], b2, &candidate.result_q16[2]) ||
          !direct_whitebox_exact_add(
              candidate.result_q16[2], c[3], &candidate.result_q16[2]))
        return false;
      candidate.result_count = 3u;
      candidate.eliminated_count = 1u;
    } else {
      constexpr std::int32_t q = 1 << 16;
      std::int32_t product = 0, quotient = 0;
      const std::int64_t denominator64 = static_cast<std::int64_t>(q) - c[3];
      if (denominator64 < INT32_MIN || denominator64 > INT32_MAX ||
          !direct_whitebox_exact_q16_mul(c[1], c[2], &product) ||
          !direct_whitebox_exact_q16_div(
              product, static_cast<std::int32_t>(denominator64), &quotient) ||
          !direct_whitebox_exact_add(c[0], quotient, &candidate.result_q16[0]))
        return false;
      candidate.result_count = 1u;
      candidate.eliminated_count = 1u;
    }
  } else {
    if (source.coefficient_count != 0u || source.state_count == 0u ||
        source.state_count > kDirectWhiteboxReductionWidth)
      return false;
    for (std::uint32_t state = 0u; state < source.state_count; ++state)
      if (source.successors[state] >= source.state_count) return false;
    std::uint8_t classes[kDirectWhiteboxReductionWidth]{};
    for (std::uint32_t state = 0u; state < source.state_count; ++state) {
      std::uint8_t assigned = 0xffu;
      for (std::uint32_t prior = 0u; prior < state; ++prior)
        if (source.outputs_q16[prior] == source.outputs_q16[state]) {
          assigned = classes[prior];
          break;
        }
      if (assigned == 0xffu) assigned = static_cast<std::uint8_t>(state);
      classes[state] = assigned;
    }
    for (std::uint32_t pass = 0u; pass < source.state_count; ++pass) {
      std::uint8_t next[kDirectWhiteboxReductionWidth]{};
      std::uint8_t next_count = 0u;
      for (std::uint32_t state = 0u; state < source.state_count; ++state) {
        std::uint8_t assigned = 0xffu;
        for (std::uint32_t prior = 0u; prior < state; ++prior)
          if (source.outputs_q16[prior] == source.outputs_q16[state] &&
              classes[source.successors[prior]] ==
                  classes[source.successors[state]]) {
            assigned = next[prior];
            break;
          }
        if (assigned == 0xffu) assigned = next_count++;
        next[state] = assigned;
      }
      bool same = true;
      for (std::uint32_t state = 0u; state < source.state_count; ++state) {
        same = same && next[state] == classes[state];
        classes[state] = next[state];
      }
      if (same) break;
    }
    std::uint32_t class_count = 0u;
    for (std::uint32_t state = 0u; state < source.state_count; ++state) {
      candidate.class_by_state[state] = classes[state];
      if (static_cast<std::uint32_t>(classes[state]) + 1u > class_count)
        class_count = static_cast<std::uint32_t>(classes[state]) + 1u;
    }
    for (std::uint32_t state = 0u; state < source.state_count; ++state) {
      const std::uint32_t cls = classes[state];
      candidate.result_u32[cls] = classes[source.successors[state]];
      candidate.result_q16[cls] = source.outputs_q16[state];
    }
    candidate.result_count = class_count;
    candidate.eliminated_count = source.state_count - class_count;
  }
  candidate.witness_identity = direct_whitebox_witness_identity(candidate);
  *out = candidate;
  return true;
}

DIRECT_ALGEBRA_HD inline bool direct_whitebox_condensation_valid(
    const DirectWhiteboxCondensationV1& value) {
  DirectWhiteboxCondensationV1 expected{};
  if (!direct_whitebox_reduce_exact(value.source, &expected)) return false;
  if (value.result_count != expected.result_count ||
      value.eliminated_count != expected.eliminated_count ||
      value.witness_identity != expected.witness_identity)
    return false;
  for (std::uint32_t i = 0u; i < kDirectWhiteboxReductionWidth; ++i)
    if (value.result_q16[i] != expected.result_q16[i] ||
        value.result_u32[i] != expected.result_u32[i] ||
        value.class_by_state[i] != expected.class_by_state[i])
      return false;
  return true;
}

DIRECT_ALGEBRA_HD inline bool direct_whitebox_rematerialize(
    const DirectWhiteboxCondensationV1& value,
    DirectWhiteboxReductionSourceV1* out) {
  if (out == nullptr || !direct_whitebox_condensation_valid(value)) return false;
  *out = value.source;
  return true;
}

}  // namespace substrate::direct_network

#undef DIRECT_ALGEBRA_HD

#endif  // HARDWARE_NATIVE_DIRECT_MATHEMATICAL_RELATION_ALGEBRA_CUH
