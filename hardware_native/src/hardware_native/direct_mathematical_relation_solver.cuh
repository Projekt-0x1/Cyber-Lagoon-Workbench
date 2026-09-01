#ifndef HARDWARE_NATIVE_DIRECT_MATHEMATICAL_RELATION_SOLVER_CUH
#define HARDWARE_NATIVE_DIRECT_MATHEMATICAL_RELATION_SOLVER_CUH

#include <cstdint>
#include <type_traits>

#include "hardware_native/direct_mathematical_relation_ontology.cuh"

#if defined(__CUDACC__)
#define DIRECT_SOLVER_HD __host__ __device__
#else
#define DIRECT_SOLVER_HD
#endif

namespace substrate::direct_network {

// Swappable numerical solvers for implicit relation rows M(V;Theta)=0, where
// row i reads  -V_i + Theta_i + sum_j coefficient_ij * V_j = 0  using the
// ontology's own self coefficient. The definition lives in the ontology and
// is READ-ONLY to every solver; swapping the kind is an execution choice
// only. github #1236 c.relation_solver_separation.
enum class DirectRelationSolverKind : std::uint32_t {
  jacobi = 0u,
  gauss_seidel = 1u,
};

inline constexpr std::uint32_t kDirectRelationSolverKindCount = 2u;
inline constexpr std::uint32_t kDirectInverseThetaMaxSteps = 64u;

struct DirectInverseThetaSolutionV1 {
  std::int64_t theta_q16;
  std::int64_t observed_residual_q16;
  std::uint32_t work_units;
  DirectRelationSolverKind solver_kind;
};
static_assert(std::is_trivially_copyable_v<DirectInverseThetaSolutionV1>);

// Relaxed update by exact half-steps toward the row target: truncating
// division keeps the step sign-symmetric (an arithmetic right shift would
// stall one short of negative odd targets), and repeated application reaches
// the target exactly in finite steps, so receipts may assert exact
// equalities where the physics allows them instead of inventing tolerances.
DIRECT_SOLVER_HD
inline std::int64_t direct_relation_relaxed_step_q16(std::int64_t current_q16,
                                                     std::int64_t target_q16) {
  const std::int64_t delta = target_q16 - current_q16;
  return current_q16 + delta - delta / 2;
}

// Invert one already-evaluated implicit row for Theta.  The caller supplies
// the observed and predicted values from the same bound occurrence, so all
// non-Theta terms cancel: Theta' = Theta + (observed - predicted).  Both
// solvers remain read-only over the relation; this is only a transient
// proposal and deliberately carries no participation or credit authority.
DIRECT_SOLVER_HD
inline bool direct_relation_inverse_theta_q16(
    const DirectImplicitRelationRowV1& row, std::int64_t observed_q16,
    std::int64_t predicted_q16, DirectRelationSolverKind kind,
    std::uint32_t max_steps, DirectInverseThetaSolutionV1* out) {
  if (out == nullptr || row.self_coefficient_q16 !=
                            kDirectImplicitSelfCoefficientQ16 ||
      (kind != DirectRelationSolverKind::jacobi &&
       kind != DirectRelationSolverKind::gauss_seidel) ||
      max_steps == 0u || max_steps > kDirectInverseThetaMaxSteps)
    return false;
  constexpr std::int64_t kMax = 0x7fffffffffffffffLL;
  constexpr std::int64_t kMin = -kMax - 1;
  if ((predicted_q16 < 0 && observed_q16 > kMax + predicted_q16) ||
      (predicted_q16 > 0 && observed_q16 < kMin + predicted_q16))
    return false;
  const std::int64_t residual = observed_q16 - predicted_q16;
  if ((residual > 0 && row.theta_q16 > kMax - residual) ||
      (residual < 0 && row.theta_q16 < kMin - residual))
    return false;
  const std::int64_t target = row.theta_q16 + residual;
  std::int64_t theta = row.theta_q16;
  std::uint32_t work = 0u;
  if (kind == DirectRelationSolverKind::jacobi) {
    theta = target;
    work = 1u;
  } else {
    while (theta != target && work < max_steps) {
      theta = direct_relation_relaxed_step_q16(theta, target);
      ++work;
    }
    if (theta != target) return false;
  }
  *out = DirectInverseThetaSolutionV1{theta, residual, work, kind};
  return true;
}

// Row target Theta_i + sum_j coefficient_ij * V_j against the caller's value
// vector. Read-only over the definition.
DIRECT_SOLVER_HD
inline std::int64_t direct_relation_row_target_q16(
    const ResidentRecipeCell* cells, const ResidentRecipeEdge* edges,
    std::uint32_t edge_count, std::uint32_t variable_index,
    const std::int64_t* values_q16) {
  const ResidentRecipeCell cell = cells[variable_index];
  std::int64_t target = cell.support_q16 + cell.credit_q16;
  for (std::uint32_t k = 0; k < cell.edge_count && cell.edge_offset + k < edge_count; ++k) {
    const ResidentRecipeEdge edge = edges[cell.edge_offset + k];
    target += static_cast<std::int64_t>(edge.weight_q16) * values_q16[edge.target_cell];
  }
  return target;
}

// Residual M_i(V) straight from the definition: negative of the row target
// plus the row's own value (the self coefficient closes the left side).
DIRECT_SOLVER_HD
inline std::int64_t direct_relation_residual_q16(
    const ResidentRecipeCell* cells, const ResidentRecipeEdge* edges,
    std::uint32_t edge_count, std::uint32_t variable_index,
    const std::int64_t* values_q16) {
  return values_q16[variable_index] -
         direct_relation_row_target_q16(cells, edges, edge_count, variable_index,
                                        values_q16);
}

// github #1236 d.structured_relation_residual: solving reports constraint
// satisfaction as a structured per-row record, not a bare scalar. The record
// is a property of definition + values alone -- it carries no participant
// identity, ticket lineage, or credit field, so a residual cannot mint a
// revision or enter settlement; only exact-history receipts with actual
// participant lineage can.
enum class DirectRelationResidualState : std::uint32_t {
  satisfied = 0u,
  above_target = 1u,
  below_target = 2u,
};

struct DirectRelationResidualRecordV1 {
  std::uint32_t variable_index;
  DirectRelationResidualState state;
  std::int64_t residual_q16;
};
static_assert(std::is_trivially_copyable_v<DirectRelationResidualRecordV1>);

DIRECT_SOLVER_HD
inline bool direct_relation_structured_residuals(
    const ResidentRecipeCell* cells, std::uint32_t cell_count,
    const ResidentRecipeEdge* edges, std::uint32_t edge_count,
    const std::int64_t* values_q16,
    DirectRelationResidualRecordV1* out_records, std::uint32_t out_capacity) {
  if (cells == nullptr || values_q16 == nullptr ||
      (cell_count != 0u && out_records == nullptr) || out_capacity < cell_count ||
      cell_count > 0xffffu)
    return false;
  for (std::uint32_t i = 0; i < cell_count; ++i) {
    const std::int64_t residual =
        direct_relation_residual_q16(cells, edges, edge_count, i, values_q16);
    DirectRelationResidualRecordV1 record{};
    record.variable_index = i;
    record.state = residual == 0 ? DirectRelationResidualState::satisfied
                   : residual > 0 ? DirectRelationResidualState::above_target
                                  : DirectRelationResidualState::below_target;
    record.residual_q16 = residual;
    out_records[i] = record;
  }
  return true;
}

// Run `iterations` in-place sweeps over the caller's value buffer. Jacobi
// reads a frozen pre-sweep snapshot; Gauss-Seidel reads freshly written
// values. Neither touches cells or edges: mutating the definition from a
// solver is exactly the defect this node forbids.
DIRECT_SOLVER_HD
inline bool direct_relation_solve(
    const ResidentRecipeCell* cells, std::uint32_t cell_count,
    const ResidentRecipeEdge* edges, std::uint32_t edge_count,
    DirectRelationSolverKind kind, std::uint32_t iterations, std::int64_t* values_q16,
    std::int64_t* scratch_q16) {
  if (cells == nullptr || values_q16 == nullptr || scratch_q16 == nullptr ||
      cell_count == 0u || cell_count > 0xffffu)
    return false;
  for (std::uint32_t sweep = 0; sweep < iterations; ++sweep) {
    const bool frozen =
        kind == DirectRelationSolverKind::jacobi;
    if (frozen)
      for (std::uint32_t i = 0; i < cell_count; ++i) scratch_q16[i] = values_q16[i];
    const std::int64_t* read_from = frozen ? scratch_q16 : values_q16;
    for (std::uint32_t i = 0; i < cell_count; ++i) {
      const ResidentRecipeCell cell = cells[i];
      std::int64_t target = cell.support_q16 + cell.credit_q16;
      for (std::uint32_t k = 0; k < cell.edge_count && cell.edge_offset + k < edge_count; ++k) {
        const ResidentRecipeEdge edge = edges[cell.edge_offset + k];
        if (edge.target_cell >= cell_count) continue;
        target += static_cast<std::int64_t>(edge.weight_q16) * read_from[edge.target_cell];
      }
      values_q16[i] = direct_relation_relaxed_step_q16(values_q16[i], target);
    }
  }
  return true;
}

}  // namespace substrate::direct_network

#undef DIRECT_SOLVER_HD

#endif  // HARDWARE_NATIVE_DIRECT_MATHEMATICAL_RELATION_SOLVER_CUH
