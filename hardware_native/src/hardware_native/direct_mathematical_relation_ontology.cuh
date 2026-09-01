#ifndef HARDWARE_NATIVE_DIRECT_MATHEMATICAL_RELATION_ONTOLOGY_CUH
#define HARDWARE_NATIVE_DIRECT_MATHEMATICAL_RELATION_ONTOLOGY_CUH

#include <cstdint>
#include <limits>
#include <type_traits>

#include "hardware_native/direct_network_brain.cuh"

namespace substrate::direct_network {

inline constexpr std::int32_t kDirectImplicitSelfCoefficientQ16 = -(1 << 16);

struct DirectImplicitRelationRowV1 {
  std::uint32_t variable_index;
  std::uint32_t term_offset;
  std::int64_t theta_q16;
  std::int32_t self_coefficient_q16;
  std::uint16_t term_count;
  std::uint16_t flags;
};
static_assert(std::is_standard_layout_v<DirectImplicitRelationRowV1> &&
              std::is_trivial_v<DirectImplicitRelationRowV1> &&
              std::has_unique_object_representations_v<DirectImplicitRelationRowV1>);

struct DirectImplicitRelationTermV1 {
  std::uint32_t target_variable;
  std::int32_t coefficient_q16;
  ResidentRecipeRelation relation;
  std::uint16_t flags;
  std::uint32_t field_index;
};
static_assert(std::is_standard_layout_v<DirectImplicitRelationTermV1> &&
              std::is_trivial_v<DirectImplicitRelationTermV1> &&
              std::has_unique_object_representations_v<DirectImplicitRelationTermV1>);

#if defined(__CUDACC__)
#define DIRECT_RELATION_HD __host__ __device__
#else
#define DIRECT_RELATION_HD
#endif

DIRECT_RELATION_HD inline std::int64_t direct_relation_saturating_add(
    std::int64_t left, std::int64_t right) {
  constexpr std::int64_t kMax = std::numeric_limits<std::int64_t>::max();
  constexpr std::int64_t kMin = std::numeric_limits<std::int64_t>::min();
  if (right > 0 && left > kMax - right) return kMax;
  if (right < 0 && left < kMin - right) return kMin;
  return left + right;
}

DIRECT_RELATION_HD inline bool direct_implicit_relation_row_v1(
    const ResidentRecipeCell* cells, std::uint32_t cell_count,
    std::uint32_t edge_count, std::uint32_t cell_index,
    DirectImplicitRelationRowV1* out) {
  if (cells == nullptr || out == nullptr || cell_index >= cell_count ||
      cell_count > 0xffffu)
    return false;
  const ResidentRecipeCell cell = cells[cell_index];
  if (cell.edge_offset > edge_count ||
      static_cast<std::uint32_t>(cell.edge_count) > edge_count - cell.edge_offset)
    return false;

  DirectImplicitRelationRowV1 row{};
  row.variable_index = cell_index;
  row.term_offset = cell.edge_offset;
  row.theta_q16 = direct_relation_saturating_add(cell.support_q16, cell.credit_q16);
  row.self_coefficient_q16 = kDirectImplicitSelfCoefficientQ16;
  row.term_count = cell.edge_count;
  row.flags = cell.flags;
  *out = row;
  return true;
}

DIRECT_RELATION_HD inline bool direct_implicit_relation_term_v1(
    const ResidentRecipeEdge* edges, std::uint32_t edge_count,
    std::uint32_t cell_count, const DirectImplicitRelationRowV1& row,
    std::uint32_t local_term_index, DirectImplicitRelationTermV1* out) {
  if (edges == nullptr || out == nullptr || row.variable_index >= cell_count ||
      local_term_index >= row.term_count || row.term_offset > edge_count ||
      static_cast<std::uint32_t>(row.term_count) > edge_count - row.term_offset)
    return false;
  const std::uint32_t edge_index = row.term_offset + local_term_index;
  const ResidentRecipeEdge edge = edges[edge_index];
  if (edge.source_cell != row.variable_index || edge.target_cell >= cell_count)
    return false;

  DirectImplicitRelationTermV1 term{};
  term.target_variable = edge.target_cell;
  term.coefficient_q16 = edge.weight_q16;
  term.relation = edge.relation;
  term.flags = edge.flags;
  term.field_index = edge.field_index;
  *out = term;
  return true;
}

#undef DIRECT_RELATION_HD

}  // namespace substrate::direct_network

#endif  // HARDWARE_NATIVE_DIRECT_MATHEMATICAL_RELATION_ONTOLOGY_CUH
