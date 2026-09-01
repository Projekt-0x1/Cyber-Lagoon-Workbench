#ifndef HARDWARE_NATIVE_DIRECT_RESIDENT_STRUCTURAL_ROLE_PROJECTION_CUH
#define HARDWARE_NATIVE_DIRECT_RESIDENT_STRUCTURAL_ROLE_PROJECTION_CUH

#include <cstdint>
#include <type_traits>

#include "hardware_native/direct_adult_core.cuh"

namespace substrate::direct_adult_core {

using direct_network::exact_history_fold_word;
using direct_network::kResidentDerivationWidth;
using direct_network::resident_recipe_ports_compatible;
#include "hardware_native/direct_adult_resident_relational_network.cuh"

#if defined(__CUDACC__)
#define DIRECT_STRUCTURAL_ROLE_HD __host__ __device__
#else
#define DIRECT_STRUCTURAL_ROLE_HD
#endif

struct ResidentStructuralRoleEntry {
  std::uint64_t occurrence_identity;
  std::uint64_t role_identity;
};

struct ResidentStructuralRoleProjection {
  std::uint64_t topology_identity;
  std::uint16_t role_count;
  std::uint16_t coupling_count;
  std::uint32_t reserved;
  ResidentStructuralRoleEntry roles[kResidentRelationalNetworkMaxOccurrences];
};

static_assert(std::is_standard_layout_v<ResidentStructuralRoleEntry> &&
              std::is_trivial_v<ResidentStructuralRoleEntry> &&
              std::has_unique_object_representations_v<ResidentStructuralRoleEntry>);
static_assert(std::is_standard_layout_v<ResidentStructuralRoleProjection> &&
              std::is_trivial_v<ResidentStructuralRoleProjection> &&
              std::has_unique_object_representations_v<ResidentStructuralRoleProjection>);

namespace structural_role_detail {

struct IncidentRow {
  std::uint64_t neighbor_logical_recipe_id;
  std::uint64_t neighbor_derivation_rank;
  std::uint64_t neighbor_morphology_identity;
  std::uint16_t local_port;
  std::uint16_t remote_port;
  std::uint16_t orientation;
  std::uint16_t reserved;
};

struct BoundaryRow {
  std::uint16_t formal_port;
  std::uint16_t domain;
  std::uint16_t direction;
  std::uint16_t arity;
};

DIRECT_STRUCTURAL_ROLE_HD inline bool incident_less(
    const IncidentRow& left, const IncidentRow& right) {
  if (left.neighbor_logical_recipe_id != right.neighbor_logical_recipe_id)
    return left.neighbor_logical_recipe_id < right.neighbor_logical_recipe_id;
  if (left.neighbor_derivation_rank != right.neighbor_derivation_rank)
    return left.neighbor_derivation_rank < right.neighbor_derivation_rank;
  if (left.neighbor_morphology_identity != right.neighbor_morphology_identity)
    return left.neighbor_morphology_identity < right.neighbor_morphology_identity;
  if (left.local_port != right.local_port) return left.local_port < right.local_port;
  if (left.remote_port != right.remote_port) return left.remote_port < right.remote_port;
  return left.orientation < right.orientation;
}

DIRECT_STRUCTURAL_ROLE_HD inline bool boundary_less(
    const BoundaryRow& left, const BoundaryRow& right) {
  if (left.formal_port != right.formal_port) return left.formal_port < right.formal_port;
  if (left.domain != right.domain) return left.domain < right.domain;
  if (left.direction != right.direction) return left.direction < right.direction;
  return left.arity < right.arity;
}

DIRECT_STRUCTURAL_ROLE_HD inline const ResidentRelationalNetworkMember*
member_for_occurrence(const ResidentRelationalNetworkClosure& closure,
                      std::uint64_t occurrence_identity) {
  if (occurrence_identity == 0u) return nullptr;
  const ResidentRelationalNetworkMember* match = nullptr;
  for (std::uint16_t i = 0u; i < closure.occurrence_count; ++i) {
    if (closure.members[i].occurrence_identity != occurrence_identity) continue;
    if (match != nullptr) return nullptr;
    match = &closure.members[i];
  }
  return match;
}

DIRECT_STRUCTURAL_ROLE_HD inline bool projection_shape_valid(
    const ResidentRelationalNetworkClosure& closure) {
  if (closure.identity == 0u || closure.occurrence_count < 2u ||
      closure.occurrence_count > kResidentRelationalNetworkMaxOccurrences ||
      closure.coupling_count < closure.occurrence_count - 1u ||
      closure.coupling_count > kResidentRelationalNetworkMaxCouplings ||
      closure.boundary_count > kResidentRelationalNetworkMaxBoundary)
    return false;
  std::uint8_t adjacency[kResidentRelationalNetworkMaxOccurrences]{};
  for (std::uint16_t i = 0u; i < closure.occurrence_count; ++i) {
    const auto& member = closure.members[i];
    if (member.occurrence_identity == 0u || member.logical_recipe_id == 0u ||
        member.derivation_rank == 0u || member.morphology_identity == 0u ||
        member.binding_count == 0u ||
        member.binding_count > direct_network::kResidentDerivationWidth)
      return false;
    for (std::uint16_t prior = 0u; prior < i; ++prior)
      if (closure.members[prior].occurrence_identity == member.occurrence_identity)
        return false;
  }
  for (std::uint16_t edge = 0u; edge < closure.coupling_count; ++edge) {
    const auto& coupling = closure.couplings[edge];
    const auto* source = member_for_occurrence(closure, coupling.source_occurrence_identity);
    const auto* target = member_for_occurrence(closure, coupling.target_occurrence_identity);
    if (source == nullptr || target == nullptr || source == target ||
        source->revision_identity != coupling.source_revision_identity ||
        target->revision_identity != coupling.target_revision_identity ||
        source->derivation_rank != coupling.source_derivation_rank ||
        target->derivation_rank != coupling.target_derivation_rank ||
        coupling.source_port_index >= source->binding_count ||
        coupling.target_port_index >= target->binding_count)
      return false;
    std::uint16_t source_index = closure.occurrence_count;
    std::uint16_t target_index = closure.occurrence_count;
    for (std::uint16_t i = 0u; i < closure.occurrence_count; ++i) {
      if (&closure.members[i] == source) source_index = i;
      if (&closure.members[i] == target) target_index = i;
    }
    if (source_index >= closure.occurrence_count || target_index >= closure.occurrence_count)
      return false;
    adjacency[source_index] |= static_cast<std::uint8_t>(1u << target_index);
    adjacency[target_index] |= static_cast<std::uint8_t>(1u << source_index);
    for (std::uint16_t prior = 0u; prior < edge; ++prior) {
      const auto& other = closure.couplings[prior];
      if (other.source_occurrence_identity == coupling.source_occurrence_identity &&
          other.target_occurrence_identity == coupling.target_occurrence_identity &&
          other.source_port_index == coupling.source_port_index &&
          other.target_port_index == coupling.target_port_index)
        return false;
    }
  }
  std::uint8_t reached = 1u;
  for (std::uint16_t pass = 0u; pass < closure.occurrence_count; ++pass)
    for (std::uint16_t member = 0u; member < closure.occurrence_count; ++member)
      if ((reached & static_cast<std::uint8_t>(1u << member)) != 0u)
        reached |= adjacency[member];
  if (reached != static_cast<std::uint8_t>((1u << closure.occurrence_count) - 1u))
    return false;
  for (std::uint16_t b = 0u; b < closure.boundary_count; ++b) {
    const auto& boundary = closure.boundary[b];
    const auto* member = member_for_occurrence(closure, boundary.occurrence_identity);
    if (member == nullptr || boundary.formal_port_index >= member->binding_count ||
        boundary.arity == 0u)
      return false;
  }
  return true;
}

}  // namespace structural_role_detail

DIRECT_STRUCTURAL_ROLE_HD inline bool project_resident_structural_roles(
    const ResidentRelationalNetworkClosure& closure,
    ResidentStructuralRoleProjection* out) {
  using direct_network::exact_history_fold_word;
  using structural_role_detail::BoundaryRow;
  using structural_role_detail::IncidentRow;
  if (out == nullptr || !structural_role_detail::projection_shape_valid(closure))
    return false;
  const std::uint64_t topology = resident_relational_network_recruitment_identity(closure);
  if (topology == 0u) return false;

  ResidentStructuralRoleProjection candidate{};
  candidate.topology_identity = topology;
  candidate.role_count = closure.occurrence_count;
  candidate.coupling_count = closure.coupling_count;

  for (std::uint16_t i = 0u; i < closure.occurrence_count; ++i) {
    const auto& member = closure.members[i];
    IncidentRow incidents[kResidentRelationalNetworkMaxCouplings]{};
    std::uint16_t incident_count = 0u;
    for (std::uint16_t edge = 0u; edge < closure.coupling_count; ++edge) {
      const auto& coupling = closure.couplings[edge];
      const ResidentRelationalNetworkMember* neighbor = nullptr;
      IncidentRow row{};
      if (coupling.source_occurrence_identity == member.occurrence_identity) {
        neighbor = structural_role_detail::member_for_occurrence(
            closure, coupling.target_occurrence_identity);
        row.local_port = coupling.source_port_index;
        row.remote_port = coupling.target_port_index;
        row.orientation = 1u;
      } else if (coupling.target_occurrence_identity == member.occurrence_identity) {
        neighbor = structural_role_detail::member_for_occurrence(
            closure, coupling.source_occurrence_identity);
        row.local_port = coupling.target_port_index;
        row.remote_port = coupling.source_port_index;
        row.orientation = 2u;
      } else {
        continue;
      }
      if (neighbor == nullptr || incident_count >= kResidentRelationalNetworkMaxCouplings)
        return false;
      row.neighbor_logical_recipe_id = neighbor->logical_recipe_id;
      row.neighbor_derivation_rank = neighbor->derivation_rank;
      row.neighbor_morphology_identity = neighbor->morphology_identity;
      incidents[incident_count++] = row;
    }
    if (incident_count == 0u) return false;
    for (std::uint16_t j = 1u; j < incident_count; ++j) {
      const IncidentRow value = incidents[j];
      std::uint16_t k = j;
      while (k != 0u && structural_role_detail::incident_less(value, incidents[k - 1u])) {
        incidents[k] = incidents[k - 1u];
        --k;
      }
      incidents[k] = value;
    }

    BoundaryRow boundaries[kResidentRelationalNetworkMaxBoundary]{};
    std::uint16_t boundary_count = 0u;
    for (std::uint16_t b = 0u; b < closure.boundary_count; ++b) {
      const auto& boundary = closure.boundary[b];
      if (boundary.occurrence_identity != member.occurrence_identity) continue;
      if (boundary_count >= kResidentRelationalNetworkMaxBoundary) return false;
      boundaries[boundary_count++] = BoundaryRow{
          boundary.formal_port_index,
          static_cast<std::uint16_t>(boundary.domain),
          static_cast<std::uint16_t>(boundary.direction), boundary.arity};
    }
    for (std::uint16_t j = 1u; j < boundary_count; ++j) {
      const BoundaryRow value = boundaries[j];
      std::uint16_t k = j;
      while (k != 0u && structural_role_detail::boundary_less(value, boundaries[k - 1u])) {
        boundaries[k] = boundaries[k - 1u];
        --k;
      }
      boundaries[k] = value;
    }

    std::uint64_t role = exact_history_fold_word(
        0x7374727563726f6cull, member.logical_recipe_id);
    role = exact_history_fold_word(role, member.derivation_rank);
    role = exact_history_fold_word(role, member.morphology_identity);
    role = exact_history_fold_word(role, member.binding_count);
    role = exact_history_fold_word(role, incident_count);
    role = exact_history_fold_word(role, boundary_count);
    for (std::uint16_t j = 0u; j < incident_count; ++j) {
      role = exact_history_fold_word(role, incidents[j].neighbor_logical_recipe_id);
      role = exact_history_fold_word(role, incidents[j].neighbor_derivation_rank);
      role = exact_history_fold_word(role, incidents[j].neighbor_morphology_identity);
      role = exact_history_fold_word(role, incidents[j].local_port);
      role = exact_history_fold_word(role, incidents[j].remote_port);
      role = exact_history_fold_word(role, incidents[j].orientation);
    }
    for (std::uint16_t j = 0u; j < boundary_count; ++j) {
      role = exact_history_fold_word(role, boundaries[j].formal_port);
      role = exact_history_fold_word(role, boundaries[j].domain);
      role = exact_history_fold_word(role, boundaries[j].direction);
      role = exact_history_fold_word(role, boundaries[j].arity);
    }
    if (role == 0u) role = 1u;
    for (std::uint16_t prior = 0u; prior < i; ++prior)
      if (candidate.roles[prior].role_identity == role) return false;
    candidate.roles[i] = ResidentStructuralRoleEntry{member.occurrence_identity, role};
  }

  for (std::uint16_t i = 1u; i < candidate.role_count; ++i) {
    const ResidentStructuralRoleEntry value = candidate.roles[i];
    std::uint16_t j = i;
    while (j != 0u && candidate.roles[j - 1u].role_identity > value.role_identity) {
      candidate.roles[j] = candidate.roles[j - 1u];
      --j;
    }
    candidate.roles[j] = value;
  }
  *out = candidate;
  return true;
}

#undef DIRECT_STRUCTURAL_ROLE_HD

}  // namespace substrate::direct_adult_core

#endif  // HARDWARE_NATIVE_DIRECT_RESIDENT_STRUCTURAL_ROLE_PROJECTION_CUH
