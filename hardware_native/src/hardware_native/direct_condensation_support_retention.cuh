#ifndef HARDWARE_NATIVE_DIRECT_CONDENSATION_SUPPORT_RETENTION_CUH
#define HARDWARE_NATIVE_DIRECT_CONDENSATION_SUPPORT_RETENTION_CUH

// e.condensation_support_retention (#1551): the complete condensation witness,
// not a host-authored keep set, names the lower derivations required to rebuild
// a condensed revision.  A bounded resident ledger reference-counts those
// supports and recycles an intermediate only after it becomes unreachable.

#include <cstdint>
#include <type_traits>

#include "hardware_native/direct_boundary_condensation.cuh"
#include "hardware_native/direct_resource_ecology_abi.cuh"

namespace substrate::direct_adult_core {

inline constexpr std::uint32_t kCondensationRetainedSupportCapacity = 64u;
inline constexpr std::uint32_t kCondensationIntermediateCapacity = 32u;

struct alignas(8) DirectRetainedCondensationSupport {
  std::uint64_t logical_recipe_id;
  std::uint64_t revision_identity;
  std::uint64_t witness_identity;
  std::uint32_t retainers;
  std::uint32_t reserved;
};

struct alignas(8) DirectCondensedIntermediateRecord {
  std::uint64_t intermediate_identity;
  std::uint64_t witness_identity;
  std::uint32_t active;
  std::uint32_t reserved;
};

struct alignas(8) DirectCondensationSupportRetentionState {
  DirectRetainedCondensationSupport supports[kCondensationRetainedSupportCapacity];
  DirectCondensedIntermediateRecord intermediates[kCondensationIntermediateCapacity];
  std::uint64_t retained_supports;
  std::uint64_t live_intermediates;
  std::uint64_t recycled_intermediates;
  std::uint64_t reclaimed_supports;
  std::uint64_t capacity_refusals;
  std::uint64_t invalid_refusals;
  std::uint64_t duplicate_refusals;
};
static_assert(std::is_trivial_v<DirectCondensationSupportRetentionState> &&
              std::is_standard_layout_v<DirectCondensationSupportRetentionState>);

#if defined(__CUDACC__)

__device__ inline DirectRetainedCondensationSupport* find_condensation_support(
    DirectCondensationSupportRetentionState* state, std::uint64_t logical_recipe_id,
    std::uint64_t revision_identity) {
  for (std::uint32_t i = 0u; i < kCondensationRetainedSupportCapacity; ++i) {
    auto* support = &state->supports[i];
    if (support->logical_recipe_id == logical_recipe_id &&
        support->revision_identity == revision_identity)
      return support;
  }
  return nullptr;
}

__device__ inline DirectCondensedIntermediateRecord* find_condensed_intermediate(
    DirectCondensationSupportRetentionState* state, std::uint64_t intermediate_identity) {
  for (std::uint32_t i = 0u; i < kCondensationIntermediateCapacity; ++i)
    if (state->intermediates[i].intermediate_identity == intermediate_identity)
      return &state->intermediates[i];
  return nullptr;
}

__device__ inline bool retain_condensation_support(
    DirectCondensationSupportRetentionState* state,
    substrate::direct_adult::DirectResourceEcologyState* ecology,
    const ResidentBoundaryCondensationWitness& witness, std::uint64_t intermediate_identity) {
  if (state == nullptr || ecology == nullptr || intermediate_identity == 0u ||
      !resident_boundary_condensation_witness_complete(witness)) {
    if (state != nullptr)
      ++state->invalid_refusals;
    return false;
  }
  if (find_condensed_intermediate(state, intermediate_identity) != nullptr) {
    ++state->duplicate_refusals;
    return false;
  }

  DirectCondensedIntermediateRecord* empty_intermediate = nullptr;
  std::uint32_t empty_supports = 0u;
  std::uint32_t new_supports = 0u;
  for (std::uint32_t i = 0u; i < kCondensationIntermediateCapacity; ++i)
    if (state->intermediates[i].intermediate_identity == 0u && empty_intermediate == nullptr)
      empty_intermediate = &state->intermediates[i];
  for (std::uint32_t i = 0u; i < kCondensationRetainedSupportCapacity; ++i)
    if (state->supports[i].logical_recipe_id == 0u)
      ++empty_supports;
  for (std::uint32_t i = 0u; i < witness.source_boundary.member_count; ++i) {
    const auto& member = witness.source_boundary.members[i];
    if (find_condensation_support(state, member.logical_recipe_id, member.revision_identity) ==
        nullptr)
      ++new_supports;
  }
  if (empty_intermediate == nullptr || empty_supports < new_supports) {
    ++state->capacity_refusals;
    return false;
  }

  const std::uint64_t new_records = static_cast<std::uint64_t>(new_supports) + 1u;
  if (!substrate::direct_adult::device_reserve_pool_units(
          ecology, substrate::direct_adult::DirectResourcePoolKind::derivation_record,
          new_records)) {
    ++state->capacity_refusals;
    return false;
  }
  if (!substrate::direct_adult::device_commit_pool_units(
          ecology, substrate::direct_adult::DirectResourcePoolKind::derivation_record,
          new_records)) {
    substrate::direct_adult::device_cancel_pool_reservation(
        ecology, substrate::direct_adult::DirectResourcePoolKind::derivation_record, new_records);
    ++state->capacity_refusals;
    return false;
  }

  for (std::uint32_t i = 0u; i < witness.source_boundary.member_count; ++i) {
    const auto& member = witness.source_boundary.members[i];
    auto* support =
        find_condensation_support(state, member.logical_recipe_id, member.revision_identity);
    if (support == nullptr) {
      for (std::uint32_t j = 0u; j < kCondensationRetainedSupportCapacity; ++j)
        if (state->supports[j].logical_recipe_id == 0u) {
          support = &state->supports[j];
          break;
        }
      support->logical_recipe_id = member.logical_recipe_id;
      support->revision_identity = member.revision_identity;
      support->witness_identity = witness.witness_identity;
      ++state->retained_supports;
    }
    ++support->retainers;
  }
  empty_intermediate->intermediate_identity = intermediate_identity;
  empty_intermediate->witness_identity = witness.witness_identity;
  empty_intermediate->active = 1u;
  ++state->live_intermediates;
  return true;
}

__device__ inline bool condensation_support_available_for_repair(
    DirectCondensationSupportRetentionState* state,
    const ResidentBoundaryCondensationWitness& witness) {
  if (state == nullptr || !resident_boundary_condensation_witness_complete(witness))
    return false;
  for (std::uint32_t i = 0u; i < witness.source_boundary.member_count; ++i) {
    const auto& member = witness.source_boundary.members[i];
    const auto* support =
        find_condensation_support(state, member.logical_recipe_id, member.revision_identity);
    if (support == nullptr || support->retainers == 0u)
      return false;
  }
  return true;
}

__device__ inline bool recycle_condensed_intermediate(
    DirectCondensationSupportRetentionState* state,
    substrate::direct_adult::DirectResourceEcologyState* ecology,
    const ResidentBoundaryCondensationWitness& witness, std::uint64_t intermediate_identity) {
  if (state == nullptr || ecology == nullptr ||
      !resident_boundary_condensation_witness_complete(witness)) {
    if (state != nullptr)
      ++state->invalid_refusals;
    return false;
  }
  auto* intermediate = find_condensed_intermediate(state, intermediate_identity);
  if (intermediate == nullptr || intermediate->active == 0u ||
      intermediate->witness_identity != witness.witness_identity) {
    ++state->duplicate_refusals;
    return false;
  }
  for (std::uint32_t i = 0u; i < witness.source_boundary.member_count; ++i) {
    const auto& member = witness.source_boundary.members[i];
    const auto* support =
        find_condensation_support(state, member.logical_recipe_id, member.revision_identity);
    if (support == nullptr || support->retainers == 0u) {
      ++state->invalid_refusals;
      return false;
    }
  }

  intermediate->active = 0u;
  --state->live_intermediates;
  ++state->recycled_intermediates;
  substrate::direct_adult::device_release_pool_units(
      ecology, substrate::direct_adult::DirectResourcePoolKind::derivation_record, 1u);
  for (std::uint32_t i = 0u; i < witness.source_boundary.member_count; ++i) {
    const auto& member = witness.source_boundary.members[i];
    auto* support =
        find_condensation_support(state, member.logical_recipe_id, member.revision_identity);
    if (support == nullptr || support->retainers == 0u)
      return false;
    --support->retainers;
    if (support->retainers == 0u) {
      *support = {};
      --state->retained_supports;
      ++state->reclaimed_supports;
      substrate::direct_adult::device_release_pool_units(
          ecology, substrate::direct_adult::DirectResourcePoolKind::derivation_record, 1u);
    }
  }
  return true;
}

#endif  // __CUDACC__

}  // namespace substrate::direct_adult_core

#endif  // HARDWARE_NATIVE_DIRECT_CONDENSATION_SUPPORT_RETENTION_CUH
