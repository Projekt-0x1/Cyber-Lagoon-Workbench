#ifndef HARDWARE_NATIVE_DIRECT_ADULT_ELIGIBILITY_RESOURCE_ECONOMICS_CUH
#define HARDWARE_NATIVE_DIRECT_ADULT_ELIGIBILITY_RESOURCE_ECONOMICS_CUH

// e.eligibility_resource_economics (#1550): residual error changes how much
// finite eligibility matter one causal context may retain.  The seven residual
// relations remain separate evidence; their magnitudes determine only bounded
// search breadth, horizon, and reinstatement pressure.  Allocation still passes
// through the organism's physical eligibility-record pool.

#include <cstdint>
#include <type_traits>

#include "hardware_native/direct_adult_eligibility_residual_meta.cuh"
#include "hardware_native/direct_resource_ecology_abi.cuh"

namespace substrate::direct_network {

inline constexpr std::uint32_t kEligibilityEconomyContextCapacity = 16u;
inline constexpr std::uint32_t kEligibilityEconomyMaxUnitsPerContext = 8u;
inline constexpr std::uint32_t kEligibilityEconomyMaxHorizonTicks = 16u;

enum class DirectEligibilityEconomyResult : std::uint32_t {
  applied = 0u,
  malformed = 1u,
  stale = 2u,
  context_capacity = 3u,
  matter_deferred = 4u,
};

struct alignas(8) DirectEligibilityContextEconomy {
  std::uint64_t context_identity;
  std::uint64_t last_evidence_identity;
  std::uint64_t last_epoch;
  std::uint64_t residual_magnitude_q16;
  std::uint32_t retained_units;
  std::uint32_t horizon_ticks;
  std::uint32_t reinstatement_q16;
  std::uint32_t observations;
};
static_assert(std::is_trivial_v<DirectEligibilityContextEconomy> &&
              std::is_standard_layout_v<DirectEligibilityContextEconomy>);

struct alignas(8) DirectEligibilityResourceEconomy {
  DirectEligibilityContextEconomy contexts[kEligibilityEconomyContextCapacity];
  std::uint64_t applied_updates;
  std::uint64_t expanded_units;
  std::uint64_t contracted_units;
  std::uint64_t deferred_units;
  std::uint64_t malformed_refusals;
  std::uint64_t stale_refusals;
  std::uint64_t context_capacity_refusals;
};
static_assert(std::is_trivial_v<DirectEligibilityResourceEconomy> &&
              std::is_standard_layout_v<DirectEligibilityResourceEconomy>);

#if defined(__CUDACC__)

__device__ inline std::uint64_t eligibility_economy_abs_q16(std::int32_t value) {
  return value < 0 ? static_cast<std::uint64_t>(-static_cast<std::int64_t>(value))
                   : static_cast<std::uint64_t>(value);
}

__device__ inline DirectEligibilityContextEconomy* eligibility_economy_find_or_empty(
    DirectEligibilityResourceEconomy* economy, std::uint64_t context_identity, bool* found) {
  DirectEligibilityContextEconomy* empty = nullptr;
  *found = false;
  for (std::uint32_t i = 0u; i < kEligibilityEconomyContextCapacity; ++i) {
    auto* entry = &economy->contexts[i];
    if (entry->context_identity == context_identity) {
      *found = true;
      return entry;
    }
    if (entry->context_identity == 0u && empty == nullptr)
      empty = entry;
  }
  return empty;
}

__device__ inline DirectEligibilityEconomyResult apply_eligibility_resource_economics(
    DirectEligibilityResourceEconomy* economy,
    substrate::direct_adult::DirectResourceEcologyState* ecology, std::uint64_t context_identity,
    std::uint64_t evidence_identity, std::uint64_t epoch,
    const DirectEligibilityResidualScore& score) {
  if (economy == nullptr || ecology == nullptr || context_identity == 0u ||
      evidence_identity == 0u || epoch == 0u) {
    if (economy != nullptr)
      ++economy->malformed_refusals;
    return DirectEligibilityEconomyResult::malformed;
  }

  bool found = false;
  DirectEligibilityContextEconomy* entry =
      eligibility_economy_find_or_empty(economy, context_identity, &found);
  if (entry == nullptr) {
    ++economy->context_capacity_refusals;
    return DirectEligibilityEconomyResult::context_capacity;
  }
  if (found && (epoch <= entry->last_epoch || evidence_identity == entry->last_evidence_identity)) {
    ++economy->stale_refusals;
    return DirectEligibilityEconomyResult::stale;
  }

  std::uint64_t magnitude_q16 = 0u;
  for (std::uint32_t d = 0u; d < kEligibilityResidualDimensions; ++d) {
    const std::uint64_t component = eligibility_economy_abs_q16(score.error.q16[d]);
    const std::uint64_t remaining = ~std::uint64_t{0} - magnitude_q16;
    magnitude_q16 += component > remaining ? remaining : component;
  }
  const std::uint64_t breadth = magnitude_q16 >> 16u;
  const std::uint32_t target_units =
      1u + static_cast<std::uint32_t>(breadth < kEligibilityEconomyMaxUnitsPerContext - 1u
                                          ? breadth
                                          : kEligibilityEconomyMaxUnitsPerContext - 1u);
  const std::uint32_t target_horizon =
      1u +
      static_cast<std::uint32_t>((magnitude_q16 >> 15u) < kEligibilityEconomyMaxHorizonTicks - 1u
                                     ? (magnitude_q16 >> 15u)
                                     : kEligibilityEconomyMaxHorizonTicks - 1u);
  const std::uint64_t mean_q16 = magnitude_q16 / kEligibilityResidualDimensions;
  const std::uint32_t reinstatement_q16 =
      static_cast<std::uint32_t>(mean_q16 < (1u << 16u) ? mean_q16 : (1u << 16u));
  const std::uint32_t old_units = found ? entry->retained_units : 0u;

  if (target_units > old_units) {
    const std::uint32_t delta = target_units - old_units;
    if (!substrate::direct_adult::device_reserve_pool_units(
            ecology, substrate::direct_adult::DirectResourcePoolKind::eligibility_record, delta)) {
      substrate::direct_adult::device_defer_pool_units(
          ecology, substrate::direct_adult::DirectResourcePoolKind::eligibility_record, delta);
      economy->deferred_units += delta;
      return DirectEligibilityEconomyResult::matter_deferred;
    }
    if (!substrate::direct_adult::device_commit_pool_units(
            ecology, substrate::direct_adult::DirectResourcePoolKind::eligibility_record, delta)) {
      substrate::direct_adult::device_cancel_pool_reservation(
          ecology, substrate::direct_adult::DirectResourcePoolKind::eligibility_record, delta);
      economy->deferred_units += delta;
      return DirectEligibilityEconomyResult::matter_deferred;
    }
    economy->expanded_units += delta;
  } else if (target_units < old_units) {
    const std::uint32_t delta = old_units - target_units;
    substrate::direct_adult::device_release_pool_units(
        ecology, substrate::direct_adult::DirectResourcePoolKind::eligibility_record, delta);
    economy->contracted_units += delta;
  }

  entry->context_identity = context_identity;
  entry->last_evidence_identity = evidence_identity;
  entry->last_epoch = epoch;
  entry->residual_magnitude_q16 = magnitude_q16;
  entry->retained_units = target_units;
  entry->horizon_ticks = target_horizon;
  entry->reinstatement_q16 = reinstatement_q16;
  ++entry->observations;
  ++economy->applied_updates;
  return DirectEligibilityEconomyResult::applied;
}

#endif  // __CUDACC__

}  // namespace substrate::direct_network

#endif  // HARDWARE_NATIVE_DIRECT_ADULT_ELIGIBILITY_RESOURCE_ECONOMICS_CUH
