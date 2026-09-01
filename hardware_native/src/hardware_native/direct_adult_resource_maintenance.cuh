#ifndef HARDWARE_NATIVE_DIRECT_ADULT_RESOURCE_MAINTENANCE_CUH
#define HARDWARE_NATIVE_DIRECT_ADULT_RESOURCE_MAINTENANCE_CUH

#include "hardware_native/direct_adult_core.cuh"
#include "hardware_native/direct_active_frontier_dispatch.cuh"
#include "hardware_native/direct_resource_ecology.cuh"
namespace substrate::direct_adult {

__device__ inline bool device_explicit_interaction_pressure(DirectResourceEcologyState* ecology) {
  DirectResourcePoolState* pool =
      direct_ecology_pool(ecology, DirectResourcePoolKind::explicit_interaction);
  if (pool == nullptr || pool->capacity_units == 0u || pool->charged_units > pool->capacity_units)
    return false;
  const std::uint64_t available =
      (pool->capacity_units - pool->charged_units) + pool->reserved_units;
  return available < pool->capacity_units / 10u + 1u;
}

// The reserve word is the concurrent permit: only the prefix needed to restore
// explicit-interaction headroom can retire live routes.
__device__ inline bool device_uncommit_explicit_interaction_under_pressure(
    DirectResourceEcologyState* ecology, std::uint64_t units) {
  DirectResourcePoolState* pool =
      direct_ecology_pool(ecology, DirectResourcePoolKind::explicit_interaction);
  if (pool == nullptr || units == 0u || pool->capacity_units == 0u ||
      pool->charged_units > pool->capacity_units)
    return false;
  const unsigned long long previous = direct_ecology_atomic_add_u64(&pool->reserved_units, units);
  const std::uint64_t base_available = pool->capacity_units - pool->charged_units;
  const std::uint64_t required_available = pool->capacity_units / 10u + 1u;
  if (previous > pool->charged_units || units > pool->charged_units - previous ||
      base_available + previous >= required_available ||
      units > required_available - (base_available + previous)) {
    direct_ecology_atomic_sub_u64(&pool->reserved_units, units);
    return false;
  }
  direct_ecology_atomic_sub_u64(&pool->live_units, units);
  direct_ecology_atomic_add_u64(&pool->compaction_units, units);
  return true;
}

}  // namespace substrate::direct_adult

namespace substrate::direct_adult_core {

__device__ inline std::int32_t maintenance_abs_q16(std::int32_t value) {
  return value < 0 ? -value : value;
}

// One owner thread per node; both executors call this exact structural law.
__device__ inline void maintain_adult_node_routes(
    DirectNode* nodes, DirectRoute* routes, std::uint64_t* route_incarnations,
    const std::uint64_t* route_opportunity_incarnations, std::uint32_t node_index,
    std::uint32_t node_count, ResidentDevelopmentState* development,
    substrate::direct_adult::DirectResourceEcologyState* ecology, std::int32_t cost_per_route_q16,
    const std::uint32_t* route_causal_pin_bits, AdultCoreMetrics* metrics) {
  if (node_index >= node_count || development == nullptr || ecology == nullptr ||
      metrics == nullptr)
    return;
  DirectNode& node = nodes[node_index];
  const bool pressure = substrate::direct_adult::device_explicit_interaction_pressure(ecology);
  const bool weak_retention = maintenance_abs_q16(node.credit_ema_q16) < (kQ16One / 64) &&
                              maintenance_abs_q16(node.activity_ema_q16) < (kQ16One / 128);
  const std::uint32_t route_end = node.route_offset + node.route_capacity;
  for (std::uint32_t route_index = node.route_offset; route_index < route_end; ++route_index) {
    DirectRoute& route = routes[route_index];
    if ((route.flags & direct_network::kRouteFlagActive) == 0u)
      continue;
    if (cost_per_route_q16 <= 0 ||
        !substrate::direct_adult::device_record_active_frontier_cost(
            ecology, metrics, 1u, static_cast<std::uint32_t>(cost_per_route_q16)))
      continue;
    if (route_incarnations == nullptr || route_opportunity_incarnations == nullptr ||
        route_opportunity_incarnations[route_index] != route_incarnations[route_index] ||
        !pressure || !weak_retention)
      continue;
    if (route_causal_pin_bits != nullptr &&
        ((route_causal_pin_bits[route_index >> 5] >> (route_index & 31u)) & 1u) != 0u) {
      substrate::direct_adult::device_defer_pool_units(
          ecology, substrate::direct_adult::DirectResourcePoolKind::explicit_interaction, 1u);
      atomicAdd(reinterpret_cast<unsigned long long*>(&metrics->retractions_deferred_causal_pin),
                1ULL);
      continue;
    }
    if (route.conductance_q16 > kMinConductanceQ16) {
      const std::int32_t decayed = route.conductance_q16 - (cost_per_route_q16 * 2);
      route.conductance_q16 = decayed < kMinConductanceQ16 ? kMinConductanceQ16 : decayed;
      continue;
    }
    if (!substrate::direct_adult::device_uncommit_explicit_interaction_under_pressure(ecology, 1u))
      continue;
    route.flags &= ~direct_network::kRouteFlagActive;
    if (route_incarnations != nullptr)
      route_incarnations[route_index] += direct_network::kRouteIncarnationStride;
    if (route.target < node_count)
      atomicSub(&nodes[route.target].active_in_degree, 1u);
    if (node.active_route_count > 0u)
      --node.active_route_count;
    atomicAdd(reinterpret_cast<unsigned long long*>(&development->constructor_reserve), 1ULL);
    atomicAdd(reinterpret_cast<unsigned long long*>(&development->reclaimed_resource), 1ULL);
    atomicAdd(reinterpret_cast<unsigned long long*>(&development->live_route_matter),
              static_cast<unsigned long long>(-1ll));
    atomicAdd(reinterpret_cast<unsigned long long*>(&metrics->routes_retracted_scarcity), 1ULL);
  }
}

}  // namespace substrate::direct_adult_core

#endif  // HARDWARE_NATIVE_DIRECT_ADULT_RESOURCE_MAINTENANCE_CUH
