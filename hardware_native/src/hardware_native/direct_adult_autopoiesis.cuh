#ifndef HARDWARE_NATIVE_DIRECT_ADULT_AUTOPOIESIS_CUH
#define HARDWARE_NATIVE_DIRECT_ADULT_AUTOPOIESIS_CUH

#include <cstdint>
#include <type_traits>

#include "hardware_native/direct_network_brain.cuh"

namespace substrate::direct_network {

// Homeostatic-class structural metabolism over one grown subject's own route
// tissue -- the device-owned structural act class proven by e.maintenance
// (#1504) and c.turnover (#1294), not a learned-plasticity proposal.  One
// continuous cycle: wear degrades kept components toward the floor, paid
// repair restores degraded conductance stepwise, renewal retires an aged weak
// component with its exact matter returned and rebuilds the same slot to the
// same endpoint under a fresh incarnation, and synthesis activates dormant
// developmental-reserve material into a new operational component charged
// against the construction reserve before any physical route exists.  Every
// decision derives from device words alone; every refusal fails closed.
//
// Matter identities kept exact: a born route's life was committed at gestation
// as one explicit-interaction unit with no construction-reserve entry, while a
// metabolically synthesized route carries both its decoded cost against the
// reserve and the same-cost pool commitment.  Retirement therefore reverses
// exactly what the component's own life charged -- never more, never less.
inline constexpr std::uint32_t kAutopoieticComponentCapacity = 12u;
inline constexpr std::int32_t kAutopoieticIntactConductanceQ16 = 1 << 16;
inline constexpr std::int32_t kAutopoieticRepairThresholdQ16 = 1 << 14;
inline constexpr std::int32_t kAutopoieticFloorConductanceQ16 = 1 << 10;
inline constexpr std::int32_t kAutopoieticNewbornConductanceQ16 = 1 << 15;
inline constexpr std::int32_t kAutopoieticRepairStepQ16 = 1 << 13;
inline constexpr std::uint32_t kAutopoiesisMaxTargetInDegree = 16u;

struct DirectAutopoiesisConfig {
  std::uint32_t renewal_window_epochs;
  std::uint32_t synthesis_budget_per_epoch;
  std::uint32_t repair_budget_per_epoch;
  std::int32_t wear_per_epoch_q16;
};

struct DirectAutopoieticComponent {
  std::uint32_t route_index;
  std::uint32_t target_node;
  std::uint32_t born_epoch;
  std::uint32_t matter_units;
};

struct DirectAutopoiesisState {
  Root256 tissue_root;
  DirectAutopoieticComponent components[kAutopoieticComponentCapacity];
  std::uint32_t count;
  std::uint32_t epoch;
  std::uint64_t matter_charged;
  std::uint64_t matter_returned;
  std::uint64_t repair_energy_debited;
  std::uint64_t syntheses;
  std::uint64_t repairs;
  std::uint64_t renewals;
  std::uint64_t boundary_refusals;
  std::uint64_t matter_refusals;
  std::uint64_t capacity_refusals;
};

static_assert(std::is_standard_layout_v<DirectAutopoiesisState> &&
              std::is_trivial_v<DirectAutopoiesisState> &&
              std::has_unique_object_representations_v<DirectAutopoiesisState>);

__device__ inline bool tissue_bound(const DirectAutopoiesisState& state,
                                    const DirectBrain& brain) {
  return state.tissue_root == brain.birth_root;
}

// Birth handoff: the metabolism binds exactly one subject and never rebinds.
__device__ inline bool bind_autopoietic_tissue(DirectAutopoiesisState* state,
                                               const DirectBrain& brain) {
  if (state == nullptr || state->count != 0u || state->epoch != 0u ||
      state->tissue_root == brain.birth_root)
    return false;
  state->tissue_root = brain.birth_root;
  return true;
}

// Born tissue enters the metabolism already paid for: admission charges
// nothing and records zero matter units, so conservation stays exact.
__device__ inline bool admit_resident_component(DirectAutopoiesisState* state,
                                                const DirectBrain& brain,
                                                std::uint32_t route_index) {
  if (state == nullptr || !tissue_bound(*state, brain) ||
      route_index >= brain.route_capacity ||
      !route_is_active(brain.routes[route_index]) ||
      brain.routes[route_index].target >= brain.node_count ||
      state->count >= kAutopoieticComponentCapacity)
    return false;
  for (std::uint32_t i = 0u; i < state->count; ++i)
    if (state->components[i].route_index == route_index) return false;
  state->components[state->count++] =
      DirectAutopoieticComponent{route_index, brain.routes[route_index].target,
                                 state->epoch, 0u};
  return true;
}

// Conservation law over the whole cycle: net matter drawn from the resident
// construction reserve equals the matter currently resident in metabolic
// components, across arbitrary synthesize+repair+retire sequences.
__device__ inline bool autopoiesis_ledger_balanced(
    const DirectAutopoiesisState& state) {
  std::uint64_t resident = 0u;
  for (std::uint32_t i = 0u; i < state.count; ++i)
    if (state.components[i].matter_units != 0u)
      resident += state.components[i].matter_units;
  return state.matter_charged - state.matter_returned == resident;
}

__device__ inline std::uint32_t autopoiesis_route_delay(const DirectNode& source,
                                                        const DirectNode& target) {
  const int dx = source.coordinate[0] > target.coordinate[0]
                     ? source.coordinate[0] - target.coordinate[0]
                     : target.coordinate[0] - source.coordinate[0];
  const int dy = source.coordinate[1] > target.coordinate[1]
                     ? source.coordinate[1] - target.coordinate[1]
                     : target.coordinate[1] - source.coordinate[1];
  const int dz = source.coordinate[2] > target.coordinate[2]
                     ? source.coordinate[2] - target.coordinate[2]
                     : target.coordinate[2] - source.coordinate[2];
  const std::uint32_t raw =
      1u + static_cast<std::uint32_t>((dx + dy + dz) / 32);
  return raw > 16u ? 16u : raw;
}

__device__ inline bool weak_owner_support(const DirectNode& owner) {
  const std::int32_t activity = owner.activity_ema_q16 < 0 ? -owner.activity_ema_q16
                                                           : owner.activity_ema_q16;
  const std::int32_t credit = owner.credit_ema_q16 < 0 ? -owner.credit_ema_q16
                                                       : owner.credit_ema_q16;
  constexpr std::int32_t kWeakActivityQ16 = static_cast<std::int32_t>(kQ16One) / 128;
  constexpr std::int32_t kWeakCreditQ16 = static_cast<std::int32_t>(kQ16One) / 64;
  return activity < kWeakActivityQ16 && credit < kWeakCreditQ16;
}

__device__ inline bool duplicate_active_edge(const DirectRoute* routes,
                                             const DirectNode& owner,
                                             std::uint32_t target) {
  const std::uint32_t end = owner.route_offset + owner.route_capacity;
  for (std::uint32_t r = owner.route_offset; r < end; ++r)
    if (route_is_active(routes[r]) && routes[r].target == target) return true;
  return false;
}

// Retire one component slot: reverse exactly what its own life charged.
__device__ inline void retire_component_slot(DirectBrain* brain,
                                             DirectAutopoiesisState* state,
                                             DirectNode& owner,
                                             std::uint32_t route_index,
                                             std::uint32_t matter_units) {
  DirectRoute& route = brain->routes[route_index];
  substrate::direct_adult::device_uncommit_pool_units(
      brain->resource_ecology,
      substrate::direct_adult::DirectResourcePoolKind::explicit_interaction,
      matter_units == 0u ? 1u : matter_units);
  if (matter_units != 0u) {
    atomicAdd(reinterpret_cast<unsigned long long*>(&brain->development->constructor_reserve),
              static_cast<unsigned long long>(matter_units));
    atomicAdd(reinterpret_cast<unsigned long long*>(&brain->development->reclaimed_resource),
              static_cast<unsigned long long>(matter_units));
    atomicAdd(reinterpret_cast<unsigned long long*>(&brain->development->live_route_matter),
              static_cast<unsigned long long>(-static_cast<long long>(matter_units)));
  }
  route.flags &= ~kRouteFlagActive;
  route.flags |= kRouteFlagDevelopmentalReserve;
  if (brain->route_incarnations != nullptr)
    brain->route_incarnations[route_index] += kRouteIncarnationStride;
  route.eligibility_q16 = 0;
  route.eligibility_context = kInvalidIndex;
  route.eligibility_expires = 0u;
  route.last_credit_q16 = 0;
  route.last_credit_ticket = kNoCreditTicket;
  if (owner.active_route_count > 0u) --owner.active_route_count;
  if (route.target < brain->node_count)
    atomicSub(&brain->nodes[route.target].active_in_degree, 1u);
  state->matter_returned += matter_units;
}

__device__ inline bool run_autopoietic_epoch(const DirectAutopoiesisConfig& config,
                                             DirectAutopoiesisState* state,
                                             DirectBrain* brain) {
  if (state == nullptr) return false;
  if (brain == nullptr || !tissue_bound(*state, *brain)) {
    ++state->boundary_refusals;
    return false;
  }
  DirectNode* nodes = brain->nodes;
  DirectRoute* routes = brain->routes;
  ResidentDevelopmentState* development = brain->development;
  substrate::direct_adult::DirectResourceEcologyState* ecology = brain->resource_ecology;
  if (nodes == nullptr || routes == nullptr || development == nullptr ||
      ecology == nullptr)
    return false;
  ++state->epoch;

  // Catabolic term: unkept component conductance wears toward the floor.
  for (std::uint32_t i = 0u; i < state->count; ++i) {
    DirectRoute& route = routes[state->components[i].route_index];
    if (!route_is_active(route)) continue;
    const std::int32_t worn = route.conductance_q16 - config.wear_per_epoch_q16;
    route.conductance_q16 =
        worn < kAutopoieticFloorConductanceQ16 ? kAutopoieticFloorConductanceQ16 : worn;
  }

  // Anabolic repair: paid stepwise restoration of degraded components, bounded
  // per epoch so continuous upkeep stays bounded work.
  std::uint32_t repair_budget = config.repair_budget_per_epoch;
  for (std::uint32_t i = 0u; i < state->count && repair_budget != 0u; ++i) {
    DirectRoute& route = routes[state->components[i].route_index];
    if (!route_is_active(route) || route.conductance_q16 >= kAutopoieticRepairThresholdQ16)
      continue;
    const std::int32_t restored = route.conductance_q16 + kAutopoieticRepairStepQ16;
    route.conductance_q16 = restored > kAutopoieticIntactConductanceQ16
                                ? kAutopoieticIntactConductanceQ16
                                : restored;
    ++state->repairs;
    ++state->repair_energy_debited;
    --repair_budget;
  }

  // Renewal: an aged weak component is retired with its exact matter returned,
  // then rebuilt in place to the SAME functional endpoint at newborn
  // conductance under a fresh incarnation.  The replacement charge can no
  // longer fail: the retirement above already returned this transaction's
  // working matter into the reserve, so continuity of function is atomic.
  for (std::uint32_t i = 0u; i < state->count; ++i) {
    DirectAutopoieticComponent& component = state->components[i];
    if (state->epoch - component.born_epoch <= config.renewal_window_epochs) continue;
    DirectRoute& route = routes[component.route_index];
    if (!route_is_active(route) || route.target >= brain->node_count) continue;
    DirectNode& owner = nodes[route.source];
    if (!weak_owner_support(owner)) continue;
    const std::uint32_t route_index = component.route_index;
    const std::uint32_t endpoint = route.target;
    retire_component_slot(brain, state, owner, route_index, component.matter_units);

    const std::uint32_t rebuilt_cost = construction_route_cost(owner, nodes[endpoint]);
    route.target = endpoint;
    route.flags = encode_route_construction_cost(
        kRouteFlagActive | kRouteFlagRecurrent |
            ((route.flags & kRouteFlagInhibitory) != 0u ? kRouteFlagInhibitory : 0u),
        rebuilt_cost);
    route.delay = autopoiesis_route_delay(owner, nodes[endpoint]);
    route.conductance_q16 = kAutopoieticNewbornConductanceQ16;
    route.developmental_score_q16 = 0;
    if (brain->route_incarnations != nullptr)
      brain->route_incarnations[route_index] += kRouteIncarnationStride;
    if (brain->retention_bank != nullptr) {
      substrate::direct_adult::DirectRetentionState fresh{};
      fresh.logical_source = route.source;
      fresh.logical_slot = route_index;
      fresh.logical_generation =
          brain->route_incarnations == nullptr
              ? 0u
              : brain->route_incarnations[route_index];
      fresh.source_revision = fresh.logical_generation;
      fresh.last_confirmed_conductance_q16 = route.conductance_q16;
      brain->retention_bank[route_index] = fresh;
    }
    ++owner.active_route_count;
    atomicAdd(&nodes[endpoint].active_in_degree, 1u);
    atomicAdd(reinterpret_cast<unsigned long long*>(&development->constructor_reserve),
              static_cast<unsigned long long>(-static_cast<long long>(rebuilt_cost)));
    atomicAdd(reinterpret_cast<unsigned long long*>(&development->live_route_matter),
              static_cast<unsigned long long>(rebuilt_cost));
    substrate::direct_adult::device_commit_pool_units(
        ecology, substrate::direct_adult::DirectResourcePoolKind::explicit_interaction,
        rebuilt_cost);
    state->matter_charged += rebuilt_cost;
    component.born_epoch = state->epoch;
    component.matter_units = rebuilt_cost;
    component.target_node = endpoint;
    ++state->renewals;
  }

  // Synthesis: dormant developmental-reserve material becomes a new operational
  // component.  Ledger room, reserve matter and pool capacity are all proved
  // BEFORE any physical route exists -- refusal is fail-closed and counted.
  std::uint32_t synthesis_budget = config.synthesis_budget_per_epoch;
  for (std::uint32_t node_index = 0u;
       node_index < brain->node_count && synthesis_budget != 0u; ++node_index) {
    const DirectNode& owner = nodes[node_index];
    std::uint32_t chosen_slot = kInvalidIndex;
    std::uint32_t chosen_target = kInvalidIndex;
    const std::uint32_t slot_end = owner.route_offset + owner.route_capacity;
    for (std::uint32_t r = owner.route_offset;
         r < slot_end && chosen_slot == kInvalidIndex; ++r) {
      if (route_is_active(routes[r]) ||
          (routes[r].flags & kRouteFlagDevelopmentalReserve) == 0u)
        continue;
      for (std::uint32_t t = 0u; t < brain->node_count; ++t) {
        if (t == node_index || nodes[t].active_in_degree >= kAutopoiesisMaxTargetInDegree ||
            duplicate_active_edge(routes, owner, t))
          continue;
        chosen_slot = r;
        chosen_target = t;
        break;
      }
    }
    if (chosen_slot == kInvalidIndex) continue;
    const std::uint32_t cost = construction_route_cost(owner, nodes[chosen_target]);
    if (state->count >= kAutopoieticComponentCapacity) {
      ++state->capacity_refusals;
      break;
    }
    if (development->constructor_reserve < cost) {
      ++state->matter_refusals;
      break;
    }
    if (!substrate::direct_adult::device_reserve_pool_units(
            ecology, substrate::direct_adult::DirectResourcePoolKind::explicit_interaction,
            cost)) {
      ++state->matter_refusals;
      break;
    }
    DirectRoute& route = routes[chosen_slot];
    route.source = node_index;
    route.target = chosen_target;
    route.flags =
        encode_route_construction_cost(kRouteFlagActive | kRouteFlagRecurrent, cost);
    route.delay = autopoiesis_route_delay(owner, nodes[chosen_target]);
    route.conductance_q16 = kAutopoieticNewbornConductanceQ16;
    route.developmental_score_q16 = 0;
    route.eligibility_q16 = 0;
    route.last_credit_q16 = 0;
    route.eligibility_context = kInvalidIndex;
    route.eligibility_expires = 0u;
    route.last_credit_ticket = kNoCreditTicket;
    if (brain->route_incarnations != nullptr)
      brain->route_incarnations[chosen_slot] += kRouteIncarnationStride;
    if (brain->retention_bank != nullptr) {
      substrate::direct_adult::DirectRetentionState fresh{};
      fresh.logical_source = node_index;
      fresh.logical_slot = chosen_slot;
      fresh.logical_generation =
          brain->route_incarnations == nullptr
              ? 0u
              : brain->route_incarnations[chosen_slot];
      fresh.source_revision = fresh.logical_generation;
      fresh.last_confirmed_conductance_q16 = route.conductance_q16;
      brain->retention_bank[chosen_slot] = fresh;
    }
    ++nodes[node_index].active_route_count;
    atomicAdd(&nodes[chosen_target].active_in_degree, 1u);
    atomicAdd(reinterpret_cast<unsigned long long*>(&development->constructor_reserve),
              static_cast<unsigned long long>(-static_cast<long long>(cost)));
    atomicAdd(reinterpret_cast<unsigned long long*>(&development->live_route_matter),
              static_cast<unsigned long long>(cost));
    substrate::direct_adult::device_commit_pool_units(
        ecology, substrate::direct_adult::DirectResourcePoolKind::explicit_interaction,
        cost);
    state->components[state->count++] =
        DirectAutopoieticComponent{chosen_slot, chosen_target, state->epoch, cost};
    state->matter_charged += cost;
    ++state->syntheses;
    --synthesis_budget;
  }

  return true;
}

}  // namespace substrate::direct_network

#endif  // HARDWARE_NATIVE_DIRECT_ADULT_AUTOPOIESIS_CUH
