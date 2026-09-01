#ifndef HARDWARE_NATIVE_DIRECT_ADULT_AUTOPOIESIS_PREDICTIVE_CUH
#define HARDWARE_NATIVE_DIRECT_ADULT_AUTOPOIESIS_PREDICTIVE_CUH

#include <cstdint>
#include <type_traits>

#include "hardware_native/direct_adult_autopoiesis.cuh"

namespace substrate::direct_network {

// Predictively-steered autopoietic metabolism over one grown subject's own
// route tissue -- j.autopoiesis_predictive (#1542) closing the loop between
// the e.autopoiesis structural metabolism (#1528) and the d.successor_shadows
// forward-predictive discipline (#1466).  One device-owned cycle, no host
// decisions anywhere:
//
//   P  form prospective per-endpoint pressure predictions from resident
//      learned weights (endogenous nominations owning zero participation,
//      evidence, or credit authority -- priming law, volume 03 section 9),
//   W  wear kept components by baseline catabolism PLUS the traffic pressure
//      their endpoint actually imposes (innervation x supply depth),
//   L  score every prediction against the measured conductance loss -- the
//      settled physical consequence -- and move the weights only from those
//      outcomes, never from whether a prediction steered work its way, so the
//      loop cannot confirm itself (self-fulfilling credit firewall),
//   M  spend the bounded repair budget and nominate synthesis endpoints by
//      predicted need through one multiplicative-gain law; gain zero is
//      exactly the uninformed index-order control, so informed and control
//      twins under identical budgets diverge byte-wise only where predictions
//      actually changed what was produced,
//   N  renewal stays outcome-driven exactly as the parent metabolism defined
//      it: aged weak components retire with their exact matter reversed and
//      rebuild in place to the same endpoint under a fresh incarnation.
//
// Conservation is inherited whole: charged minus returned equals the matter
// resident in live metabolic components, agreeing with the construction
// reserve, reclaimed-resource and live-route-matter deltas and pool
// quiescence; every refusal fails closed.
inline constexpr std::uint32_t kPredictiveTargetCapacity = 256u;
inline constexpr std::int32_t kPredictivePressurePerInDegreeQ16 = 1 << 9;
inline constexpr std::int32_t kPredictivePressurePerDepthStepQ16 = 1 << 6;
inline constexpr std::int32_t kPredictiveDepthStride = 8;
inline constexpr std::int32_t kPredictiveMaxDepthSteps = 12;
inline constexpr std::int32_t kPredictivePressureCeilingQ16 = 3 << 12;
inline constexpr std::uint32_t kPredictiveMaxInfluenceDegree = 8u;

struct DirectPredictiveMetabolismConfig {
  DirectAutopoiesisConfig metabolism;
  std::uint32_t prediction_gain_q16;
  std::uint32_t learning_rate_shift;
  std::uint32_t freeze_learning;
};

struct DirectPredictiveMetabolismState {
  DirectAutopoiesisState metabolism;
  std::int32_t endpoint_pressure_q16[kPredictiveTargetCapacity];
  std::uint32_t prediction_rounds;
  std::int32_t prediction_error_first_q16;
  std::int32_t prediction_error_last_q16;
  // Explicit pad keeping the 64-bit ledgers alignment-clean so the state
  // stays unique-represented and byte-comparable.
  std::uint32_t prediction_pad;
  std::uint64_t prediction_error_sum_q16;
  std::uint64_t prediction_terms;
  std::uint64_t steered_repairs;
  std::uint64_t steered_syntheses;
};

static_assert(std::is_standard_layout_v<DirectPredictiveMetabolismState> &&
              std::is_trivial_v<DirectPredictiveMetabolismState> &&
              std::has_unique_object_representations_v<
                  DirectPredictiveMetabolismState>);

// Traffic pressure an endpoint imposes on the components feeding it:
// innervation count plus geometric supply depth, both bounded.  The
// innervation term couples metabolic production back into the predictive
// landscape -- synthesizing into an endpoint raises the pressure its feeders
// feel -- while the depth term is fixed tissue geometry the predictor must
// discover from repeated metabolic contact alone.
__device__ inline std::int32_t endpoint_traffic_pressure_q16(
    const DirectNode& endpoint) {
  const std::uint32_t degree =
      endpoint.active_in_degree > kPredictiveMaxInfluenceDegree
          ? kPredictiveMaxInfluenceDegree
          : endpoint.active_in_degree;
  const std::int32_t depth =
      (endpoint.coordinate[0] < 0 ? -endpoint.coordinate[0]
                                  : endpoint.coordinate[0]) +
      (endpoint.coordinate[1] < 0 ? -endpoint.coordinate[1]
                                  : endpoint.coordinate[1]) +
      (endpoint.coordinate[2] < 0 ? -endpoint.coordinate[2]
                                  : endpoint.coordinate[2]);
  std::int32_t depth_steps = depth / kPredictiveDepthStride;
  if (depth_steps > kPredictiveMaxDepthSteps)
    depth_steps = kPredictiveMaxDepthSteps;
  const std::int64_t pressure =
      static_cast<std::int64_t>(degree) *
          kPredictivePressurePerInDegreeQ16 +
      static_cast<std::int64_t>(depth_steps) *
          kPredictivePressurePerDepthStepQ16;
  return pressure > kPredictivePressureCeilingQ16
             ? kPredictivePressureCeilingQ16
             : static_cast<std::int32_t>(pressure);
}

// Birth handoff: one subject, bound once, never rebound.  Subjects whose
// tissue exceeds the predictive table refuse the handoff fail-closed.
__device__ inline bool bind_predictive_metabolism(
    DirectPredictiveMetabolismState* state, const DirectBrain& brain) {
  if (state == nullptr || brain.node_count > kPredictiveTargetCapacity ||
      state->prediction_rounds != 0u || state->prediction_terms != 0u)
    return false;
  return bind_autopoietic_tissue(&state->metabolism, brain);
}

__device__ inline bool predictive_ledger_balanced(
    const DirectPredictiveMetabolismState& state) {
  return autopoiesis_ledger_balanced(state.metabolism);
}

__device__ inline void record_prediction_error(
    DirectPredictiveMetabolismState* state, std::int64_t sum_q16,
    std::uint32_t terms) {
  if (terms == 0u) return;
  const std::int32_t mean_q16 =
      static_cast<std::int32_t>(sum_q16 / static_cast<std::int64_t>(terms));
  state->prediction_error_sum_q16 += static_cast<std::uint64_t>(sum_q16);
  state->prediction_terms += terms;
  if (state->prediction_rounds == 0u)
    state->prediction_error_first_q16 = mean_q16;
  state->prediction_error_last_q16 = mean_q16;
}

// One integrated predict -> wear -> learn -> repair -> renew -> synthesize
// cycle.  Returns false only for boundary refusal (foreign or unbound
// tissue), which counts itself and touches nothing else.
__device__ inline bool run_predictive_metabolic_epoch(
    const DirectPredictiveMetabolismConfig& config,
    DirectPredictiveMetabolismState* state, DirectBrain* brain) {
  if (state == nullptr) return false;
  if (brain == nullptr ||
      !tissue_bound(state->metabolism, *brain)) {
    ++state->metabolism.boundary_refusals;
    return false;
  }
  DirectNode* nodes = brain->nodes;
  DirectRoute* routes = brain->routes;
  ResidentDevelopmentState* development = brain->development;
  substrate::direct_adult::DirectResourceEcologyState* ecology =
      brain->resource_ecology;
  if (nodes == nullptr || routes == nullptr || development == nullptr ||
      ecology == nullptr)
    return false;
  DirectAutopoiesisState& metabolism = state->metabolism;
  ++metabolism.epoch;

  // ---- P: prospective pressure nominations ------------------------------
  // Enumerated with exactly the predicate the wear phase re-applies, so each
  // prediction pairs with its own settled outcome.
  std::int32_t predicted_drop_q16[kAutopoieticComponentCapacity];
  std::uint32_t live_components = 0u;
  for (std::uint32_t i = 0u; i < metabolism.count; ++i) {
    const DirectRoute& route = routes[metabolism.components[i].route_index];
    if (!route_is_active(route)) continue;
    std::int32_t predicted = config.metabolism.wear_per_epoch_q16;
    if (route.target < brain->node_count)
      predicted += state->endpoint_pressure_q16[route.target];
    if (predicted < 0) predicted = 0;
    predicted_drop_q16[live_components++] = predicted;
  }

  // ---- W: catabolic wear under real endpoint traffic --------------------
  // A component the floor clamps carries no pressure information this epoch
  // (its measured loss is truncated), so it neither scores nor teaches.
  std::uint32_t scored = 0u;
  std::int64_t error_sum = 0;
  std::uint32_t live_index = 0u;
  for (std::uint32_t i = 0u; i < metabolism.count; ++i) {
    DirectAutopoieticComponent& component = metabolism.components[i];
    DirectRoute& route = routes[component.route_index];
    if (!route_is_active(route)) continue;
    const std::int32_t before = route.conductance_q16;
    bool saturated = false;
    if (route.target < brain->node_count) {
      const std::int32_t pressure =
          endpoint_traffic_pressure_q16(nodes[route.target]);
      const std::int32_t worn =
          before - config.metabolism.wear_per_epoch_q16 - pressure;
      saturated = worn < kAutopoieticFloorConductanceQ16;
      route.conductance_q16 =
          saturated ? kAutopoieticFloorConductanceQ16 : worn;
    }
    const std::int32_t predicted = predicted_drop_q16[live_index++];
    if (!saturated) {
      const std::int32_t actual = before - route.conductance_q16;
      const std::int32_t miss = predicted > actual ? predicted - actual
                                                   : actual - predicted;
      error_sum += miss;
      ++scored;
      if (config.freeze_learning == 0u && route.target < brain->node_count) {
        std::int32_t excess = actual - config.metabolism.wear_per_epoch_q16;
        if (excess < 0) excess = 0;
        std::int32_t& weight = state->endpoint_pressure_q16[route.target];
        const std::int32_t gap = excess - weight;
        weight += gap >= 0 ? (gap >> config.learning_rate_shift)
                           : -((-gap) >> config.learning_rate_shift);
        if (weight < 0) weight = 0;
        if (weight > kPredictivePressureCeilingQ16)
          weight = kPredictivePressureCeilingQ16;
      }
    }
  }
  record_prediction_error(state, error_sum, scored);
  ++state->prediction_rounds;

  // ---- M: paid repair nominated by predicted need -----------------------
  bool repaired[kAutopoieticComponentCapacity]{};
  std::uint32_t repair_budget = config.metabolism.repair_budget_per_epoch;
  while (repair_budget != 0u) {
    std::uint32_t chosen = kInvalidIndex;
    std::int64_t chosen_score = -1;
    std::uint32_t first_eligible = kInvalidIndex;
    for (std::uint32_t i = 0u; i < metabolism.count; ++i) {
      if (repaired[i]) continue;
      const DirectRoute& route = routes[metabolism.components[i].route_index];
      if (!route_is_active(route) ||
          route.conductance_q16 >= kAutopoieticRepairThresholdQ16 ||
          route.target >= brain->node_count)
        continue;
      if (first_eligible == kInvalidIndex) first_eligible = i;
      const std::int32_t predicted_next =
          route.conductance_q16 - config.metabolism.wear_per_epoch_q16 -
          state->endpoint_pressure_q16[route.target];
      std::int32_t urgency = kAutopoieticRepairThresholdQ16 - predicted_next;
      if (urgency < 0) urgency = 0;
      const std::int64_t score =
          static_cast<std::int64_t>(urgency) * config.prediction_gain_q16;
      if (score > chosen_score) {
        chosen_score = score;
        chosen = i;
      }
    }
    if (chosen == kInvalidIndex) break;
    repaired[chosen] = true;
    if (chosen != first_eligible) ++state->steered_repairs;
    DirectRoute& route = routes[metabolism.components[chosen].route_index];
    const std::int32_t restored = route.conductance_q16 + kAutopoieticRepairStepQ16;
    route.conductance_q16 = restored > kAutopoieticIntactConductanceQ16
                                ? kAutopoieticIntactConductanceQ16
                                : restored;
    ++metabolism.repairs;
    ++metabolism.repair_energy_debited;
    --repair_budget;
  }

  // ---- N: outcome-driven renewal, exact matter reversal -----------------
  for (std::uint32_t i = 0u; i < metabolism.count; ++i) {
    DirectAutopoieticComponent& component = metabolism.components[i];
    if (metabolism.epoch - component.born_epoch <=
        config.metabolism.renewal_window_epochs)
      continue;
    DirectRoute& route = routes[component.route_index];
    if (!route_is_active(route) || route.target >= brain->node_count) continue;
    DirectNode& owner = nodes[route.source];
    if (!weak_owner_support(owner)) continue;
    const std::uint32_t route_index = component.route_index;
    const std::uint32_t endpoint = route.target;
    retire_component_slot(brain, &metabolism, owner, route_index,
                          component.matter_units);
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
    metabolism.matter_charged += rebuilt_cost;
    component.born_epoch = metabolism.epoch;
    component.matter_units = rebuilt_cost;
    component.target_node = endpoint;
    ++metabolism.renewals;
  }

  // ---- M: synthesis into predicted-demand endpoints ---------------------
  std::uint32_t synthesis_budget = config.metabolism.synthesis_budget_per_epoch;
  for (std::uint32_t node_index = 0u;
       node_index < brain->node_count && synthesis_budget != 0u; ++node_index) {
    const DirectNode& owner = nodes[node_index];
    std::uint32_t chosen_slot = kInvalidIndex;
    for (std::uint32_t r = owner.route_offset;
         r < owner.route_offset + owner.route_capacity && chosen_slot == kInvalidIndex;
         ++r) {
      if (route_is_active(routes[r]) ||
          (routes[r].flags & kRouteFlagDevelopmentalReserve) == 0u)
        continue;
      chosen_slot = r;
    }
    if (chosen_slot == kInvalidIndex) continue;
    std::uint32_t chosen_target = kInvalidIndex;
    std::int64_t chosen_score = -1;
    std::uint32_t first_valid = kInvalidIndex;
    for (std::uint32_t t = 0u; t < brain->node_count; ++t) {
      if (t == node_index || nodes[t].active_in_degree >= kAutopoiesisMaxTargetInDegree ||
          duplicate_active_edge(routes, owner, t))
        continue;
      if (first_valid == kInvalidIndex) first_valid = t;
      const std::int64_t score =
          static_cast<std::int64_t>(state->endpoint_pressure_q16[t]) *
          config.prediction_gain_q16;
      if (score > chosen_score) {
        chosen_score = score;
        chosen_target = t;
      }
    }
    if (chosen_target == kInvalidIndex) continue;
    if (chosen_target != first_valid) ++state->steered_syntheses;
    const std::uint32_t cost = construction_route_cost(owner, nodes[chosen_target]);
    if (metabolism.count >= kAutopoieticComponentCapacity) {
      ++metabolism.capacity_refusals;
      break;
    }
    if (development->constructor_reserve < cost) {
      ++metabolism.matter_refusals;
      break;
    }
    if (!substrate::direct_adult::device_reserve_pool_units(
            ecology, substrate::direct_adult::DirectResourcePoolKind::explicit_interaction,
            cost)) {
      ++metabolism.matter_refusals;
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
    metabolism.components[metabolism.count++] =
        DirectAutopoieticComponent{chosen_slot, chosen_target, metabolism.epoch, cost};
    metabolism.matter_charged += cost;
    ++metabolism.syntheses;
    --synthesis_budget;
  }

  return true;
}

}  // namespace substrate::direct_network

#endif  // HARDWARE_NATIVE_DIRECT_ADULT_AUTOPOIESIS_PREDICTIVE_CUH
