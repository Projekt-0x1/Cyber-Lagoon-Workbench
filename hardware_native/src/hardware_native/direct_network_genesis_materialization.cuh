// Included inside direct_network_life_function.cu's anonymous namespace after
// direct_network_corridor_law.inl, whose long-tract target and delay laws the
// route kernel consumes. Owns every device writer of born matter -- nodes,
// routes, dense tiles, boundary ports -- including the pool commits and
// uncommits that move matter between reserved and live, plus overload
// refinement, prenatal stabilization, birth handoff, and census.
constexpr std::int32_t kInitialConductanceQ16 = 1 << 16;

// gh #1251: direct_network_resident_development.cu (same library target)
// names its own TU-local copy of this floor kMinConductanceQ16 instead --
// same value (1 << 10), different spelling. That spelling also shadow-
// collides, by name only, with direct_adult_core.cuh's external
// kMinConductanceQ16, whose value is 1 << 8, NOT 1 << 10. Both this file's
// and the sibling's copies are TU-local (anonymous namespace); neither
// includes the header they happen to echo.
constexpr std::int32_t kMinimumConductanceQ16 = 1 << 10;
constexpr std::int32_t kMaximumConductanceQ16 = 4 << 16;

DIRECT_NETWORK_HD inline std::uint32_t find_plan_for_node(const TerritoryPlan* plans,
                                                           std::uint32_t plan_count,
                                                           std::uint32_t node_index) {
  for (std::uint32_t p = 0; p < plan_count; ++p) {
    const TerritoryPlan& plan = plans[p];
    if (plan.active != 0u && node_index >= plan.node_offset &&
        node_index < plan.node_offset + plan.node_count) {
      return p;
    }
  }
  return kInvalidIndex;
}

__global__ void materialize_nodes_kernel(const GammaV1* gamma,
                                         const DirectDevelopmentEnvironmentV1* environment,
                                         const TerritoryPlan* plans,
                                         std::uint32_t plan_count, DirectNode* nodes,
                                         std::uint32_t node_count,
                                         CompileDiagnostics* diagnostics) {
  const std::uint32_t global = blockIdx.x * blockDim.x + threadIdx.x;
  if (global >= node_count) return;
  const std::uint32_t p = find_plan_for_node(plans, plan_count, global);
  if (p == kInvalidIndex) return;
  const TerritoryPlan plan = plans[p];
  const std::uint32_t local = global - plan.node_offset;
  const std::uint32_t logical_tick = logical_node_birth_tick(*gamma, plan, local);

  std::int32_t best_coord[3]{};
  std::uint32_t extended_draws = 0u;
  const std::int32_t best_score = select_node_site(
      *gamma, *environment, plan, local, logical_tick, best_coord,
      &extended_draws);
  if (extended_draws != 0u) atomicAdd(&diagnostics->extended_draw_nodes, 1u);
  // Still counted, and now it means what it says: every first-draw candidate AND
  // the whole extended draw were forbidden, so there was nowhere lawful to grow.
  if (environment_hard_excludes(*environment, best_coord))
    atomicAdd(&diagnostics->environment_violating_nodes, 1u);

  DirectNode node{};
  node.coordinate[0] = best_coord[0];
  node.coordinate[1] = best_coord[1];
  node.coordinate[2] = best_coord[2];
  node.lineage = plan.lineage;
  node.chemotype = plan.chemotype;
  node.territory_index = static_cast<std::uint16_t>(plan.seed_index);
  node.route_offset = plan.route_offset + local * plan.route_capacity_per_node;
  node.route_capacity = static_cast<std::uint16_t>(plan.route_capacity_per_node);
  node.active_route_count = static_cast<std::uint16_t>(plan.sparse_degree +
                                                (local < plan.long_tract_count ? 1u : 0u));
  node.attract_field = plan.attract_field < kInvalidFieldIndex16
                           ? static_cast<std::uint16_t>(plan.attract_field)
                           : kInvalidFieldIndex16;
  node.repel_field = plan.repel_field < kInvalidFieldIndex16
                         ? static_cast<std::uint16_t>(plan.repel_field)
                         : kInvalidFieldIndex16;
  node.resource_field = plan.resource_field < kInvalidFieldIndex16
                            ? static_cast<std::uint16_t>(plan.resource_field)
                            : kInvalidFieldIndex16;
  node.maturation_field = plan.maturation_field < kInvalidFieldIndex16
                              ? static_cast<std::uint16_t>(plan.maturation_field)
                              : kInvalidFieldIndex16;
  node.inhibition_field = plan.inhibition_field < kInvalidFieldIndex16
                              ? static_cast<std::uint16_t>(plan.inhibition_field)
                              : kInvalidFieldIndex16;
  node.repair_field = plan.repair_field < kInvalidFieldIndex16
                          ? static_cast<std::uint16_t>(plan.repair_field)
                          : kInvalidFieldIndex16;
  node.maintenance_q16 = static_cast<std::int32_t>(kQ16One);
  const std::uint32_t begin_tick = gamma->seeds[plan.seed_index].begin_tick;
  const std::uint32_t handoff_tick = max_u32(begin_tick + 1u, gamma->header.development_end_tick);
  const std::uint32_t phase_span = max_u32(1u, handoff_tick - begin_tick);
  const std::uint32_t phase_q16 = static_cast<std::uint32_t>(
      (static_cast<std::uint64_t>(logical_tick - begin_tick) * kQ16One) / phase_span);
  const std::uint32_t baseline_maturation_q16 =
      min_u32(kQ16One / 2u, kQ16One / 8u + phase_q16 / 4u);
  const std::int32_t authored_maturation_q16 = bound_field_kind_influence_q16(
      *gamma, plan, best_coord, plan.chemotype, logical_tick,
      DevelopmentFieldKind::maturation);
  const std::int64_t initial_maturation_q16 =
      static_cast<std::int64_t>(baseline_maturation_q16) + authored_maturation_q16;
  node.maturation_q16 = static_cast<std::uint32_t>(
      initial_maturation_q16 <= 0
          ? 0
          : min_i64(initial_maturation_q16, static_cast<std::int64_t>(kQ16One / 2u)));
  node.attractor_support_q16 = best_score / 4;
  if ((plan.flags & kRuleFlagDenseIntegrative) != 0u && local < plan.dense_width) {
    node.flags |= kNodeFlagDenseMember;
  }
  if ((plan.flags & kRuleFlagConstructorReserve) != 0u &&
      (mix64(global ^ gamma->header.development_seed) & 15ull) == 0ull) {
    node.flags |= kNodeFlagConstructor;
  }
  // gh #1359 / #1267 / #1290. Stable, domain-separated developmental priority
  // with threshold cutoff for competition realization. Uses territory-local
  // coordinate (seed_index, local) rather than global allocation index, and
  if ((plan.flags & kRuleFlagInhibitoryBias) != 0u && plan.authoring_fault == 0u) {
    // Competition is territory-wide. Density says which members supply the
    // brake, not which members may enter the local action election.
    node.flags |= kNodeFlagCompetitive;
    if (plan.inhibition_priority_threshold > 0ull) {
      const std::uint64_t priority =
          competition_priority_key(*gamma, plan.seed_index, local);
      if (priority <= plan.inhibition_priority_threshold) {
        node.flags |= kNodeFlagInhibitory;
        // gh #1310: Outbound competition strength / brake magnitude is stored
        // in competition_strength_q16, decoupled from mutable resident homeostatic inhibition.
        node.competition_strength_code_q16 = encode_competition_strength_q16(plan.competition_strength_q16);
        node.inhibition_q16 = kQ16One / 2;
      }
    }
  }
  node.flags |= kNodeFlagImmature;
  nodes[global] = node;
}

DIRECT_NETWORK_HD inline bool already_selected_target(const DirectRoute* routes,
                                                       std::uint32_t route_offset,
                                                       std::uint32_t count,
                                                       std::uint32_t target) {
  for (std::uint32_t i = 0; i < count; ++i) {
    if (routes[route_offset + i].target == target) return true;
  }
  return false;
}

__global__ void materialize_sparse_routes_kernel(const GammaV1* gamma,
                                                  const TerritoryPlan* plans,
                                                  std::uint32_t plan_count,
                                                  DirectNode* nodes,
                                                  DirectRoute* routes,
                                                  std::uint64_t* route_incarnations,
                                                  std::uint32_t* route_delay_law_indices,
                                                  std::uint32_t* route_mature_delays,
                                                  std::uint64_t* route_delay_law_incarnations,
                                                  const DirectTractDelayLawV1* rule_delay_laws,
                                                  std::uint32_t rule_delay_law_count,
                                                  std::uint32_t node_count,
                                                  std::uint32_t candidate_targets,
                                                  CompileDiagnostics* diagnostics,
                                                  substrate::direct_adult::DirectResourceEcologyState*
                                                      ecology) {
  const std::uint32_t global = blockIdx.x * blockDim.x + threadIdx.x;
  if (global >= node_count) return;
  const std::uint32_t p = find_plan_for_node(plans, plan_count, global);
  if (p == kInvalidIndex) return;
  const TerritoryPlan plan = plans[p];
  DirectNode& source = nodes[global];
  const std::uint32_t source_local = global - plan.node_offset;
  const std::uint32_t logical_tick = logical_node_birth_tick(*gamma, plan, source_local);

  // Initialize every slot in the pre-paid storage capacity to a deterministic
  // dormant reserve state.  ResidentDevelopment can later activate these
  // without reallocating/moving adjacency.
  for (std::uint32_t slot = 0; slot < source.route_capacity; ++slot) {
    DirectRoute reserve{};
    reserve.source = global;
    reserve.target = kInvalidIndex;
    reserve.flags = kRouteFlagDevelopmentalReserve;
    reserve.eligibility_context = kInvalidIndex;
    reserve.last_credit_ticket = kNoCreditTicket;
    const std::uint32_t route_index = source.route_offset + slot;
    routes[route_index] = reserve;
    route_incarnations[route_index] = initial_route_incarnation(route_index);
    if (route_delay_law_indices != nullptr) {
      route_delay_law_indices[route_index] = kInvalidIndex;
      route_mature_delays[route_index] = 0u;
      route_delay_law_incarnations[route_index] = 0u;
    }
  }

  for (std::uint32_t slot = 0; slot < plan.sparse_degree; ++slot) {
    std::uint32_t best_target = kInvalidIndex;
    std::int32_t best_score = INT32_MIN;
    const std::uint32_t samples = max_u32(2u, candidate_targets);
    for (std::uint32_t candidate = 0; candidate < samples; ++candidate) {
      const std::uint32_t target_local = candidate_local_target(*gamma, plan, source_local, slot,
                                                                candidate);
      const std::uint32_t target = plan.node_offset + target_local;
      if (target == global || already_selected_target(routes, source.route_offset, slot, target)) continue;
      const DirectNode& target_node = nodes[target];
      std::int32_t score = geometry_route_score_q16(source, target_node);
      score += combined_developmental_score_q16(*gamma, plan, target_node.coordinate, logical_tick);
      // b.local_fields: the cone sits at the source; a candidate target is
      // scored by how far its direction climbs the local morphogen gradient.
      score += combined_gradient_tilt_q16(*gamma, plan, target_node.coordinate,
                                          source.coordinate, logical_tick);
      if ((source.flags & kNodeFlagDenseMember) != 0u &&
          (target_node.flags & kNodeFlagDenseMember) != 0u) {
        score += kQ16One / 2;
      }
      if (score > best_score || (score == best_score && target < best_target)) {
        best_score = score;
        best_target = target;
      }
    }
    bool fallback_wired = false;
    if (best_target == kInvalidIndex) {
      // Every Gamma candidate for this slot was skipped -- self-target, or a
      // target this source already took. The ring below consults neither Gamma,
      // nor geometry, nor the developmental score, so it is authored wiring and
      // is counted separately (gh #1309).
      best_target = plan.node_offset + ((source_local + slot + 1u) % plan.node_count);
      best_score = 0;
      fallback_wired = true;
    }
    DirectRoute route{};
    route.source = global;
    route.target = best_target;
    route.flags = kRouteFlagActive | kRouteFlagRecurrent;
    if ((source.flags & kNodeFlagInhibitory) != 0u) route.flags |= kRouteFlagInhibitory;
    route.delay = route_delay_from_geometry(source, nodes[best_target], false);
    route.developmental_score_q16 = best_score;
    route.conductance_q16 = clamp_i32(kInitialConductanceQ16 + (best_score >> 4),
                                      kMinimumConductanceQ16, kMaximumConductanceQ16);
    route.eligibility_context = kInvalidIndex;
    route.last_credit_ticket = kNoCreditTicket;
    routes[source.route_offset + slot] = route;
    atomicAdd(&diagnostics->active_routes, 1u);
    if (fallback_wired) atomicAdd(&diagnostics->fallback_wired_routes, 1u);
    // Commit one pre-paid slot. The reservation for every slot in the arena was
    // booked on the host before the arena existed; activating a slot is what
    // turns reserved matter into live matter, which is exactly what commit means.
    substrate::direct_adult::device_commit_pool_units(
        ecology, substrate::direct_adult::DirectResourcePoolKind::explicit_interaction, 1u);
  }

  if (source_local < plan.long_tract_count && source.active_route_count > plan.sparse_degree) {
    const std::uint32_t slot = plan.sparse_degree;
    bool steered = false;
    std::int32_t tract_score = 0;
    bool tract_gain_participates = false;
    const std::uint32_t corridor_rule = corridor_rule_for_local(*gamma, plan, source_local);
    if (corridor_rule == kInvalidIndex) return;
    const ConstructionRule tract_rule = gamma->rules[corridor_rule];
    const std::uint32_t maximum_age =
        tract_rule.maximum_age == 0u ? 0xffffffffu : tract_rule.maximum_age;
    if (logical_tick < tract_rule.minimum_age || logical_tick > maximum_age) {
      // gh #1268 / #1267 tract_maturation: this node's own developmental birth
      // tick falls outside the family's authored maturation window. Nothing is
      // written or committed -- the same restraint the refusal branch below
      // already obeys for a named partner no territory satisfies.
      atomicAdd(&diagnostics->immature_deferred_tracts, 1u);
      return;
    }
    const std::uint32_t target = choose_long_tract_target(
        *gamma, plans, plan_count, p, nodes, global, corridor_rule, &steered, &tract_score,
        &tract_gain_participates);
    if (target == kInvalidIndex) {
      // Refused: this territory named a partner no territory satisfies. Nothing
      // is written, nothing is committed -- an unbuilt tract must not charge the
      // ledger for matter it never took (#1178).
      atomicAdd(&diagnostics->refused_partner_tracts, 1u);
      return;
    }
    if (steered) atomicAdd(&diagnostics->partner_steered_tracts, 1u);
    DirectRoute route{};
    route.source = global;
    route.target = target;
    route.flags = kRouteFlagActive | kRouteFlagLongTract | ((source.flags & kNodeFlagInhibitory) != 0u ? kRouteFlagInhibitory : 0u);
    const DirectTractDelayLawV1 delay_law =
        corridor_rule < rule_delay_law_count ? rule_delay_laws[corridor_rule]
                                             : DirectTractDelayLawV1{};
    route.delay = initial_tract_delay(*gamma, plan, source_local, corridor_rule, delay_law,
                                      source, nodes[target]);
    route.developmental_score_q16 = geometry_route_score_q16(source, nodes[target]);
    // distinct_effect_vector_over_gain (#1276/NET12 rung 3): every pre-rung-3
    // field has tract_gain_participates == false (see field_gain_participates),
    // so this is a strict no-op for every existing long-tract corridor --
    // route.conductance_q16 stays exactly kInitialConductanceQ16, bit for bit.
    // Only a field that opts in has its own (possibly decayed --
    // field_decayed_strength_q16 already folded into tract_score via
    // choose_long_tract_target's affinity term) influence bias the GAIN of the
    // routes it wins, mirroring the sparse-degree path's existing
    // best_score-derived conductance a few dozen lines above.
    route.conductance_q16 = tract_gain_participates
                                 ? clamp_i32(kInitialConductanceQ16 + (tract_score >> 4),
                                             kMinimumConductanceQ16, kMaximumConductanceQ16)
                                 : kInitialConductanceQ16;
    route.eligibility_context = kInvalidIndex;
    route.last_credit_ticket = kNoCreditTicket;
    const std::uint32_t route_index = source.route_offset + slot;
    routes[route_index] = route;
    if (explicit_tract_delay_law(delay_law) && route_delay_law_indices != nullptr) {
      route_delay_law_indices[route_index] = corridor_rule;
      route_mature_delays[route_index] =
          mature_tract_delay(*gamma, plan, source_local, corridor_rule, delay_law);
      route_delay_law_incarnations[route_index] = route_incarnations[route_index];
    }
    atomicAdd(&diagnostics->active_routes, 1u);
    atomicAdd(&diagnostics->long_tracts, 1u);
    substrate::direct_adult::device_commit_pool_units(
        ecology, substrate::direct_adult::DirectResourcePoolKind::explicit_interaction, 1u);
  }
}

__global__ void clear_u32_kernel(std::uint32_t* values, std::uint32_t count) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count) values[i] = 0u;
}

// gh #1298 follow-up: the three cudaMemsetAsync host dispatches that used to
// open every genesis compile (clearing diagnostics, dense_block_counts, and
// dense_weight_counts) collapse into one kernel launch. Same values, same
// sizes: diagnostics is still cleared unconditionally via value-init, and
// both per-seed count arrays are still zeroed for exactly `seed_count`
// elements each.
__global__ void clear_compile_scratch_kernel(CompileDiagnostics* diagnostics,
                                             std::uint32_t* dense_block_counts,
                                             std::uint32_t* dense_weight_counts,
                                             std::uint32_t seed_count) {
  const std::uint32_t stride = blockDim.x * gridDim.x;
  for (std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < seed_count; i += stride) {
    dense_block_counts[i] = 0u;
    dense_weight_counts[i] = 0u;
  }
  if (blockIdx.x == 0u && threadIdx.x == 0u) {
    *diagnostics = CompileDiagnostics{};
  }
}

__global__ void count_in_degree_kernel(const DirectRoute* routes, std::uint32_t route_capacity,
                                       std::uint32_t node_count, std::uint32_t* in_degree) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= route_capacity) return;
  const DirectRoute route = routes[i];
  if (!route_is_active(route) || route.target >= node_count) return;
  atomicAdd(&in_degree[route.target], 1u);
}

__global__ void install_active_in_degree_kernel(DirectNode* nodes, const std::uint32_t* in_degree,
                                                std::uint32_t node_count) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= node_count) return;
  nodes[i].active_in_degree = in_degree[i];
}

// Birth handoff: convert developmental node state into adult initial state.
//
// MEASURED 2026-08-18 by cuda_direct_network_basin_probe_contract, and this
// kernel exists because of that measurement. `attractor_support_q16` is ONE
// struct slot serving TWO different quantities:
//
//   * During development, `prenatal_stabilize_kernel` writes the node's MEAN
//     ACTIVE-ROUTE CONDUCTANCE there, clamped to [0, 4 * kQ16One], and
//     direct_network_resident_development.cu reads it as a structural score
//     (`score += target.attractor_support_q16 >> 3`).
//   * In the adult, `integrate_node_activation_kernel` reads the same slot as
//     an additive drive in ACTIVATION units, clamped to [0, kQ16One], raised
//     only when a node actually fires and decayed when it does not.
//
// These are not a badly-scaled version of each other; they are different
// quantities. Handing the first straight to the second put every node's
// net_input above threshold before any input arrived: the probe measured
// [70884..76846] at birth, entirely above the adult's kQ16One ceiling, and
// within two epochs all 4096 nodes sat at exactly kQ16One in both activation
// and support -- one dynamical state, no basins, and ~5962 units of grown
// variation gone.
//
// The newborn's adult support is ZERO. Attractor support is accumulated from
// actual coactivation and a newborn has fired nothing; a non-zero starting
// value is unearned drive. Nothing grown is discarded by this: the mean
// conductance the compiler computed still lives in routes[].conductance_q16,
// which is where the adult reads it from every tick through incoming
// excitation. What is deleted is a type confusion, not a phenotype.
//
// Corroboration that this belongs here rather than in each caller:
// cuda_direct_adult_core_full_contract already zeroed this field by hand before
// every arm.
__global__ void birth_handoff_kernel(DirectNode* nodes, std::uint32_t node_count) {
  const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= node_count) return;
  nodes[index].attractor_support_q16 = 0;
}

__global__ void refine_overloaded_targets_kernel(const GammaV1* gamma,
                                                 const TerritoryPlan* plans,
                                                 std::uint32_t plan_count,
                                                 DirectNode* nodes,
                                                 DirectRoute* routes,
                                                 std::uint32_t node_count,
                                                 const std::uint32_t* in_degree,
                                                 std::uint32_t maximum_in_degree,
                                                 std::uint32_t candidate_targets) {
  const std::uint32_t global = blockIdx.x * blockDim.x + threadIdx.x;
  if (global >= node_count) return;
  const std::uint32_t p = find_plan_for_node(plans, plan_count, global);
  if (p == kInvalidIndex) return;
  const TerritoryPlan plan = plans[p];
  DirectNode& source = nodes[global];
  const std::uint32_t source_local = global - plan.node_offset;
  const std::uint32_t logical_tick = logical_node_birth_tick(*gamma, plan, source_local);

  for (std::uint32_t slot = 0; slot < source.route_capacity; ++slot) {
    DirectRoute& route = routes[source.route_offset + slot];
    if (!route_is_active(route) || (route.flags & kRouteFlagLongTract) != 0u ||
        route.target >= node_count || in_degree[route.target] <= maximum_in_degree) {
      continue;
    }
    std::uint32_t best_target = route.target;
    std::int32_t best_score = INT32_MIN;
    for (std::uint32_t candidate = 0; candidate < max_u32(2u, candidate_targets); ++candidate) {
      const std::uint32_t target_local = candidate_local_target(*gamma, plan, source_local, slot,
                                                                candidate + 17u);
      const std::uint32_t target = plan.node_offset + target_local;
      if (target == global) continue;
      const DirectNode& target_node = nodes[target];
      std::int64_t score = geometry_route_score_q16(source, target_node);
      score += combined_developmental_score_q16(*gamma, plan, target_node.coordinate, logical_tick);
      score += combined_gradient_tilt_q16(*gamma, plan, target_node.coordinate,
                                          source.coordinate, logical_tick);
      score -= static_cast<std::int64_t>(in_degree[target]) << 12;
      if (score > best_score || (score == best_score && target < best_target)) {
        best_score = static_cast<std::int32_t>(score);
        best_target = target;
      }
    }
    route.target = best_target;
    route.developmental_score_q16 = best_score;
    route.delay = route_delay_from_geometry(source, nodes[best_target], false);
  }
}

__global__ void prenatal_stabilize_kernel(DirectNode* nodes, DirectRoute* routes,
                                          std::uint64_t* route_incarnations,
                                          std::uint32_t node_count,
                                          const std::uint32_t* in_degree,
                                          std::uint32_t maximum_in_degree,
                                          std::uint32_t pass,
                                          substrate::direct_adult::DirectResourceEcologyState* ecology,
                                          CompileDiagnostics* diagnostics) {
  const std::uint32_t node_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (node_index >= node_count) return;
  DirectNode& node = nodes[node_index];
  const std::uint32_t incoming = in_degree[node_index];
  std::int64_t support = 0;
  std::uint32_t active = 0u;
  for (std::uint32_t slot = 0; slot < node.route_capacity; ++slot) {
    DirectRoute& route = routes[node.route_offset + slot];
    if (!route_is_active(route)) continue;
    ++active;
    support += route.conductance_q16;
    // Preserve the first two recurrent scaffold routes and all long tracts;
    // prune only weak redundant routes during prenatal stabilization.
    if (slot >= 2u && (route.flags & kRouteFlagLongTract) == 0u &&
        route.developmental_score_q16 < -(1 << 15) && incoming > maximum_in_degree && pass > 0u) {
      // #1178: prenatal pruning runs AFTER materialization has committed every
      // activated slot, so a pruned route must give its unit back or the ledger
      // leaves birth already over-counting. This path had been unwired and
      // invisible: the reconciliation only agreed because the fixtures never
      // pruned anything, which is the same "a guard that examined nothing agrees
      // with every ledger" failure one layer down.
      route.flags &= ~kRouteFlagActive;
      route.flags |= kRouteFlagDevelopmentalReserve;
      route_incarnations[node.route_offset + slot] += kRouteIncarnationStride;
      substrate::direct_adult::device_uncommit_pool_units(
          ecology, substrate::direct_adult::DirectResourcePoolKind::explicit_interaction, 1u);
      if (node.active_route_count > 0u) --node.active_route_count;
      atomicAdd(&diagnostics->pruned_routes, 1u);
      atomicSub(&diagnostics->active_routes, 1u);
      --active;
    }
  }
  if (active != 0u) support /= static_cast<std::int64_t>(active);
  node.attractor_support_q16 = clamp_i32(static_cast<std::int32_t>(support), 0, 4 << 16);
  const std::uint32_t maturation_step =
      max_u32(1u, kQ16One / max_u32(2u, 8u - min_u32(pass, 6u)));
  node.maturation_q16 = min_u32(kQ16One, node.maturation_q16 + maturation_step);
  if (node.maturation_q16 >= kQ16One / 2u) node.flags &= ~kNodeFlagImmature;
}

__global__ void materialize_dense_blocks_kernel(const TerritoryPlan* plans,
                                                std::uint32_t plan_count,
                                                DirectDenseBlock* blocks,
                                                std::uint16_t* weight_bits,
                                                std::uint64_t development_seed,
                                                substrate::direct_adult::DirectResourceEcologyState*
                                                    ecology) {
  const std::uint32_t p = blockIdx.x;
  if (p >= plan_count) return;
  const TerritoryPlan plan = plans[p];
  if (plan.active == 0u || (plan.flags & kRuleFlagDenseIntegrative) == 0u ||
      plan.dense_width < 16u) {
    return;
  }
  if (threadIdx.x == 0u) {
    DirectDenseBlock block{};
    block.node_begin = plan.node_offset;
    block.node_count = plan.dense_width;
    block.weight_offset = plan.dense_weight_offset;
    block.weight_count = plan.dense_width * plan.dense_width;
    block.lineage = plan.lineage;
    block.flags = kDenseBlockFlagTensorEligible | kDenseBlockFlagRecurrent;
    if ((plan.flags & kRuleFlagSequenceIntegrator) != 0u) {
      block.flags |= kDenseBlockFlagSequenceIntegrator;
    }
    block.tile_width = 16u;
    blocks[plan.dense_block_offset] = block;
    substrate::direct_adult::device_commit_pool_units(
        ecology, substrate::direct_adult::DirectResourcePoolKind::dense_tile, 1u);
  }
  const std::uint32_t weight_count = plan.dense_width * plan.dense_width;
  __half* half_weights = reinterpret_cast<__half*>(weight_bits + plan.dense_weight_offset);
  for (std::uint32_t local = threadIdx.x; local < weight_count; local += blockDim.x) {
    const std::uint32_t row = local / plan.dense_width;
    const std::uint32_t col = local % plan.dense_width;
    const std::uint64_t h = mix64(development_seed ^
                                  (static_cast<std::uint64_t>(plan.seed_index) << 48) ^
                                  (static_cast<std::uint64_t>(row) << 24) ^ col);
    const float raw = (static_cast<int>(h & 0xffffu) - 32768) / 32768.0f;
    const float diagonal = row == col ? 0.125f : 0.0f;
    half_weights[local] = __float2half_rn(raw * 0.03125f + diagonal);
  }
}

// Where one boundary binding attaches, and what the port it produces contains.
// Extracted verbatim from attach_boundary_ports_kernel so the membrane can be
// read on the host (direct_probe_boundary_port) instead of only inside a device
// launch -- github #1322. Returns false for a binding the compiler refuses; the
// caller is what counts the refusal.
DIRECT_NETWORK_HD inline bool attach_boundary_port(const BoundaryPortBinding& binding,
                                                   const TerritoryPlan* plans,
                                                   std::uint32_t plan_count,
                                                   DirectBoundaryPort* out_port) {
  if (binding.seed_index >= plan_count) return false;
  const TerritoryPlan plan = plans[binding.seed_index];
  if (plan.active == 0u || binding.local_node >= plan.node_count) return false;
  DirectBoundaryPort port{};
  port.node = plan.node_offset + binding.local_node;
  port.channel = binding.channel;
  port.role_mask = binding.role_mask;
  port.physical_route = binding.physical_route;
  port.parent_route = binding.parent_route;
  *out_port = port;
  return true;
}

__global__ void attach_boundary_ports_kernel(const DirectBodyManifestV1* body,
                                             const TerritoryPlan* plans,
                                             std::uint32_t plan_count,
                                             DirectNode* nodes,
                                             DirectBoundaryPort* ports,
                                             CompileDiagnostics* diagnostics,
                                             substrate::direct_adult::DirectResourceEcologyState*
                                                 ecology) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= body->binding_count) return;
  const BoundaryPortBinding binding = body->bindings[i];
  DirectBoundaryPort port{};
  if (!attach_boundary_port(binding, plans, plan_count, &port)) {
    atomicAdd(&diagnostics->invalid_boundary_bindings, 1u);
    return;
  }
  const std::uint32_t node_index = port.node;
  ports[i] = port;
  // Commit only on the path that actually wrote a port. Every early return above
  // leaves the reservation standing, which is what makes the remainder equal to
  // the invalid-binding count.
  substrate::direct_adult::device_commit_pool_units(
      ecology, substrate::direct_adult::DirectResourcePoolKind::boundary_port, 1u);
  if ((binding.role_mask & static_cast<std::uint32_t>(BoundaryRole::sensor)) != 0u)
    atomicOr(&nodes[node_index].flags, kNodeFlagSensor);
  if ((binding.role_mask & static_cast<std::uint32_t>(BoundaryRole::motor)) != 0u)
    atomicOr(&nodes[node_index].flags, kNodeFlagMotor);
  if ((binding.role_mask & static_cast<std::uint32_t>(BoundaryRole::world_return)) != 0u)
    atomicOr(&nodes[node_index].flags, kNodeFlagWorldReturn);
}

__global__ void census_kernel(const DirectRoute* routes, std::uint32_t route_capacity,
                              const std::uint32_t* in_degree, std::uint32_t node_count,
                              CompileDiagnostics* diagnostics) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < node_count) atomicMax(&diagnostics->maximum_in_degree, in_degree[i]);
  (void)routes;
  (void)route_capacity;
}
