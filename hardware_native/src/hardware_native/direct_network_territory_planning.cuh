// Included inside direct_network_life_function.cu's anonymous namespace after
// direct_network_developmental_placement_law.cuh. Owns Gamma-rule ->
// TerritoryPlan derivation, the host capacity preflight, and the device
// kernels that emit per-seed counts, offsets, and totals. Kernel launch
// order stays with the Life Function orchestrator.
DIRECT_NETWORK_HD inline TerritoryPlan derive_territory_plan(
    const GammaV1& gamma, std::uint32_t seed_index,
    std::uint32_t route_reserve_per_node) {
  const SeedBlock seed = gamma.seeds[seed_index];
  TerritoryPlan plan{};
  plan.seed_index = seed_index;
  plan.lineage = seed.lineage;
  plan.chemotype = seed.chemistry;
  plan.attract_field = kInvalidField;
  plan.repel_field = kInvalidField;
  plan.resource_field = kInvalidField;
  plan.maturation_field = kInvalidField;
  plan.birth_maturation_field = kInvalidField;
  plan.inhibition_field = kInvalidField;
  plan.repair_field = kInvalidField;
  plan.radius = 32u;
  std::uint32_t referenced_field_words[kDirectMaxFieldsV1 / 32u]{};

  for (std::uint32_t i = 0; i < gamma.header.rule_count; ++i) {
    const ConstructionRule rule = gamma.rules[i];
    if (!chemistry_matches(seed.chemistry, rule.require_mask, rule.require_value)) continue;
    const std::uint32_t rule_end = rule.end_tick == 0u ? 0xffffffffu : rule.end_tick;
    const bool seed_exists_before_birth = seed.begin_tick < gamma.header.development_end_tick;
    const bool expressed_before_birth = seed_exists_before_birth &&
        rule.begin_tick < gamma.header.development_end_tick && rule_end > seed.begin_tick;
    const bool resident_after_birth =
        (rule.flags & kRuleFlagPostBirthResident) != 0u &&
        (rule.end_tick == 0u || rule.end_tick > gamma.header.development_end_tick);
    if (!expressed_before_birth && !resident_after_birth) continue;

    // Maturation polarity is not authority by itself: only a mature rule may
    // bind that field into the born plan. Other field kinds remain generic.
    const bool field_authorized =
        rule.field < gamma.header.field_count &&
        (field_kind(gamma.fields[rule.field]) != DevelopmentFieldKind::maturation ||
         rule.opcode == RuleOpcode::mature);
    if (expressed_before_birth && field_authorized) {
      referenced_field_words[rule.field / 32u] |= 1u << (rule.field % 32u);
    }

    if (!expressed_before_birth) {
      // Preserve developmental preparedness for future resident phases without
      // pretending the future rule already executed before birth.
      plan.flags |= rule.flags;
      if (rule.opcode == RuleOpcode::repair) plan.flags |= kRuleFlagConstructorReserve;
      continue;
    }

    // Authored flags belong to the rule, not to its opcode.  Propagating them
    // once here instead of inside every case is what makes it impossible to add
    // an opcode that silently drops them: `retract` -- the pruning opcode, and
    // so the most natural carrier of a pruning/inhibitory bias -- was added
    // without propagation and could not carry one (github #1315).  No case
    // below reads plan.flags, so this is exactly the previous behaviour for
    // every opcode that already propagated.
    plan.flags |= rule.flags;
    // Strength travels with the flag, and only with it: a rule that does not
    // declare an inhibitory bias cannot set one by carrying a stray threshold.
    if ((rule.flags & kRuleFlagInhibitoryBias) != 0u) {
      const bool explicit_magnitude =
          (rule.flags & kRuleFlagCompetitionMagnitudeAuthored) != 0u;
      const std::uint32_t authored_magnitude = rule.threshold_q32 >> 16;
      if (explicit_magnitude || authored_magnitude != 0u) {
        if (!plan.competition_magnitude_authored) {
          plan.competition_magnitude_authored = 1u;
          plan.competition_strength_q16 = authored_magnitude;
        } else if (plan.competition_strength_q16 != authored_magnitude) {
          plan.authoring_fault |= kTerritoryPlanConflictingCompetitionMagnitude;
        }
      }
    }
    // ... and the share travels with the flag AND with the opcode that grows
    // tissue, since `minimum_age` keeps its corridor-window meaning on
    // `long_tract`.
    if ((rule.flags & kRuleFlagInhibitoryBias) != 0u &&
        (rule.opcode == RuleOpcode::extend || rule.opcode == RuleOpcode::branch)) {
      const bool explicit_density =
          (rule.flags & kRuleFlagCompetitionDensityAuthored) != 0u;
      const std::uint32_t authored_density = rule.minimum_age;
      if (explicit_density || authored_density != 0u) {
        if (!plan.competition_density_authored) {
          plan.competition_density_authored = 1u;
          plan.inhibition_share_denominator = authored_density;
        } else if (plan.inhibition_share_denominator != authored_density) {
          plan.authoring_fault |= kTerritoryPlanConflictingCompetitionDensity;
        }
      }
    }

    switch (rule.opcode) {
      case RuleOpcode::branch:
      case RuleOpcode::extend:
        if (plan.active == 0u) {
          plan.active = 1u;
          plan.node_count = max_u32(64u, rule.branch_count == 0u ? 64u : rule.branch_count);
          plan.sparse_degree = clamp_u32(rule.child_slot == kInvalidIndex ? 4u : rule.child_slot,
                                         2u, kMaxSparseDegree);
          plan.radius = max_u32(8u, rule.extent);
        }
        break;
      case RuleOpcode::long_tract: {
        const std::uint32_t requested_tracts =
            clamp_u32(rule.branch_count == 0u ? 1u : rule.branch_count, 1u, 64u);
        plan.long_tract_count += requested_tracts;
        plan.flags |= kRuleFlagLongRangePreferred;
        break;
      }
      case RuleOpcode::fuse:
        plan.flags |= kRuleFlagDenseIntegrative;
        if (plan.node_count != 0u) {
          const std::uint32_t requested =
              rule.branch_count == 0u ? min_u32(plan.node_count, 128u) : rule.branch_count;
          plan.dense_width =
              clamp_u32(requested, 16u, min_u32(plan.node_count, kDenseWidthLimit));
          plan.dense_width &= ~15u;
          if (plan.dense_width < 16u) plan.dense_width = 16u;
        }
        break;
      case RuleOpcode::repair:
        plan.flags |= kRuleFlagConstructorReserve;
        break;
      case RuleOpcode::retract:
        break;
      case RuleOpcode::mature:
        break;
      case RuleOpcode::endogenous_source:
        // Flags already propagated above; this opcode still plans no geometry.
        // What it now declares is CAPACITY: `endogenous_source` is the opcode
        // that says a territory interfaces with something outside itself, so
        // its extent is read as the spare port capacity that territory carries
        // for bindings it has not made yet. Every genome that predates this
        // authors the opcode with extent 0, which reduces to the previous
        // behaviour exactly -- the reserve then comes from the host option
        // alone, as it always did.
        plan.port_reserve = max_u32(plan.port_reserve, rule.extent);
        break;
    }
  }

  // Canonical field identity is the sorted set of exact rule->field
  // references. Duplicate references are idempotent and opcode never changes
  // field physics. The fixed bitset is compilation scratch; every hot reader
  // below consumes only bound_field_count entries.
  for (std::uint32_t field_index = 0u; field_index < gamma.header.field_count;
       ++field_index) {
    if ((referenced_field_words[field_index / 32u] &
         (1u << (field_index % 32u))) == 0u)
      continue;
    plan.bound_field_indices[plan.bound_field_count++] =
        static_cast<std::uint16_t>(field_index);
    std::uint32_t* legacy_slot = nullptr;
    switch (field_kind(gamma.fields[field_index])) {
      case DevelopmentFieldKind::attract: legacy_slot = &plan.attract_field; break;
      case DevelopmentFieldKind::repel: legacy_slot = &plan.repel_field; break;
      case DevelopmentFieldKind::resource: legacy_slot = &plan.resource_field; break;
      case DevelopmentFieldKind::maturation: legacy_slot = &plan.maturation_field; break;
      case DevelopmentFieldKind::inhibition: legacy_slot = &plan.inhibition_field; break;
      case DevelopmentFieldKind::repair: legacy_slot = &plan.repair_field; break;
    }
    if (legacy_slot != nullptr && *legacy_slot == kInvalidField)
      *legacy_slot = field_index;
  }
  plan.birth_maturation_field = plan.maturation_field;

  if (plan.active != 0u) {
    const std::uint32_t long_slot = plan.long_tract_count == 0u ? 0u : 1u;
    // The host option is a FLOOR the caller guarantees, never a ceiling on what
    // the species may ask for: an authored reserve raises it, and a genome that
    // authors none is left exactly where it was.
    const std::uint32_t reserve = max_u32(route_reserve_per_node, plan.port_reserve);
    plan.route_capacity_per_node =
        clamp_u32(plan.sparse_degree + long_slot + reserve,
                  plan.sparse_degree + long_slot, kMaxRouteCapacityPerNode);
    plan.long_tract_count = min_u32(plan.long_tract_count, plan.node_count);
    // Keep the plan derivation itself overflow-safe because host preflight uses
    // this same function before it has proved the v0.2 index envelope. The
    // exact 64-bit total is recomputed by preflight_direct_capacity(); after
    // that proof the device planner's uint32 materialization is safe.
    const std::uint64_t active_estimate =
        static_cast<std::uint64_t>(plan.node_count) * plan.sparse_degree + plan.long_tract_count;
    plan.active_route_estimate =
        active_estimate > 0xffffffffull ? 0xffffffffu : static_cast<std::uint32_t>(active_estimate);
  }

  if ((plan.flags & kRuleFlagInhibitoryBias) != 0u) {
    if (!plan.competition_magnitude_authored) {
      // Unauthored magnitude fallback to legacy constant 32768 (kQ16One / 2)
      plan.competition_strength_q16 = 32768u;
    }
    if (plan.competition_density_authored != 0u) {
      if (plan.inhibition_share_denominator == 0u) {
        plan.inhibition_priority_threshold = 0ull;
      } else {
        plan.inhibition_priority_threshold =
            0xffffffffffffffffull / static_cast<std::uint64_t>(plan.inhibition_share_denominator);
      }
    } else {
      // Unauthored zero: fallback to legacy rate 1/5
      plan.inhibition_priority_threshold = 0xffffffffffffffffull / 5ull;
    }
  } else {
    plan.inhibition_priority_threshold = 0ull;
  }

  return plan;
}

void preflight_direct_capacity(const GammaV1& gamma,
                               std::uint32_t route_reserve_per_node) {
  std::uint64_t nodes = 0u;
  std::uint64_t route_slots = 0u;
  std::uint64_t active_routes = 0u;
  std::uint64_t dense_weights = 0u;
  for (std::uint32_t seed_index = 0; seed_index < gamma.header.seed_count; ++seed_index) {
    const TerritoryPlan plan = derive_territory_plan(gamma, seed_index, route_reserve_per_node);
    if (plan.authoring_fault != 0u) {
      if ((plan.authoring_fault & kTerritoryPlanConflictingCompetitionDensity) != 0u) {
        throw std::invalid_argument(
            "direct Gamma contains conflicting territory competition density declarations");
      }
      if ((plan.authoring_fault & kTerritoryPlanConflictingCompetitionMagnitude) != 0u) {
        throw std::invalid_argument(
            "direct Gamma contains conflicting territory competition magnitude declarations");
      }
      throw std::invalid_argument(
          "direct Gamma contains conflicting territory plan declarations");
    }
    if (plan.active == 0u) continue;
    nodes += plan.node_count;
    route_slots += static_cast<std::uint64_t>(plan.node_count) * plan.route_capacity_per_node;
    active_routes += static_cast<std::uint64_t>(plan.node_count) * plan.sparse_degree +
                     plan.long_tract_count;
    if ((plan.flags & kRuleFlagDenseIntegrative) != 0u && plan.dense_width >= 16u) {
      dense_weights += static_cast<std::uint64_t>(plan.dense_width) * plan.dense_width;
    }
  }
  const std::uint64_t u32_limit = 0xffffffffull;
  if (nodes > u32_limit || route_slots > u32_limit || active_routes > u32_limit ||
      dense_weights > u32_limit) {
    throw std::invalid_argument(
        "direct Gamma exceeds the v0.2 32-bit execution-index ABI; split/upgrade the ABI, do not overflow it");
  }
  if (nodes + active_routes + dense_weights > gamma.header.matter_budget) {
    throw std::invalid_argument("direct Gamma exceeds finite matter budget in preflight");
  }
}

__global__ void plan_territories_kernel(const GammaV1* gamma, TerritoryPlan* plans,
                                        std::uint32_t route_reserve_per_node,
                                        std::uint32_t* node_counts,
                                        std::uint32_t* route_capacity_counts,
                                        std::uint32_t* active_route_counts,
                                        std::uint32_t* dense_block_counts,
                                        std::uint32_t* dense_weight_counts) {
  const std::uint32_t seed_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (seed_index >= gamma->header.seed_count) return;
  const TerritoryPlan plan = derive_territory_plan(*gamma, seed_index, route_reserve_per_node);
  plans[seed_index] = plan;
  node_counts[seed_index] = plan.node_count;
  route_capacity_counts[seed_index] = plan.node_count * plan.route_capacity_per_node;
  active_route_counts[seed_index] = plan.active_route_estimate;
  dense_block_counts[seed_index] =
      plan.active != 0u && (plan.flags & kRuleFlagDenseIntegrative) != 0u &&
              plan.dense_width >= 16u
          ? 1u
          : 0u;
  dense_weight_counts[seed_index] = dense_block_counts[seed_index] != 0u
                                        ? plan.dense_width * plan.dense_width
                                        : 0u;
}

__global__ void install_plan_offsets_kernel(TerritoryPlan* plans,
                                            const std::uint32_t* node_offsets,
                                            const std::uint32_t* route_offsets,
                                            const std::uint32_t* active_route_offsets,
                                            const std::uint32_t* dense_block_offsets,
                                            const std::uint32_t* dense_weight_offsets,
                                            std::uint32_t count) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count) return;
  plans[i].node_offset = node_offsets[i];
  plans[i].route_offset = route_offsets[i];
  plans[i].active_route_offset = active_route_offsets[i];
  plans[i].dense_block_offset = dense_block_offsets[i];
  plans[i].dense_weight_offset = dense_weight_offsets[i];
}

__global__ void write_compile_totals_kernel(const TerritoryPlan* plans,
                                            const std::uint32_t* node_offsets,
                                            const std::uint32_t* node_counts,
                                            const std::uint32_t* route_offsets,
                                            const std::uint32_t* route_counts,
                                            const std::uint32_t* active_offsets,
                                            const std::uint32_t* active_counts,
                                            const std::uint32_t* dense_block_offsets,
                                            const std::uint32_t* dense_block_counts,
                                            const std::uint32_t* dense_weight_offsets,
                                            const std::uint32_t* dense_weight_counts,
                                            std::uint32_t count,
                                            CompileTotals* totals) {
  if (blockIdx.x != 0u || count == 0u) return;
  if (threadIdx.x == 0u) {
    const std::uint32_t last = count - 1u;
    totals->node_count = node_offsets[last] + node_counts[last];
    totals->route_capacity = route_offsets[last] + route_counts[last];
    totals->active_route_estimate = active_offsets[last] + active_counts[last];
    totals->dense_block_count = dense_block_offsets[last] + dense_block_counts[last];
    totals->dense_weight_count = dense_weight_offsets[last] + dense_weight_counts[last];
    totals->territory_count = 0u;
  }
  __syncthreads();
  for (std::uint32_t i = threadIdx.x; i < count; i += blockDim.x) {
    if (plans[i].active != 0u) atomicAdd(&totals->territory_count, 1u);
  }
}
