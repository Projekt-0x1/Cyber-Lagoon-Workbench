// Included inside direct_network_life_function.cu's anonymous namespace after
// the field, plan, geometry, and node-selection primitives are defined.

DIRECT_NETWORK_HD inline bool prenatal_long_tract_rule_applies(
    const GammaV1& gamma, const TerritoryPlan& source, const ConstructionRule& rule) {
  if (rule.opcode != RuleOpcode::long_tract ||
      !chemistry_matches(source.chemotype, rule.require_mask, rule.require_value))
    return false;
  const SeedBlock seed = gamma.seeds[source.seed_index];
  const std::uint32_t rule_end = rule.end_tick == 0u ? 0xffffffffu : rule.end_tick;
  return seed.begin_tick < gamma.header.development_end_tick &&
         rule.begin_tick < gamma.header.development_end_tick && rule_end > seed.begin_tick;
}

// A territory's authored long_tract rules are already a bounded flat corridor
// list. Scanning them avoids a second fixed family table and therefore has no
// independent partner ceiling below kMaxRules.
DIRECT_NETWORK_HD inline std::uint32_t corridor_rule_for_local(
    const GammaV1& gamma, const TerritoryPlan& source, std::uint32_t source_local) {
  std::uint32_t first_local = 0u;
  for (std::uint32_t rule_index = 0u; rule_index < gamma.header.rule_count; ++rule_index) {
    const ConstructionRule rule = gamma.rules[rule_index];
    if (!prenatal_long_tract_rule_applies(gamma, source, rule)) continue;
    const std::uint32_t count =
        clamp_u32(rule.branch_count == 0u ? 1u : rule.branch_count, 1u, 64u);
    if (source_local >= first_local && source_local - first_local < count) return rule_index;
    first_local += count;
  }
  return kInvalidIndex;
}

DIRECT_NETWORK_HD inline std::uint32_t choose_long_tract_target(
    const GammaV1& gamma, const TerritoryPlan* plans, std::uint32_t plan_count,
    std::uint32_t source_plan, const DirectNode* nodes, std::uint32_t source_node,
    std::uint32_t corridor_rule_index, bool* steered_out = nullptr,
    std::int32_t* score_out = nullptr, bool* gain_participates_out = nullptr) {
  std::int32_t best_score = INT32_MIN;
  std::uint32_t best_target = source_node;
  const TerritoryPlan source = plans[source_plan];
  const std::uint32_t source_local = source_node - source.node_offset;
  const std::uint32_t logical_tick = logical_node_birth_tick(gamma, source, source_local);
  const bool has_rule = corridor_rule_index < gamma.header.rule_count;
  const std::uint32_t rule_field = has_rule ? gamma.rules[corridor_rule_index].field : kInvalidField;
  const bool per_rule = rule_field < gamma.header.field_count;
  const bool addressed = per_rule ? field_addresses_partner(gamma, source.chemotype, rule_field)
                                  : declares_partner_affinity(gamma, source);
  bool any_partner_matched = false;
  if (steered_out != nullptr) *steered_out = false;
  for (std::uint32_t p = 0; p < plan_count; ++p) {
    if (p == source_plan || plans[p].active == 0u) continue;
    const TerritoryPlan target_plan = plans[p];
    const std::uint64_t h = mix64(gamma.header.development_seed ^
                                  (static_cast<std::uint64_t>(source_node) << 32) ^ p);
    const std::uint32_t target = target_plan.node_offset +
        static_cast<std::uint32_t>(h % target_plan.node_count);
    const DirectNode& target_node = nodes[target];
    const std::int32_t affinity =
        per_rule ? clamp_i32(field_influence_q16(gamma.fields, gamma.header.field_count,
                                                 rule_field, target_node.coordinate,
                                                 target_plan.chemotype, logical_tick),
                             -(16 << 16), 16 << 16)
                 : partner_affinity_q16(gamma, source, target_plan.chemotype,
                                        target_node.coordinate, logical_tick);
    if (addressed && affinity == 0) continue;
    if (affinity != 0) any_partner_matched = true;
    std::int32_t score = geometry_route_score_q16(nodes[source_node], target_node);
    score += combined_developmental_score_q16(gamma, source, target_node.coordinate, logical_tick);
    score += affinity;
    if (score > best_score || (score == best_score && target < best_target)) {
      best_score = score;
      best_target = target;
      if (steered_out != nullptr) *steered_out = affinity != 0;
    }
  }
  if (addressed && !any_partner_matched) return kInvalidIndex;
  if (score_out != nullptr) *score_out = best_score;
  if (gain_participates_out != nullptr) {
    const std::uint32_t used_field = per_rule ? rule_field : source.attract_field;
    *gain_participates_out = used_field < gamma.header.field_count &&
                             field_gain_participates(gamma.fields[used_field]);
  }
  return best_target;
}

DIRECT_NETWORK_HD inline bool explicit_tract_delay_law(const DirectTractDelayLawV1& law) {
  return law.initial_min_ticks != 0u;
}

DIRECT_NETWORK_HD inline std::uint32_t sample_tract_delay(
    const GammaV1& gamma, const TerritoryPlan& source, std::uint32_t source_local,
    std::uint32_t rule_index, std::uint32_t minimum, std::uint32_t maximum,
    std::uint64_t domain) {
  const std::uint32_t span = maximum - minimum + 1u;
  const std::uint64_t draw = mix64(
      gamma.header.development_seed ^ domain ^
      (static_cast<std::uint64_t>(source.seed_index) << 48u) ^
      (static_cast<std::uint64_t>(source_local) << 16u) ^ rule_index);
  return minimum + static_cast<std::uint32_t>(draw % span);
}

DIRECT_NETWORK_HD inline std::uint32_t initial_tract_delay(
    const GammaV1& gamma, const TerritoryPlan& source, std::uint32_t source_local,
    std::uint32_t rule_index, const DirectTractDelayLawV1& law,
    const DirectNode& source_node, const DirectNode& target_node) {
  if (!explicit_tract_delay_law(law)) {
    const std::uint32_t geometry = route_delay_from_geometry(source_node, target_node, true);
    const std::uint32_t bias =
        rule_index < gamma.header.rule_count ? gamma.rules[rule_index].extent : 0u;
    return clamp_u32(geometry + bias, 1u, 64u);
  }
  return sample_tract_delay(gamma, source, source_local, rule_index,
                            law.initial_min_ticks, law.initial_max_ticks,
                            0x494e4954444c5930ull);  // "INITDLY0"
}

DIRECT_NETWORK_HD inline std::uint32_t mature_tract_delay(
    const GammaV1& gamma, const TerritoryPlan& source, std::uint32_t source_local,
    std::uint32_t rule_index, const DirectTractDelayLawV1& law) {
  const std::uint32_t minimum =
      law.mature_min_ticks == 0u ? law.initial_min_ticks : law.mature_min_ticks;
  const std::uint32_t maximum =
      law.mature_max_ticks == 0u ? law.initial_max_ticks : law.mature_max_ticks;
  return sample_tract_delay(gamma, source, source_local, rule_index, minimum, maximum,
                            0x4d41545552454430ull);  // "MATURED0"
}
