// Included inside direct_network_life_function.cu's anonymous namespace after
// the arena-hash section and before the canonical arena-layout include. Owns host
// AOT compilation of the born organism's resident constructor ecology: rule
// admission across the birth handoff, recipe cell/edge relation derivation,
// the bounded chemotype closure, and resident field lowering.
DIRECT_NETWORK_HD inline std::uint32_t apply_resident_chemotype_write(
    std::uint32_t current, const ResidentConstructorRule& rule) {
  return (current & ~rule.write_mask) | (rule.write_value & rule.write_mask);
}

std::vector<ResidentConstructorRule> compile_resident_rules(const GammaV1& gamma) {
  std::vector<ResidentConstructorRule> out;
  out.reserve(gamma.header.rule_count);
  const std::uint32_t handoff = gamma.header.development_end_tick;
  for (std::uint32_t i = 0; i < gamma.header.rule_count; ++i) {
    const ConstructionRule& rule = gamma.rules[i];
    if ((rule.flags & kRuleFlagPostBirthResident) == 0u) continue;
    // Absolute developmental time remains authoritative after bulk AOT
    // compilation. A rule whose finite window ended before birth cannot be
    // resurrected merely because it carries the resident-capable flag.
    if (rule.end_tick != 0u && rule.end_tick <= handoff) continue;
    ResidentConstructorRule resident{};
    resident.opcode = rule.opcode;
    resident.flags = rule.flags;
    resident.field_index = rule.field;
    resident.begin_age = rule.begin_tick > handoff ? rule.begin_tick - handoff : 0u;
    resident.end_age = rule.end_tick == 0u ? 0xffffffffu : rule.end_tick - handoff;
    if (rule.opcode == RuleOpcode::mature) {
      resident.critical_begin_age =
          rule.minimum_age > handoff ? rule.minimum_age - handoff : 0u;
      resident.critical_end_age = rule.maximum_age == 0u
          ? 0xffffffffu
          : (rule.maximum_age < handoff ? 0u : rule.maximum_age - handoff);
      if (rule.maximum_age != 0u && rule.maximum_age < handoff)
        resident.critical_begin_age = 1u;
    }
    resident.threshold_q16 = rule.threshold_q32 >> 16;
    resident.extent = rule.extent;
    resident.branch_count = rule.branch_count;
    resident.require_mask = rule.require_mask;
    resident.require_value = rule.require_value;
    resident.write_mask = rule.write_mask;
    resident.write_value = rule.write_value;
    resident.source_rule_index = i;
    out.push_back(resident);
  }
  return out;
}

struct ResidentRecipeBuild {
  std::vector<ResidentRecipeCell> cells;
  std::vector<ResidentRecipeEdge> edges;
  std::vector<ResidentRecipeRange> ranges;
  std::vector<std::uint16_t> indices;
};

bool recipe_growth_opcode(RuleOpcode opcode) {
  return opcode == RuleOpcode::branch || opcode == RuleOpcode::extend ||
         opcode == RuleOpcode::repair || opcode == RuleOpcode::long_tract ||
         opcode == RuleOpcode::fuse;
}

bool derive_recipe_relation(const ResidentConstructorRule& source,
                            const ResidentConstructorRule& target,
                            ResidentRecipeRelation* relation,
                            std::int32_t* weight_q16) {
  const std::uint32_t shared_require = source.require_mask & target.require_mask;
  if ((source.require_value & shared_require) != (target.require_value & shared_require)) return false;
  const bool same_field = source.field_index != kInvalidIndex &&
                          source.field_index == target.field_index;
  if (source.opcode == RuleOpcode::retract && recipe_growth_opcode(target.opcode)) {
    // Retraction is anti-growth even when the two recipes are driven by
    // different fields. Shared-field antagonism is stronger, but a generic
    // retraction recipe must never become a positive trigger simply because
    // both recipes happen to write the same chemotype.
    *relation = ResidentRecipeRelation::inhibit;
    *weight_q16 = same_field ? -(1 << 14) : -(1 << 12);
    return true;
  }
  if (source.opcode == RuleOpcode::repair && recipe_growth_opcode(target.opcode)) {
    *relation = ResidentRecipeRelation::repair;
    *weight_q16 = 1 << 14;
    return true;
  }
  const bool covers_required = (source.write_mask & target.require_mask) == target.require_mask;
  const bool writes_required = covers_required &&
      (source.write_value & target.require_mask) == target.require_value;
  if (writes_required && source.write_mask != 0u) {
    *relation = ResidentRecipeRelation::trigger;
    *weight_q16 = 1 << 13;
    return true;
  }
  if (source.end_age != 0xffffffffu && target.begin_age >= source.end_age &&
      target.begin_age - source.end_age <= 64u) {
    *relation = ResidentRecipeRelation::temporal;
    *weight_q16 = 1 << 12;
    return true;
  }
  if (same_field) {
    *relation = ResidentRecipeRelation::shared_field;
    *weight_q16 = 1 << 11;
    return true;
  }
  return false;
}

ResidentRecipeBuild compile_resident_recipe_network(
    const GammaV1& gamma, const std::vector<ResidentConstructorRule>& rules) {
  ResidentRecipeBuild build{};
  if (rules.size() > 0xffffu) {
    throw std::runtime_error("resident recipe-cell index exceeds compact uint16 ABI");
  }
  build.cells.resize(rules.size());
  build.ranges.resize(gamma.header.seed_count);

  for (std::uint32_t i = 0; i < rules.size(); ++i) {
    ResidentRecipeCell cell{};
    std::uint64_t logical_id = exact_history_fold_word(0x7265636970657030ull, i);
    logical_id = exact_history_fold_word(logical_id, rules[i].source_rule_index);
    logical_id = exact_history_fold_word(logical_id, static_cast<std::uint32_t>(rules[i].opcode));
    cell.logical_recipe_id = logical_id == 0u ? 1u : logical_id;
    cell.rule_index = i;
    cell.edge_offset = static_cast<std::uint32_t>(build.edges.size());
    cell.support_q16 = 1 << 15;
    cell.credit_q16 = 0;
    if (!initialize_resident_recipe_update_ir(&cell)) {
      throw std::runtime_error("default resident Recipe IR is invalid");
    }
    cell.revision_identity = resident_recipe_revision_identity(
        cell.logical_recipe_id, 0u, 0u, 0u, cell.support_q16, cell.credit_q16);
    std::uint32_t edge_count = 0u;
    for (std::uint32_t j = 0; j < rules.size() && edge_count < kMaxRecipeEdgesPerCell; ++j) {
      if (i == j) continue;
      ResidentRecipeRelation relation{};
      std::int32_t weight_q16 = 0;
      if (!derive_recipe_relation(rules[i], rules[j], &relation, &weight_q16)) continue;
      ResidentRecipeEdge edge{};
      edge.source_cell = static_cast<std::uint16_t>(i);
      edge.target_cell = static_cast<std::uint16_t>(j);
      edge.relation = relation;
      edge.weight_q16 = weight_q16;
      edge.field_index = rules[i].field_index;
      build.edges.push_back(edge);
      ++edge_count;
    }
    cell.edge_count = static_cast<std::uint16_t>(edge_count);
    build.cells[i] = cell;
  }

  constexpr std::size_t kChemotypeClosureStateLimit = 16384u;
  for (std::uint32_t seed_index = 0; seed_index < gamma.header.seed_count; ++seed_index) {
    ResidentRecipeRange range{};
    range.index_offset = static_cast<std::uint32_t>(build.indices.size());

    std::vector<std::uint32_t> reachable_chemotypes;
    reachable_chemotypes.reserve(std::min<std::size_t>(rules.size() + 1u,
                                                       kChemotypeClosureStateLimit));
    std::unordered_set<std::uint32_t> seen_chemotypes;
    seen_chemotypes.reserve(std::min<std::size_t>((rules.size() + 1u) * 2u,
                                                  kChemotypeClosureStateLimit));
    const std::uint32_t birth_chemotype = gamma.seeds[seed_index].chemistry;
    reachable_chemotypes.push_back(birth_chemotype);
    seen_chemotypes.insert(birth_chemotype);
    std::vector<bool> reachable_cells(rules.size(), false);

    for (std::size_t state_index = 0; state_index < reachable_chemotypes.size(); ++state_index) {
      const std::uint32_t chemistry = reachable_chemotypes[state_index];
      for (std::uint32_t cell = 0; cell < rules.size(); ++cell) {
        const ResidentConstructorRule& rule = rules[cell];
        if (!chemistry_matches(chemistry, rule.require_mask, rule.require_value)) continue;
        reachable_cells[cell] = true;
        const std::uint32_t next = apply_resident_chemotype_write(chemistry, rule);
        if (next == chemistry || !seen_chemotypes.insert(next).second) continue;
        if (reachable_chemotypes.size() >= kChemotypeClosureStateLimit) {
          throw std::runtime_error(
              "resident chemotype closure exceeds bounded phenotype-state limit");
        }
        reachable_chemotypes.push_back(next);
      }
    }

    for (std::uint32_t cell = 0; cell < rules.size(); ++cell) {
      if (!reachable_cells[cell]) continue;
      if (range.index_count == 0xffffu) {
        throw std::runtime_error("territory resident recipe index exceeds compact uint16 ABI");
      }
      build.indices.push_back(static_cast<std::uint16_t>(cell));
      ++range.index_count;
    }
    build.ranges[seed_index] = range;
  }
  return build;
}

struct ResidentFieldBindingBuild {
  std::vector<ResidentFieldRange> ranges;
  std::vector<std::uint16_t> indices;
};

ResidentFieldBindingBuild compile_resident_field_bindings(
    const GammaV1& gamma, const std::vector<ResidentConstructorRule>& rules,
    const ResidentRecipeBuild& recipes) {
  ResidentFieldBindingBuild build{};
  build.ranges.resize(gamma.header.seed_count);
  for (std::uint32_t territory = 0u; territory < gamma.header.seed_count; ++territory) {
    ResidentFieldRange range{};
    range.index_offset = static_cast<std::uint32_t>(build.indices.size());
    std::vector<std::uint16_t> fields;
    const ResidentRecipeRange& recipe_range = recipes.ranges[territory];
    fields.reserve(recipe_range.index_count);
    for (std::uint32_t i = 0u; i < recipe_range.index_count; ++i) {
      const std::uint32_t cell_index = recipes.indices[recipe_range.index_offset + i];
      const std::uint32_t rule_index = recipes.cells[cell_index].rule_index;
      const std::uint32_t field_index = rules[rule_index].field_index;
      if (field_index < gamma.header.field_count)
        fields.push_back(static_cast<std::uint16_t>(field_index));
    }
    std::sort(fields.begin(), fields.end());
    fields.erase(std::unique(fields.begin(), fields.end()), fields.end());
    range.index_count = static_cast<std::uint16_t>(fields.size());
    build.indices.insert(build.indices.end(), fields.begin(), fields.end());
    build.ranges[territory] = range;
  }
  return build;
}

std::vector<ResidentDevelopmentField> compile_resident_fields(const GammaV1& gamma) {
  std::vector<ResidentDevelopmentField> out(gamma.header.field_count);
  for (std::uint32_t i = 0; i < gamma.header.field_count; ++i) {
    const FieldBlock& field = gamma.fields[i];
    ResidentDevelopmentField resident{};
    resident.center[0] = static_cast<std::int32_t>(field.center[0]);
    resident.center[1] = static_cast<std::int32_t>(field.center[1]);
    resident.center[2] = static_cast<std::int32_t>(field.center[2]);
    resident.radius = field.radius;
    resident.strength_q16 = field_strength_q16(field);
    resident.decay_q16_per_tick =
        field_decay_magnitude_q16_per_tick(field_decay_class(field));
    resident.kind = field_kind(field);
    resident.require_mask = field.require_mask;
    resident.require_value = field.require_value;
    resident.begin_tick = field.begin_tick;
    resident.end_tick = field.end_tick;
    out[i] = resident;
  }
  return out;
}
