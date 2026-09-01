namespace substrate::direct_network {
namespace {

constexpr std::uint32_t kNet02Lineage = 2u;  // NET00 lineage 0, NET01 lineage 1.
constexpr std::uint32_t kTemporalIntegrationChemistry = 0x51u;
constexpr std::uint32_t kGroundingConvergenceChemistry = 0x52u;
constexpr std::uint32_t kFrontalContextChemistry = 0x54u;

constexpr std::uint32_t kNet02DevelopmentEndTick = 4096u;
// Broader reach and lower per-node degree than NET01's fast dorsal sequence --
// the atlas motif is "continuous multi-timescale integration", not one tight
// slot: values match the canonical species table's own NET02 row so the two
// readings of the same atlas block do not disagree.
constexpr std::uint32_t kNet02Reach = 24u;
constexpr std::uint32_t kNet02Degree = 8u;
constexpr std::uint32_t kNet02Population = 192u;
constexpr std::uint32_t kNet02DenseWidth = 32u;
constexpr std::uint32_t kNet02LongTracts = 12u;
constexpr std::uint32_t kNet02PortReserve = 24u;

// Same construction as NET01's direct corridor: generous radius, chemistry
// does the matching, no host-authored coordinate.
constexpr std::uint32_t kNet02AffinityRadius = 1u << 20;
constexpr std::int32_t kNet02AffinityStrengthQ16 = 1 << 16;

DirectRuleSpecV1 net02_rule(DirectRuleOpcodeV1 opcode, std::uint32_t chemistry) {
  DirectRuleSpecV1 out{};
  out.opcode = opcode;
  out.begin_tick = 0u;
  out.end_tick = kNet02DevelopmentEndTick;
  out.require_mask = 0xffffffffu;
  out.require_value = chemistry;
  out.write_mask = 0xffffffffu;
  out.write_value = chemistry;
  return out;
}

DirectTerritorySpecV1 net02_territory(std::uint32_t ordinal, std::uint32_t chemistry) {
  DirectTerritorySpecV1 out{};
  out.identity.lineage = kNet02Lineage;
  out.identity.axis = 0u;
  out.identity.ordinal = ordinal;
  out.chemotype = chemistry;
  out.reach = kNet02Reach;
  out.begin_tick = 0u;
  return out;
}

}  // namespace

DirectGenomeV1 seed_atlas_net02_ventral_stream() {
  DirectGenomeV1 genome{};
  genome.header.life_function_version = 2u;
  genome.header.development_end_tick = kNet02DevelopmentEndTick;
  genome.header.matter_budget = 1u << 24;
  // A species constant, not an individual one: two individuals of this species
  // differ through Xi (#1342), never through this value.
  genome.header.development_seed = 0x56454e32u;  // "VEN2" -- ventral, arbitrary as NET1's

  genome.territories[genome.header.territory_count++] =
      net02_territory(0u, kTemporalIntegrationChemistry);
  genome.territories[genome.header.territory_count++] =
      net02_territory(1u, kGroundingConvergenceChemistry);
  genome.territories[genome.header.territory_count++] =
      net02_territory(2u, kFrontalContextChemistry);

  // The direct ventral corridor (github #1268's `crossmodal_territory_affinity`,
  // primary leg -- extreme_capsule/IFOF/uncinate): owned by temporal_integration,
  // attract, seeks inferior_frontal_context_control's chemistry rather than its
  // own, exactly as NET01's direct dorsal corridor does.
  DirectFieldSpecV1 direct_corridor{};
  direct_corridor.territory.lineage = kNet02Lineage;
  direct_corridor.territory.axis = 0u;
  direct_corridor.territory.ordinal = 0u;  // temporal_integration
  direct_corridor.radius = kNet02AffinityRadius;
  direct_corridor.require_mask = 0xffffffffu;
  direct_corridor.require_value = kFrontalContextChemistry;
  direct_corridor.write_mask = 0xffffffffu;
  direct_corridor.write_value = static_cast<std::uint32_t>(kNet02AffinityStrengthQ16);
  direct_corridor.begin_tick = 0u;
  direct_corridor.end_tick = kNet02DevelopmentEndTick;
  direct_corridor.polarity = 0u;  // DevelopmentFieldKind::attract
  const std::uint32_t direct_corridor_index = genome.header.field_count++;
  genome.fields[direct_corridor_index] = direct_corridor;

  // The PARALLEL ventral corridor (`multiple_parallel_ventral_tract_corridors`):
  // the same owner territory, a different named partner --
  // multimodal_grounding_convergence. Authored the same way `3ddf739421`
  // retrofitted NET01's indirect leg: an ordinary second field plus an
  // ordinary second long_tract rule, no new opcode or descriptor. A long_tract
  // rule IS a corridor family now, so this is a second family on
  // temporal_integration, not a second address fighting the first for one slot.
  DirectFieldSpecV1 parallel_corridor = direct_corridor;
  parallel_corridor.require_value = kGroundingConvergenceChemistry;
  const std::uint32_t parallel_corridor_index = genome.header.field_count++;
  genome.fields[parallel_corridor_index] = parallel_corridor;

  const std::uint32_t chemistries[] = {kTemporalIntegrationChemistry,
                                       kGroundingConvergenceChemistry, kFrontalContextChemistry};
  for (const std::uint32_t chemistry : chemistries) {
    DirectRuleSpecV1 grow = net02_rule(DirectRuleOpcodeV1::extend, chemistry);
    grow.extent = kNet02Reach;
    grow.child_slot = kNet02Degree;
    grow.branch_count = kNet02Population;
    genome.rules[genome.header.rule_count++] = grow;

    DirectRuleSpecV1 dense = net02_rule(DirectRuleOpcodeV1::fuse, chemistry);
    dense.branch_count = kNet02DenseWidth;
    genome.rules[genome.header.rule_count++] = dense;

    DirectRuleSpecV1 tract = net02_rule(DirectRuleOpcodeV1::long_tract, chemistry);
    tract.branch_count = kNet02LongTracts;
    if (chemistry == kTemporalIntegrationChemistry) {
      tract.field_index = direct_corridor_index;
    }
    genome.rules[genome.header.rule_count++] = tract;

    if (chemistry == kTemporalIntegrationChemistry) {
      DirectRuleSpecV1 parallel = net02_rule(DirectRuleOpcodeV1::long_tract, chemistry);
      parallel.branch_count = kNet02LongTracts;
      parallel.field_index = parallel_corridor_index;
      genome.rules[genome.header.rule_count++] = parallel;
    }

    DirectRuleSpecV1 ports = net02_rule(DirectRuleOpcodeV1::endogenous_source, chemistry);
    ports.extent = kNet02PortReserve;
    genome.rules[genome.header.rule_count++] = ports;
  }
  return genome;
}

}  // namespace substrate::direct_network
