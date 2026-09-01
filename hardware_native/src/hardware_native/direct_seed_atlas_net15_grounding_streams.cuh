namespace substrate::direct_network {
namespace {

constexpr std::uint32_t kNet15Lineage = 15u;  // matches the atlas's own NET15 numbering.
constexpr std::uint32_t kNet15SensorEntryChemistry = 0x91u;
constexpr std::uint32_t kNet15ObjectIdentityChemistry = 0x92u;
constexpr std::uint32_t kNet15ActionAffordanceChemistry = 0x94u;

constexpr std::uint32_t kNet15DevelopmentEndTick = 4096u;

// object_identity ("what"): the atlas's separate_what_vs_how_where_processing_
// biases term, read as broad integrative reach and low fanout -- the same
// shape NET02's temporal_integration and NET05's heteromodal territories use
// for convergent, slower-changing content.
constexpr std::uint32_t kNet15IdentityReach = 22u;
constexpr std::uint32_t kNet15IdentityDegree = 7u;

// action_affordance ("how/where"): action_adjacent_recurrent_territories,
// read as NET00's own strong_local_recurrence shape -- tight reach, high
// per-node degree, many routes all of them near.
constexpr std::uint32_t kNet15AffordanceReach = 8u;
constexpr std::uint32_t kNet15AffordanceDegree = 14u;

// sensor_entry: the port through which body events arrive, not itself a
// processing bias -- the species table's own uniform NET15 baseline.
constexpr std::uint32_t kNet15EntryReach = 14u;
constexpr std::uint32_t kNet15EntryDegree = 10u;

// Population/dense_width/long_tracts match the species table's own NET15 row
// so a reader comparing the generic and the dedicated genome does not find
// them disagreeing on scale, only on the per-territory bias the table's one
// row per family cannot express.
constexpr std::uint32_t kNet15Population = 192u;
constexpr std::uint32_t kNet15DenseWidth = 32u;
constexpr std::uint32_t kNet15LongTracts = 10u;
constexpr std::uint32_t kNet15PortReserve = 24u;

constexpr std::uint32_t kNet15AffinityRadius = 1u << 20;
constexpr std::int32_t kNet15AffinityStrengthQ16 = 1 << 16;

DirectRuleSpecV1 net15_rule(DirectRuleOpcodeV1 opcode, std::uint32_t chemistry) {
  DirectRuleSpecV1 out{};
  out.opcode = opcode;
  out.begin_tick = 0u;
  out.end_tick = kNet15DevelopmentEndTick;
  out.require_mask = 0xffffffffu;
  out.require_value = chemistry;
  out.write_mask = 0xffffffffu;
  out.write_value = chemistry;
  return out;
}

DirectTerritorySpecV1 net15_territory(std::uint32_t ordinal, std::uint32_t chemistry,
                                      std::uint32_t reach) {
  DirectTerritorySpecV1 out{};
  out.identity.lineage = kNet15Lineage;
  out.identity.axis = 0u;
  out.identity.ordinal = ordinal;
  out.chemotype = chemistry;
  out.reach = reach;
  out.begin_tick = 0u;
  return out;
}

// One directed corridor field, owned by `owner_ordinal`, seeking
// `partner_chemistry`. Same construction every corridor on this genome uses:
// a generous radius so affinity is expressed by chemistry rather than a
// guessed coordinate, and inert for the owner's own placement since
// chemistry_matches fails against the owner's own chemotype.
DirectFieldSpecV1 net15_corridor(std::uint32_t owner_ordinal, std::uint32_t partner_chemistry) {
  DirectFieldSpecV1 out{};
  out.territory.lineage = kNet15Lineage;
  out.territory.axis = 0u;
  out.territory.ordinal = owner_ordinal;
  out.radius = kNet15AffinityRadius;
  out.require_mask = 0xffffffffu;
  out.require_value = partner_chemistry;
  out.write_mask = 0xffffffffu;
  out.write_value = static_cast<std::uint32_t>(kNet15AffinityStrengthQ16);
  out.begin_tick = 0u;
  out.end_tick = kNet15DevelopmentEndTick;
  out.polarity = 0u;  // DevelopmentFieldKind::attract
  return out;
}

}  // namespace

DirectGenomeV1 seed_atlas_net15_grounding_streams() {
  DirectGenomeV1 genome{};
  genome.header.life_function_version = 2u;
  genome.header.development_end_tick = kNet15DevelopmentEndTick;
  genome.header.matter_budget = 1u << 24;
  // A species constant, not an individual one: two individuals of this
  // species differ through Xi (#1342), never through this value.
  genome.header.development_seed = 0x4e455430u + 15u;  // "NET0" + 15

  genome.territories[genome.header.territory_count++] =
      net15_territory(0u, kNet15SensorEntryChemistry, kNet15EntryReach);
  genome.territories[genome.header.territory_count++] =
      net15_territory(1u, kNet15ObjectIdentityChemistry, kNet15IdentityReach);
  genome.territories[genome.header.territory_count++] =
      net15_territory(2u, kNet15ActionAffordanceChemistry, kNet15AffordanceReach);

  // crossmodal_binding_corridors, authored bidirectionally per the atlas's own
  // recurrent_sensorimotor_loops human_basis term: sensor_entry forks to both
  // streams, and both streams loop back -- four ordinary corridor fields, the
  // same construction NET01's back-projections used one hop further down a
  // different motif.
  const std::uint32_t entry_to_identity_field = genome.header.field_count++;
  genome.fields[entry_to_identity_field] =
      net15_corridor(0u, kNet15ObjectIdentityChemistry);
  const std::uint32_t identity_to_entry_field = genome.header.field_count++;
  genome.fields[identity_to_entry_field] =
      net15_corridor(1u, kNet15SensorEntryChemistry);
  const std::uint32_t entry_to_affordance_field = genome.header.field_count++;
  genome.fields[entry_to_affordance_field] =
      net15_corridor(0u, kNet15ActionAffordanceChemistry);
  const std::uint32_t affordance_to_entry_field = genome.header.field_count++;
  genome.fields[affordance_to_entry_field] =
      net15_corridor(2u, kNet15SensorEntryChemistry);

  const std::uint32_t chemistries[] = {kNet15SensorEntryChemistry, kNet15ObjectIdentityChemistry,
                                       kNet15ActionAffordanceChemistry};
  const std::uint32_t reaches[] = {kNet15EntryReach, kNet15IdentityReach, kNet15AffordanceReach};
  const std::uint32_t degrees[] = {kNet15EntryDegree, kNet15IdentityDegree, kNet15AffordanceDegree};
  for (std::uint32_t i = 0u; i < 3u; ++i) {
    const std::uint32_t chemistry = chemistries[i];

    DirectRuleSpecV1 grow = net15_rule(DirectRuleOpcodeV1::extend, chemistry);
    grow.extent = reaches[i];
    grow.child_slot = degrees[i];
    grow.branch_count = kNet15Population;
    genome.rules[genome.header.rule_count++] = grow;

    DirectRuleSpecV1 dense = net15_rule(DirectRuleOpcodeV1::fuse, chemistry);
    dense.branch_count = kNet15DenseWidth;
    genome.rules[genome.header.rule_count++] = dense;

    // sensor_entry owns TWO corridor families (one per stream it forks to);
    // object_identity and action_affordance each own one (their loop back).
    if (chemistry == kNet15SensorEntryChemistry) {
      DirectRuleSpecV1 to_identity = net15_rule(DirectRuleOpcodeV1::long_tract, chemistry);
      to_identity.branch_count = kNet15LongTracts;
      to_identity.field_index = entry_to_identity_field;
      genome.rules[genome.header.rule_count++] = to_identity;

      DirectRuleSpecV1 to_affordance = net15_rule(DirectRuleOpcodeV1::long_tract, chemistry);
      to_affordance.branch_count = kNet15LongTracts;
      to_affordance.field_index = entry_to_affordance_field;
      genome.rules[genome.header.rule_count++] = to_affordance;
    } else if (chemistry == kNet15ObjectIdentityChemistry) {
      DirectRuleSpecV1 back = net15_rule(DirectRuleOpcodeV1::long_tract, chemistry);
      back.branch_count = kNet15LongTracts;
      back.field_index = identity_to_entry_field;
      genome.rules[genome.header.rule_count++] = back;
    } else {
      DirectRuleSpecV1 back = net15_rule(DirectRuleOpcodeV1::long_tract, chemistry);
      back.branch_count = kNet15LongTracts;
      back.field_index = affordance_to_entry_field;
      genome.rules[genome.header.rule_count++] = back;
    }

    DirectRuleSpecV1 ports = net15_rule(DirectRuleOpcodeV1::endogenous_source, chemistry);
    ports.extent = kNet15PortReserve;
    genome.rules[genome.header.rule_count++] = ports;
  }
  return genome;
}

}  // namespace substrate::direct_network
