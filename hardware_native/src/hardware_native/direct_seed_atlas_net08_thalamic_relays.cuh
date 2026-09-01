namespace substrate::direct_network {
namespace {

constexpr std::uint32_t kNet08Lineage = 8u;  // matches the atlas's own NET08 numbering.
// The two cortical chemistries are ADJACENT on purpose: they differ only in
// bit 0, so a corridor field can insist on the full value (one partner, a
// network-specific relay) or ignore that one bit (both partners, a connector
// subdivision).  How much of the chemistry a field insists on IS the breadth
// of the subdivision's partner set.
constexpr std::uint32_t kRelayAChemistry = 0xb1u;
constexpr std::uint32_t kRelayBChemistry = 0xb2u;
constexpr std::uint32_t kConnectorChemistry = 0xb3u;
constexpr std::uint32_t kCortexXChemistry = 0xb4u;
constexpr std::uint32_t kCortexYChemistry = 0xb5u;

// Full mask: (chemistry & 0xffffffff) == require_value matches exactly one.
constexpr std::uint32_t kSpecificMask = 0xffffffffu;
// Relaxed mask: (0xb4 & ~1) == (0xb5 & ~1) == 0xb4, so this matches BOTH
// cortical chemistries -- and neither relay (0xb1 & ~1 == 0xb0,
// 0xb2 & ~1 == 0xb2) nor the connector itself (0xb3 & ~1 == 0xb2), so
// field_addresses_partner() still reads it as addressing a partner.
constexpr std::uint32_t kConnectorMask = 0xfffffffeu;

constexpr std::uint32_t kNet08DevelopmentEndTick = 4096u;
// Compact subdivisions, broad cortex: the atlas's "many compact thalamic-like
// hub ecologies" against distributed cortical partners.
constexpr std::uint32_t kSubdivisionReach = 12u;
constexpr std::uint32_t kCortexReach = 24u;
// fanout: high_but_bounded -- high degree per subdivision node, small
// population.  The BOUND is #1178's resource question, not authored here.
constexpr std::uint32_t kNet08Degree = 20u;
constexpr std::uint32_t kNet08Population = 64u;
constexpr std::uint32_t kNet08LongTracts = 12u;
constexpr std::uint32_t kNet08PortReserve = 24u;
// explicit_delay_phase: the two point-to-point relays carry a FAST transport
// cost, the broader integrative connector a SLOWER one -- the same
// `long_tract.extent` carrier NET03/NET05/NET10/NET13 already prove
// (b0a747d01a), applied here to this family's own construction economy.
constexpr std::uint32_t kNet08RelayDelay = 8u;
constexpr std::uint32_t kNet08ConnectorDelay = 20u;

constexpr std::uint32_t kNet08AffinityRadius = 1u << 20;
constexpr std::int32_t kNet08AffinityStrengthQ16 = 1 << 16;

DirectRuleSpecV1 net08_rule(DirectRuleOpcodeV1 opcode, std::uint32_t chemistry) {
  DirectRuleSpecV1 out{};
  out.opcode = opcode;
  out.begin_tick = 0u;
  out.end_tick = kNet08DevelopmentEndTick;
  out.require_mask = kSpecificMask;
  out.require_value = chemistry;
  out.write_mask = kSpecificMask;
  out.write_value = chemistry;
  return out;
}

DirectTerritorySpecV1 net08_territory(std::uint32_t ordinal, std::uint32_t chemistry,
                                      std::uint32_t reach) {
  DirectTerritorySpecV1 out{};
  out.identity.lineage = kNet08Lineage;
  out.identity.axis = 0u;
  out.identity.ordinal = ordinal;
  out.chemotype = chemistry;
  out.reach = reach;
  out.begin_tick = 0u;
  return out;
}

DirectFieldSpecV1 net08_corridor(std::uint32_t source_ordinal, std::uint32_t require_mask,
                                 std::uint32_t require_value) {
  DirectFieldSpecV1 out{};
  out.territory.lineage = kNet08Lineage;
  out.territory.axis = 0u;
  out.territory.ordinal = source_ordinal;
  out.radius = kNet08AffinityRadius;
  out.require_mask = require_mask;
  out.require_value = require_value;
  out.write_mask = kSpecificMask;
  out.write_value = static_cast<std::uint32_t>(kNet08AffinityStrengthQ16);
  out.begin_tick = 0u;
  out.end_tick = kNet08DevelopmentEndTick;
  out.polarity = 0u;  // DevelopmentFieldKind::attract
  return out;
}

}  // namespace

DirectGenomeV1 seed_atlas_net08_thalamic_relays() {
  DirectGenomeV1 genome{};
  genome.header.life_function_version = 2u;
  genome.header.development_end_tick = kNet08DevelopmentEndTick;
  genome.header.matter_budget = 1u << 24;
  // A species constant, not an individual one: two individuals of this species
  // differ through Xi (#1342), never through this value.
  genome.header.development_seed = 0x4e455430u + 8u;  // "NET0" + 8

  genome.territories[genome.header.territory_count++] =
      net08_territory(0u, kRelayAChemistry, kSubdivisionReach);
  genome.territories[genome.header.territory_count++] =
      net08_territory(1u, kRelayBChemistry, kSubdivisionReach);
  genome.territories[genome.header.territory_count++] =
      net08_territory(2u, kConnectorChemistry, kSubdivisionReach);
  genome.territories[genome.header.territory_count++] =
      net08_territory(3u, kCortexXChemistry, kCortexReach);
  genome.territories[genome.header.territory_count++] =
      net08_territory(4u, kCortexYChemistry, kCortexReach);

  // network_specific_relays: each relay insists on the whole cortical
  // chemistry, so it reaches exactly one partner family.
  const std::uint32_t relay_a_field = genome.header.field_count++;
  genome.fields[relay_a_field] = net08_corridor(0u, kSpecificMask, kCortexXChemistry);
  const std::uint32_t relay_b_field = genome.header.field_count++;
  genome.fields[relay_b_field] = net08_corridor(1u, kSpecificMask, kCortexYChemistry);

  // connector_subdivisions: the SAME mechanism, insisting on less.  This is the
  // whole of the difference between a relay and a connector.
  const std::uint32_t connector_field = genome.header.field_count++;
  genome.fields[connector_field] = net08_corridor(2u, kConnectorMask, kCortexXChemistry);

  const std::uint32_t chemistries[] = {kRelayAChemistry, kRelayBChemistry, kConnectorChemistry,
                                       kCortexXChemistry, kCortexYChemistry};
  for (const std::uint32_t chemistry : chemistries) {
    const bool subdivision = chemistry == kRelayAChemistry || chemistry == kRelayBChemistry ||
                             chemistry == kConnectorChemistry;
    DirectRuleSpecV1 grow = net08_rule(DirectRuleOpcodeV1::extend, chemistry);
    grow.extent = subdivision ? kSubdivisionReach : kCortexReach;
    grow.child_slot = kNet08Degree;
    grow.branch_count = kNet08Population;
    genome.rules[genome.header.rule_count++] = grow;

    if (subdivision) {
      const bool is_connector = chemistry == kConnectorChemistry;
      DirectRuleSpecV1 tract = net08_rule(DirectRuleOpcodeV1::long_tract, chemistry);
      tract.branch_count = kNet08LongTracts;
      tract.field_index = (chemistry == kRelayAChemistry)
                              ? relay_a_field
                              : (chemistry == kRelayBChemistry) ? relay_b_field : connector_field;
      tract.extent = is_connector ? kNet08ConnectorDelay : kNet08RelayDelay;
      genome.rules[genome.header.rule_count++] = tract;
    }

    DirectRuleSpecV1 ports = net08_rule(DirectRuleOpcodeV1::endogenous_source, chemistry);
    ports.extent = kNet08PortReserve;
    genome.rules[genome.header.rule_count++] = ports;
  }
  return genome;
}

}  // namespace substrate::direct_network
