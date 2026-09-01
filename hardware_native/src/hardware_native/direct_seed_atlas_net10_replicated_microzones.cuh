namespace substrate::direct_network {
namespace {

constexpr std::uint32_t kNet10Lineage = 10u;  // matches the atlas's own NET10 numbering.

// The replicated set.  Six units is the smallest count that makes the two
// claims separable: more than one unit per domain (so a lesion can miss a
// SAME-domain sibling), and index-adjacent units in different domains (so a
// lesion can miss a fractured neighbour).
constexpr std::uint32_t kNet10MicrozoneCount = 6u;
constexpr std::uint32_t kNet10MicrozoneChemistryBase = 0xd0u;
constexpr std::uint32_t kDomainAChemistry = 0xc8u;
constexpr std::uint32_t kDomainBChemistry = 0xc9u;
constexpr std::uint32_t kDomainAOrdinal = kNet10MicrozoneCount;      // 6
constexpr std::uint32_t kDomainBOrdinal = kNet10MicrozoneCount + 1u; // 7

constexpr std::uint32_t kNet10DevelopmentEndTick = 4096u;
// The SHARED generic local motif -- written once, identical for every unit.
// "many repeated local microzone-like units" with "short state": compact reach,
// high parallelism, small population per unit.
constexpr std::uint32_t kMicrozoneReach = 8u;
constexpr std::uint32_t kMicrozoneDegree = 12u;
constexpr std::uint32_t kMicrozonePopulation = 48u;
constexpr std::uint32_t kMicrozoneLongTracts = 8u;
constexpr std::uint32_t kMicrozonePortReserve = 12u;
// The cerebral partner domains are broader and fewer.
constexpr std::uint32_t kDomainReach = 28u;
constexpr std::uint32_t kDomainDegree = 8u;
constexpr std::uint32_t kDomainPopulation = 96u;

constexpr std::uint32_t kNet10Mask = 0xffffffffu;
constexpr std::uint32_t kNet10AffinityRadius = 1u << 20;
constexpr std::int32_t kNet10AffinityStrengthQ16 = 1 << 16;

// precise_delay_timing: each unit already authors its OWN long_tract rule
// (its own corridor family, one per unit), so each one's `extent` -- the
// transport-cost carrier b0a747d01a proved, generalized past a two-family
// binary by NET03/d792ebead0 and NET05/b8a5a0b30b -- can already carry a
// value distinct from every other unit's, not just one shared per family.
// A monotonic per-unit sequence, not a per-domain pair, is the atlas's own
// distinction between "precise" and merely "two classes".
constexpr std::uint32_t kMicrozoneDelayBase = 4u;
constexpr std::uint32_t kMicrozoneDelayStep = 8u;

// fractured_noncontiguous_partner_topography: index-adjacent units serve
// DIFFERENT domains, so the units of one domain are noncontiguous in the
// generator's own ordering.
constexpr bool microzone_serves_domain_a(std::uint32_t index) { return (index % 2u) == 0u; }

DirectRuleSpecV1 net10_rule(DirectRuleOpcodeV1 opcode, std::uint32_t chemistry) {
  DirectRuleSpecV1 out{};
  out.opcode = opcode;
  out.begin_tick = 0u;
  out.end_tick = kNet10DevelopmentEndTick;
  out.require_mask = kNet10Mask;
  out.require_value = chemistry;
  out.write_mask = kNet10Mask;
  out.write_value = chemistry;
  return out;
}

DirectTerritorySpecV1 net10_territory(std::uint32_t ordinal, std::uint32_t chemistry,
                                      std::uint32_t reach) {
  DirectTerritorySpecV1 out{};
  out.identity.lineage = kNet10Lineage;
  out.identity.axis = 0u;
  out.identity.ordinal = ordinal;
  out.chemotype = chemistry;
  out.reach = reach;
  out.begin_tick = 0u;
  return out;
}

}  // namespace

std::uint32_t seed_atlas_net10_microzone_count() { return kNet10MicrozoneCount; }

std::uint32_t seed_atlas_net10_partner_ordinal(std::uint32_t index) {
  return microzone_serves_domain_a(index) ? kDomainAOrdinal : kDomainBOrdinal;
}

DirectGenomeV1 seed_atlas_net10_replicated_microzones() {
  DirectGenomeV1 genome{};
  genome.header.life_function_version = 2u;
  genome.header.development_end_tick = kNet10DevelopmentEndTick;
  genome.header.matter_budget = 1u << 24;
  // A species constant, not an individual one: two individuals of this species
  // differ through Xi (#1342), never through this value.
  genome.header.development_seed = 0x4e455431u + 10u;  // "NET1" + 10

  // THE GENERATOR.  One loop emits every unit.  The only thing that differs
  // between units is which partner domain the corridor field names -- the
  // motif itself is written once, above, and reused verbatim.
  for (std::uint32_t index = 0u; index < kNet10MicrozoneCount; ++index) {
    genome.territories[genome.header.territory_count++] =
        net10_territory(index, kNet10MicrozoneChemistryBase + index, kMicrozoneReach);
  }
  genome.territories[genome.header.territory_count++] =
      net10_territory(kDomainAOrdinal, kDomainAChemistry, kDomainReach);
  genome.territories[genome.header.territory_count++] =
      net10_territory(kDomainBOrdinal, kDomainBChemistry, kDomainReach);

  for (std::uint32_t index = 0u; index < kNet10MicrozoneCount; ++index) {
    const std::uint32_t chemistry = kNet10MicrozoneChemistryBase + index;

    DirectFieldSpecV1 corridor{};
    corridor.territory.lineage = kNet10Lineage;
    corridor.territory.axis = 0u;
    corridor.territory.ordinal = index;
    corridor.radius = kNet10AffinityRadius;
    corridor.require_mask = kNet10Mask;
    // strong_partner_specificity: each unit names exactly one domain.
    corridor.require_value =
        microzone_serves_domain_a(index) ? kDomainAChemistry : kDomainBChemistry;
    corridor.write_mask = kNet10Mask;
    corridor.write_value = static_cast<std::uint32_t>(kNet10AffinityStrengthQ16);
    corridor.begin_tick = 0u;
    corridor.end_tick = kNet10DevelopmentEndTick;
    corridor.polarity = 0u;  // DevelopmentFieldKind::attract
    const std::uint32_t corridor_index = genome.header.field_count++;
    genome.fields[corridor_index] = corridor;

    DirectRuleSpecV1 grow = net10_rule(DirectRuleOpcodeV1::extend, chemistry);
    grow.extent = kMicrozoneReach;
    grow.child_slot = kMicrozoneDegree;
    grow.branch_count = kMicrozonePopulation;
    genome.rules[genome.header.rule_count++] = grow;

    DirectRuleSpecV1 tract = net10_rule(DirectRuleOpcodeV1::long_tract, chemistry);
    tract.branch_count = kMicrozoneLongTracts;
    tract.field_index = corridor_index;
    tract.extent = kMicrozoneDelayBase + index * kMicrozoneDelayStep;
    genome.rules[genome.header.rule_count++] = tract;

    DirectRuleSpecV1 ports = net10_rule(DirectRuleOpcodeV1::endogenous_source, chemistry);
    ports.extent = kMicrozonePortReserve;
    genome.rules[genome.header.rule_count++] = ports;
  }

  const std::uint32_t domains[] = {kDomainAChemistry, kDomainBChemistry};
  for (const std::uint32_t chemistry : domains) {
    DirectRuleSpecV1 grow = net10_rule(DirectRuleOpcodeV1::extend, chemistry);
    grow.extent = kDomainReach;
    grow.child_slot = kDomainDegree;
    grow.branch_count = kDomainPopulation;
    genome.rules[genome.header.rule_count++] = grow;

    DirectRuleSpecV1 ports = net10_rule(DirectRuleOpcodeV1::endogenous_source, chemistry);
    ports.extent = kMicrozonePortReserve;
    genome.rules[genome.header.rule_count++] = ports;
  }
  return genome;
}

}  // namespace substrate::direct_network
