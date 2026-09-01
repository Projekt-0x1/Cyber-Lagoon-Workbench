namespace substrate::direct_network {
namespace {

constexpr std::uint32_t kNet13Lineage = 13u;  // atlas NET13/NET14/NET18 backbone families.

// ONE TABLE.  `provenance` is observer metadata: no rule, field or kernel
// reads it, and deleting every string would change nothing the organism grows.
struct BackboneTract {
  const char* provenance;      // human tract name -- never a dispatch key
  std::uint32_t source_ordinal;
  std::uint32_t partner_ordinal;
  bool reciprocal;             // does the partner name the source back?
  std::uint32_t delay;         // construction-economy transport cost, this row's own
};

// Distinct per-row delays: the same `long_tract.extent` carrier NET03/NET05/NET10
// already prove, applied here to each backbone tract class' own construction
// economy rather than a two-way split or a per-unit sequence.
//
// The last two rows are NET18's contribution (github #1275's
// WHITE_MATTER_TRACT_GROWTH_PALETTE): its `seed_corridors` list names six
// classes; NET13's first four rows already demonstrate four of them
// (language, cortico_subcortical, cerebro_cerebellar, commissural) through
// this table's own generic machinery. `thalamocortical radiation` and a
// `memory_context` corridor (cingulum-like) are that machinery's fifth and
// sixth row -- no other edit anywhere, exactly the property NET18's
// `growth_rule` names.
constexpr BackboneTract kNet13Tracts[] = {
    {"language dorsal long tract", 0u, 1u, false, 10u},
    {"cortico-subcortical loop tract", 2u, 3u, false, 18u},
    {"cerebro-cerebellar return tract", 4u, 5u, false, 34u},
    {"commissural homologous connection", 6u, 7u, true, 14u},
    {"thalamocortical radiation", 8u, 9u, false, 22u},
    {"memory-context association corridor (cingulum-like)", 10u, 11u, false, 26u},
};
constexpr std::uint32_t kNet13TractCount =
    static_cast<std::uint32_t>(sizeof(kNet13Tracts) / sizeof(kNet13Tracts[0]));
constexpr std::uint32_t kNet13TerritoryCount = 2u * kNet13TractCount;

// Chemistry is derived from the ordinal, so adding a row to the table above
// needs no other edit anywhere -- the generic machinery the packet asks for.
constexpr std::uint32_t kNet13ChemistryBase = 0x50u;
constexpr std::uint32_t net13_chemistry(std::uint32_t ordinal) {
  return kNet13ChemistryBase + ordinal;
}

constexpr std::uint32_t kNet13DevelopmentEndTick = 4096u;
constexpr std::uint32_t kNet13Reach = 16u;
constexpr std::uint32_t kNet13Degree = 10u;
constexpr std::uint32_t kNet13Population = 64u;
constexpr std::uint32_t kNet13LongTracts = 9u;
constexpr std::uint32_t kNet13PortReserve = 16u;
constexpr std::uint32_t kNet13Mask = 0xffffffffu;
constexpr std::uint32_t kNet13AffinityRadius = 1u << 20;
constexpr std::int32_t kNet13AffinityStrengthQ16 = 1 << 16;

DirectRuleSpecV1 net13_rule(DirectRuleOpcodeV1 opcode, std::uint32_t chemistry) {
  DirectRuleSpecV1 out{};
  out.opcode = opcode;
  out.begin_tick = 0u;
  out.end_tick = kNet13DevelopmentEndTick;
  out.require_mask = kNet13Mask;
  out.require_value = chemistry;
  out.write_mask = kNet13Mask;
  out.write_value = chemistry;
  return out;
}

DirectFieldSpecV1 net13_corridor(std::uint32_t source_ordinal, std::uint32_t partner_chemistry) {
  DirectFieldSpecV1 out{};
  out.territory.lineage = kNet13Lineage;
  out.territory.axis = 0u;
  out.territory.ordinal = source_ordinal;
  out.radius = kNet13AffinityRadius;
  out.require_mask = kNet13Mask;
  out.require_value = partner_chemistry;
  out.write_mask = kNet13Mask;
  out.write_value = static_cast<std::uint32_t>(kNet13AffinityStrengthQ16);
  out.begin_tick = 0u;
  out.end_tick = kNet13DevelopmentEndTick;
  out.polarity = 0u;  // DevelopmentFieldKind::attract
  return out;
}

// Author one directed corridor plus the long_tract rule that grows it.  Both
// the one-way classes and the commissural pair go through THIS function; the
// only difference is that the commissural row calls it twice.
void net13_author_leg(DirectGenomeV1* genome, std::uint32_t source_ordinal,
                      std::uint32_t partner_ordinal, std::uint32_t delay) {
  const std::uint32_t field_index = genome->header.field_count++;
  genome->fields[field_index] = net13_corridor(source_ordinal, net13_chemistry(partner_ordinal));
  DirectRuleSpecV1 tract =
      net13_rule(DirectRuleOpcodeV1::long_tract, net13_chemistry(source_ordinal));
  tract.branch_count = kNet13LongTracts;
  tract.field_index = field_index;
  tract.extent = delay;
  genome->rules[genome->header.rule_count++] = tract;
}

}  // namespace

std::uint32_t seed_atlas_net13_tract_count() { return kNet13TractCount; }

std::uint32_t seed_atlas_net13_tract_source(std::uint32_t index) {
  return (index < kNet13TractCount) ? kNet13Tracts[index].source_ordinal : 0u;
}

std::uint32_t seed_atlas_net13_tract_partner(std::uint32_t index) {
  return (index < kNet13TractCount) ? kNet13Tracts[index].partner_ordinal : 0u;
}

bool seed_atlas_net13_tract_is_reciprocal(std::uint32_t index) {
  return (index < kNet13TractCount) && kNet13Tracts[index].reciprocal;
}

std::uint32_t seed_atlas_net13_tract_delay(std::uint32_t index) {
  return (index < kNet13TractCount) ? kNet13Tracts[index].delay : 0u;
}

DirectGenomeV1 seed_atlas_net13_backbone_tracts() {
  DirectGenomeV1 genome{};
  genome.header.life_function_version = 2u;
  genome.header.development_end_tick = kNet13DevelopmentEndTick;
  genome.header.matter_budget = 1u << 24;
  // A species constant, not an individual one.
  genome.header.development_seed = 0x4e455431u + 13u;  // "NET1" + 13

  for (std::uint32_t ordinal = 0u; ordinal < kNet13TerritoryCount; ++ordinal) {
    DirectTerritorySpecV1 territory{};
    territory.identity.lineage = kNet13Lineage;
    territory.identity.axis = 0u;
    territory.identity.ordinal = ordinal;
    territory.chemotype = net13_chemistry(ordinal);
    territory.reach = kNet13Reach;
    territory.begin_tick = 0u;
    genome.territories[genome.header.territory_count++] = territory;

    DirectRuleSpecV1 grow = net13_rule(DirectRuleOpcodeV1::extend, net13_chemistry(ordinal));
    grow.extent = kNet13Reach;
    grow.child_slot = kNet13Degree;
    grow.branch_count = kNet13Population;
    genome.rules[genome.header.rule_count++] = grow;

    DirectRuleSpecV1 ports =
        net13_rule(DirectRuleOpcodeV1::endogenous_source, net13_chemistry(ordinal));
    ports.extent = kNet13PortReserve;
    genome.rules[genome.header.rule_count++] = ports;
  }

  // The generic machinery: one loop, no branch on which tract class a row is.
  // A reciprocal row simply authors the return leg as well.
  for (std::uint32_t index = 0u; index < kNet13TractCount; ++index) {
    const BackboneTract& tract = kNet13Tracts[index];
    net13_author_leg(&genome, tract.source_ordinal, tract.partner_ordinal, tract.delay);
    if (tract.reciprocal) {
      net13_author_leg(&genome, tract.partner_ordinal, tract.source_ordinal, tract.delay);
    }
  }
  return genome;
}

}  // namespace substrate::direct_network
