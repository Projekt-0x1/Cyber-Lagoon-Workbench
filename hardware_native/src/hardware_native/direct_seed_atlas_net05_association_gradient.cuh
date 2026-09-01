namespace substrate::direct_network {
namespace {

constexpr std::uint32_t kNet05Lineage = 5u;  // matches the atlas's own NET05 numbering.

// heteromodal and transmodal are ADJACENT (differ only in bit 0) so that one
// bit of one mask is the entire difference between the two sibling genomes.
// sensory is deliberately far from both, so a relaxed sensory corridor widens
// onto the association ranks without ever matching the body edge itself.
constexpr std::uint32_t kNet05SensoryChemistry = 0xe8u;
constexpr std::uint32_t kNet05HeteromodalChemistry = 0xe4u;
constexpr std::uint32_t kNet05TransmodalChemistry = 0xe5u;

constexpr std::uint32_t kNet05SensoryOrdinal = 0u;
constexpr std::uint32_t kNet05HeteromodalOrdinal = 1u;
constexpr std::uint32_t kNet05TransmodalOrdinal = 2u;

// THE ONE BIT.  A full mask makes the sensory corridor name heteromodal only;
// clearing bit 0 makes the same corridor name heteromodal AND transmodal,
// because (0xe4 & ~1) == (0xe5 & ~1) == 0xe4 while (0xe8 & ~1) == 0xe8.
constexpr std::uint32_t kGradedSensoryMask = 0xffffffffu;
constexpr std::uint32_t kFlatSensoryMask = 0xfffffffeu;

constexpr std::uint32_t kNet05DevelopmentEndTick = 4096u;
// Shared by BOTH siblings -- this is what "resource-matched" means here, and
// it is matched by construction rather than by assertion.
constexpr std::uint32_t kNet05Reach = 20u;
constexpr std::uint32_t kNet05Degree = 10u;
constexpr std::uint32_t kNet05Population = 80u;
constexpr std::uint32_t kNet05LongTracts = 10u;
constexpr std::uint32_t kNet05PortReserve = 16u;

constexpr std::uint32_t kNet05Mask = 0xffffffffu;
constexpr std::uint32_t kNet05AffinityRadius = 1u << 20;
constexpr std::int32_t kNet05AffinityStrengthQ16 = 1 << 16;

// The atlas's own timescale gradient: "delays: short" near the body edge,
// widening toward the association/transmodal end. `long_tract`'s `extent`
// is the transport-cost carrier this ABI already has (b0a747d01a, proven by
// NET03's chain), so the same two authored hops that carry the connectivity
// gradient also carry its delay -- no new machinery, and identical on both
// siblings since both are emitted by net05_seed(). It is a magnitude carried
// once per corridor family, not a distribution across the three ranks: a
// per-rank recurrence/persistence timescale still has no carrier here.
constexpr std::uint32_t kNet05SensoryToHeteromodalDelay = 8u;
constexpr std::uint32_t kNet05HeteromodalToTransmodalDelay = 24u;

DirectRuleSpecV1 net05_rule(DirectRuleOpcodeV1 opcode, std::uint32_t chemistry) {
  DirectRuleSpecV1 out{};
  out.opcode = opcode;
  out.begin_tick = 0u;
  out.end_tick = kNet05DevelopmentEndTick;
  out.require_mask = kNet05Mask;
  out.require_value = chemistry;
  out.write_mask = kNet05Mask;
  out.write_value = chemistry;
  return out;
}

DirectTerritorySpecV1 net05_territory(std::uint32_t ordinal, std::uint32_t chemistry) {
  DirectTerritorySpecV1 out{};
  out.identity.lineage = kNet05Lineage;
  out.identity.axis = 0u;
  out.identity.ordinal = ordinal;
  out.chemotype = chemistry;
  out.reach = kNet05Reach;
  out.begin_tick = 0u;
  return out;
}

DirectFieldSpecV1 net05_corridor(std::uint32_t source_ordinal, std::uint32_t require_mask,
                                 std::uint32_t require_value) {
  DirectFieldSpecV1 out{};
  out.territory.lineage = kNet05Lineage;
  out.territory.axis = 0u;
  out.territory.ordinal = source_ordinal;
  out.radius = kNet05AffinityRadius;
  out.require_mask = require_mask;
  out.require_value = require_value;
  out.write_mask = kNet05Mask;
  out.write_value = static_cast<std::uint32_t>(kNet05AffinityStrengthQ16);
  out.begin_tick = 0u;
  out.end_tick = kNet05DevelopmentEndTick;
  out.polarity = 0u;  // DevelopmentFieldKind::attract
  return out;
}

// Both siblings are emitted by THIS function.  The only argument that ever
// differs is the sensory corridor's mask, which is the whole point: the two
// genomes cannot drift apart in any other respect, because there is no other
// place for them to differ.
DirectGenomeV1 net05_seed(std::uint32_t sensory_mask) {
  DirectGenomeV1 genome{};
  genome.header.life_function_version = 2u;
  genome.header.development_end_tick = kNet05DevelopmentEndTick;
  genome.header.matter_budget = 1u << 24;
  // Identical across siblings: a different seed would make the comparison a
  // comparison of individuals rather than of genomes.
  genome.header.development_seed = 0x4e455430u + 5u;  // "NET0" + 5

  genome.territories[genome.header.territory_count++] =
      net05_territory(kNet05SensoryOrdinal, kNet05SensoryChemistry);
  genome.territories[genome.header.territory_count++] =
      net05_territory(kNet05HeteromodalOrdinal, kNet05HeteromodalChemistry);
  genome.territories[genome.header.territory_count++] =
      net05_territory(kNet05TransmodalOrdinal, kNet05TransmodalChemistry);

  // The body edge's corridor: graded names the middle only, flat names the
  // middle and the far edge.
  const std::uint32_t sensory_field = genome.header.field_count++;
  genome.fields[sensory_field] =
      net05_corridor(kNet05SensoryOrdinal, sensory_mask, kNet05HeteromodalChemistry);

  // The middle's corridor to the far edge -- IDENTICAL in both siblings, so
  // any difference the contract measures is attributable to the sensory mask.
  const std::uint32_t heteromodal_field = genome.header.field_count++;
  genome.fields[heteromodal_field] =
      net05_corridor(kNet05HeteromodalOrdinal, kNet05Mask, kNet05TransmodalChemistry);

  const std::uint32_t chemistries[] = {kNet05SensoryChemistry, kNet05HeteromodalChemistry,
                                       kNet05TransmodalChemistry};
  for (const std::uint32_t chemistry : chemistries) {
    DirectRuleSpecV1 grow = net05_rule(DirectRuleOpcodeV1::extend, chemistry);
    grow.extent = kNet05Reach;
    grow.child_slot = kNet05Degree;
    grow.branch_count = kNet05Population;
    genome.rules[genome.header.rule_count++] = grow;

    // The transmodal edge is the end of the chain and grows no corridor of its
    // own in either sibling, so the long-tract budget is matched too.
    if (chemistry == kNet05SensoryChemistry || chemistry == kNet05HeteromodalChemistry) {
      DirectRuleSpecV1 tract = net05_rule(DirectRuleOpcodeV1::long_tract, chemistry);
      tract.branch_count = kNet05LongTracts;
      tract.field_index = (chemistry == kNet05SensoryChemistry) ? sensory_field : heteromodal_field;
      tract.extent = (chemistry == kNet05SensoryChemistry) ? kNet05SensoryToHeteromodalDelay
                                                            : kNet05HeteromodalToTransmodalDelay;
      genome.rules[genome.header.rule_count++] = tract;
    }

    DirectRuleSpecV1 ports = net05_rule(DirectRuleOpcodeV1::endogenous_source, chemistry);
    ports.extent = kNet05PortReserve;
    genome.rules[genome.header.rule_count++] = ports;
  }
  return genome;
}

}  // namespace

DirectGenomeV1 seed_atlas_net05_association_gradient() { return net05_seed(kGradedSensoryMask); }

DirectGenomeV1 seed_atlas_net05_flat_sibling() { return net05_seed(kFlatSensoryMask); }

}  // namespace substrate::direct_network
