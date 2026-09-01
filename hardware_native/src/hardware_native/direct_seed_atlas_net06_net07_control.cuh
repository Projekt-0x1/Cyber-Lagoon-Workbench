namespace substrate::direct_network {
namespace {

constexpr std::uint32_t kNet06Lineage = 6u;  // NET06 adaptive + NET07 stable-set control.

// Four network partners on ADJACENT chemistries, so relaxing two bits names
// the whole block and insisting on all of them names exactly one.  Breadth is
// therefore carried by the mask and never by a branch on controller identity.
constexpr std::uint32_t kNet06PartnerCount = 4u;
constexpr std::uint32_t kNet06PartnerChemistryBase = 0x60u;  // 0x60..0x63
constexpr std::uint32_t kNet06AdaptiveChemistry = 0x68u;
constexpr std::uint32_t kNet06StableChemistry = 0x69u;

constexpr std::uint32_t kNet06AdaptiveOrdinal = 0u;
constexpr std::uint32_t kNet06StableOrdinal = 1u;
constexpr std::uint32_t kNet06PartnerOrdinalBase = 2u;  // partners occupy 2..5

// high cross-network port diversity: (chem & ~3) == 0x60 matches all four
// partners, and neither controller (0x68 & ~3 == 0x68, 0x69 & ~3 == 0x68).
constexpr std::uint32_t kNet06BroadMask = 0xfffffffcu;
// stable context/set maintenance: the full chemistry names exactly one.
constexpr std::uint32_t kNet06NarrowMask = 0xffffffffu;

constexpr std::uint32_t kNet06DevelopmentEndTick = 4096u;
constexpr std::uint32_t kNet06ControlReach = 18u;
constexpr std::uint32_t kNet06PartnerReach = 14u;
constexpr std::uint32_t kNet06Degree = 12u;
constexpr std::uint32_t kNet06Population = 64u;
constexpr std::uint32_t kNet06LongTracts = 12u;
constexpr std::uint32_t kNet06PortReserve = 20u;
constexpr std::uint32_t kNet06Mask = 0xffffffffu;
constexpr std::uint32_t kNet06AffinityRadius = 1u << 20;
constexpr std::int32_t kNet06AffinityStrengthQ16 = 1 << 16;

// independent_decay_timescale (#1276/NET12's mechanism, `08c38a9803`), taken
// for this seed's own named gap: the packet's SPEED/PERSISTENCE axis --
// "faster reconfiguration" (NET06 adaptive) against "slower recurrent
// persistence... resistance to incidental transient disturbance" (NET07
// stable) -- has no timescale carrier when this seed was first authored
// (`5278aea4ed`). `field_decay_class`, packed into `DirectFieldSpecV1.polarity`
// above `DevelopmentFieldKind` (`kDevelopmentFieldKindCount`, from
// direct_network_brain.cuh), closes it: the adaptive controller's field decays
// (class 2), the stable controller's stays PERMANENT (class 0, no decay) --
// the field itself now outlives or fades exactly as the packet's language
// describes, not just its reach.
//
// Class 2, not the faster class 3 or the slower class 1: measured, not
// guessed. `field_decay_magnitude_q16_per_tick` zeroes a
// `kNet06AffinityStrengthQ16` (1<<16) field at elapsed == strength/magnitude
// ticks -- class 3 (1<<10/tick) hits zero at tick 64 of this seed's
// kNet06DevelopmentEndTick=4096 window, so nearly the ENTIRE window sees a
// fully-decayed field and arm 1's own breadth measurement (adaptive_reach >
// stable_reach) collapsed to a tie (measured: adaptive dropped from reaching
// 3/4 partners to 1/4). Class 1 (1<<4/tick) reaches zero only at tick 4096 --
// the window's own end -- decaying so gradually that arm 7's own route-count
// dissociation vanished (measured: both controllers formed the identical 12
// routes to their shared partner on the flattened sibling). Class 2
// (1<<7/tick) reaches zero at tick 512 (1/8 of the window): breadth (arm 1)
// stays exactly 3/4 vs 1/4, unchanged from the pre-decay seed, while the
// decay dissociation (arm 7) shows a real, nonzero, non-collapsed effect (8
// routes vs. the reference's 12) -- the one class of the three that measurably
// moves arm 7 without breaking arm 1.
constexpr std::uint32_t kNet06AdaptiveFastDecayClass = 2u;
constexpr std::uint32_t kNet06StablePersistentDecayClass = 0u;

constexpr std::uint32_t net06_partner_chemistry(std::uint32_t index) {
  return kNet06PartnerChemistryBase + index;
}

DirectRuleSpecV1 net06_rule(DirectRuleOpcodeV1 opcode, std::uint32_t chemistry) {
  DirectRuleSpecV1 out{};
  out.opcode = opcode;
  out.begin_tick = 0u;
  out.end_tick = kNet06DevelopmentEndTick;
  out.require_mask = kNet06Mask;
  out.require_value = chemistry;
  out.write_mask = kNet06Mask;
  out.write_value = chemistry;
  return out;
}

DirectTerritorySpecV1 net06_territory(std::uint32_t ordinal, std::uint32_t chemistry,
                                      std::uint32_t reach) {
  DirectTerritorySpecV1 out{};
  out.identity.lineage = kNet06Lineage;
  out.identity.axis = 0u;
  out.identity.ordinal = ordinal;
  out.chemotype = chemistry;
  out.reach = reach;
  out.begin_tick = 0u;
  return out;
}

// Both siblings come from HERE.  The only argument that ever differs is the
// adaptive controller's mask, so the two genomes cannot drift apart anywhere
// else -- resource-matching by construction, not by assertion.
DirectGenomeV1 net06_seed(std::uint32_t adaptive_mask) {
  DirectGenomeV1 genome{};
  genome.header.life_function_version = 2u;
  genome.header.development_end_tick = kNet06DevelopmentEndTick;
  genome.header.matter_budget = 1u << 24;
  genome.header.development_seed = 0x4e455430u + 6u;  // "NET0" + 6

  genome.territories[genome.header.territory_count++] =
      net06_territory(kNet06AdaptiveOrdinal, kNet06AdaptiveChemistry, kNet06ControlReach);
  genome.territories[genome.header.territory_count++] =
      net06_territory(kNet06StableOrdinal, kNet06StableChemistry, kNet06ControlReach);
  for (std::uint32_t index = 0u; index < kNet06PartnerCount; ++index) {
    genome.territories[genome.header.territory_count++] = net06_territory(
        kNet06PartnerOrdinalBase + index, net06_partner_chemistry(index), kNet06PartnerReach);
  }

  // Both controllers name the SAME first partner chemistry as their require
  // value; only how much of it they insist on differs.
  const std::uint32_t adaptive_field = genome.header.field_count++;
  {
    DirectFieldSpecV1 corridor{};
    corridor.territory.lineage = kNet06Lineage;
    corridor.territory.axis = 0u;
    corridor.territory.ordinal = kNet06AdaptiveOrdinal;
    corridor.radius = kNet06AffinityRadius;
    corridor.require_mask = adaptive_mask;
    corridor.require_value = net06_partner_chemistry(0u);
    corridor.write_mask = kNet06Mask;
    corridor.write_value = static_cast<std::uint32_t>(kNet06AffinityStrengthQ16);
    corridor.begin_tick = 0u;
    corridor.end_tick = kNet06DevelopmentEndTick;
    // DevelopmentFieldKind::attract (0) + decay -- the adaptive
    // controller's own field fades toward zero as development proceeds,
    // authoring "faster reconfiguration" as a real timescale rather than
    // only a breadth mask.
    corridor.polarity = 0u + kDevelopmentFieldKindCount * kNet06AdaptiveFastDecayClass;
    genome.fields[adaptive_field] = corridor;
  }
  const std::uint32_t stable_field = genome.header.field_count++;
  {
    DirectFieldSpecV1 corridor = genome.fields[adaptive_field];
    corridor.territory.ordinal = kNet06StableOrdinal;
    corridor.require_mask = kNet06NarrowMask;
    // DevelopmentFieldKind::attract (0) + PERMANENT (no decay) -- the stable
    // controller's field never fades, authoring "resistance to incidental
    // transient disturbance" as a real timescale.
    corridor.polarity = 0u + kDevelopmentFieldKindCount * kNet06StablePersistentDecayClass;
    genome.fields[stable_field] = corridor;
  }

  const std::uint32_t control_chemistries[] = {kNet06AdaptiveChemistry, kNet06StableChemistry};
  for (std::uint32_t which = 0u; which < 2u; ++which) {
    const std::uint32_t chemistry = control_chemistries[which];
    DirectRuleSpecV1 grow = net06_rule(DirectRuleOpcodeV1::extend, chemistry);
    grow.extent = kNet06ControlReach;
    grow.child_slot = kNet06Degree;
    grow.branch_count = kNet06Population;
    genome.rules[genome.header.rule_count++] = grow;

    DirectRuleSpecV1 tract = net06_rule(DirectRuleOpcodeV1::long_tract, chemistry);
    tract.branch_count = kNet06LongTracts;
    tract.field_index = (which == 0u) ? adaptive_field : stable_field;
    genome.rules[genome.header.rule_count++] = tract;

    DirectRuleSpecV1 ports = net06_rule(DirectRuleOpcodeV1::endogenous_source, chemistry);
    ports.extent = kNet06PortReserve;
    genome.rules[genome.header.rule_count++] = ports;
  }

  for (std::uint32_t index = 0u; index < kNet06PartnerCount; ++index) {
    const std::uint32_t chemistry = net06_partner_chemistry(index);
    DirectRuleSpecV1 grow = net06_rule(DirectRuleOpcodeV1::extend, chemistry);
    grow.extent = kNet06PartnerReach;
    grow.child_slot = kNet06Degree;
    grow.branch_count = kNet06Population;
    genome.rules[genome.header.rule_count++] = grow;

    DirectRuleSpecV1 ports = net06_rule(DirectRuleOpcodeV1::endogenous_source, chemistry);
    ports.extent = kNet06PortReserve;
    genome.rules[genome.header.rule_count++] = ports;
  }
  return genome;
}

}  // namespace

std::uint32_t seed_atlas_net06_partner_count() { return kNet06PartnerCount; }

std::uint32_t seed_atlas_net06_partner_ordinal(std::uint32_t index) {
  return kNet06PartnerOrdinalBase + ((index < kNet06PartnerCount) ? index : 0u);
}

std::uint32_t seed_atlas_net06_adaptive_ordinal() { return kNet06AdaptiveOrdinal; }

std::uint32_t seed_atlas_net06_stable_ordinal() { return kNet06StableOrdinal; }

DirectGenomeV1 seed_atlas_net06_net07_control() { return net06_seed(kNet06BroadMask); }

DirectGenomeV1 seed_atlas_net06_net07_flattened_sibling() { return net06_seed(kNet06NarrowMask); }

}  // namespace substrate::direct_network
