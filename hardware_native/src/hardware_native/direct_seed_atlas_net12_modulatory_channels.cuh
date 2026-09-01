namespace substrate::direct_network {
namespace {

constexpr std::uint32_t kNet12Lineage = 12u;  // matches the atlas's own NET12 numbering.
constexpr std::uint32_t kChannelAChemistry = 0x91u;
constexpr std::uint32_t kChannelBChemistry = 0x92u;
// independent_decay_timescale (#1276): channel_a is FAST-decaying (class 3,
// the atlas's own "short-latency" register), channel_b is PERSISTENT (class
// 0, no decay) -- the same narrow/broad reach split the two channels already
// carry, now also distinct on timescale. Packed above DevelopmentFieldKind
// (kDevelopmentFieldKindCount, from direct_network_brain.cuh) via
// direct_network_life_function.cu's field_decay_class
// (polarity / kDevelopmentFieldKindCount).
constexpr std::uint32_t kChannelAFastDecayClass = 3u;
constexpr std::uint32_t kChannelBNoDecayClass = 0u;
// distinct_effect_vector_over_gain (#1276 rung 3): the digit above decay
// class (direct_network_life_function.cu's field_gain_participates,
// `polarity / (kDevelopmentFieldKindCount * kFieldDecayClassCount)`). The `4u`
// here duplicates that file's local kFieldDecayClassCount -- it is not
// visible from this translation unit (declared inside an anonymous
// namespace), the same reason kChannelAFastDecayClass above is a literal
// rather than a named import.
constexpr std::uint32_t kLifeFunctionDecayClassCount = 4u;
constexpr std::uint32_t kGainParticipates = 1u;
constexpr std::uint32_t kTargetXChemistry = 0x93u;
constexpr std::uint32_t kTargetYChemistry = 0x94u;

constexpr std::uint32_t kNet12DevelopmentEndTick = 4096u;
// distinct_spatial_reach: channel_a is the NARROW, fast-onset family; channel_b
// is the BROAD, diffuse family -- the atlas's own "broad_projection" term.
constexpr std::uint32_t kChannelANarrowReach = 8u;
constexpr std::uint32_t kChannelBBroadReach = 40u;
constexpr std::uint32_t kTargetReach = 16u;
constexpr std::uint32_t kNet12Degree = 8u;
constexpr std::uint32_t kNet12Population = 96u;
constexpr std::uint32_t kNet12LongTracts = 8u;
constexpr std::uint32_t kNet12PortReserve = 16u;

// Same construction as NET01/NET02: generous radius, chemistry does the
// matching, no host-authored coordinate.
constexpr std::uint32_t kNet12AffinityRadius = 1u << 20;
constexpr std::int32_t kNet12AffinityStrengthQ16 = 1 << 16;

DirectRuleSpecV1 net12_rule(DirectRuleOpcodeV1 opcode, std::uint32_t chemistry) {
  DirectRuleSpecV1 out{};
  out.opcode = opcode;
  out.begin_tick = 0u;
  out.end_tick = kNet12DevelopmentEndTick;
  out.require_mask = 0xffffffffu;
  out.require_value = chemistry;
  out.write_mask = 0xffffffffu;
  out.write_value = chemistry;
  return out;
}

DirectTerritorySpecV1 net12_territory(std::uint32_t ordinal, std::uint32_t chemistry,
                                      std::uint32_t reach) {
  DirectTerritorySpecV1 out{};
  out.identity.lineage = kNet12Lineage;
  out.identity.axis = 0u;
  out.identity.ordinal = ordinal;
  out.chemotype = chemistry;
  out.reach = reach;
  out.begin_tick = 0u;
  return out;
}

}  // namespace

DirectGenomeV1 seed_atlas_net12_modulatory_channels(bool channel_a_decays) {
  DirectGenomeV1 genome{};
  genome.header.life_function_version = 2u;
  genome.header.development_end_tick = kNet12DevelopmentEndTick;
  genome.header.matter_budget = 1u << 24;
  // A species constant, not an individual one: two individuals of this species
  // differ through Xi (#1342), never through this value.
  genome.header.development_seed = 0x4e455431u + 11u;  // "NET1" + 11 == distinct from NET01/12 clash

  genome.territories[genome.header.territory_count++] =
      net12_territory(0u, kChannelAChemistry, kChannelANarrowReach);
  genome.territories[genome.header.territory_count++] =
      net12_territory(1u, kChannelBChemistry, kChannelBBroadReach);
  genome.territories[genome.header.territory_count++] =
      net12_territory(2u, kTargetXChemistry, kTargetReach);
  genome.territories[genome.header.territory_count++] =
      net12_territory(3u, kTargetYChemistry, kTargetReach);

  // local_receptor_like_recipe_compatibility: channel_a's corridor field
  // matches ONLY target_x's chemistry, so it cannot land on target_y.
  DirectFieldSpecV1 channel_a_corridor{};
  channel_a_corridor.territory.lineage = kNet12Lineage;
  channel_a_corridor.territory.axis = 0u;
  channel_a_corridor.territory.ordinal = 0u;  // channel_a
  channel_a_corridor.radius = kNet12AffinityRadius;
  channel_a_corridor.require_mask = 0xffffffffu;
  channel_a_corridor.require_value = kTargetXChemistry;
  channel_a_corridor.write_mask = 0xffffffffu;
  channel_a_corridor.write_value = static_cast<std::uint32_t>(kNet12AffinityStrengthQ16);
  channel_a_corridor.begin_tick = 0u;
  channel_a_corridor.end_tick = kNet12DevelopmentEndTick;
  // DevelopmentFieldKind::attract (0) + decay class + gain-participation,
  // packed. The reference configuration (channel_a_decays=false) uses class 0
  // -- identical to channel_b's decay class, distinguished only by the
  // gain-participation digit both channels now share.
  channel_a_corridor.polarity =
      0u +
      kDevelopmentFieldKindCount * (channel_a_decays ? kChannelAFastDecayClass : kChannelBNoDecayClass) +
      kDevelopmentFieldKindCount * kLifeFunctionDecayClassCount * kGainParticipates;
  const std::uint32_t channel_a_corridor_index = genome.header.field_count++;
  genome.fields[channel_a_corridor_index] = channel_a_corridor;

  // channel_b's corridor field matches ONLY target_y's chemistry -- the other
  // receptor-compatible family, deliberately incompatible with target_x.
  DirectFieldSpecV1 channel_b_corridor = channel_a_corridor;
  channel_b_corridor.territory.ordinal = 1u;  // channel_b
  channel_b_corridor.require_value = kTargetYChemistry;
  // DevelopmentFieldKind::attract (0) + no decay, persistent + gain-participation.
  channel_b_corridor.polarity = 0u + kDevelopmentFieldKindCount * kChannelBNoDecayClass +
                                 kDevelopmentFieldKindCount * kLifeFunctionDecayClassCount *
                                     kGainParticipates;
  const std::uint32_t channel_b_corridor_index = genome.header.field_count++;
  genome.fields[channel_b_corridor_index] = channel_b_corridor;

  const std::uint32_t chemistries[] = {kChannelAChemistry, kChannelBChemistry, kTargetXChemistry,
                                       kTargetYChemistry};
  for (const std::uint32_t chemistry : chemistries) {
    const std::uint32_t reach =
        (chemistry == kChannelAChemistry)
            ? kChannelANarrowReach
            : (chemistry == kChannelBChemistry) ? kChannelBBroadReach : kTargetReach;
    DirectRuleSpecV1 grow = net12_rule(DirectRuleOpcodeV1::extend, chemistry);
    grow.extent = reach;
    grow.child_slot = kNet12Degree;
    grow.branch_count = kNet12Population;
    genome.rules[genome.header.rule_count++] = grow;

    if (chemistry == kChannelAChemistry || chemistry == kChannelBChemistry) {
      DirectRuleSpecV1 tract = net12_rule(DirectRuleOpcodeV1::long_tract, chemistry);
      tract.branch_count = kNet12LongTracts;
      tract.field_index =
          (chemistry == kChannelAChemistry) ? channel_a_corridor_index : channel_b_corridor_index;
      genome.rules[genome.header.rule_count++] = tract;
    }

    DirectRuleSpecV1 ports = net12_rule(DirectRuleOpcodeV1::endogenous_source, chemistry);
    ports.extent = kNet12PortReserve;
    genome.rules[genome.header.rule_count++] = ports;
  }
  return genome;
}

}  // namespace substrate::direct_network
