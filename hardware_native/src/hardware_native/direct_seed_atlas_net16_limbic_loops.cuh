namespace substrate::direct_network {
namespace {

constexpr std::uint32_t kNet16Lineage = 16u;  // matches the atlas's own NET16 numbering.
constexpr std::uint32_t kNet16SourceChemistry = 0xf3u;
constexpr std::uint32_t kNet16FastTargetChemistry = 0xf4u;
constexpr std::uint32_t kNet16SlowTargetChemistry = 0xf5u;

constexpr std::uint32_t kNet16DevelopmentEndTick = 4096u;

// fast_then_slow_parallel_paths: ONE source forks two independently-
// lesionable corridors, rather than NET17's two wholly separate loops --
// the atlas frames NET16 around a single body-consequence hub
// (high_source_context_binding), not two disjoint sources. fast_target is
// NARROW/immediate; slow_target is BROAD/integrative -- the same narrow/broad
// split NET12/NET17 already carry, retargeted to this family.
constexpr std::uint32_t kNet16FastReach = 6u;
constexpr std::uint32_t kNet16SlowReach = 36u;
// high_source_context_binding: the hub's own degree is higher than either
// target's -- many simultaneous partner bindings from one source, the
// literal reading of "context binding".
constexpr std::uint32_t kNet16SourceDegree = 16u;
constexpr std::uint32_t kNet16TargetDegree = 8u;
constexpr std::uint32_t kNet16Population = 64u;
constexpr std::uint32_t kNet16LongTracts = 8u;
constexpr std::uint32_t kNet16PortReserve = 16u;
// short_latency vs the atlas's own "then slow" timing term: the fast leg
// stays at the pure geometric delay; the slow leg's long_tract extent adds
// an authored transport cost on top -- the same b0a747d01a carrier NET01/
// NET03/NET05/NET09/NET17 already use to separate a fast leg from a slow one.
constexpr std::uint32_t kNet16SlowTransportCost = 24u;

constexpr std::uint32_t kNet16AffinityRadius = 1u << 20;
constexpr std::int32_t kNet16AffinityStrengthQ16 = 1 << 16;

DirectRuleSpecV1 net16_rule(DirectRuleOpcodeV1 opcode, std::uint32_t chemistry) {
  DirectRuleSpecV1 out{};
  out.opcode = opcode;
  out.begin_tick = 0u;
  out.end_tick = kNet16DevelopmentEndTick;
  out.require_mask = 0xffffffffu;
  out.require_value = chemistry;
  out.write_mask = 0xffffffffu;
  out.write_value = chemistry;
  return out;
}

DirectTerritorySpecV1 net16_territory(std::uint32_t ordinal, std::uint32_t chemistry,
                                      std::uint32_t reach) {
  DirectTerritorySpecV1 out{};
  out.identity.lineage = kNet16Lineage;
  out.identity.axis = 0u;
  out.identity.ordinal = ordinal;
  out.chemotype = chemistry;
  out.reach = reach;
  out.begin_tick = 0u;
  return out;
}

// One directed corridor field, owned by the source hub (ordinal 0), seeking
// `partner_chemistry`. Same construction NET12/NET15/NET17 use: a generous
// radius so affinity is expressed by chemistry rather than a guessed
// coordinate, inert for the owner's own placement.
DirectFieldSpecV1 net16_corridor(std::uint32_t partner_chemistry) {
  DirectFieldSpecV1 out{};
  out.territory.lineage = kNet16Lineage;
  out.territory.axis = 0u;
  out.territory.ordinal = 0u;  // the source hub owns both corridor families.
  out.radius = kNet16AffinityRadius;
  out.require_mask = 0xffffffffu;
  out.require_value = partner_chemistry;
  out.write_mask = 0xffffffffu;
  out.write_value = static_cast<std::uint32_t>(kNet16AffinityStrengthQ16);
  out.begin_tick = 0u;
  out.end_tick = kNet16DevelopmentEndTick;
  out.polarity = 0u;  // DevelopmentFieldKind::attract
  return out;
}

}  // namespace

// NET16_AMYGDALA_LIMBIC_SOCIAL_AFFECTIVE_LOOPS (github #1267).
//
// Three territories: source (the body-consequence hub, ordinal 0),
// fast_target (ordinal 1, NARROW/immediate) and slow_target (ordinal 2,
// BROAD/integrative -- the atlas's own transmodal_slow_path). The atlas
// names `fast_then_slow_parallel_paths` around a SINGLE hub rather than two
// disjoint sources (contrast NET17's two wholly separate loops), so source
// forks two independently-authored, independently-lesionable corridor
// fields -- the same fork-from-one-territory construction NET15's
// crossmodal binding uses, one hop shorter and with a delay asymmetry
// instead of a growth-bias split.
//
// high_source_context_binding is authored as the source's own degree, higher
// than either target's -- many simultaneous partner bindings from one hub.
//
// The atlas's own probe amygdala_like_fast_path_lesion_vs_transmodal_slow_
// path is realized directly: cutting the fast corridor removes fast_target's
// incoming connectivity while slow_target's is untouched, and the fast leg's
// mean route delay is measurably lower than the slow leg's.
//
// WHAT THIS DOES NOT AUTHOR. source_partner_conditioning, interaction_with_
// hippocampal_context, interaction_with_prefrontal_control, extinction_and_
// reversal and self_vs_other_agency are dynamics and cross-network relations,
// not developmental anatomy -- no gain/plasticity/context-conditioning path
// exists anywhere in the executor yet (same absence NET12/NET17 already
// documented). This seed and its contract prove the anatomy only: one hub,
// two independently-lesionable, differently-timed corridors.
DirectGenomeV1 seed_atlas_net16_limbic_loops() {
  DirectGenomeV1 genome{};
  genome.header.life_function_version = 2u;
  genome.header.development_end_tick = kNet16DevelopmentEndTick;
  genome.header.matter_budget = 1u << 24;
  // A species constant, not an individual one: two individuals of this
  // species differ through Xi (#1342), never through this value.
  genome.header.development_seed = 0x4e455430u + 16u;  // "NET0" + 16

  constexpr std::uint32_t kSource = 0u, kFastTarget = 1u, kSlowTarget = 2u;
  genome.territories[genome.header.territory_count++] =
      net16_territory(kSource, kNet16SourceChemistry, kNet16FastReach);
  genome.territories[genome.header.territory_count++] =
      net16_territory(kFastTarget, kNet16FastTargetChemistry, kNet16FastReach);
  genome.territories[genome.header.territory_count++] =
      net16_territory(kSlowTarget, kNet16SlowTargetChemistry, kNet16SlowReach);

  // local_receptor_like_recipe_compatibility: each corridor field matches
  // ONLY its own target's chemistry, so neither can land on the other's
  // territory.
  const std::uint32_t fast_corridor_index = genome.header.field_count++;
  genome.fields[fast_corridor_index] = net16_corridor(kNet16FastTargetChemistry);
  const std::uint32_t slow_corridor_index = genome.header.field_count++;
  genome.fields[slow_corridor_index] = net16_corridor(kNet16SlowTargetChemistry);

  const std::uint32_t chemistries[] = {kNet16SourceChemistry, kNet16FastTargetChemistry,
                                       kNet16SlowTargetChemistry};
  for (const std::uint32_t chemistry : chemistries) {
    const std::uint32_t reach =
        chemistry == kNet16SlowTargetChemistry ? kNet16SlowReach : kNet16FastReach;
    DirectRuleSpecV1 grow = net16_rule(DirectRuleOpcodeV1::extend, chemistry);
    grow.extent = reach;
    grow.child_slot = chemistry == kNet16SourceChemistry ? kNet16SourceDegree : kNet16TargetDegree;
    grow.branch_count = kNet16Population;
    genome.rules[genome.header.rule_count++] = grow;

    if (chemistry == kNet16SourceChemistry) {
      DirectRuleSpecV1 fast_tract = net16_rule(DirectRuleOpcodeV1::long_tract, chemistry);
      fast_tract.branch_count = kNet16LongTracts;
      fast_tract.field_index = fast_corridor_index;
      genome.rules[genome.header.rule_count++] = fast_tract;

      // short_to_medium_delays: the fast leg stays at the pure geometric
      // delay; the slow leg carries an authored transport cost on top.
      DirectRuleSpecV1 slow_tract = net16_rule(DirectRuleOpcodeV1::long_tract, chemistry);
      slow_tract.branch_count = kNet16LongTracts;
      slow_tract.field_index = slow_corridor_index;
      slow_tract.extent = kNet16SlowTransportCost;
      genome.rules[genome.header.rule_count++] = slow_tract;
    }

    DirectRuleSpecV1 ports = net16_rule(DirectRuleOpcodeV1::endogenous_source, chemistry);
    ports.extent = kNet16PortReserve;
    genome.rules[genome.header.rule_count++] = ports;
  }
  return genome;
}

}  // namespace substrate::direct_network
