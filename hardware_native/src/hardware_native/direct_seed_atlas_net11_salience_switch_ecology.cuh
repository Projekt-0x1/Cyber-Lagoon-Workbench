namespace substrate::direct_network {
namespace {

constexpr std::uint32_t kNet11Lineage = 11u;  // matches the atlas's own NET11 numbering.

// Two independently-lesionable corridor families, the atlas's own
// `distinct_from_world_truth` term (docs/architecture/human_connectome_to_
// gamma_seed_atlas_v0.md NET11 section). Ordinal 0/1 is the salience source/
// target pair, 2/3 the world-evidence source/target pair -- the same two-
// source/two-target shape NET17's fast/slow loops already proved, applied
// here to the atlas's salience-vs-world-truth split instead of a fast/slow
// timescale split.
constexpr std::uint32_t kSalienceSourceChemistry = 0xd1u;
constexpr std::uint32_t kSalienceTargetChemistry = 0xd2u;
constexpr std::uint32_t kWorldSourceChemistry = 0xd3u;
constexpr std::uint32_t kWorldTargetChemistry = 0xd4u;

constexpr std::uint32_t kNet11DevelopmentEndTick = 4096u;
// broad_but_bounded_cross_network_ports (the atlas's own NET11 term): the
// salience family's reach is wider than the ordinary world-evidence
// family's, but a finite territory reach -- not the near-unbounded affinity
// radius a `global_attention_scalar` would need.
constexpr std::uint32_t kSalienceReach = 24u;
constexpr std::uint32_t kWorldReach = 12u;
constexpr std::uint32_t kNet11Degree = 8u;
constexpr std::uint32_t kNet11Population = 96u;
constexpr std::uint32_t kNet11LongTracts = 8u;
constexpr std::uint32_t kNet11PortReserve = 16u;

// Same construction as NET01/NET02/NET12/NET17: generous radius, chemistry
// does the matching, no host-authored coordinate.
constexpr std::uint32_t kNet11AffinityRadius = 1u << 20;
constexpr std::int32_t kNet11AffinityStrengthQ16 = 1 << 16;

DirectRuleSpecV1 net11_rule(DirectRuleOpcodeV1 opcode, std::uint32_t chemistry) {
  DirectRuleSpecV1 out{};
  out.opcode = opcode;
  out.begin_tick = 0u;
  out.end_tick = kNet11DevelopmentEndTick;
  out.require_mask = 0xffffffffu;
  out.require_value = chemistry;
  out.write_mask = 0xffffffffu;
  out.write_value = chemistry;
  return out;
}

DirectTerritorySpecV1 net11_territory(std::uint32_t ordinal, std::uint32_t chemistry,
                                      std::uint32_t reach) {
  DirectTerritorySpecV1 out{};
  out.identity.lineage = kNet11Lineage;
  out.identity.axis = 0u;
  out.identity.ordinal = ordinal;
  out.chemotype = chemistry;
  out.reach = reach;
  out.begin_tick = 0u;
  return out;
}

}  // namespace

// NET11_SALIENCE_INTEROCEPTIVE_SWITCH_ECOLOGY (github #1276).
//
// The atlas's `distinct_from_world_truth` term, made concrete: TWO
// independently-lesionable corridor families -- `salience` (source ->
// target, chemistry-gated on the atlas's `high_access_to_body_consequence_
// channels` / `high_access_to_prediction_mismatch` terms, reaching the
// `recruit_reconfiguration` partner) and `world` (source -> target, ordinary
// perceptual evidence, on DISTINCT chemistry) -- the same DirectFieldSpecV1
// chemistry-matched corridor mechanism NET01/NET02/NET09/NET12/NET17
// already proved on a real device, applied here to the atlas's
// salience-vs-world-truth split rather than a fast/slow timescale split.
// Distinct chemistry throughout means neither family's corridor can land on
// the other's territory: salience cannot become truth merely by being
// adjacent to it, because there is no shared route for it to borrow.
//
// WHAT THIS DOES NOT AUTHOR. Every `delta_r` term
// (`context_specific_salience_relation`, `threat_or_novelty_
// overgeneralization_split`, `source_conditioned_relevance`) has no ABI
// carrier -- no relation-revision mechanism reachable from seed authoring
// exists yet, the same gap #1276's own NET12/NET17 rungs already named.
// `repeated_consequence_pattern -> compact_trigger_recipe` (n_plus_1) needs
// a condensation mechanism this seed layer cannot reach.
// `lesion_salience_ecology_vs_frontoparietal` needs the control family
// #1273 owns (cross-issue, out of scope); `spoofed_endogenous_surprise_
// cannot_gain_world_evidence` needs a learning/evidence-weighting mechanism
// that does not exist. This seed and its contract prove the anatomy only:
// two named corridor families, grown on a real device, each reaching only
// its own named partner, independently cuttable without touching the other.
DirectGenomeV1 seed_atlas_net11_salience_switch_ecology() {
  DirectGenomeV1 genome{};
  genome.header.life_function_version = 2u;
  genome.header.development_end_tick = kNet11DevelopmentEndTick;
  genome.header.matter_budget = 1u << 24;
  // A species constant, not an individual one: two individuals of this
  // species differ through Xi (#1342), never through this value.
  genome.header.development_seed = 0x4e455430u + 11u;  // "NET0" + 11

  constexpr std::uint32_t kSalienceSource = 0u, kSalienceTarget = 1u;
  constexpr std::uint32_t kWorldSource = 2u, kWorldTarget = 3u;
  genome.territories[genome.header.territory_count++] =
      net11_territory(kSalienceSource, kSalienceSourceChemistry, kSalienceReach);
  genome.territories[genome.header.territory_count++] =
      net11_territory(kSalienceTarget, kSalienceTargetChemistry, kSalienceReach);
  genome.territories[genome.header.territory_count++] =
      net11_territory(kWorldSource, kWorldSourceChemistry, kWorldReach);
  genome.territories[genome.header.territory_count++] =
      net11_territory(kWorldTarget, kWorldTargetChemistry, kWorldReach);

  // local_receptor_like_recipe_compatibility: the salience source's corridor
  // matches ONLY the salience target's chemistry, so it cannot land on the
  // world-evidence family's territories.
  DirectFieldSpecV1 salience_corridor{};
  salience_corridor.territory.lineage = kNet11Lineage;
  salience_corridor.territory.axis = 0u;
  salience_corridor.territory.ordinal = kSalienceSource;
  salience_corridor.radius = kNet11AffinityRadius;
  salience_corridor.require_mask = 0xffffffffu;
  salience_corridor.require_value = kSalienceTargetChemistry;
  salience_corridor.write_mask = 0xffffffffu;
  salience_corridor.write_value = static_cast<std::uint32_t>(kNet11AffinityStrengthQ16);
  salience_corridor.begin_tick = 0u;
  salience_corridor.end_tick = kNet11DevelopmentEndTick;
  salience_corridor.polarity = 0u;  // DevelopmentFieldKind::attract
  const std::uint32_t salience_corridor_index = genome.header.field_count++;
  genome.fields[salience_corridor_index] = salience_corridor;

  // the world-evidence source's corridor matches ONLY the world-evidence
  // target's chemistry -- the other receptor-compatible family, deliberately
  // incompatible with the salience family's territories.
  DirectFieldSpecV1 world_corridor = salience_corridor;
  world_corridor.territory.ordinal = kWorldSource;
  world_corridor.require_value = kWorldTargetChemistry;
  const std::uint32_t world_corridor_index = genome.header.field_count++;
  genome.fields[world_corridor_index] = world_corridor;

  const std::uint32_t chemistries[] = {kSalienceSourceChemistry, kSalienceTargetChemistry,
                                       kWorldSourceChemistry, kWorldTargetChemistry};
  for (const std::uint32_t chemistry : chemistries) {
    const bool is_salience = chemistry == kSalienceSourceChemistry ||
                             chemistry == kSalienceTargetChemistry;
    const std::uint32_t reach = is_salience ? kSalienceReach : kWorldReach;
    DirectRuleSpecV1 grow = net11_rule(DirectRuleOpcodeV1::extend, chemistry);
    grow.extent = reach;
    grow.child_slot = kNet11Degree;
    grow.branch_count = kNet11Population;
    genome.rules[genome.header.rule_count++] = grow;

    if (chemistry == kSalienceSourceChemistry || chemistry == kWorldSourceChemistry) {
      DirectRuleSpecV1 tract = net11_rule(DirectRuleOpcodeV1::long_tract, chemistry);
      tract.branch_count = kNet11LongTracts;
      tract.field_index =
          (chemistry == kSalienceSourceChemistry) ? salience_corridor_index : world_corridor_index;
      genome.rules[genome.header.rule_count++] = tract;
    }

    DirectRuleSpecV1 ports = net11_rule(DirectRuleOpcodeV1::endogenous_source, chemistry);
    ports.extent = kNet11PortReserve;
    genome.rules[genome.header.rule_count++] = ports;
  }
  return genome;
}

}  // namespace substrate::direct_network
