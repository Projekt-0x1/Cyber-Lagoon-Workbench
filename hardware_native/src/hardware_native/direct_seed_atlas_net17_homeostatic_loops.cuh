namespace substrate::direct_network {
namespace {

constexpr std::uint32_t kNet17Lineage = 17u;  // matches the atlas's own NET17 numbering.

// Two independently-lesionable loops, the atlas's own
// `short_latency_protective_loops` / `slow_allostatic_state` split (docs/
// architecture/human_connectome_to_gamma_seed_atlas_v0.md NET17 section).
// Ordinal 0/1 is the fast protective source/target pair, 2/3 the slow
// allostatic pair -- same shape as NET12's two-channel construction, applied
// to a different atlas family and a different territory pair per channel
// (rather than NET12's shared pair of targets), since the atlas names these
// as two SEPARATE loops rather than two sources converging on shared
// territories.
constexpr std::uint32_t kFastProtectiveSourceChemistry = 0xb1u;
constexpr std::uint32_t kFastProtectiveTargetChemistry = 0xb2u;
constexpr std::uint32_t kSlowAllostaticSourceChemistry = 0xb3u;
constexpr std::uint32_t kSlowAllostaticTargetChemistry = 0xb4u;

constexpr std::uint32_t kNet17DevelopmentEndTick = 4096u;
// short_latency_protective_loops / slow_allostatic_state: the fast loop is
// NARROW and immediate, the slow loop is BROAD and integrative -- the same
// narrow/broad reach split NET12's two channels already carry, applied here
// to the atlas's fast/slow timescale term instead of NET12's spatial-reach
// term (this family has no decay-class ABI to author yet, see the seed
// function's own comment below).
constexpr std::uint32_t kFastReach = 6u;
constexpr std::uint32_t kSlowReach = 36u;
constexpr std::uint32_t kNet17Degree = 8u;
constexpr std::uint32_t kNet17Population = 96u;
constexpr std::uint32_t kNet17LongTracts = 8u;
constexpr std::uint32_t kNet17PortReserve = 16u;
// short_to_medium_delays (the atlas's own NET17 term, shared with NET09):
// the fast protective corridor carries the pure geometric delay; the slow
// allostatic corridor's `long_tract` extent adds an authored transport cost
// on top, the same `b0a747d01a` carrier NET01/NET03/NET09 already use to
// separate a fast leg from a slow one.
constexpr std::uint32_t kSlowAllostaticTransportCost = 24u;

// Same construction as NET01/NET02/NET12: generous radius, chemistry does the
// matching, no host-authored coordinate.
constexpr std::uint32_t kNet17AffinityRadius = 1u << 20;
constexpr std::int32_t kNet17AffinityStrengthQ16 = 1 << 16;

DirectRuleSpecV1 net17_rule(DirectRuleOpcodeV1 opcode, std::uint32_t chemistry) {
  DirectRuleSpecV1 out{};
  out.opcode = opcode;
  out.begin_tick = 0u;
  out.end_tick = kNet17DevelopmentEndTick;
  out.require_mask = 0xffffffffu;
  out.require_value = chemistry;
  out.write_mask = 0xffffffffu;
  out.write_value = chemistry;
  return out;
}

DirectTerritorySpecV1 net17_territory(std::uint32_t ordinal, std::uint32_t chemistry,
                                      std::uint32_t reach) {
  DirectTerritorySpecV1 out{};
  out.identity.lineage = kNet17Lineage;
  out.identity.axis = 0u;
  out.identity.ordinal = ordinal;
  out.chemotype = chemistry;
  out.reach = reach;
  out.begin_tick = 0u;
  return out;
}

}  // namespace

// NET17_HOMEOSTATIC_BRAINSTEM_HYPOTHALAMIC_BODY_CONTROL (github #1276).
//
// Two independently-lesionable body-tethered loops -- `short_latency_
// protective_loops` (ordinals 0-1: source -> target, narrow reach, pure
// geometric delay) and `slow_allostatic_state` (ordinals 2-3: source ->
// target, broad reach, authored transport cost on top of geometric delay) --
// the same `DirectFieldSpecV1` chemistry-matched corridor mechanism NET01/
// NET02/NET09/NET12 already proved on a real device, applied here to the
// atlas's fast-protective/slow-allostatic split rather than NET12's spatial-
// reach split. Distinct chemistry throughout means neither loop's corridor
// can land on the other's territory -- the same receptor-incompatibility
// construction NET12's two channels already proved.
//
// WHAT THIS DOES NOT AUTHOR. `dedicated_body_vitality_damage_resource_
// channels`, `strong_insula_cingulate_modulatory_links`,
// `chronic_load_metaplasticity`, and `context_specific_protective_reflex`
// are NOT authored -- these are resource-ledger and learning-dynamics claims,
// and (per #1276's NET12 rungs) no gain/plasticity/resource-channel
// consumption path exists anywhere in the executor yet (`grep -rn
// "gain_bias\|plasticity_threshold\|search_pressure\|retention_bias\|
// effect_vector"` across `DirectBrain`/`direct_adult_core.cu` is zero).
// This seed and its contract prove the anatomy only: two named body-adjacent
// loops, grown on a real device, each independently cuttable without
// touching the other, with the atlas's own fast/slow delay asymmetry
// measured rather than assumed.
DirectGenomeV1 seed_atlas_net17_homeostatic_loops() {
  DirectGenomeV1 genome{};
  genome.header.life_function_version = 2u;
  genome.header.development_end_tick = kNet17DevelopmentEndTick;
  genome.header.matter_budget = 1u << 24;
  // A species constant, not an individual one: two individuals of this
  // species differ through Xi (#1342), never through this value.
  genome.header.development_seed = 0x4e455430u + 17u;  // "NET0" + 17

  constexpr std::uint32_t kFastSource = 0u, kFastTarget = 1u;
  constexpr std::uint32_t kSlowSource = 2u, kSlowTarget = 3u;
  genome.territories[genome.header.territory_count++] =
      net17_territory(kFastSource, kFastProtectiveSourceChemistry, kFastReach);
  genome.territories[genome.header.territory_count++] =
      net17_territory(kFastTarget, kFastProtectiveTargetChemistry, kFastReach);
  genome.territories[genome.header.territory_count++] =
      net17_territory(kSlowSource, kSlowAllostaticSourceChemistry, kSlowReach);
  genome.territories[genome.header.territory_count++] =
      net17_territory(kSlowTarget, kSlowAllostaticTargetChemistry, kSlowReach);

  // local_receptor_like_recipe_compatibility: the fast source's corridor
  // matches ONLY the fast target's chemistry, so it cannot land on the slow
  // loop's territories.
  DirectFieldSpecV1 fast_corridor{};
  fast_corridor.territory.lineage = kNet17Lineage;
  fast_corridor.territory.axis = 0u;
  fast_corridor.territory.ordinal = kFastSource;
  fast_corridor.radius = kNet17AffinityRadius;
  fast_corridor.require_mask = 0xffffffffu;
  fast_corridor.require_value = kFastProtectiveTargetChemistry;
  fast_corridor.write_mask = 0xffffffffu;
  fast_corridor.write_value = static_cast<std::uint32_t>(kNet17AffinityStrengthQ16);
  fast_corridor.begin_tick = 0u;
  fast_corridor.end_tick = kNet17DevelopmentEndTick;
  fast_corridor.polarity = 0u;  // DevelopmentFieldKind::attract
  const std::uint32_t fast_corridor_index = genome.header.field_count++;
  genome.fields[fast_corridor_index] = fast_corridor;

  // the slow source's corridor matches ONLY the slow target's chemistry --
  // the other receptor-compatible family, deliberately incompatible with the
  // fast loop's territories.
  DirectFieldSpecV1 slow_corridor = fast_corridor;
  slow_corridor.territory.ordinal = kSlowSource;
  slow_corridor.require_value = kSlowAllostaticTargetChemistry;
  const std::uint32_t slow_corridor_index = genome.header.field_count++;
  genome.fields[slow_corridor_index] = slow_corridor;

  const std::uint32_t chemistries[] = {kFastProtectiveSourceChemistry, kFastProtectiveTargetChemistry,
                                       kSlowAllostaticSourceChemistry, kSlowAllostaticTargetChemistry};
  for (const std::uint32_t chemistry : chemistries) {
    const bool is_fast = chemistry == kFastProtectiveSourceChemistry ||
                         chemistry == kFastProtectiveTargetChemistry;
    const std::uint32_t reach = is_fast ? kFastReach : kSlowReach;
    DirectRuleSpecV1 grow = net17_rule(DirectRuleOpcodeV1::extend, chemistry);
    grow.extent = reach;
    grow.child_slot = kNet17Degree;
    grow.branch_count = kNet17Population;
    genome.rules[genome.header.rule_count++] = grow;

    if (chemistry == kFastProtectiveSourceChemistry || chemistry == kSlowAllostaticSourceChemistry) {
      DirectRuleSpecV1 tract = net17_rule(DirectRuleOpcodeV1::long_tract, chemistry);
      tract.branch_count = kNet17LongTracts;
      const bool fast_leg = chemistry == kFastProtectiveSourceChemistry;
      tract.field_index = fast_leg ? fast_corridor_index : slow_corridor_index;
      // short_to_medium_delays: the fast leg stays at the pure geometric
      // delay; the slow leg carries an authored transport cost on top.
      if (!fast_leg) tract.extent = kSlowAllostaticTransportCost;
      genome.rules[genome.header.rule_count++] = tract;
    }

    DirectRuleSpecV1 ports = net17_rule(DirectRuleOpcodeV1::endogenous_source, chemistry);
    ports.extent = kNet17PortReserve;
    genome.rules[genome.header.rule_count++] = ports;
  }
  return genome;
}

}  // namespace substrate::direct_network
