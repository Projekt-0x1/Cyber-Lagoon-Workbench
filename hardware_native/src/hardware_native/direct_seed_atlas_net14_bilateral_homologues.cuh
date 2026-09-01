namespace substrate::direct_network {
namespace {

constexpr std::uint32_t kNet14Lineage = 14u;  // matches the atlas's own NET14 numbering.
constexpr std::uint32_t kNet14HomologueAChemistry = 0xf1u;
constexpr std::uint32_t kNet14HomologueBChemistry = 0xf2u;

constexpr std::uint32_t kNet14DevelopmentEndTick = 4096u;

// Population/dense_width/long_tracts match the species table's own NET14 row
// (territories=2, long_tracts=20) so the dedicated genome disagrees with the
// generic one only on the weak per-homologue bias the table's single reach/
// degree pair cannot express.
constexpr std::uint32_t kNet14Population = 96u;
constexpr std::uint32_t kNet14LongTracts = 20u;
constexpr std::uint32_t kNet14PortReserve = 16u;

// The table's own reach/degree (32/8) is the SYMMETRIC baseline both
// homologues start from. `asymmetry_parameter_seed_is_weak_bias_not_
// semantic_assignment` is read literally: the favored homologue's degree
// moves by one tenth of the baseline, not a qualitatively different shape --
// weak enough that both homologues still grow real, comparable capacity
// (`redundancy_without_exact_duplication`), strong enough to be a measurable,
// non-accidental bias rather than `force_perfect_symmetry`.
constexpr std::uint32_t kNet14BaselineReach = 32u;
constexpr std::uint32_t kNet14BaselineDegree = 8u;
constexpr std::uint32_t kNet14FavoredDegreeBias = 1u;

constexpr std::uint32_t kNet14AffinityRadius = 1u << 20;
constexpr std::int32_t kNet14AffinityStrengthQ16 = 1 << 16;

DirectRuleSpecV1 net14_rule(DirectRuleOpcodeV1 opcode, std::uint32_t chemistry) {
  DirectRuleSpecV1 out{};
  out.opcode = opcode;
  out.begin_tick = 0u;
  out.end_tick = kNet14DevelopmentEndTick;
  out.require_mask = 0xffffffffu;
  out.require_value = chemistry;
  out.write_mask = 0xffffffffu;
  out.write_value = chemistry;
  return out;
}

DirectTerritorySpecV1 net14_territory(std::uint32_t ordinal, std::uint32_t chemistry) {
  DirectTerritorySpecV1 out{};
  out.identity.lineage = kNet14Lineage;
  out.identity.axis = 0u;
  out.identity.ordinal = ordinal;
  out.chemotype = chemistry;
  out.reach = kNet14BaselineReach;
  out.begin_tick = 0u;
  return out;
}

// One directed commissural field, owned by `owner_ordinal`, seeking
// `partner_chemistry`. Same construction NET15's crossmodal corridors use: a
// generous radius so affinity is expressed by chemistry rather than a guessed
// coordinate, inert for the owner's own placement.
DirectFieldSpecV1 net14_corridor(std::uint32_t owner_ordinal, std::uint32_t partner_chemistry) {
  DirectFieldSpecV1 out{};
  out.territory.lineage = kNet14Lineage;
  out.territory.axis = 0u;
  out.territory.ordinal = owner_ordinal;
  out.radius = kNet14AffinityRadius;
  out.require_mask = 0xffffffffu;
  out.require_value = partner_chemistry;
  out.write_mask = 0xffffffffu;
  out.write_value = static_cast<std::uint32_t>(kNet14AffinityStrengthQ16);
  out.begin_tick = 0u;
  out.end_tick = kNet14DevelopmentEndTick;
  out.polarity = 0u;  // DevelopmentFieldKind::attract
  return out;
}

}  // namespace

// `homologue_a_favored` selects which side carries the weak degree bias
// rather than hardcoding one: `true` favors homologue A (ordinal 0), `false`
// mirrors it onto homologue B (ordinal 1) -- the atlas's own
// mirrored_Gamma_asymmetry_swap probe, and the direct refutation of
// hardcode_left_equals_language_truth (neither ordinal is "left"; whichever
// one is favored is a parameter of the call, not an identity of the
// territory). The public no-arg caller always favors homologue A.
DirectGenomeV1 seed_atlas_net14_bilateral_homologues(bool homologue_a_favored) {
  DirectGenomeV1 genome{};
  genome.header.life_function_version = 2u;
  genome.header.development_end_tick = kNet14DevelopmentEndTick;
  genome.header.matter_budget = 1u << 24;
  // A species constant, not an individual one: two individuals of this
  // species differ through Xi (#1342), never through this value.
  genome.header.development_seed = 0x4e455430u + 14u;  // "NET0" + 14

  genome.territories[genome.header.territory_count++] =
      net14_territory(0u, kNet14HomologueAChemistry);
  genome.territories[genome.header.territory_count++] =
      net14_territory(1u, kNet14HomologueBChemistry);

  // strong_commissural_growth_corridors: bidirectional, both directions
  // authored the same way -- the asymmetry lives in growth bias, never in
  // which side the commissure itself reaches.
  const std::uint32_t a_to_b_field = genome.header.field_count++;
  genome.fields[a_to_b_field] = net14_corridor(0u, kNet14HomologueBChemistry);
  const std::uint32_t b_to_a_field = genome.header.field_count++;
  genome.fields[b_to_a_field] = net14_corridor(1u, kNet14HomologueAChemistry);

  const std::uint32_t chemistries[] = {kNet14HomologueAChemistry, kNet14HomologueBChemistry};
  const std::uint32_t fields[] = {a_to_b_field, b_to_a_field};
  const std::uint32_t degrees[] = {
      kNet14BaselineDegree + (homologue_a_favored ? kNet14FavoredDegreeBias : 0u),
      kNet14BaselineDegree + (homologue_a_favored ? 0u : kNet14FavoredDegreeBias)};
  for (std::uint32_t i = 0u; i < 2u; ++i) {
    const std::uint32_t chemistry = chemistries[i];

    DirectRuleSpecV1 grow = net14_rule(DirectRuleOpcodeV1::extend, chemistry);
    grow.extent = kNet14BaselineReach;
    grow.child_slot = degrees[i];
    grow.branch_count = kNet14Population;
    genome.rules[genome.header.rule_count++] = grow;

    DirectRuleSpecV1 commissure = net14_rule(DirectRuleOpcodeV1::long_tract, chemistry);
    commissure.branch_count = kNet14LongTracts;
    commissure.field_index = fields[i];
    genome.rules[genome.header.rule_count++] = commissure;

    DirectRuleSpecV1 ports = net14_rule(DirectRuleOpcodeV1::endogenous_source, chemistry);
    ports.extent = kNet14PortReserve;
    genome.rules[genome.header.rule_count++] = ports;
  }
  return genome;
}

}  // namespace substrate::direct_network
