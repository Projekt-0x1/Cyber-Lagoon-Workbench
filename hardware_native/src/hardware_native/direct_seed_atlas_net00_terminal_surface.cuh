namespace substrate::direct_network {
namespace {

// One chemotype per atlas territory. These are IDENTITIES, not meanings: the
// Life Function matches rules to territories by chemistry, so a territory needs
// a distinct one to be addressable at all. No value here encodes a symbol.
constexpr std::uint32_t kSurfaceBinderChemistry = 0x21u;
constexpr std::uint32_t kSuccessorChemistry = 0x22u;
constexpr std::uint32_t kExpressionChemistry = 0x24u;

constexpr std::uint32_t kDevelopmentEndTick = 4096u;

// `strong_local_recurrence`: a TIGHT reach with a HIGH per-node degree. The
// planner takes plan.radius = max(8, extent) and clamps child_slot into
// sparse_degree, so a small extent with a large child_slot is precisely "many
// routes, all of them near" in this substrate's own terms.
constexpr std::uint32_t kLocalRecurrenceReach = 12u;
constexpr std::uint32_t kLocalRecurrenceDegree = 12u;
constexpr std::uint32_t kSurfaceBinderNodes = 256u;

// `dense_short_range_temporal_neighbors`: `fuse` is the only opcode that sets
// kRuleFlagDenseIntegrative and sizes dense_width, and the planner clamps that
// width into [16, min(node_count, kDenseWidthLimit)] and rounds it down to a
// multiple of 16 -- so the authored number is a request, and the contract
// measures what survives.
constexpr std::uint32_t kDenseNeighbourhood = 64u;

// `multiple_delays`: route delay is distance/32 for a sparse route and
// distance/128 for a long tract, clamped to 1..16 and 1..64 respectively. A
// territory carrying BOTH therefore spans two delay regimes rather than one.
// This is the substrate's only authored delay lever -- there is no delay field
// in DirectRuleSpecV1.
constexpr std::uint32_t kLongTractCount = 8u;

// `high_one_shot_port_capacity`: the spare adjacency each node carries for
// bindings it has not made yet. This term used to be UNAUTHORABLE -- route
// capacity was sparse_degree + long_slot + DirectCompileOptions::
// route_reserve_per_node, and that reserve is a caller's argument, so two
// organisms whose genomes agreed byte for byte grew different port capacity
// depending on who compiled them. `endogenous_source` -- the opcode that says a
// territory interfaces with something outside itself, and the one opcode the
// planner read nothing from -- now carries it in `extent`, and the host option
// is only a floor.
constexpr std::uint32_t kOneShotPortReserve = 24u;

DirectRuleSpecV1 rule(DirectRuleOpcodeV1 opcode, std::uint32_t chemistry) {
  DirectRuleSpecV1 out{};
  out.opcode = opcode;
  out.begin_tick = 0u;
  out.end_tick = kDevelopmentEndTick;
  out.require_mask = 0xffffffffu;
  out.require_value = chemistry;
  out.write_mask = 0xffffffffu;
  out.write_value = chemistry;
  return out;
}

DirectTerritorySpecV1 territory(std::uint32_t ordinal, std::uint32_t chemistry,
                                std::uint32_t reach) {
  DirectTerritorySpecV1 out{};
  out.identity.lineage = 0u;  // NET00 is one lineage; NET01+ get their own.
  out.identity.axis = 0u;
  out.identity.ordinal = ordinal;
  out.chemotype = chemistry;
  out.reach = reach;
  out.begin_tick = 0u;
  return out;
}

}  // namespace

DirectGenomeV1 seed_atlas_net00_terminal_surface() {
  DirectGenomeV1 genome{};
  genome.header.life_function_version = 2u;
  genome.header.development_end_tick = kDevelopmentEndTick;
  genome.header.matter_budget = 1u << 24;
  // A species constant, not an individual one: two individuals of this species
  // differ through Xi (#1342), never through this value.
  genome.header.development_seed = 0x4e455430u;  // "NET0"

  genome.territories[genome.header.territory_count++] =
      territory(0u, kSurfaceBinderChemistry, kLocalRecurrenceReach);
  genome.territories[genome.header.territory_count++] =
      territory(1u, kSuccessorChemistry, kLocalRecurrenceReach);
  genome.territories[genome.header.territory_count++] =
      territory(2u, kExpressionChemistry, kLocalRecurrenceReach);

  const std::uint32_t chemistries[] = {kSurfaceBinderChemistry, kSuccessorChemistry,
                                       kExpressionChemistry};
  for (const std::uint32_t chemistry : chemistries) {
    // strong_local_recurrence
    DirectRuleSpecV1 grow = rule(DirectRuleOpcodeV1::extend, chemistry);
    grow.extent = kLocalRecurrenceReach;
    grow.child_slot = kLocalRecurrenceDegree;
    grow.branch_count = kSurfaceBinderNodes;
    genome.rules[genome.header.rule_count++] = grow;

    // dense_short_range_temporal_neighbors
    DirectRuleSpecV1 dense = rule(DirectRuleOpcodeV1::fuse, chemistry);
    dense.branch_count = kDenseNeighbourhood;
    genome.rules[genome.header.rule_count++] = dense;

    // multiple_delays
    DirectRuleSpecV1 tract = rule(DirectRuleOpcodeV1::long_tract, chemistry);
    tract.branch_count = kLongTractCount;
    genome.rules[genome.header.rule_count++] = tract;

    // high_one_shot_port_capacity
    DirectRuleSpecV1 ports = rule(DirectRuleOpcodeV1::endogenous_source, chemistry);
    ports.extent = kOneShotPortReserve;
    genome.rules[genome.header.rule_count++] = ports;
  }
  return genome;
}

}  // namespace substrate::direct_network
