namespace substrate::direct_network {
namespace {

// NET09_CORTICO_STRIATO_PALLIDO_THALAMO_CORTICAL_SELECTION_LOOPS.
//
// The atlas block asks for `parallel_competitive_loop_families`,
// `output_returns_to_cortex_via_thalamus` and `short_to_medium_delays`, and
// names two kill conditions: `one_global_reward_scalar` and
// `one_global_action_winner_for_all_motor_surfaces`.
//
// What makes this family different from every one seeded before it is that its
// motif is a CYCLE, not a stream. NET01 and NET02 author corridors that leave a
// territory and arrive somewhere else; NET09 authors corridors that come back.
// Nothing new was needed for that: a return leg is just the last station's own
// corridor family addressed to the first station's chemistry. The loop closes
// because four ordinary `long_tract` rules happen to name a cycle of
// chemistries -- there is no loop opcode, no cycle descriptor, and nothing in
// the substrate that knows these twelve territories are three of anything.
//
// The channels are parallel in the strong sense the atlas means: three
// independent four-station loops whose chemistries do not overlap, so no
// corridor can cross from one channel into another and no station is shared.
// That is the structural form of `no_single_global_winner` -- cutting one
// channel's return cannot reach the others, which is what the contract
// measures with the atlas's own `thalamic_return_cut` probe.
constexpr std::uint32_t kNet09Lineage = 9u;
constexpr std::uint32_t kNet09Channels = 3u;
constexpr std::uint32_t kNet09Stations = 4u;  // cortex, striatum, pallidum, thalamus

constexpr std::uint32_t kNet09DevelopmentEndTick = 4096u;
// Deliberately smaller per-station population than NET01/NET02: this family
// contributes twelve territories rather than three, and the atlas motif is
// "many parallel channels", not "one large sheet".
constexpr std::uint32_t kNet09Population = 96u;
constexpr std::uint32_t kNet09Reach = 16u;
constexpr std::uint32_t kNet09Degree = 8u;
constexpr std::uint32_t kNet09LongTracts = 8u;
constexpr std::uint32_t kNet09PortReserve = 16u;
constexpr std::uint32_t kNet09AffinityRadius = 1u << 20;
constexpr std::int32_t kNet09AffinityStrengthQ16 = 1 << 16;

// `short_to_medium_delays`, authored through the transport cost `b0a747d01a`
// gave the `long_tract` rule's `extent`. The three intra-loop hops are short;
// the thalamic return to cortex is the medium one, which is the leg the atlas
// singles out (`output_returns_to_cortex_via_thalamus`).
constexpr std::uint32_t kNet09HopCost = 4u;
constexpr std::uint32_t kNet09ReturnCost = 40u;

// Channel c, station s. Channels are 0x10 apart so no two channels can ever
// share a chemistry, which is what keeps the loops parallel rather than
// convergent -- convergence with competition is a later rung and needs the
// arbitration this family does not yet author.
constexpr std::uint32_t net09_chemistry(std::uint32_t channel, std::uint32_t station) {
  return 0x900u + channel * 0x10u + station;
}

DirectRuleSpecV1 net09_rule(DirectRuleOpcodeV1 opcode, std::uint32_t chemistry) {
  DirectRuleSpecV1 out{};
  out.opcode = opcode;
  out.begin_tick = 0u;
  out.end_tick = kNet09DevelopmentEndTick;
  out.require_mask = 0xffffffffu;
  out.require_value = chemistry;
  out.write_mask = 0xffffffffu;
  out.write_value = chemistry;
  return out;
}

}  // namespace

DirectGenomeV1 seed_atlas_net09_selection_loops() {
  DirectGenomeV1 genome{};
  genome.header.life_function_version = 2u;
  genome.header.development_end_tick = kNet09DevelopmentEndTick;
  genome.header.matter_budget = 1u << 24;
  genome.header.development_seed = 0x4c4f4f50u;  // "LOOP"

  for (std::uint32_t channel = 0; channel < kNet09Channels; ++channel) {
    for (std::uint32_t station = 0; station < kNet09Stations; ++station) {
      DirectTerritorySpecV1 territory{};
      territory.identity.lineage = kNet09Lineage;
      territory.identity.axis = channel;
      territory.identity.ordinal = station;
      territory.chemotype = net09_chemistry(channel, station);
      territory.reach = kNet09Reach;
      territory.begin_tick = 0u;
      genome.territories[genome.header.territory_count++] = territory;
    }
  }

  // One corridor family per station, addressed to the NEXT station in its own
  // channel, wrapping from the last back to the first. The wrap is the whole
  // point: it is an ordinary field, and it is what closes the loop.
  for (std::uint32_t channel = 0; channel < kNet09Channels; ++channel) {
    for (std::uint32_t station = 0; station < kNet09Stations; ++station) {
      const std::uint32_t next = (station + 1u) % kNet09Stations;
      const bool is_return = next == 0u;

      DirectFieldSpecV1 corridor{};
      corridor.territory.lineage = kNet09Lineage;
      corridor.territory.axis = channel;
      corridor.territory.ordinal = station;
      corridor.radius = kNet09AffinityRadius;
      corridor.require_mask = 0xffffffffu;
      corridor.require_value = net09_chemistry(channel, next);
      corridor.write_mask = 0xffffffffu;
      corridor.write_value = static_cast<std::uint32_t>(kNet09AffinityStrengthQ16);
      corridor.begin_tick = 0u;
      corridor.end_tick = kNet09DevelopmentEndTick;
      corridor.polarity = 0u;  // DevelopmentFieldKind::attract
      const std::uint32_t corridor_index = genome.header.field_count++;
      genome.fields[corridor_index] = corridor;

      const std::uint32_t chemistry = net09_chemistry(channel, station);

      DirectRuleSpecV1 grow = net09_rule(DirectRuleOpcodeV1::extend, chemistry);
      grow.extent = kNet09Reach;
      grow.child_slot = kNet09Degree;
      grow.branch_count = kNet09Population;
      genome.rules[genome.header.rule_count++] = grow;

      DirectRuleSpecV1 tract = net09_rule(DirectRuleOpcodeV1::long_tract, chemistry);
      tract.branch_count = kNet09LongTracts;
      tract.field_index = corridor_index;
      tract.extent = is_return ? kNet09ReturnCost : kNet09HopCost;
      genome.rules[genome.header.rule_count++] = tract;

      DirectRuleSpecV1 ports = net09_rule(DirectRuleOpcodeV1::endogenous_source, chemistry);
      ports.extent = kNet09PortReserve;
      genome.rules[genome.header.rule_count++] = ports;
    }
  }
  return genome;
}

}  // namespace substrate::direct_network
