namespace substrate::direct_network {
namespace {

constexpr std::uint32_t kNet01Lineage = 1u;  // NET00 is lineage 0; each network is its own.
constexpr std::uint32_t kPosteriorTemporalChemistry = 0x31u;
constexpr std::uint32_t kInferiorParietalChemistry = 0x32u;
constexpr std::uint32_t kInferiorFrontalChemistry = 0x34u;

constexpr std::uint32_t kNet01DevelopmentEndTick = 4096u;
constexpr std::uint32_t kNet01Reach = 16u;
constexpr std::uint32_t kNet01Degree = 10u;
constexpr std::uint32_t kNet01Population = 192u;
constexpr std::uint32_t kNet01DenseWidth = 32u;
constexpr std::uint32_t kNet01LongTracts = 12u;
constexpr std::uint32_t kNet01PortReserve = 24u;

// gh #1268/#1267. `delay_distribution: fast_medium`: the direct dorsal
// corridor stays at the pure geometric delay (0 authored cost, the fast
// leg); each hop of the indirect route (pt->ip, then ip->if -- a real
// two-hop path since 51ba5cea71 gave the second hop somewhere to land)
// carries this transport cost, so the round-trip indirect path is slower by
// two authored hops' worth rather than one. The mechanism and its clamp
// were proven in cuda_direct_corridor_delay_carrier_contract this session;
// this is the first genome in the tree to actually author it.
constexpr std::uint32_t kNet01IndirectTransportCost = 20u;

// gh #1268/#1267. `tract_maturation`: the sixth leg (inferior_parietal's own
// back-projection to posterior_temporal, closed in e3c28dd98e) recruits only
// nodes born at or after this developmental tick. Feedback/recurrent corridors
// typically stabilize later than the feedforward pathways they trail, so this
// leg -- the last one closed, and the only purely-backward leg with no
// forward sibling terminating on the same partner -- is the first family in
// the tree to actually author a nonzero minimum_age (the mechanism landed in
// 627e280ce5, `cuda_direct_corridor_maturation_window_contract`). Not a
// tuned/measured value -- a placeholder demonstrating "recruits measurably
// later", the same role kNet01IndirectTransportCost played before any tuning
// question was asked of it.
constexpr std::uint32_t kNet01BackprojectionMaturationTick = 350u;

// The direct corridor's field reach: generous on purpose. This function does
// not know where the lattice arbiter will place inferior_frontal, and it must
// not guess a coordinate -- affinity is expressed by CHEMISTRY, matched
// against the candidate target regardless of where the arbiter lands it.
constexpr std::uint32_t kNet01AffinityRadius = 1u << 20;
constexpr std::int32_t kNet01AffinityStrengthQ16 = 1 << 16;

DirectRuleSpecV1 net01_rule(DirectRuleOpcodeV1 opcode, std::uint32_t chemistry) {
  DirectRuleSpecV1 out{};
  out.opcode = opcode;
  out.begin_tick = 0u;
  out.end_tick = kNet01DevelopmentEndTick;
  out.require_mask = 0xffffffffu;
  out.require_value = chemistry;
  out.write_mask = 0xffffffffu;
  out.write_value = chemistry;
  return out;
}

DirectTerritorySpecV1 net01_territory(std::uint32_t ordinal, std::uint32_t chemistry) {
  DirectTerritorySpecV1 out{};
  out.identity.lineage = kNet01Lineage;
  out.identity.axis = 0u;
  out.identity.ordinal = ordinal;
  out.chemotype = chemistry;
  out.reach = kNet01Reach;
  out.begin_tick = 0u;
  return out;
}

}  // namespace

DirectGenomeV1 seed_atlas_net01_dorsal_stream() {
  DirectGenomeV1 genome{};
  genome.header.life_function_version = 2u;
  genome.header.development_end_tick = kNet01DevelopmentEndTick;
  genome.header.matter_budget = 1u << 24;
  // A species constant, not an individual one: two individuals of this species
  // differ through Xi (#1342), never through this value.
  genome.header.development_seed = 0x4e455431u;  // "NET1"

  genome.territories[genome.header.territory_count++] =
      net01_territory(0u, kPosteriorTemporalChemistry);
  genome.territories[genome.header.territory_count++] =
      net01_territory(1u, kInferiorParietalChemistry);
  genome.territories[genome.header.territory_count++] =
      net01_territory(2u, kInferiorFrontalChemistry);

  // The direct dorsal corridor (github #1268's "A<->C long_tract_affinity"):
  // owned by posterior_temporal, attract, seeks inferior_frontal's chemistry
  // rather than its own. Inert for posterior_temporal's own node placement,
  // by the same construction 007ac93107 measured: chemistry_matches fails
  // when this field is evaluated against posterior_temporal's own chemotype,
  // and only fires when evaluated against a candidate corridor target's.
  DirectFieldSpecV1 direct_corridor{};
  direct_corridor.territory.lineage = kNet01Lineage;
  direct_corridor.territory.axis = 0u;
  direct_corridor.territory.ordinal = 0u;  // posterior_temporal
  direct_corridor.radius = kNet01AffinityRadius;
  direct_corridor.require_mask = 0xffffffffu;
  direct_corridor.require_value = kInferiorFrontalChemistry;
  direct_corridor.write_mask = 0xffffffffu;
  direct_corridor.write_value = static_cast<std::uint32_t>(kNet01AffinityStrengthQ16);
  direct_corridor.begin_tick = 0u;
  direct_corridor.end_tick = kNet01DevelopmentEndTick;
  direct_corridor.polarity = 0u;  // DevelopmentFieldKind::attract
  const std::uint32_t direct_corridor_index = genome.header.field_count++;
  genome.fields[direct_corridor_index] = direct_corridor;

  // The INDIRECT dorsal corridor: the same owner territory, a DIFFERENT partner
  // chemistry. This is the leg `9e87e869d0` had to leave unauthored -- the
  // planner collapsed every long_tract rule into one count and read the partner
  // address off a single last-write-wins slot, so authoring this field simply
  // overwrote the direct corridor's rather than adding a second corridor. It is
  // authored here as an ordinary second field plus an ordinary second
  // long_tract rule: no new opcode, no new descriptor, and nothing anywhere
  // that names NET01, a stream, or a language function.
  DirectFieldSpecV1 indirect_corridor = direct_corridor;
  indirect_corridor.require_value = kInferiorParietalChemistry;
  const std::uint32_t indirect_corridor_index = genome.header.field_count++;
  genome.fields[indirect_corridor_index] = indirect_corridor;

  // The indirect route's SECOND HOP (github #1268/82bef40571): the atlas calls
  // the direct and indirect dorsal routes "intentionally parallel", which means
  // alternative paths between the SAME endpoints -- not a fork that leaves
  // posterior_temporal for two destinations that never rejoin. Without this,
  // inferior_parietal's long_tract rule names no partner, and whatever routes
  // it grows toward inferior_frontal are chosen by geometry against wherever Xi
  // happened to place it -- an accident of one individual, not a species
  // corridor. Owned by inferior_parietal, attract, seeks inferior_frontal's
  // chemistry -- the same construction as both legs above, applied one hop
  // further down the indirect route.
  DirectFieldSpecV1 second_hop_corridor = direct_corridor;
  second_hop_corridor.territory.ordinal = 1u;  // inferior_parietal
  second_hop_corridor.require_value = kInferiorFrontalChemistry;
  const std::uint32_t second_hop_corridor_index = genome.header.field_count++;
  genome.fields[second_hop_corridor_index] = second_hop_corridor;

  // The BACK-PROJECTIONS (github #1268's `recurrent_bidirectional_coupling`
  // motif term; the atlas's own growth terms read `A<->C long_tract_affinity`
  // and `A<->B strong_affinity` -- `<->`, not `->`). Every leg authored above
  // runs one direction only. inferior_frontal owns two long_tract rules of its
  // own here, seeking posterior_temporal's chemistry and inferior_parietal's
  // in turn -- the reverse of the two hops that already terminate on it.
  // Without these, inferior_frontal's long_tract rule names no partner and
  // whatever it grows back is geometry's accident, the same gap 82bef40571
  // measured and 51ba5cea71 closed for the forward direction, one hop further
  // down the graph. inferior_parietal's own back-projection to
  // posterior_temporal -- the sixth and final directed leg -- is authored a
  // few lines further down, after the field this pattern shares is in scope.
  DirectFieldSpecV1 backproject_to_pt = direct_corridor;
  backproject_to_pt.territory.ordinal = 2u;  // inferior_frontal
  backproject_to_pt.require_value = kPosteriorTemporalChemistry;
  const std::uint32_t backproject_to_pt_index = genome.header.field_count++;
  genome.fields[backproject_to_pt_index] = backproject_to_pt;

  DirectFieldSpecV1 backproject_to_ip = backproject_to_pt;
  backproject_to_ip.require_value = kInferiorParietalChemistry;
  const std::uint32_t backproject_to_ip_index = genome.header.field_count++;
  genome.fields[backproject_to_ip_index] = backproject_to_ip;

  // The SIXTH and final leg (github #1268, a6f5a39a7d's own next_attack):
  // inferior_parietal's own back-projection to posterior_temporal, owned by
  // inferior_parietal, attract, seeks posterior_temporal's chemistry -- the
  // same construction as every corridor above, closing the last unauthored
  // direction of the dorsal matrix.
  DirectFieldSpecV1 backproject_ip_to_pt = direct_corridor;
  backproject_ip_to_pt.territory.ordinal = 1u;  // inferior_parietal
  backproject_ip_to_pt.require_value = kPosteriorTemporalChemistry;
  const std::uint32_t backproject_ip_to_pt_index = genome.header.field_count++;
  genome.fields[backproject_ip_to_pt_index] = backproject_ip_to_pt;

  const std::uint32_t chemistries[] = {kPosteriorTemporalChemistry, kInferiorParietalChemistry,
                                       kInferiorFrontalChemistry};
  for (const std::uint32_t chemistry : chemistries) {
    DirectRuleSpecV1 grow = net01_rule(DirectRuleOpcodeV1::extend, chemistry);
    grow.extent = kNet01Reach;
    grow.child_slot = kNet01Degree;
    grow.branch_count = kNet01Population;
    genome.rules[genome.header.rule_count++] = grow;

    DirectRuleSpecV1 dense = net01_rule(DirectRuleOpcodeV1::fuse, chemistry);
    dense.branch_count = kNet01DenseWidth;
    genome.rules[genome.header.rule_count++] = dense;

    DirectRuleSpecV1 tract = net01_rule(DirectRuleOpcodeV1::long_tract, chemistry);
    tract.branch_count = kNet01LongTracts;
    if (chemistry == kPosteriorTemporalChemistry) {
      tract.field_index = direct_corridor_index;
      // The direct leg stays at the pure geometric delay -- the fast half of
      // fast_medium.
    } else if (chemistry == kInferiorParietalChemistry) {
      tract.field_index = second_hop_corridor_index;
      // The indirect route's second hop -- the medium half of fast_medium.
      tract.extent = kNet01IndirectTransportCost;
    } else if (chemistry == kInferiorFrontalChemistry) {
      tract.field_index = backproject_to_pt_index;
    }
    genome.rules[genome.header.rule_count++] = tract;

    if (chemistry == kPosteriorTemporalChemistry) {
      // A second long_tract rule IS a second corridor family (#1268): its field
      // is the family's partner address and its branch_count is how many of
      // posterior_temporal's nodes originate it. The two families partition the
      // territory's long-tract nodes, so this adds tracts without widening any
      // node's route capacity.
      DirectRuleSpecV1 indirect = net01_rule(DirectRuleOpcodeV1::long_tract, chemistry);
      indirect.branch_count = kNet01LongTracts;
      indirect.field_index = indirect_corridor_index;
      // The indirect route's first hop -- the medium half of fast_medium.
      indirect.extent = kNet01IndirectTransportCost;
      genome.rules[genome.header.rule_count++] = indirect;
    } else if (chemistry == kInferiorFrontalChemistry) {
      // inferior_frontal's second family: the same construction, its second
      // back-projection.
      DirectRuleSpecV1 backproject = net01_rule(DirectRuleOpcodeV1::long_tract, chemistry);
      backproject.branch_count = kNet01LongTracts;
      backproject.field_index = backproject_to_ip_index;
      genome.rules[genome.header.rule_count++] = backproject;
    } else if (chemistry == kInferiorParietalChemistry) {
      // inferior_parietal's second family: its own back-projection to
      // posterior_temporal, the sixth and final directed leg. Left at the
      // pure geometric delay, matching the other two back-projections above
      // rather than the forward indirect hops' authored transport cost.
      DirectRuleSpecV1 backproject_ip = net01_rule(DirectRuleOpcodeV1::long_tract, chemistry);
      backproject_ip.branch_count = kNet01LongTracts;
      backproject_ip.field_index = backproject_ip_to_pt_index;
      // The first family in the tree to author tract_maturation for real: this
      // leg recruits only nodes born after kNet01BackprojectionMaturationTick.
      backproject_ip.minimum_age = kNet01BackprojectionMaturationTick;
      genome.rules[genome.header.rule_count++] = backproject_ip;
    }

    DirectRuleSpecV1 ports = net01_rule(DirectRuleOpcodeV1::endogenous_source, chemistry);
    ports.extent = kNet01PortReserve;
    genome.rules[genome.header.rule_count++] = ports;
  }
  return genome;
}

}  // namespace substrate::direct_network
