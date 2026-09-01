namespace substrate::direct_network {
namespace {

constexpr std::uint32_t kNet04Lineage = 4u;  // matches the atlas's own NET04 numbering.
constexpr std::uint32_t kAnteriorChemistry = 0xa1u;
constexpr std::uint32_t kPosteriorChemistry = 0xa2u;
constexpr std::uint32_t kTransmodalChemistry = 0xa3u;
constexpr std::uint32_t kSensoryChemistry = 0xa4u;

constexpr std::uint32_t kNet04DevelopmentEndTick = 4096u;
// anterior_posterior_connectivity_differentiation: the anterior segment is the
// BROAD, convergent one that reaches transmodal territory; the posterior
// segment is the NARROWER, fine-grained one that reaches sensory territory.
// The two reaches differ so the segments are not each other's relabelling.
constexpr std::uint32_t kAnteriorBroadReach = 32u;
constexpr std::uint32_t kPosteriorNarrowReach = 10u;
constexpr std::uint32_t kCorticalReach = 16u;
// sparse_distinct_bindings: high fanout per node, small population per
// territory -- the atlas's "sparse, high-dimensional" reading of the index.
constexpr std::uint32_t kNet04Degree = 16u;
constexpr std::uint32_t kNet04Population = 128u;
constexpr std::uint32_t kNet04LongTracts = 10u;
constexpr std::uint32_t kNet04PortReserve = 16u;

// Same construction as NET01/NET02/NET12: generous radius, chemistry does the
// matching, no host-authored coordinate.
constexpr std::uint32_t kNet04AffinityRadius = 1u << 20;
constexpr std::int32_t kNet04AffinityStrengthQ16 = 1 << 16;

// rapid_episode_index_distinct_from_slower_cortical_consolidation (#1269,
// timescale rung): posterior is the FAST-decaying segment, anterior stays
// PERSISTENT (class 0, no decay). Posterior already carries the narrower,
// finer-grained reach and the sensory partner -- the detail-preserving half
// of the anterior/posterior gradient this seed already authors -- while
// anterior carries the broad reach into transmodal/associative cortex, the
// partner most plausibly doing the slower consolidation this index feeds.
// Packed above DevelopmentFieldKind via
// direct_network_life_function.cu's field_decay_class, the exact mechanism
// NET12 rung 2 proved on a real device (08c38a9803) -- no new ABI field, no
// new opcode.
constexpr std::uint32_t kPosteriorFastDecayClass = 3u;
constexpr std::uint32_t kAnteriorNoDecayClass = 0u;

DirectRuleSpecV1 net04_rule(DirectRuleOpcodeV1 opcode, std::uint32_t chemistry) {
  DirectRuleSpecV1 out{};
  out.opcode = opcode;
  out.begin_tick = 0u;
  out.end_tick = kNet04DevelopmentEndTick;
  out.require_mask = 0xffffffffu;
  out.require_value = chemistry;
  out.write_mask = 0xffffffffu;
  out.write_value = chemistry;
  return out;
}

DirectTerritorySpecV1 net04_territory(std::uint32_t ordinal, std::uint32_t chemistry,
                                      std::uint32_t reach) {
  DirectTerritorySpecV1 out{};
  out.identity.lineage = kNet04Lineage;
  out.identity.axis = 0u;
  out.identity.ordinal = ordinal;
  out.chemotype = chemistry;
  out.reach = reach;
  out.begin_tick = 0u;
  return out;
}

}  // namespace

DirectGenomeV1 seed_atlas_net04_binding_index(bool posterior_decays) {
  DirectGenomeV1 genome{};
  genome.header.life_function_version = 2u;
  genome.header.development_end_tick = kNet04DevelopmentEndTick;
  genome.header.matter_budget = 1u << 24;
  // A species constant, not an individual one: two individuals of this species
  // differ through Xi (#1342), never through this value.
  genome.header.development_seed = 0x4e455430u + 4u;  // "NET0" + 4

  genome.territories[genome.header.territory_count++] =
      net04_territory(0u, kAnteriorChemistry, kAnteriorBroadReach);
  genome.territories[genome.header.territory_count++] =
      net04_territory(1u, kPosteriorChemistry, kPosteriorNarrowReach);
  genome.territories[genome.header.territory_count++] =
      net04_territory(2u, kTransmodalChemistry, kCorticalReach);
  genome.territories[genome.header.territory_count++] =
      net04_territory(3u, kSensoryChemistry, kCorticalReach);

  // The differentiation itself: anterior's corridor field matches ONLY
  // transmodal's chemistry, so it cannot land on the sensory partner.
  DirectFieldSpecV1 anterior_corridor{};
  anterior_corridor.territory.lineage = kNet04Lineage;
  anterior_corridor.territory.axis = 0u;
  anterior_corridor.territory.ordinal = 0u;  // anterior
  anterior_corridor.radius = kNet04AffinityRadius;
  anterior_corridor.require_mask = 0xffffffffu;
  anterior_corridor.require_value = kTransmodalChemistry;
  anterior_corridor.write_mask = 0xffffffffu;
  anterior_corridor.write_value = static_cast<std::uint32_t>(kNet04AffinityStrengthQ16);
  anterior_corridor.begin_tick = 0u;
  anterior_corridor.end_tick = kNet04DevelopmentEndTick;
  // DevelopmentFieldKind::attract (0) + no-decay class, packed -- anterior is
  // always persistent (the reference class, see the comment above).
  anterior_corridor.polarity = 0u + kDevelopmentFieldKindCount * kAnteriorNoDecayClass;
  const std::uint32_t anterior_corridor_index = genome.header.field_count++;
  genome.fields[anterior_corridor_index] = anterior_corridor;

  // posterior's corridor field matches ONLY sensory's chemistry -- the other
  // half of the gradient, deliberately incompatible with transmodal.
  DirectFieldSpecV1 posterior_corridor = anterior_corridor;
  posterior_corridor.territory.ordinal = 1u;  // posterior
  posterior_corridor.require_value = kSensoryChemistry;
  // DevelopmentFieldKind::attract (0) + decay class, packed. The reference
  // configuration (posterior_decays=false) uses class 0 -- identical to
  // anterior's, and identical to this seed's pre-timescale-rung behavior.
  posterior_corridor.polarity =
      0u + kDevelopmentFieldKindCount *
               (posterior_decays ? kPosteriorFastDecayClass : kAnteriorNoDecayClass);
  const std::uint32_t posterior_corridor_index = genome.header.field_count++;
  genome.fields[posterior_corridor_index] = posterior_corridor;

  const std::uint32_t chemistries[] = {kAnteriorChemistry, kPosteriorChemistry,
                                       kTransmodalChemistry, kSensoryChemistry};
  for (const std::uint32_t chemistry : chemistries) {
    const std::uint32_t reach = (chemistry == kAnteriorChemistry)
                                    ? kAnteriorBroadReach
                                    : (chemistry == kPosteriorChemistry) ? kPosteriorNarrowReach
                                                                         : kCorticalReach;
    DirectRuleSpecV1 grow = net04_rule(DirectRuleOpcodeV1::extend, chemistry);
    grow.extent = reach;
    grow.child_slot = kNet04Degree;
    grow.branch_count = kNet04Population;
    genome.rules[genome.header.rule_count++] = grow;

    if (chemistry == kAnteriorChemistry || chemistry == kPosteriorChemistry) {
      DirectRuleSpecV1 tract = net04_rule(DirectRuleOpcodeV1::long_tract, chemistry);
      tract.branch_count = kNet04LongTracts;
      tract.field_index = (chemistry == kAnteriorChemistry) ? anterior_corridor_index
                                                            : posterior_corridor_index;
      genome.rules[genome.header.rule_count++] = tract;
    }

    DirectRuleSpecV1 ports = net04_rule(DirectRuleOpcodeV1::endogenous_source, chemistry);
    ports.extent = kNet04PortReserve;
    genome.rules[genome.header.rule_count++] = ports;
  }
  return genome;
}

}  // namespace substrate::direct_network
