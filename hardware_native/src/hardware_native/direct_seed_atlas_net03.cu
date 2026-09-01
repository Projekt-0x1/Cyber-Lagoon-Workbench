#include "hardware_native/direct_seed_atlas.cuh"

namespace substrate::direct_network {
namespace {

constexpr std::uint32_t kNet03Lineage = 3u;  // matches the atlas's own NET03 numbering.
constexpr std::uint32_t kNet03LocalChemistry = 0x71u;
constexpr std::uint32_t kNet03MediumChemistry = 0x72u;
constexpr std::uint32_t kNet03DiscourseChemistry = 0x74u;

constexpr std::uint32_t kNet03DevelopmentEndTick = 4096u;

// recurrence_time_constant_gradient: reach widens and degree thins,
// MONOTONICALLY, from local to discourse -- a tight, high-fanout local window
// through to a broad, sparse discourse one.
constexpr std::uint32_t kNet03LocalReach = 10u;
constexpr std::uint32_t kNet03MediumReach = 22u;
constexpr std::uint32_t kNet03DiscourseReach = 40u;
constexpr std::uint32_t kNet03LocalDegree = 14u;
constexpr std::uint32_t kNet03MediumDegree = 8u;
constexpr std::uint32_t kNet03DiscourseDegree = 4u;
constexpr std::uint32_t kNet03Population = 160u;

// local_fast_sequence_state: dense short-range fusion, same mechanism NET00's
// strong_local_recurrence used. Thins to nothing by discourse -- a slow,
// broad window has no dense local neighbourhood to fuse.
constexpr std::uint32_t kNet03LocalDenseWidth = 64u;
constexpr std::uint32_t kNet03MediumDenseWidth = 16u;

constexpr std::uint32_t kNet03LongTracts = 12u;
constexpr std::uint32_t kNet03PortReserve = 24u;

constexpr std::uint32_t kNet03AffinityRadius = 1u << 20;
constexpr std::int32_t kNet03AffinityStrengthQ16 = 1 << 16;

// broad_delay_distribution: the medium->discourse hop costs strictly more
// than local->medium's -- a genuine three-regime gradient (local's own dense
// recurrence, then two distinct authored transport costs), broader than
// NET01's binary fast(0)/medium(20) distribution.
constexpr std::uint32_t kNet03LocalToMediumCost = 8u;
constexpr std::uint32_t kNet03MediumToDiscourseCost = 32u;
constexpr std::uint32_t kNet03MediumToDiscourseMaturationTick = 160u;

DirectRuleSpecV1 net03_rule(DirectRuleOpcodeV1 opcode, std::uint32_t chemistry) {
  DirectRuleSpecV1 out{};
  out.opcode = opcode;
  out.begin_tick = 0u;
  out.end_tick = kNet03DevelopmentEndTick;
  out.require_mask = 0xffffffffu;
  out.require_value = chemistry;
  out.write_mask = 0xffffffffu;
  out.write_value = chemistry;
  return out;
}

DirectTerritorySpecV1 net03_territory(std::uint32_t ordinal, std::uint32_t chemistry,
                                      std::uint32_t reach) {
  DirectTerritorySpecV1 out{};
  out.identity.lineage = kNet03Lineage;
  out.identity.axis = 0u;
  out.identity.ordinal = ordinal;
  out.chemotype = chemistry;
  out.reach = reach;
  out.begin_tick = 0u;
  return out;
}

}  // namespace

DirectGenomeV1 seed_atlas_net03_temporal_boundary_ecology() {
  DirectGenomeV1 genome{};
  genome.header.life_function_version = 2u;
  genome.header.development_end_tick = kNet03DevelopmentEndTick;
  genome.header.matter_budget = 1u << 24;
  genome.header.development_seed = 0x4e455430u + 3u;  // "NET0" + 3

  genome.territories[genome.header.territory_count++] =
      net03_territory(0u, kNet03LocalChemistry, kNet03LocalReach);
  genome.territories[genome.header.territory_count++] =
      net03_territory(1u, kNet03MediumChemistry, kNet03MediumReach);
  genome.territories[genome.header.territory_count++] =
      net03_territory(2u, kNet03DiscourseChemistry, kNet03DiscourseReach);

  // The gradient chain: local's long_tract seeks medium's chemistry, medium's
  // seeks discourse's. Each field is anchored on its OWNER territory and
  // targets the next link -- 007ac93107's target-chemistry mechanism, applied
  // twice to make a chain rather than a star.
  DirectFieldSpecV1 local_to_medium{};
  local_to_medium.territory.lineage = kNet03Lineage;
  local_to_medium.territory.axis = 0u;
  local_to_medium.territory.ordinal = 0u;  // local
  local_to_medium.radius = kNet03AffinityRadius;
  local_to_medium.require_mask = 0xffffffffu;
  local_to_medium.require_value = kNet03MediumChemistry;
  local_to_medium.write_mask = 0xffffffffu;
  local_to_medium.write_value = static_cast<std::uint32_t>(kNet03AffinityStrengthQ16);
  local_to_medium.begin_tick = 0u;
  local_to_medium.end_tick = kNet03DevelopmentEndTick;
  local_to_medium.polarity = 0u;  // attract
  const std::uint32_t local_to_medium_index = genome.header.field_count++;
  genome.fields[local_to_medium_index] = local_to_medium;

  DirectFieldSpecV1 medium_to_discourse = local_to_medium;
  medium_to_discourse.territory.ordinal = 1u;  // medium
  medium_to_discourse.require_value = kNet03DiscourseChemistry;
  const std::uint32_t medium_to_discourse_index = genome.header.field_count++;
  genome.fields[medium_to_discourse_index] = medium_to_discourse;

  const struct {
    std::uint32_t chemistry;
    std::uint32_t reach;
    std::uint32_t degree;
    std::uint32_t dense_width;  // 0 = author no fuse rule
  } territories[] = {
      {kNet03LocalChemistry, kNet03LocalReach, kNet03LocalDegree, kNet03LocalDenseWidth},
      {kNet03MediumChemistry, kNet03MediumReach, kNet03MediumDegree, kNet03MediumDenseWidth},
      {kNet03DiscourseChemistry, kNet03DiscourseReach, kNet03DiscourseDegree, 0u},
  };
  for (const auto& territory : territories) {
    DirectRuleSpecV1 grow = net03_rule(DirectRuleOpcodeV1::extend, territory.chemistry);
    grow.extent = territory.reach;
    grow.child_slot = territory.degree;
    grow.branch_count = kNet03Population;
    genome.rules[genome.header.rule_count++] = grow;

    if (territory.dense_width != 0u) {
      DirectRuleSpecV1 dense = net03_rule(DirectRuleOpcodeV1::fuse, territory.chemistry);
      dense.branch_count = territory.dense_width;
      genome.rules[genome.header.rule_count++] = dense;
    }

    DirectRuleSpecV1 tract = net03_rule(DirectRuleOpcodeV1::long_tract, territory.chemistry);
    tract.branch_count = kNet03LongTracts;
    if (territory.chemistry == kNet03LocalChemistry) {
      tract.field_index = local_to_medium_index;
      tract.extent = kNet03LocalToMediumCost;
    } else if (territory.chemistry == kNet03MediumChemistry) {
      tract.field_index = medium_to_discourse_index;
      tract.extent = kNet03MediumToDiscourseCost;
      tract.minimum_age = kNet03MediumToDiscourseMaturationTick;
    }
    genome.rules[genome.header.rule_count++] = tract;

    DirectRuleSpecV1 ports = net03_rule(DirectRuleOpcodeV1::endogenous_source, territory.chemistry);
    ports.extent = kNet03PortReserve;
    genome.rules[genome.header.rule_count++] = ports;
  }
  return genome;
}

}  // namespace substrate::direct_network
