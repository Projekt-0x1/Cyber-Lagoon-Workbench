#include <stdexcept>

#include "hardware_native/direct_seed_atlas.cuh"

namespace substrate::direct_network {
namespace {

// ONE ROW PER ATLAS FAMILY. Everything here is a generic developmental scalar
// the Life Function already consumes; nothing names a network, a modality, an
// endpoint or a symbol. `provenance` exists so a reader can find the atlas
// section a row came from -- it is never compiled into the genome.
struct AtlasFamilySeed {
  const char* provenance;
  std::uint32_t territories;   // how many territories this family contributes
  std::uint32_t reach;         // extent -> plan.radius: local vs broad growth
  std::uint32_t degree;        // child_slot -> sparse_degree: fanout
  std::uint32_t population;    // branch_count on extend
  std::uint32_t dense_width;   // 0 = author no fuse rule for this family
  std::uint32_t long_tracts;   // 0 = author no long_tract rule for this family
  std::uint32_t begin_tick;    // developmental order prior (atlas section 7)
  // Disposition, magnitude presence, and magnitude are independent atlas
  // dimensions. This lets a family declare competition with an explicit zero
  // strength instead of overloading zero as "no competition".
  bool competition_enabled;
  std::uint32_t competition;   // outbound strength in Q16, [0, 65535]
  bool competition_magnitude_authored;
  // gh #1268: WHICH OTHER FAMILY THIS ONE'S CORRIDORS REACH, as a row index into
  // this very table (the table's order follows the developmental prior, not the
  // NET numbering, so a NET number would be the wrong key).
  std::uint32_t partner;
  // gh #1359: HOW MANY, on the same terms as how strong. 0 = this family says
  // nothing about proportion; non-zero N = "one node in N competes", the share
  // of this family's tissue that carries the disposition. Delivered through
  // `minimum_age` on the rule that grows the territory, whose only other
  // consumer anywhere in the tree is the `long_tract` corridor window -- so the
  // Life Function gates the read on the opcode as well as the flag, and the
  // corridor rule copied from `grow` below has it cleared for the same reason
  // `extent` is cleared there.
  std::uint32_t competition_share;
  // gh #1359: Host-side presence flag allowing explicit authored 0% density
  // (competition_share=0, competition_density_authored=true) directly from atlas table.
  bool competition_density_authored;
};

constexpr std::uint32_t kCompetitionMagnitudeMaxQ16 = 65535u;

// Row indices. Only the rows that deviate from the hub default need one.
constexpr std::uint32_t kRowNet08 = 8u;
constexpr std::uint32_t kRowNet11 = 12u;
constexpr std::uint32_t kRowNet13 = 14u;
constexpr std::uint32_t kRowNet17 = 17u;
constexpr std::uint32_t kRowNet09 = 9u;

// Chemistry is the only address this ABI has -- there is no coordinate column
// (see the note on the table). A corridor field therefore carries a generous
// radius and lets the required chemistry do the addressing, exactly as every
// per-family seed genome in this file already does.
constexpr std::uint32_t kSpeciesCorridorRadius = 1u << 20;
constexpr std::int32_t kSpeciesCorridorStrengthQ16 = 1 << 16;

constexpr std::uint32_t kSpeciesDevelopmentEnd = 8192u;
constexpr std::uint32_t kSpeciesPreparedLearningEnd = kSpeciesDevelopmentEnd + 256u;

// The atlas's own developmental order prior: body-adjacent and surface ecologies
// first, integrative and control ecologies later. These are windows, not stages
// -- every rule stays open to the end tick.
constexpr std::uint32_t kPhaseSurface = 0u;
constexpr std::uint32_t kPhaseSequence = 256u;
constexpr std::uint32_t kPhaseAssociation = 768u;
constexpr std::uint32_t kPhaseControl = 1536u;

// Families the founder named for the first executable species, plus declared
// initial support from the remaining ones so their absence cannot invalidate
// the organism. `competition_enabled`, `competition`, and
// `competition_magnitude_authored` are deliberately separate fields.
constexpr AtlasFamilySeed kSpecies[] = {
  {"NET00", 3u, 12u, 12u, 256u, 64u, 8u, kPhaseSurface, false, 0u, false, kRowNet08, 0u, false},
  {"NET01", 3u, 16u, 10u, 192u, 32u, 12u, kPhaseSequence, false, 0u, false, kRowNet08, 0u, false},
  {"NET02", 3u, 24u, 8u, 192u, 32u, 12u, kPhaseSequence, false, 0u, false, kRowNet08, 0u, false},
  {"NET03", 2u, 20u, 6u, 160u, 0u, 16u, kPhaseSequence, false, 0u, false, kRowNet08, 0u, false},
  {"NET04", 2u, 10u, 16u, 128u, 0u, 10u, kPhaseSequence, false, 0u, false, kRowNet08, 0u, false},
  {"NET05", 3u, 40u, 6u, 224u, 48u, 16u, kPhaseAssociation, false, 0u, false, kRowNet08, 0u, false},
  {"NET06", 2u, 32u, 14u, 128u, 0u, 14u, kPhaseControl, false, 0u, false, kRowNet08, 0u, false},
  {"NET07", 2u, 24u, 6u, 96u, 0u, 8u, kPhaseControl, false, 0u, false, kRowNet08, 0u, false},
  {"NET08", 3u, 16u, 20u, 64u, 0u, 24u, kPhaseSequence, false, 0u, false, kRowNet13, 0u, false},
  // NET09 is the one atlas family whose gamma_seed names competitive loops.
  {"NET09", 3u, 14u, 10u, 96u, 0u, 10u, kPhaseControl, true, 32768u, true, kRowNet08, 5u, true},
  {"NET10", 4u, 10u, 8u, 64u, 16u, 8u, kPhaseAssociation, false, 0u, false, kRowNet08, 0u, false},
  {"NET15", 3u, 14u, 10u, 192u, 32u, 10u, kPhaseSurface, false, 0u, false, kRowNet08, 0u, false},
  {"NET11", 1u, 28u, 12u, 64u, 0u, 12u, kPhaseAssociation, false, 0u, false, kRowNet17, 0u, false},
  {"NET12", 2u, 48u, 4u, 48u, 0u, 16u, kPhaseAssociation, false, 0u, false, kRowNet08, 0u, false},
  {"NET13", 2u, 56u, 8u, 64u, 0u, 32u, kPhaseAssociation, false, 0u, false, kRowNet08, 0u, false},
  {"NET14", 2u, 32u, 8u, 96u, 0u, 20u, kPhaseSequence, false, 0u, false, kRowNet08, 0u, false},
  {"NET16", 1u, 18u, 8u, 64u, 0u, 8u, kPhaseAssociation, false, 0u, false, kRowNet17, 0u, false},
  {"NET17", 1u, 12u, 6u, 48u, 0u, 6u, kPhaseSurface, false, 0u, false, kRowNet11, 0u, false},
};
constexpr std::uint32_t kSpeciesFamilyCount =
    static_cast<std::uint32_t>(sizeof(kSpecies) / sizeof(kSpecies[0]));

constexpr bool same_row_name(const char* a, const char* b) {
  while (*a != '\0' && *a == *b) { ++a; ++b; }
  return *a == *b;
}
static_assert(same_row_name(kSpecies[kRowNet08].provenance, "NET08"));
static_assert(same_row_name(kSpecies[kRowNet11].provenance, "NET11"));
static_assert(same_row_name(kSpecies[kRowNet13].provenance, "NET13"));
static_assert(same_row_name(kSpecies[kRowNet17].provenance, "NET17"));

void validate_competition_authoring(const AtlasFamilySeed& seed) {
  if (!seed.competition_enabled &&
      (seed.competition != 0u || seed.competition_magnitude_authored)) {
    throw std::invalid_argument("competition magnitude requires an enabled disposition");
  }
  if (seed.competition > kCompetitionMagnitudeMaxQ16) {
    throw std::invalid_argument("competition magnitude exceeds Q16 authoring range");
  }
}

DirectGenomeV1 compile_canonical_species(
    const DirectAtlasCompetitionSpecV1* competition_override) {
  DirectGenomeV1 genome{};
  genome.header.abi_version = kDirectGenomeAbiCurrent;
  genome.header.life_function_version = 2u;
  genome.header.development_end_tick = kSpeciesDevelopmentEnd;
  genome.header.matter_budget = 1u << 30;
  genome.header.development_seed = 0x53504543u;  // "SPEC"

  std::uint32_t family_base_chemistry[kSpeciesFamilyCount] = {};
  {
    std::uint32_t running = 0x40u;
    for (std::uint32_t family = 0; family < kSpeciesFamilyCount; ++family) {
      family_base_chemistry[family] = running;
      running += kSpecies[family].territories;
    }
  }

  std::uint32_t chemistry = 0x40u;
  for (std::uint32_t family = 0; family < kSpeciesFamilyCount; ++family) {
    AtlasFamilySeed seed = kSpecies[family];
    if (competition_override != nullptr && family == kRowNet09) {
      seed.competition_enabled = competition_override->competition_enabled;
      seed.competition = competition_override->competition;
      seed.competition_magnitude_authored =
          competition_override->competition_magnitude_authored;
    }
    validate_competition_authoring(seed);
    for (std::uint32_t local = 0; local < seed.territories; ++local) {
      DirectTerritorySpecV1 territory{};
      territory.identity.lineage = family;
      territory.identity.axis = 0u;
      territory.identity.ordinal = local;
      territory.chemotype = chemistry;
      territory.reach = seed.reach;
      territory.begin_tick = seed.begin_tick;
      genome.territories[genome.header.territory_count++] = territory;

      DirectRuleSpecV1 grow{};
      grow.opcode = DirectRuleOpcodeV1::extend;
      grow.begin_tick = seed.begin_tick;
      grow.end_tick = kSpeciesDevelopmentEnd;
      grow.require_mask = 0xffffffffu;
      grow.require_value = chemistry;
      grow.write_mask = 0xffffffffu;
      grow.write_value = chemistry;
      grow.extent = seed.reach;
      grow.child_slot = seed.degree;
      grow.branch_count = seed.population;
      if (seed.competition_enabled) {
        grow.flags |= kRuleFlagInhibitoryBias;
        const bool magnitude_authored =
            seed.competition_magnitude_authored || seed.competition != 0u;
        if (magnitude_authored) {
          grow.flags |= kRuleFlagCompetitionMagnitudeAuthored;
          grow.threshold_q32 = seed.competition << 16;
        }
        grow.minimum_age = seed.competition_share;
        if (seed.competition_density_authored || seed.competition_share != 0u) {
          grow.flags |= kRuleFlagCompetitionDensityAuthored;
        }
      }
      genome.rules[genome.header.rule_count++] = grow;

      if (seed.dense_width != 0u) {
        DirectRuleSpecV1 dense = grow;
        dense.opcode = DirectRuleOpcodeV1::fuse;
        dense.branch_count = seed.dense_width;
        // Magnitude is declared once by the territory's extend rule. The
        // derived fuse rule must not create a second opcode-specific carrier.
        dense.flags &= ~kRuleFlagCompetitionMagnitudeAuthored;
        dense.threshold_q32 = 0u;
        genome.rules[genome.header.rule_count++] = dense;
      }
      if (seed.long_tracts != 0u) {
        const std::uint32_t recurrent_count =
            seed.territories > 1u ? (seed.long_tracts + 1u) / 2u : 0u;
        const std::uint32_t cross_count = seed.long_tracts - recurrent_count;
        const std::uint32_t partner_local[2] = {(local + 1u) % seed.territories,
                                                local % kSpecies[seed.partner].territories};
        const std::uint32_t partner_family[2] = {family, seed.partner};
        const std::uint32_t counts[2] = {recurrent_count, cross_count};
        for (std::uint32_t which = 0; which < 2u; ++which) {
          if (counts[which] == 0u) continue;
          const std::uint32_t corridor_field = genome.header.field_count++;
          DirectFieldSpecV1 corridor{};
          corridor.territory = territory.identity;
          corridor.radius = kSpeciesCorridorRadius;
          corridor.require_mask = 0xffffffffu;
          corridor.require_value =
              family_base_chemistry[partner_family[which]] + partner_local[which];
          corridor.write_mask = 0xffffffffu;
          corridor.write_value = static_cast<std::uint32_t>(kSpeciesCorridorStrengthQ16);
          corridor.begin_tick = seed.begin_tick;
          corridor.end_tick = kSpeciesDevelopmentEnd;
          corridor.polarity = 0u;
          genome.fields[corridor_field] = corridor;

          DirectRuleSpecV1 tract = grow;
          tract.opcode = DirectRuleOpcodeV1::long_tract;
          tract.branch_count = counts[which];
          tract.field_index = corridor_field;
          tract.extent = 0u;
          tract.minimum_age = 0u;
          // The territory's extend rule is the only magnitude declaration.
          tract.flags &= ~kRuleFlagCompetitionMagnitudeAuthored;
          tract.threshold_q32 = 0u;
          genome.rules[genome.header.rule_count++] = tract;
        }
      }
      // gh #1294 rung-10 dose arm: NET08 is the kind-matched constructor
      // family paired with NET13. Author the same resident repair term at
      // both ends of that declared corridor so ARM 13 can vary constructor
      // population without changing the producer or route protocol.
      if (family == kRowNet13 || family == kRowNet08) {
        DirectRuleSpecV1 regrow = grow;
        regrow.opcode = DirectRuleOpcodeV1::repair;
        regrow.flags |= kRuleFlagPostBirthResident;
        regrow.end_tick = 0u;
        genome.rules[genome.header.rule_count++] = regrow;
      }
      ++chemistry;
    }
  }

  const std::uint32_t maturation_field = genome.header.field_count++;
  DirectFieldSpecV1 field{};
  field.territory = genome.territories[0].identity;
  field.radius = kSpeciesCorridorRadius;
  field.write_value = static_cast<std::uint32_t>(kSpeciesCorridorStrengthQ16);
  field.begin_tick = kSpeciesDevelopmentEnd;
  field.end_tick = kSpeciesPreparedLearningEnd + 1u;
  field.polarity = static_cast<std::uint32_t>(DevelopmentFieldKind::maturation);
  genome.fields[maturation_field] = field;

  DirectRuleSpecV1 prepared_learning{};
  prepared_learning.opcode = DirectRuleOpcodeV1::mature;
  prepared_learning.flags = kRuleFlagPostBirthResident;
  prepared_learning.field_index = maturation_field;
  prepared_learning.minimum_age = kSpeciesDevelopmentEnd;
  prepared_learning.maximum_age = kSpeciesPreparedLearningEnd;
  genome.rules[genome.header.rule_count++] = prepared_learning;
  return genome;
}

}  // namespace

std::uint32_t seed_atlas_family_count() { return kSpeciesFamilyCount; }

DirectGenomeV1 seed_atlas_canonical_species() {
  return compile_canonical_species(nullptr);
}

DirectGenomeV1 seed_atlas_canonical_species(DirectAtlasCompetitionSpecV1 competition) {
  return compile_canonical_species(&competition);
}

}  // namespace substrate::direct_network
