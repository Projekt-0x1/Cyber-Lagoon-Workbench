#ifndef HARDWARE_NATIVE_DIRECT_NETWORK_GENOME_CUH
#define HARDWARE_NATIVE_DIRECT_NETWORK_GENOME_CUH

#include <cstdint>
#include <type_traits>

#include "hardware_native/direct_network_brain.cuh"

namespace substrate::direct_network {

inline constexpr std::uint32_t kDirectGenomeAbiV1 = 1u;
inline constexpr std::uint32_t kDirectGenomeAbiV2 = 2u;
inline constexpr std::uint32_t kDirectGenomeAbiCurrent = kDirectGenomeAbiV2;
inline constexpr std::uint32_t kDirectMaxTerritoriesV1 = 64u;
inline constexpr std::uint32_t kDirectMaxFieldsV1 = 256u;
inline constexpr std::uint32_t kDirectMaxRulesV1 = 1024u;

struct DirectTerritoryIdentityV1 {
  std::uint32_t lineage = 0u;
  std::uint32_t axis = 0u;
  std::uint32_t ordinal = 0u;
};
static_assert(std::is_standard_layout_v<DirectTerritoryIdentityV1> && std::is_trivially_copyable_v<DirectTerritoryIdentityV1>);
static_assert(std::has_unique_object_representations_v<DirectTerritoryIdentityV1>);

struct DirectTerritorySpecV1 {
  DirectTerritoryIdentityV1 identity{};
  std::uint32_t reach = 0u;
  std::uint32_t chemotype = 0u;
  std::uint32_t begin_tick = 0u;
  std::uint32_t flags = 0u;
};
static_assert(std::is_standard_layout_v<DirectTerritorySpecV1> && std::is_trivially_copyable_v<DirectTerritorySpecV1>);
static_assert(std::has_unique_object_representations_v<DirectTerritorySpecV1>);

// A field is expressed against the territory that owns it.  `relative_center`
// is an offset from that territory's origin, which the Direct Life Function
// derives against the body/development world at planning time.
struct DirectFieldSpecV1 {
  DirectTerritoryIdentityV1 territory{};
  std::int32_t relative_center[3] = {0, 0, 0};
  std::uint32_t radius = 0u;
  std::uint32_t require_mask = 0u;
  std::uint32_t require_value = 0u;
  std::uint32_t write_mask = 0u;
  std::uint32_t write_value = 0u;
  std::uint32_t begin_tick = 0u;
  std::uint32_t end_tick = 0u;
  // Packed: `polarity % kDevelopmentFieldKindCount` selects the
  // DevelopmentFieldKind (unchanged). `(polarity / kDevelopmentFieldKindCount)
  // % kFieldDecayClassCount` selects independent_decay_timescale (#1276/NET12)
  // -- a per-field decay class, orthogonal to kind, packed into the same
  // legacy word rather than widening `FieldBlock` (BCC-shared POD, see
  // direct_network_recipe.hpp). Class 0 reproduces the prior no-decay
  // behavior exactly, so every existing NET00-NET02 field (polarity always
  // authored < kDevelopmentFieldKindCount) is unaffected.
  std::uint32_t polarity = 0u;
};
static_assert(std::is_standard_layout_v<DirectFieldSpecV1> && std::is_trivially_copyable_v<DirectFieldSpecV1>);
static_assert(std::has_unique_object_representations_v<DirectFieldSpecV1>);

enum class DirectRuleOpcodeV1 : std::uint32_t {
  extend = 0u,
  branch = 1u,
  fuse = 2u,
  retract = 3u,
  mature = 4u,
  repair = 5u,
  long_tract = 6u,
  endogenous_source = 7u,
};
inline constexpr std::uint32_t kDirectRuleOpcodeCountV1 = 8u;

struct DirectTractDelayLawV1 {
  std::uint32_t initial_min_ticks = 0u, initial_max_ticks = 0u;
  std::uint32_t mature_min_ticks = 0u, mature_max_ticks = 0u;
  std::uint32_t maturation_use_threshold = 0u;
};
static_assert(sizeof(DirectTractDelayLawV1) == 5u * sizeof(std::uint32_t));
static_assert(std::is_standard_layout_v<DirectTractDelayLawV1> && std::is_trivially_copyable_v<DirectTractDelayLawV1> && std::has_unique_object_representations_v<DirectTractDelayLawV1>);

struct DirectRuleSpecV1 {
  DirectRuleOpcodeV1 opcode = DirectRuleOpcodeV1::extend;
  std::uint32_t direction_mode = 0u;
  std::uint32_t flags = 0u;
  std::uint32_t begin_tick = 0u;
  std::uint32_t end_tick = 0u;
  std::uint32_t require_mask = 0u;
  std::uint32_t require_value = 0u;
  std::uint32_t write_mask = 0u;
  std::uint32_t write_value = 0u;
  std::uint32_t minimum_age = 0u;
  std::uint32_t maximum_age = 0u;
  std::uint32_t threshold_q32 = 0u;
  std::uint32_t field_index = kInvalidIndex;
  std::uint32_t extent = 0u;
  std::uint32_t child_slot = 0u;
  std::uint32_t branch_count = 0u;
  DirectTractDelayLawV1 tract_delay{};
};
static_assert(offsetof(DirectRuleSpecV1, tract_delay) == 16u * sizeof(std::uint32_t));
static_assert(std::is_standard_layout_v<DirectRuleSpecV1> && std::is_trivially_copyable_v<DirectRuleSpecV1> && std::has_unique_object_representations_v<DirectRuleSpecV1>);

struct DirectGenomeHeaderV1 {
  std::uint32_t abi_version = kDirectGenomeAbiCurrent;
  std::uint32_t life_function_version = 0u;
  std::uint32_t territory_count = 0u;
  std::uint32_t field_count = 0u;
  std::uint32_t rule_count = 0u;
  std::uint32_t development_end_tick = 0u;
  std::uint32_t matter_budget = 0u;
  std::uint32_t flags = 0u;
  std::uint64_t development_seed = 0u;
  Root256 parent_root{};
  Root256 delta_root{};
};
static_assert(std::is_standard_layout_v<DirectGenomeHeaderV1> && std::is_trivially_copyable_v<DirectGenomeHeaderV1> && std::has_unique_object_representations_v<DirectGenomeHeaderV1>);

struct DirectGenomeV1 {
  DirectGenomeHeaderV1 header{};
  DirectTerritorySpecV1 territories[kDirectMaxTerritoriesV1]{};
  DirectFieldSpecV1 fields[kDirectMaxFieldsV1]{};
  DirectRuleSpecV1 rules[kDirectMaxRulesV1]{};
};
static_assert(std::is_standard_layout_v<DirectGenomeV1> &&
              std::is_trivially_copyable_v<DirectGenomeV1>);

// Root over exactly the Direct-owned authored bytes in use.
Root256 canonical_direct_genome_root_v1(const DirectGenomeV1& genome);
bool apply_observer_prose_bytes_to_direct_genome(DirectGenomeV1*, const void*, std::uint64_t);

}  // namespace substrate::direct_network

#endif  // HARDWARE_NATIVE_DIRECT_NETWORK_GENOME_CUH
