// Patch 0002 of the FULL CUDA network-recipe patch program: canonical
// fixed-width Genome Gamma ABI.
//
// A Genome is a compact, content-addressed developmental *recipe* -- it may
// be highly directive about generic silicon construction (lineage creation,
// chemotype transitions, field placement, route extension, branching,
// fusion, retraction, maturation, canalization, finite construction
// economy) but it may never contain a mature connectome, named cognitive
// regions, words/tokens/concepts/answers, a language parser, a host runtime
// controller, host-selected routing, copied somatic tissue, or a semantic
// fitness value. What is mined is Gamma/DeltaGamma; what is grown is the
// actual adult substrate (patch program section 1).
//
// Every field here is a fixed-width POD so the whole structure can be
// device-resident (no dynamic device allocation, no host pointer graph) and
// content-addressed by a canonical byte-exact hash. This header is plain
// host C++ (no CUDA attributes) so CPU tooling -- the recipe miner's host
// bootstrap, the constitutional dependency contracts, a future recipe
// compiler -- can include it directly; direct_network_recipe_abi.cuh adds
// __host__ __device__ qualified operations over the same types for use
// inside CUDA construction kernels (patches 0004+).

#ifndef HARDWARE_NATIVE_DIRECT_NETWORK_RECIPE_HPP
#define HARDWARE_NATIVE_DIRECT_NETWORK_RECIPE_HPP

#include <cstdint>
#include <type_traits>

// This header stays plain host C++ (no CUDA includes) so CPU-only tooling
// can include it without nvcc. Root256's comparison operators are still
// called from __host__ __device__ functions in direct_network_recipe_abi.cuh
// (patch 0002's device-side validator), so they need the __host__ __device__
// qualifiers when compiled by nvcc, and no qualifiers at all otherwise.
#if defined(__CUDACC__)
#define DIRECT_NETWORK_RECIPE_HD __host__ __device__
#else
#define DIRECT_NETWORK_RECIPE_HD
#endif

namespace substrate::direct_network::recipe {

using SiteWord = std::uint32_t;

// Fixed capacities. These bound a Genome to a size that is cheap to keep
// device-resident and content-address; they are recipe-authoring limits,
// not a claim about how large the *grown* morphology may become (that is
// bounded separately, by the paid network-matter budget in patch 0003).
inline constexpr std::uint32_t kMaxSeeds = 64;
inline constexpr std::uint32_t kMaxFields = 256;
inline constexpr std::uint32_t kMaxRules = 1024;

struct Root256 {
  std::uint32_t word[8];

  friend DIRECT_NETWORK_RECIPE_HD bool operator==(const Root256& a, const Root256& b) {
    for (int i = 0; i < 8; ++i) {
      if (a.word[i] != b.word[i]) return false;
    }
    return true;
  }
  friend DIRECT_NETWORK_RECIPE_HD bool operator!=(const Root256& a, const Root256& b) {
    return !(a == b);
  }
};
static_assert(sizeof(Root256) == 32, "Root256 must be exactly 32 bytes");
static_assert(std::is_standard_layout_v<Root256> && std::is_trivial_v<Root256>,
              "Root256 must be a fixed-width POD for device residency and content addressing");

struct GenomeHeader {
  std::uint32_t abi_version;
  std::uint32_t life_function_version;
  std::uint32_t seed_count;
  std::uint32_t field_count;
  std::uint32_t rule_count;
  std::uint32_t development_end_tick;
  std::uint32_t matter_budget;
  std::uint32_t flags;
  std::uint64_t development_seed;
  Root256 parent_root;
  Root256 genome_root;
  Root256 delta_root;
};
static_assert(std::is_standard_layout_v<GenomeHeader> && std::is_trivial_v<GenomeHeader>,
              "GenomeHeader must be a fixed-width POD");

// The current ABI version this header implements. A Genome whose
// header.abi_version does not match this constant must be rejected, not
// reinterpreted -- an ABI change is a new recipe language, not a silent
// reinterpretation of old bytes.
inline constexpr std::uint32_t kCurrentAbiVersion = 1;

enum class RuleOpcode : std::uint32_t {
  extend = 0,
  branch = 1,
  fuse = 2,
  retract = 3,
  mature = 4,
  repair = 5,
  long_tract = 6,
  endogenous_source = 7,
};
inline constexpr std::uint32_t kRuleOpcodeCount = 8;

// ⛔ LEGACY/DONOR (github #1206): `coordinate` is an authored absolute
// address, not a developmental field -- every genome authored against this
// struct today fills it by hand (e.g. `kBaseCoordinate + t *
// kTerritorySpacing`), which is the same address in every world and cannot
// depend on the body, the environment, or what is already grown. Kept
// unchanged here -- same layout, same size, same ABI version -- so the 20+
// existing consumers (construction kernels, certification, every Direct
// Adult test fixture) are unaffected by this slice. New genome-authoring
// code should prefer `developmental_seed_territory.hpp`'s
// `SeedTerritoryRequest` + `derive_seed_territory_origins`, which derives a
// coordinate triple by arbitrating a lineage/axis/extent request against a
// real occupancy field (dose-matched-obstruction verified in
// `developmental_seed_territory_contract`) and can fill this same field. A
// follow-up should migrate every consumer and then delete `coordinate` in
// favor of the territory-request representation.
struct SeedBlock {
  std::uint32_t coordinate[3];
  SiteWord chemistry;
  std::uint32_t lineage;
  std::uint32_t flags;
  std::uint32_t begin_tick;
};
static_assert(std::is_standard_layout_v<SeedBlock> && std::is_trivial_v<SeedBlock>,
              "SeedBlock must be a fixed-width POD");

// ⭐ STATUS 2026-08-18 for the DIRECT lane -- both `SeedBlock.coordinate` and
// `FieldBlock.center` below are now CAUSALLY DEAD there, and only their
// DIFFERENCE is genomic.
//
// `compile_direct_brain` derives each territory's anchor from Γ's
// lineage/reach against the development environment, keeps
// `SeedBlock.coordinate` only as the frame Γ's field centres were written in,
// and evaluates every developmental field at `coord - anchor + declared`, so
// the absolute halves cancel. Measured: translating every authored coordinate
// in Γ by (777001, -531999, 4242424) leaves the grown organism byte-identical
// (`genome_translation_invariant=1`), which is what makes the value dead rather
// than merely unwritten -- an unwritten field is a convention, a dead one
// cannot influence the organism whatever anybody writes into it.
//
// The BCC lane still reads both as real addresses (`bcc32_network_certification.cu`,
// the founder gestation contract, and the JSON genome schema's "center" key),
// so the fields cannot leave this struct until that lane retires.
//
// ⛔ LEGACY/DONOR (github #1206): `center` has the same authored-address
// defect as SeedBlock.coordinate above, and the same migration plan applies.
struct FieldBlock {
  std::uint32_t center[3];
  std::uint32_t radius;
  SiteWord require_mask;
  SiteWord require_value;
  SiteWord write_mask;
  SiteWord write_value;
  std::uint32_t begin_tick;
  std::uint32_t end_tick;
  std::uint32_t polarity;
};
static_assert(std::is_standard_layout_v<FieldBlock> && std::is_trivial_v<FieldBlock>,
              "FieldBlock must be a fixed-width POD");

struct ConstructionRule {
  RuleOpcode opcode;
  std::uint32_t direction_mode;
  std::uint32_t flags;
  std::uint32_t begin_tick;
  std::uint32_t end_tick;
  SiteWord require_mask;
  SiteWord require_value;
  SiteWord write_mask;
  SiteWord write_value;
  std::uint32_t minimum_age;
  std::uint32_t maximum_age;
  std::uint32_t threshold_q32;
  std::uint32_t field;
  std::uint32_t extent;
  std::uint32_t child_slot;
  std::uint32_t branch_count;
};
static_assert(std::is_standard_layout_v<ConstructionRule> && std::is_trivial_v<ConstructionRule>,
              "ConstructionRule must be a fixed-width POD");

struct Genome {
  GenomeHeader header;
  SeedBlock seeds[kMaxSeeds];
  FieldBlock fields[kMaxFields];
  ConstructionRule rules[kMaxRules];
};
static_assert(std::is_standard_layout_v<Genome> && std::is_trivial_v<Genome>,
              "Genome must be a fixed-width POD for device residency");

}  // namespace substrate::direct_network::recipe

#endif  // HARDWARE_NATIVE_DIRECT_NETWORK_RECIPE_HPP
