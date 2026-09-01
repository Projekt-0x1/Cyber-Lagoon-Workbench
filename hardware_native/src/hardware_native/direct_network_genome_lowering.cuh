#ifndef HARDWARE_NATIVE_DIRECT_NETWORK_GENOME_LOWERING_CUH
#define HARDWARE_NATIVE_DIRECT_NETWORK_GENOME_LOWERING_CUH

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "hardware_native/direct_foundry_equivalence_authority.cuh"
#include "hardware_native/direct_foundry_global_recipe_optimization.cuh"
#include "hardware_native/direct_gamma_evidence_ladder.cuh"
#include "hardware_native/direct_network_recipe_abi.cuh"
#include "hardware_native/direct_network_genome.cuh"

namespace substrate::direct_network {

// Direct-owned state carried beside the retained GammaV1 implementation IR.
struct DirectGenomeLoweringV1 {
  GammaV1 gamma{};
  DirectTractDelayLawV1 rule_delay_laws[kDirectMaxRulesV1]{};
  std::uint32_t rule_delay_law_count = 0u;
  Root256 direct_genome_root{};
  Root256 territory_layout_root{};
};

// A verified foundry pool can seed only a fresh Species Gamma.  The high bit
// is a genesis provenance marker, not runtime authority: the synced candidate
// bodies are lowered into ordinary construction rules and the external
// manifest is absent from the born Adult.
inline constexpr std::uint32_t kDirectGenomeFlagVerifiedPoolFrozen = 1u << 31u;

enum class DirectGammaVerifiedCauseProofV1 : std::uint32_t {
  engineering_equivalence = 1u,
  developmental_sibling_assay = 2u,
};

enum class DirectGammaPoolSyncRefuseV1 : std::uint32_t {
  none = 0u,
  malformed = 1u,
  stale = 2u,
  duplicate = 3u,
  incompatible_species = 4u,
  forged_chain = 5u,
  unverified_candidate = 6u,
  forged_evidence = 7u,
  semantic_authority = 8u,
  capacity = 9u,
};

// Observer-side latest-head declaration.  A caller must bind the manifest to
// the current pool/usage/optimizer heads; a stale snapshot cannot silently
// become a new Species Gamma.
struct DirectGammaFoundryLatestHeadsV1 {
  std::uint32_t abi_version = 1u;
  std::uint32_t reserved = 0u;
  std::uint64_t pool_sequence = 0u;
  DirectSha256Address pool_head{};
  DirectSha256Address usage_head{};
  DirectSha256Address optimization_head{};
};

// The only positive authority is availability as a developmental candidate.
// There is deliberately no field for words, propositions, expected outputs,
// Adult identity, Occurrence membership, activation or mature placement.
struct DirectGammaVerifiedPoolManifestHeaderV1 {
  std::uint32_t abi_version = 1u;
  std::uint32_t pool_entry_count = 0u;
  std::uint32_t usage_entry_count = 0u;
  std::uint32_t optimization_entry_count = 0u;
  std::uint32_t cause_count = 0u;
  std::uint32_t developmental_candidate_authority = 1u;
  std::uint32_t semantic_authority = 0u;
  std::uint32_t experiential_authority = 0u;
  std::uint32_t participation_authority = 0u;
  std::uint32_t current_adult_authority = 0u;
  std::uint32_t activation_authority = 0u;
  std::uint32_t mature_topology_authority = 0u;
  std::uint64_t pool_sequence = 0u;
  DirectSha256Address base_genome{};
  DirectSha256Address compatible_species{};
  DirectSha256Address pool_head{};
  DirectSha256Address usage_head{};
  DirectSha256Address optimization_head{};
  DirectSha256Address provenance{};
};

// One selected candidate contributes one generic construction rule.  A rule
// can bias counts, branching, fields and recurrent motifs, but cannot name a
// later node, route, wire table or active Network.  Life Function still binds
// it to concrete local tissue from Gamma, Xi, body, environment and resources.
struct DirectGammaVerifiedRecipeCauseV1 {
  std::uint32_t abi_version = 1u;
  std::uint32_t pool_index = 0u;
  std::uint32_t usage_index = 0u;
  std::uint32_t optimization_index = 0u;
  std::uint32_t fallback_pool_index = 0u;
  DirectGammaVerifiedCauseProofV1 proof =
      DirectGammaVerifiedCauseProofV1::engineering_equivalence;
  std::uint32_t semantic_authority = 0u;
  std::uint32_t experiential_authority = 0u;
  std::uint32_t participation_authority = 0u;
  std::uint32_t current_adult_authority = 0u;
  std::uint32_t activation_authority = 0u;
  std::uint32_t mature_topology_authority = 0u;
  DirectRuleSpecV1 developmental_rule{};
  DirectFoundryEquivalenceReceiptV1 equivalence{};
  DirectFoundryResourceReceiptV1 resources{};
  gamma_evidence::GammaFoundryNeonatalSiblingAssay sibling_assay{};
};

static_assert(std::is_standard_layout_v<DirectGammaFoundryLatestHeadsV1> &&
              std::is_trivially_copyable_v<DirectGammaFoundryLatestHeadsV1>);
static_assert(std::is_standard_layout_v<DirectGammaVerifiedPoolManifestHeaderV1> &&
              std::is_trivially_copyable_v<DirectGammaVerifiedPoolManifestHeaderV1>);
static_assert(std::is_standard_layout_v<DirectGammaVerifiedRecipeCauseV1> &&
              std::is_trivially_copyable_v<DirectGammaVerifiedRecipeCauseV1>);

DirectSha256Address direct_gamma_verified_recipe_cause_body_address(
    const DirectGammaVerifiedRecipeCauseV1& cause,
    DirectFoundryCandidateKindV1 kind);

namespace detail {
bool apply_direct_gamma_verified_recipe_pool_sync_impl(
    DirectGenomeV1* genome,
    const DirectGammaFoundryLatestHeadsV1& latest,
    const DirectGammaVerifiedPoolManifestHeaderV1& manifest,
    const DirectFoundryCandidateEntryV1* pool_entries,
    std::size_t pool_count,
    const DirectFoundryPopulationUsageEntryV1* usage_entries,
    std::size_t usage_count,
    const DirectFoundryOptimizationEntryV1* optimization_entries,
    std::size_t optimization_count,
    const DirectGammaVerifiedRecipeCauseV1* causes,
    std::size_t cause_count,
    DirectGammaPoolSyncRefuseV1* refuse);
}  // namespace detail

// Capacities are properties of one finite manifest proof, not a ceiling on
// foundry generations.  A later foundry snapshot may instantiate this verifier
// with larger archive capacities; each concrete Gamma remains matter-bounded.
template <std::size_t PoolCapacity, std::size_t UsageCapacity,
          std::size_t OptimizationCapacity>
bool apply_direct_gamma_verified_recipe_pool_sync(
    DirectGenomeV1* genome,
    const DirectGammaFoundryLatestHeadsV1& latest,
    const DirectGammaVerifiedPoolManifestHeaderV1& manifest,
    const DirectFoundryCandidateEntryV1* pool_entries,
    std::size_t pool_count,
    const DirectFoundryPopulationUsageEntryV1* usage_entries,
    std::size_t usage_count,
    const DirectFoundryOptimizationEntryV1* optimization_entries,
    std::size_t optimization_count,
    const DirectGammaVerifiedRecipeCauseV1* causes,
    std::size_t cause_count,
    DirectGammaPoolSyncRefuseV1* refuse = nullptr) {
  static_assert(PoolCapacity > 0u && UsageCapacity > 0u &&
                OptimizationCapacity > 0u);
  const bool chains_valid =
      pool_count <= PoolCapacity && usage_count <= UsageCapacity &&
      optimization_count <= OptimizationCapacity &&
      DirectFoundryRecipeCandidatePool<PoolCapacity>::verify_entries(
          pool_entries, pool_count, manifest.pool_head) &&
      DirectFoundryPopulationUsageArchive<UsageCapacity>::verify_entries(
          usage_entries, usage_count, manifest.usage_head) &&
      DirectFoundryGlobalRecipeOptimization<OptimizationCapacity>::verify_entries(
          optimization_entries, optimization_count,
          manifest.optimization_head);
  if (!chains_valid) {
    if (refuse != nullptr)
      *refuse = DirectGammaPoolSyncRefuseV1::forged_chain;
    return false;
  }
  return detail::apply_direct_gamma_verified_recipe_pool_sync_impl(
      genome, latest, manifest, pool_entries, pool_count, usage_entries,
      usage_count, optimization_entries, optimization_count, causes,
      cause_count, refuse);
}

DirectGenomeLoweringV1 lower_direct_genome_v1(
    const DirectGenomeV1& genome, const DirectBodyManifestV1& body,
    const DirectDevelopmentEnvironmentV1& environment);

}  // namespace substrate::direct_network

#endif  // HARDWARE_NATIVE_DIRECT_NETWORK_GENOME_LOWERING_CUH
