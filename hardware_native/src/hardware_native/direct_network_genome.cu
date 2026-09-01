#include "hardware_native/direct_gamma_evidence_ladder.cuh"
#include "hardware_native/direct_network_genome_lowering.cuh"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

#include "hardware_native/direct_seed_territory.hpp"

namespace substrate::direct_network {
namespace {

using substrate::direct_network::recipe::canonical_genome_root;
using substrate::direct_network::recipe::kCurrentAbiVersion;
using substrate::direct_network::recipe::kMaxFields;
using substrate::direct_network::recipe::kMaxRules;
using substrate::direct_network::recipe::kMaxSeeds;
using substrate::direct_network::recipe::validate_genome;
using substrate::direct_network::recipe::GenomeValidationError;

static_assert(kDirectMaxTerritoriesV1 <= kMaxSeeds,
              "the temporary lowered Gamma IR must hold every Direct territory");
static_assert(kDirectMaxFieldsV1 <= kMaxFields,
              "the temporary lowered Gamma IR must hold every Direct field");
static_assert(kDirectMaxRulesV1 <= kMaxRules,
              "the temporary lowered Gamma IR must hold every Direct rule");

[[nodiscard]] constexpr bool same_identity(const DirectTerritoryIdentityV1& left,
                                           const DirectTerritoryIdentityV1& right) {
  return left.lineage == right.lineage && left.axis == right.axis &&
         left.ordinal == right.ordinal;
}

[[nodiscard]] constexpr bool identity_less(const DirectTerritoryIdentityV1& left,
                                            const DirectTerritoryIdentityV1& right) {
  if (left.lineage != right.lineage) return left.lineage < right.lineage;
  if (left.axis != right.axis) return left.axis < right.axis;
  return left.ordinal < right.ordinal;
}

[[nodiscard]] std::uint32_t checked_coordinate(std::int64_t value, const char* what) {
  if (value < 0 || value > static_cast<std::int64_t>(std::numeric_limits<std::uint32_t>::max()))
    throw std::invalid_argument(what);
  return static_cast<std::uint32_t>(value);
}

void absorb_bytes(std::uint64_t (&state)[8], const void* data, std::size_t bytes) {
  const auto* input = static_cast<const unsigned char*>(data);
  static constexpr std::uint64_t kPrime[8] = {
      0x100000001b3ull, 0x9e3779b185ebca87ull, 0xc2b2ae3d27d4eb4full,
      0x165667b19e3779f9ull, 0x85ebca77c2b2ae63ull, 0x27d4eb2f165667c5ull,
      0x94d049bb133111ebull, 0xd6e8feb86659fd93ull};
  for (std::size_t index = 0u; index < bytes; ++index) {
    for (std::uint32_t lane = 0u; lane < 8u; ++lane) {
      state[lane] ^= static_cast<std::uint64_t>(input[index]) +
                     (static_cast<std::uint64_t>(lane) << 8u);
      state[lane] *= kPrime[lane];
      state[lane] ^= state[lane] >> (17u + lane);
    }
  }
}

[[nodiscard]] Root256 finish_root(const std::uint64_t (&state)[8]) {
  Root256 root{};
  for (std::uint32_t lane = 0u; lane < 8u; ++lane) {
    std::uint64_t value = state[lane] ^ (state[lane] >> 33u);
    value *= 0xff51afd7ed558ccdull;
    value ^= value >> 29u;
    root.word[lane] = static_cast<std::uint32_t>(value ^ (value >> 32u));
  }
  return root;
}

[[nodiscard]] Root256 canonical_layout_root(const Root256& genome_root,
                                             const seed_territory::DirectDerivedOrigin* origins,
                                             std::uint32_t count) {
  std::uint64_t state[8] = {0x6a09e667f3bcc909ull, 0xbb67ae8584caa73bull,
                            0x3c6ef372fe94f82bull, 0xa54ff53a5f1d36f1ull,
                            0x510e527fade682d1ull, 0x9b05688c2b3e6c1full,
                            0x1f83d9abfb41bd6bull, 0x5be0cd19137e2179ull};
  absorb_bytes(state, &genome_root, sizeof(genome_root));
  absorb_bytes(state, &count, sizeof(count));
  for (std::uint32_t index = 0u; index < count; ++index) {
    const seed_territory::DirectDerivedOrigin& origin = origins[index];
    absorb_bytes(state, origin.coordinate, sizeof(origin.coordinate));
    absorb_bytes(state, &origin.cell, sizeof(origin.cell));
    absorb_bytes(state, &origin.rounds, sizeof(origin.rounds));
    const std::uint32_t placed = origin.placed ? 1u : 0u;
    absorb_bytes(state, &placed, sizeof(placed));
  }
  return finish_root(state);
}

void validate_direct_genome(const DirectGenomeV1& genome) {
  const DirectGenomeHeaderV1& header = genome.header;
  if (header.abi_version != kDirectGenomeAbiV1 && header.abi_version != kDirectGenomeAbiV2)
    throw std::invalid_argument("DirectGenomeV1 ABI mismatch");
  if (header.territory_count == 0u || header.territory_count > kDirectMaxTerritoriesV1)
    throw std::invalid_argument("DirectGenomeV1 territory count invalid");
  if (header.field_count > kDirectMaxFieldsV1)
    throw std::invalid_argument("DirectGenomeV1 field count invalid");
  if (header.rule_count > kDirectMaxRulesV1)
    throw std::invalid_argument("DirectGenomeV1 rule count invalid");

  std::vector<DirectTerritoryIdentityV1> identities;
  identities.reserve(header.territory_count);
  for (std::uint32_t index = 0u; index < header.territory_count; ++index) {
    if (genome.territories[index].reach == 0u)
      throw std::invalid_argument("DirectGenomeV1 territory reach must be nonzero");
    identities.push_back(genome.territories[index].identity);
  }
  std::sort(identities.begin(), identities.end(), identity_less);
  for (std::size_t index = 1u; index < identities.size(); ++index) {
    if (same_identity(identities[index - 1u], identities[index]))
      throw std::invalid_argument("DirectGenomeV1 duplicate (lineage, axis, ordinal)");
  }

  const auto territory_exists = [&](const DirectTerritoryIdentityV1& identity) {
    for (std::uint32_t index = 0u; index < header.territory_count; ++index)
      if (same_identity(genome.territories[index].identity, identity)) return true;
    return false;
  };
  for (std::uint32_t index = 0u; index < header.field_count; ++index) {
    if (!territory_exists(genome.fields[index].territory))
      throw std::invalid_argument("DirectGenomeV1 field targets no territory identity");
  }
  for (std::uint32_t index = 0u; index < header.rule_count; ++index) {
    const DirectRuleSpecV1& rule = genome.rules[index];
    if (static_cast<std::uint32_t>(rule.opcode) >= kDirectRuleOpcodeCountV1)
      throw std::invalid_argument("DirectGenomeV1 rule opcode invalid");
    if (rule.field_index != kInvalidIndex && rule.field_index >= header.field_count)
      throw std::invalid_argument("DirectGenomeV1 rule field index invalid");
    const DirectTractDelayLawV1& delay = rule.tract_delay;
    const bool zero_delay = delay.initial_min_ticks == 0u && delay.initial_max_ticks == 0u &&
                            delay.mature_min_ticks == 0u && delay.mature_max_ticks == 0u &&
                            delay.maturation_use_threshold == 0u;
    if (header.abi_version == kDirectGenomeAbiV1 && !zero_delay)
      throw std::invalid_argument("DirectGenomeV1 ABI cannot author tract delay laws");
    if (rule.opcode != DirectRuleOpcodeV1::long_tract && !zero_delay)
      throw std::invalid_argument("Direct tract delay law requires long_tract opcode");
    if (!zero_delay) {
      if (delay.initial_min_ticks == 0u || delay.initial_min_ticks > delay.initial_max_ticks ||
          delay.initial_max_ticks > 64u)
        throw std::invalid_argument("Direct initial tract delay distribution invalid");
      const bool mature_pair_absent = delay.mature_min_ticks == 0u && delay.mature_max_ticks == 0u;
      const bool mature_pair_valid = delay.mature_min_ticks != 0u &&
                                     delay.mature_min_ticks <= delay.mature_max_ticks &&
                                     delay.mature_max_ticks <= 64u;
      if (!mature_pair_absent && !mature_pair_valid)
        throw std::invalid_argument("Direct mature tract delay distribution invalid");
      const bool distinct_mature = mature_pair_valid &&
          (delay.mature_min_ticks != delay.initial_min_ticks ||
           delay.mature_max_ticks != delay.initial_max_ticks);
      if ((delay.maturation_use_threshold != 0u) != distinct_mature)
        throw std::invalid_argument("Direct tract delay maturation threshold invalid");
    }
  }
}

[[nodiscard]] const seed_territory::DirectDerivedOrigin& origin_for(
    const DirectGenomeV1& genome, const seed_territory::DirectDerivedOrigin* origins,
    const DirectTerritoryIdentityV1& identity) {
  for (std::uint32_t index = 0u; index < genome.header.territory_count; ++index) {
    if (same_identity(genome.territories[index].identity, identity)) return origins[index];
  }
  throw std::logic_error("validated Direct field lost its territory identity");
}

void set_pool_sync_refuse(DirectGammaPoolSyncRefuseV1* out,
                          DirectGammaPoolSyncRefuseV1 value) {
  if (out != nullptr) *out = value;
}

[[nodiscard]] bool valid_synced_rule(const DirectGenomeV1& genome,
                                     const DirectRuleSpecV1& rule) {
  if (static_cast<std::uint32_t>(rule.opcode) >= kDirectRuleOpcodeCountV1 ||
      (rule.field_index != kInvalidIndex &&
       rule.field_index >= genome.header.field_count))
    return false;
  const DirectTractDelayLawV1& delay = rule.tract_delay;
  const bool zero_delay = delay.initial_min_ticks == 0u &&
                          delay.initial_max_ticks == 0u &&
                          delay.mature_min_ticks == 0u &&
                          delay.mature_max_ticks == 0u &&
                          delay.maturation_use_threshold == 0u;
  if (genome.header.abi_version == kDirectGenomeAbiV1) return zero_delay;
  if (rule.opcode != DirectRuleOpcodeV1::long_tract) return zero_delay;
  if (zero_delay) return true;
  if (delay.initial_min_ticks == 0u ||
      delay.initial_min_ticks > delay.initial_max_ticks ||
      delay.initial_max_ticks > 64u)
    return false;
  const bool mature_absent = delay.mature_min_ticks == 0u &&
                             delay.mature_max_ticks == 0u;
  const bool mature_valid = delay.mature_min_ticks != 0u &&
                            delay.mature_min_ticks <= delay.mature_max_ticks &&
                            delay.mature_max_ticks <= 64u;
  if (!mature_absent && !mature_valid) return false;
  const bool distinct = mature_valid &&
      (delay.mature_min_ticks != delay.initial_min_ticks ||
       delay.mature_max_ticks != delay.initial_max_ticks);
  return (delay.maturation_use_threshold != 0u) == distinct;
}

[[nodiscard]] const DirectFoundryCandidateEntryV1* find_pool_candidate(
    const DirectFoundryCandidateEntryV1* entries, std::size_t count,
    const DirectSha256Address& address) {
  for (std::size_t i = 0u; i < count; ++i)
    if (entries[i].candidate_address == address) return &entries[i];
  return nullptr;
}

[[nodiscard]] bool optimization_sources_match(
    const DirectFoundryCandidateEntryV1& candidate,
    const DirectFoundryCandidateEntryV1& fallback,
    const DirectFoundryPopulationUsageEntryV1& usage,
    const DirectFoundryOptimizationEntryV1& optimization) {
  using UsageAddressing = DirectFoundryPopulationUsageArchive<1u>;
  const auto& observed = usage.receipt;
  const auto& record = optimization.record;
  if (usage.receipt_address != UsageAddressing::receipt_address(observed) ||
      record.candidate != candidate.candidate_address ||
      record.parent_candidate != candidate.record.parent_candidate ||
      record.fallback_candidate != fallback.candidate_address ||
      record.construction_ancestry != candidate.record.construction_ancestry ||
      record.compatible_species != candidate.record.compatible_species ||
      record.usage_receipt != usage.receipt_address ||
      record.guard != candidate.record.guard ||
      record.evaluator != candidate.record.compatible_evaluator)
    return false;
  return observed.candidate == candidate.candidate_address &&
         record.task == observed.task && record.guard == observed.guard &&
         record.body_regime == observed.body_regime &&
         record.evaluator == observed.evaluator &&
         record.resource_regime == observed.resource_regime &&
         record.trial_history == observed.trial_history &&
         record.positive_receipts == observed.positive_receipts &&
         record.negative_receipts == observed.negative_receipts &&
         record.trial_count == observed.trial_count &&
         record.successful_closure_count == observed.successful_closure_count &&
         record.guarded_failure_count == observed.guarded_failure_count &&
         record.exact_causal_participation_count ==
             observed.exact_causal_participation_count &&
         record.redundant_occurrence_count == observed.redundant_occurrence_count &&
         record.cooccurrence_count == observed.cooccurrence_count &&
         record.latency_p95_ns == observed.latency_p95_ns &&
         record.peak_vram_bytes == observed.peak_vram_bytes &&
         record.energy_p95_nj == observed.energy_p95_nj;
}

[[nodiscard]] DirectSha256Address verified_pool_manifest_address(
    const DirectGammaVerifiedPoolManifestHeaderV1& manifest,
    const DirectFoundryCandidateEntryV1* pool_entries,
    std::size_t pool_count,
    const DirectFoundryPopulationUsageEntryV1* usage_entries,
    std::size_t usage_count,
    const DirectFoundryOptimizationEntryV1* optimization_entries,
    std::size_t optimization_count,
    const DirectGammaVerifiedRecipeCauseV1* causes,
    std::size_t cause_count) {
  static constexpr char kDomain[] =
      "0x1-direct-gamma-verified-recipe-pool-manifest-v1";
  detail::DirectSha256State state{};
  state.update(kDomain, sizeof(kDomain) - 1u);
  state.update(&manifest, sizeof(manifest));
  for (std::size_t i = 0u; i < pool_count; ++i)
    state.update(pool_entries[i].candidate_address.byte,
                 sizeof(pool_entries[i].candidate_address.byte));
  for (std::size_t i = 0u; i < usage_count; ++i)
    state.update(usage_entries[i].receipt_address.byte,
                 sizeof(usage_entries[i].receipt_address.byte));
  for (std::size_t i = 0u; i < optimization_count; ++i)
    state.update(optimization_entries[i].nomination_address.byte,
                 sizeof(optimization_entries[i].nomination_address.byte));
  for (std::size_t i = 0u; i < cause_count; ++i) {
    const auto& candidate = pool_entries[causes[i].pool_index];
    const DirectSha256Address body =
        direct_gamma_verified_recipe_cause_body_address(
            causes[i], candidate.record.kind);
    state.update(body.byte, sizeof(body.byte));
    const std::uint32_t proof = static_cast<std::uint32_t>(causes[i].proof);
    state.update(&proof, sizeof(proof));
  }
  return state.finish();
}

}  // namespace

DirectSha256Address direct_gamma_verified_recipe_cause_body_address(
    const DirectGammaVerifiedRecipeCauseV1& cause,
    DirectFoundryCandidateKindV1 kind) {
  static constexpr char kDomain[] =
      "0x1-direct-gamma-verified-recipe-developmental-cause-v1";
  detail::DirectSha256State state{};
  state.update(kDomain, sizeof(kDomain) - 1u);
  state.update(&cause.abi_version, sizeof(cause.abi_version));
  const std::uint32_t kind_word = static_cast<std::uint32_t>(kind);
  state.update(&kind_word, sizeof(kind_word));
  state.update(&cause.developmental_rule, sizeof(cause.developmental_rule));
  return state.finish();
}

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
    DirectGammaPoolSyncRefuseV1* refuse) {
  set_pool_sync_refuse(refuse, DirectGammaPoolSyncRefuseV1::malformed);
  if (genome == nullptr || pool_entries == nullptr || usage_entries == nullptr ||
      optimization_entries == nullptr || causes == nullptr || pool_count == 0u ||
      usage_count == 0u || optimization_count == 0u || cause_count == 0u ||
      manifest.abi_version != 1u || latest.abi_version != 1u ||
      latest.reserved != 0u || manifest.pool_entry_count != pool_count ||
      manifest.usage_entry_count != usage_count ||
      manifest.optimization_entry_count != optimization_count ||
      manifest.cause_count != cause_count || manifest.pool_sequence == 0u ||
      !direct_foundry_address_is_nonzero(manifest.provenance))
    return false;
  if (manifest.developmental_candidate_authority != 1u ||
      manifest.semantic_authority != 0u ||
      manifest.experiential_authority != 0u ||
      manifest.participation_authority != 0u ||
      manifest.current_adult_authority != 0u ||
      manifest.activation_authority != 0u ||
      manifest.mature_topology_authority != 0u) {
    set_pool_sync_refuse(refuse,
                         DirectGammaPoolSyncRefuseV1::semantic_authority);
    return false;
  }
  if (latest.pool_sequence != manifest.pool_sequence ||
      latest.pool_head != manifest.pool_head ||
      latest.usage_head != manifest.usage_head ||
      latest.optimization_head != manifest.optimization_head) {
    set_pool_sync_refuse(refuse, DirectGammaPoolSyncRefuseV1::stale);
    return false;
  }
  if ((genome->header.flags & kDirectGenomeFlagVerifiedPoolFrozen) != 0u) {
    set_pool_sync_refuse(refuse, DirectGammaPoolSyncRefuseV1::duplicate);
    return false;
  }
  if ((genome->header.abi_version != kDirectGenomeAbiV1 &&
       genome->header.abi_version != kDirectGenomeAbiV2) ||
      genome->header.rule_count > kDirectMaxRulesV1 ||
      cause_count > kDirectMaxRulesV1 - genome->header.rule_count) {
    set_pool_sync_refuse(refuse, DirectGammaPoolSyncRefuseV1::capacity);
    return false;
  }

  DirectSha256Address base_address{};
  if (!direct_sha256_genome_address(*genome, &base_address) ||
      manifest.base_genome != base_address ||
      manifest.compatible_species != base_address) {
    set_pool_sync_refuse(refuse,
                         DirectGammaPoolSyncRefuseV1::incompatible_species);
    return false;
  }
  const Root256 base_root = canonical_direct_genome_root_v1(*genome);

  for (std::size_t i = 0u; i < cause_count; ++i) {
    const DirectGammaVerifiedRecipeCauseV1& cause = causes[i];
    if (cause.abi_version != 1u || cause.pool_index >= pool_count ||
        cause.usage_index >= usage_count ||
        cause.optimization_index >= optimization_count ||
        cause.fallback_pool_index >= pool_count ||
        !valid_synced_rule(*genome, cause.developmental_rule))
      return false;
    if (cause.semantic_authority != 0u ||
        cause.experiential_authority != 0u ||
        cause.participation_authority != 0u ||
        cause.current_adult_authority != 0u ||
        cause.activation_authority != 0u ||
        cause.mature_topology_authority != 0u) {
      set_pool_sync_refuse(refuse,
                           DirectGammaPoolSyncRefuseV1::semantic_authority);
      return false;
    }
    const DirectFoundryCandidateEntryV1& candidate =
        pool_entries[cause.pool_index];
    const DirectFoundryCandidateEntryV1& fallback =
        pool_entries[cause.fallback_pool_index];
    const DirectFoundryPopulationUsageEntryV1& usage =
        usage_entries[cause.usage_index];
    const DirectFoundryOptimizationEntryV1& optimization =
        optimization_entries[cause.optimization_index];
    if (candidate.record.compatible_species != manifest.compatible_species ||
        candidate.record.candidate_body !=
            direct_gamma_verified_recipe_cause_body_address(
                cause, candidate.record.kind) ||
        !optimization_sources_match(candidate, fallback, usage, optimization)) {
      set_pool_sync_refuse(refuse,
                           DirectGammaPoolSyncRefuseV1::forged_evidence);
      return false;
    }
    for (std::size_t prior = 0u; prior < i; ++prior)
      if (pool_entries[causes[prior].pool_index].candidate_address ==
          candidate.candidate_address) {
        set_pool_sync_refuse(refuse,
                             DirectGammaPoolSyncRefuseV1::duplicate);
        return false;
      }

    if (cause.proof ==
        DirectGammaVerifiedCauseProofV1::engineering_equivalence) {
      if (candidate.record.proof_class !=
              DirectFoundryProofClassV1::logical_equivalence_claim &&
          candidate.record.proof_class !=
              DirectFoundryProofClassV1::physical_efficiency_claim) {
        set_pool_sync_refuse(
            refuse, DirectGammaPoolSyncRefuseV1::unverified_candidate);
        return false;
      }
      const DirectFoundryCandidateEntryV1* source = find_pool_candidate(
          pool_entries, pool_count, candidate.record.parent_candidate);
      if (source == nullptr ||
          !direct_foundry_equivalence_receipt_valid(
              cause.equivalence, *source, candidate, cause.resources)) {
        set_pool_sync_refuse(refuse,
                             DirectGammaPoolSyncRefuseV1::forged_evidence);
        return false;
      }
    } else if (cause.proof ==
               DirectGammaVerifiedCauseProofV1::developmental_sibling_assay) {
      DirectSha256Address assay_address{};
      if (candidate.record.proof_class !=
              DirectFoundryProofClassV1::developmental_prior_claim ||
          gamma_evidence::classify_gamma_g2_assay(
              cause.sibling_assay,
              cause.sibling_assay.experiment_identity) !=
              gamma_evidence::GammaG2Refuse::none ||
          !gamma_evidence::gamma_g2_assay_address(cause.sibling_assay,
                                                   &assay_address) ||
          candidate.record.proof_claim != assay_address ||
          cause.sibling_assay.parent_root != base_root ||
          cause.sibling_assay.matter_budget != genome->header.matter_budget ||
          cause.sibling_assay.life_function_version !=
              genome->header.life_function_version) {
        set_pool_sync_refuse(refuse,
                             DirectGammaPoolSyncRefuseV1::forged_evidence);
        return false;
      }
    } else {
      set_pool_sync_refuse(refuse,
                           DirectGammaPoolSyncRefuseV1::unverified_candidate);
      return false;
    }
  }

  DirectGenomeV1 staged = *genome;
  for (std::size_t i = 0u; i < cause_count; ++i)
    staged.rules[staged.header.rule_count++] = causes[i].developmental_rule;
  staged.header.flags |= kDirectGenomeFlagVerifiedPoolFrozen;
  staged.header.parent_root = base_root;
  const DirectSha256Address manifest_address = verified_pool_manifest_address(
      manifest, pool_entries, pool_count, usage_entries, usage_count,
      optimization_entries, optimization_count, causes, cause_count);
  std::memcpy(staged.header.delta_root.word, manifest_address.byte,
              sizeof(staged.header.delta_root.word));
  *genome = staged;
  set_pool_sync_refuse(refuse, DirectGammaPoolSyncRefuseV1::none);
  return true;
}

}  // namespace detail

bool apply_observer_prose_bytes_to_direct_genome(DirectGenomeV1* genome,
                                                 const void* bytes,
                                                 std::uint64_t byte_count) {
  (void)genome;
  (void)bytes;
  (void)byte_count;
  return false;
}

static bool stamp_direct_genome_g1_delta(DirectGenomeV1* genome, const Root256& parent_root,
                                 const Root256& delta_root,
                                 std::uint32_t life_function_version,
                                 std::uint32_t production_value) {
  if (genome == nullptr || life_function_version == 0u || production_value == 0u)
    return false;
  std::uint32_t parent_any = 0u;
  std::uint32_t delta_any = 0u;
  for (std::uint32_t i = 0u; i < 8u; ++i) {
    parent_any |= parent_root.word[i];
    delta_any |= delta_root.word[i];
  }
  if (parent_any == 0u || delta_any == 0u) return false;
  genome->header.parent_root = parent_root;
  genome->header.delta_root = delta_root;
  genome->header.life_function_version = life_function_version;
  genome->header.development_seed ^= static_cast<std::uint64_t>(production_value);
  return true;
}

namespace gamma_evidence {

bool apply_gamma_g1_executable_seed_to_direct_genome(
    DirectGenomeV1* genome, const GammaFoundryExecutableSeed& seed,
    const GammaFoundryEvidenceNote* bound_hypothesis,
    std::uint64_t bound_experiment_identity, GammaG1Refuse* refuse) {
  const GammaG1Refuse classified =
      classify_gamma_g1_seed(seed, bound_hypothesis, bound_experiment_identity);
  if (classified != GammaG1Refuse::none) {
    if (refuse != nullptr) *refuse = classified;
    return false;
  }
  if (genome == nullptr) {
    if (refuse != nullptr) *refuse = GammaG1Refuse::malformed;
    return false;
  }
  DirectSha256Address seed_address{};
  if (!gamma_g1_seed_address(seed, &seed_address)) {
    if (refuse != nullptr) *refuse = GammaG1Refuse::malformed;
    return false;
  }
  Root256 delta_root{};
  std::memcpy(delta_root.word, seed_address.byte, sizeof(delta_root.word));
  if (genome->header.delta_root == delta_root &&
      genome->header.parent_root == seed.parent_root) {
    if (refuse != nullptr) *refuse = GammaG1Refuse::duplicate;
    return false;
  }
  const Root256 current = canonical_direct_genome_root_v1(*genome);
  if (!(current == seed.parent_root)) {
    if (refuse != nullptr) *refuse = GammaG1Refuse::stale;
    return false;
  }
  if (!stamp_direct_genome_g1_delta(genome, current, delta_root,
                                   seed.life_function_version,
                                   seed.production_value)) {
    if (refuse != nullptr) *refuse = GammaG1Refuse::malformed;
    return false;
  }
  if (refuse != nullptr) *refuse = GammaG1Refuse::none;
  return true;
}

bool apply_gamma_g2_arm_to_direct_genome(
    DirectGenomeV1* genome, const GammaFoundryNeonatalSiblingAssay& assay,
    GammaG2Arm arm, std::uint64_t bound_experiment_identity,
    GammaG2Refuse* refuse) {
  const GammaG2Refuse classified =
      classify_gamma_g2_assay(assay, bound_experiment_identity);
  if (classified != GammaG2Refuse::none) {
    if (refuse != nullptr) *refuse = classified;
    return false;
  }
  if (genome == nullptr) {
    if (refuse != nullptr) *refuse = GammaG2Refuse::malformed;
    return false;
  }
  if (genome->header.matter_budget != assay.matter_budget) {
    if (refuse != nullptr) *refuse = GammaG2Refuse::unequal_matter;
    return false;
  }
  const Root256 current = canonical_direct_genome_root_v1(*genome);
  if (!(current == assay.parent_root)) {
    if (refuse != nullptr) *refuse = GammaG2Refuse::stale;
    return false;
  }
  if (arm == GammaG2Arm::minus_prior) {
    if (refuse != nullptr) *refuse = GammaG2Refuse::none;
    return true;
  }
  if (arm == GammaG2Arm::full) {
    GammaFoundryExecutableSeed seed{};
    if (!gamma_g2_arm_seed(assay, arm, &seed)) {
      if (refuse != nullptr) *refuse = GammaG2Refuse::malformed;
      return false;
    }
    GammaG1Refuse g1 = GammaG1Refuse::none;
    if (!apply_gamma_g1_executable_seed_to_direct_genome(
            genome, seed, nullptr, assay.experiment_identity, &g1)) {
      if (refuse != nullptr) *refuse = static_cast<GammaG2Refuse>(g1);
      return false;
    }
    if (refuse != nullptr) *refuse = GammaG2Refuse::none;
    return true;
  }
  if (arm != GammaG2Arm::equal_matter_sham && arm != GammaG2Arm::randomized) {
    if (refuse != nullptr) *refuse = GammaG2Refuse::unsupported;
    return false;
  }
  DirectSha256Address arm_address{};
  if (!gamma_g2_arm_address(assay, arm, &arm_address)) {
    if (refuse != nullptr) *refuse = GammaG2Refuse::malformed;
    return false;
  }
  Root256 delta_root{};
  std::memcpy(delta_root.word, arm_address.byte, sizeof(delta_root.word));
  if (genome->header.delta_root == delta_root &&
      genome->header.parent_root == assay.parent_root) {
    if (refuse != nullptr) *refuse = GammaG2Refuse::duplicate;
    return false;
  }
  if (!stamp_direct_genome_g1_delta(genome, current, delta_root,
                                   assay.life_function_version,
                                   assay.production_value)) {
    if (refuse != nullptr) *refuse = GammaG2Refuse::malformed;
    return false;
  }
  if (refuse != nullptr) *refuse = GammaG2Refuse::none;
  return true;
}

}  // namespace gamma_evidence

Root256 canonical_direct_genome_root_v1(const DirectGenomeV1& genome) {
  std::uint64_t state[8] = {0x243f6a8885a308d3ull, 0x13198a2e03707344ull,
                            0xa4093822299f31d0ull, 0x082efa98ec4e6c89ull,
                            0x452821e638d01377ull, 0xbe5466cf34e90c6cull,
                            0xc0ac29b7c97c50ddull, 0x3f84d5b5b5470917ull};
  absorb_bytes(state, &genome.header, sizeof(genome.header));
  absorb_bytes(state, genome.territories,
               sizeof(genome.territories[0]) * genome.header.territory_count);
  absorb_bytes(state, genome.fields, sizeof(genome.fields[0]) * genome.header.field_count);
  if (genome.header.abi_version == kDirectGenomeAbiV1) {
    for (std::uint32_t index = 0u; index < genome.header.rule_count; ++index)
      absorb_bytes(state, &genome.rules[index], offsetof(DirectRuleSpecV1, tract_delay));
  } else {
    absorb_bytes(state, genome.rules, sizeof(genome.rules[0]) * genome.header.rule_count);
  }
  return finish_root(state);
}

DirectGenomeLoweringV1 lower_direct_genome_v1(
    const DirectGenomeV1& genome, const DirectBodyManifestV1& body,
    const DirectDevelopmentEnvironmentV1& environment) {
  validate_direct_genome(genome);
  if (body.abi_version != kDirectBodyAbiV1 || body.binding_count > kMaxBoundaryPorts)
    throw std::invalid_argument("Direct body manifest ABI/bounds invalid");
  if (environment.abi_version != kDirectDevelopmentEnvironmentAbiV1 ||
      environment.constraint_count > kMaxDevelopmentConstraints)
    throw std::invalid_argument("Direct development environment ABI/bounds invalid");

  DirectGenomeLoweringV1 lowered{};
  lowered.direct_genome_root = canonical_direct_genome_root_v1(genome);
  GammaV1& gamma = lowered.gamma;
  gamma.header.abi_version = kCurrentAbiVersion;
  gamma.header.life_function_version = genome.header.life_function_version;
  gamma.header.seed_count = genome.header.territory_count;
  gamma.header.field_count = genome.header.field_count;
  gamma.header.rule_count = genome.header.rule_count;
  gamma.header.development_end_tick = genome.header.development_end_tick;
  gamma.header.matter_budget = genome.header.matter_budget;
  gamma.header.flags = genome.header.flags;
  gamma.header.development_seed = genome.header.development_seed;
  gamma.header.parent_root = genome.header.parent_root;
  gamma.header.delta_root = genome.header.delta_root;

  std::uint32_t maximum_reach = 1u;
  std::vector<seed_territory::DirectTerritoryRequest> requests;
  requests.reserve(genome.header.territory_count);
  for (std::uint32_t index = 0u; index < genome.header.territory_count; ++index) {
    const DirectTerritorySpecV1& territory = genome.territories[index];
    maximum_reach = std::max(maximum_reach, territory.reach);
    requests.push_back(seed_territory::DirectTerritoryRequest{
        territory.identity.ordinal, territory.identity.lineage, territory.identity.axis,
        territory.reach});
  }

  // The body owns channel attachment and is validated above, but its present
  // ABI deliberately contains no invented spatial coordinate. When it grows
  // geometry, that geometry replaces this physical-origin default rather than
  // a host-chosen territory address.
  const std::int32_t body_origin[3] = {0, 0, 0};
  const seed_territory::DirectTerritoryLattice lattice =
      seed_territory::lattice_for(requests.size(), maximum_reach, body_origin);
  seed_territory::EnvironmentOccupancyField occupancy{&lattice, &environment};
  std::vector<seed_territory::DirectDerivedOrigin> origins(genome.header.territory_count);
  // #1277/#1268: placement is seeded from the authored development_seed, not
  // the whole-genome content root. Every rule/field/territory byte is folded
  // into direct_genome_root, so seeding arbitration from it made any Γ edit
  // relocate every node -- no knockout could serve as a control. development_seed
  // is already the organism's one dedicated developmental-randomization channel
  // (read throughout direct_network_life_function.cu for branch/fuse/resource
  // draws); reusing it here keeps a deliberate reseed meaningful while making an
  // edit to unrelated genome content leave placement untouched.
  seed_territory::derive_direct_territory_origins(
      genome.header.development_seed, requests.data(), requests.size(), lattice, occupancy,
      origins.data());
  for (const seed_territory::DirectDerivedOrigin& origin : origins) {
    if (!origin.placed)
      throw std::invalid_argument("Direct territory arbitration refused a territory");
  }
  lowered.territory_layout_root =
      canonical_layout_root(lowered.direct_genome_root, origins.data(), genome.header.territory_count);

  for (std::uint32_t index = 0u; index < genome.header.territory_count; ++index) {
    const DirectTerritorySpecV1& territory = genome.territories[index];
    const seed_territory::DirectDerivedOrigin& origin = origins[index];
    gamma.seeds[index] = SeedBlock{{checked_coordinate(origin.coordinate[0],
                                                         "Direct territory x out of Gamma IR range"),
                                      checked_coordinate(origin.coordinate[1],
                                                         "Direct territory y out of Gamma IR range"),
                                      checked_coordinate(origin.coordinate[2],
                                                         "Direct territory z out of Gamma IR range")},
                                     territory.chemotype, territory.identity.lineage, territory.flags,
                                     territory.begin_tick};
  }
  for (std::uint32_t index = 0u; index < genome.header.field_count; ++index) {
    const DirectFieldSpecV1& field = genome.fields[index];
    const seed_territory::DirectDerivedOrigin& origin =
        origin_for(genome, origins.data(), field.territory);
    FieldBlock lowered_field{};
    for (std::uint32_t axis = 0u; axis < 3u; ++axis) {
      const std::int64_t coordinate = static_cast<std::int64_t>(origin.coordinate[axis]) +
                                      static_cast<std::int64_t>(field.relative_center[axis]);
      lowered_field.center[axis] = checked_coordinate(coordinate,
          "Direct relative field center out of Gamma IR range");
    }
    lowered_field.radius = field.radius;
    lowered_field.require_mask = field.require_mask;
    lowered_field.require_value = field.require_value;
    lowered_field.write_mask = field.write_mask;
    lowered_field.write_value = field.write_value;
    lowered_field.begin_tick = field.begin_tick;
    lowered_field.end_tick = field.end_tick;
    lowered_field.polarity = field.polarity;
    gamma.fields[index] = lowered_field;
  }
  for (std::uint32_t index = 0u; index < genome.header.rule_count; ++index) {
    const DirectRuleSpecV1& rule = genome.rules[index];
    lowered.rule_delay_laws[index] = rule.tract_delay;
    gamma.rules[index] = ConstructionRule{
        static_cast<RuleOpcode>(rule.opcode), rule.direction_mode, rule.flags, rule.begin_tick,
        rule.end_tick, rule.require_mask, rule.require_value, rule.write_mask, rule.write_value,
        rule.minimum_age, rule.maximum_age, rule.threshold_q32, rule.field_index, rule.extent,
        rule.child_slot, rule.branch_count};
  }
  lowered.rule_delay_law_count = genome.header.abi_version == kDirectGenomeAbiV2
                                      ? genome.header.rule_count
                                      : 0u;
  gamma.header.genome_root = canonical_genome_root(gamma);
  if (validate_genome(gamma) != GenomeValidationError::kNone)
    throw std::logic_error("Direct lowerer produced invalid Gamma IR");
  return lowered;
}

}  // namespace substrate::direct_network
