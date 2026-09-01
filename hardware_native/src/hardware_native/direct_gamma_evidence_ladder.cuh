#ifndef HARDWARE_NATIVE_DIRECT_GAMMA_EVIDENCE_LADDER_CUH
#define HARDWARE_NATIVE_DIRECT_GAMMA_EVIDENCE_LADDER_CUH

#include <cstdint>
#include <cstring>
#include <type_traits>

#include "hardware_native/direct_content_address.cuh"
#include "hardware_native/direct_network_certification.cuh"

namespace substrate::direct_network::gamma_evidence {

enum class GammaBioEvidenceGrade : std::uint32_t {
  B0_intuition_or_analogy = 0u,
  B1_animal_or_computational_property = 1u,
  B2_replicated_human_behavior = 2u,
  B3_neural_developmental_support = 3u,
  B4_causal_intervention_dissociation = 4u,
};

enum class GammaBioNoteKind : std::uint32_t {
  intuition = 1u,
  analogy = 2u,
  animal_property = 3u,
  computational_property = 4u,
  replicated_human_behavior = 5u,
  neural_developmental_support = 6u,
  causal_intervention = 7u,
  strong_natural_dissociation = 8u,
};

struct GammaBioEvidenceNote {
  GammaBioEvidenceGrade grade;
  GammaBioNoteKind kind;
  std::uint64_t note_identity;
  std::uint64_t source_identity;
  std::uint64_t protocol_identity;
  std::uint64_t intervention_control_identity;
  std::uint64_t limits_identity;
  std::uint32_t resident_authority;
  std::uint32_t gamma_authority;
};
static_assert(std::is_standard_layout_v<GammaBioEvidenceNote> &&
              std::is_trivial_v<GammaBioEvidenceNote>);

inline bool record_gamma_b0_observer_note(GammaBioEvidenceNote* out,
                                          GammaBioNoteKind kind,
                                          std::uint64_t note_identity) {
  if (out == nullptr || note_identity == 0ull) return false;
  if (kind != GammaBioNoteKind::intuition && kind != GammaBioNoteKind::analogy)
    return false;
  *out = GammaBioEvidenceNote{};
  out->grade = GammaBioEvidenceGrade::B0_intuition_or_analogy;
  out->kind = kind;
  out->note_identity = note_identity;
  out->resident_authority = 0u;
  out->gamma_authority = 0u;
  return true;
}

inline constexpr bool gamma_bio_note_has_resident_authority(
    const GammaBioEvidenceNote& note) {
  return note.resident_authority != 0u;
}

inline constexpr bool gamma_bio_note_has_gamma_authority(
    const GammaBioEvidenceNote& note) {
  return note.gamma_authority != 0u;
}

inline constexpr bool gamma_bio_b0_may_promote(const GammaBioEvidenceNote&) {
  return false;
}

inline constexpr bool gamma_bio_b0_grants_certification(
    const GammaBioEvidenceNote& note) {
  (void)note;
  return false;
}

inline bool apply_gamma_b0_to_certification(
    certification::NetworkFoundationReceipt* receipt,
    const GammaBioEvidenceNote& note) {
  (void)receipt;
  (void)note;
  return false;
}

inline bool record_gamma_b1_observer_note(GammaBioEvidenceNote* out,
                                          GammaBioNoteKind kind,
                                          std::uint64_t note_identity) {
  if (out == nullptr || note_identity == 0ull) return false;
  if (kind != GammaBioNoteKind::animal_property &&
      kind != GammaBioNoteKind::computational_property)
    return false;
  *out = GammaBioEvidenceNote{};
  out->grade = GammaBioEvidenceGrade::B1_animal_or_computational_property;
  out->kind = kind;
  out->note_identity = note_identity;
  out->resident_authority = 0u;
  out->gamma_authority = 0u;
  return true;
}

inline constexpr bool gamma_bio_b1_may_promote_silicon_prior(
    const GammaBioEvidenceNote&) {
  return false;
}

inline constexpr bool gamma_bio_b1_grants_certification(
    const GammaBioEvidenceNote& note) {
  (void)note;
  return false;
}

inline bool apply_gamma_b1_to_certification(
    certification::NetworkFoundationReceipt* receipt,
    const GammaBioEvidenceNote& note) {
  (void)receipt;
  (void)note;
  return false;
}

inline bool record_gamma_b2_observer_note(GammaBioEvidenceNote* out,
                                          GammaBioNoteKind kind,
                                          std::uint64_t source_identity,
                                          std::uint64_t protocol_identity) {
  if (out == nullptr || source_identity == 0ull || protocol_identity == 0ull)
    return false;
  if (kind != GammaBioNoteKind::replicated_human_behavior) return false;
  *out = GammaBioEvidenceNote{};
  out->grade = GammaBioEvidenceGrade::B2_replicated_human_behavior;
  out->kind = kind;
  out->note_identity = source_identity;
  out->source_identity = source_identity;
  out->protocol_identity = protocol_identity;
  out->resident_authority = 0u;
  out->gamma_authority = 0u;
  return true;
}

inline constexpr bool gamma_bio_b2_may_promote_silicon_prior(
    const GammaBioEvidenceNote&) {
  return false;
}

inline constexpr bool gamma_bio_b2_may_become_semantic_route(
    const GammaBioEvidenceNote&) {
  return false;
}

inline constexpr bool gamma_bio_b2_grants_certification(
    const GammaBioEvidenceNote& note) {
  (void)note;
  return false;
}

inline constexpr bool gamma_bio_b2_grants_eligibility(
    const GammaBioEvidenceNote& note) {
  (void)note;
  return false;
}

inline bool apply_gamma_b2_to_certification(
    certification::NetworkFoundationReceipt* receipt,
    const GammaBioEvidenceNote& note) {
  (void)receipt;
  (void)note;
  return false;
}

inline bool record_gamma_b3_observer_note(GammaBioEvidenceNote* out,
                                          GammaBioNoteKind kind,
                                          std::uint64_t source_identity,
                                          std::uint64_t protocol_identity,
                                          std::uint64_t limits_identity) {
  if (out == nullptr || source_identity == 0ull || protocol_identity == 0ull ||
      limits_identity == 0ull)
    return false;
  if (kind != GammaBioNoteKind::neural_developmental_support) return false;
  *out = GammaBioEvidenceNote{};
  out->grade = GammaBioEvidenceGrade::B3_neural_developmental_support;
  out->kind = kind;
  out->note_identity = source_identity;
  out->source_identity = source_identity;
  out->protocol_identity = protocol_identity;
  out->limits_identity = limits_identity;
  out->resident_authority = 0u;
  out->gamma_authority = 0u;
  return true;
}

inline constexpr bool gamma_bio_b3_may_promote_silicon_prior(
    const GammaBioEvidenceNote&) {
  return false;
}

inline constexpr bool gamma_bio_b3_may_become_semantic_route(
    const GammaBioEvidenceNote&) {
  return false;
}

inline constexpr bool gamma_bio_b3_grants_certification(
    const GammaBioEvidenceNote& note) {
  (void)note;
  return false;
}

inline constexpr bool gamma_bio_b3_grants_eligibility(
    const GammaBioEvidenceNote& note) {
  (void)note;
  return false;
}

inline bool apply_gamma_b3_to_certification(
    certification::NetworkFoundationReceipt* receipt,
    const GammaBioEvidenceNote& note) {
  (void)receipt;
  (void)note;
  return false;
}

inline bool record_gamma_b4_observer_note(
    GammaBioEvidenceNote* out, GammaBioNoteKind kind,
    std::uint64_t source_identity, std::uint64_t protocol_identity,
    std::uint64_t intervention_control_identity,
    std::uint64_t limits_identity) {
  if (out == nullptr || source_identity == 0ull || protocol_identity == 0ull ||
      intervention_control_identity == 0ull || limits_identity == 0ull)
    return false;
  if (kind != GammaBioNoteKind::causal_intervention &&
      kind != GammaBioNoteKind::strong_natural_dissociation)
    return false;
  *out = GammaBioEvidenceNote{};
  out->grade = GammaBioEvidenceGrade::B4_causal_intervention_dissociation;
  out->kind = kind;
  out->note_identity = source_identity;
  out->source_identity = source_identity;
  out->protocol_identity = protocol_identity;
  out->intervention_control_identity = intervention_control_identity;
  out->limits_identity = limits_identity;
  out->resident_authority = 0u;
  out->gamma_authority = 0u;
  return true;
}

inline constexpr bool gamma_bio_b4_may_promote_silicon_prior(
    const GammaBioEvidenceNote&) {
  return false;
}

inline constexpr bool gamma_bio_b4_may_become_semantic_route(
    const GammaBioEvidenceNote&) {
  return false;
}

inline constexpr bool gamma_bio_b4_grants_certification(
    const GammaBioEvidenceNote& note) {
  (void)note;
  return false;
}

inline constexpr bool gamma_bio_b4_grants_eligibility(
    const GammaBioEvidenceNote& note) {
  (void)note;
  return false;
}

inline bool apply_gamma_b4_to_certification(
    certification::NetworkFoundationReceipt* receipt,
    const GammaBioEvidenceNote& note) {
  (void)receipt;
  (void)note;
  return false;
}

enum class GammaFoundryEvidenceGrade : std::uint32_t {
  G0_prose_hypothesis = 0u,
  G1_executable_seed_authored = 1u,
  G2_neonatal_sibling_assay = 2u,
};

enum class GammaFoundryNoteKind : std::uint32_t {
  protected_property_seed_hypothesis = 1u,
  protected_property_falsifier = 2u,
  executable_semantic_blank_delta = 3u,
  neonatal_sibling_assay = 4u,
  equal_matter_sham_delta = 5u,
  randomized_fresh_delta = 6u,
};

enum class GammaG1Refuse : std::uint32_t {
  none = 0u,
  stale = 1u,
  duplicate = 2u,
  forged = 3u,
  mixed_context = 4u,
  malformed = 5u,
  unsupported = 6u,
};

struct GammaFoundryEvidenceNote {
  GammaFoundryEvidenceGrade grade;
  GammaFoundryNoteKind kind;
  std::uint64_t note_identity;
  std::uint64_t property_identity;
  std::uint64_t falsifier_identity;
  std::uint32_t resident_authority;
  std::uint32_t gamma_authority;
};
static_assert(std::is_standard_layout_v<GammaFoundryEvidenceNote> &&
              std::is_trivial_v<GammaFoundryEvidenceNote>);

inline bool record_gamma_g0_observer_note(GammaFoundryEvidenceNote* out,
                                          GammaFoundryNoteKind kind,
                                          std::uint64_t note_identity,
                                          std::uint64_t property_identity,
                                          std::uint64_t falsifier_identity) {
  if (out == nullptr || note_identity == 0ull || property_identity == 0ull)
    return false;
  if (kind == GammaFoundryNoteKind::protected_property_seed_hypothesis) {
    *out = GammaFoundryEvidenceNote{};
    out->grade = GammaFoundryEvidenceGrade::G0_prose_hypothesis;
    out->kind = kind;
    out->note_identity = note_identity;
    out->property_identity = property_identity;
    out->falsifier_identity = falsifier_identity;
    out->resident_authority = 0u;
    out->gamma_authority = 0u;
    return true;
  }
  if (kind != GammaFoundryNoteKind::protected_property_falsifier ||
      falsifier_identity == 0ull)
    return false;
  *out = GammaFoundryEvidenceNote{};
  out->grade = GammaFoundryEvidenceGrade::G0_prose_hypothesis;
  out->kind = kind;
  out->note_identity = note_identity;
  out->property_identity = property_identity;
  out->falsifier_identity = falsifier_identity;
  out->resident_authority = 0u;
  out->gamma_authority = 0u;
  return true;
}

inline constexpr bool gamma_foundry_note_has_resident_authority(
    const GammaFoundryEvidenceNote& note) {
  return note.resident_authority != 0u;
}

inline constexpr bool gamma_foundry_note_has_gamma_authority(
    const GammaFoundryEvidenceNote& note) {
  return note.gamma_authority != 0u;
}

inline constexpr bool gamma_g0_may_mutate_gamma(const GammaFoundryEvidenceNote&) {
  return false;
}

inline constexpr bool gamma_g0_may_become_semantic_route(
    const GammaFoundryEvidenceNote&) {
  return false;
}

inline constexpr bool gamma_g0_grants_certification(
    const GammaFoundryEvidenceNote& note) {
  (void)note;
  return false;
}

inline bool apply_gamma_g0_to_certification(
    certification::NetworkFoundationReceipt* receipt,
    const GammaFoundryEvidenceNote& note) {
  (void)receipt;
  (void)note;
  return false;
}

struct GammaFoundryExecutableSeed {
  GammaFoundryEvidenceGrade grade;
  GammaFoundryNoteKind kind;
  std::uint64_t source_identity;
  std::uint64_t property_identity;
  std::uint64_t version_identity;
  std::uint64_t experiment_identity;
  std::uint64_t ablation_identity;
  Root256 parent_root;
  std::uint32_t life_function_version;
  std::uint32_t production_value;
  std::uint32_t flags;
  std::uint32_t resident_authority;
  std::uint32_t gamma_authority;
};
static_assert(std::is_standard_layout_v<GammaFoundryExecutableSeed> &&
              std::is_trivial_v<GammaFoundryExecutableSeed>);

inline bool root256_is_nonzero(const Root256& root) {
  std::uint32_t any = 0u;
  for (std::uint32_t word : root.word) any |= word;
  return any != 0u;
}

inline bool identities_collide(std::uint64_t a, std::uint64_t b, std::uint64_t c,
                               std::uint64_t d, std::uint64_t e) {
  const std::uint64_t ids[5] = {a, b, c, d, e};
  for (std::uint32_t i = 0u; i < 5u; ++i)
    for (std::uint32_t j = i + 1u; j < 5u; ++j)
      if (ids[i] == ids[j]) return true;
  return false;
}

inline bool author_gamma_g1_executable_seed(
    GammaFoundryExecutableSeed* out, const GammaFoundryEvidenceNote& hypothesis,
    const Root256& parent_root, std::uint64_t version_identity,
    std::uint64_t experiment_identity, std::uint64_t ablation_identity,
    std::uint32_t life_function_version, std::uint32_t production_value) {
  if (out == nullptr) return false;
  if (hypothesis.grade != GammaFoundryEvidenceGrade::G0_prose_hypothesis ||
      hypothesis.kind != GammaFoundryNoteKind::protected_property_seed_hypothesis)
    return false;
  if (hypothesis.note_identity == 0ull || hypothesis.property_identity == 0ull)
    return false;
  if (gamma_foundry_note_has_resident_authority(hypothesis) ||
      gamma_foundry_note_has_gamma_authority(hypothesis))
    return false;
  if (version_identity == 0ull || experiment_identity == 0ull ||
      ablation_identity == 0ull || life_function_version == 0u ||
      production_value == 0u || !root256_is_nonzero(parent_root))
    return false;
  if (identities_collide(hypothesis.note_identity, hypothesis.property_identity,
                         version_identity, experiment_identity,
                         ablation_identity))
    return false;
  *out = GammaFoundryExecutableSeed{};
  out->grade = GammaFoundryEvidenceGrade::G1_executable_seed_authored;
  out->kind = GammaFoundryNoteKind::executable_semantic_blank_delta;
  out->source_identity = hypothesis.note_identity;
  out->property_identity = hypothesis.property_identity;
  out->version_identity = version_identity;
  out->experiment_identity = experiment_identity;
  out->ablation_identity = ablation_identity;
  out->parent_root = parent_root;
  out->life_function_version = life_function_version;
  out->production_value = production_value;
  out->flags = 0u;
  out->resident_authority = 0u;
  out->gamma_authority = 0u;
  return true;
}

inline bool gamma_g1_seed_address(const GammaFoundryExecutableSeed& seed,
                                  DirectSha256Address* out) {
  if (out == nullptr) return false;
  static constexpr char kDomain[] = "0x1-direct-gamma-g1-executable-seed-v1";
  detail::DirectSha256State state{};
  state.update(kDomain, sizeof(kDomain) - 1u);
  const std::uint32_t grade = static_cast<std::uint32_t>(seed.grade);
  const std::uint32_t kind = static_cast<std::uint32_t>(seed.kind);
  state.update(&grade, sizeof(grade));
  state.update(&kind, sizeof(kind));
  state.update(&seed.source_identity, sizeof(seed.source_identity));
  state.update(&seed.property_identity, sizeof(seed.property_identity));
  state.update(&seed.version_identity, sizeof(seed.version_identity));
  state.update(&seed.experiment_identity, sizeof(seed.experiment_identity));
  state.update(&seed.ablation_identity, sizeof(seed.ablation_identity));
  state.update(seed.parent_root.word, sizeof(seed.parent_root.word));
  state.update(&seed.life_function_version, sizeof(seed.life_function_version));
  state.update(&seed.production_value, sizeof(seed.production_value));
  state.update(&seed.flags, sizeof(seed.flags));
  state.update(&seed.resident_authority, sizeof(seed.resident_authority));
  state.update(&seed.gamma_authority, sizeof(seed.gamma_authority));
  *out = state.finish();
  return true;
}

inline constexpr bool gamma_g1_may_become_semantic_route(
    const GammaFoundryExecutableSeed&) {
  return false;
}

inline constexpr bool gamma_g1_grants_certification(
    const GammaFoundryExecutableSeed& seed) {
  (void)seed;
  return false;
}

inline bool apply_gamma_g1_to_certification(
    certification::NetworkFoundationReceipt* receipt,
    const GammaFoundryExecutableSeed& seed) {
  (void)receipt;
  (void)seed;
  return false;
}

inline GammaG1Refuse classify_gamma_g1_seed(
    const GammaFoundryExecutableSeed& seed,
    const GammaFoundryEvidenceNote* bound_hypothesis,
    std::uint64_t bound_experiment_identity) {
  if (seed.grade != GammaFoundryEvidenceGrade::G1_executable_seed_authored ||
      seed.kind != GammaFoundryNoteKind::executable_semantic_blank_delta)
    return GammaG1Refuse::unsupported;
  if (seed.resident_authority != 0u || seed.gamma_authority != 0u)
    return GammaG1Refuse::forged;
  if (seed.flags != 0u) return GammaG1Refuse::unsupported;
  if (seed.source_identity == 0ull || seed.property_identity == 0ull ||
      seed.version_identity == 0ull || seed.experiment_identity == 0ull ||
      seed.ablation_identity == 0ull || seed.life_function_version == 0u ||
      seed.production_value == 0u || !root256_is_nonzero(seed.parent_root))
    return GammaG1Refuse::malformed;
  if (identities_collide(seed.source_identity, seed.property_identity,
                         seed.version_identity, seed.experiment_identity,
                         seed.ablation_identity))
    return GammaG1Refuse::malformed;
  if (bound_hypothesis != nullptr) {
    if (bound_hypothesis->grade !=
            GammaFoundryEvidenceGrade::G0_prose_hypothesis ||
        bound_hypothesis->kind !=
            GammaFoundryNoteKind::protected_property_seed_hypothesis ||
        bound_hypothesis->note_identity != seed.source_identity ||
        bound_hypothesis->property_identity != seed.property_identity)
      return GammaG1Refuse::mixed_context;
  }
  if (bound_experiment_identity != 0ull &&
      bound_experiment_identity != seed.experiment_identity)
    return GammaG1Refuse::mixed_context;
  return GammaG1Refuse::none;
}

bool apply_gamma_g1_executable_seed_to_direct_genome(
    DirectGenomeV1* genome, const GammaFoundryExecutableSeed& seed,
    const GammaFoundryEvidenceNote* bound_hypothesis,
    std::uint64_t bound_experiment_identity, GammaG1Refuse* refuse);

enum class GammaG2Arm : std::uint32_t {
  full = 1u,
  minus_prior = 2u,
  equal_matter_sham = 3u,
  randomized = 4u,
};

enum class GammaG2Refuse : std::uint32_t {
  none = 0u,
  stale = 1u,
  duplicate = 2u,
  forged = 3u,
  mixed_context = 4u,
  malformed = 5u,
  unsupported = 6u,
  unequal_matter = 7u,
};

struct GammaFoundryNeonatalSiblingAssay {
  GammaFoundryEvidenceGrade grade;
  GammaFoundryNoteKind kind;
  std::uint64_t source_identity;
  std::uint64_t property_identity;
  std::uint64_t version_identity;
  std::uint64_t experiment_identity;
  std::uint64_t full_ablation_identity;
  std::uint64_t minus_ablation_identity;
  std::uint64_t sham_ablation_identity;
  std::uint64_t randomized_ablation_identity;
  Root256 parent_root;
  std::uint32_t life_function_version;
  std::uint32_t production_value;
  std::uint32_t matter_budget;
  std::uint32_t flags;
  std::uint32_t resident_authority;
  std::uint32_t gamma_authority;
};
static_assert(std::is_standard_layout_v<GammaFoundryNeonatalSiblingAssay> &&
              std::is_trivial_v<GammaFoundryNeonatalSiblingAssay>);

inline bool any_identities_collide(const std::uint64_t* ids, std::uint32_t n) {
  if (ids == nullptr) return true;
  for (std::uint32_t i = 0u; i < n; ++i)
    for (std::uint32_t j = i + 1u; j < n; ++j)
      if (ids[i] == ids[j]) return true;
  return false;
}

inline bool author_gamma_g2_neonatal_sibling_assay(
    GammaFoundryNeonatalSiblingAssay* out, const GammaFoundryExecutableSeed& seed,
    std::uint64_t minus_ablation_identity, std::uint64_t sham_ablation_identity,
    std::uint64_t randomized_ablation_identity, std::uint32_t matter_budget) {
  if (out == nullptr) return false;
  if (classify_gamma_g1_seed(seed, nullptr, seed.experiment_identity) !=
      GammaG1Refuse::none)
    return false;
  if (minus_ablation_identity == 0ull || sham_ablation_identity == 0ull ||
      randomized_ablation_identity == 0ull || matter_budget == 0u)
    return false;
  const std::uint64_t ids[] = {
      seed.source_identity,       seed.property_identity,
      seed.version_identity,      seed.experiment_identity,
      seed.ablation_identity,     minus_ablation_identity,
      sham_ablation_identity,     randomized_ablation_identity};
  if (any_identities_collide(ids, 8u)) return false;
  *out = GammaFoundryNeonatalSiblingAssay{};
  out->grade = GammaFoundryEvidenceGrade::G2_neonatal_sibling_assay;
  out->kind = GammaFoundryNoteKind::neonatal_sibling_assay;
  out->source_identity = seed.source_identity;
  out->property_identity = seed.property_identity;
  out->version_identity = seed.version_identity;
  out->experiment_identity = seed.experiment_identity;
  out->full_ablation_identity = seed.ablation_identity;
  out->minus_ablation_identity = minus_ablation_identity;
  out->sham_ablation_identity = sham_ablation_identity;
  out->randomized_ablation_identity = randomized_ablation_identity;
  out->parent_root = seed.parent_root;
  out->life_function_version = seed.life_function_version;
  out->production_value = seed.production_value;
  out->matter_budget = matter_budget;
  out->flags = 0u;
  out->resident_authority = 0u;
  out->gamma_authority = 0u;
  return true;
}

inline GammaG2Refuse classify_gamma_g2_assay(
    const GammaFoundryNeonatalSiblingAssay& assay,
    std::uint64_t bound_experiment_identity) {
  if (assay.grade != GammaFoundryEvidenceGrade::G2_neonatal_sibling_assay ||
      assay.kind != GammaFoundryNoteKind::neonatal_sibling_assay)
    return GammaG2Refuse::unsupported;
  if (assay.resident_authority != 0u || assay.gamma_authority != 0u)
    return GammaG2Refuse::forged;
  if (assay.flags != 0u) return GammaG2Refuse::unsupported;
  if (assay.source_identity == 0ull || assay.property_identity == 0ull ||
      assay.version_identity == 0ull || assay.experiment_identity == 0ull ||
      assay.full_ablation_identity == 0ull ||
      assay.minus_ablation_identity == 0ull ||
      assay.sham_ablation_identity == 0ull ||
      assay.randomized_ablation_identity == 0ull ||
      assay.life_function_version == 0u || assay.production_value == 0u ||
      assay.matter_budget == 0u || !root256_is_nonzero(assay.parent_root))
    return GammaG2Refuse::malformed;
  const std::uint64_t ids[] = {
      assay.source_identity,          assay.property_identity,
      assay.version_identity,         assay.experiment_identity,
      assay.full_ablation_identity,   assay.minus_ablation_identity,
      assay.sham_ablation_identity,   assay.randomized_ablation_identity};
  if (any_identities_collide(ids, 8u)) return GammaG2Refuse::malformed;
  if (bound_experiment_identity != 0ull &&
      bound_experiment_identity != assay.experiment_identity)
    return GammaG2Refuse::mixed_context;
  return GammaG2Refuse::none;
}

inline bool gamma_g2_arm_seed(const GammaFoundryNeonatalSiblingAssay& assay,
                              GammaG2Arm arm, GammaFoundryExecutableSeed* out) {
  if (out == nullptr || classify_gamma_g2_assay(assay, 0ull) != GammaG2Refuse::none)
    return false;
  if (arm == GammaG2Arm::minus_prior) return false;
  *out = GammaFoundryExecutableSeed{};
  out->source_identity = assay.source_identity;
  out->property_identity = assay.property_identity;
  out->version_identity = assay.version_identity;
  out->experiment_identity = assay.experiment_identity;
  out->parent_root = assay.parent_root;
  out->life_function_version = assay.life_function_version;
  out->production_value = assay.production_value;
  out->flags = 0u;
  out->resident_authority = 0u;
  out->gamma_authority = 0u;
  if (arm == GammaG2Arm::full) {
    out->grade = GammaFoundryEvidenceGrade::G1_executable_seed_authored;
    out->kind = GammaFoundryNoteKind::executable_semantic_blank_delta;
    out->ablation_identity = assay.full_ablation_identity;
    return true;
  }
  if (arm == GammaG2Arm::equal_matter_sham) {
    out->grade = GammaFoundryEvidenceGrade::G2_neonatal_sibling_assay;
    out->kind = GammaFoundryNoteKind::equal_matter_sham_delta;
    out->ablation_identity = assay.sham_ablation_identity;
    return true;
  }
  if (arm == GammaG2Arm::randomized) {
    out->grade = GammaFoundryEvidenceGrade::G2_neonatal_sibling_assay;
    out->kind = GammaFoundryNoteKind::randomized_fresh_delta;
    out->ablation_identity = assay.randomized_ablation_identity;
    return true;
  }
  return false;
}

inline bool gamma_g2_assay_address(const GammaFoundryNeonatalSiblingAssay& assay,
                                   DirectSha256Address* out) {
  if (out == nullptr) return false;
  static constexpr char kDomain[] = "0x1-direct-gamma-g2-neonatal-sibling-assay-v1";
  detail::DirectSha256State state{};
  state.update(kDomain, sizeof(kDomain) - 1u);
  const std::uint32_t grade = static_cast<std::uint32_t>(assay.grade);
  const std::uint32_t kind = static_cast<std::uint32_t>(assay.kind);
  state.update(&grade, sizeof(grade));
  state.update(&kind, sizeof(kind));
  state.update(&assay.source_identity, sizeof(assay.source_identity));
  state.update(&assay.property_identity, sizeof(assay.property_identity));
  state.update(&assay.version_identity, sizeof(assay.version_identity));
  state.update(&assay.experiment_identity, sizeof(assay.experiment_identity));
  state.update(&assay.full_ablation_identity, sizeof(assay.full_ablation_identity));
  state.update(&assay.minus_ablation_identity,
               sizeof(assay.minus_ablation_identity));
  state.update(&assay.sham_ablation_identity, sizeof(assay.sham_ablation_identity));
  state.update(&assay.randomized_ablation_identity,
               sizeof(assay.randomized_ablation_identity));
  state.update(assay.parent_root.word, sizeof(assay.parent_root.word));
  state.update(&assay.life_function_version, sizeof(assay.life_function_version));
  state.update(&assay.production_value, sizeof(assay.production_value));
  state.update(&assay.matter_budget, sizeof(assay.matter_budget));
  state.update(&assay.flags, sizeof(assay.flags));
  state.update(&assay.resident_authority, sizeof(assay.resident_authority));
  state.update(&assay.gamma_authority, sizeof(assay.gamma_authority));
  *out = state.finish();
  return true;
}

inline bool gamma_g2_arm_address(const GammaFoundryNeonatalSiblingAssay& assay,
                                 GammaG2Arm arm, DirectSha256Address* out) {
  if (out == nullptr) return false;
  if (arm == GammaG2Arm::full) {
    GammaFoundryExecutableSeed seed{};
    return gamma_g2_arm_seed(assay, arm, &seed) && gamma_g1_seed_address(seed, out);
  }
  if (arm == GammaG2Arm::minus_prior) {
    static constexpr char kMinus[] = "0x1-direct-gamma-g2-minus-prior-v1";
    detail::DirectSha256State state{};
    state.update(kMinus, sizeof(kMinus) - 1u);
    state.update(&assay.minus_ablation_identity, sizeof(assay.minus_ablation_identity));
    state.update(&assay.experiment_identity, sizeof(assay.experiment_identity));
    state.update(assay.parent_root.word, sizeof(assay.parent_root.word));
    *out = state.finish();
    return true;
  }
  GammaFoundryExecutableSeed seed{};
  if (!gamma_g2_arm_seed(assay, arm, &seed)) return false;
  static constexpr char kDomain[] = "0x1-direct-gamma-g2-control-arm-v1";
  detail::DirectSha256State state{};
  state.update(kDomain, sizeof(kDomain) - 1u);
  const std::uint32_t grade = static_cast<std::uint32_t>(seed.grade);
  const std::uint32_t kind = static_cast<std::uint32_t>(seed.kind);
  const std::uint32_t arm_word = static_cast<std::uint32_t>(arm);
  state.update(&grade, sizeof(grade));
  state.update(&kind, sizeof(kind));
  state.update(&arm_word, sizeof(arm_word));
  state.update(&seed.source_identity, sizeof(seed.source_identity));
  state.update(&seed.property_identity, sizeof(seed.property_identity));
  state.update(&seed.version_identity, sizeof(seed.version_identity));
  state.update(&seed.experiment_identity, sizeof(seed.experiment_identity));
  state.update(&seed.ablation_identity, sizeof(seed.ablation_identity));
  state.update(seed.parent_root.word, sizeof(seed.parent_root.word));
  state.update(&seed.life_function_version, sizeof(seed.life_function_version));
  state.update(&seed.production_value, sizeof(seed.production_value));
  state.update(&assay.matter_budget, sizeof(assay.matter_budget));
  *out = state.finish();
  return true;
}

inline constexpr bool gamma_g2_may_become_semantic_route(
    const GammaFoundryNeonatalSiblingAssay&) {
  return false;
}

inline constexpr bool gamma_g2_grants_certification(
    const GammaFoundryNeonatalSiblingAssay& assay) {
  (void)assay;
  return false;
}

inline bool apply_gamma_g2_to_certification(
    certification::NetworkFoundationReceipt* receipt,
    const GammaFoundryNeonatalSiblingAssay& assay) {
  (void)receipt;
  (void)assay;
  return false;
}

inline bool gamma_g2_canalized_phenotype(const DirectBirthReceiptV1& left,
                                         const DirectBirthReceiptV1& right) {
  return left.node_count == right.node_count &&
         left.territory_count == right.territory_count &&
         left.resident_rule_count == right.resident_rule_count &&
         left.external_life_function_detached ==
             right.external_life_function_detached;
}

bool apply_gamma_g2_arm_to_direct_genome(
    DirectGenomeV1* genome, const GammaFoundryNeonatalSiblingAssay& assay,
    GammaG2Arm arm, std::uint64_t bound_experiment_identity,
    GammaG2Refuse* refuse);

}  // namespace substrate::direct_network::gamma_evidence

#endif
