#ifndef HARDWARE_NATIVE_DIRECT_BEE_DANCE_CAPSULE_ABI_CUH
#define HARDWARE_NATIVE_DIRECT_BEE_DANCE_CAPSULE_ABI_CUH

#include <cstdint>
#include <type_traits>

#include "hardware_native/direct_foundry_recipe_candidate_pool.cuh"

namespace substrate::direct_network {

// Delivery can expose only one of these later assimilation routes. Neither
// route performs the assimilation. Engineering-equivalent physical backing
// must use the separate quiescent atomic transaction; every non-equivalent
// logical change remains an FC4 shadow/consequence/Marriage candidate.
enum class DirectBeeDanceAssimilationRouteV1 : std::uint32_t {
  quiescent_equivalent_backing_only = 1u,
  shadow_consequence_marriage_required = 2u,
};

// Opaque recipient binding for a cold external catalog. It carries no Adult
// bytes or mutable pointer. checkpoint_continuation is a frozen witness used
// to prove that delivery did not silently patch the continuing subject.
struct DirectBeeDanceRecipientBindingV1 {
  std::uint32_t abi_version = 1u;
  DirectSha256Address recipient_scope{};
  DirectSha256Address checkpoint_continuation{};
  DirectSha256Address compatible_species{};
  DirectSha256Address body_regime{};
  DirectSha256Address evaluator{};
  DirectSha256Address publication_trust_domain{};
  DirectSha256Address trusted_publisher{};
  DirectSha256Address recipient_capability{};
  DirectSha256Address consent_receipt{};
};

// A pool object transported by Bee-Dance. Every root is semantic-opaque and
// content addressed. The explicit authority fields make the only positive
// power cold dormant availability; the object cannot create resident matter,
// an Occurrence, participation, credit, current Network membership, a logical
// revision, world evidence, truth, activation, or current thought.
struct DirectBeeDanceVerifiedPoolCapsuleV1 {
  std::uint32_t abi_version = 1u;
  DirectBeeDanceAssimilationRouteV1 route =
      DirectBeeDanceAssimilationRouteV1::shadow_consequence_marriage_required;
  DirectFoundryCandidateKindV1 candidate_kind = DirectFoundryCandidateKindV1::recipe_revision;
  DirectFoundryProofClassV1 proof_class = DirectFoundryProofClassV1::unverified_proposal;
  std::uint32_t dormant_availability_authority = 1u;
  std::uint32_t activation_authority = 0u;
  std::uint32_t semantic_authority = 0u;
  std::uint32_t world_truth_authority = 0u;
  std::uint32_t logical_revision_authority = 0u;
  std::uint32_t resident_matter_authority = 0u;
  std::uint32_t occurrence_authority = 0u;
  std::uint32_t participation_authority = 0u;
  std::uint32_t credit_authority = 0u;
  std::uint32_t current_network_authority = 0u;
  std::uint32_t current_thought_authority = 0u;
  DirectSha256Address recipient_scope{};
  DirectSha256Address candidate{};
  DirectSha256Address fallback_candidate{};
  DirectSha256Address proof_receipt{};
  DirectSha256Address resource_receipt{};
  DirectSha256Address pool_head{};
  DirectSha256Address population_usage_head{};
  DirectSha256Address optimization_head{};
  DirectSha256Address optimization_nomination{};
  DirectSha256Address domain{};
  DirectSha256Address guard{};
  DirectSha256Address formal_ports{};
  DirectSha256Address evaluator{};
  DirectSha256Address compatible_species{};
  DirectSha256Address body_regime{};
  DirectSha256Address resource_regime{};
  DirectSha256Address source_lineage{};
  DirectSha256Address contextual_failures{};
};

// This object records the result of an authenticated Bee-Dance publication
// boundary. Signature bytes and their verifier receipt stay external; this ABI
// binds their content addresses to the exact capsule payload and the recipient
// capability/consent policy. A signature attests source and bytes, never truth.
struct DirectBeeDanceSourceAttestationV1 {
  std::uint32_t abi_version = 1u;
  std::uint32_t cryptographic_source_verified = 1u;
  std::uint32_t truth_authority = 0u;
  std::uint32_t runtime_authority = 0u;
  DirectSha256Address capsule_payload{};
  DirectSha256Address publisher{};
  DirectSha256Address trust_domain{};
  DirectSha256Address signature{};
  DirectSha256Address signature_verification_receipt{};
  DirectSha256Address publication_manifest{};
  DirectSha256Address source_lineage{};
  DirectSha256Address recipient_capability{};
  DirectSha256Address consent_receipt{};
};

static_assert(std::is_standard_layout_v<DirectBeeDanceRecipientBindingV1> &&
              std::is_trivially_copyable_v<DirectBeeDanceRecipientBindingV1>);
static_assert(std::is_standard_layout_v<DirectBeeDanceVerifiedPoolCapsuleV1> &&
              std::is_trivially_copyable_v<DirectBeeDanceVerifiedPoolCapsuleV1>);
static_assert(std::is_standard_layout_v<DirectBeeDanceSourceAttestationV1> &&
              std::is_trivially_copyable_v<DirectBeeDanceSourceAttestationV1>);

#if defined(__CUDACC__)
#define DIRECT_BEE_DANCE_HD __host__ __device__
#else
#define DIRECT_BEE_DANCE_HD
#endif

DIRECT_BEE_DANCE_HD inline bool direct_bee_dance_address_nonzero(
    const DirectSha256Address& address) {
  std::uint8_t any = 0u;
  for (std::uint8_t byte : address.byte)
    any |= byte;
  return any != 0u;
}

DIRECT_BEE_DANCE_HD inline void direct_bee_dance_update_u32(detail::DirectSha256State* state,
                                                            std::uint32_t value) {
  const std::uint8_t bytes[4] = {
      static_cast<std::uint8_t>(value), static_cast<std::uint8_t>(value >> 8u),
      static_cast<std::uint8_t>(value >> 16u), static_cast<std::uint8_t>(value >> 24u)};
  state->update(bytes, sizeof(bytes));
}

DIRECT_BEE_DANCE_HD inline void direct_bee_dance_update_u64(detail::DirectSha256State* state,
                                                            std::uint64_t value) {
  std::uint8_t bytes[8];
  for (std::uint32_t index = 0u; index < 8u; ++index)
    bytes[index] = static_cast<std::uint8_t>(value >> (index * 8u));
  state->update(bytes, sizeof(bytes));
}

DIRECT_BEE_DANCE_HD inline void direct_bee_dance_update_address(
    detail::DirectSha256State* state, const DirectSha256Address& address) {
  state->update(address.byte, sizeof(address.byte));
}

DIRECT_BEE_DANCE_HD inline DirectSha256Address direct_bee_dance_recipient_binding_address(
    const DirectBeeDanceRecipientBindingV1& binding) {
  static constexpr char kDomain[] = "0x1-bee-dance-recipient-binding-v1";
  detail::DirectSha256State state{};
  state.update(kDomain, sizeof(kDomain) - 1u);
  direct_bee_dance_update_u32(&state, binding.abi_version);
  const DirectSha256Address* roots[] = {
      &binding.recipient_scope,    &binding.checkpoint_continuation,
      &binding.compatible_species, &binding.body_regime,
      &binding.evaluator,          &binding.publication_trust_domain,
      &binding.trusted_publisher,  &binding.recipient_capability,
      &binding.consent_receipt};
  for (const auto* root : roots)
    direct_bee_dance_update_address(&state, *root);
  return state.finish();
}

DIRECT_BEE_DANCE_HD inline DirectSha256Address direct_bee_dance_capsule_address(
    const DirectBeeDanceVerifiedPoolCapsuleV1& capsule) {
  static constexpr char kDomain[] = "0x1-bee-dance-verified-pool-capsule-v1";
  detail::DirectSha256State state{};
  state.update(kDomain, sizeof(kDomain) - 1u);
  const std::uint32_t fields[] = {capsule.abi_version,
                                  static_cast<std::uint32_t>(capsule.route),
                                  static_cast<std::uint32_t>(capsule.candidate_kind),
                                  static_cast<std::uint32_t>(capsule.proof_class),
                                  capsule.dormant_availability_authority,
                                  capsule.activation_authority,
                                  capsule.semantic_authority,
                                  capsule.world_truth_authority,
                                  capsule.logical_revision_authority,
                                  capsule.resident_matter_authority,
                                  capsule.occurrence_authority,
                                  capsule.participation_authority,
                                  capsule.credit_authority,
                                  capsule.current_network_authority,
                                  capsule.current_thought_authority};
  for (std::uint32_t value : fields)
    direct_bee_dance_update_u32(&state, value);
  const DirectSha256Address* roots[] = {&capsule.recipient_scope,
                                        &capsule.candidate,
                                        &capsule.fallback_candidate,
                                        &capsule.proof_receipt,
                                        &capsule.resource_receipt,
                                        &capsule.pool_head,
                                        &capsule.population_usage_head,
                                        &capsule.optimization_head,
                                        &capsule.optimization_nomination,
                                        &capsule.domain,
                                        &capsule.guard,
                                        &capsule.formal_ports,
                                        &capsule.evaluator,
                                        &capsule.compatible_species,
                                        &capsule.body_regime,
                                        &capsule.resource_regime,
                                        &capsule.source_lineage,
                                        &capsule.contextual_failures};
  for (const auto* root : roots)
    direct_bee_dance_update_address(&state, *root);
  return state.finish();
}

DIRECT_BEE_DANCE_HD inline DirectSha256Address direct_bee_dance_source_attestation_address(
    const DirectBeeDanceSourceAttestationV1& attestation) {
  static constexpr char kDomain[] = "0x1-bee-dance-source-attestation-v1";
  detail::DirectSha256State state{};
  state.update(kDomain, sizeof(kDomain) - 1u);
  direct_bee_dance_update_u32(&state, attestation.abi_version);
  direct_bee_dance_update_u32(&state, attestation.cryptographic_source_verified);
  direct_bee_dance_update_u32(&state, attestation.truth_authority);
  direct_bee_dance_update_u32(&state, attestation.runtime_authority);
  const DirectSha256Address* roots[] = {&attestation.capsule_payload,
                                        &attestation.publisher,
                                        &attestation.trust_domain,
                                        &attestation.signature,
                                        &attestation.signature_verification_receipt,
                                        &attestation.publication_manifest,
                                        &attestation.source_lineage,
                                        &attestation.recipient_capability,
                                        &attestation.consent_receipt};
  for (const auto* root : roots)
    direct_bee_dance_update_address(&state, *root);
  return state.finish();
}

DIRECT_BEE_DANCE_HD inline bool direct_bee_dance_recipient_binding_valid(
    const DirectBeeDanceRecipientBindingV1& binding) {
  if (binding.abi_version != 1u)
    return false;
  const DirectSha256Address* roots[] = {
      &binding.recipient_scope,    &binding.checkpoint_continuation,
      &binding.compatible_species, &binding.body_regime,
      &binding.evaluator,          &binding.publication_trust_domain,
      &binding.trusted_publisher,  &binding.recipient_capability,
      &binding.consent_receipt};
  for (const auto* root : roots)
    if (!direct_bee_dance_address_nonzero(*root))
      return false;
  return true;
}

DIRECT_BEE_DANCE_HD inline bool direct_bee_dance_capsule_authority_valid(
    const DirectBeeDanceVerifiedPoolCapsuleV1& capsule) {
  return capsule.abi_version == 1u && capsule.dormant_availability_authority == 1u &&
         capsule.activation_authority == 0u && capsule.semantic_authority == 0u &&
         capsule.world_truth_authority == 0u && capsule.logical_revision_authority == 0u &&
         capsule.resident_matter_authority == 0u && capsule.occurrence_authority == 0u &&
         capsule.participation_authority == 0u && capsule.credit_authority == 0u &&
         capsule.current_network_authority == 0u && capsule.current_thought_authority == 0u;
}

DIRECT_BEE_DANCE_HD inline bool direct_bee_dance_attestation_valid(
    const DirectBeeDanceSourceAttestationV1& attestation,
    const DirectBeeDanceVerifiedPoolCapsuleV1& capsule,
    const DirectBeeDanceRecipientBindingV1& binding) {
  if (attestation.abi_version != 1u || attestation.cryptographic_source_verified != 1u ||
      attestation.truth_authority != 0u || attestation.runtime_authority != 0u ||
      attestation.capsule_payload != direct_bee_dance_capsule_address(capsule) ||
      attestation.publisher != binding.trusted_publisher ||
      attestation.trust_domain != binding.publication_trust_domain ||
      attestation.recipient_capability != binding.recipient_capability ||
      attestation.consent_receipt != binding.consent_receipt ||
      attestation.source_lineage != capsule.source_lineage)
    return false;
  const DirectSha256Address* roots[] = {&attestation.signature,
                                        &attestation.signature_verification_receipt,
                                        &attestation.publication_manifest};
  for (const auto* root : roots)
    if (!direct_bee_dance_address_nonzero(*root))
      return false;
  return true;
}

#undef DIRECT_BEE_DANCE_HD

}  // namespace substrate::direct_network

#endif  // HARDWARE_NATIVE_DIRECT_BEE_DANCE_CAPSULE_ABI_CUH
