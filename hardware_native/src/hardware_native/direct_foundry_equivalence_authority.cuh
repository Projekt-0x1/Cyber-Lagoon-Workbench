#ifndef HARDWARE_NATIVE_DIRECT_FOUNDRY_EQUIVALENCE_AUTHORITY_CUH
#define HARDWARE_NATIVE_DIRECT_FOUNDRY_EQUIVALENCE_AUTHORITY_CUH

#include <cstdint>
#include <type_traits>

#include "hardware_native/direct_foundry_recipe_candidate_pool.cuh"
#include "hardware_native/direct_foundry_resource_receipt.cuh"

namespace substrate::direct_network {

enum class DirectFoundryEquivalenceModeV1 : std::uint32_t {
  exact = 1u,
  bounded_tolerance = 2u,
};

// Observer-side computation proof. All protected behavioral surfaces are
// opaque content roots. They prove engineering equivalence only: explicit
// authority bits make world meaning, lived evidence, participation, credit,
// active Network selection and current thought unavailable.
struct DirectFoundryEquivalenceReceiptV1 {
  std::uint32_t abi_version = 1u;
  DirectFoundryEquivalenceModeV1 mode = DirectFoundryEquivalenceModeV1::exact;
  std::uint32_t domain_case_count = 0u;
  std::uint32_t evaluated_case_count = 0u;
  std::uint32_t exact_match_count = 0u;
  std::uint32_t representation_authority = 1u;
  std::uint32_t engineering_authority = 1u;
  std::uint32_t semantic_authority = 0u;
  std::uint32_t experiential_authority = 0u;
  std::uint32_t participation_authority = 0u;
  std::uint32_t credit_authority = 0u;
  std::uint32_t current_adult_authority = 0u;
  std::uint64_t tolerance_q32 = 0u;
  std::uint64_t maximum_error_q32 = 0u;
  std::int32_t guard_min_q16 = 0;
  std::int32_t guard_max_q16 = 0;
  DirectSha256Address source_candidate{};
  DirectSha256Address candidate{};
  DirectSha256Address relation_contract{};
  DirectSha256Address complete_domain{};
  DirectSha256Address guard{};
  DirectSha256Address formal_ports{};
  DirectSha256Address source_output_trace{};
  DirectSha256Address candidate_output_trace{};
  DirectSha256Address state_transition{};
  DirectSha256Address chronology{};
  DirectSha256Address provenance{};
  DirectSha256Address refusal_semantics{};
  DirectSha256Address transaction_semantics{};
  DirectSha256Address deterministic_replay{};
  DirectSha256Address resource_receipt{};
};
static_assert(std::is_trivially_copyable_v<DirectFoundryEquivalenceReceiptV1> &&
              std::is_standard_layout_v<DirectFoundryEquivalenceReceiptV1>);

#if defined(__CUDACC__)
#define DIRECT_FOUNDRY_EQ_HD __host__ __device__
#else
#define DIRECT_FOUNDRY_EQ_HD
#endif

DIRECT_FOUNDRY_EQ_HD inline DirectSha256Address
direct_foundry_equivalence_receipt_address(
    const DirectFoundryEquivalenceReceiptV1& receipt) {
  static constexpr char kDomain[] =
      "0x1-direct-foundry-equivalence-authority-receipt-v1";
  detail::DirectSha256State state{};
  state.update(kDomain, sizeof(kDomain) - 1u);
  direct_foundry_resource_update_u32(&state, receipt.abi_version);
  direct_foundry_resource_update_u32(
      &state, static_cast<std::uint32_t>(receipt.mode));
  const std::uint32_t counts[] = {
      receipt.domain_case_count,
      receipt.evaluated_case_count,
      receipt.exact_match_count,
      receipt.representation_authority,
      receipt.engineering_authority,
      receipt.semantic_authority,
      receipt.experiential_authority,
      receipt.participation_authority,
      receipt.credit_authority,
      receipt.current_adult_authority,
      static_cast<std::uint32_t>(receipt.guard_min_q16),
      static_cast<std::uint32_t>(receipt.guard_max_q16)};
  for (std::uint32_t value : counts)
    direct_foundry_resource_update_u32(&state, value);
  direct_foundry_resource_update_u64(&state, receipt.tolerance_q32);
  direct_foundry_resource_update_u64(&state, receipt.maximum_error_q32);
  const DirectSha256Address* roots[] = {
      &receipt.source_candidate,
      &receipt.candidate,
      &receipt.relation_contract,
      &receipt.complete_domain,
      &receipt.guard,
      &receipt.formal_ports,
      &receipt.source_output_trace,
      &receipt.candidate_output_trace,
      &receipt.state_transition,
      &receipt.chronology,
      &receipt.provenance,
      &receipt.refusal_semantics,
      &receipt.transaction_semantics,
      &receipt.deterministic_replay,
      &receipt.resource_receipt};
  for (const DirectSha256Address* root : roots)
    state.update(root->byte, sizeof(root->byte));
  return state.finish();
}

DIRECT_FOUNDRY_EQ_HD inline bool direct_foundry_equivalence_receipt_valid(
    const DirectFoundryEquivalenceReceiptV1& receipt,
    const DirectFoundryCandidateEntryV1& source,
    const DirectFoundryCandidateEntryV1& candidate,
    const DirectFoundryResourceReceiptV1& resources) {
  using Addressing = DirectFoundryRecipeCandidatePool<1u>;
  if (source.candidate_address !=
          Addressing::candidate_address(source.record) ||
      candidate.candidate_address !=
          Addressing::candidate_address(candidate.record) ||
      source.candidate_address == candidate.candidate_address ||
      candidate.record.parent_candidate != source.candidate_address ||
      receipt.source_candidate != source.candidate_address ||
      receipt.candidate != candidate.candidate_address)
    return false;
  const bool claim_class =
      candidate.record.proof_class ==
          DirectFoundryProofClassV1::logical_equivalence_claim ||
      candidate.record.proof_class ==
          DirectFoundryProofClassV1::physical_efficiency_claim;
  if (!claim_class || source.record.domain != candidate.record.domain ||
      source.record.guard != candidate.record.guard ||
      source.record.formal_ports != candidate.record.formal_ports ||
      source.record.compatible_evaluator !=
          candidate.record.compatible_evaluator ||
      source.record.compatible_species != candidate.record.compatible_species ||
      receipt.relation_contract != candidate.record.proof_claim ||
      receipt.complete_domain != candidate.record.domain ||
      receipt.guard != candidate.record.guard ||
      receipt.formal_ports != candidate.record.formal_ports)
    return false;
  if (receipt.abi_version != 1u || receipt.domain_case_count == 0u ||
      receipt.evaluated_case_count != receipt.domain_case_count ||
      receipt.exact_match_count > receipt.evaluated_case_count ||
      receipt.guard_min_q16 > receipt.guard_max_q16 ||
      receipt.representation_authority != 1u ||
      receipt.engineering_authority != 1u ||
      receipt.semantic_authority != 0u ||
      receipt.experiential_authority != 0u ||
      receipt.participation_authority != 0u ||
      receipt.credit_authority != 0u ||
      receipt.current_adult_authority != 0u)
    return false;
  const bool exact = receipt.mode == DirectFoundryEquivalenceModeV1::exact &&
                     receipt.tolerance_q32 == 0u &&
                     receipt.maximum_error_q32 == 0u &&
                     receipt.exact_match_count == receipt.domain_case_count &&
                     receipt.source_output_trace ==
                         receipt.candidate_output_trace;
  const bool bounded =
      receipt.mode == DirectFoundryEquivalenceModeV1::bounded_tolerance &&
      receipt.tolerance_q32 != 0u &&
      receipt.maximum_error_q32 <= receipt.tolerance_q32;
  if (!exact && !bounded) return false;
  const DirectSha256Address* roots[] = {
      &receipt.relation_contract,     &receipt.complete_domain,
      &receipt.guard,                 &receipt.formal_ports,
      &receipt.source_output_trace,   &receipt.candidate_output_trace,
      &receipt.state_transition,      &receipt.chronology,
      &receipt.provenance,            &receipt.refusal_semantics,
      &receipt.transaction_semantics, &receipt.deterministic_replay,
      &receipt.resource_receipt};
  for (const DirectSha256Address* root : roots)
    if (!direct_foundry_resource_address_nonzero(*root)) return false;
  return direct_foundry_resource_receipt_valid(resources) &&
         receipt.resource_receipt ==
             direct_foundry_resource_receipt_address(resources) &&
         resources.source_candidate == source.candidate_address &&
         resources.candidate == candidate.candidate_address &&
         resources.guard == receipt.guard &&
         resources.evaluator == candidate.record.compatible_evaluator;
}

#undef DIRECT_FOUNDRY_EQ_HD

}  // namespace substrate::direct_network

#endif  // HARDWARE_NATIVE_DIRECT_FOUNDRY_EQUIVALENCE_AUTHORITY_CUH
