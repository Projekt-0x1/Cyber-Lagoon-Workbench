#ifndef HARDWARE_NATIVE_DIRECT_FOUNDRY_EQUIVALENCE_COMMIT_CUH
#define HARDWARE_NATIVE_DIRECT_FOUNDRY_EQUIVALENCE_COMMIT_CUH

#include <cstdint>
#include <limits>
#include <type_traits>

#include "hardware_native/direct_foundry_equivalence_authority.cuh"
#include "hardware_native/direct_network_brain.cuh"
#include "hardware_native/direct_adult_recipe_ir.cuh"

namespace substrate::direct_adult_core {

// Versioned owner for one existing Adult's replaceable physical Recipe
// backing. Logical Recipe/revision identity and every occurrence/evidence
// root are independent of the selected program. The old program remains the
// deterministic out-of-guard fallback.
struct ResidentRecipeIrBackingOwnerV1 {
  std::uint64_t logical_recipe_id;
  std::uint64_t revision_identity;
  ResidentRecipeIrProgram program;
  ResidentRecipeIrProgram fallback_program;
  direct_network::DirectSha256Address backing_candidate;
  direct_network::DirectSha256Address fallback_candidate;
  direct_network::DirectSha256Address guard;
  direct_network::DirectSha256Address occurrence_binding_state;
  direct_network::DirectSha256Address causal_chronology;
  direct_network::DirectSha256Address evidence_state;
  direct_network::DirectSha256Address participation_state;
  direct_network::DirectSha256Address credit_state;
  direct_network::DirectSha256Address checkpoint_continuation;
  std::uint64_t backing_epoch;
  std::int32_t guard_min_q16;
  std::int32_t guard_max_q16;
  std::uint32_t current_occurrence_count;
  std::uint32_t reserved;
};
static_assert(std::is_trivially_copyable_v<ResidentRecipeIrBackingOwnerV1> &&
              std::is_standard_layout_v<ResidentRecipeIrBackingOwnerV1>);

enum class DirectFoundryEquivalentCommitRefusalV1 : std::uint32_t {
  none = 0u,
  null_argument = 1u,
  invalid_owner = 2u,
  active_boundary = 3u,
  stale_transaction = 4u,
  invalid_proof = 5u,
  invalid_program = 6u,
  epoch_exhausted = 7u,
};

// Actual device counters, not a host assertion, must all be quiet. Existing
// dormant/current Occurrences may remain present; only concurrent mutation is
// fenced while the backing pointer/body changes.
struct DirectFoundryQuiescentFenceV1 {
  std::uint32_t ingress_pending = 0u;
  std::uint32_t consequence_pending = 0u;
  std::uint32_t executor_inflight = 0u;
  std::uint32_t structural_transaction_inflight = 0u;
};

struct DirectFoundryEquivalentBackingTransactionV1 {
  std::uint64_t expected_logical_recipe_id = 0u;
  std::uint64_t expected_revision_identity = 0u;
  std::uint64_t expected_backing_epoch = 0u;
  direct_network::DirectSha256Address expected_preservation_root{};
  direct_network::DirectSha256Address proof_receipt{};
  ResidentRecipeIrProgram candidate_program{};
  DirectFoundryQuiescentFenceV1 quiescence{};
};

struct DirectFoundryEquivalentBackingCommitReceiptV1 {
  std::uint32_t abi_version = 1u;
  std::uint32_t representation_authority = 1u;
  std::uint32_t engineering_authority = 1u;
  std::uint32_t semantic_authority = 0u;
  std::uint32_t experiential_authority = 0u;
  std::uint32_t participation_authority = 0u;
  std::uint32_t credit_authority = 0u;
  std::uint32_t current_network_authority = 0u;
  std::uint32_t current_thought_authority = 0u;
  std::uint64_t logical_recipe_id = 0u;
  std::uint64_t revision_identity = 0u;
  std::uint64_t prior_program_identity = 0u;
  std::uint64_t candidate_program_identity = 0u;
  std::uint64_t prior_backing_epoch = 0u;
  std::uint64_t committed_backing_epoch = 0u;
  direct_network::DirectSha256Address source_candidate{};
  direct_network::DirectSha256Address candidate{};
  direct_network::DirectSha256Address proof_receipt{};
  direct_network::DirectSha256Address resource_receipt{};
  direct_network::DirectSha256Address preservation_root{};
  direct_network::DirectSha256Address checkpoint_continuation{};
};
static_assert(
    std::is_trivially_copyable_v<DirectFoundryEquivalentBackingCommitReceiptV1> &&
    std::is_standard_layout_v<DirectFoundryEquivalentBackingCommitReceiptV1>);

#if defined(__CUDACC__)
#define DIRECT_FOUNDRY_COMMIT_HD __host__ __device__
#else
#define DIRECT_FOUNDRY_COMMIT_HD
#endif

DIRECT_FOUNDRY_COMMIT_HD inline direct_network::DirectSha256Address
resident_recipe_ir_program_address(const ResidentRecipeIrProgram& program) {
  direct_network::DirectSha256Address address{};
  (void)direct_network::direct_sha256_content_address(&program, sizeof(program),
                                                       &address);
  return address;
}

DIRECT_FOUNDRY_COMMIT_HD inline bool resident_recipe_ir_backing_owner_intact(
    const ResidentRecipeIrBackingOwnerV1& owner) {
  using direct_network::direct_foundry_resource_address_nonzero;
  return owner.logical_recipe_id != 0u && owner.revision_identity != 0u &&
         resident_recipe_ir_intact(owner.program) &&
         resident_recipe_ir_intact(owner.fallback_program) &&
         direct_foundry_resource_address_nonzero(owner.backing_candidate) &&
         direct_foundry_resource_address_nonzero(owner.fallback_candidate) &&
         direct_foundry_resource_address_nonzero(owner.guard) &&
         direct_foundry_resource_address_nonzero(
             owner.occurrence_binding_state) &&
         direct_foundry_resource_address_nonzero(owner.causal_chronology) &&
         direct_foundry_resource_address_nonzero(owner.evidence_state) &&
         direct_foundry_resource_address_nonzero(owner.participation_state) &&
         direct_foundry_resource_address_nonzero(owner.credit_state) &&
         direct_foundry_resource_address_nonzero(
             owner.checkpoint_continuation) &&
         owner.guard_min_q16 <= owner.guard_max_q16;
}

// Formal numeric guards select a proved specialization. No semantic name,
// popularity or host-chosen route participates in this decision.
DIRECT_FOUNDRY_COMMIT_HD inline const ResidentRecipeIrProgram*
select_resident_recipe_ir_backing(const ResidentRecipeIrBackingOwnerV1& owner,
                                  std::int32_t input_q16) {
  return input_q16 >= owner.guard_min_q16 && input_q16 <= owner.guard_max_q16
             ? &owner.program
             : &owner.fallback_program;
}

DIRECT_FOUNDRY_COMMIT_HD inline bool direct_foundry_interpret_recipe_ir(
    const ResidentRecipeIrProgram& program,
    const ResidentRecipeIrEvidence& evidence,
    ResidentRecipeIrResult* result) {
  return execute_resident_recipe_ir(program, evidence, result);
}

// This identity deliberately excludes physical program/backing/epoch fields.
// If it changes, the transaction would be changing Adult logical or causal
// state rather than merely its equivalent execution realization.
DIRECT_FOUNDRY_COMMIT_HD inline direct_network::DirectSha256Address
resident_recipe_ir_preservation_root(
    const ResidentRecipeIrBackingOwnerV1& owner) {
  static constexpr char kDomain[] =
      "0x1-direct-adult-recipe-ir-preservation-v1";
  direct_network::detail::DirectSha256State state{};
  state.update(kDomain, sizeof(kDomain) - 1u);
  direct_network::direct_foundry_resource_update_u64(
      &state, owner.logical_recipe_id);
  direct_network::direct_foundry_resource_update_u64(
      &state, owner.revision_identity);
  direct_network::direct_foundry_resource_update_u32(
      &state, static_cast<std::uint32_t>(owner.guard_min_q16));
  direct_network::direct_foundry_resource_update_u32(
      &state, static_cast<std::uint32_t>(owner.guard_max_q16));
  direct_network::direct_foundry_resource_update_u32(
      &state, owner.current_occurrence_count);
  const direct_network::DirectSha256Address* roots[] = {
      &owner.guard,
      &owner.occurrence_binding_state,
      &owner.causal_chronology,
      &owner.evidence_state,
      &owner.participation_state,
      &owner.credit_state,
      &owner.checkpoint_continuation};
  for (const auto* root : roots) state.update(root->byte, sizeof(root->byte));
  return state.finish();
}

DIRECT_FOUNDRY_COMMIT_HD inline bool direct_foundry_quiescent(
    const DirectFoundryQuiescentFenceV1& fence) {
  return fence.ingress_pending == 0u &&
         fence.consequence_pending == 0u &&
         fence.executor_inflight == 0u &&
         fence.structural_transaction_inflight == 0u;
}

DIRECT_FOUNDRY_COMMIT_HD inline DirectFoundryEquivalentCommitRefusalV1
commit_direct_foundry_equivalent_backing(
    ResidentRecipeIrBackingOwnerV1* owner,
    const direct_network::DirectFoundryCandidateEntryV1& source,
    const direct_network::DirectFoundryCandidateEntryV1& candidate,
    const direct_network::DirectFoundryEquivalenceReceiptV1& proof,
    const direct_network::DirectFoundryResourceReceiptV1& resources,
    const DirectFoundryEquivalentBackingTransactionV1& transaction,
    DirectFoundryEquivalentBackingCommitReceiptV1* out) {
  using namespace direct_network;
  if (owner == nullptr || out == nullptr)
    return DirectFoundryEquivalentCommitRefusalV1::null_argument;
  if (!resident_recipe_ir_backing_owner_intact(*owner))
    return DirectFoundryEquivalentCommitRefusalV1::invalid_owner;
  if (!direct_foundry_quiescent(transaction.quiescence))
    return DirectFoundryEquivalentCommitRefusalV1::active_boundary;
  const DirectSha256Address preservation =
      resident_recipe_ir_preservation_root(*owner);
  if (transaction.expected_logical_recipe_id != owner->logical_recipe_id ||
      transaction.expected_revision_identity != owner->revision_identity ||
      transaction.expected_backing_epoch != owner->backing_epoch ||
      transaction.expected_preservation_root != preservation ||
      transaction.proof_receipt !=
          direct_foundry_equivalence_receipt_address(proof) ||
      owner->backing_candidate != source.candidate_address ||
      owner->guard != proof.guard ||
      owner->guard_min_q16 != proof.guard_min_q16 ||
      owner->guard_max_q16 != proof.guard_max_q16)
    return DirectFoundryEquivalentCommitRefusalV1::stale_transaction;
  if (!direct_foundry_equivalence_receipt_valid(proof, source, candidate,
                                                 resources))
    return DirectFoundryEquivalentCommitRefusalV1::invalid_proof;
  if (!resident_recipe_ir_intact(transaction.candidate_program) ||
      source.record.candidate_body !=
          resident_recipe_ir_program_address(owner->program) ||
      candidate.record.candidate_body !=
          resident_recipe_ir_program_address(transaction.candidate_program))
    return DirectFoundryEquivalentCommitRefusalV1::invalid_program;
  if (owner->backing_epoch == std::numeric_limits<std::uint64_t>::max())
    return DirectFoundryEquivalentCommitRefusalV1::epoch_exhausted;

  ResidentRecipeIrBackingOwnerV1 staged = *owner;
  staged.fallback_program = owner->program;
  staged.fallback_candidate = owner->backing_candidate;
  staged.program = transaction.candidate_program;
  staged.backing_candidate = candidate.candidate_address;
  ++staged.backing_epoch;
  if (resident_recipe_ir_preservation_root(staged) != preservation)
    return DirectFoundryEquivalentCommitRefusalV1::invalid_program;

  DirectFoundryEquivalentBackingCommitReceiptV1 receipt{};
  receipt.logical_recipe_id = owner->logical_recipe_id;
  receipt.revision_identity = owner->revision_identity;
  receipt.prior_program_identity = owner->program.program_identity;
  receipt.candidate_program_identity =
      transaction.candidate_program.program_identity;
  receipt.prior_backing_epoch = owner->backing_epoch;
  receipt.committed_backing_epoch = staged.backing_epoch;
  receipt.source_candidate = source.candidate_address;
  receipt.candidate = candidate.candidate_address;
  receipt.proof_receipt = transaction.proof_receipt;
  receipt.resource_receipt =
      direct_foundry_resource_receipt_address(resources);
  receipt.preservation_root = preservation;
  receipt.checkpoint_continuation = owner->checkpoint_continuation;
  *owner = staged;
  *out = receipt;
  return DirectFoundryEquivalentCommitRefusalV1::none;
}

#undef DIRECT_FOUNDRY_COMMIT_HD

}  // namespace substrate::direct_adult_core

#endif  // HARDWARE_NATIVE_DIRECT_FOUNDRY_EQUIVALENCE_COMMIT_CUH
