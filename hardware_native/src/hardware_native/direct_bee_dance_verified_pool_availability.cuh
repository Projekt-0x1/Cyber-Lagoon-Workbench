#ifndef HARDWARE_NATIVE_DIRECT_BEE_DANCE_VERIFIED_POOL_AVAILABILITY_CUH
#define HARDWARE_NATIVE_DIRECT_BEE_DANCE_VERIFIED_POOL_AVAILABILITY_CUH

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "hardware_native/direct_bee_dance_capsule_abi.cuh"
#include "hardware_native/direct_foundry_equivalence_authority.cuh"
#include "hardware_native/direct_foundry_global_recipe_optimization.cuh"

namespace substrate::direct_network {

struct DirectBeeDanceAvailabilityEntryV1 {
  std::uint64_t delivery_sequence = 0u;
  DirectBeeDanceVerifiedPoolCapsuleV1 capsule{};
  DirectBeeDanceSourceAttestationV1 attestation{};
  DirectSha256Address capsule_address{};
  DirectSha256Address attestation_address{};
  DirectSha256Address recipient_binding{};
  DirectSha256Address checkpoint_continuation{};
  DirectSha256Address prior_head{};
  DirectSha256Address record_address{};
  DirectSha256Address chain_head{};
};

struct DirectBeeDanceResidentNominationRequestV1 {
  std::uint32_t abi_version = 1u;
  DirectSha256Address recipient_scope{};
  DirectSha256Address candidate{};
  DirectSha256Address resident_context{};
  DirectSha256Address resident_request_receipt{};
  DirectSha256Address guard{};
  DirectSha256Address body_regime{};
  DirectSha256Address evaluator{};
  DirectSha256Address resource_regime{};
  DirectSha256Address checkpoint_continuation{};
};

// A nomination exposes an exact dormant address and the only lawful next
// route. It deliberately contains no executable body or Adult pointer and all
// resident/current/world authority stays zero.
struct DirectBeeDanceDormantNominationV1 {
  std::uint32_t abi_version = 1u;
  DirectBeeDanceAssimilationRouteV1 required_route =
      DirectBeeDanceAssimilationRouteV1::shadow_consequence_marriage_required;
  std::uint32_t dormant_nomination = 1u;
  std::uint32_t activation_authority = 0u;
  std::uint32_t logical_revision_authority = 0u;
  std::uint32_t semantic_authority = 0u;
  std::uint32_t world_truth_authority = 0u;
  std::uint32_t participation_authority = 0u;
  std::uint32_t credit_authority = 0u;
  std::uint32_t current_network_authority = 0u;
  std::uint32_t current_thought_authority = 0u;
  DirectSha256Address candidate{};
  DirectSha256Address capsule{};
  DirectSha256Address recipient_scope{};
  DirectSha256Address resident_context{};
  DirectSha256Address resident_request_receipt{};
  DirectSha256Address checkpoint_continuation{};
};

static_assert(std::is_standard_layout_v<DirectBeeDanceAvailabilityEntryV1> &&
              std::is_trivially_copyable_v<DirectBeeDanceAvailabilityEntryV1>);
static_assert(std::is_standard_layout_v<DirectBeeDanceResidentNominationRequestV1> &&
              std::is_trivially_copyable_v<DirectBeeDanceResidentNominationRequestV1>);
static_assert(std::is_standard_layout_v<DirectBeeDanceDormantNominationV1> &&
              std::is_trivially_copyable_v<DirectBeeDanceDormantNominationV1>);

#if defined(__CUDACC__)
#define DIRECT_BEE_AVAIL_HD __host__ __device__
#else
#define DIRECT_BEE_AVAIL_HD
#endif

DIRECT_BEE_AVAIL_HD inline bool direct_bee_dance_known_route(
    DirectBeeDanceAssimilationRouteV1 route) {
  return route == DirectBeeDanceAssimilationRouteV1::quiescent_equivalent_backing_only ||
         route == DirectBeeDanceAssimilationRouteV1::shadow_consequence_marriage_required;
}

DIRECT_BEE_AVAIL_HD inline bool direct_bee_dance_nomination_authority_valid(
    const DirectBeeDanceDormantNominationV1& nomination) {
  return nomination.abi_version == 1u && nomination.dormant_nomination == 1u &&
         nomination.activation_authority == 0u && nomination.logical_revision_authority == 0u &&
         nomination.semantic_authority == 0u && nomination.world_truth_authority == 0u &&
         nomination.participation_authority == 0u && nomination.credit_authority == 0u &&
         nomination.current_network_authority == 0u && nomination.current_thought_authority == 0u &&
         direct_bee_dance_known_route(nomination.required_route);
}

DIRECT_BEE_AVAIL_HD inline DirectSha256Address direct_bee_dance_availability_record_address(
    std::uint64_t sequence, const DirectSha256Address& capsule,
    const DirectSha256Address& attestation, const DirectSha256Address& binding) {
  static constexpr char kDomain[] = "0x1-bee-dance-availability-record-v1";
  detail::DirectSha256State state{};
  state.update(kDomain, sizeof(kDomain) - 1u);
  direct_bee_dance_update_u64(&state, sequence);
  direct_bee_dance_update_address(&state, capsule);
  direct_bee_dance_update_address(&state, attestation);
  direct_bee_dance_update_address(&state, binding);
  return state.finish();
}

DIRECT_BEE_AVAIL_HD inline DirectSha256Address direct_bee_dance_availability_next_head(
    const DirectSha256Address& prior, const DirectSha256Address& record) {
  static constexpr char kDomain[] = "0x1-bee-dance-availability-chain-v1";
  detail::DirectSha256State state{};
  state.update(kDomain, sizeof(kDomain) - 1u);
  direct_bee_dance_update_address(&state, prior);
  direct_bee_dance_update_address(&state, record);
  return state.finish();
}

DIRECT_BEE_AVAIL_HD inline DirectSha256Address direct_bee_dance_nomination_address(
    const DirectBeeDanceDormantNominationV1& nomination) {
  static constexpr char kDomain[] = "0x1-bee-dance-dormant-nomination-v1";
  detail::DirectSha256State state{};
  state.update(kDomain, sizeof(kDomain) - 1u);
  direct_bee_dance_update_u32(&state, nomination.abi_version);
  direct_bee_dance_update_u32(&state, static_cast<std::uint32_t>(nomination.required_route));
  direct_bee_dance_update_u32(&state, nomination.dormant_nomination);
  direct_bee_dance_update_u32(&state, nomination.activation_authority);
  direct_bee_dance_update_u32(&state, nomination.logical_revision_authority);
  direct_bee_dance_update_u32(&state, nomination.semantic_authority);
  direct_bee_dance_update_u32(&state, nomination.world_truth_authority);
  direct_bee_dance_update_u32(&state, nomination.participation_authority);
  direct_bee_dance_update_u32(&state, nomination.credit_authority);
  direct_bee_dance_update_u32(&state, nomination.current_network_authority);
  direct_bee_dance_update_u32(&state, nomination.current_thought_authority);
  const DirectSha256Address* roots[] = {&nomination.candidate,
                                        &nomination.capsule,
                                        &nomination.recipient_scope,
                                        &nomination.resident_context,
                                        &nomination.resident_request_receipt,
                                        &nomination.checkpoint_continuation};
  for (const auto* root : roots)
    direct_bee_dance_update_address(&state, *root);
  return state.finish();
}

namespace detail {

inline bool direct_bee_dance_optimizer_sources_match(
    const DirectFoundryOptimizationEntryV1& optimization,
    const DirectFoundryCandidateEntryV1& candidate, const DirectFoundryCandidateEntryV1& fallback,
    const DirectFoundryPopulationUsageEntryV1& usage) {
  using CandidateAddressing = DirectFoundryRecipeCandidatePool<1u>;
  using UsageAddressing = DirectFoundryPopulationUsageArchive<1u>;
  using OptimizationAddressing = DirectFoundryGlobalRecipeOptimization<1u>;
  if (candidate.candidate_address != CandidateAddressing::candidate_address(candidate.record) ||
      fallback.candidate_address != CandidateAddressing::candidate_address(fallback.record) ||
      usage.receipt_address != UsageAddressing::receipt_address(usage.receipt) ||
      optimization.nomination_address !=
          OptimizationAddressing::nomination_address(optimization.record))
    return false;
  const auto& record = optimization.record;
  const auto& observed = usage.receipt;
  return record.candidate == candidate.candidate_address &&
         record.parent_candidate == candidate.record.parent_candidate &&
         record.fallback_candidate == fallback.candidate_address &&
         record.construction_ancestry == candidate.record.construction_ancestry &&
         record.compatible_species == candidate.record.compatible_species &&
         record.usage_receipt == usage.receipt_address && record.task == observed.task &&
         record.guard == observed.guard && record.guard == candidate.record.guard &&
         record.body_regime == observed.body_regime && record.evaluator == observed.evaluator &&
         record.evaluator == candidate.record.compatible_evaluator &&
         record.resource_regime == observed.resource_regime &&
         record.trial_history == observed.trial_history &&
         record.positive_receipts == observed.positive_receipts &&
         record.negative_receipts == observed.negative_receipts &&
         record.trial_count == observed.trial_count &&
         record.successful_closure_count == observed.successful_closure_count &&
         record.guarded_failure_count == observed.guarded_failure_count &&
         record.exact_causal_participation_count == observed.exact_causal_participation_count &&
         record.redundant_occurrence_count == observed.redundant_occurrence_count &&
         record.cooccurrence_count == observed.cooccurrence_count &&
         record.latency_p95_ns == observed.latency_p95_ns &&
         record.peak_vram_bytes == observed.peak_vram_bytes &&
         record.energy_p95_nj == observed.energy_p95_nj;
}

}  // namespace detail

template <std::size_t PoolCapacity, std::size_t UsageCapacity, std::size_t OptimizationCapacity>
bool direct_bee_dance_verified_delivery_valid(
    const DirectBeeDanceVerifiedPoolCapsuleV1& capsule,
    const DirectBeeDanceSourceAttestationV1& attestation,
    const DirectBeeDanceRecipientBindingV1& binding,
    const DirectFoundryCandidateEntryV1* pool_entries, std::size_t pool_count,
    const DirectFoundryPopulationUsageEntryV1* usage_entries, std::size_t usage_count,
    const DirectFoundryOptimizationEntryV1* optimization_entries, std::size_t optimization_count,
    std::size_t candidate_index, std::size_t fallback_index, std::size_t usage_index,
    std::size_t optimization_index, const DirectFoundryEquivalenceReceiptV1* equivalence,
    const DirectFoundryResourceReceiptV1* engineering_resources) {
  if (!direct_bee_dance_recipient_binding_valid(binding) ||
      !direct_bee_dance_capsule_authority_valid(capsule) ||
      capsule.recipient_scope != binding.recipient_scope || candidate_index >= pool_count ||
      fallback_index >= pool_count || usage_index >= usage_count ||
      optimization_index >= optimization_count ||
      !DirectFoundryRecipeCandidatePool<PoolCapacity>::verify_entries(pool_entries, pool_count,
                                                                      capsule.pool_head) ||
      !DirectFoundryPopulationUsageArchive<UsageCapacity>::verify_entries(
          usage_entries, usage_count, capsule.population_usage_head) ||
      !DirectFoundryGlobalRecipeOptimization<OptimizationCapacity>::verify_entries(
          optimization_entries, optimization_count, capsule.optimization_head))
    return false;

  const auto& candidate = pool_entries[candidate_index];
  const auto& fallback = pool_entries[fallback_index];
  const auto& usage = usage_entries[usage_index];
  const auto& optimization = optimization_entries[optimization_index];
  if (!detail::direct_bee_dance_optimizer_sources_match(optimization, candidate, fallback, usage) ||
      capsule.candidate != candidate.candidate_address ||
      capsule.fallback_candidate != fallback.candidate_address ||
      capsule.candidate_kind != candidate.record.kind ||
      capsule.proof_class != candidate.record.proof_class ||
      capsule.optimization_nomination != optimization.nomination_address ||
      capsule.domain != candidate.record.domain || capsule.guard != candidate.record.guard ||
      capsule.formal_ports != candidate.record.formal_ports ||
      capsule.evaluator != candidate.record.compatible_evaluator ||
      capsule.compatible_species != candidate.record.compatible_species ||
      capsule.body_regime != optimization.record.body_regime ||
      capsule.resource_regime != optimization.record.resource_regime ||
      capsule.source_lineage != candidate.record.construction_ancestry ||
      capsule.contextual_failures != candidate.record.contextual_failures ||
      capsule.compatible_species != binding.compatible_species ||
      capsule.evaluator != binding.evaluator || capsule.body_regime != binding.body_regime ||
      !direct_bee_dance_attestation_valid(attestation, capsule, binding))
    return false;

  const bool engineering =
      capsule.route == DirectBeeDanceAssimilationRouteV1::quiescent_equivalent_backing_only;
  if (engineering) {
    if (equivalence == nullptr || engineering_resources == nullptr ||
        fallback_index == candidate_index ||
        fallback.candidate_address != candidate.record.parent_candidate ||
        capsule.proof_receipt != direct_foundry_equivalence_receipt_address(*equivalence) ||
        capsule.resource_receipt !=
            direct_foundry_resource_receipt_address(*engineering_resources) ||
        !direct_foundry_equivalence_receipt_valid(*equivalence, fallback, candidate,
                                                  *engineering_resources))
      return false;
  } else {
    if (capsule.route != DirectBeeDanceAssimilationRouteV1::shadow_consequence_marriage_required ||
        candidate.record.proof_class != DirectFoundryProofClassV1::developmental_prior_claim ||
        equivalence != nullptr || engineering_resources != nullptr ||
        capsule.proof_receipt != candidate.record.proof_claim ||
        capsule.resource_receipt != candidate.record.resource_receipt)
      return false;
  }
  return true;
}

template <std::size_t Capacity>
class DirectBeeDanceVerifiedPoolAvailabilityCatalog {
 public:
  static_assert(Capacity > 0u);

  explicit DirectBeeDanceVerifiedPoolAvailabilityCatalog(
      const DirectBeeDanceRecipientBindingV1& binding)
      : binding_(binding), binding_address_(direct_bee_dance_recipient_binding_address(binding)) {}

  template <std::size_t PoolCapacity, std::size_t UsageCapacity, std::size_t OptimizationCapacity>
  bool deliver_verified(const DirectBeeDanceVerifiedPoolCapsuleV1& capsule,
                        const DirectBeeDanceSourceAttestationV1& attestation,
                        const DirectFoundryCandidateEntryV1* pool_entries, std::size_t pool_count,
                        const DirectFoundryPopulationUsageEntryV1* usage_entries,
                        std::size_t usage_count,
                        const DirectFoundryOptimizationEntryV1* optimization_entries,
                        std::size_t optimization_count, std::size_t candidate_index,
                        std::size_t fallback_index, std::size_t usage_index,
                        std::size_t optimization_index,
                        const DirectFoundryEquivalenceReceiptV1* equivalence = nullptr,
                        const DirectFoundryResourceReceiptV1* engineering_resources = nullptr) {
    if (size_ == Capacity ||
        !direct_bee_dance_verified_delivery_valid<PoolCapacity, UsageCapacity,
                                                  OptimizationCapacity>(
            capsule, attestation, binding_, pool_entries, pool_count, usage_entries, usage_count,
            optimization_entries, optimization_count, candidate_index, fallback_index, usage_index,
            optimization_index, equivalence, engineering_resources))
      return false;
    for (std::size_t index = 0u; index < size_; ++index)
      if (entries_[index].capsule.candidate == capsule.candidate)
        return false;

    DirectBeeDanceAvailabilityEntryV1 entry{};
    entry.delivery_sequence = size_;
    entry.capsule = capsule;
    entry.attestation = attestation;
    entry.capsule_address = direct_bee_dance_capsule_address(capsule);
    entry.attestation_address = direct_bee_dance_source_attestation_address(attestation);
    entry.recipient_binding = binding_address_;
    entry.checkpoint_continuation = binding_.checkpoint_continuation;
    entry.prior_head = head_;
    entry.record_address = direct_bee_dance_availability_record_address(
        entry.delivery_sequence, entry.capsule_address, entry.attestation_address,
        entry.recipient_binding);
    entry.chain_head =
        direct_bee_dance_availability_next_head(entry.prior_head, entry.record_address);
    entries_[size_++] = entry;
    head_ = entry.chain_head;
    return true;
  }

  DIRECT_BEE_AVAIL_HD std::size_t size() const { return size_; }
  DIRECT_BEE_AVAIL_HD static constexpr std::size_t capacity() { return Capacity; }
  DIRECT_BEE_AVAIL_HD DirectSha256Address head() const { return head_; }
  DIRECT_BEE_AVAIL_HD DirectBeeDanceRecipientBindingV1 binding() const { return binding_; }
  DIRECT_BEE_AVAIL_HD DirectSha256Address binding_address() const { return binding_address_; }

  DIRECT_BEE_AVAIL_HD bool read(std::size_t index, DirectBeeDanceAvailabilityEntryV1* out) const {
    if (out == nullptr || index >= size_)
      return false;
    *out = entries_[index];
    return true;
  }

  // Exact-address lookup only. Provider, popularity, delivery chronology and
  // host tickets cannot participate in candidate nomination.
  DIRECT_BEE_AVAIL_HD bool nominate_exact(const DirectBeeDanceResidentNominationRequestV1& request,
                                          DirectBeeDanceDormantNominationV1* out) const {
    if (out == nullptr || request.abi_version != 1u ||
        request.recipient_scope != binding_.recipient_scope ||
        request.checkpoint_continuation != binding_.checkpoint_continuation ||
        !direct_bee_dance_address_nonzero(request.resident_context) ||
        !direct_bee_dance_address_nonzero(request.resident_request_receipt))
      return false;
    for (std::size_t index = 0u; index < size_; ++index) {
      const auto& entry = entries_[index];
      if (entry.capsule.candidate != request.candidate)
        continue;
      if (entry.capsule.guard != request.guard ||
          entry.capsule.body_regime != request.body_regime ||
          entry.capsule.evaluator != request.evaluator ||
          entry.capsule.resource_regime != request.resource_regime)
        return false;
      DirectBeeDanceDormantNominationV1 nomination{};
      nomination.required_route = entry.capsule.route;
      nomination.candidate = entry.capsule.candidate;
      nomination.capsule = entry.capsule_address;
      nomination.recipient_scope = request.recipient_scope;
      nomination.resident_context = request.resident_context;
      nomination.resident_request_receipt = request.resident_request_receipt;
      nomination.checkpoint_continuation = request.checkpoint_continuation;
      if (!direct_bee_dance_nomination_authority_valid(nomination))
        return false;
      *out = nomination;
      return true;
    }
    return false;
  }

  DIRECT_BEE_AVAIL_HD bool verify() const {
    if (!direct_bee_dance_recipient_binding_valid(binding_) ||
        binding_address_ != direct_bee_dance_recipient_binding_address(binding_) ||
        size_ > Capacity)
      return false;
    DirectSha256Address prior{};
    for (std::size_t index = 0u; index < size_; ++index) {
      const auto& entry = entries_[index];
      if (entry.delivery_sequence != index ||
          !direct_bee_dance_capsule_authority_valid(entry.capsule) ||
          !direct_bee_dance_attestation_valid(entry.attestation, entry.capsule, binding_) ||
          entry.capsule_address != direct_bee_dance_capsule_address(entry.capsule) ||
          entry.attestation_address !=
              direct_bee_dance_source_attestation_address(entry.attestation) ||
          entry.recipient_binding != binding_address_ ||
          entry.checkpoint_continuation != binding_.checkpoint_continuation ||
          entry.prior_head != prior ||
          entry.record_address != direct_bee_dance_availability_record_address(
                                      index, entry.capsule_address, entry.attestation_address,
                                      entry.recipient_binding) ||
          entry.chain_head != direct_bee_dance_availability_next_head(prior, entry.record_address))
        return false;
      for (std::size_t prior_index = 0u; prior_index < index; ++prior_index)
        if (entries_[prior_index].capsule.candidate == entry.capsule.candidate)
          return false;
      prior = entry.chain_head;
    }
    return prior == head_;
  }

 private:
  DirectBeeDanceRecipientBindingV1 binding_{};
  DirectSha256Address binding_address_{};
  DirectBeeDanceAvailabilityEntryV1 entries_[Capacity]{};
  std::size_t size_ = 0u;
  DirectSha256Address head_{};
};

#undef DIRECT_BEE_AVAIL_HD

}  // namespace substrate::direct_network

#endif  // HARDWARE_NATIVE_DIRECT_BEE_DANCE_VERIFIED_POOL_AVAILABILITY_CUH
