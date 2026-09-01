#ifndef HARDWARE_NATIVE_DIRECT_ADULT_ELIGIBILITY_CLAIM_IDENTITY_CUH
#define HARDWARE_NATIVE_DIRECT_ADULT_ELIGIBILITY_CLAIM_IDENTITY_CUH

#include "direct_adult_core.cuh"

namespace substrate::direct_adult_core {

DIRECT_ADULT_HD inline std::uint64_t eligibility_claim_hash(
    const DirectParticipationDescriptor& claim) {
  std::uint64_t h = claim.ticket_id ^ claim.route_incarnation;
  const std::uint32_t words[] = {
      claim.source_node, claim.target_node, claim.route_index,
      claim.context_signature, claim.claim_incarnation,
      static_cast<std::uint32_t>(claim.authority),
      claim.authority_incarnation,
      static_cast<std::uint32_t>(claim.contribution_kind),
      claim.ancestry_depth};
  for (std::uint32_t word : words)
    h ^= static_cast<std::uint64_t>(word) + 0x9e3779b97f4a7c15ULL +
         (h << 6) + (h >> 2);
  h ^= claim.parent_eligibility_ref + 0x9e3779b97f4a7c15ULL +
       (h << 6) + (h >> 2);
  h ^= h >> 30; h *= 0xbf58476d1ce4e5b9ULL;
  h ^= h >> 27; h *= 0x94d049bb133111ebULL;
  return h ^ (h >> 31);
}

DIRECT_ADULT_HD inline bool eligibility_claim_matches(
    const DirectParticipationDescriptor& claim,
    const EligibilityRecord& record) {
  return claim.ticket_id == record.ticket_id &&
         claim.source_node == record.source_node &&
         claim.target_node == record.target_node &&
         claim.route_index == record.route_index &&
         claim.route_incarnation == record.route_incarnation &&
         claim.context_signature == record.context_signature &&
         claim.claim_incarnation == record.claim_incarnation &&
         claim.authority == record.authority &&
         claim.authority_incarnation == record.authority_incarnation &&
         claim.parent_eligibility_ref == record.parent_eligibility_ref &&
         claim.ancestry_depth == record.ancestry_depth;
}

DIRECT_ADULT_HD inline bool eligibility_record_identity_matches(
    const EligibilityRecord& left, const EligibilityRecord& right) {
  DirectParticipationDescriptor claim{};
  claim.ticket_id = left.ticket_id;
  claim.source_node = left.source_node;
  claim.target_node = left.target_node;
  claim.route_index = left.route_index;
  claim.context_signature = left.context_signature;
  claim.claim_incarnation = left.claim_incarnation;
  claim.route_incarnation = left.route_incarnation;
  claim.authority = left.authority;
  claim.authority_incarnation = left.authority_incarnation;
  claim.parent_eligibility_ref = left.parent_eligibility_ref;
  claim.lineage_expiry_tick = left.lineage_expiry_tick;
  claim.ancestry_depth = left.ancestry_depth;
  return eligibility_claim_matches(claim, right);
}

DIRECT_ADULT_HD inline DirectParticipationDescriptor eligibility_record_claim(
    const EligibilityRecord& record) {
  DirectParticipationDescriptor claim{};
  claim.ticket_id = record.ticket_id;
  claim.source_node = record.source_node;
  claim.target_node = record.target_node;
  claim.route_index = record.route_index;
  claim.context_signature = record.context_signature;
  claim.expiry_tick = record.expiry_tick;
  claim.claim_incarnation = record.claim_incarnation;
  claim.route_incarnation = record.route_incarnation;
  claim.authority = record.authority;
  claim.authority_incarnation = record.authority_incarnation;
  claim.contribution_kind = DirectContributionKind::sparse_route;
  claim.parent_eligibility_ref = record.parent_eligibility_ref;
  claim.lineage_expiry_tick = record.lineage_expiry_tick;
  claim.ancestry_depth = record.ancestry_depth;
  return claim;
}

}  // namespace substrate::direct_adult_core

#endif
