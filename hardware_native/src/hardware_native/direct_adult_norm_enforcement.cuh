#ifndef HARDWARE_NATIVE_DIRECT_ADULT_NORM_ENFORCEMENT_CUH
#define HARDWARE_NATIVE_DIRECT_ADULT_NORM_ENFORCEMENT_CUH

#include <cstdint>
#include <type_traits>

#include "hardware_native/direct_adult_partner_reciprocity.cuh"

namespace substrate::direct_network {

inline constexpr std::uint32_t kNormPenaltyCostQ16 =
    static_cast<std::uint32_t>(direct_adult_core::kQ16One / 16);

struct DirectNormEnforcementState {
  std::uint64_t paid_cost_q16;
  std::uint32_t penalties;
  std::uint32_t refusals;
  std::uint64_t enforcement_identity;
};
static_assert(std::is_trivially_copyable_v<DirectNormEnforcementState>);

__host__ __device__ inline std::uint64_t norm_enforcement_fold(
    std::uint64_t hash, std::uint64_t value) {
  return (hash ^ (value + 0x9e3779b97f4a7c15ULL + (hash << 6u) +
                  (hash >> 2u))) *
         1099511628211ULL;
}

// Costly punishment never manufactures an interaction. It may only penalize a
// known partner whose lived reciprocity record contains defection. Every
// admitted penalty debits the enforcer's resource counter and lowers that
// partner's standing by one bounded reputation step. Crossing out of trust is
// a counted sanction in the shared reciprocity ledger.
__host__ __device__ inline bool enforce_lived_defector(
    DirectReciprocityLedger* ledger, DirectNormEnforcementState* enforcement,
    std::uint64_t partner_identity) {
  if (ledger == nullptr || enforcement == nullptr || partner_identity == 0u) {
    if (enforcement != nullptr) ++enforcement->refusals;
    return false;
  }
  const std::int32_t index = reciprocity_find_partner(*ledger, partner_identity);
  if (index < 0) {
    ++enforcement->refusals;
    return false;
  }
  DirectPartnerStanding& standing = ledger->partners[index];
  if (standing.defecting_interactions == 0u) {
    ++enforcement->refusals;
    return false;
  }
  if (enforcement->paid_cost_q16 >
      0xffffffffffffffffULL - kNormPenaltyCostQ16) {
    ++enforcement->refusals;
    return false;
  }
  const bool was_trusted = standing.reputation_q16 >= kTrustThresholdQ16;
  enforcement->paid_cost_q16 += kNormPenaltyCostQ16;
  ++enforcement->penalties;
  standing.reputation_q16 = direct_adult_core::clamp_q16(
      standing.reputation_q16 - kReputationStepQ16, 0,
      direct_adult_core::kQ16One);
  if (was_trusted && standing.reputation_q16 < kTrustThresholdQ16)
    ++ledger->sanctions;
  enforcement->enforcement_identity = norm_enforcement_fold(
      norm_enforcement_fold(enforcement->enforcement_identity, partner_identity),
      static_cast<std::uint64_t>(standing.reputation_q16));
  ledger->ledger_identity = reciprocity_fold(
      reciprocity_fold(ledger->ledger_identity, partner_identity),
      enforcement->enforcement_identity);
  return true;
}

}  // namespace substrate::direct_network

#endif
