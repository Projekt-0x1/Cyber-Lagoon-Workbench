#ifndef HARDWARE_NATIVE_DIRECT_ADULT_PARTNER_RECIPROCITY_CUH
#define HARDWARE_NATIVE_DIRECT_ADULT_PARTNER_RECIPROCITY_CUH

// h.social_reciprocity_and_rank (#1612). Partner standing grows only from
// settled interaction outcomes under device-owned participation identities.
// Contingent trust gates on reputation; resource allocation follows
// deterministic rank; falling out of trust is a counted sanction. The ledger
// never creates participation and never fabricates settlement evidence.

#include <cstdint>

#include "hardware_native/direct_adult_q16.cuh"

namespace substrate::direct_network {

inline constexpr std::uint32_t kReciprocityPartnerCapacity = 16u;
inline constexpr std::int32_t kTrustThresholdQ16 = direct_adult_core::kQ16One / 4;
inline constexpr std::int32_t kReputationStepQ16 = direct_adult_core::kQ16One / 8;

struct DirectPartnerStanding {
  std::uint64_t partner_identity;
  std::uint32_t cooperative_interactions;
  std::uint32_t defecting_interactions;
  std::int32_t reputation_q16;
};

struct DirectReciprocityLedger {
  DirectPartnerStanding partners[kReciprocityPartnerCapacity] = {};
  std::uint32_t partner_count;
  std::uint32_t sanctions;
  std::uint64_t ledger_identity;
};

static_assert(std::is_trivially_copyable_v<DirectReciprocityLedger>);

__host__ __device__ inline std::uint64_t reciprocity_fold(
    std::uint64_t hash, std::uint64_t value) {
  return (hash ^ (value + 0x9e3779b97f4a7c15ULL + (hash << 6u) +
                  (hash >> 2u))) *
         1099511628211ULL;
}

__host__ __device__ inline std::int32_t reciprocity_find_partner(
    const DirectReciprocityLedger& ledger, std::uint64_t partner_identity) {
  for (std::uint32_t i = 0u; i < ledger.partner_count; ++i)
    if (ledger.partners[i].partner_identity == partner_identity)
      return static_cast<std::int32_t>(i);
  return -1;
}

// One settled interaction with one partner. Positive rewards are cooperation,
// anything else a defection; reputation steps saturating in [0, 1]. Crossing
// out of trust is a counted sanction; the first sighting of a partner mints
// neutral standing.
__host__ __device__ inline bool observe_partner_outcome(
    DirectReciprocityLedger* ledger, std::uint64_t partner_identity,
    std::int32_t settled_reward_q16) {
  if (ledger == nullptr || partner_identity == 0u) return false;
  std::int32_t index = reciprocity_find_partner(*ledger, partner_identity);
  if (index < 0) {
    if (ledger->partner_count >= kReciprocityPartnerCapacity) return false;
    index = static_cast<std::int32_t>(ledger->partner_count);
    DirectPartnerStanding fresh{};
    fresh.partner_identity = partner_identity;
    fresh.reputation_q16 = direct_adult_core::kQ16One / 2;
    ledger->partners[index] = fresh;
    ++ledger->partner_count;
  }
  DirectPartnerStanding& standing = ledger->partners[index];
  const bool was_trusted = standing.reputation_q16 >= kTrustThresholdQ16;
  if (settled_reward_q16 > 0) {
    ++standing.cooperative_interactions;
    standing.reputation_q16 =
        direct_adult_core::clamp_q16(standing.reputation_q16 +
                                         kReputationStepQ16,
                                     0, direct_adult_core::kQ16One);
  } else {
    ++standing.defecting_interactions;
    standing.reputation_q16 =
        direct_adult_core::clamp_q16(standing.reputation_q16 -
                                         kReputationStepQ16,
                                     0, direct_adult_core::kQ16One);
    if (was_trusted && standing.reputation_q16 < kTrustThresholdQ16)
      ++ledger->sanctions;
  }
  ledger->ledger_identity =
      reciprocity_fold(reciprocity_fold(ledger->ledger_identity,
                                        partner_identity),
                       static_cast<std::uint64_t>(settled_reward_q16));
  return true;
}

// Contingent trust: partners earn the gate through lived reputation alone.
__host__ __device__ inline bool partner_trusted(
    const DirectReciprocityLedger& ledger, std::uint64_t partner_identity) {
  const std::int32_t index = reciprocity_find_partner(ledger, partner_identity);
  if (index < 0) return false;
  const DirectPartnerStanding& standing = ledger.partners[index];
  return standing.cooperative_interactions + standing.defecting_interactions >
             0u &&
         standing.reputation_q16 >= kTrustThresholdQ16;
}

// Status-dependent allocation between two known partners: higher reputation
// wins, then more cooperation, then the lower identity as deterministic
// final tie-break. Unknown or absent evidence fails closed.
__host__ __device__ inline bool allocate_by_rank(
    const DirectReciprocityLedger& ledger, std::uint64_t partner_a,
    std::uint64_t partner_b, std::uint64_t* winner_identity) {
  if (winner_identity == nullptr || partner_a == 0u || partner_b == 0u ||
      partner_a == partner_b)
    return false;
  const std::int32_t a = reciprocity_find_partner(ledger, partner_a);
  const std::int32_t b = reciprocity_find_partner(ledger, partner_b);
  if (a < 0 || b < 0) return false;
  const DirectPartnerStanding& sa = ledger.partners[a];
  const DirectPartnerStanding& sb = ledger.partners[b];
  bool a_wins;
  if (sa.reputation_q16 != sb.reputation_q16)
    a_wins = sa.reputation_q16 > sb.reputation_q16;
  else if (sa.cooperative_interactions != sb.cooperative_interactions)
    a_wins = sa.cooperative_interactions > sb.cooperative_interactions;
  else
    a_wins = partner_a < partner_b;
  *winner_identity = a_wins ? partner_a : partner_b;
  return true;
}

}  // namespace substrate::direct_network

#endif
