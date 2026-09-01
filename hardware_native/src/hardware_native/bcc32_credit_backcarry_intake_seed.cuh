#pragma once

// Compact composition of the proven local C2-backcarry synapse with the one
// free intake founder of the flux-powered BCC boundary.  Both components use
// the same generic four B founders; the composition adds only a P3-hole B3/E3
// neighbour.  The hash carries no episode schedule or weight state.

#include <array>
#include <cstdint>

#include "bcc32_credit_backcarry_seed.cuh"

namespace substrate::bcc32 {

using CreditBackcarryIntakeSeedHash = std::uint64_t;
constexpr CreditBackcarryIntakeSeedHash make_credit_backcarry_intake_seed_hash(
    CreditBackcarrySeedHash parent, bool include_intake) {
  return (static_cast<CreditBackcarryIntakeSeedHash>(parent) << 1u) |
         static_cast<CreditBackcarryIntakeSeedHash>(include_intake);
}

constexpr CreditBackcarrySeedHash credit_backcarry_intake_parent_hash(
    CreditBackcarryIntakeSeedHash hash) {
  return static_cast<CreditBackcarrySeedHash>(hash >> 1u);
}

constexpr bool credit_backcarry_intake_enabled(CreditBackcarryIntakeSeedHash hash) {
  return (hash & 1u) != 0u;
}

inline constexpr std::int8_t kCreditBackcarryIntakeX = -1;
inline constexpr std::int8_t kCreditBackcarryIntakeY = -1;
inline constexpr std::int8_t kCreditBackcarryIntakeZ = -1;
inline constexpr SiteWord kCreditBackcarryIntakeWord =
    (kQ & ~carrier_bit(3u)) | owned_bond_bit(3u) | energy_bit(3u);

constexpr std::size_t kCreditBackcarryIntakeSeedSiteCount =
    kCreditBackcarrySeedSiteCount + 1u;

inline std::array<DevelopmentalSeedSite, kCreditBackcarryIntakeSeedSiteCount>
credit_backcarry_intake_seed(CreditBackcarryIntakeSeedHash hash) {
  const CreditBackcarrySeedHash parent = credit_backcarry_intake_parent_hash(hash);
  std::array<DevelopmentalSeedSite, kCreditBackcarryIntakeSeedSiteCount> result{};
  const auto base = credit_backcarry_seed(parent);
  for (std::size_t index = 0u; index < base.size(); ++index) result[index] = base[index];
  result.back() = DevelopmentalSeedSite{
      kCreditBackcarryIntakeX, kCreditBackcarryIntakeY, kCreditBackcarryIntakeZ,
      credit_backcarry_intake_enabled(hash) ? kCreditBackcarryIntakeWord : kQ};
  return result;
}

}  // namespace substrate::bcc32
