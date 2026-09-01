#pragma once

// Finite, generic energy reservoir at the proven local C2 factor receptor.
//
// E is a source-side BCC role, so a future E2 factor cannot be delivered by a
// neighbouring cell as an ordinary destination write.  This compact hash tests
// the lawful alternative: retain generic spare E0/E1/E3 matter at the same
// receptor and let F's site dynamics decide whether it can expose another E2
// episode.  It contains no weight state or episode schedule.

#include <array>
#include <cstdint>

#include "bcc32_credit_backcarry_seed.cuh"

namespace substrate::bcc32 {

using CreditBackcarryReceptorReservoirSeedHash = std::uint64_t;
constexpr std::uint8_t kCreditBackcarryReservoirMask = 0x07u;
constexpr std::array<std::uint32_t, 3u> kCreditBackcarryReservoirBases{{0u, 1u, 3u}};

constexpr CreditBackcarryReceptorReservoirSeedHash
make_credit_backcarry_receptor_reservoir_seed_hash(CreditBackcarrySeedHash parent,
                                                   std::uint8_t reservoir_mask) {
  return (static_cast<CreditBackcarryReceptorReservoirSeedHash>(parent) << 3u) |
         static_cast<CreditBackcarryReceptorReservoirSeedHash>(reservoir_mask &
                                                                kCreditBackcarryReservoirMask);
}

constexpr CreditBackcarrySeedHash credit_backcarry_receptor_reservoir_parent_hash(
    CreditBackcarryReceptorReservoirSeedHash hash) {
  return static_cast<CreditBackcarrySeedHash>(hash >> 3u);
}

constexpr std::uint8_t credit_backcarry_receptor_reservoir_mask(
    CreditBackcarryReceptorReservoirSeedHash hash) {
  return static_cast<std::uint8_t>(hash & kCreditBackcarryReservoirMask);
}

inline std::array<DevelopmentalSeedSite, kCreditBackcarrySeedSiteCount>
credit_backcarry_receptor_reservoir_seed(CreditBackcarryReceptorReservoirSeedHash hash) {
  const CreditBackcarrySeedHash parent = credit_backcarry_receptor_reservoir_parent_hash(hash);
  auto result = credit_backcarry_seed(parent);
  SiteWord& receptor = result.back().word;
  const std::uint8_t mask = credit_backcarry_receptor_reservoir_mask(hash);
  for (std::size_t index = 0u; index < kCreditBackcarryReservoirBases.size(); ++index)
    if ((mask & (1u << index)) != 0u) receptor |= energy_bit(kCreditBackcarryReservoirBases[index]);
  return result;
}

}  // namespace substrate::bcc32
