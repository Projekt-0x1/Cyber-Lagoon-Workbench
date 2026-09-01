#pragma once

// Compact founder hash for the first C2-specific resident receiver.  The
// parent C2-export germ supplies the causal P6 vacancy; one local C2 latch at
// its measured vacant port converts that transient into C/R dynamics.  The
// hash births only ordinary BCC matter.  F remains the sole executor.

#include <array>
#include <cstddef>
#include <cstdint>

#include "bcc32_credit_bud_receiver_seed.cuh"

namespace substrate::bcc32 {

using CreditP6LatchReceiverSeedHash = unsigned __int128;

constexpr std::uint32_t kCreditP6LatchBaseShift = 0u;
constexpr std::uint32_t kCreditP6LatchGeneShift = 64u;
constexpr CreditP6LatchReceiverSeedHash kCreditP6LatchGeneMask = 0x1u;
constexpr std::size_t kCreditP6LatchReceiverSeedSiteCount =
    kCreditBudReceiverSeedSiteCount + 1u;

constexpr CreditP6LatchReceiverSeedHash make_credit_p6_latch_receiver_seed_hash(
    CreditBudReceiverSeedHash base_hash, bool c2_latch) {
  return (static_cast<CreditP6LatchReceiverSeedHash>(base_hash) << kCreditP6LatchBaseShift) |
         ((static_cast<CreditP6LatchReceiverSeedHash>(c2_latch ? 1u : 0u) &
           kCreditP6LatchGeneMask) << kCreditP6LatchGeneShift);
}

constexpr CreditBudReceiverSeedHash credit_p6_latch_receiver_base_hash(
    CreditP6LatchReceiverSeedHash hash) {
  return static_cast<CreditBudReceiverSeedHash>(hash >> kCreditP6LatchBaseShift);
}

constexpr bool credit_p6_latch_receiver_has_c2_latch(CreditP6LatchReceiverSeedHash hash) {
  return ((hash >> kCreditP6LatchGeneShift) & kCreditP6LatchGeneMask) != 0u;
}

constexpr std::array<DevelopmentalSeedSite, kCreditP6LatchReceiverSeedSiteCount>
credit_p6_latch_receiver_seed(CreditP6LatchReceiverSeedHash hash) {
  std::array<DevelopmentalSeedSite, kCreditP6LatchReceiverSeedSiteCount> result{};
  const auto base = credit_bud_receiver_seed(credit_p6_latch_receiver_base_hash(hash));
  for (std::size_t index = 0u; index < base.size(); ++index) result[index] = base[index];
  result[base.size()] = {0, 0, -2,
      credit_p6_latch_receiver_has_c2_latch(hash)
          ? static_cast<SiteWord>(kQ | channel_bit(kConformationShift, 2u))
          : kQ};
  return result;
}

inline constexpr CreditP6LatchReceiverSeedHash kCreditP6LatchReceiverSeedHash =
    make_credit_p6_latch_receiver_seed_hash(kCreditBudReceiverLocalRNoBHash, true);
inline constexpr CreditP6LatchReceiverSeedHash kCreditP6LatchReceiverC2LesionSeedHash =
    make_credit_p6_latch_receiver_seed_hash(kCreditBudReceiverLocalRNoBC2LesionHash, true);

}  // namespace substrate::bcc32
