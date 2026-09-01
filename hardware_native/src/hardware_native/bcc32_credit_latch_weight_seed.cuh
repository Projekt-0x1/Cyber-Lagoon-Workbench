#pragma once

// Compact local composition: the C2-latch receiver grows around the origin and
// the four-cell native synaptic-weight germ is translated so its basis-2
// consequence founder lies at the latch's measured E2 outlet (0,0,-3).

#include <array>
#include <cstddef>
#include <cstdint>

#include "bcc32_credit_p6_latch_receiver_seed.cuh"
#include "bcc32_synaptic_weight_seed.cuh"

namespace substrate::bcc32 {

using CreditLatchWeightSeedHash = unsigned __int128;

constexpr std::uint32_t kCreditLatchWeightParentBits = 80u;
constexpr std::uint32_t kCreditLatchWeightHashShift = kCreditLatchWeightParentBits;
constexpr CreditLatchWeightSeedHash kCreditLatchWeightParentMask =
    (CreditLatchWeightSeedHash{1u} << kCreditLatchWeightParentBits) - 1u;
constexpr std::size_t kCreditLatchWeightSeedSiteCount =
    kCreditP6LatchReceiverSeedSiteCount + kLocalFounderBasisCount;

constexpr CreditLatchWeightSeedHash make_credit_latch_weight_seed_hash(
    CreditP6LatchReceiverSeedHash latch_hash, SynapticWeightSeedHash weight_hash) {
  return (static_cast<CreditLatchWeightSeedHash>(latch_hash) & kCreditLatchWeightParentMask) |
         (static_cast<CreditLatchWeightSeedHash>(weight_hash) << kCreditLatchWeightHashShift);
}

constexpr CreditP6LatchReceiverSeedHash credit_latch_weight_latch_hash(
    CreditLatchWeightSeedHash hash) {
  return static_cast<CreditP6LatchReceiverSeedHash>(hash & kCreditLatchWeightParentMask);
}

constexpr SynapticWeightSeedHash credit_latch_weight_weight_hash(CreditLatchWeightSeedHash hash) {
  return static_cast<SynapticWeightSeedHash>(hash >> kCreditLatchWeightHashShift);
}

constexpr std::array<DevelopmentalSeedSite, kCreditLatchWeightSeedSiteCount>
credit_latch_weight_seed(CreditLatchWeightSeedHash hash) {
  std::array<DevelopmentalSeedSite, kCreditLatchWeightSeedSiteCount> result{};
  const auto latch = credit_p6_latch_receiver_seed(credit_latch_weight_latch_hash(hash));
  const auto weight = synaptic_weight_seed(credit_latch_weight_weight_hash(hash));
  for (std::size_t index = 0u; index < latch.size(); ++index) result[index] = latch[index];
  for (std::size_t index = 0u; index < weight.size(); ++index) {
    result[latch.size() + index] = {
        static_cast<std::int8_t>(weight[index].x),
        static_cast<std::int8_t>(weight[index].y),
        static_cast<std::int8_t>(weight[index].z - 2), weight[index].word};
  }
  return result;
}

inline constexpr CreditLatchWeightSeedHash kCreditLatchWeightSeedHash =
    make_credit_latch_weight_seed_hash(kCreditP6LatchReceiverSeedHash, kSynapticWeightSeedHash);
inline constexpr CreditLatchWeightSeedHash kCreditLatchWeightC2LesionSeedHash =
    make_credit_latch_weight_seed_hash(kCreditP6LatchReceiverC2LesionSeedHash,
                                       kSynapticWeightSeedHash);
inline constexpr CreditLatchWeightSeedHash kCreditNoLatchWeightSeedHash =
    make_credit_latch_weight_seed_hash(
        make_credit_p6_latch_receiver_seed_hash(kCreditBudReceiverLocalRNoBHash, false),
        kSynapticWeightSeedHash);
inline constexpr CreditLatchWeightSeedHash kCreditNoLatchWeightC2LesionSeedHash =
    make_credit_latch_weight_seed_hash(
        make_credit_p6_latch_receiver_seed_hash(kCreditBudReceiverLocalRNoBC2LesionHash, false),
        kSynapticWeightSeedHash);

}  // namespace substrate::bcc32
