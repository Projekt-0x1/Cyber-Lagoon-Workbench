#pragma once

// A compact candidate composition, not a scheduled program.  The parent is
// the measured C2-exclusive credit-port germ.  The ordinary four-founder
// synapse placement gene translates its existing B2 third-factor founder
// along -u2.  The hash therefore names only already-generic founder matter
// and a fixed developmental placement.

#include <array>
#include <cstddef>
#include <cstdint>

#include "bcc32_credit_bud_receiver_seed.cuh"
#include "bcc32_synaptic_weight_seed.cuh"

namespace substrate::bcc32 {

using CreditPortWeightSeedHash = unsigned __int128;

constexpr std::uint32_t kCreditPortWeightParentBits = 64u;
constexpr CreditPortWeightSeedHash kCreditPortWeightParentMask =
    (CreditPortWeightSeedHash{1u} << kCreditPortWeightParentBits) - 1u;
constexpr std::uint32_t kCreditPortWeightSynapseShift = kCreditPortWeightParentBits;
constexpr std::uint32_t kCreditPortWeightPlacementShift = 72u;
constexpr std::uint8_t kCreditPortWeightPlacementMask = 0x3u;
constexpr std::size_t kCreditPortWeightSeedSiteCount =
    kCreditBudReceiverSeedSiteCount + kLocalFounderBasisCount;

constexpr CreditPortWeightSeedHash make_credit_port_weight_seed_hash(
    CreditBudReceiverSeedHash parent, SynapticWeightSeedHash synapse, std::uint8_t placement) {
  return (static_cast<CreditPortWeightSeedHash>(parent) & kCreditPortWeightParentMask) |
         (static_cast<CreditPortWeightSeedHash>(synapse) << kCreditPortWeightSynapseShift) |
         ((static_cast<CreditPortWeightSeedHash>(placement) & kCreditPortWeightPlacementMask)
          << kCreditPortWeightPlacementShift);
}

constexpr CreditBudReceiverSeedHash credit_port_weight_parent_hash(CreditPortWeightSeedHash hash) {
  return static_cast<CreditBudReceiverSeedHash>(hash & kCreditPortWeightParentMask);
}

constexpr SynapticWeightSeedHash credit_port_weight_synapse_hash(CreditPortWeightSeedHash hash) {
  return static_cast<SynapticWeightSeedHash>(hash >> kCreditPortWeightSynapseShift);
}

constexpr std::uint8_t credit_port_weight_placement(CreditPortWeightSeedHash hash) {
  return static_cast<std::uint8_t>((hash >> kCreditPortWeightPlacementShift) &
                                   kCreditPortWeightPlacementMask);
}

constexpr std::array<DevelopmentalSeedSite, kCreditPortWeightSeedSiteCount>
credit_port_weight_seed(CreditPortWeightSeedHash hash) {
  std::array<DevelopmentalSeedSite, kCreditPortWeightSeedSiteCount> result{};
  const auto parent = credit_bud_receiver_seed(credit_port_weight_parent_hash(hash));
  const auto synapse = synaptic_weight_seed(credit_port_weight_synapse_hash(hash));
  const auto placement = credit_port_weight_placement(hash);
  for (std::size_t index = 0u; index < parent.size(); ++index) result[index] = parent[index];
  for (std::size_t index = 0u; index < synapse.size(); ++index) {
    result[parent.size() + index] = {
        synapse[index].x, synapse[index].y, static_cast<std::int8_t>(synapse[index].z - placement),
        synapse[index].word};
  }
  return result;
}

inline constexpr CreditPortWeightSeedHash kCreditPortWeightSeedHash =
    make_credit_port_weight_seed_hash(kCreditBudReceiverLocalRNoBHash,
                                      kSynapticWeightSeedHash, 1u);
inline constexpr CreditPortWeightSeedHash kCreditPortWeightC2LesionSeedHash =
    make_credit_port_weight_seed_hash(kCreditBudReceiverLocalRNoBC2LesionHash,
                                      kSynapticWeightSeedHash, 1u);
inline constexpr CreditPortWeightSeedHash kCreditPortDownstreamWeightSeedHash =
    make_credit_port_weight_seed_hash(kCreditBudReceiverLocalRNoBHash,
                                      kSynapticWeightSeedHash, 2u);
inline constexpr CreditPortWeightSeedHash kCreditPortDownstreamWeightC2LesionSeedHash =
    make_credit_port_weight_seed_hash(kCreditBudReceiverLocalRNoBC2LesionHash,
                                      kSynapticWeightSeedHash, 2u);

}  // namespace substrate::bcc32
