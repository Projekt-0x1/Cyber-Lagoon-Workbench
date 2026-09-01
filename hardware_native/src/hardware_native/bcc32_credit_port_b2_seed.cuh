#pragma once

// Minimal material extension of the measured C2-exclusive credit-port seed.
// The sole added founder is B2 at the vacant port (0,0,-2), directly below
// the existing upstream C2/E2 terrain.  This isolates the native same-basis
// bond/conformation exchange before any synaptic-weight founders are added.

#include <array>
#include <cstddef>
#include <cstdint>

#include "bcc32_credit_bud_receiver_seed.cuh"

namespace substrate::bcc32 {

using CreditPortB2SeedHash = unsigned __int128;
constexpr std::uint32_t kCreditPortB2ParentBits = 64u;
constexpr CreditPortB2SeedHash kCreditPortB2ParentMask =
    (CreditPortB2SeedHash{1u} << kCreditPortB2ParentBits) - 1u;
constexpr std::uint32_t kCreditPortB2GeneShift = 64u;
constexpr std::size_t kCreditPortB2SeedSiteCount = kCreditBudReceiverSeedSiteCount + 1u;

constexpr CreditPortB2SeedHash make_credit_port_b2_seed_hash(CreditBudReceiverSeedHash parent,
                                                              bool source_b2) {
  return (static_cast<CreditPortB2SeedHash>(parent) & kCreditPortB2ParentMask) |
         (static_cast<CreditPortB2SeedHash>(source_b2 ? 1u : 0u) << kCreditPortB2GeneShift);
}

constexpr CreditBudReceiverSeedHash credit_port_b2_parent_hash(CreditPortB2SeedHash hash) {
  return static_cast<CreditBudReceiverSeedHash>(hash & kCreditPortB2ParentMask);
}

constexpr bool credit_port_b2_has_source(CreditPortB2SeedHash hash) {
  return ((hash >> kCreditPortB2GeneShift) & 1u) != 0u;
}

constexpr std::array<DevelopmentalSeedSite, kCreditPortB2SeedSiteCount>
credit_port_b2_seed(CreditPortB2SeedHash hash) {
  std::array<DevelopmentalSeedSite, kCreditPortB2SeedSiteCount> result{};
  const auto parent = credit_bud_receiver_seed(credit_port_b2_parent_hash(hash));
  for (std::size_t index = 0u; index < parent.size(); ++index) result[index] = parent[index];
  result.back() = {0, 0, -2,
                   credit_port_b2_has_source(hash) ? static_cast<SiteWord>(kQ | owned_bond_bit(2u))
                                                   : kQ};
  return result;
}

inline constexpr CreditPortB2SeedHash kCreditPortB2SeedHash =
    make_credit_port_b2_seed_hash(kCreditBudReceiverLocalRNoBHash, true);
inline constexpr CreditPortB2SeedHash kCreditPortB2CutSeedHash =
    make_credit_port_b2_seed_hash(kCreditBudReceiverLocalRNoBHash, false);
inline constexpr CreditPortB2SeedHash kCreditPortB2C2LesionSeedHash =
    make_credit_port_b2_seed_hash(kCreditBudReceiverLocalRNoBC2LesionHash, true);
inline constexpr CreditPortB2SeedHash kCreditPortB2C2LesionCutSeedHash =
    make_credit_port_b2_seed_hash(kCreditBudReceiverLocalRNoBC2LesionHash, false);

// Ablate only the inherited B2/B3 founders of the credit parent.  B0/B1
// remain, because they are the measured pre/post entry points; the existing
// C2 receptor, bud, cofactor, destination, and local-state genes are copied
// verbatim from the measured receiver rather than re-authored.
inline constexpr CreditBackcarrySeedHash kCreditPortB2ScaffoldBackcarryHash =
    make_credit_backcarry_seed_hash(static_cast<SynapticWeightSeedHash>(0x05u), 0x2u, 0, 0, 1);
inline constexpr CreditBudReceiverSeedHash kCreditPortB2ScaffoldParentHash =
    make_credit_bud_receiver_seed_hash(
        kCreditPortB2ScaffoldBackcarryHash,
        credit_bud_gene(credit_bud_receiver_bud_hash(kCreditBudReceiverLocalRNoBHash)),
        credit_bud_receiver_cofactor_gene(kCreditBudReceiverLocalRNoBHash),
        credit_bud_receiver_destination_gene(kCreditBudReceiverLocalRNoBHash),
        credit_bud_receiver_local_state_gene(kCreditBudReceiverLocalRNoBHash));
inline constexpr CreditBudReceiverSeedHash kCreditPortB2ScaffoldC2LesionParentHash =
    make_credit_bud_receiver_seed_hash(
        kCreditPortB2ScaffoldBackcarryHash,
        credit_bud_gene(credit_bud_receiver_bud_hash(kCreditBudReceiverLocalRNoBHash)),
        credit_bud_receiver_cofactor_gene(kCreditBudReceiverLocalRNoBHash),
        credit_bud_receiver_destination_gene(kCreditBudReceiverLocalRNoBHash),
        0u);
inline constexpr CreditPortB2SeedHash kCreditPortB2ScaffoldSeedHash =
    make_credit_port_b2_seed_hash(kCreditPortB2ScaffoldParentHash, true);
inline constexpr CreditPortB2SeedHash kCreditPortB2ScaffoldCutSeedHash =
    make_credit_port_b2_seed_hash(kCreditPortB2ScaffoldParentHash, false);
inline constexpr CreditPortB2SeedHash kCreditPortB2ScaffoldC2LesionSeedHash =
    make_credit_port_b2_seed_hash(kCreditPortB2ScaffoldC2LesionParentHash, true);
inline constexpr CreditPortB2SeedHash kCreditPortB2ScaffoldC2LesionCutSeedHash =
    make_credit_port_b2_seed_hash(kCreditPortB2ScaffoldC2LesionParentHash, false);

}  // namespace substrate::bcc32
