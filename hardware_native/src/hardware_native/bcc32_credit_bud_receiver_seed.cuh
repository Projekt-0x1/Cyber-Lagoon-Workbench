#pragma once

// Compact C2-export receiver grammar.
//
// The existing C2 bud has a measured carrier-3 difference at (-2,-2,-2).
// Along that same u3 edge, F reads B3/E3 on the source and C3/R3 on the
// destination.  This seed therefore grows only those two ordinary local
// cofactors beside the existing bud.  It contains no target value, dispatch,
// host callback, or receiver update rule; full F determines whether the C2
// difference reaches the destination.

#include <array>
#include <cstdint>

#include "bcc32_credit_bud_seed.cuh"

namespace substrate::bcc32 {

using CreditBudReceiverSeedHash = std::uint64_t;

inline constexpr std::uint32_t kCreditBudReceiverCofactorShift = 35u;
inline constexpr std::uint32_t kCreditBudReceiverDestinationShift = 37u;
inline constexpr std::uint32_t kCreditBudReceiverLocalStateShift = 39u;
inline constexpr std::uint8_t kCreditBudReceiverGeneMask = 0x3u;
inline constexpr std::size_t kCreditBudReceiverSeedSiteCount = kCreditBudSeedSiteCount + 2u;

inline constexpr DevelopmentalSeedSite kCreditBudReceiverSource{-2, -2, -2, kQ};
inline constexpr DevelopmentalSeedSite kCreditBudReceiverDestination{-3, -3, -3, kQ};

constexpr CreditBudReceiverSeedHash make_credit_bud_receiver_seed_hash(
    CreditBackcarrySeedHash parent, std::uint8_t bud_gene, std::uint8_t cofactor_gene,
    std::uint8_t destination_gene, std::uint8_t local_state_gene = 0u) {
  return make_credit_bud_seed_hash(parent, bud_gene) |
         ((static_cast<CreditBudReceiverSeedHash>(cofactor_gene) & kCreditBudReceiverGeneMask)
          << kCreditBudReceiverCofactorShift) |
         ((static_cast<CreditBudReceiverSeedHash>(destination_gene) &
           kCreditBudReceiverGeneMask)
          << kCreditBudReceiverDestinationShift) |
         ((static_cast<CreditBudReceiverSeedHash>(local_state_gene) &
           kCreditBudReceiverGeneMask)
          << kCreditBudReceiverLocalStateShift);
}

constexpr CreditBudSeedHash credit_bud_receiver_bud_hash(CreditBudReceiverSeedHash hash) {
  return static_cast<CreditBudSeedHash>(hash & ((CreditBudReceiverSeedHash{1} <<
                                                   kCreditBudReceiverCofactorShift) -
                                                  1u));
}

constexpr std::uint8_t credit_bud_receiver_cofactor_gene(CreditBudReceiverSeedHash hash) {
  return static_cast<std::uint8_t>((hash >> kCreditBudReceiverCofactorShift) &
                                   kCreditBudReceiverGeneMask);
}

constexpr std::uint8_t credit_bud_receiver_destination_gene(CreditBudReceiverSeedHash hash) {
  return static_cast<std::uint8_t>((hash >> kCreditBudReceiverDestinationShift) &
                                   kCreditBudReceiverGeneMask);
}

constexpr std::uint8_t credit_bud_receiver_local_state_gene(CreditBudReceiverSeedHash hash) {
  return static_cast<std::uint8_t>((hash >> kCreditBudReceiverLocalStateShift) &
                                   kCreditBudReceiverGeneMask);
}

constexpr SiteWord credit_bud_receiver_cofactor_word(CreditBudReceiverSeedHash hash) {
  SiteWord word = kQ;
  const std::uint8_t gene = credit_bud_receiver_cofactor_gene(hash);
  if ((gene & 0x1u) != 0u) word |= owned_bond_bit(3u);
  if ((gene & 0x2u) != 0u) word |= energy_bit(3u);
  const std::uint8_t local = credit_bud_receiver_local_state_gene(hash);
  if ((local & 0x1u) != 0u) word |= channel_bit(kConformationShift, 3u);
  if ((local & 0x2u) != 0u) word |= channel_bit(kReactiveShift, 3u);
  return word;
}

constexpr SiteWord credit_bud_receiver_destination_word(CreditBudReceiverSeedHash hash) {
  SiteWord word = kQ;
  const std::uint8_t gene = credit_bud_receiver_destination_gene(hash);
  if ((gene & 0x1u) != 0u) word |= channel_bit(kConformationShift, 3u);
  if ((gene & 0x2u) != 0u) word |= channel_bit(kReactiveShift, 3u);
  return word;
}

constexpr std::array<DevelopmentalSeedSite, kCreditBudReceiverSeedSiteCount>
credit_bud_receiver_seed(CreditBudReceiverSeedHash hash) {
  std::array<DevelopmentalSeedSite, kCreditBudReceiverSeedSiteCount> result{};
  const auto bud = credit_bud_seed(credit_bud_receiver_bud_hash(hash));
  for (std::size_t index = 0u; index < bud.size(); ++index) result[index] = bud[index];
  result[bud.size()] = {kCreditBudReceiverSource.x, kCreditBudReceiverSource.y,
                        kCreditBudReceiverSource.z, credit_bud_receiver_cofactor_word(hash)};
  result[bud.size() + 1u] = {kCreditBudReceiverDestination.x,
                              kCreditBudReceiverDestination.y,
                              kCreditBudReceiverDestination.z,
                              credit_bud_receiver_destination_word(hash)};
  return result;
}

// Gate 8 needs only source B3 to move a P3 difference into the destination
// P-3 lane.  Gate 12 can then mix C3/R3 only when the two destination lanes
// differ, so the two complementary one-lane receivers are the only derived
// phase pair worth testing; C3|R3 was the null equal-lane control.
inline constexpr CreditBudReceiverSeedHash kCreditBudReceiverCOnlyHash =
    make_credit_bud_receiver_seed_hash(kCreditBackcarrySeedHash, 0x2u, 0x1u, 0x1u);
inline constexpr CreditBudReceiverSeedHash kCreditBudReceiverCOnlyC2LesionHash =
    make_credit_bud_receiver_seed_hash(kCreditBackcarrySeedHash, 0x0u, 0x1u, 0x1u);
inline constexpr CreditBudReceiverSeedHash kCreditBudReceiverROnlyHash =
    make_credit_bud_receiver_seed_hash(kCreditBackcarrySeedHash, 0x2u, 0x1u, 0x2u);
inline constexpr CreditBudReceiverSeedHash kCreditBudReceiverROnlyC2LesionHash =
    make_credit_bud_receiver_seed_hash(kCreditBackcarrySeedHash, 0x0u, 0x1u, 0x2u);
inline constexpr CreditBudReceiverSeedHash kCreditBudReceiverLocalCOnlyHash =
    make_credit_bud_receiver_seed_hash(kCreditBackcarrySeedHash, 0x2u, 0x1u, 0x0u, 0x1u);
inline constexpr CreditBudReceiverSeedHash kCreditBudReceiverLocalCOnlyC2LesionHash =
    make_credit_bud_receiver_seed_hash(kCreditBackcarrySeedHash, 0x0u, 0x1u, 0x0u, 0x1u);
inline constexpr CreditBudReceiverSeedHash kCreditBudReceiverLocalROnlyHash =
    make_credit_bud_receiver_seed_hash(kCreditBackcarrySeedHash, 0x2u, 0x1u, 0x0u, 0x2u);
inline constexpr CreditBudReceiverSeedHash kCreditBudReceiverLocalROnlyC2LesionHash =
    make_credit_bud_receiver_seed_hash(kCreditBackcarrySeedHash, 0x0u, 0x1u, 0x0u, 0x2u);
inline constexpr CreditBudReceiverSeedHash kCreditBudReceiverLocalRNoBHash =
    make_credit_bud_receiver_seed_hash(kCreditBackcarrySeedHash, 0x2u, 0x0u, 0x0u, 0x2u);
inline constexpr CreditBudReceiverSeedHash kCreditBudReceiverLocalRNoBC2LesionHash =
    make_credit_bud_receiver_seed_hash(kCreditBackcarrySeedHash, 0x0u, 0x0u, 0x0u, 0x2u);

// The local-R3 receiver's eight-arm map puts its C2-specific carrier at the
// adjacent -u3 target.  These are the next derived receiver states: unlike
// the earlier C-only/R-only controls, they retain the proven source R3 bit.
inline constexpr CreditBudReceiverSeedHash kCreditBudReceiverLocalRTargetCHash =
    make_credit_bud_receiver_seed_hash(kCreditBackcarrySeedHash, 0x2u, 0x0u, 0x1u, 0x2u);
inline constexpr CreditBudReceiverSeedHash kCreditBudReceiverLocalRTargetCC2LesionHash =
    make_credit_bud_receiver_seed_hash(kCreditBackcarrySeedHash, 0x0u, 0x0u, 0x1u, 0x2u);
inline constexpr CreditBudReceiverSeedHash kCreditBudReceiverLocalRTargetRHash =
    make_credit_bud_receiver_seed_hash(kCreditBackcarrySeedHash, 0x2u, 0x0u, 0x2u, 0x2u);
inline constexpr CreditBudReceiverSeedHash kCreditBudReceiverLocalRTargetRC2LesionHash =
    make_credit_bud_receiver_seed_hash(kCreditBackcarrySeedHash, 0x0u, 0x0u, 0x2u, 0x2u);

// A source R3 bit is a destination-side role for its own incoming edge.  To
// ask whether it can affect the forward u3 edge, the same source must carry
// B3, the actual forward-edge control.  These two hashes are therefore the
// complete local BR -> target C/R tableaux, not target-only adapters.
inline constexpr CreditBudReceiverSeedHash kCreditBudReceiverLocalBRTargetCHash =
    make_credit_bud_receiver_seed_hash(kCreditBackcarrySeedHash, 0x2u, 0x1u, 0x1u, 0x2u);
inline constexpr CreditBudReceiverSeedHash kCreditBudReceiverLocalBRTargetCC2LesionHash =
    make_credit_bud_receiver_seed_hash(kCreditBackcarrySeedHash, 0x0u, 0x1u, 0x1u, 0x2u);
inline constexpr CreditBudReceiverSeedHash kCreditBudReceiverLocalBRTargetRHash =
    make_credit_bud_receiver_seed_hash(kCreditBackcarrySeedHash, 0x2u, 0x1u, 0x2u, 0x2u);
inline constexpr CreditBudReceiverSeedHash kCreditBudReceiverLocalBRTargetRC2LesionHash =
    make_credit_bud_receiver_seed_hash(kCreditBackcarrySeedHash, 0x0u, 0x1u, 0x2u, 0x2u);

static_assert(credit_bud_receiver_bud_hash(kCreditBudReceiverCOnlyHash) == kCreditBudSeedHash);
static_assert(credit_bud_receiver_cofactor_word(kCreditBudReceiverCOnlyHash) ==
              (kQ | owned_bond_bit(3u)));
static_assert(credit_bud_receiver_destination_word(kCreditBudReceiverCOnlyHash) ==
              (kQ | channel_bit(kConformationShift, 3u)));
static_assert(credit_bud_receiver_destination_word(kCreditBudReceiverROnlyHash) ==
              (kQ | channel_bit(kReactiveShift, 3u)));
static_assert(credit_bud_receiver_cofactor_word(kCreditBudReceiverLocalCOnlyHash) ==
              (kQ | owned_bond_bit(3u) | channel_bit(kConformationShift, 3u)));
static_assert(credit_bud_receiver_cofactor_word(kCreditBudReceiverLocalRNoBHash) ==
              (kQ | channel_bit(kReactiveShift, 3u)));
static_assert(credit_bud_receiver_destination_word(kCreditBudReceiverLocalRTargetCHash) ==
              (kQ | channel_bit(kConformationShift, 3u)));
static_assert(credit_bud_receiver_destination_word(kCreditBudReceiverLocalRTargetRHash) ==
              (kQ | channel_bit(kReactiveShift, 3u)));
static_assert(credit_bud_receiver_cofactor_word(kCreditBudReceiverLocalBRTargetCHash) ==
              (kQ | owned_bond_bit(3u) | channel_bit(kReactiveShift, 3u)));

}  // namespace substrate::bcc32
