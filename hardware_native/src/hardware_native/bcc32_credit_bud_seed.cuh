#pragma once

// Compact developmental germ for one dormant, local successor contact.
//
// This is deliberately a *candidate*, not a claim that BCC already allocates
// new contacts.  One 64-bit seed places the measured delayed-credit germ and
// one identical latent germ two B3 steps downstream.  The only proposed
// developmental interaction is local: credit in the parent plus a B3 probe
// may, under F, change the dormant successor.  No host callback may activate
// the bud or select a result; the probe only reads ordinary BCC matter.

#include <array>
#include <cstdint>

#include "bcc32_credit_backcarry_seed.cuh"

namespace substrate::bcc32 {

using CreditBudSeedHash = std::uint64_t;

inline constexpr std::uint32_t kCreditBudParentShift = 0u;
inline constexpr std::uint32_t kCreditBudGeneShift = 32u;
inline constexpr std::uint32_t kCreditBudGeneMask = 0x7u;
inline constexpr std::int32_t kCreditBudB3Reach = 2;
inline constexpr std::array<std::int32_t, 3> kCreditBudHub{{
    kCreditBudB3Reach, kCreditBudB3Reach, kCreditBudB3Reach,
}};
inline constexpr std::size_t kCreditBudSeedSiteCount = 2u * kCreditBackcarrySeedSiteCount;

constexpr CreditBudSeedHash make_credit_bud_seed_hash(CreditBackcarrySeedHash parent,
                                                      std::uint8_t bud_gene) {
  return static_cast<CreditBudSeedHash>(parent) << kCreditBudParentShift |
         (static_cast<CreditBudSeedHash>(bud_gene) & kCreditBudGeneMask)
             << kCreditBudGeneShift;
}

constexpr CreditBackcarrySeedHash credit_bud_parent_hash(CreditBudSeedHash hash) {
  return static_cast<CreditBackcarrySeedHash>(hash >> kCreditBudParentShift);
}

constexpr std::uint8_t credit_bud_gene(CreditBudSeedHash hash) {
  return static_cast<std::uint8_t>((hash >> kCreditBudGeneShift) & kCreditBudGeneMask);
}

constexpr SiteWord credit_bud_receptor_word(CreditBudSeedHash hash) {
  SiteWord word = kQ;
  const std::uint8_t gene = credit_bud_gene(hash);
  if ((gene & 0x1u) != 0u)
    word |= owned_bond_bit(2u);
  if ((gene & 0x2u) != 0u)
    word |= channel_bit(kConformationShift, 2u);
  if ((gene & 0x4u) != 0u)
    word |= channel_bit(kReactiveShift, 2u);
  return word;
}

constexpr std::array<DevelopmentalSeedSite, kCreditBudSeedSiteCount> credit_bud_seed(
    CreditBudSeedHash hash) {
  std::array<DevelopmentalSeedSite, kCreditBudSeedSiteCount> result{};
  const auto parent = credit_backcarry_seed(credit_bud_parent_hash(hash));
  const auto bud_weight = synaptic_weight_seed(credit_backcarry_synapse_hash(
      credit_bud_parent_hash(hash)));
  for (std::size_t index = 0u; index < parent.size(); ++index)
    result[index] = parent[index];

  for (std::size_t index = 0u; index < bud_weight.size(); ++index) {
    const DevelopmentalSeedSite site = bud_weight[index];
    result[parent.size() + index] = {
        static_cast<std::int8_t>(site.x + kCreditBudHub[0]),
        static_cast<std::int8_t>(site.y + kCreditBudHub[1]),
        static_cast<std::int8_t>(site.z + kCreditBudHub[2]),
        site.word,
    };
  }
  result.back() = {
      static_cast<std::int8_t>(kCreditBudHub[0]),
      static_cast<std::int8_t>(kCreditBudHub[1]),
      static_cast<std::int8_t>(kCreditBudHub[2] + 1),
      credit_bud_receptor_word(hash),
  };
  return result;
}

constexpr DevelopmentalSeedSite credit_bud_receptor_site(CreditBudSeedHash hash) {
  return credit_bud_seed(hash).back();
}

inline constexpr CreditBudSeedHash kCreditBudSeedHash =
    make_credit_bud_seed_hash(kCreditBackcarrySeedHash, 0x2u);

static_assert(kCreditBudSeedHash == 0x0000000210c10255ull);
static_assert(credit_bud_parent_hash(kCreditBudSeedHash) == kCreditBackcarrySeedHash);
static_assert(credit_bud_gene(kCreditBudSeedHash) == 0x2u);
static_assert(credit_bud_receptor_site(kCreditBudSeedHash).x == 2 &&
              credit_bud_receptor_site(kCreditBudSeedHash).y == 2 &&
              credit_bud_receptor_site(kCreditBudSeedHash).z == 3);

// A derived composition of two already measured local blocks, not a new
// palette search.  Rotate the B0/E0 two-hop relay onto B3: its source is the
// parent contact's existing +u3 founder, its scaffold sits at 2*u3, and its
// ordinary target is 3*u3.  C2 is the only extra local receptor proposed to
// couple parent credit into that known relay.
using CreditBudRelaySeedHash = std::uint64_t;
inline constexpr std::array<std::int32_t, 3> kCreditBudRelayCenter{{2, 2, 2}};
inline constexpr std::array<std::int32_t, 3> kCreditBudRelayTarget{{3, 3, 3}};
inline constexpr std::size_t kCreditBudRelaySeedSiteCount = kCreditBackcarrySeedSiteCount + 1u;

constexpr CreditBudRelaySeedHash make_credit_bud_relay_seed_hash(
    CreditBackcarrySeedHash parent, std::uint8_t receptor_gene) {
  return make_credit_bud_seed_hash(parent, receptor_gene);
}

constexpr std::array<DevelopmentalSeedSite, kCreditBudRelaySeedSiteCount>
credit_bud_relay_seed(CreditBudRelaySeedHash hash) {
  std::array<DevelopmentalSeedSite, kCreditBudRelaySeedSiteCount> result{};
  const auto parent = credit_backcarry_seed(credit_bud_parent_hash(hash));
  for (std::size_t index = 0u; index < parent.size(); ++index)
    result[index] = parent[index];
  result.back() = {
      static_cast<std::int8_t>(kCreditBudRelayCenter[0]),
      static_cast<std::int8_t>(kCreditBudRelayCenter[1]),
      static_cast<std::int8_t>(kCreditBudRelayCenter[2]),
      static_cast<SiteWord>(kQ | owned_bond_bit(3u) | credit_bud_receptor_word(hash)),
  };
  return result;
}

inline constexpr CreditBudRelaySeedHash kCreditBudRelaySeedHash =
    make_credit_bud_relay_seed_hash(kCreditBackcarrySeedHash, 0x2u);

static_assert(kCreditBudRelaySeedHash == kCreditBudSeedHash);
static_assert(credit_bud_relay_seed(kCreditBudRelaySeedHash).back().word ==
              (kQ | owned_bond_bit(3u) | channel_bit(kConformationShift, 2u)));

}  // namespace substrate::bcc32
