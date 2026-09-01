#pragma once

// A basis-rotated composition of two already measured local rules.
//
// The credit-port B2 seed supplies a B2 source one -u2 step behind its native
// B2 upstream founder.  The local-area translated interface shows that a
// B/E source on one basis plus E/C material at its neighbour can leave a
// C/R residue on a second basis.  This hash makes only that rotation:
//
//   port P = U - u2 : B2 + optional E2     (incoming source)
//   upstream U         : B2 + optional E0 + optional C0 (local cofactor)
//   proposed output    : U + u0
//
// It names no value, clock, target action, or host callback.  Full reversible
// F is the sole post-genesis interpreter.  The companion contract decides
// whether this composition is a credit-selected exporter or another closed
// parallel scaffold.

#include <array>
#include <cstddef>
#include <cstdint>

#include "bcc32_credit_port_b2_seed.cuh"

namespace substrate::bcc32 {

using CreditPortTranslatorSeedHash = unsigned __int128;

constexpr std::uint32_t kCreditPortTranslatorParentBits = 65u;
constexpr CreditPortTranslatorSeedHash kCreditPortTranslatorParentMask =
    (CreditPortTranslatorSeedHash{1u} << kCreditPortTranslatorParentBits) - 1u;
constexpr std::uint32_t kCreditPortTranslatorGeneShift = kCreditPortTranslatorParentBits;

using CreditPortTranslatorGene = std::uint8_t;
constexpr CreditPortTranslatorGene kCreditPortTranslatorSourceEnergy = 0x1u;
constexpr CreditPortTranslatorGene kCreditPortTranslatorCofactorEnergy = 0x2u;
constexpr CreditPortTranslatorGene kCreditPortTranslatorCofactorConformation = 0x4u;
constexpr CreditPortTranslatorGene kCreditPortTranslatorGeneMask =
    kCreditPortTranslatorSourceEnergy | kCreditPortTranslatorCofactorEnergy |
    kCreditPortTranslatorCofactorConformation;
constexpr CreditPortTranslatorGene kCreditPortTranslatorFullGene =
    kCreditPortTranslatorGeneMask;

constexpr CreditPortTranslatorSeedHash make_credit_port_translator_seed_hash(
    CreditPortB2SeedHash parent, CreditPortTranslatorGene gene) {
  return (static_cast<CreditPortTranslatorSeedHash>(parent) & kCreditPortTranslatorParentMask) |
         ((static_cast<CreditPortTranslatorSeedHash>(gene) & kCreditPortTranslatorGeneMask)
          << kCreditPortTranslatorGeneShift);
}

constexpr CreditPortB2SeedHash credit_port_translator_parent_hash(
    CreditPortTranslatorSeedHash hash) {
  return static_cast<CreditPortB2SeedHash>(hash & kCreditPortTranslatorParentMask);
}

constexpr CreditPortTranslatorGene credit_port_translator_gene(
    CreditPortTranslatorSeedHash hash) {
  return static_cast<CreditPortTranslatorGene>(
      (hash >> kCreditPortTranslatorGeneShift) & kCreditPortTranslatorGeneMask);
}

constexpr bool credit_port_translator_gene_has(CreditPortTranslatorSeedHash hash,
                                                CreditPortTranslatorGene flag) {
  return (credit_port_translator_gene(hash) & flag) != 0u;
}

constexpr bool credit_port_translator_site_is(const DevelopmentalSeedSite& site, std::int8_t x,
                                               std::int8_t y, std::int8_t z) {
  return site.x == x && site.y == y && site.z == z;
}

constexpr std::array<DevelopmentalSeedSite, kCreditPortB2SeedSiteCount + 1u>
credit_port_translator_seed(CreditPortTranslatorSeedHash hash) {
  std::array<DevelopmentalSeedSite, kCreditPortB2SeedSiteCount + 1u> result{};
  const auto parent = credit_port_b2_seed(credit_port_translator_parent_hash(hash));
  for (std::size_t index = 0u; index < parent.size(); ++index) result[index] = parent[index];
  SiteWord cofactor = kQ;
  if (credit_port_translator_gene_has(hash, kCreditPortTranslatorCofactorEnergy))
    cofactor |= energy_bit(0u);
  if (credit_port_translator_gene_has(hash, kCreditPortTranslatorCofactorConformation))
    cofactor |= channel_bit(kConformationShift, 0u);
  bool found_upstream = false;
  for (std::size_t index = 0u; index < parent.size(); ++index) {
    DevelopmentalSeedSite& site = result[index];
    if (credit_port_translator_site_is(site, 0, 0, -2) &&
        credit_port_translator_gene_has(hash, kCreditPortTranslatorSourceEnergy))
      site.word |= energy_bit(2u);
    if (credit_port_translator_site_is(site, 0, 0, -1)) {
      site.word |= cofactor & ~kQ;
      found_upstream = true;
    }
  }
  result.back() = found_upstream ? DevelopmentalSeedSite{0, 0, 0, kQ}
                                  : DevelopmentalSeedSite{0, 0, -1, cofactor};
  return result;
}

constexpr SiteWord credit_port_translator_site_word(CreditPortTranslatorSeedHash hash,
                                                     std::int8_t x, std::int8_t y,
                                                     std::int8_t z) {
  for (const DevelopmentalSeedSite site : credit_port_translator_seed(hash))
    if (credit_port_translator_site_is(site, x, y, z)) return site.word;
  return kQ;
}

inline constexpr CreditPortTranslatorSeedHash kCreditPortTranslatorSeedHash =
    make_credit_port_translator_seed_hash(kCreditPortB2SeedHash,
                                          kCreditPortTranslatorFullGene);
inline constexpr CreditPortTranslatorSeedHash kCreditPortTranslatorCutSeedHash =
    make_credit_port_translator_seed_hash(kCreditPortB2SeedHash, 0u);
inline constexpr CreditPortTranslatorSeedHash kCreditPortTranslatorC2CutSeedHash =
    make_credit_port_translator_seed_hash(kCreditPortB2C2LesionSeedHash,
                                          kCreditPortTranslatorFullGene);
inline constexpr CreditPortTranslatorSeedHash kCreditPortTranslatorB2CutSeedHash =
    make_credit_port_translator_seed_hash(kCreditPortB2CutSeedHash,
                                          kCreditPortTranslatorFullGene);
inline constexpr CreditPortTranslatorSeedHash kCreditPortTranslatorC2B2CutSeedHash =
    make_credit_port_translator_seed_hash(kCreditPortB2C2LesionCutSeedHash,
                                          kCreditPortTranslatorFullGene);

static_assert(credit_port_translator_gene(kCreditPortTranslatorSeedHash) ==
              kCreditPortTranslatorFullGene);
static_assert(credit_port_translator_gene_has(kCreditPortTranslatorSeedHash,
                                              kCreditPortTranslatorSourceEnergy));
static_assert(credit_port_b2_has_source(
    credit_port_translator_parent_hash(kCreditPortTranslatorSeedHash)));
static_assert((credit_port_translator_site_word(kCreditPortTranslatorSeedHash, 0, 0, -2) &
               energy_bit(2u)) != 0u);
static_assert((credit_port_translator_site_word(kCreditPortTranslatorSeedHash, 0, 0, -1) &
               (energy_bit(0u) | channel_bit(kConformationShift, 0u))) ==
              (energy_bit(0u) | channel_bit(kConformationShift, 0u)));

}  // namespace substrate::bcc32
