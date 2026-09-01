#pragma once

// Compact developmental germ for one local, delayed credit-bearing synapse.
//
// The low byte selects the existing four-founder weight germ. Three bits
// select physical B2/C2/R2 matter at one nearby founder coordinate; three
// signed six-bit loci place that founder. The hash contains no delay, settle
// count, truth table, weight value, route, target, or host-side update.
// Canonical reversible F remains the only runtime interpreter.

#include <array>
#include <cstdint>

#include "bcc32_synaptic_weight_seed.cuh"

namespace substrate::bcc32 {

using CreditBackcarrySeedHash = std::uint32_t;

inline constexpr std::uint32_t kCreditBackcarryWeightShift = 0u;
inline constexpr std::uint32_t kCreditBackcarryGeneShift = 8u;
inline constexpr std::uint32_t kCreditBackcarryXShift = 11u;
inline constexpr std::uint32_t kCreditBackcarryYShift = 17u;
inline constexpr std::uint32_t kCreditBackcarryZShift = 23u;
inline constexpr std::uint32_t kCreditBackcarryGeneMask = 0x7u;
inline constexpr std::uint32_t kCreditBackcarryCoordinateMask = 0x3fu;
inline constexpr std::int32_t kCreditBackcarryCoordinateBias = 32;

constexpr std::uint32_t encode_credit_backcarry_coordinate(std::int32_t value) {
  return static_cast<std::uint32_t>(value + kCreditBackcarryCoordinateBias) &
         kCreditBackcarryCoordinateMask;
}

constexpr std::int32_t decode_credit_backcarry_coordinate(CreditBackcarrySeedHash hash,
                                                          std::uint32_t shift) {
  return static_cast<std::int32_t>((hash >> shift) & kCreditBackcarryCoordinateMask) -
         kCreditBackcarryCoordinateBias;
}

constexpr CreditBackcarrySeedHash make_credit_backcarry_seed_hash(
    SynapticWeightSeedHash weight_hash, std::uint8_t receptor_gene, std::int32_t receptor_x,
    std::int32_t receptor_y, std::int32_t receptor_z) {
  return static_cast<CreditBackcarrySeedHash>(weight_hash) << kCreditBackcarryWeightShift |
         (static_cast<std::uint32_t>(receptor_gene) & kCreditBackcarryGeneMask)
             << kCreditBackcarryGeneShift |
         encode_credit_backcarry_coordinate(receptor_x) << kCreditBackcarryXShift |
         encode_credit_backcarry_coordinate(receptor_y) << kCreditBackcarryYShift |
         encode_credit_backcarry_coordinate(receptor_z) << kCreditBackcarryZShift;
}

constexpr SynapticWeightSeedHash credit_backcarry_synapse_hash(CreditBackcarrySeedHash hash) {
  return static_cast<SynapticWeightSeedHash>(hash & 0xffu);
}

constexpr std::uint8_t credit_backcarry_receptor_gene(CreditBackcarrySeedHash hash) {
  return static_cast<std::uint8_t>((hash >> kCreditBackcarryGeneShift) & kCreditBackcarryGeneMask);
}

constexpr std::array<std::int32_t, 3> credit_backcarry_receptor_coordinate(
    CreditBackcarrySeedHash hash) {
  return {{
      decode_credit_backcarry_coordinate(hash, kCreditBackcarryXShift),
      decode_credit_backcarry_coordinate(hash, kCreditBackcarryYShift),
      decode_credit_backcarry_coordinate(hash, kCreditBackcarryZShift),
  }};
}

constexpr SiteWord credit_backcarry_receptor_word(CreditBackcarrySeedHash hash) {
  const std::uint8_t gene = credit_backcarry_receptor_gene(hash);
  SiteWord word = kQ;
  if ((gene & 0x1u) != 0u)
    word |= owned_bond_bit(2u);
  if ((gene & 0x2u) != 0u)
    word |= channel_bit(kConformationShift, 2u);
  if ((gene & 0x4u) != 0u)
    word |= channel_bit(kReactiveShift, 2u);
  return word;
}

inline constexpr std::size_t kCreditBackcarrySeedSiteCount = kLocalFounderBasisCount + 1u;

constexpr std::array<DevelopmentalSeedSite, kCreditBackcarrySeedSiteCount> credit_backcarry_seed(
    CreditBackcarrySeedHash hash) {
  std::array<DevelopmentalSeedSite, kCreditBackcarrySeedSiteCount> result{};
  const auto weight = synaptic_weight_seed(credit_backcarry_synapse_hash(hash));
  for (std::size_t index = 0u; index < weight.size(); ++index)
    result[index] = weight[index];

  const auto receptor = credit_backcarry_receptor_coordinate(hash);
  result[kLocalFounderBasisCount] = {
      static_cast<std::int8_t>(receptor[0]), static_cast<std::int8_t>(receptor[1]),
      static_cast<std::int8_t>(receptor[2]), credit_backcarry_receptor_word(hash)};
  return result;
}

constexpr DevelopmentalSeedSite credit_backcarry_receptor_site(CreditBackcarrySeedHash hash) {
  return credit_backcarry_seed(hash)[kLocalFounderBasisCount];
}

inline constexpr CreditBackcarrySeedHash kCreditBackcarrySeedHash =
    make_credit_backcarry_seed_hash(kSynapticWeightSeedHash, 0x2u, 0, 0, 1);

static_assert(kCreditBackcarrySeedHash == 0x10c10255u);
static_assert(credit_backcarry_synapse_hash(kCreditBackcarrySeedHash) == kSynapticWeightSeedHash);
static_assert(credit_backcarry_receptor_gene(kCreditBackcarrySeedHash) == 0x2u);
static_assert(credit_backcarry_receptor_coordinate(kCreditBackcarrySeedHash) ==
              std::array<std::int32_t, 3>{{0, 0, 1}});
static_assert(credit_backcarry_receptor_word(kCreditBackcarrySeedHash) ==
              (kQ | channel_bit(kConformationShift, 2u)));

}  // namespace substrate::bcc32
