#pragma once

// A compact, generic four-basis local founder grammar.
//
// The word is not a pointer to a host recipe.  Each two-bit gene states which
// of the B/E lanes an incoming BCC founder carries on one of the four physical
// bases.  Birth materializes that tiny local pool once; thereafter the
// unchanged reversible law F is the sole interpreter.  The grammar is generic
// -- it names no language token, region, coordinate label, or output feature.

#include <array>
#include <cstdint>

#include "bcc32_developmental_seed.cuh"
#include "bcc32_law.cuh"

namespace substrate::bcc32 {

using LocalFounderHash = std::uint8_t;
constexpr std::size_t kLocalFounderBasisCount = 4u;

enum class LocalFounderGene : std::uint8_t {
  quiescent = 0u,
  bond = 1u,
  energy = 2u,
  bond_energy = 3u,
};

constexpr LocalFounderGene local_founder_gene(LocalFounderHash hash,
                                               std::uint32_t basis) {
  return static_cast<LocalFounderGene>((hash >> (2u * basis)) & 0x3u);
}

constexpr SiteWord local_founder_word(LocalFounderGene gene, std::uint32_t basis) {
  SiteWord word = kQ;
  if ((static_cast<std::uint8_t>(gene) & 0x1u) != 0u) word |= owned_bond_bit(basis);
  if ((static_cast<std::uint8_t>(gene) & 0x2u) != 0u) word |= energy_bit(basis);
  return word;
}

constexpr DevelopmentalSeedSite local_founder_site(LocalFounderGene gene,
                                                    std::uint32_t basis) {
  switch (basis) {
    case 0u: return {-1, 0, 0, local_founder_word(gene, basis)};
    case 1u: return {0, -1, 0, local_founder_word(gene, basis)};
    case 2u: return {0, 0, -1, local_founder_word(gene, basis)};
    case 3u: return {1, 1, 1, local_founder_word(gene, basis)};
  }
  return {0, 0, 0, kQ};
}

constexpr std::array<DevelopmentalSeedSite, kLocalFounderBasisCount>
local_founder_seed(LocalFounderHash hash) {
  return {{local_founder_site(local_founder_gene(hash, 0u), 0u),
           local_founder_site(local_founder_gene(hash, 1u), 1u),
           local_founder_site(local_founder_gene(hash, 2u), 2u),
           local_founder_site(local_founder_gene(hash, 3u), 3u)}};
}

// The first executable local-area founder: two adjacent B/E founders on
// distinct BCC bases.  It is deliberately only a field nucleus; it is not
// claimed to be an adaptive synapse, bounded cluster, or language area.
constexpr LocalFounderHash kTwoBasisFieldFounderHash = 0x0fu;

// A local area word extends the generic incoming-founder grammar with local
// B/E/C scaffold genes at the receptor.  This is still just local
// matter at birth: no gene names a feature, target coordinate, graph edge, or
// runtime action.  F decides whether the scaffold becomes causal.
using LocalAreaHash = std::uint32_t;
constexpr std::uint32_t kLocalAreaFounderShift = 0u;
constexpr std::uint32_t kLocalAreaScaffoldShift = 8u;
constexpr std::uint32_t kLocalAreaEnergyScaffoldShift = 12u;
constexpr std::uint32_t kLocalAreaConformationScaffoldShift = 16u;
constexpr std::uint32_t kLocalAreaPositiveFaceScaffoldShift = 24u;

constexpr LocalFounderHash local_area_founder_hash(LocalAreaHash hash) {
  return static_cast<LocalFounderHash>((hash >> kLocalAreaFounderShift) & 0xffu);
}

constexpr SiteWord local_area_scaffold_word(LocalAreaHash hash) {
  SiteWord word = kQ;
  for (std::uint32_t basis = 0u; basis < kLocalFounderBasisCount; ++basis) {
    if (((hash >> (kLocalAreaScaffoldShift + basis)) & 0x1u) != 0u)
      word |= owned_bond_bit(basis);
    if (((hash >> (kLocalAreaEnergyScaffoldShift + basis)) & 0x1u) != 0u)
      word |= energy_bit(basis);
    if (((hash >> (kLocalAreaConformationScaffoldShift + basis)) & 0x1u) != 0u)
      word |= channel_bit(kConformationShift, basis);
  }
  for (std::uint32_t face = 0u; face < 8u; ++face) {
    if (((hash >> (kLocalAreaPositiveFaceScaffoldShift + face)) & 0x1u) != 0u)
      word |= face_bit(face);
  }
  return word;
}

constexpr std::array<DevelopmentalSeedSite, kLocalFounderBasisCount + 1u>
local_area_seed(LocalAreaHash hash) {
  const auto founders = local_founder_seed(local_area_founder_hash(hash));
  return {{founders[0], founders[1], founders[2], founders[3],
           {0, 0, 0, local_area_scaffold_word(hash)}}};
}

// One incoming B0/E0 founder plus one local B0 scaffold.  Under F the seed
// forms a two-hop, B0-gated C0/R0 relay.  It remains a minimal causal block,
// not a learned, grown, or language-capable area.
constexpr LocalAreaHash kTwoHopRelayAreaHash = 0x0103u;

// Two generic local roads share one receptor: incoming B0/E0 and B1/E1 genes,
// plus B0/B1 scaffold lanes.  The word is a compact hub candidate; its branch
// interaction and selective lesions must be measured under F before it is
// treated as an area.
constexpr LocalAreaHash kTwoBranchHubAreaHash = 0x030fu;

// A translated interface: input B0/E0 reaches a local E1/C1 scaffold.  Under
// F the interface makes the +u1 target differ between intact B0 and B0-cut
// input.  It is a compact direction-changing causal block, not yet learned.
constexpr LocalAreaHash kTranslatedInterfaceHash = 0x00022003u;
constexpr LocalAreaHash kTranslatedInterfaceB0CutHash = 0x00022002u;

// Rejected shared-road/interface candidate.  The compact union is useful as
// a regression control: its pulse-dependent target change remains after its
// helper face is removed and also occurs without the helper scaffold.  It
// therefore must never be promoted to an inhibitory composition.
constexpr LocalAreaHash kRejectedRelayCrHelperUnionHash = 0x02000303u;

// A six-bit local interface gene for a helper situated beside an incoming C/R
// residue.  The palette spans its B0/E0/B1/E1/positive-face1/negative-face1
// lanes; it is a reusable cell type, not a named graph connection.
using LocalInterfaceGene = std::uint8_t;
constexpr SiteWord local_cr_interface_word(LocalInterfaceGene gene) {
  constexpr std::array<SiteWord, 6> kLanes{{
      owned_bond_bit(0u), energy_bit(0u), owned_bond_bit(1u), energy_bit(1u),
      face_bit(1u), face_bit(5u),
  }};
  SiteWord word = kQ;
  for (std::uint32_t bit = 0u; bit < kLanes.size(); ++bit) {
    if (((gene >> bit) & 0x1u) != 0u) word |= kLanes[bit];
  }
  return word;
}

// B0+B1+positive-face1.  Adjacent C0/R0 input suppresses target R1 plus
// face5 to Q; removing helper B0 preserves face5.  This is an inhibitory local
// interface primitive, not an area or learning claim.
constexpr LocalInterfaceGene kCrInhibitoryHelperGene = 0x15u;

// A one-bit phase gene for the native parity cell.  Birth places two adjacent
// BCC cells with complementary C0/R0 phase and the carrier polarity required
// by F's existing XOR-controlled C/R transpose.  Runtime inputs are ordinary
// local E0 and R0 molecules; the hash contains no Boolean program or answer.
using LocalParityHash = std::uint8_t;
constexpr LocalParityHash kNativeParityCellHash = 0x01u;

constexpr std::array<DevelopmentalSeedSite, 2> local_parity_seed(LocalParityHash hash) {
  SiteWord receptor = 0x000000efu;
  if ((hash & 0x01u) != 0u) receptor |= channel_bit(kConformationShift, 0u);
  return {{{-1, 0, 0, 0x000000feu}, {0, 0, 0, receptor}}};
}

}  // namespace substrate::bcc32
