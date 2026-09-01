#pragma once

// Compact local molecular-corridor grammar.
//
// Each two-bit gene materializes only ordinary B/E matter at one of three
// consecutive sites along a single BCC basis.  The hash is an initial
// condition, not an instruction stream: after genesis the unchanged
// reversible law F is the sole dynamics.  A corridor is the smallest
// direction-bearing seed from which a decaying local field can be tested
// without a host field buffer or a coordinate-labelled region.

#include <array>
#include <cstdint>

#include "bcc32_developmental_seed.cuh"
#include "bcc32_law.cuh"

namespace substrate::bcc32 {

using GradientFieldHash = std::uint8_t;

enum class GradientFieldGene : std::uint8_t {
  quiescent = 0u,
  bond = 1u,
  energy = 2u,
  bond_energy = 3u,
};

constexpr std::uint32_t kGradientFieldSiteCount = 3u;

constexpr GradientFieldGene gradient_field_gene(GradientFieldHash hash,
                                                std::uint32_t site) {
  return static_cast<GradientFieldGene>((hash >> (2u * site)) & 0x3u);
}

constexpr SiteWord gradient_field_word(GradientFieldGene gene, std::uint32_t basis) {
  SiteWord word = kQ;
  if ((static_cast<std::uint8_t>(gene) & 0x1u) != 0u) word |= owned_bond_bit(basis);
  if ((static_cast<std::uint8_t>(gene) & 0x2u) != 0u) word |= energy_bit(basis);
  return word;
}

constexpr DevelopmentalSeedSite gradient_field_site(GradientFieldGene gene,
                                                     std::uint32_t distance,
                                                     std::uint32_t basis) {
  const Int3 offset = basis_offset(static_cast<Basis>(basis));
  return {static_cast<std::int8_t>(-static_cast<std::int32_t>(distance) * offset.x),
          static_cast<std::int8_t>(-static_cast<std::int32_t>(distance) * offset.y),
          static_cast<std::int8_t>(-static_cast<std::int32_t>(distance) * offset.z),
          gradient_field_word(gene, basis)};
}

constexpr std::array<DevelopmentalSeedSite, kGradientFieldSiteCount>
gradient_field_seed(GradientFieldHash hash, std::uint32_t basis = 0u) {
  return {{gradient_field_site(gradient_field_gene(hash, 0u), 1u, basis),
           gradient_field_site(gradient_field_gene(hash, 1u), 2u, basis),
           gradient_field_site(gradient_field_gene(hash, 2u), 3u, basis)}};
}

// A compact three-level occupancy profile.  It contains no target, program,
// activity schedule, or field-update code; F decides whether it has any
// receptor-visible effect.
constexpr GradientFieldHash kThreeLevelGradientFieldHash = 0x1bu;

}  // namespace substrate::bcc32
