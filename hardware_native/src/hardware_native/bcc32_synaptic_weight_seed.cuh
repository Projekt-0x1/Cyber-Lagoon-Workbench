#pragma once

// One compact physical germ for a local plastic synapse.
//
// The hash is interpreted only at birth by the generic four-basis founder
// grammar.  Its four two-bit loci all request bond-only founder matter.  It
// contains no weight, truth table, timing program, target, or host callback.
// Complete reversible F turns later local molecules into the tissue state.

#include "bcc32_local_founder_hash.cuh"

namespace substrate::bcc32 {

using SynapticWeightSeedHash = LocalFounderHash;

// 01 on each of four bases: B0, B1, B2, B3, with no authored energy.
inline constexpr SynapticWeightSeedHash kSynapticWeightSeedHash = 0x55u;

constexpr std::array<DevelopmentalSeedSite, kLocalFounderBasisCount> synaptic_weight_seed(
    SynapticWeightSeedHash hash) {
  return local_founder_seed(hash);
}

static_assert(local_founder_gene(kSynapticWeightSeedHash, 0u) == LocalFounderGene::bond);
static_assert(local_founder_gene(kSynapticWeightSeedHash, 1u) == LocalFounderGene::bond);
static_assert(local_founder_gene(kSynapticWeightSeedHash, 2u) == LocalFounderGene::bond);
static_assert(local_founder_gene(kSynapticWeightSeedHash, 3u) == LocalFounderGene::bond);

}  // namespace substrate::bcc32
