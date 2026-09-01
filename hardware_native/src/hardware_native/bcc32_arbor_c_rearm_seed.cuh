#pragma once

// A compact developmental variant of the four-branch synaptic arbor.  The
// failed second complete episode leaves B/F/E intact at each contact hub but
// clears its local C lane.  The physical upstream site is already the contact's
// C2 receptor, so Gate 2's native B+E -> C relation is expressed by giving
// that existing receptor the same branch-local B2+E2 lanes.  This is a
// law-derived seed, not a host rearm operation: after genesis, only F advances
// these cells.

#include <array>
#include <cstddef>
#include <cstdint>

#include "bcc32_synaptic_arbor_seed.cuh"

namespace substrate::bcc32 {

using ArborCRearmSeedHash = std::uint64_t;

constexpr std::uint32_t kArborCRearmMaskShift = 60u;
constexpr ArborCRearmSeedHash kArborCRearmBaseMask = (ArborCRearmSeedHash{1} << kArborCRearmMaskShift) - 1u;
constexpr ArborCRearmSeedHash kArborCRearmBranchMask = 0x0fu;
constexpr std::size_t kArborCRearmSeedSiteCount = kSynapticArborSeedSiteCount;

constexpr ArborCRearmSeedHash make_arbor_c_rearm_seed_hash(SynapticArborSeedHash arbor_hash,
                                                            std::uint8_t donor_mask) {
  return (static_cast<ArborCRearmSeedHash>(arbor_hash) & kArborCRearmBaseMask) |
         ((static_cast<ArborCRearmSeedHash>(donor_mask) & kArborCRearmBranchMask)
          << kArborCRearmMaskShift);
}

constexpr SynapticArborSeedHash arbor_c_rearm_arbor_hash(ArborCRearmSeedHash hash) {
  return static_cast<SynapticArborSeedHash>(hash & kArborCRearmBaseMask);
}

constexpr std::uint8_t arbor_c_rearm_donor_mask(ArborCRearmSeedHash hash) {
  return static_cast<std::uint8_t>((hash >> kArborCRearmMaskShift) & kArborCRearmBranchMask);
}

// The delayed-credit contact's consequence lane is local basis 2.  Its
// physical upstream neighbour is the existing receptor at local (0,0,1), so
// this seed extends that receptor rather than colliding with it.
constexpr ArborCoordinate arbor_c_rearm_receptor_coordinate(std::size_t branch) {
  return synaptic_arbor_branch_coordinate(branch, {0, 0, 1});
}

constexpr SiteWord arbor_c_rearm_receptor_lanes(std::size_t branch) {
  const std::uint32_t basis = synaptic_arbor_branch_permutation(branch)[2u];
  return static_cast<SiteWord>(owned_bond_bit(basis) | energy_bit(basis));
}

constexpr std::array<DevelopmentalSeedSite, kArborCRearmSeedSiteCount> arbor_c_rearm_seed(
    ArborCRearmSeedHash hash) {
  std::array<DevelopmentalSeedSite, kArborCRearmSeedSiteCount> result{};
  const auto arbor = synaptic_arbor_seed(arbor_c_rearm_arbor_hash(hash));
  for (std::size_t index = 0u; index < arbor.size(); ++index) result[index] = arbor[index];
  const std::uint8_t mask = arbor_c_rearm_donor_mask(hash);
  for (std::size_t branch = 0u; branch < kSynapticArborBranchCount; ++branch) {
    if ((mask & (1u << branch)) == 0u) continue;
    const ArborCoordinate receptor = arbor_c_rearm_receptor_coordinate(branch);
    for (DevelopmentalSeedSite& site : result) {
      if (site.x == receptor[0] && site.y == receptor[1] && site.z == receptor[2])
        site.word = static_cast<SiteWord>(site.word | arbor_c_rearm_receptor_lanes(branch));
    }
  }
  return result;
}

constexpr ArborCRearmSeedHash kArborCRearmSeedHash =
    make_arbor_c_rearm_seed_hash(kSynapticArborSeedHash, 0x0fu);

}  // namespace substrate::bcc32
