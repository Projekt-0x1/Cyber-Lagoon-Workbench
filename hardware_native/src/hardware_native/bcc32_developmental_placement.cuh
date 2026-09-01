#pragma once

// BCC as ONE binding of the placement instruction, not as its foundation.
//
// `developmental_recipe.hpp` states DISTRIBUTE without knowing what a site is.
// This file supplies the lattice reading of that vocabulary:
//
//     pole            -> the germ's own origin, {0,0,0}
//     lineage         -> one of the eight tetrahedral directions
//     extend          -> integer steps along that direction
//     translate       -> add a genome site's relative offset
//     clearance       -> a Chebyshev box in relative lattice coordinates
//
// The Chebyshev box is deliberately crude. It is a CONSERVATIVE structural
// condition — "leave this much room" — and not a claim about S_P range, the
// release footprint, or the aperture radius. If it were tuned to a measured
// interaction range it would be a physics assertion needing its own evidence;
// as a slack requirement it only ever costs space.
//
// Placement runs on the host at founding time, which is where the hand-picked
// constants it replaces also lived. That is sanctioned under "host builds
// structure, never directs flow": choosing where matter is founded is
// structure, and F remains the sole interpreter afterwards. Nothing here runs
// on device and nothing here reads the organism after tick zero.

#include <cstddef>
#include <cstdint>
#include <unordered_set>

#include "bcc32_developmental_seed.cuh"
#include "bcc32_geometry.cuh"
#include "developmental_recipe.hpp"

namespace substrate::bcc32 {

// The lattice binding of the Space interface. Eight lineages, one per
// tetrahedral direction; fronts fan over them in order.
struct Bcc32PlacementSpace {
  using Site = Int3;
  using Offset = DevelopmentalSeedSite;

  [[nodiscard]] Site pole() const { return Int3{0, 0, 0}; }

  [[nodiscard]] std::uint32_t lineage_count() const { return 8u; }

  [[nodiscard]] Site extend(Site from, std::uint32_t lineage, std::int32_t steps) const {
    const Int3 ray = direction_offset(static_cast<Direction>(lineage & 0x7u));
    return Int3{from.x + ray.x * steps, from.y + ray.y * steps, from.z + ray.z * steps};
  }

  [[nodiscard]] Site translate(Site origin, const Offset& offset) const {
    return Int3{origin.x + offset.x, origin.y + offset.y, origin.z + offset.z};
  }

  template <class Fn>
  void for_each_clearance(Site site, std::uint32_t clearance, Fn&& visit) const {
    const std::int32_t radius = static_cast<std::int32_t>(clearance);
    for (std::int32_t dx = -radius; dx <= radius; ++dx)
      for (std::int32_t dy = -radius; dy <= radius; ++dy)
        for (std::int32_t dz = -radius; dz <= radius; ++dz)
          visit(Int3{site.x + dx, site.y + dy, site.z + dz});
  }
};

// The world as construction finds it. Populated from whatever is ALREADY
// founded — the germ, and any earlier region or rail block — so that a
// placement's answer depends on the founding order and content, which is the
// property the contract falsifies.
class Bcc32Occupancy {
 public:
  [[nodiscard]] bool occupied(Int3 site) const { return sites_.count(key(site)) != 0u; }

  void occupy(Int3 site) { sites_.insert(key(site)); }

  void occupy_seed(const DevelopmentalSeedSite* seed, std::size_t count, Int3 origin = {0, 0, 0}) {
    for (std::size_t index = 0u; index < count; ++index)
      occupy(Int3{origin.x + seed[index].x, origin.y + seed[index].y, origin.z + seed[index].z});
  }

  [[nodiscard]] std::size_t size() const { return sites_.size(); }

 private:
  // Relative coordinates stay well inside 21 bits per axis in every founding
  // path here; the bias keeps negatives packing without a sign branch.
  static std::uint64_t key(Int3 site) {
    const std::uint64_t x = static_cast<std::uint64_t>(site.x + (1 << 20)) & 0x1fffffull;
    const std::uint64_t y = static_cast<std::uint64_t>(site.y + (1 << 20)) & 0x1fffffull;
    const std::uint64_t z = static_cast<std::uint64_t>(site.z + (1 << 20)) & 0x1fffffull;
    return (x << 42) | (y << 21) | z;
  }

  std::unordered_set<std::uint64_t> sites_;
};

using Bcc32PlacementOutcome = developmental::DistributeOutcome<Int3>;

// Convenience wrapper: DISTRIBUTE a genome expansion into the lattice.
inline Bcc32PlacementOutcome distribute_bcc32_region(Bcc32Occupancy& world,
                                                     developmental::RecipeHash hash,
                                                     const DevelopmentalSeedSite* footprint,
                                                     std::size_t count) {
  const Bcc32PlacementSpace space;
  return developmental::distribute_region(space, world, hash, footprint, count);
}

}  // namespace substrate::bcc32
