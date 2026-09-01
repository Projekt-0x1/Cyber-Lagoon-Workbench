#pragma once

// Γ TERRITORY ORIGINS FOR THE DIRECT LANE.
//
// ── Why a second binding exists ───────────────────────────────────────────
//
// GitHub #1206: `SeedBlock.coordinate[3]` / `FieldBlock.center[3]` are authored
// absolute addresses, and every genome fills them by hand as
// `kBaseCoordinate + t * kTerritorySpacing` -- territories on a straight line
// at positions a person chose, identical in every world.
//
// `developmental_seed_territory.hpp` (commit aa6670f17e) repaired that by
// binding `developmental_territory.hpp`'s CARDINALITY/ALLOCATE/FIELD/PROPOSE/
// ARBITRATE/COMMIT pipeline to a `PageOccupancyField` over
// `bcc32_network_page_directory.cuh`. That work is correct and its falsifiers
// hold. It is bound to the wrong lane.
//
// ⛔ Measured 2026-08-18: `direct_network_life_function.cu` and
// `direct_network_brain.cuh` contain no reference to the page directory at all.
// `bcc32_network_page_directory.cuh` is reachable only from
// `bcc32_network_matter.cuh`, `bcc32_network_life_function.cuh` and their
// contracts -- the BCC lane, which the founder's canon makes donor evidence to
// be translated and then deleted. Its volume is also a different size
// (`kPageGridExtent = 32` x `kPageSiteExtent = 64` = 2048 sites per axis),
// while the direct genome develops around `kBaseCoordinate = 100000` with a
// territory spacing of 16384. So #1206's own stated next attack -- "migrate
// direct_network_life_function.cu's planning phase to call
// derive_seed_territory_origins" -- cannot be executed as written: there is no
// page-occupancy field in the direct path to arbitrate against, and the two
// coordinate volumes do not overlap.
//
// ── The world the direct lane actually has ────────────────────────────────
//
// It has `DirectDevelopmentEnvironmentV1`: up to `kMaxDevelopmentConstraints`
// boxes, each with a centre, a half-extent and flags. `environment_score_q16()`
// already treats `kEnvironmentConstraintHardExclude` as "no node may be here",
// so that predicate is the honest reading of "this territory is closed" -- it
// is not a new concept invented for placement, it is the one the compiler
// already enforces on every candidate coordinate it scores.
//
// That makes the world-dependence real rather than decorative: obstruct the
// region a territory would have taken and its ORIGIN moves, which is the
// property #1206 asks for and the one an authored constant can never have.
//
// ── What is deliberately NOT decided here ─────────────────────────────────
//
// The lattice is described by Γ's own quantities -- how many territories, how
// far each one reaches -- and never by a hardcoded volume. `cell_span` comes
// from the caller's reach and `grid_extent` from the territory count, so a
// genome asking for more or larger territories gets a proportionally larger
// developmental volume rather than being squeezed into a fixed one. There is no
// `kBaseCoordinate` in this file.

#include <cstddef>
#include <cstdint>
#include <vector>

#include "developmental_territory.hpp"
#include "hardware_native/direct_network_brain.cuh"

namespace substrate::direct_network::seed_territory {

// Direct's description of a territory: a complete `(lineage, axis, ordinal)`
// identity and a reach, never a position. `reach` is the developmental extent
// the territory expects to occupy, which makes cell size a property of the
// Direct genome rather than of this file.
struct DirectTerritoryRequest {
  std::uint32_t ordinal = 0u;
  std::uint32_t lineage = 0u;
  std::uint32_t axis = 0u;
  std::uint32_t reach = 1u;
};

// The developmental volume, sized from Γ. Cells are cubes of `cell_span`
// arranged `grid_extent` per axis; `body_origin` is where the body sits, so the
// volume is expressed relative to the organism rather than to absolute zero.
struct DirectTerritoryLattice {
  std::uint32_t grid_extent = 1u;
  std::int32_t cell_span = 1;
  std::int32_t body_origin[3] = {0, 0, 0};

  [[nodiscard]] std::uint64_t cell_count() const {
    const std::uint64_t extent = grid_extent;
    return extent * extent * extent;
  }

  // The representative coordinate of a cell: its centre, so a territory's
  // origin is not biased toward a corner of its own cell.
  void cell_center(std::uint64_t cell, std::int32_t (&out)[3]) const {
    const std::uint64_t extent = grid_extent;
    const std::int32_t half = cell_span / 2;
    const std::int32_t x = static_cast<std::int32_t>(cell % extent);
    const std::int32_t y = static_cast<std::int32_t>((cell / extent) % extent);
    const std::int32_t z = static_cast<std::int32_t>(cell / (extent * extent));
    out[0] = body_origin[0] + x * cell_span + half;
    out[1] = body_origin[1] + y * cell_span + half;
    out[2] = body_origin[2] + z * cell_span + half;
  }
};

// The ONE place the world enters. A cell is closed when its representative
// coordinate is hard-excluded by the development environment -- the same
// predicate `environment_score_q16()` already applies to every candidate
// coordinate the compiler scores, not a placement-only invention.
struct EnvironmentOccupancyField {
  const DirectTerritoryLattice* lattice = nullptr;
  const DirectDevelopmentEnvironmentV1* environment = nullptr;

  [[nodiscard]] bool closed(std::uint64_t cell) const {
    if (lattice == nullptr || environment == nullptr) return false;
    std::int32_t coord[3];
    lattice->cell_center(cell, coord);
    for (std::uint32_t index = 0u; index < environment->constraint_count; ++index) {
      const DevelopmentEnvironmentConstraint& constraint = environment->constraints[index];
      if ((constraint.flags & kEnvironmentConstraintHardExclude) == 0u) continue;
      const std::int32_t dx = coord[0] > constraint.center[0] ? coord[0] - constraint.center[0]
                                                              : constraint.center[0] - coord[0];
      const std::int32_t dy = coord[1] > constraint.center[1] ? coord[1] - constraint.center[1]
                                                              : constraint.center[1] - coord[1];
      const std::int32_t dz = coord[2] > constraint.center[2] ? coord[2] - constraint.center[2]
                                                              : constraint.center[2] - coord[2];
      if (dx <= static_cast<std::int32_t>(constraint.half_extent[0]) &&
          dy <= static_cast<std::int32_t>(constraint.half_extent[1]) &&
          dz <= static_cast<std::int32_t>(constraint.half_extent[2]))
        return true;
    }
    return false;
  }
};

// Where a territory's origin ended up, plus the evidence that it searched.
struct DirectDerivedOrigin {
  std::int32_t coordinate[3] = {0, 0, 0};
  std::uint64_t cell = 0u;
  std::uint32_t rounds = 0u;
  bool placed = false;
};

// Size the developmental volume from Γ alone. `grid_extent` is the smallest
// cube that holds `count` territories with room to be displaced -- 4x the
// requested count, the same headroom developmental_territory.hpp's arbitration
// converges against in 7-11 rounds regardless of scale.
inline DirectTerritoryLattice lattice_for(std::size_t count, std::uint32_t reach,
                                          const std::int32_t (&body_origin)[3]) {
  DirectTerritoryLattice lattice;
  lattice.cell_span = static_cast<std::int32_t>(reach > 0u ? reach * 2u : 2u);
  std::uint32_t extent = 1u;
  while (extent * extent * extent < count * 4u && extent < 1024u) ++extent;
  lattice.grid_extent = extent;
  lattice.body_origin[0] = body_origin[0];
  lattice.body_origin[1] = body_origin[1];
  lattice.body_origin[2] = body_origin[2];
  return lattice;
}

// DERIVE `count` territory origins by arbitrating Γ's descriptions against the
// development environment.
//
// ⛔ The trap this must not fall into: deriving from H(genome_root, ordinal)
// ALONE is deterministic, parallel, compact -- and returns the same coordinate
// in every world, so a dose-matched obstruction cannot move it. That is the
// enumeration cancer in a better font. World-dependence lives inside
// allocate_territories, at PROPOSE (against `field`) and ARBITRATE (against the
// other territories being derived in the same pass), and the contract proves it
// by closing exactly the cells an unobstructed run chose.
inline void derive_direct_territory_origins(std::uint64_t genome_root,
                                            const DirectTerritoryRequest* requests,
                                            std::size_t count,
                                            const DirectTerritoryLattice& lattice,
                                            const EnvironmentOccupancyField& field,
                                            DirectDerivedOrigin* out,
                                            std::uint32_t max_rounds = 32u) {
  if (requests == nullptr || out == nullptr || count == 0u) return;

  std::vector<developmental::TerritoryRequest> pipeline;
  pipeline.reserve(count);
  for (std::size_t index = 0u; index < count; ++index)
    // No request identity is dropped at the generic allocator boundary. A
    // Direct territory differs semantically by `(lineage, axis, ordinal)`.
    pipeline.push_back(developmental::TerritoryRequest{
        requests[index].ordinal, requests[index].reach > 0u ? requests[index].reach : 1u,
        requests[index].lineage, requests[index].axis});

  std::vector<developmental::TerritoryPlacement> placements(count);
  developmental::allocate_territories(lattice, field, genome_root, pipeline.data(), count,
                                      placements.data(), max_rounds);

  for (std::size_t index = 0u; index < count; ++index) {
    out[index] = DirectDerivedOrigin{};
    out[index].cell = placements[index].cell;
    out[index].rounds = placements[index].rounds;
    out[index].placed = placements[index].placed;
    if (placements[index].placed)
      lattice.cell_center(placements[index].cell, out[index].coordinate);
  }
}

// ⛔ NEGATIVE CONTROL — DO NOT CALL FROM PRODUCTION.
//
// The rejected mechanism, kept in tree on purpose. It derives an origin from
// H(genome_root, ordinal) with no reference to the world, which is what an
// authored constant amounts to once you dress it in a hash. The contract runs
// it through the IDENTICAL dose-matched obstruction and requires it NOT to
// move. Without that arm, "the origin moved under obstruction" would not
// discriminate the real mechanism -- a falsifier that every candidate passes
// tests nothing.
inline void derive_direct_territory_origins_HASH_ONLY_TRAP(
    std::uint64_t genome_root, const DirectTerritoryRequest* requests, std::size_t count,
    const DirectTerritoryLattice& lattice, const EnvironmentOccupancyField&,
    DirectDerivedOrigin* out) {
  if (requests == nullptr || out == nullptr || count == 0u) return;
  const std::uint64_t cells = lattice.cell_count();
  for (std::size_t index = 0u; index < count; ++index) {
    out[index] = DirectDerivedOrigin{};
    if (cells == 0u) continue;
    out[index].cell =
        developmental::territory_subseed(genome_root, requests[index].ordinal, 0u, 0u) % cells;
    out[index].rounds = 1u;
    out[index].placed = true;
    lattice.cell_center(out[index].cell, out[index].coordinate);
  }
}

}  // namespace substrate::direct_network::seed_territory
