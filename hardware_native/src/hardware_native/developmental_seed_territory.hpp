#pragma once

// Γ SEED/FIELD TERRITORY ORIGIN DERIVATION — opt-in successor to authored
// SeedBlock.coordinate[3] / FieldBlock.center[3] (github issue #1206, patch
// 0002's canonical Γ ABI in bcc32_network_recipe.hpp).
//
// ── The defect this file answers ──────────────────────────────────────────
//
// Every genome authored against the canonical ABI today fills a seed's
// coordinate by hand, e.g. `kBaseCoordinate + t * kTerritorySpacing`. That
// places territories on a straight line at addresses a person chose, and the
// same address appears in every world the genome is gestated in: the Life
// Function's candidate-scoring machinery can only perturb that hand-chosen
// layout, never produce one from scratch, because the anchor the candidates
// are offsets from was never itself a function of the body or the
// environment. See `hardcoded-region-origins-are-the-repo-wide-enumeration-
// cancer` and issue #1206.
//
// ── Precedent this file reuses, not reinvents ─────────────────────────────
//
// `developmental_territory.hpp` already solved exactly this problem for
// territory placement in general: CARDINALITY -> ALLOCATE -> FIELD ->
// PROPOSE -> ARBITRATE -> COMMIT, representation-neutral, measured
// 0.08 us/territory flat to 4,194,304 territories (commit 33c97d5666). This
// file is a thin binding of that pipeline to the Γ seed/field vocabulary: a
// `SeedTerritoryRequest` is a lineage/axis/extent DESCRIPTION (an ordinal,
// which developmental axis it continues along, how much room it needs) --
// never an address -- and `derive_seed_territory_origins` runs PROPOSE /
// ARBITRATE against a real world-occupancy field to produce concrete
// `coordinate[3]` triples usable as SeedBlock.coordinate / FieldBlock.center
// values without changing either struct's layout.
//
// The world-occupancy field bound here is `PageOccupancyField`, one entry per
// page at exactly the granularity `bcc32_network_page_directory.cuh`'s
// `page_index_for_coordinate` already addresses: "closed" means "this page
// already holds grown network matter". That is a real reading of what has
// already been grown in this specific run (the body manifest / development
// environment the issue asks for), not a second, fabricated occupancy
// concept layered on top of the real one.
//
// ── ⛔ The trap this design must not fall into ────────────────────────────
//
// Deriving an origin from H(Γ || ordinal) ALONE looks like it removes the
// hardcoded constant but is exactly as enumerative as the original: it
// returns the same page in every world, so a dose-matched obstruction cannot
// move it. `derive_seed_territory_origins_HASH_ONLY_TRAP` below is kept
// deliberately as that excluded mechanism, purely as the contract's negative
// control -- it must FAIL the obstruction falsifier every time, which is what
// proves the falsifier discriminates the real fix rather than always passing.
// World-dependence lives in `allocate_territories`'s PROPOSE/ARBITRATE, which
// is why `derive_seed_territory_origins` takes the occupancy field as a
// required argument and the trap function does not.
//
// ── Scope of this slice (github issue #1206) ──────────────────────────────
//
// This is an ADDITIVE, opt-in path. `SeedBlock.coordinate[3]` and
// `FieldBlock.center[3]` in bcc32_network_recipe.hpp are UNCHANGED -- same
// layout, same size, same ABI version -- so none of the 20+ existing
// consumers (construction kernels, certification, the founder gestation
// contract, every Direct Adult test fixture) are touched by this file. They
// remain the legacy/donor representation, explicitly marked as such at their
// declaration, until a follow-up migrates genome-authoring call sites to call
// `derive_seed_territory_origins` instead of hand-picking coordinates and
// then deletes the donor path. See #1206 for what is left.

#include <cstddef>
#include <cstdint>
#include <vector>

#include "bcc32_network_page_directory.cuh"
#include "developmental_territory.hpp"

namespace substrate::bcc32::network_recipe {

// The persistent unit a genome author (or, once migrated, the Life
// Function's planning phase) hands to the derivation: a SIZE, an ordinal and
// a developmental axis -- never a position. `ordinal` is the territory's
// identity within the developmental program (mirrors the slot it will
// occupy in Genome.seeds/fields, exactly as Γ's own authoring order already
// does); `axis` records which developmental axis/lineage-relative direction
// this territory continues along; `footprint_extent` is how much room it
// needs, not where that room is.
struct SeedTerritoryRequest {
  std::uint32_t ordinal = 0u;
  std::uint32_t axis = 0u;
  std::uint32_t footprint_extent = 1u;
};

// World occupancy the placement is arbitrated against, at page granularity.
// A caller building this from the real page directory (one entry per page's
// live `node_count`) is reading what has actually already been grown in this
// run; a caller building it from all-zero pages is describing an empty
// world. Either way this is the field the ISSUE requires: obstruction and
// its dose-matched sham are properties the CALLER controls, not this file.
struct PageOccupancyField {
  const std::uint32_t* page_node_counts = nullptr;  // page_count entries
  std::uint32_t page_count = 0u;

  [[nodiscard]] bool closed(std::uint64_t cell) const {
    return page_node_counts != nullptr && cell < page_count && page_node_counts[cell] > 0u;
  }
};

// Where a territory's origin ended up, plus the evidence that it searched.
// `page_cell` is the page-directory cell the pipeline arbitrated (the same
// space `page_index_for_coordinate` reads), and `coordinate` is a concrete
// triple directly assignable to SeedBlock.coordinate / FieldBlock.center.
struct DerivedTerritoryOrigin {
  std::uint32_t coordinate[3] = {0u, 0u, 0u};
  std::uint32_t page_cell = 0u;
  std::uint32_t rounds = 0u;
  bool placed = false;
};

namespace detail {

// Binds developmental_territory.hpp's Space concept (only `cell_count()` is
// actually called by `allocate_territories`) to the page-grid volume: one
// cell per page, exactly `bcc32_network_page_directory.cuh`'s addressing.
struct PageGridSpace {
  [[nodiscard]] std::uint64_t cell_count() const { return kPageCount; }
};

// The lowest-coordinate corner of a page, in the SAME coordinate volume
// `page_index_for_coordinate` addresses -- so a derived origin round-trips
// through the existing page directory without a second coordinate system.
inline void page_corner(std::uint64_t cell, std::uint32_t (&out)[3]) {
  const std::uint32_t page_id = static_cast<std::uint32_t>(cell);
  out[0] = (page_id % kPageGridExtent) * kPageSiteExtent;
  out[1] = ((page_id / kPageGridExtent) % kPageGridExtent) * kPageSiteExtent;
  out[2] = (page_id / (kPageGridExtent * kPageGridExtent)) * kPageSiteExtent;
}

}  // namespace detail

// Derive `count` seed/field origins, arbitrated against `body_manifest`.
// World-dependence enters exactly once, at the `allocate_territories` call:
// the same genome_root and the same requests, arbitrated against a world
// that already occupies the page an unobstructed run would have chosen,
// must land elsewhere. The in-page offset applied afterwards (`reach`) is a
// deterministic function of Γ's own request fields only -- the
// world-dependence has already happened at page selection, which is the
// anchor the issue is about.
inline void derive_seed_territory_origins(std::uint64_t genome_root,
                                           const SeedTerritoryRequest* requests,
                                           std::size_t count,
                                           const PageOccupancyField& body_manifest,
                                           DerivedTerritoryOrigin* out,
                                           std::uint32_t max_rounds = 32u) {
  if (requests == nullptr || out == nullptr || count == 0u) return;

  detail::PageGridSpace space;
  std::vector<developmental::TerritoryRequest> territory_requests(count);
  for (std::size_t index = 0u; index < count; ++index) {
    territory_requests[index].ordinal = requests[index].ordinal;
    territory_requests[index].footprint_count =
        requests[index].footprint_extent == 0u ? 1u : requests[index].footprint_extent;
  }
  std::vector<developmental::TerritoryPlacement> placements(count);
  developmental::allocate_territories(space, body_manifest, genome_root,
                                       territory_requests.data(), count, placements.data(),
                                       max_rounds);

  for (std::size_t index = 0u; index < count; ++index) {
    out[index] = DerivedTerritoryOrigin{};
    out[index].rounds = placements[index].rounds;
    out[index].placed = placements[index].placed;
    if (!placements[index].placed) continue;
    out[index].page_cell = static_cast<std::uint32_t>(placements[index].cell);
    std::uint32_t corner[3];
    detail::page_corner(placements[index].cell, corner);
    const std::uint32_t reach = requests[index].footprint_extent % (kPageSiteExtent - 1u);
    out[index].coordinate[0] = corner[0];
    out[index].coordinate[1] = corner[1];
    out[index].coordinate[2] = corner[2];
    out[index].coordinate[requests[index].axis % 3u] += reach;
  }
}

// ⛔ TRAP, kept ONLY as the contract's excluded-mechanism negative control.
// This is H(genome || ordinal) with NO PROPOSE/ARBITRATE against any
// occupancy field at all -- deterministic, parallel, and precisely the
// enumeration cancer in a better font that #1206 warns against. It must not
// be called by real genome-authoring code; it exists so the contract can
// prove its own obstruction falsifier actually discriminates the fixed
// mechanism from the trap, rather than passing regardless of what it is
// handed.
inline void derive_seed_territory_origins_HASH_ONLY_TRAP(std::uint64_t genome_root,
                                                           const SeedTerritoryRequest* requests,
                                                           std::size_t count,
                                                           DerivedTerritoryOrigin* out) {
  for (std::size_t index = 0u; index < count; ++index) {
    const std::uint64_t cell =
        developmental::territory_subseed(genome_root, requests[index].ordinal, 0u, 0u) %
        kPageCount;
    out[index] = DerivedTerritoryOrigin{};
    out[index].placed = true;
    out[index].page_cell = static_cast<std::uint32_t>(cell);
    std::uint32_t corner[3];
    detail::page_corner(cell, corner);
    out[index].coordinate[0] = corner[0];
    out[index].coordinate[1] = corner[1];
    out[index].coordinate[2] = corner[2];
  }
}

}  // namespace substrate::bcc32::network_recipe
