#pragma once

// Γ TERRITORY ALLOCATION — the compiler shape, not the embryology shape.
//
// ── Why this file exists, measured ────────────────────────────────────────
//
// `developmental_recipe.hpp`'s DISTRIBUTE places ONE territory by seeding a
// construction front at the pole and stepping it outward until it stands in
// open ground. That is a faithful picture of a growth cone and it is the wrong
// program for a compiler. Measured 2026-08-17 with the real executor and the
// real lattice binding (footprint 32, clearance 3, stride 8, margin 3):
//
//     territories   sites      wall        probes        refusals
//         256       8,192      0.062 s        33,664        32,640
//       1,024      32,768      0.620 s       527,872       523,776
//       2,048      65,536      2.584 s     2,104,320     2,096,128
//       4,096     131,072     10.330 s     8,402,944     8,386,560
//       8,192     262,144     39.536 s    33,583,104    33,550,336
//
// Every doubling costs 3.8x-4.0x. That is O(R^2), measured across five
// octaves rather than fitted, and 99.90% of every probe is a REFUSAL: front
// number R has no way to learn where the first R-1 territories went except by
// stepping through all of them. One million sites extrapolates to ~10 minutes
// single-threaded; ten million to ~16 hours. An adult cannot be compiled by
// this, and no inner-loop optimisation fixes an exponent.
//
// ── The distinction that matters ──────────────────────────────────────────
//
// Developmental SEMANTICS are not developmental SLOWNESS. A biological brain
// takes months because cells physically divide and axons physically grow;
// there is no architectural virtue in reproducing that wall clock. What we
// want is the CAUSAL RESULT of development — territories whose positions were
// determined by the organism's own state rather than by a person — and that
// result does not require the serial trajectory that produced it in biology.
//
// So the same instruction is re-expressed as a compiler pipeline:
//
//     CARDINALITY   parallel scan of every territory's size -> pool offsets
//     ALLOCATE      one bounded allocation, never per-territory growth
//     FIELD         parallel histogram of the world into coarse cells
//     PROPOSE       every territory independently hashes to a candidate cell
//     ARBITRATE     sort claims, lowest ordinal wins, losers re-propose
//     COMMIT        parallel footprint write
//
// Work per round is O(R) plus one O(R log R) sort, and the rounds are bounded
// by occupancy rather than by R. Nothing steps past anything.
//
// ── The trap this design has to avoid ─────────────────────────────────────
//
// ⛔ Deriving an origin from H(Γ, ordinal) ALONE recreates the enumeration
// cancer in a more sophisticated font. A pure function of the ordinal is a
// constant table with extra steps: it returns the same address in every world,
// so a dose-matched obstruction cannot move it. See
// `hardcoded-region-origins-are-the-repo-wide-enumeration-cancer`.
//
// The world-dependence is therefore load-bearing and lives in two places:
// PROPOSE tests the candidate against the coarse field built from the world as
// it actually is, and ARBITRATE tests it against the other territories being
// grown at the same time. Obstruct exactly the cells a territory would take
// and it must land elsewhere; leave the world alone and it must reproduce the
// previous answer byte for byte. That pair is the falsifier, unchanged from
// the serial instruction — the mechanism is superseded, the invariant is not.
//
// ── Order independence ────────────────────────────────────────────────────
//
// A parallel allocator whose answer depends on which thread arrived first is
// not a developmental program, it is a race. ARBITRATE resolves every contested
// cell in favour of the LOWEST ORDINAL, which is a property of Γ rather than of
// the schedule, so the committed layout is identical under any evaluation
// order. `allocate_territories()` is written so a caller can prove that by
// permuting the proposal order and comparing the result.
//
// ── Representation neutrality, unchanged ──────────────────────────────────
//
// Nothing here knows what a Site is. The Space supplies a coarse cell field on
// top of the four operations DISTRIBUTE already required:
//
//     using Site;
//     Site translate(Site origin, const Offset&) const;
//     std::uint64_t cell_count() const;            developmental field extent
//     Site cell_origin(std::uint64_t cell) const;  where a cell's matter begins
//     std::uint64_t cell_of(Site) const;           which cell a site falls in
//
// A BCC lattice, a node-pool index space and a plain integer lane space all
// bind this equally.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace developmental {

// The persistent unit hands the allocator a size and a complete developmental
// identity, never a position. `footprint` is the genome's own expansion;
// `(lineage, axis, ordinal)` is the semantic identity used for proposals,
// deterministic pool order, and arbitration. The trailing defaults preserve
// current non-Direct callers while giving the canonical Direct path its full
// identity rather than a lossy ordinal winner key.
struct TerritoryRequest {
  std::uint32_t ordinal = 0u;
  std::uint32_t footprint_count = 0u;
  std::uint32_t lineage = 0u;
  std::uint32_t axis = 0u;
};

// Where a territory ended up, plus the evidence that it searched. `rounds` is
// how many times this territory had to re-propose; a territory that never
// re-proposed in any world has not demonstrated that its answer depends on one.
struct TerritoryPlacement {
  std::uint64_t cell = 0u;
  std::uint32_t pool_offset = 0u;
  std::uint32_t footprint_count = 0u;
  std::uint32_t rounds = 0u;
  bool placed = false;
};

struct AllocationReport {
  std::uint32_t territories = 0u;
  std::uint32_t placed = 0u;
  std::uint32_t pool_sites = 0u;      // CARDINALITY's answer, known before any search
  std::uint32_t field_cells = 0u;     // coarse cells the world was reduced to
  std::uint32_t occupied_before = 0u; // cells the world already held
  std::uint32_t rounds = 0u;          // ARBITRATE passes actually needed
  std::uint64_t proposals = 0u;       // total candidate evaluations, all rounds
  std::uint64_t displacements = 0u;   // proposals refused by field or rival
  std::uint32_t ambiguous_identities = 0u;  // duplicate complete IDs fail closed
};

[[nodiscard]] constexpr bool territory_identity_less(const TerritoryRequest& left,
                                                      const TerritoryRequest& right) {
  if (left.lineage != right.lineage) return left.lineage < right.lineage;
  if (left.axis != right.axis) return left.axis < right.axis;
  return left.ordinal < right.ordinal;
}

[[nodiscard]] constexpr bool territory_identity_equal(const TerritoryRequest& left,
                                                       const TerritoryRequest& right) {
  return left.lineage == right.lineage && left.axis == right.axis &&
         left.ordinal == right.ordinal;
}

// A hash proposes a candidate, but it never chooses a winner. The full
// `(lineage, axis, ordinal)` identity enters the proposal; deterministic
// arbitration below uses that same non-lossy tuple. Zero axis exactly retains
// the already-landed lineage behavior for legacy callers.
constexpr std::uint64_t territory_subseed(std::uint64_t genome, std::uint32_t ordinal,
                                          std::uint32_t lineage, std::uint32_t axis,
                                          std::uint32_t round) {
  std::uint64_t value = genome ^ (static_cast<std::uint64_t>(ordinal) * 0x9e3779b97f4a7c15ull) ^
                        (static_cast<std::uint64_t>(lineage) * 0x94d049bb133111ebull) ^
                        (static_cast<std::uint64_t>(axis) * 0xd6e8feb86659fd93ull) ^
                        (static_cast<std::uint64_t>(round) * 0xbf58476d1ce4e5b9ull);
  value ^= value >> 33;
  value *= 0xff51afd7ed558ccdull;
  value ^= value >> 33;
  value *= 0xc4ceb9fe1a85ec53ull;
  value ^= value >> 33;
  return value;
}

constexpr std::uint64_t territory_subseed(std::uint64_t genome, std::uint32_t ordinal,
                                          std::uint32_t lineage, std::uint32_t round) {
  return territory_subseed(genome, ordinal, lineage, 0u, round);
}

constexpr std::uint64_t territory_subseed(std::uint64_t genome, std::uint32_t ordinal,
                                          std::uint32_t round) {
  return territory_subseed(genome, ordinal, 0u, 0u, round);
}

// ALLOCATE every territory of a developmental program at once.
//
// `field` is the coarse occupancy the world already imposes: cell `c` is closed
// to construction when `field[c]` is true. The caller builds it from the world
// as it actually is (a parallel histogram in the CUDA lowering; a loop here),
// which is precisely where the world-dependence enters.
//
// `max_rounds` bounds ARBITRATE. Refusal is a real outcome and is reported, not
// papered over with a fallback cell.
template <class Space, class Field>
AllocationReport allocate_territories(const Space& space, const Field& field,
                                      std::uint64_t genome, const TerritoryRequest* requests,
                                      std::size_t count, TerritoryPlacement* out,
                                      std::uint32_t max_rounds = 32u) {
  AllocationReport report;
  if (requests == nullptr || out == nullptr || count == 0u)
    return report;

  const std::uint64_t cells = space.cell_count();
  report.territories = static_cast<std::uint32_t>(count);
  report.field_cells = static_cast<std::uint32_t>(cells);
  if (cells == 0u)
    return report;

  // ── CARDINALITY + ALLOCATE ──────────────────────────────────────────────
  // An exclusive prefix sum over the footprint sizes. The whole pool is sized
  // and carved before a single position is searched for, which is the thing
  // the serial front could never do: it had to place territory R-1 to know
  // where territory R would even start looking.
  //
  // ⚠ The scan runs in complete IDENTITY order, not in the order the caller happened to
  // hand the requests over. Measured 2026-08-17: scanning in array order makes
  // the birth artifact depend on the sequence a parallel expansion emitted its
  // territories in — the cells were already ordinal-keyed and invariant, but
  // the pool offsets were not, so two identical genomes expanded by different
  // thread counts produced different layouts. A developmental program's output
  // may depend on the world; it may not depend on the schedule.
  std::vector<std::uint32_t> by_identity(count);
  for (std::size_t index = 0u; index < count; ++index) {
    out[index] = TerritoryPlacement{};
    out[index].footprint_count = requests[index].footprint_count;
    by_identity[index] = static_cast<std::uint32_t>(index);
  }
  std::sort(by_identity.begin(), by_identity.end(),
            [&](std::uint32_t left, std::uint32_t right) {
              return territory_identity_less(requests[left], requests[right]);
            });
  for (std::size_t index = 1u; index < by_identity.size(); ++index) {
    if (territory_identity_equal(requests[by_identity[index - 1u]],
                                requests[by_identity[index]]))
      ++report.ambiguous_identities;
  }
  // There is no semantic tie-break below the full identity. An array position
  // would make the schedule authoritative, so identical IDs fail closed.
  if (report.ambiguous_identities != 0u) return report;
  std::uint32_t running = 0u;
  for (const std::uint32_t index : by_identity) {
    out[index].pool_offset = running;
    running += requests[index].footprint_count;
  }
  report.pool_sites = running;

  for (std::uint64_t cell = 0u; cell < cells; ++cell)
    if (field.closed(cell))
      ++report.occupied_before;

  // ── PROPOSE / ARBITRATE ─────────────────────────────────────────────────
  // Claims are (cell, lineage, axis, ordinal) tuples. Every unplaced territory
  // proposes once per round, all independently; then one sort resolves every
  // contested cell at once. Sorting by the full tuple and keeping
  // the first of each run is the whole arbitration, and it is why the answer
  // cannot depend on order. A claim carries its own index
  // alongside the arbitration key. Resolving the winner by searching for its
  // ordinal would put an O(R) scan inside a loop that already runs O(R) times
  // and hand the quadratic straight back.
  struct Claim {
    std::uint64_t cell;
    std::uint32_t lineage;
    std::uint32_t axis;
    std::uint32_t ordinal;
    std::uint32_t index;
    bool operator<(const Claim& other) const {
      if (cell != other.cell) return cell < other.cell;
      if (lineage != other.lineage) return lineage < other.lineage;
      if (axis != other.axis) return axis < other.axis;
      return ordinal < other.ordinal;
    }
  };

  std::vector<bool> taken(static_cast<std::size_t>(cells), false);
  std::vector<Claim> claims;
  claims.reserve(count);

  std::uint32_t remaining = static_cast<std::uint32_t>(count);
  for (std::uint32_t round = 0u; round < max_rounds && remaining > 0u; ++round) {
    ++report.rounds;
    claims.clear();

    for (std::size_t index = 0u; index < count; ++index) {
      if (out[index].placed)
        continue;
      ++report.proposals;
      ++out[index].rounds;
      const std::uint64_t cell = territory_subseed(
          genome, requests[index].ordinal, requests[index].lineage, requests[index].axis,
          round) % cells;
      // The world refuses first: a cell the organism already occupies is not
      // territory, however free the pool slot behind it happens to be.
      if (field.closed(cell) || taken[static_cast<std::size_t>(cell)]) {
        ++report.displacements;
        continue;
      }
      claims.push_back(Claim{cell, requests[index].lineage, requests[index].axis,
                             requests[index].ordinal, static_cast<std::uint32_t>(index)});
    }

    std::sort(claims.begin(), claims.end());

    for (std::size_t claim = 0u; claim < claims.size(); ++claim) {
      if (claim > 0u && claims[claim - 1u].cell == claims[claim].cell) {
        // Lost to a lower full identity. A property of the genome, not of the
        // schedule.
        ++report.displacements;
        continue;
      }
      TerritoryPlacement& placement = out[claims[claim].index];
      placement.cell = claims[claim].cell;
      placement.placed = true;
      taken[static_cast<std::size_t>(claims[claim].cell)] = true;
      --remaining;
      ++report.placed;
    }
  }
  return report;
}

// COMMIT. Kept separate from allocation on purpose: the placement is a pure
// decision over the field, so it can be checked, permuted and compared without
// any matter being written, and the write is a flat parallel pass afterwards.
template <class Space, class Offset, class Sink>
void commit_territories(const Space& space, const TerritoryPlacement* placements,
                        std::size_t count, const Offset* const* footprints, Sink&& sink) {
  for (std::size_t index = 0u; index < count; ++index) {
    if (!placements[index].placed)
      continue;
    const auto origin = space.cell_origin(placements[index].cell);
    for (std::uint32_t site = 0u; site < placements[index].footprint_count; ++site)
      sink(placements[index].pool_offset + site,
           space.translate(origin, footprints[index][site]));
  }
}

}  // namespace developmental
