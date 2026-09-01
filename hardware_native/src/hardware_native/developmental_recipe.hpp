#pragma once

// 🔴 SUPERSEDED AS A COMPILER, 2026-08-17 — see developmental_territory.hpp.
//
// The invariant below survived; this MECHANISM did not. `distribute_region` is
// a faithful growth cone, and measured the day it landed it costs O(R^2):
// 39.536 s for 8,192 territories, 3.8x-4.0x per doubling across five octaves,
// 99.90% of every probe a refusal, because front R can only discover where
// fronts 1..R-1 went by stepping through them. That is days of gestation for a
// language-capable adult, and an exponent is not fixable in the inner loop.
// Developmental semantics are not developmental slowness. The replacement
// measures 0.08 us/territory flat to 134M sites, carries the dose-matched
// obstruction across unchanged, and adds the order-independence a serial front
// never needed. Kept only as a green regression for three arithmetic contracts
// -- do not build anything new on it. Full measurement: the handbook section
// "The instruction above was right and its mechanism was disqualifying".
//
// Γ INSTRUCTIONS — construction operations, not coordinates, and not BCC.
//
// No `bcc32_` prefix and no substrate header, on purpose: this is the first
// piece of the repository written above the lattice rather than inside it.
//
// ── What this replaces ────────────────────────────────────────────────────
//
// Every hash-seeded region here is expanded from a compact genome (good) and
// then STAMPED at an absolute coordinate a human picked by hand (not good):
// `kRibbonOrigin{-80,-80,-80}`, `kJournalCounterOrigin{-20,-80,-80}`,
// `kRoadOrigin{-60,-60,-60}`, `kAccumulatorOrigin{-32,-21,20}`. Those constants
// are the recipe becoming the brain. `-20` is not a developmental fact; it is a
// person reasoning "the rails are near +80, so -20 is probably free" and then
// verifying the guess. Verifying a guess is not deriving a placement: the
// information about WHERE the region belongs lives in the source file, so the
// source file grows with the structure — exactly the enumeration Γ avoids.
//
// ── What a placement instruction says instead ─────────────────────────────
//
// SEED K construction fronts from the body-relative pole.
// EXTEND each front along its lineage by a fixed stride.
// REQUIRE a clearance condition against the world AS IT ACTUALLY IS.
// COMMIT where the condition is first met; BRANCH to the next front when a
//   front exhausts its budget.
//
// The load-bearing property, and the one the contract falsifies: THE ANSWER
// IS A FUNCTION OF THE WORLD, NOT OF THE SOURCE. Found the rails first and
// the region lands somewhere else than if you do not. A "recipe" whose output
// is the same in every world is a constant wearing a procedure's clothes, and
// `distribute_region()` is written so that a test can catch that.
//
// ── What is deliberately NOT here ─────────────────────────────────────────
//
// Γ's vocabulary is longer than this: BRANCH a lineage three ways under a
// dynamical condition, CREATE long tracts under a cost law, DIFFERENTIATE
// local time constants, MATURE cooperating structure, RETRACT what fails its
// condition. None of those exist yet. This file implements ONE instruction —
// DISTRIBUTE — and names the rest as absent rather than implying a whole
// developmental program is present. One executed instruction is worth more
// than a vocabulary that does not run.
//
// ── Why it is representation-neutral ──────────────────────────────────────
//
// A recipe written in `Int3` and tetrahedral directions has BCC baked into Γ,
// which makes the lattice load-bearing for the developmental program rather
// than one implementation of it. So the instruction is parameterized on a
// Space that supplies four operations, and BCC is one binding of that Space
// (`bcc32_developmental_placement.cuh`). A graph, a packed bit matrix or an
// integer lane space can bind it equally, and the contract proves that by
// executing the IDENTICAL recipe against a second, non-lattice Space.
//
//     Space:      Site, Offset, pole(), lineage_count(),
//                 extend(Site, lineage, steps), translate(Site, Offset),
//                 for_each_clearance(Site, clearance, Fn)
//     Occupancy:  occupied(Site), occupy(Site)
//
// The recipe never learns what a Site is.

#include <cstddef>
#include <cstdint>

namespace developmental {

using RecipeHash = std::uint32_t;

inline constexpr std::uint32_t kRecipeFrontShift = 0u;
inline constexpr std::uint32_t kRecipeLineageShift = 3u;
inline constexpr std::uint32_t kRecipeClearanceShift = 6u;
inline constexpr std::uint32_t kRecipeStrideShift = 10u;
inline constexpr std::uint32_t kRecipeGeneShift = 15u;
inline constexpr std::uint32_t kRecipeMarginShift = 23u;

// The reusable placement gene. Like kProcessiveWeightGene it is a validity
// tag, not an address: a hash that does not carry it is not a placement
// instruction and expands to nothing.
inline constexpr std::uint32_t kRecipeGene = 0x2du;
inline constexpr std::uint32_t kRecipeMaxFronts = 8u;
inline constexpr std::uint32_t kRecipeDefaultProbeBudget = 512u;

constexpr RecipeHash make_distribute_hash(std::uint32_t fronts, std::uint32_t lineage,
                                          std::uint32_t clearance, std::uint32_t stride,
                                          std::uint32_t margin, std::uint32_t gene) {
  return static_cast<RecipeHash>((fronts - 1u) & 0x7u) |
         static_cast<RecipeHash>((lineage & 0x7u) << kRecipeLineageShift) |
         static_cast<RecipeHash>((clearance & 0xfu) << kRecipeClearanceShift) |
         static_cast<RecipeHash>((stride & 0x1fu) << kRecipeStrideShift) |
         static_cast<RecipeHash>((gene & 0xffu) << kRecipeGeneShift) |
         static_cast<RecipeHash>((margin & 0x7u) << kRecipeMarginShift);
}

constexpr std::uint32_t recipe_fronts(RecipeHash hash) {
  return ((hash >> kRecipeFrontShift) & 0x7u) + 1u;
}

constexpr std::uint32_t recipe_lineage(RecipeHash hash) {
  return (hash >> kRecipeLineageShift) & 0x7u;
}

constexpr std::uint32_t recipe_clearance(RecipeHash hash) {
  return (hash >> kRecipeClearanceShift) & 0xfu;
}

constexpr std::uint32_t recipe_stride(RecipeHash hash) {
  return (hash >> kRecipeStrideShift) & 0x1fu;
}

constexpr std::uint32_t recipe_gene(RecipeHash hash) {
  return (hash >> kRecipeGeneShift) & 0xffu;
}

// ⭐ THE CONDITION THAT SITE-COLLISION ALONE CANNOT EXPRESS.
//
// Measured 2026-08-17, and the reason this field exists. With clearance alone,
// the journal counter's front committed at the POLE, on its first probe,
// inside the germ. Nothing was wrong with the search: the germ is a sparse
// 8-stride lattice of loci on two planes at z ~ +/-8, so a clearance-3 halo
// around a footprint lying near z == 0 genuinely touches no occupied site. The
// front had found a hollow mid-plane of the parent body and called it room.
//
// A construction front cannot thread a needle through existing tissue. "Vacant
// at every site I occupy" is not the developmental condition; "out in open
// territory" is. `margin` states it: the front must find margin + 1
// CONSECUTIVE clear candidates and commits at the FIRST of them, so a lucky
// gap inside the body cannot satisfy it while genuinely free space can.
//
// This is why arm A of the assay refuses a first-probe placement. A placement
// that never had to refuse anything has not shown that it left the body.
constexpr std::uint32_t recipe_margin(RecipeHash hash) {
  return (hash >> kRecipeMarginShift) & 0x7u;
}

constexpr bool valid_distribute_hash(RecipeHash hash) {
  return recipe_gene(hash) == kRecipeGene && recipe_stride(hash) > 0u &&
         recipe_fronts(hash) <= kRecipeMaxFronts;
}

// The construction receipt. `probes` and `refusals` are what distinguish a
// search from a lookup: a placement that lands on its first probe in every
// world has not demonstrably searched, and a contract should say so rather
// than accept the coordinate.
template <class Site>
struct DistributeOutcome {
  Site site{};
  std::uint32_t front = 0u;
  std::uint32_t probes = 0u;
  std::uint32_t refusals = 0u;
  bool placed = false;
};

// DISTRIBUTE a region's footprint into whatever room the world actually has.
//
// The footprint is the genome's own expansion (for BCC that is
// processive_weight_region_seed()'s relative sites); this instruction decides
// only WHERE it goes, never WHAT it is. On success the footprint sites — not
// the clearance halo — become occupied, so a second call sees the first
// region and routes around it without any source constant claiming it will.
template <class Space, class Occupancy, class Offset>
DistributeOutcome<typename Space::Site> distribute_region(
    const Space& space, Occupancy& world, RecipeHash hash, const Offset* footprint,
    std::size_t footprint_count, std::uint32_t probe_budget = kRecipeDefaultProbeBudget) {
  using Site = typename Space::Site;
  DistributeOutcome<Site> outcome;
  if (!valid_distribute_hash(hash) || footprint == nullptr || footprint_count == 0u)
    return outcome;

  const std::uint32_t fronts = recipe_fronts(hash);
  const std::uint32_t clearance = recipe_clearance(hash);
  const std::uint32_t margin = recipe_margin(hash);
  const std::int32_t stride = static_cast<std::int32_t>(recipe_stride(hash));
  const std::uint32_t lineages = space.lineage_count();
  if (lineages == 0u)
    return outcome;

  const auto footprint_clear = [&](const Site& candidate) {
    // REQUIRE CLEARANCE against the world as it actually is. The halo is a
    // conservative condition on the footprint's neighbourhood, not a claim
    // about any particular interaction range.
    bool clear = true;
    for (std::size_t index = 0u; index < footprint_count && clear; ++index) {
      const Site site = space.translate(candidate, footprint[index]);
      space.for_each_clearance(site, clearance, [&](const Site& probe) {
        if (world.occupied(probe))
          clear = false;
      });
    }
    return clear;
  };

  // SEED K construction fronts from the body-relative pole. Every front starts
  // where the organism already is; none starts at a chosen coordinate.
  for (std::uint32_t front = 0u; front < fronts; ++front) {
    const std::uint32_t lineage = (recipe_lineage(hash) + front) % lineages;
    std::uint32_t run = 0u;  // consecutive clear candidates seen on this front

    // EXTEND the front until it stands margin + 1 steps deep in open territory.
    for (std::uint32_t step = 0u; step < probe_budget; ++step) {
      const Site candidate =
          space.extend(space.pole(), lineage, stride * static_cast<std::int32_t>(step));
      ++outcome.probes;

      if (!footprint_clear(candidate)) {
        // Any occupancy resets the run. A gap INSIDE the body is not open
        // territory however clear its own halo happens to be — that is exactly
        // the failure recipe_margin() documents.
        ++outcome.refusals;
        run = 0u;
        continue;
      }
      if (++run <= margin)
        continue;

      // COMMIT at the FIRST clear step of the run: the near edge of open
      // territory, so the margin buys certainty rather than distance. The
      // region exists here because this operation put it here.
      const Site chosen = space.extend(space.pole(), lineage,
                                       stride * static_cast<std::int32_t>(step - margin));
      for (std::size_t index = 0u; index < footprint_count; ++index)
        world.occupy(space.translate(chosen, footprint[index]));
      outcome.site = chosen;
      outcome.front = front;
      outcome.placed = true;
      return outcome;
    }
    // BRANCH: this front is exhausted, so construction continues on the next
    // one. With K == 1 there is nowhere to branch to and the instruction
    // refuses — a refusal the caller must handle, never paper over with a
    // fallback coordinate.
  }
  return outcome;
}

}  // namespace developmental
