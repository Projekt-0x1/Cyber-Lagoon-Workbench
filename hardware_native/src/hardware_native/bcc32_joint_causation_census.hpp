#pragma once

// DOES THE UNMODIFIED LAW CONTAIN A TWO-CAUSE OUTPUT JUNCTION?
//
// The grounded-language arc closed on a measurement, not a guess: across eleven
// arms, twenty-eight emitted bytes and two disjoint alphabets, every byte the
// organism replied with was exactly the byte its route already held
// (`GATE7 displaced=28 same_byte_same_route=28`). The reply TRANSPORTS resident
// matter. It CONSTRUCTS NOTHING. External review confirmed this is structural
// rather than statistical -- the reciprocal exchange swaps corresponding
// eight-bit fields and performs no value transformation, so the query can
// change WHICH routes open and can never change WHAT byte comes out of one.
//
// ⭐⭐ THE CORRECTION THAT OPENS THIS FILE. The standing assumption was that a
// reversible law cannot construct, because construction looked like the
// many-to-one map `(A, B) -> O` that a bijection forbids. That is wrong, and
// naming why is the whole point:
//
//     REVERSIBILITY DOES NOT FORBID CONSTRUCTION. IT FORBIDS UNRECORDED
//     ERASURE.
//
//         (A, B, O)  |->  (A, B, O XOR f(A, B))
//
//     is a bijection -- applying it twice restores O, since f(A,B) XOR f(A,B)
//     vanishes. If O began blank it now carries f(A, B), and A and B are still
//     resident, so nothing was destroyed. What a bijection actually forbids is
//     `A, B -> O` with A and B GONE and no record of which inputs produced O.
//     That is bookkeeping, not impossibility.
//
// So "can this substrate construct?" is not answered by the language path at
// all. It is a question about the law, and it is CHEAP to ask, because the
// witness is one bit, not one sentence.
//
// ⭐ THE OBSERVABLE: THE SECOND-ORDER DIFFERENCE.
//
//     N(a, b) = F(x^a^b) XOR F(x^a) XOR F(x^b) XOR F(x)
//
// This is the exact algebraic residue of joint causation. If F were XOR-affine
// -- if every output bit were a fixed linear combination of input bits -- then
// N vanishes identically, because the two single-cause differences would sum to
// the joint one. N is nonzero at a site precisely when that site's update
// depends on a AND b TOGETHER in a way neither dependence alone accounts for.
//
// ⛔ AND IT IS NOT ENTAILED, WHICH IS THE POINT. This file exists downstream of
// a repeat offender: `live_bits = 32/32` looked like a probe-validity check and
// was in fact forced by injectivity -- any input change forces some output
// change whether or not the factor read the bit. `differing_sites != 0` has the
// same defect. N does NOT: a bijection can be perfectly XOR-affine (every
// invertible linear map is one), so `N == 0` and `N != 0` are BOTH reachable
// states of the world. The reading can come back empty and that is a real
// outcome, not an instrument failure.
//
// ⭐⭐ THE CAPABILITY FLOOR, AND WHY `joint_sites` IS NOT IT. N != 0 is
// necessary and not sufficient. N can be nonzero at a site that BOTH single
// causes already changed -- there the joint effect is a modulation of an
// existing write, not the filling of a blank. The witness external review
// specified is stricter and is `constructed_sites`:
//
//     cause A alone      -> site s is UNCHANGED from baseline
//     cause B alone      -> site s is UNCHANGED from baseline
//     A and B together   -> site s DIFFERS from baseline
//     and s is neither cause's own site
//
// ⭐ THE DOUBLE DISSOCIATION IS BUILT INTO THE DEFINITION, not bolted on. The
// two single-cause arms ARE the two lesions: "lesion A -> output lost" is
// literally the B-alone arm, and "lesion B -> output lost" is the A-alone arm.
// A site can only be counted when both of them are silent. There is no separate
// lesion step that could be dose-mismatched, and no sham that could be drawn
// from the wrong population -- the arms are the same three worlds throughout.
//
// ⚠ WHAT THIS DOES NOT ESTABLISH, so it is not assumed. A junction in the law
// is a CAPABILITY, not a faculty. It shows the substrate can write a site from
// two causes that neither writes alone; it does not show anything resident ever
// arranges two learned causes to meet at one, nor that the result is read. Those
// are the next questions and this file answers none of them. It also speaks only
// for the perturbations, separations and factors actually swept: a null result
// earns "no such junction within these bounds", never "no such junction".

#include <bit>
#include <cstddef>
#include <cstdint>

#include "bcc32_coordinate.hpp"
#include "bcc32_law.cuh"
#include "bcc32_law_netlist.cuh"
#include "bcc32_payload_difference_census.hpp"
#include "bcc32_reference.hpp"
#include "bcc32_types.cuh"

namespace substrate::bcc32::joint_causation {

struct JointReading {
  // Sites where N != 0 -- the update depends on both causes jointly. Necessary
  // for a junction, and NOT sufficient: this counts modulation of an existing
  // write as well as construction of a blank one.
  std::size_t joint_sites = 0u;

  // ⭐ THE CAPABILITY FLOOR. Sites that neither cause alone moved off baseline,
  // that both together did move, and that are neither cause's own site. Each
  // one is a constructed output with its own double dissociation.
  std::size_t constructed_sites = 0u;

  // Bits, not sites. External review put the floor at ONE CONSTRUCTED BIT, so
  // the honest unit is reported alongside the site count.
  std::size_t constructed_bits = 0u;

  // The strictest reading of "blank": the site held the quiescent word in the
  // BASE state, so the causes did not merely modulate a site that was already
  // carrying something.
  std::size_t constructed_from_quiescent = 0u;

  // ⭐⭐⭐ THE CORRECT FLOOR, AND IT IS PER-BIT, NOT PER-SITE.
  //
  //     blank = ~(F(x^a) XOR F(x)) AND ~(F(x^b) XOR F(x))     bits neither
  //                                                            cause moved
  //     built = blank AND (F(x^a^b) XOR F(x))                  of those, bits
  //                                                            both together did
  //
  // `constructed_sites` above requires the WHOLE SITE to be untouched by both
  // single arms, and that is too coarse for the shape the law actually
  // provides. A controlled transpose swaps two roles that may live on the SAME
  // site as the control's own cause: the target BIT is blank in both single
  // arms while the site is not. Site granularity scores that as nothing.
  //
  // It also removes the need for the cause-site exclusion rule: the injected
  // bits are changed by their own single arm, so `blank` excludes them
  // automatically rather than by a hand-written condition that could be wrong.
  //
  // ONE CONSTRUCTED BIT IS THE CAPABILITY FLOOR. This is that number.
  std::size_t built_bits = 0u;
  std::size_t built_bit_sites = 0u;

  // ⛔⛔ TRANSPORT OR CONSTRUCTION? THE QUESTION THIS ARC ALREADY GOT WRONG ONCE.
  //
  // The grounded-reply path was closed by measuring that every emitted byte was
  // the byte its route already held -- it TRANSPORTED and constructed nothing.
  // A CONTROLLED TRANSPOSE is vulnerable to exactly that objection: with the
  // control set it SWAPS source and target, so the target's new content is the
  // source's old content and the source is left empty. Neither cause alone
  // writes the target, so the strict floor is satisfied -- and it would still
  // be a gated MOVE rather than a computed function of two causes.
  //
  // The discriminator is whether the CAUSES SURVIVE. External review's own form
  // makes this explicit: `(A,B,O) |-> (A,B, O XOR f(A,B))` leaves A and B
  // resident, which is precisely why it destroys nothing. A swap does not.
  //
  //     source_retained = the first cause's injected bits are STILL SET at
  //                       their site in the joint world
  //
  // ⭐ A built bit with `source_retained = false` is a MOVE and must not be
  // reported as construction. One with `source_retained = true` is a genuine
  // two-cause write that cost neither cause its matter.
  bool source_retained = false;
  bool source_probe_valid = false;

  // ⭐⭐⭐ THE FOUR-ROLE LEDGER, AND WHY `source_retained` ALONE WAS THE WRONG
  // QUESTION. External review located the modelling error exactly: in the
  // controlled-transpose probe, cause A WAS the swap's source bit, so
  // `retained = 0` is entailed by the opcode rather than discovered. Asking
  // "did the source stay after being swapped" is tautological.
  //
  // The right question needs a FOURTH role -- generic feedstock:
  //
  //     (A, B, R=1, O=0)  <->  (A, B, R=0, O=1)     controlled by f(A,B)
  //
  // Both causes REMAIN, the output becomes occupied, and a generic resource
  // quantum is consumed. Occupied-channel count is unchanged, so this is legal
  // under exact conservation -- while the old three-register form
  // `(A,B,O=0) -> (A,B,O=1)` is NOT, since it creates inventory.
  //
  // ⛔ THAT SPLITS THE TARGET IN TWO, and only one half is excluded:
  //   Target A -- literal matter creation, nothing else loses a quantum:
  //               structurally impossible under conservation. STOP CHASING IT.
  //   Target B -- two retained causes, generic feedstock consumed, a jointly
  //               selected output appears that was not a pre-encoded cause
  //               payload: NOT EXCLUDED. This is the correct target.
  //
  // `both_causes_retained` plus `built_bits_offsite` is the witness: if both
  // causes still hold their injected bits and a site that is neither of them
  // gained a jointly-caused bit, conservation forces the matter to have come
  // from somewhere that is not a cause.
  bool second_retained = false;
  bool both_causes_retained = false;
  std::size_t built_bits_offsite = 0u;

  // ⛔ CHECK B, THE COLUMN THIS ARC KEPT SKIPPING. A positive is not a
  // construction until the matter is accounted for: if the output gained
  // occupied channels and NOTHING lost any, the claim would violate exact
  // conservation and something in the measurement is wrong. If the totals
  // balance, the gained matter came from resident feedstock -- which is the
  // conservative constructor -- and the causes are shown to have survived
  // separately.
  std::int64_t occupied_delta = 0;      // both-arm total minus baseline total
  // The three terms Check B was missing. `accounted` is the identity exact
  // conservation actually asserts: D_C + D_U == I.
  std::int64_t initial_arm_delta = 0;   // I  -- the arms are seeded differently
  std::int64_t cause_sector_delta = 0;  // D_C -- the excluded cause bits
  bool accounted = false;
  std::size_t feedstock_lost_bits = 0u;  // bits lost at sites that are not the causes

  // Chebyshev distance from a constructed site to the NEARER cause. A junction
  // is a local object; a constructed site far from both causes would mean the
  // interaction is not the local one this file claims to be measuring.
  CoordinateComponent max_constructed_radius = 0;

  // ⛔ PROBE VALIDITY, and it can fail. `influence` subtracts the injected
  // difference, so it is zero exactly when the law relayed the perturbation
  // untouched. If either cause was merely carried, the pair was never really
  // presented to the law and a null N says nothing about the law.
  bool first_reached = false;
  bool second_reached = false;

  // ⛔ REACH, AND WHY A ROW WITHOUT IT IS NOT EVIDENCE OF ABSENCE. Each is the
  // largest Chebyshev distance from a cause to any site that cause alone
  // changed -- the measured extent of its own influence under this exact step.
  //
  // Two causes cannot interact at all unless their influences overlap, so a row
  // whose separation exceeds `first_radius + second_radius` is GUARANTEED to
  // report zero, whatever the law does. Counting such rows as nulls would be
  // the classic vacuous control: a test that cannot fail is not a test. They
  // are excluded from the search and reported separately as out-of-reach.
  //
  // These are MEASURED from the arms themselves, never a constant. The first
  // version of this file swept separations 1..8 at a single superstep and read
  // the resulting zeros as a bounded negative; most of those rows were simply
  // out of reach, and the "bound" was an artifact of the geometry.
  CoordinateComponent first_radius = 0;
  CoordinateComponent second_radius = 0;

  // ⛔⛔ THE SUFFICIENT CONDITION, AND THE ONE THAT DECIDES WHETHER A NULL MEANS
  // ANYTHING. Radii summing past the separation is only NECESSARY for the two
  // causes to interact -- their influence regions can be disjoint shells that
  // pass each other without ever sharing a site. On such a row `N` is zero by
  // set theory, not by the law.
  //
  // `overlap_sites` is the number of sites BOTH causes changed on their own.
  // A row with zero overlap is the purest form of the vacuous control this
  // project keeps rediscovering: it cannot fail, so its null is not evidence.
  // The verdict below is computed over overlapping rows ONLY, and the
  // non-overlapping count is printed rather than quietly folded in.
  std::size_t overlap_sites = 0u;

  // ⛔⛔ AND ONE LEVEL DEEPER: DID THE TWO CAUSES TOUCH THE SAME BITS THERE?
  //
  // `overlap_sites` counts sites both causes changed. That is still not enough
  // to make a null meaningful. If cause A moves only bits in one field and
  // cause B only bits in another, the two never contend for anything, and their
  // effects compose additively BY CONSTRUCTION -- `N` is zero for the same
  // reason a disjoint influence region gives zero, just one level down.
  //
  // `contended_sites` counts overlap sites where the two single-cause bit masks
  // INTERSECT. Those are the only places a conditional update could ever have
  // had two competing causes to condition on, and they are the only rows a
  // bounded negative may be built from.
  std::size_t contended_sites = 0u;
  SiteWord widest_contended_mask = 0u;

  // The two single-cause difference-set sizes, kept so a zero overlap can be
  // told apart from an empty probe.
  std::size_t first_sites = 0u;
  std::size_t second_sites = 0u;

  [[nodiscard]] bool in_reach(CoordinateComponent separation) const {
    return separation <= first_radius + second_radius;
  }
  // ⛔🔴 `contended_sites` IS NOT A VALID ELIGIBILITY CRITERION AND MUST NOT BE
  // USED AS ONE. It was introduced as "the two causes must contend for the same
  // bit", and external review refuted it with a construction the criterion
  // rejects outright:
  //
  //     (S, C, O) |-> (S, C, O XOR (S AND C))
  //
  // Cause A moves the source bit S, cause B moves the control bit C, neither
  // touches the other's bit, their single-cause masks NEVER intersect -- and
  // together they write a blank output O. A controlled transpose is the same
  // shape: control off leaves S and O alone, control on moves S into O. That
  // satisfies the strict construction floor with zero contention.
  //
  // So same-bit contention is neither necessary (a gate can read distinct
  // controls and write a third bit) nor sufficient (two causes can touch one
  // bit with no gate reading them as two causes). The valid criterion is
  // GATE-LEVEL JOINT ELIGIBILITY -- some gate instance whose read tuple is
  // changed by both causes and whose truth function has a nonzero mixed
  // derivative there -- which this file does not yet compute.
  //
  // The field is kept and reported because it is informative, never gating.
  [[nodiscard]] bool can_couple() const { return overlap_sites != 0u; }
};

namespace detail {

[[nodiscard]] inline bool influence_present(const ReferenceLattice& base,
                                            const ReferenceLattice& evolved_base,
                                            const ReferenceLattice& evolved_arm,
                                            const Z3Coordinate& site,
                                            SiteWord perturbed_word) {
  const SiteWord mask = base.read(site) ^ perturbed_word;
  const payload_difference::CoordinateSet differing =
      payload_difference::difference_sites(evolved_base, evolved_arm);
  for (const Z3Coordinate& coordinate : differing) {
    SiteWord residual =
        evolved_base.read(coordinate) ^ evolved_arm.read(coordinate);
    if (coordinate == site) residual ^= mask;  // subtract the INJECTED change
    if (residual != 0u) return true;
  }
  return false;
}

[[nodiscard]] inline CoordinateComponent chebyshev(const Z3Coordinate& a,
                                                   const Z3Coordinate& b) {
  const CoordinateComponent dx =
      payload_difference::detail::abs_component(a.x - b.x);
  const CoordinateComponent dy =
      payload_difference::detail::abs_component(a.y - b.y);
  const CoordinateComponent dz =
      payload_difference::detail::abs_component(a.z - b.z);
  return dx > dy ? (dx > dz ? dx : dz) : (dy > dz ? dy : dz);
}

}  // namespace detail

// Present two causes to the law separately and together, and census the residue
// that only the joint presentation produces.
//
// `Step` is any callable taking `ReferenceLattice&` -- `apply_superstep` for the
// whole law, or a lambda closing over one `LawFactor` to resolve WHICH factor
// carries a junction. All four worlds are ordinary lattice states reached by
// ordinary writes, so nothing here restricts, truncates or inverts anything.
template <typename Step>
[[nodiscard]] JointReading measure_joint(const ReferenceLattice& base,
                                         const Z3Coordinate& first,
                                         SiteWord first_word,
                                         const Z3Coordinate& second,
                                         SiteWord second_word,
                                         Step step) {
  ReferenceLattice neither = base;
  ReferenceLattice only_first = base;
  ReferenceLattice only_second = base;
  ReferenceLattice both = base;

  only_first.write(first, first_word);
  only_second.write(second, second_word);
  both.write(first, first_word);
  both.write(second, second_word);

  step(neither);
  step(only_first);
  step(only_second);
  step(both);

  JointReading out;
  out.first_reached =
      detail::influence_present(base, neither, only_first, first, first_word);
  out.second_reached =
      detail::influence_present(base, neither, only_second, second, second_word);

  for (const Z3Coordinate& site :
       payload_difference::difference_sites(neither, only_first)) {
    const CoordinateComponent reach = detail::chebyshev(site, first);
    if (reach > out.first_radius) out.first_radius = reach;
  }
  const payload_difference::CoordinateSet second_touched =
      payload_difference::difference_sites(neither, only_second);
  for (const Z3Coordinate& site : second_touched) {
    const CoordinateComponent reach = detail::chebyshev(site, second);
    if (reach > out.second_radius) out.second_radius = reach;
  }
  {
    const SiteWord injected = SiteWord(base.read(first) ^ first_word);
    out.source_probe_valid = injected != 0u;
    // Same correction as the mask form below: `injected` names FLIPPED bits, so
    // retention is "still differs from baseline in those positions".
    out.source_retained =
        out.source_probe_valid &&
        ((SiteWord(both.read(first) ^ base.read(first)) & injected) == injected);
    const SiteWord injected_second = SiteWord(base.read(second) ^ second_word);
    out.second_retained =
        injected_second != 0u &&
        ((SiteWord(both.read(second) ^ base.read(second)) & injected_second) ==
         injected_second);
    out.both_causes_retained = out.source_retained && out.second_retained;
  }

  const payload_difference::CoordinateSet first_touched =
      payload_difference::difference_sites(neither, only_first);
  out.first_sites = first_touched.size();
  out.second_sites = second_touched.size();
  for (const Z3Coordinate& site : first_touched) {
    if (second_touched.count(site) == 0u) continue;
    ++out.overlap_sites;
    const SiteWord baseline = neither.read(site);
    const SiteWord contended =
        (only_first.read(site) ^ baseline) & (only_second.read(site) ^ baseline);
    if (contended == 0u) continue;
    ++out.contended_sites;
    if (contended != 0u && out.widest_contended_mask == 0u)
      out.widest_contended_mask = contended;
  }

  // Every site that can carry a nonzero N differs between `both` and `neither`
  // or between one single arm and `neither`; the union of those difference sets
  // is a superset, and iterating it is exact.
  payload_difference::CoordinateSet candidates =
      payload_difference::difference_sites(neither, both);
  for (const Z3Coordinate& site :
       payload_difference::difference_sites(neither, only_first))
    candidates.insert(site);
  for (const Z3Coordinate& site :
       payload_difference::difference_sites(neither, only_second))
    candidates.insert(site);

  for (const Z3Coordinate& site : candidates) {
    const SiteWord baseline = neither.read(site);
    const SiteWord with_first = only_first.read(site);
    const SiteWord with_second = only_second.read(site);
    const SiteWord with_both = both.read(site);

    const SiteWord residue =
        with_both ^ with_first ^ with_second ^ baseline;
    if (residue != 0u) ++out.joint_sites;

    const SiteWord blank =
        SiteWord(~(with_first ^ baseline) & ~(with_second ^ baseline));
    const SiteWord built = SiteWord(blank & (with_both ^ baseline));
    if (built != 0u) {
      ++out.built_bit_sites;
      out.built_bits += std::size_t(std::popcount(std::uint32_t(built)));
      // Offsite: the output is neither cause's own site, so it cannot be the
      // injected perturbation reappearing under another name.
      if (!(site == first) && !(site == second))
        out.built_bits_offsite +=
            std::size_t(std::popcount(std::uint32_t(built)));
    }

    const bool cause_site = site == first || site == second;
    const bool first_silent = with_first == baseline;
    const bool second_silent = with_second == baseline;
    const bool joint_writes = with_both != baseline;
    if (cause_site || !first_silent || !second_silent || !joint_writes) continue;

    ++out.constructed_sites;
    out.constructed_bits +=
        std::size_t(std::popcount(std::uint32_t(with_both ^ baseline)));
    if (base.read(site) == kQuiescentWord) ++out.constructed_from_quiescent;
    const CoordinateComponent nearest =
        detail::chebyshev(site, first) < detail::chebyshev(site, second)
            ? detail::chebyshev(site, first)
            : detail::chebyshev(site, second);
    if (nearest > out.max_constructed_radius)
      out.max_constructed_radius = nearest;
  }
  return out;
}

// ⭐⭐⭐ BOTH CAUSES IN ONE WORD. THE PROBE THE TWO-COORDINATE API CANNOT EXPRESS.
//
// `apply_k_site()` traverses each occupied coordinate independently and applies
// `apply_site_word_forward()` to that ONE word. It NEVER jointly reads two
// different coordinates. Every probe above places the two causes at two
// distinct coordinates, so the site factor's `eligible = 0` is ENTAILED BY THE
// PRESENTATION GEOMETRY -- not a fact about the factor. External review caught
// this after the two-coordinate sweeps had already been landed.
//
// The same correction repairs a granularity error that crept back in.
// `built_bits_offsite` requires the output COORDINATE to differ from both cause
// COORDINATES, which contradicts the per-bit floor established earlier: a cause
// bit, a control bit, feedstock and output may all live in different FIELDS of
// the SAME SiteWord. `pair_rotor` is exactly that shape -- it reads one tetrad
// as a control pattern and, at popcount two, swaps two bits of another tetrad
// in the same word.
//
// So here the causes are disjoint BIT MASKS on ONE site, and "the output is not
// a cause" is enforced per bit by excluding the union of the two masks instead
// of by excluding coordinates.
template <typename Step>
[[nodiscard]] JointReading measure_joint_same_site(const ReferenceLattice& base,
                                                   const Z3Coordinate& site,
                                                   SiteWord first_mask,
                                                   SiteWord second_mask,
                                                   Step step) {
  JointReading out;
  if ((first_mask & second_mask) != 0u || first_mask == 0u || second_mask == 0u)
    return out;  // overlapping masks are not two independent causes

  const SiteWord baseline_word = base.read(site);
  ReferenceLattice neither = base;
  ReferenceLattice only_first = base;
  ReferenceLattice only_second = base;
  ReferenceLattice both = base;
  only_first.write(site, SiteWord(baseline_word ^ first_mask));
  only_second.write(site, SiteWord(baseline_word ^ second_mask));
  both.write(site, SiteWord(baseline_word ^ first_mask ^ second_mask));

  step(neither);
  step(only_first);
  step(only_second);
  step(both);

  out.source_probe_valid = true;
  // ⛔ AN XOR MASK NAMES BITS THAT WERE FLIPPED, NOT BITS THAT MUST BE SET.
  // The old test demanded every masked bit be SET afterwards. A cause applied by
  // XOR may CLEAR a bit -- and every carrier bit IS set in kQ, so XOR clears it
  // and "still set" is the opposite of retention for exactly those causes.
  //
  // ⭐ Retention means the site still differs from baseline in the same
  // positions, whichever direction each flip went: set (0->1) still 1, cleared
  // (1->0) still 0. One expression covers both.
  //
  // ⚠ The old form was CONSERVATIVE -- it could only MISS a witness, never
  // manufacture one -- so this can only raise the count, and a FALL would refute
  // that reasoning.
  const SiteWord retained_now = SiteWord(both.read(site) ^ baseline_word);
  out.source_retained = (retained_now & first_mask) == first_mask;
  out.second_retained = (retained_now & second_mask) == second_mask;
  out.both_causes_retained = out.source_retained && out.second_retained;

  payload_difference::CoordinateSet candidates =
      payload_difference::difference_sites(neither, both);
  for (const Z3Coordinate& other :
       payload_difference::difference_sites(neither, only_first))
    candidates.insert(other);
  for (const Z3Coordinate& other :
       payload_difference::difference_sites(neither, only_second))
    candidates.insert(other);

  const SiteWord cause_bits = SiteWord(first_mask | second_mask);
  // ⛔ THE FIRST VERSION OF THIS LEDGER WAS WRONG AND CHECK B CAUGHT IT. It
  // compared whole-lattice occupied totals between `both` and `neither`, which
  // includes THE INJECTED CAUSE BITS THEMSELVES -- the two arms differ by those
  // bits before the law even runs. Every witness therefore read a nonzero
  // delta, which looked like a conservation violation and was an accounting
  // error. The balance that matters is between what NON-CAUSE matter GAINED and
  // what NON-CAUSE matter LOST.
  for (const Z3Coordinate& where : candidates) {
    const SiteWord baseline = neither.read(where);
    const SiteWord with_first = only_first.read(where);
    const SiteWord with_second = only_second.read(where);
    const SiteWord with_both = both.read(where);
    if ((with_both ^ with_first ^ with_second ^ baseline) != 0u)
      ++out.joint_sites;
    const SiteWord blank =
        SiteWord(~(with_first ^ baseline) & ~(with_second ^ baseline));
    SiteWord built = SiteWord(blank & (with_both ^ baseline));
    if (where == site) built &= ~cause_bits;  // the output is never a cause bit
    if (built == 0u) continue;
    ++out.built_bit_sites;
    const std::size_t count =
        std::size_t(std::popcount(std::uint32_t(built)));
    out.built_bits += count;
    out.built_bits_offsite += count;  // "not a cause" is enforced per BIT here
  }
  std::size_t gained_noncause = 0u;
  for (const Z3Coordinate& where : candidates) {
    SiteWord lost = SiteWord(neither.read(where) & ~both.read(where));
    SiteWord gained = SiteWord(both.read(where) & ~neither.read(where));
    if (where == site) {
      lost &= ~cause_bits;
      gained &= ~cause_bits;
    }
    out.feedstock_lost_bits += std::size_t(std::popcount(std::uint32_t(lost)));
    gained_noncause += std::size_t(std::popcount(std::uint32_t(gained)));
  }
  // ⭐⭐ CHECK B COMPARED TWO DIFFERENTLY SEEDED ARMS AND DEMANDED THE REMAINDER
  // BALANCE TO ZERO. Its own verdict called `unbalanced_on_all=14` a sign the
  // measurement was wrong. It was, but not in the way the message says: the
  // imbalance is LAWFUL and expected.
  //
  // Before the step, `both` already differs from `neither` at the junction site
  // by the injected causes. Call that INITIAL ARM DIFFERENCE `I`. Exact
  // conservation carries `I` to the post-state, so with the final difference
  // split into the excluded cause sector `D_C` and everything else `D_U`:
  //
  //     D_C + D_U = I           and Check B required D_U == 0
  //
  // which holds only if `D_C == I`. Nothing guarantees that, so a zero test on
  // `D_U` alone tests a proposition conservation never asserts.
  //
  // ⛔ TWO ESCAPES CHECKED AND REJECTED BY READING, not assumed:
  //   * switching to ΔN_Q changes NOTHING -- `occupied_channels` IS popcount, so
  //     gained-minus-lost over a fixed site set already equals
  //     ΔN_Q(both) - ΔN_Q(neither); the per-site kQ baseline cancels in a
  //     difference.
  //   * there is no missing SPATIAL boundary term -- `candidates` is the union of
  //     difference sites over all arms, so the neither/both iteration is exact.
  //
  // ⇒ the missing terms are `I` and `D_C`, and both are recorded here so the
  // caller can test the identity that conservation DOES assert.
  std::size_t initial_gained = 0u;
  std::size_t initial_lost = 0u;
  std::size_t cause_sector_gained = 0u;
  std::size_t cause_sector_lost = 0u;
  for (const Z3Coordinate& where : candidates) {
    const SiteWord before_neither = base.read(where);
    const SiteWord before_both =
        where == site ? SiteWord(base.read(where) ^ cause_bits) : before_neither;
    initial_lost += std::size_t(
        std::popcount(std::uint32_t(before_neither & ~before_both)));
    initial_gained += std::size_t(
        std::popcount(std::uint32_t(before_both & ~before_neither)));
    if (where != site) continue;
    const SiteWord lost_here =
        SiteWord(neither.read(where) & ~both.read(where) & cause_bits);
    const SiteWord gained_here =
        SiteWord(both.read(where) & ~neither.read(where) & cause_bits);
    cause_sector_lost += std::size_t(std::popcount(std::uint32_t(lost_here)));
    cause_sector_gained += std::size_t(std::popcount(std::uint32_t(gained_here)));
  }
  out.initial_arm_delta =
      std::int64_t(initial_gained) - std::int64_t(initial_lost);
  out.cause_sector_delta =
      std::int64_t(cause_sector_gained) - std::int64_t(cause_sector_lost);
  out.occupied_delta =
      std::int64_t(gained_noncause) - std::int64_t(out.feedstock_lost_bits);
  // ⭐ THE IDENTITY CONSERVATION ACTUALLY ASSERTS. This is what a caller should
  // test instead of `occupied_delta == 0`.
  out.accounted = (out.cause_sector_delta + out.occupied_delta) ==
                  out.initial_arm_delta;
  return out;
}

// The whole law, one superstep -- the composition of all twelve forward
// factors, which is what the organism actually runs.
[[nodiscard]] inline JointReading measure_joint_superstep(
    const ReferenceLattice& base, const Z3Coordinate& first,
    SiteWord first_word, const Z3Coordinate& second, SiteWord second_word) {
  return measure_joint(base, first, first_word, second, second_word,
                       [](ReferenceLattice& world) { apply_superstep(world); });
}

// One factor in isolation -- resolves WHICH part of the law carries a junction,
// so a positive is attributable rather than merely present.
[[nodiscard]] inline JointReading measure_joint_factor(
    const ReferenceLattice& base, const Z3Coordinate& first,
    SiteWord first_word, const Z3Coordinate& second, SiteWord second_word,
    LawFactor factor) {
  return measure_joint(
      base, first, first_word, second, second_word,
      [factor](ReferenceLattice& world) { apply_factor(world, factor, false); });
}

}  // namespace substrate::bcc32::joint_causation
