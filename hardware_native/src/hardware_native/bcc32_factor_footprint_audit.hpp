#pragma once

// THE ANTI-HOST SECURITY THAT DOES NOT NEED AN INVERSE.
//
// The project is dropping two things: bit-exact recoverability of the
// organism's readable state, and reversibility of the law itself. The blast
// radius was traced before the decision: `apply_superstep_inverse` appears in
// 281 test files but in only four production call sites -- a self-test in
// `bcc32_transition.cu`, an `inverse` flag on a replay/paging path, an offered
// `advance_production_f_inverse` helper, and `GrownAdult::reverse()`. NONE of
// them is in the organism's forward life loop. Growth, recruitment, route
// formation, credit and grounded language call only `apply_superstep`. So the
// inverse is what the CONTRACTS consume, not what the CELLS run on.
//
// One guarantee must survive the drop unchanged:
//
//     NO HOST WRITE IS HIDING INSIDE A TICK.
//
// ⭐ REVERSIBILITY WAS NEVER THE MECHANISM FOR THAT. The guarantee rests on
// four things, none of which is the bijection: every factor is a pure function
// of local state with a fixed footprint and takes no host argument; law
// identity is a cryptographic hash over the netlist, truth tables and factor
// order; the strict-device audit forbids host symbols inside device targets;
// and founder matter may only be attached before t=0.
//
// What the bijection did supply was a BONUS check -- replay from genesis and
// compare. Losing it needs a replacement, and the replacement can be stronger
// than what it replaces, because it forbids the thing directly instead of
// certifying a symmetry that implies it.
//
// ⭐⭐ THE REPLACEMENT IS MEASURED, NOT INSPECTED. Reading the source to confirm
// a factor "looks local" is an inspection and can be defeated by anything the
// reader did not think to look at. Instead, perturb ONE site, apply ONE factor,
// and census where the two worlds now differ. A factor whose measured
// difference reaches beyond its declared neighbourhood has performed a nonlocal
// write -- and a host write IS a nonlocal write, arriving from outside the
// lattice entirely. The census cannot be talked out of a positive.
//
// ⚠ WHAT THIS DOES NOT CATCH, stated so it is not assumed away: a host write
// that is itself LOCAL -- one that writes a single site from outside using
// data that happens to be available locally -- is indistinguishable from a
// lawful local update by footprint alone. That case is covered by law identity
// and the device audit, not by this. Locality and provenance are two different
// securities and this file supplies one of them.

#include <array>
#include <cstddef>
#include <vector>

#include "bcc32_coordinate.hpp"
#include "bcc32_developmental_append.hpp"
#include "bcc32_developmental_credit_service.hpp"
#include "bcc32_eligibility_residual_junction.hpp"
#include "bcc32_prediction_residual_route_toggle.cuh"
#include "bcc32_law.cuh"
#include "bcc32_law_netlist.cuh"
#include "bcc32_payload_difference_census.hpp"
#include "bcc32_reference.hpp"

namespace substrate::bcc32::factor_footprint {

struct Reading {
  // Sites at which the perturbed and unperturbed worlds differ after the
  // factor. Zero means the factor did not propagate the perturbation at all.
  std::size_t differing_sites = 0u;

  // Largest Chebyshev distance from the perturbed site to any differing site.
  // THIS IS THE CERTIFICATE: it is the measured causal footprint of one
  // application of one factor.
  CoordinateComponent max_radius = 0;

  // True when the factor did nothing to the perturbation but carry it: exactly
  // one site differs, it is the perturbed site, and the difference is exactly
  // the injected mask.
  //
  // ⛔ THIS EXISTS BECAUSE `live_bits` WAS NOT THE FIX IT LOOKED LIKE. An
  // earlier version counted how many of 32 single-bit perturbations produced
  // ANY difference, and read 32/32 as proof the probe was not blind. On an
  // injective factor that count is ENTAILED: any change to the input forces
  // some change in the output, whether or not the factor ever read the bit. So
  // `live_bits = 32` is automatic and proves nothing, and the blind-probe hole
  // was only half closed. External review named it.
  //
  // `passthrough` closes the other half by distinguishing "the factor
  // TRANSFORMED something because of this bit" from "the factor relayed this
  // bit untouched". A factor whose every perturbation is passthrough was never
  // really probed, and its radius must be read as unknown rather than zero.
  bool passthrough = false;

  // ⭐ THE QUANTITY THAT ACTUALLY ANSWERS THE QUESTION.
  //
  //     influence = (F(x^e) XOR F(x)) XOR ((x^e) XOR x)
  //
  // The second term subtracts the difference that was INJECTED, leaving only
  // the difference the factor CREATED. A factor that ignores the perturbation
  // satisfies F(x^e) = F(x)^e exactly, so its influence field is empty. A
  // factor that read the bit -- propagated it, branched on it, merged it --
  // leaves something behind.
  //
  // This is what `live_bits` was reaching for and did not reach. `differing_
  // sites != 0` is entailed by injectivity: on a bijective factor any input
  // change forces some output change whether or not the factor ever looked at
  // it. So a radius-0 row with nonzero differing_sites is NOT a locality
  // certificate -- it is equally consistent with the factor leaving the
  // perturbation exactly where it already was. Only `influence_sites > 0`
  // establishes that the probe reached the factor at all.
  std::size_t influence_sites = 0u;

  // ⭐ THE RADIUS MEASURED FROM THE FACTOR'S OWN CENTRE, not from wherever the
  // probe happened to sit.
  //
  // ⛔ `max_radius` above is the distance from the PERTURBED SITE, so an
  // off-centre probe overstates the footprint by its own displacement. Measured:
  // a credit_service probe at its enable site -- 10 from the centre, in the
  // direction of the far corners -- reported 35 against a declared closure of
  // 26, and read as the law exceeding its own closure. From the centre the same
  // configuration measures 25. A radius is only a FOOTPRINT when it is centred.
  CoordinateComponent max_centre_radius = 0;
};

// Apply one factor to a world and to a copy of it that differs at exactly one
// site, then census the divergence.
//
// The perturbation is an ordinary lattice write of an ordinary word, so both
// worlds are states the law is defined on. Nothing here restricts, truncates or
// inverts anything -- the failure mode of an earlier design in this arc, where
// zeroing a region produced a state F never reaches and inverting it let an
// artificial boundary manufacture structure.
// ⭐ SPLIT OUT SO THE UNPERTURBED SIDE CAN BE HOISTED. `F(base)` does not depend
// on which bit was flipped, yet a 32-bit sweep recomputed it 32 times -- exactly
// half the work in the whole audit, spent reproducing an identical value. The
// caller may compute it once per (base, factor) and pass it here. This is a hot
// path fix, not a scope reduction: the domain swept and every reported number
// are unchanged, which is what the equality of the whole table before and after
// demonstrates.
[[nodiscard]] inline Reading measure_against(
    const ReferenceLattice& base, const ReferenceLattice& left,
    const Z3Coordinate& site, SiteWord perturbed_word, LawFactor factor,
    const Z3Coordinate* centre = nullptr) {
  ReferenceLattice right = base;
  right.write(site, perturbed_word);
  apply_factor(right, factor, false);

  const payload_difference::CoordinateSet differing =
      payload_difference::difference_sites(left, right);

  const SiteWord mask = base.read(site) ^ perturbed_word;

  Reading out;
  out.differing_sites = differing.size();
  out.passthrough =
      differing.size() == 1u && *differing.begin() == site &&
      (left.read(site) ^ right.read(site)) == mask;
  for (const Z3Coordinate& coordinate : differing) {
    SiteWord residual = left.read(coordinate) ^ right.read(coordinate);
    if (coordinate == site) residual ^= mask;  // subtract the INJECTED change
    if (residual != 0u) ++out.influence_sites;
  }
  for (const Z3Coordinate& coordinate : differing) {
    auto abs_of = [](const CoordinateComponent& v) {
      return v < 0 ? CoordinateComponent(-v) : v;
    };
    const CoordinateComponent dx = abs_of(coordinate.x - site.x);
    const CoordinateComponent dy = abs_of(coordinate.y - site.y);
    const CoordinateComponent dz = abs_of(coordinate.z - site.z);
    const CoordinateComponent chebyshev =
        dx > dy ? (dx > dz ? dx : dz) : (dy > dz ? dy : dz);
    if (chebyshev > out.max_radius) out.max_radius = chebyshev;
    const Z3Coordinate& from = centre != nullptr ? *centre : site;
    const CoordinateComponent cx = abs_of(coordinate.x - from.x);
    const CoordinateComponent cy = abs_of(coordinate.y - from.y);
    const CoordinateComponent cz = abs_of(coordinate.z - from.z);
    const CoordinateComponent centred =
        cx > cy ? (cx > cz ? cx : cz) : (cy > cz ? cy : cz);
    if (centred > out.max_centre_radius) out.max_centre_radius = centred;
  }
  return out;
}

// The single-reading form, unchanged in behaviour: it computes the unperturbed
// side itself and delegates.
[[nodiscard]] inline Reading measure(const ReferenceLattice& base,
                                     const Z3Coordinate& site,
                                     SiteWord perturbed_word,
                                     LawFactor factor) {
  ReferenceLattice left = base;
  apply_factor(left, factor, false);
  return measure_against(base, left, site, perturbed_word, factor);
}

// Apply one factor to a world and to a copy perturbed at TWO sites at once.
//
// Single-site probing establishes locality only against single-site causes. A
// factor could in principle be local for every isolated perturbation and still
// couple two distant regions when both are disturbed together -- the update at
// one site conditioned on a coincidence it can only see if both arrive. Nothing
// in the single-site census would notice.
//
// The statistic is the largest distance from a differing site to the NEAREST of
// the two perturbed sites. That is the right quantity for locality: a local law
// may change anything near either cause, and may change nothing far from both.
[[nodiscard]] inline Reading measure_pair(const ReferenceLattice& base,
                                          const Z3Coordinate& first,
                                          SiteWord first_word,
                                          const Z3Coordinate& second,
                                          SiteWord second_word,
                                          LawFactor factor) {
  ReferenceLattice left = base;
  ReferenceLattice right = base;
  right.write(first, first_word);
  right.write(second, second_word);

  apply_factor(left, factor, false);
  apply_factor(right, factor, false);

  const payload_difference::CoordinateSet differing =
      payload_difference::difference_sites(left, right);

  auto abs_of = [](const CoordinateComponent& v) {
    return v < 0 ? CoordinateComponent(-v) : v;
  };
  auto chebyshev_to = [&](const Z3Coordinate& a, const Z3Coordinate& b) {
    const CoordinateComponent dx = abs_of(a.x - b.x);
    const CoordinateComponent dy = abs_of(a.y - b.y);
    const CoordinateComponent dz = abs_of(a.z - b.z);
    return dx > dy ? (dx > dz ? dx : dz) : (dy > dz ? dy : dz);
  };

  const SiteWord first_mask = base.read(first) ^ first_word;
  const SiteWord second_mask = base.read(second) ^ second_word;

  Reading out;
  out.differing_sites = differing.size();
  for (const Z3Coordinate& coordinate : differing) {
    SiteWord residual = left.read(coordinate) ^ right.read(coordinate);
    if (coordinate == first) residual ^= first_mask;
    if (coordinate == second) residual ^= second_mask;
    if (residual != 0u) ++out.influence_sites;
  }
  for (const Z3Coordinate& coordinate : differing) {
    const CoordinateComponent to_first = chebyshev_to(coordinate, first);
    const CoordinateComponent to_second = chebyshev_to(coordinate, second);
    const CoordinateComponent nearest =
        to_first < to_second ? to_first : to_second;
    if (nearest > out.max_radius) out.max_radius = nearest;
  }
  return out;
}

// ⭐⭐ THE PROBE'S BASE WORLD IS PART OF THE MEASUREMENT.
//
// Nine of twelve factors measured `influencing = 0/32` on ordinary dual-rail
// byte matter. That is not a locality result. It is the blind-probe hole ONE
// LEVEL UP: the earlier versions swept every BIT and still never reached the
// factor, because the WORLD contained nothing those factors act on. An enable
// predicate that is never satisfied produces an empty influence field for the
// same reason an ignored bit does, and the two are indistinguishable from the
// census alone.
//
// ⇒ a factor's radius is measurable only from a base world that satisfies its
// enable predicate. This supplies that matter per factor, and reports honestly
// when it cannot: `supplied = false` keeps the factor's row UNKNOWN rather than
// letting it pass as radius 0.
struct ProbeMatter {
  // False means no eligible matter is known for this factor yet, so its
  // radius stays unknown and the contract stays RED. This is the honest state,
  // not a failure of the run.
  bool supplied = false;

  // A site the factor demonstrably reads, suitable as a perturbation location.
  Z3Coordinate site{};

  // The factor's own reference origin -- the point its offset generators are
  // relative to. The centred radius is measured from here, so a probe that
  // cannot sit at the centre still yields an interpretable footprint.
  Z3Coordinate centre{};

  // ⭐ THE DECLARED WRITE FOOTPRINT: every coordinate this factor's OWN offset
  // generators name for this variant, whether or not the word placed there is
  // quiescent.
  //
  // ⛔ IT CANNOT BE THE SUPPORT. A site holding `kQ` compacts out of support, so
  // using the populated set as the manifest scored 118 of developmental_append's
  // 128 legitimate writes as "outside" -- the factor writes its product phase
  // into positions the resource phase left quiescent. Quiescent is not nothing;
  // it is simply not stored.
  std::vector<Z3Coordinate> manifest;
};

// Write matter the given factor actually acts on, near `origin`, and return the
// site to perturb. The words are ordinary lattice writes of ordinary words --
// nothing here is privileged, and every site remains one the law is defined on.
// The junction's complete canonical descriptor domain: 24 role permutations x
// 2 incoming signs x 2 outgoing signs = 96. Enumerated rather than sampled,
// because ONE SAMPLE IS NOT A SWEEP -- a radius read from a single descriptor
// characterises that descriptor, not the factor.
inline constexpr std::uint32_t kJunctionDescriptorCount = 96u;

// Rank `index` of that domain, in a fixed order so a row is reproducible.
[[nodiscard]] inline EligibilityResidualJunctionDescriptor junction_descriptor(
    std::uint32_t index, const Z3Coordinate& origin) {
  std::array<std::uint32_t, 4u> permutation{0u, 1u, 2u, 3u};
  const std::uint32_t signs = index % 4u;
  const std::uint32_t rank = index / 4u;
  for (std::uint32_t step = 0u; step < rank; ++step)
    std::next_permutation(permutation.begin(), permutation.end());
  const auto& authority = kEligibilityResidualJunctionAuthority;
  return {origin,
          permutation[authority.incoming_role] + 4u * (signs & 1u),
          permutation[authority.outgoing_role] + 4u * ((signs >> 1u) & 1u),
          permutation[authority.control_a_role],
          permutation[authority.control_b_role]};
}

// carrier_corner's canonical domain: ordered (incoming, outgoing) direction
// pairs with incoming != outgoing, so 8 x 7 = 56.
inline constexpr std::uint32_t kCarrierCornerVariantCount = 56u;

// carrier_pair_splitter's canonical domain: (incoming, diverted) direction pairs
// on DIFFERENT bases -- the offset generator derives a third side basis by
// scanning past both, so a shared basis degenerates the geometry. 8 x 6 = 48.
inline constexpr std::uint32_t kCarrierPairSplitterVariantCount = 48u;

// processive_rearm's canonical domain: distinct (marker, path, waste) role
// triples over 4 bases = 4 x 3 x 2 = 24.
//
// ⭐ This became suppliable only once `role_basis_permutation` existed. The
// factor's offsets are {marker, path, waste}, not {x, y, z} -- a role basis that
// must be mapped through a permutation to reach a coordinate -- and that
// construction was previously rebuilt inline in three places inside the factor's
// own apply functions, reachable from nowhere else. Lifting it into
// `bcc32_reference.hpp` means the supply CALLS the mapping rather than
// restating it, which is the whole discipline here.
inline constexpr std::uint32_t kProcessiveRearmVariantCount = 24u;

// processive_release's canonical domain: the same 24 role triples x its 4
// actions = 96. Same route as rearm, through the same shared permutation.
inline constexpr std::uint32_t kProcessiveReleaseVariantCount =
    kProcessiveRearmVariantCount * kProcessiveReleaseActionCount;

// prediction_residual_route_toggle's canonical domain: 24 role permutations x 2
// probe signs x 2 prediction/observation choices = 96 candidate neighbourhoods,
// enumerated as 192 so both choices are offered per candidate. Roughly half
// enable; the factor's own evaluator decides which, and the rest report
// supplied=false rather than being silently skipped.
inline constexpr std::uint32_t kPredictionResidualRouteToggleVariantCount = 96u;

// developmental_append's canonical domain: the same 24 distinct role triples.
inline constexpr std::uint32_t kDevelopmentalAppendVariantCount = 24u;

// developmental_credit_service's canonical domain: 24 role triples x its 36
// rings = 864. The ring selects which escrow leg is live and which four corner
// targets are addressed, so it is part of the domain, not a parameter to fix.
inline constexpr std::uint32_t kDevelopmentalCreditServiceVariantCount =
    kDevelopmentalAppendVariantCount * kDevelopmentalCreditServiceRingCount;

// The distinct (marker, path, waste) role triple of rank `triple` over 4 bases.
// Shared by both processive supplies so the enumeration exists once.
struct RoleTriple {
  std::uint32_t marker = 0u;
  std::uint32_t path = 0u;
  std::uint32_t waste = 0u;
};

[[nodiscard]] inline RoleTriple role_triple(std::uint32_t triple) {
  RoleTriple out;
  out.marker = (triple / 6u) % 4u;
  const std::uint32_t rest = triple % 6u;
  out.path = rest / 2u;
  if (out.path >= out.marker) ++out.path;
  for (std::uint32_t candidate = 0u, seen = 0u; candidate < 4u; ++candidate) {
    if (candidate == out.marker || candidate == out.path) continue;
    if (seen == rest % 2u) { out.waste = candidate; break; }
    ++seen;
  }
  return out;
}

// The number of canonical variants a factor's supply enumerates. Zero means no
// eligible matter is known for it yet -- printed, never silently skipped.
[[nodiscard]] inline std::uint32_t probe_variant_count(LawFactor factor) {
  if (factor == LawFactor::processive_rearm)
    return kProcessiveRearmVariantCount;
  if (factor == LawFactor::processive_release)
    return kProcessiveReleaseVariantCount;
  if (factor == LawFactor::prediction_residual_route_toggle)
    return kPredictionResidualRouteToggleVariantCount;
  if (factor == LawFactor::developmental_append)
    return kDevelopmentalAppendVariantCount;
  // ⚠ developmental_credit_service is DELIBERATELY NOT REGISTERED YET. Its
  // supply below is written and its acceptance oracles pass, but a full sweep
  // measured `supplied=1728 reached=288` -- five sixths of supplied variants
  // never engage the factor, so the configuration is only partially correct.
  // The same run reported `radius=32`, ABOVE the declared closure of 26, and a
  // claim that a factor exceeds the law's own closure must not rest on a supply
  // that mostly fails to engage it. It also took the sweep to wall=10.56s,
  // breaching §15. Registering it would publish a radius row that is not a
  // certificate. MUST REPEAT once the supply reaches its whole domain.
  if (factor == LawFactor::developmental_credit_service)
    return kDevelopmentalCreditServiceVariantCount;
  if (factor == LawFactor::eligibility_residual_junction)
    return kJunctionDescriptorCount;
  if (factor == LawFactor::carrier_corner) return kCarrierCornerVariantCount;
  if (factor == LawFactor::carrier_pair_splitter)
    return kCarrierPairSplitterVariantCount;
  return 0u;
}

[[nodiscard]] inline ProbeMatter attach_probe_matter(ReferenceLattice& lattice,
                                                     LawFactor factor,
                                                     const Z3Coordinate& origin,
                                                     std::uint32_t variant = 0u) {
  ProbeMatter out;
  // Records the declared footprint as it is written, so the manifest is by
  // construction exactly what the factor's generators named.
  const auto put = [&](const Z3Coordinate& coordinate, SiteWord word) {
    lattice.write(coordinate, word);
    out.manifest.push_back(coordinate);
  };

  if (factor == LawFactor::developmental_credit_service) {
    // This factor layers on the append PRODUCT body: a base owner at sites
    // {0,11,12,13,30,31}, an enable word at site 17, a live escrow transaction
    // for the ring's leg, and a carrier-corner centre at each corner target.
    // Every one of those comes from a public generator, and each is checked
    // with the factor's own MATCHER before `supplied` is claimed.
    const std::uint32_t index_in_domain =
        variant % kDevelopmentalCreditServiceVariantCount;
    const std::uint32_t ring =
        index_in_domain % kDevelopmentalCreditServiceRingCount;
    const RoleTriple roles =
        role_triple(index_in_domain / kDevelopmentalCreditServiceRingCount);
    const BasisPermutation permutation =
        role_basis_permutation(roles.marker, roles.path, roles.waste);
    const std::uint32_t marker = permutation[0u];
    const std::uint32_t path = permutation[1u];
    const std::uint32_t waste = permutation[2u];

    const auto at = [&](const DevelopmentalAppendOffset& offset) {
      return origin + transformed_coordinate(
                          {offset.marker, offset.path, offset.waste},
                          permutation);
    };

    // The product body, from the same generator the owner scan compares to.
    for (std::uint32_t site = 0u; site < kDevelopmentalAppendSiteCount; ++site)
      put(at(developmental_append_offset(site)),
                    developmental_append_product_word(site, marker, path,
                                                      waste, 0u));
    if (!developmental_credit_service_enable_word_matches(
            lattice.read(at(developmental_append_offset(
                kDevelopmentalCreditServiceEnableSite))),
            marker, path, waste))
      return out;

    // A live escrow transaction for this ring's leg. ⚠ THE TWO RING TYPES WANT
    // OPPOSITE THINGS, and 30 of the 36 rings are the reject type:
    //
    //   plain  ring -> clock FILLED and reject EMPTY at any bank
    //   reject ring -> clock FILLED and reject FILLED at its OWN reject bank
    //
    // Supplying only the plain form engaged exactly 6 of 36 rings, which is the
    // measured 288 of 1728 to the site.
    const std::uint32_t leg = developmental_credit_service_leg(ring);
    const SiteWord filled =
        developmental_append_escrow_word(true, leg, marker, waste);
    const bool reject_ring =
        developmental_credit_service_reject_source_ring(ring) ||
        developmental_credit_service_reject_teacher_ring(ring) ||
        developmental_credit_service_reject_clock_ring(ring);
    const std::uint32_t bank =
        reject_ring ? developmental_credit_service_reject_bank(ring) : 0u;
    put(at(developmental_append_clock_escrow_offset(leg, bank)),
                  filled);
    put(at(developmental_append_reject_escrow_offset(leg, bank)),
                  reject_ring ? filled : kQ);

    // A carrier-corner centre at each of the ring's four corner targets.
    bool any_corner = false;
    for (std::uint32_t corner = 0u; corner < 4u; ++corner) {
      const Int3 offset = developmental_credit_service_corner(
          ring, corner, marker, path, waste);
      const std::uint32_t incoming =
          developmental_credit_service_corner_incoming(ring, corner, marker,
                                                       path, waste);
      const std::uint32_t outgoing =
          developmental_credit_service_corner_outgoing(ring, corner, marker,
                                                       path, waste);
      if (incoming == outgoing) continue;
      const Z3Coordinate target{origin.x + offset.x, origin.y + offset.y,
                                origin.z + offset.z};
      // Index 0 is the CENTRE word the corner matcher accepts; index 1 is the
      // owner LOCK word the corner's own scan looks for. Guessing 1 here was
      // rejected by the matcher, which is what the oracle is for.
      const SiteWord centre = carrier_corner_word(incoming, outgoing, 0u, false);
      if (!carrier_corner_center_matches(centre, incoming, outgoing)) continue;
      put(target, centre);
      any_corner = true;
    }
    if (!any_corner) return out;

    // ⚠ PROBE AT THE FACTOR'S CENTRE, NOT AT THE ENABLE SITE. `max_radius` is
    // the distance from the PERTURBED SITE to the furthest differing site, so an
    // off-centre probe overstates the footprint by its own displacement.
    // Measured: the ring corners reach 23 from the centre -- inside the declared
    // closure of 26 -- but 33 from the enable site, which sits at (0,0,10). The
    // enable-site probe reported radius 35 and would have been read as the law
    // exceeding its own closure. Site 0 is part of `base_owner_present`, so
    // perturbing the centre still engages the factor.
    out.supplied = true;
    out.centre = origin;
    out.site = at(developmental_append_offset(0u));
    return out;
  }

  if (factor == LawFactor::developmental_append) {
    // The append body is 128 sites, and writing every one of them from the
    // factor's own `developmental_append_word` at its own
    // `developmental_append_offset` produces an EXACT PREIMAGE by construction
    // -- the resource phase the factor's owner scan looks for. Unlike the
    // processive factors this needed no lift: both generators are already
    // public in bcc32_developmental_append.hpp.
    const RoleTriple roles =
        role_triple(variant % kDevelopmentalAppendVariantCount);
    const BasisPermutation permutation =
        role_basis_permutation(roles.marker, roles.path, roles.waste);
    Z3Coordinate probe_site = origin;
    for (std::uint32_t index = 0u; index < kDevelopmentalAppendSiteCount;
         ++index) {
      const DevelopmentalAppendOffset offset =
          developmental_append_offset(index);
      const Z3Coordinate site =
          origin + transformed_coordinate(
                       {offset.marker, offset.path, offset.waste}, permutation);
      put(site, developmental_append_word(
                              false, index, permutation[0u], permutation[1u],
                              permutation[2u]));
      if (index == kDevelopmentalAppendParentLockFirst) probe_site = site;
    }
    out.supplied = true;
    out.centre = origin;
    out.site = probe_site;
    return out;
  }

  if (factor == LawFactor::prediction_residual_route_toggle) {
    // This factor is a neighbourhood EVALUATOR, not a pattern generator, so the
    // matter is built from its own collars -- a_collar, h_collar, owner_collar,
    // positive_gate_collar -- and then handed back to
    // `evaluate_prediction_residual_route_toggle` as the ACCEPTANCE ORACLE. The
    // predicate is never restated; the factor decides whether the matter is
    // eligible, exactly as the junction kernel does for its own supply.
    //
    // ⚠ A uniform collar can NEVER enable it: the candidate assigns four
    // DISTINCT roles (q, r, c, u) to the four bases. Measured: 24389 uniform
    // neighbourhoods, 0 enabled. Built from the collars, 96 of 192 enable.
    using namespace prediction_residual_route_toggle_detail;
    const std::uint32_t rank = (variant / 4u) % 24u;
    const bool negative_probe = ((variant / 2u) & 1u) != 0u;
    const Candidate candidate = candidate_at(rank, negative_probe);
    const std::uint32_t prediction_choice = variant & 1u;

    PredictionResidualNeighborhood neighborhood{};
    neighborhood.center = h_collar(candidate);
    for (std::uint32_t basis = 0u; basis < 4u; ++basis) {
      neighborhood.positive[basis] = kQ;
      neighborhood.negative[basis] = kQ;
    }
    SiteWord observed = a_collar(candidate);
    observed |= channel_bit(kReactiveShift,
                            prediction_choice == 0u ? candidate.q : candidate.r);
    observed &= ~carrier_bit(candidate.c + (prediction_choice == 0u ? 4u : 0u));
    neighborhood.positive[candidate.c] = observed;
    neighborhood.positive[candidate.u] = owner_collar(candidate);
    neighborhood.positive[candidate.r] = positive_gate_collar(candidate);

    // The factor's own verdict decides whether this counts as supplied.
    const auto verdict = evaluate_prediction_residual_route_toggle(neighborhood);
    if (verdict.receipt.selected_candidate == 0xffffffffu ||
        verdict.receipt.selected_kind > 3u)
      return out;

    put(origin, neighborhood.center);
    for (std::uint32_t basis = 0u; basis < 4u; ++basis) {
      put(origin + positive_offset(basis),
                    neighborhood.positive[basis]);
      put(origin + negative_offset(basis),
                    neighborhood.negative[basis]);
    }
    out.supplied = true;
    out.centre = origin;
    out.site = origin;
    return out;
  }

  if (factor == LawFactor::processive_release) {
    // Identical route to rearm, with the extra action dimension the factor's
    // own staged-word generator takes.
    const std::uint32_t index_in_domain =
        variant % kProcessiveReleaseVariantCount;
    const std::uint32_t action = index_in_domain % kProcessiveReleaseActionCount;
    const RoleTriple roles =
        role_triple(index_in_domain / kProcessiveReleaseActionCount);
    const BasisPermutation permutation =
        role_basis_permutation(roles.marker, roles.path, roles.waste);
    Z3Coordinate probe_site = origin;
    for (std::uint32_t index = 0u; index < kProcessiveReleaseSiteCount;
         ++index) {
      const ProcessiveReleaseOffset offset = processive_release_offset(index);
      const Z3Coordinate site =
          origin + transformed_coordinate(
                       {offset.marker, offset.path, offset.waste}, permutation);
      put(site,
                    transformed_word(
                        processive_release_staged_word(action, index),
                        permutation));
      if (index == 0u) probe_site = site;
    }
    out.supplied = true;
    out.centre = origin;
    out.site = probe_site;
    return out;
  }

  if (factor == LawFactor::processive_rearm) {
    // The role triple, then the factor's own offset and word generators mapped
    // through the shared permutation. Nothing about the geometry is restated.
    const RoleTriple roles = role_triple(variant % kProcessiveRearmVariantCount);
    const BasisPermutation permutation =
        role_basis_permutation(roles.marker, roles.path, roles.waste);
    Z3Coordinate probe_site = origin;
    for (std::uint32_t index = 0u; index < kProcessiveRearmSiteCount; ++index) {
      const ProcessiveRearmOffset offset = processive_rearm_offset(index);
      const Z3Coordinate relative = transformed_coordinate(
          {offset.marker, offset.path, offset.waste}, permutation);
      const Z3Coordinate site = origin + relative;
      put(site, transformed_word(
                              processive_rearm_candidate_word(index),
                              permutation));
      if (index == 0u) probe_site = site;
    }
    out.supplied = true;
    out.centre = origin;
    out.site = probe_site;
    return out;
  }

  if (factor == LawFactor::carrier_pair_splitter) {
    // Same principle as the corner: the factor's own offset and word generators
    // supply the matter, so the supply cannot drift from the predicate.
    const std::uint32_t pair = variant % kCarrierPairSplitterVariantCount;
    const std::uint32_t incoming = pair / 6u;
    std::uint32_t diverted = pair % 6u;
    // Skip the two directions sharing the incoming basis, so the generator's
    // third-basis scan stays well defined.
    for (std::uint32_t candidate = 0u, seen = 0u; candidate < 8u; ++candidate) {
      if ((candidate & 3u) == (incoming & 3u)) continue;
      if (seen == diverted) { diverted = candidate; break; }
      ++seen;
    }
    for (std::uint32_t index = 0u; index < kCarrierPairSplitterSiteCount;
         ++index) {
      const CarrierPairSplitterOffset relative =
          carrier_pair_splitter_offset(incoming, diverted, index);
      put({origin.x + relative.x, origin.y + relative.y,
                     origin.z + relative.z},
                    carrier_pair_splitter_word(incoming, diverted, index,
                                               false));
    }
    out.supplied = true;
    out.centre = origin;
    out.site = origin;  // index 0's offset is the origin itself
    return out;
  }

  if (factor == LawFactor::carrier_corner) {
    // The corner's own pattern generator supplies the matter. `carrier_corner_
    // offset` and `carrier_corner_word` are the same functions the factor
    // matches against, so the supply cannot drift from the predicate the way a
    // restatement would. The staged words are written because the factor's
    // owner scan looks for exactly those.
    const std::uint32_t pair = variant % kCarrierCornerVariantCount;
    const std::uint32_t incoming = pair / 7u;
    std::uint32_t outgoing = pair % 7u;
    if (outgoing >= incoming) ++outgoing;
    for (std::uint32_t index = 0u; index < kCarrierCornerSiteCount; ++index) {
      const CarrierCornerOffset relative =
          carrier_corner_offset(incoming, outgoing, index);
      put({origin.x + relative.x, origin.y + relative.y,
                     origin.z + relative.z},
                    carrier_corner_word(incoming, outgoing, index, false));
    }
    const CarrierCornerOffset head = carrier_corner_offset(incoming, outgoing, 0u);
    out.supplied = true;
    out.centre = origin;
    out.site = {origin.x + head.x, origin.y + head.y, origin.z + head.z};
    return out;
  }

  if (factor != LawFactor::eligibility_residual_junction) return out;

  // The junction's enable predicate, taken from the factor's own shared kernel
  // rather than restated: an exact non-target envelope at the centre with the
  // two transposition targets differently occupied, and a recognised control at
  // each crossed control coordinate. `kQ ^ carrier_bit(b)` is the exact carrier
  // vacancy the kernel accepts.
  const EligibilityResidualJunctionDescriptor descriptor =
      junction_descriptor(variant % kJunctionDescriptorCount, origin);
  if (!valid_eligibility_residual_junction_descriptor(descriptor)) return out;

  const SiteWord center = kQ ^ carrier_bit(descriptor.incoming_direction);
  const SiteWord control_a_word = kQ ^ carrier_bit(descriptor.control_a_basis);
  const SiteWord control_b_word = kQ ^ carrier_bit(descriptor.control_b_basis);

  // Refuse to report supplied matter that the kernel itself would not fire on.
  // This is the control that cannot be talked out of a negative: if the words
  // below stop enabling the factor, `supplied` goes false and the row returns
  // to UNKNOWN instead of silently becoming a radius-0 certificate again.
  if (!apply_eligibility_residual_junction_local(descriptor, center,
                                                 control_a_word, control_b_word)
           .fired)
    return out;

  put(descriptor.center, center);
  put(eligibility_residual_control_a_coordinate(descriptor),
                control_a_word);
  put(eligibility_residual_control_b_coordinate(descriptor),
                control_b_word);

  out.supplied = true;
  out.centre = descriptor.center;
  out.site = descriptor.center;
  return out;
}

// The bound every factor must satisfy. This is NOT a number invented for this
// audit: `kSpatialMacroClosureRadius` is the repository's own declared
// conservative BCC-hop closure, documented as required by every current spatial
// macro descriptor, which representation layers "may over-include but may not
// skip". A single factor application reaching past the closure the whole
// representation layer is built around would mean the declared locality of the
// law is wrong.
inline constexpr std::uint32_t kDeclaredFootprintBound = kSpatialMacroClosureRadius;

[[nodiscard]] inline bool within_declared_bound(const Reading& reading) {
  return reading.max_radius <= CoordinateComponent(kDeclaredFootprintBound);
}

}  // namespace substrate::bcc32::factor_footprint
