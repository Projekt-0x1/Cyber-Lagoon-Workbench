#pragma once

// WHERE IS THE IDENTITY OF A PAYLOAD REPRESENTED?
//
// bb51259987 and adc02db33c measured the support of a written byte chain
// against the EMPTY lattice: 18 sites at t=0, then 68, 69, 70, 72 over seven
// ticks, with no dual-rail pair decodable anywhere after tick 1. That is a
// measurement of where A PERTURBATION differs from vacuum. It is NOT a
// measurement of where the IDENTITY OF THE PAYLOAD lives, and the two are
// different quantities: the first tells you that something was written, the
// second tells you which region a decoder would have to read to know WHAT was
// written.
//
// External review (2026-08-06) named the conflation and the fix. Evolve two
// worlds that are identical except for the payload bytes, and census the set of
// sites at which they DIFFER. Every site outside that set is payload-invariant
// by construction -- not by inference, and with no decoder, no restriction of
// the lattice, and no inverse run involved. A restriction-and-invert test
// cannot do this job: zeroing the exterior produces a state F never reaches, so
// inverting it lets an artificial boundary propagate inward and manufacture
// valid-looking rails. This primitive avoids that failure mode entirely because
// it never edits a state; it only compares two states F actually produced.
//
// The census reports four quantities that the earlier support measurement
// conflated into one, and they answer different questions:
//
//   sites        how much state depends on the payload
//   max_radius   how FAR from the write origin that dependence reaches
//   extent       the diameter of the payload-dependent region itself
//   centroid     whether that region is sitting still or travelling
//
// A bounded `sites` with a growing `max_radius` is a region of constant size
// that is moving away -- which reads as a co-moving carrier, not confinement.
// Only bounded `max_radius` AND bounded `extent` license a bounded decoder at a
// fixed place.

#include <cstddef>
#include <cstdint>
#include <map>
#include <set>
#include <vector>

#include "bcc32_coordinate.hpp"
#include "bcc32_reference.hpp"

namespace substrate::bcc32::payload_difference {

struct Census {
  // Number of coordinates whose word differs between the two worlds. A site
  // present in one support and absent from the other counts as differing,
  // because absence is the quiescent word and that is a real state.
  std::size_t sites = 0u;

  // Largest Chebyshev (L-infinity) distance from `origin` to any differing
  // site. Bounded => the payload's identity never influences state beyond this
  // radius of where it was written.
  CoordinateComponent max_radius = 0;

  // Per-axis span of the differing set: max - min. The largest of the three is
  // `extent`. This is the size of the region a decoder would have to cover,
  // independent of where that region has drifted to.
  CoordinateComponent extent = 0;

  // Sum of differing-site coordinates. Reported unaveraged so that no rounding
  // enters, and so a caller can compare centroids exactly by cross-multiplying
  // rather than dividing.
  CoordinateComponent sum_x = 0;
  CoordinateComponent sum_y = 0;
  CoordinateComponent sum_z = 0;

  [[nodiscard]] bool empty() const { return sites == 0u; }
};

namespace detail {

[[nodiscard]] inline CoordinateComponent abs_component(
    const CoordinateComponent& value) {
  return value < 0 ? CoordinateComponent(-value) : value;
}

}  // namespace detail

using CoordinateSet = std::set<Z3Coordinate, Z3CoordinateLess>;

// The set of sites at which two worlds differ.
//
// Both supports are unioned first, so a site that exists in exactly one world
// is compared against that world's quiescent word rather than skipped. Skipping
// it would under-report the difference set and would bias every radius downward
// -- the specific way this measurement could quietly flatter a confinement
// reading.
[[nodiscard]] inline CoordinateSet difference_sites(const ReferenceLattice& left,
                                                    const ReferenceLattice& right) {
  CoordinateSet union_support;
  for (const ReferenceSite& site : left.support())
    union_support.insert(site.coordinate);
  for (const ReferenceSite& site : right.support())
    union_support.insert(site.coordinate);

  CoordinateSet out;
  for (const Z3Coordinate& coordinate : union_support)
    if (left.read(coordinate) != right.read(coordinate)) out.insert(coordinate);
  return out;
}

// The restriction of one world to a fixed set of coordinates, in the set's own
// order. This is what a reader confined to a region would actually see, and
// nothing else: two worlds with equal restrictions are indistinguishable to any
// reader of that region, whatever they do elsewhere.
//
// The coordinate set is supplied rather than derived, so the caller must commit
// to WHERE the reader looks before comparing what it finds. Deriving the region
// per pair would let the region absorb the difference and make distinctness
// unfalsifiable.
[[nodiscard]] inline std::vector<SiteWord> restrict_to(
    const ReferenceLattice& lattice, const CoordinateSet& region) {
  std::vector<SiteWord> out;
  out.reserve(region.size());
  for (const Z3Coordinate& coordinate : region)
    out.push_back(lattice.read(coordinate));
  return out;
}

// The partition of a value domain induced by what a reader of one region sees:
// values whose restrictions are equal get equal labels, numbered by order of
// first appearance so two partitions compare equal exactly when they group the
// values the same way, whatever the signatures themselves are.
//
// This is the readout side of the same mechanism `restrict_to` implements, and
// it lives here rather than in a contract because more than one contract needs
// it and a second copy would drift from the first.
[[nodiscard]] inline std::vector<std::size_t> partition_of(
    const std::vector<std::vector<SiteWord>>& signatures) {
  std::map<std::vector<SiteWord>, std::size_t> seen;
  std::vector<std::size_t> labels;
  labels.reserve(signatures.size());
  for (const auto& signature : signatures) {
    const auto found = seen.find(signature);
    if (found != seen.end()) {
      labels.push_back(found->second);
      continue;
    }
    const std::size_t next = seen.size();
    seen.emplace(signature, next);
    labels.push_back(next);
  }
  return labels;
}

[[nodiscard]] inline std::size_t class_count(
    const std::vector<std::size_t>& labels) {
  std::size_t highest = 0u;
  for (const std::size_t label : labels)
    if (label + 1u > highest) highest = label + 1u;
  return highest;
}

// Census the sites at which two worlds differ.
[[nodiscard]] inline Census census(const ReferenceLattice& left,
                                   const ReferenceLattice& right,
                                   const Z3Coordinate& origin) {
  const CoordinateSet differing = difference_sites(left, right);

  Census out;
  bool seen = false;
  CoordinateComponent min_x = 0, max_x = 0;
  CoordinateComponent min_y = 0, max_y = 0;
  CoordinateComponent min_z = 0, max_z = 0;

  for (const Z3Coordinate& coordinate : differing) {
    ++out.sites;

    const CoordinateComponent dx =
        detail::abs_component(coordinate.x - origin.x);
    const CoordinateComponent dy =
        detail::abs_component(coordinate.y - origin.y);
    const CoordinateComponent dz =
        detail::abs_component(coordinate.z - origin.z);
    const CoordinateComponent chebyshev =
        dx > dy ? (dx > dz ? dx : dz) : (dy > dz ? dy : dz);
    if (chebyshev > out.max_radius) out.max_radius = chebyshev;

    out.sum_x += coordinate.x;
    out.sum_y += coordinate.y;
    out.sum_z += coordinate.z;

    if (!seen) {
      min_x = max_x = coordinate.x;
      min_y = max_y = coordinate.y;
      min_z = max_z = coordinate.z;
      seen = true;
      continue;
    }
    if (coordinate.x < min_x) min_x = coordinate.x;
    if (coordinate.x > max_x) max_x = coordinate.x;
    if (coordinate.y < min_y) min_y = coordinate.y;
    if (coordinate.y > max_y) max_y = coordinate.y;
    if (coordinate.z < min_z) min_z = coordinate.z;
    if (coordinate.z > max_z) max_z = coordinate.z;
  }

  if (seen) {
    const CoordinateComponent span_x = max_x - min_x;
    const CoordinateComponent span_y = max_y - min_y;
    const CoordinateComponent span_z = max_z - min_z;
    out.extent = span_x > span_y ? (span_x > span_z ? span_x : span_z)
                                 : (span_y > span_z ? span_y : span_z);
  }
  return out;
}

}  // namespace substrate::bcc32::payload_difference
