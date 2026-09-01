#pragma once

// THE GROUNDED ANATOMY'S SHAPE, AS CODE RATHER THAN AS A COMMENT.
//
// `bcc32_same_adult_resident_completion_contract.cu` carries the seven-route
// anatomy as `const std::array<Route, 7> kRoutes` and the readout as
// `kTargets`. Both are typed-out coordinates. Both are ALSO accompanied by a
// comment describing the rule the entries follow -- which is the exact shape
// METHOD forbids: the rule is authored as prose and its expansion is authored
// as data, so a word longer than the table needs a SOURCE EDIT. The test METHOD
// gives is "does it have to be re-authored when the task grows?", and the answer
// today is yes.
//
// This header makes the recorded rules executable so they can be CHECKED rather
// than believed. Nothing here replaces the authored tables yet: the first
// question is how much of the anatomy the recorded shape actually generates.
//
// ⛔ THIS IS NOT AN ATTEMPT TO REPAIR ROUTE 3. That prohibition stands and is
// correct: repairing one route buys a single payload position while preserving
// an architecture that cannot scale, and the six-byte ceiling was measured to be
// a private-line readout rather than a broken route. The target here is the
// opposite one -- whether the ANATOMY ITSELF is authored global structure, which
// is a question about the architecture, not about one entry in it.
//
// The three recorded rules, quoted from the tables they describe:
//
//   R1  d(first - source) is 8*e_axis for axis x/y/z, or (-8,-8,-8) for a
//       fourth "diagonal" kind
//   R2  axis-aligned routes have owner = source + (1,1,1) and basis 3
//   R3  target = first + e_axis, bit = 0x00100000 << axis, tick = 14
//
// R3 is the one that also exposes a malformed entry: index 3's target bears no
// stated relation to its own first waypoint and its tick is 40 rather than 14.

#include <cstdint>

#include "bcc32_coordinate.hpp"

namespace substrate::bcc32::route_shape {

struct TargetShape {
  Z3Coordinate coordinate;
  std::uint32_t bit = 0u;
  int tick = 0;
};

// R1, axis-aligned: the first waypoint is eight sites out along one axis.
[[nodiscard]] inline Z3Coordinate first_from_source(const Z3Coordinate& source,
                                                    int axis) {
  Z3Coordinate out = source;
  if (axis == 0) out.x += 8;
  if (axis == 1) out.y += 8;
  if (axis == 2) out.z += 8;
  return out;
}

// R1, diagonal kind.
[[nodiscard]] inline Z3Coordinate diagonal_first_from_source(
    const Z3Coordinate& source) {
  return {source.x - 8, source.y - 8, source.z - 8};
}

// R2: the attachment owner of an axis-aligned route.
[[nodiscard]] inline Z3Coordinate owner_from_source(const Z3Coordinate& source) {
  return {source.x + 1, source.y + 1, source.z + 1};
}

inline constexpr std::uint32_t kAxisAlignedBasis = 3u;

// R3: the readout of an axis-aligned route.
[[nodiscard]] inline TargetShape target_from_first(const Z3Coordinate& first,
                                                   int axis) {
  Z3Coordinate coordinate = first;
  if (axis == 0) coordinate.x += 1;
  if (axis == 1) coordinate.y += 1;
  if (axis == 2) coordinate.z += 1;
  return {coordinate, 0x00100000u << static_cast<std::uint32_t>(axis), 14};
}

// R4, FOUND AFTER R1-R3 WERE LANDED. The first pass concluded the second
// waypoint had no rule, because it looked for an AXIS -> OFFSET map and the two
// populations disagree under that reading. The route's own `permutation` field
// was never consulted.
//
//   second = first + 4 * e_{permutation[3]}
//
// `permutation[3]` is `first_bond`, which the repository already documents as
// the bond the route's first waypoint carries. So the second waypoint is not
// keyed to the route's AXIS at all -- it is keyed to the route's own BOND, and
// under that reading both populations obey one rule.
[[nodiscard]] inline Z3Coordinate second_from_first(const Z3Coordinate& first,
                                                    std::uint32_t first_bond) {
  Z3Coordinate out = first;
  if (first_bond == 0u) out.x += 4;
  if (first_bond == 1u) out.y += 4;
  if (first_bond == 2u) out.z += 4;
  return out;
}

}  // namespace substrate::bcc32::route_shape
