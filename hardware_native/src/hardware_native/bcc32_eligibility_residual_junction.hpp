#pragma once

#include <cstddef>
#include <cstdint>

#include "bcc32_coordinate.hpp"
#include "bcc32_geometry.cuh"
#include "bcc32_law.cuh"

namespace substrate::bcc32 {

class ReferenceLattice;

// A proposed ordinary-matter junction descriptor.  The two local eligibility
// sites catalyse a carrier-hole transpose at center; they are controls, never
// write targets.  Direction and control roles are explicit so the construction
// has no preferred BCC basis.
struct EligibilityResidualJunctionDescriptor {
  Z3Coordinate center{};
  std::uint32_t incoming_direction = 0u;
  std::uint32_t outgoing_direction = 0u;
  std::uint32_t control_a_basis = 0u;
  std::uint32_t control_b_basis = 0u;
};

namespace eligibility_residual_detail {

inline Z3Coordinate offset(const Z3Coordinate& coordinate, Int3 delta) {
  return {coordinate.x + delta.x, coordinate.y + delta.y,
          coordinate.z + delta.z};
}

inline Int3 scaled(Int3 value, std::int32_t multiplier) {
  return {multiplier * value.x, multiplier * value.y, multiplier * value.z};
}

// The control offset that distinguishes a positive incoming leg.  Shared by
// both control coordinates so the two never disagree about the aim.
inline Int3 incoming_control_offset(
    const EligibilityResidualJunctionDescriptor& descriptor) {
  return descriptor.incoming_direction < 4u
             ? scaled(basis_offset(
                          static_cast<Basis>(descriptor.incoming_direction)),
                      kEligibilityResidualJunctionAuthority
                          .positive_incoming_control_offset)
             : Int3{};
}

}  // namespace eligibility_residual_detail

[[nodiscard]] inline bool valid_eligibility_residual_junction_descriptor(
    const EligibilityResidualJunctionDescriptor& descriptor) {
  if (descriptor.incoming_direction >= 8u ||
      descriptor.outgoing_direction >= 8u ||
      descriptor.incoming_direction == descriptor.outgoing_direction ||
      descriptor.control_a_basis >= 4u || descriptor.control_b_basis >= 4u ||
      descriptor.control_a_basis == descriptor.control_b_basis)
    return false;

  const std::uint32_t incoming_basis = descriptor.incoming_direction & 3u;
  const std::uint32_t outgoing_basis = descriptor.outgoing_direction & 3u;
  return incoming_basis != outgoing_basis &&
         descriptor.control_a_basis != incoming_basis &&
         descriptor.control_a_basis != outgoing_basis &&
         descriptor.control_b_basis != incoming_basis &&
         descriptor.control_b_basis != outgoing_basis;
}

[[nodiscard]] inline Z3Coordinate eligibility_residual_control_a_coordinate(
    const EligibilityResidualJunctionDescriptor& descriptor) {
  using namespace eligibility_residual_detail;
  const Int3 outgoing =
      direction_offset(static_cast<Direction>(descriptor.outgoing_direction));
  const Int3 crossed =
      basis_offset(static_cast<Basis>(descriptor.control_b_basis));
  return offset(descriptor.center,
                scaled(outgoing, kEligibilityResidualJunctionAuthority
                                     .outgoing_distance) +
                    crossed + incoming_control_offset(descriptor));
}

[[nodiscard]] inline Z3Coordinate eligibility_residual_control_b_coordinate(
    const EligibilityResidualJunctionDescriptor& descriptor) {
  using namespace eligibility_residual_detail;
  const Int3 outgoing =
      direction_offset(static_cast<Direction>(descriptor.outgoing_direction));
  const Int3 crossed =
      basis_offset(static_cast<Basis>(descriptor.control_a_basis));
  return offset(descriptor.center,
                scaled(outgoing, kEligibilityResidualJunctionAuthority
                                     .outgoing_distance) +
                    crossed + incoming_control_offset(descriptor));
}

[[nodiscard]] inline Z3Coordinate eligibility_residual_owner_coordinate(
    const EligibilityResidualJunctionDescriptor& descriptor) {
  using namespace eligibility_residual_detail;
  const Int3 outgoing =
      direction_offset(static_cast<Direction>(descriptor.outgoing_direction));
  const Int3 incoming =
      direction_offset(static_cast<Direction>(descriptor.incoming_direction));
  const Int3 control_a =
      basis_offset(static_cast<Basis>(descriptor.control_a_basis));
  const Int3 control_b =
      basis_offset(static_cast<Basis>(descriptor.control_b_basis));
  return offset(descriptor.center, scaled(outgoing, 2) + scaled(incoming, 3) +
                                       control_a + control_b);
}

[[nodiscard]] inline SiteWord eligibility_residual_owner_staged_word(
    const EligibilityResidualJunctionDescriptor& descriptor) {
  return kQ | channel_bit(kReactiveShift, descriptor.incoming_direction & 3u);
}

[[nodiscard]] inline SiteWord eligibility_residual_owner_released_word(
    const EligibilityResidualJunctionDescriptor& descriptor) {
  return kQ | channel_bit(kReactiveShift, descriptor.outgoing_direction & 3u);
}

// A local eligibility site is recognised either by represented eligibility or
// by an exact carrier vacancy on its own basis.  Nothing else enables.
[[nodiscard]] inline bool eligibility_residual_control_present(
    SiteWord word, std::uint32_t basis) {
  const bool represented_eligibility = (word & energy_bit(basis)) != 0u;
  const bool exact_carrier_vacancy = word == (kQ ^ carrier_bit(basis));
  return represented_eligibility || exact_carrier_vacancy;
}

// The complete single-descriptor local transition, with no lattice storage.
// `fired` reports whether the descriptor enabled; `center` is the post-state
// centre word and equals the input when it did not.  The controls are never
// write targets, so they are inputs only.  A strict-device caller may certify
// or execute this transition without linking the host reference interpreter.
struct EligibilityResidualLocalResult {
  SiteWord center = 0u;
  bool fired = false;
};

[[nodiscard]] inline EligibilityResidualLocalResult
apply_eligibility_residual_junction_local(
    const EligibilityResidualJunctionDescriptor& descriptor, SiteWord center,
    SiteWord control_a, SiteWord control_b) {
  EligibilityResidualLocalResult result{center, false};
  if (!valid_eligibility_residual_junction_descriptor(descriptor))
    return result;

  const SiteWord incoming_bit = carrier_bit(descriptor.incoming_direction);
  const SiteWord outgoing_bit = carrier_bit(descriptor.outgoing_direction);
  const SiteWord target_mask = incoming_bit | outgoing_bit;
  // The enable predicate deliberately excludes both transposition targets.
  const bool exact_non_target_envelope =
      (center & ~target_mask) == (kQ & ~target_mask);
  const bool target_occupancy_differs =
      ((center & incoming_bit) != 0u) != ((center & outgoing_bit) != 0u);
  const bool controls_present =
      eligibility_residual_control_present(control_a,
                                           descriptor.control_a_basis) &&
      eligibility_residual_control_present(control_b,
                                           descriptor.control_b_basis);
  if (!(exact_non_target_envelope && target_occupancy_differs &&
        controls_present))
    return result;

  // Overlap abstention is a property of the descriptor's geometry alone: the
  // owner site participates here and nowhere else on this path.  The whole
  // predicate is a conjunction, so testing the four coordinates last is
  // semantically identical and keeps them off the multi-descriptor hot path.
  const Z3Coordinate sites[4] = {
      descriptor.center, eligibility_residual_control_a_coordinate(descriptor),
      eligibility_residual_control_b_coordinate(descriptor),
      eligibility_residual_owner_coordinate(descriptor)};
  for (std::size_t left = 0u; left < 4u; ++left)
    for (std::size_t right = left + 1u; right < 4u; ++right)
      if (sites[left] == sites[right]) return result;

  SiteWord after = center;
  controlled_transpose(after, incoming_bit, after, outgoing_bit, true);
  result.center = after;
  result.fired = true;
  return result;
}

// Discovery reads one immutable snapshot.  Enabled descriptors whose
// center/control footprints overlap abstain together; disjoint owners commit
// simultaneously.  Applying the same operation twice is the exact inverse.
void apply_eligibility_residual_junction(
    ReferenceLattice& lattice,
    const EligibilityResidualJunctionDescriptor* descriptors,
    std::size_t descriptor_count);

// Canonical CPU factor: discovers the complete S4 x outgoing-sign orbit from
// ordinary finite support.  The bool is present for factor-dispatch symmetry;
// this transposition factor is its own inverse.
void apply_k_eligibility_residual_junction(ReferenceLattice& lattice,
                                           bool inverse);

}  // namespace substrate::bcc32
