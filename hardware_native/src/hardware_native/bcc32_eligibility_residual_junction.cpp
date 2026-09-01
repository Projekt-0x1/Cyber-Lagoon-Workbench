#include "bcc32_eligibility_residual_junction.hpp"

#include <algorithm>
#include <array>
#include <vector>

#include "bcc32_geometry.cuh"
#include "bcc32_law.cuh"
#include "bcc32_reference.hpp"

namespace substrate::bcc32 {
namespace {

static_assert(kEligibilityResidualJunctionAuthority.target_independent == 1u);
static_assert(kEligibilityResidualJunctionAuthority.overlap_abstains == 1u);
static_assert(kEligibilityResidualJunctionAuthority.exact_carrier_vacancy_controls == 1u);
static_assert(kEligibilityResidualJunctionAuthority.positive_incoming_control_offset == 3u);
static_assert(kEligibilityResidualJunctionAuthority.endogenous_owner_required == 1u);

bool junction_control_present(SiteWord word, std::uint32_t basis) {
  return eligibility_residual_control_present(word, basis);
}

using Footprint = std::array<Z3Coordinate, 4u>;

Footprint footprint(const EligibilityResidualJunctionDescriptor& descriptor) {
  return {descriptor.center,
          eligibility_residual_control_a_coordinate(descriptor),
          eligibility_residual_control_b_coordinate(descriptor),
          eligibility_residual_owner_coordinate(descriptor)};
}

bool has_internal_overlap(const Footprint& value) {
  for (std::size_t left = 0u; left < value.size(); ++left)
    for (std::size_t right = left + 1u; right < value.size(); ++right)
      if (value[left] == value[right]) return true;
  return false;
}

bool overlaps(const Footprint& left, std::size_t left_count,
              const Footprint& right, std::size_t right_count) {
  for (std::size_t left_index = 0u; left_index < left_count; ++left_index)
    for (std::size_t right_index = 0u; right_index < right_count;
         ++right_index)
      {
      const Z3Coordinate& a = left[left_index];
      const Z3Coordinate& b = right[right_index];
      if (a == b)
        return true;
      }
  return false;
}

bool same_footprint_set(const Footprint& left, std::size_t left_count,
                        const Footprint& right, std::size_t right_count) {
  if (left_count != right_count) return false;
  for (std::size_t left_index = 0u; left_index < left_count; ++left_index) {
    const Z3Coordinate& a = left[left_index];
    bool found = false;
    for (std::size_t right_index = 0u; right_index < right_count;
         ++right_index)
      found = found || a == right[right_index];
    if (!found)
      return false;
  }
  return true;
}

struct Match {
  EligibilityResidualJunctionDescriptor descriptor{};
  Footprint footprint{};
  // The kernel's post-state centre word; nothing here recomputes the action.
  SiteWord post_center = 0u;
  std::size_t footprint_count = 3u;
  Z3Coordinate phase_token{};
  std::uint32_t phase_basis = 0u;
  // 0: legacy immutable controls, 1: staged endogenous lane lock,
  // 2: released endogenous lane lock.
  std::uint8_t phase = 0u;
  bool collided = false;
};

bool equivalent_action(const Match& left, const Match& right) {
  return left.descriptor.center == right.descriptor.center &&
         left.descriptor.incoming_direction ==
             right.descriptor.incoming_direction &&
         left.descriptor.outgoing_direction ==
             right.descriptor.outgoing_direction &&
         same_footprint_set(left.footprint, left.footprint_count,
                            right.footprint, right.footprint_count);
}

std::uint8_t owner_phase(
    SiteWord word, const EligibilityResidualJunctionDescriptor& descriptor) {
  if (word == eligibility_residual_owner_staged_word(descriptor)) return 1u;
  if (word == eligibility_residual_owner_released_word(descriptor)) return 2u;
  return 0u;
}

void apply_impl(ReferenceLattice& lattice,
                const EligibilityResidualJunctionDescriptor* descriptors,
                std::size_t descriptor_count,
                bool collapse_s4_stabilizers) {
  if (descriptors == nullptr || descriptor_count == 0u)
    return;

  const ReferenceLattice before = lattice;
  std::vector<Match> matches;
  matches.reserve(descriptor_count);

  for (std::size_t index = 0; index < descriptor_count; ++index) {
    const EligibilityResidualJunctionDescriptor& descriptor = descriptors[index];
    if (!valid_eligibility_residual_junction_descriptor(descriptor))
      continue;
    const Footprint sites = footprint(descriptor);
    if (has_internal_overlap(sites))
      continue;

    // The enable predicate and the transposition both live in the shared
    // lattice-free kernel; this loop supplies the words and owns only the
    // multi-descriptor arbitration around it.
    const SiteWord center = before.read(descriptor.center);
    const EligibilityResidualLocalResult local =
        apply_eligibility_residual_junction_local(
            descriptor, center, before.read(sites[1]), before.read(sites[2]));
    const std::uint8_t phase = collapse_s4_stabilizers
        ? owner_phase(before.read(sites[3]), descriptor)
        : 0u;
    // Production discovery requires a resident phase token. Arbitrary words
    // that happen to satisfy the centre and control predicates do not acquire
    // authority merely by coincidence. The staged/released token survives the
    // centre transpose and therefore names the same local action in F^-1.
    if (collapse_s4_stabilizers && phase == 0u)
      continue;
    if (!local.fired)
      continue;

    Match candidate{descriptor, sites, local.center};
    if (phase != 0u) {
      candidate.phase_token = sites[3];
      candidate.phase = phase;
      candidate.footprint_count = 4u;
    }
    bool equivalent = false;
    if (collapse_s4_stabilizers) {
      for (const Match& match : matches)
        equivalent = equivalent || equivalent_action(match, candidate);
    }
    if (!equivalent)
      matches.push_back(candidate);
  }


  // A released local lane lock is the endogenous witness of the action that
  // actually ran.  It outranks compatible staged owners at the same center;
  // two inequivalent released actions still collide and abstain.
  for (std::size_t left = 0u; left < matches.size(); ++left) {
    if (matches[left].phase == 2u) continue;
    for (const Match& right : matches)
      if (right.descriptor.center == matches[left].descriptor.center &&
          right.phase == 2u && !equivalent_action(matches[left], right)) {
        matches[left].collided = true;
        break;
      }
  }

  for (std::size_t left = 0; left < matches.size(); ++left) {
    for (std::size_t right = left + 1u; right < matches.size(); ++right) {
      if (overlaps(matches[left].footprint, matches[left].footprint_count,
                   matches[right].footprint,
                   matches[right].footprint_count)) {
        matches[left].collided = true;
        matches[right].collided = true;
      }
    }
  }

  ReferenceLattice after = before;
  for (const Match& match : matches) {
    if (match.collided)
      continue;
    after.write(match.descriptor.center, match.post_center);
    if (match.phase != 0u) {
      SiteWord token = before.read(match.phase_token);
      controlled_transpose(
          token,
          channel_bit(kReactiveShift,
                      match.descriptor.incoming_direction & 3u),
          token,
          channel_bit(kReactiveShift,
                      match.descriptor.outgoing_direction & 3u),
          true);
      after.write(match.phase_token, token);
    }
  }
  lattice = after;
}

}  // namespace

void apply_eligibility_residual_junction(
    ReferenceLattice& lattice,
    const EligibilityResidualJunctionDescriptor* descriptors,
    std::size_t descriptor_count) {
  apply_impl(lattice, descriptors, descriptor_count, false);
}

void apply_k_eligibility_residual_junction(ReferenceLattice& lattice,
                                           bool inverse) {
  (void)inverse;
  const ReferenceLattice before = lattice;
  constexpr EligibilityResidualJunctionAuthority authority =
      kEligibilityResidualJunctionAuthority;
  std::vector<EligibilityResidualJunctionDescriptor> descriptors;
  descriptors.reserve(before.support_size() * 24u *
                      authority.outgoing_sign_count * authority.incoming_sign_count);
  for (const ReferenceSite& site : before.support()) {
    std::array<std::uint32_t, 4u> permutation{0u, 1u, 2u, 3u};
    do {
      for (std::uint32_t outgoing_sign = 0u;
           outgoing_sign < authority.outgoing_sign_count;
           ++outgoing_sign) {
        for (std::uint32_t incoming_sign = 0u;
             incoming_sign < authority.incoming_sign_count;
             ++incoming_sign) {
          descriptors.push_back(
              {site.coordinate,
               permutation[authority.incoming_role] + 4u * incoming_sign,
               permutation[authority.outgoing_role] + 4u * outgoing_sign,
               permutation[authority.control_a_role],
               permutation[authority.control_b_role]});
        }
      }
    } while (std::next_permutation(permutation.begin(), permutation.end()));
  }
  apply_impl(lattice, descriptors.data(), descriptors.size(), true);
}

}  // namespace substrate::bcc32
