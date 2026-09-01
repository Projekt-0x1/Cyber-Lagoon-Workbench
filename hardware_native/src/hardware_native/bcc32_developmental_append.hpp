#pragma once

#include <cstdint>

#include "bcc32_types.cuh"

namespace substrate::bcc32 {

class ReferenceLattice;

// A role-relative, content-neutral construction transaction.  `marker`,
// `path`, and `waste` are coefficients of three distinct tetrahedral bases;
// the fourth basis is the generic feedstock lane.
struct DevelopmentalAppendOffset {
  std::int8_t marker = 0;
  std::int8_t path = 0;
  std::int8_t waste = 0;
};

inline constexpr std::uint32_t kDevelopmentalAppendSiteCount = 128u;
inline constexpr std::uint32_t kDevelopmentalAppendFirstVirginSite = 10u;
inline constexpr std::uint32_t kDevelopmentalAppendVirginSiteCount = 118u;
inline constexpr std::uint32_t kDevelopmentalAppendParentLockFirst = 1u;
inline constexpr std::uint32_t kDevelopmentalAppendParentLockSecond = 2u;
inline constexpr std::uint32_t kDevelopmentalAppendChildHead = 14u;
inline constexpr std::uint32_t kDevelopmentalAppendChildLockFirst = 15u;
inline constexpr std::uint32_t kDevelopmentalAppendChildLockSecond = 16u;
inline constexpr std::uint32_t kDevelopmentalAppendChildFeedstockFirst = 18u;
inline constexpr std::uint32_t kDevelopmentalAppendChildFeedstockCount = 6u;
inline constexpr std::uint32_t kDevelopmentalAppendRetainedHoleAnchorFirst =
    4u;
inline constexpr std::uint32_t kDevelopmentalAppendRetainedHoleLockFirst =
    24u;
inline constexpr std::uint32_t kDevelopmentalAppendRetainedHoleCount = 6u;
inline constexpr std::uint32_t kDevelopmentalAppendHandoffSiteCount = 9u;
inline constexpr std::uint32_t kDevelopmentalAppendAgeFirst = 32u;
inline constexpr std::uint32_t kDevelopmentalAppendAgeDigitCount = 24u;
inline constexpr std::uint32_t kDevelopmentalAppendAgeSitesPerDigit = 3u;
inline constexpr std::uint32_t kDevelopmentalAppendJournalFirst =
    kDevelopmentalAppendAgeFirst +
    kDevelopmentalAppendAgeDigitCount * kDevelopmentalAppendAgeSitesPerDigit;
inline constexpr std::uint32_t kDevelopmentalAppendJournalDigitCount = 8u;
inline constexpr std::uint64_t kDevelopmentalAppendMaxAge =
    (std::uint64_t{1u} << (2u * kDevelopmentalAppendAgeDigitCount)) - 1u;
inline constexpr std::uint32_t kDevelopmentalAppendReceptorLegCount = 2u;
inline constexpr std::uint32_t kDevelopmentalAppendReceptorJournalEmpty = 0u;
inline constexpr std::uint32_t kDevelopmentalAppendReceptorJournalA = 1u;
inline constexpr std::uint32_t kDevelopmentalAppendReceptorJournalB = 2u;
inline constexpr std::uint32_t kDevelopmentalAppendReceptorInletSite = 10u;
inline constexpr std::uint32_t kDevelopmentalAppendAuthorityDigitCount = 2u;
inline constexpr std::uint32_t kDevelopmentalAppendEventJournalFirst = 2u;
inline constexpr std::uint32_t kDevelopmentalAppendEventJournalCount =
    kDevelopmentalAppendJournalDigitCount -
    kDevelopmentalAppendEventJournalFirst;

[[nodiscard]] __host__ __device__ constexpr std::uint32_t
developmental_append_receptor_authority_site(std::uint32_t leg) {
  return kDevelopmentalAppendJournalFirst +
         leg * kDevelopmentalAppendAgeSitesPerDigit;
}

[[nodiscard]] __host__ __device__ constexpr std::uint32_t
developmental_append_journal_state_site(std::uint32_t digit) {
  return kDevelopmentalAppendJournalFirst +
         digit * kDevelopmentalAppendAgeSitesPerDigit + 2u;
}

[[nodiscard]] __host__ __device__ constexpr std::uint32_t
developmental_append_receptor_authority_state_site(std::uint32_t leg) {
  return developmental_append_journal_state_site(leg);
}

[[nodiscard]] __host__ __device__ constexpr bool
developmental_append_handoff_pair(std::uint32_t parent_site,
                                  std::uint32_t child_site) {
  return parent_site == child_site + kDevelopmentalAppendChildHead &&
         (child_site < 3u || (child_site >= 4u && child_site <= 9u));
}

[[nodiscard]] __host__ __device__ constexpr DevelopmentalAppendOffset
developmental_append_offset(std::uint32_t index) {
  switch (index) {
    case 0u:  // armed/spent parent head
      return {};
    case 1u:  // first owner lock
      return {-6, 0, 0};
    case 2u:  // second owner lock
      return {-8, 0, 0};
    case 3u:  // demand
      return {0, -2, 0};
    case 4u:
      return {0, 0, -2};
    case 5u:
      return {0, 0, -3};
    case 6u:
      return {0, 0, -4};
    case 7u:
      return {0, 0, -5};
    case 8u:
      return {0, 0, -6};
    case 9u:
      return {0, 0, -7};
    case 10u:  // blank processive body, one path step after the head
      return {0, 1, 0};
    case 11u:  // processive marker lock
      return {1, 1, 0};
    case 12u:  // processive free-basis lock
      return {0, 1, 6};
    case 13u:  // processive marker-face lock
      return {0, 1, 8};
    case 14u:  // armed child; next body begins six path steps later
      return {0, 6, 0};
    case 15u:  // child owner lock one
      return {-6, 6, 0};
    case 16u:  // child owner lock two
      return {-8, 6, 0};
    case 17u:  // represented construction waste
      return {0, 0, 10};
    case 18u:  // exact child feedstock
      return {0, 6, -2};
    case 19u:
      return {0, 6, -3};
    case 20u:
      return {0, 6, -4};
    case 21u:
      return {0, 6, -5};
    case 22u:
      return {0, 6, -6};
    case 23u:
      return {0, 6, -7};
    case 24u:  // retained-hole lock beside consumed parent resource 4
      return {1, 0, -2};
    case 25u:
      return {1, 0, -3};
    case 26u:
      return {1, 0, -4};
    case 27u:
      return {1, 0, -5};
    case 28u:
      return {1, 0, -6};
    case 29u:
      return {1, 0, -7};
    case 30u:  // second represented construction-waste singleton
      return {1, 0, 10};
    case 31u:  // third represented construction-waste singleton
      return {2, 0, 10};
    default:
      if (index >= kDevelopmentalAppendAgeFirst &&
          index < kDevelopmentalAppendSiteCount) {
        const std::uint32_t relative = index - kDevelopmentalAppendAgeFirst;
        std::uint32_t digit =
            relative / kDevelopmentalAppendAgeSitesPerDigit;
        const std::uint32_t cell = digit;
        const std::uint32_t component =
            relative % kDevelopmentalAppendAgeSitesPerDigit;
        // Thirty-two graph-separated three-site cells in the path=4 plane.
        // Every coordinate is inside the spatial macro closure radius 15;
        // adjacent lineage banks are shifted to path=10 and cannot alias.
        constexpr std::uint32_t row_counts[9] =
            {1u, 2u, 3u, 4u, 5u, 6u, 5u, 4u, 2u};
        std::uint32_t row = 0u;
        while (digit >= row_counts[row]) digit -= row_counts[row++];
        const std::int32_t waste_value = -10 + 2 * static_cast<std::int32_t>(row);
        const std::int32_t marker_extent =
            11 - (waste_value < 0 ? -waste_value : waste_value);
        const std::int8_t marker = static_cast<std::int8_t>(
            -marker_extent + 4 * static_cast<std::int32_t>(digit) +
            static_cast<std::int32_t>(component));
        // Cell 18's original {marker=1..3,path=4,waste=0} triplet crosses the
        // lawful terminal demand-routing orbit before adult birth.  The
        // adjacent waste=1 plane is graph-disjoint, remains inside radius 15,
        // and preserves separation from the path+6 child ledger.
        if (cell == 18u)
          return {static_cast<std::int8_t>(1 + component), 4, 1};
        const std::int8_t path = 4;
        const std::int8_t waste = static_cast<std::int8_t>(waste_value);
        return {marker, path, waste};
      }
      return {};
  }
}

[[nodiscard]] __host__ __device__ constexpr std::uint32_t
developmental_append_free_basis(std::uint32_t marker, std::uint32_t path,
                                std::uint32_t waste) {
  for (std::uint32_t basis = 0u; basis < 4u; ++basis)
    if (basis != marker && basis != path && basis != waste) return basis;
  return 0u;
}

[[nodiscard]] __host__ __device__ constexpr std::uint32_t
developmental_append_receptor_basis(std::uint32_t leg,
                                    std::uint32_t marker,
                                    std::uint32_t waste) {
  return leg == 0u ? marker : waste;
}

[[nodiscard]] __host__ __device__ constexpr std::uint32_t
developmental_append_spent_source_basis(std::uint32_t leg,
                                        std::uint32_t marker,
                                        std::uint32_t path,
                                        std::uint32_t waste) {
  return leg == 0u ? developmental_append_free_basis(marker, path, waste)
                   : path;
}

[[nodiscard]] __host__ __device__ constexpr SiteWord
developmental_append_spent_source_vacancy_word(std::uint32_t leg,
                                               std::uint32_t marker,
                                               std::uint32_t path,
                                               std::uint32_t waste) {
  return kQuiescentWord ^ carrier_bit(
                              developmental_append_spent_source_basis(
                                  leg, marker, path, waste));
}

[[nodiscard]] __host__ __device__ constexpr DevelopmentalAppendOffset
developmental_append_clock_ingress_offset(std::uint32_t leg) {
  return leg == 0u ? DevelopmentalAppendOffset{-12, 0, 0}
                   : DevelopmentalAppendOffset{0, 0, -12};
}

[[nodiscard]] __host__ __device__ constexpr DevelopmentalAppendOffset
developmental_append_clock_escrow_offset(std::uint32_t leg,
                                         std::uint32_t slot = 0u) {
  constexpr std::int8_t marker_or_waste[5u]{-13, 5, -12, -7, 10};
  constexpr std::int8_t path[5u]{1, 8, 3, -1, 3};
  const std::uint32_t bounded = slot < 5u ? slot : 0u;
  return leg == 0u
             ? DevelopmentalAppendOffset{marker_or_waste[bounded],
                                         path[bounded], 0}
             : DevelopmentalAppendOffset{0, path[bounded],
                                         marker_or_waste[bounded]};
}

[[nodiscard]] __host__ __device__ constexpr DevelopmentalAppendOffset
developmental_append_reject_escrow_offset(std::uint32_t leg,
                                          std::uint32_t slot = 0u) {
  constexpr std::int8_t marker_or_waste[5u]{9, 4, -8, -11, 2};
  constexpr std::int8_t path[5u]{3, 5, -7, 3, 6};
  const std::uint32_t bounded = slot < 5u ? slot : 0u;
  return leg == 0u
             ? DevelopmentalAppendOffset{marker_or_waste[bounded],
                                         path[bounded], 0}
             : DevelopmentalAppendOffset{0, path[bounded],
                                         marker_or_waste[bounded]};
}

[[nodiscard]] __host__ __device__ constexpr DevelopmentalAppendOffset
developmental_append_reject_clock_ingress_offset(std::uint32_t leg,
                                                  std::uint32_t bank) {
  constexpr std::int8_t marker_or_waste[5u]{4, 11, -13, -15, -13};
  constexpr std::int8_t path[5u]{4, -1, 0, 0, -2};
  const std::uint32_t bounded = bank < 5u ? bank : 0u;
  return leg == 0u
             ? DevelopmentalAppendOffset{marker_or_waste[bounded],
                                         path[bounded], 0}
             : DevelopmentalAppendOffset{0, path[bounded],
                                         marker_or_waste[bounded]};
}

[[nodiscard]] __host__ __device__ constexpr DevelopmentalAppendOffset
developmental_append_reject_teacher_ingress_offset(std::uint32_t leg,
                                                    std::uint32_t bank) {
  constexpr std::int8_t marker_or_waste[5u]{6, -9, 8, -14, 12};
  constexpr std::int8_t path[5u]{-4, 0, -1, 0, -2};
  const std::uint32_t bounded = bank < 5u ? bank : 0u;
  return leg == 0u
             ? DevelopmentalAppendOffset{marker_or_waste[bounded],
                                         path[bounded], 0}
             : DevelopmentalAppendOffset{0, path[bounded],
                                         marker_or_waste[bounded]};
}

[[nodiscard]] __host__ __device__ constexpr DevelopmentalAppendOffset
developmental_append_reject_source_ingress_offset(std::uint32_t leg,
                                                   std::uint32_t bank) {
  constexpr std::int8_t marker_or_waste[5u]{-2, 3, 4, 3, -5};
  constexpr std::int8_t path[5u]{-3, -1, 2, -5, 2};
  const std::uint32_t bounded = bank < 5u ? bank : 0u;
  return leg == 0u
             ? DevelopmentalAppendOffset{marker_or_waste[bounded],
                                         path[bounded], 0}
             : DevelopmentalAppendOffset{0, path[bounded],
                                         marker_or_waste[bounded]};
}

inline constexpr std::uint32_t kDevelopmentalAppendWitnessBankCount = 5u;

[[nodiscard]] __host__ __device__ constexpr std::uint32_t
developmental_append_clock_basis(std::uint32_t leg, std::uint32_t marker,
                                 std::uint32_t path,
                                 std::uint32_t waste) {
  return leg == 0u ? developmental_append_free_basis(marker, path, waste)
                   : path;
}

[[nodiscard]] __host__ __device__ constexpr SiteWord
developmental_append_clock_vacancy_word(std::uint32_t leg,
                                        std::uint32_t marker,
                                        std::uint32_t path,
                                        std::uint32_t waste) {
  return kQuiescentWord ^ carrier_bit(
                              developmental_append_clock_basis(
                                  leg, marker, path, waste));
}

[[nodiscard]] __host__ __device__ constexpr SiteWord
developmental_append_escrow_word(bool filled, std::uint32_t leg,
                                 std::uint32_t marker,
                                 std::uint32_t waste) {
  return filled ? (kQuiescentWord |
                   face_bit(developmental_append_receptor_basis(
                       leg, marker, waste)))
                : kQuiescentWord;
}

[[nodiscard]] __host__ __device__ constexpr DevelopmentalAppendOffset
developmental_append_receptor_port_offset(std::uint32_t leg) {
  return leg == 0u ? DevelopmentalAppendOffset{12, 0, 0}
                   : DevelopmentalAppendOffset{0, 0, 12};
}

// Physical receptor ports are intake-only.  Once a source/teacher/clock
// triplet is accepted, append transfers the spent source vacancy to this
// leg-specific ingress before the persistent credit ring advances.  Held-out
// operand traffic can then reuse the physical intake at any ring phase without
// colliding with the retained lesson witness.
[[nodiscard]] __host__ __device__ constexpr DevelopmentalAppendOffset
developmental_append_accepted_source_ingress_offset(std::uint32_t leg) {
  return leg == 0u ? DevelopmentalAppendOffset{-3, 3, 0}
                   : DevelopmentalAppendOffset{0, 3, -3};
}

[[nodiscard]] __host__ __device__ constexpr DevelopmentalAppendOffset
developmental_append_teacher_offset() {
  return {-1, 1, 0};
}

// The common teacher site is an intake, not persistent service storage.
// Accepted append atomically transfers its spent witness to this leg-specific
// ingress before the closed credit ring advances.  Distinct ingresses keep an
// older lawful lesson from periodically reoccupying the shared intake.
[[nodiscard]] __host__ __device__ constexpr DevelopmentalAppendOffset
developmental_append_accepted_teacher_ingress_offset(std::uint32_t leg) {
  return leg == 0u ? DevelopmentalAppendOffset{1, -1, 0}
                   : DevelopmentalAppendOffset{0, -1, 1};
}

[[nodiscard]] __host__ __device__ constexpr SiteWord
developmental_append_teacher_vacancy_word(std::uint32_t leg,
                                          std::uint32_t marker,
                                          std::uint32_t waste) {
  return kQuiescentWord ^ carrier_bit(
                              developmental_append_receptor_basis(
                                  leg, marker, waste));
}

// A teaching vacancy is consumed into a leg-specific service role.  A uses
// path and B uses the unused fourth basis, so the postimage retains local
// inverse provenance until the resident credit-service ring captures it.
// Both roles are distinct from the fresh marker/waste words in every S4 frame.
[[nodiscard]] __host__ __device__ constexpr std::uint32_t
developmental_append_spent_teacher_basis(std::uint32_t leg,
                                         std::uint32_t marker,
                                         std::uint32_t path,
                                         std::uint32_t waste) {
  (void)marker;
  return leg == 0u ? path
                   : developmental_append_free_basis(marker, path, waste);
}

[[nodiscard]] __host__ __device__ constexpr SiteWord
developmental_append_spent_teacher_vacancy_word(std::uint32_t leg,
                                                std::uint32_t marker,
                                                std::uint32_t path,
                                                std::uint32_t waste) {
  return kQuiescentWord ^ carrier_bit(
                              developmental_append_spent_teacher_basis(
                                  leg, marker, path, waste));
}

[[nodiscard]] __host__ __device__ constexpr SiteWord
developmental_append_armed_head_word(std::uint32_t marker,
                                     std::uint32_t path) {
  return kQuiescentWord | face_bit(marker) |
         channel_bit(kReactiveShift, path);
}

[[nodiscard]] __host__ __device__ constexpr SiteWord
developmental_append_owner_one_word(std::uint32_t marker,
                                    std::uint32_t path) {
  return kQuiescentWord | face_bit(marker) |
         channel_bit(kReactiveShift, path);
}

[[nodiscard]] __host__ __device__ constexpr SiteWord
developmental_append_owner_two_word(std::uint32_t waste,
                                    std::uint32_t free_basis) {
  (void)free_basis;
  // A differentiated singleton face is a complete-F fixed point.  It names
  // the third role without carrying a timing-sensitive energy phase.
  return kQuiescentWord | face_bit(waste);
}

// The adult head is the one-shot physical claim shared by learned capture and
// processive release. Positive marker face is resting/processed; negative
// marker face is captured/unconsumed. Both are differentiated singleton fixed
// points with identical represented matter.
[[nodiscard]] __host__ __device__ constexpr SiteWord
developmental_append_receptor_head_word(bool captured,
                                         std::uint32_t marker) {
  return kQuiescentWord | face_bit(marker + (captured ? 4u : 0u));
}

[[nodiscard]] __host__ __device__ constexpr SiteWord
developmental_append_retained_hole_anchor_word(std::uint32_t axis) {
  return (kCarrierMask ^ carrier_bit(axis) ^ carrier_bit(axis + 4u)) |
         face_bit(axis + 4u) | energy_bit(axis);
}

[[nodiscard]] __host__ __device__ constexpr SiteWord
developmental_append_retained_hole_lock_word(std::uint32_t axis) {
  return (kCarrierMask ^ carrier_bit(axis) ^ carrier_bit(axis + 4u)) |
         owned_bond_bit(axis);
}

[[nodiscard]] __host__ __device__ constexpr SiteWord
developmental_append_age_word(std::uint32_t index, std::uint32_t marker,
                              std::uint32_t path, std::uint32_t waste,
                              std::uint64_t age) {
  const std::uint32_t relative = index - kDevelopmentalAppendAgeFirst;
  const std::uint32_t component =
      relative % kDevelopmentalAppendAgeSitesPerDigit;
  if (component == 0u)
    return developmental_append_retained_hole_anchor_word(marker);
  if (component == 1u)
    return developmental_append_retained_hole_lock_word(marker);
  const std::uint32_t digit =
      relative / kDevelopmentalAppendAgeSitesPerDigit;
  const std::uint32_t encoded =
      static_cast<std::uint32_t>((age >> (2u * digit)) & 3u);
  const std::uint32_t free_basis =
      developmental_append_free_basis(marker, path, waste);
  const std::uint32_t role_basis[4] = {marker, path, waste, free_basis};
  return kQuiescentWord | face_bit(role_basis[encoded]);
}

[[nodiscard]] __host__ __device__ constexpr SiteWord
developmental_append_journal_word(std::uint32_t index,
                                  std::uint32_t marker,
                                  std::uint32_t path,
                                  std::uint32_t waste,
                                  std::uint32_t encoded) {
  const std::uint32_t relative = index - kDevelopmentalAppendJournalFirst;
  const std::uint32_t component =
      relative % kDevelopmentalAppendAgeSitesPerDigit;
  if (component == 0u)
    return developmental_append_retained_hole_anchor_word(marker);
  if (component == 1u)
    return developmental_append_retained_hole_lock_word(marker);
  const std::uint32_t free_basis =
      developmental_append_free_basis(marker, path, waste);
  const std::uint32_t role_basis[4] = {marker, path, waste, free_basis};
  return kQuiescentWord | face_bit(role_basis[encoded & 3u]);
}

[[nodiscard]] __host__ __device__ constexpr SiteWord
developmental_append_word(bool product, std::uint32_t index,
                          std::uint32_t marker, std::uint32_t path,
                          std::uint32_t waste) {
  const std::uint32_t free_basis =
      developmental_append_free_basis(marker, path, waste);
  if (!product) {
    if (index == 0u)
      return developmental_append_armed_head_word(marker, path);
    if (index == 1u)
      return developmental_append_owner_one_word(marker, path);
    if (index == 2u)
      return developmental_append_owner_two_word(waste, free_basis);
    if (index == 3u) return kQuiescentWord ^ carrier_bit(path);
    if (index >= 4u && index <= 9u)
      return kQuiescentWord | face_bit(free_basis);
    return kQuiescentWord;
  }

  if (index >= kDevelopmentalAppendJournalFirst)
    return developmental_append_journal_word(index, marker, path, waste, 0u);
  if (index >= kDevelopmentalAppendAgeFirst)
    return developmental_append_age_word(index, marker, path, waste, 0u);

  if (index == 0u)
    return developmental_append_receptor_head_word(false, marker);
  if (index >= 1u && index <= 3u) return kQuiescentWord;
  if (index >= kDevelopmentalAppendRetainedHoleAnchorFirst && index < 10u)
    return developmental_append_retained_hole_anchor_word(marker);
  if (index == 10u)
    return developmental_append_retained_hole_anchor_word(marker);
  if (index == 11u)
    return developmental_append_retained_hole_lock_word(marker);
  if (index == 12u) return kQuiescentWord | face_bit(free_basis);
  if (index == 13u) return kQuiescentWord | face_bit(marker);
  if (index == 14u)
    return developmental_append_armed_head_word(marker, path);
  if (index == 15u)
    return developmental_append_owner_one_word(marker, path);
  if (index == 16u)
    return developmental_append_owner_two_word(waste, free_basis);
  // Three differentiated singleton faces represent the construction surplus
  // without launching an unbounded terminal-waste orbit under complete F.
  if (index == 17u) return kQuiescentWord | face_bit(waste);
  if (index >= kDevelopmentalAppendChildFeedstockFirst &&
      index < kDevelopmentalAppendChildFeedstockFirst +
                  kDevelopmentalAppendChildFeedstockCount)
    return kQuiescentWord | face_bit(free_basis);
  if (index >= kDevelopmentalAppendRetainedHoleLockFirst &&
      index < kDevelopmentalAppendRetainedHoleLockFirst +
                  kDevelopmentalAppendRetainedHoleCount)
    return developmental_append_retained_hole_lock_word(marker);
  if (index == 30u) return kQuiescentWord | face_bit(marker);
  if (index == 31u) return kQuiescentWord | face_bit(path);
  return kQuiescentWord;
}

[[nodiscard]] __host__ __device__ constexpr SiteWord
developmental_append_product_word(std::uint32_t index, std::uint32_t marker,
                                  std::uint32_t path, std::uint32_t waste,
                                  std::uint64_t age) {
  if (index >= kDevelopmentalAppendJournalFirst)
    return developmental_append_journal_word(index, marker, path, waste, 0u);
  if (index >= kDevelopmentalAppendAgeFirst)
    return developmental_append_age_word(index, marker, path, waste, age);
  return developmental_append_word(true, index, marker, path, waste);
}

// Applies one immutable-snapshot spatial factor. Forward recognizes only the
// resource preimage; inverse recognizes only the fully represented product.
void apply_k_developmental_append(ReferenceLattice& lattice, bool inverse);

}  // namespace substrate::bcc32
