#include "bcc32_developmental_credit_service.hpp"

#include <cstdint>
#include <utility>
#include <vector>

#include "bcc32_developmental_append.hpp"
#include "bcc32_law.cuh"
#include "bcc32_reference.hpp"

namespace substrate::bcc32 {
namespace {

struct Match {
  Z3Coordinate target{};
  std::uint32_t incoming = 0u;
  std::uint32_t outgoing = 0u;
};

[[nodiscard]] std::vector<BasisPermutation> all_permutations() {
  std::vector<BasisPermutation> result;
  for (std::uint32_t marker = 0u; marker < 4u; ++marker)
    for (std::uint32_t path = 0u; path < 4u; ++path) {
      if (path == marker) continue;
      for (std::uint32_t waste = 0u; waste < 4u; ++waste) {
        if (waste == marker || waste == path) continue;
        BasisPermutation permutation{marker, path, waste, 0u};
        for (std::uint32_t basis = 0u; basis < 4u; ++basis)
          if (basis != marker && basis != path && basis != waste)
            permutation[3u] = basis;
        result.push_back(permutation);
      }
    }
  return result;
}

[[nodiscard]] Z3Coordinate relative(
    const Z3Coordinate& center, const BasisPermutation& permutation,
    std::uint32_t site) {
  const DevelopmentalAppendOffset offset = developmental_append_offset(site);
  return center + transformed_coordinate(
                      {offset.marker, offset.path, offset.waste}, permutation);
}

[[nodiscard]] bool base_owner_present(
    const ReferenceLattice& world, const Z3Coordinate& center,
    const BasisPermutation& permutation) {
  for (const std::uint32_t site : {0u, 11u, 12u, 13u, 30u, 31u})
    if ((world.read(relative(center, permutation, site)) & ~kCarrierMask) !=
        (developmental_append_product_word(
             site, permutation[0u], permutation[1u], permutation[2u], 0u) &
         ~kCarrierMask))
      return false;
  return true;
}

[[nodiscard]] bool ring_enabled(
    const ReferenceLattice& world, const Z3Coordinate& center,
    const BasisPermutation& permutation) {
  return developmental_credit_service_enable_word_matches(
      world.read(relative(center, permutation,
                          kDevelopmentalCreditServiceEnableSite)),
      permutation[0u], permutation[1u], permutation[2u]);
}

[[nodiscard]] Z3Coordinate external_relative(
    const Z3Coordinate& center, const BasisPermutation& permutation,
    DevelopmentalAppendOffset offset) {
  return center + transformed_coordinate(
                      {offset.marker, offset.path, offset.waste}, permutation);
}

[[nodiscard]] bool transaction_live(
    const ReferenceLattice& world, const Z3Coordinate& center,
    const BasisPermutation& permutation, std::uint32_t ring) {
  const std::uint32_t leg = developmental_credit_service_leg(ring);
  const SiteWord filled = developmental_append_escrow_word(
      true, leg, permutation[0u], permutation[2u]);
  if (developmental_credit_service_reject_source_ring(ring) ||
      developmental_credit_service_reject_teacher_ring(ring) ||
      developmental_credit_service_reject_clock_ring(ring)) {
    const std::uint32_t bank = developmental_credit_service_reject_bank(ring);
    return world.read(external_relative(
               center, permutation,
               developmental_append_clock_escrow_offset(leg, bank))) ==
               filled &&
           world.read(external_relative(
               center, permutation,
               developmental_append_reject_escrow_offset(leg, bank))) ==
               filled;
  }
  for (std::uint32_t bank = 0u;
       bank < kDevelopmentalAppendWitnessBankCount; ++bank)
    if (world.read(external_relative(
            center, permutation,
            developmental_append_clock_escrow_offset(leg, bank))) == filled &&
        world.read(external_relative(
            center, permutation,
            developmental_append_reject_escrow_offset(leg, bank))) == kQ)
      return true;
  return false;
}

}  // namespace

void apply_k_developmental_credit_service(ReferenceLattice& lattice,
                                          bool inverse) {
  (void)inverse;
  const ReferenceLattice snapshot = lattice;
  const std::vector<ReferenceSite> support = snapshot.support();
  const std::vector<BasisPermutation> permutations = all_permutations();
  std::vector<Match> matches;
  for (const ReferenceSite& candidate : support) {
    for (const BasisPermutation& permutation : permutations) {
      if ((candidate.word & ~kCarrierMask) !=
          (developmental_append_product_word(
               11u, permutation[0u], permutation[1u], permutation[2u], 0u) &
           ~kCarrierMask))
        continue;
      const Z3Coordinate center =
          candidate.coordinate -
          transformed_coordinate(
              {developmental_append_offset(11u).marker,
               developmental_append_offset(11u).path,
               developmental_append_offset(11u).waste},
              permutation);
      if (!base_owner_present(snapshot, center, permutation) ||
          !ring_enabled(snapshot, center, permutation))
        continue;
      for (std::uint32_t ring = 0u;
           ring < kDevelopmentalCreditServiceRingCount; ++ring) {
        if (!transaction_live(snapshot, center, permutation, ring)) continue;
        for (std::uint32_t corner = 0u; corner < 4u; ++corner) {
          const Int3 offset = developmental_credit_service_corner(
              ring, corner, permutation[0u], permutation[1u],
              permutation[2u]);
          const Z3Coordinate target = center + Z3Coordinate{
              offset.x, offset.y, offset.z};
          const std::uint32_t incoming =
              developmental_credit_service_corner_incoming(
                  ring, corner, permutation[0u], permutation[1u],
                  permutation[2u]);
          const std::uint32_t outgoing =
              developmental_credit_service_corner_outgoing(
                  ring, corner, permutation[0u], permutation[1u],
                  permutation[2u]);
          if (carrier_corner_center_matches(snapshot.read(target), incoming,
                                            outgoing))
            matches.push_back({target, incoming, outgoing});
        }
      }
    }
  }

  ReferenceLattice result = snapshot;
  for (std::size_t index = 0u; index < matches.size(); ++index) {
    bool unique = true;
    for (std::size_t other = 0u; other < matches.size(); ++other)
      if (index != other && matches[index].target == matches[other].target) {
        unique = false;
        break;
      }
    if (!unique) continue;
    const Match& match = matches[index];
    result.write(match.target,
                 carrier_corner_transpose(snapshot.read(match.target),
                                          match.incoming, match.outgoing));
  }
  lattice = std::move(result);
}

}  // namespace substrate::bcc32
