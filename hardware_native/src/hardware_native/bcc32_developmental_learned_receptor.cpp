#include "bcc32_developmental_learned_receptor.hpp"

#include <array>
#include <cstdint>
#include <utility>
#include <vector>

#include "bcc32_developmental_append.hpp"
#include "bcc32_law.cuh"
#include "bcc32_reference.hpp"

namespace substrate::bcc32 {
namespace {

inline constexpr std::uint32_t kLearnedJournalInvalid = 3u;

struct Match {
  Z3Coordinate center{};
  BasisPermutation permutation{};
  std::uint32_t leg = kDevelopmentalAppendReceptorLegCount;
  std::uint32_t event_digit = kDevelopmentalAppendJournalDigitCount;
  Z3Coordinate port{};
  Z3Coordinate inlet{};
  Z3Coordinate journal{};
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

[[nodiscard]] Z3Coordinate shifted(
    const Z3Coordinate& center, const BasisPermutation& permutation,
    DevelopmentalAppendOffset offset) {
  return center + transformed_coordinate(
                      {offset.marker, offset.path, offset.waste}, permutation);
}

[[nodiscard]] Z3Coordinate relative(
    const Z3Coordinate& center, const BasisPermutation& permutation,
    std::uint32_t site) {
  return shifted(center, permutation, developmental_append_offset(site));
}

[[nodiscard]] std::uint32_t journal_state(
    const ReferenceLattice& world, const Z3Coordinate& center,
    const BasisPermutation& permutation, std::uint32_t digit) {
  const std::uint32_t site = developmental_append_journal_state_site(digit);
  const SiteWord observed = world.read(relative(center, permutation, site));
  for (std::uint32_t state = 0u; state <= 2u; ++state)
    if (observed == developmental_append_journal_word(
                        site, permutation[0u], permutation[1u],
                        permutation[2u], state))
      return state;
  return kLearnedJournalInvalid;
}

[[nodiscard]] bool owner_present(
    const ReferenceLattice& world, const Z3Coordinate& center,
    const BasisPermutation& permutation) {
  for (const std::uint32_t site :
       {11u, 12u, 13u, 30u, 31u})
    if (world.read(relative(center, permutation, site)) !=
        developmental_append_product_word(
            site, permutation[0u], permutation[1u], permutation[2u], 0u))
      return false;
  for (std::uint32_t digit = 0u;
       digit < kDevelopmentalAppendJournalDigitCount; ++digit) {
    const std::uint32_t first = kDevelopmentalAppendJournalFirst +
                                digit * kDevelopmentalAppendAgeSitesPerDigit;
    if (world.read(relative(center, permutation, first + 1u)) !=
        developmental_append_journal_word(
            first + 1u, permutation[0u], permutation[1u], permutation[2u],
            0u))
      return false;
    if (world.read(relative(center, permutation, first)) !=
        developmental_append_journal_word(
            first, permutation[0u], permutation[1u], permutation[2u], 0u))
      return false;
    const std::uint32_t state = journal_state(world, center, permutation, digit);
    if (digit < kDevelopmentalAppendAuthorityDigitCount) {
      if (state > kDevelopmentalAppendReceptorJournalB ||
          (state != kDevelopmentalAppendReceptorJournalEmpty &&
           state != digit + 1u))
        return false;
    } else if (digit == kDevelopmentalAppendEventJournalFirst ||
               digit == kDevelopmentalAppendEventJournalFirst + 1u) {
      if (state > kDevelopmentalAppendReceptorJournalB)
        return false;
    } else {
      if (state != kDevelopmentalAppendReceptorJournalEmpty)
        return false;
    }
  }
  return true;
}

[[nodiscard]] bool same_target(const Match& a, const Match& b) {
  const std::array<Z3Coordinate, 3u> left{a.port, a.inlet, a.journal};
  const std::array<Z3Coordinate, 3u> right{b.port, b.inlet, b.journal};
  for (const Z3Coordinate& first : left)
    for (const Z3Coordinate& second : right)
      if (first == second) return true;
  return false;
}

}  // namespace

void apply_k_developmental_learned_receptor(ReferenceLattice& lattice,
                                             bool inverse) {
  const ReferenceLattice snapshot = lattice;
  const std::vector<ReferenceSite> support = snapshot.support();
  const std::vector<BasisPermutation> permutations = all_permutations();
  std::vector<Match> matches;
  for (const ReferenceSite& candidate : support) {
    for (const BasisPermutation& permutation : permutations) {
      if (candidate.word != developmental_append_product_word(
                                11u, permutation[0u], permutation[1u],
                                permutation[2u], 0u))
        continue;
      const Z3Coordinate center =
          candidate.coordinate -
          transformed_coordinate(
              {developmental_append_offset(11u).marker,
               developmental_append_offset(11u).path,
               developmental_append_offset(11u).waste},
              permutation);
      if (!owner_present(snapshot, center, permutation)) continue;
      if (snapshot.read(shifted(center, permutation,
                                developmental_append_teacher_offset())) != kQ)
        continue;
      const Z3Coordinate inlet = relative(
          center, permutation, kDevelopmentalAppendReceptorInletSite);
      const SiteWord inlet_bit = carrier_bit(permutation[1u]);
      if ((!inverse && (snapshot.read(inlet) & inlet_bit) == 0u) ||
          (inverse && (snapshot.read(inlet) & inlet_bit) != 0u))
        continue;

      std::array<Z3Coordinate, kDevelopmentalAppendReceptorLegCount> ports{};
      for (std::uint32_t leg = 0u;
           leg < kDevelopmentalAppendReceptorLegCount; ++leg)
        ports[leg] = shifted(center, permutation,
                             developmental_append_receptor_port_offset(leg));
      std::uint32_t leg = kDevelopmentalAppendReceptorLegCount;
      const std::uint32_t event_digit =
          kDevelopmentalAppendEventJournalFirst;
      if (!inverse) {
        for (std::uint32_t candidate_leg = 0u;
             candidate_leg < kDevelopmentalAppendReceptorLegCount;
             ++candidate_leg) {
          const std::uint32_t basis = developmental_append_receptor_basis(
              candidate_leg, permutation[0u], permutation[2u]);
          const SiteWord port_word = snapshot.read(ports[candidate_leg]);
          if (port_word == (kQ ^ carrier_bit(basis))) {
            if (leg != kDevelopmentalAppendReceptorLegCount) {
              leg = kDevelopmentalAppendReceptorLegCount;
              break;
            }
            leg = candidate_leg;
          } else if (port_word != kQ) {
            leg = kDevelopmentalAppendReceptorLegCount;
            break;
          }
        }
        if (leg == kDevelopmentalAppendReceptorLegCount) continue;
        if (journal_state(snapshot, center, permutation, event_digit) !=
            kDevelopmentalAppendReceptorJournalEmpty)
          continue;
      } else {
        bool ports_valid = true;
        for (const Z3Coordinate& port : ports)
          if (snapshot.read(port) != kQ) {
            ports_valid = false;
            break;
        }
        if (!ports_valid) continue;
        const std::uint32_t state =
            journal_state(snapshot, center, permutation, event_digit);
        if (state == kDevelopmentalAppendReceptorJournalA ||
            state == kDevelopmentalAppendReceptorJournalB)
          leg = state - 1u;
      }
      if (leg == kDevelopmentalAppendReceptorLegCount ||
          event_digit >= kDevelopmentalAppendJournalDigitCount)
        continue;
      if (journal_state(snapshot, center, permutation, leg) != leg + 1u)
        continue;
      const std::uint32_t authority_site =
          developmental_append_receptor_authority_site(leg);
      if (snapshot.read(relative(center, permutation, authority_site)) !=
          developmental_append_journal_word(
              authority_site, permutation[0u], permutation[1u],
              permutation[2u], 0u))
        continue;
      Match match;
      match.center = center;
      match.permutation = permutation;
      match.leg = leg;
      match.event_digit = event_digit;
      match.port = ports[leg];
      match.inlet = inlet;
      match.journal = relative(
          center, permutation,
          developmental_append_journal_state_site(event_digit));
      matches.push_back(match);
    }
  }

  ReferenceLattice result = snapshot;
  for (std::size_t index = 0u; index < matches.size(); ++index) {
    bool collision = false;
    for (std::size_t other = 0u; other < matches.size(); ++other)
      if (index != other && same_target(matches[index], matches[other])) {
        collision = true;
        break;
      }
    if (collision) continue;
    const Match& match = matches[index];
    const std::uint32_t basis = developmental_append_receptor_basis(
        match.leg, match.permutation[0u], match.permutation[2u]);
    result.write(match.port,
                 snapshot.read(match.port) ^ carrier_bit(basis));
    result.write(match.inlet,
                 snapshot.read(match.inlet) ^
                     carrier_bit(match.permutation[1u]));
    const std::uint32_t journal_site =
        developmental_append_journal_state_site(match.event_digit);
    result.write(
        match.journal,
        developmental_append_journal_word(
            journal_site, match.permutation[0u], match.permutation[1u],
            match.permutation[2u],
            inverse ? kDevelopmentalAppendReceptorJournalEmpty
                    : match.leg + 1u));
  }
  lattice = std::move(result);
}

}  // namespace substrate::bcc32
