#include "bcc32_developmental_append.hpp"

#include <array>
#include <cstdint>
#include <utility>
#include <vector>

#include "bcc32_law.cuh"
#include "bcc32_reference.hpp"

namespace substrate::bcc32 {
namespace {

struct PatternSite {
  Z3Coordinate coordinate{};
  SiteWord before = kQ;
  SiteWord after = kQ;
};

struct Owner {
  Z3Coordinate center{};
  BasisPermutation permutation{};
  std::array<Z3Coordinate, kDevelopmentalAppendSiteCount> footprint{};
  bool product = false;
};

struct Match {
  std::array<PatternSite, kDevelopmentalAppendSiteCount> sites{};
  Z3Coordinate center{};
  std::uint32_t owner_index = 0u;
  bool ledger_only = false;
};

[[nodiscard]] Z3Coordinate relative(
    std::uint32_t index, const BasisPermutation& permutation);

[[nodiscard]] bool same_coordinate(const Z3Coordinate& first,
                                   const Z3Coordinate& second) {
  return first == second;
}

[[nodiscard]] bool footprints_overlap(
    const std::array<Z3Coordinate, kDevelopmentalAppendSiteCount>& first,
    const std::array<Z3Coordinate, kDevelopmentalAppendSiteCount>& second) {
  for (const Z3Coordinate& left : first)
    for (const Z3Coordinate& right : second)
      if (same_coordinate(left, right)) return true;
  return false;
}

[[nodiscard]] bool lineage_handoff(const Owner& first, const Owner& second) {
  if (first.permutation != second.permutation) return false;
  const Owner* parent = &first;
  const Owner* child = &second;
  if (second.center !=
      first.center +
          relative(kDevelopmentalAppendChildHead, first.permutation)) {
    parent = &second;
    child = &first;
    if (child->center !=
        parent->center +
            relative(kDevelopmentalAppendChildHead, parent->permutation))
      return false;
  }
  std::uint32_t shared = 0u;
  for (std::uint32_t left = 0u; left < parent->footprint.size(); ++left) {
    for (std::uint32_t right = 0u; right < child->footprint.size(); ++right) {
      if (parent->footprint[left] != child->footprint[right]) continue;
      if (!developmental_append_handoff_pair(left, right))
        return false;
      ++shared;
    }
  }
  return shared == kDevelopmentalAppendHandoffSiteCount;
}

[[nodiscard]] std::vector<BasisPermutation> all_permutations() {
  std::vector<BasisPermutation> result;
  for (std::uint32_t marker = 0u; marker < 4u; ++marker) {
    for (std::uint32_t path = 0u; path < 4u; ++path) {
      if (path == marker) continue;
      for (std::uint32_t waste = 0u; waste < 4u; ++waste) {
        if (waste == marker || waste == path) continue;
        BasisPermutation permutation{};
        permutation[0u] = marker;
        permutation[1u] = path;
        permutation[2u] = waste;
        for (std::uint32_t basis = 0u; basis < 4u; ++basis)
          if (basis != marker && basis != path && basis != waste)
            permutation[3u] = basis;
        result.push_back(permutation);
      }
    }
  }
  return result;
}

[[nodiscard]] Z3Coordinate relative(std::uint32_t index,
                                    const BasisPermutation& permutation) {
  const DevelopmentalAppendOffset offset = developmental_append_offset(index);
  return transformed_coordinate({offset.marker, offset.path, offset.waste},
                                permutation);
}

[[nodiscard]] Z3Coordinate relative(
    DevelopmentalAppendOffset offset,
    const BasisPermutation& permutation) {
  return transformed_coordinate({offset.marker, offset.path, offset.waste},
                                permutation);
}

[[nodiscard]] SiteWord phase_word(bool product, std::uint32_t index,
                                  const BasisPermutation& permutation) {
  return developmental_append_word(product, index, permutation[0u],
                                   permutation[1u], permutation[2u]);
}

[[nodiscard]] SiteWord product_word(std::uint32_t index,
                                    const BasisPermutation& permutation,
                                    std::uint64_t age) {
  return developmental_append_product_word(
      index, permutation[0u], permutation[1u], permutation[2u], age);
}

[[nodiscard]] bool exact_preimage(const ReferenceLattice& snapshot,
                                  const Owner& owner) {
  for (std::uint32_t index = 0u; index < owner.footprint.size(); ++index)
    if (snapshot.read(owner.footprint[index]) !=
        phase_word(false, index, owner.permutation))
      return false;
  return true;
}

[[nodiscard]] bool decode_product_age(const ReferenceLattice& snapshot,
                                      const Owner& owner,
                                      std::uint64_t* age) {
  for (std::uint32_t index = 0u; index < kDevelopmentalAppendAgeFirst;
       ++index) {
    if (index == 10u) continue;  // mutable processive/receptor inlet
    if (snapshot.read(owner.footprint[index]) !=
        product_word(index, owner.permutation, 0u)) {
      if (index < kDevelopmentalAppendChildHead ||
          !developmental_append_handoff_pair(
              index, index - kDevelopmentalAppendChildHead) ||
          snapshot.read(owner.footprint[index]) !=
              product_word(index - kDevelopmentalAppendChildHead,
                           owner.permutation, 0u))
        return false;
    }
  }
  std::uint64_t decoded = 0u;
  for (std::uint32_t digit = 0u;
       digit < kDevelopmentalAppendAgeDigitCount; ++digit) {
    const std::uint32_t first = kDevelopmentalAppendAgeFirst +
                                digit * kDevelopmentalAppendAgeSitesPerDigit;
    if (snapshot.read(owner.footprint[first]) !=
            product_word(first, owner.permutation, 0u) ||
        snapshot.read(owner.footprint[first + 1u]) !=
            product_word(first + 1u, owner.permutation, 0u))
      return false;
    const SiteWord observed = snapshot.read(owner.footprint[first + 2u]);
    std::uint32_t encoded = 0u;
    while (encoded < 4u &&
           observed != product_word(first + 2u, owner.permutation,
                                    std::uint64_t{encoded} << (2u * digit)))
      ++encoded;
    if (encoded == 4u) return false;
    decoded |= std::uint64_t{encoded} << (2u * digit);
  }
  for (std::uint32_t digit = 0u;
       digit < kDevelopmentalAppendJournalDigitCount; ++digit) {
    const std::uint32_t first = kDevelopmentalAppendJournalFirst +
                                digit * kDevelopmentalAppendAgeSitesPerDigit;
    if (snapshot.read(owner.footprint[first]) !=
            developmental_append_journal_word(
                first, owner.permutation[0u], owner.permutation[1u],
                owner.permutation[2u], 0u) ||
        snapshot.read(owner.footprint[first + 1u]) !=
            developmental_append_journal_word(
                first + 1u, owner.permutation[0u], owner.permutation[1u],
                owner.permutation[2u], 0u))
      return false;
    const SiteWord observed = snapshot.read(owner.footprint[first + 2u]);
    bool valid = false;
    for (std::uint32_t encoded = 0u; encoded <= 2u; ++encoded)
      valid = valid ||
              observed == developmental_append_journal_word(
                              first + 2u, owner.permutation[0u],
                              owner.permutation[1u], owner.permutation[2u],
                              encoded);
    if (!valid) return false;
  }
  *age = decoded;
  return true;
}

[[nodiscard]] Z3Coordinate receptor_port(
    const Z3Coordinate& center, const BasisPermutation& permutation,
    std::uint32_t leg) {
  const DevelopmentalAppendOffset offset =
      developmental_append_receptor_port_offset(leg);
  return center + transformed_coordinate(
                      {offset.marker, offset.path, offset.waste}, permutation);
}

[[nodiscard]] Z3Coordinate receptor_teacher(
    const Z3Coordinate& center, const BasisPermutation& permutation) {
  const DevelopmentalAppendOffset offset = developmental_append_teacher_offset();
  return center + transformed_coordinate(
                      {offset.marker, offset.path, offset.waste}, permutation);
}

[[nodiscard]] bool receptor_owner_present(
    const ReferenceLattice& snapshot, const Z3Coordinate& center,
    const BasisPermutation& permutation) {
  for (const std::uint32_t site :
       {0u, 11u, 12u, 13u, 30u, 31u})
    if (snapshot.read(center + relative(site, permutation)) !=
        phase_word(true, site, permutation))
      return false;
  bool event_empty_seen = false;
  for (std::uint32_t digit = 0u;
       digit < kDevelopmentalAppendJournalDigitCount; ++digit) {
    const std::uint32_t first = kDevelopmentalAppendJournalFirst +
                                digit * kDevelopmentalAppendAgeSitesPerDigit;
    const bool leg_authority =
        digit < kDevelopmentalAppendReceptorLegCount;
    if ((!leg_authority &&
         snapshot.read(center + relative(first, permutation)) !=
            developmental_append_journal_word(
                first, permutation[0u], permutation[1u], permutation[2u],
                0u)) ||
        snapshot.read(center + relative(first + 1u, permutation)) !=
            developmental_append_journal_word(
                first + 1u, permutation[0u], permutation[1u],
                permutation[2u], 0u))
      return false;
    const SiteWord state =
        snapshot.read(center + relative(first + 2u, permutation));
    std::uint32_t decoded = 3u;
    for (std::uint32_t encoded = 0u; encoded <= 2u; ++encoded)
      if (state == developmental_append_journal_word(
                       first + 2u, permutation[0u], permutation[1u],
                       permutation[2u], encoded))
        decoded = encoded;
    if (decoded > kDevelopmentalAppendReceptorJournalB)
      return false;
    if (digit < kDevelopmentalAppendAuthorityDigitCount) {
      if (decoded != kDevelopmentalAppendReceptorJournalEmpty &&
          decoded != digit + 1u)
        return false;
    } else {
      if (event_empty_seen &&
          decoded != kDevelopmentalAppendReceptorJournalEmpty)
        return false;
      event_empty_seen = event_empty_seen ||
                         decoded == kDevelopmentalAppendReceptorJournalEmpty;
    }
  }
  return true;
}

[[nodiscard]] std::uint32_t receptor_journal_state(
    const ReferenceLattice& snapshot, const Z3Coordinate& center,
    const BasisPermutation& permutation, std::uint32_t digit) {
  const std::uint32_t site = kDevelopmentalAppendJournalFirst +
                             digit * kDevelopmentalAppendAgeSitesPerDigit + 2u;
  const SiteWord observed = snapshot.read(center + relative(site, permutation));
  for (std::uint32_t encoded = 0u; encoded <= 2u; ++encoded)
    if (observed == developmental_append_journal_word(
                        site, permutation[0u], permutation[1u],
                        permutation[2u], encoded))
      return encoded;
  return 3u;
}

void apply_receptor_transactions(const ReferenceLattice& snapshot,
                                 ReferenceLattice* result, bool inverse) {
  const std::vector<ReferenceSite> support = snapshot.support();
  const std::vector<BasisPermutation> permutations = all_permutations();
  for (const ReferenceSite& candidate : support) {
    for (const BasisPermutation& permutation : permutations) {
      if (candidate.word != phase_word(true, 11u, permutation)) continue;
      const Z3Coordinate center = candidate.coordinate - relative(11u, permutation);
      if (!receptor_owner_present(snapshot, center, permutation)) continue;
      const SiteWord teacher_word =
          snapshot.read(receptor_teacher(center, permutation));
      const Z3Coordinate inlet = center + relative(10u, permutation);
      const SiteWord inlet_word = snapshot.read(inlet);
      const SiteWord inlet_bit = carrier_bit(permutation[1u]);
      if ((inlet_word & inlet_bit) == 0u)
        continue;

      std::uint32_t selected_leg = kDevelopmentalAppendReceptorLegCount;
      std::uint32_t selected_digit = kDevelopmentalAppendJournalDigitCount;
      bool selected_source_accepted = false;
      for (std::uint32_t leg = 0u;
           leg < kDevelopmentalAppendReceptorLegCount; ++leg) {
        const std::uint32_t basis =
            inverse ? developmental_append_spent_source_basis(
                          leg, permutation[0u], permutation[1u],
                          permutation[2u])
                    : developmental_append_receptor_basis(
                          leg, permutation[0u], permutation[2u]);
        const SiteWord source_word = kQ ^ carrier_bit(basis);
        const SiteWord port_word =
            snapshot.read(receptor_port(center, permutation, leg));
        bool source_matches = port_word == source_word;
        bool source_accepted = false;
        if (inverse && port_word == kQ) {
          const SiteWord accepted_source_word = snapshot.read(
              center + relative(
                           developmental_append_accepted_source_ingress_offset(
                               leg),
                           permutation));
          const SiteWord filled_escrow = developmental_append_escrow_word(
              true, leg, permutation[0u], permutation[2u]);
          std::uint32_t top_bank = kDevelopmentalAppendWitnessBankCount;
          bool valid_bank = true;
          bool empty_seen = false;
          for (std::uint32_t bank = 0u;
               bank < kDevelopmentalAppendWitnessBankCount; ++bank) {
            const SiteWord clock_state = snapshot.read(
                center + relative(developmental_append_clock_escrow_offset(
                                      leg, bank),
                                  permutation));
            const SiteWord reject_state = snapshot.read(
                center + relative(developmental_append_reject_escrow_offset(
                                      leg, bank),
                                  permutation));
            const bool occupied = clock_state == filled_escrow;
            valid_bank = valid_bank &&
                         (occupied || clock_state == kQ) &&
                         (reject_state == kQ ||
                          reject_state == filled_escrow) &&
                         (!empty_seen || !occupied) &&
                         (occupied || reject_state == kQ);
            if (occupied) top_bank = bank;
            empty_seen = empty_seen || !occupied;
          }
          const bool rejected_source =
              valid_bank &&
              top_bank != kDevelopmentalAppendWitnessBankCount &&
              snapshot.read(
                  center + relative(
                               developmental_append_reject_source_ingress_offset(
                                   leg, top_bank),
                               permutation)) == source_word;
          source_accepted = accepted_source_word == source_word;
          source_matches = source_accepted != rejected_source;
        }
        if (source_matches) {
          if (selected_leg != kDevelopmentalAppendReceptorLegCount) {
            selected_leg = kDevelopmentalAppendReceptorLegCount;
            break;
          }
          selected_leg = leg;
          selected_source_accepted = source_accepted;
        } else if (port_word != kQ) {
          selected_leg = kDevelopmentalAppendReceptorLegCount;
          break;
        }
      }
      if (selected_leg == kDevelopmentalAppendReceptorLegCount) continue;
      selected_digit = selected_leg;
      bool teaching = false;
      const std::uint32_t journal_state = receptor_journal_state(
          snapshot, center, permutation, selected_digit);
      const Z3Coordinate accepted_clock =
          center + relative(developmental_append_clock_ingress_offset(
                                selected_leg),
                            permutation);
      const Z3Coordinate accepted_teacher =
          center + relative(
                       developmental_append_accepted_teacher_ingress_offset(
                           selected_leg),
                           permutation);
      const Z3Coordinate accepted_source =
          center + relative(
                       developmental_append_accepted_source_ingress_offset(
                           selected_leg),
                       permutation);
      const SiteWord clock_vacancy =
          developmental_append_clock_vacancy_word(
              selected_leg, permutation[0u], permutation[1u],
              permutation[2u]);
      const SiteWord filled_escrow = developmental_append_escrow_word(
          true, selected_leg, permutation[0u], permutation[2u]);
      std::uint32_t witness_bank = kDevelopmentalAppendWitnessBankCount;
      bool empty_seen = false;
      bool valid_bank = true;
      for (std::uint32_t bank = 0u;
           bank < kDevelopmentalAppendWitnessBankCount; ++bank) {
        const SiteWord clock_state = snapshot.read(
            center + relative(developmental_append_clock_escrow_offset(
                                  selected_leg, bank),
                              permutation));
        const SiteWord reject_state = snapshot.read(
            center + relative(developmental_append_reject_escrow_offset(
                                  selected_leg, bank),
                              permutation));
        const bool occupied = clock_state == filled_escrow;
        valid_bank = valid_bank &&
                     (occupied || clock_state == kQ) &&
                     (reject_state == kQ || reject_state == filled_escrow) &&
                     (!empty_seen || !occupied) &&
                     (occupied || reject_state == kQ);
        if (inverse && occupied) witness_bank = bank;
        if (!inverse && !occupied &&
            witness_bank == kDevelopmentalAppendWitnessBankCount)
          witness_bank = bank;
        empty_seen = empty_seen || !occupied;
      }
      if (!valid_bank ||
          witness_bank == kDevelopmentalAppendWitnessBankCount)
        continue;
      const Z3Coordinate clock_escrow =
          center + relative(developmental_append_clock_escrow_offset(
                                selected_leg, witness_bank),
                            permutation);
      const Z3Coordinate reject_escrow =
          center + relative(developmental_append_reject_escrow_offset(
                                selected_leg, witness_bank),
                                permutation);
      const Z3Coordinate reject_source =
          center + relative(
                       developmental_append_reject_source_ingress_offset(
                           selected_leg, witness_bank),
                       permutation);
      const Z3Coordinate reject_teacher =
          center + relative(
                       developmental_append_reject_teacher_ingress_offset(
                           selected_leg, witness_bank),
                       permutation);
      const Z3Coordinate reject_clock =
          center + relative(
                       developmental_append_reject_clock_ingress_offset(
                           selected_leg, witness_bank),
                       permutation);
      if (!inverse) {
        if (teacher_word == developmental_append_teacher_vacancy_word(
                                selected_leg, permutation[0u],
                                permutation[2u])) {
          teaching = true;
        } else if (teacher_word != kQ) {
          continue;
        }
        if (journal_state != kDevelopmentalAppendReceptorJournalEmpty)
          continue;
        if (snapshot.read(accepted_source) != kQ ||
            snapshot.read(accepted_clock) != kQ ||
            snapshot.read(accepted_teacher) != kQ ||
            snapshot.read(reject_teacher) != kQ ||
            snapshot.read(reject_clock) != kQ ||
            snapshot.read(clock_escrow) != kQ ||
            snapshot.read(reject_escrow) != kQ)
          continue;
      } else {
        if (selected_source_accepted && teacher_word == kQ &&
            snapshot.read(accepted_teacher) ==
                developmental_append_spent_teacher_vacancy_word(
                    selected_leg, permutation[0u], permutation[1u],
                    permutation[2u]) &&
            journal_state == selected_leg + 1u &&
            snapshot.read(reject_escrow) == kQ &&
            snapshot.read(reject_teacher) == kQ &&
            snapshot.read(reject_clock) == kQ) {
          teaching = true;
        } else if (selected_source_accepted || teacher_word != kQ ||
                   snapshot.read(accepted_teacher) != kQ ||
                   journal_state !=
                       kDevelopmentalAppendReceptorJournalEmpty ||
                   snapshot.read(reject_escrow) != filled_escrow ||
                   snapshot.read(reject_teacher) !=
                       developmental_append_spent_teacher_vacancy_word(
                           selected_leg, permutation[0u], permutation[1u],
                           permutation[2u]) ||
                   snapshot.read(reject_clock) != clock_vacancy) {
          continue;
        }
        if (snapshot.read(teaching ? accepted_clock : reject_clock) !=
                clock_vacancy ||
            snapshot.read(clock_escrow) != filled_escrow)
          continue;
      }
      if (selected_leg == kDevelopmentalAppendReceptorLegCount ||
          selected_digit == kDevelopmentalAppendJournalDigitCount)
        continue;
      const std::uint32_t authority_site =
          developmental_append_receptor_authority_site(selected_leg);
      if (snapshot.read(center + relative(authority_site, permutation)) !=
          developmental_append_journal_word(
              authority_site, permutation[0u], permutation[1u],
              permutation[2u], 0u))
        continue;
      const std::uint32_t journal_site =
          developmental_append_journal_state_site(selected_digit);
      result->write(
          center + relative(journal_site, permutation),
          developmental_append_journal_word(
              journal_site, permutation[0u], permutation[1u],
              permutation[2u],
              teaching
                  ? (inverse ? kDevelopmentalAppendReceptorJournalEmpty
                             : selected_leg + 1u)
                  : kDevelopmentalAppendReceptorJournalEmpty));
      result->write(
          receptor_teacher(center, permutation),
          inverse && teaching
              ? developmental_append_teacher_vacancy_word(
                    selected_leg, permutation[0u], permutation[2u])
              : kQ);
      result->write(
          receptor_port(center, permutation, selected_leg),
          inverse
              ? (kQ ^ carrier_bit(developmental_append_receptor_basis(
                            selected_leg, permutation[0u], permutation[2u])))
              : kQ);
      if (teaching)
        result->write(
            accepted_source,
            inverse ? kQ
                    : developmental_append_spent_source_vacancy_word(
                          selected_leg, permutation[0u], permutation[1u],
                          permutation[2u]));
      if (!teaching)
        result->write(
            reject_source,
            inverse ? kQ
                    : developmental_append_spent_source_vacancy_word(
                          selected_leg, permutation[0u], permutation[1u],
                          permutation[2u]));
      if (teaching)
      {
        result->write(accepted_clock, inverse ? kQ : clock_vacancy);
        result->write(
            accepted_teacher,
            inverse ? kQ
                    : developmental_append_spent_teacher_vacancy_word(
                          selected_leg, permutation[0u], permutation[1u],
                          permutation[2u]));
      }
      else {
        result->write(
            reject_teacher,
            inverse ? kQ
                    : developmental_append_spent_teacher_vacancy_word(
                          selected_leg, permutation[0u], permutation[1u],
                          permutation[2u]));
        result->write(reject_clock, inverse ? kQ : clock_vacancy);
      }
      result->write(clock_escrow, inverse ? kQ : filled_escrow);
      result->write(reject_escrow,
                    inverse || teaching ? kQ : filled_escrow);
    }
  }
}

}  // namespace

void apply_k_developmental_append(ReferenceLattice& lattice, bool inverse) {
  if (inverse) {
    const ReferenceLattice receptor_snapshot = lattice;
    ReferenceLattice receptor_result = receptor_snapshot;
    apply_receptor_transactions(receptor_snapshot, &receptor_result, true);
    lattice = std::move(receptor_result);
  }
  const ReferenceLattice snapshot = lattice;
  const std::vector<ReferenceSite> support = snapshot.support();
  const std::vector<BasisPermutation> permutations = all_permutations();
  std::vector<Owner> owners;

  // Ownership is phase-independent: an opposite-phase product can block a
  // resource candidate and vice versa.  The only permitted overlap is the
  // exact nine-site physical handoff from a product's child tip to the next
  // resource preimage in the same orientation.
  for (const bool product : {false, true}) {
    const std::uint32_t first_lock =
        product ? 11u
                : kDevelopmentalAppendParentLockFirst;
    const std::uint32_t second_lock =
        product ? 12u
                : kDevelopmentalAppendParentLockSecond;
    for (const ReferenceSite& candidate : support) {
      for (const BasisPermutation& permutation : permutations) {
        if (candidate.word != phase_word(product, first_lock, permutation))
          continue;
        const Z3Coordinate center =
            candidate.coordinate - relative(first_lock, permutation);
        if (snapshot.read(center + relative(second_lock, permutation)) !=
            phase_word(product, second_lock, permutation))
          continue;
        if (product) {
          bool signature = true;
          for (const std::uint32_t signature_site :
               {0u, 11u, 12u, 13u, 17u, 30u, 31u,
                kDevelopmentalAppendJournalFirst,
                kDevelopmentalAppendJournalFirst + 1u})
            signature = signature &&
                        snapshot.read(center +
                                      relative(signature_site, permutation)) ==
                            phase_word(true, signature_site, permutation);
          if (!signature) continue;
        }
        Owner owner;
        owner.center = center;
        owner.permutation = permutation;
        owner.product = product;
        for (std::uint32_t index = 0u; index < owner.footprint.size(); ++index)
          owner.footprint[index] = center + relative(index, permutation);
        owners.push_back(std::move(owner));
      }
    }
  }

  std::vector<Match> matches;
  for (std::uint32_t owner_index = 0u; owner_index < owners.size(); ++owner_index) {
    const Owner& owner = owners[owner_index];
    std::uint64_t age = 0u;
    const bool exact = owner.product ? decode_product_age(snapshot, owner, &age)
                                     : exact_preimage(snapshot, owner);
    if (!exact) continue;
    bool after_product = true;
    std::uint64_t after_age = 0u;
    if (!owner.product) {
      after_age = inverse ? kDevelopmentalAppendMaxAge : 0u;
    } else if (!inverse && age == kDevelopmentalAppendMaxAge) {
      after_product = false;
    } else if (inverse && age == 0u) {
      after_product = false;
    } else {
      after_age = inverse ? age - 1u : age + 1u;
    }
    Match match;
    match.center = owner.center;
    match.owner_index = owner_index;
    match.ledger_only = owner.product && after_product;
    for (std::uint32_t index = 0u; index < match.sites.size(); ++index) {
      PatternSite& site = match.sites[index];
      site.coordinate = owner.footprint[index];
      site.before = owner.product
                        ? product_word(index, owner.permutation, age)
                        : phase_word(false, index, owner.permutation);
      site.after = after_product
                       ? product_word(index, owner.permutation, after_age)
                       : phase_word(false, index, owner.permutation);
    }
    matches.push_back(std::move(match));
  }

  ReferenceLattice result = snapshot;
  for (std::uint32_t index = 0u; index < matches.size(); ++index) {
    bool collided = false;
    std::array<Z3Coordinate, kDevelopmentalAppendSiteCount> footprint{};
    for (std::uint32_t site = 0u; site < footprint.size(); ++site)
      footprint[site] = matches[index].sites[site].coordinate;
    const Owner& own = owners[matches[index].owner_index];
    for (std::uint32_t owner_index = 0u; owner_index < owners.size(); ++owner_index) {
      if (owner_index == matches[index].owner_index) continue;
      const Owner& owner = owners[owner_index];
      if (!footprints_overlap(footprint, owner.footprint)) continue;
      if (!lineage_handoff(own, owner)) {
        collided = true;
        break;
      }
      if (inverse && !own.product && owner.product) {
        std::uint64_t parent_age = 0u;
        if (decode_product_age(snapshot, owner, &parent_age) &&
            parent_age == 0u) {
          collided = true;
          break;
        }
      }
    }
    if (collided) continue;
    for (std::uint32_t site_index = 0u;
         site_index < matches[index].sites.size(); ++site_index) {
      if (matches[index].ledger_only &&
          (site_index < kDevelopmentalAppendAgeFirst ||
           site_index >= kDevelopmentalAppendJournalFirst ||
           (site_index - kDevelopmentalAppendAgeFirst) %
                   kDevelopmentalAppendAgeSitesPerDigit !=
               2u))
        continue;
      const PatternSite& site = matches[index].sites[site_index];
      result.write(site.coordinate, site.after);
    }
  }
  if (!inverse) apply_receptor_transactions(snapshot, &result, false);
  lattice = std::move(result);
}

}  // namespace substrate::bcc32
