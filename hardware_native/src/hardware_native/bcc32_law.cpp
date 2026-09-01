#include <optional>
#include <stdexcept>

#include "bcc32_developmental_append.hpp"
#include "bcc32_developmental_credit_service.hpp"
#include "bcc32_developmental_learned_receptor.hpp"
#include "bcc32_law.cuh"

#include <array>
#include <vector>
#include "bcc32_eligibility_residual_junction.hpp"
#include "bcc32_prediction_residual_route_toggle.cuh"
#include "bcc32_reference.hpp"

namespace substrate::bcc32 {
namespace {

[[nodiscard]] Z3Coordinate coordinate_offset(Int3 offset) {
    return {offset.x, offset.y, offset.z};
}

// positive_offset / negative_offset now live in bcc32_reference.hpp so an
// external probe can build a neighbourhood through the same geometry.

void replace_masked(ReferenceLattice& lattice, const Z3Coordinate& coordinate,
                    SiteWord mask, SiteWord value) {
    lattice.write(coordinate,
                  static_cast<SiteWord>((lattice.read(coordinate) & ~mask) |
                                        (value & mask)));
}

[[nodiscard]] SiteWord source_edge_mask(std::uint32_t basis) {
    return carrier_bit(basis) | face_bit(basis) | owned_bond_bit(basis) |
           energy_bit(basis);
}

[[nodiscard]] SiteWord destination_edge_mask(std::uint32_t basis) {
    return carrier_bit(basis + 4u) | face_bit(basis + 4u) |
           channel_bit(kConformationShift, basis) |
           channel_bit(kReactiveShift, basis);
}

void apply_edge_factor(ReferenceLattice& lattice, bool inverse) {
    const ReferenceLattice before = lattice;
    ReferenceLattice after = before;
    const std::array<Z3Coordinate, 4> incoming_sources = {
        negative_offset(0u), negative_offset(1u), negative_offset(2u),
        negative_offset(3u)};
    const std::vector<Z3Coordinate> sources =
        before.causal_closure(incoming_sources);

    for (const Z3Coordinate& source : sources) {
        for (std::uint32_t basis = 0u; basis < 4u; ++basis) {
            const Z3Coordinate destination = source + positive_offset(basis);
            SiteWord source_word = before.read(source);
            SiteWord destination_word = before.read(destination);
            if (inverse) {
                apply_edge_block_inverse(source_word, destination_word, basis);
            } else {
                apply_edge_block_forward(source_word, destination_word, basis);
            }
            replace_masked(after, source, source_edge_mask(basis), source_word);
            replace_masked(after, destination, destination_edge_mask(basis),
                           destination_word);
        }
    }
    lattice = std::move(after);
}

void stream(ReferenceLattice& lattice, bool inverse) {
    const ReferenceLattice before = lattice;
    std::vector<Z3Coordinate> halo;
    halo.reserve(kStreamChannelCount);
    for (std::uint32_t index = 0u; index < kStreamChannelCount; ++index) {
        halo.push_back(coordinate_offset(
            direction_offset(stream_channel(index).direction)));
    }
    const std::vector<Z3Coordinate> destinations = before.causal_closure(halo);
    ReferenceLattice after;

    for (const Z3Coordinate& destination : destinations) {
        SiteWord result = before.read(destination) & ~kCarrierMask;
        for (std::uint32_t index = 0u; index < kStreamChannelCount; ++index) {
            const StreamChannel mapping = stream_channel(index);
            const Int3 offset = direction_offset(mapping.direction);
            const Z3Coordinate source =
                destination + coordinate_offset(inverse ? offset : -offset);
            const SiteWord channel = carrier_bit(mapping.channel);
            if (bit_is_set(before.read(source), channel)) {
                result |= channel;
            }
        }
        after.write(destination, result);
    }
    lattice = std::move(after);
}

}  // namespace

void apply_k_site(ReferenceLattice& lattice) {
    const std::vector<ReferenceSite> before = lattice.support();
    std::vector<ReferenceSite> after;
    after.reserve(before.size());
    for (ReferenceSite site : before) {
        apply_site_word_forward(site.word);
        if (site.word != kQ) {
            after.push_back(std::move(site));
        }
    }
    lattice.replace_support(after);
}

void apply_k_site_inverse(ReferenceLattice& lattice) {
    const std::vector<ReferenceSite> before = lattice.support();
    std::vector<ReferenceSite> after;
    after.reserve(before.size());
    for (ReferenceSite site : before) {
        apply_site_word_inverse(site.word);
        if (site.word != kQ) {
            after.push_back(std::move(site));
        }
    }
    lattice.replace_support(after);
}

void apply_k_edge(ReferenceLattice& lattice) {
    apply_edge_factor(lattice, false);
}

void apply_k_edge_inverse(ReferenceLattice& lattice) {
    apply_edge_factor(lattice, true);
}

void apply_k_carrier_pair_splitter(ReferenceLattice& lattice, bool inverse) {
  struct PatternSite {
    Z3Coordinate relative{};
    SiteWord staged = kQ;
    SiteWord released = kQ;
  };
  struct Match {
    std::array<PatternSite, kCarrierPairSplitterSiteCount> sites{};
    Z3Coordinate center{};
    bool staged_to_released = true;
  };
  struct Owner {
    std::array<Z3Coordinate, kCarrierPairSplitterSiteCount> footprint{};
    Z3Coordinate center{};
  };

  const auto relative = [](std::uint32_t incoming, std::uint32_t diverted, std::uint32_t index) {
    const CarrierPairSplitterOffset value = carrier_pair_splitter_offset(incoming, diverted, index);
    return Z3Coordinate{value.x, value.y, value.z};
  };

  const ReferenceLattice before = lattice;
  const std::vector<ReferenceSite> support = before.support();
  std::vector<Owner> owners;
  std::vector<Match> matches;

  // The centre word a candidate must equal depends only on (incoming, diverted),
  // never on the candidate, so it is computed once per call here instead of once
  // per candidate.  Index 2 and staged=false are the fixed arguments the pass-1
  // test below uses.
  std::array<std::array<SiteWord, 8u>, 8u> centre_word{};
  for (std::uint32_t incoming = 0u; incoming < 8u; ++incoming)
    for (std::uint32_t diverted = 0u; diverted < 8u; ++diverted)
      if (incoming != diverted)
        centre_word[incoming][diverted] =
            carrier_pair_splitter_word(incoming, diverted, 2u, false);

  // Owner discovery, native. Every footprint site is candidate + a SMALL
  // constant, so the base is narrowed once per candidate and the derivation
  // stays in machine words. Two things matter beyond the arithmetic:
  //
  //   * the exact footprint is no longer built before the test. It used to
  //     construct all thirteen arbitrary-precision coordinates and only then
  //     check site 3, so the overwhelming majority of that work was discarded.
  //     Now it tests natively, breaks on the first mismatch, and materializes
  //     exact coordinates ONLY for an owner that survived;
  //   * fallback is per CANDIDATE, never per site. A candidate is evaluated
  //     wholly natively or wholly exactly, so the two representations can never
  //     be mixed inside one owner.
  std::array<std::array<std::array<NativeCoordinate, kCarrierPairSplitterSiteCount>, 8u>, 8u>
      footprint_delta{};
  for (std::uint32_t incoming = 0u; incoming < 8u; ++incoming) {
    for (std::uint32_t diverted = 0u; diverted < 8u; ++diverted) {
      if (incoming == diverted)
        continue;
      const CarrierPairSplitterOffset centre = carrier_pair_splitter_offset(incoming, diverted, 2u);
      for (std::uint32_t index = 0u; index < kCarrierPairSplitterSiteCount; ++index) {
        const CarrierPairSplitterOffset site =
            carrier_pair_splitter_offset(incoming, diverted, index);
        // candidate + (site - centre) == (candidate - centre) + site, without
        // ever materializing the intermediate centre coordinate.
        footprint_delta[incoming][diverted][index] = native_offset(
            static_cast<std::int64_t>(site.x) - static_cast<std::int64_t>(centre.x),
            static_cast<std::int64_t>(site.y) - static_cast<std::int64_t>(centre.y),
            static_cast<std::int64_t>(site.z) - static_cast<std::int64_t>(centre.z));
        if (!native_offset_in_guard_band(footprint_delta[incoming][diverted][index]))
          throw std::logic_error("carrier pair splitter offset escapes the native guard band");
      }
    }
  }

  const ReferenceLattice::NativeReadView native_before = before.build_native_read_view();

  for (const ReferenceSite& candidate : support) {
    for (std::uint32_t incoming = 0u; incoming < 8u; ++incoming) {
      for (std::uint32_t diverted = 0u; diverted < 8u; ++diverted) {
        if (incoming == diverted)
          continue;
        if (candidate.word != centre_word[incoming][diverted])
          continue;

        const std::optional<NativeCoordinate> base =
            ReferenceLattice::try_narrow_guarded(candidate.coordinate);
        bool complete_owner = true;
        if (base) {
          for (std::uint32_t index = 3u; index < kCarrierPairSplitterSiteCount; ++index) {
            if (native_before.read(
                    add_unchecked(*base, footprint_delta[incoming][diverted][index])) !=
                carrier_pair_splitter_word(incoming, diverted, index, false)) {
              complete_owner = false;
              break;
            }
          }
        } else {
          // Exact path, preserved verbatim as the semantic oracle.
          const Z3Coordinate center = candidate.coordinate - relative(incoming, diverted, 2u);
          for (std::uint32_t index = 3u; index < kCarrierPairSplitterSiteCount; ++index)
            complete_owner =
                complete_owner && before.read(center + relative(incoming, diverted, index)) ==
                                      carrier_pair_splitter_word(incoming, diverted, index, false);
        }
        if (!complete_owner)
          continue;

        Owner owner;
        owner.center = candidate.coordinate - relative(incoming, diverted, 2u);
        for (std::uint32_t index = 0u; index < kCarrierPairSplitterSiteCount; ++index)
          owner.footprint[index] = owner.center + relative(incoming, diverted, index);
        owners.push_back(owner);
      }
    }
  }

  // The site geometry and its staged/released words depend only on
  // (incoming, diverted, index) -- never on the candidate.  Built lazily on the
  // first candidate that survives the carrier-only filter, so a superstep where
  // none survives costs nothing, exactly as before the hoist.
  std::array<std::array<std::array<PatternSite, kCarrierPairSplitterSiteCount>, 8u>, 8u>
      pair_sites{};
  bool pair_sites_built = false;
  const auto build_pair_sites = [&pair_sites, &relative]() {
    for (std::uint32_t incoming = 0u; incoming < 8u; ++incoming) {
      for (std::uint32_t diverted = 0u; diverted < 8u; ++diverted) {
        if (incoming == diverted)
          continue;
        for (std::uint32_t index = 0u; index < kCarrierPairSplitterSiteCount; ++index) {
          PatternSite& site = pair_sites[incoming][diverted][index];
          site.relative = relative(incoming, diverted, index);
          site.staged = carrier_pair_splitter_word(incoming, diverted, index, false);
          site.released = carrier_pair_splitter_word(incoming, diverted, index, true);
        }
      }
    }
  };

  for (const ReferenceSite& candidate : support) {
    if ((candidate.word & ~kCarrierMask) != 0u)
      continue;
    if (!pair_sites_built) {
      build_pair_sites();
      pair_sites_built = true;
    }
    for (std::uint32_t incoming = 0u; incoming < 8u; ++incoming) {
      for (std::uint32_t diverted = 0u; diverted < 8u; ++diverted) {
        if (incoming == diverted)
          continue;
        Match match;
        match.center = candidate.coordinate;
        bool staged_match = true;
        bool released_match = true;
        for (std::uint32_t index = 0u; index < kCarrierPairSplitterSiteCount; ++index) {
          const PatternSite& source = pair_sites[incoming][diverted][index];
          match.sites[index] = source;
          const SiteWord actual = before.read(match.center + source.relative);
          staged_match = staged_match && actual == source.staged;
          released_match = released_match && actual == source.released;
          // Only these two booleans are computed from a const read here, so once
          // both are false the remaining sites cannot change the test below.
          // match.sites is fully populated before any use: a break leaves later
          // entries at their default, and they are only read when a match is
          // recorded, which requires reaching the end of this loop.
          if (!staged_match && !released_match)
            break;
        }
        if (!staged_match && !released_match)
          continue;
        match.staged_to_released = staged_match;
        matches.push_back(match);
        goto next_pair_center;
      }
    }
  next_pair_center:;
  }

  ReferenceLattice after = before;
  for (const Match& match : matches) {
    bool collided = false;
    for (const Owner& owner : owners) {
      if (owner.center == match.center)
        continue;
      for (const PatternSite& first : match.sites) {
        for (const Z3Coordinate& second : owner.footprint) {
          if (match.center + first.relative == second) {
            collided = true;
            break;
          }
        }
        if (collided)
          break;
      }
      if (collided)
        break;
    }
    if (collided)
      continue;
    for (const PatternSite& site : match.sites)
      after.write(match.center + site.relative,
                  match.staged_to_released ? site.released : site.staged);
  }
  lattice = std::move(after);
}

void apply_k_processive_rearm(ReferenceLattice& lattice, bool inverse) {
  (void)inverse;
  struct PatternSite {
    Z3Coordinate relative{};
    SiteWord candidate = kQ;
    SiteWord rearmed = kQ;
  };
  struct Match {
    std::array<PatternSite, kProcessiveRearmSiteCount> sites{};
    Z3Coordinate center{};
    bool candidate_to_rearmed = true;
  };
  struct Owner {
    std::array<Z3Coordinate, kProcessiveRearmSiteCount> footprint{};
    Z3Coordinate center{};
    BasisPermutation permutation{};
  };

  const ReferenceLattice before = lattice;
  const std::vector<ReferenceSite> support = before.support();
  std::vector<Owner> owners;
  std::vector<Match> matches;

  // The basis permutation and the lock word a candidate must equal depend only on
  // (marker, path, waste) -- never on the candidate -- so both are built once per
  // call instead of once per candidate.  Indexed by the same triple the loop
  // below walks, so iteration order is unchanged.
  std::array<std::array<std::array<BasisPermutation, 4u>, 4u>, 4u> rearm_permutation{};
  std::array<std::array<std::array<SiteWord, 4u>, 4u>, 4u> rearm_lock_word{};
  for (std::uint32_t marker = 0u; marker < 4u; ++marker) {
    for (std::uint32_t path = 0u; path < 4u; ++path) {
      if (path == marker)
        continue;
      for (std::uint32_t waste = 0u; waste < 4u; ++waste) {
        if (waste == marker || waste == path)
          continue;
        const BasisPermutation permutation =
            role_basis_permutation(marker, path, waste);
        rearm_permutation[marker][path][waste] = permutation;
        rearm_lock_word[marker][path][waste] =
            transformed_word(processive_rearm_candidate_word(8u), permutation);
      }
    }
  }

  for (const ReferenceSite& lock : support) {
    for (std::uint32_t marker = 0u; marker < 4u; ++marker) {
      for (std::uint32_t path = 0u; path < 4u; ++path) {
        if (path == marker)
          continue;
        for (std::uint32_t waste = 0u; waste < 4u; ++waste) {
          if (waste == marker || waste == path)
            continue;
          const BasisPermutation& permutation = rearm_permutation[marker][path][waste];
          if (lock.word != rearm_lock_word[marker][path][waste])
            continue;
          const ProcessiveRearmOffset lock_offset = processive_rearm_offset(8u);
          Owner owner;
          owner.permutation = permutation;
          owner.center =
              lock.coordinate -
              transformed_coordinate({lock_offset.marker, lock_offset.path, lock_offset.waste},
                                     permutation);
          for (std::uint32_t index = 0u; index < kProcessiveRearmSiteCount; ++index) {
            const ProcessiveRearmOffset offset = processive_rearm_offset(index);
            owner.footprint[index] =
                owner.center +
                transformed_coordinate({offset.marker, offset.path, offset.waste}, permutation);
          }
          bool locks_match = true;
          for (std::uint32_t index = 6u; index < 8u; ++index) {
            locks_match = locks_match &&
                          before.read(owner.footprint[index]) ==
                              transformed_word(processive_rearm_candidate_word(index), permutation);
          }
          if (locks_match)
            owners.push_back(owner);
        }
      }
    }
  }

  for (const Owner& owner : owners) {
    Match match;
    match.center = owner.center;
    bool candidate_match = true;
    bool rearmed_match = true;
    for (std::uint32_t index = 0u; index < kProcessiveRearmSiteCount; ++index) {
      const ProcessiveRearmOffset offset = processive_rearm_offset(index);
      PatternSite& site = match.sites[index];
      site.relative =
          transformed_coordinate({offset.marker, offset.path, offset.waste}, owner.permutation);
      site.candidate = transformed_word(processive_rearm_candidate_word(index), owner.permutation);
      site.rearmed = transformed_word(processive_rearm_rearmed_word(index), owner.permutation);
      const SiteWord actual = before.read(match.center + site.relative);
      candidate_match = candidate_match && actual == site.candidate;
      rearmed_match = rearmed_match && actual == site.rearmed;
    }
    if (!candidate_match && !rearmed_match)
      continue;
    match.candidate_to_rearmed = candidate_match;
    matches.push_back(match);
  }

  ReferenceLattice after = before;
  for (std::size_t index = 0u; index < matches.size(); ++index) {
    bool collided = false;
    for (const Owner& owner : owners) {
      if (owner.center == matches[index].center)
        continue;
      for (const PatternSite& site : matches[index].sites) {
        for (const Z3Coordinate& occupied : owner.footprint) {
          collided = collided || matches[index].center + site.relative == occupied;
        }
      }
    }
    if (collided)
      continue;
    for (const PatternSite& site : matches[index].sites) {
      after.write(matches[index].center + site.relative,
                  matches[index].candidate_to_rearmed ? site.rearmed : site.candidate);
    }
  }
  lattice = std::move(after);
}

void apply_k_processive_release(ReferenceLattice& lattice, bool inverse) {
  struct PatternSite {
    Z3Coordinate relative{};
    SiteWord staged = kQ;
    SiteWord released = kQ;
  };
  struct Match {
    std::array<PatternSite, kProcessiveReleaseSiteCount> sites{};
    Z3Coordinate center{};
    BasisPermutation permutation{};
    std::array<Z3Coordinate, kProcessiveReleaseClaimDigitCount>
        claim_relatives{};
    std::array<Z3Coordinate, 2u> role_targets{};
    std::array<SiteWord, 2u> role_products{};
    Z3Coordinate successor_center{};
    Z3Coordinate successor_ingress{};
    std::uint32_t role_write_count = 0u;
    bool carries = false;
    bool staged_to_released = true;
  };
  struct Owner {
    std::array<Z3Coordinate, kProcessiveReleaseFootprintCount> footprint{};
    Z3Coordinate center{};
    BasisPermutation permutation{};
  };

  const ReferenceLattice before = lattice;
  const std::vector<ReferenceSite> centers = before.support();
  std::vector<Match> matches;
  std::vector<Owner> owners;

  for (const ReferenceSite& candidate : centers) {
    for (std::uint32_t marker = 0u; marker < 4u; ++marker) {
      for (std::uint32_t path = 0u; path < 4u; ++path) {
        if (path == marker)
          continue;
        for (std::uint32_t waste = 0u; waste < 4u; ++waste) {
          if (waste == marker || waste == path)
            continue;
          const BasisPermutation permutation =
              role_basis_permutation(marker, path, waste);
          if (candidate.word !=
              transformed_word(processive_release_staged_word(0u, 10u), permutation))
            continue;
          const ProcessiveReleaseOffset first_offset = processive_release_offset(10u);
          const Z3Coordinate first_relative = transformed_coordinate(
              {first_offset.marker, first_offset.path, first_offset.waste}, permutation);
          Owner owner;
          owner.center = candidate.coordinate - first_relative;
          owner.permutation = permutation;
          for (std::uint32_t index = 0u; index < kProcessiveReleaseSiteCount; ++index) {
            const ProcessiveReleaseOffset offset = processive_release_offset(index);
            owner.footprint[index] =
                owner.center +
                transformed_coordinate({offset.marker, offset.path, offset.waste}, permutation);
          }
          for (std::uint32_t digit = 0u;
               digit < kProcessiveReleaseClaimDigitCount; ++digit) {
            const ProcessiveReleaseOffset offset =
                processive_release_claim_offset(digit);
            owner.footprint[kProcessiveReleaseSiteCount + digit] =
                owner.center + transformed_coordinate(
                                   {offset.marker, offset.path, offset.waste},
                                   permutation);
          }
          if (before.read(owner.footprint[11u]) !=
              transformed_word(processive_release_staged_word(0u, 11u), permutation))
            continue;
          owners.push_back(std::move(owner));
        }
      }
    }
  }

  // The pattern geometry and its staged/released words depend only on
  // (permutation, action, index) -- never on the candidate.  Only `center` did.
  // Building the rows once per call instead of once per candidate is a pure
  // hoist of loop-invariant work: it removes a candidate-count-fold repetition
  // of transformed_coordinate (three cpp_int constructions each) and
  // transformed_word.  The rows are built in the SAME nested order the candidate
  // loop used to walk them -- marker, path, waste, then action -- so iteration
  // order, and therefore which row wins the first-match break below, is
  // unchanged.
  struct PatternRow {
    std::array<PatternSite, kProcessiveReleaseSiteCount> sites{};
    BasisPermutation permutation{};
    std::uint32_t action = 0u;
    std::array<Z3Coordinate, kProcessiveReleaseClaimDigitCount>
        claim_relatives{};
    Z3Coordinate successor_ingress_relative{};
  };
  // Built lazily, on the first candidate that survives the differentiation
  // filter below.  Streaming carrier holes dominate sparse support at scale, so
  // a superstep in which NO candidate qualifies is common; building the table
  // unconditionally would make those supersteps slower than before the hoist.
  std::vector<PatternRow> rows;
  const auto build_rows = [&rows]() {
    rows.reserve(24u * kProcessiveReleaseActionCount);
    for (std::uint32_t marker = 0u; marker < 4u; ++marker) {
      for (std::uint32_t path = 0u; path < 4u; ++path) {
        if (path == marker)
          continue;
        for (std::uint32_t waste = 0u; waste < 4u; ++waste) {
          if (waste == marker || waste == path)
            continue;
          const BasisPermutation permutation =
              role_basis_permutation(marker, path, waste);

          for (std::uint32_t action = 0u; action < 4u; ++action) {
            PatternRow row;
            row.permutation = permutation;
            row.action = action;
            for (std::uint32_t index = 0u; index < row.sites.size(); ++index) {
              const ProcessiveReleaseOffset offset = processive_release_offset(index);
              row.sites[index].relative = transformed_coordinate(
                  {offset.marker, offset.path, offset.waste}, permutation);
              row.sites[index].staged = transformed_word(
                  processive_release_staged_word(action, index), permutation);
              row.sites[index].released = transformed_word(
                  processive_release_released_word(action, index), permutation);
            }
            for (std::uint32_t digit = 0u;
                 digit < kProcessiveReleaseClaimDigitCount; ++digit) {
              const ProcessiveReleaseOffset offset =
                  processive_release_claim_offset(digit);
              row.claim_relatives[digit] = transformed_coordinate(
                  {offset.marker, offset.path, offset.waste}, permutation);
            }
            const ProcessiveReleaseOffset successor =
                processive_successor_claim_offset(
                    kProcessiveRoleIngressDigit);
            row.successor_ingress_relative = transformed_coordinate(
                {successor.marker, successor.path, successor.waste},
                permutation);
            rows.push_back(std::move(row));
          }
        }
      }
    }
  };

  for (const ReferenceSite& candidate : centers) {
    // Streaming carrier holes dominate sparse support at scale. Only a
    // differentiated body/token can be the centre of this factor.
    if ((candidate.word & ~kCarrierMask) == 0u)
      continue;
    if (rows.empty())
      build_rows();
    for (const PatternRow& row : rows) {
      bool staged_match = true;
      bool released_match = true;
      for (const PatternSite& site : row.sites) {
        const Z3Coordinate at = candidate.coordinate + site.relative;
        const SiteWord actual = before.read(at);
        staged_match = staged_match && actual == site.staged;
        released_match = released_match && actual == site.released;
        // This body computes only these two booleans from a const read and has
        // no side effects, so once both are false the remaining sites cannot
        // change the decision immediately below.
        if (!staged_match && !released_match)
          break;
      }
      if (!staged_match && !released_match)
        continue;

      std::array<SiteWord, kProcessiveReleaseClaimDigitCount> actual_claims{};
      bool bare = true;
      bool resting = true;
      for (std::uint32_t digit = 0u;
           digit < kProcessiveReleaseClaimDigitCount; ++digit) {
        actual_claims[digit] =
            before.read(candidate.coordinate + row.claim_relatives[digit]);
        bare = bare && actual_claims[digit] == transformed_word(
            processive_release_claim_word(ProcessiveReleaseClaim::bare),
            row.permutation);
        resting = resting && actual_claims[digit] == transformed_word(
            processive_release_claim_word(ProcessiveReleaseClaim::empty),
            row.permutation);
      }
      Match match;
      match.center = candidate.coordinate;
      match.permutation = row.permutation;
      match.claim_relatives = row.claim_relatives;
      match.sites = row.sites;
      if (bare || resting) {
        match.staged_to_released = staged_match;
      } else {
        const SiteWord empty = transformed_word(
            processive_release_claim_word(ProcessiveReleaseClaim::empty),
            row.permutation);
        const SiteWord role_a = transformed_word(
            processive_release_claim_word(ProcessiveReleaseClaim::adult_a),
            row.permutation);
        const SiteWord role_b = transformed_word(
            processive_release_claim_word(ProcessiveReleaseClaim::adult_b),
            row.permutation);
        const auto role = [role_a, role_b](SiteWord word) {
          return word == role_a ? 0 : (word == role_b ? 1 : -1);
        };
        bool guards_empty = true;
        for (std::uint32_t digit = kProcessiveRoleGuardFirst;
             digit < kProcessiveReleaseClaimDigitCount; ++digit)
          guards_empty = guards_empty && actual_claims[digit] == empty;
        if (!guards_empty)
          continue;

        match.carries = processive_action_carries(row.action);
        match.staged_to_released = staged_match;
        const Z3Coordinate ingress =
            candidate.coordinate +
            row.claim_relatives[kProcessiveRoleIngressDigit];
        const Z3Coordinate landed =
            candidate.coordinate +
            row.claim_relatives[kProcessiveRoleLandedDigit];
        if (!inverse) {
          const int incoming_role =
              role(actual_claims[kProcessiveRoleIngressDigit]);
          if (incoming_role < 0)
            continue;
          if (match.carries) {
            if (role(actual_claims[kProcessiveRoleLandedDigit]) < 0)
              continue;
            match.successor_center =
                candidate.coordinate + transformed_coordinate(
                    {0, 6, 0}, row.permutation);
            match.successor_ingress =
                candidate.coordinate + row.successor_ingress_relative;
            std::uint32_t successor_count = 0u;
            for (const Owner& owner : owners)
              if (owner.center == match.successor_center &&
                  owner.permutation == row.permutation)
                ++successor_count;
            if (successor_count != 1u ||
                before.read(match.successor_ingress) != empty)
              continue;
            match.role_targets[0u] = ingress;
            match.role_products[0u] = empty;
            match.role_targets[1u] = match.successor_ingress;
            match.role_products[1u] =
                incoming_role == 0 ? role_a : role_b;
          } else {
            if (actual_claims[kProcessiveRoleLandedDigit] != empty)
              continue;
            match.role_targets[0u] = ingress;
            match.role_products[0u] = empty;
            match.role_targets[1u] = landed;
            match.role_products[1u] =
                incoming_role == 0 ? role_a : role_b;
          }
        } else {
          if (actual_claims[kProcessiveRoleIngressDigit] != empty)
            continue;
          SiteWord restored_role = kQ;
          if (match.carries) {
            if (role(actual_claims[kProcessiveRoleLandedDigit]) < 0)
              continue;
            match.successor_center =
                candidate.coordinate + transformed_coordinate(
                    {0, 6, 0}, row.permutation);
            match.successor_ingress =
                candidate.coordinate + row.successor_ingress_relative;
            std::uint32_t successor_count = 0u;
            for (const Owner& owner : owners)
              if (owner.center == match.successor_center &&
                  owner.permutation == row.permutation)
                ++successor_count;
            restored_role = before.read(match.successor_ingress);
            if (successor_count != 1u || role(restored_role) < 0)
              continue;
            match.role_targets[1u] = match.successor_ingress;
          } else {
            restored_role = actual_claims[kProcessiveRoleLandedDigit];
            if (role(restored_role) < 0)
              continue;
            match.role_targets[1u] = landed;
          }
          match.role_targets[0u] = ingress;
          match.role_products[0u] = restored_role;
          match.role_products[1u] = empty;
        }
        match.role_write_count = 2u;
      }
      matches.push_back(std::move(match));
      break;  // first match wins, exactly as the original goto next_center did
    }
  }

  ReferenceLattice after = before;
  std::vector<bool> collided(matches.size(), false);
  for (std::size_t source = 0u; source < matches.size(); ++source) {
    if (!matches[source].carries ||
        matches[source].role_write_count == 0u)
      continue;
    for (std::size_t successor = 0u; successor < matches.size(); ++successor)
      if (source != successor &&
          matches[successor].center == matches[source].successor_center &&
          matches[successor].role_write_count == 0u)
        collided[successor] = true;
  }
  for (std::size_t index = 0u; index < matches.size(); ++index) {
    for (const Owner& owner : owners) {
      if (owner.center == matches[index].center)
        continue;
      bool overlap = false;
      for (const PatternSite& first : matches[index].sites) {
        for (const Z3Coordinate& second : owner.footprint) {
          overlap = overlap || matches[index].center + first.relative == second;
        }
      }
      for (const Z3Coordinate& claim_relative :
           matches[index].claim_relatives)
        for (const Z3Coordinate& second : owner.footprint)
          overlap = overlap ||
                    matches[index].center + claim_relative == second;
      if (matches[index].carries) {
        for (const Z3Coordinate& second : owner.footprint) {
          if (matches[index].successor_ingress != second)
            continue;
          const bool intentional_successor =
              owner.center == matches[index].successor_center &&
              owner.permutation == matches[index].permutation;
          overlap = overlap || !intentional_successor;
        }
      }
      if (overlap) {
        collided[index] = true;
        break;
      }
    }
  }
  for (std::size_t index = 0u; index < matches.size(); ++index) {
    if (collided[index])
      continue;
    const Match& match = matches[index];
    for (const PatternSite& site : match.sites) {
      const SiteWord word = match.staged_to_released ? site.released : site.staged;
      after.write(match.center + site.relative, word);
    }
    for (std::uint32_t role_write = 0u;
         role_write < match.role_write_count; ++role_write)
      after.write(match.role_targets[role_write],
                  match.role_products[role_write]);
  }
  lattice = std::move(after);
}

void apply_k_carrier_corner(ReferenceLattice& lattice, bool inverse) {
  struct PatternSite {
    Z3Coordinate relative{};
    SiteWord staged = kQ;
    SiteWord released = kQ;
  };
  struct Match {
    std::array<PatternSite, kCarrierCornerSiteCount> sites{};
    Z3Coordinate center{};
    std::uint32_t incoming = 0u;
    std::uint32_t outgoing = 0u;
  };
  struct Owner {
    std::array<Z3Coordinate, kCarrierCornerSiteCount> footprint{};
    Z3Coordinate center{};
    std::uint32_t incoming = 0u;
    std::uint32_t outgoing = 0u;
  };

  const auto relative = [](std::uint32_t incoming, std::uint32_t outgoing, std::uint32_t index) {
    const CarrierCornerOffset value = carrier_corner_offset(incoming, outgoing, index);
    return Z3Coordinate{value.x, value.y, value.z};
  };

  const ReferenceLattice before = lattice;
  const std::vector<ReferenceSite> support = before.support();
  std::vector<Owner> owners;
  std::vector<Match> matches;

  // The centre word a candidate must equal depends only on (incoming, outgoing),
  // never on the candidate, so it is computed once per call rather than once per
  // candidate.  Index 1 and staged=false are the fixed arguments the lock scan
  // below uses.
  std::array<std::array<SiteWord, 8u>, 8u> corner_centre_word{};
  for (std::uint32_t incoming = 0u; incoming < 8u; ++incoming)
    for (std::uint32_t outgoing = 0u; outgoing < 8u; ++outgoing)
      if (incoming != outgoing)
        corner_centre_word[incoming][outgoing] =
            carrier_corner_word(incoming, outgoing, 1u, false);

  // Locks are persistent owner matter even while no vacancy is at the
  // centre. Discover them independently of the transient endpoint.
  for (const ReferenceSite& candidate : support) {
    for (std::uint32_t incoming = 0u; incoming < 8u; ++incoming) {
      for (std::uint32_t outgoing = 0u; outgoing < 8u; ++outgoing) {
        if (incoming == outgoing)
          continue;
        if (candidate.word != corner_centre_word[incoming][outgoing])
          continue;
        Owner owner;
        owner.center = candidate.coordinate - relative(incoming, outgoing, 1u);
        owner.incoming = incoming;
        owner.outgoing = outgoing;
        for (std::uint32_t index = 0u; index < kCarrierCornerSiteCount; ++index)
          owner.footprint[index] = owner.center + relative(incoming, outgoing, index);
        if (before.read(owner.footprint[2u]) != carrier_corner_word(incoming, outgoing, 2u, false))
          continue;
        owners.push_back(owner);
      }
    }
  }

  // Site geometry and staged/released words depend only on
  // (incoming, outgoing, index).  Built lazily on the first candidate surviving
  // the carrier-only filter, so a superstep where none survives costs nothing.
  std::array<std::array<std::array<PatternSite, kCarrierCornerSiteCount>, 8u>, 8u>
      corner_sites{};
  bool corner_sites_built = false;
  const auto build_corner_sites = [&corner_sites, &relative]() {
    for (std::uint32_t incoming = 0u; incoming < 8u; ++incoming) {
      for (std::uint32_t outgoing = 0u; outgoing < 8u; ++outgoing) {
        if (incoming == outgoing)
          continue;
        for (std::uint32_t index = 0u; index < kCarrierCornerSiteCount; ++index) {
          PatternSite& site = corner_sites[incoming][outgoing][index];
          site.relative = relative(incoming, outgoing, index);
          site.staged = carrier_corner_word(incoming, outgoing, index, false);
          site.released = carrier_corner_word(incoming, outgoing, index, true);
        }
      }
    }
  };

  for (const ReferenceSite& candidate : support) {
    if ((candidate.word & ~kCarrierMask) != 0u)
      continue;
    if (!corner_sites_built) {
      build_corner_sites();
      corner_sites_built = true;
    }
    Match unique_match;
    std::uint32_t present_owners = 0u;
    std::uint32_t compatible_owners = 0u;
    for (const Owner& owner : owners) {
      if (owner.center != candidate.coordinate) continue;
      ++present_owners;
      if (!carrier_corner_center_matches(
              candidate.word, owner.incoming, owner.outgoing))
        continue;
      unique_match.center = candidate.coordinate;
      unique_match.incoming = owner.incoming;
      unique_match.outgoing = owner.outgoing;
      for (std::uint32_t index = 0u;
           index < kCarrierCornerSiteCount; ++index)
        unique_match.sites[index] =
            corner_sites[owner.incoming][owner.outgoing][index];
      ++compatible_owners;
    }
    // Persistent locks, not the transient centre byte, define ownership.
    // A second co-centred owner must make both phases abstain even when only
    // one of those owners happens to match before the proposed transpose.
    if (carrier_corner_unique_owner_matches(
            present_owners, compatible_owners))
      matches.push_back(unique_match);
  }

  ReferenceLattice after = before;
  for (const Match& match : matches) {
    bool collided = false;
    for (const Owner& owner : owners) {
      if (owner.center == match.center)
        continue;
      for (const PatternSite& first : match.sites) {
        for (const Z3Coordinate& second : owner.footprint)
          collided = collided || match.center + first.relative == second;
      }
    }
    if (collided)
      continue;
    const SiteWord center = before.read(match.center);
    after.write(
        match.center,
        carrier_corner_transpose(
            center, match.incoming, match.outgoing));
  }
  lattice = std::move(after);
}

void apply_s_p(ReferenceLattice& lattice) {
    stream(lattice, false);
}

void apply_s_p_inverse(ReferenceLattice& lattice) {
    stream(lattice, true);
}

void apply_k_prediction_residual_route_toggle(ReferenceLattice& lattice) {
    const ReferenceLattice before = lattice;
    struct Match {
        Z3Coordinate center{};
        prediction_residual_route_toggle_detail::Action action{};
        Z3Coordinate sites[prediction_residual_route_toggle_detail::kPhysicalSites]{};
    };
    std::vector<Match> matches;
    for (const ReferenceSite& site : before.support()) {
        PredictionResidualNeighborhood neighborhood{};
        neighborhood.center = before.read(site.coordinate);
        for (std::uint32_t basis = 0u; basis < 4u; ++basis) {
            neighborhood.positive[basis] = before.read(
                site.coordinate + positive_offset(basis));
            neighborhood.negative[basis] = before.read(
                site.coordinate + negative_offset(basis));
        }
        const auto result = evaluate_prediction_residual_route_toggle(neighborhood);
        if (result.receipt.selected_candidate == 0xffffffffu ||
            result.receipt.selected_kind > 3u)
            continue;
        const auto candidate = prediction_residual_route_toggle_detail::candidate_at(
            prediction_residual_route_toggle_detail::selected_permutation(
                result.receipt.selected_candidate),
            prediction_residual_route_toggle_detail::selected_negative_probe(
                result.receipt.selected_candidate));
        Match match{};
        match.center = site.coordinate;
        match.action = prediction_residual_route_toggle_detail::make_selected_action(
            candidate, static_cast<prediction_residual_route_toggle_detail::ActionKind>(
                           result.receipt.selected_kind));
        match.sites[0] = site.coordinate;
        for (std::uint32_t basis = 0u; basis < 4u; ++basis) {
            match.sites[1u + basis] = site.coordinate + positive_offset(basis);
            match.sites[5u + basis] = site.coordinate + negative_offset(basis);
        }
        matches.push_back(match);
    }
    ReferenceLattice after = before;
    for (std::size_t i = 0u; i < matches.size(); ++i) {
        bool collision = false;
        for (std::size_t j = 0u; j < matches.size(); ++j) {
            if (i == j) continue;
            for (std::uint32_t left = 0u; left < 9u; ++left) {
                for (std::uint32_t right = 0u; right < 9u; ++right) {
                    if (matches[i].sites[left] != matches[j].sites[right]) continue;
                    const auto& a = matches[i].action;
                    const auto& b = matches[j].action;
                    collision = collision ||
                        (a.write_masks[left] & (b.read_masks[right] | b.write_masks[right])) != 0u ||
                        (b.write_masks[right] & (a.read_masks[left] | a.write_masks[left])) != 0u;
                }
            }
        }
        if (collision) continue;
        const auto& action = matches[i].action;
        const Z3Coordinate& target = matches[i].sites[action.site];
        SiteWord word = before.read(target);
        controlled_transpose(word, action.first_bit, word, action.second_bit, true);
        after.write(target, word);
    }
    lattice = std::move(after);
}

void apply_factor(ReferenceLattice& lattice, LawFactor factor, bool inverse) {
    switch (factor) {
        case LawFactor::site:
            inverse ? apply_k_site_inverse(lattice) : apply_k_site(lattice);
            return;
        case LawFactor::edge:
            inverse ? apply_k_edge_inverse(lattice) : apply_k_edge(lattice);
      return;
    case LawFactor::carrier_pair_splitter:
      apply_k_carrier_pair_splitter(lattice, inverse);
      return;
    case LawFactor::processive_rearm:
      apply_k_processive_rearm(lattice, inverse);
      return;
    case LawFactor::processive_release:
      apply_k_processive_release(lattice, inverse);
      return;
    case LawFactor::carrier_corner:
      apply_k_carrier_corner(lattice, inverse);
      return;
    case LawFactor::eligibility_residual_junction:
      apply_k_eligibility_residual_junction(lattice, inverse);
      return;
    case LawFactor::developmental_append:
      apply_k_developmental_append(lattice, inverse);
      return;
    case LawFactor::developmental_learned_receptor:
      apply_k_developmental_learned_receptor(lattice, inverse);
      return;
    case LawFactor::developmental_credit_service:
      apply_k_developmental_credit_service(lattice, inverse);
      return;
    case LawFactor::prediction_residual_route_toggle:
      apply_k_prediction_residual_route_toggle(lattice);
      return;
        case LawFactor::stream:
            inverse ? apply_s_p_inverse(lattice) : apply_s_p(lattice);
            return;
    }
}

void apply_superstep(ReferenceLattice& lattice) {
    for (std::uint32_t index = 0u; index < kForwardFactorCount; ++index) {
        apply_factor(lattice, forward_factor(index), false);
    }
}

void apply_superstep_inverse(ReferenceLattice& lattice) {
    for (std::uint32_t index = kForwardFactorCount; index > 0u; --index) {
        apply_factor(lattice, forward_factor(index - 1u), true);
    }
}

}  // namespace substrate::bcc32
