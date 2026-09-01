#include "bcc32_law_identity.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>

#include "bcc32_developmental_append.hpp"
#include "bcc32_developmental_credit_service.hpp"
#include "bcc32_geometry.cuh"
#include "bcc32_eligibility_residual_junction.hpp"
#include "bcc32_law.cuh"
#include "bcc32_law_netlist.cuh"
#include "bcc32_prediction_residual_route_toggle.cuh"
#include "bcc32_types.cuh"

namespace substrate::bcc32 {
namespace {

class CanonicalHasher {
 public:
  void u32(std::uint32_t value) {
    std::array<std::byte, 4u> encoded{};
    for (std::uint32_t byte = 0u; byte < encoded.size(); ++byte) {
      encoded[byte] = static_cast<std::byte>(value >> (byte * 8u));
    }
    append(encoded);
  }

  void u64(std::uint64_t value) {
    std::array<std::byte, 8u> encoded{};
    for (std::uint32_t byte = 0u; byte < encoded.size(); ++byte) {
      encoded[byte] = static_cast<std::byte>(value >> (byte * 8u));
    }
    append(encoded);
  }

  void i32(std::int32_t value) { u32(static_cast<std::uint32_t>(value)); }

  void word(SiteWord value) { u32(value); }

  [[nodiscard]] ContentAddress finish() { return {hasher_.finish(), bytes_}; }

 private:
  template <std::size_t N>
  void append(const std::array<std::byte, N>& encoded) {
    if (bytes_ > std::numeric_limits<std::uint64_t>::max() - encoded.size()) {
      throw std::overflow_error("BCC32 canonical fingerprint is too large");
    }
    hasher_.update(std::span<const std::byte>(encoded));
    bytes_ += encoded.size();
  }

  ContentHasher hasher_;
  std::uint64_t bytes_ = 0u;
};

template <typename Enum>
[[nodiscard]] constexpr std::uint32_t enum_value(Enum value) {
  return static_cast<std::uint32_t>(value);
}

void append_site_gate(CanonicalHasher* hasher, const SiteGate& gate) {
  hasher->u32(enum_value(gate.opcode));
  hasher->u32(gate.target_shift);
  hasher->u32(gate.control_shift);
  std::uint32_t inputs = 256u;
  if (gate.opcode == SiteOpcode::complement_pair_tetrad)
    inputs = 16u;
  else if (gate.opcode == SiteOpcode::differentiated_three_factor_capture)
    inputs = 2097152u;
  else if (gate.opcode == SiteOpcode::differentiated_signed_synapse)
    inputs = 2048u;
  else if (gate.opcode == SiteOpcode::differentiated_processive_stage)
    inputs = 96u;
  else if (gate.opcode == SiteOpcode::differentiated_terminal_maturation)
    // All 8 patterns over C_p/R_p/E_m for all 12 ordered marker/path pairs,
    // plus every one-bit contamination around both frozen endpoints. This
    // binds abstention semantics as well as the 24 moving archetypes.
    inputs = 160u;
  hasher->u32(inputs);
  for (std::uint32_t input = 0u; input < inputs; ++input) {
    SiteWord word = static_cast<SiteWord>((input & 0x0fu) << gate.target_shift);
    if (gate.opcode == SiteOpcode::pair_rotor) {
      word |= static_cast<SiteWord>(((input >> 4u) & 0x0fu) << gate.control_shift);
    } else if (gate.opcode == SiteOpcode::differentiated_signed_synapse) {
      word = static_cast<SiteWord>(input & 0xffu);
      if ((input & (1u << 8u)) != 0u)
        word |= face_bit(4u);
      if ((input & (1u << 9u)) != 0u)
        word |= energy_bit(0u);
      if ((input & (1u << 10u)) != 0u)
        word |= owned_bond_bit(0u);
    } else if (gate.opcode == SiteOpcode::differentiated_three_factor_capture) {
      constexpr std::array<SiteWord, 21u> kRoles{{
          energy_bit(0u),
          channel_bit(kConformationShift, 0u),
          channel_bit(kReactiveShift, 0u),
          owned_bond_bit(0u),
          carrier_bit(0u),
          owned_bond_bit(1u),
          owned_bond_bit(2u),
          owned_bond_bit(3u),
          face_bit(0u),
          face_bit(1u),
          face_bit(2u),
          face_bit(3u),
          face_bit(4u),
          face_bit(5u),
          face_bit(6u),
          face_bit(7u),
          energy_bit(1u),
          channel_bit(kReactiveShift, 1u),
          channel_bit(kConformationShift, 1u),
          carrier_bit(1u),
          carrier_bit(5u),
      }};
      word = 0u;
      for (std::uint32_t bit = 0u; bit < kRoles.size(); ++bit)
        if ((input & (1u << bit)) != 0u)
          word |= kRoles[bit];
    } else if (gate.opcode == SiteOpcode::differentiated_processive_stage) {
      const std::uint32_t endpoint = input & 1u;
      const std::uint32_t action = (input >> 1u) & 3u;
      const std::uint32_t path_rank = (input >> 3u) % 3u;
      const std::uint32_t marker = input / 24u;
      std::uint32_t path = path_rank;
      if (path >= marker)
        ++path;
      const SiteWord rest = kCarrierMask ^ carrier_bit(marker) ^ carrier_bit(marker + 4u);
      const SiteWord zero_marker = face_bit(marker + 4u) | energy_bit(marker);
      const SiteWord one_marker = face_bit(marker + 4u) | owned_bond_bit(marker);
      const SiteWord raw_positive = rest ^ carrier_bit(path);
      const SiteWord raw_negative = rest ^ carrier_bit(path + 4u);
      const std::array<SiteWord, 4u> bodies{{
          zero_marker | (raw_positive ^ SiteWord{0x0fu}),
          one_marker | (raw_positive ^ SiteWord{0x0fu}),
          one_marker | (raw_negative ^ SiteWord{0xf0u}),
          zero_marker | (raw_negative ^ SiteWord{0xf0u}),
      }};
      const std::array<SiteWord, 4u> tokens{{
          owned_bond_bit(path) | rest,
          channel_bit(kReactiveShift, path) | rest,
          energy_bit(path) | rest,
          channel_bit(kConformationShift, path) | rest,
      }};
      word = endpoint == 0u ? bodies[action] : tokens[action];
    } else if (gate.opcode == SiteOpcode::differentiated_terminal_maturation) {
      if (input < 96u) {
        const std::uint32_t ordered_pair = input / 8u;
        const std::uint32_t marker = ordered_pair / 3u;
        const std::uint32_t path_rank = ordered_pair % 3u;
        std::uint32_t path = path_rank;
        if (path >= marker)
          ++path;
        const std::uint32_t pattern = input & 0x07u;
        const std::array<SiteWord, 3u> targets{{
            channel_bit(kConformationShift, path),
            channel_bit(kReactiveShift, path), energy_bit(marker)}};
        word = kQ | face_bit(marker);
        for (std::uint32_t bit = 0u; bit < targets.size(); ++bit)
          if ((pattern & (1u << bit)) != 0u)
            word |= targets[bit];
      } else {
        const std::uint32_t contamination = input - 96u;
        const SiteWord endpoint =
            contamination < 32u
                ? kQ | face_bit(1u) |
                      channel_bit(kConformationShift, 0u) |
                      channel_bit(kReactiveShift, 0u)
                : kQ | face_bit(1u) |
                      channel_bit(kReactiveShift, 0u) | energy_bit(1u);
        word = endpoint ^ (SiteWord{1u} << (contamination & 31u));
      }
    }
    apply_site_gate(word, gate);
    hasher->word(word);
  }
}

constexpr std::array<EdgeRole, 8u> kEdgeRoles = {
    EdgeRole::positive_carrier, EdgeRole::negative_carrier, EdgeRole::positive_face,
    EdgeRole::negative_face,    EdgeRole::owned_bond,       EdgeRole::conformation,
    EdgeRole::reactive,         EdgeRole::energy,
};

void append_edge_block(CanonicalHasher* hasher, SiteWord source, SiteWord destination,
                       std::uint32_t basis) {
  for (const EdgeRole role : kEdgeRoles) {
    hasher->u32(role_is_set(source, destination, role, basis) ? 1u : 0u);
  }
}

void append_edge_gate(CanonicalHasher* hasher, const EdgeGate& gate) {
  hasher->u32(enum_value(gate.opcode));
  hasher->u32(enum_value(gate.first));
  hasher->u32(enum_value(gate.second));
  hasher->u32(enum_value(gate.third));
  hasher->u32(enum_value(gate.fourth));
  hasher->u32(enum_value(gate.control));
  hasher->u32(enum_value(gate.control_first));
  hasher->u32(enum_value(gate.control_second));
  hasher->u32(256u);
  for (std::uint32_t input = 0u; input < 256u; ++input) {
    SiteWord source = 0u;
    SiteWord destination = 0u;
    for (std::uint32_t bit = 0u; bit < kEdgeRoles.size(); ++bit) {
      if ((input & (1u << bit)) != 0u) {
        toggle_role(source, destination, kEdgeRoles[bit], 0u);
      }
    }
    apply_edge_gate(source, destination, gate, 0u);
    append_edge_block(hasher, source, destination, 0u);
  }
}

SiteWord junction_control_truth_word(std::uint32_t basis,
                                     std::uint32_t control_class) {
  const std::uint32_t other = (basis + 1u) & 3u;
  switch (control_class) {
    case 0u: return kQ;
    case 1u: return kQ | energy_bit(basis);
    case 2u: return kQ | energy_bit(basis) | face_bit(other);
    case 3u: return kQ ^ carrier_bit(basis);
    case 4u: return (kQ ^ carrier_bit(basis)) | face_bit(other);
    case 5u: return kQ ^ carrier_bit(other);
    case 6u: return kQ | energy_bit(other);
    case 7u: return kQ | owned_bond_bit(basis);
    default: throw std::logic_error("invalid NL08 control truth-table class");
  }
}

void append_eligibility_residual_junction_semantics(CanonicalHasher* hasher) {
  constexpr std::uint32_t kRolePermutations = 24u;
  constexpr std::uint32_t kCenterClasses = 8u;
  constexpr std::uint32_t kControlClasses = 8u;
  hasher->u32(0x4a523039u);  // JR09: resident owner required in production.
  hasher->u32(kRolePermutations);
  hasher->u32(kEligibilityResidualJunctionAuthority.incoming_sign_count);
  hasher->u32(kEligibilityResidualJunctionAuthority.outgoing_sign_count);
  hasher->u32(kEligibilityResidualJunctionAuthority.endogenous_owner_required);
  hasher->u32(kCenterClasses);
  hasher->u32(kControlClasses);

  std::array<std::uint32_t, 4u> permutation{0u, 1u, 2u, 3u};
  do {
    for (std::uint32_t incoming_sign = 0u;
         incoming_sign < kEligibilityResidualJunctionAuthority.incoming_sign_count;
         ++incoming_sign) {
      for (std::uint32_t outgoing_sign = 0u;
           outgoing_sign < kEligibilityResidualJunctionAuthority.outgoing_sign_count;
           ++outgoing_sign) {
        const EligibilityResidualJunctionDescriptor descriptor{
            {0, 0, 0},
            permutation[kEligibilityResidualJunctionAuthority.incoming_role] +
                4u * incoming_sign,
            permutation[kEligibilityResidualJunctionAuthority.outgoing_role] +
                4u * outgoing_sign,
            permutation[kEligibilityResidualJunctionAuthority.control_a_role],
            permutation[kEligibilityResidualJunctionAuthority.control_b_role],
        };
        if (!valid_eligibility_residual_junction_descriptor(descriptor))
          throw std::logic_error("canonical NL08 descriptor is invalid");
        const Z3Coordinate control_a =
            eligibility_residual_control_a_coordinate(descriptor);
        const Z3Coordinate control_b =
            eligibility_residual_control_b_coordinate(descriptor);
        hasher->u32(descriptor.incoming_direction);
        hasher->u32(descriptor.outgoing_direction);
        hasher->u32(descriptor.control_a_basis);
        hasher->u32(descriptor.control_b_basis);
        hasher->i32(static_cast<std::int32_t>(control_a.x));
        hasher->i32(static_cast<std::int32_t>(control_a.y));
        hasher->i32(static_cast<std::int32_t>(control_a.z));
        hasher->i32(static_cast<std::int32_t>(control_b.x));
        hasher->i32(static_cast<std::int32_t>(control_b.y));
        hasher->i32(static_cast<std::int32_t>(control_b.z));

        const SiteWord incoming_bit = carrier_bit(descriptor.incoming_direction);
        const SiteWord outgoing_bit = carrier_bit(descriptor.outgoing_direction);
        const SiteWord target_mask = incoming_bit | outgoing_bit;
        for (std::uint32_t center_class = 0u; center_class < kCenterClasses;
             ++center_class) {
          SiteWord center_word = kQ & ~target_mask;
          if ((center_class & 1u) != 0u)
            center_word |= incoming_bit;
          if ((center_class & 2u) != 0u)
            center_word |= outgoing_bit;
          if ((center_class & 4u) != 0u)
            center_word |= face_bit(0u);
          for (std::uint32_t a_class = 0u; a_class < kControlClasses;
               ++a_class) {
            for (std::uint32_t b_class = 0u; b_class < kControlClasses;
                 ++b_class) {
              const SiteWord a_word = junction_control_truth_word(
                  descriptor.control_a_basis, a_class);
              const SiteWord b_word = junction_control_truth_word(
                  descriptor.control_b_basis, b_class);
              // The identity certifies the factor by RUNNING its shared
              // single-descriptor kernel, not a private restatement of it.
              // The host reference lattice was only a three-word container
              // here, and a strict-device binary must not link it merely to
              // ask the law its name.
              hasher->word(center_word);
              hasher->word(a_word);
              hasher->word(b_word);
              const EligibilityResidualLocalResult local =
                  apply_eligibility_residual_junction_local(
                      descriptor, center_word, a_word, b_word);
              hasher->word(local.fired ? local.center : center_word);
              hasher->word(a_word);
              hasher->word(b_word);
            }
          }
        }
      }
    }
  } while (std::next_permutation(permutation.begin(), permutation.end()));
}

}  // namespace

ContentAddress canonical_law_identity() {
  CanonicalHasher hasher;
  // The identity serializes the executable descriptors and the exhaustive
  // truth table of each primitive descriptor, not sampled complete worlds.
  hasher.u32(0x4c4e4932u);
  hasher.u32(kLawNetlistSchema);
  hasher.u32(kBitsPerSite);
  hasher.word(kQ);
  hasher.word(kCarrierMask);
  hasher.word(kFaceMask);
  hasher.word(kOwnedBondMask);
  hasher.word(kConformationMask);
  hasher.word(kReactiveMask);
  hasher.word(kEnergyMask);
  hasher.u32(kCarrierShift);
  hasher.u32(kFaceShift);
  hasher.u32(kOwnedBondShift);
  hasher.u32(kConformationShift);
  hasher.u32(kReactiveShift);
  hasher.u32(kEnergyShift);

  for (std::uint32_t direction = 0u; direction < 8u; ++direction) {
    const auto value = static_cast<Direction>(direction);
    const Int3 logical = direction_offset(value);
    const Int3 physical = physical_tetrahedral_vector(value);
    hasher.u32(direction);
    hasher.u32(direction_index(opposite_direction(value)));
    hasher.i32(logical.x);
    hasher.i32(logical.y);
    hasher.i32(logical.z);
    hasher.i32(physical.x);
    hasher.i32(physical.y);
    hasher.i32(physical.z);
  }

  hasher.u32(kSiteGateCount);
  for (std::uint32_t index = 0u; index < kSiteGateCount; ++index) {
    append_site_gate(&hasher, site_gate(index));
  }
  hasher.u32(kEdgeGateCount);
  for (std::uint32_t index = 0u; index < kEdgeGateCount; ++index) {
    append_edge_gate(&hasher, edge_gate(index));
  }
  hasher.u32(kForwardFactorCount);
  for (std::uint32_t index = 0u; index < kForwardFactorCount; ++index) {
    hasher.u32(enum_value(forward_factor(index)));
  }
  // PRT2 records exact-footprint deduplication, local unique-action
  // abstention, and symmetric global overlap rejection.
  hasher.u32(0x50525432u);
  hasher.u32(prediction_residual_route_toggle_detail::kPhysicalSites);
  hasher.u32(prediction_residual_route_toggle_detail::kActionCapacity);
  for (std::uint32_t code = 1u; code <= 48u; ++code) {
    const auto candidate = prediction_residual_route_toggle_detail::candidate_at(
        prediction_residual_route_toggle_detail::selected_permutation(code),
        prediction_residual_route_toggle_detail::selected_negative_probe(code));
    hasher.u32(code);
    hasher.word(prediction_residual_route_toggle_detail::h_collar(candidate));
    hasher.word(prediction_residual_route_toggle_detail::a_collar(candidate));
    hasher.word(prediction_residual_route_toggle_detail::positive_gate_collar(candidate));
    hasher.word(prediction_residual_route_toggle_detail::negative_gate_collar(candidate));
    hasher.word(prediction_residual_route_toggle_detail::owner_collar(candidate));
    for (std::uint32_t kind = 0u; kind < 4u; ++kind) {
      const auto action = prediction_residual_route_toggle_detail::make_selected_action(
          candidate, static_cast<prediction_residual_route_toggle_detail::ActionKind>(kind));
      hasher.u32(kind);
      for (std::uint32_t site = 0u; site < 9u; ++site) {
        hasher.word(action.read_masks[site]);
        hasher.word(action.predicate_masks[site]);
        hasher.word(action.write_masks[site]);
      }
    }
  }
  hasher.u32(kEligibilityResidualJunctionAuthority.outgoing_role);
  hasher.u32(kEligibilityResidualJunctionAuthority.control_a_role);
  hasher.u32(kEligibilityResidualJunctionAuthority.incoming_role);
  hasher.u32(kEligibilityResidualJunctionAuthority.control_b_role);
  hasher.i32(kEligibilityResidualJunctionAuthority.outgoing_distance);
  hasher.u32(kEligibilityResidualJunctionAuthority.outgoing_sign_count);
  hasher.u32(kEligibilityResidualJunctionAuthority.incoming_sign_count);
  hasher.u32(kEligibilityResidualJunctionAuthority.target_independent);
  hasher.u32(kEligibilityResidualJunctionAuthority.overlap_abstains);
  hasher.u32(kEligibilityResidualJunctionAuthority.exact_carrier_vacancy_controls);
  hasher.u32(kEligibilityResidualJunctionAuthority.positive_incoming_control_offset);
  append_eligibility_residual_junction_semantics(&hasher);
  hasher.u32(kDevelopmentalAppendSiteCount);
  hasher.u32(kDevelopmentalAppendFirstVirginSite);
  hasher.u32(kDevelopmentalAppendVirginSiteCount);
  hasher.u32(kDevelopmentalAppendParentLockFirst);
  hasher.u32(kDevelopmentalAppendParentLockSecond);
  hasher.u32(kDevelopmentalAppendChildHead);
  hasher.u32(kDevelopmentalAppendChildLockFirst);
  hasher.u32(kDevelopmentalAppendChildLockSecond);
  hasher.u32(kDevelopmentalAppendHandoffSiteCount);
  hasher.u32(kDevelopmentalAppendAgeFirst);
  hasher.u32(kDevelopmentalAppendAgeDigitCount);
  hasher.u32(kDevelopmentalAppendAgeSitesPerDigit);
  hasher.u32(kDevelopmentalAppendJournalFirst);
  hasher.u32(kDevelopmentalAppendJournalDigitCount);
  hasher.u32(kDevelopmentalAppendReceptorLegCount);
  hasher.u32(kDevelopmentalAppendAuthorityDigitCount);
  hasher.u32(kDevelopmentalAppendEventJournalFirst);
  hasher.u32(kDevelopmentalAppendEventJournalCount);
  // Exact finite lifecycle permutation:
  // X->P0, Pn->P(n+1), Pmax->X; inverse traverses the same cycle backward.
  hasher.u32(0x58415030u);
  hasher.u32(0x504e5049u);
  hasher.u32(0x504d4158u);
  hasher.u64(kDevelopmentalAppendMaxAge);
  constexpr std::uint32_t append_product_signature[9] = {
      0u, 11u, 12u, 13u, 17u, 30u, 31u,
      kDevelopmentalAppendJournalFirst,
      kDevelopmentalAppendJournalFirst + 1u};
  for (const std::uint32_t site : append_product_signature)
    hasher.u32(site);
  for (std::uint32_t parent_site = 0u;
       parent_site < kDevelopmentalAppendSiteCount; ++parent_site)
    for (std::uint32_t child_site = 0u;
         child_site < kDevelopmentalAppendSiteCount; ++child_site)
      hasher.u32(developmental_append_handoff_pair(parent_site, child_site));
  for (std::uint32_t index = 0u; index < kDevelopmentalAppendSiteCount; ++index) {
    const DevelopmentalAppendOffset offset = developmental_append_offset(index);
    hasher.i32(offset.marker);
    hasher.i32(offset.path);
    hasher.i32(offset.waste);
    for (std::uint32_t marker = 0u; marker < 4u; ++marker) {
      for (std::uint32_t path = 0u; path < 4u; ++path) {
        if (path == marker) continue;
        for (std::uint32_t waste = 0u; waste < 4u; ++waste) {
          if (waste == marker || waste == path) continue;
          hasher.word(developmental_append_word(false, index, marker, path, waste));
          hasher.word(developmental_append_word(true, index, marker, path, waste));
        }
      }
    }
  }
  for (std::uint32_t digit = 0u;
       digit < kDevelopmentalAppendAgeDigitCount; ++digit) {
    const std::uint32_t site = kDevelopmentalAppendAgeFirst +
                               digit * kDevelopmentalAppendAgeSitesPerDigit +
                               2u;
    for (std::uint32_t encoded = 0u; encoded < 4u; ++encoded) {
      const std::uint64_t age =
          std::uint64_t{encoded} << (2u * digit);
      for (std::uint32_t marker = 0u; marker < 4u; ++marker)
        for (std::uint32_t path = 0u; path < 4u; ++path) {
          if (path == marker) continue;
          for (std::uint32_t waste = 0u; waste < 4u; ++waste) {
            if (waste == marker || waste == path) continue;
            hasher.word(developmental_append_product_word(
                site, marker, path, waste, age));
          }
        }
    }
  }
  // Two teacher-gated authority digits persist learned A/B routes.  Six
  // transient event digits then preserve the inverse provenance of held-out
  // arrivals while both legs converge on the same processive inlet.
  hasher.u32(0x52435032u);
  // Immutable-snapshot composition: receptor predicates read the pre-factor
  // state; on CUDA the receptor kernel therefore precedes append matching.
  // Existing-product age writes commute, while birth-coincident vacancies are
  // intentionally deferred to birth+1.  Authority digits are fixed A/B slots;
  // event policy is contiguous-prefix, forward-first-empty,
  // inverse-last-occupied, and exact-two-port abstention.
  hasher.u32(0x494d4d55u);
  hasher.u32(0x52425031u);
  hasher.u32(0x4a505246u);
  hasher.u32(0x4a4c4946u);
  hasher.u32(0x41424232u);
  hasher.u32(0x54474154u);
  hasher.u32(0x41554232u);
  hasher.u32(0x45564c36u);
  // NL13 learned-receptor predicate/action map.  These tags bind the CPU K,
  // dense CUDA, dynamic-active CUDA, and paged factor-window implementations
  // to the same learned->append ordering and declared macro-closure contract.
  hasher.u32(0x4c523133u);
  hasher.u32(enum_value(LawFactor::developmental_learned_receptor));
  hasher.u32(kSpatialMacroClosureRadius);
  hasher.u32(0x4350554bu);
  hasher.u32(0x43554444u);
  hasher.u32(0x43554143u);
  hasher.u32(0x50414745u);
  hasher.u32(0x41504c52u);
  // Predicate: append training reads a leg-specific teacher vacancy; learned
  // routing reads exact-Q teacher withdrawal, one exact source vacancy, the
  // path carrier at site 10, and only the matching fixed authority digit.
  hasher.u32(0x54525643u);
  hasher.u32(0x54525130u);
  hasher.u32(0x31505643u);
  hasher.u32(0x494e5031u);
  hasher.u32(0x41555448u);
  // Training action transposes the fixed authority face and consumes fresh
  // credit by redirecting its carrier vacancy onto the unused fourth basis.
  // Source and common inlet remain controls, so training cannot leak an
  // operand.  The spent-teacher witness makes duplicate abstention injective.
  hasher.u32(0x41554643u);
  hasher.u32(0x54535054u);
  hasher.u32(0x44555049u);
  // Learned action/inverse: transpose source-basis vacancy to the common path
  // inlet, push first-empty event digit, and pop last-occupied event digit.
  // That positive A/B event is the one-shot claim shared with processive
  // release; the adult head, authority digits, and teacher word are immutable.
  hasher.u32(0x50544649u);
  hasher.u32(0x494e484fu);
  hasher.u32(0x45565055u);
  hasher.u32(0x4556504fu);
  hasher.u32(0x4155494du);
  hasher.u32(0x5445494du);
  hasher.u32(0x53415436u);
  hasher.u32(0x4556434cu);  // EVCL: event journal claim
  hasher.u32(3u);           // source, inlet, journal
  hasher.u32(kDevelopmentalAppendReceptorInletSite);
  hasher.u32(kDevelopmentalAppendReceptorJournalEmpty);
  hasher.u32(kDevelopmentalAppendReceptorJournalA);
  hasher.u32(kDevelopmentalAppendReceptorJournalB);
  const DevelopmentalAppendOffset teacher =
      developmental_append_teacher_offset();
  hasher.i32(teacher.marker);
  hasher.i32(teacher.path);
  hasher.i32(teacher.waste);
  for (std::uint32_t leg = 0u;
       leg < kDevelopmentalAppendReceptorLegCount; ++leg) {
    const DevelopmentalAppendOffset port =
        developmental_append_receptor_port_offset(leg);
    hasher.u32(leg);
    hasher.u32(developmental_append_receptor_authority_site(leg));
    hasher.u32(developmental_append_receptor_authority_state_site(leg));
    hasher.i32(port.marker);
    hasher.i32(port.path);
    hasher.i32(port.waste);
    for (std::uint32_t marker = 0u; marker < 4u; ++marker)
      for (std::uint32_t waste = 0u; waste < 4u; ++waste) {
        hasher.u32(
            developmental_append_receptor_basis(leg, marker, waste));
        hasher.word(developmental_append_teacher_vacancy_word(
            leg, marker, waste));
      }
  }
  // Exact all-S4 fresh/spent teaching-credit map.  The path role is required
  // to identify the free fourth basis used by the conserved spent trajectory.
  for (std::uint32_t marker = 0u; marker < 4u; ++marker)
    for (std::uint32_t path = 0u; path < 4u; ++path) {
      if (path == marker) continue;
      for (std::uint32_t waste = 0u; waste < 4u; ++waste) {
        if (waste == marker || waste == path) continue;
        hasher.u32(marker);
        hasher.u32(path);
        hasher.u32(waste);
        hasher.u32(developmental_append_free_basis(marker, path, waste));
        hasher.word(developmental_append_teacher_vacancy_word(
            0u, marker, waste));
        hasher.word(developmental_append_teacher_vacancy_word(
            1u, marker, waste));
        hasher.word(developmental_append_spent_teacher_vacancy_word(
            0u, marker, path, waste));
        hasher.word(developmental_append_spent_teacher_vacancy_word(
            1u, marker, path, waste));
      }
    }
  // NL12 admission identity.  Accepted teaching and rejected source-only
  // admissions have different conserved postimages.  Rejections allocate the
  // first empty per-leg bank and require that same bank's three spatially
  // distinct P18/P20/P22 witnesses before the last-filled bank can invert.
  hasher.u32(0x41444d33u);  // ADM3: three bank-bound witnesses
  hasher.u32(kDevelopmentalAppendWitnessBankCount);
  hasher.u32(0x46454d50u);  // first-empty forward
  hasher.u32(0x4c46494cu);  // last-filled inverse
  hasher.u32(0x41434350u);  // accepted: teacher/source/clock
  hasher.u32(0x52454a50u);  // rejected: source/teacher/clock + two escrows
  hasher.u32(0x54494e54u);  // TINT: common teacher is intake-only
  hasher.u32(0x53494e54u);  // SINT: physical source ports are intake-only
  for (std::uint32_t leg = 0u;
       leg < kDevelopmentalAppendReceptorLegCount; ++leg) {
    const DevelopmentalAppendOffset accepted_source =
        developmental_append_accepted_source_ingress_offset(leg);
    const DevelopmentalAppendOffset accepted_clock =
        developmental_append_clock_ingress_offset(leg);
    const DevelopmentalAppendOffset accepted_teacher =
        developmental_append_accepted_teacher_ingress_offset(leg);
    hasher.u32(leg);
    hasher.i32(accepted_source.marker);
    hasher.i32(accepted_source.path);
    hasher.i32(accepted_source.waste);
    hasher.i32(accepted_clock.marker);
    hasher.i32(accepted_clock.path);
    hasher.i32(accepted_clock.waste);
    hasher.i32(accepted_teacher.marker);
    hasher.i32(accepted_teacher.path);
    hasher.i32(accepted_teacher.waste);
    for (std::uint32_t bank = 0u;
         bank < kDevelopmentalAppendWitnessBankCount; ++bank) {
      hasher.u32(bank);
      for (const DevelopmentalAppendOffset offset :
           {developmental_append_reject_source_ingress_offset(leg, bank),
            developmental_append_reject_teacher_ingress_offset(leg, bank),
            developmental_append_reject_clock_ingress_offset(leg, bank),
            developmental_append_clock_escrow_offset(leg, bank),
            developmental_append_reject_escrow_offset(leg, bank)}) {
        hasher.i32(offset.marker);
        hasher.i32(offset.path);
        hasher.i32(offset.waste);
      }
    }
    for (std::uint32_t marker = 0u; marker < 4u; ++marker)
      for (std::uint32_t path = 0u; path < 4u; ++path) {
        if (path == marker) continue;
        for (std::uint32_t waste = 0u; waste < 4u; ++waste) {
          if (waste == marker || waste == path) continue;
          hasher.u32(marker);
          hasher.u32(path);
          hasher.u32(waste);
          hasher.u32(developmental_append_spent_source_basis(
              leg, marker, path, waste));
          hasher.word(developmental_append_spent_source_vacancy_word(
              leg, marker, path, waste));
          hasher.u32(developmental_append_clock_basis(
              leg, marker, path, waste));
          hasher.word(developmental_append_clock_vacancy_word(
              leg, marker, path, waste));
          hasher.word(developmental_append_escrow_word(
              true, leg, marker, waste));
        }
      }
  }
  for (std::uint32_t digit = 0u;
       digit < kDevelopmentalAppendJournalDigitCount; ++digit) {
    const std::uint32_t site = kDevelopmentalAppendJournalFirst +
                               digit * kDevelopmentalAppendAgeSitesPerDigit +
                               2u;
    hasher.u32(site);
    for (std::uint32_t encoded = 0u; encoded <= 2u; ++encoded)
      for (std::uint32_t marker = 0u; marker < 4u; ++marker)
        for (std::uint32_t path = 0u; path < 4u; ++path) {
          if (path == marker) continue;
          for (std::uint32_t waste = 0u; waste < 4u; ++waste) {
            if (waste == marker || waste == path) continue;
            hasher.word(developmental_append_journal_word(
                site, marker, path, waste, encoded));
          }
        }
  }
  // NL12 resident credit service.  The differentiated adult and its
  // separately lesionable site-17 enable turn accepted witnesses plus five
  // bank-bound rejection triples per leg through 36 P18/P20/P22 circuits
  // immediately before S_P.  Bind the exact owner policy, all S4 geometry and
  // all corner channel transpositions used by CPU and CUDA.
  hasher.u32(0x43533336u);
  hasher.u32(enum_value(LawFactor::developmental_credit_service));
  hasher.u32(kDevelopmentalCreditServiceRingCount);
  hasher.u32(kDevelopmentalCreditServiceMaxPeriod);
  hasher.u32(kDevelopmentalCreditServiceEnableSite);
  hasher.u32(kSpatialMacroClosureRadius);
  hasher.u32(0x4f574e4du);  // structural owner words mask carrier channels
  hasher.u32(0x454e4d43u);  // site-17 owner ignores only carrier traffic
  hasher.u32(0x554e4951u);  // ambiguous target ownership abstains
  hasher.u32(0x4350554bu);
  hasher.u32(0x43554444u);
  hasher.u32(0x43554143u);
  hasher.u32(0x50414745u);
  constexpr std::uint32_t kServiceOwnerSites[]{0u, 11u, 12u, 13u, 30u, 31u};
  hasher.u32(static_cast<std::uint32_t>(std::size(kServiceOwnerSites)));
  for (const std::uint32_t site : kServiceOwnerSites)
    hasher.u32(site);
  for (std::uint32_t marker = 0u; marker < 4u; ++marker)
    for (std::uint32_t path = 0u; path < 4u; ++path) {
      if (path == marker) continue;
      for (std::uint32_t waste = 0u; waste < 4u; ++waste) {
        if (waste == marker || waste == path) continue;
        hasher.u32(marker);
        hasher.u32(path);
        hasher.u32(waste);
        const SiteWord enable_base = developmental_append_product_word(
            kDevelopmentalCreditServiceEnableSite, marker, path, waste, 0u);
        hasher.word(enable_base);
        for (std::uint32_t carrier = 0u; carrier <= kCarrierMask; ++carrier)
          hasher.u32(developmental_credit_service_enable_word_matches(
              (enable_base & ~kCarrierMask) | carrier, marker, path, waste));
        hasher.u32(developmental_credit_service_enable_word_matches(
            kQ, marker, path, waste));
        hasher.u32(developmental_credit_service_enable_word_matches(
            enable_base ^ 0x00010000u, marker, path, waste));
        for (std::uint32_t ring = 0u;
             ring < kDevelopmentalCreditServiceRingCount; ++ring) {
          hasher.u32(ring);
          hasher.u32(developmental_credit_service_leg(ring));
          hasher.u32(developmental_credit_service_teacher_ring(ring));
          hasher.u32(developmental_credit_service_reject_source_ring(ring));
          hasher.u32(developmental_credit_service_reject_teacher_ring(ring));
          hasher.u32(developmental_credit_service_reject_clock_ring(ring));
          hasher.u32(developmental_credit_service_reject_bank(ring));
          hasher.u32(developmental_credit_service_period(ring));
          hasher.u32(developmental_credit_service_first_length(ring));
          hasher.u32(developmental_credit_service_second_length(ring));
          hasher.u32(developmental_credit_service_incoming(
              ring, marker, path, waste));
          hasher.u32(developmental_credit_service_outgoing(
              ring, marker, path, waste));
          const Int3 ingress = developmental_credit_service_ingress(
              ring, marker, path, waste);
          hasher.i32(ingress.x);
          hasher.i32(ingress.y);
          hasher.i32(ingress.z);
          for (std::uint32_t corner = 0u; corner < 4u; ++corner) {
            const Int3 offset = developmental_credit_service_corner(
                ring, corner, marker, path, waste);
            hasher.u32(corner);
            hasher.i32(offset.x);
            hasher.i32(offset.y);
            hasher.i32(offset.z);
            hasher.u32(developmental_credit_service_corner_incoming(
                ring, corner, marker, path, waste));
            hasher.u32(developmental_credit_service_corner_outgoing(
                ring, corner, marker, path, waste));
          }
        }
      }
    }
  hasher.u32(kCarrierPairSplitterSiteCount);
  for (std::uint32_t incoming = 0u; incoming < 8u; ++incoming) {
    for (std::uint32_t diverted = 0u; diverted < 8u; ++diverted) {
      if (incoming == diverted)
        continue;
      hasher.u32(incoming);
      hasher.u32(diverted);
      for (std::uint32_t index = 0u; index < kCarrierPairSplitterSiteCount; ++index) {
        const CarrierPairSplitterOffset offset =
            carrier_pair_splitter_offset(incoming, diverted, index);
        hasher.i32(offset.x);
        hasher.i32(offset.y);
        hasher.i32(offset.z);
        hasher.word(carrier_pair_splitter_word(incoming, diverted, index, false));
        hasher.word(carrier_pair_splitter_word(incoming, diverted, index, true));
      }
    }
  }
  hasher.u32(kProcessiveRearmSiteCount);
  for (std::uint32_t index = 0u; index < kProcessiveRearmSiteCount; ++index) {
    const ProcessiveRearmOffset offset = processive_rearm_offset(index);
    hasher.i32(offset.marker);
    hasher.i32(offset.path);
    hasher.i32(offset.waste);
    hasher.word(processive_rearm_candidate_word(index));
    hasher.word(processive_rearm_rearmed_word(index));
  }
  hasher.u32(kProcessiveReleaseSiteCount);
  hasher.u32(kProcessiveReleaseActionCount);
  hasher.u32(kProcessiveReleaseClaimDigitCount);
  hasher.u32(kProcessiveReleaseClaimStateCount);
  for (std::uint32_t digit = 0u;
       digit < kProcessiveReleaseClaimDigitCount; ++digit) {
    const ProcessiveReleaseOffset offset =
        processive_release_claim_offset(digit);
    hasher.i32(offset.marker);
    hasher.i32(offset.path);
    hasher.i32(offset.waste);
  }
  hasher.u32(0x554e5253u);  // unresolved footprint rejects atomically
  hasher.u32(0x45583138u);  // exact 12-row + six-claim footprint
  hasher.u32(0x42415236u);  // Q^6 is the ordinary bare endpoint
  hasher.u32(0x52455354u);  // Empty^6 is downstream carrier state
  hasher.u32(0x4c49464fu);  // learned events consume/replay LIFO
  for (std::uint32_t claim = 0u;
       claim < kProcessiveReleaseClaimStateCount;
       ++claim) {
    hasher.word(processive_release_claim_word(
        static_cast<ProcessiveReleaseClaim>(claim)));
  }
  // Bind the complete five-role adult stack grammar, including its selected
  // source and digit. Each base-5 word enumerates Empty,A,B,A-,B- over d2..d7.
  for (std::uint32_t packed = 0u; packed < 15625u; ++packed) {
    std::array<std::uint32_t, kProcessiveReleaseClaimDigitCount> roles{};
    std::uint32_t remaining = packed;
    for (std::uint32_t digit = 0u;
         digit < kProcessiveReleaseClaimDigitCount; ++digit) {
      roles[digit] = 1u + (remaining % 5u);
      remaining /= 5u;
    }
    std::uint32_t forward = 0u;
    bool found_positive = false;
    bool forward_valid = true;
    for (std::uint32_t digit = 0u;
         digit < kProcessiveReleaseClaimDigitCount; ++digit) {
      const std::uint32_t role = roles[digit];
      if (!found_positive && (role == 4u || role == 5u))
        continue;
      if (!found_positive && (role == 2u || role == 3u)) {
        found_positive = true;
        forward = 1u + digit * 2u + (role == 3u ? 1u : 0u);
        continue;
      }
      if (found_positive && role == 1u)
        continue;
      forward_valid = false;
      break;
    }
    if (!found_positive || !forward_valid)
      forward = 0u;

    std::uint32_t inverse_selected = 0u;
    bool found_negative = false;
    bool found_empty = false;
    bool inverse_valid = true;
    for (std::uint32_t digit = 0u;
         digit < kProcessiveReleaseClaimDigitCount; ++digit) {
      const std::uint32_t role = roles[digit];
      if (!found_empty && (role == 4u || role == 5u)) {
        found_negative = true;
        inverse_selected =
            1u + digit * 2u + (role == 5u ? 1u : 0u);
        continue;
      }
      if (role == 1u) {
        found_empty = true;
        continue;
      }
      inverse_valid = false;
      break;
    }
    if (!found_negative || !inverse_valid)
      inverse_selected = 0u;
    hasher.u32(forward);
    hasher.u32(inverse_selected);
  }
  for (std::uint32_t index = 0u; index < kProcessiveReleaseSiteCount; ++index) {
    const ProcessiveReleaseOffset offset = processive_release_offset(index);
    hasher.i32(offset.marker);
    hasher.i32(offset.path);
    hasher.i32(offset.waste);
    for (std::uint32_t action = 0u; action < kProcessiveReleaseActionCount; ++action) {
      hasher.word(processive_release_staged_word(action, index));
      hasher.word(processive_release_released_word(action, index));
    }
  }
  hasher.u32(kCarrierCornerSiteCount);
  // A center can be claimed by any of the 56 ordered lock-pair owners.  Bind
  // the complete owner-cardinality policy separately from the carrier-byte
  // predicate so a one-compatible-of-two asymmetric rule cannot share an
  // identity with the reversible unique-owner rule.
  hasher.u32(57u);
  for (std::uint32_t present = 0u; present <= 56u; ++present)
    for (std::uint32_t compatible = 0u; compatible <= 56u; ++compatible) {
      hasher.u32(present);
      hasher.u32(compatible);
      hasher.u32(carrier_corner_unique_owner_matches(
          present, compatible));
    }
  for (std::uint32_t incoming = 0u; incoming < 8u; ++incoming) {
    for (std::uint32_t outgoing = 0u; outgoing < 8u; ++outgoing) {
      if (incoming == outgoing)
        continue;
      hasher.u32(incoming);
      hasher.u32(outgoing);
      for (std::uint32_t index = 0u; index < kCarrierCornerSiteCount; ++index) {
        const CarrierCornerOffset offset = carrier_corner_offset(incoming, outgoing, index);
        hasher.i32(offset.x);
        hasher.i32(offset.y);
        hasher.i32(offset.z);
        hasher.word(carrier_corner_word(incoming, outgoing, index, false));
        hasher.word(carrier_corner_word(incoming, outgoing, index, true));
      }
      // K_corner is not only its three-site geometry.  Its executable centre
      // predicate and transposition own exactly two of the eight carrier
      // channels and preserve the other six.  Serialize the complete carrier
      // byte truth table so a context-erasing implementation changes the law
      // identity even when the lock geometry is untouched.
      hasher.u32(256u);
      for (std::uint32_t byte = 0u; byte < 256u; ++byte) {
        const SiteWord word = static_cast<SiteWord>(byte);
        hasher.word(word);
        hasher.u32(carrier_corner_center_matches(word, incoming, outgoing));
        hasher.u32(carrier_corner_center_released(word, incoming, outgoing));
        hasher.word(carrier_corner_transpose(word, incoming, outgoing));
      }
      // Every non-carrier bit is an explicit negative domain, in both named
      // phases.  This binds the structural guard rather than inferring it from
      // the carrier-only table above.
      hasher.u32(kBitsPerSite - kFaceShift);
      for (std::uint32_t bit = kFaceShift; bit < kBitsPerSite; ++bit) {
        for (const SiteWord named : {carrier_bit(incoming),
                                     carrier_bit(outgoing)}) {
          const SiteWord word = named | (SiteWord{1u} << bit);
          hasher.word(word);
          hasher.u32(carrier_corner_center_matches(word, incoming, outgoing));
          hasher.u32(carrier_corner_center_released(word, incoming, outgoing));
          hasher.word(carrier_corner_transpose(word, incoming, outgoing));
        }
      }
    }
  }
  hasher.u32(kStreamChannelCount);
  for (std::uint32_t index = 0u; index < kStreamChannelCount; ++index) {
    const StreamChannel mapping = stream_channel(index);
    hasher.u32(mapping.channel);
    hasher.u32(direction_index(mapping.direction));
  }
  return hasher.finish();
}

ContentAddress execution_profile_identity(std::uint32_t aperture_chunks) {
  CanonicalHasher hasher;
  hasher.u32(0x45585031u);
  const ContentAddress law = canonical_law_identity();
  for (std::uint8_t byte : law.digest)
    hasher.u32(byte);
  hasher.u64(law.byte_count);
  const std::uint64_t aperture_sites = static_cast<std::uint64_t>(aperture_chunks) * kChunkSites;
  const std::uint64_t aperture_bits = aperture_sites * kBitsPerSite;
  const std::uint64_t aperture_bytes = aperture_sites * sizeof(SiteWord);
  hasher.u64(aperture_sites);
  hasher.u64(aperture_bits);
  hasher.u64(aperture_bytes);
  hasher.u32(kChunkEdge);
  hasher.u64(kChunkSites);
  hasher.u64(kChunkBytes);
  hasher.u32(aperture_chunks);
  return hasher.finish();
}

ContentAddress canonical_execution_profile_identity() {
  return execution_profile_identity(kProductionChunkSlots);
}

}  // namespace substrate::bcc32
