#pragma once

#include <cuda_runtime.h>

#include <cstdint>

#include "bcc32_geometry.cuh"
#include "bcc32_types.cuh"

namespace substrate::bcc32 {

// These descriptors are the one canonical executable circuit. CPU, CUDA,
// provenance, and symbolic tooling consume the same arrays.
enum class SiteOpcode : std::uint8_t {
    complement_pair_tetrad = 1u,
    pair_rotor = 2u,
    differentiated_three_factor_capture = 3u,
    differentiated_signed_synapse = 4u,
  differentiated_processive_stage = 5u,
  differentiated_terminal_maturation = 6u,
};

struct SiteGate {
    SiteOpcode opcode{};
    std::uint8_t target_shift = 0u;
    std::uint8_t control_shift = 0u;
};

inline constexpr std::uint32_t kSiteGateCount = 14u;

[[nodiscard]] __host__ __device__ constexpr SiteGate site_gate(std::uint32_t index) {
    switch (index) {
    case 0u:
        return {SiteOpcode::differentiated_three_factor_capture, 0u, 0u};
    case 1u:
        return {SiteOpcode::complement_pair_tetrad, 0u, 0u};
    case 2u:
        return {SiteOpcode::complement_pair_tetrad, 4u, 0u};
    case 3u:
        return {SiteOpcode::complement_pair_tetrad, 8u, 0u};
    case 4u:
        return {SiteOpcode::complement_pair_tetrad, 12u, 0u};
    case 5u:
        return {SiteOpcode::complement_pair_tetrad, 16u, 0u};
    case 6u:
        return {SiteOpcode::complement_pair_tetrad, 20u, 0u};
    case 7u:
        return {SiteOpcode::complement_pair_tetrad, 24u, 0u};
    case 8u:
        return {SiteOpcode::complement_pair_tetrad, 28u, 0u};
    case 9u:
        return {SiteOpcode::differentiated_processive_stage, 0u, 0u};
    case 10u:
        return {SiteOpcode::pair_rotor, 28u, 0u};
    case 11u:
      return {SiteOpcode::pair_rotor, 28u, 4u};
    case 12u:
        return {SiteOpcode::differentiated_signed_synapse, 0u, 0u};
    case 13u:
        return {SiteOpcode::differentiated_terminal_maturation, 0u, 0u};
    }
    return {};
}

enum class EdgeOpcode : std::uint8_t {
    pair_exchange = 1u,
    controlled_transpose = 2u,
};

enum class EdgeRole : std::uint8_t {
    positive_carrier = 1u,
    negative_carrier = 2u,
    positive_face = 3u,
    negative_face = 4u,
    owned_bond = 5u,
    conformation = 6u,
    reactive = 7u,
    energy = 8u,
};

enum class EdgeControl : std::uint8_t {
    none = 0u,
    bit = 1u,
    xor_bits = 2u,
};

struct EdgeGate {
    EdgeOpcode opcode{};
    EdgeRole first{};
    EdgeRole second{};
    EdgeRole third{};
    EdgeRole fourth{};
    EdgeControl control{};
    EdgeRole control_first{};
    EdgeRole control_second{};
};

[[nodiscard]] __host__ __device__ constexpr EdgeGate pair_exchange_gate() {
    return {EdgeOpcode::pair_exchange,  EdgeRole::positive_carrier, EdgeRole::energy,
            EdgeRole::owned_bond,       EdgeRole::reactive,         EdgeControl::none,
            EdgeRole::positive_carrier, EdgeRole::positive_carrier};
}

[[nodiscard]] __host__ __device__ constexpr EdgeGate transpose_gate(EdgeRole first, EdgeRole second,
                                                                    EdgeRole control) {
    return {EdgeOpcode::controlled_transpose,
            first,
            second,
            EdgeRole::positive_carrier,
            EdgeRole::positive_carrier,
            EdgeControl::bit,
            control,
            control};
}

[[nodiscard]] __host__ __device__ constexpr EdgeGate xor_transpose_gate(EdgeRole first,
                                                                        EdgeRole second,
                                                                        EdgeRole control_first,
                                                                        EdgeRole control_second) {
    return {EdgeOpcode::controlled_transpose,
            first,
            second,
            EdgeRole::positive_carrier,
            EdgeRole::positive_carrier,
            EdgeControl::xor_bits,
            control_first,
            control_second};
}

inline constexpr std::uint32_t kEdgeGateCount = 13u;

[[nodiscard]] __host__ __device__ constexpr EdgeGate edge_gate(std::uint32_t index) {
    switch (index) {
    case 0u:
        return pair_exchange_gate();
    case 1u:
        return transpose_gate(EdgeRole::positive_carrier, EdgeRole::energy, EdgeRole::conformation);
    case 2u:
        return transpose_gate(EdgeRole::energy, EdgeRole::conformation, EdgeRole::owned_bond);
    case 3u:
        return transpose_gate(EdgeRole::conformation, EdgeRole::positive_face, EdgeRole::reactive);
    case 4u:
        return transpose_gate(EdgeRole::positive_face, EdgeRole::negative_face,
                              EdgeRole::conformation);
    case 5u:
        return transpose_gate(EdgeRole::negative_face, EdgeRole::reactive, EdgeRole::energy);
    case 6u:
        return transpose_gate(EdgeRole::reactive, EdgeRole::negative_carrier,
                              EdgeRole::conformation);
    case 7u:
        return transpose_gate(EdgeRole::negative_carrier, EdgeRole::positive_carrier,
                              EdgeRole::negative_face);
    case 8u:
        return transpose_gate(EdgeRole::positive_carrier, EdgeRole::negative_carrier,
                              EdgeRole::owned_bond);
    case 9u:
        return transpose_gate(EdgeRole::positive_face, EdgeRole::negative_face,
                              EdgeRole::owned_bond);
    case 10u:
        return transpose_gate(EdgeRole::owned_bond, EdgeRole::energy, EdgeRole::positive_face);
    case 11u:
        return transpose_gate(EdgeRole::owned_bond, EdgeRole::reactive, EdgeRole::negative_face);
    case 12u:
        return xor_transpose_gate(EdgeRole::conformation, EdgeRole::reactive,
                                  EdgeRole::positive_carrier, EdgeRole::negative_carrier);
    }
    return {};
}

enum class LawFactor : std::uint8_t {
    site = 1u,
    edge = 2u,
  carrier_pair_splitter = 3u,
  processive_release = 4u,
  carrier_corner = 5u,
  stream = 6u,
  processive_rearm = 7u,
  developmental_append = 8u,
  eligibility_residual_junction = 9u,
  prediction_residual_route_toggle = 10u,
  developmental_learned_receptor = 11u,
  developmental_credit_service = 12u,
};

// Basis-role authority for the eligibility/residual junction.  A complete S4
// orbit assigns these four distinct roles; each assignment is evaluated for
// both signs of the incoming and outgoing directions.  Controls are crossed around the
// center at 2*outgoing + the other control basis.  The enable predicate reads
// neither carrier target.
struct EligibilityResidualJunctionAuthority {
  std::uint8_t outgoing_role = 0u;
  std::uint8_t control_a_role = 1u;
  std::uint8_t incoming_role = 2u;
  std::uint8_t control_b_role = 3u;
  std::int8_t outgoing_distance = 2;
  std::uint8_t outgoing_sign_count = 2u;
  std::uint8_t incoming_sign_count = 2u;
  std::uint8_t target_independent = 1u;
  std::uint8_t overlap_abstains = 1u;
  std::uint8_t exact_carrier_vacancy_controls = 1u;
  std::uint8_t positive_incoming_control_offset = 3u;
  std::uint8_t endogenous_owner_required = 1u;
};

inline constexpr EligibilityResidualJunctionAuthority
    kEligibilityResidualJunctionAuthority{};

// One processive release is a bounded local collision, not a host-side
// instruction.  These coefficients are expressed in the three selected BCC
// bases (marker, path, waste); a fourth basis remains free.  CPU reference,
// CUDA, paged execution, provenance, and symbolic mirrors must consume this
// descriptor rather than restating the tissue pattern.
struct ProcessiveReleaseOffset {
  std::int8_t marker = 0;
  std::int8_t path = 0;
  std::int8_t waste = 0;
};

inline constexpr std::uint32_t kProcessiveReleaseSiteCount = 12u;
inline constexpr std::uint32_t kProcessiveReleaseActionCount = 4u;
enum class ProcessiveReleaseClaim : std::uint32_t {
  bare = 0u,
  empty = 1u,
  adult_a = 2u,
  adult_b = 3u,
  consumed_a = 4u,
  consumed_b = 5u,
};
inline constexpr std::uint32_t kProcessiveReleaseClaimStateCount = 6u;
inline constexpr std::uint32_t kProcessiveReleaseClaimDigitCount = 6u;
inline constexpr std::uint32_t kProcessiveRoleIngressDigit = 0u;
inline constexpr std::uint32_t kProcessiveRoleLandedDigit = 1u;
inline constexpr std::uint32_t kProcessiveRoleGuardFirst = 2u;
inline constexpr std::uint32_t kProcessiveReleaseFootprintCount =
    kProcessiveReleaseSiteCount + kProcessiveReleaseClaimDigitCount;
inline constexpr std::uint32_t kProcessiveReleaseRoleFootprintCount =
    kProcessiveReleaseFootprintCount + 1u;

[[nodiscard]] __host__ __device__ constexpr bool
processive_action_carries(std::uint32_t action) {
  return action >= 2u && action < kProcessiveReleaseActionCount;
}

// Append journal digits 2..7, expressed relative to append site 10 (the
// processive centre).  Their adult-frame append sites are respectively
// 112,115,118,121,124,127.  Every release match owns all six atoms, including
// the unchanged atoms, so no partial stack can commit through a collision.
[[nodiscard]] __host__ __device__ constexpr ProcessiveReleaseOffset
processive_release_claim_offset(std::uint32_t digit) {
  switch (digit) {
    case 0u:
      return {-5, 3, 4};
    case 1u:
      return {-1, 3, 4};
    case 2u:
      return {3, 3, 4};
    case 3u:
      return {7, 3, 4};
    case 4u:
      return {-3, 3, 6};
    case 5u:
      return {1, 3, 6};
  }
  return {};
}

[[nodiscard]] __host__ __device__ constexpr ProcessiveReleaseOffset
processive_successor_claim_offset(std::uint32_t digit) {
  ProcessiveReleaseOffset result = processive_release_claim_offset(digit);
  result.path = static_cast<std::int8_t>(result.path + 6);
  return result;
}

[[nodiscard]] __host__ __device__ constexpr ProcessiveReleaseOffset processive_release_offset(
    std::uint32_t index) {
  switch (index) {
    case 0u:
      return {-1, 0, 0};
    case 1u:
      return {0, 0, 0};
    case 2u:
      return {1, 0, 0};
    case 3u:
      return {2, 0, 0};
    case 4u:
      return {0, 1, 0};
    case 5u:
      return {0, -1, 0};
    case 6u:
      return {0, 2, 0};
    case 7u:
      return {0, -2, 0};
    case 8u:
      return {0, 0, 2};
    case 9u:
      return {0, 0, -2};
    case 10u:
      return {0, 0, 6};
    case 11u:
      return {0, 0, 8};
  }
  return {};
}

[[nodiscard]] __host__ __device__ constexpr SiteWord processive_release_staged_word(
    std::uint32_t action, std::uint32_t index) {
  if (index == 1u) {
    switch (action) {
      case 0u:
        return 0x000200eeu;
      case 1u:
        return 0x020000eeu;
      case 2u:
        return 0x000200eeu;
      case 3u:
        return 0x022000ceu;
    }
  }
  if (index == 2u)
    return 0x000100efu;
  if (index == 3u)
    return 0x000000efu;
  if (action == 2u && index == 4u)
    return 0x002000dfu;
  if (action == 3u && index == 5u)
    return 0x200000fdu;
  // The lock pair is unchanged by either endpoint. Its first word names the
  // unused fourth basis; its second names the marker basis. Their separation
  // names waste, leaving path as the only remaining basis. Both are stable
  // differentiated singleton cells under a complete superstep.
  if (index == 10u)
    return 0x000008ffu;
  if (index == 11u)
    return 0x000001ffu;
  return kQuiescentWord;
}

[[nodiscard]] __host__ __device__ constexpr SiteWord processive_release_released_word(
    std::uint32_t action, std::uint32_t index) {
  if (index == 0u)
    return 0x000000feu;
  if (index == 1u) {
    switch (action) {
      case 0u:
      case 1u:
        return 0x000110feu;
      case 2u:
      case 3u:
        return 0x100010feu;
    }
  }
  if (index == 2u)
    return 0x000100efu;
  if (index == 3u)
    return 0x000000efu;
  if ((action == 0u || action == 2u) && index == 9u)
    return 0x000000bfu;
  if (action == 1u && index == 6u)
    return 0x000000fdu;
  if (action == 3u && index == 7u)
    return 0x000000dfu;
  if (index == 10u)
    return 0x000008ffu;
  if (index == 11u)
    return 0x000001ffu;
  return kQuiescentWord;
}

// The six sideband atoms are graph-separated from the ordinary twelve-site
// processive row. Digit 0 is a reusable role ingress, digit 1 stores the role
// permanently landed in this body, and digits 2..5 are exact collision guards.
// A carry moves ingress provenance to the unique successor body's ingress;
// a terminal transition moves it to this body's landed atom. Consumed A-/B-
// remain named only so legacy checkpoints are rejected deterministically.
[[nodiscard]] __host__ __device__ constexpr SiteWord
processive_release_claim_word(ProcessiveReleaseClaim claim) {
  if (claim == ProcessiveReleaseClaim::bare) return kQuiescentWord;
  if (claim == ProcessiveReleaseClaim::empty)
    return kQuiescentWord | face_bit(0u);
  if (claim == ProcessiveReleaseClaim::adult_a)
    return kQuiescentWord | face_bit(1u);
  if (claim == ProcessiveReleaseClaim::adult_b)
    return kQuiescentWord | face_bit(2u);
  if (claim == ProcessiveReleaseClaim::consumed_a)
    return kQuiescentWord | face_bit(5u);
  return kQuiescentWord | face_bit(6u);
}

// The zero-underflow product is a local candidate, not waste. After S_P has
// settled action zero, K_edge exposes its exact candidate phase. This
// nine-site receptor maps that phase back to the staged processive cell before
// K_processive_release runs. The mapping is a transposition, so the factor is
// its own inverse; a third differentiated lock makes rearm independently
// lesionable.
struct ProcessiveRearmOffset {
  std::int8_t marker = 0;
  std::int8_t path = 0;
  std::int8_t waste = 0;
};

inline constexpr std::uint32_t kProcessiveRearmSiteCount = 9u;

[[nodiscard]] __host__ __device__ constexpr ProcessiveRearmOffset processive_rearm_offset(
    std::uint32_t index) {
  switch (index) {
    case 0u:
      return {0, 0, -3};
    case 1u:
      return {-1, 0, 0};
    case 2u:
      return {0, 0, 0};
    case 3u:
      return {1, 0, 0};
    case 4u:
      return {2, 0, 0};
    case 5u:
      return {3, 0, 0};
    case 6u:
      return {0, 0, 6};
    case 7u:
      return {0, 0, 8};
    case 8u:
      return {0, 0, 10};
  }
  return {};
}

[[nodiscard]] __host__ __device__ constexpr SiteWord processive_rearm_candidate_word(
    std::uint32_t index) {
  switch (index) {
    case 0u:
      return 0x000000bfu;
    case 1u:
      return 0x000000feu;
    case 2u:
      return 0x100010feu;
    case 3u:
      return 0x000100efu;
    case 4u:
      return 0x000000efu;
    case 6u:
      return 0x000008ffu;
    case 7u:
      return 0x000001ffu;
    case 8u:
      return 0x000004ffu;
    default:
      return kQuiescentWord;
  }
}

[[nodiscard]] __host__ __device__ constexpr SiteWord processive_rearm_rearmed_word(
    std::uint32_t index) {
  switch (index) {
    case 2u:
      return 0x000200eeu;
    case 3u:
      return 0x000100efu;
    case 4u:
      return 0x000000efu;
    case 6u:
      return 0x000008ffu;
    case 7u:
      return 0x000001ffu;
    case 8u:
      return 0x000004ffu;
    default:
      return kQuiescentWord;
  }
}

// A carrier corner is a three-site grown road joint. The two immutable,
// differentiated singleton locks name the signed incoming and outgoing rays
// by their positions. The centre contains only the transported vacancy. The
// factor swaps the vacancy's signed lane; S_P performs the actual movement.
// This keeps the routing authority in persistent tissue rather than in a
// movable one-site direction tag.
struct CarrierCornerOffset {
  std::int8_t x = 0;
  std::int8_t y = 0;
  std::int8_t z = 0;
};

inline constexpr std::uint32_t kCarrierCornerSiteCount = 3u;

[[nodiscard]] __host__ __device__ constexpr CarrierCornerOffset carrier_corner_offset(
    std::uint32_t incoming, std::uint32_t outgoing, std::uint32_t index) {
  if (index == 0u)
    return {};
  const std::uint32_t direction = index == 1u ? incoming : outgoing;
  const std::int32_t scale = index == 1u ? 6 : 8;
  const Int3 ray = direction_offset(static_cast<Direction>(direction));
  std::uint32_t side_basis = 0u;
  while (side_basis == (incoming & 3u) || side_basis == (outgoing & 3u))
    ++side_basis;
  const Int3 side = direction_offset(static_cast<Direction>(side_basis));
  const std::int32_t side_scale = index == 1u ? 2 : 3;
  return {static_cast<std::int8_t>(scale * ray.x + side_scale * side.x),
          static_cast<std::int8_t>(scale * ray.y + side_scale * side.y),
          static_cast<std::int8_t>(scale * ray.z + side_scale * side.z)};
}

[[nodiscard]] __host__ __device__ constexpr SiteWord carrier_corner_word(std::uint32_t incoming,
                                                                         std::uint32_t outgoing,
                                                                         std::uint32_t index,
                                                                         bool released) {
  if (index == 0u)
    return kQuiescentWord ^ carrier_bit(released ? outgoing : incoming);
  const std::uint32_t direction = index == 1u ? incoming : outgoing;
  return kQuiescentWord |
         (index == 1u ? face_bit(direction & 3u) : channel_bit(kReactiveShift, direction & 3u));
}

// A carrier corner owns only its named incoming/outgoing carrier channels.
// The other six channels are independent traffic and must survive the local
// turn bit-for-bit.  Structural-bearing centre words are never carrier-road
// traffic.  Exactly one named channel must be occupied, making the transpose
// an exact involution in every orthogonal carrier context.
[[nodiscard]] __host__ __device__ constexpr bool
carrier_corner_center_matches(SiteWord word, std::uint32_t incoming,
                              std::uint32_t outgoing) {
  if (incoming >= 8u || outgoing >= 8u || incoming == outgoing ||
      (word & ~kCarrierMask) != 0u)
    return false;
  const bool incoming_occupied = (word & carrier_bit(incoming)) != 0u;
  const bool outgoing_occupied = (word & carrier_bit(outgoing)) != 0u;
  return incoming_occupied != outgoing_occupied;
}

[[nodiscard]] __host__ __device__ constexpr bool
carrier_corner_center_released(SiteWord word, std::uint32_t incoming,
                               std::uint32_t outgoing) {
  return carrier_corner_center_matches(word, incoming, outgoing) &&
         (word & carrier_bit(incoming)) != 0u;
}

[[nodiscard]] __host__ __device__ constexpr SiteWord
carrier_corner_transpose(SiteWord word, std::uint32_t incoming,
                         std::uint32_t outgoing) {
  return carrier_corner_center_matches(word, incoming, outgoing)
      ? word ^ carrier_bit(incoming) ^ carrier_bit(outgoing)
      : word;
}

// Persistent lock-pair multiplicity is part of the reversible local law.  A
// co-centred second owner makes both phases abstain even when only one owner is
// compatible with the transient centre byte before the proposed transpose.
[[nodiscard]] __host__ __device__ constexpr bool
carrier_corner_unique_owner_matches(std::uint32_t present_owners,
                                    std::uint32_t compatible_owners) {
  return present_owners == 1u && compatible_owners == 1u;
}

// A carrier-pair splitter is a four-site, locked coincidence organ. Two
// adjacent vacancies on one signed ray are a spatial representation of a
// two-quantum packet. The downstream vacancy is retained while the upstream
// vacancy is turned onto a different ray. Occupancy is conserved exactly;
// a singleton vacancy cannot trigger the organ.
struct CarrierPairSplitterOffset {
  std::int8_t x = 0;
  std::int8_t y = 0;
  std::int8_t z = 0;
};

inline constexpr std::uint32_t kCarrierPairSplitterSiteCount = 5u;

[[nodiscard]] __host__ __device__ constexpr CarrierPairSplitterOffset carrier_pair_splitter_offset(
    std::uint32_t incoming, std::uint32_t diverted, std::uint32_t index) {
  const Int3 incoming_ray = direction_offset(static_cast<Direction>(incoming));
  const Int3 diverted_ray = direction_offset(static_cast<Direction>(diverted));
  if (index == 1u)
    return {static_cast<std::int8_t>(-incoming_ray.x), static_cast<std::int8_t>(-incoming_ray.y),
            static_cast<std::int8_t>(-incoming_ray.z)};
  if (index == 2u)
    return {static_cast<std::int8_t>(7 * incoming_ray.x),
            static_cast<std::int8_t>(7 * incoming_ray.y),
            static_cast<std::int8_t>(7 * incoming_ray.z)};
  if (index == 3u)
    return {static_cast<std::int8_t>(9 * diverted_ray.x),
            static_cast<std::int8_t>(9 * diverted_ray.y),
            static_cast<std::int8_t>(9 * diverted_ray.z)};
  std::uint32_t side_basis = 0u;
  while (side_basis == (incoming & 3u) || side_basis == (diverted & 3u))
    ++side_basis;
  const Int3 side_ray = direction_offset(static_cast<Direction>(side_basis));
  if (index == 4u)
    return {static_cast<std::int8_t>(11 * side_ray.x + 4 * incoming_ray.x),
            static_cast<std::int8_t>(11 * side_ray.y + 4 * incoming_ray.y),
            static_cast<std::int8_t>(11 * side_ray.z + 4 * incoming_ray.z)};
  return {};
}

[[nodiscard]] __host__ __device__ constexpr SiteWord carrier_pair_splitter_word(
    std::uint32_t incoming, std::uint32_t diverted, std::uint32_t index, bool released) {
  if (index == 0u)
    return kQuiescentWord ^ carrier_bit(incoming);
  if (index == 1u)
    return kQuiescentWord ^ carrier_bit(released ? diverted : incoming);
  if (index == 2u)
    return kQuiescentWord | face_bit(incoming & 3u);
  if (index == 3u)
    return kQuiescentWord | channel_bit(kReactiveShift, diverted & 3u);
  std::uint32_t side_basis = 0u;
  while (side_basis == (incoming & 3u) || side_basis == (diverted & 3u))
    ++side_basis;
  return kQuiescentWord | face_bit(side_basis);
}

inline constexpr std::uint32_t kForwardFactorCount = 12u;
// Conservative BCC-hop closure required by every current spatial macro
// descriptor. Representation layers may over-include this radius but may not
// skip it.
inline constexpr std::uint32_t kSpatialMacroClosureRadius = 26u;

[[nodiscard]] __host__ __device__ constexpr LawFactor forward_factor(std::uint32_t index) {
    switch (index) {
    case 0u:
      return LawFactor::developmental_learned_receptor;
    case 1u:
      return LawFactor::developmental_append;
    case 2u:
      return LawFactor::eligibility_residual_junction;
    case 3u:
        return LawFactor::site;
    case 4u:
        return LawFactor::edge;
    case 5u:
      return LawFactor::carrier_pair_splitter;
    case 6u:
      return LawFactor::carrier_corner;
    case 7u:
      return LawFactor::processive_rearm;
    case 8u:
      return LawFactor::processive_release;
    case 9u:
      return LawFactor::prediction_residual_route_toggle;
    case 10u:
      return LawFactor::developmental_credit_service;
    case 11u:
        return LawFactor::stream;
    }
    return {};
}

struct StreamChannel {
    std::uint8_t channel = 0u;
    Direction direction = Direction::positive_u0;
};

inline constexpr std::uint32_t kStreamChannelCount = 8u;

[[nodiscard]] __host__ __device__ constexpr StreamChannel stream_channel(std::uint32_t index) {
    return {static_cast<std::uint8_t>(index), static_cast<Direction>(index)};
}

inline constexpr std::uint32_t kLawNetlistSchema = 0x4e4c3133u;  // "NL13"

} // namespace substrate::bcc32
