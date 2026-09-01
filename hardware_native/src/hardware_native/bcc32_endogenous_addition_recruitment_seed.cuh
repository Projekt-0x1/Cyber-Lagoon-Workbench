#pragma once

// Preregistered current-F boundary for endogenous unary recruitment.
//
// This is a seed and a falsifiable local target, not an implementation of an
// adder.  Current F must turn one physical token at the unique cursor into one
// recruited balanced digit and move that same cursor to the next generic
// balanced digit.  No host-selected index or result word occurs here.

#include <array>
#include <cstdint>

#include "bcc32_grown_multicontact_integrator.cuh"
#include "bcc32_processive_weight_region_seed.cuh"

namespace substrate::bcc32::endogenous_addition_recruitment {

using FounderHash = std::uint32_t;
using Permutation = grown_multicontact_integrator::Permutation;

inline constexpr std::array<FounderHash, 3u> kPreregisteredFounderHashes{
    {0xa6310101u, 0xa6310202u, 0xa6310304u}};
inline constexpr FounderHash kHeldOutFounderHash = 0x9e3779b9u;

inline constexpr std::uint32_t kRoadDigitCount = 7u;
inline constexpr std::uint32_t kSitesPerDigit = 2u;
inline constexpr std::uint32_t kFounderSiteCount = kRoadDigitCount * kSitesPerDigit;
inline constexpr std::uint32_t kInternalBasis = 0u;
inline constexpr std::uint32_t kRoadBasis = 1u;
inline constexpr std::uint32_t kRouteABasis = 0u;
inline constexpr std::uint32_t kRouteBBasis = 1u;
inline constexpr std::uint32_t kTokenDistance = 8u;
inline constexpr ProcessiveWeightRegionSeedHash kAuthoredPrefixHash =
    make_processive_weight_region_seed_hash(3u, 0u, 1u, 2u, kProcessiveWeightGene);

// Exact codes already measured by bcc32_hole_neutral_write_contract.
// A code packs the two eight-bit single-basis cell descriptions low/high.
inline constexpr std::uint16_t kUncommittedCode = 0x0236u;
inline constexpr std::uint16_t kRecruitedCode = 0x02a6u;
inline constexpr std::uint16_t kTipCode = 0x02acu;
inline constexpr std::uint16_t kSpentTipCode = 0x222cu;

struct Grammar {
  Permutation outer{{0u, 1u, 2u, 3u}};
  std::uint32_t road_pitch = 1u;
};

[[nodiscard]] constexpr Grammar grammar(FounderHash hash) {
  Grammar result{};
  const std::uint32_t rotation = hash & 0x3u;
  for (std::uint32_t basis = 0u; basis < 4u; ++basis)
    result.outer[basis] = (basis + rotation) & 0x3u;
  return result;
}

[[nodiscard]] constexpr SiteWord code_bit(std::uint32_t bit) {
  switch (bit) {
    case 0u:
      return carrier_bit(kInternalBasis);
    case 1u:
      return carrier_bit(kInternalBasis + 4u);
    case 2u:
      return face_bit(kInternalBasis);
    case 3u:
      return face_bit(kInternalBasis + 4u);
    case 4u:
      return owned_bond_bit(kInternalBasis);
    case 5u:
      return channel_bit(kConformationShift, kInternalBasis);
    case 6u:
      return channel_bit(kReactiveShift, kInternalBasis);
    default:
      return energy_bit(kInternalBasis);
  }
}

[[nodiscard]] constexpr SiteWord cell_word(std::uint8_t code) {
  SiteWord word = kQ;
  for (std::uint32_t bit = 0u; bit < 8u; ++bit) {
    const SiteWord field = code_bit(bit);
    const bool present = (code & (1u << bit)) != 0u;
    if (bit < 2u) {
      if (!present)
        word &= ~field;
    } else if (present) {
      word |= field;
    }
  }
  return word;
}

[[nodiscard]] constexpr std::array<SiteWord, 2u> code_words(std::uint16_t code) {
  return {{cell_word(static_cast<std::uint8_t>(code & 0xffu)),
           cell_word(static_cast<std::uint8_t>((code >> 8u) & 0xffu))}};
}

[[nodiscard]] constexpr Int3 add(Int3 left, Int3 right) {
  return {left.x + right.x, left.y + right.y, left.z + right.z};
}

[[nodiscard]] constexpr Int3 scale(Int3 value, std::int32_t factor) {
  return {value.x * factor, value.y * factor, value.z * factor};
}

[[nodiscard]] constexpr DevelopmentalSeedSite make_site(Int3 coordinate, SiteWord word) {
  return {static_cast<std::int8_t>(coordinate.x), static_cast<std::int8_t>(coordinate.y),
          static_cast<std::int8_t>(coordinate.z), word};
}

[[nodiscard]] constexpr Int3 digit_origin(FounderHash hash, std::uint32_t digit) {
  const Grammar selected = grammar(hash);
  const Int3 base = scale(direction_offset(static_cast<Direction>(kRoadBasis)),
                          static_cast<std::int32_t>(digit * selected.road_pitch));
  return grown_multicontact_integrator::permute_coordinate(base, selected.outer);
}

[[nodiscard]] constexpr std::array<DevelopmentalSeedSite, kFounderSiteCount> founder_seed(
    FounderHash hash) {
  std::array<DevelopmentalSeedSite, kFounderSiteCount> result{};
  const Grammar selected = grammar(hash);
  const Int3 internal = grown_multicontact_integrator::permute_coordinate(
      direction_offset(static_cast<Direction>(kInternalBasis)), selected.outer);
  std::uint32_t write = 0u;
  for (std::uint32_t digit = 0u; digit < kRoadDigitCount; ++digit) {
    const auto words = code_words(digit == 0u ? kTipCode : kUncommittedCode);
    const Int3 origin = digit_origin(hash, digit);
    result[write++] =
        make_site(origin, grown_multicontact_integrator::permute_word(words[0], selected.outer));
    result[write++] = make_site(add(origin, internal), grown_multicontact_integrator::permute_word(
                                                           words[1], selected.outer));
  }
  return result;
}

[[nodiscard]] constexpr std::uint32_t route_lane(FounderHash hash, std::uint32_t source) {
  const std::uint32_t base = source == 0u ? kRouteABasis : kRouteBBasis;
  return grammar(hash).outer[base];
}

[[nodiscard]] constexpr bool generic_geometry_is_valid(FounderHash hash) {
  return route_lane(hash, 0u) != route_lane(hash, 1u) && grammar(hash).road_pitch == 1u;
}

// Minimal law-collision proposal if current F is RED.  Each symbol is a
// two-cell codeword.  The carrier deficit is part of the left codeword at
// contact and is returned on the complementary lane.  This one transposition,
// plus every S4 image, is its own inverse when input/product are exchanged:
//
//   (TIP, U, P_in=0) <-> (R, TIP, P_return=0)
//
// U and R have equal Delta N_Q.  One TIP and one carrier deficit occur on both
// sides.  Support is exactly two adjacent two-cell digits; a non-tip U/U edge
// cannot match, which is the no-avalanche guard.
struct CollisionProposal {
  std::uint16_t input_left = kTipCode;
  std::uint16_t input_right = kUncommittedCode;
  std::uint16_t product_left = kRecruitedCode;
  std::uint16_t product_right = kTipCode;
  std::uint32_t incoming_lane = kRouteABasis;
  std::uint32_t return_lane = kRouteABasis + 4u;
};

[[nodiscard]] constexpr CollisionProposal collision_proposal() {
  return {};
}

static_assert(generic_geometry_is_valid(kPreregisteredFounderHashes[0]));
static_assert(generic_geometry_is_valid(kPreregisteredFounderHashes[1]));
static_assert(generic_geometry_is_valid(kPreregisteredFounderHashes[2]));
static_assert(generic_geometry_is_valid(kHeldOutFounderHash));
static_assert(valid_processive_weight_region_hash(kAuthoredPrefixHash));
static_assert(processive_weight_length(kAuthoredPrefixHash) == 3u);

}  // namespace substrate::bcc32::endogenous_addition_recruitment
