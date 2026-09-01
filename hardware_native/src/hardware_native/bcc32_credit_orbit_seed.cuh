#pragma once

// A compact, physical candidate for the next missing link in plasticity:
// the C2-dependent carrier-3 difference already emitted by the local credit
// receiver is aligned with the upstream vacancy port of a two-state orbit.
//
// This is deliberately a *candidate seed*, not a new law.  The parent is the
// measured credit-bud receiver.  The only new matter is the existing balanced
// two-site orbit, rotated so its input is lane 3 and translated so that its
// input port is the measured first R3-dependent export (-4,-4,-4).  F decides
// whether that carrier difference can write the orbit.  No host transfer,
// timing program, answer, or update instruction appears in the hash.

#include <array>
#include <cstddef>
#include <cstdint>

#include "bcc32_credit_bud_receiver_seed.cuh"
#include "bcc32_reference.hpp"

namespace substrate::bcc32 {

using CreditOrbitSeedHash = unsigned __int128;

inline constexpr std::uint32_t kCreditOrbitParentShift = 0u;
inline constexpr std::uint32_t kCreditOrbitXShift = 64u;
inline constexpr std::uint32_t kCreditOrbitYShift = 70u;
inline constexpr std::uint32_t kCreditOrbitZShift = 76u;
inline constexpr std::uint32_t kCreditOrbitCoordinateMask = 0x3fu;
inline constexpr std::int32_t kCreditOrbitCoordinateBias = 32;

constexpr CreditOrbitSeedHash encode_credit_orbit_coordinate(std::int32_t value) {
  return (static_cast<CreditOrbitSeedHash>(value + kCreditOrbitCoordinateBias) &
          kCreditOrbitCoordinateMask);
}

constexpr std::int32_t decode_credit_orbit_coordinate(CreditOrbitSeedHash hash,
                                                       std::uint32_t shift) {
  return static_cast<std::int32_t>((hash >> shift) & kCreditOrbitCoordinateMask) -
         kCreditOrbitCoordinateBias;
}

constexpr CreditOrbitSeedHash make_credit_orbit_seed_hash(CreditBudReceiverSeedHash parent,
                                                           std::int32_t orbit_x,
                                                           std::int32_t orbit_y,
                                                           std::int32_t orbit_z) {
  return (static_cast<CreditOrbitSeedHash>(parent) << kCreditOrbitParentShift) |
         (encode_credit_orbit_coordinate(orbit_x) << kCreditOrbitXShift) |
         (encode_credit_orbit_coordinate(orbit_y) << kCreditOrbitYShift) |
         (encode_credit_orbit_coordinate(orbit_z) << kCreditOrbitZShift);
}

constexpr CreditBudReceiverSeedHash credit_orbit_parent_hash(CreditOrbitSeedHash hash) {
  return static_cast<CreditBudReceiverSeedHash>(hash);
}

constexpr std::array<std::int32_t, 3> credit_orbit_origin(CreditOrbitSeedHash hash) {
  return {{decode_credit_orbit_coordinate(hash, kCreditOrbitXShift),
           decode_credit_orbit_coordinate(hash, kCreditOrbitYShift),
           decode_credit_orbit_coordinate(hash, kCreditOrbitZShift)}};
}

// This S4 element maps the old lane 0 input to lane 3.  It is not a bespoke
// "credit" opcode: word and coordinate symmetry are both native BCC symmetry.
inline constexpr BasisPermutation kCreditOrbitLane0ToLane3{3u, 1u, 2u, 0u};

inline constexpr std::size_t kCreditOrbitSeedSiteCount =
    kCreditBudReceiverSeedSiteCount + 2u;

inline std::array<DevelopmentalSeedSite, kCreditOrbitSeedSiteCount> credit_orbit_seed(
    CreditOrbitSeedHash hash) {
  std::array<DevelopmentalSeedSite, kCreditOrbitSeedSiteCount> result{};
  const auto parent = credit_bud_receiver_seed(credit_orbit_parent_hash(hash));
  for (std::size_t index = 0u; index < parent.size(); ++index) result[index] = parent[index];

  const auto origin = credit_orbit_origin(hash);
  const Z3Coordinate source{origin[0], origin[1], origin[2]};
  const Int3 old_lane_zero = direction_offset(static_cast<Direction>(0u));
  const Z3Coordinate destination_relative = transformed_coordinate(
      Z3Coordinate{old_lane_zero.x, old_lane_zero.y, old_lane_zero.z}, kCreditOrbitLane0ToLane3);
  const Z3Coordinate destination{source.x + destination_relative.x,
                                 source.y + destination_relative.y,
                                 source.z + destination_relative.z};
  result[parent.size()] = {static_cast<std::int8_t>(source.x), static_cast<std::int8_t>(source.y),
                           static_cast<std::int8_t>(source.z),
                           transformed_word(0x101001feu, kCreditOrbitLane0ToLane3)};
  result[parent.size() + 1u] = {
      static_cast<std::int8_t>(destination.x), static_cast<std::int8_t>(destination.y),
      static_cast<std::int8_t>(destination.z),
      transformed_word(0x000000efu, kCreditOrbitLane0ToLane3)};
  return result;
}

inline constexpr CreditOrbitSeedHash kCreditOrbitSeedHash =
    make_credit_orbit_seed_hash(kCreditBudReceiverLocalRNoBHash, -7, -7, -7);

static_assert(credit_orbit_parent_hash(kCreditOrbitSeedHash) == kCreditBudReceiverLocalRNoBHash);
static_assert(credit_orbit_origin(kCreditOrbitSeedHash) == std::array<std::int32_t, 3>{{-7, -7, -7}});

}  // namespace substrate::bcc32
