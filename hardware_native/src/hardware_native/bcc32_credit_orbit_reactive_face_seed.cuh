#pragma once

// A compact, interpreted founder seed for the first C2 wake receiver grammar.
// The parent supplies the measured C2 credit orbit.  The three low hash bits choose
// one physical neighbour of its R3 wake site; birth places exactly one X-3 controller
// there.  The generic law, not this decoder, decides whether that controller ever
// converts R3 into B3.

#include <array>
#include <cstdint>

#include "bcc32_credit_orbit_seed.cuh"

namespace substrate::bcc32 {

using CreditOrbitReactiveFaceSeedHash = std::uint8_t;
constexpr std::size_t kCreditOrbitReactiveFaceSeedSiteCount = kCreditOrbitSeedSiteCount + 1u;
constexpr std::uint32_t kReactiveWakeDirection = 0u;
constexpr std::uint32_t kReactiveFaceDirection = 7u;  // X-3 in the edge grammar.

constexpr CreditOrbitReactiveFaceSeedHash make_credit_orbit_reactive_face_seed_hash(
    std::uint32_t controller_direction) {
  return static_cast<CreditOrbitReactiveFaceSeedHash>(controller_direction & 0x07u);
}

constexpr std::uint32_t credit_orbit_reactive_face_controller_direction(
    CreditOrbitReactiveFaceSeedHash hash) {
  return static_cast<std::uint32_t>(hash & 0x07u);
}

inline Z3Coordinate credit_orbit_reactive_face_output() {
  const DevelopmentalSeedSite source = credit_orbit_seed(kCreditOrbitSeedHash)[kCreditBudReceiverSeedSiteCount];
  return {source.x, source.y, source.z};
}

inline Z3Coordinate credit_orbit_reactive_face_target() {
  const Z3Coordinate output = credit_orbit_reactive_face_output();
  const Int3 step = direction_offset(static_cast<Direction>(kReactiveWakeDirection));
  return {output.x + 2 * step.x, output.y + 2 * step.y, output.z + 2 * step.z};
}

inline Z3Coordinate credit_orbit_reactive_face_controller(CreditOrbitReactiveFaceSeedHash hash) {
  const Z3Coordinate target = credit_orbit_reactive_face_target();
  const Int3 step = direction_offset(
      static_cast<Direction>(credit_orbit_reactive_face_controller_direction(hash)));
  return {target.x + step.x, target.y + step.y, target.z + step.z};
}

inline constexpr SiteWord kCreditOrbitReactiveFaceControllerWord =
    kQ | face_bit(kReactiveFaceDirection);

inline std::array<DevelopmentalSeedSite, kCreditOrbitReactiveFaceSeedSiteCount>
credit_orbit_reactive_face_seed(CreditOrbitReactiveFaceSeedHash hash) {
  std::array<DevelopmentalSeedSite, kCreditOrbitReactiveFaceSeedSiteCount> result{};
  const auto parent = credit_orbit_seed(kCreditOrbitSeedHash);
  for (std::size_t index = 0u; index < parent.size(); ++index) result[index] = parent[index];
  const Z3Coordinate controller = credit_orbit_reactive_face_controller(hash);
  result[parent.size()] = {static_cast<std::int8_t>(controller.x),
                           static_cast<std::int8_t>(controller.y),
                           static_cast<std::int8_t>(controller.z),
                           kCreditOrbitReactiveFaceControllerWord};
  return result;
}

}  // namespace substrate::bcc32
