#pragma once

// Nonabsorbing local observer grammar for the autonomous period-23 clock.
//
// The receiver is not a schedule or an externally read clock.  It is one
// already-proved strict-fixed saturated dimer.  The compact hash chooses a
// clock anchor, one BCC-neighbour direction, and one dimer lane.  Full F must
// decide both whether the complete composite retains the clock's F23 return
// and whether material phase reaches the dimer.  A survivor is therefore a
// candidate local region-to-region interface, not a host observation.

#include <array>
#include <cstdint>

#include "bcc32_credit_reflector_seed.cuh"
#include "bcc32_geometry.cuh"
#include "bcc32_period23_clock_seed.cuh"

namespace substrate::bcc32 {

using Period23ReflectorObserverGeneHash = std::uint8_t;

inline constexpr std::uint32_t kPeriod23ReflectorAnchorMask = 0x07u;
inline constexpr std::uint32_t kPeriod23ReflectorDirectionShift = 3u;
inline constexpr std::uint32_t kPeriod23ReflectorLaneShift = 6u;

constexpr Period23ReflectorObserverGeneHash make_period23_reflector_observer_gene(
    std::uint32_t anchor, std::uint32_t direction, std::uint32_t lane) {
  return static_cast<Period23ReflectorObserverGeneHash>(
      (anchor & kPeriod23ReflectorAnchorMask) |
      ((direction & 0x07u) << kPeriod23ReflectorDirectionShift) |
      ((lane & 0x03u) << kPeriod23ReflectorLaneShift));
}

constexpr std::uint32_t period23_reflector_observer_anchor(Period23ReflectorObserverGeneHash hash) {
  return hash & kPeriod23ReflectorAnchorMask;
}

constexpr std::uint32_t period23_reflector_observer_direction(Period23ReflectorObserverGeneHash hash) {
  return (hash >> kPeriod23ReflectorDirectionShift) & 0x07u;
}

constexpr CreditReflectorGeneHash period23_reflector_observer_lane(
    Period23ReflectorObserverGeneHash hash) {
  return static_cast<CreditReflectorGeneHash>((hash >> kPeriod23ReflectorLaneShift) & 0x03u);
}

inline Z3Coordinate period23_reflector_observer_symbol(Period23ReflectorObserverGeneHash hash) {
  const auto clock = period23_clock_seed(kPeriod23ClockSeedHash);
  const DevelopmentalSeedSite anchor = clock[period23_reflector_observer_anchor(hash)];
  const Int3 delta = direction_offset(
      static_cast<Direction>(period23_reflector_observer_direction(hash)));
  return {static_cast<std::int32_t>(anchor.x) + delta.x,
          static_cast<std::int32_t>(anchor.y) + delta.y,
          static_cast<std::int32_t>(anchor.z) + delta.z};
}

inline std::array<DevelopmentalSeedSite, 2> period23_reflector_observer_seed(
    Period23ReflectorObserverGeneHash hash) {
  return credit_reflector_seed(period23_reflector_observer_lane(hash),
                               period23_reflector_observer_symbol(hash));
}

}  // namespace substrate::bcc32
