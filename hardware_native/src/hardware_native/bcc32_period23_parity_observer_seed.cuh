#pragma once

// Compact developmental grammar for testing whether the measured two-cell
// native parity material can read the autonomous F23 clock without becoming a
// host-scheduled observer.  The word selects only one clock founder and one
// BCC-neighbour offset; the parity cells themselves are the already measured
// local C/R material.

#include <array>
#include <cstdint>

#include "bcc32_local_founder_hash.cuh"
#include "bcc32_period23_clock_seed.cuh"

namespace substrate::bcc32 {

using Period23ParityObserverHash = std::uint8_t;
constexpr std::uint32_t kPeriod23ParityObserverAnchorShift = 0u;
constexpr std::uint32_t kPeriod23ParityObserverDirectionShift = 3u;
constexpr Period23ParityObserverHash kPeriod23ParityObserverHashMask = 0x3fu;
constexpr std::size_t kPeriod23ParityObserverSeedSiteCount = kPeriod23ClockSeedSiteCount + 2u;

constexpr Period23ParityObserverHash make_period23_parity_observer_hash(
    std::uint32_t anchor, std::uint32_t direction) {
  return static_cast<Period23ParityObserverHash>(
      ((anchor & 0x07u) << kPeriod23ParityObserverAnchorShift) |
      ((direction & 0x07u) << kPeriod23ParityObserverDirectionShift));
}

constexpr std::uint32_t period23_parity_observer_anchor(Period23ParityObserverHash hash) {
  return (hash >> kPeriod23ParityObserverAnchorShift) & 0x07u;
}

constexpr std::uint32_t period23_parity_observer_direction(Period23ParityObserverHash hash) {
  return (hash >> kPeriod23ParityObserverDirectionShift) & 0x07u;
}

constexpr std::array<DevelopmentalSeedSite, kPeriod23ParityObserverSeedSiteCount>
period23_parity_observer_seed(Period23ParityObserverHash hash, Period23ClockSeedHash clock_hash) {
  const auto clock = period23_clock_seed(clock_hash);
  const auto parity = local_parity_seed(kNativeParityCellHash);
  const DevelopmentalSeedSite anchor =
      clock[period23_parity_observer_anchor(hash) % kPeriod23ClockSeedSiteCount];
  const Int3 offset = direction_offset(
      static_cast<Direction>(period23_parity_observer_direction(hash)));
  const std::int8_t origin_x = static_cast<std::int8_t>(anchor.x + offset.x);
  const std::int8_t origin_y = static_cast<std::int8_t>(anchor.y + offset.y);
  const std::int8_t origin_z = static_cast<std::int8_t>(anchor.z + offset.z);
  return {{clock[0], clock[1], clock[2], clock[3], clock[4],
           {static_cast<std::int8_t>(origin_x + parity[0].x),
            static_cast<std::int8_t>(origin_y + parity[0].y),
            static_cast<std::int8_t>(origin_z + parity[0].z), parity[0].word},
           {static_cast<std::int8_t>(origin_x + parity[1].x),
            static_cast<std::int8_t>(origin_y + parity[1].y),
            static_cast<std::int8_t>(origin_z + parity[1].z), parity[1].word}}};
}

}  // namespace substrate::bcc32
