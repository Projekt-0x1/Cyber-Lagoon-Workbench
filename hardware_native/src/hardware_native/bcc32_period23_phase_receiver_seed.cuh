#pragma once

// A compact receiver candidate for a clock's unused B1/B2 hub roads.
//
// The clock hub already provides an owned B1 or B2 source lane.  The only
// phase-relevant destination palette on that edge is its local C/R molecular
// state, because gate 2 transposes source E with destination C under source B.
// This hash therefore names exactly a route basis and C/R receptor state, not
// a target action, clock tick, or host-side decoder.  F decides whether the
// receptor becomes a recurring, selective phase route.

#include <array>
#include <cstddef>
#include <cstdint>

#include "bcc32_developmental_seed.cuh"
#include "bcc32_law.cuh"
#include "bcc32_period23_clock_seed.cuh"

namespace substrate::bcc32 {

using Period23PhaseReceiverSeedHash = std::uint8_t;

inline constexpr std::uint32_t kPeriod23PhaseReceiverClockShift = 0u;
inline constexpr std::uint32_t kPeriod23PhaseReceiverBasisShift = 4u;
inline constexpr std::uint32_t kPeriod23PhaseReceiverCShift = 5u;
inline constexpr std::uint32_t kPeriod23PhaseReceiverRShift = 6u;
inline constexpr std::size_t kPeriod23PhaseReceiverSeedSiteCount = 6u;

constexpr Period23ClockSeedHash period23_phase_receiver_clock_hash(
    Period23PhaseReceiverSeedHash hash) {
  return static_cast<Period23ClockSeedHash>((hash >> kPeriod23PhaseReceiverClockShift) & 0x0fu);
}

constexpr std::uint32_t period23_phase_receiver_basis(Period23PhaseReceiverSeedHash hash) {
  return 1u + ((hash >> kPeriod23PhaseReceiverBasisShift) & 0x01u);
}

constexpr bool period23_phase_receiver_c(Period23PhaseReceiverSeedHash hash) {
  return ((hash >> kPeriod23PhaseReceiverCShift) & 0x01u) != 0u;
}

constexpr bool period23_phase_receiver_r(Period23PhaseReceiverSeedHash hash) {
  return ((hash >> kPeriod23PhaseReceiverRShift) & 0x01u) != 0u;
}

constexpr Period23PhaseReceiverSeedHash make_period23_phase_receiver_seed_hash(
    Period23ClockSeedHash clock_hash, std::uint32_t route_basis, bool c, bool r) {
  return static_cast<Period23PhaseReceiverSeedHash>(
      (clock_hash & 0x0fu) | (((route_basis - 1u) & 0x01u) << kPeriod23PhaseReceiverBasisShift) |
      ((c ? 1u : 0u) << kPeriod23PhaseReceiverCShift) |
      ((r ? 1u : 0u) << kPeriod23PhaseReceiverRShift));
}

constexpr DevelopmentalSeedSite period23_phase_receiver_site(
    Period23PhaseReceiverSeedHash hash) {
  const std::uint32_t basis = period23_phase_receiver_basis(hash);
  SiteWord word = kQuiescentWord;
  if (period23_phase_receiver_c(hash)) word |= channel_bit(kConformationShift, basis);
  if (period23_phase_receiver_r(hash)) word |= channel_bit(kReactiveShift, basis);
  switch (basis) {
    case 1u: return {0, 1, 0, word};
    case 2u: return {0, 0, 1, word};
  }
  return {0, 0, 0, kQuiescentWord};
}

constexpr std::array<DevelopmentalSeedSite, kPeriod23PhaseReceiverSeedSiteCount>
period23_phase_receiver_seed(Period23PhaseReceiverSeedHash hash) {
  const auto clock = period23_clock_seed(period23_phase_receiver_clock_hash(hash));
  return {{clock[0], clock[1], clock[2], clock[3], clock[4], period23_phase_receiver_site(hash)}};
}

}  // namespace substrate::bcc32
