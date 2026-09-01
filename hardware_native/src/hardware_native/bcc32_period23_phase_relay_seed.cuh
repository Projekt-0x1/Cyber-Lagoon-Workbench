#pragma once

// A compact, falsifiable phase-road candidate for the autonomous period-23
// germ.  The clock already owns B1/B2 at its hub; this hash supplies one
// matching E lane and asks whether an otherwise unused native BCC road can
// carry a phase-dependent state to its adjacent receptor.  It contains only
// local matter at birth.  F remains the sole growth and execution rule.

#include <array>
#include <cstddef>
#include <cstdint>

#include "bcc32_developmental_seed.cuh"
#include "bcc32_law.cuh"
#include "bcc32_period23_clock_seed.cuh"

namespace substrate::bcc32 {

using Period23PhaseRelaySeedHash = std::uint8_t;

inline constexpr std::uint32_t kPeriod23PhaseRelayClockShift = 0u;
inline constexpr std::uint32_t kPeriod23PhaseRelayBasisShift = 4u;
inline constexpr std::uint32_t kPeriod23PhaseRelayEnableShift = 5u;
inline constexpr std::size_t kPeriod23PhaseRelaySeedSiteCount = 5u;

constexpr Period23ClockSeedHash period23_phase_relay_clock_hash(
    Period23PhaseRelaySeedHash hash) {
  return static_cast<Period23ClockSeedHash>((hash >> kPeriod23PhaseRelayClockShift) & 0x0fu);
}

// The clock consumes basis 0 for its internal C0/R0 phase.  Its hub already
// carries B1 and B2, so this finite candidate grammar deliberately exposes
// only those two unused physical roads.  The contract determines whether
// either becomes a relay; it does not assume that one will.
constexpr std::uint32_t period23_phase_relay_basis(Period23PhaseRelaySeedHash hash) {
  return 1u + ((hash >> kPeriod23PhaseRelayBasisShift) & 0x01u);
}

constexpr bool period23_phase_relay_enabled(Period23PhaseRelaySeedHash hash) {
  return ((hash >> kPeriod23PhaseRelayEnableShift) & 0x01u) != 0u;
}

constexpr Period23PhaseRelaySeedHash make_period23_phase_relay_seed_hash(
    Period23ClockSeedHash clock_hash, std::uint32_t route_basis, bool enabled) {
  return static_cast<Period23PhaseRelaySeedHash>(
      (clock_hash & 0x0fu) | (((route_basis - 1u) & 0x01u) << kPeriod23PhaseRelayBasisShift) |
      ((enabled ? 1u : 0u) << kPeriod23PhaseRelayEnableShift));
}

constexpr std::array<DevelopmentalSeedSite, kPeriod23PhaseRelaySeedSiteCount>
period23_phase_relay_seed(Period23PhaseRelaySeedHash hash) {
  const auto clock = period23_clock_seed(period23_phase_relay_clock_hash(hash));
  std::array<DevelopmentalSeedSite, kPeriod23PhaseRelaySeedSiteCount> result = clock;
  const std::uint32_t route_basis = period23_phase_relay_basis(hash);
  if (period23_phase_relay_enabled(hash)) {
    result[2].word = static_cast<SiteWord>(result[2].word | energy_bit(route_basis));
  }
  return result;
}

}  // namespace substrate::bcc32
