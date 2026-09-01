#pragma once

// Three-cell delayed-rearm candidate for the period-23 clock.
//
// A C1 receiver one edge from the hub consumes the hub's B1-controlled E/C
// exchange.  Instead of putting replacement energy on the hub (which closes
// the clock), the receiver carries B1 and a second C1 receptor one edge
// downstream.  Gate 2 can therefore move a received E1 on a later local edge.
// The hash selects only that B1/C1/C1 matter; F decides whether it closes a
// recurrent selective route.

#include <array>
#include <cstddef>
#include <cstdint>

#include "bcc32_developmental_seed.cuh"
#include "bcc32_law.cuh"
#include "bcc32_period23_clock_seed.cuh"

namespace substrate::bcc32 {

using Period23PhaseRearmSeedHash = std::uint8_t;
inline constexpr std::uint32_t kPeriod23PhaseRearmClockShift = 0u;
inline constexpr std::uint32_t kPeriod23PhaseRearmReceiverBondShift = 4u;
inline constexpr std::uint32_t kPeriod23PhaseRearmReceiverCShift = 5u;
inline constexpr std::uint32_t kPeriod23PhaseRearmBufferCShift = 6u;
inline constexpr std::size_t kPeriod23PhaseRearmSeedSiteCount = 7u;

constexpr bool period23_phase_rearm_bit(Period23PhaseRearmSeedHash hash, std::uint32_t shift) {
  return ((hash >> shift) & 0x01u) != 0u;
}

constexpr Period23ClockSeedHash period23_phase_rearm_clock_hash(Period23PhaseRearmSeedHash hash) {
  return static_cast<Period23ClockSeedHash>((hash >> kPeriod23PhaseRearmClockShift) & 0x0fu);
}

constexpr Period23PhaseRearmSeedHash make_period23_phase_rearm_seed_hash(
    Period23ClockSeedHash clock_hash, bool receiver_bond, bool receiver_c, bool buffer_c) {
  return static_cast<Period23PhaseRearmSeedHash>(
      (clock_hash & 0x0fu) | ((receiver_bond ? 1u : 0u) << kPeriod23PhaseRearmReceiverBondShift) |
      ((receiver_c ? 1u : 0u) << kPeriod23PhaseRearmReceiverCShift) |
      ((buffer_c ? 1u : 0u) << kPeriod23PhaseRearmBufferCShift));
}

constexpr std::array<DevelopmentalSeedSite, kPeriod23PhaseRearmSeedSiteCount>
period23_phase_rearm_seed(Period23PhaseRearmSeedHash hash) {
  const auto clock = period23_clock_seed(period23_phase_rearm_clock_hash(hash));
  SiteWord receiver = kQuiescentWord;
  if (period23_phase_rearm_bit(hash, kPeriod23PhaseRearmReceiverBondShift)) receiver |= owned_bond_bit(1u);
  if (period23_phase_rearm_bit(hash, kPeriod23PhaseRearmReceiverCShift)) receiver |= channel_bit(kConformationShift, 1u);
  SiteWord buffer = kQuiescentWord;
  if (period23_phase_rearm_bit(hash, kPeriod23PhaseRearmBufferCShift)) buffer |= channel_bit(kConformationShift, 1u);
  return {{clock[0], clock[1], clock[2], clock[3], clock[4], {0, 1, 0, receiver},
           {0, 2, 0, buffer}}};
}

}  // namespace substrate::bcc32
