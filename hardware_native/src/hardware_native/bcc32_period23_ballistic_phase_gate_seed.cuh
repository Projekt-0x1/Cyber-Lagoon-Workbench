#pragma once

// Ballistic vacancy phase-gate candidate.
//
// A lane-5 carrier vacancy travels one BCC cell per F step.  At its physical
// arrival it makes the destination P-1 absent while the clock hub still owns
// P+1.  Gate 12 can then transpose the passive receiver's R1/C1 state.  The
// delay is geometry (the initial vacancy distance), never a host counter.

#include <array>
#include <cstddef>
#include <cstdint>

#include "bcc32_developmental_seed.cuh"
#include "bcc32_law.cuh"
#include "bcc32_period23_clock_seed.cuh"

namespace substrate::bcc32 {

using Period23BallisticPhaseGateHash = std::uint8_t;
inline constexpr std::size_t kPeriod23BallisticPhaseGateSeedSiteCount = 7u;
inline constexpr std::uint32_t kPeriod23BallisticDelay = 12u;

constexpr std::array<DevelopmentalSeedSite, kPeriod23BallisticPhaseGateSeedSiteCount>
period23_ballistic_phase_gate_seed(Period23BallisticPhaseGateHash clock_hash,
                                   std::uint32_t delay = kPeriod23BallisticDelay) {
  const auto clock = period23_clock_seed(clock_hash);
  // Carrier lane 5 is -u1, so the vacancy launched +u1*delay from the receiver
  // reaches it after exactly `delay` complete F applications.
  return {{clock[0], clock[1], clock[2], clock[3], clock[4],
           {0, 1, 0, static_cast<SiteWord>(kQuiescentWord | channel_bit(kReactiveShift, 1u))},
           {0, static_cast<std::int8_t>(1u + delay), 0,
            static_cast<SiteWord>(kQuiescentWord & ~carrier_bit(5u))}}};
}

}  // namespace substrate::bcc32
