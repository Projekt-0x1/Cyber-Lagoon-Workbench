#pragma once

// Compact autonomous clock germ.  Five sites of ordinary BCC matter return
// exactly under F every 23 ticks; no host counter, pulse injection, or phase
// schedule exists after genesis.  The hash selects only payload and collars.

#include <array>
#include <cstddef>
#include <cstdint>

#include "bcc32_developmental_seed.cuh"
#include "bcc32_law.cuh"

namespace substrate::bcc32 {

using Period23ClockSeedHash = std::uint8_t;

inline constexpr std::uint32_t kPeriod23ClockBondCollarShift = 0u;
inline constexpr std::uint32_t kPeriod23ClockFaceCollarShift = 1u;
inline constexpr std::uint32_t kPeriod23ClockPayloadShift = 2u;
inline constexpr std::uint32_t kPeriod23ClockPayloadBasisShift = 3u;
inline constexpr std::size_t kPeriod23ClockSeedSiteCount = 5u;

inline constexpr Period23ClockSeedHash kPeriod23ClockSeedHash = 0x07u;

constexpr bool period23_clock_bit(Period23ClockSeedHash hash, std::uint32_t shift) {
  return ((hash >> shift) & Period23ClockSeedHash{1u}) != 0u;
}

constexpr std::uint32_t period23_clock_payload_basis(Period23ClockSeedHash hash) {
  return (hash >> kPeriod23ClockPayloadBasisShift) & 0x3u;
}

constexpr Period23ClockSeedHash make_period23_clock_seed_hash(bool bond_collar, bool face_collar,
                                                                bool payload, std::uint32_t payload_basis) {
  return static_cast<Period23ClockSeedHash>((bond_collar ? 1u : 0u) |
                                             ((face_collar ? 1u : 0u) << 1u) |
                                             ((payload ? 1u : 0u) << 2u) |
                                             ((payload_basis & 0x3u) << 3u));
}

constexpr std::array<DevelopmentalSeedSite, kPeriod23ClockSeedSiteCount> period23_clock_seed(
    Period23ClockSeedHash hash) {
  const SiteWord payload = period23_clock_bit(hash, kPeriod23ClockPayloadShift)
                               ? energy_bit(period23_clock_payload_basis(hash))
                               : 0u;
  return {{{-1, -1, -1, static_cast<SiteWord>(kQuiescentWord | channel_bit(kReactiveShift, 0u))},
           {-1, 0, 0, static_cast<SiteWord>(kQuiescentWord |
                                             (period23_clock_bit(hash, kPeriod23ClockBondCollarShift)
                                                  ? owned_bond_bit(0u)
                                                  : 0u))},
           {0, 0, 0, static_cast<SiteWord>(kQuiescentWord | owned_bond_bit(0u) |
                                            owned_bond_bit(1u) | owned_bond_bit(2u) | payload)},
           {1, 0, 0, static_cast<SiteWord>(kQuiescentWord | channel_bit(kReactiveShift, 0u) |
                                            face_bit(1u))},
           {2, 0, 0, static_cast<SiteWord>(kQuiescentWord |
                                             (period23_clock_bit(hash, kPeriod23ClockFaceCollarShift)
                                                  ? face_bit(4u)
                                                  : 0u))}}};
}

}  // namespace substrate::bcc32
