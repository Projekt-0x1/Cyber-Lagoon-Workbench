#pragma once

// Co-grown timing body plus one ordinary B/E output-port founder.  Unlike a
// receiver docked onto an adult clock, this compact hash births the sixth site
// as part of the periodic body and lets F determine whether it matures into a
// payload-causal C/R phase port.

#include <array>
#include <cstdint>

#include "bcc32_local_founder_hash.cuh"
#include "bcc32_period23_clock_seed.cuh"

namespace substrate::bcc32 {

using Period23OutputPortHash = std::uint16_t;
constexpr std::uint32_t kPeriod23OutputPortAnchorShift = 0u;
constexpr std::uint32_t kPeriod23OutputPortDirectionShift = 3u;
constexpr std::uint32_t kPeriod23OutputPortBasisShift = 6u;
constexpr std::uint32_t kPeriod23OutputPortGeneShift = 8u;
constexpr Period23OutputPortHash kPeriod23OutputPortHashMask = 0x03ffu;
constexpr std::size_t kPeriod23OutputPortSeedSiteCount = kPeriod23ClockSeedSiteCount + 1u;

constexpr Period23OutputPortHash make_period23_output_port_hash(std::uint32_t anchor,
                                                                 std::uint32_t direction,
                                                                 std::uint32_t basis,
                                                                 LocalFounderGene gene) {
  return static_cast<Period23OutputPortHash>(
      ((anchor & 0x07u) << kPeriod23OutputPortAnchorShift) |
      ((direction & 0x07u) << kPeriod23OutputPortDirectionShift) |
      ((basis & 0x03u) << kPeriod23OutputPortBasisShift) |
      ((static_cast<std::uint32_t>(gene) & 0x03u) << kPeriod23OutputPortGeneShift));
}

constexpr std::uint32_t period23_output_port_anchor(Period23OutputPortHash hash) {
  return (hash >> kPeriod23OutputPortAnchorShift) & 0x07u;
}
constexpr std::uint32_t period23_output_port_direction(Period23OutputPortHash hash) {
  return (hash >> kPeriod23OutputPortDirectionShift) & 0x07u;
}
constexpr std::uint32_t period23_output_port_basis(Period23OutputPortHash hash) {
  return (hash >> kPeriod23OutputPortBasisShift) & 0x03u;
}
constexpr LocalFounderGene period23_output_port_gene(Period23OutputPortHash hash) {
  return static_cast<LocalFounderGene>((hash >> kPeriod23OutputPortGeneShift) & 0x03u);
}

constexpr std::array<DevelopmentalSeedSite, kPeriod23OutputPortSeedSiteCount>
period23_output_port_seed(Period23OutputPortHash hash, Period23ClockSeedHash clock_hash) {
  const auto clock = period23_clock_seed(clock_hash);
  const DevelopmentalSeedSite anchor =
      clock[period23_output_port_anchor(hash) % kPeriod23ClockSeedSiteCount];
  const Int3 offset =
      direction_offset(static_cast<Direction>(period23_output_port_direction(hash)));
  return {{clock[0], clock[1], clock[2], clock[3], clock[4],
           {static_cast<std::int8_t>(anchor.x + offset.x),
            static_cast<std::int8_t>(anchor.y + offset.y),
            static_cast<std::int8_t>(anchor.z + offset.z),
            local_founder_word(period23_output_port_gene(hash), period23_output_port_basis(hash))}}};
}

}  // namespace substrate::bcc32
