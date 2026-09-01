#pragma once

// Smallest co-grown bridge grammar after the one-founder port closure: one
// generic B/E port grows with the F23 body and one same-basis B/E helper sits
// at a distinct local neighbour of that port.  The hash never encodes a phase
// schedule or an output value; F must either organize a causal C/R phase road
// or the grammar is rejected.

#include <array>
#include <cstdint>

#include "bcc32_local_founder_hash.cuh"
#include "bcc32_period23_clock_seed.cuh"

namespace substrate::bcc32 {

using Period23TwoFounderBridgeHash = std::uint16_t;
constexpr std::uint32_t kPeriod23BridgeAnchorShift = 0u;
constexpr std::uint32_t kPeriod23BridgePortDirectionShift = 3u;
constexpr std::uint32_t kPeriod23BridgeBasisShift = 6u;
constexpr std::uint32_t kPeriod23BridgePortGeneShift = 8u;
constexpr std::uint32_t kPeriod23BridgeHelperDirectionShift = 10u;
constexpr std::uint32_t kPeriod23BridgeHelperGeneShift = 13u;
constexpr Period23TwoFounderBridgeHash kPeriod23TwoFounderBridgeHashMask = 0x7fffu;
constexpr std::size_t kPeriod23TwoFounderBridgeSeedSiteCount = kPeriod23ClockSeedSiteCount + 2u;

constexpr Period23TwoFounderBridgeHash make_period23_twofounder_bridge_hash(
    std::uint32_t anchor, std::uint32_t port_direction, std::uint32_t basis,
    LocalFounderGene port_gene, std::uint32_t helper_direction, LocalFounderGene helper_gene) {
  return static_cast<Period23TwoFounderBridgeHash>(
      ((anchor & 0x07u) << kPeriod23BridgeAnchorShift) |
      ((port_direction & 0x07u) << kPeriod23BridgePortDirectionShift) |
      ((basis & 0x03u) << kPeriod23BridgeBasisShift) |
      ((static_cast<std::uint32_t>(port_gene) & 0x03u) << kPeriod23BridgePortGeneShift) |
      ((helper_direction & 0x07u) << kPeriod23BridgeHelperDirectionShift) |
      ((static_cast<std::uint32_t>(helper_gene) & 0x03u) << kPeriod23BridgeHelperGeneShift));
}

constexpr std::uint32_t period23_bridge_anchor(Period23TwoFounderBridgeHash hash) {
  return (hash >> kPeriod23BridgeAnchorShift) & 0x07u;
}
constexpr std::uint32_t period23_bridge_port_direction(Period23TwoFounderBridgeHash hash) {
  return (hash >> kPeriod23BridgePortDirectionShift) & 0x07u;
}
constexpr std::uint32_t period23_bridge_basis(Period23TwoFounderBridgeHash hash) {
  return (hash >> kPeriod23BridgeBasisShift) & 0x03u;
}
constexpr LocalFounderGene period23_bridge_port_gene(Period23TwoFounderBridgeHash hash) {
  return static_cast<LocalFounderGene>((hash >> kPeriod23BridgePortGeneShift) & 0x03u);
}
constexpr std::uint32_t period23_bridge_helper_direction(Period23TwoFounderBridgeHash hash) {
  return (hash >> kPeriod23BridgeHelperDirectionShift) & 0x07u;
}
constexpr LocalFounderGene period23_bridge_helper_gene(Period23TwoFounderBridgeHash hash) {
  return static_cast<LocalFounderGene>((hash >> kPeriod23BridgeHelperGeneShift) & 0x03u);
}

constexpr std::array<DevelopmentalSeedSite, kPeriod23TwoFounderBridgeSeedSiteCount>
period23_twofounder_bridge_seed(Period23TwoFounderBridgeHash hash, Period23ClockSeedHash clock_hash,
                                bool include_helper = true) {
  const auto clock = period23_clock_seed(clock_hash);
  const DevelopmentalSeedSite anchor =
      clock[period23_bridge_anchor(hash) % kPeriod23ClockSeedSiteCount];
  const Int3 port_offset =
      direction_offset(static_cast<Direction>(period23_bridge_port_direction(hash)));
  const std::int8_t port_x = static_cast<std::int8_t>(anchor.x + port_offset.x);
  const std::int8_t port_y = static_cast<std::int8_t>(anchor.y + port_offset.y);
  const std::int8_t port_z = static_cast<std::int8_t>(anchor.z + port_offset.z);
  const Int3 helper_offset =
      direction_offset(static_cast<Direction>(period23_bridge_helper_direction(hash)));
  const SiteWord helper = include_helper
                              ? local_founder_word(period23_bridge_helper_gene(hash),
                                                   period23_bridge_basis(hash))
                              : kQ;
  return {{clock[0], clock[1], clock[2], clock[3], clock[4],
           {port_x, port_y, port_z,
            local_founder_word(period23_bridge_port_gene(hash), period23_bridge_basis(hash))},
           {static_cast<std::int8_t>(port_x + helper_offset.x),
            static_cast<std::int8_t>(port_y + helper_offset.y),
            static_cast<std::int8_t>(port_z + helper_offset.z), helper}}};
}

}  // namespace substrate::bcc32
