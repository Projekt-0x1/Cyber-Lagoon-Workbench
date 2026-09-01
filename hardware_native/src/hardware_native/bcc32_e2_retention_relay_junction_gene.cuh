#pragma once

// The direct composition of the measured E2-retention hash 0x50 with the
// measured two-hop relay's required +x B0 scaffold.  The two cells remain
// matter, not a graph instruction: rotor B0+E1 at the origin; collar
// B0+R0+face-0 at +x.  Full F determines whether both functions coexist.

#include <array>

#include "bcc32_developmental_seed.cuh"
#include "bcc32_law.cuh"

namespace substrate::bcc32 {

using E2RetentionRelayJunctionHash = std::uint8_t;
constexpr E2RetentionRelayJunctionHash kE2RetentionRelayJunctionHash = 0x01u;

constexpr std::array<DevelopmentalSeedSite, 2u> e2_retention_relay_junction_seed(
    E2RetentionRelayJunctionHash hash) {
  const bool enabled = hash == kE2RetentionRelayJunctionHash;
  const SiteWord rotor = kQ | owned_bond_bit(0u) | energy_bit(1u);
  SiteWord collar = kQ | channel_bit(kReactiveShift, 0u) | face_bit(4u);
  if (enabled) collar |= owned_bond_bit(0u);
  return {{{0, 0, 0, rotor}, {1, 0, 0, collar}}};
}

}  // namespace substrate::bcc32
