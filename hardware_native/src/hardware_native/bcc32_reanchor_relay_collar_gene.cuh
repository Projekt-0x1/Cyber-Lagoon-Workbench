#pragma once

// A two-bit developmental collar for the first construction-area seed.
//
// Each bit reuses one already-present B0/E0 reanchor parent as the incoming
// half of the measured two-hop relay, and grows one B0 scaffold immediately
// one +x step away.  This is a compact matter grammar: the mask selects a
// local cell type and placement; it does not name an edge, output, or runtime
// action.  The unchanged reversible law F decides whether the collar becomes
// an actual relay in the grown tissue.

#include <array>
#include <cstdint>

#include "bcc32_reanchor_pair_gene.cuh"

namespace substrate::bcc32 {

using ReanchorRelayCollarHash = std::uint8_t;
constexpr ReanchorRelayCollarHash kReanchorRelayFirstCollarHash = 0x01u;
constexpr ReanchorRelayCollarHash kReanchorRelaySecondCollarHash = 0x02u;
constexpr ReanchorRelayCollarHash kReanchorRelayCollarMask = 0x03u;

[[nodiscard]] constexpr bool reanchor_relay_collar_hash_valid(
    ReanchorRelayCollarHash hash) {
  return hash == kReanchorRelayFirstCollarHash ||
         hash == kReanchorRelaySecondCollarHash;
}

[[nodiscard]] constexpr std::array<DevelopmentalSeedSite, 5u>
reanchor_relay_collar_seed(ReanchorRelayCollarHash hash) {
  const auto base = reanchor_pair_seed(kFirstConstructionAreaHash);
  const DevelopmentalSeedSite parent =
      hash == kReanchorRelayFirstCollarHash ? base[0] : base[2];
  return {{base[0], base[1], base[2], base[3],
           {parent.x + 1, parent.y, parent.z, kQ | owned_bond_bit(0u)}}};
}

}  // namespace substrate::bcc32
