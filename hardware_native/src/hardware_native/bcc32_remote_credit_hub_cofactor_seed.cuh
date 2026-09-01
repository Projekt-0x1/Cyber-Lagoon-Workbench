#pragma once

// Compact generic-matter overlay for the measured remote credit route.
//
// The hash names only an edge basis and whether generic bond/energy matter is
// present at the route's otherwise untouched R3 origin.  It does not encode a
// receiver state, an output rule, a clock, or an observer protocol.  After
// Genesis, ordinary F alone decides whether this local B/E cofactor can turn a
// routed carrier into a physical successor state.

#include <array>
#include <cstdint>

#include "bcc32_remote_credit_hole_transducer_seed.cuh"

namespace substrate::bcc32 {

using RemoteCreditHubCofactorSeedHash = std::uint8_t;
constexpr RemoteCreditHubCofactorSeedHash kRemoteCreditHubCofactorBasisMask = 0x03u;
constexpr RemoteCreditHubCofactorSeedHash kRemoteCreditHubCofactorBondBit = 0x04u;
constexpr RemoteCreditHubCofactorSeedHash kRemoteCreditHubCofactorEnergyBit = 0x08u;

constexpr RemoteCreditHubCofactorSeedHash make_remote_credit_hub_cofactor_seed_hash(
    std::uint32_t basis, bool bond, bool energy) {
  return static_cast<RemoteCreditHubCofactorSeedHash>(
      (basis & kRemoteCreditHubCofactorBasisMask) |
      (bond ? kRemoteCreditHubCofactorBondBit : 0u) |
      (energy ? kRemoteCreditHubCofactorEnergyBit : 0u));
}

constexpr std::uint32_t remote_credit_hub_cofactor_basis(
    RemoteCreditHubCofactorSeedHash hash) {
  return static_cast<std::uint32_t>(hash & kRemoteCreditHubCofactorBasisMask);
}

constexpr bool remote_credit_hub_cofactor_has_bond(RemoteCreditHubCofactorSeedHash hash) {
  return (hash & kRemoteCreditHubCofactorBondBit) != 0u;
}

constexpr bool remote_credit_hub_cofactor_has_energy(RemoteCreditHubCofactorSeedHash hash) {
  return (hash & kRemoteCreditHubCofactorEnergyBit) != 0u;
}

inline std::array<DevelopmentalSeedSite, kRemoteCreditHoleTransducerSeedSiteCount>
remote_credit_hub_cofactor_seed(CreditOrbitSeedHash parent,
                                RemoteCreditHoleTransducerSeedHash route_hash,
                                RemoteCreditHubCofactorSeedHash cofactor_hash,
                                bool receptor) {
  auto result = remote_credit_hole_transducer_seed(parent, route_hash);
  SiteWord hub = receptor ? static_cast<SiteWord>(kQ | channel_bit(kReactiveShift, 3u)) : kQ;
  const std::uint32_t basis = remote_credit_hub_cofactor_basis(cofactor_hash);
  if (remote_credit_hub_cofactor_has_bond(cofactor_hash)) hub |= owned_bond_bit(basis);
  if (remote_credit_hub_cofactor_has_energy(cofactor_hash)) hub |= energy_bit(basis);
  result.back().word = hub;
  return result;
}

}  // namespace substrate::bcc32
