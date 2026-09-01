#pragma once

// Minimal catalytic-readout material grammar.
//
// On one native BCC edge, enumerate the eight local roles under the canonical
// edge permutation with a quiescent carrier background (P+ and P- present).
// Requiring a face query to create a carrier difference while leaving the
// source B matter present leaves exactly two local states; the minimal state
// is P+,P-,B,C.  This hash grows only its non-background matter:
//
//   source (-u0): B0       -- the candidate durable weight / catalyst
//   receiver (0): C0       -- the local phase needed by gate 1
//
// A represented X+0 face query is not written by the seed.  Full F alone
// decides whether the query emits carrier matter and whether the same B0 can
// serve another query.  There is no timer, answer value, or host-side rearm.

#include <array>
#include <cstdint>

#include "bcc32_developmental_seed.cuh"
#include "bcc32_law.cuh"

namespace substrate::bcc32 {

using CatalyticReadoutSeedHash = std::uint8_t;
constexpr CatalyticReadoutSeedHash kCatalyticReadoutSourceBond = 0x01u;
constexpr CatalyticReadoutSeedHash kCatalyticReadoutReceiverConformation = 0x02u;
constexpr CatalyticReadoutSeedHash kCatalyticReadoutGeneMask =
    kCatalyticReadoutSourceBond | kCatalyticReadoutReceiverConformation;
constexpr CatalyticReadoutSeedHash kCatalyticReadoutSeedHash =
    kCatalyticReadoutGeneMask;
constexpr CatalyticReadoutSeedHash kCatalyticReadoutBondCutSeedHash =
    kCatalyticReadoutReceiverConformation;
constexpr CatalyticReadoutSeedHash kCatalyticReadoutConformationCutSeedHash =
    kCatalyticReadoutSourceBond;

constexpr bool catalytic_readout_has(CatalyticReadoutSeedHash hash,
                                     CatalyticReadoutSeedHash flag) {
  return (hash & flag) != 0u;
}

constexpr std::array<DevelopmentalSeedSite, 2u> catalytic_readout_seed(
    CatalyticReadoutSeedHash hash) {
  SiteWord source = kQ;
  SiteWord receiver = kQ;
  if (catalytic_readout_has(hash, kCatalyticReadoutSourceBond))
    source |= owned_bond_bit(0u);
  if (catalytic_readout_has(hash, kCatalyticReadoutReceiverConformation))
    receiver |= channel_bit(kConformationShift, 0u);
  return {{{-1, 0, 0, source}, {0, 0, 0, receiver}}};
}

}  // namespace substrate::bcc32
