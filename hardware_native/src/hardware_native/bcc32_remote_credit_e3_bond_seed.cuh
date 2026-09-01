#pragma once

// Compact joint precursor: a remote P7-hole route, its local R3 receptor, and
// one X+3 helper at a non-overlapping edge of the measured +u3 E3 bath site.
// The six-bit hash names only route distance and one of the seven available
// local ports.  F, rather than a recipe schedule, determines whether E3 can
// write B3 across that edge.

#include <array>

#include "bcc32_remote_credit_hole_transducer_seed.cuh"

namespace substrate::bcc32 {

using RemoteCreditE3BondSeedHash = std::uint8_t;
constexpr std::uint32_t kRemoteCreditE3BondRouteMask = 0x07u;
constexpr std::uint32_t kRemoteCreditE3BondHelperShift = 3u;
constexpr std::uint32_t kRemoteCreditE3BondHelperCount = 7u;
constexpr std::size_t kRemoteCreditE3BondSeedSiteCount =
    kRemoteCreditHoleTransducerSeedSiteCount + 1u;

constexpr RemoteCreditE3BondSeedHash make_remote_credit_e3_bond_seed_hash(
    std::uint32_t route_length, std::uint32_t helper_port) {
  return static_cast<RemoteCreditE3BondSeedHash>(
      (route_length & kRemoteCreditE3BondRouteMask) |
      ((helper_port & 0x07u) << kRemoteCreditE3BondHelperShift));
}

constexpr RemoteCreditHoleTransducerSeedHash remote_credit_e3_bond_route_hash(
    RemoteCreditE3BondSeedHash hash) {
  return static_cast<RemoteCreditHoleTransducerSeedHash>(hash & kRemoteCreditE3BondRouteMask);
}

constexpr std::uint32_t remote_credit_e3_bond_helper_port(RemoteCreditE3BondSeedHash hash) {
  return static_cast<std::uint32_t>(hash >> kRemoteCreditE3BondHelperShift);
}

inline Z3Coordinate remote_credit_e3_bond_source() { return {1, 1, 1}; }

inline Z3Coordinate remote_credit_e3_bond_helper(RemoteCreditE3BondSeedHash hash) {
  const std::uint32_t port = remote_credit_e3_bond_helper_port(hash);
  const Int3 offset = direction_offset(static_cast<Direction>(port));
  const Z3Coordinate source = remote_credit_e3_bond_source();
  return {source.x + offset.x, source.y + offset.y, source.z + offset.z};
}

inline std::array<DevelopmentalSeedSite, kRemoteCreditE3BondSeedSiteCount>
remote_credit_e3_bond_seed(CreditOrbitSeedHash parent, RemoteCreditE3BondSeedHash hash) {
  std::array<DevelopmentalSeedSite, kRemoteCreditE3BondSeedSiteCount> result{};
  const auto base = remote_credit_hole_transducer_seed(parent, remote_credit_e3_bond_route_hash(hash));
  for (std::size_t index = 0u; index < base.size(); ++index) result[index] = base[index];
  const Z3Coordinate helper = remote_credit_e3_bond_helper(hash);
  result.back() = {static_cast<std::int8_t>(helper.x), static_cast<std::int8_t>(helper.y),
                   static_cast<std::int8_t>(helper.z), static_cast<SiteWord>(kQ | face_bit(3u))};
  return result;
}

}  // namespace substrate::bcc32
