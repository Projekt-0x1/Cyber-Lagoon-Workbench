#pragma once

// One compact developmental route seed.  Its low three bits state how far the
// measured credit emitter is translated along -u3.  The emitter keeps the parent
// weight/receptor, receiver, and orbit but deliberately omits the unrelated dormant
// successor-weight founders: in the translated geometry that dormant B3 founder sits
// directly on the P7 road.  The interpreter merely births these local types; ordinary
// F carries any later credit output back toward the untouched origin.  No route table,
// timing program, or target write is stored in the hash.

#include <array>
#include <cstdint>

#include "bcc32_credit_orbit_seed.cuh"

namespace substrate::bcc32 {

using RemoteCreditRouteSeedHash = std::uint8_t;
constexpr std::uint32_t kRemoteCreditRouteMinLength = 1u;
constexpr std::uint32_t kRemoteCreditRouteMaxLength = 7u;
constexpr std::size_t kRemoteCreditRouteExcludedBudBegin = kCreditBackcarrySeedSiteCount;
constexpr std::size_t kRemoteCreditRouteExcludedBudEnd =
    kRemoteCreditRouteExcludedBudBegin + kLocalFounderBasisCount;
constexpr std::size_t kRemoteCreditRouteSeedSiteCount =
    kCreditOrbitSeedSiteCount - kLocalFounderBasisCount;

constexpr RemoteCreditRouteSeedHash make_remote_credit_route_seed_hash(std::uint32_t length) {
  return static_cast<RemoteCreditRouteSeedHash>(length & 0x07u);
}

constexpr std::uint32_t remote_credit_route_length(RemoteCreditRouteSeedHash hash) {
  return static_cast<std::uint32_t>(hash & 0x07u);
}

inline std::array<DevelopmentalSeedSite, kRemoteCreditRouteSeedSiteCount>
remote_credit_route_seed(CreditOrbitSeedHash parent, RemoteCreditRouteSeedHash hash) {
  std::array<DevelopmentalSeedSite, kRemoteCreditRouteSeedSiteCount> result{};
  const auto source = credit_orbit_seed(parent);
  const std::int32_t length = static_cast<std::int32_t>(remote_credit_route_length(hash));
  std::size_t output = 0u;
  for (std::size_t index = 0u; index < source.size(); ++index) {
    if (index >= kRemoteCreditRouteExcludedBudBegin && index < kRemoteCreditRouteExcludedBudEnd)
      continue;
    result[output++] = {static_cast<std::int8_t>(source[index].x - length),
                        static_cast<std::int8_t>(source[index].y - length),
                        static_cast<std::int8_t>(source[index].z - length), source[index].word};
  }
  static_cast<void>(output);
  return result;
}

inline Z3Coordinate remote_credit_route_receptor(RemoteCreditRouteSeedHash hash) {
  const std::int32_t length = static_cast<std::int32_t>(remote_credit_route_length(hash));
  return {-length, -length, 1 - length};
}

inline Z3Coordinate remote_credit_route_output(CreditOrbitSeedHash parent,
                                               RemoteCreditRouteSeedHash hash) {
  const auto source = credit_orbit_seed(parent);
  const DevelopmentalSeedSite output = source[kCreditBudReceiverSeedSiteCount];
  const std::int32_t length = static_cast<std::int32_t>(remote_credit_route_length(hash));
  return {output.x - length, output.y - length, output.z - length};
}

inline Z3Coordinate remote_credit_route_founder(std::uint32_t basis,
                                                RemoteCreditRouteSeedHash hash) {
  const std::array<Z3Coordinate, 4> founders{{{-1, 0, 0}, {0, -1, 0},
                                               {0, 0, -1}, {1, 1, 1}}};
  const std::int32_t length = static_cast<std::int32_t>(remote_credit_route_length(hash));
  return {founders[basis].x - length, founders[basis].y - length, founders[basis].z - length};
}

}  // namespace substrate::bcc32
