#pragma once

// A minimal directly seeded receiver for the measured remote P7 credit route.
// The hash selects only the route translation; the one local R3 receptor is
// placed in the developmental seed at the route's otherwise untouched origin.
// The neighbouring carrier bath remains Q, so F—not a host write—must decide
// whether remote P7 can create a local molecular event.  This seed alone is
// not evidence that F grew the receptor from an earlier germ.

#include <array>

#include "bcc32_remote_credit_route_seed.cuh"

namespace substrate::bcc32 {

using RemoteCreditHoleTransducerSeedHash = RemoteCreditRouteSeedHash;
constexpr std::size_t kRemoteCreditHoleTransducerSeedSiteCount =
    kRemoteCreditRouteSeedSiteCount + 1u;

inline std::array<DevelopmentalSeedSite, kRemoteCreditHoleTransducerSeedSiteCount>
remote_credit_hole_transducer_seed(CreditOrbitSeedHash parent,
                                   RemoteCreditHoleTransducerSeedHash hash) {
  std::array<DevelopmentalSeedSite, kRemoteCreditHoleTransducerSeedSiteCount> result{};
  const auto route = remote_credit_route_seed(parent, hash);
  for (std::size_t index = 0u; index < route.size(); ++index) result[index] = route[index];
  result.back() = {0, 0, 0, static_cast<SiteWord>(kQ | channel_bit(kReactiveShift, 3u))};
  return result;
}

}  // namespace substrate::bcc32
