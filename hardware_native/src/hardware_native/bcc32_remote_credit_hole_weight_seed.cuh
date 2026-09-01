#pragma once

// One extra local receptor gene composes the measured P-3-hole -> E3 chemistry
// with the oriented remote-credit weight seed.  It is ordinary R3 matter at the
// origin hub; the hash still specifies only route length and S4 orientation.

#include <array>

#include "bcc32_remote_credit_weight_seed.cuh"

namespace substrate::bcc32 {

constexpr std::size_t kRemoteCreditHoleWeightSeedSiteCount =
    kRemoteCreditWeightSeedSiteCount + 1u;

inline std::array<DevelopmentalSeedSite, kRemoteCreditHoleWeightSeedSiteCount>
remote_credit_hole_weight_seed(CreditOrbitSeedHash parent, RemoteCreditWeightSeedHash hash,
                               SynapticWeightSeedHash weight_hash = kSynapticWeightSeedHash) {
  std::array<DevelopmentalSeedSite, kRemoteCreditHoleWeightSeedSiteCount> result{};
  const auto base = remote_credit_weight_seed(parent, hash, weight_hash);
  for (std::size_t index = 0u; index < base.size(); ++index) result[index] = base[index];
  result.back() = {0, 0, 0, static_cast<SiteWord>(kQ | channel_bit(kReactiveShift, 3u))};
  return result;
}

}  // namespace substrate::bcc32
