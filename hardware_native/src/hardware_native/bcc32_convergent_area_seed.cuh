#pragma once

// Compact germ for a bounded convergent local area.
//
// This is one physical 128-bit seed value, split only because the existing
// arbor and orbit grammars each already occupy a 64-bit content word.  The
// arbor is four S4-related delayed-credit contacts.  Its four physical routes
// meet at local zero.  The core is the measured rotated orbit with its C3
// source at that same local zero.  Thus the only authored relation is a local
// co-location of two already measured physical interfaces; no hash bit names
// a task, output, schedule, or cell identity.  Complete F remains the sole
// runtime interpreter after birth.

#include <array>
#include <cstddef>
#include <cstdint>

#include "bcc32_credit_orbit_seed.cuh"
#include "bcc32_synaptic_arbor_seed.cuh"

namespace substrate::bcc32 {

struct ConvergentAreaSeedHash {
  SynapticArborSeedHash arbor = 0u;
  CreditOrbitSeedHash core = 0u;

  constexpr bool operator==(const ConvergentAreaSeedHash&) const = default;
};

constexpr ConvergentAreaSeedHash make_convergent_area_seed_hash(
    SynapticArborSeedHash arbor, CreditOrbitSeedHash core) {
  return {arbor, core};
}

inline constexpr CreditOrbitSeedHash kConvergentAreaCoreHash =
    make_credit_orbit_seed_hash(kCreditBudReceiverLocalRNoBHash, 0, 0, 0);
inline constexpr ConvergentAreaSeedHash kConvergentAreaSeedHash =
    make_convergent_area_seed_hash(kSynapticArborSeedHash, kConvergentAreaCoreHash);

inline constexpr std::size_t kConvergentAreaSeedSiteCount =
    kSynapticArborSeedSiteCount + kCreditOrbitSeedSiteCount;

inline std::array<DevelopmentalSeedSite, kConvergentAreaSeedSiteCount>
convergent_area_seed(ConvergentAreaSeedHash hash) {
  std::array<DevelopmentalSeedSite, kConvergentAreaSeedSiteCount> result{};
  const auto arbor = synaptic_arbor_seed(hash.arbor);
  const auto core = credit_orbit_seed(hash.core);
  std::size_t cursor = 0u;
  for (const DevelopmentalSeedSite& site : arbor) result[cursor++] = site;
  for (const DevelopmentalSeedSite& site : core) result[cursor++] = site;
  return result;
}

static_assert(credit_orbit_origin(kConvergentAreaCoreHash) ==
              std::array<std::int32_t, 3>{{0, 0, 0}});

}  // namespace substrate::bcc32
