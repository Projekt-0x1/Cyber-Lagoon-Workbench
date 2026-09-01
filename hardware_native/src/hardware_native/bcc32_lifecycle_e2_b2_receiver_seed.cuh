#pragma once

// Compact F-only interface for the measured lifecycle E2 source.  The seed
// contains the two generic lifecycle catalysts and the one +u2 B2/B2 edge
// selected by the law-derived receiver preimage map.  Its bits select only
// local anatomy; resource remains ordinary environmental matter.

#include <array>
#include <cstdint>

#include "bcc32_developmental_seed.cuh"
#include "bcc32_law.cuh"
#include "bcc32_lifecycle_seed.cuh"

namespace substrate::bcc32 {

using LifecycleE2B2ReceiverSeedHash = std::uint8_t;
constexpr std::uint32_t kLifecycleE2SourceB2Bit = 0u;
constexpr std::uint32_t kLifecycleE2TargetB2Bit = 1u;
constexpr std::uint32_t kLifecycleE2TargetPplusSiblingBit = 2u;
constexpr LifecycleE2B2ReceiverSeedHash make_lifecycle_e2_b2_receiver_seed_hash(
    bool source_b2, bool target_b2, bool target_pplus_sibling = false) {
  return static_cast<LifecycleE2B2ReceiverSeedHash>(
      (source_b2 ? (1u << kLifecycleE2SourceB2Bit) : 0u) |
      (target_b2 ? (1u << kLifecycleE2TargetB2Bit) : 0u) |
      (target_pplus_sibling ? (1u << kLifecycleE2TargetPplusSiblingBit) : 0u));
}

inline constexpr LifecycleE2B2ReceiverSeedHash kLifecycleE2B2ReceiverSeedHash =
    make_lifecycle_e2_b2_receiver_seed_hash(true, true);
inline constexpr std::size_t kLifecycleE2B2ReceiverSeedSiteCount = 4u;

constexpr std::array<DevelopmentalSeedSite, kLifecycleE2B2ReceiverSeedSiteCount>
lifecycle_e2_b2_receiver_seed(LifecycleE2B2ReceiverSeedHash hash) {
  const bool source_b2 = (hash & (1u << kLifecycleE2SourceB2Bit)) != 0u;
  const bool target_b2 = (hash & (1u << kLifecycleE2TargetB2Bit)) != 0u;
  const bool target_pplus = (hash & (1u << kLifecycleE2TargetPplusSiblingBit)) != 0u;
  return {{{-1, 0, 0, kMetabolicCatalystSeed[0].word},
           {0, -1, 0, kMetabolicCatalystSeed[1].word},
           {0, 0, -1, static_cast<SiteWord>(kQ | (source_b2 ? owned_bond_bit(2u) : 0u))},
           {0, 0, 0, static_cast<SiteWord>(kQ | (target_b2 ? owned_bond_bit(2u) : 0u) |
                                             (target_pplus ? carrier_bit(2u) : 0u))}}};
}

static_assert(kLifecycleE2B2ReceiverSeedHash == 0x3u);

}  // namespace substrate::bcc32
