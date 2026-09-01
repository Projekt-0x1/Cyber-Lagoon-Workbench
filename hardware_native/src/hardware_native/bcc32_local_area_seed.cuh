#pragma once

// First composite local-area seed: error contact plus eligibility contact.
//
// A seven-bit genesis hash selects three local founder genes: the five-cell
// prediction-error nucleus and the two B/E eligibility parents.  The decoder
// runs only at birth to place ordinary BCC matter; the hash is never present
// in, read by, or interpreted during F.  Complete F plus reciprocal raw
// contact remain the sole runtime dynamics.

#include <array>
#include <cstdint>

#include "bcc32_attraction_seed.cuh"
#include "bcc32_error_attraction_seed.cuh"
#include "bcc32_reanchor_pair_gene.cuh"

namespace substrate::bcc32 {

using ErrorEligibilityAreaSeedHash = std::uint8_t;
constexpr ErrorEligibilityAreaSeedHash kErrorEligibilityErrorGene = 0x01u;
constexpr ErrorEligibilityAreaSeedHash kErrorEligibilityParent0Gene = 0x02u;
constexpr ErrorEligibilityAreaSeedHash kErrorEligibilityParent1Gene = 0x04u;
constexpr ErrorEligibilityAreaSeedHash kErrorEligibilityAreaSeedHash =
    kErrorEligibilityErrorGene | kErrorEligibilityParent0Gene |
    kErrorEligibilityParent1Gene;
constexpr ErrorEligibilityAreaSeedHash kErrorEligibilityAreaParent0LesionHash =
    kErrorEligibilityAreaSeedHash & static_cast<ErrorEligibilityAreaSeedHash>(~kErrorEligibilityParent0Gene);
constexpr ErrorEligibilityAreaSeedHash kErrorEligibilityAreaParent1LesionHash =
    kErrorEligibilityAreaSeedHash & static_cast<ErrorEligibilityAreaSeedHash>(~kErrorEligibilityParent1Gene);

constexpr std::size_t kErrorEligibilityAreaErrorCount = kErrorAttractionSeed.size();
constexpr std::size_t kErrorEligibilityAreaEligibilityCount = kEligibilityAttractionSeed.size();

constexpr DevelopmentalSeedSite shifted_seed_site(DevelopmentalSeedSite site,
                                                  std::int8_t x,
                                                  std::int8_t y,
                                                  std::int8_t z) {
  site.x = static_cast<std::int8_t>(site.x + x);
  site.y = static_cast<std::int8_t>(site.y + y);
  site.z = static_cast<std::int8_t>(site.z + z);
  return site;
}

constexpr std::array<DevelopmentalSeedSite,
                     kErrorEligibilityAreaErrorCount + kErrorEligibilityAreaEligibilityCount>
error_eligibility_area_seed(ErrorEligibilityAreaSeedHash hash) {
  std::array<DevelopmentalSeedSite,
             kErrorEligibilityAreaErrorCount + kErrorEligibilityAreaEligibilityCount> result{};
  for (DevelopmentalSeedSite& site : result) site = {0, 0, 0, kQ};
  std::size_t cursor = 0u;
  for (const DevelopmentalSeedSite& site : kErrorAttractionSeed) {
    if ((hash & kErrorEligibilityErrorGene) != 0u) result[cursor] = site;
    ++cursor;
  }
  // The error receptor is (-1,-1,1); grow the two eligibility founders around it.
  for (std::size_t index = 0u; index < kEligibilityAttractionSeed.size(); ++index) {
    const ErrorEligibilityAreaSeedHash gene =
        index == 0u ? kErrorEligibilityParent0Gene : kErrorEligibilityParent1Gene;
    if ((hash & gene) != 0u)
      result[cursor] = shifted_seed_site(kEligibilityAttractionSeed[index], -1, -1, 1);
    ++cursor;
  }
  return result;
}

inline constexpr auto kErrorEligibilityAreaSeed =
    error_eligibility_area_seed(kErrorEligibilityAreaSeedHash);

constexpr std::uint64_t error_eligibility_area_seed_fingerprint() {
  std::uint64_t hash = 14695981039346656037ull;
  const auto add = [&hash](std::uint8_t byte) constexpr {
    hash ^= byte;
    hash *= 1099511628211ull;
  };
  for (const DevelopmentalSeedSite& site : error_eligibility_area_seed(kErrorEligibilityAreaSeedHash)) {
    add(static_cast<std::uint8_t>(site.x));
    add(static_cast<std::uint8_t>(site.y));
    add(static_cast<std::uint8_t>(site.z));
    for (std::uint32_t shift = 0u; shift < 32u; shift += 8u)
      add(static_cast<std::uint8_t>(site.word >> shift));
  }
  return hash;
}

inline constexpr std::uint64_t kErrorEligibilityAreaSeedFingerprint =
    error_eligibility_area_seed_fingerprint();

// A compact, developmentally active local area.  Four B/E founders are born
// from this hash; all relay C/R lanes and the non-adjacent activity path are
// subsequently ordinary matter evolved by F.  The hash is genesis-only, not a
// runtime controller or an edge table.
using GrownActivityAreaSeedHash = ReanchorPairHash;
constexpr GrownActivityAreaSeedHash kGrownActivityAreaSeedHash =
    kFirstConstructionAreaHash;

[[nodiscard]] constexpr bool grown_activity_area_seed_hash_valid(
    GrownActivityAreaSeedHash hash) {
  return reanchor_pair_hash_valid(hash);
}

[[nodiscard]] constexpr std::array<DevelopmentalSeedSite, 4u>
grown_activity_area_seed(GrownActivityAreaSeedHash hash) {
  return reanchor_pair_seed(hash);
}

constexpr std::uint64_t grown_activity_area_seed_fingerprint() {
  std::uint64_t hash = 14695981039346656037ull;
  const auto add = [&hash](std::uint8_t byte) constexpr {
    hash ^= byte;
    hash *= 1099511628211ull;
  };
  for (const DevelopmentalSeedSite& site :
       grown_activity_area_seed(kGrownActivityAreaSeedHash)) {
    add(static_cast<std::uint8_t>(site.x));
    add(static_cast<std::uint8_t>(site.y));
    add(static_cast<std::uint8_t>(site.z));
    for (std::uint32_t shift = 0u; shift < 32u; shift += 8u)
      add(static_cast<std::uint8_t>(site.word >> shift));
  }
  return hash;
}

inline constexpr std::uint64_t kGrownActivityAreaSeedFingerprint =
    grown_activity_area_seed_fingerprint();

}  // namespace substrate::bcc32
