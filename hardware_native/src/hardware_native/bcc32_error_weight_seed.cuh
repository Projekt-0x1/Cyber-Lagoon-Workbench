#pragma once

// Compact composition gene for a weight germ plus a nearby mismatch organ.
//
// The low byte is the ordinary four-basis weight germ. Three signed six-bit
// loci place the already-physical five-cell mismatch germ. The hash contains
// no timing, expected answer, weight value, route, or update instruction.
// Birth expands known cell-pool genes once; canonical F remains the only
// runtime interpreter.

#include <array>
#include <cstdint>

#include "bcc32_error_attraction_seed.cuh"
#include "bcc32_synaptic_weight_seed.cuh"

namespace substrate::bcc32 {

using ErrorWeightSeedHash = std::uint32_t;

inline constexpr std::uint32_t kErrorWeightHashWeightShift = 0u;
inline constexpr std::uint32_t kErrorWeightHashOffsetXShift = 8u;
inline constexpr std::uint32_t kErrorWeightHashOffsetYShift = 14u;
inline constexpr std::uint32_t kErrorWeightHashOffsetZShift = 20u;
inline constexpr std::uint32_t kErrorWeightHashOffsetMask = 0x3fu;
inline constexpr std::int32_t kErrorWeightHashOffsetBias = 32;

constexpr std::uint32_t encode_error_weight_offset(std::int32_t value) {
  return static_cast<std::uint32_t>(value + kErrorWeightHashOffsetBias) &
         kErrorWeightHashOffsetMask;
}

constexpr std::int32_t decode_error_weight_offset(ErrorWeightSeedHash hash, std::uint32_t shift) {
  return static_cast<std::int32_t>((hash >> shift) & kErrorWeightHashOffsetMask) -
         kErrorWeightHashOffsetBias;
}

constexpr ErrorWeightSeedHash make_error_weight_seed_hash(SynapticWeightSeedHash weight_hash,
                                                          std::int32_t offset_x,
                                                          std::int32_t offset_y,
                                                          std::int32_t offset_z) {
  return static_cast<ErrorWeightSeedHash>(weight_hash) << kErrorWeightHashWeightShift |
         encode_error_weight_offset(offset_x) << kErrorWeightHashOffsetXShift |
         encode_error_weight_offset(offset_y) << kErrorWeightHashOffsetYShift |
         encode_error_weight_offset(offset_z) << kErrorWeightHashOffsetZShift;
}

constexpr SynapticWeightSeedHash error_weight_synapse_hash(ErrorWeightSeedHash hash) {
  return static_cast<SynapticWeightSeedHash>(hash & 0xffu);
}

constexpr std::array<std::int32_t, 3> error_weight_offset(ErrorWeightSeedHash hash) {
  return {{
      decode_error_weight_offset(hash, kErrorWeightHashOffsetXShift),
      decode_error_weight_offset(hash, kErrorWeightHashOffsetYShift),
      decode_error_weight_offset(hash, kErrorWeightHashOffsetZShift),
  }};
}

inline constexpr std::size_t kErrorWeightSeedSiteCount =
    kLocalFounderBasisCount + kErrorAttractionSeed.size();

constexpr std::array<DevelopmentalSeedSite, kErrorWeightSeedSiteCount> error_weight_seed(
    ErrorWeightSeedHash hash) {
  std::array<DevelopmentalSeedSite, kErrorWeightSeedSiteCount> result{};
  const auto weight = synaptic_weight_seed(error_weight_synapse_hash(hash));
  for (std::size_t index = 0u; index < weight.size(); ++index)
    result[index] = weight[index];

  const auto offset = error_weight_offset(hash);
  for (std::size_t index = 0u; index < kErrorAttractionSeed.size(); ++index) {
    const DevelopmentalSeedSite& site = kErrorAttractionSeed[index];
    result[kLocalFounderBasisCount + index] = {
        static_cast<std::int8_t>(site.x + offset[0]), static_cast<std::int8_t>(site.y + offset[1]),
        static_cast<std::int8_t>(site.z + offset[2]), site.word};
  }
  return result;
}

constexpr DevelopmentalSeedSite error_weight_error_site(ErrorWeightSeedHash hash,
                                                        std::size_t error_site_index) {
  return error_weight_seed(hash)[kLocalFounderBasisCount + error_site_index];
}

inline constexpr ErrorWeightSeedHash kErrorWeightSeedHash =
    make_error_weight_seed_hash(kSynapticWeightSeedHash, 2, 3, 1);

static_assert(kErrorWeightSeedHash == 0x0218e255u);
static_assert(error_weight_synapse_hash(kErrorWeightSeedHash) == kSynapticWeightSeedHash);
static_assert(error_weight_offset(kErrorWeightSeedHash) == std::array<std::int32_t, 3>{{2, 3, 1}});

}  // namespace substrate::bcc32
