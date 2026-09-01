#pragma once

// A compact developmental gene closes the successful signed-credit waste
// paths of an eight-stage processive weight.
//
// Each stage grows one local carrier corner at the release factor's negative
// waste outlet.  The corner turns the spent P-_w vacancy onto +marker.  The
// vacancy then reaches a stage-specific, fixed world port at tick 64.  A
// reciprocal boundary tape can receive the vacancy there, so the tissue does
// not retain an ever-longer ballistic tail.  The gene names only local roads
// and ports; it does not contain a learned level, target, or host instruction.

#include <array>
#include <cstdint>

#include "bcc32_processive_weight_region_seed.cuh"

namespace substrate::bcc32 {

using ProcessiveCreditReturnSeedHash = std::uint64_t;

inline constexpr std::uint32_t kProcessiveCreditReturnGene = 0x8au;
inline constexpr std::uint32_t kProcessiveCreditReturnStageCount = 8u;
inline constexpr std::uint32_t kProcessiveCreditReturnSettleTicks = 64u;
inline constexpr std::uint32_t kProcessiveCreditReturnSeedSiteCount =
    kProcessiveCreditReturnStageCount * kProcessiveWeightSitesPerCell +
    kProcessiveCreditReturnStageCount * (kCarrierCornerSiteCount - 1u);

constexpr ProcessiveCreditReturnSeedHash make_processive_credit_return_seed_hash(
    ProcessiveWeightRegionSeedHash weight_hash, std::uint32_t gene) {
  return static_cast<ProcessiveCreditReturnSeedHash>(weight_hash) |
         (static_cast<ProcessiveCreditReturnSeedHash>(gene & 0xffu) << 32u);
}

constexpr ProcessiveWeightRegionSeedHash processive_credit_return_weight_hash(
    ProcessiveCreditReturnSeedHash hash) {
  return static_cast<ProcessiveWeightRegionSeedHash>(hash);
}

constexpr std::uint32_t processive_credit_return_gene(
    ProcessiveCreditReturnSeedHash hash) {
  return static_cast<std::uint32_t>((hash >> 32u) & 0xffu);
}

constexpr bool valid_processive_credit_return_seed_hash(
    ProcessiveCreditReturnSeedHash hash) {
  const ProcessiveWeightRegionSeedHash weight_hash =
      processive_credit_return_weight_hash(hash);
  return processive_credit_return_gene(hash) ==
             kProcessiveCreditReturnGene &&
         valid_processive_weight_region_hash(weight_hash) &&
         processive_weight_length(weight_hash) ==
             kProcessiveCreditReturnStageCount;
}

constexpr DevelopmentalSeedSite processive_credit_return_site(
    Int3 coordinate, SiteWord word) {
  return {static_cast<std::int8_t>(coordinate.x),
          static_cast<std::int8_t>(coordinate.y),
          static_cast<std::int8_t>(coordinate.z), word};
}

constexpr Int3 processive_credit_scaled(Int3 value,
                                        std::int32_t factor) {
  return {value.x * factor, value.y * factor, value.z * factor};
}

constexpr Int3 processive_credit_return_body(
    ProcessiveCreditReturnSeedHash hash, std::uint32_t stage) {
  const ProcessiveWeightRegionSeedHash weight_hash =
      processive_credit_return_weight_hash(hash);
  const Int3 path =
      direction_offset(static_cast<Direction>(processive_weight_path(weight_hash)));
  return processive_credit_scaled(
      path, static_cast<std::int32_t>(6u * stage));
}

constexpr Int3 processive_positive_credit_outlet(
    ProcessiveCreditReturnSeedHash hash, std::uint32_t stage) {
  const ProcessiveWeightRegionSeedHash weight_hash =
      processive_credit_return_weight_hash(hash);
  const Int3 waste = direction_offset(
      static_cast<Direction>(processive_weight_waste(weight_hash)));
  return processive_credit_return_body(hash, stage) +
         processive_credit_scaled(waste, -3);
}

// A successful positive pulse crosses one six-site stage per four supersteps:
// release, S_P flight, next-stage contact, and restaging overlap in the
// seven-factor epoch. It reaches stage s's waste outlet at tick 4*s+1. The
// carrier corner turns
// it on the following factor pass; S_P then advances it once per remaining
// tick, so this is the exact tick-64 boundary coordinate.
constexpr Int3 processive_positive_credit_return_port(
    ProcessiveCreditReturnSeedHash hash, std::uint32_t stage) {
  const ProcessiveWeightRegionSeedHash weight_hash =
      processive_credit_return_weight_hash(hash);
  const Int3 outgoing = direction_offset(
      static_cast<Direction>(processive_weight_marker(weight_hash)));
  const std::int32_t remaining =
      static_cast<std::int32_t>(kProcessiveCreditReturnSettleTicks -
                                (4u * stage + 1u));
  return processive_positive_credit_outlet(hash, stage) +
         processive_credit_scaled(outgoing, remaining);
}

// Successful negative updates emit the same P-_w product, but the root contact
// begins at the tail.  The first changed stage is seven, so the propagation
// delay is four ticks for every already-zero stage crossed toward stage zero.
constexpr Int3 processive_negative_credit_return_port(
    ProcessiveCreditReturnSeedHash hash, std::uint32_t stage) {
  const ProcessiveWeightRegionSeedHash weight_hash =
      processive_credit_return_weight_hash(hash);
  const Int3 outgoing = direction_offset(
      static_cast<Direction>(processive_weight_marker(weight_hash)));
  const std::uint32_t crossed =
      (kProcessiveCreditReturnStageCount - 1u) - stage;
  const std::int32_t remaining =
      static_cast<std::int32_t>(kProcessiveCreditReturnSettleTicks -
                                (4u * crossed + 1u));
  return processive_positive_credit_outlet(hash, stage) +
         processive_credit_scaled(outgoing, remaining);
}

constexpr std::array<DevelopmentalSeedSite,
                     kProcessiveCreditReturnSeedSiteCount>
processive_credit_return_seed(ProcessiveCreditReturnSeedHash hash) {
  std::array<DevelopmentalSeedSite,
             kProcessiveCreditReturnSeedSiteCount>
      result{};
  if (!valid_processive_credit_return_seed_hash(hash)) return result;

  const ProcessiveWeightRegionSeedHash weight_hash =
      processive_credit_return_weight_hash(hash);
  const auto weight = processive_weight_region_seed(weight_hash);
  std::uint32_t next = 0u;
  for (std::uint32_t index = 0u;
       index < kProcessiveCreditReturnStageCount *
                   kProcessiveWeightSitesPerCell;
       ++index)
    result[next++] = weight[index];

  const std::uint32_t incoming =
      processive_weight_waste(weight_hash) + 4u;
  const std::uint32_t outgoing = processive_weight_marker(weight_hash);
  for (std::uint32_t stage = 0u;
       stage < kProcessiveCreditReturnStageCount; ++stage) {
    const Int3 center = processive_positive_credit_outlet(hash, stage);
    for (std::uint32_t index = 1u; index < kCarrierCornerSiteCount;
         ++index) {
      const CarrierCornerOffset offset =
          carrier_corner_offset(incoming, outgoing, index);
      result[next++] = processive_credit_return_site(
          center + Int3{offset.x, offset.y, offset.z},
          carrier_corner_word(incoming, outgoing, index, false));
    }
  }
  return result;
}

inline constexpr ProcessiveCreditReturnSeedHash
    kProcessiveCreditReturnSeedHash =
        make_processive_credit_return_seed_hash(
            kProcessiveWeightRegionSeedHash,
            kProcessiveCreditReturnGene);

static_assert(
    valid_processive_credit_return_seed_hash(
        kProcessiveCreditReturnSeedHash));
static_assert(processive_positive_credit_return_port(
                  kProcessiveCreditReturnSeedHash, 0u) ==
              Int3{63, 0, -3});
static_assert(processive_positive_credit_return_port(
                  kProcessiveCreditReturnSeedHash, 7u) ==
              Int3{35, 42, -3});
static_assert(processive_negative_credit_return_port(
                  kProcessiveCreditReturnSeedHash, 7u) ==
              Int3{63, 42, -3});
static_assert(processive_negative_credit_return_port(
                  kProcessiveCreditReturnSeedHash, 0u) ==
              Int3{35, 0, -3});

}  // namespace substrate::bcc32
