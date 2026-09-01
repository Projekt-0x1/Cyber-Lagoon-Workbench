#pragma once

// Compact developmental extension of the measured reanchor area.
//
// The five-bit hash contains only a reanchor attachment choice plus one
// ordinary B/E founder gene on the already-measured activity target.  It does
// not encode R2, a route, a schedule, or a response.  Whether F grows an R2
// receptor while preserving the activity tissue is a contract question.

#include <array>
#include <cstdint>

#include "bcc32_local_founder_hash.cuh"
#include "bcc32_reanchor_pair_gene.cuh"

namespace substrate::bcc32 {

using ReanchorR2AreaHash = std::uint8_t;
constexpr ReanchorR2AreaHash kReanchorR2AreaHashMask = 0x1fu;
constexpr std::int8_t kReanchorR2AreaTargetX = 0;
constexpr std::int8_t kReanchorR2AreaTargetY = 0;
constexpr std::int8_t kReanchorR2AreaTargetZ = -2;

[[nodiscard]] constexpr bool reanchor_r2_area_hash_valid(ReanchorR2AreaHash hash) {
  return (hash & ~kReanchorR2AreaHashMask) == 0u;
}

[[nodiscard]] constexpr ReanchorPairHash reanchor_r2_area_parent_hash(
    ReanchorR2AreaHash hash) {
  return static_cast<ReanchorPairHash>(hash & 0x01u);
}

[[nodiscard]] constexpr std::uint32_t reanchor_r2_area_founder_basis(
    ReanchorR2AreaHash hash) {
  return (hash >> 1u) & 0x03u;
}

[[nodiscard]] constexpr LocalFounderGene reanchor_r2_area_founder_gene(
    ReanchorR2AreaHash hash) {
  return static_cast<LocalFounderGene>((hash >> 3u) & 0x03u);
}

[[nodiscard]] constexpr std::array<DevelopmentalSeedSite, 5u> reanchor_r2_area_seed(
    ReanchorR2AreaHash hash) {
  const auto parent = reanchor_pair_seed(reanchor_r2_area_parent_hash(hash));
  const std::uint32_t basis = reanchor_r2_area_founder_basis(hash);
  return {{parent[0], parent[1], parent[2], parent[3],
           {kReanchorR2AreaTargetX, kReanchorR2AreaTargetY, kReanchorR2AreaTargetZ,
            local_founder_word(reanchor_r2_area_founder_gene(hash), basis)}}};
}

}  // namespace substrate::bcc32
