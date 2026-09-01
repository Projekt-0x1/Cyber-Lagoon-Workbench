#pragma once

// Two-founder developmental grammar for the R2 receptor frontier.
//
// A compact word selects only generic B/E matter: one founder at the measured
// P-2 activity target and one at an adjacent BCC port.  F remains the sole
// interpreter.  The hash contains neither R2 nor an activity/readout rule.

#include <array>
#include <cstdint>

#include "bcc32_local_founder_hash.cuh"
#include "bcc32_reanchor_pair_gene.cuh"

namespace substrate::bcc32 {

using ReanchorR2PairHash = std::uint16_t;
constexpr ReanchorR2PairHash kReanchorR2PairHashMask = 0x03ffu;
constexpr std::int8_t kReanchorR2PairTargetX = 0;
constexpr std::int8_t kReanchorR2PairTargetY = 0;
constexpr std::int8_t kReanchorR2PairTargetZ = -2;

[[nodiscard]] constexpr ReanchorR2PairHash make_reanchor_r2_pair_hash(
    ReanchorPairHash parent, std::uint32_t target_basis, LocalFounderGene target_gene,
    std::uint32_t helper_port, LocalFounderGene helper_gene) {
  return static_cast<ReanchorR2PairHash>((parent & 0x01u) | ((target_basis & 0x03u) << 1u) |
                                         ((static_cast<std::uint32_t>(target_gene) & 0x03u) << 3u) |
                                         ((helper_port & 0x07u) << 5u) |
                                         ((static_cast<std::uint32_t>(helper_gene) & 0x03u) << 8u));
}

[[nodiscard]] constexpr bool reanchor_r2_pair_hash_valid(ReanchorR2PairHash hash) {
  return (hash & ~kReanchorR2PairHashMask) == 0u;
}

[[nodiscard]] constexpr ReanchorPairHash reanchor_r2_pair_parent_hash(
    ReanchorR2PairHash hash) { return static_cast<ReanchorPairHash>(hash & 0x01u); }
[[nodiscard]] constexpr std::uint32_t reanchor_r2_pair_target_basis(
    ReanchorR2PairHash hash) { return (hash >> 1u) & 0x03u; }
[[nodiscard]] constexpr LocalFounderGene reanchor_r2_pair_target_gene(
    ReanchorR2PairHash hash) { return static_cast<LocalFounderGene>((hash >> 3u) & 0x03u); }
[[nodiscard]] constexpr std::uint32_t reanchor_r2_pair_helper_port(
    ReanchorR2PairHash hash) { return (hash >> 5u) & 0x07u; }
[[nodiscard]] constexpr LocalFounderGene reanchor_r2_pair_helper_gene(
    ReanchorR2PairHash hash) { return static_cast<LocalFounderGene>((hash >> 8u) & 0x03u); }

[[nodiscard]] constexpr std::array<DevelopmentalSeedSite, 6u> reanchor_r2_pair_seed(
    ReanchorR2PairHash hash) {
  const auto parent = reanchor_pair_seed(reanchor_r2_pair_parent_hash(hash));
  const std::uint32_t target_basis = reanchor_r2_pair_target_basis(hash);
  const std::uint32_t helper_port = reanchor_r2_pair_helper_port(hash);
  const Int3 helper_offset = direction_offset(static_cast<Direction>(helper_port));
  const std::uint32_t helper_basis = helper_port & 0x03u;
  return {{parent[0], parent[1], parent[2], parent[3],
           {kReanchorR2PairTargetX, kReanchorR2PairTargetY, kReanchorR2PairTargetZ,
            local_founder_word(reanchor_r2_pair_target_gene(hash), target_basis)},
           {static_cast<std::int8_t>(kReanchorR2PairTargetX + helper_offset.x),
            static_cast<std::int8_t>(kReanchorR2PairTargetY + helper_offset.y),
            static_cast<std::int8_t>(kReanchorR2PairTargetZ + helper_offset.z),
            local_founder_word(reanchor_r2_pair_helper_gene(hash), helper_basis)}}};
}

}  // namespace substrate::bcc32
