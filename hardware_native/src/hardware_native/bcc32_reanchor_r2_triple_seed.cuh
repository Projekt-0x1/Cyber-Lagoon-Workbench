#pragma once

// Three-founder grammar with developmental support separated from the local
// molecular output.  The word selects only B/E founder lanes; no hash bit
// writes R2, an activity hole, or E2 output.

#include <array>
#include <cstdint>

#include "bcc32_local_founder_hash.cuh"
#include "bcc32_reanchor_pair_gene.cuh"

namespace substrate::bcc32 {
using ReanchorR2TripleHash = std::uint16_t;
constexpr ReanchorR2TripleHash kReanchorR2TripleHashMask = 0x7fffu;
constexpr std::int8_t kReanchorR2TripleTargetX = 0;
constexpr std::int8_t kReanchorR2TripleTargetY = 0;
constexpr std::int8_t kReanchorR2TripleTargetZ = -2;

[[nodiscard]] constexpr ReanchorR2TripleHash make_reanchor_r2_triple_hash(
    ReanchorPairHash parent, std::uint32_t target_basis, LocalFounderGene target_gene,
    std::uint32_t receptor_port, LocalFounderGene receptor_gene, std::uint32_t support_port,
    LocalFounderGene support_gene) {
  return static_cast<ReanchorR2TripleHash>(
      (parent & 0x01u) | ((target_basis & 0x03u) << 1u) |
      ((static_cast<std::uint32_t>(target_gene) & 0x03u) << 3u) | ((receptor_port & 0x07u) << 5u) |
      ((static_cast<std::uint32_t>(receptor_gene) & 0x03u) << 8u) |
      ((support_port & 0x07u) << 10u) |
      ((static_cast<std::uint32_t>(support_gene) & 0x03u) << 13u));
}
[[nodiscard]] constexpr bool reanchor_r2_triple_hash_valid(ReanchorR2TripleHash hash) {
  return (hash & ~kReanchorR2TripleHashMask) == 0u;
}
[[nodiscard]] constexpr ReanchorPairHash reanchor_r2_triple_parent_hash(ReanchorR2TripleHash hash) {
  return static_cast<ReanchorPairHash>(hash & 0x01u);
}
[[nodiscard]] constexpr std::uint32_t reanchor_r2_triple_target_basis(ReanchorR2TripleHash hash) {
  return (hash >> 1u) & 0x03u;
}
[[nodiscard]] constexpr LocalFounderGene reanchor_r2_triple_target_gene(ReanchorR2TripleHash hash) {
  return static_cast<LocalFounderGene>((hash >> 3u) & 0x03u);
}
[[nodiscard]] constexpr std::uint32_t reanchor_r2_triple_receptor_port(ReanchorR2TripleHash hash) {
  return (hash >> 5u) & 0x07u;
}
[[nodiscard]] constexpr LocalFounderGene reanchor_r2_triple_receptor_gene(
    ReanchorR2TripleHash hash) {
  return static_cast<LocalFounderGene>((hash >> 8u) & 0x03u);
}
[[nodiscard]] constexpr std::uint32_t reanchor_r2_triple_support_port(ReanchorR2TripleHash hash) {
  return (hash >> 10u) & 0x07u;
}
[[nodiscard]] constexpr LocalFounderGene reanchor_r2_triple_support_gene(
    ReanchorR2TripleHash hash) {
  return static_cast<LocalFounderGene>((hash >> 13u) & 0x03u);
}

[[nodiscard]] constexpr std::array<DevelopmentalSeedSite, 7u> reanchor_r2_triple_seed(
    ReanchorR2TripleHash hash) {
  const auto parent = reanchor_pair_seed(reanchor_r2_triple_parent_hash(hash));
  const std::uint32_t receptor_port = reanchor_r2_triple_receptor_port(hash);
  const std::uint32_t support_port = reanchor_r2_triple_support_port(hash);
  const Int3 receptor_offset = direction_offset(static_cast<Direction>(receptor_port));
  const Int3 support_offset = direction_offset(static_cast<Direction>(support_port));
  return {{parent[0],
           parent[1],
           parent[2],
           parent[3],
           {kReanchorR2TripleTargetX, kReanchorR2TripleTargetY, kReanchorR2TripleTargetZ,
            local_founder_word(reanchor_r2_triple_target_gene(hash),
                               reanchor_r2_triple_target_basis(hash))},
           {static_cast<std::int8_t>(kReanchorR2TripleTargetX + receptor_offset.x),
            static_cast<std::int8_t>(kReanchorR2TripleTargetY + receptor_offset.y),
            static_cast<std::int8_t>(kReanchorR2TripleTargetZ + receptor_offset.z),
            local_founder_word(reanchor_r2_triple_receptor_gene(hash), receptor_port & 0x03u)},
           {static_cast<std::int8_t>(kReanchorR2TripleTargetX + support_offset.x),
            static_cast<std::int8_t>(kReanchorR2TripleTargetY + support_offset.y),
            static_cast<std::int8_t>(kReanchorR2TripleTargetZ + support_offset.z),
            local_founder_word(reanchor_r2_triple_support_gene(hash), support_port & 0x03u)}}};
}
}  // namespace substrate::bcc32
