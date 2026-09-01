#pragma once

// Compose the measured E2 rotor-retention gene with the measured remote E3
// source.  The one fixed S4 relabelling maps the retention gene's incoming E2
// lane to physical E3; the compact hash carries only the route length and the
// existing retention-gene word.  Ordinary F decides whether the joint tissue
// preserves any credit-dependent material.

#include <array>
#include <cstdint>

#include "bcc32_e2_rotor_retention_gene.cuh"
#include "bcc32_remote_credit_hole_transducer_seed.cuh"

namespace substrate::bcc32 {

using RemoteCreditE3RetentionSeedHash = std::uint16_t;
constexpr std::uint32_t kRemoteCreditE3RetentionRouteMask = 0x07u;
constexpr std::uint32_t kRemoteCreditE3RetentionGeneShift = 3u;
constexpr BasisPermutation kRemoteCreditE3RetentionPermutation{0u, 1u, 3u, 2u};
constexpr std::size_t kRemoteCreditE3RetentionSeedSiteCount =
    kRemoteCreditHoleTransducerSeedSiteCount + 2u;

constexpr RemoteCreditE3RetentionSeedHash make_remote_credit_e3_retention_seed_hash(
    std::uint32_t route_length, E2RotorRetentionHash retention_hash) {
  return static_cast<RemoteCreditE3RetentionSeedHash>(
      (route_length & kRemoteCreditE3RetentionRouteMask) |
      (static_cast<std::uint32_t>(retention_hash) << kRemoteCreditE3RetentionGeneShift));
}

constexpr RemoteCreditHoleTransducerSeedHash remote_credit_e3_retention_route_hash(
    RemoteCreditE3RetentionSeedHash hash) {
  return static_cast<RemoteCreditHoleTransducerSeedHash>(hash & kRemoteCreditE3RetentionRouteMask);
}

constexpr E2RotorRetentionHash remote_credit_e3_retention_gene_hash(
    RemoteCreditE3RetentionSeedHash hash) {
  return static_cast<E2RotorRetentionHash>(hash >> kRemoteCreditE3RetentionGeneShift);
}

inline Z3Coordinate remote_credit_e3_retention_source() { return {1, 1, 1}; }

inline std::array<DevelopmentalSeedSite, kRemoteCreditE3RetentionSeedSiteCount>
remote_credit_e3_retention_seed(CreditOrbitSeedHash parent, RemoteCreditE3RetentionSeedHash hash,
                                bool prime = true) {
  std::array<DevelopmentalSeedSite, kRemoteCreditE3RetentionSeedSiteCount> result{};
  const auto base = remote_credit_hole_transducer_seed(parent, remote_credit_e3_retention_route_hash(hash));
  for (std::size_t index = 0u; index < base.size(); ++index) result[index] = base[index];
  const auto retention = e2_rotor_retention_seed(remote_credit_e3_retention_gene_hash(hash), prime);
  const Z3Coordinate source = remote_credit_e3_retention_source();
  for (std::size_t index = 0u; index < retention.size(); ++index) {
    const Z3Coordinate rotated = transformed_coordinate(
        Z3Coordinate{retention[index].x, retention[index].y, retention[index].z},
        kRemoteCreditE3RetentionPermutation);
    result[base.size() + index] = {
        static_cast<std::int8_t>(source.x + rotated.x),
        static_cast<std::int8_t>(source.y + rotated.y),
        static_cast<std::int8_t>(source.z + rotated.z),
        transformed_word(retention[index].word, kRemoteCreditE3RetentionPermutation)};
  }
  return result;
}

}  // namespace substrate::bcc32
