#pragma once

// Compact molecular interface seed: E2 retention beside an error nucleus.
//
// The hash carries only founder material selections and local BCC directions.
// It is decoded once at genesis into nine ordinary sites.  The error receptor
// itself remains exclusively a product of the error nucleus seed; neither the
// retention rotor nor its collar is injected there.  F is the sole runtime.

#include <array>
#include <cstdint>

#include "bcc32_e2_rotor_retention_gene.cuh"
#include "bcc32_local_area_seed.cuh"

namespace substrate::bcc32 {

using RetentionErrorAreaSeedHash = std::uint16_t;
constexpr std::uint32_t kRetentionErrorErrorShift = 0u;
constexpr std::uint32_t kRetentionErrorRotorPortShift = 3u;
constexpr std::uint32_t kRetentionErrorCollarPortShift = 6u;
constexpr std::uint32_t kRetentionErrorGeneShift = 9u;

constexpr RetentionErrorAreaSeedHash make_retention_error_area_seed_hash(
    ErrorEligibilityAreaSeedHash error_hash, std::uint32_t rotor_port,
    std::uint32_t collar_port, E2RotorRetentionGene gene) {
  return static_cast<RetentionErrorAreaSeedHash>(
      (static_cast<RetentionErrorAreaSeedHash>(error_hash) << kRetentionErrorErrorShift) |
      ((rotor_port & 0x07u) << kRetentionErrorRotorPortShift) |
      ((collar_port & 0x07u) << kRetentionErrorCollarPortShift) |
      ((static_cast<RetentionErrorAreaSeedHash>(gene) & kRetentionPaletteMask)
       << kRetentionErrorGeneShift));
}

constexpr ErrorEligibilityAreaSeedHash retention_error_area_error_hash(
    RetentionErrorAreaSeedHash hash) {
  return static_cast<ErrorEligibilityAreaSeedHash>((hash >> kRetentionErrorErrorShift) & 0x07u);
}
constexpr std::uint32_t retention_error_area_rotor_port(RetentionErrorAreaSeedHash hash) {
  return (hash >> kRetentionErrorRotorPortShift) & 0x07u;
}
constexpr std::uint32_t retention_error_area_collar_port(RetentionErrorAreaSeedHash hash) {
  return (hash >> kRetentionErrorCollarPortShift) & 0x07u;
}
constexpr E2RotorRetentionGene retention_error_area_gene(RetentionErrorAreaSeedHash hash) {
  return static_cast<E2RotorRetentionGene>(
      (hash >> kRetentionErrorGeneShift) & kRetentionPaletteMask);
}

constexpr std::int32_t kRetentionErrorAreaReceptorX = -1;
constexpr std::int32_t kRetentionErrorAreaReceptorY = -1;
constexpr std::int32_t kRetentionErrorAreaReceptorZ = 1;
constexpr RetentionErrorAreaSeedHash kRetentionErrorAreaSeedHash =
    make_retention_error_area_seed_hash(
        kErrorEligibilityAreaSeedHash, 0u, 1u,
        static_cast<E2RotorRetentionGene>(kRetentionR0 | kRetentionFaceNegative0));

constexpr std::array<DevelopmentalSeedSite, 9u> retention_error_area_seed(
    RetentionErrorAreaSeedHash hash) {
  std::array<DevelopmentalSeedSite, 9u> result{};
  const auto error = error_eligibility_area_seed(retention_error_area_error_hash(hash));
  for (std::size_t index = 0u; index < error.size(); ++index) result[index] = error[index];

  const Int3 rotor_offset =
      direction_offset(static_cast<Direction>(retention_error_area_rotor_port(hash)));
  const std::int32_t rotor_x = kRetentionErrorAreaReceptorX + rotor_offset.x;
  const std::int32_t rotor_y = kRetentionErrorAreaReceptorY + rotor_offset.y;
  const std::int32_t rotor_z = kRetentionErrorAreaReceptorZ + rotor_offset.z;
  const auto retention = e2_rotor_retention_seed(retention_error_area_collar_port(hash),
                                                  retention_error_area_gene(hash));
  for (std::size_t index = 0u; index < retention.size(); ++index) {
    result[error.size() + index] = {
        static_cast<std::int8_t>(retention[index].x + rotor_x),
        static_cast<std::int8_t>(retention[index].y + rotor_y),
        static_cast<std::int8_t>(retention[index].z + rotor_z), retention[index].word};
  }
  return result;
}

constexpr std::uint64_t retention_error_area_seed_fingerprint() {
  std::uint64_t hash = 14695981039346656037ull;
  const auto add = [&hash](std::uint8_t byte) constexpr {
    hash ^= byte;
    hash *= 1099511628211ull;
  };
  for (const DevelopmentalSeedSite& site :
       retention_error_area_seed(kRetentionErrorAreaSeedHash)) {
    add(static_cast<std::uint8_t>(site.x));
    add(static_cast<std::uint8_t>(site.y));
    add(static_cast<std::uint8_t>(site.z));
    for (std::uint32_t shift = 0u; shift < 32u; shift += 8u)
      add(static_cast<std::uint8_t>(site.word >> shift));
  }
  return hash;
}

inline constexpr std::uint64_t kRetentionErrorAreaSeedFingerprint =
    retention_error_area_seed_fingerprint();

}  // namespace substrate::bcc32
