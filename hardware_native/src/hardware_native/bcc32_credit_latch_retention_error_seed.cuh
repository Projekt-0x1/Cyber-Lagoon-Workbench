#pragma once

// One compact developmental composition of two measured local mechanisms.
// The C2-latch receiver's native E2 outlet is the rotor location of an
// E2-retention/error molecule.  The hash names only the existing parent hash
// and two BCC-relative ports; neither an E2 pulse nor an error result is
// authored by the host after birth.

#include <array>
#include <cstddef>
#include <cstdint>

#include "bcc32_credit_p6_latch_receiver_seed.cuh"
#include "bcc32_reference.hpp"
#include "bcc32_retention_error_area_seed.cuh"

namespace substrate::bcc32 {

using CreditLatchRetentionErrorSeedHash = unsigned __int128;

constexpr std::uint32_t kCreditLatchRetentionParentBits = 65u;
constexpr std::uint32_t kCreditLatchRetentionRotorShift = kCreditLatchRetentionParentBits;
constexpr std::uint32_t kCreditLatchRetentionCollarShift = kCreditLatchRetentionRotorShift + 3u;
constexpr CreditLatchRetentionErrorSeedHash kCreditLatchRetentionParentMask =
    (CreditLatchRetentionErrorSeedHash{1u} << kCreditLatchRetentionParentBits) - 1u;
constexpr std::size_t kCreditLatchRetentionErrorSeedSiteCount =
    kCreditP6LatchReceiverSeedSiteCount + 9u;

// The latch has already been shown to make this local E2 lane causal.  It is
// an ordinary initially-Q contact in the parent seed, so a rotor may lawfully
// be born there without overlapping parent matter.
inline const Z3Coordinate kCreditLatchRetentionE2Outlet{0, 0, -3};

constexpr CreditLatchRetentionErrorSeedHash make_credit_latch_retention_error_seed_hash(
    CreditP6LatchReceiverSeedHash parent, std::uint32_t rotor_port,
    std::uint32_t collar_port) {
  return (static_cast<CreditLatchRetentionErrorSeedHash>(parent) &
          kCreditLatchRetentionParentMask) |
         (static_cast<CreditLatchRetentionErrorSeedHash>(rotor_port & 0x07u)
          << kCreditLatchRetentionRotorShift) |
         (static_cast<CreditLatchRetentionErrorSeedHash>(collar_port & 0x07u)
          << kCreditLatchRetentionCollarShift);
}

constexpr CreditP6LatchReceiverSeedHash credit_latch_retention_error_parent_hash(
    CreditLatchRetentionErrorSeedHash hash) {
  return static_cast<CreditP6LatchReceiverSeedHash>(hash & kCreditLatchRetentionParentMask);
}

constexpr std::uint32_t credit_latch_retention_error_rotor_port(
    CreditLatchRetentionErrorSeedHash hash) {
  return static_cast<std::uint32_t>((hash >> kCreditLatchRetentionRotorShift) & 0x07u);
}

constexpr std::uint32_t credit_latch_retention_error_collar_port(
    CreditLatchRetentionErrorSeedHash hash) {
  return static_cast<std::uint32_t>((hash >> kCreditLatchRetentionCollarShift) & 0x07u);
}

constexpr RetentionErrorAreaSeedHash credit_latch_retention_error_area_hash(
    CreditLatchRetentionErrorSeedHash hash) {
  return make_retention_error_area_seed_hash(
      kErrorEligibilityAreaSeedHash, credit_latch_retention_error_rotor_port(hash),
      credit_latch_retention_error_collar_port(hash),
      static_cast<E2RotorRetentionGene>(kRetentionR0 | kRetentionFaceNegative0));
}

inline Z3Coordinate credit_latch_retention_error_untranslated_rotor(
    CreditLatchRetentionErrorSeedHash hash) {
  const Int3 offset = direction_offset(
      static_cast<Direction>(credit_latch_retention_error_rotor_port(hash)));
  return {kRetentionErrorAreaReceptorX + offset.x, kRetentionErrorAreaReceptorY + offset.y,
          kRetentionErrorAreaReceptorZ + offset.z};
}

inline Z3Coordinate credit_latch_retention_error_shift(CreditLatchRetentionErrorSeedHash hash) {
  const Z3Coordinate rotor = credit_latch_retention_error_untranslated_rotor(hash);
  return {kCreditLatchRetentionE2Outlet.x - rotor.x, kCreditLatchRetentionE2Outlet.y - rotor.y,
          kCreditLatchRetentionE2Outlet.z - rotor.z};
}

inline Z3Coordinate credit_latch_retention_error_receptor(
    CreditLatchRetentionErrorSeedHash hash) {
  const Z3Coordinate shift = credit_latch_retention_error_shift(hash);
  return {kRetentionErrorAreaReceptorX + shift.x, kRetentionErrorAreaReceptorY + shift.y,
          kRetentionErrorAreaReceptorZ + shift.z};
}

inline Z3Coordinate credit_latch_retention_error_collar(CreditLatchRetentionErrorSeedHash hash) {
  const Int3 offset = direction_offset(
      static_cast<Direction>(credit_latch_retention_error_collar_port(hash)));
  return {kCreditLatchRetentionE2Outlet.x + offset.x, kCreditLatchRetentionE2Outlet.y + offset.y,
          kCreditLatchRetentionE2Outlet.z + offset.z};
}

inline std::array<DevelopmentalSeedSite, kCreditLatchRetentionErrorSeedSiteCount>
credit_latch_retention_error_seed(CreditLatchRetentionErrorSeedHash hash) {
  std::array<DevelopmentalSeedSite, kCreditLatchRetentionErrorSeedSiteCount> result{};
  const auto parent = credit_p6_latch_receiver_seed(credit_latch_retention_error_parent_hash(hash));
  const auto area = retention_error_area_seed(credit_latch_retention_error_area_hash(hash));
  for (std::size_t index = 0u; index < parent.size(); ++index) result[index] = parent[index];
  const Z3Coordinate shift = credit_latch_retention_error_shift(hash);
  for (std::size_t index = 0u; index < area.size(); ++index) {
    result[parent.size() + index] = {
        static_cast<std::int8_t>(area[index].x + shift.x),
        static_cast<std::int8_t>(area[index].y + shift.y),
        static_cast<std::int8_t>(area[index].z + shift.z), area[index].word};
  }
  return result;
}

}  // namespace substrate::bcc32
