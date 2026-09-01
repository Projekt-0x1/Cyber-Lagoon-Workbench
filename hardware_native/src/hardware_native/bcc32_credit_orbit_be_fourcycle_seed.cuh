#pragma once

// The smallest parity-closed B/E receiver grammar at the measured
// C2-conditioned C3 orbit output.  A B3/P3-hole arm and an E3/P3-hole arm
// leave the output on distinct BCC edges and converge on one initially-vacuum
// site.  The compact hash names geometry and which arm carries which existing
// molecule; F alone decides whether the vacuum becomes resident storage.

#include <array>
#include <cstddef>
#include <cstdint>

#include "bcc32_credit_orbit_seed.cuh"

namespace substrate::bcc32 {

using CreditOrbitBEFourcycleSeedHash = unsigned __int128;
inline constexpr std::uint32_t kCreditOrbitBEFourcycleParentBits = 82u;
inline constexpr std::uint32_t kCreditOrbitBEFourcycleArmAShift =
    kCreditOrbitBEFourcycleParentBits;
inline constexpr std::uint32_t kCreditOrbitBEFourcycleArmBShift =
    kCreditOrbitBEFourcycleArmAShift + 3u;
inline constexpr std::uint32_t kCreditOrbitBEFourcycleAssignmentShift =
    kCreditOrbitBEFourcycleArmBShift + 3u;
inline constexpr CreditOrbitBEFourcycleSeedHash kCreditOrbitBEFourcycleParentMask =
    (CreditOrbitBEFourcycleSeedHash{1u} << kCreditOrbitBEFourcycleParentBits) - 1u;
inline constexpr std::size_t kCreditOrbitBEFourcycleSeedSiteCount =
    kCreditOrbitSeedSiteCount + 2u;
inline constexpr std::uint32_t kCreditOrbitBEFourcycleBasis = 3u;

constexpr CreditOrbitBEFourcycleSeedHash make_credit_orbit_be_fourcycle_seed_hash(
    CreditOrbitSeedHash parent, std::uint32_t arm_a_direction, std::uint32_t arm_b_direction,
    bool swap_assignment) {
  return (static_cast<CreditOrbitBEFourcycleSeedHash>(parent) &
          kCreditOrbitBEFourcycleParentMask) |
         (static_cast<CreditOrbitBEFourcycleSeedHash>(arm_a_direction & 0x07u)
          << kCreditOrbitBEFourcycleArmAShift) |
         (static_cast<CreditOrbitBEFourcycleSeedHash>(arm_b_direction & 0x07u)
          << kCreditOrbitBEFourcycleArmBShift) |
         (static_cast<CreditOrbitBEFourcycleSeedHash>(swap_assignment ? 1u : 0u)
          << kCreditOrbitBEFourcycleAssignmentShift);
}

constexpr CreditOrbitSeedHash credit_orbit_be_fourcycle_parent_hash(
    CreditOrbitBEFourcycleSeedHash hash) {
  return static_cast<CreditOrbitSeedHash>(hash & kCreditOrbitBEFourcycleParentMask);
}
constexpr std::uint32_t credit_orbit_be_fourcycle_arm_a_direction(
    CreditOrbitBEFourcycleSeedHash hash) {
  return static_cast<std::uint32_t>((hash >> kCreditOrbitBEFourcycleArmAShift) & 0x07u);
}
constexpr std::uint32_t credit_orbit_be_fourcycle_arm_b_direction(
    CreditOrbitBEFourcycleSeedHash hash) {
  return static_cast<std::uint32_t>((hash >> kCreditOrbitBEFourcycleArmBShift) & 0x07u);
}
constexpr bool credit_orbit_be_fourcycle_swaps_assignment(CreditOrbitBEFourcycleSeedHash hash) {
  return ((hash >> kCreditOrbitBEFourcycleAssignmentShift) & 1u) != 0u;
}

inline Z3Coordinate credit_orbit_be_fourcycle_output(CreditOrbitBEFourcycleSeedHash hash) {
  const auto parent = credit_orbit_seed(credit_orbit_be_fourcycle_parent_hash(hash));
  const DevelopmentalSeedSite source = parent[kCreditBudReceiverSeedSiteCount];
  return {source.x, source.y, source.z};
}
inline Z3Coordinate credit_orbit_be_fourcycle_arm_a(CreditOrbitBEFourcycleSeedHash hash) {
  const Z3Coordinate output = credit_orbit_be_fourcycle_output(hash);
  const Int3 offset = direction_offset(
      static_cast<Direction>(credit_orbit_be_fourcycle_arm_a_direction(hash)));
  return {output.x + offset.x, output.y + offset.y, output.z + offset.z};
}
inline Z3Coordinate credit_orbit_be_fourcycle_arm_b(CreditOrbitBEFourcycleSeedHash hash) {
  const Z3Coordinate output = credit_orbit_be_fourcycle_output(hash);
  const Int3 offset = direction_offset(
      static_cast<Direction>(credit_orbit_be_fourcycle_arm_b_direction(hash)));
  return {output.x + offset.x, output.y + offset.y, output.z + offset.z};
}
inline Z3Coordinate credit_orbit_be_fourcycle_register(CreditOrbitBEFourcycleSeedHash hash) {
  const Z3Coordinate arm_a = credit_orbit_be_fourcycle_arm_a(hash);
  const Int3 arm_b_offset = direction_offset(
      static_cast<Direction>(credit_orbit_be_fourcycle_arm_b_direction(hash)));
  return {arm_a.x + arm_b_offset.x, arm_a.y + arm_b_offset.y, arm_a.z + arm_b_offset.z};
}

inline constexpr SiteWord kCreditOrbitBEFourcycleBondWord =
    static_cast<SiteWord>((kQ & ~carrier_bit(kCreditOrbitBEFourcycleBasis)) |
                          owned_bond_bit(kCreditOrbitBEFourcycleBasis));
inline constexpr SiteWord kCreditOrbitBEFourcycleEnergyWord =
    static_cast<SiteWord>((kQ & ~carrier_bit(kCreditOrbitBEFourcycleBasis)) |
                          energy_bit(kCreditOrbitBEFourcycleBasis));

inline std::array<DevelopmentalSeedSite, kCreditOrbitBEFourcycleSeedSiteCount>
credit_orbit_be_fourcycle_seed(CreditOrbitBEFourcycleSeedHash hash) {
  std::array<DevelopmentalSeedSite, kCreditOrbitBEFourcycleSeedSiteCount> result{};
  const auto parent = credit_orbit_seed(credit_orbit_be_fourcycle_parent_hash(hash));
  for (std::size_t index = 0u; index < parent.size(); ++index) result[index] = parent[index];
  const Z3Coordinate arm_a = credit_orbit_be_fourcycle_arm_a(hash);
  const Z3Coordinate arm_b = credit_orbit_be_fourcycle_arm_b(hash);
  const bool swap = credit_orbit_be_fourcycle_swaps_assignment(hash);
  result[parent.size()] = {static_cast<std::int8_t>(arm_a.x), static_cast<std::int8_t>(arm_a.y),
                           static_cast<std::int8_t>(arm_a.z),
                           swap ? kCreditOrbitBEFourcycleEnergyWord
                                : kCreditOrbitBEFourcycleBondWord};
  result[parent.size() + 1u] = {
      static_cast<std::int8_t>(arm_b.x), static_cast<std::int8_t>(arm_b.y),
      static_cast<std::int8_t>(arm_b.z),
      swap ? kCreditOrbitBEFourcycleBondWord : kCreditOrbitBEFourcycleEnergyWord};
  return result;
}

inline SiteWord credit_orbit_be_fourcycle_arm_a_lane(CreditOrbitBEFourcycleSeedHash hash) {
  return credit_orbit_be_fourcycle_swaps_assignment(hash)
             ? energy_bit(kCreditOrbitBEFourcycleBasis)
             : owned_bond_bit(kCreditOrbitBEFourcycleBasis);
}
inline SiteWord credit_orbit_be_fourcycle_arm_b_lane(CreditOrbitBEFourcycleSeedHash hash) {
  return credit_orbit_be_fourcycle_swaps_assignment(hash)
             ? owned_bond_bit(kCreditOrbitBEFourcycleBasis)
             : energy_bit(kCreditOrbitBEFourcycleBasis);
}

}  // namespace substrate::bcc32
