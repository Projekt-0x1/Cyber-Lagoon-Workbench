#pragma once

// The smallest edge-local B/E escrow grammar at the measured C2-conditioned
// C3 orbit output.  Both additions are occupancy-neutral relative to Q: the
// donor owns B3 while carrying a P3 hole; the acceptor owns E3 while carrying
// its own P3 hole.  The compact hash chooses only the two adjacent BCC edges.

#include <array>
#include <cstddef>
#include <cstdint>

#include "bcc32_credit_orbit_seed.cuh"

namespace substrate::bcc32 {

using CreditOrbitBEPairSeedHash = unsigned __int128;
constexpr std::uint32_t kCreditOrbitBEPairParentBits = 82u;
constexpr std::uint32_t kCreditOrbitBEPairDonorShift = kCreditOrbitBEPairParentBits;
constexpr std::uint32_t kCreditOrbitBEPairAcceptorShift = kCreditOrbitBEPairDonorShift + 3u;
constexpr CreditOrbitBEPairSeedHash kCreditOrbitBEPairParentMask =
    (CreditOrbitBEPairSeedHash{1u} << kCreditOrbitBEPairParentBits) - 1u;
constexpr std::size_t kCreditOrbitBEPairSeedSiteCount = kCreditOrbitSeedSiteCount + 2u;
constexpr std::uint32_t kCreditOrbitBEPairBasis = 3u;

constexpr CreditOrbitBEPairSeedHash make_credit_orbit_be_pair_seed_hash(
    CreditOrbitSeedHash parent, std::uint32_t donor_direction, std::uint32_t acceptor_direction) {
  return (static_cast<CreditOrbitBEPairSeedHash>(parent) & kCreditOrbitBEPairParentMask) |
         (static_cast<CreditOrbitBEPairSeedHash>(donor_direction & 0x07u)
          << kCreditOrbitBEPairDonorShift) |
         (static_cast<CreditOrbitBEPairSeedHash>(acceptor_direction & 0x07u)
          << kCreditOrbitBEPairAcceptorShift);
}

constexpr CreditOrbitSeedHash credit_orbit_be_pair_parent_hash(CreditOrbitBEPairSeedHash hash) {
  return static_cast<CreditOrbitSeedHash>(hash & kCreditOrbitBEPairParentMask);
}
constexpr std::uint32_t credit_orbit_be_pair_donor_direction(CreditOrbitBEPairSeedHash hash) {
  return static_cast<std::uint32_t>((hash >> kCreditOrbitBEPairDonorShift) & 0x07u);
}
constexpr std::uint32_t credit_orbit_be_pair_acceptor_direction(CreditOrbitBEPairSeedHash hash) {
  return static_cast<std::uint32_t>((hash >> kCreditOrbitBEPairAcceptorShift) & 0x07u);
}

inline Z3Coordinate credit_orbit_be_pair_output(CreditOrbitBEPairSeedHash hash) {
  const auto parent = credit_orbit_seed(credit_orbit_be_pair_parent_hash(hash));
  const DevelopmentalSeedSite source = parent[kCreditBudReceiverSeedSiteCount];
  return {source.x, source.y, source.z};
}
inline Z3Coordinate credit_orbit_be_pair_donor(CreditOrbitBEPairSeedHash hash) {
  const Z3Coordinate output = credit_orbit_be_pair_output(hash);
  const Int3 offset = direction_offset(
      static_cast<Direction>(credit_orbit_be_pair_donor_direction(hash)));
  return {output.x + offset.x, output.y + offset.y, output.z + offset.z};
}
inline Z3Coordinate credit_orbit_be_pair_acceptor(CreditOrbitBEPairSeedHash hash) {
  const Z3Coordinate donor = credit_orbit_be_pair_donor(hash);
  const Int3 offset = direction_offset(
      static_cast<Direction>(credit_orbit_be_pair_acceptor_direction(hash)));
  return {donor.x + offset.x, donor.y + offset.y, donor.z + offset.z};
}

inline constexpr SiteWord kCreditOrbitBEPairDonorWord =
    static_cast<SiteWord>((kQ & ~carrier_bit(kCreditOrbitBEPairBasis)) |
                          owned_bond_bit(kCreditOrbitBEPairBasis));
inline constexpr SiteWord kCreditOrbitBEPairAcceptorWord =
    static_cast<SiteWord>((kQ & ~carrier_bit(kCreditOrbitBEPairBasis)) |
                          energy_bit(kCreditOrbitBEPairBasis));

inline std::array<DevelopmentalSeedSite, kCreditOrbitBEPairSeedSiteCount>
credit_orbit_be_pair_seed(CreditOrbitBEPairSeedHash hash) {
  std::array<DevelopmentalSeedSite, kCreditOrbitBEPairSeedSiteCount> result{};
  const auto parent = credit_orbit_seed(credit_orbit_be_pair_parent_hash(hash));
  for (std::size_t index = 0u; index < parent.size(); ++index) result[index] = parent[index];
  const Z3Coordinate donor = credit_orbit_be_pair_donor(hash);
  const Z3Coordinate acceptor = credit_orbit_be_pair_acceptor(hash);
  result[parent.size()] = {static_cast<std::int8_t>(donor.x), static_cast<std::int8_t>(donor.y),
                           static_cast<std::int8_t>(donor.z), kCreditOrbitBEPairDonorWord};
  result[parent.size() + 1u] = {static_cast<std::int8_t>(acceptor.x),
                                static_cast<std::int8_t>(acceptor.y),
                                static_cast<std::int8_t>(acceptor.z),
                                kCreditOrbitBEPairAcceptorWord};
  return result;
}

}  // namespace substrate::bcc32
