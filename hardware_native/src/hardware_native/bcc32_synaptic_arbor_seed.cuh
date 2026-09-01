#pragma once

// Compact gene for a convergent four-contact BCC synaptic arbor.
//
// The low word selects one delayed-credit contact gene. Four high bits select
// which tetrahedral precursor branches exist. The birth interpreter applies
// the same S4-covariant local gene on four rays whose downstream observations
// meet at one coordinate. It contains no weight values, training history,
// truth table, episode counter, target answer, or host update callback.
// After birth, canonical reversible F is the only runtime interpreter.

#include <array>
#include <cstddef>
#include <cstdint>

#include "bcc32_credit_backcarry_seed.cuh"

namespace substrate::bcc32 {

using SynapticArborSeedHash = std::uint64_t;

inline constexpr std::uint32_t kSynapticArborContactShift = 32u;
inline constexpr std::uint32_t kSynapticArborContactMask = 0x0fu;
inline constexpr std::size_t kSynapticArborBranchCount = 4u;
inline constexpr std::int32_t kSynapticArborReach = 15;
inline constexpr std::size_t kSynapticArborSeedSiteCount =
    kSynapticArborBranchCount * kCreditBackcarrySeedSiteCount;

using ArborBasisPermutation = std::array<std::uint8_t, 4>;
using ArborCoordinate = std::array<std::int32_t, 3>;

constexpr SynapticArborSeedHash make_synaptic_arbor_seed_hash(CreditBackcarrySeedHash contact_hash,
                                                              std::uint8_t contact_mask) {
  return static_cast<SynapticArborSeedHash>(contact_hash) |
         (static_cast<SynapticArborSeedHash>(contact_mask & kSynapticArborContactMask)
          << kSynapticArborContactShift);
}

constexpr CreditBackcarrySeedHash synaptic_arbor_contact_hash(SynapticArborSeedHash hash) {
  return static_cast<CreditBackcarrySeedHash>(hash & 0xffffffffull);
}

constexpr std::uint8_t synaptic_arbor_contact_mask(SynapticArborSeedHash hash) {
  return static_cast<std::uint8_t>((hash >> kSynapticArborContactShift) &
                                   kSynapticArborContactMask);
}

// One branch approaches from each BCC tetrahedral ray. These are basis
// relabellings of one gene, not four separately authored contact recipes.
constexpr ArborBasisPermutation synaptic_arbor_branch_permutation(std::size_t branch) {
  switch (branch & 3u) {
    case 0u:
      return {0u, 1u, 2u, 3u};
    case 1u:
      return {3u, 1u, 2u, 0u};
    case 2u:
      return {0u, 3u, 2u, 1u};
    default:
      return {0u, 1u, 3u, 2u};
  }
}

constexpr ArborCoordinate synaptic_arbor_basis_offset(std::uint8_t basis) {
  switch (basis & 3u) {
    case 0u:
      return {1, 0, 0};
    case 1u:
      return {0, 1, 0};
    case 2u:
      return {0, 0, 1};
    default:
      return {-1, -1, -1};
  }
}

constexpr ArborCoordinate synaptic_arbor_transform_coordinate(ArborCoordinate coordinate,
                                                              ArborBasisPermutation permutation) {
  ArborCoordinate result{0, 0, 0};
  for (std::size_t axis = 0u; axis < 3u; ++axis) {
    const ArborCoordinate image = synaptic_arbor_basis_offset(permutation[axis]);
    for (std::size_t component = 0u; component < 3u; ++component)
      result[component] += coordinate[axis] * image[component];
  }
  return result;
}

constexpr SiteWord synaptic_arbor_transform_word(SiteWord word, ArborBasisPermutation permutation) {
  SiteWord result = 0u;
  for (std::uint32_t basis = 0u; basis < 4u; ++basis) {
    const std::uint32_t image = permutation[basis];
    if ((word & carrier_bit(basis)) != 0u)
      result |= carrier_bit(image);
    if ((word & carrier_bit(basis + 4u)) != 0u)
      result |= carrier_bit(image + 4u);
    if ((word & face_bit(basis)) != 0u)
      result |= face_bit(image);
    if ((word & face_bit(basis + 4u)) != 0u)
      result |= face_bit(image + 4u);
    if ((word & owned_bond_bit(basis)) != 0u)
      result |= owned_bond_bit(image);
    if ((word & channel_bit(kConformationShift, basis)) != 0u)
      result |= channel_bit(kConformationShift, image);
    if ((word & channel_bit(kReactiveShift, basis)) != 0u)
      result |= channel_bit(kReactiveShift, image);
    if ((word & energy_bit(basis)) != 0u)
      result |= energy_bit(image);
  }
  return result;
}

constexpr ArborCoordinate synaptic_arbor_branch_origin(std::size_t branch) {
  const ArborCoordinate transformed_target = synaptic_arbor_transform_coordinate(
      {kSynapticArborReach, kSynapticArborReach, kSynapticArborReach},
      synaptic_arbor_branch_permutation(branch));
  return {-transformed_target[0], -transformed_target[1], -transformed_target[2]};
}

constexpr ArborCoordinate synaptic_arbor_branch_coordinate(std::size_t branch,
                                                           ArborCoordinate local_coordinate) {
  const ArborCoordinate transformed = synaptic_arbor_transform_coordinate(
      local_coordinate, synaptic_arbor_branch_permutation(branch));
  const ArborCoordinate origin = synaptic_arbor_branch_origin(branch);
  return {transformed[0] + origin[0], transformed[1] + origin[1], transformed[2] + origin[2]};
}

constexpr std::array<DevelopmentalSeedSite, kSynapticArborSeedSiteCount> synaptic_arbor_seed(
    SynapticArborSeedHash hash) {
  std::array<DevelopmentalSeedSite, kSynapticArborSeedSiteCount> result{};
  const auto contact = credit_backcarry_seed(synaptic_arbor_contact_hash(hash));
  const std::uint8_t mask = synaptic_arbor_contact_mask(hash);
  std::size_t cursor = 0u;
  for (std::size_t branch = 0u; branch < kSynapticArborBranchCount; ++branch) {
    for (const DevelopmentalSeedSite& site : contact) {
      const ArborCoordinate coordinate =
          synaptic_arbor_branch_coordinate(branch, {site.x, site.y, site.z});
      result[cursor++] = {
          static_cast<std::int8_t>(coordinate[0]), static_cast<std::int8_t>(coordinate[1]),
          static_cast<std::int8_t>(coordinate[2]),
          (mask & (1u << branch)) != 0u
              ? synaptic_arbor_transform_word(site.word, synaptic_arbor_branch_permutation(branch))
              : kQ};
    }
  }
  return result;
}

inline constexpr SynapticArborSeedHash kSynapticArborSeedHash =
    make_synaptic_arbor_seed_hash(kCreditBackcarrySeedHash, 0x0fu);

static_assert(kSynapticArborSeedHash == 0x0000000f10c10255ull);
static_assert(synaptic_arbor_contact_hash(kSynapticArborSeedHash) == kCreditBackcarrySeedHash);
static_assert(synaptic_arbor_contact_mask(kSynapticArborSeedHash) == 0x0fu);
static_assert(synaptic_arbor_branch_origin(0u) == ArborCoordinate{-15, -15, -15});
static_assert(synaptic_arbor_branch_origin(1u) == ArborCoordinate{15, 0, 0});
static_assert(synaptic_arbor_branch_origin(2u) == ArborCoordinate{0, 15, 0});
static_assert(synaptic_arbor_branch_origin(3u) == ArborCoordinate{0, 0, 15});

}  // namespace substrate::bcc32
