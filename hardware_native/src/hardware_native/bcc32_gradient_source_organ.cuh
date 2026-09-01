#pragma once

// Generic source geometry for the gradient-spine search.  This is deliberately
// only a founder description: it contains no destination, semantic label, or
// growth decision.  F remains the sole interpreter of the seeded matter.

#include <array>
#include <cstdint>
#include <vector>

#include "bcc32_developmental_seed.cuh"
#include "bcc32_law.cuh"
#include "bcc32_reference.hpp"

namespace substrate::bcc32 {

using GradientSourceHash = unsigned __int128;

struct GradientSourceGeometry {
  BasisPermutation frame{};
  Int3 origin{};
  std::uint32_t period = 0u;
  std::uint32_t outbound_phase = 0u;
  std::uint32_t return_phase = 0u;
};

constexpr GradientSourceHash make_gradient_source_hash(
    std::uint32_t frame_code, std::uint32_t loop_code,
    std::uint32_t phase_code) {
  return static_cast<GradientSourceHash>(frame_code & 0x1fu) |
         (static_cast<GradientSourceHash>(loop_code & 0xffu) << 8u) |
         (static_cast<GradientSourceHash>(phase_code & 0xffu) << 16u);
}

constexpr BasisPermutation gradient_source_frame(std::uint32_t frame_code) {
  // The founder API keeps the frame explicit, but every emitted frame remains
  // a real S4 permutation. No frame selects a target region.
  constexpr std::array<BasisPermutation, 24u> kFrames{{
      {{0u, 1u, 2u, 3u}}, {{0u, 1u, 3u, 2u}}, {{0u, 2u, 1u, 3u}},
      {{0u, 2u, 3u, 1u}}, {{0u, 3u, 1u, 2u}}, {{0u, 3u, 2u, 1u}},
      {{1u, 0u, 2u, 3u}}, {{1u, 0u, 3u, 2u}}, {{1u, 2u, 0u, 3u}},
      {{1u, 2u, 3u, 0u}}, {{1u, 3u, 0u, 2u}}, {{1u, 3u, 2u, 0u}},
      {{2u, 0u, 1u, 3u}}, {{2u, 0u, 3u, 1u}}, {{2u, 1u, 0u, 3u}},
      {{2u, 1u, 3u, 0u}}, {{2u, 3u, 0u, 1u}}, {{2u, 3u, 1u, 0u}},
      {{3u, 0u, 1u, 2u}}, {{3u, 0u, 2u, 1u}}, {{3u, 1u, 0u, 2u}},
      {{3u, 1u, 2u, 0u}}, {{3u, 2u, 0u, 1u}}, {{3u, 2u, 1u, 0u}}}};
  return kFrames[frame_code % kFrames.size()];
}

constexpr GradientSourceGeometry gradient_source_geometry(
    GradientSourceHash hash) {
  const std::uint32_t frame_code = static_cast<std::uint32_t>(hash & 0xffu);
  const std::uint32_t loop_code = static_cast<std::uint32_t>((hash >> 8u) & 0xffu);
  const std::uint32_t phase_code = static_cast<std::uint32_t>((hash >> 16u) & 0xffu);
  return {gradient_source_frame(frame_code), {0, 0, 0},
          1u + (loop_code & 0x7fu), phase_code & 0x7fu,
          (phase_code >> 1u) & 0x7fu};
}

inline Z3Coordinate gradient_source_outlet(GradientSourceHash hash) {
  const GradientSourceGeometry geometry = gradient_source_geometry(hash);
  const Int3 direction = basis_offset(static_cast<Basis>(geometry.frame[0] & 3u));
  return {direction.x, direction.y, direction.z};
}

inline Z3Coordinate gradient_source_return_inlet(GradientSourceHash hash) {
  const Z3Coordinate outlet = gradient_source_outlet(hash);
  return {-outlet.x, -outlet.y, -outlet.z};
}

constexpr SiteWord gradient_source_word(GradientSourceHash hash,
                                        std::uint32_t phase) {
  const std::uint32_t basis =
      static_cast<std::uint32_t>((hash >> 24u) & 3u);
  SiteWord word = kQ | carrier_bit(basis) | carrier_bit(basis + 4u);
  if ((phase & 1u) != 0u) word |= owned_bond_bit(basis);
  if ((phase & 2u) != 0u) word |= energy_bit(basis);
  return word;
}

inline std::vector<DevelopmentalSeedSite> gradient_source_organ_seed(
    GradientSourceHash hash) {
  const GradientSourceGeometry geometry = gradient_source_geometry(hash);
  const Int3 local = basis_offset(static_cast<Basis>(geometry.frame[1] & 3u));
  const std::uint32_t phase =
      static_cast<std::uint32_t>((hash >> 16u) & 0xffu);
  return {{static_cast<std::int8_t>(geometry.origin.x),
           static_cast<std::int8_t>(geometry.origin.y),
           static_cast<std::int8_t>(geometry.origin.z),
           gradient_source_word(hash, phase)},
          {static_cast<std::int8_t>(geometry.origin.x + local.x),
           static_cast<std::int8_t>(geometry.origin.y + local.y),
           static_cast<std::int8_t>(geometry.origin.z + local.z),
           gradient_source_word(hash, phase >> 2u)}};
}

}  // namespace substrate::bcc32
