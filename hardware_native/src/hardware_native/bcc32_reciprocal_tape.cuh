#pragma once

#include <cstdint>

#include "bcc32_types.cuh"

namespace substrate::bcc32 {

// External tape state is part of the complete reversible boundary. These
// exchanges move represented fields between the world and that tape; they do
// not create, erase, classify, or interpret a signal.
__host__ __device__ constexpr void reciprocal_quantum_exchange(SiteWord& world_word,
                                                               SiteWord world_bit,
                                                               SiteWord& tape_word,
                                                               SiteWord tape_bit) {
  const bool world_set = (world_word & world_bit) != 0u;
  const bool tape_set = (tape_word & tape_bit) != 0u;
  if (world_set != tape_set) {
    world_word ^= world_bit;
    tape_word ^= tape_bit;
  }
}

template <std::uint32_t WorldShift, std::uint32_t TapeShift, std::uint32_t Width>
__host__ __device__ constexpr void reciprocal_field_exchange(SiteWord& world_word,
                                                             SiteWord& tape_word) {
  static_assert(Width > 0u && Width <= 32u);
  static_assert(WorldShift <= 32u - Width);
  static_assert(TapeShift <= 32u - Width);
  constexpr SiteWord field = ~SiteWord{0u} >> (32u - Width);
  constexpr SiteWord world_mask = field << WorldShift;
  constexpr SiteWord tape_mask = field << TapeShift;
  const SiteWord world_payload = (world_word >> WorldShift) & field;
  const SiteWord tape_payload = (tape_word >> TapeShift) & field;
  world_word = (world_word & ~world_mask) | (tape_payload << WorldShift);
  tape_word = (tape_word & ~tape_mask) | (world_payload << TapeShift);
}

}  // namespace substrate::bcc32
