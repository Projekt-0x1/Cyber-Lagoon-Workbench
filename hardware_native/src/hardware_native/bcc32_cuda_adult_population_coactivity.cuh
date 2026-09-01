#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace bcc32_cuda_adult_v1 {

__device__ __forceinline__ std::uint32_t resident_surface_mix32(std::uint32_t x) {
  x ^= x >> 16u;
  x *= 0x7feb352du;
  x ^= x >> 15u;
  x *= 0x846ca68bu;
  x ^= x >> 16u;
  return x;
}
// A population is born from the raw surface, but it must not remain a frozen
// spelling address. Each observed adjacency recruits one directional context
// cell on each side. Units repeatedly used before the same resident population
// therefore acquire shared forward matter; units repeatedly used after it
// acquire shared backward matter. The rule carries no word, role, question, or
// host-selected category. Finite replaceable slots make incompatible contexts
// compete instead of creating one universal bus.
__global__ void adapt_resident_population_coactivity_kernel(
    const std::uint32_t* sequence, std::uint32_t sequence_count,
    const std::uint32_t* segment_ids,
    const std::uint32_t* identity_populations,
    std::uint32_t* context_populations, std::uint16_t* context_mass,
    std::uint32_t unit_count,
    std::uint32_t population_width, std::uint32_t population_capacity) {
  if (blockIdx.x != 0u || threadIdx.x != 0u || sequence == nullptr ||
      identity_populations == nullptr || context_populations == nullptr ||
      context_mass == nullptr || sequence_count < 2u || population_width == 0u ||
      population_capacity == 0u)
    return;
  if (population_width < 2u)
    return;
  for (std::uint32_t position = 0u; position + 1u < sequence_count; ++position) {
    if (segment_ids != nullptr &&
        segment_ids[position] != segment_ids[position + 1u])
      continue;
    const std::uint32_t left_unit = sequence[position];
    const std::uint32_t right_unit = sequence[position + 1u];
    if (left_unit >= unit_count || right_unit >= unit_count || left_unit == right_unit)
      continue;
    const std::uint32_t* left_identity =
        identity_populations +
        static_cast<std::size_t>(left_unit) * population_width;
    const std::uint32_t* right_identity =
        identity_populations +
        static_cast<std::size_t>(right_unit) * population_width;
    std::uint32_t* left =
        context_populations +
        static_cast<std::size_t>(left_unit) * population_width;
    std::uint32_t* right =
        context_populations +
        static_cast<std::size_t>(right_unit) * population_width;
    std::uint16_t* left_mass =
        context_mass + static_cast<std::size_t>(left_unit) * population_width;
    std::uint16_t* right_mass =
        context_mass + static_cast<std::size_t>(right_unit) * population_width;
    // Slot zero remains the birth identity.  Learned context may replace only
    // the remaining slots, so a repeated encounter reopens the same bridge
    // rather than wandering as its own prior adaptations accumulate.
    const std::uint32_t left_seed = left_identity[0u];
    const std::uint32_t right_seed = right_identity[0u];
    if (left_seed == 0xffffffffu || right_seed == 0xffffffffu)
      continue;
    const std::uint32_t forward_state =
        resident_surface_mix32(right_seed ^ 0x6d2b79f5u);
    const std::uint32_t backward_state =
        resident_surface_mix32(left_seed ^ 0x1b873593u);
    const std::uint32_t forward_cell =
        resident_surface_mix32(forward_state) % population_capacity;
    const std::uint32_t backward_cell =
        resident_surface_mix32(backward_state) % population_capacity;
    const std::uint32_t left_slot =
        forward_state % population_width;
    const std::uint32_t right_slot =
        backward_state % population_width;
    if (left[left_slot] == forward_cell) {
      if (left_mass[left_slot] != 0xffffu) ++left_mass[left_slot];
    } else if (left_mass[left_slot] == 0u) {
      left[left_slot] = forward_cell;
      left_mass[left_slot] = 1u;
    } else {
      --left_mass[left_slot];
    }
    if (right[right_slot] == backward_cell) {
      if (right_mass[right_slot] != 0xffffu) ++right_mass[right_slot];
    } else if (right_mass[right_slot] == 0u) {
      right[right_slot] = backward_cell;
      right_mass[right_slot] = 1u;
    } else {
      --right_mass[right_slot];
    }
  }
}

}  // namespace bcc32_cuda_adult_v1
