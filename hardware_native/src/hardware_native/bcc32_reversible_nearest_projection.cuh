#pragma once

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

namespace bcc32::reversible_projection {

struct DualRailWord {
  std::uint32_t zero;
  std::uint32_t one;
};

// Observer-only diagnostics. Receipts are not part of the reversible physical
// state and make no conservation or inverse claim.
struct ProjectionReceipt {
  std::uint32_t admitted;
  std::uint32_t invalid_instances;
  std::uint32_t selected_prototype;
  std::uint32_t tied;
};

__host__ __device__ inline DualRailWord encode(std::uint32_t value) {
  return DualRailWord{~value, value};
}

__host__ __device__ inline bool valid(DualRailWord value) {
  return (value.zero & value.one) == 0u &&
         (value.zero | value.one) == 0xffffffffu;
}

__host__ __device__ inline void xor_into(DualRailWord& target,
                                         std::uint32_t control) {
  const std::uint32_t old_zero = target.zero;
  target.zero = (target.zero & ~control) | (target.one & control);
  target.one = (target.one & ~control) | (old_zero & control);
}

__host__ __device__ inline void swap_words(DualRailWord& left,
                                           DualRailWord& right) {
  const DualRailWord temporary = left;
  left = right;
  right = temporary;
}

__host__ __device__ inline std::uint32_t majority_five(
    std::uint32_t a,
    std::uint32_t b,
    std::uint32_t c,
    std::uint32_t d,
    std::uint32_t e) {
  return (a & b & (c | d | e)) |
         (a & c & (d | e)) |
         (a & d & e) |
         (b & c & (d | e)) |
         (b & d & e) |
         (c & d & e);
}

// prototype and exposures must not overlap.
__global__ void derive_majority_five_kernel(
    DualRailWord* prototype,
    const DualRailWord* exposures,
    std::uint32_t word_count) {
  const std::size_t total_words = static_cast<std::size_t>(word_count);
  const std::size_t stride =
      static_cast<std::size_t>(blockDim.x) *
      static_cast<std::size_t>(gridDim.x);
  for (std::size_t word =
           static_cast<std::size_t>(blockIdx.x) *
               static_cast<std::size_t>(blockDim.x) +
           static_cast<std::size_t>(threadIdx.x);
       word < total_words;
       word += stride) {
    const std::uint32_t learned = majority_five(
        exposures[word].one,
        exposures[total_words + word].one,
        exposures[2u * total_words + word].one,
        exposures[3u * total_words + word].one,
        exposures[4u * total_words + word].one);
    xor_into(prototype[word], learned);
  }
}

// Preconditions: cognitive, prototypes, history, and branch are disjoint.
//
// Forward with clear history and branch rails:
//   history ^= cognitive; history ^= basin; cognitive ^= history.
//   branch ^= selected_prototype + 1.
// The visible cognitive rail becomes the selected prototype, history retains
// the exact cue residual, and branch records which prototype participated.
// Reverse consumes that physical branch record rather than recomputing a
// nearest-neighbour decision from mutated state.
__global__ void project_batch_kernel(
    DualRailWord* cognitive,
    const DualRailWord* prototypes,
    DualRailWord* history,
    DualRailWord* branch,
    ProjectionReceipt* receipts,
    std::uint32_t cue_count,
    std::uint32_t word_count,
    std::uint32_t prototype_count,
    bool reverse) {
  const std::uint32_t cue = blockIdx.x;
  if (cue >= cue_count || threadIdx.x != 0u) return;

  const std::size_t cue_offset =
      static_cast<std::size_t>(cue) * static_cast<std::size_t>(word_count);
  DualRailWord* cue_words = cognitive + cue_offset;
  DualRailWord* history_words = history + cue_offset;
  std::uint32_t invalid_instances = 0u;
  for (std::uint32_t word = 0u; word < word_count; ++word) {
    const bool rails_valid =
        valid(cue_words[word]) && valid(history_words[word]);
    const bool sink_clear = reverse || history_words[word].one == 0u;
    if (!rails_valid || !sink_clear) ++invalid_instances;
    for (std::uint32_t prototype = 0u;
         prototype < prototype_count;
         ++prototype) {
      const std::size_t prototype_offset =
          static_cast<std::size_t>(prototype) *
          static_cast<std::size_t>(word_count);
      if (!valid(prototypes[prototype_offset + word])) {
        ++invalid_instances;
      }
    }
  }
  if (!valid(branch[cue])) ++invalid_instances;

  std::uint32_t selected_prototype = 0xffffffffu;
  bool tied = false;
  if (reverse) {
    const std::uint32_t branch_code = branch[cue].one;
    if (branch_code == 0u || branch_code > prototype_count) {
      ++invalid_instances;
    } else {
      selected_prototype = branch_code - 1u;
    }
  } else {
    if (branch[cue].one != 0u) ++invalid_instances;
    std::uint64_t best_distance = ~std::uint64_t{0};
    for (std::uint32_t prototype = 0u;
         prototype < prototype_count;
         ++prototype) {
      const std::size_t prototype_offset =
          static_cast<std::size_t>(prototype) *
          static_cast<std::size_t>(word_count);
      std::uint64_t distance = 0u;
      for (std::uint32_t word = 0u; word < word_count; ++word) {
        distance += static_cast<std::uint64_t>(
            __popc(cue_words[word].one ^
                   prototypes[prototype_offset + word].one));
      }
      if (distance < best_distance) {
        best_distance = distance;
        selected_prototype = prototype;
        tied = false;
      } else if (distance == best_distance) {
        tied = true;
      }
    }
  }
  const bool admitted =
      invalid_instances == 0u &&
      selected_prototype != 0xffffffffu &&
      !tied;
  receipts[cue] = ProjectionReceipt{
      admitted ? 1u : 0u,
      invalid_instances,
      selected_prototype,
      tied ? 1u : 0u};
  if (!admitted) return;

  const std::size_t prototype_offset =
      static_cast<std::size_t>(selected_prototype) *
      static_cast<std::size_t>(word_count);
  const DualRailWord* prototype_words = prototypes + prototype_offset;
  for (std::uint32_t word = 0u; word < word_count; ++word) {
    if (!reverse) {
      xor_into(history_words[word], cue_words[word].one);
      xor_into(history_words[word], prototype_words[word].one);
      xor_into(cue_words[word], history_words[word].one);
    } else {
      xor_into(cue_words[word], history_words[word].one);
      xor_into(history_words[word], prototype_words[word].one);
      xor_into(history_words[word], cue_words[word].one);
    }
  }
  xor_into(branch[cue], selected_prototype + 1u);
}

// history and environment must be disjoint.
__global__ void swap_history_environment_kernel(
    DualRailWord* history,
    DualRailWord* environment,
    std::size_t total_words) {
  const std::size_t stride =
      static_cast<std::size_t>(blockDim.x) *
      static_cast<std::size_t>(gridDim.x);
  for (std::size_t word =
           static_cast<std::size_t>(blockIdx.x) *
               static_cast<std::size_t>(blockDim.x) +
           static_cast<std::size_t>(threadIdx.x);
       word < total_words;
       word += stride) {
    swap_words(history[word], environment[word]);
  }
}

__global__ void lesion_word_kernel(
    DualRailWord* field,
    std::uint32_t word,
    std::uint32_t mask) {
  if (blockIdx.x == 0u && threadIdx.x == 0u) xor_into(field[word], mask);
}

}  // namespace bcc32::reversible_projection
