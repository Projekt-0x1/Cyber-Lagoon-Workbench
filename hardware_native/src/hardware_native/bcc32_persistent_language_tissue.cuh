#pragma once

// Device-resident form tissue for the autonomous persistent adult.
//
// Raw boundary bytes do not index strings, words, records, or answer slots.
// Repeated temporal contact recruits overlapping cells, strengthens local
// transitions, and changes eight signed motor dispositions per cell.  A later
// partial cue settles onto the learned populations and unfolds the remainder
// of the trajectory from those resident transitions.

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace substrate::bcc32::persistent_language_tissue {

inline constexpr std::uint32_t kCellCount = 512u;
inline constexpr std::uint32_t kPopulationWidth = 2u;
inline constexpr std::uint32_t kEdgeCapacity = 6u;
inline constexpr std::uint32_t kMotorBits = 8u;
inline constexpr std::uint32_t kOutputCapacity = 128u;
inline constexpr std::uint16_t kNoCell = 0xffffu;
inline constexpr std::int16_t kMotorLimit = 2047;
inline constexpr std::uint16_t kEdgeLimit = 65535u;

struct FormCell {
  std::int16_t motor[kMotorBits]{};
  std::uint16_t targets[kEdgeCapacity]{kNoCell, kNoCell, kNoCell,
                                       kNoCell, kNoCell, kNoCell};
  std::uint16_t strengths[kEdgeCapacity]{};
  std::uint16_t use_count = 0u;
  std::uint16_t claimed = 0u;
};

struct RawTrajectoryFrame {
  std::uint32_t length = 0u;
  std::uint32_t complete = 0u;
  std::uint64_t generation = 0u;
  std::uint8_t bytes[kOutputCapacity]{};
};

struct Tissue {
  FormCell cells[kCellCount]{};
  // Recruited on first contact. This is a generic episode-onset coalition,
  // not a byte, word, semantic role, or authored coordinate.
  std::uint16_t onset[kPopulationWidth]{};
  std::uint32_t onset_count = 0u;
  std::uint16_t active[kPopulationWidth]{};
  std::uint32_t active_count = 0u;
  std::uint32_t growth_cursor = 0u;
  std::uint64_t learned_revision = 0u;
  std::uint64_t learned_contacts = 0u;
  std::uint64_t recruited_cells = 0u;
  std::uint64_t reused_cells = 0u;
  std::uint64_t strengthened_edges = 0u;
  std::uint64_t completed_trajectories = 0u;
  std::uint32_t lesion_begin = kCellCount;
  std::uint32_t lesion_count = 0u;
  RawTrajectoryFrame frame{};
};

__device__ inline void initialize(Tissue* tissue) {
  *tissue = {};
  tissue->lesion_begin = kCellCount;
  for (std::uint32_t lane = 0u; lane < kPopulationWidth; ++lane) {
    tissue->onset[lane] = kNoCell;
    tissue->active[lane] = kNoCell;
  }
  for (std::uint32_t cell = 0u; cell < kCellCount; ++cell)
    for (std::uint32_t edge = 0u; edge < kEdgeCapacity; ++edge)
      tissue->cells[cell].targets[edge] = kNoCell;
}

__device__ __forceinline__ std::int16_t clamp_motor(std::int32_t value) {
  return static_cast<std::int16_t>(
      value < -kMotorLimit ? -kMotorLimit
                           : value > kMotorLimit ? kMotorLimit : value);
}

__device__ __forceinline__ bool lesioned(const Tissue* tissue,
                                         std::uint32_t cell) {
  return cell >= tissue->lesion_begin &&
         cell < tissue->lesion_begin + tissue->lesion_count;
}

__device__ __forceinline__ std::int32_t motor_score(const FormCell& cell,
                                                     std::uint8_t byte) {
  std::int32_t score = 0;
  for (std::uint32_t bit = 0u; bit < kMotorBits; ++bit)
    score += ((byte >> bit) & 1u) != 0u ? cell.motor[bit] : -cell.motor[bit];
  return score;
}

__device__ __forceinline__ bool motor_compatible(const FormCell& cell,
                                                  std::uint8_t byte) {
  for (std::uint32_t bit = 0u; bit < kMotorBits; ++bit) {
    const bool expected = ((byte >> bit) & 1u) != 0u;
    if ((expected && cell.motor[bit] <= 0) ||
        (!expected && cell.motor[bit] >= 0))
      return false;
  }
  return true;
}

__device__ __forceinline__ std::uint8_t decode_motor(
    const Tissue* tissue, const std::uint16_t population[kPopulationWidth],
    std::uint32_t count) {
  std::uint8_t byte = 0u;
  for (std::uint32_t bit = 0u; bit < kMotorBits; ++bit) {
    std::int32_t support = 0;
    for (std::uint32_t lane = 0u; lane < count; ++lane) {
      const std::uint16_t cell = population[lane];
      if (cell != kNoCell && cell < kCellCount && !lesioned(tissue, cell))
        support += tissue->cells[cell].motor[bit];
    }
    if (support > 0) byte |= static_cast<std::uint8_t>(1u << bit);
  }
  return byte;
}

__device__ __forceinline__ std::uint32_t edge_support(
    const Tissue* tissue, const std::uint16_t previous[kPopulationWidth],
    std::uint32_t previous_count, std::uint32_t candidate) {
  std::uint32_t support = 0u;
  for (std::uint32_t lane = 0u; lane < previous_count; ++lane) {
    const std::uint16_t source = previous[lane];
    if (source == kNoCell || source >= kCellCount || lesioned(tissue, source))
      continue;
    const FormCell& cell = tissue->cells[source];
    for (std::uint32_t edge = 0u; edge < kEdgeCapacity; ++edge)
      if (cell.targets[edge] == candidate) support += cell.strengths[edge];
  }
  return support;
}

__device__ __forceinline__ bool in_population(
    const std::uint16_t population[kPopulationWidth], std::uint32_t count,
    std::uint32_t cell) {
  for (std::uint32_t lane = 0u; lane < count; ++lane)
    if (population[lane] == cell) return true;
  return false;
}

__device__ inline std::uint16_t recruit_cell(Tissue* tissue,
                                      const std::uint16_t* previous,
                                      std::uint32_t previous_count,
                                      std::uint32_t lane,
                                      const std::uint16_t* selected,
                                      std::uint32_t selected_count) {
  std::uint32_t anchor = tissue->growth_cursor;
  if (previous_count != 0u && previous[lane % previous_count] != kNoCell)
    anchor = (static_cast<std::uint32_t>(previous[lane % previous_count]) + 1u +
              lane) %
             kCellCount;
  for (std::uint32_t offset = 0u; offset < kCellCount; ++offset) {
    const std::uint32_t candidate = (anchor + offset) % kCellCount;
    if (lesioned(tissue, candidate) ||
        in_population(selected, selected_count, candidate) ||
        tissue->cells[candidate].claimed != 0u)
      continue;
    tissue->cells[candidate].claimed = 1u;
    tissue->cells[candidate].use_count = 1u;
    tissue->growth_cursor = (candidate + 1u) % kCellCount;
    ++tissue->recruited_cells;
    return static_cast<std::uint16_t>(candidate);
  }

  // Capacity pressure prunes the least-used non-active cell.  This is local
  // forgetting, not table widening; incoming references are overwritten only
  // when their source next learns a replacement edge.
  std::uint32_t weakest = kCellCount;
  std::uint16_t weakest_use = 0xffffu;
  for (std::uint32_t candidate = 0u; candidate < kCellCount; ++candidate) {
    if (lesioned(tissue, candidate) ||
        in_population(previous, previous_count, candidate) ||
        in_population(selected, selected_count, candidate))
      continue;
    if (tissue->cells[candidate].use_count < weakest_use) {
      weakest = candidate;
      weakest_use = tissue->cells[candidate].use_count;
    }
  }
  if (weakest == kCellCount) return kNoCell;
  tissue->cells[weakest] = FormCell{};
  tissue->cells[weakest].claimed = 1u;
  tissue->cells[weakest].use_count = 1u;
  ++tissue->recruited_cells;
  return static_cast<std::uint16_t>(weakest);
}

__device__ inline void select_population(
    Tissue* tissue, std::uint8_t byte,
    const std::uint16_t previous[kPopulationWidth],
    std::uint32_t previous_count,
    std::uint16_t selected[kPopulationWidth]) {
  std::uint32_t selected_count = 0u;
  for (; selected_count < kPopulationWidth; ++selected_count) {
    std::int64_t best_score = -0x7fffffffffffffffll;
    std::uint32_t best_cell = kCellCount;
    for (std::uint32_t candidate = 0u; candidate < kCellCount; ++candidate) {
      const FormCell& cell = tissue->cells[candidate];
      if (cell.claimed == 0u || lesioned(tissue, candidate) ||
          in_population(selected, selected_count, candidate) ||
          !motor_compatible(cell, byte))
        continue;
      const std::uint32_t support =
          edge_support(tissue, previous, previous_count, candidate);
      // Rank within the currently active trajectory. A strongly learned copy
      // of the same motor byte elsewhere in the tissue is not a viable
      // continuation merely because its surface disposition is stronger.
      if (previous_count != 0u && support == 0u) continue;
      const std::int64_t score =
          static_cast<std::int64_t>(motor_score(cell, byte)) * 8ll +
          static_cast<std::int64_t>(support) * 16ll +
          cell.use_count;
      if (score > best_score) {
        best_score = score;
        best_cell = candidate;
      }
    }
    // A population must be supported by both its motor disposition and the
    // preceding trajectory. Otherwise co-activity recruits adjacent matter.
    const bool supported =
        best_cell != kCellCount && best_score > 64ll;
    if (supported) {
      selected[selected_count] = static_cast<std::uint16_t>(best_cell);
      ++tissue->reused_cells;
    } else {
      selected[selected_count] =
          recruit_cell(tissue, previous, previous_count, selected_count,
                       selected, selected_count);
    }
  }
}

__device__ inline void strengthen_motor(FormCell* cell, std::uint8_t byte) {
  for (std::uint32_t bit = 0u; bit < kMotorBits; ++bit) {
    const std::int32_t delta = ((byte >> bit) & 1u) != 0u ? 12 : -12;
    cell->motor[bit] = clamp_motor(static_cast<std::int32_t>(cell->motor[bit]) +
                                    delta);
  }
  if (cell->use_count != 0xffffu) ++cell->use_count;
}

__device__ inline void strengthen_edge(Tissue* tissue, std::uint16_t source,
                                std::uint16_t target) {
  if (source == kNoCell || target == kNoCell || source >= kCellCount ||
      target >= kCellCount || lesioned(tissue, source) ||
      lesioned(tissue, target))
    return;
  FormCell& cell = tissue->cells[source];
  std::uint32_t weakest = 0u;
  for (std::uint32_t edge = 0u; edge < kEdgeCapacity; ++edge) {
    if (cell.targets[edge] == target) {
      const std::uint32_t next = static_cast<std::uint32_t>(cell.strengths[edge]) + 8u;
      cell.strengths[edge] = static_cast<std::uint16_t>(
          next > kEdgeLimit ? kEdgeLimit : next);
      ++tissue->strengthened_edges;
      return;
    }
    if (cell.targets[edge] == kNoCell) {
      cell.targets[edge] = target;
      cell.strengths[edge] = 8u;
      ++tissue->strengthened_edges;
      return;
    }
    if (cell.strengths[edge] < cell.strengths[weakest]) weakest = edge;
  }
  cell.targets[weakest] = target;
  cell.strengths[weakest] = 1u;
  ++tissue->strengthened_edges;
}

__device__ inline void learn_transition(
    Tissue* tissue, const std::uint16_t previous[kPopulationWidth],
    std::uint32_t previous_count,
    const std::uint16_t current[kPopulationWidth]) {
  for (std::uint32_t left = 0u; left < previous_count; ++left)
    for (std::uint32_t right = 0u; right < kPopulationWidth; ++right)
      strengthen_edge(tissue, previous[left], current[right]);
}

__device__ inline bool next_population(
    const Tissue* tissue, const std::uint16_t current[kPopulationWidth],
    std::uint32_t current_count,
    std::uint16_t next[kPopulationWidth]) {
  std::uint32_t selected = 0u;
  for (; selected < kPopulationWidth; ++selected) {
    std::uint32_t best_target = kCellCount;
    std::uint32_t best_support = 0u;
    for (std::uint32_t source_lane = 0u; source_lane < current_count;
         ++source_lane) {
      const std::uint16_t source = current[source_lane];
      if (source == kNoCell || source >= kCellCount || lesioned(tissue, source))
        continue;
      const FormCell& cell = tissue->cells[source];
      for (std::uint32_t edge = 0u; edge < kEdgeCapacity; ++edge) {
        const std::uint16_t target = cell.targets[edge];
        if (target == kNoCell || target >= kCellCount || lesioned(tissue, target) ||
            in_population(next, selected, target))
          continue;
        std::uint32_t support = 0u;
        for (std::uint32_t lane = 0u; lane < current_count; ++lane) {
          const std::uint16_t peer = current[lane];
          if (peer == kNoCell || peer >= kCellCount || lesioned(tissue, peer))
            continue;
          const FormCell& peer_cell = tissue->cells[peer];
          for (std::uint32_t peer_edge = 0u; peer_edge < kEdgeCapacity;
               ++peer_edge)
            if (peer_cell.targets[peer_edge] == target)
              support += peer_cell.strengths[peer_edge];
        }
        if (support > best_support) {
          best_support = support;
          best_target = target;
        }
      }
    }
    if (best_target == kCellCount || best_support == 0u) break;
    next[selected] = static_cast<std::uint16_t>(best_target);
  }
  for (std::uint32_t lane = selected; lane < kPopulationWidth; ++lane)
    next[lane] = kNoCell;
  return selected == kPopulationWidth;
}

__device__ inline void clear_frame(RawTrajectoryFrame* frame) {
  frame->length = 0u;
  frame->complete = 0u;
  for (std::uint32_t index = 0u; index < kOutputCapacity; ++index)
    frame->bytes[index] = 0u;
}

__device__ inline void unfold(Tissue* tissue) {
  clear_frame(&tissue->frame);
  if (tissue->active_count == 0u) return;
  std::uint16_t current[kPopulationWidth]{};
  for (std::uint32_t lane = 0u; lane < kPopulationWidth; ++lane)
    current[lane] = tissue->active[lane];
  for (std::uint32_t step = 0u; step < kOutputCapacity; ++step) {
    std::uint16_t next[kPopulationWidth]{};
    if (!next_population(tissue, current, kPopulationWidth, next)) {
      tissue->frame.complete = tissue->frame.length != 0u ? 1u : 0u;
      break;
    }
    const std::uint8_t byte = decode_motor(tissue, next, kPopulationWidth);
    tissue->frame.bytes[tissue->frame.length++] = byte;
    bool repeated = true;
    for (std::uint32_t lane = 0u; lane < kPopulationWidth; ++lane)
      repeated = repeated && in_population(current, kPopulationWidth, next[lane]);
    for (std::uint32_t lane = 0u; lane < kPopulationWidth; ++lane)
      current[lane] = next[lane];
    if (repeated) break;
  }
  ++tissue->frame.generation;
  if (tissue->frame.complete != 0u) ++tissue->completed_trajectories;
}

template <typename Word>
__device__ void contact(Tissue* tissue, const Word* words,
                        std::uint32_t count, std::uint64_t revision) {
  if (revision == tissue->learned_revision) return;
  tissue->learned_revision = revision;
  clear_frame(&tissue->frame);
  if (count == 0u) {
    tissue->active_count = 0u;
    for (std::uint32_t lane = 0u; lane < kPopulationWidth; ++lane)
      tissue->active[lane] = kNoCell;
    return;
  }

  if (tissue->onset_count == 0u) {
    for (std::uint32_t lane = 0u; lane < kPopulationWidth; ++lane)
      tissue->onset[lane] = recruit_cell(tissue, nullptr, 0u, lane,
                                         tissue->onset, lane);
    tissue->onset_count = kPopulationWidth;
  }
  std::uint16_t previous[kPopulationWidth]{};
  for (std::uint32_t lane = 0u; lane < kPopulationWidth; ++lane)
    previous[lane] = tissue->onset[lane];
  std::uint32_t previous_count = tissue->onset_count;
  for (std::uint32_t position = 0u; position < count; ++position) {
    const std::uint8_t byte = static_cast<std::uint8_t>(words[position] & 0xffu);
    std::uint16_t current[kPopulationWidth]{};
    select_population(tissue, byte, previous, previous_count, current);
    for (std::uint32_t lane = 0u; lane < kPopulationWidth; ++lane)
      if (current[lane] != kNoCell)
        strengthen_motor(&tissue->cells[current[lane]], byte);
    if (previous_count != 0u)
      learn_transition(tissue, previous, previous_count, current);
    for (std::uint32_t lane = 0u; lane < kPopulationWidth; ++lane)
      previous[lane] = current[lane];
    previous_count = kPopulationWidth;
  }
  for (std::uint32_t lane = 0u; lane < kPopulationWidth; ++lane)
    tissue->active[lane] = previous[lane];
  tissue->active_count = previous_count;
  ++tissue->learned_contacts;
  unfold(tissue);
}

}  // namespace substrate::bcc32::persistent_language_tissue
