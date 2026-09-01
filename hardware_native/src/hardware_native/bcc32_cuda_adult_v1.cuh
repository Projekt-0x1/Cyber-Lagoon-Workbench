#pragma once

#include <cuda_runtime.h>

#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/merge.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/sort.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <cmath>
#include <numeric>
#include <span>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "bcc32_cuda_resident_synthesis.cuh"
#include "bcc32_cuda_adult_population_coactivity.cuh"

// This is an explicitly engineered raw-byte warm-start organ. It is useful as a
// CUDA learning frontier and scaffold, but it is not the final evolved BCC brain,
// BCC-F integration, a semantic evaluator, or evidence of GPT-level understanding.
#include "bcc32_cuda_adult_state.cuh"

namespace bcc32_cuda_adult_v1 {

#if !defined(BCC32_CUDA_ADULT_STATE_ONLY)

#include "bcc32_cuda_adult_surface_population.inl"
#include "bcc32_cuda_adult_construction_organ.inl"

// Public diagnostic entry point used by bcc32_cuda_adult_v1.cu. Keep this
// compatibility surface explicit: the helper is a deliberate construction
// lesion, not dead code merely because no library caller uses it.
inline void lesion_construction_state(AdultState& state) {
  if (state.construction_store_count.get() == nullptr) return;
  cuda_require(cudaMemset(state.construction_store_count.get(), 0,
                          state.construction_store_count.bytes()),
               "lesion resident construction extent");
  cuda_require(cudaMemset(state.construction_hash_slots.get(), 0,
                          state.construction_hash_slots.bytes()),
               "lesion resident construction hash");
  cuda_require(cudaMemset(state.construction_lengths.get(), 0,
                          state.construction_lengths.bytes()),
               "lesion resident construction lengths");
  cuda_require(cudaMemset(state.construction_slot_units.get(), 0xff,
                          state.construction_slot_units.bytes()),
               "lesion resident construction slot units");
  cuda_require(cudaMemset(state.construction_slot_masses.get(), 0,
                          state.construction_slot_masses.bytes()),
               "lesion resident construction slot masses");
  cuda_require(cudaMemset(state.construction_slot_totals.get(), 0,
                          state.construction_slot_totals.bytes()),
               "lesion resident construction slot totals");
  cuda_require(cudaMemset(state.construction_slot_overflow.get(), 0,
                          state.construction_slot_overflow.bytes()),
               "lesion resident construction slot overflow");
  state.construction_count_host = 0u;
  state.construction_lesioned = true;
}

#include "bcc32_cuda_adult_surface_projection.inl"
#include "bcc32_cuda_adult_surface_population_adaptation.inl"
#include "bcc32_cuda_adult_surface_episode_learning.inl"
#include "bcc32_cuda_adult_construction_learning.inl"
#include "bcc32_cuda_adult_distributed_motor_contact.inl"
__global__ void fill_shuffle_keys_kernel(std::uint32_t count, std::uint32_t seed,
                                         std::uint32_t* keys) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count) keys[i] = mix32(i ^ seed);
}

__global__ void byte_statistics_kernel(const std::uint8_t* bytes, std::uint32_t count,
                                       std::uint32_t* histogram, std::uint32_t* pairs) {
  const std::uint32_t stride = gridDim.x * blockDim.x;
  for (std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += stride) {
    const std::uint32_t current = bytes[i];
    atomicAdd(histogram + current, 1u);
    if (i != 0u) {
      atomicAdd(pairs + static_cast<std::uint32_t>(bytes[i - 1u]) * 256u + current, 1u);
    }
  }
}

__global__ void detect_exact_resident_bytes_replay_kernel(
    const std::uint8_t* incoming_bytes, std::uint32_t incoming_count,
    const std::uint32_t* resident_units, std::uint32_t resident_count,
    const std::uint32_t* episode_breaks, std::uint32_t episode_break_count,
    const std::uint32_t* unit_lengths, const std::uint32_t* unit_content,
    std::uint32_t unit_count, std::uint32_t require_episode_alignment,
    std::uint32_t replay_allowed,
    std::uint32_t* exact_replay) {
  const std::uint32_t start = blockIdx.x * blockDim.x + threadIdx.x;
  if (replay_allowed == 0u || exact_replay[0] != 0u || incoming_count == 0u ||
      start >= resident_count) {
    return;
  }
  std::uint32_t episode_end = resident_count;
  if (require_episode_alignment != 0u) {
    std::uint32_t lo = 0u, hi = episode_break_count;
    while (lo < hi) {
      const std::uint32_t mid = lo + (hi - lo) / 2u;
      if (episode_breaks[mid] < start) lo = mid + 1u; else hi = mid;
    }
    std::uint32_t episode = 0u;
    if (start != 0u) {
      if (lo >= episode_break_count || episode_breaks[lo] != start) return;
      episode = lo + 1u;
    }
    if (episode >= episode_break_count) return;
    episode_end = episode_breaks[episode];
    if (episode_end <= start || episode_end > resident_count) return;
  } else if (start != 0u) {
    return;
  }
  const std::uint32_t first_unit = resident_units[start];
  if (first_unit >= unit_count || unit_lengths[first_unit] == 0u) return;

  std::uint32_t cursor = 0u;
  for (std::uint32_t position = start;
       position < episode_end && cursor < incoming_count; ++position) {
    const std::uint32_t unit = resident_units[position];
    if (unit >= unit_count || unit_lengths[unit] == 0u ||
        unit_lengths[unit] > kUnitWords * 4u) {
      return;
    }
    for (std::uint32_t offset = 0u;
         offset < unit_lengths[unit] && cursor < incoming_count; ++offset) {
      const std::uint32_t packed = unit_content[unit * kUnitWords + offset / 4u];
      const std::uint8_t resident = static_cast<std::uint8_t>(
          (packed >> ((offset % 4u) * 8u)) & 0xffu);
      if (incoming_bytes[cursor] != resident) return;
      ++cursor;
      if (cursor == incoming_count) {
        const std::uint32_t end = position + 1u;
        if (offset + 1u != unit_lengths[unit]) return;
        if (require_episode_alignment == 0u) {
          if (end != resident_count) return;
        } else {
          if (end != episode_end) return;
        }
        atomicExch(exact_replay, 1u);
        return;
      }
    }
  }
}

__global__ void byte_statistics_if_novel_kernel(
    const std::uint8_t* bytes, std::uint32_t count,
    const std::uint32_t* exact_replay, std::uint32_t* histogram,
    std::uint32_t* pairs) {
  if (exact_replay[0] != 0u) return;
  const std::uint32_t stride = gridDim.x * blockDim.x;
  for (std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count;
       i += stride) {
    const std::uint32_t current = bytes[i];
    atomicAdd(histogram + current, 1u);
    if (i != 0u) {
      atomicAdd(pairs + static_cast<std::uint32_t>(bytes[i - 1u]) * 256u + current,
                1u);
    }
  }
}

#include "bcc32_cuda_adult_body_feedback_controls.inl"
#include "bcc32_cuda_adult_surface_segmentation.inl"
__global__ void mark_unique_units_kernel(const std::uint8_t* bytes, std::uint32_t byte_count,
                                         const std::uint32_t* starts, std::uint32_t unit_count,
                                         const UnitKey* sorted_keys,
                                         const std::uint32_t* sorted_occurrences,
                                         std::uint32_t* unique_flags) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= unit_count) return;
  unique_flags[i] = i == 0u || !(sorted_keys[i] == sorted_keys[i - 1u]) ||
                    !same_unit_bytes(bytes, starts, byte_count, unit_count,
                                     sorted_occurrences[i], sorted_occurrences[i - 1u]);
}

__global__ void materialize_dictionary_kernel(
    const std::uint8_t* bytes, std::uint32_t byte_count, const std::uint32_t* starts,
    std::uint32_t unit_occurrences, const std::uint32_t* sorted_occurrences,
    const std::uint32_t* unique_flags, const std::uint32_t* group_ids,
    std::uint32_t* occurrence_to_unit, std::uint32_t* lengths,
    std::uint32_t* packed_content, std::uint32_t* vitality) {
  const std::uint32_t sorted = blockIdx.x * blockDim.x + threadIdx.x;
  if (sorted >= unit_occurrences) return;
  const std::uint32_t occurrence = sorted_occurrences[sorted];
  const std::uint32_t unit = group_ids[sorted] - 1u;
  occurrence_to_unit[occurrence] = unit;
  atomicAdd(vitality + unit, 1u);
  if (unique_flags[sorted] == 0u) return;

  const std::uint32_t begin = starts[occurrence];
  const std::uint32_t end = occurrence + 1u < unit_occurrences
                                ? starts[occurrence + 1u]
                                : byte_count;
  lengths[unit] = end - begin;
  for (std::uint32_t word = 0u; word < kUnitWords; ++word) {
    std::uint32_t packed = 0u;
    for (std::uint32_t lane = 0u; lane < 4u; ++lane) {
      const std::uint32_t offset = word * 4u + lane;
      if (begin + offset < end) packed |= static_cast<std::uint32_t>(bytes[begin + offset]) << (lane * 8u);
    }
    packed_content[unit * kUnitWords + word] = packed;
  }
}

#include "bcc32_cuda_adult_ngram_associations.inl"
#include "bcc32_cuda_adult_window_answer_candidates.inl"
__global__ void stage_answer_frame_selection_kernel(
    const bcc32_cuda_resident_synthesis::ResidentSynthesisResult* synthesis_result,
    const std::uint32_t* synthesis_units,
    answer_frame::MutableSelectionState* selection,
    std::uint32_t* relation_tail, std::uint32_t* relation_tail_count) {
  if (blockIdx.x != 0u || threadIdx.x != 0u) return;
  *relation_tail_count = 0u;
  for (std::uint32_t i = 0u;
       i < bcc32_cuda_resident_synthesis::kResidentSynthesisMaxRelationTail; ++i)
    relation_tail[i] = answer_frame::kInvalidUnit;
  ++selection->revision;
  selection->intact = 0u;
  selection->value = answer_frame::kInvalidUnit;
  selection->subject = answer_frame::kInvalidUnit;
  selection->predicate = answer_frame::kInvalidUnit;
  selection->evidence = 0u;
  if (synthesis_result->ready == 0u || synthesis_result->conditioned == 0u ||
      synthesis_result->unit_count != 1u) {
    return;
  }
  selection->value = synthesis_units[0];
  selection->subject = synthesis_result->relation_subject;
  selection->predicate = synthesis_result->relation_predicate;
  selection->evidence = synthesis_result->score_high;
  selection->intact = 1u;
  const std::uint32_t tail_count = min(
      synthesis_result->relation_tail_count,
      bcc32_cuda_resident_synthesis::kResidentSynthesisMaxRelationTail);
  for (std::uint32_t i = 0u; i < tail_count; ++i)
    relation_tail[i] = synthesis_result->relation_tail[i];
  *relation_tail_count = tail_count;
}

__global__ void adopt_answer_frame_kernel(
    const answer_frame::Result* frame_result,
    const std::uint32_t* frame_units, std::uint32_t* motor_context,
    std::uint32_t* motor_completion) {
  if (blockIdx.x != 0u || frame_result->ready == 0u) return;
  const std::uint32_t count = min(frame_result->unit_count, kCompositionUnits);
  for (std::uint32_t index = threadIdx.x; index < count; index += blockDim.x)
    motor_completion[index] = frame_units[index];
  __syncthreads();
  if (threadIdx.x != 0u) return;
  if (count == 0u) return;
  const std::uint32_t focus = frame_result->value_position < count
      ? frame_result->value_position : min(2u, count - 1u);
  motor_context[0] = 1u;
  motor_context[1] = frame_units[focus];
  motor_context[2] = frame_result->score_high;
  motor_context[3] = count;
  motor_context[4] = 1u;
  motor_context[5] = 4u;
  motor_context[12] = count;
  motor_context[13] = 1u;
  motor_context[15] = 1u;
}

__global__ void build_role_compositor_evidence_kernel(
    const std::uint32_t* unit_flags,
    const substrate::bcc32::resident_roles::MutableStructuralRole* roles,
    std::uint32_t unit_count, std::uint32_t* boundary_evidence,
    std::uint32_t* closure_evidence) {
  const std::uint32_t unit = blockIdx.x * blockDim.x + threadIdx.x;
  if (unit >= unit_count) return;
  const bool eligible = (unit_flags[unit] & answer_frame::kUnitEligible) != 0u;
  const std::uint32_t depth = max(1u, roles[unit].evidence_depth);
  boundary_evidence[unit] = eligible ? depth : 0u;
  closure_evidence[unit] =
      eligible && (unit_flags[unit] & answer_frame::kUnitClosure) != 0u
          ? depth : 0u;
}

__global__ void seed_role_compositor_activation_kernel(
    const std::uint32_t* unit_lengths, const std::uint32_t* unit_content,
    const std::uint32_t* unit_vitality,
    std::uint32_t unit_count, const std::uint32_t* learned_boundary_bytes,
    const std::uint8_t* raw_cue,
    std::uint32_t raw_cue_count, unsigned long long* cue_activation,
    std::uint32_t* cue_groups) {
  const std::uint32_t unit = blockIdx.x * blockDim.x + threadIdx.x;
  if (unit >= unit_count) return;
  unsigned long long activation = 0ull;
  std::uint32_t activation_group = 0xffffffffu;
  const std::uint32_t length = unit_lengths[unit];
  auto resident_byte = [&](std::uint32_t offset) {
    const std::uint32_t word =
        unit_content[unit * kUnitWords + offset / 4u];
    return static_cast<std::uint8_t>(
        (word >> (8u * (offset & 3u))) & 0xffu);
  };
  auto learned_boundary = [&](std::uint8_t byte) {
    for (std::uint32_t index = 0u; index < kBoundaryCount; ++index)
      if (byte == static_cast<std::uint8_t>(learned_boundary_bytes[index]))
        return true;
    return false;
  };
  std::uint32_t unit_begin = 0u;
  std::uint32_t unit_end = length;
  while (unit_begin < unit_end && learned_boundary(resident_byte(unit_begin)))
    ++unit_begin;
  while (unit_end > unit_begin && learned_boundary(resident_byte(unit_end - 1u)))
    --unit_end;
  const std::uint32_t cue_extent = unit_end - unit_begin;
  if (cue_extent >= 4u && cue_extent <= raw_cue_count) {
    for (std::uint32_t begin = 0u; begin + cue_extent <= raw_cue_count; ++begin) {
      bool exact = true;
      for (std::uint32_t offset = 0u; offset < cue_extent; ++offset) {
        const std::uint8_t expected = resident_byte(unit_begin + offset);
        if (raw_cue[begin + offset] != expected) {
          exact = false;
          break;
        }
      }
      if (!exact) continue;
      const unsigned long long strength =
          static_cast<unsigned long long>(cue_extent) * cue_extent * 4096ull *
          (begin + cue_extent) / max(1u, unit_vitality[unit]);
      if (strength > activation) {
        activation = strength;
        activation_group = min(31u, begin);
      }
    }
  }
  cue_activation[unit] = activation;
  cue_groups[unit] = activation_group;
}

__global__ void build_conditioned_role_traces_kernel(
    const std::uint32_t* episode_units, std::uint32_t episode_unit_count,
    const std::uint32_t* episode_breaks, std::uint32_t episode_break_count,
    const std::uint32_t* unit_lengths, const std::uint32_t* unit_content,
    const std::uint32_t* closure_bytes,
    const unsigned long long* cue_activation, const std::uint32_t* cue_groups,
    role_compositor::SubjectConditionedRelationTrace* traces,
    std::uint32_t trace_capacity, std::uint32_t* trace_count,
    std::uint32_t* trace_units, std::uint32_t trace_unit_capacity,
    std::uint32_t* trace_unit_count) {
  if (blockIdx.x != 0u || threadIdx.x != 0u) return;
  std::uint32_t traces_written = 0u;
  std::uint32_t units_written = 0u;
  std::uint32_t clause_begin = 0u;
  std::uint32_t episode = 0u;
  for (std::uint32_t position = 0u; position < episode_unit_count; ++position) {
    while (episode < episode_break_count &&
           clause_begin >= episode_breaks[episode])
      ++episode;
    const bool closes = resident_unit_contains_any(
        unit_lengths, unit_content, episode_units[position], closure_bytes, 1u);
    const bool episode_closes =
        episode < episode_break_count && position + 1u == episode_breaks[episode];
    if (!closes && !episode_closes && position + 1u != episode_unit_count)
      continue;
    const std::uint32_t clause_end = position + 1u;
    const std::uint32_t clause_count = clause_end - clause_begin;
    if (clause_count >= 2u && clause_count <= kCompositionUnits &&
        traces_written < trace_capacity &&
        clause_count <= trace_unit_capacity - units_written) {
      std::uint32_t subject = 0xffffffffu;
      unsigned long long strongest = 0ull;
      std::uint32_t cue_group_mask = 0u;
      for (std::uint32_t at = clause_begin; at < clause_end; ++at) {
        const std::uint32_t unit = episode_units[at];
        const unsigned long long activation = cue_activation[unit];
        const std::uint32_t group = cue_groups[unit];
        if (activation != 0ull && group < 32u)
          cue_group_mask |= 1u << group;
        if (activation > strongest) {
          strongest = activation;
          subject = unit;
        }
      }
      const std::uint32_t cue_hits = __popc(cue_group_mask);
      if (subject != 0xffffffffu && cue_hits >= 2u) {
        traces[traces_written] = role_compositor::SubjectConditionedRelationTrace{
            subject, units_written, clause_count, cue_hits, 1u};
        for (std::uint32_t at = clause_begin; at < clause_end; ++at)
          trace_units[units_written++] = episode_units[at];
        ++traces_written;
      }
    }
    clause_begin = clause_end;
    if (episode_closes) ++episode;
  }
  std::uint32_t episode_begin = 0u;
  for (std::uint32_t episode_index = 0u;
       episode_index < episode_break_count && traces_written < trace_capacity;
       ++episode_index) {
    const std::uint32_t episode_end = min(
        episode_unit_count, episode_breaks[episode_index]);
    std::uint32_t previous_window_begin = 0xffffffffu;
    for (std::uint32_t position = episode_begin;
         position < episode_end && traces_written < trace_capacity; ++position) {
      const std::uint32_t seed_unit = episode_units[position];
      if (cue_groups[seed_unit] >= 32u || cue_activation[seed_unit] == 0ull)
        continue;
      std::uint32_t window_begin = position > episode_begin + 16u
          ? position - 16u : episode_begin;
      if (window_begin == previous_window_begin) continue;
      const std::uint32_t window_end = min(
          episode_end, window_begin + kCompositionUnits);
      const std::uint32_t window_count = window_end - window_begin;
      if (window_count < 2u || window_count > trace_unit_capacity - units_written)
        continue;
      std::uint32_t cue_group_mask = 0u;
      std::uint32_t subject = 0xffffffffu;
      unsigned long long strongest = 0ull;
      for (std::uint32_t at = window_begin; at < window_end; ++at) {
        const std::uint32_t unit = episode_units[at];
        const unsigned long long activation = cue_activation[unit];
        const std::uint32_t group = cue_groups[unit];
        if (activation != 0ull && group < 32u)
          cue_group_mask |= 1u << group;
        if (activation > strongest) {
          strongest = activation;
          subject = unit;
        }
      }
      const std::uint32_t cue_hits = __popc(cue_group_mask);
      if (subject == 0xffffffffu || cue_hits < 2u) continue;
      traces[traces_written] = role_compositor::SubjectConditionedRelationTrace{
          subject, units_written, window_count, cue_hits, 1u};
      for (std::uint32_t at = window_begin; at < window_end; ++at)
        trace_units[units_written++] = episode_units[at];
      ++traces_written;
      previous_window_begin = window_begin;
    }
    episode_begin = episode_end;
  }
  trace_count[0] = traces_written;
  trace_unit_count[0] = units_written;
}


__global__ void activate_selected_relation_trace_kernel(
    const role_compositor::RoleCompositorChoice* choice,
    const role_compositor::SubjectConditionedRelationTrace* traces,
    const std::uint32_t* trace_units, const std::uint32_t* unit_lengths,
    const std::uint32_t* unit_vitality,
    const unsigned long long* cue_activation, std::uint32_t* subject_ids,
    std::uint32_t* subject_weights, std::uint32_t* subject_count) {
  if (blockIdx.x != 0u || threadIdx.x != 0u || choice->trace == 0xffffffffu)
    return;
  constexpr std::uint32_t kTraceFocus = kCompositionUnits;
  std::uint32_t focus_ids[kTraceFocus]{};
  std::uint32_t focus_scores[kTraceFocus]{};
  const role_compositor::SubjectConditionedRelationTrace trace =
      traces[choice->trace];
  for (std::uint32_t offset = 0u; offset < trace.unit_count; ++offset) {
    const std::uint32_t unit = trace_units[trace.unit_begin + offset];
    const std::uint32_t length = unit_lengths[unit];
    if (length < 3u) continue;
    std::uint32_t cue_distance = trace.unit_count;
    for (std::uint32_t other = 0u; other < trace.unit_count; ++other) {
      const std::uint32_t other_unit = trace_units[trace.unit_begin + other];
      if (cue_activation[other_unit] == 0ull) continue;
      const std::uint32_t distance = offset > other ? offset - other : other - offset;
      cue_distance = min(cue_distance, distance);
    }
    const std::uint32_t proximity = trace.unit_count - cue_distance;
    std::uint32_t vitality_depth = 0u;
    for (std::uint32_t value = unit_vitality[unit]; value != 0u; value >>= 1u)
      ++vitality_depth;
    const std::uint32_t score = proximity * 24u +
        min(16u, vitality_depth) * 32u + min(15u, length);
    std::uint32_t slot = kTraceFocus;
    for (std::uint32_t at = 0u; at < kTraceFocus; ++at) {
      if (score > focus_scores[at]) {
        slot = at;
        break;
      }
    }
    if (slot == kTraceFocus) continue;
    for (std::uint32_t at = kTraceFocus - 1u; at > slot; --at) {
      focus_ids[at] = focus_ids[at - 1u];
      focus_scores[at] = focus_scores[at - 1u];
    }
    focus_ids[slot] = unit;
    focus_scores[slot] = score;
  }
  std::uint32_t count = min(subject_count[0], kSubjectCap);
  for (std::uint32_t focus = 0u;
       focus < kTraceFocus && focus_scores[focus] != 0u;
       ++focus) {
    const std::uint32_t unit = focus_ids[focus];
    const std::uint32_t weight = min(
        kSubjectCapWeight, kSubjectInitWeight + focus_scores[focus]);
    std::uint32_t destination = count;
    for (std::uint32_t prior = 0u; prior < count; ++prior) {
      if (subject_ids[prior] == unit) {
        destination = prior;
        break;
      }
    }
    bool appended = false;
    if (destination == count && count < kSubjectCap) {
      ++count;
      appended = true;
    } else if (destination == count) {
      destination = 0u;
      for (std::uint32_t prior = 1u; prior < count; ++prior) {
        if (subject_weights[prior] < subject_weights[destination])
          destination = prior;
      }
    }
    subject_ids[destination] = unit;
    subject_weights[destination] = appended
        ? weight : max(subject_weights[destination], weight);
  }
  subject_count[0] = count;
}

__global__ void seed_motor_from_selected_relation_trace_kernel(
    const role_compositor::RoleCompositorChoice* choice,
    const role_compositor::SubjectConditionedRelationTrace* traces,
    const std::uint32_t* trace_units,
    const unsigned long long* cue_activation,
    const std::uint32_t* unit_lengths, const std::uint32_t* unit_vitality,
    std::uint32_t* motor_context, std::uint32_t* motor_completion) {
  if (blockIdx.x != 0u || threadIdx.x != 0u || choice->trace == 0xffffffffu)
    return;
  const role_compositor::SubjectConditionedRelationTrace trace =
      traces[choice->trace];
  std::uint32_t context = 0xffffffffu;
  std::uint32_t strongest_offset = 0u;
  unsigned long long strongest = 0ull;
  for (std::uint32_t offset = 0u; offset < trace.unit_count; ++offset) {
    const std::uint32_t unit = trace_units[trace.unit_begin + offset];
    motor_completion[offset] = unit;
    if (cue_activation[unit] > strongest) {
      strongest = cue_activation[unit];
      context = unit;
      strongest_offset = offset;
    }
  }
  if (context == 0xffffffffu) return;
  motor_context[0] = 1u;
  if (strongest_offset >= 2u) {
    motor_context[1] = trace_units[trace.unit_begin + strongest_offset - 2u];
    motor_context[6] = trace_units[trace.unit_begin + strongest_offset - 1u];
    motor_context[7] = 2u;
  } else {
    motor_context[1] = context;
    motor_context[6] = context;
    motor_context[7] = 1u;
  }
  motor_context[2] = static_cast<std::uint32_t>(choice->score >> 32u);
  motor_context[3] = trace.unit_count;
  motor_context[4] = choice->cue_hits;
  motor_context[5] = 6u;
  motor_context[12] = 0u;
  motor_context[13] = trace.evidence;
  motor_context[15] = 1u;
}

// --frame-emit: multi-clause relation-frame composer (substrate-generator wiring).
// Builds motor_completion = [S, P1, V1, P2, V2, ...] from the top subject-field
// subject S and its strongest resident conditioned relations (anchor==S) -- an
// ordered multi-clause answer recombining several resident relations, the
// frame+content emission proven in bcc32_resident_relation_frame_emission_contract,
// driven here by the adult's OWN subject field + relation store. Overwrites the
// existing mode-4 composition only when it finds usable relations (lesionable via
// the flag; otherwise the existing composition stands). Emits into the mode-4 path;
// no new emission code. No token table, no answer key -- content is resident matter.
__global__ void compose_frame_content_kernel(
    const std::uint32_t* subject_ids, const std::uint32_t* subject_weights,
    const std::uint32_t* subject_count_ptr,
    const ConditionedTransitionKey* cond, const std::uint32_t* cond_counts,
    std::uint32_t cond_count, const std::uint32_t* unit_lengths,
    std::uint32_t* motor_context, std::uint32_t* motor_completion) {
  if (blockIdx.x != 0u || threadIdx.x != 0u) return;
  const std::uint32_t subj_n = subject_count_ptr[0];
  if (subj_n == 0u || cond_count == 0u) return;        // no subject field -> keep existing composition
  std::uint32_t subject = subject_ids[0], best_w = subject_weights[0];
  for (std::uint32_t i = 1u; i < subj_n; ++i)
    if (subject_weights[i] > best_w) { best_w = subject_weights[i]; subject = subject_ids[i]; }
  const std::uint32_t lo =
      bcc32_cuda_resident_synthesis::lower_subject_transition(cond, cond_count, subject, 0u);
  const std::uint32_t hi =
      bcc32_cuda_resident_synthesis::upper_subject_transition(cond, cond_count, subject, 0xffffffffu);
  constexpr std::uint32_t kClauses = 3u;
  std::uint32_t bp[kClauses] = {}, bv[kClauses] = {}, bc[kClauses] = {};
  std::uint32_t nb = 0u;
  for (std::uint32_t i = lo; i < hi; ++i) {
    const std::uint32_t P = cond[i].previous, V = cond[i].next, c = cond_counts[i];
    if (c == 0u || P == subject || V == subject || V == P) continue;
    if (unit_lengths[P] < 2u || unit_lengths[V] < 2u) continue;   // skip empty units
    bool dup = false;
    for (std::uint32_t k = 0u; k < nb; ++k)
      if (bv[k] == V || bp[k] == P) { dup = true; break; }        // one clause per predicate/value
    if (dup) continue;
    if (nb < kClauses) { bp[nb] = P; bv[nb] = V; bc[nb] = c; ++nb; }
    else {
      std::uint32_t m = 0u;
      for (std::uint32_t k = 1u; k < kClauses; ++k) if (bc[k] < bc[m]) m = k;
      if (c > bc[m]) { bp[m] = P; bv[m] = V; bc[m] = c; }         // keep top-kClauses by evidence
    }
  }
  for (std::uint32_t a = 0u; a < nb; ++a)              // order clauses by evidence desc
    for (std::uint32_t b = a + 1u; b < nb; ++b)
      if (bc[b] > bc[a]) {
        std::uint32_t t;
        t = bp[a]; bp[a] = bp[b]; bp[b] = t;
        t = bv[a]; bv[a] = bv[b]; bv[b] = t;
        t = bc[a]; bc[a] = bc[b]; bc[b] = t;
      }
  // Confidence gate: only override the baseline composition when the plan is
  // well-evidenced (a full multi-clause plan whose weakest clause is still solid).
  // For subjects with thin/noisy relations, keep the existing composition -- so
  // frame-emit can only help, never regress.
  if (nb < kClauses || bc[nb - 1u] < 4u) return;
  std::uint32_t n = 0u;
  motor_completion[n++] = subject;
  for (std::uint32_t k = 0u; k < nb && n + 2u <= kCompositionUnits; ++k) {
    motor_completion[n++] = bp[k];
    motor_completion[n++] = bv[k];
  }
  motor_context[0] = 1u;
  motor_context[1] = motor_completion[n - 1u];         // re-enter the subject-field walk from the last value
  motor_context[2] = bc[0];
  motor_context[3] = n;
  motor_context[4] = 1u;
  motor_context[5] = 4u;
  motor_context[12] = n;
  motor_context[13] = 1u;
  motor_context[15] = 1u;
}

__global__ void sum_answer_frame_role_evidence_kernel(
    const std::uint64_t* base, const std::uint64_t* online,
    std::uint64_t* combined, std::size_t count) {
  const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x +
                            threadIdx.x;
  if (index < count) combined[index] = base[index] + online[index];
}

__device__ void insert_top(std::uint32_t id, std::uint32_t weight,
                           std::uint32_t* ids, std::uint32_t* weights,
                           std::uint32_t count);

__global__ void mark_large_bigram_contexts_kernel(const BigramKey* keys,
                                                   std::uint32_t count,
                                                   std::uint32_t* flags) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count) return;
  if (i != 0u && keys[i - 1u].previous == keys[i].previous) {
    flags[i] = 0u;
    return;
  }
  std::uint32_t end = i + 1u;
  while (end < count && keys[end].previous == keys[i].previous) ++end;
  flags[i] = end - i > kTopK;
}

__global__ void materialize_bigram_cache_kernel(
    const BigramKey* keys, const std::uint32_t* counts, std::uint32_t count,
    const std::uint32_t* flags, const std::uint32_t* ids,
    std::uint32_t* contexts, std::uint32_t* entries) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count || flags[i] == 0u) return;
  const std::uint32_t cache = ids[i] - 1u;
  contexts[cache] = keys[i].previous;
  std::uint32_t top_entries[kTopK] = {};
  std::uint32_t top_weights[kTopK] = {};
  std::uint32_t end = i;
  while (end < count && keys[end].previous == keys[i].previous) {
    insert_top(end, counts[end], top_entries, top_weights, kTopK);
    ++end;
  }
  for (std::uint32_t slot = 0u; slot < kTopK; ++slot) {
    entries[cache * kTopK + slot] = top_entries[slot];
  }
}

__global__ void mark_large_trigram_contexts_kernel(const TrigramKey* keys,
                                                    std::uint32_t count,
                                                    std::uint32_t* flags) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count) return;
  if (i != 0u && keys[i - 1u].first == keys[i].first &&
      keys[i - 1u].second == keys[i].second) {
    flags[i] = 0u;
    return;
  }
  std::uint32_t end = i + 1u;
  while (end < count && keys[end].first == keys[i].first &&
         keys[end].second == keys[i].second) ++end;
  flags[i] = end - i > kTopK;
}

__global__ void materialize_trigram_cache_kernel(
    const TrigramKey* keys, const std::uint32_t* counts, std::uint32_t count,
    const std::uint32_t* flags, const std::uint32_t* ids,
    BigramKey* contexts, std::uint32_t* entries) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count || flags[i] == 0u) return;
  const std::uint32_t cache = ids[i] - 1u;
  contexts[cache] = BigramKey{keys[i].first, keys[i].second};
  std::uint32_t top_entries[kTopK] = {};
  std::uint32_t top_weights[kTopK] = {};
  std::uint32_t end = i;
  while (end < count && keys[end].first == keys[i].first &&
         keys[end].second == keys[i].second) {
    insert_top(end, counts[end], top_entries, top_weights, kTopK);
    ++end;
  }
  for (std::uint32_t slot = 0u; slot < kTopK; ++slot) {
    entries[cache * kTopK + slot] = top_entries[slot];
  }
}

__device__ void insert_top(std::uint32_t id, std::uint32_t weight,
                           std::uint32_t* ids, std::uint32_t* weights,
                           std::uint32_t count) {
  for (std::uint32_t slot = 0u; slot < count; ++slot) {
    if (weight > weights[slot] || (weight == weights[slot] && id < ids[slot])) {
      for (std::uint32_t move = count - 1u; move > slot; --move) {
        ids[move] = ids[move - 1u];
        weights[move] = weights[move - 1u];
      }
      ids[slot] = id;
      weights[slot] = weight;
      return;
    }
  }
}

__global__ void build_unigram_top_kernel(const std::uint32_t* vitality,
                                         std::uint32_t unit_count,
                                         std::uint32_t* top_ids) {
  __shared__ std::uint32_t ids[kBlock];
  __shared__ std::uint32_t weights[kBlock];
  __shared__ std::uint32_t prior_id;
  __shared__ std::uint32_t prior_weight;
  if (threadIdx.x == 0u) {
    prior_id = 0u;
    prior_weight = 0xffffffffu;
  }
  __syncthreads();
  for (std::uint32_t rank = 0u; rank < kUnigramTop; ++rank) {
    std::uint32_t local_id = 0xffffffffu;
    std::uint32_t local_weight = 0u;
    for (std::uint32_t unit = threadIdx.x; unit < unit_count; unit += blockDim.x) {
      const std::uint32_t weight = vitality[unit];
      const bool below_prior = weight < prior_weight ||
          (weight == prior_weight && unit > prior_id);
      if (below_prior && (weight > local_weight ||
          (weight == local_weight && unit < local_id))) {
        local_weight = weight;
        local_id = unit;
      }
    }
    ids[threadIdx.x] = local_id;
    weights[threadIdx.x] = local_weight;
    __syncthreads();
    for (std::uint32_t offset = blockDim.x / 2u; offset != 0u; offset >>= 1u) {
      if (threadIdx.x < offset &&
          (weights[threadIdx.x + offset] > weights[threadIdx.x] ||
           (weights[threadIdx.x + offset] == weights[threadIdx.x] &&
            ids[threadIdx.x + offset] < ids[threadIdx.x]))) {
        weights[threadIdx.x] = weights[threadIdx.x + offset];
        ids[threadIdx.x] = ids[threadIdx.x + offset];
      }
      __syncthreads();
    }
    if (threadIdx.x == 0u) {
      top_ids[rank] = ids[0] == 0xffffffffu ? 0u : ids[0];
      prior_id = ids[0];
      prior_weight = weights[0];
    }
    __syncthreads();
  }
}

__global__ void initialize_ledger_kernel(std::uint32_t budget, std::uint32_t seed,
                                         std::uint32_t* ledger, std::uint32_t* rng) {
  if (blockIdx.x == 0u && threadIdx.x == 0u) {
    ledger[0] = budget;
    ledger[1] = budget;
    ledger[2] = 0u;
    ledger[3] = 1u;
    rng[0] = seed == 0u ? 0x6d2b79f5u : seed;
  }
}

__global__ void debit_counts_kernel(const std::uint32_t* counts, std::uint32_t count,
                                    std::uint32_t* ledger) {
  __shared__ std::uint32_t block_mass[kBlock];
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  block_mass[threadIdx.x] = i < count ? counts[i] : 0u;
  __syncthreads();
  for (std::uint32_t offset = blockDim.x / 2u; offset != 0u; offset >>= 1u) {
    if (threadIdx.x < offset) block_mass[threadIdx.x] += block_mass[threadIdx.x + offset];
    __syncthreads();
  }
  if (threadIdx.x == 0u) {
    atomicSub(ledger + 1u, block_mass[0]);
    atomicAdd(ledger + 2u, block_mass[0]);
  }
}

__global__ void audit_ledger_kernel(const std::uint32_t* vitality, std::uint32_t unit_count,
                                    const std::uint32_t* bigram_counts,
                                    std::uint32_t bigram_count,
                                    const std::uint32_t* trigram_counts,
                                    std::uint32_t trigram_count,
                                    const std::uint32_t* online_bigram_counts,
                                    std::uint32_t online_bigram_count,
                                    const std::uint32_t* online_trigram_counts,
                                    std::uint32_t online_trigram_count,
                                    const std::uint32_t* association_counts,
                                    std::uint32_t association_count,
                                    const std::uint32_t* conditioned_transition_counts,
                                    std::uint32_t conditioned_transition_count,
                                    std::uint32_t episode_count,
                                    const std::uint32_t* boundary_histogram,
                                    const std::uint32_t* boundary_pairs,
                                    std::uint32_t* ledger) {
  __shared__ unsigned long long sums[kBlock];
  unsigned long long local = 0u;
  for (std::uint32_t i = threadIdx.x; i < unit_count; i += blockDim.x) local += vitality[i];
  for (std::uint32_t i = threadIdx.x; i < bigram_count; i += blockDim.x) local += bigram_counts[i];
  for (std::uint32_t i = threadIdx.x; i < trigram_count; i += blockDim.x) local += trigram_counts[i];
  for (std::uint32_t i = threadIdx.x; i < online_bigram_count; i += blockDim.x) {
    local += online_bigram_counts[i];
  }
  for (std::uint32_t i = threadIdx.x; i < online_trigram_count; i += blockDim.x) {
    local += online_trigram_counts[i];
  }
  for (std::uint32_t i = threadIdx.x; i < association_count; i += blockDim.x) {
    local += association_counts[i];
  }
  for (std::uint32_t i = threadIdx.x; i < conditioned_transition_count;
       i += blockDim.x) {
    local += conditioned_transition_counts[i];
  }
  for (std::uint32_t i = threadIdx.x; i < episode_count; i += blockDim.x) ++local;
  for (std::uint32_t i = threadIdx.x; i < 256u; i += blockDim.x) {
    local += boundary_histogram[i];
  }
  for (std::uint32_t i = threadIdx.x; i < 256u * 256u; i += blockDim.x) {
    local += boundary_pairs[i];
  }
  sums[threadIdx.x] = local;
  __syncthreads();
  for (std::uint32_t offset = blockDim.x / 2u; offset != 0u; offset >>= 1u) {
    if (threadIdx.x < offset) sums[threadIdx.x] += sums[threadIdx.x + offset];
    __syncthreads();
  }
  if (threadIdx.x == 0u) {
    const unsigned long long resident = sums[0];
    ledger[3] = resident <= 0xffffffffull &&
                ledger[0] == ledger[1] + ledger[2] && ledger[2] == resident;
  }
}

__global__ void lesion_counts_kernel(std::uint32_t* counts, std::uint32_t count,
                                     std::uint32_t* ledger) {
  __shared__ std::uint32_t returned[kBlock];
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  const std::uint32_t mass = i < count ? counts[i] : 0u;
  if (i < count) counts[i] = 0u;
  returned[threadIdx.x] = mass;
  __syncthreads();
  for (std::uint32_t offset = blockDim.x / 2u; offset != 0u; offset >>= 1u) {
    if (threadIdx.x < offset) returned[threadIdx.x] += returned[threadIdx.x + offset];
    __syncthreads();
  }
  if (threadIdx.x == 0u) {
    atomicAdd(ledger + 1u, returned[0]);
    atomicSub(ledger + 2u, returned[0]);
  }
}

__global__ void lesion_episode_kernel(std::uint32_t* mutable_sizes,
                                      std::uint32_t* ledger) {
  if (blockIdx.x == 0u && threadIdx.x == 0u) {
    const std::uint32_t returned = mutable_sizes[4];
    mutable_sizes[4] = 0u;
    mutable_sizes[5] = 0u;
    ledger[1] += returned;
    ledger[2] -= returned;
  }
}

__global__ void lesion_recent_episode_kernel(std::uint32_t* mutable_sizes,
                                             std::uint32_t episode_begin,
                                             std::uint32_t break_begin,
                                             std::uint32_t* ledger) {
  if (blockIdx.x == 0u && threadIdx.x == 0u) {
    const std::uint32_t returned = mutable_sizes[4] - episode_begin;
    mutable_sizes[4] = episode_begin;
    mutable_sizes[5] = break_begin;
    ledger[1] += returned;
    ledger[2] -= returned;
  }
}

__global__ void normalize_episode_breaks_kernel(const std::uint32_t* source,
                                                std::uint32_t count,
                                                std::uint32_t episode_begin,
                                                std::uint32_t* normalized) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count) normalized[i] = source[i] - episode_begin;
}

__global__ void offset_local_seed_candidates_kernel(LocalSeedCandidate* candidates,
                                                    std::uint32_t count,
                                                    std::uint32_t episode_begin) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count || candidates[i].score == 0u) return;
  candidates[i].position += episode_begin;
  candidates[i].anchor_position += episode_begin;
  candidates[i].local_launch_end += episode_begin;
}

__global__ void return_fixed_mass_kernel(std::uint32_t mass, std::uint32_t* ledger) {
  if (blockIdx.x == 0u && threadIdx.x == 0u) {
    ledger[1] += mass;
    ledger[2] -= mass;
  }
}

__global__ void reserve_fixed_mass_kernel(std::uint32_t mass, std::uint32_t* ledger,
                                          std::uint32_t* status) {
  if (blockIdx.x == 0u && threadIdx.x == 0u) {
    status[0] = ledger[1] < mass ? 1u : 0u;
    if (status[0] == 0u) {
      ledger[1] -= mass;
      ledger[2] += mass;
    }
  }
}

__global__ void reserve_fixed_mass_if_novel_kernel(
    std::uint32_t mass, const std::uint32_t* exact_replay,
    std::uint32_t* ledger, std::uint32_t* status) {
  if (blockIdx.x == 0u && threadIdx.x == 0u) {
    status[0] = exact_replay[0] == 0u && ledger[1] < mass ? 1u : 0u;
    if (status[0] == 0u && exact_replay[0] == 0u) {
      ledger[1] -= mass;
      ledger[2] += mass;
    }
  }
}

#include "bcc32_cuda_adult_raw_assimilation_kernels.inl"

__device__ std::uint32_t fuzzy_unit_score(const std::uint8_t* bytes, std::uint32_t begin,
                                          std::uint32_t length, std::uint32_t unit,
                                          const std::uint32_t* unit_lengths,
                                          const std::uint32_t* unit_content) {
  const std::uint32_t other_length = unit_lengths[unit];
  const std::uint32_t common = min(length, other_length);
  std::uint32_t prefix = 0u;
  std::uint32_t positional = 0u;
  bool prefix_open = true;
  for (std::uint32_t i = 0u; i < common; ++i) {
    const std::uint32_t packed = unit_content[unit * kUnitWords + i / 4u];
    const bool equal = bytes[begin + i] == static_cast<std::uint8_t>(packed >> ((i % 4u) * 8u));
    positional += equal;
    if (prefix_open && equal) ++prefix; else prefix_open = false;
  }
  std::uint32_t normalized_common = common;
  std::uint32_t extent = max(length, other_length);
  // Identity modulo trailing boundary bytes (the same trimming rule the
  // exact-cue marker applies): "revolution?" and "revolution " are the SAME
  // word for cue identity, and must score a true 65535 so the exact-cue
  // marker fires for the word family.
  auto trailing_boundary_byte = [](std::uint32_t b) {
    return b == static_cast<std::uint32_t>(' ') ||
           b == static_cast<std::uint32_t>('\t') ||
           b == static_cast<std::uint32_t>('\n') ||
           b == static_cast<std::uint32_t>('\r') ||
           b == static_cast<std::uint32_t>('.') ||
           b == static_cast<std::uint32_t>(',') ||
           b == static_cast<std::uint32_t>('!') ||
           b == static_cast<std::uint32_t>('?') ||
           b == static_cast<std::uint32_t>(';') ||
           b == static_cast<std::uint32_t>(':');
  };
  std::uint32_t segment_core = length;
  while (segment_core != 0u &&
         trailing_boundary_byte(bytes[begin + segment_core - 1u]))
    --segment_core;
  std::uint32_t unit_core = other_length;
  while (unit_core != 0u) {
    const std::uint32_t packed =
        unit_content[unit * kUnitWords + (unit_core - 1u) / 4u];
    if (!trailing_boundary_byte((packed >> (((unit_core - 1u) % 4u) * 8u)) &
                                0xffu))
      break;
    --unit_core;
  }
  const bool trimmed_exact =
      segment_core != 0u && segment_core == unit_core && prefix >= segment_core;
  if (trimmed_exact) return 65535u;
  const bool byte_exact = length == other_length && positional == common;
  bool forgiven = false;
  if (length == other_length && common > 1u && prefix + 1u == common &&
      positional + 1u == common) {
    normalized_common = common - 1u;
    extent -= 1u;
    forgiven = true;
  } else if (!byte_exact && prefix >= 3u && prefix + 1u >= common &&
             max(length, other_length) - min(length, other_length) <= 1u) {
    normalized_common = prefix;
    positional = prefix;
    extent = prefix;
    forgiven = true;
  }
  if (extent == 0u) return 0u;
  const unsigned long long matched =
      prefix * 16u + positional * 4u + normalized_common;
  std::uint32_t score =
      static_cast<std::uint32_t>(matched * 65535ull / (21ull * extent));
  // A forgiven (suffix/interior-tolerant) match must rank STRICTLY below a
  // byte-exact one. Both previously scored a perfect 65535, and the
  // lowest-unit-id tie-break then let a same-length neighbor STEAL the cue
  // word's own best-match slot ('the'->'then', 'was'->'wash'), stamping
  // exact-cue identity on the wrong word family -- the measured cause of
  // the topic latch failing to ground ('then' out-ranking 'revolution').
  if (forgiven && score > 65534u) score = 65534u;
  return score;
}

#include "bcc32_cuda_adult_cue_matching_association.inl"

#include "bcc32_cuda_adult_cue_contact_order.inl"
#include "bcc32_cuda_adult_base_completion_policy.inl"

#include "bcc32_cuda_adult_completion_evidence.inl"
#include "bcc32_cuda_adult_resident_relation_composition.inl"

#include "bcc32_cuda_adult_resident_synthesis_surface.inl"
#include "bcc32_cuda_adult_conditioned_relation_bindings.inl"

#include "bcc32_cuda_adult_conditioned_surface_learning.inl"
#include "bcc32_cuda_adult_generation_kernel.inl"

#include "bcc32_cuda_adult_index_consolidation.inl"

#include "bcc32_cuda_adult_training_pipeline.inl"

#include "bcc32_cuda_adult_v1_episode_lesion.inl"

#include "bcc32_cuda_adult_raw_assimilation_pipeline.inl"

#include "bcc32_cuda_adult_contact_control.inl"
#include "bcc32_cuda_adult_v1_checkpoint.inl"

inline std::vector<std::uint8_t> generate_construction_reply(
    AdultState& state, std::uint32_t output_bytes,
    std::uint32_t subject_field_count, bool allow_completion);

static __global__ void collect_subject_proposition_population_kernel(
    const std::uint32_t* subject_units, std::uint32_t subject_count,
    const std::uint32_t* unit_population,
    std::uint32_t unit_count, std::uint32_t population_width,
    std::uint32_t* cue_cells, std::uint32_t cue_capacity,
    std::uint32_t* cue_count) {
  if (blockIdx.x != 0u || threadIdx.x != 0u) return;
  cue_count[0] = 0u;
  for (std::uint32_t subject = 0u; subject < subject_count; ++subject) {
    const std::uint32_t unit = subject_units[subject];
    if (unit >= unit_count) continue;
    for (std::uint32_t offset = 0u; offset < population_width; ++offset) {
      const std::uint32_t cell =
          unit_population[static_cast<std::size_t>(unit) * population_width + offset];
      bool present = false;
      for (std::uint32_t prior = 0u; prior < cue_count[0]; ++prior)
        present = present || cue_cells[prior] == cell;
      if (present) continue;
      if (cue_count[0] >= cue_capacity) return;
      cue_cells[cue_count[0]++] = cell;
    }
  }
}

static __global__ void collect_exact_cue_proposition_population_kernel(
    const std::uint32_t* cue_exact, const std::uint32_t* cue_scores,
    const std::uint32_t* cue_orders, const std::uint32_t* unit_population,
    std::uint32_t unit_count, std::uint32_t population_width,
    const std::uint32_t* exact_sequence,
    const std::uint32_t* exact_sequence_count,
    const roles::MutableStructuralRole* unit_roles,
    const std::uint32_t* construction_tokens,
    const std::uint32_t* construction_lengths,
    const std::uint32_t* construction_slot_counts,
    const std::uint32_t* construction_supports,
    const std::uint32_t* construction_count,
    std::uint32_t* cue_cells, std::uint32_t cue_capacity,
    std::uint32_t* cue_count, std::uint32_t* ordered_units,
    std::uint32_t ordered_capacity, std::uint32_t* ordered_count,
    std::uint32_t* completion_role, std::uint32_t* completion_construction,
    std::uint32_t* completion_slot) {
  if (blockIdx.x != 0u || threadIdx.x != 0u) return;
  cue_count[0] = 0u;
  ordered_count[0] = 0u;
  completion_role[0] = 0xffffffffu;
  completion_construction[0] = construction::kNoConstruction;
  completion_slot[0] = 0xffffffffu;
  if (exact_sequence != nullptr && exact_sequence_count != nullptr &&
      unit_roles != nullptr && construction_tokens != nullptr &&
      construction_lengths != nullptr && construction_slot_counts != nullptr &&
      construction_supports != nullptr && construction_count != nullptr) {
    const std::uint32_t sequence_count = exact_sequence_count[0];
    const std::uint32_t count =
        construction_count[0] < construction::kConstructionCap
            ? construction_count[0]
            : construction::kConstructionCap;
    std::uint32_t selected_prefix = 0u;
    std::uint32_t selected_support = 0u;
    std::uint32_t selected_role = 0xffffffffu;
    std::uint32_t selected_construction = construction::kNoConstruction;
    bool selected_ambiguous = false;
    for (std::uint32_t index = 0u; index < count; ++index) {
      const std::uint32_t extent = construction_lengths[index];
      const std::uint32_t slots = construction_slot_counts[index];
      if (construction_supports[index] < construction::kConstructionMinRoleEvidence ||
          slots != extent || slots < 2u ||
          slots > construction::kConstructionMaxSlots)
        continue;
      const std::uint32_t prefix = slots - 1u;
      if (prefix > sequence_count || prefix > ordered_capacity) continue;
      bool compatible = true;
      for (std::uint32_t position = 0u; position < prefix; ++position) {
        const std::uint32_t unit =
            exact_sequence[sequence_count - prefix + position];
        const std::uint32_t token =
            construction_tokens[index * construction::kConstructionMaxTokens +
                                position];
        compatible = compatible && unit < unit_count &&
                     construction::token_is_slot(token) &&
                     unit_roles[unit].confidence != 0u &&
                     unit_roles[unit].role == construction::token_role(token);
      }
      if (!compatible) continue;
      const std::uint32_t completion_token =
          construction_tokens[index * construction::kConstructionMaxTokens + prefix];
      if (!construction::token_is_slot(completion_token)) continue;
      const std::uint32_t candidate_role = construction::token_role(completion_token);
      const std::uint32_t support = construction_supports[index];
      if (selected_prefix == 0u || prefix > selected_prefix ||
          (prefix == selected_prefix && support > selected_support)) {
        selected_prefix = prefix;
        selected_support = support;
        selected_role = candidate_role;
        selected_construction = index;
        selected_ambiguous = false;
      } else if (prefix == selected_prefix && support == selected_support &&
                 index != selected_construction) {
        selected_ambiguous = true;
      }
    }
    if (selected_prefix != 0u && !selected_ambiguous) {
      completion_role[0] = selected_role;
      completion_construction[0] = selected_construction;
      completion_slot[0] = selected_prefix;
      for (std::uint32_t position = 0u; position < selected_prefix; ++position) {
        const std::uint32_t unit =
            exact_sequence[sequence_count - selected_prefix + position];
        ordered_units[ordered_count[0]++] = unit;
        for (std::uint32_t offset = 0u; offset < population_width; ++offset) {
          const std::uint32_t cell = unit_population[
              static_cast<std::size_t>(unit) * population_width + offset];
          bool present = false;
          for (std::uint32_t prior = 0u; prior < cue_count[0]; ++prior)
            present = present || cue_cells[prior] == cell;
          if (present) continue;
          if (cue_count[0] >= cue_capacity) return;
          cue_cells[cue_count[0]++] = cell;
        }
      }
      return;
    }
  }
  std::uint32_t maximum_order = 0u;
  for (std::uint32_t unit = 0u; unit < unit_count; ++unit)
    if (cue_orders[unit] != 0xffffffffu && cue_orders[unit] > maximum_order)
      maximum_order = cue_orders[unit];
  std::uint32_t learned_prefix_width = 0u;
  if (construction_lengths != nullptr && construction_slot_counts != nullptr &&
      construction_supports != nullptr && construction_count != nullptr) {
    const std::uint32_t count =
        construction_count[0] < construction::kConstructionCap
            ? construction_count[0]
            : construction::kConstructionCap;
    for (std::uint32_t index = 0u; index < count; ++index) {
      const std::uint32_t extent = construction_lengths[index];
      const std::uint32_t slots = construction_slot_counts[index];
      if (construction_supports[index] < construction::kConstructionMinRoleEvidence ||
          slots != extent || slots < 2u ||
          slots > construction::kConstructionMaxSlots)
        continue;
      if (slots - 1u > learned_prefix_width)
        learned_prefix_width = slots - 1u;
    }
  }
  if (learned_prefix_width == 0u || learned_prefix_width > maximum_order)
    learned_prefix_width = maximum_order;
  const std::uint32_t first_order =
      maximum_order >= learned_prefix_width
          ? maximum_order - learned_prefix_width + 1u
          : 1u;
  for (std::uint32_t order = first_order; order <= maximum_order; ++order) {
    std::uint32_t selected = 0xffffffffu;
    std::uint32_t selected_score = 0u;
    for (std::uint32_t unit = 0u; unit < unit_count; ++unit) {
      if (cue_orders[unit] != order) continue;
      const std::uint32_t score = cue_scores[unit];
      if (score > selected_score ||
          (score == selected_score && score != 0u && unit < selected)) {
        selected = unit;
        selected_score = score;
      }
    }
    if (selected == 0xffffffffu) continue;
    const std::uint32_t unit = selected;
    if (ordered_count[0] < ordered_capacity)
      ordered_units[ordered_count[0]++] = unit;
    for (std::uint32_t offset = 0u; offset < population_width; ++offset) {
      const std::uint32_t cell =
          unit_population[static_cast<std::size_t>(unit) * population_width + offset];
      bool present = false;
      for (std::uint32_t prior = 0u; prior < cue_count[0]; ++prior)
        present = present || cue_cells[prior] == cell;
      if (present) continue;
      if (cue_count[0] >= cue_capacity) return;
      cue_cells[cue_count[0]++] = cell;
    }
  }
  if (cue_count[0] != 0u) return;
  for (std::uint32_t unit = 0u; unit < unit_count; ++unit) {
    if (cue_exact[unit] == 0u) continue;
    if (ordered_count[0] < ordered_capacity)
      ordered_units[ordered_count[0]++] = unit;
    for (std::uint32_t offset = 0u; offset < population_width; ++offset) {
      const std::uint32_t cell =
          unit_population[static_cast<std::size_t>(unit) * population_width + offset];
      bool present = false;
      for (std::uint32_t prior = 0u; prior < cue_count[0]; ++prior)
        present = present || cue_cells[prior] == cell;
      if (present) continue;
      if (cue_count[0] >= cue_capacity) return;
      cue_cells[cue_count[0]++] = cell;
    }
  }
}

static __global__ void ground_proposition_completion_units_kernel(
    const std::uint32_t* completed_cells,
    const std::uint64_t* completed_scores, std::uint32_t completed_count,
    const std::uint32_t* unit_population, std::uint32_t unit_count,
    std::uint32_t population_width,
    const roles::MutableStructuralRole* unit_roles,
    const std::uint32_t* completion_role,
    const std::uint32_t* completion_construction,
    const std::uint32_t* completion_slot,
    const std::uint32_t* construction_slot_units,
    const std::uint32_t* construction_slot_masses,
    const std::uint32_t* construction_slot_overflow,
    std::uint32_t* resident_rng,
    std::uint32_t* anchor_units,
    std::uint32_t anchor_capacity, std::uint32_t initial_count,
    std::uint32_t* anchor_count) {
  if (blockIdx.x != 0u || threadIdx.x != 0u) return;
  anchor_count[0] = initial_count;
  if (completion_construction == nullptr || completion_slot == nullptr ||
      construction_slot_units == nullptr || construction_slot_masses == nullptr ||
      construction_slot_overflow == nullptr || resident_rng == nullptr ||
      completion_construction[0] >= construction::kConstructionCap ||
      completion_slot[0] >= construction::kConstructionMaxSlots)
    return;
  const std::uint32_t construction_index = completion_construction[0];
  const std::uint32_t slot = completion_slot[0];
  if (construction_slot_overflow[
          construction::construction_slot_index(construction_index, slot)] != 0u)
    return;
  for (std::uint32_t rank = initial_count; rank < anchor_capacity; ++rank) {
    std::uint32_t best_unit = 0xffffffffu;
    std::uint64_t best_weight_high = 0u;
    std::uint64_t best_weight_low = 0u;
    std::uint32_t tied_candidates = 0u;
    for (std::uint32_t member = 0u;
         member < construction::kConstructionSlotPopulationCap; ++member) {
      const std::size_t member_index = construction::construction_slot_member_index(
          construction_index, slot, member);
      const std::uint32_t unit = construction_slot_units[member_index];
      const std::uint32_t slot_mass = construction_slot_masses[member_index];
      if (unit >= unit_count || slot_mass == 0u)
        continue;
      bool used = false;
      for (std::uint32_t prior = 0u; prior < anchor_count[0]; ++prior)
        used = used || anchor_units[prior] == unit;
      if (used) continue;
      const bool role_constrained =
          completion_role != nullptr && completion_role[0] != 0xffffffffu;
      if (role_constrained &&
          (unit_roles[unit].confidence == 0u ||
           unit_roles[unit].role != completion_role[0]))
        continue;
      std::uint64_t proposition_score = 0u;
      for (std::uint32_t offset = 0u; offset < population_width; ++offset) {
        const std::uint32_t cell =
            unit_population[static_cast<std::size_t>(unit) * population_width + offset];
        for (std::uint32_t completed = 0u; completed < completed_count; ++completed) {
          if (cell == completed_cells[completed])
            proposition_score += completed_scores[completed];
        }
      }
      if (proposition_score == 0u)
        continue;
      const std::uint64_t weight_high =
          __umul64hi(proposition_score, static_cast<std::uint64_t>(slot_mass));
      const std::uint64_t weight_low =
          proposition_score * static_cast<std::uint64_t>(slot_mass);
      if (best_unit == 0xffffffffu || weight_high > best_weight_high ||
          (weight_high == best_weight_high && weight_low > best_weight_low)) {
        best_unit = unit;
        best_weight_high = weight_high;
        best_weight_low = weight_low;
        tied_candidates = 1u;
      } else if (weight_high == best_weight_high && weight_low == best_weight_low) {
        ++tied_candidates;
        std::uint32_t random = resident_rng[0];
        random ^= random << 13u;
        random ^= random >> 17u;
        random ^= random << 5u;
        resident_rng[0] = random;
        if (random % tied_candidates == 0u)
          best_unit = unit;
      }
    }
    if (best_unit == 0xffffffffu) break;
    anchor_units[anchor_count[0]++] = best_unit;
  }
}

inline std::vector<std::uint8_t> realize_resident_proposition_construction(
    AdultState& state, proposition_tissue::SparsePopulationView cue,
    std::uint32_t output_bytes, bool discourse_evidence = true,
    const std::uint32_t* committed_units = nullptr,
    std::uint32_t committed_count = 0u,
    const std::uint32_t* learned_completion_role = nullptr,
    const std::uint32_t* learned_completion_construction = nullptr,
    const std::uint32_t* learned_completion_slot = nullptr) {
  const bool trace = std::getenv("BCC32_PROPOSITION_CONSTRUCTION_TRACE") != nullptr;
  if (trace)
    std::fprintf(stderr,
                 "proposition_construction enter surface=%u lesioned=%u store=%u cue=%u bytes=%u populations=%u\n",
                 state.surface_organ_enabled ? 1u : 0u,
                 state.construction_lesioned ? 1u : 0u,
                 state.construction_count_host, cue.count, output_bytes,
                 state.surface_unit_population.get() != nullptr ? 1u : 0u);
  if (!state.surface_organ_enabled || state.construction_lesioned ||
      state.construction_count_host == 0u || cue.count == 0u ||
      output_bytes == 0u || state.surface_unit_population.get() == nullptr)
    return {};
  const proposition_tissue::CompletionResult completion =
      discourse_evidence ? complete_resident_discourse_proposition(state, cue)
                         : complete_resident_proposition(state, cue);
  if (trace)
    std::fprintf(stderr,
                 "proposition_construction settle cue=%u ready=%u cells=%u qualified=%u uncertain=%llu\n",
                 cue.count, completion.ready, completion.output_count,
                 completion.qualified_synapses,
                 static_cast<unsigned long long>(completion.uncertain_mass));
  if (completion.ready == 0u || completion.output_count == 0u) return {};

  if (committed_count >= construction::kConstructionMaxSlots) return {};
  DeviceArray<std::uint32_t> anchor_units(construction::kConstructionMaxSlots);
  DeviceArray<std::uint32_t> anchor_count(1u);
  if (committed_count != 0u) {
    cuda_require(cudaMemcpy(anchor_units.get(), committed_units,
                            committed_count * sizeof(std::uint32_t),
                            cudaMemcpyDeviceToDevice),
                 "retain ordered proposition cue units");
  }
  const std::uint32_t anchor_capacity = committed_count + 1u;
  const std::uint32_t* completed_cells = state.proposition_output_cells.get();
  const std::uint64_t* completed_scores = state.proposition_output_scores.get();
  std::uint32_t completed_count = completion.output_count;
  const std::uint32_t* unit_population = state.surface_unit_population.get();
  std::uint32_t unit_count = state.unit_count;
  std::uint32_t population_width = kDistributedMotorActiveWidth;
  const roles::MutableStructuralRole* unit_roles = state.construction_roles.get();
  const std::uint32_t* completion_role = learned_completion_role;
  const std::uint32_t* completion_construction = learned_completion_construction;
  const std::uint32_t* completion_slot = learned_completion_slot;
  const std::uint32_t* construction_slot_units = state.construction_slot_units.get();
  const std::uint32_t* construction_slot_masses = state.construction_slot_masses.get();
  const std::uint32_t* construction_slot_overflow =
      state.construction_slot_overflow.get();
  std::uint32_t* resident_rng = state.rng.get();
  std::uint32_t* anchor_units_device = anchor_units.get();
  std::uint32_t anchor_capacity_value = anchor_capacity;
  std::uint32_t initial_count = committed_count;
  std::uint32_t* anchor_count_device = anchor_count.get();
  void* kernel_arguments[] = {
      &completed_cells, &completed_scores, &completed_count, &unit_population,
      &unit_count, &population_width, &unit_roles, &completion_role,
      &completion_construction, &completion_slot, &construction_slot_units,
      &construction_slot_masses, &construction_slot_overflow, &resident_rng,
      &anchor_units_device, &anchor_capacity_value, &initial_count,
      &anchor_count_device};
  cuda_require(
      cudaLaunchKernel(
          reinterpret_cast<const void*>(ground_proposition_completion_units_kernel),
          dim3{1u, 1u, 1u}, dim3{2u, 1u, 1u}, kernel_arguments, 0u, nullptr),
      "launch ground resident proposition completion units");
  cuda_require(cudaGetLastError(), "ground resident proposition completion units");
  std::uint32_t grounded_count = 0u;
  cuda_require(cudaMemcpy(&grounded_count, anchor_count.get(), sizeof(grounded_count),
                          cudaMemcpyDeviceToHost),
               "read resident proposition anchor count");
  if (trace)
    std::fprintf(stderr, "proposition_construction grounded=%u\n", grounded_count);
  if (grounded_count == 0u) return {};
  if (trace) {
    std::vector<std::uint32_t> host_anchors(grounded_count);
    cuda_require(cudaMemcpy(host_anchors.data(), anchor_units.get(),
                            grounded_count * sizeof(std::uint32_t),
                            cudaMemcpyDeviceToHost),
                 "trace proposition anchor units");
    std::fprintf(stderr, "proposition_construction anchor_roles");
    for (const std::uint32_t unit : host_anchors) {
      roles::MutableStructuralRole role{};
      cuda_require(cudaMemcpy(&role, state.construction_roles.get() + unit,
                              sizeof(role), cudaMemcpyDeviceToHost),
                   "trace proposition anchor role");
      std::fprintf(stderr, " u%u:r%u/c%u", unit, role.role, role.confidence);
    }
    std::fprintf(stderr, "\nproposition_construction skeleton_roles");
    const std::uint32_t inspected =
        std::min<std::uint32_t>(state.construction_count_host, 16u);
    for (std::uint32_t index = 0u; index < inspected; ++index) {
      std::uint32_t length = 0u;
      cuda_require(cudaMemcpy(&length, state.construction_lengths.get() + index,
                              sizeof(length), cudaMemcpyDeviceToHost),
                   "trace proposition construction length");
      std::vector<std::uint32_t> tokens(length);
      cuda_require(cudaMemcpy(
                       tokens.data(),
                       state.construction_tokens.get() +
                           static_cast<std::size_t>(index) *
                               construction::kConstructionMaxTokens,
                       length * sizeof(std::uint32_t), cudaMemcpyDeviceToHost),
                   "trace proposition construction roles");
      std::fprintf(stderr, " c%u[", index);
      for (const std::uint32_t token : tokens)
        std::fprintf(stderr, "%s%u,", construction::token_is_slot(token) ? "r" : "u",
                     construction::token_is_slot(token)
                         ? construction::token_role(token)
                         : token);
      std::fprintf(stderr, "]");
    }
    std::fprintf(stderr, "\n");
  }
  const std::uint32_t byte_capacity = std::min(
      output_bytes, static_cast<std::uint32_t>(state.surface_output_bytes.size()));
  const surface_organ::OpaqueConstructionWitnessView witness{
      state.construction_tokens.get(), state.construction_lengths.get(),
      state.construction_slot_counts.get(), state.construction_supports.get(),
      state.construction_store_count.get(), state.construction_roles.get(),
      construction::kConstructionCap, learned_completion_construction,
      state.construction_slot_units.get(), state.construction_slot_masses.get(),
      state.construction_slot_overflow.get(),
      state.construction_closed_class_mask.get()};
  cuda_require(cudaMemset(state.motor_context.get(), 0, state.motor_context.bytes()),
               "clear resident proposition Plan metadata");
  cuda_require(cudaMemcpy(state.motor_completion.get(), anchor_units.get(),
                          grounded_count * sizeof(std::uint32_t),
                          cudaMemcpyDeviceToDevice),
               "commit resident proposition Plan anchors");
  cuda_require(cudaMemcpy(state.motor_context.get() + 3u, &grounded_count,
                          sizeof(grounded_count), cudaMemcpyHostToDevice),
               "commit resident proposition Plan extent");
  if (learned_completion_construction != nullptr) {
    cuda_require(cudaMemcpy(state.motor_context.get() + 4u,
                            learned_completion_construction,
                            sizeof(std::uint32_t), cudaMemcpyDeviceToDevice),
                 "commit resident proposition construction reference");
  }
  const surface_organ::OpaqueContentPlanView plan{
      state.motor_completion.get(), grounded_count};
  const surface_organ::SurfaceRealizationWorkspaceView workspace{
      state.surface_bridges.get(), state.surface_prefixes.get(),
      state.surface_suffixes.get(), state.surface_permutation_scores.get(),
      state.surface_permutation_valid.get(),
      static_cast<std::uint32_t>(state.surface_permutation_scores.size()),
      state.surface_output_units.get(), state.surface_output_anchor_mask.get(),
      static_cast<std::uint32_t>(state.surface_output_units.size()),
      state.surface_output_bytes.get(), byte_capacity, state.surface_result.get()};
  cuda_require(surface_organ::realize_surface_construction_cuda(
                   surface_unit_view(state), witness, plan, workspace),
               "realize resident proposition through learned construction");
  cuda_require(cudaDeviceSynchronize(),
               "complete resident proposition construction realization");
  surface_organ::SurfaceOrganResult result{};
  cuda_require(cudaMemcpy(&result, state.surface_result.get(), sizeof(result),
                          cudaMemcpyDeviceToHost),
               "read resident proposition construction result");
  if (trace)
    std::fprintf(stderr,
                 "proposition_construction surface ready=%u grammar=%u anchors=%u/%u units=%u bytes=%u\n",
                 result.ready, result.grammar_supported, result.anchors_preserved,
                 grounded_count, result.output_unit_count, result.output_byte_count);
  if (result.ready == 0u || result.grammar_supported == 0u ||
      result.anchors_preserved != grounded_count ||
      result.output_byte_count == 0u || result.output_byte_count > byte_capacity)
    return {};
  std::vector<std::uint8_t> output(result.output_byte_count);
  cuda_require(cudaMemcpy(output.data(), state.surface_output_bytes.get(), output.size(),
                          cudaMemcpyDeviceToHost),
               "read resident proposition construction bytes");
  cuda_require(cudaMemset(state.motor_context.get(), 0, state.motor_context.bytes()),
               "consume resident proposition Plan");
  return output;
}

inline std::vector<std::uint8_t> realize_unit_proposition_construction(
    AdultState& state, const std::uint32_t* units, std::uint32_t unit_extent,
    std::uint32_t output_bytes) {
  if (units == nullptr || unit_extent == 0u ||
      state.surface_unit_population.get() == nullptr)
    return {};
  DeviceArray<std::uint32_t> cue_cells(proposition_tissue::kMaximumPopulationCells);
  DeviceArray<std::uint32_t> cue_count(1u);
  const std::uint32_t* subject_units = units;
  std::uint32_t subject_count = unit_extent;
  const std::uint32_t* unit_population = state.surface_unit_population.get();
  std::uint32_t unit_count = state.unit_count;
  std::uint32_t population_width = kDistributedMotorActiveWidth;
  std::uint32_t* cue_cells_device = cue_cells.get();
  std::uint32_t cue_capacity = proposition_tissue::kMaximumPopulationCells;
  std::uint32_t* cue_count_device = cue_count.get();
  void* kernel_arguments[] = {
      &subject_units, &subject_count, &unit_population, &unit_count,
      &population_width, &cue_cells_device, &cue_capacity, &cue_count_device};
  cuda_require(
      cudaLaunchKernel(
          reinterpret_cast<const void*>(collect_subject_proposition_population_kernel),
          dim3{1u, 1u, 1u}, dim3{2u, 1u, 1u}, kernel_arguments, 0u, nullptr),
      "launch collect subject proposition population");
  cuda_require(cudaGetLastError(), "collect subject proposition population");
  std::uint32_t count = 0u;
  cuda_require(cudaMemcpy(&count, cue_count.get(), sizeof(count), cudaMemcpyDeviceToHost),
               "read subject proposition population count");
  if (count == 0u) return {};
  return realize_resident_proposition_construction(
      state, proposition_tissue::SparsePopulationView{cue_cells.get(), count},
      output_bytes, true);
}

inline std::vector<std::uint8_t> realize_subject_proposition_construction(
    AdultState& state, std::uint32_t subject_count,
    std::uint32_t output_bytes) {
  return realize_unit_proposition_construction(
      state, state.subject_ids.get(), subject_count, output_bytes);
}

inline std::vector<std::uint8_t> realize_exact_cue_proposition_construction(
    AdultState& state, std::uint32_t output_bytes) {
  if (state.relation_cue_exact.get() == nullptr ||
      state.surface_unit_population.get() == nullptr)
    return {};
  DeviceArray<std::uint32_t> cue_cells(proposition_tissue::kMaximumPopulationCells);
  DeviceArray<std::uint32_t> cue_count(1u);
  DeviceArray<std::uint32_t> cue_units(kCueAnchorLimit);
  DeviceArray<std::uint32_t> cue_unit_count(1u);
  DeviceArray<std::uint32_t> completion_role(1u);
  DeviceArray<std::uint32_t> completion_construction(1u);
  DeviceArray<std::uint32_t> completion_slot(1u);
  const std::uint32_t* cue_exact_device = state.relation_cue_exact.get();
  const std::uint32_t* cue_scores_device = state.relation_cue_scores.get();
  const std::uint32_t* cue_orders_device = state.relation_cue_orders.get();
  const std::uint32_t* unit_population_device = state.surface_unit_population.get();
  std::uint32_t unit_count_value = state.unit_count;
  std::uint32_t population_width_value = kDistributedMotorActiveWidth;
  const std::uint32_t* exact_sequence = state.proposition_cue_sequence.get();
  const std::uint32_t* exact_sequence_count =
      state.proposition_cue_sequence_count.get();
  const roles::MutableStructuralRole* unit_roles_device =
      state.construction_roles.get();
  const std::uint32_t* construction_tokens = state.construction_tokens.get();
  const std::uint32_t* construction_lengths = state.construction_lengths.get();
  const std::uint32_t* construction_slot_counts =
      state.construction_slot_counts.get();
  const std::uint32_t* construction_supports = state.construction_supports.get();
  const std::uint32_t* construction_count = state.construction_store_count.get();
  std::uint32_t* cue_cells_device = cue_cells.get();
  std::uint32_t cue_capacity = proposition_tissue::kMaximumPopulationCells;
  std::uint32_t* cue_count_device = cue_count.get();
  std::uint32_t* ordered_units_device = cue_units.get();
  std::uint32_t ordered_capacity = kCueAnchorLimit;
  std::uint32_t* ordered_count_device = cue_unit_count.get();
  std::uint32_t* completion_role_device = completion_role.get();
  std::uint32_t* completion_construction_device = completion_construction.get();
  std::uint32_t* completion_slot_device = completion_slot.get();
  void* kernel_arguments[] = {
      &cue_exact_device,
      &cue_scores_device,
      &cue_orders_device,
      &unit_population_device,
      &unit_count_value,
      &population_width_value,
      &exact_sequence,
      &exact_sequence_count,
      &unit_roles_device,
      &construction_tokens,
      &construction_lengths,
      &construction_slot_counts,
      &construction_supports,
      &construction_count,
      &cue_cells_device,
      &cue_capacity,
      &cue_count_device,
      &ordered_units_device,
      &ordered_capacity,
      &ordered_count_device,
      &completion_role_device,
      &completion_construction_device,
      &completion_slot_device};
  cuda_require(
      cudaLaunchKernel(
          reinterpret_cast<const void*>(collect_exact_cue_proposition_population_kernel),
          dim3{1u, 1u, 1u}, dim3{2u, 1u, 1u}, kernel_arguments, 0u, nullptr),
      "launch collect exact cue proposition population");
  cuda_require(cudaGetLastError(), "collect exact cue proposition population");
  std::uint32_t count = 0u;
  cuda_require(cudaMemcpy(&count, cue_count.get(), sizeof(count), cudaMemcpyDeviceToHost),
               "read exact cue proposition population count");
  std::uint32_t unit_count = 0u;
  cuda_require(cudaMemcpy(&unit_count, cue_unit_count.get(), sizeof(unit_count),
                          cudaMemcpyDeviceToHost),
               "read ordered exact cue unit count");
  if (count == 0u || unit_count == 0u) return {};
  if (std::getenv("BCC32_PROPOSITION_CONSTRUCTION_TRACE") != nullptr) {
    std::vector<std::uint32_t> host_units(unit_count);
    std::uint32_t host_sequence_count = 0u;
    std::uint32_t host_completion_role = 0xffffffffu;
    std::uint32_t host_completion_construction = construction::kNoConstruction;
    std::uint32_t host_completion_slot = 0xffffffffu;
    cuda_require(cudaMemcpy(host_units.data(), cue_units.get(),
                            unit_count * sizeof(std::uint32_t),
                            cudaMemcpyDeviceToHost),
                 "trace ordered exact cue units");
    cuda_require(cudaMemcpy(&host_completion_role, completion_role.get(),
                            sizeof(host_completion_role), cudaMemcpyDeviceToHost),
                 "trace learned completion role");
    cuda_require(cudaMemcpy(&host_completion_construction,
                            completion_construction.get(),
                            sizeof(host_completion_construction),
                            cudaMemcpyDeviceToHost),
                 "trace learned completion construction");
    cuda_require(cudaMemcpy(&host_completion_slot, completion_slot.get(),
                            sizeof(host_completion_slot), cudaMemcpyDeviceToHost),
                 "trace learned completion slot");
    cuda_require(cudaMemcpy(&host_sequence_count,
                            state.proposition_cue_sequence_count.get(),
                            sizeof(host_sequence_count), cudaMemcpyDeviceToHost),
                 "trace exact proposition cue sequence extent");
    std::fprintf(stderr, "proposition_construction cue_units");
    for (const std::uint32_t unit : host_units) {
      roles::MutableStructuralRole role{};
      cuda_require(cudaMemcpy(&role, state.construction_roles.get() + unit,
                              sizeof(role), cudaMemcpyDeviceToHost),
                   "trace ordered exact cue role");
      std::fprintf(stderr, " u%u:r%u", unit, role.role);
    }
    std::fprintf(stderr, " completion_role=%u construction=%u slot=%u\n",
                 host_completion_role, host_completion_construction,
                 host_completion_slot);
    if (host_sequence_count != 0u) {
      std::vector<std::uint32_t> host_sequence(host_sequence_count);
      cuda_require(cudaMemcpy(host_sequence.data(),
                              state.proposition_cue_sequence.get(),
                              host_sequence_count * sizeof(std::uint32_t),
                              cudaMemcpyDeviceToHost),
                   "trace exact proposition cue sequence");
      std::fprintf(stderr, "proposition_construction exact_sequence");
      for (const std::uint32_t unit : host_sequence) {
        roles::MutableStructuralRole role{};
        cuda_require(cudaMemcpy(&role, state.construction_roles.get() + unit,
                                sizeof(role), cudaMemcpyDeviceToHost),
                     "trace exact proposition cue sequence role");
        std::fprintf(stderr, " u%u:r%u/c%u", unit, role.role, role.confidence);
      }
      std::fprintf(stderr, "\n");
    }
  }
  return realize_resident_proposition_construction(
      state, proposition_tissue::SparsePopulationView{cue_cells.get(), count},
      output_bytes, true, cue_units.get(), unit_count,
      completion_role.get(), completion_construction.get(), completion_slot.get());
}

inline std::vector<std::uint8_t> generate_resident_surface_plan(
    AdultState& state, std::uint32_t output_bytes) {
  if (!state.surface_organ_enabled || output_bytes == 0u)
    return {};
  std::uint32_t plan_count = 0u;
  cuda_require(cudaMemcpy(&plan_count, state.motor_context.get() + 3u,
                          sizeof(plan_count), cudaMemcpyDeviceToHost),
               "read resident surface plan extent");
  if (plan_count == 0u || plan_count > surface_organ::kSurfaceOrganMaxAnchors)
    return {};
  std::uint32_t subject_field_count = 0u;
  if (state.subject_count.get() != nullptr) {
    cuda_require(cudaMemcpy(&subject_field_count, state.subject_count.get(),
                            sizeof(subject_field_count), cudaMemcpyDeviceToHost),
                 "read construction subject field extent");
  }
  // Keep the same authority ordering as the raw source-free route: once the
  // adult has formed relational construction matter, the resident composer
  // must get first refusal even when the distributed motor has a surface plan.
  const bool resident_composer_ready =
      resident_construction_admission_open(state);
  if (resident_composer_ready) {
    std::vector<std::uint8_t> constructed =
        generate_construction_reply(state, output_bytes, subject_field_count,
                                    true);
    if (!constructed.empty()) return constructed;
  }
  // The distributed population has already committed opaque content units.
  // Realize that commitment through a learned construction before falling
  // back to the older greedy surface bridge.
  std::vector<std::uint8_t> proposition_constructed =
      realize_unit_proposition_construction(
          state, state.motor_completion.get(), plan_count, output_bytes);
  if (!proposition_constructed.empty()) return proposition_constructed;
  if (!resident_composer_ready) {
    std::vector<std::uint8_t> constructed = generate_construction_reply(
        state, output_bytes, subject_field_count, true);
    if (!constructed.empty()) return constructed;
  }
  const std::uint32_t byte_capacity =
      std::min(output_bytes, static_cast<std::uint32_t>(state.surface_output_bytes.size()));
  surface_organ::SurfaceOrganConfig config{};
  config.max_bridge_roles = surface_organ::kSurfaceOrganMaxBridgeRoles;
  config.min_link_probability_q20 = 1u;
  const surface_organ::SurfaceRealizationWorkspaceView workspace{
      state.surface_bridges.get(),
      state.surface_prefixes.get(),
      state.surface_suffixes.get(),
      state.surface_permutation_scores.get(),
      state.surface_permutation_valid.get(),
      static_cast<std::uint32_t>(state.surface_permutation_scores.size()),
      state.surface_output_units.get(),
      state.surface_output_anchor_mask.get(),
      static_cast<std::uint32_t>(state.surface_output_units.size()),
      state.surface_output_bytes.get(),
      byte_capacity,
      state.surface_result.get()};
  const surface_organ::OpaqueContentPlanView surface_plan{
      state.motor_completion.get(), plan_count};
  const bool has_sequence_evidence =
      state.bigram_count != 0u || state.online_bigram_count != 0u;
  const cudaError_t realization_status = has_sequence_evidence
      ? surface_organ::realize_surface_conditioned_cuda(
            surface_unit_view(state), surface_evidence_view(state),
            surface_sequence_evidence_view(state), surface_plan, config,
            workspace, true)
      : surface_organ::realize_surface_greedy_cuda(
            surface_unit_view(state), surface_evidence_view(state),
            surface_plan, config, workspace);
  cuda_require(realization_status,
               "realize resident distributed surface plan");
  cuda_require(cudaDeviceSynchronize(), "complete resident distributed surface plan");
  surface_organ::SurfaceOrganResult result{};
  cuda_require(cudaMemcpy(&result, state.surface_result.get(), sizeof(result),
                          cudaMemcpyDeviceToHost),
               "read resident surface result");
  if (result.ready == 0u || result.output_byte_count == 0u ||
      result.output_byte_count > byte_capacity)
    return {};
  std::vector<std::uint8_t> output(result.output_byte_count);
  cuda_require(cudaMemcpy(output.data(), state.surface_output_bytes.get(), output.size(),
                          cudaMemcpyDeviceToHost),
               "read resident surface bytes");
  cuda_require(cudaMemset(state.motor_context.get(), 0, state.motor_context.bytes()),
               "consume resident surface plan");
  return output;
}

#include "bcc32_cuda_adult_v1_construction_reply.inl"

inline std::vector<std::uint8_t> generate_source_free(AdultState& state,
                                                       std::uint32_t output_bytes) {
  const bool learned_only = std::getenv("BCC32_LEARNED_ONLY") != nullptr;
  if (learned_only) {
    static bool announced = false;
    if (!announced) {
      std::fprintf(stderr, "learned_only_active\n");
      announced = true;
    }
  }
  if (state.streaming_cue_mode && state.streaming_cue_meta.get() != nullptr) {
    std::uint32_t stream_meta[2] = {};
    cuda_require(cudaMemcpy(stream_meta, state.streaming_cue_meta.get(),
                            sizeof(stream_meta), cudaMemcpyDeviceToHost),
                 "read streaming cue readiness");
    if (stream_meta[0] != 0u && stream_meta[1] == 0u) return {};
  }
  std::uint32_t subject_field_count = 0u;
  if (state.subject_count.get() != nullptr) {
    cuda_require(cudaMemcpy(&subject_field_count, state.subject_count.get(),
                            sizeof(subject_field_count), cudaMemcpyDeviceToHost),
                 "read resident subject field before route selection");
  }
  // Route selection belongs to the resident adult, not to whichever downstream
  // surface projection happens to be enabled. A formed construction therefore
  // gets first refusal across all generation modes.
  if (!learned_only && resident_construction_admission_open(state)) {
    std::vector<std::uint8_t> constructed =
        generate_construction_reply(state, output_bytes, subject_field_count,
                                    false);
    if (!constructed.empty()) return constructed;
  }
  DeviceArray<std::uint8_t> device_output(output_bytes);
  DeviceArray<std::uint32_t> device_generated_count(1u);
  if (state.distributed_motor_enabled && !learned_only) {
    cuda_require(distributed_motor::generate(
                     distributed_motor_view(state), device_output.get(),
                     output_bytes, device_generated_count.get()),
                 "generate from distributed raw-event sequence motor");
    cuda_require(cudaDeviceSynchronize(),
                 "complete distributed raw-event sequence generation");
    if (!project_distributed_surface_plan(state)) {
      // A distributed surface projection is an optional downstream view. It
      // must not suppress a resident learned construction that already has a
      // real question-time cue and role evidence.
      if (resident_construction_admission_open(state)) {
        std::uint32_t subject_field_count = 0u;
        if (state.subject_count.get() != nullptr)
          cuda_require(cudaMemcpy(&subject_field_count, state.subject_count.get(),
                                  sizeof(subject_field_count),
                                  cudaMemcpyDeviceToHost),
                       "read subject field after surface projection abstention");
        return generate_construction_reply(state, output_bytes,
                                            subject_field_count, false);
      }
      return {};
    }
    std::uint32_t generated_count = 0u;
    cuda_require(cudaMemcpy(&generated_count, device_generated_count.get(),
                            sizeof(generated_count), cudaMemcpyDeviceToHost),
                 "read distributed emitted byte extent");
    if (generated_count > output_bytes) {
      throw std::runtime_error("invalid distributed emitted byte extent");
    }
    return generate_resident_surface_plan(state, output_bytes);
  }
  if (learned_only) {
    return realize_exact_cue_proposition_construction(state, output_bytes);
  }
  // A formed resident construction gets first refusal.  The older proposition
  // surface is a fallback, not an authority that may hide an untested
  // composer.  This ordering is itself observable through the bind receipt in
  // construction_last_selected and keeps the adult's learned route load-
  // bearing once its relational evidence exists.
  const bool resident_composer_ready =
      resident_construction_admission_open(state);
  if (resident_composer_ready) {
    std::vector<std::uint8_t> constructed =
        generate_construction_reply(state, output_bytes, subject_field_count,
                                    false);
    if (!constructed.empty()) return constructed;
  }
  if (subject_field_count != 0u) {
    std::vector<std::uint8_t> proposition_constructed =
        realize_subject_proposition_construction(state, subject_field_count,
                                                 output_bytes);
    if (!proposition_constructed.empty()) return proposition_constructed;
  }
  if (!resident_composer_ready) {
    std::vector<std::uint8_t> constructed =
        generate_construction_reply(state, output_bytes, subject_field_count, false);
    if (!constructed.empty()) return constructed;
  }
  generate_kernel<<<1u, kBlock>>>(
      state.unit_lengths.get(), state.unit_content.get(), state.unit_vitality.get(),
      state.unigram_top_ids.get(), state.bigrams.get(), state.bigram_counts.get(),
      state.bigram_count, state.trigrams.get(), state.trigram_counts.get(),
      state.trigram_count, state.online_bigrams.get(),
      state.online_bigram_counts.get(), state.online_bigram_count,
      state.online_trigrams.get(), state.online_trigram_counts.get(),
      state.online_trigram_count,
      state.online_conditioned_transitions.get(),
      state.online_conditioned_transition_conductance.get(),
      state.online_conditioned_transition_count,
      state.motor_context.get(), state.motor_completion.get(),
      state.subject_ids.get(), state.subject_weights.get(), subject_field_count,
      state.qonset_count.get(), state.qterm_count.get(),
      state.qorig_onset.get(), state.qorig_onset_w.get(), state.qorig_onset_n.get(),
      state.qorig_term.get(), state.qorig_term_w.get(), state.qorig_term_n.get(),
      state.qorig_on ? 1u : 0u,
      state.rng.get(), device_output.get(), output_bytes, device_generated_count.get());
  cuda_require(cudaGetLastError(), "launch source-free generator");
  cuda_require(cudaDeviceSynchronize(), "complete source-free generation");
  std::uint32_t generated_count = 0u;
  cuda_require(cudaMemcpy(&generated_count, device_generated_count.get(), sizeof(generated_count),
                          cudaMemcpyDeviceToHost), "read emitted byte extent");
  if (generated_count > output_bytes) throw std::runtime_error("invalid emitted byte extent");
  std::vector<std::uint8_t> output(generated_count);
  if (generated_count != 0u) {
    cuda_require(cudaMemcpy(output.data(), device_output.get(), generated_count,
                            cudaMemcpyDeviceToHost), "read emitted bytes");
  }
  return output;
}


#include "bcc32_cuda_adult_v1_reports.inl"

#endif  // !defined(BCC32_CUDA_ADULT_STATE_ONLY)

}  // namespace bcc32_cuda_adult_v1
