#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

#include "bcc32_cuda_resident_roles.cuh"

// A CUDA-resident warm-start affordance for ordering opaque unit matter.  A
// cue selects one consolidated subject-conditioned relation population, then
// its flat variable-length unit multiset is ordered only by mutable role,
// boundary, and closure evidence.  Unit bytes may be copied for inspection but
// are never interpreted.  The API has no word classes, authored frame, source
// location, episode retrieval, host score, or host-selected winner.
namespace substrate::bcc32::resident_role_compositor {

inline constexpr std::uint32_t kNoUnit = 0xffffffffu;
inline constexpr std::uint32_t kNoTrace = 0xffffffffu;
inline constexpr std::uint32_t kRoleCompositorBlockSize = 128u;

// unit_begin addresses a resident flattened relation population, not a corpus
// or source position.  The caller should materialize these variable populations
// from its existing conditioned-transition path; this sidecar consumes the
// caller's existing synthesis_roles and role evidence without deriving either.
// Repeated observations increase evidence instead of adding score-identical
// rows.
struct SubjectConditionedRelationTrace {
  std::uint32_t subject_unit = kNoUnit;
  std::uint32_t unit_begin = 0u;
  std::uint32_t unit_count = 0u;
  std::uint32_t evidence = 0u;
  std::uint32_t revision = 0u;
};

// This disclosed policy is ordinary mutable device state.  The constants are
// generic warm-start biases; all linguistic organization remains in the input
// evidence and can be shuffled, lesioned, or replaced without changing code.
struct MutableRoleCompositorPolicy {
  std::uint32_t enabled = 1u;
  std::uint32_t revision = 1u;
  std::uint32_t lesion_events = 0u;
  std::uint32_t min_plan_units = 2u;
  std::uint32_t max_plan_units = 32u;
  std::uint32_t min_relation_cue_hits = 1u;
  std::uint32_t subject_cue_weight = 16u;
  std::uint32_t relation_cue_weight = 24u;
  std::uint32_t relation_evidence_weight = 64u;
  std::uint32_t role_bigram_weight = 128u;
  std::uint32_t role_trigram_weight = 512u;
  std::uint32_t boundary_weight = 256u;
  std::uint32_t closure_weight = 256u;
  std::uint32_t activation_weight = 2u;
  std::uint32_t vitality_weight = 1u;
};

struct RoleCompositorModelView {
  // Opaque raw units learned at the reciprocal byte surface.
  std::uint32_t* unit_lengths = nullptr;
  std::uint32_t* unit_content = nullptr;
  std::uint32_t unit_words = 0u;
  std::uint32_t* unit_vitality = nullptr;
  std::uint32_t unit_count = 0u;

  // Mutable resident evidence.  No array index has executable semantics.
  unsigned long long* cue_activation = nullptr;
  std::uint32_t* cue_groups = nullptr;
  std::uint32_t* boundary_evidence = nullptr;
  std::uint32_t* closure_evidence = nullptr;
  resident_roles::MutableStructuralRole* unit_roles = nullptr;
  resident_roles::MutableRoleEvidenceTables role_evidence{};

  SubjectConditionedRelationTrace* relation_traces = nullptr;
  std::uint32_t relation_trace_count = 0u;
  std::uint32_t* relation_units = nullptr;
  std::uint32_t relation_unit_count = 0u;

  MutableRoleCompositorPolicy* policy = nullptr;
};

struct RoleCompositorChoice {
  std::uint32_t trace = kNoTrace;
  std::uint32_t cue_hits = 0u;
  std::uint32_t required_units = 0u;
  std::uint32_t tied_or_invalid = 0u;
  unsigned long long score = 0ull;
};

struct RoleCompositorWorkspaceView {
  unsigned long long* trace_scores = nullptr;
  std::uint32_t trace_capacity = 0u;
  std::uint32_t* required_counts = nullptr;
  std::uint32_t* selected_counts = nullptr;
  std::uint32_t unit_capacity = 0u;
  RoleCompositorChoice* choice = nullptr;

  // Composition is staged here so a failed plan never partially overwrites a
  // caller-owned motor_completion buffer.
  std::uint32_t* ordered_units = nullptr;
  std::uint32_t ordered_unit_capacity = 0u;

  std::uint32_t* output_units = nullptr;
  std::uint32_t output_unit_capacity = 0u;
  std::uint8_t* output_bytes = nullptr;
  std::uint32_t output_byte_capacity = 0u;
};

struct RoleCompositorResult {
  std::uint32_t ready = 0u;
  std::uint32_t selected_trace = kNoTrace;
  std::uint32_t selected_anchor = kNoUnit;
  std::uint32_t output_unit_count = 0u;
  std::uint32_t output_byte_count = 0u;
  std::uint32_t relation_cue_hits = 0u;
  std::uint32_t relation_evidence = 0u;
  std::uint32_t boundary_evidence = 0u;
  std::uint32_t closure_evidence = 0u;
  std::uint32_t policy_revision = 0u;
  std::uint32_t role_transition_count = 0u;
  std::uint32_t score_low = 0u;
  std::uint32_t score_high = 0u;
  std::uint32_t minimum_ordering_margin_low = 0u;
  std::uint32_t minimum_ordering_margin_high = 0u;
};

[[nodiscard]] __host__ __device__ inline unsigned long long compositor_sat_add(
    unsigned long long first, unsigned long long second) {
  const unsigned long long limit = ~0ull;
  return limit - first < second ? limit : first + second;
}

[[nodiscard]] __host__ __device__ inline unsigned long long compositor_sat_mul(
    unsigned long long first, unsigned long long second) {
  const unsigned long long limit = ~0ull;
  return first != 0ull && second > limit / first ? limit : first * second;
}

[[nodiscard]] __device__ inline unsigned long long compositor_role_count(
    const std::uint64_t* table, std::uint32_t index) {
  return table == nullptr ? 0ull : static_cast<unsigned long long>(table[index]);
}

[[nodiscard]] __device__ inline unsigned long long compositor_transition_score(
    const RoleCompositorModelView& model, const MutableRoleCompositorPolicy& policy,
    std::uint32_t first, std::uint32_t previous, std::uint32_t candidate,
    std::uint32_t position) {
  if (first >= model.unit_count || previous >= model.unit_count || candidate >= model.unit_count)
    return 0ull;
  const std::uint32_t first_role = model.unit_roles[first].role;
  const std::uint32_t previous_role = model.unit_roles[previous].role;
  const std::uint32_t candidate_role = model.unit_roles[candidate].role;
  if (first_role >= resident_roles::kStructuralRoleCount ||
      previous_role >= resident_roles::kStructuralRoleCount ||
      candidate_role >= resident_roles::kStructuralRoleCount) {
    return 0ull;
  }

  const resident_roles::RoleEvidenceTables evidence = resident_roles::as_const(model.role_evidence);
  const std::uint32_t pair_index =
      resident_roles::role_bigram_index(previous_role, candidate_role);
  const unsigned long long pair_count = compositor_sat_add(
      compositor_role_count(evidence.base_grammar.bigrams, pair_index),
      compositor_role_count(evidence.online_content.bigrams, pair_index));
  unsigned long long score = compositor_sat_mul(
      resident_roles::integer_log_depth(pair_count), policy.role_bigram_weight);
  if (position >= 2u) {
    const std::uint32_t triple_index =
        resident_roles::role_trigram_index(first_role, previous_role, candidate_role);
    const unsigned long long triple_count = compositor_sat_add(
        compositor_role_count(evidence.base_grammar.trigrams, triple_index),
        compositor_role_count(evidence.online_content.trigrams, triple_index));
    score = compositor_sat_add(
        score, compositor_sat_mul(resident_roles::integer_log_depth(triple_count),
                                  policy.role_trigram_weight));
  }
  return score;
}

static __global__ void score_relation_traces_kernel(RoleCompositorModelView model,
                                                     unsigned long long* trace_scores) {
  const std::uint32_t trace_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (trace_index >= model.relation_trace_count)
    return;
  trace_scores[trace_index] = 0ull;
  const MutableRoleCompositorPolicy policy = *model.policy;
  const SubjectConditionedRelationTrace trace = model.relation_traces[trace_index];
  if (policy.enabled == 0u || trace.evidence == 0u || trace.unit_count == 0u ||
      trace.subject_unit >= model.unit_count || trace.unit_begin > model.relation_unit_count ||
      trace.unit_count > model.relation_unit_count - trace.unit_begin) {
    return;
  }
  const unsigned long long subject_activation = model.cue_activation[trace.subject_unit];
  if (subject_activation == 0u)
    return;

  unsigned long long group_mass[32]{};
  std::uint32_t group_extent[32]{};
  std::uint32_t group_last_position[32];
  for (std::uint32_t group = 0u; group < 32u; ++group)
    group_last_position[group] = 0xffffffffu;
  std::uint32_t local_pair_extent = 0u;
  unsigned long long content_specificity = 0ull;
  for (std::uint32_t offset = 0u; offset < trace.unit_count; ++offset) {
    const std::uint32_t unit = model.relation_units[trace.unit_begin + offset];
    if (unit >= model.unit_count)
      return;
    const std::uint32_t group = model.cue_groups[unit];
    if (group < 32u && model.cue_activation[unit] > group_mass[group]) {
      group_mass[group] = model.cue_activation[unit];
    }
    if (group < 32u && model.cue_activation[unit] != 0ull) {
      const std::uint32_t extent = model.unit_lengths[unit];
      for (std::uint32_t other = 0u; other < 32u; ++other) {
        if (other == group || group_last_position[other] == 0xffffffffu) continue;
        const std::uint32_t cue_gap = group > other ? group - other : other - group;
        const std::uint32_t trace_gap = offset - group_last_position[other];
        if (cue_gap <= 12u && trace_gap <= 8u) {
          local_pair_extent = max(local_pair_extent,
                                  min(31u, extent + group_extent[other]));
        }
      }
      group_extent[group] = max(group_extent[group], extent);
      group_last_position[group] = offset;
    }
    const unsigned long long extent = model.unit_lengths[unit];
    if (extent >= 4ull) {
      const unsigned long long rarity =
          4096ull / max(1u, model.unit_vitality[unit]);
      content_specificity = compositor_sat_add(
          content_specificity,
          compositor_sat_mul(compositor_sat_mul(extent, extent),
                             rarity == 0ull ? 1ull : rarity));
    }
  }
  std::uint32_t cue_hits = 0u;
  unsigned long long cue_mass = 0ull;
  unsigned long long minimum_group_mass = ~0ull;
  unsigned long long adjacent_group_mass = 0ull;
  for (std::uint32_t group = 0u; group < 32u; ++group) {
    if (group_mass[group] == 0ull) continue;
    ++cue_hits;
    cue_mass = compositor_sat_add(cue_mass, group_mass[group]);
    minimum_group_mass = min(minimum_group_mass, group_mass[group]);
  }
  for (std::uint32_t left = 0u; left < 32u; ++left) {
    if (group_mass[left] == 0ull) continue;
    for (std::uint32_t right = left + 1u; right < 32u && right <= left + 12u;
         ++right) {
      if (group_mass[right] == 0ull) continue;
      const unsigned long long shared = min(group_mass[left], group_mass[right]);
      adjacent_group_mass = compositor_sat_add(
          adjacent_group_mass,
          compositor_sat_mul(shared, 13ull - (right - left)));
    }
  }
  if (cue_hits < policy.min_relation_cue_hits)
    return;

  unsigned long long score = compositor_sat_mul(
      static_cast<unsigned long long>(trace.evidence), policy.relation_evidence_weight);
  score = compositor_sat_add(
      score, compositor_sat_mul(subject_activation, policy.subject_cue_weight));
  score = compositor_sat_add(score, compositor_sat_mul(cue_mass, policy.relation_cue_weight));
  if (minimum_group_mass != ~0ull) {
    score = compositor_sat_add(
        score, compositor_sat_mul(minimum_group_mass,
                                  8ull * policy.relation_cue_weight));
  }
  score = compositor_sat_add(
      score, compositor_sat_mul(adjacent_group_mass,
                                4ull * policy.relation_cue_weight));
  score = compositor_sat_add(score, content_specificity);
  score = compositor_sat_mul(score, static_cast<unsigned long long>(cue_hits) + 1ull);
  constexpr unsigned long long kTopologyDetailMask = (1ull << 56u) - 1ull;
  score = (static_cast<unsigned long long>(local_pair_extent) << 59u) |
          (static_cast<unsigned long long>(min(7u, cue_hits)) << 56u) |
          (score & kTopologyDetailMask);
  trace_scores[trace_index] = score;
}

struct RelationTraceWinner {
  unsigned long long score = 0ull;
  std::uint32_t trace = kNoTrace;
  std::uint32_t tied = 0u;
};

static __global__ void select_relation_trace_kernel(
    const RoleCompositorModelView model, const unsigned long long* trace_scores,
    RoleCompositorChoice* choice) {
  const std::uint32_t lane = threadIdx.x;
  if (choice == nullptr || trace_scores == nullptr) {
    if (lane == 0u && choice != nullptr)
      *choice = {};
    return;
  }
  extern __shared__ RelationTraceWinner winners[];
  RelationTraceWinner local{};
  for (std::uint32_t trace = lane; trace < model.relation_trace_count;
       trace += blockDim.x) {
    const unsigned long long score = trace_scores[trace];
    if (score > local.score) {
      local.score = score;
      local.trace = trace;
      local.tied = 0u;
    } else if (score != 0ull && score == local.score) {
      local.tied = 1u;
    }
  }
  winners[lane] = local;
  __syncthreads();
  for (std::uint32_t stride = blockDim.x / 2u; stride != 0u; stride >>= 1u) {
    if (lane < stride) {
      const RelationTraceWinner other = winners[lane + stride];
      RelationTraceWinner& current = winners[lane];
      if (other.score > current.score) {
        current = other;
      } else if (other.score == current.score && other.score != 0ull) {
        current.tied = current.tied != 0u || other.tied != 0u ||
                               current.trace != kNoTrace && other.trace != kNoTrace
                           ? 1u
                           : 0u;
        if (current.trace == kNoTrace)
          current.trace = other.trace;
      }
    }
    __syncthreads();
  }
  if (lane != 0u)
    return;

  *choice = {};
  choice->trace = winners[0].trace;
  const unsigned long long best = winners[0].score;
  const bool tied = winners[0].tied != 0u;
  if (choice->trace == kNoTrace || tied) {
    choice->trace = kNoTrace;
    choice->tied_or_invalid = tied ? 1u : 0u;
    return;
  }

  const SubjectConditionedRelationTrace trace = model.relation_traces[choice->trace];
  choice->score = best;
  choice->required_units = trace.unit_count;
  for (std::uint32_t offset = 0u; offset < trace.unit_count; ++offset) {
    const std::uint32_t unit = model.relation_units[trace.unit_begin + offset];
    if (unit != trace.subject_unit && model.cue_activation[unit] != 0u)
      ++choice->cue_hits;
  }
}

static __global__ void mark_selected_relation_units_kernel(
    RoleCompositorModelView model, RoleCompositorChoice* choice,
    std::uint32_t* required_counts) {
  const std::uint32_t offset = blockIdx.x * blockDim.x + threadIdx.x;
  if (choice->trace == kNoTrace)
    return;
  const SubjectConditionedRelationTrace trace = model.relation_traces[choice->trace];
  if (offset >= trace.unit_count)
    return;
  const std::uint32_t unit = model.relation_units[trace.unit_begin + offset];
  if (unit >= model.unit_count || model.unit_lengths[unit] == 0u ||
      model.unit_lengths[unit] > model.unit_words * 4u) {
    atomicExch(&choice->tied_or_invalid, 1u);
    return;
  }
  atomicAdd(required_counts + unit, 1u);
}

static __global__ void compose_role_plan_kernel(
    RoleCompositorModelView model, RoleCompositorWorkspaceView workspace,
    RoleCompositorResult* result) {
  if (blockIdx.x != 0u || threadIdx.x != 0u)
    return;
  *result = {};
  result->selected_trace = kNoTrace;
  const MutableRoleCompositorPolicy policy = *model.policy;
  const RoleCompositorChoice choice = *workspace.choice;
  if (policy.enabled == 0u || choice.trace == kNoTrace || choice.tied_or_invalid != 0u ||
      choice.required_units < policy.min_plan_units ||
      policy.max_plan_units < policy.min_plan_units) {
    return;
  }
  const std::uint32_t plan_units = min(
      min(choice.required_units, policy.max_plan_units),
      workspace.output_unit_capacity);
  if (plan_units < policy.min_plan_units) return;

  unsigned long long total_score = choice.score;
  unsigned long long minimum_margin = ~0ull;
  std::uint32_t first = kNoUnit;
  std::uint32_t previous = kNoUnit;
  std::uint32_t first_boundary = 0u;
  std::uint32_t final_closure = 0u;
  std::uint32_t transition_count = 0u;

  for (std::uint32_t position = 0u; position < plan_units; ++position) {
    std::uint32_t winner = kNoUnit;
    unsigned long long best = 0ull;
    unsigned long long second = 0ull;
    bool tied = false;
    const bool final_position = position + 1u == plan_units;

    for (std::uint32_t unit = 0u; unit < model.unit_count; ++unit) {
      if (workspace.selected_counts[unit] >= workspace.required_counts[unit])
        continue;
      const bool closes = model.closure_evidence[unit] != 0u;
      if ((final_position && !closes) || (!final_position && closes))
        continue;
      if (position == 0u && model.boundary_evidence[unit] == 0u)
        continue;

      unsigned long long score = 0ull;
      if (position == 0u) {
        score = compositor_sat_mul(model.boundary_evidence[unit], policy.boundary_weight);
      } else {
        score = compositor_transition_score(model, policy, first, previous, unit, position);
        if (score == 0ull)
          continue;
      }
      if (closes) {
        score = compositor_sat_add(
            score, compositor_sat_mul(model.closure_evidence[unit], policy.closure_weight));
      }
      score = compositor_sat_add(
          score, compositor_sat_mul(model.cue_activation[unit], policy.activation_weight));
      score = compositor_sat_add(
          score, compositor_sat_mul(model.unit_vitality[unit], policy.vitality_weight));
      score = compositor_sat_add(
          score, compositor_sat_mul(workspace.required_counts[unit],
                                    policy.relation_evidence_weight));

      if (score > best) {
        second = best;
        best = score;
        winner = unit;
        tied = false;
      } else if (score != 0ull && score == best) {
        tied = true;
      } else if (score > second) {
        second = score;
      }
    }

    if (winner == kNoUnit || tied)
      return;
    if (position == 0u) {
      first = winner;
      first_boundary = model.boundary_evidence[winner];
    } else {
      ++transition_count;
      const unsigned long long margin = best - second;
      if (margin < minimum_margin)
        minimum_margin = margin;
    }
    previous = winner;
    final_closure = model.closure_evidence[winner];
    ++workspace.selected_counts[winner];
    workspace.ordered_units[position] = winner;
    total_score = compositor_sat_add(total_score, best);
  }

  if (first_boundary == 0u || final_closure == 0u)
    return;
  std::uint32_t byte_count = 0u;
  if (workspace.output_bytes != nullptr) {
    for (std::uint32_t position = 0u; position < plan_units; ++position) {
      const std::uint32_t unit = workspace.ordered_units[position];
      if (model.unit_lengths[unit] > workspace.output_byte_capacity - byte_count)
        return;
      byte_count += model.unit_lengths[unit];
    }
  }

  for (std::uint32_t position = 0u; position < plan_units; ++position)
    workspace.output_units[position] = workspace.ordered_units[position];
  if (workspace.output_bytes != nullptr) {
    byte_count = 0u;
    for (std::uint32_t position = 0u; position < plan_units; ++position) {
      const std::uint32_t unit = workspace.ordered_units[position];
      for (std::uint32_t offset = 0u; offset < model.unit_lengths[unit]; ++offset) {
        const std::uint32_t packed = model.unit_content[
            static_cast<std::size_t>(unit) * model.unit_words + offset / 4u];
        workspace.output_bytes[byte_count++] =
            static_cast<std::uint8_t>((packed >> (8u * (offset & 3u))) & 0xffu);
      }
    }
  }

  const SubjectConditionedRelationTrace trace = model.relation_traces[choice.trace];
  result->ready = 1u;
  result->selected_trace = choice.trace;
  result->output_unit_count = plan_units;
  result->output_byte_count = byte_count;
  result->relation_cue_hits = choice.cue_hits;
  result->relation_evidence = trace.evidence;
  result->boundary_evidence = first_boundary;
  result->closure_evidence = final_closure;
  result->policy_revision = policy.revision;
  result->role_transition_count = transition_count;
  result->score_low = static_cast<std::uint32_t>(total_score);
  result->score_high = static_cast<std::uint32_t>(total_score >> 32u);
  if (minimum_margin == ~0ull)
    minimum_margin = 0ull;
  result->minimum_ordering_margin_low = static_cast<std::uint32_t>(minimum_margin);
  result->minimum_ordering_margin_high = static_cast<std::uint32_t>(minimum_margin >> 32u);
}

[[nodiscard]] inline constexpr std::uint32_t role_compositor_blocks(std::uint32_t count) {
  return (count + kRoleCompositorBlockSize - 1u) / kRoleCompositorBlockSize;
}

inline cudaError_t compose_resident_role_plan_cuda(
    RoleCompositorModelView model, RoleCompositorWorkspaceView workspace,
    RoleCompositorResult* result, cudaStream_t stream = nullptr) {
  const bool has_bigram_evidence = model.role_evidence.base_grammar.bigrams != nullptr ||
                                   model.role_evidence.online_content.bigrams != nullptr;
  if (model.unit_lengths == nullptr || model.unit_content == nullptr || model.unit_words == 0u ||
      model.unit_vitality == nullptr || model.unit_count == 0u ||
      model.cue_activation == nullptr || model.cue_groups == nullptr ||
      model.boundary_evidence == nullptr ||
      model.closure_evidence == nullptr || model.unit_roles == nullptr || !has_bigram_evidence ||
      model.relation_traces == nullptr || model.relation_trace_count == 0u ||
      model.relation_units == nullptr || model.relation_unit_count == 0u ||
      model.policy == nullptr || workspace.trace_scores == nullptr ||
      workspace.trace_capacity < model.relation_trace_count ||
      workspace.required_counts == nullptr || workspace.selected_counts == nullptr ||
      workspace.unit_capacity < model.unit_count || workspace.choice == nullptr ||
      workspace.ordered_units == nullptr ||
      workspace.ordered_unit_capacity < workspace.output_unit_capacity ||
      workspace.output_units == nullptr || workspace.output_unit_capacity == 0u ||
      (workspace.output_bytes != nullptr && workspace.output_byte_capacity == 0u) ||
      result == nullptr) {
    return cudaErrorInvalidValue;
  }

  cudaError_t status = cudaMemsetAsync(workspace.trace_scores, 0,
                                       static_cast<std::size_t>(model.relation_trace_count) *
                                           sizeof(unsigned long long),
                                       stream);
  if (status != cudaSuccess)
    return status;
  status = cudaMemsetAsync(workspace.required_counts, 0,
                           static_cast<std::size_t>(model.unit_count) * sizeof(std::uint32_t),
                           stream);
  if (status != cudaSuccess)
    return status;
  status = cudaMemsetAsync(workspace.selected_counts, 0,
                           static_cast<std::size_t>(model.unit_count) * sizeof(std::uint32_t),
                           stream);
  if (status != cudaSuccess)
    return status;
  status = cudaMemsetAsync(workspace.ordered_units, 0xff,
                           static_cast<std::size_t>(workspace.ordered_unit_capacity) *
                               sizeof(std::uint32_t),
                           stream);
  if (status != cudaSuccess)
    return status;
  status = cudaMemsetAsync(result, 0, sizeof(RoleCompositorResult), stream);
  if (status != cudaSuccess)
    return status;

  score_relation_traces_kernel<<<role_compositor_blocks(model.relation_trace_count),
                                  kRoleCompositorBlockSize, 0u, stream>>>(model,
                                                                         workspace.trace_scores);
  status = cudaPeekAtLastError();
  if (status != cudaSuccess)
    return status;
  select_relation_trace_kernel<<<1u, kRoleCompositorBlockSize,
                                  kRoleCompositorBlockSize * sizeof(RelationTraceWinner), stream>>>(
      model, workspace.trace_scores, workspace.choice);
  status = cudaPeekAtLastError();
  if (status != cudaSuccess)
    return status;
  mark_selected_relation_units_kernel<<<role_compositor_blocks(model.relation_unit_count),
                                         kRoleCompositorBlockSize, 0u, stream>>>(
      model, workspace.choice, workspace.required_counts);
  status = cudaPeekAtLastError();
  if (status != cudaSuccess)
    return status;
  compose_role_plan_kernel<<<1u, 1u, 0u, stream>>>(model, workspace, result);
  return cudaPeekAtLastError();
}

// Direct view of adult_v1's existing ConditionedTransitionKey table.  Access is
// structural: anchor/previous/next are opaque unit IDs, never semantic slots.
struct DefaultConditionedTransitionAccess {
  template <typename TransitionKey>
  [[nodiscard]] __host__ __device__ static std::uint32_t anchor(
      const TransitionKey& key) {
    return static_cast<std::uint32_t>(key.anchor);
  }

  template <typename TransitionKey>
  [[nodiscard]] __host__ __device__ static std::uint32_t previous(
      const TransitionKey& key) {
    return static_cast<std::uint32_t>(key.previous);
  }

  template <typename TransitionKey>
  [[nodiscard]] __host__ __device__ static std::uint32_t next(
      const TransitionKey& key) {
    return static_cast<std::uint32_t>(key.next);
  }
};

template <typename TransitionKey>
struct ConditionedRelationTableView {
  TransitionKey* transitions = nullptr;
  std::uint32_t* counts = nullptr;
  std::uint32_t transition_count = 0u;
};

struct ConditionedRoleTrajectoryState {
  std::uint32_t anchor = kNoUnit;
  std::uint32_t first = kNoUnit;
  std::uint32_t previous = kNoUnit;
  std::uint32_t unit_count = 0u;
  std::uint32_t relation_edge_count = 0u;
  std::uint32_t ready = 0u;
  std::uint32_t failed = 0u;
  unsigned long long total_score = 0ull;
  unsigned long long minimum_margin = ~0ull;
};

struct ConditionedRoleTrajectoryWorkspaceView {
  unsigned long long* candidate_scores = nullptr;
  std::uint32_t* selected_counts = nullptr;
  std::uint32_t unit_capacity = 0u;
  ConditionedRoleTrajectoryState* state = nullptr;

  std::uint32_t* ordered_units = nullptr;
  std::uint32_t ordered_unit_capacity = 0u;
  std::uint32_t* output_units = nullptr;
  std::uint32_t output_unit_capacity = 0u;
  std::uint8_t* output_bytes = nullptr;
  std::uint32_t output_byte_capacity = 0u;
};

template <typename TransitionKey, typename Access>
static __global__ void score_conditioned_anchors_kernel(
    RoleCompositorModelView model, ConditionedRelationTableView<TransitionKey> table,
    unsigned long long* candidate_scores) {
  const std::uint32_t edge = blockIdx.x * blockDim.x + threadIdx.x;
  if (edge >= table.transition_count || table.counts[edge] == 0u ||
      model.policy->enabled == 0u) {
    return;
  }
  const TransitionKey key = table.transitions[edge];
  const std::uint32_t anchor = Access::anchor(key);
  const std::uint32_t previous = Access::previous(key);
  const std::uint32_t next = Access::next(key);
  if (anchor >= model.unit_count || previous >= model.unit_count || next >= model.unit_count ||
      model.cue_activation[anchor] == 0ull) {
    return;
  }
  unsigned long long overlap = 0ull;
  if (previous != anchor)
    overlap = compositor_sat_add(overlap, model.cue_activation[previous]);
  if (next != anchor && next != previous)
    overlap = compositor_sat_add(overlap, model.cue_activation[next]);
  const MutableRoleCompositorPolicy policy = *model.policy;
  const std::uint32_t span_specificity = min(model.unit_lengths[anchor], 32u);
  unsigned long long score = compositor_sat_mul(
      compositor_sat_mul(model.cue_activation[anchor], span_specificity),
      policy.subject_cue_weight);
  score = compositor_sat_add(
      score, compositor_sat_mul(overlap, policy.relation_cue_weight));
  score = compositor_sat_add(
      score, compositor_sat_mul(resident_roles::integer_log_depth(table.counts[edge]),
                                policy.relation_evidence_weight));
  atomicMax(candidate_scores + anchor, score);
}

static __global__ void select_conditioned_anchor_kernel(
    const unsigned long long* candidate_scores, std::uint32_t unit_count,
    ConditionedRoleTrajectoryState* state) {
  if (blockIdx.x != 0u || threadIdx.x != 0u)
    return;
  *state = {};
  state->anchor = kNoUnit;
  state->first = kNoUnit;
  state->previous = kNoUnit;
  state->minimum_margin = ~0ull;
  unsigned long long best = 0ull;
  bool tied = false;
  for (std::uint32_t unit = 0u; unit < unit_count; ++unit) {
    const unsigned long long score = candidate_scores[unit];
    if (score > best) {
      best = score;
      state->anchor = unit;
      tied = false;
    } else if (score != 0ull && score == best) {
      tied = true;
    }
  }
  if (state->anchor == kNoUnit || tied) {
    state->anchor = kNoUnit;
    state->failed = 1u;
    return;
  }
  state->total_score = best;
}

__device__ inline void score_conditioned_start_candidate(
    const RoleCompositorModelView& model, std::uint32_t unit,
    std::uint32_t relation_count, unsigned long long* candidate_scores) {
  if (unit >= model.unit_count || model.boundary_evidence[unit] == 0u ||
      model.unit_lengths[unit] == 0u || model.unit_lengths[unit] > model.unit_words * 4u) {
    return;
  }
  const MutableRoleCompositorPolicy policy = *model.policy;
  unsigned long long score = compositor_sat_mul(model.boundary_evidence[unit],
                                                  policy.boundary_weight);
  score = compositor_sat_add(
      score, compositor_sat_mul(model.cue_activation[unit], policy.activation_weight));
  score = compositor_sat_add(
      score, compositor_sat_mul(model.unit_vitality[unit], policy.vitality_weight));
  score = compositor_sat_add(
      score, compositor_sat_mul(resident_roles::integer_log_depth(relation_count),
                                policy.relation_evidence_weight));
  atomicMax(candidate_scores + unit, score);
}

template <typename TransitionKey, typename Access>
static __global__ void score_conditioned_starts_kernel(
    RoleCompositorModelView model, ConditionedRelationTableView<TransitionKey> table,
    const ConditionedRoleTrajectoryState* state, unsigned long long* candidate_scores) {
  const std::uint32_t edge = blockIdx.x * blockDim.x + threadIdx.x;
  if (edge >= table.transition_count || state->failed != 0u || table.counts[edge] == 0u)
    return;
  const TransitionKey key = table.transitions[edge];
  const std::uint32_t anchor = Access::anchor(key);
  if (anchor != state->anchor)
    return;
  score_conditioned_start_candidate(model, anchor, table.counts[edge], candidate_scores);
  score_conditioned_start_candidate(model, Access::previous(key), table.counts[edge],
                                    candidate_scores);
  score_conditioned_start_candidate(model, Access::next(key), table.counts[edge],
                                    candidate_scores);
}

static __global__ void select_conditioned_start_kernel(
    RoleCompositorModelView model, const unsigned long long* candidate_scores,
    ConditionedRoleTrajectoryWorkspaceView workspace) {
  if (blockIdx.x != 0u || threadIdx.x != 0u || workspace.state->failed != 0u)
    return;
  unsigned long long best = 0ull;
  std::uint32_t winner = kNoUnit;
  bool tied = false;
  for (std::uint32_t unit = 0u; unit < model.unit_count; ++unit) {
    const unsigned long long score = candidate_scores[unit];
    if (score > best) {
      best = score;
      winner = unit;
      tied = false;
    } else if (score != 0ull && score == best) {
      tied = true;
    }
  }
  if (winner == kNoUnit || tied) {
    workspace.state->failed = 1u;
    return;
  }
  workspace.state->first = winner;
  workspace.state->previous = winner;
  workspace.state->unit_count = 1u;
  workspace.state->total_score = compositor_sat_add(workspace.state->total_score, best);
  workspace.selected_counts[winner] = 1u;
  workspace.ordered_units[0] = winner;
  const MutableRoleCompositorPolicy policy = *model.policy;
  if (model.closure_evidence[winner] != 0u && policy.min_plan_units <= 1u)
    workspace.state->ready = 1u;
}

template <typename TransitionKey, typename Access>
static __global__ void score_conditioned_next_kernel(
    RoleCompositorModelView model, ConditionedRelationTableView<TransitionKey> table,
    ConditionedRoleTrajectoryWorkspaceView workspace, std::uint32_t position) {
  const std::uint32_t edge = blockIdx.x * blockDim.x + threadIdx.x;
  if (edge >= table.transition_count || workspace.state->failed != 0u ||
      workspace.state->ready != 0u || workspace.state->unit_count != position ||
      table.counts[edge] == 0u) {
    return;
  }
  const TransitionKey key = table.transitions[edge];
  const std::uint32_t anchor = Access::anchor(key);
  const std::uint32_t previous = Access::previous(key);
  const std::uint32_t next = Access::next(key);
  if (anchor != workspace.state->anchor || previous != workspace.state->previous ||
      next >= model.unit_count || workspace.selected_counts[next] != 0u ||
      model.unit_lengths[next] == 0u || model.unit_lengths[next] > model.unit_words * 4u) {
    return;
  }

  const MutableRoleCompositorPolicy policy = *model.policy;
  const bool closes = model.closure_evidence[next] != 0u;
  const std::uint32_t next_count = position + 1u;
  if ((closes && next_count < policy.min_plan_units) ||
      (!closes && next_count >= policy.max_plan_units)) {
    return;
  }
  const unsigned long long role_score = compositor_transition_score(
      model, policy, workspace.state->first, workspace.state->previous, next, position);
  if (role_score == 0ull)
    return;
  unsigned long long score = role_score;
  score = compositor_sat_add(
      score, compositor_sat_mul(resident_roles::integer_log_depth(table.counts[edge]),
                                policy.relation_evidence_weight));
  score = compositor_sat_add(
      score, compositor_sat_mul(model.cue_activation[next], policy.activation_weight));
  score = compositor_sat_add(
      score, compositor_sat_mul(model.unit_vitality[next], policy.vitality_weight));
  if (closes) {
    score = compositor_sat_add(
        score, compositor_sat_mul(model.closure_evidence[next], policy.closure_weight));
  }
  atomicMax(workspace.candidate_scores + next, score);
}

static __global__ void select_conditioned_next_kernel(
    RoleCompositorModelView model, ConditionedRoleTrajectoryWorkspaceView workspace,
    std::uint32_t position) {
  if (blockIdx.x != 0u || threadIdx.x != 0u || workspace.state->failed != 0u ||
      workspace.state->ready != 0u || workspace.state->unit_count != position) {
    return;
  }
  unsigned long long best = 0ull;
  unsigned long long second = 0ull;
  std::uint32_t winner = kNoUnit;
  bool tied = false;
  for (std::uint32_t unit = 0u; unit < model.unit_count; ++unit) {
    const unsigned long long score = workspace.candidate_scores[unit];
    if (score > best) {
      second = best;
      best = score;
      winner = unit;
      tied = false;
    } else if (score != 0ull && score == best) {
      tied = true;
    } else if (score > second) {
      second = score;
    }
  }
  if (winner == kNoUnit || tied) {
    workspace.state->failed = 1u;
    return;
  }
  workspace.ordered_units[position] = winner;
  workspace.selected_counts[winner] = 1u;
  workspace.state->previous = winner;
  workspace.state->unit_count = position + 1u;
  ++workspace.state->relation_edge_count;
  workspace.state->total_score = compositor_sat_add(workspace.state->total_score, best);
  const unsigned long long margin = best - second;
  if (margin < workspace.state->minimum_margin)
    workspace.state->minimum_margin = margin;
  if (model.closure_evidence[winner] != 0u)
    workspace.state->ready = 1u;
}

static __global__ void emit_conditioned_trajectory_kernel(
    RoleCompositorModelView model, ConditionedRoleTrajectoryWorkspaceView workspace,
    RoleCompositorResult* result) {
  if (blockIdx.x != 0u || threadIdx.x != 0u || workspace.state->ready == 0u ||
      workspace.state->failed != 0u || workspace.state->unit_count == 0u) {
    return;
  }
  const ConditionedRoleTrajectoryState state = *workspace.state;
  bool anchor_present = false;
  for (std::uint32_t position = 0u; position < state.unit_count; ++position) {
    anchor_present |= workspace.ordered_units[position] == state.anchor;
  }
  const std::uint32_t prepend_anchor = anchor_present ? 0u : 1u;
  const std::uint32_t output_count =
      min(workspace.output_unit_capacity, state.unit_count + prepend_anchor);
  std::uint32_t byte_count = 0u;
  std::uint32_t cue_hits = 0u;
  if (workspace.output_bytes != nullptr) {
    for (std::uint32_t position = 0u; position < output_count; ++position) {
      const std::uint32_t unit = prepend_anchor != 0u && position == 0u
                                     ? state.anchor
                                     : workspace.ordered_units[position - prepend_anchor];
      if (model.unit_lengths[unit] > workspace.output_byte_capacity - byte_count)
        return;
      byte_count += model.unit_lengths[unit];
    }
  }
  for (std::uint32_t position = 0u; position < output_count; ++position) {
    const std::uint32_t unit = prepend_anchor != 0u && position == 0u
                                   ? state.anchor
                                   : workspace.ordered_units[position - prepend_anchor];
    workspace.output_units[position] = unit;
    cue_hits += model.cue_activation[unit] != 0ull;
  }
  if (workspace.output_bytes != nullptr) {
    byte_count = 0u;
    for (std::uint32_t position = 0u; position < output_count; ++position) {
      const std::uint32_t unit = prepend_anchor != 0u && position == 0u
                                     ? state.anchor
                                     : workspace.ordered_units[position - prepend_anchor];
      for (std::uint32_t offset = 0u; offset < model.unit_lengths[unit]; ++offset) {
        const std::uint32_t packed = model.unit_content[
            static_cast<std::size_t>(unit) * model.unit_words + offset / 4u];
        workspace.output_bytes[byte_count++] =
            static_cast<std::uint8_t>((packed >> (8u * (offset & 3u))) & 0xffu);
      }
    }
  }

  result->ready = 1u;
  result->selected_trace = kNoTrace;
  result->selected_anchor = state.anchor;
  result->output_unit_count = output_count;
  result->output_byte_count = byte_count;
  result->relation_cue_hits = cue_hits;
  result->relation_evidence = state.relation_edge_count;
  result->boundary_evidence = model.boundary_evidence[state.first];
  result->closure_evidence = model.closure_evidence[state.previous];
  result->policy_revision = model.policy->revision;
  result->role_transition_count = state.unit_count - 1u;
  result->score_low = static_cast<std::uint32_t>(state.total_score);
  result->score_high = static_cast<std::uint32_t>(state.total_score >> 32u);
  const unsigned long long margin =
      state.minimum_margin == ~0ull ? 0ull : state.minimum_margin;
  result->minimum_ordering_margin_low = static_cast<std::uint32_t>(margin);
  result->minimum_ordering_margin_high = static_cast<std::uint32_t>(margin >> 32u);
}

template <typename TransitionKey, typename Access = DefaultConditionedTransitionAccess>
inline cudaError_t compose_conditioned_role_trajectory_cuda(
    RoleCompositorModelView model, ConditionedRelationTableView<TransitionKey> table,
    ConditionedRoleTrajectoryWorkspaceView workspace, RoleCompositorResult* result,
    cudaStream_t stream = nullptr) {
  const bool has_bigram_evidence = model.role_evidence.base_grammar.bigrams != nullptr ||
                                   model.role_evidence.online_content.bigrams != nullptr;
  if (model.unit_lengths == nullptr || model.unit_content == nullptr || model.unit_words == 0u ||
      model.unit_vitality == nullptr || model.unit_count == 0u ||
      model.cue_activation == nullptr || model.boundary_evidence == nullptr ||
      model.closure_evidence == nullptr || model.unit_roles == nullptr || !has_bigram_evidence ||
      model.policy == nullptr || table.transitions == nullptr || table.counts == nullptr ||
      table.transition_count == 0u || workspace.candidate_scores == nullptr ||
      workspace.selected_counts == nullptr || workspace.unit_capacity < model.unit_count ||
      workspace.state == nullptr || workspace.ordered_units == nullptr ||
      workspace.ordered_unit_capacity < workspace.output_unit_capacity ||
      workspace.output_units == nullptr || workspace.output_unit_capacity == 0u ||
      (workspace.output_bytes != nullptr && workspace.output_byte_capacity == 0u) ||
      result == nullptr) {
    return cudaErrorInvalidValue;
  }

  cudaError_t status = cudaMemsetAsync(
      workspace.candidate_scores, 0,
      static_cast<std::size_t>(model.unit_count) * sizeof(unsigned long long), stream);
  if (status != cudaSuccess)
    return status;
  status = cudaMemsetAsync(workspace.selected_counts, 0,
                           static_cast<std::size_t>(model.unit_count) * sizeof(std::uint32_t),
                           stream);
  if (status != cudaSuccess)
    return status;
  status = cudaMemsetAsync(workspace.ordered_units, 0xff,
                           static_cast<std::size_t>(workspace.ordered_unit_capacity) *
                               sizeof(std::uint32_t),
                           stream);
  if (status != cudaSuccess)
    return status;
  status = cudaMemsetAsync(result, 0, sizeof(RoleCompositorResult), stream);
  if (status != cudaSuccess)
    return status;

  const std::uint32_t edge_blocks = role_compositor_blocks(table.transition_count);
  score_conditioned_anchors_kernel<TransitionKey, Access><<<
      edge_blocks, kRoleCompositorBlockSize, 0u, stream>>>(model, table,
                                                           workspace.candidate_scores);
  status = cudaPeekAtLastError();
  if (status != cudaSuccess)
    return status;
  select_conditioned_anchor_kernel<<<1u, 1u, 0u, stream>>>(
      workspace.candidate_scores, model.unit_count, workspace.state);
  status = cudaPeekAtLastError();
  if (status != cudaSuccess)
    return status;
  status = cudaMemsetAsync(
      workspace.candidate_scores, 0,
      static_cast<std::size_t>(model.unit_count) * sizeof(unsigned long long), stream);
  if (status != cudaSuccess)
    return status;
  score_conditioned_starts_kernel<TransitionKey, Access><<<
      edge_blocks, kRoleCompositorBlockSize, 0u, stream>>>(model, table, workspace.state,
                                                           workspace.candidate_scores);
  status = cudaPeekAtLastError();
  if (status != cudaSuccess)
    return status;
  select_conditioned_start_kernel<<<1u, 1u, 0u, stream>>>(
      model, workspace.candidate_scores, workspace);
  status = cudaPeekAtLastError();
  if (status != cudaSuccess)
    return status;

  for (std::uint32_t position = 1u; position < workspace.output_unit_capacity; ++position) {
    status = cudaMemsetAsync(
        workspace.candidate_scores, 0,
        static_cast<std::size_t>(model.unit_count) * sizeof(unsigned long long), stream);
    if (status != cudaSuccess)
      return status;
    score_conditioned_next_kernel<TransitionKey, Access><<<
        edge_blocks, kRoleCompositorBlockSize, 0u, stream>>>(model, table, workspace, position);
    status = cudaPeekAtLastError();
    if (status != cudaSuccess)
      return status;
    select_conditioned_next_kernel<<<1u, 1u, 0u, stream>>>(model, workspace, position);
    status = cudaPeekAtLastError();
    if (status != cudaSuccess)
      return status;
  }
  emit_conditioned_trajectory_kernel<<<1u, 1u, 0u, stream>>>(model, workspace, result);
  return cudaPeekAtLastError();
}

static __global__ void lesion_relation_trace_kernel(SubjectConditionedRelationTrace* traces,
                                                     std::uint32_t trace_count,
                                                     std::uint32_t trace) {
  if (blockIdx.x == 0u && threadIdx.x == 0u && traces != nullptr && trace < trace_count) {
    traces[trace].evidence = 0u;
    ++traces[trace].revision;
  }
}

static __global__ void lesion_role_compositor_policy_kernel(MutableRoleCompositorPolicy* policy) {
  if (blockIdx.x == 0u && threadIdx.x == 0u && policy != nullptr) {
    policy->enabled = 0u;
    ++policy->lesion_events;
    ++policy->revision;
  }
}

static __global__ void lesion_conditioned_relation_edge_kernel(
    std::uint32_t* counts, std::uint32_t transition_count, std::uint32_t edge) {
  if (blockIdx.x == 0u && threadIdx.x == 0u && counts != nullptr && edge < transition_count)
    counts[edge] = 0u;
}

}  // namespace substrate::bcc32::resident_role_compositor
