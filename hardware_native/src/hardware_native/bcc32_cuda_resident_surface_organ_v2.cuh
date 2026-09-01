#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

#include "bcc32_cuda_resident_context_state.cuh"
#include "bcc32_cuda_resident_construction_composer.cuh"
#include "bcc32_cuda_resident_roles.cuh"

namespace substrate::bcc32::resident_surface_organ_v2 {

namespace context_state = resident_context_state;
namespace construction = resident_construction;

inline constexpr std::uint32_t kSurfaceOrganBlockSize = 128u;
inline constexpr std::uint32_t kSurfaceOrganMaxAnchors = 64u;
inline constexpr std::uint32_t kSurfaceOrganMaxExhaustiveAnchors = 8u;
inline constexpr std::uint32_t kSurfaceOrganMaxBridgeRoles = 2u;
inline constexpr std::uint32_t kSurfaceOrganMaxOutputUnits = 256u;
inline constexpr std::uint32_t kSurfaceOrganRoleCount = resident_roles::kStructuralRoleCount;
inline constexpr std::uint32_t kSurfaceOrganRoleBigramCount =
    kSurfaceOrganRoleCount * kSurfaceOrganRoleCount;
inline constexpr std::uint32_t kSurfaceOrganRoleTrigramCount =
    kSurfaceOrganRoleBigramCount * kSurfaceOrganRoleCount;
inline constexpr std::uint32_t kSurfaceOrganProbabilityOne = 1u << 20u;
inline constexpr std::uint64_t kSurfaceOrganDocumentMass = 1ull << 32u;
inline constexpr std::uint32_t kSurfaceOrganNoUnit = 0xffffffffu;

struct SurfaceOrganConfig {
  std::uint32_t max_bridge_roles = kSurfaceOrganMaxBridgeRoles;
  std::uint32_t min_link_probability_q20 = kSurfaceOrganProbabilityOne / 4u;
  std::uint32_t bridge_penalty_q20 = kSurfaceOrganProbabilityOne / 32u;
};

struct SurfaceUnitView {
  const std::uint32_t* lengths = nullptr;
  const std::uint32_t* packed_bytes = nullptr;
  const resident_roles::MutableStructuralRole* roles = nullptr;
  std::uint32_t unit_count = 0u;
  std::uint32_t unit_words = 0u;
};

// The batch is available only to learning. Realization receives aggregate
// evidence and therefore has no source sentence, source position, or source span.
struct SurfaceGrammarBatchView {
  const std::uint32_t* units = nullptr;
  const std::uint32_t* episode_offsets = nullptr;
  const std::uint32_t* document_episode_offsets = nullptr;
  std::uint32_t episode_count = 0u;
  std::uint32_t document_count = 0u;
};

// Optional sparse unit-level sequence evidence for production realization.
// Role transitions choose the shape of a clause; these learned n-grams bind
// each filler choice to the actual committed anchor(s).  Keeping this as a
// view over the adult's existing persistent n-grams avoids a second authored
// vocabulary or a dense unit-by-unit table.
template <typename BigramKey, typename TrigramKey>
struct SurfaceSequenceEvidenceView {
  const BigramKey* base_bigrams = nullptr;
  const std::uint32_t* base_bigram_counts = nullptr;
  std::uint32_t base_bigram_count = 0u;
  const TrigramKey* base_trigrams = nullptr;
  const std::uint32_t* base_trigram_counts = nullptr;
  std::uint32_t base_trigram_count = 0u;
  const BigramKey* online_bigrams = nullptr;
  const std::uint32_t* online_bigram_counts = nullptr;
  std::uint32_t online_bigram_count = 0u;
  const TrigramKey* online_trigrams = nullptr;
  const std::uint32_t* online_trigram_counts = nullptr;
  std::uint32_t online_trigram_count = 0u;
};

struct SurfaceClosureView {
  const std::uint32_t* learned_bytes = nullptr;
  std::uint32_t byte_count = 0u;
};

struct SurfaceGrammarStats {
  std::uint64_t learned_documents = 0u;
  std::uint64_t learned_episodes = 0u;
  std::uint64_t raw_unit_occurrences = 0u;
  std::uint64_t invalid_unit_occurrences = 0u;
};

struct MutableSurfaceGrammarEvidenceView {
  std::uint64_t* unit_mass = nullptr;
  std::uint64_t* unit_start_mass = nullptr;
  std::uint64_t* unit_end_mass = nullptr;
  std::uint32_t unit_capacity = 0u;

  std::uint64_t* role_mass = nullptr;
  std::uint64_t* role_start_mass = nullptr;
  std::uint64_t* role_end_mass = nullptr;
  std::uint64_t* role_bigram_mass = nullptr;
  std::uint64_t* role_bigram_context_mass = nullptr;
  std::uint64_t* role_trigram_mass = nullptr;
  std::uint64_t* role_trigram_context_mass = nullptr;
  SurfaceGrammarStats* stats = nullptr;
  context_state::DeviceView context_state_field{};
};

struct SurfaceLearningWorkspaceView {
  std::uint64_t* document_unit_events = nullptr;
  std::uint64_t* document_bigram_events = nullptr;
  std::uint64_t* document_trigram_events = nullptr;
  std::uint32_t document_capacity = 0u;
};

struct OpaqueContentPlanView {
  const std::uint32_t* anchor_units = nullptr;
  std::uint32_t anchor_count = 0u;
};

// A read-only view over the construction organ's learned joint records.  The
// surface organ may inspect only the opaque token/role pattern: literal tokens
// remain unit identities and slot tokens remain structural-role identities.
// It never interprets the bytes carried by a literal.
struct OpaqueConstructionWitnessView {
  const std::uint32_t* tokens = nullptr;
  const std::uint32_t* lengths = nullptr;
  const std::uint32_t* slot_counts = nullptr;
  const std::uint32_t* supports = nullptr;
  const std::uint32_t* count = nullptr;
  const resident_roles::MutableStructuralRole* roles = nullptr;
  std::uint32_t capacity = 0u;
  const std::uint32_t* required_construction = nullptr;
  const std::uint32_t* slot_units = nullptr;
  const std::uint32_t* slot_masses = nullptr;
  const std::uint32_t* slot_overflow = nullptr;
  // Only units learned as closed-class matter may be copied literally.
  const std::uint32_t* closed_class_mask = nullptr;
};

struct SurfaceRolePathChoice {
  std::uint32_t roles[kSurfaceOrganMaxBridgeRoles]{};
  std::uint32_t length = 0u;
  std::uint32_t quality_q20 = 0u;
  std::uint32_t valid = 0u;
};

struct SurfaceOrganResult {
  std::uint32_t ready = 0u;
  std::uint32_t grammar_supported = 0u;
  std::uint32_t closure_supported = 0u;
  std::uint32_t anchors_preserved = 0u;
  std::uint32_t output_unit_count = 0u;
  std::uint32_t output_byte_count = 0u;
  std::uint32_t connector_count = 0u;
  std::uint32_t plan_reordered = 0u;
  std::uint32_t capacity_exceeded = 0u;
  std::uint32_t selected_permutation = kSurfaceOrganNoUnit;
  std::uint32_t path_quality_q20 = 0u;
  std::uint32_t construction_count = 0u;
  std::uint32_t construction_supported = 0u;
  std::uint32_t construction_shape_matched = 0u;
  std::uint32_t construction_mapping_matched = 0u;
  std::uint32_t construction_tied = 0u;
};

struct SurfaceRealizationWorkspaceView {
  SurfaceRolePathChoice* role_bridges = nullptr;
  SurfaceRolePathChoice* prefixes = nullptr;
  SurfaceRolePathChoice* suffixes = nullptr;
  std::uint64_t* permutation_scores = nullptr;
  std::uint32_t* permutation_valid = nullptr;
  std::uint32_t permutation_capacity = 0u;

  std::uint32_t* output_units = nullptr;
  std::uint32_t* output_anchor_mask = nullptr;
  std::uint32_t output_unit_capacity = 0u;
  std::uint8_t* output_bytes = nullptr;
  std::uint32_t output_byte_capacity = 0u;
  SurfaceOrganResult* result = nullptr;
};

[[nodiscard]] __host__ __device__ constexpr std::size_t surface_unit_evidence_words(
    std::uint32_t unit_capacity) {
  return static_cast<std::size_t>(unit_capacity);
}

[[nodiscard]] __host__ __device__ constexpr std::size_t surface_role_bigram_words() {
  return kSurfaceOrganRoleBigramCount;
}

[[nodiscard]] __host__ __device__ constexpr std::size_t surface_role_trigram_words() {
  return kSurfaceOrganRoleTrigramCount;
}

[[nodiscard]] __host__ __device__ constexpr std::uint32_t surface_factorial(std::uint32_t count) {
  std::uint32_t result = 1u;
  for (std::uint32_t value = 2u; value <= count; ++value)
    result *= value;
  return result;
}

[[nodiscard]] inline constexpr std::uint32_t surface_organ_blocks(std::uint32_t count) {
  return (count + kSurfaceOrganBlockSize - 1u) / kSurfaceOrganBlockSize;
}

template <typename AdultState>
[[nodiscard]] inline SurfaceUnitView bind_bcc32_cuda_adult_v1_units(
    const AdultState& adult, const resident_roles::MutableStructuralRole* roles,
    std::uint32_t unit_words) {
  return {adult.unit_lengths.get(), adult.unit_content.get(), roles, adult.unit_count, unit_words};
}

template <typename AdultState>
[[nodiscard]] inline OpaqueContentPlanView bind_bcc32_cuda_adult_v1_motor_completion(
    const AdultState& adult, std::uint32_t completion_count) {
  return {adult.motor_completion.get(), completion_count};
}

[[nodiscard]] __device__ inline std::uint32_t surface_document_for_episode(
    const std::uint32_t* document_episode_offsets, std::uint32_t document_count,
    std::uint32_t episode) {
  std::uint32_t low = 0u;
  std::uint32_t high = document_count;
  while (low < high) {
    const std::uint32_t middle = low + (high - low) / 2u;
    if (document_episode_offsets[middle + 1u] <= episode)
      low = middle + 1u;
    else
      high = middle;
  }
  return low;
}

__device__ inline void surface_atomic_add(std::uint64_t* target, std::uint64_t value) {
  atomicAdd(reinterpret_cast<unsigned long long*>(target), static_cast<unsigned long long>(value));
}

[[nodiscard]] __host__ __device__ inline std::uint64_t surface_event_mass(
    std::uint64_t document_events) {
  if (document_events == 0u)
    return 0u;
  const std::uint64_t mass = kSurfaceOrganDocumentMass / document_events;
  return mass == 0u ? 1u : mass;
}

__global__ void count_surface_document_events_kernel(SurfaceGrammarBatchView batch,
                                                     SurfaceLearningWorkspaceView workspace) {
  const std::uint32_t episode = blockIdx.x * blockDim.x + threadIdx.x;
  if (episode >= batch.episode_count)
    return;
  const std::uint32_t document =
      surface_document_for_episode(batch.document_episode_offsets, batch.document_count, episode);
  if (document >= batch.document_count || document >= workspace.document_capacity)
    return;
  const std::uint32_t begin = batch.episode_offsets[episode];
  const std::uint32_t end = batch.episode_offsets[episode + 1u];
  if (end < begin)
    return;
  const std::uint64_t length = end - begin;
  surface_atomic_add(workspace.document_unit_events + document, length);
  surface_atomic_add(workspace.document_bigram_events + document, length > 1u ? length - 1u : 0u);
  surface_atomic_add(workspace.document_trigram_events + document, length > 2u ? length - 2u : 0u);
}

__global__ void accumulate_surface_grammar_kernel(SurfaceUnitView units,
                                                  SurfaceGrammarBatchView batch,
                                                  MutableSurfaceGrammarEvidenceView evidence,
                                                  SurfaceLearningWorkspaceView workspace) {
  const std::uint32_t episode = blockIdx.x * blockDim.x + threadIdx.x;
  if (episode >= batch.episode_count)
    return;
  const std::uint32_t document =
      surface_document_for_episode(batch.document_episode_offsets, batch.document_count, episode);
  if (document >= batch.document_count || document >= workspace.document_capacity)
    return;
  const std::uint32_t begin = batch.episode_offsets[episode];
  const std::uint32_t end = batch.episode_offsets[episode + 1u];
  if (end < begin)
    return;

  const std::uint64_t unit_weight = surface_event_mass(workspace.document_unit_events[document]);
  const std::uint64_t bigram_weight =
      surface_event_mass(workspace.document_bigram_events[document]);
  const std::uint64_t trigram_weight =
      surface_event_mass(workspace.document_trigram_events[document]);
  const std::uint32_t document_episodes =
      batch.document_episode_offsets[document + 1u] - batch.document_episode_offsets[document];
  const std::uint64_t boundary_weight = surface_event_mass(document_episodes);

  if (evidence.stats != nullptr)
    surface_atomic_add(&evidence.stats->raw_unit_occurrences, end - begin);
  for (std::uint32_t position = begin; position < end; ++position) {
    const std::uint32_t unit = batch.units[position];
    if (unit >= units.unit_count || unit >= evidence.unit_capacity) {
      if (evidence.stats != nullptr)
        surface_atomic_add(&evidence.stats->invalid_unit_occurrences, 1u);
      continue;
    }
    const std::uint32_t role = units.roles[unit].role;
    if (role >= kSurfaceOrganRoleCount) {
      if (evidence.stats != nullptr)
        surface_atomic_add(&evidence.stats->invalid_unit_occurrences, 1u);
      continue;
    }
    surface_atomic_add(evidence.unit_mass + unit, unit_weight);
    surface_atomic_add(evidence.role_mass + role, unit_weight);
    if (position == begin) {
      surface_atomic_add(evidence.unit_start_mass + unit, boundary_weight);
      surface_atomic_add(evidence.role_start_mass + role, boundary_weight);
    }
    if (position + 1u == end) {
      surface_atomic_add(evidence.unit_end_mass + unit, boundary_weight);
      surface_atomic_add(evidence.role_end_mass + role, boundary_weight);
    }
    if (position + 1u < end) {
      const std::uint32_t next = batch.units[position + 1u];
      if (next < units.unit_count && next < evidence.unit_capacity) {
        const std::uint32_t next_role = units.roles[next].role;
        if (next_role < kSurfaceOrganRoleCount) {
          const std::uint32_t index = role * kSurfaceOrganRoleCount + next_role;
          surface_atomic_add(evidence.role_bigram_mass + index, bigram_weight);
          surface_atomic_add(evidence.role_bigram_context_mass + role, bigram_weight);
        }
      }
    }
    if (position + 2u < end) {
      const std::uint32_t second = batch.units[position + 1u];
      const std::uint32_t next = batch.units[position + 2u];
      if (second < units.unit_count && next < units.unit_count && second < evidence.unit_capacity &&
          next < evidence.unit_capacity) {
        const std::uint32_t second_role = units.roles[second].role;
        const std::uint32_t next_role = units.roles[next].role;
        if (second_role < kSurfaceOrganRoleCount && next_role < kSurfaceOrganRoleCount) {
          const std::uint32_t context = role * kSurfaceOrganRoleCount + second_role;
          const std::uint32_t index = context * kSurfaceOrganRoleCount + next_role;
          surface_atomic_add(evidence.role_trigram_mass + index, trigram_weight);
          surface_atomic_add(evidence.role_trigram_context_mass + context, trigram_weight);
        }
      }
    }
  }
}

__global__ void finish_surface_grammar_batch_kernel(SurfaceGrammarBatchView batch,
                                                    SurfaceGrammarStats* stats) {
  if (blockIdx.x != 0u || threadIdx.x != 0u || stats == nullptr)
    return;
  stats->learned_documents += batch.document_count;
  stats->learned_episodes += batch.episode_count;
}

[[nodiscard]] __device__ inline bool surface_unit_has_closure(
    SurfaceUnitView units, std::uint32_t unit, SurfaceClosureView closure) {
  if (unit >= units.unit_count)
    return false;
  const std::uint32_t length = units.lengths[unit];
  for (std::uint32_t offset = 0u; offset < length; ++offset) {
    const std::uint32_t packed =
        units.packed_bytes[static_cast<std::size_t>(unit) * units.unit_words + offset / 4u];
    const std::uint32_t byte = (packed >> (8u * (offset & 3u))) & 0xffu;
    for (std::uint32_t index = 0u; index < closure.byte_count; ++index)
      if (closure.learned_bytes[index] <= 0xffu && byte == closure.learned_bytes[index])
        return true;
  }
  return false;
}

template <typename BigramKey, typename Access = resident_roles::DefaultNgramAccess>
__global__ void accumulate_surface_bigram_presence_kernel(
    SurfaceUnitView units, const BigramKey* bigrams, const std::uint32_t* counts,
    std::uint32_t bigram_count, SurfaceClosureView closure,
    MutableSurfaceGrammarEvidenceView evidence) {
  const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= bigram_count || counts[index] == 0u)
    return;
  const std::uint32_t previous = Access::bigram_previous(bigrams[index]);
  const std::uint32_t next = Access::bigram_next(bigrams[index]);
  if (previous >= units.unit_count || next >= units.unit_count)
    return;
  const std::uint32_t previous_role = units.roles[previous].role;
  const std::uint32_t next_role = units.roles[next].role;
  if (previous_role >= kSurfaceOrganRoleCount || next_role >= kSurfaceOrganRoleCount)
    return;
  constexpr std::uint64_t kEdgePresenceMass = 1ull << 20u;
  surface_atomic_add(evidence.unit_mass + previous, kEdgePresenceMass);
  surface_atomic_add(evidence.unit_mass + next, kEdgePresenceMass);
  surface_atomic_add(evidence.role_mass + previous_role, kEdgePresenceMass);
  surface_atomic_add(evidence.role_mass + next_role, kEdgePresenceMass);
  if (surface_unit_has_closure(units, previous, closure)) {
    surface_atomic_add(evidence.unit_end_mass + previous, kEdgePresenceMass);
    surface_atomic_add(evidence.role_end_mass + previous_role, kEdgePresenceMass);
    surface_atomic_add(evidence.unit_start_mass + next, kEdgePresenceMass);
    surface_atomic_add(evidence.role_start_mass + next_role, kEdgePresenceMass);
    return;
  }
  const std::uint32_t role_edge = previous_role * kSurfaceOrganRoleCount + next_role;
  surface_atomic_add(evidence.role_bigram_mass + role_edge, kEdgePresenceMass);
  surface_atomic_add(evidence.role_bigram_context_mass + previous_role, kEdgePresenceMass);
}

template <typename TrigramKey, typename Access = resident_roles::DefaultNgramAccess>
__global__ void accumulate_surface_trigram_presence_kernel(
    SurfaceUnitView units, const TrigramKey* trigrams, const std::uint32_t* counts,
    std::uint32_t trigram_count, SurfaceClosureView closure,
    MutableSurfaceGrammarEvidenceView evidence) {
  const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= trigram_count || counts[index] == 0u)
    return;
  const std::uint32_t first = Access::trigram_first(trigrams[index]);
  const std::uint32_t second = Access::trigram_second(trigrams[index]);
  const std::uint32_t next = Access::trigram_next(trigrams[index]);
  if (first >= units.unit_count || second >= units.unit_count || next >= units.unit_count ||
      surface_unit_has_closure(units, first, closure) ||
      surface_unit_has_closure(units, second, closure))
    return;
  const std::uint32_t first_role = units.roles[first].role;
  const std::uint32_t second_role = units.roles[second].role;
  const std::uint32_t next_role = units.roles[next].role;
  if (first_role >= kSurfaceOrganRoleCount || second_role >= kSurfaceOrganRoleCount ||
      next_role >= kSurfaceOrganRoleCount)
    return;
  constexpr std::uint64_t kEdgePresenceMass = 1ull << 20u;
  const std::uint32_t context = first_role * kSurfaceOrganRoleCount + second_role;
  surface_atomic_add(evidence.role_trigram_mass +
                         context * kSurfaceOrganRoleCount + next_role,
                     kEdgePresenceMass);
  surface_atomic_add(evidence.role_trigram_context_mass + context, kEdgePresenceMass);
}

__global__ void finish_surface_ngram_rebuild_kernel(SurfaceGrammarStats* stats,
                                                     std::uint32_t edge_count) {
  if (blockIdx.x == 0u && threadIdx.x == 0u) {
    stats->learned_documents = 1u;
    stats->learned_episodes = 1u;
    stats->raw_unit_occurrences = edge_count;
  }
}

[[nodiscard]] __host__ __device__ inline std::uint32_t surface_probability_q20(
    std::uint64_t mass, std::uint64_t context_mass) {
  if (mass == 0u || context_mass == 0u)
    return 0u;
  if (mass >= context_mass)
    return kSurfaceOrganProbabilityOne;
  std::uint64_t remainder = mass;
  std::uint32_t result = 0u;
  for (std::uint32_t bit = 0u; bit < 20u; ++bit) {
    result <<= 1u;
    if (remainder >= context_mass - remainder) {
      remainder -= context_mass - remainder;
      result |= 1u;
    } else {
      remainder += remainder;
    }
  }
  return result;
}

[[nodiscard]] __device__ inline std::uint32_t surface_pair_probability(
    MutableSurfaceGrammarEvidenceView evidence, std::uint32_t previous, std::uint32_t next) {
  return surface_probability_q20(
      evidence.role_bigram_mass[previous * kSurfaceOrganRoleCount + next],
      evidence.role_bigram_context_mass[previous]);
}

[[nodiscard]] __device__ inline std::uint32_t surface_triple_probability(
    MutableSurfaceGrammarEvidenceView evidence, std::uint32_t first, std::uint32_t second,
    std::uint32_t next) {
  const std::uint32_t context = first * kSurfaceOrganRoleCount + second;
  return surface_probability_q20(
      evidence.role_trigram_mass[context * kSurfaceOrganRoleCount + next],
      evidence.role_trigram_context_mass[context]);
}

[[nodiscard]] __host__ __device__ inline std::uint32_t surface_min(std::uint32_t first,
                                                                   std::uint32_t second) {
  return first < second ? first : second;
}

[[nodiscard]] __device__ inline bool surface_choice_better(const SurfaceRolePathChoice& candidate,
                                                           const SurfaceRolePathChoice& current) {
  if (candidate.valid != current.valid)
    return candidate.valid != 0u;
  if (candidate.valid == 0u)
    return false;
  if (candidate.quality_q20 != current.quality_q20)
    return candidate.quality_q20 > current.quality_q20;
  if (candidate.length != current.length)
    return candidate.length < current.length;
  if (candidate.roles[0] != current.roles[0])
    return candidate.roles[0] < current.roles[0];
  return candidate.roles[1] < current.roles[1];
}

[[nodiscard]] __device__ inline SurfaceRolePathChoice surface_candidate_choice(
    std::uint32_t first_role, std::uint32_t second_role, std::uint32_t length,
    std::uint32_t quality_q20) {
  SurfaceRolePathChoice result{};
  result.roles[0] = first_role;
  result.roles[1] = second_role;
  result.length = length;
  result.quality_q20 = quality_q20;
  result.valid = 1u;
  return result;
}

[[nodiscard]] __device__ inline std::uint32_t surface_penalized_quality(std::uint32_t quality,
                                                                        std::uint32_t length,
                                                                        SurfaceOrganConfig config) {
  const std::uint64_t penalty = static_cast<std::uint64_t>(length) * config.bridge_penalty_q20;
  return penalty >= quality ? 1u : quality - static_cast<std::uint32_t>(penalty);
}

__global__ void build_surface_role_bridges_kernel(MutableSurfaceGrammarEvidenceView evidence,
                                                  SurfaceOrganConfig config,
                                                  SurfaceRolePathChoice* bridges) {
  const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= kSurfaceOrganRoleBigramCount)
    return;
  const std::uint32_t first = index / kSurfaceOrganRoleCount;
  const std::uint32_t next = index % kSurfaceOrganRoleCount;
  SurfaceRolePathChoice best{};

  const std::uint32_t direct = surface_pair_probability(evidence, first, next);
  if (direct >= config.min_link_probability_q20)
    best = surface_candidate_choice(0u, 0u, 0u, direct);

  if (config.max_bridge_roles >= 1u) {
    for (std::uint32_t middle = 0u; middle < kSurfaceOrganRoleCount; ++middle) {
      const std::uint32_t left = surface_pair_probability(evidence, first, middle);
      const std::uint32_t right = surface_pair_probability(evidence, middle, next);
      const std::uint32_t triple = surface_triple_probability(evidence, first, middle, next);
      if (left < config.min_link_probability_q20 || right < config.min_link_probability_q20 ||
          triple < config.min_link_probability_q20)
        continue;
      const std::uint32_t quality =
          surface_penalized_quality(surface_min(left, right) + triple / 2u, 1u, config);
      const SurfaceRolePathChoice candidate = surface_candidate_choice(middle, 0u, 1u, quality);
      if (surface_choice_better(candidate, best))
        best = candidate;
    }
  }

  if (config.max_bridge_roles >= 2u) {
    for (std::uint32_t middle0 = 0u; middle0 < kSurfaceOrganRoleCount; ++middle0) {
      const std::uint32_t edge0 = surface_pair_probability(evidence, first, middle0);
      if (edge0 < config.min_link_probability_q20)
        continue;
      for (std::uint32_t middle1 = 0u; middle1 < kSurfaceOrganRoleCount; ++middle1) {
        const std::uint32_t edge1 = surface_pair_probability(evidence, middle0, middle1);
        const std::uint32_t edge2 = surface_pair_probability(evidence, middle1, next);
        const std::uint32_t triple0 = surface_triple_probability(evidence, first, middle0, middle1);
        const std::uint32_t triple1 = surface_triple_probability(evidence, middle0, middle1, next);
        if (edge1 < config.min_link_probability_q20 || edge2 < config.min_link_probability_q20 ||
            triple0 < config.min_link_probability_q20 || triple1 < config.min_link_probability_q20)
          continue;
        const std::uint32_t weakest = surface_min(edge0, surface_min(edge1, edge2));
        const std::uint32_t quality =
            surface_penalized_quality(weakest + triple0 / 4u + triple1 / 4u, 2u, config);
        const SurfaceRolePathChoice candidate =
            surface_candidate_choice(middle0, middle1, 2u, quality);
        if (surface_choice_better(candidate, best))
          best = candidate;
      }
    }
  }
  bridges[index] = best;
}

[[nodiscard]] __device__ inline std::uint64_t surface_role_mass_sum(const std::uint64_t* values) {
  std::uint64_t total = 0u;
  for (std::uint32_t role = 0u; role < kSurfaceOrganRoleCount; ++role)
    total += values[role];
  return total;
}

__global__ void build_surface_boundary_paths_kernel(MutableSurfaceGrammarEvidenceView evidence,
                                                    SurfaceOrganConfig config,
                                                    SurfaceRolePathChoice* prefixes,
                                                    SurfaceRolePathChoice* suffixes) {
  const std::uint32_t target = blockIdx.x * blockDim.x + threadIdx.x;
  if (target >= kSurfaceOrganRoleCount)
    return;
  const std::uint64_t start_total = surface_role_mass_sum(evidence.role_start_mass);
  const std::uint64_t end_total = surface_role_mass_sum(evidence.role_end_mass);
  SurfaceRolePathChoice prefix{};
  SurfaceRolePathChoice suffix{};

  const std::uint32_t direct_start =
      surface_probability_q20(evidence.role_start_mass[target], start_total);
  if (direct_start >= config.min_link_probability_q20)
    prefix = surface_candidate_choice(0u, 0u, 0u, direct_start);
  const std::uint32_t direct_end =
      surface_probability_q20(evidence.role_end_mass[target], end_total);
  if (direct_end >= config.min_link_probability_q20)
    suffix = surface_candidate_choice(0u, 0u, 0u, direct_end);

  if (config.max_bridge_roles >= 1u) {
    for (std::uint32_t role = 0u; role < kSurfaceOrganRoleCount; ++role) {
      const std::uint32_t start =
          surface_probability_q20(evidence.role_start_mass[role], start_total);
      const std::uint32_t into_target = surface_pair_probability(evidence, role, target);
      if (start >= config.min_link_probability_q20 &&
          into_target >= config.min_link_probability_q20) {
        const SurfaceRolePathChoice candidate = surface_candidate_choice(
            role, 0u, 1u, surface_penalized_quality(surface_min(start, into_target), 1u, config));
        if (surface_choice_better(candidate, prefix))
          prefix = candidate;
      }

      const std::uint32_t from_target = surface_pair_probability(evidence, target, role);
      const std::uint32_t end = surface_probability_q20(evidence.role_end_mass[role], end_total);
      if (from_target >= config.min_link_probability_q20 &&
          end >= config.min_link_probability_q20) {
        const SurfaceRolePathChoice candidate = surface_candidate_choice(
            role, 0u, 1u, surface_penalized_quality(surface_min(from_target, end), 1u, config));
        if (surface_choice_better(candidate, suffix))
          suffix = candidate;
      }
    }
  }

  if (config.max_bridge_roles >= 2u) {
    for (std::uint32_t role0 = 0u; role0 < kSurfaceOrganRoleCount; ++role0) {
      const std::uint32_t start =
          surface_probability_q20(evidence.role_start_mass[role0], start_total);
      const std::uint32_t from_target = surface_pair_probability(evidence, target, role0);
      if (start < config.min_link_probability_q20 && from_target < config.min_link_probability_q20)
        continue;
      for (std::uint32_t role1 = 0u; role1 < kSurfaceOrganRoleCount; ++role1) {
        const std::uint32_t middle = surface_pair_probability(evidence, role0, role1);
        const std::uint32_t into_target = surface_pair_probability(evidence, role1, target);
        const std::uint32_t prefix_triple =
            surface_triple_probability(evidence, role0, role1, target);
        if (start >= config.min_link_probability_q20 && middle >= config.min_link_probability_q20 &&
            into_target >= config.min_link_probability_q20 &&
            prefix_triple >= config.min_link_probability_q20) {
          const std::uint32_t quality = surface_penalized_quality(
              surface_min(start, surface_min(middle, into_target)) + prefix_triple / 2u, 2u,
              config);
          const SurfaceRolePathChoice candidate =
              surface_candidate_choice(role0, role1, 2u, quality);
          if (surface_choice_better(candidate, prefix))
            prefix = candidate;
        }

        const std::uint32_t suffix_middle = surface_pair_probability(evidence, role0, role1);
        const std::uint32_t end = surface_probability_q20(evidence.role_end_mass[role1], end_total);
        const std::uint32_t suffix_triple =
            surface_triple_probability(evidence, target, role0, role1);
        if (from_target >= config.min_link_probability_q20 &&
            suffix_middle >= config.min_link_probability_q20 &&
            end >= config.min_link_probability_q20 &&
            suffix_triple >= config.min_link_probability_q20) {
          const std::uint32_t quality = surface_penalized_quality(
              surface_min(from_target, surface_min(suffix_middle, end)) + suffix_triple / 2u, 2u,
              config);
          const SurfaceRolePathChoice candidate =
              surface_candidate_choice(role0, role1, 2u, quality);
          if (surface_choice_better(candidate, suffix))
            suffix = candidate;
        }
      }
    }
  }
  prefixes[target] = prefix;
  suffixes[target] = suffix;
}

__device__ inline void surface_decode_permutation(std::uint32_t rank, std::uint32_t count,
                                                  std::uint32_t* permutation) {
  std::uint32_t available[kSurfaceOrganMaxExhaustiveAnchors]{};
  for (std::uint32_t index = 0u; index < count; ++index)
    available[index] = index;
  for (std::uint32_t position = 0u; position < count; ++position) {
    const std::uint32_t block = surface_factorial(count - position - 1u);
    const std::uint32_t selected = block == 0u ? 0u : rank / block;
    rank = block == 0u ? 0u : rank % block;
    permutation[position] = available[selected];
    for (std::uint32_t move = selected; move + 1u < count - position; ++move)
      available[move] = available[move + 1u];
  }
}

__global__ void score_surface_anchor_permutations_kernel(
    SurfaceUnitView units, MutableSurfaceGrammarEvidenceView evidence, OpaqueContentPlanView plan,
    const SurfaceRolePathChoice* bridges, const SurfaceRolePathChoice* prefixes,
    const SurfaceRolePathChoice* suffixes, std::uint64_t* scores, std::uint32_t* valid) {
  const std::uint32_t permutation_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (plan.anchor_count > kSurfaceOrganMaxExhaustiveAnchors)
    return;
  const std::uint32_t permutation_count = surface_factorial(plan.anchor_count);
  if (permutation_index >= permutation_count)
    return;
  scores[permutation_index] = 0u;
  valid[permutation_index] = 0u;
  std::uint32_t permutation[kSurfaceOrganMaxExhaustiveAnchors]{};
  surface_decode_permutation(permutation_index, plan.anchor_count, permutation);

  std::uint32_t roles[kSurfaceOrganMaxExhaustiveAnchors]{};
  for (std::uint32_t position = 0u; position < plan.anchor_count; ++position) {
    const std::uint32_t unit = plan.anchor_units[permutation[position]];
    if (unit >= units.unit_count || unit >= evidence.unit_capacity || units.lengths[unit] == 0u)
      return;
    const resident_roles::MutableStructuralRole role = units.roles[unit];
    if (role.role >= kSurfaceOrganRoleCount || role.confidence == 0u)
      return;
    roles[position] = role.role;
  }

  const SurfaceRolePathChoice prefix = prefixes[roles[0]];
  const SurfaceRolePathChoice suffix = suffixes[roles[plan.anchor_count - 1u]];
  if (prefix.valid == 0u || suffix.valid == 0u)
    return;
  std::uint64_t score = prefix.quality_q20 + suffix.quality_q20;
  for (std::uint32_t position = 1u; position < plan.anchor_count; ++position) {
    const SurfaceRolePathChoice bridge =
        bridges[roles[position - 1u] * kSurfaceOrganRoleCount + roles[position]];
    if (bridge.valid == 0u)
      return;
    score += bridge.quality_q20;
  }
  for (std::uint32_t position = 0u; position < plan.anchor_count; ++position)
    score += units.roles[plan.anchor_units[permutation[position]]].confidence;
  scores[permutation_index] = score;
  valid[permutation_index] = 1u;
}

[[nodiscard]] __device__ inline bool surface_unit_is_anchor(std::uint32_t unit,
                                                            OpaqueContentPlanView plan) {
  for (std::uint32_t index = 0u; index < plan.anchor_count; ++index) {
    if (plan.anchor_units[index] == unit)
      return true;
  }
  return false;
}

[[nodiscard]] __device__ inline bool surface_unit_already_emitted(std::uint32_t unit,
                                                                  const std::uint32_t* output_units,
                                                                  std::uint32_t output_count) {
  for (std::uint32_t index = 0u; index < output_count; ++index) {
    if (output_units[index] == unit)
      return true;
  }
  return false;
}

enum class SurfaceUnitCompetition : std::uint32_t { kStart, kInterior, kEnd };

[[nodiscard]] __device__ inline std::uint32_t surface_select_unit_for_role(
    SurfaceUnitView units, MutableSurfaceGrammarEvidenceView evidence, OpaqueContentPlanView plan,
    std::uint32_t role, SurfaceUnitCompetition competition, const std::uint32_t* output_units,
    std::uint32_t output_count, std::uint32_t right_boundary_unit) {
  std::uint32_t winner = kSurfaceOrganNoUnit;
  std::uint32_t winner_context_edges = 0u;
  std::uint64_t winner_context_mass = 0u;
  std::uint64_t winner_score = 0u;
  const std::uint64_t role_context =
      competition == SurfaceUnitCompetition::kStart
          ? evidence.role_start_mass[role]
          : (competition == SurfaceUnitCompetition::kEnd ? evidence.role_end_mass[role]
                                                         : evidence.role_mass[role]);
  if (role_context == 0u)
    return winner;
  const std::uint32_t left_boundary_unit =
      output_count == 0u ? kSurfaceOrganNoUnit : output_units[output_count - 1u];
  const bool context_available =
      evidence.context_state_field.unit_bindings != nullptr &&
      evidence.context_state_field.transitions != nullptr &&
      evidence.context_state_field.scalars != nullptr;
  const context_state::StateRef left_state =
      context_available && left_boundary_unit < units.unit_count
          ? context_state::state_for_unit(evidence.context_state_field, left_boundary_unit)
          : context_state::StateRef{};
  const context_state::StateRef right_state =
      context_available && right_boundary_unit < units.unit_count
          ? context_state::state_for_unit(evidence.context_state_field, right_boundary_unit)
          : context_state::StateRef{};
  for (std::uint32_t unit = 0u; unit < units.unit_count; ++unit) {
    if (unit >= evidence.unit_capacity || units.lengths[unit] == 0u ||
        units.roles[unit].role != role || units.roles[unit].confidence == 0u ||
        surface_unit_is_anchor(unit, plan) ||
        surface_unit_already_emitted(unit, output_units, output_count))
      continue;
    const std::uint64_t mass =
        competition == SurfaceUnitCompetition::kStart
            ? evidence.unit_start_mass[unit]
            : (competition == SurfaceUnitCompetition::kEnd ? evidence.unit_end_mass[unit]
                                                           : evidence.unit_mass[unit]);
    const std::uint64_t score =
        static_cast<std::uint64_t>(surface_probability_q20(mass, role_context)) * 256u +
        units.roles[unit].confidence;
    std::uint32_t context_edges = 0u;
    std::uint64_t context_mass = 0u;
    if (context_available) {
      const context_state::StateRef candidate_state =
          context_state::state_for_unit(evidence.context_state_field, unit);
      if (candidate_state.ready != 0u) {
        if (left_state.ready != 0u) {
          const std::uint64_t support = context_state::transition_support(
              evidence.context_state_field, left_state.state_id, candidate_state.state_id);
          context_edges += support != 0u ? 1u : 0u;
          context_mass += support;
        }
        if (right_state.ready != 0u) {
          const std::uint64_t support = context_state::transition_support(
              evidence.context_state_field, candidate_state.state_id, right_state.state_id);
          context_edges += support != 0u ? 1u : 0u;
          context_mass += support;
        }
      }
    }
    const bool better =
        winner == kSurfaceOrganNoUnit || context_edges > winner_context_edges ||
        (context_edges == winner_context_edges && context_mass > winner_context_mass) ||
        (context_edges == winner_context_edges && context_mass == winner_context_mass &&
         score > winner_score) ||
        (context_edges == winner_context_edges && context_mass == winner_context_mass &&
         score == winner_score && unit < winner);
    if (mass != 0u && better) {
      winner = unit;
      winner_context_edges = context_edges;
      winner_context_mass = context_mass;
      winner_score = score;
    }
  }
  return winner;
}

template <typename BigramKey, typename Access>
[[nodiscard]] __device__ inline bool surface_bigram_less(
    const BigramKey& key, std::uint32_t previous, std::uint32_t next) {
  const std::uint32_t key_previous =
      Access::template bigram_previous<BigramKey>(key);
  const std::uint32_t key_next =
      Access::template bigram_next<BigramKey>(key);
  return key_previous < previous ||
         (key_previous == previous && key_next < next);
}

template <typename BigramKey, typename Access>
[[nodiscard]] __device__ inline std::uint64_t surface_exact_bigram_mass(
    const BigramKey* keys, const std::uint32_t* counts, std::uint32_t count,
    std::uint32_t previous, std::uint32_t next) {
  if (keys == nullptr || counts == nullptr || count == 0u)
    return 0u;
  std::uint32_t lower = 0u;
  std::uint32_t upper = count;
  while (lower < upper) {
    const std::uint32_t middle = lower + (upper - lower) / 2u;
    if (surface_bigram_less<BigramKey, Access>(keys[middle], previous,
                                               next))
      lower = middle + 1u;
    else
      upper = middle;
  }
  if (lower >= count ||
      Access::template bigram_previous<BigramKey>(keys[lower]) != previous ||
      Access::template bigram_next<BigramKey>(keys[lower]) != next)
    return 0u;
  return counts[lower];
}

template <typename TrigramKey, typename Access>
[[nodiscard]] __device__ inline bool surface_trigram_less(
    const TrigramKey& key, std::uint32_t first, std::uint32_t second,
    std::uint32_t next) {
  const std::uint32_t key_first =
      Access::template trigram_first<TrigramKey>(key);
  const std::uint32_t key_second =
      Access::template trigram_second<TrigramKey>(key);
  const std::uint32_t key_next =
      Access::template trigram_next<TrigramKey>(key);
  return key_first < first ||
         (key_first == first &&
          (key_second < second ||
           (key_second == second && key_next < next)));
}

template <typename TrigramKey, typename Access>
[[nodiscard]] __device__ inline std::uint64_t surface_exact_trigram_mass(
    const TrigramKey* keys, const std::uint32_t* counts,
    std::uint32_t count, std::uint32_t first, std::uint32_t second,
    std::uint32_t next) {
  if (keys == nullptr || counts == nullptr || count == 0u)
    return 0u;
  std::uint32_t lower = 0u;
  std::uint32_t upper = count;
  while (lower < upper) {
    const std::uint32_t middle = lower + (upper - lower) / 2u;
    if (surface_trigram_less<TrigramKey, Access>(keys[middle], first,
                                                 second, next))
      lower = middle + 1u;
    else
      upper = middle;
  }
  if (lower >= count ||
      Access::template trigram_first<TrigramKey>(keys[lower]) != first ||
      Access::template trigram_second<TrigramKey>(keys[lower]) != second ||
      Access::template trigram_next<TrigramKey>(keys[lower]) != next)
    return 0u;
  return counts[lower];
}

template <typename BigramKey, typename TrigramKey, typename Access>
[[nodiscard]] __device__ inline std::uint64_t surface_sequence_bigram_mass(
    SurfaceSequenceEvidenceView<BigramKey, TrigramKey> sequence,
    std::uint32_t previous, std::uint32_t next) {
  return surface_exact_bigram_mass<BigramKey, Access>(
             sequence.base_bigrams, sequence.base_bigram_counts,
             sequence.base_bigram_count, previous, next) +
         surface_exact_bigram_mass<BigramKey, Access>(
             sequence.online_bigrams, sequence.online_bigram_counts,
             sequence.online_bigram_count, previous, next);
}

template <typename BigramKey, typename TrigramKey, typename Access>
[[nodiscard]] __device__ inline std::uint64_t surface_sequence_trigram_mass(
    SurfaceSequenceEvidenceView<BigramKey, TrigramKey> sequence,
    std::uint32_t first, std::uint32_t second, std::uint32_t next) {
  return surface_exact_trigram_mass<TrigramKey, Access>(
             sequence.base_trigrams, sequence.base_trigram_counts,
             sequence.base_trigram_count, first, second, next) +
         surface_exact_trigram_mass<TrigramKey, Access>(
             sequence.online_trigrams, sequence.online_trigram_counts,
             sequence.online_trigram_count, first, second, next);
}

[[nodiscard]] __device__ inline bool surface_contextual_unit_eligible(
    SurfaceUnitView units, OpaqueContentPlanView plan, std::uint32_t unit,
    std::uint32_t role, const std::uint32_t* output_units,
    std::uint32_t output_count) {
  return unit < units.unit_count && units.lengths[unit] != 0u &&
         units.roles[unit].role == role && units.roles[unit].confidence != 0u &&
         !surface_unit_is_anchor(unit, plan) &&
         !surface_unit_already_emitted(unit, output_units, output_count);
}

[[nodiscard]] __device__ inline std::uint64_t surface_context_transition_mass(
    MutableSurfaceGrammarEvidenceView evidence, std::uint32_t from_unit,
    std::uint32_t to_unit) {
  if (evidence.context_state_field.unit_bindings == nullptr ||
      evidence.context_state_field.transitions == nullptr ||
      evidence.context_state_field.scalars == nullptr ||
      from_unit == kSurfaceOrganNoUnit || to_unit == kSurfaceOrganNoUnit)
    return 0u;
  const context_state::StateRef from =
      context_state::state_for_unit(evidence.context_state_field, from_unit);
  const context_state::StateRef to =
      context_state::state_for_unit(evidence.context_state_field, to_unit);
  if (from.ready == 0u || to.ready == 0u)
    return 0u;
  return context_state::transition_support(evidence.context_state_field, from.state_id,
                                           to.state_id);
}

// Select a complete one- or two-role filler path against the local committed
// anchors.  Every selected edge must have resident unit-level sequence mass;
// when no such path exists realization abstains to anchor-only fallback rather
// than splicing globally frequent role fillers into unrelated content.
template <typename BigramKey, typename TrigramKey,
          typename Access = resident_roles::DefaultNgramAccess>
[[nodiscard]] __device__ inline bool surface_select_contextual_role_path(
    SurfaceUnitView units, MutableSurfaceGrammarEvidenceView evidence,
    OpaqueContentPlanView plan,
    SurfaceSequenceEvidenceView<BigramKey, TrigramKey> sequence,
    SurfaceRolePathChoice path, std::uint32_t left_anchor,
    std::uint32_t right_anchor, const std::uint32_t* output_units,
    std::uint32_t output_count, std::uint32_t* selected) {
  if (path.length == 0u)
    return true;
  if (path.length > kSurfaceOrganMaxBridgeRoles || selected == nullptr ||
      (left_anchor == kSurfaceOrganNoUnit &&
       right_anchor == kSurfaceOrganNoUnit))
    return false;

  std::uint64_t winner_context = 0u;
  std::uint64_t winner_state_mass = 0u;
  std::uint32_t winner_state_edges = 0u;
  std::uint64_t winner_frequency = 0u;
  std::uint32_t winner0 = kSurfaceOrganNoUnit;
  std::uint32_t winner1 = kSurfaceOrganNoUnit;
  if (path.length == 1u) {
    for (std::uint32_t unit = 0u; unit < units.unit_count; ++unit) {
      if (!surface_contextual_unit_eligible(
              units, plan, unit, path.roles[0], output_units, output_count))
        continue;
      const std::uint64_t left =
          left_anchor == kSurfaceOrganNoUnit
              ? 0u
              : surface_sequence_bigram_mass<BigramKey, TrigramKey, Access>(
                    sequence, left_anchor, unit);
      const std::uint64_t right =
          right_anchor == kSurfaceOrganNoUnit
              ? 0u
              : surface_sequence_bigram_mass<BigramKey, TrigramKey, Access>(
                    sequence, unit, right_anchor);
      const std::uint64_t left_state =
          surface_context_transition_mass(evidence, left_anchor, unit);
      const std::uint64_t right_state =
          surface_context_transition_mass(evidence, unit, right_anchor);
      if ((left_anchor != kSurfaceOrganNoUnit && left == 0u && left_state == 0u) ||
          (right_anchor != kSurfaceOrganNoUnit && right == 0u && right_state == 0u))
        continue;
      const std::uint64_t triple =
          left_anchor != kSurfaceOrganNoUnit &&
                  right_anchor != kSurfaceOrganNoUnit
              ? surface_sequence_trigram_mass<BigramKey, TrigramKey, Access>(
                    sequence, left_anchor, unit, right_anchor)
              : 0u;
      const std::uint64_t context = left + right + 4u * triple;
      const std::uint32_t state_edges =
          (left_state != 0u ? 1u : 0u) + (right_state != 0u ? 1u : 0u);
      const std::uint64_t state_mass = left_state + right_state;
      const std::uint64_t frequency = evidence.unit_mass[unit];
      if (winner0 == kSurfaceOrganNoUnit || state_edges > winner_state_edges ||
          (state_edges == winner_state_edges && state_mass > winner_state_mass) ||
          (state_edges == winner_state_edges && state_mass == winner_state_mass &&
           context > winner_context) ||
          (state_edges == winner_state_edges && state_mass == winner_state_mass &&
           context == winner_context && frequency > winner_frequency) ||
          (state_edges == winner_state_edges && state_mass == winner_state_mass &&
           context == winner_context && frequency == winner_frequency && unit < winner0)) {
        winner0 = unit;
        winner_state_edges = state_edges;
        winner_state_mass = state_mass;
        winner_context = context;
        winner_frequency = frequency;
      }
    }
  } else {
    for (std::uint32_t source = 0u; source < 2u; ++source) {
      const BigramKey* keys =
          source == 0u ? sequence.base_bigrams : sequence.online_bigrams;
      const std::uint32_t count = source == 0u
                                      ? sequence.base_bigram_count
                                      : sequence.online_bigram_count;
      if (keys == nullptr)
        continue;
      for (std::uint32_t edge = 0u; edge < count; ++edge) {
        const std::uint32_t first =
            Access::template bigram_previous<BigramKey>(keys[edge]);
        const std::uint32_t second =
            Access::template bigram_next<BigramKey>(keys[edge]);
        if (first == second ||
            !surface_contextual_unit_eligible(
                units, plan, first, path.roles[0], output_units,
                output_count) ||
            !surface_contextual_unit_eligible(
                units, plan, second, path.roles[1], output_units,
                output_count))
          continue;
        const std::uint64_t middle =
            surface_sequence_bigram_mass<BigramKey, TrigramKey, Access>(
                sequence, first, second);
        const std::uint64_t left =
            left_anchor == kSurfaceOrganNoUnit
                ? 0u
                : surface_sequence_bigram_mass<BigramKey, TrigramKey, Access>(
                      sequence, left_anchor, first);
        const std::uint64_t right =
            right_anchor == kSurfaceOrganNoUnit
                ? 0u
                : surface_sequence_bigram_mass<BigramKey, TrigramKey, Access>(
                      sequence, second, right_anchor);
        const std::uint64_t left_state =
            surface_context_transition_mass(evidence, left_anchor, first);
        const std::uint64_t middle_state =
            surface_context_transition_mass(evidence, first, second);
        const std::uint64_t right_state =
            surface_context_transition_mass(evidence, second, right_anchor);
        if (middle == 0u ||
            (left_anchor != kSurfaceOrganNoUnit && left == 0u && left_state == 0u) ||
            (right_anchor != kSurfaceOrganNoUnit && right == 0u && right_state == 0u))
          continue;
        const std::uint64_t left_triple =
            left_anchor == kSurfaceOrganNoUnit
                ? 0u
                : surface_sequence_trigram_mass<BigramKey, TrigramKey, Access>(
                      sequence, left_anchor, first, second);
        const std::uint64_t right_triple =
            right_anchor == kSurfaceOrganNoUnit
                ? 0u
                : surface_sequence_trigram_mass<BigramKey, TrigramKey, Access>(
                      sequence, first, second, right_anchor);
        const std::uint64_t context =
            middle + left + right + 4u * (left_triple + right_triple);
        const std::uint32_t state_edges =
            (left_state != 0u ? 1u : 0u) + (middle_state != 0u ? 1u : 0u) +
            (right_state != 0u ? 1u : 0u);
        const std::uint64_t state_mass = left_state + middle_state + right_state;
        const std::uint64_t frequency =
            evidence.unit_mass[first] + evidence.unit_mass[second];
        if (winner0 == kSurfaceOrganNoUnit || state_edges > winner_state_edges ||
            (state_edges == winner_state_edges && state_mass > winner_state_mass) ||
            (state_edges == winner_state_edges && state_mass == winner_state_mass &&
             context > winner_context) ||
            (state_edges == winner_state_edges && state_mass == winner_state_mass &&
             context == winner_context && frequency > winner_frequency) ||
            (state_edges == winner_state_edges && state_mass == winner_state_mass &&
             context == winner_context && frequency == winner_frequency &&
             (first < winner0 ||
              (first == winner0 && second < winner1)))) {
          winner0 = first;
          winner1 = second;
          winner_state_edges = state_edges;
          winner_state_mass = state_mass;
          winner_context = context;
          winner_frequency = frequency;
        }
      }
    }
  }

  if (winner0 == kSurfaceOrganNoUnit)
    return false;
  selected[0] = winner0;
  if (path.length == 2u) {
    if (winner1 == kSurfaceOrganNoUnit)
      return false;
    selected[1] = winner1;
  }
  return true;
}

__device__ inline bool surface_append_unit(SurfaceRealizationWorkspaceView workspace,
                                           std::uint32_t unit, bool anchor,
                                           std::uint32_t* output_count) {
  if (*output_count >= workspace.output_unit_capacity ||
      *output_count >= kSurfaceOrganMaxOutputUnits)
    return false;
  workspace.output_units[*output_count] = unit;
  workspace.output_anchor_mask[*output_count] = anchor ? 1u : 0u;
  ++*output_count;
  return true;
}

template <typename BigramKey, typename TrigramKey,
          typename Access = resident_roles::DefaultNgramAccess>
__device__ inline bool surface_append_contextual_role_path(
    SurfaceUnitView units, MutableSurfaceGrammarEvidenceView evidence,
    OpaqueContentPlanView plan,
    SurfaceSequenceEvidenceView<BigramKey, TrigramKey> sequence,
    SurfaceRealizationWorkspaceView workspace, SurfaceRolePathChoice path,
    std::uint32_t left_anchor, std::uint32_t right_anchor,
    std::uint32_t* output_count) {
  std::uint32_t selected[kSurfaceOrganMaxBridgeRoles]{};
  if (!surface_select_contextual_role_path<BigramKey, TrigramKey, Access>(
          units, evidence, plan, sequence, path, left_anchor, right_anchor,
          workspace.output_units, *output_count, selected))
    return false;
  for (std::uint32_t index = 0u; index < path.length; ++index)
    if (!surface_append_unit(workspace, selected[index], false, output_count))
      return false;
  return true;
}

__device__ inline bool surface_append_role_path(
    SurfaceUnitView units, MutableSurfaceGrammarEvidenceView evidence, OpaqueContentPlanView plan,
    SurfaceRealizationWorkspaceView workspace, SurfaceRolePathChoice path,
    SurfaceUnitCompetition first_competition, SurfaceUnitCompetition last_competition,
    std::uint32_t right_boundary_unit, std::uint32_t* output_count) {
  for (std::uint32_t index = 0u; index < path.length; ++index) {
    SurfaceUnitCompetition competition = SurfaceUnitCompetition::kInterior;
    if (path.length == 1u && first_competition != SurfaceUnitCompetition::kInterior)
      competition = first_competition;
    else if (path.length == 1u)
      competition = last_competition;
    else if (index == 0u)
      competition = first_competition;
    else if (index + 1u == path.length)
      competition = last_competition;
    const std::uint32_t right_unit =
        index + 1u == path.length ? right_boundary_unit : kSurfaceOrganNoUnit;
    const std::uint32_t unit =
        surface_select_unit_for_role(units, evidence, plan, path.roles[index], competition,
                                     workspace.output_units, *output_count, right_unit);
    if (unit == kSurfaceOrganNoUnit || !surface_append_unit(workspace, unit, false, output_count))
      return false;
  }
  return true;
}

__device__ inline bool surface_emit_bytes(SurfaceUnitView units,
                                          SurfaceRealizationWorkspaceView workspace,
                                          std::uint32_t output_count, std::uint32_t* output_bytes) {
  std::uint32_t byte_count = 0u;
  bool visible = false;
  for (std::uint32_t position = 0u; position < output_count; ++position) {
    const std::uint32_t unit = workspace.output_units[position];
    if (unit >= units.unit_count)
      return false;
    const std::uint32_t length = units.lengths[unit];
    if (length > units.unit_words * 4u || byte_count > workspace.output_byte_capacity ||
        length > workspace.output_byte_capacity - byte_count)
      return false;
    for (std::uint32_t offset = 0u; offset < length; ++offset) {
      const std::uint32_t word =
          units.packed_bytes[static_cast<std::size_t>(unit) * units.unit_words + offset / 4u];
      const std::uint8_t byte =
          static_cast<std::uint8_t>((word >> ((offset % 4u) * 8u)) & 0xffu);
      workspace.output_bytes[byte_count++] = byte;
      visible = visible || (byte != 0u && byte != '\t' && byte != '\n' &&
                            byte != '\r' && byte != ' ');
    }
  }
  if (!visible)
    return false;
  *output_bytes = byte_count;
  return true;
}

__device__ inline void surface_clear_output(SurfaceRealizationWorkspaceView workspace) {
  for (std::uint32_t index = 0u; index < workspace.output_unit_capacity; ++index) {
    workspace.output_units[index] = kSurfaceOrganNoUnit;
    workspace.output_anchor_mask[index] = 0u;
  }
  for (std::uint32_t index = 0u; index < workspace.output_byte_capacity; ++index)
    workspace.output_bytes[index] = 0u;
}

[[nodiscard]] __device__ inline bool construction_complete_slot_path(
    SurfaceUnitView units, OpaqueContentPlanView plan,
    OpaqueConstructionWitnessView witness, std::uint32_t construction_index,
    std::uint32_t* slot_roles) {
  const std::uint32_t extent = witness.lengths[construction_index];
  const std::uint32_t slots = witness.slot_counts[construction_index];
  const bool population_only = slots == extent;
  const std::uint32_t minimum_extent =
      population_only ? construction::kConstructionMinSlots
                      : construction::kConstructionMinTokens;
  if (extent < minimum_extent ||
      extent > construction::kConstructionMaxTokens ||
      slots != plan.anchor_count || slots < construction::kConstructionMinSlots ||
      slots > construction::kConstructionMaxSlots)
    return false;

  std::uint32_t slot_count = 0u;
  std::uint32_t literal_run = 0u;
  std::uint32_t max_literal_run = 0u;
  for (std::uint32_t index = 0u; index < extent; ++index) {
    const std::uint32_t token =
        witness.tokens[construction_index * construction::kConstructionMaxTokens + index];
    if (construction::token_is_slot(token)) {
      const std::uint32_t role = construction::token_role(token);
      if (slot_count >= slots || role >= resident_roles::kStructuralRoleCount)
        return false;
      slot_roles[slot_count++] = role;
      literal_run = 0u;
      continue;
    }
    if (token >= units.unit_count || witness.closed_class_mask == nullptr ||
        witness.closed_class_mask[token] == 0u || units.lengths[token] < 2u ||
        units.lengths[token] > units.unit_words * 4u)
      return false;
    ++literal_run;
    if (literal_run > max_literal_run)
      max_literal_run = literal_run;
  }
  const std::uint32_t final_token =
      witness.tokens[construction_index * construction::kConstructionMaxTokens + extent - 1u];
  // A learned construction may end in a content slot. Literal closure remains
  // valid only when acquisition classified the whole unit as closed-class.
  const bool exact_selected_witness =
      witness.required_construction != nullptr &&
      witness.required_construction[0] == construction_index;
  const bool terminal_valid = construction::token_is_slot(final_token)
      ? exact_selected_witness
      : units.lengths[final_token] <= 12u;
  return slot_count == slots &&
         (population_only ||
          (terminal_valid &&
           max_literal_run <= construction::kConstructionMaxLiteralRun));
}

[[nodiscard]] __device__ inline bool construction_unique_anchor_mapping(
    SurfaceUnitView units, OpaqueContentPlanView plan,
    OpaqueConstructionWitnessView witness, const std::uint32_t* slot_roles,
    std::uint32_t construction_index, std::uint32_t* mapping,
    std::uint32_t* mapping_distance, std::uint64_t* mapping_evidence) {
  if (plan.anchor_count == 0u ||
      plan.anchor_count > construction::kConstructionMaxSlots)
    return false;
  for (std::uint32_t anchor = 0u; anchor < plan.anchor_count; ++anchor) {
    const std::uint32_t unit = plan.anchor_units[anchor];
    if (unit >= units.unit_count || units.lengths[unit] == 0u ||
        witness.roles[unit].confidence == 0u ||
        witness.roles[unit].role >= resident_roles::kStructuralRoleCount)
      return false;
  }

  const std::uint32_t permutation_count = surface_factorial(plan.anchor_count);
  bool found = false;
  bool tied = false;
  std::uint32_t best_population_slots = 0u;
  std::uint64_t best_population_mass = 0u;
  for (std::uint32_t candidate = 0u; candidate < permutation_count; ++candidate) {
    std::uint32_t permutation[kSurfaceOrganMaxAnchors]{};
    surface_decode_permutation(candidate, plan.anchor_count, permutation);
    bool supported = true;
    std::uint32_t population_slots = 0u;
    std::uint64_t population_mass = 0u;
    for (std::uint32_t slot = 0u; slot < plan.anchor_count; ++slot) {
      const std::uint32_t unit = plan.anchor_units[permutation[slot]];
      std::uint32_t learned_mass = 0u;
      std::uint64_t population_role_mass = 0u;
      if (witness.slot_units != nullptr && witness.slot_masses != nullptr &&
          (witness.slot_overflow == nullptr ||
           witness.slot_overflow[construction::construction_slot_index(
               construction_index, slot)] == 0u)) {
        for (std::uint32_t member = 0u;
             member < construction::kConstructionSlotPopulationCap; ++member) {
          const std::size_t member_index =
              construction::construction_slot_member_index(construction_index, slot,
                                                            member);
          const std::uint32_t member_unit = witness.slot_units[member_index];
          const std::uint32_t member_mass = witness.slot_masses[member_index];
          if (member_unit >= units.unit_count || member_mass == 0u) continue;
          if (member_unit == unit) learned_mass = member_mass;
          if (witness.roles[member_unit].confidence != 0u) {
            const std::uint32_t role_similarity =
                resident_roles::kRoleProjectionBits -
                __popc(witness.roles[member_unit].role ^
                       witness.roles[unit].role);
            population_role_mass +=
                static_cast<std::uint64_t>(member_mass) * role_similarity;
          }
        }
      }
      if (learned_mass != 0u) {
        ++population_slots;
        population_mass +=
            (static_cast<std::uint64_t>(learned_mass) << 32u) +
            population_role_mass;
      } else if (population_role_mass != 0u) {
        ++population_slots;
        population_mass += population_role_mass;
      } else if (witness.roles[unit].role != slot_roles[slot]) {
        supported = false;
        break;
      }
    }
    if (!supported)
      continue;
    if (!found || population_slots > best_population_slots ||
        (population_slots == best_population_slots &&
         population_mass > best_population_mass)) {
      found = true;
      tied = false;
      best_population_slots = population_slots;
      best_population_mass = population_mass;
      for (std::uint32_t slot = 0u; slot < plan.anchor_count; ++slot)
        mapping[slot] = permutation[slot];
    } else if (population_slots == best_population_slots &&
               population_mass == best_population_mass) {
      tied = true;
    }
  }
  if (!found || tied)
    return false;
  *mapping_distance = plan.anchor_count - best_population_slots;
  *mapping_evidence = best_population_mass;
  return true;
}

// A one-observation skeleton is episodic rather than a transferable grammar
// witness. It may therefore realize only when every slot is grounded by the
// exact unit population that formed that observation. This is an ordinary
// population-membership check: no word, question, or role label is privileged.
[[nodiscard]] __device__ inline bool construction_exact_anchor_population(
    SurfaceUnitView units, OpaqueContentPlanView plan,
    OpaqueConstructionWitnessView witness, std::uint32_t construction_index,
    const std::uint32_t* mapping) {
  if (witness.slot_units == nullptr || witness.slot_masses == nullptr)
    return false;
  for (std::uint32_t slot = 0u; slot < plan.anchor_count; ++slot) {
    if (witness.slot_overflow != nullptr &&
        witness.slot_overflow[construction::construction_slot_index(
            construction_index, slot)] != 0u)
      return false;
    const std::uint32_t unit = plan.anchor_units[mapping[slot]];
    bool found = false;
    for (std::uint32_t member = 0u;
         member < construction::kConstructionSlotPopulationCap; ++member) {
      const std::size_t member_index =
          construction::construction_slot_member_index(construction_index, slot,
                                                        member);
      if (witness.slot_units[member_index] == unit &&
          witness.slot_masses[member_index] != 0u) {
        found = true;
        break;
      }
    }
    if (!found || unit >= units.unit_count)
      return false;
  }
  return true;
}

// A joint learned skeleton is only a grammar witness. Its slot path may
// permute committed anchors inside this Plan step, but it cannot select or
// replace them. A role projection is authorized only after it recurs as the
// exact learned slot witness; geometric proximity alone is not evidence that a
// previously unseen role belongs in that slot. Ties remain silent.
__device__ inline void realize_construction_witness(
    SurfaceUnitView units, OpaqueContentPlanView plan,
    OpaqueConstructionWitnessView witness,
    SurfaceRealizationWorkspaceView workspace) {
  *workspace.result = {};
  surface_clear_output(workspace);
  if (witness.tokens == nullptr || witness.lengths == nullptr ||
      witness.slot_counts == nullptr || witness.supports == nullptr ||
      witness.count == nullptr || witness.roles == nullptr ||
      witness.capacity == 0u || plan.anchor_count == 0u ||
      plan.anchor_count > construction::kConstructionMaxSlots)
    return;

  const std::uint32_t total = witness.count[0] < witness.capacity
                                  ? witness.count[0]
                                  : witness.capacity;
  workspace.result->construction_count = total;
  std::uint32_t selected = kSurfaceOrganNoUnit;
  std::uint32_t selected_support = 0u;
  std::uint32_t selected_distance = 0xffffffffu;
  std::uint64_t selected_mapping_evidence = 0u;
  std::uint32_t selected_mapping[construction::kConstructionMaxSlots]{};
  bool tied = false;
  for (std::uint32_t construction_index = 0u; construction_index < total;
       ++construction_index) {
    if (witness.required_construction != nullptr &&
        witness.required_construction[0] != construction::kNoConstruction &&
        construction_index != witness.required_construction[0])
      continue;
    const std::uint32_t support = witness.supports[construction_index];
    const bool recurrent = support >= construction::kConstructionMinRoleEvidence;
    std::uint32_t slot_roles[construction::kConstructionMaxSlots]{};
    std::uint32_t mapping[construction::kConstructionMaxSlots]{};
    std::uint32_t mapping_distance = 0xffffffffu;
    std::uint64_t mapping_evidence = 0u;
    if (!construction_complete_slot_path(units, plan, witness,
                                         construction_index, slot_roles))
      continue;
    ++workspace.result->construction_shape_matched;
    if (!construction_unique_anchor_mapping(units, plan, witness, slot_roles,
                                            construction_index, mapping,
                                            &mapping_distance, &mapping_evidence))
      continue;
    ++workspace.result->construction_mapping_matched;
    if (!recurrent && !construction_exact_anchor_population(
                          units, plan, witness, construction_index, mapping))
      continue;
    ++workspace.result->construction_supported;
    if (selected == kSurfaceOrganNoUnit || mapping_distance < selected_distance ||
        (mapping_distance == selected_distance &&
         (mapping_evidence > selected_mapping_evidence ||
          (mapping_evidence == selected_mapping_evidence &&
           support > selected_support)))) {
      selected = construction_index;
      selected_support = support;
      selected_distance = mapping_distance;
      selected_mapping_evidence = mapping_evidence;
      tied = false;
      for (std::uint32_t slot = 0u; slot < plan.anchor_count; ++slot)
        selected_mapping[slot] = mapping[slot];
    } else if (mapping_distance == selected_distance &&
               mapping_evidence == selected_mapping_evidence &&
               support == selected_support) {
      tied = true;
    }
  }
  workspace.result->construction_tied = tied ? 1u : 0u;
  if (selected == kSurfaceOrganNoUnit || tied)
    return;

  const std::uint32_t extent = witness.lengths[selected];
  if (extent > workspace.output_unit_capacity) {
    workspace.result->capacity_exceeded = 1u;
    return;
  }
  std::uint32_t required_bytes = 0u;
  std::uint32_t slot = 0u;
  for (std::uint32_t index = 0u; index < extent; ++index) {
    const std::uint32_t token =
        witness.tokens[selected * construction::kConstructionMaxTokens + index];
    const std::uint32_t unit = construction::token_is_slot(token)
                                   ? plan.anchor_units[selected_mapping[slot++]]
                                   : token;
    const std::uint32_t length = units.lengths[unit];
    if (length > workspace.output_byte_capacity - required_bytes) {
      workspace.result->capacity_exceeded = 1u;
      return;
    }
    required_bytes += length;
  }

  slot = 0u;
  bool reordered = false;
  for (std::uint32_t index = 0u; index < extent; ++index) {
    const std::uint32_t token =
        witness.tokens[selected * construction::kConstructionMaxTokens + index];
    const bool is_slot = construction::token_is_slot(token);
    if (is_slot) {
      reordered = reordered || selected_mapping[slot] != slot;
      workspace.output_units[index] = plan.anchor_units[selected_mapping[slot++]];
    } else {
      workspace.output_units[index] = token;
    }
    workspace.output_anchor_mask[index] = is_slot ? 1u : 0u;
  }
  std::uint32_t byte_count = 0u;
  if (!surface_emit_bytes(units, workspace, extent, &byte_count)) {
    *workspace.result = {};
    workspace.result->capacity_exceeded = 1u;
    surface_clear_output(workspace);
    return;
  }
  workspace.result->ready = 1u;
  workspace.result->grammar_supported = 1u;
  workspace.result->closure_supported = 1u;
  workspace.result->anchors_preserved = plan.anchor_count;
  workspace.result->output_unit_count = extent;
  workspace.result->output_byte_count = byte_count;
  workspace.result->connector_count = extent - plan.anchor_count;
  workspace.result->plan_reordered = reordered ? 1u : 0u;
  workspace.result->path_quality_q20 = selected_support;
}

// Realize an exact event-backed construction whose anchors have already been
// recovered in learned slot order. This path does not rerun permutation
// competition: doing so would discard event identity by asking an aggregate
// slot population to rediscover an order the retained event already proves.
// Every anchor is still required to occur in that construction's learned slot
// population, and all ordinary shape, capacity, and byte checks remain.
__device__ inline void realize_ordered_construction_witness(
    SurfaceUnitView units, OpaqueContentPlanView plan,
    OpaqueConstructionWitnessView witness,
    SurfaceRealizationWorkspaceView workspace) {
  *workspace.result = {};
  surface_clear_output(workspace);
  if (witness.tokens == nullptr || witness.lengths == nullptr ||
      witness.slot_counts == nullptr || witness.supports == nullptr ||
      witness.count == nullptr || witness.roles == nullptr ||
      witness.required_construction == nullptr || witness.capacity == 0u ||
      plan.anchor_count == 0u ||
      plan.anchor_count > construction::kConstructionMaxSlots)
    return;
  const std::uint32_t total = min(witness.count[0], witness.capacity);
  const std::uint32_t selected = witness.required_construction[0];
  workspace.result->construction_count = total;
  if (selected >= total ||
      selected == construction::kNoConstruction)
    return;
  std::uint32_t slot_roles[construction::kConstructionMaxSlots]{};
  if (!construction_complete_slot_path(units, plan, witness, selected,
                                       slot_roles))
    return;
  ++workspace.result->construction_shape_matched;
  std::uint32_t identity[construction::kConstructionMaxSlots]{};
  for (std::uint32_t slot = 0u; slot < plan.anchor_count; ++slot)
    identity[slot] = slot;
  if (!construction_exact_anchor_population(units, plan, witness, selected,
                                            identity))
    return;
  ++workspace.result->construction_mapping_matched;
  ++workspace.result->construction_supported;

  const std::uint32_t extent = witness.lengths[selected];
  if (extent > workspace.output_unit_capacity) {
    workspace.result->capacity_exceeded = 1u;
    return;
  }
  std::uint32_t slot = 0u;
  for (std::uint32_t index = 0u; index < extent; ++index) {
    const std::uint32_t token =
        witness.tokens[selected * construction::kConstructionMaxTokens + index];
    const bool is_slot = construction::token_is_slot(token);
    workspace.output_units[index] =
        is_slot ? plan.anchor_units[slot++] : token;
    workspace.output_anchor_mask[index] = is_slot ? 1u : 0u;
  }
  std::uint32_t byte_count = 0u;
  if (!surface_emit_bytes(units, workspace, extent, &byte_count)) {
    *workspace.result = {};
    workspace.result->capacity_exceeded = 1u;
    surface_clear_output(workspace);
    return;
  }
  workspace.result->ready = 1u;
  workspace.result->grammar_supported = 1u;
  workspace.result->closure_supported = 1u;
  workspace.result->anchors_preserved = plan.anchor_count;
  workspace.result->output_unit_count = extent;
  workspace.result->output_byte_count = byte_count;
  workspace.result->connector_count = extent - plan.anchor_count;
  workspace.result->path_quality_q20 = witness.supports[selected];
}

#include "bcc32_cuda_resident_surface_organ_v2_tail.inl"
