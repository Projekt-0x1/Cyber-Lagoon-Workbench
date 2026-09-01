#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

#include "bcc32_cuda_resident_roles.cuh"

// A bounded language-surface organ over units already discovered from raw byte
// contact.  Episodes contribute only mutable structural evidence: their unit
// identities are reduced to resident roles and terminal support before
// rendering.  Content anchors, skeleton choice, glue choice, novelty rejection,
// and final selection all remain on the device.
namespace substrate::bcc32::resident_surface_renderer {

inline constexpr std::uint32_t kSurfaceRendererBlockSize = 128u;
inline constexpr std::uint32_t kSurfaceRendererMaxSkeletonUnits = 64u;
inline constexpr std::uint32_t kSurfaceRendererMaxAnchors = 8u;
inline constexpr std::uint32_t kSurfaceRendererNoUnit = 0xffffffffu;

struct SurfaceRendererConfig {
  std::uint32_t max_units = 32u;
  std::uint32_t min_units = 4u;
  std::uint32_t source_run_limit = 4u;
};

struct SurfaceSourceWindowSignature {
  std::uint32_t hash_a = 0u;
  std::uint32_t hash_b = 0u;

  [[nodiscard]] __host__ __device__ bool operator==(const SurfaceSourceWindowSignature& other) const {
    return hash_a == other.hash_a && hash_b == other.hash_b;
  }
};

struct SurfaceRendererResult {
  std::uint32_t ready = 0u;
  std::uint32_t selected_skeleton = kSurfaceRendererNoUnit;
  std::uint32_t output_count = 0u;
  std::uint32_t anchors_preserved = 0u;
  std::uint32_t closed = 0u;
  std::uint32_t source_novel = 0u;
  std::uint32_t glue_count = 0u;
  std::uint32_t glue_with_evidence = 0u;
  std::uint32_t recurring_support = 0u;
  std::uint32_t score_low = 0u;
  std::uint32_t score_high = 0u;
};

template <typename BigramKeyT, typename TrigramKeyT>
struct SurfaceRendererModelView {
  // Units are opaque, mutable chunks discovered at the raw body surface.
  const std::uint32_t* unit_lengths = nullptr;
  const std::uint32_t* unit_content = nullptr;
  const std::uint32_t* unit_vitality = nullptr;
  std::uint32_t unit_count = 0u;
  std::uint32_t unit_words = 0u;

  const resident_roles::MutableStructuralRole* unit_roles = nullptr;
  resident_roles::RoleEvidenceTables role_evidence{};

  const BigramKeyT* base_bigrams = nullptr;
  const std::uint32_t* base_bigram_counts = nullptr;
  std::uint32_t base_bigram_count = 0u;
  const TrigramKeyT* base_trigrams = nullptr;
  const std::uint32_t* base_trigram_counts = nullptr;
  std::uint32_t base_trigram_count = 0u;

  // Offsets has episode_count + 1 entries.  Episode boundaries are the only
  // closure signal; no punctuation, word class, or authored frame is supplied.
  const std::uint32_t* grammar_episode_units = nullptr;
  const std::uint32_t* grammar_episode_offsets = nullptr;
  std::uint32_t grammar_episode_count = 0u;

  const std::uint32_t* content_anchors = nullptr;
  std::uint32_t content_anchor_count = 0u;

  // Hashes disclose whether a bounded output run occurred in any source, but
  // provide neither source positions nor source text to the renderer.
  const SurfaceSourceWindowSignature* source_windows = nullptr;
  std::uint32_t source_window_count = 0u;
};

struct SurfaceRendererWorkspaceView {
  std::uint32_t* skeleton_roles = nullptr;
  std::uint32_t* skeleton_lengths = nullptr;
  std::uint64_t* skeleton_scores = nullptr;
  std::uint32_t* skeleton_support = nullptr;
  std::uint32_t* skeleton_valid = nullptr;
  std::uint32_t* anchor_positions = nullptr;
  std::uint32_t skeleton_capacity = 0u;

  std::uint32_t* terminal_counts = nullptr;
  std::uint32_t terminal_capacity = 0u;
  std::uint32_t* output_units = nullptr;
  std::uint32_t output_capacity = 0u;
  SurfaceRendererResult* result = nullptr;
};

[[nodiscard]] __host__ __device__ constexpr std::uint32_t surface_mix32(std::uint32_t value) {
  value ^= value >> 16u;
  value *= 0x7feb352du;
  value ^= value >> 15u;
  value *= 0x846ca68bu;
  value ^= value >> 16u;
  return value;
}

[[nodiscard]] __host__ __device__ inline SurfaceSourceWindowSignature surface_window_signature(
    const std::uint32_t* units, std::uint32_t count) {
  std::uint32_t a = 0x243f6a88u ^ count;
  std::uint32_t b = 0x85a308d3u + count;
  for (std::uint32_t index = 0u; index < count; ++index) {
    a = surface_mix32(a ^ units[index] ^ ((index + 1u) * 0x9e3779b9u));
    b = surface_mix32(b + units[index] * 0x85ebca6bu + index);
  }
  return {a, b};
}

[[nodiscard]] __device__ inline bool surface_window_seen(
    const SurfaceSourceWindowSignature* windows, std::uint32_t window_count,
    SurfaceSourceWindowSignature target) {
  for (std::uint32_t index = 0u; index < window_count; ++index) {
    if (windows[index] == target)
      return true;
  }
  return false;
}

[[nodiscard]] __host__ __device__ inline std::uint64_t surface_sat_add(std::uint64_t first,
                                                                       std::uint64_t second) {
  const std::uint64_t result = first + second;
  return result < first ? ~0ull : result;
}

[[nodiscard]] __host__ __device__ inline std::uint64_t surface_sat_mul(std::uint64_t value,
                                                                       std::uint64_t factor) {
  if (factor != 0u && value > ~0ull / factor)
    return ~0ull;
  return value * factor;
}

template <typename BigramKeyT, typename Access = resident_roles::DefaultNgramAccess>
[[nodiscard]] __host__ __device__ inline std::uint64_t surface_bigram_count(
    const BigramKeyT* keys, const std::uint32_t* counts, std::uint32_t key_count,
    std::uint32_t previous, std::uint32_t next) {
  std::uint64_t total = 0u;
  for (std::uint32_t index = 0u; index < key_count; ++index) {
    if (Access::bigram_previous(keys[index]) == previous && Access::bigram_next(keys[index]) == next)
      total = surface_sat_add(total, counts[index]);
  }
  return total;
}

template <typename TrigramKeyT, typename Access = resident_roles::DefaultNgramAccess>
[[nodiscard]] __host__ __device__ inline std::uint64_t surface_trigram_count(
    const TrigramKeyT* keys, const std::uint32_t* counts, std::uint32_t key_count,
    std::uint32_t first, std::uint32_t previous, std::uint32_t next) {
  std::uint64_t total = 0u;
  for (std::uint32_t index = 0u; index < key_count; ++index) {
    if (Access::trigram_first(keys[index]) == first &&
        Access::trigram_second(keys[index]) == previous &&
        Access::trigram_next(keys[index]) == next) {
      total = surface_sat_add(total, counts[index]);
    }
  }
  return total;
}

template <typename BigramKeyT, typename Access = resident_roles::DefaultNgramAccess>
[[nodiscard]] __device__ inline bool surface_has_base_incident(
    const BigramKeyT* keys, const std::uint32_t* counts, std::uint32_t key_count,
    std::uint32_t unit) {
  for (std::uint32_t index = 0u; index < key_count; ++index) {
    if (counts[index] != 0u &&
        (Access::bigram_previous(keys[index]) == unit || Access::bigram_next(keys[index]) == unit))
      return true;
  }
  return false;
}

[[nodiscard]] __device__ inline std::uint64_t surface_role_path_score(
    const std::uint32_t* roles, std::uint32_t length,
    resident_roles::RoleEvidenceTables evidence) {
  std::uint64_t score = 0u;
  for (std::uint32_t position = 1u; position < length; ++position) {
    const std::uint64_t pair = evidence.base_grammar.bigrams == nullptr
                                   ? 0u
                                   : evidence.base_grammar.bigrams[resident_roles::role_bigram_index(
                                         roles[position - 1u], roles[position])];
    score = surface_sat_add(score, surface_sat_mul(resident_roles::integer_log_depth(pair), 64u));
    if (position >= 2u && evidence.base_grammar.trigrams != nullptr) {
      const std::uint64_t triple = evidence.base_grammar.trigrams[resident_roles::role_trigram_index(
          roles[position - 2u], roles[position - 1u], roles[position])];
      score = surface_sat_add(score,
                              surface_sat_mul(resident_roles::integer_log_depth(triple), 256u));
    }
  }
  return score;
}

template <typename BigramKeyT, typename TrigramKeyT>
__global__ void learn_surface_skeletons_kernel(
    SurfaceRendererModelView<BigramKeyT, TrigramKeyT> model, SurfaceRendererConfig config,
    SurfaceRendererWorkspaceView workspace) {
  const std::uint32_t episode = blockIdx.x * blockDim.x + threadIdx.x;
  if (episode >= model.grammar_episode_count || episode >= workspace.skeleton_capacity)
    return;
  const std::uint32_t begin = model.grammar_episode_offsets[episode];
  const std::uint32_t end = model.grammar_episode_offsets[episode + 1u];
  const std::uint32_t length = end >= begin ? end - begin : 0u;
  workspace.skeleton_lengths[episode] = 0u;
  workspace.skeleton_scores[episode] = 0u;
  workspace.skeleton_support[episode] = 0u;
  workspace.skeleton_valid[episode] = 0u;
  if (length < config.min_units || length > config.max_units ||
      length > kSurfaceRendererMaxSkeletonUnits)
    return;

  std::uint32_t* skeleton = workspace.skeleton_roles +
      static_cast<std::size_t>(episode) * kSurfaceRendererMaxSkeletonUnits;
  for (std::uint32_t position = 0u; position < length; ++position) {
    const std::uint32_t unit = model.grammar_episode_units[begin + position];
    if (unit >= model.unit_count || model.unit_lengths[unit] == 0u)
      return;
    skeleton[position] = model.unit_roles[unit].role;
  }
  const std::uint32_t terminal = model.grammar_episode_units[end - 1u];
  atomicAdd(workspace.terminal_counts + terminal, 1u);
  workspace.skeleton_lengths[episode] = length;
  workspace.skeleton_scores[episode] = surface_role_path_score(skeleton, length, model.role_evidence);
}

template <typename BigramKeyT, typename TrigramKeyT>
__global__ void score_surface_skeletons_kernel(
    SurfaceRendererModelView<BigramKeyT, TrigramKeyT> model,
    SurfaceRendererWorkspaceView workspace) {
  const std::uint32_t episode = blockIdx.x * blockDim.x + threadIdx.x;
  if (episode >= model.grammar_episode_count || episode >= workspace.skeleton_capacity)
    return;
  const std::uint32_t length = workspace.skeleton_lengths[episode];
  if (length == 0u || length <= model.content_anchor_count)
    return;
  const std::uint32_t* skeleton = workspace.skeleton_roles +
      static_cast<std::size_t>(episode) * kSurfaceRendererMaxSkeletonUnits;
  std::uint32_t* positions = workspace.anchor_positions +
      static_cast<std::size_t>(episode) * kSurfaceRendererMaxAnchors;

  std::uint32_t cursor = 0u;
  std::uint64_t anchor_score = 0u;
  for (std::uint32_t anchor_index = 0u; anchor_index < model.content_anchor_count; ++anchor_index) {
    const std::uint32_t anchor = model.content_anchors[anchor_index];
    if (anchor >= model.unit_count)
      return;
    const std::uint32_t role = model.unit_roles[anchor].role;
    const std::uint32_t remaining = model.content_anchor_count - anchor_index;
    const std::uint32_t upper = length - remaining;  // Leaves a learned terminal slot.
    std::uint32_t match = kSurfaceRendererNoUnit;
    for (std::uint32_t position = cursor; position < upper; ++position) {
      if (skeleton[position] == role) {
        match = position;
        break;
      }
    }
    if (match == kSurfaceRendererNoUnit)
      return;
    positions[anchor_index] = match;
    cursor = match + 1u;
    anchor_score = surface_sat_add(
        anchor_score, static_cast<std::uint64_t>(model.unit_roles[anchor].confidence) +
                          model.unit_roles[anchor].evidence_depth + 1u);
  }

  std::uint32_t support = 0u;
  for (std::uint32_t other = 0u;
       other < model.grammar_episode_count && other < workspace.skeleton_capacity; ++other) {
    if (workspace.skeleton_lengths[other] != length)
      continue;
    const std::uint32_t* comparison = workspace.skeleton_roles +
        static_cast<std::size_t>(other) * kSurfaceRendererMaxSkeletonUnits;
    bool equal = true;
    for (std::uint32_t position = 0u; position < length; ++position)
      equal = equal && skeleton[position] == comparison[position];
    support += equal ? 1u : 0u;
  }
  workspace.skeleton_support[episode] = support;
  workspace.skeleton_scores[episode] = surface_sat_add(
      workspace.skeleton_scores[episode],
      surface_sat_add(surface_sat_mul(support, 512u), anchor_score));
  workspace.skeleton_valid[episode] = 1u;
}

template <typename BigramKeyT, typename TrigramKeyT,
          typename Access = resident_roles::DefaultNgramAccess>
[[nodiscard]] __device__ inline std::uint64_t surface_candidate_score(
    const SurfaceRendererModelView<BigramKeyT, TrigramKeyT>& model, std::uint32_t first,
    std::uint32_t previous, std::uint32_t candidate, std::uint32_t terminal_support,
    bool has_first, bool has_previous, bool* has_learned_evidence) {
  *has_learned_evidence = terminal_support != 0u;
  std::uint64_t score = static_cast<std::uint64_t>(model.unit_vitality == nullptr
                                                       ? 1u
                                                       : model.unit_vitality[candidate]) +
      terminal_support * 128u;
  if (!has_previous)
    return score;
  const resident_roles::RoleCompatibilityScore role = resident_roles::score_role_compatibility(
      has_first ? first : previous, previous, candidate, model.unit_roles, model.unit_count,
      model.role_evidence);
  *has_learned_evidence = *has_learned_evidence || role.total() != 0u;
  score = surface_sat_add(score, surface_sat_mul(role.total(), 8u));
  const std::uint64_t pair = surface_bigram_count<BigramKeyT, Access>(
      model.base_bigrams, model.base_bigram_counts, model.base_bigram_count, previous, candidate);
  *has_learned_evidence = *has_learned_evidence || pair != 0u;
  score = surface_sat_add(score,
                          surface_sat_mul(resident_roles::integer_log_depth(pair), 256u));
  if (has_first) {
    const std::uint64_t triple = surface_trigram_count<TrigramKeyT, Access>(
        model.base_trigrams, model.base_trigram_counts, model.base_trigram_count, first, previous,
        candidate);
    *has_learned_evidence = *has_learned_evidence || triple != 0u;
    score = surface_sat_add(score,
                            surface_sat_mul(resident_roles::integer_log_depth(triple), 1024u));
  }
  return score;
}

template <typename BigramKeyT, typename TrigramKeyT,
          typename Access = resident_roles::DefaultNgramAccess>
__global__ void select_and_render_surface_kernel(
    SurfaceRendererModelView<BigramKeyT, TrigramKeyT> model, SurfaceRendererConfig config,
    SurfaceRendererWorkspaceView workspace) {
  if (blockIdx.x != 0u || threadIdx.x != 0u)
    return;
  *workspace.result = {};
  for (std::uint32_t index = 0u; index < workspace.output_capacity; ++index)
    workspace.output_units[index] = kSurfaceRendererNoUnit;

  std::uint32_t winner = kSurfaceRendererNoUnit;
  std::uint64_t winner_score = 0u;
  for (std::uint32_t episode = 0u;
       episode < model.grammar_episode_count && episode < workspace.skeleton_capacity; ++episode) {
    if (workspace.skeleton_valid[episode] == 0u)
      continue;
    const std::uint64_t score = workspace.skeleton_scores[episode];
    if (winner == kSurfaceRendererNoUnit || score > winner_score ||
        (score == winner_score && episode < winner)) {
      winner = episode;
      winner_score = score;
    }
  }
  if (winner == kSurfaceRendererNoUnit)
    return;

  const std::uint32_t length = workspace.skeleton_lengths[winner];
  if (length > workspace.output_capacity)
    return;
  const std::uint32_t* skeleton = workspace.skeleton_roles +
      static_cast<std::size_t>(winner) * kSurfaceRendererMaxSkeletonUnits;
  const std::uint32_t* anchor_positions = workspace.anchor_positions +
      static_cast<std::size_t>(winner) * kSurfaceRendererMaxAnchors;
  std::uint32_t next_anchor = 0u;
  std::uint32_t glue_count = 0u;
  std::uint32_t glue_with_evidence = 0u;

  for (std::uint32_t position = 0u; position < length; ++position) {
    std::uint32_t selected = kSurfaceRendererNoUnit;
    std::uint64_t selected_score = 0u;
    const bool anchor_slot = next_anchor < model.content_anchor_count &&
        anchor_positions[next_anchor] == position;
    if (anchor_slot) {
      selected = model.content_anchors[next_anchor++];
    } else {
      ++glue_count;
      const bool final_slot = position + 1u == length;
      bool selected_has_evidence = false;
      for (std::uint32_t candidate = 0u; candidate < model.unit_count; ++candidate) {
        if (model.unit_lengths[candidate] == 0u || model.unit_roles[candidate].role != skeleton[position])
          continue;
        bool is_anchor = false;
        for (std::uint32_t anchor = 0u; anchor < model.content_anchor_count; ++anchor)
          is_anchor = is_anchor || model.content_anchors[anchor] == candidate;
        if (is_anchor || (final_slot && workspace.terminal_counts[candidate] == 0u) ||
            !surface_has_base_incident<BigramKeyT, Access>(
                model.base_bigrams, model.base_bigram_counts, model.base_bigram_count, candidate))
          continue;

        const bool has_previous = position != 0u;
        const bool has_first = position >= 2u;
        bool candidate_has_evidence = false;
        std::uint64_t score = surface_candidate_score<BigramKeyT, TrigramKeyT, Access>(
            model, has_first ? workspace.output_units[position - 2u] : 0u,
            has_previous ? workspace.output_units[position - 1u] : 0u, candidate,
            workspace.terminal_counts[candidate], has_first, has_previous,
            &candidate_has_evidence);

        if (next_anchor < model.content_anchor_count &&
            anchor_positions[next_anchor] == position + 1u) {
          const std::uint32_t upcoming = model.content_anchors[next_anchor];
          const resident_roles::RoleCompatibilityScore lookahead =
              resident_roles::score_role_compatibility(
                  has_previous ? workspace.output_units[position - 1u] : candidate, candidate,
                  upcoming, model.unit_roles, model.unit_count, model.role_evidence);
          score = surface_sat_add(score, surface_sat_mul(lookahead.total(), 4u));
          const std::uint64_t pair = surface_bigram_count<BigramKeyT, Access>(
              model.base_bigrams, model.base_bigram_counts, model.base_bigram_count, candidate,
              upcoming);
          candidate_has_evidence = candidate_has_evidence || lookahead.total() != 0u || pair != 0u;
          score = surface_sat_add(
              score, surface_sat_mul(resident_roles::integer_log_depth(pair), 128u));
        }

        workspace.output_units[position] = candidate;
        const bool makes_source_run = config.source_run_limit != 0u &&
            position + 1u >= config.source_run_limit &&
            surface_window_seen(
                model.source_windows, model.source_window_count,
                surface_window_signature(workspace.output_units + position + 1u -
                                             config.source_run_limit,
                                         config.source_run_limit));
        workspace.output_units[position] = kSurfaceRendererNoUnit;
        if (makes_source_run)
          continue;
        if (selected == kSurfaceRendererNoUnit || score > selected_score ||
            (score == selected_score && candidate < selected)) {
          selected = candidate;
          selected_score = score;
          selected_has_evidence = candidate_has_evidence;
        }
      }
      if (selected == kSurfaceRendererNoUnit)
        return;
      if (selected_has_evidence)
        ++glue_with_evidence;
    }
    workspace.output_units[position] = selected;
  }

  bool novel = length >= config.source_run_limit;
  if (config.source_run_limit == 0u)
    novel = true;
  for (std::uint32_t offset = 0u;
       config.source_run_limit != 0u && offset + config.source_run_limit <= length; ++offset) {
    if (surface_window_seen(model.source_windows, model.source_window_count,
                            surface_window_signature(workspace.output_units + offset,
                                                     config.source_run_limit))) {
      novel = false;
    }
  }
  const bool closed = length != 0u && workspace.terminal_counts[workspace.output_units[length - 1u]] != 0u;
  workspace.result->selected_skeleton = winner;
  workspace.result->output_count = length;
  workspace.result->anchors_preserved = next_anchor;
  workspace.result->closed = closed ? 1u : 0u;
  workspace.result->source_novel = novel ? 1u : 0u;
  workspace.result->glue_count = glue_count;
  workspace.result->glue_with_evidence = glue_with_evidence;
  workspace.result->recurring_support = workspace.skeleton_support[winner];
  workspace.result->score_low = static_cast<std::uint32_t>(winner_score);
  workspace.result->score_high = static_cast<std::uint32_t>(winner_score >> 32u);
  workspace.result->ready = closed && novel && next_anchor == model.content_anchor_count &&
          glue_count == glue_with_evidence
      ? 1u
      : 0u;
}

[[nodiscard]] inline constexpr std::uint32_t surface_renderer_blocks(std::uint32_t count) {
  return (count + kSurfaceRendererBlockSize - 1u) / kSurfaceRendererBlockSize;
}

template <typename BigramKeyT, typename TrigramKeyT,
          typename Access = resident_roles::DefaultNgramAccess>
inline cudaError_t render_learned_surface_cuda(
    SurfaceRendererModelView<BigramKeyT, TrigramKeyT> model, SurfaceRendererConfig config,
    SurfaceRendererWorkspaceView workspace, cudaStream_t stream = nullptr) {
  if (model.unit_count == 0u || model.unit_lengths == nullptr || model.unit_content == nullptr ||
      model.unit_words == 0u || model.unit_roles == nullptr ||
      (model.base_bigram_count != 0u &&
       (model.base_bigrams == nullptr || model.base_bigram_counts == nullptr)) ||
      (model.base_trigram_count != 0u &&
       (model.base_trigrams == nullptr || model.base_trigram_counts == nullptr)) ||
      (model.source_window_count != 0u && model.source_windows == nullptr) ||
      model.grammar_episode_units == nullptr ||
      model.grammar_episode_offsets == nullptr || model.grammar_episode_count == 0u ||
      model.content_anchors == nullptr || model.content_anchor_count == 0u ||
      model.content_anchor_count > kSurfaceRendererMaxAnchors || config.min_units == 0u ||
      config.max_units < config.min_units || config.max_units > kSurfaceRendererMaxSkeletonUnits ||
      config.source_run_limit > config.max_units || workspace.skeleton_capacity < model.grammar_episode_count ||
      workspace.terminal_capacity < model.unit_count || workspace.output_capacity < config.max_units ||
      workspace.skeleton_roles == nullptr || workspace.skeleton_lengths == nullptr ||
      workspace.skeleton_scores == nullptr || workspace.skeleton_support == nullptr ||
      workspace.skeleton_valid == nullptr || workspace.anchor_positions == nullptr ||
      workspace.terminal_counts == nullptr || workspace.output_units == nullptr ||
      workspace.result == nullptr) {
    return cudaErrorInvalidValue;
  }
  cudaError_t status = cudaMemsetAsync(workspace.terminal_counts, 0,
                                       model.unit_count * sizeof(std::uint32_t), stream);
  if (status != cudaSuccess)
    return status;
  status = cudaMemsetAsync(workspace.anchor_positions, 0xff,
                           static_cast<std::size_t>(workspace.skeleton_capacity) *
                               kSurfaceRendererMaxAnchors * sizeof(std::uint32_t),
                           stream);
  if (status != cudaSuccess)
    return status;

  const std::uint32_t blocks = surface_renderer_blocks(model.grammar_episode_count);
  learn_surface_skeletons_kernel<<<blocks, kSurfaceRendererBlockSize, 0, stream>>>(
      model, config, workspace);
  status = cudaPeekAtLastError();
  if (status != cudaSuccess)
    return status;
  score_surface_skeletons_kernel<<<blocks, kSurfaceRendererBlockSize, 0, stream>>>(model,
                                                                                   workspace);
  status = cudaPeekAtLastError();
  if (status != cudaSuccess)
    return status;
  select_and_render_surface_kernel<BigramKeyT, TrigramKeyT, Access><<<1u, 1u, 0, stream>>>(
      model, config, workspace);
  return cudaPeekAtLastError();
}

}  // namespace substrate::bcc32::resident_surface_renderer
