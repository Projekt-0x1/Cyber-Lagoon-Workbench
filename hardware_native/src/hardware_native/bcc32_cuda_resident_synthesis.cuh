#pragma once

#include <cuda_runtime.h>

#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/system/cuda/execution_policy.h>

#include <cstddef>
#include <cstdint>

#include "bcc32_cuda_resident_roles.cuh"
#include "bcc32_cuda_resident_synthesis_policy.cuh"

// Generic resident synthesis over units learned from raw byte contact.  This
// module never receives source positions, episodes, postings, words, tokens, or
// answer labels.  The host launches one fixed device pipeline; semantic seed
// choice, draft construction, repair, appraisal, and final selection all remain
// on the device.
namespace bcc32_cuda_resident_synthesis {

constexpr std::uint32_t kResidentSynthesisBlock = 256u;
constexpr std::uint32_t kResidentSynthesisMaxSeeds = 16u;
constexpr std::uint32_t kResidentSynthesisBaseVariants = 8u;
constexpr std::uint32_t kResidentSynthesisVariantsPerSeed = 16u;
constexpr std::uint32_t kResidentSynthesisMaxCandidates =
    kResidentSynthesisMaxSeeds * kResidentSynthesisVariantsPerSeed;
constexpr std::uint32_t kResidentSynthesisMaxUnits = 64u;
constexpr std::uint32_t kResidentSynthesisDefaultEdgeScan = 64u;
constexpr std::uint32_t kResidentSynthesisInvalid = 0xffffffffu;
constexpr std::uint32_t kResidentSynthesisNoveltyUnits = 6u;
constexpr std::uint32_t kResidentSynthesisConditionedMinUnits = 1u;
constexpr std::uint32_t kResidentSynthesisRecentUnits = 256u;
constexpr std::uint32_t kResidentSynthesisSubjectLag = 16u;
constexpr std::uint32_t kResidentSynthesisMaxRelationTail = 4u;

__host__ __device__ inline std::uint32_t synthesis_variant_trigram_weight(
    std::uint32_t variant) {
  constexpr std::uint32_t weights[kResidentSynthesisBaseVariants] = {
      1u, 2u, 4u, 8u, 16u, 1u, 2u, 4u};
  return weights[variant % kResidentSynthesisBaseVariants];
}

struct ResidentSynthesisConfig {
  std::uint32_t seed_count = kResidentSynthesisMaxSeeds;
  std::uint32_t candidate_count = kResidentSynthesisMaxCandidates;
  std::uint32_t max_units = kResidentSynthesisMaxUnits;
  std::uint32_t min_units = 8u;
  std::uint32_t edge_scan_limit = kResidentSynthesisDefaultEdgeScan;
  std::uint32_t repair_passes = 2u;
};

struct ResidentSourceWindowSignature {
  std::uint32_t hash_a;
  std::uint32_t hash_b;

  __host__ __device__ bool operator<(
      const ResidentSourceWindowSignature& other) const {
    return hash_a < other.hash_a ||
        (hash_a == other.hash_a && hash_b < other.hash_b);
  }
};

struct ResidentSubjectTransitionKey {
  std::uint32_t anchor;
  std::uint32_t previous;
  std::uint32_t next;

  __host__ __device__ bool operator<(
      const ResidentSubjectTransitionKey& other) const {
    if (anchor != other.anchor) return anchor < other.anchor;
    if (previous != other.previous) return previous < other.previous;
    return next < other.next;
  }

  __host__ __device__ bool operator==(
      const ResidentSubjectTransitionKey& other) const {
    return anchor == other.anchor && previous == other.previous &&
        next == other.next;
  }
};

struct ReverseBigramEdge {
  std::uint32_t current;
  std::uint32_t previous;
  std::uint32_t support;

  __host__ __device__ bool operator<(const ReverseBigramEdge& other) const {
    if (current != other.current) return current < other.current;
    if (previous != other.previous) return previous < other.previous;
    return support > other.support;
  }
};

struct ReverseTrigramEdge {
  std::uint32_t second;
  std::uint32_t next;
  std::uint32_t previous;
  std::uint32_t support;

  __host__ __device__ bool operator<(const ReverseTrigramEdge& other) const {
    if (second != other.second) return second < other.second;
    if (next != other.next) return next < other.next;
    if (previous != other.previous) return previous < other.previous;
    return support > other.support;
  }
};

struct ResidentSynthesisResult {
  std::uint32_t ready;
  std::uint32_t winner;
  std::uint32_t unit_count;
  std::uint32_t cue_coverage;
  std::uint32_t closed;
  std::uint32_t conditioned;
  std::uint32_t score_low;
  std::uint32_t score_high;
  std::uint32_t relation_subject;
  std::uint32_t relation_predicate;
  std::uint32_t relation_value;
  std::uint32_t relation_tail_count;
  std::uint32_t relation_tail[kResidentSynthesisMaxRelationTail];
};

// All pointers are device pointers.  `cue_activation`, `forward_activation`,
// and `backward_activation` are mutable resident fields produced by cue contact
// and recurrent association propagation.  Any of the three may be null.
template <typename BigramKeyT, typename TrigramKeyT>
struct ResidentSynthesisModelView {
  const std::uint32_t* unit_lengths = nullptr;
  const std::uint32_t* unit_content = nullptr;
  const std::uint32_t* unit_vitality = nullptr;
  std::uint32_t unit_count = 0u;
  std::uint32_t unit_words = 0u;

  const std::uint32_t* boundary_bytes = nullptr;
  std::uint32_t boundary_count = 0u;
  const std::uint32_t* closure_bytes = nullptr;
  std::uint32_t closure_count = 0u;

  const unsigned long long* cue_activation = nullptr;
  const unsigned long long* forward_activation = nullptr;
  const unsigned long long* backward_activation = nullptr;
  const std::uint32_t* cue_masks = nullptr;
  const std::uint32_t* cue_orders = nullptr;
  const std::uint32_t* salient_masks = nullptr;
  const std::uint32_t* forward_support = nullptr;
  const std::uint32_t* backward_support = nullptr;
  const ResidentSynthesisPolicyState* policy_state = nullptr;

  const ResidentSourceWindowSignature* source_windows = nullptr;
  std::uint32_t source_window_count = 0u;

  const substrate::bcc32::resident_roles::MutableStructuralRole* unit_roles = nullptr;
  std::uint32_t role_count = 0u;
  const std::uint64_t* base_role_bigrams = nullptr;
  const std::uint64_t* base_role_trigrams = nullptr;
  const std::uint64_t* online_role_bigrams = nullptr;
  const std::uint64_t* online_role_trigrams = nullptr;

  const BigramKeyT* base_bigrams = nullptr;
  const std::uint32_t* base_bigram_counts = nullptr;
  std::uint32_t base_bigram_count = 0u;
  const TrigramKeyT* base_trigrams = nullptr;
  const std::uint32_t* base_trigram_counts = nullptr;
  std::uint32_t base_trigram_count = 0u;

  const BigramKeyT* online_bigrams = nullptr;
  const std::uint32_t* online_bigram_counts = nullptr;
  std::uint32_t online_bigram_count = 0u;
  const TrigramKeyT* online_trigrams = nullptr;
  const std::uint32_t* online_trigram_counts = nullptr;
  std::uint32_t online_trigram_count = 0u;

  const ResidentSubjectTransitionKey* subject_transitions = nullptr;
  const std::uint32_t* subject_transition_counts = nullptr;
  std::uint32_t subject_transition_count = 0u;
  const std::uint32_t* subject_anchors = nullptr;
  std::uint32_t subject_anchor_count = 0u;
  const std::uint32_t* cue_units = nullptr;
  std::uint32_t cue_unit_count = 0u;
  const std::uint32_t* episode_units = nullptr;
  std::uint32_t episode_count = 0u;
  const std::uint32_t* episode_breaks = nullptr;
  std::uint32_t episode_break_count = 0u;
};

struct ResidentSynthesisWorkspaceView {
  ReverseBigramEdge* reverse_bigrams = nullptr;
  std::size_t reverse_bigram_capacity = 0u;
  ReverseTrigramEdge* reverse_trigrams = nullptr;
  std::size_t reverse_trigram_capacity = 0u;

  std::uint32_t* seed_units = nullptr;
  unsigned long long* seed_scores = nullptr;
  std::uint32_t* drafts = nullptr;
  std::uint32_t* draft_lengths = nullptr;
  unsigned long long* draft_scores = nullptr;
  std::uint32_t* selected_units = nullptr;
  std::size_t selected_capacity = 0u;
  ResidentSynthesisResult* result = nullptr;
};

__host__ __device__ inline std::uint32_t synthesis_min(std::uint32_t a,
                                                       std::uint32_t b) {
  return a < b ? a : b;
}

__host__ __device__ inline std::uint32_t synthesis_mix32(std::uint32_t value) {
  value ^= value >> 16u;
  value *= 0x7feb352du;
  value ^= value >> 15u;
  value *= 0x846ca68bu;
  return value ^ (value >> 16u);
}

__host__ __device__ inline ResidentSourceWindowSignature
synthesis_source_window_signature(const std::uint32_t* units) {
  std::uint32_t a = 0x811c9dc5u;
  std::uint32_t b = 0x9e3779b9u;
  for (std::uint32_t offset = 0u; offset < kResidentSynthesisNoveltyUnits;
       ++offset) {
    a = synthesis_mix32(a ^ units[offset] ^ (offset + 1u));
    b = synthesis_mix32(b + units[offset] * 0x85ebca6bu + offset);
  }
  return {a, b};
}

#include "bcc32_cuda_resident_synthesis_scoring.inl"

template <typename BigramKeyT>
__global__ void materialize_reverse_bigrams_kernel(
    const BigramKeyT* keys, const std::uint32_t* counts, std::uint32_t count,
    ReverseBigramEdge* output, std::uint32_t output_offset) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count) return;
  output[output_offset + i] = {keys[i].next, keys[i].previous, counts[i]};
}

template <typename TrigramKeyT>
__global__ void materialize_reverse_trigrams_kernel(
    const TrigramKeyT* keys, const std::uint32_t* counts, std::uint32_t count,
    ReverseTrigramEdge* output, std::uint32_t output_offset) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count) return;
  output[output_offset + i] = {
      keys[i].second, keys[i].next, keys[i].first, counts[i]};
}

template <typename BigramKeyT, typename TrigramKeyT>
__device__ inline unsigned long long synthesis_activation(
    const ResidentSynthesisModelView<BigramKeyT, TrigramKeyT>& model,
    std::uint32_t unit) {
  unsigned long long activation = 0ull;
  if (model.cue_activation != nullptr) {
    activation = synthesis_sat_add(activation, model.cue_activation[unit]);
  }
  if (model.forward_activation != nullptr) {
    activation = synthesis_sat_add(activation, model.forward_activation[unit]);
  }
  if (model.backward_activation != nullptr) {
    activation = synthesis_sat_add(activation, model.backward_activation[unit]);
  }
  return activation;
}

template <typename BigramKeyT, typename TrigramKeyT>
__device__ inline unsigned long long synthesis_directional_activation(
    const ResidentSynthesisModelView<BigramKeyT, TrigramKeyT>& model,
    std::uint32_t unit, bool forward) {
  const unsigned long long* directional =
      forward ? model.forward_activation : model.backward_activation;
  return directional == nullptr ? 0ull : directional[unit];
}

template <typename BigramKeyT, typename TrigramKeyT>
__device__ inline unsigned long long synthesis_semantic_score(
    const ResidentSynthesisModelView<BigramKeyT, TrigramKeyT>& model,
    std::uint32_t unit) {
  const unsigned long long activation = synthesis_activation(model, unit);
  const std::uint32_t cue_mask =
      model.cue_masks == nullptr ? 0u : model.cue_masks[unit];
  const std::uint32_t salient_mask =
      model.salient_masks == nullptr ? 0u : model.salient_masks[unit];
  const std::uint32_t directional_mask =
      (model.forward_support == nullptr ? 0u : model.forward_support[unit]) |
      (model.backward_support == nullptr ? 0u : model.backward_support[unit]);
  if (activation == 0ull && cue_mask == 0u && salient_mask == 0u &&
      directional_mask == 0u) {
    return 0ull;
  }
  const std::uint32_t vitality =
      model.unit_vitality == nullptr ? 0u : model.unit_vitality[unit];
  unsigned long long score = activation;
  score = synthesis_sat_add(score,
      static_cast<unsigned long long>(__popc(cue_mask)) << 32u);
  score = synthesis_sat_add(score,
      static_cast<unsigned long long>(__popc(salient_mask)) << 36u);
  score = synthesis_sat_add(score,
      static_cast<unsigned long long>(__popc(directional_mask)) << 40u);
  score = synthesis_sat_add(score, synthesis_depth(vitality));
  return score;
}

template <typename BigramKeyT, typename TrigramKeyT>
__global__ void select_semantic_seeds_kernel(
    ResidentSynthesisModelView<BigramKeyT, TrigramKeyT> model,
    std::uint32_t seed_count, std::uint32_t* seed_units,
    unsigned long long* seed_scores) {
  __shared__ unsigned long long block_scores[kResidentSynthesisBlock];
  __shared__ std::uint32_t block_units[kResidentSynthesisBlock];
  __shared__ std::uint32_t reserved_count;
  if (threadIdx.x == 0u) {
    reserved_count = 0u;
    for (std::uint32_t anchor_index = 0u;
         anchor_index < model.subject_anchor_count &&
         reserved_count < seed_count;
         ++anchor_index) {
      const std::uint32_t unit = model.subject_anchors == nullptr
          ? kResidentSynthesisInvalid : model.subject_anchors[anchor_index];
      if (unit >= model.unit_count) continue;
      bool duplicate = false;
      for (std::uint32_t prior = 0u; prior < reserved_count; ++prior) {
        duplicate |= seed_units[prior] == unit;
      }
      if (duplicate) continue;
      const std::uint32_t begin = lower_subject_transition(
          model.subject_transitions, model.subject_transition_count, unit, 0u);
      const std::uint32_t end = upper_subject_transition(
          model.subject_transitions, model.subject_transition_count,
          unit, 0xffffffffu);
      if (begin == end) continue;
      seed_units[reserved_count] = unit;
      const unsigned long long score = synthesis_semantic_score(model, unit);
      seed_scores[reserved_count] = score == 0ull ? 1ull : score;
      ++reserved_count;
    }
  }
  __syncthreads();
  for (std::uint32_t rank = reserved_count; rank < seed_count; ++rank) {
    unsigned long long best_score = 0ull;
    std::uint32_t best_unit = kResidentSynthesisInvalid;
    for (std::uint32_t unit = threadIdx.x; unit < model.unit_count;
         unit += blockDim.x) {
      const bool direct = model.cue_masks != nullptr && model.cue_masks[unit] != 0u;
      const bool require_direct = rank < seed_count / 2u;
      if (direct != require_direct) continue;
      if (model.online_bigram_count != 0u) {
        const std::uint32_t begin = lower_forward_bigram(
            model.online_bigrams, model.online_bigram_count, unit);
        const std::uint32_t end = upper_forward_bigram(
            model.online_bigrams, model.online_bigram_count, unit);
        if (begin == end) continue;
      }
      bool selected = false;
      for (std::uint32_t prior = 0u; prior < rank; ++prior) {
        selected |= seed_units[prior] == unit;
      }
      if (selected) continue;
      const unsigned long long score = synthesis_semantic_score(model, unit);
      if (score > best_score ||
          (score == best_score && score != 0ull && unit < best_unit)) {
        best_score = score;
        best_unit = unit;
      }
    }
    block_scores[threadIdx.x] = best_score;
    block_units[threadIdx.x] = best_unit;
    __syncthreads();
    for (std::uint32_t stride = blockDim.x / 2u; stride != 0u; stride >>= 1u) {
      if (threadIdx.x < stride) {
        const unsigned long long other_score = block_scores[threadIdx.x + stride];
        const std::uint32_t other_unit = block_units[threadIdx.x + stride];
        if (other_score > block_scores[threadIdx.x] ||
            (other_score == block_scores[threadIdx.x] && other_score != 0ull &&
             other_unit < block_units[threadIdx.x])) {
          block_scores[threadIdx.x] = other_score;
          block_units[threadIdx.x] = other_unit;
        }
      }
      __syncthreads();
    }
    if (threadIdx.x == 0u) {
      seed_units[rank] = block_scores[0] == 0ull
          ? kResidentSynthesisInvalid : block_units[0];
      seed_scores[rank] = block_scores[0];
    }
    __syncthreads();
  }
}

__device__ inline std::uint32_t synthesis_sample_index(
    std::uint32_t begin, std::uint32_t end, std::uint32_t sample,
    std::uint32_t sample_count) {
  const std::uint32_t span = end - begin;
  return span <= sample_count ? begin + sample
      : begin + static_cast<std::uint32_t>(
          (static_cast<unsigned long long>(sample) * span) / sample_count);
}

template <typename BigramKeyT, typename TrigramKeyT>
__device__ inline unsigned long long synthesis_forward_fit(
    const ResidentSynthesisModelView<BigramKeyT, TrigramKeyT>& model,
    std::uint32_t first, std::uint32_t previous, std::uint32_t candidate,
    std::uint32_t trigram_weight) {
  const std::uint32_t online_bi = synthesis_bigram_count(
      model.online_bigrams, model.online_bigram_counts,
      model.online_bigram_count, previous, candidate);
  const std::uint32_t online_tri = first == kResidentSynthesisInvalid ? 0u
      : synthesis_trigram_count(
          model.online_trigrams, model.online_trigram_counts,
          model.online_trigram_count, first, previous, candidate);
  const unsigned long long online_support =
      static_cast<unsigned long long>(online_bi) +
      static_cast<unsigned long long>(trigram_weight) * online_tri;
  const unsigned long long role_online_support = synthesis_role_support(
      model, model.online_role_bigrams, model.online_role_trigrams,
      first, previous, candidate);
  if ((model.online_bigram_count != 0u || model.online_trigram_count != 0u) &&
      online_support == 0ull && role_online_support == 0ull) {
    return 0ull;
  }
  const unsigned long long relevance =
      synthesis_depth(synthesis_directional_activation(model, candidate, true)) +
      (model.forward_support == nullptr ? 0u :
          4u * __popc(model.forward_support[candidate]));
  return synthesis_fit_priority_score(relevance, online_support);
}

template <typename BigramKeyT, typename TrigramKeyT>
__device__ inline unsigned long long synthesis_backward_fit(
    const ResidentSynthesisModelView<BigramKeyT, TrigramKeyT>& model,
    std::uint32_t candidate, std::uint32_t current, std::uint32_t next,
    std::uint32_t trigram_weight) {
  const std::uint32_t online_bi = synthesis_bigram_count(
      model.online_bigrams, model.online_bigram_counts,
      model.online_bigram_count, candidate, current);
  const std::uint32_t online_tri = next == kResidentSynthesisInvalid ? 0u
      : synthesis_trigram_count(
          model.online_trigrams, model.online_trigram_counts,
          model.online_trigram_count, candidate, current, next);
  const unsigned long long online_support =
      static_cast<unsigned long long>(online_bi) +
      static_cast<unsigned long long>(trigram_weight) * online_tri;
  const unsigned long long role_online_support = synthesis_role_support(
      model, model.online_role_bigrams, model.online_role_trigrams,
      candidate, current, next);
  if ((model.online_bigram_count != 0u || model.online_trigram_count != 0u) &&
      online_support == 0ull && role_online_support == 0ull) {
    return 0ull;
  }
  const unsigned long long relevance =
      synthesis_depth(synthesis_directional_activation(model, candidate, false)) +
      (model.backward_support == nullptr ? 0u :
          4u * __popc(model.backward_support[candidate]));
  return synthesis_fit_priority_score(relevance, online_support);
}

template <typename BigramKeyT, typename TrigramKeyT>
__device__ inline void synthesis_consider_ranked(
    std::uint32_t unit, unsigned long long score, std::uint32_t* best_unit,
    unsigned long long* best_score, std::uint32_t* alternate_unit,
    unsigned long long* alternate_score) {
  if (unit == *best_unit) {
    if (score > *best_score) *best_score = score;
    return;
  }
  if (unit == *alternate_unit) {
    if (score > *alternate_score) *alternate_score = score;
    return;
  }
  if (score > *best_score ||
      (score == *best_score && unit < *best_unit)) {
    *alternate_unit = *best_unit;
    *alternate_score = *best_score;
    *best_unit = unit;
    *best_score = score;
  } else if (score > *alternate_score ||
             (score == *alternate_score && unit < *alternate_unit)) {
    *alternate_unit = unit;
    *alternate_score = score;
  }
}

template <typename BigramKeyT, typename TrigramKeyT>
__device__ inline void consider_forward_candidate(
    const ResidentSynthesisModelView<BigramKeyT, TrigramKeyT>& model,
    std::uint32_t first, std::uint32_t previous, std::uint32_t candidate,
    std::uint32_t support, std::uint32_t* best_unit,
    unsigned long long* best_score, std::uint32_t* alternate_unit,
    unsigned long long* alternate_score, std::uint32_t trigram_weight,
    bool allow_base_only) {
  if (candidate >= model.unit_count || support == 0u) return;
  unsigned long long score = synthesis_forward_fit(
      model, first, previous, candidate, trigram_weight);
  if (score == 0ull && allow_base_only) {
    const unsigned long long relevance =
        synthesis_depth(synthesis_directional_activation(model, candidate, true)) +
        (model.forward_support == nullptr ? 0u :
            4u * __popc(model.forward_support[candidate]));
    score = synthesis_fit_priority_score(relevance, support);
  }
  if (score == 0ull) return;
  score = synthesis_sat_add(score, support);
  synthesis_consider_ranked<BigramKeyT, TrigramKeyT>(
      candidate, score, best_unit, best_score, alternate_unit, alternate_score);
}

template <typename BigramKeyT, typename TrigramKeyT>
__device__ inline std::uint32_t choose_forward_unit(
    const ResidentSynthesisModelView<BigramKeyT, TrigramKeyT>& model,
    std::uint32_t first, std::uint32_t previous, std::uint32_t scan_limit,
    std::uint32_t variant, std::uint32_t subject,
    std::uint32_t trigram_weight, bool allow_base_only = false) {
  std::uint32_t best = kResidentSynthesisInvalid;
  std::uint32_t alternate = kResidentSynthesisInvalid;
  unsigned long long best_score = 0ull;
  unsigned long long alternate_score = 0ull;
  if (subject != kResidentSynthesisInvalid &&
      synthesis_subject_is_selected(model, subject) &&
      model.subject_transitions != nullptr &&
      model.subject_transition_counts != nullptr) {
    const std::uint32_t begin = lower_subject_transition(
        model.subject_transitions, model.subject_transition_count,
        subject, previous);
    const std::uint32_t end = upper_subject_transition(
        model.subject_transitions, model.subject_transition_count,
        subject, previous);
    const std::uint32_t samples = synthesis_min(end - begin, scan_limit);
    for (std::uint32_t sample = 0u; sample < samples; ++sample) {
      const std::uint32_t edge =
          synthesis_sample_index(begin, end, sample, samples);
      consider_forward_candidate(
          model, first, previous, model.subject_transitions[edge].next,
          model.subject_transition_counts[edge], &best, &best_score,
          &alternate, &alternate_score, trigram_weight, allow_base_only);
    }
  }
  if (best == kResidentSynthesisInvalid && first != kResidentSynthesisInvalid) {
    const TrigramKeyT* tri_keys[2] = {
        model.base_trigrams, model.online_trigrams};
    const std::uint32_t* tri_counts[2] = {
        model.base_trigram_counts, model.online_trigram_counts};
    const std::uint32_t tri_sizes[2] = {
        model.base_trigram_count, model.online_trigram_count};
    for (std::uint32_t source = 0u; source < 2u; ++source) {
      if (tri_keys[source] == nullptr || tri_counts[source] == nullptr) continue;
      const std::uint32_t begin = lower_forward_trigram(
          tri_keys[source], tri_sizes[source], first, previous);
      const std::uint32_t end = upper_forward_trigram(
          tri_keys[source], tri_sizes[source], first, previous);
      const std::uint32_t samples = synthesis_min(end - begin, scan_limit);
      for (std::uint32_t sample = 0u; sample < samples; ++sample) {
        const std::uint32_t edge =
            synthesis_sample_index(begin, end, sample, samples);
        consider_forward_candidate(model, first, previous,
                                   tri_keys[source][edge].next,
                                   tri_counts[source][edge], &best, &best_score,
                                   &alternate, &alternate_score, trigram_weight,
                                   allow_base_only);
      }
    }
  }
  if (best == kResidentSynthesisInvalid) {
    const BigramKeyT* bi_keys[2] = {model.base_bigrams, model.online_bigrams};
    const std::uint32_t* bi_counts[2] = {
        model.base_bigram_counts, model.online_bigram_counts};
    const std::uint32_t bi_sizes[2] = {
        model.base_bigram_count, model.online_bigram_count};
    for (std::uint32_t source = 0u; source < 2u; ++source) {
      if (bi_keys[source] == nullptr || bi_counts[source] == nullptr) continue;
      const std::uint32_t begin =
          lower_forward_bigram(bi_keys[source], bi_sizes[source], previous);
      const std::uint32_t end =
          upper_forward_bigram(bi_keys[source], bi_sizes[source], previous);
      const std::uint32_t samples = synthesis_min(end - begin, scan_limit);
      for (std::uint32_t sample = 0u; sample < samples; ++sample) {
        const std::uint32_t edge =
            synthesis_sample_index(begin, end, sample, samples);
        consider_forward_candidate(
            model, first, previous, bi_keys[source][edge].next,
            bi_counts[source][edge], &best, &best_score, &alternate,
            &alternate_score, trigram_weight, allow_base_only);
      }
    }
  }
  return (variant & 1u) != 0u && alternate != kResidentSynthesisInvalid
      ? alternate : best;
}

template <typename BigramKeyT, typename TrigramKeyT>
__device__ inline std::uint32_t choose_forward_base_escape(
    const ResidentSynthesisModelView<BigramKeyT, TrigramKeyT>& model,
    std::uint32_t previous, std::uint32_t forbidden,
    std::uint32_t scan_limit, std::uint32_t variant) {
  if (model.base_bigrams == nullptr || model.base_bigram_counts == nullptr) {
    return kResidentSynthesisInvalid;
  }
  std::uint32_t best = kResidentSynthesisInvalid;
  std::uint32_t alternate = kResidentSynthesisInvalid;
  unsigned long long best_score = 0ull;
  unsigned long long alternate_score = 0ull;
  const std::uint32_t begin =
      lower_forward_bigram(model.base_bigrams, model.base_bigram_count, previous);
  const std::uint32_t end =
      upper_forward_bigram(model.base_bigrams, model.base_bigram_count, previous);
  const std::uint32_t samples = synthesis_min(end - begin, scan_limit);
  for (std::uint32_t sample = 0u; sample < samples; ++sample) {
    const std::uint32_t edge =
        synthesis_sample_index(begin, end, sample, samples);
    const std::uint32_t candidate = model.base_bigrams[edge].next;
    const std::uint32_t support = model.base_bigram_counts[edge];
    if (candidate == forbidden || candidate >= model.unit_count || support == 0u) {
      continue;
    }
    const unsigned long long relevance =
        synthesis_depth(synthesis_directional_activation(model, candidate, true)) +
        (model.forward_support == nullptr ? 0u :
            4u * __popc(model.forward_support[candidate]));
    const unsigned long long score = synthesis_fit_priority_score(relevance, support);
    synthesis_consider_ranked<BigramKeyT, TrigramKeyT>(
        candidate, score, &best, &best_score, &alternate, &alternate_score);
  }
  return (variant & 1u) != 0u && alternate != kResidentSynthesisInvalid
      ? alternate : best;
}

template <typename BigramKeyT, typename TrigramKeyT>
__device__ inline std::uint32_t choose_backward_unit(
    const ResidentSynthesisModelView<BigramKeyT, TrigramKeyT>& model,
    const ReverseBigramEdge* reverse_bigrams, std::uint32_t reverse_bigram_count,
    const ReverseTrigramEdge* reverse_trigrams, std::uint32_t reverse_trigram_count,
    std::uint32_t current, std::uint32_t next, std::uint32_t scan_limit,
    std::uint32_t variant, std::uint32_t trigram_weight) {
  std::uint32_t best = kResidentSynthesisInvalid;
  std::uint32_t alternate = kResidentSynthesisInvalid;
  unsigned long long best_score = 0ull;
  unsigned long long alternate_score = 0ull;
  if (next != kResidentSynthesisInvalid) {
    std::uint32_t begin = lower_reverse_trigram(
        reverse_trigrams, reverse_trigram_count, current, next);
    const std::uint32_t end = upper_reverse_trigram(
        reverse_trigrams, reverse_trigram_count, current, next);
    const std::uint32_t samples = synthesis_min(end - begin, scan_limit);
    for (std::uint32_t sample = 0u; sample < samples; ++sample) {
      const ReverseTrigramEdge edge = reverse_trigrams[
          synthesis_sample_index(begin, end, sample, samples)];
      if (edge.support == 0u || edge.previous >= model.unit_count) continue;
      const unsigned long long fit =
          synthesis_backward_fit(model, edge.previous, current, next, trigram_weight);
      if (fit == 0ull) continue;
      const unsigned long long score = synthesis_sat_add(
          fit, static_cast<unsigned long long>(edge.support) * 4ull);
      synthesis_consider_ranked<BigramKeyT, TrigramKeyT>(
          edge.previous, score, &best, &best_score, &alternate,
          &alternate_score);
    }
  }
  if (best == kResidentSynthesisInvalid) {
    const std::uint32_t begin = lower_reverse_bigram(
        reverse_bigrams, reverse_bigram_count, current);
    const std::uint32_t end = upper_reverse_bigram(
        reverse_bigrams, reverse_bigram_count, current);
    const std::uint32_t samples = synthesis_min(end - begin, scan_limit);
    for (std::uint32_t sample = 0u; sample < samples; ++sample) {
      const ReverseBigramEdge edge = reverse_bigrams[
          synthesis_sample_index(begin, end, sample, samples)];
      if (edge.support == 0u || edge.previous >= model.unit_count) continue;
      const unsigned long long fit =
          synthesis_backward_fit(model, edge.previous, current, next, trigram_weight);
      if (fit == 0ull) continue;
      const unsigned long long score = synthesis_sat_add(fit, edge.support);
      synthesis_consider_ranked<BigramKeyT, TrigramKeyT>(
          edge.previous, score, &best, &best_score, &alternate,
          &alternate_score);
    }
  }
  return (variant & 1u) != 0u && alternate != kResidentSynthesisInvalid
      ? alternate : best;
}

__device__ inline bool synthesis_forward_transition_seen(
    const std::uint32_t* draft, std::uint32_t leftmost,
    std::uint32_t rightmost, std::uint32_t candidate) {
  if (rightmost <= leftmost) return false;
  const std::uint32_t previous = draft[rightmost - 1u];
  for (std::uint32_t i = leftmost + 1u; i < rightmost; ++i) {
    if (draft[i - 1u] == previous && draft[i] == candidate) return true;
  }
  if (rightmost < leftmost + 2u) return false;
  const std::uint32_t first = draft[rightmost - 2u];
  for (std::uint32_t i = leftmost + 2u; i < rightmost; ++i) {
    if (draft[i - 2u] == first && draft[i - 1u] == previous &&
        draft[i] == candidate) {
      return true;
    }
  }
  return false;
}

__device__ inline bool synthesis_backward_transition_seen(
    const std::uint32_t* draft, std::uint32_t leftmost,
    std::uint32_t rightmost, std::uint32_t candidate) {
  if (rightmost <= leftmost) return false;
  const std::uint32_t current = draft[leftmost];
  for (std::uint32_t i = leftmost + 1u; i < rightmost; ++i) {
    if (draft[i - 1u] == candidate && draft[i] == current) return true;
  }
  if (rightmost < leftmost + 2u) return false;
  const std::uint32_t next = draft[leftmost + 1u];
  for (std::uint32_t i = leftmost + 2u; i < rightmost; ++i) {
    if (draft[i - 2u] == candidate && draft[i - 1u] == current &&
        draft[i] == next) {
      return true;
    }
  }
  return false;
}

template <typename BigramKeyT, typename TrigramKeyT>
__device__ inline bool synthesis_forward_source_window_present(
    const ResidentSynthesisModelView<BigramKeyT, TrigramKeyT>& model,
    const std::uint32_t* draft, std::uint32_t leftmost,
    std::uint32_t rightmost, std::uint32_t candidate) {
  if (model.source_windows == nullptr ||
      rightmost - leftmost + 1u < kResidentSynthesisNoveltyUnits) {
    return false;
  }
  std::uint32_t window[kResidentSynthesisNoveltyUnits];
  const std::uint32_t retained = kResidentSynthesisNoveltyUnits - 1u;
  for (std::uint32_t offset = 0u; offset < retained; ++offset) {
    window[offset] = draft[rightmost - retained + offset];
  }
  window[retained] = candidate;
  return synthesis_source_window_present(
      model.source_windows, model.source_window_count,
      synthesis_source_window_signature(window));
}

template <typename BigramKeyT, typename TrigramKeyT>
__device__ inline bool synthesis_backward_source_window_present(
    const ResidentSynthesisModelView<BigramKeyT, TrigramKeyT>& model,
    const std::uint32_t* draft, std::uint32_t leftmost,
    std::uint32_t rightmost, std::uint32_t candidate) {
  if (model.source_windows == nullptr ||
      rightmost - leftmost + 1u < kResidentSynthesisNoveltyUnits) {
    return false;
  }
  std::uint32_t window[kResidentSynthesisNoveltyUnits];
  window[0] = candidate;
  for (std::uint32_t offset = 1u;
       offset < kResidentSynthesisNoveltyUnits; ++offset) {
    window[offset] = draft[leftmost + offset - 1u];
  }
  return synthesis_source_window_present(
      model.source_windows, model.source_window_count,
      synthesis_source_window_signature(window));
}

template <typename BigramKeyT, typename TrigramKeyT>
__device__ inline std::uint32_t synthesis_conditioned_relation_recency(
    const ResidentSynthesisModelView<BigramKeyT, TrigramKeyT>& model,
    std::uint32_t subject, std::uint32_t previous, std::uint32_t next) {
  if (model.episode_units == nullptr || model.episode_count < 2u) return 0u;
  const std::uint32_t window_begin = model.episode_count > kResidentSynthesisRecentUnits
      ? model.episode_count - kResidentSynthesisRecentUnits : 0u;
  for (std::uint32_t position = model.episode_count - 1u;
       position > window_begin; --position) {
    if (model.episode_units[position] != next ||
        model.episode_units[position - 1u] != previous) {
      continue;
    }
    std::uint32_t episode_begin = window_begin;
    if (model.episode_breaks != nullptr && model.episode_break_count != 0u) {
      std::uint32_t lo = 0u;
      std::uint32_t hi = model.episode_break_count;
      while (lo < hi) {
        const std::uint32_t middle = lo + (hi - lo) / 2u;
        if (model.episode_breaks[middle] <= position) lo = middle + 1u;
        else hi = middle;
      }
      const std::uint32_t resident_begin = lo == 0u ? 0u : model.episode_breaks[lo - 1u];
      episode_begin = window_begin > resident_begin ? window_begin : resident_begin;
    }
    const std::uint32_t previous_position = position - 1u;
    for (std::uint32_t lag = 1u;
         lag <= kResidentSynthesisSubjectLag && previous_position >= episode_begin + lag;
         ++lag) {
      if (model.episode_units[previous_position - lag] == subject) {
        return position + 1u;
      }
    }
  }
  return 0u;
}

template <typename BigramKeyT, typename TrigramKeyT>
__device__ inline std::uint32_t synthesis_conditioned_relation_value(
    const ResidentSynthesisModelView<BigramKeyT, TrigramKeyT>& model,
    std::uint32_t subject, std::uint32_t scan_limit,
    std::uint32_t* selected_previous = nullptr) {
  if (model.subject_transitions == nullptr ||
      model.subject_transition_counts == nullptr) {
    return kResidentSynthesisInvalid;
  }
  const std::uint32_t begin = lower_subject_transition(
      model.subject_transitions, model.subject_transition_count, subject, 0u);
  const std::uint32_t end = upper_subject_transition(
      model.subject_transitions, model.subject_transition_count,
      subject, 0xffffffffu);
  const std::uint32_t samples = synthesis_min(end - begin, scan_limit);
  std::uint32_t best = kResidentSynthesisInvalid;
  std::uint32_t best_previous = kResidentSynthesisInvalid;
  std::uint32_t best_distance = 0xffffffffu;
  std::uint32_t best_recency = 0u;
  unsigned long long best_score = 0ull;
  std::uint32_t subject_order = model.cue_orders == nullptr
      ? 0xffffffffu : model.cue_orders[subject];
  if (model.cue_units != nullptr) {
    for (std::uint32_t cue = 0u; cue < model.cue_unit_count; ++cue) {
      if (model.cue_units[cue] == subject) {
        subject_order = cue + 1u;
        break;
      }
    }
  }
  for (std::uint32_t sample = 0u; sample < samples; ++sample) {
    const std::uint32_t edge =
        synthesis_sample_index(begin, end, sample, samples);
    const ResidentSubjectTransitionKey key = model.subject_transitions[edge];
    if (key.previous >= model.unit_count || key.next >= model.unit_count ||
        model.subject_transition_counts[edge] == 0u) {
      continue;
    }
    const std::uint32_t next_cue =
        model.cue_masks == nullptr ? 0u : model.cue_masks[key.next];
    if (next_cue != 0u) continue;
    const std::uint32_t previous_cue =
        model.cue_masks == nullptr ? 0u : model.cue_masks[key.previous];
    const unsigned long long activation =
        synthesis_directional_activation(model, key.previous, true);
    std::uint32_t raw_similarity = 0u;
    std::uint32_t predicate_order = previous_cue != 0u &&
            model.cue_orders != nullptr
        ? model.cue_orders[key.previous] : 0xffffffffu;
    if (model.cue_units != nullptr && model.unit_lengths != nullptr &&
        model.unit_content != nullptr) {
      for (std::uint32_t anchor_index = 0u;
           anchor_index < model.cue_unit_count; ++anchor_index) {
        const std::uint32_t anchor = model.cue_units[anchor_index];
        if (anchor >= model.unit_count || anchor == subject) continue;
        const std::uint32_t extent = synthesis_min(
            model.unit_lengths[key.previous], model.unit_lengths[anchor]);
        std::uint32_t prefix = 0u;
        while (prefix < extent &&
               synthesis_unit_byte(model.unit_content, model.unit_words,
                                   key.previous, prefix) ==
                   synthesis_unit_byte(model.unit_content, model.unit_words,
                                       anchor, prefix)) {
          ++prefix;
        }
        const std::uint32_t anchor_order = anchor_index + 1u;
        const std::uint32_t prior_distance =
            predicate_order == 0xffffffffu || subject_order == 0xffffffffu
                ? 0xffffffffu
                : (predicate_order > subject_order
                       ? predicate_order - subject_order
                       : subject_order - predicate_order);
        const std::uint32_t candidate_distance =
            anchor_order == 0xffffffffu || subject_order == 0xffffffffu
                ? 0xffffffffu
                : (anchor_order > subject_order
                       ? anchor_order - subject_order
                       : subject_order - anchor_order);
        if (prefix > raw_similarity ||
            (prefix == raw_similarity && candidate_distance < prior_distance)) {
          raw_similarity = prefix;
          predicate_order = anchor_order;
        }
      }
    }
    if ((previous_cue == 0u && raw_similarity < 2u) ||
        key.previous == subject || predicate_order == subject_order) {
      continue;
    }
    const std::uint32_t predicate_distance =
        predicate_order == 0xffffffffu || subject_order == 0xffffffffu
            ? 0xffffffffu
            : (predicate_order > subject_order
                   ? predicate_order - subject_order
                   : subject_order - predicate_order);
    const std::uint32_t recency = synthesis_conditioned_relation_recency(
        model, subject, key.previous, key.next);
    const unsigned long long vitality =
        model.unit_vitality == nullptr ? 0ull : model.unit_vitality[key.next];
    const std::uint32_t information =
        33u - synthesis_min(32u, synthesis_depth(vitality + 1ull));
    const std::uint32_t relevance =
        synthesis_min(16u, synthesis_depth(activation)) +
        8u * __popc(previous_cue) + 32u * raw_similarity;
    const unsigned long long quality =
        static_cast<unsigned long long>(relevance) * information * information;
    const unsigned long long score =
        (quality << 32u) |
        synthesis_depth(model.subject_transition_counts[edge]);
    bool better = predicate_distance < best_distance;
    if (predicate_distance == best_distance) {
      if (key.previous == best_previous) {
        better = recency > best_recency ||
            (recency == best_recency &&
             (score > best_score ||
              (score == best_score && score != 0ull && key.next < best)));
      } else {
        better = score > best_score ||
            (score == best_score && score != 0ull && key.next < best);
      }
    }
    if (better) {
      best = key.next;
      best_previous = key.previous;
      best_distance = predicate_distance;
      best_recency = recency;
      best_score = score;
    }
  }
  if (selected_previous != nullptr) *selected_previous = best_previous;
  return best;
}

template <typename BigramKeyT, typename TrigramKeyT>
__device__ inline std::uint32_t synthesis_conditioned_relation_tail(
    const ResidentSynthesisModelView<BigramKeyT, TrigramKeyT>& model,
    std::uint32_t subject, std::uint32_t previous, std::uint32_t scan_limit,
    std::uint32_t* tail, std::uint32_t tail_capacity) {
  if (tail == nullptr || tail_capacity == 0u ||
      model.subject_transitions == nullptr ||
      model.subject_transition_counts == nullptr) {
    return 0u;
  }
  std::uint32_t length = 0u;
  for (; length < tail_capacity; ++length) {
    const std::uint32_t begin = lower_subject_transition(
        model.subject_transitions, model.subject_transition_count,
        subject, previous);
    const std::uint32_t end = upper_subject_transition(
        model.subject_transitions, model.subject_transition_count,
        subject, previous);
    const std::uint32_t samples = synthesis_min(end - begin, scan_limit);
    std::uint32_t best = kResidentSynthesisInvalid;
    std::uint32_t best_recency = 0u;
    std::uint32_t best_count = 0u;
    for (std::uint32_t sample = 0u; sample < samples; ++sample) {
      const std::uint32_t edge =
          synthesis_sample_index(begin, end, sample, samples);
      const ResidentSubjectTransitionKey key = model.subject_transitions[edge];
      const std::uint32_t count = model.subject_transition_counts[edge];
      if (key.anchor != subject || key.previous != previous ||
          key.next >= model.unit_count || count == 0u ||
          key.next == subject || key.next == previous) {
        continue;
      }
      bool cycle = false;
      for (std::uint32_t prior = 0u; prior < length; ++prior) {
        cycle |= tail[prior] == key.next;
      }
      if (cycle) continue;
      const std::uint32_t recency = synthesis_conditioned_relation_recency(
          model, subject, key.previous, key.next);
      if (recency == 0u) continue;
      if (recency > best_recency ||
          (recency == best_recency &&
           (count > best_count ||
            (count == best_count && key.next < best)))) {
        best = key.next;
        best_recency = recency;
        best_count = count;
      }
    }
    if (best == kResidentSynthesisInvalid) break;
    tail[length] = best;
    previous = best;
    if (synthesis_unit_contains(
            model.unit_lengths, model.unit_content, model.unit_words, best,
            model.closure_bytes, model.closure_count)) {
      ++length;
      break;
    }
  }
  if (length == tail_capacity && length != 0u &&
      !synthesis_unit_contains(
          model.unit_lengths, model.unit_content, model.unit_words,
          tail[length - 1u], model.closure_bytes, model.closure_count)) {
    --length;
  }
  return length;
}

template <typename BigramKeyT, typename TrigramKeyT>
__device__ inline bool synthesis_recent_source_prefix_present(
    const ResidentSynthesisModelView<BigramKeyT, TrigramKeyT>& model,
    const std::uint32_t* draft, std::uint32_t length) {
  if (model.episode_units == nullptr || model.episode_count < 2u || length < 2u) {
    return false;
  }
  const std::uint32_t begin = model.episode_count > kResidentSynthesisRecentUnits
      ? model.episode_count - kResidentSynthesisRecentUnits : 0u;
  for (std::uint32_t position = begin; position + 1u < model.episode_count;
       ++position) {
    if (model.episode_units[position] == draft[0] &&
        model.episode_units[position + 1u] == draft[1]) {
      return true;
    }
  }
  return false;
}

template <typename BigramKeyT, typename TrigramKeyT>
__global__ void generate_bidirectional_drafts_kernel(
    ResidentSynthesisModelView<BigramKeyT, TrigramKeyT> model,
    const ReverseBigramEdge* reverse_bigrams, std::uint32_t reverse_bigram_count,
    const ReverseTrigramEdge* reverse_trigrams, std::uint32_t reverse_trigram_count,
    const std::uint32_t* seed_units, ResidentSynthesisConfig config,
    std::uint32_t* drafts, std::uint32_t* draft_lengths) {
  const std::uint32_t draft_id = blockIdx.x;
  if (draft_id >= config.candidate_count || threadIdx.x != 0u) return;
  std::uint32_t* draft = drafts + draft_id * kResidentSynthesisMaxUnits;
  for (std::uint32_t i = 0u; i < kResidentSynthesisMaxUnits; ++i) {
    draft[i] = kResidentSynthesisInvalid;
  }
  const std::uint32_t seed = seed_units[draft_id % config.seed_count];
  if (seed == kResidentSynthesisInvalid || seed >= model.unit_count) {
    draft_lengths[draft_id] = 0u;
    return;
  }

  const std::uint32_t variant = draft_id / config.seed_count;
  const std::uint32_t base_variant = variant % kResidentSynthesisBaseVariants;
  const bool novelty_branch = variant >= kResidentSynthesisBaseVariants;
  const std::uint32_t trigram_weight = synthesis_variant_trigram_weight(base_variant);
  const bool conditioned_subject =
      synthesis_subject_is_selected(model, seed);
  const std::uint32_t center =
      conditioned_subject ? 0u : config.max_units / 2u;
  const std::uint32_t left_limit = center;
  const std::uint32_t right_limit = config.max_units - center;
  const std::uint32_t forward_branch_step =
      base_variant >= 1u && base_variant <= 4u
          ? 1u << (base_variant - 1u) : kResidentSynthesisInvalid;
  const std::uint32_t backward_branch_step =
      base_variant >= 5u ? 1u << (base_variant - 5u)
                         : kResidentSynthesisInvalid;
  const std::uint32_t relation_value = conditioned_subject
      ? synthesis_conditioned_relation_value(
            model, seed, config.edge_scan_limit)
      : kResidentSynthesisInvalid;
  if (conditioned_subject) {
    if (relation_value == kResidentSynthesisInvalid) {
      draft_lengths[draft_id] = 0u;
      return;
    }
    draft[0] = relation_value;
    draft_lengths[draft_id] = 1u;
    return;
  }
  draft[center] = relation_value == kResidentSynthesisInvalid
      ? seed : relation_value;
  std::uint32_t leftmost = center;
  std::uint32_t rightmost = center + 1u;

  while (leftmost != 0u && center - leftmost < left_limit) {
    const std::uint32_t current = draft[leftmost];
    const std::uint32_t next = leftmost + 1u < rightmost
        ? draft[leftmost + 1u] : kResidentSynthesisInvalid;
    const std::uint32_t backward_step = center - leftmost + 1u;
    const std::uint32_t backward_branch =
        backward_step == backward_branch_step ? 1u : 0u;
    std::uint32_t previous = choose_backward_unit(
        model, reverse_bigrams, reverse_bigram_count,
        reverse_trigrams, reverse_trigram_count, current, next,
        config.edge_scan_limit, backward_branch, trigram_weight);
    if (previous != kResidentSynthesisInvalid &&
        (synthesis_backward_transition_seen(
             draft, leftmost, rightmost, previous) ||
         (novelty_branch && synthesis_backward_source_window_present(
             model, draft, leftmost, rightmost, previous)))) {
      const std::uint32_t alternate = choose_backward_unit(
          model, reverse_bigrams, reverse_bigram_count,
          reverse_trigrams, reverse_trigram_count, current, next,
          config.edge_scan_limit, backward_branch ^ 1u, trigram_weight);
      previous = alternate != previous &&
              !synthesis_backward_transition_seen(
                  draft, leftmost, rightmost, alternate) &&
              (!novelty_branch || !synthesis_backward_source_window_present(
                  model, draft, leftmost, rightmost, alternate))
          ? alternate : kResidentSynthesisInvalid;
    }
    if (previous == kResidentSynthesisInvalid) break;
    draft[--leftmost] = previous;
    if (synthesis_unit_is_boundary(
            model.unit_lengths, model.unit_content, model.unit_words, previous,
            model.boundary_bytes, model.boundary_count) ||
        synthesis_unit_contains(
            model.unit_lengths, model.unit_content, model.unit_words, previous,
            model.closure_bytes, model.closure_count)) {
      break;
    }
  }

  while (rightmost < config.max_units && rightmost - center <= right_limit) {
    const std::uint32_t previous = draft[rightmost - 1u];
    const std::uint32_t first = rightmost >= leftmost + 2u
        ? draft[rightmost - 2u] : kResidentSynthesisInvalid;
    const std::uint32_t forward_step = rightmost - center;
    const std::uint32_t branch =
        forward_step == forward_branch_step ? 1u : 0u;
    std::uint32_t next = choose_forward_unit(
        model, first, previous, config.edge_scan_limit, branch, seed,
        trigram_weight);
    const bool repeated_transition = next != kResidentSynthesisInvalid &&
        synthesis_forward_transition_seen(draft, leftmost, rightmost, next);
    const bool repeated_source = next != kResidentSynthesisInvalid &&
        novelty_branch && synthesis_forward_source_window_present(
            model, draft, leftmost, rightmost, next);
    if (repeated_transition || repeated_source) {
      const std::uint32_t alternate = repeated_source
          ? choose_forward_base_escape(
                model, previous, next, config.edge_scan_limit, branch ^ 1u)
          : choose_forward_unit(
                model, first, previous, config.edge_scan_limit, branch ^ 1u,
                seed, trigram_weight);
      next = alternate != next &&
              !synthesis_forward_transition_seen(
                  draft, leftmost, rightmost, alternate) &&
              (!novelty_branch || !synthesis_forward_source_window_present(
                  model, draft, leftmost, rightmost, alternate))
          ? alternate : kResidentSynthesisInvalid;
    }
    if (next == kResidentSynthesisInvalid) break;
    draft[rightmost++] = next;
    if (synthesis_unit_contains(
            model.unit_lengths, model.unit_content, model.unit_words, next,
            model.closure_bytes, model.closure_count)) {
      break;
    }
  }

  const std::uint32_t length = rightmost - leftmost;
  for (std::uint32_t i = 0u; i < length; ++i) draft[i] = draft[leftmost + i];
  for (std::uint32_t i = length; i < config.max_units; ++i) {
    draft[i] = kResidentSynthesisInvalid;
  }
  draft_lengths[draft_id] = length;
}

template <typename BigramKeyT, typename TrigramKeyT>
__device__ inline unsigned long long synthesis_repair_score(
    const ResidentSynthesisModelView<BigramKeyT, TrigramKeyT>& model,
    const std::uint32_t* draft, std::uint32_t length, std::uint32_t position,
    std::uint32_t candidate, std::uint32_t trigram_weight) {
  if (candidate >= model.unit_count) return 0ull;
  const std::uint32_t left = draft[position - 1u];
  const std::uint32_t right = draft[position + 1u];
  unsigned long long left_support = synthesis_combined_bigram(model, left, candidate);
  unsigned long long right_support = synthesis_combined_bigram(model, candidate, right);
  if (position >= 2u) {
    left_support += static_cast<unsigned long long>(trigram_weight) *
        synthesis_combined_trigram(
        model, draft[position - 2u], left, candidate);
  }
  if (position + 2u < length) {
    right_support += static_cast<unsigned long long>(trigram_weight) *
        synthesis_combined_trigram(
        model, candidate, right, draft[position + 2u]);
  }
  const std::uint32_t vitality =
      model.unit_vitality == nullptr ? 0u : model.unit_vitality[candidate];
  unsigned long long semantic = synthesis_activation(model, candidate) /
      static_cast<unsigned long long>(1u + vitality);
  semantic = synthesis_depth(semantic);
  if (model.cue_masks != nullptr) semantic += __popc(model.cue_masks[candidate]);
  if (model.salient_masks != nullptr) {
    semantic += 2u * __popc(model.salient_masks[candidate]);
  }
  return synthesis_sat_mul(
      synthesis_sat_mul(1ull + left_support, 1ull + right_support),
      1ull + semantic);
}

template <typename BigramKeyT, typename TrigramKeyT>
__device__ inline void consider_repair_candidate(
    const ResidentSynthesisModelView<BigramKeyT, TrigramKeyT>& model,
    const std::uint32_t* draft, std::uint32_t length, std::uint32_t position,
    std::uint32_t candidate, std::uint32_t* best,
    unsigned long long* best_score, std::uint32_t trigram_weight) {
  const unsigned long long score =
      synthesis_repair_score(
          model, draft, length, position, candidate, trigram_weight);
  if (score > *best_score) {
    *best_score = score;
    *best = candidate;
  }
}

template <typename BigramKeyT, typename TrigramKeyT>
__global__ void repair_draft_interiors_kernel(
    ResidentSynthesisModelView<BigramKeyT, TrigramKeyT> model,
    const ReverseBigramEdge* reverse_bigrams, std::uint32_t reverse_bigram_count,
    ResidentSynthesisConfig config, std::uint32_t* drafts,
    const std::uint32_t* draft_lengths) {
  const std::uint32_t draft_id = blockIdx.x;
  if (draft_id >= config.candidate_count || threadIdx.x != 0u) return;
  std::uint32_t* draft = drafts + draft_id * kResidentSynthesisMaxUnits;
  const std::uint32_t length = draft_lengths[draft_id];
  const std::uint32_t trigram_weight =
      synthesis_variant_trigram_weight(draft_id / config.seed_count);
  if (length < 3u) return;
  for (std::uint32_t position = 1u; position + 1u < length; ++position) {
    std::uint32_t best = draft[position];
    unsigned long long best_score =
        synthesis_repair_score(
            model, draft, length, position, best, trigram_weight);
    const std::uint32_t left = draft[position - 1u];
    const BigramKeyT* keys[2] = {model.base_bigrams, model.online_bigrams};
    const std::uint32_t* counts[2] = {
        model.base_bigram_counts, model.online_bigram_counts};
    const std::uint32_t sizes[2] = {
        model.base_bigram_count, model.online_bigram_count};
    for (std::uint32_t source = 0u; source < 2u; ++source) {
      if (keys[source] == nullptr || counts[source] == nullptr) continue;
      const std::uint32_t begin = lower_forward_bigram(keys[source], sizes[source], left);
      const std::uint32_t end = upper_forward_bigram(keys[source], sizes[source], left);
      const std::uint32_t samples = synthesis_min(end - begin, config.edge_scan_limit);
      for (std::uint32_t sample = 0u; sample < samples; ++sample) {
        const std::uint32_t edge =
            synthesis_sample_index(begin, end, sample, samples);
        if (counts[source][edge] != 0u) {
          consider_repair_candidate(model, draft, length, position,
                                    keys[source][edge].next, &best, &best_score,
                                    trigram_weight);
        }
      }
    }
    const std::uint32_t right = draft[position + 1u];
    const std::uint32_t reverse_begin = lower_reverse_bigram(
        reverse_bigrams, reverse_bigram_count, right);
    const std::uint32_t reverse_end = upper_reverse_bigram(
        reverse_bigrams, reverse_bigram_count, right);
    const std::uint32_t reverse_samples =
        synthesis_min(reverse_end - reverse_begin, config.edge_scan_limit);
    for (std::uint32_t sample = 0u; sample < reverse_samples; ++sample) {
      const ReverseBigramEdge edge = reverse_bigrams[
          synthesis_sample_index(reverse_begin, reverse_end, sample, reverse_samples)];
      if (edge.support != 0u) {
        consider_repair_candidate(model, draft, length, position,
                                  edge.previous, &best, &best_score,
                                  trigram_weight);
      }
    }
    draft[position] = best;
  }
}

template <typename BigramKeyT, typename TrigramKeyT>
__device__ inline bool synthesis_has_repeated_cycle(
    const std::uint32_t* draft, std::uint32_t length) {
  for (std::uint32_t period = 1u; period <= 16u && 3u * period <= length;
       ++period) {
    for (std::uint32_t begin = 0u; begin + 3u * period <= length; ++begin) {
      bool repeated = true;
      for (std::uint32_t offset = 0u; offset < period; ++offset) {
        repeated &= draft[begin + offset] == draft[begin + period + offset];
        repeated &= draft[begin + offset] == draft[begin + 2u * period + offset];
      }
      if (repeated) return true;
    }
  }
  return false;
}

template <typename BigramKeyT, typename TrigramKeyT>
__global__ void appraise_drafts_kernel(
    ResidentSynthesisModelView<BigramKeyT, TrigramKeyT> model,
    ResidentSynthesisConfig config, const std::uint32_t* seed_units,
    const std::uint32_t* drafts,
    const std::uint32_t* draft_lengths, unsigned long long* draft_scores) {
  const std::uint32_t draft_id = blockIdx.x * blockDim.x + threadIdx.x;
  if (draft_id >= config.candidate_count) return;
  const std::uint32_t length = draft_lengths[draft_id];
  const std::uint32_t subject = seed_units[draft_id % config.seed_count];
  const bool subject_is_salient =
      synthesis_subject_is_selected(model, subject);
  if (length > config.max_units ||
      (length < config.min_units && !subject_is_salient)) {
    draft_scores[draft_id] = 0ull;
    return;
  }
  const std::uint32_t* draft = drafts + draft_id * kResidentSynthesisMaxUnits;
  const bool causal_value = subject_is_salient && length == 1u;
  const std::uint32_t trigram_weight = synthesis_variant_trigram_weight(
      draft_id / config.seed_count);
  for (std::uint32_t i = 0u; i < length; ++i) {
    if (draft[i] >= model.unit_count) {
      draft_scores[draft_id] = 0ull;
      return;
    }
  }
  if (synthesis_has_repeated_cycle<BigramKeyT, TrigramKeyT>(draft, length)) {
    draft_scores[draft_id] = 0ull;
    return;
  }
  if (model.source_windows != nullptr &&
      length >= kResidentSynthesisNoveltyUnits) {
    for (std::uint32_t begin = 0u;
         begin + kResidentSynthesisNoveltyUnits <= length; ++begin) {
      if (synthesis_source_window_present(
              model.source_windows, model.source_window_count,
              synthesis_source_window_signature(draft + begin))) {
        draft_scores[draft_id] = 0ull;
        return;
      }
    }
  }
  if (!causal_value && synthesis_recent_source_prefix_present(model, draft, length)) {
    draft_scores[draft_id] = 0ull;
    return;
  }
  unsigned long long score = length;
  std::uint32_t cue_coverage = 0u;
  std::uint32_t salient_coverage = 0u;
  std::uint32_t directional_coverage = 0u;
  std::uint32_t conditioned_edges = 0u;
  unsigned long long conditioned_support = 0ull;
  std::uint32_t repetitions = 0u;
  bool closed = false;
  for (std::uint32_t i = 0u; i < length; ++i) {
    const std::uint32_t unit = draft[i];
    if (unit >= model.unit_count) {
      draft_scores[draft_id] = 0ull;
      return;
    }
    cue_coverage |= model.cue_masks == nullptr ? 0u : model.cue_masks[unit];
    salient_coverage |=
        model.salient_masks == nullptr ? 0u : model.salient_masks[unit];
    directional_coverage |=
        (model.forward_support == nullptr ? 0u : model.forward_support[unit]) |
        (model.backward_support == nullptr ? 0u : model.backward_support[unit]);
    score = synthesis_sat_add(score,
        synthesis_depth(synthesis_activation(model, unit)));
    if (i != 0u) {
      score = synthesis_sat_add(score,
          synthesis_combined_bigram(model, draft[i - 1u], unit));
      const std::uint32_t support = subject_is_salient
          ? synthesis_subject_transition_count(
                model.subject_transitions, model.subject_transition_counts,
                model.subject_transition_count, subject, draft[i - 1u], unit)
          : 0u;
      conditioned_edges += support != 0u;
      conditioned_support = synthesis_sat_add(conditioned_support, support);
    }
    if (i >= 2u) {
      score = synthesis_sat_add(score,
          static_cast<unsigned long long>(trigram_weight) *
              synthesis_combined_trigram(
              model, draft[i - 2u], draft[i - 1u], unit));
    }
    for (std::uint32_t prior = 0u; prior < i; ++prior) {
      repetitions += draft[prior] == unit;
    }
    if (i + 1u == length) {
      closed = synthesis_unit_contains(
          model.unit_lengths, model.unit_content, model.unit_words, unit,
          model.closure_bytes, model.closure_count);
    }
  }
  if (causal_value) {
    conditioned_edges = 1u;
    conditioned_support = synthesis_sat_add(
        conditioned_support, synthesis_activation(model, draft[0]));
  }
  if (length < config.min_units && !causal_value && conditioned_edges < 2u) {
    draft_scores[draft_id] = 0ull;
    return;
  }
  score = synthesis_sat_add(score,
      static_cast<unsigned long long>(__popc(cue_coverage)) << 40u);
  score = synthesis_sat_add(score,
      static_cast<unsigned long long>(__popc(salient_coverage)) << 44u);
  score = synthesis_sat_add(score,
      static_cast<unsigned long long>(__popc(directional_coverage)) << 48u);
  score = synthesis_sat_add(score,
      static_cast<unsigned long long>(conditioned_edges) << 56u);
  score = synthesis_sat_add(score,
      static_cast<unsigned long long>(synthesis_depth(conditioned_support)) << 52u);
  if (closed) score = synthesis_sat_add(score, 1ull << 44u);
  if (model.policy_state != nullptr &&
      model.policy_state->history_size >= 16u &&
      trigram_weight == resident_synthesis_policy_weight(
          resident_synthesis_policy_select(*model.policy_state))) {
    score = synthesis_sat_add(score, 1ull << 34u);
  }
  const unsigned long long penalty =
      static_cast<unsigned long long>(repetitions) << 32u;
  draft_scores[draft_id] = score > penalty ? score - penalty : 1ull;
}

template <typename BigramKeyT, typename TrigramKeyT>
__global__ void select_synthesis_winner_kernel(
    ResidentSynthesisModelView<BigramKeyT, TrigramKeyT> model,
    ResidentSynthesisConfig config, const std::uint32_t* seed_units,
    const std::uint32_t* drafts,
    const std::uint32_t* draft_lengths, const unsigned long long* draft_scores,
    std::uint32_t* selected_units, ResidentSynthesisResult* result) {
  if (blockIdx.x != 0u || threadIdx.x != 0u) return;
  std::uint32_t winner = kResidentSynthesisInvalid;
  unsigned long long winner_score = 0ull;
  std::uint32_t winner_coverage = 0u;
  const std::uint32_t baseline_count = min(
      config.candidate_count,
      config.seed_count * kResidentSynthesisBaseVariants);
  for (std::uint32_t candidate = 0u; candidate < baseline_count; ++candidate) {
    const unsigned long long score = draft_scores[candidate];
    if (score > winner_score) {
      winner_score = score;
      winner = candidate;
    }
  }
  if (winner != kResidentSynthesisInvalid && model.cue_masks != nullptr) {
    const std::uint32_t* baseline =
        drafts + winner * kResidentSynthesisMaxUnits;
    for (std::uint32_t i = 0u; i < draft_lengths[winner]; ++i) {
      winner_coverage |= model.cue_masks[baseline[i]];
    }
  }
  for (std::uint32_t candidate = baseline_count;
       candidate < config.candidate_count; ++candidate) {
    const unsigned long long score = draft_scores[candidate];
    if (score <= winner_score) continue;
    std::uint32_t candidate_coverage = 0u;
    if (model.cue_masks != nullptr) {
      const std::uint32_t* draft =
          drafts + candidate * kResidentSynthesisMaxUnits;
      for (std::uint32_t i = 0u; i < draft_lengths[candidate]; ++i) {
        candidate_coverage |= model.cue_masks[draft[i]];
      }
    }
    if (winner != kResidentSynthesisInvalid &&
        __popc(candidate_coverage) < __popc(winner_coverage) + 2) {
      continue;
    }
    winner_score = score;
    winner = candidate;
    winner_coverage = candidate_coverage;
  }
  for (std::uint32_t seed_rank = 0u; seed_rank < config.seed_count; ++seed_rank) {
    const std::uint32_t subject = seed_units[seed_rank];
    if (!synthesis_subject_is_selected(model, subject)) continue;
    std::uint32_t conditioned_winner = kResidentSynthesisInvalid;
    std::uint32_t conditioned_edges = 0u;
    unsigned long long conditioned_score = 0ull;
    for (std::uint32_t candidate = seed_rank;
         candidate < config.candidate_count; candidate += config.seed_count) {
      const unsigned long long score = draft_scores[candidate];
      if (score == 0ull) continue;
      const std::uint32_t* draft =
          drafts + candidate * kResidentSynthesisMaxUnits;
      std::uint32_t edges = 0u;
      for (std::uint32_t i = 1u; i < draft_lengths[candidate]; ++i) {
        edges += synthesis_subject_transition_count(
                     model.subject_transitions, model.subject_transition_counts,
                     model.subject_transition_count, subject,
                     draft[i - 1u], draft[i]) != 0u;
      }
      const bool causal_value = draft_lengths[candidate] == 1u;
      edges += causal_value ? 1u : 0u;
      if (edges > conditioned_edges ||
          (edges == conditioned_edges && edges != 0u && score > conditioned_score)) {
        conditioned_winner = candidate;
        conditioned_edges = edges;
        conditioned_score = score;
      }
    }
    if (conditioned_winner != kResidentSynthesisInvalid) {
      winner = conditioned_winner;
      winner_score = conditioned_score;
      break;
    }
  }
  *result = {};
  result->winner = winner;
  if (winner == kResidentSynthesisInvalid || winner_score == 0ull) return;
  const std::uint32_t length = draft_lengths[winner];
  const std::uint32_t* draft = drafts + winner * kResidentSynthesisMaxUnits;
  const std::uint32_t subject = seed_units[winner % config.seed_count];
  bool conditioned = false;
  std::uint32_t relation_predicate = kResidentSynthesisInvalid;
  std::uint32_t relation_value = kResidentSynthesisInvalid;
  std::uint32_t relation_tail[kResidentSynthesisMaxRelationTail];
  for (std::uint32_t index = 0u;
       index < kResidentSynthesisMaxRelationTail; ++index) {
    relation_tail[index] = kResidentSynthesisInvalid;
  }
  std::uint32_t relation_tail_count = 0u;
  if (synthesis_subject_is_selected(model, subject)) {
    for (std::uint32_t i = 1u; i < length; ++i) {
      conditioned |= synthesis_subject_transition_count(
                         model.subject_transitions,
                         model.subject_transition_counts,
                         model.subject_transition_count, subject,
                         draft[i - 1u], draft[i]) != 0u;
    }
    conditioned |= length == 1u;
    if (conditioned) {
      relation_value = synthesis_conditioned_relation_value(
          model, subject, config.edge_scan_limit, &relation_predicate);
      if (relation_value != kResidentSynthesisInvalid) {
        relation_tail_count = synthesis_conditioned_relation_tail(
            model, subject, relation_value, config.edge_scan_limit,
            relation_tail, kResidentSynthesisMaxRelationTail);
      }
    }
  }
  std::uint32_t cue_coverage = 0u;
  for (std::uint32_t i = 0u; i < length; ++i) {
    selected_units[i] = draft[i];
    if (model.cue_masks != nullptr) cue_coverage |= model.cue_masks[draft[i]];
  }
  result->ready = 1u;
  result->unit_count = length;
  result->cue_coverage = __popc(cue_coverage);
  result->conditioned = conditioned ? 1u : 0u;
  result->closed = length != 0u && synthesis_unit_contains(
      model.unit_lengths, model.unit_content, model.unit_words, draft[length - 1u],
      model.closure_bytes, model.closure_count);
  result->score_low = static_cast<std::uint32_t>(winner_score);
  result->score_high = static_cast<std::uint32_t>(winner_score >> 32u);
  result->relation_subject = conditioned ? subject : kResidentSynthesisInvalid;
  result->relation_predicate = relation_predicate;
  result->relation_value = relation_value;
  result->relation_tail_count = relation_tail_count;
  for (std::uint32_t index = 0u;
       index < kResidentSynthesisMaxRelationTail; ++index) {
    result->relation_tail[index] = relation_tail[index];
  }
}

#include "bcc32_cuda_resident_synthesis_tail.inl"
