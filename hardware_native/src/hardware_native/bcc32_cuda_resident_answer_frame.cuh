#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

#include "bcc32_cuda_resident_roles.cuh"

// A bounded CUDA organ that turns an already-resident factual value into a
// source-novel response frame.  It consumes only mutable resident unit matter,
// literal transition counts, structural-role counts, activation, and vitality.
// It has no cue text, source offsets, vocabulary, token classes, answer labels,
// retrieval path, or host-selected candidate.  The selected value remains the
// causal seed; grammar decides the material on either side of it.
namespace substrate::bcc32::resident_answer_frame {

inline constexpr std::uint32_t kInvalidUnit = 0xffffffffu;
inline constexpr std::uint32_t kMaxFrameUnits = 16u;
inline constexpr std::uint32_t kMaxNoveltyWindow = 6u;
inline constexpr std::uint32_t kMaxRankedBranches = 4u;
inline constexpr std::uint32_t kUnitEligible = 1u << 0u;
inline constexpr std::uint32_t kUnitClosure = 1u << 1u;
inline constexpr std::uint32_t kUnitTerminalOnly = 1u << 2u;

struct BigramKey {
  std::uint32_t previous = 0u;
  std::uint32_t next = 0u;

  [[nodiscard]] __host__ __device__ bool operator<(const BigramKey& other) const {
    return previous < other.previous || (previous == other.previous && next < other.next);
  }
};

struct TrigramKey {
  std::uint32_t first = 0u;
  std::uint32_t second = 0u;
  std::uint32_t next = 0u;

  [[nodiscard]] __host__ __device__ bool operator<(const TrigramKey& other) const {
    if (first != other.first)
      return first < other.first;
    if (second != other.second)
      return second < other.second;
    return next < other.next;
  }
};

using MutableRoleState = resident_roles::MutableStructuralRole;

struct SourceWindowSignature {
  std::uint32_t hash_a = 0u;
  std::uint32_t hash_b = 0u;

  [[nodiscard]] __host__ __device__ bool operator<(const SourceWindowSignature& other) const {
    return hash_a < other.hash_a || (hash_a == other.hash_a && hash_b < other.hash_b);
  }

  [[nodiscard]] __host__ __device__ bool operator==(const SourceWindowSignature& other) const {
    return hash_a == other.hash_a && hash_b == other.hash_b;
  }
};

// Every field is resident device state.  Hosts launch a fixed pipeline and do
// not inspect these values to choose an answer.  The organ can revise them, and
// the lesion kernels below can remove either policy, selection, activation, or
// role grammar without changing executable code.
struct MutablePolicyState {
  std::uint32_t enabled = 1u;
  std::uint32_t revision = 1u;
  std::uint32_t lesion_events = 0u;
  std::uint32_t candidate_count = 192u;
  std::uint32_t branch_width = 4u;
  std::uint32_t max_left_units = 3u;
  std::uint32_t max_right_units = 2u;
  std::uint32_t min_output_units = 4u;
  std::uint32_t novelty_window = 4u;
  std::uint32_t require_closure = 1u;
  std::uint32_t literal_bigram_weight = 8u;
  std::uint32_t literal_trigram_weight = 16u;
  std::uint32_t role_bigram_weight = 12u;
  std::uint32_t role_trigram_weight = 24u;
  std::uint32_t activation_weight = 4u;
  std::uint32_t vitality_weight = 1u;
  std::uint32_t closure_bonus = 64u;
  std::uint32_t relation_tail_min_vitality = 4u;
  std::uint32_t raw_analogy_weight = 16384u;
  std::uint32_t causal_value_bonus = 1048576u;
};

struct MutableSelectionState {
  std::uint32_t value = kInvalidUnit;
  std::uint32_t subject = kInvalidUnit;
  std::uint32_t predicate = kInvalidUnit;
  std::uint32_t evidence = 0u;
  std::uint32_t revision = 0u;
  std::uint32_t intact = 1u;
};

// All pointers address device-resident mutable storage.  The compose kernels
// only read the view, but no array is made immutable by the API.  candidate_units
// is a bounded resident active set, avoiding a vocabulary-wide host search.
template <typename BigramKeyT, typename TrigramKeyT>
struct BasicMutableModelView {
  std::uint32_t* unit_lengths = nullptr;
  std::uint32_t* unit_content = nullptr;
  std::uint32_t unit_words = 0u;
  std::uint32_t* unit_vitality = nullptr;
  std::uint32_t* unit_activation = nullptr;
  std::uint32_t* unit_flags = nullptr;
  MutableRoleState* unit_roles = nullptr;
  std::uint32_t unit_count = 0u;

  std::uint32_t* candidate_units = nullptr;
  std::uint32_t candidate_unit_count = 0u;
  std::uint32_t* relation_tail = nullptr;
  std::uint32_t* relation_tail_count = nullptr;
  std::uint32_t relation_tail_capacity = 0u;

  BigramKeyT* bigrams = nullptr;
  std::uint32_t* bigram_counts = nullptr;
  std::uint32_t bigram_count = 0u;
  TrigramKeyT* trigrams = nullptr;
  std::uint32_t* trigram_counts = nullptr;
  std::uint32_t trigram_count = 0u;

  BigramKeyT* online_bigrams = nullptr;
  std::uint32_t* online_bigram_counts = nullptr;
  std::uint32_t online_bigram_count = 0u;
  TrigramKeyT* online_trigrams = nullptr;
  std::uint32_t* online_trigram_counts = nullptr;
  std::uint32_t online_trigram_count = 0u;

  std::uint64_t* role_bigrams = nullptr;
  std::uint64_t* role_trigrams = nullptr;
  std::uint32_t role_count = 0u;

  SourceWindowSignature* source_windows = nullptr;
  std::uint32_t source_window_count = 0u;

  MutablePolicyState* policy = nullptr;
};

using MutableModelView = BasicMutableModelView<BigramKey, TrigramKey>;

struct MutableWorkspaceView {
  std::uint32_t* drafts = nullptr;
  std::uint32_t* draft_lengths = nullptr;
  unsigned long long* draft_scores = nullptr;
  std::uint32_t candidate_capacity = 0u;

  std::uint32_t* output_units = nullptr;
  std::uint32_t output_unit_capacity = 0u;
  std::uint8_t* output_bytes = nullptr;
  std::uint32_t output_byte_capacity = 0u;
};

struct Result {
  std::uint32_t ready = 0u;
  std::uint32_t candidate = kInvalidUnit;
  std::uint32_t unit_count = 0u;
  std::uint32_t byte_count = 0u;
  std::uint32_t value_position = kInvalidUnit;
  std::uint32_t closed = 0u;
  std::uint32_t source_novel = 0u;
  std::uint32_t policy_revision = 0u;
  std::uint32_t selection_revision = 0u;
  std::uint32_t score_low = 0u;
  std::uint32_t score_high = 0u;
};

[[nodiscard]] __host__ __device__ inline std::uint32_t mix32(std::uint32_t value) {
  value ^= value >> 16u;
  value *= 0x7feb352du;
  value ^= value >> 15u;
  value *= 0x846ca68bu;
  return value ^ (value >> 16u);
}

[[nodiscard]] __host__ __device__ inline SourceWindowSignature source_window_signature(
    const std::uint32_t* units, std::uint32_t count) {
  std::uint32_t a = 0x811c9dc5u ^ count;
  std::uint32_t b = 0x9e3779b9u + count;
  for (std::uint32_t i = 0u; i < count; ++i) {
    a = mix32(a ^ units[i] ^ (i + 1u));
    b = mix32(b + units[i] * 0x85ebca6bu + i);
  }
  return {a, b};
}

[[nodiscard]] __device__ inline unsigned long long sat_add(unsigned long long a,
                                                           unsigned long long b) {
  return a > ~0ull - b ? ~0ull : a + b;
}

[[nodiscard]] __device__ inline unsigned long long sat_mul(unsigned long long a,
                                                           unsigned long long b) {
  return a != 0ull && b > ~0ull / a ? ~0ull : a * b;
}

[[nodiscard]] __device__ inline std::uint32_t depth64(unsigned long long value) {
  std::uint32_t depth = 0u;
  while (value != 0ull) {
    ++depth;
    value >>= 1u;
  }
  return depth;
}

template <typename BigramKeyT, typename TrigramKeyT>
[[nodiscard]] __device__ inline std::uint8_t unit_byte(
    const BasicMutableModelView<BigramKeyT, TrigramKeyT>& model, std::uint32_t unit,
    std::uint32_t offset) {
  const std::uint32_t packed = model.unit_content[unit * model.unit_words + offset / 4u];
  return static_cast<std::uint8_t>((packed >> (8u * (offset & 3u))) & 0xffu);
}

template <typename BigramKeyT>
[[nodiscard]] __device__ inline std::uint32_t literal_bigram_table_count(
    const BigramKeyT* keys, const std::uint32_t* counts, std::uint32_t count,
    std::uint32_t previous, std::uint32_t next) {
  if (keys == nullptr || counts == nullptr || count == 0u) return 0u;
  std::uint32_t lo = 0u;
  std::uint32_t hi = count;
  const BigramKeyT target{previous, next};
  while (lo < hi) {
    const std::uint32_t mid = lo + (hi - lo) / 2u;
    if (keys[mid] < target)
      lo = mid + 1u;
    else
      hi = mid;
  }
  return lo < count && keys[lo].previous == previous && keys[lo].next == next
             ? counts[lo]
             : 0u;
}

template <typename BigramKeyT, typename TrigramKeyT>
[[nodiscard]] __device__ inline unsigned long long bigram_count(
    const BasicMutableModelView<BigramKeyT, TrigramKeyT>& model,
    std::uint32_t previous, std::uint32_t next) {
  return static_cast<unsigned long long>(literal_bigram_table_count(
             model.bigrams, model.bigram_counts, model.bigram_count,
             previous, next)) +
      literal_bigram_table_count(
             model.online_bigrams, model.online_bigram_counts,
             model.online_bigram_count, previous, next);
}

template <typename TrigramKeyT>
[[nodiscard]] __device__ inline std::uint32_t literal_trigram_table_count(
    const TrigramKeyT* keys, const std::uint32_t* counts, std::uint32_t count,
    std::uint32_t first, std::uint32_t second, std::uint32_t next) {
  if (keys == nullptr || counts == nullptr || count == 0u) return 0u;
  std::uint32_t lo = 0u;
  std::uint32_t hi = count;
  const TrigramKeyT target{first, second, next};
  while (lo < hi) {
    const std::uint32_t mid = lo + (hi - lo) / 2u;
    if (keys[mid] < target)
      lo = mid + 1u;
    else
      hi = mid;
  }
  return lo < count && keys[lo].first == first &&
                 keys[lo].second == second && keys[lo].next == next
             ? counts[lo]
             : 0u;
}

template <typename BigramKeyT, typename TrigramKeyT>
[[nodiscard]] __device__ inline unsigned long long trigram_count(
    const BasicMutableModelView<BigramKeyT, TrigramKeyT>& model,
    std::uint32_t first, std::uint32_t second, std::uint32_t next) {
  return static_cast<unsigned long long>(literal_trigram_table_count(
             model.trigrams, model.trigram_counts, model.trigram_count,
             first, second, next)) +
      literal_trigram_table_count(
             model.online_trigrams, model.online_trigram_counts,
             model.online_trigram_count, first, second, next);
}

template <typename BigramKeyT, typename TrigramKeyT>
[[nodiscard]] __device__ inline unsigned long long role_bigram_count(
    const BasicMutableModelView<BigramKeyT, TrigramKeyT>& model, std::uint32_t previous,
    std::uint32_t next) {
  if (model.unit_roles == nullptr || model.role_bigrams == nullptr ||
      previous >= model.unit_count || next >= model.unit_count) {
    return 0ull;
  }
  const MutableRoleState a = model.unit_roles[previous];
  const MutableRoleState b = model.unit_roles[next];
  if (a.role >= model.role_count || b.role >= model.role_count)
    return 0ull;
  const unsigned long long support =
      model.role_bigrams[static_cast<std::size_t>(a.role) * model.role_count + b.role];
  const std::uint32_t confidence = a.confidence < b.confidence ? a.confidence : b.confidence;
  return sat_mul(support, 1ull + depth64(confidence));
}

template <typename BigramKeyT, typename TrigramKeyT>
[[nodiscard]] __device__ inline unsigned long long role_trigram_count(
    const BasicMutableModelView<BigramKeyT, TrigramKeyT>& model, std::uint32_t first,
    std::uint32_t second, std::uint32_t next) {
  if (model.unit_roles == nullptr || model.role_trigrams == nullptr || first >= model.unit_count ||
      second >= model.unit_count || next >= model.unit_count) {
    return 0ull;
  }
  const MutableRoleState a = model.unit_roles[first];
  const MutableRoleState b = model.unit_roles[second];
  const MutableRoleState c = model.unit_roles[next];
  if (a.role >= model.role_count || b.role >= model.role_count || c.role >= model.role_count) {
    return 0ull;
  }
  const std::size_t index =
      (static_cast<std::size_t>(a.role) * model.role_count + b.role) * model.role_count + c.role;
  const unsigned long long support = model.role_trigrams[index];
  std::uint32_t confidence = a.confidence < b.confidence ? a.confidence : b.confidence;
  if (c.confidence < confidence)
    confidence = c.confidence;
  return sat_mul(support, 1ull + depth64(confidence));
}

template <typename BigramKeyT, typename TrigramKeyT>
[[nodiscard]] __device__ inline bool source_window_present(
    const BasicMutableModelView<BigramKeyT, TrigramKeyT>& model, SourceWindowSignature target) {
  std::uint32_t lo = 0u;
  std::uint32_t hi = model.source_window_count;
  while (lo < hi) {
    const std::uint32_t mid = lo + (hi - lo) / 2u;
    if (model.source_windows[mid] < target)
      lo = mid + 1u;
    else
      hi = mid;
  }
  return lo < model.source_window_count && model.source_windows[lo] == target;
}

[[nodiscard]] __device__ inline bool draft_contains(const std::uint32_t* draft,
                                                    std::uint32_t length, std::uint32_t unit) {
  for (std::uint32_t i = 0u; i < length; ++i) {
    if (draft[i] == unit)
      return true;
  }
  return false;
}

template <typename BigramKeyT, typename TrigramKeyT>
[[nodiscard]] __device__ inline unsigned long long unit_state_score(
    const BasicMutableModelView<BigramKeyT, TrigramKeyT>& model, const MutablePolicyState& policy,
    std::uint32_t unit) {
  const unsigned long long activation =
      model.unit_activation == nullptr ? 0ull : depth64(model.unit_activation[unit]);
  const unsigned long long vitality =
      model.unit_vitality == nullptr ? 0ull : depth64(model.unit_vitality[unit]);
  return sat_add(sat_mul(activation, policy.activation_weight),
                 sat_mul(vitality, policy.vitality_weight));
}

template <typename BigramKeyT, typename TrigramKeyT>
[[nodiscard]] __device__ inline unsigned long long forward_score(
    const BasicMutableModelView<BigramKeyT, TrigramKeyT>& model, const MutablePolicyState& policy,
    std::uint32_t previous_previous, std::uint32_t previous, std::uint32_t candidate) {
  const unsigned long long literal_bi = bigram_count(model, previous, candidate);
  const unsigned long long literal_tri =
      previous_previous == kInvalidUnit
          ? 0ull
          : trigram_count(model, previous_previous, previous, candidate);
  const unsigned long long role_bi = role_bigram_count(model, previous, candidate);
  const unsigned long long role_tri =
      previous_previous == kInvalidUnit
          ? 0ull
          : role_trigram_count(model, previous_previous, previous, candidate);
  if (literal_bi == 0ull && literal_tri == 0ull && role_bi == 0ull && role_tri == 0ull) {
    return 0ull;
  }
  unsigned long long score = unit_state_score(model, policy, candidate);
  score = sat_add(score, sat_mul(literal_bi, policy.literal_bigram_weight));
  score = sat_add(score, sat_mul(literal_tri, policy.literal_trigram_weight));
  score = sat_add(score, sat_mul(role_bi, policy.role_bigram_weight));
  score = sat_add(score, sat_mul(role_tri, policy.role_trigram_weight));
  if ((model.unit_flags[candidate] & kUnitClosure) != 0u) {
    score = sat_add(score, policy.closure_bonus);
  }
  return score;
}

template <typename BigramKeyT, typename TrigramKeyT>
[[nodiscard]] __device__ inline unsigned long long backward_score(
    const BasicMutableModelView<BigramKeyT, TrigramKeyT>& model, const MutablePolicyState& policy,
    std::uint32_t candidate, std::uint32_t current, std::uint32_t next) {
  const unsigned long long literal_bi = bigram_count(model, candidate, current);
  const unsigned long long literal_tri =
      next == kInvalidUnit ? 0ull : trigram_count(model, candidate, current, next);
  const unsigned long long role_bi = role_bigram_count(model, candidate, current);
  const unsigned long long role_tri =
      next == kInvalidUnit ? 0ull : role_trigram_count(model, candidate, current, next);
  if (literal_bi == 0ull && literal_tri == 0ull && role_bi == 0ull && role_tri == 0ull) {
    return 0ull;
  }
  unsigned long long score = unit_state_score(model, policy, candidate);
  score = sat_add(score, sat_mul(literal_bi, policy.literal_bigram_weight));
  score = sat_add(score, sat_mul(literal_tri, policy.literal_trigram_weight));
  score = sat_add(score, sat_mul(role_bi, policy.role_bigram_weight));
  score = sat_add(score, sat_mul(role_tri, policy.role_trigram_weight));
  return score;
}

__device__ inline void insert_ranked(std::uint32_t unit, unsigned long long score,
                                     std::uint32_t* units, unsigned long long* scores) {
  for (std::uint32_t rank = 0u; rank < kMaxRankedBranches; ++rank) {
    if (score > scores[rank] || (score == scores[rank] && score != 0ull && unit < units[rank])) {
      for (std::uint32_t move = kMaxRankedBranches - 1u; move > rank; --move) {
        units[move] = units[move - 1u];
        scores[move] = scores[move - 1u];
      }
      units[rank] = unit;
      scores[rank] = score;
      return;
    }
  }
}

template <bool Forward, typename BigramKeyT, typename TrigramKeyT>
[[nodiscard]] __device__ inline std::uint32_t choose_unit(
    const BasicMutableModelView<BigramKeyT, TrigramKeyT>& model, const MutablePolicyState& policy,
    const std::uint32_t* draft, std::uint32_t left, std::uint32_t right, std::uint32_t variant,
    std::uint32_t step) {
  std::uint32_t ranked_units[kMaxRankedBranches];
  unsigned long long ranked_scores[kMaxRankedBranches];
  for (std::uint32_t i = 0u; i < kMaxRankedBranches; ++i) {
    ranked_units[i] = kInvalidUnit;
    ranked_scores[i] = 0ull;
  }

  for (std::uint32_t index = 0u; index < model.candidate_unit_count; ++index) {
    const std::uint32_t candidate = model.candidate_units[index];
    if (candidate >= model.unit_count || (model.unit_flags[candidate] & kUnitEligible) == 0u ||
        draft_contains(draft + left, right - left + 1u, candidate)) {
      continue;
    }
    if (!Forward && (model.unit_flags[candidate] & kUnitClosure) != 0u)
      continue;
    if (Forward && (model.unit_flags[candidate] & kUnitClosure) != 0u &&
        (model.unit_flags[candidate] & kUnitTerminalOnly) == 0u)
      continue;
    const unsigned long long score =
        Forward ? forward_score(model, policy, right > left ? draft[right - 1u] : kInvalidUnit,
                                draft[right], candidate)
                : backward_score(model, policy, candidate, draft[left],
                                 left < right ? draft[left + 1u] : kInvalidUnit);
    if (score == 0ull)
      continue;
    insert_ranked(candidate, score, ranked_units, ranked_scores);
  }
  const std::uint32_t width =
      policy.branch_width == 0u
          ? 1u
          : (policy.branch_width > kMaxRankedBranches ? kMaxRankedBranches : policy.branch_width);
  const std::uint32_t rank = mix32(variant + 0x9e3779b9u * (step + 1u)) % width;
  return ranked_units[rank] == kInvalidUnit ? ranked_units[0] : ranked_units[rank];
}

template <typename BigramKeyT, typename TrigramKeyT>
[[nodiscard]] __device__ inline bool appraise_draft(
    const BasicMutableModelView<BigramKeyT, TrigramKeyT>& model, const MutablePolicyState& policy,
    const MutableSelectionState& selection, const std::uint32_t* draft, std::uint32_t length,
    unsigned long long* score_out) {
  if (length < policy.min_output_units || length > kMaxFrameUnits)
    return false;
  std::uint32_t value_count = 0u;
  bool closed = false;
  bool literal_predicate_value = false;
  bool role_subject_predicate = false;
  bool role_predicate_value = false;
  unsigned long long score = selection.evidence;
  for (std::uint32_t i = 0u; i < length; ++i) {
    const std::uint32_t unit = draft[i];
    if (unit >= model.unit_count)
      return false;
    value_count += unit == selection.value ? 1u : 0u;
    closed |= (model.unit_flags[unit] & kUnitClosure) != 0u;
    score = sat_add(score, unit_state_score(model, policy, unit));
    if (i != 0u) {
      const std::uint32_t first = i > 1u ? draft[i - 2u] : kInvalidUnit;
      const unsigned long long literal_bi =
          bigram_count(model, draft[i - 1u], unit);
      const unsigned long long literal_tri = first == kInvalidUnit
          ? 0ull
          : trigram_count(model, first, draft[i - 1u], unit);
      const bool literal_edge = literal_bi != 0ull || literal_tri != 0ull;
      literal_predicate_value |=
          draft[i - 1u] == selection.predicate && unit == selection.value && literal_edge;
      role_subject_predicate |=
          draft[i - 1u] == selection.subject && unit == selection.predicate &&
          role_bigram_count(model, draft[i - 1u], unit) != 0ull;
      role_predicate_value |=
          draft[i - 1u] == selection.predicate && unit == selection.value &&
          role_bigram_count(model, draft[i - 1u], unit) != 0ull;
      const std::uint32_t causal_tail_count = model.relation_tail_count == nullptr
          ? 0u : *model.relation_tail_count;
      const bool exhausted_causal_tail = model.relation_tail != nullptr &&
          causal_tail_count != 0u && causal_tail_count < model.relation_tail_capacity &&
          unit == model.relation_tail[causal_tail_count - 1u];
      const bool structural_terminal = i + 1u == length &&
          ((model.unit_flags[unit] & kUnitClosure) != 0u || exhausted_causal_tail);
      unsigned long long edge_score = structural_terminal
          ? policy.closure_bonus
          : forward_score(model, policy, first, draft[i - 1u], unit);
      if (edge_score == 0ull && model.relation_tail != nullptr &&
          model.relation_tail_count != nullptr) {
        const std::uint32_t tail_count = *model.relation_tail_count;
        for (std::uint32_t skipped = 0u; skipped + 1u < tail_count; ++skipped) {
          const std::uint32_t bridge_previous = skipped == 0u
              ? selection.value : model.relation_tail[skipped - 1u];
          const std::uint32_t bridge_middle = model.relation_tail[skipped];
          const std::uint32_t bridge_next = model.relation_tail[skipped + 1u];
          if (draft[i - 1u] != bridge_previous || unit != bridge_next) continue;
          const unsigned long long left =
              bigram_count(model, bridge_previous, bridge_middle);
          const unsigned long long right =
              bigram_count(model, bridge_middle, bridge_next);
          if (left != 0ull && right != 0ull) {
            edge_score = sat_add(
                sat_mul(left, policy.literal_bigram_weight),
                sat_mul(right, policy.literal_bigram_weight));
            break;
          }
        }
      }
      if (edge_score == 0ull)
        return false;
      score = sat_add(score, edge_score);
    }
  }
  const std::uint32_t causal_tail_count = model.relation_tail_count == nullptr
      ? 0u : *model.relation_tail_count;
  const bool exhausted_causal_tail = length != 0u && model.relation_tail != nullptr &&
      causal_tail_count != 0u && causal_tail_count < model.relation_tail_capacity &&
      draft[length - 1u] == model.relation_tail[causal_tail_count - 1u];
  const bool terminal = length != 0u &&
      ((model.unit_flags[draft[length - 1u]] & kUnitClosure) != 0u ||
       exhausted_causal_tail);
  if (value_count != 1u || selection.evidence == 0u || !literal_predicate_value ||
      !role_subject_predicate || !role_predicate_value ||
      (policy.require_closure != 0u && (!closed || !terminal)))
    return false;

  const std::uint32_t novelty = policy.novelty_window;
  if (model.source_window_count != 0u) {
    if (novelty == 0u || novelty > kMaxNoveltyWindow || length < novelty) {
      return false;
    }
    for (std::uint32_t i = 0u; i + novelty <= length; ++i) {
      if (source_window_present(model, source_window_signature(draft + i, novelty))) {
        return false;
      }
    }
  }
  *score_out = score;
  return true;
}

template <typename BigramKeyT, typename TrigramKeyT>
[[nodiscard]] __device__ inline bool appraise_conditioned_tail_draft(
    const BasicMutableModelView<BigramKeyT, TrigramKeyT>& model,
    const MutablePolicyState& policy, const MutableSelectionState& selection,
    const std::uint32_t* draft, std::uint32_t length,
    std::uint32_t connector, std::uint32_t tail_count,
    unsigned long long* score_out) {
  if (selection.evidence == 0u || model.relation_tail == nullptr ||
      tail_count < 2u ||
      length != 3u + tail_count || length > kMaxFrameUnits ||
      connector >= model.unit_count || selection.value >= model.unit_count ||
      connector == selection.subject ||
      connector == selection.predicate) {
    return false;
  }
  if (model.unit_vitality == nullptr || model.unit_vitality[connector] == 0u ||
      bigram_count(model, connector, model.relation_tail[0]) == 0ull) {
    return false;
  }

  auto raw_form_similarity = [&](std::uint32_t left, std::uint32_t right) {
    if (left >= model.unit_count || right >= model.unit_count)
      return 0u;
    const std::uint32_t left_length = model.unit_lengths[left];
    const std::uint32_t right_length = model.unit_lengths[right];
    const std::uint32_t shorter = min(left_length, right_length);
    const std::uint32_t longer = max(left_length, right_length);
    if (shorter == 0u || longer - shorter > 2u)
      return 0u;
    std::uint32_t prefix = 0u;
    while (prefix < shorter &&
           unit_byte(model, left, prefix) == unit_byte(model, right, prefix)) {
      ++prefix;
    }
    std::uint32_t suffix = 0u;
    while (suffix < shorter - prefix &&
           unit_byte(model, left, left_length - 1u - suffix) ==
               unit_byte(model, right, right_length - 1u - suffix)) {
      ++suffix;
    }
    const std::uint32_t shared = prefix + suffix;
    return shared + 1u >= shorter ? shared : 0u;
  };
  auto analogical_support = [&](const BigramKeyT* keys,
                                const std::uint32_t* counts,
                                std::uint32_t count) {
    unsigned long long support = 0ull;
    if (keys == nullptr || counts == nullptr) return support;
    for (std::uint32_t i = 0u; i < count; ++i) {
      if (keys[i].next != connector) continue;
      const std::uint32_t similarity =
          raw_form_similarity(selection.predicate, keys[i].previous);
      support = sat_add(support, sat_mul(similarity, counts[i]));
    }
    return support;
  };

  unsigned long long score = selection.evidence;
  if (connector == selection.value) {
    score = sat_add(score, policy.causal_value_bonus);
  } else {
    const unsigned long long analogy = sat_add(
        analogical_support(model.bigrams, model.bigram_counts, model.bigram_count),
        analogical_support(model.online_bigrams, model.online_bigram_counts,
                           model.online_bigram_count));
    score = sat_add(score, sat_mul(analogy, policy.raw_analogy_weight));
  }
  for (std::uint32_t i = 0u; i < length; ++i) {
    const std::uint32_t unit = draft[i];
    if (unit >= model.unit_count) return false;
    if (i >= 3u && unit != model.relation_tail[i - 3u]) return false;
    score = sat_add(score, unit_state_score(model, policy, unit));
    if (i == 0u) continue;
    const std::uint32_t first = i > 1u ? draft[i - 2u] : kInvalidUnit;
    if (i >= 4u && bigram_count(model, draft[i - 1u], unit) == 0ull) {
      return false;
    }
    const bool structural_terminal = i + 1u == length &&
        ((model.unit_flags[unit] & kUnitClosure) != 0u ||
         i == 2u + tail_count);
    const unsigned long long edge_score = structural_terminal
        ? policy.closure_bonus
        : forward_score(model, policy, first, draft[i - 1u], unit);
    if (edge_score == 0ull && i != 2u) return false;
    score = sat_add(score, edge_score);
  }
  if ((model.unit_flags[draft[length - 1u]] & kUnitClosure) == 0u &&
      length != 3u + tail_count) {
    return false;
  }
  const std::uint32_t novelty = policy.novelty_window;
  if (model.source_window_count != 0u) {
    if (novelty == 0u || novelty > kMaxNoveltyWindow || length < novelty) {
      return false;
    }
    const std::uint32_t windows = connector == selection.value
        ? 1u : length - novelty + 1u;
    for (std::uint32_t i = 0u; i < windows; ++i) {
      if (source_window_present(model,
                                source_window_signature(draft + i, novelty))) {
        return false;
      }
    }
  }
  *score_out = score;
  return true;
}

template <typename BigramKeyT, typename TrigramKeyT>
[[nodiscard]] __device__ inline bool appraise_reordered_causal_tail_draft(
    const BasicMutableModelView<BigramKeyT, TrigramKeyT>& model,
    const MutablePolicyState& policy, const MutableSelectionState& selection,
    const std::uint32_t* draft, std::uint32_t length,
    std::uint32_t tail_count, unsigned long long* score_out) {
  if (selection.evidence == 0u || model.relation_tail == nullptr ||
      tail_count < 2u || length != 3u + tail_count ||
      length > kMaxFrameUnits || selection.subject >= model.unit_count ||
      selection.predicate >= model.unit_count || selection.value >= model.unit_count) {
    return false;
  }
  if (draft[0] != selection.subject ||
      draft[1u + tail_count] != selection.predicate ||
      draft[2u + tail_count] != selection.value) {
    return false;
  }
  unsigned long long score = sat_add(selection.evidence,
                                     policy.causal_value_bonus);
  for (std::uint32_t i = 0u; i < length; ++i) {
    const std::uint32_t unit = draft[i];
    if (unit >= model.unit_count) return false;
    if (i >= 1u && i <= tail_count &&
        unit != model.relation_tail[i - 1u]) {
      return false;
    }
    score = sat_add(score, unit_state_score(model, policy, unit));
    if (i == 0u) continue;
    const std::uint32_t first = i > 1u ? draft[i - 2u] : kInvalidUnit;
    const unsigned long long edge_score =
        forward_score(model, policy, first, draft[i - 1u], unit);
    const bool frame_seam = i == 1u || i == 1u + tail_count;
    if (edge_score == 0ull && !frame_seam) return false;
    score = sat_add(score, edge_score);
  }
  const std::uint32_t novelty = policy.novelty_window;
  if (model.source_window_count != 0u) {
    if (novelty == 0u || novelty > kMaxNoveltyWindow || length < novelty)
      return false;
    for (std::uint32_t i = 0u; i + novelty <= length; ++i) {
      if (source_window_present(model,
                                source_window_signature(draft + i, novelty))) {
        return false;
      }
    }
  }
  *score_out = score;
  return true;
}

template <typename BigramKeyT, typename TrigramKeyT>
static __global__ void generate_and_appraise_kernel(
    BasicMutableModelView<BigramKeyT, TrigramKeyT> model, const MutableSelectionState* selection,
    MutableWorkspaceView workspace) {
  const std::uint32_t candidate = blockIdx.x * blockDim.x + threadIdx.x;
  if (candidate >= workspace.candidate_capacity)
    return;
  workspace.draft_lengths[candidate] = 0u;
  workspace.draft_scores[candidate] = 0ull;
  if (model.policy == nullptr || selection == nullptr || model.policy->enabled == 0u ||
      selection->intact == 0u || selection->value >= model.unit_count ||
      candidate >= model.policy->candidate_count ||
      candidate >= model.candidate_unit_count) {
    return;
  }
  const MutablePolicyState policy = *model.policy;
  std::uint32_t* draft = workspace.drafts + static_cast<std::size_t>(candidate) * kMaxFrameUnits;
  for (std::uint32_t i = 0u; i < kMaxFrameUnits; ++i)
    draft[i] = kInvalidUnit;

  // A conditioned relation already contains the resident subject, predicate,
  // and value that caused the answer. Close that proposition with resident
  // terminal matter before attempting a free role walk.
  if (selection->subject < model.unit_count &&
      selection->predicate < model.unit_count &&
      candidate < model.candidate_unit_count) {
    const std::uint32_t connector = model.candidate_units[candidate];
    const std::uint32_t connector_tail_count = model.relation_tail == nullptr ||
            model.relation_tail_count == nullptr
        ? 0u
        : (*model.relation_tail_count < kMaxFrameUnits - 3u
               ? *model.relation_tail_count
               : kMaxFrameUnits - 3u);
    const std::uint32_t value_vitality = model.unit_vitality == nullptr
        ? 0u : model.unit_vitality[selection->value];
    if (connector_tail_count >= 2u &&
        (connector == selection->value ||
         value_vitality >= policy.relation_tail_min_vitality) &&
        connector < model.unit_count) {
      draft[0] = selection->subject;
      draft[1] = selection->predicate;
      draft[2] = connector;
      for (std::uint32_t i = 0u; i < connector_tail_count; ++i) {
        draft[3u + i] = model.relation_tail[i];
      }
      unsigned long long connector_score = 0ull;
      const std::uint32_t connector_length = 3u + connector_tail_count;
      if (appraise_conditioned_tail_draft(
              model, policy, *selection, draft, connector_length, connector,
              connector_tail_count, &connector_score)) {
        workspace.draft_lengths[candidate] = connector_length;
        workspace.draft_scores[candidate] = connector_score;
        return;
      }
      if (connector == selection->value) {
        draft[0] = selection->subject;
        for (std::uint32_t i = 0u; i < connector_tail_count; ++i)
          draft[1u + i] = model.relation_tail[i];
        draft[1u + connector_tail_count] = selection->predicate;
        draft[2u + connector_tail_count] = selection->value;
        if (appraise_reordered_causal_tail_draft(
                model, policy, *selection, draft, connector_length,
                connector_tail_count, &connector_score)) {
          workspace.draft_lengths[candidate] = connector_length;
          workspace.draft_scores[candidate] = connector_score;
          return;
        }
      }
      for (std::uint32_t i = 0u; i < kMaxFrameUnits; ++i)
        draft[i] = kInvalidUnit;
    }
    const std::uint32_t terminal = model.candidate_units[candidate];
    const bool extended_relation = connector_tail_count >= 2u;
    if (!extended_relation && terminal < model.unit_count &&
        terminal != selection->subject &&
        terminal != selection->predicate && terminal != selection->value &&
        (model.unit_flags[terminal] & kUnitTerminalOnly) != 0u) {
      draft[0] = selection->subject;
      draft[1] = selection->predicate;
      draft[2] = selection->value;
      std::uint32_t length = 3u;
      const std::uint32_t resident_tail_count = model.relation_tail == nullptr ||
              model.relation_tail_count == nullptr
          ? 0u
          : (*model.relation_tail_count < kMaxFrameUnits - 4u
                 ? *model.relation_tail_count
                 : kMaxFrameUnits - 4u);
      const std::uint32_t tail_count = resident_tail_count != 0u &&
              resident_tail_count < model.relation_tail_capacity
          ? resident_tail_count : 0u;
      std::uint32_t skipped = kInvalidUnit;
      std::uint32_t replacement = kInvalidUnit;
      if (tail_count >= 2u && model.unit_vitality != nullptr) {
        std::uint32_t best_vitality = 0u;
        for (std::uint32_t i = 0u; i < tail_count; ++i) {
          const std::uint32_t unit = model.relation_tail[i];
          if (unit < model.unit_count && model.unit_vitality[unit] > best_vitality) {
            best_vitality = model.unit_vitality[unit];
            skipped = i;
          }
        }
        if (skipped != kInvalidUnit && skipped + 1u < tail_count &&
            model.unit_roles != nullptr) {
          const std::uint32_t original = model.relation_tail[skipped];
          const std::uint32_t previous = skipped == 0u
              ? selection->value : model.relation_tail[skipped - 1u];
          const std::uint32_t next = model.relation_tail[skipped + 1u];
          const std::uint32_t original_role = model.unit_roles[original].role;
          unsigned long long best_support = 0ull;
          for (std::uint32_t index = 0u;
               index < model.candidate_unit_count; ++index) {
            const std::uint32_t unit = model.candidate_units[index];
            if (unit >= model.unit_count || unit == original ||
                unit == selection->subject || unit == selection->predicate ||
                unit == selection->value ||
                (model.unit_flags[unit] & kUnitClosure) != 0u ||
                model.unit_roles[unit].role != original_role) {
              continue;
            }
            const unsigned long long left =
                role_bigram_count(model, previous, unit);
            const unsigned long long right =
                role_bigram_count(model, unit, next);
            if (left == 0ull || right == 0ull) continue;
            const unsigned long long support = sat_add(left, right);
            if (support > best_support ||
                (support == best_support && support != 0ull && unit < replacement)) {
              replacement = unit;
              best_support = support;
            }
          }
        }
      }
      for (std::uint32_t i = 0u; i < tail_count; ++i) {
        const std::uint32_t unit = i == skipped && replacement != kInvalidUnit
            ? replacement : model.relation_tail[i];
        if (i == skipped && replacement == kInvalidUnit) continue;
        if (unit >= model.unit_count || draft_contains(draft, length, unit)) continue;
        draft[length++] = unit;
      }
      const bool exhausted_tail = tail_count != 0u;
      if (!exhausted_tail &&
          (model.unit_flags[draft[length - 1u]] & kUnitClosure) == 0u)
        draft[length++] = terminal;
      unsigned long long score = 0ull;
      if (appraise_draft(model, policy, *selection, draft, length, &score)) {
        workspace.draft_lengths[candidate] = length;
        workspace.draft_scores[candidate] = score;
        return;
      }
      draft[0] = selection->subject;
      draft[1] = selection->predicate;
      draft[2] = selection->value;
      draft[3] = terminal;
      score = 0ull;
      if (appraise_draft(model, policy, *selection, draft, 4u, &score)) {
        workspace.draft_lengths[candidate] = 4u;
        workspace.draft_scores[candidate] = score;
        return;
      }
    }
    for (std::uint32_t i = 0u; i < kMaxFrameUnits; ++i)
      draft[i] = kInvalidUnit;
  }

  std::uint32_t center = policy.max_left_units;
  if (center >= kMaxFrameUnits)
    center = kMaxFrameUnits - 1u;
  std::uint32_t left = center;
  std::uint32_t right = center;
  draft[center] = selection->value;

  const std::uint32_t max_left = policy.max_left_units < center ? policy.max_left_units : center;
  for (std::uint32_t step = 0u; step < max_left; ++step) {
    const std::uint32_t chosen =
        choose_unit<false>(model, policy, draft, left, right, candidate, step);
    if (chosen == kInvalidUnit)
      break;
    draft[--left] = chosen;
  }

  const std::uint32_t max_right = policy.max_right_units < kMaxFrameUnits - 1u - right
                                      ? policy.max_right_units
                                      : kMaxFrameUnits - 1u - right;
  for (std::uint32_t step = 0u; step < max_right; ++step) {
    const std::uint32_t chosen =
        choose_unit<true>(model, policy, draft, left, right, candidate, step);
    if (chosen == kInvalidUnit)
      break;
    draft[++right] = chosen;
    if ((model.unit_flags[chosen] & kUnitClosure) != 0u)
      break;
  }

  const std::uint32_t length = right - left + 1u;
  for (std::uint32_t i = 0u; i < length; ++i)
    draft[i] = draft[left + i];
  for (std::uint32_t i = length; i < kMaxFrameUnits; ++i) {
    draft[i] = kInvalidUnit;
  }
  unsigned long long score = 0ull;
  if (!appraise_draft(model, policy, *selection, draft, length, &score))
    return;
  workspace.draft_lengths[candidate] = length;
  workspace.draft_scores[candidate] = score;
}

struct DecodeWinner {
  unsigned long long score = 0ull;
  std::uint32_t candidate = kInvalidUnit;
};

template <typename BigramKeyT, typename TrigramKeyT>
static __global__ void select_and_decode_kernel(
    BasicMutableModelView<BigramKeyT, TrigramKeyT> model, const MutableSelectionState* selection,
    MutableWorkspaceView workspace, Result* result) {
  const std::uint32_t lane = threadIdx.x;
  if (result == nullptr || model.policy == nullptr || selection == nullptr ||
      model.policy->enabled == 0u || selection->intact == 0u) {
    if (lane == 0u && result != nullptr)
      *result = {};
    return;
  }
  extern __shared__ DecodeWinner winners[];
  DecodeWinner local{};
  for (std::uint32_t candidate = lane; candidate < workspace.candidate_capacity;
       candidate += blockDim.x) {
    const unsigned long long score = workspace.draft_scores[candidate];
    if (score > local.score ||
        (score == local.score && score != 0ull &&
         (local.candidate == kInvalidUnit || candidate < local.candidate))) {
      local.score = score;
      local.candidate = candidate;
    }
  }
  winners[lane] = local;
  __syncthreads();
  for (std::uint32_t stride = blockDim.x / 2u; stride != 0u; stride >>= 1u) {
    if (lane < stride) {
      const DecodeWinner other = winners[lane + stride];
      DecodeWinner& current = winners[lane];
      if (other.score > current.score ||
          (other.score == current.score && other.score != 0ull &&
           (current.candidate == kInvalidUnit || other.candidate < current.candidate)))
        current = other;
    }
    __syncthreads();
  }
  if (lane != 0u)
    return;
  *result = {};
  const std::uint32_t winner = winners[0].candidate;
  const unsigned long long best = winners[0].score;
  if (winner == kInvalidUnit)
    return;

  const std::uint32_t length = workspace.draft_lengths[winner];
  if (length > workspace.output_unit_capacity)
    return;
  const std::uint32_t* draft = workspace.drafts + static_cast<std::size_t>(winner) * kMaxFrameUnits;
  std::uint32_t byte_count = 0u;
  std::uint32_t value_position = kInvalidUnit;
  bool closed = false;
  for (std::uint32_t i = 0u; i < length; ++i) {
    const std::uint32_t unit = draft[i];
    workspace.output_units[i] = unit;
    if (unit == selection->value)
      value_position = i;
    closed |= (model.unit_flags[unit] & kUnitClosure) != 0u;
    for (std::uint32_t offset = 0u; offset < model.unit_lengths[unit]; ++offset) {
      if (byte_count >= workspace.output_byte_capacity)
        return;
      workspace.output_bytes[byte_count++] = unit_byte(model, unit, offset);
    }
  }
  if (value_position == kInvalidUnit && model.relation_tail_count != nullptr &&
      *model.relation_tail_count >= 2u &&
      length == 3u + *model.relation_tail_count) {
    value_position = 2u;
  }
  result->ready = 1u;
  result->candidate = winner;
  result->unit_count = length;
  result->byte_count = byte_count;
  result->value_position = value_position;
  result->closed = closed ? 1u : 0u;
  result->source_novel = 1u;
  result->policy_revision = model.policy->revision;
  result->selection_revision = selection->revision;
  result->score_low = static_cast<std::uint32_t>(best);
  result->score_high = static_cast<std::uint32_t>(best >> 32u);
}

template <typename BigramKeyT, typename TrigramKeyT>
inline cudaError_t launch_resident_answer_frame(
    BasicMutableModelView<BigramKeyT, TrigramKeyT> model, const MutableSelectionState* selection,
    MutableWorkspaceView workspace, Result* result, cudaStream_t stream = nullptr) {
  if (selection == nullptr || result == nullptr || model.policy == nullptr ||
      model.unit_lengths == nullptr || model.unit_content == nullptr || model.unit_words == 0u ||
      model.unit_flags == nullptr || model.unit_count == 0u || model.candidate_units == nullptr ||
      model.candidate_unit_count == 0u ||
      (model.bigram_count != 0u && (model.bigrams == nullptr || model.bigram_counts == nullptr)) ||
      (model.trigram_count != 0u &&
       (model.trigrams == nullptr || model.trigram_counts == nullptr)) ||
      (model.source_window_count != 0u && model.source_windows == nullptr) ||
      workspace.drafts == nullptr || workspace.draft_lengths == nullptr ||
      workspace.draft_scores == nullptr || workspace.output_units == nullptr ||
      workspace.output_bytes == nullptr || workspace.candidate_capacity == 0u ||
      workspace.output_unit_capacity == 0u || workspace.output_byte_capacity == 0u) {
    return cudaErrorInvalidValue;
  }
  const std::uint32_t threads = 128u;
  const std::uint32_t blocks = (workspace.candidate_capacity + threads - 1u) / threads;
  generate_and_appraise_kernel<<<blocks, threads, 0u, stream>>>(model, selection, workspace);
  select_and_decode_kernel<<<1u, threads, threads * sizeof(DecodeWinner), stream>>>(
      model, selection, workspace, result);
  return cudaGetLastError();
}

static __global__ void lesion_answer_frame_policy_kernel(MutablePolicyState* policy) {
  if (blockIdx.x == 0u && threadIdx.x == 0u && policy != nullptr) {
    policy->enabled = 0u;
    ++policy->lesion_events;
    ++policy->revision;
  }
}

static __global__ void lesion_answer_frame_selection_kernel(MutableSelectionState* selection) {
  if (blockIdx.x == 0u && threadIdx.x == 0u && selection != nullptr) {
    selection->intact = 0u;
    selection->value = kInvalidUnit;
    selection->subject = kInvalidUnit;
    selection->predicate = kInvalidUnit;
    ++selection->revision;
  }
}

static __global__ void lesion_answer_frame_activation_kernel(std::uint32_t* activation,
                                                             std::uint32_t count) {
  const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (activation != nullptr && index < count)
    activation[index] = 0u;
}

static __global__ void lesion_answer_frame_role_grammar_kernel(std::uint64_t* role_bigrams,
                                                               std::size_t bigram_count,
                                                               std::uint64_t* role_trigrams,
                                                               std::size_t trigram_count,
                                                               MutablePolicyState* policy) {
  const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (role_bigrams != nullptr && index < bigram_count)
    role_bigrams[index] = 0ull;
  if (role_trigrams != nullptr && index < trigram_count)
    role_trigrams[index] = 0ull;
  if (index == 0u && policy != nullptr) {
    ++policy->lesion_events;
    ++policy->revision;
  }
}

}  // namespace substrate::bcc32::resident_answer_frame
