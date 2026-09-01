#ifndef HARDWARE_NATIVE_DIRECT_ADULT_DISCOURSE_FOCUS_CUH
#define HARDWARE_NATIVE_DIRECT_ADULT_DISCOURSE_FOCUS_CUH

#include <cstdint>
#include <type_traits>

#include "hardware_native/direct_exact_history.cuh"

namespace substrate::direct_network {

inline constexpr std::uint32_t kDirectDiscourseFocusCapacity = 128u;
inline constexpr std::uint32_t kDirectDiscourseRelationFocusCapacity = 64u;
inline constexpr std::uint32_t kDirectDiscourseRelationRecipeCapacity = 8u;

enum class DirectDiscourseRelationSelectionStatus : std::uint32_t {
  none = 0u,
  unique = 1u,
  ambiguous = 2u,
};

struct DirectDiscourseRelationFocusEntry {
  std::uint64_t relation_identity;
  std::uint64_t recruitment_identity;
  std::uint64_t network_identity;
  std::uint64_t last_action_ticket;
  std::uint64_t source_identity;
  std::uint64_t context_identity;
  std::uint64_t logical_recipe_ids[kDirectDiscourseRelationRecipeCapacity];
  std::uint64_t revision_identities[kDirectDiscourseRelationRecipeCapacity];
  std::int64_t credit_q16;
  std::uint32_t last_tick;
  std::uint32_t positive_settlements;
  std::uint32_t recipe_count;
  std::uint32_t active;
};

struct DirectDiscourseFocusEntry {
  std::uint64_t recruitment_identity;
  std::uint64_t network_identity;
  std::uint64_t last_action_ticket;
  std::uint64_t source_identity;
  std::uint64_t context_identity;
  std::uint32_t last_tick;
  std::uint32_t positive_settlements;
  std::int64_t credit_q16;
  std::uint32_t active;
  std::uint32_t reserved;
};

struct DirectDiscourseFocusState {
  DirectDiscourseFocusEntry entries[kDirectDiscourseFocusCapacity];
  DirectDiscourseRelationFocusEntry
      relation_entries[kDirectDiscourseRelationFocusCapacity];
  std::uint32_t count;
  std::uint32_t cursor;
  std::uint32_t relation_count;
  std::uint32_t relation_reserved;
  std::uint64_t current_source_identity;
  std::uint64_t current_context_identity;
  std::uint32_t current_source_tick;
  std::uint32_t reserved;
  std::uint64_t revision_identity;
};

static_assert(std::is_trivially_copyable_v<DirectDiscourseFocusEntry>);
static_assert(std::is_trivially_copyable_v<DirectDiscourseRelationFocusEntry>);
static_assert(std::is_trivially_copyable_v<DirectDiscourseFocusState>);

__host__ __device__ inline std::uint64_t direct_discourse_focus_fold(
    std::uint64_t h, std::uint64_t value) {
  return exact_history_fold_word(h, value);
}

// A discourse relation is a compact set of consequence-participating current
// RecipeRevisions. It contains no Occurrence identity, surface bytes, lexical
// anchor, or semantic label. Canonical ordering makes equality independent of
// frozen action-link order while keeping revision staleness observable.
__host__ __device__ inline std::uint64_t direct_discourse_relation_identity(
    const std::uint64_t* logical_recipe_ids,
    const std::uint64_t* revision_identities, std::uint32_t recipe_count) {
  if (logical_recipe_ids == nullptr || revision_identities == nullptr ||
      recipe_count == 0u ||
      recipe_count > kDirectDiscourseRelationRecipeCapacity)
    return 0u;
  std::uint64_t identity = direct_discourse_focus_fold(
      0x64736372656c7631ull, recipe_count);
  for (std::uint32_t i = 0u; i < recipe_count; ++i) {
    if (logical_recipe_ids[i] == 0u || revision_identities[i] == 0u)
      return 0u;
    // Logical Recipe identity supplies cross-revision continuity. The exact
    // revision remains payload and must be advanced through authenticated
    // revision ancestry before it can constrain a later current Network.
    if (i != 0u && logical_recipe_ids[i - 1u] >= logical_recipe_ids[i])
      return 0u;
    identity = direct_discourse_focus_fold(identity, logical_recipe_ids[i]);
  }
  return identity == 0u ? 1u : identity;
}

__host__ __device__ inline bool direct_discourse_relation_entry_valid(
    const DirectDiscourseRelationFocusEntry& entry) {
  if (entry.relation_identity == 0u || entry.recipe_count == 0u ||
      entry.recipe_count > kDirectDiscourseRelationRecipeCapacity ||
      ((entry.source_identity == 0u) != (entry.context_identity == 0u)))
    return false;
  return direct_discourse_relation_identity(
             entry.logical_recipe_ids, entry.revision_identities,
             entry.recipe_count) == entry.relation_identity;
}

__host__ __device__ inline bool direct_discourse_relation_payload_equal(
    const DirectDiscourseRelationFocusEntry& left,
    const DirectDiscourseRelationFocusEntry& right) {
  if (left.relation_identity != right.relation_identity ||
      left.recipe_count != right.recipe_count)
    return false;
  for (std::uint32_t i = 0u; i < left.recipe_count; ++i)
    if (left.logical_recipe_ids[i] != right.logical_recipe_ids[i] ||
        left.revision_identities[i] != right.revision_identities[i])
      return false;
  return true;
}

// Advance one already-resident relation to post-consequence RecipeRevisions.
// The caller must have proven the exact revision ancestry. A missing entry is
// a valid first settlement; a mismatched resident version fails closed.
__host__ __device__ inline bool direct_discourse_relation_focus_rebind(
    DirectDiscourseFocusState* state,
    const std::uint64_t* logical_recipe_ids,
    const std::uint64_t* prior_revision_identities,
    const std::uint64_t* next_revision_identities, std::uint32_t recipe_count,
    std::uint64_t source_identity, std::uint64_t context_identity) {
  if (state == nullptr || logical_recipe_ids == nullptr ||
      prior_revision_identities == nullptr || next_revision_identities == nullptr ||
      ((source_identity == 0u) != (context_identity == 0u)))
    return false;
  const std::uint64_t relation_identity = direct_discourse_relation_identity(
      logical_recipe_ids, prior_revision_identities, recipe_count);
  if (relation_identity == 0u ||
      direct_discourse_relation_identity(
          logical_recipe_ids, next_revision_identities, recipe_count) !=
          relation_identity ||
      state->relation_count > kDirectDiscourseRelationFocusCapacity)
    return false;
  for (std::uint32_t i = 0u; i < state->relation_count; ++i) {
    auto& entry = state->relation_entries[i];
    if (entry.relation_identity != relation_identity ||
        entry.source_identity != source_identity ||
        entry.context_identity != context_identity)
      continue;
    if (!direct_discourse_relation_entry_valid(entry) ||
        entry.recipe_count != recipe_count)
      return false;
    for (std::uint32_t r = 0u; r < recipe_count; ++r)
      if (entry.logical_recipe_ids[r] != logical_recipe_ids[r] ||
          entry.revision_identities[r] != prior_revision_identities[r])
        return false;
    for (std::uint32_t r = 0u; r < recipe_count; ++r)
      entry.revision_identities[r] = next_revision_identities[r];
    return direct_discourse_relation_entry_valid(entry);
  }
  return true;
}

// Consequence can keep a compact relation active across episodes, but this
// state is retrieval pressure only. It never creates a current Occurrence,
// Network member, or causal credit. Negative settlement can retire an already
// learned relation; it cannot create a new negative "fact" entry.
__host__ __device__ inline bool direct_discourse_relation_focus_note(
    DirectDiscourseFocusState* state, const DirectExactHistoryRecord& record,
    const std::uint64_t* logical_recipe_ids,
    const std::uint64_t* revision_identities, std::uint32_t recipe_count,
    std::uint64_t source_identity, std::uint64_t context_identity) {
  if (state == nullptr || record.kind != DirectExactHistoryKind::network_credit ||
      record.identity == 0u || record.parent_identity == 0u ||
      record.resource_delta == 0 ||
      ((source_identity == 0u) != (context_identity == 0u)))
    return false;
  const std::uint64_t recruitment_identity =
      (static_cast<std::uint64_t>(record.subject) << 32u) | record.source;
  const std::uint64_t relation_identity = direct_discourse_relation_identity(
      logical_recipe_ids, revision_identities, recipe_count);
  if (recruitment_identity == 0u || relation_identity == 0u ||
      state->relation_count > kDirectDiscourseRelationFocusCapacity)
    return false;

  std::uint32_t slot = kDirectDiscourseRelationFocusCapacity;
  for (std::uint32_t i = 0u; i < state->relation_count; ++i) {
    const auto& candidate = state->relation_entries[i];
    if (candidate.relation_identity != relation_identity ||
        candidate.source_identity != source_identity ||
        candidate.context_identity != context_identity)
      continue;
    if (!direct_discourse_relation_entry_valid(candidate) ||
        candidate.recipe_count != recipe_count)
      return false;
    for (std::uint32_t r = 0u; r < recipe_count; ++r)
      if (candidate.logical_recipe_ids[r] != logical_recipe_ids[r] ||
          candidate.revision_identities[r] != revision_identities[r])
        return false;
    slot = i;
    break;
  }
  if (slot == kDirectDiscourseRelationFocusCapacity) {
    if (record.resource_delta <= 0) return false;
    if (state->relation_count < kDirectDiscourseRelationFocusCapacity) {
      slot = state->relation_count++;
    } else {
      slot = 0u;
      for (std::uint32_t i = 1u; i < state->relation_count; ++i)
        if (state->relation_entries[i].last_tick <
            state->relation_entries[slot].last_tick)
          slot = i;
    }
    state->relation_entries[slot] = {};
    auto& created = state->relation_entries[slot];
    created.relation_identity = relation_identity;
    created.source_identity = source_identity;
    created.context_identity = context_identity;
    created.recipe_count = recipe_count;
    for (std::uint32_t r = 0u; r < recipe_count; ++r) {
      created.logical_recipe_ids[r] = logical_recipe_ids[r];
      created.revision_identities[r] = revision_identities[r];
    }
  }

  auto& entry = state->relation_entries[slot];
  entry.recruitment_identity = recruitment_identity;
  entry.network_identity = record.parent_identity;
  entry.last_action_ticket = record.identity;
  entry.last_tick = record.resident_tick;
  const std::int64_t delta = record.resource_delta;
  constexpr std::int64_t kMaxI64 = 0x7fffffffffffffffll;
  constexpr std::int64_t kMinI64 = -0x7fffffffffffffffll - 1ll;
  if (delta > 0 && entry.credit_q16 > kMaxI64 - delta)
    entry.credit_q16 = kMaxI64;
  else if (delta < 0 && entry.credit_q16 < kMinI64 - delta)
    entry.credit_q16 = kMinI64;
  else
    entry.credit_q16 += delta;
  if (delta > 0 && entry.positive_settlements != 0xffffffffu)
    ++entry.positive_settlements;
  entry.active = entry.credit_q16 > 0 ? 1u : 0u;
  state->revision_identity = direct_discourse_focus_fold(
      direct_discourse_focus_fold(state->revision_identity, relation_identity),
      static_cast<std::uint64_t>(entry.credit_q16));
  return true;
}

// Select a resident relation, not a historical recruitment. Different active
// relation identities are genuine unresolved alternatives and never lose to
// recency. Several entries for the same relation (for example after different
// transient recruitments) collapse only when their exact RecipeRevision payload
// agrees. A current authenticated source/context first demands an exact
// qualified relation. If the episode context changed, the same authenticated
// source is the next admissible retrieval tier; only when that source has no
// resident relation may source-independent relation matter transfer. Different
// authenticated sources never inherit one another's qualified relation.
__host__ __device__ inline DirectDiscourseRelationSelectionStatus
direct_discourse_relation_focus_select(
    const DirectDiscourseFocusState& state, std::uint32_t* selected_index) {
  if (selected_index != nullptr)
    *selected_index = kDirectDiscourseRelationFocusCapacity;
  if (state.relation_count > kDirectDiscourseRelationFocusCapacity)
    return DirectDiscourseRelationSelectionStatus::ambiguous;
  bool have_exact_source_context = false;
  bool have_same_source = false;
  for (std::uint32_t i = 0u; i < state.relation_count; ++i) {
    const auto& candidate = state.relation_entries[i];
    if (candidate.active == 0u || candidate.credit_q16 <= 0) continue;
    if (!direct_discourse_relation_entry_valid(candidate))
      return DirectDiscourseRelationSelectionStatus::ambiguous;
    if (state.current_source_identity == 0u ||
        candidate.source_identity != state.current_source_identity)
      continue;
    have_same_source = true;
    if (candidate.context_identity == state.current_context_identity)
      have_exact_source_context = true;
  }

  const DirectDiscourseRelationFocusEntry* selected = nullptr;
  std::uint32_t selected_slot = kDirectDiscourseRelationFocusCapacity;
  for (std::uint32_t i = 0u; i < state.relation_count; ++i) {
    const auto& candidate = state.relation_entries[i];
    if (candidate.active == 0u || candidate.credit_q16 <= 0) continue;
    if (state.current_source_identity != 0u) {
      if (have_exact_source_context) {
        if (candidate.source_identity != state.current_source_identity ||
            candidate.context_identity != state.current_context_identity)
          continue;
      } else if (have_same_source) {
        if (candidate.source_identity != state.current_source_identity)
          continue;
      } else if (candidate.source_identity != 0u ||
                 candidate.context_identity != 0u) {
        continue;
      }
    }
    if (selected == nullptr) {
      selected = &candidate;
      selected_slot = i;
      continue;
    }
    if (!direct_discourse_relation_payload_equal(*selected, candidate))
      return DirectDiscourseRelationSelectionStatus::ambiguous;
    if (candidate.last_tick > selected->last_tick ||
        (candidate.last_tick == selected->last_tick &&
         candidate.positive_settlements > selected->positive_settlements)) {
      selected = &candidate;
      selected_slot = i;
    }
  }
  if (selected == nullptr)
    return DirectDiscourseRelationSelectionStatus::none;
  if (selected_index != nullptr) *selected_index = selected_slot;
  return DirectDiscourseRelationSelectionStatus::unique;
}

__host__ __device__ inline void direct_discourse_focus_note(
    DirectDiscourseFocusState* state, const DirectExactHistoryRecord& record) {
  if (state == nullptr || record.kind != DirectExactHistoryKind::network_credit ||
      record.identity == 0u || record.parent_identity == 0u)
    return;
  const std::uint64_t recruitment_identity =
      (static_cast<std::uint64_t>(record.subject) << 32u) | record.source;
  if (recruitment_identity == 0u) return;
  std::uint32_t slot = kDirectDiscourseFocusCapacity;
  for (std::uint32_t i = 0u; i < state->count; ++i)
    if (state->entries[i].recruitment_identity == recruitment_identity) {
      slot = i;
      break;
    }
  if (slot == kDirectDiscourseFocusCapacity) {
    if (state->count < kDirectDiscourseFocusCapacity) {
      slot = state->count++;
    } else {
      slot = 0u;
      for (std::uint32_t i = 1u; i < state->count; ++i)
        if (state->entries[i].last_tick < state->entries[slot].last_tick)
          slot = i;
    }
    state->entries[slot] = {};
    state->entries[slot].recruitment_identity = recruitment_identity;
  }
  auto& entry = state->entries[slot];
  entry.network_identity = record.parent_identity;
  entry.last_action_ticket = record.identity;
  entry.last_tick = record.resident_tick;
  entry.credit_q16 = static_cast<std::int64_t>(record.incarnation_after);
  if (record.resource_delta > 0) {
    if (entry.positive_settlements != 0xffffffffu) ++entry.positive_settlements;
    entry.active = 1u;
  } else if (record.resource_delta < 0 && record.incarnation_after <= 0) {
    entry.active = 0u;
  }
  state->revision_identity = direct_discourse_focus_fold(
      direct_discourse_focus_fold(state->revision_identity, recruitment_identity),
      static_cast<std::uint64_t>(record.incarnation_after));
}

__device__ inline void direct_discourse_focus_assimilate(
    DirectDiscourseFocusState* state, const DirectExactHistoryRecord* records,
    std::uint32_t count) {
  if (state == nullptr || records == nullptr) return;
  if (count < state->cursor) state->cursor = 0u;
  for (std::uint32_t i = state->cursor; i < count; ++i) {
    const auto& record = records[i];
    direct_discourse_focus_note(state, record);
    if (record.kind == DirectExactHistoryKind::source_assertion &&
        record.identity != 0u && record.parent_identity != 0u &&
        (record.flags & kDirectHistoryVerifiedObservation) != 0u &&
        record.incarnation_before != 0u) {
      const std::uint64_t session = static_cast<std::uint64_t>(record.source) |
          (static_cast<std::uint64_t>(record.subject) << 32u);
      const std::uint64_t ingress = static_cast<std::uint64_t>(record.value) |
          (static_cast<std::uint64_t>(record.context) << 32u);
      state->current_source_identity = record.incarnation_before;
      state->current_context_identity = exact_history_fold_word(
          exact_history_fold_word(0x6173737274437874ull, session), ingress);
      if (state->current_context_identity == 0u) state->current_context_identity = 1u;
      state->current_source_tick = record.resident_tick;
    }
  }
  state->cursor = count;
}

__host__ __device__ inline bool direct_discourse_focus_unresolved(
    const DirectDiscourseFocusState& state) {
  if (state.relation_count != 0u)
    return direct_discourse_relation_focus_select(state, nullptr) !=
        DirectDiscourseRelationSelectionStatus::unique;
  if (state.current_source_identity == 0u) return false;
  std::uint32_t matches = 0u;
  for (std::uint32_t i = 0u; i < state.count; ++i) {
    const auto& candidate = state.entries[i];
    if (candidate.active != 0u && candidate.credit_q16 > 0 &&
        candidate.source_identity == state.current_source_identity &&
        candidate.context_identity == state.current_context_identity)
      ++matches;
  }
  return matches != 1u;
}

__host__ __device__ inline std::uint64_t direct_discourse_focus_select(
    const DirectDiscourseFocusState& state) {
  const DirectDiscourseFocusEntry* selected = nullptr;
  bool tied = false;
  bool source_conditioned = false;
  if (state.current_source_identity != 0u) {
    for (std::uint32_t i = 0u; i < state.count; ++i) {
      const auto& candidate = state.entries[i];
      source_conditioned |= candidate.active != 0u && candidate.credit_q16 > 0 &&
          candidate.source_identity == state.current_source_identity &&
          candidate.context_identity == state.current_context_identity;
    }
  }
  for (std::uint32_t i = 0u; i < state.count; ++i) {
    const auto& candidate = state.entries[i];
    if (candidate.active == 0u || candidate.recruitment_identity == 0u ||
        candidate.credit_q16 <= 0 ||
        (source_conditioned &&
         (candidate.source_identity != state.current_source_identity ||
          candidate.context_identity != state.current_context_identity)))
      continue;
    if (selected == nullptr || candidate.last_tick > selected->last_tick ||
        (candidate.last_tick == selected->last_tick &&
         candidate.positive_settlements > selected->positive_settlements)) {
      selected = &candidate;
      tied = false;
    } else if (candidate.last_tick == selected->last_tick &&
               candidate.positive_settlements == selected->positive_settlements &&
               candidate.recruitment_identity != selected->recruitment_identity) {
      tied = true;
    }
  }
  return selected != nullptr && !tied ? selected->recruitment_identity : 0u;
}

}  // namespace substrate::direct_network

#endif
