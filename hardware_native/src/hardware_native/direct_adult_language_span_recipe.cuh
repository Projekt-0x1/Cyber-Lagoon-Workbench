#ifndef HARDWARE_NATIVE_DIRECT_ADULT_LANGUAGE_SPAN_RECIPE_CUH
#define HARDWARE_NATIVE_DIRECT_ADULT_LANGUAGE_SPAN_RECIPE_CUH

#include <cstdint>
#include <type_traits>

#include "hardware_native/direct_adult_relational_sequence_bridge.cuh"
#include "hardware_native/direct_exact_history.cuh"

#if defined(__CUDACC__)
#define DIRECT_LANGUAGE_SPAN_HD __host__ __device__
#else
#define DIRECT_LANGUAGE_SPAN_HD
#endif

namespace substrate::direct_adult_core {

// Outer-language p(n+1). Persistent rows store only structural mathematics and
// exact RecipeRevision witnesses; current surface units exist only in the
// transient RelSeq occurrence used to unfold the Recipe.
inline constexpr std::uint32_t kResidentLanguageSpanMaxChildren = 16u;
inline constexpr std::uint32_t kResidentLanguageSpanCandidateCapacity = 256u;
inline constexpr std::uint32_t kResidentLanguageSpanRecipeCapacity = 1024u;
inline constexpr std::uint32_t kResidentLanguageSpanMaxUnfoldNodes = 31u;

struct ResidentLanguageSpanCandidateV1 {
  std::uint64_t identity;
  std::uint64_t action_ticket_id;
  std::uint64_t recruitment_identity;
  std::uint64_t child_logical_recipe_ids[kResidentLanguageSpanMaxChildren];
  std::uint64_t child_revision_identities[kResidentLanguageSpanMaxChildren];
  std::uint32_t child_count;
  std::uint32_t active;
};

struct ResidentLanguageSpanRecipeV1 {
  // `identity` is the logical p(n+1) identity. The flat child sequence remains
  // the rematerialization witness even when `components` call lower spans.
  std::uint64_t identity;
  std::uint64_t recruitment_identity;
  std::uint64_t child_logical_recipe_ids[kResidentLanguageSpanMaxChildren];
  std::uint64_t child_revision_identities[kResidentLanguageSpanMaxChildren];
  // Zero means this component is one flat leaf. Non-zero names a learned child
  // span whose flat witness must exactly cover `component_extent` leaves.
  std::uint64_t component_span_identities[kResidentLanguageSpanMaxChildren];
  std::uint16_t component_extents[kResidentLanguageSpanMaxChildren];
  std::uint16_t child_count;
  std::uint16_t component_count;
  std::uint16_t rank;
  std::uint32_t support;
  std::uint32_t active;
};

struct ResidentLanguageSpanBankV1 {
  ResidentLanguageSpanCandidateV1 candidates[kResidentLanguageSpanCandidateCapacity];
  ResidentLanguageSpanRecipeV1 recipes[kResidentLanguageSpanRecipeCapacity];
  std::uint32_t candidate_count;
  std::uint32_t recipe_count;
  std::uint32_t history_cursor;
  std::uint32_t refusals;
  std::uint32_t promotions;
  std::uint32_t reserved;
};

static_assert(std::is_trivially_copyable_v<ResidentLanguageSpanCandidateV1> &&
              std::is_trivially_copyable_v<ResidentLanguageSpanRecipeV1> &&
              std::is_trivially_copyable_v<ResidentLanguageSpanBankV1>);

DIRECT_LANGUAGE_SPAN_HD inline std::uint64_t resident_language_span_identity(
    std::uint64_t recruitment_identity,
    const std::uint64_t* logical_recipe_ids,
    const std::uint64_t* revision_identities,
    std::uint32_t child_count) {
  using direct_network::exact_history_fold_word;
  if (recruitment_identity == 0u || logical_recipe_ids == nullptr ||
      revision_identities == nullptr || child_count < 2u ||
      child_count > kResidentLanguageSpanMaxChildren)
    return 0u;
  std::uint64_t identity = exact_history_fold_word(
      0x6c616e677370616eull, recruitment_identity);
  identity = exact_history_fold_word(identity, child_count);
  for (std::uint32_t i = 0u; i < child_count; ++i) {
    if (logical_recipe_ids[i] == 0u || revision_identities[i] == 0u)
      return 0u;
    identity = exact_history_fold_word(identity, logical_recipe_ids[i]);
    identity = exact_history_fold_word(identity, revision_identities[i]);
  }
  return identity == 0u ? 1u : identity;
}

DIRECT_LANGUAGE_SPAN_HD inline bool resident_language_span_witness_equal(
    const ResidentLanguageSpanRecipeV1& span,
    const std::uint64_t* logical_recipe_ids,
    const std::uint64_t* revision_identities,
    std::uint32_t begin, std::uint32_t count) {
  if (logical_recipe_ids == nullptr || revision_identities == nullptr ||
      span.active == 0u || span.child_count != count || count < 2u)
    return false;
  for (std::uint32_t i = 0u; i < count; ++i)
    if (span.child_logical_recipe_ids[i] != logical_recipe_ids[begin + i] ||
        span.child_revision_identities[i] != revision_identities[begin + i])
      return false;
  return true;
}

DIRECT_LANGUAGE_SPAN_HD inline bool record_resident_language_span_candidate(
    ResidentLanguageSpanBankV1* bank, std::uint64_t action_ticket_id,
    std::uint64_t recruitment_identity, const std::uint64_t* logical_recipe_ids,
    const std::uint64_t* revision_identities, std::uint32_t child_count) {
  if (bank == nullptr || action_ticket_id == 0u ||
      action_ticket_id == static_cast<std::uint64_t>(-1) ||
      child_count < 2u || child_count > kResidentLanguageSpanMaxChildren)
    return false;
  const std::uint64_t identity = resident_language_span_identity(
      recruitment_identity, logical_recipe_ids, revision_identities, child_count);
  if (identity == 0u) {
    ++bank->refusals;
    return false;
  }
  for (std::uint32_t i = 0u; i < bank->candidate_count; ++i) {
    auto& row = bank->candidates[i];
    if (row.active != 0u && row.action_ticket_id == action_ticket_id &&
        row.recruitment_identity == recruitment_identity) {
      if (row.identity == identity) return true;
      ++bank->refusals;
      return false;
    }
  }
  std::uint32_t slot = kResidentLanguageSpanCandidateCapacity;
  for (std::uint32_t i = 0u; i < bank->candidate_count; ++i)
    if (bank->candidates[i].active == 0u) {
      slot = i;
      break;
    }
  if (slot == kResidentLanguageSpanCandidateCapacity) {
    if (bank->candidate_count >= kResidentLanguageSpanCandidateCapacity) {
      ++bank->refusals;
      return false;
    }
    slot = bank->candidate_count++;
  }
  ResidentLanguageSpanCandidateV1 row{};
  row.identity = identity;
  row.action_ticket_id = action_ticket_id;
  row.recruitment_identity = recruitment_identity;
  row.child_count = child_count;
  row.active = 1u;
  for (std::uint32_t i = 0u; i < child_count; ++i) {
    row.child_logical_recipe_ids[i] = logical_recipe_ids[i];
    row.child_revision_identities[i] = revision_identities[i];
  }
  bank->candidates[slot] = row;
  return true;
}

// Choose a unique minimum-component decomposition using only Recipes that were
// already durable before this candidate. Ambiguous equal-cost decompositions do
// not acquire authority; the flat witness remains the representation.
DIRECT_LANGUAGE_SPAN_HD inline void resident_language_span_choose_components(
    const ResidentLanguageSpanBankV1& bank,
    const ResidentLanguageSpanCandidateV1& candidate,
    ResidentLanguageSpanRecipeV1* out) {
  if (out == nullptr) return;
  out->component_count = static_cast<std::uint16_t>(candidate.child_count);
  out->rank = 1u;
  for (std::uint32_t i = 0u; i < candidate.child_count; ++i) {
    out->component_extents[i] = 1u;
    out->component_span_identities[i] = 0u;
  }
  if (candidate.child_count < 3u) return;

  constexpr std::uint16_t kUnreachable = 0xffffu;
  std::uint16_t cost[kResidentLanguageSpanMaxChildren + 1u]{};
  std::uint8_t ways[kResidentLanguageSpanMaxChildren + 1u]{};
  std::uint16_t previous[kResidentLanguageSpanMaxChildren + 1u]{};
  std::uint16_t chosen_extent[kResidentLanguageSpanMaxChildren + 1u]{};
  std::uint16_t path_rank[kResidentLanguageSpanMaxChildren + 1u]{};
  std::uint64_t chosen_span[kResidentLanguageSpanMaxChildren + 1u]{};
  for (std::uint32_t i = 1u; i <= candidate.child_count; ++i)
    cost[i] = kUnreachable;
  ways[0] = 1u;

  for (std::uint32_t begin = 0u; begin < candidate.child_count; ++begin) {
    if (ways[begin] == 0u || cost[begin] == kUnreachable) continue;
    for (std::uint32_t extent = 1u; begin + extent <= candidate.child_count; ++extent) {
      std::uint64_t span_identity = 0u;
      std::uint16_t component_rank = 1u;
      if (extent > 1u) {
        const ResidentLanguageSpanRecipeV1* match = nullptr;
        std::uint32_t matches = 0u;
        for (std::uint32_t r = 0u; r < bank.recipe_count; ++r) {
          const auto& existing = bank.recipes[r];
          if (existing.active != 0u && existing.identity != candidate.identity &&
              existing.child_count == extent &&
              resident_language_span_witness_equal(
                  existing, candidate.child_logical_recipe_ids,
                  candidate.child_revision_identities, begin, extent)) {
            match = &existing;
            ++matches;
          }
        }
        if (matches != 1u || match == nullptr) continue;
        span_identity = match->identity;
        component_rank = static_cast<std::uint16_t>(match->rank + 1u);
      }
      const std::uint32_t end = begin + extent;
      const std::uint16_t next_cost = static_cast<std::uint16_t>(cost[begin] + 1u);
      const std::uint16_t next_rank =
          path_rank[begin] > component_rank ? path_rank[begin] : component_rank;
      if (cost[end] == kUnreachable || next_cost < cost[end]) {
        cost[end] = next_cost;
        ways[end] = ways[begin] > 1u ? 2u : ways[begin];
        previous[end] = static_cast<std::uint16_t>(begin);
        chosen_extent[end] = static_cast<std::uint16_t>(extent);
        chosen_span[end] = span_identity;
        path_rank[end] = next_rank;
      } else if (next_cost == cost[end]) {
        ways[end] = 2u;
      }
    }
  }

  const std::uint32_t end = candidate.child_count;
  if (ways[end] != 1u || cost[end] == kUnreachable ||
      cost[end] >= candidate.child_count)
    return;
  std::uint16_t extents[kResidentLanguageSpanMaxChildren]{};
  std::uint64_t spans[kResidentLanguageSpanMaxChildren]{};
  std::uint32_t component_count = 0u;
  std::uint32_t cursor = end;
  while (cursor != 0u && component_count < kResidentLanguageSpanMaxChildren) {
    extents[component_count] = chosen_extent[cursor];
    spans[component_count] = chosen_span[cursor];
    cursor = previous[cursor];
    ++component_count;
  }
  if (cursor != 0u || component_count != cost[end]) return;
  out->component_count = static_cast<std::uint16_t>(component_count);
  out->rank = path_rank[end];
  for (std::uint32_t i = 0u; i < component_count; ++i) {
    const std::uint32_t source = component_count - 1u - i;
    out->component_span_identities[i] = spans[source];
    out->component_extents[i] = extents[source];
  }
}

DIRECT_LANGUAGE_SPAN_HD inline bool promote_resident_language_span_candidate(
    ResidentLanguageSpanBankV1* bank,
    const direct_network::DirectExactHistoryRecord& record) {
  using direct_network::DirectExactHistoryKind;
  if (bank == nullptr || record.kind != DirectExactHistoryKind::network_credit ||
      record.identity == 0u || record.resource_delta <= 0 ||
      record.incarnation_after <= record.incarnation_before)
    return false;
  const std::uint64_t recruitment_identity =
      (static_cast<std::uint64_t>(record.subject) << 32u) | record.source;
  ResidentLanguageSpanCandidateV1* candidate = nullptr;
  std::uint32_t matches = 0u;
  for (std::uint32_t i = 0u; i < bank->candidate_count; ++i) {
    auto& row = bank->candidates[i];
    if (row.active != 0u && row.action_ticket_id == record.identity &&
        row.recruitment_identity == recruitment_identity) {
      candidate = &row;
      ++matches;
    }
  }
  if (matches != 1u || candidate == nullptr) return false;

  ResidentLanguageSpanRecipeV1* recipe = nullptr;
  for (std::uint32_t i = 0u; i < bank->recipe_count; ++i)
    if (bank->recipes[i].identity == candidate->identity) {
      recipe = &bank->recipes[i];
      break;
    }
  if (recipe == nullptr) {
    if (bank->recipe_count >= kResidentLanguageSpanRecipeCapacity) {
      ++bank->refusals;
      return false;
    }
    recipe = &bank->recipes[bank->recipe_count++];
    *recipe = {};
    recipe->identity = candidate->identity;
    recipe->recruitment_identity = candidate->recruitment_identity;
    recipe->child_count = static_cast<std::uint16_t>(candidate->child_count);
    recipe->active = 1u;
    for (std::uint32_t i = 0u; i < candidate->child_count; ++i) {
      recipe->child_logical_recipe_ids[i] = candidate->child_logical_recipe_ids[i];
      recipe->child_revision_identities[i] = candidate->child_revision_identities[i];
    }
    resident_language_span_choose_components(*bank, *candidate, recipe);
  } else {
    ResidentLanguageSpanRecipeV1 optimized = *recipe;
    resident_language_span_choose_components(*bank, *candidate, &optimized);
    if (optimized.component_count < recipe->component_count) {
      recipe->component_count = optimized.component_count;
      recipe->rank = optimized.rank;
      for (std::uint32_t i = 0u; i < optimized.component_count; ++i) {
        recipe->component_span_identities[i] = optimized.component_span_identities[i];
        recipe->component_extents[i] = optimized.component_extents[i];
      }
    }
  }
  if (recipe->support != 0xffffffffu) ++recipe->support;
  recipe->active = 1u;
  candidate->active = 0u;
  ++bank->promotions;
  return true;
}

DIRECT_LANGUAGE_SPAN_HD inline void assimilate_resident_language_span_credit(
    ResidentLanguageSpanBankV1* bank,
    const direct_network::DirectExactHistoryRecord* records, std::uint32_t count) {
  if (bank == nullptr || records == nullptr) return;
  if (count < bank->history_cursor) bank->history_cursor = 0u;
  for (std::uint32_t i = bank->history_cursor; i < count; ++i)
    (void)promote_resident_language_span_candidate(bank, records[i]);
  bank->history_cursor = count;
}

DIRECT_LANGUAGE_SPAN_HD inline const ResidentLanguageSpanRecipeV1*
resident_language_span_by_identity(const ResidentLanguageSpanBankV1& bank,
                                   std::uint64_t identity) {
  if (identity == 0u) return nullptr;
  const ResidentLanguageSpanRecipeV1* found = nullptr;
  for (std::uint32_t i = 0u; i < bank.recipe_count; ++i)
    if (bank.recipes[i].active != 0u && bank.recipes[i].identity == identity) {
      if (found != nullptr) return nullptr;
      found = &bank.recipes[i];
    }
  return found;
}

DIRECT_LANGUAGE_SPAN_HD inline const ResidentLanguageSpanRecipeV1*
select_resident_language_span_recipe(
    const ResidentLanguageSpanBankV1& bank, std::uint64_t recruitment_identity) {
  if (recruitment_identity == 0u) return nullptr;
  const ResidentLanguageSpanRecipeV1* selected = nullptr;
  for (std::uint32_t i = 0u; i < bank.recipe_count; ++i) {
    const auto& row = bank.recipes[i];
    if (row.active == 0u || row.recruitment_identity != recruitment_identity)
      continue;
    if (selected != nullptr && selected->identity != row.identity)
      return nullptr;
    selected = &row;
  }
  return selected;
}

DIRECT_LANGUAGE_SPAN_HD inline const ResidentLanguageSpanRecipeV1*
select_resident_language_span_recipe_by_children(
    const ResidentLanguageSpanBankV1& bank,
    const std::uint64_t* logical_recipe_ids,
    const std::uint64_t* revision_identities, std::uint32_t child_count) {
  if (logical_recipe_ids == nullptr || revision_identities == nullptr ||
      child_count < 2u || child_count > kResidentLanguageSpanMaxChildren)
    return nullptr;
  const ResidentLanguageSpanRecipeV1* selected = nullptr;
  for (std::uint32_t i = 0u; i < bank.recipe_count; ++i) {
    const auto& row = bank.recipes[i];
    if (!resident_language_span_witness_equal(
            row, logical_recipe_ids, revision_identities, 0u, child_count))
      continue;
    if (selected != nullptr && selected->identity != row.identity)
      return nullptr;
    selected = &row;
  }
  return selected;
}

// Consequence-aware higher-rank arbitration. Structural witness remains the
// admission gate. Whole-Network consequence credit is consulted only when more
// than one distinct learned span realizes the same current child witness; exact
// credit ties remain unresolved rather than letting array order choose language.
template <typename RecruitedNetworkStateT>
DIRECT_LANGUAGE_SPAN_HD inline const ResidentLanguageSpanRecipeV1*
select_resident_language_span_recipe_by_children_and_consequence(
    const ResidentLanguageSpanBankV1& bank,
    const RecruitedNetworkStateT& recruited,
    const std::uint64_t* logical_recipe_ids,
    const std::uint64_t* revision_identities, std::uint32_t child_count) {
  if (logical_recipe_ids == nullptr || revision_identities == nullptr ||
      child_count < 2u || child_count > kResidentLanguageSpanMaxChildren)
    return nullptr;
  const ResidentLanguageSpanRecipeV1* selected = nullptr;
  std::int64_t selected_credit = 0;
  bool have_credit = false;
  bool tied = false;
  std::uint32_t matches = 0u;
  for (std::uint32_t i = 0u; i < bank.recipe_count; ++i) {
    const auto& row = bank.recipes[i];
    if (!resident_language_span_witness_equal(
            row, logical_recipe_ids, revision_identities, 0u, child_count))
      continue;
    ++matches;
    std::int32_t incidence_index = -1;
    std::uint32_t incidence_matches = 0u;
    for (std::uint32_t j = 0u; j < recruited.incidence_count; ++j)
      if (recruited.incidences[j].recruitment_identity == row.recruitment_identity) {
        incidence_index = static_cast<std::int32_t>(j);
        ++incidence_matches;
      }
    if (incidence_matches > 1u) return nullptr;
    const std::int64_t credit = incidence_index >= 0
        ? recruited.incidences[static_cast<std::uint32_t>(incidence_index)].credit_q16
        : 0;
    if (!have_credit || credit > selected_credit) {
      selected = &row;
      selected_credit = credit;
      have_credit = true;
      tied = false;
    } else if (credit == selected_credit && selected != nullptr &&
               selected->identity != row.identity) {
      tied = true;
    }
  }
  if (matches <= 1u) return selected;
  return tied ? nullptr : selected;
}

DIRECT_LANGUAGE_SPAN_HD inline std::uint64_t resident_language_span_relation_identity(
    const ResidentLanguageSpanRecipeV1& span) {
  return direct_network::exact_history_fold_word(0x7370616e72656c31ull, span.identity);
}

DIRECT_LANGUAGE_SPAN_HD inline bool resident_language_span_relseq_body(
    const ResidentLanguageSpanRecipeV1& span, ResidentRelationalSequenceBody* body) {
  if (body == nullptr || span.active == 0u || span.identity == 0u ||
      span.component_count == 0u || span.component_count > span.child_count ||
      span.component_count > kResidentRelSeqMaxPorts)
    return false;
  ResidentRelationalSequenceBody out{};
  out.logical_recipe_id = span.identity;
  out.revision_identity = direct_network::exact_history_fold_word(
      0x7370616e72657631ull, span.identity);
  out.relation_identity = resident_language_span_relation_identity(span);
  out.piece_count = span.component_count;
  out.port_count = span.component_count;
  out.support = span.support;
  out.active = 1u;
  for (std::uint32_t i = 0u; i < span.component_count; ++i) {
    out.pieces[i].value = i;
    out.pieces[i].kind = static_cast<std::uint16_t>(
        direct_network::DirectRelSeqPieceKind::port);
  }
  if (!resident_relseq_body_valid(out)) return false;
  *body = out;
  return true;
}

struct ResidentLanguageSpanUnfoldScratchV1 {
  direct_network::DirectRelSeqRecipe recipes[kResidentLanguageSpanMaxUnfoldNodes];
  direct_network::DirectRelSeqOccurrence occurrences[kResidentLanguageSpanMaxUnfoldNodes];
  std::uint32_t count;
};

DIRECT_LANGUAGE_SPAN_HD inline bool resident_language_span_build_unfold_node(
    const ResidentLanguageSpanBankV1& bank,
    const ResidentLanguageSpanRecipeV1& span,
    const std::uint32_t* units, std::uint32_t unit_count,
    ResidentLanguageSpanUnfoldScratchV1* scratch,
    std::uint64_t* occurrence_identity) {
  if (units == nullptr || scratch == nullptr || occurrence_identity == nullptr ||
      unit_count != span.child_count || scratch->count >= kResidentLanguageSpanMaxUnfoldNodes)
    return false;
  ResidentRelationalSequenceBody body{};
  if (!resident_language_span_relseq_body(span, &body)) return false;
  const std::uint32_t slot = scratch->count++;
  auto& recipe = scratch->recipes[slot];
  recipe.logical_recipe_id = body.logical_recipe_id;
  recipe.relation_identity = body.relation_identity;
  recipe.piece_count = body.piece_count;
  recipe.port_count = body.port_count;
  recipe.support = body.support;
  recipe.active = body.active;
  for (std::uint32_t i = 0u; i < body.piece_count; ++i) recipe.pieces[i] = body.pieces[i];
  recipe.revision_identity = direct_network::direct_relseq_recipe_revision_identity(recipe);

  auto& occurrence = scratch->occurrences[slot];
  std::uint64_t oid = direct_network::exact_history_fold_word(
      0x7370616e6f636331ull, span.identity);
  for (std::uint32_t i = 0u; i < unit_count; ++i)
    oid = direct_network::exact_history_fold_word(oid, units[i]);
  oid = direct_network::exact_history_fold_word(oid, slot + 1u);
  occurrence.occurrence_identity = oid == 0u ? slot + 1u : oid;
  occurrence.logical_recipe_id = recipe.logical_recipe_id;
  occurrence.revision_identity = recipe.revision_identity;
  occurrence.binding_count = span.component_count;

  std::uint32_t begin = 0u;
  for (std::uint32_t component = 0u; component < span.component_count; ++component) {
    const std::uint32_t extent = span.component_extents[component];
    if (extent == 0u || begin + extent > unit_count) return false;
    auto& binding = occurrence.bindings[component];
    binding.formal_port = component;
    const std::uint64_t child_span = span.component_span_identities[component];
    if (child_span == 0u) {
      if (extent != 1u || units[begin] == 0u) return false;
      binding.unit_identity = units[begin];
    } else {
      const auto* child = resident_language_span_by_identity(bank, child_span);
      if (child == nullptr || child->child_count != extent)
        return false;
      for (std::uint32_t i = 0u; i < extent; ++i)
        if (child->child_logical_recipe_ids[i] != span.child_logical_recipe_ids[begin + i] ||
            child->child_revision_identities[i] != span.child_revision_identities[begin + i])
          return false;
      std::uint64_t child_oid = 0u;
      if (!resident_language_span_build_unfold_node(
              bank, *child, units + begin, extent, scratch, &child_oid))
        return false;
      binding.child_occurrence_identity = child_oid;
    }
    begin += extent;
  }
  if (begin != unit_count) return false;
  *occurrence_identity = occurrence.occurrence_identity;
  return true;
}

DIRECT_LANGUAGE_SPAN_HD inline bool evaluate_resident_language_span_from_units(
    const ResidentLanguageSpanBankV1& bank,
    const ResidentLanguageSpanRecipeV1& span,
    const std::uint32_t* units, std::uint32_t unit_count,
    direct_network::DirectRelSeqOutput* output) {
  if (output == nullptr) return false;
  *output = {};
  ResidentLanguageSpanUnfoldScratchV1 scratch{};
  std::uint64_t root = 0u;
  if (!resident_language_span_build_unfold_node(
          bank, span, units, unit_count, &scratch, &root))
    return false;
  return direct_network::direct_relseq_evaluate_occurrence(
      scratch.recipes, scratch.count, scratch.occurrences, scratch.count, root, output);
}

template <typename RecipeCellT>
DIRECT_LANGUAGE_SPAN_HD inline bool resident_language_span_children_current(
    const ResidentLanguageSpanRecipeV1& span, const RecipeCellT* cells,
    std::uint32_t cell_count) {
  if (span.active == 0u || span.identity == 0u || cells == nullptr ||
      span.child_count < 2u || span.child_count > kResidentLanguageSpanMaxChildren)
    return false;
  for (std::uint32_t child = 0u; child < span.child_count; ++child) {
    std::uint32_t matches = 0u;
    for (std::uint32_t i = 0u; i < cell_count; ++i)
      matches += cells[i].logical_recipe_id == span.child_logical_recipe_ids[child] &&
                 cells[i].revision_identity == span.child_revision_identities[child]
                     ? 1u
                     : 0u;
    if (matches != 1u) return false;
  }
  return true;
}

}  // namespace substrate::direct_adult_core

#undef DIRECT_LANGUAGE_SPAN_HD
#endif
