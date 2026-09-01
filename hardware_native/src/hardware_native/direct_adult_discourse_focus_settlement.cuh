#ifndef HARDWARE_NATIVE_DIRECT_ADULT_DISCOURSE_FOCUS_SETTLEMENT_CUH
#define HARDWARE_NATIVE_DIRECT_ADULT_DISCOURSE_FOCUS_SETTLEMENT_CUH

#include "hardware_native/direct_adult_core.cuh"
#include "hardware_native/direct_adult_discourse_focus.cuh"
#include "hardware_native/direct_adult_resident_language_runtime.cuh"
#include "hardware_native/direct_adult_source_epistemics.cuh"

namespace substrate::direct_adult_core {

// Resolve the exact RecipeRevision visible after this consequence transaction.
// A frozen action link names the version that actually participated. Only an
// unambiguous experience-revision chain from that same action may advance it.
DIRECT_ADULT_HD inline bool resident_discourse_settlement_final_revision(
    const DirectActionOccurrence& action,
    const DirectActionParticipationLink& link,
    const direct_network::DirectExactHistoryRecord& network_credit,
    const direct_network::DirectExactHistoryRecord* records,
    std::uint32_t record_count, std::uint64_t* final_revision_identity) {
  using namespace direct_network;
  if (records == nullptr || final_revision_identity == nullptr ||
      link.logical_recipe_id == 0u || link.revision_identity == 0u ||
      link.occurrence_identity == 0u || link.participation_identity == 0u ||
      action.action_ticket_id == 0u || action.emission_tick == 0u ||
      network_credit.kind != DirectExactHistoryKind::network_credit ||
      network_credit.identity != action.action_ticket_id)
    return false;
  std::uint64_t current = link.revision_identity;
  for (std::uint32_t depth = 0u; depth < record_count; ++depth) {
    std::uint64_t next = 0u;
    std::uint32_t matches = 0u;
    for (std::uint32_t i = 0u; i < record_count; ++i) {
      const auto& revision = records[i];
      if (revision.kind != DirectExactHistoryKind::recipe_revision ||
          revision.parent_identity != current || revision.identity == 0u ||
          revision.identity == current ||
          revision.subject != static_cast<std::uint32_t>(
              ResidentRecipeRevisionAuthority::experience) ||
          revision.incarnation_before != link.occurrence_identity ||
          revision.incarnation_after != link.participation_identity ||
          revision.resident_tick != network_credit.resident_tick ||
          revision.event_tick != action.emission_tick)
        continue;
      next = revision.identity;
      ++matches;
      if (matches > 1u) return false;
    }
    if (matches == 0u) {
      *final_revision_identity = current;
      return true;
    }
    current = next;
  }
  return false;
}

// Extract the consequence-revised learned relation without preserving transient
// Occurrence identity. Network members whose RecipeRevision did not change are
// surrounding action context, not automatically part of the durable relation.
// Repeated instances of one logical Recipe are one relation coordinate here;
// inconsistent versions of that coordinate fail.
DIRECT_ADULT_HD inline bool resident_discourse_settlement_relation_recipes(
    const DirectActionOccurrence& action,
    const DirectActionParticipationLink* action_links,
    std::uint32_t action_participant_capacity,
    const direct_network::DirectExactHistoryRecord& network_credit,
    const direct_network::DirectExactHistoryRecord* records,
    std::uint32_t record_count, std::uint64_t* logical_recipe_ids,
    std::uint64_t* prior_revision_identities,
    std::uint64_t* final_revision_identities, std::uint32_t* recipe_count) {
  using namespace direct_network;
  if (action_links == nullptr || records == nullptr ||
      logical_recipe_ids == nullptr || prior_revision_identities == nullptr ||
      final_revision_identities == nullptr || recipe_count == nullptr ||
      action.participant_offset == kInvalidIndex ||
      action.participant_count == 0u ||
      action.participant_count > action_participant_capacity)
    return false;
  *recipe_count = 0u;
  for (std::uint32_t p = 0u; p < action.participant_count; ++p) {
    const auto& link = action_links[action.participant_offset + p];
    if (link.logical_recipe_id == 0u && link.revision_identity == 0u) continue;
    if (link.logical_recipe_id == 0u || link.revision_identity == 0u ||
        link.occurrence_identity == 0u || link.participation_identity == 0u)
      return false;
    std::uint64_t final_revision = 0u;
    if (!resident_discourse_settlement_final_revision(
            action, link, network_credit, records, record_count,
            &final_revision) ||
        final_revision == 0u)
      return false;
    if (final_revision == link.revision_identity) continue;

    bool duplicate = false;
    for (std::uint32_t r = 0u; r < *recipe_count; ++r) {
      if (logical_recipe_ids[r] != link.logical_recipe_id) continue;
      if (prior_revision_identities[r] != link.revision_identity ||
          final_revision_identities[r] != final_revision)
        return false;
      duplicate = true;
      break;
    }
    if (duplicate) continue;
    if (*recipe_count >= kDirectDiscourseRelationRecipeCapacity) return false;
    const std::uint32_t slot = (*recipe_count)++;
    logical_recipe_ids[slot] = link.logical_recipe_id;
    prior_revision_identities[slot] = link.revision_identity;
    final_revision_identities[slot] = final_revision;
  }
  if (*recipe_count == 0u) return false;

  for (std::uint32_t i = 1u; i < *recipe_count; ++i) {
    const std::uint64_t logical = logical_recipe_ids[i];
    const std::uint64_t prior = prior_revision_identities[i];
    const std::uint64_t final = final_revision_identities[i];
    std::uint32_t j = i;
    while (j != 0u && logical < logical_recipe_ids[j - 1u]) {
      logical_recipe_ids[j] = logical_recipe_ids[j - 1u];
      prior_revision_identities[j] = prior_revision_identities[j - 1u];
      final_revision_identities[j] = final_revision_identities[j - 1u];
      --j;
    }
    logical_recipe_ids[j] = logical;
    prior_revision_identities[j] = prior;
    final_revision_identities[j] = final;
  }
  return direct_discourse_relation_identity(
             logical_recipe_ids, final_revision_identities, *recipe_count) != 0u;
}

__device__ inline void note_resident_discourse_settlement(
    direct_network::DirectResidentLanguageRuntimeBlock* language,
    const DirectActionOccurrence& action,
    const DirectActionParticipationLink* action_links,
    std::uint32_t action_participant_capacity,
    const direct_network::DirectExactHistoryRecord* records,
    std::uint32_t record_count) {
  using namespace direct_network;
  if (language == nullptr || action_links == nullptr || records == nullptr ||
      action.action_ticket_id == 0u || action.network_identity == 0u ||
      action.recruitment_identity == 0u || action.participant_offset == kInvalidIndex ||
      action.participant_count == 0u ||
      action.participant_count > action_participant_capacity)
    return;

  const DirectExactHistoryRecord* credit = nullptr;
  for (std::uint32_t i = 0u; i < record_count; ++i) {
    const auto& record = records[i];
    if (record.kind != DirectExactHistoryKind::network_credit ||
        record.identity != action.action_ticket_id ||
        record.parent_identity != action.network_identity)
      continue;
    const std::uint64_t rid =
        (static_cast<std::uint64_t>(record.subject) << 32u) | record.source;
    if (rid != action.recruitment_identity) continue;
    if (credit != nullptr) return;
    credit = &record;
  }
  if (credit == nullptr) return;

  std::uint64_t source_identity = 0u;
  std::uint64_t context_identity = 0u;
  bool ambiguous = false;
  for (std::uint32_t p = 0u; p < action.participant_count && !ambiguous; ++p) {
    const auto& link = action_links[action.participant_offset + p];
    if (link.participant_ticket_id == 0u) continue;
    for (std::uint32_t i = 0u; i < record_count; ++i) {
      const auto& record = records[i];
      if (record.kind != DirectExactHistoryKind::source_assertion ||
          record.parent_identity != link.participant_ticket_id ||
          record.identity == 0u ||
          (record.flags & kDirectHistoryVerifiedObservation) == 0u)
        continue;
      DirectSourceAssertionClaim claim{};
      if (!read_source_assertion_claim(record, &claim) ||
          claim.contact_authenticated == 0u)
        continue;
      if (source_identity == 0u) {
        source_identity = claim.source_identity;
        context_identity = claim.context_identity;
      } else if (source_identity != claim.source_identity ||
                 context_identity != claim.context_identity) {
        ambiguous = true;
        break;
      }
    }
  }

  direct_discourse_focus_note(&language->language.discourse_focus, *credit);
  for (std::uint32_t i = 0u; i < language->language.discourse_focus.count; ++i) {
    auto& entry = language->language.discourse_focus.entries[i];
    if (entry.recruitment_identity != action.recruitment_identity) continue;
    if (!ambiguous) {
      entry.source_identity = source_identity;
      entry.context_identity = context_identity;
    } else {
      entry.source_identity = 0u;
      entry.context_identity = 0u;
    }
    break;
  }

  std::uint64_t logical_recipe_ids[kDirectDiscourseRelationRecipeCapacity]{};
  std::uint64_t prior_revision_identities[kDirectDiscourseRelationRecipeCapacity]{};
  std::uint64_t final_revision_identities[kDirectDiscourseRelationRecipeCapacity]{};
  std::uint32_t relation_recipe_count = 0u;
  if (!resident_discourse_settlement_relation_recipes(
          action, action_links, action_participant_capacity, *credit, records,
          record_count, logical_recipe_ids, prior_revision_identities,
          final_revision_identities, &relation_recipe_count))
    return;
  const std::uint64_t qualified_source = ambiguous ? 0u : source_identity;
  const std::uint64_t qualified_context = ambiguous ? 0u : context_identity;
  if (!direct_discourse_relation_focus_rebind(
          &language->language.discourse_focus, logical_recipe_ids,
          prior_revision_identities, final_revision_identities,
          relation_recipe_count, qualified_source, qualified_context))
    return;
  (void)direct_discourse_relation_focus_note(
      &language->language.discourse_focus, *credit, logical_recipe_ids,
      final_revision_identities, relation_recipe_count, qualified_source,
      qualified_context);
}

}  // namespace substrate::direct_adult_core

#endif
